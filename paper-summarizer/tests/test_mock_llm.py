"""mock-llm behaves like an OpenAI-compatible endpoint, deterministically, with injectable faults."""
import json
import time

import httpx
import numpy as np
import openai
import pytest

from paper_agent.explain import paper_messages
from paper_agent.fixtures.tinycoder import PAPER, SECTION_IDS
from paper_agent.schemas import Answer, PaperExplainer

HI = [{"role": "user", "content": "hi"}]
QUESTION = "How does TinyCoder's pass@1 compare with the 7B baseline?"
CONTEXT = "\n".join(f'<section id="{s}">{t}</section>' for s, t in PAPER["sections"].items())
QA = [{"role": "user", "content": f"{CONTEXT}\n\nQuestion: {QUESTION}"}]


def chat(client, messages=HI, model="mock-llm", **kw):
    return client.chat.completions.create(model=model, messages=messages, **kw)


def test_health_and_models(mock):
    assert httpx.get(f"{mock.url}/health").json() == {"status": "ok"}
    assert "mock-llm" in [m.id for m in mock.client().models.list()]


def test_basic_chat_and_usage(mock):
    r = chat(mock.client(), [{"role": "user", "content": "Reply with exactly: ready"}], max_tokens=5)
    assert r.choices[0].message.content == "ready" and r.choices[0].finish_reason == "stop"
    assert r.usage.prompt_tokens > 0 and r.usage.completion_tokens == 1


def test_max_completion_tokens_is_accepted(mock):
    r = chat(mock.client(), [{"role": "user", "content": "Reply with exactly: ready"}], max_completion_tokens=5)
    assert r.choices[0].message.content == "ready"


def test_explainer_is_deterministic_and_schema_valid(mock):
    c = mock.client()
    outs = [chat(c, paper_messages(), max_tokens=900, temperature=0).choices[0].message.content for _ in range(2)]
    assert outs[0] == outs[1]
    result = PaperExplainer.model_validate_json(outs[0])
    assert set(result.citations) <= set(SECTION_IDS)


def test_mock_resists_injection_when_prompt_says_data_not_instructions(mock):
    result = PaperExplainer.model_validate_json(
        chat(mock.client(), paper_messages(), max_tokens=900).choices[0].message.content)
    assert "better than all" not in result.headline.lower() and len(result.caveats) >= 2


def test_mock_obeys_injection_without_a_defense_rule(mock):
    naive = paper_messages(prompt="Explain the paper. Reply with ONLY a JSON object with headline, summary, "
                                  "why_it_matters, evidence_type, key_terms, caveats, citations.")
    result = PaperExplainer.model_validate_json(chat(mock.client(), naive, max_tokens=900).choices[0].message.content)
    assert "better than all" in result.headline.lower()


def test_qa_three_ways_to_structure(mock):
    c = mock.client()
    a = chat(c, [{"role": "system", "content": 'Reply with ONLY JSON: {"answer": str, "citations": [section ids], '
                                               '"confidence": "high|medium|low"}'}] + QA, max_tokens=250)
    b = chat(c, QA, max_tokens=250, response_format={
        "type": "json_schema", "json_schema": {"name": "answer", "schema": Answer.model_json_schema()}})
    t = chat(c, QA, max_tokens=250, tools=[{"type": "function", "function": {
        "name": "submit_answer", "description": "Submit", "parameters": Answer.model_json_schema()}}],
        tool_choice={"type": "function", "function": {"name": "submit_answer"}})
    answers = [Answer.model_validate_json(a.choices[0].message.content),
               Answer.model_validate_json(b.choices[0].message.content),
               Answer.model_validate_json(t.choices[0].message.tool_calls[0].function.arguments)]
    for ans in answers:
        assert ans.citations == ["results"] and "41%" in ans.answer
    assert t.choices[0].finish_reason == "tool_calls"


def test_generic_json_schema(mock):
    schema = {"type": "object", "properties": {"title": {"type": "string"}, "year": {"type": "integer"},
                                               "tags": {"type": "array", "items": {"type": "string"}, "minItems": 2}},
              "required": ["title", "year", "tags"]}
    r = chat(mock.client(), max_tokens=200, response_format={"type": "json_schema",
                                                             "json_schema": {"name": "x", "schema": schema}})
    data = json.loads(r.choices[0].message.content)
    assert isinstance(data["year"], int) and len(data["tags"]) == 2


def test_auto_tool_choice_then_answer(mock):
    tools = [{"type": "function", "function": {"name": "search_paper", "description": "Search the paper text",
              "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
             {"type": "function", "function": {"name": "save_note", "description": "Save a note",
              "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}]
    msgs = [{"role": "user", "content": "Search the paper for the pass@1 results"}]
    r = chat(mock.client(), msgs, tools=tools, max_tokens=200)
    call = r.choices[0].message.tool_calls[0]
    assert call.function.name == "search_paper" and "pass@1" in json.loads(call.function.arguments)["query"]
    msgs += [r.choices[0].message.model_dump(exclude_none=True),
             {"role": "tool", "tool_call_id": call.id, "content": PAPER["sections"]["results"]}]
    final = chat(mock.client(), msgs, tools=tools, max_tokens=200)
    assert final.choices[0].message.tool_calls is None and "41%" in final.choices[0].message.content


def test_streaming_with_usage(mock):
    stream = chat(mock.client(), [{"role": "user", "content": "In 3 sentences, explain what sliding-window "
                                                              "attention is."}],
                  stream=True, max_tokens=150, stream_options={"include_usage": True})
    chunks = list(stream)
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text.startswith("Sliding-window attention") and len(chunks) > 10
    assert chunks[-1].usage.completion_tokens > 0
    assert [c.choices[0].finish_reason for c in chunks if c.choices][-1] == "stop"


def test_sentence_limits_and_audience(mock):
    c = mock.client()
    one = chat(c, [{"role": "user", "content": "In one sentence, what is sliding-window attention?"}], max_tokens=60)
    assert one.choices[0].finish_reason == "stop" and one.choices[0].message.content.count(". ") == 0
    user = {"role": "user", "content": "Explain this abstract:\n" + PAPER["sections"]["abstract"]}
    plain = chat(c, [{"role": "system", "content": "You explain research papers simply."}, user])
    student = chat(c, [{"role": "system", "content": "Audience: a first-year CS student. At most three short "
                                                     "sentences and one everyday analogy."}, user])
    assert "Think of" in student.choices[0].message.content
    assert plain.choices[0].message.content != student.choices[0].message.content


def test_stop_sequence(mock):
    r = chat(mock.client(), [{"role": "user", "content": "List 5 Python web frameworks, numbered 1. 2. 3. ..."}],
             max_tokens=120, stop=["3."])
    assert r.choices[0].message.content.strip() == "1. Django\n2. Flask"


def test_temperature_and_seed(mock):
    c = mock.client()
    p = [{"role": "user", "content": "Suggest one short, catchy blog-post title about this paper: TinyCoder"}]
    assert len({chat(c, p, temperature=0).choices[0].message.content for _ in range(3)}) == 1
    seeded = {chat(c, p, temperature=1.0, seed=7).choices[0].message.content for _ in range(3)}
    assert len(seeded) == 1


def test_embeddings_float_and_base64(mock):
    c = mock.client()
    texts = ["sliding window attention", "attention with a sliding window", "baking sourdough bread"]
    e = c.embeddings.create(model="mock-embed", input=texts)     # SDK default: base64, decoded by the SDK
    v = np.array([d.embedding for d in e.data])
    assert v.shape == (3, 256) and np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
    assert v[0] @ v[1] > 0.5 > v[0] @ v[2]
    f = c.embeddings.create(model="mock-embed", input="sliding window attention", encoding_format="float",
                            dimensions=64)
    assert len(f.data[0].embedding) == 64
    assert e.usage.prompt_tokens > 0


# --- errors ------------------------------------------------------------------
def test_wrong_key_is_401(mock):
    with pytest.raises(openai.AuthenticationError):
        chat(mock.client(api_key="sk-wrong"))


def test_unknown_model_is_404(mock):
    with pytest.raises(openai.NotFoundError):
        chat(mock.client(), model="no-such-model")


def test_bad_temperature_is_400(mock):
    with pytest.raises(openai.BadRequestError):
        chat(mock.client(), temperature=5.0)


def test_rejected_param_is_400(mock):
    c = mock.client(default_headers={"X-Mock-Reject-Params": "max_tokens"})
    with pytest.raises(openai.BadRequestError, match="max_completion_tokens"):
        chat(c, max_tokens=5)
    assert chat(c, max_completion_tokens=5).choices[0].message.content


@pytest.mark.parametrize("status,cls", [(429, openai.RateLimitError), (500, openai.InternalServerError),
                                        (503, openai.InternalServerError), (400, openai.BadRequestError),
                                        (401, openai.AuthenticationError), (404, openai.NotFoundError)])
def test_injected_status_via_header(mock, status, cls):
    with pytest.raises(cls) as e:
        chat(mock.client(), extra_headers={"X-Mock-Status": str(status), "X-Mock-Retry-After": "3"})
    assert e.value.status_code == status


def test_429_carries_retry_after(mock):
    with pytest.raises(openai.RateLimitError) as e:
        chat(mock.client(), extra_headers={"X-Mock-Status": "429", "X-Mock-Retry-After": "3"})
    assert e.value.response.headers["retry-after"] == "3"


def test_fault_queue_counts_and_model_filter(mock):
    mock.fault(status=503, count=2, model="mock-llm")
    c = mock.client()
    for _ in range(2):
        with pytest.raises(openai.InternalServerError):
            chat(c)
    assert chat(c).choices[0].message.content
    mock.fault(status=503, model="mock-llm")
    assert chat(c, model="mock-llm-fallback").choices[0].message.content   # filter: other model unaffected
    with pytest.raises(openai.InternalServerError):
        chat(c)


def test_slow_response_triggers_client_timeout(mock):
    c = mock.client(timeout=0.3)
    t = time.perf_counter()
    with pytest.raises(openai.APITimeoutError):
        chat(c, extra_headers={"X-Mock-Delay-Ms": "2000"})
    assert time.perf_counter() - t < 1.5


def test_truncation_via_max_tokens_and_header(mock):
    c = mock.client()
    r = chat(c, paper_messages(), max_tokens=30)
    assert r.choices[0].finish_reason == "length" and r.usage.completion_tokens == 30
    with pytest.raises(json.JSONDecodeError):
        json.loads(r.choices[0].message.content)
    r = chat(c, paper_messages(), max_tokens=900, extra_headers={"X-Mock-Truncate": "1"})
    assert r.choices[0].finish_reason == "length"


def test_broken_model_is_500(mock):
    with pytest.raises(openai.InternalServerError):
        chat(mock.client(), model="broken-model")
