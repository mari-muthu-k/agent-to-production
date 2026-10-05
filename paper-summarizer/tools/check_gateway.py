"""Probe OpenRouter (or any OpenAI-compatible URL) and report what the configured model supports.

Plain HTTP with `requests` (paper_agent's OpenRouterClient), no SDK. Reads LLM_BASE_URL, LLM_API_KEY,
LLM_MODEL (first entry if it is a list) and optional EMBED_MODEL from the environment (.env).
About 15 small requests: mind the free-tier cap of 50 requests/day.

    python tools/check_gateway.py                      # human-readable report
    python tools/check_gateway.py --json               # machine-readable, to paste back
    python tools/check_gateway.py --model <id>:free    # probe another model without editing .env

Exit code 0 if the basic chat call works, 1 otherwise.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

from paper_agent.llm_client import BadRequestError, OpenRouterClient, env_models
from paper_agent.openrouter import assistant_turn, is_openrouter, key_info, list_models, reasoning_tokens, stream_chat

PROBE = [{"role": "user", "content": "Reply with exactly: ready"}]


@dataclass
class Check:
    name: str
    status: str          # ok | unsupported | error | skipped
    detail: str
    ms: int = 0


def _err(e: Exception) -> str:
    code = getattr(e, "status_code", None)
    msg = getattr(e, "message", None) or str(e)
    return f"{type(e).__name__}{f' HTTP {code}' if code else ''}: {msg[:160]}"


def _text(reply: dict) -> str:
    return (reply["choices"][0]["message"].get("content") or "").strip()


def run_checks(client: OpenRouterClient, model: str, embed_model) -> list:
    results: list = []
    state: dict = {}
    tok = {"max_tokens": 40}

    def check(name):
        def wrap(fn):
            t = time.perf_counter()
            try:
                status, detail = fn()
            except BadRequestError as e:
                status, detail = "unsupported", _err(e)
            except Exception as e:   # report everything; this is a diagnostic tool
                status, detail = "error", _err(e)
            results.append(Check(name, status, detail, int((time.perf_counter() - t) * 1000)))
            return fn
        return wrap

    def create(**kw):
        return client.chat(model=model, messages=kw.pop("messages", PROBE), **kw)

    @check("max_tokens")
    def _():
        r = create(max_tokens=40)
        return "ok", f"accepted (finish_reason={r['choices'][0].get('finish_reason')})"

    @check("max_completion_tokens")
    def _():
        r = create(max_completion_tokens=40)
        return "ok", f"accepted (finish_reason={r['choices'][0].get('finish_reason')})"

    if results[0].status != "ok" and results[1].status == "ok":
        tok = {"max_completion_tokens": 40}   # use what works for every later probe
    param = next(iter(tok))

    @check("chat")
    def _():
        r = create(**tok)
        u = r.get("usage")
        usage = f"usage {u.get('prompt_tokens')} in / {u.get('completion_tokens')} out" if u else "NO usage returned"
        thinking = reasoning_tokens(u)
        note = f", {thinking} of them reasoning (on by default!)" if thinking else ""
        status = "ok" if _text(r) else "error"
        return status, f"{_text(r)!r}, finish_reason={r['choices'][0].get('finish_reason')}, {usage}{note}"

    @check(f"{param} enforced")
    def _():
        r = create(**{param: 5}, messages=[{"role": "user", "content": "Count from 1 to 50, comma-separated."}])
        fr = r["choices"][0].get("finish_reason")
        return ("ok" if fr == "length" else "error"), f"finish_reason={fr} (expect 'length')"

    @check("temperature")
    def _():
        notes = []
        for t in (0, 1.0, 1.5):
            try:
                create(temperature=t, **tok)
                notes.append(f"{t}: ok")
            except BadRequestError as e:
                notes.append(f"{t}: rejected ({_err(e)[:60]})")
        return ("ok" if all("ok" in n for n in notes) else "unsupported"), "; ".join(notes)

    @check("temperature out of range (5.0)")
    def _():
        try:
            create(temperature=5.0, **tok)
            return "unsupported", "accepted 5.0 (the endpoint is lenient: Day 1 3.2 will say 'succeeded')"
        except BadRequestError as e:
            return "ok", f"rejected as expected: {_err(e)[:80]}"

    @check("seed")
    def _():
        msgs = [{"role": "user", "content": "Suggest one catchy title for a blog post about small code models."}]
        a = _text(create(messages=msgs, temperature=1.0, seed=7, **tok))
        b = _text(create(messages=msgs, temperature=1.0, seed=7, **tok))
        return "ok", f"accepted; same output twice: {a == b}"

    schema = {"type": "object", "properties": {"answer": {"type": "string"},
                                               "confidence": {"type": "string", "enum": ["high", "medium", "low"]}},
              "required": ["answer", "confidence"], "additionalProperties": False}
    q = [{"role": "user", "content": "What is pass@1 in code generation? Answer briefly as JSON."}]

    @check("response_format json_schema")
    def _():
        r = create(messages=q, **{param: 120}, response_format={
            "type": "json_schema", "json_schema": {"name": "answer", "schema": schema, "strict": True}})
        return "ok", f"valid JSON with keys {sorted(json.loads(_text(r)))}"

    @check("response_format json_object")
    def _():
        json.loads(_text(create(messages=q, **{param: 120}, response_format={"type": "json_object"})))
        return "ok", "valid JSON"

    tool = {"type": "function", "function": {"name": "submit_answer", "description": "Submit the answer",
                                             "parameters": schema}}

    @check("tools (forced tool_choice)")
    def _():
        r = create(messages=q, **{param: 120}, tools=[tool],
                   tool_choice={"type": "function", "function": {"name": "submit_answer"}})
        calls = r["choices"][0]["message"].get("tool_calls") or []
        if not calls:
            return "error", "no tool_calls returned"
        json.loads(calls[0]["function"]["arguments"])
        return "ok", f"called {calls[0]['function']['name']} with valid JSON arguments"

    @check("tools (auto)")
    def _():
        search = {"type": "function", "function": {
            "name": "search_paper", "description": "Search the uploaded research paper",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
        r = create(messages=[{"role": "user", "content": "Search the paper for its pass@1 results."}],
                   **{param: 120}, tools=[search], tool_choice="auto")
        calls = r["choices"][0]["message"].get("tool_calls") or []
        return "ok", (f"called {calls[0]['function']['name']}" if calls else "answered without a tool call")

    @check("streaming")
    def _():
        t, first, parts, usage = time.perf_counter(), None, [], None
        for event in stream_chat(client, model=model, **{param: 60}, stream_options={"include_usage": True},
                                 messages=[{"role": "user", "content": "In 2 sentences, what is attention?"}]):
            delta = (event.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content"):
                first = first or time.perf_counter()
                parts.append(delta["content"])
            usage = event.get("usage") or usage
        ttft = f"{(first - t) * 1000:.0f} ms" if first else "n/a"
        return "ok", f"{len(parts)} chunks, time to first token {ttft}, usage in stream: {usage is not None}"

    question = [{"role": "user", "content": "How many r's are in the word 'strawberry'?"}]

    @check("reasoning on (OpenRouter `reasoning`)")
    def _():
        r = create(messages=question, **{param: 1500}, reasoning={"enabled": True})
        m = r["choices"][0]["message"]
        state["turn"] = assistant_turn(m)
        return "ok", (f"answer={(m.get('content') or '').strip()[:40]!r}, reasoning tokens="
                      f"{reasoning_tokens(r.get('usage'))}, reasoning text={'yes' if m.get('reasoning') else 'no'}, "
                      f"reasoning_details={'yes' if m.get('reasoning_details') else 'no'}")

    @check("reasoning off")
    def _():
        r = create(messages=question, **{param: 60}, reasoning={"enabled": False})
        thinking = reasoning_tokens(r.get("usage"))
        return ("ok" if not thinking else "unsupported"), (
            f"reasoning tokens={thinking}, answer={_text(r)[:40]!r}"
            + (" (this model always reasons: give it a large max_tokens)" if thinking else ""))

    @check("reasoning_details passed back")
    def _():
        if not state.get("turn"):
            return "skipped", "no assistant turn from the reasoning probe"
        r = create(messages=question + [state["turn"], {"role": "user", "content": "Are you sure? Think carefully."}],
                   **{param: 1500}, reasoning={"enabled": True})
        return "ok", f"follow-up accepted: {_text(r)[:60]!r}"

    @check("embeddings")
    def _():
        if not embed_model:
            return "skipped", "EMBED_MODEL not set"
        r = client.embeddings(model=embed_model, input=["sliding-window attention", "pass@1"])
        return "ok", f"{len(r['data'])} vectors, dim={len(r['data'][0]['embedding'])}"

    @check("unknown model")
    def _():
        try:
            client.chat(model="no-such-model-xyz", messages=PROBE, **tok)
            return "error", "accepted an unknown model name"
        except Exception as e:     # noqa: BLE001  (we report whichever status comes back)
            code = getattr(e, "status_code", None)
            return "ok", f"HTTP {code} {type(e).__name__} (Day 1 3.2 expects 404, OpenRouter may say 400)"

    @check("OpenRouter key limits")
    def _():
        if not is_openrouter(client.base_url):
            return "skipped", "not OpenRouter"
        d = key_info(client)
        return "ok", (f"free tier={d.get('is_free_tier')}, credit limit={d.get('limit')}, "
                      f"used={d.get('usage')}, label={d.get('label')!r}")

    @check("free models available")
    def _():
        rows = list_models(client, free_only=True)
        tools_ok = [m["id"] for m in rows if "tools" in m["parameters"]]
        return "ok", (f"{len(rows)} models ({len(tools_ok)} with tools), e.g. "
                      f"{', '.join(m['id'] for m in rows[:4])}")

    return results


def recommendations(results: list) -> list:
    by = {r.name: r for r in results}
    recs = []
    if by["max_tokens"].status != "ok" and by["max_completion_tokens"].status == "ok":
        recs.append("Set LLM_MAX_TOKENS_PARAM=max_completion_tokens (LLMConfig.max_tokens_param).")
    if by["temperature"].status != "ok":
        recs.append("Model rejects some temperatures: use LLMClient.chat(..., temperature=None).")
    if by["response_format json_schema"].status != "ok":
        recs.append("No json_schema mode: rely on prompt-only JSON + Pydantic validation + repair (Day 1 2.3a).")
    if by["tools (forced tool_choice)"].status != "ok":
        recs.append("Forced tool calls unsupported: the Day 3 agent needs tool calling; "
                    "pick one from free_models(needs=('tools',)).")
    if by["embeddings"].status not in ("ok", "skipped"):
        recs.append("Embeddings unavailable: Day 2 needs EMBED_MODEL, e.g. nvidia/nemotron-3-embed-1b:free.")
    if "(on by default!)" in by["chat"].detail:
        recs.append("The model reasons by default and spends max_tokens on it: use LLMConfig(reasoning={'enabled': "
                    "False}) (or effort 'low'), and give raw calls a larger max_tokens.")
    if by["reasoning off"].status == "unsupported":
        recs.append("Reasoning cannot be switched off for this model: cells with max_tokens below ~200 may return "
                    "empty answers. Prefer a model where reasoning is optional for class.")
    if "free tier=True" in by["OpenRouter key limits"].detail:
        recs.append("Free-tier key: :free models allow 50 requests/day per account without credits (1000/day with "
                    "$10+). Switching models does not reset it; the Day 1 code-along alone makes ~55 calls.")
    return recs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--model", help="model to probe instead of LLM_MODEL")
    args = parser.parse_args()
    missing = [n for n in ("LLM_BASE_URL", "LLM_API_KEY") if not os.environ.get(n)]
    model = args.model or (env_models("LLM_MODEL") or [None])[0]
    if missing or not model:
        print("missing:", ", ".join(missing + ([] if model else ["LLM_MODEL"])))
        return 1
    client = OpenRouterClient(timeout=60)
    embed = os.environ.get("EMBED_MODEL")
    results = run_checks(client, model, embed)
    recs = recommendations(results)
    if args.json:
        print(json.dumps({"base_url": client.base_url, "model": model, "embed_model": embed,
                          "checks": [asdict(r) for r in results], "recommendations": recs}, indent=2))
    else:
        print(f"endpoint: {client.base_url}\nmodel:    {model}\nembed:    {embed or '-'}\n")
        icon = {"ok": "✅", "unsupported": "⚠️ ", "error": "❌", "skipped": "⏭ "}
        for r in results:
            print(f"{icon[r.status]} {r.name:<36} {r.ms:>6} ms  {r.detail}")
        if recs:
            print("\nRecommendations:")
            for rec in recs:
                print(" -", rec)
    return 0 if next(r for r in results if r.name == "chat").status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
