"""Probe an OpenAI-compatible endpoint and report what it supports.

Reads LLM_BASE_URL, LLM_API_KEY, LLM_MODEL and (optional) EMBED_MODEL from the environment (.env).
Every probe is one small request (max 40 output tokens), so a full run costs well under a cent.

    python tools/check_gateway.py            # human-readable report
    python tools/check_gateway.py --json     # machine-readable, to paste back to the course team

Exit code 0 if the basic chat call works, 1 otherwise.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import httpx
import openai
from openai import OpenAI

from paper_agent.openrouter import assistant_turn, is_openrouter, make_client, reasoning_tokens

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


def run_checks(client: OpenAI, model: str, embed_model) -> list:
    results: list = []
    state: dict = {}
    tok = {"max_tokens": 40}

    def check(name):
        def wrap(fn):
            t = time.perf_counter()
            try:
                status, detail = fn()
            except openai.BadRequestError as e:
                status, detail = "unsupported", _err(e)
            except Exception as e:   # report everything; this is a diagnostic tool
                status, detail = "error", _err(e)
            results.append(Check(name, status, detail, int((time.perf_counter() - t) * 1000)))
            return fn
        return wrap

    def create(**kw):
        return client.chat.completions.create(model=model, messages=kw.pop("messages", PROBE), **kw)

    @check("max_tokens")
    def _():
        r = create(max_tokens=40)
        return "ok", f"accepted (finish_reason={r.choices[0].finish_reason})"

    @check("max_completion_tokens")
    def _():
        r = create(max_completion_tokens=40)
        return "ok", f"accepted (finish_reason={r.choices[0].finish_reason})"

    if results[0].status != "ok" and results[1].status == "ok":
        tok = {"max_completion_tokens": 40}   # use what works for every later probe
    param = next(iter(tok))

    @check("chat")
    def _():
        r = create(**tok)
        u = r.usage
        usage = f"usage {u.prompt_tokens} in / {u.completion_tokens} out" if u else "NO usage returned"
        thinking = reasoning_tokens(u)
        note = f", {thinking} of them reasoning (on by default!)" if thinking else ""
        status = "ok" if r.choices[0].message.content else "error"
        return status, f"{r.choices[0].message.content!r}, finish_reason={r.choices[0].finish_reason}, {usage}{note}"

    @check(f"{param} enforced")
    def _():
        r = create(**{param: 5}, messages=[{"role": "user", "content": "Count from 1 to 50, comma-separated."}])
        fr = r.choices[0].finish_reason
        return ("ok" if fr == "length" else "error"), f"finish_reason={fr} (expect 'length')"

    @check("temperature")
    def _():
        notes = []
        for t in (0, 1.0, 1.5):
            try:
                create(temperature=t, **tok)
                notes.append(f"{t}: ok")
            except openai.BadRequestError as e:
                notes.append(f"{t}: rejected ({_err(e)[:60]})")
        return ("ok" if all("ok" in n for n in notes) else "unsupported"), "; ".join(notes)

    @check("temperature out of range (5.0)")
    def _():
        try:
            create(temperature=5.0, **tok)
            return "unsupported", "accepted 5.0 (gateway is lenient: Day 1 6.2 will say 'succeeded')"
        except openai.BadRequestError as e:
            return "ok", f"rejected as expected: {_err(e)[:80]}"

    @check("seed")
    def _():
        msgs = [{"role": "user", "content": "Suggest one catchy title for a blog post about small code models."}]
        a = create(messages=msgs, temperature=1.0, seed=7, **tok).choices[0].message.content
        b = create(messages=msgs, temperature=1.0, seed=7, **tok).choices[0].message.content
        return "ok", f"accepted; same output twice: {a == b}"

    schema = {"type": "object", "properties": {"answer": {"type": "string"},
                                               "confidence": {"type": "string", "enum": ["high", "medium", "low"]}},
              "required": ["answer", "confidence"], "additionalProperties": False}
    q = [{"role": "user", "content": "What is pass@1 in code generation? Answer briefly as JSON."}]

    @check("response_format json_schema")
    def _():
        r = create(messages=q, **{param: 120}, response_format={
            "type": "json_schema", "json_schema": {"name": "answer", "schema": schema, "strict": True}})
        data = json.loads(r.choices[0].message.content)
        return "ok", f"valid JSON with keys {sorted(data)}"

    @check("response_format json_object")
    def _():
        r = create(messages=q, **{param: 120}, response_format={"type": "json_object"})
        json.loads(r.choices[0].message.content)
        return "ok", "valid JSON"

    tool = {"type": "function", "function": {"name": "submit_answer", "description": "Submit the answer",
                                             "parameters": schema}}

    @check("tools (forced tool_choice)")
    def _():
        r = create(messages=q, **{param: 120}, tools=[tool],
                   tool_choice={"type": "function", "function": {"name": "submit_answer"}})
        calls = r.choices[0].message.tool_calls or []
        if not calls:
            return "error", "no tool_calls returned"
        json.loads(calls[0].function.arguments)
        return "ok", f"called {calls[0].function.name} with valid JSON arguments"

    @check("tools (auto)")
    def _():
        search = {"type": "function", "function": {
            "name": "search_paper", "description": "Search the uploaded research paper",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
        r = create(messages=[{"role": "user", "content": "Search the paper for its pass@1 results."}],
                   **{param: 120}, tools=[search], tool_choice="auto")
        calls = r.choices[0].message.tool_calls or []
        return "ok", (f"called {calls[0].function.name}" if calls else "answered without a tool call")

    @check("streaming")
    def _():
        t, first, parts, usage = time.perf_counter(), None, [], None
        for chunk in create(stream=True, stream_options={"include_usage": True}, **{param: 60},
                            messages=[{"role": "user", "content": "In 2 sentences, what is attention?"}]):
            if chunk.choices and chunk.choices[0].delta.content:
                first = first or time.perf_counter()
                parts.append(chunk.choices[0].delta.content)
            usage = chunk.usage or usage
        ttft = f"{(first - t) * 1000:.0f} ms" if first else "n/a"
        return "ok", f"{len(parts)} chunks, time to first token {ttft}, usage in stream: {usage is not None}"

    question = [{"role": "user", "content": "How many r's are in the word 'strawberry'?"}]

    @check("reasoning on (OpenRouter `reasoning`)")
    def _():
        r = create(messages=question, **{param: 1500}, extra_body={"reasoning": {"enabled": True}})
        m = r.choices[0].message
        details = getattr(m, "reasoning_details", None)
        state["turn"] = assistant_turn(m)
        return "ok", (f"answer={(m.content or '').strip()[:40]!r}, reasoning tokens={reasoning_tokens(r.usage)}, "
                      f"reasoning text={'yes' if getattr(m, 'reasoning', None) else 'no'}, "
                      f"reasoning_details={'yes' if details else 'no'}")

    @check("reasoning off")
    def _():
        r = create(messages=question, **{param: 60}, extra_body={"reasoning": {"enabled": False}})
        thinking = reasoning_tokens(r.usage)
        return ("ok" if not thinking else "unsupported"), (
            f"reasoning tokens={thinking}, answer={(r.choices[0].message.content or '').strip()[:40]!r}"
            + (" (this model always reasons: give it a large max_tokens)" if thinking else ""))

    @check("reasoning_details passed back")
    def _():
        if not state.get("turn"):
            return "skipped", "no assistant turn from the reasoning probe"
        r = create(messages=question + [state["turn"], {"role": "user", "content": "Are you sure? Think carefully."}],
                   **{param: 1500}, extra_body={"reasoning": {"enabled": True}})
        return "ok", f"follow-up accepted: {(r.choices[0].message.content or '').strip()[:60]!r}"

    @check("embeddings")
    def _():
        if not embed_model:
            return "skipped", "EMBED_MODEL not set"
        r = client.embeddings.create(model=embed_model, input=["sliding-window attention", "pass@1"])
        return "ok", f"{len(r.data)} vectors, dim={len(r.data[0].embedding)}"

    @check("unknown model")
    def _():
        try:
            client.chat.completions.create(model="no-such-model-xyz", messages=PROBE, **tok)
            return "error", "accepted an unknown model name"
        except openai.APIStatusError as e:
            return "ok", f"HTTP {e.status_code} {type(e).__name__} (Day 1 6.2 expects 404 NotFoundError)"

    @check("OpenRouter key limits")
    def _():
        if not is_openrouter(str(client.base_url)):
            return "skipped", "not OpenRouter"
        resp = httpx.get(str(client.base_url).rstrip("/") + "/key", timeout=20,
                         headers={"Authorization": f"Bearer {client.api_key}"})
        resp.raise_for_status()
        d = resp.json().get("data", {})
        state["free_tier"] = d.get("is_free_tier")
        return "ok", (f"free tier={d.get('is_free_tier')}, credit limit={d.get('limit')}, "
                      f"used={d.get('usage')}, label={d.get('label')!r}")

    @check("list models")
    def _():
        ids = [m.id for m in client.models.list()]
        return "ok", f"{len(ids)} models: {', '.join(ids[:8])}{' ...' if len(ids) > 8 else ''}"

    return results


def recommendations(results: list) -> list:
    by = {r.name: r for r in results}
    recs = []
    if by["max_tokens"].status != "ok" and by["max_completion_tokens"].status == "ok":
        recs.append("Set LLM_MAX_TOKENS_PARAM=max_completion_tokens (LLMConfig.max_tokens_param). Day 1 notebook "
                    "cells call max_tokens directly: see the 4.1 'Failed?' table.")
    if by["temperature"].status != "ok":
        recs.append("Model rejects some temperatures: use LLMClient.chat(..., temperature=None).")
    if by["response_format json_schema"].status != "ok":
        recs.append("No json_schema mode: rely on prompt-only JSON + Pydantic validation + repair (Day 1 5.3a).")
    if by["tools (forced tool_choice)"].status != "ok":
        recs.append("Forced tool calls unsupported: Day 3 agent needs tool calling; check another model.")
    if by["embeddings"].status not in ("ok",):
        recs.append("Embeddings unavailable: Day 2 needs EMBED_MODEL or the local sentence-transformers fallback.")
    if "(on by default!)" in by["chat"].detail:
        recs.append("The model reasons by default and spends max_tokens on it: use LLMConfig(reasoning={'enabled': "
                    "False}) (or effort 'low'), and give raw calls a larger max_tokens.")
    if by.get("reasoning off") and by["reasoning off"].status == "unsupported":
        recs.append("Reasoning cannot be switched off for this model: Day 1 cells with max_tokens below ~200 may "
                    "return empty answers. Prefer a model where reasoning is optional for class.")
    key = by.get("OpenRouter key limits")
    if key and "free tier=True" in key.detail:
        recs.append("Free-tier key: :free models allow 50 requests/day without credits (1000/day with $10+). "
                    "The Day 1 code-along alone makes ~55 calls: participants need credits or a paid model.")
    return recs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    missing = [n for n in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL") if not os.environ.get(n)]
    if missing:
        print("missing:", ", ".join(missing))
        return 1
    base_url, model, embed = os.environ["LLM_BASE_URL"], os.environ["LLM_MODEL"], os.environ.get("EMBED_MODEL")
    client = make_client(os.environ["LLM_API_KEY"], base_url, timeout=60)
    results = run_checks(client, model, embed)
    recs = recommendations(results)
    if args.json:
        print(json.dumps({"base_url": base_url, "model": model, "embed_model": embed,
                          "openai_sdk": openai.__version__, "checks": [asdict(r) for r in results],
                          "recommendations": recs}, indent=2))
    else:
        print(f"endpoint: {base_url}\nmodel:    {model}\nembed:    {embed or '-'}\n"
              f"sdk:      openai {openai.__version__}\n")
        icon = {"ok": "✅", "unsupported": "⚠️ ", "error": "❌", "skipped": "⏭ "}
        for r in results:
            print(f"{icon[r.status]} {r.name:<32} {r.ms:>6} ms  {r.detail}")
        if recs:
            print("\nRecommendations:")
            for rec in recs:
                print(" -", rec)
    return 0 if next(r for r in results if r.name == "chat").status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
