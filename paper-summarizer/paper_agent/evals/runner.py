"""Run the golden set (Day 4, 7.3 and `make eval`).

In the notebook: run_eval("v1") asks every question through the agent in this process (ask_paper).
From the command line, the same questions go to the api over HTTP, as CI would run them:

    python -m paper_agent.evals.runner --api http://api:8000 --key <eval-bot key> [--prompt-version v2]
        [--write-baseline]          # make eval-baseline
Exit code 1 when the regression gate blocks the run (or a request fails), 0 when it passes.
"""
import argparse
import datetime
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Optional

from paper_agent.evals import gate
from paper_agent.evals.metrics import JUDGE_VERSION, judge, score

HERE = Path(__file__).parent
GOLDEN = HERE / "golden_tinycoder.json"
BASELINE = HERE / "baseline.json"
EVAL_PAUSE_S = float(os.environ.get("EVAL_PAUSE_S", "0"))      # free tiers: 4 keeps you under 20 requests/minute
NOTEBOOK_METRICS = ("valid_citations", "cited", "right_section", "faithful", "refused_correctly", "hype_free",
                    "readability")
ALL_METRICS = NOTEBOOK_METRICS + ("recall_at_k", "page_accuracy")


def golden_set(which: str = "notebook") -> list:
    """The 8 questions of cell 7.1 (which="notebook"), or all 20 (which="all")."""
    items = json.loads(GOLDEN.read_text())["questions"]
    return [q for q in items if which == "all" or q["set"] == which]


def chunk_index(paper: dict) -> dict:
    """chunk id -> {"section": section id, "page": n, "text": ...} for one registered paper."""
    from paper_agent.agent.papers import section_id
    return {c["id"]: {"section": section_id(c["section"]), "page": c["page"], "text": c["text"]}
            for c in paper["chunks"]}


def average(rows: list, metrics=NOTEBOOK_METRICS) -> dict:
    out = {}
    for m in metrics:
        values = [r["scores"][m] for r in rows if r["scores"].get(m) is not None]
        out[m] = None if not values else round(sum(values) / len(values), 1 if m == "readability" else 2)
    return out


def run_eval(prompt_version: str, ask: Callable, complete: Callable, chunks: dict, golden: Optional[list] = None,
             metrics=NOTEBOOK_METRICS, pause_s: Optional[float] = None, show: bool = True) -> tuple:
    """Ask every golden question, score it, judge it. ask(question) -> ask_paper()-style dict;
    complete(messages, max_tokens) -> text, for the judge. Returns (averages, rows)."""
    golden = golden or golden_set()
    pause_s = EVAL_PAUSE_S if pause_s is None else pause_s
    rows = []
    for item in golden:
        result = ask(item["question"])
        sources = "\n\n".join(f"[{c}] {chunks[c]['text']}" for c in result.get("retrieved") or [] if c in chunks)
        verdict = judge(item["question"], sources or "(nothing was retrieved)", result["answer"], complete)
        scores = score(item, result, chunks, faithful=verdict["faithful"])
        failed = [m for m in metrics if scores.get(m) == 0.0]
        rows.append({"id": item["id"], "question": item["question"], "scores": scores, "judge": verdict,
                     "llm_calls": result.get("llm_calls", 0), "cost_usd": result.get("cost_usd", 0.0)})
        if show:
            mark = "✅" if not failed else "❌"
            print(f"{mark} {item['question'][:62]:62} {'failed: ' + ', '.join(failed) if failed else ''}")
        if pause_s:
            time.sleep(pause_s)
    return average(rows, metrics), rows


def baseline_record(metrics: dict, prompt_version: str, model: str) -> dict:
    return {"metrics": metrics, "prompt_version": prompt_version, "judge_version": JUDGE_VERSION, "model": model,
            "date": datetime.date.today().isoformat()}


# --- command line: the api over HTTP (make eval) ------------------------------------------------------
def http_ask(client, paper_id: str, prompt_version: str) -> Callable:
    def ask(question: str) -> dict:
        r = client.post(f"/papers/{paper_id}/ask", json={"question": question, "prompt_version": prompt_version})
        if r.status_code != 200:
            raise SystemExit(f"api returned {r.status_code} for {question!r}: {r.text[:200]}")
        return r.json()
    return ask


def gateway_complete(gateway_url: str, key: str) -> Callable:
    """The judge through the LiteLLM proxy, with the eval bot's own key and budget."""
    import httpx

    def complete(messages: list, max_tokens: int) -> str:
        r = httpx.post(f"{gateway_url}/v1/chat/completions", headers={"Authorization": f"Bearer {key}"}, timeout=60,
                       json={"model": "paper-explainer", "messages": messages, "max_tokens": max_tokens,
                             "temperature": 0})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""
    return complete


def main(argv=None) -> int:
    import httpx

    from paper_agent.agent.papers import register_paper
    from paper_agent.fixtures.tinycoder_pdf import make_pdf
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default=os.environ.get("API_URL", "http://api:8000"))
    parser.add_argument("--gateway", default=os.environ.get("GATEWAY_URL", "http://gateway:4000"))
    parser.add_argument("--key", default=os.environ.get("EVAL_KEY"), help="the eval bot's virtual key")
    parser.add_argument("--prompt-version", default=os.environ.get("PROMPT_VERSION", "v1"))
    parser.add_argument("--set", default="all", choices=["all", "notebook"])
    parser.add_argument("--baseline", default=str(BASELINE))
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args(argv)
    if not args.key:
        from paper_agent.service.keys import load_keys
        args.key = load_keys().get("eval-bot")
    if not args.key:
        print("no eval key: run `make keys` first (keys.local.json) or pass --key")
        return 2
    client = httpx.Client(base_url=args.api, headers={"Authorization": f"Bearer {args.key}"}, timeout=120)
    pdf = make_pdf()
    up = client.post("/papers", files={"file": ("tinycoder.pdf", pdf, "application/pdf")})
    up.raise_for_status()
    paper_id = up.json()["paper_id"]
    Path("/tmp/eval_tinycoder.pdf").write_bytes(pdf)
    chunks = chunk_index(register_paper("/tmp/eval_tinycoder.pdf", paper_id))      # parse only: no API call
    golden = golden_set(args.set)
    print(f"golden set: {len(golden)} questions · prompt {args.prompt_version} · judge {JUDGE_VERSION} · "
          f"paper {paper_id} via {args.api}\n")
    metrics, rows = run_eval(args.prompt_version, http_ask(client, paper_id, args.prompt_version),
                             gateway_complete(args.gateway, args.key), chunks, golden, metrics=ALL_METRICS)
    calls, cost = sum(r["llm_calls"] for r in rows), sum(r["cost_usd"] for r in rows)
    print(f"\n{len(rows)} questions, {calls} agent model calls + {len(rows)} judge calls, agent cost ${cost:.6f}\n")
    model = os.environ.get("PROVIDER_MODEL") or os.environ.get("LLM_MODEL", "?")
    if args.write_baseline:
        record = baseline_record(metrics, args.prompt_version, model)
        Path(args.baseline).write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(metrics, indent=2))
        print(f"\nbaseline written to {args.baseline}")
        return 0
    if not Path(args.baseline).exists():
        print(f"no baseline at {args.baseline}: run `make eval-baseline` first")
        return 2
    base = json.loads(Path(args.baseline).read_text())
    gate.compare(base["metrics"], metrics, (f"baseline {base['prompt_version']}", args.prompt_version))
    failures = gate.regression_gate(metrics, base["metrics"])
    if failures:
        print(f"\n❌ BLOCKED: {args.prompt_version} can't ship. Worse than the baseline: " + " · ".join(failures))
        return 1
    print(f"\n✅ PASSED: {args.prompt_version} is no worse than the baseline "
          f"({base['prompt_version']}, {base['date']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
