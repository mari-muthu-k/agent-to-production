"""The regression gate (Day 4, 7.4: TODO 3 in the notebook). Answers wobble a little between runs, so a small
drop is allowed, except for valid citations: a made-up citation is never acceptable."""

TOLERANCE = {"valid_citations": 0.0, "readability": 10.0}      # every other metric: 0.10


def regression_gate(current: dict, baseline: dict) -> list:
    """The metrics that got worse than the baseline by more than their tolerance; [] means pass."""
    failures = []
    for metric, base in baseline.items():
        if metric not in current or base is None or current[metric] is None:
            continue
        allowed = TOLERANCE.get(metric, 0.10)
        if current[metric] < base - allowed:
            failures.append(f"{metric}: {base} -> {current[metric]}")
    return failures


def compare(baseline: dict, current: dict, names=("baseline", "current")) -> None:
    """The comparison table of 7.5 and `make eval`."""
    print(f"{'metric':20} {names[0]:>10} {names[1]:>10}")
    for metric in baseline:
        if metric in current:
            b, c = baseline[metric], current[metric]
            mark = "" if b is None or c is None or c >= b - TOLERANCE.get(metric, 0.10) else "  ❌"
            print(f"{metric:20} {fmt(b):>10} {fmt(c):>10}{mark}")


def fmt(value) -> str:
    return "-" if value is None else f"{value:g}"
