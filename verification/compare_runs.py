#!/usr/bin/env python3
"""Compare a new simulation run against the recorded results, number for number.

The simulation is deterministic, so a correct re-run must reproduce every cycle count and hardware counter
exactly. run-flow.sh writes new runs to verification/runs/<stamp>/ with these log names:

  fair benchmark      nm24.log  nm48.log  berkeley.log  nm24_sameprog.log
  original six tests  orig-nm24-<test>.log  orig-nm48-<test>.log   (<test> = nm_sparse_debug|matmul|perf)

usage: ~/chipyard/.conda-env/bin/python verification/compare_runs.py verification/runs/<stamp>
Exit code 0 only if every log present matches its recorded counterpart (and at least one log was found).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nmdata  # noqa: E402


def pairs(run):
    for key, rec in nmdata.FAIR_LOGS.items():
        yield run / f"{key}.log", rec, f"fair benchmark, {key}"
    for cfg, stamp in nmdata.ORIG_STAMP.items():
        for t in nmdata.TESTS:
            yield run / f"orig-{cfg}-{t}.log", nmdata.ROOT / f"logs/runs/bench-{cfg}-{t}-{stamp}.log", f"original {t}, {cfg}"


def numbers(p):
    return {k: p[k] for k in ("perf", "matmul", "debug", "ref_checks")}


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run = Path(sys.argv[1]).resolve()
    found = fails = 0
    print(f"comparing {run} with the recorded results")
    for new, rec, label in pairs(run):
        if not new.exists():
            continue
        found += 1
        a, b = nmdata.parse_log(rec), nmdata.parse_log(new)
        ok_run = b["passed"] and not b["failed"]
        same = numbers(a) == numbers(b)
        n = sum(len(v) for v in a["perf"].values()) + len(a["matmul"]) + len(a["debug"])
        if not (ok_run and same):
            fails += 1
        status = "PASS" if ok_run and same else "FAIL"
        detail = f"all {n} numbers identical" if same else "numbers differ from the recorded run"
        print(f"  {status}  {label}: {'passed' if ok_run else 'DID NOT PASS'}, {detail}")
        if not same:
            for v in sorted(set(a["perf"]) | set(b["perf"])):
                ra, rb = a["perf"].get(v), b["perf"].get(v)
                if ra is None or rb is None:
                    print(f"        {v}: {'missing from the new run' if rb is None else 'not in the recorded run'}")
                elif ra != rb:
                    diffs = ", ".join(f"{k} {ra[k]} -> {rb.get(k)}" for k in ra if ra[k] != rb.get(k))
                    print(f"        {v}: {diffs}")
            for k in ("matmul", "debug", "ref_checks"):
                if a[k] != b[k]:
                    print(f"        {k}: recorded {a[k]}, new {b[k]}")
    if not found:
        print("  FAIL  no recognised logs in this folder")
        sys.exit(1)
    print(f"\n{'ALL RUNS MATCH' if not fails else f'{fails} RUN(S) DIFFER'}  ({found} logs compared)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
