#!/usr/bin/env python3
# traces every benchmark number back to the raw simulator output and checks it three ways:
#   1. logs/runs/bench-*.log (original 02:52-03:21 runs) vs results/cycles.csv and analysis/all_data.csv
#   2. original logs vs the fresh re-run in verification/rerun-logs (same simulators, same binaries)
#   3. every speedup / percentage quoted in the reports, recomputed from the raw numbers
#
# run: ~/chipyard/.conda-env/bin/python verification/cross_check.py
import csv, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORIG = {"nm24": "20261006-025209", "nm48": "20261006-030607"}
fails = 0


def say(ok, msg):
    global fails
    fails += not ok
    print(("  ok    " if ok else "  FAIL  ") + msg)


def parse(path):
    """-> {'passed': bool, 'perf': {variant: {...}}, 'matmul': {dense, sparse}, 'debug': {name: hits}}"""
    t = Path(path).read_text(errors="replace")
    r = {"passed": bool(re.search(r"^nm_sparse_\w+ passed$", t, re.M)),
         "failed": "FAILED" in t, "perf": {}, "matmul": {}, "debug": {}}
    for m in re.finditer(r"^(\w+)\s+total\s+(\d+)\s+cpu_issue_done\s+(\d+)\s+pairs\s+(\d+)\s+cyc/pair\s+(\d+) \|(.*)$", t, re.M):
        ctr = dict((k, int(v)) for k, v in re.findall(r"(\w+)=(\d+)", m[6]))
        r["perf"][m[1]] = {"total": int(m[2]), "issued": int(m[3]), "tiles": int(m[4]), **ctr}
    for m in re.finditer(r"^(dense|sparse)\s+cycles: (\d+)", t, re.M):
        r["matmul"][m[1]] = int(m[2])
    for m in re.finditer(r"^  (.+?)\s+(\d+) / 256$", t, re.M):
        r["debug"][m[1].strip()] = int(m[2])
    return r


def load(dirname, pattern):
    out = {}
    for cfg in ("nm24", "nm48"):
        out[cfg] = {}
        for t in ("nm_sparse_debug", "nm_sparse_matmul", "nm_sparse_perf"):
            p = ROOT / dirname / pattern.format(cfg=cfg, t=t, stamp=ORIG[cfg])
            out[cfg][t] = parse(p) if p.exists() else None
    return out


orig = load("logs/runs", "bench-{cfg}-{t}-{stamp}.log")
rerun = load("verification/rerun-logs", "{cfg}-{t}.log")

print("1. every test passed its own correctness check (original runs)")
for cfg in orig:
    for t, r in orig[cfg].items():
        say(r["passed"] and not r["failed"], f"{cfg} {t}: printed '{t} passed' and no failure line")

print("\n2. results/cycles.csv matches the raw logs, row by row")
cfgname = {"GemminiNMRocketConfig": "nm24", "GemminiNM48RocketConfig": "nm48"}
rows = list(csv.DictReader(open(ROOT / "results/cycles.csv")))
for row in rows:
    cfg = cfgname[row["config"]]
    r = orig[cfg][row["test"]]
    if row["test"] == "nm_sparse_matmul":
        say(int(row["total_cycles"]) == r["matmul"][row["variant"]], f"{cfg} matmul {row['variant']}: csv {row['total_cycles']} = log {r['matmul'][row['variant']]}")
    else:
        p = r["perf"][row["variant"]]
        ok = (int(row["total_cycles"]), int(row["array_busy_cycles"]), int(row["tiles"])) == (p["total"], p["ex_active"], p["tiles"])
        say(ok, f"{cfg} perf {row['variant']:<17} csv total/busy/tiles {row['total_cycles']}/{row['array_busy_cycles']}/{row['tiles']} = log")
say(len(rows) == 26, f"cycles.csv has {len(rows)} data rows (2 configs x (2 matmul + 11 perf))")

print("\n3. analysis/all_data.csv (the chart data) matches the raw logs")
n_checked = 0
for row in csv.DictReader(open(ROOT / "analysis/all_data.csv")):
    cfg = "nm24" if row["pattern"] == "2:4" else "nm48"
    if row["test"] == "nm_sparse_perf":
        p = orig[cfg]["nm_sparse_perf"]["perf"][row["variant"]]
        for col, key in [("total", "total"), ("cpu_issue_done", "issued"), ("tiles", "tiles"), ("ex_active", "ex_active"),
                         ("rs_full", "rs_full"), ("spad_a_wait", "spad_a_wait"), ("spad_b_wait", "spad_b_wait"),
                         ("spad_d_wait", "spad_d_wait"), ("main_ex", "main_ex")]:
            n_checked += 1
            if int(row[col]) != p[key]:
                say(False, f"{cfg} {row['variant']} {col}: csv {row[col]} vs log {p[key]}")
    elif row["test"] == "nm_sparse_debug":
        n_checked += 1
        if int(row["matches"]) != orig[cfg]["nm_sparse_debug"]["debug"][row["check"]]:
            say(False, f"{cfg} debug {row['check']}")
say(True, f"{n_checked} values in all_data.csv compared, see any FAIL lines above")

print("\n4. fresh re-run (verification/rerun-logs) reproduces the original numbers exactly")
if all(rerun[c][t] for c in rerun for t in rerun[c]):
    for cfg in orig:
        for t in orig[cfg]:
            a, b = orig[cfg][t], rerun[cfg][t]
            say(b["passed"] and not b["failed"], f"{cfg} {t}: re-run passed")
            say(a["perf"] == b["perf"] and a["matmul"] == b["matmul"] and a["debug"] == b["debug"],
                f"{cfg} {t}: all {sum(len(v) for v in a['perf'].values()) + len(a['matmul']) + len(a['debug'])} numbers identical to the original")
else:
    say(False, "re-run logs missing or incomplete")

print("\n5. every quoted number, recomputed from the raw logs")
def perf(cfg, v, k="total"): return orig[cfg]["nm_sparse_perf"]["perf"][v][k]
quotes = []
for cfg, pat in (("nm24", "2:4"), ("nm48", "4:8")):
    quotes += [
        (f"{pat} array busy, no reuse: dense {perf(cfg,'dense','ex_active')} / sparse {perf(cfg,'sparse','ex_active')}", perf(cfg, "dense", "ex_active") / perf(cfg, "sparse", "ex_active"), 1.88),
        (f"{pat} array busy, weight reuse: dense {perf(cfg,'dense_reuse','ex_active')} / sparse {perf(cfg,'sparse_reuse','ex_active')}", perf(cfg, "dense_reuse", "ex_active") / perf(cfg, "sparse_reuse", "ex_active"), 2.00),
        (f"{pat} end to end, weight reuse: {perf(cfg,'dense_reuse')} / {perf(cfg,'sparse_reuse')}", perf(cfg, "dense_reuse") / perf(cfg, "sparse_reuse"), {"nm24": 1.44, "nm48": 1.36}[cfg]),
        (f"{pat} end to end, no reuse, meta ahead: {perf(cfg,'dense')} / {perf(cfg,'sparse')}", perf(cfg, "dense") / perf(cfg, "sparse"), {"nm24": 0.74, "nm48": 0.64}[cfg]),
        (f"{pat} end to end, no reuse, natural: {perf(cfg,'dense')} / {perf(cfg,'sparse_nat')}", perf(cfg, "dense") / perf(cfg, "sparse_nat"), {"nm24": 0.84, "nm48": 0.65}[cfg]),
        (f"{pat} array busy per tile, reuse: dense {perf(cfg,'dense_reuse','ex_active')}/64, sparse {perf(cfg,'sparse_reuse','ex_active')}/32",
         (perf(cfg, "dense_reuse", "ex_active") / 64) / (perf(cfg, "sparse_reuse", "ex_active") / 32), 1.00),
    ]
for text, got, quoted in quotes:
    say(round(got, 2) == quoted, f"{text} = {got:.3f}  (report says {quoted:.2f})")
fr = []
for cfg in ("nm24", "nm48"):
    for v, c in [("dense", "cpu_dense"), ("sparse", "cpu_sparse"), ("sparse_nat", "cpu_sparse_nat"),
                 ("dense_reuse", "cpu_dense_reuse"), ("sparse_reuse", "cpu_sparse_reuse")]:
        fr.append(perf(cfg, c) / perf(cfg, v))
say(round(min(fr) * 100) == 67 and round(max(fr) * 100) == 86,
    f"CPU-only issue time as a share of each total: {min(fr)*100:.0f}% to {max(fr)*100:.0f}%  (report says 67-86%)")

print(f"\n{'ALL CHECKS PASSED' if not fails else str(fails) + ' CHECK(S) FAILED'}")
sys.exit(1 if fails else 0)
