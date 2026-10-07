"""Shared loader for the N:M sparsity results, used by verify_all.py and analysis/advanced_visuals.py.

Reads the raw simulator logs directly (no hand-copied numbers). Every function returns plain dicts.

Data sets
  fair       verification/fair_perf.c (identical lean loops on every chip); the numbers to cite
  original   logs/runs/bench-*.log from nm_sparse_perf.c (generic loop; whole-job speedups superseded)
  rerun      verification/rerun-logs/ (fresh re-run of the original six tests)
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VER = ROOT / "verification"

FAIR_LOGS = {
    "berkeley": VER / "berkeley-logs/stock-fair_perf_dense_only.log",       # unmodified Gemmini, fair_perf -DDENSE_ONLY
    "nm24": VER / "fair-logs/fair_perf_24.log",                             # our chip, 2:4
    "nm48": VER / "fair-logs/fair_perf_48.log",                             # our chip, 4:8
    "nm24_sameprog": VER / "fair-logs/nm24-fair_perf_dense_only.log",       # our chip running berkeley's exact program
}
ORIG_STAMP = {"nm24": "20261006-025209", "nm48": "20261006-030607"}
TESTS = ("nm_sparse_debug", "nm_sparse_matmul", "nm_sparse_perf")

PERF_RE = re.compile(r"^(\w+)\s+total\s+(\d+)\s+cpu_issue_done\s+(\d+)\s+pairs\s+(\d+)\s+cyc/pair\s+(\d+) \|(.*)$", re.M)


def parse_log(path):
    """Everything a benchmark log reports: pass/fail, perf rows, matmul cycles, debug scores, reference checks."""
    t = Path(path).read_text(errors="replace")
    out = {"path": str(path), "passed": bool(re.search(r"^\w+ passed$", t, re.M)),
           "failed": "FAILED" in t or "mismatch" in t, "perf": {}, "matmul": {}, "debug": {},
           "ref_checks": t.count("matches the cpu reference")}
    for m in PERF_RE.finditer(t):
        ctr = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", m[6])}
        out["perf"][m[1]] = {"total": int(m[2]), "issued": int(m[3]), "tiles": int(m[4]), **ctr}
    for m in re.finditer(r"^(dense|sparse)\s+cycles: (\d+)", t, re.M):
        out["matmul"][m[1]] = int(m[2])
    for m in re.finditer(r"^  (.+?)\s+(\d+) / 256$", t, re.M):
        out["debug"][m[1].strip()] = int(m[2])
    return out


def load_fair(require_all=True):
    """{'berkeley','nm24','nm48','nm24_sameprog'} -> parsed log. Missing logs are skipped unless require_all."""
    res = {}
    for k, p in FAIR_LOGS.items():
        if p.exists():
            res[k] = parse_log(p)
        elif require_all and k != "nm24_sameprog":
            raise FileNotFoundError(p)
    return res


def load_original():
    return {c: {t: parse_log(ROOT / f"logs/runs/bench-{c}-{t}-{s}.log") for t in TESTS} for c, s in ORIG_STAMP.items()}


def load_rerun():
    out = {}
    for c in ORIG_STAMP:
        for t in TESTS:
            p = VER / f"rerun-logs/{c}-{t}.log"
            if p.exists():
                out.setdefault(c, {})[t] = parse_log(p)
    return out


def summary(fair):
    """The headline table: berkeley vs ours (2:4, 4:8) for grid / whole job (reuse, fresh) / processor issue time."""
    b, s24, s48 = fair["berkeley"]["perf"], fair["nm24"]["perf"], fair["nm48"]["perf"]
    def row(p, dense):
        k = "dense" if dense else "sparse"
        return {"tiles": p[k]["tiles"], "busy_reuse": p[k + "_reuse"]["ex_active"], "busy_fresh": p[k]["ex_active"],
                "reuse": p[k + "_reuse"]["total"], "fresh": p[k]["total"],
                "cpu_reuse": p["cpu_" + k + "_reuse"]["total"], "cpu_fresh": p["cpu_" + k]["total"]}
    out = {"berkeley": row(b, True), "2:4": row(s24, False), "4:8": row(s48, False),
           "ours_off_2:4": row(s24, True), "ours_off_4:8": row(s48, True)}
    if "nm24_sameprog" in fair:
        sp = fair["nm24_sameprog"]["perf"]
        out["ours_off_sameprog"] = {"reuse": sp["dense_reuse"]["total"], "fresh": sp["dense"]["total"],
                                    "busy_reuse": sp["dense_reuse"]["ex_active"]}
    return out


def load_fidelity():
    """pattern_fidelity.out -> {(distribution, rule): (energy_kept_pct, answer_error_pct)}"""
    fid = {}
    for l in (VER / "pattern_fidelity.out").read_text().splitlines():
        m = re.match(r"^(bell curve|heavy tail)\s+(paired 4:8|2:4|4:8|no pattern)\s+([\d.]+)%\s+([\d.]+)%", l)
        if m:
            fid[(m[1], m[2])] = (float(m[3]), float(m[4]))
    return fid
