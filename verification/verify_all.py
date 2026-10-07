#!/usr/bin/env python3
"""One command that re-checks every claim in the N:M sparsity work.

  A  original runs: logs <-> results/cycles.csv <-> analysis/all_data.csv, the re-run, quoted numbers  (cross_check.py)
  B  independent math check: sparse math == dense math, injected faults are caught        (independent_math_check.py)
  C  pattern comparison reproduces exactly                                                 (pattern_fidelity.py)
  D  fair benchmark: every run correct, grid halved, our chip == berkeley in normal mode, paper_data.json == logs
  E  the paper summary PDF prints exactly the numbers in the logs, and no superseded figures
  F  the code patches in the zip apply cleanly to berkeley's original code and rebuild the sparse-nm branches
  G  integrity: logs, results and test binaries match their recorded SHA-256 fingerprints

usage (from the project folder, with the chipyard conda python):
  ~/chipyard/.conda-env/bin/python verification/verify_all.py                 # everything
  ~/chipyard/.conda-env/bin/python verification/verify_all.py --skip-patches  # no git work
  ~/chipyard/.conda-env/bin/python verification/verify_all.py --write-manifest  # (re)record fingerprints, then verify
Writes verification/verify_all_report.md. Exit code 0 only if every check passes.
"""
import argparse, hashlib, json, re, subprocess, sys, tempfile, zipfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nmdata  # noqa: E402

ROOT, VER = nmdata.ROOT, nmdata.VER
PY = sys.executable
GEM = Path.home() / "chipyard/generators/gemmini"
REPOS = {"gemmini": (GEM, "8c3f9923"), "libgemmini": (GEM / "software/libgemmini", "ea8f7ed"),
         "gemmini-rocc-tests": (GEM / "software/gemmini-rocc-tests", "7c540b3")}
MANIFEST = VER / "MANIFEST.sha256"
MANIFEST_GLOBS = ["logs/**/*.log", "logs/**/*.txt", "logs/**/*.patch", "results/*.csv", "analysis/all_data.csv",
                  "verification/**/*.log", "verification/bin/*-baremetal", "verification/*.c", "verification/*.out"]

results = []   # (section, ok, message); ok is None for a skipped check


def check(section, ok, msg):
    results.append((section, bool(ok), msg))
    print(f"  {'PASS' if ok else 'FAIL'}  {msg}")
    return ok


def skip(section, msg):
    results.append((section, None, msg))
    print(f"  SKIP  {msg}")


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, **kw)


# ---------------------------------------------------------------- A
def sec_a():
    print("\nA. original runs, CSVs, re-run and quoted numbers (cross_check.py)")
    r = run([PY, str(VER / "cross_check.py")])
    n_ok = r.stdout.count("\n  ok ")
    fails = [l.strip() for l in r.stdout.splitlines() if l.strip().startswith("FAIL")]
    check("A", r.returncode == 0 and "ALL CHECKS PASSED" in r.stdout, f"cross_check.py: {n_ok} checks ok, {len(fails)} failed")
    for f in fails:
        check("A", False, f)


# ---------------------------------------------------------------- B
def sec_b():
    print("\nB. independent math check (numpy model, no shared code with the C tests)")
    r = run([PY, str(VER / "independent_math_check.py")])
    out = r.stdout.splitlines()
    eq = [l for l in out if "== dense" in l and "every case" in l]
    det = [re.search(r"detected in (\d+)/(\d+)", l) for l in out if "injected fault" in l]
    check("B", r.returncode == 0 and len(eq) == 2, f"sparse math equals dense math in every random case (2:4 and 4:8): {len(eq)}/2 patterns")
    check("B", len(det) == 8 and all(m and m[1] == m[2] for m in det), f"every injected hardware fault detected: {sum(1 for m in det if m and m[1] == m[2])}/8 fault types")
    stored = (VER / "independent_math_check.out").read_text().splitlines()
    check("B", out == stored, "output identical to the stored verification/independent_math_check.out")


# ---------------------------------------------------------------- C
def sec_c():
    print("\nC. pattern comparison (2:4 vs 4:8 vs paired 4:8)")
    r = run([PY, str(VER / "pattern_fidelity.py")])
    stored = (VER / "pattern_fidelity.out").read_text()
    check("C", r.returncode == 0 and r.stdout == stored, "pattern_fidelity.py reproduces pattern_fidelity.out exactly")
    fid = nmdata.load_fidelity()
    for dist in ("bell curve", "heavy tail"):
        k = [fid[(dist, rl)][0] for rl in ("paired 4:8", "2:4", "4:8", "no pattern")]
        check("C", k == sorted(k), f"{dist}: share kept ranks paired 4:8 < 2:4 < 4:8 < no pattern ({' < '.join(f'{v:.1f}%' for v in k)})")


# ---------------------------------------------------------------- D
def sec_d():
    print("\nD. fair benchmark (identical program on every chip)")
    fair = nmdata.load_fair()
    want = {"berkeley": 2, "nm24": 4, "nm48": 4, "nm24_sameprog": 2}
    for k, n in want.items():
        if k not in fair:
            check("D", False, f"{k}: log missing ({nmdata.FAIR_LOGS[k]})")
            continue
        f = fair[k]
        check("D", f["passed"] and not f["failed"] and f["ref_checks"] == n,
              f"{k}: passed, {f['ref_checks']}/{n} results match the processor's reference answer (2,048 values each)")
    s = nmdata.summary(fair)
    b = s["berkeley"]
    for pat in ("2:4", "4:8"):
        check("D", s[pat]["tiles"] * 2 == b["tiles"], f"{pat}: half the weight blocks ({s[pat]['tiles']} vs {b['tiles']})")
        check("D", s[pat]["busy_reuse"] * 2 <= b["busy_reuse"] * 1.1,
              f"{pat}: grid working time about halved ({b['busy_reuse']} -> {s[pat]['busy_reuse']}, {b['busy_reuse'] / s[pat]['busy_reuse']:.2f}x)")
    if "ours_off_sameprog" in s:
        o = s["ours_off_sameprog"]
        for k in ("reuse", "fresh"):
            dev = abs(o[k] - b[k]) / b[k]
            check("D", dev <= 0.01, f"our chip, sparsity off, berkeley's exact program: {k} {o[k]} vs berkeley {b[k]} ({dev * 100:.1f}% apart)")
    pj = VER / "paper_data.json"
    if pj.exists():
        d = json.loads(pj.read_text())
        pairs = [("berkeley", "berkeley"), ("ours_2:4", "2:4"), ("ours_4:8", "4:8")]
        same = all(d[a]["reuse"] == s[c]["reuse"] and d[a]["fresh"] == s[c]["fresh"] and d[a]["busy"] == s[c]["busy_reuse"]
                   and d[a]["tiles"] == s[c]["tiles"] for a, c in pairs)
        check("D", same, "paper_data.json (the paper's numbers) matches the raw logs")
    return s


# ---------------------------------------------------------------- E
def sec_e(s):
    print("\nE. the paper summary PDF prints the logged numbers")
    pdf = ROOT / "N-M-Sparsity-Paper-Summary.pdf"
    if not pdf.exists():
        return skip("E", f"{pdf.name} not found (reports were removed; rebuild with tools/report-builders/build_paper.py)")
    txt = run(["pdftotext", "-layout", str(pdf), "-"]).stdout.replace(" ", " ")
    b = s["berkeley"]
    need = {}
    for c in ("berkeley", "2:4", "4:8"):
        for k in ("busy_reuse", "reuse", "fresh"):
            need[f"{c} {k}"] = f"{s[c][k]:,}"
    for pat in ("2:4", "4:8"):
        need[f"{pat} grid speed-up"] = f"{b['busy_reuse'] / s[pat]['busy_reuse']:.2f}×"
        need[f"{pat} whole-job speed, reuse"] = f"{b['reuse'] / s[pat]['reuse']:.2f}×"
        need[f"{pat} whole-job speed, fresh"] = f"{b['fresh'] / s[pat]['fresh']:.2f}×"
    missing = [f"{k} ({v})" for k, v in need.items() if v not in txt]
    check("E", not missing, f"all {len(need)} key numbers appear in the PDF" + (f"; missing: {', '.join(missing)}" if missing else ""))
    # the only allowed mention is the "1.36–1.44× ... shouldn't be cited" note in the limits section
    note = len(re.findall(r"1\.36–1\.44×", txt))
    stale = len(re.findall(r"(?<!1\.36–)1\.44×", txt)) + len(re.findall(r"1\.36×(?!–)", txt))
    check("E", stale == 0 and note <= 1, f"superseded 1.44×/1.36× figures appear only in the 'don't cite' note ({stale} stray mention(s))")


# ---------------------------------------------------------------- F
def sec_f():
    print("\nF. code patches in the zip rebuild the sparse-nm branches from berkeley's code")
    zips = sorted(ROOT.glob("Gemmini-NM-Sparsity-*.zip"))
    if not zips:
        return skip("F", "no Gemmini-NM-Sparsity-*.zip in the project folder (remake the zip to test its patches)")
    if not GEM.exists():
        return check("F", False, f"{GEM} not found (needs the local chipyard checkout)")
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(zips[-1]) as z:
        z.extractall(tmp, [n for n in z.namelist() if "/code-patches/" in n])
        pdir = next(Path(tmp).rglob("code-patches"))
        for name, (repo, base) in REPOS.items():
            patches = sorted((pdir / name).glob("*.patch"))
            clone = Path(tmp) / f"clone-{name}"
            ok = run(["git", "clone", "-q", "--no-checkout", str(repo), str(clone)]).returncode == 0
            ok = ok and run(["git", "-C", str(clone), "checkout", "-q", base]).returncode == 0
            am = run(["git", "-C", str(clone), "-c", "user.name=verify", "-c", "user.email=verify@localhost", "am", "-q", *map(str, patches)])
            tree = run(["git", "-C", str(clone), "rev-parse", "HEAD^{tree}"]).stdout.strip()
            want = run(["git", "-C", str(repo), "rev-parse", "sparse-nm^{tree}"]).stdout.strip()
            check("F", ok and am.returncode == 0 and tree == want,
                  f"{name}: {len(patches)} patches from {zips[-1].name} apply to {base} and match sparse-nm exactly")


# ---------------------------------------------------------------- G
def manifest_files():
    files = set()
    for g in MANIFEST_GLOBS:
        files.update(p for p in ROOT.glob(g) if p.is_file())
    return sorted(files)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_manifest():
    lines = [f"{sha(p)}  {p.relative_to(ROOT)}" for p in manifest_files()]
    MANIFEST.write_text("\n".join(lines) + "\n")
    print(f"wrote {MANIFEST.relative_to(ROOT)} ({len(lines)} files)")


def sec_g():
    print("\nG. integrity of logs, results and test programs")
    if not MANIFEST.exists():
        return check("G", False, "no MANIFEST.sha256 yet (run with --write-manifest once)")
    rec = dict(reversed(l.split("  ", 1)) for l in MANIFEST.read_text().splitlines() if l)
    changed = [n for n, h in rec.items() if not (ROOT / n).exists() or sha(ROOT / n) != h]
    new = [str(p.relative_to(ROOT)) for p in manifest_files() if str(p.relative_to(ROOT)) not in rec]
    msg = (f"all {len(rec)} recorded files unchanged since the manifest was written" if not changed
           else f"{len(changed)} of {len(rec)} recorded files changed or missing: {', '.join(changed[:5])}")
    check("G", not changed, msg)
    if new:
        print(f"  note  {len(new)} newer file(s) not in the manifest (e.g. {new[0]}); --write-manifest to add them")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-patches", action="store_true", help="skip section F (git clone + am)")
    ap.add_argument("--write-manifest", action="store_true", help="record SHA-256 fingerprints before verifying")
    a = ap.parse_args()
    print(f"verify_all.py  {datetime.now():%Y-%m-%d %H:%M}  project: {ROOT}")
    if a.write_manifest:
        write_manifest()
    sec_a(); sec_b(); sec_c()
    s = sec_d()
    sec_e(s)
    if not a.skip_patches:
        sec_f()
    sec_g()

    n_fail = sum(1 for _, ok, _ in results if ok is False)
    n_skip = sum(1 for _, ok, _ in results if ok is None)
    verdict = "ALL CHECKS PASSED" if n_fail == 0 else f"{n_fail} CHECK(S) FAILED"
    tail = f"{len(results) - n_skip} checks" + (f", {n_skip} skipped" if n_skip else "")
    print(f"\n{verdict}  ({tail})")
    names = {"A": "Original runs and quoted numbers", "B": "Independent math check", "C": "Pattern comparison",
             "D": "Fair benchmark", "E": "Paper summary PDF", "F": "Code patches in the zip", "G": "File integrity"}
    md = [f"# Verification report", "", f"Run {datetime.now():%Y-%m-%d %H:%M} with `verification/verify_all.py`. "
          f"**{verdict}** ({tail}).", ""]
    for sec in names:
        rows = [(ok, m) for s_, ok, m in results if s_ == sec]
        if rows:
            md += [f"## {sec}. {names[sec]}", ""] + [f"- {'SKIP' if ok is None else 'PASS' if ok else '**FAIL**'}: {m}" for ok, m in rows] + [""]
    (VER / "verify_all_report.md").write_text("\n".join(md))
    print(f"report: {(VER / 'verify_all_report.md').relative_to(ROOT)}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
