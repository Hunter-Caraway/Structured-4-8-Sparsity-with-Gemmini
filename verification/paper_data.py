#!/usr/bin/env python3
# pulls every number the paper summary uses straight out of the simulator logs, writes
# verification/paper_data.json and draws verification/paper_chart.png. nothing is typed in by hand.
#
# run: ~/chipyard/.conda-env/bin/python verification/paper_data.py
import json, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
LOGS = {
    "berkeley": ROOT / "verification/berkeley-logs/stock-fair_perf_dense_only.log",   # fair_perf.c built -DDENSE_ONLY
    # fair_perf: the same lean loops as berkeley_dense_perf, plus sparse versions written the same way
    "ours_2:4": ROOT / "verification/fair-logs/fair_perf_24.log",
    "ours_4:8": ROOT / "verification/fair-logs/fair_perf_48.log",
}


def perf(path):
    t = Path(path).read_text(errors="replace")
    assert re.search(r"^\w+ passed$", t, re.M) and "FAILED" not in t, f"{path} did not pass"
    out = {}
    for m in re.finditer(r"^(\w+)\s+total\s+(\d+)\s+cpu_issue_done\s+\d+\s+pairs\s+(\d+)\s.*?ex_active=(\d+)", t, re.M):
        out[m[1]] = {"total": int(m[2]), "tiles": int(m[3]), "busy": int(m[4])}
    out["_checks"] = t.count("matches the cpu reference")
    return out, t


berk, berk_txt = perf(LOGS["berkeley"])
assert berk_txt.count("matches the cpu reference") == 2, "berkeley run must check both results against the cpu"
p24, _ = perf(LOGS["ours_2:4"])
p48, _ = perf(LOGS["ours_4:8"])
assert p24["_checks"] == 4 and p48["_checks"] == 4, "every fair_perf run must match the cpu reference"

d = {
    "berkeley": {"busy": berk["dense_reuse"]["busy"], "reuse": berk["dense_reuse"]["total"], "fresh": berk["dense"]["total"],
                 "busy_fresh": berk["dense"]["busy"], "tiles": berk["dense"]["tiles"]},
    "ours_2:4": {"busy": p24["sparse_reuse"]["busy"], "reuse": p24["sparse_reuse"]["total"], "fresh": p24["sparse"]["total"],
                 "busy_fresh": p24["sparse"]["busy"], "tiles": p24["sparse"]["tiles"]},
    "ours_4:8": {"busy": p48["sparse_reuse"]["busy"], "reuse": p48["sparse_reuse"]["total"], "fresh": p48["sparse"]["total"],
                 "busy_fresh": p48["sparse"]["busy"], "tiles": p48["sparse"]["tiles"]},
    # our modified chip with sparse mode OFF, to show the change doesn't slow down normal use
    "ours_off_2:4": {"busy": p24["dense_reuse"]["busy"], "reuse": p24["dense_reuse"]["total"], "fresh": p24["dense"]["total"]},
    "ours_off_4:8": {"busy": p48["dense_reuse"]["busy"], "reuse": p48["dense_reuse"]["total"], "fresh": p48["dense"]["total"]},
    # processor-only time (accelerator removed), for explaining where the time goes
    "cpu": {"berkeley_fresh": berk["cpu_dense"]["total"], "berkeley_reuse": berk["cpu_dense_reuse"]["total"],
            "2:4_fresh": p24["cpu_sparse"]["total"], "2:4_reuse": p24["cpu_sparse_reuse"]["total"],
            "4:8_fresh": p48["cpu_sparse"]["total"], "4:8_reuse": p48["cpu_sparse_reuse"]["total"]},
}
# berkeley's exact program (fair_perf.c built -DDENSE_ONLY) run on OUR 2:4 chip: separates a hardware
# difference from a program-layout difference when comparing our normal mode with berkeley's
same = ROOT / "verification/fair-logs/nm24-fair_perf_dense_only.log"
if same.exists() and "fair_perf passed" in same.read_text():
    ps, st = perf(same)
    assert st.count("matches the cpu reference") == 2
    d["ours_off_sameprog"] = {"busy": ps["dense_reuse"]["busy"], "reuse": ps["dense_reuse"]["total"], "fresh": ps["dense"]["total"]}
json.dump(d, open(ROOT / "verification/paper_data.json", "w"), indent=1)
print(json.dumps(d, indent=1))

# ---- chart: one measure (cycles), one axis, three series in fixed palette order ----
SER = [("berkeley", "Berkeley original", "#2a78d6"), ("ours_2:4", "Ours, 2:4", "#eb6834"), ("ours_4:8", "Ours, 4:8", "#1baf7a")]
ROWS = [("busy", "Multiplier grid working time\n(weights reused)"), ("reuse", "Whole job, weights reused\n(the realistic case)"),
        ("fresh", "Whole job, fresh weights\nevery step (stress test)")]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=200)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
bar_h, gap = 0.22, 0.03
for r, (key, label) in enumerate(ROWS):
    for s, (sk, sname, col) in enumerate(SER):
        y = r + (s - 1) * (bar_h + gap)
        v = d[sk][key]
        ax.barh(y, v, height=bar_h, color=col, edgecolor=SURF, linewidth=1, zorder=3)
        txt = f"{v:,}"
        if sk != "berkeley":
            sp = d["berkeley"][key] / v
            txt += f"   {sp:.2f}× {'faster' if sp > 1.02 else 'slower' if sp < 0.98 else '(about the same)'}"
        ax.text(v + 45, y, txt, va="center", ha="left", fontsize=8.6, color=INK2 if sk == "berkeley" else INK, zorder=4)
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([l for _, l in ROWS], color=INK)
ax.invert_yaxis()
ax.set_xlim(0, max(d[s][k] for s, _, _ in SER for k, _ in ROWS) * 1.42)
ax.set_xlabel("clock cycles to finish the same job (shorter is better)", color=INK2, fontsize=9)
ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0); ax.set_axisbelow(True)
for side in ("top", "right", "left"): ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#c9c8c2"); ax.tick_params(axis="x", colors=INK2, labelsize=8.5); ax.tick_params(axis="y", length=0)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
handles = [matplotlib.patches.Patch(color=c, label=n) for _, n, c in SER]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=3, frameon=False, fontsize=9, labelcolor=INK)
fig.tight_layout()
fig.savefig(ROOT / "verification/paper_chart.png", facecolor=SURF)
print("chart written")
