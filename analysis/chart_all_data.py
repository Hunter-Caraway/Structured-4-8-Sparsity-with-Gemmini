#!/usr/bin/env python3
# charts every number the benchmark runs printed, not just the ones in cycles.csv
# changed: new script. parses logs/runs/bench-*.log directly (all 11 perf counters,
# the matmul test, the debug test scores), checks them against results/cycles.csv,
# and reuses the theme + helpers from make_charts.py so the figures match fig1-7
#
# run with the chipyard conda env, it has matplotlib + pandas:
#   ~/chipyard/.conda-env/bin/python analysis/chart_all_data.py
# writes analysis/charts/fig8..fig11-<theme>.png and analysis/all_data.csv (table view)

import re
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from make_charts import THEMES, style, legend, fmt, save, root, here

# ---------------------------------------------------------------------------
# data

PATTERN = {"nm24": "2:4", "nm48": "4:8"}
PATTERNS = ["2:4", "4:8"]

def latest_logs():
    # bench-<nm24|nm48>-<test>-<timestamp>.log, keep the newest per (pattern, test)
    logs = {}
    for f in sorted((root / "logs" / "runs").glob("bench-*.log")):
        m = re.match(r"bench-(nm\d\d)-(\w+)-(\d{8}-\d{6})\.log", f.name)
        logs[(PATTERN[m[1]], m[2])] = (m[3], f)
    return logs

perf_line = re.compile(r"^(\w+)\s+total\s+(\d+)\s+cpu_issue_done\s+(\d+)\s+pairs\s+(\d+)\s+cyc/pair\s+(\d+)\s+\|(.*)$")
matmul_line = re.compile(r"^(dense|sparse)\s+cycles:\s+(\d+)\s+\((\d+) preload/compute pairs")
debug_line = re.compile(r"^\s+(.+?)\s+(\d+) / (\d+)$")

perf_rows, matmul_rows, debug_rows = [], [], []
for (p, test), (stamp, f) in latest_logs().items():
    for line in f.read_text().splitlines():
        if test == "nm_sparse_perf" and (m := perf_line.match(line)):
            row = dict(pattern=p, timestamp=stamp, variant=m[1], total=int(m[2]),
                       cpu_issue_done=int(m[3]), tiles=int(m[4]), cyc_per_tile=int(m[5]))
            row.update({k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", m[6])})
            perf_rows.append(row)
        elif test == "nm_sparse_matmul" and (m := matmul_line.match(line)):
            matmul_rows.append(dict(pattern=p, timestamp=stamp, variant=m[1], total=int(m[2]), tiles=int(m[3])))
        elif test == "nm_sparse_debug" and (m := debug_line.match(line)):
            debug_rows.append(dict(pattern=p, timestamp=stamp, check=m[1], matches=int(m[2]), of=int(m[3])))

perf = pd.DataFrame(perf_rows)
matmul = pd.DataFrame(matmul_rows)
debug = pd.DataFrame(debug_rows)

# the logs and cycles.csv come from the same runs, so the totals have to agree
csv = pd.read_csv(root / "results" / "cycles.csv")
csv["pattern"] = csv["config"].map({"GemminiNMRocketConfig": "2:4", "GemminiNM48RocketConfig": "4:8"})
csv = csv.sort_values("timestamp").groupby(["pattern", "test", "variant"]).last()
for name, df in [("nm_sparse_perf", perf), ("nm_sparse_matmul", matmul)]:
    for r in df.itertuples():
        assert csv.loc[(r.pattern, name, r.variant), "total_cycles"] == r.total, (name, r)

# table view of everything charted below
(pd.concat([perf.assign(test="nm_sparse_perf"), matmul.assign(test="nm_sparse_matmul"),
            debug.assign(test="nm_sparse_debug")], ignore_index=True).convert_dtypes()
   .to_csv(here / "all_data.csv", index=False))

VARIANTS = list(dict.fromkeys(perf.variant))  # log order: accelerator runs, then cpu_*
ACCEL = [v for v in VARIANTS if not v.startswith("cpu_")]
COUNTERS = ["total", "cpu_issue_done", "main_ex", "ex_active", "rs_full", "ctrl_q_block",
            "overlap_haz", "spad_a_wait", "spad_b_wait", "spad_d_wait"]

def pv(p, v):
    return perf[(perf.pattern == p) & (perf.variant == v)].iloc[0]

# blue sequential ramp from the reference palette. light: near zero = lightest step.
# dark: near zero recedes toward the dark surface, so the ramp runs the other way
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
RAMP = {"light": BLUE, "dark": BLUE[::-1][:-1]}

# ---------------------------------------------------------------------------
# layout helpers (title/legend in inches so tall figures keep the same spacing)

def header(fig, t, text, sub):
    h = fig.get_figheight()
    fig.text(0.012, 1 - 0.06 / h, text, ha="left", va="top", fontsize=13, fontweight="bold", color=t["fg"])
    fig.text(0.012, 1 - 0.32 / h, sub, ha="left", va="top", fontsize=9.5, color=t["fg2"])

def legend_at(fig, t, items, inches=0.58):
    legend(fig, t, items, y=1 - inches / fig.get_figheight())

def top(fig, inches):
    return 1 - inches / fig.get_figheight()

def figure(t, w, h, nrows=1, ncols=1, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), squeeze=False, **kw)
    fig.patch.set_facecolor(t["surface"])
    return fig, axes.ravel()

def ink_on(hexcolor):
    r, g, b = (int(hexcolor[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return "#0b0b0b" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5 else "#ffffff"

# ---------------------------------------------------------------------------
# charts

def chart_counters(theme):
    # every perf counter for every variant: it's a table, so every cell gets its number
    t = THEMES[theme]
    cmap = LinearSegmentedColormap.from_list("seq", RAMP[theme])
    vmax = perf[COUNTERS].to_numpy().max()
    fig, axes = figure(t, 9.6, 10.2, nrows=2)
    header(fig, t, "Every hardware counter from nm_sparse_perf, both sparsity patterns",
           "64×128×32 matmul · cycles, snapshot at the fence · " + ("darker" if theme == "light" else "brighter") + " = more cycles · cpu_* rows issue no Gemmini instructions")
    for ax, p in zip(axes, PATTERNS):
        grid = [[int(pv(p, v)[c]) for c in COUNTERS] for v in VARIANTS]
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        for i, row in enumerate(grid):
            for j, val in enumerate(row):
                ax.text(j, i, fmt(val), ha="center", va="center", fontsize=8,
                        color=ink_on(matplotlib_hex(cmap(val / vmax))))
        # 2px surface gaps between cells
        ax.set_xticks([x - 0.5 for x in range(1, len(COUNTERS))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(VARIANTS))], minor=True)
        ax.grid(which="minor", color=t["surface"], linewidth=2)
        ax.tick_params(which="both", length=0)
        ax.set_xticks(range(len(COUNTERS)), COUNTERS, rotation=30, ha="right", fontsize=9, color=t["fg"])
        ax.set_yticks(range(len(VARIANTS)),
                      [f"{v}  ({pv(p, v).tiles} tiles, {pv(p, v).cyc_per_tile}/tile)" for v in VARIANTS],
                      fontsize=9, color=t["fg"])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(p, loc="left", fontsize=11, color=t["fg"], fontweight="bold")
    fig.subplots_adjust(left=0.27, right=0.99, top=top(fig, 0.95), bottom=0.07, hspace=0.32)
    save(fig, "fig8-all-counters", theme)

def matplotlib_hex(rgba):
    return "#" + "".join(f"{round(c * 255):02x}" for c in rgba[:3])

def chart_waits(theme):
    # the three scratchpad wait counters, one panel each, color = pattern like fig3
    t = THEMES[theme]
    fig, axes = figure(t, 9.6, 4.4, ncols=3, sharey=True)
    header(fig, t, "Scratchpad wait counters: how long the array sat waiting on each operand",
           "accelerator variants only · cycles with no A / B / D row ready to stream · lower is better")
    legend_at(fig, t, [("2:4", t["p24"], "box"), ("4:8", t["p48"], "box")])
    for ax, c in zip(axes, ["spad_a_wait", "spad_b_wait", "spad_d_wait"]):
        style(ax, t)
        for i, v in enumerate(ACCEL):
            for k, (p, col) in enumerate([("2:4", t["p24"]), ("4:8", t["p48"])]):
                y = -i - k * 0.38
                val = int(pv(p, v)[c])
                ax.barh(y, val, height=0.32, color=col)
                ax.text(val + 40, y, fmt(val), va="center", fontsize=8, color=t["fg2"])
        ax.set_yticks([-i - 0.19 for i in range(len(ACCEL))], ACCEL, fontsize=9.5, color=t["fg"])
        ax.set_xlim(0, 3200)
        ax.set_title(c, loc="left", fontsize=11, color=t["fg"], fontweight="bold")
        ax.set_xlabel("cycles", fontsize=9, color=t["fg3"])
    fig.subplots_adjust(left=0.11, right=0.99, top=top(fig, 1.15), bottom=0.12, wspace=0.12)
    save(fig, "fig9-scratchpad-waits", theme)

def chart_patterns(theme):
    # every benchmark's total cycles, 2:4 next to 4:8
    t = THEMES[theme]
    rows = [("nm_sparse_matmul", v, matmul[(matmul.pattern == "2:4") & (matmul.variant == v)].total.iloc[0],
             matmul[(matmul.pattern == "4:8") & (matmul.variant == v)].total.iloc[0]) for v in ["dense", "sparse"]]
    rows += [("nm_sparse_perf", v, pv("2:4", v).total, pv("4:8", v).total) for v in VARIANTS]
    fig, axes = figure(t, 9.6, 6.6)
    ax = axes[0]
    style(ax, t)
    header(fig, t, "Total cycles for every benchmark run: 2:4 against 4:8",
           "same matrices for both patterns · right column = 4:8 cycles ÷ 2:4 cycles (above 1 = 4:8 is slower)")
    legend_at(fig, t, [("2:4", t["p24"], "o"), ("4:8", t["p48"], "o")])
    labels = []
    for i, (test, v, a, b) in enumerate(rows):
        y = -i
        # 2:4 sits just above 4:8 so equal values don't hide each other
        ax.plot([a, b], [y + 0.13, y - 0.13], color=t["grid"], linewidth=2, zorder=1)
        for x, dy, col in [(a, 0.13, t["p24"]), (b, -0.13, t["p48"])]:
            ax.plot([x], [y + dy], marker="o", markersize=8, color=col, markeredgecolor=t["surface"], markeredgewidth=2, zorder=3)
        ax.text(4150, y, f"{b / a:.2f}×", va="center", ha="right", fontsize=9, color=t["fg2"])
        labels.append(f"{v}" + ("  (matmul test)" if test == "nm_sparse_matmul" else ""))
    ax.axhline(-1.5, color=t["grid"], linewidth=0.8)
    ax.set_yticks([-i for i in range(len(rows))], labels, fontsize=9.5, color=t["fg"])
    ax.set_xlim(0, 4200)
    ax.set_ylim(-len(rows) + 0.4, 0.6)
    ax.set_xlabel("total cycles", fontsize=9, color=t["fg3"])
    fig.subplots_adjust(left=0.23, right=0.98, top=top(fig, 0.95), bottom=0.08)
    save(fig, "fig10-pattern-totals", theme)

def chart_debug(theme):
    # nm_sparse_debug: how many of the 256 outputs match each hypothesis about what the array did
    t = THEMES[theme]
    checks = list(dict.fromkeys(debug.check))
    fig, axes = figure(t, 9.6, 4.2)
    ax = axes[0]
    style(ax, t)
    header(fig, t, "nm_sparse_debug: only the correct sparse result matches the hardware output",
           "outputs (of 256) that match each candidate result · the wrong candidates are the bugs this test catches")
    legend_at(fig, t, [("2:4", t["p24"], "box"), ("4:8", t["p48"], "box")])
    for i, c in enumerate(checks):
        for k, (p, col) in enumerate([("2:4", t["p24"]), ("4:8", t["p48"])]):
            y = -i - k * 0.38
            val = int(debug[(debug.pattern == p) & (debug.check == c)].matches.iloc[0])
            ax.barh(y, val, height=0.32, color=col)
            ax.text(val + 3, y, f"{val}", va="center", fontsize=8.5, color=t["fg2"])
    ax.set_yticks([-i - 0.19 for i in range(len(checks))], checks, fontsize=9.5, color=t["fg"])
    ax.set_xlim(0, 280)
    ax.set_xlabel("matching outputs (of 256)", fontsize=9, color=t["fg3"])
    fig.subplots_adjust(left=0.2, right=0.98, top=top(fig, 1.0), bottom=0.13)
    save(fig, "fig11-debug-checks", theme)

if __name__ == "__main__":
    for theme in THEMES:
        chart_counters(theme)
        chart_waits(theme)
        chart_patterns(theme)
        chart_debug(theme)
    print(f"parsed {len(perf)} perf rows, {len(matmul)} matmul rows, {len(debug)} debug rows")
    print("wrote fig8-fig11 (light + dark) to", here / "charts", "and table to", here / "all_data.csv")
