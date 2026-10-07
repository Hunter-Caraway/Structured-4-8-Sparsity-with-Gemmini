#!/usr/bin/env python3
# makes the charts for the n:m sparsity writeup
# changed: new script. reads results/cycles.csv for the benchmark numbers, the rest
# (stall counters, design history, debug scores) is copied out of the logs named next to it
#
# run with the chipyard conda env, it has matplotlib + pandas:
#   ~/chipyard/.conda-env/bin/python analysis/make_charts.py
# writes analysis/charts/<name>-light.png and <name>-dark.png

import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

here = pathlib.Path(__file__).resolve().parent
root = here.parent
out = here / "charts"
out.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# data

df = pd.read_csv(root / "results" / "cycles.csv")
# keep the latest run of each (config, test, variant)
df = df.sort_values("timestamp").groupby(["config", "test", "variant"], as_index=False).last()
df["pattern"] = df["config"].map({"GemminiNMRocketConfig": "2:4", "GemminiNM48RocketConfig": "4:8"})
perf = df[df.test == "nm_sparse_perf"].set_index(["pattern", "variant"])

def total(p, v): return int(perf.loc[(p, v), "total_cycles"])
def busy(p, v): return int(perf.loc[(p, v), "array_busy_cycles"])
def tiles(p, v): return int(perf.loc[(p, v), "tiles"])

PATTERNS = ["2:4", "4:8"]

# stall counters from the same runs (snapshot taken right at the fence)
# logs/sparse/nm24-rtl-fifo-perf2-cpuonly.log and logs/sparse/nm48-rtl-nm_sparse_perf.log
idle_gap = {  # spad_a_wait: cycles the array had nothing to stream (stale control = no work queued)
    "2:4": {"dense": 93, "dense_bias": 207, "sparse": 1984, "sparse_nat": 1640, "dense_reuse": 2194, "sparse_reuse": 1825},
    "4:8": {"dense": 82, "dense_bias": 215, "sparse": 2520, "sparse_nat": 2477, "dense_reuse": 2332, "sparse_reuse": 2088},
}

# design history
# nm_sparse_matmul sparse cycles: logs/sparse/nm24-rtl-run1.log (packing inside the timed loop, results wrong)
#                                 logs/sparse/nm24-rtl-run3-nm_sparse_matmul.log (prepacked, results right)
history_matmul = [("first RTL run\n(index packing timed,\nresults wrong)", 22674),
                  ("index packing moved\nout of the timed loop", 815)]
# nm_sparse_perf sparse (metadata sent ahead), same test binary:
#   logs/sparse/nm24-rtl-perf2.log (meta through the reservation station)
#   logs/sparse/nm24-rtl-fifo-nm_sparse_perf.log (meta through the index fifo)
history_meta = [("metadata through the\nreservation station", 2815),
                ("metadata through the\nindex FIFO", 2859),
                ("no metadata at all\n(results junk, bound)", 1375)]

# ---------------------------------------------------------------------------
# style: two themes, same layout. colors are the reference categorical palette
# (slot 1 blue, slot 2 orange; slot 7 violet + slot 3 aqua where the color means pattern)

THEMES = {
    "light": dict(surface="#fcfcfb", fg="#0b0b0b", fg2="#52514e", fg3="#7b7a75", grid="#e9e8e4",
                  dense="#2a78d6", sparse="#eb6834", p24="#4a3aa7", p48="#1baf7a", ref="#52514e"),
    "dark": dict(surface="#1a1a19", fg="#ffffff", fg2="#c3c2b7", fg3="#97968d", grid="#2e2e2c",
                 dense="#3987e5", sparse="#d95926", p24="#9085e9", p48="#199e70", ref="#c3c2b7"),
}

def style(ax, t, xgrid=True):
    ax.set_facecolor(t["surface"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["fg2"], labelsize=9.5, length=0)
    if xgrid:
        ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)

def new_fig(t, w, h, ncols=1, sharex=False):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), sharex=sharex, squeeze=False)
    fig.patch.set_facecolor(t["surface"])
    return fig, axes[0]

def title(fig, t, text, sub):
    fig.text(0.012, 0.985, text, ha="left", va="top", fontsize=13, fontweight="bold", color=t["fg"])
    fig.text(0.012, 0.915, sub, ha="left", va="top", fontsize=9.5, color=t["fg2"])

def legend(fig, t, items, y=0.83):
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) if kind == "box" else
               plt.Line2D([0], [0], color=c, lw=0, marker=kind, markersize=8, markeredgecolor=t["surface"])
               for (lbl, c, kind) in items]
    fig.legend(handles, [i[0] for i in items], loc="upper left", bbox_to_anchor=(0.008, y), ncol=len(items),
               frameon=False, fontsize=9.5, labelcolor=t["fg2"], handlelength=1.0, columnspacing=1.6)

def fmt(n):
    return f"{n:,.0f}"

def save(fig, name, theme):
    fig.savefig(out / f"{name}-{theme}.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

# ---------------------------------------------------------------------------
# charts

def chart_busy(theme):
    t = THEMES[theme]
    fig, axes = new_fig(t, 9.2, 3.6, ncols=2)
    title(fig, t, "Array-busy cycles: the sparse array does the same matmul in half the time",
          "64×128×32 matmul, cycles the execute controller spends streaming rows through the 16×16 array")
    legend(fig, t, [("Dense", t["dense"], "box"), ("Sparse", t["sparse"], "box")])
    rows = [("No weight reuse", "dense", "sparse"), ("Weight reuse", "dense_reuse", "sparse_reuse")]
    for ax, p in zip(axes, PATTERNS):
        style(ax, t)
        y = 0
        yt, yl = [], []
        for label, dv, sv in rows:
            for k, (v, c) in enumerate([(busy(p, dv), t["dense"]), (busy(p, sv), t["sparse"])]):
                yy = y - k * 0.38
                ax.barh(yy, v, height=0.32, color=c)
                ax.text(v + 30, yy, fmt(v), va="center", fontsize=9, color=t["fg2"])
            yt.append(y - 0.19)
            yl.append(f"{label}\n{busy(p, dv) / busy(p, sv):.2f}× fewer")
            y -= 1.2
        ax.set_yticks(yt, yl, fontsize=9.5, color=t["fg"])
        ax.set_xlim(0, 2150)
        ax.set_title(f"{p}", loc="left", fontsize=11, color=t["fg"], fontweight="bold")
        ax.set_xlabel("cycles", fontsize=9, color=t["fg3"])
    fig.subplots_adjust(left=0.13, right=0.98, top=0.68, bottom=0.14, wspace=0.42)
    save(fig, "fig1-array-busy", theme)

def chart_end_to_end(theme):
    t = THEMES[theme]
    fig, axes = new_fig(t, 9.2, 4.6, ncols=2)
    title(fig, t, "End-to-end cycles, and how much of that the CPU needs just to issue commands",
          "bar = total cycles from first command to fence · dot = the same loop with the accelerator removed")
    legend(fig, t, [("Dense", t["dense"], "box"), ("Sparse", t["sparse"], "box"), ("CPU-only issue time", t["fg"], "o")], y=0.85)
    rows = [("No reuse,\nmeta sent ahead", "dense", "sparse"),
            ("No reuse,\nnatural order", "dense", "sparse_nat"),
            ("Weight reuse", "dense_reuse", "sparse_reuse")]
    for ax, p in zip(axes, PATTERNS):
        style(ax, t)
        y = 0
        yt, yl = [], []
        for label, dv, sv in rows:
            for k, (v, c) in enumerate([(dv, t["dense"]), (sv, t["sparse"])]):
                yy = y - k * 0.38
                tot = total(p, v)
                cpu = total(p, "cpu_" + v)
                ax.barh(yy, tot, height=0.32, color=c)
                ax.plot([cpu], [yy], marker="o", markersize=7, color=t["fg"], markeredgecolor=t["surface"], markeredgewidth=1.5)
                ax.text(tot + 50, yy, fmt(tot), va="center", fontsize=9, color=t["fg2"])
            ratio = total(p, dv) / total(p, sv)
            yt.append(y - 0.19)
            yl.append(f"{label}\nsparse {ratio:.2f}×")
            y -= 1.25
        ax.set_yticks(yt, yl, fontsize=9.5, color=t["fg"])
        ax.set_xlim(0, 4400)
        ax.set_title(p, loc="left", fontsize=11, color=t["fg"], fontweight="bold")
        ax.set_xlabel("cycles", fontsize=9, color=t["fg3"])
    fig.subplots_adjust(left=0.14, right=0.98, top=0.72, bottom=0.11, wspace=0.45)
    save(fig, "fig2-end-to-end", theme)

def chart_speedup(theme):
    t = THEMES[theme]
    fig, axes = new_fig(t, 9.2, 3.9)
    ax = axes[0]
    style(ax, t)
    title(fig, t, "Speedup of sparse over dense (dense cycles ÷ sparse cycles)",
          "above 1 = sparse is faster · dashed lines: break-even (1×) and the 2× limit when M = 2N")
    legend(fig, t, [("2:4", t["p24"], "box"), ("4:8", t["p48"], "box")])
    cats = [("Array busy,\nno reuse", lambda p: busy(p, "dense") / busy(p, "sparse")),
            ("Array busy,\nweight reuse", lambda p: busy(p, "dense_reuse") / busy(p, "sparse_reuse")),
            ("End to end,\nno reuse", lambda p: total(p, "dense") / total(p, "sparse")),
            ("End to end,\nnatural order", lambda p: total(p, "dense") / total(p, "sparse_nat")),
            ("End to end,\nweight reuse", lambda p: total(p, "dense_reuse") / total(p, "sparse_reuse"))]
    w = 0.3
    for i, (lbl, f) in enumerate(cats):
        for k, (p, c) in enumerate([("2:4", t["p24"]), ("4:8", t["p48"])]):
            v = f(p)
            x = i + (k - 0.5) * (w + 0.04)
            ax.bar(x, v, width=w, color=c)
            # label sits on a surface-colored patch so the reference lines pass behind it
            ax.text(x, v + 0.06, f"{v:.2f}", ha="center", fontsize=8.5, color=t["fg2"], zorder=4,
                    bbox=dict(facecolor=t["surface"], edgecolor="none", pad=1.2))
    for ref, lbl in [(1.0, "break-even"), (2.0, "2× limit")]:
        ax.axhline(ref, color=t["ref"], linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
        # reference labels live in their own strip right of the last group
        ax.text(len(cats) + 0.25, ref, lbl, fontsize=8.5, color=t["fg3"], ha="right", va="center",
                bbox=dict(facecolor=t["surface"], edgecolor="none", pad=1.2), zorder=4)
    ax.set_xticks(range(len(cats)), [c[0] for c in cats], fontsize=9.5, color=t["fg"])
    ax.set_xlim(-0.6, len(cats) + 0.3)
    ax.set_ylim(0, 2.3)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_ylabel("speedup (×)", fontsize=9, color=t["fg3"])
    fig.subplots_adjust(left=0.08, right=0.98, top=0.7, bottom=0.17)
    save(fig, "fig3-speedup", theme)

def chart_per_tile(theme):
    t = THEMES[theme]
    fig, axes = new_fig(t, 9.2, 4.4, ncols=2)
    title(fig, t, "Cycles per weight tile: where each tile's time goes",
          "each sparse tile does the work of two dense tiles · 16 = rows streamed per tile, the floor for any tile")
    legend(fig, t, [("Array busy", t["dense"], "o"), ("CPU-only issue", t["sparse"], "o"), ("Total", t["fg"], "o")], y=0.85)
    variants = [("Dense", "dense"), ("Sparse, sent ahead", "sparse"), ("Sparse, natural", "sparse_nat"),
                ("Dense, reuse", "dense_reuse"), ("Sparse, reuse", "sparse_reuse")]
    for ax, p in zip(axes, PATTERNS):
        style(ax, t)
        for i, (lbl, v) in enumerate(variants):
            n = tiles(p, v)
            vals = [(busy(p, v) / n, t["dense"]), (total(p, "cpu_" + v) / n, t["sparse"]), (total(p, v) / n, t["fg"])]
            lo, hi = min(x for x, _ in vals), max(x for x, _ in vals)
            ax.plot([lo, hi], [-i, -i], color=t["grid"], linewidth=2, zorder=1)
            for x, c in vals:
                ax.plot([x], [-i], marker="o", markersize=8, color=c, markeredgecolor=t["surface"], markeredgewidth=1.5, zorder=3)
            ax.text(hi + 3, -i, f"{total(p, v) / n:.0f}", va="center", fontsize=8.5, color=t["fg2"])
        ax.axvline(16, color=t["ref"], linewidth=0.9, linestyle=(0, (4, 3)))
        ax.text(17, 0.55, "16", fontsize=8.5, color=t["fg3"])
        ax.set_yticks([-i for i in range(len(variants))], [v[0] for v in variants], fontsize=9.5, color=t["fg"])
        ax.set_xlim(0, 125)
        ax.set_ylim(-len(variants) + 0.4, 0.9)
        ax.set_title(p, loc="left", fontsize=11, color=t["fg"], fontweight="bold")
        ax.set_xlabel("cycles per tile", fontsize=9, color=t["fg3"])
    fig.subplots_adjust(left=0.16, right=0.98, top=0.72, bottom=0.11, wspace=0.5)
    save(fig, "fig4-per-tile", theme)

def chart_utilization(theme):
    t = THEMES[theme]
    fig, axes = new_fig(t, 9.2, 3.9, ncols=2)
    title(fig, t, "Array utilization: share of the run the array actually spends computing",
          "array-busy cycles ÷ total cycles · the rest is mostly the array waiting for the CPU to issue the next command")
    legend(fig, t, [("Dense", t["dense"], "box"), ("Sparse", t["sparse"], "box")])
    rows = [("No reuse", "dense", "sparse"), ("Natural order", "dense", "sparse_nat"), ("Weight reuse", "dense_reuse", "sparse_reuse")]
    for ax, p in zip(axes, PATTERNS):
        style(ax, t)
        y = 0
        yt, yl = [], []
        for label, dv, sv in rows:
            for k, (v, c) in enumerate([(dv, t["dense"]), (sv, t["sparse"])]):
                yy = y - k * 0.38
                u = 100 * busy(p, v) / total(p, v)
                ax.barh(yy, u, height=0.32, color=c)
                ax.text(u + 1.5, yy, f"{u:.0f}%", va="center", fontsize=9, color=t["fg2"])
            yt.append(y - 0.19)
            yl.append(label)
            y -= 1.2
        ax.set_yticks(yt, yl, fontsize=9.5, color=t["fg"])
        ax.set_xlim(0, 100)
        ax.set_title(p, loc="left", fontsize=11, color=t["fg"], fontweight="bold")
        ax.set_xlabel("% of total cycles", fontsize=9, color=t["fg3"])
    fig.subplots_adjust(left=0.11, right=0.98, top=0.68, bottom=0.14, wspace=0.35)
    save(fig, "fig5-utilization", theme)

def chart_metadata(theme):
    t = THEMES[theme]
    fig, axes = new_fig(t, 9.2, 3.5)
    ax = axes[0]
    style(ax, t)
    title(fig, t, "CPU cost of issuing one tile's commands (accelerator removed)",
          "a dense tile is 1 preload + 1 compute · a sparse tile adds 4 (2:4) or 8 (4:8) metadata commands")
    rows = [("Dense tile (2 commands)", total("2:4", "cpu_dense") / 64, t["dense"]),
            ("Sparse 2:4, natural order (6)", total("2:4", "cpu_sparse_nat") / 32, t["sparse"]),
            ("Sparse 2:4, sent ahead (6)", total("2:4", "cpu_sparse") / 32, t["sparse"]),
            ("Sparse 4:8, natural order (10)", total("4:8", "cpu_sparse_nat") / 32, t["sparse"]),
            ("Sparse 4:8, sent ahead (10)", total("4:8", "cpu_sparse") / 32, t["sparse"])]
    for i, (lbl, v, c) in enumerate(rows):
        ax.barh(-i, v, height=0.5, color=c)
        ax.text(v + 1, -i, f"{v:.1f}", va="center", fontsize=9, color=t["fg2"])
    # one sparse tile does the work of two dense tiles, so that's the fair comparison
    two = 2 * rows[0][1]
    ax.axvline(two, color=t["ref"], linewidth=0.9, linestyle=(0, (4, 3)))
    ax.text(two + 0.8, 0.45, f"two dense tiles = {two:.1f}\n(same amount of work)", fontsize=8.5, color=t["fg3"], va="center")
    ax.set_yticks([-i for i in range(len(rows))], [r[0] for r in rows], fontsize=9.5, color=t["fg"])
    ax.set_xlim(0, 95)
    ax.set_ylim(-len(rows) + 0.4, 0.9)
    ax.set_xlabel("CPU cycles per tile", fontsize=9, color=t["fg3"])
    fig.subplots_adjust(left=0.27, right=0.98, top=0.76, bottom=0.15)
    save(fig, "fig6-metadata-cost", theme)

def chart_history(theme):
    t = THEMES[theme]
    fig, axes = new_fig(t, 9.2, 3.6, ncols=2)
    title(fig, t, "How the sparse timing changed across design iterations (2:4)",
          "left: nm_sparse_matmul 32×64×32 · right: nm_sparse_perf 64×128×32, metadata sent ahead, same test binary")
    for ax, data, xmax in [(axes[0], history_matmul, 26000), (axes[1], history_meta, 3400)]:
        style(ax, t)
        for i, (lbl, v) in enumerate(data):
            ax.barh(-i, v, height=0.45, color=t["sparse"])
            ax.text(v + xmax * 0.015, -i, fmt(v), va="center", fontsize=9, color=t["fg2"])
        ax.set_yticks([-i for i in range(len(data))], [d[0] for d in data], fontsize=9, color=t["fg"])
        ax.set_xlim(0, xmax)
        ax.set_ylim(-2.5, 0.5)  # same row spacing in both panels
        ax.set_xlabel("sparse cycles", fontsize=9, color=t["fg3"])
    fig.subplots_adjust(left=0.17, right=0.98, top=0.78, bottom=0.15, wspace=0.75)
    save(fig, "fig7-history", theme)

if __name__ == "__main__":
    for theme in THEMES:
        chart_busy(theme)
        chart_end_to_end(theme)
        chart_speedup(theme)
        chart_per_tile(theme)
        chart_utilization(theme)
        chart_metadata(theme)
        chart_history(theme)
    print("wrote", len(list(out.glob("*.png"))), "charts to", out)
