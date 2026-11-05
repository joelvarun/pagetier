"""Draw docs/before_after.png from docs/results.json."""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
MILLION = 1e6
GIB = float(2**30)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)


def frontier_panel(ax, res):
    hues = {"x86_64": BLUE, "arm64_4k": ORANGE}
    none = res["arch"]["x86_64"]["no_huge_pages"] / MILLION
    ax.axhline(none, color=AXIS, linewidth=1.0, linestyle=(0, (1, 3)), zorder=2)
    ax.annotate("no huge pages, %.2fM translations" % none, xy=(0.017, none),
                xytext=(0, 5), textcoords="offset points", fontsize=9, color=MUTED)

    for key in ("x86_64", "arm64_4k"):
        block = res["arch"][key]
        hue = hues[key]
        env = np.array(block["envelope_local"], dtype=float)
        keep = env[:, 0] > 0
        ax.plot(env[keep, 0] / GIB, env[keep, 1] / MILLION, color=hue, linewidth=2.0,
                linestyle=(0, (5, 3)), zorder=3)
        front = block["frontier"]
        w = np.array(front["waste_bytes"], dtype=float)
        e = np.array(front["entries"], dtype=float)
        keep = w > 0
        ax.plot(w[keep] / GIB, e[keep] / MILLION, color=hue, linewidth=2.0, zorder=4)
        default = block["khugepaged_default"]
        ax.plot([default["bytes"] / GIB], [default["entries"] / MILLION], marker="o",
                markersize=9, color=SURFACE, markeredgecolor=hue, markeredgewidth=2.0,
                zorder=5, linestyle="none")

    dflt = res["arch"]["x86_64"]["khugepaged_default"]
    ax.annotate("khugepaged default, max_ptes_none=511:\n%.2f GB extra for %.2fM translations"
                % (dflt["bytes"] / GIB, dflt["entries"] / MILLION),
                xy=(dflt["bytes"] / GIB, dflt["entries"] / MILLION),
                xytext=(1.05, 2.30), fontsize=9, color=INK_2, ha="left",
                arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=1.0))

    hb = res["headline_mb"] * 2**20 / GIB
    ax.axvline(hb, color=AXIS, linewidth=1.0, linestyle=(0, (1, 3)), zorder=2)
    ax.annotate("%d MB budget" % res["headline_mb"], xy=(hb, 3.20), xytext=(4, 0),
                textcoords="offset points", fontsize=9, color=MUTED, ha="left")
    for key, label, y in (("x86_64", "x86-64", 1.97), ("arm64_4k", "ARM64", 1.03)):
        ax.annotate(label, xy=(6.9, y), fontsize=10, color=hues[key], ha="left",
                    va="center", fontweight="medium")

    ax.set_xscale("log")
    ax.set_xlim(0.015, 9.0)
    ax.set_ylim(0.86, 3.28)
    ax.set_xticks([0.0156, 0.0625, 0.25, 1.0, 4.0, 8.0])
    ax.set_xticklabels(["16 MB", "64 MB", "256 MB", "1 GB", "4 GB", "8 GB"])
    ax.set_xlabel("extra physical memory committed, log scale", fontsize=10, color=INK_2)
    ax.set_ylabel("translations covering the resident set (millions)", fontsize=10, color=INK_2)
    ax.set_title("Translations against the memory they cost", fontsize=11, color=INK,
                 loc="left", pad=10)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%.1f" % v))
    style_axes(ax)

    handles = [
        plt.Line2D([], [], color=BLUE, linewidth=2.0, label="x86-64, pagetier"),
        plt.Line2D([], [], color=BLUE, linewidth=2.0, linestyle=(0, (5, 3)),
                   label="x86-64, tuned per-mm threshold"),
        plt.Line2D([], [], color=ORANGE, linewidth=2.0, label="ARM64, pagetier"),
        plt.Line2D([], [], color=ORANGE, linewidth=2.0, linestyle=(0, (5, 3)),
                   label="ARM64, tuned per-mm threshold"),
    ]
    leg = ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=9,
                    labelcolor=INK_2, handlelength=2.4, borderaxespad=0.6,
                    facecolor=SURFACE, edgecolor="none", framealpha=1.0)
    leg.set_zorder(7)


def budget_panel(ax, res):
    hb = res["headline_mb"]
    labels, deltas = [], []
    series = {"no huge pages": [], "tuned per-mm threshold": [], "pagetier exact": []}
    for key in ("x86_64", "arm64_4k"):
        block = res["arch"][key]
        row = next(r for r in block["budget_rows"] if r["budget_mb"] == hb)
        labels.append(block["label"])
        series["no huge pages"].append(block["no_huge_pages"] / MILLION)
        series["tuned per-mm threshold"].append(row["per_mm_entries"] / MILLION)
        series["pagetier exact"].append(row["exact_entries"] / MILLION)
        deltas.append(100 * (row["per_mm_entries"] - row["exact_entries"]) / row["per_mm_entries"])

    hues = [BLUE, ORANGE, AQUA]
    x = np.arange(len(labels), dtype=float)
    width = 0.24
    gap = 0.015
    for i, (name, values) in enumerate(series.items()):
        off = (i - 1) * (width + gap)
        bars = ax.bar(x + off, values, width, label=name, color=hues[i], zorder=3,
                      edgecolor=SURFACE, linewidth=1.6)
        for j, (rect, v) in enumerate(zip(bars, values)):
            cx = rect.get_x() + rect.get_width() / 2
            ax.annotate("%.2fM" % v, xy=(cx, v), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=9, color=INK_2)
            if name == "pagetier exact":
                ax.annotate("%.1f%% below the\ntuned threshold" % deltas[j], xy=(cx, v),
                            xytext=(0, 22), textcoords="offset points", ha="center",
                            fontsize=9, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color=INK_2)
    ax.set_ylabel("translations covering the resident set (millions)", fontsize=10, color=INK_2)
    ax.set_title("At a %d MB extra-memory budget" % hb, fontsize=11, color=INK,
                 loc="left", pad=28)
    ax.set_ylim(0, 3.55)
    style_axes(ax)
    ax.grid(False, axis="x")
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.008), ncol=3, frameon=False,
              fontsize=9, labelcolor=INK_2, columnspacing=1.4, handlelength=1.1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=os.path.join("docs", "results.json"))
    ap.add_argument("--out", default=os.path.join("docs", "before_after.png"))
    args = ap.parse_args()
    with open(args.results) as fh:
        res = json.load(fh)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.28, 1.0]})
    frontier_panel(axes[0], res)
    budget_panel(axes[1], res)
    census = res["census"]
    fig.suptitle(
        "pagetier: translations needed to cover a real resident-page census",
        fontsize=13, color=INK, x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.930,
             "%d Windows processes, %d translations over %.2f GB of distinct physical pages. "
             "Lower is better on both axes."
             % (census["processes"], census["resident_pages"],
                census["distinct_bytes"] / GIB),
             fontsize=10, color=INK_2, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.895])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
