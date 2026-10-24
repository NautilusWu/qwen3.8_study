"""Static figure for the confirmation phase, from ../data/summary.csv (no other inputs).

    python3 make_figures.py      # needs matplotlib; writes confirmation.png
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# One color per method (categorical slots 1-4 of the palette used in Parts 1-3).
STYLE = {"A65": ("Deep thinking (default)", "#2a78d6"), "D2": ("Low thinking", "#1baf7a"),
         "H": ("Deep direction + low execution", "#eb6834"), "CL4": ("4 low-thinking branches + vote", "#8f5bd6")}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def confirmation():
    rows = {r["method"]: r for r in csv.DictReader(open(DATA / "summary.csv")) if r["phase"] == "confirmation"}
    order = ["CL4", "H", "D2", "A65"]
    labels = [STYLE[m][0] for m in order]
    colors = [STYLE[m][1] for m in order]
    panels = [("correct", "Correct answers\n48 new tasks", "{:.0f} / 48", "higher is better"),
              ("min_per_correct", "Minutes per correct answer\n(all time, including wrong answers)", "{:.2f} min", "lower is better")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), dpi=200, sharey=True)
    for ax, (col, title, fmt, hint) in zip(axes, panels):
        vals = [float(rows[m][col]) for m in order]
        ax.barh(range(len(order)), vals, height=0.56, color=colors, edgecolor=SURFACE, linewidth=2)
        for i, v in enumerate(vals):
            ax.annotate(fmt.format(v), (v, i), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
        ax.set_xlim(0, max(vals) * 1.28)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel(hint, fontsize=8.5)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(order)), labels, fontsize=9)
    fig.suptitle("On 48 new tasks, low thinking reaches a correct answer about 3x faster than deep thinking",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "confirmation.png")


if __name__ == "__main__":
    confirmation()
    print("wrote", HERE / "confirmation.png")
