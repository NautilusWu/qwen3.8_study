"""Static figure for the final validation, from ../data/summary_final.csv (no other inputs).

    python3 make_figures.py      # needs matplotlib; writes time_to_correct_answer.png
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# One color per configuration, same in both panels (categorical slots 1-3, as in Parts 1 and 2).
STYLE = {"vLLM 0.27.1 / NVIDIA NVFP4 / MTP5": ("vLLM NVFP4 MTP5", "#2a78d6"),
         "TensorFold 0.6.0 / NVIDIA NVFP4 / DFlash2": ("TensorFold NVFP4 DFlash2", "#1baf7a"),
         "TensorFold 0.6.0 / MLX 4-bit / DFlash2": ("TensorFold MLX 4-bit DFlash2", "#eb6834")}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def time_to_correct_answer():
    rows = {r["config"]: r for r in csv.DictReader(open(DATA / "summary_final.csv"))}
    order = list(STYLE)[::-1]  # top to bottom: vLLM, TensorFold NVFP4, TensorFold MLX 4-bit
    labels = [STYLE[c][0] for c in order]
    colors = [STYLE[c][1] for c in order]
    panels = [("sealed_median_output_tok_s", "Output speed (tokens/s)\nmedian, 48 sealed prompts", "{:.1f}", "higher is better"),
              ("minutes_per_correct_answer_72", "Minutes per correct answer\nall 72 prompts", "{:.1f} min", "lower is better")]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), dpi=200, sharey=True)
    for ax, (col, title, fmt, hint) in zip(axes, panels):
        vals = [float(rows[c][col]) for c in order]
        ax.barh(range(len(order)), vals, height=0.56, color=colors, edgecolor=SURFACE, linewidth=2)
        for i, v in enumerate(vals):
            ax.annotate(fmt.format(v), (v, i), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
        ax.set_xlim(0, max(vals) * 1.25)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel(hint, fontsize=8.5)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(order)), labels, fontsize=9)
    fig.suptitle("The fastest by tokens/s is not the fastest to a correct answer", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "time_to_correct_answer.png")


if __name__ == "__main__":
    time_to_correct_answer()
    print("wrote", HERE / "time_to_correct_answer.png")
