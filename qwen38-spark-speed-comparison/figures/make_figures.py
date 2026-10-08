"""Static figures for the write-up, from ../data/*.csv (no other inputs).

    python3 make_figures.py      # needs matplotlib; writes speed_vs_correctness.png, pass_matrix.png
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ENGINE = {  # categorical slots 1-3 (validated all-pairs, light), plus a marker shape per engine
    "vLLM 0.27.1": ("#2a78d6", "o"), "vLLM 0.29.0": ("#eb6834", "s"), "TensorFold 0.6.0": ("#1baf7a", "^")}
SHORT = {"vLLM 0.27.1 / NVIDIA NVFP4 / no speculation": "NVFP4, no spec", "vLLM 0.27.1 / NVIDIA NVFP4 / MTP3": "NVFP4 MTP3",
         "vLLM 0.27.1 / NVIDIA NVFP4 / MTP4": "NVFP4 MTP4", "vLLM 0.27.1 / NVIDIA NVFP4 / MTP5": "NVFP4 MTP5",
         "vLLM 0.29.0 / Qwen FP8 / MTP3": "FP8 MTP3", "vLLM 0.29.0 / Qwen FP8 / DFlash2 k=7": "FP8 DFlash2",
         "TensorFold 0.6.0 / MLX 4-bit / DFlash2": "TF MLX 4-bit DFlash2", "TensorFold 0.6.0 / NVIDIA NVFP4 / DFlash2": "TF NVFP4 DFlash2"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def speed_vs_correctness():
    rows = list(csv.DictReader(open(DATA / "summary_table.csv")))
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
    ax.axhline(18, color=GRID, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.text(9.9, 17.85, "vLLM NVFP4 MTP3 baseline: 18/24", ha="right", va="top", color=INK2, fontsize=8.5)
    offsets = {"NVFP4 MTP3": (8, -12), "FP8 DFlash2": (8, -12), "NVFP4 MTP4": (8, -4), "NVFP4 MTP5": (8, 4),
               "TF MLX 4-bit DFlash2": (8, -12), "TF NVFP4 DFlash2": (8, 10), "FP8 MTP3": (8, -4), "NVFP4, no spec": (-8, 10)}
    for r in rows:
        engine = r["config"].split(" / ")[0]
        color, marker = ENGINE[engine]
        x, y, lab = float(r["total_hours_24_prompts"]), int(r["passed"]), SHORT[r["config"]]
        ax.scatter([x], [y], s=70, color=color, marker=marker, edgecolor=SURFACE, linewidth=2, zorder=3)
        dx, dy = offsets[lab]
        ax.annotate(f"{lab}  ({float(r['median_output_tok_s']):.0f} tok/s)", (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="left" if dx > 0 else "right", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, 10.2)
    ax.set_ylim(12.5, 20.5)
    ax.set_xlabel("Total wall time for the 24 prompts (hours)  ← faster")
    ax.set_ylabel("Prompts passed (of 24)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Line2D([], [], color=c, marker=m, ls="", markersize=7, label=e) for e, (c, m) in ENGINE.items()]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK2)
    ax.set_title("Qwen3.8-27B on one DGX Spark, single stream, thinking on (one launch per config)",
                 loc="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "speed_vs_correctness.png")


def pass_matrix():
    rows = list(csv.DictReader(open(DATA / "pass_matrix.csv")))
    configs = [c for c in rows[0] if c not in ("prompt_id", "category", "passes_of_8")]
    rows.sort(key=lambda r: (-int(r["passes_of_8"]), r["category"], r["prompt_id"]))
    fig, ax = plt.subplots(figsize=(7.6, 7.4), dpi=200)
    PASS, FAIL = "#2a78d6", "#ebeae6"
    for i, r in enumerate(rows):
        for j, c in enumerate(configs):
            ok = r[c] == "1"
            ax.add_patch(plt.Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, color=PASS if ok else FAIL, lw=0))
            if ok:
                ax.text(j + 0.5, i + 0.5, "✓", ha="center", va="center", color=SURFACE, fontsize=8)
        ax.text(len(configs) + 0.15, i + 0.5, f"{r['passes_of_8']}/8", va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, len(configs) + 0.8)
    ax.set_ylim(len(rows), 0)
    ax.set_yticks([i + 0.5 for i in range(len(rows))], [f"{r['prompt_id']}" for r in rows], fontsize=8)
    ax.set_xticks([j + 0.5 for j in range(len(configs))], [SHORT[c] for c in configs], rotation=35, ha="right", fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    groups, start = [], 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or (rows[i]["passes_of_8"] == "8") != (rows[i - 1]["passes_of_8"] == "8") or \
                (rows[i]["passes_of_8"] == "0") != (rows[i - 1]["passes_of_8"] == "0"):
            groups.append((start, i)); start = i
    for a, b in groups[:-1]:
        ax.axhline(b, color=INK2, lw=0.8)
    ax.set_title("Pass/fail per prompt: 9 always pass, 4 never finish within 24,576 tokens,\n"
                 "11 flip between configurations (blue ✓ = passed)", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "pass_matrix.png")


if __name__ == "__main__":
    speed_vs_correctness()
    pass_matrix()
    print("wrote", HERE / "speed_vs_correctness.png", HERE / "pass_matrix.png")
