"""Static figures for Part 2, from ../data/*.csv (no other inputs).

    python3 make_figures.py      # needs matplotlib; writes display_on_vs_off.png, mtp_depth_display_off.png
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OFF, ON = "#2a78d6", "#eb6834"  # categorical slots 1-2, same as Part 1
SHORT = {"TensorFold 0.6.0 / NVIDIA NVFP4 / DFlash2": "TensorFold NVFP4 DFlash2",
         "vLLM 0.27.1 / NVIDIA NVFP4 / MTP3": "vLLM NVFP4 MTP3",
         "vLLM 0.27.1 / NVIDIA NVFP4 / MTP4": "vLLM NVFP4 MTP4",
         "vLLM 0.27.1 / NVIDIA NVFP4 / MTP5": "vLLM NVFP4 MTP5"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def display_on_vs_off():
    # Only configurations whose earlier run had the display on (the earlier MLX 4-bit run was mostly display-off).
    rows = [r for r in csv.DictReader(open(DATA / "display_on_vs_off.csv")) if r["earlier_run_display"] == "on"]
    rows.sort(key=lambda r: float(r["median_output_tok_s_display_off"]))
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=200)
    for i, r in enumerate(rows):
        on, off = float(r["median_output_tok_s_display_on"]), float(r["median_output_tok_s_display_off"])
        ax.plot([on, off], [i, i], color=GRID, lw=2.5, zorder=1, solid_capstyle="round")
        ax.scatter([on], [i], s=70, color=ON, marker="s", edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.scatter([off], [i], s=80, color=OFF, marker="o", edgecolor=SURFACE, linewidth=2, zorder=3)
        gain = (float(r["output_speed_ratio_off_vs_on_paired"]) - 1) * 100
        ax.annotate(f"+{gain:.0f}%", (off, i), xytext=(10, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), [SHORT[r["config"]] for r in rows], fontsize=9)
    ax.set_xlim(0, 68)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Median output speed (tokens/s); % = paired per-prompt gain")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Line2D([], [], color=ON, marker="s", ls="", markersize=7, label="display on (earlier runs)"),
               plt.Line2D([], [], color=OFF, marker="o", ls="", markersize=7, label="display off (this run)")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK2)
    ax.set_title("Same outputs token for token, only the display changed", loc="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "display_on_vs_off.png")


def mtp_depth():
    rows = {r["config"]: r for r in csv.DictReader(open(DATA / "summary_display_off.csv"))}
    depths = [3, 4, 5, 6, 7]
    get = lambda k, col: float(rows[f"vLLM 0.27.1 / NVIDIA NVFP4 / MTP{k}"][col])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8), dpi=200)

    tps = [get(k, "tokens_per_step") for k in depths]
    a1.plot(depths, tps, color=INK2, lw=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
    for k, v in zip(depths, tps):
        a1.annotate(f"{v:.2f}", (k, v), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    a1.set_ylim(2.8, 4.4)
    a1.set_title("Tokens accepted per verification step", loc="left", fontsize=10, color=INK)

    speed = [get(k, "output_speed_ratio_vs_vllm_mtp3_display_off") for k in depths]
    total = [get(k, "total_time_speedup_vs_vllm_mtp3_display_off") for k in depths]
    a2.axhline(1.0, color=GRID, lw=1.5, ls=(0, (4, 3)), zorder=1)
    a2.plot(depths, total, color=OFF, lw=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    a2.plot(depths, speed, color=ON, lw=2, marker="s", markersize=6, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    a2.annotate("total time to answer", (7, total[-1]), xytext=(8, -2), textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    a2.annotate("output speed", (7, speed[-1]), xytext=(8, 4), textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    a2.annotate(f"MTP5: {total[2]:.2f}x", (5, total[2]), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    a2.set_xlim(2.7, 8.6)
    a2.set_ylim(0.95, 1.22)
    a2.set_title("Speed-up vs MTP3 (paired per prompt)", loc="left", fontsize=10, color=INK)

    for ax in (a1, a2):
        ax.set_xticks(depths, [f"MTP{k}" for k in depths])
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("vLLM + NVIDIA NVFP4, display off, same day: the speed-up peaks at MTP5", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "mtp_depth_display_off.png")


if __name__ == "__main__":
    display_on_vs_off()
    mtp_depth()
    print("wrote", HERE / "display_on_vs_off.png", HERE / "mtp_depth_display_off.png")
