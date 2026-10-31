"""Draw figures/accuracy_vs_time.png from data/summary.csv: correct answers against minutes per correct answer."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE.parent / 'data' / 'summary.csv')))
LABEL = {'deep_A65': 'deep thinking', 'low_D2': 'low thinking', 'plan_then_low_H': 'deep plan + low execution', 'vote4_low_CL4': '4 low branches + vote'}
COLOR = {'original': '#2f6fb3', 'finetune': '#d1702a'}
MARK = {'deep_A65': 'o', 'low_D2': 's', 'plan_then_low_H': 'D', 'vote4_low_CL4': '^'}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, ts, title in ((axes[0], '72', '72 tasks (dev + confirm of Part 4)'), (axes[1], '400', '400 new tasks (note the narrower y-range)')):
    for r in rows:
        if r['task_set'] != ts:
            continue
        model, method = r['config'].split('_', 1)
        x, y = float(r['min_per_correct']), 100 * int(r['correct']) / int(r['tasks'])
        ax.scatter(x, y, s=70, color=COLOR[model], marker=MARK[method], zorder=3,
                   label=f"{'original' if model == 'original' else 'fine-tune'}, {LABEL[method]}")
        ax.annotate(f"{r['correct']}", (x, y), textcoords='offset points', xytext=(6, 4), fontsize=8)
    ax.set_xscale('log')
    ticks = [t for t in (0.5, 0.7, 1, 1.5, 2, 3, 4, 6, 8, 10) if 0.6 * min(float(r['min_per_correct']) for r in rows if r['task_set'] == ts) <= t <= 1.4 * max(float(r['min_per_correct']) for r in rows if r['task_set'] == ts)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t:g}' for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('minutes per correct answer (log scale; left is faster)')
    ax.set_ylabel('correct answers (%)')
    ax.set_title(title)
    ax.grid(alpha=0.3, which='both')
handles, labels = axes[1].get_legend_handles_labels()
h0, l0 = axes[0].get_legend_handles_labels()
seen = {}
for h, l in list(zip(h0, l0)) + list(zip(handles, labels)):
    seen.setdefault(l, h)
fig.legend(list(seen.values()), list(seen.keys()), loc='lower center', ncol=4, fontsize=8, frameon=False)
fig.suptitle('Qwen3.8-27B original NVFP4 vs a community fine-tune (NVFP4 lm_head), one DGX Spark', fontsize=11)
fig.tight_layout(rect=(0, 0.1, 1, 0.95))
fig.savefig(HERE / 'accuracy_vs_time.png', dpi=150)
print('wrote figures/accuracy_vs_time.png')
