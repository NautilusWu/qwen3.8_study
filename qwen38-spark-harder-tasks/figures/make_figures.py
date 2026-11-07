"""Draw the three figures from data/:
  by_group.png        h250 and ArXivMath, per group: correct answers and minutes per correct answer,
                      Flash Next deep thinking + 32K budget against the 27B's vote over 4 low-thinking branches
  output_length.png   Flash Next deep thinking + 32K budget: how long each reply was, on h250 and on ArXivMath
  old72.png           step 1 (72 tasks of Part 4): correct answers against minutes per correct answer, all configurations
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'data'
COLOR = {'flashnext': '#7a3fb0', 'qwen27b': '#2f6fb3'}          # the same model colors as Parts 6-7
MARK = {'flashnext': 'D', 'qwen27b': 'o'}
GREY = '#9aa0a6'
FB, CL = 'flashnext_deep_budget', 'qwen27b_vote4_low_CL4'
LABEL = {FB: 'Flash Next, deep thinking + 32K budget', CL: '27B, vote over 4 low-thinking branches'}
summary = {(r['set'], r['group'], r['config']): r for r in csv.DictReader(open(DATA / 'summary.csv'))}

# --- 1. per group ---------------------------------------------------------------------------------------------------
GROUPS = [('h250', 'all', 'h250, all 250'), ('h250', 'dev', 'dev 63'), ('h250', 'test', 'test 187'),
          ('h250', 'python', 'Python 140'), ('h250', 'cpp', 'C++ 60'), ('h250', 'process', 'process simulation 50'),
          ('h250', 'A', 'tier A 78'), ('h250', 'B', 'tier B 100'), ('h250', 'C', 'tier C 72'),
          ('arxivmath', 'all', 'ArXivMath, all 110'), ('arxivmath', 'after_release', 'paper after release 58'),
          ('arxivmath', 'before_release', 'other 52')]
ys = [i + (1 if s == 'arxivmath' else 0) for i, (s, _, _) in enumerate(GROUPS)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 6.2), sharey=True, gridspec_kw={'width_ratios': [1.15, 1]})
for (s, g, _), y in zip(GROUPS, ys):
    pts = {}
    for cfg in (FB, CL):
        r = summary[s, g, cfg]
        pct = 100 * int(r['correct']) / int(r['tasks'])
        mpc = float(r['min_per_correct']) if r['min_per_correct'] else None
        pts[cfg] = (pct, mpc, r)
    a1.plot([pts[FB][0], pts[CL][0]], [y, y], color=GREY, lw=1.2, zorder=1)
    if pts[FB][1] and pts[CL][1]:
        a2.plot([pts[FB][1], pts[CL][1]], [y, y], color=GREY, lw=1.2, zorder=1)
    same = pts[FB][2]['correct'] == pts[CL][2]['correct']
    close = pts[FB][1] and pts[CL][1] and max(pts[FB][1], pts[CL][1]) / min(pts[FB][1], pts[CL][1]) < 1.25
    for cfg, model in ((FB, 'flashnext'), (CL, 'qwen27b')):
        pct, mpc, r = pts[cfg]
        right = pct >= pts[FB if cfg == CL else CL][0]
        a1.scatter(pct, y, s=60, marker=MARK[model], color=COLOR[model], zorder=3 if cfg == FB else 4,
                   label=LABEL[cfg] if y == 0 else None)
        if same and cfg == CL:
            pass
        else:
            a1.annotate(f"{r['correct']}/{r['tasks']}" + (' (both)' if same else ''), (pct, y), textcoords='offset points',
                        xytext=(7 if right else -7, -3), ha='left' if right else 'right', fontsize=7.5)
        if mpc:
            a2.scatter(mpc, y, s=60, marker=MARK[model], color=COLOR[model], zorder=3)
            dy = (8 if cfg == FB else -14) if close else 6
            a2.annotate(f'{mpc:g}', (mpc, y), textcoords='offset points', xytext=(0, dy), ha='center', fontsize=7.5)
        else:
            a2.annotate('27B: 0 correct', (3, y), fontsize=7.5, va='center', color=COLOR[model])
a1.set_yticks(ys)
a1.set_yticklabels([lab for _, _, lab in GROUPS], fontsize=8.5)
a1.invert_yaxis()
a1.set_xlim(-16, 112)
a1.set_xlabel('correct answers (%)')
a2.set_xscale('log')
a2.set_xticks([0.5, 1, 2, 5, 10, 30, 100, 300, 1000])
a2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
a2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a2.set_xlim(0.4, 1500)
a2.set_xlabel('minutes per correct answer (log scale; left is faster)')
for ax in (a1, a2):
    ax.grid(alpha=0.3, axis='x', which='major')
    ax.axhline(9, color='#cccccc', lw=0.8)
fig.suptitle('One DGX Spark: harder tasks, each model in its best configuration from step 1', fontsize=10)
fig.legend(loc='lower center', ncol=2, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.05, 1, 0.97))
fig.savefig(HERE / 'by_group.png', dpi=150)

# --- 2. Flash Next reply lengths ------------------------------------------------------------------------------------
BIN = 2048
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, (folder, title) in zip(axes, (('per_task_h250', 'h250 (250 private tasks)'), ('per_task_arxivmath', 'ArXivMath (110 problems)'))):
    per = list(csv.DictReader(open(DATA / folder / f'{FB}.csv')))
    count = {True: defaultdict(int), False: defaultdict(int)}
    for r in per:
        count[r['passed'] == '1'][min(int(r['output_tokens']) // BIN, 65536 // BIN - 1)] += 1
    xs = range(65536 // BIN)
    ok = [count[True][i] for i in xs]
    bad = [count[False][i] for i in xs]
    left = [i * BIN / 1000 for i in xs]
    ax.bar(left, ok, width=BIN / 1000 * 0.9, align='edge', color=COLOR['flashnext'], label='correct')
    ax.bar(left, bad, width=BIN / 1000 * 0.9, align='edge', bottom=ok, color=GREY, label='wrong')
    ax.axvline(32.768, color='#333333', ls='--', lw=1)
    ax.annotate('32,768: the thinking budget', (32.768, 0.93), xycoords=('data', 'axes fraction'), xytext=(4, 0),
                textcoords='offset points', fontsize=8)
    over = sum(r['over_32768_tokens'] == '1' for r in per)
    ax.set_title(f'{title}: {over} of {len(per)} replies over 32,768 tokens', fontsize=9.5)
    ax.set_yscale('symlog', linthresh=10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yticks([0, 2, 5, 10, 20, 50, 100])
    ax.set_xlim(0, 65.536)
    ax.set_xlabel('output tokens of the reply (thousands)')
    ax.grid(alpha=0.3, axis='y')
axes[0].set_ylabel('replies (symmetric log scale)')
axes[0].legend(fontsize=8, frameon=False, loc='upper left')
fig.suptitle('Flash Next, deep thinking + 32K thinking budget (greedy)', fontsize=10)
fig.tight_layout()
fig.savefig(HERE / 'output_length.png', dpi=150)

# --- 3. step 1, the 72 old tasks -------------------------------------------------------------------------------------
STYLE = {'low_D2': ('low thinking', 's', 'none'), 'vote4_low_CL4': ('vote over 4 low-thinking branches', '^', None),
         'deep_A65': ('deep thinking', 'o', 'none'), 'deep_budget': ('deep thinking + 32K budget', 'D', None)}
NAME = {'flashnext': 'Flash Next', 'qwen27b': '27B'}
fig, ax = plt.subplots(figsize=(7.5, 5.6))
for model in COLOR:
    for key, (label, marker, face) in STYLE.items():
        r = summary.get(('old72', 'all', f'{model}_{key}'))
        if not r:
            continue
        x, y = float(r['min_per_correct']), int(r['correct'])
        ax.scatter(x, y, s=85, marker=marker, zorder=3, linewidths=1.4, edgecolors=COLOR[model],
                   facecolors=COLOR[model] if face is None else 'none', label=f'{NAME[model]}, {label}')
        ax.annotate(f'{y}/72', (x, y), textcoords='offset points', xytext=(7, 4), fontsize=8)
ax.set_xscale('log')
ax.set_xticks([2, 3, 4, 5, 7, 10, 15])
ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel('minutes per correct answer (log scale; left is faster)')
ax.set_ylabel('correct answers of 72')
ax.set_title("Step 1: Part 4's 72 hard and confirmation tasks", fontsize=10)
ax.grid(alpha=0.3, which='both')
fig.legend(*ax.get_legend_handles_labels(), loc='lower center', ncol=2, fontsize=7.5, frameon=False)
fig.tight_layout(rect=(0, 0.16, 1, 1))
fig.savefig(HERE / 'old72.png', dpi=150)
print('wrote by_group.png, output_length.png, old72.png')
