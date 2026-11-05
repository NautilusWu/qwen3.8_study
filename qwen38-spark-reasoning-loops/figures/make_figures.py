"""Draw the three figures from data/:
  accuracy_vs_time.png  400 tasks: correct answers against minutes per correct answer, both models; deep thinking
                        greedy, greedy + 32K thinking budget, sampling + budget, with low thinking and the vote of
                        Parts 5-6 for reference
  greedy_reply_length.png  400 tasks, deep thinking, greedy: how long each reply was, correct or not
  stage1.png            stage 1 (79 tasks, Flash Next deep thinking): correct on the 29 looping tasks, and minutes per
                        correct answer, per arm
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
COLOR = {'flashnext': '#7a3fb0', 'qwen27b': '#2f6fb3'}          # the same model colors as Part 6
NAME = {'flashnext': 'Flash Next', 'qwen27b': '27B'}
GREY = '#9aa0a6'

# --- 1. accuracy vs time ----------------------------------------------------------------------------------------
rows = {r['config']: r for r in csv.DictReader(open(DATA / 'summary.csv'))}
STYLE = {'deep_greedy': ('deep thinking, greedy', 'o', 'none'), 'deep_greedy_budget': ('deep, greedy + 32K budget', 'D', None),
         'deep_sampling_budget': ('deep, sampling + 32K budget', 'P', None), 'low': ('low thinking (Parts 5-6)', 's', 'none'),
         'vote4_low': ('4 low branches + vote (Parts 5-6)', '^', 'none')}
fig, ax = plt.subplots(figsize=(7.5, 5.8))
pos = {}
for model in COLOR:
    for key, (label, marker, face) in STYLE.items():
        r = rows.get(f'{model}_{key}')
        if not r:
            continue
        x, y = float(r['min_per_correct']), 100 * int(r['correct']) / int(r['tasks'])
        pos[model, key] = (x, y)
        ref = key in ('low', 'vote4_low')
        ax.scatter(x, y, s=55 if ref else 85, marker=marker, zorder=3, linewidths=1.4, alpha=0.55 if ref else 1,
                   edgecolors=COLOR[model], facecolors=COLOR[model] if face is None else 'none',
                   label=f'{NAME[model]}, {label}')
        ax.annotate(r['correct'], (x, y), textcoords='offset points', xytext=(7, 4), fontsize=8)
    ax.annotate('', pos[model, 'deep_greedy_budget'], pos[model, 'deep_greedy'],
                arrowprops={'arrowstyle': '->', 'color': COLOR[model], 'lw': 1.2, 'shrinkA': 6, 'shrinkB': 6})
ax.set_xscale('log')
ax.set_xticks([0.7, 1, 1.5, 2, 3, 4, 5])
ax.set_xticklabels(['0.7', '1', '1.5', '2', '3', '4', '5'])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel('minutes per correct answer (log scale; left is faster)')
ax.set_ylabel('correct answers (%)')
ax.set_title('400 private tasks, one DGX Spark: a 32K thinking budget on deep thinking (arrows)', fontsize=10)
ax.grid(alpha=0.3, which='both')
fig.legend(*ax.get_legend_handles_labels(), loc='lower center', ncol=2, fontsize=7.5, frameon=False)
fig.tight_layout(rect=(0, 0.2, 1, 1))
fig.savefig(HERE / 'accuracy_vs_time.png', dpi=150)

# --- 2. how long the greedy deep-thinking replies were ------------------------------------------------------------
BIN = 4096
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, model in zip(axes, COLOR):
    per = list(csv.DictReader(open(DATA / 'per_task_400' / f'{model}_deep_greedy.csv')))
    count = {True: defaultdict(int), False: defaultdict(int)}
    for r in per:
        count[r['passed'] == '1'][min(int(r['output_tokens']) // BIN, 65536 // BIN - 1)] += 1
    xs = range(65536 // BIN)
    ok = [count[True][i] for i in xs]
    bad = [count[False][i] for i in xs]
    left = [i * BIN / 1000 for i in xs]
    ax.bar(left, ok, width=BIN / 1000 * 0.92, align='edge', color=COLOR[model], label='correct')
    ax.bar(left, bad, width=BIN / 1000 * 0.92, align='edge', bottom=ok, color=GREY, label='wrong or no answer')
    ax.axvline(32.768, color='#333333', ls='--', lw=1)
    ax.annotate('32,768: the thinking budget', (32.768, 0.9), xycoords=('data', 'axes fraction'), xytext=(4, 0),
                textcoords='offset points', fontsize=8)
    capped = sum(r['hit_output_limit'] == '1' for r in per)
    ax.annotate(f'{capped} hit the\n65,536 limit', (63.5, (ok[-1] + bad[-1])), textcoords='offset points',
                xytext=(-4, 6), fontsize=8, ha='right')
    ax.set_yscale('symlog', linthresh=10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yticks([0, 2, 5, 10, 20, 50, 100])
    ax.set_xlim(0, 65.536)
    ax.set_xlabel('output tokens of the greedy reply (thousands; thinking + answer)')
    n = sum(r['passed'] == '1' for r in per)
    over = sum(r['passed'] == '1' and int(r['output_tokens']) > 32768 for r in per)
    ax.set_title(f'{NAME[model]}: {n} correct, {over} of them longer than 32,768 tokens', fontsize=10)
    ax.grid(alpha=0.3, axis='y')
axes[0].set_ylabel('tasks (symmetric log scale)')
axes[0].legend(fontsize=8, loc='upper left', bbox_to_anchor=(0.53, 0.5))
fig.suptitle('400 tasks, deep thinking, greedy (Parts 5 and 6): replies in bins of 4,096 tokens', fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(HERE / 'greedy_reply_length.png', dpi=150)

# --- 3. stage 1 ---------------------------------------------------------------------------------------------------
pilot = list(csv.DictReader(open(DATA / 'stage1_pilot79.csv')))
ARMS = [('G', 'G  greedy'), ('S', 'S  sampling'), ('B', 'B  greedy + 32K budget'), ('D', 'D  detect loop, then answer')]
hard = {a: sum(r['passed'] == '1' for r in pilot if r['arm'] == a and r['group'] == 'hard') for a, _ in ARMS}
mpc = {}
for a, _ in ARMS:
    rs = [r for r in pilot if r['arm'] == a]
    mpc[a] = sum(min(float(r['wall_s']), 2400) for r in rs) / 60 / sum(r['passed'] == '1' for r in rs)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
ys = list(range(len(ARMS)))[::-1]
a1.barh(ys, [hard[a] for a, _ in ARMS], color=COLOR['flashnext'], height=0.6)
for y, (a, _) in zip(ys, ARMS):
    a1.annotate(str(hard[a]), (hard[a], y), textcoords='offset points', xytext=(4, -3), fontsize=9)
a1.set_xlim(0, 29)
a1.set_yticks(ys)
a1.set_yticklabels([l for _, l in ARMS])
a1.set_xlabel('correct, of the 29 tasks where greedy hit the limit')
a2.barh(ys, [mpc[a] for a, _ in ARMS], color=COLOR['flashnext'], height=0.6)
for y, (a, _) in zip(ys, ARMS):
    a2.annotate(f'{mpc[a]:.2f}', (mpc[a], y), textcoords='offset points', xytext=(4, -3), fontsize=9)
a2.set_xlim(0, 15)
a2.set_xlabel('minutes per correct answer, all 79 tasks')
for ax in (a1, a2):
    ax.grid(alpha=0.3, axis='x')
fig.suptitle('Stage 1, Flash Next deep thinking, 79 tasks (29 that looped + 50 controls; every arm 49 of 50 controls)',
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(HERE / 'stage1.png', dpi=150)
print('wrote figures/accuracy_vs_time.png, figures/greedy_reply_length.png, figures/stage1.png')
