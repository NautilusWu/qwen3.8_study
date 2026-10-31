"""Draw the two figures from data/:
  accuracy_vs_time.png  correct answers against minutes per correct answer, 400 tasks, both models
  mtp_settings.png      time of the long example reply (ex_gcd_cpp, deep thinking, 10,308 tokens) against the MTP
                        draft settings, in the two sessions with the display on throughout
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'data'

# --- 1. accuracy vs time ----------------------------------------------------------------------------------------
rows = {r['config']: r for r in csv.DictReader(open(DATA / 'summary.csv'))}
LABEL = {'deep_A65': 'deep thinking', 'low_D2': 'low thinking', 'vote4_low_CL4': '4 low branches + vote'}
COLOR = {'flashnext': '#7a3fb0', 'qwen27b': '#2f6fb3'}
NAME = {'flashnext': 'Flash Next', 'qwen27b': '27B'}
MARK = {'deep_A65': 'o', 'low_D2': 's', 'vote4_low_CL4': '^'}
fig, ax = plt.subplots(figsize=(7.5, 5.6))
for model in COLOR:
    for method in LABEL:
        r = rows[f'{model}_{method}']
        x, y = float(r['min_per_correct']), 100 * int(r['correct']) / int(r['tasks'])
        ax.scatter(x, y, s=80, color=COLOR[model], marker=MARK[method], zorder=3,
                   label=f'{NAME[model]}, {LABEL[method]}')
        ax.annotate(r['correct'], (x, y), textcoords='offset points', xytext=(7, 4), fontsize=8)
r = rows['flashnext_low_D2_maxlimit']
x, y = float(r['min_per_correct']), 100 * int(r['correct']) / int(r['tasks'])
ax.scatter(x, y, s=80, facecolors='none', edgecolors=COLOR['flashnext'], marker='s', zorder=3,
           label='Flash Next, low thinking, 260,096-token limit (added later)')
ax.set_xscale('log')
ax.set_xticks([0.7, 1, 1.5, 2, 3, 4, 5])
ax.set_xticklabels(['0.7', '1', '1.5', '2', '3', '4', '5'])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel('minutes per correct answer (log scale; left is faster)')
ax.set_ylabel('correct answers (%)')
ax.set_title('400 private tasks, one DGX Spark: Qwen3.8-Flash-Next vs Qwen3.8-27B', fontsize=10)
ax.grid(alpha=0.3, which='both')
fig.legend(*ax.get_legend_handles_labels(), loc='lower center', ncol=2, fontsize=7.5, frameon=False)
fig.tight_layout(rect=(0, 0.17, 1, 1))
fig.savefig(HERE / 'accuracy_vs_time.png', dpi=150)

# --- 2. MTP settings --------------------------------------------------------------------------------------------
mtp = [r for r in csv.DictReader(open(DATA / 'mtp' / 'mtp_settings.csv'))
       if r['task'] == 'ex_gcd_cpp' and r['effort'] == 'deep']
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
c = [r for r in mtp if r['session'] == 'C_display_on']
fixed = sorted((int(r['max_drafts']), float(r['wall_s'])) for r in c if r['confidence'] in ('0.0', 'serial'))
a1.plot([d for d, _ in fixed], [s for _, s in fixed], 'o-', color='#2f6fb3', label='fixed number of drafts (no threshold)')
default = next(float(r['wall_s']) for r in c if r['confidence'] == '0.7')
a1.axhline(default, color='#d1702a', ls='--', label=f'default: threshold 0.7, up to 6 ({default:.0f} s)')
a1.set_xlabel('MTP drafts a round (0 = no drafts)')
a1.set_ylabel('seconds for the 10,308-token reply (lower is faster)')
a1.set_title('session C: drafts a round', fontsize=10)
a1.legend(fontsize=8)
a1.grid(alpha=0.3)
d = [r for r in mtp if r['session'] == 'D_display_on']
conf = sorted((float(r['confidence']), float(r['wall_s'])) for r in d)
zero = next(float(r['wall_s']) for r in c if r['confidence'] == '0.0' and r['max_drafts'] == '6')
a2.plot([0.0] + [x for x, _ in conf], [zero] + [s for _, s in conf], 'o-', color='#7a3fb0')
a2.annotate('0 = from session C', (0.0, zero), textcoords='offset points', xytext=(6, -12), fontsize=8)
a2.annotate(f'0.7 again in session C: {default:.0f} s', (0.7, default), textcoords='offset points', xytext=(8, -14),
            fontsize=8, color='#d1702a')
a2.set_ylim(bottom=215)
a2.scatter([0.7], [default], color='#d1702a', zorder=3)
a2.set_xlabel('confidence threshold (up to 6 drafts)')
a2.set_title('session D: threshold', fontsize=10)
a2.grid(alpha=0.3)
fig.suptitle('Flash Next MTP drafting on one DGX Spark, display on throughout; replies identical in every setting',
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(HERE / 'mtp_settings.png', dpi=150)
print('wrote figures/accuracy_vs_time.png and figures/mtp_settings.png')
