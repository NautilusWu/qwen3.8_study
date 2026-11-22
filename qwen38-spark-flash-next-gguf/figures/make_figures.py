"""Draw the three figures from data/ (needs matplotlib):
  accuracy_vs_time.png  correct answers against minutes per correct answer, every configuration (400 tasks, D2)
  per_task_time.png     minutes per task, TensorFold against UD-IQ3_XXS + MTP (PLAN_18c, log scales)
  mtp_speed.png         llama.cpp decoding speed with MTP: draft count (PLAN_18c P0) and each quantization
"""
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'data'
TF, LL = '#7a3fb0', '#d1702a'        # Flash Next on TensorFold (Parts 6-9); llama.cpp (passes the palette checks)
INK, MUTED, GREY = '#202124', '#5f6368', '#9aa0a6'
LIMIT = 120


def rows(path):
    return list(csv.DictReader(open(path)))


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.3)


S = {r['config']: r for r in rows(DATA / 'summary.csv')}

# --- 1. accuracy vs time -----------------------------------------------------------------------------------------------
POINTS = [  # config, label, color, MTP on, label offset (points), alignment
    ('IQ3_S', 'GSQ-RCO IQ3_S', LL, False, (0, 9), 'center'),
    ('UD-IQ4_XS', 'UD-IQ4_XS', LL, False, (0, -14), 'center'),
    ('UD-IQ3_XXS', 'UD-IQ3_XXS', LL, False, (0, -14), 'center'),
    ('UD-IQ3_XXS_MTP', 'UD-IQ3_XXS + MTP', LL, True, (0, -14), 'center'),
    ('TensorFold_NVFP4', 'TensorFold NVFP4 (this study)', TF, True, (-8, 9), 'right'),
    ('TensorFold_NVFP4_part6', 'same, Part 6 run (display on)', TF, True, (8, 9), 'left'),
]
fig, ax = plt.subplots(figsize=(7.6, 4.8))
for c, label, col, mtp, off, ha in POINTS:
    x, y = float(S[c]['min_per_correct']), int(S[c]['correct'])
    ax.scatter([x], [y], s=90, zorder=3, linewidths=2, edgecolors=col, facecolors=col if mtp else 'white',
               alpha=0.55 if c.endswith('part6') else 1)
    ax.annotate(f'{label}\n{y}/400, {x:.2f} min', (x, y), textcoords='offset points', xytext=off, ha=ha,
                va='bottom' if off[1] > 0 else 'top', fontsize=8, color=INK)
a, b = S['UD-IQ3_XXS'], S['UD-IQ3_XXS_MTP']
ax.annotate('', xy=(float(b['min_per_correct']) + 0.06, int(b['correct'])),
            xytext=(float(a['min_per_correct']) - 0.06, int(a['correct'])),
            arrowprops=dict(arrowstyle='->', color=GREY, lw=1.2), zorder=2)
ax.text((float(a['min_per_correct']) + float(b['min_per_correct'])) / 2, int(a['correct']) + 0.75, 'MTP on',
        ha='center', fontsize=8, color=MUTED)
ax.set_xlim(0.8, 3.6)
ax.set_ylim(375.5, 387)
ax.set_xlabel('minutes per correct answer (all time / correct answers; lower is better)')
ax.set_ylabel('correct answers of 400')
ax.legend(handles=[Line2D([], [], marker='o', ls='', color=LL, markersize=8, label='llama.cpp v1.0.0, GGUF'),
                   Line2D([], [], marker='o', ls='', color=TF, markersize=8, label='TensorFold 0.6.0, NVFP4'),
                   Line2D([], [], marker='o', ls='', markerfacecolor='white', markeredgecolor=MUTED, markersize=8,
                          label='hollow: MTP off')],
          loc='lower right', fontsize=8, frameon=False)
ax.set_title('Qwen3.8-Flash-Next on one DGX Spark: 400 private tasks, low thinking (D2), greedy', fontsize=9.5)
style(ax)
fig.tight_layout()
fig.savefig(HERE / 'accuracy_vs_time.png', dpi=150)

# --- 2. per task ---------------------------------------------------------------------------------------------------------
m = {r['task_id']: r for r in rows(DATA / 'per_task_400' / 'UD-IQ3_XXS_MTP.csv')}
t = {r['task_id']: r for r in rows(DATA / 'per_task_400' / 'TensorFold_NVFP4.csv')}
ids = sorted(m)
x = [min(float(t[i]['wall_s']) / 60, LIMIT) for i in ids]
y = [min(float(m[i]['wall_s']) / 60, LIMIT) for i in ids]
faster = sum(b < a for a, b in zip(x, y))
top = sorted(range(len(ids)), key=lambda k: x[k] - y[k], reverse=True)[:10]
gap = (sum(x) - sum(y))
gap10 = sum(x[k] - y[k] for k in top)
fig, ax = plt.subplots(figsize=(6.4, 6))
ax.scatter(x, y, s=14, color=LL, alpha=0.55, linewidths=0, zorder=3)
ax.scatter([x[k] for k in top], [y[k] for k in top], s=60, facecolors='none', edgecolors=INK, linewidths=1.2, zorder=4)
lo, hi = 0.05, 60
ax.plot([lo, hi], [lo, hi], color=GREY, lw=1, ls='--', zorder=2)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ticks = [0.1, 0.3, 1, 3, 10, 30]
ax.set_xticks(ticks, [str(v) for v in ticks])
ax.set_yticks(ticks, [str(v) for v in ticks])
ax.set_xlabel('minutes per task, TensorFold NVFP4 + MTP')
ax.set_ylabel('minutes per task, llama.cpp UD-IQ3_XXS + MTP')
ax.text(0.97, 0.04, f'below the line, llama.cpp was faster: {faster} of {len(ids)} tasks\n'
        f'median per-task time ratio (llama.cpp / TensorFold): {statistics.median(b / a for a, b in zip(x, y)):.3f}\n'
        f'circled: the 10 tasks with the largest time difference,\n'
        f'{gap10:.1f} of the {gap:.1f} minutes llama.cpp saved in total',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color=MUTED)
ax.set_title('Each of the 400 tasks, same session, alternating blocks (PLAN_18c)', fontsize=9.5)
style(ax)
fig.tight_layout()
fig.savefig(HERE / 'per_task_time.png', dpi=150)

# --- 3. MTP speed ------------------------------------------------------------------------------------------------------
ps = {(r['session'], r['config']): r for r in rows(DATA / 'probe' / 'probe_summary.csv')}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={'width_ratios': [1, 1.25]})
ns = [0, 2, 3, 4]
v = [float(ps[('p0', 'UD-IQ3_XXS_off' if n == 0 else f'UD-IQ3_XXS_mtp{n}')]['decode_tok_s']) for n in ns]
a1.plot(ns, v, 'o-', color=LL, lw=2, markersize=7, zorder=3)
for n, val in zip(ns, v):
    a1.annotate(f'{val:.1f}', (n, val), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=8, color=INK)
acc = [ps[('p0', f'UD-IQ3_XXS_mtp{n}')]['draft_acceptance'] for n in ns[1:]]
a1.axhline(0.97 * max(v), color=GREY, lw=1, ls=':')
a1.text(-0.1, 0.97 * max(v) - 4, '97% of the fastest', fontsize=7.5, color=MUTED)
a1.set_xticks(ns, ['off'] + [f'{n}\n{float(x):.0%}' for n, x in zip(ns[1:], acc)])
a1.set_xlabel('MTP drafts per round (--spec-draft-n-max)\nand the share of drafts accepted')
a1.set_ylabel('decoding speed, tokens/s')
a1.set_ylim(0, 70)
a1.set_title('UD-IQ3_XXS, 12 probe tasks (P0)', fontsize=9.5, loc='left')
style(a1)
ROWS = [('IQ3_S, 12 probe tasks (P3)', ps[('p3', 'IQ3_S_off')]['decode_tok_s'], ps[('p3', 'IQ3_S_mtp4')]['decode_tok_s']),
        ('UD-IQ4_XS, 12 probe tasks (P3)', ps[('p3', 'UD-IQ4_XS_off')]['decode_tok_s'], ps[('p3', 'UD-IQ4_XS_mtp4')]['decode_tok_s']),
        ('UD-IQ3_XXS, 12 probe tasks (P3)*', ps[('p3', 'UD-IQ3_XXS_off')]['decode_tok_s'], ps[('p3', 'UD-IQ3_XXS_mtp4')]['decode_tok_s']),
        ('UD-IQ3_XXS, all 400 tasks', S['UD-IQ3_XXS']['decode_tok_s'], S['UD-IQ3_XXS_MTP']['decode_tok_s'])]
pos = list(range(len(ROWS)))[::-1]
for p, (name, off, on) in zip(pos, ROWS):
    off, on = float(off), float(on)
    a2.plot([off, on], [p, p], color=GREY, lw=1.5, zorder=2)
    a2.scatter([off], [p], s=60, facecolors='white', edgecolors=LL, linewidths=2, zorder=3)
    a2.scatter([on], [p], s=60, color=LL, zorder=3)
    a2.text(on + 1.2, p, f'x{on / off:.2f}', va='center', fontsize=8, color=INK)
a2.set_yticks(pos, [r[0] for r in ROWS], fontsize=8)
a2.set_xlim(0, 70)
a2.set_xlabel('decoding speed, tokens/s (hollow: MTP off; filled: 4 drafts)')
a2.set_title('Each quantization, MTP off and on', fontsize=9.5, loc='left')
fig.text(0.42, 0.015, '* this MTP-off run was 15% slower than the same 12 tasks in P0 (identical text; cause unknown).\n'
         'The rule chose N = 4: no smaller N came within 97% of its speed.', fontsize=7.5, color=MUTED)
style(a2)
a2.grid(axis='y', visible=False)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(HERE / 'mtp_speed.png', dpi=150)
print('figures written')
