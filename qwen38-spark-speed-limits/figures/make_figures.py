"""Draw the three figures from data/ (needs matplotlib):
  round_time.png       where one decoding round of the 27B goes (PLAN_14, Nsight Systems): one stream and four streams,
                       with the time a tuned kernel needs just to read the same weights once (PLAN_22 M1)
  tree_width.png       wider verification trees (PLAN_22 M2): tokens per round, today's speed, and the ceiling if
                       verifying more rows cost nothing extra (C_hi), all relative to the default 16 rows
  tokens_vs_speed.png  for each comparison, how much the total output tokens and the token speed changed
                       (cross-plan analysis, report only)
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'data'
GREEN, ORANGE, BLUE = '#1b8a6b', '#d97a0b', '#3567d6'   # 27B keeps Parts 10-12's green; the three pass CVD checks
INK, MUTED, GREY, LIGHT = '#202124', '#5f6368', '#9aa0a6', '#dadce0'


def style(ax, grid='x'):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, alpha=0.3)
    ax.set_axisbelow(True)


prof = json.loads((DATA / 'profile_round.json').read_text())
tests = json.loads((DATA / 'tests_preregistered.json').read_text())
rep = json.loads((DATA / 'reported_only.json').read_text())

# --- 1. one round -------------------------------------------------------------------------------------------------------
parts = [('verify:mm', 'weight matmuls', GREEN), ('draft', 'drafting', ORANGE), ('commit', 'commit', BLUE),
         (('verify:gdn', 'verify:attn', 'verify', 'sample', 'other'), 'DeltaNet, attention, other', GREY), ('idle', 'GPU idle', None)]
fig, ax = plt.subplots(figsize=(8.2, 3.4))
for y, (path, label) in enumerate((('multi', 'four streams'), ('single', 'one stream'))):
    g = prof['paths'][path]['nsys']['gpu_ms_per_round']
    left = 0.0
    for key, name, color in parts:
        v = prof['paths'][path]['nsys']['idle_ms'] if key == 'idle' else sum(g.get(k, 0) for k in (key if isinstance(key, tuple) else (key,)))
        ax.barh(y, v, left=left, height=0.55, color=color or 'white', edgecolor='white' if color else GREY, linewidth=1.5,
                hatch=None if color else '///', label=name if y == 0 else None)
        if v >= 6:
            ax.text(left + v / 2, y, f'{v:.0f}', ha='center', va='center', fontsize=9, color='white' if color not in (GREY, None) else INK)
        left += v
    ax.text(left + 1.5, y, f'{left:.0f} ms', va='center', fontsize=9, color=INK)
F = tests['PLAN_22']['F_ms']
ax.axvline(F, color=INK, linestyle='--', linewidth=1)
ax.text(F - 1, 1.42, f'tuned read of the same weights: {F:.1f} ms', ha='right', va='center', fontsize=8.5, color=INK)
ax.set_yticks([0, 1])
ax.set_yticklabels(['four streams', 'one stream'])
ax.set_xlim(0, 160)
ax.set_ylim(-0.5, 1.65)
ax.set_xlabel('GPU time per decoding round (ms), TensorFold 0.6.0, Qwen3.8-27B NVFP4')
ax.legend(ncol=5, fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.28))
style(ax)
fig.tight_layout()
fig.savefig(HERE / 'round_time.png', dpi=150)
plt.close(fig)

# --- 2. tree width ------------------------------------------------------------------------------------------------------
rows = [r for r in csv.DictReader(open(DATA / 'tree_width.csv')) if r['study'] == 'PLAN_22 M2']
by = {}
for r in rows:
    by.setdefault(int(r['tree_rows']), []).append(r)
W = sorted(by)
mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
s16 = mean([float(r['tok_s']) for r in by[16]])
pw = tests['PLAN_22']['by_width']
series = [('tokens per round', [pw[str(w)]['g'] for w in W], GREEN, 'o', '-'),
          ('speed today (current kernels)', [mean([float(r['tok_s']) for r in by[w]]) / s16 for w in W], ORANGE, 's', '-'),
          ('ceiling C_hi (verify cost flat, weights read at the floor)', [pw[str(w)]['C_hi'] for w in W], BLUE, '^', '--')]
fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.axvspan(80, 136, color=LIGHT, alpha=0.5, linewidth=0)
ax.text(108, 0.585, 'report only', ha='center', fontsize=8.5, color=MUTED)
ax.axhline(1, color=GREY, linewidth=1)
for name, ys, c, m, ls in series:
    ax.plot(W, ys, color=c, marker=m, markersize=7, linewidth=2, linestyle=ls, label=name)
    ax.text(W[-1] + 3, ys[-1], f'{ys[-1]:.2f}', va='center', fontsize=9, color=INK)
cs = tests['PLAN_22']['C_star']
ax.annotate(f'C* = {cs:.3f} at 64 rows', (64, cs), xytext=(30, 1.33), fontsize=9, color=INK,
            arrowprops={'arrowstyle': '-', 'color': MUTED, 'linewidth': 1})
ax.set_xticks(W)
ax.set_xlim(8, 140)
ax.set_ylim(0.55, 1.4)
ax.set_xlabel('verification tree rows per round (default 16)')
ax.set_ylabel('relative to 16 rows')
ax.legend(fontsize=8.5, frameon=False, loc='lower left')
style(ax, 'y')
fig.tight_layout()
fig.savefig(HERE / 'tree_width.png', dpi=150)
plt.close(fig)

# --- 3. tokens vs speed ------------------------------------------------------------------------------------------------
cx = [c for c in rep['cross_plan'] if 'skipped' not in c]
short = {'PLAN_15 NVFP4 projections vs original (CL4)': 'FP4 projections, CL4 (PLAN_15)',
         'PLAN_15 NVFP4 projections vs original (D2)': 'FP4 projections, D2 (PLAN_15)',
         'PLAN_17 TensorFold 1.2.7 vs 0.6.0 (D2, one stream)': 'engine 1.2.7 vs 0.6.0, D2 (PLAN_17)',
         'PLAN_13 TensorFold 1.2.5 vs 0.6.0 (CL4), Part 10': 'engine 1.2.5 vs 0.6.0, CL4 (Part 10)',
         'Flash Next UD-IQ3_XXS, MTP on vs off (text changes, model does not), Part 11': 'MTP on vs off, same model (Part 11)',
         'Re-run: 27B D2 0.6.0, PLAN_15 vs Part 5': 're-run D2: PLAN_15 vs Part 5',
         'Re-run: 27B D2 0.6.0, PLAN_17 (one stream) vs Part 5': 're-run D2: PLAN_17 vs Part 5',
         'Re-run: 27B CL4 0.6.0, PLAN_15 vs Part 5': 're-run CL4: PLAN_15 vs Part 5',
         'Re-run: 27B CL4 0.6.0, PLAN_13 vs Part 5, Part 10': 're-run CL4: PLAN_13 vs Part 5'}
fig, ax = plt.subplots(figsize=(8.0, 4.6))
ys = list(range(len(cx)))[::-1]
for y, c in zip(ys, cx):
    t, s = c['ratio_tokens'], c['ratio_speed']
    ax.plot([min(t, s), max(t, s)], [y, y], color=LIGHT, linewidth=3, zorder=1)
    ax.scatter([t], [y], s=60, color=GREEN, zorder=3, label='total output tokens' if y == ys[0] else None)
    ax.scatter([s], [y], s=60, facecolor='white', edgecolor=ORANGE, linewidth=2, zorder=3,
               label='tokens per second' if y == ys[0] else None)
    if c['kind'] == 'repeat' and y == ys[5]:
        ax.axhline(y + 0.5, color=GREY, linewidth=0.8)
ax.axvline(1, color=GREY, linewidth=1)
ax.set_yticks(ys)
ax.set_yticklabels([short.get(c['comparison'], c['comparison']) for c in cx], fontsize=8.5)
ax.set_xscale('log')
ax.set_xticks([0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0])
ax.set_xticklabels(['0.7', '0.8', '0.9', '1', '1.25', '1.5', '2'])
ax.set_xlim(0.65, 2.1)
ax.set_xlabel('ratio, first configuration / second (log scale)')
ax.legend(fontsize=8.5, frameon=False, loc='lower right')
style(ax)
fig.tight_layout()
fig.savefig(HERE / 'tokens_vs_speed.png', dpi=150)
plt.close(fig)
print('ok')
