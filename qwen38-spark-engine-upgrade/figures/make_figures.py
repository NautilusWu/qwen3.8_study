"""Draw the two figures from data/ (needs matplotlib):
  per_task_time.png  minutes per task on 0.6.0 against 1.2.5, CL4, all 400 tasks (log scales)
  speed.png          speed ratio 1.2.5 / 0.6.0: decoding inside the run, the microbench, and prompt processing
"""
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'data'
BLUE = '#2f6fb3'                 # the 27B's color in Parts 6-9
INK, MUTED, GREY = '#202124', '#5f6368', '#9aa0a6'
LIMIT = 40


def rows(path):
    return list(csv.DictReader(open(path)))


def minutes(r):
    return min(float(r['wall_s']) / 60, LIMIT)


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.3)


# --- 1. per task -----------------------------------------------------------------------------------------------------
new = {r['task_id']: r for r in rows(DATA / 'per_task_400' / 'qwen27b_vote4_low_CL4_tf125.csv')}
old = {r['task_id']: r for r in rows(DATA / 'per_task_400' / 'qwen27b_vote4_low_CL4_tf060.csv')}
ids = sorted(new)
x = [minutes(old[t]) for t in ids]
y = [minutes(new[t]) for t in ids]
faster = sum(b < a for a, b in zip(x, y))
tail = sorted(ids, key=lambda t: minutes(old[t]) - minutes(new[t]))[:2]
fig, ax = plt.subplots(figsize=(6.4, 6))
ax.scatter(x, y, s=14, color=BLUE, alpha=0.55, linewidths=0, zorder=3)
lo, hi = 0.08, 40
ax.plot([lo, hi], [lo, hi], color=GREY, lw=1, ls='--', zorder=2)
for t in tail:
    a, b = minutes(old[t]), minutes(new[t])
    ax.scatter([a], [b], s=60, facecolors='none', edgecolors=INK, linewidths=1.4, zorder=4)
    ax.annotate(f'{t}: {b:.1f} vs {a:.1f} min', (a, b), textcoords='offset points', xytext=(-10, 0), ha='right',
                fontsize=8, color=INK, va='center')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ticks = [0.1, 0.3, 1, 3, 10, 30]
ax.set_xticks(ticks, [str(v) for v in ticks])
ax.set_yticks(ticks, [str(v) for v in ticks])
ax.set_xlabel('minutes per task, TensorFold 0.6.0')
ax.set_ylabel('minutes per task, TensorFold 1.2.5')
ax.text(0.97, 0.04, f'below the line, 1.2.5 was faster: {faster} of {len(ids)} tasks\n'
        f'median per-task time ratio 1.2.5 / 0.6.0: {statistics.median(b / a for a, b in zip(x, y)):.3f}',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color=MUTED)
ax.set_title('Each of the 400 tasks, 4 low-thinking branches + vote (CL4), one DGX Spark', fontsize=9.5)
style(ax)
fig.tight_layout()
fig.savefig(HERE / 'per_task_time.png', dpi=150)

# --- 2. speed ratios ---------------------------------------------------------------------------------------------------
br = rows(DATA / 'branches.csv')
mb = rows(DATA / 'microbench.csv')


def run_tok_s(e):
    return statistics.median(float(b['tok_s_from_wall']) for b in br if b['engine'] == e and b['tok_s_from_wall'])


def bench(e, item, key):
    vals = [float(v) for r in mb if r['engine'] == e and r['item'] == item
            for v in (r[key].split() if key == 'per_stream_tok_s' else [r[key]])]
    return statistics.median(vals)


ITEMS = [('Decoding inside the 400-task run (4 streams, finished branches)', run_tok_s('tf125') / run_tok_s('tf060')),
         ('Microbench: 4 streams, per stream', bench('tf125', 'four_streams', 'per_stream_tok_s') / bench('tf060', 'four_streams', 'per_stream_tok_s')),
         ('Microbench: 4 streams, total', bench('tf125', 'four_streams', 'aggregate_tok_s') / bench('tf060', 'four_streams', 'aggregate_tok_s')),
         ('Microbench: 1 stream, long thinking answer', bench('tf125', 'long_thinking', 'tok_s') / bench('tf060', 'long_thinking', 'tok_s')),
         ('Microbench: 1 stream, short chat', bench('tf125', 'chat', 'tok_s') / bench('tf060', 'chat', 'tok_s')),
         ('Microbench: 1 stream, short code', bench('tf125', 'code', 'tok_s') / bench('tf060', 'code', 'tok_s')),
         ('Prompt processing, about 2K tokens', bench('tf125', 'cold_prompt_1500w', 'prefill_tok_s') / bench('tf060', 'cold_prompt_1500w', 'prefill_tok_s')),
         ('Prompt processing, about 8.6K tokens', bench('tf125', 'cold_prompt_6000w', 'prefill_tok_s') / bench('tf060', 'cold_prompt_6000w', 'prefill_tok_s')),
         ('Prompt processing, about 36K tokens', bench('tf125', 'cold_prompt_24000w', 'prefill_tok_s') / bench('tf060', 'cold_prompt_24000w', 'prefill_tok_s'))]
fig, ax = plt.subplots(figsize=(9, 4.6))
pos = list(range(len(ITEMS)))[::-1]
ax.barh(pos, [v - 1 for _, v in ITEMS], left=1, height=0.55, color=BLUE, zorder=3)
for p, (_, v) in zip(pos, ITEMS):
    ax.text(v + (0.006 if v >= 1 else -0.006), p, f'{v:.2f}', va='center', ha='left' if v >= 1 else 'right', fontsize=8, color=INK)
ax.axvline(1, color=GREY, lw=1, zorder=2)
ax.set_yticks(pos, [n for n, _ in ITEMS], fontsize=8)
ax.set_xlim(0.65, 1.15)
ax.set_xlabel('tokens per second, 1.2.5 / 0.6.0 (above 1: 1.2.5 faster)')
ax.set_title('TensorFold 1.2.5 vs 0.6.0, same checkpoint and drafter (medians)', fontsize=9.5, loc='right')
style(ax)
ax.grid(axis='y', visible=False)
fig.tight_layout()
fig.savefig(HERE / 'speed.png', dpi=150)
print('figures written')
