"""Draw the four figures from data/ (needs matplotlib):
  throughput.png        correct answers per hour: one stream, four streams in blocks of 50 (measured), and four
                        streams on a server that never runs dry (estimated from the steady speed)
  tasks_running.png     how many of the four streams were busy, block after block (PLAN_20)
  per_task_latency.png  each task's own time, one stream against four streams (log scales)
  streams_27b.png       the 27B at 1 (PLAN_19), 2, 4 and 8 streams (PLAN_21b): correct answers per hour, and seconds
                        per task (median and 90th percentile)
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'data'
sys.path.insert(0, str(HERE.parent / 'code'))
import shared_time as ST  # noqa: E402

COLOR = {'27b': '#1b8a6b', 'flashnext': '#7a3fb0'}   # Flash Next keeps Parts 6-11's purple; the pair passes CVD checks
NAME = {'27b': 'Qwen3.8-27B', 'flashnext': 'Flash Next'}
INK, MUTED, GREY = '#202124', '#5f6368', '#9aa0a6'
MODELS = ('27b', 'flashnext')


def rows(path):
    return list(csv.DictReader(open(path)))


def style(ax, grid='y'):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, alpha=0.3)
    ax.set_axisbelow(True)


rep = json.loads((DATA / 'reported_only.json').read_text())
sc = rep['Two ways to use the server']

# --- 1. throughput -----------------------------------------------------------------------------------------------------
groups = [('one_stream', 'one stream\n(measured)'), ('batch_of_50', 'four streams, blocks of 50\n(measured)'),
          ('never_dry_estimate', 'four streams, never runs dry\n(estimate)')]
fig, ax = plt.subplots(figsize=(7.6, 4.4))
w = 0.36
for i, m in enumerate(MODELS):
    for j, (k, _) in enumerate(groups):
        v = sc[m]['correct_per_hour'][k]
        est = k.startswith('never')
        x = j + (i - 0.5) * (w + 0.02)
        ax.bar(x, v, w, color='white' if est else COLOR[m], edgecolor=COLOR[m], linewidth=2, hatch='///' if est else None,
               label=NAME[m] if j == 0 else None)
        ax.text(x, v + 12, f'{v:.0f}', ha='center', va='bottom', fontsize=9, color=INK)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([g for _, g in groups], fontsize=9)
ax.set_ylabel('correct answers per hour (higher is better)')
ax.set_ylim(0, 900)
style(ax)
ax.legend(frameon=False, loc='upper left')
ax.set_title('Second private set (400 tasks), low thinking, greedy, one DGX Spark', fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(HERE / 'throughput.png', dpi=150)
plt.close(fig)

# --- 2. tasks running, block after block --------------------------------------------------------------------------------
four = {m: rows(DATA / 'per_task_400' / f'{m}_4streams.csv') for m in MODELS}
blocks = rows(DATA / 'blocks.csv')
order = {m: [int(b['block']) for b in sorted((b for b in blocks if b['study'] == 'PLAN_20' and b['model'] == m),
                                             key=lambda b: int(b['position']))] for m in MODELS}
fig, axes = plt.subplots(2, 1, figsize=(9, 4.8), sharex=True)
for ax, m in zip(axes, MODELS):
    t0 = 0.0
    for g in order[m]:
        rs = [{'task_id': r['task_id'], 'start_s': float(r['start_s']), 'end_s': float(r['end_s'])}
              for r in four[m] if int(r['block']) == g]
        c = ST.cuts(rs)
        xs, ys = [], []
        for a, b in zip(c, c[1:]):
            k = sum(1 for r in rs if r['start_s'] <= a and r['end_s'] >= b)
            xs += [t0 + a / 60, t0 + b / 60]
            ys += [k, k]
        ax.fill_between(xs, ys, step=None, color=COLOR[m], alpha=0.85, linewidth=0)
        span = (c[-1] - c[0]) / 60
        ax.text(t0 + span / 2, 4.25, str(g), ha='center', fontsize=7, color=MUTED)
        t0 += span
        ax.axvline(t0, color=GREY, lw=0.6)
    ax.set_ylim(0, 4.7)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_ylabel('tasks running')
    under = float(next(r for r in rows(DATA / 'summary.csv') if r['study'] == 'PLAN_20' and r['model'] == m)['under_n_share'])
    ax.text(1.0, 1.02, f'{NAME[m]}: {t0:.1f} busy minutes, {under:.0%} of them with fewer than 4 tasks running',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9, color=INK)
    style(ax, grid='y')
axes[-1].set_xlabel('minutes; blocks (numbered at the top) back to back in running order; server starts not counted')
fig.tight_layout()
fig.savefig(HERE / 'tasks_running.png', dpi=150)
plt.close(fig)

# --- 3. per-task latency -----------------------------------------------------------------------------------------------
one = {m: {r['task_id']: r for r in rows(DATA / 'per_task_400' / f'{m}_1stream.csv')} for m in MODELS}
fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharex=True, sharey=True)
for ax, m in zip(axes, MODELS):
    f4 = {r['task_id']: r for r in four[m]}
    x = [float(one[m][t]['wall_s']) for t in sorted(f4)]
    y = [float(f4[t]['wall_s']) for t in sorted(f4)]
    ax.scatter(x, y, s=10, color=COLOR[m], alpha=0.6, linewidths=0)
    ax.set_xscale('log')
    ax.set_yscale('log')
    lim = (1, 3000)
    ax.plot(lim, lim, color=GREY, lw=1, ls='--')
    ax.text(2000, 1500, 'same time', fontsize=8, color=MUTED, ha='right', rotation=45, rotation_mode='anchor')
    ratio = rep['PLAN_20 four streams']['latency_ratio_median_per_task'][m]
    ax.set_title(f'{NAME[m]}: median task {ratio:.2f}x slower with four streams', fontsize=9, color=INK)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel('seconds, one stream')
    style(ax, grid='both')
axes[0].set_ylabel('seconds, four streams')
fig.tight_layout()
fig.savefig(HERE / 'per_task_latency.png', dpi=150)
plt.close(fig)
# --- 4. the 27B at 1, 2, 4 and 8 streams -------------------------------------------------------------------------------
summ = rows(DATA / 'summary.csv')
runs = [next(r for r in summ if r['study'] == 'PLAN_19' and r['model'] == '27b')] + \
    [next(r for r in summ if r['study'] == 'PLAN_21b' and r['streams'] == str(n)) for n in (2, 4, 8)]
labels = ['1\n(PLAN_19)', '2', '4', '8']
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.2))
xs = range(len(runs))
for x, r in zip(xs, runs):
    v = float(r['correct_per_hour'])
    a1.bar(x, v, 0.6, color=COLOR['27b'])
    a1.text(x, v + 12, f'{v:.0f}', ha='center', va='bottom', fontsize=9, color=INK)
a1.set_xticks(list(xs))
a1.set_xticklabels(labels, fontsize=9)
a1.set_xlabel('streams (--parallel)')
a1.set_ylabel('correct answers per hour (higher is better)')
a1.set_ylim(0, 1000)
a1.set_title('Throughput', fontsize=10, color=INK)
style(a1)
med = [float(r['latency_median_s']) for r in runs]
p90 = [float(r['latency_p90_s']) for r in runs]
a2.plot(list(xs), p90, color=COLOR['27b'], lw=2, ls='--', marker='o', ms=8, label='90th percentile')
a2.plot(list(xs), med, color=COLOR['27b'], lw=2, marker='o', ms=8, label='median')
for x, a, b in zip(xs, med, p90):
    a2.text(x, a - 3.2, f'{a:.1f}', ha='center', va='top', fontsize=8, color=INK)
    a2.text(x, b + 2.0, f'{b:.1f}', ha='center', va='bottom', fontsize=8, color=INK)
a2.set_xticks(list(xs))
a2.set_xticklabels(labels, fontsize=9)
a2.set_xlabel('streams (--parallel)')
a2.set_ylabel("seconds per task, a task's own time (lower is better)")
a2.set_ylim(0, 55)
a2.set_title('Waiting time of one task', fontsize=10, color=INK)
a2.legend(frameon=False, loc='upper left', fontsize=9)
style(a2)
fig.suptitle('Qwen3.8-27B on the second private set (400 tasks), low thinking, greedy; replies identical at every count',
             fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(HERE / 'streams_27b.png', dpi=150)
plt.close(fig)
print('wrote throughput.png, tasks_running.png, per_task_latency.png, streams_27b.png')
