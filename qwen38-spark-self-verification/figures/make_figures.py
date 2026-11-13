"""Draw the two figures from data/ (needs matplotlib):
  tradeoff.png   correct answers against minutes per correct answer, on h250-dev (50 coding tasks, where the method was
                 developed) and h250-test (150 coding tasks, the locked check)
  tester.png     the test writer T in the four runs: how often its tests accept the reference solution, and where the
                 reference first fails when they do not
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'data'
COLOR = {'flashnext': '#7a3fb0', 'qwen27b': '#2f6fb3'}          # the same model colors as Parts 6-8
GREY = '#9aa0a6'
summary = {(r['run'], r['group'], r['config']): r for r in csv.DictReader(open(DATA / 'summary.csv'))}
notes = {}


def arm(run, config):
    r = summary[run, 'all', config]
    return int(r['correct']), int(r['tasks']), float(r['min_per_correct']) if r.get('min_per_correct') else None


def escalation(run, main):
    """B2's unverified tasks answered by Flash Next B instead (composed from the two runs, not run)."""
    def rows(path):
        return {r['task_id']: r for r in csv.DictReader(open(path))}
    m = rows(DATA / run / f'arm_{main}.csv')
    fn = rows(DATA / 'plan10_reference' / 'flashnext_deep_budget.csv')
    mins = lambda r: min(float(r['wall_s']) / 60, 40)
    un = [t for t in m if m[t]['verified'] == '0']
    ok = sum(m[t]['passed'] == '1' for t in m if t not in un) + sum(fn[t]['passed'] == '1' for t in un)
    return ok, len(m), (sum(mins(r) for r in m.values()) + sum(mins(fn[t]) for t in un)) / ok


# --- 1. the trade-off on dev and test ---------------------------------------------------------------------------------
PANELS = [('run3_dev_tester_new_prompt', 'h250-dev: 50 coding tasks (the method was developed here)'),
          ('run4_test_locked', 'h250-test: 150 coding tasks (locked check)')]
POINTS = [('plan10_qwen27b_vote4_low_CL4', 'qwen27b', 'o', '27B vote over 4 branches (CL4, Part 8)'),
          ('A', 'qwen27b', 's', '27B first code (A, same run as B2)'),
          ('B2', 'qwen27b', '^', '27B self-verified (B2)'),
          ('plan10_flashnext_deep_budget', 'flashnext', 'D', 'Flash Next deep thinking + 32K budget (Part 8)'),
          ('escalation', 'flashnext', '^', 'B2, unverified tasks sent to Flash Next (estimate, not run)')]
# label offsets in points, per panel, to keep the labels apart: (dx, dy, ha)
OFFSET = {'run3_dev_tester_new_prompt': {'plan10_qwen27b_vote4_low_CL4': (0, -16, 'center'), 'A': (0, 10, 'center'),
                                         'B2': (10, -3, 'left'), 'plan10_flashnext_deep_budget': (10, -3, 'left'),
                                         'escalation': (-10, -3, 'right')},
          'run4_test_locked': {'plan10_qwen27b_vote4_low_CL4': (0, -16, 'center'), 'A': (10, -3, 'left'),
                               'B2': (0, 10, 'center'), 'plan10_flashnext_deep_budget': (0, -16, 'center'),
                               'escalation': (0, 10, 'center')}}
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
for ax, (run, title) in zip(axes, PANELS):
    o, n, _ = arm(run, 'O')
    for cfg, model, mk, label in POINTS:
        c, n, mpc = escalation(run, 'B2') if cfg == 'escalation' else arm(run, cfg)
        hollow = cfg == 'escalation'
        ax.scatter(mpc, 100 * c / n, s=70, marker=mk, zorder=3, linewidths=1.6, label=label if run.startswith('run3') else None,
                   facecolors='white' if hollow else COLOR[model], edgecolors=COLOR[model])
        dx, dy, ha = OFFSET[run][cfg]
        ax.annotate(f'{c}/{n}, {mpc:.2f} min', (mpc, 100 * c / n), textcoords='offset points', xytext=(dx, dy), ha=ha,
                    va='center', fontsize=8, color='#202124')
    ax.axhline(100 * o / n, color=GREY, lw=1, ls='--', zorder=1)
    ax.annotate(f'at least one of the 4 branches right: {o}/{n}', (0.15, 100 * o / n), textcoords='offset points',
                xytext=(0, 4), ha='left', fontsize=7.5, color='#5f6368')
    ax.set_xlim(0, 8)
    ax.set_ylim(84, 100)
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel('minutes per correct answer (left is faster)')
    ax.grid(alpha=0.3)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel('correct answers (%)')
fig.suptitle('The 27B checking its own code with its own tests, one DGX Spark', fontsize=10)
fig.legend(loc='lower center', ncol=3, fontsize=7.5, frameon=False)
fig.tight_layout(rect=(0, 0.1, 1, 0.95))
fig.savefig(HERE / 'tradeoff.png', dpi=150)

# --- 2. the test writer -----------------------------------------------------------------------------------------------
RUNS = [('run1_dev_tester_low', 'run 1: dev, low thinking'), ('run2_dev_tester_medium', 'run 2: dev, medium thinking'),
        ('run3_dev_tester_new_prompt', 'run 3: dev, new prompt'), ('run4_test_locked', 'run 4: test, new prompt')]
CATS = [('accepts', 'tests accept the reference', COLOR['qwen27b']),
        ('constant', 'reference fails a hand-written expected value', '#d9822b'),
        ('check', 'fails another check (property, brute-force loop, other)', '#7d8590'),
        ('error', 'test code error (exception, compile error) or timeout', '#b9bec4'),
        ('none', 'no usable tests', '#e3e5e8')]
fig, ax = plt.subplots(figsize=(10, 3.6))
for y, (run, label) in enumerate(RUNS):
    rows = list(csv.DictReader(open(DATA / run / 'tasks.csv')))
    n = len(rows)
    count = {'accepts': sum(r['accepts_reference'] == '1' for r in rows),
             'constant': sum(r['reference_first_failure'] == 'hand-written constant' for r in rows),
             'check': sum(r['reference_first_failure'] in ('other check', 'check inside a loop') for r in rows),
             'error': sum(r['reference_first_failure'] in ('exception', 'compile error', 'timeout', 'no line in output')
                          for r in rows),
             'none': sum(r['tests_given'] == '0' for r in rows)}
    assert sum(count.values()) == n, (run, count)
    left = 0
    for key, name, color in CATS:
        w = 100 * count[key] / n
        ax.barh(y, w, left=left, color=color, edgecolor='white', linewidth=2, height=0.62, label=name if y == 0 else None)
        if count[key] and w >= 3.5:
            ax.text(left + w / 2, y, str(count[key]), ha='center', va='center', fontsize=8,
                    color='white' if key in ('accepts', 'constant', 'check') else '#202124')
        left += w
    ax.text(101, y, f'{count["accepts"]}/{n}', va='center', fontsize=8.5)
ax.set_yticks(range(len(RUNS)))
ax.set_yticklabels([lab for _, lab in RUNS], fontsize=8.5)
ax.invert_yaxis()
ax.set_xlim(0, 108)
ax.set_xlabel('share of tasks (%); numbers are task counts')
ax.set_title("The test writer's tests on the reference solution (where the reference first fails)", fontsize=10)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.legend(loc='lower center', ncol=3, fontsize=7.5, frameon=False)
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig(HERE / 'tester.png', dpi=150)
print('wrote tradeoff.png, tester.png')
