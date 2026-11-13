"""Recompute every selection rule (A, B, B2, C) from the recorded times in data/<run>/branches.csv and tasks.csv, with
the rules exactly as they ran (code/rules/p11_rules.py, p11c_rules.py), and check the result against data/<run>/arm_*.csv.
No GPU, no task files needed.

    python3 code/arms.py                         the four runs in data/
    python3 code/arms.py --dir results/<run_id>  a run of code/selfverify.py (its arm_A, arm_B and arm_B2.csv)

Times are seconds from the moment a task's five requests were sent; each task is capped at 40 minutes.
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code' / 'rules'))
import p11_rules  # noqa: E402
import p11c_rules  # noqa: E402

LIMIT = 40 * 60
RUNS = {'run1_dev_tester_low': ('A', 'B', 'C'), 'run2_dev_tester_medium': ('A', 'B'),
        'run3_dev_tester_new_prompt': ('A', 'B', 'B2'), 'run4_test_locked': ('A', 'B', 'B2')}


def flag(x):
    return x == '1'


def records(d):
    """One record per task of the folder d, in the shape the rules read (see the docstring of p11_rules.py)."""
    tasks = {r['task_id']: r for r in csv.DictReader(open(d / 'tasks.csv'))}
    recs = {}
    for t, r in tasks.items():
        tester = {'end_s': float(r['tester_end_s']), 'tests': '' if flag(r['tests_given']) else None}
        if r['empty_run_passed'] != '':
            tester['empty_rt'] = {'passed': flag(r['empty_run_passed']), 'container_s': float(r['empty_run_container_s'])}
        critic = None
        if flag(r.get('critic_ran', '0')):
            critic = {'duration_s': float(r['critic_duration_s']), 'tests': '' if flag(r['critic_tests_given']) else None}
        recs[t] = {'branches': [], 'tester': tester, 'critic': critic}
    for b in csv.DictReader(open(d / 'branches.csv')):
        x = {'branch': int(b['branch']), 'end_s': float(b['end_s']), 'candidate': flag(b['has_code']),
             'hidden': {'passed': flag(b['hidden_passed'])}}
        if b['tests_passed'] != '':
            x['t_run'] = {'passed': flag(b['tests_passed']), 'container_s': float(b['tests_container_s'])}
        if b.get('critic_tests_passed', '') != '':
            x['k_run'] = {'passed': flag(b['critic_tests_passed']), 'container_s': float(b['critic_container_s'])}
        recs[b['task_id']]['branches'].append(x)
    for r in recs.values():
        r['branches'].sort(key=lambda x: x['branch'])
        assert [x['branch'] for x in r['branches']] == list(range(len(r['branches'])))
    return recs


def arm(rec, name):
    x = {'A': p11_rules.arm_a, 'B': p11_rules.arm_b, 'C': p11_rules.arm_c, 'B2': p11c_rules.arm_b2}[name](rec, LIMIT)
    passed = x['choice'] is not None and not x['timed_out'] and rec['branches'][x['choice']]['hidden']['passed']
    why = ('first branch with code' if x['choice'] is not None else 'no code') if name == 'A' else \
        (x['k'] if name == 'C' else x['why'])
    return {'passed': str(int(passed)), 'wall_s': f"{min(x['time_s'], LIMIT):.3f}",
            'chosen_branch': '' if x['choice'] is None else str(x['choice']), 'verified': str(int(x['verified'])), 'why': why}


def check(d, names):
    """Recompute each rule in names for every task of the folder d and compare with d/arm_<name>.csv -> tasks that differ."""
    recs = records(d)
    bad = 0
    for name in names:
        pub = {r['task_id']: r for r in csv.DictReader(open(d / f'arm_{name}.csv'))}
        assert sorted(pub) == sorted(recs), (d.name, name)
        diff = [t for t in sorted(recs) if arm(recs[t], name) != {k: pub[t][k] for k in
                                                                   ('passed', 'wall_s', 'chosen_branch', 'verified', 'why')}]
        bad += len(diff)
        print(f'{d.name:28s} {name:2s} {len(recs) - len(diff):>3}/{len(recs)} tasks match'
              + (f'; differ: {", ".join(diff[:10])}' if diff else ''))
    return bad


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--dir':
        d = Path(sys.argv[2])
        bad = check(d, [n for n in ('A', 'B', 'B2', 'C') if (d / f'arm_{n}.csv').exists()])
    elif len(sys.argv) == 1:
        bad = sum(check(ROOT / 'data' / run, names) for run, names in RUNS.items())
    else:
        sys.exit(__doc__)
    print(f'\narms: {"all match" if not bad else f"{bad} differ"}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
