"""Recompute the README's tables from the per-task files in data/ (no GPU, no task files needed).

    python3 code/tables.py            print the tables and check them against data/summary.csv and data/tester_summary.csv
    python3 code/tables.py --write    (re)write the two summary files

Times are capped at 40 minutes per task. Minutes per correct answer = all minutes (wrong answers included) / correct.
"""
import csv
import math
import sys
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / 'data'
LIMIT_MIN = 40
RUNS = {'run1_dev_tester_low': ('A', 'B', 'C'), 'run2_dev_tester_medium': ('A', 'B'),
        'run3_dev_tester_new_prompt': ('A', 'B', 'B2'), 'run4_test_locked': ('A', 'B', 'B2')}
MAIN = {'run1_dev_tester_low': 'B', 'run2_dev_tester_medium': 'B', 'run3_dev_tester_new_prompt': 'B2',
        'run4_test_locked': 'B2'}            # the arm each run is read by (run 3 also reports B; B2 = B there)
GROUPS = {'all': lambda m: True, 'python': lambda m: m['category'] == 'python', 'cpp': lambda m: m['category'] == 'cpp',
          'tier A': lambda m: m['tier'] == 'A', 'tier B': lambda m: m['tier'] == 'B', 'tier C': lambda m: m['tier'] == 'C',
          'spec': lambda m: m['spec'] == '1', 'other': lambda m: m['spec'] == '0'}
REF_KINDS = ('assert failed', 'check failed', 'exception', 'crash', 'compile error', 'timeout')
SITES = ('hand-written constant', 'check inside a loop', 'other check', 'exception', 'compile error', 'timeout',
         'no line in output')


def read(path):
    return {r['task_id']: r for r in csv.DictReader(open(path))}


def minutes(r):
    return min(float(r['wall_s']) / 60, LIMIT_MIN)


def mcnemar(b, c):
    """Exact two-sided McNemar p (as in code/stats.py)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = lambda k: sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * min(tail(b), tail(c)))


def clopper_pearson(k, n, alpha=0.05):
    """Exact two-sided 95% interval for k of n (bisection, as in the locked PLAN_12 analysis)."""
    cdf = lambda j, p: sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(j + 1))

    def root(f):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < alpha / 2 else (lo, mid)
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else root(lambda p: 1 - cdf(k - 1, p))
    upper = 1.0 if k == n else root(lambda p: alpha - cdf(k, p))
    return lower, upper


def arm_row(xs, ids):
    ok = sum(xs[t]['passed'] == '1' for t in ids)
    mins = sum(minutes(xs[t]) for t in ids)
    out = {'tasks': len(ids), 'correct': ok, 'total_min': f'{mins:.1f}', 'min_per_correct': f'{mins / ok:.2f}' if ok else ''}
    if 'verified' in next(iter(xs.values())):
        v = [t for t in ids if xs[t]['verified'] == '1']
        out.update(verified=len(v), verified_wrong=sum(xs[t]['passed'] == '0' for t in v), unverified=len(ids) - len(v),
                   unverified_correct=sum(xs[t]['passed'] == '1' for t in ids if t not in v))
    return out


def load_run(run):
    d = DATA / run
    tasks = read(d / 'tasks.csv')
    arms = {a: read(d / f'arm_{a}.csv') for a in RUNS[run]}
    branches = list(csv.DictReader(open(d / 'branches.csv')))
    return tasks, arms, branches


def tester_row(tasks, branches, prefix=''):
    """The quality of the model's tests (T, or the critic K in run 1 with prefix 'critic_')."""
    have = [r for r in tasks.values() if r[prefix + 'tests_given'] == '1'] if prefix else \
        [r for r in tasks.values() if r['tests_given'] == '1']
    acc = [r for r in have if r[prefix + 'accepts_reference'] == '1']
    out = {'tasks': len(tasks), 'tests_given': len(have), 'accepts_reference': len(acc)}
    if not prefix:
        kinds = Counter(r['reference_failure'] for r in have if r['accepts_reference'] == '0')
        out.update({f'ref_{k.replace(" ", "_")}': kinds.get(k, 0) for k in REF_KINDS})
        out['test_code_errors'] = sum(kinds.get(k, 0) for k in ('exception', 'crash', 'compile error'))
        out['passes_on_empty_code'] = sum(r['passes_on_empty_code'] == '1' for r in have)
        sites = Counter(r['reference_first_failure'] for r in have if r['accepts_reference'] == '0')
        assert set(sites) <= set(SITES), sites
        out.update({f'first_failure_{s.replace(" ", "_").replace("-", "_")}': sites.get(s, 0) for s in SITES})
    for cls in ('wrong', 'slow'):
        out[f'bads_{cls}_rejected'] = sum(int(r[f'{prefix}bads_{cls}_rejected']) for r in have)
        out[f'bads_{cls}_total'] = sum(int(r[f'{prefix}bads_{cls}_total']) for r in have)
    col = 'critic_tests_passed' if prefix else 'tests_passed'
    cands = [b for b in branches if b['has_code'] == '1' and b[col] != '']
    out.update(candidates_tested=len(cands),
               false_rejections=sum(b['hidden_passed'] == '1' and b[col] == '0' for b in cands),
               hidden_correct=sum(b['hidden_passed'] == '1' for b in cands),
               false_acceptances=sum(b['hidden_passed'] == '0' and b[col] == '1' for b in cands),
               hidden_wrong=sum(b['hidden_passed'] == '0' for b in cands))
    if not prefix:
        py = [r for r in have if r['category'] == 'python']
        out.update(random_inputs=sum(r['random_inputs'] == '1' for r in have),
                   python_tests=len(py), python_tests_with_unused_functions=sum(int(r['python_unused_functions'] or 0) > 0 for r in py),
                   python_unused_functions=sum(int(r['python_unused_functions'] or 0) for r in py))
    return out


def build():
    summary, tester, notes = [], [], []
    p10 = {n: read(DATA / 'plan10_reference' / f'{n}.csv') for n in ('qwen27b_vote4_low_CL4', 'flashnext_deep_budget')}
    for run, arms in RUNS.items():
        tasks, xs, branches = load_run(run)
        split = 'dev' if 'dev' in run else 'test'
        assert all(p10[n][t]['split'] == split for n in p10 for t in tasks)
        hidden = {}
        for b in branches:
            hidden.setdefault(b['task_id'], []).append(b['hidden_passed'] == '1')
        for g, f in GROUPS.items():
            ids = sorted(t for t, m in tasks.items() if f(m))
            for a in arms:
                summary.append({'run': run, 'group': g, 'config': a, **arm_row(xs[a], ids)})
            summary.append({'run': run, 'group': g, 'config': 'O', 'tasks': len(ids), 'correct': sum(any(hidden[t]) for t in ids)})
            for n in p10:
                summary.append({'run': run, 'group': g, 'config': f'plan10_{n}', **arm_row(p10[n], ids)})
        tester.append({'run': run, 'writer': 'T', **tester_row(tasks, branches)})
        if run.startswith('run1'):
            tester.append({'run': run, 'writer': 'K (critic)', **tester_row(tasks, branches, 'critic_')})
        # reported only: the plans' gates, where "verified" comes from, the escalation estimate
        m, a = xs[MAIN[run]], xs['A']
        ids = sorted(tasks)
        ra, rm = arm_row(a, ids), arm_row(m, ids)
        if split == 'dev':
            for name in arms[1:]:
                r = arm_row(xs[name], ids)
                gates = (r['correct'] >= ra['correct'], r['verified_wrong'] <= 2,
                         float(r['min_per_correct']) <= 2 * float(ra['min_per_correct']))
                notes.append(f'{run:27s} gates for {name}: correct >= A {gates[0]}, verified wrong <= 2 {gates[1]}, '
                             f'time <= 2x A {gates[2]} ({r["min_per_correct"]} vs {ra["min_per_correct"]})')
        ver = [t for t in ids if m[t]['verified'] == '1']
        un = [t for t in ids if t not in ver]
        notes.append(f'{run:27s} {MAIN[run]}: verified {len(ver)} (wrong: {MAIN[run]} {sum(m[t]["passed"] == "0" for t in ver)}, '
                     f'A {sum(a[t]["passed"] == "0" for t in ver)}), unverified {len(un)} (wrong: {MAIN[run]} '
                     f'{sum(m[t]["passed"] == "0" for t in un)}, A {sum(a[t]["passed"] == "0" for t in un)}); '
                     f'{MAIN[run]} chose differently from A on {sum(m[t]["chosen_branch"] != a[t]["chosen_branch"] for t in ids)}')
        fn = p10['flashnext_deep_budget']
        ok = sum(m[t]['passed'] == '1' for t in ver) + sum(fn[t]['passed'] == '1' for t in un)
        mins = sum(minutes(m[t]) for t in ids) + sum(minutes(fn[t]) for t in un)
        notes.append(f'{run:27s} escalation estimate ({MAIN[run]} unverified -> Flash Next B): {ok}/{len(ids)}, {mins:.1f} min, '
                     f'{mins / ok:.2f} per correct; on the {len(un)} escalated: {MAIN[run]} {sum(m[t]["passed"] == "1" for t in un)}, '
                     f'Flash Next B {sum(fn[t]["passed"] == "1" for t in un)}')
    # the plans' main comparisons
    acc = {run: {t: r['accepts_reference'] == '1' for t, r in load_run(run)[0].items()} for run in RUNS}
    base = acc['run1_dev_tester_low']
    for run in ('run2_dev_tester_medium', 'run3_dev_tester_new_prompt'):
        gained = sum(acc[run][t] and not base[t] for t in base)
        lost = sum(base[t] and not acc[run][t] for t in base)
        notes.append(f'{run:27s} T accepts the reference {sum(acc[run].values())}/50 vs run 1 {sum(base.values())}/50: '
                     f'gained {gained}, lost {lost}, exact McNemar p = {mcnemar(gained, lost):.4f}')
    b2 = read(DATA / 'run4_test_locked' / 'arm_B2.csv')
    ver = [t for t in b2 if b2[t]['verified'] == '1']
    wrong = sum(b2[t]['passed'] == '0' for t in ver)
    lo, hi = clopper_pearson(wrong, len(ver))
    notes.append(f'run4_test_locked            M2: verified wrong {wrong}/{len(ver)}, 95% exact interval '
                 f'{100 * lo:.2f}%-{100 * hi:.2f}% (threshold 17/150 = 11.33%)')
    return summary, tester, notes


def check(path, rows):
    old = [dict(r) for r in csv.DictReader(open(path))]
    cols = list(old[0]) if old else []
    new = [{k: str(r.get(k, '')) for k in cols} for r in rows]
    return old == new and all(set(r) <= set(cols) for r in rows)


def main():
    summary, tester, notes = build()
    files = {'summary.csv': summary, 'tester_summary.csv': tester}
    if '--write' in sys.argv:
        for name, rows in files.items():
            cols = list(dict.fromkeys(k for r in rows for k in r))
            with open(DATA / name, 'w', newline='') as f:
                w = csv.DictWriter(f, cols, lineterminator='\n')
                w.writeheader()
                w.writerows(rows)
            print(f'wrote {name}: {len(rows)} rows')
        return
    for r in summary:
        if r['group'] == 'all':
            v = f", verified {r['verified']} (wrong {r['verified_wrong']})" if 'verified' in r else ''
            per = f"{r['total_min']:>6} min {r['min_per_correct']:>5} per correct" if 'total_min' in r else ''
            print(f"{r['run']:27s} {r['config']:34s} {r['correct']:>3}/{r['tasks']:<3} {per}{v}")
    print()
    for r in tester:
        print(f"{r['run']:27s} {r['writer']:10s} tests {r['tests_given']}/{r['tasks']}, accepts reference {r['accepts_reference']}, "
              f"wrong answers rejected {r['bads_wrong_rejected']}/{r['bads_wrong_total']}, slow {r['bads_slow_rejected']}/{r['bads_slow_total']}, "
              f"false rejections {r['false_rejections']}/{r['hidden_correct']}, false acceptances {r['false_acceptances']}/{r['hidden_wrong']}")
    print('\n' + '\n'.join(notes))
    same = {name: check(DATA / name, rows) for name, rows in files.items()}
    print('\n' + '; '.join(f'{name}: {"all " + str(len(files[name])) + " rows match" if ok else "DIFFERS"}'
                           for name, ok in same.items()))
    sys.exit(0 if all(same.values()) else 1)


if __name__ == '__main__':
    main()
