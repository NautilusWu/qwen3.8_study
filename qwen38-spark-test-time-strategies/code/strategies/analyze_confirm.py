"""PLAN_04 tests C1-C5 on the 48 confirmation tasks, exactly as locked.

    python3 analyze_confirm.py conf-001     -> results/conf-001-tests.json and a printed table
    python3 analyze_confirm.py --csv ../../data/per_task.csv
                                            -> the same tests recomputed from the published per-task results
"""
import csv
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIMIT_MIN = 40


def load_csv(path, m):
    """Published per-task rows (data/per_task.csv, phase 'confirmation') in the shape of graded.jsonl."""
    return {r['task_id']: {'task_id': r['task_id'], 'category': r['category'], 'passed': r['passed'] == '1',
                           'wall_s': float(r['wall_s']), 'finish': 'length' if r['hit_output_limit'] == '1' else 'stop'}
            for r in csv.DictReader(open(path)) if r['phase'] == 'confirmation' and r['method'] == m}


def load(run, m):
    return {json.loads(l)['task_id']: json.loads(l) for l in (HERE / 'results' / f'{run}-{m}' / 'graded.jsonl').read_text().splitlines()}


def minutes(r):
    return min((r.get('wall_s') or LIMIT_MIN * 60) / 60, LIMIT_MIN)


def summary(rows):
    c = sum(r['passed'] for r in rows.values())
    t = sum(minutes(r) for r in rows.values())
    return {'correct': c, 'coding': sum(r['passed'] for r in rows.values() if r['category'] == 'coding'),
            'reasoning': sum(r['passed'] for r in rows.values() if r['category'] == 'reasoning'),
            'total_min': round(t, 1), 'min_per_correct': round(t / c, 3) if c else None,
            'hit_cap': sum(r.get('finish') == 'length' or r.get('exec_finish') == 'length' for r in rows.values()),
            'timed_out': sum(bool(r.get('timed_out')) for r in rows.values())}


def sign_test(x, y):
    """Exact McNemar: x-only and y-only counts; two-sided and one-sided (x better) p."""
    b = sum(x[t]['passed'] and not y[t]['passed'] for t in x)
    c = sum(y[t]['passed'] and not x[t]['passed'] for t in x)
    n = b + c
    if n == 0:
        return b, c, 1.0, 1.0
    tail = lambda k: sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    one = tail(b)                                    # P(X >= b) under p = 1/2
    two = min(1.0, 2 * min(tail(b), tail(c)))
    return b, c, two, one


def ratio_ci(x, y, n_boot=10000, seed=2026):
    """95% paired bootstrap interval of (min per correct of x) / (min per correct of y), resampling tasks."""
    ids = sorted(x)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        s = [rng.choice(ids) for _ in ids]
        cx = sum(x[t]['passed'] for t in s); cy = sum(y[t]['passed'] for t in s)
        if not cx or not cy:
            continue
        vals.append((sum(minutes(x[t]) for t in s) / cx) / (sum(minutes(y[t]) for t in s) / cy))
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def main():
    from_csv = sys.argv[1] == '--csv'
    run = sys.argv[2] if from_csv else sys.argv[1]
    d = {m: (load_csv(run, m) if from_csv else load(run, m)) for m in ('A65', 'D2', 'H', 'CL4')}
    s = {m: summary(r) for m, r in d.items()}
    tests = {}
    r1 = s['D2']['min_per_correct'] / s['A65']['min_per_correct']
    lo, hi = ratio_ci(d['D2'], d['A65'])
    tests['C1'] = {'ratio': round(r1, 3), 'ci95': [round(lo, 3), round(hi, 3)], 'confirmed': r1 <= 0.5 and hi < 1}
    b, c, two, _ = sign_test(d['D2'], d['A65'])
    tests['C2'] = {'D2_only': b, 'A65_only': c, 'p_two_sided': round(two, 4),
                   'confirmed': two >= 0.05 and s['D2']['correct'] >= s['A65']['correct'] - 3}
    r3 = s['H']['min_per_correct'] / s['A65']['min_per_correct']
    lo, hi = ratio_ci(d['H'], d['A65'])
    tests['C3'] = {'H_correct': s['H']['correct'], 'A65_correct': s['A65']['correct'], 'ratio': round(r3, 3),
                   'ci95': [round(lo, 3), round(hi, 3)],
                   'confirmed': s['H']['correct'] >= s['A65']['correct'] - 2 and r3 <= 0.5 and hi < 1}
    b, c, two, one = sign_test(d['CL4'], d['D2'])
    tests['C4'] = {'CL4_only': b, 'D2_only': c, 'p_one_sided': round(one, 4), 'confirmed': one < 0.05}
    r5 = s['CL4']['min_per_correct'] / s['D2']['min_per_correct']
    lo, hi = ratio_ci(d['CL4'], d['D2'])
    tests['C5'] = {'ratio': round(r5, 3), 'ci95': [round(lo, 3), round(hi, 3)], 'confirmed': r5 <= 1.5}
    b, c, two, _ = sign_test(d['H'], d['D2'])
    extra = {'H_vs_D2': {'H_only': b, 'D2_only': c, 'p_two_sided': round(two, 4)}}
    out = {'summary': s, 'tests': tests, 'reported_only': extra}
    if not from_csv:
        (HERE / 'results' / f'{run}-tests.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
