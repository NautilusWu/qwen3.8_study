"""Paired comparison of configurations on the same tasks (time per correct answer as the main measure).

    python3 stats.py [--limit-min 40] [--boot 10000] [--seed 2026] NAME=results.jsonl NAME=results.jsonl ...

Each file is JSONL (one task per line) or CSV with columns: task id ('task_id' or 'id'), 'passed' (true/false or
1/0), and time ('wall_s' seconds, or 'minutes'). Missing or capped times count as --limit-min minutes.
Prints JSON: per configuration correct / total minutes / minutes per correct; for every pair the tasks only one
side solved, the exact two-sided McNemar p, Holm-adjusted p across all pairs, and paired bootstrap 95% intervals
of the accuracy difference (percentage points) and of the minutes-per-correct ratio (first / second).
"""
import argparse
import csv
import json
import math
import random
import sys


def load(path, limit):
    rows = []
    if path.endswith('.csv'):
        rows = list(csv.DictReader(open(path)))
    else:
        rows = [json.loads(l) for l in open(path) if l.strip()]
    out = {}
    for r in rows:
        tid = r.get('task_id', r.get('id'))
        p = r.get('passed')
        passed = p if isinstance(p, bool) else str(p).strip().lower() in ('1', 'true', 'yes')
        if r.get('minutes') not in (None, ''):
            m = float(r['minutes'])
        elif r.get('wall_s') not in (None, ''):
            m = float(r['wall_s']) / 60
        else:
            m = limit
        out[tid] = (passed, min(m, limit))
    return out


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = lambda k: sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * min(tail(b), tail(c)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('configs', nargs='+')
    ap.add_argument('--limit-min', type=float, default=40)
    ap.add_argument('--boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=2026)
    a = ap.parse_args()
    data = {}
    for spec in a.configs:
        name, path = spec.split('=', 1)
        data[name] = load(path, a.limit_min)
    names = list(data)
    ids = sorted(set.intersection(*(set(d) for d in data.values())))
    dropped = {n: len(d) - len(ids) for n, d in data.items() if len(d) != len(ids)}
    out = {'tasks': len(ids), 'dropped_unpaired': dropped, 'totals': {}, 'pairs': {}}
    for n, d in data.items():
        c = sum(d[t][0] for t in ids)
        tm = sum(d[t][1] for t in ids)
        out['totals'][n] = {'correct': c, 'total_min': round(tm, 1), 'min_per_correct': round(tm / c, 3) if c else None}
    rng = random.Random(a.seed)
    boot = [[rng.choice(ids) for _ in ids] for _ in range(a.boot)]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            x, y = data[names[i]], data[names[j]]
            b = sum(1 for t in ids if x[t][0] and not y[t][0])
            c = sum(1 for t in ids if y[t][0] and not x[t][0])
            diffs, ratios = [], []
            for s in boot:
                cx = sum(x[t][0] for t in s)
                cy = sum(y[t][0] for t in s)
                diffs.append(100 * (cx - cy) / len(s))
                if cx and cy:
                    ratios.append((sum(x[t][1] for t in s) / cx) / (sum(y[t][1] for t in s) / cy))
            diffs.sort()
            ratios.sort()
            tx, ty = out['totals'][names[i]], out['totals'][names[j]]
            out['pairs'][f'{names[i]} vs {names[j]}'] = {
                'only_first': b, 'only_second': c, 'p': mcnemar(b, c),
                'acc_diff_pp': round(100 * (b - c) / len(ids), 2),
                'acc_ci95': [round(diffs[int(0.025 * len(diffs))], 2), round(diffs[int(0.975 * len(diffs)) - 1], 2)],
                'time_ratio': round(tx['min_per_correct'] / ty['min_per_correct'], 3) if tx['min_per_correct'] and ty['min_per_correct'] else None,
                'time_ci95': [round(ratios[int(0.025 * len(ratios))], 3), round(ratios[int(0.975 * len(ratios)) - 1], 3)] if ratios else None}
    order = sorted(out['pairs'], key=lambda k: out['pairs'][k]['p'])
    prev = 0.0
    for k_i, k in enumerate(order):
        adj = min(1.0, max(prev, (len(order) - k_i) * out['pairs'][k]['p']))
        out['pairs'][k]['p_holm'] = round(adj, 4)
        out['pairs'][k]['p'] = round(out['pairs'][k]['p'], 4)
        prev = adj
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == '__main__':
    main()
