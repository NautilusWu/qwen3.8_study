"""Recompute the README's tables from the per-task files in data/ (no GPU, no task files needed).

    python3 code/tables.py            print the tables and check them against data/summary.csv
    python3 code/tables.py --write    (re)write data/summary.csv

Times are capped at 40 minutes per task. Minutes per correct answer = all minutes (wrong answers included) / correct.
"""
import csv
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / 'data'
CONFIGS = {'per_task_h250': ('flashnext_deep_budget', 'qwen27b_vote4_low_CL4'),
           'per_task_arxivmath': ('flashnext_deep_budget', 'qwen27b_vote4_low_CL4'),
           'per_task_72': ('flashnext_low_D2', 'flashnext_vote4_low_CL4', 'flashnext_deep_A65', 'flashnext_deep_budget',
                           'qwen27b_low_D2', 'qwen27b_vote4_low_CL4', 'qwen27b_deep_A65')}
SET = {'per_task_h250': 'h250', 'per_task_arxivmath': 'arxivmath', 'per_task_72': 'old72'}


def load(folder, name):
    return {r['task_id']: r for r in csv.DictReader(open(DATA / folder / f'{name}.csv'))}


def groups(folder, rr):
    ids = sorted(rr)
    if folder == 'per_task_h250':
        g = {'all': ids}
        for col, vals in (('split', ('dev', 'test')), ('category', ('python', 'cpp', 'process')), ('tier', ('A', 'B', 'C'))):
            g.update({v: [t for t in ids if rr[t][col] == v] for v in vals})
        return g
    if folder == 'per_task_arxivmath':
        return {'all': ids, 'after_release': [t for t in ids if rr[t]['after_release'] == '1'],
                'before_release': [t for t in ids if rr[t]['after_release'] == '0']}
    return {'all': ids, 'hard': [t for t in ids if rr[t]['set'] == 'hard'], 'confirm': [t for t in ids if rr[t]['set'] == 'confirm']}


def summarize(rr, ids):
    ok = sum(rr[t]['passed'] == '1' for t in ids)
    mins = sum(min(float(rr[t]['wall_s']) / 60, 40) for t in ids)
    return {'tasks': len(ids), 'correct': ok, 'total_min': f'{mins:.1f}', 'min_per_correct': f'{mins / ok:.2f}' if ok else '',
            'hit_output_limit': sum(int(rr[t]['hit_output_limit']) for t in ids),
            'no_answer': sum(rr[t]['answered'] == '0' for t in ids)}


def main():
    out, extra = [], []
    for folder, names in CONFIGS.items():
        data = {n: load(folder, n) for n in names}
        first = data[names[0]]
        for g, ids in groups(folder, first).items():
            for n in names:
                out.append({'set': SET[folder], 'group': g, 'config': n, **summarize(data[n], ids)})
            if folder != 'per_task_72':
                fb, cl = ({t for t in ids if data[n][t]['passed'] == '1'} for n in names)
                extra.append(f'{SET[folder]:9s} {g:15s} union {len(fb | cl)}/{len(ids)}, both wrong {len(ids) - len(fb | cl)}/{len(ids)}, '
                             f'only Flash Next {len(fb - cl)}, only 27B {len(cl - fb)}')
        if folder != 'per_task_72':
            cl, fb = data['qwen27b_vote4_low_CL4'], data['flashnext_deep_budget']
            for g, ids in groups(folder, first).items():
                maj = [t for t in ids if cl[t]['selection_rule'] == 'majority']
                extra.append(f'{SET[folder]:9s} {g:15s} 27B CL4: >=3 of 4 agree {len(maj)} ({sum(cl[t]["passed"] == "1" for t in maj)} correct), '
                             f'pass@4 reference {sum(int(cl[t]["branches_correct"]) > 0 for t in ids)}/{len(ids)}; '
                             f'Flash Next B over 32,768 tokens {sum(fb[t]["over_32768_tokens"] == "1" for t in ids)}/{len(ids)} '
                             f'({sum(fb[t]["over_32768_tokens"] == "1" and fb[t]["passed"] == "1" for t in ids)} correct)')
    cols = ['set', 'group', 'config', 'tasks', 'correct', 'total_min', 'min_per_correct', 'hit_output_limit', 'no_answer']
    path = DATA / 'summary.csv'
    if '--write' in sys.argv:
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, cols)
            w.writeheader()
            w.writerows(out)
        print(f'wrote {path.name}: {len(out)} rows')
        return
    for r in out:
        print(f"{r['set']:9s} {r['group']:15s} {r['config']:24s} {r['correct']:>3}/{r['tasks']:<3} {r['total_min']:>7} min "
              f"{r['min_per_correct']:>6} per correct, limit hits {r['hit_output_limit']}, no answer {r['no_answer']}")
    print('\n'.join(extra))
    old = [dict(r) for r in csv.DictReader(open(path))]
    same = old == [{k: str(v) for k, v in r.items()} for r in out]
    print(f'\nsummary.csv: {"all " + str(len(out)) + " rows match" if same else "DIFFERS"}')
    sys.exit(0 if same else 1)


if __name__ == '__main__':
    main()
