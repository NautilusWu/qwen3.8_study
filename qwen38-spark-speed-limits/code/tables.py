"""Part 13: recompute every table and test of the README from data/ (no GPU, no task text).

    python3 code/tables.py           print the tables
    python3 code/tables.py --check   also compare with data/summary.csv, data/tests_preregistered.json and
                                     data/reported_only.json (exit 1 on any difference)
    python3 code/tables.py --write   (re)write those three files

Paired tests use stats.py (exact McNemar, paired bootstrap 95% intervals, 10,000 resamples, seed 2026, 40-minute cap),
called the way the locked analyses called it. The cross-plan section also reads Part 5, Part 10 and Part 11's
published per-task files from the sibling folders when they are present (the repository has them side by side).
"""
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / 'data'
PT = D / 'per_task_400'
STATS = ROOT / 'code' / 'stats.py'
REPO = ROOT.parent
SIB = {'part5_CL4_run': REPO / 'qwen38-spark-engine-upgrade/data/per_task_400/qwen27b_vote4_low_CL4_tf060_part5_run.csv',
       'p13_tf125': REPO / 'qwen38-spark-engine-upgrade/data/per_task_400/qwen27b_vote4_low_CL4_tf125.csv',
       'p13_tf060': REPO / 'qwen38-spark-engine-upgrade/data/per_task_400/qwen27b_vote4_low_CL4_tf060.csv',
       'fn_U3X_MTP': REPO / 'qwen38-spark-flash-next-gguf/data/per_task_400/UD-IQ3_XXS_MTP.csv',
       'fn_U3X': REPO / 'qwen38-spark-flash-next-gguf/data/per_task_400/UD-IQ3_XXS.csv'}
LIMIT = {'fn_U3X_MTP': 120, 'fn_U3X': 120}


def rows(path):
    return {r['task_id']: r for r in csv.DictReader(open(path))}


def run(name):
    p = SIB.get(name) or PT / f'{name}.csv'
    lim = LIMIT.get(name, 40) * 60
    return {t: {'passed': r['passed'] in ('1', 'True', 'true'), 'wall': min(float(r['wall_s']), lim),
                'tokens': int(r['output_tokens'])} for t, r in rows(p).items()}


def totals(x):
    c = sum(v['passed'] for v in x.values())
    t = sum(v['wall'] for v in x.values())
    n = sum(v['tokens'] for v in x.values())
    return {'correct': c, 'total_min': round(t / 60, 1), 'min_per_correct': round(t / 60 / c, 3), 'output_tokens': n,
            'tokens_per_s': round(n / t, 2)}


def stats(a, b, lim=40):
    pa, pb = SIB.get(a) or PT / f'{a}.csv', SIB.get(b) or PT / f'{b}.csv'
    out = subprocess.run([sys.executable, str(STATS), '--limit-min', str(lim), f'A={pa}', f'B={pb}'],
                         capture_output=True, text=True, check=True).stdout
    p = json.loads(out)['pairs']['A vs B']
    return {k: p[k] for k in ('only_first', 'only_second', 'p', 'acc_diff_pp', 'acc_ci95', 'time_ratio', 'time_ci95')}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run_max = [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        run_max = max(run_max, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = round(run_max, 4)
    return adj


def csvrows(path):
    return list(csv.DictReader(open(path)))


def main():
    T, R = {}, {}
    # ------------------------------------------------------------ summary of the per-task runs
    names = ['27b_CL4_fp4proj', '27b_CL4_original', '27b_D2_fp4proj', '27b_D2_original', '27b_D2_p1_tf127', '27b_D2_p1_tf060',
             '27b_D2_original_part5_run']
    study = {'27b_CL4_fp4proj': 'PLAN_15', '27b_CL4_original': 'PLAN_15', '27b_D2_fp4proj': 'PLAN_15', '27b_D2_original': 'PLAN_15',
             '27b_D2_p1_tf127': 'PLAN_17', '27b_D2_p1_tf060': 'PLAN_17', '27b_D2_original_part5_run': 'Part 5 (re-run reference)'}
    X = {n: run(n) for n in names}
    summary = []
    for n in names:
        t = totals(X[n])
        lim_hits = sum(int(r['hit_output_limit']) for r in csvrows(PT / f'{n}.csv'))
        by = {}
        for r in csvrows(PT / f'{n}.csv'):
            g = r['group']
            by.setdefault(g, [0, 0])
            by[g][0] += int(r['passed'])
            by[g][1] += 1
        summary.append({'run': n, 'study': study[n], **t, 'hit_output_limit': lim_hits,
                        'correct_python': by['coding_python'][0], 'correct_cpp': by['coding_cpp'][0], 'correct_reasoning': by['reasoning'][0]})

    # ------------------------------------------------------------ pre-registered tests
    prof = json.load(open(D / 'profile_round.json'))
    for path in ('single', 'multi'):
        e = prof['paths'][path]['estimate']
        s = round(e['T_ms'] / (e['T_ms'] - e['H_k_ms'] - e['H_g_ms']), 3)
        T[f'PLAN_14 S {path}'] = s
    T['PLAN_14 decision'] = 'write step 2' if max(T['PLAN_14 S single'], T['PLAN_14 S multi']) >= 1.10 else 'stop'

    p15 = {ph: stats(f'27b_{ph}_fp4proj', f'27b_{ph}_original') for ph in ('CL4', 'D2')}
    for ph, h in zip(('CL4', 'D2'), holm([p15['CL4']['p'], p15['D2']['p']])):
        p15[ph]['p_holm'] = h
        p15[ph]['accuracy_ok'] = p15[ph]['acc_ci95'][0] >= -2.5
        p15[ph]['faster'] = p15[ph]['time_ratio'] < 1 and p15[ph]['time_ci95'][1] < 1
    T['PLAN_15'] = p15
    T['PLAN_15 decision'] = ('no accuracy drop detected; no speed difference detected'
                             if all(p15[ph]['accuracy_ok'] and not p15[ph]['faster'] for ph in p15) else 'see rows')

    tw = csvrows(D / 'tree_width.csv')
    b0 = {}
    for r in tw:
        if r['study'] == 'PLAN_16 B0':
            b0.setdefault(int(r['tree_rows']), []).append(float(r['tok_s']))
    base = statistics.mean(b0[16])
    T['PLAN_16 B0'] = {w: {'ratio_vs_16': round(statistics.mean(v) / base, 4), 'both_beat_both_16': min(v) > max(b0[16])}
                       for w, v in sorted(b0.items())}
    cands = [w for w, v in T['PLAN_16 B0'].items() if w != 16 and v['ratio_vs_16'] >= 1.03 and v['both_beat_both_16']]
    T['PLAN_16 B0 keep'] = max(cands, key=lambda w: T['PLAN_16 B0'][w]['ratio_vs_16']) if cands else 16
    dr = json.load(open(D / 'drafter.json'))
    T['PLAN_16 B1c'] = {'ratio': round(dr['B1c']['proxy_finetuned'] / dr['B1c']['proxy_original_served'], 4),
                        'passed': dr['B1c']['proxy_finetuned'] / dr['B1c']['proxy_original_served'] >= 1.05}

    q1 = stats('27b_D2_p1_tf127', '27b_D2_p1_tf060')
    q1['accuracy_ok'] = q1['acc_ci95'][0] >= -2.5
    q1['faster'] = q1['time_ratio'] < 1 and q1['time_ci95'][1] < 1
    T['PLAN_17 Q1'] = q1
    sp = [r for r in csvrows(D / 'engine_speed.csv') if r['comparison'] == 'S1_vs_N1' and r['category'] == 'agent']
    T['PLAN_17 Q2 agent speed ratios'] = [float(r['speed_ratio']) for r in sp]
    T['PLAN_17 Q2 decision'] = 'use DSpine' if min(T['PLAN_17 Q2 agent speed ratios']) >= 1.05 else 'do not use DSpine'

    # PLAN_22: the ceiling, from the confirmed forward floor and the M2 widths (fixed numbers from PLAN_14)
    fw = {r['kind']: r for r in csvrows(D / 'ceiling_forward.csv')}
    F = float(fw['confirmation (F)']['ms_median'])
    TT, M16, HG = 105.9, 89.2, 2.1
    m2 = {}
    for r in tw:
        if r['study'] == 'PLAN_22 M2':
            m2.setdefault(int(r['tree_rows']), []).append(r)
    mean = statistics.mean
    D16 = mean(float(r['tokens_per_round']) for r in m2[16])
    nonv = {w: mean(float(r['draft_ms']) + float(r['sample_ms']) + float(r['commit_ms']) for r in v) for w, v in m2.items()}
    cub = {int(r['rows']): float(r['ms_median']) for r in csvrows(D / 'ceiling_cublas.csv')}
    basec = TT - (M16 - F) - HG
    p22 = {}
    for w, v in sorted(m2.items()):
        g = mean(float(r['tokens_per_round']) for r in v) / D16
        dS = nonv[w] - nonv[16]
        p22[w] = {'g': round(g, 4), 'dS_ms': round(dS, 3), 'C_hi': round(g * TT / basec, 4), 'C_mid': round(g * TT / (basec + dS), 4)}
        if w in cub:
            p22[w]['C_lib'] = round(g * TT / (TT - M16 + cub[w] - HG), 4)
    cstar_w = max((w for w in (16, 32, 48, 64) if w in p22), key=lambda w: p22[w]['C_hi'])
    T['PLAN_22'] = {'F_ms': F, 'S_kernel_only': round(TT / basec, 4), 'by_width': p22, 'C_star': p22[cstar_w]['C_hi'],
                    'C_star_width': cstar_w,
                    'verdict': 'stop: even the optimistic ceiling is below 1.20' if p22[cstar_w]['C_hi'] < 1.20 else 'ceiling >= 1.20: the user decides'}

    # ------------------------------------------------------------ reported only: the cross-plan decomposition
    comps = [('PLAN_15 NVFP4 projections vs original (CL4)', '27b_CL4_fp4proj', '27b_CL4_original', 'change'),
             ('PLAN_15 NVFP4 projections vs original (D2)', '27b_D2_fp4proj', '27b_D2_original', 'change'),
             ('PLAN_17 TensorFold 1.2.7 vs 0.6.0 (D2, one stream)', '27b_D2_p1_tf127', '27b_D2_p1_tf060', 'change'),
             ('PLAN_13 TensorFold 1.2.5 vs 0.6.0 (CL4), Part 10', 'p13_tf125', 'p13_tf060', 'change'),
             ('Flash Next UD-IQ3_XXS, MTP on vs off (text changes, model does not), Part 11', 'fn_U3X_MTP', 'fn_U3X', 'text-only change'),
             ('Re-run: 27B D2 0.6.0, PLAN_15 vs Part 5', '27b_D2_original', '27b_D2_original_part5_run', 'repeat'),
             ('Re-run: 27B D2 0.6.0, PLAN_17 (one stream) vs Part 5', '27b_D2_p1_tf060', '27b_D2_original_part5_run', 'repeat'),
             ('Re-run: 27B CL4 0.6.0, PLAN_15 vs Part 5', '27b_CL4_original', 'part5_CL4_run', 'repeat'),
             ('Re-run: 27B CL4 0.6.0, PLAN_13 vs Part 5, Part 10', 'p13_tf060', 'part5_CL4_run', 'repeat')]
    st = csvrows(D / 'same_text.csv')
    cross = []
    for label, a, b, kind in comps:
        if any(n in SIB and not SIB[n].exists() for n in (a, b)):
            cross.append({'comparison': label, 'skipped': 'sibling folder not present'})
            continue
        x, y = run(a), run(b)
        ids = sorted(x)
        ta, tb = totals(x), totals(y)
        diff = {t: x[t]['wall'] - y[t]['wall'] for t in ids}
        gap = sum(diff.values())
        top = sorted(ids, key=lambda t: diff[t], reverse=gap > 0)[:10]
        rest = [t for t in ids if t not in top]
        key = f'{a}__vs__{b}'
        row = {'comparison': label, 'kind': kind,
               'ratio_time_per_correct': round(ta['min_per_correct'] / tb['min_per_correct'], 3),
               'ratio_tokens': round(ta['output_tokens'] / tb['output_tokens'], 3),
               'ratio_speed': round(ta['tokens_per_s'] / tb['tokens_per_s'], 3),
               'ratio_correct': round(ta['correct'] / tb['correct'], 4),
               'tasks_A_faster': sum(diff[t] < 0 for t in ids),
               'median_task_time_ratio': round(statistics.median(x[t]['wall'] / y[t]['wall'] for t in ids), 3),
               'gap_min': round(gap / 60, 1), 'gap_from_top10_min': round(sum(diff[t] for t in top) / 60, 1),
               'ratio_without_top10': round(sum(x[t]['wall'] for t in rest) / sum(y[t]['wall'] for t in rest), 3),
               'tasks_tokens_within_10pct': sum(abs(x[t]['tokens'] / y[t]['tokens'] - 1) <= 0.10 for t in ids if y[t]['tokens']),
               'tasks_tokens_over_2x': sum(max(x[t]['tokens'], y[t]['tokens']) > 2 * min(x[t]['tokens'], y[t]['tokens']) for t in ids)}
        if st and key in st[0]:
            row['identical_text'] = sum(int(r[key]) for r in st)
        if kind == 'repeat':
            s = stats(a, b, max(LIMIT.get(a, 40), LIMIT.get(b, 40)))
            row['time_ratio_ci95'] = s['time_ci95']
        cross.append(row)
    R['cross_plan'] = cross
    R['PLAN_22_forward_all_850_ms'] = float(fw['all 850 groups']['ms_median'])
    R['PLAN_22_B_star_gb_s'] = max(float(r['gb_s_median']) for r in csvrows(D / 'bandwidth.csv')
                                   if r['study'] == 'PLAN_22' and float(r['size_gb']) >= 8)

    # ------------------------------------------------------------ print / check / write
    for s in summary:
        print(s)
    print(json.dumps(T, indent=1))
    for c in cross:
        print(c)
    files = {'summary.csv': summary, 'tests_preregistered.json': T, 'reported_only.json': R}
    norm = lambda o: json.loads(json.dumps(o))  # noqa: E731  (int keys -> str, tuples -> lists)
    if '--write' in sys.argv:
        with open(D / 'summary.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator='\n')
            w.writeheader()
            w.writerows(summary)
        (D / 'tests_preregistered.json').write_text(json.dumps(T, indent=1) + '\n')
        (D / 'reported_only.json').write_text(json.dumps(R, indent=1) + '\n')
    if '--check' in sys.argv:
        bad = 0
        old = list(csv.DictReader(open(D / 'summary.csv')))
        new = [{k: str(v) for k, v in s.items()} for s in summary]
        ok = old == new
        print(f"summary.csv: {'all %d rows match' % len(new) if ok else 'DIFFERS'}")
        bad += not ok
        for fn, obj in (('tests_preregistered.json', T), ('reported_only.json', R)):
            ok = json.load(open(D / fn)) == norm(obj)
            print(f"{ {'tests_preregistered.json': 'tests', 'reported_only.json': 'reported only'}[fn]}: {'all match' if ok else 'DIFFERS'}")
            bad += not ok
        sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
