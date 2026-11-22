"""Recompute every table of this part from the published files in data/ (no GPU, no task files).

    python3 code/tables.py           print the tables
    python3 code/tables.py --check   also compare with data/summary.csv and data/tests_preregistered.json

The locked tests are recomputed with code/stats.py (the same file as in Parts 4-10): exact McNemar, paired bootstrap
95% intervals with 10,000 resamples and seed 2026, a 120-minute cap per task. PLAN_18's token-ratio intervals use the
resampling of its own analysis (tasks in the order IQ3_S ran them, the `run_order` column), so they come out the same.
"""
import csv
import json
import math
import random
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / 'data'
P = D / 'per_task_400'
STATS = Path(__file__).resolve().parent / 'stats.py'
LIMIT_MIN = 120
CONFIGS = ['IQ3_S', 'UD-IQ4_XS', 'UD-IQ3_XXS', 'UD-IQ3_XXS_MTP', 'TensorFold_NVFP4', 'TensorFold_NVFP4_part6']


def rows(path):
    return list(csv.DictReader(open(path)))


def load(c):
    return {r['task_id']: r for r in rows(P / f'{c}.csv')}


T = {c: load(c) for c in CONFIGS}
IDS = sorted(T['IQ3_S'])
assert all(sorted(T[c]) == IDS for c in CONFIGS) and len(IDS) == 400


def num(x):
    return float(x) if x not in ('', None) else None


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def boot_ratio(a, b, ids, seed=2026, n=10000):
    """PLAN_18's interval for a ratio of totals (its analysis code, unchanged)."""
    rng = random.Random(seed)
    va, vb = [a[i] for i in ids], [b[i] for i in ids]
    res = []
    for _ in range(n):
        idx = [rng.randrange(len(ids)) for _ in ids]
        sb = sum(vb[i] for i in idx)
        if sb:
            res.append(sum(va[i] for i in idx) / sb)
    res.sort()
    return [round(res[int(0.025 * len(res))], 3), round(res[int(0.975 * len(res)) - 1], 3)]


def stats_py(a, b):
    out = subprocess.run([sys.executable, str(STATS), '--limit-min', str(LIMIT_MIN), f'{a}={P / (a + ".csv")}',
                          f'{b}={P / (b + ".csv")}'], capture_output=True, text=True, check=True).stdout
    st = json.loads(out)
    return st['totals'], st['pairs'][f'{a} vs {b}']


def summary():
    out = {}
    for c in CONFIGS:
        t = T[c]
        corr = sum(int(t[i]['passed']) for i in IDS)
        tot = sum(min(float(t[i]['wall_s']), LIMIT_MIN * 60) for i in IDS) / 60
        tok = sum(int(t[i]['output_tokens']) for i in IDS)
        dec = [(int(t[i]['output_tokens']), num(t[i]['decode_tok_s'])) for i in IDS if num(t[i]['decode_tok_s'])]
        ft = [num(t[i]['first_token_s']) for i in IDS if num(t[i]['first_token_s']) is not None]
        dr = [(int(t[i]['draft_tokens']), int(t[i]['draft_accepted'])) for i in IDS if t[i]['draft_tokens']]
        out[c] = {'correct': corr, 'total_min': round(tot, 1), 'min_per_correct': round(tot / corr, 3), 'output_tokens': tok,
                  'hit_output_limit': sum(t[i]['finish'] == 'length' for i in IDS),
                  'loop_detected': sum(int(t[i]['loop_detected']) for i in IDS),
                  'over_40_min': sum(float(t[i]['wall_s']) > 2400 for i in IDS),
                  'timed_out': sum(int(t[i]['timed_out']) for i in IDS),
                  'decode_tok_s': round(sum(n for n, _ in dec) / sum(n / v for n, v in dec), 2) if dec else None,
                  'first_token_median_s': round(statistics.median(ft), 2) if ft else None,
                  'draft_acceptance': round(sum(a for _, a in dr) / sum(n for n, _ in dr), 4) if dr else None}
        for g in ('coding_python', 'coding_cpp', 'reasoning'):
            out[c][f'correct_{g}'] = sum(int(t[i]['passed']) for i in IDS if t[i]['group'] == g)
    return out


def plan18():
    pairs = [('IQ3_S', 'UD-IQ4_XS'), ('IQ3_S', 'UD-IQ3_XXS'), ('UD-IQ4_XS', 'UD-IQ3_XXS')]
    q1 = {}
    for a, b in pairs:
        q1[f'{a} vs {b}'] = stats_py(a, b)[1]
    adj = holm([q1[f'{a} vs {b}']['p'] for a, b in pairs[:2]])
    for (a, b), p in zip(pairs[:2], adj):
        q = q1[f'{a} vs {b}']
        q['p_holm_primary'] = round(p, 4)
        q['decision'] = ('no detectable accuracy difference' if p >= 0.05 else
                         'IQ3_S more accurate (detectable)' if q['acc_diff_pp'] > 0 else 'IQ3_S less accurate (detectable)')
    order = sorted(IDS, key=lambda i: int(T['IQ3_S'][i]['run_order']))
    q2 = {}
    for a, b in pairs:
        ta = {i: int(T[a][i]['output_tokens']) for i in IDS}
        tb = {i: int(T[b][i]['output_tokens']) for i in IDS}
        la = {i: T[a][i]['loop_detected'] == '1' for i in IDS}
        lb = {i: T[b][i]['loop_detected'] == '1' for i in IDS}
        oa, ob = sum(la[i] and not lb[i] for i in IDS), sum(lb[i] and not la[i] for i in IDS)
        q2[f'{a} vs {b}'] = {'tokens_ratio_total': round(sum(ta.values()) / sum(tb.values()), 3),
                             'tokens_ratio_ci95': boot_ratio(ta, tb, order),
                             'tokens_ratio_median': round(statistics.median(ta[i] / tb[i] for i in IDS if ta[i] and tb[i]), 3),
                             'loops_only_first': oa, 'loops_only_second': ob, 'loops_p': round(mcnemar(oa, ob), 4)}
    tc = rows(D / 'tool_calls.csv')
    counts = {}
    for r in tc:
        counts.setdefault(r['config'], {k: 0 for k in ('valid', 'malformed', 'text', 'cut', 'error')})[r['label']] += 1
    lab = {(r['config'], r['variant'], r['turn']): r['label'] for r in tc}
    q3 = {}
    for a, b in pairs:
        keys = sorted({(v, t) for c, v, t in lab if c == a} & {(v, t) for c, v, t in lab if c == b})
        ma = {k: lab[(a, *k)] == 'malformed' for k in keys}
        mb = {k: lab[(b, *k)] == 'malformed' for k in keys}
        oa, ob = sum(ma[k] and not mb[k] for k in keys), sum(mb[k] and not ma[k] for k in keys)
        q3[f'{a} vs {b}'] = {'paired': len(keys), 'malformed_only_first': oa, 'malformed_only_second': ob,
                             'p': round(mcnemar(oa, ob), 4)}
    return q1, q2, counts, q3


def plan18c(summ):
    q1 = stats_py('UD-IQ3_XXS_MTP', 'TensorFold_NVFP4')[1]
    lo, hi = q1['time_ci95']
    q1['decision'] = ('GGUF + MTP faster (detectable)' if hi < 1 else 'TensorFold faster (detectable)' if lo > 1
                      else 'no detectable time difference')
    q2 = stats_py('UD-IQ3_XXS_MTP', 'UD-IQ3_XXS')[1]
    q4 = stats_py('TensorFold_NVFP4', 'TensorFold_NVFP4_part6')[1]
    same = {r: rows(D / 'same_text.csv') for r in [0]}[0]
    for q, key in ((q1, 'UD-IQ3_XXS_MTP__vs__TensorFold_NVFP4'), (q2, 'UD-IQ3_XXS_MTP__vs__UD-IQ3_XXS'),
                   (q4, 'TensorFold_NVFP4__vs__TensorFold_NVFP4_part6')):
        q['identical_text'] = sum(int(r[key]) for r in same)
    ps = {(r['session'], r['config']): r for r in rows(D / 'probe' / 'probe_summary.csv')}
    q3 = {}
    for c in ('IQ3_S', 'UD-IQ3_XXS', 'UD-IQ4_XS'):
        off, on = ps[('p3', f'{c}_off')], ps[('p3', f'{c}_mtp4')]
        v0, v1 = float(off['decode_tok_s']), float(on['decode_tok_s'])
        sp = round(v1 / v0, 3)
        q3[c] = {'decode_off': v0, 'decode_mtp': v1, 'speedup': sp, 'acceptance_rate': float(on['draft_acceptance']),
                 'identical_text': int(on['probe_same_text_as_mtp_off_of_12']),
                 'p18_min_per_correct': summ[c]['min_per_correct'],
                 'estimated_min_per_correct_mtp': round(summ[c]['min_per_correct'] / sp, 3)}
    q3['UD-IQ3_XXS']['measured_min_per_correct_U3M'] = summ['UD-IQ3_XXS_MTP']['min_per_correct']
    return q1, q2, q3, q4


def post_hoc():
    """Looked at after the results (report only): where the time difference of PLAN_18c's Q1 comes from."""
    a, b = T['UD-IQ3_XXS_MTP'], T['TensorFold_NVFP4']
    tu = {i: int(a[i]['output_tokens']) for i in IDS}
    tr = {i: int(b[i]['output_tokens']) for i in IDS}
    wu = {i: float(a[i]['wall_s']) for i in IDS}
    wr = {i: float(b[i]['wall_s']) for i in IDS}
    top_tok = sorted(IDS, key=lambda i: tr[i] - tu[i], reverse=True)[:10]
    top_wall = sorted((wr[i] - wu[i] for i in IDS), reverse=True)[:10]
    u3x = {i: int(T['UD-IQ3_XXS'][i]['output_tokens']) for i in IDS}
    long_ = {c: {i for i in IDS if int(T[c][i]['output_tokens']) > 20000} for c in CONFIGS}
    return {'tokens_ratio_total': round(sum(tu.values()) / sum(tr.values()), 3),
            'tokens_per_second_overall': {'UD-IQ3_XXS_MTP': round(sum(tu.values()) / sum(wu.values()), 1),
                                          'TensorFold_NVFP4': round(sum(tr.values()) / sum(wr.values()), 1)},
            'token_gap': sum(tr.values()) - sum(tu.values()), 'token_gap_top10_tasks': sum(tr[i] - tu[i] for i in top_tok),
            'minutes_gap': round((sum(wr.values()) - sum(wu.values())) / 60, 1),
            'minutes_gap_top10_tasks': round(sum(top_wall) / 60, 1),
            'tasks_faster_with_UD-IQ3_XXS_MTP': sum(wu[i] < wr[i] for i in IDS),
            'median_per_task_time_ratio': round(statistics.median(wu[i] / wr[i] for i in IDS), 3),
            'top10_tasks_short_in_UD-IQ3_XXS_without_MTP_too': sum(u3x[i] / tr[i] <= 0.25 for i in top_tok),
            'replies_over_20000_tokens': {c: len(long_[c]) for c in CONFIGS},
            'over_20000_in_both_UD-IQ3_XXS_runs': len(long_['UD-IQ3_XXS_MTP'] & long_['UD-IQ3_XXS'])}


def same(a, b, keys):
    """True if every listed key agrees (floats to 1e-9; lists element-wise)."""
    bad = []
    for k in keys:
        x, y = a.get(k), b.get(k)
        if isinstance(x, float) or isinstance(y, float):
            ok = x is not None and y is not None and abs(x - y) < 1e-9
        else:
            ok = x == y
        if not ok:
            bad.append((k, x, y))
    return bad


def main():
    check = '--check' in sys.argv
    summ = summary()
    q1, q2, counts, q3 = plan18()
    c1, c2, c3, c4 = plan18c(summ)
    ph = post_hoc()
    print('== Summary (400 tasks, D2)')
    for c in CONFIGS:
        print(c, json.dumps(summ[c]))
    print('\n== PLAN_18 Q1 (primary: the first two pairs, Holm over 2)')
    for k, v in q1.items():
        print(k, json.dumps({x: v.get(x) for x in ('only_first', 'only_second', 'p', 'p_holm_primary', 'acc_diff_pp',
                                                   'acc_ci95', 'time_ratio', 'time_ci95', 'decision')}))
    print('\n== PLAN_18 Q2 (secondary): output length and loops')
    for k, v in q2.items():
        print(k, json.dumps(v))
    print('\n== PLAN_18 Q3 (secondary): tool calls, 96 replies per config')
    print(json.dumps(counts))
    for k, v in q3.items():
        print(k, json.dumps(v))
    print('\n== PLAN_18c Q1 (primary): UD-IQ3_XXS + MTP vs TensorFold')
    print(json.dumps({x: c1.get(x) for x in ('only_first', 'only_second', 'p', 'acc_diff_pp', 'acc_ci95', 'time_ratio',
                                             'time_ci95', 'identical_text', 'decision')}))
    print('\n== PLAN_18c Q2 (MTP on vs off; the time ratio is report only)')
    print(json.dumps({x: c2.get(x) for x in ('only_first', 'only_second', 'p', 'acc_diff_pp', 'acc_ci95', 'time_ratio',
                                             'time_ci95', 'identical_text')}))
    print('\n== PLAN_18c Q3 (report only): MTP speed-up on the 12 probe tasks, estimates')
    for k, v in c3.items():
        print(k, json.dumps(v))
    print('\n== PLAN_18c Q4 (report only): TensorFold now vs Part 6')
    print(json.dumps({x: c4.get(x) for x in ('only_first', 'only_second', 'p', 'time_ratio', 'time_ci95', 'identical_text')}))
    print('\n== Looked at after the results (report only)')
    print(json.dumps(ph, indent=1))
    if not check:
        return 0
    bad = []
    keys = ['correct', 'correct_coding_python', 'correct_coding_cpp', 'correct_reasoning', 'total_min', 'min_per_correct',
            'output_tokens', 'hit_output_limit', 'loop_detected', 'over_40_min', 'timed_out', 'draft_acceptance']
    for r in rows(D / 'summary.csv'):
        mine = summ[r['config']]
        for k in keys:
            want = None if r[k] == '' else float(r[k])
            got = mine[k]
            if (want is None) != (got is None) or (want is not None and abs(want - got) > 1e-6):
                bad.append(('summary', r['config'], k, want, got))
        for k in ('decode_tok_s', 'first_token_median_s'):      # per-task values are rounded to 3 decimals
            if r[k] and abs(float(r[k]) - mine[k]) > 0.011:
                bad.append(('summary', r['config'], k, r[k], mine[k]))
    print(f"\nsummary.csv: {'all 6 rows match' if not bad else 'MISMATCH'}")
    tests = json.loads((D / 'tests_preregistered.json').read_text())
    t18, t18c = tests['PLAN_18'], tests['PLAN_18c']
    k1 = ['only_first', 'only_second', 'p', 'p_holm_primary', 'acc_diff_pp', 'acc_ci95', 'time_ratio', 'time_ci95', 'decision']
    nb = len(bad)
    for k, v in t18['Q1_primary_holm_over_2'].items():
        bad += [('PLAN_18 Q1', k) + x for x in same(q1[k], v, k1)]
    for k, v in t18['Q1_report_only'].items():
        bad += [('PLAN_18 Q1', k) + x for x in same(q1[k], v, k1[:3] + k1[4:8])]
    for k, v in t18['Q2_secondary_output_length_and_loops'].items():
        bad += [('PLAN_18 Q2', k) + x for x in same(q2[k], v, list(v))]
    for k, v in t18['Q3_secondary_tool_calls']['counts'].items():
        bad += [('PLAN_18 Q3', k) + x for x in same(counts[k], v, list(v))]
    for k, v in t18['Q3_secondary_tool_calls']['malformed_pairs'].items():
        bad += [('PLAN_18 Q3', k) + x for x in same(q3[k], v, list(v))]
    kc = ['only_first', 'only_second', 'p', 'acc_diff_pp', 'acc_ci95', 'time_ratio', 'time_ci95', 'identical_text']
    bad += [('PLAN_18c Q1',) + x for x in same(c1, t18c['Q1_primary']['UD-IQ3_XXS_MTP vs TensorFold_NVFP4'], kc + ['decision'])]
    bad += [('PLAN_18c Q2',) + x for x in same(c2, t18c['Q2_report_time_only']['UD-IQ3_XXS_MTP vs UD-IQ3_XXS'], kc)]
    bad += [('PLAN_18c Q4',) + x for x in same(c4, t18c['Q4_report_only']['TensorFold_NVFP4 vs TensorFold_NVFP4_part6'], kc)]
    for k, v in t18c['Q3_report_only'].items():
        mine = dict(c3[k], acceptance=v['acceptance'])
        bad += [('PLAN_18c Q3', k) + x for x in same(mine, v, [x for x in v if x != 'acceptance'])]
        if abs(v['acceptance']['rate'] - c3[k]['acceptance_rate']) > 1e-9:
            bad.append(('PLAN_18c Q3', k, 'acceptance', v['acceptance']['rate'], c3[k]['acceptance_rate']))
    print(f"tests: {'all match' if len(bad) == nb else 'MISMATCH'}")
    for b in bad:
        print('  ', b)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
