"""Recompute every number of Part 12 from the published per-task files (no GPU, no task files).

    python3 code/tables.py           print the tables
    python3 code/tables.py --write   write data/summary.csv and data/reported_only.json
    python3 code/tables.py --check   recompute; compare with data/summary.csv, data/reported_only.json and
                                     data/tests_preregistered.json (the locked analyses' own output)

Locked tests (PLAN_19, PLAN_20, PLAN_21b) are recomputed exactly as the locked analyses did: stats.py for paired
comparisons (10,000 resamples, seed 2026, 40-minute limit), and for PLAN_19 Q2 an independent bootstrap of each task set
in the order the locked analysis read them (column analysis_order of the set T files). PLAN_21b's report-only
comparisons were written in its plan; everything else in reported_only.json was looked at after the results.
"""
import csv
import json
import random
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'data'
sys.path.insert(0, str(HERE))
import shared_time as ST  # noqa: E402

LIMIT = 2400
MODELS = ('27b', 'flashnext')
N21 = (2, 4, 8)          # PLAN_21b: the 27B at 2, 4 and 8 streams, interleaved
NAME = {'27b': '27B', 'flashnext': 'Flash Next'}
INT = ('topic_similar', 'block', 'send_order', 'worker', 'passed', 'output_tokens', 'timed_out', 'analysis_order')
FLOAT = ('start_s', 'end_s', 'wall_s')


def load(name):
    rows = list(csv.DictReader(open(DATA / 'per_task_400' / f'{name}.csv')))
    for r in rows:
        for k in INT:
            if k in r:
                r[k] = int(r[k])
        for k in FLOAT:
            if k in r:
                r[k] = float(r[k])
    return {r['task_id']: r for r in rows}


def stats(arms):
    """arms: {name: {task: (passed, seconds)}} -> stats.py output."""
    with tempfile.TemporaryDirectory() as d:
        args = []
        for n, v in arms.items():
            p = Path(d) / f'{n}.csv'
            p.write_text('task_id,passed,wall_s\n' + ''.join(f'{t},{int(bool(a))},{min(s, LIMIT):.3f}\n'
                                                             for t, (a, s) in sorted(v.items())))
            args.append(f'{n}={p}')
        out = subprocess.run([sys.executable, str(HERE / 'stats.py'), '--limit-min', str(LIMIT // 60), *args],
                             capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def boot_diff(a, b, seed=2026, n=10000):
    """95% interval of mean(b) - mean(a), resampling a and b independently (two different task sets)."""
    rng = random.Random(seed)
    res = []
    for _ in range(n):
        ma = sum(a[rng.randrange(len(a))] for _ in a) / len(a)
        mb = sum(b[rng.randrange(len(b))] for _ in b) / len(b)
        res.append(mb - ma)
    res.sort()
    return [round(100 * res[int(0.025 * n)], 2), round(100 * res[int(0.975 * n) - 1], 2)]


def boot_gap_change(aq, af, oq, of, seed=2026, n=10000):
    """95% interval of (Flash Next - 27B on set O) - (the same on set T), resampling tasks within each set."""
    rng = random.Random(seed)
    A, O = list(zip(aq, af)), list(zip(oq, of))
    res = []
    for _ in range(n):
        sa = [A[rng.randrange(len(A))] for _ in A]
        so = [O[rng.randrange(len(O))] for _ in O]
        res.append(sum(f - q for q, f in so) / len(so) - sum(f - q for q, f in sa) / len(sa))
    res.sort()
    return [round(100 * res[int(0.025 * n)], 2), round(100 * res[int(0.975 * n) - 1], 2)]


def reading(lo, hi, faster, slower, none='no detectable time difference'):
    return faster if hi < 1 else slower if lo > 1 else none


def per_run(R, n):
    """One multi-stream run -> (shared seconds per task, per-block facts, always-full rows, {tasks running: seconds})."""
    S, blk, full, prof = {}, {}, [], {}
    for g in range(8):
        rows = [r for r in R.values() if r['block'] == g]
        sh, busy, under, fs, ft = ST.shares(rows, n)
        S.update(sh)
        blk[g] = {'busy_s': busy, 'under_s': under, 'full_s': fs, 'full_tok': ft, 'correct': sum(r['passed'] for r in rows)}
        full += ST.always_full(rows, n)
        for k, v in ST.running_profile(rows).items():
            prof[k] = prof.get(k, 0.0) + v
    return S, blk, full, prof


def compute():
    one = {m: load(f'{m}_1stream') for m in MODELS}
    four = {m: load(f'{m}_4streams') for m in MODELS}
    setT = {m: load(f'setT_{m}_1stream') for m in MODELS}
    p21 = {n: load(f'27b_plan21b_{n}streams') for n in N21}
    ids = sorted(one['27b'])
    assert len(ids) == 400 and all(sorted(x) == ids for x in (*one.values(), *four.values(), *p21.values()))

    # shared time, block by block
    S4, blk, prof, full = {}, {}, {}, {}
    for m in MODELS:
        S4[m], blk[m], prof[m], full[m] = {}, {}, {}, []
        for g in range(8):
            rows = [r for r in four[m].values() if r['block'] == g]
            sh, busy, under, fs, ft = ST.shares(rows)
            S4[m].update(sh)
            blk[m][g] = {'busy_s': busy, 'under_s': under, 'full_s': fs, 'full_tok': ft,
                         'wall_1stream_s': sum(one[m][r['task_id']]['wall_s'] for r in rows),
                         'correct': sum(r['passed'] for r in rows)}
            for k, v in ST.running_profile(rows).items():
                prof[m][k] = prof[m].get(k, 0.0) + v
            full[m] += ST.always_full(rows)
    S1 = {m: {t: min(one[m][t]['wall_s'], LIMIT) for t in ids} for m in MODELS}

    # ---------- locked tests ----------
    T = {}
    st = stats({'FN': {t: (one['flashnext'][t]['passed'], S1['flashnext'][t]) for t in ids},
                'Q27': {t: (one['27b'][t]['passed'], S1['27b'][t]) for t in ids}})
    p = st['pairs']['FN vs Q27']
    K = ('only_first', 'only_second', 'p', 'acc_diff_pp', 'acc_ci95', 'time_ratio', 'time_ci95')
    T19 = {'Q1 (primary): Flash Next vs 27B, one stream': {
        'totals': {'flashnext': st['totals']['FN'], '27b': st['totals']['Q27']},
        'pair': {k: p[k] for k in K},
        'accuracy_reading': ('no detectable accuracy difference' if p['p'] >= 0.05 else
                             'Flash Next more accurate (detectable)' if p['acc_diff_pp'] > 0 else '27B more accurate (detectable)'),
        'time_reading': reading(*p['time_ci95'], 'Flash Next faster (detectable)', 'Flash Next slower (detectable)')}}
    q2 = {}
    for m in MODELS:
        a_rows = sorted(setT[m].values(), key=lambda r: r['analysis_order'])
        row = {}
        for g in ('all', 'py', 'cpp', 'reason'):
            av = [r['passed'] for r in a_rows if g == 'all' or r['stratum'] == g]
            ov = [one[m][t]['passed'] for t in ids if g == 'all' or
                  {'py': 'py', 'cpp': 'cpp', 'json': 'reason'}[one[m][t]['language']] == g]
            row[g] = {'setT': f'{sum(av)}/{len(av)}', 'setO': f'{sum(ov)}/{len(ov)}',
                      'setT_pct': round(100 * sum(av) / len(av), 2), 'setO_pct': round(100 * sum(ov) / len(ov), 2),
                      'setO_minus_setT_pp': round(100 * (sum(ov) / len(ov) - sum(av) / len(av)), 2),
                      'ci95': boot_diff(av, ov)}
        q2[m] = row
    both = [q2[m]['all']['ci95'] for m in MODELS]
    q2['reading'] = ('set O harder for both models (detectable)' if all(h < 0 for _, h in both) else
                     'set T harder for both models (detectable)' if all(lo > 0 for lo, _ in both) else
                     'no consistent detectable difference')
    tid = sorted(setT['27b'])
    q2['model_gap_report_only'] = {
        'setT_flashnext_minus_27b_pp': round(100 * (sum(setT['flashnext'][t]['passed'] for t in tid) -
                                                    sum(setT['27b'][t]['passed'] for t in tid)) / len(tid), 2),
        'setO_flashnext_minus_27b_pp': round(100 * (sum(one['flashnext'][t]['passed'] for t in ids) -
                                                    sum(one['27b'][t]['passed'] for t in ids)) / len(ids), 2),
        'change_ci95': boot_gap_change([setT['27b'][t]['passed'] for t in tid], [setT['flashnext'][t]['passed'] for t in tid],
                                       [one['27b'][t]['passed'] for t in ids], [one['flashnext'][t]['passed'] for t in ids]),
        'note': 'report only'}
    q2['time_report_only'] = {m: {
        'setT_min_per_correct': round(sum(r['wall_s'] for r in setT[m].values()) / 60 /
                                      sum(r['passed'] for r in setT[m].values()), 3),
        'setO_min_per_correct': T19['Q1 (primary): Flash Next vs 27B, one stream']['totals'][m]['min_per_correct'],
        'setT_tokens_median': statistics.median(r['output_tokens'] for r in setT[m].values()),
        'setO_tokens_median': statistics.median(one[m][t]['output_tokens'] for t in ids)} for m in MODELS}
    T19['Q2 (secondary): accuracy on set O minus set T, sets resampled independently'] = q2
    T['PLAN_19'] = T19

    T20 = {}
    st = stats({'FN4': {t: (four['flashnext'][t]['passed'], S4['flashnext'][t]) for t in ids},
                'Q27_4': {t: (four['27b'][t]['passed'], S4['27b'][t]) for t in ids}})
    p = st['pairs']['FN4 vs Q27_4']
    T20['Q1 (primary): Flash Next vs 27B, four streams, shared time'] = {
        'totals': {'flashnext': st['totals']['FN4'], '27b': st['totals']['Q27_4']}, 'pair': {k: p[k] for k in K},
        'time_reading': reading(*p['time_ci95'], 'Flash Next faster at 4 streams (detectable)',
                                'Flash Next slower at 4 streams (detectable)')}
    for m in MODELS:
        st = stats({'p4': {t: (four[m][t]['passed'], S4[m][t]) for t in ids},
                    'p1': {t: (one[m][t]['passed'], S1[m][t]) for t in ids}})
        p = st['pairs']['p4 vs p1']
        T20[f'Q2 (primary): {m}, four streams vs one stream'] = {
            'totals': {'4streams': st['totals']['p4'], '1stream': st['totals']['p1']},
            'pair': {**{k: p[k] for k in K}, 'speedup': round(1 / p['time_ratio'], 3),
                     'speedup_ci95': [round(1 / p['time_ci95'][1], 3), round(1 / p['time_ci95'][0], 3)]},
            'time_reading': reading(*p['time_ci95'], 'concurrency raises throughput (detectable)',
                                    'concurrency is slower (detectable)')}
    T['PLAN_20'] = T20

    # PLAN_21b: shared time per run, then its locked test (8 vs 4 streams, both this run)
    S21, blk21, full21, prof21 = {}, {}, {}, {}
    for n in N21:
        S21[n], blk21[n], full21[n], prof21[n] = per_run(p21[n], n)
    arm21 = {f'Q27_p{n}': {t: (p21[n][t]['passed'], S21[n][t]) for t in ids} for n in N21}
    arm21['Q27_p1_plan19'] = {t: (one['27b'][t]['passed'], S1['27b'][t]) for t in ids}
    arm21['Q27_p4_plan20'] = {t: (four['27b'][t]['passed'], S4['27b'][t]) for t in ids}

    def cmp21(a, b):
        st = stats({a: arm21[a], b: arm21[b]})
        p = st['pairs'][f'{a} vs {b}']
        return st['totals'], {**{k: p[k] for k in K}, 'speedup': round(1 / p['time_ratio'], 3),
                              'speedup_ci95': [round(1 / p['time_ci95'][1], 3), round(1 / p['time_ci95'][0], 3)]}
    tot, p = cmp21('Q27_p8', 'Q27_p4')
    T['PLAN_21b'] = {'Q1 (primary): 27B, 8 streams vs 4 streams, both this run, shared time': {
        'totals': {'8streams': tot['Q27_p8'], '4streams': tot['Q27_p4']}, 'pair': p,
        'time_reading': reading(*p['time_ci95'], '8 streams raise throughput over 4 (detectable)',
                                '8 streams slower than 4 (detectable)')}}

    # ---------- summary, one row per run ----------
    def p90(xs):
        return statistics.quantiles(xs, n=10, method='inclusive')[-1]

    runs = [('PLAN_19' if k == 1 else 'PLAN_20', m, k, (one if k == 1 else four)[m], (S1 if k == 1 else S4)[m],
             blk[m] if k == 4 else None, full[m] if k == 4 else None) for m in MODELS for k in (1, 4)]
    runs += [('PLAN_21b', '27b', n, p21[n], S21[n], blk21[n], full21[n]) for n in N21]
    summary = []
    for study, m, streams, R, S, B, F in runs:
        corr = sum(R[t]['passed'] for t in ids)
        toks = sum(R[t]['output_tokens'] for t in ids)
        busy = sum(S.values())
        lat = [R[t]['wall_s'] for t in ids]
        sts = {}
        for t in ids:
            sts[R[t]['status']] = sts.get(R[t]['status'], 0) + 1
        row = {'study': study, 'model': m, 'streams': streams, 'correct': corr, 'minutes': round(busy / 60, 1),
               'min_per_correct': round(busy / 60 / corr, 3), 'correct_per_hour': round(corr / (busy / 3600), 1),
               'output_tokens': toks, 'statuses': '; '.join(f'{k} {v}' for k, v in sorted(sts.items())),
               'hit_limit': sum(R[t]['finish'] == 'length' for t in ids), 'timed_out': sum(R[t]['timed_out'] for t in ids),
               'latency_median_s': round(statistics.median(lat), 1), 'latency_p90_s': round(p90(lat), 1),
               'latency_max_s': round(max(lat), 1),
               'per_stream_tok_s_median': round(statistics.median(R[t]['output_tokens'] / R[t]['wall_s']
                                                                  for t in ids if R[t]['wall_s'] > 5), 1),
               'total_tok_s': round(toks / busy, 1)}
        if streams > 1:
            fs = sum(b['full_s'] for b in B.values())
            row.update(steady_tok_s_spread=round(sum(b['full_tok'] for b in B.values()) / fs, 1),
                       steady_tok_s_full_tasks=round(streams * sum(r['output_tokens'] for r in F) /
                                                     sum(r['wall_s'] for r in F), 1),
                       full_tasks=len(F),
                       under_n_share=round(sum(b['under_s'] for b in B.values()) / busy, 3))
        else:
            row.update(steady_tok_s_spread='', steady_tok_s_full_tasks='', full_tasks='', under_n_share='')
        summary.append(row)

    # ---------- reported only ----------
    rep = {}
    o1 = {r['model']: r for r in summary if r['study'] == 'PLAN_19'}
    o4 = {r['model']: r for r in summary if r['study'] == 'PLAN_20'}
    o21 = {r['streams']: r for r in summary if r['study'] == 'PLAN_21b'}
    q, f = one['27b'], one['flashnext']
    tq, tf = sum(S1['27b'].values()), sum(S1['flashnext'].values())
    extra = sorted(ids, key=lambda t: -(S1['flashnext'][t] - S1['27b'][t]))
    gap = tf - tq

    def without(k, S):
        keep = [t for t in ids if t not in sorted(ids, key=lambda t: -(S['flashnext'][t] - S['27b'][t]))[:k]]
        a = sum(S['flashnext'][t] for t in keep) / sum(four['flashnext'][t]['passed'] for t in keep)
        b = sum(S['27b'][t] for t in keep) / sum(four['27b'][t]['passed'] for t in keep)
        return round(a / b, 3)

    def without1(k):
        keep = [t for t in ids if t not in extra[:k]]
        a = sum(S1['flashnext'][t] for t in keep) / sum(f[t]['passed'] for t in keep)
        b = sum(S1['27b'][t] for t in keep) / sum(q[t]['passed'] for t in keep)
        return round(a / b, 3)

    rep['PLAN_19 one stream, Flash Next vs 27B'] = {
        'token_ratio': round(o1['flashnext']['output_tokens'] / o1['27b']['output_tokens'], 3),
        'speed_ratio': round(o1['flashnext']['total_tok_s'] / o1['27b']['total_tok_s'], 3),
        'correct_ratio': round(o1['flashnext']['correct'] / o1['27b']['correct'], 3),
        'flashnext_faster_on_tasks': sum(S1['flashnext'][t] < S1['27b'][t] for t in ids),
        'median_per_task_time_ratio': round(statistics.median(S1['flashnext'][t] / S1['27b'][t] for t in ids), 3),
        'share_of_extra_time_top1_top5_top10': [round(sum(S1['flashnext'][t] - S1['27b'][t] for t in extra[:k]) / gap, 3)
                                                for k in (1, 5, 10)],
        'time_ratio_without_top5_top10': [without1(5), without1(10)],
        'longest_two': [{'task_id': t, 'flashnext_min': round(f[t]['wall_s'] / 60, 1), 'flashnext_tokens': f[t]['output_tokens'],
                         'flashnext_status': f[t]['status'], '27b_min': round(q[t]['wall_s'] / 60, 1)} for t in extra[:2]],
        'both_wrong': [t for t in ids if not q[t]['passed'] and not f[t]['passed']]}
    by = {}
    for t in ids:
        k = f"{q[t]['kind']} / {q[t]['language']}"
        b = by.setdefault(k, {'tasks': 0, **{f'{m}_{x}': [] for m in MODELS for x in ('correct', 'tokens', 'seconds')}})
        b['tasks'] += 1
        for m in MODELS:
            b[f'{m}_correct'].append(one[m][t]['passed'])
            b[f'{m}_tokens'].append(one[m][t]['output_tokens'])
            b[f'{m}_seconds'].append(S1[m][t])
    rep['PLAN_19 by kind and language, one stream'] = {
        k: {'tasks': b['tasks'], **{f'{m}_correct': sum(b[f'{m}_correct']) for m in MODELS},
            **{f'{m}_tokens_median': statistics.median(b[f'{m}_tokens']) for m in MODELS},
            **{f'{m}_minutes': round(sum(b[f'{m}_seconds']) / 60, 1) for m in MODELS}} for k, b in by.items()}
    keep = [t for t in ids if not q[t]['topic_similar']]
    rep['PLAN_19 without the topic-similar tasks'] = {
        'excluded': 400 - len(keep), **{m: f"{sum(one[m][t]['passed'] for t in keep)}/{len(keep)}" for m in MODELS},
        'excluded_correct': {m: sum(one[m][t]['passed'] for t in ids if q[t]['topic_similar']) for m in MODELS}}

    top = sorted(ids, key=lambda t: -(S4['flashnext'][t] - S4['27b'][t]))
    gap4 = sum(S4['flashnext'][t] - S4['27b'][t] for t in ids)
    rep['PLAN_20 four streams'] = {
        'same_text_as_one_stream': {m: sum(int(r[f'{m}_1stream__vs__{m}_4streams']) for r in csv.DictReader(open(DATA / 'same_text.csv')))
                                    for m in MODELS},
        'latency_ratio_median_per_task': {m: round(statistics.median(four[m][t]['wall_s'] / one[m][t]['wall_s'] for t in ids), 3)
                                          for m in MODELS},
        'per_stream_speed_vs_one_stream': {m: round(o4[m]['per_stream_tok_s_median'] / o1[m]['per_stream_tok_s_median'], 3)
                                           for m in MODELS},
        'total_speed_gain': {m: round(o4[m]['total_tok_s'] / o1[m]['total_tok_s'], 2) for m in MODELS},
        'steady_speed_gain_spread': {m: round(o4[m]['steady_tok_s_spread'] / o1[m]['total_tok_s'], 2) for m in MODELS},
        'steady_speed_gain_full_tasks': {m: round(o4[m]['steady_tok_s_full_tasks'] / o1[m]['total_tok_s'], 2) for m in MODELS},
        'flashnext_over_27b_without_top5_top10': [without(5, S4), without(10, S4)],
        'top5_share_of_flashnext_extra_time': round(sum(S4['flashnext'][t] - S4['27b'][t] for t in top[:5]) / gap4, 3),
        'top5_tasks': [{'task_id': t, 'flashnext_shared_min': round(S4['flashnext'][t] / 60, 1),
                        'flashnext_tokens': four['flashnext'][t]['output_tokens'], 'flashnext_status': four['flashnext'][t]['status']}
                       for t in top[:5]],
        'minutes_by_tasks_running': {m: {str(k): round(v / 60, 1) for k, v in sorted(prof[m].items())} for m in MODELS}}
    rep['PLAN_20 per block (no time sharing needed)'] = {
        m: [{'block': g, 'busy_min_4streams': round(b['busy_s'] / 60, 1), 'sum_wall_min_1stream': round(b['wall_1stream_s'] / 60, 1),
             'speedup': round(b['wall_1stream_s'] / b['busy_s'], 2), 'under_4_share': round(b['under_s'] / b['busy_s'], 3),
             'correct': b['correct']} for g, b in blk[m].items()] for m in MODELS}
    rep['PLAN_20 per block (no time sharing needed)']['flashnext_over_27b_busy'] = [
        round(blk['flashnext'][g]['busy_s'] / blk['27b'][g]['busy_s'], 2) for g in range(8)]
    # two ways to use one Spark: a batch of 50 tasks, waited for (measured), or a server that never runs dry (estimated)
    sc = {}
    for m in MODELS:
        steady = o4[m]['steady_tok_s_spread']
        cont = o4[m]['output_tokens'] / steady / 60 / o4[m]['correct']
        sc[m] = {'one_stream_min_per_correct': o1[m]['min_per_correct'], 'batch_of_50_min_per_correct': o4[m]['min_per_correct'],
                 'never_dry_min_per_correct_estimate': round(cont, 3),
                 'correct_per_hour': {'one_stream': o1[m]['correct_per_hour'], 'batch_of_50': o4[m]['correct_per_hour'],
                                      'never_dry_estimate': round(60 / cont, 1)}}
    sc['flashnext_over_27b'] = {k: round(sc['flashnext'][k] / sc['27b'][k], 2) for k in
                                ('one_stream_min_per_correct', 'batch_of_50_min_per_correct', 'never_dry_min_per_correct_estimate')}
    sc['note'] = ('never_dry = all output tokens / steady speed (spread estimate) / correct answers; not measured, '
                  'it assumes a queue that always has four tasks waiting')
    rep['Two ways to use the server'] = sc

    # PLAN_21b: the comparisons its plan lists as report only, then the per-run numbers
    same = list(csv.DictReader(open(DATA / 'same_text.csv')))
    r21 = {'2_vs_4': dict(zip(('totals', 'pair'), cmp21('Q27_p2', 'Q27_p4'))),
           'vs_one_stream_PLAN_19': {f'{n}streams': cmp21(f'Q27_p{n}', 'Q27_p1_plan19')[1] for n in N21}}
    tot, p = cmp21('Q27_p4', 'Q27_p4_plan20')
    r21['replication_4streams_vs_PLAN_20'] = {
        'totals': tot, 'pair': p,
        'block_busy_ratio': [round(blk21[4][g]['busy_s'] / blk['27b'][g]['busy_s'], 3) for g in range(8)],
        'same_text_as_PLAN_20': sum(int(r['27b_4streams__vs__27b_plan21b_4streams']) for r in same)}
    r21['same_text_as_one_stream'] = {f'{n}streams': sum(int(r[f'27b_1stream__vs__27b_plan21b_{n}streams']) for r in same)
                                      for n in N21}
    sp1 = o1['27b']['output_tokens'] / sum(S1['27b'].values())         # one stream's speed, unrounded
    r21['per_run'] = {f'{n}streams': {
        'never_dry_min_per_correct_estimate': round(o21[n]['output_tokens'] / o21[n]['steady_tok_s_spread'] / 60 / o21[n]['correct'], 4),
        'total_speed_vs_one_stream': round(o21[n]['total_tok_s'] / sp1, 3),
        'steady_speed_vs_one_stream': round(o21[n]['steady_tok_s_spread'] / sp1, 3),
        'per_stream_speed_vs_one_stream': round(o21[n]['per_stream_tok_s_median'] / o1['27b']['per_stream_tok_s_median'], 3),
        'latency_ratio_median_per_task': round(statistics.median(p21[n][t]['wall_s'] / one['27b'][t]['wall_s'] for t in ids), 3),
        'busy_s_per_block': [round(blk21[n][g]['busy_s'], 1) for g in range(8)],
        'minutes_by_tasks_running': {str(k): round(v / 60, 1) for k, v in sorted(prof21[n].items())}} for n in N21}
    r21['busy_ratio_8_over_4_per_block'] = [round(blk21[8][g]['busy_s'] / blk21[4][g]['busy_s'], 3) for g in range(8)]
    rep['PLAN_21b 27B at 2, 4 and 8 streams'] = r21
    return T, summary, rep


SUMMARY_COLS = ['study', 'model', 'streams', 'correct', 'minutes', 'min_per_correct', 'correct_per_hour', 'output_tokens', 'statuses',
                'hit_limit', 'timed_out', 'latency_median_s', 'latency_p90_s', 'latency_max_s', 'per_stream_tok_s_median',
                'total_tok_s', 'steady_tok_s_spread', 'steady_tok_s_full_tasks', 'full_tasks', 'under_n_share']


def as_text(rows):
    return [[str(r[c]) for c in SUMMARY_COLS] for r in rows]


def main():
    T, summary, rep = compute()
    if '--write' in sys.argv:
        with open(DATA / 'summary.csv', 'w', newline='') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerow(SUMMARY_COLS)
            w.writerows(as_text(summary))
        (DATA / 'reported_only.json').write_text(json.dumps(rep, indent=1) + '\n')
        print('wrote data/summary.csv and data/reported_only.json')
        return
    if '--check' in sys.argv:
        ok = True
        have = [list(r.values()) for r in csv.DictReader(open(DATA / 'summary.csv'))]
        bad = [i for i, (a, b) in enumerate(zip(have, as_text(summary))) if a != b]
        if bad or len(have) != len(summary):
            ok = False
            print('summary.csv: rows differ:', bad)
        else:
            print(f'summary.csv: all {len(summary)} rows match')
        want = json.loads((DATA / 'tests_preregistered.json').read_text())
        if want == json.loads(json.dumps(T)):
            print('tests: all match')
        else:
            ok = False
            for s in want:
                for k in want[s]:
                    if want[s][k] != json.loads(json.dumps(T[s].get(k))):
                        print('tests differ:', s, k)
        if json.loads((DATA / 'reported_only.json').read_text()) == json.loads(json.dumps(rep)):
            print('reported only: all match')
        else:
            ok = False
            print('reported_only.json differs')
        sys.exit(0 if ok else 1)
    for r in summary:
        print(json.dumps(r))
    print(json.dumps(T, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
