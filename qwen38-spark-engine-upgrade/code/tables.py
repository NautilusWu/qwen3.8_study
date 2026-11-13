"""Recompute every table of the README from the published files in data/ (no GPU, no task files needed).

    python3 code/tables.py            print the tables
    python3 code/tables.py --check    also check data/summary.csv and data/tests_preregistered.json against a fresh
                                      computation (stats.py is run on the per-task files)

Engines: tf125 = TensorFold 1.2.5 (native), tf060 = TensorFold 0.6.0, both run in this study in alternating blocks;
tf060_part5_run = Part 5's run of the same configuration on 0.6.0 (October). Times are capped at 40 minutes per task.
"""
import csv
import json
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIMIT = 40
CONFIGS = ('tf125', 'tf060', 'tf060_part5_run')
GROUPS = ('coding_python', 'coding_cpp', 'reasoning')
SUMMARY_COLUMNS = ['config', 'engine', 'run', 'tasks', 'correct', 'correct_coding_python', 'correct_coding_cpp',
                   'correct_reasoning', 'total_min', 'min_per_correct', 'output_tokens', 'finished_branches',
                   'cancelled_branches', 'branches_hit_output_limit', 'timed_out', 'no_answer', 'any_branch_correct']
ABOUT = {'tf125': ('TensorFold 1.2.5', 'this study'), 'tf060': ('TensorFold 0.6.0', 'this study'),
         'tf060_part5_run': ('TensorFold 0.6.0', 'Part 5 (October)')}


def rows(path):
    return list(csv.DictReader(open(path)))


def minutes(r):
    return min(float(r['wall_s']) / 60, LIMIT)


def med(xs, nd=3):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), nd) if xs else None


def f(x):
    return float(x) if x not in ('', None) else None


def compute(data):
    data = Path(data)
    per = {c: {r['task_id']: r for r in rows(data / 'per_task_400' / f'qwen27b_vote4_low_CL4_{c}.csv')} for c in CONFIGS}
    br = rows(data / 'branches.csv')
    ids = sorted(per['tf125'])
    T = {'configs': {}, 'groups': {}}
    for c, p in per.items():
        r = list(p.values())
        ok = [x for x in r if x['passed'] == '1']
        tot = sum(minutes(x) for x in r)
        s = {'tasks': len(r), 'correct': len(ok), 'total_min': round(tot, 1), 'min_per_correct': round(tot / len(ok), 3),
             'output_tokens': sum(int(x['output_tokens']) for x in r), 'vote_rules': dict(Counter(x['vote_rule'] for x in r))}
        for g in GROUPS:
            s[f'correct_{g}'] = sum(x['group'] == g for x in ok)
        if 'finished_branches' in r[0]:
            for k in ('finished_branches', 'cancelled_branches', 'branches_hit_output_limit', 'timed_out', 'any_branch_correct'):
                s[k] = sum(int(x[k]) for x in r)
        T['configs'][c] = s
        T['groups'][c] = {}
        for g in GROUPS:
            gi = [x for x in r if x['group'] == g]
            gc = sum(x['passed'] == '1' for x in gi)
            gt = sum(minutes(x) for x in gi)
            T['groups'][c][g] = {'correct': f'{gc}/{len(gi)}', 'total_min': round(gt, 1),
                                 'min_per_correct': round(gt / gc, 3) if gc else None}
    # speed inside the main run (4 streams at once), from every branch
    T['speed'] = {}
    for e in ('tf125', 'tf060'):
        mine = [b for b in br if b['engine'] == e]
        fin = [b for b in mine if b['cancelled'] == '0']
        T['speed'][e] = {'branches': len(mine), 'finished': len(fin),
                         'tok_s_from_wall_median': med(f(b['tok_s_from_wall']) for b in fin),
                         'tok_s_engine_median': med(f(b['tok_s_engine']) for b in fin),
                         'first_token_s_median': med(f(b['first_token_s']) for b in mine)}
    # same text (as program checks; the texts themselves are not published)
    T['same_text'] = {
        'tf125_tf060': {'chosen_same_text': sum(int(per['tf125'][t]['chosen_same_text_as_tf060']) for t in ids),
                        'finished_branch_pairs_same_text': sum(b['same_text_tf125_tf060'] == '1' for b in br if b['engine'] == 'tf125')},
        'tf060_part5_run': {'chosen_same_text': sum(int(per['tf060'][t]['chosen_same_text_as_part5_run']) for t in ids),
                            'finished_branch_pairs_same_text': sum(b['same_text_tf060_part5_run'] == '1' for b in br if b['engine'] == 'tf060')}}
    T['same_correct_tf060_part5_run'] = sum(per['tf060'][t]['passed'] == per['tf060_part5_run'][t]['passed'] for t in ids)
    T['only_one_correct'] = {a: sorted(t for t in ids if per[a][t]['passed'] == '1' and per[b][t]['passed'] == '0')
                             for a, b in (('tf125', 'tf060'), ('tf060', 'tf125'))}
    fin_tok = {e: Counter() for e in ('tf125', 'tf060')}
    for b in br:
        if b['cancelled'] == '0':
            fin_tok[b['engine']][b['task_id']] += int(b['output_tokens'])
    T['tokens_ratio_median'] = med(fin_tok['tf125'][t] / max(fin_tok['tf060'][t], 1) for t in ids)
    # per task (looked at after the results; not a pre-registered test)
    ratio = {t: minutes(per['tf125'][t]) / minutes(per['tf060'][t]) for t in ids}
    T['per_task'] = {'all': {'tf125_faster': sum(ratio[t] < 1 for t in ids), 'tasks': len(ids), 'median_ratio': med(ratio.values())}}
    for g in GROUPS:
        gi = [t for t in ids if per['tf125'][t]['group'] == g]
        T['per_task'][g] = {'tf125_faster': sum(ratio[t] < 1 for t in gi), 'tasks': len(gi), 'median_ratio': med(ratio[t] for t in gi)}
    diff = sorted(ids, key=lambda t: minutes(per['tf060'][t]) - minutes(per['tf125'][t]))
    total_diff = sum(minutes(per['tf125'][t]) - minutes(per['tf060'][t]) for t in ids)
    T['tail'] = {'total_extra_min_tf125': round(total_diff, 1),
                 'largest': [{'task_id': t, 'group': per['tf125'][t]['group'], 'tf125_min': round(minutes(per['tf125'][t]), 1),
                              'tf060_min': round(minutes(per['tf060'][t]), 1), 'tf125_passed': int(per['tf125'][t]['passed']),
                              'tf060_passed': int(per['tf060'][t]['passed'])} for t in diff[:5]],
                 'largest_two_extra_min': round(sum(minutes(per['tf125'][t]) - minutes(per['tf060'][t]) for t in diff[:2]), 1),
                 'without_largest_two_ratio': round(sum(minutes(per['tf125'][t]) for t in ids if t not in diff[:2]) /
                                                    sum(minutes(per['tf060'][t]) for t in ids if t not in diff[:2]), 3)}
    # speed microbench (public prompts), medians over the rounds of each engine
    mb = rows(data / 'microbench.csv')
    T['microbench'] = {}
    for e in ('tf125', 'tf060'):
        mine = [r for r in mb if r['engine'] == e]
        s = {}
        for item in sorted({r['item'] for r in mine}):
            rs = [r for r in mine if r['item'] == item]
            if item == 'four_streams':
                s[item] = {'aggregate_tok_s': statistics.median(f(r['aggregate_tok_s']) for r in rs),
                           'per_stream_tok_s': statistics.median(float(v) for r in rs for v in r['per_stream_tok_s'].split())}
            elif item.startswith('cold_prompt'):
                s[item] = {'prompt_tokens': int(rs[0]['prompt_tokens']), 'ttft_s': statistics.median(f(r['ttft_s']) for r in rs),
                           'prefill_tok_s': statistics.median(f(r['prefill_tok_s']) for r in rs)}
            else:
                s[item] = {'tok_s': statistics.median(f(r['tok_s']) for r in rs),
                           'completion_tokens': sorted({int(r['completion_tokens']) for r in rs})}
        T['microbench'][e] = s
    bl = rows(data / 'blocks.csv')
    T['blocks'] = {e: {'blocks': sum(r['engine'] == e for r in bl), 'ok': sum(r['engine'] == e and r['ok'] == '1' for r in bl),
                       'startup_s': sorted(float(r['startup_s']) for r in bl if r['engine'] == e),
                       'mem_available_low_gib': min(float(r['mem_available_low_gib']) for r in bl if r['engine'] == e)}
                   for e in ('tf125', 'tf060')}
    return T


def summary_rows(T):
    out = []
    for c in CONFIGS:
        s = T['configs'][c]
        out.append([c, *ABOUT[c], s['tasks'], s['correct'], s['correct_coding_python'], s['correct_coding_cpp'],
                    s['correct_reasoning'], s['total_min'], s['min_per_correct'], s['output_tokens'],
                    s.get('finished_branches', ''), s.get('cancelled_branches', ''), s.get('branches_hit_output_limit', ''),
                    s.get('timed_out', ''), s['vote_rules'].get('no_answer', 0), s.get('any_branch_correct', '')])
    return out


def show(T):
    print('== Main table (CL4, 400 tasks)')
    print(f'{"config":16s} {"correct":>8s} {"total min":>10s} {"min/correct":>12s} {"tokens":>10s}')
    for c in CONFIGS:
        s = T['configs'][c]
        print(f'{c:16s} {s["correct"]:>4d}/400 {s["total_min"]:>10.1f} {s["min_per_correct"]:>12.3f} {s["output_tokens"]:>10,d}')
    print('\n== By group')
    for c in ('tf125', 'tf060'):
        for g in GROUPS:
            v = T['groups'][c][g]
            print(f'{c:8s} {g:14s} {v["correct"]:>8s} {v["total_min"]:>7.1f} min {v["min_per_correct"]:.3f} min/correct')
    print('\n== Vote rules, branches')
    for c in ('tf125', 'tf060'):
        s = T['configs'][c]
        print(f'{c:8s} rules {s["vote_rules"]}  finished {s["finished_branches"]} cancelled {s["cancelled_branches"]} '
              f'hit limit {s["branches_hit_output_limit"]} timed out {s["timed_out"]} any branch right {s["any_branch_correct"]}')
    print(f'finished-branch tokens per task, tf125/tf060, median: {T["tokens_ratio_median"]}')
    print('\n== Speed inside the run (4 streams), medians')
    for e, s in T['speed'].items():
        print(f'{e:8s} tok/s from wall clock {s["tok_s_from_wall_median"]}, as the engine reports {s["tok_s_engine_median"]}, '
              f'first token {s["first_token_s_median"]} s ({s["finished"]} finished of {s["branches"]} branches)')
    print('\n== Same text, same answers')
    for k, v in T['same_text'].items():
        print(f'{k:16s} chosen answers with the same text {v["chosen_same_text"]}/400, finished same-seed branch pairs '
              f'with the same text {v["finished_branch_pairs_same_text"]}')
    print(f'tf060 vs Part 5 run, same right/wrong on {T["same_correct_tf060_part5_run"]}/400 tasks')
    for k, v in T['only_one_correct'].items():
        print(f'only {k} correct ({len(v)}): {" ".join(v)}')
    print('\n== Per task (looked at after the results)')
    for g, v in T['per_task'].items():
        print(f'{g:14s} tf125 faster on {v["tf125_faster"]}/{v["tasks"]}, median time ratio tf125/tf060 {v["median_ratio"]}')
    t = T['tail']
    print(f'tf125 total extra minutes {t["total_extra_min_tf125"]}; the two largest extra times {t["largest_two_extra_min"]} min; '
          f'without these two tasks the total time ratio is {t["without_largest_two_ratio"]}')
    for x in t['largest']:
        print(f'  {x["task_id"]} {x["group"]:14s} tf125 {x["tf125_min"]:5.1f} min ({"right" if x["tf125_passed"] else "wrong"}), '
              f'tf060 {x["tf060_min"]:5.1f} min ({"right" if x["tf060_passed"] else "wrong"})')
    print('\n== Speed microbench (6 rounds, greedy), medians: tf060 | tf125')
    a, b = T['microbench']['tf060'], T['microbench']['tf125']
    for item in a:
        print(f'{item:22s} {json.dumps(a[item])} | {json.dumps(b[item])}')
    print('\n== Server starts and memory')
    for e, v in T['blocks'].items():
        print(f'{e:8s} blocks {v["ok"]}/{v["blocks"]} ok, startup s {v["startup_s"]}, lowest available memory {v["mem_available_low_gib"]} GiB')


def check(T, data):
    data = Path(data)
    bad = 0
    old = rows(data / 'summary.csv')
    new = [dict(zip(SUMMARY_COLUMNS, map(str, r))) for r in summary_rows(T)]
    if old == new:
        print(f'summary.csv: all {len(new)} rows match')
    else:
        bad += 1
        print('summary.csv: DIFFERS')
    tests = json.loads((data / 'tests_preregistered.json').read_text())
    D = data / 'per_task_400'
    for key, names in (('M1', ('tf125', 'tf060')), ('tf060_this_run_vs_part5_run_reported_only', ('tf060', 'tf060_part5_run'))):
        out = json.loads(subprocess.run([sys.executable, str(HERE / 'stats.py')] +
                                        [f'{n}={D}/qwen27b_vote4_low_CL4_{n}.csv' for n in names],
                                        check=True, capture_output=True, text=True).stdout)
        ok = out == tests[key]['stats_py']
        bad += not ok
        print(f'tests: {key} {"matches" if ok else "DIFFERS"}')
    print('tests: all match' if not bad else 'CHECK FAILED')
    return bad


def main():
    data = HERE.parent / 'data'
    T = compute(data)
    show(T)
    if '--check' in sys.argv[1:]:
        print()
        sys.exit(1 if check(T, data) else 0)


if __name__ == '__main__':
    main()
