"""PLAN_02_cooperative.md: P1 (model splits the task) and P2 (fixed roles + sandbox), every call low effort + greedy.

Used by run_method.py (methods P1, P2). The server runs with --parallel 4; branches are threads making
non-streaming calls. Each record keeps the time of every stage so the waiting time can be split.
"""
import json
import re
import threading
import time
from collections import Counter

import common
from common import SYSTEM, answer_key, body_for, extract_code, last_block, plain_chat, run_code

LOW = {'reasoning_effort': 'low'}
PLAN_TOKENS, WORK_TOKENS, MERGE_TOKENS = 2048, 8192, 8192
MAX_FIX = 2


def call(user_text, max_tokens, history=None):
    msgs = [{'role': 'system', 'content': SYSTEM}] + (history or [{'role': 'user', 'content': user_text}])
    body = body_for(msgs, max_tokens, extra=LOW)
    body.pop('return_token_ids')
    r = plain_chat(body)
    return {'content': r['message'].get('content') or '', 'tokens': r['usage'].get('completion_tokens') or 0,
            'wall_s': round(r['wall_s'], 2), 'finish': r['finish']}


def parallel(jobs):
    """jobs: {name: zero-arg function}; run together, return {name: result} and the wall time."""
    out, t0 = {}, time.perf_counter()

    def run(name, fn):
        try:
            out[name] = fn()
        except Exception as exc:  # noqa: BLE001 - recorded; the branch counts as empty
            out[name] = {'content': '', 'tokens': 0, 'wall_s': 0, 'error': repr(exc)[:300]}

    threads = [threading.Thread(target=run, args=(n, f)) for n, f in jobs.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out, round(time.perf_counter() - t0, 2)


def task_text(task):
    return task['messages'][-1]['content']


def tokens(*results):
    return sum(r.get('tokens') or 0 for r in results)


# --- P1 ----------------------------------------------------------------------------------------------------------

P1_PLAN = ('{task}\n\n---\nDo not solve the task yet. Decide whether it can be split into 2 to 4 independent subtasks '
           'that different people could work on at the same time, each solvable without the others\' results. '
           'Reply with JSON only: {{"split": true, "subtasks": ["...", "..."]}} or {{"split": false}}.')
P1_WORK = ('Original task:\n{task}\n\n---\nYou are one of several workers on this task. Your part only:\n{sub}\n\n'
           'Work on this part alone and reply with a short, precise result for it (not the answer to the whole task).')
P1_MERGE = ('Original task:\n{task}\n\n---\nWorkers solved these parts of it:\n{parts}\n\n---\nUsing their results '
            '(check them; fix anything wrong), give the answer to the original task, following exactly the answer '
            'format it asks for.')


def parse_plan(text):
    try:
        plan = json.loads(text[text.index('{'): text.rindex('}') + 1])
    except ValueError:
        return None
    subs = [str(s) for s in plan.get('subtasks') or [] if str(s).strip()][:4]
    return subs if plan.get('split') and len(subs) >= 2 else None


def run_p1(task):
    t0, stages = time.perf_counter(), {}
    text = task_text(task)
    plan = call(P1_PLAN.format(task=text), PLAN_TOKENS)
    stages['plan_s'] = plan['wall_s']
    subs = parse_plan(plan['content'])
    if not subs:                                            # not worth splitting: one low-effort answer (as D2)
        single = call(None, WORK_TOKENS * 3, history=task['messages'])
        stages['single_s'] = single['wall_s']
        return {'wall_s': time.perf_counter() - t0, 'text': single['content'], 'thinking': False, 'split': False,
                'stages': stages, 'completion_tokens': tokens(plan, single), 'plan': plan['content'][:500]}
    work, stages['workers_s'] = parallel({i: (lambda s=s: call(P1_WORK.format(task=text, sub=s), WORK_TOKENS))
                                          for i, s in enumerate(subs)})
    parts = '\n\n'.join(f'### Part {i + 1}: {s}\n{work[i]["content"]}' for i, s in enumerate(subs))
    merge = call(P1_MERGE.format(task=text, parts=parts), MERGE_TOKENS)
    stages['merge_s'] = merge['wall_s']
    return {'wall_s': time.perf_counter() - t0, 'text': merge['content'], 'thinking': False, 'split': True,
            'subtasks': subs, 'stages': stages, 'completion_tokens': tokens(plan, merge, *work.values()),
            'worker_s': {i: w['wall_s'] for i, w in work.items()}}


# --- P2 ----------------------------------------------------------------------------------------------------------

def lang_of(task):
    return 'cpp' if task['grader']['type'] == 'cpp_tests' else 'python'


TEST_FORMAT = {
    'python': 'Reply with one ```python block of plain assert statements that call the function or class by the name '
              'the task gives. They will be appended to the same file as the implementation: do not import it and do '
              'not define it yourself.',
    'cpp': 'Reply with one ```cpp block containing a function `void run_tests()` that uses assert(...) from <cassert> '
           'and calls the function or class by the name the task gives. It will be appended to the same file as the '
           'implementation: do not #include it, do not define it yourself and do not write main().'}
P2_SPEC_TESTS = ('Original task:\n{task}\n\n---\nDo not implement anything. Write tests for what the task asks for, '
                 'from its specification only, covering the normal behaviour. {fmt}')
P2_EDGE_TESTS = ('Original task:\n{task}\n\n---\nDo not implement anything. List to yourself the tricky edge cases '
                 'of this specification (empty inputs, limits, ties, invalid input, rounding, ordering...) and write '
                 'them as tests. {fmt}')
P2_SECOND = ('{task}\n\n(Write your own independent implementation; if a less obvious but correct approach exists, '
             'prefer it.)')
P2_FIX = ('Original task:\n{task}\n\n---\nThis implementation:\n```{lang}\n{code}\n```\nfails these tests:\n{fails}\n\n'
          'Tests are written from the specification but may themselves be wrong; follow the specification. Reply '
          'with the full corrected implementation in one ```{lang} block, in the form the task asks for.')
P2_PROGRAM = ('Original task:\n{task}\n\n---\nWrite one complete Python program that computes the answer to this '
              'task and prints only the answer. Reply with only the program in one ```python block.')
P2_OTHER = ('{task}\n\n(Solve it with a different method than the most direct one, e.g. work backwards, enumerate, or '
            'check with an independent calculation, then answer in the required format.)')
P2_CHECKER = ('Original task:\n{task}\n\n---\nDo not solve the task. Write a Python function `check(answer: str) -> bool` '
              'that returns True only if `answer` (a complete reply to this task, in its required format) satisfies '
              'everything the task requires that can be checked by a program. Reply with only the function in one '
              '```python block.')
P2_JUDGE = ('Original task:\n{task}\n\n---\nCandidate answers from independent solvers:\n{cands}\n\n---\nDecide which '
            'is right (or correct them), then give the answer to the original task, following exactly the answer '
            'format it asks for.')
CHECK_RUN = '{checker}\n\nimport json\nanswers = json.loads({payload!r})\nprint(json.dumps([bool(check(a)) for a in answers]))\n'


def code_of(text, lang):
    return extract_code(text or '', ('cpp', 'c++', 'cc', 'cxx') if lang == 'cpp' else ('python', 'py'))


def defined_names(impl):
    return set(re.findall(r'^\s*(?:def|class)\s+(\w+)', impl, flags=re.M))


def strip_self_imports(tests, impl):
    """Tests sometimes import the implementation as a module; it is in the same file, so drop those lines."""
    names = defined_names(impl)
    keep = []
    for line in tests.splitlines():
        m = re.match(r'\s*from\s+([\w.]+)\s+import\s+(.+)$', line)
        if m:
            imported = {x.strip().split(' as ')[0] for x in m.group(2).strip('() ').split(',')}
            if imported and imported <= names | {'*'}:
                continue
        m = re.match(r'\s*import\s+(\w+)\s*$', line)
        if m and m.group(1) in names:
            continue
        keep.append(line)
    return '\n'.join(keep)


def test_run(lang, impl, tests):
    if not impl or not tests:
        return {'passed': False, 'output': 'missing code or tests'}
    if lang == 'python':
        tests = strip_self_imports(tests, impl)
    else:
        tests = '\n'.join(l for l in tests.splitlines() if not re.match(r'\s*#include\s+"', l))
    if lang == 'cpp':
        program = ('#include <bits/stdc++.h>\n#include <cassert>\nusing namespace std;\n' + impl + '\n' + tests +
                   '\nint main() { run_tests(); std::cout << "TESTS OK" << std::endl; return 0; }\n')
    else:
        program = impl + '\n\n' + tests + '\n\nprint("TESTS OK")\n'
    r = run_code(lang, program)
    return {'passed': r['exit_code'] == 0 and 'TESTS OK' in r['output'], 'output': r['output'][-1500:],
            'run_s': r['wall_s']}


def run_p2_code(task, t0):
    text, lang = task_text(task), lang_of(task)
    fmt = TEST_FORMAT[lang]
    work, branches_s = parallel({
        'impl': lambda: call(None, WORK_TOKENS * 2, history=task['messages']),
        'spec_tests': lambda: call(P2_SPEC_TESTS.format(task=text, fmt=fmt), WORK_TOKENS),
        'edge_tests': lambda: call(P2_EDGE_TESTS.format(task=text, fmt=fmt), WORK_TOKENS),
        'second': lambda: call(None, WORK_TOKENS * 2, history=[{'role': 'user', 'content': P2_SECOND.format(task=text)}]),
    })
    tests = {k: last_block(work[k]['content'] or '') for k in ('spec_tests', 'edge_tests')}
    impls = {k: code_of(work[k]['content'], lang) for k in ('impl', 'second')}
    t_runs = time.perf_counter()
    results = {(i, t): test_run(lang, impls[i], tests[t]) for i in impls for t in tests}
    score = {i: sum(results[(i, t)]['passed'] for t in tests) for i in impls}
    runs_s = round(time.perf_counter() - t_runs, 2)
    full = [i for i in impls if impls[i] and score[i] == len(tests)]
    fixes, fix_calls = [], []
    if 'impl' in full or not full:
        chosen = 'impl' if 'impl' in full else None
    else:
        chosen = full[0]
    code = impls[chosen] if chosen else impls['impl']
    failed = {i: {t for t in tests if not results[(i, t)]['passed']} for i in impls}
    if chosen is None and code and impls['second'] and failed['impl'] == failed['second']:
        chosen = 'impl_tests_suspect'                # two independent implementations fail the same tests: trust them
    if chosen is None and code:                      # nothing passes everything: fix the main implementation
        best = code
        best_score = score['impl']
        for _ in range(MAX_FIX):
            fails = '\n\n'.join(f'{t}:\n{results_t["output"]}' for t, results_t in
                                ((t, test_run(lang, code, tests[t])) for t in tests) if not results_t['passed'])
            if not fails:
                break
            fix = call(P2_FIX.format(task=text, lang=lang, code=code, fails=fails[:3000]), MERGE_TOKENS)
            fix_calls.append(fix)
            new = code_of(fix['content'], lang)
            if not new:
                fixes.append({'score_before': best_score, 'score_after': None, 'wall_s': fix['wall_s'], 'no_code': True})
                break
            new_score = sum(test_run(lang, new, tests[t])['passed'] for t in tests)
            fixes.append({'score_before': best_score, 'score_after': new_score, 'wall_s': fix['wall_s']})
            code = new
            if new_score >= best_score:
                best, best_score = new, new_score
            if new_score == len(tests):
                break
        code = best
        chosen = 'impl_fixed' if any(f.get('score_after') is not None for f in fixes) else 'impl'
    if not code:
        code = impls['second'] or ''
        chosen = 'second_fallback'
    final = f'```{lang}\n{code}\n```' if code else (work['impl']['content'] or '')
    return {'wall_s': time.perf_counter() - t0, 'text': final, 'thinking': False, 'chosen': chosen,
            'stages': {'branches_s': branches_s, 'test_runs_s': runs_s, 'fix_s': round(sum(f['wall_s'] for f in fix_calls), 2)},
            'test_scores': score, 'fixes': fixes,
            'test_outputs': {f'{i}|{t}': r['output'][-400:] for (i, t), r in results.items()}, 'branch_s': {k: v['wall_s'] for k, v in work.items()},
            'completion_tokens': tokens(*work.values(), *fix_calls)}


def run_p2_answer(task, t0):
    text = task_text(task)
    work, branches_s = parallel({
        'direct': lambda: call(None, WORK_TOKENS * 2, history=task['messages']),
        'program': lambda: call(P2_PROGRAM.format(task=text), WORK_TOKENS),
        'other': lambda: call(None, WORK_TOKENS * 2, history=[{'role': 'user', 'content': P2_OTHER.format(task=text)}]),
        'checker': lambda: call(P2_CHECKER.format(task=text), WORK_TOKENS),
    })
    t_runs = time.perf_counter()
    prog = extract_code(work['program']['content'] or '', ('python', 'py'))
    prog_out = run_code('python', prog)['output'].strip() if prog else ''
    cands = {k: work[k]['content'] for k in ('direct', 'other') if work[k]['content']}
    keys = {k: answer_key(task, '</think>\n' + c) for k, c in cands.items()}
    kind = task['grader']['type']
    if prog_out and kind == 'numeric':
        val = common.parse_number(prog_out.splitlines()[-1])
        keys['program'] = None if val is None else ('num', float(f'{val:.6g}'))
    elif prog_out and kind == 'exact_output':
        keys['program'] = ('out', tuple(l.rstrip() for l in prog_out.strip('\n').splitlines()))
    checker = extract_code(work['checker']['content'] or '', ('python', 'py'))
    checks = {}
    if checker and cands:
        payload = json.dumps(list(cands.values()))
        r = run_code('python', CHECK_RUN.format(checker=checker, payload=payload))
        try:
            checks = dict(zip(cands, json.loads(r['output'].strip().splitlines()[-1])))
        except (ValueError, IndexError):
            checks = {}
    runs_s = round(time.perf_counter() - t_runs, 2)
    counts = Counter(k for k in keys.values() if k is not None)
    top = counts.most_common(1)
    judge = None
    if top and top[0][1] >= 2 and any(keys.get(k) == top[0][0] for k in cands):
        chosen = next(k for k in ('direct', 'other') if keys.get(k) == top[0][0])
        final, rule = cands[chosen], 'majority'
    elif sum(bool(v) for v in checks.values()) == 1:
        chosen = next(k for k, v in checks.items() if v)
        final, rule = cands[chosen], 'checker'
    else:
        listing = '\n\n'.join([f'### Solver "{k}":\n{c}' for k, c in cands.items()] +
                              ([f'### A program computed and printed:\n{prog_out[:1500]}'] if prog_out else []) +
                              ([f'### A constraint checker accepted: {[k for k, v in checks.items() if v]}'] if checks else []))
        judge = call(P2_JUDGE.format(task=text, cands=listing), MERGE_TOKENS)
        final, chosen, rule = judge['content'], 'judge', 'judge'
    return {'wall_s': time.perf_counter() - t0, 'text': final, 'thinking': False, 'chosen': chosen, 'rule': rule,
            'keys': {k: repr(v) for k, v in keys.items()}, 'checks': checks, 'program_output': prog_out[:300],
            'stages': {'branches_s': branches_s, 'runs_s': runs_s, 'judge_s': judge['wall_s'] if judge else 0},
            'branch_s': {k: v['wall_s'] for k, v in work.items()},
            'completion_tokens': tokens(*work.values(), *([judge] if judge else []))}


def run_p2(task):
    t0 = time.perf_counter()
    if task['grader']['type'] in ('python_tests', 'cpp_tests'):
        return run_p2_code(task, t0)
    return run_p2_answer(task, t0)
