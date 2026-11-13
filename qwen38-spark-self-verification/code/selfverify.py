"""Part 9's method (runs 3 and 4) on your own coding tasks, on top of Part 4's runner (code/strategies of
qwen38-spark-test-time-strategies). Per task, five requests are sent together and none is cancelled:
  - the 4 code branches of Part 4's vote CL4 (the task as the prompt, low thinking, temperature 0.6, top_p 0.95,
    top_k 20, seeds 1000-1003);
  - the test writer T (prompts.T_PROMPT, the same sampling, low thinking, seed 2000), which sees only the task.
When all five have ended, T's tests run once on an empty code block and then on every branch with code, each run in
its own sandbox container. The rules A, B and B2 (code/rules, unchanged) then choose from the recorded times, as in
our runs. After the server has stopped, every branch is graded with your grader (the hidden tests).

    python3 selfverify.py <part4>/code/strategies <run_id> [--split confirm] [--tasks id,id]
    python3 selfverify.py <part4>/code/strategies <run_id> --split dev --check      (Part 4's example tasks)

Only the coding tasks are run (grader types python_tests and cpp_tests). Each task must end with the sentence about
the answer format that prompts.RETURN_TAIL removes for T ("Return only ..." or "Do not write a main function. Return
only ..."); this is checked before the server starts. 40 minutes per task, output limit 65,536, context 81,920,
server --parallel 5 (Part 4's server and settings otherwise).

Environment as in Part 4 (HF_DIR, TASKS_DIR, GRADER_DIR, TF_CACHE_DIR, SANDBOX_IMAGE), and:
  SANDBOX_CPUS       cores for the sandbox containers, for T's tests and for grading (default 0-4,10-14: the DGX
                     Spark's Cortex-A725 cores, as in our runs; the model server runs on the others; empty: not pinned)
  CPP_FAIL_COUNTER   the failure counter of your grader's C++ harness, which the C++ wrapper increments (default
                     fails, as in Part 4's example grader; ours was called failures)

Results in results/<run_id>/ of this folder:
  records.jsonl      every request's text, T's tests, every test run and every hidden grade (stays local)
  branches.csv, tasks.csv, arm_A.csv, arm_B.csv, arm_B2.csv   in the columns of data/ (no text)
  summary.json       per rule: correct, minutes per correct, verified, verified but wrong
`python3 code/arms.py --dir results/<run_id>` recomputes the rules from the CSV files (the runner does this at the end).

--check: first runs hand-written tests for Part 4's two example coding tasks on a right answer, a wrong answer and
empty code (the sandbox, the C++ wrapper and the failure counter); then the method on those two tasks. It ends with
CHECK PASSED when every step ran. How many answers are right or verified is printed, not checked.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code' / 'rules'))
import arms as ARMS  # noqa: E402  (code/arms.py: the rules applied to one record, and the recheck from CSV)
import prompts as P  # noqa: E402

LIMIT_S = ARMS.LIMIT                     # 40 minutes
MAX_TOKENS, CONTEXT, K = 65536, 81920, 4
LOW = {'reasoning_effort': 'low'}
LANGS = {'python_tests': ('python', 'py'), 'cpp_tests': ('cpp', 'c++', 'cc', 'cxx')}
EMPTY = {'python_tests': '```python\n# empty\n```', 'cpp_tests': '```cpp\n// empty\n```'}
NAMES = ('A', 'B', 'B2')
COUNTER = os.environ.get('CPP_FAIL_COUNTER', 'fails')
CPUS = os.environ.get('SANDBOX_CPUS', '0-4,10-14')
C = RM = None                            # Part 4's common and run_method, set by setup()


def setup(strategies):
    global C, RM
    sys.path.insert(0, str(Path(strategies).resolve()))
    import common
    import run_method
    if CPUS:                             # also used by common.grade_all for the hidden tests
        common.SANDBOX = common.SANDBOX + ['--cpuset-cpus', CPUS]
    C, RM = common, run_method


# --- the test writer's prompt and tests ----------------------------------------------------------------------------

def task_text(task):
    assert len(task['messages']) == 1, f"{task['id']}: one user message expected"
    text = task['messages'][0]['content']
    m = P.RETURN_TAIL.search(text)
    assert m, f"{task['id']}: the task must end with its answer-format sentence ('Return only ...'), see prompts.py"
    return text[:m.start()]


def tester_messages(task):
    return [{'role': 'system', 'content': P.SYSTEM},
            {'role': 'user', 'content': P.T_PROMPT.format(task=task_text(task), fmt=P.T_FORMAT[task['grader']['type']])}]


def tests_in(task, r):
    """The code block of T's finished answer, or None."""
    if r.get('cancelled') or r.get('error'):
        return None
    final = C.final_section(r.get('text', ''), True)
    return None if final is None else C.extract_code(final, LANGS[task['grader']['type']])


MAIN_DEF = re.compile(r'\bint\s+main\s*\([^)]*\)\s*\{')
FILE_SCOPE_OK = re.compile(r'\s*(#include\b.*|using\s+namespace\s+std\s*;)?\s*')


def test_statements(kind, tests):
    """The tests as written, except a C++ block that is a whole `int main() {...}`: its body is used and whatever came
    before it is dropped. -> (tests, form)"""
    if kind != 'cpp_tests':
        return tests, 'as written'
    m = MAIN_DEF.search(tests)
    if not m:
        return tests, 'statements'
    end = tests.rfind('}')
    before = tests[:m.start()].splitlines()
    form = 'main body' if all(FILE_SCOPE_OK.fullmatch(l) for l in before) else 'main body, code before it dropped'
    return tests[m.end():end if end > m.end() else len(tests)], form


def model_grader(task, tests):
    kind = task['grader']['type']
    body, _ = test_statements(kind, tests)
    if kind == 'cpp_tests':
        body = P.CPP_WRAP.replace('failures++', COUNTER + '++').format(body=body)
    return {'type': kind, 'tests': body, 'timeout': P.TEST_TIMEOUT_S}


# --- sandbox: your grader's code-test path, one container per run, timed ----------------------------------------------

RUN_SCRIPT = '''import json, time
from pathlib import Path
from grading import grade
out = []
for item in json.loads(Path('/inputs/items.json').read_text()):
    t0 = time.perf_counter()
    try:
        score, passed, detail = grade(item['grader'], item['text'], thinking=item['thinking'])
        error = None
    except Exception as exc:
        passed, detail, error = False, repr(exc), type(exc).__name__
    out.append({'key': item['key'], 'passed': bool(passed), 'detail': str(detail)[-500:], 'grader_error': error,
                'run_s': round(time.perf_counter() - t0, 3)})
print(json.dumps(out))
'''


def sandbox(items, workdir):
    """items: [{'key', 'grader', 'text', 'thinking'}] in one container -> ({key: result}, container wall seconds)."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / 'items.json').write_text(json.dumps(items))
    (workdir / 'run.py').write_text(RUN_SCRIPT)
    args = C.SANDBOX + ['-e', 'PYTHONPATH=/grader', '-e', 'PYTHONDONTWRITEBYTECODE=1',
                        '-v', f'{Path(C.GRADER_DIR).resolve()}:/grader:ro', '-v', f'{workdir.resolve()}:/inputs:ro',
                        '--entrypoint', 'python3', C.SANDBOX_IMAGE, '/inputs/run.py']
    t0 = time.perf_counter()
    r = subprocess.run(args, capture_output=True, text=True, timeout=120 * len(items) + 120, check=True)
    return {x['key']: x for x in json.loads(r.stdout)}, round(time.perf_counter() - t0, 3)


def sandbox_safe(items, workdir):
    """sandbox(), but a container that fails as a whole (e.g. a test that exhausts its memory) is retried one item per
    container, and an item that still fails counts as not passed."""
    t0 = time.perf_counter()
    try:
        return sandbox(items, workdir)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as exc:
        if len(items) > 1:
            out = {}
            for j, item in enumerate(items):
                out.update(sandbox_safe([item], Path(workdir) / f'single{j:02d}')[0])
            return out, round(time.perf_counter() - t0, 3)
        return ({items[0]['key']: {'key': items[0]['key'], 'passed': False, 'detail': f'sandbox failed: {exc!r}'[:500],
                                   'grader_error': 'sandbox', 'run_s': None}}, round(time.perf_counter() - t0, 3))


def timed_run(task, tests, key, text, thinking, workdir):
    res, wall = sandbox_safe([{'key': key, 'grader': model_grader(task, tests), 'text': text, 'thinking': thinking}],
                             workdir / key.replace('#', '_'))
    return {**{k: res[key][k] for k in ('passed', 'detail', 'grader_error', 'run_s')}, 'container_s': wall}


# --- one task ------------------------------------------------------------------------------------------------------

def run_task(task, workdir):
    kind = task['grader']['type']
    t_zero = time.perf_counter()
    bodies = [C.body_for(RM.messages(task), MAX_TOKENS, sampling=RM.SAMPLED, seed=1000 + i, extra=LOW) for i in range(K)]
    bodies.append(C.body_for(tester_messages(task), MAX_TOKENS, sampling=RM.SAMPLED, seed=2000, extra=LOW))
    out = [None] * len(bodies)

    def go(j):
        try:
            out[j] = C.stream_chat(bodies[j], t_zero, deadline=t_zero + LIMIT_S)
        except Exception as exc:  # noqa: BLE001 - recorded; that request counts as failed
            out[j] = {'error': repr(exc)[:300], 'end': time.perf_counter() - t_zero, 'text': ''}

    threads = [threading.Thread(target=go, args=(j,), daemon=True) for j in range(len(bodies))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rec = {'branches': [], 'tester': None}
    for i, r in enumerate(out[:K]):
        finished = not r.get('cancelled') and not r.get('error')
        rec['branches'].append({
            'branch': i, 'end_s': r['end'], 'cancelled': bool(r.get('cancelled')), 'error': r.get('error'),
            'finish': r.get('finish'), 'completion_tokens': r.get('completion_tokens'), 'text': r.get('text', ''),
            'candidate': finished and C.answer_key(task, r.get('text', '')) is not None})
    t = out[K]
    rec['tester'] = {'end_s': t['end'], 'cancelled': bool(t.get('cancelled')), 'error': t.get('error'),
                     'finish': t.get('finish'), 'completion_tokens': t.get('completion_tokens'), 'text': t.get('text', ''),
                     'tests': tests_in(task, t)}
    tests = rec['tester']['tests']
    if tests is not None:
        rec['tester']['tests_form'] = test_statements(kind, tests)[1]
        rec['tester']['empty_rt'] = timed_run(task, tests, 'T#empty', EMPTY[kind], False, workdir)
        for b in rec['branches']:
            if b['candidate']:
                b['t_run'] = timed_run(task, tests, f"T#{b['branch']}", b['text'], True, workdir)
    return rec


# --- after the run: hidden tests, rules, files -----------------------------------------------------------------------

def evaluate(out, tasks, rows, chunk=50):
    by_id = {t['id']: t for t in tasks}
    hidden = [{'key': f"{r['task_id']}#{b['branch']}", 'grader': by_id[r['task_id']]['grader'], 'text': b['text'],
               'thinking': True} for r in rows for b in r['branches'] if b['text'] and not b['cancelled']]
    grades = {}
    for start in range(0, len(hidden), chunk):
        grades.update(C.grade_all(hidden[start:start + chunk], out / 'grading_hidden' / f'part{start // chunk:03d}'))
    for r in rows:
        for b in r['branches']:
            g = grades.get(f"{r['task_id']}#{b['branch']}")
            b['hidden'] = {'passed': bool(g and g['passed']), 'graded': g is not None,
                           'detail': g['detail'] if g else 'not graded (cut or error)'}
        r['arms'] = {name: ARMS.arm(r, name) for name in NAMES}
    (out / 'records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return len(hidden)


def write_csv(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(header)
        w.writerows(rows)


def write_files(out, rows):
    """The per-task files in the columns of data/ (no text), and summary.json."""
    one = lambda x: '' if x is None else str(int(bool(x)))  # noqa: E731
    blank = lambda x: '' if x is None else x  # noqa: E731
    write_csv(out / 'branches.csv', ['task_id', 'branch', 'end_s', 'output_tokens', 'finish', 'has_code', 'hidden_passed',
                                     'tests_passed', 'tests_container_s'],
              [[r['task_id'], b['branch'], f"{b['end_s']:.6f}", blank(b['completion_tokens']), b['finish'] or '',
                one(b['candidate']), one(b['hidden']['passed']), one((b.get('t_run') or {}).get('passed')),
                f"{b['t_run']['container_s']:.6f}" if b.get('t_run') else ''] for r in rows for b in r['branches']])
    write_csv(out / 'tasks.csv', ['task_id', 'tester_end_s', 'tester_output_tokens', 'tester_finish', 'tests_given',
                                  'tests_form', 'empty_run_passed', 'empty_run_container_s'],
              [[r['task_id'], f"{r['tester']['end_s']:.6f}", blank(r['tester']['completion_tokens']),
                r['tester']['finish'] or '', one(r['tester']['tests'] is not None), r['tester'].get('tests_form', ''),
                one((r['tester'].get('empty_rt') or {}).get('passed')),
                f"{r['tester']['empty_rt']['container_s']:.6f}" if r['tester'].get('empty_rt') else ''] for r in rows])
    summary = {'tasks': len(rows), 'limit_s': LIMIT_S,
               'any_branch_right': sum(any(b['hidden']['passed'] for b in r['branches']) for r in rows)}
    for name in NAMES:
        write_csv(out / f'arm_{name}.csv', ['task_id', 'passed', 'wall_s', 'chosen_branch', 'verified', 'why'],
                  [[r['task_id']] + [r['arms'][name][k] for k in ('passed', 'wall_s', 'chosen_branch', 'verified', 'why')]
                   for r in rows])
        xs = [r['arms'][name] for r in rows]
        right = sum(x['passed'] == '1' for x in xs)
        minutes = sum(float(x['wall_s']) for x in xs) / 60
        summary[name] = {'correct': right, 'total_min': round(minutes, 2),
                         'min_per_correct': round(minutes / right, 2) if right else None,
                         'verified': sum(x['verified'] == '1' for x in xs),
                         'verified_wrong': sum(x['verified'] == '1' and x['passed'] == '0' for x in xs)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    return summary


def print_summary(rows, summary):
    print(f"\n{'task':<18}{'branches ended (s)':<30}{'T ended':>9}{'tests':>7}{'empty':>7}{'T passed':>10}"
          f"{'right':>7}  B2")
    for r in rows:
        t = r['tester']
        runs = ''.join('-' if not b.get('t_run') else ('P' if b['t_run']['passed'] else 'f') for b in r['branches'])
        right = ''.join('1' if b['hidden']['passed'] else '0' for b in r['branches'])
        b2 = r['arms']['B2']
        ends = ' '.join(f"{b['end_s']:.0f}" for b in r['branches'])
        print(f"{r['task_id']:<18}{ends:<30}{t['end_s']:>9.0f}"
              f"{'yes' if t['tests'] is not None else 'no':>7}"
              f"{'' if not t.get('empty_rt') else ('pass' if t['empty_rt']['passed'] else 'fail'):>7}{runs:>10}{right:>7}"
              f"  branch {b2['chosen_branch'] or '-'}, {'verified' if b2['verified'] == '1' else 'unverified'}, "
              f"{'right' if b2['passed'] == '1' else 'wrong'}, {float(b2['wall_s']):.1f} s")
    print(f"\n{'rule':<6}{'correct':>10}{'total min':>11}{'min/correct':>13}{'verified':>10}{'verified wrong':>16}")
    for name in NAMES:
        s = summary[name]
        print(f"{name:<6}{s['correct']:>6}/{summary['tasks']:<3}{s['total_min']:>11}{s['min_per_correct'] or '-':>13}"
              f"{s['verified']:>10}{s['verified_wrong']:>16}")
    print(f"any branch right: {summary['any_branch_right']}/{summary['tasks']}")


# --- --check: the test path on hand-written tests (Part 4's two example coding tasks) ---------------------------------

PY_TESTS = '''import random
random.seed(0)
def brute_palindrome(s):
    t = [c.lower() for c in s if c.isalnum()]
    return t == t[::-1]
for _ in range(200):
    s = ''.join(random.choice('aAbB1 ,!') for _ in range(random.randint(0, 8)))
    assert is_palindrome(s) == brute_palindrome(s), s
assert is_palindrome('A man, a plan, a canal: Panama')'''
CPP_TESTS = '''auto brute_gcd = [](long long a, long long b) {
    a = llabs(a); b = llabs(b); long long g = 0;
    for (long long d = 1; d <= std::max(a, b); d++) if (a % d == 0 && b % d == 0) g = d;
    return g;
};
std::mt19937 rng(0);
for (int i = 0; i < 200; i++) {
    long long a = (long long)(rng() % 61) - 30, b = (long long)(rng() % 61) - 30;
    CHECK(gcd_ll(a, b) == brute_gcd(a, b));
}
CHECK(gcd_ll(0, 0) == 0);'''
PY_RIGHT = 'def is_palindrome(s: str) -> bool:\n    t = [c.lower() for c in s if c.isalnum()]\n    return t == t[::-1]\n'
PY_WRONG = 'def is_palindrome(s: str) -> bool:\n    t = [c for c in s if c.isalnum()]\n    return t == t[::-1]\n'
CPP_RIGHT = ('#include <cstdlib>\nlong long gcd_ll(long long a, long long b) {\n    a = llabs(a); b = llabs(b);\n'
             '    while (b) { long long t = a % b; a = b; b = t; }\n    return a;\n}\n')
CPP_WRONG = ('long long gcd_ll(long long a, long long b) {\n    while (b) { long long t = a % b; a = b; b = t; }\n'
             '    return a;\n}\n')
# (task, label, tests, code or None for the empty block, expected pass)
CHECK_CASES = [
    ('ex_palindrome', 'right answer', PY_TESTS, PY_RIGHT, True),
    ('ex_palindrome', 'wrong answer (case)', PY_TESTS, PY_WRONG, False),
    ('ex_palindrome', 'empty code', PY_TESTS, None, False),
    ('ex_gcd_cpp', 'right answer', CPP_TESTS, CPP_RIGHT, True),
    ('ex_gcd_cpp', 'wrong answer (sign)', CPP_TESTS, CPP_WRONG, False),
    ('ex_gcd_cpp', 'empty code', CPP_TESTS, None, False),
    ('ex_gcd_cpp', 'tests as int main(), right', '#include <bits/stdc++.h>\nint main() {\n' + CPP_TESTS + '\nreturn 0;\n}',
     CPP_RIGHT, True),
    ('ex_gcd_cpp', 'tests as int main(), wrong', '#include <bits/stdc++.h>\nint main() {\n' + CPP_TESTS + '\nreturn 0;\n}',
     CPP_WRONG, False),
    ('ex_gcd_cpp', 'failed CHECK, then return 0', 'CHECK(gcd_ll(4, 6) == 3);\nreturn 0;', CPP_RIGHT, False),
]


def harness_check(tasks, workdir):
    by_id = {t['id']: t for t in tasks}
    missing = sorted({c[0] for c in CHECK_CASES} - set(by_id))
    assert not missing, f'--check needs Part 4 example tasks {missing}: TASKS_DIR=<part4>/examples/tasks --split dev'
    items, expect = [], {}
    for j, (tid, label, tests, code, want) in enumerate(CHECK_CASES):
        kind = by_id[tid]['grader']['type']
        fence = 'python' if kind == 'python_tests' else 'cpp'
        text = EMPTY[kind] if code is None else f'(thinking)\n</think>\n```{fence}\n{code}```'
        items.append({'key': str(j), 'grader': model_grader(by_id[tid], tests), 'text': text, 'thinking': code is not None})
        expect[str(j)] = want
    res, wall = sandbox_safe(items, workdir)
    ok = True
    print(f"{'task':<16}{'case':<32}{'expected':>10}{'got':>6}")
    for j, (tid, label, *_rest) in enumerate(CHECK_CASES):
        got = res[str(j)]['passed']
        ok &= got == expect[str(j)]
        print(f"{tid:<16}{label:<32}{'pass' if expect[str(j)] else 'fail':>10}{'pass' if got else 'fail':>6}"
              + ('' if got == expect[str(j)] else f"   MISMATCH: {res[str(j)]['detail'][-200:]!r}"))
    print(f'harness check: {"passed" if ok else "FAILED"} ({len(items)} runs, {wall:.1f} s)')
    return ok


# --- driver ----------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('strategies', help="Part 4's code/strategies folder")
    ap.add_argument('run_id')
    ap.add_argument('--split', default='confirm', help='TASKS_DIR/<split>.jsonl')
    ap.add_argument('--tasks', default='', help='only these task ids, comma-separated')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    setup(a.strategies)
    assert P.run34_sha256() == P.RUN34_SHA256, 'prompts.py differs from the prompts of runs 3-4'
    tasks = [t for t in C.load_tasks(a.split) if t['grader']['type'] in LANGS]
    if a.tasks:
        tasks = [t for t in tasks if t['id'] in a.tasks.split(',')]
    assert tasks, 'no coding tasks (grader type python_tests or cpp_tests)'
    for t in tasks:
        tester_messages(t)               # every task has the answer-format sentence (checked before the server starts)
    out = ROOT / 'results' / a.run_id
    out.mkdir(parents=True, exist_ok=False)
    if a.check and not harness_check(tasks, out / 'harness_check'):
        print('CHECK FAILED: the test path (sandbox, C++ wrapper or failure counter)')
        sys.exit(1)
    guard = C.MemoryGuard()
    info = {'tasks': [t['id'] for t in tasks], 'split': a.split, 'limit_s': LIMIT_S, 'max_tokens': MAX_TOKENS,
            'context': CONTEXT, 'parallel': K + 1, 'test_timeout_s': P.TEST_TIMEOUT_S, 'tester_prompt_sha256':
            P.RUN34_SHA256, 'sandbox_cpus': CPUS, 'cpp_fail_counter': COUNTER, 'sandbox_image': C.SANDBOX_IMAGE,
            'started': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    rows = []
    try:
        info['server'] = C.start_server(K + 1, out / 'server_start.log', context=CONTEXT)
        (out / 'run.json').write_text(json.dumps(info, indent=1))
        warm = C.stream_chat(C.body_for([{'role': 'user', 'content': 'Compute 2+3. Reply exactly FINAL: 5.'}], 64,
                                        extra={'chat_template_kwargs': {'enable_thinking': False}}), time.perf_counter())
        assert 'FINAL: 5' in warm.get('answer', ''), warm
        with open(out / 'requests.jsonl', 'w') as f:
            for task in tasks:
                if guard.fired:
                    raise RuntimeError('memory guard fired')
                waited = C.wait_idle()       # not timed
                rec = run_task(task, out / 'sandbox' / task['id'])
                if guard.fired:
                    raise RuntimeError(f"memory guard fired during {task['id']}")
                if all(b['error'] for b in rec['branches']) and rec['tester']['error']:
                    raise RuntimeError(f"all five requests failed on {task['id']}")
                rec.update(task_id=task['id'], type=task['grader']['type'], waited_idle_s=waited)
                rows.append(rec)
                f.write(json.dumps(rec) + '\n')
                f.flush()
                print(json.dumps({'task': task['id'], 'branches_end_s': [round(x['end_s'], 1) for x in rec['branches']],
                                  'tester_end_s': round(rec['tester']['end_s'], 1),
                                  'tests': rec['tester']['tests'] is not None,
                                  'empty_passed': (rec['tester'].get('empty_rt') or {}).get('passed')}), flush=True)
    finally:
        guard.close()
        C.stop_server()
        info['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        info['memory_guard_fired'] = guard.fired
        (out / 'run.json').write_text(json.dumps(info, indent=1))
    graded = evaluate(out, tasks, rows)
    summary = write_files(out, rows)
    print_summary(rows, summary)
    print()
    bad = ARMS.check(out, NAMES)
    if not a.check:
        return
    problems = []
    if bad:
        problems.append(f'{bad} rule results differ when recomputed from the CSV files')
    for r in rows:
        reqs = r['branches'] + [r['tester']]
        if any(x['error'] or x['cancelled'] for x in reqs):
            problems.append(f"{r['task_id']}: a request failed or was cut")
        if any(b['text'] and not b['cancelled'] and not b['hidden']['graded'] for b in r['branches']):
            problems.append(f"{r['task_id']}: a branch was not graded")
        if r['tester']['tests'] is not None and (not r['tester'].get('empty_rt') or
                                                 any(b['candidate'] and not b.get('t_run') for b in r['branches'])):
            problems.append(f"{r['task_id']}: a test run is missing")
    print(f'\n{len(rows)} tasks, {5 * len(rows)} requests, {graded} branches graded')
    print('CHECK PASSED: the method runs end to end' if not problems else 'CHECK FAILED: ' + '; '.join(problems))
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
