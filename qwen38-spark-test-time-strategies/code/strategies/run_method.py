"""Run one method on the dev (or sealed) tasks, grade the answers in the sandbox, write a summary.

    python3 run_method.py <method> <run_id> [--k 4] [--tasks c_avoid,r_jugs] [--split dev]

Methods (PLAN.md): A single long think, D1 effort medium, D2 effort low, D3 thinking budget 8192,
C K parallel sampled branches with early-stopping majority vote, B agent with a run_code tool.
Each task gets at most LIMIT_S of wall time; past it the task fails and counts LIMIT_S.
"""
import argparse
import json
import threading
import time
from collections import Counter
from pathlib import Path

import common
import coop
from common import SYSTEM, answer_key, body_for, load_tasks, plain_chat, run_code, stream_chat, votes_on

LIMIT_S = 25 * 60
MAX_TOKENS = 24576
SAMPLED = {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20}
SINGLE = {'A': {}, 'D1': {'reasoning_effort': 'medium'}, 'D2': {'reasoning_effort': 'low'},
          'D3': {'thinking_budget': 8192}}


def messages(task):
    return [{'role': 'system', 'content': SYSTEM}] + task['messages']


# --- A, D: one request -------------------------------------------------------------------------------------------

def run_single(task, method, max_tokens=None):
    t_zero = time.perf_counter()
    r = stream_chat(body_for(messages(task), max_tokens or MAX_TOKENS, extra=SINGLE[method]), t_zero,
                    deadline=t_zero + LIMIT_S)
    return {'wall_s': r['end'], 'text': r['text'], 'thinking': True, 'completion_tokens': r.get('completion_tokens'),
            'finish': r.get('finish'), 'timed_out': bool(r.get('cancelled')), 'error': r.get('error'),
            'token_ids': r.get('token_ids')}


# --- C: K branches, early-stopping majority vote -----------------------------------------------------------------

def run_parallel(task, k, extra=None):
    t_zero = time.perf_counter()
    cancel = threading.Event()
    branches = [None] * k
    finished = []                       # (branch, end time, answer key), in finishing order
    lock = threading.Condition()

    def branch(i):
        r = stream_chat(body_for(messages(task), MAX_TOKENS, sampling=SAMPLED, seed=1000 + i, extra=extra), t_zero,
                        cancel=cancel, deadline=t_zero + LIMIT_S)
        branches[i] = r
        with lock:
            if not r.get('cancelled') and not r.get('error'):
                finished.append((i, r['end'], answer_key(task, r['text'])))
            lock.notify_all()

    threads = [threading.Thread(target=branch, args=(i,), daemon=True) for i in range(k)]
    for t in threads:
        t.start()
    choice, rule, decided_at = None, None, None
    with lock:
        while True:
            done = sum(b is not None for b in branches)
            keyed = [(i, end, key) for i, end, key in finished if key is not None]
            if votes_on(task):
                counts = Counter(key for _, _, key in keyed)
                top = counts.most_common(1)
                if top and top[0][1] > k / 2:
                    choice = next(i for i, _, key in keyed if key == top[0][0])
                    rule = 'majority'
                elif done == k:
                    if counts:
                        best = max(counts.values())
                        choice = next(i for i, _, key in keyed if counts[key] == best)   # earliest of the tied
                        rule = 'plurality' if list(counts.values()).count(best) == 1 else 'tie_earliest'
                    rule = rule or 'no_answer'
            else:
                if keyed:
                    choice, rule = keyed[0][0], 'first_with_code'
                elif done == k:
                    rule = 'no_answer'
            if rule:
                decided_at = time.perf_counter() - t_zero
                break
            lock.wait(timeout=1)
    cancel.set()
    for t in threads:
        t.join(timeout=60)
    if choice is None and finished:
        choice = finished[0][0]          # nothing usable: grade the first finished branch (it fails)
    text = branches[choice]['text'] if choice is not None else ''
    keys = {i: key for i, _, key in finished}
    return {'wall_s': decided_at, 'text': text, 'thinking': True, 'chosen': choice, 'rule': rule,
            'timed_out': decided_at >= LIMIT_S - 1,
            'completion_tokens': sum((b or {}).get('completion_tokens') or 0 for b in branches),
            'branches': [{'branch': i, 'end_s': (b or {}).get('end'), 'cancelled': bool((b or {}).get('cancelled')),
                          'completion_tokens': (b or {}).get('completion_tokens'), 'finish': (b or {}).get('finish'),
                          'answer_key': repr(keys.get(i)), 'text': (b or {}).get('text', ''),
                          'error': (b or {}).get('error')} for i, b in enumerate(branches)]}


# --- H (PLAN_03): deep thinking writes a direction, low effort executes it ---------------------------------------

H_PLAN = ('{task}\n\n---\nDo not solve this task yet. Give only the direction for solving it: the key ideas, the method '
          'to use, and the places where it is easy to go wrong. At most 300 words. Do not write complete code and do '
          'not give the final answer.')
H_EXEC = ('{task}\n\n---\nAn expert suggested this direction for solving it (use it, but check it):\n{plan}')
H_PLAN_TOKENS, H_EXEC_TOKENS = 16384, 65536


def run_hybrid(task):
    t0 = time.perf_counter()
    text = task['messages'][-1]['content']
    plan_body = body_for([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': H_PLAN.format(task=text)}],
                         H_PLAN_TOKENS)
    plan_body.pop('return_token_ids')
    plan = plain_chat(plan_body, timeout=LIMIT_S)
    direction = (plan['message'].get('content') or '').strip() if plan['finish'] != 'length' else ''
    if direction:
        user = H_EXEC.format(task=text, plan=direction)
    else:
        user = text                                   # no finished direction: the original task alone (as D2)
    exec_body = body_for([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}], H_EXEC_TOKENS,
                         extra={'reasoning_effort': 'low'})
    exec_body.pop('return_token_ids')
    left = LIMIT_S - (time.perf_counter() - t0)
    if left <= 5:
        return {'wall_s': time.perf_counter() - t0, 'text': '', 'thinking': False, 'timed_out': True,
                'plan_s': round(plan['wall_s'], 1), 'completion_tokens': plan['usage'].get('completion_tokens') or 0}
    ex = plain_chat(exec_body, timeout=left)
    return {'wall_s': time.perf_counter() - t0, 'text': ex['message'].get('content') or '', 'thinking': False,
            'timed_out': time.perf_counter() - t0 > LIMIT_S, 'plan_finished': bool(direction),
            'plan_s': round(plan['wall_s'], 1), 'exec_s': round(ex['wall_s'], 1), 'direction': direction[:3000],
            'plan_tokens': plan['usage'].get('completion_tokens'), 'exec_tokens': ex['usage'].get('completion_tokens'),
            'exec_finish': ex['finish'],
            'completion_tokens': (plan['usage'].get('completion_tokens') or 0) + (ex['usage'].get('completion_tokens') or 0)}


# --- B: agent with a code tool ----------------------------------------------------------------------------------

AGENT_SYSTEM = (SYSTEM + ' You can run code with the run_code tool (Python 3 or C++17, no network, 20-second limit) '
                'to test your ideas and check your answer before you give it. Your final reply, the one without a '
                'tool call, is what gets graded, so it must follow exactly the answer format the task asks for.')
TOOLS = [{'type': 'function', 'function': {
    'name': 'run_code',
    'description': 'Run a complete program and get its stdout/stderr (truncated to 4000 characters). Python runs as '
                   'a script; C++ is compiled with g++ -std=c++17 and run. No network, no files besides the program.',
    'parameters': {'type': 'object', 'properties': {
        'language': {'type': 'string', 'enum': ['python', 'cpp']},
        'code': {'type': 'string', 'description': 'The whole program, with prints for what you want to see.'}},
        'required': ['language', 'code']}}}]
MAX_CALLS = 12
AGENT_CONTEXT = 65536
TURN_TOKENS = 16384


def run_agent(task):
    t0 = time.perf_counter()
    msgs = [{'role': 'system', 'content': AGENT_SYSTEM}] + task['messages']
    turns, calls, out_tokens = [], 0, 0
    final, timed_out, error = '', False, None
    while True:
        if time.perf_counter() - t0 > LIMIT_S:
            timed_out = True
            break
        body = body_for(msgs, TURN_TOKENS)
        body.pop('return_token_ids')
        if calls < MAX_CALLS:
            body['tools'] = TOOLS
        try:
            r = plain_chat(body, timeout=max(60, LIMIT_S - (time.perf_counter() - t0)))
        except Exception as exc:  # noqa: BLE001 - recorded; the task fails
            error = repr(exc)[:300]
            break
        msg = r['message']
        out_tokens += r['usage'].get('completion_tokens') or 0
        turn = {'wall_s': round(r['wall_s'], 2), 'finish': r['finish'], 'completion_tokens': r['usage'].get('completion_tokens'),
                'content': msg.get('content'), 'reasoning_chars': len(msg.get('reasoning_content') or ''),
                'tool_calls': []}
        turns.append(turn)
        tool_calls = msg.get('tool_calls') or []
        if not tool_calls:
            final = msg.get('content') or ''
            break
        # earlier turns' thinking is not sent back (only what the model said and did), so the history stays short
        msgs.append({'role': 'assistant', 'content': msg.get('content') or '', 'tool_calls': tool_calls})
        for call in tool_calls:
            calls += 1
            try:
                args = json.loads(call['function'].get('arguments') or '{}')
                result = run_code(args.get('language', 'python'), str(args.get('code', '')))
                shown = f"exit code {result['exit_code']}\n{result['output']}"
            except (ValueError, KeyError, TypeError) as exc:
                result, shown = {'exit_code': None, 'output': '', 'wall_s': 0}, f'bad tool call: {exc}'
            turn['tool_calls'].append({'language': (args if isinstance(args, dict) else {}).get('language'),
                                       'code_chars': len(str((args if isinstance(args, dict) else {}).get('code', ''))),
                                       'exit_code': result['exit_code'], 'run_s': result['wall_s'],
                                       'output_head': result['output'][:300]})
            msgs.append({'role': 'tool', 'tool_call_id': call.get('id', ''), 'content': shown})
    return {'wall_s': time.perf_counter() - t0, 'text': final, 'thinking': False, 'timed_out': timed_out,
            'error': error, 'completion_tokens': out_tokens, 'tool_calls': calls, 'turns': turns}


# --- driver ------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('method', choices=['A', 'D1', 'D2', 'D3', 'C', 'CL', 'B', 'P1', 'P2', 'A65', 'H'])
    ap.add_argument('run_id')
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--tasks', default='')
    ap.add_argument('--split', default='dev', choices=['dev', 'sealed', 'hard', 'confirm'])
    ap.add_argument('--grade-only', action='store_true', help='grade an existing run again')
    ap.add_argument('--limit-min', type=float, default=25)
    ap.add_argument('--context', type=int, default=None)
    ap.add_argument('--max-tokens', type=int, default=None, help='output limit for A, D*, C, CL (default 24576)')
    a = ap.parse_args()
    global LIMIT_S, MAX_TOKENS
    LIMIT_S = int(a.limit_min * 60)
    if a.max_tokens:
        MAX_TOKENS = a.max_tokens
    out = common.HERE / 'results' / a.run_id
    if a.grade_only:
        tasks = load_tasks(a.split)
        return grade_run(out, [t for t in tasks if t['id'] in {json.loads(l)['task_id'] for l in (out / 'requests.jsonl').read_text().splitlines()}])
    out.mkdir(parents=True, exist_ok=False)
    tasks = load_tasks(a.split)
    if a.tasks:
        tasks = [t for t in tasks if t['id'] in a.tasks.split(',')]
    parallel = a.k if a.method in ('C', 'CL') else (4 if a.method in ('P1', 'P2') else 1)
    guard = common.MemoryGuard()
    info = {'method': a.method, 'k': a.k if a.method in ('C', 'CL') else None, 'split': a.split,
            'tasks': [t['id'] for t in tasks], 'limit_s': LIMIT_S, 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    try:
        # the agent's conversation grows over its turns: 64K context instead of 32K (all other settings the same);
        # A65 and H write up to 65,536 tokens: 80K, as in the runs
        context = a.context or {'B': AGENT_CONTEXT, 'A65': 81920, 'H': 81920}.get(a.method, 32768)
        info['server'] = common.start_server(parallel, out / 'server_start.log', context=context)
        info['context'] = context
        (out / 'run.json').write_text(json.dumps(info, indent=1))
        warm = stream_chat(body_for([{'role': 'user', 'content': 'Compute 2+3. Reply exactly FINAL: 5.'}], 64,
                                    extra={'chat_template_kwargs': {'enable_thinking': False}}), time.perf_counter())
        assert 'FINAL: 5' in warm.get('answer', ''), warm
        with open(out / 'requests.jsonl', 'w') as f:
            for task in tasks:
                if guard.fired:
                    raise RuntimeError('memory guard fired')
                waited = common.wait_idle()          # not timed: a previous request still finishing on the server
                if a.method in SINGLE:
                    rec = run_single(task, a.method)
                elif a.method == 'A65':          # PLAN_03: A with a 65,536-token output limit
                    rec = run_single(task, 'A', max_tokens=65536)
                elif a.method == 'H':            # PLAN_03: deep direction, low-effort execution
                    rec = run_hybrid(task)
                elif a.method == 'C':
                    rec = run_parallel(task, a.k)
                elif a.method in ('P1', 'P2'):   # PLAN_02_cooperative
                    rec = (coop.run_p1 if a.method == 'P1' else coop.run_p2)(task)
                elif a.method == 'CL':           # PLAN_addendum_01: every branch at low effort
                    rec = run_parallel(task, a.k, extra={'reasoning_effort': 'low'})
                else:
                    rec = run_agent(task)
                rec.update(task_id=task['id'], category=task['category'], method=a.method, waited_idle_s=waited,
                   max_tokens=MAX_TOKENS)
                f.write(json.dumps(rec) + '\n')
                f.flush()
                print(json.dumps({'task': task['id'], 'wall_s': round(rec['wall_s'] or 0, 1),
                                  'tokens': rec.get('completion_tokens'), 'timed_out': rec.get('timed_out'),
                                  'rule': rec.get('rule'), 'tool_calls': rec.get('tool_calls')}), flush=True)
    finally:
        guard.close()
        common.stop_server()
        info['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        info['memory_guard_fired'] = guard.fired
        (out / 'run.json').write_text(json.dumps(info, indent=1))
    grade_run(out, tasks)


def grade_run(out, tasks):
    rows = [json.loads(l) for l in (out / 'requests.jsonl').read_text().splitlines()]
    by_id = {t['id']: t for t in tasks}
    items = [{'key': r['task_id'], 'grader': by_id[r['task_id']]['grader'], 'text': r['text'] or '',
              'thinking': r['thinking']} for r in rows]
    for r in rows:                     # C: every branch too (the pass@K reference and per-branch accuracy)
        for b in r.get('branches') or []:
            if b['text'] and not b['cancelled']:
                items.append({'key': f"{r['task_id']}#{b['branch']}", 'grader': by_id[r['task_id']]['grader'],
                              'text': b['text'], 'thinking': True})
    grades = common.grade_all(items, out / 'grading')
    for r in rows:
        g = grades[r['task_id']]
        r['passed'] = bool(g['passed']) and not r.get('timed_out')
        r['grade_detail'] = g['detail']
        for b in r.get('branches') or []:
            b['passed'] = bool(grades.get(f"{r['task_id']}#{b['branch']}", {}).get('passed'))
    wall = sum(min(r['wall_s'] or common_limit(), common_limit()) for r in rows)
    correct = sum(r['passed'] for r in rows)
    summary = {'tasks': len(rows), 'correct': correct,
               'correct_coding': sum(r['passed'] for r in rows if r['category'] == 'coding'),
               'correct_reasoning': sum(r['passed'] for r in rows if r['category'] == 'reasoning'),
               'total_wall_min': round(wall / 60, 2),
               'min_per_correct': round(wall / 60 / correct, 2) if correct else None,
               'output_tokens': sum(r.get('completion_tokens') or 0 for r in rows),
               'timed_out': sum(bool(r.get('timed_out')) for r in rows)}
    if any(r.get('branches') for r in rows):
        summary['pass_at_k_reference'] = sum(any(b.get('passed') for b in r['branches']) for r in rows)
        done = [b for r in rows for b in r['branches'] if not b['cancelled']]
        summary['finished_branches'] = len(done)
        summary['finished_branch_pass_rate'] = round(sum(b['passed'] for b in done) / len(done), 3) if done else None
    (out / 'graded.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


def common_limit():
    return LIMIT_S


if __name__ == '__main__':
    main()
