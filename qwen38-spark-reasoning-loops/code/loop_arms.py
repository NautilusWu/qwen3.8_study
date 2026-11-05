"""Ways to stop reasoning loops, on top of Part 4's runner (code/strategies of qwen38-spark-test-time-strategies),
for Qwen3.8-27B (Part 4's server) or Qwen3.8-Flash-Next (Part 6's server, code/flash_next.py).

    python3 loop_arms.py <part4>/code/strategies <run_id> [--model 27b|flashnext --part6-code <part6>/code]
        [--arms G,B,SB,S,D] [--split confirm] [--budget 32768] [--tasks id,id] [--check]

Every arm is deep thinking (the chat template's default effort), output limit 65,536, context 81,920, 40 minutes per
task, as in Parts 4-6:
  G   greedy (temperature 0), as in Parts 4-6
  B   greedy + thinking_budget: the server closes the think block at --budget thinking tokens (recommended)
  SB  the model's recommended sampling (temperature 1.0, top_p 0.95, top_k 20, seed 1234) + thinking_budget
  S   the recommended sampling alone
  D   greedy + loopdetect.LoopDetector on the streamed reasoning; when it fires, the request is stopped and the
      reasoning, cut where the loop began, is continued as a raw completion with a short sentence and </think>
Results: <part4>/code/strategies/results/<run_id>/graded.jsonl, one line per task and arm; a summary per arm.
--check (with Part 4's example tasks, --split dev): first checks that the locally rendered prompt is the server's (same
prompt tokens, same greedy token ids as the chat reply), then runs the arms with a 256-token budget and a detector that
fires after 1,500 characters, so that every path runs on tasks that do not loop. Prints CHECK PASSED or CHECK FAILED.
Environment as in Part 4 (HF_DIR, TASKS_DIR, GRADER_DIR, TF_CACHE_DIR, SANDBOX_IMAGE); Part 6's for Flash Next.
"""
import argparse
import http.client
import json
import os
import sys
import time
from pathlib import Path

import jinja2

from loopdetect import LoopDetector

MAX_TOKENS, CONTEXT, LIMIT_S, ANSWER_TOKENS = 65536, 81920, 40 * 60, 8192
SAMPLE = {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20}
CLOSE = '\n\nI keep repeating myself, so I will stop thinking here and give my final answer now.\n</think>\n\n'


def setup(a):
    sys.path.insert(0, str(Path(a.strategies).resolve()))
    if a.model == 'flashnext':
        sys.path.insert(0, str(Path(a.part6_code).resolve()))
        import flash_next
        return flash_next.patch(a.strategies)
    import common
    return common


def render(common, messages):
    """The chat prompt as the server renders it (thinking on, default effort), from the checkpoint's own template."""
    path = Path(common.HF_DIR) / common.TARGET.removeprefix('/hf/') / 'chat_template.jinja'
    env = jinja2.Environment(extensions=['jinja2.ext.loopcontrols'])

    def raise_exception(msg):
        raise jinja2.TemplateError(msg)
    env.globals['raise_exception'] = raise_exception
    return env.from_string(path.read_text()).render(messages=messages, add_generation_prompt=True, enable_thinking=True)


class ForcedDetector:
    """--check only: fires after 1,500 characters of reasoning."""
    def feed(self, text):
        return {'rule': 'forced-check', 'cut': 1500, 'at': len(text)} if len(text) >= 1500 else None


def post(common, path, body):
    conn = http.client.HTTPConnection('127.0.0.1', common.PORT, timeout=600)
    try:
        conn.request('POST', path, json.dumps(body), {'Content-Type': 'application/json'})
        return json.loads(conn.getresponse().read())
    finally:
        conn.close()


def render_check(common, task):
    messages = [{'role': 'system', 'content': common.SYSTEM}] + task['messages']
    body = {**common.body_for(messages, 300), 'stream': False}
    body.pop('stream_options')
    chat = post(common, '/v1/chat/completions', body)
    raw = post(common, '/v1/completions', {'model': common.MODEL, 'prompt': render(common, messages), 'max_tokens': 300,
                                           'stream': False, 'temperature': 0, 'top_p': 1, 'top_k': 0, 'seed': 1234,
                                           'return_token_ids': True})
    ids = [(x.get('tensorfold') or {}).get('token_ids') for x in (chat, raw)]
    return {'task_id': task['id'], 'prompt_tokens': [x['usage']['prompt_tokens'] for x in (chat, raw)],
            'same_token_ids': bool(ids[0]) and ids[0] == ids[1]}


def stream(common, path, body, t0, deadline, on_reasoning=None):
    conn = http.client.HTTPConnection('127.0.0.1', common.PORT, timeout=1800)
    reasoning, answer, usage, finish, stop, out = '', '', {}, None, None, {}
    try:
        conn.request('POST', path, json.dumps(body), {'Content-Type': 'application/json'})
        resp = conn.getresponse()
        if resp.status != 200:
            return {'error': f'HTTP {resp.status}: {resp.read()[:300]!r}', 'end': time.perf_counter() - t0}
        while True:
            if time.perf_counter() > deadline:
                out['timed_out'] = True
                break
            line = resp.readline()
            if not line:
                break
            raw = line[5:].strip() if line.startswith(b'data:') else b''
            if not raw:
                continue
            if raw == b'[DONE]':
                break
            x = json.loads(raw)
            if x.get('error'):
                out['error'] = str(x['error'])[:300]
                break
            usage.update(x.get('usage') or {})
            for c in x.get('choices') or []:
                finish = c.get('finish_reason') or finish
                d = c.get('delta') or {}
                if d.get('reasoning_content'):
                    reasoning += d['reasoning_content']
                    if on_reasoning is not None and (stop := on_reasoning(reasoning)):
                        break
                answer += d.get('content') or c.get('text') or ''
            if stop:
                break
    finally:
        conn.close()                           # the server stops a request at its next round once the client leaves
    return out | {'end': time.perf_counter() - t0, 'reasoning': reasoning, 'answer': answer, 'finish': finish,
                  'tokens': usage.get('completion_tokens'), 'stop': stop}


def run_arm(common, task, arm, budget, detector=LoopDetector):
    messages = [{'role': 'system', 'content': common.SYSTEM}] + task['messages']
    extra = {'B': {'thinking_budget': budget}, 'SB': {**SAMPLE, 'thinking_budget': budget}, 'S': SAMPLE}.get(arm, {})
    body = common.body_for(messages, MAX_TOKENS, extra=extra)
    body.pop('return_token_ids', None)
    t0 = time.perf_counter()
    deadline = t0 + LIMIT_S
    det = detector() if arm == 'D' else None
    r = stream(common, '/v1/chat/completions', body, t0, deadline, det.feed if det else None)
    rec = {'arm': arm, 'tokens': r.get('tokens'), 'finish': r.get('finish'), 'error': r.get('error'),
           'timed_out': bool(r.get('timed_out')), 'detector': r.get('stop')}
    if not r.get('stop'):
        text = r.get('reasoning', '') + ('</think>\n' + r['answer'] if r.get('answer') else '')
        return rec | {'wall_s': r['end'], 'text': text, 'answered': bool(r.get('answer'))}
    kept = r['reasoning'][:r['stop']['cut']]
    c = stream(common, '/v1/completions', {'model': common.MODEL, 'prompt': render(common, messages) + kept + CLOSE,
               'max_tokens': ANSWER_TOKENS, 'stream': True, 'stream_options': {'include_usage': True},
               'temperature': 0, 'top_p': 1, 'top_k': 0, 'seed': 1234}, t0, deadline)
    text = kept + CLOSE.replace('\n</think>\n\n', '</think>\n') + (c.get('answer') or '')
    return rec | {'wall_s': c['end'], 'text': text, 'answered': bool((c.get('answer') or '').strip()),
                  'continuation_tokens': c.get('tokens'), 'timed_out': rec['timed_out'] or bool(c.get('timed_out'))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('strategies')
    ap.add_argument('run_id')
    ap.add_argument('--model', choices=['27b', 'flashnext'], default='27b')
    ap.add_argument('--part6-code', default='')
    ap.add_argument('--arms', default='G,B')
    ap.add_argument('--split', default='confirm')
    ap.add_argument('--budget', type=int, default=32768)
    ap.add_argument('--tasks', default='')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    if a.check:
        a.budget = 256
    common = setup(a)
    tasks = common.load_tasks(a.split)
    if a.tasks:
        tasks = [t for t in tasks if t['id'] in a.tasks.split(',')]
    out = common.HERE / 'results' / a.run_id
    out.mkdir(parents=True, exist_ok=False)
    guard = common.MemoryGuard()
    rows, checks = [], []
    try:
        common.start_server(1, out / 'server_start.log', context=CONTEXT)
        if a.check:
            for task in tasks[:2]:
                checks.append(render_check(common, task))
                print(json.dumps(checks[-1]), flush=True)
        with open(out / 'requests.jsonl', 'w') as f:
            for task in tasks:
                for arm in a.arms.split(','):
                    if guard.fired:
                        raise RuntimeError('memory guard fired')
                    common.wait_idle()
                    rec = run_arm(common, task, arm, a.budget, ForcedDetector if a.check else LoopDetector)
                    rec |= {'task_id': task['id'], 'category': task['category']}
                    rows.append(rec)
                    f.write(json.dumps(rec) + '\n')
                    f.flush()
                    print(json.dumps({k: rec.get(k) for k in ('task_id', 'arm', 'wall_s', 'tokens', 'answered')}), flush=True)
    finally:
        guard.close()
        common.stop_server()
    by = {t['id']: t for t in tasks}
    g = common.grade_all([{'key': f"{r['task_id']}#{r['arm']}", 'grader': by[r['task_id']]['grader'],
                           'text': r.get('text') or '', 'thinking': True} for r in rows], out / 'grading')
    for r in rows:
        r['passed'] = bool(g[f"{r['task_id']}#{r['arm']}"]['passed']) and not r.get('timed_out')
    (out / 'graded.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    for arm in a.arms.split(','):
        rs = [r for r in rows if r['arm'] == arm]
        wall = sum(min(r['wall_s'] or LIMIT_S, LIMIT_S) for r in rs) / 60
        c = sum(r['passed'] for r in rs)
        print(json.dumps({'arm': arm, 'tasks': len(rs), 'correct': c, 'answered': sum(bool(r['answered']) for r in rs),
                          'minutes': round(wall, 1), 'min_per_correct': round(wall / c, 2) if c else None}))
    if a.check:
        ok = all(c['same_token_ids'] and c['prompt_tokens'][0] == c['prompt_tokens'][1] for c in checks) and all(
            not r.get('error') for r in rows) and all(
            sum(bool(r['answered']) for r in rows if r['arm'] == arm) >= len(tasks) - 1 for arm in a.arms.split(','))
        print('CHECK PASSED' if ok else 'CHECK FAILED')


if __name__ == '__main__':
    main()
