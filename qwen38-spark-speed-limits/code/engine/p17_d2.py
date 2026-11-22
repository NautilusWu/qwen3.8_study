"""PLAN_17 P3 (Q1): D2 (run_method.run_single 'D2': one greedy low-effort request) on the 400 private tasks (set T) with O1 (0.6.0 +
DFlash2, --parallel 1) and N1 (1.2.7 + DFlash2, --parallel 1). The 400 tasks in their fixed order form 8 groups of
50; both configurations run every group, the first one per group taken from ORDER, so 16 blocks, and the server
restarts before every block. Same structure as PLAN_15's run_ckpt.py, with p17_engine's servers and idle wait.

    python3 p17_d2.py <prefix>                                     (the formal run)
    python3 p17_d2.py <prefix> --tasks-file <examples.jsonl> --group 5 --order N1,O1   (a smoke test)

Writes results/<prefix>-D2-<config>/ (requests.jsonl in task order, blocks.json, server logs, stats.jsonl with each
request's engine statistics without token ids; then graded.jsonl and summary.json from run_method.grade_run).
Task text and model output are never printed.
"""
import argparse
import json
import os
import threading
import time
from pathlib import Path

import p17_engine as P
import common
import run_method as RM

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
ARMS = ('O1', 'N1')
ORDER = 'N1,O1,O1,N1,O1,N1,N1,O1'
GROUP = 50
KEEP = ('engine', 'tokens_per_second', 'seconds', 'prefill_seconds', 'time_to_first_token', 'drafts',
        'accepted_per_step', 'acceptance', 'draft_tokens', 'accepted_tokens', 'steps', 'reasoning_effort')

_local = threading.local()


class _JsonShim:
    def __getattr__(self, name):
        return getattr(json, name)

    @staticmethod
    def loads(s, *a, **k):
        x = json.loads(s, *a, **k)
        if isinstance(x, dict) and isinstance(x.get('tensorfold'), dict) and hasattr(_local, 'stats'):
            _local.stats.update({k2: v for k2, v in x['tensorfold'].items() if k2 in KEEP})
        return x


common.json = _JsonShim()
STATS, _lock, _current = [], threading.Lock(), {'task': None}
_plain_stream = RM.stream_chat


def stream_chat_recorded(body, t_zero, cancel=None, deadline=None):
    task = _current['task']
    _local.stats = {}
    r = _plain_stream(body, t_zero, cancel=cancel, deadline=deadline)
    with _lock:
        STATS.append({'task_id': task, 'stats': dict(_local.stats), 'sent': r.get('sent'), 'first': r.get('first'),
                      'end': r.get('end'), 'completion_tokens': r.get('completion_tokens'),
                      'prompt_tokens': r.get('prompt_tokens'), 'cancelled': bool(r.get('cancelled')),
                      'error': r.get('error'), 'finish': r.get('finish')})
    del _local.stats
    return r


RM.stream_chat = stream_chat_recorded


class MemLow:
    def __init__(self):
        self.low, self._stop = None, threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.wait(5):
            for line in Path('/proc/meminfo').read_text().splitlines():
                if line.startswith('MemAvailable:'):
                    v = int(line.split()[1]) / 2**20
                    self.low = v if self.low is None else min(self.low, v)

    def close(self):
        self._stop.set()
        return round(self.low, 1) if self.low is not None else None


def run_task(task):
    box = {}

    def go():
        try:
            box['rec'] = RM.run_single(task, 'D2')
        except Exception as exc:  # noqa: BLE001
            box['exc'] = exc

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(RM.LIMIT_S + 600)
    if t.is_alive():
        common.stop_server()
        raise RuntimeError('task did not finish within its limit + 10 minutes')
    if 'exc' in box:
        raise box['exc']
    return box['rec']


def run_block(cfg, tasks, out, group, f, guard):
    info = {'config': cfg, 'group': group, 'tasks': [t['id'] for t in tasks],
            'started': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    low, done = MemLow(), 0
    try:
        s = P.start(cfg, out / f'server_start_g{group}.log')
        info['server'] = {k: v for k, v in s.items() if k != 'argv'}
        info['argv'] = s['argv']
        warm = common.stream_chat(common.body_for([{'role': 'user', 'content': 'Compute 2+3. Reply exactly FINAL: 5.'}], 64,
                                                  extra={'chat_template_kwargs': {'enable_thinking': False}}), time.perf_counter())
        assert 'FINAL: 5' in warm.get('answer', ''), 'warm-up reply'
        for task in tasks:
            if guard.fired:
                raise RuntimeError('memory guard fired')
            waited = P.wait_idle()
            _current['task'] = task['id']
            rec = run_task(task)
            rec.pop('token_ids', None)
            rec.update(task_id=task['id'], category=task['category'], method='D2', waited_idle_s=waited,
                       max_tokens=RM.MAX_TOKENS, config=cfg, group=group)
            f.write(json.dumps(rec) + '\n')
            f.flush()
            done += 1
            print(json.dumps({'config': cfg, 'group': group, 'task': task['id'], 'wall_s': round(rec['wall_s'] or 0, 1),
                              'tokens': rec.get('completion_tokens'), 'timed_out': rec.get('timed_out')}), flush=True)
        info['ok'] = True
    except Exception as exc:  # noqa: BLE001
        info['ok'], info['error'] = False, repr(exc)[:500]
        print(f'FAILED {out.name} group {group}: {exc!r}'[:400], flush=True)
    finally:
        common.stop_server()
        info['done'], info['mem_available_low_gib'] = done, low.close()
        info['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('prefix')
    ap.add_argument('--order', default=ORDER)
    ap.add_argument('--group', type=int, default=GROUP)
    ap.add_argument('--tasks-file')
    a = ap.parse_args()
    RM.LIMIT_S, RM.MAX_TOKENS = 40 * 60, 65536
    order = a.order.split(',')
    assert set(order) <= set(ARMS)
    if a.tasks_file:
        tasks = [json.loads(l) for l in Path(a.tasks_file).read_text().splitlines()][:len(order) * a.group]
    else:
        tasks = common.load_tasks(os.environ.get('TASK_SPLIT', 'set_t'))   # TASKS_DIR/<split>.jsonl, 400 tasks
        assert len(tasks) == 400 and len(order) * a.group == 400
    assert len(tasks) == len(order) * a.group
    outs = {c: RESULTS / f'{a.prefix}-D2-{c}' for c in ARMS}
    for o in outs.values():
        o.mkdir(parents=True, exist_ok=False)
    seq = []
    for g, c in enumerate(order):
        seq += [(g, c), (g, 'O1' if c == 'N1' else 'N1')]
    guard = common.MemoryGuard()
    files = {c: open(outs[c] / 'requests.jsonl', 'w') for c in outs}
    blocks = {c: [] for c in outs}
    try:
        for n, (g, c) in enumerate(seq):
            part = tasks[g * a.group:(g + 1) * a.group]
            STATS.clear()
            info = run_block(c, part, outs[c], g, files[c], guard)
            info['position'] = n
            blocks[c].append(info)
            with open(outs[c] / 'stats.jsonl', 'a') as sf:
                sf.write(''.join(json.dumps({**s, 'group': g}) + '\n' for s in STATS))
            (outs[c] / 'blocks.json').write_text(json.dumps(blocks[c], indent=1))
            if guard.fired:
                print('FAILED memory guard fired: stopping', flush=True)
                break
    finally:
        guard.close()
        common.stop_server()
        for f in files.values():
            f.close()
    for c, o in outs.items():
        engine, parallel, drafter = P.CONFIGS[c]
        run = {'config': c, 'engine': engine, 'parallel': parallel, 'drafter': drafter, 'limit_s': RM.LIMIT_S,
               'max_tokens': RM.MAX_TOKENS, 'context': 81920, 'order': seq, 'tasks': [t['id'] for t in tasks],
               'memory_guard_fired': guard.fired}
        (o / 'run.json').write_text(json.dumps(run, indent=1))
        rows = {json.loads(l)['task_id'] for l in (o / 'requests.jsonl').read_text().splitlines()}
        try:
            summary = RM.grade_run(o, [t for t in tasks if t['id'] in rows])
            print(json.dumps({'config': c, **(summary or {})}), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f'FAILED grading {o.name}: {exc!r}', flush=True)


if __name__ == '__main__':
    main()
