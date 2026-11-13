"""This study's comparison on your own tasks: CL4 (4 low-effort branches, early-stopping vote; Part 4's
run_method.run_parallel, unchanged) with TensorFold 1.2.5 and 0.6.0 in alternating blocks.

    python3 run_engines.py <part4>/code/strategies <run_id> [--split confirm] [--tasks id,id]
                           [--first new,old,old,new,old,new,new,old] [--limit-min 40] [--max-tokens 65536]

The tasks, in their file order, form len(--first) groups of (nearly) equal size; each group is run by both engines,
the one named in --first first, so there are 2 x len(--first) blocks, and the server restarts before every block
(with the same warm-up request as Part 4). The study: 400 tasks, the default --first, 8 groups of 50.
Before each task the runner waits until the server is idle (a cancelled branch still finishing is not timed).

Writes <this folder>/results/<run_id>/: tf125/ and tf060/ (requests.jsonl, graded.jsonl, summary.json, blocks.json,
stats.jsonl with each branch's engine statistics, server logs), and per_task_tf125.csv, per_task_tf060.csv,
branches.csv and blocks.csv with the columns of data/ (without the columns that compare texts with other runs);
then prints stats.py's comparison tf125 vs tf060. Task text and model output are never printed.

Environment: as native_engine.py, and
  SANDBOX_CPUS   cores for grading containers (default 0-4,10-14: the GB10's Cortex-A725 cores, away from the
                 server's 5-9,15-19; time limits in graders depend on the core type). Empty: not pinned.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import native_engine as E

ROOT = Path(__file__).resolve().parent.parent
NAME = {'new': 'tf125', 'old': 'tf060'}
FIRST = 'new,old,old,new,old,new,new,old'
CPUS = os.environ.get('SANDBOX_CPUS', '0-4,10-14')
KEEP = ('engine', 'tokens_per_second', 'seconds', 'prefill_seconds', 'time_to_first_token', 'sampling', 'drafts',
        'token_sha256', 'reasoning_effort', 'enable_thinking', 'accepted_per_step', 'acceptance', 'draft_tokens',
        'accepted_tokens', 'steps')
common = RM = None
STATS = []                     # one row per request: task, seed, engine statistics, times, tokens
_stats_lock = threading.Lock()
_local = threading.local()
_current = {'task': None}


def setup(strategies):
    """Import Part 4's common and run_method; record each streamed request's engine statistics (common.stream_chat
    reads them but keeps only token ids): a json shim copies every 'tensorfold' object, so the request path stays
    common.stream_chat itself, for both engines."""
    global common, RM
    common = E.patch(strategies)
    import run_method
    RM = run_method
    import json as _json

    class JsonShim:
        def __getattr__(self, name):
            return getattr(_json, name)

        @staticmethod
        def loads(s, *a, **k):
            x = _json.loads(s, *a, **k)
            if isinstance(x, dict) and isinstance(x.get('tensorfold'), dict) and hasattr(_local, 'stats'):
                _local.stats.update({k2: v for k2, v in x['tensorfold'].items() if k2 in KEEP})
            return x

    common.json = JsonShim()
    RM.stream_chat = stream_chat_recorded
    if CPUS and '--cpuset-cpus' not in common.SANDBOX:
        common.SANDBOX += ['--cpuset-cpus', CPUS]
    grade_all = common.grade_all

    def grade_all_chunked(items, workdir, chunk=50):
        """Part 4's grade_all, 50 items per container, so a large run cannot reach the per-container time limit."""
        out = {}
        for i in range(0, max(len(items), 1), chunk):
            if items[i:i + chunk]:
                out.update(grade_all(items[i:i + chunk], Path(workdir) / f'part{i // chunk:03d}'))
        return out

    common.grade_all = grade_all_chunked
    return common, RM


def stream_chat_recorded(body, t_zero, cancel=None, deadline=None):
    task = _current['task']
    _local.stats = {}
    r = common.stream_chat(body, t_zero, cancel=cancel, deadline=deadline)
    with _stats_lock:
        STATS.append({'task_id': task, 'seed': body.get('seed'), 'stats': dict(_local.stats),
                      'sent': r.get('sent'), 'first': r.get('first'), 'end': r.get('end'),
                      'completion_tokens': r.get('completion_tokens'), 'prompt_tokens': r.get('prompt_tokens'),
                      'cancelled': bool(r.get('cancelled')), 'error': r.get('error')})
    del _local.stats
    return r


class MemLow:
    """The lowest MemAvailable (GiB) seen while it runs (report only; Part 4's MemoryGuard does the stopping)."""

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
    """run_method.run_parallel (CL4) with a watchdog: a task that outlives its limit by 10 minutes stops the block."""
    box = {}

    def go():
        try:
            box['rec'] = RM.run_parallel(task, 4, extra={'reasoning_effort': 'low'})
        except Exception as exc:  # noqa: BLE001 - the block fails and is reported
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


def run_block(engine, tasks, out, group, f, guard):
    info = {'engine': engine, 'block': group, 'tasks': [t['id'] for t in tasks], 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    low = MemLow()
    done = 0
    try:
        info['server'] = E.start(engine, 4, out / f'server_start_b{group}.log', context=CONTEXT)
        if engine == 'new':
            info['sockets_after_start'] = E.outside_sockets()
        warm = common.stream_chat(common.body_for([{'role': 'user', 'content': 'Compute 2+3. Reply exactly FINAL: 5.'}], 64,
                                                  extra={'chat_template_kwargs': {'enable_thinking': False}}), time.perf_counter())
        assert 'FINAL: 5' in warm.get('answer', ''), 'warm-up reply'
        for task in tasks:
            if guard.fired:
                raise RuntimeError('memory guard fired')
            waited = E.wait_idle()
            _current['task'] = task['id']
            rec = run_task(task)
            rec.update(task_id=task['id'], category=task['category'], method='CL', waited_idle_s=waited,
                       max_tokens=RM.MAX_TOKENS, engine=engine, block=group)
            f.write(json.dumps(rec) + '\n')
            f.flush()
            done += 1
            print(json.dumps({'engine': NAME[engine], 'block': group, 'task': task['id'], 'wall_s': round(rec['wall_s'] or 0, 1),
                              'tokens': rec.get('completion_tokens'), 'timed_out': rec.get('timed_out'), 'rule': rec.get('rule')}),
                  flush=True)
        if engine == 'new':
            info['sockets_after_block'] = E.outside_sockets()
        info['ok'] = True
    except Exception as exc:  # noqa: BLE001 - recorded; the next block runs
        info['ok'] = False
        info['error'] = repr(exc)[:500]
        print(f'FAILED {NAME[engine]} block {group}: {exc!r}'[:400], flush=True)
    finally:
        common.stop_server()
        info['done'] = done
        info['mem_available_low_gib'] = low.close()
        info['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    return info


def write(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(header)
        w.writerows(rows)


def write_tables(base, tasks, first):
    """The data/ columns, from both engines' graded runs. Tasks a failed block never ran count as wrong at the time
    limit (the plan's rule), with empty branch columns."""
    limit = RM.LIMIT_S
    graded = {e: {r['task_id']: r for r in map(json.loads, (base / NAME[e] / 'graded.jsonl').read_text().splitlines())}
              if (base / NAME[e] / 'graded.jsonl').exists() else {} for e in NAME}
    group_of = {t['id']: ('coding_cpp' if t['grader']['type'] == 'cpp_tests' else 'coding_python')
                if t['category'] == 'coding' else 'reasoning' for t in tasks}
    for e in NAME:
        rows = []
        for t in tasks:
            r = graded[e].get(t['id'])
            if r is None:
                rows.append([t['id'], t['category'], group_of[t['id']], '', '', 0, f'{limit:.3f}', '', '', '', '', '', 'not_run', ''])
                continue
            br = r['branches']
            rows.append([t['id'], t['category'], group_of[t['id']], r['block'], int(first[r['block']] == e), int(bool(r['passed'])),
                         f'{min(r["wall_s"], limit):.3f}', r['completion_tokens'],
                         sum(not b.get('cancelled') and not b.get('error') for b in br), sum(bool(b.get('cancelled')) for b in br),
                         sum(b.get('finish') == 'length' for b in br), int(any(b.get('passed') for b in br)), r['rule'],
                         int(bool(r.get('timed_out')))])
        write(base / f'per_task_{NAME[e]}.csv', ['task_id', 'category', 'group', 'block', 'ran_first_in_block', 'passed', 'wall_s',
                                                 'output_tokens', 'finished_branches', 'cancelled_branches',
                                                 'branches_hit_output_limit', 'any_branch_correct', 'vote_rule', 'timed_out'], rows)
    rows = []
    for e in NAME:
        sp = base / NAME[e] / 'stats.jsonl'
        st = {(s['task_id'], s['seed'] - 1000): s for s in map(json.loads, sp.read_text().splitlines())} if sp.exists() else {}
        for t in tasks:
            r = graded[e].get(t['id'])
            for b in (r or {}).get('branches') or []:
                s = st.get((t['id'], b['branch']), {})
                fin = not b.get('cancelled') and not b.get('error')
                span = (s['end'] - s['first']) if s.get('first') is not None and s.get('end') is not None else None
                eng = (s.get('stats') or {}).get('tokens_per_second')
                rows.append([t['id'], NAME[e], b['branch'], 1000 + b['branch'], int(bool(b.get('cancelled'))), b.get('finish') or '',
                             b.get('completion_tokens') or '', s.get('prompt_tokens') or '', f'{b["end_s"]:.3f}' if b.get('end_s') is not None else '',
                             f'{s["first"] - s["sent"]:.6f}' if s.get('first') is not None and s.get('sent') is not None else '',
                             f'{b["completion_tokens"] / span:.6f}' if fin and b.get('completion_tokens') and span and span > 0 else '',
                             repr(eng) if isinstance(eng, (int, float)) and fin else '', int(bool(b.get('passed'))) if fin else '',
                             int(r['chosen'] == b['branch'])])
    write(base / 'branches.csv', ['task_id', 'engine', 'branch', 'seed', 'cancelled', 'finish', 'output_tokens', 'prompt_tokens',
                                  'end_s', 'first_token_s', 'tok_s_from_wall', 'tok_s_engine', 'passed', 'chosen'], rows)
    rows = []
    for e in NAME:
        bp = base / NAME[e] / 'blocks.json'
        for x in json.loads(bp.read_text()) if bp.exists() else []:
            rows.append([x['position'], x['block'], NAME[e], int(first[x['block']] == e), len(x['tasks']), x['done'], int(x['ok']),
                         (x.get('server') or {}).get('startup_s', ''), x.get('mem_available_low_gib', '')])
    rows.sort()
    write(base / 'blocks.csv', ['position', 'group', 'engine', 'ran_first_in_group', 'tasks', 'tasks_done', 'ok', 'startup_s',
                                'mem_available_low_gib'], rows)


CONTEXT = 81920


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('strategies', help="Part 4's code/strategies folder")
    ap.add_argument('run_id')
    ap.add_argument('--split', default='confirm', help='TASKS_DIR/<split>.jsonl')
    ap.add_argument('--tasks', default='', help='only these task ids (comma-separated)')
    ap.add_argument('--first', default=FIRST, help='per group, the engine that runs first (new = 1.2.5, old = 0.6.0)')
    ap.add_argument('--limit-min', type=float, default=40)
    ap.add_argument('--max-tokens', type=int, default=65536)
    a = ap.parse_args()
    setup(a.strategies)
    RM.LIMIT_S = int(a.limit_min * 60)
    RM.MAX_TOKENS = a.max_tokens
    first = a.first.split(',')
    assert first and set(first) <= {'new', 'old'}, '--first takes new/old per group'
    tasks = common.load_tasks(a.split)
    if a.tasks:
        tasks = [t for t in tasks if t['id'] in a.tasks.split(',')]
    size = math.ceil(len(tasks) / len(first))
    groups = [tasks[g * size:(g + 1) * size] for g in range(len(first))]
    assert all(groups), f'{len(tasks)} tasks cannot fill {len(first)} groups'
    base = ROOT / 'results' / a.run_id
    outs = {e: base / NAME[e] for e in NAME}
    for o in outs.values():
        o.mkdir(parents=True, exist_ok=False)
    order = []
    for g, e in enumerate(first):
        order += [(g, e), (g, 'old' if e == 'new' else 'new')]
    print(json.dumps({'tasks': len(tasks), 'groups': [len(x) for x in groups],
                      'blocks': [f'{NAME[e]}:{g}' for g, e in order], 'release': E.check_release()}), flush=True)
    files = {e: open(outs[e] / 'requests.jsonl', 'w') for e in outs}
    blocks = {e: [] for e in outs}
    guard = common.MemoryGuard()
    try:
        for n, (g, e) in enumerate(order):
            STATS.clear()
            info = run_block(e, groups[g], outs[e], g, files[e], guard)
            info['position'] = n
            blocks[e].append(info)
            with open(outs[e] / 'stats.jsonl', 'a') as sf:
                sf.write(''.join(json.dumps({**s, 'block': g}) + '\n' for s in STATS))
            (outs[e] / 'blocks.json').write_text(json.dumps(blocks[e], indent=1))
            if guard.fired:
                print('FAILED memory guard fired: stopping', flush=True)
                break
    finally:
        guard.close()
        common.stop_server()
        for fh in files.values():
            fh.close()
    for e, o in outs.items():
        run = {'method': 'CL', 'k': 4, 'split': a.split, 'engine': NAME[e], 'limit_s': RM.LIMIT_S, 'max_tokens': RM.MAX_TOKENS,
               'context': CONTEXT, 'order': order, 'tasks': [t['id'] for t in tasks], 'memory_guard_fired': guard.fired}
        (o / 'run.json').write_text(json.dumps(run, indent=1))
        ran = {json.loads(l)['task_id'] for l in (o / 'requests.jsonl').read_text().splitlines()}
        if ran:
            try:
                RM.grade_run(o, [t for t in tasks if t['id'] in ran])
            except Exception as exc:  # noqa: BLE001
                print(f'FAILED grading {NAME[e]}: {exc!r}', flush=True)
    write_tables(base, tasks, first)
    st = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / 'stats.py'),
                         f'tf125={base / "per_task_tf125.csv"}', f'tf060={base / "per_task_tf060.csv"}'],
                        capture_output=True, text=True)
    (base / 'stats.json').write_text(st.stdout)
    print(st.stdout or st.stderr, flush=True)


if __name__ == '__main__':
    main()
