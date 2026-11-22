"""This study's runs on your own tasks: Qwen3.8-27B and/or Qwen3.8-Flash-Next on TensorFold 0.6.0, in alternating
blocks, with N tasks in flight at once (N = 1 is PLAN_19, N = 4 is PLAN_20, the 27B at N = 2, 4, 8 interleaved is
PLAN_21b).
One reply per task with Part 4's
run_method.run_single (D2 = low thinking by default), greedy.

    python3 run_streams.py <part4>/code/strategies <run_id> --model NAME=SPEC [--model NAME=SPEC]
                           [--streams 4] [--orders "A,B;B,A;B,A;A,B;A,B;B,A;B,A;A,B"] [--shuffle-seed 2026]
                           [--split confirm] [--tasks id,id] [--method D2] [--limit-min 40] [--max-tokens 65536]
                           [--context 81920]

  SPEC  27b                       Part 4's server unchanged: Qwen3.8-27B NVFP4 + DFlash2 drafts, FP8 KV cache
        flashnext:<Part 6 code>   Part 6's flash_next.py: Flash Next NVFP4 with its MTP head (threshold 0.6, up to 6
                                  drafts unless FN_EXTRA says otherwise), bf16 KV cache
        ...@N                     this arm runs N streams instead of --streams (e.g. 27b_p8=27b@8), so one run can
                                  interleave several stream counts (PLAN_21b: three 27B arms, --orders lists all three
                                  per group)

The tasks (in file order, or shuffled with --shuffle-seed) form one group per entry of --orders (nearly equal sizes).
Every model runs every group in the order given, so each group is one block per model, and the server is started
fresh with `--parallel N` for every block (each block runs in its own process; before each start the runner waits
until nvidia-smi lists no process, at most 300 s). Two failed blocks in a row stop the run.

Inside a block:
  N = 1  one task at a time, waiting for the server to be idle before each (PLAN_19)
  N > 1  N workers take tasks from one queue in group order, so N different tasks run at once until the queue is
         empty; the server is waited for once, before the first task (PLAN_20)
Each task has its own time limit from its own request. Every row records start_s / end_s on the block's clock, the
worker, and the order sent; the server's /health is polled every 2 s for the largest `requests_running`.

Writes <this folder>/results/<run_id>/: one folder per model (requests.jsonl, graded.jsonl, blocks.jsonl, server
logs), per_task_<model>.csv (the columns of data/per_task_400, plus shared_s: the PLAN_20 shared time, see
shared_time.py; with N = 1 it is the task's own time), stats_<model>.csv (task_id, passed, wall_s = shared_s, ready for
stats.py, also against another run), blocks.csv, same_text.csv (0/1 for every pair of arms), and with exactly two arms
stats.py's comparison of them (with more arms, run stats.py on the pairs you want).
Task text and model output are never printed.

Grading is Part 4's grade_run (your task files' graders), after all blocks, with the grading containers pinned:
  SANDBOX_CPUS  cores for grading containers (default 0-4,10-14: the GB10's Cortex-A725 cores, away from the
                server's 5-9,15-19; time limits in graders depend on the core type). Empty: not pinned.
Environment otherwise as Part 4 (HF_DIR, TF_CACHE_DIR, TASKS_DIR, GRADER_DIR, SANDBOX_IMAGE) and Part 6 (FN_TARGET,
FN_EXTRA, FN_FLOOR_GIB).
"""
import argparse
import csv
import json
import math
import os
import queue
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import shared_time as ST  # noqa: E402

SANDBOX_CPUS = os.environ.get('SANDBOX_CPUS', '0-4,10-14')
ABBA = 'A,B;B,A;B,A;A,B;A,B;B,A;B,A;A,B'
WARM = [{'role': 'user', 'content': 'Compute 2+3. Reply exactly FINAL: 5.'}]
GPU_WAIT_S = 300


def parse_model(text):
    name, spec = text.split('=', 1)
    extra = {}
    if '@' in spec:
        spec, n = spec.rsplit('@', 1)
        extra = {'streams': int(n)}
    if spec == '27b':
        return name, {'kind': '27b', **extra}
    if spec.startswith('flashnext:'):
        return name, {'kind': 'flashnext', 'part6_code': str(Path(spec.split(':', 1)[1]).resolve()), **extra}
    raise SystemExit(f'--model {text}: SPEC must be 27b or flashnext:<Part 6 code folder>, optionally with @N')


def engine(model, strategies):
    """Part 4's common (patched by Part 6's flash_next.py for Flash Next) and run_method, in this process."""
    sys.path.insert(0, str(Path(strategies).resolve()))
    if model['kind'] == 'flashnext':
        sys.path.insert(0, model['part6_code'])
        import flash_next
        common = flash_next.patch(strategies)
    else:
        import common
    import run_method
    return common, run_method


def gpu_pids():
    return subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                          capture_output=True, text=True).stdout.split()


def wait_gpu_free():
    t0 = time.time()
    while gpu_pids() and time.time() - t0 < GPU_WAIT_S:
        time.sleep(1)
    return round(time.time() - t0, 1)


class MemLow:
    """Lowest MemAvailable (GiB) seen every 5 s."""

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


class Running:
    """Polls the server's /health every 2 s; keeps the largest requests_running seen."""

    def __init__(self, common):
        self.common, self.max, self._stop = common, 0, threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.wait(2):
            try:
                self.max = max(self.max, int(self.common.health().get('requests_running') or 0))
            except Exception:  # noqa: BLE001 - the server may be starting or stopping
                pass

    def close(self):
        self._stop.set()
        return self.max


def run_with_watchdog(fn, limit_s, on_hang):
    box = {}

    def go():
        try:
            box['rec'] = fn()
        except Exception as exc:  # noqa: BLE001 - the block fails and is reported
            box['exc'] = exc

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(limit_s + 600)
    if t.is_alive():
        on_hang()
        raise RuntimeError('task did not finish within its limit + 10 minutes')
    if 'exc' in box:
        raise box['exc']
    return box['rec']


def warm_up(common):
    warm = common.stream_chat(common.body_for(WARM, 64, extra={'chat_template_kwargs': {'enable_thinking': False}}),
                              time.perf_counter())
    assert 'FINAL: 5' in warm.get('answer', ''), 'warm-up reply'


def run_group(common, RM, tasks, path, guard, spec):
    """N workers on one queue in group order (N = 1: one task at a time, idle wait before each)."""
    n_streams = spec['streams']
    todo = queue.Queue()
    for i, t in enumerate(tasks):
        todo.put((i + 1, t))
    lock, state = threading.Lock(), {'n': 0, 'err': None}
    f = open(path, 'a')
    t0 = time.perf_counter()

    def worker(w):
        while state['err'] is None:
            if guard.fired:
                state['err'] = 'memory guard fired'
                return
            try:
                order, task = todo.get_nowait()
            except queue.Empty:
                return
            try:
                waited = common.wait_idle() if n_streams == 1 else None
                start = time.perf_counter() - t0
                rec = run_with_watchdog(lambda: RM.run_single(task, spec['method']), RM.LIMIT_S, common.stop_server)
                end = time.perf_counter() - t0
            except Exception as exc:  # noqa: BLE001
                state['err'] = repr(exc)[:300]
                return
            rec.pop('token_ids', None)
            rec.update(task_id=task['id'], category=task.get('category'), method=spec['method'], max_tokens=RM.MAX_TOKENS,
                       model=spec['name'], group=spec['group'], streams=n_streams, worker=w, send_order=order,
                       waited_idle_s=waited, start_s=round(start, 3), end_s=round(end, 3))
            with lock:
                f.write(json.dumps(rec) + '\n')
                f.flush()
                state['n'] += 1
                print(json.dumps({'model': spec['name'], 'group': spec['group'], 'task': task['id'], 'worker': w,
                                  'wall_s': round(rec['wall_s'] or 0, 1), 'tokens': rec.get('completion_tokens'),
                                  'timed_out': rec.get('timed_out')}), flush=True)

    threads = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(n_streams)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    f.close()
    if state['err']:
        raise RuntimeError(state['err'])
    return state['n'], round(time.perf_counter() - t0, 3)


def one_block(spec):
    """Child process: one server start, one group, one model. Exit 0 ok, 2 failed, 3 memory guard."""
    out = Path(spec['out'])
    common, RM = engine(spec['model'], spec['strategies'])
    RM.LIMIT_S, RM.MAX_TOKENS = spec['limit_s'], spec['max_tokens']
    tasks = {t['id']: t for t in common.load_tasks(spec['split'])}
    part = [tasks[i] for i in spec['tasks']]
    info = {'model': spec['name'], 'group': spec['group'], 'position': spec['position'], 'streams': spec['streams'],
            'tasks': spec['tasks'], 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    low, guard, run, done = MemLow(), common.MemoryGuard(), None, 0
    try:
        info['gpu_release_wait_s'] = wait_gpu_free()
        s = common.start_server(spec['streams'], out / f'server_start_g{spec["group"]}.log', context=spec['context'])
        info['server'] = {k: s.get(k) for k in ('startup_s', 'image_id')}
        warm_up(common)
        common.wait_idle()
        run = Running(common)
        done, info['block_s'] = run_group(common, RM, part, out / 'requests.jsonl', guard, spec)
        info['ok'] = True
    except Exception as exc:  # noqa: BLE001 - recorded; the parent decides whether to go on
        info['ok'], info['error'] = False, repr(exc)[:500]
        print(f'FAILED {spec["name"]} group {spec["group"]}: {exc!r}'[:400], flush=True)
    finally:
        info['max_requests_running'] = run.close() if run else None
        common.stop_server()
        guard.close()
        info.update(done=done, mem_available_low_gib=low.close(), memory_guard_fired=guard.fired,
                    finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    with open(out / 'blocks.jsonl', 'a') as bf:
        bf.write(json.dumps(info) + '\n')
    sys.exit(3 if guard.fired else (0 if info['ok'] else 2))


def write(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(header)
        w.writerows(rows)


def jl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines()] if Path(p).exists() else []


def f3(x):
    return '' if x is None else f'{x:.3f}'


def write_tables(base, names, tasks, limit_s, streams):
    """streams: {arm: N}."""
    """Per-task CSVs (tasks a failed block never ran count as wrong at the time limit), shared time per block."""
    graded = {n: {r['task_id']: r for r in jl(base / n / 'graded.jsonl')} for n in names}
    for n in names:
        shared = {}
        for g in sorted({r['group'] for r in graded[n].values()}):
            rows = [{'task_id': t, 'start_s': r['start_s'], 'end_s': r['end_s'], 'output_tokens': r.get('completion_tokens')}
                    for t, r in graded[n].items() if r['group'] == g]
            shared.update(ST.shares(rows, streams[n])[0])
        rows, srows = [], []
        for t in tasks:
            r = graded[n].get(t['id'])
            if r is None:
                rows.append([t['id'], t.get('category', ''), '', '', '', '', '', 0, f3(limit_s), '', 'not_run', '', f3(limit_s)])
                srows.append([t['id'], 0, f3(limit_s)])
                continue
            sh = min(shared[t['id']], limit_s)
            rows.append([t['id'], t.get('category', ''), r['group'], r['send_order'], r['worker'], f3(r['start_s']),
                         f3(r['end_s']), int(bool(r['passed'])), f3(min(r['wall_s'], limit_s)), r.get('completion_tokens'),
                         r.get('finish') or '', int(bool(r.get('timed_out'))), f3(sh)])
            srows.append([t['id'], int(bool(r['passed'])), f3(sh)])
        write(base / f'per_task_{n}.csv', ['task_id', 'category', 'block', 'send_order', 'worker', 'start_s', 'end_s',
                                           'passed', 'wall_s', 'output_tokens', 'finish', 'timed_out', 'shared_s'], rows)
        write(base / f'stats_{n}.csv', ['task_id', 'passed', 'wall_s'], srows)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    if pairs:
        write(base / 'same_text.csv', ['task_id', *[f'{a}__vs__{b}' for a, b in pairs]],
              [[t['id'], *[int(t['id'] in graded[a] and t['id'] in graded[b] and
                               (graded[a][t['id']].get('text') or '') == (graded[b][t['id']].get('text') or ''))
                           for a, b in pairs]] for t in tasks])
    rows = []
    for n in names:
        for b in jl(base / n / 'blocks.jsonl'):
            s = b.get('server') or {}
            rows.append([b['position'], b['group'], n, b['streams'], len(b['tasks']), b['done'], int(b['ok']),
                         s.get('startup_s', ''), b.get('gpu_release_wait_s', ''), b.get('mem_available_low_gib', ''),
                         f3(b.get('block_s')), b.get('max_requests_running', '')])
    rows.sort()
    write(base / 'blocks.csv', ['position', 'block', 'model', 'streams', 'tasks', 'tasks_done', 'ok', 'startup_s',
                                'gpu_release_wait_s', 'mem_available_low_gib', 'block_s', 'max_requests_running'], rows)


def grade(strategies, base, names, tasks):
    """Part 4's grade_run, with the grading containers pinned (SANDBOX_CPUS) and 50 items per container."""
    sys.path.insert(0, str(Path(strategies).resolve()))
    import common
    import run_method as RM
    if SANDBOX_CPUS and '--cpuset-cpus' not in common.SANDBOX:
        common.SANDBOX += ['--cpuset-cpus', SANDBOX_CPUS]
    plain = common.grade_all

    def grade_all_chunked(items, workdir, chunk=50):
        out = {}
        for i in range(0, max(len(items), 1), chunk):
            if items[i:i + chunk]:
                out.update(plain(items[i:i + chunk], Path(workdir) / f'part{i // chunk:03d}'))
        return out

    common.grade_all = grade_all_chunked
    for n in names:
        ran = {r['task_id'] for r in jl(base / n / 'requests.jsonl')}
        if ran:
            try:
                RM.grade_run(base / n, [t for t in tasks if t['id'] in ran])
            except Exception as exc:  # noqa: BLE001
                print(f'FAILED grading {n}: {exc!r}', flush=True)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--block':
        one_block(json.loads(sys.argv[2]))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument('strategies', help="Part 4's code/strategies folder")
    ap.add_argument('run_id')
    ap.add_argument('--model', action='append', required=True, help='NAME=27b or NAME=flashnext:<Part 6 code folder>')
    ap.add_argument('--streams', type=int, default=4, help='tasks in flight at once, and the server\'s --parallel '
                    '(an arm\'s own @N overrides it)')
    ap.add_argument('--orders', default='', help='per group, the models in running order: "A,B;B,A;..."')
    ap.add_argument('--shuffle-seed', type=int, default=None, help='shuffle the tasks before cutting groups (ours: 2026)')
    ap.add_argument('--split', default='confirm', help='TASKS_DIR/<split>.jsonl')
    ap.add_argument('--tasks', default='', help='only these task ids (comma-separated)')
    ap.add_argument('--method', default='D2', help='run_method.run_single method (D2 = reasoning_effort low)')
    ap.add_argument('--limit-min', type=float, default=40)
    ap.add_argument('--max-tokens', type=int, default=65536)
    ap.add_argument('--context', type=int, default=81920)
    a = ap.parse_args()
    models = dict(parse_model(x) for x in a.model)
    names = list(models)
    streams = {n: models[n].get('streams', a.streams) for n in names}
    if a.orders:
        orders = [g.split(',') for g in a.orders.replace(' ', '').replace('\n', '').split(';') if g]
    elif len(names) == 2:
        orders = [[names[0] if x == 'A' else names[1] for x in g.split(',')] for g in ABBA.split(';')]
    else:
        orders = [names] * 8
    assert all(sorted(g) == sorted(names) for g in orders), '--orders must list every model once per group'
    sys.path.insert(0, str(Path(a.strategies).resolve()))
    import common
    tasks = common.load_tasks(a.split)
    if a.tasks:
        tasks = [t for t in tasks if t['id'] in a.tasks.split(',')]
    order = [t['id'] for t in tasks]
    if a.shuffle_seed is not None:
        random.Random(a.shuffle_seed).shuffle(order)
    size = math.ceil(len(order) / len(orders))
    groups = [order[g * size:(g + 1) * size] for g in range(len(orders))]
    assert all(groups), f'{len(tasks)} tasks cannot fill {len(orders)} groups'
    base = ROOT / 'results' / a.run_id
    for n in names:
        (base / n).mkdir(parents=True, exist_ok=False)
    seq = [(g, n) for g, o in enumerate(orders) for n in o]
    print(json.dumps({'tasks': len(tasks), 'streams': streams, 'groups': [len(x) for x in groups],
                      'blocks': [f'{n}:{g}' for g, n in seq], 'models': models}), flush=True)
    fails, stopped = 0, None
    for pos, (g, n) in enumerate(seq):
        spec = {'strategies': str(Path(a.strategies).resolve()), 'name': n, 'model': models[n], 'group': g, 'position': pos,
                'tasks': groups[g], 'split': a.split, 'method': a.method, 'streams': streams[n], 'context': a.context,
                'limit_s': int(a.limit_min * 60), 'max_tokens': a.max_tokens, 'out': str(base / n)}
        rc = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--block', json.dumps(spec)]).returncode
        fails = 0 if rc == 0 else fails + 1
        if rc == 3:
            stopped = 'memory guard fired'
        elif fails >= 2:
            stopped = 'two blocks failed in a row'
        if stopped:
            print(f'STOPPED: {stopped}', flush=True)
            break
    (base / 'run.json').write_text(json.dumps({'models': models, 'streams': streams, 'orders': orders, 'groups': groups,
                                               'split': a.split, 'method': a.method, 'limit_s': int(a.limit_min * 60),
                                               'max_tokens': a.max_tokens, 'context': a.context, 'stopped': stopped},
                                              indent=1))
    grade(a.strategies, base, names, tasks)
    write_tables(base, names, tasks, int(a.limit_min * 60), streams)
    if len(names) == 2:
        x, y = names
        st = subprocess.run([sys.executable, str(HERE / 'stats.py'), '--limit-min', str(a.limit_min),
                             f'{x}={base / f"stats_{x}.csv"}', f'{y}={base / f"stats_{y}.csv"}'], capture_output=True, text=True)
        out = json.loads(st.stdout) if st.returncode == 0 else {'error': st.stderr[-2000:]}
        (base / 'stats.json').write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1), flush=True)


if __name__ == '__main__':
    main()
