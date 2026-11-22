"""This study's comparison on your own tasks: two or more configurations of Qwen3.8-Flash-Next in alternating blocks,
one reply per task with Part 4's run_method.run_single (D2 = low thinking by default), greedy.

    python3 run_blocks.py <part4>/code/strategies <run_id> --arm NAME=SPEC [--arm NAME=SPEC ...]
                          [--orders "A,B;B,A;B,A;A,B;A,B;B,A;B,A;A,B"] [--split confirm] [--tasks id,id]
                          [--method D2] [--limit-min 120] [--max-tokens 65536]

  SPEC  gguf:<first .gguf shard>[:mtp=N]   llama.cpp (llama_cpp.py), with Unsloth's MTP head and up to N drafts
        tensorfold:<Part 6 code folder>    TensorFold 0.6.0 + NVFP4 with Part 6's flash_next.py (MTP threshold 0.6,
                                           up to 6 drafts, unless FN_EXTRA says otherwise)

The tasks, in their file order, form one group per entry of --orders (nearly equal sizes); every arm runs every group
in the order given, so each group is len(arms) blocks, and the server is started fresh for every block (each block runs
in its own process; before each start the runner waits until nvidia-smi lists no process, at most 300 s). Two failed
blocks in a row stop the run. Without --orders: PLAN_18c's order for two arms (AB BA BA AB AB BA BA AB), or a
rotation for more.

This study:
  PLAN_18   --arm IQ3_S=gguf:... --arm UD-IQ3_XXS=gguf:... --arm UD-IQ4_XS=gguf:...
            --orders "IQ3_S,UD-IQ3_XXS,UD-IQ4_XS;UD-IQ3_XXS,UD-IQ4_XS,IQ3_S;UD-IQ4_XS,IQ3_S,UD-IQ3_XXS;
                      IQ3_S,UD-IQ4_XS,UD-IQ3_XXS;UD-IQ3_XXS,IQ3_S,UD-IQ4_XS;UD-IQ4_XS,UD-IQ3_XXS,IQ3_S;
                      IQ3_S,UD-IQ3_XXS,UD-IQ4_XS;UD-IQ4_XS,IQ3_S,UD-IQ3_XXS"
  PLAN_18c  --arm UD-IQ3_XXS_MTP=gguf:...:mtp=4 --arm TensorFold_NVFP4=tensorfold:<Part 6>/code   (default order)

Writes <this folder>/results/<run_id>/: one folder per arm (requests.jsonl, graded.jsonl, summary.json, stats.jsonl
with llama.cpp's timings and draft counts per request, blocks.jsonl, server logs), per_task_<arm>.csv with the
columns of data/per_task_400/ that a new run has, blocks.csv, same_text.csv (0/1 per task and pair), and stats.py's
comparison of every pair. Task text and model output are never printed.

Environment: as llama_cpp.py (LLAMA_SERVER, FN_TEMPLATE, MTP_HEAD) and Part 6's flash_next.py (HF_DIR, ...), and
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
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import llama_cpp as L  # noqa: E402

SANDBOX_CPUS = os.environ.get('SANDBOX_CPUS', '0-4,10-14')
PLAN18C_ORDER = 'A,B;B,A;B,A;A,B;A,B;B,A;B,A;A,B'
WARM = [{'role': 'user', 'content': 'Compute 2+3. Reply exactly FINAL: 5.'}]


def parse_arm(text):
    name, spec = text.split('=', 1)
    kind, _, rest = spec.partition(':')
    if kind == 'gguf':
        path, mtp = rest, 0
        if ':mtp=' in rest:
            path, n = rest.rsplit(':mtp=', 1)
            mtp = int(n)
        return name, {'kind': 'gguf', 'model': str(Path(path).resolve()), 'mtp_n': mtp}
    if kind == 'tensorfold':
        return name, {'kind': 'tensorfold', 'part6_code': str(Path(rest).resolve())}
    raise SystemExit(f'--arm {text}: SPEC must start with gguf: or tensorfold:')


def engine(arm, strategies):
    """Patch Part 4's common for this arm's engine (in this process); returns (common, run_method, start)."""
    if arm['kind'] == 'gguf':
        common = L.patch(strategies)

        def start(log):
            return L.start(arm['model'], log, mtp_n=arm['mtp_n'])
    else:
        sys.path.insert(0, arm['part6_code'])
        import flash_next
        common = flash_next.patch(strategies)

        def start(log):
            waited = L.wait_gpu_free()
            s = common.start_server(1, log, context=L.CONTEXT)
            s['gpu_release_wait_s'] = waited
            return s
    import run_method
    return common, run_method, start


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


def one_block(spec):
    """Child process: one server start, one group of tasks, for one arm. Exit 0 ok, 2 failed, 3 memory guard."""
    out = Path(spec['out'])
    L.PIDFILE = out / 'llama-server.pid'
    common, RM, start = engine(spec['arm'], spec['strategies'])
    RM.LIMIT_S, RM.MAX_TOKENS = spec['limit_s'], spec['max_tokens']
    rec = L.Recorder(common, RM)
    tasks = {t['id']: t for t in common.load_tasks(spec['split'])}
    part = [tasks[i] for i in spec['tasks']]
    info = {'arm': spec['name'], 'group': spec['group'], 'position': spec['position'], 'tasks': spec['tasks'],
            'started': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    low, guard, done = L.MemLow(), common.MemoryGuard(), 0
    try:
        info['server'] = start(out / f'server_start_g{spec["group"]}.log')
        if spec['arm']['kind'] == 'gguf':
            info['sockets_after_start'] = L.outside_sockets(info['server']['pid'])
        warm_up(common)
        with open(out / 'requests.jsonl', 'a') as f:
            for task in part:
                if guard.fired:
                    raise RuntimeError('memory guard fired')
                waited = common.wait_idle()
                rec.task = task['id']
                r = run_with_watchdog(lambda: RM.run_single(task, spec['method']), RM.LIMIT_S, common.stop_server)
                r.pop('token_ids', None)
                r.update(task_id=task['id'], category=task['category'], method=spec['method'], waited_idle_s=waited,
                         max_tokens=RM.MAX_TOKENS, arm=spec['name'], group=spec['group'])
                f.write(json.dumps(r) + '\n')
                f.flush()
                done += 1
                print(json.dumps({'arm': spec['name'], 'group': spec['group'], 'task': task['id'],
                                  'wall_s': round(r['wall_s'] or 0, 1), 'tokens': r.get('completion_tokens'),
                                  'timed_out': r.get('timed_out')}), flush=True)
        info['ok'] = True
    except Exception as exc:  # noqa: BLE001 - recorded; the parent decides whether to go on
        info['ok'], info['error'] = False, repr(exc)[:500]
        print(f'FAILED {spec["name"]} group {spec["group"]}: {exc!r}'[:400], flush=True)
    finally:
        common.stop_server()
        guard.close()
        info.update(done=done, mem_available_low_gib=low.close(), memory_guard_fired=guard.fired,
                    finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    with open(out / 'stats.jsonl', 'a') as sf:
        sf.write(''.join(json.dumps({**s, 'group': spec['group']}) + '\n' for s in rec.rows))
    with open(out / 'blocks.jsonl', 'a') as bf:
        bf.write(json.dumps(info) + '\n')
    sys.exit(3 if guard.fired else (0 if info['ok'] else 2))


def write(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(header)
        w.writerows(rows)


def loop_detected(r):
    """PLAN_18's automatic loop label: the reply hit the output limit and its last 8,000 characters compress (zlib)
    to less than 10% of their size."""
    tail = (r.get('text') or '')[-8000:]
    return int(r.get('finish') == 'length' and bool(tail) and len(zlib.compress(tail.encode())) / len(tail.encode()) < 0.10)


def f3(x):
    return '' if x is None else f'{x:.3f}'


def jl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines()] if Path(p).exists() else []


def write_tables(base, names, tasks, limit_s):
    """The data/per_task_400 columns a new run has; tasks a failed block never ran count as wrong at the time limit."""
    graded = {n: {r['task_id']: r for r in jl(base / n / 'graded.jsonl')} for n in names}
    for n in names:
        order = {r['task_id']: i + 1 for i, r in enumerate(jl(base / n / 'requests.jsonl'))}
        st = {s['task_id']: s for s in jl(base / n / 'stats.jsonl')}
        rows = []
        for t in tasks:
            r = graded[n].get(t['id'])
            if r is None:
                rows.append([t['id'], t['category'], '', '', 0, f'{limit_s:.3f}', '', 'not_run', '', '', '', '', '', ''])
                continue
            s = st.get(t['id']) or {}
            x = s.get('stats') or {}
            dec = x['predicted_n'] / x['predicted_ms'] * 1000 if x.get('predicted_ms') else None
            rows.append([t['id'], t['category'], r['group'], order.get(t['id'], ''), int(bool(r['passed'])),
                         f3(min(r['wall_s'], limit_s)), r.get('completion_tokens'), r.get('finish') or '',
                         int(bool(r.get('timed_out'))), loop_detected(r), f3(s.get('first')), f3(dec),
                         x.get('draft_n', ''), x.get('draft_n_accepted', '')])
        write(base / f'per_task_{n}.csv', ['task_id', 'category', 'block', 'run_order', 'passed', 'wall_s', 'output_tokens',
                                           'finish', 'timed_out', 'loop_detected', 'first_token_s', 'decode_tok_s',
                                           'draft_tokens', 'draft_accepted'], rows)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    write(base / 'same_text.csv', ['task_id'] + [f'{a}__vs__{b}' for a, b in pairs],
          [[t['id']] + [int(t['id'] in graded[a] and t['id'] in graded[b] and
                            (graded[a][t['id']].get('text') or '') == (graded[b][t['id']].get('text') or '')) for a, b in pairs]
           for t in tasks])
    rows = []
    for n in names:
        for b in jl(base / n / 'blocks.jsonl'):
            s = b.get('server') or {}
            rows.append([b['position'], b['group'], n, len(b['tasks']), b['done'], int(b['ok']), s.get('startup_s', ''),
                         s.get('gpu_release_wait_s', ''), b.get('mem_available_low_gib', ''),
                         '' if s.get('draft_loaded') is None else int(s['draft_loaded'])])
    rows.sort()
    write(base / 'blocks.csv', ['position', 'block', 'arm', 'tasks', 'tasks_done', 'ok', 'startup_s', 'gpu_release_wait_s',
                                'mem_available_low_gib', 'draft_loaded'], rows)
    return pairs


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
    return common


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--block':
        one_block(json.loads(sys.argv[2]))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument('strategies', help="Part 4's code/strategies folder")
    ap.add_argument('run_id')
    ap.add_argument('--arm', action='append', required=True, help='NAME=gguf:<file>[:mtp=N] or NAME=tensorfold:<Part 6 code>')
    ap.add_argument('--orders', default='', help='per group, the arms in running order: "A,B;B,A;..."')
    ap.add_argument('--split', default='confirm', help='TASKS_DIR/<split>.jsonl')
    ap.add_argument('--tasks', default='', help='only these task ids (comma-separated)')
    ap.add_argument('--method', default='D2', help="run_method.run_single method (D2 = reasoning_effort low)")
    ap.add_argument('--limit-min', type=float, default=120)
    ap.add_argument('--max-tokens', type=int, default=65536)
    a = ap.parse_args()
    arms = dict(parse_arm(x) for x in a.arm)
    names = list(arms)
    assert len(names) >= 2, 'give at least two --arm'
    if a.orders:
        orders = [g.split(',') for g in a.orders.replace(' ', '').replace('\n', '').split(';') if g]
    elif len(names) == 2:
        orders = [[names[0] if x == 'A' else names[1] for x in g.split(',')] for g in PLAN18C_ORDER.split(';')]
    else:
        orders = [names[i:] + names[:i] for i in range(len(names))] * 2
    assert all(sorted(g) == sorted(names) for g in orders), '--orders must list every arm once per group'
    sys.path.insert(0, str(Path(a.strategies).resolve()))
    import common
    tasks = common.load_tasks(a.split)
    if a.tasks:
        tasks = [t for t in tasks if t['id'] in a.tasks.split(',')]
    size = math.ceil(len(tasks) / len(orders))
    groups = [tasks[g * size:(g + 1) * size] for g in range(len(orders))]
    assert all(groups), f'{len(tasks)} tasks cannot fill {len(orders)} groups'
    base = ROOT / 'results' / a.run_id
    for n in names:
        (base / n).mkdir(parents=True, exist_ok=False)
    seq = [(g, n) for g, order in enumerate(orders) for n in order]
    print(json.dumps({'tasks': len(tasks), 'groups': [len(x) for x in groups], 'blocks': [f'{n}:{g}' for g, n in seq],
                      'arms': arms}), flush=True)
    fails, stopped = 0, None
    for pos, (g, n) in enumerate(seq):
        spec = {'strategies': str(Path(a.strategies).resolve()), 'name': n, 'arm': arms[n], 'group': g, 'position': pos,
                'tasks': [t['id'] for t in groups[g]], 'split': a.split, 'method': a.method,
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
    (base / 'run.json').write_text(json.dumps({'arms': arms, 'orders': orders, 'split': a.split, 'method': a.method,
                                               'limit_s': int(a.limit_min * 60), 'max_tokens': a.max_tokens,
                                               'context': L.CONTEXT, 'tasks': [t['id'] for t in tasks],
                                               'stopped': stopped}, indent=1))
    grade(a.strategies, base, names, tasks)
    pairs = write_tables(base, names, tasks, int(a.limit_min * 60))
    out = {}
    for x, y in pairs:
        st = subprocess.run([sys.executable, str(HERE / 'stats.py'), '--limit-min', str(a.limit_min),
                             f'{x}={base / f"per_task_{x}.csv"}', f'{y}={base / f"per_task_{y}.csv"}'],
                            capture_output=True, text=True)
        out[f'{x} vs {y}'] = json.loads(st.stdout) if st.returncode == 0 else {'error': st.stderr[-2000:]}
    (base / 'stats.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == '__main__':
    main()
