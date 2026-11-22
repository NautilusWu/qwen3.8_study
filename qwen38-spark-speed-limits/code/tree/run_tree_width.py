"""PLAN_16 B0 and PLAN_22 M2: a lone stream's verification-tree width on TensorFold 0.6.0 (default 16 rows).

    python3 run_tree_width.py <part4>/code/strategies <out_dir> [--order 16,32,48,64,96,128,128,96,64,48,32,16]

Uses Part 4's public code (common.py and run_method.py: the server slot, streaming, the 5 public example tasks in
<part4>/examples/tasks/dev.jsonl). Each start: `tensorfold serve` with --parallel 1, the original checkpoint and
DFlash2 drafter, the per-round hook (../profile/hook/serve_hooked.py) and serve_rows.py, which sets the one constructor
argument tree_rows (nothing on disk changes). Then the warm-up, three public prompts (a long thinking reply, 8,192-token
cap; code and chat, 1,024 tokens, thinking off) and the 5 example tasks as D2 (65,536-token cap). Every reply's token
ids must be identical across all starts (verification is exact); otherwise the run stops.
PLAN_16 B0 used the order 16,24,32,32,24,16; PLAN_22 M2 the default order above.

Writes <out>/runs.json, <out>/summary.json and per start <run>.rounds.jsonl and <run>.server.log. Speed: decode tokens /
decode seconds from the hook's per-request trace (prefill excluded), tokens per round, ms per round, stage times.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOOK = HERE.parent / 'profile' / 'hook'
CONTEXT = 81920
OFF = {'chat_template_kwargs': {'enable_thinking': False}}
CODE = 'Write a Python function merge_intervals(intervals) that merges overlapping intervals and returns them sorted. Include a docstring and two example calls.'
CHAT = 'Explain how a hash table handles collisions, comparing chaining and open addressing, in about 300 words.'
LONG = 'How many positive integers below 1,000,000 have digits that sum to 27? Explain your reasoning.'
ITEMS = (('long', LONG, 8192, None), ('code', CODE, 1024, OFF), ('chat', CHAT, 1024, OFF))


def sha(x):
    return hashlib.sha256(json.dumps(x).encode()).hexdigest()


def server_args(common, rows, out, run):
    serve = ['serve', common.TARGET, '--drafter', common.DRAFT, '--backend', 'cuda', '--name', common.MODEL,
             '--host', '127.0.0.1', '--port', str(common.PORT), '--parallel', '1', '--context', str(CONTEXT),
             '--temperature', '0', '--top-p', '1', '--top-k', '0', '--thinking', '--prefill-fp8']
    return ['docker', 'run', '-d', '--name', common.NAME, '--gpus', 'all', '--ipc', 'host', '--network', 'host',
            '--cpuset-cpus', '5-9,15-19', '--memory', '80g', '--memory-swap', '80g',
            '--ulimit', 'memlock=-1', '--ulimit', 'stack=67108864',
            '-e', 'TENSORFOLD_KV_DTYPE=fp8', '-e', 'TENSORFOLD_MEMORY_RESERVE_GIB=40', '-e', 'TENSORFOLD_KV_POOL_GIB=0',
            '-e', 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',
            '-e', 'P14_HOOK=1', '-e', 'P14_NVTX_DETAIL=0', '-e', f'P14_ROUNDS=/out/{run}.rounds.jsonl',
            '-e', f'P16_TREE_ROWS={rows}',
            '-v', f'{common.HF_DIR}:/hf:ro', '-v', f'{common.CACHE}:/cache', '-v', f'{HOOK}:/p14:ro', '-v', f'{HERE}:/p16:ro',
            '-v', f'{out}:/out', common.IMAGE, 'python', '/p16/serve_rows.py'] + serve


def gpu_free(timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                              capture_output=True, text=True).stdout.strip():
            return True
        time.sleep(2)
    return False


def start(common, rows, out, run):
    common.stop_server()
    assert gpu_free(), 'GPU is in use'
    f = out / f'{run}.rounds.jsonl'
    f.touch()
    f.chmod(0o666)
    args = server_args(common, rows, out, run)
    subprocess.run(args, check=True, capture_output=True)
    t0 = time.time()
    while True:
        try:
            common.health()
            break
        except OSError:
            running = subprocess.run(['docker', 'inspect', '-f', '{{.State.Running}}', common.NAME],
                                     capture_output=True, text=True).stdout.strip()
            if running != 'true' or time.time() - t0 > 1800:
                tail = subprocess.run(['docker', 'logs', '--tail', '60', common.NAME], capture_output=True, text=True)
                raise RuntimeError('server did not start: ' + tail.stdout[-3000:] + tail.stderr[-3000:])
            time.sleep(3)
    return {'startup_s': round(time.time() - t0, 1), 'argv': args}


def one(common, text, max_tokens, extra):
    msgs = [{'role': 'system', 'content': common.SYSTEM}, {'role': 'user', 'content': text}]
    r = common.stream_chat(common.body_for(msgs, max_tokens, extra=extra), time.perf_counter())
    return {'completion_tokens': r.get('completion_tokens'), 'finish': r.get('finish'), 'error': r.get('error'),
            'token_sha': sha(r.get('token_ids')) if r.get('token_ids') is not None else None}


def warm(common):
    r = common.stream_chat(common.body_for([{'role': 'user', 'content': 'Compute 2+3. Reply exactly FINAL: 5.'}], 64, extra=OFF),
                           time.perf_counter())
    assert 'FINAL: 5' in r.get('answer', ''), 'warm-up reply'


def decode_stats(out, run):
    recs = [json.loads(l) for l in (out / f'{run}.rounds.jsonl').read_text().splitlines() if '"single"' in l]
    tok = sum(r['tokens'] for r in recs)
    sec = sum(r['seconds'] for r in recs)
    rounds = sum(r['rounds'] for r in recs)
    stage = {k: round(1000 * sum(r['stage_s'][k] for r in recs) / rounds, 3) for k in ('draft', 'verify', 'sample', 'commit')} if rounds else {}
    return {'requests': len(recs), 'tokens': tok, 'seconds': round(sec, 3), 'rounds': rounds,
            'tok_s': round(tok / sec, 3) if sec else None, 'tokens_per_round': round(tok / rounds, 3) if rounds else None,
            'ms_per_round': round(1000 * sec / rounds, 3) if rounds else None, 'stage_ms_per_round': stage}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('strategies')
    ap.add_argument('out')
    ap.add_argument('--order', default='16,32,48,64,96,128,128,96,64,48,32,16')
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.strategies).resolve()))
    import common
    import run_method as RM
    examples = Path(a.strategies).resolve().parent.parent / 'examples' / 'tasks' / 'dev.jsonl'
    tasks = [json.loads(l) for l in examples.read_text().splitlines() if l.strip()]
    RM.LIMIT_S, RM.MAX_TOKENS = 40 * 60, 65536
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    order = [int(x) for x in a.order.split(',')]
    ref, runs, stop = {}, [], False
    for n, rows in enumerate(order):
        run = f'w{rows}-{n}'
        rec = {'run': run, 'rows': rows}
        guard = common.MemoryGuard()
        items = []
        try:
            rec['server'] = start(common, rows, out, run)
            warm(common)
            for name, text, mt, extra in ITEMS:
                common.wait_idle()
                items.append({'item': name, **one(common, text, mt, extra)})
            for t in tasks:
                common.wait_idle()
                r = RM.run_single(t, 'D2')
                items.append({'item': t['id'], 'completion_tokens': r.get('completion_tokens'), 'finish': r.get('finish'),
                              'token_sha': sha(r.get('token_ids'))})
            rec['ok'] = True
        except Exception as exc:  # noqa: BLE001
            rec['ok'], rec['error'] = False, repr(exc)[:600]
            print(f'FAILED {run}: {exc!r}'[:500], flush=True)
        finally:
            logs = subprocess.run(['docker', 'logs', common.NAME], capture_output=True, text=True)
            (out / f'{run}.server.log').write_text(logs.stdout + logs.stderr)
            rec['p16_line'] = [l for l in (logs.stdout + logs.stderr).splitlines() if l.startswith('[p16]')]
            common.stop_server()
            rec['mem_guard_fired'] = guard.fired
            guard.close()
        rec['items'] = items
        rec['differs'] = [it['item'] for it in items if ref.setdefault(it['item'], it['token_sha']) != it['token_sha']]
        rec['decode'] = decode_stats(out, run)
        runs.append(rec)
        print(json.dumps({'run': run, 'ok': rec['ok'], 'differs': rec['differs'], **{k: rec['decode'][k] for k in
                          ('tokens', 'tok_s', 'tokens_per_round', 'ms_per_round')}}), flush=True)
        (out / 'runs.json').write_text(json.dumps(runs, indent=1))
        if not rec['ok'] or rec['differs'] or rec['mem_guard_fired']:
            stop = True
            print('STOP: a start failed or a reply changed with the tree width', flush=True)
            break
    summary = {'stopped': stop, 'by_rows': {}}
    for rows in sorted(set(order)):
        rs = [r for r in runs if r['rows'] == rows and r['ok']]
        if rs:
            summary['by_rows'][rows] = {k: [r['decode'][k] for r in rs] for k in ('tok_s', 'tokens_per_round', 'ms_per_round')}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print('SUMMARY ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
