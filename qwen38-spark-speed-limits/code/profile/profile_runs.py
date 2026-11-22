"""PLAN_14 step 1: the runs. Every server is TensorFold 0.6.0 exactly as in the studies (common.start_server's container
and `serve` arguments), started through hook/serve_hooked.py (P14_HOOK=0: plain `tensorfold serve`; 1: the timing
hook), optionally inside `nsys launch`. Only public prompts: the speed microbench's (Part 10) and Part 4's example
tasks. Replies' token ids (texts for the vote's branches) are compared across runs: any difference stops the runs.

    python3 profile_runs.py <prefix> [--compat]

Writes results/<prefix>/: bw.json, <run>.requests.jsonl (one line per reply, no text), <run>.rounds.jsonl (the
hook's records, with the runner's markers), <run>.server.log, <run>.nsys-rep and <run>.sqlite (Nsight runs),
identity.json, ncu_probe.json, runs.json.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
import os
TTS = Path(os.environ['PART4_STRATEGIES'])   # <part4>/code/strategies (Part 4's public common.py and run_method.py)
sys.path.insert(0, str(TTS))
import common  # noqa: E402
import run_method as RM  # noqa: E402

EXAMPLES = TTS.parent.parent / 'examples/tasks/dev.jsonl'      # Part 4's 5 public example tasks
HF = common.HF_DIR
CONTEXT = 81920
SYS = {'role': 'system', 'content': common.SYSTEM}
OFF = {'chat_template_kwargs': {'enable_thinking': False}}
# the speed microbench's public prompts (Part 10's code/microbench.py, copied verbatim; the compat check compares them)
CODE = 'Write a Python function merge_intervals(intervals) that merges overlapping intervals and returns them sorted. Include a docstring and two example calls.'
CHAT = 'Explain how a hash table handles collisions, comparing chaining and open addressing, in about 300 words.'
LONG = 'How many positive integers below 1,000,000 have digits that sum to 27? Explain your reasoning.'
FOUR = ['Write a Python class for a fixed-size ring buffer with push, pop and len, with a docstring.',
        'Describe the differences between TCP and UDP in about 250 words.',
        'Write a C++ function that returns the k smallest elements of a vector, with a short explanation.',
        'Summarize the main ideas of dynamic programming in about 250 words, with one small example.']
SINGLE_ITEMS = {'long': (LONG, 8192, None), 'code': (CODE, 1024, OFF), 'chat': (CHAT, 1024, OFF)}
NSYS = ['nsys', 'launch', '--session-new=p14', '--trace=cuda-sw,nvtx']   # software CUDA trace: hardware trace needs
# GPU counter privileges (the driver restricts them to admins); sampling switches go on `nsys start`


def sha(x):
    return hashlib.sha256(json.dumps(x).encode()).hexdigest()


def msg(text):
    return [SYS, {'role': 'user', 'content': text}]


def server_args(parallel, hook, detail, nsys, out, run):
    """common.start_server's container and serve arguments, plus the hook's mounts and environment."""
    serve = ['serve', common.TARGET, '--drafter', common.DRAFT, '--backend', 'cuda', '--name', common.MODEL,
             '--host', '127.0.0.1', '--port', str(common.PORT), '--parallel', str(parallel), '--context', str(CONTEXT),
             '--temperature', '0', '--top-p', '1', '--top-k', '0', '--thinking', '--prefill-fp8']
    docker = ['docker', 'run', '-d', '--name', common.NAME, '--gpus', 'all', '--ipc', 'host', '--network', 'host',
              '--cpuset-cpus', '5-9,15-19', '--memory', '80g', '--memory-swap', '80g',
              '--ulimit', 'memlock=-1', '--ulimit', 'stack=67108864',
              '-e', 'TENSORFOLD_KV_DTYPE=fp8', '-e', 'TENSORFOLD_MEMORY_RESERVE_GIB=40', '-e', 'TENSORFOLD_KV_POOL_GIB=0',
              '-e', 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',
              '-e', f'P14_HOOK={int(hook)}', '-e', f'P14_NVTX_DETAIL={int(detail)}', '-e', f'P14_ROUNDS=/out/{run}.rounds.jsonl',
              '-v', f'{HF}:/hf:ro', '-v', f'{common.CACHE}:/cache', '-v', f'{HERE / "hook"}:/p14:ro', '-v', f'{out}:/out',
              common.IMAGE]
    return docker + (NSYS if nsys else []) + ['python', '/p14/serve_hooked.py'] + serve, serve


def gpu_free(timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                              capture_output=True, text=True).stdout.strip():
            return True
        time.sleep(2)
    return False


def start(parallel, hook, detail, nsys, out, run):
    common.stop_server()
    assert gpu_free(), 'GPU is in use'
    rounds = out / f'{run}.rounds.jsonl'          # created here, writable by the container's root and by the runner
    rounds.touch()
    rounds.chmod(0o666)
    args, serve = server_args(parallel, hook, detail, nsys, out, run)
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
    return {'startup_s': round(time.time() - t0, 1), 'argv': args, 'serve': serve}


def marker(out, run, **kw):
    with open(out / f'{run}.rounds.jsonl', 'a') as f:
        f.write(json.dumps({'kind': 'marker', 'host_time': time.time(), **kw}) + '\n')


def one(text, max_tokens, extra):
    r = common.stream_chat(common.body_for(msg(text), max_tokens, extra=extra), time.perf_counter())
    span = (r.get('end') or 0) - (r.get('first') or 0)
    return {'completion_tokens': r.get('completion_tokens'), 'prompt_tokens': r.get('prompt_tokens'),
            'ttft_s': round(r['first'] - r['sent'], 4) if r.get('first') is not None else None, 'wall_s': round(r['end'], 3),
            'tok_s': round(r['completion_tokens'] / span, 2) if r.get('completion_tokens') and span > 0 else None,
            'finish': r.get('finish'), 'error': r.get('error'), 'cancelled': bool(r.get('cancelled')),
            'token_sha': sha(r.get('token_ids')) if r.get('token_ids') is not None else None}


def warm():
    r = common.stream_chat(common.body_for([{'role': 'user', 'content': 'Compute 2+3. Reply exactly FINAL: 5.'}], 64, extra=OFF),
                           time.perf_counter())
    assert 'FINAL: 5' in r.get('answer', ''), 'warm-up reply'


def single_workload(out, run, rep, items, rows):
    for item in items:
        common.wait_idle()
        text, mt, extra = SINGLE_ITEMS[item]
        marker(out, run, phase=item, rep=rep)
        rows.append({'run': run, 'rep': rep, 'item': item, **one(text, mt, extra)})
        print(json.dumps({k: rows[-1][k] for k in ('run', 'rep', 'item', 'completion_tokens', 'tok_s')}), flush=True)


def four_workload(out, run, rep, tasks, rows):
    common.wait_idle()
    marker(out, run, phase='four', rep=rep)
    box = [None] * 4
    th = [threading.Thread(target=lambda i=i: box.__setitem__(i, one(FOUR[i], 1024, OFF))) for i in range(4)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    for i, b in enumerate(box):
        rows.append({'run': run, 'rep': rep, 'item': f'four{i}', **b})
    print(json.dumps({'run': run, 'rep': rep, 'item': 'four', 'tok_s': [b['tok_s'] for b in box]}), flush=True)
    for task in tasks:
        common.wait_idle()
        marker(out, run, phase=task['id'], rep=rep)
        rec = RM.run_parallel(task, 4, extra={'reasoning_effort': 'low'})
        rows.append({'run': run, 'rep': rep, 'item': task['id'], 'rule': rec.get('rule'), 'chosen': rec.get('chosen'),
                     'wall_s': round(rec['wall_s'], 3), 'completion_tokens': rec.get('completion_tokens'),
                     'branches': [{'branch': b['branch'], 'cancelled': bool(b.get('cancelled')), 'finish': b.get('finish'),
                                   'completion_tokens': b.get('completion_tokens'), 'end_s': b.get('end_s'),
                                   'text_sha': sha(b.get('text'))} for b in rec['branches']]})
        print(json.dumps({'run': run, 'rep': rep, 'item': task['id'], 'rule': rec.get('rule'), 'wall_s': round(rec['wall_s'], 1)}), flush=True)


def identical(ref, rows):
    """Greedy replies: the same token ids; the vote's branches: the same text wherever both runs finished the branch."""
    bad = []
    for r in rows:
        key = r['item']
        if key not in ref:
            ref[key] = r
            continue
        a = ref[key]
        if 'branches' in r:
            for x, y in zip(a['branches'], r['branches']):
                if not x['cancelled'] and not y['cancelled'] and x['finish'] and y['finish'] and x['text_sha'] != y['text_sha']:
                    bad.append(f'{key} branch {x["branch"]}')
        elif a['token_sha'] != r['token_sha']:
            bad.append(key)
    return bad


def nsys_ctl(cmd, out, run):
    args = ['docker', 'exec', common.NAME, 'nsys', cmd, '--session=p14']
    if cmd == 'start':
        args += [f'--output=/out/{run}', '--force-overwrite=true', '--sample=none', '--cpuctxsw=none']
    r = subprocess.run(args, capture_output=True, text=True, timeout=1800)
    return {'cmd': cmd, 'rc': r.returncode, 'out': (r.stdout + r.stderr)[-1500:]}


def export_sqlite(out, run):
    r = subprocess.run(['docker', 'run', '--rm', '--network', 'none', '-v', f'{out}:/out', '--entrypoint', 'nsys', common.IMAGE,
                        'export', '--type=sqlite', '--force-overwrite=true', f'--output=/out/{run}.sqlite', f'/out/{run}.nsys-rep'],
                       capture_output=True, text=True, timeout=3600)
    return {'rc': r.returncode, 'out': (r.stdout + r.stderr)[-1500:]}


def run_server(out, run, parallel, hook, detail, nsys, reps, workload, rows, refs, info):
    rec = {'run': run, 'parallel': parallel, 'hook': hook, 'detail': detail, 'nsys': nsys, 'reps': reps}
    guard = common.MemoryGuard()
    try:
        rec['server'] = start(parallel, hook, detail, nsys, out, run)
        warm()
        if nsys:
            rec['nsys_start'] = nsys_ctl('start', out, run)
        mine = []
        for rep in range(reps):
            if guard.fired:
                raise RuntimeError('memory guard fired')
            workload(rep, mine)
        if nsys:
            common.wait_idle()
            rec['nsys_stop'] = nsys_ctl('stop', out, run)
        rows += mine
        rec['identity_differs'] = identical(refs, mine)
        rec['ok'] = not rec['identity_differs']
    except Exception as exc:  # noqa: BLE001 - recorded; the caller stops or goes on
        rec['ok'] = False
        rec['error'] = repr(exc)[:600]
        print(f'FAILED {run}: {exc!r}'[:500], flush=True)
    finally:
        logs = subprocess.run(['docker', 'logs', common.NAME], capture_output=True, text=True)
        (out / f'{run}.server.log').write_text(logs.stdout + logs.stderr)
        common.stop_server()
        rec['mem_guard_fired'] = guard.fired
        guard.close()
    if nsys and rec.get('nsys_stop', {}).get('rc') == 0:
        rec['export'] = export_sqlite(out, run)
    info.append(rec)
    (out / f'{run}.requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows if r['run'] == run))
    return rec


def bw(out, quick):
    env = ['-e', 'P14_BW_QUICK=1'] if quick else []
    r = subprocess.run(['docker', 'run', '--rm', '--gpus', 'all', '--ipc', 'host', '--network', 'none', '--cpuset-cpus', '5-9,15-19',
                        *env, '-v', f'{HERE / "hook"}:/p14:ro', '--entrypoint', 'python', common.IMAGE, '/p14/bw_bench.py'],
                       capture_output=True, text=True, timeout=3600)
    (out / 'bw.json').write_text(r.stdout)
    return {'rc': r.returncode, 'stderr': r.stderr[-2000:]}


def ncu_probe(out):
    r = subprocess.run(['docker', 'run', '--rm', '--gpus', 'all', '--network', 'none', '--entrypoint', 'ncu', common.IMAGE,
                        '--target-processes', 'all', '-c', '1', 'python', '-c',
                        "import torch; x = torch.ones(1024, device='cuda'); print(float((x + 1).sum()))"],
                       capture_output=True, text=True, timeout=600)
    text = r.stdout + r.stderr
    res = {'rc': r.returncode, 'permission_error': 'ERR_NVGPUCTRPERM' in text, 'tail': text[-1500:]}
    res['usable'] = r.returncode == 0 and not res['permission_error'] and '==PROF==' in text
    (out / 'ncu_probe.json').write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('prefix')
    ap.add_argument('--compat', action='store_true')
    a = ap.parse_args()
    out = HERE / 'results' / a.prefix
    out.mkdir(parents=True, exist_ok=False)
    RM.LIMIT_S, RM.MAX_TOKENS = 2400, 65536
    tasks = [json.loads(l) for l in EXAMPLES.read_text().splitlines()]
    info, rows, refs = [], [], {'single': {}, 'four': {}}
    single = lambda items: (lambda rep, mine: single_workload(out, run_name[0], rep, items, mine))
    four = lambda ts: (lambda rep, mine: four_workload(out, run_name[0], rep, ts, mine))
    run_name = [None]
    if a.compat:
        plan = [('C1', 1, False, False, False, 1, single(['code', 'chat']), 'single'),
                ('C2', 1, True, False, False, 1, single(['code', 'chat']), 'single'),
                ('C3', 1, True, True, True, 1, single(['code', 'chat']), 'single'),
                ('C4', 4, False, False, False, 1, four([t for t in tasks if t['id'] in ('ex_primes', 'ex_palindrome')]), 'four'),
                ('C5', 4, True, True, True, 1, four([t for t in tasks if t['id'] in ('ex_primes', 'ex_palindrome')]), 'four')]
    else:
        plan = [('R1n', 1, False, False, False, 1, single(['long', 'code', 'chat']), 'single'),
                ('R1', 1, True, False, False, 3, single(['long', 'code', 'chat']), 'single'),
                ('R2', 1, True, True, True, 1, single(['long', 'code', 'chat']), 'single'),
                ('R3n', 4, False, False, False, 1, four(tasks), 'four'),
                ('R3', 4, True, False, False, 3, four(tasks), 'four'),
                ('R4', 4, True, True, True, 1, four(tasks), 'four')]
    status = {'prefix': a.prefix, 'compat': a.compat, 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    status['bw'] = bw(out, a.compat)
    print(json.dumps({'bw': status['bw']['rc']}), flush=True)
    stopped = None
    for run, parallel, hook, detail, nsys, reps, workload, kind in plan:
        run_name[0] = run
        rec = run_server(out, run, parallel, hook, detail, nsys, reps, workload, rows, refs[kind], info)
        (out / 'runs.json').write_text(json.dumps({**status, 'runs': info}, indent=1))
        if rec.get('identity_differs'):
            stopped = f'{run}: replies differ from the first run ({rec["identity_differs"][:5]})'
            print(f'FAILED identity {stopped}', flush=True)
            break
    status['ncu'] = ncu_probe(out)
    status['stopped'] = stopped
    status['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    (out / 'runs.json').write_text(json.dumps({**status, 'runs': info}, indent=1))
    (out / 'identity.json').write_text(json.dumps({'stopped': stopped, 'checked_runs': [r['run'] for r in info],
                                                   'differs': {r['run']: r.get('identity_differs') for r in info}}, indent=1))
    subprocess.run(['docker', 'run', '--rm', '--network', 'none', '-v', f'{out}:/out', '--entrypoint', 'chown', common.IMAGE,
                    '-R', '1000:1000', '/out'], capture_output=True)
    print(json.dumps({'stopped': stopped, 'runs': [(r['run'], r['ok']) for r in info], 'ncu_usable': status['ncu']['usable']}), flush=True)


if __name__ == '__main__':
    main()
