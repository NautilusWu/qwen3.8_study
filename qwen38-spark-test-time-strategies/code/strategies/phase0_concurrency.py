"""Phase 0: how much throughput do parallel requests add on one DGX Spark?

    python3 phase0_concurrency.py <out_dir> [--streams 1,2,4,8] [--tokens 2048] [--rounds 2]

For each stream count N: a TensorFold server with --parallel N (otherwise the single-stream study's settings:
NVIDIA NVFP4 + DFlash2, FP8 KV, FP8 prefill, thinking on, 32K context per stream), one untimed warm-up, then for
greedy and for sampled decoding (Qwen's thinking-mode settings: temperature 0.6, top_p 0.95, top_k 20, one seed per
request) `rounds` batches of N simultaneous requests, each a different dev prompt, each writing exactly `tokens`
tokens (ignore_eos), so every request decodes the same amount. Recorded per request: time to first token, decode
rate after the first token; per batch: total tokens / wall time from the first send to the last end.

The daily service must be stopped first (the GPU must be free). A memory guard stops the server if MemAvailable
drops under 12 GiB. Standard library only.
"""
import argparse
import http.client
import json
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

IMAGE = 'qwen38-tensorfold-repro:0.6.0-trace'
PORT = 8113
NAME = 'tts-tf-concurrency'
TARGET = '/hf/hub/models--nvidia--Qwen3.8-27B-NVFP4/snapshots/482ca0f3832238542f8f5295dde86b5f22711d80'
DRAFT = '/hf/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/50307d4c4cde6860d4eee73e2547cd786fe8e8a4'
import os
CACHE = os.environ.get('TF_CACHE_DIR', str(Path.home() / '.cache/qwen38-tensorfold-cache'))
DEV = Path(os.environ.get('TASKS_DIR', 'tasks')) / 'dev.jsonl'
HF_DIR = os.environ.get('HF_DIR', str(Path.home() / '.cache/huggingface'))
SAMPLED = {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20}
GREEDY = {'temperature': 0, 'top_p': 1, 'top_k': 0}


def mem_available_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 2**20
    return 0.0


def health():
    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=5) as r:
        return json.load(r)


def start_server(streams, log):
    subprocess.run(['docker', 'rm', '-f', NAME], capture_output=True)
    args = ['docker', 'run', '-d', '--name', NAME, '--gpus', 'all', '--ipc', 'host', '--network', 'host',
            '--cpuset-cpus', '5-9,15-19', '--memory', '80g', '--memory-swap', '80g',
            '--ulimit', 'memlock=-1', '--ulimit', 'stack=67108864',
            '-e', 'TENSORFOLD_KV_DTYPE=fp8', '-e', 'TENSORFOLD_MEMORY_RESERVE_GIB=40', '-e', 'TENSORFOLD_KV_POOL_GIB=0',
            '-e', 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',
            '-v', f'{HF_DIR}:/hf:ro', '-v', f'{CACHE}:/cache', IMAGE,
            'tensorfold', 'serve', TARGET, '--drafter', DRAFT, '--backend', 'cuda', '--name', 'Qwen3.8-27B',
            '--host', '127.0.0.1', '--port', str(PORT), '--parallel', str(streams), '--context', '32768',
            '--temperature', '0', '--top-p', '1', '--top-k', '0', '--thinking', '--prefill-fp8']
    subprocess.run(args, check=True, capture_output=True)
    t0 = time.time()
    while time.time() - t0 < 1800:
        try:
            health()
            break
        except OSError:
            if subprocess.run(['docker', 'inspect', '-f', '{{.State.Running}}', NAME],
                              capture_output=True, text=True).stdout.strip() != 'true':
                raise RuntimeError('server exited: ' + subprocess.run(['docker', 'logs', '--tail', '40', NAME],
                                                                      capture_output=True, text=True).stderr)
            time.sleep(3)
    else:
        raise RuntimeError('server did not start')
    lines = subprocess.run(['docker', 'logs', NAME], capture_output=True, text=True)
    log.write(lines.stdout + lines.stderr)
    return {'argv': args, 'startup_s': round(time.time() - t0, 1),
            'startup_lines': [l for l in (lines.stdout + lines.stderr).splitlines() if l.startswith('[tensorfold]')]}


def stop_server():
    subprocess.run(['docker', 'rm', '-f', NAME], capture_output=True)


def one_request(messages, tokens, sampling, seed, t_zero, out):
    body = {'model': 'Qwen3.8-27B', 'messages': messages, 'max_tokens': tokens, 'ignore_eos': True, 'seed': seed,
            'stream': True, 'stream_options': {'include_usage': True},
            'chat_template_kwargs': {'enable_thinking': True}, **sampling}
    conn = http.client.HTTPConnection('127.0.0.1', PORT, timeout=1800)
    sent = time.perf_counter() - t_zero
    conn.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json'})
    resp = conn.getresponse()
    if resp.status != 200:
        out.update(error=f'HTTP {resp.status}: {resp.read()[:300]!r}')
        return
    first = last = None
    usage = {}
    while True:
        line = resp.readline()
        if not line:
            break
        if not line.startswith(b'data:'):
            continue
        raw = line[5:].strip()
        if raw == b'[DONE]':
            break
        if not raw:
            continue
        x = json.loads(raw)
        usage.update(x.get('usage') or {})
        for c in x.get('choices') or []:
            d = c.get('delta') or {}
            if d.get('content') or d.get('reasoning_content'):
                now = time.perf_counter() - t_zero
                first = now if first is None else first
                last = now
    conn.close()
    end = time.perf_counter() - t_zero
    n = usage.get('completion_tokens') or 0
    out.update(sent=sent, first=first, last=last, end=end, tokens=n,
               ttft_s=round(first - sent, 3) if first is not None else None,
               decode_tok_s=round((n - 1) / (last - first), 2) if first is not None and last > first else None)


def batch(prompts, tokens, sampling, seed0):
    t_zero = time.perf_counter()
    rows = [{'prompt_id': p['id']} for p in prompts]
    threads = [threading.Thread(target=one_request, args=(p['messages'], tokens, sampling, seed0 + i, t_zero, rows[i]))
               for i, p in enumerate(prompts)]
    peak = {'running': 0}
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            try:
                peak['running'] = max(peak['running'], health().get('requests_running', 0))
            except OSError:
                pass
            stop.wait(0.5)

    w = threading.Thread(target=watch, daemon=True)
    w.start()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop.set()
    wall = max(r.get('end', 0) for r in rows) - min(r.get('sent', 0) for r in rows)
    total = sum(r.get('tokens', 0) for r in rows)
    return {'requests': rows, 'wall_s': round(wall, 2), 'total_tokens': total,
            'aggregate_tok_s': round(total / wall, 2) if wall else None, 'peak_requests_running': peak['running'],
            'errors': [r['error'] for r in rows if 'error' in r]}


def guard(stop_event, fired):
    while not stop_event.is_set():
        if mem_available_gib() < 12:
            fired.append(time.time())
            stop_server()
            return
        stop_event.wait(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--streams', default='1,2,4,8')
    ap.add_argument('--tokens', type=int, default=2048)
    ap.add_argument('--rounds', type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    busy = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                          capture_output=True, text=True).stdout.strip()
    assert not busy, 'GPU is in use: stop the daily service (docker stop qwen38-tensorfold) first'
    prompts = [json.loads(l) for l in DEV.read_text().splitlines()]
    messages = lambda p: [{'role': 'system', 'content': 'You are a helpful assistant.'}] + p['messages']
    prompts = [{'id': p['id'], 'messages': messages(p)} for p in prompts]
    stop_event, fired = threading.Event(), []
    threading.Thread(target=guard, args=(stop_event, fired), daemon=True).start()
    results = {'tokens_per_request': a.tokens, 'rounds': a.rounds, 'image': IMAGE, 'configs': []}
    try:
        for n in map(int, a.streams.split(',')):
            with open(out / f'server_p{n}.log', 'w') as log:
                launch = start_server(n, log)
            print(json.dumps({'streams': n, 'startup_s': launch['startup_s'], 'lines': launch['startup_lines']}), flush=True)
            batch(prompts[:1], 64, GREEDY, 1)                       # untimed warm-up
            cfg = {'streams': n, 'launch': launch, 'modes': {}}
            for mode, sampling in (('greedy', GREEDY), ('sampled', SAMPLED)):
                runs = []
                for r in range(a.rounds):
                    chosen = [prompts[(r * n + i) % len(prompts)] for i in range(n)]
                    res = batch(chosen, a.tokens, sampling, seed0=1000 * r + 1)
                    runs.append(res)
                    per = [x['decode_tok_s'] for x in res['requests'] if x.get('decode_tok_s')]
                    print(json.dumps({'streams': n, 'mode': mode, 'round': r, 'aggregate_tok_s': res['aggregate_tok_s'],
                                      'per_stream_tok_s': per, 'peak_running': res['peak_requests_running'],
                                      'errors': res['errors']}), flush=True)
                cfg['modes'][mode] = runs
            results['configs'].append(cfg)
            (out / 'results.json').write_text(json.dumps(results, indent=1))
            stop_server()
            if fired:
                raise RuntimeError('memory guard fired')
    finally:
        stop_event.set()
        stop_server()
        results['memory_guard_fired'] = bool(fired)
        (out / 'results.json').write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
