"""Which kind of long reply makes TensorFold's GPU memory grow? One request per variant, server restarted before each.

    python3 leak_probe.py <out.json> [variant ...]

Variants (thinking off, greedy, the same ~4K-token file in the prompt):
  copy_tools    write the file verbatim through a Write tool call (the reply copies the prompt: copy drafts)
  copy_nodraft  the same with "draft": false (no DFlash2, no copy drafts)
  copy_plain    repeat the file verbatim as plain text, no tools
  novel_tools   write a new, different file through the Write tool (little to copy)
  long_*        the same as the variant after "long_", after ~50K tokens of other reference files (an agent's
                long conversation); xlong_*: after ~100K tokens
The GPU memory of the server process is sampled every second; a request is cut off past CAP_GIB.
"""
import json
import subprocess
import sys
import threading
import time
import urllib.request

URL = 'http://127.0.0.1:8091'
MODEL = 'qwen3.8-27b-tensorfold-nvfp4'
CAP_GIB = 60
WRITE = {'type': 'function', 'function': {'name': 'Write', 'description': 'Write a file.', 'parameters': {
    'type': 'object', 'properties': {'file_path': {'type': 'string'}, 'content': {'type': 'string'}},
    'required': ['file_path', 'content']}}}


def source_file():
    lines = ['"""Inventory helpers (synthetic test file)."""', '', 'import math', '']
    for i in range(60):
        lines += [f'def item_{i}_price(quantity, unit={i + 1}.25, discount=0.{i % 9}):',
                  f'    """Price of {i} items with discount."""',
                  f'    total = quantity * unit * (1 - discount)',
                  f'    if total > {100 * (i + 1)}:',
                  f'        total -= math.floor(total / {i + 7})',
                  f'    return round(total, 2)', '']
    return '\n'.join(lines)


def reference_files(n=12):
    """~50K tokens of other modules, so the reply is written after a long prompt (as late in an agent session)."""
    parts = []
    for f in range(n):
        lines = [f'# reference module ledger_{f}.py', '']
        for i in range(60):
            lines += [f'def ledger_{f}_{i}(rows, rate={f}.{i % 10}5, cap={i * 37 + f}):',
                      f'    """Sum ledger {f} row {i} with a rate cap."""',
                      f'    value = sum(r * rate for r in rows[:{i % 13 + 2}])',
                      f'    return min(value, cap) if value > {f * 3 + i} else value', '']
        parts.append('\n'.join(lines))
    return '\n\n'.join(parts)


def gpu_gib():
    out = subprocess.run(['nvidia-smi', '--query-compute-apps=used_memory', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout
    return sum(int(x) for x in out.split()) / 1024


def restart():
    subprocess.run(['docker', 'restart', '-t', '30', 'qwen38-tensorfold'], check=True, capture_output=True)
    for _ in range(300):
        try:
            urllib.request.urlopen(URL + '/health', timeout=5).read()
            return
        except OSError:
            time.sleep(1)
    raise RuntimeError('server did not come back')


def body(variant, text):
    files = 24 if variant.startswith('xlong_') else 12 if variant.startswith('long_') else 0
    variant = variant.split('long_', 1)[1] if files else variant
    prefix = ('Reference files from the repository (for context only):\n\n' + reference_files(files) + '\n\n'
              if files else '')
    if variant == 'novel_tools':
        ask = ('Write a NEW Python module /work/shipping.py with 60 different functions computing shipping costs '
               '(each with a docstring and some arithmetic), using the Write tool. Do not repeat the file below.\n\n')
    elif variant == 'copy_plain':
        ask = 'Repeat the following file exactly, character for character, as plain text and nothing else.\n\n'
    else:
        ask = 'Write the following content exactly, character for character, to /work/inventory.py with the Write tool.\n\n'
    b = {'model': MODEL, 'stream': True, 'max_tokens': 12000, 'reasoning_effort': 'none',
         'messages': [{'role': 'user', 'content': prefix + ask + text}]}
    if variant != 'copy_plain':
        b['tools'] = [WRITE]
    if variant == 'copy_nodraft':
        b['draft'] = False
    return b


def run(variant, text):
    restart()
    time.sleep(3)
    base = gpu_gib()
    samples, done = [], threading.Event()

    def sample():
        while not done.is_set():
            samples.append((round(time.time() - t0, 1), round(gpu_gib(), 2)))
            done.wait(1)

    t0 = time.time()
    threading.Thread(target=sample, daemon=True).start()
    req = urllib.request.Request(URL + '/v1/chat/completions', json.dumps(body(variant, text)).encode(),
                                 {'Content-Type': 'application/json'})
    chars, cut, usage = 0, False, None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b'data: ') or line == b'data: [DONE]':
                continue
            event = json.loads(line[6:])
            usage = event.get('usage') or usage
            for choice in event.get('choices') or []:
                d = choice.get('delta') or {}
                chars += len(d.get('content') or '') + sum(len((c.get('function') or {}).get('arguments') or '')
                                                           for c in d.get('tool_calls') or [])
            if samples and samples[-1][1] > CAP_GIB:
                cut = True
                break
    wall = time.time() - t0
    time.sleep(3)
    done.set()
    after = gpu_gib()
    health = json.load(urllib.request.urlopen(URL + '/health', timeout=10))
    tokens = health['completion_tokens_total'] or (usage or {}).get('completion_tokens') or 0
    peak = max(g for _, g in samples) if samples else after
    return {'variant': variant, 'wall_s': round(wall, 1), 'reply_chars': chars, 'completion_tokens': tokens,
            'gpu_before_gib': round(base, 2), 'gpu_peak_gib': round(peak, 2), 'gpu_after_gib': round(after, 2),
            'growth_gib': round(after - base, 2),
            'mib_per_token': round((after - base) * 1024 / tokens, 2) if tokens else None,
            'rounds': health['rounds_total'], 'drafted': health['drafted_total'], 'accepted': health['accepted_total'],
            'cut_at_cap': cut, 'samples': samples}


def main():
    out = sys.argv[1]
    variants = sys.argv[2:] or ['copy_tools', 'copy_nodraft', 'copy_plain', 'novel_tools']
    text = source_file()
    results = []
    for v in variants:
        rec = run(v, text)
        results.append(rec)
        print(json.dumps({k: rec[k] for k in rec if k != 'samples'}), flush=True)
        with open(out, 'w') as f:
            json.dump(results, f, indent=1)


if __name__ == '__main__':
    main()
