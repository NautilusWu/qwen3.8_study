"""Memory log and guard for a harness run.

    python3 memwatch.py <log.jsonl> [--floor-gib 12] [--every 5]

Every few seconds: the host's MemAvailable/MemFree/swap and each GPU process's memory, one JSON line each.
On the GB10 the GPU allocates from system memory outside any container limit, so a server whose caches keep
growing can take the whole machine down (2026-10-10). If MemAvailable falls under the floor, the model server
container is stopped and a `<log>.fired` file is written; run_all.py then stops the run.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

CONTAINER = os.environ.get('TF_CONTAINER', 'qwen38-tensorfold')


def meminfo():
    out = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemAvailable', 'MemFree', 'SwapTotal', 'SwapFree'):
            out[key] = int(value.split()[0]) // 1024        # MiB
    return out


def gpu_processes():
    try:
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    return {pid.strip(): int(mib) for pid, mib in (l.split(',') for l in out.strip().splitlines() if ',' in l)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--floor-gib', type=float, default=12)
    ap.add_argument('--every', type=float, default=5)
    a = ap.parse_args()
    fired = Path(a.log + '.fired')
    while True:
        rec = {'t': round(time.time(), 1), **meminfo(), 'gpu_mib': gpu_processes()}
        if rec['MemAvailable'] < a.floor_gib * 1024 and not fired.exists():
            subprocess.run(['docker', 'stop', CONTAINER], capture_output=True, timeout=120)
            rec['action'] = f'stopped {CONTAINER}: MemAvailable under {a.floor_gib} GiB'
            fired.write_text(json.dumps(rec) + '\n')
        with open(a.log, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        time.sleep(a.every)


if __name__ == '__main__':
    main()
