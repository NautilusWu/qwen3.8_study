"""TensorFold 1.2.5 (the native engine) as the server in Part 4's runner (code/strategies of
qwen38-spark-test-time-strategies). Call patch(<Part 4 strategies folder>) first; it imports Part 4's `common`.

The 1.2.5 server takes the slot of Part 4's 0.6.0 server: the same container name, port, CPU set (5-9,15-19), 80 GB
container limit, host network, checkpoint (nvidia/Qwen3.8-27B-NVFP4 @482ca0f3), drafter (z-lab/Qwen3.8-27B-DFlash2
@50307d4c), model name and 40 GiB memory reserve, so Part 4's stop_server() and memory guard work for both engines.
The official release archive, extracted, is mounted read-only into the base image of Part 4's TensorFold image, which
serves only as the runtime environment (the native engine needs no Python).

Differences from Part 4's 0.6.0 server that come with the engine (none of them can be set to match):
- prompt processing (prefill) with bf16 activations: 1.2.5 refuses --prefill-fp8 (0.6.0 used FP8 activations there)
- --precision full: bf16 activations on the NVFP4/FP8 weights, the only mode 1.2.5 offers for this checkpoint
  (the same decode math as 0.6.0)
- FP8 KV cache via --kv-cache fp8 (0.6.0: TENSORFOLD_KV_DTYPE=fp8, a recipe patch), and the engine's own KV pool
  (0.6.0: TENSORFOLD_KV_POOL_GIB=0)
- no PyTorch, so no PYTORCH_CUDA_ALLOC_CONF and no kernel cache: every start measures the verification window cost
  again (about 43 s, kept only inside the container)
- its own sampler: the same temperature, top_p, top_k and seed draw different tokens than 0.6.0
- /health reports live.connections instead of busy / requests_running (wait_idle() below reads both)

Environment: as Part 4 (HF_DIR, TASKS_DIR, GRADER_DIR, SANDBOX_IMAGE; TF_CACHE_DIR for 0.6.0) and
  TF_NATIVE_DIR      the extracted release folder tensorfold-1.2.5-linux-aarch64 (needed for engine 'new')
  TF_NATIVE_ARCHIVE  optional: the downloaded .tar.gz, so that check_release() also checks its sha256
"""
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

VERSION = '1.2.5'
COMMIT = '8be9c408c2825102b475dc6049c6aa3cf5d9e910'           # tag v1.2.5
ARCHIVE_SHA256 = '1a0192b28d6ab6ec2749f3ff25515379dba4895171e781b30f810afc971ae4f8'
BASE_IMAGE = 'nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c'
BINARY = '/opt/tensorfold/bin/tensorfold-native'
ROOT = Path(os.environ.get('TF_NATIVE_DIR', 'tensorfold-1.2.5-linux-aarch64')).resolve()
common = None                   # Part 4's common module, set by patch()
start_060 = None                # Part 4's 0.6.0 start_server, kept before patch(serve_native=True) replaces it


def patch(strategies, serve_native=False):
    """Import Part 4's common module. With serve_native, common.start_server and common.wait_idle become the 1.2.5
    versions (for Part 4's run_method.py, see run_native.py); otherwise use start('new' | 'old')."""
    global common, start_060
    sys.path.insert(0, str(Path(strategies).resolve()))
    import common as c
    common = c
    start_060 = c.start_server
    if serve_native:
        c.start_server = start_server
        c.wait_idle = wait_idle
    return c


def check_release(archive=os.environ.get('TF_NATIVE_ARCHIVE')):
    """The archive's sha256 (if given), the extracted files against the archive's SHA256SUMS, and the binary's version
    (`--version` prints only "tensorfold-native 1.2.5"; the commit is fixed by the tag and the archive's sha256)."""
    if not (ROOT / 'SHA256SUMS').exists():
        return {'files_ok': False, 'error': f'no extracted release at {ROOT}: set TF_NATIVE_DIR'}
    out = {}
    if archive:
        h = hashlib.sha256()
        with open(archive, 'rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                h.update(block)
        out.update(archive_sha256=h.hexdigest(), archive_ok=h.hexdigest() == ARCHIVE_SHA256)
    sums = subprocess.run(['sha256sum', '-c', '--quiet', 'SHA256SUMS'], cwd=ROOT, capture_output=True, text=True)
    version = subprocess.run(['docker', 'run', '--rm', '--network', 'none', '-v', f'{ROOT}:/opt/tensorfold:ro',
                              '--entrypoint', BINARY, BASE_IMAGE, '--version'], capture_output=True, text=True)
    out.update(files_ok=sums.returncode == 0, version=version.stdout.strip(),
               version_ok=version.stdout.strip() == f'tensorfold-native {VERSION}')
    return out


def server_args(parallel, context):
    return ['docker', 'run', '-d', '--name', common.NAME, '--gpus', 'all', '--ipc', 'host', '--network', 'host',
            '--cpuset-cpus', '5-9,15-19', '--memory', '80g', '--memory-swap', '80g',
            '--ulimit', 'memlock=-1', '--ulimit', 'stack=67108864',
            '-e', 'TENSORFOLD_MEMORY_RESERVE_GIB=40',
            '-v', f'{common.HF_DIR}:/hf:ro', '-v', f'{ROOT}:/opt/tensorfold:ro',
            '--entrypoint', BINARY, BASE_IMAGE,
            'serve', common.TARGET, '--drafter', common.DRAFT, '--backend', 'cuda', '--name', common.MODEL,
            '--host', '127.0.0.1', '--port', str(common.PORT), '--parallel', str(parallel), '--context', str(context),
            '--temperature', '0', '--top-p', '1', '--top-k', '0', '--thinking', '--kv-cache', 'fp8', '--precision', 'full']


def start_server(parallel, log_path, context=81920):
    """As Part 4's common.start_server, for the native engine: returns once /health no longer reports warming."""
    assert (ROOT / 'bin' / 'tensorfold-native').exists(), 'set TF_NATIVE_DIR to the extracted 1.2.5 release folder'
    common.stop_server()
    busy = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                          capture_output=True, text=True).stdout.strip()
    assert not busy, 'GPU is in use: stop other model servers first'
    args = server_args(parallel, context)
    subprocess.run(args, check=True, capture_output=True)
    t0 = time.time()
    while True:
        try:
            if not common.health().get('warming'):
                break
            time.sleep(3)
        except OSError:
            running = subprocess.run(['docker', 'inspect', '-f', '{{.State.Running}}', common.NAME],
                                     capture_output=True, text=True).stdout.strip()
            if running != 'true' or time.time() - t0 > 1800:
                tail = subprocess.run(['docker', 'logs', '--tail', '60', common.NAME], capture_output=True, text=True)
                raise RuntimeError('server did not start: ' + tail.stdout + tail.stderr)
            time.sleep(3)
    logs = subprocess.run(['docker', 'logs', common.NAME], capture_output=True, text=True)
    Path(log_path).write_text(logs.stdout + logs.stderr)
    return {'engine': f'native {VERSION}', 'argv': args, 'startup_s': round(time.time() - t0, 1),
            'startup_lines': (logs.stdout + logs.stderr).splitlines()[-80:]}


def gpu_free(timeout=180):
    """After a server stops, its GPU process can stay listed for a moment; both start functions refuse a busy GPU."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                              capture_output=True, text=True).stdout.strip():
            return round(time.time() - t0, 1)
        time.sleep(2)
    return None


def start(engine, parallel, log_path, context=81920):
    """engine 'old' (0.6.0: Part 4's start_server, unchanged) or 'new' (1.2.5)."""
    common.stop_server()
    gpu_free()
    if engine == 'old':
        info = start_060(parallel, log_path, context=context)
        info['engine'] = '0.6.0'
        return info
    return start_server(parallel, log_path, context=context)


def active(h):
    """Requests running or waiting on the server, from either engine's /health."""
    live = h.get('live')
    if isinstance(live, dict):                       # native: connections = active + waiting
        return int(live.get('connections') or 0)
    return int(bool(h.get('busy'))) + int(h.get('requests_running') or 0)


def wait_idle(timeout=1800):
    """As Part 4's common.wait_idle, for both engines: nothing left running (e.g. a cancelled branch still finishing)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not active(common.health()):
            return round(time.time() - t0, 1)
        time.sleep(1)
    raise RuntimeError('server stayed busy')


def outside_sockets():
    """Report only: TCP/UDP sockets of the server's process whose remote end is not this host (read inside the
    container, which shares the host's network)."""
    script = r'''
import os
ino = set()
for fd in os.listdir('/proc/1/fd'):
    try:
        t = os.readlink('/proc/1/fd/' + fd)
    except OSError:
        continue
    if t.startswith('socket:['):
        ino.add(t[8:-1])
out = []
for name in ('tcp', 'tcp6', 'udp', 'udp6'):
    for line in open('/proc/net/' + name).read().splitlines()[1:]:
        f = line.split()
        if f[9] in ino:
            out.append(name + ' ' + f[1] + ' ' + f[2] + ' ' + f[3])
print('\n'.join(out))
'''
    r = subprocess.run(['docker', 'exec', common.NAME, 'python3', '-c', script], capture_output=True, text=True)
    rows = [l.split() for l in r.stdout.splitlines() if l.strip()]
    local = ('0100007F', '00000000', '00000000000000000000000000000000', '00000000000000000000000001000000',
             '0000000000000000FFFF00000100007F')
    return {'sockets': len(rows), 'outside': [' '.join(x) for x in rows if x[2].split(':')[0] not in local],
            'error': r.stderr.strip()[:300] or None}
