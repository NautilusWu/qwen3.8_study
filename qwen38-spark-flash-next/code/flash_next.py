"""Serve Qwen3.8-Flash-Next instead of Qwen3.8-27B in Part 4's runner (code/strategies of
qwen38-spark-test-time-strategies). Import, then call patch(<Part 4 strategies folder>); it changes Part 4's `common`
module in place, so run_method.py and everything else run unchanged.

What changes against Part 4's 27B server (everything else, including the requests, is Part 4's):
- target: local-inference-lab/Qwen3.8-Flash-Next-NVFP4 at commit 7c4f1bc1, inside HF_DIR (mounted as /hf)
- drafting: the model's own MTP head (no DFlash2); FN_EXTRA sets the draft flags (default: threshold 0.6, up to 6)
- KV cache bf16 (Flash Next does not take TENSORFOLD_KV_DTYPE=fp8)
- TensorFold's default memory reserve instead of 40 GiB, and no container memory limit: the model takes about 90 GiB
  and its n-gram tables are file-backed page cache, so Part 4's 40 GiB reserve and 80 GB limit do not fit
- the memory guard stops the server below FN_FLOOR_GIB of available memory (default 3) instead of 12

Environment: HF_DIR, TF_CACHE_DIR, TASKS_DIR, GRADER_DIR, SANDBOX_IMAGE as in Part 4, and
  FN_TARGET     checkpoint path inside /hf (default: the hub snapshot of the commit above)
  FN_EXTRA      extra `tensorfold serve` flags (default '--mtp-confidence 0.6 --mtp-drafts 6')
  FN_FLOOR_GIB  memory guard floor in GiB (default 3)
"""
import os
import subprocess
import sys
import time
from pathlib import Path

REVISION = '7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd'
TARGET = os.environ.get('FN_TARGET', '/hf/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/'
                        + REVISION)
EXTRA = tuple(os.environ.get('FN_EXTRA', '--mtp-confidence 0.6 --mtp-drafts 6').split())
FLOOR = float(os.environ.get('FN_FLOOR_GIB', '3'))


def patch(strategies):
    """Point Part 4's common.py at Flash Next; returns the patched module."""
    sys.path.insert(0, str(Path(strategies).resolve()))
    import common

    common.TARGET = TARGET
    common.MODEL = 'Qwen3.8-Flash-Next'
    common.NAME = 'fn-server'
    common.PORT = 8115
    guard = common.MemoryGuard
    common.MemoryGuard = lambda: guard(floor_gib=FLOOR)

    def start_server(parallel, log_path, context=32768, extra_args=()):
        common.stop_server()
        busy = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                              capture_output=True, text=True).stdout.strip()
        assert not busy, 'GPU is in use: stop other model servers first'
        Path(common.CACHE).mkdir(parents=True, exist_ok=True)
        args = ['docker', 'run', '-d', '--name', common.NAME, '--gpus', 'all', '--ipc', 'host', '--network', 'host',
                '--cpuset-cpus', '5-9,15-19', '--ulimit', 'memlock=-1', '--ulimit', 'stack=67108864',
                '-e', 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',
                '-v', f'{common.HF_DIR}:/hf:ro', '-v', f'{common.CACHE}:/cache', common.IMAGE,
                'tensorfold', 'serve', common.TARGET, '--backend', 'cuda', '--name', common.MODEL,
                '--host', '127.0.0.1', '--port', str(common.PORT), '--parallel', str(parallel),
                '--context', str(context), '--temperature', '0', '--top-p', '1', '--top-k', '0', '--thinking',
                '--prefill-fp8', *EXTRA, *extra_args]
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
                    tail = subprocess.run(['docker', 'logs', '--tail', '60', common.NAME], capture_output=True,
                                          text=True)
                    raise RuntimeError('server did not start: ' + tail.stdout[-3000:] + tail.stderr[-3000:])
                time.sleep(3)
        logs = subprocess.run(['docker', 'logs', common.NAME], capture_output=True, text=True)
        Path(log_path).write_text(logs.stdout + logs.stderr)
        image_id = subprocess.run(['docker', 'image', 'inspect', '-f', '{{.Id}}', common.IMAGE],
                                  capture_output=True, text=True).stdout.strip()
        return {'argv': args, 'image_id': image_id, 'startup_s': round(time.time() - t0, 1),
                'startup_lines': [l for l in (logs.stdout + logs.stderr).splitlines() if l.startswith('[tensorfold]')]}

    common.start_server = start_server
    return common
