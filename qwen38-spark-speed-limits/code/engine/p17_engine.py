"""PLAN_17: the six server configurations, all in the same slot as common.py's server (container name, port, CPU set,
memory limit, host network, checkpoint, model name).

    O1, O4  0.6.0 + DFlash2, --parallel 1 / 4      common.start_server, unchanged
    N1, N4  1.2.7 + DFlash2, --parallel 1 / 4      PLAN_13's 1.2.5 arguments with the 1.2.7 release mounted instead
    S1      1.2.7 + DSpine-b16, --parallel 1       as N1, --drafter names the DSpine folder
    N0      1.2.7, no drafts, --parallel 1         as N1, --no-drafts instead of --drafter (only the lossless check)
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

import native_engine as E                     # Part 10's code/native_engine.py (PLAN_13's module), not modified
import common

VERSION = '1.2.7'
COMMIT = '93a2e93ecebff8aa52b88e58c5dd222516e69c5f'
import os
DIST = Path(os.environ['TF127_DIR'])           # folder holding the 1.2.7 release archive (see REPRODUCE.md)
ARCHIVE = DIST / 'tensorfold-1.2.7-linux-aarch64.tar.gz'
ARCHIVE_SHA256 = '250b7cd3d1aa2685cbf011f501c4f10bbff44c6ccd876dc63a7a4bbf337ed01b'
ROOT = DIST / 'tensorfold-1.2.7-linux-aarch64'
DSPINE_HOST = Path(common.HF_DIR) / 'extra/tencent/Qwen3.8-27B-DSpine-b16-ea909c0f'
DSPINE = '/hf/extra/tencent/Qwen3.8-27B-DSpine-b16-ea909c0f'
DSPINE_REV = 'ea909c0f5fab8865f9e14600334f55f7ce47308f'
DSPINE_SHA256 = 'ac90f1df842ea66b23e4224f3e86624add547526f689ffe9e9c6b7a5937116b1'

CONFIGS = {'O1': ('old', 1, 'dflash2'), 'O4': ('old', 4, 'dflash2'),
           'N1': ('new', 1, 'dflash2'), 'N4': ('new', 4, 'dflash2'),
           'S1': ('new', 1, 'dspine'), 'N0': ('new', 1, None)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def check_release():
    digest = sha256(ARCHIVE)
    sums = subprocess.run(['sha256sum', '-c', '--quiet', 'SHA256SUMS'], cwd=ROOT, capture_output=True, text=True)
    version = subprocess.run(['docker', 'run', '--rm', '--network', 'none', '-v', f'{ROOT}:/opt/tensorfold:ro',
                              '--entrypoint', E.BINARY, E.BASE_IMAGE, '--version'], capture_output=True, text=True)
    return {'archive_sha256': digest, 'archive_ok': digest == ARCHIVE_SHA256, 'files_ok': sums.returncode == 0,
            'version': version.stdout.strip(), 'version_ok': version.stdout.strip() == f'tensorfold-native {VERSION}'}


def check_dspine():
    digest = sha256(DSPINE_HOST / 'model.safetensors')
    cfg = json.loads((DSPINE_HOST / 'config.json').read_text())
    return {'weights_sha256': digest, 'weights_ok': digest == DSPINE_SHA256, 'has_dflash_config': 'dflash_config' in cfg}


def server_args(parallel, drafter, context=81920):
    """PLAN_13's 1.2.5 argument list with only the mounted release and the drafter changed."""
    base = E.server_args(parallel, context)
    old_mount = f'{E.ROOT}:/opt/tensorfold:ro'
    assert base.count(old_mount) == 1
    args = [f'{ROOT}:/opt/tensorfold:ro' if a == old_mount else a for a in base]
    i = args.index('--drafter')
    assert args[i + 1] == common.DRAFT
    if drafter == 'dspine':
        args[i + 1] = DSPINE
    elif drafter is None:
        args[i:i + 2] = ['--no-drafts']
    return args


def start_native(parallel, drafter, log_path, context=81920):
    """As engine_native.start_server, with server_args above."""
    common.stop_server()
    E.gpu_free()
    busy = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                          capture_output=True, text=True).stdout.strip()
    assert not busy, 'GPU is in use'
    args = server_args(parallel, drafter, context)
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
            'startup_lines': [l for l in (logs.stdout + logs.stderr).splitlines() if l.startswith('[tensorfold]')][-20:]}


def start(cfg, log_path, context=81920):
    engine, parallel, drafter = CONFIGS[cfg]
    common.stop_server()
    E.gpu_free()
    if engine == 'old':
        info = common.start_server(parallel, log_path, context=context)
        info['engine'] = '0.6.0'
    else:
        info = start_native(parallel, drafter, log_path, context=context)
    info.update(config=cfg, parallel=parallel, drafter=drafter)
    return info


wait_idle = E.wait_idle
outside_sockets = E.outside_sockets
