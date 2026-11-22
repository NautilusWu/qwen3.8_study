"""Serve a Qwen3.8-Flash-Next GGUF file with llama.cpp in Part 4's runner (code/strategies of
qwen38-spark-test-time-strategies). Call patch(<Part 4 strategies folder>) before importing run_method: it changes
Part 4's `common` module in place (run_method binds body_for and stream_chat when it is imported).

What changes against Part 4's server (the requests are otherwise Part 4's, unchanged):
- the server: llama.cpp's llama-server on the host (no container), pinned to cores 5-9,15-19 (where Part 4's
  TensorFold container runs), 10 threads, port 8118, listening on 127.0.0.1 only;
- every request carries "cache_prompt": false and no "return_token_ids" (a TensorFold-only field). Without it
  llama.cpp reuses the previous request's prompt cache, and a greedy reply then differs from the same request sent to
  a fresh server (in our smoke test 3 identical requests gave 727, 864 and 864 tokens with the cache, 727 three times
  without it). So every request is processed from scratch;
- health() reads llama.cpp's /health, wait_idle() its /slots, stop_server() stops the process by its pid file;
- the memory guard stops the server below 3 GiB of available memory (Part 4: 12).

The server flags are this study's (identical for every quantization; only the model file differs):
  -ngl 999 -c 81920 -np 1 -fa on -lm mmap --lazy-mode on --jinja --chat-template-file <template>
  --reasoning-format deepseek --min-p 0 -t 10 -tb 10 --no-webui
and, with MTP, `-md <MTP head> --spec-type draft-mtp --spec-draft-n-max N` (draft p-min left at its default 0).

Environment:
  LLAMA_SERVER   the llama-server binary built from llama.cpp v1.0.0 (REPRODUCE.md step 1)
  FN_TEMPLATE    chat_template.jinja of local-inference-lab/Qwen3.8-Flash-Next-NVFP4 at commit 7c4f1bc1
                 (the same official template for every GGUF file, instead of the one inside each file)
  MTP_HEAD       Unsloth's mtp-Qwen3.8-Flash-Next-Q8_0.gguf (only needed with MTP)
  LLAMA_CPUS     cores for the server (default 5-9,15-19); LLAMA_PORT (default 8118)
"""
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

PORT = int(os.environ.get('LLAMA_PORT', '8118'))
CPUS = os.environ.get('LLAMA_CPUS', '5-9,15-19')
CONTEXT = 81920
START_LIMIT_S = 600
GPU_WAIT_S = 300
MODEL = 'Qwen3.8-Flash-Next'
KEEP = ('prompt_n', 'prompt_ms', 'prompt_per_second', 'predicted_n', 'predicted_ms', 'predicted_per_second', 'cache_n',
        'draft_n', 'draft_n_accepted')
PIDFILE = Path(os.environ.get('LLAMA_PIDFILE', Path.cwd() / 'llama-server.pid'))
common = None


def _env_path(name):
    v = os.environ.get(name)
    assert v and Path(v).exists(), f'set {name} to an existing file (see the docstring)'
    return os.path.abspath(v)       # as given (a Hugging Face snapshot path is a symlink; it is not resolved)


def patch(strategies):
    """Point Part 4's common.py at llama.cpp; returns the patched module."""
    global common
    sys.path.insert(0, str(Path(strategies).resolve()))
    import common as c
    common = c
    c.PORT, c.MODEL, c.NAME = PORT, MODEL, 'llama-server'
    guard = c.MemoryGuard
    c.MemoryGuard = lambda: guard(floor_gib=3)
    plain_body = c.body_for

    def body_for(*a, **k):
        body = plain_body(*a, **k)
        body.pop('return_token_ids', None)
        body['cache_prompt'] = False
        return body

    c.body_for, c.health, c.wait_idle, c.stop_server = body_for, health, wait_idle, stop_server
    return c


def _get(path, timeout=5):
    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}{path}', timeout=timeout) as r:
        return json.load(r)


def health():
    return _get('/health')


def wait_idle(timeout=1800):
    """Before each request: no slot still processing (e.g. a request a client gave up on)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not any(s.get('is_processing') for s in _get('/slots')):
            return round(time.time() - t0, 1)
        time.sleep(1)
    raise RuntimeError('server stayed busy')


def _alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:   # a zombie child counts as gone
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except OSError:
        return False


def stop_server():
    """Stop the server started from this module (by its pid file; never by name matching)."""
    if not PIDFILE.exists():
        return
    pid = int(PIDFILE.read_text().strip() or 0)
    if pid and _alive(pid):
        os.kill(pid, signal.SIGTERM)
        t0 = time.time()
        while _alive(pid) and time.time() - t0 < 60:
            time.sleep(0.5)
        if _alive(pid):
            os.kill(pid, signal.SIGKILL)
            while _alive(pid):
                time.sleep(0.5)
    try:
        os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, OSError):
        pass
    PIDFILE.unlink(missing_ok=True)


def gpu_pids():
    return subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                          capture_output=True, text=True).stdout.split()


def wait_gpu_free():
    """A stopped server stays in nvidia-smi for a few seconds; starting the next one then fails. Wait until nvidia-smi
    lists no compute process (at most GPU_WAIT_S). Returns the seconds waited."""
    t0 = time.time()
    while gpu_pids() and time.time() - t0 < GPU_WAIT_S:
        time.sleep(1)
    return round(time.time() - t0, 1)


def server_args(model, mtp_n=0, budget=None):
    args = [_env_path('LLAMA_SERVER'), '-m', str(model), '--alias', MODEL, '--host', '127.0.0.1', '--port', str(PORT),
            '-ngl', '999', '-c', str(CONTEXT), '-np', '1', '-fa', 'on', '-lm', 'mmap', '--lazy-mode', 'on',
            '--jinja', '--chat-template-file', _env_path('FN_TEMPLATE'), '--reasoning-format', 'deepseek',
            '--min-p', '0', '-t', '10', '-tb', '10', '--no-webui']
    if budget is not None:
        args += ['--reasoning-budget', str(budget)]
    if mtp_n:
        args += ['-md', _env_path('MTP_HEAD'), '--spec-type', 'draft-mtp', '--spec-draft-n-max', str(mtp_n)]
    return args


def port_in_use():
    return bool(subprocess.run(['ss', '-ltnH', f'sport = :{PORT}'], capture_output=True, text=True).stdout.strip())


def start(model, log_path, mtp_n=0, budget=None):
    """Start llama-server with the given GGUF file (the first shard of a split file); returns startup facts."""
    stop_server()
    waited = wait_gpu_free()
    assert not gpu_pids(), 'GPU is in use: stop other model servers first'
    assert not port_in_use(), f'port {PORT} is in use'
    argv = ['taskset', '-c', CPUS] + server_args(model, mtp_n, budget)
    log = open(log_path, 'w')
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    PIDFILE.write_text(str(proc.pid))
    t0 = time.time()
    while True:
        try:
            if health().get('status') == 'ok':
                break
        except (OSError, urllib.error.HTTPError, ValueError):
            pass
        if proc.poll() is not None or time.time() - t0 > START_LIMIT_S:
            stop_server()
            log.flush()
            raise RuntimeError('server did not start: ' + Path(log_path).read_text(errors='replace')[-3000:])
        time.sleep(2)
    log.flush()
    text = Path(log_path).read_text(errors='replace')
    return {'argv': argv, 'pid': proc.pid, 'mtp_n': mtp_n, 'budget': budget, 'startup_s': round(time.time() - t0, 1),
            'gpu_release_wait_s': waited, 'draft_loaded': 'loading draft model' in text}


def outside_sockets(pid):
    """TCP/UDP sockets of the server process: listening addresses and any remote end that is not this host."""
    ino = set()
    for fd in os.listdir(f'/proc/{pid}/fd'):
        try:
            t = os.readlink(f'/proc/{pid}/fd/{fd}')
        except OSError:
            continue
        if t.startswith('socket:['):
            ino.add(t[8:-1])
    listen, remote = [], []
    for name in ('tcp', 'tcp6', 'udp', 'udp6'):
        for line in Path(f'/proc/net/{name}').read_text().splitlines()[1:]:
            f = line.split()
            if f[9] not in ino:
                continue
            local, rem, state = f[1], f[2], f[3]
            if name.startswith('tcp') and state == '0A':
                listen.append(f'{name} {local}')
            elif not rem.startswith(('00000000:0000', '0100007F', '00000000000000000000000000000000:0000',
                                     '0000000000000000FFFF00000100007F', '00000000000000000000000001000000')):
                remote.append(f'{name} {local} {rem} {state}')
    return {'listen': listen, 'remote': remote}


class Recorder:
    """Records llama.cpp's per-request 'timings' (the last stream chunk carries them, with the draft counts) without
    touching the request path: a json shim inside common copies them; run_method's stream_chat is wrapped to keep
    one row per request. Works with any engine (a TensorFold reply simply has no 'timings')."""

    def __init__(self, common_mod, run_method):
        self.rows, self.task, self._local, self._lock = [], None, threading.local(), threading.Lock()
        rec = self

        class JsonShim:
            def __getattr__(self, name):
                return getattr(json, name)

            @staticmethod
            def loads(s, *a, **k):
                x = json.loads(s, *a, **k)
                if isinstance(x, dict) and isinstance(x.get('timings'), dict) and hasattr(rec._local, 'stats'):
                    rec._local.stats.update({k2: v for k2, v in x['timings'].items() if k2 in KEEP})
                return x

        common_mod.json = JsonShim()
        plain = run_method.stream_chat

        def stream_chat_recorded(body, t_zero, cancel=None, deadline=None):
            rec._local.stats = {}
            r = plain(body, t_zero, cancel=cancel, deadline=deadline)
            with rec._lock:
                rec.rows.append({'task_id': rec.task, 'stats': dict(rec._local.stats), 'sent': r.get('sent'),
                                 'first': r.get('first'), 'end': r.get('end'),
                                 'completion_tokens': r.get('completion_tokens'), 'prompt_tokens': r.get('prompt_tokens'),
                                 'cancelled': bool(r.get('cancelled')), 'error': r.get('error'), 'finish': r.get('finish')})
            del rec._local.stats
            return r

        run_method.stream_chat = stream_chat_recorded


class MemLow:
    """The lowest MemAvailable (GiB) seen while it runs (report only; Part 4's MemoryGuard does the stopping)."""

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
