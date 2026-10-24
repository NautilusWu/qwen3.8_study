"""Shared pieces: the TensorFold server, streaming and plain requests, the code sandbox, grading, a memory guard.

Configure with environment variables (no paths are built in):
  TASKS_DIR    folder with your task files dev.jsonl / hard.jsonl / confirm.jsonl (one JSON object per line:
               id, category, messages, grader); the task sets used in this study are not published
  GRADER_DIR   folder containing a Python package `grading` with `grade(grader: dict, text: str, thinking: bool)
               -> (score, passed, detail)`; it runs inside the sandbox image, network off
  HF_DIR       Hugging Face cache with the model snapshots (default ~/.cache/huggingface)
  TF_CACHE_DIR TensorFold kernel cache (default ~/.cache/qwen38-tensorfold-cache)
  SANDBOX_IMAGE an image with python3 and g++ for running code (default agent-harness:0.1, built by scripts/setup.sh)
"""
import http.client
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
import os
TASKS_DIR = Path(os.environ.get('TASKS_DIR', HERE / 'tasks'))
GRADER_DIR = Path(os.environ.get('GRADER_DIR', HERE / 'grader'))
HF_DIR = os.environ.get('HF_DIR', str(Path.home() / '.cache/huggingface'))
IMAGE = 'qwen38-tensorfold-repro:0.6.0-trace'
SANDBOX_IMAGE = os.environ.get('SANDBOX_IMAGE', 'agent-harness:0.1')
PORT = 8114
NAME = 'tts-tf-server'
TARGET = '/hf/hub/models--nvidia--Qwen3.8-27B-NVFP4/snapshots/482ca0f3832238542f8f5295dde86b5f22711d80'
DRAFT = '/hf/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/50307d4c4cde6860d4eee73e2547cd786fe8e8a4'
CACHE = os.environ.get('TF_CACHE_DIR', str(Path.home() / '.cache/qwen38-tensorfold-cache'))
SYSTEM = 'You are a helpful assistant.'
MODEL = 'Qwen3.8-27B'

from answers import extract_code, final_section, last_block, last_marker, parse_number  # noqa: E402  (text parsing only)


def load_tasks(split='dev'):
    path = TASKS_DIR / f'{split}.jsonl'
    return [json.loads(l) for l in path.read_text().splitlines()]


# --- server ---------------------------------------------------------------------------------------------------

def health():
    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=5) as r:
        return json.load(r)


def start_server(parallel, log_path, context=32768):
    stop_server()
    busy = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                          capture_output=True, text=True).stdout.strip()
    assert not busy, 'GPU is in use: stop the daily service (docker stop qwen38-tensorfold) first'
    args = ['docker', 'run', '-d', '--name', NAME, '--gpus', 'all', '--ipc', 'host', '--network', 'host',
            '--cpuset-cpus', '5-9,15-19', '--memory', '80g', '--memory-swap', '80g',
            '--ulimit', 'memlock=-1', '--ulimit', 'stack=67108864',
            '-e', 'TENSORFOLD_KV_DTYPE=fp8', '-e', 'TENSORFOLD_MEMORY_RESERVE_GIB=40', '-e', 'TENSORFOLD_KV_POOL_GIB=0',
            '-e', 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',
            '-v', f'{HF_DIR}:/hf:ro', '-v', f'{CACHE}:/cache', IMAGE,
            'tensorfold', 'serve', TARGET, '--drafter', DRAFT, '--backend', 'cuda', '--name', MODEL,
            '--host', '127.0.0.1', '--port', str(PORT), '--parallel', str(parallel), '--context', str(context),
            '--temperature', '0', '--top-p', '1', '--top-k', '0', '--thinking', '--prefill-fp8']
    subprocess.run(args, check=True, capture_output=True)
    t0 = time.time()
    while True:
        try:
            health()
            break
        except OSError:
            running = subprocess.run(['docker', 'inspect', '-f', '{{.State.Running}}', NAME],
                                     capture_output=True, text=True).stdout.strip()
            if running != 'true' or time.time() - t0 > 1800:
                tail = subprocess.run(['docker', 'logs', '--tail', '40', NAME], capture_output=True, text=True)
                raise RuntimeError('server did not start: ' + tail.stdout + tail.stderr)
            time.sleep(3)
    logs = subprocess.run(['docker', 'logs', NAME], capture_output=True, text=True)
    Path(log_path).write_text(logs.stdout + logs.stderr)
    image_id = subprocess.run(['docker', 'image', 'inspect', '-f', '{{.Id}}', IMAGE], capture_output=True,
                              text=True).stdout.strip()
    return {'argv': args, 'image_id': image_id, 'startup_s': round(time.time() - t0, 1),
            'startup_lines': [l for l in (logs.stdout + logs.stderr).splitlines() if l.startswith('[tensorfold]')]}


def wait_idle(timeout=1800):
    """Before each task: nothing left running on the server (e.g. a request a client gave up on)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = health()
        if not h.get('busy') and not h.get('requests_running'):
            return round(time.time() - t0, 1)
        time.sleep(1)
    raise RuntimeError('server stayed busy')


def stop_server():
    subprocess.run(['docker', 'rm', '-f', NAME], capture_output=True)


class MemoryGuard:
    """Stops the server if the host runs low on memory (on the GB10 the GPU allocates from system memory)."""

    def __init__(self, floor_gib=12):
        self.floor, self.fired, self._stop = floor_gib, False, threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.wait(1):
            for line in Path('/proc/meminfo').read_text().splitlines():
                if line.startswith('MemAvailable:') and int(line.split()[1]) / 2**20 < self.floor:
                    self.fired = True
                    stop_server()
                    return

    def close(self):
        self._stop.set()


# --- requests ---------------------------------------------------------------------------------------------------

def body_for(messages, max_tokens, sampling=None, seed=1234, extra=None):
    body = {'model': MODEL, 'messages': messages, 'max_tokens': max_tokens, 'seed': seed, 'stream': True,
            'stream_options': {'include_usage': True}, 'return_token_ids': True,
            'chat_template_kwargs': {'enable_thinking': True},
            **(sampling or {'temperature': 0, 'top_p': 1, 'top_k': 0})}
    body.update(extra or {})
    return body


def stream_chat(body, t_zero, cancel=None, deadline=None):
    """One streaming request. Stops early (result['cancelled']) when ``cancel`` is set or ``deadline`` passes."""
    out = {'sent': time.perf_counter() - t_zero}
    conn = http.client.HTTPConnection('127.0.0.1', PORT, timeout=1800)
    reasoning, answer, usage, stats, finish, first = '', '', {}, {}, None, None
    try:
        conn.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json'})
        resp = conn.getresponse()
        if resp.status != 200:
            out.update(error=f'HTTP {resp.status}: {resp.read()[:300]!r}', end=time.perf_counter() - t_zero)
            return out
        while True:
            if (cancel is not None and cancel.is_set()) or (deadline is not None and time.perf_counter() > deadline):
                out['cancelled'] = True
                break
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
            if x.get('error'):
                out['error'] = str(x['error'])[:300]
                break
            usage.update(x.get('usage') or {})
            stats.update(x.get('tensorfold') or {})
            for c in x.get('choices') or []:
                finish = c.get('finish_reason') or finish
                d = c.get('delta') or {}
                if d.get('reasoning_content') or d.get('content'):
                    first = first if first is not None else time.perf_counter() - t_zero
                reasoning += d.get('reasoning_content') or ''
                answer += d.get('content') or ''
    finally:
        conn.close()
    out.update(end=time.perf_counter() - t_zero, first=first, reasoning=reasoning, answer=answer, finish=finish,
               completion_tokens=usage.get('completion_tokens'), prompt_tokens=usage.get('prompt_tokens'),
               token_ids=stats.get('token_ids'))
    # the graded text: thinking, then </think> and the answer (no answer: thinking never closed, graded as failure)
    out['text'] = reasoning + ('</think>\n' + answer if answer else '')
    return out


def plain_chat(body, timeout=1800):
    """One non-streaming request (the agent loop): the message, usage and wall time."""
    body = {**body, 'stream': False}
    body.pop('stream_options', None)
    t0 = time.perf_counter()
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as exc:            # keep the server's reason (e.g. context exceeded)
        raise RuntimeError(f'HTTP {exc.code}: {exc.read()[:400]!r}') from None
    choice = resp['choices'][0]
    return {'message': choice['message'], 'finish': choice.get('finish_reason'), 'usage': resp.get('usage') or {},
            'wall_s': time.perf_counter() - t0}


# --- answers (for voting; never the expected answer) -----------------------------------------------------------

def answer_key(task, text):
    """What two branches must share to count as the same answer, by the task's answer format; None: no answer."""
    final = final_section(text, True)
    if final is None:
        return None
    kind = task['grader']['type']
    if kind == 'numeric':
        line = last_marker(final, 'Final answer:')
        val = parse_number(line) if line is not None else None
        return None if val is None else ('num', float(f'{val:.6g}'))
    if kind == 'exact_output':
        block = last_block(final)
        return None if block is None else ('out', tuple(l.rstrip() for l in block.strip('\n').splitlines()))
    if kind == 'python_check':
        line = last_marker(final, 'Final answer:')
        body = line if line else final
        return ('text', ' '.join(body.split()).lower())
    if kind in ('python_tests', 'cpp_tests'):
        langs = ('python', 'py') if kind == 'python_tests' else ('cpp', 'c++', 'cc', 'cxx')
        return None if extract_code(final, langs) is None else ('code', None)   # code is never voted on
    return ('text', ' '.join(final.split()).lower())


def votes_on(task):
    return task['grader']['type'] in ('numeric', 'exact_output', 'python_check')


# --- code sandbox (the agent's tool) and grading ----------------------------------------------------------------

SANDBOX = ['docker', 'run', '--rm', '--network', 'none', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
           '--read-only', '--memory', '2g', '--memory-swap', '2g', '--cpus', '2', '--pids-limit', '128',
           '--tmpfs', '/tmp:rw,exec,size=256m', '--user', '1000:1000']


def run_code(language, code, timeout=20, limit=4000):
    """Run the agent's code with no network and nothing but the code itself; stdout+stderr, truncated."""
    (HERE / '.sandbox').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=HERE / '.sandbox') as d:
        d = Path(d)
        d.chmod(0o755)
        if language == 'cpp':
            (d / 'main.cpp').write_text(code)
            script = f'cd /tmp && g++ -std=c++17 -O1 -o a /code/main.cpp 2>&1 && timeout {timeout} ./a'
        else:
            (d / 'main.py').write_text(code)
            script = f'cd /tmp && timeout {timeout} python3 /code/main.py'
        for f in d.iterdir():
            f.chmod(0o644)
        t0 = time.perf_counter()
        try:
            r = subprocess.run(SANDBOX + ['-v', f'{d}:/code:ro', '--entrypoint', 'bash', SANDBOX_IMAGE, '-c', script],
                               capture_output=True, text=True, timeout=timeout + 60)
            output, rc = (r.stdout + r.stderr), r.returncode
        except subprocess.TimeoutExpired:
            output, rc = 'timed out', 124
    if rc == 124:
        output += f'\n[timed out after {timeout} s]'
    if len(output) > limit:
        output = output[:limit] + f'\n[... {len(output) - limit} more characters cut]'
    return {'exit_code': rc, 'output': output, 'wall_s': round(time.perf_counter() - t0, 2)}


GRADE_SCRIPT = '''import json
from pathlib import Path
from grading import grade
out = []
for item in json.loads(Path('/inputs/grade_inputs.json').read_text()):
    try:
        score, passed, detail = grade(item['grader'], item['text'], thinking=item['thinking'])
        error = None
    except Exception as exc:
        score, passed, detail, error = 0.0, False, repr(exc), type(exc).__name__
    out.append({'key': item['key'], 'passed': passed, 'score': score, 'detail': str(detail)[:300], 'grader_error': error})
print(json.dumps(out))
'''


def grade_all(items, workdir):
    """items: [{'key', 'grader', 'text', 'thinking'}] -> {key: result}; graded in the study's sandbox."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / 'grade_inputs.json').write_text(json.dumps(items))
    (workdir / 'grade.py').write_text(GRADE_SCRIPT)
    args = SANDBOX + ['-e', 'PYTHONPATH=/grader', '-e', 'PYTHONDONTWRITEBYTECODE=1',
                      '-v', f'{GRADER_DIR}:/grader:ro', '-v', f'{workdir}:/inputs:ro',
                      '--entrypoint', 'python3', SANDBOX_IMAGE, '/inputs/grade.py']
    r = subprocess.run(args, capture_output=True, text=True, timeout=3600, check=True)
    return {x['key']: x for x in json.loads(r.stdout)}
