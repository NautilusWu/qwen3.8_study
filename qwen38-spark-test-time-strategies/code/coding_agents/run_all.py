"""Run every agent on every task once, grade with hidden tests, record time and model usage.

    python3 run_all.py <run_id> [--agents qwen,opencode,codex,claude] [--tasks t1,t2] [--limit 1800]

Tasks: AGENT_TASKS_DIR (default ./tasks), one folder per task with repo/, prompt.txt and hidden/ (see
examples/agent_tasks/). The run starts the model server (serve_model.sh), the normalizing proxy and, for Claude Code,
LiteLLM (litellm_ctl.sh), and stops all three at the end. Results go to results/<run_id>/.

Per (task, agent): a fresh copy of the task repo as a git repository, a freshly restarted model server (empty
prompt cache, one untimed warm-up request), the agent in its own container through the normalizing proxy (fresh
proxy log per run), the hidden tests afterwards, and the TensorFold counters before and after the run (requests,
prompt/output tokens, prefill/decode seconds). The agent order is shuffled per task with a fixed seed, so no agent
always runs first or last. ../memory/memwatch.py logs memory for the whole run and stops the server if the host runs low;
the run then stops too.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENTS = ['qwen', 'opencode', 'codex', 'claude']
SERVER = os.environ.get('TF_CONTAINER', 'qwen38-tensorfold')
TASKS = Path(os.environ.get('AGENT_TASKS_DIR', HERE / 'tasks'))
MODEL = 'qwen3.8-27b-tensorfold-nvfp4'


def health():
    with urllib.request.urlopen('http://127.0.0.1:8091/health', timeout=10) as r:
        return json.load(r)


def wait_idle():
    for _ in range(600):
        h = health()
        if not h['busy'] and h['requests_running'] == 0:
            return h
        time.sleep(1)
    raise RuntimeError('model server stayed busy')


def restart_server():
    """Restart the model server (its prompt cache starts empty) and send one short untimed request."""
    t0 = time.time()
    subprocess.run(['docker', 'restart', '-t', '30', SERVER], check=True, capture_output=True)
    for _ in range(300):
        try:
            health()
            break
        except OSError:
            time.sleep(1)
    else:
        raise RuntimeError('model server did not come back')
    body = json.dumps({'model': MODEL, 'messages': [{'role': 'user', 'content': 'Say OK.'}], 'max_tokens': 16,
                       'reasoning_effort': 'none'}).encode()
    req = urllib.request.Request('http://127.0.0.1:8091/v1/chat/completions', body,
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as r:
        r.read()
    return round(time.time() - t0, 1)


def memory_during(log, t0, t1):
    rows = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
    rows = [r for r in rows if t0 <= r['t'] <= t1]
    if not rows:
        return {}
    return {'min_mem_available_gib': round(min(r['MemAvailable'] for r in rows) / 1024, 1),
            'max_gpu_gib': round(max(sum((r.get('gpu_mib') or {}).values()) for r in rows) / 1024, 1)}


def prepare(task, work):
    shutil.copytree(task / 'repo', work)
    git = ['git', '-c', 'user.name=task', '-c', 'user.email=task@example.invalid', '-C', str(work)]
    subprocess.run(['git', 'init', '-q', str(work)], check=True)
    subprocess.run(git + ['add', '-A'], check=True)
    subprocess.run(git + ['commit', '-q', '-m', 'task start'], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_id')
    ap.add_argument('--agents', default=','.join(AGENTS))
    ap.add_argument('--tasks', default='')
    ap.add_argument('--limit', type=int, default=1800)
    a = ap.parse_args()
    root = HERE / 'results' / a.run_id
    root.mkdir(parents=True, exist_ok=True)
    tasks = sorted(p for p in TASKS.iterdir() if p.is_dir())
    if a.tasks:
        tasks = [t for t in tasks if t.name.split('_')[0] in a.tasks.split(',')]
    agents = a.agents.split(',')
    done = set()
    results_file = root / 'results.jsonl'
    if results_file.exists():
        done = {(r['task'], r['agent']) for r in map(json.loads, results_file.read_text().splitlines())}
    memlog = root / 'memory.jsonl'
    fired = Path(str(memlog) + '.fired')
    if fired.exists():
        raise SystemExit(f'{fired} exists: the memory guard stopped the server; look at it before resuming')
    subprocess.run([str(HERE / 'serve_model.sh'), 'start'], check=True)
    if 'claude' in agents:
        subprocess.run([str(HERE / 'litellm_ctl.sh'), 'start'], check=True)
    watch = subprocess.Popen(['python3', str(HERE.parent / 'memory' / 'memwatch.py'), str(memlog)])
    try:
        run_tasks(a, root, tasks, agents, done, results_file, memlog, fired)
    finally:
        watch.terminate()
        subprocess.run([str(HERE / 'proxy_ctl.sh'), 'stop'], capture_output=True)
        subprocess.run([str(HERE / 'litellm_ctl.sh'), 'stop'], capture_output=True)
        subprocess.run([str(HERE / 'serve_model.sh'), 'stop'], capture_output=True)


def run_tasks(a, root, tasks, agents, done, results_file, memlog, fired):
    for task in tasks:
        order = agents[:]
        random.Random(f'harness-{task.name}').shuffle(order)
        for agent in order:
            if (task.name, agent) in done:
                continue
            if fired.exists():
                return
            out = root / task.name / agent
            if out.exists():
                shutil.rmtree(out)
            out.mkdir(parents=True)
            work = out / 'work'
            prepare(task, work)
            restart_s = restart_server()
            subprocess.run([str(HERE / 'proxy_ctl.sh'), 'start', str(out / 'proxy.jsonl')], check=True, capture_output=True)
            before = wait_idle()
            t0 = time.time()
            subprocess.run([str(HERE / 'run_agent.sh'), agent, str(work), str(task / 'prompt.txt'), str(out), str(a.limit)],
                           capture_output=True)
            wall = time.time() - t0
            if fired.exists():
                print(json.dumps({'task': task.name, 'agent': agent, 'stopped': 'memory guard fired'}), flush=True)
                return
            after = wait_idle()
            subprocess.run([str(HERE / 'grade.sh'), str(task), str(work), str(out / 'grade.json')], check=True)
            run = json.loads((out / 'run.json').read_text())
            grade = json.loads((out / 'grade.json').read_text())
            proxy = [json.loads(l) for l in (out / 'proxy.jsonl').read_text().splitlines()] if (out / 'proxy.jsonl').exists() else []
            delta = {k: after[k] - before[k] for k in ('requests_total', 'prompt_tokens_total', 'completion_tokens_total',
                                                        'cached_tokens_total', 'prefill_seconds_total', 'decode_seconds_total',
                                                        'rounds_total')}
            rec = {'task': task.name, 'agent': agent, 'passed': grade['passed'], 'grade_summary': grade['summary'],
                   'wall_s': round(wall, 1), 'exit_code': run['exit_code'], 'timed_out': run['timed_out'],
                   'proxy_requests': sum(1 for p in proxy if p.get('method') == 'POST'),
                   'proxy_errors': sum(1 for p in proxy if p.get('error') or (p.get('status') or 200) >= 400),
                   **delta, **memory_during(memlog, t0, t0 + wall), 'server_restart_s': restart_s,
                   'utc_start': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t0))}
            with open(results_file, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(json.dumps({k: rec.get(k) for k in ('task', 'agent', 'passed', 'wall_s', 'requests_total',
                                                      'completion_tokens_total', 'timed_out', 'proxy_errors',
                                                      'min_mem_available_gib', 'max_gpu_gib')}), flush=True)


if __name__ == '__main__':
    main()
