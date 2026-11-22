"""PLAN_18c's speed probe on your own tasks: one GGUF file with MTP off and with up to N drafts, decode speed from
llama.cpp's own timings, draft acceptance, and whether each reply is identical to the reply without MTP.

    python3 probe.py <part4>/code/strategies <run_id> --gguf <first .gguf shard> [--mtp 0,2,3,4]
                     [--split confirm] [--tasks id,id] [--max-tokens 8192]
                     [--examples-split dev] [--examples-max-tokens 65536] [--examples-limit-min 40]

For each N (0 = off) the server is started fresh (each in its own process) and the tasks run with D2, greedy, the
output capped at --max-tokens (8,192 in this study; the probe measures speed, it is not graded). With --examples-split
the tasks of that split also run afterwards, uncapped as in the study (65,536 tokens, 40 minutes) and graded.

Decode speed = sum of predicted_n / sum of predicted_ms over the capped tasks (llama.cpp's per-request timings).
Acceptance = accepted drafts / drafted tokens. The rule this study used to pick N: the fastest N, unless a smaller N
is within 97% of its speed, then the smallest such N.

Writes <this folder>/results/<run_id>/: one folder per N (requests.jsonl, stats.jsonl, server.log, probe.json) and
summary.json. Task text and model output are never printed (only ids, token counts, speeds; text equality as 0/1).
Environment: as llama_cpp.py (LLAMA_SERVER, FN_TEMPLATE, MTP_HEAD), and SANDBOX_CPUS as run_blocks.py.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import llama_cpp as L  # noqa: E402
import run_blocks as B  # noqa: E402

TOLERANCE = 0.97
PROBE_LIMIT_S = 120 * 60      # per capped probe task (a safety limit only)


def sha(s):
    return hashlib.sha256((s or '').encode()).hexdigest()


def decode_speed(stats):
    n = sum((s.get('stats') or {}).get('predicted_n') or 0 for s in stats)
    ms = sum((s.get('stats') or {}).get('predicted_ms') or 0 for s in stats)
    return round(n / ms * 1000, 3) if ms else None


def acceptance(stats):
    d = sum((s.get('stats') or {}).get('draft_n') or 0 for s in stats)
    a = sum((s.get('stats') or {}).get('draft_n_accepted') or 0 for s in stats)
    return {'draft_n': d, 'accepted': a, 'rate': round(a / d, 4) if d else None}


def run_tasks(common, RM, rec, tasks, path, guard, limit_s, max_tokens):
    RM.LIMIT_S, RM.MAX_TOKENS = limit_s, max_tokens
    start = len(rec.rows)
    with open(path, 'w') as f:
        for t in tasks:
            if guard.fired:
                raise RuntimeError('memory guard fired')
            common.wait_idle()
            rec.task = t['id']
            r = B.run_with_watchdog(lambda: RM.run_single(t, 'D2'), RM.LIMIT_S, common.stop_server)
            r.pop('token_ids', None)
            r.update(task_id=t['id'], category=t['category'], method='D2', max_tokens=RM.MAX_TOKENS)
            f.write(json.dumps(r) + '\n')
            f.flush()
            print(json.dumps({'task': t['id'], 'tokens': r.get('completion_tokens'), 'wall_s': round(r.get('wall_s') or 0, 1),
                              'error': r.get('error')}), flush=True)
    return rec.rows[start:]


def per_task(path, stats):
    by_id = {s['task_id']: s for s in stats}
    rows = []
    for r in B.jl(path):
        s = (by_id.get(r['task_id']) or {}).get('stats') or {}
        rows.append({'task_id': r['task_id'], 'completion_tokens': r.get('completion_tokens'), 'finish': r.get('finish'),
                     'error': r.get('error'), 'text_sha256': sha(r.get('text')),
                     **{k: s.get(k) for k in ('predicted_n', 'predicted_ms', 'draft_n', 'draft_n_accepted')}})
    return rows


def one(spec):
    """Child process: one server start with N drafts (0 = off)."""
    out = Path(spec['out'])
    out.mkdir(parents=True, exist_ok=False)
    L.PIDFILE = out / 'llama-server.pid'
    common = L.patch(spec['strategies'])
    import run_method as RM
    rec = L.Recorder(common, RM)
    load = common.load_tasks
    tasks = load(spec['split'])
    if spec['tasks']:
        tasks = [t for t in tasks if t['id'] in spec['tasks']]
    ex = load(spec['examples_split']) if spec['examples_split'] else []
    rep = {'mtp_n': spec['mtp_n'], 'tasks': [t['id'] for t in tasks], 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    low, guard = L.MemLow(), common.MemoryGuard()
    probe_stats, ex_stats = [], []
    try:
        rep['server'] = L.start(spec['gguf'], out / 'server.log', mtp_n=spec['mtp_n'])
        B.warm_up(common)
        probe_stats = run_tasks(common, RM, rec, tasks, out / 'requests.jsonl', guard, PROBE_LIMIT_S, spec['max_tokens'])
        if ex:
            (out / 'examples').mkdir()
            ex_stats = run_tasks(common, RM, rec, ex, out / 'examples' / 'requests.jsonl', guard,
                                 spec['examples_limit_s'], spec['examples_max_tokens'])
        rep['error'] = None
    except Exception as exc:  # noqa: BLE001
        rep['error'] = repr(exc)[:1000]
    finally:
        common.stop_server()
        guard.close()
        rep['mem_available_low_gib'], rep['memory_guard_fired'] = low.close(), guard.fired
    (out / 'stats.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in rec.rows))
    rep['per_task'] = per_task(out / 'requests.jsonl', probe_stats)
    rep['probe'] = {'requests': len(rep['per_task']), 'decode_tok_s': decode_speed(probe_stats),
                    'acceptance': acceptance(probe_stats),
                    'output_tokens': sum(r['completion_tokens'] or 0 for r in rep['per_task'])}
    if ex and (out / 'examples' / 'requests.jsonl').exists():
        B.grade(spec['strategies'], out, ['examples'], ex)
        g = {r['task_id']: r for r in B.jl(out / 'examples' / 'graded.jsonl')}
        rows = per_task(out / 'examples' / 'requests.jsonl', ex_stats)
        for r in rows:
            r['passed'] = bool((g.get(r['task_id']) or {}).get('passed'))
        rep['examples'] = {'correct': sum(r['passed'] for r in rows), 'of': len(ex), 'per_task': rows,
                           'tokens': sum(r['completion_tokens'] or 0 for r in rows)}
    rep['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    (out / 'probe.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps({'mtp_n': spec['mtp_n'], 'error': rep['error'], 'draft_loaded': (rep.get('server') or {}).get('draft_loaded'),
                      'decode_tok_s': rep['probe']['decode_tok_s'], 'acceptance': rep['probe']['acceptance']['rate'],
                      'examples_correct': (rep.get('examples') or {}).get('correct')}), flush=True)
    sys.exit(0 if rep['error'] is None else 2)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--one':
        one(json.loads(sys.argv[2]))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument('strategies')
    ap.add_argument('run_id')
    ap.add_argument('--gguf', required=True)
    ap.add_argument('--mtp', default='0,2,3,4')
    ap.add_argument('--split', default='confirm')
    ap.add_argument('--tasks', default='')
    ap.add_argument('--max-tokens', type=int, default=8192)
    ap.add_argument('--examples-split', default='')
    ap.add_argument('--examples-max-tokens', type=int, default=65536)
    ap.add_argument('--examples-limit-min', type=float, default=40)
    a = ap.parse_args()
    ns = [int(x) for x in a.mtp.split(',')]
    base = ROOT / 'results' / a.run_id
    base.mkdir(parents=True, exist_ok=False)
    for n in ns:
        spec = {'strategies': str(Path(a.strategies).resolve()), 'gguf': str(Path(a.gguf).resolve()), 'mtp_n': n,
                'split': a.split, 'tasks': [x for x in a.tasks.split(',') if x], 'max_tokens': a.max_tokens,
                'examples_split': a.examples_split, 'examples_max_tokens': a.examples_max_tokens,
                'examples_limit_s': int(a.examples_limit_min * 60), 'out': str(base / f'mtp{n}')}
        subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--one', json.dumps(spec)])
    reps = {n: json.loads((base / f'mtp{n}' / 'probe.json').read_text()) for n in ns if (base / f'mtp{n}' / 'probe.json').exists()}
    off = {r['task_id']: r['text_sha256'] for r in (reps.get(0) or {}).get('per_task') or []}
    summ = {}
    for n, r in reps.items():
        summ[n] = {'error': r['error'], 'draft_loaded': (r.get('server') or {}).get('draft_loaded'),
                   'startup_s': (r.get('server') or {}).get('startup_s'), 'decode_tok_s': r['probe']['decode_tok_s'],
                   'acceptance': r['probe']['acceptance']['rate'], 'output_tokens': r['probe']['output_tokens'],
                   'same_text_as_off': (sum(off.get(x['task_id']) == x['text_sha256'] for x in r['per_task'])
                                        if n and off else None),
                   'examples_correct': (r.get('examples') or {}).get('correct'),
                   'mem_available_low_gib': r['mem_available_low_gib']}
    v = {n: s['decode_tok_s'] for n, s in summ.items() if n and s['decode_tok_s']}
    chosen = min(n for n, x in v.items() if x >= TOLERANCE * max(v.values())) if v else None
    out = {'runs': summ, 'n_rule': 'fastest N; a smaller N within 97% of its speed is preferred', 'n_chosen': chosen}
    (base / 'summary.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
