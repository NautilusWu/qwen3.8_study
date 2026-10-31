"""Time replies under different MTP draft settings; drafts are verified exactly, so every setting must give the
same tokens (checked by hash against the first setting), and only the time should change.

    python3 mtp_sweep.py <part4>/code/strategies <run_id> [--settings 0.7:6,0.6:6,0:4,serial] \
        [--deep ex_primes,ex_gcd_cpp] [--low ex_primes,ex_trains,ex_palindrome,ex_gcd_cpp,ex_five_numbers]

A setting is THRESHOLD:DRAFTS (--mtp-confidence, --mtp-drafts) or 'serial' (--no-drafts). Tasks come from
TASKS_DIR/dev.jsonl (Part 4's examples/tasks by default in its smoke test). The server restarts for every setting.
Writes <part4>/code/strategies/results/<run_id>/sweep.jsonl, one line per reply, and prints a table.
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.request

import flash_next


def ask(common, messages, max_tokens, extra):
    body = common.body_for([{'role': 'system', 'content': common.SYSTEM}] + messages, max_tokens, extra=extra)
    body.update(stream=False)
    body.pop('stream_options', None)
    t0 = time.perf_counter()
    req = urllib.request.Request(f'http://127.0.0.1:{common.PORT}/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=7200) as r:
        resp = json.load(r)
    tf = dict(resp.get('tensorfold') or {})
    ids = tf.pop('token_ids', None)
    return {'wall_s': round(time.perf_counter() - t0, 2), 'tokens': (resp.get('usage') or {}).get('completion_tokens'),
            'ids_sha': hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16] if ids else None, 'tf': tf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('strategies')
    ap.add_argument('run_id')
    ap.add_argument('--settings', default='0.7:6,0.6:6,0:3,0:4,serial')
    ap.add_argument('--deep', default='ex_primes,ex_gcd_cpp')
    ap.add_argument('--low', default='ex_primes,ex_trains,ex_palindrome,ex_gcd_cpp,ex_five_numbers')
    ap.add_argument('--split', default='dev')
    ap.add_argument('--context', type=int, default=81920)
    a = ap.parse_args()
    flash_next.EXTRA = ()                      # the settings below replace the default draft flags
    common = flash_next.patch(a.strategies)
    out = common.HERE / 'results' / a.run_id
    out.mkdir(parents=True, exist_ok=False)
    tasks = {t['id']: t for t in common.load_tasks(a.split)}
    plan = [(tasks[i], 'deep') for i in a.deep.split(',') if i] + [(tasks[i], 'low') for i in a.low.split(',') if i]
    ref, table = {}, []
    for s in a.settings.split(','):
        flags = ('--no-drafts',) if s == 'serial' else ('--mtp-confidence', s.split(':')[0], '--mtp-drafts', s.split(':')[1])
        guard = common.MemoryGuard()
        recs = []
        try:
            common.start_server(1, out / f'server_{s.replace(":", "_")}.log', context=a.context, extra_args=flags)
            for t, effort in plan:
                r = ask(common, t['messages'], 65536, {'reasoning_effort': 'low'} if effort == 'low' else {})
                r.update(setting=s, task=t['id'], effort=effort)
                ref.setdefault((t['id'], effort), r['ids_sha'])
                r['same_tokens'] = r['ids_sha'] == ref[(t['id'], effort)]
                print(json.dumps(r), flush=True)
                recs.append(r)
        finally:
            guard.close()
            common.stop_server()
        with open(out / 'sweep.jsonl', 'a') as f:
            f.writelines(json.dumps(r) + '\n' for r in recs)
        table.append((s, sum(r['wall_s'] for r in recs if r['effort'] == 'deep'),
                      sum(r['wall_s'] for r in recs if r['effort'] == 'low'), all(r['same_tokens'] for r in recs)))
    print(f"\n{'setting':<10}{'deep s':>10}{'low s':>10}  same tokens")
    for s, d, l, same in table:
        print(f'{s:<10}{d:>10.1f}{l:>10.1f}  {same}')
    if not all(row[3] for row in table):
        sys.exit('some replies changed between settings')


if __name__ == '__main__':
    main()
