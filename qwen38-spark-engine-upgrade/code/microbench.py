"""The speed microbench (report only): both engines on fixed public prompts, in rounds that alternate the engines
(default old new new old old new, as in the study). Every round starts the server as the main run does
(--parallel 4, context 81,920) and measures, greedy:
- one stream: short code and short chat (thinking off, up to 1,024 tokens), one long thinking answer (default
  effort, up to 8,192 tokens);
- 4 streams at once (4 different prompts, thinking off, up to 1,024 tokens each): per stream and in total;
- cold prompts of about 2K, 8K and 32K tokens (each round's prompts start with their own text, so no prompt cache
  can help): time to first token and prompt tokens per second.

    python3 microbench.py <part4>/code/strategies <run_id> [--order old,new,new,old,old,new]

Writes <this folder>/results/<run_id>/: microbench.jsonl, summary.json (medians per engine) and microbench.csv (the
columns of data/microbench.csv). Environment as in run_engines.py.
"""
import argparse
import csv
import json
import random
import statistics
import threading
import time
from pathlib import Path

import native_engine as E
import run_engines as RE

ROOT = Path(__file__).resolve().parent.parent
OFF = {'chat_template_kwargs': {'enable_thinking': False}}
CODE = 'Write a Python function merge_intervals(intervals) that merges overlapping intervals and returns them sorted. Include a docstring and two example calls.'
CHAT = 'Explain how a hash table handles collisions, comparing chaining and open addressing, in about 300 words.'
LONG = 'How many positive integers below 1,000,000 have digits that sum to 27? Explain your reasoning.'
FOUR = ['Write a Python class for a fixed-size ring buffer with push, pop and len, with a docstring.',
        'Describe the differences between TCP and UDP in about 250 words.',
        'Write a C++ function that returns the k smallest elements of a vector, with a short explanation.',
        'Summarize the main ideas of dynamic programming in about 250 words, with one small example.']
WORDS = ('river stone garden window paper silver orange forest market candle engine valley mirror ladder pocket '
         'thunder harbor violin meadow lantern copper bridge saddle feather marble canyon whistle blanket compass').split()
common = None


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def msg(text):
    return [{'role': 'system', 'content': common.SYSTEM}, {'role': 'user', 'content': text}]


def filler(n_words, tag):
    rng = random.Random(tag)
    lines = [f'Record {tag}.']
    for i in range(n_words // 12):
        lines.append(f'{i}: ' + ' '.join(rng.choice(WORDS) for _ in range(12)))
    return '\n'.join(lines) + '\n\nHow many lines above start with a number? Reply with the number only.'


def one(messages, max_tokens, extra=None):
    r = RE.stream_chat_recorded(common.body_for(messages, max_tokens, extra=extra), time.perf_counter())
    span = (r['end'] or 0) - (r['first'] or 0)
    return {'completion_tokens': r.get('completion_tokens'), 'prompt_tokens': r.get('prompt_tokens'),
            'ttft_s': round(r['first'], 3) if r.get('first') is not None else None, 'wall_s': round(r['end'], 2),
            'tok_s': round(r['completion_tokens'] / span, 2) if r.get('completion_tokens') and span > 0 else None,
            'finish': r.get('finish'), 'error': r.get('error')}


def round_(engine, n, out):
    """One round: one server start, then every measurement. The tag in the cold prompts is engine-round-words."""
    info = E.start(engine, 4, out / f'server_r{n}_{engine}.log', context=81920)
    rows = []
    add = lambda item, x: rows.append({'round': n, 'engine': engine, 'item': item, **x})
    one(msg('Compute 2+3. Reply exactly FINAL: 5.'), 64, OFF)                    # warm-up
    E.wait_idle()
    add('code', one(msg(CODE), 1024, OFF))
    E.wait_idle()
    add('chat', one(msg(CHAT), 1024, OFF))
    E.wait_idle()
    add('long_thinking', one(msg(LONG), 8192))
    E.wait_idle()
    box = [None] * 4
    t0 = time.perf_counter()

    def go(i):
        box[i] = one(msg(FOUR[i]), 1024, OFF)
    th = [threading.Thread(target=go, args=(i,)) for i in range(4)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.perf_counter() - t0
    add('four_streams', {'per_stream_tok_s': [b['tok_s'] for b in box],
                         'aggregate_tok_s': round(sum(b['completion_tokens'] or 0 for b in box) / wall, 2),
                         'errors': [b['error'] for b in box if b['error']]})
    E.wait_idle()
    for words in (1500, 6000, 24000):
        x = one(msg(filler(words, f'{engine}-{n}-{words}')), 8, OFF)
        x['prefill_tok_s'] = round(x['prompt_tokens'] / x['ttft_s'], 1) if x.get('prompt_tokens') and x.get('ttft_s') else None
        add(f'cold_prompt_{words}w', x)
        E.wait_idle()
    common.stop_server()
    return rows, info['startup_s']


def main():
    global common
    ap = argparse.ArgumentParser()
    ap.add_argument('strategies', help="Part 4's code/strategies folder")
    ap.add_argument('run_id')
    ap.add_argument('--order', default='old,new,new,old,old,new')
    a = ap.parse_args()
    common, _ = RE.setup(a.strategies)
    out = ROOT / 'results' / a.run_id
    out.mkdir(parents=True, exist_ok=False)
    guard = common.MemoryGuard()
    rows, starts = [], []
    try:
        for n, engine in enumerate(a.order.split(',')):
            try:
                r, s = round_(engine, n, out)
                rows += r
                starts.append({'round': n, 'engine': engine, 'startup_s': s})
            except Exception as exc:  # noqa: BLE001 - report-only step: note it and go on
                print(f'FAILED microbench round {n} {engine}: {exc!r}'[:400], flush=True)
                starts.append({'round': n, 'engine': engine, 'error': repr(exc)[:300]})
            finally:
                common.stop_server()
    finally:
        guard.close()
        common.stop_server()
    (out / 'microbench.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    summary = {'starts': starts, 'median': {}}
    for engine in ('old', 'new'):
        mine = [r for r in rows if r['engine'] == engine]
        s = {}
        for item in sorted({r['item'] for r in mine}):
            rs = [r for r in mine if r['item'] == item]
            if item == 'four_streams':
                s[item] = {'aggregate_tok_s': med(r['aggregate_tok_s'] for r in rs),
                           'per_stream_tok_s': med(v for r in rs for v in r['per_stream_tok_s'])}
            elif item.startswith('cold_prompt'):
                s[item] = {'prompt_tokens': rs[0].get('prompt_tokens'), 'ttft_s': med(r['ttft_s'] for r in rs),
                           'prefill_tok_s': med(r['prefill_tok_s'] for r in rs)}
            else:
                s[item] = {'tok_s': med(r['tok_s'] for r in rs), 'completion_tokens': [r['completion_tokens'] for r in rs]}
        summary['median'][RE.NAME[engine]] = s
    summary['memory_guard_fired'] = guard.fired
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    startup = {(s['round'], s['engine']): s.get('startup_s') for s in starts}
    with open(out / 'microbench.csv', 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['round', 'engine', 'item', 'round_startup_s', 'prompt_tokens', 'completion_tokens', 'ttft_s', 'wall_s',
                    'tok_s', 'prefill_tok_s', 'aggregate_tok_s', 'per_stream_tok_s', 'finish', 'error'])
        for x in rows:
            four = x['item'] == 'four_streams'
            vals = [x['round'], RE.NAME[x['engine']], x['item'], startup.get((x['round'], x['engine'])), x.get('prompt_tokens'),
                    x.get('completion_tokens'), x.get('ttft_s'), x.get('wall_s'), x.get('tok_s'), x.get('prefill_tok_s'),
                    x['aggregate_tok_s'] if four else None, ' '.join(str(v) for v in x['per_stream_tok_s']) if four else None,
                    x.get('finish'), '; '.join(x['errors']) if four else x.get('error')]
            w.writerow(['' if v is None else v for v in vals])
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
