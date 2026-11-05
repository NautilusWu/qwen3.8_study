"""Replay stored replies through loopdetect.LoopDetector, 256 characters at a time as a stream would arrive, and
report where it fires. Input: graded.jsonl files from Part 4's runner (fields task_id, text, completion_tokens or
tokens, finish, passed; vote runs: branches). No GPU needed.

    python3 replay_detector.py <run>/graded.jsonl [<run>/graded.jsonl ...]
"""
import json
import sys

from loopdetect import LoopDetector


def replay(text, tokens):
    reasoning = text.split('</think>')[0]
    d = LoopDetector()
    for end in range(256, len(reasoning) + 256, 256):
        hit = d.feed(reasoning[:end])
        if hit:
            return hit | {'at_tokens': round(hit['at'] * tokens / max(1, len(text)))}
    return None


for path in sys.argv[1:]:
    replies = []
    for line in open(path):
        r = json.loads(line)
        for b in r.get('branches') or [r]:
            if not b.get('cancelled'):
                replies.append((r['task_id'], b))
    fired = [(t, b, h) for t, b in replies
             if (h := replay(b.get('text') or '', b.get('completion_tokens') or b.get('tokens') or 1))]
    capped = sum(b.get('finish') == 'length' for _, b in replies)
    answered = [(t, h['rule']) for t, b, h in fired if '</think>' in (b.get('text') or '')]
    print(f'{path}: {len(replies)} replies, {capped} hit the output limit; detector fired on {len(fired)} '
          f'(on replies that did answer: {len(answered)} {answered})')
    for t, b, h in fired:
        print(f'   {t}: {h["rule"]} at about {h["at_tokens"]} tokens; reply {b.get("completion_tokens") or b.get("tokens")} '
              f'tokens, finish {b.get("finish")}, passed {b.get("passed")}')
