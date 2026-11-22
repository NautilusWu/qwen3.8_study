"""Shared time for tasks that run at the same time on one server (PLAN_20's rule), and two estimates of the steady
speed while all streams are busy.

Shared time: inside one block, every stretch of time is split evenly between the tasks running in it. A block's
shared times add up to its busy time (from the first request to the last reply, counting only time when at least
one task runs). With one stream, a task's shared time is its own wall time.

    rows: dicts with 'task_id', 'start_s', 'end_s' (seconds on the block's clock) and 'output_tokens'
"""


def cuts(rows):
    return sorted({x for r in rows for x in (r['start_s'], r['end_s'])})


def shares(rows, streams=4):
    """One block -> ({task: shared seconds}, busy seconds, seconds with fewer than `streams` running,
    seconds with `streams` running, output tokens assigned to those seconds).

    The last two give the locked plan's steady speed: each task's tokens are spread evenly over its own interval
    (an approximation: a task runs faster while fewer tasks share the GPU)."""
    c = cuts(rows)
    out, busy, under, full_s, full_tok = {r['task_id']: 0.0 for r in rows}, 0.0, 0.0, 0.0, 0.0
    for a, b in zip(c, c[1:]):
        on = [r for r in rows if r['start_s'] <= a and r['end_s'] >= b]
        if not on:
            continue
        busy += b - a
        under += (b - a) if len(on) < streams else 0.0
        if len(on) == streams:
            full_s += b - a
            full_tok += sum((r.get('output_tokens') or 0) * (b - a) / max(1e-9, r['end_s'] - r['start_s']) for r in on)
        for r in on:
            out[r['task_id']] += (b - a) / len(on)
    return out, busy, under, full_s, full_tok


def running_profile(rows):
    """One block -> {number of tasks running: seconds}."""
    c = cuts(rows)
    prof = {}
    for a, b in zip(c, c[1:]):
        k = sum(1 for r in rows if r['start_s'] <= a and r['end_s'] >= b)
        if k:
            prof[k] = prof.get(k, 0.0) + (b - a)
    return prof


def always_full(rows, streams=4):
    """One block -> the tasks that had exactly `streams` tasks running (themselves included) for their whole time.
    Their own tokens / own time is the speed of one stream at full load, with no spreading assumption."""
    c = cuts(rows)
    n = {(a, b): sum(1 for r in rows if r['start_s'] <= a and r['end_s'] >= b) for a, b in zip(c, c[1:])}
    return [r for r in rows if all(k == streams for (a, b), k in n.items() if a >= r['start_s'] and b <= r['end_s'])]
