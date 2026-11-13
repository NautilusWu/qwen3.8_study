"""PLAN_11: the arms A, B and C, computed from one task's record (run_p11.py writes it, analyze_p11.py reads it).

All times are seconds from the moment the task's five requests were sent. A run of the model's tests on a candidate
starts as soon as both the candidate and the tests exist and takes its measured sandbox time (container start
included); runs do not wait for each other. Nothing finished after the task's limit is used: an arm that has not
accepted a candidate by then returns its fallback at the limit.

Record fields used here:
  branches[i]: branch, end_s, candidate (finished, not cut, code block in the answer), t_run / k_run
               ({passed, container_s}, only on candidates)
  tester:      end_s, tests (None: no usable tests)
  critic:      duration_s (its own generation time), tests; None when it was not run
"""


def candidates(rec):
    return [b for b in rec['branches'] if b['candidate']]


def arm_a(rec, limit):
    """CL4's rule for code: the first branch that finished with a code block."""
    cands = sorted((b['end_s'], b['branch']) for b in candidates(rec))
    if cands:
        t, choice = cands[0]
    else:
        t, choice = max(b['end_s'] for b in rec['branches']), None
    return {'choice': choice, 'time_s': min(t, limit), 'timed_out': t >= limit - 1, 'verified': False}


def t_done(rec):
    """{branch: time its run of T's tests ended}, for every candidate; {} when T gave no tests."""
    t_end = rec['tester']['end_s']
    if rec['tester']['tests'] is None:
        return {}
    return {b['branch']: max(b['end_s'], t_end) + b['t_run']['container_s'] for b in candidates(rec)}


def arm_b(rec, limit):
    """The candidate whose run of T's tests is the first to end with a pass; otherwise A's choice, unverified."""
    a = arm_a(rec, limit)
    t_end = rec['tester']['end_s']
    if rec['tester']['tests'] is None:            # nothing to check with: A's choice once A has chosen and T has ended
        return {**a, 'time_s': min(max(a['time_s'], t_end), limit), 'why': 'no tests'}
    done = t_done(rec)
    by_id = {b['branch']: b for b in candidates(rec)}
    passing = sorted((t, i) for i, t in done.items() if by_id[i]['t_run']['passed'] and t <= limit)
    if passing:
        t, i = passing[0]
        return {'choice': i, 'time_s': t, 'timed_out': False, 'verified': True, 'why': 'passed T'}
    last = max([b['end_s'] for b in rec['branches']] + [t_end] + list(done.values()))
    return {**a, 'time_s': min(last, limit), 'why': 'none passed T'}


def arm_c(rec, limit):
    """B, then K's extra tests: the candidate (among those that passed T) whose run of K's tests is the first to end
    with a pass. None passes: B's choice, unverified. K not run or without usable tests: B's choice and label."""
    b = arm_b(rec, limit)
    if not b['verified']:
        return {**b, 'k': 'not run'}
    k = rec.get('critic')
    if k is None:
        return {**b, 'k': 'no time'}
    t_k = b['time_s'] + k['duration_s']
    if t_k >= limit:
        return {**b, 'time_s': limit, 'k': 'no time'}
    if k['tests'] is None:
        return {**b, 'time_s': min(t_k, limit), 'k': 'no tests'}
    by_id = {x['branch']: x for x in candidates(rec)}
    passed_t = {i: t for i, t in t_done(rec).items() if by_id[i]['t_run']['passed']}
    k_done = {i: max(t_k, t) + by_id[i]['k_run']['container_s'] for i, t in passed_t.items()}
    ok = sorted((t, i) for i, t in k_done.items() if by_id[i]['k_run']['passed'] and t <= limit)
    if ok:
        t, i = ok[0]
        return {'choice': i, 'time_s': t, 'timed_out': False, 'verified': True, 'k': 'passed K'}
    last = max([x['end_s'] for x in rec['branches']] + [rec['tester']['end_s']] + list(k_done.values()))
    return {**b, 'time_s': min(last, limit), 'verified': False, 'k': 'none passed K'}
