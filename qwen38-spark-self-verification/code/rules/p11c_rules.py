"""PLAN_11c: the report-only arm B2, computed on top of p11_rules (unchanged) from one task's record.

B2 is B, except that T's tests must first fail on an empty code block: tests that pass there check nothing and count
as no tests (A's choice). The empty-code run starts when T has finished and takes its measured sandbox time
(`tester.empty_rt.container_s`), so no candidate can be accepted before it ends. Among candidates whose run of T's
tests passed, B2 takes the one ready first, ties broken by when its own run ended (B's order).
"""
import p11_rules


def arm_b2(rec, limit):
    t = rec['tester']
    if t['tests'] is None:
        return p11_rules.arm_b(rec, limit)
    a = p11_rules.arm_a(rec, limit)
    e = t['empty_rt']
    ready = t['end_s'] + e['container_s']
    if e['passed']:
        return {**a, 'time_s': min(max(a['time_s'], ready), limit), 'why': 'pass on empty code'}
    done = p11_rules.t_done(rec)
    by_id = {b['branch']: b for b in p11_rules.candidates(rec)}
    passing = sorted((max(d, ready), d, i) for i, d in done.items()
                     if by_id[i]['t_run']['passed'] and max(d, ready) <= limit)
    if passing:
        t2, _, i = passing[0]
        return {'choice': i, 'time_s': t2, 'timed_out': False, 'verified': True, 'why': 'passed T'}
    last = max([b['end_s'] for b in rec['branches']] + [t['end_s'], ready] + list(done.values()))
    return {**a, 'time_s': min(last, limit), 'why': 'none passed T'}
