"""Checks of p11_rules on hand-made records (no model, no sandbox): python3 check_p11_rules.py"""
import p11_rules as P

LIMIT = 2400


def rec(ends, t_end, t_runs, tests=True, critic=None, k_runs=None, cand=None):
    """t_runs / k_runs: {branch: (passed, container_s)}; cand: branches that are candidates (default: all)."""
    bs = []
    for i, e in enumerate(ends):
        b = {'branch': i, 'end_s': e, 'candidate': cand is None or i in cand}
        if b['candidate'] and i in t_runs:
            b['t_run'] = {'passed': t_runs[i][0], 'container_s': t_runs[i][1]}
        if b['candidate'] and k_runs and i in k_runs:
            b['k_run'] = {'passed': k_runs[i][0], 'container_s': k_runs[i][1]}
        bs.append(b)
    return {'branches': bs, 'tester': {'end_s': t_end, 'tests': 'x' if tests else None}, 'critic': critic}


def check(got, **want):
    for k, v in want.items():
        assert got[k] == v, (k, got[k], v, got)


# normal: B takes the first run of T's tests that passes, C the first run of K's tests that passes
r = rec([60, 90, 120, 200], 100, {0: (False, 2), 1: (True, 3), 2: (True, 1), 3: (True, 1)},
        critic={'duration_s': 50, 'tests': 'k'}, k_runs={0: (True, 1), 1: (False, 2), 2: (True, 1), 3: (True, 1)})
check(P.arm_a(r, LIMIT), choice=0, time_s=60, timed_out=False, verified=False)
check(P.arm_b(r, LIMIT), choice=1, time_s=103, verified=True)
check(P.arm_c(r, LIMIT), choice=2, time_s=154, verified=True, k='passed K')   # K at 103 + 50; branch 2: 153 + 1

# K rejects every candidate that passed T: B's choice, unverified, once everything is known
r2 = rec([60, 90, 120, 200], 100, {0: (False, 2), 1: (True, 3), 2: (True, 1), 3: (True, 1)},
         critic={'duration_s': 50, 'tests': 'k'}, k_runs={0: (True, 1), 1: (False, 2), 2: (False, 1), 3: (False, 4)})
check(P.arm_c(r2, LIMIT), choice=1, time_s=205, verified=False, k='none passed K')   # branch 3: T done 201, K 201+4

# K without usable tests: B's choice and label, after K's time
r3 = rec([60, 90], 100, {0: (True, 2), 1: (True, 2)}, critic={'duration_s': 40, 'tests': None})
check(P.arm_c(r3, LIMIT), choice=0, time_s=142, verified=True, k='no tests')

# K not run (too little time left)
check(P.arm_c(rec([60], 100, {0: (True, 2)}), LIMIT), choice=0, time_s=102, verified=True, k='no time')

# T without tests: A's choice when both A and T are done
r4 = rec([60, 90], 300, {}, tests=False)
check(P.arm_b(r4, LIMIT), choice=0, time_s=300, verified=False, why='no tests')
check(P.arm_c(r4, LIMIT), choice=0, verified=False, k='not run')

# no candidate passes T: A's choice when all branches, T and all runs are done
r5 = rec([60, 90, 500], 100, {0: (False, 2), 1: (False, 2), 2: (False, 5)})
check(P.arm_b(r5, LIMIT), choice=0, time_s=505, verified=False, why='none passed T')

# no candidate at all: A has no choice; B the same
r6 = rec([60, 2400], 100, {}, cand=set())
check(P.arm_a(r6, LIMIT), choice=None, time_s=2400, timed_out=True)
check(P.arm_b(r6, LIMIT), choice=None, timed_out=True, verified=False)

# a pass after the limit is not used: fallback to A's choice at the limit
r7 = rec([60, 2398], 100, {0: (False, 2), 1: (True, 5)})
check(P.arm_b(r7, LIMIT), choice=0, time_s=LIMIT, timed_out=False, verified=False)

# K would end after the limit
r8 = rec([60], 100, {0: (True, 2)}, critic={'duration_s': 2300, 'tests': 'k'}, k_runs={0: (True, 1)})
check(P.arm_c(r8, LIMIT), choice=0, time_s=LIMIT, verified=True, k='no time')

# only some branches are candidates: A ignores the others
r9 = rec([30, 60, 90], 100, {1: (True, 1), 2: (True, 1)}, cand={1, 2})
check(P.arm_a(r9, LIMIT), choice=1, time_s=60)
check(P.arm_b(r9, LIMIT), choice=1, time_s=101, verified=True)
print('p11_rules checks passed')
