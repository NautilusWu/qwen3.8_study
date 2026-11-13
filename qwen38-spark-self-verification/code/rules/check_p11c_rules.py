"""Checks of p11c_rules on hand-made records (no model, no sandbox): python3 check_p11c_rules.py"""
import p11_rules as P
import p11c_rules as Q
from check_p11_rules import LIMIT, check, rec


def with_empty(r, passed, container_s):
    r['tester']['empty_rt'] = {'passed': passed, 'container_s': container_s}
    return r


T_RUNS = {0: (False, 2), 1: (True, 3), 2: (True, 1), 3: (True, 1)}

# the empty-code run ends before the first passing run: B2 = B
r = with_empty(rec([60, 90, 120, 200], 100, T_RUNS), False, 2)
check(Q.arm_b2(r, LIMIT), choice=1, time_s=103, verified=True, why='passed T')
check(P.arm_b(r, LIMIT), choice=1, time_s=103, verified=True)

# the empty-code run ends later: B2 waits for it; both passing runs are ready then, B's order decides (branch 1, 103)
r = with_empty(rec([60, 90, 120, 200], 100, T_RUNS), False, 30)
check(Q.arm_b2(r, LIMIT), choice=1, time_s=130, verified=True)
r = with_empty(rec([60, 90, 120, 200], 100, {0: (False, 2), 1: (True, 25), 2: (True, 1), 3: (True, 1)}), False, 30)
check(Q.arm_b2(r, LIMIT), choice=2, time_s=130, verified=True)       # branch 2 done 121 before branch 1 done 125

# tests that pass on empty code: A's choice, unverified, once A has chosen and the empty-code run has ended
r = with_empty(rec([60, 90, 120, 200], 100, T_RUNS), True, 4)
check(Q.arm_b2(r, LIMIT), choice=0, time_s=104, verified=False, why='pass on empty code')
r = with_empty(rec([300, 400], 100, {0: (True, 2), 1: (True, 2)}), True, 4)
check(Q.arm_b2(r, LIMIT), choice=0, time_s=300, verified=False, why='pass on empty code')

# T without tests: exactly B
r = rec([60, 90], 300, {}, tests=False)
check(Q.arm_b2(r, LIMIT), **P.arm_b(r, LIMIT))

# no candidate passes T: A's choice when everything, the empty-code run included, has ended
r = with_empty(rec([60, 90, 500], 100, {0: (False, 2), 1: (False, 2), 2: (False, 5)}), False, 3)
check(Q.arm_b2(r, LIMIT), choice=0, time_s=505, verified=False, why='none passed T')
r = with_empty(rec([60, 90], 600, {0: (False, 2), 1: (False, 2)}), False, 7)
check(Q.arm_b2(r, LIMIT), choice=0, time_s=607, verified=False, why='none passed T')

# the empty-code run pushes a pass past the limit: fallback to A's choice at the limit
r = with_empty(rec([60, 90], 2390, {0: (False, 2), 1: (True, 2)}), False, 20)
check(P.arm_b(r, LIMIT), choice=1, time_s=2392, verified=True)
check(Q.arm_b2(r, LIMIT), choice=0, time_s=LIMIT, verified=False, why='none passed T')
print('p11c_rules checks passed')
