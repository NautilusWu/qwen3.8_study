"""Run the paired tests of the four runs with code/stats.py on the published per-task files, and check them against the
outputs of the locked analyses in data/tests.json (no GPU, no task files needed).

    python3 code/tests_check.py           every call: compare and print one line each
    python3 code/tests_check.py --print   also print the full stats.py output of each call

stats.py: exact two-sided McNemar test; paired bootstrap 95% intervals (10,000 resamples, seed 2026) of the accuracy
difference and of the ratio of minutes per correct answer; each task capped at 40 minutes. Its `p_holm` is over the
pairs of one call; the plans read the unadjusted p of the one pair they name.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    spec = json.loads((ROOT / 'data' / 'tests.json').read_text())
    bad = 0
    for call in spec['stats_calls']:
        r = subprocess.run([sys.executable, str(ROOT / 'code' / 'stats.py'), '--limit-min', '40', '--boot', '10000',
                            '--seed', '2026', *call['args']], cwd=ROOT, capture_output=True, text=True, check=True)
        got = json.loads(r.stdout)
        same = all(got[k] == call[k] for k in ('tasks', 'totals', 'pairs'))
        bad += not same
        if '--print' in sys.argv:
            print(json.dumps({k: got[k] for k in ('tasks', 'totals', 'pairs')}, indent=1))
        print(f"{call['name']:34s} {'matches the locked analysis' if same else 'DIFFERS'} "
              f"({len(call['pairs'])} pairs, {call['tasks']} tasks)")
    print(f'\ntests: {"all match" if not bad else f"{bad} differ"}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
