"""Summarize a harness run: per agent and per task, with the user's lead metric (time per successful task).

    python3 analyze.py <run_id> [--repeat <run_id> ...]

Success is counted two ways: strict (hidden tests pass and the agent finished by itself) and lenient (hidden tests
pass, even if the agent was stopped at the time limit). Minutes per success divide all the agent's wall time,
failures included, by its successes: what a user waits per working result. --repeat runs give the same
(task, agent) pairs again, to show how much one run varies.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENTS = ['opencode', 'codex', 'claude', 'qwen']
NAMES = {'opencode': 'OpenCode', 'codex': 'Codex', 'claude': 'Claude Code', 'qwen': 'Qwen Code'}


def load(run_id):
    root = HERE / 'results' / run_id
    rows = [json.loads(l) for l in (root / 'results.jsonl').read_text().splitlines()]
    for r in rows:
        proxy = root / r['task'] / r['agent'] / 'proxy.jsonl'
        entries = [json.loads(l) for l in proxy.read_text().splitlines()] if proxy.exists() else []
        calls = {c['call_id'] for e in entries for c in (e.get('removed') or {}).get('repaired_tool_arguments', [])}
        r['repaired_calls'] = len(calls)
        r['strict'] = bool(r['passed'] and not r['timed_out'])
    return rows


def per_agent(rows):
    out = {}
    for agent in AGENTS:
        mine = [r for r in rows if r['agent'] == agent]
        if not mine:
            continue
        wall = sum(r['wall_s'] for r in mine)
        strict = sum(r['strict'] for r in mine)
        lenient = sum(bool(r['passed']) for r in mine)
        out[agent] = {
            'tasks': len(mine), 'strict': strict, 'lenient': lenient, 'wall_min': round(wall / 60, 1),
            'min_per_strict_success': round(wall / 60 / strict, 1) if strict else None,
            'min_per_lenient_success': round(wall / 60 / lenient, 1) if lenient else None,
            'requests': sum(r['requests_total'] for r in mine),
            'output_tokens': sum(r['completion_tokens_total'] for r in mine),
            'prompt_tokens': sum(r['prompt_tokens_total'] for r in mine),
            'cached_tokens': sum(r['cached_tokens_total'] for r in mine),
            'prefill_min': round(sum(r['prefill_seconds_total'] for r in mine) / 60, 1),
            'decode_min': round(sum(r['decode_seconds_total'] for r in mine) / 60, 1),
            'repaired_calls': sum(r['repaired_calls'] for r in mine),
            'timeouts': sum(bool(r['timed_out']) for r in mine),
            'max_gpu_gib': max((r.get('max_gpu_gib') or 0) for r in mine),
            'min_mem_available_gib': min((r.get('min_mem_available_gib') or 999) for r in mine),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_id')
    ap.add_argument('--repeat', action='append', default=[])
    a = ap.parse_args()
    rows = load(a.run_id)
    agents = per_agent(rows)
    tasks = sorted({r['task'] for r in rows})
    grid = {t: {r['agent']: r for r in rows if r['task'] == t} for t in tasks}

    print(f'# {a.run_id}: {len(rows)} runs\n')
    print('| agent | strict / lenient success | wall min | min per success (strict / lenient) | requests | '
          'output tokens | prefill / decode min | repaired calls | timeouts |')
    print('|---|---|---|---|---|---|---|---|---|')
    for agent, s in agents.items():
        print(f"| {NAMES[agent]} | {s['strict']} / {s['lenient']} of {s['tasks']} | {s['wall_min']} | "
              f"{s['min_per_strict_success']} / {s['min_per_lenient_success']} | {s['requests']} | {s['output_tokens']:,} | "
              f"{s['prefill_min']} / {s['decode_min']} | {s['repaired_calls']} | {s['timeouts']} |")
    print('\n| task | ' + ' | '.join(NAMES[x] for x in agents) + ' |')
    print('|---|' + '---|' * len(agents))
    for t in tasks:
        cells = []
        for agent in agents:
            r = grid[t].get(agent)
            if r is None:
                cells.append('')
                continue
            mark = 'pass' if r['strict'] else ('pass, timed out' if r['passed'] else 'FAIL')
            cells.append(f"{mark} {r['wall_s']:.0f} s")
        print(f'| {t} | ' + ' | '.join(cells) + ' |')

    if a.repeat:
        print('\n## The same (task, agent) in other runs: wall seconds\n')
        others = {rid: {(r['task'], r['agent']): r for r in load(rid)} for rid in a.repeat}
        print('| task | agent | ' + ' | '.join([a.run_id, *a.repeat]) + ' | max / min |')
        print('|---|---|' + '---|' * (1 + len(a.repeat)) + '---|')
        for r in rows:
            same = [o.get((r['task'], r['agent'])) for o in others.values()]
            if not any(same):
                continue
            walls = [r['wall_s']] + [s['wall_s'] if s else None for s in same]
            known = [w for w in walls if w]
            print(f"| {r['task']} | {NAMES[r['agent']]} | " + ' | '.join('' if w is None else f'{w:.0f}' for w in walls)
                  + f' | {max(known) / min(known):.2f} |')

    out = HERE / 'results' / a.run_id / 'summary.json'
    out.write_text(json.dumps({'agents': agents, 'runs': rows}, indent=1))


if __name__ == '__main__':
    main()
