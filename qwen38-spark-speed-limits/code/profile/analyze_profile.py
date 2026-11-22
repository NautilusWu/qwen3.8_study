"""PLAN_14 step 1: from the runs' files to the plan's numbers and its decision.

    python3 analyze_profile.py <prefix> [--compat]

- B: the achievable read bandwidth (bw.json).
- T: time per decode round, a mean (rounds are additive): one stream = the engine's decode seconds / rounds of the
  long thinking reply (R1, the hook only; code and chat reported too); 4 streams = the hook's round wall time over
  rounds with 4 streams and no prompt fill (R3).
- From the Nsight runs (R2, R4), per round (means over the same kinds of rounds): GPU busy time (the union of the
  round's kernels), GPU idle H_g = round time - busy, and H_k = the sum over the round's projection matmuls of
  max(0, GPU time - weight bytes / (0.9 B)). Kernels are attributed to NVTX ranges through their launch calls.
- Nsight's own cost: T_nsys (the hook's round time in the Nsight run) - T; H_g is reduced by it (the extra time
  shows up as idle GPU), then S = T / (T - H_k - H_g).
- Decision (written before the run): S < 1.10 for both paths -> stop this direction; otherwise step 2.
"""
import argparse
import bisect
import json
import re
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ('copy', 'draft', 'tree', 'verify', 'sample', 'commit')
ATTAINABLE = 0.9
THRESHOLD = 1.10


def jl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()] if Path(path).exists() else []


def phased(records):
    """The hook's records with the runner's phase markers applied (records before the first marker: the warm-up)."""
    phase, rep, out = None, None, []
    for r in records:
        if r.get('kind') == 'marker':
            phase, rep = r['phase'], r['rep']
        else:
            out.append({**r, 'phase': phase, 'rep': rep})
    return out


def single_T(records, phases):
    rs = [r for r in records if r['kind'] == 'single' and r['phase'] in phases]
    rounds = sum(r['rounds'] for r in rs)
    if not rounds:
        return None
    tr = [t for r in rs for t in r['trace']]
    return {'requests': len(rs), 'rounds': rounds, 'tokens': sum(r['tokens'] - 1 for r in rs),
            'ms_per_round': round(1e3 * sum(r['seconds'] for r in rs) / rounds, 3),
            'tokens_per_round': round(sum(r['tokens'] - 1 for r in rs) / rounds, 3),
            'stage_ms_per_round': {k: round(1e3 * sum(r['stage_s'][k] for r in rs) / rounds, 3) for k in ('draft', 'verify', 'sample', 'commit')},
            'rows_per_round': round(statistics.mean(t['rows'] for t in tr), 2),
            'accepted_per_round': round(statistics.mean(t['accepted'] for t in tr), 3),
            'copy_rounds': sum(t['source'] == 'copy' for t in tr)}


def multi_rounds(records, phases=None, streams=4):
    return [r for r in records if r['kind'] == 'multi' and not r['fill'] and len(r['streams']) == streams
            and (phases is None or r['phase'] in phases)]


def multi_T(records, phases=None):
    rs = multi_rounds(records, phases)
    if not rs:
        return None
    ms = defaultdict(float)
    for r in rs:
        for k, v in r['ms'].items():
            ms[k] += v
    return {'rounds': len(rs), 'ms_per_round': round(statistics.mean(r['wall_ms'] for r in rs), 3),
            'host_ms_per_round': {k: round(v / len(rs), 3) for k, v in sorted(ms.items())},
            'rows_per_stream': round(statistics.mean(s[0] for r in rs for s in r['streams']), 2),
            'accepted_per_stream': round(statistics.mean(s[1] for r in rs for s in r['streams']), 3),
            'tokens_per_round': round(statistics.mean(sum(s[1] + 1 for s in r['streams']) for r in rs), 3)}


# --- Nsight Systems ----------------------------------------------------------------------------------------------

def load_nsys(path):
    db = sqlite3.connect(path)
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    strings = dict(db.execute('select id, value from StringIds')) if 'StringIds' in tables else {}
    cols = {r[1] for r in db.execute('pragma table_info(NVTX_EVENTS)')}
    text_col = 'coalesce(text, (select value from StringIds where id = textId))' if 'textId' in cols else 'text'
    ranges = [(s, e, tid, t) for s, e, tid, t in
              db.execute(f'select start, end, globalTid, {text_col} from NVTX_EVENTS where end is not null and end > start')]
    launches = dict((c, (s, tid)) for c, s, tid in db.execute('select correlationId, start, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME'))
    kcols = {r[1] for r in db.execute('pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)')}
    name_col = 'shortName' if 'shortName' in kcols else 'demangledName'
    kernels = [(s, e, c, strings.get(n, str(n))) for s, e, c, n in
               db.execute(f'select start, end, correlationId, {name_col} from CUPTI_ACTIVITY_KIND_KERNEL')]
    return ranges, launches, kernels, tables


def attribute(ranges, launches, kernels):
    """Each kernel: the stack of NVTX ranges open on its launching thread when it was launched."""
    by_tid = defaultdict(list)
    for i, (s, e, tid, t) in enumerate(ranges):
        by_tid[tid] += [(s, 1, i), (e, -1, i)]
    klaunch = defaultdict(list)
    for k, (s, e, c, n) in enumerate(kernels):
        if c in launches:
            ts, tid = launches[c]
            klaunch[tid].append((ts, 0, k))
    stacks = [None] * len(kernels)
    for tid in set(by_tid) | set(klaunch):
        ev = sorted(by_tid.get(tid, []) + klaunch.get(tid, []), key=lambda x: (x[0], {-1: 0, 1: 1, 0: 2}[x[1]]))
        stack = []
        for ts, kind, idx in ev:
            if kind == 1:
                stack.append(idx)
            elif kind == -1:
                if idx in stack:
                    stack.remove(idx)
            else:
                stacks[idx] = tuple(stack)
    return stacks


def nsys_rounds(path, kind, hook_records, B):
    ranges, launches, kernels, tables = load_nsys(path)
    stacks = attribute(ranges, launches, kernels)
    label = lambda i: ranges[i][3] or ''
    # rounds: 4 streams = the hook's 'round' ranges that launched a verify; one stream = stage ranges grouped per round
    verify_ids = [i for i, (s, e, tid, t) in enumerate(ranges) if t == 'verify']
    if kind == 'multi':
        rnds = sorted((i for i, r in enumerate(ranges) if r[3] == 'round'), key=lambda i: ranges[i][0])
        has_verify = set()
        vstarts = sorted(ranges[i][0] for i in verify_ids)
        spans = []
        for i in rnds:
            s, e = ranges[i][0], ranges[i][1]
            j = bisect.bisect_left(vstarts, s)
            if j < len(vstarts) and vstarts[j] < e:
                spans.append((s, e))
        recs = [r for r in hook_records if r['kind'] == 'multi' and r['phase'] is not None]
    else:
        groups = []
        for tid in {ranges[i][2] for i in verify_ids}:     # each request decodes on its own server thread
            stage = sorted((r for r in ranges if r[2] == tid and r[3] in STAGES), key=lambda r: r[0])
            top, last_end = [], -1
            for r in stage:                               # top-level stage ranges (nested draft > tree are skipped)
                if r[0] >= last_end:
                    top.append(r)
                    last_end = r[1]
            cur, seen_verify = [], False                  # a round starts at its copy/draft; ranges outside rounds
            for r in top:                                 # (a prompt's first sample) are left out
                if r[3] in ('copy', 'draft') and (not cur or seen_verify):
                    if cur:
                        groups.append(cur)
                    cur, seen_verify = [r], False
                elif cur and not (seen_verify and r[3] not in ('sample', 'commit')):
                    if seen_verify and r[3] == 'sample' and any(x[3] == 'commit' for x in cur):
                        continue                          # a sample after this round's commit: the next prompt's
                    cur.append(r)
                    seen_verify |= r[3] == 'verify'
            if cur:
                groups.append(cur)
        groups.sort(key=lambda g: g[0][0])                # requests run one after another: time order is round order
        end_of = lambda g: max(x[1] for x in g if x[3] in ('commit', 'sample', 'verify'))
        recs = [r for r in hook_records if r['kind'] == 'single' and r['phase'] is not None]
        counts = [r['rounds'] for r in recs]
        spans, k = [], 0
        for n in counts:                                   # within a request a round lasts until the next one starts
            g = groups[k:k + n]
            for a, b in zip(g, g[1:] + [None]):
                spans.append((a[0][0], b[0][0] if b is not None else end_of(a)))
            k += n
        if k != len(groups):
            return {'error': f'{len(groups)} rounds in the Nsight trace, {k} in the hook records', 'tables': sorted(tables)}
    expect = len(recs) if kind == 'multi' else sum(r['rounds'] for r in recs)
    if len(spans) != expect:
        return {'error': f'{len(spans)} rounds in the Nsight trace, {expect} in the hook records'}
    # per round: kernels launched in it, their union, the matmuls' time against their bytes
    starts = [s for s, _ in spans]
    per = [{'busy': 0.0, 'gpu': defaultdict(float), 'mm': defaultdict(lambda: [0.0, 0])} for _ in spans]
    order = sorted(range(len(kernels)), key=lambda k: launches.get(kernels[k][2], (0,))[0])
    intervals = defaultdict(list)
    for k in order:
        c = kernels[k][2]
        if c not in launches or stacks[k] is None:
            continue
        ts = launches[c][0]
        j = bisect.bisect_right(starts, ts) - 1
        if j < 0 or ts >= spans[j][1]:
            continue
        s, e = max(kernels[k][0], spans[j][0]), min(kernels[k][1], spans[j][1])
        if e <= s:
            continue
        intervals[j].append((s, e))
        labs = [label(i) for i in stacks[k]]
        st = next((l for l in labs if l in STAGES), 'other')
        mm = next((i for i in reversed(stacks[k]) if label(i).startswith('mm|')), None)
        detail = 'mm' if mm is not None else next((l for l in reversed(labs) if l in ('gdn', 'attn')), '')
        per[j]['gpu'][f'{st}:{detail}' if detail else st] += (e - s) / 1e6
        if mm is not None:
            per[j]['mm'][mm][0] += (kernels[k][1] - kernels[k][0]) / 1e6
    for j, iv in intervals.items():
        iv.sort()
        busy, cs, ce = 0, None, None
        for s, e in iv:
            if cs is None or s > ce:
                if cs is not None:
                    busy += ce - cs
                cs, ce = s, e
            else:
                ce = max(ce, e)
        if cs is not None:
            busy += ce - cs
        per[j]['busy'] = busy / 1e6
    out = []
    for j, (s, e) in enumerate(spans):
        hk, by_type = 0.0, defaultdict(lambda: [0.0, 0, 0])
        for i, (t, _) in per[j]['mm'].items():
            _, typ, nbytes, rows, shape = label(i).split('|')
            ideal = int(nbytes) / (ATTAINABLE * B * 1e9) * 1e3
            hk += max(0.0, t - ideal)
            by_type[typ][0] += t
            by_type[typ][1] += int(nbytes)
            by_type[typ][2] += 1
        out.append({'wall_ms': (e - s) / 1e6, 'busy_ms': per[j]['busy'], 'idle_ms': (e - s) / 1e6 - per[j]['busy'],
                    'hk_ms': hk, 'gpu_ms': dict(per[j]['gpu']), 'mm': {k: list(v) for k, v in by_type.items()}})
    return {'rounds': out, 'records': recs}


def summarize(rounds):
    n = len(rounds)
    if not n:
        return None
    mean = lambda key: round(sum(r[key] for r in rounds) / n, 3)
    gpu = defaultdict(float)
    mm = defaultdict(lambda: [0.0, 0, 0])
    for r in rounds:
        for k, v in r['gpu_ms'].items():
            gpu[k] += v
        for k, (t, b, c) in r['mm'].items():
            mm[k][0] += t
            mm[k][1] += b
            mm[k][2] += c
    return {'rounds': n, 'wall_ms': mean('wall_ms'), 'busy_ms': mean('busy_ms'), 'idle_ms': mean('idle_ms'), 'hk_ms': mean('hk_ms'),
            'gpu_ms_per_round': {k: round(v / n, 3) for k, v in sorted(gpu.items(), key=lambda x: -x[1])},
            'matmul_per_round': {k: {'ms': round(t / n, 3), 'gb': round(b / n / 1e9, 3), 'calls': round(c / n, 1),
                                     'gb_s': round(b / 1e9 / (t / 1e3), 1) if t else None} for k, (t, b, c) in mm.items()}}


def curve(log):
    m = re.search(r'verify ms by rows[^:]*: ([^\n]+)', Path(log).read_text()) if Path(log).exists() else None
    return m.group(1).strip() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('prefix')
    ap.add_argument('--compat', action='store_true')
    a = ap.parse_args()
    d = HERE / 'results' / a.prefix
    bwj = json.loads((d / 'bw.json').read_text())
    B = bwj['B_gb_s']
    names = {'single': ('C2', 'C3', ['code', 'chat']), 'multi': ('C5', 'C5', None)} if a.compat else \
            {'single': ('R1', 'R2', ['long']), 'multi': ('R3', 'R4', None)}
    wts = json.loads((HERE / 'weights.json').read_text())
    res = {'prefix': a.prefix, 'B_gb_s': B, 'B_from': bwj['B_from'], 'bw_rows': [{k: r[k] for k in ('method', 'size_gb', 'gb_s_median', 'gb_s_best')} for r in bwj['rows']],
           'nominal_share': round(B / 273, 3), 'weights': wts, 'paths': {}}
    for path, (hook_run, ns_run, phases) in names.items():
        hook = phased(jl(d / f'{hook_run}.rounds.jsonl'))
        ns_hook = phased(jl(d / f'{ns_run}.rounds.jsonl'))
        p = {'hook_run': hook_run, 'nsys_run': ns_run, 'verify_curve': curve(d / f'{ns_run}.server.log')}
        if path == 'single':
            p['T_hook'] = single_T(hook, phases)
            p['T_nsys_run'] = single_T(ns_hook, phases)
            p['other_items'] = {ph: single_T(hook, [ph]) for ph in ('long', 'code', 'chat') if ph not in phases}
        else:
            p['T_hook'] = multi_T(hook)
            p['T_nsys_run'] = multi_T(ns_hook)
        sq = d / f'{ns_run}.sqlite'
        if sq.exists() and p['T_hook']:
            nr = nsys_rounds(sq, path, ns_hook, B)
            if 'error' in nr:
                p['nsys'] = nr
            else:
                recs, rounds = nr['records'], nr['rounds']
                if path == 'single':
                    flat = [r['phase'] for r in recs for _ in range(r['rounds'])]
                    sel = [x for x, ph in zip(rounds, flat) if ph in phases]
                else:
                    sel = [x for x, r in zip(rounds, recs) if not r['fill'] and len(r['streams']) == 4]
                p['nsys'] = summarize(sel)
                T, Tn = p['T_hook']['ms_per_round'], p['T_nsys_run']['ms_per_round']
                infl = max(0.0, Tn - T)
                hg = max(0.0, p['nsys']['idle_ms'] - infl)
                hk = p['nsys']['hk_ms']
                p['estimate'] = {'T_ms': T, 'T_nsys_ms': Tn, 'nsys_inflation': round(Tn / T, 3), 'H_k_ms': hk,
                                 'H_g_ms_raw': p['nsys']['idle_ms'], 'H_g_ms': round(hg, 3),
                                 'S': round(T / (T - hk - hg), 3) if T - hk - hg > 0 else None,
                                 'proportions_only': Tn / T > 1.10}
        res['paths'][path] = p
    S = [p.get('estimate', {}).get('S') for p in res['paths'].values()]
    if all(s is not None for s in S):
        res['decision'] = ('stop this direction: S < 1.10 for both paths' if all(s < THRESHOLD for s in S)
                           else 'step 2: S >= 1.10 for at least one path')
    else:
        res['decision'] = 'not decided: an estimate is missing'
    (d / 'p14-analysis.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
