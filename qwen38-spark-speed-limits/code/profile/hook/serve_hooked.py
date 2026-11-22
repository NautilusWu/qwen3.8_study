"""PLAN_14 step 1: `tensorfold serve` with an observation-only timing hook (run inside the 0.6.0 image).

    python serve_hooked.py serve <the usual serve arguments...>

P14_HOOK=1 installs the hook in this process before the server starts (nothing on disk changes):
- NVTX ranges around the stages of a decode round (draft, copy, tree, verify, sample, commit; multi-stream: round),
  so Nsight Systems can attribute every kernel to a stage and a round;
- P14_NVTX_DETAIL=1 also wraps every projection matmul (`forward._mm`: weight type, bytes read, rows), the DeltaNet
  tree kernel and tree attention (only for the Nsight runs; the labels cost a little host time);
- one stream: `draft_decode` runs with its own `trace=` (rows, accepted tokens and the four stage times of every round,
  which the engine measures itself); several streams: each round's host time per stage, rows and accepted tokens.
Records go to P14_ROUNDS (JSON lines). Nothing here changes a kernel, a number, a draft or a sample; the runner
checks that the replies' token ids are identical with and without the hook and under Nsight Systems.
"""
import json
import os
import sys
import threading
import time

HOOK = os.environ.get('P14_HOOK') == '1'
DETAIL = os.environ.get('P14_NVTX_DETAIL') == '1'
OUT = os.environ.get('P14_ROUNDS', '/out/rounds.jsonl')


def install():
    import torch
    from torch.cuda import nvtx

    from tensorfold.cuda import sampling as sampling_mod
    from tensorfold.cuda import streams as streams_mod
    from tensorfold.cuda.kernels import attention as attn_mod
    from tensorfold.cuda.kernels import gdn as gdn_mod
    from tensorfold.families.qwen3_5.cuda import decode, dflash2, forward, multi

    lock = threading.Lock()
    sink = open(OUT, 'a', buffering=1)
    local = threading.local()

    def emit(rec):
        with lock:
            sink.write(json.dumps(rec) + '\n')

    def ranged(obj, name, label, timed=None):
        """Wrap obj.name in an NVTX range; with ``timed``, add its host time to the current multi-stream round."""
        orig = getattr(obj, name)
        assert callable(orig), f'{obj}.{name}'

        def wrapper(*a, **k):
            nvtx.range_push(label)
            t0 = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                nvtx.range_pop()
                cur = getattr(local, 'round', None)
                if timed and cur is not None:
                    cur['ms'][timed] = cur['ms'].get(timed, 0.0) + 1e3 * (time.perf_counter() - t0)
        wrapper.__wrapped__ = orig
        setattr(obj, name, wrapper)

    # --- stages (both paths) ------------------------------------------------------------------------------------
    ranged(decode.CopyIndex, 'propose', 'copy', 'copy')
    ranged(dflash2.DFlash2, 'propose_tree', 'draft')
    ranged(dflash2.DFlash2, 'launch_blocks', 'draft', 'draft_launch')
    ranged(dflash2.DFlash2, 'finish_tree', 'tree', 'tree')
    ranged(dflash2.DFlash2, 'add_taps', 'commit')
    ranged(dflash2.DFlash2, 'add_taps_streams', 'commit', 'draft_taps')
    ranged(decode, 'tree_forward', 'verify')
    ranged(decode, 'sample_rows', 'sample')
    ranged(decode, 'commit', 'commit')
    ranged(multi, 'multi_tree_forward', 'verify', 'verify_launch')
    ranged(multi, 'sample_streams', 'sample', 'sample')
    ranged(multi.MultiDecoder, '_commit', 'commit', 'commit')
    ranged(multi.MultiDecoder, '_mode', 'copy', 'copy')

    # --- one stream: the engine's own per-round trace -----------------------------------------------------------
    orig_dd = decode.draft_decode

    def draft_decode(*a, **k):
        trace = k.get('trace')
        if trace is None:
            trace = []
            k['trace'] = trace
        res = orig_dd(*a, **k)
        emit({'kind': 'single', 'tokens': len(res.tokens), 'seconds': res.seconds, 'rounds': res.rounds,
              'drafted_rows': res.drafted_rows, 'accepted_drafts': res.accepted_drafts,
              'stage_s': {'draft': res.draft_seconds, 'verify': res.verify_seconds, 'sample': res.sample_seconds,
                          'commit': res.commit_seconds}, 'trace': trace})
        return res
    decode.draft_decode = draft_decode

    # --- several streams: one record per round -----------------------------------------------------------------
    orig_round = multi.MultiDecoder.round
    orig_fill = multi.MultiDecoder._fill
    orig_accept = multi.accept

    def fill(self, *a, **k):
        cur = getattr(local, 'round', None)
        if cur is not None:
            cur['fill'] = True
        return orig_fill(self, *a, **k)

    def accept(tokens, parents, rows, *a, **k):
        path, end = orig_accept(tokens, parents, rows, *a, **k)
        cur = getattr(local, 'round', None)
        if cur is not None:
            cur['streams'].append([len(tokens), len(path) - 1])
        return path, end

    def round_(self):
        local.round = {'kind': 'multi', 'ms': {}, 'streams': [], 'fill': False}
        nvtx.range_push('round')
        t0 = time.perf_counter()
        try:
            return orig_round(self)
        finally:
            nvtx.range_pop()
            cur, local.round = local.round, None
            if cur['streams']:
                cur['wall_ms'] = 1e3 * (time.perf_counter() - t0)
                emit(cur)
    multi.MultiDecoder.round = round_
    multi.MultiDecoder._fill = fill
    multi.accept = accept

    # --- detail for Nsight: every projection's bytes and rows, the DeltaNet tree kernel, tree attention ------------
    if DETAIL:
        orig_mm = forward._mm
        labels = {}

        def nbytes(w):
            try:
                return int(w.nbytes())
            except Exception:  # noqa: BLE001 - a weight without nbytes(): its tensors
                return sum(t.numel() * t.element_size() for t in vars(w).values() if isinstance(t, torch.Tensor))

        def mm(x, w, xs=None):
            key = (id(w), int(x.shape[0]))
            lab = labels.get(key)
            if lab is None:
                lab = labels[key] = f'mm|{type(w).__name__}|{nbytes(w)}|{key[1]}|{getattr(w, "n", 0)}x{getattr(w, "k", 0)}'
            nvtx.range_push(lab)
            try:
                return orig_mm(x, w, xs)
            finally:
                nvtx.range_pop()
        forward._mm = mm
        ranged(gdn_mod, 'tree', 'gdn')
        ranged(attn_mod, 'attention', 'attn')
    print(f'[p14] hook installed (detail: {DETAIL}), records to {OUT}', flush=True)


if __name__ == '__main__':
    if HOOK:
        install()
    from tensorfold.cli import main
    sys.exit(main(sys.argv[1:]))
