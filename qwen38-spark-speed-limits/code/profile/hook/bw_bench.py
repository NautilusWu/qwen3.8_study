"""PLAN_14 R0 (inside the 0.6.0 image, no model loaded): how fast the GB10's GPU reads unified memory in practice.

Five plain methods, sizes 2, 8 and 17.6 GB (17.6 GB = the weights one forward reads), 10 timed repetitions each
after one warm-up, CUDA events. Reads (bytes = the buffer): a float32 sum, a bf16 sum, a bf16 sum per 8,192-wide row,
and a bf16 matrix [rows, 8192] times one vector (torch.mv, the decode pattern). A float32 copy (bytes moved = 2 x the
buffer) is reported only. Prints one JSON object; B (the plan's achievable read bandwidth) is the highest median GB/s
of the four reads at 8 GB or more.
"""
import json
import statistics
import sys

import torch

import os
QUICK = os.environ.get('P14_BW_QUICK') == '1'          # the compatibility check: 2 GB, 3 repetitions
SIZES_GB = (2.0,) if QUICK else (2.0, 8.0, 17.6)
REPS = 3 if QUICK else 10


def timed(fn):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(REPS):
        start.record()
        fn()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end) / 1e3)
    return out


def main():
    dev = torch.device('cuda')
    rows = []
    for gb in SIZES_GB:
        n_bytes = int(gb * 1e9)
        for name, dtype, factor in (('sum_fp32', torch.float32, 1), ('sum_bf16', torch.bfloat16, 1), ('rowsum_bf16', torch.bfloat16, 1),
                                    ('gemv_bf16', torch.bfloat16, 1), ('copy_fp32', torch.float32, 2)):
            n = n_bytes // torch.tensor([], dtype=dtype).element_size() // 8192 * 8192
            x = torch.ones(n, dtype=dtype, device=dev)
            if name.startswith('copy'):
                y = torch.empty_like(x)
                fn = lambda: y.copy_(x)
            elif name.startswith('rowsum'):                     # [rows, 8192]: one output per row
                fn = lambda: x.view(-1, 8192).sum(dim=1)
            elif name.startswith('gemv'):                       # the decode pattern: a weight matrix times one vector
                v = torch.ones(8192, dtype=dtype, device=dev)
                fn = lambda: torch.mv(x.view(-1, 8192), v)
            else:
                fn = lambda: x.sum(dtype=torch.float32)
            secs = timed(fn)
            moved = factor * x.numel() * x.element_size()
            rows.append({'method': name, 'size_gb': gb, 'bytes_moved': moved, 'seconds': secs,
                         'gb_s_median': round(moved / statistics.median(secs) / 1e9, 1),
                         'gb_s_best': round(moved / min(secs) / 1e9, 1)})
            print(json.dumps({k: rows[-1][k] for k in ('method', 'size_gb', 'gb_s_median', 'gb_s_best')}), file=sys.stderr, flush=True)
            del x
            if name.startswith('copy'):
                del y
            torch.cuda.empty_cache()
    reads = [r for r in rows if not r['method'].startswith('copy') and (r['size_gb'] >= 8 or QUICK)]
    best = max(reads, key=lambda r: r['gb_s_median'])
    print(json.dumps({'rows': rows, 'B_gb_s': best['gb_s_median'], 'B_from': [best['method'], best['size_gb']],
                      'device': torch.cuda.get_device_name(), 'torch': torch.__version__}))


if __name__ == '__main__':
    main()
