"""PLAN_22 M3 (report only): cuBLAS (torch._scaled_mm) running the 401 weight matmuls of one decode forward at M rows.

Shapes from tensors.json: MLP and head NVFP4 (packed uint8 [N, K/2], block scales fp8 [N, K/16]), DeltaNet and
attention FP8 [N, K]. Random weights and activations (only the time matters; the values and the scale layout are not
checked). For M in 16, 32, 48, 64, 96, 128: the whole sequence of 401 matmuls, 2 warm-ups, 10 timed repetitions,
CUDA events. Prints one JSON object. A failure of the NVFP4 path is recorded, not worked around.
"""
import json
import statistics
import sys

import torch

MS = (16, 32, 48, 64, 96, 128)


def main():
    t = json.load(open(sys.argv[1]))
    dev = torch.device('cuda')
    groups = [g for g in t['rows'] if g['bytes'] >= 1 << 20]
    one = torch.ones((), device=dev)
    mats = []
    for g in groups:
        w = [x for x in g['tensors'] if x[0] == '.weight'][0]
        if w[1] == 'U8':
            n, k = w[2][0], w[2][1] * 2
            wq = torch.randint(0, 255, (n, k // 2), dtype=torch.uint8, device=dev).view(torch.float4_e2m1fn_x2)
            sw = torch.rand(n, k // 16, device=dev).to(torch.float8_e4m3fn)
            mats.append(('fp4', n, k, wq, sw))
        else:
            n, k = w[2]
            mats.append(('fp8', n, k, torch.randn(n, k, device=dev).to(torch.float8_e4m3fn), one))
    res = {'groups': len(mats), 'weight_gb': round(sum(g['bytes'] for g in groups) / 1e9, 4), 'by_m': {}}
    for m in MS:
        acts = {}
        for kind, n, k, _, _ in mats:
            if (kind, k) in acts:
                continue
            if kind == 'fp4':
                a = torch.randint(0, 255, (m, k // 2), dtype=torch.uint8, device=dev).view(torch.float4_e2m1fn_x2)
                sa = torch.rand(-(-m // 128) * 128, k // 16, device=dev).to(torch.float8_e4m3fn)
            else:
                a, sa = torch.randn(m, k, device=dev).to(torch.float8_e4m3fn), one
            acts[(kind, k)] = (a, sa)

        def seq():
            for kind, n, k, w, sw in mats:
                a, sa = acts[(kind, k)]
                torch._scaled_mm(a, w.t(), scale_a=sa, scale_b=sw, out_dtype=torch.bfloat16)
        try:
            for _ in range(2):
                seq()
            torch.cuda.synchronize()
            times = []
            for _ in range(10):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                seq()
                e.record()
                e.synchronize()
                times.append(s.elapsed_time(e))
            med = statistics.median(times)
            res['by_m'][m] = {'ms': [round(x, 3) for x in times], 'ms_median': round(med, 3),
                              'gb_s': round(res['weight_gb'] / med * 1e3, 1)}
        except Exception as exc:  # noqa: BLE001
            res['by_m'][m] = {'error': repr(exc)[:400]}
        print(json.dumps({m: res['by_m'][m].get('ms_median', res['by_m'][m].get('error'))}), file=sys.stderr, flush=True)
    res['device'], res['torch'] = torch.cuda.get_device_name(), torch.__version__
    print(json.dumps(res))


if __name__ == '__main__':
    main()
