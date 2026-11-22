"""PLAN_22 M1 and M3 (no model is loaded; a GPU and Docker with the NVIDIA Container Toolkit are needed).

  m1  tuned read bandwidth: bw_tuned.cu compiled with the image's nvcc (sm_121); sweeps at 2 GB and 8 GB (288 settings,
      10 repetitions); candidates = the 8 best at 8 GB plus the 4 best at 2 GB; each runs the forward pattern (the 401
      matmul weight groups of sizes.txt, one launch each) in separate buffers and in one arena, 10 repetitions; the best
      candidate (median) is confirmed in a fresh process with 30 repetitions -> F. The best 8 GB setting is confirmed at
      8 GB and 17.6 GB with 30 repetitions -> B*. All 850 groups with F's setting, 10 repetitions (report only).
  m3  cuBLAS (torch._scaled_mm, NVFP4 and FP8) running the same 401 matmul shapes at 16-128 rows (report only).

    python3 run_ceiling.py m1|m3 <out_dir>        (IMAGE defaults to Part 4's qwen38-tensorfold-repro:0.6.0-trace)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGE = os.environ.get('IMAGE', 'qwen38-tensorfold-repro:0.6.0-trace')
NAME = 'ceiling-bench'


def docker(out, cmd, timeout=3600):
    args = ['docker', 'run', '--rm', '--name', NAME, '--gpus', 'all', '--network', 'none', '--cpuset-cpus', '5-9,15-19',
            '-u', f'{os.getuid()}:{os.getgid()}', '-e', 'PYTHONDONTWRITEBYTECODE=1', '-v', f'{HERE}:/src:ro', '-v', f'{out}:/out',
            '-w', '/out', '--entrypoint', 'bash', IMAGE, '-c', cmd]
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f'rc={r.returncode}: {r.stderr[-2000:]}')
    return r.stdout, r.stderr


def jl(text):
    return [json.loads(l) for l in text.splitlines() if l.startswith('{')]


def m1(out):
    out.mkdir(parents=True, exist_ok=False)
    docker(out, 'nvcc --version > nvcc.txt && nvcc -O3 -std=c++17 -gencode arch=compute_121,code=sm_121 -o bw_tuned /src/bw_tuned.cu')
    sweeps = {}
    for gb in ('2', '8'):
        so, se = docker(out, f'./bw_tuned sweep {gb} 10')
        (out / f'sweep{gb}.jsonl').write_text(so)
        (out / 'device.json').write_text(se.splitlines()[0] + '\n')
        sweeps[gb] = sorted(jl(so), key=lambda r: -r['gb_s_median'])
    cands = []
    for r in sweeps['8'][:8] + sweeps['2'][:4]:
        if r['cfg'] not in cands:
            cands.append(r['cfg'])
    rows = []
    for c in cands:
        for lay in ('sep', 'arena'):
            so, _ = docker(out, f'./bw_tuned forward /src/sizes.txt 1048576 10 {c} {lay}')
            rows += [{**x, 'layout': lay} for x in jl(so)]
    (out / 'forward_candidates.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
    best = min(rows, key=lambda r: r['ms_median'])
    so, _ = docker(out, f'./bw_tuned forward /src/sizes.txt 1048576 30 {best["cfg"]} {best["layout"]}')
    conf = jl(so)[0]
    so, _ = docker(out, f'./bw_tuned forward /src/sizes.txt 0 10 {best["cfg"]} {best["layout"]}')
    all850 = jl(so)[0]
    b8 = sweeps['8'][0]['cfg']
    singles = []
    for gb in ('8', '17.6'):
        so, _ = docker(out, f'./bw_tuned single {gb} 30 {b8}')
        singles += jl(so)
    res = {'candidates': cands, 'best_candidate': {k: best[k] for k in ('cfg', 'layout', 'ms_median', 'gb_s_median')},
           'F_confirm': conf, 'F_ms': conf['ms_median'], 'forward_all_850': all850,
           'B_cfg': b8, 'B_confirm': singles, 'B_star_gb_s': max(s['gb_s_median'] for s in singles if s['size_gb'] >= 8)}
    (out / 'm1.json').write_text(json.dumps(res, indent=1))
    print('M1 ' + json.dumps({'F_ms': res['F_ms'], 'F_gb_s': conf['gb_s_median'], 'cfg': best['cfg'], 'layout': best['layout'],
                              'B_star_gb_s': res['B_star_gb_s'], 'all850_ms': all850['ms_median']}), flush=True)


def m3(out):
    out.mkdir(parents=True, exist_ok=True)
    so, se = docker(out, 'python3 -I /src/m3_mm.py /src/tensors.json')
    (out / 'm3.json').write_text(so)
    print('M3 ' + json.dumps({m: v.get('ms_median', v.get('error', '')[:80]) for m, v in json.loads(so)['by_m'].items()}), flush=True)


if __name__ == '__main__':
    {'m1': m1, 'm3': m3}[sys.argv[1]](Path(sys.argv[2]).resolve())
