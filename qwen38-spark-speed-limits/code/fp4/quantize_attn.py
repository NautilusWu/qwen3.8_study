"""PLAN_15 (A): the NVIDIA Qwen3.8-27B NVFP4 checkpoint with its 208 FP8 projections (linear-attention in_proj_qkv,
in_proj_z, out_proj; full-attention q/k/v/o) re-quantized to NVFP4 from the original BF16 weights. Everything else
(MLP, lm_head, norms, gates, embeddings, vision, MTP) is copied tensor for tensor.

Runs inside the TensorFold 0.6.0 image (torch + safetensors), GPU for the arithmetic:
    python -I /p15/quantize_attn.py --nvfp4 <NVIDIA dir> --bf16 <Qwen BF16 dir> --out <new dir> [--method mse|rtn]
    python -I /p15/quantize_attn.py --selftest --nvfp4 <NVIDIA dir> --bf16 <Qwen BF16 dir>

NVFP4 as ModelOpt writes it: weight uint8 [N, K/2] (e2m1 codes, low nibble first), weight_scale e4m3 [N, K/16]
(one scale per 16 inputs), weight_scale_2 fp32 scalar g with value = code * e4m3(scale) * g, g = amax|W| / (6 * 448).
Block scales: 'rtn' takes amax_block / 6 (ModelOpt's default); 'mse' (the plan's method) tries c * amax_block / 6 for
c in CANDIDATES and keeps, per block, the one with the smallest squared error against the BF16 weights.
"""
import argparse
import hashlib
import json
import shutil
import struct
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])       # e2m1 magnitudes; the index is the 3-bit code
BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])       # midpoints
CANDIDATES = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)
FP8_MAX, FP4_MAX, GS = 448.0, 6.0, 16
DEV = 'cuda'


def headers(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob('*.safetensors')):
        with open(f, 'rb') as fh:
            n = struct.unpack('<Q', fh.read(8))[0]
            h = json.loads(fh.read(n))
        h.pop('__metadata__', None)
        for k, v in h.items():
            out[k] = (v['dtype'], v['shape'], f.name)
    return out


class Reader:
    def __init__(self, d: Path):
        self.d, self.h, self.open = d, headers(d), {}

    def get(self, name: str) -> torch.Tensor:
        f = self.h[name][2]
        if f not in self.open:
            self.open[f] = safe_open(str(self.d / f), framework='pt', device='cpu')
        return self.open[f].get_tensor(name)


def decode_fp4(packed: torch.Tensor, scale8: torch.Tensor, g: float) -> torch.Tensor:
    """uint8 [N, K/2], e4m3 [N, K/16], fp32 g -> fp32 [N, K]."""
    p = packed.to(DEV)
    codes = torch.stack([p & 0xF, p >> 4], dim=-1).reshape(p.shape[0], -1).long()
    mag = GRID.to(DEV)[codes & 7]
    val = torch.where((codes & 8) != 0, -mag, mag)
    s = scale8.to(DEV).float().repeat_interleave(GS, dim=1) * g
    return val * s


def encode_codes(x: torch.Tensor) -> torch.Tensor:
    """fp32 values already divided by their scale (|x| <= 6 after clamping) -> 4-bit codes (int64)."""
    a = x.abs().clamp(max=FP4_MAX)
    idx = torch.bucketize(a, BOUNDS.to(x.device), right=False)
    return idx | torch.signbit(x).long() << 3            # small negatives become -0, as ModelOpt writes them


def pack(codes: torch.Tensor) -> torch.Tensor:
    c = codes.reshape(codes.shape[0], -1, 2)
    return (c[..., 0] | (c[..., 1] << 4)).to(torch.uint8)


def e4m3(v: torch.Tensor) -> torch.Tensor:
    """Positive fp32 -> e4m3 (round to nearest even, clamped to 448) as float8 tensor."""
    return v.clamp(min=0.0, max=FP8_MAX).to(torch.float8_e4m3fn)


def quantize(w: torch.Tensor, method: str):
    """bf16/fp32 [N, K] -> (packed uint8 [N, K/2], e4m3 [N, K/16], g, fp32 dequantized, stats)."""
    w = w.to(DEV).float()
    n, k = w.shape
    assert k % 64 == 0, (n, k)
    g = float(w.abs().max()) / (FP4_MAX * FP8_MAX)
    blocks = w.view(n, k // GS, GS)
    amax = blocks.abs().amax(dim=-1)                                         # [N, K/16]
    cands = CANDIDATES if method == 'mse' else (1.0,)
    best_err = best_s8 = best_codes = None
    chosen = torch.zeros(len(cands), dtype=torch.long)
    for ci, c in enumerate(cands):
        s8 = e4m3(c * amax / FP4_MAX / g)                                    # [N, K/16] float8
        s = s8.float() * g
        safe = torch.where(s > 0, s, torch.ones_like(s))
        codes = encode_codes(blocks / safe[..., None])
        codes = torch.where((s > 0)[..., None], codes, torch.zeros_like(codes))
        mag = GRID.to(DEV)[codes & 7]
        deq = torch.where((codes & 8) != 0, -mag, mag) * s[..., None]
        err = (deq - blocks).pow(2).sum(dim=-1)                              # [N, K/16]
        if best_err is None:
            best_err, best_s8, best_codes, pick = err, s8, codes, torch.zeros_like(err, dtype=torch.long)
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_s8 = torch.where(better, s8.float(), best_s8.float()).to(torch.float8_e4m3fn)
            best_codes = torch.where(better[..., None], codes, best_codes)
            pick = torch.where(better, torch.full_like(pick, ci), pick)
    for ci in range(len(cands)):
        chosen[ci] = int((pick == ci).sum())
    packed = pack(best_codes.reshape(n, k))
    deq = decode_fp4(packed, best_s8, g)
    rel = float((deq - w).norm() / w.norm())
    return packed.cpu(), best_s8.cpu(), g, deq, {'rel_err': rel, 'choice_counts': chosen.tolist()}


def fp8_projections(h: dict) -> list:
    names = sorted(k[:-len('.weight')] for k, (dt, sh, _) in h.items()
                   if k.endswith('.weight') and dt == 'F8_E4M3' and not k.startswith(('mtp.', 'model.visual.')))
    return names


def selftest(a) -> int:
    nv, bf = Reader(Path(a.nvfp4)), Reader(Path(a.bf16))
    ok = True
    # 1. the encoder reproduces NVIDIA's own NVFP4 bytes from their decoded values and scales
    for name in ('model.language_model.layers.0.mlp.gate_proj', 'model.language_model.layers.33.mlp.down_proj',
                 'lm_head'):
        packed, s8, g = nv.get(name + '.weight'), nv.get(name + '.weight_scale'), float(nv.get(name + '.weight_scale_2'))
        val = decode_fp4(packed, s8, g)
        s = s8.to(DEV).float() * g
        blocks = val.view(val.shape[0], -1, GS)
        safe = torch.where(s > 0, s, torch.ones_like(s))
        codes = encode_codes(blocks / safe[..., None])
        codes = torch.where((s > 0)[..., None], codes, torch.zeros_like(codes))
        again = pack(codes.reshape(val.shape[0], -1)).cpu()
        equal = bool((packed == again).all())
        print(json.dumps({'test': 're-encode NVIDIA NVFP4', 'tensor': name, 'bytes_equal': equal}), flush=True)
        ok &= equal
    # 2. NVIDIA's FP8 projections come from these BF16 weights; NVFP4 error of our method vs NVIDIA's MLP NVFP4 error
    for name in ('model.language_model.layers.0.linear_attn.in_proj_qkv', 'model.language_model.layers.3.self_attn.o_proj',
                 'model.language_model.layers.61.linear_attn.out_proj'):
        w = bf.get(name + '.weight').to(DEV).float()
        w8 = nv.get(name + '.weight').to(DEV).float() * float(nv.get(name + '.weight_scale'))
        fp8_rel = float((w8 - w).norm() / w.norm())
        r = {m: quantize(w, m)[4] for m in ('rtn', 'mse')}
        print(json.dumps({'test': 'source and error', 'tensor': name, 'fp8_rel_err': round(fp8_rel, 5),
                          'nvfp4_rtn_rel_err': round(r['rtn']['rel_err'], 5), 'nvfp4_mse_rel_err': round(r['mse']['rel_err'], 5),
                          'mse_choices': r['mse']['choice_counts']}), flush=True)
        ok &= fp8_rel < 0.08
    for name in ('model.language_model.layers.0.mlp.gate_proj', 'model.language_model.layers.33.mlp.down_proj'):
        w = bf.get(name + '.weight').to(DEV).float()
        val = decode_fp4(nv.get(name + '.weight'), nv.get(name + '.weight_scale'), float(nv.get(name + '.weight_scale_2')))
        ours = quantize(w, 'mse')[4]['rel_err']
        print(json.dumps({'test': 'NVIDIA MLP NVFP4 (Local Hessian) vs ours on the same tensor', 'tensor': name,
                          'nvidia_rel_err': round(float((val - w).norm() / w.norm()), 5), 'ours_mse_rel_err': round(ours, 5)}),
              flush=True)
    print('SELFTEST', 'PASSED' if ok else 'FAILED', flush=True)
    return 0 if ok else 1


def build(a) -> int:
    src, out = Path(a.nvfp4), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    nv, bf = Reader(src), Reader(Path(a.bf16))
    names = fp8_projections(nv.h)
    assert len(names) == 208, len(names)
    stats, t0 = {}, time.time()
    replaced = {}
    for i, name in enumerate(names):
        w = bf.get(name + '.weight')
        assert list(w.shape) == nv.h[name + '.weight'][1], name
        packed, s8, g, deq, st = quantize(w, a.method)
        w8 = nv.get(name + '.weight').to(DEV).float() * float(nv.get(name + '.weight_scale'))
        wf = w.to(DEV).float()
        st['fp8_rel_err'] = float((w8 - wf).norm() / wf.norm())
        assert st['fp8_rel_err'] < 0.08, (name, st)                       # the BF16 source matches NVIDIA's FP8 weights
        in8 = nv.get(name + '.input_scale')
        replaced[name] = {'weight': packed, 'weight_scale': s8, 'weight_scale_2': torch.tensor(g, dtype=torch.float32),
                          'input_scale': (in8.float() / FP4_MAX).reshape(())}   # unused by TensorFold (W4A16)
        stats[name] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in st.items()}
        if i % 20 == 0:
            print(json.dumps({'done': i, 'of': len(names), 'tensor': name, **stats[name], 's': round(time.time() - t0)}), flush=True)
        del deq, w8, wf
    # write shards: same shard for each tensor as in NVIDIA's checkpoint
    index = json.loads((src / 'model.safetensors.index.json').read_text())
    wm = index['weight_map']
    by_shard = {}
    for k, (dt, sh, f) in nv.h.items():
        by_shard.setdefault(f, []).append(k)
    new_map = {}
    for f, keys in sorted(by_shard.items()):
        tensors = {}
        for k in sorted(keys):
            base, _, suffix = k.rpartition('.')
            if base in replaced:
                if suffix in ('weight', 'weight_scale', 'input_scale'):
                    tensors[k] = replaced[base][suffix]
                continue
            tensors[k] = nv.get(k)
        for base in sorted(replaced):
            if wm.get(base + '.weight') == f:
                tensors[base + '.weight_scale_2'] = replaced[base]['weight_scale_2']
        save_file(tensors, str(out / f), metadata={'format': 'pt'})
        for k in tensors:
            new_map[k] = f
        print(json.dumps({'wrote': f, 'tensors': len(tensors)}), flush=True)
        del tensors
    index['weight_map'] = dict(sorted(new_map.items()))
    (out / 'model.safetensors.index.json').write_text(json.dumps(index, indent=2))
    # configs: the 208 projections move from the FP8 group to the NVFP4 group
    cfg = json.loads((src / 'config.json').read_text())
    q = cfg['quantization_config']
    fp8_group = [g for g, d in q['config_groups'].items() if d['weights']['num_bits'] == 8]
    fp4_group = [g for g, d in q['config_groups'].items() if d['weights']['num_bits'] == 4]
    assert len(fp8_group) == 1 and len(fp4_group) == 1
    q['config_groups'][fp4_group[0]]['targets'] = sorted(q['config_groups'][fp4_group[0]]['targets'] +
                                                          q['config_groups'][fp8_group[0]]['targets'])
    del q['config_groups'][fp8_group[0]]
    for name in names:
        q['quantized_layers'][name] = {'quant_algo': 'NVFP4', 'group_size': 16}
    (out / 'config.json').write_text(json.dumps(cfg, indent=2))
    hq = json.loads((src / 'hf_quant_config.json').read_text())
    for name in names:
        hq['quantization']['quantized_layers'][name] = {'quant_algo': 'NVFP4', 'group_size': 16}
    (out / 'hf_quant_config.json').write_text(json.dumps(hq, indent=4))
    for p in sorted(src.iterdir()):
        if p.suffix != '.safetensors' and p.name not in ('model.safetensors.index.json', 'config.json',
                                                           'hf_quant_config.json', 'README.md'):
            shutil.copyfile(p, out / p.name)
    me = Path(__file__).read_bytes()
    prov = {'what': 'NVIDIA Qwen3.8-27B NVFP4 with the 208 FP8 projections re-quantized to NVFP4 (PLAN_15, A)',
            'nvfp4_source': str(src), 'bf16_source': a.bf16, 'method': a.method, 'candidates': list(CANDIDATES),
            'script_sha256': hashlib.sha256(me).hexdigest(), 'torch': torch.__version__, 'seconds': round(time.time() - t0),
            'per_tensor': stats}
    (out / 'PROVENANCE.json').write_text(json.dumps(prov, indent=1))
    errs = [v['rel_err'] for v in stats.values()]
    print(json.dumps({'built': str(out), 'projections': len(names), 'rel_err_mean': round(sum(errs) / len(errs), 5),
                      'rel_err_max': round(max(errs), 5)}), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--nvfp4', required=True)
    ap.add_argument('--bf16', required=True)
    ap.add_argument('--out')
    ap.add_argument('--method', default='mse', choices=('mse', 'rtn'))
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    return selftest(a) if a.selftest else build(a)


if __name__ == '__main__':
    sys.exit(main())
