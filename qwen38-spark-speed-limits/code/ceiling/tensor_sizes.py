"""PLAN_22 M1: the weight tensors one decode forward reads, grouped by module, from the checkpoint's safetensors headers
(no weight is read). Embedding (one row per token), vision tower and MTP head are excluded, as in PLAN_14. A module's
weight, block scales and global scales form one group (one matmul reads them together). Writes tensors.json and
sizes.txt (one byte count per line, in layer order) next to this file; bw_tuned.cu's forward pattern reads sizes.txt
and m3_mm.py reads tensors.json.

    python3 tensor_sizes.py <snapshot folder of nvidia/Qwen3.8-27B-NVFP4 @482ca0f3>
"""
import json
import re
import struct
import sys
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUFFIX = re.compile(r'\.(weight|weight_scale|weight_scale_2|input_scale|bias)$')


def part(name):
    if 'visual' in name or 'vision' in name:
        return 'vision (not read)'
    if 'embed_tokens' in name:
        return 'embedding (one row a token)'
    if 'mtp' in name:
        return 'mtp head (not used)'
    if 'lm_head' in name:
        return 'head'
    if any(s in name for s in ('mlp.', 'gate_proj', 'up_proj', 'down_proj')):
        return 'MLP'
    if 'linear_attn' in name:
        return 'DeltaNet'
    if 'self_attn' in name:
        return 'full attention'
    return 'other'


def layer_key(name):
    m = re.search(r'layers\.(\d+)\.', name)
    return (0, int(m.group(1))) if m else ((1, 0) if 'lm_head' in name else (-1, 0))


def main():
    ckpt = Path(sys.argv[1])
    groups = OrderedDict()
    for f in sorted(ckpt.glob('*.safetensors')):
        with open(f, 'rb') as fh:
            n = struct.unpack('<Q', fh.read(8))[0]
            header = json.loads(fh.read(n))
        for k, v in header.items():
            if k == '__metadata__' or '(' in part(k):
                continue
            mod = SUFFIX.sub('', k)
            g = groups.setdefault(mod, {'module': mod, 'part': part(k), 'bytes': 0, 'tensors': []})
            g['bytes'] += v['data_offsets'][1] - v['data_offsets'][0]
            g['tensors'].append([k[len(mod):] or '(self)', v['dtype'], v['shape']])
    rows = sorted(groups.values(), key=lambda g: (layer_key(g['module']), g['module']))
    total = sum(g['bytes'] for g in rows)
    big = [g for g in rows if g['bytes'] >= 1 << 20]
    out = {'checkpoint': 'nvidia/Qwen3.8-27B-NVFP4@482ca0f3', 'groups': len(rows), 'total_gb': round(total / 1e9, 4),
           'groups_ge_1MiB': len(big), 'bytes_ge_1MiB_gb': round(sum(g['bytes'] for g in big) / 1e9, 4), 'rows': rows}
    (HERE / 'tensors.json').write_text(json.dumps(out, indent=1))
    (HERE / 'sizes.txt').write_text(''.join(f"{g['bytes']}\n" for g in rows))
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
