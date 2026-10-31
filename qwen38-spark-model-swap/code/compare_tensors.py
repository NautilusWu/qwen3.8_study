"""Compare two safetensors checkpoints by their headers: tensor names, dtypes and shapes (no weights are loaded).

    python3 compare_tensors.py <folder A> <folder B>
"""
import json
import os
import struct
import sys


def load(d):
    idx = json.load(open(os.path.join(d, 'model.safetensors.index.json')))['weight_map']
    out = {}
    for fn in sorted(set(idx.values())):
        with open(os.path.join(d, fn), 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            h = json.loads(f.read(n))
        h.pop('__metadata__', None)
        out.update({k: (v['dtype'], v['shape']) for k, v in h.items()})
    return out


a, b = load(sys.argv[1]), load(sys.argv[2])
print('tensors', len(a), len(b))
print('only in A:', sorted(set(a) - set(b)))
print('only in B:', sorted(set(b) - set(a)))
diff = [(k, a[k], b[k]) for k in sorted(set(a) & set(b)) if a[k] != b[k]]
print('dtype/shape differences:', len(diff))
for d in diff[:20]:
    print('  ', d)
