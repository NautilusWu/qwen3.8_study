"""Make a checkpoint folder that is the published NVFP4/W4A4-W8A8 folder with only lm_head replaced by NVFP4.

    python3 make_overlay.py <published W4A4-W8A8 folder> <new folder> <quantized text-06.safetensors>

Every other file is a relative symlink to the published file (nothing is modified). The index gains the two scale
tensors; hf_quant_config.json and config.json list lm_head as NVFP4 group 16, as in NVIDIA's own checkpoint.
In the published folder lm_head is the only tensor in text-06.safetensors; the script checks that.
"""
import json
import os
import shutil
import sys
from pathlib import Path

src, out, head = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=False)
EDIT = {'model.safetensors.index.json', 'hf_quant_config.json', 'config.json', 'text-06.safetensors', 'SHA256SUMS'}
for f in sorted(src.iterdir()):
    if f.name not in EDIT:
        os.symlink(os.path.relpath(f, out), out / f.name)
idx = json.loads((src / 'model.safetensors.index.json').read_text())
assert [k for k, v in idx['weight_map'].items() if v == 'text-06.safetensors'] == ['lm_head.weight']
for k in ('lm_head.weight_scale', 'lm_head.weight_scale_2'):
    idx['weight_map'][k] = 'text-06.safetensors'
idx.pop('metadata', None)
(out / 'model.safetensors.index.json').write_text(json.dumps(idx, indent=2))
hq = json.loads((src / 'hf_quant_config.json').read_text())
hq['quantization']['quantized_layers']['lm_head'] = {'quant_algo': 'NVFP4', 'group_size': 16}
(out / 'hf_quant_config.json').write_text(json.dumps(hq, indent=4))
cfg = json.loads((src / 'config.json').read_text())
cfg['quantization_config'].setdefault('quantized_layers', {})['lm_head'] = {'quant_algo': 'NVFP4', 'group_size': 16}
(out / 'config.json').write_text(json.dumps(cfg, indent=2))
shutil.copy(head, out / 'text-06.safetensors')
print('wrote', out)
