"""PLAN_05 addendum 01: the community checkpoint's BF16 lm_head -> NVFP4 (ModelOpt, block 16), as the original's.
Runs in the TensorFold image, CPU only, no network:  quantize_head.py <src text-06.safetensors> <out text-06.safetensors>
Writes lm_head.weight (U8 [N, K/2]), lm_head.weight_scale (F8_E4M3 [N, K/16]), lm_head.weight_scale_2 (F32 []),
then checks: TensorFold's own dequant against the BF16 weight (relative Frobenius error) and ModelOpt's dequant
against TensorFold's on sample rows (packing order)."""
import json
import sys

import numpy as np
import torch
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
from safetensors.torch import load_file, save_file
from tensorfold.cuda.nvfp4 import format as fmt

src, out = sys.argv[1], sys.argv[2]
w = load_file(src)['lm_head.weight']
assert w.dtype == torch.bfloat16 and list(w.shape) == [248320, 5120], (w.dtype, w.shape)
q, scale, scale2 = NVFP4QTensor.quantize(w, 16)
packed = q._quantized_data.contiguous()
scale2 = scale2.float().reshape(())
tensors = {'lm_head.weight': packed, 'lm_head.weight_scale': scale.contiguous(), 'lm_head.weight_scale_2': scale2}
for k, v in tensors.items():
    print(k, v.dtype, list(v.shape))
assert packed.dtype == torch.uint8 and list(packed.shape) == [248320, 2560]
assert scale.dtype == torch.float8_e4m3fn and list(scale.shape) == [248320, 320]
save_file(tensors, out, metadata={'format': 'pt'})

# check 1: TensorFold's dequant vs the BF16 weight, in row chunks (float32)
num = den = 0.0
for i in range(0, 248320, 16384):
    d = fmt.dequant('nvfp4', packed[i:i + 16384].numpy(), scale[i:i + 16384].view(torch.uint8).numpy(), float(scale2))
    ref = w[i:i + 16384].float().numpy()
    num += float(((d - ref) ** 2).sum()); den += float((ref ** 2).sum())
rel = (num / den) ** 0.5
# check 2: ModelOpt's dequant vs TensorFold's on 512 sample rows (packing order and scale layout)
rows = torch.linspace(0, 248319, 512).long()
sub = NVFP4QTensor(torch.Size([512, 5120]), torch.bfloat16, packed[rows])
mo = sub.dequantize(dtype=torch.float32, scale=scale[rows], double_scale=scale2, block_sizes={-1: 16}).numpy()
tf = fmt.dequant('nvfp4', packed[rows].numpy(), scale[rows].view(torch.uint8).numpy(), float(scale2))
agree = float(np.abs(mo - tf).max())
print(json.dumps({'relative_frobenius_error': round(rel, 4), 'modelopt_vs_tensorfold_max_abs_diff': agree,
                  'weight_scale_2': float(scale2), 'bf16_absmax': float(w.abs().max())}))
