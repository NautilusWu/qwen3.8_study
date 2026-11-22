"""PLAN_16 B1a: a trainable PyTorch copy of TensorFold 0.6.0's DFlash2 drafter (the reference path of
families/qwen3_5/cuda/dflash2.py: DFlash2._layer, _attention, add_taps and the non-fast branch of launch_blocks).

Inference, as TensorFold runs it: committed target rows 0..c-1 give taps (layers 5, 19, 33, 47, 61 concatenated,
5 x 5120); fc + hidden_norm project them; every drafter layer projects the projected rows with its own k/v (k_norm,
RoPE at the row's position) as context keys, a sliding window of 2,047 rows. The block is [pending token t_c, mask x
(L-1)] at positions c..c+L-1, embedded with the target's embedding; five layers (dynamic two-tap convolutions around
non-causal attention over context + block, and around the MLP); the final norm on block rows 1..L-1 and the target's
head restricted to the token spans give the logits for t_{c+1}..t_{c+L-1}.

Training uses many anchors per sequence: the projected rows and context keys/values are computed once per sequence,
every anchor's block attends its own window through one attention mask.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SPANS = ((0, 98304), (248032, 248320))           # the drafter head's token ids (TensorFold's vocab_spans)
TRAINABLE_SKIP = ('candidate_selector.',)        # kept frozen


def packed_draft(name: str, shape) -> bool:
    """TensorFold's rule (affine_memory.packed_draft): which drafter tensors it packs to 4 bits at load."""
    return (len(shape) == 2 and name.endswith('.weight') and shape[0] % 64 == 0 and shape[1] % 64 == 0
            and shape[0] * shape[1] >= 1 << 20)


def fake_quant4(w: torch.Tensor) -> torch.Tensor:
    """TensorFold's quantize4 (MLX affine 4-bit, groups of 64 along K, bf16 scale and bias) as values, with a
    straight-through gradient: forward sees the packed weights, backward passes the gradient to w."""
    n, k = w.shape
    g = w.to(torch.bfloat16).float().view(n, k // 64, 64)      # TensorFold packs the exported bf16 weights
    lo, hi = g.amin(-1), g.amax(-1)
    scale = ((hi - lo) / 15).clamp_min(1e-8).to(torch.bfloat16).float()
    bias = lo.to(torch.bfloat16).float()
    q = torch.round((g - bias[..., None]) / scale[..., None]).clamp(0, 15)
    wq = (q * scale[..., None] + bias[..., None]).view(n, k)
    return w + (wq.to(w.dtype) - w).detach()


def rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def rope(x: torch.Tensor, pos: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """x (..., rows, head_dim), pos (rows,) -> rotate-half RoPE."""
    half = x.shape[-1] // 2
    ph = pos.float()[:, None] * inv_freq[None, :]
    cos, sin = ph.cos(), ph.sin()
    a, b = x[..., :half].float(), x[..., half:].float()
    return torch.cat((a * cos - b * sin, b * cos + a * sin), -1).to(x.dtype)


def dconv(x: torch.Tensor, dyn: torch.Tensor, base: torch.Tensor, branch: int, gs: int, seg: int) -> torch.Tensor:
    """Two-tap grouped dynamic convolution inside each ``seg``-row block: row r mixes rows r and r-1."""
    n, d = x.shape
    xb = x.view(-1, seg, d)
    prev = torch.cat((torch.zeros_like(xb[:, :1]), xb[:, :-1]), 1).view(n, d)
    # TensorFold adds base and dynamic kernels and rounds the sum to bf16 before the fp32 product (dflash2._conv/_dconv)
    k0 = (base[branch, 0].float() + dyn[:, branch, 0].float().repeat_interleave(gs, -1)).to(torch.bfloat16).float()
    k1 = (base[branch, 1].float() + dyn[:, branch, 1].float().repeat_interleave(gs, -1)).to(torch.bfloat16).float()
    return (x.float() * k0 + prev.float() * k1).to(x.dtype)


class Twin(nn.Module):
    def __init__(self, draft_dir: str | Path, embed: torch.Tensor, head_rows: torch.Tensor, device='cuda',
                 dtype=torch.bfloat16):
        super().__init__()
        from safetensors.torch import load_file

        cfg = json.loads((Path(draft_dir) / 'config.json').read_text())
        self.cfg = cfg
        self.hidden, self.hd = cfg['hidden_size'], cfg['head_dim']
        self.heads, self.kv = cfg['num_attention_heads'], cfg['num_key_value_heads']
        self.eps, self.layers_n = cfg['rms_norm_eps'], cfg['num_hidden_layers']
        self.gs = cfg['dflash_config']['conv_group_size']
        self.mask_id = cfg['dflash_config']['mask_token_id']
        self.window = cfg['sliding_window'] - 1
        assert not cfg.get('is_causal', True)
        w = load_file(str(Path(draft_dir) / 'model.safetensors'), device='cpu')
        self.names = sorted(w)
        self.p = nn.ParameterDict({n.replace('.', '__'): nn.Parameter(t.to(device, dtype if t.dtype != torch.float32 else t.dtype),
                                                                      requires_grad=not n.startswith(TRAINABLE_SKIP))
                                   for n, t in w.items()})
        self.register_buffer('embed', embed.to(device, dtype), persistent=False)          # frozen target embedding
        self.register_buffer('head', head_rows.to(device, dtype), persistent=False)       # frozen target head, span rows
        self.register_buffer('head_ids', torch.cat([torch.arange(a, b) for a, b in SPANS]).to(device), persistent=False)
        lut = torch.full((cfg['vocab_size'],), -1, dtype=torch.long)
        lut[self.head_ids.cpu()] = torch.arange(len(self.head_ids))
        self.register_buffer('local_of', lut.to(device), persistent=False)
        self.register_buffer('inv_freq', 1.0 / cfg['rope_parameters']['rope_theta'] **
                             (torch.arange(self.hd // 2, device=device, dtype=torch.float32) * 2 / self.hd), persistent=False)

    quant4 = False                                   # True: packed tensors as TensorFold serves them (4-bit)

    def w(self, name: str) -> torch.Tensor:
        t = self.p[name.replace('.', '__')]
        if self.quant4 and packed_draft(name, t.shape):
            return fake_quant4(t)
        return t

    def state_for_export(self) -> dict:
        """The master weights: TensorFold packs the exported bf16 tensors at load exactly as fake_quant4 does."""
        return {n: self.p[n.replace('.', '__')].detach() for n in self.names}

    # --- context -----------------------------------------------------------------------------------------------
    def project(self, taps: torch.Tensor) -> torch.Tensor:
        """taps (T, 5 * hidden) -> projected rows (T, hidden)."""
        return rms(F.linear(taps, self.w('fc.weight')), self.w('hidden_norm.weight'), self.eps)

    def context_kv(self, layer: int, projected: torch.Tensor, pos: torch.Tensor):
        pre = f'layers.{layer}.self_attn.'
        k = F.linear(projected, self.w(pre + 'k_proj.weight')).view(-1, self.kv, self.hd)
        v = F.linear(projected, self.w(pre + 'v_proj.weight')).view(-1, self.kv, self.hd)
        k = rope(rms(k, self.w(pre + 'k_norm.weight'), self.eps).transpose(0, 1), pos, self.inv_freq)
        return k, v.transpose(0, 1)                                      # (kv, T, hd)

    # --- blocks ------------------------------------------------------------------------------------------------
    def forward(self, taps: torch.Tensor, ctx_pos: torch.Tensor, anchors: torch.Tensor, pending: torch.Tensor,
                length: int) -> torch.Tensor:
        """taps (T, 5h) of committed rows at positions ctx_pos (T,) (0..T-1 for one sequence); anchors (A,): anchor c
        means context rows < c (the last ``window`` of them) and the block at positions c..c+L-1; pending (A,): t_c.
        Returns hidden states of block rows 1..L-1 after the final norm, (A, L-1, hidden)."""
        A, L, dev = anchors.shape[0], length, taps.device
        projected = self.project(taps)
        tokens = torch.full((A, L), self.mask_id, dtype=torch.long, device=dev)
        tokens[:, 0] = pending
        x = self.embed[tokens.view(-1)]                                   # (A*L, h)
        bpos = (anchors[:, None] + torch.arange(L, device=dev)[None, :]).reshape(-1)
        T = taps.shape[0]
        # mask (as DFlash2._attention): the block query at absolute position p of anchor c attends context rows j < c
        # with p - j <= window, and every row of its own block (non-causal)
        q_anchor = anchors.repeat_interleave(L)                          # (A*L,)
        j = ctx_pos[None, :]
        ctx_ok = (j < q_anchor[:, None]) & (bpos[:, None] - j <= self.window)
        blk_id = torch.arange(A, device=dev).repeat_interleave(L)
        blk_ok = blk_id[:, None] == blk_id[None, :]
        allowed = torch.cat((ctx_ok, blk_ok), 1)                         # (A*L, T + A*L)
        for i in range(self.layers_n):
            x = self.layer(i, x, projected, ctx_pos, bpos, allowed, L)
        h = rms(x.view(A, L, -1)[:, 1:], self.w('norm.weight'), self.eps)
        return h

    def layer(self, i, x, projected, ctx_pos, bpos, allowed, L):
        b = f'layers.{i}.'
        res = x
        normed = rms(x, self.w(b + 'input_layernorm.weight'), self.eps)
        dyn = F.linear(normed, self.w(b + 'attention_conv.kernel_projection.weight')).view(len(x), 2, 2, self.hidden // self.gs)
        base = self.w(b + 'attention_conv.base_kernel')
        xc = dconv(normed, dyn, base, 0, self.gs, L)
        x = (res.float() + dconv(self.attention(i, xc, projected, ctx_pos, bpos, allowed), dyn, base, 1, self.gs, L).float()).to(x.dtype)
        res = x
        normed = rms(x, self.w(b + 'post_attention_layernorm.weight'), self.eps)
        dyn = F.linear(normed, self.w(b + 'mlp_conv.kernel_projection.weight')).view(len(x), 2, 2, self.hidden // self.gs)
        base = self.w(b + 'mlp_conv.base_kernel')
        h = dconv(normed, dyn, base, 0, self.gs, L)
        m = F.linear(F.silu(F.linear(h, self.w(b + 'mlp.gate_proj.weight'))) * F.linear(h, self.w(b + 'mlp.up_proj.weight')),
                     self.w(b + 'mlp.down_proj.weight'))
        return (res.float() + dconv(m, dyn, base, 1, self.gs, L).float()).to(x.dtype)

    def attention(self, i, x, projected, ctx_pos, bpos, allowed):
        pre = f'layers.{i}.self_attn.'
        n = len(x)
        q = F.linear(x, self.w(pre + 'q_proj.weight')).view(n, self.heads, self.hd)
        k = F.linear(x, self.w(pre + 'k_proj.weight')).view(n, self.kv, self.hd)
        v = F.linear(x, self.w(pre + 'v_proj.weight')).view(n, self.kv, self.hd)
        q = rope(rms(q, self.w(pre + 'q_norm.weight'), self.eps).transpose(0, 1), bpos, self.inv_freq)
        k = rope(rms(k, self.w(pre + 'k_norm.weight'), self.eps).transpose(0, 1), bpos, self.inv_freq)
        v = v.transpose(0, 1)
        kc, vc = self.context_kv(i, projected, ctx_pos)
        keys = torch.cat((kc, k), 1).repeat_interleave(self.heads // self.kv, 0)
        vals = torch.cat((vc, v), 1).repeat_interleave(self.heads // self.kv, 0)
        out = F.scaled_dot_product_attention(q[None], keys[None], vals[None], attn_mask=allowed[None, None],
                                             scale=self.hd ** -0.5)
        return F.linear(out[0].transpose(0, 1).reshape(n, self.heads * self.hd), self.w(pre + 'o_proj.weight'))

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """Block hidden states -> logits over the head's span rows (local ids; ``head_ids`` maps them back)."""
        return F.linear(h, self.head)


def position_weights(n: int, gamma: float = 7.0) -> torch.Tensor:
    """DFlash's loss weight by block position k = 1..n: exp(-(k - 1) / gamma)."""
    return torch.exp(-torch.arange(n, dtype=torch.float32) / gamma)
