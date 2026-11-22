"""PLAN_16 B1c: fine-tune DFlash2 on the NVFP4 target's own taps and replies, block 16 (as TensorFold drafts).
Runs inside the 0.6.0 image:

    python -I /p16/b1c_train.py <drafter dir> <NVIDIA NVFP4 dir> <data dir> <out dir> [--eval-only] [options]

Data: <data>/taps/*.safetensors (taps, ids) and index.jsonl from b1b_taps.py. Held out: sequences with
crc32(id) % 20 == 0. A step = one sequence, up to --anchors anchors drawn in its reply (c from the prompt end to
T - 16); the block [t_c, mask x 15] predicts t_{c+1..c+15}.
Loss = sum_k w_k CE_k / sum_k w_k with w_k = exp(-(k-1)/gamma) over targets inside the head's spans, plus
--kl x KL(original drafter || new) on the same rows. Trained: everything except the candidate selector (the
embedding and head are the target's, frozen). fp32 master weights, bf16 autocast, AdamW, warm-up then cosine.
Quantization-aware: the student's packed tensors go through TensorFold's 4-bit packing (dflash_twin.fake_quant4,
straight-through), so training and held-out numbers are for the drafter as served; the KL teacher is the original
drafter unpacked (bf16). Held-out metrics (same anchors every time): acceptance proxy = mean count of consecutive positions whose top-1 is
the reply's token (0..15), and top-16 recall by position. The best checkpoint by the proxy is exported in the
original format (bf16 safetensors + config.json), the original's numbers are measured first (step 0).
"""
import argparse
import json
import math
import random
import shutil
import sys
import time
import zlib
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, '/p16')
sys.path.insert(0, '/p15')
import dflash_twin as DT  # noqa: E402
import quantize_attn as QA  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

L = 16


def held_out(i):
    return zlib.crc32(i.encode()) % 20 == 0


def load_seq(path):
    d = load_file(str(path), device='cuda')
    return d['taps'], d['ids'].long()


def anchors_for(n, prompt_len, k, rnd):
    lo, hi = max(prompt_len, 1), n - L
    if hi <= lo:
        return []
    cand = list(range(lo, hi + 1))
    return sorted(rnd.sample(cand, min(k, len(cand))))


def batch_forward(model, taps, ids, anchors):
    A = torch.tensor(anchors, device='cuda')
    lo = max(0, anchors[0] - model.window - L)
    hi = anchors[-1]
    h = model(taps[lo:hi], torch.arange(lo, hi, device='cuda'), A, ids[A], L)          # (A, 15, h)
    tgt = ids[A[:, None] + torch.arange(1, L, device='cuda')[None, :]]                    # (A, 15)
    return h, model.local_of[tgt]                                                         # local ids, -1 outside spans


def evaluate(model, items, data, n_anchor=32):
    model.eval()
    acc_len, recall, rows = [], torch.zeros(L - 1), torch.zeros(L - 1)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for it in items:
            taps, ids = load_seq(data / 'taps' / f"{it['id']}.safetensors")
            an = anchors_for(it['len'], it['prompt_len'], n_anchor, random.Random(it['id']))
            if not an:
                continue
            h, tl = batch_forward(model, taps, ids, an)
            logits = model.logits(h).float()
            top = logits.topk(16, -1).indices                                             # (A, 15, 16)
            hit1 = (top[..., 0] == tl) & (tl >= 0)
            # consecutive top-1 hits from position 1
            run = torch.cumprod(hit1.int(), dim=1).sum(1)
            acc_len += run.tolist()
            in16 = ((top == tl[..., None]).any(-1) & (tl >= 0)).float()
            recall += in16.sum(0).cpu()
            rows += torch.ones_like(in16).sum(0).cpu()
            del taps, ids, h, logits
    model.train()
    return {'accept_proxy': round(sum(acc_len) / max(1, len(acc_len)), 4), 'anchors': len(acc_len),
            'top16_recall_by_pos': [round(float(x), 4) for x in (recall / rows.clamp(min=1))]}


def export(model, src_dir, out_dir, meta):
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {n: t.to(torch.bfloat16).contiguous().cpu() for n, t in model.state_for_export().items()}
    save_file(state, str(out_dir / 'model.safetensors'), metadata={'format': 'pt'})
    shutil.copyfile(Path(src_dir) / 'config.json', out_dir / 'config.json')
    (out_dir / 'FINETUNE.json').write_text(json.dumps(meta, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('drafter')
    ap.add_argument('target')
    ap.add_argument('data')
    ap.add_argument('out')
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--epochs', type=float, default=2.0)
    ap.add_argument('--anchors', type=int, default=64)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--warmup', type=int, default=50)
    ap.add_argument('--gamma', type=float, default=7.0)
    ap.add_argument('--kl', type=float, default=0.5)
    ap.add_argument('--eval-every', type=int, default=200)
    ap.add_argument('--seed', type=int, default=2026)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index = [json.loads(l) for l in (data / 'taps' / 'index.jsonl').read_text().splitlines()]
    train = [it for it in index if not held_out(it['id'])]
    hold = [it for it in index if held_out(it['id'])]
    nv = QA.Reader(Path(a.target))
    embed = nv.get('model.language_model.embed_tokens.weight').to('cuda', torch.bfloat16)
    head = QA.decode_fp4(nv.get('lm_head.weight'), nv.get('lm_head.weight_scale'),
                         float(nv.get('lm_head.weight_scale_2'))).to(torch.bfloat16)
    head_rows = torch.cat([head[x:y] for x, y in DT.SPANS])
    del head
    model = DT.Twin(a.drafter, embed, head_rows, dtype=torch.float32).train()
    log = open(out / 'train_log.jsonl', 'a')

    def emit(rec):
        log.write(json.dumps(rec) + '\n')
        log.flush()
        print(json.dumps(rec), flush=True)

    base_bf16 = evaluate(model, hold, data)                 # the original drafter unpacked (an upper reference)
    model.quant4 = True                                      # from here on: as TensorFold serves it, 4-bit packed
    base = evaluate(model, hold, data)
    emit({'step': 0, 'train_sequences': len(train), 'held_out_sequences': len(hold), 'original_bf16': base_bf16,
          'original_served_4bit': base})
    if a.eval_only:
        return
    ref = DT.Twin(a.drafter, embed, head_rows, dtype=torch.bfloat16).eval()     # teacher: original, unpacked
    for p in ref.parameters():
        p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    steps = int(a.epochs * len(train))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) *
                                              (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps))))))
    pw = DT.position_weights(L - 1, a.gamma).cuda()
    rnd = random.Random(a.seed)
    best, t0, step = base['accept_proxy'], time.time(), 0
    order = []
    while step < steps:
        if not order:
            order = list(range(len(train)))
            rnd.shuffle(order)
        it = train[order.pop()]
        an = anchors_for(it['len'], it['prompt_len'], a.anchors, rnd)
        if not an:
            continue
        taps, ids = load_seq(data / 'taps' / f"{it['id']}.safetensors")
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h, tl = batch_forward(model, taps, ids, an)
            logits = model.logits(h).float()
            with torch.no_grad():
                rh, _ = batch_forward(ref, taps, ids, an)
                rlog = ref.logits(rh).float()
        ok = tl >= 0
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tl.clamp(min=0).reshape(-1), reduction='none').view(tl.shape)
        wts = pw[None, :] * ok
        loss_ce = (ce * wts).sum() / wts.sum().clamp(min=1)
        kl = F.kl_div(F.log_softmax(logits, -1), F.log_softmax(rlog, -1), log_target=True, reduction='none').sum(-1)
        loss_kl = (kl * pw[None, :]).sum() / (pw.sum() * kl.shape[0])
        loss = loss_ce + a.kl * loss_kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 20 == 0:
            emit({'step': step, 'loss_ce': round(float(loss_ce), 4), 'loss_kl': round(float(loss_kl), 4),
                  'grad_norm': round(float(gn), 3), 'lr': sched.get_last_lr()[0], 's': round(time.time() - t0)})
        if step % a.eval_every == 0 or step == steps:
            ev = evaluate(model, hold, data)
            emit({'step': step, 'eval': True, **ev})
            if ev['accept_proxy'] > best:
                best = ev['accept_proxy']
                export(model, a.drafter, out / 'best', {'step': step, 'held_out': ev, 'original_served_4bit': base,
                                                       'original_bf16': base_bf16, 'args': vars(a)})
        del taps, ids, h, logits, rh, rlog
    emit({'done': True, 'best_accept_proxy': best, 'original_accept_proxy': base['accept_proxy'],
          'gain': round(best / base['accept_proxy'], 4) if base['accept_proxy'] else None})


if __name__ == '__main__':
    main()
