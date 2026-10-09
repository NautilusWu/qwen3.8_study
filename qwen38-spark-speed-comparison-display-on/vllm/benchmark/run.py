"""Run one condition (one server launch) of the study.

    python -m benchmark.run --config configs/nvidia_nvfp4_mtp3.yaml --out results/<run>/nvidia_nvfp4_mtp3 \
        --block 0 --subset full

Expects the server for this condition to be up (scripts/serve.sh). Refuses to
run if the served model name does not match the condition, so a stale server
can never be measured under the wrong label.

Outputs in --out:
    requests.jsonl   one line per measured request (raw tokens + timings + metrics)
    warmup.jsonl     warm-up requests (never used in results)
    telemetry.csv    power / clocks / temperature / memory samples
    run_meta.json    config, versions, prompt order, timestamps
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import socket
import sys
import time
from pathlib import Path

from benchmark import client, config, metrics


def load_prompts(path: str, subset: str) -> list[dict]:
    prompts = [json.loads(l) for l in open(path) if l.strip()]
    if subset == "full":
        return prompts
    if subset == "pilot":
        return [p for p in prompts if p.get("pilot")]
    ids = set(subset.split(","))
    sel = [p for p in prompts if p["id"] in ids]
    missing = ids - {p["id"] for p in sel}
    if missing:
        raise SystemExit(f"unknown prompt ids: {sorted(missing)}")
    return sel


def messages_for(prompt: dict, gen: dict) -> list[dict]:
    return [{"role": "system", "content": gen["system_prompt"]}] + prompt["messages"]


def sha256_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prompt_sha256(prompt: dict) -> str:
    return hashlib.sha256(json.dumps(prompt, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def server_load(host: str, port: int) -> tuple[float, float]:
    m = metrics.parse_prometheus(client.get_text(host, port, "/metrics"))
    return m.get("vllm:num_requests_running", 0.0), m.get("vllm:num_requests_waiting", 0.0)


def wait_idle(host: str, port: int, timeout: float) -> float:
    """Block until the server has no running or waiting request; return the wait in s.

    Single-stream rule: after a timeout or broken stream the next request may
    only start once the server has really dropped the old one, otherwise the leftover
    generation would run concurrently with the next measured request.
    """
    t0 = time.time()
    while True:
        running, waiting = server_load(host, port)
        if running == 0 and waiting == 0:
            return time.time() - t0
        if time.time() - t0 > timeout:
            raise SystemExit(f"server still busy after {timeout:.0f}s (running={running}, waiting={waiting}); "
                             "refusing to start a concurrent request")
        time.sleep(1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=0, help="randomisation block (one server launch)")
    ap.add_argument("--repeats", type=int, default=1, help="passes over the prompt set within this launch")
    ap.add_argument("--subset", default="full", help="full | pilot | comma-separated prompt ids")
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts (smoke tests)")
    ap.add_argument("--max-tokens", type=int, default=0, help="override generation.max_tokens (smoke tests only)")
    ap.add_argument("--no-max-tokens", action="store_true",
                    help="no generation cap (answers end at EOS or the context limit); for output-length checks only")
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    cfg = config.load(args.config)
    gen = dict(cfg["generation"])
    if args.max_tokens:
        gen["max_tokens"] = args.max_tokens
    if args.no_max_tokens:
        gen["max_tokens"] = None
    bench = cfg["benchmark"]
    port = cfg["server"]["port"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    served = client.get_json(args.host, port, "/v1/models")
    served_names = [m["id"] for m in served["data"]]
    if served_names != [cfg["served_model_name"]]:
        raise SystemExit(f"server is serving {served_names}, expected [{cfg['served_model_name']}]")
    try:
        vllm_version = client.get_json(args.host, port, "/version")
    except Exception:
        vllm_version = None

    prompts = load_prompts(bench["prompts"], args.subset)
    if args.limit:
        prompts = prompts[: args.limit]
    warmups = [json.loads(l) for l in open(bench["warmup_prompts"]) if l.strip()]

    # Prompt order depends only on (seed, block, repeat): identical across the
    # conditions of a block (paired design), different between blocks.
    orders = []
    for rep in range(args.repeats):
        order = [p["id"] for p in prompts]
        random.Random(f"{gen['seed']}-block{args.block}-rep{rep}").shuffle(order)
        orders.append(order)
    by_id = {p["id"]: p for p in prompts}

    meta = {
        "condition": cfg["condition"],
        "precision": cfg["precision"],
        "mtp": cfg["mtp"],
        "block": args.block,
        "repeats": args.repeats,
        "subset": args.subset,
        "limit": args.limit,
        "generation": gen,
        "config": cfg,
        "serve_args": config.serve_args(cfg),
        "served_models": served,
        "vllm_version": vllm_version,
        "prompts_sha256": sha256_file(bench["prompts"]),
        "warmup_sha256": sha256_file(bench["warmup_prompts"]),
        "orders": orders,
        "hostname": socket.gethostname(),
        "git_commit": os.environ.get("STUDY_GIT_COMMIT"),
        "git_dirty": os.environ.get("STUDY_GIT_DIRTY"),
        "image_id": os.environ.get("STUDY_IMAGE_ID"),
        "run_id": os.environ.get("STUDY_RUN_ID"),
        "started_at": time.time(),
    }

    host = args.host
    timeout = bench["request_timeout_s"]
    with metrics.Telemetry(out / "telemetry.csv", bench["telemetry_interval_s"]):
        # ---- warm-up (discarded) ------------------------------------------------
        with open(out / "warmup.jsonl", "w") as wf:
            for w in warmups:
                payload = client.build_payload(cfg["served_model_name"], messages_for(w, gen), gen,
                                               max_tokens=w.get("max_tokens"))
                raw = client.stream_chat(host, port, payload, timeout)
                rec = {"id": w["id"], **metrics.request_metrics(raw),
                       "wall_start": raw["wall_start"], "wall_end": raw["wall_end"]}
                wf.write(json.dumps(rec) + "\n")
                print(f"[warmup] {w['id']}: {rec['completion_tokens']} tok, "
                      f"ttft {rec['ttft_s']:.2f}s, {rec['decode_tok_s'] or 0:.2f} tok/s", flush=True)
        meta["warmup_done_at"] = time.time()

        # ---- measured requests ----------------------------------------------------
        n_err = 0
        with open(out / "requests.jsonl", "w") as rf:
            seq = 0
            for rep, order in enumerate(orders):
                for pid in order:
                    p = by_id[pid]
                    payload = client.build_payload(cfg["served_model_name"], messages_for(p, gen), gen,
                                                   max_tokens=p.get("max_tokens"))
                    idle_wait = wait_idle(host, port, 300)
                    before = metrics.parse_prometheus(client.get_text(host, port, "/metrics"))
                    try:
                        raw = client.stream_chat(host, port, payload, timeout)
                    except Exception as e:  # noqa: BLE001 - recorded, then continue
                        n_err += 1
                        # Keep the failure, and make sure the server dropped the request first.
                        drained = wait_idle(host, port, max(600, timeout))
                        rf.write(json.dumps({"condition": cfg["condition"], "block": args.block,
                                             "repeat": rep, "seq": seq, "prompt_id": pid,
                                             "prompt_sha256": prompt_sha256(p),
                                             "max_tokens": payload.get("max_tokens"),
                                             "error": repr(e), "drain_wait_s": drained,
                                             "wall_time": time.time()}) + "\n")
                        rf.flush()
                        print(f"[error] {pid}: {e!r}", flush=True)
                        if n_err >= 3:
                            raise SystemExit("3 request errors; aborting condition")
                        seq += 1
                        continue
                    n_err = 0
                    after = metrics.parse_prometheus(client.get_text(host, port, "/metrics"))
                    delta = metrics.counter_delta(before, after)
                    prompt_text = raw.pop("prompt_text") or ""
                    finished = delta.get("vllm:request_success_total")
                    gen_tokens = delta.get("vllm:generation_tokens_total")
                    # Exactly one request finished and every generated token is ours: no other
                    # client (e.g. a UI on the host network) generated during the measurement.
                    exclusive = finished == 1 and gen_tokens == len(raw["token_ids"])
                    rec = {
                        "condition": cfg["condition"],
                        "precision": cfg["precision"],
                        "mtp": cfg["mtp"],
                        "block": args.block,
                        "repeat": rep,
                        "seq": seq,
                        "prompt_id": pid,
                        "category": p["category"],
                        "subcategory": p.get("subcategory"),
                        "length_bucket": p.get("length_bucket"),
                        **metrics.request_metrics(raw),
                        **metrics.spec_decode_stats(delta, cfg["spec"]["num_speculative_tokens"] if cfg.get("spec") else cfg["mtp"]),
                        "server_prompt_tokens": delta.get("vllm:prompt_tokens_total"),
                        "server_generation_tokens": delta.get("vllm:generation_tokens_total"),
                        "server_preemptions": delta.get("vllm:num_preemptions_total"),
                        "prompt_text_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
                        "prompt_sha256": prompt_sha256(p),
                        "max_tokens": payload.get("max_tokens"),
                        "server_requests_finished": finished,
                        "exclusive_ok": exclusive,
                        "idle_wait_s": idle_wait,
                        **raw,
                    }
                    rf.write(json.dumps(rec) + "\n")
                    rf.flush()
                    sd = rec.get("sd_mean_acceptance_length")
                    print(f"[{cfg['condition']} b{args.block} r{rep} {seq + 1}/{len(order) * args.repeats}] "
                          f"{pid}: in {rec['prompt_tokens']} out {rec['completion_tokens']} "
                          f"ttft {rec['ttft_s']:.2f}s tpot {1000 * (rec['tpot_s'] or 0):.1f}ms "
                          f"{rec['decode_tok_s'] or 0:.2f} tok/s"
                          + (f" MAL {sd:.2f}" if sd else "")
                          + (f" think {rec['think_tokens']} answer@{rec['ttft_answer_s']:.1f}s"
                             if rec.get("thinking") and rec.get("ttft_answer_s") is not None else
                             " NO-ANSWER" if rec.get("thinking") else "")
                          + f" {rec['finish_reason']}" + ("" if exclusive else " NOT-EXCLUSIVE"), flush=True)
                    seq += 1

    meta["finished_at"] = time.time()
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done: {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
