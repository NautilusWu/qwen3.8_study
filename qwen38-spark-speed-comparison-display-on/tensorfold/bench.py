"""Single-stream measurement client for the TensorFold server started by serve.sh.

    python3 tensorfold/bench.py --prompts vllm/prompts/prompts.jsonl --out results/tf_nvfp4

Standard library only. Uses the same prompt file format, system prompt, sampling settings and
prompt order as vllm/benchmark/run.py, so both engines see the same requests in the same order.
Needs the instrumented image (Dockerfile.trace): it streams the committed token ids of every
event, returns the actual prompt token ids, and clears the prompt cache before each request.
None of this changes logits, drafting, sampling or kernels.

Outputs in --out:
    warmup.jsonl     warm-up requests (never used in results)
    requests.jsonl   one line per measured request: timings, token ids, thinking and answer text
    telemetry.csv    GPU temperature / power / utilization and host memory, every 2 s
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import random
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

HOST = "127.0.0.1"
SYSTEM_PROMPT = "You are a helpful assistant."
SEED = 1234
MAX_TOKENS = 24576

# Warm-up, identical to the published runs: a smoke test, kernel / long-output warm-ups, and nine
# 400-token requests (fixed length via ignore_eos), the last one with drafts off.
_CODE_TAIL = ("Output only Python source. No comments, no docstrings, no markdown fences. "
              "Then add tests and the helpers this needs. Keep writing code.")
_CODE = "binary_search\ndef binary_search(nums, target) -> int: index of target in a sorted list, or -1.\n" + _CODE_TAIL
_PROSE = ("Write a detailed step-by-step explanation of how a hash map works, including collision handling, "
          "resizing, and time complexity. Be thorough.")
_WARM = "warmup_noop\ndef warmup_noop(x): return x unchanged.\n" + _CODE_TAIL


def payload(messages, limit, thinking=False, draft=True, fixed=False):
    return {"model": "Qwen3.8-27B", "messages": messages, "max_tokens": limit, "temperature": 0, "top_p": 1,
            "seed": SEED, "n": 1, "stream": True, "stream_options": {"include_usage": True},
            "return_token_ids": True, "chat_template_kwargs": {"enable_thinking": thinking}, "draft": draft,
            "ignore_eos": fixed, "study_trace": True, "study_cold_cache": True}


def warmups():
    user = lambda text: [{"role": "user", "content": text}]  # noqa: E731
    tasks = [("smoke", payload(user("Compute 2+3. Reply exactly FINAL: 5."), 64)),
             ("code_kernel_warmup", payload(user(_WARM), 32, fixed=True)),
             ("code_full_warmup", payload(user(_CODE), 400, fixed=True)),
             ("prose_full_warmup", payload(user(_PROSE), 400, fixed=True))]
    for i in range(5):
        tasks.append((f"short_code_{i}", payload(user(_CODE), 400, fixed=True)))
        if i < 3:
            tasks.append((f"short_prose_{i}", payload(user(_PROSE), 400, fixed=True)))
    tasks.append(("short_code_no_drafts", payload(user(_CODE), 400, draft=False, fixed=True)))
    return tasks


def health(port):
    with urllib.request.urlopen(f"http://{HOST}:{port}/health", timeout=5) as r:
        return json.load(r)


def stream(port, body, timeout):
    """Send one request; return timings, token ids and text. Refuses to overlap with another request."""
    before = health(port)
    if before.get("busy") or before.get("requests_running", 0):
        raise RuntimeError(f"server busy before request: {before}")
    conn = http.client.HTTPConnection(HOST, port, timeout=timeout)
    start = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:2000]!r}")
    events, ids, answer, reasoning = [], [], "", ""
    answer_start, stats, usage, finish = None, {}, {}, None
    try:
        while True:
            line = resp.readline()
            if not line:
                break
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if raw == b"[DONE]":
                break
            if not raw:
                continue
            x = json.loads(raw)
            now = time.perf_counter() - start
            if x.get("error"):
                raise RuntimeError(x["error"])
            stats.update(x.get("tensorfold") or {})
            usage.update(x.get("usage") or {})
            for choice in x.get("choices", []):
                finish = choice.get("finish_reason") or finish
                d = choice.get("delta") or {}
                new = d.get("study_token_ids") or []
                if new:
                    events.append((now, len(new)))
                    ids.extend(new)
                a = d.get("content") or ""
                answer += a
                reasoning += d.get("reasoning_content") or ""
                if a.strip() and answer_start is None:
                    answer_start = now
    finally:
        conn.close()
    total = time.perf_counter() - start
    after = health(port)
    if not events:
        raise RuntimeError("no token ids received; is the instrumented image (Dockerfile.trace) running?")
    if after["requests_total"] - before["requests_total"] != 1:
        raise RuntimeError("another client used the server during the request")
    prompt_ids = stats.pop("study_prompt_token_ids", [])
    stats.pop("token_ids", None)
    n, n_first = len(ids), events[0][1]
    span = events[-1][0] - events[0][0]
    rounds = stats.get("rounds") or 0
    return {
        "prompt_tokens": len(prompt_ids),
        "prompt_token_ids_sha256": hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest(),
        "completion_tokens": n,
        "usage": usage,
        "finish_reason": finish,
        "ttft_s": events[0][0],
        "ttft_answer_s": answer_start,   # first non-whitespace answer text after thinking
        "e2e_s": total,                  # request sent -> stream finished
        # committed tokens after the first token-carrying event / time from that event to the last
        "post_first_event_tok_s": (n - n_first) / span if len(events) > 1 and span > 0 else None,
        "tokens_per_round": n / rounds if rounds else None,
        "drafted_per_round": stats.get("drafted", 0) / rounds if rounds else None,
        "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
        "tensorfold": stats,
        "events": events,
        "token_ids": ids,
        "reasoning": reasoning,
        "answer": answer,
    }


class Telemetry:
    """GPU temperature / power / utilization and host MemAvailable every `interval` s."""

    def __init__(self, path, interval=2.0):
        self.path, self.interval = path, interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=10)

    def _loop(self):
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["wall_time", "temperature_c", "power_w", "utilization_pct", "mem_available_gib"])
            while not self._stop.is_set():
                try:
                    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,utilization.gpu",
                                          "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                         timeout=5).stdout.strip().splitlines()[0]
                    gpu = [v.strip() for v in out.split(",")]
                except Exception:
                    gpu = ["", "", ""]
                with open("/proc/meminfo") as m:
                    avail = next(int(l.split()[1]) for l in m if l.startswith("MemAvailable:"))
                w.writerow([f"{time.time():.3f}", *gpu, f"{avail / 2**20:.2f}"])
                f.flush()
                self._stop.wait(self.interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="JSONL: id, category, messages, optional max_tokens")
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8112)
    ap.add_argument("--block", type=int, default=0, help="prompt-order seed, as in vllm/benchmark/run.py")
    ap.add_argument("--timeout", type=float, default=600, help="per socket read, not per request")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    prompts = [json.loads(l) for l in open(args.prompts) if l.strip()]
    by_id = {p["id"]: p for p in prompts}
    order = [p["id"] for p in prompts]
    random.Random(f"{SEED}-block{args.block}-rep0").shuffle(order)   # same order as run.py
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with Telemetry(out / "telemetry.csv"):
        if not args.no_warmup:
            with open(out / "warmup.jsonl", "w") as f:
                for name, body in warmups():
                    row = stream(args.port, body, args.timeout)
                    if name == "smoke" and "FINAL: 5" not in row["answer"]:
                        raise SystemExit(f"smoke test failed: {row['answer']!r}")
                    f.write(json.dumps({"id": name, **row}) + "\n")
                    print(f"[warmup] {name}: {row['completion_tokens']} tok", flush=True)
        with open(out / "requests.jsonl", "w") as f:
            for i, pid in enumerate(order, 1):
                p = by_id[pid]
                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + p["messages"]
                body = payload(messages, p.get("max_tokens") or MAX_TOKENS, thinking=True)
                row = stream(args.port, body, args.timeout)
                if row["cached_tokens"]:
                    raise SystemExit("prompt cache was reused; the instrumented image is required")
                row.update(prompt_id=pid, category=p.get("category"), max_tokens=body["max_tokens"],
                           wall_time=time.time())
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"[{i}/{len(order)}] {pid}: out {row['completion_tokens']} "
                      f"{row['post_first_event_tok_s'] or 0:.1f} tok/s e2e {row['e2e_s']:.0f}s "
                      f"tok/round {row['tokens_per_round'] or 0:.2f} {row['finish_reason']}", flush=True)
    print(f"done: {out}", flush=True)


if __name__ == "__main__":
    main()
