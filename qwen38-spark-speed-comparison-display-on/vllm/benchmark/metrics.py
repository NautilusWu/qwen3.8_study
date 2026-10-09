"""Request-level metric derivation, Prometheus scraping and hardware telemetry."""

from __future__ import annotations

import csv
import re
import subprocess
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Prometheus /metrics
# ---------------------------------------------------------------------------

_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")


def parse_prometheus(text: str) -> dict[str, float]:
    """Return {'name{labels}': value} for vLLM counters/gauges (histogram buckets dropped).

    Labels identifying the engine/model are stripped so keys are stable across servers;
    only the `position` label (spec-decode per-position counters) is kept.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        m = _SAMPLE.match(line)
        if not m:
            continue
        name, labels, value = m.groups()
        if name.endswith("_bucket") or name.endswith("_created"):
            continue
        pos = re.search(r'position="(\d+)"', labels or "")
        key = f"{name}[{pos.group(1)}]" if pos else name
        try:
            out[key] = out.get(key, 0.0) + float(value)
        except ValueError:
            pass
    return out


def counter_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {
        k: after[k] - before.get(k, 0.0)
        for k in after
        if k.endswith(("_total", "_sum", "_count")) or "_total[" in k
    }


def spec_decode_stats(delta: dict[str, float], mtp: int) -> dict:
    """Spec-decode statistics for one request from /metrics counter deltas."""
    drafts = delta.get("vllm:spec_decode_num_drafts_total", 0.0)
    drafted = delta.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    accepted = delta.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    out = {
        "sd_drafts": drafts,
        "sd_draft_tokens": drafted,
        "sd_accepted_tokens": accepted,
        "sd_acceptance_rate": accepted / drafted if drafted else None,
        # Mean acceptance length = tokens emitted per verification step (1 bonus/corrected + accepted).
        "sd_mean_acceptance_length": 1.0 + accepted / drafts if drafts else None,
    }
    for p in range(mtp):
        acc_p = delta.get(f"vllm:spec_decode_num_accepted_tokens_per_pos_total[{p}]", 0.0)
        out[f"sd_accept_pos{p + 1}"] = acc_p / drafts if drafts else None
    return out


# ---------------------------------------------------------------------------
# Client-side request metrics
# ---------------------------------------------------------------------------

def request_metrics(raw: dict) -> dict:
    n_out = len(raw["token_ids"])
    ttft = raw["t_first_token"]
    decode_time = raw["t_last_token"] - raw["t_first_token"]
    e2e = raw["t_done"]
    usage = raw.get("usage") or {}
    n_prompt = usage.get("prompt_tokens")
    decode_chunks = len(raw["chunk_n"]) - 1
    return {
        "prompt_tokens": n_prompt,
        "completion_tokens": usage.get("completion_tokens"),
        "n_token_ids": n_out,
        "ttft_s": ttft,
        "e2e_s": e2e,
        "decode_time_s": decode_time,
        # TPOT excludes the first token (produced by prefill).
        "tpot_s": decode_time / (n_out - 1) if n_out > 1 else None,
        "decode_tok_s": (n_out - 1) / decode_time if n_out > 1 and decode_time > 0 else None,
        "e2e_tok_s": n_out / e2e if e2e > 0 else None,
        "prefill_tok_s": n_prompt / ttft if n_prompt and ttft > 0 else None,
        # Client-side tokens per streamed chunk after the first (≈ tokens per engine step).
        "client_tokens_per_step": (n_out - raw["chunk_n"][0]) / decode_chunks if decode_chunks > 0 else None,
        "max_inter_chunk_s": max((b - a for a, b in zip(raw["chunk_t"], raw["chunk_t"][1:])), default=None),
        **single_stream_metrics(raw),
    }


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def single_stream_metrics(raw: dict) -> dict:
    """Single-stream metrics; fields missing in older records -> None.

    post_first_event_tok_s counts only tokens that arrived after the first token-carrying
    event, so a multi-token first event cannot inflate the rate the way
    (n - 1) / (t_last - t_first) would. Both are client-side arrival rates, not GPU decode
    throughput.
    """
    n_out = len(raw["token_ids"])
    n_first = raw["chunk_n"][0]
    span = raw["t_last_token"] - raw["t_first_token"]
    gaps = [b - a for a, b in zip(raw["chunk_t"], raw["chunk_t"][1:])]
    n_think = raw.get("n_think_tokens")
    t_answer = raw.get("t_answer")
    return {
        "n_first_event_tokens": n_first,
        "post_first_event_tok_s": (n_out - n_first) / span if len(raw["chunk_n"]) > 1 and span > 0 else None,
        "ttft_answer_s": t_answer,
        "think_tokens": n_think,
        "answer_tokens": n_out - n_think if n_think is not None else None,
        "gap_p50_s": _pct(gaps, 0.50),
        "gap_p95_s": _pct(gaps, 0.95),
        "gap_p99_s": _pct(gaps, 0.99),
        "n_gaps_over_1s": sum(g > 1.0 for g in gaps),
    }


# ---------------------------------------------------------------------------
# Hardware telemetry (GPU power/clock/temperature via nvidia-smi, unified memory via /proc/meminfo)
# ---------------------------------------------------------------------------

def _meminfo() -> dict[str, int]:
    vals = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            vals[k] = int(v.split()[0])
    return vals


class Telemetry:
    """Samples GPU and memory state every `interval` seconds into a CSV file."""

    FIELDS = ["wall_time", "power_w", "temp_c", "sm_clock_mhz", "gpu_util_pct",
              "mem_total_kb", "mem_available_kb", "swap_used_kb", "cpu_busy_pct", "loadavg_1m"]

    def __init__(self, path: Path, interval: float):
        self.path = path
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=10)

    @staticmethod
    def _gpu() -> list[str]:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,clocks.sm,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip().splitlines()[0]
            return [v.strip() for v in out.split(",")]
        except Exception:
            return ["", "", "", ""]

    @staticmethod
    def _cpu_times() -> tuple[int, int]:
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
        idle = v[3] + v[4]  # idle + iowait
        return sum(v) - idle, sum(v)

    def _loop(self):
        # Whole-machine CPU load and load average. In our runs launches were 12-22%
        # slower while the desktop was in use (likely shared LPDDR5x bandwidth / GPU
        # compositing), and nothing in the telemetry showed it; this makes it visible.
        prev = self._cpu_times()
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.FIELDS)
            while not self._stop.is_set():
                t = time.time()
                gpu = self._gpu()
                m = _meminfo()
                cur = self._cpu_times()
                busy = 100.0 * (cur[0] - prev[0]) / max(1, cur[1] - prev[1])
                prev = cur
                with open("/proc/loadavg") as la:
                    load1 = la.read().split()[0]
                w.writerow([f"{t:.3f}", *gpu, m["MemTotal"], m["MemAvailable"],
                            m["SwapTotal"] - m["SwapFree"], f"{busy:.1f}", load1])
                f.flush()
                self._stop.wait(max(0.0, self.interval - (time.time() - t)))
