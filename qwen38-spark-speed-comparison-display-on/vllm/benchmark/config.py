"""Condition config loading and `vllm serve` argument rendering.

Usage (inside the runtime image):
    python -m benchmark.config serve-args configs/nvidia_nvfp4_mtp3.yaml   # NUL-separated argv
    python -m benchmark.config show configs/nvidia_nvfp4_mtp3.yaml         # merged config as JSON
    python -m benchmark.config serve-env configs/nvidia_nvfp4_mtp3.yaml    # NUL-separated KEY=VALUE for the server

A common file may itself name a `base:` file (path relative to it); it is
deep-merged on top of that base. Experiments use this to change one server
setting without copying, or touching, configs/common.yaml.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

# Keys a condition file may set. Everything else must come from common.yaml,
# so two conditions can never differ in anything but precision and MTP depth.
CONDITION_KEYS = {"inherit", "condition", "precision", "mtp", "spec"}
# `spec` (optional): a full vLLM --speculative-config for draft-model methods such as DFlash2;
# a condition sets either `spec` or an MTP depth, never both.


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load_common(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    if "base" in data:
        data = _deep_merge(_load_common(path.parent / data.pop("base")), data)
    return data


def load(path: str | Path) -> dict:
    path = Path(path)
    cond = yaml.safe_load(path.read_text())
    extra = set(cond) - CONDITION_KEYS
    if extra:
        raise ValueError(f"{path}: condition files may only set {sorted(CONDITION_KEYS)}; got {sorted(extra)}")
    base = _load_common(path.parent / cond["inherit"])
    cfg = dict(base)
    cfg.update({k: v for k, v in cond.items() if k != "inherit"})
    if cfg["precision"] not in cfg["models"]:
        raise ValueError(f"unknown precision {cfg['precision']!r}")
    if not (isinstance(cfg["mtp"], int) and 0 <= cfg["mtp"] <= 8):
        raise ValueError(f"bad mtp depth {cfg['mtp']!r}")
    if cfg.get("spec") and cfg["mtp"] != 0:
        raise ValueError("set either spec or an mtp depth, not both")
    cfg["model"] = cfg["models"][cfg["precision"]]
    cfg["served_model_name"] = cfg["condition"]
    return cfg


def serve_args(cfg: dict) -> list[str]:
    s = cfg["server"]
    args = [
        "vllm", "serve", cfg["model"]["repo"],
        "--revision", cfg["model"]["revision"],
        "--tokenizer", cfg["tokenizer"]["repo"],
        "--tokenizer-revision", cfg["tokenizer"]["revision"],
        "--chat-template", cfg["chat_template"],
        "--served-model-name", cfg["served_model_name"],
        "--host", "127.0.0.1",
        "--port", str(s["port"]),
        "--dtype", s["dtype"],
        "--max-model-len", str(s["max_model_len"]),
        "--kv-cache-dtype", s["kv_cache_dtype"],
        "--gpu-memory-utilization", str(s["gpu_memory_utilization"]),
        "--max-num-seqs", str(s["max_num_seqs"]),
        "--max-num-batched-tokens", str(s["max_num_batched_tokens"]),
        "--enable-chunked-prefill" if s["enable_chunked_prefill"] else "--no-enable-chunked-prefill",
        "--enable-prefix-caching" if s["enable_prefix_caching"] else "--no-enable-prefix-caching",
        "--generation-config", s["generation_config"],
    ]
    if s["language_model_only"]:
        args.append("--language-model-only")
    if cfg.get("spec"):
        args += ["--speculative-config", json.dumps(cfg["spec"], separators=(",", ":"))]
    elif cfg["mtp"] > 0:
        spec = {"method": s["mtp_method"], "num_speculative_tokens": cfg["mtp"]}
        args += ["--speculative-config", json.dumps(spec, separators=(",", ":"))]
    args += [str(a) for a in s["extra_args"]]
    return args


def main(argv: list[str]) -> None:
    cmd, path = argv[1], argv[2]
    cfg = load(path)
    if cmd == "serve-args":
        sys.stdout.write("\0".join(serve_args(cfg)))
    elif cmd == "serve-env":
        sys.stdout.write("\0".join(f"{k}={v}" for k, v in (cfg["server"].get("env") or {}).items()))
    elif cmd == "show":
        print(json.dumps(cfg, indent=2))
    elif cmd == "get":
        # Dotted lookup, e.g. `get configs/x.yaml server.port`
        v = cfg
        for part in argv[3].split("."):
            v = v[part]
        print(v)
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main(sys.argv)
