"""Minimal, dependency-free streaming client for vLLM's OpenAI chat endpoint.

Concurrency is 1 by design, so a blocking http.client connection is enough and
keeps the timing path free of event-loop scheduling noise. Every SSE chunk is
timestamped on arrival with time.perf_counter().

With `return_token_ids`, vLLM attaches the token ids emitted in each chunk.
Under MTP one engine step can emit several tokens, so the per-chunk token
counts give a client-side view of tokens-per-step that is cross-checked
against the server's spec-decode counters.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import time
import urllib.request


class RequestError(RuntimeError):
    pass


# `</think>` in the official Qwen3.8 tokenizer (used for every checkpoint, see common.yaml).
# It is an added, non-special token, so it is also present in the streamed text.
THINK_END_ID = 248069
THINK_END_TEXT = "</think>"


def get_json(host: str, port: int, path: str, timeout: float = 10.0):
    with urllib.request.urlopen(f"http://{host}:{port}{path}", timeout=timeout) as r:
        return json.loads(r.read())


def get_text(host: str, port: int, path: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(f"http://{host}:{port}{path}", timeout=timeout) as r:
        return r.read().decode()


def build_payload(model: str, messages: list[dict], gen: dict, max_tokens: int | None = None) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": gen["temperature"],
        "seed": gen["seed"],
        "chat_template_kwargs": {"enable_thinking": gen["enable_thinking"]},
        "return_token_ids": True,
        "return_prompt_text": True,
    }
    if gen.get("reasoning_effort"):  # unset = the chat template default
        payload["chat_template_kwargs"]["reasoning_effort"] = gen["reasoning_effort"]
    limit = max_tokens or gen["max_tokens"]
    if limit:  # None/0: no cap, generation stops at EOS or the context limit (length checks only)
        payload["max_tokens"] = limit
    return payload


def stream_chat(host: str, port: int, payload: dict, timeout: float) -> dict:
    """Send one streaming chat request and return raw timing + token data.

    With thinking on, the chat template ends the prompt with `<think>\\n`, so the output is
    `<thinking> </think> <answer>`. The answer boundary is located twice: by token id
    (n_think_tokens = tokens up to and including `</think>`) and in the streamed text
    (t_answer = first event whose text after `</think>` is not whitespace). With thinking
    off, the template pre-fills an empty think block and every output token is answer.
    """
    thinking = bool((payload.get("chat_template_kwargs") or {}).get("enable_thinking"))
    body = json.dumps(payload).encode()
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    wall_start = time.time()
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RequestError(f"HTTP {resp.status}: {resp.read()[:2000]!r}")

    chunk_t: list[float] = []      # arrival time (s since send) of every chunk carrying tokens
    chunk_n: list[int] = []        # tokens in that chunk
    token_ids: list[int] = []
    text_parts: list[str] = []
    prompt_ids_sha = None
    n_prompt_ids = None
    prompt_text = None
    usage = None
    finish_reason = None
    t_done = None
    n_think = None if thinking else 0     # tokens up to and including </think>
    t_think_end = None                    # arrival of the event carrying </think>
    t_answer = None                       # first event with non-whitespace answer text
    answer_seen_text = not thinking       # whether </think> has appeared in the text yet
    n_events = 0                          # SSE data events (incl. role / usage / empty)

    while True:
        line = resp.readline()
        if not line:
            break
        now = time.perf_counter() - t0
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            t_done = now
            break
        obj = json.loads(data)
        n_events += 1
        if obj.get("prompt_token_ids") is not None:
            ids = obj["prompt_token_ids"]
            n_prompt_ids = len(ids)
            prompt_ids_sha = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        if obj.get("prompt_text") is not None:
            prompt_text = obj["prompt_text"]
        if obj.get("usage"):
            usage = obj["usage"]
        for ch in obj.get("choices") or []:
            ids = ch.get("token_ids") or []
            if ids:
                if n_think is None and THINK_END_ID in ids:
                    n_think = len(token_ids) + ids.index(THINK_END_ID) + 1
                    t_think_end = now
                chunk_t.append(now)
                chunk_n.append(len(ids))
                token_ids.extend(ids)
            content = (ch.get("delta") or {}).get("content")
            if content:
                if answer_seen_text:
                    answer_part = content
                elif THINK_END_TEXT in content:
                    answer_seen_text = True
                    answer_part = content.split(THINK_END_TEXT, 1)[1]
                else:
                    answer_part = ""
                if t_answer is None and answer_part.strip():
                    t_answer = now
                text_parts.append(content)
            if ch.get("finish_reason"):
                finish_reason = ch["finish_reason"]
    conn.close()
    wall_end = time.time()
    if t_done is None:
        raise RequestError("stream ended without [DONE]")
    if not chunk_t:
        raise RequestError("no tokens received")

    return {
        "wall_start": wall_start,
        "wall_end": wall_end,
        "t_first_token": chunk_t[0],
        "t_last_token": chunk_t[-1],
        "t_done": t_done,
        "chunk_t": [round(t, 5) for t in chunk_t],
        "chunk_n": chunk_n,
        "token_ids": token_ids,
        "text": "".join(text_parts),
        "finish_reason": finish_reason,
        "usage": usage,
        "thinking": thinking,
        "n_think_tokens": n_think,
        "t_think_end": t_think_end,
        "t_answer": t_answer,
        "think_end_in_text": answer_seen_text if thinking else None,
        "n_events": n_events,
        "prompt_token_ids_sha256": prompt_ids_sha,
        "n_prompt_token_ids": n_prompt_ids,
        "prompt_text": prompt_text,
    }
