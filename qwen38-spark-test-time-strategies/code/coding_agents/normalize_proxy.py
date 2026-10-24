"""Normalizing proxy between coding agents and the local TensorFold server.

    python3 normalize_proxy.py --listen 8099 --upstream 127.0.0.1:8091 --log run/requests.jsonl

Every agent sends its requests here instead of to TensorFold, so all agents get the same generation
settings and the comparison isolates the harness (prompts, tools, loop, context handling):

- thinking: on, at the chat template's default effort (xhigh); any agent-chosen effort or switch is removed;
- output limit: MAX_OUTPUT tokens for every request (chat: max_tokens; responses: max_output_tokens);
- sampling: the server's defaults (greedy); agent-chosen temperature/top_p/top_k/seed are removed;
- model name: rewritten to the served model;
- compatibility: in Responses API requests, the `include` field (encrypted reasoning) and non-function
  tools (hosted web search) are dropped; both exist only on OpenAI's cloud;
- repair: a tool call in the history whose arguments are not a JSON object is closed (or, failing that,
  replaced by {}). TensorFold streams a call's arguments while the model writes it and does not close
  them when the model leaves the call unfinished; agents that send such a call back would otherwise get
  a 400 ("Can only get item pairs from a mapping") from the chat template and lose the session.

Each request is logged as one JSON line with what was removed, so the agents' own defaults are on record.
Standard library only; streams responses through unchanged.
"""
import argparse
import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = 'qwen3.8-27b-tensorfold-nvfp4'
MAX_OUTPUT = 32768
DROP = ('reasoning_effort', 'reasoning', 'thinking', 'temperature', 'top_p', 'top_k', 'min_p', 'seed',
        'max_tokens', 'max_completion_tokens', 'max_output_tokens')
HOP = {'connection', 'keep-alive', 'transfer-encoding', 'content-length', 'te', 'trailer', 'upgrade',
       'proxy-authorization', 'proxy-authenticate', 'accept-encoding', 'host'}


def repair_arguments(args):
    """The arguments as a JSON-object string, or None if they already are one."""
    if not isinstance(args, str):
        return None
    for candidate in (args, args + '}', args + '"}'):
        try:
            if isinstance(json.loads(candidate), dict):
                return None if candidate is args else candidate
        except ValueError:
            pass
    return '{}'


def repair_history(body):
    repaired = []
    items = body.get('input') if isinstance(body.get('input'), list) else []
    for item in items:
        if isinstance(item, dict) and item.get('type') == 'function_call':
            fixed = repair_arguments(item.get('arguments'))
            if fixed is not None:
                repaired.append({'call_id': item.get('call_id'), 'before': item['arguments'], 'after': fixed})
                item['arguments'] = fixed
    for message in body.get('messages') or []:
        for call in (message.get('tool_calls') or []) if isinstance(message, dict) else []:
            fn = call.get('function') if isinstance(call, dict) else None
            fixed = repair_arguments(fn.get('arguments')) if isinstance(fn, dict) else None
            if fixed is not None:
                repaired.append({'call_id': call.get('id'), 'before': fn['arguments'], 'after': fixed})
                fn['arguments'] = fixed
    return repaired


def normalize(path, body):
    removed = {k: body.pop(k) for k in DROP if k in body}
    kwargs = body.get('chat_template_kwargs')
    if isinstance(kwargs, dict):
        for k in ('enable_thinking', 'reasoning_effort', 'thinking'):
            if k in kwargs:
                removed[f'chat_template_kwargs.{k}'] = kwargs.pop(k)
        if not kwargs:
            body.pop('chat_template_kwargs')
    if 'model' in body:
        removed['model'] = body['model']
        body['model'] = MODEL
    if path.endswith('/responses'):
        body['max_output_tokens'] = MAX_OUTPUT
        if 'include' in body:  # e.g. Codex asks for reasoning.encrypted_content, an OpenAI-cloud feature TensorFold lacks
            removed['include'] = body.pop('include')
        hosted = [t for t in body.get('tools') or [] if t.get('type') != 'function']
        if hosted:  # e.g. Codex's hosted web_search: runs only on OpenAI's servers, TensorFold runs function tools only
            removed['hosted_tools'] = [t.get('type') for t in hosted]
            body['tools'] = [t for t in body['tools'] if t.get('type') == 'function']
    elif path.endswith('/chat/completions') or path.endswith('/completions'):
        body['max_tokens'] = MAX_OUTPUT
    repaired = repair_history(body)
    if repaired:
        removed['repaired_tool_arguments'] = repaired
    return removed


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def _forward(self, method):
        t0 = time.time()
        length = int(self.headers.get('Content-Length') or 0)
        raw = self.rfile.read(length) if length else b''
        entry = {'t_start': t0, 'method': method, 'path': self.path}
        if raw and 'json' in (self.headers.get('Content-Type') or ''):
            try:
                body = json.loads(raw)
                entry['removed'] = normalize(self.path, body)
                entry['stream'] = bool(body.get('stream'))
                entry['n_messages'] = len(body.get('messages') or body.get('input') or [])
                entry['n_tools'] = len(body.get('tools') or [])
                raw = json.dumps(body).encode()
            except ValueError:
                entry['note'] = 'non-json body passed through'
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        headers['Content-Length'] = str(len(raw))
        conn = http.client.HTTPConnection(self.server.upstream_host, self.server.upstream_port, timeout=3600)
        try:
            conn.request(method, self.path, body=raw if method == 'POST' else None, headers=headers)
            resp = conn.getresponse()
            entry['status'] = resp.status
            if resp.status >= 400 and raw:  # keep the rejected request for diagnosis
                dump = self.server.log_path + f'.rejected-{int(t0 * 1000)}.json'
                with open(dump, 'wb') as f:
                    f.write(raw)
                entry['rejected_body'] = dump
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            self.send_header('Transfer-Encoding', 'chunked')
            self.send_header('Connection', 'close')
            self.end_headers()
            n = 0
            while True:
                chunk = resp.readline()
                if not chunk:
                    break
                n += len(chunk)
                self.wfile.write(b'%x\r\n%s\r\n' % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b'0\r\n\r\n')
            self.wfile.flush()
            entry['bytes'] = n
        except Exception as exc:  # noqa: BLE001 - recorded, and the agent sees a broken response
            entry['error'] = repr(exc)
        finally:
            conn.close()
            entry['t_end'] = time.time()
            with self.server.lock, open(self.server.log_path, 'a') as f:
                f.write(json.dumps(entry, default=str) + '\n')
        self.close_connection = True

    def do_POST(self):
        self._forward('POST')

    def do_GET(self):
        self._forward('GET')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--listen', type=int, default=8099)
    ap.add_argument('--upstream', default='127.0.0.1:8091')
    ap.add_argument('--log', required=True)
    a = ap.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', a.listen), Handler)
    server.upstream_host, port = a.upstream.split(':')
    server.upstream_port = int(port)
    server.log_path = a.log
    server.lock = threading.Lock()
    server.serve_forever()


if __name__ == '__main__':
    main()
