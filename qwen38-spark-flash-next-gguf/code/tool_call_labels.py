"""The automatic labels of PLAN_18's tool-call check (data/tool_calls.csv), as the locked code applied them.

The check replayed 32 recorded turns of a coding agent (OpenCode, on our own private agent tasks) against each server,
exactly as recorded, with the output capped at 8,192 tokens; three replies per turn: greedy, and sampled with the
model's recommended thinking settings (temperature 1.0, top_p 0.95, top_k 20) with seeds 1 and 2. The recorded turns
contain the private tasks, so the replay itself is not published; this is the labelling rule.

    label(reply, tools, finish) -> 'cut' | 'valid' | 'malformed' | 'text'

  reply   {'answer': str, 'reasoning': str, 'tool_calls': [{'name': str, 'arguments': str}, ...]} as the server
          returned it (llama.cpp and TensorFold both parse tool calls into OpenAI-style tool_calls)
  tools   the request's tool definitions (OpenAI format)
  finish  the reply's finish reason
"""
import json

MARKUP = ('<tool_call', '<function=')


def label(reply, tools, finish):
    if finish == 'length':
        return 'cut'                       # reached the 8,192-token limit (checked first)
    schema = {t['function']['name']: t['function'].get('parameters') or {} for t in tools}
    calls = reply['tool_calls']
    if calls:
        for c in calls:
            if c['name'] not in schema:
                return 'malformed'         # a tool that is not in the request
            try:
                args = json.loads(c['arguments'] or '{}')
            except ValueError:
                return 'malformed'         # arguments are not JSON
            if not isinstance(args, dict) or any(k not in args for k in schema[c['name']].get('required') or []):
                return 'malformed'         # not an object, or a required parameter is missing
        return 'valid'
    if any(m in (reply['answer'] or '') or m in (reply['reasoning'] or '') for m in MARKUP):
        return 'malformed'                 # tool-call markup the server could not parse
    return 'text'                          # no tool call, normal stop
