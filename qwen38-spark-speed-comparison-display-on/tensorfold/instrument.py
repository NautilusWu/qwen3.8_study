"""Observational SSE token/prompt accounting, plus requested cold-cache isolation.

Does not change logits, drafting, sampling, token stopping, or CUDA kernels.
Only active for requests carrying study_trace / study_cold_cache.
"""
from pathlib import Path
import tensorfold.cuda.server

p = Path(tensorfold.cuda.server.__file__)
s = p.read_text()
old = '        prompt, max_tokens = prepared.prompt, prepared.max_tokens\n'
new = old + '''        if body.get("study_cold_cache"):
            if getattr(self.engine, "concurrent", False):
                raise ValueError("study cold cache requires one stream")
            self.engine.cache.entries.clear()
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '                if delta and not emit(delta):\n'
new = '''                if body.get("study_trace"):
                    delta["study_token_ids"] = [int(t) for t in new]
''' + old
assert s.count(old) == 1
s = s.replace(old, new)
old = '        stats = {**(stats or {}), "token_sha": token_sha(out)}\n'
new = old + '''        if body.get("study_trace"):
            stats["study_prompt_token_ids"] = [int(t) for t in prompt]
'''
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s)
print('Installed observational accounting and explicitly requested cold-cache adapter')
