"""Check downloaded files against data/downloads.json (size and sha256).

    python3 scripts/check_downloads.py <root>

expects each repository under <root>/<repository id>/, as `hf download <id> --local-dir <root>/<id>` puts it, for
example <root>/unsloth/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/...gguf. Files you did not download are listed as missing;
the exit code is 1 if any present file has the wrong size or sha256.
"""
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
d = json.loads((Path(__file__).resolve().parent.parent / 'data' / 'downloads.json').read_text())
entries = d['gguf'] + [d['mtp_head'], d['chat_template']]
bad = 0
for e in entries:
    p = root / e['repo'] / e['path']
    if not p.exists():
        print(f'missing  {e["repo"]}/{e["path"]}')
        continue
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    ok = p.stat().st_size == e['size'] and h.hexdigest() == e['sha256']
    bad += not ok
    print(f'{"ok      " if ok else "MISMATCH"} {e["repo"]}/{e["path"]}')
sys.exit(1 if bad else 0)
