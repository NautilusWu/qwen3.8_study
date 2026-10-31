"""Run Part 4's runner (code/strategies/run_method.py) with a different target checkpoint, nothing else changed.

    python3 run_with_target.py <part4>/code/strategies <target path inside the /hf mount> <run_method arguments...>

Example (the fine-tune with the NVFP4 lm_head, prepared as in REPRODUCE.md, stored under $HF_DIR/extra/):
    python3 run_with_target.py ../../qwen38-spark-test-time-strategies/code/strategies /hf/extra/coder390-nvfp4head \
        D2 ft-D2 --split confirm --limit-min 40 --context 81920 --max-tokens 65536
"""
import runpy
import sys
from pathlib import Path

strategies, target, *rest = sys.argv[1:]
strategies = str(Path(strategies).resolve())
sys.path.insert(0, strategies)
import common  # noqa: E402  (Part 4's module; run_method imports this same object)

common.TARGET = target
sys.argv = [str(Path(strategies) / 'run_method.py')] + rest
runpy.run_path(sys.argv[0], run_name='__main__')
