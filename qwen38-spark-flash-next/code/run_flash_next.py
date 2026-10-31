"""Run Part 4's runner (code/strategies/run_method.py) with Qwen3.8-Flash-Next as the model (see flash_next.py).

    python3 run_flash_next.py <part4>/code/strategies <run_method arguments...>

Example (the settings of this study's three configurations, on your own TASKS_DIR/confirm.jsonl):
    python3 run_flash_next.py ../../qwen38-spark-test-time-strategies/code/strategies \
        D2 fn-D2 --split confirm --limit-min 40 --context 81920 --max-tokens 65536
Results go to <part4>/code/strategies/results/<run_id>/, as in Part 4.
"""
import runpy
import sys
from pathlib import Path

import flash_next

strategies, *rest = sys.argv[1:]
flash_next.patch(strategies)
sys.argv = [str(Path(strategies).resolve() / 'run_method.py')] + rest
runpy.run_path(sys.argv[0], run_name='__main__')
