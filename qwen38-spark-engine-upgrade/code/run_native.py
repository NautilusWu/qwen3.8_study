"""Run Part 4's runner (code/strategies/run_method.py) with TensorFold 1.2.5 as the server (see native_engine.py).

    python3 run_native.py <part4>/code/strategies <run_method arguments...>

Example (this study's configuration, CL4, on your own TASKS_DIR/confirm.jsonl):
    python3 run_native.py ../../qwen38-spark-test-time-strategies/code/strategies \
        CL tf125-CL4 --k 4 --split confirm --limit-min 40 --context 81920 --max-tokens 65536
Results go to <part4>/code/strategies/results/<run_id>/, as in Part 4. For the paired comparison with 0.6.0 in
alternating blocks, use run_engines.py.
"""
import runpy
import sys
from pathlib import Path

import native_engine

strategies, *rest = sys.argv[1:]
native_engine.patch(strategies, serve_native=True)
sys.argv = [str(Path(strategies).resolve() / 'run_method.py')] + rest
runpy.run_path(sys.argv[0], run_name='__main__')
