"""PLAN_16 B0: `tensorfold serve` (0.6.0, inside its image) with a lone stream's tree width set from P16_TREE_ROWS,
then the PLAN_14 observation hook (/p14/serve_hooked.py, mounted read-only) for the per-round trace.

TensorFold 0.6.0 gives one stream on one GB10 `tree_rows=16` (families/qwen3_5/__init__.py cuda_engine). This sets
that one constructor argument before the server starts; nothing on disk changes. Verification is exact (lane
matmuls: a row's bits never depend on how many rows are verified), so the replies must stay token for token the same;
the runner checks that.
"""
import os
import runpy
import sys

ROWS = int(os.environ.get('P16_TREE_ROWS', '0') or 0)

if ROWS:
    from tensorfold.families.qwen3_5.cuda import engine as _engine

    _orig = _engine.Qwen27Engine.__init__

    def _init(self, *a, **k):
        if k.get('tree_rows') is not None:            # only the lone-stream wide path sets it (16)
            k['tree_rows'] = ROWS
        _orig(self, *a, **k)
        print(f'[p16] tree_rows={self.tree_rows} max_rows={self.max_rows}', flush=True)

    _engine.Qwen27Engine.__init__ = _init

sys.argv = ['/p14/serve_hooked.py'] + sys.argv[1:]
runpy.run_path('/p14/serve_hooked.py', run_name='__main__')
