"""Log host memory every 2 s and, every 20 s, how much of the checkpoint's n-gram tables is in the page cache
(Flash Next maps them from the checkpoint files instead of loading them).

    python3 memlog.py <checkpoint folder on the host> <out.jsonl>        (runs until stopped)
"""
import ctypes
import json
import os
import sys
import time
from pathlib import Path

PAGE = os.sysconf('SC_PAGE_SIZE')
libc = ctypes.CDLL('libc.so.6', use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]


def ngram_shards(model_dir):
    """Shards holding only n-gram tensors, from the checkpoint's index."""
    index = json.loads((model_dir / 'model.safetensors.index.json').read_text())['weight_map']
    kinds = {}
    for name, shard in index.items():
        kinds.setdefault(shard, set()).add('ngram' in name)
    return [(model_dir / s).resolve() for s, k in sorted(kinds.items()) if k == {True}]


def resident_bytes(path):
    size = path.stat().st_size
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, 1, 1, fd, 0)          # PROT_READ, MAP_SHARED: maps without reading
        if addr in (None, ctypes.c_void_p(-1).value):
            return None
        try:
            pages = (size + PAGE - 1) // PAGE
            vec = (ctypes.c_ubyte * pages)()
            if libc.mincore(addr, size, vec) != 0:
                return None
            return (pages - bytes(vec).count(0)) * PAGE
        finally:
            libc.munmap(addr, size)
    finally:
        os.close(fd)


def main():
    paths = ngram_shards(Path(sys.argv[1]))
    out = open(sys.argv[2], 'a')
    last = 0
    while True:
        m = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            k, v = line.split(':')
            m[k] = int(v.split()[0]) * 1024
        rec = {'t': round(time.time(), 1), 'avail_gib': round(m['MemAvailable'] / 2**30, 2),
               'cached_gib': round(m['Cached'] / 2**30, 2)}
        if time.time() - last >= 20:
            last = time.time()
            r = [resident_bytes(p) for p in paths]
            if None not in r:
                rec['ngram_resident_gib'] = round(sum(r) / 2**30, 2)
        out.write(json.dumps(rec) + '\n')
        out.flush()
        time.sleep(2)


if __name__ == '__main__':
    main()
