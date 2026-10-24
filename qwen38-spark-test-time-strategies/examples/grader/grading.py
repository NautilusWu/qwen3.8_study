"""Example grader for the example tasks (examples/tasks/). Replace with your own.

grade(grader, text, thinking) -> (score, passed, detail). With thinking=True, `text` is the thinking, then </think>,
then the answer; only the answer is graded (no </think>: the model never answered). Grader types:
  numeric       {"answer": float, "tol": float}         number after the last "Final answer:"
  python_tests  {"tests": str, "timeout": s}            the longest ```python block + the tests, run with python3
  cpp_tests     {"tests": str, "timeout": s}            the longest ```cpp block + tests inside main(), with CHECK(...)
  exact_output  {"expected": str}                       last code block equals the expected lines
  python_check  {"check": str}                          source defining check(answer) -> (ok, detail)
"""
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def answer_part(text, thinking):
    if not thinking:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return text.split("</think>", 1)[1].strip() if "</think>" in text else None


def blocks(text):
    return re.findall(r"```([A-Za-z0-9+#]*)[^\n]*\n(.*?)```", text, flags=re.S)


def code(text, langs):
    found = [b for lang, b in blocks(text) if lang.lower() in langs] or [b for _, b in blocks(text)]
    return max(found, key=len) if found else None


def final_line(text):
    hits = [m.end() for m in re.finditer(r"Final answer:", text, flags=re.I)]
    return text[hits[-1]:].strip().splitlines()[0] if hits and text[hits[-1]:].strip() else None


def run(cmd, cwd, timeout):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, (r.stdout + r.stderr)[-500:]
    except subprocess.TimeoutExpired:
        return False, "timeout"


CPP = "#include <bits/stdc++.h>\n{code}\nstatic int fails = 0;\n#define CHECK(c) do {{ if (!(c)) {{ std::printf(\"FAIL %d\\n\", __LINE__); fails++; }} }} while (0)\nint main() {{\n{tests}\nreturn fails ? 1 : 0;\n}}\n"


def grade(grader, text, thinking=False):
    ans = answer_part(text, thinking)
    if ans is None:
        return 0.0, False, "no answer: thinking did not finish"
    kind = grader["type"]
    if kind == "numeric":
        line = final_line(ans)
        m = re.search(r"-?\d+(?:\.\d+)?", (line or "").replace(",", ""))
        if not m:
            return 0.0, False, f"unparsed: {line!r}"
        ok = math.isclose(float(m.group(0)), grader["answer"], abs_tol=max(grader.get("tol", 0), 1e-9))
        return float(ok), ok, m.group(0)
    if kind in ("python_tests", "cpp_tests"):
        src = code(ans, ("python", "py") if kind == "python_tests" else ("cpp", "c++", "cc"))
        if src is None:
            return 0.0, False, "no code block"
        with tempfile.TemporaryDirectory() as d:
            if kind == "python_tests":
                Path(d, "t.py").write_text(src + "\n\n" + grader["tests"] + "\n")
                ok, out = run([sys.executable, "t.py"], d, grader.get("timeout", 20))
            else:
                Path(d, "t.cpp").write_text(CPP.format(code=src, tests=grader["tests"]))
                ok, out = run(["g++", "-std=c++17", "-O1", "-o", "t", "t.cpp"], d, 60)
                if ok:
                    ok, out = run(["./t"], d, grader.get("timeout", 20))
        return float(ok), ok, out
    if kind == "exact_output":
        bl = blocks(ans)
        got = [l.rstrip() for l in bl[-1][1].strip("\n").splitlines()] if bl else None
        ok = got == [l.rstrip() for l in grader["expected"].strip("\n").splitlines()]
        return float(ok), ok, "" if ok else f"got {got!r}"
    if kind == "python_check":
        ns = {}
        exec(grader["check"], ns)
        ok, detail = ns["check"](ans)
        return float(bool(ok)), bool(ok), str(detail)
    raise ValueError(f"unknown grader type {kind}")
