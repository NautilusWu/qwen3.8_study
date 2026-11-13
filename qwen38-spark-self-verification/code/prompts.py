"""The prompts of the test writer T and the critic K, and the C++ wrapper of the model's tests, exactly as they ran
(copied from the run code). Every request has the system message "You are a helpful assistant." The four code branches
get the task itself, exactly as Part 4's vote (CL4) does.

    python3 code/prompts.py      print the two hashes and check run 3-4's against the one recorded in its runs

{task} is the task text without its last sentence about the answer format ("Return only the complete function ..."
or "Do not write a main function. Return only the code ..."), which RETURN_TAIL removes. {fmt} is the format
paragraph for the task's language.
"""
import hashlib
import json
import re

SYSTEM = 'You are a helpful assistant.'
RETURN_TAIL = re.compile('\\s*(?:Do not write a main function\\.\\s*)?Return only [^\\n]*\\Z')

# --- runs 1 and 2 (run 2 changed only T's thinking, low -> medium) ---------------------------------------------------
T_FORMAT_RUN1 = {'cpp_tests': 'Return exactly one ```cpp code block containing only statements for the body of `main()`; the '
              'candidate solution and `#include <bits/stdc++.h>` are placed above it. Do not write `int main()` '
              'and do not `return` from it. Check each condition with `CHECK(condition)`. Helper functions must be '
              'lambdas inside it, with names different from the functions the task asks for. You may compare a '
              'simple brute-force version that is obviously correct for small inputs with the solution on random '
              'small inputs (use `std::mt19937 rng(0)`).',
 'python_tests': 'Return exactly one ```python code block. It will be appended after a candidate solution and run '
                 'as a script, so use plain `assert` statements; do not use pytest or unittest, the script must '
                 'run all the tests itself when it is executed. You may define helper functions, for example a '
                 'simple brute-force version that is obviously correct for small inputs, and compare it with the '
                 'solution on random small inputs (use `random.seed(0)`); give helpers names different from the '
                 'functions and classes the task asks for.'}
T_PROMPT_RUN1 = ('{task}\n'
 '\n'
 '---\n'
 'You are writing tests for the task above. Do not write the solution.\n'
 '\n'
 '{fmt} Cover the edge cases, and include one test at the largest input size stated in the task (a correct, '
 'efficient solution must finish it within a few seconds). Every test must hold for any correct solution.')
# the critic K (run 1 only), on the tasks B verified; {code} is B's choice, {tests} T's tests (as run)
K_PROMPT = ('{task}\n'
 '\n'
 '---\n'
 'A candidate solution:\n'
 '\n'
 '```{fence}\n'
 '{code}\n'
 '```\n'
 '\n'
 'It passed these tests:\n'
 '\n'
 '```{fence}\n'
 '{tests}\n'
 '```\n'
 '\n'
 'List the requirements of the task and the edge cases that these tests do not check. Then write additional tests '
 'for them. Do not repeat the tests above. {fmt} Every test must hold for any correct solution.')

# --- runs 3 and 4: expected values only from a brute force, a property, or an example in the task --------------------
T_FORMAT = {'cpp_tests': 'Return exactly one ```cpp code block containing only statements for the body of `main()`; the '
              'candidate solution and `#include <bits/stdc++.h>` are placed above it. Do not write `int main()` '
              'and do not `return` from it. Check each condition with `CHECK(condition)`. Helper functions must be '
              'lambdas inside it, with names different from the functions the task asks for. Use `std::mt19937 '
              'rng(0)` for random inputs.',
 'python_tests': 'Return exactly one ```python code block. It will be appended after a candidate solution and run '
                 'as a script, so use plain `assert` statements; do not use pytest or unittest. Every check must '
                 'run when the script is executed: call every helper function you define, and do not leave checks '
                 'inside functions that are never called. Give helpers names different from the functions and '
                 'classes the task asks for. Use `random.seed(0)` for random inputs.'}
T_PROMPT = ('{task}\n'
 '\n'
 '---\n'
 'You are writing tests for the task above. Do not write the solution.\n'
 '\n'
 '{fmt}\n'
 '\n'
 'Do not work out expected outputs by hand; hand-computed values are the most common cause of wrong tests. Every '
 'expected value must come from one of these sources:\n'
 '1. A brute-force implementation that you write inside the tests, simple enough to be obviously correct for small '
 'inputs. Compare it with the solution on about 200 random small inputs, and on the smallest and other edge-case '
 'inputs. Every input must satisfy all constraints stated in the task.\n'
 '2. A property that every correct output must have and that can be checked without knowing the answer, for '
 'example that a returned path is valid or that a returned list is a permutation of the input.\n'
 '3. An example given in the task statement, with exactly the values stated there.\n'
 'If a requirement cannot be checked in one of these ways, leave it untested.\n'
 '\n'
 'Also include one test at the largest input size stated in the task. Check only properties on it (do not run the '
 'brute force on it); a correct, efficient solution must finish it within a few seconds. Every test must hold for '
 'any correct solution.')
RUN34_SHA256 = '0dab2a0b6ef7424a281e9fe0e1d77c055f3df1862544adaf32486015f61b0993'

# --- running the model's tests ---------------------------------------------------------------------------------------
# Python: the tests are appended after the candidate's code and run as a script (plain asserts).
# C++: the statements become the body of a static member function returning int, placed inside the grader's main()
# below the candidate's code and a CHECK(cond) macro that counts failures; a `return 0;` in the tests then ends only
# the tests (a failed CHECK before it still counts), and a non-zero value returned counts as a failure. A C++ block
# that is a whole `int main() {...}` is replaced by its body. `failures` is the failure counter of our grader's C++
# harness (Part 4's example grader calls it `fails`).
CPP_WRAP = 'struct p11_tests {{ static int run() {{\n{body}\nreturn 0; }} }};\nif (p11_tests::run() != 0) failures++;'
TEST_TIMEOUT_S = 30              # each run of the model's tests; C++ compilation up to 60 s, as in the grader


def run34_sha256():
    return hashlib.sha256(json.dumps([T_FORMAT, T_PROMPT], sort_keys=True).encode()).hexdigest()


if __name__ == '__main__':
    print('runs 1-2:', hashlib.sha256(json.dumps([T_FORMAT_RUN1, T_PROMPT_RUN1], sort_keys=True).encode()).hexdigest())
    print('runs 3-4:', run34_sha256())
    assert run34_sha256() == RUN34_SHA256
    print('prompts check passed: the runs 3-4 hash equals the one recorded in those runs')
