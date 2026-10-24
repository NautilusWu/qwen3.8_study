#!/usr/bin/env bash
# grade.sh <task_dir> <workdir> [out.json]: run the task's hidden tests against the agent's work.
# No network, the work folder read-only, hidden tests mounted separately (the agent never saw them).
task="$(realpath "$1")"; work="$(realpath "$2")"; out="${3:-/dev/stdout}"
if [ -f "$task/hidden/hidden.test.js" ]; then
  cmd=(node --test /grade/hidden.test.js)
else
  cmd=(python3 -m pytest -q -p no:cacheprovider /grade)
fi
res="$(docker run --rm --network none --user 1000:1000 --memory 2g --pids-limit 256 -e PYTHONPATH=/work -e REPO=/work \
  -e PYTHONDONTWRITEBYTECODE=1 -v "$work:/work:ro" -v "$task/hidden:/grade:ro" -w /tmp agent-harness:0.1 timeout 300 "${cmd[@]}" 2>&1)"
rc=$?
summary="$(printf '%s\n' "$res" | grep -E "passed|failed|error|^# (pass|fail)" | tail -3 | tr '\n' ' ' | sed 's/"/\\"/g')"
printf '{"passed": %s, "exit_code": %d, "summary": "%s"}\n' "$([ $rc -eq 0 ] && echo true || echo false)" "$rc" "$summary" > "$out"
