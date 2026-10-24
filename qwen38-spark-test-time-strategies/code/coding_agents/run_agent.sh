#!/usr/bin/env bash
# Run one coding agent on one task folder, non-interactively, in a throwaway container.
#   run_agent.sh <qwen|opencode|codex|claude> <workdir> <prompt_file> <out_dir> [timeout_s]
# The agent sees only <workdir> (mounted at /work). It reaches the model only through the local
# normalizing proxy (127.0.0.1:8099) or, for Claude Code, LiteLLM (127.0.0.1:4000). No real credentials
# are present: Claude Code has no claude.ai login and uses a local-only token.
set -u
agent="$1"; work="$(realpath "$2")"; prompt_file="$(realpath "$3")"; out="$(realpath -m "$4")"; limit="${5:-1800}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$out" "$HERE/cache/opencode"
prompt="$(cat "$prompt_file")"
# A fresh, writable home per run: agents keep no state between runs and see only their own config.
home="$out/home"; rm -rf "$home"; mkdir -p "$home/.qwen" "$home/.config/opencode" "$home/.codex" "$home/.cache"
cp "$HERE/configs/qwen/settings.json" "$home/.qwen/"
cp "$HERE/configs/opencode/opencode.json" "$home/.config/opencode/"
cp "$HERE/configs/codex/config.toml" "$home/.codex/"
common=(--rm --name "harness-$agent-$$" --network host --user 1000:1000 --memory 8g --pids-limit 1024
        -v "$work:/work" -v "$home:/home/node" -w /work -e HOME=/home/node -e NO_COLOR=1 -e CI=1)
case "$agent" in
  qwen)
    cmd=(docker run "${common[@]}"
         -e OPENAI_API_KEY=local-only -e OPENAI_BASE_URL=http://127.0.0.1:8099/v1 -e OPENAI_MODEL=qwen3.8-27b-tensorfold-nvfp4
         agent-harness:0.1 qwen --yolo -o json "$prompt") ;;
  opencode)
    cmd=(docker run "${common[@]}"
         -v "$HERE/cache/opencode:/home/node/.cache/opencode" -e OPENCODE_DISABLE_AUTOUPDATE=1
         agent-harness:0.1 opencode run --auto --format json "$prompt") ;;
  codex)
    cmd=(docker run "${common[@]}" -e LOCAL_API_KEY=local-only
         agent-harness:0.1 codex exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check --json "$prompt") ;;
  claude)
    cmd=(docker run "${common[@]}" -e ANTHROPIC_BASE_URL=http://127.0.0.1:4000 -e ANTHROPIC_AUTH_TOKEN=sk-local-harness-test
         -e ANTHROPIC_MODEL=qwen3.8-27b -e ANTHROPIC_DEFAULT_OPUS_MODEL=qwen3.8-27b -e ANTHROPIC_DEFAULT_SONNET_MODEL=qwen3.8-27b
         -e ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3.8-27b -e CLAUDE_CODE_SUBAGENT_MODEL=qwen3.8-27b
         -e DISABLE_TELEMETRY=1 -e DISABLE_AUTOUPDATER=1 -e CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
         agent-harness:0.1 claude -p --dangerously-skip-permissions --output-format json "$prompt") ;;
  *) echo "unknown agent $agent" >&2; exit 2 ;;
esac
t0=$(date +%s.%N)
timeout --kill-after=30 "$limit" "${cmd[@]}" > "$out/agent_stdout.txt" 2> "$out/agent_stderr.txt"
rc=$?
docker rm -f "harness-$agent-$$" >/dev/null 2>&1
t1=$(date +%s.%N)
printf '{"agent": "%s", "exit_code": %d, "timed_out": %s, "wall_s": %s}\n' "$agent" "$rc" \
  "$([ $rc -eq 124 ] || [ $rc -eq 137 ] && echo true || echo false)" "$(echo "$t1 - $t0" | bc)" > "$out/run.json"
cat "$out/run.json"
