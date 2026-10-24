# Test image for comparing coding-agent harnesses on the same local model. Agents only; no model weights.
# Also the sandbox for running and grading code in code/strategies. Versions are the ones used for the results.
FROM node:22.23.3-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv python3-pytest git ripgrep jq \
    && rm -rf /var/lib/apt/lists/*
RUN npm install -g @qwen-code/qwen-code@0.25.0 opencode-ai@1.18.35 @openai/codex@0.162.1 @anthropic-ai/claude-code@2.1.296 \
    && npm cache clean --force
RUN mkdir -p /work && chown node:node /work
USER node
WORKDIR /work
RUN qwen --version && opencode --version && codex --version && claude --version
