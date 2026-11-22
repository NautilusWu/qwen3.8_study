# Reproduce

## What can and cannot be reproduced

| | How |
|---|---|
| Every table in the README, from the per-task results | `scripts/recheck.sh` (no GPU) |
| The figures | `figures/make_figures.py` (needs matplotlib) |
| Building llama.cpp v1.0.0 and serving the GGUF files with this study's settings | steps 1-3 |
| A check on Part 4's public example tasks, with and without MTP, and against TensorFold | step 4 |
| The comparisons and the MTP speed probe on your own tasks | steps 5-6 |
| Re-running our scores | **not possible from this repository**: the 400 tasks are private (kept unseen for future model comparisons) |

## 0. Prerequisites

The runner and grader are Part 4's. Check out [Part 4's folder](https://github.com/NautilusWu/qwen3.8_study/blob/main/qwen38-spark-test-time-strategies/REPRODUCE.md) next to this one, and build its sandbox image with Part 4's `scripts/setup.sh`. `SKIP_MODELS=1` skips the 27B checkpoints.

For the TensorFold comparison you also need:
- Part 4's TensorFold 0.6.0 image (built by the same setup);
- [Part 6's folder](https://github.com/NautilusWu/qwen3.8_study/blob/main/qwen38-spark-flash-next/REPRODUCE.md), also next to this one;
- the NVFP4 checkpoint from Part 6's step 1.

To build llama.cpp you need git, CMake (3.28 here), GCC (13.3 here) and the CUDA 13.0 toolkit (`nvcc`), all present on DGX OS. The machine and versions are in the README's Conditions.

Nothing else may run on the GPU during a run. The GGUF files take 84-94 GB; TensorFold + NVFP4 takes about 90 GiB.

## 1. Build llama.cpp v1.0.0 (only `llama-server`)

```bash
git clone --depth 1 --branch v1.0.0 https://github.com/ggml-org/llama.cpp llama.cpp-v1.0.0
cd llama.cpp-v1.0.0 && git rev-parse HEAD
```
Expected: `b1b770d90d94cad9e9ea5b993bdbc383a522fb28`.

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121 \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_OPENSSL=OFF
cmake --build build --target llama-server -j 20
build/bin/llama-server --version
```
CMake prints `Replacing 121 in CMAKE_CUDA_ARCHITECTURES with 121a`. The version line says `1.0.0-dev (build 1, commit b1b770d)`. Nothing is downloaded during the build. The binary and its shared libraries are all in `build/bin/`.

```bash
export LLAMA_SERVER=$PWD/build/bin/llama-server
```

## 2. Download the files (pinned commits)

Every file, with its commit, size and sha256, is in [data/downloads.json](data/downloads.json).

| What | Size |
|---|---|
| GSQ-RCO IQ3_S | 83.6 GB |
| UD-IQ3_XXS | 83.6 GB |
| UD-IQ4_XS | 93.7 GB |
| MTP head | 4.1 GB |
| chat template | 8,952 bytes |

The two Unsloth files have the same fourth shard (28.8 GB), so downloading both is 148.5 GB plus the head. Download only what you need. Set `G` to a folder with enough space:

```bash
G=/path/to/gguf
hf download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF --revision ed59f92082b1e93c0e96d60a8b11aab089b52f09 \
  --include "IQ3_S/*" --local-dir $G/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
hf download unsloth/Qwen3.8-Flash-Next-GGUF --revision f5d175de2cf7f720d45d6064019a81fcf056e986 \
  --include "UD-IQ3_XXS/*" "UD-IQ4_XS/*" "mtp-Qwen3.8-Flash-Next-Q8_0.gguf" --local-dir $G/unsloth/Qwen3.8-Flash-Next-GGUF
hf download local-inference-lab/Qwen3.8-Flash-Next-NVFP4 chat_template.jinja --revision 7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd \
  --local-dir $G/local-inference-lab/Qwen3.8-Flash-Next-NVFP4
```

```bash
python3 scripts/check_downloads.py $G
```
Expected: `ok` for every file you downloaded, `missing` for the rest, and exit code 0.

Then set the paths the code reads:

```bash
export FN_TEMPLATE=$G/local-inference-lab/Qwen3.8-Flash-Next-NVFP4/chat_template.jinja
export MTP_HEAD=$G/unsloth/Qwen3.8-Flash-Next-GGUF/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
U3X=$G/unsloth/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf
U4X=$G/unsloth/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf
Q3S=$G/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
```

## 3. The server

[code/llama_cpp.py](code/llama_cpp.py) starts llama-server with this study's flags. It changes Part 4's runner in a few ways, each explained in that file's docstring:
- every request carries `"cache_prompt": false`, without which greedy replies depend on the previous request;
- health, idle and stop checks use llama.cpp's endpoints;
- the memory guard floor is 3 GiB;
- before each start it waits until `nvidia-smi` lists no process.

The same command by hand, for a quick look (MTP with up to 4 drafts):

```bash
taskset -c 5-9,15-19 $LLAMA_SERVER -m $U3X --alias Qwen3.8-Flash-Next --host 127.0.0.1 --port 8118 \
  -ngl 999 -c 81920 -np 1 -fa on -lm mmap --lazy-mode on --jinja --chat-template-file $FN_TEMPLATE \
  --reasoning-format deepseek --min-p 0 -t 10 -tb 10 --no-webui \
  -md $MTP_HEAD --spec-type draft-mtp --spec-draft-n-max 4
```
With MTP, the log contains `loading draft model`. Leave out the last line for MTP off.

## 4. Check on Part 4's example tasks

All commands in this step run in `code/`, with this environment:

```bash
cd code
export TASKS_DIR=../../qwen38-spark-test-time-strategies/examples/tasks GRADER_DIR=../../qwen38-spark-test-time-strategies/examples/grader
P4=../../qwen38-spark-test-time-strategies/code/strategies
P6=../../qwen38-spark-flash-next/code
```

**MTP speed probe, off and with 4 drafts** (output capped at 8,192 tokens; about 10 minutes):

```bash
python3 probe.py $P4 check-probe --gguf $U3X --mtp 0,4 --split dev
```
Expected: both runs start, and only the second log says the draft model loaded. With greedy decoding the same software should give the same tokens.

| Run | Output tokens (`ex_primes`, `ex_trains`, `ex_palindrome`, `ex_gcd_cpp`, `ex_five_numbers`) | Decoding | Drafts accepted |
|---|---|---|---|
| MTP off | 783, 512, 115, 418, 1,290 | 21.2 tokens/s | |
| 4 drafts | 783, 527, 117, 517, 8,192 (capped) | 53.0 tokens/s | 94% |

Our replies were character for character the same as in this study's compatibility check (all 9 that were not capped). `results/check-probe/summary.json` reports 1 of 5 replies identical to MTP off (`ex_primes`).

**Alternating blocks, GGUF + MTP against TensorFold** (2 groups, 4 blocks):

```bash
python3 run_blocks.py $P4 check-blocks --arm UD-IQ3_XXS_MTP=gguf:$U3X:mtp=4 --arm TensorFold_NVFP4=tensorfold:$P6 \
  --orders "UD-IQ3_XXS_MTP,TensorFold_NVFP4;TensorFold_NVFP4,UD-IQ3_XXS_MTP" --split dev
```
For TensorFold, set Part 6's environment too (`HF_DIR`, `TF_CACHE_DIR`, and `FN_TARGET` if the checkpoint is elsewhere).

Ours, about 21 minutes in all. The run writes `per_task_*.csv`, `blocks.csv`, `same_text.csv` and `stats.json` (on 5 tasks the comparison means nothing).

| | UD-IQ3_XXS + MTP | TensorFold + NVFP4 |
|---|---|---|
| Correct | 4 of 5 | 5 of 5 |
| Output tokens | 67,480 | 2,541 (the same as Part 6's check) |
| Server start | 12-52 s | 91-116 s |
| Lowest available memory | 55.2 GiB | 35.3 GiB |

- On the GGUF side, `ex_five_numbers` ran to the 65,536-token limit (1,100 s).
- All 5 GGUF replies were character for character the same as in this study's compatibility check at N = 4.
- No start had to wait for the GPU.
- The draft model loaded in both llama.cpp blocks, and llama-server listened on 127.0.0.1 only.

## 5. The comparisons on your own tasks

All commands run in `code/`, with `TASKS_DIR` and `GRADER_DIR` pointing at your tasks. The task and grader format is in Part 4's REPRODUCE.md.

[code/run_blocks.py](code/run_blocks.py) runs each configuration on every group of tasks, in alternating blocks:
- a fresh server for every block, each block in its own process;
- two failed blocks in a row stop the run;
- grading at the end in the network-isolated sandbox, pinned to the Cortex-A725 cores (`SANDBOX_CPUS`).

It writes per-task CSVs with the columns of `data/per_task_400/`, `same_text.csv` and `blocks.csv`, then runs `stats.py` on every pair. Task text and model output stay in `results/` and are never printed.

**PLAN_18: three files, MTP off, this study's order of 8 groups:**

```bash
python3 run_blocks.py $P4 three-ggufs --split confirm --limit-min 120 --max-tokens 65536 \
  --arm IQ3_S=gguf:$Q3S --arm UD-IQ3_XXS=gguf:$U3X --arm UD-IQ4_XS=gguf:$U4X \
  --orders "IQ3_S,UD-IQ3_XXS,UD-IQ4_XS;UD-IQ3_XXS,UD-IQ4_XS,IQ3_S;UD-IQ4_XS,IQ3_S,UD-IQ3_XXS;IQ3_S,UD-IQ4_XS,UD-IQ3_XXS;UD-IQ3_XXS,IQ3_S,UD-IQ4_XS;UD-IQ4_XS,UD-IQ3_XXS,IQ3_S;IQ3_S,UD-IQ3_XXS,UD-IQ4_XS;UD-IQ4_XS,IQ3_S,UD-IQ3_XXS"
```

**PLAN_18c: GGUF + MTP against TensorFold.** With two configurations, the default order is AB BA BA AB AB BA BA AB:

```bash
python3 run_blocks.py $P4 gguf-mtp-vs-tensorfold --split confirm --limit-min 120 --max-tokens 65536 \
  --arm UD-IQ3_XXS_MTP=gguf:$U3X:mtp=4 --arm TensorFold_NVFP4=tensorfold:$P6
```

`--method` takes any of Part 4's single-reply methods (default `D2`, low thinking).

PLAN_18's tool-call check replayed recorded turns of our private agent tasks, so it is not included. Its labelling rule is in [code/tool_call_labels.py](code/tool_call_labels.py).

## 6. MTP speed probe on your own tasks

```bash
python3 probe.py $P4 my-probe --gguf $U3X --mtp 0,2,3,4 --split confirm --tasks id1,id2,...
```

`summary.json` gives, per N:
- decoding speed (llama.cpp's `predicted_n` / `predicted_ms`);
- draft acceptance;
- replies identical to MTP off;
- the N this study's rule picks: the fastest, unless a smaller N is within 97% of it.

Add `--examples-split dev` to also run and grade the example tasks uncapped, as the compatibility check did.

## 7. Recompute the tables

```bash
scripts/recheck.sh
```
Expected last lines: `summary.csv: all 6 rows match` and `tests: all match`. It prints every table of the README, including the analysis done after the results (marked as such).

```bash
python3 figures/make_figures.py
```
Redraws the figures (needs matplotlib).
