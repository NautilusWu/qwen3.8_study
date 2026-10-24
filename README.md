# Qwen3.8-27B on NVIDIA DGX Spark: Inference and Test-Time Compute Studies

An empirical research repository investigating how to run **Qwen3.8-27B** efficiently on a single **NVIDIA DGX Spark**, and how to turn inference-time computation into more correct answers rather than merely more generated tokens.

The project covers inference backends, quantization, speculative decoding, performance–accuracy trade-offs, and test-time reasoning strategies. Each study retains its own methodology and experimental artifacts; this page is the entry point.

> **Scope:** These are measurements on specific model snapshots, software versions, workloads, and hardware settings—not universal rankings of inference engines or reasoning methods.

## Research questions

1. How much inference performance can a single DGX Spark deliver for Qwen3.8-27B?
2. What changes when we switch inference engines, numerical formats, and decoding techniques?
3. Does higher token throughput actually reduce end-to-end time to a correct answer?
4. When is additional test-time computation useful—and should it be spent on longer reasoning, parallel candidates, or tools?

## Key findings

The most recent test-time compute study compares strategies on a **48-task confirmation set**, separate from earlier exploratory tasks. The published summary reports:

| Strategy | Correct / 48 | Accuracy | Total task time | Minutes per correct answer |
|---|---:|---:|---:|---:|
| A65 — extended/deep reasoning | 35 | 72.9% | 336.5 min | 9.61 |
| D2 — low-effort reasoning | 38 | 79.2% | 112.4 min | 2.96 |
| H — hybrid reasoning | 40 | 83.3% | 124.0 min | 3.10 |
| CL4 — four concurrent low-effort candidates | 44 | 91.7% | 124.3 min | 2.82 |

**Interpretation:** In this particular test set, longer single-path reasoning did not improve accuracy or time efficiency. Concurrent candidates produced the highest observed accuracy while retaining a competitive time per correct answer. This does **not** establish a universal advantage: the one-sided paired test comparing CL4 with D2 was reported as *p* = 0.055, above the conventional 0.05 threshold.

The earlier engine-comparison study also explores why raw output tokens/second and end-to-end task completion time can rank configurations differently. Refer to the individual reports for setup, caveats, and precise metric definitions.

## Studies

| Study | Focus | Where to start |
|---|---|---|
| Part 1 | Initial Qwen3.8-27B deployment and inference measurements | Browse the repository directories below |
| Part 2 | Inference optimization and decoding configurations | Browse the repository directories below |
| Part 3 | Engine, quantization, throughput, and accuracy comparisons | [Inference speed comparison](qwen38-spark-speed-comparison-final/README.md) |
| Part 4 | Reasoning budgets, parallel sampling, and cooperative tool-using strategies | [Test-time strategies](qwen38-spark-test-time-strategies/README.md) |

The studies are related but should not be interpreted as a single controlled experiment: workloads and configurations may differ between parts.

## Part 4: Test-time compute

The fourth study investigates several ways to allocate inference-time work:

- **Single-path reasoning:** compare low and extended reasoning budgets.
- **Independent parallel candidates:** generate multiple solutions and choose a result using task-dependent selection rules.
- **Hybrid reasoning:** combine stages with different reasoning budgets.
- **Cooperative strategies:** divide tasks or assign specialized roles, including sandbox-based code execution and verification.

Its evaluation includes exploratory development tasks, a harder exploratory set, and a separate confirmation set. Published per-task data enables readers to inspect failures, elapsed time, output tokens, and selection rules rather than relying only on aggregate accuracy.

### Recompute published confirmation statistics

From the repository root:

```bash
cd qwen38-spark-test-time-strategies/code/strategies
python3 analyze_confirm.py --csv ../../data/per_task.csv
```

This analysis script computes paired comparisons and bootstrap intervals from the published per-task CSV. It does **not** rerun model inference.

### Data and implementation

- [Aggregate results](qwen38-spark-test-time-strategies/data/summary.csv)
- [Per-task results](qwen38-spark-test-time-strategies/data/per_task.csv)
- [Confirmation analysis](qwen38-spark-test-time-strategies/code/strategies/analyze_confirm.py)
- [Cooperative strategy implementation](qwen38-spark-test-time-strategies/code/strategies/coop.py)
- [Shared inference, grading, and sandbox utilities](qwen38-spark-test-time-strategies/code/strategies/common.py)

## Reproducibility

The repository provides strategy implementations, aggregated and per-task measurements, and analysis scripts. For full inference reproduction, follow the instructions and pinned environment details in each study's own README.

**Important limitation:** The Part 4 task prompts and grading package used in the original study are not published. As a result, the public CSV-based statistical analysis is reproducible, but an independent end-to-end reproduction of the original scores requires equivalent task and grader inputs. The strategy code expects these inputs via `TASKS_DIR` and `GRADER_DIR`.

Additional dependencies for Part 4 include a compatible TensorFold serving environment, model snapshots, Docker, and an isolated execution image. The code is tuned for the tested DGX Spark environment; do not assume it is portable without adjustments.

## Reading the results responsibly

- **Tokens/second is not the same as time to a correct answer.** Generated length, prompt processing, parallel scheduling, and tool overhead matter.
- **A higher observed score is not necessarily statistically conclusive.** Paired task-level comparisons are more informative than comparing two percentages alone.
- **More computation is not automatically better.** The value of longer reasoning or parallel sampling depends on task type and the selection mechanism.
- **Results are configuration-specific.** Quantization, model snapshot, backend, task selection, and hardware constraints can change the outcome.

## Future directions

Potential extensions include adaptive selection between low-effort reasoning, parallel candidates, and tool-assisted verification; evaluation on fully public benchmarks; additional independent runs; and explicit latency–accuracy Pareto comparisons across hardware and models.

These are research directions, not claims of completed results.

## Repository

[GitHub: NautilusWu/qwen3.8_study](https://github.com/NautilusWu/qwen3.8_study)

For detailed methodology, configuration flags, and interpretation, begin with the [Part 3](qwen38-spark-speed-comparison-final/README.md) and [Part 4](qwen38-spark-test-time-strategies/README.md) reports.
