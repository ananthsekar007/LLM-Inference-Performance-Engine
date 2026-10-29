# CSE 598 — LLM Serving Systems: Assignment Report

**Name:** _TODO_
**Date:** _TODO_

---

## Part 0. Environment and Methodology

### 0.1 Hardware and software environment

All experiments were run interactively on a single GPU node of the ASU Sol cluster. The raw output is saved in `results/env.txt`.

| Component | Version / Value |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB (80 GiB HBM2e, 81,920 MiB reported by `nvidia-smi`) |
| GPU power cap | 500 W |
| MIG | Disabled (full GPU available) |
| NVIDIA driver (kernel-mode) | 615.71.09 |
| Max CUDA supported by driver | 13.4 |
| CUDA version PyTorch was built with | 13.0 |
| cuDNN | 9.20.0 (reported as `92000`) |
| Python | 3.11.18 |
| PyTorch | 2.13.0+cu130 |
| Hugging Face Transformers | 5.19.0 |
| vLLM | 0.32.0 |

`nvidia-smi` reports the highest CUDA version the installed *driver* can run (13.4). `torch.version.cuda` reports the CUDA toolkit PyTorch was *compiled* against (13.0). A driver that supports a newer CUDA version can run binaries built for an older one, so this pairing is compatible.

At the time the environment was recorded, `nvidia-smi` showed 0 MiB in use and no running processes, so no other jobs were sharing the GPU.

### 0.2 Model and numerical precision

| Setting | Value |
|---|---|
| Model | `gpt2` (124M parameters, decoder-only causal LM) |
| Architecture | 12 layers, 12 attention heads, hidden size 768, head dim 64 |
| KV heads | 12 (multi-head attention; no grouped-query attention) |
| Maximum context | 1024 positions (`n_positions`) |
| Precision | FP16 for all GPU runs (Hugging Face and vLLM) |
| Decoding | Greedy (argmax / `temperature=0`) |

**Context-length constraint.** GPT-2 has learned absolute position embeddings for only 1024 positions, so every run must satisfy *prompt length + output length ≤ 1024*. The experiment parameters were chosen to respect this:

- Part 1 uses prompt lengths {128, 256, 512, 768} and output lengths {32, 64, 128, 256}, so the largest case is 768 + 256 = 1024.
- The Part 4 workload uses prompt lengths {32, 64, 128, 256, 512} and output lengths {16, 32, 64, 128, 256}. Static batching pads each batch to its longest prompt and runs to its longest output. Capping prompts at 512 keeps every batch, and every vLLM request, within the 1024-token `max_model_len`.

### 0.3 Measurement methodology

- **GPU synchronization.** Every timed region is wrapped by `utils.time_cuda_safe`, which calls `torch.cuda.synchronize()` immediately before reading the start time and immediately after the work finishes (see Question 0.1).
- **Warm-up.** Every script runs a short warm-up (prompt 32, output 4) before any measurement. The first CUDA calls pay one-time costs: creating the CUDA context, loading kernels, cuBLAS/cuDNN heuristic and algorithm selection, and growing the caching allocator. Excluding the warm-up keeps those costs out of the reported numbers.
- **Model loading excluded.** The model and tokenizer are loaded and moved to the GPU in `main()` before any timer starts. Download and load time is therefore never included in latency.
- **Fixed model, precision and hardware.** Every comparison uses gpt2 in FP16 on the same A100-SXM4-80GB.
- **Repetitions.** _TODO: describe the repetition policy as implemented (e.g., each configuration is run 3 times after warm-up and the mean ± standard deviation is reported; raw per-run rows are kept in the CSV)._
- **Synthetic inputs.** Prompts are random token IDs (`utils.synthetic_tokens`) with a fixed seed (`seed_everything(42)`). Content does not affect compute cost for a fixed sequence length, and fixed seeds make runs reproducible.
- **Memory metric.** Peak memory is `torch.cuda.max_memory_allocated()`, reset before each measured case. It counts tensors allocated by PyTorch, including the model weights. It is lower than the usage `nvidia-smi` shows, because `nvidia-smi` also counts the CUDA context and memory held in PyTorch's caching allocator but not currently allocated.

### Question 0.1 — Why is naive `time.perf_counter()` timing incorrect on a GPU?

```python
start = time.perf_counter()
output = model(input_ids)
end = time.perf_counter()
```

CUDA execution is **asynchronous** with respect to the host. When Python calls `model(input_ids)`, PyTorch *enqueues* a sequence of kernels on a CUDA stream and returns control to the CPU, usually before the GPU has finished, or even started, running them. The interval `end - start` therefore measures mostly CPU-side work: Python overhead, operator dispatch and kernel launch. It does not measure the time the GPU spends computing, so it usually **underestimates** latency.

There is also a second error in the opposite direction. Work enqueued *before* `start` may still be running on the GPU. If a later operation forces a sync inside the timed region, for example copying a result to the CPU, that leftover work is wrongly counted in this measurement.

`torch.cuda.synchronize()` blocks the host until every kernel queued on the device has completed. Correct timing calls it twice:

1. **before `start`**, so the GPU is idle and no earlier work leaks into the measurement;
2. **after the model call, before `end`**, so `end` is taken only after the GPU has actually finished.

This is the pattern in `utils.time_cuda_safe`. An alternative is `torch.cuda.Event(enable_timing=True)`, which records timestamps on the GPU stream itself.

**Empirical check.** A single FP16 8192 × 8192 matrix multiply was timed on the A100, after three warm-up multiplies:

| Timing method | Measured time |
|---|---|
| `perf_counter()` only (no synchronize) | 0.084 ms |
| `synchronize()` before start and before end | 4.528 ms |

Without synchronization, the measured time is about 54× too small. It captures only the kernel launch. The synchronized time agrees with the expected cost of the matmul: 2 · 8192³ ≈ 1.10 × 10¹² FLOPs in 4.528 ms ≈ 243 TFLOP/s, about 78% of the A100's 312 TFLOP/s FP16 Tensor Core peak. So the synchronized number reflects the GPU's actual compute time, and the unsynchronized one does not.

---

## Part 1. Prefill and Decode

### What is asked

- **Task 1.1** — For at least 3 prompt lengths × 3 output lengths, measure prefill latency, decode latency, end-to-end latency, average decode time per output token (TPOT), and peak GPU memory.
- **Task 1.2** — Plots: prompt length vs. prefill latency; output length vs. decode latency; output length vs. TPOT.
- **Task 1.3** — Answer the six analysis questions below.

### Experimental parameters

| Parameter | Value |
|---|---|
| Prompt lengths | 128, 256, 512, 768 |
| Output lengths | 32, 64, 128, 256 |
| Batch size | 1 |
| Script | `python part1_prefill_decode.py` |
| Output | `part1_prefill_decode.csv` |

### Results

_TODO: table of mean prefill_ms, decode_ms, total_ms, tpot_ms, peak_memory_gb per (prompt, output) configuration._

_TODO: insert plots (prefill vs prompt length, decode vs output length, TPOT vs output length)._

### Analysis

1. **Why does prompt length affect prefill?** _TODO_
2. **Why does output length affect decode?** _TODO_
3. **Why can prefill expose more GPU parallelism than decode?** _TODO_
4. **Which phase appears more compute intensive?** _TODO_
5. **Which phase appears more memory-bandwidth sensitive?** _TODO_
6. **Does TPOT remain constant as the context grows? If not, why?** _TODO_

---

## Part 2. KV Cache

### What is asked

- **Task 2.1** — Complete `generate_without_cache()`: pass the whole sequence at every step, use `use_cache=False`, append each generated token.
- **Task 2.2** — Compare cache on vs. off for several output lengths: total latency, ms per output token, peak GPU memory.
- **Task 2.3** — Derive the KV-cache size formula, explain the factor of 2, and check it with `part2_kv_size.py` for at least one configuration.
- **Task 2.4** — Answer the four analysis questions below.

### Implementation (Task 2.1)

_TODO: briefly describe the three steps of `generate_without_cache()` and the agreement-check result (PASS, or the divergence position and the fp16 explanation)._

### Experimental parameters

| Parameter | Value |
|---|---|
| Prompt length | 128 |
| Output lengths | _TODO_ |
| Script | `python part2_kv_cache.py` |
| Output | `part2_kv_cache.csv` |

### Results (Task 2.2)

_TODO: table of latency_sec, ms_per_token, peak_memory_gb for use_cache ∈ {True, False} × output length._

_TODO: insert plot(s)._

### KV-cache memory derivation (Task 2.3)

_TODO: derive the formula from the per-layer K and V tensor shapes, explain the factor of 2, and show the `part2_kv_size.py` check for gpt2 (layers=12, kv-heads=12, head-dim=64, bytes=2) and at least one other configuration._

### Analysis (Task 2.4)

1. **Why does caching reduce redundant computation?** _TODO_
2. **Why does cache memory grow with sequence length?** _TODO_
3. **Why does long-context serving reduce concurrency?** _TODO_
4. **Why can a serving system become memory-limited even when compute is available?** _TODO_

---

## Part 3. Batching

### What is asked

- **Task 3.1** — For several batch sizes, measure TTFT (approximation), TPOT, end-to-end latency, requests/s, output tokens/s, and peak GPU memory.
- **Task 3.2** — Plots: batch size vs. tokens/s; batch size vs. latency; batch size vs. peak memory.
- **Task 3.3** — Answer the five analysis questions below.

### Experimental parameters

| Parameter | Value |
|---|---|
| Batch sizes | _TODO_ |
| Prompt length | 128 |
| Output length | _TODO_ |
| Script | `python part3_batching.py` |
| Output | `part3_batching.csv` |

### Metric definitions (TODO 2 in the starter code)

_TODO: state and justify each formula: `requests_per_sec`, `tokens_per_sec`, `tpot_ms`, and why `ttft_ms` (prefill time) is only an approximation of TTFT._

### Results

_TODO: results table and the three plots._

### Analysis

1. **Why can larger batches improve GPU utilization?** _TODO_
2. **Why does throughput not increase linearly forever?** _TODO_
3. **Where do you observe diminishing returns?** _TODO_
4. **Why can latency increase even when throughput improves?** _TODO_
5. **For an interactive chatbot, would you choose the max-throughput batch size? Why or why not?** _TODO_

---

## Part 4. Heterogeneous Requests and Serving Engines

### What is asked

- **Task 4.1** — Run static batching on a heterogeneous workload; report throughput, mean latency, P95 latency, and prompt-padding efficiency. Explain padding waste using the prompt lengths 128, 150, 180, 1024.
- **Task 4.2** — Explain what happens with output lengths 20, 25, 200, 30 in naive static batching, and what hardware utilization is lost.
- **Task 4.3** — Compare vLLM against static batching (at minimum requests/s and output tokens/s).
- **Task 4.4** — Explain at least two serving mechanisms: the problem each solves, how it works, and the link to your measurements.

### Workload

| Parameter | Value |
|---|---|
| Requests | 50 (`python workload.py --n 50`) |
| Arrival process | Poisson, rate 5 req/s (arrival times are not used by the offline scripts) |
| Prompt lengths | {32, 64, 128, 256, 512}, uniform |
| Output lengths | {16, 32, 64, 128, 256}, uniform |
| Seed | 42 |

### Task 4.1 — Static batching results

_TODO: table of throughput, mean latency, P95 latency, padding efficiency for each batch size tested._

**Padding example (128, 150, 180, 1024):** _TODO: useful tokens vs. tokens processed after padding, and the resulting efficiency._

### Task 4.2 — Output-length heterogeneity (20, 25, 200, 30)

_TODO (also covers TODO 3 in `part4_static_batching.py`): explain why the batch runs to max_output, quantify the wasted slot-steps, and describe what changes if requests could leave the batch as soon as they finish._

### Task 4.3 — vLLM comparison

_TODO: requests/s and output tokens/s for vLLM vs. static batching, with the vLLM settings used (dtype, max_tokens / per-request SamplingParams)._

_TODO (TODO 4 in `part4_vllm_offline.py`): explain why a single `max_tokens` value does not reproduce heterogeneous output lengths, and how that affects the comparison._

### Task 4.4 — Systems mechanisms

#### Mechanism 1: _TODO_
- **Problem addressed:** _TODO_
- **How it works:** _TODO_
- **Connection to measurements:** _TODO_

#### Mechanism 2: _TODO_
- **Problem addressed:** _TODO_
- **How it works:** _TODO_
- **Connection to measurements:** _TODO_

---

## Part 5. Serving Design Challenge

### What is asked

Design a policy for a single-GPU service with continuous arrivals and heterogeneous requests that meets the TTFT and TPOT SLOs while maximizing throughput. Cover each topic below and discuss tradeoffs.

1. **Request scheduling:** _TODO_
2. **Continuous / dynamic batching:** _TODO_
3. **Prefill vs. decode scheduling:** _TODO_
4. **KV-cache allocation:** _TODO_
5. **Handling long-context requests:** _TODO_
6. **Admission control:** _TODO_
7. **Overload behavior:** _TODO_

**Tradeoffs:** _TODO_

---

## Part 6. Interpretation and Reflection

### What is asked

300–500 words: *Why is LLM serving fundamentally a systems problem rather than simply a model-inference problem?* Use evidence from the experiments.

_TODO_

---

## Reproducibility

_TODO: point to README.md, which lists environment setup, the exact command for each part, and how the plots are generated._

## Deliverables

- [ ] Source code
- [ ] This report (exported to PDF)
- [ ] Plots
- [ ] CSV result files
- [ ] README with reproduction steps
