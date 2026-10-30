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
- **Repetitions.** Each configuration is run 3 times, after a single warm-up at the start of the script. Tables report the mean ± standard deviation across the 3 runs. The raw per-run rows (with a `rep` column) are kept in `results/part1_prefill_decode.csv`, and the aggregates are in `*_mean.csv` and `*_std.csv`.
- **First run at a new input shape.** In Part 1, the first repetition (rep 0) at a new prompt length was often slower than reps 1–2. At prompt 512 / output 32 it took 20.9 ms vs. 9.7 and 9.3 ms. The warm-up only exercised a 32-token prompt, so the first call at each new sequence length still pays one-time costs such as choosing a GEMM/attention kernel for that shape. Averaging three runs reduces the effect, and the standard-deviation columns make it visible.
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
| Decoding | Greedy, manual loop with `past_key_values` (no `generate()`, no early EOS stop) |
| Precision | FP16 |
| Warm-up | 1 run (prompt 32, output 4) before measurement |
| Repetitions | 3 per configuration (48 runs in total) |
| Script | `python part1_prefill_decode.py --out results/part1_prefill_decode.csv` |
| Output | `results/part1_prefill_decode.csv` (raw), `_mean.csv`, `_std.csv` |

**Metric definitions.**
- **Prefill** is one forward pass over the full prompt, which also produces the first output token.
- **Decode** is the loop of `output_length` single-token forward passes that reuse the KV cache.
- **End-to-end** = prefill + decode.
- **TPOT** = decode time / `output_length`, i.e. the mean time per decode step.
- **Peak memory** is `torch.cuda.max_memory_allocated()` and includes the FP16 weights (124.4M params × 2 B ≈ 0.232 GiB).

### Task 1.1 — Results (mean ± std over 3 runs)

| Prompt | Output | Prefill (ms) | Decode (ms) | End-to-end (ms) | TPOT (ms/token) | Peak memory (GiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 32 | 8.01 ± 1.01 | 177.1 ± 0.7 | 185.1 ± 0.8 | 5.534 ± 0.023 | 0.2635 |
| 128 | 64 | 7.45 ± 0.03 | 357.5 ± 1.5 | 365.0 ± 1.5 | 5.586 ± 0.023 | 0.2647 |
| 128 | 128 | 7.57 ± 0.12 | 717.4 ± 8.4 | 725.0 ± 8.5 | 5.605 ± 0.066 | 0.2670 |
| 128 | 256 | 7.49 ± 0.02 | 1445.2 ± 10.4 | 1452.7 ± 10.4 | 5.645 ± 0.041 | 0.2715 |
| 256 | 32 | 8.06 ± 0.71 | 183.7 ± 3.0 | 191.8 ± 3.7 | 5.741 ± 0.094 | 0.2801 |
| 256 | 64 | 8.85 ± 2.18 | 363.8 ± 4.8 | 372.6 ± 3.0 | 5.684 ± 0.074 | 0.2812 |
| 256 | 128 | 9.42 ± 3.21 | 724.3 ± 3.9 | 733.7 ± 0.9 | 5.659 ± 0.030 | 0.2835 |
| 256 | 256 | 7.67 ± 0.06 | 1454.4 ± 10.3 | 1462.1 ± 10.2 | 5.681 ± 0.040 | 0.2881 |
| 512 | 32 | 13.28 ± 6.61 | 180.0 ± 0.4 | 193.3 ± 6.3 | 5.625 ± 0.012 | 0.3141 |
| 512 | 64 | 9.40 ± 0.15 | 361.0 ± 0.5 | 370.4 ± 0.4 | 5.640 ± 0.007 | 0.3152 |
| 512 | 128 | 9.34 ± 0.05 | 720.1 ± 4.0 | 729.5 ± 4.0 | 5.626 ± 0.032 | 0.3175 |
| 512 | 256 | 9.45 ± 0.10 | 1451.0 ± 10.2 | 1460.5 ± 10.1 | 5.668 ± 0.040 | 0.3353 |
| 768 | 32 | 8.35 ± 1.23 | 179.3 ± 0.3 | 187.7 ± 1.1 | 5.604 ± 0.009 | 0.3486 |
| 768 | 64 | 7.56 ± 0.04 | 357.3 ± 3.1 | 364.8 ± 3.1 | 5.583 ± 0.048 | 0.3497 |
| 768 | 128 | 8.07 ± 0.63 | 713.8 ± 5.9 | 721.9 ± 6.2 | 5.577 ± 0.046 | 0.3511 |
| 768 | 256 | 7.67 ± 0.12 | 1425.9 ± 11.9 | 1433.6 ± 11.8 | 5.570 ± 0.047 | 0.3555 |

The 13.28 ± 6.61 ms prefill at prompt 512 / output 32 is caused by a single slow first run (20.9 ms) at a new prompt length (see Part 0). The other two runs took 9.7 and 9.3 ms.

**Summary statistics.**
- **Prefill:** with the slow first runs excluded (reps 1–2 only), prefill is 7.5, 8.1, 9.4 and 7.9 ms for prompts of 128, 256, 512 and 768 tokens. That is nearly flat across a 6× change in prompt length.
- **Decode:** a linear fit of decode latency against output length gives a slope of **5.64 ms/token**, an intercept of −1.5 ms, and **R² = 0.9998**.
- **TPOT:** across all 16 configurations it ranges from 5.53 to 5.74 ms (mean 5.63 ms, a spread of about ±2%).

### Task 1.2 — Plots

![Prompt length vs prefill latency](plots/part1_prefill_vs_prompt.png)

![Output length vs decode latency](plots/part1_decode_vs_output.png)

![Output length vs TPOT](plots/part1_tpot_vs_output.png)

![Prompt length vs TPOT](plots/part1_tpot_vs_prompt.png)

PDF versions of each figure are in `plots/`.

### Analysis

Reference numbers used below:
- A100-SXM4-80GB FP16 Tensor Core peak: about 312 TFLOP/s. Measured earlier: 243 TFLOP/s on an 8192² matmul.
- HBM bandwidth: about 2.0 TB/s.
- gpt2 FP16 weights: 124.4M params × 2 B ≈ 249 MB.
- KV cache per token: 2 × 12 layers × 12 heads × 64 × 2 B = 36,864 B ≈ 36 KiB.

**1. Why does prompt length affect prefill?**

Prefill runs every prompt token through every layer in one forward pass. The linear layers cost about 2 × (parameters) FLOPs per token, so their work grows linearly with prompt length *n*. Self-attention compares every token with every earlier token, so its work and its score matrix grow as *n*².

In principle, then, prefill latency should rise with *n*. In our measurements it barely does: the clean runs went from 7.5 ms at 128 tokens to 7.9 ms at 768 tokens. The work is simply too small to fill the GPU. A 768-token prefill is about 2 × 124.4M × 768 ≈ 1.9 × 10¹¹ FLOPs, which takes about **0.8 ms** at our measured 243 TFLOP/s. The measured time is about 7.6 ms. Roughly 90% of prefill time is therefore a fixed per-forward-pass cost: Python and Hugging Face module dispatch, plus launching a few hundred small kernels across 12 layers. That cost does not depend on *n*.

The slightly higher time at 512 tokens (about 9.4 ms) than at 768 (about 7.9 ms) is also consistent with an overhead-dominated regime. The library selects different GEMM/attention kernels for different sequence shapes, and at this scale those per-shape differences outweigh the extra FLOPs.

Prompt length *does* clearly show up in memory, through the tensors prefill creates. Moving from 128 to 768 tokens raises peak memory by 87.2 MiB (269.8 → 357.0 MiB at output 32). That matches the prediction for 640 extra tokens:
- full-prompt logits: 640 × 50,257 vocab × 2 B ≈ 61.4 MiB;
- KV cache: 640 × 36 KiB ≈ 22.5 MiB;
- total predicted: 83.9 MiB.

The same per-token estimate (≈ 0.131 MiB per prompt token) also predicts the 128 → 256 step (16.8 MiB predicted, 17.0 MiB measured).

**2. Why does output length affect decode?**

Autoregressive decoding is sequential. Token *t* + 1 depends on token *t*, so *m* output tokens require *m* forward passes that cannot run in parallel. Decode latency is therefore proportional to *m*. Our data fits this almost perfectly: decode latency = 5.64 ms × *m* − 1.5 ms, with R² = 0.9998. The four prompt-length lines in the decode plot lie on top of each other.

**3. Why can prefill expose more GPU parallelism than decode?**

In prefill, the activations entering each layer form an *n* × 768 matrix, so every weight matrix multiplies against *n* rows at once. The GPU can spread those rows across thousands of cores and Tensor Cores, and the weights are loaded from memory once and reused *n* times.

In decode, the input is a single 1 × 768 row. Each layer becomes a matrix-vector product with very little independent work per kernel. Most of the GPU sits idle, and each weight is used for only one token before the next step reloads it.

Our numbers show this directly. Processing **768 tokens in one prefill takes about 7.6 ms, while generating one token takes 5.6 ms**. Per token, prefill costs about 0.010 ms versus 5.6 ms for decode, roughly **570× cheaper**.

**4. Which phase appears more compute intensive?**

**Prefill.** Its arithmetic intensity (FLOPs per byte loaded) is high. Each FP16 weight (2 B) is used in about 2*n* FLOPs, so intensity is about *n* FLOP/byte. That is 128–768 FLOP/byte for our prompts, at or above the A100's balance point of about 312 TFLOP/s ÷ 2.0 TB/s ≈ 150 FLOP/byte. Prefill FLOPs also grow with *n* (and *n*² for attention), while decode does almost no arithmetic per step: about 2.5 × 10⁸ FLOPs, roughly 1 µs of work at the A100's measured throughput.

In our gpt2 runs, prefill latency is not yet *dominated* by compute, because only about 0.8 ms of its 7.6 ms is arithmetic. But it is the phase whose cost would scale with compute for a larger model or a longer prompt.

**5. Which phase appears more memory-bandwidth sensitive?**

**Decode.** Every decode step must read all the weights (about 249 MB) plus the whole KV cache, only to produce one token. Its arithmetic intensity is about 1 FLOP/byte, far below the A100's balance point. That puts decode firmly in the memory-bound regime: the speed limit is how fast bytes come from HBM, not how fast the GPU computes.

For gpt2 specifically, even this bound is small. 249 MB at 2.0 TB/s ≈ **0.12 ms**, yet our measured TPOT is 5.6 ms, about 45× higher. The remaining time is fixed per-step overhead (Python, Hugging Face module dispatch, and launching hundreds of small kernels per step). Our measurement is therefore *overhead-bound*, which is a worse version of the same problem: the GPU does very little useful work per step.

Decode is the phase sensitive to anything that adds bytes per step (larger weights, longer KV cache). Prefill is limited by arithmetic and does not have this problem. Batching (Part 3) is the standard fix: it reuses each weight read across many sequences.

**6. Does TPOT remain constant as the context grows? If not, why?**

**In our measurements, yes, to within noise.** TPOT stays between 5.53 and 5.74 ms (±2%) across every configuration. There is no consistent trend with prompt length: averaged over output lengths it is 5.59, 5.69, 5.64 and 5.58 ms for prompts of 128, 256, 512 and 768. Nor is there a trend with output length: 5.63, 5.62, 5.62 and 5.64 ms for outputs of 32, 64, 128 and 256.

**In principle, TPOT should grow with context.** At decode step *t*, attention has to read the K and V of all *n* + *t* earlier tokens, so the bytes read and the attention FLOPs per step grow linearly with context length.

The effect is invisible here because it is tiny compared with the fixed overhead. At the largest context we ran (1024 tokens), the KV cache is 1024 × 36 KiB ≈ 36 MiB. Reading it takes about 0.018 ms at 2.0 TB/s, about **0.3% of a 5.6 ms step**, which is below our run-to-run variation (std ≈ 0.01–0.09 ms).

For a larger model, long contexts, or large batches, the KV read per step becomes comparable to or larger than the weight read, and TPOT rises roughly linearly with context. That effect is what makes long-context serving expensive (see Part 2).

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
