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

Interactive sessions on Sol were allocated to different compute nodes (Part 1 on `sg029`, Part 2 on `sg046`). Every node used had the same GPU model (A100-SXM4-80GB), driver and software stack, and each comparison within a part was run in a single session on a single GPU.

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

`generate_without_cache()` repeats three steps for each output token:

1. Run the model over the **entire** current sequence (prompt plus every token generated so far) with `use_cache=False`, so no keys or values are reused from earlier steps.
2. Take the logits at the last position and choose the next token greedily with `argmax(..., keepdim=True)`, which keeps the shape `[batch, 1]`.
3. Append that token to the sequence with `torch.cat(..., dim=1)`.

**Agreement check.**
- At batch size 1, cached and uncached generation produced **identical tokens** at every output length (match rate 1.0).
- At batch size 32, **exactly 2 generated tokens differ** at every output length, out of 32 × *m* (up to 28,672), the first at generated position 2 (match rate ≥ 0.998).
  - The cause is FP16 rounding, not a logic error. The cached path multiplies one new token against stored K/V, while the uncached path recomputes the full sequence. The two use different kernels and summation orders, so a near-tie between two candidate tokens can flip.
  - With greedy decoding on random prompts, gpt2 falls quickly into a repetitive pattern, so the affected row rejoins the same token sequence after the flip.

**Two fixes were needed to run batched experiments.**
1. **`agreement_check()` crash.** As provided, it looked for the first mismatch only in row 0, so it raised `IndexError` when row 0 matched but another row diverged. It now reports the earliest mismatching position across all rows in that case.
2. **Allocator fragmentation.** At batch 32 without the cache, every step allocates slightly larger tensors than the step before. PyTorch's caching allocator kept reserving new blocks until all 80 GB was reserved for only a few GB of live data. It then hit OOM, flushed its cache and retried, which added time to the measurements. Both batch sizes were therefore run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

### Experimental parameters

| Parameter | Value |
|---|---|
| Prompt length | 128 (random tokens, fixed seed) |
| Output lengths | 32, 128, 256, 512, 896 (128 + 896 = 1024, gpt2's maximum) |
| Batch sizes | 1 and 32 (separate runs) |
| Repetitions | 3 per configuration, after one warm-up of each method |
| Precision | FP16 |
| Allocator | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Command | `python part2_kv_cache.py --batch-size {1,32} --out results/part2_kv_cache_b{1,32}.csv` |
| Output | `results/part2_kv_cache_b{1,32}.csv` (raw), `_mean.csv`, `_std.csv` |

**Metric definitions.**
- **Latency** is the whole generation call: prefill plus all decode steps.
- **ms per step** is latency ÷ `output_length`. At batch 32 each step produces 32 tokens, one per sequence.
- **Peak memory** is `torch.cuda.max_memory_allocated()` and includes the 0.232 GiB of FP16 weights.

### Results (Task 2.2)

**Batch size 1** (mean ± std over 3 runs)

| Output | Cache | Latency (s) | ms per step | Peak memory (GiB) | Match rate |
|---:|:---:|---:|---:|---:|---:|
| 32 | on | 0.178 ± 0.001 | 5.56 ± 0.03 | 0.256 | 1.00000 |
| 32 | off | 0.177 ± 0.002 | 5.53 ± 0.05 | 0.270 | 1.00000 |
| 128 | on | 0.709 ± 0.002 | 5.54 ± 0.02 | 0.256 | 1.00000 |
| 128 | off | 0.703 ± 0.001 | 5.49 ± 0.01 | 0.288 | 1.00000 |
| 256 | on | 1.443 ± 0.006 | 5.64 ± 0.02 | 0.256 | 1.00000 |
| 256 | off | 1.422 ± 0.014 | 5.56 ± 0.06 | 0.312 | 1.00000 |
| 512 | on | 2.854 ± 0.004 | 5.57 ± 0.01 | 0.263 | 1.00000 |
| 512 | off | 2.917 ± 0.015 | 5.70 ± 0.03 | 0.360 | 1.00000 |
| 896 | on | 5.024 ± 0.016 | 5.61 ± 0.02 | 0.277 | 1.00000 |
| 896 | off | 5.308 ± 0.021 | 5.92 ± 0.02 | 0.433 | 1.00000 |

**Batch size 32** (mean ± std over 3 runs)

| Output | Cache | Latency (s) | ms per step | Peak memory (GiB) | Match rate |
|---:|:---:|---:|---:|---:|---:|
| 32 | on | 0.188 ± 0.000 | 5.89 ± 0.00 | 0.771 | 0.99805 |
| 32 | off | 0.472 ± 0.006 | 14.76 ± 0.18 | 1.197 | 0.99805 |
| 128 | on | 0.745 ± 0.004 | 5.82 ± 0.03 | 0.771 | 0.99951 |
| 128 | off | 2.443 ± 0.014 | 19.09 ± 0.11 | 1.776 | 0.99951 |
| 256 | on | 1.488 ± 0.007 | 5.81 ± 0.03 | 0.771 | 0.99976 |
| 256 | off | 6.434 ± 0.017 | 25.13 ± 0.07 | 2.549 | 0.99976 |
| 512 | on | 2.963 ± 0.010 | 5.79 ± 0.02 | 0.975 | 0.99988 |
| 512 | off | 19.045 ± 0.004 | 37.20 ± 0.01 | 4.095 | 0.99988 |
| 896 | on | 5.400 ± 0.006 | 6.03 ± 0.01 | 1.415 | 0.99993 |
| 896 | off | 49.702 ± 0.017 | 55.47 ± 0.02 | 6.413 | 0.99993 |

**Slowdown from disabling the cache** (latency without cache ÷ latency with cache):

| Output length | 32 | 128 | 256 | 512 | 896 |
|---|---:|---:|---:|---:|---:|
| Batch 1 | 0.99× | 0.99× | 0.99× | 1.02× | 1.06× |
| Batch 32 | 2.5× | 3.3× | 4.3× | 6.4× | **9.2×** |

**Key observations.**
- **Batch 32 without the cache grows quadratically.** Total latency fits *a·m + b·m²* with **R² = 0.9999995**, while a straight line fits with R² = 0.962. Dividing latency by the number of token positions the model actually processes (32 × Σₜ (128 + *t*)) gives a nearly constant **3.0–3.2 µs per token** at every output length. The extra time is accounted for entirely by recomputation.
- **With the cache, cost per step is flat** at 5.8–6.0 ms (batch 32) and 5.5–5.6 ms (batch 1), regardless of length.
- **At batch 1 the two methods are indistinguishable** until 896 tokens (1.06×). As in Part 1, a single-sequence forward pass is dominated by a fixed ~5.5 ms of Python and kernel-launch overhead, which hides the recomputation.

![Batch 32: latency, cache on vs off](plots/part2_latency_b32.png)

![Batch 32: time per decode step, cache on vs off](plots/part2_ms_per_token_b32.png)

![Batch 32: peak memory, cache on vs off](plots/part2_memory_b32.png)

![Batch 1: latency, cache on vs off](plots/part2_latency_b1.png)

![Batch 1: time per decode step, cache on vs off](plots/part2_ms_per_token_b1.png)

![Batch 1: peak memory, cache on vs off](plots/part2_memory_b1.png)

### KV-cache memory derivation (Task 2.3)

**Derivation.** In each attention layer, every token's hidden state is projected to a key vector and a value vector for each of the *H_kv* KV heads, each of dimension *D*. For *B* sequences of cached length *S*, one layer therefore stores:

- a key tensor of shape [*B*, *H_kv*, *S*, *D*], i.e. *B·H_kv·S·D* elements;
- a value tensor of the same shape, another *B·H_kv·S·D* elements.

Summing over *L* layers and multiplying by *b* bytes per element:

$$
\text{KV bytes} \;=\; 2 \cdot B \cdot L \cdot S \cdot H_{kv} \cdot D \cdot b
$$

**The factor of 2** is there because attention needs both keys and values of every earlier token, and they are separate tensors with identical shapes. Keys are compared with the current query to produce attention weights, and values are the vectors those weights average. Queries are *not* cached: each token's query is used only at the step that produced it.

With grouped-query attention, *H_kv* is smaller than the number of query heads, which shrinks the cache proportionally. gpt2 uses standard multi-head attention, so *H_kv* = 12.

**Per-token cost:**
- gpt2: 2 × 12 × 12 × 64 × 2 B = 36,864 B = **36 KiB per token per sequence**.
- Llama-8B-style configuration: 2 × 32 × 8 × 128 × 2 B = 131,072 B = **128 KiB per token**.

**Checks with `part2_kv_size.py`:**

| Configuration | Command | Result |
|---|---|---|
| gpt2, *B* = 1, *S* = 1024 | `python part2_kv_size.py --layers 12 --kv-heads 12 --head-dim 64 --seq-len 1024` | 37,748,736 B = **36.00 MiB** |
| gpt2, *B* = 32, *S* = 1024 | `... --batch-size 32` | 1,207,959,552 B = **1.125 GiB** |
| Llama-8B-style (*L* = 32, *H_kv* = 8, *D* = 128, *S* = 4096), *B* = 1 | `python part2_kv_size.py` (defaults) | 536,870,912 B = **512 MiB** |

**Agreement with measured memory.** At batch 32 with 896 output tokens, the cached sequence reaches 1024 tokens. The formula predicts 1.125 GiB of KV cache. Measured peak memory was 1.415 GiB, or 1.183 GiB after subtracting the 0.232 GiB of weights. That is within about 5% of the prediction; the small remainder is temporaries such as the copy made when each layer's cache grows by concatenation.

The cached memory curve is also explained by the formula. Up to 256 output tokens, peak memory stays at 0.771 GiB because the peak is set by **prefill**: the prompt's logits (32 × 128 × 50,257 × 2 B = 0.38 GiB) plus the weights and the 128-token cache (0.14 GiB). Those logits are freed after the first step. The cache grows by 32 × 36 KiB = 1.125 MiB per generated token, so it needs about 350 more tokens (roughly 480 in total) to exceed that earlier peak. After that point, memory rises in proportion to *S*.

### Analysis (Task 2.4)

**1. Why does caching reduce redundant computation?**

In a causal Transformer, the key and value of token *i* depend only on tokens 1…*i*. Once computed, they never change, no matter how many tokens follow.

Without a cache, every decode step recomputes K and V (and the full layer stack) for every earlier token. Generating *m* tokens after an *n*-token prompt therefore processes Σₜ (*n* + *t*) ≈ *nm* + *m*²/2 token positions instead of *n* + *m*. For *n* = 128 and *m* = 896, that is **515,648 positions per sequence versus 1,024**, about 500× more work. With a cache, each step processes only the newest token and reads the stored K/V of the rest.

Our measurements confirm this. At batch 32:
- no-cache latency grows quadratically (R² = 0.9999995) and costs a constant ~3 µs per processed token position;
- cached latency grows linearly at about 5.9 ms per step;
- disabling the cache is **9.2× slower at 896 tokens**, and the gap widens with length.

The measured slowdown is far below 500× for two reasons. Recomputation is done in large, efficient prefill-style passes, and every cached step still pays the fixed ~5.5 ms per-forward overhead. At batch 1 that overhead hides the redundant work almost completely (0.99–1.06×). The recomputation is still happening, but it is cheap relative to the per-step overhead until the batch and sequence are large.

**2. Why does cache memory grow with sequence length?**

The cache must keep K and V for *every* token the model may still attend to. Each new token adds one key and one value per layer per KV head, so cache size is linear in *S*: 2·*L*·*H_kv*·*D*·*b* bytes per token (36 KiB for gpt2) for each sequence.

We observed this directly. At batch 32, cached peak memory rose from 0.77 to 1.41 GiB as the sequence grew to 1024 tokens, within about 5% of the 1.125 GiB the formula predicts. Memory without the cache grew even faster (1.20 → 6.41 GiB), because each step re-creates logits for every position (3.06 GiB at 1023 tokens). Here the cache saves memory as well as time.

**3. Why does long-context serving reduce concurrency?**

GPU memory is fixed. After the weights (and activation workspace), the remainder is a budget for KV caches, and each active request needs 2·*L*·*H_kv*·*D*·*b*·*S* bytes. The maximum number of concurrent requests is therefore roughly:

> KV budget ÷ (bytes per token × context length)

so it falls in **inverse proportion to context length**.

For the Llama-8B-style configuration on our 80 GB A100:
- the weights take about 16 GB, leaving roughly 60 GB for KV;
- each 4096-token request needs 512 MiB, allowing about **110 concurrent requests**;
- at 32k tokens each request needs 4 GiB, allowing about **14**.

Even gpt2's tiny cache reaches 1.125 GiB for 32 sequences of 1024 tokens. Fewer concurrent requests means smaller decode batches, which (Part 3) means lower throughput.

**4. Why can a serving system become memory-limited even when compute is available?**

Decode does very little arithmetic per step. It is limited by overhead and memory bandwidth, not FLOPs (Part 1). Batching is how a serving system raises GPU utilization, and our data shows how cheap extra sequences are in compute terms: a cached decode step at **batch 32 took 5.89 ms versus 5.56 ms at batch 1, 32× the tokens for 6% more time**. In principle the server could keep adding requests to the batch.

Every additional request, however, brings its own KV cache, which grows as the request generates. The batch can only grow until the KV caches fill the memory budget. Past that point the system must queue or reject new requests, or evict existing ones, even though the Tensor Cores are mostly idle. Memory capacity, not compute, sets the limit.

This gets worse when memory is wasted:
- reserving each request's maximum possible context up front;
- fragmentation from caches of different, growing sizes.

We saw an extreme case of fragmentation in the no-cache batch-32 run. PyTorch's allocator had reserved all 80 GB while only a few GB were in use, and it hit OOM. Mechanisms such as paged KV-cache management (Part 4) exist to reduce exactly this waste.

---

## Part 3. Batching

### What is asked

- **Task 3.1** — For several batch sizes, measure TTFT (approximation), TPOT, end-to-end latency, requests/s, output tokens/s, and peak GPU memory.
- **Task 3.2** — Plots: batch size vs. tokens/s; batch size vs. latency; batch size vs. peak memory.
- **Task 3.3** — Answer the five analysis questions below.

### Experimental parameters

| Parameter | Value |
|---|---|
| Batch sizes | 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 |
| Prompt length | 128 (random tokens; identical length for every request) |
| Output length | 128 |
| Repetitions | 3 per batch size, after one warm-up (batch 1, prompt 32, output 4) |
| Precision | FP16 |
| Allocator | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Command | `python part3_batching.py --out results/part3_batching.csv` |
| Output | `results/part3_batching.csv` (raw), `_mean.csv`, `_std.csv` |

The starter script stopped at batch 16. We extended it to 512 because, at 16, throughput was still rising in proportion to batch size, so the point of diminishing returns (Q3) would not have been visible. Batch 512 used 11.1 GiB, so the run never hit the script's out-of-memory guard.

### Metric definitions (TODO 2 in the starter code)

All *B* requests in a batch start together, have the same prompt and output lengths, and therefore finish together. Each formula relies on that.

| Metric | Formula | Meaning and caveats |
|---|---|---|
| `ttft_ms` | prefill time × 1000 | Time for one forward pass over all *B* × 128 prompt tokens, which produces every request's first token. It is only an **approximation** of TTFT. A real TTFT is measured from when a request *arrives* until its first token reaches the user, so it also includes time queued waiting for a batch, scheduling, tokenization and network delay. None of those exist here, because every request is ready at time 0. |
| `tpot_ms` | decode time × 1000 ÷ `output_length` | Average time per decode step. Each step produces one token for *every* request in the batch, so this is also the inter-token latency each individual user sees. |
| `latency_sec` | prefill + decode | End-to-end latency of every request, since they all start and finish together. |
| `requests_per_sec` | *B* ÷ `latency_sec` | Correct only because the whole batch finishes at once. With mixed output lengths, requests would finish at different times and the batch would be held until the longest one finished (Part 4). |
| `tokens_per_sec` | *B* × `output_length` ÷ `latency_sec` | Aggregate output throughput. Prompt tokens are not counted, but prefill time *is* in the denominator, as a user would experience. Prefill also produces one token per request before the `output_length` decode steps, so each request actually generates `output_length` + 1 tokens. The formula therefore undercounts by 1/129 (< 1%). |
| `peak_memory_gb` | `torch.cuda.max_memory_allocated()` | Includes the 0.232 GiB of weights. The prefill `outputs` (full prompt logits) stay referenced during decode in this script, so they are part of the peak. |

### Task 3.1 — Results (mean ± std over 3 runs)

| Batch | TTFT approx. (ms) | TPOT (ms) | Latency (s) | Requests/s | Output tokens/s | Peak memory (GiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.7 ± 1.0 | 5.49 ± 0.03 | 0.711 ± 0.003 | 1.4 | 180 ± 1 | 0.261 |
| 2 | 7.6 ± 0.2 | 6.03 ± 0.72 | 0.779 ± 0.092 | 2.6 | 331 ± 37 | 0.282 |
| 4 | 8.8 ± 0.1 | 5.56 ± 0.00 | 0.721 ± 0.001 | 5.6 | 711 ± 1 | 0.325 |
| 8 | 11.6 ± 2.9 | 5.54 ± 0.01 | 0.721 ± 0.004 | 11.1 | 1,420 ± 8 | 0.410 |
| 16 | 15.6 ± 2.8 | 5.51 ± 0.03 | 0.721 ± 0.006 | 22.2 | 2,840 ± 25 | 0.580 |
| 32 | 15.3 ± 0.2 | 5.78 ± 0.03 | 0.755 ± 0.003 | 42.4 | 5,425 ± 25 | 0.919 |
| 64 | 28.5 ± 0.7 | 5.81 ± 0.03 | 0.772 ± 0.004 | 82.9 | 10,612 ± 55 | 1.599 |
| 128 | 56.4 ± 4.4 | 6.44 ± 0.02 | 0.881 ± 0.007 | 145.2 | 18,590 ± 139 | 2.958 |
| 256 | 104.1 ± 0.1 | 11.41 ± 0.00 | 1.565 ± 0.001 | 163.6 | 20,941 ± 8 | 5.677 |
| 512 | 205.5 ± 0.0 | 21.57 ± 0.01 | 2.966 ± 0.001 | 172.6 | 22,092 ± 6 | 11.114 |

The larger standard deviations at batch 2 (TPOT) and at batches 8–16 (TTFT) come from the first repetition at each new batch shape, as in Part 1. At batch 2, rep 0 had a TPOT of 6.86 ms versus 5.63 and 5.59 ms for reps 1–2. Likewise, batch 16's TTFT is 14.1 ms without its first run, so the apparent dip at batch 32 is not a real effect.

**Scaling summary.**

| Batch | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Throughput gain over the previous batch size | — | 1.84× | 2.14× | 2.00× | 2.00× | 1.91× | 1.96× | 1.75× | **1.13×** | **1.05×** |
| Throughput ÷ (batch × batch-1 throughput) | 1.00 | 0.92 | 0.99 | 0.99 | 0.99 | 0.94 | 0.92 | 0.81 | 0.45 | 0.24 |

- Peak memory grows linearly at about **21.7 MiB per additional sequence**. That matches 256 tokens × 36 KiB of KV cache (9 MiB) plus the 128 × 50,257 × 2 B prompt logits kept from prefill (12.3 MiB), a total of 21.3 MiB.
- At batch 512, the prefill pass processes 65,536 tokens in 205 ms. That is about 320k tokens/s, or about 79 TFLOP/s for the linear layers alone.

### Task 3.2 — Plots

Batch size is shown on a log₂ axis so that every doubling is evenly spaced. The starter code's linear-axis versions are also in `plots/` (`batch_vs_throughput.pdf`, `batch_vs_latency.pdf`, `batch_vs_memory.pdf`).

![Batch size vs throughput](plots/part3_throughput.png)

![Batch size vs end-to-end latency](plots/part3_latency.png)

![Batch size vs peak memory](plots/part3_memory.png)

![Batch size vs TTFT](plots/part3_ttft.png)

![Batch size vs TPOT](plots/part3_tpot.png)

![Throughput vs latency tradeoff](plots/part3_tradeoff.png)

### Analysis

**1. Why can larger batches improve GPU utilization?**

A decode step has two kinds of cost:
- a **fixed** cost paid once per step whatever the batch: reading all the weights from HBM, launching hundreds of kernels, and Python/Hugging Face dispatch;
- a **per-sequence** cost: the arithmetic for that sequence's token and reading its own KV cache.

At batch 1, the fixed cost is almost everything (Parts 1–2): each weight is loaded to do one multiply-add, and the Tensor Cores sit idle. Batching processes *B* sequences in the same step. Every weight read and every kernel launch is shared by *B* tokens, and each layer becomes a *B*-row matrix multiply instead of a matrix-vector product, so arithmetic intensity rises roughly in proportion to *B*.

Our data shows this clearly. From batch 1 to batch 64, TPOT rose only from 5.49 to 5.81 ms, while each step produced 64× more tokens. Throughput went from **180 to 10,612 tokens/s, 59× for a 64× batch** (92% of perfectly linear scaling).

**2. Why does throughput not increase linearly forever?**

Linear scaling lasts only while the fixed cost dominates. The per-sequence cost grows with *B*. A sequence's K/V cannot be shared with other sequences, so the KV bytes read per step grow as *B* × context length. At batch 512, the cache is 2.25–4.5 GiB during decode, and it is read every step (and partly re-copied as Hugging Face's cache grows by concatenation). Matmul FLOPs also grow with *B*.

Once that per-sequence work exceeds the fixed cost, step time grows with *B*. TPOT doubles each time the batch doubles beyond 128 (6.44 → 11.41 → 21.57 ms), so tokens per second stops growing. Decode converges to a constant cost of about **42 µs per token**, and throughput saturates at about **22k tokens/s**, whether the limit is memory bandwidth or compute.

Prefill saturates even earlier. TTFT is flat at batches 1–2, then doubles with each doubling from 64 onward, because a 65,536-token prefill pass is fully compute-bound. Beyond these limits, memory capacity would eventually cap the batch outright: 21.7 MiB per sequence here, far more for a larger model or longer contexts (Part 2).

**3. Where do you observe diminishing returns?**

Between **batch 128 and batch 256**:
- every doubling up to 64 gave 1.9–2.1× more throughput, and 64 → 128 still gave 1.75×;
- **128 → 256 gave only 1.13×, and 256 → 512 only 1.05×**;
- relative to perfectly linear scaling, efficiency falls from 0.92 at 64 to 0.81 at 128, then 0.45 at 256 and 0.24 at 512.

The same knee shows in latency. TPOT first rises noticeably at 128 (6.44 ms), and end-to-end latency jumps from 0.88 s at 128 to 1.57 s at 256 and 2.97 s at 512. In the throughput-vs-latency plot, the curve turns from nearly vertical to nearly horizontal at batch 128.

**4. Why can latency increase even when throughput improves?**

Throughput is a **system-wide** rate: total tokens across all users ÷ time. Latency is what **each individual request** experiences. Batching raises the first by making each request share the GPU with more others, which can slow every one of them down:

- **TTFT** — all *B* prompts are prefilled in one pass before anyone gets a first token. Once prefill is compute-bound, that pass takes time proportional to *B*: 7.7 ms at batch 1, 205 ms at 512.
- **TPOT** — every decode step serves all *B* requests. Once the GPU is saturated, each step takes longer: 5.5 ms → 21.6 ms per token for every user.
- **Batch-level waiting** — in a static batch, every request is tied to the others: it starts when the batch starts and finishes when the batch finishes. In a live system, requests must also wait in a queue for a batch to be formed.

Between batch 128 and 512, throughput improved by only 19% (18,590 → 22,092 tokens/s), but every request's end-to-end latency grew **3.4×** (0.88 → 2.97 s). The system serves more tokens per second by making each user wait longer, which is consistent with Little's law (concurrency = throughput × latency).

**5. For an interactive chatbot, would you choose the max-throughput batch size? Why or why not?**

**No.** A chatbot user experiences TTFT (how long before text starts appearing) and TPOT (how fast it streams), not the aggregate tokens per second. The right choice is the **largest batch that still meets the latency targets, with headroom**, not the maximum-throughput point.

In our measurements, batch 512 maximizes throughput, but the last two doublings buy only 19% more throughput for 3.6× worse TTFT, 3.4× worse TPOT and 3.4× worse end-to-end latency. **Batch 128** gets 84% of the maximum throughput while keeping TPOT at 6.4 ms and TTFT at 56 ms. **Batch 64** is the conservative choice: about half the peak throughput, but essentially no latency cost compared with batch 1.

Running at the knee rather than at saturation also leaves headroom for real traffic. Arrivals are bursty, prompts are longer and varied (a long prompt in a large batch stalls everyone's decode), and queueing delay adds to TTFT. A server that is already saturated has no slack to absorb these, so tail latency grows quickly.

For gpt2 even batch 512 streams faster than people read (about 46 tokens/s per user). For a production-size model, however, every one of these latencies would be one to two orders of magnitude larger, and the same knee would decide whether SLOs are met.

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
