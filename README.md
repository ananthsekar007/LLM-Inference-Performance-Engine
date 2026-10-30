# CSE 598 — LLM Serving Systems Assignment

**Author:** Ananth Sundararajan Sekar

This repository measures the systems behaviour of autoregressive LLM serving: prefill vs. decode, KV caching, batching, and static batching vs. a modern serving engine (vLLM). The analysis and results are in [`REPORT.md`](REPORT.md).

| | |
|---|---|
| Model | `gpt2` (124M parameters, 1024-token context), FP16 everywhere |
| Hardware | One NVIDIA A100-SXM4-80GB on ASU Sol (full environment in [`results/env.txt`](results/env.txt)) |
| Software | Python 3.11.18, PyTorch 2.13.0+cu130, Transformers 5.19.0, vLLM 0.32.0 |
| Methodology | One warm-up before measuring, model loading excluded, 3 repetitions per configuration, mean ± std reported |

## Repository layout

| Path | Purpose |
|---|---|
| `setup.sh` | Creates and activates the `llmserve` conda env on Sol; optionally records the environment |
| `utils.py` | Shared helpers: seeding, GPU-synchronized timing, peak-memory tracking, synthetic tokens |
| `part1_prefill_decode.py` | Part 1: prefill / decode latency, TPOT and memory over prompt × output lengths |
| `part2_kv_cache.py` | Part 2: generation with vs. without the KV cache (`--batch-size`, `--repeats`) |
| `part2_kv_size.py` | Part 2: analytical KV-cache size calculator |
| `part3_batching.py` | Part 3: batch-size sweep (1–512): TTFT, TPOT, latency, throughput, memory |
| `workload.py` | Part 4: generates the 50-request heterogeneous workload (`workload.json`, seed 42) |
| `part4_static_batching.py` | Part 4: Hugging Face static batching over the workload |
| `part4_vllm_offline.py` | Part 4: vLLM offline engine over the same workload |
| `plot_results.py` | Generates every plot from the CSVs |
| `results/` | All CSV results (raw per run, `_mean`, `_std`) and `env.txt` |
| `plots/` | All figures (PDF, and PNG for the report) |
| `REPORT.md` | The report |

## 1. Environment setup (ASU Sol)

All experiments were run **interactively** on a Sol GPU node.

```bash
ssh <asurite>@login.sol.rc.asu.edu
tmux new -s llm
interactive -p general -q public -G a100:1 -c 8 --mem=32G -t 0-03:00
cd ~/Desktop/LLM-Inference-Performance-Engine
source setup.sh --env-info
```

`setup.sh` must be **sourced**, not executed, so that the activated environment stays in your shell. It:
- loads `mamba/latest`;
- points the Hugging Face, pip and conda caches at `/scratch/$USER`;
- creates `llmserve` with Python 3.11 if it doesn't exist, then activates it;
- installs vLLM first (it pins its own `torch` and `transformers`), then the remaining packages;
- pre-downloads gpt2 and checks that CUDA sees a GPU.

With `--env-info` it also writes `results/env.txt`. On later sessions, `source setup.sh` only re-activates the env.

To install the packages without the script, install vLLM first and then the rest:

```bash
pip install vllm
pip install -r requirements.txt
```

### Extra setup for vLLM (Part 4 only)

vLLM 0.32 compiles a FlashInfer sampling kernel at startup, which needs `nvcc` and `ninja`. Without them it fails with `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`. Fix:

```bash
module avail cuda
module load <cuda module closest to 13.0>
pip install ninja
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
```

## 2. Running the experiments

Run every command from the repository root on the GPU node. Each script writes a raw CSV (one row per repetition) plus `_mean.csv` and `_std.csv` next to it.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is used for Parts 2–4. Without it, the growing tensors in no-cache and large-batch runs fragment PyTorch's caching allocator, which then hits OOM and retries, distorting the timings.

### Part 0: environment and timing check

```bash
source setup.sh --env-info
```

The synchronize-vs-no-synchronize matmul timing used for Question 0.1 is in `REPORT.md`, Part 0.

### Part 1: prefill and decode (about 2 min)

```bash
python part1_prefill_decode.py --out results/part1_prefill_decode.csv
```

Prompt lengths are 128/256/512/768 and output lengths 32/64/128/256 (prompt + output ≤ 1024, gpt2's limit). Each configuration is repeated 3 times.

### Part 2: KV cache (batch 1: a few minutes; batch 32: about 10 min)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python part2_kv_cache.py --batch-size 1 --out results/part2_kv_cache_b1.csv
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python part2_kv_cache.py --batch-size 32 --out results/part2_kv_cache_b32.csv
```

KV-cache size checks (no GPU needed):

```bash
python part2_kv_size.py --layers 12 --kv-heads 12 --head-dim 64 --seq-len 1024
python part2_kv_size.py --batch-size 32 --layers 12 --kv-heads 12 --head-dim 64 --seq-len 1024
python part2_kv_size.py
```

The first two use gpt2's configuration. The third uses the script's defaults, a Llama-8B-style configuration.

`results/part2_kv_cache.csv` is an earlier single-run smoke test (output lengths 32/128/256) and is not used in the report tables.

### Part 3: batching (a few minutes)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python part3_batching.py --out results/part3_batching.csv
```

Batch sizes 1, 2, 4, …, 512 with prompt 128 and output 128.

### Part 4: static batching vs. vLLM (a few minutes)

```bash
python workload.py --n 50

for b in 1 4 8 16; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python part4_static_batching.py --batch-size $b --out results/part4_static_b$b.csv
done

python part4_vllm_offline.py --out results/part4_vllm.csv
python part4_vllm_offline.py --no-exact-workload --out results/part4_vllm_starter.csv
```

- **`workload.json`** is deterministic (seed 42): 11,040 prompt tokens and 4,592 output tokens. Prompts are capped at 512 so that every request fits gpt2's context.
- **`part4_vllm_offline.py`** defaults to `--exact-workload`: exact-length token prompts and per-request `max_tokens` with `ignore_eos=True`, so vLLM does the same work as static batching. It also forces `--dtype float16`, because vLLM's `auto` dtype would load gpt2 in bfloat16. Each repetition uses fresh random prompts so prefix caching cannot reuse earlier ones.
- **`--no-exact-workload`** reproduces the original starter configuration (`"hello "` prompts, a single `max_tokens=64`). It is used only for the TODO 4 comparison in the report.

## 3. Generating the plots

`plot_results.py` saves figures into the current directory, so run it from `plots/`:

```bash
cd plots
python ../plot_results.py --part1 ../results/part1_prefill_decode_mean.csv
python ../plot_results.py --part2 ../results/part2_kv_cache_b1_mean.csv
python ../plot_results.py --part2 ../results/part2_kv_cache_b32_mean.csv
python ../plot_results.py --part3 ../results/part3_batching_mean.csv
python ../plot_results.py --part4 ../results
cd ..
```

`--part2` takes one CSV per call; the batch size is read from the file and added to the plot filenames. The starter's `part2_kv_latency.pdf` has a fixed name and is overwritten by the last call. `--part4` takes the results directory.

## 4. Results and figures

| Part | Results | Figures |
|---|---|---|
| 0 | `results/env.txt` | — |
| 1 | `results/part1_prefill_decode{,_mean,_std}.csv` | `plots/part1_*` |
| 2 | `results/part2_kv_cache_b{1,32}{,_mean,_std}.csv` | `plots/part2_*` |
| 3 | `results/part3_batching{,_mean,_std}.csv` | `plots/part3_*`, `plots/batch_vs_*.pdf` |
| 4 | `results/part4_static_b{1,4,8,16}{,_mean,_std}.csv`, `results/part4_vllm{,_starter}{,_mean,_std}.csv`, `workload.json` | `plots/part4_*` |

## 5. Changes to the starter code

All changes are additions, except for a few one-line edits noted here.

- **`part1_prefill_decode.py`:** prompt/output lengths changed to fit gpt2's 1024-token context *(edited line)*; 3 repetitions and mean/std CSVs added.
- **`part2_kv_cache.py`:**
  - the `generate_without_cache()` TODOs completed;
  - `--batch-size` and `--repeats` options added;
  - output lengths extended to 512 and 896 *(edited line)*;
  - mean/std CSVs added;
  - `agreement_check()` fixed for batches where only a later row diverges.
- **`part3_batching.py`:** batch sizes extended to 512 *(edited line)*; `--repeats` option and mean/std CSVs added.
- **`part4_static_batching.py`:**
  - warm-up added;
  - `min_new_tokens` added so batches cannot stop early on EOS;
  - `run_static_workload()` added, with output tokens/s, output efficiency and completion latency;
  - repeats and CSV output added.
- **`part4_vllm_offline.py`:**
  - `--exact-workload`, `--repeats`, `--out` and `--dtype` options added; dtype is now passed to `LLM()` *(edited line)*;
  - warm-up and CSV output added.
- **`workload.py`:** prompt lengths capped at 512 *(edited line)*.
- **`plot_results.py`:** Part 1–3 plots added, plus a new `--part4` option.
