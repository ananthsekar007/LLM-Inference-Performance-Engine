# CSE 598 — LLM Serving Systems Assignment

**Author:** Ananth Sundararajan Sekar (ASU ID: 1237228176)

This project measures the systems behaviour of autoregressive LLM serving: prefill vs. decode, KV caching, batching, and static batching vs. a modern serving engine (vLLM). The report is `Ananth_Sundararajan_Sekar_CSE598_Assignment3_Report.docx`.

| | |
|---|---|
| Model | `gpt2` (124M parameters, 1024-token context), FP16 everywhere |
| Hardware | One NVIDIA A100-SXM4-80GB on ASU Sol (full environment in `results/env.txt`) |
| Software | Python 3.11.18, PyTorch 2.13.0+cu130, Transformers 5.19.0, vLLM 0.32.0 |
| Methodology | One warm-up before measuring, model loading excluded, 3 repetitions per configuration, mean reported |

## Files

| Path | Purpose |
|---|---|
| `setup.sh` | Creates and activates the `llmserve` conda env on Sol; optionally records the environment |
| `utils.py` | Shared helpers: seeding, GPU-synchronized timing, peak-memory tracking, synthetic tokens |
| `part1_prefill_decode.py` | Part 1: prefill/decode latency, TPOT and memory over prompt × output lengths |
| `part2_kv_cache.py` | Part 2: generation with vs. without the KV cache (`--batch-size`, `--repeats`) |
| `part2_kv_size.py` | Part 2: analytical KV-cache size calculator |
| `part3_batching.py` | Part 3: batch-size sweep (1–512) |
| `workload.py` | Part 4: generates the 50-request heterogeneous workload (`workload.json`, seed 42) |
| `part4_static_batching.py` | Part 4: Hugging Face static batching over the workload |
| `part4_vllm_offline.py` | Part 4: vLLM offline engine over the same workload |
| `plot_results.py` | Generates the per-part plots in `plots/` |
| `results/` | All CSV results (raw per run, `_mean`, `_std`) and `env.txt` |
| `plots/` | All figures (PDF and PNG) |

## How to run everything

These steps assume you have the project as a zip file and run everything on ASU Sol in an **interactive shell on a GPU node**. Replace the placeholders:
- `<ASURITE>`: your ASU username;
- `<project-zip>`: the name of the zip file;
- `<project-folder>`: the folder you unzip it into.

`results/` and `plots/` already contain the results reported in the report. Re-running an experiment overwrites the matching files.

### Step 1: Copy the project to Sol (your computer)

```bash
scp <project-zip>.zip <ASURITE>@login.sol.rc.asu.edu:~/
```

You can also upload the zip through the Sol web portal (https://sol.asu.edu, Files menu).

### Step 2: Log in and start a persistent session (your computer → Sol login node)

```bash
ssh <ASURITE>@login.sol.rc.asu.edu
tmux new -s llm
```

`tmux` keeps the session alive if the connection drops. Reattach later with `tmux attach -t llm`.

### Step 3: Unzip the project (Sol login node)

```bash
unzip <project-zip>.zip -d <project-folder>
```

`<project-folder>` should directly contain `setup.sh` and the `part*.py` scripts. If the zip has its own top-level folder, use that folder as `<project-folder>` in the steps below.

### Step 4: Request a GPU interactively (Sol login node)

```bash
salloc -G a100:1
```

This allocates one A100 and opens a shell on a GPU node. Check the GPU:

```bash
nvidia-smi
```

### Step 5: Set up the environment (GPU node)

```bash
cd ~/<project-folder>
source setup.sh --env-info
```

**`setup.sh` must be sourced before running anything else, in every new shell on the GPU node.** Every command in Steps 6–12 assumes the environment it sets up.

It must be **sourced**, not run with `bash`, so the activated environment stays in your shell. It:
- loads `mamba/latest`;
- points the Hugging Face, pip and conda caches at `/scratch/$USER`;
- creates the `llmserve` env with Python 3.11 if it doesn't exist, then activates it;
- installs vLLM first (it pins its own `torch`/`transformers`), then `accelerate`, `pandas`, `numpy` and `matplotlib`;
- installs `ninja` and loads the CUDA module `cuda-13.4.3-gcc-13.2.0` so `nvcc` is available, then sets `CUDA_HOME`. vLLM needs these in Part 4 (see Step 10). To use a different module, run `CUDA_MODULE=<name> source setup.sh`;
- pre-downloads gpt2 and checks that CUDA sees the GPU.

The first run creates the env and installs the packages, which takes several minutes. Later runs skip anything already installed, so in later sessions just use `source setup.sh`.

`--env-info` also writes `nvidia-smi` plus the Python/PyTorch/CUDA/cuDNN/Transformers/vLLM versions to `results/env.txt`. It only needs to be used once.

### Step 6: Part 0, timing check for Question 0.1 (GPU node)

Run in `python`:

```python
import torch, time
a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
for _ in range(3): a @ a
torch.cuda.synchronize()

s = time.perf_counter(); b = a @ a; e = time.perf_counter()
print("no sync  (ms):", (e - s) * 1000)
torch.cuda.synchronize()

torch.cuda.synchronize(); s = time.perf_counter(); b = a @ a; torch.cuda.synchronize(); e = time.perf_counter()
print("with sync (ms):", (e - s) * 1000)
```

Reported result: 0.084 ms without synchronization, 4.528 ms with it.

### Step 7: Part 1, prefill and decode (GPU node, about 2 min)

```bash
python part1_prefill_decode.py --out results/part1_prefill_decode.csv
```

Prompts 128/256/512/768 × outputs 32/64/128/256, 3 repetitions each. Writes `results/part1_prefill_decode.csv` (raw), `_mean.csv` and `_std.csv`.

### Step 8: Part 2, KV cache (GPU node)

Batch 1 takes a few minutes; batch 32 takes about 10 minutes:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python part2_kv_cache.py --batch-size 1 --out results/part2_kv_cache_b1.csv
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python part2_kv_cache.py --batch-size 32 --out results/part2_kv_cache_b32.csv
```

`expandable_segments:True` is needed at batch 32. Without the cache, every step allocates slightly larger tensors, which fragments PyTorch's caching allocator until it hits OOM and retries, adding time to the measurements.

`results/part2_kv_cache.csv` is an earlier single-run smoke test that is not used in the report. It was produced by `python part2_kv_cache.py --out results/part2_kv_cache.csv`.

KV-cache size checks (no GPU needed):

```bash
python part2_kv_size.py --layers 12 --kv-heads 12 --head-dim 64 --seq-len 1024
python part2_kv_size.py --batch-size 32 --layers 12 --kv-heads 12 --head-dim 64 --seq-len 1024
python part2_kv_size.py
```

These are gpt2 at 1 and 32 sequences (36 MiB and 1.125 GiB), and the script's default Llama-8B-style configuration (512 MiB).

### Step 9: Part 3, batching (GPU node, a few minutes)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python part3_batching.py --out results/part3_batching.csv
```

Batch sizes 1, 2, 4, …, 512 with prompt 128 and output 128, 3 repetitions each.

### Step 10: Part 4, static batching vs. vLLM (GPU node)

Generate the workload (deterministic, seed 42):

```bash
python workload.py --n 50
```

Static batching at four batch sizes:

```bash
for b in 1 4 8 16; do PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python part4_static_batching.py --batch-size $b --out results/part4_static_b$b.csv; done
```

vLLM needs a compiler on Sol. At startup, FlashInfer compiles its sampling kernel and fails with `Could not find nvcc` unless the CUDA toolkit and `ninja` are available. `setup.sh` (Step 5) takes care of this. Before running vLLM, confirm that `nvcc` is on the path:

```bash
which nvcc && echo $CUDA_HOME
```

If it prints nothing, pick a module from `module avail cuda` and re-source `setup.sh` with it:

```bash
CUDA_MODULE=<cuda module name> source setup.sh
```

Then run vLLM on the exact workload, and once more with the original starter settings for the TODO 4 comparison:

```bash
python part4_vllm_offline.py --out results/part4_vllm.csv
python part4_vllm_offline.py --no-exact-workload --out results/part4_vllm_starter.csv
```

- **Exact workload (default):** exact-length token prompts, per-request `max_tokens` with `ignore_eos=True`, and `dtype=float16`. vLLM's default would load gpt2 in bfloat16.
- **Starter settings:** `--no-exact-workload` uses `"hello "` prompts and a single `max_tokens=64`.
- **Check:** confirm the vLLM startup log shows `dtype=torch.float16`.

### Step 11: Generate the plots (GPU node or login node)

`plot_results.py` saves into the current directory, so run it from `plots/`:

```bash
cd plots
python ../plot_results.py --part1 ../results/part1_prefill_decode_mean.csv
python ../plot_results.py --part2 ../results/part2_kv_cache_b1_mean.csv
python ../plot_results.py --part2 ../results/part2_kv_cache_b32_mean.csv
python ../plot_results.py --part3 ../results/part3_batching_mean.csv
python ../plot_results.py --part4 ../results
cd ..
```

### Step 12 (optional): Copy results to your computer (your computer, not Sol)

```bash
scp -r <ASURITE>@login.sol.rc.asu.edu:~/<project-folder>/results <ASURITE>@login.sol.rc.asu.edu:~/<project-folder>/plots .
```

`scp` has to be run on your own computer. Running it from inside Sol tries to SSH back into Sol and hangs.

## Results and figures

| Part | Results | Figures |
|---|---|---|
| 0 | `results/env.txt` | — |
| 1 | `results/part1_prefill_decode{,_mean,_std}.csv` | `plots/part1_*` |
| 2 | `results/part2_kv_cache_b{1,32}{,_mean,_std}.csv` | `plots/part2_*` |
| 3 | `results/part3_batching{,_mean,_std}.csv` | `plots/part3_*`, `plots/batch_vs_*.pdf` |
| 4 | `results/part4_static_b{1,4,8,16}{,_mean,_std}.csv`, `results/part4_vllm{,_starter}{,_mean,_std}.csv`, `workload.json` | `plots/part4_*` |

## Changes to the starter code

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
