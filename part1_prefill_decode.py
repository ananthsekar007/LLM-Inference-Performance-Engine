import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    device,
    peak_memory_gb,
    reset_peak_memory,
    seed_everything,
    synthetic_tokens,
    time_cuda_safe,
)


@torch.no_grad()
def benchmark_case(model, prompt_length, output_length, dev):
    input_ids = synthetic_tokens(
        batch_size=1,
        seq_len=prompt_length,
        vocab_size=model.config.vocab_size,
        dev=dev,
    )

    reset_peak_memory()

    # -----------------------------
    # PREFILL
    # -----------------------------
    def prefill():
        return model(input_ids=input_ids, use_cache=True)

    outputs, prefill_sec = time_cuda_safe(prefill)

    past = outputs.past_key_values
    next_token = torch.argmax(
        outputs.logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    # -----------------------------
    # DECODE
    # -----------------------------
    def decode():
        nonlocal past, next_token

        for _ in range(output_length):
            outputs = model(
                input_ids=next_token,
                past_key_values=past,
                use_cache=True,
            )
            past = outputs.past_key_values
            next_token = torch.argmax(
                outputs.logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

    _, decode_sec = time_cuda_safe(decode)

    return {
        "prompt_length": prompt_length,
        "output_length": output_length,
        "prefill_ms": prefill_sec * 1000,
        "decode_ms": decode_sec * 1000,
        "total_ms": (prefill_sec + decode_sec) * 1000,
        "tpot_ms": decode_sec * 1000 / output_length,
        "peak_memory_gb": peak_memory_gb(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--out", default="part1_prefill_decode.csv")
    args = parser.parse_args()

    seed_everything(42)
    dev = device()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if dev == "cuda" else torch.float32,
    ).to(dev)
    model.eval()

    # Warm-up
    benchmark_case(model, 32, 4, dev)

    rows = []
    for n in [128, 256, 512, 768]:
        for m in [32, 64, 128, 256]:
            print(f"Running prompt={n}, output={m}")
            rows.append(benchmark_case(model, n, m, dev))
            rows[-1]["rep"] = 0
            for rep in range(1, 3):
                print(f"Running prompt={n}, output={m}, rep={rep}")
                rows.append(benchmark_case(model, n, m, dev))
                rows[-1]["rep"] = rep

    df = pd.DataFrame(rows)
    print(df)
    df.to_csv(args.out, index=False)

    group_cols = ["prompt_length", "output_length"]
    mean_df = df.groupby(group_cols).mean().reset_index().drop(columns=["rep"])
    std_df = df.groupby(group_cols).std().reset_index().drop(columns=["rep"])
    print(mean_df)
    mean_df.to_csv(args.out.removesuffix(".csv") + "_mean.csv", index=False)
    std_df.to_csv(args.out.removesuffix(".csv") + "_std.csv", index=False)


if __name__ == "__main__":
    main()
