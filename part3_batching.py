import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM

from utils import (
    device,
    peak_memory_gb,
    reset_peak_memory,
    seed_everything,
    synthetic_tokens,
    time_cuda_safe,
)


@torch.no_grad()
def benchmark_batch(
    model,
    batch_size,
    prompt_length,
    output_length,
    dev,
):
    input_ids = synthetic_tokens(
        batch_size=batch_size,
        seq_len=prompt_length,
        vocab_size=model.config.vocab_size,
        dev=dev,
    )

    reset_peak_memory()

    # -----------------------------
    # TTFT approximation / prefill
    # -----------------------------
    outputs, prefill_sec = time_cuda_safe(
        lambda: model(
            input_ids=input_ids,
            use_cache=True,
        )
    )

    past = outputs.past_key_values

    next_token = torch.argmax(
        outputs.logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    # -----------------------------
    # Decode
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

    total_sec = prefill_sec + decode_sec

    # TODO 2:
    # Verify the metric formulas below and explain them in your report.
    requests_per_sec = batch_size / total_sec
    output_tokens = batch_size * output_length
    tokens_per_sec = output_tokens / total_sec
    tpot_ms = decode_sec * 1000 / output_length

    return {
        "batch_size": batch_size,
        "prompt_length": prompt_length,
        "output_length": output_length,
        "ttft_ms": prefill_sec * 1000,
        "tpot_ms": tpot_ms,
        "latency_sec": total_sec,
        "requests_per_sec": requests_per_sec,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_gb": peak_memory_gb(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--out", default="part3_batching.csv")
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--output-length", type=int, default=128)
    args = parser.parse_args()

    seed_everything(42)
    dev = device()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if dev == "cuda" else torch.float32,
    ).to(dev)
    model.eval()

    # Warm-up
    benchmark_batch(model, 1, 32, 4, dev)

    rows = []

    for batch_size in [1, 2, 4, 8, 16]:
        try:
            print(f"Running batch_size={batch_size}")
            rows.append(
                benchmark_batch(
                    model=model,
                    batch_size=batch_size,
                    prompt_length=args.prompt_length,
                    output_length=args.output_length,
                    dev=dev,
                )
            )
        except torch.cuda.OutOfMemoryError:
            print(f"OOM at batch_size={batch_size}; stopping.")
            break

    df = pd.DataFrame(rows)
    print(df)
    df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
