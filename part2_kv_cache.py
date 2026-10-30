"""
Part 2: KV Cache Benchmark

Study the provided `generate_with_cache()` implementation, then complete
`generate_without_cache()` by filling in the three TODO blocks.

The no-cache version must:
  * pass the entire sequence to the model at each iteration;
  * use `use_cache=False`;
  * append each generated token to the sequence.

Everything else (the cached generator, the benchmark harness, and the
agreement check) is provided. When your implementation is correct, the
agreement check in main() should report that the cached and uncached
outputs match.

Run with:
    python part2_kv_cache.py --model gpt2
"""

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
def generate_with_cache(model, input_ids, output_length):
    """
    Autoregressive generation WITH a KV cache.

    The model processes the full prompt once (prefill).
    For subsequent decoding steps, it processes only
    the newest token while reusing cached keys and values.

    Returns:
        Complete sequence: prompt + generated tokens.
    """
    if output_length < 0:
        raise ValueError("output_length must be nonnegative")

    if output_length == 0:
        return input_ids

    # Step 1: Prefill the KV cache with the prompt
    outputs = model(
        input_ids=input_ids,
        use_cache=True,
    )

    past = outputs.past_key_values

    # Step 2: Predict the first new token
    next_token = torch.argmax(
        outputs.logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    generated = [next_token]

    # Step 3: Generate the remaining tokens
    for _ in range(1, output_length):

        # Only process the newest token
        outputs = model(
            input_ids=next_token,
            past_key_values=past,
            use_cache=True,
        )

        # Update the KV cache
        past = outputs.past_key_values

        # Predict the next token
        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        generated.append(next_token)

    # Return prompt + generated tokens
    return torch.cat([input_ids] + generated, dim=1)


@torch.no_grad()
def generate_without_cache(model, input_ids, output_length):
    """
    Autoregressive generation WITHOUT a KV cache.

    At every decoding step:
      1. Process the entire current sequence.
      2. Predict the next token.
      3. Append the predicted token to the sequence.

    Returns:
        Complete sequence: prompt + generated tokens.
    """
    if output_length < 0:
        raise ValueError("output_length must be nonnegative")

    sequence = input_ids

    for _ in range(output_length):

        # --------------------------------------------------------
        # TODO (a): Forward pass over the ENTIRE current sequence.
        #
        # Unlike generate_with_cache(), there is no cache to reuse,
        # so every step must re-process all tokens in `sequence`.
        # Be sure to pass `use_cache=False`.
        # --------------------------------------------------------
        outputs = model(
            input_ids=sequence,
            use_cache=False,
        )

        # --------------------------------------------------------
        # TODO (b): Greedily predict the next token.
        #
        # Take the logits at the last position and argmax over the
        # vocabulary. Keep the result shaped [batch, 1] so it can be
        # concatenated to `sequence` (see generate_with_cache()).
        # --------------------------------------------------------
        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        # --------------------------------------------------------
        # TODO (c): Append the new token to the complete sequence
        # along the sequence dimension (dim=1).
        # --------------------------------------------------------
        sequence = torch.cat([sequence, next_token], dim=1)

    return sequence


def benchmark(model, input_ids, output_length, use_cache):
    """
    Benchmark autoregressive generation.

    Measures:
      - Total generation latency
      - Average latency per generated token
      - Peak memory usage

    Returns:
        (result_dict, generated_sequence)
    """

    prompt_length = input_ids.shape[1]

    # Select the appropriate generation function
    fn = (
        generate_with_cache
        if use_cache
        else generate_without_cache
    )

    # Reset peak memory statistics before measurement
    reset_peak_memory()

    # Measure execution time
    sequence, elapsed = time_cuda_safe(
        lambda: fn(model, input_ids, output_length)
    )

    # Record results
    result = {
        "use_cache": use_cache,
        "prompt_length": prompt_length,
        "output_length": output_length,
        "latency_sec": elapsed,
        "ms_per_token": elapsed * 1000 / output_length,
        "peak_memory_gb": peak_memory_gb(),
    }

    return result, sequence


def agreement_check(seq_cached, seq_uncached, prompt_length):
    """
    Compare the tokens generated with and without the KV cache.

    Only the generated portion is compared (the prompt is shared).

    Returns:
        exact_match (bool): True if every generated token is identical.
        match_rate (float): Fraction of positions where the tokens agree.
        first_mismatch (int or None): Index (relative to the first
            generated token) of the first divergence, or None if all
            tokens match.
    """
    gen_cached = seq_cached[:, prompt_length:]
    gen_uncached = seq_uncached[:, prompt_length:]

    if gen_cached.shape != gen_uncached.shape:
        raise RuntimeError(
            f"Generated lengths differ: "
            f"{gen_cached.shape} vs {gen_uncached.shape}"
        )

    matches = gen_cached == gen_uncached
    exact_match = bool(matches.all())
    match_rate = matches.float().mean().item()

    first_mismatch = None
    if not exact_match:
        first_mismatch = int((~matches[0]).nonzero()[0].item())

    return exact_match, match_rate, first_mismatch


def main():

    parser = argparse.ArgumentParser(
        description="Benchmark LLM generation with and without KV cache"
    )

    parser.add_argument(
        "--model",
        default="gpt2",
        help="Hugging Face model name",
    )

    parser.add_argument(
        "--out",
        default="part2_kv_cache.csv",
        help="Output CSV file",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of sequences generated together",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Measured runs per configuration",
    )

    args = parser.parse_args()

    # Reproducibility
    seed_everything(42)

    # Choose CPU or GPU
    dev = device()

    print(f"Using device: {dev}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=(
            torch.float16
            if torch.device(dev).type == "cuda"
            else torch.float32
        ),
    ).to(dev)

    model.eval()

    # -----------------------------
    # Warm-up
    # -----------------------------

    print("Warming up model...")

    warmup_ids = synthetic_tokens(
        batch_size=1,
        seq_len=32,
        vocab_size=model.config.vocab_size,
        dev=dev,
    )

    # Warm up both generation methods
    benchmark(
        model=model,
        input_ids=warmup_ids,
        output_length=4,
        use_cache=True,
    )

    benchmark(
        model=model,
        input_ids=warmup_ids,
        output_length=4,
        use_cache=False,
    )

    # -----------------------------
    # Benchmark experiments
    # -----------------------------

    rows = []

    prompt_length = 128

    # Use identical input tokens for both cache settings
    input_ids = synthetic_tokens(
        batch_size=1,
        seq_len=prompt_length,
        vocab_size=model.config.vocab_size,
        dev=dev,
    )

    if args.batch_size > 1:
        input_ids = synthetic_tokens(
            batch_size=args.batch_size,
            seq_len=prompt_length,
            vocab_size=model.config.vocab_size,
            dev=dev,
        )

    for output_length in [32, 128, 256, 512, 896]:

        sequences = {}

        for use_cache in [True, False]:

            print(
                f"output_length={output_length}, "
                f"use_cache={use_cache}"
            )

            result, sequence = benchmark(
                model=model,
                input_ids=input_ids,
                output_length=output_length,
                use_cache=use_cache,
            )

            rows.append(result)
            sequences[use_cache] = sequence
            rows[-1]["rep"] = 0

            for rep in range(1, args.repeats):
                print(
                    f"output_length={output_length}, "
                    f"use_cache={use_cache}, rep={rep}"
                )

                result, _ = benchmark(
                    model=model,
                    input_ids=input_ids,
                    output_length=output_length,
                    use_cache=use_cache,
                )

                result["rep"] = rep
                rows.append(result)

        # -----------------------------
        # Agreement check
        # -----------------------------

        exact_match, match_rate, first_mismatch = agreement_check(
            sequences[True],
            sequences[False],
            prompt_length,
        )

        if exact_match:
            print(
                f"  Agreement check: PASS "
                f"(all {output_length} generated tokens identical)"
            )
        else:
            print(
                f"  Agreement check: tokens diverge at generated "
                f"position {first_mismatch} "
                f"(match rate {match_rate:.1%}). "
                f"Small divergences are expected in fp16 due to "
                f"accumulated rounding differences."
            )

        # Record the check alongside the benchmark rows
        for row in rows[-2:]:
            row["exact_match"] = exact_match
            row["match_rate"] = match_rate

        for row in rows[-2 * args.repeats:]:
            row["batch_size"] = args.batch_size
            row["exact_match"] = exact_match
            row["match_rate"] = match_rate

    # -----------------------------
    # Save benchmark results
    # -----------------------------

    df = pd.DataFrame(rows)

    print("\nBenchmark results:")
    print(df.to_string(index=False))

    df.to_csv(args.out, index=False)

    print(f"\nResults saved to {args.out}")

    group_cols = ["batch_size", "use_cache", "prompt_length", "output_length"]
    mean_df = (
        df.groupby(group_cols)
        .mean(numeric_only=True)
        .reset_index()
        .drop(columns=["rep"])
    )
    std_df = (
        df.groupby(group_cols)
        .std(numeric_only=True)
        .reset_index()
        .drop(columns=["rep"])
    )

    print("\nMean over repeats:")
    print(mean_df.to_string(index=False))

    mean_df.to_csv(args.out.removesuffix(".csv") + "_mean.csv", index=False)
    std_df.to_csv(args.out.removesuffix(".csv") + "_std.csv", index=False)


if __name__ == "__main__":
    main()
