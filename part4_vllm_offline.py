"""
This file demonstrates a minimal vLLM comparison.

Install separately:
    pip install vllm

Note:
Offline llm.generate() is convenient for throughput experiments, but a true
request-arrival / TTFT experiment is better performed with the vLLM server
and an asynchronous client. For this assignment, students may use offline
generation unless the instructor requires online arrival-time measurements.
"""

import argparse
import json
import random
import time

import pandas as pd
from vllm import LLM, SamplingParams


def load_workload(path):
    with open(path) as f:
        return json.load(f)


def make_prompt(n_words):
    # This approximates a prompt of the desired scale but is not guaranteed
    # to tokenize to exactly n_words tokens.
    return "hello " * n_words


def make_token_prompts(workload, vocab_size, seed):
    rng = random.Random(seed)
    return [
        {
            "prompt_token_ids": [
                rng.randrange(vocab_size) for _ in range(r["prompt_length"])
            ]
        }
        for r in workload
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--workload", default="workload.json")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--exact-workload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use exact prompt token counts and per-request output lengths",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="part4_vllm.csv")
    args = parser.parse_args()

    workload = load_workload(args.workload)

    prompts = [
        make_prompt(r["prompt_length"])
        for r in workload
    ]

    llm = LLM(model=args.model)

    # Simple common max-token setting for an offline comparison.
    # TODO 4:
    # Explain why this does not perfectly reproduce heterogeneous output lengths.
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
    )

    vocab_size = len(llm.get_tokenizer())

    if args.exact_workload:
        prompts = make_token_prompts(workload, vocab_size, seed=0)
        sampling_params = [
            SamplingParams(
                temperature=0,
                max_tokens=r["output_length"],
                ignore_eos=True,
            )
            for r in workload
        ]

    warmup_params = (
        sampling_params[:4]
        if isinstance(sampling_params, list)
        else sampling_params
    )
    llm.generate(prompts[:4], warmup_params, use_tqdm=False)

    start = time.perf_counter()

    outputs = llm.generate(
        prompts,
        sampling_params,
    )

    end = time.perf_counter()

    total_time = end - start

    output_tokens = sum(
        len(output.outputs[0].token_ids)
        for output in outputs
    )

    print(f"Requests/s: {len(prompts) / total_time:.3f}")
    print(f"Output tokens/s: {output_tokens / total_time:.3f}")
    print(f"Total time: {total_time:.3f} s")

    rows = []
    for rep in range(args.repeats):
        if args.exact_workload:
            prompts = make_token_prompts(workload, vocab_size, seed=rep + 1)

        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        total_time = time.perf_counter() - start

        output_tokens = sum(
            len(output.outputs[0].token_ids)
            for output in outputs
        )
        prompt_tokens = sum(
            len(output.prompt_token_ids)
            for output in outputs
        )

        rows.append({
            "engine": "vllm",
            "exact_workload": args.exact_workload,
            "num_requests": len(prompts),
            "total_time_sec": total_time,
            "requests_per_sec": len(prompts) / total_time,
            "output_tokens_per_sec": output_tokens / total_time,
            "output_tokens": output_tokens,
            "prompt_tokens": prompt_tokens,
            "rep": rep,
        })
        print(
            f"rep={rep}: {rows[-1]['requests_per_sec']:.3f} requests/s, "
            f"{rows[-1]['output_tokens_per_sec']:.1f} output tokens/s"
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    numeric = df.drop(columns=["engine", "exact_workload", "rep"])
    summary = numeric.agg(["mean", "std"])
    print(summary.T.to_string())
    summary.loc[["mean"]].to_csv(
        args.out.removesuffix(".csv") + "_mean.csv", index=False
    )
    summary.loc[["std"]].to_csv(
        args.out.removesuffix(".csv") + "_std.csv", index=False
    )


if __name__ == "__main__":
    main()
