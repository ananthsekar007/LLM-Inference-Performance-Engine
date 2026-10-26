import argparse
import json
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM

from utils import device, seed_everything, sync


def load_workload(path):
    with open(path) as f:
        return json.load(f)


@torch.no_grad()
def process_static_batch(model, batch, dev):
    batch_size = len(batch)

    max_prompt = max(r["prompt_length"] for r in batch)
    max_output = max(r["output_length"] for r in batch)

    input_ids = torch.zeros(
        (batch_size, max_prompt),
        dtype=torch.long,
        device=dev,
    )

    attention_mask = torch.zeros(
        (batch_size, max_prompt),
        dtype=torch.long,
        device=dev,
    )

    for i, request in enumerate(batch):
        length = request["prompt_length"]

        tokens = torch.randint(
            0,
            model.config.vocab_size,
            (length,),
            device=dev,
        )

        # Left pad to the longest prompt in this batch.
        input_ids[i, max_prompt - length :] = tokens
        attention_mask[i, max_prompt - length :] = 1

    sync()
    start = time.perf_counter()

    _ = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_output,
        do_sample=False,
        pad_token_id=0,
    )

    sync()
    end = time.perf_counter()

    return end - start, max_prompt, max_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--workload", default="workload.json")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    seed_everything(42)
    dev = device()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if dev == "cuda" else torch.float32,
    ).to(dev)
    model.eval()

    workload = load_workload(args.workload)

    request_latencies = []
    total_useful_prompt_tokens = 0
    total_padded_prompt_tokens = 0

    all_start = time.perf_counter()

    for start_idx in range(0, len(workload), args.batch_size):
        batch = workload[start_idx : start_idx + args.batch_size]

        batch_latency, max_prompt, max_output = process_static_batch(
            model, batch, dev
        )

        # Naive static-batch approximation:
        # every request in the batch experiences the same batch latency.
        request_latencies.extend([batch_latency] * len(batch))

        useful = sum(r["prompt_length"] for r in batch)
        padded = len(batch) * max_prompt

        total_useful_prompt_tokens += useful
        total_padded_prompt_tokens += padded

    all_end = time.perf_counter()

    total_time = all_end - all_start

    print(f"Throughput: {len(workload) / total_time:.3f} requests/s")
    print(f"Mean latency: {np.mean(request_latencies):.3f} s")
    print(f"P95 latency: {np.percentile(request_latencies, 95):.3f} s")

    padding_efficiency = (
        total_useful_prompt_tokens / total_padded_prompt_tokens
        if total_padded_prompt_tokens > 0
        else 1.0
    )

    print(f"Prompt-padding efficiency: {padding_efficiency:.3f}")

    # TODO 3:
    # Explain why max_output is also a source of inefficiency in naive batching.
    # How would the behavior change if requests could leave the batch immediately
    # after completing?


if __name__ == "__main__":
    main()
