import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_part1(path):
    df = pd.read_csv(path)

    for output_length in sorted(df["output_length"].unique()):
        subset = df[df["output_length"] == output_length]

        plt.figure()
        plt.plot(
            subset["prompt_length"],
            subset["prefill_ms"],
            marker="o",
        )
        plt.xlabel("Prompt Length")
        plt.ylabel("Prefill Latency (ms)")
        plt.title(
            f"Prompt Length vs Prefill Latency "
            f"(Output={output_length})"
        )
        plt.savefig(
            f"part1_prefill_output_{output_length}.pdf",
            bbox_inches="tight",
        )
        plt.close()


def plot_part2(path):
    df = pd.read_csv(path)

    plt.figure()
    for cache_value in [True, False]:
        subset = df[df["use_cache"] == cache_value]

        plt.plot(
            subset["output_length"],
            subset["latency_sec"],
            marker="o",
            label=f"use_cache={cache_value}",
        )

    plt.xlabel("Output Length")
    plt.ylabel("Generation Latency (s)")
    plt.title("KV Cache vs Generation Latency")
    plt.legend()
    plt.savefig("part2_kv_latency.pdf", bbox_inches="tight")
    plt.close()


def plot_part3(path):
    df = pd.read_csv(path)

    metrics = [
        ("tokens_per_sec", "Tokens / second", "batch_vs_throughput.pdf"),
        ("latency_sec", "Latency (s)", "batch_vs_latency.pdf"),
        ("peak_memory_gb", "Peak GPU Memory (GB)", "batch_vs_memory.pdf"),
    ]

    for column, ylabel, filename in metrics:
        plt.figure()
        plt.plot(
            df["batch_size"],
            df[column],
            marker="o",
        )
        plt.xlabel("Batch Size")
        plt.ylabel(ylabel)
        plt.title(f"Batch Size vs {ylabel}")
        plt.savefig(filename, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part1")
    parser.add_argument("--part2")
    parser.add_argument("--part3")
    args = parser.parse_args()

    if args.part1:
        plot_part1(args.part1)

    if args.part2:
        plot_part2(args.part2)

    if args.part3:
        plot_part3(args.part3)


if __name__ == "__main__":
    main()
