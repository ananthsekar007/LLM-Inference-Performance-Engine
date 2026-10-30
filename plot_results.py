import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_lines(df, x, y, group, xlabel, ylabel, title, filename):
    plt.figure()
    for value, subset in df.groupby(group):
        subset = subset.sort_values(x)
        plt.plot(
            subset[x],
            subset[y],
            marker="o",
            label=f"{group}={value}",
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{filename}.pdf", bbox_inches="tight")
    plt.savefig(f"{filename}.png", dpi=150, bbox_inches="tight")
    plt.close()


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

    plot_lines(
        df,
        x="prompt_length",
        y="prefill_ms",
        group="output_length",
        xlabel="Prompt Length (tokens)",
        ylabel="Prefill Latency (ms)",
        title="Prompt Length vs Prefill Latency",
        filename="part1_prefill_vs_prompt",
    )
    plot_lines(
        df,
        x="output_length",
        y="decode_ms",
        group="prompt_length",
        xlabel="Output Length (tokens)",
        ylabel="Decode Latency (ms)",
        title="Output Length vs Decode Latency",
        filename="part1_decode_vs_output",
    )
    plot_lines(
        df,
        x="output_length",
        y="tpot_ms",
        group="prompt_length",
        xlabel="Output Length (tokens)",
        ylabel="TPOT (ms/token)",
        title="Output Length vs TPOT",
        filename="part1_tpot_vs_output",
    )
    plot_lines(
        df,
        x="prompt_length",
        y="tpot_ms",
        group="output_length",
        xlabel="Prompt Length (tokens)",
        ylabel="TPOT (ms/token)",
        title="Prompt Length vs TPOT",
        filename="part1_tpot_vs_prompt",
    )


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

    batch_size = int(df["batch_size"].iloc[0]) if "batch_size" in df else 1
    metrics = [
        ("latency_sec", "Generation Latency (s)", "latency"),
        ("ms_per_token", "Time per Decode Step (ms)", "ms_per_token"),
        ("peak_memory_gb", "Peak GPU Memory (GB)", "memory"),
    ]

    for column, ylabel, name in metrics:
        plot_lines(
            df,
            x="output_length",
            y=column,
            group="use_cache",
            xlabel="Output Length (tokens)",
            ylabel=ylabel,
            title=f"KV Cache On vs Off: {ylabel} (batch={batch_size})",
            filename=f"part2_{name}_b{batch_size}",
        )


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

    log_metrics = [
        ("tokens_per_sec", "Output Tokens / second", "part3_throughput"),
        ("requests_per_sec", "Requests / second", "part3_requests"),
        ("latency_sec", "End-to-End Latency (s)", "part3_latency"),
        ("ttft_ms", "TTFT Approximation (ms)", "part3_ttft"),
        ("tpot_ms", "TPOT (ms/token)", "part3_tpot"),
        ("peak_memory_gb", "Peak GPU Memory (GB)", "part3_memory"),
    ]

    for column, ylabel, filename in log_metrics:
        plt.figure()
        plt.plot(
            df["batch_size"],
            df[column],
            marker="o",
        )
        plt.xscale("log", base=2)
        plt.xticks(df["batch_size"], [str(b) for b in df["batch_size"]])
        plt.ylim(bottom=0)
        plt.grid(alpha=0.3)
        plt.xlabel("Batch Size (log scale)")
        plt.ylabel(ylabel)
        plt.title(f"Batch Size vs {ylabel}")
        plt.savefig(f"{filename}.pdf", bbox_inches="tight")
        plt.savefig(f"{filename}.png", dpi=150, bbox_inches="tight")
        plt.close()

    plt.figure()
    plt.plot(
        df["latency_sec"],
        df["tokens_per_sec"],
        marker="o",
    )
    for _, row in df.iterrows():
        if row["batch_size"] in (2, 4, 8):
            continue
        plt.annotate(
            f"b={int(row['batch_size'])}",
            (row["latency_sec"], row["tokens_per_sec"]),
            textcoords="offset points",
            xytext=(6, -10),
            fontsize=8,
        )
    plt.ylim(bottom=0)
    plt.grid(alpha=0.3)
    plt.xlabel("End-to-End Latency (s)")
    plt.ylabel("Output Tokens / second")
    plt.title("Throughput vs Latency Tradeoff")
    plt.savefig("part3_tradeoff.pdf", bbox_inches="tight")
    plt.savefig("part3_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_bars(labels, values, ylabel, title, filename, colors):
    plt.figure()
    bars = plt.bar(labels, values, color=colors)
    plt.bar_label(
        bars,
        labels=[f"{v:,.0f}" if v >= 100 else f"{v:.2f}" for v in values],
        fontsize=8,
    )
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(f"{filename}.pdf", bbox_inches="tight")
    plt.savefig(f"{filename}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_part4(results_dir):
    static = pd.concat(
        pd.read_csv(f"{results_dir}/part4_static_b{b}_mean.csv")
        for b in [1, 4, 8, 16]
    )
    vllm = pd.read_csv(f"{results_dir}/part4_vllm_mean.csv").iloc[0]

    labels = [f"static b={int(b)}" for b in static["batch_size"]] + ["vLLM"]
    colors = ["tab:blue"] * len(static) + ["tab:orange"]

    plot_bars(
        labels,
        list(static["output_tokens_per_sec"]) + [vllm["output_tokens_per_sec"]],
        "Output Tokens / second",
        "Static Batching vs vLLM: Output Throughput",
        "part4_throughput",
        colors,
    )
    plot_bars(
        labels,
        list(static["requests_per_sec"]) + [vllm["requests_per_sec"]],
        "Requests / second",
        "Static Batching vs vLLM: Request Throughput",
        "part4_requests",
        colors,
    )
    plot_bars(
        labels,
        list(static["total_time_sec"]) + [vllm["total_time_sec"]],
        "Time to Finish All 50 Requests (s)",
        "Static Batching vs vLLM: Workload Completion Time",
        "part4_total_time",
        colors,
    )

    plt.figure()
    x = range(len(static))
    width = 0.38
    plt.bar(
        [i - width / 2 for i in x],
        static["prompt_padding_efficiency"],
        width,
        label="prompt padding efficiency",
    )
    plt.bar(
        [i + width / 2 for i in x],
        static["output_efficiency"],
        width,
        label="output (decode) efficiency",
    )
    plt.xticks(list(x), [f"b={int(b)}" for b in static["batch_size"]])
    plt.ylim(0, 1.05)
    plt.ylabel("Useful / Processed Tokens")
    plt.xlabel("Static Batch Size")
    plt.title("Static Batching Efficiency")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.savefig("part4_efficiency.pdf", bbox_inches="tight")
    plt.savefig("part4_efficiency.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(
        static["batch_size"],
        static["mean_completion_sec"],
        marker="o",
        label="mean completion time",
    )
    plt.plot(
        static["batch_size"],
        static["p95_completion_sec"],
        marker="o",
        label="P95 completion time",
    )
    plt.axhline(
        vllm["total_time_sec"],
        color="tab:orange",
        linestyle="--",
        label="vLLM: all requests done",
    )
    plt.xscale("log", base=2)
    plt.xticks(static["batch_size"], [str(int(b)) for b in static["batch_size"]])
    plt.ylim(bottom=0)
    plt.xlabel("Static Batch Size (log scale)")
    plt.ylabel("Time Since Start of Run (s)")
    plt.title("Request Completion Time (all requests ready at t=0)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("part4_completion.pdf", bbox_inches="tight")
    plt.savefig("part4_completion.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part1")
    parser.add_argument("--part2")
    parser.add_argument("--part3")
    parser.add_argument("--part4", help="Directory with Part 4 result CSVs")
    args = parser.parse_args()

    if args.part1:
        plot_part1(args.part1)

    if args.part2:
        plot_part2(args.part2)

    if args.part3:
        plot_part3(args.part3)

    if args.part4:
        plot_part4(args.part4)


if __name__ == "__main__":
    main()
