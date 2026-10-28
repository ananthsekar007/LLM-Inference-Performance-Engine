import json
import random
import argparse


def generate_workload(
    n_requests=50,
    seed=42,
    arrival_rate=5.0,
    prompt_lengths=(32, 64, 128, 256, 512),
    output_lengths=(16, 32, 64, 128, 256),
):
    random.seed(seed)
    workload = []
    current_time = 0.0

    for request_id in range(n_requests):
        # Poisson-arrival workload using exponential inter-arrival times.
        current_time += random.expovariate(arrival_rate)

        workload.append(
            {
                "request_id": request_id,
                "arrival_time": current_time,
                "prompt_length": random.choice(prompt_lengths),
                "output_length": random.choice(output_lengths),
            }
        )

    return workload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arrival-rate", type=float, default=5.0)
    parser.add_argument("--out", type=str, default="workload.json")
    args = parser.parse_args()

    workload = generate_workload(
        n_requests=args.n,
        seed=args.seed,
        arrival_rate=args.arrival_rate,
    )

    with open(args.out, "w") as f:
        json.dump(workload, f, indent=2)

    print(f"Wrote {len(workload)} requests to {args.out}")
