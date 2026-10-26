import argparse


def kv_cache_size_bytes(
    batch_size,
    num_layers,
    seq_len,
    num_kv_heads,
    head_dim,
    bytes_per_element,
):
    """
    KV cache memory:

        batch_size
        * num_layers
        * seq_len
        * num_kv_heads
        * head_dim
        * 2      # keys + values
        * bytes_per_element
    """
    return (
        batch_size
        * num_layers
        * seq_len
        * num_kv_heads
        * head_dim
        * 2
        * bytes_per_element
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--bytes", type=int, default=2)
    args = parser.parse_args()

    size = kv_cache_size_bytes(
        args.batch_size,
        args.layers,
        args.seq_len,
        args.kv_heads,
        args.head_dim,
        args.bytes,
    )

    print(f"Bytes: {size}")
    print(f"MiB:   {size / 1024**2:.2f}")
    print(f"GiB:   {size / 1024**3:.4f}")


if __name__ == "__main__":
    main()
