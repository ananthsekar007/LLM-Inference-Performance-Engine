import time
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def time_cuda_safe(fn):
    sync()
    start = time.perf_counter()
    result = fn()
    sync()
    end = time.perf_counter()
    return result, end - start


def percentile(values, p):
    return float(np.percentile(values, p))


def synthetic_tokens(batch_size, seq_len, vocab_size, dev):
    return torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        device=dev,
        dtype=torch.long,
    )
