from __future__ import annotations
import os
import time
from pathlib import Path
import psutil
import numpy as np


def benchmark_inference(fn, *args, n_runs: int = 20, **kwargs) -> dict:
    process = psutil.Process(os.getpid())
    times = []
    mem_before = process.memory_info().rss / (1024**2)
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append((time.perf_counter() - t0) * 1000)
    mem_after = process.memory_info().rss / (1024**2)
    return {
        "latency_ms_mean": float(np.mean(times)),
        "latency_ms_p50": float(np.percentile(times, 50)),
        "latency_ms_p95": float(np.percentile(times, 95)),
        "ram_mb_delta": float(mem_after - mem_before),
        "cpu_only": True,
    }


def file_size_mb(path: str | Path) -> float:
    p = Path(path)
    return p.stat().st_size / (1024**2) if p.exists() else float("nan")
