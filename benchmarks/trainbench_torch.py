#!/usr/bin/env python3
"""
#1662 lever #1 (§5d) PyTorch CPU baseline for the fused optimizer-in-backward proof.

Mirrors ConvParallelProbe's `--trainbench` shape EXACTLY (residual-FFN MLP stack:
per layer  h = h + W2( gelu( W1 h ) ),  scalar loss = sum(h*h),  SGD) so the two
runtimes can be diffed apples-to-apples on the metrics that are comparable across a
managed (.NET) and a native (libtorch) runtime: per-step wall time and peak process
RSS. (Per-step *managed* allocation is not comparable — torch allocates in C++ — so
it is not reported here; the AiDotNet side reports it separately as the GC-churn win.)

PyTorch does the classic collect-then-step: loss.backward() materializes the full
gradient set, then a separate SGD sweep updates every parameter. AiDotNet's streaming
path applies the optimizer to each gradient the moment it is produced and frees it
(optimizer-in-backward), so the comparison is exactly the architecture difference.

Usage:
  python trainbench_torch.py --s 128 --d 384 --layers 10 --reps 20 --threads 8
"""
import argparse
import statistics
import threading
import time

import psutil
import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=128)        # sequence length (rows)
    ap.add_argument("--d", type=int, default=384)        # model dim
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--threads", type=int, default=psutil.cpu_count(logical=True))
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    S, D, L = args.s, args.d, args.layers
    x = (torch.rand(S, D) - 0.5)  # fixed input, no grad
    w1 = [((torch.rand(D, 4 * D) - 0.5) * 0.02).requires_grad_(True) for _ in range(L)]
    w2 = [((torch.rand(4 * D, D) - 0.5) * 0.02).requires_grad_(True) for _ in range(L)]
    params = w1 + w2

    def step():
        for p in params:
            p.grad = None
        h = x
        for l in range(L):
            f = h @ w1[l]
            f = F.gelu(f)
            f = f @ w2[l]
            h = h + f
        loss = (h * h).sum()
        loss.backward()
        with torch.no_grad():
            for p in params:               # classic collect-then-step SGD sweep
                p -= args.lr * p.grad
        return float(loss.detach())

    for _ in range(args.warmup):
        step()

    proc = psutil.Process()
    peak_rss = proc.memory_info().rss
    stop = False

    def sampler():
        nonlocal peak_rss
        while not stop:
            rss = proc.memory_info().rss
            if rss > peak_rss:
                peak_rss = rss
            time.sleep(0.002)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()

    times = []
    last_loss = 0.0
    for _ in range(args.reps):
        t0 = time.perf_counter()
        last_loss = step()
        times.append((time.perf_counter() - t0) * 1000.0)

    stop = True
    t.join()
    times.sort()

    print(
        f"TRAINBENCH engine=torch block=mlp S={S} D={D} layers={L} threads={args.threads} "
        f"median_ms={times[len(times)//2]:.3f} min_ms={times[0]:.3f} "
        f"peak_rss_mb={peak_rss/(1024*1024):.1f} last_loss={last_loss:.3e}"
    )


if __name__ == "__main__":
    main()
