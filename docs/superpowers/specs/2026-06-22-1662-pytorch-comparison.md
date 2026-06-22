# #1662 lever #1 — PyTorch CPU comparison (§5d proof)

**Date:** 2026-06-22
**Shape:** residual-FFN MLP stack — S=128, D=384, 10 layers, SGD, 20 reps, 8 threads/cores.
**AiDotNet path:** `ConvParallelProbe --trainbench` (fwd → `ComputeGradientsStreaming` with the
optimizer applied per-gradient = optimizer-in-backward → arena recycling).
**PyTorch path:** `benchmarks/trainbench_torch.py` (classic `loss.backward()` + separate SGD sweep).

## Result (honest)

| Metric | PyTorch 2.11 CPU | AiDotNet | Verdict |
|---|---|---|---|
| median ms/step | **69.6** | 252.9 | **torch 3.6× faster** |
| min ms/step | 62.0 | 244.6 | torch |
| peak process RSS | **312 MB** | 489 MB | torch |
| per-step managed alloc | n/a (C++) | **0.127 MB** | AiDotNet (near-zero GC churn) |
| final loss | 1.727e3 | 1.726e3 | identical computation ✓ |

## Conclusion

**The fused optimizer-in-backward does NOT make AiDotNet beat PyTorch on per-step time or peak
RSS.** It is ~3.6× slower per step on this shape. The one thing it wins is **allocation / GC
churn** (0.127 MB/step vs PyTorch's per-tensor malloc/free) — a memory-stability and scaling
benefit (bounded peak gradient memory = O(largest layer), zero steady-state GC), not a throughput
win.

The 3.6× per-step gap is **GEMM + autodiff overhead** — the core #653 CPU-parity problem.
Optimizer-in-backward does not touch matmul speed, so it cannot close that gap. Closing it is a
separate, larger effort (managed/native GEMM tuning, autodiff dispatch overhead).

## Implications for lever #1

- The **§5a default-on flip is NOT justified by this proof** — there is no speed win to make
  single-pass fused optimizer-in-backward the default; doing so would not make training faster and
  would change the default code path without a throughput benefit. **Keep it opt-in** (`ForceOn`),
  valued for its memory/allocation properties (bounded peak grads, zero GC churn → enables larger
  models, smoother long runs), which are bit-identical to classic.
- The honest, defensible claims for the PR are: (1) optimizer-in-backward is bit-identical to
  classic Adam for both clipped and unclipped (verified); (2) it bounds peak gradient memory to
  O(largest layer) and drives per-step allocation to ~0 (verified); (3) it enables clipped
  optimizer-in-backward, which PyTorch's `apply_optimizer_in_backward` does not support at all.
- "Handily beat PyTorch on all metrics" is **not** currently true and should not be claimed.

## Gap investigation — it's small-batch GEMM parallel-scaling

Decomposing the per-step deficit by batch size (D=384, 10 layers, 8 threads):

| Batch | PyTorch ms | AiDotNet ms | gap |
|---|---|---|---|
| S=128  (M=128)  | 69.6  | 252.9 | 3.6× |
| S=1024 (M=1024) | 392.5 | 559.1 | 1.43× |

Isolated single-GEMM (128×384×1536) measured 9.65 ms (~15 GF) for AiDotNet vs 0.63 ms (239 GF)
for torch — but the `--gemm` probe path is the known unused-overload red herring; the *training*
path (`TensorMatMul` → in-house BlasManaged kernels) is what the trainbench exercises.

The signal is clear: the deficit is **GEMM-throughput-bound and concentrated at small batch**. At
M=1024 the gap collapses to 1.43×, matching the documented small-M finding (#475): the managed
microkernel is healthy (MKL-parity on large DiT-XL shapes), but small-M GEMM **parallel scaling
plateaus (~2×)** — not enough rows to amortize thread dispatch. So:

- The per-step speed gap is the **#653 core CPU-parity problem (small-M GEMM parallel scaling)**,
  NOT something optimizer-in-backward (lever #1) can address — it doesn't change matmul cost.
- Closing it is a separate, scoped, sizeable effort (core-perf sprint #368/#375/#475 territory):
  better small-M parallelization / batched-GEMM dispatch.
- The optimizer-in-backward allocation/memory win (0.127–0.173 MB/step, zero steady-state GC,
  O(largest-grad) peak) holds at every batch size and is bit-identical — that remains lever #1's
  real, shippable contribution.

## Caveats

- Whole-process RSS baselines differ between the CLR and Python+libtorch runtimes, so absolute
  peak RSS is only roughly comparable; per-step wall time is the clean comparison and AiDotNet
  clearly loses it.
- The AiDotNet trainbench's SGD update is a scalar per-element C# loop in the streaming callback,
  which adds some overhead; even discounting it, the GEMM fwd+bwd dominates and the gap remains.
