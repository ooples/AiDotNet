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

## Caveats

- Whole-process RSS baselines differ between the CLR and Python+libtorch runtimes, so absolute
  peak RSS is only roughly comparable; per-step wall time is the clean comparison and AiDotNet
  clearly loses it.
- The AiDotNet trainbench's SGD update is a scalar per-element C# loop in the streaming callback,
  which adds some overhead; even discounting it, the GEMM fwd+bwd dominates and the gap remains.
