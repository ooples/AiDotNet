# Design — #1662 Backward-pass memory/throughput optimizations

**Issue:** [#1662](https://github.com/ooples/AiDotNet/issues/1662) — backward/training side of #653 (match/beat PyTorch CPU training overhead), part of the #1624 training-bound timeout/OOM effort.
**Date:** 2026-06-22
**Scope this design:** levers **#4** (backward buffer-reuse audit), **#1** (optimizer-in-backward), **#3** (FlashAttention-style backward). **NOT** lever #2 (checkpointing default-on — owned by the open PR #1633).

---

## 1. Context & constraints

The forward caching allocator is handled elsewhere (#1661 / Tensors #661). Already in place and **NOT to be redone**: per-step `TensorArena` (`NeuralNetworkBase`, recycles fwd+bwd+grad buffers per iteration), opt-in gradient checkpointing (#643/#645), streaming backward (`GradientTape.ComputeGradientsStreaming`, Tensors #564), and the Adam8Bit fused epilogue.

This design covers the three remaining backward levers. Lever #2 (checkpointing default-on) is **explicitly out of scope**: PR #1633 (branch `perf/1624-training-scale`, still open) owns activation checkpointing and its package fixes (Tensors 0.101.5 — #643 weight-grad + #645 multi-segment detach). We do not touch checkpointing.

### Cross-repo reality

The backward kernels live in the **AiDotNet.Tensors** repo, not AiDotNet:
- `src/AiDotNet.Tensors/Engines/Autodiff/BackwardFunctions.cs` — backward ops (lever #4).
- `src/AiDotNet.Tensors/Engines/Autodiff/FusedAttention.cs` — `FusedAttention.Backward` (lever #3).
- `src/AiDotNet.Tensors/Engines/Autodiff/GradientTape.cs` — `ComputeGradientsStreaming` (already exposes a per-source gradient callback; basis for lever #1).
- `tools/ConvParallelProbe/Program.cs` — measurement probe (lever #4 adds a mode).

The AiDotNet-side work is the lever #1 wiring in `NeuralNetworkBase.TrainWithTape` plus the acceptance/gate tests.

Master pins `AiDotNet.Tensors` **0.94.2**; the Tensors repo is published through **v0.101.6** (dev branch `fix/fp16-resident-store-backward-grad`).

### Acceptance gate (from the issue)

Measured per-step training **peak-RSS and allocation reduction** on the #1624 canonical shape — **SimCSE\<float\> dim=384, 10 layers, fused-optimizer path** — with the **loss trajectory unchanged**. Each lever ships with a bit-identical-trajectory (or fp-tol bit-close) test. No test weakening; paper-scale shapes preserved.

---

## 2. Work structure — two worktrees, Tensors-first

| Worktree | Repo | Branch | Levers |
|---|---|---|---|
| `AiDotNet.Tensors/.claude/worktrees/pr1662` | AiDotNet.Tensors | `perf/1662-backward` | #4, #3, + fused grad-norm reduction helper for #1's two-pass path |
| `AiDotNet/.claude/worktrees/pr1662` | AiDotNet | `perf/1662-optimizer-in-backward` | #1 wiring + gate tests + Tensors version bump |

**Sequencing:** #4 (audit, cheapest, immediate signal) → #3 (FlashAttention backward, largest) in Tensors → publish a new Tensors release (next after v0.101.6, e.g. **v0.102.0**) → bump in AiDotNet and land #1.

The AiDotNet **draft** PR for #1662 may sit red until the Tensors release is published — expected for a draft and called out in the PR body.

---

## 3. Lever #4 — backward buffer-reuse audit (Tensors)

**Problem:** the per-step arena only recycles backward grad buffers that flow through `Rent`/arena. Allocations that bypass it (`new Tensor<T>`, `AutoTensorCache.RentOrAllocate` when its pool is disabled, temporary `new T[]` / `Vector<T>`) still churn GC every training step.

**Approach:**
1. **Measurement first.** Add a `--trainbench` mode to `ConvParallelProbe` (mirrors the existing `--attnblock` / `--arena alloc_MB/fwd` style): builds a small transformer/MLP block, runs `forward → backward → optimizer-step` in a loop, and reports per-step **managed allocation** (`GC.GetAllocatedBytesForCurrentThread`) and peak working set. This gives a before/after number for the audit.
2. **Audit & fix.** Walk `BackwardFunctions.cs` and the CpuEngine `*Backward` ops; route arena-bypassing allocations through the per-step arena `Rent`. One op at a time, re-measuring with `--trainbench`.
3. **Fused grad-norm reduction helper.** While in the backward path, add a reduction that produces each gradient's sum-of-squares contribution as it is computed (the grad is already hot in cache). This is consumed by lever #1's two-pass clipped path so the norm pass costs ~0 extra sweeps.

**Correctness gate:** grads bit-identical before/after; a new `BackwardArenaTests` asserting per-step backward allocation drops, guarded exactly like the existing `InferenceArenaForwardTests`.

---

## 4. Lever #3 — FlashAttention-style tiled backward (Tensors)

**Problem:** `FusedAttention.Backward` currently **recomputes and materializes the full `P` (S×S) probability matrix** during backward (`pFlatT`, full `P [B*H, Sq, Sk]`). That is the dominant activation term for long sequences — O(S²) memory.

**Approach:** rewrite `Backward` to **tile over K/Q blocks** in the FlashAttention-2 style:
- Carry the forward softmax statistics (per-row max `m`, per-row sum-exp `l`) so each tile's contribution to `P` can be reconstructed without storing the whole matrix.
- Recompute `S = QKᵀ` per tile; accumulate `dV = Pᵀ dO`, `dS = P * (dP - rowsum(dP * P))`, `dQ = dS·K·scale`, `dK = dSᵀ·Q·scale` tile-by-tile.
- The full S×S `P` is never resident → **O(S) memory** instead of O(S²).

**Correctness gate:** `dQ / dK / dV` **bit-close (fp tolerance)** to the current stored-`P` backward across a matrix of sequence lengths and head counts; parity test added to the Tensors autodiff suite. This is the largest and most novel of the three; the forward FlashAttention kernel (#89) already exists, so this completes the matched backward.

---

## 5. Lever #1 — optimizer-in-backward, adaptive hybrid (AiDotNet)

**Problem:** training currently collects all gradients, then steps the optimizer once. Peak grad memory is O(all params' grads).

**Foundation that already exists:** `GradientTape.ComputeGradientsStreaming(loss, sources, onSourceGradient)` walks the graph in reverse, releases activations topologically, and fires a **per-source gradient callback** — but today it only *frees*; it does not step the optimizer. Lever #1 pushes the optimizer update into that callback.

**The clipping barrier.** Exact global-norm gradient clipping computes a single scalar `c = min(1, max_norm / ‖g‖)` that depends on *every* gradient, and Adam consumes `c·g` / `(c·g)²` non-linearly. So no parameter can be stepped until the final gradient is produced — a hard synchronization barrier. PyTorch lives with this by holding all grads in memory and doing a separate post-backward foreach clip+step. Three regimes follow:

### 5a. Unclipped path — single-pass fused step (the double win)
In the `onSourceGradient` callback, apply the optimizer's per-parameter fused update for that gradient the moment it is produced, then free it.
- **Memory:** peak grad = O(largest layer's grad), not O(all params). Strict win over PyTorch.
- **Throughput:** each gradient is stepped while still hot in L2, and we never run PyTorch's separate optimizer sweep over all params. Locality win over PyTorch.
- **Bit-identical** to collect-then-step.

### 5b. Clipped path (global-norm) — two-pass with fused norm
- **Pass 1:** stream gradients, accumulate global sum-of-squares (via lever #4's in-backward reduction — near-free), free each grad. O(largest-grad) memory.
- Compute global norm and clip scale `c`.
- **Pass 2:** stream again, applying the `c`-scaled fused optimizer step per parameter.
- **Bit-identical** to collect-then-step. Wins on memory; the cost is one extra backward recompute (so ~matches PyTorch throughput, never worse on trajectory). Beating PyTorch on *both* axes here is mathematically precluded by the barrier while staying bit-identical.

### 5c. Opt-in fast-clip — single-pass even when clipping (OFF by default)
An explicitly opt-in mode using a running/EMA grad-norm from the prior step (NFNet-style adaptive clipping, Brock et al. 2021) to set the clip scale, enabling a single fused pass even under clipping.
- **NOT bit-identical** — a documented approximation. **OFF by default.**
- Excluded from the bit-identical-trajectory gate; has its own convergence test (training converges, does not diverge).
- Never engages unless explicitly enabled.

### Optimizer-interface integration
Today the fused step is `optimizer.Step(TapeStepContext<T>)`, stepping the whole parameter set, with `ApplyGradientClipping` called just before it. Lever #1 needs a **per-source/per-slice fused step** entry point the streaming callback can invoke, mapping each source tensor to its optimizer-state slice (Adam `m`/`v`). The per-slice updates **must advance Adam `m`/`v` identically** to the whole-vector `Step`, which the bit-identical gate verifies. Exact mechanism (extend `IGradientBasedOptimizer` / `TapeStepContext` vs a new per-parameter step API) is resolved in the implementation plan.

**Correctness gate:** bit-identical training trajectory vs collect-then-step on the #1624 SimCSE\<float\> dim=384, 10-layer shape (`FusedOptimizerIntegrationTests`-style); Adam `m`/`v` identical per step; per-step training peak-RSS and allocation reduction asserted.

---

## 6. Out of scope

- **Lever #2** (gradient checkpointing default-on) — owned by open PR #1633 and its Tensors 0.101.5 checkpoint fixes. Not touched here.
- GPU backward paths — this design targets the CPU training overhead of #653/#1624.
- Forward caching allocator — #1661 / Tensors #661.

---

## 7. Risks & open questions (resolved during planning)

- **Per-slice Adam state mapping** in `TrainWithTape` must line up exactly with the existing whole-vector ordering used by `GetParameters`/`Step`; mis-mapping silently corrupts `m`/`v`. Verified by the bit-identical gate on a multi-layer model.
- **Tensors release timing** — the AiDotNet PR cannot go green until v0.102.0 is published. Acceptable for a draft; flagged in the PR body.
- **`--trainbench` shape selection** must be representative enough that arena-bypass regressions are caught, while staying fast enough for CI. Default to a small transformer block; document the shape.
