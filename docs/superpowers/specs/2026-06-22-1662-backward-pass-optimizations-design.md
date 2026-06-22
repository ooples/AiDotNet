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

Measured per-step training **peak-RSS and allocation reduction** on the #1624 canonical shape — **SimCSE\<float\> dim=384, 10 layers, fused-optimizer path** — with the **loss trajectory unchanged**. Each lever ships with a bit-identical-trajectory (or fp-tol bit-close) test. No test weakening; paper-scale shapes preserved. **Additionally (lever #1): a `--trainbench` head-to-head must show AiDotNet handily beating a torch CPU baseline on per-step wall time, peak RSS, and per-step allocation on the acceptance shape — the default-on flip (§5a) is gated on this proof.**

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
1. **Measurement first.** Add a `--trainbench` mode to `ConvParallelProbe` (mirrors the existing `--attnblock` / `--arena alloc_MB/fwd` style): builds a small transformer/MLP block, runs `forward → backward → optimizer-step` in a loop, and reports per-step **managed allocation** (`GC.GetAllocatedBytesForCurrentThread`), **peak working set**, and **per-step wall time**. This is both the before/after number for the audit AND the §5d proof harness: it emits machine-readable metrics so a torch CPU baseline (a committed `trainbench_torch.py` on the same shape/optimizer) can be diffed against it to prove AiDotNet wins on every metric.
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

**Problem:** for the common case (models that fit in memory), training collects the *full* gradient set, then steps the optimizer once — the same collect-then-step strategy as PyTorch, and **not superior**.

**What already exists (and why it is NOT enough).** `GradientTape.ComputeGradientsStreaming(loss, sources, onSourceGradient)` walks the graph in reverse, releases activations topologically, and fires a per-source gradient callback. `NeuralNetworkBase.TrainWithTapeStreaming` (~L6294-6382) *already* uses it to do single-pass fused optimizer-in-backward (unclipped) and two-pass-norm (clipped). **BUT** this path is wired purely as an **OOM-survival** mechanism: `ShouldUseStreamingTraining()` Auto engages it **only when `footprint > 0.5 × available RAM`** (L6207). For models that fit, the **classic collect-then-step path** (L6999-7028) runs — the full gradient set is resident and a single whole-set `opt.Step(context)` sweep follows. So:
- The common case ties PyTorch (collect-then-step), it does not beat it.
- The clipped streaming path does a **full second backward** (2× backward compute) — a throughput *loss* on a fitting model.

**What lever #1 must actually deliver (non-duplicative):**

### 5a. Promote single-pass fused optimizer-in-backward to the common-case default (unclipped)
Engage the existing `streamingOptimizer.Apply(source, grad)` single-pass path for unclipped training on models **that fit**, not just on OOM. This is the clean double-win: peak grad = O(largest layer) and each gradient is stepped while hot in L2 (no separate optimizer sweep). **Bit-identical** to collect-then-step. Lower/replace the `ShouldUseStreamingTraining` Auto gate (or add a dedicated `FusedOptimizerInBackward` engagement) so it is **default-on for unclipped**, with a `ForceOff` escape hatch. **Rollout is gated on the §5d PyTorch-comparison proof.**

### 5b. Clipped path — bit-identical stays two-pass
The global-norm barrier forces clipped + bit-identical into 2× backward (compute the global norm before any param can be stepped). Keep the existing two-pass-norm path for clipped runs; it wins on memory and ties/loses on throughput. We do not regress it.

### 5c. Opt-in fast-clip — single-pass even when clipping (OFF by default)
The only way to win *single-pass* under clipping is to drop the barrier: a running/EMA grad-norm from the prior step (NFNet-style adaptive clipping, Brock et al. 2021) sets the clip scale, enabling one fused pass. **NOT bit-identical** — a documented approximation, **OFF by default**, excluded from the bit-identical gate, with its own convergence test. This is what makes clipped training also beat PyTorch when the user opts in.

### 5d. Proof harness — handily beat PyTorch on ALL metrics
A `--trainbench` PyTorch-comparison (see §3) producing hard numbers on the acceptance shape: **per-step wall time, peak RSS, and per-step managed allocation**, AiDotNet vs a torch CPU baseline. The §5a default-on flip does not land until this shows AiDotNet winning on every metric. This proof is a required deliverable, not a nice-to-have.

### Optimizer-interface integration
The single-pass path already calls `streamingOptimizer.Apply(source, grad)` per gradient (the per-slice fused step exists). The §5a work is to **engage that path for fitting models** when unclipped, not to rebuild it — i.e. replace/augment the `ShouldUseStreamingTraining` Auto gate with a `FusedOptimizerInBackward` default-on-for-unclipped decision, retaining `ForceOff`. The per-slice updates **must advance Adam `m`/`v` identically** to the whole-vector classic `opt.Step(context)`; the bit-identical gate verifies this on a multi-layer model.

**Correctness gate:** bit-identical training trajectory (and per-step Adam `m`/`v`) of fused-optimizer-in-backward vs the classic collect-then-step path on the #1624 SimCSE\<float\> dim=384, 10-layer shape; per-step peak-RSS and managed-allocation reduction asserted; the §5d harness shows AiDotNet beating the torch CPU baseline on per-step wall time, peak RSS, and allocation. Fast-clip (§5c) is excluded from the bit-identical gate and has its own convergence test.

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
