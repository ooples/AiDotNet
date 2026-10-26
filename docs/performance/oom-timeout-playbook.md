# OOM and timeout playbook

How the foundation-scale OOM and timeout CI shards were cleared, in the order the levers mattered, and the
diagnostic signature that points at each. Moved here from issue #1684 so it lives with the code. Remaining
foundation-scale OOM work is tracked in #1622.

Two additions since the issue was written (2026-10-24, #2366):
- An arena held across a long training LOOP in a test has the same signature as Breakthrough 1: live heap
  after a forced GC grows linearly per step. Cascade R-CNN grew 1.2 GB per step to OutOfMemory inside one
  `TensorArena` and stayed flat at ~3.9 GB without it. Separate retained growth from garbage first:
  `GC.GetTotalMemory(true)` per step, then `dotnet-gcdump collect -p <pid>` and `dotnet-gcdump report`.
- Weight streaming (Breakthrough 5) trades memory for allocation churn. On ParaformerLarge (661M) most
  of the ~18 GB allocated per step is weight materialization and write-back (#2322). Profile it with
  `dotnet-trace collect --providers Microsoft-Windows-DotNETRuntime:0x1:5` and aggregate
  `GCAllocationTick` events by stack.
## TL;DR — the lever order that actually fixed the #1677 OOM/timeout shards

**OOM/timeout was fixed primarily by MEMORY-ARCHITECTURE changes, not by switching to float.**
`float`-by-default was the **last and smallest** lever — applied only to the residual
training-perf-bound models *after* the memory architecture was already bounded. The big wins
(in impact order) were caching-allocator arenas, fused optimizer-in-backward, copy-on-write
clone, and weight streaming. This playbook captures each, with the exact mechanism + the
diagnostic signature that points you at it, so the rest of #1677 can be cleared the same way.

When you hit a new OOM/timeout shard, **first classify it** (this decides the fix):
- **OOM that grows ~linearly with step/iteration count** → un-recycled per-step scratch under an
  outer arena → Breakthrough 1.
- **OOM at `Clone()` / in `MoreData` / `Clone_ShouldProduceIdenticalOutput`** → double-
  materialized weights → Breakthrough 4 (COW).
- **OOM at construction / weights alone exceed RAM** → Breakthroughs 5–6 (lazy construct +
  weight/optimizer streaming).
- **OOM/slow in the forward only** → Breakthrough 2 (forward caching allocator).
- **Still OOM/timeout purely from FP64 cost after the above** → float-by-default (last lever).

---

## Breakthrough 1 — Per-step caching-allocator arena bounds training memory (#1641 / #1624 / #1640) — the single biggest OOM fix

**The discovery:** the *default* training route for a float model with plain Adam/SGD is the
fused-optimizer path (`NeuralNetworkBase.TryTrainWithFusedOptimizer`). Unlike the eager-tape and
streaming paths, it opened **no per-step `TensorArena`**. So when a caller wraps the whole loop in
ONE un-`Reset` outer arena — which the shared **ModelFamilyTests base does** for cross-step buffer
pooling — every step's forward+backward activations piled into that outer arena's ring and
**accumulated → linear growth → OOM**. Measured: **+403 MB/step on ViT-Base** (this is #1640's
"100 GB over a long run"), **+290 MB/step on SimCSE** (#1624).

**The fix:** open a per-step arena around the fused step (Dispose in a `finally` immediately after
it returns), matching how the eager + streaming paths already scope — and exactly how PyTorch's
caching allocator bounds training memory: each iteration's freed blocks return to a reuse pool so
the process reaches a steady state, while the optimizer state (Adam m/v) and persistent
input/target live OUTSIDE the per-iteration churn (GC-heap-held by the compiled plan, NOT
arena-ring allocated, so the per-step scope recycles only transients and leaves optimizer state
intact).

**Reusable lesson:** an OOM whose peak RSS climbs ~linearly with the number of training steps is
almost never the model's size — it's un-recycled per-step scratch accumulating under an outer
arena. Find the training code path that doesn't scope (and Reset/Dispose) a per-step arena.

## Breakthrough 2 — Forward caching allocator, arena default-ON (#1661/#1663) + grad-accum arena reuse (#1651)

- **Inference**: extracted `Predict` into a `PredictCore` funnel that runs inside a per-call
  `TensorArena`, recycling intermediate-tensor buffers (~**98% allocation reduction**,
  bit-identical), copying the result out via `DetachFromArena` (`ToArray()` → fresh GC array that
  survives `Dispose`). Default-ON via the migration; nesting-safe for composite/GAN sub-models.
- **Gradient accumulation** (#1651): `TrainWithGradientAccumulation` created a *fresh* arena per
  chunk (zero cross-chunk reuse). Hoisting ONE arena over the batch + `Reset()` per chunk gave
  **117.2 MB/fwd → 0.12 MB/fwd (~970×)** and GC-pause tail **2380 ms → 638 ms**. (Cross-chunk
  accumulators must be `RentPinned` so they survive `Reset`.)

## Breakthrough 3 — Fused optimizer-in-backward as the common-case default (#1662/#1664)

Promote single-pass **fused optimizer-in-backward** for unclipped training: the optimizer update
runs INSIDE the backward pass, so the per-parameter gradient tensors are **never materialized**
between backward and the step → lower peak memory + fewer allocations. Bit-identical to
collect-then-step. (Previously single-pass-fused only engaged above ~0.5× RAM in
`TrainWithTapeStreaming`; this makes it the default in the fits-in-memory case too.)

## Breakthrough 4 — Copy-on-write (COW) Clone kills Clone/MoreData OOMs (#1633 "G6", #1669/#1676)

`Clone_ShouldProduceIdenticalOutput` / `MoreData_ShouldNotDegrade` OOM'd because the default clone
is `new(...); SetParameters(GetParameters())` — which materializes the **entire weight set a SECOND
time** plus a giant intermediate flat vector. Fix: share parameters **copy-on-write** via Tensors
`CloneShared()` (O(1)-until-write). `TryShareParametersFrom` / `TryDeepCopyCopyOnWrite` do a
parallel reflection walk of source+clone trainable layers and re-bind the clone's tensors to COW
views — fidelity-equivalent (same trainable tensors transferred), O(1) memory until first write,
with a fallback to the eager flat copy on any 1:1 structure mismatch. (#1669: also stop the COW
layer-walk recursing through pointer fields.)

**Reusable lesson:** a `Clone()`/`MoreData` OOM is double-materialization, not training cost — COW
param-share fixes it without changing fidelity.

## Breakthrough 5 — Weight streaming for foundation-scale models (#1596/#1610/#1621 + #1633 lazy construct)

Models whose weights alone exceed RAM auto-enable transparent **weight streaming** (stream weights
to/from disk above a parameter threshold) and **lazy foundation-scale construction** (never
materialize all weights at construction). Covers the DiT diffusion family, foundation VLMs, and
diffusion noise predictors.

## Breakthrough 6 — Streaming first-order optimizer variants (#1603/#1609)

Optimizer state (Adam m/v) otherwise 2–3×'s the weight memory resident. Streaming variants of every
gradient-based optimizer keep that state bounded, paired with the streaming training path.

## Breakthrough 7 — Diffusion denoise-loop + parallel-test specifics (#1668/#1674, #1620, #1671/#1677)

- Disable the unsafe inference arena INSIDE the denoise loop (the per-call arena recycled buffers
  the multi-step loop still referenced → corruption/non-determinism); use an eager per-step
  denoising forward (avoid compile-cache staleness).
- **Cap BLAS threads in parallel tests** — thread explosion across parallel test workers caused
  both memory blow-ups and timeouts.
- TCD clone + default-construction timeouts (#1671/#1677).

---

## Last (smallest) lever — float-by-default, AFTER the memory architecture was bounded (#1679/#1680)

Only the residual training-perf-bound models that still OOM'd/timed-out purely from FP64 cost
(double = 2× resident + slower per-op) were converted to `T=float` test scaffolds — FP32 exercises
the same code paths at ~half the memory and meaningfully faster, and the model-family invariants
don't need FP64 precision.
- Auto scaffolds: add the class to `TestScaffoldGenerator.Fp32TestClassNames`; `FloatifyGenericArgs`
  rewrites the generic args / return type / ctor / factory to `<float>`.
- Manual scaffolds (EmotiVoice/TinyBERTNER/UNet3D — ctors need an architecture/options or ONNX
  path): float the base + `CreateNetwork()` directly; generic-ize the family base `<T>` first if
  it's double-only.
- Reproduce CI-faithfully with `AIDOTNET_DISABLE_GPU=1` (CI runners are CPU-only).

---

## Then: float UNMASKS pre-existing functional failures — separate them, don't chase ghosts

1. **Runner-shutdown false positives**: many shards report `failure` from
   `##[error]The runner has received a shutdown signal` mid-run, not a test failure. Always grep for
   real `[FAIL]` markers.
2. **Test-unmasking**: as the memory fixes + float let the runner reach further into the queue, it
   *reveals* pre-existing downstream failures. **New FAIL counts ≠ regressions** — confirm against
   the pre-fix commit.

## Functional-bug debug playbook (never weaken the test)

Isolate (`--filter --no-build AIDOTNET_DISABLE_GPU=1`) → **elimination probe** in `testconsole`
(toggle one variable: params? masks/seeds? COW-vs-eager? thread-static? forward-vs-backward?
layout?) → **per-op output-checksum bisection** (find the first divergent op) → **reflection dumps**
of private layer state (fresh vs cloned). Then **verify paper fidelity**: if the model matches the
paper, fix the TEST with a paper-grounded override; if not, fix the model.

## Reusable functional lessons found via the above

1. **Layer ctor-args MUST be persisted via `GetMetadata()` — including NON-shape args.**
   `DropoutLayer` never persisted `dropoutRate`, so the serialize/deserialize clone path rebuilt it
   via `DropoutLayer(double=0.5)` and used the 0.5 DEFAULT — a configured 0.2-dropout model became
   0.5-dropout on clone and diverged (`EmotiVoice MoreData`). No shape mismatch (dropout has no
   params) → it failed silently. **Audit every no-params / hyperparameter-only layer.**
2. **Argmax-decoding (NER) models must measure `GetLastLoss()`, not `MeasureLoss(Predict(...))`.**
   NER `Predict` argmax-decodes to discrete labels, so the base computed cross-entropy over integer
   label IDs (meaningless, never moves) — making `Training_ShouldReduceLoss` and `MoreData` fail/
   flaky. Override both in `NERModelTestBase` to measure the optimizer's real objective.
3. **LayerNorm architectures: scale-invariant + uniform-collapse.** `ScaledInput_ShouldChangeOutput`
   (BERT LayerNorm is exactly magnitude-invariant) and the PRE-training `DifferentInputs_*`
   (untrained Viterbi/argmax is scale-insensitive → collapses, flaky under non-deterministic init)
   need random-pattern / learn-to-differentiate overrides — the CRF-aware
   `AssertModelCanLearnToDifferentiateInputs` pattern in `NERModelTestBase`.
4. **Dropout has a second, latent reproducibility bug.** `AiDotNet.Tensors CpuEngine.TensorDropoutMask`
   parallel path (`totalElements > 10000`) keys its per-thread RNG on `ManagedThreadId`, so a SEEDED
   mask is still non-reproducible run-to-run (and context-dependent → flaky NER/clone tests). A
   per-element SplitMix/Philox hash fixes it. **Separate AiDotNet.Tensors PR** (cross-repo).

---

*Source: the #1679 / PR #1680 commit history (#1641, #1661, #1651, #1662, #1633, #1669, #1596,
#1610, #1603, #1668, #1620, #1671) and the EmotiVoice/TinyBERTNER/BiLSTMCRF functional fixes.*
