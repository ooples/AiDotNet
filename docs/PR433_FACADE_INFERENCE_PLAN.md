# PR #433 Facade Integration Plan (Inference Optimizations)

## 0) Purpose

Integrate PR #433 inference optimizations into the *existing facade pattern* so that:
- Users configure everything via `PredictionModelBuilder.ConfigureInferenceOptimizations(...)`.
- Users consume everything via `PredictionModelResult` (e.g., `Predict`, plus a session API).
- Internal implementation details (optimizer, caches, kernels, batching, speculative decoding) remain hidden behind the facade.

This plan is written to be actionable for a junior engineer without requiring deep prior context.

---

## 1) Constraints / Non‑Goals

### 1.1 Facade constraints (must keep)
- No new user-facing entry points besides:
  - `PredictionModelBuilder` for configuration + training/build.
  - `PredictionModelResult` for inference.
- Any “advanced” types should be:
  - `internal`, or
  - nested under `PredictionModelResult` (if we must expose a session object).

### 1.2 Correctness constraints (must keep)
- "Industry standard defaults" when user omits config values.
- Avoid semantic changes by default for non-autoregressive models, but prefer serving-grade defaults when inference optimizations are explicitly enabled:
  - `AttentionMasking=Auto` should assume causal for inference sessions unless the model/layer explicitly indicates bidirectional attention.
  - Paged KV-cache should be auto-enabled by default (opt-out via config).
  - KV-cache must not silently change results for models that are not compatible with caching.

### 1.3 Non-goals (for this integration pass)
- Rewriting training-time behavior.
- Full rewrite of the `src/InferenceOptimization/*` SIMD/IR plan (tracked separately in `INTEGRATION_PLAN_PR433.md`).

---

## 2) Current Issues / Gaps (What to Fix)

### 2.1 “Disconnected” implementations
New PR #433 classes exist but are not consistently invoked through the facade:
- `FlashAttention.cs`, `CachedMultiHeadAttention.cs`, `InferenceOptimizer.cs`, KV cache variants, etc.

### 2.2 Model cloning/serialization assumptions
We need to treat `NeuralNetworkBase<T>.Clone()` behavior carefully:
- Confirm whether it is deep copy vs shallow copy in practice.
- If cloning relies on serialization, ensure all attention layers can be deserialized with correct constructor arguments (constructor mismatch is a known risk).
- If serialization cannot faithfully round-trip all layer types, implement a robust deep-clone path that still preserves the facade (users never touch these internals).

### 2.3 Layer coverage
The optimizer rewrite logic must recognize more attention layer types, not just:
- `MultiHeadAttentionLayer<T>` and `FlashAttentionLayer<T>`.

Add support/coverage for (at minimum):
- `AttentionLayer<T>` (mask-aware)
- `SelfAttentionLayer<T>`
- `GraphAttentionLayer<T>`
- Any transformer encoder/decoder attention layers in `src/NeuralNetworks/Layers/*`

### 2.4 Missing facade integration
`EnableBatching` and `EnableSpeculativeDecoding` must be wired end-to-end:
- Builder stores config.
- Result uses config during inference.
- Serving project should be able to leverage the same internal components.

### 2.5 Paged attention integration missing
`PagedAttention` / `PagedKVCache` exist but are not selected/used based on config.

### 2.6 Tests
We need:
- Unit tests for low-level correctness (some exist).
- Integration tests that validate the facade wiring end-to-end via `PredictionModelBuilder` + `PredictionModelResult`.

---

## 3) Target UX (What the user sees)

### 3.1 Builder usage (no new user-facing components)
```csharp
var result = await new PredictionModelBuilder<float, Tensor<float>, Tensor<float>>()
    .ConfigureModel(model)
    .ConfigureInferenceOptimizations(new InferenceOptimizationConfig
    {
        EnableFlashAttention = true,
        EnableKVCache = true,
        EnableBatching = true,
        EnableSpeculativeDecoding = true
    })
    .BuildAsync(x, y);
```

### 3.2 Result usage
Default usage stays the same:
```csharp
var y = result.Predict(x);
```

For sequence-based inference (KV-cache, generation, serving), add a *single* facade-friendly entry:
- `result.BeginInferenceSession()` returns a session object that manages caches/batching internally.

Example shape:
```csharp
using var session = result.BeginInferenceSession();
var y1 = session.Predict(x1);
var y2 = session.Predict(x2);
```

Notes:
- The session type should be nested under `PredictionModelResult` (or otherwise kept hidden).
- "Clear cache" should be internal to session lifecycle; avoid exposing raw cache types.
- Avoid adding new public cache-control methods on `PredictionModelResult`; prefer session `Dispose()`/`Reset()`.

---

## 4) Implementation Plan (Phased)

### Phase A - Baseline facade wiring + safety (no batching/spec yet)

**Goal:** Ensure optimizations are *actually applied* and correct during inference for supported model types.

1) **Confirm cloning semantics**
   - Inspect `NeuralNetworkBase<T>.Clone()` and `DeepCopy()` implementation.
   - Decide policy:
     - If deep clone is reliable for all relevant layers, prefer cloning before rewrites.
     - If cloning is unreliable, either:
       - Fix serialization/deserialization for attention layers, or
       - Apply rewrites in-place but only inside an inference-session-scoped model instance owned by the result (never mutate the user's original model object).
   - Required outcome for facade safety:
     - Never mutate the user's original model object.
     - Any optimized/mutated model instance is owned by the result/session internally.
   - Acceptance criteria:
     - No runtime errors when inference optimizations are enabled.
     - No cross-request contamination (weights/caches).

2) **Make serialization/deserialization round-trip attention layers (if used by clone)**
   - Inventory layer constructors that require metadata:
     - `MultiHeadAttentionLayer<T>` (e.g., head count)
     - `SelfAttentionLayer<T>`
     - `AttentionLayer<T>` (mask semantics)
     - `GraphAttentionLayer<T>` (graph-specific parameters)
   - Update the model serialization format in a backward-compatible way:
     - Preserve current header/structure so existing saved models still load.
     - Add per-layer metadata payload (versioned) needed to reconstruct constructors.
   - Extend deserialization to use metadata when present and fall back to safe defaults when absent.
   - Acceptance criteria:
     - `Clone()` becomes a true deep copy for supported model types.
     - `OptimizeForInference()` can safely operate on a cloned instance.

3) **Extend optimizer layer detection/rewrite coverage**
   - Identify all attention-related layer types and their expected shapes/masking behavior:
     - `MultiHeadAttentionLayer<T>`
     - `FlashAttentionLayer<T>`
     - `AttentionLayer<T>` (mask input)
     - `SelfAttentionLayer<T>`
     - `GraphAttentionLayer<T>` (graph adjacency/attention specifics)
   - For each layer type, decide:
     - Can we rewrite to `FlashAttentionLayer<T>` safely?
     - Can we wrap/replace with KV-cache variants safely?
     - If not, skip and record diagnostics (internal).
   - Acceptance criteria:
     - Transformer models built via default layer helper are optimized.
     - Non-transformer models are unchanged.

4) **Masking policy**
   - Make masking decision consistent and safe:
     - `AttentionMasking=Auto` defaults to causal masking for inference sessions unless a bidirectional model is clearly indicated.
     - `AttentionMasking=Causal` forces causal mask.
     - `AttentionMasking=Disabled` disables causal mask even for text generation.
   - Heuristics for `Auto` (ordered, conservative):
     - If the layer explicitly takes an attention mask input and one is provided, honor it (don’t infer).
     - If KV-cache is enabled (or a session is created), assume causal unless explicitly overridden.
     - If model metadata/task type exists and indicates non-causal (e.g., encoder/classification), prefer non-causal.
     - If uncertain, default to causal only inside a session (so plain `Predict()` stays safest).
   - Acceptance criteria:
     - No causal mask applied for classification/encoder tasks unless forced.

5) **Paged attention config plumbing (no swap yet)**
   - Add config surface for paged attention selection with industry standard defaults:
     - `EnablePagedKVCache = true` by default (opt-out).
     - `PagedKVCacheBlockSize = 16` by default (configurable).
   - Validate config values.
   - Acceptance criteria:
     - Configuration exists and is validated; actual usage comes in Phase C.

---

### Phase B — Inference Session API (facade-compliant)

**Goal:** Provide a safe, explicit lifecycle for stateful inference features (KV-cache, batching queues).

1) **Add `BeginInferenceSession()`**
   - Implement as a method on `PredictionModelResult`:
     - `public InferenceSession BeginInferenceSession(...)`
   - Session responsibilities:
     - Own (or reference) an optimized model instance.
     - Own cache state (KV cache or paged KV cache).
     - Provide session-scoped prediction methods.
     - Reset/clear caches on `Dispose()`.

2) **Decide session API surface**
   - Minimum recommended:
     - `Predict(TInput input)` for session-scoped inference.
     - Optional: `Reset()` (but keep it on session, not on result).
   - Avoid exposing raw cache objects.

3) **Backwards compatibility**
   - Keep `PredictionModelResult.Predict` working:
     - It may internally use a “default session” with conservative behavior.
     - But for KV-cache and generation patterns, prefer explicit session use.

4) **Serving integration point**
   - Provide internal hooks so `AiDotNet.Serving` can:
     - Create sessions per request/sequence.
     - Route speculative decoding/batching through the same internal components.

Acceptance criteria:
- Session correctly isolates cache state across concurrent requests.
- No public exposure of internal optimization classes beyond session wrapper.
- No public cache-control surface on `PredictionModelResult` (session owns lifecycle).

---

### Phase C — PagedAttention / PagedKVCache integration

**Goal:** Use paged KV-cache for long-context / many-concurrent-sequence serving.

1) **Select KV-cache backend**
   - Based on config and/or heuristics (but default to paged when enabled):
     - If `EnablePagedKVCache` is true, prefer `PagedKVCache<T>`.
     - Otherwise use contiguous `KVCache<T>` (optionally with sliding window).

2) **Bridge cached attention layers to paged cache**
   - Options:
     - Implement a new cached attention layer variant that reads/writes via `PagedKVCache`.
     - Or implement an adapter interface (internal) that abstracts KV-cache operations used by cached attention.
   - Ensure:
     - Prefill path supported.
     - Decode path supported.
     - Sliding window works without O(n) shifting (paged/ring behavior).
   - Ensure causal masking logic supports both:
     - Prefill (many query tokens at once)
     - Decode (queryOffset / position-based masking)

3) **Integrate with serving continuous batching**
   - Ensure a single source of truth for per-sequence cache state.
   - Session ID / sequence ID mapping must be deterministic and internal.

Acceptance criteria:
- Multi-sequence inference works without cache corruption.
- Memory growth is bounded via paging/windowing.

---

### Phase D — EnableBatching integration (facade + serving)

**Goal:** Make `EnableBatching` actually affect inference.

1) **Define batching scope**
   - Offline “single-thread user calling Predict” batching is usually not beneficial.
   - Batching matters for:
     - `AiDotNet.Serving` pipelines
     - Concurrent inference workloads

2) **Implement batching in session/serving**
   - In `PredictionModelResult.BeginInferenceSession()` optionally accept:
     - `BatchingMode` (Auto/Disabled/Force) (or use config only).
   - In serving:
     - Use `RequestBatcher`/`ContinuousBatchingRequestBatcher` to batch requests.
     - Ensure batched forward uses the optimized model instance.

3) **Metrics & safety**
   - Ensure per-request latency bounds (timeout).
   - Ensure correctness when mixing sequence lengths.

Acceptance criteria:
- When enabled and used via serving, throughput increases measurably.
- Batching never changes numerical outputs (only performance).

---

### Phase E — EnableSpeculativeDecoding integration

**Goal:** Route speculative decoding through facade/session/serving for text generation models.

1) **Add a facade entry point for generation**
   - If generation already exists elsewhere, reuse it.
   - Otherwise, add a method on the *session*:
     - `GenerateNextToken(...)` or `Generate(...)` (scope depends on existing LLM infra).

2) **Wire `SpeculativeDecoder<T>`**
   - Construct from config when:
     - Task type is text generation (or user forces).
     - Draft model is configured (NGram default is allowed if desired).
   - Ensure the “target model forward” used by the decoder is the optimized model forward.

3) **Serving integration**
   - Ensure speculative decoding works with:
     - KV-cache
     - Paged cache (if enabled)
     - Continuous batching (optional but desirable)

Acceptance criteria:
- Speculative decoding can be enabled end-to-end via builder config.
- No public exposure of speculative internals; only facade methods (prefer serving-only unless a strong session use-case exists).

---

## 4.1) Gap Analysis Backlog (to exceed industry standards)

These are common inference optimizations worth confirming (or adding) beyond the basic wiring work:
- **Attention kernels:** prefill + decode correctness, queryOffset-based causal masking, multi-query attention support, stable softmax.
- **KV-cache:** per-layer/per-batch correctness, sliding window, paged/ring behavior, cache eviction policy.
- **Inference quantization (missing today):**
  - **Weight-only quantization** (INT8/INT4, per-channel scales, optional GPTQ/AWQ style offline calibration) for transformer blocks and projection matrices.
  - **Activation quantization** (INT8) for matmuls/MLP with calibration (min/max, percentile, KL) and safe fallbacks.
  - **KV-cache quantization** (INT8/FP8) for K/V storage with dequant-on-read or fused quantized attention kernels; configurable per-layer/per-head.
  - **Mixed precision** defaults (FP16/BF16/FP8 where supported) with numerically safe softmax/LayerNorm paths.
  - **Quantization-aware cache policies** (paged KV-cache block reuse/eviction with quantized blocks).
- **Thread safety:** concurrent sessions, cache isolation, avoiding shared mutable state.
- **Serving throughput:** continuous batching, request timeouts, micro-batching heuristics.
- **Speculative decoding:** draft model choices (N-gram vs small neural draft), accept/reject efficiency metrics.
- **Speculation + batching co-scheduling (missing today):**
  - Avoid “double spending” compute when both continuous batching and speculative decoding are enabled.
  - Add policies that trade throughput vs latency under load.
  - Add modern multi-candidate methods (Medusa/EAGLE) and dynamic speculation (“speculative scheduling”).
- **Multi-LoRA (missing today):**
  - Per-session/per-sequence adapter selection, hot-swap, and safe isolation across concurrent requests.
  - Multi-adapter composition policies (merge vs stack) and caching of merged weights.
- **Model cloning/serialization:** versioning + backward compatibility for saved models, deterministic round-trips.
- **Telemetry (internal):** rewrite decisions, cache hit rates, kernel selection, disable-on-failure behavior (internal only).

---

## 4.2) Inference Quantization Roadmap (New)

Goal: add inference-side quantization without expanding public surface beyond `InferenceOptimizationConfig` (builder) and session/serving behavior (result).

### 4.2.1) What exists today
- Deployment/quantization configuration exists, but current usage is primarily training/export oriented.
- PR#433 inference optimizations currently operate on FP32/FP64 tensors and caches.

### 4.2.2) Target capabilities (industry standard baseline → exceed)
1) **Weight-only quantization (WOQ)** for inference
   - INT8 first (simpler), then INT4.
   - Per-channel scales for linear projections (Q/K/V/O, FFN).
   - Offline calibration supported through builder tooling/agents (hidden behind facade).

2) **Activation quantization**
   - Optional INT8 activations for matmul-heavy blocks.
   - Calibration strategies:
     - min/max
     - percentile
     - KL-divergence
   - Safe fallback per-layer when calibration insufficient.

3) **KV-cache quantization**
   - Quantize K/V storage (INT8 or FP8) with dequant-on-read OR fused kernel support.
   - Must work with:
     - contiguous KV cache
     - paged KV cache
     - sliding window mode
   - Defaults:
     - Off by default until kernels are proven stable; once stable, enable by default for serving workloads with opt-out.

### 4.2.3) Facade integration (no new public types)
Add to `InferenceOptimizationConfig` (or reuse `QuantizationConfig` internally):
- `EnableWeightOnlyQuantization` (default: false until validated)
- `WeightQuantizationBits` (8/4)
- `EnableActivationQuantization` (default: false)
- `EnableKVCacheQuantization` (default: false initially; planned default true for serving after validation)
- `KVCacheQuantizationFormat` (INT8/FP8)
- `QuantizationCalibrationMode` (Auto/MinMax/Percentile/KL)

Implementation detail: keep all quantized kernels/types `internal` and selected via the optimizer/session/serving pipeline.

### 4.2.4) Testing/acceptance for quantization
- Golden-output tests with tolerances per quant mode.
- Determinism tests (same input → same output) under identical config.
- Memory-budget tests: confirm KV-cache footprint reduction.
- Regression tests: ensure non-quantized path unchanged.

---

## 4.3) Speculation vs Continuous Batching (Scheduling) (New)

### 4.3.1) Problem
Continuous batching and speculative decoding can compete for the same compute/memory bandwidth:
- Speculation increases per-step compute (draft + verify), but may reduce total steps.
- Continuous batching improves utilization by packing many sequences, but adds scheduling complexity.

### 4.3.2) Policy-based approach (recommended)
Keep `EnableSpeculativeDecoding` usable in sessions and serving, but add **internal policies** that decide *when* to apply it:
- **Latency-first** (small batches): enable speculation more often.
- **Throughput-first** (large batches): reduce speculation depth or disable speculation under high load.
- **Auto** (default): dynamic based on queue depth, batch size, and recent accept-rate.

Add to config (still facade-only):
- `SpeculationPolicy` (Auto/ForceOn/ForceOff/LatencyFirst/ThroughputFirst)
- `MaxSpeculationComputeFraction` (cap draft overhead)

### 4.3.3) Dynamic speculation (“speculative scheduling”)
Implement “dynamic speculation” that adapts:
- speculation depth
- draft batch size
- whether to speculate at all
based on:
- rolling accept-rate
- queue depth / batcher load
- KV-cache pressure (paged block availability)

### 4.3.4) Medusa / EAGLE support
These methods reduce “draft model” overhead by producing multiple candidate tokens with lightweight heads:
- **Medusa**: extra heads on top of the target model to propose multiple tokens.
- **EAGLE**: enhanced draft proposals with improved verification efficiency.

Plan:
1) Add internal capability detection: “model supports Medusa/EAGLE heads”.
2) Extend session/serving generation to use these heads when enabled.
3) Add config flags:
   - `SpeculativeMethod` (ClassicDraftModel/Medusa/Eagle/Auto)
   - `NumSpeculationHeads` (for Medusa-like methods)
4) Ensure policies still apply (auto-disable under high batching load).

---

## 4.4) Multi-LoRA for Inference Sessions (New)

### 4.4.1) Goals
- Allow multiple LoRA adapters to be applied per-session/per-sequence without exposing LoRA internals publicly.
- Make it compatible with serving: per-request adapter selection.

### 4.4.2) Required behaviors
1) **Selection**
   - Session/serving chooses adapter by ID (string) via internal route/metadata.
2) **Isolation**
   - No cross-request weight pollution; adapters never mutate the base weights.
3) **Performance**
   - Cache merged weights per adapter (and per precision/quantization mode).
4) **Composition**
   - Support multiple adapters:
     - merge (weighted sum) OR
     - stack (apply sequential deltas)
   - Keep composition policy internal, configurable via builder options.

### 4.4.3) Test plan for multi-LoRA
- Two sequences with different adapters must produce different outputs for same input.
- Same adapter reused across sequences must hit cache and remain deterministic.
- Adapter hot-swap mid-session must not corrupt caches (KV-cache reset rules must be defined).

## 5) Testing Plan

### 5.1 Unit tests (fast, deterministic)
Add/extend tests for:
- FlashAttention masking correctness (including cached decoding offsets).
- KV-cache correctness across layers, batches, and truncation.
- Optimizer rewrite decisions for each supported attention layer type.

### 5.2 Integration tests (facade end-to-end)
Create tests that only use the public facade:
1) Build a small transformer with `PredictionModelBuilder` + `ConfigureInferenceOptimizations()`.
2) Call `Predict()` and assert:
   - No exceptions.
   - Output shape matches expected.
3) Call `BeginInferenceSession()` and run multiple steps:
   - Verify caches don't leak across sessions.
   - Verify `Dispose()` resets state.
4) Validate attention layer coverage:
   - Ensure models using `AttentionLayer<T>`, `SelfAttentionLayer<T>`, and `GraphAttentionLayer<T>` either:
     - optimize safely, or
     - are explicitly skipped with internal diagnostics (but still function).

If `AiDotNet.Serving` has a test harness, add a serving integration test:
- Spin up a batcher with a model + config.
- Submit concurrent requests.
- Validate outputs and ensure no deadlocks/file-lock issues.

### 5.3 Performance smoke tests (optional)
- Benchmarks belong in benchmark projects; for this PR, a smoke test is enough:
  - Validate the optimized path is selected (internal diagnostics).

---

## 6) Acceptance Criteria Checklist

- [ ] No new public entrypoints besides builder/result (session may be nested under result).
- [ ] `ConfigureInferenceOptimizations()` has full effect in inference.
- [ ] KV-cache correctness for multi-layer models (no cross-layer corruption).
- [ ] Attention optimization supports major attention layer types used in the repo.
- [ ] Paged KV-cache can be enabled (backend selection + attention integration).
- [ ] Batching and speculative decoding are usable via facade and serving.
- [ ] Speculation + batching policy prevents throughput regressions under load (Auto backoff).
- [ ] Inference WOQ (INT8) works end-to-end with safe fallback.
- [ ] Multi-LoRA works per-request/per-sequence with cache isolation (KV reset on adapter change).
- [ ] Unit tests + integration tests cover the end-to-end wiring.

---

## 7) Resolved Decisions (from discussion)

- **Facade:** Keep public surface minimal; avoid public cache-control methods on the result.
- **`BeginInferenceSession()` shape:** Choose whatever is best; prefer a nested session type under `PredictionModelResult`.
- **`AttentionMasking=Auto`:** Assume causal in typical inference/session usage when task type isn’t set; provide opt-out/override via config.
- **Paged KV-cache:** Auto-enabled by default; users can opt-out via config.
- **Speculative decoding:** Serving-first; session support only if it fits cleanly without expanding public surface.
- **Cloning policy:** Improving cloning via better serialization/deserialization is acceptable to ensure a true deep copy.

## 8) Remaining Questions (small, but useful before coding)

1) Should a session support multiple independent sequences (e.g., `session.CreateSequence()` / `sequenceId`), or is “one session = one sequence” acceptable for now?
2) Do you already have a preferred public API for text generation (e.g., `Generate(...)`) elsewhere, or should speculative decoding remain strictly within `AiDotNet.Serving` for now?

---

## 9) MVP Sequencing (to raise implementation confidence)

This section turns the backlog into a concrete, low-risk execution order with explicit “first targets” and acceptance checks.
It is written so a junior engineer can start implementation without having to make major architectural decisions.

### 9.1) MVP-0: Guardrails (do first)
1) Keep public API surface unchanged:
   - Only `PredictionModelBuilder` and `PredictionModelResult` are user-facing.
   - Sessions remain nested under `PredictionModelResult`.
   - All new inference types remain `internal` (kernels/caches/schedulers/draft models).
2) Add internal policy switches (config-driven) but keep defaults safe:
   - If anything fails (unsupported model/layer), auto-disable that optimization and fall back to baseline inference.
3) Add internal diagnostics (non-user-facing) to record which optimizations were applied and why others were skipped.

Acceptance:
- `dotnet build AiDotNet.sln` passes.
- Existing unit tests still pass (warnings acceptable, no new failures introduced by MVP-0).

### 9.2) MVP-1: Speculative Decoding in Sessions + Serving with Auto Policy

**Goal:** Speculation is available wherever sessions are used when `EnableSpeculativeDecoding=true`, but it does not tank throughput under load.

**First target method:** Classic “draft-model speculation” (existing draft model support) with a new internal policy layer.

Implementation steps:
1) Introduce a single internal decision point (used by both session and serving):
   - “Should we speculate this step?” and “What depth should we use?”
2) Implement `SpeculationPolicy=Auto` (internal default recommendation):
   - Inputs to policy:
     - rolling accept-rate
     - current batch size / queue depth (serving)
     - KV-cache pressure (paged block availability, optional)
   - Behavior:
     - Under high batching load: reduce depth or disable speculation.
     - Under low load / latency-sensitive: increase depth up to configured max.
3) Respect `EnableBatching` and `EnableSpeculativeDecoding` together:
   - When both true, `Auto` policy prevents “double spending” compute.
   - Provide `ForceOn/ForceOff` to override for experimentation (still configured via builder, not new public APIs).

Defaults:
- Sessions: if `EnableSpeculativeDecoding=true` and `SpeculationPolicy=Auto`, speculate when accept-rate is good and batch load is low/moderate.
- Serving: if continuous batching queue is deep, speculation backs off automatically.

Acceptance:
- Integration tests show sessions still isolate state across sequences.
- Serving throughput under batching load does not regress materially when `EnableSpeculativeDecoding=true` (policy must disable speculation under heavy load).

### 9.3) MVP-2: Inference Quantization “First Target” (Weight-Only INT8)

**Goal:** Get a real inference quantization win with minimal kernel churn and low correctness risk.

**First target mode:** **Weight-only INT8 (WOQ)** for dense/linear matmuls in transformer blocks:
- Q/K/V projections
- output projection
- FFN projections

Non-goals for MVP-2 (explicitly deferred):
- Activation quantization (INT8) (Phase 2)
- KV-cache quantization (INT8/FP8) (Phase 3)
- INT4 weight-only (Phase 4)

Implementation steps:
1) Add internal quantized weight containers (per-channel scales) and an `internal` matmul kernel that supports:
   - FP32/FP16 activations * INT8 weights → FP32/FP16 output
2) Add configuration wiring via `InferenceOptimizationConfig` (builder-only surface):
   - `EnableWeightOnlyQuantization` (default false until validated)
   - `WeightQuantizationBits` (start with 8)
   - `QuantizationCalibrationMode` (Auto/MinMax/Percentile/KL) — for WOQ, “calibration” is mostly scale estimation; keep it simple initially.
3) Selection rules:
   - Only apply WOQ when model/task matches supported inference path (e.g., transformer-style models).
   - Skip layers not yet supported; do not partially quantize in ways that break determinism.
4) Agent support (optional but planned):
   - Agents can recommend enabling WOQ for serving workloads; still configured via builder.

Acceptance:
- Accuracy within tolerance against FP baseline on deterministic tests.
- Measurable memory reduction for weights (and ideally throughput gain on CPU/GPU depending on engine).

### 9.4) MVP-3: Multi-LoRA (Per-Session/Per-Request Adapter Selection)

**Goal:** Multiple LoRA adapters can be used concurrently (serving) or per-sequence (sessions) without exposing LoRA internals publicly.

First target behavior:
1) Adapter selection:
   - Serving: adapter ID selected per request (e.g., metadata field in serving request model).
   - Sessions: adapter ID set at sequence creation time (internal hook; no new public surface required beyond existing builder/result/session shape).
2) Weight handling:
   - Never mutate base weights.
   - Cache merged weights per adapter ID (and per precision/quantization mode) to avoid recomputing merges.
3) KV-cache interaction rules:
   - If adapter changes for a given sequence, **reset KV-cache** for that sequence (deterministic + correctness first).

Non-goals for MVP-3 (defer):
- Multi-adapter composition beyond simple “single adapter at a time” (Phase 2: merge/stack).
- Hot-swapping adapters mid-generation without KV reset (Phase 3: advanced cache partitioning).

Acceptance:
- Two concurrent sequences using different adapters produce different outputs for same input.
- Same adapter reused across sequences hits merge cache and remains deterministic.

### 9.5) Phase 2+ (after MVPs)
1) Dynamic speculation improvements (speculative scheduling refinements).
2) Medusa vs EAGLE:
   - Recommended order: implement **Medusa first** if it fits the model architecture best (multiple lightweight heads), then add EAGLE if needed.
3) Activation quantization (INT8) and then KV-cache quantization (INT8/FP8), including paged KV-cache support.
4) Multi-adapter LoRA composition (merge/stack), plus better cache invalidation rules.
