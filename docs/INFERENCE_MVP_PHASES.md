# Inference MVP Phases (PR #433) — Implementation Plan

This plan breaks down the remaining inference MVP work into phases that can be implemented and verified independently, while preserving the project’s facade philosophy (minimal public API surface) and default-first configuration approach.

## Goals

- Keep the public surface area limited to `PredictionModelBuilder` (build/train) and `PredictionModelResult` (inference), with a small number of carefully chosen inference entrypoints (e.g., session start).
- Use industry-standard defaults when users do not specify options; allow opt-out via `InferenceOptimizationConfig`.
- Ensure all PR #433 components are wired end-to-end: config → inference optimizer → optimized layers/kernels → serving integration.
- Exceed industry standards where feasible (paged KV-cache, FP16 KV-cache, batching, speculative decoding, dynamic scheduling), without exposing internal IP.

## Non-Goals (MVP)

- Exposing layer-level kernels or optimizers publicly.
- Full-blown public text-generation UX/API surface (tokenization, sampling, streaming) beyond what serving needs.
- GPU-specific paging kernels (CPU-first correctness is priority for MVP).

---

## Phase 0 — Baseline Safety & Diagnostics

**Outcome:** Optimization pipeline is observable and safe-by-default.

1. Add internal inference diagnostics:
   - Record decisions (enabled/disabled) per feature (KV-cache, masking mode resolution, flash attention, paging, speculative).
   - Record exceptions with a reason and feature tag (do not throw from the facade in normal inference).
2. Add stability guardrails:
   - Avoid mutating user-supplied model instances unless explicitly requested (e.g., `cloneModel: false` internal path).
   - When deep copy fails for a model, fall back to baseline inference and record diagnostics.
3. Verification:
   - Unit tests for optimizer decisions and non-throwing fallbacks.

---

## Phase 1 — Attention Rewrite Integration

**Outcome:** All supported attention layers are rewritten consistently based on config, and the optimizer can run end-to-end.

1. Wire attention rewrite decisions in `InferenceOptimizer`:
   - `MultiHeadAttentionLayer` → `FlashAttentionLayer` when enabled.
   - `MultiHeadAttentionLayer`/`FlashAttentionLayer` → cached attention when KV-cache enabled (causal default for sessions).
   - `SelfAttentionLayer` conversion path to multi-head for downstream rewrites (when feasible).
2. Add missing attention layer support:
   - Ensure `SelfAttentionLayer`, `AttentionLayer`, and `GraphAttentionLayer` are handled for cloning/serialization and/or have clear non-supported fallbacks.
3. Ensure cloning is truly deep:
   - Serialization/deserialization round-trip must preserve parameters exactly.
4. Verification:
   - Unit tests that assert rewritten layers exist as expected per config.
   - Clone tests verifying parameters match before mutation.

---

## Phase 2 — Paged Attention + Paged KV-Cache (Industry Standard Default)

**Outcome:** Paged KV-cache is available and auto-enabled by default for KV-cache workloads, with opt-out.

1. Default behavior:
   - `EnablePagedKVCache = true` by default; user can disable.
2. Wiring:
   - `InferenceOptimizer` initializes paged cache when paged attention layers exist; otherwise falls back to contiguous KV-cache.
3. Session behavior:
   - Sessions should prefer causal masking when user sets `AttentionMasking=Auto`.
4. Verification:
   - Unit tests for paged attention kernel and cache mechanics (block tables, COW).
   - Integration tests for session cache growth and reset.

---

## Phase 3 — KV-Cache Precision (FP16 default, opt-out)

**Outcome:** KV-cache can store keys/values in FP16 to reduce memory and improve capacity/throughput, with opt-out to FP32.

1. Configuration:
   - Add `KVCachePrecision` with default `Auto` selecting FP16 when possible.
2. Implementation:
   - KV-cache uses FP16 backing storage when enabled and `T` supports conversion.
3. Wiring:
   - `InferenceOptimizer` resolves cache data type and records decision.
4. Verification:
   - Unit tests for FP16 round-trip and memory usage calculations.

---

## Phase 4 — Inference Sessions (Multi-Sequence, Facade-Friendly)

**Outcome:** `PredictionModelResult.BeginInferenceSession()` supports multiple independent sequences for serving-style workloads without exposing internal implementation details.

1. API:
   - Keep public API minimal (session + sequence objects), do not expose layer internals.
2. Behavior:
   - Each sequence maintains independent KV-cache state and can `Reset()`.
   - Session is safe to use for multiple sequences in parallel (thread-safety where feasible).
3. Internal diagnostics:
   - Provide internal-only stats hooks to validate caching behavior in integration tests without expanding public API.
4. Verification:
   - Integration tests for:
     - Stateless `Predict()` behavior.
     - Sequence independence.
     - Reset restoring initial state.

---

## Phase 5 — Batching (Serving-First) + Resource Arbitration

**Outcome:** `EnableBatching` is honored in serving and session contexts, with guardrails around incompatibilities with speculation.

1. Serving integration:
   - Use `AiDotNet.Serving` to host batching behavior and backpressure.
2. Conflict policy:
   - Document and implement a policy when batching and speculative decoding both enabled:
     - Default: prioritize batching for throughput, optional override to prioritize speculation for latency.
3. Verification:
   - Serving-side tests around batch coalescing and max batch size enforcement.

---

## Phase 6 — Speculative Decoding MVP (Draft Model + Policy)

**Outcome:** `EnableSpeculativeDecoding` and `SpeculationPolicy` are fully wired, with a draft model option and safe defaults.

1. Configuration:
   - Draft model selection via `DraftModelType` (NGram, small neural) and speculation depth.
2. Session + serving:
   - Enable speculation wherever sessions are used when flag is enabled.
   - Serving integration for production usage (streaming/latency).
3. Verification:
   - Unit tests for policy decisions and safe fallback when draft model not available.

---

## Phase 7 — Dynamic Speculation & Alternative Speculators (Medusa/EAGLE)

**Outcome:** Add next-gen speculative methods and dynamic scheduling options.

1. Dynamic scheduling:
   - Adaptive speculation depth based on acceptance rate, queue pressure, and compute budget.
2. Alternative methods:
   - Add config hooks for Medusa/EAGLE-style multi-head draft proposals as a future opt-in.
3. Verification:
   - Bench-style tests (non-flaky) for acceptance-rate-driven behavior.

---

## Phase 8 — Inference Quantization (Gap-Closing)

**Outcome:** Extend quantization support beyond training into inference areas where it is industry standard.

1. KV-cache quantization:
   - Optional per-layer KV-cache quantization (e.g., int8) with dequant on read.
2. Weight-only quantization:
   - Optional weight-only quant for inference (e.g., int8/int4) with fast matmul paths.
3. Weight + activation quantization (advanced):
   - Add as opt-in; ensure correctness-first.
4. Verification:
   - Unit tests validating numerics and shape correctness.

---

## Phase 9 — Multi-LoRA (Serving-First, Secure Defaults)

**Outcome:** Multi-LoRA can be selected per request without leaking internal implementation details.

1. Serving integration:
   - Prefer selecting LoRA adapters from headers/metadata on serving side.
2. Session integration:
   - Optional adapter selection per sequence, but keep surface minimal.
3. Verification:
   - Serving tests for adapter routing and isolation.

---

## Release Checklist (Per Phase)

- `dotnet build AiDotNet.sln -c Release`
- Targeted `dotnet test` runs for touched areas
- Update docs and XML comments to match project conventions (summary/remarks + “For Beginners” sections where appropriate)
- Commit with conventional prefix (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`) in small, regular increments

