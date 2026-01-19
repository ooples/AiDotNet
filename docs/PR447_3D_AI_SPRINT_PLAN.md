# PR #447 Sprint Plan — 3D AI Models (Issue #399)

This document is the exhaustive, phased plan to take PR #447 (“3D AI capabilities: point clouds + neural radiance fields”) from its current state to production-ready, while preserving AiDotNet’s facade philosophy: end users primarily interact via `AiModelBuilder` (build/train) and `AiModelResult` (inference).

## Goals (User-Facing Outcome)

- No change to user-facing entry points:
  - Users can still provide a concrete model via `AiModelBuilder.ConfigureModel(...)`.
  - AutoML can still select a model automatically when enabled.
  - Agents can still recommend/assemble a model and configuration.
- Add production-ready 3D-focused models and workflows:
  - Point cloud processing models.
  - Neural radiance field models.
  - Industry-standard defaults for all new options, with opt-in customization via options classes.

## Non-Negotiable Constraints

- Public API surface area remains minimal and consistent:
  - No new “power user” public entry points that bypass the builder/result flow.
  - Prefer configuration option classes (with defaults) over exposing low-level delegates/callbacks.
- All new models must follow the existing architecture patterns:
  - Concrete model types must ultimately implement `IFullModel<T, TInput, TOutput>` (directly or through a base type).
  - Ensure compatibility with serialization/cloning patterns used by other models.
- Use AiDotNet’s optimized data types and numeric abstractions:
  - Prefer `Tensor<T>`, `Matrix<T>`, `Vector<T>`, and `INumericOperations<T>` patterns over raw arrays and `where T : INumber<T>` constraints.

## Current PR #447 Snapshot (What’s in Scope Today)

**Files added/modified (high-level):**
- Point cloud: `src/PointCloud/**` (PointNet, PointNet++, DGCNN, layers, interfaces, data).
- Radiance fields: `src/NeuralRadianceFields/**` (NeRF, InstantNGP, GaussianSplatting, ray + interface).
- Docs: `docs/3D_AI_Features.md`.
- Tests: a small set of unit tests for PointNet/NeRF.

**CI status (as of latest check):**
- Failing: `gate`, `SonarCloud Analysis`, `CodeQL Analysis` (skips some downstream jobs).
- Passing: `commitlint`, `Fix Commit Messages`, `Codacy Security Scan`, etc.

## Phase 0 — Pre-Implementation Audit (Gap Analysis Deliverables)

This phase is about producing a precise checklist of what must change before any large refactors.

### 0.1 Architecture Fit Check (Model Contracts)
- Confirm how existing “core” model families implement `IFullModel`:
  - Identify the canonical base class(es) used by other neural-network-based models.
  - Confirm expected overrides: `Predict`, `Train` (internal use), `Serialize/Deserialize`, `Clone/DeepCopy`, metadata.
- For each new 3D model:
  - Define the exact `TInput` / `TOutput` types to be supported.
  - Verify the model can be passed to `AiModelBuilder.ConfigureModel(...)` without adapters.
  - Verify `AiModelResult<T,TInput,TOutput>` can run inference for those types.

**Acceptance criteria:** For every 3D model, we have a one-page “contract sheet”:
- Model type name + namespace.
- Implements `IFullModel<T, TInput, TOutput>` (how/where).
- Supported input/output structures.
- Training expectations (if used through builder).
- Serialization and cloning policy.

### 0.2 Options Inventory (Defaults-First)
- Decide which options belong in:
  - Per-model options (`PointNetOptions`, `PointNetPlusPlusOptions`, `DgcnnOptions`, `NeRFOptions`, `InstantNgpOptions`, `GaussianSplattingOptions`).
  - Shared pipeline options (`PointCloudPipelineOptions`, `RadianceFieldPipelineOptions`) if needed.
- For each option:
  - Default value (industry standard).
  - “Auto” behavior (when user doesn’t specify).
  - Validation rules and failure modes.

**Acceptance criteria:** Options matrix with defaults and validation rules, written before code changes.

### 0.3 Integration Inventory (Facade Wiring)
- Identify required integration points in:
  - `src/AiModelBuilder.cs` (AutoML/Agents selection flow, model family routing).
  - `src/Models/Results/AiModelResult.cs` (inference session flow, tensor shapes, batching expectations).
  - Any existing “model registry” / “task family” mapping used by AutoML and agents.

**Acceptance criteria:** A routing map that explicitly answers:
- “How does AutoML pick a 3D model?”
- “How do Agents recommend a 3D model?”
- “What defaults are applied when task type is not specified?”

## Phase 1 — CI/Quality Gate Stabilization (Production-Readiness Baseline)

### 1.1 Fix `gate` failures
- Use GraphQL to enumerate unresolved review threads.
- Address unresolved items one-by-one:
  - Fix the underlying issue.
  - Resolve the thread (or comment when it is a code-scanning thread that cannot be manually resolved).

### 1.2 Fix Sonar + CodeQL failures
- Remove or correct:
  - Invalid encodings / non-ASCII artifacts in XML docs (these frequently trip analyzers).
  - Any unsafe patterns flagged by CodeQL.
  - Any obvious performance pitfalls (alloc-heavy paths) flagged by analyzers.

**Acceptance criteria:** PR #447 is back to “green checks” with no workflow workarounds.

## Phase 2 — Point Cloud Core (Correctness + Performance + Options)

### 2.0 Internal `PointCloudData<T>` + Tensor Adapters (Facade-Safe)

**Decision:** Public-facing model inputs remain `Tensor<T>` (builder/result consistency). Internally, we introduce a `PointCloudData<T>` container plus helpers/adapters for correctness, ergonomics, and performance.

- `PointCloudData<T>` (internal helper/container)
  - Stores point positions and optional per-point attributes without forcing padding/masks by default:
    - Positions: `Tensor<T>` shaped `[N, 3]` (or `[B, N, 3]` when explicitly batched).
    - Optional attributes: normals/colors/intensity/features as tensors (explicitly typed and shaped).
    - Optional metadata: coordinate frame, units, and precomputed neighborhood indices (when beneficial).
  - Centralizes invariants and validation (finite values, expected ranks, consistent first dimension, etc.).

- Adapters (internal)
  - `PointCloudTensorAdapter` to convert between:
    - `Tensor<T>` representations used by `IFullModel<T, TInput, TOutput>` implementations.
    - The internal `PointCloudData<T>` representation used by dataset loaders, augmentations, and neighborhood construction.
  - Defines a single canonical `Tensor<T>` layout for point clouds (documented and enforced).

**Acceptance criteria:**
- The model types still accept `Tensor<T>` as `TInput` (no new public surface required).
- Internal code (training pipelines, augmentation, sampling/grouping) can operate on `PointCloudData<T>` without repeated shape checks and without unnecessary padding allocations.

### 2.1 Canonical Input/Output Shapes
- Establish and document supported representations:
  - Point clouds: `[numPoints, 3]` positions; optional normals/features `[numPoints, F]`.
  - Batching strategy: either `[batch, numPoints, channels]` tensors or a `PointCloudData<T>` batch abstraction.
- Define how labels are represented:
  - Classification: class index vector / one-hot (match existing conventions).
  - Segmentation: per-point labels.

### 2.2 PointNet
- Verify and complete:
  - Permutation invariance correctness.
  - Input transform (T-Net) and feature transform behavior.
  - Global feature vector caching rules (thread-safety / inference correctness).
- Options coverage:
  - Toggle transforms, hidden widths, pooling types, dropout, normalization, activations, etc.

### 2.3 PointNet++
- Implement the missing “industry standard” components if not present:
  - Farthest point sampling (FPS).
  - Local neighborhood grouping (ball query / kNN).
  - Set abstraction + feature propagation layers.
  - Multi-scale grouping (MSG) option.
- Add defaults aligned to the paper and typical implementations.

### 2.4 DGCNN
- Ensure dynamic graph construction is production-ready:
  - Efficient kNN (reuse existing tensor ops where possible).
  - EdgeConv implementation using AiDotNet tensor primitives.
  - Deterministic behavior when a random seed is configured.

**Acceptance criteria (Point Cloud):**
- Unit tests: shape checks, permutation invariance (PointNet), basic forward pass determinism.
- Integration tests: `AiModelBuilder -> Build -> AiModelResult -> Predict` for at least one point cloud task.

## Phase 3 — Radiance Fields Core (Correctness + Options + Scalability)

### 3.1 NeRF (baseline)
- Ensure correct volume rendering pipeline:
  - Ray sampling, stratified sampling, compositing.
  - Positional encoding implementation and parameterization.
  - Chunked inference (avoid OOM).
- Options coverage:
  - Encoding levels, sample counts, near/far, background handling, activation choices.

### 3.2 Instant-NGP (hash encoding)
- Complete “industry standard” parts if missing:
  - Multiresolution hash grid encoding.
  - Tiny MLP and level parameters.
  - Optional occupancy grid acceleration.
- Performance-focused defaults with opt-out switches.

### 3.3 3D Gaussian Splatting
- Define the production boundary:
  - MVP: inference/rendering interface + basic training hooks (if training is intended in AiDotNet scope).
  - If training is in scope: densification, pruning, and optimization schedule options.

**Acceptance criteria (Radiance Fields):**
- Unit tests: ray math, positional encoding, stable forward pass.
- Integration tests: builder/result flow for a tiny synthetic scene (minimal render sanity test).

## Phase 4 — AutoML + Agents Integration (Exhaustive Defaults)

### 4.1 AutoML model selection
- Extend AutoML model family selection logic to include:
  - Point cloud classification/segmentation.
  - Radiance-field training/inference tasks (if supported by AutoML).
- Ensure defaults when user does not specify a task:
  - Define a deterministic fallback (documented).

### 4.2 Agents recommendations
- Add rule-based recommendations for:
  - When to use PointNet vs PointNet++ vs DGCNN.
  - When to use NeRF vs Instant-NGP vs Gaussian splatting.
- Ensure the agent output maps to concrete option classes + builder configuration.

**Acceptance criteria:** A user can enable AutoML/Agents and get a valid 3D model chosen and configured without touching internal types.

## Phase 5 — Inference & Serving Alignment (Coordinate With Inference Optimization PRs)

This phase is explicitly dependent on what is already merged vs still in-flight in inference optimization PRs.

### 5.1 Dependency audit
- List which inference features are already merged in `master` vs in open PRs:
  - KV cache, paged attention/KV, batching, speculative decoding, session APIs, etc.
- Decide merge order:
  - If 3D inference can reuse existing infrastructure, integrate now.
  - If not, defer integration and track as a follow-on PR with explicit acceptance criteria.

### 5.2 Serving integration (optional, use-case driven)
- Only add serving-specific toggles where they provide clear value:
  - Multi-scene batching for radiance fields.
  - Render scheduling policies.
  - Safety/resource limits by default (timeouts, max rays, max points).

**Acceptance criteria:** 3D models can run inference reliably via `AiModelResult`; serving extensions are optional and isolated.

## Phase 6 — Serialization, Cloning, and Reproducibility

- Ensure 3D models support:
  - `Serialize/Deserialize` round-trip.
  - Deep clone where required (avoid shared mutable state between sessions).
  - Reproducible initialization with configured random seeds.

**Acceptance criteria:** Integration test that:
- Builds a model, serializes, reloads, and produces identical predictions for a fixed input/seed.

## Phase 7 — Documentation (Match Repo Standards)

- Update `docs/3D_AI_Features.md` to:
  - Use builder/result examples (primary flow).
  - Keep direct model construction examples only as “advanced” and still routed through `ConfigureModel`.
  - Normalize formatting and remove any encoding artifacts.
- Add a module-level README if needed (per your preference), e.g.:
  - `src/PointCloud/README.md`
  - `src/NeuralRadianceFields/README.md`

**Acceptance criteria:** Docs are consistent with existing repo doc structure and compile cleanly (no analyzer-breaking characters).

## Phase 8 — Tests & Benchmarks (Industry Standard + Beyond)

### 8.1 Unit tests
- Determinism with fixed seeds.
- Shape validation and error messages.
- Null/empty input handling.

### 8.2 Integration tests
- End-to-end builder/result flow for:
  - Point cloud classification (tiny synthetic dataset).
  - NeRF forward/render for a tiny synthetic scene.

### 8.3 Performance checks
- Add at least one benchmark per category (if the repo already has benchmark patterns):
  - Point cloud forward pass throughput.
  - NeRF render chunk throughput.

## Phase 9 — PR Hardening / Merge Readiness

- Re-run GraphQL unresolved-thread loop until no resolvable review threads remain.
- Ensure all required checks pass without special-casing.
- Update PR description to match `docs/PRODUCTION_READY_PR_PROCESS.md` checklist items:
  - Verification evidence and final SHA.

## Open Questions (Need Your Confirmation Before Implementation)

These are now resolved based on your guidance:

1. **Radiance fields training is in scope.**
   - Plan impact: NeRF / Instant-NGP / Gaussian Splatting phases include training loops, optimizer schedules, and dataset/input pipelines as first-class requirements (not “inference-only MVPs”).

2. **Standardize point cloud inputs on `Tensor<T>` (public-facing).**
   - Default plan: `TInput` for point cloud models is `Tensor<T>` (or other existing core tensor primitives) so the builder/result flow stays consistent and avoids proliferating new public data containers.
   - Why `PointCloudData<T>` might still be useful (internal): real point cloud workloads often have variable point counts, optional per-point attributes (normals, colors, intensities), and metadata (coordinate frames). A dedicated container can avoid padding/masks, preserve semantics, and make dataset loaders cleaner.
   - Recommendation: keep any `PointCloudData<T>`-style helpers **internal** (or “advanced/internal”) and provide adapters that materialize to/from `Tensor<T>` so the facade remains consistent.

3. **AutoML should treat 3D tasks as new task families.**
   - Plan impact: add explicit 3D task families (e.g., point-cloud classification/segmentation; radiance-field reconstruction/rendering) and wire them through AutoML + Agents routing rather than inferring via input shape alone.
