# PR #431 — Uncertainty Quantification: Gap Analysis & Exhaustive MVP Plan

This document is an engineering gap analysis and implementation plan for PR #431 (Issue #418) to reach an “exhaustive MVP” for uncertainty quantification (UQ) while respecting AiDotNet’s facade philosophy:

- User-facing configuration through `PredictionModelBuilder`
- User-facing inference through `PredictionModelResult`
- Keep internal complexity hidden; expose only what is necessary for user value

## 0. Definitions

### “Exhaustive MVP” (for this PR)

For PR #431, “exhaustive MVP” means:

1. The facade workflow supports uncertainty for the common model/task families AiDotNet already supports (at least: regression + classification; optionally time series).
2. All major UQ families that are already partially implemented (or implied by code) are integrated end-to-end:
   - Monte Carlo Dropout
   - Deep ensembles
   - Conformal prediction (intervals/sets with guarantees)
   - Calibration (temperature scaling, ECE)
3. Defaults are safe, “industry standard”, and require minimal user input (opt-out rather than opt-in where appropriate).
4. Behavior is correct, deterministic when requested, thread-safe for concurrent inference, and documented.
5. Tests prove correctness, facade wiring, and concurrency safety.

## 1. Current State (what PR #431 already has)

### Facade integration (present, but partial)

- `PredictionModelBuilder.ConfigureUncertaintyQuantification(UncertaintyQuantificationOptions)` exists.
- `PredictionModelResult.PredictWithUncertainty(...)` exists and returns `(mean, variance)` as `TOutput`.
- `UncertaintyQuantificationOptions` exists with:
  - `Enabled`, `Method` (incl. `Auto`), `NumSamples`, `MonteCarloDropoutRate`, `RandomSeed`, `DenormalizeUncertainty`.

### Implementations / building blocks (present)

- Monte Carlo Dropout:
  - `MCDropoutLayer<T>`
  - `MCDropoutNeuralNetwork<T>`
- Bayesian layers:
  - `BayesianDenseLayer<T>`
  - `BayesianNeuralNetwork<T>`
- Deep ensembles:
  - `DeepEnsemble<T>`
- Calibration:
  - `TemperatureScaling<T>`
  - `ExpectedCalibrationError<T>`
- Conformal:
  - `SplitConformalPredictor<T>` (regression intervals)
  - `ConformalClassifier<T>` (prediction sets)

### Tests (present, but incomplete for “exhaustive”)

- Unit tests for: MC Dropout layer, temperature scaling, ECE
- A facade integration test scaffold: `UncertaintyQuantificationFacadeTests`

## 2. Key Gaps (what prevents “exhaustive MVP”)

### 2.1 Facade behavior gaps (core)

1. **Method selection is not actually integrated.**
   - Builder currently only applies MC Dropout (and fails/returns early for other methods).
   - `PredictWithUncertainty` only samples MC Dropout layers; it does not route to:
     - `BayesianNeuralNetwork<T>`
     - `DeepEnsemble<T>`
     - Conformal prediction
     - Calibration

2. **Thread-safety / concurrency risk.**
   - `PredictWithUncertainty` toggles MC mode by mutating dropout layers on the model instance.
   - If the same `PredictionModelResult` is used concurrently (common in serving), this can cause:
     - cross-request interference
     - nondeterminism even with seeding
     - corrupted dropout mode state

3. **Output contract is too narrow for “exhaustive” UQ.**
   - Returning only `(mean, variance)` is insufficient for:
     - classification uncertainty (entropy, confidence, calibrated probs)
     - conformal prediction sets / intervals
     - decomposition (aleatoric vs epistemic) where supported

4. **`Auto` behavior is underspecified.**
   - “Auto should choose best available method” is not implemented (and has no policy definition).

### 2.2 Model coverage gaps

1. **Non-neural models do not have an uncertainty pathway.**
   - AiDotNet supports many models besides neural networks; exhaustive MVP needs a baseline method that works broadly (industry standard is conformal or bootstrap-based ensembling).

2. **Multi-output support is unclear.**
   - For vector/tensor outputs: how should variance/intervals be returned (per-dimension vs aggregated)?

### 2.3 Conformal prediction gaps

1. **Facade integration is missing.**
   - Conformal predictors exist but there is no builder configuration path, training-time calibration workflow, or inference-time API.

2. **Calibration-set management is not defined.**
   - Split conformal requires a calibration dataset (separate from training/test). The facade currently doesn’t define how users provide this.

### 2.4 Calibration gaps

1. **Calibration is not wired into `PredictionModelResult`.**
   - Temperature scaling must be fit on validation/calibration data, then applied to inference outputs (classification probabilities).

2. **ECE is a metric, not an inference-time feature.**
   - It should appear in diagnostics/metadata/evaluation outputs, not required for normal inference.

### 2.5 Bayesian training gaps

1. **Bayesian layers are present but not “production wired”.**
   - To be truly MVP:
     - training must be supported end-to-end (loss, optimizers, serialization)
     - clear constraints on which base models can be Bayesian

### 2.6 Defaults, documentation, and configuration gaps

1. Defaults are only partially “industry standard”.
2. Documentation exists (`src/UncertaintyQuantification/README.md`) but does not cover:
   - method routing rules
   - classification vs regression outputs
   - concurrency guarantees
   - required data splits for conformal / calibration

## 3. Target User-Facing Outcomes (facade-first)

### 3.1 Minimal public API (recommended)

Keep existing `Predict(...)` unchanged.

Add/standardize a single facade UQ entrypoint:

- `PredictionModelResult.PredictWithUncertainty(...)` remains, but becomes a thin facade that routes to the configured strategy.

For richer outputs without exploding surface area, prefer:

- extending existing stats containers (e.g., `PredictionStats`) and/or
- returning a single small result type (if necessary) rather than many methods.

### 3.2 Proposed user-visible output (updated decision)

Introduce a single public result container (example name):

- `UncertaintyPredictionResult<TOutput>`

Recommended fields (exact naming TBD to match codebase conventions):

- `TOutput Prediction` (mean / point estimate; matches the “main” user-facing value)
- `TOutput? Variance` (optional; regression + probability-variance)
- `TOutput? StdDev` (optional convenience)
- `Dictionary<string, object> Metrics` (for scalar / per-sample metrics without API explosion)
  - Classification (must include to exceed industry standards):
    - `predictive_entropy`
    - `mutual_information` (epistemic)
    - optionally `expected_entropy` (aleatoric proxy)
- `object? Conformal` (typed sub-object recommended; see below)

Metrics integration requirements:

- Reuse the codebase's existing "metrics" conventions for train/validation/test reporting (where applicable).
- When a metric cannot be computed for a given method/model, the key must still be present and set to `0` (or `null` if you prefer nullable semantics later), so downstream consumers do not have to branch on missing keys.
- `Prediction` for classification should be **calibrated probabilities** (builder-time calibrated when configured), and the metrics should be computed from the probability distribution.

Conformal outputs should be represented by small, typed result types (public, minimal):

- `RegressionConformalInterval<TOutput>` (e.g., `Lower`, `Upper`)
- `ClassificationConformalSet` (e.g., `int[] ClassIndices` per sample)

Backward compatibility options:

- Keep the old tuple-returning method as `[Obsolete]` wrapper for regression-like outputs, OR
- keep the tuple overload but only as a convenience that maps from `UncertaintyPredictionResult<TOutput>`.

Stakeholder direction: classification UQ must include **entropy + mutual information + conformal prediction sets** (B + C + D).

### 3.3 How this fits AiDotNet evaluation/metrics (existing patterns)

AiDotNet already has a standard evaluation pipeline:

- `IModelEvaluator<T, TInput, TOutput>.EvaluateModel(...)` returns `ModelEvaluationData<T, TInput, TOutput>`
- `ModelEvaluationData` carries `TrainingSet`, `ValidationSet`, `TestSet` as `DataSetStats<T, TInput, TOutput>`
- `DataSetStats` stores:
  - `ErrorStats<T>` (error metrics like MSE/MAE, and classification metrics where applicable)
  - `PredictionStats<T>` (intervals + prediction-quality stats)
  - raw `Features`, `Actual`, `Predicted` (for downstream analysis)

Plan requirement: UQ diagnostics must align with this pattern so users (and internal AutoML / agents) get consistent reporting across training/validation/test.

Recommended integration (additive, low-risk):

- Extend `DataSetStats<T, TInput, TOutput>` with an optional `UncertaintyStats` container (name TBD to match conventions), e.g.:
  - per-dataset aggregates (ECE, NLL/Brier, coverage, average entropy/MI)
  - optionally per-sample summaries if needed (kept minimal and/or opt-in to avoid large memory use)
- Update (or provide an alternate) `IModelEvaluator` implementation to populate `UncertaintyStats` when:
  - the evaluated model is a `PredictionModelResult` with UQ enabled, OR
  - the model supports an internal UQ strategy interface (see §4.1)

Key requirement from stakeholder: always include both `predictive_entropy` and `mutual_information` keys in the returned UQ metrics (even if MI is not computable, set to `0` by default for consistency with existing non-nullable metrics patterns).

## 4. Architecture Proposal (internal-first, facade-friendly)

### 4.1 Internal strategy interface (not user-facing)

Create an internal strategy abstraction (example naming):

- `internal interface IUncertaintyStrategy<T, TInput, TOutput>`
  - `UncertaintyPrediction<TOutput> Predict(...)`
  - optionally: `Fit(...)` for conformal/calibration

`PredictionModelResult.PredictWithUncertainty`:

1. Validates config
2. Selects strategy (based on `options.Method` + model capabilities)
3. Executes strategy with required locking / cloning for safety
4. Returns mean + variance (and stores richer info in diagnostics/metadata if available)

### 4.2 Concurrency policy (must be explicit)

Recommended policy:

- `PredictionModelResult` is safe for concurrent `PredictWithUncertainty` calls.
- Any strategy that requires mutable model state must:
  - operate on a cloned model instance, OR
  - use per-call state that does not mutate shared model fields, OR
  - lock around the mutation window (last resort; can harm throughput).

### 4.3 Method routing policy for `Auto`

Define an internal decision tree, e.g.:

1. If user provided a conformal/calibration configuration and a calibration set exists → use conformal for intervals/sets.
2. Else if model implements an uncertainty-capable interface (e.g., internal strategy wrapper) → use it.
3. Else if neural network and MC Dropout layers available (or can be injected safely) → MC Dropout.
4. Else fallback → bootstrap ensemble (generic model-agnostic baseline).

## 5. Implementation Plan (Phased)

Each phase includes: goal, tasks, tests, acceptance criteria.

### Phase 0 — Design freeze and contract definition

Goal: lock the user-facing behavior and minimize API churn.

Tasks:
- Write the definitive “what user gets” for regression vs classification.
- Decide whether conformal/calibration require new builder methods or reuse `UncertaintyQuantificationOptions`.
- Decide how calibration data is provided (recommended: additional builder configuration, or inference-time overload).
- Define concurrency guarantees in `PredictionModelResult` XML docs.

Acceptance:
- A short spec section is added to `src/UncertaintyQuantification/README.md` describing:
  - routing rules
  - outputs
  - defaults
  - concurrency guarantees

### Phase 1 — Strategy routing + thread-safety foundation

Goal: turn UQ into a routed strategy rather than hard-coded MC Dropout toggling.

Tasks:
- Add internal strategy abstraction.
- Implement strategy selection for:
  - MC Dropout (existing logic)
  - “None/Disabled”
  - placeholder for others (throws internal NotSupported, returns zeros at facade)
- Make MC Dropout execution thread-safe (preferred: clone model per call; fallback: internal lock).

Tests:
- Concurrency test: multiple parallel `PredictWithUncertainty` calls must not corrupt dropout state.
- Determinism test: fixed seed + fixed inputs produce stable mean/variance.

Acceptance:
- No shared mutable mode toggles escape the strategy boundary.

### Phase 2 — Generic model-agnostic baseline (bootstrap / bagging)

Goal: provide UQ for non-neural models (industry-standard baseline).

Tasks:
- Add a model-agnostic uncertainty method:
  - bootstrap resampling + train N models (builder-time)
  - or inference-time bootstrap if training-time is not feasible for all models
- Integrate into `Auto` routing for non-neural models.

Tests:
- Regression: bootstrap variance non-zero for noisy data.
- Classification: bootstrap yields probability variance / entropy.

Acceptance:
- `PredictWithUncertainty` returns meaningful uncertainty for non-neural models.

### Phase 3 — Deep ensembles (neural-focused)

Goal: integrate `DeepEnsemble<T>` end-to-end through facade.

Tasks:
- Define how ensembles are built in `PredictionModelBuilder`:
  - clone base model N times
  - train independently with different seeds
  - store ensemble inside `PredictionModelResult` (internals)
- Route `PredictWithUncertainty` to ensemble strategy.
- Expose configuration options (numMembers, seed policy, aggregation).

Tests:
- Ensemble mean/variance matches manual aggregation.
- Reproducibility with seed policy.

Acceptance:
- Ensemble strategy works with the facade without exposing internal models.

### Phase 4 — Conformal prediction (regression + classification)

Goal: integrate existing conformal classes into the facade.

Tasks:
- Define calibration data workflow:
  - builder: accept `(xCal, yCal)` and store conformal calibrator, OR
  - inference: accept calibration set once and cache calibrated predictor
- Regression:
  - return interval(s) (lower/upper) in output scale
- Classification:
  - return prediction set(s)
- Add configuration options:
  - alpha / confidence
  - score function choices

Tests:
- Coverage tests on synthetic data (approximate but stable).
- Regression: interval contains true label at expected frequency (within tolerance).
- Classification: set size increases when model is uncertain.

Acceptance:
- Users can obtain intervals/sets from the facade without managing internal objects.

### Phase 5 — Calibration (temperature scaling) integrated

Goal: calibrated classification probabilities through the facade.

Tasks:
- Decide where calibration occurs:
  - builder-time (preferred) using validation set
  - or post-build `Calibrate(...)` on `PredictionModelResult` (avoid if minimizing API)
- Apply temperature scaling to logits/probabilities in inference.
- Record ECE in model metadata / diagnostics (not required for inference).

Tests:
- Temperature scaling reduces ECE on a known synthetic miscalibrated model.

Acceptance:
- Calibration is applied consistently and documented.

### Phase 6 — Bayesian layers / Bayes-by-Backprop integration (if in MVP scope)

Goal: ensure Bayesian layers are not “dead code”.

Tasks:
- Confirm training loop compatibility (loss, gradients, optimizers).
- Ensure serialization/deserialization works.
- Route UQ methods through Bayesian models.

Tests:
- BayesianDenseLayer produces non-zero epistemic uncertainty.
- Serialization round-trips.

Acceptance:
- Bayesian models train + infer with uncertainty through the facade.

### Phase 7 — Normalization and output-scale correctness

Goal: ensure uncertainty outputs match user-facing output scale.

Tasks:
- Expand variance denormalization support:
  - affine transforms (ZScore/MinMax/Robust) should scale variance correctly
  - document “non-linear transforms return normalized uncertainty”
- Define classification output conventions (variance over probs vs logits).

Tests:
- Variance scaling matches expected factor for affine normalizers.

Acceptance:
- Documented, consistent behavior across normalizers.

### Phase 8 — Performance and memory (production readiness)

Goal: avoid extreme overhead (sampling can be expensive).

Tasks:
- Reduce allocations in sampling loop:
  - avoid `List<Tensor<T>>` clones if possible; compute streaming mean/variance (Welford).
- Ensure dropout sampling does not allocate per token unnecessarily.
- Add a small benchmark or perf test if the repo already uses benchmarks in tests.

Acceptance:
- Sampling path uses O(1) extra memory (streaming aggregation).

## 6. Acceptance Checklist (PR ready-to-merge)

- Facade-only flow works:
  - `PredictionModelBuilder.ConfigureUncertaintyQuantification(...)`
  - `PredictionModelResult.PredictWithUncertainty(...)`
- `Auto` policy documented and implemented
- Regression + classification supported (with clear outputs)
- At least one model-agnostic UQ method exists (bootstrap or conformal)
- Concurrency-safe (parallel inference does not corrupt state)
- net471 compatible (no unsupported APIs; use helpers where required)
- Tests:
  - unit tests per method
  - facade integration tests
  - concurrency test
- Documentation:
  - module README updated with routing + outputs + examples

## 7. Open Questions (need your decision)

### 7.1 Resolved by stakeholder

1. Bayesian training (Bayes-by-Backprop) is **required for MVP** (end-to-end wiring for `BayesianDenseLayer<T>` / `BayesianNeuralNetwork<T>`).
2. Conformal calibration data must be supplied **at builder-time**.
3. Classification UQ must exceed industry standards and include **B + C + D**:
   - predictive entropy
   - mutual information
   - conformal prediction sets
4. Keep customizable components **public**; only purely internal plumbing should be `internal`.
5. Classification `Prediction` should be **calibrated probabilities**.
6. Entropy + MI metrics must **always be present**; when MI is not computable for a method/model, default to `0` (or later support nullable semantics if the broader codebase adopts `T?` metric patterns).

### 7.2 Remaining open questions (engineering)

1. Should we store UQ aggregate metrics inside:
   - `DataSetStats` (preferred for consistency with existing evaluation reports), OR
   - only on `UncertaintyPredictionResult<TOutput>` (simpler, but less integrated with the evaluation pipeline)?
2. Should the legacy tuple-returning `PredictWithUncertainty` be kept as `[Obsolete]` compatibility wrapper, or removed outright before merge?
