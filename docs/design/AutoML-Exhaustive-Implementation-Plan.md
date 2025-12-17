# AutoML Exhaustive Implementation Plan (v1)

Status: Draft (expanded, code-free)

Date: 2025-12-17

Purpose: Define a production-ready, facade-compliant plan to make AutoML "exhaustive" for AiDotNet while protecting IP (no weights/model recreation leakage).

---

## 1) Context

- Active PR: #444 (branch: `claude/fix-issue-403-011CUvxMg2tE1BB8Nm5u94H5`)
- Issue anchor: #403 (NAS implementations) is a subset of "Exhaustive AutoML".
- This document is intentionally code-free and focuses on architecture, task/metric coverage, and production readiness.

---

## 2) Non-Negotiables (Project Guardrails)

- Facade / Public API: users must build/train via `PredictionModelBuilder` and infer via `PredictionModelResult`.
- IP protection: users must not be able to recreate/clone models outside the system using exposed artifacts (trial history, metadata, etc.).
- Options-driven UX: one configuration object per feature area, with industry-standard defaults; allow optional injection of interfaces (nullable -> default internally).
- Architecture standards:
  - Avoid generic constraints; use `INumericOperations` for conversions/comparisons.
  - Use intermediate base classes between interfaces and concrete implementations.
  - One top-level type per file (classes/enums/interfaces).
  - Documentation must match existing XML doc style with "For Beginners" sections (for C# code).
- Compatibility: must remain compatible with `.NET Framework 4.7.1`.
- JSON: always `Newtonsoft.Json` (never `System.Text.Json`).

---

## 3) What "Exhaustive AutoML v1" Means Here

Exhaustive means:
- AutoML supports any `IFullModel<T, TInput, TOutput>` across all input/output types represented in the repo.
- AutoML supports all task families the repo can actually train and evaluate, with industry-recommended default metrics per task family and user override via configuration.
- RL is in-scope for AutoML v1 (environment-based training/evaluation through the existing facade RL path).
- AutoML is integrated behind the facade with minimal required user inputs.
- AutoML exposes a redacted trial history (no weights, no serialized model bytes, no raw hyperparameter values, no architecture dumps).

Non-goal (v1):
- "Any future research task under the sun" if the repo does not yet have an evaluatable training+prediction loop for it.

### 3.1 Compute Budget Presets (v1)

AutoML must ship with opinionated, industry-standard budget presets, while still allowing full user override.

Preset goals:
- `CI`: deterministic, very fast, intended for pipelines; prioritizes compilation + correctness over peak quality.
- `Fast`: quick iteration for local dev; good-enough model quality.
- `Standard`: default; balanced quality vs cost.
- `Thorough`: best-quality search under reasonable limits; intended for release candidates or offline tuning.

Each preset must define defaults for:
- time budget (overall wall-clock limit)
- trial budget (max trials)
- multi-fidelity policy (enabled/disabled; early stopping rules)
- evaluation rigor (single split vs CV where meaningful)
- parallelism (max concurrent trials; safe defaults)
- deterministic seeding policy (on by default)

Recommended first-pass defaults (to be tuned after benchmarking on representative datasets):
- `CI`: time limit 2–5 minutes; max trials 5–10; single split only; multi-fidelity off; parallelism 1.
- `Fast`: time limit 10–15 minutes; max trials 20–30; single split; multi-fidelity on for neural candidates; parallelism 1–2.
- `Standard` (default): time limit 30 minutes; max trials 100; single split by default, with CV enabled only when task adapters support it; multi-fidelity on for neural candidates; parallelism 2–4.
- `Thorough`: time limit 2 hours; max trials 300; CV enabled where supported; multi-fidelity on; parallelism 4–8 (bounded by available CPU/GPU and memory).

Important: presets must be task-aware. For example, “trial” for a tree model may mean a full training run, while for deep learning it should support multi-fidelity (short runs promoted to longer runs).

### 3.2 Facade Integration Requirements (AutoML)

AutoML must remain “options-first” behind `PredictionModelBuilder`:
- Typical users should configure AutoML using a single configuration options object with safe defaults.
- Advanced users may optionally inject interfaces (nullable -> internal defaults).
- AutoML must not require users to manually instantiate internal components or provide custom evaluators for common task families.

AutoML must also remain compatible with the existing facade rule:
- build/train via `PredictionModelBuilder`
- inference via `PredictionModelResult`

---

## 4) Current Task Surface in Repo (Inventory)

### 4.1 Task-type signals currently present

- `PredictionType` only distinguishes `Binary` vs `Regression` (insufficient for multi-class, multi-label, ranking, seq2seq, detection, segmentation, etc.).
- `NeuralNetworkTaskType` enumerates many task families (multi-class/multi-label, seq2seq, forecasting, detection/segmentation, RL, clustering, generative, ASR, translation, etc.).
- `TransformerTaskType` includes classification/regression/text generation/sequence tagging/translation.
- Graph ML task containers exist:
  - `NodeClassificationTask<T>`
  - `GraphClassificationTask<T>`
  - `LinkPredictionTask<T>`
  - `GraphGenerationTask<T>`
- Time series modeling has a full model family via `ITimeSeriesModel<T> : IFullModel<T, Matrix<T>, Vector<T>>` and `TimeSeriesModelType`.

### 4.2 Reality check

Enums describe potential coverage, but evaluation + metric extraction is currently only reliable for a subset end-to-end.

### 4.3 Task coverage matrix (current vs required)

Legend:
- Model exists: a meaningful `IFullModel` training+prediction loop exists in-repo for this task family.
- Evaluator ready: the default evaluation pipeline can score this task family without user-provided glue code.
- AutoML ready: AutoML can run trials and select a best candidate using the correct metric by default.

| Task family | Model exists | Evaluator ready | AutoML ready | Default metric (v1) | Notes / gaps |
|---|---:|---:|---:|---|---|
| Tabular regression (`Matrix/Vector`) | Yes | Partial | Partial | RMSE | Default evaluator omits task/prediction type configuration and over-assumes `Vector<T>` outputs for all tasks. |
| Binary classification (`Matrix/Vector`) | Yes | Partial | Partial | AUCROC (or F1 if imbalanced) | Current `PredictionType` is only binary vs regression; multi-class is not representable here. |
| Multi-class classification | Mixed | No | No | Macro-F1 | Requires multi-class-aware evaluation and output conventions (class id vs logits/probabilities). |
| Multi-label classification | Mixed | No | No | Micro-F1 | Requires multi-hot label format and proper metric implementation. |
| Time-series forecasting | Yes | Partial | Partial | SMAPE | Requires time-aware splits and horizon-aware evaluation. |
| Time-series anomaly detection | Yes | No | No | AUCPR | Requires explicit anomaly label conventions and threshold policy. |
| Ranking / recommendation | Mixed | No | No | NDCG@K | Metrics exist in parts of the codebase but not wired as task-aware evaluation with grouping. |
| Graph node classification | Yes | No | No | Accuracy / Macro-F1 | Graph models use bespoke evaluation methods; not integrated into `IModelEvaluator`/AutoML scoring. |
| Graph classification | Yes | No | No | Accuracy / Macro-F1 | Same as above. |
| Graph link prediction | Yes | No | No | AUCROC / hits@K | Same as above; protocol-dependent metrics. |
| Graph generation | Mixed | No | No | TBD (policy) | Requires explicit generation evaluation policy; may be domain dependent. |
| NLP classification | Mixed | No | No | Macro-F1 | Needs task adapters and tokenization/data pipeline integration. |
| Sequence tagging | Mixed | No | No | F1 (entity-level) | Needs task adapters and metrics (token-level vs span-level). |
| Translation | Mixed | No | No | BLEU/chrF | Metrics not implemented/wired into evaluation pipeline. |
| Text generation | Mixed | No | No | Perplexity | `MetricType` includes Perplexity but no unified implementation/wiring. |
| Vision image classification | Mixed | No | No | Accuracy / Macro-F1 | Requires consistent label formats and evaluation adapters for tensor outputs. |
| Object detection | Unknown | No | No | mAP | Requires standardized box formats and evaluation metrics. |
| Image segmentation | Unknown | No | No | mIoU / Dice | Requires standardized mask formats and metrics. |
| Generative modeling (images) | Mixed | No | No | FID / IS | Implementations exist as standalone metrics but not integrated; also requires feature extractor policy. |
| Unsupervised clustering | Unclear | No | No | Silhouette | Task exists in enums/metrics, but model+builder+evaluation integration is not end-to-end. |
| Dimensionality reduction | Mixed | No | No | Reconstruction error | Requires evaluation definition and builder support where labels are absent. |
| Reinforcement learning | Mixed | No | No | Avg episodic return | Requires environment-based evaluation (rollouts) and episode/step-based budgets. |

---

## 5) Current Evaluation + Metrics Plumbing (Inventory)

### 5.1 Evaluator implementation

- Only `DefaultModelEvaluator<T, TInput, TOutput>` exists as the main `IModelEvaluator`.
- It assumes predictions and targets can be converted to `Vector<T>` and evaluated via `ErrorStats<T>` and `PredictionStats<T>`.
- It does not set a prediction/task type for `ErrorStatsInputs<T>` / `PredictionStatsInputs<T>`, so they default to regression.
- Cross-validation defaults only apply to `Matrix<T>/Vector<T>`. For all other `TInput/TOutput`, a cross-validator must be supplied explicitly, which conflicts with "industry defaults everywhere".

### 5.2 Metrics reality

- `MetricType` is very broad, but:
  - `ErrorStats.GetMetric()` and `PredictionStats.GetMetric()` only expose a small subset.
  - Many `MetricType` entries are not reachable through the evaluator pipeline today.
- Advanced metrics exist as standalone components (e.g., Inception Score, FID) but are not integrated into the unified evaluator pipeline nor exposed as first-class evaluation outputs.

### 5.3 AutoML implementation reality (today)

AutoML is currently not production-correct end-to-end:
- The only concrete AutoML implementations in-repo are NAS-focused and do not perform full trial training + evaluation.
- `AutoMLModelBase` delegates evaluation to `IModelEvaluator` (if provided) but does not define how trials are trained; the fallback evaluation path is a placeholder.
- `ValidateConstraints` is a stub (always returns true), so constraint-based search is not implemented.
- The public `TrialResult` model stores raw hyperparameter values, which conflicts with IP protection requirements.
- The public `IAutoMLModel` contract includes methods that return or suggest raw hyperparameter values; this must be reconciled with facade/IP goals in Phase E.

---

## 6) Default Metrics Policy (v1)

This section defines "industry recommended defaults" as primary optimization metric + secondary reported metrics.

Note: where the evaluator lacks correct implementations today, the gap is explicitly called out in Section 7.

### 6.1 Supervised regression (tabular / general numeric)

- Primary: `RMSE` (minimize) unless user opts into robust; robust default can be `MAE` (minimize).
- Secondary: `R2`, `MAPE`/`SMAPE` (when meaningful), `MeanBiasError`.

### 6.2 Binary classification

- Primary: `AUCROC` (maximize) by default; switch to `F1Score` (maximize) when imbalance is detected or declared.
- Secondary: `AUCPR`, `Precision`, `Recall`, `BrierScore` (calibration), `CrossEntropyLoss` (if available).

### 6.3 Multi-class classification (single-label)

- Primary: Macro-F1 (maximize) or Log Loss (minimize) as calibrated alternative.
- Secondary: Accuracy, top-k accuracy, per-class precision/recall.
- Gap: current evaluation plumbing is binary-oriented; requires multi-class aware adapters/metrics.

### 6.4 Multi-label classification

- Primary: Micro-F1 (maximize).
- Secondary: macro-F1, per-label AP, AUCPR (macro/micro), Hamming loss.
- Gap: requires multi-label aware adapters/metrics and output shape conventions.

### 6.5 Ranking / recommendation (query-grouped)

- Primary: `NormalizedDiscountedCumulativeGain`@K (maximize).
- Secondary: `MeanAveragePrecision`@K, `MeanReciprocalRank`, hits@K.
- Gap: requires group-aware evaluation; current evaluator assumes independent samples.

### 6.6 Time series forecasting

- Primary: `SMAPE` (minimize) for business-like forecasting; allow `RMSE` for scale-sensitive use cases.
- Secondary: `MAE`, `MAPE` (when safe), interval coverage metrics where available.
- Gap: time-series-aware splitting and horizon-aware evaluation needed.

### 6.7 Anomaly detection (time series or tabular)

- Primary: `AUCPR` (maximize) for imbalance; fallback `AUCROC`.
- Secondary: precision/recall at chosen threshold; calibration metrics where applicable.
- Gap: requires clear definition of anomaly label format and thresholding policy.

### 6.8 Clustering (unsupervised)

- Primary: `SilhouetteScore` (maximize).
- Secondary: `DaviesBouldinIndex` (minimize), `CalinskiHarabaszIndex` (maximize).
- Gap: evaluator must support tasks where no `y` exists (builder overloads and/or task definitions).

### 6.9 Dimensionality reduction (unsupervised)

- Primary: reconstruction error (minimize) for reconstruction-based methods.
- Secondary: neighborhood preservation metrics (if implemented).
- Gap: missing unified evaluation support.

### 6.10 NLP generation / translation / ASR

- Primary (task-specific):
  - Translation: BLEU/chrF (maximize)
  - Summarization: ROUGE (maximize)
  - ASR: WER/CER (minimize)
  - Language modeling: Perplexity (minimize)
- Gap: most of these metrics are not implemented/wired into the evaluator pipeline today.

### 6.11 Vision detection / segmentation

- Detection primary: mAP (maximize) across IoU thresholds.
- Segmentation primary: mIoU or Dice (maximize).
- Gap: missing metrics and missing standardized prediction/label formats in evaluation.

### 6.12 Graph ML

- Node/graph classification primary: Accuracy or macro-F1 depending on imbalance.
- Link prediction primary: AUCROC / hits@K (protocol dependent).
- Graph generation: requires explicit policy choice (domain dependent); candidate: FID-like proxies where meaningful.
- Gap: graph models evaluate via bespoke methods today; not integrated into `IModelEvaluator`/AutoML scoring.

### 6.13 Reinforcement learning (RL)

- Primary: average episodic return (maximize) or success rate (maximize), depending on environment/task definition.
- Secondary: stability (variance of return), sample efficiency (return vs steps), constraint violations (safety), time-to-threshold.
- Gap: requires environment-based evaluation adapters (rollouts) and episode/step-aware splitting/budget definitions.

---

## 7) Gap Analysis (What Must Change To Be Truly Exhaustive)

### 7.1 Missing universal task definition for `IFullModel`

AutoML needs an internal task contract that can represent:
- supervised vs unsupervised vs self-supervised tasks
- output/label formats (scalar, class id, one-hot, multi-hot, sequence, boxes, masks, graph objects)
- split strategy (standard, stratified, group, time series, predefined splits)

Today, there is no unified way to infer task family from `IFullModel` alone.

### 7.2 Evaluator assumes vectorizable outputs

Many `IFullModel` implementations can output tensors, sequences, graphs, or structured results; a universal evaluator cannot require `ConvertToVector` as the primary path.

### 7.3 Classification evaluation is not production-correct by default

The default evaluator path does not set prediction type and relies on helpers that assume binary classification or regression.

### 7.4 Task families not integrated into the evaluation pipeline

- Graph tasks: bespoke evaluation methods not wired into `IModelEvaluator` or AutoML scoring.
- Transformers/NLP: task enums exist but evaluator does not compute task-specific metrics (BLEU/ROUGE/WER/etc.).
- Vision detection/segmentation: task enums exist but metrics and evaluation formats are missing.
- Generative metrics (IS/FID) exist but are not part of unified evaluation outputs.

### 7.5 Metric surface mismatch

`MetricType` is large, but AutoML metric extraction supports only a handful of metrics. AutoML must be able to optimize any metric it offers for a given task family.

### 7.6 IP protection conflict: trial history leaks hyperparameter values

AutoML trial results currently store raw hyperparameters and return them as-is. For monetization and IP goals, hyperparameter values must be redacted from public results.

### 7.7 AutoML search is missing core “trial execution” mechanics

To be production-ready, AutoML must:
- train each candidate trial (or run a well-defined multi-fidelity approximation) using the same training infrastructure as the facade
- evaluate trials using task-aware evaluators/adapters
- support early stopping, pruning, and failure isolation

Today, the AutoML surface does not consistently define or implement these responsibilities.

### 7.8 Hyperparameter optimization strategies are missing or incomplete

Exhaustive AutoML requires multiple complementary strategies to meet industry standards across problem shapes and budgets:
- Random search baseline (always)
- Bayesian optimization (recommended default for mixed spaces; e.g., TPE-style)
- Multi-fidelity search (HyperBand/ASHA-style scheduling for budgets)
- Evolutionary/Genetic strategies (robust for discrete/conditional spaces; also helpful for NAS)
- Optional grid search (small spaces only; mainly for debugging)

The repository currently has enums and documentation references implying several strategies, but does not provide production-ready implementations.

### 7.9 Model-family search spaces are not defined comprehensively

“Exhaustive” requires:
- a default search space per model family (bounded, safe, and task-aware)
- domain packs that provide task-appropriate search spaces/constraints (especially for neural/NLP/vision/time-series)
- a registry that maps model families to valid hyperparameters, constraints, and default optimizers

Today, the search-space surface exists only in limited form and is not populated exhaustively across the model catalog.

### 7.10 Ensemble/selection policies are not defined

Industry-standard AutoML typically includes:
- robust model selection across folds/splits
- optional ensembling (bagging/stacking/averaging) where it improves generalization
- tie-breaking policies (accuracy vs latency vs size)

These policies must be explicitly defined to claim “exhaustive” behavior.

### 7.11 RL evaluation and AutoML integration are not defined

If RL models are treated as AutoML-eligible (they implement `IFullModel` and are trainable through the facade’s RL path), then AutoML must define:
- how to parameterize environments/tasks for trials
- how to evaluate (rollouts, seeds, horizon limits)
- what the default objective is (return vs success rate) and how to report stability
- how budgets map (episodes, steps, wall time)

Without an RL-aware evaluator and budget policy, “exhaustive AutoML” will remain incomplete for RL task families.

---

## 8) Implementation Phases (All Phases Fully Defined)

Each phase includes: goals, deliverables, acceptance criteria, and a junior-friendly checklist.

### 8.0 First-Pass Sprint Plan (Junior-Friendly)

This sprint plan is aligned to Phases A–H. It is intentionally “first-pass” and should be revalidated after Phase A’s taxonomy is finalized.

Sprint 1 (Phase A):
- Deliver the canonical task taxonomy + default metric policy + required label/prediction formats.
- Add a “supported vs excluded” task list with rationale (exclusions are allowed only when training+prediction+evaluation cannot be made end-to-end in-repo in v1).

Sprint 2 (Phase B, tabular baseline correctness):
- Fix the default evaluator assumptions (task/prediction type must be explicit).
- Ensure tabular regression + binary classification work end-to-end with correct default metrics.
- Ensure cross-validation does not silently produce incorrect metrics (either works correctly or is explicitly disabled for unsupported task shapes).

Sprint 3 (Phase B/C, supervised classification expansion):
- Add multi-class + multi-label evaluation adapters and metric correctness tests.
- Ensure AutoML can optimize the selected default metric and report secondary metrics.

Sprint 4 (Phase B/C, time series):
- Add time-series-aware splitting and forecasting evaluation adapter.
- Add anomaly detection evaluation policy (labels + thresholding) and metrics wiring.

Sprint 5 (Phase B/C, ranking/recommendation):
- Add group-aware evaluation adapter (query/user grouping) and ranking metrics wiring (NDCG@K, MAP@K, MRR).

Sprint 6 (Phase B/C, graph ML):
- Integrate graph task containers into the evaluator pipeline (node/graph/link).
- Standardize evaluation outputs so AutoML scoring is consistent across graph tasks.

Sprint 7 (Phase B/C, NLP):
- Add task adapters for NLP classification and sequence tagging.
- Add translation/generation evaluation policies and metrics wiring (BLEU/chrF, ROUGE, WER/CER, perplexity).

Sprint 8 (Phase B/C, vision):
- Add standardized formats + evaluation adapters for detection and segmentation.
- Add mAP and mIoU/Dice metric wiring.

Sprint 9 (Phase D):
- Implement the internal AutoML registry mapping model families to supported tasks and searchable knobs.
- Add coverage enforcement so “exhaustiveness” cannot regress silently.

Sprint 10 (Phase E):
- Implement redacted public trial summaries + internal replay records.
- Ensure serialization and results exposure do not leak hyperparameter values or architecture details.

Sprint 11 (Phase F):
- Implement the trial execution engine (train/evaluate per trial) with failure isolation and pruning hooks.
- Implement baseline HPO strategies (random search + simple Bayesian/TPE) on top of the engine.

Sprint 12 (Phase F/H):
- Add multi-fidelity scheduling (ASHA/HyperBand-style) and budget promotion policies.
- Add initial ensembling options + final model selection policies.

Sprint 13 (Phase G):
- Populate the model-family registry/search spaces across the model catalog in a task-aware way.
- Add domain packs to keep defaults safe and “industry standard” per modality.

### 8.0.1 v1 User Stories (Acceptance-Oriented)

US1 (Beginner AutoML run):
- As a beginner, I can run AutoML through `PredictionModelBuilder` with minimal configuration and receive a trained `PredictionModelResult`.
- Acceptance: no custom evaluator/cross-validator required for supported task families; defaults produce a valid best model and a primary metric.

US2 (Task-aware defaults):
- As a user, if I do not specify task family or metric, AutoML chooses industry-standard defaults based on the task family and data shape.
- Acceptance: default metric selection is deterministic and documented; users can override.

US3 (Budget control):
- As a user, I can select a budget preset (CI/Fast/Standard/Thorough) or explicitly set time/trial limits.
- Acceptance: AutoML respects the budget and stops deterministically; results include a summary of budget used.

US4 (Correct evaluation per task):
- As a user, I can run AutoML for any supported task family (tabular, time series, ranking, graph, NLP, vision, unsupervised) and get correct metrics by default.
- Acceptance: evaluation does not rely on incorrect “vector-only” assumptions for non-vector tasks; adapters exist per supported family.

US5 (AutoML “exhaustiveness” enforcement):
- As a library maintainer, when a new model family is added, AutoML coverage cannot silently regress.
- Acceptance: registry/validation fails CI if a model family is AutoML-eligible but not registered or explicitly excluded.

US6 (IP-safe results):
- As a facade user, I can inspect AutoML trial history and model selection rationale without learning raw hyperparameter values or architecture internals.
- Acceptance: `PredictionModelResult` (and any other facade outputs) expose only safe summaries + fingerprints; no weights/bytes/hyperparameter values/architecture dumps in default output.

### Phase A - Task taxonomy + default metric policy

Goal: define what tasks AutoML supports and how AutoML chooses default metrics/splitting/evaluation.

Deliverables:
- Internal task definition design (doc + taxonomy list).
- Default metric policy mapping (primary + secondary metrics per task family).
- AutoML-eligible task families list (supported, excluded, rationale).

Acceptance criteria:
- Every AutoML run can declare its task family explicitly (no ambiguous "binary vs regression only").
- Default metric selection is deterministic and documented.

Junior checklist:
- Inventory all existing task signals (enums, options, model types, task containers).
- Define a canonical internal taxonomy that can map from those signals.
- Define per-task defaults: metric, maximize/minimize, CV strategy, preprocessing defaults.
- Define required label/prediction formats for each task family.

### Phase B - Universal evaluation adapters

Goal: evaluate any `IFullModel` without assuming `Vector<T>` outputs.

Deliverables:
- Adapter-based evaluation architecture (internal).
- At least one evaluator adapter per supported task family.
- Default splitting/CV strategies for non-`Matrix/Vector` tasks (where meaningful).

Acceptance criteria:
- AutoML can evaluate candidates for each supported task family without requiring user-supplied evaluators/CV.
- Evaluation output contains a standardized metric dictionary + task-specific artifacts needed for scoring.

Junior checklist:
- Implement adapters in priority order:
  1) Tabular `Matrix/Vector` (baseline correctness fixes)
  2) Tensor supervised tasks (classification/regression)
  3) Time series forecasting (horizon + time split)
  4) Graph tasks (node/graph/link; reuse existing task containers)
  5) NLP tasks (classification, tagging, translation, generation) with task-specific metrics
  6) Vision detection/segmentation formats + metrics
  7) Unsupervised tasks (clustering, dimensionality reduction) using unlabeled metrics
- Ensure each adapter supports deterministic seeding and strict budget enforcement.

### Phase C - Metrics expansion + wiring

Goal: implement and wire "industry standard" metrics for each task family into the evaluation pipeline and AutoML scoring.

Deliverables:
- Metric calculators for missing task families (see Section 6).
- Unified metric exposure so AutoML can optimize any supported metric per task family.
- Validation tests for metric correctness on small known examples.

Acceptance criteria:
- Each task family's primary metric is computed correctly and is selectable as an optimization objective.
- AutoML scoring supports more than the current limited subset of metrics.

Junior checklist:
- Add multi-class + multi-label metric support (macro/micro/weighted variants).
- Add ranking metric support with group handling.
- Add NLP generation/translation/ASR metric implementations and wiring.
- Add detection/segmentation metrics and standardized evaluation formats.
- Integrate existing generative metrics (IS/FID) into the evaluation surface.
- Ensure metrics are compatible with `.NET Framework 4.7.1` and follow numeric ops patterns.

### Phase D - AutoML registry + coverage enforcement

Goal: guarantee "exhaustiveness" doesn't regress when new models/task types are added.

Deliverables:
- Internal registry mapping candidate families to:
  - supported task families + input/output types
  - default searchable hyperparameters (internal-only)
  - supported optimizers + training recipes
  - constraints (latency/size/memory) and prerequisites
- Coverage enforcement (tests or validation step) that fails when registry is incomplete.

Acceptance criteria:
- Every AutoML-eligible model family is either registered or explicitly excluded with rationale.

Junior checklist:
- Build registry entries for each model family and/or `ModelType` group.
- Add explicit exclusions for models that cannot be evaluated/trained via facade yet.
- Add validation that prevents silent support gaps.

### Phase E - Trial history redaction (IP protection) + safe exposure

Goal: expose trial history that is useful but does not enable model recreation outside the system.

Deliverables:
- Public trial summary schema (no hyperparameter values).
- Internal replay schema (full details), stored only internally and excluded from default result serialization.
- Documentation describing what is and isn't exposed.

Acceptance criteria:
- Trial history exposed via facade outputs contains no weights, no serialized model bytes, and no raw hyperparameter values.
- Facade outputs do not include enough detail to recreate models outside AiDotNet by copying trial dumps.

Junior checklist:
- Replace "parameters dictionary exposed" with a redacted summary structure.
- Represent config fingerprint as a stable hash plus a small set of non-sensitive descriptors (model family, search strategy name, budget preset, etc.).
- Ensure serialized artifacts do not contain secret/internal config payloads by default.

### Phase F - Trial execution engine + HPO strategies

Goal: make AutoML trials real (train + evaluate) and support multiple complementary hyperparameter optimization strategies.

Deliverables:
- Internal “trial executor” that can:
  - build a candidate (model + training recipe)
  - train it under a budget
  - evaluate it via task adapters
  - report metrics + artifacts in a standardized way
- At least these HPO strategies implemented on top of the executor:
  - Random search (baseline)
  - Bayesian optimization for mixed spaces (recommended default)
  - Evolutionary strategy (for discrete/conditional spaces)
  - Multi-fidelity scheduler (ASHA/HyperBand-style)

Acceptance criteria:
- AutoML can run end-to-end for at least one tabular task family with real training + correct evaluation, using at least random search + one Bayesian strategy.
- Budget enforcement works (time/trials/multi-fidelity) and is deterministic under seeding.

Junior checklist:
- Start with random search + deterministic sampling over `ParameterRange`.
- Add pruning hooks (early stopping and budget-based termination) before adding advanced strategies.
- Add one Bayesian strategy for mixed spaces (prefer approaches that work well with categorical/discrete knobs).
- Add ASHA/HyperBand scheduling once the evaluator adapters are reliable.

### Phase G - Model-family search spaces + domain packs

Goal: define safe, bounded default search spaces and candidate sets for the full model catalog, without leaking IP.

Deliverables:
- Search-space definitions for each AutoML-eligible model family (bounded, task-aware).
- Domain packs (vision/NLP/time-series/tabular/graph) that provide:
  - default candidate sets
  - default search spaces
  - default preprocessing and constraints
- Documentation of what “eligible” means for a model family (inputs/outputs, training recipe availability, evaluation readiness).

Acceptance criteria:
- The registry (Phase D) is populated for all eligible model families or each omission is explicitly excluded with rationale.
- A user can run AutoML on supported modalities using defaults without manually selecting model types.

Junior checklist:
- Populate the registry in priority order: tabular -> time series -> graph -> NLP -> vision.
- Ensure each search space is bounded and has “safe defaults” for CI/Fast presets.
- Verify that public trial summaries never include raw hyperparameter values (even when internal search spaces are rich).

### Phase H - Ensemble + final selection policy

Goal: exceed industry baselines by adding robust selection and optional ensembling where appropriate.

Deliverables:
- Selection policy rules (metric direction, tie-breakers using constraints like latency/size).
- Optional ensemble mechanisms for supported task families (where it improves generalization).
- Consistent reporting of “best single model” vs “best ensemble” without leaking model internals.

Acceptance criteria:
- AutoML produces stable selections under deterministic seeding and does not regress single-model behavior when ensembles are disabled.
- Ensemble options are configurable and default to off unless they measurably improve results within the chosen budget preset.

Junior checklist:
- Implement tie-breakers (metric -> constraints -> simplicity).
- Add simple, safe ensembling first (e.g., averaging for compatible outputs) before stacking.
- Ensure inference through `PredictionModelResult` remains consistent and facade-friendly.

---

## 9) Trial History Redaction Policy (v1)

Allowed to expose:
- Trial id, timestamps, status, duration, budget used (epochs/time/trials).
- Primary + secondary metric values and direction (maximize/minimize).
- Model family identifier and safe capability labels (e.g., "tree-based", "neural", "time-series").
- A stable configuration fingerprint (hash) for internal reproducibility within AiDotNet.

Not allowed to expose:
- Raw weights, serialized model bytes, optimizer state, gradients.
- Exact hyperparameter values.
- Full architecture dumps / full computation graphs / NAS cell definitions.

### 9.1 Public vs Internal Trial Records (Schema-Level Guidance)

Public trial summary (safe to return from `PredictionModelResult` / facade):
- Identifiers: trial id, run id, status, timestamps, duration
- Budgets: preset name, time used, trial index, early-stopping flags
- Metrics: objective metric name/value/direction + secondary metrics
- Safe model descriptors: model family id, high-level tags/capabilities
- Fingerprints: stable configuration fingerprint (hash) + safe provenance (e.g., “search strategy name”)

Internal replay record (never exposed by default; used only for in-system reproducibility):
- Full hyperparameter assignments and search-space details
- All random seeds and split strategy definitions
- Full candidate model construction metadata required to reproduce inside AiDotNet
- Optional internal-only artifacts (e.g., logs, traces) that must not be serialized into default user results

Note: the current public `IAutoMLModel` surface includes methods that return or suggest raw hyperparameter values. v1 must reconcile this with IP protection by ensuring the facade results do not expose those values, and by moving any “raw hyperparameter” APIs behind internal-only pathways or premium-gated surfaces (design choice to be made in Phase E).
Decision (v1): hyperparameter values are redacted from facade outputs only (i.e., anything returned via `PredictionModelResult` / builder path). Non-facade AutoML surfaces may continue to expose raw hyperparameters.

---

## 10) Notes: NAS within Exhaustive AutoML

- NAS should be a general capability for neural candidates, but domain packs must supply:
  - task-appropriate search spaces (vision vs NLP vs time-series differ materially)
  - default constraints and budgets
  - task-aware evaluation (metrics and data handling)
