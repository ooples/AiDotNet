# Phase 2a — AiModelBuilder DI refactor migration plan

Status: **foundation in progress** (this PR is slice 1 of ~7–9 follow-up PRs).

## Why

`src/AiModelBuilder.cs` is a 9,511-line god class that owns ~30 distinct concerns. Audit finding #12 (2026-05) flagged it as:

- Untestable in isolation (single huge surface mixing concerns)
- Hard to extend (adding a new concern means surgery on the same file every contributor touches)
- A code-review bottleneck (any change to one concern requires reading 9.5K LoC of context)

The audit's recommendation: split into ~30 cohesive interfaces with default implementations, keep `AiModelBuilder` as a facade that composes them, and migrate one concern per PR with backward-compat semantics on every public method.

## Backward-compat contract (HARD INVARIANT)

Every public API surface that exists on master at commit `0019269e1` (the LICENSE change merge) MUST remain callable with identical signatures and observable behaviour at the end of every PR in this series. No `[Obsolete]` deprecations during the refactor — the externally-visible facade stays unchanged. Only the internal composition changes.

Specifically:
- Every existing `Configure*` method keeps its signature and return type (`IAiModelBuilder<T, TInput, TOutput>` for chaining).
- Every existing `BuildAsync` overload returns the same `AiModelResult<T, TInput, TOutput>` shape.
- Every `IConfiguredView<T, TInput, TOutput>` property continues to expose the same field values.
- Tests that exist on master at the start of Phase 2a MUST continue to pass on every subsequent commit.

## Concern groupings (the ~30 components)

Each row is one extraction PR. They are roughly ordered by independence (top of list = fewest cross-cuts to other concerns).

| # | Component | Configure methods consumed | LoC est. | Touched by BuildAsync? |
|---|---|---|---|---|
| 1 | **DataPipeline** (this PR) | `ConfigurePreprocessing` (3 overloads), `ConfigurePostprocessing` (3 overloads), `ConfigureDataLoader`, `ConfigureDataPreparation`, `ConfigureAugmentation` (2 overloads), `SetPostprocessingFitMaxRows` | ~700 | yes (provides X/y inputs) |
| 2 | **TrainingCore** | `ConfigureModel`, `ConfigureOptimizer`, `ConfigureRegularization`, `ConfigureFitnessCalculator`, `ConfigureFitDetector`, `ConfigureTrainingPipeline`, `ConfigureTrainingMonitor`, `ConfigureCheckpointManager`, `ConfigureMemoryManagement` | ~900 | yes (core training loop) |
| 3 | **CrossValidation** | `ConfigureCrossValidation` | ~150 | yes (split orchestration) |
| 4 | **Compliance** | `ConfigureBiasDetector`, `ConfigureFairnessEvaluator`, `ConfigureAdversarialRobustness`, `ConfigureSafety`, `ConfigureInterpretability` | ~600 | yes (post-train evaluation) |
| 5 | **Performance** | `ConfigureMixedPrecision`, `ConfigureInferenceOptimizations`, `ConfigureJitCompilation`, `ConfigurePlanCaching`, `ConfigureGpuAcceleration`, `ConfigureQuantization`, `ConfigureCompression` | ~500 | yes (BuildAsync prelude) |
| 6 | **WorkflowOrchestration** | `ConfigureFederatedLearning`, `ConfigureDistributedTraining`, `ConfigurePipelineParallelism`, `ConfigureReinforcementLearning`, `ConfigureAutoML`, `ConfigureHyperparameterOptimizer`, `ConfigureCurriculumLearning`, `ConfigureMetaLearning` | ~1,400 | yes (replaces standard train) |
| 7 | **AdvancedLearning** | `ConfigureKnowledgeDistillation`, `ConfigureLoRA`, `ConfigureFineTuning`, `ConfigureSelfSupervisedLearning` (2 overloads), `ConfigureProgramSynthesis`, `ConfigureProgramSynthesisServing` | ~800 | yes (training-loop variant) |
| 8 | **RagAndKnowledge** | `ConfigureRetrievalAugmentedGeneration`, `ConfigureKnowledgeGraph` | ~500 | partial (predict-path only) |
| 9 | **Storage** | `ConfigureExperimentTracker`, `ConfigureModelRegistry`, `ConfigureDataVersionControl`, `ConfigureVersioning`, `ConfigureCaching`, `ConfigureABTesting` | ~400 | yes (post-train artifacts) |
| 10 | **Observability** | `ConfigureBenchmarking`, `ConfigureProfiling`, `ConfigureTelemetry`, `ConfigureGpuDiagnostics` | ~300 | partial (metrics emission) |
| 11 | **AgentAndExport** | `ConfigureAgentAssistance`, `AskAgentAsync`, `ConfigureReasoning`, `ConfigureExport`, `ConfigureWeightStreaming` | ~600 | yes (artifact export) |
| 12 | **LicenseAndCompat** | `ConfigureLicenseKey`, license-key resolution in constructor, build-time license check | ~200 | yes (BuildAsync precondition) |

After all 12 component PRs land, `AiModelBuilder.cs` itself shrinks from ~9.5K LoC to roughly ~1.5K (constructor + facade delegation + the `BuildAsync` orchestration that composes the components).

## Per-PR shape

Every concern-extraction PR follows this template:

1. **Create the interface** under `src/Configuration/I<Component>.cs`. Interface methods mirror the existing `Configure*` signatures plus the readout properties that `BuildAsync` currently accesses via private fields.
2. **Create the default implementation** `src/Configuration/<Component>.cs`. This is where the migrated fields and Configure-method bodies live. The implementation is internally mutable (matches the existing builder's mutable-state pattern); the interface exposes get-only access to consumers.
3. **Update `AiModelBuilder`** to:
   - Hold a private `_<component>` field.
   - Initialise it in every constructor.
   - Delegate the moved `Configure*` methods to it (signatures unchanged).
   - Replace internal field-access sites in `BuildAsync` and partial-class siblings with property reads on `_<component>`.
4. **Add unit tests** under `tests/AiDotNet.Tests/UnitTests/Configuration/<Component>Tests.cs`. Tests exercise the component in isolation (without the rest of the builder) to prove it can be used by alternative composition roots.
5. **Verify no behaviour regressions** by running every test under `tests/AiDotNet.Tests/IntegrationTests/Regression/AiModelBuilder*` — these are the load-bearing end-to-end tests for the builder.

## Why composition over inheritance

I considered inheritance (`AiModelBuilder` → `BuilderWithDataPipeline` → `BuilderWithTraining` → …) and rejected it:

- Multiple-inheritance-like behaviour needs the diamond — every leaf class would need every concern.
- Adding a concern would require modifying the inheritance chain (versus simply registering a new component).
- Inheritance chains make `BuildAsync` orchestration brittle because hooks fire in fixed order.
- C# doesn't support mixins, so you can't compose orthogonal concerns at the type level.

Composition gives:
- Independent testability (mock or swap any one component).
- Free reordering of `BuildAsync` orchestration (each component exposes pure hooks).
- A natural seam for third-party plugins (consumers can replace a default with their own implementation by passing it to a constructor parameter).

## Why no `[Obsolete]` annotations during the refactor

The previous audit cycle marked unfinished VLA stubs `[Obsolete]` and the maintainer rejected it as a lazy hand-wave. The same principle applies here: the public Configure methods are not "obsolete" — they are the supported way to configure the builder and they will remain so indefinitely. The refactor changes implementation, not contract.

After all 12 PRs land, **if** the project decides to also offer the new DI components as a public alternative API surface (allowing consumers to bypass `AiModelBuilder` entirely and compose just the components they need), that's a separate API-design conversation tracked as Phase 2c.

## This PR's scope (slice 1 of 12)

- Adds `src/Configuration/IAiModelDataPipeline.cs` and `src/Configuration/AiModelDataPipeline.cs` (the DataPipeline component).
- Updates `AiModelBuilder` to hold a `_dataPipeline` field and delegate the 5 data-pipeline `Configure*` methods to it. Field-access sites in `BuildAsync` and partial-class siblings are replaced with property reads on `_dataPipeline`.
- Adds `tests/AiDotNet.Tests/UnitTests/Configuration/AiModelDataPipelineTests.cs`.
- This document.

Slices 2–12 are deliberately deferred to follow-up PRs so each one stays reviewable (~500–1,500 LoC) and so any one failed slice doesn't block the others.

## Sequencing dependencies between slices

| Slice | Blocked by |
|---|---|
| 1 — DataPipeline | (none — landing first) |
| 2 — TrainingCore | 1 (needs `IAiModelDataPipeline` to fetch X/y) |
| 3 — CrossValidation | 1, 2 (needs both data and training) |
| 4 — Compliance | 2 (post-train evaluation runs after training) |
| 5 — Performance | (none — orthogonal to data and training) |
| 6 — WorkflowOrchestration | 1, 2 (FL/DDP replace the standard train path) |
| 7 — AdvancedLearning | 1, 2 |
| 8 — RagAndKnowledge | (mostly orthogonal; minor coupling to 2 for serving) |
| 9 — Storage | 2 (consumes trained model) |
| 10 — Observability | 2 (emits training metrics) |
| 11 — AgentAndExport | 2, 9 (export depends on stored model) |
| 12 — LicenseAndCompat | (orthogonal — runs at constructor + BuildAsync entry) |

Slices 1, 5, 8, 12 can land in any order. Slices 2, 3, 4, 6, 7, 9, 10, 11 depend on 2 having landed first. The natural critical-path ordering: 1 → 2 → (3, 4, 6, 7, 9, 10, 11 in parallel) → … with 5, 8, 12 landing whenever convenient.

## Testing strategy

- **Per-component unit tests** prove each component works in isolation (called without the rest of the builder). These run fast (<100 ms each).
- **The existing `tests/AiDotNet.Tests/IntegrationTests/Regression/AiModelBuilder*` end-to-end tests** stay green on every slice. They are the load-bearing safety net — any behavioural regression in the facade surfaces there.
- **CI gates on both TFMs** (net10.0 + net471) for every slice.

## Open questions / known risks

- **Cross-component state**: a few `BuildAsync` code paths read from multiple components in interleaved patterns (e.g. mixed-precision config influences how the optimizer constructs its accumulators). The migration plan flags these as comments in the relevant slices' source; the resolution is generally to pass the cross-cutting context as a method parameter rather than letting components peek at each other.
- **Partial classes**: `AiModelBuilder` is a partial class with siblings in several `src/Domains/*` folders. The slice PRs must update those siblings in lockstep with the facade — covered by the per-slice "Update `AiModelBuilder` to delegate" step.
- **net471 compat**: every extracted component compiles for both `net471` and `net10.0`. No new language features that the older TFM doesn't support.

## Tracking

This PR opens the foundation. Subsequent slices each link back to audit finding #12 and reference this document by relative path.
