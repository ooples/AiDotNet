# Configure* method coverage tests

End-to-end integration coverage for the `Configure*` methods on `AiModelBuilder`.

## Why this exists

Field-testing in downstream consumers (HarmonicEngine) found several "drop-in"
Configure* methods that produce broken behavior with zero existing test coverage:

| Configure* | Bug | Status |
|---|---|---|
| `BuildAsync + ConfigureOptimizer(Adam)` | top1=0%, uniform output on Transformer LM | Upstream PR in flight ([#1351](https://github.com/ooples/AiDotNet/issues/1351)) |
| `ConfigureQuantization` (Int8Quantizer) | 0.36× inference slowdown (fake quantization, no real INT8 matmul) | [AiDotNet#1342](https://github.com/ooples/AiDotNet/issues/1342) |
| `EnableMemoryManagement` (GradientCheckpointing) | Wrong chain rule for non-elementwise ops | [AiDotNet#1341](https://github.com/ooples/AiDotNet/issues/1341) (fixed in Tensors PR #361) |
| FlashAttentionLayer<T> swap | top1=0%, top5=100% (uniform output), 3.76× slower instead of 2-4× faster | Not yet filed |
| `ConfigureFitnessCalculator` (CategoricalCE) | Collapses post-build model to uniform output | **Discovered by this suite** |
| `ConfigureModelRegistry` | BuildAsync throws "Model not found in registry" | **Discovered by this suite** ([#1367](https://github.com/ooples/AiDotNet/issues/1367)) |
| OpenCL DirectGpu backend | `SetKernelArg` 0xC0000005 access violation during MultiHeadAttention training | **Discovered by this suite** — worked around with `AiDotNetEngine.ResetToCpu()` in fixture |

Each of those would be caught by "train tiny model, assert top-1 > random chance
AND assert prediction spread non-zero". This suite is that bar.

## Test categorization

Configure* methods are grouped into 13 buckets. Buckets 1-3 are the
existing PR #1345 baseline (28 wired methods); buckets 4-13 are this
PR's extension closing the remaining surface.

1. **Training-pipeline** (`Bucket1_TrainingPipelineTests`) — affects how training works
   (ConfigureModel, ConfigureOptimizer, ConfigureDataLoader, ConfigureFitnessCalculator,
   ConfigureFitDetector). Note: `ConfigureRegularization` is exercised in Bucket 7
   (training-pipeline auxiliaries) where its wiring-bug-fix tests live.
2. **Acceleration** (`Bucket2_AccelerationTests`) — affects perf (ConfigureMixedPrecision,
   ConfigureJitCompilation, ConfigurePlanCaching, ConfigureGpuAcceleration,
   ConfigureMemoryManagement, ConfigureQuantization, ConfigureCompression,
   ConfigureWeightStreaming, ConfigureInferenceOptimizations).
3. **Quality-of-life** (`Bucket3_QualityOfLifeTests`) — observation/analysis
   (ConfigureProfiling, ConfigureTelemetry, ConfigureBenchmarking, ConfigureInterpretability,
   ConfigureAdversarialRobustness, ConfigureBiasDetector, ConfigureFairnessEvaluator,
   ConfigureCrossValidation, ConfigureExperimentTracker, ConfigureCheckpointManager,
   ConfigureTrainingMonitor, ConfigureModelRegistry).
4. **Deployment metadata** (`Bucket4_DeploymentMetadataTests`) — ConfigureCaching,
   ConfigureVersioning, ConfigureABTesting, ConfigureExport, ConfigureGpuDiagnostics.
5. **Build lifecycle** (`Bucket5_LifecycleTests`) — ConfigureLicenseKey,
   ConfigureDataVersionControl (with recording DVC asserting LinkDatasetToRun was called),
   ConfigureSafety.
6. **Pre/post-processing pipelines** (`Bucket6_PrePostProcessingTests`) — 3 overloads
   each of ConfigurePreprocessing + ConfigurePostprocessing, using a
   RecordingTensorTransformer to assert Fit/Transform actually fires.
   **WIRING BUG FIX**: ConfigurePostprocessing was stored-but-never-consumed;
   added PostprocessingPipeline slot on AiModelResultOptions/AiModelResult
   and invoked in Predict.
7. **Training-pipeline auxiliaries** (`Bucket7_TrainingPipelineAuxTests`) —
   ConfigureRegularization, ConfigureDataPreparation, ConfigureHyperparameterOptimizer.
   **WIRING BUG FIX**: ConfigureRegularization was stored-but-never-consumed;
   added SetRegularization on GradientBasedOptimizerBase and wired in
   AiModelBuilder.
8. **Augmentation** (`Bucket8_AugmentationTests`) — ConfigureAugmentation.
   **WIRING BUG FIX**: AugmentationConfig was completely unused;
   added CustomAugmenter slot + invocation in BuildSupervisedInternalAsync.
9. **Advanced AI features** (`Bucket9_AdvancedAITests`) — ConfigureReasoning,
   ConfigureRetrievalAugmentedGeneration (knowledge graph), ConfigureKnowledgeGraph,
   ConfigureKnowledgeDistillation.
   **WIRING BUG FIX**: KnowledgeDistillation options were stored on the builder
   but dropped at AiModelResultOptions; added the slot, captured on
   AiModelResult, and on the DIRECT-TRAINING / LoRA-wrapped paths the
   options now flow through to result.KnowledgeDistillationOptions
   without going through the KD-aware training loop. On the REGULAR
   (non-LoRA, non-direct-training) NN training path the second
   NotSupportedException throw site was intentionally KEPT — that gate
   surfaces the missing tape-based KD integration at Build time instead
   of silently substituting standard supervised training for the
   requested distillation (matches review feedback: fail-fast on
   genuinely-unintegrated paths beats silent fall-through). The
   Bucket9 test asserts the throw, then bucket-coverage on direct-
   training-eligible models will confirm options propagation once
   such a fixture lands.
10. **LoRA** (`Bucket10_LoRATests`) — ConfigureLoRA.
   **3 STACKED WIRING BUG FIXES**:
   - Lazy-layer wrap crash (IsShapeResolved guard + warmup forward).
   - GetInputShape()[0] read batch dim instead of feature dim (prefer
     weight-inferred dims, fall back to last-axis).
   - NormalOptimizer Clone-roundtrip incompatible with LoRA-wrapped
     models (route NN + LoRA through direct-training path).
11. **Hijack-path methods** (`Bucket11_HijackPathTests`) — ConfigureMetaLearning,
   ConfigureAutoML (IAutoMLModel overload), ConfigureReinforcementLearning,
   ConfigureAgentAssistance.
12. **Distributed/federated** (`Bucket12_DistributedTests`) — ConfigureDistributedTraining,
   ConfigurePipelineParallelism, ConfigureFederatedLearning.
13. **Program synthesis** (`Bucket13_ProgramSynthesisTests`) — ConfigureProgramSynthesis,
   ConfigureProgramSynthesisServing (both options + pre-built client overloads).

### Suite roll-up

| Bucket | Tests | Pass | Skip (tracked elsewhere) |
|---|---|---|---|
| 1. Training-pipeline (existing) | 6 | 3 | 3 (Adam batched [#1351](https://github.com/ooples/AiDotNet/issues/1351), default-opt [#1351](https://github.com/ooples/AiDotNet/issues/1351), CategoricalCE) |
| 2. Acceleration (existing) | 13 | 12 | 1 (INT8 cached-B [#1349](https://github.com/ooples/AiDotNet/issues/1349)/[#1363](https://github.com/ooples/AiDotNet/issues/1363)) |
| 3. Quality-of-life (existing) | 14 | 13 | 1 (ModelRegistry [#1367](https://github.com/ooples/AiDotNet/issues/1367)) |
| 4. Deployment metadata | 5 | 5 | 0 |
| 5. Lifecycle | 3 | 3 | 0 |
| 6. Pre/post-processing | 6 | 6 | 0 |
| 7. Training-pipeline aux | 3 | 3 | 0 |
| 8. Augmentation | 2 | 2 | 0 |
| 9. Advanced AI | 4 | 4 | 0 |
| 10. LoRA | 1 | 1 | 0 |
| 11. Hijack-path | 4 | 4 | 0 |
| 12. Distributed/federated | 3 | 3 | 0 |
| 13. Program synthesis | 3 | 3 | 0 |
| **Total** | **67** | **62** | **5** |

The 5 skips are all tracked by other open PRs
([#1351](https://github.com/ooples/AiDotNet/issues/1351),
[#1349](https://github.com/ooples/AiDotNet/issues/1349),
[#1363](https://github.com/ooples/AiDotNet/pull/1363),
[#1367](https://github.com/ooples/AiDotNet/issues/1367)) or are
[PR #1345](https://github.com/ooples/AiDotNet/pull/1345)'s own
discovered bugs — out of scope for this PR which extends coverage to
the Configure* methods those PRs don't touch.

### Real source bugs fixed in this PR

| Method | Bug | Fix |
|---|---|---|
| ConfigurePostprocessing (all 3 overloads) | Pipeline stored on builder but never invoked by result.Predict | Wired through AiModelResultOptions.PostprocessingPipeline → AiModelResult.Predict |
| ConfigureRegularization | Stored on builder but optimizer never read it | Added SetRegularization on GradientBasedOptimizerBase + wired in AiModelBuilder |
| ConfigureAugmentation | AugmentationConfig was completely unused | Added CustomAugmenter slot on AugmentationConfig + Apply invocation in BuildSupervisedInternalAsync |
| ConfigureKnowledgeDistillation | Options dropped at AiModelResultOptions | Added KnowledgeDistillationOptions slot — the second NotSupportedException throw was *kept* per reviewer feedback (fail-fast: surfacing the missing tape-based integration at Build time beats a silent fall-through to standard supervised training on the regular-NN path) |
| ConfigureLoRA (3 stacked bugs) | Lazy-layer wrap crash; CreateLoRALayer read batch dim; NormalOptimizer Clone-roundtrip incompatibility | IsShapeResolved guard + warmup forward; weight-inferred dims + last-axis fallback; route NN+LoRA through direct-training path |

## Adding a test for a new Configure* method

Pick the right bucket file, then copy this template:

```csharp
/// <summary>
/// ConfigureYourMethod — one sentence summarizing what the method does and what
/// failure mode this test is screening for.
/// </summary>
[Fact]
[Trait("category", "integration-configure-method")]
public async Task ConfigureYourMethod_DefaultConfig_ProducesNonDegenerateOutput()
{
    var (features, labels) = MakeMemorizationSet();
    var loader = MakeCanaryLoader(features, labels);
    var model = MakeCanaryModel();

    // If the Configure* method is on the IAiModelBuilder interface, fluent chain works:
    var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
        .ConfigureModel(model)
        .ConfigureDataLoader(loader)
        .ConfigureYourMethod()
        .BuildAsync();

    // If the method is only on the concrete AiModelBuilder class (e.g. ConfigureProfiling,
    // ConfigureMemoryManagement, ConfigureWeightStreaming, ConfigureInterpretability),
    // call it on a concrete builder reference before the interface methods:
    //   var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>();
    //   builder.ConfigureYourMethod();
    //   builder.ConfigureModel(model);
    //   builder.ConfigureDataLoader(loader);
    //   var result = await builder.BuildAsync();

    var probe = new Tensor<float>([1, CanaryCtxLen]);
    for (int s = 0; s < CanaryCtxLen; s++) probe[0, s] = features[0, s];
    AssertFacadePredictNonDegenerate(result.Predict(probe), "ConfigureYourMethod");
}
```

For perf claims (FlashAttention, Quantization, JIT), add a speedup assertion:

```csharp
double baselineTime = TimeAction(() => baselineResult.Predict(probe));
double featTime = TimeAction(() => featureResult.Predict(probe));
double speedup = baselineTime / featTime;
AssertSpeedupBetween(speedup, lowerBound: 0.8, upperBound: 10.0, "ConfigureYourMethod");
```

The 0.8× lower bound is non-negotiable — a drop-in optimization should never be slower
than baseline; this catches the Int8Quantization 0.36× and FlashAttention 3.76× slower
regressions.

## Filter

```bash
dotnet test --filter "FullyQualifiedName~ConfigureMethodCoverage"
dotnet test --filter "category=integration-configure-method"
```

## Notes

- All tests force CPU via `AiDotNetEngine.ResetToCpu()` in the
  `ConfigureMethodTestCpuFixture` collection fixture. This avoids the OpenCL
  `SetKernelArg` access-violation crash that surfaces under MultiHeadAttention
  forward pass on a development GPU.
- The canary model is intentionally tiny (V=8, B=8, dModel=16, layers=1, heads=2).
  Convergence isn't the goal — degenerate-output detection (uniform predictions,
  NaN/Inf, all-zero facade output) is the goal.
- Tests use `[Trait("category", "integration-configure-method")]` for filtering.
- Skipped tests document upstream bugs the suite discovered; remove the `Skip`
  attribute once the upstream fix lands.
