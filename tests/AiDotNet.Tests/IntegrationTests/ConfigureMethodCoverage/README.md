# Configure* method coverage tests

End-to-end integration coverage for the `Configure*` methods on `AiModelBuilder`.

## Why this exists

Field-testing in downstream consumers (HarmonicEngine) found several "drop-in"
Configure* methods that produce broken behavior with zero existing test coverage:

| Configure* | Bug | Status |
|---|---|---|
| `BuildAsync + ConfigureOptimizer(Adam)` | top1=0%, uniform output on Transformer LM | Upstream PR in flight |
| `ConfigureQuantization` (Int8Quantizer) | 0.36× inference slowdown (fake quantization, no real INT8 matmul) | AiDotNet#1342 |
| `EnableMemoryManagement` (GradientCheckpointing) | Wrong chain rule for non-elementwise ops | AiDotNet#1341 (fixed in Tensors PR #361) |
| FlashAttentionLayer<T> swap | top1=0%, top5=100% (uniform output), 3.76× slower instead of 2-4× faster | Not yet filed |
| `ConfigureFitnessCalculator` (CategoricalCE) | Collapses post-build model to uniform output | **Discovered by this suite** |
| `ConfigureModelRegistry` | BuildAsync throws "Model not found in registry" | **Discovered by this suite** |
| OpenCL DirectGpu backend | `SetKernelArg` 0xC0000005 access violation during MultiHeadAttention training | **Discovered by this suite** — worked around with `AiDotNetEngine.ResetToCpu()` in fixture |

Each of those would be caught by "train tiny model, assert top-1 > random chance
AND assert prediction spread non-zero". This suite is that bar.

## Test categorization

Configure* methods are grouped into 4 buckets:

1. **Training-pipeline** (`Bucket1_TrainingPipelineTests`) — affects how training works
   (ConfigureModel, ConfigureOptimizer, ConfigureDataLoader, ConfigureFitnessCalculator,
   ConfigureFitDetector, ConfigureRegularization).
2. **Acceleration** (`Bucket2_AccelerationTests`) — affects perf (ConfigureMixedPrecision,
   ConfigureJitCompilation, ConfigurePlanCaching, ConfigureGpuAcceleration,
   ConfigureMemoryManagement, ConfigureQuantization, ConfigureCompression,
   ConfigureWeightStreaming, ConfigureInferenceOptimizations).
3. **Quality-of-life** (`Bucket3_QualityOfLifeTests`) — observation/analysis
   (ConfigureProfiling, ConfigureTelemetry, ConfigureBenchmarking, ConfigureInterpretability,
   ConfigureAdversarialRobustness, ConfigureBiasDetector, ConfigureFairnessEvaluator,
   ConfigureCrossValidation, ConfigureExperimentTracker, ConfigureCheckpointManager,
   ConfigureTrainingMonitor, ConfigureModelRegistry).
4. **Out-of-scope-for-this-suite** — need their own dedicated infrastructure:
   ConfigureReasoning, ConfigureFederatedLearning, ConfigureAgentAssistance,
   ConfigureReinforcementLearning, ConfigureKnowledgeGraph, ConfigureAutoML,
   ConfigureFineTuning, ConfigureLoRA, ConfigurePipelineParallelism,
   ConfigureDistributedTraining, ConfigureRetrievalAugmentedGeneration,
   ConfigureCurriculumLearning, ConfigureMetaLearning, ConfigureSelfSupervisedLearning,
   ConfigureKnowledgeDistillation, ConfigureProgramSynthesis,
   ConfigureSafety, ConfigureAugmentation, ConfigureHyperparameterOptimizer,
   ConfigureDataPreparation, ConfigurePreprocessing, ConfigurePostprocessing,
   ConfigureExport, ConfigureCaching, ConfigureVersioning, ConfigureABTesting,
   ConfigureLicenseKey, ConfigureGpuDiagnostics, ConfigureDataVersionControl,
   ConfigureProgramSynthesisServing.

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
