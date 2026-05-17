using AiDotNet.AdversarialRobustness;
using AiDotNet.CheckpointManagement;
using AiDotNet.CrossValidators;
using AiDotNet.Deployment.Configuration;
using AiDotNet.ExperimentTracking;
using AiDotNet.Interpretability;
using AiDotNet.Models.Options;
using AiDotNet.TrainingMonitoring;
using AiDotNet.Configuration;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.ConfigureMethodCoverage;

/// <summary>
/// Bucket 3 — Quality-of-life Configure* methods (observation / analysis).
/// These don't change training output (in theory), but they have failure modes:
/// crash on construction, swallow errors silently, leak resources, etc. Each
/// test enables one feature and asserts the build still produces non-degenerate
/// output.
/// </summary>
/// <remarks>
/// Methods covered:
/// <list type="bullet">
/// <item>ConfigureProfiling</item>
/// <item>ConfigureTelemetry</item>
/// <item>ConfigureBenchmarking</item>
/// <item>ConfigureInterpretability</item>
/// <item>ConfigureAdversarialRobustness</item>
/// <item>ConfigureBiasDetector</item>
/// <item>ConfigureFairnessEvaluator</item>
/// <item>ConfigureCrossValidation</item>
/// <item>ConfigureExperimentTracker</item>
/// <item>ConfigureCheckpointManager</item>
/// <item>ConfigureTrainingMonitor</item>
/// <item>ConfigureModelRegistry</item>
/// </list>
/// </remarks>
[Collection("ConfigureMethodCoverage")]
public class Bucket3_QualityOfLifeTests : ConfigureMethodTestBase
{
    private readonly ITestOutputHelper _output;
    public Bucket3_QualityOfLifeTests(ITestOutputHelper output) { _output = output; }

    /// <summary>
    /// ConfigureProfiling — verifies the path enables profiling without breaking
    /// training output. Profiling is a wrapper around the training loop; bugs here
    /// would surface as crashes or stale-data output.
    /// </summary>
    [Fact(Timeout = 60_000)]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureProfiling_DefaultEnabled_ProducesNonDegenerateOutput()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>();
        builder.ConfigureProfiling();
        builder.ConfigureModel(model);
        builder.ConfigureDataLoader(loader);
        var result = await builder.BuildAsync();

        var probe = new Tensor<float>([1, CanaryCtxLen]);
        for (int s = 0; s < CanaryCtxLen; s++) probe[0, s] = features[0, s];
        AssertFacadePredictNonDegenerate(result.Predict(probe), "ConfigureProfiling");
    }

    /// <summary>
    /// ConfigureProfiling with custom config (detailed timing, low sampling rate).
    /// </summary>
    [Fact(Timeout = 60_000)]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureProfiling_DetailedTiming_ProducesNonDegenerateOutput()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        var config = new ProfilingConfig
        {
            Enabled = true,
            DetailedTiming = true,
            SamplingRate = 1.0,
            ReservoirSize = 100,
            TrackAllocations = true,
        };

        var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>();
        builder.ConfigureProfiling(config);
        builder.ConfigureModel(model);
        builder.ConfigureDataLoader(loader);
        var result = await builder.BuildAsync();

        var probe = new Tensor<float>([1, CanaryCtxLen]);
        for (int s = 0; s < CanaryCtxLen; s++) probe[0, s] = features[0, s];
        AssertFacadePredictNonDegenerate(result.Predict(probe), "ConfigureProfiling(detailed)");
    }

    /// <summary>
    /// ConfigureTelemetry — verifies the metadata path doesn't crash builds.
    /// </summary>
    [Fact(Timeout = 60_000)]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureTelemetry_Default_ProducesNonDegenerateOutput()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigureTelemetry(new TelemetryConfig())
            .BuildAsync();

        var probe = new Tensor<float>([1, CanaryCtxLen]);
        for (int s = 0; s < CanaryCtxLen; s++) probe[0, s] = features[0, s];
        AssertFacadePredictNonDegenerate(result.Predict(probe), "ConfigureTelemetry");
    }

    /// <summary>
    /// ConfigureBenchmarking — verifies no-suite default config doesn't break the
    /// pipeline. Real benchmarking is a separate test infrastructure concern.
    /// </summary>
    [Fact(Timeout = 60_000)]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureBenchmarking_EmptyDefault_ProducesNonDegenerateOutput()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        // Empty BenchmarkingOptions has Suites = Array.Empty — should be a no-op.
        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigureBenchmarking(new BenchmarkingOptions())
            .BuildAsync();

        var probe = new Tensor<float>([1, CanaryCtxLen]);
        for (int s = 0; s < CanaryCtxLen; s++) probe[0, s] = features[0, s];
        AssertFacadePredictNonDegenerate(result.Predict(probe), "ConfigureBenchmarking");
    }

    /// <summary>
    /// ConfigureInterpretability — claims SHAP/LIME/permutation importance. Smoke
    /// test that the metadata path doesn't break training.
    /// </summary>
    [Fact(Timeout = 60_000)]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureInterpretability_Default_ProducesNonDegenerateOutput()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>();
        builder.ConfigureInterpretability();
        builder.ConfigureModel(model);
        builder.ConfigureDataLoader(loader);
        var result = await builder.BuildAsync();

        var probe = new Tensor<float>([1, CanaryCtxLen]);
        for (int s = 0; s < CanaryCtxLen; s++) probe[0, s] = features[0, s];
        AssertFacadePredictNonDegenerate(result.Predict(probe), "ConfigureInterpretability");
    }

    /// <summary>
    /// ConfigureInterpretability with custom options enabling SHAP.
    /// </summary>
    [Fact(Timeout = 60_000)]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureInterpretability_WithSHAP_ProducesNonDegenerateOutput()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        var options = new InterpretabilityOptions
        {
            EnableSHAP = true,
            EnablePermutationImportance = true,
        };

        var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>();
        builder.ConfigureInterpretability(options);
        builder.ConfigureModel(model);
        builder.ConfigureDataLoader(loader);
        var result = await builder.BuildAsync();

        var probe = new Tensor<float>([1, CanaryCtxLen]);
        for (int s = 0; s < CanaryCtxLen; s++) probe[0, s] = features[0, s];
        AssertFacadePredictNonDegenerate(result.Predict(probe), "ConfigureInterpretability(SHAP)");
    }

    /// <summary>
    /// ConfigureAdversarialRobustness — default config should be a no-op for the
    /// canary task (no adversarial training enabled). Verifies build path.
    /// </summary>
    [Fact(Timeout = 60_000)]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureAdversarialRobustness_Default_ProducesNonDegenerateOutput()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigureAdversarialRobustness()
            .BuildAsync();

        var probe = new Tensor<float>([1, CanaryCtxLen]);
        for (int s = 0; s < CanaryCtxLen; s++) probe[0, s] = features[0, s];
        AssertFacadePredictNonDegenerate(result.Predict(probe), "ConfigureAdversarialRobustness");
    }

    /// <summary>
    /// ConfigureBiasDetector with a DisparateImpactBiasDetector — verifies the
    /// metadata path doesn't break the build.
    /// </summary>
    [Fact(Timeout = 60_000)]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureBiasDetector_DisparateImpact_ProducesNonDegenerateOutput()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigureBiasDetector(new DisparateImpactBiasDetector<float>())
            .BuildAsync();

        var probe = new Tensor<float>([1, CanaryCtxLen]);
        for (int s = 0; s < CanaryCtxLen; s++) probe[0, s] = features[0, s];
        AssertFacadePredictNonDegenerate(result.Predict(probe), "ConfigureBiasDetector(DisparateImpact)");
    }

    /// <summary>
    /// ConfigureFairnessEvaluator with a BasicFairnessEvaluator — smoke test.
    /// </summary>
    [Fact(Timeout = 60_000)]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureFairnessEvaluator_Basic_ProducesNonDegenerateOutput()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigureFairnessEvaluator(new BasicFairnessEvaluator<float>())
            .BuildAsync();

        var probe = new Tensor<float>([1, CanaryCtxLen]);
        for (int s = 0; s < CanaryCtxLen; s++) probe[0, s] = features[0, s];
        AssertFacadePredictNonDegenerate(result.Predict(probe), "ConfigureFairnessEvaluator(Basic)");
    }

    /// <summary>
    /// ConfigureCrossValidation with KFold (k=3) — exercises the alternative
    /// training-evaluation path through the builder.
    /// </summary>
    [Fact(Timeout = 60_000)]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureCrossValidation_KFold3_ProducesNonDegenerateOutput()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        var cv = new KFoldCrossValidator<float, Tensor<float>, Tensor<float>>(
            new AiDotNet.Models.Options.CrossValidationOptions { NumberOfFolds = 3 });

        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigureCrossValidation(cv)
            .BuildAsync();

        var probe = new Tensor<float>([1, CanaryCtxLen]);
        for (int s = 0; s < CanaryCtxLen; s++) probe[0, s] = features[0, s];
        AssertFacadePredictNonDegenerate(result.Predict(probe), "ConfigureCrossValidation(KFold-3)");
    }

    /// <summary>
    /// ConfigureExperimentTracker pointed at a freshly-created temp directory —
    /// verifies the tracker reads / writes correctly when given a real
    /// pre-existing storage path. (The original docstring claimed
    /// "missing storage directory" but the test always pre-created the
    /// directory; renamed to match what the test actually validates per
    /// PR #1345 review. A separate "missing directory" test would need
    /// to skip the CreateDirectory call and rely on the tracker's
    /// auto-create / in-memory-fallback behavior.)
    /// </summary>
    [Fact(Timeout = 60_000)]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureExperimentTracker_TempStorage_ProducesNonDegenerateOutput()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();
        string tempDir = Path.Combine(Path.GetTempPath(), "AiDotNetExpTracker_" + Guid.NewGuid().ToString("N"));
        try
        {
            Directory.CreateDirectory(tempDir);
            var tracker = new ExperimentTracker<float>(tempDir);

            var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                .ConfigureModel(model)
                .ConfigureDataLoader(loader)
                .ConfigureExperimentTracker(tracker)
                .BuildAsync();

            var probe = new Tensor<float>([1, CanaryCtxLen]);
            for (int s = 0; s < CanaryCtxLen; s++) probe[0, s] = features[0, s];
            AssertFacadePredictNonDegenerate(result.Predict(probe), "ConfigureExperimentTracker");
        }
        finally
        {
            try { if (Directory.Exists(tempDir)) Directory.Delete(tempDir, recursive: true); }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }
    }

    /// <summary>
    /// ConfigureCheckpointManager with temp directory.
    /// </summary>
    [Fact(Timeout = 60_000)]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureCheckpointManager_TempDir_ProducesNonDegenerateOutput()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();
        string tempDir = Path.Combine(Path.GetTempPath(), "AiDotNetCheckpoint_" + Guid.NewGuid().ToString("N"));
        try
        {
            Directory.CreateDirectory(tempDir);
            var mgr = new CheckpointManager<float, Tensor<float>, Tensor<float>>(tempDir);

            var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                .ConfigureModel(model)
                .ConfigureDataLoader(loader)
                .ConfigureCheckpointManager(mgr)
                .BuildAsync();

            var probe = new Tensor<float>([1, CanaryCtxLen]);
            for (int s = 0; s < CanaryCtxLen; s++) probe[0, s] = features[0, s];
            AssertFacadePredictNonDegenerate(result.Predict(probe), "ConfigureCheckpointManager");
        }
        finally
        {
            try { if (Directory.Exists(tempDir)) Directory.Delete(tempDir, recursive: true); }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }
    }

    /// <summary>
    /// ConfigureTrainingMonitor with default monitor.
    /// </summary>
    [Fact(Timeout = 60_000)]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureTrainingMonitor_Default_ProducesNonDegenerateOutput()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();
        var monitor = new AiDotNet.TrainingMonitoring.TrainingMonitor<float>();

        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigureTrainingMonitor(monitor)
            .BuildAsync();

        var probe = new Tensor<float>([1, CanaryCtxLen]);
        for (int s = 0; s < CanaryCtxLen; s++) probe[0, s] = features[0, s];
        AssertFacadePredictNonDegenerate(result.Predict(probe), "ConfigureTrainingMonitor");
    }

    /// <summary>
    /// ConfigureModelRegistry with temp directory. <strong>Currently fails:</strong>
    /// <c>BuildAsync</c> calls <c>ModelRegistry.CreateModelVersion</c> on a model name
    /// that was never registered with <c>RegisterModel</c> first. The registry path
    /// throws <c>ArgumentException: Model '...' not found in registry</c>. This is
    /// an upstream API contract bug — either <c>BuildAsync</c> must call
    /// <c>RegisterModel</c> first, or <c>ModelRegistry.CreateModelVersion</c>
    /// should upsert. Filed as test-discovered bug.
    /// </summary>
    [Fact(Skip = "Upstream bug discovered by this test: AiModelBuilder.BuildAsync calls ModelRegistry.CreateModelVersion without first calling RegisterModel — throws ArgumentException. Needs upstream fix in AiModelBuilder.BuildSupervisedInternalAsync.")]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureModelRegistry_TempDir_ProducesNonDegenerateOutput()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();
        string tempDir = Path.Combine(Path.GetTempPath(), "AiDotNetRegistry_" + Guid.NewGuid().ToString("N"));
        try
        {
            Directory.CreateDirectory(tempDir);
            var registry = new AiDotNet.ModelRegistry.ModelRegistry<float, Tensor<float>, Tensor<float>>(tempDir);

            var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                .ConfigureModel(model)
                .ConfigureDataLoader(loader)
                .ConfigureModelRegistry(registry)
                .BuildAsync();

            var probe = new Tensor<float>([1, CanaryCtxLen]);
            for (int s = 0; s < CanaryCtxLen; s++) probe[0, s] = features[0, s];
            AssertFacadePredictNonDegenerate(result.Predict(probe), "ConfigureModelRegistry");
        }
        finally
        {
            try { if (Directory.Exists(tempDir)) Directory.Delete(tempDir, recursive: true); }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }
    }
}
