using AiDotNet.HyperparameterOptimization;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Optimizers;
using AiDotNet.Preprocessing.DataPreparation;
using AiDotNet.Regularization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.ConfigureMethodCoverage;

/// <summary>
/// Bucket 7 — Configure* methods that affect the training pipeline
/// auxiliaries: regularization, data preparation, hyperparameter
/// optimization. Each test exercises an observable side-effect of the
/// configure call on the training trajectory.
/// </summary>
[Collection("ConfigureMethodCoverage")]
public class Bucket7_TrainingPipelineAuxTests : ConfigureMethodTestBase
{
    private readonly ITestOutputHelper _output;
    public Bucket7_TrainingPipelineAuxTests(ITestOutputHelper output) { _output = output; }

    /// <summary>
    /// ConfigureRegularization — verifies the configured regularization
    /// instance is propagated to the gradient-based optimizer that owns
    /// the L1/L2/elastic-net term during gradient application. Without
    /// the wiring fix in this PR, <c>_regularization</c> was set on the
    /// builder by the configure call but never read anywhere else in
    /// src/ — the optimizer always used its default L2 regardless of
    /// what the user asked for.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureRegularization_NoRegularization_ReachesGradientOptimizer()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        // The user wants NO regularization. Without the SetRegularization
        // wiring fix, the optimizer keeps its default L2Regularization
        // and silently applies it.
        var sentinel = new NoRegularization<float, Tensor<float>, Tensor<float>>();
        var adam = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null);

        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigureOptimizer(adam)
            .ConfigureRegularization(sentinel)
            .BuildAsync();

        // GradientBasedOptimizerBase exposes ActiveRegularization as a
        // public read-only property (promoted from the test-only
        // GetRegularizationForTests accessor in this PR's review — test
        // coupling on production APIs was flagged for removal).
        // Contract: after BuildAsync, the optimizer's regularization is
        // the user-supplied instance. Stored-but-not-consumed would
        // leave it at the default L2.
        Assert.Same(sentinel, adam.ActiveRegularization);
    }

    /// <summary>
    /// ConfigureDataPreparation — the data prep pipeline runs in
    /// BuildSupervisedInternalAsync (FitResample call) when there's at
    /// least one row operation. Adds a recording operation that captures
    /// FitResample invocations, then asserts post-build that the
    /// recorder saw the call.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureDataPreparation_WithStep_ActuallyRunsFitResample()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        var recorder = new RecordingRowOperation<float>();

        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigureDataPreparation(prep => prep.Add(recorder))
            .BuildAsync();

        // BuildSupervisedInternalAsync calls _dataPreparationPipeline
        // .FitResampleTensor (or FitResample) when the pipeline is
        // non-empty (AiModelBuilder.cs:2349, 2619, 2692). A stored-but-
        // not-consumed regression would leave the recorder's counter at 0.
        int totalCalls = recorder.FitResampleCalls + recorder.FitResampleTensorCalls;
        Assert.True(totalCalls > 0,
            $"ConfigureDataPreparation added a step but BuildAsync never invoked FitResample on it (Matrix calls={recorder.FitResampleCalls}, Tensor calls={recorder.FitResampleTensorCalls}). Stored-but-not-consumed regression.");
    }

    /// <summary>
    /// ConfigureHyperparameterOptimizer — when both an HPO instance AND
    /// a search space are configured, BuildSupervisedInternalAsync calls
    /// <c>_hyperparameterOptimizer.Optimize(...)</c> at
    /// <c>AiModelBuilder.cs:2944</c>. The recording HPO overrides
    /// <c>Optimize</c> to capture the call so the test can assert the
    /// wiring actually fires.
    /// </summary>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public async Task ConfigureHyperparameterOptimizer_WithSearchSpace_ActuallyRunsOptimize()
    {
        var (features, labels) = MakeMemorizationSet();
        var loader = MakeCanaryLoader(features, labels);
        var model = MakeCanaryModel();

        var recordingHpo = new RecordingHyperparameterOptimizer<float, Tensor<float>, Tensor<float>>();
        var searchSpace = new HyperparameterSearchSpace();
        searchSpace.AddContinuous("learning_rate", 1e-4, 1e-2);

        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigureHyperparameterOptimizer(recordingHpo, searchSpace, nTrials: 1)
            .BuildAsync();

        Assert.True(recordingHpo.OptimizeCalls > 0,
            $"ConfigureHyperparameterOptimizer set the HPO + search space but BuildAsync never invoked Optimize (calls={recordingHpo.OptimizeCalls}). Stored-but-not-consumed regression.");
    }

    /// <summary>
    /// Identity row operation that records every FitResample / FitResampleTensor
    /// invocation so the test can assert the configure → build path actually
    /// invoked it. Returns the input as-is to avoid disturbing the training
    /// trajectory.
    /// </summary>
    private sealed class RecordingRowOperation<TNum> : IRowOperation<TNum>
    {
        public int FitResampleCalls;
        public int FitResampleTensorCalls;
        public bool IsFitted { get; private set; }
        public string Description => nameof(RecordingRowOperation<TNum>);

        public (Matrix<TNum> X, Vector<TNum> y) FitResample(Matrix<TNum> X, Vector<TNum> y)
        {
            FitResampleCalls++;
            IsFitted = true;
            return (X, y);
        }

        public (Tensor<TNum> X, Tensor<TNum> y) FitResampleTensor(Tensor<TNum> X, Tensor<TNum> y)
        {
            FitResampleTensorCalls++;
            IsFitted = true;
            return (X, y);
        }
    }

    /// <summary>
    /// Recording HPO that subclasses <see cref="RandomSearchOptimizer{T, TInput, TOutput}"/>
    /// and overrides <c>Optimize</c> to count invocations without actually
    /// running a search loop. Returns a minimal valid optimization result so
    /// downstream code in BuildSupervisedInternalAsync doesn't crash.
    /// </summary>
    private sealed class RecordingHyperparameterOptimizer<TNum, TIn, TOut>
        : RandomSearchOptimizer<TNum, TIn, TOut>
    {
        public int OptimizeCalls;

        public override HyperparameterOptimizationResult<TNum> Optimize(
            System.Func<System.Collections.Generic.Dictionary<string, object>, TNum> objectiveFunction,
            HyperparameterSearchSpace searchSpace,
            int nTrials)
        {
            OptimizeCalls++;
            // Short-circuit with a structurally-valid empty result so
            // the test only validates the wiring proof (OptimizeCalls > 0)
            // without the extra training round-trip that base.Optimize
            // would trigger via objectiveFunction. Calling base.Optimize
            // here would invoke the user's training loop nTrials times
            // for what should be a pure wiring assertion — adds flakiness
            // sources unrelated to the wiring claim.
            return new HyperparameterOptimizationResult<TNum>
            {
                BestParameters = new System.Collections.Generic.Dictionary<string, object>(),
                AllTrials = new System.Collections.Generic.List<HyperparameterTrial<TNum>>(),
                SearchSpace = searchSpace,
                TotalTrials = nTrials,
                CompletedTrials = 0,
                TotalTime = System.TimeSpan.Zero,
            };
        }
    }
}
