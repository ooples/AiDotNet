using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.ActivationFunctions;
using AiDotNet.CheckpointManagement;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.FitnessCalculators;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.MixedPrecision;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TrainingMonitoring;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.TrainingCallbacks;

/// <summary>
/// End-to-end tests for the facade training-callback feature: per-epoch monitor streaming,
/// user-registerable abortable callbacks, the in-box <see cref="HealthMonitorCallback{T}"/>,
/// and the result-surface observability fields. Everything is driven ONLY through the
/// <see cref="AiModelBuilder{T,TInput,TOutput}"/> facade + <c>BuildAsync</c>, mirroring the
/// capability-audit setup (tiny FeedForwardNeuralNetwork + InMemoryDataLoader).
/// </summary>
public class FacadeTrainingCallbackTests
{
    private const int InDim = 4;
    private const int N = 48;

    /// <summary>A monitor that records the session id the facade opens so the test can query its history.</summary>
    private sealed class CapturingMonitor : TrainingMonitor<float>
    {
        public string SessionId { get; private set; } = string.Empty;

        public override string StartSession(string sessionName, Dictionary<string, object>? metadata = null)
        {
            var id = base.StartSession(sessionName, metadata);
            SessionId = id;
            return id;
        }
    }

    private static (Tensor<float> x, Tensor<float> y) BuildData()
    {
        var rng = new System.Random(1234);
        var x = new Tensor<float>(new[] { N, InDim });
        var y = new Tensor<float>(new[] { N, 1 });
        for (int i = 0; i < N; i++)
        {
            float x0 = (float)(rng.NextDouble() * 2 - 1);
            float x1 = (float)(rng.NextDouble() * 2 - 1);
            float x2 = (float)(rng.NextDouble() * 2 - 1);
            float x3 = (float)(rng.NextDouble() * 2 - 1);
            x[i, 0] = x0; x[i, 1] = x1; x[i, 2] = x2; x[i, 3] = x3;
            y[i, 0] = 0.5f * x0 - 0.3f * x1 + 0.2f * x2 - 0.1f * x3 + 0.05f;
        }
        return (x, y);
    }

    private static FeedForwardNeuralNetwork<float> BuildModel(AdamOptimizer<float, Tensor<float>, Tensor<float>> optimizer)
    {
        var layers = new List<ILayer<float>>
        {
            new InputLayer<float>(InDim),
            new DenseLayer<float>(8, activationFunction: new ReLUActivation<float>()),
            new DenseLayer<float>(1, activationFunction: new IdentityActivation<float>()),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: InDim,
            outputSize: 1,
            layers: layers);
        return new FeedForwardNeuralNetwork<float>(
            arch, optimizer: optimizer, lossFunction: new MeanSquaredErrorLoss<float>());
    }

    private static AdamOptimizer<float, Tensor<float>, Tensor<float>> BuildOptimizer(
        int maxIterations, double learningRate)
    {
        var options = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = learningRate,
            MaxIterations = maxIterations,
            UseAdaptiveLearningRate = false,
            UseEarlyStopping = false,
            // Deterministic epoch count: never converge out of the loop early.
            Tolerance = 0.0,
            // Stream loss (not R^2) so TrainingProgress.Loss is a genuine loss value.
            FitnessCalculator = new MeanSquaredErrorFitnessCalculator<float, Tensor<float>, Tensor<float>>()
        };
        return new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, options);
    }

    [Fact(Timeout = 120000)]
    public async Task PerEpochStreaming_MonitorHistoryLengthEqualsEpochsRun()
    {
        const int epochs = 4;
        var (x, y) = BuildData();
        var optimizer = BuildOptimizer(epochs, 0.05);
        var model = BuildModel(optimizer);
        var monitor = new CapturingMonitor();

        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureDataLoader(new InMemoryDataLoader<float, Tensor<float>, Tensor<float>>(x, y))
            .ConfigureTrainingMonitor(monitor)
            .BuildAsync();

        Assert.NotNull(result);
        Assert.False(string.IsNullOrEmpty(monitor.SessionId));

        // The monitor received a genuine per-epoch time-series (not a single final snapshot).
        var lossHistory = monitor.GetMetricHistory(monitor.SessionId, "loss");
        Assert.Equal(epochs, lossHistory.Count);
    }

    [Fact(Timeout = 120000)]
    public async Task Callback_FiresOncePerEpoch()
    {
        const int epochs = 4;
        var (x, y) = BuildData();
        var optimizer = BuildOptimizer(epochs, 0.05);
        var model = BuildModel(optimizer);

        int epochEndCount = 0;
        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureDataLoader(new InMemoryDataLoader<float, Tensor<float>, Tensor<float>>(x, y))
            .ConfigureTrainingCallback(_ => { epochEndCount++; return true; })
            .BuildAsync();

        Assert.NotNull(result);
        Assert.Equal(epochs, epochEndCount);
        Assert.False(result.EarlyStopTriggered);
        Assert.Null(result.StopReason);
    }

    [Fact(Timeout = 120000)]
    public async Task Callback_ReturningFalse_AbortsTraining()
    {
        const int epochs = 3;
        var (x, y) = BuildData();
        var optimizer = BuildOptimizer(epochs, 0.05);
        var model = BuildModel(optimizer);

        int epochEndCount = 0;
        // Abort at the 2nd completed epoch (zero-based index 1). Training should stop there
        // rather than running the full 3 planned epochs.
        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureDataLoader(new InMemoryDataLoader<float, Tensor<float>, Tensor<float>>(x, y))
            .ConfigureTrainingCallback(p => { epochEndCount++; return p.Epoch < 1; })
            .BuildAsync();

        Assert.NotNull(result);
        Assert.Equal(2, epochEndCount); // epoch 0 (continue) + epoch 1 (abort)
        Assert.True(result.EarlyStopTriggered);
        Assert.NotNull(result.StopReason);
        Assert.Contains("abort", result.StopReason!, System.StringComparison.OrdinalIgnoreCase);
    }

    [Fact(Timeout = 120000)]
    public async Task HealthMonitorCallback_AbortsOnDivergingLoss()
    {
        const int epochs = 8;
        var (x, y) = BuildData();
        // Inject a NaN target so the very first epoch's loss becomes NaN.
        y[0, 0] = float.NaN;
        // A large learning rate also guarantees divergence even absent the NaN.
        var optimizer = BuildOptimizer(epochs, 5.0);
        var model = BuildModel(optimizer);

        var health = new HealthMonitorCallback<float>();
        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureDataLoader(new InMemoryDataLoader<float, Tensor<float>, Tensor<float>>(x, y))
            .ConfigureTrainingCallback(health)
            .BuildAsync();

        Assert.NotNull(result);
        Assert.True(result.EarlyStopTriggered, "Diverging/NaN training should be aborted by HealthMonitorCallback.");
        Assert.NotNull(health.AbortReason);
        Assert.NotNull(result.StopReason);
    }

    [Fact(Timeout = 120000)]
    public async Task Result_ExposesConfiguredMonitorCheckpointAndMixedPrecisionStatus()
    {
        const int epochs = 3;
        var (x, y) = BuildData();
        var optimizer = BuildOptimizer(epochs, 0.05);
        var model = BuildModel(optimizer);
        var monitor = new CapturingMonitor();
        var ckptDir = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(), "aidotnet_cb_ckpt_" + System.Guid.NewGuid().ToString("N").Substring(0, 8));
        System.IO.Directory.CreateDirectory(ckptDir);
        var checkpointManager = new CheckpointManager<float, Tensor<float>, Tensor<float>>(ckptDir);

        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureDataLoader(new InMemoryDataLoader<float, Tensor<float>, Tensor<float>>(x, y))
            .ConfigureTrainingMonitor(monitor)
            .ConfigureCheckpointManager(checkpointManager)
            .ConfigureMixedPrecision(new MixedPrecisionConfig { PrecisionType = MixedPrecisionType.FP16 })
            .BuildAsync();

        Assert.NotNull(result);
        // The result surfaces the CONFIGURED instances (previously null).
        Assert.Same(monitor, result.TrainingMonitor);
        Assert.Same(checkpointManager, result.CheckpointManager);
        // Mixed-precision status is populated and reports that FP16 engaged.
        Assert.False(string.IsNullOrEmpty(result.MixedPrecisionStatus));
        Assert.True(result.MixedPrecisionEngaged, $"FP16 should have engaged; status='{result.MixedPrecisionStatus}'.");
        Assert.Contains("engaged", result.MixedPrecisionStatus!, System.StringComparison.OrdinalIgnoreCase);
    }
}
