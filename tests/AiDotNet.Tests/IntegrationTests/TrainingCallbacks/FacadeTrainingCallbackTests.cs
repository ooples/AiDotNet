using System.Collections.Generic;
using System.Linq;
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

    /// <summary>A monitor whose <see cref="CheckForIssues"/> always throws, to exercise fail-closed handling.</summary>
    private sealed class ThrowingMonitor : TrainingMonitor<float>
    {
        public override List<string> CheckForIssues(string sessionId) =>
            throw new System.InvalidOperationException("monitor check failed");
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
        await Task.Yield(); // ensure the test yields so [Fact(Timeout)] can enforce on a sync-completing BuildAsync
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
        await Task.Yield();
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
        await Task.Yield();
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
    public async Task HealthMonitorCallback_AbortsOnNaNLoss()
    {
        await Task.Yield();
        const int epochs = 8;
        var (x, y) = BuildData();
        // Inject a NaN target so the very first epoch's loss becomes NaN — this exercises the NaN branch.
        y[0, 0] = float.NaN;
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
        Assert.True(result.EarlyStopTriggered, "NaN training should be aborted by HealthMonitorCallback.");
        Assert.NotNull(health.AbortReason);
        Assert.Contains("NaN", health.AbortReason!, System.StringComparison.OrdinalIgnoreCase);
        Assert.NotNull(result.StopReason);
    }

    [Fact(Timeout = 120000)]
    public async Task DivergenceGuard_StopsOptimizer_EvenWhenNoObserverVetoes()
    {
        await Task.Yield();
        const int epochs = 30;
        var (x, y) = BuildData();
        // Inject a NaN target so the fitness is non-finite from the first epoch. The callback below only COUNTS
        // epochs and never asks to stop, so the ONLY thing that can cut training short is the optimizer's
        // always-on divergence guard. Without it, all 30 epochs run (propagating NaNs).
        y[0, 0] = float.NaN;
        var optimizer = BuildOptimizer(epochs, 5.0);
        var model = BuildModel(optimizer);

        int epochCount = 0;
        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureDataLoader(new InMemoryDataLoader<float, Tensor<float>, Tensor<float>>(x, y))
            .ConfigureTrainingCallback(_ => { epochCount++; return true; }) // never vetoes — only counts
            .BuildAsync();

        Assert.NotNull(result);
        Assert.True(epochCount < epochs,
            $"divergence guard did not stop the optimizer on non-finite fitness (ran {epochCount}/{epochs} epochs)");
    }

    private static AdamOptimizer<float, Tensor<float>, Tensor<float>> BuildEarlyStopOptimizer(
        int maxIterations, double learningRate, int patience, int plateauReductions)
    {
        var options = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = learningRate,
            MaxIterations = maxIterations,
            UseAdaptiveLearningRate = false,
            UseEarlyStopping = true,
            EarlyStoppingPatience = patience,
            Tolerance = 0.0,
            MaxLearningRateReductionsOnPlateau = plateauReductions,
            PlateauLearningRateReductionFactor = 0.5,
            FitnessCalculator = new MeanSquaredErrorFitnessCalculator<float, Tensor<float>, Tensor<float>>()
        };
        return new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, options);
    }

    [Fact(Timeout = 120000)]
    public async Task ReduceLROnPlateau_defers_early_stop_and_trains_longer()
    {
        await Task.Yield();
        const int maxEpochs = 80;
        const int patience = 3;

        // Unlearnable (noise) target → the model plateaus fast at the variance floor, so early stopping fires
        // well before maxEpochs. With plateau LR reductions enabled, each reduction grants a fresh patience
        // window, so training must run STRICTLY MORE epochs before it finally gives up.
        var (x, y) = BuildData();
        var noise = new System.Random(99);
        for (int i = 0; i < N; i++) y[i, 0] = (float)(noise.NextDouble() * 2 - 1);

        int epochsBaseline = 0;
        var opt0 = BuildEarlyStopOptimizer(maxEpochs, 0.01, patience, plateauReductions: 0);
        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(BuildModel(opt0))
            .ConfigureOptimizer(opt0)
            .ConfigureDataLoader(new InMemoryDataLoader<float, Tensor<float>, Tensor<float>>(x, y))
            .ConfigureTrainingCallback(_ => { epochsBaseline++; return true; })
            .BuildAsync();

        int epochsWithReductions = 0;
        var opt3 = BuildEarlyStopOptimizer(maxEpochs, 0.01, patience, plateauReductions: 3);
        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(BuildModel(opt3))
            .ConfigureOptimizer(opt3)
            .ConfigureDataLoader(new InMemoryDataLoader<float, Tensor<float>, Tensor<float>>(x, y))
            .ConfigureTrainingCallback(_ => { epochsWithReductions++; return true; })
            .BuildAsync();

        Assert.True(epochsBaseline < maxEpochs,
            $"baseline should early-stop on the plateau, not run all {maxEpochs} (ran {epochsBaseline})");
        Assert.True(epochsWithReductions > epochsBaseline,
            $"ReduceLROnPlateau should train longer before stopping ({epochsWithReductions} vs baseline {epochsBaseline})");
    }

    // --- Learnable probe task: "most-frequent token in the window" ---------------------------
    // Order-invariant and generalizing: a strict-majority token m is planted in each window and
    // the label is m. A correct SequenceClassification transformer learns this FAR above chance
    // (1/V). Train and held-out windows are drawn from the same generator so accuracy measures
    // GENERALIZATION, not memorization.
    private const int ProbeV = 10;   // vocab / class count → chance top-1 = 0.10
    private const int ProbeCtx = 8;
    private const int ProbeMajority = 5; // > Ctx/2 → strict mode → deterministic label

    private static (Tensor<float> x, Tensor<float> y, int[] labels) BuildMajorityData(int n, int seed)
    {
        var rng = new System.Random(seed);
        var x = new Tensor<float>(new[] { n, ProbeCtx });
        var y = new Tensor<float>(new[] { n, ProbeV });
        var labels = new int[n];
        for (int i = 0; i < n; i++)
        {
            int m = rng.Next(ProbeV);
            var toks = new int[ProbeCtx];
            for (int c = 0; c < ProbeCtx; c++) toks[c] = c < ProbeMajority ? m : rng.Next(ProbeV);
            for (int c = ProbeCtx - 1; c > 0; c--) { int j = rng.Next(c + 1); (toks[c], toks[j]) = (toks[j], toks[c]); }
            for (int c = 0; c < ProbeCtx; c++) x[i, c] = toks[c];
            labels[i] = m; y[i, m] = 1f;
        }
        return (x, y, labels);
    }

    // Held-out top-1 accuracy from the model's raw Predict output. NOTE: a SequenceClassification
    // Transformer applies a Softmax output head, so Predict returns PROBABILITIES (rows sum to 1,
    // all ≥ 0) — argmax is well-defined directly on them (no logits / no extra softmax).
    private static double HeldOutAccuracy(object result, Tensor<float> vx, int[] vlabels)
    {
        var predMethod = result.GetType().GetMethod("Predict", new[] { typeof(Tensor<float>) });
        Assert.NotNull(predMethod);
        int correct = 0, cnt = 0;
        for (int i = 0; i < vlabels.Length; i++)
        {
            var xi = new Tensor<float>(new[] { 1, ProbeCtx });
            for (int c = 0; c < ProbeCtx; c++) xi[0, c] = vx[i, c];
            var pred = predMethod!.Invoke(result, new object[] { xi }) as Tensor<float>;
            if (pred is null) continue;
            var flat = pred.ToArray();
            int off = System.Math.Max(0, flat.Length - ProbeV);
            int argmax = 0; float best = float.NegativeInfinity;
            for (int v = 0; v < ProbeV; v++) if (flat[off + v] > best) { best = flat[off + v]; argmax = v; }
            if (argmax == vlabels[i]) correct++;
            cnt++;
        }
        return cnt > 0 ? (double)correct / cnt : double.NaN;
    }

    private static Transformer<float> BuildProbeTransformer(AdamOptimizer<float, Tensor<float>, Tensor<float>> opt)
    {
        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional, taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 2, numDecoderLayers: 0, numHeads: 4, modelDimension: 32,
            feedForwardDimension: 64, inputSize: ProbeCtx, outputSize: ProbeV,
            maxSequenceLength: ProbeCtx, vocabularySize: ProbeV, randomSeed: 42);
        return new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>(), optimizer: opt);
    }

    private static AdamOptimizer<float, Tensor<float>, Tensor<float>> BuildProbeOptimizer(int epochs)
        => new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null,
            new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 0.01, MaxIterations = epochs, UseAdaptiveLearningRate = false,
                UseEarlyStopping = false, Tolerance = 0.0, BatchSize = 32,
                FitnessCalculator = new MeanSquaredErrorFitnessCalculator<float, Tensor<float>, Tensor<float>>()
            });

    [Fact(Timeout = 300000)]
    public async Task Transformer_TrainsThroughFacade_LearnsHeldOutTask()
    {
        // Regression: a plain Transformer<T> (NO mixed-precision / memory config) must actually
        // LEARN through BuildAsync — measured with a CORRECT metric (held-out top-1 accuracy on a
        // genuinely-learnable task), NOT the optimizer's MSE-fitness value (which is minimised by a
        // near-uniform output and therefore does NOT indicate learning). A plain transformer takes
        // the standard OPTIMIZER path (AdamOptimizer.Optimize → tape-backprop gradient steps); this
        // asserts that path drives held-out accuracy from ~chance to well above it.
        const int epochs = 30;
        var (tx, ty, _) = BuildMajorityData(384, 1);
        var (vx, _, vlabels) = BuildMajorityData(192, 999);

        // Untrained baseline of the identical architecture: held-out accuracy ≈ chance (1/V).
        double accBefore = HeldOutAccuracy(BuildProbeTransformer(BuildProbeOptimizer(epochs)), vx, vlabels);

        var opt = BuildProbeOptimizer(epochs);
        var model = BuildProbeTransformer(opt);
        var losses = new List<double>();
        var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model).ConfigureOptimizer(opt)
            .ConfigureDataLoader(new InMemoryDataLoader<float, Tensor<float>, Tensor<float>>(tx, ty))
            .ConfigureTrainingCallback(p => { losses.Add(System.Convert.ToDouble(p.Loss)); return true; }) // NO ConfigureMixedPrecision
            .BuildAsync();

        Assert.NotNull(result);
        Assert.True(losses.Count >= 2, $"expected per-epoch loss stream, got {losses.Count}");
        Assert.All(losses, l => Assert.False(double.IsNaN(l) || double.IsInfinity(l)));

        double accAfter = HeldOutAccuracy(result!, vx, vlabels);
        double chance = 1.0 / ProbeV;
        // REAL learning: held-out accuracy clearly beats chance AND clearly beats the untrained baseline.
        Assert.True(accAfter > 0.50,
            $"transformer did not learn through the facade (optimizer path): held-out acc={accAfter:F4} (chance={chance:F4})");
        Assert.True(accAfter > accBefore + 0.20,
            $"training did not improve held-out accuracy: {accBefore:F4} -> {accAfter:F4}");
    }

    [Fact(Timeout = 300000)]
    public async Task Transformer_TrainsThroughFacade_DirectPath_WithMemoryConfig_Learns()
    {
        // Companion regression for the DIRECT in-memory path: configuring memory management routes
        // a transformer through TrainTensorNeuralNetworkRows (minibatched model.Train). That path
        // must ALSO learn the held-out task — this guards the minibatching of the direct loop.
        const int epochs = 30;
        var (tx, ty, _) = BuildMajorityData(384, 1);
        var (vx, _, vlabels) = BuildMajorityData(192, 999);

        double accBefore = HeldOutAccuracy(BuildProbeTransformer(BuildProbeOptimizer(epochs)), vx, vlabels);

        var opt = BuildProbeOptimizer(epochs);
        var model = BuildProbeTransformer(opt);
        var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>();
        builder.ConfigureMemoryManagement(AiDotNet.Training.Memory.TrainingMemoryConfig.ForTransformers());
        var result = await builder
            .ConfigureModel(model).ConfigureOptimizer(opt)
            .ConfigureDataLoader(new InMemoryDataLoader<float, Tensor<float>, Tensor<float>>(tx, ty))
            .BuildAsync();

        Assert.NotNull(result);
        double accAfter = HeldOutAccuracy(result!, vx, vlabels);
        Assert.True(accAfter > 0.50,
            $"direct-path transformer did not learn the held-out task: acc={accAfter:F4} (chance={1.0 / ProbeV:F4})");
        Assert.True(accAfter > accBefore + 0.20,
            $"direct-path training did not improve held-out accuracy: {accBefore:F4} -> {accAfter:F4}");
    }

    [Fact(Timeout = 120000)]
    public async Task HealthMonitorCallback_AbortsOnDivergingLoss()
    {
        await Task.Yield();
        // Drive OnEpochEnd directly with a deterministic loss sequence that exercises the sliding-window
        // DIVERGENCE branch (not the NaN branch): establish a stable recent best of 1.0, then feed a loss
        // more than lossRisePercent (50%) above it.
        var health = new HealthMonitorCallback<float>(lossRisePercent: 50.0, windowSize: 5);
        health.OnTrainBegin(Progress(0, 1.0f));
        Assert.True(health.OnEpochEnd(Progress(0, 1.0f)));
        Assert.True(health.OnEpochEnd(Progress(1, 1.0f)));
        Assert.True(health.OnEpochEnd(Progress(2, 1.0f)));
        // 2.0 > 1.0 + 1.0 * 0.5  =>  diverging; must return false and record a "diverging" reason.
        Assert.False(health.OnEpochEnd(Progress(3, 2.0f)));
        Assert.NotNull(health.AbortReason);
        Assert.Contains("diverging", health.AbortReason!, System.StringComparison.OrdinalIgnoreCase);
    }

    private static TrainingProgress<float> Progress(int epoch, float loss) =>
        new TrainingProgress<float>(epoch, totalEpochs: 8, loss: loss, metrics: null, elapsed: System.TimeSpan.Zero);

    [Fact(Timeout = 120000)]
    public async Task HealthMonitorCallback_FailsClosedWhenMonitorCheckThrows()
    {
        await Task.Yield();
        var monitor = new ThrowingMonitor();
        var sessionId = monitor.StartSession("failclosed");
        var health = new HealthMonitorCallback<float>(monitor: monitor, monitorSessionId: sessionId);
        health.OnTrainBegin(Progress(0, 1.0f));
        // A healthy (non-NaN, non-diverging) loss: the ONLY abort trigger is the monitor's CheckForIssues
        // throwing, which must fail CLOSED (return false) rather than be treated as healthy.
        Assert.False(health.OnEpochEnd(Progress(0, 1.0f)));
        Assert.NotNull(health.AbortReason);
        Assert.Contains("health check failed", health.AbortReason!, System.StringComparison.OrdinalIgnoreCase);
    }

    [Fact(Timeout = 120000)]
    public async Task Result_ExposesConfiguredMonitorCheckpointAndMixedPrecisionStatus()
    {
        await Task.Yield();
        const int epochs = 3;
        var (x, y) = BuildData();
        var optimizer = BuildOptimizer(epochs, 0.05);
        var model = BuildModel(optimizer);
        var monitor = new CapturingMonitor();
        var ckptDir = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(), "aidotnet_cb_ckpt_" + System.Guid.NewGuid().ToString("N").Substring(0, 8));
        System.IO.Directory.CreateDirectory(ckptDir);
        try
        {
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
        finally
        {
            // Best-effort cleanup of the unique temp checkpoint dir so runs don't leak directories.
            try { System.IO.Directory.Delete(ckptDir, recursive: true); }
            catch (System.IO.IOException) { /* still locked — leave it for the OS temp sweep */ }
        }
    }
}
