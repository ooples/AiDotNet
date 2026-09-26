using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.ActivationFunctions;
using AiDotNet.CheckpointManagement;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.FitnessCalculators;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TrainingMonitoring;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Step-level streaming training through the facade (<c>ConfigureStreamingTraining</c>): the resumed run must be
/// indistinguishable from an uninterrupted one, a WSD run must be extendable, held-out validation must be logged,
/// and a configured learning-rate scheduler must actually drive training.
/// </summary>
public class StreamingStepTrainingTests : IDisposable
{
    private const int InDim = 4;
    private const int N = 40;
    private const int BatchSize = 4;          // 10 batches per epoch
    private readonly string _root = Path.Combine(Path.GetTempPath(), "aidotnet-streaming-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    private static (Tensor<float>[] x, Tensor<float>[] y) Data(int seed)
    {
        var rng = new Random(seed);
        var x = new Tensor<float>[N];
        var y = new Tensor<float>[N];
        for (int i = 0; i < N; i++)
        {
            var xi = new Tensor<float>(new[] { InDim });
            float target = 0.05f;
            for (int d = 0; d < InDim; d++)
            {
                float v = (float)(rng.NextDouble() * 2 - 1);
                xi[d] = v;
                target += (d % 2 == 0 ? 0.5f : -0.3f) * v;
            }
            x[i] = xi;
            y[i] = new Tensor<float>(new[] { 1 }, new Vector<float>(new[] { target }));
        }
        return (x, y);
    }

    private static StreamingDataLoader<float, Tensor<float>, Tensor<float>> Loader(int seed)
    {
        var (x, y) = Data(seed);
        return new StreamingDataLoader<float, Tensor<float>, Tensor<float>>(
            N, (i, _) => Task.FromResult((x[i], y[i])), BatchSize);
    }

    private static AdamOptimizer<float, Tensor<float>, Tensor<float>> Optimizer(int epochs, ILearningRateScheduler? scheduler = null)
    {
        var options = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 0.02,
            MaxIterations = epochs,
            UseAdaptiveLearningRate = false,
            UseEarlyStopping = false,
            Tolerance = 0.0,
            LearningRateScheduler = scheduler,
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
            FitnessCalculator = new MeanSquaredErrorFitnessCalculator<float, Tensor<float>, Tensor<float>>()
        };
        return new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, options);
    }

    private static FeedForwardNeuralNetwork<float> Model(AdamOptimizer<float, Tensor<float>, Tensor<float>> optimizer)
    {
        var layers = new List<ILayer<float>>
        {
            new InputLayer<float>(InDim),
            new DenseLayer<float>(8, activationFunction: new ReLUActivation<float>()),
            new DenseLayer<float>(1, activationFunction: new IdentityActivation<float>()),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: InDim, outputSize: 1, layers: layers);
        return new FeedForwardNeuralNetwork<float>(arch, optimizer: optimizer, lossFunction: new MeanSquaredErrorLoss<float>());
    }

    /// <summary>A fixed initial weight vector shared by every run, so runs differ only in how they were trained.</summary>
    private static Vector<float> InitialWeights()
    {
        var probe = Model(Optimizer(1));
        var rng = new Random(7);
        var w = new float[probe.GetParameters().Length];
        for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() * 0.6 - 0.3);
        return new Vector<float>(w);
    }

    private CheckpointManager<float, Tensor<float>, Tensor<float>> Checkpoints(string name, int saveEvery)
    {
        var manager = new CheckpointManager<float, Tensor<float>, Tensor<float>>(Path.Combine(_root, name));
        manager.ConfigureAutoCheckpointing(saveFrequency: saveEvery, keepLast: 3, saveOnImprovement: false);
        return manager;
    }

    private static async Task<(FeedForwardNeuralNetwork<float> Model, AdamOptimizer<float, Tensor<float>, Tensor<float>> Opt)> Train(
        long maxSteps,
        bool resume,
        CheckpointManager<float, Tensor<float>, Tensor<float>>? checkpoints,
        Vector<float> init,
        ILearningRateScheduler? scheduler = null,
        bool useConfiguredSchedulerOnResume = false,
        int seed = 1234,
        ITrainingMonitor<float>? monitor = null,
        IStreamingDataLoader<float, Tensor<float>, Tensor<float>>? validation = null,
        int validateEvery = 0,
        IDataTransformer<float, Tensor<float>, Tensor<float>>? featureTransform = null)
    {
        var optimizer = Optimizer(epochs: 100, scheduler);
        var model = Model(optimizer);
        model.SetParameters(init.Clone());
        var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureDataLoader(Loader(1))
            .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
            {
                Seed = seed,
                MaxSteps = maxSteps,
                ResumeFromLatestCheckpoint = resume,
                UseConfiguredSchedulerOnResume = useConfiguredSchedulerOnResume,
                ValidationLoader = validation,
                ValidateEveryNSteps = validateEvery,
            });
        if (checkpoints is not null) builder.ConfigureCheckpointManager(checkpoints);
        if (monitor is not null) builder.ConfigureTrainingMonitor(monitor);
        if (featureTransform is not null) builder.ConfigurePreprocessing(featureTransform);
        await builder.BuildAsync();
        return (model, optimizer);
    }

    private static double MaxAbsDiff(Vector<float> a, Vector<float> b)
    {
        Assert.Equal(a.Length, b.Length);
        double max = 0;
        for (int i = 0; i < a.Length; i++) max = Math.Max(max, Math.Abs(a[i] - b[i]));
        return max;
    }

    [Fact(Timeout = 180000)]
    public async Task StoppedAndResumedRun_MatchesUninterruptedRun()
    {
        await Task.Yield();
        var init = InitialWeights();
        // 25 steps crosses an epoch boundary (10 batches/epoch) on both sides of the stop at step 13.
        var straight = await Train(25, resume: false, Checkpoints("straight", saveEvery: 1000), init,
            new WarmupStableDecayScheduler(0.02, warmupSteps: 5, decayStartStep: 15, decaySteps: 10));

        var ckpt = Checkpoints("resumed", saveEvery: 4);
        var first = await Train(13, resume: true, ckpt, init,
            new WarmupStableDecayScheduler(0.02, warmupSteps: 5, decayStartStep: 15, decaySteps: 10));
        // The comparison is only meaningful if training moved the weights between the stop and the end.
        Assert.True(MaxAbsDiff(first.Model.GetParameters(), straight.Model.GetParameters()) > 1e-3,
            "the stop point must differ from the end point, or the equality below proves nothing");

        // A NEW model object with a DIFFERENT starting point: everything must come from the checkpoint.
        var scrambled = new Vector<float>(Enumerable.Repeat(0.123f, init.Length).ToArray());
        var resumed = await Train(25, resume: true, Checkpoints("resumed", saveEvery: 4), scrambled,
            new WarmupStableDecayScheduler(0.02, warmupSteps: 5, decayStartStep: 15, decaySteps: 10));

        Assert.True(MaxAbsDiff(resumed.Model.GetParameters(), straight.Model.GetParameters()) < 1e-6,
            "resumed run diverged from the uninterrupted run");
        Assert.Equal(straight.Opt.GetCurrentLearningRate(), resumed.Opt.GetCurrentLearningRate(), 12);
    }

    /// <summary>
    /// Subtracts the mean of whichever batch it was fitted on. Fitting on a different batch gives a different
    /// shift, so a resumed run that fits its (fresh, unfitted) pipeline on another batch trains on differently
    /// scaled data and diverges from the run it resumes.
    /// </summary>
    private sealed class FirstFitMeanShift : IDataTransformer<float, Tensor<float>, Tensor<float>>
    {
        private float _mean;

        public bool IsFitted { get; private set; }
        public bool SupportsInverseTransform => true;
        public int[]? ColumnIndices => null;

        public void Fit(Tensor<float> data)
        {
            double sum = 0;
            for (int i = 0; i < data.Length; i++) sum += data[i];
            _mean = (float)(sum / Math.Max(1, data.Length));
            IsFitted = true;
        }

        public Tensor<float> Transform(Tensor<float> data) => Shift(data, -_mean);
        public Tensor<float> FitTransform(Tensor<float> data) { Fit(data); return Transform(data); }
        public Tensor<float> InverseTransform(Tensor<float> data) => Shift(data, _mean);
        public string[] GetFeatureNamesOut(string[]? inputFeatureNames = null) => inputFeatureNames ?? [];

        private static Tensor<float> Shift(Tensor<float> data, float by)
        {
            var result = new Tensor<float>(data.Shape.ToArray());
            for (int i = 0; i < data.Length; i++) result[i] = data[i] + by;
            return result;
        }
    }

    [Fact(Timeout = 180000)]
    public async Task StoppedAndResumedRun_WithFittedPreprocessing_MatchesUninterruptedRun()
    {
        await Task.Yield();
        var init = InitialWeights();
        WarmupStableDecayScheduler Wsd() => new(0.02, warmupSteps: 5, decayStartStep: 15, decaySteps: 10);
        var baselineTransform = new FirstFitMeanShift();
        var straight = await Train(25, resume: false, Checkpoints("pp-straight", saveEvery: 1000), init, Wsd(),
            featureTransform: baselineTransform);

        // Positive controls: the pipeline must actually take part in training, or the resume comparison below
        // would pass with preprocessing ignored (both runs would train on the same unprocessed data).
        Assert.True(baselineTransform.IsFitted, "the configured preprocessing was never fitted");
        var unprocessed = await Train(25, resume: false, Checkpoints("pp-none", saveEvery: 1000), init, Wsd());
        Assert.True(MaxAbsDiff(straight.Model.GetParameters(), unprocessed.Model.GetParameters()) > 1e-3,
            "preprocessing did not change the trained parameters, so the resume comparison proves nothing");

        await Train(13, resume: true, Checkpoints("pp-resumed", saveEvery: 4), init, Wsd(),
            featureTransform: new FirstFitMeanShift());
        var scrambled = new Vector<float>(Enumerable.Repeat(0.123f, init.Length).ToArray());
        var resumed = await Train(25, resume: true, Checkpoints("pp-resumed", saveEvery: 4), scrambled, Wsd(),
            featureTransform: new FirstFitMeanShift());

        Assert.True(MaxAbsDiff(resumed.Model.GetParameters(), straight.Model.GetParameters()) < 1e-6,
            "a resumed run fitted its preprocessing on a different batch than the uninterrupted run");
    }

    [Fact(Timeout = 60000)]
    public async Task StreamingTrainingOptions_OnANonStreamingBuild_AreRefused()
    {
        await Task.Yield();
        var optimizer = Optimizer(epochs: 1);
        var x = new Tensor<float>(new[] { N, InDim });
        var y = new Tensor<float>(new[] { N, 1 });
        var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(Model(optimizer))
            .ConfigureOptimizer(optimizer)
            .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>> { Seed = 1 });

        Assert.Throws<NotSupportedException>(() => builder.Build(x, y));
    }

    [Fact(Timeout = 120000)]
    public async Task ResumeWithDifferentSeed_IsRefused()
    {
        await Task.Yield();
        var init = InitialWeights();
        await Train(6, resume: true, Checkpoints("seed", saveEvery: 3), init, seed: 1);
        await Assert.ThrowsAsync<InvalidOperationException>(
            () => Train(12, resume: true, Checkpoints("seed", saveEvery: 3), init, seed: 2));
    }

    [Fact(Timeout = 180000)]
    public async Task StableWsdRun_ExtendedWithConfiguredDecay_MatchesRunPlannedThatWay()
    {
        await Task.Yield();
        var init = InitialWeights();
        var planned = await Train(24, resume: false, null, init,
            new WarmupStableDecayScheduler(0.02, warmupSteps: 4, decayStartStep: 16, decaySteps: 8));

        // Stable-only run (decay never scheduled), stopped at 16 ...
        await Train(16, resume: true, Checkpoints("extend", saveEvery: 1000), init,
            new WarmupStableDecayScheduler(0.02, warmupSteps: 4, decayStartStep: int.MaxValue, decaySteps: 8));
        // ... then extended with the decay decided afterwards.
        var extended = await Train(24, resume: true, Checkpoints("extend", saveEvery: 1000), init,
            new WarmupStableDecayScheduler(0.02, warmupSteps: 4, decayStartStep: 16, decaySteps: 8),
            useConfiguredSchedulerOnResume: true);

        Assert.True(MaxAbsDiff(extended.Model.GetParameters(), planned.Model.GetParameters()) < 1e-6);
        Assert.Equal(0.0, extended.Opt.GetCurrentLearningRate(), 12);
    }

    [Fact(Timeout = 120000)]
    public async Task StepBudgetStop_AlwaysLeavesACheckpointAtTheStopStep()
    {
        await Task.Yield();
        var ckpt = Checkpoints("budget", saveEvery: 1000);
        await Train(7, resume: false, ckpt, InitialWeights());
        var latest = ckpt.LoadLatestCheckpoint();
        Assert.NotNull(latest);
        Assert.Equal(7, latest!.Step);
    }

    [Fact(Timeout = 120000)]
    public async Task Validation_IsLoggedEveryNStepsAndAtTheEnd()
    {
        await Task.Yield();
        var monitor = new CapturingMonitor();
        await Train(10, resume: false, null, InitialWeights(), monitor: monitor, validation: Loader(99), validateEvery: 4);
        var history = monitor.GetMetricHistory(monitor.SessionId, "validation_loss");
        // steps 4, 8, then the final evaluation at step 10
        Assert.Equal(new[] { 4, 8, 10 }, history.Select(entry => entry.Step).ToArray());
        foreach (var (step, value, _) in history)
        {
            Assert.False(float.IsNaN(value) || float.IsInfinity(value), $"validation loss at step {step} is not finite: {value}");
            Assert.True(value >= 0f, $"validation loss at step {step} is negative: {value}");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureLearningRateScheduler_DrivesTheOptimizer_OnTheStreamingPath()
    {
        await Task.Yield();
        var scheduler = new WarmupStableDecayScheduler(0.02, warmupSteps: 0, decayStartStep: 0, decaySteps: 10);
        var optimizer = Optimizer(epochs: 100);
        var model = Model(optimizer);
        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureDataLoader(Loader(1))
            .ConfigureLearningRateScheduler(scheduler)
            .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>> { Seed = 1, MaxSteps = 10 })
            .BuildAsync();

        Assert.Same(scheduler, optimizer.LearningRateScheduler);
        Assert.Equal(10, scheduler.CurrentStep);
        Assert.Equal(0.0, optimizer.GetCurrentLearningRate(), 12);
    }

    [Fact(Timeout = 120000)]
    public async Task ReloadedCheckpoint_KeepsItsIdAndCreationTime_AcrossManagerInstances()
    {
        await Task.Yield();
        var dir = Path.Combine(_root, "ids");
        var first = new CheckpointManager<float, Tensor<float>, Tensor<float>>(dir);
        var optimizer = Optimizer(1);
        var id = first.SaveCheckpoint(Model(optimizer), optimizer, 0, 3, new Dictionary<string, float> { ["loss"] = 1f });
        var saved = first.LoadCheckpoint(id);

        var reopened = new CheckpointManager<float, Tensor<float>, Tensor<float>>(dir);
        var latest = reopened.LoadLatestCheckpoint();
        Assert.NotNull(latest);
        Assert.Equal(id, saved.CheckpointId);
        Assert.Equal(id, latest!.CheckpointId);
        Assert.Equal(saved.CreatedAt, latest.CreatedAt);
    }

    [Fact]
    public void Wsd_WarmupStableDecay_Values()
    {
        var s = new WarmupStableDecayScheduler(1.0, warmupSteps: 4, decayStartStep: 10, decaySteps: 4);
        Assert.Equal(0.25, s.GetLearningRateAtStep(1), 12);
        Assert.Equal(1.0, s.GetLearningRateAtStep(4), 12);
        Assert.Equal(1.0, s.GetLearningRateAtStep(9), 12);
        Assert.Equal(0.5, s.GetLearningRateAtStep(12), 12);
        Assert.Equal(0.0, s.GetLearningRateAtStep(14), 12);
        Assert.Equal(0.0, s.GetLearningRateAtStep(100), 12);
        var sqrt = new WarmupStableDecayScheduler(1.0, 0, 0, 4, WarmupStableDecayScheduler.DecayShape.OneMinusSqrt);
        Assert.Equal(0.5, sqrt.GetLearningRateAtStep(1), 12);
    }

    private sealed class CapturingMonitor : TrainingMonitor<float>
    {
        public string SessionId { get; private set; } = string.Empty;

        public override string StartSession(string sessionName, Dictionary<string, object>? metadata = null)
        {
            SessionId = base.StartSession(sessionName, metadata);
            return SessionId;
        }
    }
}
