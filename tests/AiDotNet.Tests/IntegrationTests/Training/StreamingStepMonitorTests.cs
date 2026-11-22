using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.ActivationFunctions;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.FitnessCalculators;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TrainingMonitoring;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// The streaming loop reports step-level metrics to the configured training monitor every
/// <see cref="StreamingTrainingOptions{T, TInput, TOutput}.MonitorEveryNSteps"/> optimizer steps.
/// </summary>
public class StreamingStepMonitorTests
{
    private const int InDim = 4;
    private const int N = 40;
    private const int BatchSize = 4; // 10 batches (steps) per epoch

    private readonly ITestOutputHelper _output;

    public StreamingStepMonitorTests(ITestOutputHelper output) => _output = output;

    private static StreamingDataLoader<float, Tensor<float>, Tensor<float>> Loader(int samples = N)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(3);
        var x = new Tensor<float>[samples];
        var y = new Tensor<float>[samples];
        for (int i = 0; i < samples; i++)
        {
            var xi = new Tensor<float>(new[] { InDim });
            float target = 0.1f;
            for (int d = 0; d < InDim; d++)
            {
                float v = (float)(rng.NextDouble() * 2 - 1);
                xi[d] = v;
                target += 0.4f * v;
            }
            x[i] = xi;
            y[i] = new Tensor<float>(new[] { 1 }, new Vector<float>(new[] { target }));
        }

        return new StreamingDataLoader<float, Tensor<float>, Tensor<float>>(
            samples, (i, _) => Task.FromResult((x[i], y[i])), BatchSize);
    }

    private static (FeedForwardNeuralNetwork<float> Model, AdamOptimizer<float, Tensor<float>, Tensor<float>> Optimizer) TinyModel(
        int epochs, bool eager = false, bool clip = true)
    {
        var options = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 0.01,
            MaxIterations = epochs,
            UseAdaptiveLearningRate = false,
            UseEarlyStopping = false,
            Tolerance = 0.0,
            EnableGradientClipping = clip,
            FitnessCalculator = new MeanSquaredErrorFitnessCalculator<float, Tensor<float>, Tensor<float>>()
        };
        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, options);
        var layers = new List<ILayer<float>>
        {
            new InputLayer<float>(InDim),
            new DenseLayer<float>(8, activationFunction: new ReLUActivation<float>()),
            new DenseLayer<float>(1, activationFunction: new IdentityActivation<float>()),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: InDim, outputSize: 1, layers: layers);
        var networkOptions = new FeedForwardNeuralNetworkOptions { MaxGradNorm = clip ? 1.0 : 0.0 };
        var model = eager
            ? new EagerFeedForwardNetwork(arch, optimizer, networkOptions)
            : new FeedForwardNeuralNetwork<float>(arch, optimizer: optimizer, lossFunction: new MeanSquaredErrorLoss<float>(), options: networkOptions);
        return (model, optimizer);
    }

    /// <summary>A feed-forward network whose steps run on the eager tape path instead of a fused compiled plan.</summary>
    private sealed class EagerFeedForwardNetwork : FeedForwardNeuralNetwork<float>
    {
        public EagerFeedForwardNetwork(
            NeuralNetworkArchitecture<float> architecture,
            AdamOptimizer<float, Tensor<float>, Tensor<float>> optimizer,
            FeedForwardNeuralNetworkOptions options)
            : base(architecture, optimizer, new MeanSquaredErrorLoss<float>(), options)
        {
        }

        protected override bool FusedTrainingDisabledByConfiguration => true;
    }

    private static async Task<RecordingMonitor> Train(
        int epochs,
        long? maxSteps,
        int? monitorEvery,
        int? tokensPerSample = null,
        double resourceIntervalSeconds = 0,
        bool monitorGradNorm = false)
    {
        var (model, optimizer) = TinyModel(epochs);
        var monitor = new RecordingMonitor();
        var options = new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
        {
            Seed = 11,
            MaxSteps = maxSteps,
            TokensPerSample = tokensPerSample,
            MonitorResourceIntervalSeconds = resourceIntervalSeconds,
            MonitorGradNorm = monitorGradNorm,
        };
        if (monitorEvery is int every) options.MonitorEveryNSteps = every;

        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureDataLoader(Loader())
            .ConfigureStreamingTraining(options)
            .ConfigureTrainingMonitor(monitor)
            .BuildAsync();
        return monitor;
    }

    [Fact(Timeout = 180000)]
    public async Task DefaultOptions_ReportEveryTenStepsAcrossEpochs_WithGlobalStepNumbers()
    {
        await Task.Yield();
        // 3 epochs x 10 steps = 30 global steps; reports at 10, 20, 30.
        var monitor = await Train(epochs: 3, maxSteps: null, monitorEvery: null);

        Assert.Equal(new[] { 10, 20, 30 }, monitor.Steps("train_loss"));
        Assert.Equal(new[] { 10, 20, 30 }, monitor.Steps("train_loss_avg"));
        Assert.Equal(new[] { 10, 20, 30 }, monitor.Steps("learning_rate"));
        Assert.Equal(new[] { 10, 20, 30 }, monitor.Steps("step_ms"));
        Assert.Equal(new[] { 10, 20, 30 }, monitor.Steps("samples_per_second"));
        Assert.Empty(monitor.Steps("tokens_per_second"));

        Assert.All(monitor.Values("train_loss"), v => Assert.True(v >= 0 && !double.IsNaN(v)));
        Assert.All(monitor.Values("learning_rate"), v => Assert.Equal(0.01, v, 6));
        Assert.All(monitor.Values("step_ms"), v => Assert.True(v > 0));
        Assert.All(monitor.Values("samples_per_second"), v => Assert.True(v > 0));

        // Progress: every report, plus the end of every epoch; total steps known from the loader (40 / 4 * 3).
        Assert.Contains(monitor.Progress, p => p.Step == 10 && p.TotalSteps == 30 && p.Epoch == 1 && p.TotalEpochs == 3);
        Assert.Contains(monitor.Progress, p => p.Step == 30 && p.TotalSteps == 30 && p.Epoch == 3);

        // End-of-epoch loss is still reported at the epoch number.
        Assert.Equal(3, monitor.Steps("loss").Length);
    }

    [Fact(Timeout = 180000)]
    public async Task MonitorEveryNSteps_IsHonoured_AndZeroDisablesStepReports()
    {
        await Task.Yield();
        var every4 = await Train(epochs: 100, maxSteps: 13, monitorEvery: 4);
        Assert.Equal(new[] { 4, 8, 12 }, every4.Steps("train_loss"));
        Assert.Equal(new[] { 4, 8, 12 }, every4.Steps("step_ms"));
        Assert.Contains(every4.Progress, p => p.Step == 12 && p.TotalSteps == 13);

        var off = await Train(epochs: 2, maxSteps: null, monitorEvery: 0);
        Assert.Empty(off.Steps("train_loss"));
        Assert.Empty(off.Steps("step_ms"));
        Assert.Equal(2, off.Steps("loss").Length);
    }

    [Fact(Timeout = 180000)]
    public async Task TokensPerSample_AddsTokenThroughput_ProportionalToSamples()
    {
        await Task.Yield();
        var monitor = await Train(epochs: 1, maxSteps: null, monitorEvery: 5, tokensPerSample: 128);
        var samples = monitor.Values("samples_per_second");
        var tokens = monitor.Values("tokens_per_second");
        Assert.Equal(new[] { 5, 10 }, monitor.Steps("tokens_per_second"));
        for (int i = 0; i < samples.Length; i++)
            Assert.Equal(samples[i] * 128, tokens[i], 3);
    }

    [Fact(Timeout = 180000)]
    public async Task ResourceUsage_IsReportedWhenTheIntervalElapses()
    {
        await Task.Yield();
        // A tiny interval elapses between every report.
        var monitor = await Train(epochs: 2, maxSteps: null, monitorEvery: 5, resourceIntervalSeconds: 1e-6);
        Assert.NotEmpty(monitor.Resources);
        Assert.All(monitor.Resources, r =>
        {
            Assert.True(r.MemoryMb > 0);
            Assert.True(r.CpuPercent >= 0);
            Assert.Null(r.GpuPercent);
        });
    }

    [Fact(Timeout = 180000)]
    public async Task MonitorGradNorm_WithoutAnyClip_ReportsTheNormFromTheOptimizer()
    {
        await Task.Yield();
        var (model, optimizer) = TinyModel(epochs: 1, eager: true, clip: false);
        var monitor = new RecordingMonitor();
        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureDataLoader(Loader())
            .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
            {
                Seed = 11,
                MonitorEveryNSteps = 5,
                MonitorGradNorm = true,
                MonitorResourceIntervalSeconds = 0,
            })
            .ConfigureTrainingMonitor(monitor)
            .BuildAsync();

        Assert.Equal(new[] { 5, 10 }, monitor.Steps("grad_norm"));
        Assert.All(monitor.Values("grad_norm"), v => Assert.True(v > 0));
        Assert.False(optimizer.ComputeGradientNormEveryStep, "the opt-in reduction must be switched off after the run");
    }

    [Fact(Timeout = 180000)]
    public async Task WithoutAnyClipAndWithoutMonitorGradNorm_ReportsTheAnomalyGuardNorm_WithoutAnExtraReduction()
    {
        await Task.Yield();
        var (model, optimizer) = TinyModel(epochs: 1, eager: true, clip: false);
        var monitor = new RecordingMonitor();
        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureDataLoader(Loader())
            .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
            {
                Seed = 11,
                MonitorEveryNSteps = 5,
                MonitorResourceIntervalSeconds = 0,
            })
            .ConfigureTrainingMonitor(monitor)
            .BuildAsync();

        Assert.Equal(new[] { 5, 10 }, monitor.Steps("train_loss"));
        // Adam's anomaly guard already reduces the global norm on every step, so the norm is still reported, at no
        // extra cost: the opt-in reduction stays off.
        Assert.Equal(new[] { 5, 10 }, monitor.Steps("grad_norm"));
        Assert.False(optimizer.ComputeGradientNormEveryStep);
    }

    [Fact(Timeout = 180000)]
    public async Task EagerStepWithTheDefaultClip_ReportsTheNormTheClipAlreadyComputed()
    {
        await Task.Yield();
        var (model, optimizer) = TinyModel(epochs: 2, eager: true, clip: true);
        var monitor = new RecordingMonitor();
        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureDataLoader(Loader())
            .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
            {
                Seed = 11,
                MonitorResourceIntervalSeconds = 0,
            })
            .ConfigureTrainingMonitor(monitor)
            .BuildAsync();

        Assert.Equal(new[] { 10, 20 }, monitor.Steps("grad_norm"));
        Assert.All(monitor.Values("grad_norm"), v => Assert.True(v > 0 && !double.IsInfinity(v)));
        Assert.False(optimizer.ComputeGradientNormEveryStep, "the default path must not add a reduction");
    }

    /// <summary>
    /// Wall time of a small streaming run with step reporting on (every 10 steps) and off, paired and interleaved.
    /// Prints the raw medians; asserts only that both arms trained.
    /// </summary>
    [Fact(Timeout = 600000)]
    [Trait("Category", "Benchmark")]
    public async Task Overhead_MonitorOnVersusOff_PairedInterleaved()
    {
        await Task.Yield();
        const int Repeats = 15;
        const int Steps = 1000;
        var on = new List<double>();
        var off = new List<double>();

        // Warm both arms once (JIT, first-step allocations).
        await TimedRun(10);
        await TimedRun(0);

        for (int r = 0; r < Repeats; r++)
        {
            // Alternate the order so neither arm always runs first.
            if (r % 2 == 0)
            {
                on.Add(await TimedRun(10));
                off.Add(await TimedRun(0));
            }
            else
            {
                off.Add(await TimedRun(0));
                on.Add(await TimedRun(10));
            }
        }

        double Median(List<double> xs)
        {
            var sorted = xs.OrderBy(v => v).ToArray();
            return sorted.Length % 2 == 1 ? sorted[sorted.Length / 2] : (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]) / 2;
        }

        _output.WriteLine($"steps per run: {Steps}, repeats: {Repeats}");
        _output.WriteLine($"monitor on  (N=10) ms: {string.Join(", ", on.Select(v => v.ToString("F1")))}  median {Median(on):F1}");
        _output.WriteLine($"monitor off (N=0)  ms: {string.Join(", ", off.Select(v => v.ToString("F1")))}  median {Median(off):F1}");
        Assert.Equal(Repeats, on.Count);
        Assert.Equal(Repeats, off.Count);

        async Task<double> TimedRun(int every)
        {
            var (model, optimizer) = TinyModel(epochs: 1000);
            var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                .ConfigureModel(model)
                .ConfigureOptimizer(optimizer)
                .ConfigureDataLoader(Loader())
                .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
                {
                    Seed = 11,
                    MaxSteps = Steps,
                    MonitorEveryNSteps = every,
                })
                .ConfigureTrainingMonitor(new TrainingMonitor<float>());
            long start = Stopwatch.GetTimestamp();
            await builder.BuildAsync();
            return (Stopwatch.GetTimestamp() - start) * 1000.0 / Stopwatch.Frequency;
        }
    }

    private sealed class RecordingMonitor : TrainingMonitor<float>
    {
        private readonly object _gate = new();
        private readonly List<(string Name, int Step, double Value)> _metrics = new();

        public List<(int Step, int TotalSteps, int Epoch, int TotalEpochs)> Progress { get; } = new();
        public List<(double CpuPercent, double MemoryMb, double? GpuPercent, double? GpuMemoryMb)> Resources { get; } = new();

        public override void LogMetric(string sessionId, string metricName, float value, int step, DateTime? timestamp = null)
        {
            lock (_gate) _metrics.Add((metricName, step, value));
            base.LogMetric(sessionId, metricName, value, step, timestamp);
        }

        public override void UpdateProgress(string sessionId, int currentStep, int totalSteps, int currentEpoch, int totalEpochs)
        {
            lock (_gate) Progress.Add((currentStep, totalSteps, currentEpoch, totalEpochs));
            base.UpdateProgress(sessionId, currentStep, totalSteps, currentEpoch, totalEpochs);
        }

        public override void LogResourceUsage(string sessionId, double cpuUsage, double memoryUsage, double? gpuUsage = null, double? gpuMemory = null)
        {
            lock (_gate) Resources.Add((cpuUsage, memoryUsage, gpuUsage, gpuMemory));
            base.LogResourceUsage(sessionId, cpuUsage, memoryUsage, gpuUsage, gpuMemory);
        }

        public int[] Steps(string name)
        {
            lock (_gate) return _metrics.Where(m => m.Name == name).Select(m => m.Step).ToArray();
        }

        public double[] Values(string name)
        {
            lock (_gate) return _metrics.Where(m => m.Name == name).Select(m => m.Value).ToArray();
        }
    }
}
