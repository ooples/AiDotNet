using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Threading.Tasks;
using AiDotNet.ActiveLearning.Data;
using AiDotNet.CheckpointManagement;
using AiDotNet.Configuration;
using AiDotNet.CurriculumLearning.Interfaces;
using AiDotNet.Data.Loaders;
using AiDotNet.DistributedTraining;
using AiDotNet.Engines;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Regression;
using AiDotNet.Regularization;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.DistributedTraining;

/// <summary>
/// Issue #2554: <c>ConfigureDistributedTraining</c> on the streaming path for models that train through
/// <c>IGradientComputable</c> instead of the tape (classical online learners), plus a direct check that a
/// streaming loader is sharded into disjoint per-rank slices.
/// </summary>
[Collection("SequentialDistributedStreamingEnablers")]
public sealed class StreamingDistributedGradientModelTests
{
    private const int FeatureCount = 3;

    private static (Matrix<double>[] Inputs, Vector<double>[] Targets) CreateDataset(int sampleCount, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var inputs = new Matrix<double>[sampleCount];
        var targets = new Vector<double>[sampleCount];
        for (int i = 0; i < sampleCount; i++)
        {
            var row = new Matrix<double>(1, FeatureCount);
            double y = 0.25;
            for (int j = 0; j < FeatureCount; j++)
            {
                double v = rng.NextDouble() * 2.0 - 1.0;
                row[0, j] = v;
                y += (j + 1) * 0.5 * v;
            }

            inputs[i] = row;
            targets[i] = new Vector<double>(new[] { y + (rng.NextDouble() - 0.5) * 0.1 });
        }

        return (inputs, targets);
    }

    private static StreamingDataLoader<double, Matrix<double>, Vector<double>> CreateLoader(
        Matrix<double>[] inputs,
        Vector<double>[] targets,
        int batchSize)
    {
        return new StreamingDataLoader<double, Matrix<double>, Vector<double>>(
            sampleCount: inputs.Length,
            sampleReader: (idx, ct) => Task.FromResult((inputs[idx], targets[idx])),
            batchSize: batchSize,
            prefetchCount: 1,
            numWorkers: 1);
    }

    private static Vector<double> InitialParameters()
    {
        // FeatureCount coefficients followed by the intercept.
        return new Vector<double>(new[] { 0.3, -0.2, 0.1, 0.05 });
    }

    private static MultipleRegression<double> CreateModel()
    {
        var model = new MultipleRegression<double>();
        model.SetParameters(InitialParameters());
        return model;
    }

    private static InMemoryCommunicationBackend<double>[] CreateBackends(int worldSize)
    {
        string sessionId = "dist_grad_" + Guid.NewGuid().ToString("N");
        var backends = new InMemoryCommunicationBackend<double>[worldSize];
        for (int r = 0; r < worldSize; r++)
        {
            backends[r] = new InMemoryCommunicationBackend<double>(r, worldSize, sessionId);
        }

        return backends;
    }

    private static void ShutdownBackends(IEnumerable<InMemoryCommunicationBackend<double>> backends)
    {
        foreach (var backend in backends)
        {
            if (backend.IsInitialized)
            {
                backend.Shutdown();
            }
        }
    }

    private static GpuAccelerationConfig CpuOnlyConfig() => new GpuAccelerationConfig
    {
        DeviceType = GpuDeviceType.CPU,
        UsageLevel = GpuUsageLevel.AlwaysCpu,
    };

    /// <summary>
    /// Trains <paramref name="worldSize"/> in-process ranks over the same dataset and returns each rank's final
    /// parameters. Every rank starts from <paramref name="rankInitialParameters"/> when given (to prove the
    /// initial broadcast), otherwise from <see cref="InitialParameters"/>.
    /// </summary>
    private static async Task<Vector<double>[]> TrainRanksAsync(
        Matrix<double>[] inputs,
        Vector<double>[] targets,
        int worldSize,
        int[] rankBatchSizes,
        long maxSteps,
        int seed,
        Vector<double>[]? rankInitialParameters = null,
        Func<IAiModelBuilder<double, Matrix<double>, Vector<double>>, IAiModelBuilder<double, Matrix<double>, Vector<double>>>? configure = null,
        bool resume = false)
    {
        var backends = CreateBackends(worldSize);
        try
        {
            var finals = new Vector<double>[worldSize];
            var tasks = Enumerable.Range(0, worldSize).Select(rank => Task.Run(async () =>
            {
                var model = CreateModel();
                if (rankInitialParameters is not null)
                {
                    model.SetParameters(rankInitialParameters[rank].Clone());
                }

                IAiModelBuilder<double, Matrix<double>, Vector<double>> builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                    .ConfigureModel(model)
                    .ConfigureDataLoader(CreateLoader(inputs, targets, rankBatchSizes[rank]))
                    .ConfigureGpuAcceleration(CpuOnlyConfig())
                    .ConfigureDistributedTraining(backends[rank], DistributedStrategy.DDP)
                    .ConfigureStreamingTraining(new StreamingTrainingOptions<double, Matrix<double>, Vector<double>>
                    {
                        Seed = seed,
                        MaxSteps = maxSteps,
                        ResumeFromLatestCheckpoint = resume,
                    });
                await (configure is null ? builder : configure(builder)).BuildAsync();

                finals[rank] = model.GetParameters();
            })).ToArray();

            await Task.WhenAll(tasks);
            return finals;
        }
        finally
        {
            ShutdownBackends(backends);
        }
    }

    private static double MaxAbsDiff(Vector<double> a, Vector<double> b)
    {
        Assert.Equal(a.Length, b.Length);
        double max = 0.0;
        for (int i = 0; i < a.Length; i++)
        {
            Assert.False(double.IsNaN(a[i]) || double.IsInfinity(a[i]), $"Non-finite value at index {i}: {a[i]}");
            Assert.False(double.IsNaN(b[i]) || double.IsInfinity(b[i]), $"Non-finite value at index {i}: {b[i]}");
            max = Math.Max(max, Math.Abs(a[i] - b[i]));
        }

        return max;
    }

    /// <summary>Mean over every sample of the per-sample MSE gradient at <paramref name="parameters"/>.</summary>
    private static double[] FullDatasetMeanGradient(
        Matrix<double>[] inputs,
        Vector<double>[] targets,
        Vector<double> parameters)
    {
        var mean = new double[FeatureCount + 1];
        for (int i = 0; i < inputs.Length; i++)
        {
            double prediction = parameters[FeatureCount];
            for (int j = 0; j < FeatureCount; j++)
            {
                prediction += parameters[j] * inputs[i][0, j];
            }

            double error = prediction - targets[i][0];
            for (int j = 0; j < FeatureCount; j++)
            {
                mean[j] += error * inputs[i][0, j];
            }

            mean[FeatureCount] += error;
        }

        for (int j = 0; j < mean.Length; j++)
        {
            mean[j] /= inputs.Length;
        }

        return mean;
    }

    /// <summary>
    /// Gradient descent on the mean gradient of each global batch, computed without the builder or any distributed
    /// code: the batches come from an undistributed loader of <paramref name="globalBatch"/> samples, shuffled with
    /// the seed the streaming loop gives epoch <c>e</c> (<c>seed + e</c>), and the step uses the 0.01 learning rate
    /// the loop applies when no optimizer is configured, unless <paramref name="learningRateAtStep"/> gives the
    /// rate of each step. <paramref name="l2Strength"/> adds <c>strength * parameters</c> to the gradient.
    /// </summary>
    private static async Task<Vector<double>> GlobalBatchMeanGradientDescentAsync(
        Matrix<double>[] inputs,
        Vector<double>[] targets,
        int globalBatch,
        long steps,
        int seed,
        Func<long, double>? learningRateAtStep = null,
        double l2Strength = 0.0)
    {
        var loader = CreateLoader(inputs, targets, globalBatch);
        await loader.LoadAsync();

        var parameters = new double[FeatureCount + 1];
        Vector<double> initial = InitialParameters();
        for (int j = 0; j < parameters.Length; j++)
        {
            parameters[j] = initial[j];
        }

        long taken = 0;
        for (int epoch = 0; taken < steps; epoch++)
        {
            await foreach (var (batchInputs, batchTargets) in loader.GetBatchesAsync(shuffle: true, seed: seed + epoch))
            {
                double[] gradient = FullDatasetMeanGradient(batchInputs, batchTargets, new Vector<double>(parameters));
                double learningRate = learningRateAtStep is null ? 0.01 : learningRateAtStep(taken);
                for (int j = 0; j < parameters.Length; j++)
                {
                    parameters[j] -= learningRate * (gradient[j] + l2Strength * parameters[j]);
                }

                if (++taken == steps)
                {
                    break;
                }
            }
        }

        return new Vector<double>(parameters);
    }

    [Fact(Timeout = 90000)]
    public async Task GradientModel_TwoRanksFullBatch_StepAlongTheWholeDatasetMeanGradient()
    {
        var (inputs, targets) = CreateDataset(sampleCount: 16, seed: 11);

        // Two ranks of 8 samples each: one global step covers the whole dataset, so the update is independent
        // of the shuffle and must be the mean gradient over all 16 samples. Ranks that both trained on the
        // same 8 samples, or that each applied only their own gradient, cannot produce it.
        var finals = await TrainRanksAsync(inputs, targets, worldSize: 2, rankBatchSizes: new[] { 8, 8 }, maxSteps: 1, seed: 5);

        Assert.True(MaxAbsDiff(finals[0], finals[1]) == 0.0, "Both ranks must hold identical parameters after a synchronized step.");

        var start = InitialParameters();
        double[] expectedDirection = FullDatasetMeanGradient(inputs, targets, start);
        var delta = new double[expectedDirection.Length];
        double dot = 0.0, deltaNorm = 0.0, expectedNorm = 0.0;
        for (int j = 0; j < delta.Length; j++)
        {
            delta[j] = start[j] - finals[0][j];
            dot += delta[j] * expectedDirection[j];
            deltaNorm += delta[j] * delta[j];
            expectedNorm += expectedDirection[j] * expectedDirection[j];
        }

        Assert.True(deltaNorm > 0.0, "The distributed step must change the parameters.");
        double cosine = dot / Math.Sqrt(deltaNorm * expectedNorm);
        Assert.True(cosine > 1.0 - 1e-12, $"Update is not along the whole-dataset mean gradient (cosine {cosine:R}).");

        // The step length is learningRate * |mean gradient|: the same learning rate for every component.
        double learningRate = Math.Sqrt(deltaNorm / expectedNorm);
        for (int j = 0; j < delta.Length; j++)
        {
            Assert.Equal(learningRate * expectedDirection[j], delta[j], 12);
        }
    }

    // The two tests below used a one-rank backend run as their reference. A one-rank backend now trains exactly like
    // no backend (per-sample updates, see GradientModel_OneRankBackend_TrainsExactlyLikeNoBackend), so the reference
    // is computed here instead: plain gradient descent on the mean gradient of each global batch, with the batches
    // read from an undistributed loader.
    [Theory(Timeout = 90000)]
    [InlineData(2, 16, 2)]
    [InlineData(3, 16, 2)]
    [InlineData(3, 7, 2)]
    public async Task GradientModel_ManyRanks_MatchMeanGradientDescentOverEachGlobalBatch(int worldSize, int sampleCount, int perRankBatch)
    {
        var (inputs, targets) = CreateDataset(sampleCount, seed: 23);
        int globalBatch = worldSize * perRankBatch;
        const long steps = 9;

        var expected = await GlobalBatchMeanGradientDescentAsync(inputs, targets, globalBatch, steps, seed: 41);
        var multi = await TrainRanksAsync(
            inputs, targets, worldSize, Enumerable.Repeat(perRankBatch, worldSize).ToArray(), maxSteps: steps, seed: 41);

        Assert.True(MaxAbsDiff(expected, InitialParameters()) > 1e-6, "The reference must have moved the parameters.");
        for (int r = 0; r < worldSize; r++)
        {
            Assert.True(MaxAbsDiff(multi[0], multi[r]) == 0.0, $"Rank {r} diverged from rank 0.");
            double diff = MaxAbsDiff(expected, multi[r]);
            Assert.True(diff < 1e-12, $"Rank {r} differs from mean-gradient descent over the global batches by {diff:R}.");
        }
    }

    [Fact(Timeout = 90000)]
    public async Task GradientModel_UnevenRankBatchSizes_WeightBySampleCount()
    {
        var (inputs, targets) = CreateDataset(sampleCount: 18, seed: 31);

        var expected = await GlobalBatchMeanGradientDescentAsync(inputs, targets, globalBatch: 6, steps: 6, seed: 3);
        var multi = await TrainRanksAsync(
            inputs, targets, worldSize: 2, rankBatchSizes: new[] { 4, 2 }, maxSteps: 6, seed: 3);

        Assert.True(MaxAbsDiff(multi[0], multi[1]) == 0.0, "Ranks diverged.");
        double diff = MaxAbsDiff(expected, multi[0]);
        Assert.True(diff < 1e-12, $"Uneven ranks differ from mean-gradient descent over the global batches by {diff:R}.");
    }

    [Theory(Timeout = 90000)]
    [InlineData(16, 4, 9)]
    [InlineData(7, 3, 8)]
    public async Task GradientModel_OneRankBackend_TrainsExactlyLikeNoBackend(int sampleCount, int batchSize, long steps)
    {
        var (inputs, targets) = CreateDataset(sampleCount, seed: 59);

        var withoutBackend = CreateModel();
        await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(withoutBackend)
            .ConfigureDataLoader(CreateLoader(inputs, targets, batchSize))
            .ConfigureGpuAcceleration(CpuOnlyConfig())
            .ConfigureStreamingTraining(new StreamingTrainingOptions<double, Matrix<double>, Vector<double>>
            {
                Seed = 13,
                MaxSteps = steps,
            })
            .BuildAsync();

        var oneRank = await TrainRanksAsync(
            inputs, targets, worldSize: 1, rankBatchSizes: new[] { batchSize }, maxSteps: steps, seed: 13);

        Vector<double> expected = withoutBackend.GetParameters();
        Assert.True(MaxAbsDiff(expected, InitialParameters()) > 1e-6, "The run without a backend must have trained.");
        double diff = MaxAbsDiff(expected, oneRank[0]);
        Assert.True(diff == 0.0, $"A one-rank backend changed the training result by {diff:R}.");

        // Per-sample updates are not one mean-gradient step per batch, so the equality above would not hold if
        // the one-rank run had taken the synchronized step that several ranks take.
        var meanGradientRun = await GlobalBatchMeanGradientDescentAsync(inputs, targets, batchSize, steps, seed: 13);
        Assert.True(MaxAbsDiff(expected, meanGradientRun) > 1e-6,
            "This data cannot tell per-sample updates from a mean-gradient step, so the equality above proves nothing.");
    }

    [Fact(Timeout = 90000)]
    public async Task GradientModel_RanksStartingFromDifferentParameters_TrainFromRankZeros()
    {
        var (inputs, targets) = CreateDataset(sampleCount: 16, seed: 47);

        var reference = await TrainRanksAsync(
            inputs, targets, worldSize: 2, rankBatchSizes: new[] { 2, 2 }, maxSteps: 5, seed: 9);
        var mismatched = await TrainRanksAsync(
            inputs, targets, worldSize: 2, rankBatchSizes: new[] { 2, 2 }, maxSteps: 5, seed: 9,
            rankInitialParameters: new[]
            {
                InitialParameters(),
                new Vector<double>(new[] { 9.0, 9.0, 9.0, 9.0 }),
            });

        Assert.True(MaxAbsDiff(mismatched[0], mismatched[1]) == 0.0, "Rank 1 did not adopt rank 0's initial parameters.");
        Assert.True(MaxAbsDiff(reference[0], mismatched[0]) == 0.0, "Rank 0's initial parameters must be the ones trained.");
    }

    [Theory(Timeout = 90000)]
    [InlineData(2, 3, 20)]
    [InlineData(3, 2, 17)]
    public async Task StreamingLoader_PerRankSlices_AreDisjointAndCoverEveryGlobalBatch(int worldSize, int perRankBatch, int sampleCount)
    {
        var inputs = new Matrix<double>[sampleCount];
        var targets = new Vector<double>[sampleCount];
        for (int i = 0; i < sampleCount; i++)
        {
            inputs[i] = new Matrix<double>(1, 1);
            inputs[i][0, 0] = i;
            targets[i] = new Vector<double>(new[] { (double)i });
        }

        var backends = CreateBackends(worldSize);
        try
        {
            var perRankSteps = new List<int[]>[worldSize];
            var tasks = Enumerable.Range(0, worldSize).Select(rank => Task.Run(async () =>
            {
                backends[rank].Initialize();
                var loader = new DistributedStreamingDataLoader<double, Matrix<double>, Vector<double>>(
                    CreateLoader(inputs, targets, perRankBatch), backends[rank]);
                await loader.LoadAsync();

                var steps = new List<int[]>();
                await foreach (var (batchInputs, _) in loader.GetBatchesAsync(shuffle: true, seed: 77))
                {
                    steps.Add(batchInputs.Select(m => (int)m[0, 0]).ToArray());
                }

                perRankSteps[rank] = steps;
            })).ToArray();

            await Task.WhenAll(tasks);

            int stepCount = perRankSteps[0].Count;
            int globalBatch = worldSize * perRankBatch;
            Assert.Equal((sampleCount + globalBatch - 1) / globalBatch, stepCount);

            var seen = new HashSet<int>();
            for (int step = 0; step < stepCount; step++)
            {
                int stepSamples = 0;
                for (int r = 0; r < worldSize; r++)
                {
                    Assert.Equal(stepCount, perRankSteps[r].Count);
                    foreach (int id in perRankSteps[r][step])
                    {
                        Assert.True(seen.Add(id), $"Sample {id} was handed to more than one rank or step (rank {r}, step {step}).");
                        stepSamples++;
                    }
                }

                Assert.Equal(Math.Min(globalBatch, sampleCount - step * globalBatch), stepSamples);
            }

            Assert.Equal(sampleCount, seen.Count);
        }
        finally
        {
            ShutdownBackends(backends);
        }
    }

    private static StochasticGradientDescentOptimizer<double, Matrix<double>, Vector<double>> CreateScheduledSgd(
        double baseLearningRate,
        double gamma)
    {
        return new StochasticGradientDescentOptimizer<double, Matrix<double>, Vector<double>>(
            null,
            new StochasticGradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = baseLearningRate,
                MaxIterations = 50,
                UseAdaptiveLearningRate = false,
                UseEarlyStopping = false,
                LearningRateScheduler = new ExponentialLRScheduler(baseLearningRate, gamma),
                SchedulerStepMode = SchedulerStepMode.StepPerBatch,
            });
    }

    [Fact(Timeout = 90000)]
    public async Task GradientModel_ManyRanks_UseTheRateThePerBatchSchedulerHasReached()
    {
        var (inputs, targets) = CreateDataset(sampleCount: 16, seed: 71);
        const long steps = 6;
        const double baseRate = 0.04;
        const double gamma = 0.5;

        // Step k (from 0) must use baseRate * gamma^k. A scheduler that never advanced would use baseRate throughout.
        var expected = await GlobalBatchMeanGradientDescentAsync(
            inputs, targets, globalBatch: 4, steps, seed: 19, learningRateAtStep: step => baseRate * Math.Pow(gamma, step));
        var constantRate = await GlobalBatchMeanGradientDescentAsync(
            inputs, targets, globalBatch: 4, steps, seed: 19, learningRateAtStep: _ => baseRate);
        Assert.True(MaxAbsDiff(expected, constantRate) > 1e-4, "This data cannot tell a decaying rate from a constant one.");

        var multi = await TrainRanksAsync(
            inputs, targets, worldSize: 2, rankBatchSizes: new[] { 2, 2 }, maxSteps: steps, seed: 19,
            configure: builder => builder.ConfigureOptimizer(CreateScheduledSgd(baseRate, gamma)));

        Assert.True(MaxAbsDiff(multi[0], multi[1]) == 0.0, "Ranks diverged.");
        double diff = MaxAbsDiff(expected, multi[0]);
        Assert.True(diff < 1e-12, $"Two ranks differ from descent at the scheduled rates by {diff:R}.");
    }

    [Fact(Timeout = 90000)]
    public async Task GradientModel_WithoutABackend_UsesTheRateThePerBatchSchedulerHasReached()
    {
        var (inputs, targets) = CreateDataset(sampleCount: 12, seed: 73);
        const long steps = 7;
        const double baseRate = 0.04;
        const double gamma = 0.5;

        // Batches of one sample, so a batch is one update whichever way the path steps.
        var expected = await GlobalBatchMeanGradientDescentAsync(
            inputs, targets, globalBatch: 1, steps, seed: 29, learningRateAtStep: step => baseRate * Math.Pow(gamma, step));

        var model = CreateModel();
        await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(model)
            .ConfigureOptimizer(CreateScheduledSgd(baseRate, gamma))
            .ConfigureDataLoader(CreateLoader(inputs, targets, batchSize: 1))
            .ConfigureGpuAcceleration(CpuOnlyConfig())
            .ConfigureStreamingTraining(new StreamingTrainingOptions<double, Matrix<double>, Vector<double>> { Seed = 29, MaxSteps = steps })
            .BuildAsync();

        double diff = MaxAbsDiff(expected, model.GetParameters());
        Assert.True(diff < 1e-12, $"One process differs from descent at the scheduled rates by {diff:R}.");
    }

    [Fact(Timeout = 90000)]
    public async Task GradientModel_ManyRanks_AddTheRegularizationPenaltyToTheReducedGradient()
    {
        var (inputs, targets) = CreateDataset(sampleCount: 16, seed: 79);
        const long steps = 6;
        const double strength = 0.75;

        var expected = await GlobalBatchMeanGradientDescentAsync(inputs, targets, globalBatch: 4, steps, seed: 37, l2Strength: strength);
        var unregularized = await GlobalBatchMeanGradientDescentAsync(inputs, targets, globalBatch: 4, steps, seed: 37);
        Assert.True(MaxAbsDiff(expected, unregularized) > 1e-4, "This data cannot tell a penalized run from a plain one.");

        var multi = await TrainRanksAsync(
            inputs, targets, worldSize: 2, rankBatchSizes: new[] { 2, 2 }, maxSteps: steps, seed: 37,
            configure: builder => builder.ConfigureRegularization(new L2Regularization<double, Matrix<double>, Vector<double>>(
                new RegularizationOptions { Type = RegularizationType.L2, Strength = strength })));

        Assert.True(MaxAbsDiff(multi[0], multi[1]) == 0.0, "Ranks diverged.");
        double diff = MaxAbsDiff(expected, multi[0]);
        Assert.True(diff < 1e-12, $"Two ranks differ from L2-penalized descent by {diff:R}.");
    }

    [Fact(Timeout = 120000)]
    public async Task GradientModel_ManyRanks_ResumeFromACheckpointOntoTheUninterruptedRun()
    {
        var (inputs, targets) = CreateDataset(sampleCount: 20, seed: 83);
        string checkpointDir = Path.Combine(Path.GetTempPath(), "aidotnet-grad-resume-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(checkpointDir);
        try
        {
            IAiModelBuilder<double, Matrix<double>, Vector<double>> WithCheckpoints(
                IAiModelBuilder<double, Matrix<double>, Vector<double>> builder)
            {
                var manager = new CheckpointManager<double, Matrix<double>, Vector<double>>(checkpointDir);
                manager.ConfigureAutoCheckpointing(saveFrequency: 100, keepLast: 5, saveOnImprovement: false);
                return WithScheduledOptimizer(builder).ConfigureCheckpointManager(manager);
            }

            // A rate that decays every step, so resuming at the wrong step or with a restarted schedule shows.
            IAiModelBuilder<double, Matrix<double>, Vector<double>> WithScheduledOptimizer(
                IAiModelBuilder<double, Matrix<double>, Vector<double>> builder) =>
                builder.ConfigureOptimizer(CreateScheduledSgd(baseLearningRate: 0.04, gamma: 0.8));

            var uninterrupted = await TrainRanksAsync(
                inputs, targets, worldSize: 2, rankBatchSizes: new[] { 3, 3 }, maxSteps: 9, seed: 43,
                configure: WithScheduledOptimizer);

            // Four steps, which stops inside the second epoch and leaves a checkpoint; then both ranks start from
            // parameters that are nothing like the trained ones and resume to step 9.
            await TrainRanksAsync(
                inputs, targets, worldSize: 2, rankBatchSizes: new[] { 3, 3 }, maxSteps: 4, seed: 43, configure: WithCheckpoints);
            var scrambled = new Vector<double>(new[] { 5.0, -5.0, 5.0, -5.0 });
            var resumed = await TrainRanksAsync(
                inputs, targets, worldSize: 2, rankBatchSizes: new[] { 3, 3 }, maxSteps: 9, seed: 43,
                rankInitialParameters: new[] { scrambled, scrambled.Clone() }, configure: WithCheckpoints, resume: true);

            Assert.True(MaxAbsDiff(resumed[0], resumed[1]) == 0.0, "Ranks diverged after the resume.");
            double diff = MaxAbsDiff(uninterrupted[0], resumed[0]);
            Assert.True(diff == 0.0, $"The resumed run differs from the uninterrupted one by {diff:R}.");
            Assert.True(MaxAbsDiff(uninterrupted[0], InitialParameters()) > 1e-6, "The uninterrupted run must have trained.");
        }
        finally
        {
            try
            {
                Directory.Delete(checkpointDir, recursive: true);
            }
            catch (IOException ex)
            {
                System.Diagnostics.Trace.TraceWarning($"Could not delete the test checkpoint directory {checkpointDir}: {ex}");
            }
        }
    }

    [Theory(Timeout = 90000)]
    [InlineData(1)]
    [InlineData(2)]
    public async Task GradientModel_GradientAccumulation_IsRefusedInsteadOfIgnored(int worldSize)
    {
        var (inputs, targets) = CreateDataset(sampleCount: 8, seed: 89);
        var backends = CreateBackends(worldSize);
        try
        {
            await Task.WhenAll(Enumerable.Range(0, worldSize).Select(rank => Task.Run(async () =>
            {
                var model = CreateModel();
                var refused = await Assert.ThrowsAsync<NotSupportedException>(() =>
                    new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                        .ConfigureModel(model)
                        .ConfigureDataLoader(CreateLoader(inputs, targets, batchSize: 2))
                        .ConfigureGpuAcceleration(CpuOnlyConfig())
                        .ConfigureDistributedTraining(backends[rank], DistributedStrategy.DDP)
                        .ConfigureStreamingTraining(new StreamingTrainingOptions<double, Matrix<double>, Vector<double>>
                        {
                            Seed = 1,
                            MaxSteps = 1,
                            GradientAccumulationSteps = 2,
                        })
                        .BuildAsync());
                Assert.Contains("GradientAccumulationSteps", refused.Message, StringComparison.Ordinal);
                Assert.True(MaxAbsDiff(InitialParameters(), model.GetParameters()) == 0.0, "The refused build still trained the model.");
            })));
        }
        finally
        {
            ShutdownBackends(backends);
        }
    }

    /// <summary>Scores samples by their position and records what the curriculum pass handed it.</summary>
    private sealed class RecordingDifficultyEstimator : IDifficultyEstimator<double, Matrix<double>, Vector<double>>
    {
        public int EstimateDifficultiesCalls { get; private set; }

        public IDataset<double, Matrix<double>, Vector<double>>? LastDataset { get; private set; }

        public Vector<double>? ParametersOfTheModelFirstSeen { get; private set; }

        public string Name => nameof(RecordingDifficultyEstimator);

        public bool RequiresModel => false;

        public double EstimateDifficulty(
            Matrix<double> input,
            Vector<double> expectedOutput,
            IFullModel<double, Matrix<double>, Vector<double>>? model = null) => 0.5;

        public Vector<double> EstimateDifficulties(
            IDataset<double, Matrix<double>, Vector<double>> dataset,
            IFullModel<double, Matrix<double>, Vector<double>>? model = null)
        {
            EstimateDifficultiesCalls++;
            LastDataset = dataset;
            if (model is not null && ParametersOfTheModelFirstSeen is null)
            {
                ParametersOfTheModelFirstSeen = model.GetParameters().Clone();
            }

            var scores = new double[dataset.Count];
            for (int i = 0; i < scores.Length; i++)
            {
                scores[i] = (double)i / Math.Max(1, scores.Length - 1);
            }

            return new Vector<double>(scores);
        }

        public void Update(int epoch, IFullModel<double, Matrix<double>, Vector<double>> model)
        {
        }

        public void Reset()
        {
        }

        public int[] GetSortedIndices(Vector<double> difficulties)
        {
            int[] indices = Enumerable.Range(0, difficulties.Length).ToArray();
            Array.Sort(indices, (a, b) => difficulties[a].CompareTo(difficulties[b]));
            return indices;
        }
    }

    [Fact(Timeout = 120000)]
    public async Task CurriculumLearning_OnStreaming_RunsItsPassOnTheStreamTrainedModel()
    {
        var (inputs, targets) = CreateDataset(sampleCount: 16, seed: 97);
        var (curriculumInputs, curriculumTargets) = CreateDataset(sampleCount: 8, seed: 98);
        var curriculumDataset = new InMemoryDataset<double, Matrix<double>, Vector<double>>(curriculumInputs, curriculumTargets);
        var estimator = new RecordingDifficultyEstimator();

        async Task<Vector<double>> TrainAsync(bool withCurriculum)
        {
            var model = CreateModel();
            IAiModelBuilder<double, Matrix<double>, Vector<double>> builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureModel(model)
                .ConfigureDataLoader(CreateLoader(inputs, targets, batchSize: 4))
                .ConfigureGpuAcceleration(CpuOnlyConfig())
                .ConfigureStreamingTraining(new StreamingTrainingOptions<double, Matrix<double>, Vector<double>> { Seed = 7, MaxSteps = 6 });
            if (withCurriculum)
            {
                builder = builder.ConfigureCurriculumLearning(new CurriculumLearningOptions<double, Matrix<double>, Vector<double>>
                {
                    Dataset = curriculumDataset,
                    CustomDifficultyEstimator = estimator,
                    TotalEpochs = 4,
                    NumPhases = 2,
                    RandomSeed = 5,
                });
            }

            await builder.BuildAsync();
            return model.GetParameters();
        }

        Vector<double> streamOnly = await TrainAsync(withCurriculum: false);
        Vector<double> withCurriculumPass = await TrainAsync(withCurriculum: true);

        Assert.True(estimator.EstimateDifficultiesCalls >= 1, "The curriculum pass never ranked its dataset.");
        Assert.Same(curriculumDataset, estimator.LastDataset);

        // The pass started from the stream-trained model, not from the untrained one.
        Vector<double> seen = estimator.ParametersOfTheModelFirstSeen
            ?? throw new InvalidOperationException("The estimator was not given the model.");
        Assert.True(MaxAbsDiff(streamOnly, seen) == 0.0, "The curriculum pass did not start from the stream-trained model.");
        Assert.True(MaxAbsDiff(streamOnly, InitialParameters()) > 1e-6, "The stream run must have trained.");

        // And it trained the model further.
        Assert.True(MaxAbsDiff(streamOnly, withCurriculumPass) > 1e-9, "The curriculum pass left the model as the stream run did.");
    }
}
