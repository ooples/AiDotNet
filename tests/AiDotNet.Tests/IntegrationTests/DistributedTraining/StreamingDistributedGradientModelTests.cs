using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Data.Loaders;
using AiDotNet.DistributedTraining;
using AiDotNet.Engines;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
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
        Vector<double>[]? rankInitialParameters = null)
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

                await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                    .ConfigureModel(model)
                    .ConfigureDataLoader(CreateLoader(inputs, targets, rankBatchSizes[rank]))
                    .ConfigureGpuAcceleration(CpuOnlyConfig())
                    .ConfigureDistributedTraining(backends[rank], DistributedStrategy.DDP)
                    .ConfigureStreamingTraining(new StreamingTrainingOptions<double, Matrix<double>, Vector<double>>
                    {
                        Seed = seed,
                        MaxSteps = maxSteps,
                    })
                    .BuildAsync();

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

    [Theory(Timeout = 90000)]
    [InlineData(2, 16, 2)]
    [InlineData(3, 16, 2)]
    [InlineData(3, 7, 2)]
    public async Task GradientModel_ManyRanks_MatchOneRankAtTheSameGlobalBatch(int worldSize, int sampleCount, int perRankBatch)
    {
        var (inputs, targets) = CreateDataset(sampleCount, seed: 23);
        int globalBatch = worldSize * perRankBatch;
        const long steps = 9;

        var single = await TrainRanksAsync(
            inputs, targets, worldSize: 1, rankBatchSizes: new[] { globalBatch }, maxSteps: steps, seed: 41);
        var multi = await TrainRanksAsync(
            inputs, targets, worldSize, Enumerable.Repeat(perRankBatch, worldSize).ToArray(), maxSteps: steps, seed: 41);

        Assert.True(MaxAbsDiff(single[0], InitialParameters()) > 1e-6, "The reference run must have trained.");
        for (int r = 0; r < worldSize; r++)
        {
            Assert.True(MaxAbsDiff(multi[0], multi[r]) == 0.0, $"Rank {r} diverged from rank 0.");
            double diff = MaxAbsDiff(single[0], multi[r]);
            Assert.True(diff < 1e-12, $"Rank {r} differs from the one-rank run at the same global batch by {diff:R}.");
        }
    }

    [Fact(Timeout = 90000)]
    public async Task GradientModel_UnevenRankBatchSizes_WeightBySampleCount()
    {
        var (inputs, targets) = CreateDataset(sampleCount: 18, seed: 31);

        var single = await TrainRanksAsync(
            inputs, targets, worldSize: 1, rankBatchSizes: new[] { 6 }, maxSteps: 6, seed: 3);
        var multi = await TrainRanksAsync(
            inputs, targets, worldSize: 2, rankBatchSizes: new[] { 4, 2 }, maxSteps: 6, seed: 3);

        Assert.True(MaxAbsDiff(multi[0], multi[1]) == 0.0, "Ranks diverged.");
        double diff = MaxAbsDiff(single[0], multi[0]);
        Assert.True(diff < 1e-12, $"Uneven ranks differ from one rank at the same global batch by {diff:R}.");
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
}
