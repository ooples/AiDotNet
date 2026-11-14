using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.CheckpointManagement;
using AiDotNet.Data.Loaders;
using AiDotNet.DistributedTraining;
using AiDotNet.Engines;
using AiDotNet.Enums;
using AiDotNet.FitnessCalculators;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LossFunctions;
using AiDotNet.MixedPrecision;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.DistributedTraining;

/// <summary>
/// Acceptance and integration tests for distributed streaming training enablers (Issue #2554):
/// bucketed gradient reduction overlap, uneven micro-batch token-weighted synchronization,
/// empty tail-batch rank handling, all four data/state-parallel strategies (<c>DDP</c>, <c>ZeRO1</c>,
/// <c>ZeRO2</c>, <c>FSDP</c>), every tape step path (eager, accumulated, compiled/fused, streaming
/// 8-bit optimizer, and <see cref="TapeTrainingStepper{T}"/>), mixed precision with coordinated
/// overflow skipping, wall-clock auto-checkpointing, and cross-world-size checkpoint resume.
/// </summary>
[Collection("SequentialDistributedStreamingEnablers")]
public sealed class StreamingDistributedEnablersTests : IDisposable
{
    private const int InDim = 8;
    private const int HiddenDim = 16;
    private const int OutDim = 4;

    private readonly string _tempRoot = Path.Combine(
        Path.GetTempPath(),
        "aidotnet-dist-enablers-" + Guid.NewGuid().ToString("N"));

    public StreamingDistributedEnablersTests()
    {
        Directory.CreateDirectory(_tempRoot);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_tempRoot))
            {
                Directory.Delete(_tempRoot, recursive: true);
            }
        }
        catch (IOException)
        {
        }
        catch (UnauthorizedAccessException)
        {
        }
    }

    #region Deterministic Fixtures & Helpers

    private static (Tensor<float>[] Inputs, Tensor<float>[] Targets) CreateRegressionDataset(int sampleCount, int seed = 101)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var inputs = new Tensor<float>[sampleCount];
        var targets = new Tensor<float>[sampleCount];

        for (int i = 0; i < sampleCount; i++)
        {
            var xArr = new float[InDim];
            for (int d = 0; d < InDim; d++)
            {
                xArr[d] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }

            var yArr = new float[OutDim];
            for (int o = 0; o < OutDim; o++)
            {
                float acc = 0.1f * (o + 1);
                for (int d = 0; d < InDim; d++)
                {
                    float w = ((d + o) % 2 == 0) ? 0.65f : -0.45f;
                    acc += w * xArr[d];
                }

                yArr[o] = acc;
            }

            inputs[i] = new Tensor<float>(new[] { InDim }, new Vector<float>(xArr));
            targets[i] = new Tensor<float>(new[] { OutDim }, new Vector<float>(yArr));
        }

        return (inputs, targets);
    }

    private static StreamingDataLoader<float, Tensor<float>, Tensor<float>> CreateStreamingLoader(
        Tensor<float>[] inputs,
        Tensor<float>[] targets,
        int batchSize,
        int sampleDelayMs = 0)
    {
        return new StreamingDataLoader<float, Tensor<float>, Tensor<float>>(
            sampleCount: inputs.Length,
            sampleReader: async (idx, ct) =>
            {
                if (sampleDelayMs > 0)
                {
                    await Task.Delay(sampleDelayMs, ct).ConfigureAwait(false);
                }

                return (inputs[idx], targets[idx]);
            },
            batchSize: batchSize,
            prefetchCount: 1,
            numWorkers: 1);
    }

    private static AdamOptimizer<float, Tensor<float>, Tensor<float>> CreateAdamOptimizer(
        double lr = 0.01,
        int epochs = 50,
        bool enableClipping = true,
        double maxGradNorm = 1.0,
        ILearningRateScheduler? scheduler = null)
    {
        var opts = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = lr,
            MaxIterations = epochs,
            UseAdaptiveLearningRate = false,
            UseEarlyStopping = false,
            Tolerance = 0.0,
            EnableGradientClipping = enableClipping,
            MaxGradientNorm = maxGradNorm,
            LearningRateScheduler = scheduler,
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
            FitnessCalculator = new MeanSquaredErrorFitnessCalculator<float, Tensor<float>, Tensor<float>>(),
        };
        return new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, opts);
    }

    private static FeedForwardNeuralNetwork<float> CreateMlp(
        IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>>? optimizer = null)
    {
        var layers = new List<ILayer<float>>
        {
            new InputLayer<float>(InDim),
            new DenseLayer<float>(outputSize: HiddenDim, activationFunction: new ReLUActivation<float>()),
            new DenseLayer<float>(outputSize: HiddenDim, activationFunction: new ReLUActivation<float>()),
            new DenseLayer<float>(outputSize: OutDim, activationFunction: new IdentityActivation<float>()),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: InDim,
            outputSize: OutDim,
            layers: layers);
        var net = new FeedForwardNeuralNetwork<float>(
            arch,
            optimizer: optimizer,
            lossFunction: new MeanSquaredErrorLoss<float>());
        _ = net.Predict(new Tensor<float>(new[] { 1, InDim }));
        return net;
    }

    private static Vector<float> CreateInitialWeights(int seed = 77)
    {
        var probe = CreateMlp(CreateAdamOptimizer());
        int len = probe.GetParameters().Length;
        var rng = RandomHelper.CreateSeededRandom(seed);
        var arr = new float[len];
        for (int i = 0; i < len; i++)
        {
            arr[i] = (float)(rng.NextDouble() * 0.8 - 0.4);
        }

        return new Vector<float>(arr);
    }

    private static Transformer<float> CreateLazyCanaryTransformer(int seed)
    {
        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 1,
            numDecoderLayers: 0,
            numHeads: 2,
            modelDimension: 16,
            feedForwardDimension: 32,
            inputSize: 4,
            outputSize: 8,
            maxSequenceLength: 4,
            vocabularySize: 8,
            warmupSteps: 5,
            randomSeed: seed);
        return new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());
    }

    private static (Tensor<float>[] Inputs, Tensor<float>[] Targets) CreateSequenceClassificationDataset(int sampleCount, int seed = 303)
    {
        const int seqLen = 4;
        const int vocab = 8;
        var rng = RandomHelper.CreateSeededRandom(seed);
        var inputs = new Tensor<float>[sampleCount];
        var targets = new Tensor<float>[sampleCount];

        for (int i = 0; i < sampleCount; i++)
        {
            int cls = i % vocab;
            var tok = new float[seqLen];
            for (int t = 0; t < seqLen; t++)
            {
                tok[t] = (float)((cls + t + rng.Next(0, 2)) % vocab);
            }

            var oneHot = new float[vocab];
            oneHot[cls] = 1.0f;

            inputs[i] = new Tensor<float>(new[] { seqLen }, new Vector<float>(tok));
            targets[i] = new Tensor<float>(new[] { vocab }, new Vector<float>(oneHot));
        }

        return (inputs, targets);
    }

    private static Tensor<float> StackRows(Tensor<float>[] rows, int start, int count)
    {
        int width = rows[0].Length;
        var flat = new float[count * width];
        for (int i = 0; i < count; i++)
        {
            rows[start + i].Data.Span.CopyTo(flat.AsSpan(i * width, width));
        }

        return new Tensor<float>(new[] { count, width }, new Vector<float>(flat));
    }

    private static double MaxAbsDiff(Vector<float> a, Vector<float> b)
    {
        Assert.Equal(a.Length, b.Length);
        double max = 0.0;
        for (int i = 0; i < a.Length; i++)
        {
            double diff = Math.Abs((double)a[i] - b[i]);
            if (diff > max)
            {
                max = diff;
            }
        }

        return max;
    }

    private static double ComputeMse(FeedForwardNeuralNetwork<float> model, Tensor<float>[] inputs, Tensor<float>[] targets)
    {
        var x = StackRows(inputs, 0, inputs.Length);
        var y = StackRows(targets, 0, targets.Length);
        var pred = model.Predict(x);
        Assert.Equal(y.Length, pred.Length);
        double sumSq = 0.0;
        var pSpan = pred.AsSpan();
        var ySpan = y.AsSpan();
        for (int i = 0; i < pSpan.Length; i++)
        {
            double d = (double)pSpan[i] - ySpan[i];
            sumSq += d * d;
        }

        return sumSq / pSpan.Length;
    }

    private static ICommunicationBackend<float>[] CreateBackends(int worldSize, bool useSharedMemory)
    {
        string sessionId = "dist_" + Guid.NewGuid().ToString("N");
        var backends = new ICommunicationBackend<float>[worldSize];
        for (int r = 0; r < worldSize; r++)
        {
            backends[r] = useSharedMemory
                ? new SharedMemoryCommunicationBackend<float>(r, worldSize, sessionId, slotCapacityBytes: 2L * 1024L * 1024L, timeoutMilliseconds: 30_000)
                : new InMemoryCommunicationBackend<float>(r, worldSize, sessionId);
        }

        return backends;
    }

    private static void ShutdownBackends(IEnumerable<ICommunicationBackend<float>> backends)
    {
        foreach (var backend in backends)
        {
            try
            {
                if (backend.IsInitialized)
                {
                    backend.Shutdown();
                }

                if (backend is IDisposable disp)
                {
                    disp.Dispose();
                }
            }
            catch (Exception)
            {
            }
        }
    }

    private static IReadOnlyList<Tensor<float>> GetTrainableParameters(NeuralNetworkBase<float> model) =>
        model.Layers.OfType<ITrainableLayer<float>>().SelectMany(l => l.GetTrainableParameters()).ToList();

    private static GpuAccelerationConfig CpuOnlyConfig() => new GpuAccelerationConfig
    {
        DeviceType = GpuDeviceType.CPU,
        UsageLevel = GpuUsageLevel.AlwaysCpu,
    };

    #endregion

    #region (a) Exact Gradient & Trajectory Parity

    [Fact(Timeout = 60000)]
    public async Task ExactGradientParity_MultiBucketAsyncOverlap_MatchesSingleRankGradients()
    {
        var init = CreateInitialWeights(seed: 51);
        var (inputs, targets) = CreateRegressionDataset(sampleCount: 8, seed: 901);
        var lossFn = new MeanSquaredErrorLoss<float>();

        // Single-rank B=8 reference gradients
        var singleModel = CreateMlp(CreateAdamOptimizer());
        singleModel.SetParameters(init.Clone());
        var singleParams = GetTrainableParameters(singleModel);
        var fullX = StackRows(inputs, 0, 8);
        var fullY = StackRows(targets, 0, 8);
        Dictionary<Tensor<float>, Tensor<float>> singleGrads;
        using (var tape = new GradientTape<float>())
        {
            var pred = singleModel.ForwardForTraining(fullX);
            var loss = lossFn.ComputeTapeLoss(pred, fullY);
            singleGrads = tape.ComputeGradients(loss, sources: singleParams);
        }

        // Two-rank B0=4, B1=4 with tiny 64-byte buckets to force multi-bucket async overlap
        var backends = CreateBackends(worldSize: 2, useSharedMemory: true);
        try
        {
            var rankParams = new IReadOnlyList<Tensor<float>>[2];
            var rankGrads = new Dictionary<Tensor<float>, Tensor<float>>[2];
            var rankBucketCounts = new int[2];

            var tasks = Enumerable.Range(0, 2).Select(rank => Task.Run(() =>
            {
                backends[rank].Initialize();
                var model = CreateMlp(CreateAdamOptimizer());
                model.SetParameters(init.Clone());
                var prms = GetTrainableParameters(model);
                rankParams[rank] = prms;

                using var reducer = new GradientBucketReducer<float>(
                    backends[rank],
                    prms,
                    bucketSizeBytes: 64,
                    overlapCommunication: true,
                    useReduceScatter: false);
                rankBucketCounts[rank] = reducer.BucketCount;

                var localX = StackRows(inputs, rank * 4, 4);
                var localY = StackRows(targets, rank * 4, 4);
                var grads = new Dictionary<Tensor<float>, Tensor<float>>();

                reducer.BeginStep(rankWeight: 0.5);
                using (var tape = new GradientTape<float>())
                {
                    var pred = model.ForwardForTraining(localX);
                    var loss = lossFn.ComputeTapeLoss(pred, localY);
                    tape.ComputeGradientsStreaming(loss, prms, (p, g) =>
                    {
                        grads[p] = g;
                        reducer.NotifyGradientReady(p, g);
                    });
                }

                reducer.FlushAndSynchronize(grads, out _);
                rankGrads[rank] = grads;
            })).ToArray();

            await Task.WhenAll(tasks);

            Assert.True(rankBucketCounts[0] >= 3, $"Expected >=3 buckets with 64B cap, got {rankBucketCounts[0]}.");
            Assert.Equal(rankBucketCounts[0], rankBucketCounts[1]);

            for (int pIdx = 0; pIdx < singleParams.Count; pIdx++)
            {
                var refG = singleGrads[singleParams[pIdx]].ToVector();
                var r0Param = rankParams[0][pIdx];
                var r1Param = rankParams[1][pIdx];

                Assert.True(MaxAbsDiff(refG, rankGrads[0][r0Param].ToVector()) < 1e-5,
                    $"Rank 0 reduced gradient mismatch on param {pIdx}.");
                Assert.True(MaxAbsDiff(rankGrads[0][r0Param].ToVector(), rankGrads[1][r1Param].ToVector()) < 1e-6,
                    $"Rank 0 vs Rank 1 reduced gradient mismatch on param {pIdx}.");
            }
        }
        finally
        {
            ShutdownBackends(backends);
        }
    }

    [Theory(Timeout = 90000)]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ExactTrajectoryParity_TwoRanksStreaming_MatchesSingleRank(bool useSharedMemory)
    {
        const int sampleCount = 24;
        const int dataSeed = 2026;
        const long maxSteps = 6;
        var init = CreateInitialWeights(seed: 88);
        var (inputs, targets) = CreateRegressionDataset(sampleCount, seed: 404);

        // 1-Rank B=8 reference
        var singleOpt = CreateAdamOptimizer(lr: 0.015, enableClipping: true, maxGradNorm: 0.5);
        var singleModel = CreateMlp(singleOpt);
        singleModel.SetParameters(init.Clone());
        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(singleModel)
            .ConfigureOptimizer(singleOpt)
            .ConfigureDataLoader(CreateStreamingLoader(inputs, targets, batchSize: 8))
            .ConfigureGpuAcceleration(CpuOnlyConfig())
            .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
            {
                Seed = dataSeed,
                MaxSteps = maxSteps,
            })
            .BuildAsync();
        var expectedWeights = singleModel.GetParameters();
        Assert.True(MaxAbsDiff(init, expectedWeights) > 1e-3, "Training must move weights meaningfully.");

        // 2-Rank B0=4, B1=4
        var backends = CreateBackends(worldSize: 2, useSharedMemory: useSharedMemory);
        try
        {
            var rankModels = new FeedForwardNeuralNetwork<float>[2];
            var rankResults = new AiModelResult<float, Tensor<float>, Tensor<float>>[2];

            var tasks = Enumerable.Range(0, 2).Select(rank => Task.Run(async () =>
            {
                var opt = CreateAdamOptimizer(lr: 0.015, enableClipping: true, maxGradNorm: 0.5);
                var model = CreateMlp(opt);
                model.SetParameters(init.Clone());
                rankModels[rank] = model;

                var shardingCfg = new ShardingConfiguration<float>(backends[rank])
                {
                    OverlapCommunication = true,
                    GradientBucketSizeBytes = 128,
                };

                rankResults[rank] = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                    .ConfigureModel(model)
                    .ConfigureOptimizer(opt)
                    .ConfigureDataLoader(CreateStreamingLoader(inputs, targets, batchSize: 4))
                    .ConfigureGpuAcceleration(CpuOnlyConfig())
                    .ConfigureDistributedTraining(backends[rank], DistributedStrategy.DDP, shardingCfg)
                    .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
                    {
                        Seed = dataSeed,
                        MaxSteps = maxSteps,
                    })
                    .BuildAsync();
            })).ToArray();

            await Task.WhenAll(tasks);

            Assert.IsAssignableFrom<IShardedModel<float, Tensor<float>, Tensor<float>>>(rankResults[0].Model);
            Assert.IsAssignableFrom<IShardedModel<float, Tensor<float>, Tensor<float>>>(rankResults[1].Model);

            var r0Weights = rankModels[0].GetParameters();
            var r1Weights = rankModels[1].GetParameters();
            Assert.True(MaxAbsDiff(r0Weights, r1Weights) < 1e-6,
                $"Rank 0 and Rank 1 weights diverged: {MaxAbsDiff(r0Weights, r1Weights)}");
            Assert.True(MaxAbsDiff(expectedWeights, r0Weights) < 1e-5,
                $"2-rank trajectory diverged from 1-rank reference: {MaxAbsDiff(expectedWeights, r0Weights)}");
        }
        finally
        {
            ShutdownBackends(backends);
        }
    }

    #endregion

    #region (b) Uneven Micro-Batches + Token-Weighted Reduction + Tail Batches

    [Theory(Timeout = 90000)]
    [InlineData(20, 3)] // Global=8 (6+2): steps 0,1 = 6+2; step 2 tail = 4 samples split 3+1
    [InlineData(17, 6)] // Global=8 (6+2): 2 epochs; step 2 & step 5 tail = 1 sample split 1+0 (Rank 1 empty!)
    public async Task UnevenMicroBatches_TokenWeightedAndEmptyTailRank_MatchesSingleRankWithoutHanging(
        int sampleCount,
        long maxSteps)
    {
        const int dataSeed = 777;
        var init = CreateInitialWeights(seed: 123);
        var (inputs, targets) = CreateRegressionDataset(sampleCount, seed: 555);

        // 1-Rank B=8 reference (dropLast defaults to false, so tail batch has 4 or 1 samples)
        var singleOpt = CreateAdamOptimizer(lr: 0.01, enableClipping: true, maxGradNorm: 0.4);
        var singleModel = CreateMlp(singleOpt);
        singleModel.SetParameters(init.Clone());
        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(singleModel)
            .ConfigureOptimizer(singleOpt)
            .ConfigureDataLoader(CreateStreamingLoader(inputs, targets, batchSize: 8))
            .ConfigureGpuAcceleration(CpuOnlyConfig())
            .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
            {
                Seed = dataSeed,
                MaxSteps = maxSteps,
            })
            .BuildAsync();
        var expected = singleModel.GetParameters();

        // 2-Rank uneven B0=6, B1=2 (GlobalBatch=8)
        int[] rankBatchSizes = { 6, 2 };
        var backends = CreateBackends(worldSize: 2, useSharedMemory: true);
        try
        {
            var rankModels = new FeedForwardNeuralNetwork<float>[2];
            var tasks = Enumerable.Range(0, 2).Select(rank => Task.Run(async () =>
            {
                var opt = CreateAdamOptimizer(lr: 0.01, enableClipping: true, maxGradNorm: 0.4);
                var model = CreateMlp(opt);
                model.SetParameters(init.Clone());
                rankModels[rank] = model;

                await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                    .ConfigureModel(model)
                    .ConfigureOptimizer(opt)
                    .ConfigureDataLoader(CreateStreamingLoader(inputs, targets, batchSize: rankBatchSizes[rank]))
                    .ConfigureGpuAcceleration(CpuOnlyConfig())
                    .ConfigureDistributedTraining(backends[rank], DistributedStrategy.DDP)
                    .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
                    {
                        Seed = dataSeed,
                        MaxSteps = maxSteps,
                    })
                    .BuildAsync();
            })).ToArray();

            await Task.WhenAll(tasks);

            var r0 = rankModels[0].GetParameters();
            var r1 = rankModels[1].GetParameters();
            Assert.True(MaxAbsDiff(r0, r1) < 1e-6,
                $"Uneven SampleCount={sampleCount}: Rank 0 and Rank 1 diverged by {MaxAbsDiff(r0, r1)}.");
            Assert.True(MaxAbsDiff(expected, r0) < 1e-5,
                $"Uneven SampleCount={sampleCount}: 2-rank (6+2) diverged from 1-rank (8) by {MaxAbsDiff(expected, r0)}.");
        }
        finally
        {
            ShutdownBackends(backends);
        }
    }

    #endregion

    #region (c) All 4 Strategies on Streaming + Lazy Layer Initialization

    [Theory(Timeout = 120000)]
    [InlineData(DistributedStrategy.DDP)]
    [InlineData(DistributedStrategy.ZeRO1)]
    [InlineData(DistributedStrategy.ZeRO2)]
    [InlineData(DistributedStrategy.FSDP)]
    public async Task AllFourStrategies_LazyLayerInit_SynchronizeInitialParamsAndReturnShardedModel(
        DistributedStrategy strategy)
    {
        var (trainX, trainY) = CreateSequenceClassificationDataset(sampleCount: 16, seed: 801);
        var (valX, valY) = CreateSequenceClassificationDataset(sampleCount: 8, seed: 802);

        var backends = CreateBackends(worldSize: 2, useSharedMemory: false);
        try
        {
            var initialCounts = new int[2];
            var results = new AiModelResult<float, Tensor<float>, Tensor<float>>[2];
            var models = new Transformer<float>[2];

            // Give Rank 0 and Rank 1 DIFFERENT constructor seeds (11 vs 99) so Rank-0 initial broadcast
            // AFTER lazy parameter materialization is strictly required for weight agreement!
            int[] modelSeeds = { 11, 99 };

            var tasks = Enumerable.Range(0, 2).Select(rank => Task.Run(async () =>
            {
                var model = CreateLazyCanaryTransformer(seed: modelSeeds[rank]);
                initialCounts[rank] = model.GetParameters().Length;
                models[rank] = model;

                var opt = CreateAdamOptimizer(lr: 0.005, epochs: 10, enableClipping: true, maxGradNorm: 1.0);
                results[rank] = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                    .ConfigureModel(model)
                    .ConfigureOptimizer(opt)
                    .ConfigureDataLoader(CreateStreamingLoader(trainX, trainY, batchSize: 4))
                    .ConfigureGpuAcceleration(CpuOnlyConfig())
                    .ConfigureDistributedTraining(backends[rank], strategy)
                    .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
                    {
                        Seed = 42,
                        MaxSteps = 6,
                        ValidationLoader = CreateStreamingLoader(valX, valY, batchSize: 4),
                        ValidateEveryNSteps = 3,
                    })
                    .BuildAsync();
            })).ToArray();

            await Task.WhenAll(tasks);

            for (int r = 0; r < 2; r++)
            {
                var sharded = Assert.IsAssignableFrom<IShardedModel<float, Tensor<float>, Tensor<float>>>(results[r].Model);
                Assert.True(sharded.ParameterCount > initialCounts[r],
                    $"Strategy {strategy}: expected lazy layer parameters to materialize ({sharded.ParameterCount} <= {initialCounts[r]}).");
            }

            var w0 = models[0].GetParameters();
            var w1 = models[1].GetParameters();
            Assert.True(MaxAbsDiff(w0, w1) < 1e-5,
                $"Strategy {strategy}: Rank 0 and Rank 1 weights were not synchronized ({MaxAbsDiff(w0, w1)}).");

            var evalBatch = StackRows(valX, 0, 4);
            var pred = results[0].Predict(evalBatch);
            Assert.All(pred.ToArray(), v => Assert.False(float.IsNaN(v) || float.IsInfinity(v)));
        }
        finally
        {
            ShutdownBackends(backends);
        }
    }

    [Theory(Timeout = 90000)]
    [InlineData(DistributedStrategy.DDP)]
    [InlineData(DistributedStrategy.ZeRO1)]
    [InlineData(DistributedStrategy.ZeRO2)]
    [InlineData(DistributedStrategy.FSDP)]
    public async Task AllFourStrategies_ConvergeAndMatchSingleRankTrajectory(DistributedStrategy strategy)
    {
        var init = CreateInitialWeights(seed: 64);
        var (inputs, targets) = CreateRegressionDataset(sampleCount: 32, seed: 606);
        double initialMse = ComputeMse(
            new Func<FeedForwardNeuralNetwork<float>>(() =>
            {
                var m = CreateMlp(CreateAdamOptimizer());
                m.SetParameters(init.Clone());
                return m;
            })(),
            inputs,
            targets);

        // Single-rank B=8 reference over 10 steps
        var refOpt = CreateAdamOptimizer(lr: 0.02, enableClipping: true, maxGradNorm: 0.3);
        var refModel = CreateMlp(refOpt);
        refModel.SetParameters(init.Clone());
        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(refModel)
            .ConfigureOptimizer(refOpt)
            .ConfigureDataLoader(CreateStreamingLoader(inputs, targets, batchSize: 8))
            .ConfigureGpuAcceleration(CpuOnlyConfig())
            .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
            {
                Seed = 999,
                MaxSteps = 10,
            })
            .BuildAsync();
        var refWeights = refModel.GetParameters();

        var backends = CreateBackends(worldSize: 2, useSharedMemory: false);
        try
        {
            var models = new FeedForwardNeuralNetwork<float>[2];
            var tasks = Enumerable.Range(0, 2).Select(rank => Task.Run(async () =>
            {
                var opt = CreateAdamOptimizer(lr: 0.02, enableClipping: true, maxGradNorm: 0.3);
                var model = CreateMlp(opt);
                model.SetParameters(init.Clone());
                models[rank] = model;

                await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                    .ConfigureModel(model)
                    .ConfigureOptimizer(opt)
                    .ConfigureDataLoader(CreateStreamingLoader(inputs, targets, batchSize: 4))
                    .ConfigureGpuAcceleration(CpuOnlyConfig())
                    .ConfigureDistributedTraining(backends[rank], strategy)
                    .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
                    {
                        Seed = 999,
                        MaxSteps = 10,
                    })
                    .BuildAsync();
            })).ToArray();

            await Task.WhenAll(tasks);

            double finalMse = ComputeMse(models[0], inputs, targets);
            Assert.True(finalMse < initialMse,
                $"Strategy {strategy} did not reduce MSE: initial={initialMse:F4}, final={finalMse:F4}.");
            Assert.True(MaxAbsDiff(models[0].GetParameters(), models[1].GetParameters()) < 1e-6);
            Assert.True(MaxAbsDiff(refWeights, models[0].GetParameters()) < 1e-4,
                $"Strategy {strategy} trajectory drifted from 1-rank reference by {MaxAbsDiff(refWeights, models[0].GetParameters())}.");
        }
        finally
        {
            ShutdownBackends(backends);
        }
    }

    #endregion

    #region (d) Every Tape Step Path + Global-Norm Clipping Ahead of Optimizer

    [Theory(Timeout = 90000)]
    [InlineData("Eager")]
    [InlineData("CompiledFused")]
    [InlineData("GradientAccumulation")]
    public async Task TapePaths_SynchronizeBeforeGlobalNormClipping_MatchesSingleRankClippedStep(string pathMode)
    {
        // Tight MaxGradientNorm = 0.05 guarantees global-norm clipping fires on every step.
        // If clipping ran per-rank BEFORE AllReduce, 0.5*clip(g0) + 0.5*clip(g1) != clip(0.5*(g0+g1))!
        const double tightClipNorm = 0.05;
        const int dataSeed = 31415;
        var init = CreateInitialWeights(seed: 314);
        var (inputs, targets) = CreateRegressionDataset(sampleCount: 24, seed: 2718);

        var refOpt = CreateAdamOptimizer(lr: 0.02, enableClipping: true, maxGradNorm: tightClipNorm);
        var refModel = CreateMlp(refOpt);
        refModel.SetParameters(init.Clone());
        refModel.SetBaseTrainOptimizer(refOpt);
        if (pathMode == "Eager" || pathMode == "GradientAccumulation")
        {
            refModel.FusedSession.IsDisabled = true;
        }

        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(refModel)
            .ConfigureOptimizer(refOpt)
            .ConfigureDataLoader(CreateStreamingLoader(inputs, targets, batchSize: 8))
            .ConfigureGpuAcceleration(CpuOnlyConfig())
            .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
            {
                Seed = dataSeed,
                MaxSteps = 5,
                GradientAccumulationSteps = 1,
            })
            .BuildAsync();
        var refWeights = refModel.GetParameters();

        int perRankMicroBatch = pathMode == "GradientAccumulation" ? 2 : 4;
        int accumSteps = pathMode == "GradientAccumulation" ? 2 : 1; // 2 ranks * 2 microbatches * 2 samples = 8 global!

        var backends = CreateBackends(worldSize: 2, useSharedMemory: false);
        try
        {
            var models = new FeedForwardNeuralNetwork<float>[2];
            var tasks = Enumerable.Range(0, 2).Select(rank => Task.Run(async () =>
            {
                var opt = CreateAdamOptimizer(lr: 0.02, enableClipping: true, maxGradNorm: tightClipNorm);
                var model = CreateMlp(opt);
                model.SetParameters(init.Clone());
                model.SetBaseTrainOptimizer(opt);
                if (pathMode == "Eager")
                {
                    model.FusedSession.IsDisabled = true;
                }

                models[rank] = model;

                await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                    .ConfigureModel(model)
                    .ConfigureOptimizer(opt)
                    .ConfigureDataLoader(CreateStreamingLoader(inputs, targets, batchSize: perRankMicroBatch))
                    .ConfigureGpuAcceleration(CpuOnlyConfig())
                    .ConfigureDistributedTraining(backends[rank], DistributedStrategy.DDP)
                    .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
                    {
                        Seed = dataSeed,
                        MaxSteps = 5,
                        GradientAccumulationSteps = accumSteps,
                    })
                    .BuildAsync();
            })).ToArray();

            await Task.WhenAll(tasks);

            Assert.True(MaxAbsDiff(models[0].GetParameters(), models[1].GetParameters()) < 1e-6);
            Assert.True(MaxAbsDiff(refWeights, models[0].GetParameters()) < 1e-5,
                $"Tape path '{pathMode}' diverged from 1-rank clipped trajectory by {MaxAbsDiff(refWeights, models[0].GetParameters())}.");
        }
        finally
        {
            ShutdownBackends(backends);
        }
    }

    [Fact(Timeout = 90000)]
    public async Task TapePath_Streaming8BitOptimizer_SynchronizesAcrossRanksAndMatchesSingleRank()
    {
        var init = CreateInitialWeights(seed: 420);
        var (inputs, targets) = CreateRegressionDataset(sampleCount: 24, seed: 421);

        var refOpt = CreateAdamOptimizer(lr: 0.01, enableClipping: false);
        var refModel = CreateMlp(refOpt);
        refModel.SetParameters(init.Clone());
        refModel.StreamingTraining = StreamingTrainingMode.ForceOn;
        refModel.StreamingTrainingLearningRate = 0.01;
        double initialMse = ComputeMse(refModel, inputs, targets);

        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(refModel)
            .ConfigureOptimizer(refOpt)
            .ConfigureDataLoader(CreateStreamingLoader(inputs, targets, batchSize: 8))
            .ConfigureGpuAcceleration(CpuOnlyConfig())
            .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
            {
                Seed = 505,
                MaxSteps = 8,
            })
            .BuildAsync();
        var refWeights = refModel.GetParameters();

        var backends = CreateBackends(worldSize: 2, useSharedMemory: false);
        try
        {
            var models = new FeedForwardNeuralNetwork<float>[2];
            var tasks = Enumerable.Range(0, 2).Select(rank => Task.Run(async () =>
            {
                var opt = CreateAdamOptimizer(lr: 0.01, enableClipping: false);
                var model = CreateMlp(opt);
                model.SetParameters(init.Clone());
                model.StreamingTraining = StreamingTrainingMode.ForceOn;
                model.StreamingTrainingLearningRate = 0.01;
                models[rank] = model;

                await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                    .ConfigureModel(model)
                    .ConfigureOptimizer(opt)
                    .ConfigureDataLoader(CreateStreamingLoader(inputs, targets, batchSize: 4))
                    .ConfigureGpuAcceleration(CpuOnlyConfig())
                    .ConfigureDistributedTraining(backends[rank], DistributedStrategy.DDP)
                    .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
                    {
                        Seed = 505,
                        MaxSteps = 8,
                    })
                    .BuildAsync();
            })).ToArray();

            await Task.WhenAll(tasks);

            double finalMse = ComputeMse(models[0], inputs, targets);
            Assert.True(finalMse < initialMse, $"Streaming 8-bit distributed path did not reduce MSE ({initialMse} -> {finalMse}).");
            Assert.True(MaxAbsDiff(models[0].GetParameters(), models[1].GetParameters()) < 1e-6,
                "Rank 0 and Rank 1 diverged under StreamingTrainingMode.ForceOn.");
            Assert.True(MaxAbsDiff(refWeights, models[0].GetParameters()) < 1e-4,
                $"2-rank StreamingTrainingMode.ForceOn diverged from 1-rank by {MaxAbsDiff(refWeights, models[0].GetParameters())}.");
        }
        finally
        {
            ShutdownBackends(backends);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task TapePath_TapeTrainingStepperDirect_SynchronizesAndClipsAcrossRanks()
    {
        var init = CreateInitialWeights(seed: 19);
        var (inputs, targets) = CreateRegressionDataset(sampleCount: 8, seed: 29);
        var lossFn = new MeanSquaredErrorLoss<float>();

        // 1-Rank direct TapeTrainingStepper.EagerStep reference (2 steps from fresh optimizer)
        var refEagerOpt = CreateAdamOptimizer(lr: 0.02, enableClipping: true, maxGradNorm: 0.05);
        var refEagerModel = CreateMlp(refEagerOpt);
        refEagerModel.SetParameters(init.Clone());
        var refEagerLayers = refEagerModel.Layers.OfType<ITrainableLayer<float>>().ToList();
        for (int s = 0; s < 2; s++)
        {
            TapeTrainingStepper<float>.EagerStep(new FusedTrainingStepRequest<float>
            {
                Layers = refEagerLayers,
                Input = StackRows(inputs, 0, 8),
                Target = StackRows(targets, 0, 8),
                Forward = x => refEagerModel.ForwardForTraining(x),
                ComputeLoss = (pred, y) => lossFn.ComputeTapeLoss(pred, y),
                Optimizer = refEagerOpt,
                ModelGradientClip = 0.05,
            }, refEagerModel);
        }

        var expectedAfterEager = refEagerModel.GetParameters().Clone();

        // 1-Rank direct TapeTrainingStepper.Step reference (2 compiled steps from fresh optimizer)
        var refCompOpt = CreateAdamOptimizer(lr: 0.02, enableClipping: true, maxGradNorm: 0.05);
        var refCompModel = CreateMlp(refCompOpt);
        refCompModel.SetParameters(init.Clone());
        var refStepper = TapeTrainingStepper<float>.ForOwner(refCompModel);
        var refCompLayers = refCompModel.Layers.OfType<ITrainableLayer<float>>().ToList();
        for (int s = 0; s < 2; s++)
        {
            refStepper.Step(new FusedTrainingStepRequest<float>
            {
                Layers = refCompLayers,
                Input = StackRows(inputs, 0, 8),
                Target = StackRows(targets, 0, 8),
                Forward = x => refCompModel.ForwardForTraining(x),
                ComputeLoss = (pred, y) => lossFn.ComputeTapeLoss(pred, y),
                Optimizer = refCompOpt,
                ModelGradientClip = 0.05,
            });
        }

        var expectedAfterCompiled = refCompModel.GetParameters();

        // 2-Rank direct TapeTrainingStepper with DistributedCoordinator attached (both EagerStep and compiled Step)
        var backends = CreateBackends(worldSize: 2, useSharedMemory: false);
        try
        {
            var eagerModels = new FeedForwardNeuralNetwork<float>[2];
            var compiledModels = new FeedForwardNeuralNetwork<float>[2];
            var tasks = Enumerable.Range(0, 2).Select(rank => Task.Run(() =>
            {
                backends[rank].Initialize();

                using (var eagerCoord = new DistributedTapeCoordinator<float>(backends[rank], DistributedStrategy.DDP))
                {
                    var opt = CreateAdamOptimizer(lr: 0.02, enableClipping: true, maxGradNorm: 0.05);
                    var model = CreateMlp(opt);
                    model.SetParameters(init.Clone());
                    model.DistributedCoordinator = eagerCoord;
                    eagerModels[rank] = model;
                    var layers = model.Layers.OfType<ITrainableLayer<float>>().ToList();
                    for (int s = 0; s < 2; s++)
                    {
                        TapeTrainingStepper<float>.EagerStep(new FusedTrainingStepRequest<float>
                        {
                            Layers = layers,
                            Input = StackRows(inputs, rank * 4, 4),
                            Target = StackRows(targets, rank * 4, 4),
                            Forward = x => model.ForwardForTraining(x),
                            ComputeLoss = (pred, y) => lossFn.ComputeTapeLoss(pred, y),
                            Optimizer = opt,
                            ModelGradientClip = 0.05,
                        }, model);
                    }

                    model.DistributedCoordinator = null;
                }

                using (var compCoord = new DistributedTapeCoordinator<float>(backends[rank], DistributedStrategy.DDP))
                {
                    var opt = CreateAdamOptimizer(lr: 0.02, enableClipping: true, maxGradNorm: 0.05);
                    var model = CreateMlp(opt);
                    model.SetParameters(init.Clone());
                    model.DistributedCoordinator = compCoord;
                    compiledModels[rank] = model;
                    var stepper = TapeTrainingStepper<float>.ForOwner(model);
                    var layers = model.Layers.OfType<ITrainableLayer<float>>().ToList();
                    for (int s = 0; s < 2; s++)
                    {
                        stepper.Step(new FusedTrainingStepRequest<float>
                        {
                            Layers = layers,
                            Input = StackRows(inputs, rank * 4, 4),
                            Target = StackRows(targets, rank * 4, 4),
                            Forward = x => model.ForwardForTraining(x),
                            ComputeLoss = (pred, y) => lossFn.ComputeTapeLoss(pred, y),
                            Optimizer = opt,
                            ModelGradientClip = 0.05,
                        });
                    }

                    model.DistributedCoordinator = null;
                }
            })).ToArray();

            await Task.WhenAll(tasks);

            Assert.True(MaxAbsDiff(eagerModels[0].GetParameters(), eagerModels[1].GetParameters()) < 1e-6);
            Assert.True(MaxAbsDiff(expectedAfterEager, eagerModels[0].GetParameters()) < 1e-5,
                $"Direct TapeTrainingStepper.EagerStep 2-rank diverged from 1-rank by {MaxAbsDiff(expectedAfterEager, eagerModels[0].GetParameters())}.");
            Assert.True(MaxAbsDiff(compiledModels[0].GetParameters(), compiledModels[1].GetParameters()) < 1e-6);
            Assert.True(MaxAbsDiff(expectedAfterCompiled, compiledModels[0].GetParameters()) < 1e-5,
                $"Direct TapeTrainingStepper.Step 2-rank diverged from 1-rank by {MaxAbsDiff(expectedAfterCompiled, compiledModels[0].GetParameters())}.");
        }
        finally
        {
            ShutdownBackends(backends);
        }
    }

    #endregion

    #region (e) Mixed Precision on Streaming + Dynamic LossScaler Overflow Coordination

    [Theory(Timeout = 90000)]
    [InlineData(DistributedStrategy.DDP)]
    [InlineData(DistributedStrategy.ZeRO2)]
    public async Task MixedPrecision_OnStreaming_EngagesAndCoordinatesOverflowAcrossRanks(DistributedStrategy strategy)
    {
        var init = CreateInitialWeights(seed: 71);
        var (inputs, targets) = CreateRegressionDataset(sampleCount: 24, seed: 72);

        var backends = CreateBackends(worldSize: 2, useSharedMemory: false);
        try
        {
            var results = new AiModelResult<float, Tensor<float>, Tensor<float>>[2];
            var models = new FeedForwardNeuralNetwork<float>[2];

            var tasks = Enumerable.Range(0, 2).Select(rank => Task.Run(async () =>
            {
                var opt = CreateAdamOptimizer(lr: 0.01, enableClipping: true, maxGradNorm: 1.0);
                var model = CreateMlp(opt);
                model.SetParameters(init.Clone());
                models[rank] = model;

                var mpConfig = new AiDotNet.MixedPrecision.MixedPrecisionConfig
                {
                    PrecisionType = MixedPrecisionType.FP16,
                    InitialLossScale = 512.0,
                    EnableDynamicScaling = true,
                    ScaleGrowthInterval = 4,
                };

                results[rank] = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                    .ConfigureModel(model)
                    .ConfigureOptimizer(opt)
                    .ConfigureDataLoader(CreateStreamingLoader(inputs, targets, batchSize: 4))
                    .ConfigureGpuAcceleration(CpuOnlyConfig())
                    .ConfigureMixedPrecision(mpConfig)
                    .ConfigureDistributedTraining(backends[rank], strategy)
                    .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
                    {
                        Seed = 88,
                        MaxSteps = 6,
                    })
                    .BuildAsync();
            })).ToArray();

            await Task.WhenAll(tasks);

            for (int r = 0; r < 2; r++)
            {
                Assert.True(results[r].MixedPrecisionEngaged, $"Rank {r}: MixedPrecisionEngaged should be true.");
                Assert.NotNull(results[r].MixedPrecisionStatus);
                Assert.StartsWith("engaged:", results[r].MixedPrecisionStatus!, StringComparison.Ordinal);
            }

            Assert.True(MaxAbsDiff(models[0].GetParameters(), models[1].GetParameters()) < 1e-5);

            // Verify ZeRO2/FSDP single-shard overflow coordination: inject Inf ONLY into Rank 1's owned tail parameter
            // and verify BOTH Rank 0 and Rank 1 skip the update and back off LossScaler identically!
            var overflowBackends = CreateBackends(worldSize: 2, useSharedMemory: false);
            try
            {
                var steppedFlags = new bool[2];
                var scalesAfter = new double[2];
                var postWeights = new Vector<float>[2];

                var overflowTasks = Enumerable.Range(0, 2).Select(rank => Task.Run(() =>
                {
                    overflowBackends[rank].Initialize();
                    using var coord = new DistributedTapeCoordinator<float>(overflowBackends[rank], DistributedStrategy.ZeRO2);
                    var opt = CreateAdamOptimizer(lr: 0.05);
                    var m = CreateMlp(opt);
                    m.SetParameters(init.Clone());
                    m.EnableMixedPrecision(new AiDotNet.MixedPrecision.MixedPrecisionConfig
                    {
                        PrecisionType = MixedPrecisionType.FP16,
                        InitialLossScale = 1024.0,
                        EnableDynamicScaling = true,
                        ScaleBackoffFactor = 0.5,
                    });

                    var prms = GetTrainableParameters(m);
                    coord.EnsureInitialParametersSynchronized(prms);
                    var (rw, _, _) = coord.SynchronizeStepMetadata(4, 1.0);
                    coord.BeginBackwardStep(prms, rw);

                    var grads = new Dictionary<Tensor<float>, Tensor<float>>();
                    for (int i = 0; i < prms.Count; i++)
                    {
                        var gArr = new float[prms[i].Length];
                        for (int k = 0; k < gArr.Length; k++)
                        {
                            gArr[k] = 1.0f;
                        }

                        // Poison ONLY the very last element (in Rank 1's shard) on Rank 1!
                        if (rank == 1 && i == prms.Count - 1)
                        {
                            gArr[gArr.Length - 1] = float.PositiveInfinity;
                        }

                        var gTensor = new Tensor<float>(prms[i].Shape.ToArray(), new Vector<float>(gArr));
                        grads[prms[i]] = gTensor;
                        coord.OnParameterGradientReady(prms[i], gTensor);
                    }

                    steppedFlags[rank] = coord.SynchronizeUnscaleClipAndStep(
                        prms,
                        grads,
                        opt,
                        m.ActiveMixedPrecisionContext,
                        restoreFp32Masters: null,
                        syncFp32MastersFromWorking: null,
                        regularizeGradients: null,
                        maxGradNorm: 1.0,
                        executeFullParameterStep: ctx => opt.Step(ctx));

                    scalesAfter[rank] = m.ActiveMixedPrecisionContext?.LossScaler?.Scale ?? -1.0;
                    postWeights[rank] = m.GetParameters();
                })).ToArray();

                await Task.WhenAll(overflowTasks);

                Assert.False(steppedFlags[0], "Rank 0 must skip step when Rank 1's ZeRO2 shard overflows.");
                Assert.False(steppedFlags[1], "Rank 1 must skip step when its ZeRO2 shard overflows.");
                Assert.Equal(512.0, scalesAfter[0], 6);
                Assert.Equal(512.0, scalesAfter[1], 6);
                Assert.Equal(0.0, MaxAbsDiff(init, postWeights[0]), 6);
                Assert.Equal(0.0, MaxAbsDiff(init, postWeights[1]), 6);
            }
            finally
            {
                ShutdownBackends(overflowBackends);
            }
        }
        finally
        {
            ShutdownBackends(backends);
        }
    }

    #endregion

    #region (f) Time-Based Auto-Checkpointing

    [Fact(Timeout = 90000)]
    public async Task TimeBasedAutoCheckpointing_SaveIntervalOnly_WritesCheckpointsWithoutRankHang()
    {
        var init = CreateInitialWeights(seed: 91);
        var (inputs, targets) = CreateRegressionDataset(sampleCount: 32, seed: 92);
        string ckptDir = Path.Combine(_tempRoot, "wallclock_ckpt");

        var backends = CreateBackends(worldSize: 2, useSharedMemory: true);
        try
        {
            var tasks = Enumerable.Range(0, 2).Select(rank => Task.Run(async () =>
            {
                var opt = CreateAdamOptimizer(lr: 0.01);
                var model = CreateMlp(opt);
                model.SetParameters(init.Clone());

                var ckptMgr = new CheckpointManager<float, Tensor<float>, Tensor<float>>(ckptDir);
                ckptMgr.ConfigureAutoCheckpointing(
                    saveFrequency: 0,
                    keepLast: 5,
                    saveOnImprovement: false,
                    metricName: "loss",
                    minimizeMetric: true,
                    saveInterval: TimeSpan.FromMilliseconds(75));

                // Skew Rank 1 slightly via sample delay so naive uncoordinated wall-clock checks would diverge!
                int delayMs = rank == 0 ? 12 : 18;

                await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                    .ConfigureModel(model)
                    .ConfigureOptimizer(opt)
                    .ConfigureDataLoader(CreateStreamingLoader(inputs, targets, batchSize: 4, sampleDelayMs: delayMs))
                    .ConfigureGpuAcceleration(CpuOnlyConfig())
                    .ConfigureCheckpointManager(ckptMgr)
                    .ConfigureDistributedTraining(backends[rank], DistributedStrategy.ZeRO1)
                    .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
                    {
                        Seed = 1001,
                        MaxSteps = 8,
                    })
                    .BuildAsync();
            })).ToArray();

            await Task.WhenAll(tasks);

            Assert.True(Directory.Exists(ckptDir), "Checkpoint directory was not created.");
            var jsonFiles = Directory.GetFiles(ckptDir, "*.json")
                .Where(p => p.IndexOf(".tmp", StringComparison.OrdinalIgnoreCase) < 0)
                .ToArray();
            Assert.True(jsonFiles.Length >= 1,
                "Expected >=1 checkpoint written purely from wall-clock SaveInterval (saveFrequency=0, saveOnImprovement=false).");
        }
        finally
        {
            ShutdownBackends(backends);
        }
    }

    #endregion

    #region (g) Cross-World-Size Checkpoint Resume

    [Theory(Timeout = 120000)]
    [InlineData(DistributedStrategy.DDP)]
    [InlineData(DistributedStrategy.ZeRO2)]
    public async Task CrossWorldSizeResume_TwoRanksToOneRank_MatchesUninterruptedSingleRank(DistributedStrategy strategy)
    {
        const int dataSeed = 4242;
        var init = CreateInitialWeights(seed: 135);
        var (inputs, targets) = CreateRegressionDataset(sampleCount: 32, seed: 246);
        WarmupStableDecayScheduler NewScheduler() =>
            new WarmupStableDecayScheduler(0.02, warmupSteps: 2, decayStartStep: 4, decaySteps: 4);

        // Uninterrupted 6-step 1-rank (B=8) ground truth
        var straightOpt = CreateAdamOptimizer(lr: 0.02, enableClipping: true, maxGradNorm: 0.5, scheduler: NewScheduler());
        var straightModel = CreateMlp(straightOpt);
        straightModel.SetParameters(init.Clone());
        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(straightModel)
            .ConfigureOptimizer(straightOpt)
            .ConfigureDataLoader(CreateStreamingLoader(inputs, targets, batchSize: 8))
            .ConfigureGpuAcceleration(CpuOnlyConfig())
            .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
            {
                Seed = dataSeed,
                MaxSteps = 6,
            })
            .BuildAsync();
        var expectedWeights = straightModel.GetParameters();

        // Phase 1: Train 3 steps on 2 ranks (B0=4, B1=4 -> GlobalBatch=8) and save checkpoint at step 3
        string ckptDir = Path.Combine(_tempRoot, $"resume_2to1_{strategy}");
        var backends = CreateBackends(worldSize: 2, useSharedMemory: false);
        Vector<float> step3Weights;
        try
        {
            var phase1Models = new FeedForwardNeuralNetwork<float>[2];
            var tasks = Enumerable.Range(0, 2).Select(rank => Task.Run(async () =>
            {
                var opt = CreateAdamOptimizer(lr: 0.02, enableClipping: true, maxGradNorm: 0.5, scheduler: NewScheduler());
                var model = CreateMlp(opt);
                model.SetParameters(init.Clone());
                phase1Models[rank] = model;

                var ckpt = new CheckpointManager<float, Tensor<float>, Tensor<float>>(ckptDir);
                ckpt.ConfigureAutoCheckpointing(saveFrequency: 3, keepLast: 3, saveOnImprovement: false);

                await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                    .ConfigureModel(model)
                    .ConfigureOptimizer(opt)
                    .ConfigureDataLoader(CreateStreamingLoader(inputs, targets, batchSize: 4))
                    .ConfigureGpuAcceleration(CpuOnlyConfig())
                    .ConfigureCheckpointManager(ckpt)
                    .ConfigureDistributedTraining(backends[rank], strategy)
                    .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
                    {
                        Seed = dataSeed,
                        MaxSteps = 3,
                        ResumeFromLatestCheckpoint = false,
                    })
                    .BuildAsync();
            })).ToArray();

            await Task.WhenAll(tasks);
            step3Weights = phase1Models[0].GetParameters();
        }
        finally
        {
            ShutdownBackends(backends);
        }

        Assert.True(MaxAbsDiff(step3Weights, expectedWeights) > 1e-3,
            "Step 3 weights must differ from step 6 weights for resume parity to be non-vacuous.");

        // Phase 2: Resume on 1 rank (B=8) from scrambled weights up to MaxSteps=6
        var scrambled = new Vector<float>(Enumerable.Repeat(0.25f, init.Length).ToArray());
        var resumeOpt = CreateAdamOptimizer(lr: 0.02, enableClipping: true, maxGradNorm: 0.5, scheduler: NewScheduler());
        var resumeModel = CreateMlp(resumeOpt);
        resumeModel.SetParameters(scrambled);

        var resumeCkpt = new CheckpointManager<float, Tensor<float>, Tensor<float>>(ckptDir);
        resumeCkpt.ConfigureAutoCheckpointing(saveFrequency: 100, keepLast: 3, saveOnImprovement: false);

        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(resumeModel)
            .ConfigureOptimizer(resumeOpt)
            .ConfigureDataLoader(CreateStreamingLoader(inputs, targets, batchSize: 8))
            .ConfigureGpuAcceleration(CpuOnlyConfig())
            .ConfigureCheckpointManager(resumeCkpt)
            .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
            {
                Seed = dataSeed,
                MaxSteps = 6,
                ResumeFromLatestCheckpoint = true,
            })
            .BuildAsync();

        Assert.True(MaxAbsDiff(expectedWeights, resumeModel.GetParameters()) < 1e-5,
            $"2-rank ({strategy}) -> 1-rank resume diverged from uninterrupted 1-rank run by {MaxAbsDiff(expectedWeights, resumeModel.GetParameters())}.");
        Assert.Equal(straightOpt.GetCurrentLearningRate(), resumeOpt.GetCurrentLearningRate(), 9);
    }

    [Theory(Timeout = 120000)]
    [InlineData(DistributedStrategy.DDP)]
    [InlineData(DistributedStrategy.FSDP)]
    public async Task CrossWorldSizeResume_OneRankToTwoRanks_MatchesUninterruptedSingleRank(DistributedStrategy strategy)
    {
        const int dataSeed = 5353;
        var init = CreateInitialWeights(seed: 246);
        var (inputs, targets) = CreateRegressionDataset(sampleCount: 32, seed: 357);
        WarmupStableDecayScheduler NewScheduler() =>
            new WarmupStableDecayScheduler(0.02, warmupSteps: 2, decayStartStep: 4, decaySteps: 4);

        // Uninterrupted 6-step 1-rank (B=8) ground truth
        var straightOpt = CreateAdamOptimizer(lr: 0.02, enableClipping: true, maxGradNorm: 0.5, scheduler: NewScheduler());
        var straightModel = CreateMlp(straightOpt);
        straightModel.SetParameters(init.Clone());
        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(straightModel)
            .ConfigureOptimizer(straightOpt)
            .ConfigureDataLoader(CreateStreamingLoader(inputs, targets, batchSize: 8))
            .ConfigureGpuAcceleration(CpuOnlyConfig())
            .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
            {
                Seed = dataSeed,
                MaxSteps = 6,
            })
            .BuildAsync();
        var expectedWeights = straightModel.GetParameters();

        // Phase 1: Train 3 steps on 1 rank (B=8) and checkpoint at step 3
        string ckptDir = Path.Combine(_tempRoot, $"resume_1to2_{strategy}");
        var p1Opt = CreateAdamOptimizer(lr: 0.02, enableClipping: true, maxGradNorm: 0.5, scheduler: NewScheduler());
        var p1Model = CreateMlp(p1Opt);
        p1Model.SetParameters(init.Clone());
        var p1Ckpt = new CheckpointManager<float, Tensor<float>, Tensor<float>>(ckptDir);
        p1Ckpt.ConfigureAutoCheckpointing(saveFrequency: 3, keepLast: 3, saveOnImprovement: false);

        await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(p1Model)
            .ConfigureOptimizer(p1Opt)
            .ConfigureDataLoader(CreateStreamingLoader(inputs, targets, batchSize: 8))
            .ConfigureGpuAcceleration(CpuOnlyConfig())
            .ConfigureCheckpointManager(p1Ckpt)
            .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
            {
                Seed = dataSeed,
                MaxSteps = 3,
                ResumeFromLatestCheckpoint = false,
            })
            .BuildAsync();

        Assert.True(MaxAbsDiff(p1Model.GetParameters(), expectedWeights) > 1e-3);

        // Phase 2: Resume to step 6 on 2 UNEVEN ranks (B0=6, B1=2 -> GlobalBatch=8) from scrambled weights!
        int[] rankBatchSizes = { 6, 2 };
        var backends = CreateBackends(worldSize: 2, useSharedMemory: false);
        try
        {
            var resumedModels = new FeedForwardNeuralNetwork<float>[2];
            var tasks = Enumerable.Range(0, 2).Select(rank => Task.Run(async () =>
            {
                var scrambled = new Vector<float>(Enumerable.Repeat(-0.33f * (rank + 1), init.Length).ToArray());
                var opt = CreateAdamOptimizer(lr: 0.02, enableClipping: true, maxGradNorm: 0.5, scheduler: NewScheduler());
                var model = CreateMlp(opt);
                model.SetParameters(scrambled);
                resumedModels[rank] = model;

                var ckpt = new CheckpointManager<float, Tensor<float>, Tensor<float>>(ckptDir);
                ckpt.ConfigureAutoCheckpointing(saveFrequency: 100, keepLast: 3, saveOnImprovement: false);

                await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                    .ConfigureModel(model)
                    .ConfigureOptimizer(opt)
                    .ConfigureDataLoader(CreateStreamingLoader(inputs, targets, batchSize: rankBatchSizes[rank]))
                    .ConfigureGpuAcceleration(CpuOnlyConfig())
                    .ConfigureCheckpointManager(ckpt)
                    .ConfigureDistributedTraining(backends[rank], strategy)
                    .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
                    {
                        Seed = dataSeed,
                        MaxSteps = 6,
                        ResumeFromLatestCheckpoint = true,
                    })
                    .BuildAsync();
            })).ToArray();

            await Task.WhenAll(tasks);

            Assert.True(MaxAbsDiff(resumedModels[0].GetParameters(), resumedModels[1].GetParameters()) < 1e-6);
            Assert.True(MaxAbsDiff(expectedWeights, resumedModels[0].GetParameters()) < 1e-5,
                $"1-rank -> 2-rank ({strategy}, 6+2) resume diverged from uninterrupted 1-rank run by {MaxAbsDiff(expectedWeights, resumedModels[0].GetParameters())}.");
        }
        finally
        {
            ShutdownBackends(backends);
        }
    }

    #endregion
}
