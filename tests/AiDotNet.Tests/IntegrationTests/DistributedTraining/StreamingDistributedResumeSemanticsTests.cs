using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.CheckpointManagement;
using AiDotNet.DistributedTraining;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.DistributedTraining;

/// <summary>
/// PyTorch DDP semantics the streaming path must keep: a resumed run continues the data order and the dropout masks of
/// the run it resumes (with or without a configured seed), rank 0's buffers are every rank's buffers, and gradients
/// are weighted by the targets the loss actually averages over.
/// </summary>
public sealed partial class StreamingDistributedEnablersTests
{
    private const int ResumeSamples = 32;

    private static FeedForwardNeuralNetwork<float> CreateNet(
        IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>>? optimizer,
        Func<List<ILayer<float>>> layers,
        ILossFunction<float>? loss = null,
        NeuralNetworkTaskType task = NeuralNetworkTaskType.Regression)
    {
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: task,
            inputSize: InDim,
            outputSize: OutDim,
            layers: layers());
        var net = new FeedForwardNeuralNetwork<float>(arch, optimizer: optimizer, lossFunction: loss ?? new MeanSquaredErrorLoss<float>());
        _ = net.Predict(new Tensor<float>(new[] { 1, InDim }));
        return net;
    }

    private static List<ILayer<float>> DropoutMlpLayers() => new()
    {
        new InputLayer<float>(InDim),
        new DenseLayer<float>(outputSize: HiddenDim, activationFunction: new ReLUActivation<float>()),
        new DropoutLayer<float>(0.3),
        new DenseLayer<float>(outputSize: HiddenDim, activationFunction: new ReLUActivation<float>()),
        new DropoutLayer<float>(0.3),
        new DenseLayer<float>(outputSize: OutDim, activationFunction: new IdentityActivation<float>()),
    };

    private static Vector<float> InitFor(Func<FeedForwardNeuralNetwork<float>> probe, int seed)
    {
        int len = probe().GetParameters().Length;
        var rng = RandomHelper.CreateSeededRandom(seed);
        var arr = new float[len];
        for (int i = 0; i < len; i++) arr[i] = (float)(rng.NextDouble() * 0.8 - 0.4);
        return new Vector<float>(arr);
    }

    /// <summary>Trains <paramref name="worldSize"/> ranks (global batch 8) and returns rank 0's model.</summary>
    private async Task<FeedForwardNeuralNetwork<float>[]> TrainRanksAsync(
        int worldSize,
        Func<AdamOptimizer<float, Tensor<float>, Tensor<float>>, FeedForwardNeuralNetwork<float>> createModel,
        Vector<float> init,
        Tensor<float>[] inputs,
        Tensor<float>[] targets,
        int? dataSeed,
        long maxSteps,
        string? checkpointDir,
        int saveFrequency,
        bool resume,
        Action<int, FeedForwardNeuralNetwork<float>>? beforeBuild = null)
    {
        var models = new FeedForwardNeuralNetwork<float>[worldSize];
        async Task RunRank(int rank, ICommunicationBackend<float>? backend)
        {
            var opt = CreateAdamOptimizer(lr: 0.02, enableClipping: true, maxGradNorm: 0.5);
            var model = createModel(opt);
            model.SetParameters(init.Clone());
            beforeBuild?.Invoke(rank, model);
            models[rank] = model;
            var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                .ConfigureModel(model)
                .ConfigureOptimizer(opt)
                .ConfigureDataLoader(CreateStreamingLoader(inputs, targets, batchSize: 8 / worldSize))
                .ConfigureGpuAcceleration(CpuOnlyConfig())
                .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
                {
                    Seed = dataSeed,
                    MaxSteps = maxSteps,
                    ResumeFromLatestCheckpoint = resume,
                });
            if (checkpointDir is not null)
            {
                var ckpt = new CheckpointManager<float, Tensor<float>, Tensor<float>>(checkpointDir);
                ckpt.ConfigureAutoCheckpointing(saveFrequency: saveFrequency, keepLast: 5, saveOnImprovement: false);
                builder = builder.ConfigureCheckpointManager(ckpt);
            }
            if (backend is not null) builder = builder.ConfigureDistributedTraining(backend, DistributedStrategy.DDP);
            await builder.BuildAsync();
        }

        if (worldSize == 1)
        {
            await RunRank(0, null);
            return models;
        }

        var backends = CreateBackends(worldSize, useSharedMemory: false);
        try
        {
            await Task.WhenAll(Enumerable.Range(0, worldSize).Select(rank => Task.Run(() => RunRank(rank, backends[rank]))));
        }
        finally
        {
            ShutdownBackends(backends);
        }
        return models;
    }

    private static void DeleteCheckpointsAfter(string dir, int step)
    {
        var manager = new CheckpointManager<float, Tensor<float>, Tensor<float>>(dir);
        var later = manager.ListCheckpoints().Where(c => c.Step > step).ToList();
        Assert.NotEmpty(later);
        foreach (var c in later) manager.DeleteCheckpoint(c.CheckpointId);
        Assert.Contains(manager.ListCheckpoints(), c => c.Step == step);
    }

    /// <summary>
    /// (c) and G4: checkpoint at world size 2 mid-epoch, resume at 2 (and at 1), and finish on the trajectory the
    /// uninterrupted run took -- also with no configured seed, where the shuffle order used to be redrawn on resume.
    /// </summary>
    [Theory(Timeout = 180000)]
    [InlineData(7171, 2)]
    [InlineData(null, 2)]
    [InlineData(null, 1)]
    public async Task Resume_AtTheSameOrASmallerWorldSize_ContinuesTheUninterruptedTrajectory(int? dataSeed, int resumeWorldSize)
    {
        var (inputs, targets) = CreateRegressionDataset(ResumeSamples, seed: 612);
        var init = CreateInitialWeights(seed: 613);
        string dir = Path.Combine(_tempRoot, $"resume_{dataSeed?.ToString() ?? "noseed"}_{resumeWorldSize}");

        // Uninterrupted: 6 steps of global batch 8 over 32 samples crosses into epoch 1; it checkpoints at 3 and 6.
        var straight = await TrainRanksAsync(2, CreateMlp, init, inputs, targets, dataSeed, 6, dir, 3, resume: false);
        var expected = straight[0].GetParameters();
        Assert.True(MaxAbsDiff(straight[0].GetParameters(), straight[1].GetParameters()) < 1e-6);

        DeleteCheckpointsAfter(dir, 3);
        var scrambled = new Vector<float>(Enumerable.Repeat(0.25f, init.Length).ToArray());
        var resumed = await TrainRanksAsync(resumeWorldSize, CreateMlp, scrambled, inputs, targets, dataSeed, 6, dir, 100, resume: true);
        double diff = MaxAbsDiff(expected, resumed[0].GetParameters());
        Assert.True(diff < 1e-5, $"2 -> {resumeWorldSize} resume (seed {dataSeed?.ToString() ?? "none"}) diverged from the uninterrupted run by {diff}.");
    }

    /// <summary>
    /// G6: with dropout, a run resumed at the same world size draws the masks the uninterrupted run drew -- the draw
    /// counters and the seed drawn for unseeded layers are checkpointed -- and data-parallel ranks draw distinct masks.
    /// </summary>
    [Theory(Timeout = 180000)]
    [InlineData(1)]
    [InlineData(2)]
    public async Task Resume_WithDropout_DrawsTheUninterruptedRunsMasks(int worldSize)
    {
        var (inputs, targets) = CreateRegressionDataset(ResumeSamples, seed: 712);
        FeedForwardNeuralNetwork<float> Create(AdamOptimizer<float, Tensor<float>, Tensor<float>> opt) => CreateNet(opt, DropoutMlpLayers);
        var init = InitFor(() => Create(CreateAdamOptimizer()), seed: 713);
        string dir = Path.Combine(_tempRoot, $"dropout_resume_{worldSize}");

        var straight = await TrainRanksAsync(worldSize, Create, init, inputs, targets, 8181, 6, dir, 3, resume: false);
        var expected = straight[0].GetParameters();
        if (worldSize == 2)
        {
            var streams = straight.Select(m => m.CollectStochasticLayers().Select(l => l.StochasticStream).Distinct().Single()).ToArray();
            Assert.Equal(new[] { 0, 1 }, streams);
            Assert.True(MaxAbsDiff(straight[0].GetParameters(), straight[1].GetParameters()) < 1e-6);
        }

        DeleteCheckpointsAfter(dir, 3);
        var scrambled = new Vector<float>(Enumerable.Repeat(0.25f, init.Length).ToArray());
        var resumed = await TrainRanksAsync(worldSize, Create, scrambled, inputs, targets, 8181, 6, dir, 100, resume: true);
        double diff = MaxAbsDiff(expected, resumed[0].GetParameters());
        Assert.True(diff < 1e-5, $"{worldSize}-rank dropout resume diverged from the uninterrupted run by {diff}.");
    }

    /// <summary>
    /// G6: the stream a rank is given is the stream its dropout masks come from. Two layers with the same seed, stream
    /// and counter draw the same sequence of masks; changing only the stream changes the masks drawn.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task Dropout_SameSeedStreamAndCounter_DrawsTheSameMasks_AnotherStreamDrawsOthers()
    {
        await Task.CompletedTask;
        const int Draws = 4;
        var input = new Tensor<float>(new[] { 1, 64 }, new Vector<float>(Enumerable.Repeat(1.5f, 64).ToArray()));

        float[][] Sample(int stream)
        {
            var layer = new DropoutLayer<float>(0.5) { RandomSeed = 4242 };
            layer.StochasticStream = stream;
            layer.StochasticDrawCounter = 3;
            layer.SetTrainingMode(true);
            return Enumerable.Range(0, Draws).Select(_ => layer.Forward(input).ToArray()).ToArray();
        }

        var first = Sample(1);
        var again = Sample(1);
        var other = Sample(2);
        for (int d = 0; d < Draws; d++)
        {
            Assert.Equal(first[d], again[d]);
            Assert.Contains(first[d], v => v == 0f);
            Assert.Contains(first[d], v => v != 0f);
        }

        // Every draw of the other stream differs from this stream's draw at the same counter.
        for (int d = 0; d < Draws; d++)
            Assert.NotEqual(first[d], other[d]);
    }

    /// <summary>G7: rank 0's buffers (batch-norm running statistics) replace every other rank's when training starts.</summary>
    [Fact(Timeout = 120000)]
    public async Task InitialSync_BroadcastsRankZerosBuffers()
    {
        // Every sample identical, so both ranks see the same batch statistics and their running statistics after a
        // step differ only by what they started from.
        var (inputs, targets) = CreateRegressionDataset(16, seed: 812);
        for (int i = 1; i < inputs.Length; i++) { inputs[i] = inputs[0]; targets[i] = targets[0]; }
        List<ILayer<float>> BnLayers() => new()
        {
            new InputLayer<float>(InDim),
            new DenseLayer<float>(outputSize: HiddenDim, activationFunction: new ReLUActivation<float>()),
            new BatchNormalizationLayer<float>(HiddenDim),
            new DenseLayer<float>(outputSize: OutDim, activationFunction: new IdentityActivation<float>()),
        };
        FeedForwardNeuralNetwork<float> Create(AdamOptimizer<float, Tensor<float>, Tensor<float>> opt) => CreateNet(opt, BnLayers);
        var init = InitFor(() => Create(CreateAdamOptimizer()), seed: 813);

        var models = await TrainRanksAsync(2, Create, init, inputs, targets, 9191, 1, null, 100, resume: false,
            beforeBuild: (rank, model) =>
            {
                if (rank == 0) return;
                var bn = model.Layers.OfType<BatchNormalizationLayer<float>>().Single();
                var mean = bn.GetRunningMean().AsWritableSpan();
                for (int i = 0; i < mean.Length; i++) mean[i] = 5f + i;
            });

        var m0 = models[0].Layers.OfType<BatchNormalizationLayer<float>>().Single().GetRunningMean().AsSpan().ToArray();
        var m1 = models[1].Layers.OfType<BatchNormalizationLayer<float>>().Single().GetRunningMean().AsSpan().ToArray();
        Assert.Equal(m0, m1);
    }

    /// <summary>
    /// G5: with ignored targets spread unevenly over the ranks, 2 ranks x B/2 match 1 rank x B: each rank's gradient is
    /// weighted by its supervised targets over the global count, as cross-entropy's mean over non-ignored targets is.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task IgnoredTargets_WeightRanksByTheirSupervisedCount_MatchingOneRankOverTheGlobalBatch()
    {
        var rng = RandomHelper.CreateSeededRandom(912);
        var inputs = new Tensor<float>[24];
        var targets = new Tensor<float>[24];
        for (int i = 0; i < inputs.Length; i++)
        {
            var x = new float[InDim];
            for (int d = 0; d < InDim; d++) x[d] = (float)(rng.NextDouble() * 2 - 1);
            inputs[i] = new Tensor<float>(new[] { InDim }, new Vector<float>(x));
            // About half the targets are the ignore sentinel, so the ranks' supervised counts differ from step to step.
            float label = rng.NextDouble() < 0.5 ? -1f : rng.Next(0, OutDim);
            targets[i] = new Tensor<float>(new[] { 1 }, new Vector<float>(new[] { label }));
        }
        FeedForwardNeuralNetwork<float> Create(AdamOptimizer<float, Tensor<float>, Tensor<float>> opt) =>
            CreateNet(opt, () => new List<ILayer<float>>
            {
                new InputLayer<float>(InDim),
                new DenseLayer<float>(outputSize: HiddenDim, activationFunction: new ReLUActivation<float>()),
                new DenseLayer<float>(outputSize: OutDim, activationFunction: new IdentityActivation<float>()),
            }, new CrossEntropyWithLogitsLoss<float>(), NeuralNetworkTaskType.MultiClassClassification);
        var init = InitFor(() => Create(CreateAdamOptimizer()), seed: 913);

        var single = await TrainRanksAsync(1, Create, init, inputs, targets, 9292, 3, null, 100, resume: false);
        var ranks = await TrainRanksAsync(2, Create, init, inputs, targets, 9292, 3, null, 100, resume: false);
        Assert.True(MaxAbsDiff(init, single[0].GetParameters()) > 1e-4, "Training must move the weights.");
        double diff = MaxAbsDiff(single[0].GetParameters(), ranks[0].GetParameters());
        Assert.True(diff < 1e-5, $"2 ranks x B/2 with ignored targets diverged from 1 rank x B by {diff}.");
    }
}
