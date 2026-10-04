using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Enhancement;
using AiDotNet.Video.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Video;

/// <summary>
/// MIA-VSR's paper-specific behaviour (#1761): sparse inference over the learned keep-mask, and the
/// mask-sparsity loss reaching the gradient through the base auxiliary tape-loss hook.
/// </summary>
public class MIAVSRMaskedAttentionTests
{
    private const int Frames = 3, Channels = 3, Side = 8;

    private static MIAVSR<double> CreateModel(double maskLossWeight = 5e-4, SpyNetLayer<double>? flow = null) => new(
        new NeuralNetworkArchitecture<double>(
            inputType: InputType.FourDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputFrames: Frames, inputDepth: Channels, inputHeight: Side, inputWidth: Side,
            outputSize: 4),
        new MIAVSROptions
        {
            NumFeatures = 8, NumHeads = 2, WindowSize = 4, FeedForwardRatio = 2,
            NumPropagationBranches = 4, BlocksPerBranch = 1, ScaleFactor = 2,
            ReconstructionChannels = 8, MaskLossWeight = maskLossWeight, Seed = 1234
        }, flowEstimator: flow);

    private static Tensor<double> CreateClip(int seed)
    {
        var rng = new Random(seed);
        var clip = new Tensor<double>(new[] { 1, Frames, Channels, Side, Side });
        for (int i = 0; i < clip.Length; i++) clip[i] = rng.NextDouble();
        return clip;
    }

    [Fact]
    public void Output_UpscalesEveryFrame()
    {
        var model = CreateModel();
        var output = model.Predict(CreateClip(1));
        Assert.Equal(new[] { 1, Frames, Channels, Side * 2, Side * 2 }, output.Shape.ToArray());
        for (int i = 0; i < output.Length; i++) Assert.False(double.IsNaN(output[i]) || double.IsInfinity(output[i]));
    }

    [Fact]
    public void SparseInference_MatchesDenseMaskedBlend()
    {
        // Only kept positions are computed sparsely; dense inference computes all of them and blends.
        // The two must agree exactly, or the sparse gather/merge is wrong.
        var model = CreateModel();
        var network = Assert.IsType<MiaVsrNetwork<double>>(model.Network);
        var clip = CreateClip(2);

        network.DenseInference = false;
        var sparse = model.Predict(clip);
        network.DenseInference = true;
        var dense = model.Predict(clip);

        Assert.Equal(dense.Length, sparse.Length);
        for (int i = 0; i < dense.Length; i++) Assert.Equal(dense[i], sparse[i], 10);
    }

    [Fact]
    public void MaskLoss_ReachesTheMaskScorerGradient()
    {
        // Two identically seeded models differ only in λ. If the sparsity term reached the gradient,
        // one training step moves the mask scorers differently; if it were dropped, identically.
        var withLoss = CreateModel(maskLossWeight: 1.0);
        var withoutLoss = CreateModel(maskLossWeight: 0.0);
        var clip = CreateClip(3);
        var target = new Tensor<double>(new[] { 1, Frames, Channels, Side * 2, Side * 2 });
        var rng = new Random(4);
        for (int i = 0; i < target.Length; i++) target[i] = rng.NextDouble();

        withLoss.Predict(clip);
        withoutLoss.Predict(clip);
        // Initialization seeds come from the architecture, not these options; give both models the same
        // starting weights explicitly so lambda is the only difference.
        withoutLoss.SetParameters(withLoss.GetParameters());
        var before = MaskScorerWeights(withLoss);
        Assert.NotEmpty(before);
        Assert.Equal(before, MaskScorerWeights(withoutLoss));

        withLoss.Train(clip, target);
        withoutLoss.Train(clip, target);

        var deltaWith = MaskScorerWeights(withLoss).Zip(before, (a, b) => a - b).ToArray();
        var deltaWithout = MaskScorerWeights(withoutLoss).Zip(before, (a, b) => a - b).ToArray();
        double difference = deltaWith.Zip(deltaWithout, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(difference > 1e-9, $"The mask-sparsity loss did not change the mask scorers' update (max difference {difference}).");
    }

    [Fact]
    public void PretrainedFlowEstimator_IsUsedFrozen()
    {
        // A supplied pretrained estimator aligns windows but stays frozen by default, as in PSRT: its flow is
        // computed outside the tape, so it receives no gradient. It is the model's flow layer, so it is
        // serialized and cloned with the model (and can be opted into fine-tuning).
        var flow = new SpyNetLayer<double>(numLevels: 3);
        var model = CreateModel(flow: flow);
        var clip = CreateClip(9);
        model.Predict(clip);
        var before = flow.GetParameters().ToArray();
        Assert.NotEmpty(before);


        var target = new Tensor<double>(new[] { 1, Frames, Channels, Side * 2, Side * 2 });
        model.Train(clip, target);

        Assert.Equal(before, flow.GetParameters().ToArray());
        Assert.Same(flow, (model.Network ?? throw new InvalidOperationException("No native network.")).FlowEstimator);
        Assert.Contains(flow, model.Layers);
        var output = model.Predict(clip);
        for (int i = 0; i < output.Length; i++) Assert.False(double.IsNaN(output[i]) || double.IsInfinity(output[i]));
    }

    private static double[] MaskScorerWeights(MIAVSR<double> model)
        => model.Layers.OfType<DenseLayer<double>>()
            .Where(layer => layer.GetOutputShape().LastOrDefault() == 1)
            .SelectMany(layer => Enumerable.Range(0, layer.GetParameters().Length).Select(i => layer.GetParameters()[i]))
            .ToArray();

    [Theory]
    [InlineData(-1e-4)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void InvalidMaskLossWeight_IsRejectedAtConstruction(double weight)
        => Assert.Throws<ArgumentOutOfRangeException>(() => CreateModel(maskLossWeight: weight));

    [Fact]
    public void SuppliedLayers_AreBoundToThePaperRoles_NotRunAsAChain()
    {
        var bare = new NeuralNetworkArchitecture<double>(
            inputType: InputType.FourDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputFrames: Frames, inputDepth: Channels, inputHeight: Side, inputWidth: Side, outputSize: 4);
        var supplied = LayerHelper<double>.CreateDefaultMIAVSRLayers(bare, numFeatures: 8, windowSize: 4,
            numHeads: 2, feedForwardRatio: 2, numPropagationBranches: 4, blocksPerBranch: 1, scaleFactor: 2,
            reconstructionChannels: 8).ToList();
        var model = new MIAVSR<double>(
            new NeuralNetworkArchitecture<double>(
                inputType: InputType.FourDimensional, taskType: NeuralNetworkTaskType.Regression,
                inputFrames: Frames, inputDepth: Channels, inputHeight: Side, inputWidth: Side, outputSize: 4,
                layers: supplied),
            new MIAVSROptions
            {
                NumFeatures = 8, NumHeads = 2, WindowSize = 4, FeedForwardRatio = 2,
                NumPropagationBranches = 4, BlocksPerBranch = 1, ScaleFactor = 2,
                ReconstructionChannels = 8, Seed = 1234
            });

        // The model runs the caller's instances in the paper's roles: the clip is upscaled, which a plain
        // sequential walk over the same layers could not do.
        Assert.True(supplied.SequenceEqual(model.Layers));
        var output = model.Predict(CreateClip(5));
        Assert.Equal(new[] { 1, Frames, Channels, Side * 2, Side * 2 }, output.Shape.ToArray());
    }

    [Fact]
    public void SuppliedLayers_ThatDoNotMatchTheLayout_AreRefused()
    {
        var mismatched = new System.Collections.Generic.List<AiDotNet.Interfaces.ILayer<double>>
        {
            new DenseLayer<double>(4, (AiDotNet.Interfaces.IActivationFunction<double>)new AiDotNet.ActivationFunctions.IdentityActivation<double>())
        };
        Assert.Throws<InvalidOperationException>(() => new MIAVSR<double>(
            new NeuralNetworkArchitecture<double>(
                inputType: InputType.FourDimensional, taskType: NeuralNetworkTaskType.Regression,
                inputFrames: Frames, inputDepth: Channels, inputHeight: Side, inputWidth: Side, outputSize: 4,
                layers: mismatched),
            new MIAVSROptions { NumFeatures = 8, NumHeads = 2, WindowSize = 4, BlocksPerBranch = 1, ScaleFactor = 2 }));
    }

    private static MIAVSROptions FlowOptions(bool fineTune = false, int freezeSteps = 5000) => new()
    {
        NumFeatures = 8, NumHeads = 2, WindowSize = 4, FeedForwardRatio = 2,
        NumPropagationBranches = 4, BlocksPerBranch = 1, ScaleFactor = 2,
        ReconstructionChannels = 8, Seed = 1234, FineTuneFlowEstimator = fineTune, FlowFreezeSteps = freezeSteps
    };

    private static MIAVSR<double> CreateModel(MIAVSROptions options, SpyNetLayer<double>? flow) => new(
        new NeuralNetworkArchitecture<double>(
            inputType: InputType.FourDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputFrames: Frames, inputDepth: Channels, inputHeight: Side, inputWidth: Side, outputSize: 4),
        options, flowEstimator: flow);

    private static Tensor<double> HighResTarget(int seed)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var target = new Tensor<double>(new[] { 1, Frames, Channels, Side * 2, Side * 2 });
        for (int i = 0; i < target.Length; i++) target[i] = rng.NextDouble();
        return target;
    }

    private static double[] FlowWeights(MIAVSR<double> model)
        => (model.Network ?? throw new InvalidOperationException("No native network.")).FlowEstimator.GetParameters().ToArray();

    [Fact]
    public void PretrainedFlow_StaysFrozenByDefault()
    {
        var model = CreateModel(FlowOptions(), new SpyNetLayer<double>());
        model.Train(CreateClip(21), HighResTarget(22));
        var before = FlowWeights(model);

        model.Train(CreateClip(23), HighResTarget(24));

        Assert.Equal(before, FlowWeights(model));
    }

    [Fact]
    public void BuiltFlow_TrainsFromTheFirstStep()
    {
        var model = CreateModel(FlowOptions(), flow: null);
        model.Train(CreateClip(31), HighResTarget(32));
        var before = FlowWeights(model);

        model.Train(CreateClip(33), HighResTarget(34));

        var after = FlowWeights(model);
        Assert.All(after, w => Assert.False(double.IsNaN(w) || double.IsInfinity(w)));
        Assert.True(before.Zip(after, (a, b) => a != b).Any(changed => changed),
            "The SPyNet MIA-VSR built for itself did not train: no photometric gradient reached it.");
    }

    [Fact]
    public void FineTunedFlow_IsFrozenForTheWarmUp_ThenTrains()
    {
        var model = CreateModel(FlowOptions(fineTune: true, freezeSteps: 2), new SpyNetLayer<double>());
        model.Train(CreateClip(41), HighResTarget(42));
        var afterFirst = FlowWeights(model);

        model.Train(CreateClip(43), HighResTarget(44));
        Assert.Equal(afterFirst, FlowWeights(model));

        model.Train(CreateClip(45), HighResTarget(46));
        Assert.True(afterFirst.Zip(FlowWeights(model), (a, b) => a != b).Any(changed => changed),
            "After the freeze, the opted-in SPyNet did not fine-tune.");
    }

    [Fact]
    public void MaskLossWeight_ChangedAfterConstruction_IsRefusedWhenTraining()
    {
        var options = FlowOptions();
        var model = CreateModel(options, new SpyNetLayer<double>());
        options.MaskLossWeight = double.NaN;

        Assert.Throws<InvalidOperationException>(() => model.Train(CreateClip(51), HighResTarget(52)));
    }

    [Fact]
    public void FlowEstimator_ResolvedForOtherFrames_IsRefused()
    {
        var flow = new SpyNetLayer<double>();
        var grey = new Tensor<double>(new[] { 1, 1, 16, 16 });
        flow.EstimateFlow(grey, grey);

        Assert.Throws<ArgumentException>(() => CreateModel(FlowOptions(), flow));
    }

    [Fact]
    public void ChunkedTraining_AdvancesTheFineTuningFreeze()
    {
        var model = CreateModel(FlowOptions(fineTune: true, freezeSteps: 1), new SpyNetLayer<double>());
        var clip = new Tensor<double>(new[] { 2, Frames, Channels, Side, Side });
        var target = new Tensor<double>(new[] { 2, Frames, Channels, Side * 2, Side * 2 });
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(61);
        for (int i = 0; i < clip.Length; i++) clip[i] = rng.NextDouble();
        for (int i = 0; i < target.Length; i++) target[i] = rng.NextDouble();

        // Two samples in chunks of one: the base trainer steps the optimizer itself, not through Train.
        model.TrainWithGradientAccumulation(clip, target, batchSize: 1);
        var afterFrozenStep = FlowWeights(model);

        model.Train(CreateClip(62), HighResTarget(63));
        Assert.True(afterFrozenStep.Zip(FlowWeights(model), (a, b) => a != b).Any(changed => changed),
            "The chunked step did not count toward the freeze, so the opted-in SPyNet never started training.");
    }
}