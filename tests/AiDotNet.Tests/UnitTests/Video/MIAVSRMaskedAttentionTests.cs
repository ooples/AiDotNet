using System;
using System.Linq;
using AiDotNet.Enums;
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
        // The flow estimator aligns windows but must never train: PSRT's patch moves are discrete, so it
        // receives no gradient, and it sits outside the model's parameters so no optimizer touches it.
        var flow = new SpyNetLayer<double>(numLevels: 3);
        var model = CreateModel(flow: flow);
        var clip = CreateClip(9);
        model.Predict(clip);
        var before = flow.GetParameters().ToArray();
        Assert.NotEmpty(before);
        int modelParameters = model.GetParameters().Length;

        var target = new Tensor<double>(new[] { 1, Frames, Channels, Side * 2, Side * 2 });
        model.Train(clip, target);

        Assert.Equal(before, flow.GetParameters().ToArray());
        Assert.Equal(modelParameters, CreateModel().GetParameters().Length);
        var output = model.Predict(clip);
        for (int i = 0; i < output.Length; i++) Assert.False(double.IsNaN(output[i]) || double.IsInfinity(output[i]));
    }

    private static double[] MaskScorerWeights(MIAVSR<double> model)
        => model.Layers.OfType<DenseLayer<double>>()
            .Where(layer => layer.GetOutputShape().LastOrDefault() == 1)
            .SelectMany(layer => Enumerable.Range(0, layer.GetParameters().Length).Select(i => layer.GetParameters()[i]))
            .ToArray();
}
