using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.FrameInterpolation;
using AiDotNet.Video.Options;
using Xunit;

namespace AiDotNet.Tests.Video;

/// <summary>
/// Kalluri et al. 2023, FLAVR (arXiv 2012.08512), and the release (tarun005/FLAVR: model/FLAVR_arch.py,
/// model/resnet_3D.py, config.py, main.py): the configurations and their exact parameter counts, the clip-mean
/// normalization, the pair-as-window contract, the output times and a training step that reaches every weight (#2397).
/// </summary>
public class FLAVRPaperTests
{
    private static IEngine Engine => AiDotNetEngine.Current;

    private static FLAVROptions Small() => new() { Seed = 1, BaseChannels = 4, LayerBlocks = new[] { 1, 1, 1, 1 } };

    private static FLAVR<double> Model(FLAVROptions options) =>
        new(new NeuralNetworkArchitecture<double>(inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 32, inputWidth: 32, inputDepth: 3, outputSize: 3), options);

    private static Tensor<double> Random(int[] shape, int seed, double scale = 1.0)
    {
        var t = new Tensor<double>(shape);
        var random = new System.Random(seed);
        for (int i = 0; i < t.Length; i++) t[i] = scale * random.NextDouble();
        return t;
    }

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreUnet18AndTheReleaseTraining()
    {
        await Task.Yield();
        var o = new FLAVROptions();
        Assert.Equal(64, o.BaseChannels);
        Assert.Equal(new[] { 2, 2, 2, 2 }, o.LayerBlocks);
        Assert.Equal((4, 1), (o.NumInputFrames, o.NumOutputFrames));
        Assert.Equal((2e-4, 0.9, 0.99, 0.5, 5), (o.LearningRate, o.Beta1, o.Beta2, o.PlateauFactor, o.PlateauPatience));
        Assert.Equal(new[] { 3, 4, 6, 3 }, FLAVROptions.Deep().LayerBlocks);
    }

    // UNet_3D_3D counted module by module: unet_18 and unet_34 at 2x, and unet_18 at 8x, where resnet_3D.useBias adds
    // the encoder's biases.
    [Theory(Timeout = 240000)]
    [InlineData(0, 42061571)]
    [InlineData(1, 72926787)]
    [InlineData(2, 42121941)]
    public async Task ParameterCount_MatchesTheRelease(int variant, int expected)
    {
        await Task.Yield();
        var options = variant switch { 1 => FLAVROptions.Deep(), 2 => new FLAVROptions { NumOutputFrames = 7 }, _ => new FLAVROptions() };
        Assert.Equal(expected, Model(options).NetworkParameters.Sum(t => t.Length));
    }

    [Fact(Timeout = 60000)]
    public async Task Padding_IsZerosToMultiplesOf8()
    {
        await Task.Yield();
        Assert.Equal((6, 4), FlavrNetwork<double>.Padding(100, 130));
        Assert.Equal((0, 0), FlavrNetwork<double>.Padding(64, 96));
    }

    // The clip's per-channel mean is subtracted before the encoder and added to every output, so shifting each channel of
    // every frame by a constant shifts the output by the same constant and changes nothing else.
    [Fact(Timeout = 240000)]
    public async Task ClipMean_IsRemovedAndRestored()
    {
        await Task.Yield();
        var model = Model(Small());
        var window = Random(new[] { 12, 20, 28 }, 1, 190.0);
        for (int i = 0; i < window.Length; i++) window[i] += 30.0;
        var shifted = window.Clone();
        double[] delta = { 10, -20, 25 };
        for (int c = 0; c < 12; c++)
            for (int i = 0; i < 20 * 28; i++)
                shifted[c * 20 * 28 + i] += delta[c % 3];
        var a = model.Predict(window);
        var b = model.Predict(shifted);
        Assert.Equal(new[] { 3, 20, 28 }, a.Shape.ToArray());
        for (int i = 0; i < a.Length; i++) Assert.Equal(a[i] + delta[i / (20 * 28)], b[i], 8);
    }

    // A pair is the window with frame 0 over the first half and frame 1 over the second.
    [Fact(Timeout = 240000)]
    public async Task Pair_IsTheWindowWithEachFrameRepeated()
    {
        await Task.Yield();
        var model = Model(Small());
        var frame0 = Random(new[] { 3, 16, 24 }, 2, 255.0);
        var frame1 = Random(new[] { 3, 16, 24 }, 3, 255.0);
        var window = Engine.TensorConcatenate(new[] { frame0, frame0, frame1, frame1 }, 0);
        var expected = model.Predict(window).ToArray();
        Assert.Equal(expected, model.Interpolate(frame0, frame1).ToArray());
        Assert.Equal(expected, model.Predict(Engine.TensorConcatenate(new[] { frame0, frame1 }, 0)).ToArray());
    }

    [Fact(Timeout = 240000)]
    public async Task Interpolate_SynthesizesTheReleaseTimes()
    {
        await Task.Yield();
        var frame0 = Random(new[] { 3, 16, 24 }, 4, 255.0);
        var frame1 = Random(new[] { 3, 16, 24 }, 5, 255.0);
        var twice = Model(Small());
        Assert.Throws<ArgumentOutOfRangeException>(() => twice.Interpolate(frame0, frame1, 0.25));
        var options = Small();
        options.NumOutputFrames = 3;
        var four = Model(options);
        var all = four.Predict(Engine.TensorConcatenate(new[] { frame0, frame0, frame1, frame1 }, 0));
        Assert.Equal(new[] { 9, 16, 24 }, all.Shape.ToArray());
        for (int k = 0; k < 3; k++)
            Assert.Equal(Engine.TensorNarrow(all, 0, 3 * k, 3).ToArray(), four.Interpolate(frame0, frame1, (k + 1) / 4.0).ToArray());
        Assert.Throws<ArgumentOutOfRangeException>(() => four.Interpolate(frame0, frame1, 0.6));
    }

    // Adam moves a weight only on a nonzero gradient: the L1 loss reaches the stem, every stage, the gates, the decoder,
    // the fusion and the output convolution.
    [Fact(Timeout = 240000)]
    public async Task TrainingStep_ReachesEveryWeight()
    {
        await Task.Yield();
        var model = Model(Small());
        var before = model.NetworkParameters.Select(t => t.ToArray()).ToArray();
        model.Train(Random(new[] { 12, 32, 32 }, 6, 255.0), Random(new[] { 3, 32, 32 }, 7, 255.0));
        var after = model.NetworkParameters.Select(t => t.ToArray()).ToArray();
        var frozen = Enumerable.Range(0, before.Length).Where(i => before[i].SequenceEqual(after[i])).ToArray();
        Assert.True(frozen.Length == 0, $"{frozen.Length} of {before.Length} weight tensors received no gradient: {string.Join(", ", frozen)}");
    }

    [Fact(Timeout = 60000)]
    public async Task Options_RejectAnOddWindow()
    {
        await Task.Yield();
        Assert.Throws<ArgumentOutOfRangeException>(() => Model(new FLAVROptions { NumInputFrames = 3 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => Model(new FLAVROptions { LayerBlocks = new[] { 2, 2, 2 } }));
    }
}
