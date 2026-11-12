using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Enhancement;
using AiDotNet.Video.Options;
using Xunit;

namespace AiDotNet.Tests.Video;

/// <summary>
/// Chan et al. 2021, BasicVSR, against the authors' release (open-mmlab/mmagic BasicVSRNet,
/// configs/basicvsr/basicvsr_2xb4_reds4.py): SPyNet, flow warping, bidirectional propagation, x4 reconstruction and
/// the training recipe (#2397).
/// </summary>
public class BasicVSRPaperTests
{
    private const int Size = 16;

    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(inputType: InputType.FourDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputFrames: 3, inputDepth: 3, inputHeight: Size, inputWidth: Size, outputSize: 3);

    private static BasicVSR<double> Small(int freeze = 5000, double learningRate = 2e-4) => new(Architecture(), new BasicVSROptions
    {
        Seed = 1, MidChannels = 8, NumResBlocks = 1, NumFrames = 3, FlowFreezeIterations = freeze, LearningRate = learningRate,
    });

    private static Tensor<double> Clip(int frames)
    {
        var clip = new Tensor<double>(new[] { frames, 3, Size, Size });
        for (int i = 0; i < clip.Length; i++) clip[i] = 0.5 + 0.3 * Math.Sin(i * 0.37);
        return clip;
    }

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTheRedsRecipe()
    {
        await Task.Yield();
        var o = new BasicVSROptions();
        Assert.Equal((64, 30, 15), (o.MidChannels, o.NumResBlocks, o.NumFrames));
        Assert.Equal((2e-4, 0.125, 5000), (o.LearningRate, o.SpyNetLearningRateMultiplier, o.FlowFreezeIterations));
        Assert.Equal((1e-7, 300000, 1e-6), (o.MinLearningRate, o.CosineIterations, o.CharbonnierEpsilon));
    }

    [Fact(Timeout = 60000)]
    public async Task FlowWarp_SamplesAtThePixelPlusItsFlow()
    {
        await Task.Yield();
        var model = Small();
        var image = new Tensor<double>(new[] { 1, 1, 4, 5 });
        for (int i = 0; i < image.Length; i++) image[i] = i;
        var flow = new Tensor<double>(new[] { 1, 2, 4, 5 });
        for (int i = 0; i < 20; i++) flow[i] = 1.0; // dx = +1, dy = 0
        var warped = model.WarpForTest(image, flow);
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 5; x++)
                Assert.Equal(x < 4 ? image[y * 5 + x + 1] : 0.0, warped[y * 5 + x], 10); // zero padding past the edge
    }

    [Fact(Timeout = 60000)]
    public async Task SpyNet_IsSixLevelsOfFiveConvolutions_AtAnyInputSize()
    {
        await Task.Yield();
        var model = Small();
        var a = new Tensor<double>(new[] { 2, 3, 20, 24 });
        for (int i = 0; i < a.Length; i++) a[i] = 0.5 + 0.2 * Math.Cos(i * 0.11);
        var flow = model.FlowForTest(a, a);
        Assert.Equal(new[] { 2, 2, 20, 24 }, flow.Shape.ToArray());
        // 30 convolutions, each a weight and a bias; widths 8-32-64-32-16-2 with 7x7 kernels.
        var parameters = model.FlowParameters;
        Assert.Equal(60, parameters.Count);
        long weights = 6L * 49 * (8 * 32 + 32 * 64 + 64 * 32 + 32 * 16 + 16 * 2) + 6L * (32 + 64 + 32 + 16 + 2);
        Assert.Equal(weights, parameters.Sum(p => (long)p.Length));
    }

    [Fact(Timeout = 120000)]
    public async Task Propagation_CarriesInformationBothWays_AndUpscalesFourTimes()
    {
        await Task.Yield();
        var model = Small();
        var clip = Clip(3);
        var baseline = model.UpscaleNormalizedForTest(clip);
        Assert.Equal(new[] { 3, 3, 4 * Size, 4 * Size }, baseline.Shape.ToArray());
        int frame = 3 * Size * Size, outFrame = 3 * 16 * Size * Size;

        var lastChanged = clip.Clone();
        for (int i = 2 * frame; i < 3 * frame; i++) lastChanged[i] += 0.2;
        var backward = model.UpscaleNormalizedForTest(lastChanged);
        Assert.True(Enumerable.Range(0, outFrame).Max(i => Math.Abs(backward[i] - baseline[i])) > 1e-9,
            "frame 0 must hear about frame 2 through the backward branch");

        var firstChanged = clip.Clone();
        for (int i = 0; i < frame; i++) firstChanged[i] += 0.2;
        var forward = model.UpscaleNormalizedForTest(firstChanged);
        Assert.True(Enumerable.Range(2 * outFrame, outFrame).Max(i => Math.Abs(forward[i] - baseline[i])) > 1e-9,
            "frame 2 must hear about frame 0 through the forward branch");
    }

    [Fact(Timeout = 300000)]
    public async Task SpyNet_StaysFrozenForTheFixIterations()
    {
        await Task.Yield();
        var (low, high) = Pair();
        var frozen = Small(freeze: 100, learningRate: 1e-3);
        // SPyNet's convolutions are lazily shaped: snapshot the weights after a forward has created them.
        frozen.Predict(low);
        var before = frozen.FlowParameters.Select(p => p.ToArray()).ToList();
        for (int step = 0; step < 2; step++) frozen.Train(low, high);
        var after = frozen.FlowParameters.Select(p => p.ToArray()).ToList();
        Assert.True(before.Zip(after, (b, a) => b.SequenceEqual(a)).All(x => x), "SPyNet moved inside its freeze window");

        var released = Small(freeze: 0, learningRate: 1e-3);
        released.Predict(low);
        before = released.FlowParameters.Select(p => p.ToArray()).ToList();
        for (int step = 0; step < 2; step++) released.Train(low, high);
        after = released.FlowParameters.Select(p => p.ToArray()).ToList();
        Assert.False(before.Zip(after, (b, a) => b.SequenceEqual(a)).All(x => x), "SPyNet did not train after its freeze window");
    }

    [Fact(Timeout = 300000)]
    public async Task Training_ReducesTheError()
    {
        await Task.Yield();
        var (low, high) = Pair();
        var model = Small(learningRate: 1e-3);
        double Error()
        {
            var output = model.Predict(low);
            return Enumerable.Range(0, output.Length).Average(i => Math.Abs(output[i] - high[i]));
        }

        double before = Error();
        for (int step = 0; step < 30; step++) model.Train(low, high);
        double after = Error();
        Assert.True(after < 0.7 * before, $"L1 {before:F2} -> {after:F2}");
    }

    // A smooth 64x64 clip and its 4x box-downsampled 16x16 version, pixel values 0-255.
    private static (Tensor<double> Low, Tensor<double> High) Pair()
    {
        int big = 4 * Size;
        var high = new Tensor<double>(new[] { 3, 3, big, big });
        for (int f = 0; f < 3; f++)
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < big; y++)
                    for (int x = 0; x < big; x++)
                        high[((f * 3 + c) * big + y) * big + x] = 128 + 80 * Math.Sin((x + 2 * f) * 0.21 + c) * Math.Cos(y * 0.17);
        var low = new Tensor<double>(new[] { 3, 3, Size, Size });
        for (int f = 0; f < 3; f++)
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < Size; y++)
                    for (int x = 0; x < Size; x++)
                    {
                        double s = 0;
                        for (int dy = 0; dy < 4; dy++)
                            for (int dx = 0; dx < 4; dx++) s += high[((f * 3 + c) * big + 4 * y + dy) * big + 4 * x + dx];
                        low[((f * 3 + c) * Size + y) * Size + x] = s / 16;
                    }
        return (low, high);
    }
}
