using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Enhancement;
using AiDotNet.Video.Options;
using Xunit;

namespace AiDotNet.Tests.Video;

/// <summary>
/// Chan et al. 2021, IconVSR, against the authors' release (open-mmlab/mmagic IconVSRNet, EDVRFeatureExtractor,
/// PCDAlignment, TSAFusion, ModulatedDCNPack and configs/iconvsr/iconvsr_2xb4_reds4.py): keyframes, the refill
/// padding, the zero-initialized deformable alignment, spatial padding, coupled propagation and the freeze (#2397).
/// </summary>
public class IconVSRPaperTests
{
    private const int Size = 8;

    private static NeuralNetworkArchitecture<double> Architecture(int size = Size) =>
        new(inputType: InputType.FourDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputFrames: 3, inputDepth: 3, inputHeight: size, inputWidth: size, outputSize: 3);

    private static IconVSR<double> Small(int freeze = 5000, double learningRate = 2e-4, int size = Size) => new(Architecture(size), new IconVSROptions
    {
        Seed = 1, MidChannels = 8, NumResBlocks = 1, NumFrames = 3, EdvrChannels = 8, DeformableGroups = 2,
        EdvrExtractionBlocks = 1, KeyframeStride = 2, FlowFreezeIterations = freeze, LearningRate = learningRate,
    });

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTheReleaseRecipe()
    {
        await Task.Yield();
        var o = new IconVSROptions();
        Assert.Equal((64, 30, 15, 5, 2), (o.MidChannels, o.NumResBlocks, o.NumFrames, o.KeyframeStride, o.RefillPadding));
        Assert.Equal((64, 8, 5), (o.EdvrChannels, o.DeformableGroups, o.EdvrExtractionBlocks));
        Assert.Equal((2e-4, 0.125, 5000, 1e-7, 300000), (o.LearningRate, o.SpyNetLearningRateMultiplier, o.FlowFreezeIterations, o.MinLearningRate, o.CosineIterations));
    }

    [Fact(Timeout = 60000)]
    public async Task Keyframes_AreEveryStrideAndTheLastFrame()
    {
        await Task.Yield();
        Assert.Equal(new[] { 0, 5, 10, 14 }, IconVsrBackbone<double>.Keyframes(15, 5));
        Assert.Equal(new[] { 0, 5, 10 }, IconVsrBackbone<double>.Keyframes(11, 5));
    }

    [Fact(Timeout = 60000)]
    public async Task RefillPadding_IsTheReleasesFrameOrder()
    {
        await Task.Yield();
        // padding 2: lrs[:, [4, 3]], lrs, lrs[:, [-4, -5]]; padding 3: [6, 5, 4] and [-5, -6, -7].
        Assert.Equal(new[] { 4, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 3 }, IconVsrBackbone<double>.RefillIndices(8, 2));
        Assert.Equal(new[] { 6, 5, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 5, 4, 3 }, IconVsrBackbone<double>.RefillIndices(10, 3));
    }

    [Fact(Timeout = 60000)]
    public async Task ModulatedDcnPack_StartsAsHalfAPlainConvolution()
    {
        await Task.Yield();
        // conv_offset is zero-initialized: zero offsets and sigmoid(0) = 0.5 masks, so the pack is 0.5·conv(x) + bias.
        var engine = AiDotNetEngine.Current;
        var layers = new List<ILayer<double>>();
        var pack = new ModulatedDcnPack<double>(layers, 4, 2);
        var x = new Tensor<double>(new[] { 1, 4, 6, 6 });
        var guide = new Tensor<double>(new[] { 1, 4, 6, 6 });
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = Math.Sin(i * 0.31);
            guide[i] = Math.Cos(i * 0.17);
        }
        var y = pack.Forward(engine, layers, x, guide);
        var main = (AiDotNet.NeuralNetworks.Layers.ConvolutionalLayer<double>)layers[1];
        var conv = engine.Conv2D(x, main.KernelsForFolding, 1, 1, 1);
        Assert.Equal(new[] { 1, 4, 6, 6 }, y.Shape.ToArray());
        for (int i = 0; i < y.Length; i++) Assert.Equal(0.5 * conv[i], y[i], 10);
    }

    [Fact(Timeout = 60000)]
    public async Task SpatialPadding_ReflectsToMultiplesOfFour()
    {
        await Task.Yield();
        var frames = new Tensor<double>(new[] { 1, 1, 2, 3 });
        for (int i = 0; i < 6; i++) frames[i] = i;
        var padded = IconVsrBackbone<double>.ReflectPadTo4(frames);
        Assert.Equal(new[] { 1, 1, 4, 4 }, padded.Shape.ToArray());
        // F.pad(mode='reflect') rows 0 1 | 0 1, columns 0 1 2 | 1.
        Assert.Equal(new double[] { 0, 1, 2, 1, 3, 4, 5, 4, 0, 1, 2, 1, 3, 4, 5, 4 }, padded.ToArray());
    }

    [Fact(Timeout = 300000)]
    public async Task Output_IsFourTimesTheInput_AndCropsTheSpatialPadding()
    {
        await Task.Yield();
        var model = Small(size: 10);
        var clip = new Tensor<double>(new[] { 3, 3, 10, 10 });
        for (int i = 0; i < clip.Length; i++) clip[i] = 0.5 + 0.3 * Math.Sin(i * 0.37);
        var output = model.UpscaleNormalizedForTest(clip);
        Assert.Equal(new[] { 3, 3, 40, 40 }, output.Shape.ToArray());
        Assert.All(output.ToArray(), v => Assert.False(double.IsNaN(v)));
    }

    [Fact(Timeout = 300000)]
    public async Task Propagation_RunsBothWays()
    {
        await Task.Yield();
        var model = Small();
        var clip = new Tensor<double>(new[] { 3, 3, Size, Size });
        for (int i = 0; i < clip.Length; i++) clip[i] = 0.5 + 0.3 * Math.Sin(i * 0.37);
        var baseline = model.UpscaleNormalizedForTest(clip);
        int frame = 3 * Size * Size, outFrame = 3 * 16 * Size * Size;

        var lastChanged = clip.Clone();
        for (int i = 2 * frame; i < 3 * frame; i++) lastChanged[i] += 0.2;
        var backward = model.UpscaleNormalizedForTest(lastChanged);
        Assert.True(Enumerable.Range(0, outFrame).Max(i => Math.Abs(backward[i] - baseline[i])) > 1e-9, "frame 0 must hear about frame 2");

        var firstChanged = clip.Clone();
        for (int i = 0; i < frame; i++) firstChanged[i] += 0.2;
        var forward = model.UpscaleNormalizedForTest(firstChanged);
        Assert.True(Enumerable.Range(2 * outFrame, outFrame).Max(i => Math.Abs(forward[i] - baseline[i])) > 1e-9, "frame 2 must hear about frame 0");
    }

    [Fact(Timeout = 600000)]
    public async Task SpyNetAndEdvr_StayFrozenForTheFixIterations()
    {
        await Task.Yield();
        var (low, high) = Pair();
        var model = Small(freeze: 100, learningRate: 1e-3);
        // The convolutions are lazily shaped: snapshot the weights after a forward has created them.
        model.Predict(low);
        var flow = model.FlowParameters.Select(p => p.ToArray()).ToList();
        var edvr = model.EdvrParameters.Select(p => p.ToArray()).ToList();
        var output = model.Predict(low).ToArray();
        model.Train(low, high);
        Assert.True(flow.Zip(model.FlowParameters, (b, a) => b.SequenceEqual(a.ToArray())).All(x => x), "SPyNet moved inside its freeze window");
        Assert.True(edvr.Zip(model.EdvrParameters, (b, a) => b.SequenceEqual(a.ToArray())).All(x => x), "EDVR moved inside its freeze window");
        Assert.False(output.SequenceEqual(model.Predict(low).ToArray()), "the rest of the network must train");
    }

    [Fact(Timeout = 600000)]
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

    // A smooth 32x32 clip and its 4x box-downsampled 8x8 version, pixel values 0-255.
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
