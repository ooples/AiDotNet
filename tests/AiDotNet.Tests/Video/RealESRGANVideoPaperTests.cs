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
/// Wang et al. 2021, Real-ESRGAN, against the authors' release (xinntao/Real-ESRGAN RealESRGAN_x4plus,
/// options/train_realesrgan_x4plus.yml, inference_realesrgan_video.py; BasicSR RRDBNet): the RRDBNet layout, the
/// frame-by-frame video pipeline, and the alternating generator/discriminator steps with a frozen VGG19 (#2397).
/// </summary>
public class RealESRGANVideoPaperTests
{
    private const int Size = 8;

    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(inputType: InputType.FourDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputFrames: 2, inputDepth: 3, inputHeight: Size, inputWidth: Size, outputSize: 3);

    private static RealESRGANVideo<double> Small(double perceptual = 0, double? ema = null) => new(Architecture(), new RealESRGANVideoOptions
    {
        Seed = 1, NumFeatures = 8, NumRRDBBlocks = 1, GrowthChannels = 4, DiscriminatorChannels = 8,
        PerceptualWeight = perceptual, EmaMomentum = ema, SharpenTarget = false, LearningRate = 1e-3, DiscriminatorLearningRate = 1e-3,
    });

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreRealEsrganX4Plus()
    {
        await Task.Yield();
        var o = new RealESRGANVideoOptions();
        Assert.Equal((64, 23, 32, 64), (o.NumFeatures, o.NumRRDBBlocks, o.GrowthChannels, o.DiscriminatorChannels));
        Assert.Equal((1e-4, 1e-4), (o.LearningRate, o.DiscriminatorLearningRate));
        Assert.Equal((1.0, 1.0, 0.1), (o.PixelLossWeight, o.PerceptualWeight, o.GANWeight));
        Assert.Equal((true, 0.5, 51, 10.0), (o.SharpenTarget, o.UnsharpWeight, o.UnsharpKernelSize, o.UnsharpThreshold));
        Assert.Equal(0.001, o.EmaMomentum);
    }

    [Fact(Timeout = 120000)]
    public async Task Generator_IsRrdbNet()
    {
        await Task.Yield();
        var model = Small();
        model.UpscaleNormalizedForTest(new Tensor<double>(new[] { 1, 3, Size, Size }));
        // conv_first, 3 dense blocks x 5 convs, conv_body, conv_up1, conv_up2, conv_hr, conv_last: 21 convs with bias.
        var parameters = model.GeneratorParametersForTest;
        Assert.Equal(42, parameters.Count);
        int f = 8, g = 4;
        int dense = 0;
        for (int c = 0; c < 5; c++) dense += (f + c * g) * (c < 4 ? g : f) * 9 + (c < 4 ? g : f);
        int expected = (3 * f * 9 + f) + 3 * dense + 4 * (f * f * 9 + f) + (f * 3 * 9 + 3);
        Assert.Equal(expected, parameters.Sum(p => p.Length));
    }

    [Fact(Timeout = 120000)]
    public async Task Video_IsUpscaledFrameByFrame()
    {
        await Task.Yield();
        var model = Small();
        var clip = new Tensor<double>(new[] { 2, 3, Size, Size });
        for (int i = 0; i < clip.Length; i++) clip[i] = 0.5 + 0.3 * Math.Sin(i * 0.37);
        var output = model.UpscaleNormalizedForTest(clip);
        Assert.Equal(new[] { 2, 3, 4 * Size, 4 * Size }, output.Shape.ToArray());
        // Changing frame 1 leaves frame 0's output untouched: no temporal path.
        var changed = clip.Clone();
        for (int i = 3 * Size * Size; i < clip.Length; i++) changed[i] += 0.2;
        var other = model.UpscaleNormalizedForTest(changed);
        int outFrame = 3 * 16 * Size * Size;
        Assert.True(Enumerable.Range(0, outFrame).All(i => output[i] == other[i]), "frame 0 must not depend on frame 1");
        Assert.True(Enumerable.Range(outFrame, outFrame).Any(i => output[i] != other[i]));
    }

    [Fact(Timeout = 600000)]
    public async Task Training_StepsBothNetworks_AndFreezesVgg()
    {
        await Task.Yield();
        var (low, high) = Pair();
        var model = Small(perceptual: 1.0);
        model.Train(low, high);
        var discriminator = model.DiscriminatorParameters.Select(p => p.ToArray()).ToList();
        var vgg = model.PerceptualParameters.Select(p => p.ToArray()).ToList();
        var output = model.Predict(low).ToArray();
        model.Train(low, high);
        Assert.Equal(32, model.PerceptualParameters.Count);
        Assert.False(discriminator.Zip(model.DiscriminatorParameters, (b, a) => b.SequenceEqual(a.ToArray())).All(x => x), "the discriminator must step");
        Assert.False(output.SequenceEqual(model.Predict(low).ToArray()), "the generator must step");
        Assert.True(vgg.Zip(model.PerceptualParameters, (b, a) => b.SequenceEqual(a.ToArray())).All(x => x), "VGG19 must stay frozen");
    }

    [Fact(Timeout = 600000)]
    public async Task Training_ReducesTheError()
    {
        await Task.Yield();
        var (low, high) = Pair();
        var model = Small();
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

    [Fact(Timeout = 600000)]
    public async Task Inference_UsesTheMovingAverageOfTheGenerator()
    {
        await Task.Yield();
        var (low, high) = Pair();
        var averaged = Small(ema: 0.001);
        var plain = Small();
        for (int step = 0; step < 3; step++)
        {
            averaged.Train(low, high);
            plain.Train(low, high);
        }
        var a = averaged.Predict(low);
        var p = plain.Predict(low);
        Assert.True(Enumerable.Range(0, a.Length).Max(i => Math.Abs(a[i] - p[i])) > 1e-6, "inference must use the average");
    }

    // A smooth 32x32 two-frame clip and its 4x box-downsampled 8x8 version, pixel values 0-255.
    private static (Tensor<double> Low, Tensor<double> High) Pair()
    {
        int big = 4 * Size;
        var high = new Tensor<double>(new[] { 2, 3, big, big });
        for (int f = 0; f < 2; f++)
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < big; y++)
                    for (int x = 0; x < big; x++)
                        high[((f * 3 + c) * big + y) * big + x] = 128 + 80 * Math.Sin((x + 2 * f) * 0.21 + c) * Math.Cos(y * 0.17);
        var low = new Tensor<double>(new[] { 2, 3, Size, Size });
        for (int f = 0; f < 2; f++)
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
