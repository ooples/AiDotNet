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
/// Chan et al. 2022, RealBasicVSR's adversarial stage, against the authors' release (open-mmlab/mmagic RealBasicVSR,
/// UNetDiscriminatorWithSpectralNorm, PerceptualLoss and
/// configs/real_basicvsr/realbasicvsr_c64b20-1x30x8_8xb1-lr5e-5-150k_reds.py): the discriminator's layout and spectral
/// normalization, the VGG19 perceptual loss, and the alternating generator/discriminator steps (#2397).
/// </summary>
public class RealBasicVSRSharpPaperTests
{
    private const int Size = 8;

    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(inputType: InputType.FourDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputFrames: 2, inputDepth: 3, inputHeight: Size, inputWidth: Size, outputSize: 3);

    private static RealBasicVSRSharp<double> Small(double perceptual = 1.0, double? ema = null) => new(Architecture(), new RealBasicVSRSharpOptions
    {
        Seed = 1, MidChannels = 8, NumPropagationBlocks = 1, NumCleaningBlocks = 1, DiscriminatorChannels = 8, NumFrames = 2,
        PerceptualWeight = perceptual, EmaMomentum = ema, SharpenTarget = false, LearningRate = 1e-3, DiscriminatorLearningRate = 1e-3,
    });

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTheSecondStageRecipe()
    {
        await Task.Yield();
        var o = new RealBasicVSRSharpOptions();
        Assert.Equal((64, 20, 20, 64, 15), (o.MidChannels, o.NumPropagationBlocks, o.NumCleaningBlocks, o.DiscriminatorChannels, o.NumFrames));
        Assert.Equal((5e-5, 1e-4), (o.LearningRate, o.DiscriminatorLearningRate));
        Assert.Equal((1.0, 1.0, 1.0, 5e-2), (o.PixelLossWeight, o.CleaningLossWeight, o.PerceptualWeight, o.GANWeight));
        Assert.Equal((true, 5.0), (o.SharpenTarget, o.InferenceCleaningThreshold));
        Assert.Equal(0.001, o.EmaMomentum);
    }

    [Fact(Timeout = 120000)]
    public async Task Discriminator_HasTheReleaseLayout_AndOneLogitPerPixel()
    {
        await Task.Yield();
        var model = Small(perceptual: 0);
        var images = Images(2, 32);
        var logits = model.DiscriminateForTest(images);
        Assert.Equal(new[] { 2, 1, 32, 32 }, logits.Shape.ToArray());
        // conv0 (w, b), conv1..conv8 bias-free, conv9 (w, b): 12 tensors; with m = 8 the weights are
        // 3·8·9+8 + 8·16·16 + 16·32·16 + 32·64·16 + 64·32·9 + 32·16·9 + 16·8·9 + 8·8·9·2 + 8·9+1.
        var parameters = model.DiscriminatorParameters;
        Assert.Equal(12, parameters.Count);
        int expected = (3 * 8 * 9 + 8) + 8 * 16 * 16 + 16 * 32 * 16 + 32 * 64 * 16 + 64 * 32 * 9 + 32 * 16 * 9 + 16 * 8 * 9
            + 2 * 8 * 8 * 9 + (8 * 9 + 1);
        Assert.Equal(expected, parameters.Sum(p => p.Length));
    }

    [Fact(Timeout = 120000)]
    public async Task SpectralNorm_MakesTheDiscriminatorInvariantToKernelScale()
    {
        await Task.Yield();
        // W / σ(W) is unchanged when W is scaled, so scaling every normalized kernel leaves the logits unchanged
        // (the bias-free normalized convs are conv1..conv8, parameters 2..9).
        var model = Small(perceptual: 0);
        var images = Images(1, 16);
        var before = model.DiscriminateForTest(images).ToArray();
        var normalized = model.DiscriminatorParameters.Skip(2).Take(8).ToList();
        foreach (var kernel in normalized)
            for (int i = 0; i < kernel.Length; i++) kernel[i] *= 7.0;
        // Evaluation reuses the kept u and v, so σ = uᵀWv scales exactly with W.
        var after = model.DiscriminateForTest(images).ToArray();
        // Before any training step u and v are still torch's random draws, so σ is far below the top singular value
        // and the logits are large; the comparison is relative to their size.
        double scale = before.Max(Math.Abs);
        Assert.True(before.Zip(after, (b, a) => Math.Abs(b - a)).Max() <= 1e-9 * scale, "spectral norm must remove the scale");
    }

    [Fact(Timeout = 300000)]
    public async Task PerceptualLoss_IsZeroForIdenticalImages_AndPositiveOtherwise()
    {
        await Task.Yield();
        var model = Small();
        var a = Images(1, 32);
        var b = Images(1, 32, phase: 1.3);
        Assert.Equal(0.0, model.PerceptualLossForTest(a, a)[0], 12);
        Assert.True(model.PerceptualLossForTest(a, b)[0] > 0);
        // VGG19 through conv5_4: 16 convs with weights and biases.
        Assert.Equal(32, model.PerceptualParameters.Count);
        Assert.Empty(Small(perceptual: 0).PerceptualParameters);
    }

    [Fact(Timeout = 600000)]
    public async Task Training_StepsBothNetworks_AndFreezesSpyNetAndVgg()
    {
        await Task.Yield();
        var (low, high) = Pair();
        var model = Small();
        model.Train(low, high);
        var discriminator = model.DiscriminatorParameters.Select(p => p.ToArray()).ToList();
        var flow = model.FlowParameters.Select(p => p.ToArray()).ToList();
        var vgg = model.PerceptualParameters.Select(p => p.ToArray()).ToList();
        var output = model.Predict(low).ToArray();
        model.Train(low, high);
        Assert.False(discriminator.Zip(model.DiscriminatorParameters, (b, a) => b.SequenceEqual(a.ToArray())).All(x => x), "the discriminator must step");
        Assert.False(output.SequenceEqual(model.Predict(low).ToArray()), "the generator must step");
        Assert.True(flow.Zip(model.FlowParameters, (b, a) => b.SequenceEqual(a.ToArray())).All(x => x), "SPyNet must stay frozen");
        Assert.True(vgg.Zip(model.PerceptualParameters, (b, a) => b.SequenceEqual(a.ToArray())).All(x => x), "VGG19 must stay frozen");
    }

    [Fact(Timeout = 600000)]
    public async Task Training_ReducesTheError()
    {
        await Task.Yield();
        var (low, high) = Pair();
        var model = Small(perceptual: 0);
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

    private static Tensor<double> Images(int count, int size, double phase = 0)
    {
        var images = new Tensor<double>(new[] { count, 3, size, size });
        for (int i = 0; i < images.Length; i++) images[i] = 0.5 + 0.4 * Math.Sin(i * 0.29 + phase);
        return images;
    }

    // A smooth 32x32 clip and its 4x box-downsampled 8x8 version, pixel values 0-255.
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
