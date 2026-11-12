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
/// Chan et al. 2022, RealBasicVSR, against the authors' release (open-mmlab/mmagic RealBasicVSRNet, UnsharpMasking and
/// configs/real_basicvsr/realbasicvsr_wogan-c64b20-2x30x8_8xb2-lr1e-4-300k_reds.py): the cleaning loop, the
/// unsharp-masked target, the two L1 losses, the frozen SPyNet and the weight average (#2397).
/// </summary>
public class RealBasicVSRPaperTests
{
    private const int Size = 16;

    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(inputType: InputType.FourDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputFrames: 3, inputDepth: 3, inputHeight: Size, inputWidth: Size, outputSize: 3);

    private static RealBasicVSR<double> Small(double learningRate = 1e-4, double? ema = 0.001, bool sharpen = true) => new(Architecture(), new RealBasicVSROptions
    {
        Seed = 1, MidChannels = 8, NumPropagationBlocks = 1, NumCleaningBlocks = 1, NumFrames = 3,
        LearningRate = learningRate, EmaMomentum = ema, SharpenTarget = sharpen,
    });

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTheFirstStageRecipe()
    {
        await Task.Yield();
        var o = new RealBasicVSROptions();
        Assert.Equal((64, 20, 20, 15), (o.MidChannels, o.NumPropagationBlocks, o.NumCleaningBlocks, o.NumFrames));
        Assert.Equal((1e-4, 1.0, 1.0, 1.5), (o.LearningRate, o.PixelLossWeight, o.CleaningLossWeight, o.InferenceCleaningThreshold));
        Assert.Equal((true, 0.5, 51, 10.0), (o.SharpenTarget, o.UnsharpWeight, o.UnsharpKernelSize, o.UnsharpThreshold));
        Assert.Equal(0.001, o.EmaMomentum);
    }

    [Fact(Timeout = 60000)]
    public async Task UnsharpMask_SharpensEdges_AndLeavesFlatRegions()
    {
        await Task.Yield();
        const int w = 64;
        var image = new Tensor<double>(new[] { 1, 1, 8, w });
        for (int y = 0; y < 8; y++)
            for (int x = 0; x < w; x++) image[y * w + x] = x < w / 2 ? 0.3 : 0.7;
        var sharp = RealBasicVsrCore<double>.UnsharpMask(image, 0.5, 51, 10);
        // Beside the edge the residue exceeds 10 levels, so the edge overshoots on both sides...
        Assert.True(sharp[4 * w + w / 2 - 1] < 0.3 - 0.01, $"dark side {sharp[4 * w + w / 2 - 1]}");
        Assert.True(sharp[4 * w + w / 2] > 0.7 + 0.01, $"bright side {sharp[4 * w + w / 2]}");
        // ...while a flat image has no residue and is returned unchanged, and outputs stay in [0, 1].
        var flat = new Tensor<double>(new[] { 1, 1, 8, w });
        for (int i = 0; i < flat.Length; i++) flat[i] = 0.4;
        Assert.All(RealBasicVsrCore<double>.UnsharpMask(flat, 0.5, 51, 10).ToArray(), v => Assert.Equal(0.4, v, 12));
        Assert.All(sharp.ToArray(), v => Assert.InRange(v, 0.0, 1.0));
    }

    [Fact(Timeout = 60000)]
    public async Task Reflect101_IsCv2sDefaultBorder()
    {
        await Task.Yield();
        Assert.Equal(new[] { 2, 1, 0, 1, 2, 3, 4, 3, 2 }, Enumerable.Range(-2, 9).Select(i => RealBasicVsrCore<double>.Reflect101(i, 5)).ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task Cleaning_RepeatsUntilTheResidueIsSmall_AtMostThreeTimes()
    {
        await Task.Yield();
        var model = Small();
        var clip = new Tensor<double>(new[] { 3, 3, Size, Size });
        for (int i = 0; i < clip.Length; i++) clip[i] = 0.5 + 0.3 * Math.Sin(i * 0.37);
        var (output, cleaned) = model.RunNormalizedForTest(clip, threshold: 10.0);
        Assert.Equal(1, model.LastCleaningPasses);
        Assert.Equal(new[] { 3, 3, 4 * Size, 4 * Size }, output.Shape.ToArray());
        Assert.Equal(clip.Shape.ToArray(), cleaned.Shape.ToArray());
        model.RunNormalizedForTest(clip, threshold: 0.0);
        Assert.Equal(3, model.LastCleaningPasses);
    }

    [Fact(Timeout = 300000)]
    public async Task Training_ReducesTheError_AndNeverTouchesSpyNet()
    {
        await Task.Yield();
        var (low, high) = Pair();
        var model = Small(learningRate: 1e-3, ema: null, sharpen: false);
        // SPyNet's convolutions are lazily shaped: snapshot the weights after a forward has created them.
        model.Predict(low);
        var flowBefore = model.FlowParameters.Select(p => p.ToArray()).ToList();
        double Error()
        {
            var output = model.Predict(low);
            return Enumerable.Range(0, output.Length).Average(i => Math.Abs(output[i] - high[i]));
        }

        double before = Error();
        for (int step = 0; step < 30; step++) model.Train(low, high);
        double after = Error();
        Assert.True(after < 0.7 * before, $"L1 {before:F2} -> {after:F2}");
        var flowAfter = model.FlowParameters.Select(p => p.ToArray()).ToList();
        Assert.True(flowBefore.Zip(flowAfter, (b, a) => b.SequenceEqual(a)).All(x => x), "SPyNet must stay frozen");
    }

    [Fact(Timeout = 300000)]
    public async Task Inference_UsesTheMovingAverageOfTheWeights()
    {
        await Task.Yield();
        var (low, high) = Pair();
        var averaged = Small(learningRate: 1e-3);
        var plain = Small(learningRate: 1e-3, ema: null);
        for (int step = 0; step < 3; step++)
        {
            averaged.Train(low, high);
            plain.Train(low, high);
        }
        // Same seed and steps, so the trained weights agree; with momentum 0.001 the average still sits at the weights
        // after the first step.
        var a = averaged.Predict(low);
        var p = plain.Predict(low);
        Assert.True(Enumerable.Range(0, a.Length).Max(i => Math.Abs(a[i] - p[i])) > 1e-6, "inference must use the average");
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
