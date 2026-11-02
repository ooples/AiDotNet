using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Enhancement;
using AiDotNet.Video.Options;
using Xunit;

namespace AiDotNet.Tests.Video;

/// <summary>
/// Shi et al. 2022, PSRT-recurrent, against the authors' release (XPixelGroup/RethinkVSRAlignment,
/// psrt_recurrent_arch.py, psrt_sliding_arch.py and options/4126_PSRTRecurrent_mix_precision_REDS_600K_N16.yml):
/// patch alignment, the 3x8x8 shift mask, two-way propagation, the flow freeze, the moving average and the training
/// recipe (#2397).
/// </summary>
public class PSRTPaperTests
{
    private const int Size = 8;

    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(inputType: InputType.FourDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputFrames: 3, inputDepth: 3, inputHeight: Size, inputWidth: Size, outputSize: 3);

    private static PSRT<double> Small(int freeze = 5000, double learningRate = 2e-4, double? ema = 0.001) => new(Architecture(), new PSRTOptions
    {
        Seed = 1, MidChannels = 8, EmbedDim = 12, Depths = new[] { 2 }, NumHeads = 2, NumFrames = 3,
        FlowFreezeIterations = freeze, LearningRate = learningRate, EmaMomentum = ema,
    });

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTheReleaseRecipe()
    {
        await Task.Yield();
        var o = new PSRTOptions();
        Assert.Equal((64, 120, 6, 8), (o.MidChannels, o.EmbedDim, o.NumHeads, o.WindowSize));
        Assert.Equal(new[] { 6, 6, 6 }, o.Depths);
        Assert.Equal((2.0, 0.1, 16), (o.MlpRatio, o.DropPathRate, o.NumFrames));
        Assert.Equal((2e-4, 0.125, 5000, 1e-7, 600000), (o.LearningRate, o.SpyNetLearningRateMultiplier, o.FlowFreezeIterations, o.MinLearningRate, o.CosineIterations));
        Assert.Equal(1e-6, o.CharbonnierEpsilon, 15);
        Assert.Equal(0.001, o.EmaMomentum);
    }

    [Fact(Timeout = 60000)]
    public async Task ShiftMask_TemporalShiftOfAWholeWindowAddsNoRegions()
    {
        await Task.Yield();
        // compute_mask with window (3, 8, 8) and shift (3, 4, 4) on three frames: slice(-3) and slice(-3, -3) are empty,
        // so depth is one region and the mask is the spatial one alone.
        var withDepth = RvrtBlocks.ShiftMask(3, 8, 8, new[] { 3, 8, 8 }, new[] { 3, 4, 4 }, out int windows);
        var spatialOnly = RvrtBlocks.ShiftMask(3, 8, 8, new[] { 3, 8, 8 }, new[] { 0, 4, 4 }, out _);
        Assert.Equal(1, windows);
        Assert.Equal(spatialOnly, withDepth);
        Assert.Contains(-100.0, withDepth);
    }

    [Theory(Timeout = 60000)]
    [InlineData(8, 8)]
    [InlineData(10, 12)]
    public async Task PatchWarp_MovesEachPatchByItsMeanFlow(int h, int w)
    {
        await Task.Yield();
        const int c = 2;
        var x = new Tensor<double>(new[] { 1, c, h, w });
        for (int i = 0; i < x.Length; i++) x[i] = Math.Sin(i * 0.41) + 0.2 * i;
        var flow = new Tensor<double>(new[] { 1, 2, h, w });
        for (int i = 0; i < flow.Length; i++) flow[i] = ((i * 7) % 11 - 5) * 0.37;

        var actual = PsrtBackbone<double>.PatchWarp(AiDotNetEngine.Current, x, flow);
        var expected = ReferencePatchWarp(x, flow, c, h, w);
        Assert.Equal(new[] { 1, c, h, w }, actual.Shape.ToArray());
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 10);
    }

    // flow_warp_avg_patch written out: reflect pad to multiples of 8, 8x8 average, nearest spread, then
    // grid_sample(nearest, zeros, align_corners=True), which samples pixel round(x + dx), round(y + dy).
    private static double[] ReferencePatchWarp(Tensor<double> x, Tensor<double> flow, int c, int h, int w)
    {
        int Reflect(int i, int n) => i < n ? i : 2 * (n - 1) - i;
        double Mean(int axis, int py, int px)
        {
            double s = 0;
            for (int y = py * 8; y < py * 8 + 8; y++)
                for (int xx = px * 8; xx < px * 8 + 8; xx++) s += flow[(axis * h + Reflect(y, h)) * w + Reflect(xx, w)];
            return s / 64;
        }

        var output = new double[c * h * w];
        for (int y = 0; y < h; y++)
            for (int xx = 0; xx < w; xx++)
            {
                int sx = (int)Math.Round(xx + Mean(0, y / 8, xx / 8), MidpointRounding.ToEven);
                int sy = (int)Math.Round(y + Mean(1, y / 8, xx / 8), MidpointRounding.ToEven);
                for (int ch = 0; ch < c; ch++)
                    output[(ch * h + y) * w + xx] = sx < 0 || sy < 0 || sx >= w || sy >= h ? 0 : x[(ch * h + sy) * w + sx];
            }
        return output;
    }

    [Fact(Timeout = 300000)]
    public async Task Output_IsFourTimesTheInput_AndPropagatesBothWays()
    {
        await Task.Yield();
        var model = Small();
        var clip = new Tensor<double>(new[] { 3, 3, Size, Size });
        for (int i = 0; i < clip.Length; i++) clip[i] = 0.5 + 0.3 * Math.Sin(i * 0.37);
        var baseline = model.UpscaleNormalizedForTest(clip);
        Assert.Equal(new[] { 3, 3, 4 * Size, 4 * Size }, baseline.Shape.ToArray());
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
    public async Task SpyNet_StaysFrozenForTheFixIterations()
    {
        await Task.Yield();
        var (low, high) = Pair();
        var model = Small(freeze: 100, learningRate: 1e-3, ema: null);
        // SPyNet's convolutions are lazily shaped: snapshot the weights after a forward has created them.
        model.Predict(low);
        var flow = model.FlowParameters.Select(t => t.ToArray()).ToList();
        var output = model.Predict(low).ToArray();
        model.Train(low, high);
        Assert.True(flow.Zip(model.FlowParameters, (b, a) => b.SequenceEqual(a.ToArray())).All(x => x), "SPyNet moved inside its freeze window");
        Assert.False(output.SequenceEqual(model.Predict(low).ToArray()), "the rest of the network must train");
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
        // Same seed, steps and stochastic-depth draws, so the trained weights agree; with momentum 0.001 the average
        // still sits near the weights after the first step.
        var a = averaged.Predict(low);
        var p = plain.Predict(low);
        Assert.True(Enumerable.Range(0, a.Length).Max(i => Math.Abs(a[i] - p[i])) > 1e-6, "inference must use the average");
    }

    [Fact(Timeout = 600000)]
    public async Task Training_ReducesTheError()
    {
        await Task.Yield();
        var (low, high) = Pair();
        var model = Small(learningRate: 1e-3, ema: null);
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

    // A smooth 32x32 three-frame clip and its 4x box-downsampled 8x8 version, pixel values 0-255.
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
