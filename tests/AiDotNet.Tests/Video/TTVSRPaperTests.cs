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
/// Liu et al. 2022, TTVSR, against the authors' release (researchmm/TTVSR, TTVSRNet, LTAM and configs/TTVSR_reds4.py):
/// the three-scale tokenization, the trajectory location maps, long-range propagation and the training recipe (#2397).
/// </summary>
public class TTVSRPaperTests
{
    private const int Size = 8;

    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(inputType: InputType.FourDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputFrames: 4, inputDepth: 3, inputHeight: Size, inputWidth: Size, outputSize: 3);

    private static TTVSR<double> Small(int freeze = 5000, double learningRate = 2e-4, int keyframeStride = 3) => new(Architecture(), new TTVSROptions
    {
        Seed = 1, MidChannels = 8, NumResBlocks = 1, NumFrames = 4, KeyframeStride = keyframeStride,
        FlowFreezeIterations = freeze, LearningRate = learningRate,
    });

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTheReleaseRecipe()
    {
        await Task.Yield();
        var o = new TTVSROptions();
        Assert.Equal((64, 60, 50, 4, 3), (o.MidChannels, o.NumResBlocks, o.NumFrames, o.TokenStride, o.KeyframeStride));
        Assert.Equal((2e-4, 0.125, 5000, 1e-7, 400000), (o.LearningRate, o.SpyNetLearningRateMultiplier, o.FlowFreezeIterations, o.MinLearningRate, o.CosineIterations));
    }

    [Fact(Timeout = 60000)]
    public async Task Tokenization_MatchesTheReleasesUnfoldFoldPipeline()
    {
        await Task.Yield();
        const int c = 2, h = 8, w = 12;
        var feature = new Tensor<double>(new[] { 1, c, h, w });
        for (int i = 0; i < feature.Length; i++) feature[i] = Math.Sin(i * 0.37) + 0.01 * i;
        var tokens = Small().TokenizeForTest(feature);
        var scales = new[] { (4, 0), (6, 1), (8, 2) };
        for (int s = 0; s < 3; s++)
        {
            var expected = Reference(feature.ToArray(), c, h, w, scales[s].Item1, scales[s].Item2);
            Assert.Equal(new[] { 1, c * 16, h / 4, w / 4 }, tokens[s].Shape.ToArray());
            for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], tokens[s][i], 10);
        }
    }

    // torch: F.unfold(k, pad, stride 4) -> F.fold to (k/4·H, k/4·W) with k x k tiles -> adaptive_avg_pool2d to (H, W)
    // -> F.unfold(4, stride 4) refolded to (H/4, W/4); for k = 4 the first three steps are the identity.
    private static double[] Reference(double[] x, int c, int h, int w, int k, int pad)
    {
        int th = k * h / 4, tw = k * w / 4;
        var tiled = new double[c * th * tw];
        for (int ch = 0; ch < c; ch++)
            for (int py = 0; py < h / 4; py++)
                for (int px = 0; px < w / 4; px++)
                    for (int ky = 0; ky < k; ky++)
                        for (int kx = 0; kx < k; kx++)
                        {
                            int sy = 4 * py + ky - pad, sx = 4 * px + kx - pad;
                            double v = sy < 0 || sy >= h || sx < 0 || sx >= w ? 0 : x[(ch * h + sy) * w + sx];
                            tiled[(ch * th + k * py + ky) * tw + k * px + kx] = v;
                        }
        var pooled = new double[c * h * w];
        for (int ch = 0; ch < c; ch++)
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++)
                {
                    int y0 = i * th / h, y1 = ((i + 1) * th + h - 1) / h, x0 = j * tw / w, x1 = ((j + 1) * tw + w - 1) / w;
                    double sum = 0;
                    for (int y = y0; y < y1; y++)
                        for (int xx = x0; xx < x1; xx++) sum += tiled[(ch * th + y) * tw + xx];
                    pooled[(ch * h + i) * w + j] = sum / ((y1 - y0) * (x1 - x0));
                }
        var tokens = new double[c * 16 * (h / 4) * (w / 4)];
        for (int ch = 0; ch < c; ch++)
            for (int ky = 0; ky < 4; ky++)
                for (int kx = 0; kx < 4; kx++)
                    for (int y = 0; y < h / 4; y++)
                        for (int xx = 0; xx < w / 4; xx++)
                            tokens[((ch * 16 + ky * 4 + kx) * (h / 4) + y) * (w / 4) + xx] = pooled[(ch * h + 4 * y + ky) * w + 4 * xx + kx];
        return tokens;
    }

    [Fact(Timeout = 60000)]
    public async Task LocationMaps_FollowTheFlowWithNearestSamplingAndBorderPadding()
    {
        await Task.Yield();
        const int h = 2, w = 3;
        var map = new double[] { 0, 1, 2, 0, 1, 2, 0, 0, 0, 1, 1, 1 };
        // A flow of +1.4 px in x samples one column to the right (nearest), clamped at the border.
        var fx = Enumerable.Repeat(1.4, h * w).ToArray();
        var fy = new double[h * w];
        var warped = TtvsrBackbone<double>.WarpLocation(map, fx, fy, h, w);
        Assert.Equal(new double[] { 1, 2, 2, 1, 2, 2, 0, 0, 0, 1, 1, 1 }, warped);
        // Exact halves round to even, as grid_sample's nearest mode does: 0.5 -> 0, 1.5 -> 2.
        var half = TtvsrBackbone<double>.WarpLocation(map, Enumerable.Repeat(0.5, h * w).ToArray(), fy, h, w);
        Assert.Equal(new double[] { 0, 2, 2, 0, 2, 2 }, half.Take(6).ToArray());
    }

    [Fact(Timeout = 300000)]
    public async Task Output_IsFourTimesTheInput_AndPropagatesBothWays()
    {
        await Task.Yield();
        var model = Small(keyframeStride: 1);
        var clip = new Tensor<double>(new[] { 4, 3, Size, Size });
        for (int i = 0; i < clip.Length; i++) clip[i] = 0.5 + 0.3 * Math.Sin(i * 0.37);
        var baseline = model.UpscaleNormalizedForTest(clip);
        Assert.Equal(new[] { 4, 3, 4 * Size, 4 * Size }, baseline.Shape.ToArray());
        int frame = 3 * Size * Size, outFrame = 3 * 16 * Size * Size;

        var lastChanged = clip.Clone();
        for (int i = 3 * frame; i < 4 * frame; i++) lastChanged[i] += 0.2;
        var backward = model.UpscaleNormalizedForTest(lastChanged);
        Assert.True(Enumerable.Range(0, outFrame).Max(i => Math.Abs(backward[i] - baseline[i])) > 1e-9, "frame 0 must hear about frame 3");

        var firstChanged = clip.Clone();
        for (int i = 0; i < frame; i++) firstChanged[i] += 0.2;
        var forward = model.UpscaleNormalizedForTest(firstChanged);
        Assert.True(Enumerable.Range(3 * outFrame, outFrame).Max(i => Math.Abs(forward[i] - baseline[i])) > 1e-9, "frame 3 must hear about frame 0");
    }

    [Fact(Timeout = 600000)]
    public async Task SpyNet_StaysFrozenForTheFixIterations()
    {
        await Task.Yield();
        var (low, high) = Pair();
        var model = Small(freeze: 100, learningRate: 1e-3);
        // SPyNet's convolutions are lazily shaped: snapshot the weights after a forward has created them.
        model.Predict(low);
        var flow = model.FlowParameters.Select(p => p.ToArray()).ToList();
        var output = model.Predict(low).ToArray();
        model.Train(low, high);
        Assert.True(flow.Zip(model.FlowParameters, (b, a) => b.SequenceEqual(a.ToArray())).All(x => x), "SPyNet moved inside its freeze window");
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

    // A smooth 32x32 four-frame clip and its 4x box-downsampled 8x8 version, pixel values 0-255.
    private static (Tensor<double> Low, Tensor<double> High) Pair()
    {
        int big = 4 * Size;
        var high = new Tensor<double>(new[] { 4, 3, big, big });
        for (int f = 0; f < 4; f++)
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < big; y++)
                    for (int x = 0; x < big; x++)
                        high[((f * 3 + c) * big + y) * big + x] = 128 + 80 * Math.Sin((x + 2 * f) * 0.21 + c) * Math.Cos(y * 0.17);
        var low = new Tensor<double>(new[] { 4, 3, Size, Size });
        for (int f = 0; f < 4; f++)
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
