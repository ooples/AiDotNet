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
/// Liang et al. 2022, RVRT, against the authors' release (JingyunLiang/RVRT network_rvrt.py and op/deform_attn, KAIR
/// options/rvrt): the window attention's position index and shift mask, the guided deformable attention op, two-way
/// clip propagation and the training recipe (#2397).
/// </summary>
public class RVRTPaperTests
{
    private const int Size = 8;

    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(inputType: InputType.FourDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputFrames: 4, inputDepth: 3, inputHeight: Size, inputWidth: Size, outputSize: 3);

    private static RVRT<double> Small(int freeze = 30000, double learningRate = 4e-4) => new(Architecture(), new RVRTOptions
    {
        Seed = 1, EmbedDim = 12, NumHeads = 2, AttentionHeads = 2, DeformableGroups = 2, PropagationBlocks = 1, NumFrames = 4,
        FlowFreezeIterations = freeze, LearningRate = learningRate,
    });

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTheReleaseRecipe()
    {
        await Task.Yield();
        var o = new RVRTOptions();
        Assert.Equal((144, 6, 8, 2), (o.EmbedDim, o.NumHeads, o.WindowSize, o.Depth));
        Assert.Equal((1, 2, 1, 2.0), (o.ExtractionBlocks, o.PropagationBlocks, o.ReconstructionBlocks, o.MlpRatio));
        Assert.Equal((12, 12, 10.0, 30), (o.DeformableGroups, o.AttentionHeads, o.MaxResidueMagnitude, o.NumFrames));
        Assert.Equal((4e-4, 0.25, 30000, 1e-7, 600000), (o.LearningRate, o.SpyNetLearningRateMultiplier, o.FlowFreezeIterations, o.MinLearningRate, o.CosineIterations));
        Assert.Equal(1e-9, o.CharbonnierEpsilon * o.CharbonnierEpsilon, 15);
    }

    [Fact(Timeout = 60000)]
    public async Task PositionIndex_IsTheReleaseTable()
    {
        await Task.Yield();
        var index = RvrtBlocks.PositionIndex(new[] { 2, 8, 8 });
        // Token 0 is (0, 0, 0) and token 127 is (1, 7, 7): offsets shifted by window - 1 into a 3 x 15 x 15 table.
        Assert.Equal(1 * 225 + 7 * 15 + 7, index[0, 0]);
        Assert.Equal(0, index[0, 127]);
        Assert.Equal(2 * 225 + 14 * 15 + 14, index[127, 0]);
    }

    [Fact(Timeout = 60000)]
    public async Task ShiftMask_SeparatesRegions_AndAnUnshiftedAxisIsOneRegion()
    {
        await Task.Yield();
        // D = 2 with window 2 and shift 0: slice(-0, None) covers every frame, so depth adds no regions.
        var mask = RvrtBlocks.ShiftMask(2, 16, 16, new[] { 2, 8, 8 }, new[] { 0, 4, 4 }, out int windows);
        Assert.Equal(4, windows);
        const int n = 128;
        Assert.All(Enumerable.Range(0, n * n), i => Assert.Equal(0.0, mask[i]));
        int Token(int z, int y, int x) => (z * 8 + y) * 8 + x;
        double At(int w, int i, int j) => mask[(w * n + i) * n + j];
        // The bottom-right window holds rows/columns 8-11 (one region) and 12-15 (the wrapped region).
        Assert.Equal(0.0, At(3, Token(0, 0, 0), Token(1, 3, 3)));
        Assert.Equal(-100.0, At(3, Token(0, 0, 0), Token(0, 7, 7)));
        Assert.Equal(-100.0, At(3, Token(0, 0, 0), Token(0, 0, 4)));
        Assert.Equal(0.0, At(3, Token(0, 4, 4), Token(1, 7, 7)));
    }

    [Fact(Timeout = 60000)]
    public async Task DeformAttention_MatchesTheReleaseOp()
    {
        await Task.Yield();
        const int clip = 2, p = 4, heads = 2, groups = 2, h = 4, w = 4;
        var q = Filled(new[] { 1, clip, p, h, w }, 0.31);
        var kv = Filled(new[] { 1, clip, 2 * p, h, w }, 0.17);
        var offsets = new Tensor<double>[clip];
        for (int b = 0; b < clip; b++)
        {
            offsets[b] = new Tensor<double>(new[] { 1, clip, groups * 18, h, w });
            for (int i = 0; i < offsets[b].Length; i++) offsets[b][i] = ((i * 7 + b * 3) % 5 - 2) * 0.5;
        }

        var actual = RvrtGuidedDeformableAttention<double>.DeformAttention(AiDotNetEngine.Current, q, kv, offsets, groups, heads);
        var expected = ReferenceDeformAttention(q, kv, offsets, clip, p, heads, groups, h, w);
        Assert.Equal(new[] { 1, clip, p, h, w }, actual.Shape.ToArray());
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 10);
    }

    private static Tensor<double> Filled(int[] shape, double rate)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = Math.Sin(i * rate) + 0.1 * Math.Cos(i * 0.07);
        return t;
    }

    // deform_attn_cuda_forward with modulated_deformable_im2col (mask of ones, 3x3, pad 1) written out directly.
    private static double[] ReferenceDeformAttention(Tensor<double> q, Tensor<double> kv, Tensor<double>[] offsets,
        int clip, int p, int heads, int groups, int h, int w)
    {
        int ad = p / heads, perGroup = 2 * p / groups;
        double Kv(int frame, int c, int y, int x) => kv[((frame * 2 * p + c) * h + y) * w + x];
        double Bilinear(int frame, int c, double y, double x)
        {
            if (y <= -1 || x <= -1 || y >= h || x >= w) return 0;
            int y0 = (int)Math.Floor(y), x0 = (int)Math.Floor(x);
            double ly = y - y0, lx = x - x0, sum = 0;
            foreach (var (yy, wy) in new[] { (y0, 1 - ly), (y0 + 1, ly) })
                foreach (var (xx, wx) in new[] { (x0, 1 - lx), (x0 + 1, lx) })
                    if (yy >= 0 && yy < h && xx >= 0 && xx < w) sum += wy * wx * Kv(frame, c, yy, xx);
            return sum;
        }

        var output = new double[clip * p * h * w];
        for (int i = 0; i < clip; i++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    for (int head = 0; head < heads; head++)
                    {
                        var logits = new double[clip * 9];
                        var vals = new double[clip * 9, ad];
                        for (int b = 0; b < clip; b++)
                        {
                            int frame = (b + i) % clip;
                            for (int t = 0; t < 9; t++)
                            {
                                int ky = t / 3, kx = t % 3;
                                double Sample(int c)
                                {
                                    int g = c / perGroup;
                                    double dy = offsets[b][((i * groups * 18) + g * 18 + 2 * t) * h * w + y * w + x];
                                    double dx = offsets[b][((i * groups * 18) + g * 18 + 2 * t + 1) * h * w + y * w + x];
                                    return Bilinear(frame, c, y - 1 + ky + dy, x - 1 + kx + dx);
                                }
                                double dot = 0;
                                for (int d = 0; d < ad; d++)
                                {
                                    int c = head * ad + d;
                                    dot += q[((i * p + c) * h + y) * w + x] * Math.Pow(ad, -0.5) * Sample(c);
                                    vals[b * 9 + t, d] = Sample(p + c);
                                }
                                logits[b * 9 + t] = dot;
                            }
                        }
                        double max = logits.Max(), z = logits.Sum(l => Math.Exp(l - max));
                        for (int d = 0; d < ad; d++)
                        {
                            double o = 0;
                            for (int s = 0; s < clip * 9; s++) o += Math.Exp(logits[s] - max) / z * vals[s, d];
                            output[((i * p + head * ad + d) * h + y) * w + x] = o;
                        }
                    }
        return output;
    }

    [Fact(Timeout = 300000)]
    public async Task Output_IsFourTimesTheInput_AndPropagatesBothWays()
    {
        await Task.Yield();
        var model = Small();
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

    [Fact(Timeout = 60000)]
    public async Task OddFrameCounts_AreRejected()
    {
        await Task.Yield();
        Assert.Throws<InvalidOperationException>(() => new RVRT<double>(Architecture(), new RVRTOptions { NumFrames = 5 }));
    }

    [Fact(Timeout = 600000)]
    public async Task SpyNet_StaysFrozenForTheFixIterations()
    {
        await Task.Yield();
        var (low, high) = Pair();
        var model = Small(freeze: 100, learningRate: 1e-3);
        // SPyNet's convolutions are lazily shaped: snapshot the weights after a forward has created them.
        model.Predict(low);
        var flow = model.FlowParameters.Select(t => t.ToArray()).ToList();
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
