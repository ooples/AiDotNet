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
/// Zhang et al. 2023, EMA-VFI (arXiv 2303.00440), and the release (MCG-NJU/EMA-VFI: config.py,
/// model/feature_extractor.py, flow_estimation.py, refine.py, loss.py, train.py): the configurations, InputPadder,
/// window partitioning and the padded/shifted attention masks against brute-force transcriptions, the Laplacian loss,
/// arbitrary-time synthesis and a training step (#2397).
/// </summary>
public class EMAVFIPaperTests
{
    private static IEngine Engine => AiDotNetEngine.Current;

    private static EMAVFIOptions Small() => new()
    {
        Seed = 1, FeatureChannels = 4, WindowSize = 4, Depths = new[] { 1, 1, 1, 2, 2 }, WarmupSteps = 0, TotalIterations = 100,
    };

    private static EMAVFI<double> Model(EMAVFIOptions options) =>
        new(new NeuralNetworkArchitecture<double>(inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 64, inputWidth: 64, inputDepth: 3, outputSize: 3), options);

    private static Tensor<double> Random(int[] shape, int seed, double scale = 1.0)
    {
        var t = new Tensor<double>(shape);
        var random = new System.Random(seed);
        for (int i = 0; i < t.Length; i++) t[i] = scale * random.NextDouble();
        return t;
    }

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTheReleaseConfigurations()
    {
        await Task.Yield();
        var o = new EMAVFIOptions();
        Assert.Equal((32, 7, 4.0), (o.FeatureChannels, o.WindowSize, o.MlpRatio));
        Assert.Equal(new[] { 2, 2, 2, 4, 4 }, o.Depths);
        Assert.Equal((2e-4, 2e-5, 1e-4, 2000), (o.LearningRate, o.MinLearningRate, o.WeightDecay, o.WarmupSteps));
        Assert.Equal(300 * 1603, o.TotalIterations);
        Assert.Equal((0.5, 5), (o.MergedWeight, o.LaplacianLevels));
        var small = EMAVFIOptions.Small();
        Assert.Equal(16, small.FeatureChannels);
        Assert.Equal(new[] { 2, 2, 2, 2, 2 }, small.Depths);
    }

    [Fact(Timeout = 60000)]
    public async Task Padding_IsInputPadderAt32()
    {
        await Task.Yield();
        Assert.Equal((15, 15, 14, 14), EMAVFI<double>.Padding(100, 130));
        Assert.Equal((0, 0, 0, 0), EMAVFI<double>.Padding(64, 96));
        Assert.Equal((0, 1, 7, 8), EMAVFI<double>.Padding(17, 31));
    }

    // window_partition: window k of image b holds rows wy·w.. and columns wx·w.., windows row-major per image.
    [Fact(Timeout = 60000)]
    public async Task WindowPartition_MatchesTheRelease()
    {
        await Task.Yield();
        var x = Random(new[] { 2, 4, 6, 3 }, 1);
        var windows = EmaOps.Partition(Engine, x, 2);
        Assert.Equal(new[] { 12, 4, 3 }, windows.Shape.ToArray());
        for (int b = 0; b < 2; b++)
            for (int wy = 0; wy < 2; wy++)
                for (int wx = 0; wx < 3; wx++)
                    for (int i = 0; i < 4; i++)
                        for (int c = 0; c < 3; c++)
                            Assert.Equal(x[b, wy * 2 + i / 2, wx * 2 + i % 2, c], windows[b * 6 + wy * 3 + wx, i, c]);
        Assert.Equal(x.ToArray(), EmaOps.Reverse(Engine, windows, 2, 4, 6).ToArray());
    }

    // pad_if_needed and MotionFormerBlock's shifted mask, transcribed: regions labelled by three row and column bands,
    // −100 between different regions within a window, and the padding mask's nonzeros forced to −100 when shifted.
    [Theory(Timeout = 60000)]
    [InlineData(8, 8, 4, 0)]
    [InlineData(8, 8, 4, 2)]
    [InlineData(6, 7, 4, 0)]
    [InlineData(6, 7, 4, 2)]
    public async Task AttentionMask_MatchesTheRelease(int h, int w, int window, int shift)
    {
        await Task.Yield();
        int padH = (h + window - 1) / window * window - h, padW = (w + window - 1) / window * window - w;
        int hp = h + padH, wp = w + padW;
        int Band(int v, int a, int b) => v < a ? 0 : v < b ? 1 : 2;
        double[] Mask(Func<int, int, int> region)
        {
            int nh = hp / window, nw = wp / window, n = window * window;
            var m = new double[nh * nw * n * n];
            for (int k = 0; k < nh * nw; k++)
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                    {
                        int yi = k / nw * window + i / window, xi = k % nw * window + i % window;
                        int yj = k / nw * window + j / window, xj = k % nw * window + j % window;
                        m[(k * n + i) * n + j] = region(yi, xi) == region(yj, xj) ? 0.0 : -100.0;
                    }
            return m;
        }
        double[]? pad = padH + padW > 0 ? Mask((y, x) => Band(y, padH / 2, h + padH / 2) * 3 + Band(x, padW / 2, w + padW / 2)) : null;
        double[]? expected = pad;
        if (shift > 0)
        {
            expected = Mask((y, x) => Band(y, hp - window, hp - shift) * 3 + Band(x, wp - window, wp - shift));
            if (pad is not null)
                for (int i = 0; i < expected.Length; i++)
                    if (pad[i] != 0) expected[i] = -100.0;
        }
        var actual = EmaOps.AttentionMask(h, w, window, shift);
        if (expected is null) Assert.Null(actual);
        else Assert.Equal(expected, actual);
    }

    [Fact(Timeout = 60000)]
    public async Task CoordinateGrid_IsLinspaceXThenY()
    {
        await Task.Yield();
        var grid = EmaOps.CoordinateGrid<double>(1, 3, 5);
        Assert.Equal(-1.0, grid[0, 0, 0, 0]);
        Assert.Equal(0.5, grid[0, 2, 3, 0], 12);
        Assert.Equal(1.0, grid[0, 2, 3, 1], 12);
        Assert.Equal(0.0, grid[0, 1, 4, 1], 12);
    }

    // The pyramid keeps only detail levels, so a constant offset leaves LapLoss at zero (the reflected zero-inserted
    // upsampling reproduces a constant exactly).
    [Fact(Timeout = 60000)]
    public async Task LaplacianLoss_IgnoresTheLowpassAndSeesDetail()
    {
        await Task.Yield();
        var a = Random(new[] { 1, 3, 32, 32 }, 2);
        var b = Random(new[] { 1, 3, 32, 32 }, 3);
        Assert.Equal(0.0, EmaOps.LaplacianLoss(Engine, a, a, 5)[0], 12);
        Assert.Equal(0.0, EmaOps.LaplacianLoss(Engine, a, Engine.TensorAddScalar(a, 0.1), 5)[0], 9);
        Assert.True(EmaOps.LaplacianLoss(Engine, a, b, 5)[0] > 0.1);
        var constant = new Tensor<double>(new[] { 1, 1, 16, 16 });
        constant.Fill(0.7);
        Assert.All(EmaOps.LaplacianPyramid(Engine, constant, 3).SelectMany(level => level.ToArray()), v => Assert.Equal(0.0, v, 12));
    }

    [Fact(Timeout = 240000)]
    public async Task Interpolate_SynthesizesAnyTimeAtAnySize()
    {
        await Task.Yield();
        var model = Model(Small());
        var frame0 = Random(new[] { 3, 40, 52 }, 4, 255.0);
        var frame1 = Random(new[] { 3, 40, 52 }, 5, 255.0);
        var middle = model.Interpolate(frame0, frame1);
        var early = model.Interpolate(frame0, frame1, 0.25);
        Assert.Equal(new[] { 3, 40, 52 }, middle.Shape.ToArray());
        Assert.All(middle.ToArray(), v => Assert.True(v >= 0 && v <= 255));
        Assert.False(middle.ToArray().SequenceEqual(early.ToArray()));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Interpolate(frame0, frame1, -0.1));
    }

    [Fact(Timeout = 240000)]
    public async Task TrainingStep_UpdatesTheNetwork()
    {
        await Task.Yield();
        var model = Model(Small());
        var before = model.NetworkParameters.SelectMany(t => t.ToArray()).ToArray();
        model.Train(Random(new[] { 6, 64, 64 }, 6, 255.0), Random(new[] { 3, 64, 64 }, 7, 255.0));
        var after = model.NetworkParameters.SelectMany(t => t.ToArray()).ToArray();
        Assert.Equal(before.Length, after.Length);
        Assert.False(before.SequenceEqual(after));
        Assert.All(after, v => Assert.False(double.IsNaN(v)));
    }

    // Without weight decay a weight moves only if its gradient is nonzero, so every tensor moving shows the loss reaches
    // the attention stages, the flow heads and the U-Net alike.
    [Fact(Timeout = 240000)]
    public async Task TrainingStep_ReachesEveryWeight()
    {
        await Task.Yield();
        var options = Small();
        options.WeightDecay = 0;
        var model = Model(options);
        var before = model.NetworkParameters.Select(t => t.ToArray()).ToArray();
        model.Train(Random(new[] { 6, 64, 64 }, 8, 255.0), Random(new[] { 3, 64, 64 }, 9, 255.0));
        var after = model.NetworkParameters.Select(t => t.ToArray()).ToArray();
        var frozen = Enumerable.Range(0, before.Length).Where(i => before[i].SequenceEqual(after[i])).ToArray();
        Assert.True(frozen.Length == 0, $"{frozen.Length} of {before.Length} weight tensors received no gradient: {string.Join(", ", frozen)}");
    }

    [Fact(Timeout = 240000)]
    public async Task Options_RejectWidthsWithoutWholeHeads()
    {
        await Task.Yield();
        Assert.Throws<ArgumentOutOfRangeException>(() => Model(new EMAVFIOptions { FeatureChannels = 6 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => Model(new EMAVFIOptions { Depths = new[] { 2, 2, 2, 3, 4 } }));
    }
}
