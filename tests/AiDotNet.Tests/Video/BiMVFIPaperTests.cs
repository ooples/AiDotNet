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
/// Seo, Oh and Kim 2025, BiM-VFI (arXiv 2412.11365), and the release (KAIST-VICLab/BiM-VFI: cfgs/bim_vfi.yaml,
/// modules/components/bim_vfi, modules/loss.py, utils/padder.py): the configuration, InputPadder, the cost volume,
/// anti-aliased downsampling and CAUN's convex upsampling against brute-force transcriptions, the ternary loss,
/// arbitrary-time synthesis and a training step (#2397).
/// </summary>
public class BiMVFIPaperTests
{
    private static IEngine Engine => AiDotNetEngine.Current;

    private static BiMVFI<double> Model(BiMVFIOptions options) =>
        new(new NeuralNetworkArchitecture<double>(inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 32, inputWidth: 32, inputDepth: 3, outputSize: 3), options);

    private static BiMVFIOptions Small() => new() { Seed = 1, FeatureChannels = 4, TotalIterations = 100 };

    private static Tensor<double> Random(int[] shape, int seed, double scale = 1.0)
    {
        var t = new Tensor<double>(shape);
        var random = new System.Random(seed);
        for (int i = 0; i < t.Length; i++) t[i] = scale * random.NextDouble();
        return t;
    }

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTheReleaseConfiguration()
    {
        await Task.Yield();
        var o = new BiMVFIOptions();
        Assert.Equal((32, 3), (o.FeatureChannels, o.PyramidLevels));
        Assert.Equal((1.5e-4, 1e-4, 0.01, 1.0), (o.LearningRate, o.WeightDecay, o.WarmupFraction, o.MaxGradientNorm));
        Assert.Equal(400 * 2019, o.TotalIterations);
        Assert.Equal((0.25, 0.5), (o.LevelDecay, o.FlowLevelDecay));
        Assert.Equal((0.005, 0.005, 0.05, 150.0), (o.DistillationWeight, o.RegularizationWeight, o.SmoothnessWeight, o.SmoothnessEdgeConstant));
    }

    // utils/padder.InputPadder(divisor = 2^(levels + 1)): zeros on the right and bottom only.
    [Fact(Timeout = 60000)]
    public async Task Padding_IsInputPadder()
    {
        await Task.Yield();
        Assert.Equal((14, 12), BimVfiNetwork<double>.Padding(100, 130, 3));
        Assert.Equal((0, 0), BimVfiNetwork<double>.Padding(64, 64, 3));
        Assert.Equal((30, 28), BimVfiNetwork<double>.Padding(100, 130, 4));
    }

    // costvol_out: tap (dy, dx) y-major, the channel sum of one(p)·two(p + d − r), zero outside.
    [Fact(Timeout = 60000)]
    public async Task CostVolume_MatchesTheKernel()
    {
        await Task.Yield();
        const int c = 3, h = 5, w = 6, k = 3;
        var one = Random(new[] { 1, c, h, w }, 1);
        var two = Random(new[] { 1, c, h, w }, 2);
        var volume = BimOps.CostVolume(Engine, one, two, k);
        Assert.Equal(new[] { 1, k * k, h, w }, volume.Shape.ToArray());
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                for (int dy = 0; dy < k; dy++)
                    for (int dx = 0; dx < k; dx++)
                    {
                        int oy = y + dy - 1, ox = x + dx - 1;
                        double expected = oy < 0 || oy >= h || ox < 0 || ox >= w ? 0.0
                            : Enumerable.Range(0, c).Sum(ch => one[0, ch, y, x] * two[0, ch, oy, ox]);
                        Assert.Equal(expected, volume[0, dy * k + dx, y, x], 10);
                    }
    }

    // aten's anti-aliased bilinear: by 2 the triangle covers four inputs (1, 3, 3, 1) / 8, renormalized at the borders.
    [Fact(Timeout = 60000)]
    public async Task AntialiasDownsample_IsTorchsTriangleFilter()
    {
        await Task.Yield();
        var m = BimOps.AntialiasWeights<double>(8, 4, 2);
        Assert.Equal(new[] { 3 / 7.0, 3 / 7.0, 1 / 7.0, 0, 0, 0, 0, 0 }, Enumerable.Range(0, 8).Select(j => m[0, j]).ToArray(), new ToleranceComparer(1e-12));
        Assert.Equal(new[] { 0, 0.125, 0.375, 0.375, 0.125, 0, 0, 0 }, Enumerable.Range(0, 8).Select(j => m[1, j]).ToArray(), new ToleranceComparer(1e-12));
        var by4 = BimOps.AntialiasWeights<double>(16, 4, 4);
        Assert.Equal(new[] { 1, 3, 5, 7, 7, 5, 3, 1 }.Select(v => v / 32.0).ToArray(), Enumerable.Range(2, 8).Select(j => by4[1, j]).ToArray(), new ToleranceComparer(1e-12));

        var image = Random(new[] { 1, 2, 8, 8 }, 3);
        var down = BimOps.AntialiasDownsample(Engine, image, 2);
        Assert.Equal(new[] { 1, 2, 4, 4 }, down.Shape.ToArray());
        double expected = 0;
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++) expected += m[1, i] * m[2, j] * image[0, 1, i, j];
        Assert.Equal(expected, down[0, 1, 1, 2], 10);
    }

    // CAUN.upsample_input: uniform logits average the replicate-padded 3x3 neighbourhood; a dominant centre logit
    // repeats the coarse pixel.
    [Fact(Timeout = 60000)]
    public async Task ConvexUpsample_CombinesTheNeighbourhood()
    {
        await Task.Yield();
        var input = Random(new[] { 1, 2, 3, 4 }, 4);
        var uniform = BimOps.ConvexUpsample(Engine, input, new Tensor<double>(new[] { 1, 9, 6, 8 }), 2);
        Assert.Equal(new[] { 1, 2, 6, 8 }, uniform.Shape.ToArray());
        double Clamped(int ch, int y, int x) => input[0, ch, Math.Min(Math.Max(y, 0), 2), Math.Min(Math.Max(x, 0), 3)];
        for (int y = 0; y < 3; y++)
            for (int x = 0; x < 4; x++)
            {
                double mean = 0;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++) mean += Clamped(1, y + dy, x + dx) / 9;
                Assert.Equal(mean, uniform[0, 1, 2 * y + 1, 2 * x], 10);
            }
        var centre = new Tensor<double>(new[] { 1, 9, 6, 8 });
        for (int i = 0; i < 48; i++) centre[4 * 48 + i] = 50;
        var nearest = BimOps.ConvexUpsample(Engine, input, centre, 2);
        Assert.Equal(input[0, 0, 2, 3], nearest[0, 0, 5, 6], 10);
    }

    [Fact(Timeout = 60000)]
    public async Task Ternary_VanishesForMatchingStructure()
    {
        await Task.Yield();
        var a = Random(new[] { 1, 3, 12, 12 }, 5);
        var b = Random(new[] { 1, 3, 12, 12 }, 6);
        Assert.Equal(0.0, BimOps.Ternary(Engine, a, a)[0], 12);
        Assert.True(BimOps.Ternary(Engine, a, b)[0] > 0.05);
        Assert.Equal(BruteForceTernary(a, b), BimOps.Ternary(Engine, a, b)[0], 10);
    }

    // loss.Ternary transcribed: rgb2gray, the 7x7 census zero-padded by 3, δ²/(0.1 + δ²) averaged over the 49 taps,
    // a one-pixel border mask and the mean over all pixels. The zero padding reaches 3 pixels in while the mask covers
    // only 1, so unlike AMT's ternary this one is not invariant to a brightness shift near the border.
    private static double BruteForceTernary(Tensor<double> prediction, Tensor<double> target)
    {
        int h = prediction.Shape[2], w = prediction.Shape[3];
        double Gray(Tensor<double> im, int y, int x)
            => y < 0 || y >= h || x < 0 || x >= w ? 0.0 : 0.2989 * im[0, 0, y, x] + 0.5870 * im[0, 1, y, x] + 0.1140 * im[0, 2, y, x];
        double Census(Tensor<double> im, int y, int x, int dy, int dx)
        {
            double d = Gray(im, y + dy, x + dx) - Gray(im, y, x);
            return d / Math.Sqrt(0.81 + d * d);
        }
        double total = 0;
        for (int y = 1; y < h - 1; y++)
            for (int x = 1; x < w - 1; x++)
            {
                double sum = 0;
                for (int dy = -3; dy <= 3; dy++)
                    for (int dx = -3; dx <= 3; dx++)
                    {
                        double delta = Census(target, y, x, dy, dx) - Census(prediction, y, x, dy, dx);
                        sum += delta * delta / (0.1 + delta * delta);
                    }
                total += sum / 49;
            }
        return total / (h * w);
    }

    [Fact(Timeout = 240000)]
    public async Task Interpolate_SynthesizesAnyTimeAtAnySize()
    {
        await Task.Yield();
        var model = Model(Small());
        var frame0 = Random(new[] { 3, 20, 28 }, 7, 255.0);
        var frame1 = Random(new[] { 3, 20, 28 }, 8, 255.0);
        var middle = model.Interpolate(frame0, frame1);
        var early = model.Interpolate(frame0, frame1, 0.25);
        Assert.Equal(new[] { 3, 20, 28 }, middle.Shape.ToArray());
        Assert.All(middle.ToArray(), v => Assert.False(double.IsNaN(v)));
        Assert.False(middle.ToArray().SequenceEqual(early.ToArray()));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Interpolate(frame0, frame1, 1.5));
    }

    [Fact(Timeout = 240000)]
    public async Task TrainingStep_UpdatesTheNetwork()
    {
        await Task.Yield();
        var model = Model(Small());
        var before = model.NetworkParameters.SelectMany(t => t.ToArray()).ToArray();
        model.Train(Random(new[] { 6, 32, 32 }, 9, 255.0), Random(new[] { 3, 32, 32 }, 10, 255.0));
        var after = model.NetworkParameters.SelectMany(t => t.ToArray()).ToArray();
        Assert.Equal(before.Length, after.Length);
        Assert.False(before.SequenceEqual(after));
        Assert.All(after, v => Assert.False(double.IsNaN(v)));
    }

    [Fact(Timeout = 240000)]
    public async Task ReleaseLoss_RequiresTheTrainingDivisor()
    {
        await Task.Yield();
        var model = Model(Small());
        var frame = Random(new[] { 1, 3, 24, 24 }, 11);
        Assert.Throws<ArgumentException>(() => model.ReleaseLoss(frame, frame, frame));
        var aligned = Random(new[] { 1, 3, 32, 32 }, 12);
        var loss = model.ReleaseLoss(aligned, Random(new[] { 1, 3, 32, 32 }, 13), Random(new[] { 1, 3, 32, 32 }, 14))[0];
        Assert.True(loss > 0 && !double.IsNaN(loss));
    }

    private sealed class ToleranceComparer : System.Collections.Generic.IEqualityComparer<double>
    {
        private readonly double _tolerance;
        public ToleranceComparer(double tolerance) => _tolerance = tolerance;
        public bool Equals(double x, double y) => Math.Abs(x - y) <= _tolerance;
        public int GetHashCode(double obj) => 0;
    }
}
