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
/// Niklaus and Liu 2020, softmax splatting (arXiv 2003.05534), and the release (sniklaus/softmax-splatting: run.py,
/// softsplat.py, correlation/correlation.py): the parameter layout of the released checkpoint, the correlation and the
/// splatting kernel against brute-force transcriptions, softmax splatting's invariance to shifting Z, the input
/// standardization, the Laplacian loss, arbitrary-time synthesis and a training step that reaches every weight (#2397).
/// </summary>
public class SoftSplatPaperTests
{
    private static IEngine Engine => AiDotNetEngine.Current;

    private static SoftSplatOptions Small() => new()
    {
        Seed = 1, FlowChannels = new[] { 2, 2, 2, 2, 2, 2 }, FlowDecoderChannels = new[] { 2, 2, 2, 2, 2 },
        MetricChannels = new[] { 6, 2, 2, 2 }, SynthesisChannels = new[] { 2, 2, 2 },
    };

    private static SoftSplat<double> Model(SoftSplatOptions options) =>
        new(new NeuralNetworkArchitecture<double>(inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 32, inputWidth: 32, inputDepth: 3, outputSize: 3), options);

    private static Tensor<double> Random(int[] shape, int seed, double scale = 1.0, double offset = 0.0)
    {
        var t = new Tensor<double>(shape);
        var random = new System.Random(seed);
        for (int i = 0; i < t.Length; i++) t[i] = offset + scale * random.NextDouble();
        return t;
    }

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTheReleaseAndCtxSynTraining()
    {
        await Task.Yield();
        var o = new SoftSplatOptions();
        Assert.Equal(new[] { 16, 32, 64, 96, 128, 192 }, o.FlowChannels);
        Assert.Equal(new[] { 128, 128, 96, 64, 32 }, o.FlowDecoderChannels);
        Assert.Equal(new[] { 16, 32, 64, 96 }, o.MetricChannels);
        Assert.Equal(new[] { 32, 64, 96 }, o.SynthesisChannels);
        Assert.Equal((1e-3, 0.9, 0.999, 5), (o.LearningRate, o.Beta1, o.Beta2, o.LaplacianLevels));
    }

    // network-lf.pytorch, the released weights run.py loads, holds 278 tensors and 7,668,852 parameters.
    [Fact(Timeout = 240000)]
    public async Task Parameters_MatchTheReleasedCheckpoint()
    {
        await Task.Yield();
        var parameters = Model(new SoftSplatOptions()).NetworkParameters;
        Assert.Equal(278, parameters.Count);
        Assert.Equal(7668852, parameters.Sum(t => t.Length));
    }

    // correlation.py's kernel: channel (dy + 4)·9 + (dx + 4) holds the channel mean of one(y, x)·two(y + dy, x + dx),
    // zero outside.
    [Fact(Timeout = 60000)]
    public async Task Correlation_MatchesTheRelease()
    {
        await Task.Yield();
        int n = 2, c = 3, h = 5, w = 6;
        var one = Random(new[] { n, c, h, w }, 1);
        var two = Random(new[] { n, c, h, w }, 2);
        var volume = SoftSplatOps.Correlation(Engine, one, two);
        Assert.Equal(new[] { n, 81, h, w }, volume.Shape.ToArray());
        for (int b = 0; b < n; b++)
            for (int k = 0; k < 81; k++)
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        int dy = k / 9 - 4, dx = k % 9 - 4;
                        double sum = 0;
                        if (y + dy >= 0 && y + dy < h && x + dx >= 0 && x + dx < w)
                            for (int ch = 0; ch < c; ch++) sum += one[b, ch, y, x] * two[b, ch, y + dy, x + dx];
                        Assert.Equal(sum / c, volume[b, k, y, x], 12);
                    }
    }

    // softsplat_out transcribed: each source pixel adds its value to the four pixels around (x + u, y + v) with the
    // bilinear weights, dropping whatever lands outside.
    [Fact(Timeout = 60000)]
    public async Task Splat_MatchesTheReleaseKernel()
    {
        await Task.Yield();
        int n = 2, c = 2, h = 5, w = 7;
        var x = Random(new[] { n, c, h, w }, 3);
        var flow = Random(new[] { n, 2, h, w }, 4, 6.0, -3.0);
        var expected = new double[n * c * h * w];
        for (int b = 0; b < n; b++)
            for (int y = 0; y < h; y++)
                for (int xx = 0; xx < w; xx++)
                {
                    double fx = xx + flow[b, 0, y, xx], fy = y + flow[b, 1, y, xx];
                    int nx = (int)Math.Floor(fx), ny = (int)Math.Floor(fy);
                    var corners = new[]
                    {
                        (nx, ny, (nx + 1 - fx) * (ny + 1 - fy)), (nx + 1, ny, (fx - nx) * (ny + 1 - fy)),
                        (nx, ny + 1, (nx + 1 - fx) * (fy - ny)), (nx + 1, ny + 1, (fx - nx) * (fy - ny)),
                    };
                    foreach (var (tx, ty, weight) in corners)
                        if (tx >= 0 && tx < w && ty >= 0 && ty < h)
                            for (int ch = 0; ch < c; ch++) expected[((b * c + ch) * h + ty) * w + tx] += weight * x[b, ch, y, xx];
                }
        var actual = SoftSplatOps.Splat(Engine, x, flow).ToArray();
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 12);
    }

    // Softmax splatting divides by the splatted exp(Z), so adding a constant to Z changes nothing beyond the release's
    // 1e-7 added to the divisor, and with zero flow every pixel stays put.
    [Fact(Timeout = 60000)]
    public async Task SoftmaxSplat_IsInvariantToShiftingZ()
    {
        await Task.Yield();
        var x = Random(new[] { 1, 3, 6, 6 }, 5);
        var flow = Random(new[] { 1, 2, 6, 6 }, 6, 4.0, -2.0);
        var metric = Random(new[] { 1, 1, 6, 6 }, 7, 4.0, -2.0);
        var a = SoftSplatOps.SoftmaxSplat(Engine, x, flow, metric).ToArray();
        var b = SoftSplatOps.SoftmaxSplat(Engine, x, flow, Engine.TensorAddScalar(metric, 3.0)).ToArray();
        // With N the splatted exp(Z), a = S / (N + ε) and b = S·e³ / (N·e³ + ε): equal up to ε alone.
        var normalizer = SoftSplatOps.Splat(Engine, Engine.TensorExp(metric), flow);
        for (int i = 0; i < a.Length; i++)
        {
            double n = normalizer[i % 36];
            Assert.Equal(a[i] * (n + 1e-7) / (n + 1e-7 * Math.Exp(-3.0)), b[i], 10);
        }
        var still = SoftSplatOps.SoftmaxSplat(Engine, x, new Tensor<double>(flow.Shape.ToArray()), metric).ToArray();
        for (int i = 0; i < still.Length; i++) Assert.Equal(x[i], still[i], 5);
    }

    // run.Network.forward: the mean of both frames' means, and the square root of the mean over the frames of each one's
    // population variance plus its mean's squared distance to the joint mean.
    [Fact(Timeout = 60000)]
    public async Task Statistics_AreTheReleaseStandardization()
    {
        await Task.Yield();
        var one = Random(new[] { 1, 3, 4, 5 }, 8);
        var two = Random(new[] { 1, 3, 4, 5 }, 9, 2.0);
        var all = one.ToArray().Concat(two.ToArray()).ToArray();
        double mean = all.Average();
        double std = Math.Sqrt(all.Select(v => (v - mean) * (v - mean)).Average());
        var (m, s) = SoftSplatNetwork<double>.Statistics(one, two);
        Assert.Equal(mean, m[0], 12);
        Assert.Equal(std, s[0], 12);
    }

    // The pyramid's details ignore a constant offset, which only the low-pass residual, weighted 2^4, sees.
    [Fact(Timeout = 60000)]
    public async Task LaplacianLoss_WeightsTheResidualBy16()
    {
        await Task.Yield();
        var a = Random(new[] { 1, 3, 32, 32 }, 10);
        Assert.Equal(0.0, SoftSplat<double>.LaplacianLoss(Engine, a, a, 5)[0], 12);
        Assert.Equal(16 * 0.1, SoftSplat<double>.LaplacianLoss(Engine, a, Engine.TensorAddScalar(a, 0.1), 5)[0], 9);
    }

    [Fact(Timeout = 240000)]
    public async Task Interpolate_SynthesizesAnyTimeAtAnySize()
    {
        await Task.Yield();
        var model = Model(Small());
        var frame0 = Random(new[] { 3, 21, 27 }, 11, 255.0);
        var frame1 = Random(new[] { 3, 21, 27 }, 12, 255.0);
        var middle = model.Interpolate(frame0, frame1);
        var early = model.Interpolate(frame0, frame1, 0.25);
        Assert.Equal(new[] { 3, 21, 27 }, middle.Shape.ToArray());
        Assert.All(middle.ToArray(), v => Assert.False(double.IsNaN(v)));
        Assert.False(middle.ToArray().SequenceEqual(early.ToArray()));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Interpolate(frame0, frame1, 1.5));
    }

    [Fact(Timeout = 240000)]
    public async Task Training_RejectsCropsThePyramidCannotHalve()
    {
        await Task.Yield();
        var model = Model(Small());
        Assert.Throws<ArgumentException>(() => model.Train(Random(new[] { 6, 24, 24 }, 13, 255.0), Random(new[] { 3, 24, 24 }, 14, 255.0)));
    }

    // The Laplacian loss reaches every weight: the flow estimator (fine-tuned end to end through the splatting weights
    // and the warps), the metric U-Net, the pyramid and the GridNet. The gradients are read from the tape, because at
    // these 2-channel widths the coarsest flow decoders' gradients fall to ~1e-30 through the coarse-to-fine chain,
    // far below AdaMax's epsilon, so whether a weight moves cannot witness them.
    [Fact(Timeout = 240000)]
    public async Task Loss_ReachesEveryWeight()
    {
        await Task.Yield();
        var model = Model(Small());
        var frame0 = Random(new[] { 1, 3, 32, 32 }, 15);
        var frame1 = Random(new[] { 1, 3, 32, 32 }, 16);
        var target = Random(new[] { 1, 3, 32, 32 }, 17);
        var parameters = model.NetworkParameters.ToArray();
        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> gradients;
        using (var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<double>())
        {
            var loss = model.ReleaseLoss(frame0, frame1, target);
            gradients = tape.ComputeGradients(loss, sources: parameters);
        }
        var unreached = Enumerable.Range(0, parameters.Length)
            .Where(i => !gradients.TryGetValue(parameters[i], out var g) || g.ToArray().All(v => v == 0.0)).ToArray();
        Assert.True(unreached.Length == 0, $"{unreached.Length} of {parameters.Length} weight tensors received no gradient: {string.Join(", ", unreached)}");
    }

    [Fact(Timeout = 240000)]
    public async Task TrainingStep_UpdatesTheNetwork()
    {
        await Task.Yield();
        var model = Model(Small());
        var before = model.NetworkParameters.SelectMany(t => t.ToArray()).ToArray();
        model.Train(Random(new[] { 6, 32, 32 }, 18, 255.0), Random(new[] { 3, 32, 32 }, 19, 255.0));
        var after = model.NetworkParameters.SelectMany(t => t.ToArray()).ToArray();
        Assert.False(before.SequenceEqual(after));
        Assert.All(after, v => Assert.False(double.IsNaN(v) || double.IsInfinity(v)));
    }
}
