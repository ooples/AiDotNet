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
/// Sim, Oh and Kim 2021, XVFI (arXiv 2103.16206), and the release (JihyongOh/XVFI: XVFInet.py, main.py, utils.py): the
/// configurations and their parameter counts, the padding, the masked backward warp, the z-weighted Gaussian forward
/// warp and the bicubic downscaling against brute-force transcriptions, the smoothness loss, arbitrary-time synthesis at
/// a test depth beyond the training depth, and tape gradients reaching every weight (#2397).
/// </summary>
public class XVFIPaperTests
{
    private static IEngine Engine => AiDotNetEngine.Current;

    private static XVFIOptions Small() => new() { Seed = 1, NumFeatures = 4, ScaleFactor = 2, TrainingDepth = 1, TestDepth = 1 };

    private static XVFI<double> Model(XVFIOptions options) =>
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
    public async Task Defaults_AreTheReleaseX4KModelAndTraining()
    {
        await Task.Yield();
        var o = new XVFIOptions();
        Assert.Equal((64, 4, 3, 5), (o.NumFeatures, o.ScaleFactor, o.TrainingDepth, o.TestDepth));
        Assert.Equal(1e-4, o.LearningRate);
        Assert.Equal(new[] { 100, 150, 180 }, o.LearningRateMilestones);
        Assert.Equal((0.25, 1.0, 0.5, 150.0), (o.LearningRateDecay, o.ReconstructionWeight, o.SmoothnessWeight, o.EdgeConstant));
        var v = XVFIOptions.Vimeo();
        Assert.Equal((2, 1, 1), (v.ScaleFactor, v.TrainingDepth, v.TestDepth));
    }

    // The release's modules counted layer by layer (5,612,052 for X4K1000FPS), less RefineUNet's conv1 and conv2, which
    // its forward never uses: 5,538,196, the paper's 5.5M.
    [Theory(Timeout = 240000)]
    [InlineData(false, 5538196)]
    [InlineData(true, 5587348)]
    public async Task ParameterCount_MatchesTheRelease(bool vimeo, int expected)
    {
        await Task.Yield();
        Assert.Equal(expected, Model(vimeo ? XVFIOptions.Vimeo() : new XVFIOptions()).NetworkParameters.Sum(t => t.Length));
    }

    // main.test: args.divide = 2^S_tst · module_scale_factor · 4, zeros on the right and bottom.
    [Fact(Timeout = 60000)]
    public async Task Padding_IsTheReleaseDivide()
    {
        await Task.Yield();
        var model = Model(new XVFIOptions());
        Assert.Equal(512, model.Divisor(5));
        Assert.Equal((382, 412), model.Padding(100, 130, 5));
        Assert.Equal(16, Model(XVFIOptions.Vimeo()).Divisor(1));
    }

    // bwarp: grid_sample with zeros and align_corners=True, kept only where grid_sample of ones reaches 0.999.
    [Fact(Timeout = 60000)]
    public async Task Backwarp_MatchesTheRelease()
    {
        await Task.Yield();
        int n = 1, c = 2, h = 5, w = 6;
        var x = Random(new[] { n, c, h, w }, 1);
        var flow = Random(new[] { n, 2, h, w }, 2, 3.0, -1.5);
        var actual = XvfiOps.Backwarp(Engine, x, flow);
        for (int y = 0; y < h; y++)
            for (int xx = 0; xx < w; xx++)
            {
                double sx = xx + flow[0, 0, y, xx], sy = y + flow[0, 1, y, xx];
                int x0 = (int)Math.Floor(sx), y0 = (int)Math.Floor(sy);
                double fx = sx - x0, fy = sy - y0, inside = 0;
                var sample = new double[c];
                foreach (var (cx, cy, weight) in new[] { (x0, y0, (1 - fx) * (1 - fy)), (x0 + 1, y0, fx * (1 - fy)), (x0, y0 + 1, (1 - fx) * fy), (x0 + 1, y0 + 1, fx * fy) })
                    if (cx >= 0 && cx < w && cy >= 0 && cy < h)
                    {
                        inside += weight;
                        for (int ch = 0; ch < c; ch++) sample[ch] += weight * x[0, ch, cy, cx];
                    }
                for (int ch = 0; ch < c; ch++) Assert.Equal(inside >= 0.999 ? sample[ch] : 0.0, actual[0, ch, y, xx], 10);
            }
    }

    // z_fwarp / sample_one: each pixel adds value · (z + 1e-5) · exp(−((Δy − r)² + (Δx − c)²)) to (floor(y + Δy) + r,
    // floor(x + Δx) + c) for r, c in {0, 1}, inside the image only; the second output accumulates the weights.
    [Fact(Timeout = 60000)]
    public async Task ZForwardWarp_MatchesTheRelease()
    {
        await Task.Yield();
        int c = 2, h = 5, w = 6;
        var image = Random(new[] { 1, c, h, w }, 3);
        var flow = Random(new[] { 1, 2, h, w }, 4, 5.0, -2.5);
        var z = Random(new[] { 1, 1, h, w }, 5);
        var expected = new double[c * h * w];
        var weights = new double[h * w];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                double dx = flow[0, 0, y, x], dy = flow[0, 1, y, x];
                int fx = (int)Math.Floor(dx), fy = (int)Math.Floor(dy);
                for (int r = 0; r < 2; r++)
                    for (int col = 0; col < 2; col++)
                    {
                        int ty = y + fy + r, tx = x + fx + col;
                        if (ty < 0 || ty >= h || tx < 0 || tx >= w) continue;
                        double weight = (z[0, 0, y, x] + 1e-5) * Math.Exp(-(Math.Pow(dy - fy - r, 2) + Math.Pow(dx - fx - col, 2)));
                        weights[ty * w + tx] += weight;
                        for (int ch = 0; ch < c; ch++) expected[(ch * h + ty) * w + tx] += weight * image[0, ch, y, x];
                    }
            }
        var (warped, norm) = XvfiOps.ZForwardWarp(Engine, image, flow, z);
        Assert.Equal(expected.Length, warped.Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], warped[i], 10);
        for (int i = 0; i < weights.Length; i++) Assert.Equal(weights[i], norm[i], 10);
    }

    // F.interpolate(scale_factor=1/2, mode='bicubic', align_corners=False): source (i + 0.5)·2 − 0.5, Keys' cubic with
    // a = −0.75 over the four neighbours, indices clamped to the border.
    [Fact(Timeout = 60000)]
    public async Task Downscale_IsTorchBicubic()
    {
        await Task.Yield();
        int h = 8, w = 10;
        var x = Random(new[] { 1, 1, h, w }, 6);
        double Cubic(double d)
        {
            const double a = -0.75;
            d = Math.Abs(d);
            return d <= 1 ? ((a + 2) * d - (a + 3)) * d * d + 1 : d < 2 ? ((a * d - 5 * a) * d + 8 * a) * d - 4 * a : 0;
        }
        var actual = XvfiOps.Downscale(Engine, x, 2);
        Assert.Equal(new[] { 1, 1, h / 2, w / 2 }, actual.Shape.ToArray());
        for (int i = 0; i < h / 2; i++)
            for (int j = 0; j < w / 2; j++)
            {
                double sy = (i + 0.5) * 2 - 0.5, sx = (j + 0.5) * 2 - 0.5;
                int y0 = (int)Math.Floor(sy), x0 = (int)Math.Floor(sx);
                double sum = 0;
                for (int m = -1; m <= 2; m++)
                    for (int k = -1; k <= 2; k++)
                    {
                        int yy = Math.Min(Math.Max(y0 + m, 0), h - 1), xx = Math.Min(Math.Max(x0 + k, 0), w - 1);
                        sum += Cubic(sy - (y0 + m)) * Cubic(sx - (x0 + k)) * x[0, 0, yy, xx];
                    }
                Assert.Equal(sum, actual[0, 0, i, j], 10);
            }
    }

    // set_smoothness_loss: half the sum of the mean vertical and horizontal |Δflow| weighted by exp(−150²·mean_c Δimage²).
    [Fact(Timeout = 60000)]
    public async Task Smoothness_MatchesTheRelease()
    {
        await Task.Yield();
        var flow = Random(new[] { 1, 2, 4, 5 }, 7);
        var image = Random(new[] { 1, 3, 4, 5 }, 8, 0.01);
        double Term(bool vertical)
        {
            double sum = 0;
            int count = 0, hh = vertical ? 3 : 4, ww = vertical ? 5 : 4;
            for (int y = 0; y < hh; y++)
                for (int x = 0; x < ww; x++)
                {
                    int y2 = vertical ? y + 1 : y, x2 = vertical ? x : x + 1;
                    double g = 0;
                    for (int ch = 0; ch < 3; ch++) g += Math.Pow(image[0, ch, y2, x2] - image[0, ch, y, x], 2);
                    double weight = Math.Exp(-150.0 * 150.0 * g / 3);
                    for (int ch = 0; ch < 2; ch++) { sum += weight * Math.Abs(flow[0, ch, y2, x2] - flow[0, ch, y, x]); count++; }
                }
            return sum / count;
        }
        Assert.Equal(0.5 * (Term(true) + Term(false)), XvfiOps.Smoothness(Engine, flow, image, 150.0)[0], 10);
    }

    // VFInet is shared by every level, so the test depth may exceed the training depth.
    [Fact(Timeout = 240000)]
    public async Task Interpolate_SynthesizesAnyTimeAtAnySizeAndDepth()
    {
        await Task.Yield();
        var options = Small();
        options.TestDepth = 2;
        var model = Model(options);
        var frame0 = Random(new[] { 3, 20, 28 }, 9, 255.0);
        var frame1 = Random(new[] { 3, 20, 28 }, 10, 255.0);
        var middle = model.Interpolate(frame0, frame1);
        var early = model.Interpolate(frame0, frame1, 0.25);
        Assert.Equal(new[] { 3, 20, 28 }, middle.Shape.ToArray());
        Assert.All(middle.ToArray(), v => Assert.True(v >= 0 && v <= 255));
        Assert.False(middle.ToArray().SequenceEqual(early.ToArray()));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Interpolate(frame0, frame1, -0.5));
    }

    [Fact(Timeout = 240000)]
    public async Task Training_RejectsCropsOffTheDivide()
    {
        await Task.Yield();
        var model = Model(Small());
        Assert.Throws<ArgumentException>(() => model.Train(Random(new[] { 6, 24, 24 }, 11, 255.0), Random(new[] { 3, 24, 24 }, 12, 255.0)));
    }

    // The multi-scale reconstruction and smoothness losses reach the extractor, the shared down-sampling, every flow
    // stack and the refinement U-Net; read from the tape.
    [Fact(Timeout = 240000)]
    public async Task Loss_ReachesEveryWeight()
    {
        await Task.Yield();
        var model = Model(Small());
        var parameters = model.NetworkParameters.ToArray();
        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> gradients;
        using (var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<double>())
        {
            var loss = model.ReleaseLoss(Random(new[] { 1, 3, 32, 32 }, 13), Random(new[] { 1, 3, 32, 32 }, 14), Random(new[] { 1, 3, 32, 32 }, 15));
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
        model.Train(Random(new[] { 6, 32, 32 }, 16, 255.0), Random(new[] { 3, 32, 32 }, 17, 255.0));
        var after = model.NetworkParameters.SelectMany(t => t.ToArray()).ToArray();
        Assert.False(before.SequenceEqual(after));
        Assert.All(after, v => Assert.False(double.IsNaN(v) || double.IsInfinity(v)));
    }
}
