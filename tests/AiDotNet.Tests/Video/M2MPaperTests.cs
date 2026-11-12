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
/// Hu, Niklaus, Sclaroff and Saenko 2022, M2M (arXiv 2204.03513), and the release (feinanshan/M2M_VFI: m2m.py,
/// pwcnet.py, costvol.py, loss.py, optimizer.py): the configuration and its parameter count, the padding, the cost
/// volume, the many-to-many splatting and the ternary loss against brute-force transcriptions, arbitrary-time synthesis,
/// the flow estimator held at a zero learning rate, and tape gradients reaching every weight (#2397).
/// </summary>
public class M2MPaperTests
{
    private static IEngine Engine => AiDotNetEngine.Current;

    private static M2MOptions Small() => new()
    {
        Seed = 1, BaseChannels = 2, NumBranches = 2, CubeRank = 2, Ratio = 2, FlowChannels = 4, FlowDecoderChannels = new[] { 4, 4, 4, 4, 4 },
    };

    private static M2M<double> Model(M2MOptions options) =>
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
    public async Task Defaults_AreTheReleaseM2MPwcAndTraining()
    {
        await Task.Yield();
        var o = new M2MOptions();
        Assert.Equal((16, 4, 16, 4, 32), (o.BaseChannels, o.NumBranches, o.CubeRank, o.Ratio, o.FlowChannels));
        Assert.Equal(new[] { 128, 128, 96, 64, 32 }, o.FlowDecoderChannels);
        Assert.Equal((1e-4, 1e-4, 2000, 900000, 1.0), (o.LearningRate, o.WeightDecay, o.WarmupSteps, o.TotalSteps, o.MaxGradientNorm));
        Assert.Equal((0.0, 1.0, 1.0), (o.FlowLearningRateMultiplier, o.CharbonnierWeight, o.TernaryWeight));
    }

    // The release's modules counted layer by layer: PWC-Net 2,388,305, the Motion Refinement Network 5,222,409 and α, the
    // paper's 7.61M.
    [Fact(Timeout = 240000)]
    public async Task ParameterCount_MatchesTheRelease()
    {
        await Task.Yield();
        var model = Model(new M2MOptions());
        Assert.Equal(7610715, model.NetworkParameters.Sum(t => t.Length));
    }

    // M2M_PWC.forward: replicate padding on the right and bottom to multiples of 16·ratio.
    [Fact(Timeout = 60000)]
    public async Task Padding_IsTheReleaseMultiple()
    {
        await Task.Yield();
        Assert.Equal((62, 28), Model(new M2MOptions()).Padding(100, 130));
        Assert.Equal((0, 0), Model(new M2MOptions()).Padding(128, 192));
        Assert.Equal((14, 0), Model(Small()).Padding(64, 50));
    }

    // costvol_func: the channel mean of |one − two(y + dy, x + dx)| for dy, dx in −4..4, |one| outside.
    [Fact(Timeout = 60000)]
    public async Task CostVolume_MatchesTheRelease()
    {
        await Task.Yield();
        int c = 3, h = 6, w = 7;
        var one = Random(new[] { 1, c, h, w }, 1, 2.0, -1.0);
        var two = Random(new[] { 1, c, h, w }, 2, 2.0, -1.0);
        var actual = M2mOps.CostVolume(Engine, one, two);
        Assert.Equal(new[] { 1, 81, h, w }, actual.Shape.ToArray());
        for (int dy = -4; dy <= 4; dy++)
            for (int dx = -4; dx <= 4; dx++)
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        int yy = y + dy, xx = x + dx;
                        bool inside = yy >= 0 && yy < h && xx >= 0 && xx < w;
                        double sum = 0;
                        for (int ch = 0; ch < c; ch++) sum += Math.Abs(one[0, ch, y, x] - (inside ? two[0, ch, yy, xx] : 0.0));
                        Assert.Equal(sum / c, actual[0, (dy + 4) * 9 + dx + 4, y, x], 10);
                    }
    }

    // forwarp_mframe_mask: every branch and direction splats value·td·exp(clip(Z, −20, 20)) and the weight bilinearly;
    // the sums are divided by the summed weights, each direction's plus 1e-7, and sums below 1e-5 are holes.
    [Fact(Timeout = 60000)]
    public async Task ManyToManySplatting_MatchesTheRelease()
    {
        await Task.Yield();
        int c = 2, h = 5, w = 6, branches = 2;
        var images0 = Enumerable.Range(0, branches).Select(b => Random(new[] { 1, c, h, w }, 10 + b)).ToArray();
        var images1 = Enumerable.Range(0, branches).Select(b => Random(new[] { 1, c, h, w }, 20 + b)).ToArray();
        var flows0 = Enumerable.Range(0, branches).Select(b => Random(new[] { 1, 2, h, w }, 30 + b, 4.0, -2.0)).ToArray();
        var flows1 = Enumerable.Range(0, branches).Select(b => Random(new[] { 1, 2, h, w }, 40 + b, 4.0, -2.0)).ToArray();
        var metrics0 = Enumerable.Range(0, branches).Select(b => Random(new[] { 1, 1, h, w }, 50 + b, 50.0, -25.0)).ToArray();
        var metrics1 = Enumerable.Range(0, branches).Select(b => Random(new[] { 1, 1, h, w }, 60 + b, 50.0, -25.0)).ToArray();
        double time0 = 0.7, time1 = 0.3;
        var values = new double[c * h * w];
        var weights = new double[h * w];
        void Splat(Tensor<double> image, Tensor<double> flow, Tensor<double> metric, double td)
        {
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    double weight = td * Math.Exp(Math.Min(Math.Max(metric[0, 0, y, x], -20.0), 20.0));
                    double fx = x + flow[0, 0, y, x], fy = y + flow[0, 1, y, x];
                    int x0 = (int)Math.Floor(fx), y0 = (int)Math.Floor(fy);
                    double ax = fx - x0, ay = fy - y0;
                    foreach (var (tx, ty, bilinear) in new[] { (x0, y0, (1 - ax) * (1 - ay)), (x0 + 1, y0, ax * (1 - ay)), (x0, y0 + 1, (1 - ax) * ay), (x0 + 1, y0 + 1, ax * ay) })
                    {
                        if (tx < 0 || tx >= w || ty < 0 || ty >= h) continue;
                        weights[ty * w + tx] += weight * bilinear;
                        for (int ch = 0; ch < c; ch++) values[(ch * h + ty) * w + tx] += weight * bilinear * image[0, ch, y, x];
                    }
                }
        }
        for (int b = 0; b < branches; b++)
        {
            Splat(images0[b], flows0[b], metrics0[b], time0);
            Splat(images1[b], flows1[b], metrics1[b], time1);
        }
        var (output, holes) = M2mOps.ForwardWarpFrames(Engine, images0, flows0, time0, images1, flows1, time1, metrics0, metrics1);
        for (int p = 0; p < h * w; p++)
        {
            double normalize = weights[p] + 2 * branches * 1e-7;
            Assert.Equal(normalize < 1e-5 ? 1.0 : 0.0, holes[p]);
            for (int ch = 0; ch < c; ch++) Assert.Equal(values[ch * h * w + p] / normalize, output[ch * h * w + p], 8);
        }

        // Everything pushed out of the image leaves only holes.
        var away = Enumerable.Range(0, branches).Select(_ => { var f = new Tensor<double>(new[] { 1, 2, h, w }); f.Fill(100.0); return f; }).ToArray();
        var (_, allHoles) = M2mOps.ForwardWarpFrames(Engine, images0, away, time0, images1, away, time1, metrics0, metrics1);
        Assert.All(allHoles.ToArray(), v => Assert.Equal(1.0, v));
    }

    // loss.Ternary: rgb2gray (0.2989, 0.5870, 0.1140), 7x7 zero-padded differences d / √(0.81 + d²), the tap mean of
    // δ²/(0.1 + δ²), zero on the one-pixel border, averaged over every pixel.
    [Fact(Timeout = 60000)]
    public async Task Ternary_MatchesTheRelease()
    {
        await Task.Yield();
        int h = 9, w = 10;
        var a = Random(new[] { 1, 3, h, w }, 70);
        var b = Random(new[] { 1, 3, h, w }, 71);
        double[,] Gray(Tensor<double> x)
        {
            var g = new double[h, w];
            for (int y = 0; y < h; y++)
                for (int xx = 0; xx < w; xx++)
                    g[y, xx] = 0.2989 * x[0, 0, y, xx] + 0.5870 * x[0, 1, y, xx] + 0.1140 * x[0, 2, y, xx];
            return g;
        }
        double Census(double[,] g, int y, int x, int dy, int dx)
        {
            int yy = y + dy, xx = x + dx;
            double d = (yy >= 0 && yy < h && xx >= 0 && xx < w ? g[yy, xx] : 0.0) - g[y, x];
            return d / Math.Sqrt(0.81 + d * d);
        }
        var ga = Gray(a);
        var gb = Gray(b);
        double total = 0;
        for (int y = 1; y < h - 1; y++)
            for (int x = 1; x < w - 1; x++)
            {
                double sum = 0;
                for (int dy = -3; dy <= 3; dy++)
                    for (int dx = -3; dx <= 3; dx++)
                    {
                        double delta = Census(ga, y, x, dy, dx) - Census(gb, y, x, dy, dx);
                        sum += delta * delta / (0.1 + delta * delta);
                    }
                total += sum / 49;
            }
        Assert.Equal(total / (h * w), M2mOps.Ternary(Engine, a, b)[0], 10);
    }

    // One motion estimate serves every time; the frame size need not be a multiple of the padding.
    [Fact(Timeout = 240000)]
    public async Task Interpolate_SynthesizesAnyTimeAtAnySize()
    {
        await Task.Yield();
        var model = Model(Small());
        var frame0 = Random(new[] { 3, 20, 28 }, 80, 255.0);
        var frame1 = Random(new[] { 3, 20, 28 }, 81, 255.0);
        var middle = model.Interpolate(frame0, frame1);
        var early = model.Interpolate(frame0, frame1, 0.25);
        Assert.Equal(new[] { 3, 20, 28 }, middle.Shape.ToArray());
        Assert.All(middle.ToArray(), v => Assert.False(double.IsNaN(v) || double.IsInfinity(v)));
        Assert.False(middle.ToArray().SequenceEqual(early.ToArray()));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Interpolate(frame0, frame1, 1.5));
    }

    // The Charbonnier and ternary losses reach PWC-Net, the Motion Refinement Network and α; read from the tape. The only
    // exemptions are PReLU slopes whose input is never negative, proven by the loss not changing when the slope is set to
    // 3: PWC-Net's cost-volume activations always (a mean of absolute differences), and a few more at CI width.
    [Fact(Timeout = 240000)]
    public async Task Loss_ReachesEveryWeight()
    {
        await Task.Yield();
        var model = Model(Small());
        var parameters = model.NetworkParameters.ToArray();
        var frame0 = Random(new[] { 1, 3, 32, 32 }, 82);
        var frame1 = Random(new[] { 1, 3, 32, 32 }, 83);
        var target = Random(new[] { 1, 3, 32, 32 }, 84);
        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> gradients;
        using (var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<double>())
        {
            gradients = tape.ComputeGradients(model.ReleaseLoss(frame0, frame1, target), sources: parameters);
        }
        double Loss()
        {
            using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<double>();
            return model.ReleaseLoss(frame0, frame1, target)[0];
        }
        double loss = Loss();
        bool IsSlope(int i) => parameters[i].Length == 1 && parameters[i][0] == 0.25;
        bool LossIgnores(int i)
        {
            parameters[i][0] = 3.0;
            bool ignores = Loss() == loss;
            parameters[i][0] = 0.25;
            return ignores;
        }
        var unreached = Enumerable.Range(0, parameters.Length)
            .Where(i => !gradients.TryGetValue(parameters[i], out var g) || g.ToArray().All(v => v == 0.0)).ToArray();
        var unexplained = unreached.Where(i => !(IsSlope(i) && LossIgnores(i))).ToArray();
        Assert.True(unexplained.Length == 0, $"{unexplained.Length} of {parameters.Length} weight tensors received no gradient: "
            + string.Join(", ", unexplained.Select(i => $"{i} [{string.Join("x", parameters[i].Shape.ToArray())}]")));
        int slopes = Enumerable.Range(0, parameters.Length).Count(IsSlope);
        Assert.True(unreached.Length >= 5 && unreached.Length < slopes / 4, $"{unreached.Length} of {slopes} single PReLU slopes never see a negative input");
    }

    // get_params: PWC-Net at lr·0.0, so AdamW neither steps nor decays it; the rest trains. A positive multiple trains it.
    [Theory(Timeout = 240000)]
    [InlineData(0.0)]
    [InlineData(1.0)]
    public async Task TrainingStep_HoldsTheFlowEstimatorAtItsMultiple(double multiplier)
    {
        await Task.Yield();
        var options = Small();
        options.FlowLearningRateMultiplier = multiplier;
        options.WarmupSteps = 0;
        var model = Model(options);
        var parameters = model.NetworkParameters.ToArray();
        var flow = model.FlowParameters;
        var isFlow = parameters.Select(p => flow.Any(f => ReferenceEquals(f, p))).ToArray();
        Assert.Equal(flow.Count, isFlow.Count(x => x));
        var before = parameters.Select(t => t.ToArray()).ToArray();
        model.Train(Random(new[] { 6, 32, 32 }, 85, 255.0), Random(new[] { 3, 32, 32 }, 86, 255.0));
        var after = model.NetworkParameters.Select(t => t.ToArray()).ToArray();
        Assert.All(after.SelectMany(v => v), v => Assert.False(double.IsNaN(v) || double.IsInfinity(v)));
        bool flowMoved = Enumerable.Range(0, before.Length).Any(i => isFlow[i] && !before[i].SequenceEqual(after[i]));
        bool restMoved = Enumerable.Range(0, before.Length).Any(i => !isFlow[i] && !before[i].SequenceEqual(after[i]));
        Assert.Equal(multiplier > 0, flowMoved);
        Assert.True(restMoved);
    }
}
