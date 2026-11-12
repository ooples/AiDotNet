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
/// Shi et al. 2022, Video Frame Interpolation Transformer (arXiv 2111.13817), and the release
/// (zhshi0816/Video-Frame-Interpolation-Transformer: model/VFIT_B.py, VFIT_S.py, Sep_STS_Layer.py, cupy_module/adacof.py,
/// main.py): the configuration and both variants' parameter counts, AdaCoF and its gradient against the CUDA kernels, the
/// shifted-window mask and relative position index against the release, the midpoint-only interpolation, the training
/// inputs, and tape gradients reaching every weight (#2397).
/// </summary>
public class VFITPaperTests
{
    private static VFITOptions Small() => new()
    {
        Seed = 1, Widths = new[] { 8, 8, 8, 8 }, StageDepths = new[] { 2, 1, 1, 1 }, StageHeads = new[] { 2, 2, 2, 2 }, SynthesisChannels = 4,
    };

    private static VFIT<double> Model(VFITOptions options) =>
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
    public async Task Defaults_AreTheReleaseVfitBAndTraining()
    {
        await Task.Yield();
        var o = new VFITOptions();
        Assert.Equal(new[] { 512, 256, 128, 64 }, o.Widths);
        Assert.Equal(new[] { 2, 2, 6, 2 }, o.StageDepths);
        Assert.Equal(new[] { 2, 4, 8, 16 }, o.StageHeads);
        Assert.Equal((4, 8, 4.0, 64, 5, 1), (o.NumInputFrames, o.WindowSize, o.MlpRatio, o.SynthesisChannels, o.KernelSize, o.Dilation));
        Assert.Equal((2e-4, 0.9, 0.999), (o.LearningRate, o.Beta1, o.Beta2));
        Assert.Equal(new[] { 646120, 969180, 1211475, 1373005, 1534535 }, o.LearningRateMilestones);
        Assert.Equal(new[] { 1e-4, 5e-5, 2.5e-5, 5e-6, 1e-6 }, o.LearningRateSteps);
    }

    // UNet_3D_3D instantiated with torch: VFIT-B 29,082,065 (the encoder 20,165,920, the decoder 5,751,232, the smoothing
    // 2,212,416 and the three SynBlocks 317,499 each) and VFIT-S 7,537,489.
    [Theory(Timeout = 240000)]
    [InlineData(new[] { 512, 256, 128, 64 }, 29082065)]
    [InlineData(new[] { 192, 128, 64, 32 }, 7537489)]
    public async Task ParameterCount_MatchesTheRelease(int[] widths, int expected)
    {
        await Task.Yield();
        var model = Model(new VFITOptions { Widths = widths });
        Assert.Equal(expected, model.NetworkParameters.Sum(t => t.Length));
    }

    // kernel_AdaCoF_updateOutput transcribed: A = (int)alpha truncates toward zero and every index is clamped.
    private static double AdaCofReference(Tensor<double> frame, Tensor<double> weight, Tensor<double> alpha, Tensor<double> beta, int n, int c, int i, int j, int size, int dilation)
    {
        int hin = frame.Shape[2], win = frame.Shape[3];
        int Clamp(int v, int m) => v < 0 ? 0 : v > m - 1 ? m - 1 : v;
        double output = 0;
        for (int k = 0; k < size; k++)
            for (int l = 0; l < size; l++)
            {
                double w = weight[n, k * size + l, i, j], a = alpha[n, k * size + l, i, j], b = beta[n, k * size + l, i, j];
                int A = (int)a, B = (int)b;
                int y0 = Clamp(i + k * dilation + A, hin), y1 = Clamp(i + k * dilation + A + 1, hin);
                int x0 = Clamp(j + l * dilation + B, win), x1 = Clamp(j + l * dilation + B + 1, win);
                double fa = a - A, fb = b - B;
                output += w * (frame[n, c, y0, x0] * (1 - fa) * (1 - fb) + frame[n, c, y1, x0] * fa * (1 - fb)
                    + frame[n, c, y0, x1] * (1 - fa) * fb + frame[n, c, y1, x1] * fa * fb);
            }
        return output;
    }

    [Theory(Timeout = 60000)]
    [InlineData(3, 1)]
    [InlineData(5, 2)]
    public async Task AdaCof_MatchesTheKernel(int size, int dilation)
    {
        await Task.Yield();
        int b = 2, c = 3, h = 6, w = 7, pad = (size - 1) * dilation;
        var frame = Random(new[] { b, c, h + pad, w + pad }, 1);
        var weight = Random(new[] { b, size * size, h, w }, 2, 2.0, -1.0);
        // Offsets on both sides of zero and past the frame, so truncation and clamping both matter.
        var alpha = Random(new[] { b, size * size, h, w }, 3, 12.0, -6.0);
        var beta = Random(new[] { b, size * size, h, w }, 4, 12.0, -6.0);
        var output = VfitOps.AdaCof(AiDotNetEngine.Current, frame, weight, alpha, beta, dilation);
        Assert.Equal(new[] { b, c, h, w }, output.Shape.ToArray());
        for (int n = 0; n < b; n++)
            for (int ch = 0; ch < c; ch++)
                for (int i = 0; i < h; i++)
                    for (int j = 0; j < w; j++)
                        Assert.Equal(AdaCofReference(frame, weight, alpha, beta, n, ch, i, j, size, dilation), output[n, ch, i, j], 12);
    }

    // kernel_AdaCoF_updateGradWeight and updateGradAlpha/Beta transcribed, against the tape's gradients of Σ G·out.
    [Fact(Timeout = 60000)]
    public async Task AdaCofGradient_MatchesTheKernelBackward()
    {
        await Task.Yield();
        int b = 1, c = 3, h = 5, w = 4, size = 3;
        var frame = Random(new[] { b, c, h + 2, w + 2 }, 5);
        var weight = Random(new[] { b, 9, h, w }, 6, 2.0, -1.0);
        var alpha = Random(new[] { b, 9, h, w }, 7, 8.0, -4.0);
        var beta = Random(new[] { b, 9, h, w }, 8, 8.0, -4.0);
        var upstream = Random(new[] { b, c, h, w }, 9, 2.0, -1.0);
        var engine = AiDotNetEngine.Current;
        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> gradients;
        using (var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<double>())
        {
            var output = VfitOps.AdaCof(engine, frame, weight, alpha, beta, 1);
            var loss = engine.ReduceSum(engine.TensorMultiply(output, upstream), new[] { 0, 1, 2, 3 }, false);
            gradients = tape.ComputeGradients(loss, sources: new[] { weight, alpha, beta });
        }
        int Clamp(int v, int m) => v < 0 ? 0 : v > m - 1 ? m - 1 : v;
        for (int t = 0; t < 9; t++)
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++)
                {
                    double a = alpha[0, t, i, j], bb = beta[0, t, i, j], wt = weight[0, t, i, j];
                    int A = (int)a, B = (int)bb, k = t / size, l = t % size;
                    int y0 = Clamp(i + k + A, h + 2), y1 = Clamp(i + k + A + 1, h + 2), x0 = Clamp(j + l + B, w + 2), x1 = Clamp(j + l + B + 1, w + 2);
                    double fa = a - A, fb = bb - B, gw = 0, ga = 0, gb = 0;
                    for (int ch = 0; ch < c; ch++)
                    {
                        double delta = upstream[0, ch, i, j];
                        double p00 = frame[0, ch, y0, x0], p10 = frame[0, ch, y1, x0], p01 = frame[0, ch, y0, x1], p11 = frame[0, ch, y1, x1];
                        gw += delta * (p00 * (1 - fa) * (1 - fb) + p10 * fa * (1 - fb) + p01 * (1 - fa) * fb + p11 * fa * fb);
                        ga += delta * wt * (-p00 * (1 - fb) + p10 * (1 - fb) - p01 * fb + p11 * fb);
                        gb += delta * wt * (-p00 * (1 - fa) - p10 * fa + p01 * (1 - fa) + p11 * fa);
                    }
                    Assert.Equal(gw, gradients[weight][0, t, i, j], 12);
                    Assert.Equal(ga, gradients[alpha][0, t, i, j], 12);
                    Assert.Equal(gb, gradients[beta][0, t, i, j], 12);
                }
    }

    // compute_mask run with torch on the release: the masked count, the sum of their flat indices, and the size; the last
    // case is a 4-row clip, where get_window_size shrinks the window to 4 rows and drops their shift.
    [Theory(Timeout = 60000)]
    [InlineData(4, 16, 16, 8, 8, 4, 4, 28672, 1015007232L)]
    [InlineData(4, 16, 24, 8, 8, 4, 4, 36864, 1988081664L)]
    [InlineData(4, 4, 16, 4, 8, 0, 4, 2048, 9436160L)]
    public async Task ShiftMask_MatchesTheRelease(int d, int h, int w, int wh, int ww, int sh, int sw, int count, long sum)
    {
        await Task.Yield();
        var mask = VfitOps.ShiftMask(d, h, w, wh, ww, sh, sw);
        Assert.Equal(d * h * w * wh * ww, mask.Length);
        var masked = Enumerable.Range(0, mask.Length).Where(i => mask[i] != 0).ToArray();
        Assert.All(masked, i => Assert.Equal(-100.0, mask[i]));
        Assert.Equal((count, sum), (masked.Length, masked.Sum(i => (long)i)));
    }

    // WindowAttention3D.relative_position_index from torch: its sum and its index-weighted sum.
    [Theory(Timeout = 60000)]
    [InlineData(1, 8, 8, 458752L, 1103219712L)]
    [InlineData(4, 1, 1, 48L, 420L)]
    [InlineData(1, 4, 8, 53248L, 32164608L)]
    public async Task RelativeIndex_MatchesTheRelease(int wd, int wh, int ww, long sum, long weighted)
    {
        await Task.Yield();
        var index = VfitOps.RelativeIndex(wd, wh, ww);
        Assert.Equal((sum, weighted), (index.Sum(v => (long)v), Enumerable.Range(0, index.Length).Sum(i => (long)i * index[i])));
    }

    // The release synthesizes the midpoint only, at any frame size (no padding).
    [Fact(Timeout = 240000)]
    public async Task Interpolate_IsTheMidpointAtAnySize()
    {
        await Task.Yield();
        var model = Model(Small());
        Assert.False(model.SupportsArbitraryTimestep);
        var frame0 = Random(new[] { 3, 20, 30 }, 10, 255.0);
        var frame1 = Random(new[] { 3, 20, 30 }, 11, 255.0);
        var middle = model.Interpolate(frame0, frame1, 0.5);
        Assert.Equal(new[] { 3, 20, 30 }, middle.Shape.ToArray());
        Assert.All(middle.ToArray(), v => Assert.False(double.IsNaN(v) || double.IsInfinity(v)));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Interpolate(frame0, frame1, 0.25));
    }

    // A training input is the four frames [12, H, W], or a pair with each frame repeated over its half; other widths are
    // rejected.
    [Fact(Timeout = 240000)]
    public async Task Training_TakesTheWindowOrAPair()
    {
        await Task.Yield();
        var model = Model(Small());
        var objective = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var target = Random(new[] { 3, 32, 32 }, 12, 255.0);
        var pair = Random(new[] { 6, 32, 32 }, 13, 255.0);
        var window = new Tensor<double>(new[] { 12, 32, 32 });
        int plane = 3 * 32 * 32;
        for (int i = 0; i < plane; i++)
        {
            window[i] = window[plane + i] = pair[i];
            window[2 * plane + i] = window[3 * plane + i] = pair[plane + i];
        }
        double fromPair = objective.EvaluateTrainingObjective(pair, target);
        Assert.False(double.IsNaN(fromPair) || double.IsInfinity(fromPair));
        Assert.Equal(fromPair, objective.EvaluateTrainingObjective(window, target));
        Assert.NotEqual(fromPair, objective.EvaluateTrainingObjective(Random(new[] { 12, 32, 32 }, 14, 255.0), target));
        Assert.Throws<ArgumentException>(() => objective.EvaluateTrainingObjective(Random(new[] { 9, 32, 32 }, 15, 255.0), target));
    }

    // The L1 loss reaches the encoder, every Sep-STS block (spatial and temporal attention and their bias tables), the
    // decoder and all three synthesis scales; read from the tape.
    [Fact(Timeout = 240000)]
    public async Task Loss_ReachesEveryWeight()
    {
        await Task.Yield();
        var model = Model(Small());
        var parameters = model.NetworkParameters.ToArray();
        var window = Random(new[] { 1, 12, 32, 32 }, 16);
        var target = Random(new[] { 1, 3, 32, 32 }, 17);
        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> gradients;
        using (var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<double>())
        {
            gradients = tape.ComputeGradients(model.ReleaseLoss(window, target), sources: parameters);
        }
        var unreached = Enumerable.Range(0, parameters.Length)
            .Where(i => !gradients.TryGetValue(parameters[i], out var g) || g.ToArray().All(v => v == 0.0)).ToArray();
        Assert.True(unreached.Length == 0, $"{unreached.Length} of {parameters.Length} weight tensors received no gradient: "
            + string.Join(", ", unreached.Select(i => $"{i} [{string.Join("x", parameters[i].Shape.ToArray())}]")));
    }

    [Fact(Timeout = 240000)]
    public async Task TrainingStep_UpdatesTheNetwork()
    {
        await Task.Yield();
        var model = Model(Small());
        var before = model.NetworkParameters.SelectMany(t => t.ToArray()).ToArray();
        model.Train(Random(new[] { 12, 32, 32 }, 18, 255.0), Random(new[] { 3, 32, 32 }, 19, 255.0));
        var after = model.NetworkParameters.SelectMany(t => t.ToArray()).ToArray();
        Assert.False(before.SequenceEqual(after));
        Assert.All(after, v => Assert.False(double.IsNaN(v) || double.IsInfinity(v)));
    }
}
