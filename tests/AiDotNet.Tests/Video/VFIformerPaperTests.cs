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
/// Lu et al. 2022, Video Frame Interpolation with Transformer (arXiv 2205.07230), and the release (dvlab-research/VFIformer:
/// VFIformer_arch.py, transformer_layers.py, trainer.py): the configuration and its parameter count, the padding, the
/// unfold, the shifted-window and cross-scale masks, the coarse windows, the midpoint-only interpolation, the training
/// inputs, and tape gradients reaching every weight (#2397).
/// </summary>
public class VFIformerPaperTests
{
    private static VFIformerOptions Small() => new()
    {
        Seed = 1, FlowWidths = new[] { 4, 4, 4 }, RefineChannels = 2, EmbedDim = 8, WindowSize = 4, NumHeads = 2,
        Depths = new[] { 1, 1, 1, 1 }, WarmupSteps = 0,
    };

    private static VFIformer<double> Model(VFIformerOptions options) =>
        new(new NeuralNetworkArchitecture<double>(inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 32, inputWidth: 32, inputDepth: 3, outputSize: 3), options);

    private static Tensor<double> Random(int[] shape, int seed, double scale = 1.0, double offset = 0.0)
    {
        var t = new Tensor<double>(shape);
        var random = new System.Random(seed);
        for (int i = 0; i < t.Length; i++) t[i] = offset + scale * random.NextDouble();
        return t;
    }

    private static Tensor<double> Concat(params Tensor<double>[] parts)
    {
        int h = parts[0].Shape[1], w = parts[0].Shape[2];
        var result = new Tensor<double>(new[] { parts.Sum(p => p.Shape[0]), h, w });
        int offset = 0;
        foreach (var part in parts)
        {
            for (int i = 0; i < part.Length; i++) result[offset + i] = part[i];
            offset += part.Length;
        }
        return result;
    }

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTheReleaseVfiformerAndTraining()
    {
        await Task.Yield();
        var o = new VFIformerOptions();
        Assert.Equal(new[] { 240, 150, 90 }, o.FlowWidths);
        Assert.Equal((24, 160, 8, 2, 2.0, 0.1), (o.RefineChannels, o.EmbedDim, o.WindowSize, o.NumHeads, o.MlpRatio, o.DropPathRate));
        Assert.Equal(new[] { 3, 3, 3, 1 }, o.Depths);
        Assert.Equal(new[] { false, true, false, true, false, true, false, false }, o.CrossScaleAttention.SelectMany(g => g).ToArray());
        Assert.Equal((1e-4, 1e-4, 2000, 641400), (o.LearningRate, o.WeightDecay, o.WarmupSteps, o.TotalSteps));
        Assert.Equal((1.0, 1.0, 0.01), (o.L1Weight, o.TernaryWeight, o.FlowWeight));
    }

    // The release's VFIformer instantiated with torch: IFNet 6,804,432, FlowRefineNet_Multis 4,411,668, the fuse block
    // 26,016 and the transformer 12,924,814.
    [Fact(Timeout = 240000)]
    public async Task ParameterCount_MatchesTheRelease()
    {
        await Task.Yield();
        var model = Model(new VFIformerOptions());
        Assert.Equal(24166930, model.NetworkParameters.Sum(t => t.Length));
    }

    // The demo pads zeros around the centre to multiples of 8·window (32 for a 4x4 window).
    [Fact(Timeout = 60000)]
    public async Task Padding_IsCentredOnTheMultiple()
    {
        await Task.Yield();
        var model = Model(Small());
        Assert.Equal((14, 14, 15, 15), model.Padding(100, 130));
        Assert.Equal((6, 6, 2, 2), model.Padding(20, 28));
        Assert.Equal((0, 0, 0, 0), model.Padding(64, 96));
    }

    [Fact(Timeout = 60000)]
    public async Task Unfold3_IsTheZeroPaddedNeighbourhood()
    {
        await Task.Yield();
        var x = Random(new[] { 2, 3, 5, 6 }, 1);
        var u = VfiformerOps.Unfold3(AiDotNetEngine.Current, x);
        Assert.Equal(new[] { 2, 3, 9, 5, 6 }, u.Shape.ToArray());
        for (int b = 0; b < 2; b++)
            for (int c = 0; c < 3; c++)
                for (int k = 0; k < 9; k++)
                    for (int y = 0; y < 5; y++)
                        for (int xx = 0; xx < 6; xx++)
                        {
                            int sy = y + k / 3 - 1, sx = xx + k % 3 - 1;
                            double expected = sy < 0 || sy >= 5 || sx < 0 || sx >= 6 ? 0.0 : x[b, c, sy, sx];
                            Assert.Equal(expected, u[b, c, k, y, xx]);
                        }
    }

    // calculate_mask and calculate_mask2 run with torch on the release's TFL: the number of masked entries and the sum of
    // their flat indices.
    [Theory(Timeout = 60000)]
    [InlineData(16, 16, 8, 7168, 77591040L, 11200, 102377184L)]
    [InlineData(32, 24, 8, 13312, 450881024L, 21952, 685919712L)]
    [InlineData(16, 24, 4, 1216, 5643680L, 2044, 8829360L)]
    public async Task Masks_MatchTheRelease(int h, int w, int window, int shiftCount, long shiftSum, int crossCount, long crossSum)
    {
        await Task.Yield();
        (int Count, long Sum) Digest(double[] mask)
        {
            var masked = Enumerable.Range(0, mask.Length).Where(i => mask[i] != 0).ToArray();
            Assert.All(masked, i => Assert.Equal(-100.0, mask[i]));
            return (masked.Length, masked.Sum(i => (long)i));
        }
        Assert.Equal((shiftCount, shiftSum), Digest(VfiformerOps.ShiftMask(h, w, window, window / 2)));
        Assert.Equal((crossCount, crossSum), Digest(VfiformerOps.CrossMask(h, w, window, window / 2)));
    }

    // The coarse windows are F.unfold(window, stride window/2) of the map halved (an exact 2x2 mean) and reflect-padded
    // by window/4, one per window of the full map, row-major.
    [Fact(Timeout = 60000)]
    public async Task CoarseWindows_AreTheUnfoldedHalvedMap()
    {
        await Task.Yield();
        int b = 2, h = 16, w = 24, c = 3, window = 8, half = window / 2, pad = window / 4, hd = h / 2, wd = w / 2;
        var x = Random(new[] { b, h, w, c }, 2);
        var windows = VfiformerOps.CoarseWindows(AiDotNetEngine.Current, x, window);
        int nh = h / window, nw = w / window;
        Assert.Equal(new[] { b * nh * nw, window * window, c }, windows.Shape.ToArray());
        int Reflect(int i, int n) => i < 0 ? -i : i >= n ? 2 * (n - 1) - i : i;
        for (int n = 0; n < b; n++)
            for (int wy = 0; wy < nh; wy++)
                for (int wx = 0; wx < nw; wx++)
                    for (int j = 0; j < window * window; j++)
                        for (int ch = 0; ch < c; ch++)
                        {
                            int y = Reflect(wy * half + j / window - pad, hd), xx = Reflect(wx * half + j % window - pad, wd);
                            double expected = (x[n, 2 * y, 2 * xx, ch] + x[n, 2 * y, 2 * xx + 1, ch] + x[n, 2 * y + 1, 2 * xx, ch]
                                + x[n, 2 * y + 1, 2 * xx + 1, ch]) / 4;
                            Assert.Equal(expected, windows[(n * nh + wy) * nw + wx, j, ch], 12);
                        }
    }

    // The release synthesizes the midpoint only.
    [Fact(Timeout = 240000)]
    public async Task Interpolate_IsTheMidpoint()
    {
        await Task.Yield();
        var model = Model(Small());
        Assert.False(model.SupportsArbitraryTimestep);
        var frame0 = Random(new[] { 3, 20, 28 }, 3, 255.0);
        var frame1 = Random(new[] { 3, 20, 28 }, 4, 255.0);
        var middle = model.Interpolate(frame0, frame1, 0.5);
        Assert.Equal(new[] { 3, 20, 28 }, middle.Shape.ToArray());
        Assert.All(middle.ToArray(), v => Assert.True(v >= 0 && v <= 255));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Interpolate(frame0, frame1, 0.3));
    }

    // A training input is the pair, optionally followed by the flows from the midpoint to both frames (adding their
    // end-point error); other widths and crops off the multiple of 8·window are rejected.
    [Fact(Timeout = 240000)]
    public async Task Training_TakesThePairAndOptionalFlows()
    {
        await Task.Yield();
        var model = Model(Small());
        var objective = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var pair = Random(new[] { 6, 32, 32 }, 5, 255.0);
        var target = Random(new[] { 3, 32, 32 }, 6, 255.0);
        var flows = Random(new[] { 4, 32, 32 }, 7, 4.0, -2.0);
        double plain = objective.EvaluateTrainingObjective(pair, target);
        double flowed = objective.EvaluateTrainingObjective(Concat(pair, flows), target);
        Assert.All(new[] { plain, flowed }, v => Assert.False(double.IsNaN(v) || double.IsInfinity(v)));
        Assert.True(flowed > plain);
        Assert.Throws<ArgumentException>(() => objective.EvaluateTrainingObjective(Concat(pair, Random(new[] { 2, 32, 32 }, 8)), target));
        Assert.Throws<ArgumentException>(() => model.Train(Random(new[] { 6, 48, 48 }, 9, 255.0), Random(new[] { 3, 48, 48 }, 10, 255.0)));
    }

    // The L1, ternary and flow losses reach IFNet, the flow refinement, the fuse block and the transformer; read from the
    // tape. A PReLU slope is exempt only when setting it to 3 leaves the loss unchanged (it never sees a negative).
    [Fact(Timeout = 240000)]
    public async Task Loss_ReachesEveryWeight()
    {
        await Task.Yield();
        var model = Model(Small());
        var parameters = model.NetworkParameters.ToArray();
        var frame0 = Random(new[] { 1, 3, 32, 32 }, 11);
        var frame1 = Random(new[] { 1, 3, 32, 32 }, 12);
        var target = Random(new[] { 1, 3, 32, 32 }, 13);
        var flows = Random(new[] { 1, 4, 32, 32 }, 14, 4.0, -2.0);
        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> gradients;
        using (var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<double>())
        {
            gradients = tape.ComputeGradients(model.ReleaseLoss(frame0, frame1, target, flows, false), sources: parameters);
        }
        double Loss()
        {
            using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<double>();
            return model.ReleaseLoss(frame0, frame1, target, flows, false)[0];
        }
        double loss = Loss();
        bool IsSlope(int i) => parameters[i].ToArray().All(v => v == 0.25);
        bool LossIgnores(int i)
        {
            var saved = parameters[i].ToArray();
            for (int k = 0; k < saved.Length; k++) parameters[i][k] = 3.0;
            bool ignores = Loss() == loss;
            for (int k = 0; k < saved.Length; k++) parameters[i][k] = saved[k];
            return ignores;
        }
        var unreached = Enumerable.Range(0, parameters.Length)
            .Where(i => !gradients.TryGetValue(parameters[i], out var g) || g.ToArray().All(v => v == 0.0)).ToArray();
        var unexplained = unreached.Where(i => !(IsSlope(i) && LossIgnores(i))).ToArray();
        Assert.True(unexplained.Length == 0, $"{unexplained.Length} of {parameters.Length} weight tensors received no gradient: "
            + string.Join(", ", unexplained.Select(i => $"{i} [{string.Join("x", parameters[i].Shape.ToArray())}]")));
        int slopes = Enumerable.Range(0, parameters.Length).Count(IsSlope);
        Assert.True(unreached.Length < Math.Max(1, slopes / 4), $"{unreached.Length} of {slopes} PReLU slopes never see a negative input");
    }

    [Fact(Timeout = 240000)]
    public async Task TrainingStep_UpdatesTheNetwork()
    {
        await Task.Yield();
        var model = Model(Small());
        var before = model.NetworkParameters.SelectMany(t => t.ToArray()).ToArray();
        model.Train(Random(new[] { 6, 32, 32 }, 15, 255.0), Random(new[] { 3, 32, 32 }, 16, 255.0));
        var after = model.NetworkParameters.SelectMany(t => t.ToArray()).ToArray();
        Assert.False(before.SequenceEqual(after));
        Assert.All(after, v => Assert.False(double.IsNaN(v) || double.IsInfinity(v)));
    }
}
