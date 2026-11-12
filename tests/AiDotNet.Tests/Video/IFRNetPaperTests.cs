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
/// Kong et al. 2022, IFRNet (arXiv 2205.14620), and the release (ltkong218/IFRNet: models/IFRNet.py, IFRNet_L.py,
/// IFRNet_S.py, loss.py, train_vimeo90k.py): the three configurations and their exact parameter counts, the padding,
/// the geometry loss against a brute-force transcription, arbitrary-time synthesis and a training step that reaches
/// every weight (#2397).
/// </summary>
public class IFRNetPaperTests
{
    private static IEngine Engine => AiDotNetEngine.Current;

    private static IFRNetOptions Small() => new() { Seed = 1, EncoderChannels = new[] { 4, 4, 4, 4 }, SideChannels = 2 };

    private static IFRNet<double> Model(IFRNetOptions options) =>
        new(new NeuralNetworkArchitecture<double>(inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 32, inputWidth: 32, inputDepth: 3, outputSize: 3), options);

    private static Tensor<double> Random(int[] shape, int seed, double scale = 1.0)
    {
        var t = new Tensor<double>(shape);
        var random = new System.Random(seed);
        for (int i = 0; i < t.Length; i++) t[i] = scale * random.NextDouble();
        return t;
    }

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreIFRNetAndTheReleaseTraining()
    {
        await Task.Yield();
        var o = new IFRNetOptions();
        Assert.Equal(new[] { 32, 48, 72, 96 }, o.EncoderChannels);
        Assert.Equal((32, 3), (o.SideChannels, o.FirstKernelSize));
        Assert.Equal((1e-4, 1e-5, 0.0, 0.01), (o.LearningRate, o.MinLearningRate, o.WeightDecay, o.GeometryWeight));
        Assert.Equal(300 * 2138, o.TotalIterations);
        var large = IFRNetOptions.Large();
        Assert.Equal(new[] { 64, 96, 144, 192 }, large.EncoderChannels);
        Assert.Equal((64, 7), (large.SideChannels, large.FirstKernelSize));
        var small = IFRNetOptions.Small();
        Assert.Equal(new[] { 24, 36, 54, 72 }, small.EncoderChannels);
        Assert.Equal(24, small.SideChannels);
    }

    // The release's modules counted layer by layer: 4.96M, 19.7M and 2.8M parameters as the paper's Table 1 reports.
    [Theory(Timeout = 240000)]
    [InlineData(0, 4959692)]
    [InlineData(1, 19699460)]
    [InlineData(2, 2803610)]
    public async Task ParameterCount_MatchesTheRelease(int variant, int expected)
    {
        await Task.Yield();
        var options = variant switch { 1 => IFRNetOptions.Large(), 2 => IFRNetOptions.Small(), _ => new IFRNetOptions() };
        var model = Model(options);
        Assert.Equal(expected, model.NetworkParameters.Sum(t => t.Length));
    }

    [Fact(Timeout = 60000)]
    public async Task Padding_IsZerosToMultiplesOf16()
    {
        await Task.Yield();
        Assert.Equal((14, 12), IFRNet<double>.Padding(100, 130));
        Assert.Equal((0, 0), IFRNet<double>.Padding(64, 96));
    }

    // loss.Geometry(3) transcribed: each channel's 3x3 census zero-padded by 1, δ²/(0.1 + δ²) averaged over channels and
    // taps, a one-pixel border mask, the mean over every pixel.
    [Fact(Timeout = 60000)]
    public async Task Geometry_MatchesTheRelease()
    {
        await Task.Yield();
        var x = Random(new[] { 2, 3, 6, 7 }, 1);
        var y = Random(new[] { 2, 3, 6, 7 }, 2);
        int b = 2, c = 3, h = 6, w = 7;
        double Value(Tensor<double> f, int n, int ch, int yy, int xx) => yy < 0 || yy >= h || xx < 0 || xx >= w ? 0.0 : f[n, ch, yy, xx];
        double Census(Tensor<double> f, int n, int ch, int yy, int xx, int dy, int dx)
        {
            double d = Value(f, n, ch, yy + dy, xx + dx) - f[n, ch, yy, xx];
            return d / Math.Sqrt(0.81 + d * d);
        }
        double total = 0;
        for (int n = 0; n < b; n++)
            for (int yy = 1; yy < h - 1; yy++)
                for (int xx = 1; xx < w - 1; xx++)
                {
                    double sum = 0;
                    for (int ch = 0; ch < c; ch++)
                        for (int dy = -1; dy <= 1; dy++)
                            for (int dx = -1; dx <= 1; dx++)
                            {
                                double delta = Census(x, n, ch, yy, xx, dy, dx) - Census(y, n, ch, yy, xx, dy, dx);
                                sum += delta * delta / (0.1 + delta * delta);
                            }
                    total += sum / (c * 9);
                }
        Assert.Equal(total / (b * h * w), IfrOps.Geometry(Engine, x, y)[0], 10);
        Assert.Equal(0.0, IfrOps.Geometry(Engine, x, x)[0], 12);
    }

    [Fact(Timeout = 240000)]
    public async Task Interpolate_SynthesizesAnyTimeAtAnySize()
    {
        await Task.Yield();
        var model = Model(Small());
        var frame0 = Random(new[] { 3, 20, 28 }, 3, 255.0);
        var frame1 = Random(new[] { 3, 20, 28 }, 4, 255.0);
        var middle = model.Interpolate(frame0, frame1);
        var early = model.Interpolate(frame0, frame1, 0.25);
        Assert.Equal(new[] { 3, 20, 28 }, middle.Shape.ToArray());
        Assert.All(middle.ToArray(), v => Assert.True(v >= 0 && v <= 255));
        Assert.False(middle.ToArray().SequenceEqual(early.ToArray()));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Interpolate(frame0, frame1, 1.1));
    }

    // Without weight decay a weight moves only on a nonzero gradient: the reconstruction and geometry losses reach every
    // convolution of the encoder and the decoders. A PReLU slope's gradient is exactly zero when its channels never go
    // negative, which at these 2- and 4-channel widths happens to some (as it would in PyTorch), so a slope may stay
    // put only if the loss provably does not depend on it.
    [Fact(Timeout = 240000)]
    public async Task TrainingStep_ReachesEveryWeight()
    {
        await Task.Yield();
        var model = Model(Small());
        var input = Random(new[] { 6, 32, 32 }, 5, 255.0);
        var target = Random(new[] { 3, 32, 32 }, 6, 255.0);
        var before = model.NetworkParameters.Select(t => t.ToArray()).ToArray();
        bool IsSlope(int i) => before[i].All(v => v == 0.25);
        var objective = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        double loss = objective.EvaluateTrainingObjective(input, target);
        bool LossIgnores(int i)
        {
            var weight = model.NetworkParameters[i];
            for (int k = 0; k < weight.Length; k++) weight[k] = 3.0;
            bool ignores = objective.EvaluateTrainingObjective(input, target) == loss;
            for (int k = 0; k < weight.Length; k++) weight[k] = before[i][k];
            return ignores;
        }
        var inactiveSlopes = Enumerable.Range(0, before.Length).Where(i => IsSlope(i) && LossIgnores(i)).ToHashSet();

        model.Train(input, target);
        var after = model.NetworkParameters.Select(t => t.ToArray()).ToArray();
        var frozen = Enumerable.Range(0, before.Length).Where(i => before[i].SequenceEqual(after[i]) && !inactiveSlopes.Contains(i)).ToArray();
        Assert.True(frozen.Length == 0, $"{frozen.Length} of {before.Length} weight tensors received no gradient: {string.Join(", ", frozen)}");
        Assert.True(inactiveSlopes.Count < before.Count(t => t.All(v => v == 0.25)) / 4, $"{inactiveSlopes.Count} PReLU slopes never see a negative input");
    }
}
