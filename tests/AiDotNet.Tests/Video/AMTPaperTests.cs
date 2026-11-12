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
/// Li et al. 2023, AMT (arXiv 2304.09790), and the release (MCG-NKU/AMT: networks/AMT-S.py, AMT-L.py,
/// networks/blocks/raft.py, losses/loss.py, utils/utils.py, cfgs/AMT-S.yaml): the AMT-S and AMT-L configurations,
/// InputPadder, the bidirectional correlation and RAFT's lookup window against brute-force transcriptions, the ternary
/// loss, arbitrary-time synthesis and a training step (#2397).
/// </summary>
public class AMTPaperTests
{
    private static IEngine Engine => AiDotNetEngine.Current;

    private static AMTOptions SmallOptions() => new()
    {
        Seed = 1,
        Channels = new[] { 4, 4, 4, 4 },
        SkipChannels = 2,
        NumFlowFields = 2,
        NumCorrelationLevels = 3,
        CorrelationRadius = 1,
        FeatureEncoderWidths = new[] { 4, 4, 4 },
        FeatureDim = 4,
        UpdateHiddenDim = 4,
        UpdateFlowDim = 4,
        UpdateCorrelationDim = 4,
        UpdateFusionDim = 4,
    };

    private static AMT<double> Model(AMTOptions options) =>
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
    public async Task Defaults_AreAmtSAndTheReleaseTraining()
    {
        await Task.Yield();
        var o = new AMTOptions();
        Assert.Equal(new[] { 20, 32, 44, 56 }, o.Channels);
        Assert.Equal((20, 3, 4, 3), (o.SkipChannels, o.NumFlowFields, o.NumCorrelationLevels, o.CorrelationRadius));
        Assert.Equal(new[] { 32, 64, 96 }, o.FeatureEncoderWidths);
        Assert.Equal((84, false), (o.FeatureDim, o.LargeEncoders));
        Assert.Equal((76, 20, 64, 0, 68, 3), (o.UpdateHiddenDim, o.UpdateFlowDim, o.UpdateCorrelationDim, o.UpdateSecondCorrelationDim, o.UpdateFusionDim, o.CombineKernelSize));
        Assert.Equal((2e-4, 2e-5, 0.0, 1.0), (o.LearningRate, o.MinLearningRate, o.WeightDecay, o.TernaryWeight));
        Assert.Equal(300 * 2138, o.TotalIterations);

        var l = AMTOptions.AmtL();
        Assert.Equal(new[] { 48, 64, 72, 128 }, l.Channels);
        Assert.Equal((48, 5, true, 128), (l.SkipChannels, l.NumFlowFields, l.LargeEncoders, l.FeatureDim));
        Assert.Equal((128, 48, 256, 160, 124, 7), (l.UpdateHiddenDim, l.UpdateFlowDim, l.UpdateCorrelationDim, l.UpdateSecondCorrelationDim, l.UpdateFusionDim, l.CombineKernelSize));
    }

    // utils.InputPadder(divisor=16): centered, the extra pixel on the right and bottom.
    [Fact(Timeout = 60000)]
    public async Task Padding_IsInputPadder()
    {
        await Task.Yield();
        Assert.Equal((7, 7, 6, 6), AMT<double>.Padding(100, 130));
        Assert.Equal((0, 0, 0, 0), AMT<double>.Padding(64, 64));
        Assert.Equal((0, 1, 7, 8), AMT<double>.Padding(17, 31));
    }

    // BidirCorrBlock.corr: f0(p)·f1(q) / √D, the forward volume indexed by p, the transposed one by q.
    [Fact(Timeout = 60000)]
    public async Task CorrelationPyramids_MatchTheRelease()
    {
        await Task.Yield();
        const int d = 3, h = 4, w = 4;
        var f0 = Random(new[] { 1, d, h, w }, 1);
        var f1 = Random(new[] { 1, d, h, w }, 2);
        var (forward, transposed) = AmtOps.CorrelationPyramids(Engine, f0, f1, 2);
        Assert.Equal(new[] { 16, 1, 4, 4 }, forward[0].Shape.ToArray());
        Assert.Equal(new[] { 16, 1, 2, 2 }, forward[1].Shape.ToArray());
        for (int p = 0; p < h * w; p++)
            for (int q = 0; q < h * w; q++)
            {
                double expected = Enumerable.Range(0, d).Sum(c => f0[0, c, p / w, p % w] * f1[0, c, q / w, q % w]) / Math.Sqrt(d);
                Assert.Equal(expected, forward[0][p, 0, q / w, q % w], 10);
                Assert.Equal(expected, transposed[0][q, 0, p / w, p % w], 10);
            }
        double pooled = (forward[0][5, 0, 0, 0] + forward[0][5, 0, 0, 1] + forward[0][5, 0, 1, 0] + forward[0][5, 0, 1, 1]) / 4;
        Assert.Equal(pooled, forward[1][5, 0, 0, 0], 10);
    }

    // RAFT's window is stacked (dy, dx) yet added to (x, y): tap (i, j) samples (x + i − r, y + j − r), zero outside.
    [Fact(Timeout = 60000)]
    public async Task Lookup_KeepsRaftsWindowOrder()
    {
        await Task.Yield();
        var volume = new Tensor<double>(new[] { 1, 1, 5, 5 });
        for (int y = 0; y < 5; y++)
            for (int x = 0; x < 5; x++) volume[0, 0, y, x] = 10 * y + x + 1;
        var coords = new Tensor<double>(new[] { 1, 2, 1, 1 });
        coords[0] = 3.5; coords[1] = 1;
        var taps = AmtOps.Lookup(Engine, new[] { volume }, coords, 1);
        Assert.Equal(new[] { 1, 9, 1, 1 }, taps.Shape.ToArray());
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                double x = 3.5 + i - 1, y = 1 + j - 1;
                double Value(int yy, int xx) => xx < 5 ? 10 * yy + xx + 1 : 0.0;
                double expected = 0.5 * (Value((int)y, (int)Math.Floor(x)) + Value((int)y, (int)Math.Floor(x) + 1));
                Assert.Equal(expected, taps[0, i * 3 + j, 0, 0], 10);
            }
    }

    [Fact(Timeout = 60000)]
    public async Task Ternary_VanishesForMatchingStructure()
    {
        await Task.Yield();
        var a = Random(new[] { 1, 3, 12, 12 }, 3);
        var b = Random(new[] { 1, 3, 12, 12 }, 4);
        Assert.Equal(0.0, AmtOps.Ternary(Engine, a, a)[0], 12);
        Assert.Equal(0.0, AmtOps.Ternary(Engine, a, Engine.TensorAddScalar(a, 0.2))[0], 9);
        Assert.True(AmtOps.Ternary(Engine, a, b)[0] > 0.05);
    }

    [Fact(Timeout = 120000)]
    public async Task Interpolate_SynthesizesAnyTimeBetweenTheFrames()
    {
        await Task.Yield();
        var model = Model(SmallOptions());
        var frame0 = Random(new[] { 3, 20, 28 }, 5, 255.0);
        var frame1 = Random(new[] { 3, 20, 28 }, 6, 255.0);
        var middle = model.Interpolate(frame0, frame1);
        var early = model.Interpolate(frame0, frame1, 0.25);
        Assert.Equal(new[] { 3, 20, 28 }, middle.Shape.ToArray());
        Assert.All(middle.ToArray(), v => Assert.True(v >= 0 && v <= 255));
        Assert.False(middle.ToArray().SequenceEqual(early.ToArray()));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Interpolate(frame0, frame1, 0.0));
    }

    [Fact(Timeout = 240000)]
    public async Task AmtL_RunsAtItsReleaseWidths()
    {
        await Task.Yield();
        var model = Model(AMTOptions.AmtL());
        var output = model.Interpolate(Random(new[] { 3, 32, 32 }, 7, 255.0), Random(new[] { 3, 32, 32 }, 8, 255.0));
        Assert.Equal(new[] { 3, 32, 32 }, output.Shape.ToArray());
    }

    [Fact(Timeout = 240000)]
    public async Task TrainingStep_UpdatesTheNetwork()
    {
        await Task.Yield();
        var model = Model(SmallOptions());
        var before = model.NetworkParameters.SelectMany(t => t.ToArray()).ToArray();
        model.Train(Random(new[] { 6, 32, 32 }, 9, 255.0), Random(new[] { 3, 32, 32 }, 10, 255.0));
        var after = model.NetworkParameters.SelectMany(t => t.ToArray()).ToArray();
        Assert.Equal(before.Length, after.Length);
        Assert.False(before.SequenceEqual(after));
    }
}
