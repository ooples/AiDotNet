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
/// Park et al. 2021, ABME (arXiv 2108.06815), and the release (JunHeum/ABME: run.py, utils.py, Upsample.py,
/// Bilateral_Correlation.py, correlation_package, model/*.py): the defaults, run.py's sizes, the release's tensor
/// operations against brute-force transcriptions, the census loss, the midpoint-only contract and the staged training
/// (#2397).
/// </summary>
public class ABMEPaperTests
{
    private static IEngine Engine => AiDotNetEngine.Current;

    private static ABMEOptions SmallOptions(AbmeTrainingStage stage) => new()
    {
        Seed = 1,
        Stage = stage,
        SymmetricEncoderWidths = new[] { 4, 4, 4, 4, 4, 4 },
        SymmetricDecoderWidths = new[] { 4, 4, 4, 4, 4 },
        SymmetricDisplacements = new[] { 1, 1, 1, 1, 1 },
        AsymmetricEncoderWidths = new[] { 4, 4 },
        AsymmetricDecoderWidths = new[] { 4, 4, 4, 4, 4 },
        AsymmetricDisplacement = 1,
        SynthesisWidths = new[] { 4, 4, 4 },
    };

    private static ABME<double> Small(AbmeTrainingStage stage = AbmeTrainingStage.Synthesis) =>
        new(new NeuralNetworkArchitecture<double>(inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.Regression,
                inputHeight: 64, inputWidth: 64, inputDepth: 3, outputSize: 3), SmallOptions(stage));

    private static Tensor<double> Random(int[] shape, int seed, double scale = 1.0)
    {
        var t = new Tensor<double>(shape);
        var random = new System.Random(seed);
        for (int i = 0; i < t.Length; i++) t[i] = scale * random.NextDouble();
        return t;
    }

    private static double[] Snapshot(System.Collections.Generic.IReadOnlyList<Tensor<double>> tensors)
        => tensors.SelectMany(t => t.ToArray()).ToArray();

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTheReleaseAndThePaper()
    {
        await Task.Yield();
        var o = new ABMEOptions();
        Assert.Equal(new[] { 16, 32, 64, 96, 128, 196 }, o.SymmetricEncoderWidths);
        Assert.Equal(new[] { 128, 128, 96, 64, 32 }, o.SymmetricDecoderWidths);
        Assert.Equal(new[] { 6, 4, 4, 2, 2 }, o.SymmetricDisplacements);
        Assert.Equal(new[] { 16, 32 }, o.AsymmetricEncoderWidths);
        Assert.Equal(new[] { 64, 64, 48, 32, 16 }, o.AsymmetricDecoderWidths);
        Assert.Equal(2, o.AsymmetricDisplacement);
        Assert.Equal(new[] { 32, 64, 96 }, o.SynthesisWidths);
        Assert.Equal((1e-4, 0.5, 1e-6), (o.LearningRate, o.DecayRate, o.CharbonnierEpsilon));
        Assert.Equal(new[] { 100000, 140000, 180000 }, o.SymmetricMilestones);
        Assert.Equal(new[] { 120000, 180000, 240000 }, o.AsymmetricMilestones);
        Assert.Equal(new[] { 350000, 500000, 650000 }, o.SynthesisMilestones);
        Assert.Equal(AbmeTrainingStage.Synthesis, o.Stage);
    }

    // run.py: multiples of 64 below a height of 512, half of multiples of 128 from there.
    [Fact(Timeout = 60000)]
    public async Task MotionSize_FollowsRunPy()
    {
        await Task.Yield();
        Assert.Equal((256, 448), ABME<double>.MotionSize(256, 448));
        Assert.Equal((128, 192), ABME<double>.MotionSize(100, 130));
        Assert.Equal((320, 512), ABME<double>.MotionSize(540, 960));
    }

    // Upsample.Upsample(x, 2): even samples copy, odd samples average a pair, the last edge replicated.
    [Fact(Timeout = 60000)]
    public async Task Upsample2_IsTheReleaseKernel()
    {
        await Task.Yield();
        var x = new Tensor<double>(new[] { 1, 1, 1, 3 });
        x[0] = 1; x[1] = 3; x[2] = 7;
        var y = AbmeOps.Upsample2(Engine, x);
        Assert.Equal(new[] { 1, 1, 2, 6 }, y.Shape.ToArray());
        Assert.Equal(new double[] { 1, 2, 3, 5, 7, 7, 1, 2, 3, 5, 7, 7 }, y.ToArray());
    }

    // utils.warp: a sample that reaches outside the frame is masked to zero, one inside is the bilinear value.
    [Fact(Timeout = 60000)]
    public async Task Warp_MasksSamplesLeavingTheFrame()
    {
        await Task.Yield();
        var image = new Tensor<double>(new[] { 1, 1, 2, 4 });
        for (int i = 0; i < image.Length; i++) image[i] = i % 4 + 1;
        var flow = new Tensor<double>(new[] { 1, 2, 2, 4 });
        for (int i = 0; i < 8; i++) flow[i] = 0.5;
        var warped = AbmeOps.Warp(Engine, image, flow);
        for (int row = 0; row < 2; row++)
        {
            Assert.Equal(1.5, warped[0, 0, row, 0], 12);
            Assert.Equal(2.5, warped[0, 0, row, 1], 12);
            Assert.Equal(3.5, warped[0, 0, row, 2], 12);
            Assert.Equal(0.0, warped[0, 0, row, 3], 12);
        }
    }

    // correlation_cuda with kernel 1, stride 1, padding = displacement: channel (dy, dx) y-major, the channel mean of
    // f1(x) f2(x + d), f2 zero outside the frame.
    [Fact(Timeout = 60000)]
    public async Task Correlation_MatchesTheCudaPackage()
    {
        await Task.Yield();
        const int md = 2, c = 3, h = 5, w = 6;
        var f1 = Random(new[] { 1, c, h, w }, 1);
        var f2 = Random(new[] { 1, c, h, w }, 2);
        var result = AbmeOps.Correlation(Engine, f1, f2, md);
        Assert.Equal(new[] { 1, 25, h, w }, result.Shape.ToArray());
        for (int dy = -md; dy <= md; dy++)
            for (int dx = -md; dx <= md; dx++)
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        double expected = 0;
                        if (y + dy >= 0 && y + dy < h && x + dx >= 0 && x + dx < w)
                            for (int k = 0; k < c; k++) expected += f1[0, k, y, x] * f2[0, k, y + dy, x + dx];
                        Assert.Equal(expected / c, result[0, (dy + md) * 5 + dx + md, y, x], 10);
                    }
    }

    // BilateralCorrelation at zero motion: displacement (u, v), x fastest, compares the normalized first frame at
    // p − d with the second at p + d (border sampling, masked to zero where either leaves the frame).
    [Fact(Timeout = 60000)]
    public async Task BilateralCorrelation_MatchesTheRelease()
    {
        await Task.Yield();
        const int md = 1, c = 3, h = 4, w = 5;
        var f1 = Random(new[] { 1, c, h, w }, 3);
        var f2 = Random(new[] { 1, c, h, w }, 4);
        var result = AbmeOps.BilateralCorrelation(Engine, f1, f2, new Tensor<double>(new[] { 1, 2, h, w }), md);
        double Norm(Tensor<double> f, int y, int x) => Math.Sqrt(Enumerable.Range(0, c).Sum(k => f[0, k, y, x] * f[0, k, y, x]) + 1e-6);
        for (int v = -md; v <= md; v++)
            for (int u = -md; u <= md; u++)
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        int y1 = y - v, x1 = x - u, y2 = y + v, x2 = x + u;
                        double expected = 0;
                        if (y1 >= 0 && y1 < h && x1 >= 0 && x1 < w && y2 >= 0 && y2 < h && x2 >= 0 && x2 < w)
                        {
                            double n1 = Norm(f1, y1, x1), n2 = Norm(f2, y2, x2);
                            for (int k = 0; k < c; k++) expected += f1[0, k, y1, x1] / n1 * (f2[0, k, y2, x2] / n2);
                        }
                        Assert.Equal(expected, result[0, (v + md) * 3 + u + md, y, x], 10);
                    }
    }

    [Fact(Timeout = 60000)]
    public async Task CensusLoss_VanishesOnlyForMatchingImages()
    {
        await Task.Yield();
        var a = Random(new[] { 1, 3, 12, 12 }, 5);
        var b = Random(new[] { 1, 3, 12, 12 }, 6);
        Assert.Equal(0.0, AbmeOps.CensusLoss(Engine, a, a)[0], 12);
        Assert.True(AbmeOps.CensusLoss(Engine, a, b)[0] > 0.1);
        // The census transform is invariant to a brightness offset.
        var shifted = Engine.TensorAddScalar(a, 0.1);
        Assert.Equal(0.0, AbmeOps.CensusLoss(Engine, a, shifted)[0], 9);
    }

    [Fact(Timeout = 120000)]
    public async Task Interpolate_ReturnsTheMidpointAtTheInputSize()
    {
        await Task.Yield();
        var model = Small();
        var frame0 = Random(new[] { 3, 20, 28 }, 7, 255.0);
        var frame1 = Random(new[] { 3, 20, 28 }, 8, 255.0);
        var middle = model.Interpolate(frame0, frame1);
        Assert.Equal(new[] { 3, 20, 28 }, middle.Shape.ToArray());
        Assert.All(middle.ToArray(), value => Assert.True(value >= 0 && value <= 255));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Interpolate(frame0, frame1, 0.25));
    }

    // Sec. 4.1: the estimator trains alone, then the refiner with the estimator fixed.
    [Theory(Timeout = 240000)]
    [InlineData(AbmeTrainingStage.SymmetricMotion)]
    [InlineData(AbmeTrainingStage.AsymmetricMotion)]
    public async Task MotionStages_TrainOnlyTheirNetwork(AbmeTrainingStage stage)
    {
        await Task.Yield();
        var model = Small(stage);
        var sbme = Snapshot(model.StageParameters(AbmeTrainingStage.SymmetricMotion));
        var abmr = Snapshot(model.StageParameters(AbmeTrainingStage.AsymmetricMotion));
        model.Train(Random(new[] { 6, 64, 64 }, 9, 255.0), Random(new[] { 3, 64, 64 }, 10, 255.0));
        bool sbmeChanged = !sbme.SequenceEqual(Snapshot(model.StageParameters(AbmeTrainingStage.SymmetricMotion)));
        bool abmrChanged = !abmr.SequenceEqual(Snapshot(model.StageParameters(AbmeTrainingStage.AsymmetricMotion)));
        Assert.Equal(stage == AbmeTrainingStage.SymmetricMotion, sbmeChanged);
        Assert.Equal(stage == AbmeTrainingStage.AsymmetricMotion, abmrChanged);
    }

    // The last stage trains all three networks end to end.
    [Fact(Timeout = 240000)]
    public async Task SynthesisStage_TrainsEndToEnd()
    {
        await Task.Yield();
        var model = Small();
        int motionTensors = model.StageParameters(AbmeTrainingStage.SymmetricMotion).Count
            + model.StageParameters(AbmeTrainingStage.AsymmetricMotion).Count;
        double[] Synthesis() => Snapshot(model.StageParameters(AbmeTrainingStage.Synthesis).Skip(motionTensors).ToList());
        var sbme = Snapshot(model.StageParameters(AbmeTrainingStage.SymmetricMotion));
        var abmr = Snapshot(model.StageParameters(AbmeTrainingStage.AsymmetricMotion));
        var synthesis = Synthesis();
        Assert.NotEmpty(synthesis);
        model.Train(Random(new[] { 6, 64, 64 }, 11, 255.0), Random(new[] { 3, 64, 64 }, 12, 255.0));
        Assert.False(sbme.SequenceEqual(Snapshot(model.StageParameters(AbmeTrainingStage.SymmetricMotion))));
        Assert.False(abmr.SequenceEqual(Snapshot(model.StageParameters(AbmeTrainingStage.AsymmetricMotion))));
        Assert.False(synthesis.SequenceEqual(Synthesis()));
    }
}
