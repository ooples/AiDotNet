using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Enhancement;
using AiDotNet.Video.Options;
using Xunit;

namespace AiDotNet.Tests.Video;

/// <summary>
/// Cao et al. 2025, DualX-VSR (arXiv 2506.04830), Tables 6-8, Sec. 5.1 and App. C.1: the configuration, the parameter
/// count, RoPE's relative positions, the motion-free reach of the dual axial attention, the two training stages and
/// training (#2397).
/// </summary>
public class DualXVSRPaperTests
{
    private const int Size = 8;

    private static NeuralNetworkArchitecture<T> Architecture<T>(int frames = 2) =>
        new(inputType: InputType.FourDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputFrames: frames, inputDepth: 3, inputHeight: Size, inputWidth: Size, outputSize: 3);

    private static DualXVSR<double> Small(int reconstruction = 650000, double learningRate = 5e-5) =>
        new(Architecture<double>(), new DualXVSROptions
        {
            Seed = 1, NumFeatures = 8, EmbedDim = 16, NumHeads = 2, HeadDim = 8, EmbeddingTemporalDepth = 1, VerticalDepth = 1,
            HorizontalDepth = 1, ReconstructionTemporalDepth = 1, NumFrames = 2, DiscriminatorChannels = 4, PerceptualWeight = 0,
            ReconstructionIterations = reconstruction, LearningRate = learningRate,
        });

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreThePaperConfiguration()
    {
        await Task.Yield();
        var o = new DualXVSROptions();
        Assert.Equal((64, 2, 1280, 12, 2.0), (o.NumFeatures, o.PatchSize, o.EmbedDim, o.NumHeads, o.MlpRatio));
        Assert.Equal((2, 6, 6, 2), (o.EmbeddingTemporalDepth, o.VerticalDepth, o.HorizontalDepth, o.ReconstructionTemporalDepth));
        Assert.Equal((10000.0, 4, 16), (o.RopeBase, o.ScaleFactor, o.NumFrames));
        Assert.Equal((5e-5, 1e-5, 1e-5, 650000), (o.LearningRate, o.AdversarialLearningRate, o.DiscriminatorLearningRate, o.ReconstructionIterations));
        Assert.Equal((0.01, 1.0, 0.005), (o.AdversarialPixelWeight, o.PerceptualWeight, o.GANWeight));
    }

    // Tab. 4 reports 127.95M parameters. The paper gives no code; ViViT's 64-channel heads reproduce the count.
    [Fact(Timeout = 300000)]
    public async Task DefaultNetwork_HasThePapersParameterCount()
    {
        await Task.Yield();
        var model = new DualXVSR<float>(Architecture<float>(), new DualXVSROptions { Seed = 1, PerceptualWeight = 0 });
        model.Predict(new Tensor<float>(new[] { 2, 3, Size, Size }));
        long generator = model.ParameterCount - model.DiscriminatorParameters.Sum(p => (long)p.Length);
        Assert.InRange(generator / 1e6, 127.95 * 0.995, 127.95 * 1.005);
    }

    [Fact(Timeout = 60000)]
    public async Task Rope_ScoresDependOnlyOnTheOffset()
    {
        await Task.Yield();
        var engine = AiDotNetEngine.Current;
        const int d = 8;
        var q = new Tensor<double>(new[] { 1, 1, d });
        var k = new Tensor<double>(new[] { 1, 1, d });
        for (int i = 0; i < d; i++) { q[i] = Math.Sin(i + 1.0); k[i] = Math.Cos(2.0 * i); }
        double Score(int[] qPos, int[] kPos)
        {
            var (qc, qs) = DualXRope.Tables<double>(new[] { qPos }, d, 10000);
            var (kc, ks) = DualXRope.Tables<double>(new[] { kPos }, d, 10000);
            var rq = DualXRope.Apply(engine, q, qc, qs, qPos.Length);
            var rk = DualXRope.Apply(engine, k, kc, ks, kPos.Length);
            return Enumerable.Range(0, d).Sum(i => rq[i] * rk[i]);
        }

        Assert.Equal(Score(new[] { 1, 2 }, new[] { 4, 3 }), Score(new[] { 6, 9 }, new[] { 9, 10 }), 10);
        Assert.Equal(Score(new[] { 3 }, new[] { 0 }), Score(new[] { 12 }, new[] { 9 }), 10);
        Assert.Equal(Enumerable.Range(0, d).Sum(i => q[i] * k[i]), Score(new[] { 5, 7 }, new[] { 5, 7 }), 10);
        Assert.NotEqual(Score(new[] { 0, 0 }, new[] { 0, 1 }), Score(new[] { 0, 0 }, new[] { 1, 0 }), 6);
    }

    // Without flow or alignment, one pixel of the first frame must still reach the opposite corner of the last frame:
    // the vertical-temporal pass carries it down its column and across time, the horizontal-temporal pass along the row.
    [Fact(Timeout = 300000)]
    public async Task DualAxialAttention_ReachesEveryPositionOfEveryFrame()
    {
        await Task.Yield();
        var model = Small();
        var clip = new Tensor<double>(new[] { 2, 3, Size, Size });
        for (int i = 0; i < clip.Length; i++) clip[i] = 0.5 + 0.3 * Math.Sin(i * 0.37);
        var baseline = model.UpscaleNormalizedForTest(clip);
        Assert.Equal(new[] { 2, 3, 4 * Size, 4 * Size }, baseline.Shape.ToArray());
        var changed = clip.Clone();
        for (int c = 0; c < 3; c++) changed[c * Size * Size] += 0.3;
        var output = model.UpscaleNormalizedForTest(changed);
        int big = 4 * Size;
        double corner = Enumerable.Range(0, 3).Max(c => Math.Abs(output[((3 + c) * big + big - 1) * big + big - 1] - baseline[((3 + c) * big + big - 1) * big + big - 1]));
        Assert.True(corner > 1e-9, "frame 0's top-left pixel never reached frame 1's bottom-right output");
    }

    [Fact(Timeout = 600000)]
    public async Task Discriminator_TrainsOnlyInTheGanStage()
    {
        await Task.Yield();
        var (low, high) = Pair();
        var model = Small(reconstruction: 1);
        model.Predict(low);
        var before = model.DiscriminatorParameters.Select(t => t.ToArray()).ToList();
        Assert.False(model.InAdversarialStage);
        model.Train(low, high);
        Assert.True(before.Zip(model.DiscriminatorParameters, (b, a) => b.SequenceEqual(a.ToArray())).All(x => x), "the Charbonnier stages have no discriminator");
        Assert.True(model.InAdversarialStage);
        model.Train(low, high);
        Assert.False(before.Zip(model.DiscriminatorParameters, (b, a) => b.SequenceEqual(a.ToArray())).All(x => x), "the GAN stage trains the discriminator");
    }

    [Fact(Timeout = 600000)]
    public async Task Training_ReducesTheError()
    {
        await Task.Yield();
        var (low, high) = Pair();
        var model = Small(learningRate: 1e-3);
        double Error()
        {
            var output = model.Predict(low);
            return Enumerable.Range(0, output.Length).Average(i => Math.Abs(output[i] - high[i]));
        }

        double before = Error();
        for (int step = 0; step < 30; step++) model.Train(low, high);
        double after = Error();
        Assert.True(after < 0.7 * before, $"L1 {before:F2} -> {after:F2}");
    }

    // A smooth 32x32 two-frame clip and its 4x box-downsampled 8x8 version, pixel values 0-255.
    private static (Tensor<double> Low, Tensor<double> High) Pair()
    {
        int big = 4 * Size;
        var high = new Tensor<double>(new[] { 2, 3, big, big });
        for (int f = 0; f < 2; f++)
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < big; y++)
                    for (int x = 0; x < big; x++)
                        high[((f * 3 + c) * big + y) * big + x] = 128 + 80 * Math.Sin((x + 2 * f) * 0.21 + c) * Math.Cos(y * 0.17);
        var low = new Tensor<double>(new[] { 2, 3, Size, Size });
        for (int f = 0; f < 2; f++)
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < Size; y++)
                    for (int x = 0; x < Size; x++)
                    {
                        double s = 0;
                        for (int dy = 0; dy < 4; dy++)
                            for (int dx = 0; dx < 4; dx++) s += high[((f * 3 + c) * big + 4 * y + dy) * big + 4 * x + dx];
                        low[((f * 3 + c) * Size + y) * Size + x] = s / 16;
                    }
        return (low, high);
    }
}
