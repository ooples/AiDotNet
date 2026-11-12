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
/// Zhang and Yao 2024, RealViformer, against the authors' release (Yuehan717/RealViformer, archs/realviformer_arch.py,
/// archs/arch_util.py, inference_realviformer.py) and Sec. 4.4: transposed channel attention, the SSIM loss, the
/// unidirectional recurrence, the two training stages and the flow freeze (#2397).
/// </summary>
public class RealViformerPaperTests
{
    private const int Size = 8;

    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(inputType: InputType.FourDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputFrames: 2, inputDepth: 3, inputHeight: Size, inputWidth: Size, outputSize: 3);

    private static RealViformer<double> Small(int reconstruction = 300000, int freeze = 5000, double learningRate = 1e-4) =>
        new(Architecture(), new RealViformerOptions
        {
            Seed = 1, NumFeatures = 8, NumBlocks = new[] { 1, 1, 1, 1 }, Heads = new[] { 1, 1, 1 }, MergeHeads = 1, SqueezeFactor = 2,
            DiscriminatorChannels = 4, NumFrames = 2, PerceptualWeight = 0, ReconstructionIterations = reconstruction,
            FlowFreezeIterations = freeze, LearningRate = learningRate, AdversarialLearningRate = learningRate,
        });

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreThePaperConfiguration()
    {
        await Task.Yield();
        var o = new RealViformerOptions();
        Assert.Equal((48, 2.66, 2, 4, true), (o.NumFeatures, o.FfnExpansionFactor, o.MergeHeads, o.SqueezeFactor, o.ChannelMask));
        Assert.Equal(new[] { 2, 3, 4, 1 }, o.NumBlocks);
        Assert.Equal(new[] { 1, 2, 4 }, o.Heads);
        Assert.Equal((15, 300000, 5000), (o.NumFrames, o.ReconstructionIterations, o.FlowFreezeIterations));
        Assert.Equal((1.0, 0.001, 1.0, 0.005), (o.CharbonnierWeight, o.SsimWeight, o.PerceptualWeight, o.GANWeight));
        Assert.Equal((1e-4, 5e-5, 1e-4), (o.LearningRate, o.AdversarialLearningRate, o.DiscriminatorLearningRate));
        Assert.Null(o.EmaMomentum);
    }

    [Fact(Timeout = 60000)]
    public async Task ChannelAttention_MatchesTheReleaseFormula()
    {
        await Task.Yield();
        const int heads = 2, c = 3, h = 2, w = 3;
        var q = Filled(new[] { 1, heads * c, h, w }, 0.31);
        var k = Filled(new[] { 1, heads * c, h, w }, 0.17);
        var v = Filled(new[] { 1, heads * c, h, w }, 0.23);
        var temperature = new Tensor<double>(new[] { 1, heads }, new Vector<double>(new[] { 0.7, 1.9 }));
        var (_, actual) = RealViformerBlocks.ChannelAttention(AiDotNetEngine.Current, q, k, v, temperature, heads);

        int hw = h * w;
        double[] Row(Tensor<double> t, int head, int ch) => Enumerable.Range(0, hw).Select(p => t[(head * c + ch) * hw + p]).ToArray();
        double[] Unit(double[] r) { double n = Math.Max(Math.Sqrt(r.Sum(x => x * x)), 1e-12); return r.Select(x => x / n).ToArray(); }
        for (int head = 0; head < heads; head++)
            for (int i = 0; i < c; i++)
            {
                var qi = Unit(Row(q, head, i));
                var logits = Enumerable.Range(0, c).Select(j => qi.Zip(Unit(Row(k, head, j)), (a, b) => a * b).Sum() * temperature[head]).ToArray();
                double max = logits.Max(), z = logits.Sum(l => Math.Exp(l - max));
                for (int p = 0; p < hw; p++)
                {
                    double expected = Enumerable.Range(0, c).Sum(j => Math.Exp(logits[j] - max) / z * v[(head * c + j) * hw + p]);
                    Assert.Equal(expected, actual[(head * c + i) * hw + p], 10);
                }
            }
    }

    [Fact(Timeout = 60000)]
    public async Task SsimLoss_IsZeroForIdenticalImages_AndGrowsWithNoise()
    {
        await Task.Yield();
        var x = Filled(new[] { 1, 3, 16, 16 }, 0.05);
        for (int i = 0; i < x.Length; i++) x[i] = 0.5 + 0.4 * x[i] / 1.2;
        var noisy = x.Clone();
        for (int i = 0; i < noisy.Length; i++) noisy[i] += 0.1 * Math.Sin(i * 1.7);
        var engine = AiDotNetEngine.Current;
        Assert.Equal(0.0, RealViformerBlocks.SsimLoss(engine, x, x)[0], 10);
        double loss = RealViformerBlocks.SsimLoss(engine, x, noisy)[0];
        Assert.InRange(loss, 1e-3, 1.0);
    }

    private static Tensor<double> Filled(int[] shape, double rate)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = Math.Sin(i * rate) + 0.1 * Math.Cos(i * 0.07);
        return t;
    }

    [Fact(Timeout = 300000)]
    public async Task Output_IsFourTimesTheInput_AndPropagatesForwardOnly()
    {
        await Task.Yield();
        var model = Small();
        var clip = new Tensor<double>(new[] { 2, 3, Size, Size });
        for (int i = 0; i < clip.Length; i++) clip[i] = 0.5 + 0.3 * Math.Sin(i * 0.37);
        var baseline = model.UpscaleNormalizedForTest(clip);
        Assert.Equal(new[] { 2, 3, 4 * Size, 4 * Size }, baseline.Shape.ToArray());
        int frame = 3 * Size * Size, outFrame = 3 * 16 * Size * Size;

        var firstChanged = clip.Clone();
        for (int i = 0; i < frame; i++) firstChanged[i] += 0.2;
        var forward = model.UpscaleNormalizedForTest(firstChanged);
        Assert.True(Enumerable.Range(outFrame, outFrame).Max(i => Math.Abs(forward[i] - baseline[i])) > 1e-9, "frame 1 must hear about frame 0");

        var lastChanged = clip.Clone();
        for (int i = frame; i < 2 * frame; i++) lastChanged[i] += 0.2;
        var backward = model.UpscaleNormalizedForTest(lastChanged);
        Assert.Equal(0.0, Enumerable.Range(0, outFrame).Max(i => Math.Abs(backward[i] - baseline[i])), 12);
    }

    [Fact(Timeout = 600000)]
    public async Task Discriminator_TrainsOnlyInTheAdversarialStage()
    {
        await Task.Yield();
        var (low, high) = Pair();
        var model = Small(reconstruction: 1);
        model.Predict(low);
        var before = model.DiscriminatorParameters.Select(t => t.ToArray()).ToList();
        Assert.False(model.InAdversarialStage);
        model.Train(low, high);
        Assert.True(before.Zip(model.DiscriminatorParameters, (b, a) => b.SequenceEqual(a.ToArray())).All(x => x), "the first stage has no discriminator");
        Assert.True(model.InAdversarialStage);
        model.Train(low, high);
        Assert.False(before.Zip(model.DiscriminatorParameters, (b, a) => b.SequenceEqual(a.ToArray())).All(x => x), "the second stage trains the discriminator");
    }

    [Fact(Timeout = 600000)]
    public async Task SpyNet_StaysFrozenForTheFixIterations()
    {
        await Task.Yield();
        var (low, high) = Pair();
        var model = Small(freeze: 100, learningRate: 1e-3);
        model.Predict(low);
        var flow = model.FlowParameters.Select(t => t.ToArray()).ToList();
        var output = model.Predict(low).ToArray();
        model.Train(low, high);
        Assert.True(flow.Zip(model.FlowParameters, (b, a) => b.SequenceEqual(a.ToArray())).All(x => x), "SPyNet moved inside its freeze window");
        Assert.False(output.SequenceEqual(model.Predict(low).ToArray()), "the rest of the network must train");
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
