using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Enhancement;
using AiDotNet.Video.Options;
using Xunit;

namespace AiDotNet.Tests.Video;

/// <summary>
/// Xu et al. 2024, IART, against the authors' release (kai422/IART, archs/iart_arch.py, archs/implicit_alignment.py and
/// options/IART_REDS_N16_600K.yml): the implicit warp, two-way propagation, the flow freeze, the moving average and the
/// training recipe (#2397).
/// </summary>
public class IARTPaperTests
{
    private const int Size = 8;

    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(inputType: InputType.FourDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputFrames: 3, inputDepth: 3, inputHeight: Size, inputWidth: Size, outputSize: 3);

    private static IART<double> Small(int freeze = 5000, double learningRate = 2e-4, double? ema = 0.001) => new(Architecture(), new IARTOptions
    {
        Seed = 1, MidChannels = 8, EmbedDim = 12, Depths = new[] { 2 }, NumHeads = 2, NumFrames = 3,
        FlowFreezeIterations = freeze, LearningRate = learningRate, EmaMomentum = ema,
    });

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTheReleaseRecipe()
    {
        await Task.Yield();
        var o = new IARTOptions();
        Assert.Equal((64, 120, 6, 8), (o.MidChannels, o.EmbedDim, o.NumHeads, o.WindowSize));
        Assert.Equal(new[] { 6, 6, 6 }, o.Depths);
        Assert.Equal((2.0, 0.1, 16, 0.01), (o.MlpRatio, o.DropPathRate, o.NumFrames, o.PositionTemperature));
        Assert.Equal((2e-4, 0.125, 5000, 1e-7, 600000), (o.LearningRate, o.SpyNetLearningRateMultiplier, o.FlowFreezeIterations, o.MinLearningRate, o.CosineIterations));
        Assert.Equal(1e-12, o.CharbonnierEpsilon * o.CharbonnierEpsilon, 20);
        Assert.Equal(0.001, o.EmaMomentum);
    }

    [Fact(Timeout = 60000)]
    public async Task WindowCode_IsTheReleaseSineEncoding()
    {
        await Task.Yield();
        var layers = new List<ILayer<double>>();
        var warp = new IartImplicitWarp<double>(layers, 8, 2, 0.01);
        // get_sine_position_encoding((2, 2), 4, temperature 0.01): dim_t = 0.01^(2 floor(i/2) / 4) = 1, 1, 0.1, 0.1; the
        // window position (1, 0) is y = 2*pi / (1 + 1e-6), x = 0, and the code is [sin, cos, sin, cos] of y then of x.
        double y = 2 * Math.PI / (1 + 1e-6);
        var expected = new[] { Math.Sin(y), Math.Cos(y), Math.Sin(y / 0.1), Math.Cos(y / 0.1), 0, 1, 0, 1 };
        for (int i = 0; i < 8; i++) Assert.Equal(expected[i], warp.WindowCode[2 * 8 + i], 12);
    }

    [Fact(Timeout = 60000)]
    public async Task ImplicitWarp_MatchesTheReleaseModule()
    {
        await Task.Yield();
        const int c = 8, heads = 2, h = 5, w = 6;
        var layers = new List<ILayer<double>>();
        var warp = new IartImplicitWarp<double>(layers, c, heads, 0.01);
        var propagated = Filled(new[] { 1, c, h, w }, 0.31);
        var current = Filled(new[] { 1, c, h, w }, 0.17);
        var flow = new Tensor<double>(new[] { 1, 2, h, w });
        for (int i = 0; i < flow.Length; i++) flow[i] = ((i * 5) % 9 - 4) * 0.83;

        var actual = warp.Align(AiDotNetEngine.Current, layers, propagated, current, flow);
        var expected = Reference(layers, warp, propagated, current, flow, c, heads, h, w);
        Assert.Equal(new[] { 1, c, h, w }, actual.Shape.ToArray());
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 9);
    }

    private static Tensor<double> Filled(int[] shape, double rate)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = Math.Sin(i * rate) + 0.1 * Math.Cos(i * 0.07);
        return t;
    }

    // ImplicitWarpModule.forward written out for one batch item: duplicate-padded 2x2 gather, window and offset sine codes,
    // biased q/k/v, per-head softmax over the four neighbours, no output projection.
    private static double[] Reference(List<ILayer<double>> layers, IartImplicitWarp<double> warp, Tensor<double> y,
        Tensor<double> x, Tensor<double> flow, int c, int heads, int h, int w)
    {
        double[] Linear(int layer, double[] input)
        {
            var dense = (AiDotNet.NeuralNetworks.Layers.DenseLayer<double>)layers[layer];
            var weights = dense.GetWeights();
            var biases = dense.GetBiases();
            var output = new double[c];
            for (int o = 0; o < c; o++)
            {
                double s = biases[o];
                for (int i = 0; i < c; i++) s += input[i] * weights[i * c + o];
                output[o] = s;
            }
            return output;
        }

        int features = c / 2;
        double Code(double v, int i) => i % 2 == 0 ? Math.Sin(v / Math.Pow(0.01, 2.0 * (i / 2) / features)) : Math.Cos(v / Math.Pow(0.01, 2.0 * (i / 2) / features));
        int hd = c / heads;
        var output = new double[c * h * w];
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++)
            {
                double fy = i + flow[(1 * h + i) * w + j], fx = j + flow[(0 * h + i) * w + j];
                int iy = (int)Math.Floor(fy), ix = (int)Math.Floor(fx);
                double oy = (fy - iy) / (2 + 1e-6) * 2 * Math.PI, ox = (fx - ix) / (2 + 1e-6) * 2 * Math.PI;
                var query = new double[c];
                for (int ch = 0; ch < c; ch++)
                    query[ch] = x[(ch * h + i) * w + j] + (ch < features ? Code(oy, ch) : Code(ox, ch - features));
                var q = Linear(0, query);
                var keys = new double[4][];
                var values = new double[4][];
                for (int k = 0; k < 4; k++)
                {
                    int sy = Math.Min(Math.Max(iy + k / 2, 0), h - 1), sx = Math.Min(Math.Max(ix + k % 2, 0), w - 1);
                    var neighbour = new double[c];
                    for (int ch = 0; ch < c; ch++) neighbour[ch] = y[(ch * h + sy) * w + sx] + warp.WindowCode[k * c + ch];
                    keys[k] = Linear(1, neighbour);
                    values[k] = Linear(2, neighbour);
                }
                for (int head = 0; head < heads; head++)
                {
                    var logits = new double[4];
                    for (int k = 0; k < 4; k++)
                        for (int d = 0; d < hd; d++) logits[k] += q[head * hd + d] * Math.Pow(hd, -0.5) * keys[k][head * hd + d];
                    double max = logits.Max(), z = logits.Sum(l => Math.Exp(l - max));
                    for (int d = 0; d < hd; d++)
                        output[((head * hd + d) * h + i) * w + j] = Enumerable.Range(0, 4).Sum(k => Math.Exp(logits[k] - max) / z * values[k][head * hd + d]);
                }
            }
        return output;
    }

    [Fact(Timeout = 300000)]
    public async Task Output_IsFourTimesTheInput_AndPropagatesBothWays()
    {
        await Task.Yield();
        var model = Small();
        var clip = new Tensor<double>(new[] { 3, 3, Size, Size });
        for (int i = 0; i < clip.Length; i++) clip[i] = 0.5 + 0.3 * Math.Sin(i * 0.37);
        var baseline = model.UpscaleNormalizedForTest(clip);
        Assert.Equal(new[] { 3, 3, 4 * Size, 4 * Size }, baseline.Shape.ToArray());
        int frame = 3 * Size * Size, outFrame = 3 * 16 * Size * Size;

        var lastChanged = clip.Clone();
        for (int i = 2 * frame; i < 3 * frame; i++) lastChanged[i] += 0.2;
        var backward = model.UpscaleNormalizedForTest(lastChanged);
        Assert.True(Enumerable.Range(0, outFrame).Max(i => Math.Abs(backward[i] - baseline[i])) > 1e-9, "frame 0 must hear about frame 2");

        var firstChanged = clip.Clone();
        for (int i = 0; i < frame; i++) firstChanged[i] += 0.2;
        var forward = model.UpscaleNormalizedForTest(firstChanged);
        Assert.True(Enumerable.Range(2 * outFrame, outFrame).Max(i => Math.Abs(forward[i] - baseline[i])) > 1e-9, "frame 2 must hear about frame 0");
    }

    [Fact(Timeout = 600000)]
    public async Task SpyNet_StaysFrozenForTheFixIterations()
    {
        await Task.Yield();
        var (low, high) = Pair();
        var model = Small(freeze: 100, learningRate: 1e-3, ema: null);
        // SPyNet's convolutions are lazily shaped: snapshot the weights after a forward has created them.
        model.Predict(low);
        var flow = model.FlowParameters.Select(t => t.ToArray()).ToList();
        var output = model.Predict(low).ToArray();
        model.Train(low, high);
        Assert.True(flow.Zip(model.FlowParameters, (b, a) => b.SequenceEqual(a.ToArray())).All(x => x), "SPyNet moved inside its freeze window");
        Assert.False(output.SequenceEqual(model.Predict(low).ToArray()), "the rest of the network must train");
    }

    [Fact(Timeout = 300000)]
    public async Task Inference_UsesTheMovingAverageOfTheWeights()
    {
        await Task.Yield();
        var (low, high) = Pair();
        var averaged = Small(learningRate: 1e-3);
        var plain = Small(learningRate: 1e-3, ema: null);
        for (int step = 0; step < 3; step++)
        {
            averaged.Train(low, high);
            plain.Train(low, high);
        }
        var a = averaged.Predict(low);
        var p = plain.Predict(low);
        Assert.True(Enumerable.Range(0, a.Length).Max(i => Math.Abs(a[i] - p[i])) > 1e-6, "inference must use the average");
    }

    [Fact(Timeout = 600000)]
    public async Task Training_ReducesTheError()
    {
        await Task.Yield();
        var (low, high) = Pair();
        var model = Small(learningRate: 1e-3, ema: null);
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

    // A smooth 32x32 three-frame clip and its 4x box-downsampled 8x8 version, pixel values 0-255.
    private static (Tensor<double> Low, Tensor<double> High) Pair()
    {
        int big = 4 * Size;
        var high = new Tensor<double>(new[] { 3, 3, big, big });
        for (int f = 0; f < 3; f++)
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < big; y++)
                    for (int x = 0; x < big; x++)
                        high[((f * 3 + c) * big + y) * big + x] = 128 + 80 * Math.Sin((x + 2 * f) * 0.21 + c) * Math.Cos(y * 0.17);
        var low = new Tensor<double>(new[] { 3, 3, Size, Size });
        for (int f = 0; f < 3; f++)
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
