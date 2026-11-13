using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Classic;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Glow-TTS trains and synthesizes as its paper and reference implementation specify (Kim et al. 2020;
/// jaywalnut310/glow-tts): a relative-position Transformer encoder producing the prior means, monotonic alignment search,
/// an invertible flow decoder, and the maximum-likelihood plus duration objective.
/// </summary>
/// <remarks>
/// Before this change Glow-TTS had no flow, no alignment search and no likelihood: its synthesis computed durations with
/// a fixed formula on encoder values, and threw on every call because its encoder boundary pointed past the layer stack.
/// </remarks>
public class GlowTTSPaperTests
{
    private const int MelBins = 8;

    private static GlowTTS<double> CreateModel() => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: MelBins) { RandomSeed = 11 },
        new GlowTTSOptions
        {
            EncoderDim = 16, HiddenDim = 16, NumHeads = 2, NumEncoderLayers = 1, FilterChannels = 32, PrenetDropout = 0.0,
            DropoutRate = 0.0, DurationPredictorFilterChannels = 16, NumFlowBlocks = 2, DecoderHiddenChannels = 16,
            CouplingLayers = 2, DecoderDropout = 0.0, MelChannels = MelBins,
        },
        new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private static Tensor<double> Tokens(int count)
    {
        var tokens = new Tensor<double>(new[] { count });
        for (int i = 0; i < count; i++) tokens[i] = 10 + (i * 7) % 60;
        return tokens;
    }

    private static Tensor<double> Mel(int frames)
    {
        var mel = new Tensor<double>(new[] { frames, MelBins });
        for (int f = 0; f < frames; f++)
            for (int c = 0; c < MelBins; c++) mel[f, c] = Math.Sin(0.4 * f + 0.3 * c);
        return mel;
    }

    private static Tensor<double> Sequence(int channels, int time, double phase)
    {
        var x = new Tensor<double>(new[] { 1, channels, time });
        for (int i = 0; i < x.Length; i++) x[i] = Math.Sin(0.37 * i + phase);
        return x;
    }

    [Fact(Timeout = 60000)]
    public async Task FlowSteps_AreExactlyInvertible()
    {
        await Task.Yield();
        var x = Sequence(8, 6, 0.2);
        var actNorm = new ActNormFlowLayer<double>(8);
        actNorm.SetTrainingMode(true);
        actNorm.Transform(x, false); // data-dependent initialization
        actNorm.SetTrainingMode(false);
        var coupling = new AffineCouplingFlowLayer<double>(8, 12, 3, 1, 2, 0.0);
        // Move the zero-initialized end projection so the coupling is not the identity.
        var p = coupling.GetParameters();
        for (int i = 0; i < p.Length; i++) p[i] = 0.05 * Math.Sin(1.3 * i);
        coupling.SetParameters(p);
        foreach (IInvertibleFlowStep<double> step in new IInvertibleFlowStep<double>[]
                 { actNorm, new GroupedInvertibleConvFlowLayer<double>(8, 4), coupling })
        {
            var (z, _) = step.Transform(x, false);
            var (back, _) = step.Transform(z, true);
            for (int i = 0; i < x.Length; i++) Assert.Equal(x[i], back[i], 9);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task InvertibleConv_LogDeterminant_IsTheGroupMatrixLogDeterminantPerGroupAndFrame()
    {
        await Task.Yield();
        var layer = new GroupedInvertibleConvFlowLayer<double>(8, 4);
        var x = Sequence(8, 5, 0.0);
        var (_, logDet) = layer.Transform(x, false);
        // A rotation has |det| = 1, so log|det W| = 0 at initialization; perturb W and compare to the host determinant.
        var w = layer.GetParameters();
        for (int i = 0; i < w.Length; i++) w[i] += 0.1 * Math.Cos(i);
        layer.SetParameters(w);
        (_, logDet) = layer.Transform(x, false);
        var m = new double[4, 4];
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) m[i, j] = w[i * 4 + j];
        double det = Det4(m);
        Assert.Equal((8 / 4) * 5 * Math.Log(Math.Abs(det)), logDet![0], 9);
    }

    private static double Det4(double[,] a)
    {
        double Det3(int skipRow, int skipCol)
        {
            var r = Enumerable.Range(0, 4).Where(i => i != skipRow).ToArray();
            var c = Enumerable.Range(0, 4).Where(j => j != skipCol).ToArray();
            return a[r[0], c[0]] * (a[r[1], c[1]] * a[r[2], c[2]] - a[r[1], c[2]] * a[r[2], c[1]])
                 - a[r[0], c[1]] * (a[r[1], c[0]] * a[r[2], c[2]] - a[r[1], c[2]] * a[r[2], c[0]])
                 + a[r[0], c[2]] * (a[r[1], c[0]] * a[r[2], c[1]] - a[r[1], c[1]] * a[r[2], c[0]]);
        }
        double det = 0;
        for (int j = 0; j < 4; j++) det += (j % 2 == 0 ? 1 : -1) * a[0, j] * Det3(0, j);
        return det;
    }

    [Fact(Timeout = 60000)]
    public async Task RelativeAttention_MatchesTheReferenceFormula()
    {
        await Task.Yield();
        // attentions.MultiHeadAttention with window w and shared relative tables: for each head,
        // logits_ij = (q_i . k_j + [|j-i| <= w] q_i . rK[j-i+w]) / sqrt(dk), p = softmax_j,
        // out_i = sum_j p_ij v_j + sum_{|j-i| <= w} p_ij rV[j-i+w], then the output projection.
        const int hidden = 4, heads = 2, window = 1, length = 5, dk = hidden / heads;
        var block = new RelativePositionTransformerBlock<double>(hidden, heads, filter: 8, kernelSize: 1, dropoutRate: 0.0, window);
        block.SetTrainingMode(false);
        var x = new Tensor<double>(new[] { length, hidden });
        for (int i = 0; i < x.Length; i++) x[i] = Math.Sin(0.61 * i + 0.3);
        var actual = block.SelfAttention(x);

        double[,] Project(DenseLayer<double> layer, double[,] input)
        {
            var p = layer.GetParameters();
            int inSize = input.GetLength(1), outSize = p.Length / (inSize + 1);
            var y = new double[input.GetLength(0), outSize];
            for (int r = 0; r < input.GetLength(0); r++)
                for (int o = 0; o < outSize; o++)
                {
                    double sum = p[inSize * outSize + o];
                    for (int c = 0; c < inSize; c++) sum += input[r, c] * p[c * outSize + o];
                    y[r, o] = sum;
                }
            return y;
        }
        var xs = new double[length, hidden];
        for (int r = 0; r < length; r++) for (int c = 0; c < hidden; c++) xs[r, c] = x[r, c];
        var q = Project(block.QueryProjection, xs);
        var k = Project(block.KeyProjection, xs);
        var v = Project(block.ValueProjection, xs);
        var concat = new double[length, hidden];
        for (int h = 0; h < heads; h++)
        {
            for (int i = 0; i < length; i++)
            {
                var logits = new double[length];
                for (int j = 0; j < length; j++)
                {
                    double s = 0;
                    for (int d = 0; d < dk; d++) s += q[i, h * dk + d] * k[j, h * dk + d];
                    int r = j - i + window;
                    if (r >= 0 && r <= 2 * window)
                        for (int d = 0; d < dk; d++) s += q[i, h * dk + d] * block.RelativeKeys[r, d];
                    logits[j] = s / Math.Sqrt(dk);
                }
                double max = logits.Max();
                var p = logits.Select(l => Math.Exp(l - max)).ToArray();
                double total = p.Sum();
                for (int j = 0; j < length; j++) p[j] /= total;
                for (int d = 0; d < dk; d++)
                {
                    double o = 0;
                    for (int j = 0; j < length; j++)
                    {
                        o += p[j] * v[j, h * dk + d];
                        int r = j - i + window;
                        if (r >= 0 && r <= 2 * window) o += p[j] * block.RelativeValues[r, d];
                    }
                    concat[i, h * dk + d] = o;
                }
            }
        }
        var expected = Project(block.OutputProjection, concat);
        for (int i = 0; i < length; i++)
            for (int c = 0; c < hidden; c++) Assert.Equal(expected[i, c], actual[i, c], 10);
    }

    [Fact(Timeout = 180000)]
    public async Task Training_ReducesTheNegativeLogLikelihood()
    {
        await Task.Yield();
        var model = CreateModel();
        var tokens = Tokens(5);
        var mel = Mel(14);
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        model.Train(tokens, mel); // data-dependent ActNorm initialization runs on the first step
        double before = provider.EvaluateTrainingObjective(tokens, mel);
        for (int i = 0; i < 30; i++) model.Train(tokens, mel);
        double after = provider.EvaluateTrainingObjective(tokens, mel);
        Assert.True(after < before, $"Objective did not fall ({before} -> {after}).");
    }

    [Fact(Timeout = 60000)]
    public async Task Synthesis_IsRepeatable_AndHasAnEvenFrameCount()
    {
        await Task.Yield();
        var model = CreateModel();
        var first = model.Synthesize("hello");
        var second = model.Synthesize("hello");
        Assert.Equal(2, first.Rank);
        Assert.Equal(MelBins, first.Shape[1]);
        Assert.Equal(0, first.Shape[0] % 2);
        Assert.Equal(first.ToVector().ToArray(), second.ToVector().ToArray());
        for (int i = 0; i < first.Length; i++) Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(first[i]), $"mel[{i}] = {first[i]}");
    }
}
