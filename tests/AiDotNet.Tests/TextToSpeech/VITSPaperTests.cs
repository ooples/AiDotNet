using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.EndToEnd;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// VITS trains and synthesizes as its paper specifies (Kim et al. 2021): posterior encoder, flow prior aligned by
/// monotonic alignment search, stochastic duration predictor with spline flows, HiFi-GAN decoder on a random window, and
/// the mel + KL + duration + adversarial objective.
/// </summary>
/// <remarks>
/// Before this change VITS ran a stack of generic layers once and trained it by regression; it had no posterior, no
/// flow, no alignment search, no duration model and no discriminator.
/// </remarks>
public class VITSPaperTests
{
    private static VITS<double> CreateModel() => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: 64) { RandomSeed = 11 },
        new VITSOptions
        {
            VocabSize = 32, HiddenDim = 16, InterChannels = 8, FilterChannels = 32, NumHeads = 2, NumEncoderLayers = 1,
            DropoutRate = 0.0, PosteriorLayers = 2, FlowLayers = 2, NumFlowSteps = 2, DurationPredictorDropout = 0.0,
            DurationPredictorFlows = 2, UpsampleRates = [4, 4], UpsampleKernelSizes = [8, 8], UpsampleInitialChannels = 16,
            ResblockKernelSizes = [3], ResblockDilationSizes = [[1, 3]], DiscriminatorPeriods = [2, 3], DiscriminatorWidthDivisor = 32,
            FftSize = 64, WindowSize = 64, HopSize = 16, SampleRate = 4000, MelChannels = 8, SegmentSize = 128,
        },
        new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private static (Tensor<double> Tokens, Tensor<double> Audio) Example()
    {
        var tokens = new Tensor<double>(new[] { 3 });
        for (int i = 0; i < 3; i++) tokens[i] = 4 + i * 7;
        var audio = new Tensor<double>(new[] { 512 });
        for (int i = 0; i < 512; i++) audio[i] = 0.5 * Math.Sin(2 * Math.PI * (200 + i / 4.0) * i / 4000.0);
        return (tokens, audio);
    }

    [Fact(Timeout = 300000)]
    public async Task Training_ReducesTheMelKlAndDurationObjective()
    {
        await Task.Yield();
        var model = CreateModel();
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var (tokens, audio) = Example();
        double before = provider.EvaluateTrainingObjective(tokens, audio);
        for (int i = 0; i < 25; i++) model.Train(tokens, audio);
        double after = provider.EvaluateTrainingObjective(tokens, audio);
        Assert.True(after < before, $"Objective did not fall ({before} -> {after}).");
    }

    [Fact(Timeout = 120000)]
    public async Task Synthesis_ProducesAWaveformOfWholeFrames_Repeatably()
    {
        await Task.Yield();
        var model = CreateModel();
        var wave = model.Synthesize("abc");
        Assert.Equal(1, wave.Rank);
        Assert.Equal(0, wave.Length % 16);
        Assert.Equal(wave.ToVector().ToArray(), model.Synthesize("abc").ToVector().ToArray());
        for (int i = 0; i < wave.Length; i++) Assert.InRange(wave[i], -1.0, 1.0);
    }

    [Fact(Timeout = 60000)]
    public async Task RationalQuadraticSpline_InvertsItself_AndPassesTheTailsThrough()
    {
        await Task.Yield();
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var rng = new Random(3);
        const int n = 6, k = 10;
        var x = new Tensor<double>(new[] { n });
        double[] values = { -4.2, -1.0, 0.0, 0.7, 3.9, 6.5 };
        for (int i = 0; i < n; i++) x[i] = values[i];
        Tensor<double> Random(int cols)
        {
            var t = new Tensor<double>(new[] { n, cols });
            for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() * 2 - 1;
            return t;
        }
        var w = Random(k);
        var h = Random(k);
        var d = Random(k - 1);
        var (y, logForward) = AiDotNet.TextToSpeech.EndToEnd.RationalQuadraticSpline.Apply(engine, x, w, h, d, false, 5.0);
        var (back, logInverse) = AiDotNet.TextToSpeech.EndToEnd.RationalQuadraticSpline.Apply(engine, y, w, h, d, true, 5.0);
        for (int i = 0; i < n; i++)
        {
            Assert.Equal(values[i], back[i], 6);
            Assert.Equal(0.0, logForward[i] + logInverse[i], 6);
        }
        Assert.Equal(6.5, y[n - 1], 12);           // outside the tail bound: identity
        Assert.Equal(0.0, logForward[n - 1], 12);
    }
}
