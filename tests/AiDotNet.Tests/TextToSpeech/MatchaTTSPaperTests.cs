using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.FlowDiffusion;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Matcha-TTS trains and synthesizes as its paper and reference implementation specify (Mehta et al. 2024;
/// shivammehta25/Matcha-TTS): a rotary-position text encoder aligned by monotonic alignment search, a 1-D U-Net vector
/// field trained with the OT-CFM loss on normalized mels, and Euler integration from τ-scaled noise.
/// </summary>
/// <remarks>
/// Before this change Matcha-TTS had no flow matching: synthesis ignored its layers, set durations from character codes
/// and produced a sine-shaped "waveform" from a hand-written velocity formula.
/// </remarks>
public class MatchaTTSPaperTests
{
    private const int MelBins = 8;

    private static MatchaTTS<double> CreateModel(int steps = 3) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: MelBins) { RandomSeed = 11 },
        new MatchaTTSOptions
        {
            HiddenDim = 16, NumHeads = 2, NumEncoderLayers = 1, FilterChannels = 32, PrenetDropout = 0.0, DropoutRate = 0.0,
            DurationPredictorFilterChannels = 16, FlowDim = 16, DecoderHeads = 2, DecoderHeadDim = 8, DecoderMidBlocks = 1,
            DecoderDropout = 0.0, MelChannels = MelBins, NumFlowSteps = steps, MelMean = 0.0, MelStd = 1.0,
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

    [Fact(Timeout = 180000)]
    public async Task Training_ReducesTheDurationPriorAndFlowMatchingObjective()
    {
        await Task.Yield();
        var model = CreateModel();
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var tokens = Tokens(5);
        var mel = Mel(14);
        double before = provider.EvaluateTrainingObjective(tokens, mel);
        for (int i = 0; i < 40; i++) model.Train(tokens, mel);
        double after = provider.EvaluateTrainingObjective(tokens, mel);
        Assert.True(after < before, $"Objective did not fall ({before} -> {after}).");
    }

    [Fact(Timeout = 120000)]
    public async Task Synthesis_IsRepeatable_MelShaped_AndDenormalized()
    {
        await Task.Yield();
        var model = CreateModel();
        var first = model.Synthesize("hello");
        var second = model.Synthesize("hello");
        Assert.Equal(2, first.Rank);
        Assert.Equal(MelBins, first.Shape[1]);
        Assert.Equal(first.ToVector().ToArray(), second.ToVector().ToArray());
        for (int i = 0; i < first.Length; i++) Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(first[i]), $"mel[{i}] = {first[i]}");

        // The output is x·std + mean of the same normalized sample.
        var options = (MatchaTTSOptions)model.GetOptions();
        options.MelMean = -5.0;
        options.MelStd = 2.0;
        var shifted = model.Synthesize("hello");
        for (int i = 0; i < first.Length; i++) Assert.Equal(first[i] * 2.0 - 5.0, shifted[i], 9);
    }

    [Fact(Timeout = 120000)]
    public async Task MoreEulerSteps_ChangeTheSolution()
    {
        await Task.Yield();
        var model = CreateModel(steps: 2);
        var coarse = model.Synthesize("hello");
        ((MatchaTTSOptions)model.GetOptions()).NumFlowSteps = 6;
        var fine = model.Synthesize("hello");
        Assert.Equal(coarse.Shape, fine.Shape);
        Assert.True(Enumerable.Range(0, coarse.Length).Any(i => Math.Abs(coarse[i] - fine[i]) > 1e-9));
    }

    [Fact(Timeout = 60000)]
    public async Task RotaryEncoder_HasNoRelativeTables_AndDependsOnPosition()
    {
        await Task.Yield();
        var block = new RelativePositionTransformerBlock<double>(8, 2, 16, 3, 0.0, 0, rotary: true);
        var relative = new RelativePositionTransformerBlock<double>(8, 2, 16, 3, 0.0, 0, rotary: false);
        // The relative block carries two [1, 4] tables the rotary block does not.
        Assert.Equal(relative.ParameterCount - 8, block.ParameterCount);

        // Two identical tokens at different positions attend differently once positions are rotated in.
        var x = new Tensor<double>(new[] { 3, 8 });
        for (int t = 0; t < 3; t++)
            for (int c = 0; c < 8; c++) x[t, c] = t == 1 ? 0.5 : Math.Cos(c + 1.0);
        var y = block.Forward(x);
        Assert.True(Enumerable.Range(0, 8).Any(c => Math.Abs(y[0, c] - y[2, c]) > 1e-9));
    }

    [Fact(Timeout = 60000)]
    public async Task SnakeBeta_AtInitialization_IsXPlusSinSquared()
    {
        await Task.Yield();
        // log α = log β = 0, so the activation is x + sin²(x) (to the 1e-9 guard) of the projection.
        var snake = new SnakeBetaLayer<double>(2, 3);
        var input = new Tensor<double>(new[] { 1, 2 });
        input[0, 0] = 0.3; input[0, 1] = -0.7;
        var y = snake.Forward(input);
        var z = snake.Projection.Forward(input);
        for (int c = 0; c < 3; c++)
            Assert.Equal(z[0, c] + Math.Pow(Math.Sin(z[0, c]), 2) / (1 + 1e-9), y[0, c], 12);
    }
}
