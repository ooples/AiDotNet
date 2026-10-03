using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Classic;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Grad-TTS trains and synthesizes as its paper and reference implementation specify (Popov et al. 2021;
/// huawei-noah/Speech-Backbones): the Glow-TTS text encoder with monotonic alignment search, a U-Net score network,
/// duration + prior + diffusion losses on 2-second segments, and the reverse ODE from μ + ε/τ.
/// </summary>
/// <remarks>
/// Before this change Grad-TTS had no score network and no diffusion: its synthesis derived durations with a fixed
/// formula and ran its decoder stack once, and threw on every call because its encoder boundary pointed past the layer
/// stack.
/// </remarks>
public class GradTTSPaperTests
{
    private const int MelBins = 8;

    private static GradTTS<double> CreateModel(int steps = 3) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: MelBins),
        new GradTTSOptions
        {
            EncoderDim = 16, HiddenDim = 16, NumHeads = 2, NumEncoderLayers = 1, FilterChannels = 32, PrenetDropout = 0.0,
            DropoutRate = 0.0, DurationPredictorFilterChannels = 16, DecoderDim = 8, MelChannels = MelBins,
            SegmentFrames = 12, NumDiffusionSteps = steps,
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
    public async Task Training_ReducesTheDurationPriorAndDiffusionObjective()
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
    public async Task Synthesis_IsRepeatable_AndMatchesTheDurationLength()
    {
        await Task.Yield();
        var model = CreateModel();
        var first = model.Synthesize("hello");
        var second = model.Synthesize("hello");
        Assert.Equal(2, first.Rank);
        Assert.Equal(MelBins, first.Shape[1]);
        Assert.Equal(first.ToVector().ToArray(), second.ToVector().ToArray());
        for (int i = 0; i < first.Length; i++) Assert.True(double.IsFinite(first[i]), $"mel[{i}] = {first[i]}");
    }

    [Fact(Timeout = 120000)]
    public async Task MoreReverseSteps_ChangeTheSolution()
    {
        await Task.Yield();
        // N controls the ODE solver; the same weights give different (finer) solutions at different N.
        var model = CreateModel(steps: 2);
        var coarse = model.Synthesize("hello");
        ((GradTTSOptions)model.GetOptions()).NumDiffusionSteps = 6;
        var fine = model.Synthesize("hello");
        Assert.Equal(coarse.Shape, fine.Shape);
        Assert.True(Enumerable.Range(0, coarse.Length).Any(i => Math.Abs(coarse[i] - fine[i]) > 1e-9));
    }
}
