using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.FlowDiffusion;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// E2 TTS and F5-TTS train and sample as their papers specify (Eskimez et al. 2024; Chen et al. 2024): characters padded
/// with filler tokens, the text-guided infilling flow-matching loss over a masked span, classifier-free guidance and
/// sway-sampled Euler steps, with a voice prompt prefixing the condition and the text.
/// </summary>
/// <remarks>
/// Before this change both were declared codec language models; neither had a flow-matching objective or sampler, and
/// synthesis ran their layer stacks once.
/// </remarks>
public class FlowMatchingInfillingPaperTests
{
    private const int MelBins = 8;

    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 8, outputSize: MelBins) { RandomSeed = 11 };

    private static F5TTS<double> CreateF5() => new(Architecture(), new F5TTSOptions
    {
        HiddenDim = 16, NumHeads = 2, HeadDim = 8, NumLayers = 2, TextDim = 8, TextConvLayers = 1, VocabSize = 32,
        MelChannels = MelBins, DropoutRate = 0.0, NumFunctionEvaluations = 4,
    }, new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private static E2TTS<double> CreateE2() => new(Architecture(), new E2TTSOptions
    {
        HiddenDim = 16, NumHeads = 2, HeadDim = 8, NumLayers = 2, VocabSize = 32, MelChannels = MelBins, DropoutRate = 0.0,
        NumFunctionEvaluations = 4,
    }, new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private static (Tensor<double> Tokens, Tensor<double> Mel) Example()
    {
        var tokens = new Tensor<double>(new[] { 5 });
        for (int i = 0; i < 5; i++) tokens[i] = 3 + i * 5;
        var mel = new Tensor<double>(new[] { 16, MelBins });
        for (int f = 0; f < 16; f++)
            for (int c = 0; c < MelBins; c++) mel[f, c] = Math.Sin(0.4 * f + 0.3 * c);
        return (tokens, mel);
    }

    private static double Objective(TtsModelBase<double> model, Tensor<double> tokens, Tensor<double> mel)
        => ((AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model).EvaluateTrainingObjective(tokens, mel);

    [Fact(Timeout = 240000)]
    public async Task F5_Training_ReducesTheInfillingObjective()
    {
        await Task.Yield();
        var model = CreateF5();
        var (tokens, mel) = Example();
        double before = Objective(model, tokens, mel);
        for (int i = 0; i < 40; i++) model.Train(tokens, mel);
        double after = Objective(model, tokens, mel);
        Assert.True(after < before, $"Objective did not fall ({before} -> {after}).");
    }

    [Fact(Timeout = 240000)]
    public async Task E2_Training_ReducesTheInfillingObjective()
    {
        await Task.Yield();
        var model = CreateE2();
        var (tokens, mel) = Example();
        double before = Objective(model, tokens, mel);
        for (int i = 0; i < 40; i++) model.Train(tokens, mel);
        double after = Objective(model, tokens, mel);
        Assert.True(after < before, $"Objective did not fall ({before} -> {after}).");
    }

    [Fact(Timeout = 120000)]
    public async Task Synthesis_WithoutAPrompt_UsesTheCharacterRate_AndIsRepeatable()
    {
        await Task.Yield();
        var model = CreateF5();
        var mel = model.Synthesize("hello");
        Assert.Equal(2, mel.Rank);
        Assert.Equal(MelBins, mel.Shape[1]);
        Assert.Equal((int)Math.Ceiling(5 * 6.0), mel.Shape[0]);
        Assert.Equal(mel.ToVector().ToArray(), model.Synthesize("hello").ToVector().ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task Synthesis_WithAPrompt_FollowsThePromptsSpeakingRate()
    {
        await Task.Yield();
        var model = CreateE2();
        var (tokens, reference) = Example();
        // 16 prompt frames for 5 prompt characters: "hello" (5 characters) gets 16 generated frames.
        model.Voice = new TtsVoice<double> { Reference = reference, ReferenceTokens = tokens };
        var mel = model.Synthesize("hello");
        Assert.Equal(16, mel.Shape[0]);
        for (int i = 0; i < mel.Length; i++) Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(mel[i]));
    }
}
