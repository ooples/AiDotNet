using System.Text.Json;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.FlowDiffusion;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class DiTToPromptConditioningTests
{
    [Theory]
    [InlineData("old", "new")]
    [InlineData("an earlier sentence", "next words")]
    [InlineData("a considerably longer reference sentence with many words", "new")]
    public void PublicCrossSentencePath_ConditionsOnJointReferenceAndTargetText(string transcript, string target)
    {
        using var inference = new NoGradScope<double>();
        using var model = Tiny();
        var promptMel = new Tensor<double>(new[] { 6, 4 });
        int[] promptIds = model.TokenizeText(transcript);
        var promptTokens = new Tensor<double>(new[] { promptIds.Length }, new Vector<double>(promptIds.Select(x => (double)x).ToArray()));
        model.Voice = new TtsVoice<double> { Reference = promptMel, ReferenceTokens = promptTokens };
        Assert.Equal(model.TokenizeText(transcript + " " + target), model.TextWithPrompt(target));
        // Independent wiring expectation: run the verified text encoder once on
        // the complete transcript, rather than concatenate contextual features.
        var encoded = model.TextEncoder!.Forward(model.TokenizeText(transcript + " " + target));
        var expected = model.SynthesizeFromEncoding(encoded, model.MelVae!.Encode(promptMel));
        var actual = model.Synthesize(target);
        Assert.Equal(expected.Length, actual.Length);
        Assert.Equal(expected.ToArray(), actual.ToArray());
        Assert.Equal(promptIds.Select(x => (double)x), promptTokens.ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void WithoutATranscribedAcousticPrompt_TargetTextRemainsTheCondition(bool tokensWithoutAudio)
    {
        using var inference = new NoGradScope<double>();
        using var model = Tiny();
        var promptMel = tokensWithoutAudio ? null : new Tensor<double>(new[] { 6, 4 });
        model.Voice = new TtsVoice<double> { Reference = promptMel,
            ReferenceTokens = tokensWithoutAudio ? new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { 10.0, 2.0 })) : null };
        Assert.Equal(model.TokenizeText("new words"), model.TextWithPrompt("new words"));
        var encoded = model.TextEncoder!.Forward(model.TokenizeText("new words"));
        var expected = model.SynthesizeFromEncoding(encoded, promptMel is null ? null : model.MelVae!.Encode(promptMel));
        Assert.Equal(expected.ToArray(), model.Synthesize("new words").ToArray());
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(81.0)]
    [InlineData(0.5)]
    public void InvalidReferenceTranscriptIds_AreRejectedBeforeSynthesis(double id)
    {
        using var model = Tiny();
        model.Voice = new TtsVoice<double> { Reference = new Tensor<double>(new[] { 6, 4 }),
            ReferenceTokens = new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { id, 2.0 })) };
        Assert.Throws<ArgumentException>(() => model.Synthesize("new words"));
    }

    private static DiTToTTS<double> Tiny()
    {
        var options = DiTToTTSPaperTests.TinyOptions();
        options.TextHiddenDim = 6;
        options.LatentDim = 4;
        options.DiTLayers = 2;
        var model = new DiTToTTS<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options);
        model.SetTrainingMode(false);
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "ditto_diffusion_reference.json")));
        model.DiffusionTransformer!.Load((name, shape) => document.RootElement.GetProperty("weights")
            .GetProperty(name).EnumerateArray().Select(value => value.GetDouble()).ToArray(), "");
        // Keep the real length network but fix its posterior for short rollouts.
        var bias = Enumerable.Repeat(-40.0, options.MaxLatentLength + 1).ToArray();
        bias[2] = 40;
        TorchParameters.Linear(AiDotNetEngine.Current, model.LengthPredictor!.Classifier, options.LengthHiddenDim,
            options.MaxLatentLength + 1, new double[(options.MaxLatentLength + 1) * options.LengthHiddenDim], bias);
        return model;
    }
}
