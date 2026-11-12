using System;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Task-facade contracts that keep CosyVoice cloning on the one released CosyVoice implementation.</summary>
public class CosyVoiceCloneFacadeTests
{
    [Fact]
    public void Options_InheritTheReleasedCosyVoice300MConfiguration()
    {
        var options = new CosyVoiceCloneOptions();

        Assert.IsAssignableFrom<CosyVoiceOptions>(options);
        Assert.Equal(51866, options.VocabSize);
        Assert.Equal(4096, options.CodebookSize);
        Assert.Equal(14, options.NumLLMLayers);
        Assert.Equal(192, options.SpeakerEmbeddingDim);
        Assert.Equal(3.0, options.MinReferenceDurationSec);

        var copy = new CosyVoiceCloneOptions(options);
        options.HiftUpsampleRates[0] = 1;
        options.MinReferenceDurationSec = 9;
        Assert.Equal(8, copy.HiftUpsampleRates[0]);
        Assert.Equal(3.0, copy.MinReferenceDurationSec);
    }

    [Fact]
    public void NativeFacade_ContainsTheRealLanguageModelFlowTokenizerSpeakerEncoderAndVocoder()
    {
        using var model = new CosyVoiceClone<double>(Architecture(), SmallOptions());

        Assert.IsAssignableFrom<CosyVoice<double>>(model);
        Assert.NotEmpty(model.LanguageModelLayers);
        Assert.NotEmpty(model.FlowLayers);
        Assert.NotEmpty(model.VocoderLayers);
        Assert.NotNull(model.FlowModel);
        Assert.NotNull(model.Vocoder);
        Assert.Contains("CosyVoice-Clone", model.GetModelMetadata().Name, StringComparison.Ordinal);
    }

    [Fact]
    public void VoiceCloningFacade_UsesCrossLingualPromptAndRestoresThePreviousVoice()
    {
        var options = SmallOptions();
        options.MinReferenceDurationSec = 0.001;
        using var model = new ProbeClone(Architecture(), options);
        var previous = new TtsVoice<double> { SpeakerEmbedding = new Tensor<double>(new[] { options.SpeakerEmbeddingDim }) };
        model.Voice = previous;
        var reference = new Tensor<double>(new[] { 32 });

        var audio = model.SynthesizeWithVoice("hello", reference);

        Assert.Equal(22050, model.ObservedSampleRate);
        Assert.Null(model.ObservedTranscript);
        Assert.Same(model.CreatedVoice, model.ObservedVoiceDuringSynthesis);
        Assert.Same(previous, model.Voice);
        Assert.Equal(new[] { 0.25, -0.25 }, audio.ToArray());
    }

    [Fact]
    public void VoiceCloningFacade_RejectsReferencesShorterThanItsDocumentedContract()
    {
        var options = SmallOptions();
        options.MinReferenceDurationSec = 0.1;
        using var model = new CosyVoiceClone<double>(Architecture(), options);
        var tooShort = new Tensor<double>(new[] { 2204 });

        var error = Assert.Throws<ArgumentException>(() => model.ExtractSpeakerEmbedding(tooShort));
        Assert.Contains("2205 samples", error.Message, StringComparison.Ordinal);
    }

    private static NeuralNetworkArchitecture<double> Architecture() => new(
        AiDotNet.Enums.InputType.OneDimensional,
        AiDotNet.Enums.NeuralNetworkTaskType.Regression,
        inputSize: 1,
        outputSize: 1);

    private static CosyVoiceCloneOptions SmallOptions() => new()
    {
        TextEncoderDim = 16,
        LLMDim = 16,
        NumEncoderLayers = 1,
        NumLLMLayers = 1,
        NumHeads = 2,
        LmFeedForwardDim = 32,
        CodebookSize = 16,
        FlowDim = 16,
        FlowHeads = 2,
        FlowFeedForwardDim = 32,
        FlowEncoderLayers = 1,
        DecoderChannels = new[] { 16 },
        DecoderHeads = 2,
        DecoderHeadDim = 8,
        DecoderBlocks = 1,
        DecoderMidBlocks = 1,
        FlowTimesteps = 2,
        HiftBaseChannels = 16,
        HiftHarmonics = 2,
        SpeechTokenizerState = 16,
        SpeechTokenizerHeads = 2,
        SpeechTokenizerLayers = 1,
        SpeakerEncoderChannels = 8,
        SpeakerEncoderGrowthRate = 4,
        SpeakerEmbeddingDim = 8,
        MelChannels = 8,
        DropoutRate = 0,
        WarmupSteps = 1,
    };

    private sealed class ProbeClone : CosyVoiceClone<double>
    {
        internal ProbeClone(NeuralNetworkArchitecture<double> architecture, CosyVoiceCloneOptions options)
            : base(architecture, options)
        {
        }

        internal TtsVoice<double> CreatedVoice { get; } = new();
        internal int ObservedSampleRate { get; private set; }
        internal string? ObservedTranscript { get; private set; }
        internal TtsVoice<double>? ObservedVoiceDuringSynthesis { get; private set; }

        public override TtsVoice<double> CreateVoice(Tensor<double> recording, int sampleRate, string? transcript = null)
        {
            ObservedSampleRate = sampleRate;
            ObservedTranscript = transcript;
            return CreatedVoice;
        }

        public override Tensor<double> Synthesize(string text)
        {
            ObservedVoiceDuringSynthesis = Voice;
            var audio = new Tensor<double>(new[] { 2 });
            audio[0] = 0.25;
            audio[1] = -0.25;
            return audio;
        }
    }
}
