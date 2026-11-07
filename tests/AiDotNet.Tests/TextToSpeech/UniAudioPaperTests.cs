using System;
using System.IO;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Regression tests for UniAudio's universal codec and global/local multi-scale Transformer.</summary>
public class UniAudioPaperTests
{
    [Fact]
    public void PublicInputDomain_DescribesPhoneIdsRatherThanCodecAudio()
    {
        using var model = Create();
        var domain = model.GetInputDomain(new[] { 4 });
        Assert.True(domain.IsIndices);
        Assert.Equal(0, domain.MinInclusive);
        Assert.Equal(80, domain.MaxExclusive);
    }

    private static UniAudioOptions SmallOptions() => new()
    {
        SampleRate = 320,
        UseReleasedTextFrontend = false,
        HopSize = 32,
        CodecFrameRate = 10,
        MelChannels = 8,
        NumCodebooks = 3,
        CodebookSize = 8,
        CodecTotalQuantizers = 3,
        CodecFilters = 2,
        CodecDimension = 6,
        CodecRatios = new[] { 4, 2, 2, 2 },
        CodecKMeansIterations = 0,
        CodecDeadCodeThreshold = 0,
        LLMDim = 12,
        NumHeads = 3,
        NumEncoderLayers = 1,
        NumLLMLayers = 1,
        GlobalFeedForwardDim = 24,
        LocalFeedForwardDim = 24,
        JointVocabularySize = 160,
        VocabSize = 160,
        AudioTokenOffset = 32,
        PhoneTokenOffset = 57,
        PhoneVocabularySize = 80,
        MaxTextLength = 8,
        MaxContextPatches = 64,
        MaxCodecFrames = 64,
        PromptSeconds = 0.1,
        TopK = 4,
        Temperature = 0.8,
        MinimumLengthRatio = 0,
        MaximumLengthRatio = 1,
        MinimumGeneratedFrames = 1,
        MaximumGeneratedFrames = 2,
        SamplingSeed = 7,
    };

    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.Generative, inputSize: 8, outputSize: 160);

    private static UniAudio<double> Create(UniAudioOptions? options = null) =>
        new(Architecture(), options ?? SmallOptions());

    private static Tensor<double> Audio(int samples = 96, double phase = 0)
    {
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++)
            audio[i] = 0.3 * Math.Sin(0.17 * i + phase) + 0.09 * Math.Cos(0.071 * i);
        return audio;
    }

    private static Tensor<double> Ids(params int[] ids)
    {
        var tensor = new Tensor<double>(new[] { ids.Length });
        for (int i = 0; i < ids.Length; i++) tensor[i] = ids[i];
        return tensor;
    }

    [Fact]
    public void ReferenceDefaults_MatchPaperAndRelease()
    {
        var options = new UniAudioOptions();

        Assert.Equal(16000, options.SampleRate);
        Assert.Equal(320, options.HopSize);
        Assert.Equal(50, options.CodecFrameRate);
        Assert.Equal(3, options.NumCodebooks);
        Assert.Equal(1024, options.CodebookSize);
        Assert.Equal(12, options.CodecTotalQuantizers);
        Assert.Equal(new[] { 8, 5, 4, 2 }, options.CodecRatios);
        Assert.Equal(24, options.NumEncoderLayers);
        Assert.Equal(8, options.NumLLMLayers);
        Assert.Equal(1536, options.LLMDim);
        Assert.Equal(12, options.NumHeads);
        Assert.Equal(6144, options.GlobalFeedForwardDim);
        Assert.Equal(3000, options.MaxContextPatches);
        Assert.Equal(4212, options.JointVocabularySize);
        Assert.Equal(30, options.TopK);
        Assert.Equal(0.8, options.Temperature);
        Assert.Equal(3.0, options.PromptSeconds);
    }

    [Fact]
    public void NativeModel_ExposesUniversalCodecAndBothTransformerScales()
    {
        var model = Create();
        var metadata = model.GetModelMetadata();

        Assert.True(model.HasPaperPipeline);
        Assert.Equal(1, model.GlobalLayerCount);
        Assert.Equal(1, model.LocalLayerCount);
        Assert.Equal("16 kHz SEANet/RVQ -> 24-layer global Transformer -> 8-layer local Transformer",
            metadata.AdditionalInfo["Architecture"]);
        Assert.Equal(3, metadata.AdditionalInfo["CodecQuantizersUsed"]);
        Assert.Equal(true, metadata.AdditionalInfo["OfficialWeightsAvailable"]);
    }

    [Fact]
    public void CmuDictFrontEnd_ProducesReservedPhonemeIds()
    {
        var model = Create();

        var hello = model.TokenizePhones("hello world").ToArray();

        Assert.NotEmpty(hello);
        Assert.All(hello, id => Assert.InRange(id, 0, 79));
        Assert.Contains(0.0, hello); // the optional native CMUDict frontend's word-boundary token
    }

    [Fact]
    public void UniversalCodec_ProducesThreeOffsetCodeStreamsAtFiftyHertzSemantics()
    {
        var model = Create();

        var tokens = model.EncodeToTokens(Audio());
        var values = tokens.ToArray();
        var wave = model.DecodeFromTokens(tokens);

        Assert.True(model.CodecCodebooksInitialized);
        Assert.Equal(0, values.Length % 3);
        for (int i = 0; i < values.Length; i++)
        {
            int q = i % 3;
            Assert.InRange(values[i], q * 8, (q + 1) * 8 - 1);
        }
        Assert.Equal(values.Length / 3 * 32, wave.Length);
        Assert.All(wave.ToArray(), value => Assert.True(TtsPaperNumerics.IsFinite(value)));
    }

    [Fact]
    public void Synthesis_RequiresSpeakerPromptAndRunsGlobalLocalGenerationAndCodecDecode()
    {
        var model = Create();
        Assert.Throws<InvalidOperationException>(() => model.Synthesize("hello"));

        model.Voice = model.CreateVoice(Audio(), 320);
        var wave = model.Synthesize("hello");

        Assert.NotEmpty(wave.ToArray());
        Assert.All(wave.ToArray(), value => Assert.True(TtsPaperNumerics.IsFinite(value)));
    }

    [Fact]
    public void NextTokenObjective_IsFiniteAndUsesTargetAudio()
    {
        var model = Create();
        var sample = new TtsTrainingSample<double>
        {
            Text = "hello",
            Tokens = Ids(1, 2, 3),
            Audio = Audio(),
            SpeakerReference = Audio(64, 0.8),
        };

        double objective = model.EvaluateTrainingObjective(sample);

        Assert.True(TtsPaperNumerics.IsFinite(objective), $"Objective was {objective}.");
        Assert.True(objective >= 0);
    }

    [Fact]
    public void MultiScaleLanguageModelTraining_UpdatesParameters()
    {
        var model = Create();
        var sample = new TtsTrainingSample<double>
        {
            Text = "hello",
            Tokens = Ids(1, 2, 3),
            Audio = Audio(),
            SpeakerReference = Audio(64, 0.4),
        };
        var before = model.GetParameters().ToArray();

        double loss = model.Train(sample);
        var after = model.GetParameters().ToArray();

        Assert.True(TtsPaperNumerics.IsFinite(loss));
        Assert.True(before.Zip(after, (a, b) => Math.Abs(a - b)).Any(delta => delta > 0));
    }

    [Fact]
    public void ConvertedCheckpoint_IsStrictAndRestoresCodecAndBothScales()
    {
        var source = Create();
        var expected = source.GetParameters().ToArray();
        var restored = Create();

        restored.LoadConvertedWeights(name => name == "uniaudio.parameters" ? expected : null);

        Assert.Equal(expected, restored.GetParameters().ToArray());
        Assert.Throws<InvalidDataException>(() =>
            restored.LoadConvertedWeights(_ => new double[expected.Length - 1]));
    }

    [Fact]
    public void CopyConstructor_PreservesUniAudioSpecificSettingsWithoutAliasingRatios()
    {
        var original = SmallOptions();
        var copy = new UniAudioOptions(original);
        copy.CodecRatios[0] = 2;

        Assert.Equal(4, original.CodecRatios[0]);
        Assert.Equal(original.GlobalFeedForwardDim, copy.GlobalFeedForwardDim);
        Assert.Equal(original.PhoneTokenOffset, copy.PhoneTokenOffset);
        Assert.Equal(original.SamplingSeed, copy.SamplingSeed);
    }
}
