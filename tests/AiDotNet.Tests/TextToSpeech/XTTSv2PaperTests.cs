using AiDotNet.NeuralNetworks;
using AiDotNet.Enums;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Regression tests for the released XTTS-v2 dVAE/GPT/Perceiver/H-ASP/HiFi-GAN graph.</summary>
public class XTTSv2PaperTests
{
    internal static XTTSv2Options SmallOptions() => new()
    {
        LLMDim = 16,
        NumLLMLayers = 1,
        NumHeads = 4,
        ConditioningBlocks = 1,
        ConditioningHeads = 4,
        PerceiverDepth = 1,
        PerceiverLatents = 2,
        PerceiverHeads = 1,
        TextVocabularySize = 32,
        MaxTextLength = 16,
        MaxCodecFrames = 1,
        CodebookSize = 8,
        AudioVocabularySize = 10,
        StartAudioToken = 8,
        StopAudioToken = 9,
        DvaeHiddenDim = 8,
        DvaeCodebookDim = 8,
        DvaeResidualBlocks = 1,
        SpeakerEmbeddingDim = 8,
        SpeakerEncoderFilters = [8, 8, 8, 8],
        DecoderInitialChannels = 16,
        DecoderUpsampleRates = [2, 2],
        DecoderUpsampleKernels = [4, 4],
        DecoderResblockKernels = [3],
        DecoderResblockDilations = [[1]],
        ConditioningFftSize = 256,
        ConditioningWindowSize = 256,
        ConditioningHopSize = 64,
        DvaeFftSize = 256,
        DvaeWindowSize = 256,
        DvaeHopSize = 64,
        MaximumReferenceSeconds = 0.34,
        ConditioningSeconds = 0.34,
        ConditioningChunkSeconds = 0.34,
        MinimumConditioningChunkSeconds = 0.1,
        DropoutRate = 0,
        TopK = 2,
        TopP = 1,
        SamplingSeed = 7,
        LearningRate = 1e-4,
    };

    private static XTTSv2CloneOptions SmallCloneOptions()
    {
        var source = SmallOptions();
        return new XTTSv2CloneOptions
        {
            LLMDim = source.LLMDim,
            NumLLMLayers = source.NumLLMLayers,
            NumHeads = source.NumHeads,
            ConditioningBlocks = source.ConditioningBlocks,
            ConditioningHeads = source.ConditioningHeads,
            PerceiverDepth = source.PerceiverDepth,
            PerceiverLatents = source.PerceiverLatents,
            PerceiverHeads = source.PerceiverHeads,
            TextVocabularySize = source.TextVocabularySize,
            MaxTextLength = source.MaxTextLength,
            MaxCodecFrames = source.MaxCodecFrames,
            CodebookSize = source.CodebookSize,
            AudioVocabularySize = source.AudioVocabularySize,
            StartAudioToken = source.StartAudioToken,
            StopAudioToken = source.StopAudioToken,
            DvaeHiddenDim = source.DvaeHiddenDim,
            DvaeCodebookDim = source.DvaeCodebookDim,
            DvaeResidualBlocks = source.DvaeResidualBlocks,
            SpeakerEmbeddingDim = source.SpeakerEmbeddingDim,
            SpeakerEncoderFilters = (int[])source.SpeakerEncoderFilters.Clone(),
            DecoderInitialChannels = source.DecoderInitialChannels,
            DecoderUpsampleRates = (int[])source.DecoderUpsampleRates.Clone(),
            DecoderUpsampleKernels = (int[])source.DecoderUpsampleKernels.Clone(),
            DecoderResblockKernels = (int[])source.DecoderResblockKernels.Clone(),
            DecoderResblockDilations = source.DecoderResblockDilations.Select(x => (int[])x.Clone()).ToArray(),
            ConditioningFftSize = source.ConditioningFftSize,
            ConditioningWindowSize = source.ConditioningWindowSize,
            ConditioningHopSize = source.ConditioningHopSize,
            DvaeFftSize = source.DvaeFftSize,
            DvaeWindowSize = source.DvaeWindowSize,
            DvaeHopSize = source.DvaeHopSize,
            MaximumReferenceSeconds = source.MaximumReferenceSeconds,
            ConditioningSeconds = source.ConditioningSeconds,
            ConditioningChunkSeconds = source.ConditioningChunkSeconds,
            MinimumConditioningChunkSeconds = source.MinimumConditioningChunkSeconds,
            DropoutRate = 0,
            TopK = source.TopK,
            TopP = source.TopP,
            SamplingSeed = source.SamplingSeed,
            LearningRate = source.LearningRate,
            MinReferenceDurationSec = 0.1,
        };
    }

    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.Generative, inputSize: 16, outputSize: 12);

    private static XTTSv2<double> Create(XTTSv2Options? options = null) => new(Architecture(), options ?? SmallOptions());

    private static Tensor<double> Audio(int samples = 8160)
    {
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++)
            audio[i] = 0.2 * Math.Sin(2 * Math.PI * 180 * i / 24000.0)
                + 0.04 * Math.Sin(2 * Math.PI * 311 * i / 24000.0);
        return audio;
    }

    private static Tensor<double> Ids(params int[] ids)
    {
        var tensor = new Tensor<double>(new[] { ids.Length });
        for (int i = 0; i < ids.Length; i++) tensor[i] = ids[i];
        return tensor;
    }

    [Fact]
    public void ReferenceDefaults_MatchReleasedCheckpointAndPaperRecipe()
    {
        var options = new XTTSv2Options();

        Assert.Equal(22050, options.InputSampleRate);
        Assert.Equal(24000, options.SampleRate);
        Assert.Equal(1024, options.CodebookSize);
        Assert.Equal(1026, options.AudioVocabularySize);
        Assert.Equal(1024, options.StartAudioToken);
        Assert.Equal(1025, options.StopAudioToken);
        Assert.Equal(6681, options.TextVocabularySize);
        Assert.Equal(6681, options.PaperTextVocabularySize);
        Assert.Equal(30, options.NumLLMLayers);
        Assert.Equal(1024, options.LLMDim);
        Assert.Equal(16, options.NumHeads);
        Assert.Equal(6, options.ConditioningBlocks);
        Assert.Equal(32, options.PerceiverLatents);
        Assert.Equal(new[] { 8, 8, 2, 2 }, options.DecoderUpsampleRates);
        Assert.Equal(5e-5, options.LearningRate);
        Assert.Equal(5e-6, XTTSv2Options.OfficialGptFineTuning().LearningRate);
        Assert.Equal(17, XTTSv2Options.ReleasedLanguages.Length);
        Assert.Equal(22050d / 1024, options.AcousticTokenRate, 10);
    }

    [Fact]
    public void NativeGraph_ExposesEveryReleasedComponent()
    {
        var model = Create();
        var metadata = model.GetModelMetadata();

        Assert.True(model.HasPaperPipeline);
        Assert.Equal(1, model.TransformerLayerCount);
        Assert.Equal(1, model.ConditioningAttentionBlocks);
        Assert.Equal(2, model.ConditioningLatentCount);
        Assert.Equal(4, model.DvaeDownsampleFactor);
        Assert.Equal(1, model.CodecTokenCodebooks);
        Assert.Equal(8, model.CodecTokenVocabulary);
        Assert.Equal(true, metadata.AdditionalInfo["OfficialWeightsAvailable"]);
        Assert.Contains("Perceiver", metadata.AdditionalInfo["Architecture"]?.ToString());
    }

    [Fact]
    public void RawText_RequiresReleasedTokenizerAndNormalizerUsesLanguageMarkersPreprocessing()
    {
        var model = Create();

        Assert.Throws<InvalidOperationException>(() => model.TokenizeText("hello", "en"));
        Assert.Equal("hello world", XTTSv2<double>.NormalizeText("  HELLO   WORLD  ", "en"));
        Assert.Equal("istanbul", XTTSv2<double>.NormalizeText("İSTANBUL", "tr"));
        Assert.Throws<NotSupportedException>(() => XTTSv2<double>.NormalizeText("hello", "xx"));
    }

    [Fact]
    public void Dvae_ProducesSingleCodeStreamAndReconstructsMel()
    {
        var model = Create();

        var codes = model.EncodeToTokens(Audio());
        var mel = model.DecodeTokensToMel(codes);

        Assert.Equal(1, codes.Shape.Length);
        Assert.All(codes.ToArray(), value => Assert.InRange(value, 0, 7));
        Assert.Equal(80, mel.Shape[1]);
        Assert.True(mel.Shape[0] >= codes.Length * 4 - 3);
        Assert.All(mel.ToArray(), value => Assert.True(TtsPaperNumerics.IsFinite(value)));
    }

    [Fact]
    public void ReferenceEncoders_ReturnPerceiverSequenceAndNormalizedHaspVector()
    {
        var model = Create();

        var condition = model.ExtractConditioningLatents(Audio());
        var speaker = model.ExtractSpeakerEmbedding(Audio());

        Assert.Equal(new[] { 2, 16 }, condition.Shape.ToArray());
        Assert.Equal(8, speaker.Length);
        double norm = Math.Sqrt(speaker.ToArray().Sum(value => value * value));
        Assert.Equal(1, norm, 5);
    }

    [Fact]
    public void NativeSynthesis_UsesReferenceAndReturnsFiniteWaveform()
    {
        var model = Create();
        Assert.Throws<InvalidOperationException>(() => model.Synthesize("hello"));

        var waveform = model.SynthesizeTokens(Ids(2, 3, 4), Audio());

        Assert.NotEmpty(waveform.ToArray());
        Assert.All(waveform.ToArray(), value => Assert.True(TtsPaperNumerics.IsFinite(value)));
    }

    [Fact]
    public void ReleasedGptObjective_IsFiniteAndUpdatesTrainableGraph()
    {
        var model = Create();
        var sample = new TtsTrainingSample<double>
        {
            Tokens = Ids(2, 3, 4),
            Audio = Audio(),
            SpeakerReference = Audio(),
            CodecTokens = Ids(1, 2)
        };
        double objective = model.EvaluateTrainingObjective(sample);
        var before = model.GetParameters().ToArray();

        double loss = model.Train(sample);
        var after = model.GetParameters().ToArray();

        Assert.True(TtsPaperNumerics.IsFinite(objective));
        Assert.True(objective >= 0);
        Assert.True(TtsPaperNumerics.IsFinite(loss));
        Assert.True(before.Zip(after, (a, b) => Math.Abs(a - b)).Any(delta => delta > 0));
    }

    [Fact]
    public void ConvertedCheckpoint_IsStrictAndRestoresWholePaperGraph()
    {
        var source = Create();
        var expected = source.GetParameters().ToArray();
        var restored = Create();

        restored.LoadConvertedWeights(name => name == "xttsv2.parameters" ? expected : null);

        Assert.Equal(expected, restored.GetParameters().ToArray());
        Assert.Throws<InvalidDataException>(() => restored.LoadConvertedWeights(_ => new double[expected.Length - 1]));
    }

    [Fact]
    public void Clone_IsThinFacadeAndEnforcesEnrollmentDuration()
    {
        var options = SmallCloneOptions();
        var clone = new XTTSv2Clone<double>(Architecture(), options);

        Assert.True(clone.HasPaperPipeline);
        Assert.Equal(0.1, clone.MinReferenceDuration);
        Assert.Equal(8, clone.SpeakerEmbeddingDim);
        Assert.Throws<ArgumentException>(() => clone.SynthesizeWithVoice("hello", Audio(100)));
        Assert.Equal(8, clone.ExtractSpeakerEmbedding(Audio()).Length);
        Assert.Equal("VoiceCloning", clone.GetModelMetadata().AdditionalInfo["TaskFacade"]);
    }

    [Fact]
    public void CopyConstructor_DeepCopiesMutableArchitectureArrays()
    {
        var original = SmallOptions();
        var copy = new XTTSv2Options(original);
        copy.DecoderUpsampleRates[0] = 9;
        copy.DecoderResblockDilations[0][0] = 7;
        copy.SpeakerEncoderFilters[0] = 16;
        copy.LearningRateMilestones[0] = 1;

        Assert.Equal(2, original.DecoderUpsampleRates[0]);
        Assert.Equal(1, original.DecoderResblockDilations[0][0]);
        Assert.Equal(8, original.SpeakerEncoderFilters[0]);
        Assert.Equal(5000, original.LearningRateMilestones[0]);
    }
}
