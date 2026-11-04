using System;
using System.IO;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Seed-TTS follows the four-block pipeline disclosed in Figure 1 of Anastassiou et al. (2024), rather than the
/// former generic sequential stack and handcrafted sine-wave clone.
/// </summary>
public class SeedTTSPaperTests
{
    [Theory]
    [InlineData(SeedTTSTrainingComponent.TokenDiffusion)]
    [InlineData(SeedTTSTrainingComponent.TimbreSelfDistillation)]
    public void DiffusionTraining_UpdatesOnlyItsStageAndKeepsEvaluationRepeatable(SeedTTSTrainingComponent stage)
    {
        using var model = Create(SmallOptions(stage));
        model.DisableAutoStreaming();
        var sample = new TtsTrainingSample<double>
        {
            Tokens = Ids(3, 4, 5), Audio = Audio(24), CodecTokens = Ids(1, 2, 3),
            PromptCodecTokens = Ids(4, 5, 6), SpeakerReference = Audio(24, .8)
        };
        double initial = model.EvaluateTrainingObjective(sample);
        Assert.Equal(initial, model.EvaluateTrainingObjective(sample), 12);
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        var components = (List<LayerBase<double>>)typeof(TtsModelBase<double>).GetField("ComponentLayers", flags)!.GetValue(model)!;
        var all = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(model.Layers.Concat(components.Cast<AiDotNet.Interfaces.ILayer<double>>()).ToList());
        var diffusionLayers = (List<LayerBase<double>>)typeof(SeedTTS<double>).GetField("_diffusionLayers", flags)!.GetValue(model)!;
        var selected = new HashSet<Tensor<double>>(AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(
            diffusionLayers.Cast<AiDotNet.Interfaces.ILayer<double>>().ToList()));
        var before = all.ToDictionary(parameter => parameter, parameter => parameter.ToArray());
        Assert.True(TtsPaperNumerics.IsFinite(model.Train(sample)));
        Assert.Contains(all, parameter => selected.Contains(parameter) &&
            parameter.ToArray().Zip(before[parameter], (a, b) => Math.Abs(a - b)).Any(delta => delta > 0));
        foreach (var parameter in all.Where(parameter => !selected.Contains(parameter)))
            Assert.Equal(before[parameter], parameter.ToArray());
        double after = model.EvaluateTrainingObjective(sample);
        Assert.Equal(after, model.EvaluateTrainingObjective(sample), 12);
    }

    [Fact]
    public void DiffusionSchedule_CopyPreservesCanonicalAndCloningOptions()
    {
        var canonical = new SeedTTSOptions { DiffusionTrainingSteps = 64, DiffusionSteps = 4,
            DiffusionBetaStart = .002, DiffusionBetaEnd = .08 };
        var clone = new SeedTTSCloneOptions { DiffusionTrainingSteps = 80, DiffusionSteps = 5,
            DiffusionBetaStart = .003, DiffusionBetaEnd = .09 };
        foreach (var (source, copy) in new[] { (canonical, new SeedTTSOptions(canonical)),
            ((SeedTTSOptions)clone, (SeedTTSOptions)new SeedTTSCloneOptions(clone)) })
        {
            Assert.Equal(source.DiffusionTrainingSteps, copy.DiffusionTrainingSteps);
            Assert.Equal(source.DiffusionSteps, copy.DiffusionSteps);
            Assert.Equal(source.DiffusionBetaStart, copy.DiffusionBetaStart);
            Assert.Equal(source.DiffusionBetaEnd, copy.DiffusionBetaEnd);
        }
    }


    [Fact]
    public void NativeTextFrontend_UsesLosslessUtf8BytesWithoutCharacterCollisions()
    {
        using var model = Create(SmallOptions());
        const string text = "Seed café 中文 🎤";
        Assert.Equal(System.Text.Encoding.UTF8.GetBytes(text).Select(x => (double)x), model.TextToTokens(text).ToArray());
        Assert.False(model.TextToTokens("a").ToArray().SequenceEqual(model.TextToTokens("Ł").ToArray()));
    }

    [Fact]
    public void ReducedTextVocabulary_RejectsUnsupportedBytesInsteadOfWrappingThem()
    {
        var options = SmallOptions(); options.VocabSize = 64;
        using var model = Create(options);
        Assert.Throws<ArgumentException>(() => model.TextToTokens("A"));
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(.5)]
    [InlineData(256)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void InvalidTokenIds_AreRejectedInsteadOfClamped(double value)
    {
        using var model = Create(SmallOptions());
        var input = new Tensor<double>(new[] { 1 }); input[0] = value;
        Assert.ThrowsAny<ArgumentException>(() => model.Predict(input));
    }

    [Fact]
    public void TokenizerPretraining_UpdatesEncoderCodebookAndWaveformDecoder()
    {
        using var model = Create(SmallOptions(SeedTTSTrainingComponent.SpeechTokenizer));
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        var tokenizer = (SeedTTSSpeechTokenizer<double>)typeof(SeedTTS<double>).GetField("_speechTokenizer", flags)!.GetValue(model)!;
        var vocoder = (SeedTTSAcousticVocoder<double>)typeof(SeedTTS<double>).GetField("_acousticVocoder", flags)!.GetValue(model)!;
        var encoderBefore = tokenizer.Layers[0].GetParameters().ToArray();
        var codebookBefore = tokenizer.Quantizer.Codebook(0).ToArray();
        var decoderBefore = vocoder.Layers.Last().GetParameters().ToArray();
        double loss = model.Train(new TtsTrainingSample<double> { Tokens = Ids(3, 4), Audio = Audio(32) });
        Assert.True(TtsPaperNumerics.IsFinite(loss));
        Assert.False(encoderBefore.SequenceEqual(tokenizer.Layers[0].GetParameters().ToArray()));
        Assert.False(codebookBefore.SequenceEqual(tokenizer.Quantizer.Codebook(0).ToArray()));
        Assert.False(decoderBefore.SequenceEqual(vocoder.Layers.Last().GetParameters().ToArray()));
    }

    private static SeedTTSOptions SmallOptions(SeedTTSTrainingComponent stage = SeedTTSTrainingComponent.TokenLanguageModel)
        => new()
        {
            SampleRate = 800,
            CodecRates = new[] { 2, 2, 2 },
            TokenizerChannels = 4,
            TokenizerCodeDimension = 2,
            TrainingMelWindows = new[] { 16, 32 },
            TrainingMelBins = new[] { 4, 8 },
            DiscriminatorWindows = new[] { 64 },
            DiscriminatorChannels = 2,
            DiscriminatorWidthDivisor = 64,
            CodecFrameRate = 100,
            VocabSize = 256,
            CodebookSize = 16,
            NumCodebooks = 1,
            LLMDim = 16,
            TextEncoderDim = 16,
            NumLLMLayers = 1,
            NumHeads = 2,
            AcousticDimension = 8,
            DiffusionLayers = 1,
            DiffusionSteps = 2,
            VocoderHiddenDimension = 16,
            SpeakerEmbeddingDim = 4,
            MaxCodecFrames = 12,
            MaxPromptCodecFrames = 4,
            FramesPerTextToken = 1,
            MinimumGeneratedCodecFrames = 2,
            TopK = 4,
            TrainingComponent = stage,
        };

    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 8, outputSize: 16);

    private static SeedTTS<double> Create(SeedTTSOptions? options = null) =>
        new(Architecture(), options ?? SmallOptions(),
            new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private static Tensor<double> Audio(int samples, double phase = 0)
    {
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++) audio[i] = 0.4 * Math.Sin(0.31 * i + phase) + 0.1 * Math.Cos(0.07 * i);
        return audio;
    }

    private static Tensor<double> Ids(params int[] ids)
    {
        var tensor = new Tensor<double>(new[] { ids.Length });
        for (int i = 0; i < ids.Length; i++) tensor[i] = ids[i];
        return tensor;
    }

    [Fact]
    public void NativeModel_ContainsAllFourDisclosedBlocks_AndMasksTextLoss()
    {
        var model = Create();
        var metadata = model.GetModelMetadata();

        Assert.True(model.HasPaperPipeline);
        Assert.Equal("SpeechTokenizer -> CausalTokenLM -> TokenDiffusion -> AcousticVocoder",
            metadata.AdditionalInfo["Architecture"]);
        Assert.Equal(true, metadata.AdditionalInfo["TextLossMasked"]);
        Assert.Equal(1, model.CodecTokenCodebooks);
        Assert.Equal(16, model.CodecTokenVocabulary);
    }

    [Fact]
    public void TokenizerDiffusionAndVocoder_RoundTripThroughLearnedModules()
    {
        var model = Create();
        var audio = Audio(24);

        var tokens = model.EncodeToTokens(audio);
        var reconstructed = model.DecodeFromTokens(tokens);

        Assert.Equal(3, tokens.Length);
        Assert.All(tokens.ToArray(), value => Assert.InRange(value, 0, 15));
        Assert.Equal(24, reconstructed.Length);
        Assert.All(reconstructed.ToArray(), value => Assert.True(TtsPaperNumerics.IsFinite(value)));
        Assert.Contains(reconstructed.ToArray(), value => Math.Abs(value) > 1e-9);
    }

    [Fact]
    public void AudioPrompt_ChangesTimbreConditionedGeneration()
    {
        var model = Create();

        var first = model.SynthesizeWithPrompt("seed", "prompt", Audio(24, 0.0));
        var second = model.SynthesizeWithPrompt("seed", "prompt", Audio(24, 1.2));

        Assert.True(first.Length > 0 && second.Length > 0);
        // The acoustic prompt can change LM token count/EOS timing. Hold semantic tokens fixed below
        // so this test also verifies timbre conditioning rather than merely observing a duration change.
        Assert.True(first.Length != second.Length || first.ToArray().Zip(second.ToArray(), (a, b) => Math.Abs(a - b)).Any(delta => delta > 1e-9));
        var firstTimbre = model.ExtractTimbreEmbedding(Audio(24, 0.0));
        var secondTimbre = model.ExtractTimbreEmbedding(Audio(24, 1.2));
        Assert.True(firstTimbre.ToArray().Zip(secondTimbre.ToArray(), (a, b) => Math.Abs(a - b)).Any(delta => delta > 1e-9));
        var diffusion = (SeedTTSTokenDiffusion<double>)typeof(SeedTTS<double>).GetField("_tokenDiffusion",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.GetValue(model)!;
        var firstAcoustic = diffusion.Refine(Ids(1, 5, 7, 2), firstTimbre);
        var secondAcoustic = diffusion.Refine(Ids(1, 5, 7, 2), secondTimbre);
        Assert.Equal(firstAcoustic.Shape.ToArray(), secondAcoustic.Shape.ToArray());
        Assert.True(firstAcoustic.ToArray().Zip(secondAcoustic.ToArray(), (a, b) => Math.Abs(a - b)).Any(delta => delta > 1e-9));
    }

    [Fact]
    public void LanguageModelTraining_UsesCodecTargetsAndUpdatesParameters()
    {
        var model = Create();
        var sample = new TtsTrainingSample<double>
        {
            Text = "abc",
            Tokens = Ids(3, 4, 5),
            CodecTokens = Ids(1, 5, 7, 2),
        };
        var before = model.GetParameters().ToArray();

        double loss = model.Train(sample);
        var after = model.GetParameters().ToArray();

        Assert.True(TtsPaperNumerics.IsFinite(loss));
        Assert.True(before.Zip(after, (a, b) => Math.Abs(a - b)).Any(delta => delta > 0));
    }

    [Theory]
    [InlineData(SeedTTSTrainingComponent.TokenDiffusion)]
    [InlineData(SeedTTSTrainingComponent.AcousticVocoder)]
    [InlineData(SeedTTSTrainingComponent.TimbreSelfDistillation)]
    public void ComponentObjectives_AreFinite(SeedTTSTrainingComponent stage)
    {
        var model = Create(SmallOptions(stage));
        var sample = new TtsTrainingSample<double>
        {
            Text = "abc",
            Tokens = Ids(3, 4, 5),
            Audio = Audio(24),
            CodecTokens = Ids(1, 2, 3),
            PromptCodecTokens = Ids(4, 5, 6),
            SpeakerReference = Audio(24, 0.8),
        };

        double objective = model.EvaluateTrainingObjective(sample);

        Assert.True(TtsPaperNumerics.IsFinite(objective));
        Assert.True(objective >= 0);
    }

    [Fact]
    public void ConvertedCheckpoint_RestoresEveryPipelineComponent()
    {
        var source = Create();
        var expected = source.GetParameters().ToArray();
        var restored = Create();

        restored.LoadConvertedWeights(name => name == "seedtts.parameters" ? expected : null);

        Assert.Equal(expected, restored.GetParameters().ToArray());
        Assert.Throws<InvalidDataException>(() => restored.LoadConvertedWeights(_ => new double[expected.Length - 1]));
    }

    [Fact]
    public void Clone_IsATaskFacadeOverTheSamePipeline()
    {
        var options = new SeedTTSCloneOptions
        {
            CodecRates = new[] { 2, 2, 2 },
            TokenizerChannels = 4,
            TokenizerCodeDimension = 2,
            SampleRate = 800,
            CodecFrameRate = 100,
            VocabSize = 256,
            CodebookSize = 16,
            NumCodebooks = 1,
            LLMDim = 16,
            TextEncoderDim = 16,
            NumLLMLayers = 1,
            NumHeads = 2,
            AcousticDimension = 8,
            DiffusionLayers = 1,
            DiffusionSteps = 2,
            VocoderHiddenDimension = 16,
            SpeakerEmbeddingDim = 4,
            MaxCodecFrames = 8,
            MaxPromptCodecFrames = 4,
            FramesPerTextToken = 1,
            MinimumGeneratedCodecFrames = 2,
            TopK = 4,
            MinReferenceDurationSec = 0.01,
            ReferenceText = "reference",
        };
        var clone = new SeedTTSClone<double>(Architecture(), options,
            new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

        var embedding = clone.ExtractSpeakerEmbedding(Audio(16));
        var speech = clone.SynthesizeWithVoice("hello", Audio(16));

        Assert.True(clone.HasPaperPipeline);
        Assert.Equal(4, embedding.Length);
        Assert.NotEmpty(speech.ToArray());
        Assert.Equal("Seed-TTS", clone.GetModelMetadata().AdditionalInfo["Architecture"]);
    }
}
