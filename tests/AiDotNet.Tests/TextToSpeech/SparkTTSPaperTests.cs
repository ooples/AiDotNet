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

/// <summary>
/// Spark-TTS follows Wang et al. (2025): Qwen predicts decoupled global/semantic tokens and BiCodec combines
/// XLSR/Vocos/FVQ semantic coding with ECAPA/Perceiver/FSQ speaker coding and a neural waveform generator.
/// </summary>
public class SparkTTSPaperTests
{
    [Fact]
    public void ReleasedFrontend_UsesBiasedXlsrAndSixSecondSpeakerReference()
    {
        Assert.Equal(6, new SparkTTSOptions().ReferenceSegmentSeconds);
        using var model = Create();
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        var codec = (SparkBiCodec<double>)typeof(SparkTTS<double>).GetField("_biCodec", flags)!.GetValue(model)!;
        Assert.True(codec.Wav2Vec.Configuration.ConvBias);
        var raw = new Tensor<double>(new[] { 4 }, new Vector<double>(new[] { 1d, 2d, 3d, 4d }));
        var normalized = codec.NormalizeXlsrWaveform(raw).ToArray();
        for (int i = 0; i < 4; i++) Assert.Equal((i + 1 - 2.5) / Math.Sqrt(1.25 + 1e-7), normalized[i], 13);
    }

    [Theory]
    [InlineData("", false)]
    [InlineData("wav2vec2.", false)]
    [InlineData("", true)]
    [InlineData("wav2vec2.", true)]
    public void XlsrAuthorImport_AcceptsBackboneAndPretrainingNamespaces(string prefix, bool legacyWeightNorm)
    {
        var state = new System.Collections.Generic.Dictionary<string, Tensor<float>>
        {
            [prefix + "feature_extractor.conv_layers.0.conv.weight"] = new Tensor<float>(new[] { 1, 1, 2 }, new Vector<float>(new[] { .25f, -.5f })),
            [prefix + "feature_projection.layer_norm.weight"] = new Tensor<float>(new[] { 2 }, new Vector<float>(new[] { 1f, 2f })),
            ["project_q.weight"] = new Tensor<float>(new[] { 1 }, new Vector<float>(new[] { 99f })),
        };
        string positional = prefix + "encoder.pos_conv_embed.conv.";
        state[positional + (legacyWeightNorm ? "weight_v" : "parametrizations.weight.original1")] = new Tensor<float>(new[] { 1, 1, 2 }, new Vector<float>(new[] { 3f, 4f }));
        state[positional + (legacyWeightNorm ? "weight_g" : "parametrizations.weight.original0")] = new Tensor<float>(new[] { 1, 1, 2 }, new Vector<float>(new[] { 5f, 6f }));
        var reader = SparkTTS<double>.AuthorStateReader(state, "wav2vec2.");
        Assert.Equal(new[] { .25, -.5 }, reader("feature_extractor.conv_layers.0.conv.weight", new[] { 1, 1, 2 }));
        Assert.Equal(new[] { 1d, 2d }, reader("feature_projection.layer_norm.weight", new[] { 2 }));
        Assert.Equal(new[] { 3d, 4d }, reader("encoder.pos_conv_embed.conv.parametrizations.weight.original1", new[] { 1, 1, 2 }));
        Assert.Equal(new[] { 5d, 6d }, reader("encoder.pos_conv_embed.conv.parametrizations.weight.original0", new[] { 1, 1, 2 }));
        Assert.Throws<InvalidDataException>(() => reader("feature_projection.layer_norm.weight", new[] { 1, 2 }));
        Assert.Throws<InvalidDataException>(() => reader("encoder.missing.weight", new[] { 2 }));
    }

    [Fact]
    public void AnnotatedMixedObjective_UpdatesQwenWhileKeepingBiCodecFrozen()
    {
        using var model = Create();
        model.EncodeToTokens(Audio()); // Materialize ECAPA's input-width-dependent convolution parameters before comparing fitted state.
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        var codec = (SparkBiCodec<double>)typeof(SparkTTS<double>).GetField("_biCodec", flags)!.GetValue(model)!;
        var before = model.GetParameters().ToArray();
        var beforeCodec = codec.TrainableLayers.Select(layer => layer.GetParameters().ToArray()).ToArray();
        double loss = model.TrainAnnotatedSample(new TtsTrainingSample<double> { Tokens = Ids(3, 4), Audio = Audio() },
            true, SparkTTSControlLevel.Moderate, SparkTTSControlLevel.High, 2, 3);
        Assert.True(TtsPaperNumerics.IsFinite(loss));
        Assert.False(before.SequenceEqual(model.GetParameters().ToArray()));
        for (int i = 0; i < codec.TrainableLayers.Count; i++)
            Assert.True(beforeCodec[i].SequenceEqual(codec.TrainableLayers[i].GetParameters().ToArray()),
                $"Frozen BiCodec layer {i} ({codec.TrainableLayers[i].GetType().Name}) changed during LM training.");
    }

    [Fact]
    public void BiCodecAdversarialTraining_UpdatesBothEndsOfTheWaveformGenerator()
    {
        using var model = Create(SmallOptions(SparkTTSTrainingComponent.BiCodec));
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        var codec = typeof(SparkTTS<double>).GetField("_biCodec", flags)!.GetValue(model)!;
        var generator = codec.GetType().GetField("_waveGenerator", flags)!.GetValue(codec)!;
        var first = (AiDotNet.NeuralNetworks.Layers.LayerBase<double>)generator.GetType().GetField("_first", flags)!.GetValue(generator)!;
        var last = (AiDotNet.NeuralNetworks.Layers.LayerBase<double>)generator.GetType().GetField("_last", flags)!.GetValue(generator)!;
        var beforeFirst = first.GetParameters().ToArray();
        var beforeLast = last.GetParameters().ToArray();
        double loss = model.Train(new TtsTrainingSample<double> { Text = "hi", Tokens = Ids(3, 4), Audio = Audio() });
        Assert.True(TtsPaperNumerics.IsFinite(loss));
        Assert.False(beforeFirst.SequenceEqual(first.GetParameters().ToArray()));
        Assert.False(beforeLast.SequenceEqual(last.GetParameters().ToArray()));
    }

    internal static SparkTTSOptions SmallOptions(
        SparkTTSTrainingComponent component = SparkTTSTrainingComponent.LanguageModel) => new()
        {
            SampleRate = 1600,
            CodecFrameRate = 10,
            MelChannels = 8,
            HopSize = 16,
            MelFftSize = 32,
            MelWindowSize = 24,
            ReferenceSegmentSeconds = 0.08,
            VocabSize = 128,
            TextSymbols = Enumerable.Range(0, 26).Select(i => ((char)('a' + i)).ToString()).Concat(new[] { " ", ".", ",", "!", "?", "'" }).ToArray(),
            ConstrainedSpeechDecoding = true,
            TaskTtsToken = 64, TaskControllableTtsToken = 65, StartContentToken = 66, EndContentToken = 67,
            StartGlobalToken = 68, EndGlobalToken = 69, StartSemanticToken = 70, EndSemanticToken = 71,
            StartStyleLabelToken = 72, EndStyleLabelToken = 73, GenderTokenOffset = 74,
            PitchLabelOffset = 76, SpeedLabelOffset = 81, EosToken = 86,
            StartStylePromptToken = 87, EndStylePromptToken = 88, PitchValueOffset = 89, SpeedValueOffset = 100,
            CodebookSize = 16,
            NumCodebooks = 1,
            GlobalCodebookSize = 16,
            GlobalTokenCount = 2,
            GlobalTokenOffset = 32,
            SemanticTokenOffset = 48,
            LLMDim = 16,
            NumLLMLayers = 1,
            NumHeads = 2,
            KeyValueHeads = 1,
            LmIntermediateDim = 32,
            Wav2VecConvDims = new[] { 8, 8 },
            Wav2VecConvKernels = new[] { 4, 2 },
            Wav2VecConvStrides = new[] { 2, 2 },
            Wav2VecHiddenDim = 8,
            Wav2VecLayers = 2,
            Wav2VecHeads = 2,
            Wav2VecIntermediateDim = 16,
            Wav2VecMixedLayers = new[] { 0, 1, 2 },
            BiCodecDim = 8,
            VocosDim = 8,
            VocosIntermediateDim = 16,
            SemanticEncoderLayers = 1,
            PrenetLayers = 1,
            PostnetLayers = 1,
            SemanticCodeDimension = 2,
            SpeakerEmbeddingDim = 8,
            SpeakerLatentDim = 8,
            SpeakerEcapaChannels = 8,
            PerceiverHeads = 1,
            PerceiverDepth = 1,
            FsqLevels = new[] { 2, 2, 2, 2 },
            WaveDecoderChannels = 32,
            WaveDecoderRates = new[] { 4, 4, 5, 2 },
            WaveDecoderKernels = new[] { 8, 8, 10, 4 },
            MaxCodecFrames = 2,
            MinimumGeneratedCodecFrames = 2,
            TopK = 4,
            TrainingComponent = component,
            TrainingMelWindows = new[] { 16, 32, 64 },
            TrainingMelBins = new[] { 4, 8, 16 },
            DiscriminatorWindows = new[] { 64 },
            DiscriminatorChannels = 2,
            DiscriminatorWidthDivisor = 64,
        };

    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 8, outputSize: 16);

    private static SparkTTS<double> Create(SparkTTSOptions? options = null) =>
        new(Architecture(), options ?? SmallOptions());

    private static Tensor<double> Audio(int samples = 320, double phase = 0)
    {
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++)
            audio[i] = 0.4 * Math.Sin(0.17 * i + phase) + 0.1 * Math.Cos(0.037 * i);
        return audio;
    }

    private static Tensor<double> Ids(params int[] ids)
    {
        var tensor = new Tensor<double>(new[] { ids.Length });
        for (int i = 0; i < ids.Length; i++) tensor[i] = ids[i];
        return tensor;
    }

    [Fact]
    public void ReferenceDefaults_MatchReleasedSparkTts05B()
    {
        var options = new SparkTTSOptions();

        Assert.Equal(896, options.LLMDim);
        Assert.Equal(24, options.NumLLMLayers);
        Assert.Equal(14, options.NumHeads);
        Assert.Equal(2, options.KeyValueHeads);
        Assert.Equal(8192, options.CodebookSize);
        Assert.Equal(32, options.GlobalTokenCount);
        Assert.Equal(4096, options.GlobalCodebookSize);
        Assert.Equal(320, options.WaveDecoderRates.Aggregate(1, (product, rate) => product * rate));
    }

    [Fact]
    public void NativeModel_ExposesTheReleasedQwenAndBiCodecPipeline()
    {
        var model = Create();
        var metadata = model.GetModelMetadata();
        var domain = model.GetInputDomain(new[] { 4 });
        Assert.True(domain.IsIndices);
        Assert.Equal(0, domain.MinInclusive);
        Assert.Equal(128, domain.MaxExclusive);

        Assert.True(model.HasPaperPipeline);
        Assert.Equal("Qwen2.5-0.5B -> BiCodec(Vocos+FVQ, ECAPA+Perceiver+FSQ, WaveGenerator)",
            metadata.AdditionalInfo["Architecture"]);
        Assert.Equal(16, metadata.AdditionalInfo["SemanticVocabulary"]);
        Assert.Equal(2, metadata.AdditionalInfo["GlobalTokenCount"]);
        Assert.Equal(true, metadata.AdditionalInfo["OfficialWeightsAvailable"]);
    }

    [Fact]
    public void BiCodec_ProducesDecoupledSemanticAndGlobalTokenStreams()
    {
        var model = Create();

        var semantic = model.EncodeToTokens(Audio());
        var global = model.EncodeGlobalTokens(Audio(320, 0.7));

        Assert.NotEmpty(semantic.ToArray());
        Assert.All(semantic.ToArray(), token => Assert.InRange(token, 0, 15));
        Assert.Equal(2, global.Length);
        Assert.All(global.ToArray(), token => Assert.InRange(token, 0, 15));
    }

    [Fact]
    public void WaveGenerator_DecodesSemanticTokensWithSpeakerConditioning()
    {
        var model = Create();
        var semantic = Ids(1, 7);

        var first = model.DecodeFromTokens(semantic, Ids(0, 0));
        var second = model.DecodeFromTokens(semantic, Ids(15, 15));

        Assert.Equal(320, first.Length);
        Assert.All(first.ToArray(), value => Assert.True(TtsPaperNumerics.IsFinite(value)));
        Assert.Contains(first.ToArray(), value => Math.Abs(value) > 1e-9);
        Assert.True(first.ToArray().Zip(second.ToArray(), (a, b) => Math.Abs(a - b)).Any(delta => delta > 1e-9));
    }

    [Fact]
    public void ControllablePrompt_GeneratesGlobalChainOfThoughtThenSemanticSpeech()
    {
        var model = Create();

        var speech = model.SynthesizeControlled(
            "hello", female: true, SparkTTSControlLevel.High, SparkTTSControlLevel.Moderate);

        Assert.Equal(320, speech.Length);
        Assert.All(speech.ToArray(), value => Assert.True(TtsPaperNumerics.IsFinite(value)));
    }

    [Theory]
    [InlineData(SparkTTSTrainingComponent.LanguageModel)]
    [InlineData(SparkTTSTrainingComponent.BiCodec)]
    public void DisclosedTrainingObjectives_AreFinite(SparkTTSTrainingComponent component)
    {
        var model = Create(SmallOptions(component));
        var sample = new TtsTrainingSample<double>
        {
            Text = "abc",
            Tokens = Ids(3, 4, 5),
            Audio = Audio(),
        };

        double objective = model.EvaluateTrainingObjective(sample);

        Assert.True(TtsPaperNumerics.IsFinite(objective), $"Objective was {objective}.");
        Assert.True(objective >= 0);
    }

    [Fact]
    public void LanguageModelTraining_UpdatesNativeParameters()
    {
        var model = Create();
        var sample = new TtsTrainingSample<double>
        {
            Text = "abc",
            Tokens = Ids(3, 4, 5),
            Audio = Audio(),
        };
        var before = model.GetParameters().ToArray();

        double loss = model.Train(sample);
        var after = model.GetParameters().ToArray();

        Assert.True(TtsPaperNumerics.IsFinite(loss));
        Assert.True(before.Zip(after, (a, b) => Math.Abs(a - b)).Any(delta => delta > 0));
    }

    [Fact]
    public void ConvertedCheckpoint_RequiresAndRestoresTheCompleteNativeGraph()
    {
        var source = Create();
        var expected = source.GetParameters().ToArray();
        var restored = Create();

        restored.LoadConvertedWeights(name => name == "sparktts.parameters" ? expected : null);

        Assert.Equal(expected, restored.GetParameters().ToArray());
        Assert.Throws<InvalidDataException>(() =>
            restored.LoadConvertedWeights(_ => new double[expected.Length - 1]));
    }
}
