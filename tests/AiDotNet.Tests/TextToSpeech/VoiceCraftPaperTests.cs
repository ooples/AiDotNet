using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Regression tests for VoiceCraft's GigaSpeech codec, causal masking and delayed stacking.</summary>
public class VoiceCraftPaperTests
{
    [Theory]
    [InlineData("MinimumGeneratedFrames")]
    [InlineData("StopSilenceRepetition")]
    [InlineData("SilenceTokens")]
    [InlineData("SampleBatchSize")]
    [InlineData("MaxCodecFrames")]
    [InlineData("MaxTextLength")]
    public void NativeOptionContextsExerciseTheirBoundaryAndRestoreReadoutParameters(string option)
    {
        var options = SmallOptions();
        VoiceCraftOptionFixture.PrepareOptions(options, option);
        using var model = Create(options);
        model.Voice = model.CreateVoice(Audio(), "hello");
        var control = VoiceCraftOptionFixture.Observe(model, option);
        var parameters = model.GetParameters().ToArray();
        switch (option)
        {
            case "MinimumGeneratedFrames": options.MinimumGeneratedFrames = 3; break;
            case "StopSilenceRepetition": options.StopSilenceRepetition = 6; break;
            case "SilenceTokens": options.SilenceTokens = new[] { 2 }; break;
            case "SampleBatchSize": options.SampleBatchSize = 1; break;
            case "MaxCodecFrames": options.MaxCodecFrames = 4; break;
            case "MaxTextLength": options.MaxTextLength = 4; break;
        }
        Assert.NotNull(control);
        Assert.NotEqual(control, VoiceCraftOptionFixture.Observe(model, option));
        Assert.Equal(parameters, model.GetParameters().ToArray());
    }

    [Theory]
    [InlineData("SampleRate", 8000)]
    [InlineData("SampleRate", 24000)]
    [InlineData("HopSize", 160)]
    [InlineData("HopSize", 640)]
    [InlineData("CodecFrameRate", 25)]
    [InlineData("CodecFrameRate", 75)]
    [InlineData("NumCodebooks", 2)]
    [InlineData("CodebookSize", 16)]
    public void NativeCodecRejectsUnpairedGeometry(string property, int value)
    {
        var options = SmallOptions();
        typeof(VoiceCraftOptions).GetProperty(property)!.SetValue(options, value);
        Assert.Throws<ArgumentException>(() => Create(options));
    }

    [Theory]
    [InlineData(2, 8)]
    [InlineData(4, 16)]
    public void PairedCodecGeometryEncodesAndDecodesNativeAudio(int codebooks, int vocabulary)
    {
        var options = SmallOptions();
        options.NumCodebooks = options.Codec.NumQuantizers = codebooks;
        options.CodebookSize = options.Codec.CodebookSize = vocabulary;
        options.CodebookLossWeights = Enumerable.Range(0, codebooks).Select(q => q == 0 ? 5.0 : 1.0).ToArray();
        double bandwidth = codebooks * 50.0 * Math.Log(vocabulary, 2) / 1000;
        options.Codec.TargetBandwidthKbps = bandwidth;
        options.Codec.TargetBandwidths = new[] { bandwidth };
        using var model = Create(options);
        var codes = model.EncodeToTokens(Audio());
        Assert.Equal(2 * codebooks, codes.Length);
        var audio = model.DecodeFromTokens(codes);
        Assert.Equal(640, audio.Length);
        Assert.All(audio.ToArray(), value => Assert.True(TtsPaperNumerics.IsFinite(value)));
        Assert.Equal(16000, model.SampleRate);
        Assert.Equal(50, model.CodecFrameRate);
    }

    [Fact]
    public void EdenUpdatesTheLearningRateUsedByScaledAdam()
    {
        var options = SmallOptions();
        options.UseScaledAdam = true;
        options.LearningRate = .02;
        options.EdenWarmupSteps = 2;
        options.EdenStep = 4;
        options.EdenEpoch = 2;
        options.PseudoEpochSteps = 2;
        using var model = Create(options);
        var codes = new Tensor<double>(new[] { 12, options.NumCodebooks });
        for (int i = 0; i < codes.Length; i++) codes[i] = i % options.CodebookSize;
        var sample = new TtsTrainingSample<double> { Text = "hello", Tokens = model.TokenizeText("hello"), CodecTokens = codes };
        var optimizer = (AiDotNet.Interfaces.IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>)typeof(VoiceCraft<double>)
            .GetField("_optimizer", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.GetValue(model)!;
        model.Train(sample);
        Assert.Equal(.01, optimizer.GetCurrentLearningRate(), 12);
        model.Train(sample);
        Assert.True(optimizer.GetCurrentLearningRate() > .01);
    }

    [Fact]
    public void NestedCodecConstructionPreservesTheTransformerOptionsSeed()
    {
        var previousScope = AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.CaptureScope();
        int? previousFallback = AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.AmbientFallbackSeed;
        try
        {
            AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.AmbientFallbackSeed = 1337;
            double[] Snapshot(int optionsSeed, int? architectureSeed = null)
            {
                var options = SmallOptions();
                options.Seed = optionsSeed;
                using var model = new VoiceCraft<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
                    NeuralNetworkTaskType.Generative, inputSize: 4, outputSize: 1) { RandomSeed = architectureSeed }, options);
                return model.GetParameters().ToArray();
            }
            Assert.Equal(Snapshot(7), Snapshot(7));
            Assert.False(Snapshot(7).SequenceEqual(Snapshot(11)));
            Assert.Equal(Snapshot(7, 2294), Snapshot(11, 2294));
        }
        finally
        {
            AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.AmbientFallbackSeed = previousFallback;
            AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.RestoreScope(previousScope);
        }
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void PublicGeneration_UsesDistinctTtsAndEditingEndTokensAndDurationGates(bool tts)
    {
        var options = SmallOptions();
        options.TopK = 1;
        options.SampleBatchSize = 3;
        using var model = Create(options);
        model.Voice = model.CreateVoice(Audio(), "hello");
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        var network = (VoiceCraftNetwork<double>)typeof(VoiceCraft<double>).GetField("_network", flags)!.GetValue(model)!;
        var heads = ((AiDotNet.NeuralNetworks.Layers.DenseLayer<double> Hidden,
            AiDotNet.NeuralNetworks.Layers.DenseLayer<double> Output)[])typeof(VoiceCraftNetwork<double>)
            .GetField("_predictionHeads", flags)!.GetValue(network)!;
        for (int q = 0; q < options.NumCodebooks; q++)
        {
            var bias = Enumerable.Repeat(-10000.0, options.AudioVocabularySize).ToArray();
            bias[1] = 0;
            if (q == 0)
            {
                bias[tts ? options.EndOfUtteranceToken : options.EndOfGenerationToken] = 100;
                // The opposite mode's end marker must be ignored even if larger.
                bias[tts ? options.EndOfGenerationToken : options.EndOfUtteranceToken] = 200;
            }
            TorchParameters.Linear(AiDotNetEngine.Current, heads[q].Output, options.CodebookSize / 2,
                options.AudioVocabularySize, new double[options.CodebookSize / 2 * options.AudioVocabularySize], bias);
        }
        var waveform = tts ? model.Synthesize("hello") : model.EditSpeech("hello", Audio(), new[] { (1, 2) });
        // TTS suppresses EOS through step 1, retaining two generated frames.
        // Editing permits an immediate EOG, removing its second frame while
        // preserving the first original frame through the real EnCodec decoder.
        Assert.Equal((tts ? 2 : 1) * options.HopSize, waveform.Length);
        Assert.All(waveform.ToArray(), value => Assert.True(TtsPaperNumerics.IsFinite(value)));
    }

    [Fact]
    public void OptionsCopy_PreservesPhoneFrontendAndReleasedCodecConfiguration()
    {
        var original = new VoiceCraftOptions { TextPhonemizer = _ => new[] { "a" } };
        var copy = new VoiceCraftOptions(original);
        Assert.Same(original.TextPhonemizer, copy.TextPhonemizer);
        Assert.NotSame(original.Codec, copy.Codec);
        Assert.False(copy.Codec.TimeGroupNormalization);
        Assert.False(copy.Codec.ReflectPadding);
        Assert.Equal(128, copy.Codec.Dimension);
        Assert.Equal(new[] { 3, 1 }, copy.Codec.ResidualKernelSizes);
    }

    [Theory]
    [InlineData("")]
    [InlineData("model.")]
    public void AuthorTransformerImport_AcceptsOriginalWrapperAndRawState(string prefix)
    {
        var state = new Dictionary<string, Tensor<float>>
        {
            [prefix + "text_embedding.word_embeddings.weight"] = new Tensor<float>(new[] { 2, 2 }, new Vector<float>(new[] { 1f, 2f, 3f, 4f })),
        };
        var read = VoiceCraft<double>.AuthorTransformerReader(state);
        Assert.Equal(new[] { 1d, 2d, 3d, 4d }, read("text_embedding.word_embeddings.weight", new[] { 2, 2 }));
        Assert.Throws<InvalidDataException>(() => read("text_embedding.word_embeddings.weight", new[] { 4, 1 }));
        Assert.Throws<InvalidDataException>(() => read("decoder.missing.weight", new[] { 2, 2 }));
    }

    private static IReadOnlyList<string> RecordedPhones(string text)
    {
        using var document = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "voicecraft_text_reference.json")));
        return document.RootElement.GetProperty("cases").EnumerateArray().First(item => item.GetProperty("text").GetString() == text)
            .GetProperty("phones").EnumerateArray().Select(phone => phone.GetString()!).ToArray();
    }

    [Fact]
    public void CheckpointPhoneVocabularyPreservesIdsAndCopiesTheMapping()
    {
        using var model = new VoiceCraft<double>(Architecture(), SmallOptions());
        var domain = model.GetInputDomain(new[] { 4 });
        Assert.True(domain.IsIndices);
        Assert.Equal(0, domain.MinInclusive);
        Assert.Equal(100, domain.MaxExclusive);
        var vocabulary = VoiceCraftPhoneTokenizer.Vocabulary.ToDictionary(kv => kv.Key, kv => 79 - kv.Value);
        int[] expected = VoiceCraftPhoneTokenizer.Tokenize("hello", 32, phonemizer: RecordedPhones).Select(id => 79 - id).ToArray();
        model.SetPhoneVocabulary(vocabulary);
        vocabulary.Clear();
        Assert.Equal(expected.Select(id => (double)id), model.TokenizeText("hello").ToArray());
        Assert.Equal(80, model.PhoneVocabularySize);
        Assert.Throws<ArgumentException>(() => model.SetPhoneVocabulary(new Dictionary<string, int> { ["a"] = 100 }));
        Assert.Throws<ArgumentException>(() => model.SetPhoneVocabulary(new Dictionary<string, int> { ["a"] = 1, ["b"] = 1 }));
    }

    internal static VoiceCraftOptions SmallOptions()
    {
        var options = new VoiceCraftOptions
        {
            TextPhonemizer = RecordedPhones,
            LLMDim = 16,
            FeedForwardDim = 32,
            NumLLMLayers = 2,
            NumHeads = 4,
            TextVocabularySize = 100,
            MaxTextLength = 32,
            MaxCodecFrames = 8,
            MinimumGeneratedFrames = 1,
            MaximumFramesPerPhone = 1,
            SampleBatchSize = 1,
            TopK = 4,
            TopP = 1,
            TextEmbeddingDropout = 0,
            AudioEmbeddingDropout = 0,
            TextPositionDropout = 0,
            AudioPositionDropout = 0,
            TransformerDropout = 0,
            CodebookSize = 8,
            CodebookLossWeights = [5, 1, 0.5, 0.1],
            SilenceTokens = [1, 2, 3],
            SamplingSeed = 7,
            UseScaledAdam = false,
            LearningRate = 1e-4,
        };
        options.Codec = new EnCodecOptions
        {
            SampleRate = 16000,
            Filters = 2,
            Dimension = 8,
            Ratios = [8, 5, 4, 2],
            NumQuantizers = 4,
            CodebookSize = 8,
            TargetBandwidthKbps = 0.6,
            TargetBandwidths = [0.6],
            SegmentSize = 640,
            IncludeDiscriminators = false,
            Causal = true,
            KMeansIterations = 0,
            DeadCodeThreshold = 0,
            LstmLayers = 1,
        };
        return options;
    }

    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.Generative, inputSize: 16, outputSize: 12);

    private static VoiceCraft<double> Create(VoiceCraftOptions? options = null) =>
        new(Architecture(), options ?? SmallOptions());

    private static Tensor<double> Audio(int samples = 640, double phase = 0)
    {
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++)
            audio[i] = 0.25 * Math.Sin(0.07 * i + phase) + 0.05 * Math.Cos(0.031 * i);
        return audio;
    }

    private static Tensor<double> Ids(params int[] ids)
    {
        var tensor = new Tensor<double>(new[] { ids.Length });
        for (int i = 0; i < ids.Length; i++) tensor[i] = ids[i];
        return tensor;
    }

    private static Tensor<double> Codes(int frames = 4)
    {
        var tensor = new Tensor<double>(new[] { frames, 4 });
        for (int frame = 0; frame < frames; frame++)
            for (int q = 0; q < 4; q++) tensor[frame, q] = (frame * 3 + q) % 8;
        return tensor;
    }

    [Fact]
    public void ReferenceDefaults_MatchPaperTableAndTrainingRecipe()
    {
        var options = new VoiceCraftOptions();

        Assert.Equal(16000, options.SampleRate);
        Assert.Equal(320, options.HopSize);
        Assert.Equal(50, options.CodecFrameRate);
        Assert.Equal(4, options.NumCodebooks);
        Assert.Equal(128, options.Codec.Dimension);
        Assert.Equal(64, options.Codec.Filters);
        Assert.False(options.Codec.Causal);
        Assert.False(options.Codec.TimeGroupNormalization);
        Assert.False(options.Codec.ReflectPadding);
        Assert.True(options.Codec.TrueSkip);
        Assert.Equal(new[] { 3, 1 }, options.Codec.ResidualKernelSizes);
        Assert.Equal(2048, options.CodebookSize);
        Assert.Equal(2048, options.LLMDim);
        Assert.Equal(16, options.NumLLMLayers);
        Assert.Equal(16, options.NumHeads);
        Assert.Equal(8192, options.FeedForwardDim);
        Assert.Equal(new[] { 5.0, 1.0, 0.5, 0.1 }, options.CodebookLossWeights);
        Assert.Equal(0.8, options.TopP);
        Assert.Equal(3, options.SampleBatchSize);
        Assert.Equal(50000, options.TotalTrainingSteps);
        Assert.Equal(3000, options.EdenStep);
        Assert.Equal(500, options.EdenWarmupSteps);
    }

    [Fact]
    public void ReleasedPresets_EncodeCheckpointSpecificHeadCountsAndTtsRecipe()
    {
        var large = VoiceCraftOptions.Official830M();
        var small = VoiceCraftOptions.Official330M();
        var tts = VoiceCraftOptions.Official330MTtsEnhanced();

        Assert.Equal(16, large.NumHeads);
        Assert.Equal(40, large.TopK);
        Assert.Equal(1024, small.LLMDim);
        Assert.Equal(24, small.NumLLMLayers);
        Assert.Equal(16, small.NumHeads);
        Assert.Equal(120, tts.TextVocabularySize);
        Assert.Equal(new[] { 2.0, 1.0, 1.0, 1.0 }, tts.CodebookLossWeights);
        Assert.False(tts.UseScaledAdam);
        Assert.Equal(500000, tts.TotalTrainingSteps);
    }

    [Fact]
    public void NativeModel_ExposesCodecCausalTransformerAndOfficialVocabulary()
    {
        var model = Create();
        var metadata = model.GetModelMetadata();

        Assert.True(model.HasPaperPipeline);
        Assert.Equal(2, model.TransformerLayerCount);
        Assert.Equal(80, model.PhoneVocabularySize);
        Assert.Equal(4, model.CodecTokenCodebooks);
        Assert.Equal(8, model.CodecTokenVocabulary);
        Assert.Equal(true, metadata.AdditionalInfo["OfficialWeightsAvailable"]);
        Assert.Contains("causal decoder Transformer", metadata.AdditionalInfo["Architecture"]?.ToString());
    }

    [Fact]
    public void ReleasedPhoneFrontEnd_ProducesOnlyExactPhn2numIds()
    {
        var values = Create().TokenizeText("hello world").ToArray();

        Assert.NotEmpty(values);
        Assert.All(values, value => Assert.InRange(value, 0, 79));
    }

    [Fact]
    public void DelayedStacking_ShiftsEachCodebookByItsIndex()
    {
        var model = Create();
        var input = new Tensor<double>(new[] { 2, 4 });
        input[0, 0] = 1; input[0, 1] = 2; input[0, 2] = 3; input[0, 3] = 4;
        input[1, 0] = 5; input[1, 1] = 6; input[1, 2] = 7; input[1, 3] = 0;

        var delayed = model.DelayCodecTokens(input);

        Assert.Equal(new[] { 5, 4 }, delayed.Shape.ToArray());
        Assert.Equal(new[]
        {
            1d, 8, 8, 8,
            5, 2, 8, 8,
            8, 6, 3, 8,
            8, 8, 7, 4,
            8, 8, 8, 0,
        }, delayed.ToArray());
    }

    [Fact]
    public void GigaSpeechCodecPath_EncodesFourStreamsAndDecodesWaveform()
    {
        var model = Create();

        var codes = model.EncodeToTokens(Audio());
        var waveform = model.DecodeFromTokens(codes);

        Assert.True(model.CodecCodebooksInitialized);
        Assert.Equal(4, codes.Shape[1]);
        Assert.All(codes.ToArray(), value => Assert.InRange(value, 0, 7));
        Assert.NotEmpty(waveform.ToArray());
        Assert.All(waveform.ToArray(), value => Assert.True(TtsPaperNumerics.IsFinite(value)));
    }

    [Fact]
    public void ZeroShotSynthesis_RequiresPromptTranscriptAndRunsDelayedGeneration()
    {
        var model = Create();
        Assert.Throws<InvalidOperationException>(() => model.Synthesize("hi"));

        model.Voice = model.CreateVoice(Audio(), "hello", 16000);
        var waveform = model.Synthesize("hi");

        Assert.NotEmpty(waveform.ToArray());
        Assert.All(waveform.ToArray(), value => Assert.True(TtsPaperNumerics.IsFinite(value)));
    }

    [Fact]
    public void CausalMaskAndWeightedCodebookObjective_IsFiniteAndTrainable()
    {
        var model = Create();
        var sample = new TtsTrainingSample<double>
        {
            Text = "hello",
            Tokens = Ids(14, 18, 63),
            CodecTokens = Codes()
        };
        var before = model.GetParameters().ToArray();

        double objective = model.EvaluateTrainingObjective(sample);
        double loss = model.Train(sample);
        var after = model.GetParameters().ToArray();

        Assert.True(TtsPaperNumerics.IsFinite(objective));
        Assert.True(objective >= 0);
        Assert.True(TtsPaperNumerics.IsFinite(loss));
        Assert.True(before.Zip(after, (a, b) => Math.Abs(a - b)).Any(delta => delta > 0));
    }

    [Fact]
    public void ConvertedCheckpoint_IsStrictAndRestoresCodecAndTransformer()
    {
        var source = Create();
        var expected = source.GetParameters().ToArray();
        var restored = Create();

        restored.LoadConvertedWeights(name => name == "voicecraft.parameters" ? expected : null);

        Assert.Equal(expected, restored.GetParameters().ToArray());
        Assert.Throws<InvalidDataException>(() =>
            restored.LoadConvertedWeights(_ => new double[expected.Length - 1]));
    }

    [Fact]
    public void CopyConstructor_DeepCopiesMutablePaperConfiguration()
    {
        var original = SmallOptions();
        var copy = new VoiceCraftOptions(original);
        copy.CodebookLossWeights[0] = 9;
        copy.SilenceTokens[0] = 7;
        copy.Codec.Ratios[0] = 4;

        Assert.Equal(5, original.CodebookLossWeights[0]);
        Assert.Equal(1, original.SilenceTokens[0]);
        Assert.Equal(8, original.Codec.Ratios[0]);
        Assert.Equal(original.LLMDim, copy.LLMDim);
        Assert.Equal(original.SamplingSeed, copy.SamplingSeed);
    }
}
