using System.Text.Json;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.VoiceCloning;
using AiDotNet.TextToSpeech.FlowDiffusion;
using AiDotNet.TextToSpeech.Vocoders;
using AiDotNet.Audio;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class ReleasedTtsModuleReferenceTests
{
    [Fact]
    public void VoiceCraft_EspeakSeparatorPunctuationAndPhoneSplittingMatchAuthorsTokenizer()
    {
        using var document = Read("voicecraft_text_reference.json");
        foreach (var item in document.RootElement.GetProperty("cases").EnumerateArray())
        {
            var calls = item.GetProperty("calls").EnumerateArray().ToArray();
            string Espeak(string language, string text)
            {
                Assert.Equal("en-us", language);
                return calls.First(call => call.GetProperty("text").GetString() == text).GetProperty("output").GetString()!;
            }
            Assert.Equal(item.GetProperty("phones").EnumerateArray().Select(phone => phone.GetString()),
                VoiceCraftPhoneTokenizer.Phonemize(item.GetProperty("text").GetString()!, Espeak));
        }
    }

    [Fact]
    public void Voicebox_MaskedOtcFlowLossAndAllInputGradientsMatchIndependentEquations()
    {
        using var document = Read("voicebox_objective_reference.json"); var f = document.RootElement;
        var engine = AiDotNetEngine.Current;
        Tensor<double> Matrix(string name) => new(new[] { 5, 3 }, new Vector<double>(Values(f.GetProperty(name))));
        var target = Matrix("target"); var prior = Matrix("prior"); var prediction = Matrix("prediction");
        using var tape = new GradientTape<double>();
        var (state, velocity) = VoiceboxObjectives<double>.Flow(engine, target, prior, .37, 1e-5);
        Close(state, Values(f.GetProperty("state")), 1e-12);
        Close(velocity, Values(f.GetProperty("velocity")), 1e-12);
        var loss = VoiceboxObjectives<double>.MaskedMse(engine, engine.TensorSubtract(prediction, velocity), [true, false, true, false, true]);
        Assert.Equal(f.GetProperty("loss").GetDouble(), loss[0], 12);
        var sources = new List<Tensor<double>> { target, prior, prediction };
        var gradients = tape.ComputeGradients(loss, sources: sources);
        string[] keys = ["target_gradient", "prior_gradient", "prediction_gradient"];
        for (int i = 0; i < sources.Count; i++) Close(gradients[sources[i]], Values(f.GetProperty(keys[i])), 1e-12);
        foreach (var item in f.GetProperty("duration_cases").EnumerateArray())
            Assert.Equal(item.GetProperty("value").GetDouble(), VoiceboxObjectives<double>.TransformDuration(
                item.GetProperty("frames").GetInt32(), item.GetProperty("uniform").GetDouble()), 12);
    }

    [Fact]
    public void Spark_EncoderPrenetPostnetAndWaveformMatchReleasedModulesAndSamplingResiduals()
    {
        using var document = Read("spark_codec_modules_reference.json"); var f = document.RootElement;
        var options = new SparkTTSOptions { Wav2VecConvDims = [4], Wav2VecConvKernels = [4], Wav2VecConvStrides = [2],
            Wav2VecHiddenDim = 4, Wav2VecLayers = 1, Wav2VecHeads = 1, Wav2VecIntermediateDim = 8,
            BiCodecDim = 8, VocosDim = 8, VocosIntermediateDim = 16, SemanticEncoderLayers = 1,
            PrenetLayers = 1, PostnetLayers = 1, SpeakerEmbeddingDim = 6, SpeakerEcapaChannels = 8,
            SpeakerLatentDim = 8, GlobalTokenCount = 2, FsqLevels = [4, 4], GlobalCodebookSize = 16,
            PerceiverHeads = 2, PerceiverDepth = 1, CodebookSize = 16, SemanticCodeDimension = 2,
            WaveDecoderChannels = 32, WaveDecoderRates = [2, 2], WaveDecoderKernels = [4, 4] };
        var engine = AiDotNetEngine.Current;
        var model = new SparkBiCodec<double>(engine, new List<LayerBase<double>>(), options, new Random(1));
        model.LoadFeatureWeights(Weights(f));
        Tensor<double> TensorOf(string key, params int[] shape) => new(shape, new Vector<double>(Values(f.GetProperty(key))));
        var encoded = model.EncodeFeatures(TensorOf("input", 1, 4, 5));
        Close(encoded, Values(f.GetProperty("encoded")), 1e-9);
        var pre = model.PrenetFeatures(encoded, TensorOf("condition", 6));
        Close(pre, Values(f.GetProperty("prenet")), 1e-9);
        Close(model.PostnetFeatures(pre), Values(f.GetProperty("postnet")), 1e-9);
        var wave = new SparkWaveGenerator<double>(engine, new List<LayerBase<double>>(), options);
        wave.LoadTorchWeights(Weights(f));
        Close(wave.Forward(TensorOf("wave_input", 1, 8, 5)), Values(f.GetProperty("waveform")), 1e-9);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Spark_VocosAndConditionalAdaLayerNormSgdMatchReleasedModules(bool conditional)
    {
        using var document = Read("spark_vocos_reference.json"); var f = document.RootElement.GetProperty("cases")[conditional ? 1 : 0];
        var layers = new List<LayerBase<double>>();
        var model = new SparkVocosBackbone<double>(AiDotNetEngine.Current, layers, new Random(1), 4, 8, 16, 2, conditional ? 6 : 0);
        model.LoadTorchWeights(Weights(f));
        var input = new Tensor<double>(new[] { 1, 4, 5 }, new Vector<double>(Values(f.GetProperty("input"))));
        var condition = conditional ? new Tensor<double>(new[] { 6 }, new Vector<double>(Values(f.GetProperty("condition")))) : null;
        Close(model.Forward(input, condition), Values(f.GetProperty("output")), 1e-9);
        ReferenceSgdStep(layers, () => model.Forward(input, condition), f, 1e-9);
        Close(model.Forward(input, condition), Values(f.GetProperty("after_sgd")), 1e-9);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Spark_GenerationHonorsEndTokensOutsideSemanticCodeRange(bool constrained)
    {
        var engine = AiDotNetEngine.Current;
        var options = new SparkTTSOptions { VocabSize = 64, LLMDim = 16, NumLLMLayers = 1, NumHeads = 4,
            KeyValueHeads = 2, LmIntermediateDim = 32, TieWordEmbeddings = false, TopK = 1,
            EosToken = 60, EndSemanticToken = 61, SemanticTokenOffset = 32, CodebookSize = 16,
            MaxCodecFrames = 5, MinimumGeneratedCodecFrames = 0, ConstrainedSpeechDecoding = constrained };
        var model = new SparkTokenLanguageModel<double>(engine, new List<LayerBase<double>>(), options);
        var hidden = model.Transformer.Forward(model.Transformer.Embed(new int[,] { { 3 } })).Hidden;
        var head = model.Transformer.LmHead!.Weights;
        for (int i = 0; i < head.Length; i++) head[i] = 0;
        int end = constrained ? options.EndSemanticToken : options.EosToken;
        for (int d = 0; d < 16; d++) head[d, end] = hidden[0, 0, d];
        engine.InvalidatePersistentTensor(head);
        int[] tokens = constrained ? model.GenerateSemantic(new[] { 3 }, new Random(1)) : model.GenerateAll(new[] { 3 }, new Random(1));
        Assert.Empty(tokens);
    }

    [Fact]
    public void Spark_PerceiverFsqIndicesAndEveryCodeDigitMatchReleasedSpeakerModules()
    {
        using var document = Read("spark_global_reference.json"); var f = document.RootElement;
        var options = new SparkTTSOptions { SpeakerEcapaChannels = 8, SpeakerLatentDim = 8,
            SpeakerEmbeddingDim = 6, GlobalTokenCount = 2, PerceiverHeads = 2, PerceiverDepth = 1,
            FsqLevels = [4, 4], GlobalCodebookSize = 16 };
        var model = new SparkSpeakerTokenizer<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), options, new Random(1));
        model.LoadTorchWeights(Weights(f), "", includeEcapa: false);
        var context = new Tensor<double>(new[] { 5, 24 }, new Vector<double>(Values(f.GetProperty("context"))));
        var encoded = model.EncodeContext(context);
        Assert.Equal(Values(f.GetProperty("tokens")), encoded.GlobalTokens.ToArray());
        Close(encoded.DVector, Values(f.GetProperty("dvector")), 1e-9);
        foreach (var item in f.GetProperty("decode_cases").EnumerateArray())
        {
            var ids = Values(item.GetProperty("tokens"));
            Close(model.Decode(new Tensor<double>(new[] { 2 }, new Vector<double>(ids))), Values(item.GetProperty("dvector")), 1e-9);
        }
    }

    [Fact]
    public void Spark_ReleasedMultilingualTokenizerAndAddedSpeechTokensMatchRustTokenizer()
    {
        using var document = Read("spark_text_reference.json");
        var tokenizer = AiDotNet.Tokenization.Algorithms.QwenTokenizer.Spark;
        Assert.Equal(165158, tokenizer.VocabularySize);
        foreach (var item in document.RootElement.GetProperty("cases").EnumerateArray())
            Assert.Equal(item.GetProperty("ids").EnumerateArray().Select(id => id.GetInt32()),
                tokenizer.Encode(item.GetProperty("text").GetString()!));
    }

    [Fact]
    public void Spark_RawSlaneyMagnitudeMelMatchesReleasedTorchaudioConfiguration()
    {
        using var document = Read("spark_mel_reference.json");
        var mel = new CenteredLogMel<double>(AiDotNetEngine.Current, 16000, 1024, 320, 640, 128,
            10, 8000, 1e-5, false, logarithm: false);
        foreach (var item in document.RootElement.GetProperty("cases").EnumerateArray())
        {
            var values = Values(item.GetProperty("waveform"));
            Close(mel.Forward(new Tensor<double>(new[] { values.Length }, new Vector<double>(values))),
                Values(item.GetProperty("output")), 2e-5);
        }
    }

    [Fact]
    public void Spark_QwenLogitsAndSuffixLikelihoodSgdMatchHuggingFace()
    {
        using var document = Read("spark_qwen_reference.json"); var f = document.RootElement;
        var engine = AiDotNetEngine.Current; var layers = new List<LayerBase<double>>();
        var options = new SparkTTSOptions { VocabSize = 64, LLMDim = 16, NumLLMLayers = 1,
            NumHeads = 4, KeyValueHeads = 2, LmIntermediateDim = 32 };
        var model = new SparkTokenLanguageModel<double>(engine, layers, options);
        model.Transformer.LoadTorchWeights(Weights(f));
        int[] tokens = Values(f.GetProperty("tokens")).Select(v => (int)v).ToArray();
        Close(model.Logits(tokens.Take(tokens.Length - 1).ToArray()), Values(f.GetProperty("logits")), 2e-6);
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers.Cast<AiDotNet.Interfaces.ILayer<double>>().ToList());
        Dictionary<Tensor<double>, Tensor<double>> gradients;
        using (var tape = new GradientTape<double>())
        {
            var loss = model.Loss(tokens, f.GetProperty("first_target").GetInt32());
            Assert.True(Math.Abs(loss[0] - f.GetProperty("loss").GetDouble()) < 2e-6);
            gradients = tape.ComputeGradients(loss, sources: parameters);
        }
        foreach (var parameter in parameters)
        {
            if (!gradients.TryGetValue(parameter, out var gradient)) continue;
            for (int i = 0; i < parameter.Length; i++) parameter[i] -= .01 * gradient[i];
            engine.InvalidatePersistentTensor(parameter);
        }
        Close(model.Logits(tokens.Take(tokens.Length - 1).ToArray()), Values(f.GetProperty("after_sgd")), 2e-6);
    }

    [Fact]
    public void Spark_EcapaFrameFeaturesAndSgdStepMatchTheAuthorsRes2AndAggregationLayout()
    {
        using var document = Read("spark_ecapa_reference.json"); var f = document.RootElement;
        var model = new EcapaTdnnBackbone<double>([8, 8, 8, 8, 1536], [5, 3, 3, 3, 1], [1, 2, 3, 4, 1],
            8, 128, 128, 6, sparkLayout: true);
        model.LoadSparkFrameWeights(Weights(f), 4);
        var input = new Tensor<double>(new[] { 1, 4, 5 }, new Vector<double>(Values(f.GetProperty("input"))));
        Close(model.ForwardFrameFeatures(input), Values(f.GetProperty("output")), 1e-9);
        ReferenceSgdStep(model.Layers.Cast<LayerBase<double>>().ToList(), () => model.ForwardFrameFeatures(input), f, 1e-9);
        Close(model.ForwardFrameFeatures(input), Values(f.GetProperty("after_sgd")), 1e-9);
    }

    [Fact]
    public void Spark_AstpPoolingAndSgdStepMatchUnbiasedContextVarianceAndUnnormalizedAttention()
    {
        using var document = Read("spark_astp_reference.json"); var f = document.RootElement;
        var model = new EcapaTdnnBackbone<double>([8, 8, 8, 8, 6], [5, 3, 3, 3, 1], [1, 2, 3, 4, 1],
            8, 128, 4, 6, sparkLayout: true);
        model.LoadSparkPoolingWeights(Weights(f), includeProjection: false, prefix: "");
        var input = new Tensor<double>(new[] { 1, 6, 7 }, new Vector<double>(Values(f.GetProperty("input"))));
        Close(model.PoolStatistics(input), Values(f.GetProperty("output")), 1e-9);
        ReferenceSgdStep(model.Layers.Cast<LayerBase<double>>().ToList(), () => model.PoolStatistics(input), f, 1e-9);
        Close(model.PoolStatistics(input), Values(f.GetProperty("after_sgd")), 1e-9);
    }

    [Fact]
    public void Chatterbox_PerceiverOutputAndSgdStepMatchTheAuthorsModule()
    {
        using var document = Read("chatterbox_perceiver_reference.json"); var f = document.RootElement;
        var layers = new List<LayerBase<double>>();
        var options = new ChatterboxOptions { LLMDim = 16, PerceiverQueries = 2, PerceiverHeads = 2, PerceiverDropout = 0 };
        var model = new ChatterboxPerceiver<double>(AiDotNetEngine.Current, layers, options, new Random(1));
        model.LoadTorchWeights(Weights(f), "");
        var input = new Tensor<double>(new[] { 3, 16 }, new Vector<double>(Values(f.GetProperty("input"))));
        Close(model.Forward(input), Values(f.GetProperty("output")), 1e-9);
        ReferenceSgdStep(layers, () => model.Forward(input), f, 1e-9);
        Close(model.Forward(input), Values(f.GetProperty("after_sgd")), 1e-9);
    }

    [Fact]
    public void CosyVoice3_OfflineStreamingAndSgdStepMatchTheAuthorsDiT()
    {
        using var document = Read("cosyvoice3_dit_reference.json"); var f = document.RootElement;
        var layers = new List<LayerBase<double>>(); var engine = AiDotNetEngine.Current;
        var model = new GptSoVitsDiT<double>(engine, layers, new GptSoVitsDiTConfiguration(Dim: 32, Depth: 1,
            Heads: 2, HeadDim: 16, FfMult: 2, MelDim: 4, TextDim: 4, ConvLayers: 0, Dropout: 0,
            UseStepEmbedding: false, SpeakerDim: 4, CausalPosition: true, StaticChunkSize: 2), new Random(1));
        model.Load(name => Values(f.GetProperty("weights").GetProperty(name)), "");
        Tensor<double> Rows(string name) => new(new[] { 6, 4 }, new Vector<double>(Values(f.GetProperty(name))));
        var speaker = new Tensor<double>(new[] { 4 }, new Vector<double>(Values(f.GetProperty("speaker"))));
        Tensor<double> Run(bool streaming) => model.Forward(Rows("x"), Rows("cond"), model.TextEmbedding(Rows("mu"), false),
            model.TimeEmbedding(.37, null), false, false, null, speaker, streaming);
        Close(Run(false), Values(f.GetProperty("offline")), 2e-9);
        Close(Run(true), Values(f.GetProperty("streaming")), 2e-9);
        ReferenceSgdStep(layers, () => Run(false), f, 2e-9);
        Close(Run(false), Values(f.GetProperty("after_sgd")), 2e-9);
    }

    private static void ReferenceSgdStep(List<LayerBase<double>> layers, Func<Tensor<double>> forward, JsonElement fixture, double tolerance)
    {
        var engine = AiDotNetEngine.Current;
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers.Cast<AiDotNet.Interfaces.ILayer<double>>().ToList());
        Dictionary<Tensor<double>, Tensor<double>> gradients;
        using (var tape = new GradientTape<double>())
        {
            var output = forward();
            var loss = engine.ReduceMean(engine.TensorMultiply(output, output), Enumerable.Range(0, output.Rank).ToArray(), true);
            Assert.True(Math.Abs(loss[0] - fixture.GetProperty("loss").GetDouble()) < tolerance);
            gradients = tape.ComputeGradients(loss, sources: parameters);
        }
        double rate = fixture.GetProperty("learning_rate").GetDouble();
        foreach (var parameter in parameters)
        {
            if (!gradients.TryGetValue(parameter, out var gradient)) continue;
            for (int i = 0; i < parameter.Length; i++) parameter[i] -= rate * gradient[i];
            engine.InvalidatePersistentTensor(parameter);
        }
    }

    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(2)] [InlineData(3)] [InlineData(4)] [InlineData(5)]
    [InlineData(6)] [InlineData(7)] [InlineData(8)] [InlineData(9)] [InlineData(10)]
    public void Tortoise_EnglishCleanerAndExactBpeIdsMatchReleasedVoiceTokenizer(int index)
    {
        using var document = Read("tortoise_text_reference.json"); var f = document.RootElement.GetProperty("cases")[index];
        string text = f.GetProperty("text").GetString()!;
        Assert.Equal(f.GetProperty("cleaned").GetString(), TortoiseTextTokenizer.Clean(text));
        Assert.Equal(f.GetProperty("ids").EnumerateArray().Select(x => x.GetInt32()), TortoiseTextTokenizer.Default.Value.Encode(text));
    }

    [Fact]
    public void Tortoise_UnivNetNamedWeightsWaveformAndSgdStepMatchTheAuthorsVocoder()
    {
        using var document = Read("tortoise_vocoder_reference.json"); var f = document.RootElement;
        var engine = AiDotNetEngine.Current;
        var model = new UnivNetGenerator<double>(engine, 4, 4, 2, [2, 2], [1, 3], .2, 64, 3);
        model.LoadTorchWeights(Weights(f));
        var mel = new Tensor<double>(new[] { 1, 4, 5 }, new Vector<double>(Values(f.GetProperty("mel"))));
        var noise = new Tensor<double>(new[] { 1, 4, 5 }, new Vector<double>(Values(f.GetProperty("noise"))));
        Close(model.Forward(mel, noise), Values(f.GetProperty("waveform")), 1e-9);
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(model.Layers.Cast<AiDotNet.Interfaces.ILayer<double>>().ToList());
        Dictionary<Tensor<double>, Tensor<double>> gradients;
        using (var tape = new GradientTape<double>())
        {
            var waveform = model.Forward(mel, noise);
            var loss = engine.ReduceMean(engine.TensorMultiply(waveform, waveform), new[] { 0, 1, 2 }, true);
            Assert.Equal(f.GetProperty("loss").GetDouble(), loss[0], 9);
            gradients = tape.ComputeGradients(loss, sources: parameters);
        }
        double rate = f.GetProperty("learning_rate").GetDouble();
        foreach (var parameter in parameters)
        {
            if (!gradients.TryGetValue(parameter, out var gradient)) continue;
            for (int i = 0; i < parameter.Length; i++) parameter[i] -= rate * gradient[i];
            engine.InvalidatePersistentTensor(parameter);
        }
        Close(model.Forward(mel, noise), Values(f.GetProperty("after_sgd")), 1e-9);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void Tortoise_BothMelFrontendsMatchAuthorTransformsAndReleasedNormalization(int index)
    {
        using var document = Read("tortoise_mel_reference.json"); var f = document.RootElement.GetProperty("cases")[index];
        var model = new TortoiseMelFrontend<double>(AiDotNetEngine.Current, new TortoiseTTSOptions());
        var samples = Values(f.GetProperty("audio"));
        var audio = new Tensor<double>(new[] { samples.Length }, new Vector<double>(samples));
        var ar = model.Autoregressive(audio); var diffusion = model.Diffusion(audio);
        Assert.Equal(f.GetProperty("ar_frames").GetInt32(), ar.Shape[0]);
        Assert.Equal(f.GetProperty("diffusion_frames").GetInt32(), diffusion.Shape[0]);
        Close(ar, Values(f.GetProperty("autoregressive")), 2e-5);
        Close(diffusion, Values(f.GetProperty("diffusion")), 2e-5);
    }

    [Fact]
    public void Tortoise_OriginalDvaeInferenceAndTwoEmaTrainingForwardsMatchTheAuthors()
    {
        using var document = Read("tortoise_tokenizer_reference.json"); var f = document.RootElement;
        var engine = AiDotNetEngine.Current;
        var options = new TortoiseTTSOptions { AutoregressiveMelChannels = 4, TokenizerLatentDim = 4,
            TokenizerCodeDim = 4, TokenizerResidualBlocks = 1, CodebookSize = 8 };
        var model = new TortoiseSpeechTokenizer<double>(engine, new List<LayerBase<double>>(), options);
        model.LoadTorchWeights(Weights(f));
        var mel = engine.TensorTranspose(new Tensor<double>(new[] { 4, 8 }, new Vector<double>(Values(f.GetProperty("mel")))));
        var codes = Values(f.GetProperty("codes")).Select(x => (int)x).ToArray();
        Assert.Equal(codes, model.Encode(mel));
        Close(model.Decode(codes), Values(f.GetProperty("reconstructed")), 1e-9);
        model.Quantizer.SetTrainingMode(true);
        foreach (var expected in f.GetProperty("training").EnumerateArray())
        {
            var input = engine.TensorTranspose(new Tensor<double>(new[] { 4, 8 }, new Vector<double>(Values(expected.GetProperty("mel")))));
            Close(model.Reconstruct(input), Values(expected.GetProperty("reconstructed")), 1e-9);
            Assert.Equal(expected.GetProperty("commitment").GetDouble(), model.Quantizer.CommitmentLoss![0], 9);
            Close(model.Quantizer.Codebook(0), Values(expected.GetProperty("embed")), 1e-9);
            Assert.Equal(Values(expected.GetProperty("codes")).Select(x => (int)x),
                Enumerable.Range(0, 2).Select(i => model.Quantizer.LastCodes![0, i]));
        }
    }

    [Fact]
    public void Xtts_DiscreteVaeCodesAndReconstructionMatchCoquisActualCodec()
    {
        using var document = Read("xtts_dvae_reference.json"); var f = document.RootElement;
        var options = XttsOptions(); options.MelChannels = 4; options.CodebookSize = 8;
        options.DvaeHiddenDim = 4; options.DvaeCodebookDim = 4; options.DvaeResidualBlocks = 1;
        var model = new XttsDiscreteVae<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), options);
        model.LoadTorchWeights(Weights(f));
        var mel = new Tensor<double>(new[] { 4, 8 }, new Vector<double>(Values(f.GetProperty("mel"))));
        var codes = model.Encode(mel);
        Assert.Equal(Values(f.GetProperty("codes")), codes.ToArray());
        Close(model.Decode(codes), Values(f.GetProperty("reconstructed")), 2e-6);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void Tortoise_SpacedDdpmMatchesOriginalTimestepsVarianceClippingAndGuidanceRamp(int index)
    {
        using var document = Read("tortoise_diffusion_reference.json"); var f = document.RootElement;
        var expected = f.GetProperty("spaced")[index];
        var schedule = new TortoiseDiffusionSchedule<double>(AiDotNetEngine.Current, 4000, 4);
        var noisy = new Tensor<double>(new[] { 1, 4, 9 }, new Vector<double>(Values(f.GetProperty("noisy"))));
        var prediction = new Tensor<double>(new[] { 1, 8, 9 }, new Vector<double>(Values(f.GetProperty("prediction"))));
        var free = new Tensor<double>(new[] { 1, 8, 9 }, new Vector<double>(Values(f.GetProperty("unconditioned"))));
        var actual = schedule.Statistics(noisy, prediction, expected.GetProperty("step").GetInt32(),
            expected.GetProperty("previous").GetInt32(), expected.GetProperty("clipped_variance").GetDouble(),
            free, guidanceIndex: index, guidanceSteps: 3);
        Close(actual.Mean, Values(expected.GetProperty("mean")), 2e-5);
        Close(actual.LogVariance, Values(expected.GetProperty("log_variance")), 2e-5);
        Close(actual.Clean, Values(expected.GetProperty("clean")), 2e-5);
    }

    [Fact]
    public void Tortoise_DiffusionNetworkSgdStepMatchesTheAuthorsConditioningAndDecoderGradients()
    {
        using var document = Read("tortoise_diffusion_reference.json"); var f = document.RootElement;
        var engine = AiDotNetEngine.Current; var layers = new List<LayerBase<double>>();
        var options = new TortoiseTTSOptions { DiffusionDim = 32, DiffusionHeads = 4, DiffusionLayers = 1,
            MelChannels = 4, LLMDim = 8, CodebookSize = 8, DropoutRate = 0 };
        var model = new TortoiseDiffusionCore<double>(engine, layers, options); model.LoadTorchWeights(Weights(f));
        var mel = engine.TensorTranspose(new Tensor<double>(new[] { 4, 12 }, new Vector<double>(Values(f.GetProperty("mel")))));
        var latents = new Tensor<double>(new[] { 3, 8 }, new Vector<double>(Values(f.GetProperty("latents"))));
        var noisy = new Tensor<double>(new[] { 1, 4, 9 }, new Vector<double>(Values(f.GetProperty("noisy"))));
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers.Cast<AiDotNet.Interfaces.ILayer<double>>().ToList());
        Dictionary<Tensor<double>, Tensor<double>> gradients;
        using (var tape = new GradientTape<double>())
        {
            var prediction = model.Forward(noisy, 137, latents, mel);
            var loss = engine.ReduceMean(engine.TensorMultiply(prediction, prediction), new[] { 0, 1, 2 }, true);
            Assert.True(Math.Abs(loss[0] - f.GetProperty("sgd_loss").GetDouble()) < 2e-5);
            gradients = tape.ComputeGradients(loss, sources: parameters);
        }
        double rate = f.GetProperty("sgd_rate").GetDouble();
        foreach (var parameter in parameters)
        {
            if (!gradients.TryGetValue(parameter, out var gradient)) continue;
            for (int i = 0; i < parameter.Length; i++) parameter[i] -= rate * gradient[i];
            engine.InvalidatePersistentTensor(parameter);
        }
        Close(model.Forward(noisy, 137, latents, mel), Values(f.GetProperty("sgd_prediction_after")), 2e-5);
    }

    [Fact]
    public void Tortoise_DdpmMeanLearnedVarianceAndGuidanceMatchReleasedGaussianDiffusion()
    {
        using var document = Read("tortoise_diffusion_reference.json"); var f = document.RootElement;
        var schedule = new TortoiseDiffusionSchedule<double>(AiDotNetEngine.Current, 4000, 4);
        var noisy = new Tensor<double>(new[] { 1, 4, 9 }, new Vector<double>(Values(f.GetProperty("noisy"))));
        var prediction = new Tensor<double>(new[] { 1, 8, 9 }, new Vector<double>(Values(f.GetProperty("prediction"))));
        var free = new Tensor<double>(new[] { 1, 8, 9 }, new Vector<double>(Values(f.GetProperty("unconditioned"))));
        var actual = schedule.Statistics(noisy, prediction, 137, 136, f.GetProperty("posterior_variance").GetDouble(), free, 2);
        var expected = f.GetProperty("statistics");
        Close(actual.Mean, Values(expected.GetProperty("mean")), 2e-5);
        Close(actual.LogVariance, Values(expected.GetProperty("log_variance")), 2e-5);
        Close(actual.Clean, Values(expected.GetProperty("pred_xstart")), 2e-5);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void Tortoise_DiffusionTrainingLossAndVarianceGradientsMatchTheAuthors(int index)
    {
        using var document = Read("tortoise_diffusion_reference.json"); var f = document.RootElement;
        var expected = f.GetProperty("training")[index];
        var schedule = new TortoiseDiffusionSchedule<double>(AiDotNetEngine.Current, 4000, 4);
        var clean = new Tensor<double>(new[] { 1, 4, 9 }, new Vector<double>(Values(f.GetProperty("training_clean"))));
        var noise = new Tensor<double>(new[] { 1, 4, 9 }, new Vector<double>(Values(f.GetProperty("training_noise"))));
        var prediction = new Tensor<double>(new[] { 1, 8, 9 }, new Vector<double>(Values(f.GetProperty("prediction"))));
        using var tape = new GradientTape<double>();
        var result = schedule.TrainingTerms(clean, noise, prediction, expected.GetProperty("step").GetInt32());
        Assert.True(Math.Abs(result.Mse[0] - expected.GetProperty("mse").GetDouble()) < 2e-5);
        Assert.True(Math.Abs(result.Variational[0] - expected.GetProperty("vb").GetDouble()) < 2e-5,
            $"Variance loss: actual {result.Variational[0]:R}, reference {expected.GetProperty("vb").GetDouble():R}.");
        Assert.True(Math.Abs(result.Total[0] - expected.GetProperty("loss").GetDouble()) < 2e-5);
        var gradients = tape.ComputeGradients(result.Total, sources: new List<Tensor<double>> { prediction });
        Assert.True(gradients.ContainsKey(prediction));
        Close(gradients[prediction], Values(expected.GetProperty("gradient")), 2e-5);
    }

    [Fact]
    public void Tortoise_DiffusionMatchesReleasedConditioningAndBothEpsilonVarianceOutputs()
    {
        using var document = Read("tortoise_diffusion_reference.json"); var f = document.RootElement;
        var engine = AiDotNetEngine.Current;
        var options = new TortoiseTTSOptions { DiffusionDim = 32, DiffusionHeads = 4, DiffusionLayers = 1,
            MelChannels = 4, LLMDim = 8, CodebookSize = 8, DropoutRate = 0 };
        var model = new TortoiseDiffusionCore<double>(engine, new List<LayerBase<double>>(), options);
        model.LoadTorchWeights(Weights(f));
        var mel = engine.TensorTranspose(new Tensor<double>(new[] { 4, 12 }, new Vector<double>(Values(f.GetProperty("mel")))));
        var latents = new Tensor<double>(new[] { 3, 8 }, new Vector<double>(Values(f.GetProperty("latents"))));
        var noisy = new Tensor<double>(new[] { 1, 4, 9 }, new Vector<double>(Values(f.GetProperty("noisy"))));
        var condition = model.Conditioning(mel);
        Close(condition, Values(f.GetProperty("condition")), 2e-5);
        Close(model.Aligned(latents, condition, 9), Values(f.GetProperty("aligned")), 2e-5);
        double time = f.GetProperty("timestep").GetDouble();
        Close(model.Forward(noisy, time, latents, mel), Values(f.GetProperty("prediction")), 2e-5);
        Close(model.Forward(noisy, time, latents, null, conditioningFree: true), Values(f.GetProperty("unconditioned")), 2e-5);
        var codes = f.GetProperty("codes").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        Close(model.Forward(noisy, time, latents, mel, codes: codes), Values(f.GetProperty("code_prediction")), 2e-5);
    }

    [Fact]
    public void Tortoise_ReferenceConditioningAndGptOutputsMatchReleasedUnifiedVoice()
    {
        using var document = Read("tortoise_autoregressive_reference.json"); var f = document.RootElement;
        var engine = AiDotNetEngine.Current;
        var options = new TortoiseTTSOptions { LLMDim = 32, NumHeads = 4, NumLLMLayers = 1,
            NumEncoderLayers = 6, AutoregressiveFeedForwardDim = 128, VocabSize = 16, CodebookSize = 8,
            TextStartToken = 15, TextStopToken = 0, SpeechStartToken = 8, SpeechStopToken = 9,
            MaxTextLength = 8, MaxCodecFrames = 6, MaxConditioningInputs = 2, DropoutRate = 0 };
        var model = new TortoiseAutoregressive<double>(engine, new List<LayerBase<double>>(), options);
        model.LoadTorchWeights(Weights(f));
        var mel = new Tensor<double>(new[] { 80, 7 }, new Vector<double>(Values(f.GetProperty("mel"))));
        var condition = model.Conditioning(engine.TensorTranspose(mel), false, new Random(1));
        Close(condition, Values(f.GetProperty("condition")), 2e-6);
        var text = f.GetProperty("text").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        var speech = f.GetProperty("speech").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        var output = model.TeacherForced(text, speech, condition, false, new Random(1));
        Close(output.TextLogits, Values(f.GetProperty("text_logits")), 2e-6);
        Close(output.SpeechLogits, Values(f.GetProperty("speech_logits")), 2e-6);
        Close(output.SpeechLatents, Values(f.GetProperty("speech_latents")), 2e-6);
        double expectedLoss = f.GetProperty("speech_loss").GetDouble() + options.TextLossWeight * f.GetProperty("text_loss").GetDouble();
        Assert.Equal(expectedLoss, model.Loss(text, speech, condition, false, new Random(1))[0], 6);
    }

    [Fact]
    public void Clam_ProbabilisticRvqLossAndEveryGradientMatchIndependentPaperEquations()
    {
        using var document = Read("clam_rvq_reference.json"); var f = document.RootElement;
        var engine = AiDotNetEngine.Current;
        var layers = new List<LayerBase<double>>();
        var quantizer = new DiTToProbabilisticRvq<double>(engine, layers, new Random(1), 2, 4, 3);
        var fixtureParameters = f.GetProperty("parameters");
        for (int d = 0; d < 2; d++) quantizer.Codebooks[d].LoadTable(Values(fixtureParameters[d]));
        quantizer.MaximumScaleLogit.SetParameters(new Vector<double>(Values(fixtureParameters[2])));
        quantizer.ScaleLogits.SetParameters(new Vector<double>(Values(fixtureParameters[3])));
        quantizer.LogVariance.SetParameters(new Vector<double>(Values(fixtureParameters[4])));
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers.Cast<AiDotNet.Interfaces.ILayer<double>>().ToList());
        Assert.Equal(5, parameters.Count);
        var latents = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(Values(f.GetProperty("latents"))));
        var sources = parameters.Concat(new[] { latents }).ToList();
        using var tape = new GradientTape<double>();
        var result = quantizer.TrainingForward(latents);
        Close(result.Quantized, Values(f.GetProperty("quantized")), 1e-12);
        Assert.Equal(f.GetProperty("variational").GetDouble(), result.VariationalLoss[0], 10);
        Assert.Equal(f.GetProperty("commitment").GetDouble(), result.CommitmentLoss[0], 10);
        var loss = engine.TensorAdd(engine.TensorAdd(result.VariationalLoss,
            engine.TensorMultiplyScalar(result.CommitmentLoss, 0.25)),
            engine.TensorMultiplyScalar(engine.ReduceSum(result.Quantized, new[] { 0, 1 }, true), 0.1));
        Assert.Equal(f.GetProperty("loss").GetDouble(), loss[0], 10);
        var gradients = tape.ComputeGradients(loss, sources: sources);
        for (int i = 0; i < parameters.Count; i++)
        {
            Assert.True(gradients.ContainsKey(parameters[i]), $"Missing gradient for parameter {i}.");
            Close(gradients[parameters[i]], Values(f.GetProperty("gradients")[i]), 1e-9);
        }
        Assert.True(gradients.ContainsKey(latents), "Missing straight-through encoder gradient.");
        Close(gradients[latents], Values(f.GetProperty("latent_gradient")), 1e-9);
    }

    private static JsonDocument Read(string name) => JsonDocument.Parse(File.ReadAllText(Path.Combine(
        AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", name)));
    private static double[] Values(JsonElement value) => value.EnumerateArray().Select(v => v.GetDouble()).ToArray();
    private static Func<string, int[], double[]> Weights(JsonElement root) => (name, shape) =>
    {
        var result = Values(root.GetProperty("weights").GetProperty(name));
        Assert.Equal(shape, root.GetProperty("weight_shapes").GetProperty(name).EnumerateArray().Select(x => x.GetInt32()).ToArray());
        Assert.Equal(shape.Aggregate(1, (a, b) => a * b), result.Length);
        return result;
    };
    private static void Close(Tensor<double> actual, double[] expected, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = actual.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(error < tolerance, $"Maximum absolute error {error:R} exceeds {tolerance:R}.");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void UniAudio_PrefixTeacherForcingMatchesTheReleasedMultiScaleLm(bool fileImport)
    {
        using var document = Read("uniaudio_lm_reference.json"); var f = document.RootElement;
        var options = new UniAudioOptions { LLMDim = 12, NumHeads = 3, NumEncoderLayers = 2, NumLLMLayers = 1,
            GlobalFeedForwardDim = 48, LocalFeedForwardDim = 48, NumCodebooks = 3, JointVocabularySize = 64,
            MaxContextPatches = 32, GlobalStartToken = 2, LocalStartToken = 1, PrefixLanguageModel = true };
        var network = new UniAudioMultiScaleTransformer<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), options);
        network.LoadTorchWeights(fileImport ? FileWeights("uniaudio_lm_reference.safetensors") : Weights(f));
        var sequence = f.GetProperty("sequence").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        Close(network.Forward(sequence, f.GetProperty("prefix_tokens").GetInt32()), Values(f.GetProperty("logits")), 2e-5);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void VoiceCraft_MaskedRowsAndEveryPredictionHeadMatchTheAuthorsDecoder(bool fileImport)
    {
        using var document = Read("voicecraft_lm_reference.json"); var f = document.RootElement;
        var options = new VoiceCraftOptions { LLMDim = 16, NumHeads = 4, NumLLMLayers = 1, FeedForwardDim = 64,
            TextVocabularySize = 80, CodebookSize = 8, MaxMaskSpans = 3, TransformerDropout = 0,
            TextEmbeddingDropout = 0, AudioEmbeddingDropout = 0, TextPositionDropout = 0, AudioPositionDropout = 0 };
        var network = new VoiceCraftNetwork<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), options);
        network.LoadTorchWeights(fileImport ? FileWeights("voicecraft_lm_reference.safetensors") : Weights(f));
        var text = f.GetProperty("text").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        var rows = f.GetProperty("rows").EnumerateArray().Select(r => r.EnumerateArray().Select(x => x.GetInt32()).ToArray()).ToArray();
        var masks = f.GetProperty("masks").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        var outputs = network.Forward(text, rows, masks);
        for (int q = 0; q < outputs.Length; q++) Close(outputs[q], Values(f.GetProperty("logits")[q]), 2e-6);
    }

    private static Func<string, int[], double[]> FileWeights(string name, bool unwrapXttsRoot = false)
    {
        var loaded = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(Path.Combine(
            AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", name));
        var state = unwrapXttsRoot ? XTTSv2<double>.UnwrapAuthorState(loaded) : loaded;
        return (key, shape) =>
        {
            Assert.True(state.ContainsKey(key), $"Missing author tensor {key}.");
            Assert.Equal(shape, state[key].Shape.ToArray());
            return state[key].ToArray().Select(value => (double)value).ToArray();
        };
    }

    private static XTTSv2Options XttsOptions() => new()
    {
        LLMDim = 32, TextVocabularySize = 32, AudioVocabularySize = 18, CodebookSize = 16,
        NumLLMLayers = 1, NumHeads = 4, MaxTextLength = 64, MaxCodecFrames = 2,
        ConditioningBlocks = 1, ConditioningHeads = 4, PerceiverDepth = 1, PerceiverLatents = 2,
        PerceiverHeads = 1, DropoutRate = 0, DecoderInitialChannels = 16,
        DecoderUpsampleRates = [2, 2], DecoderUpsampleKernels = [4, 4], DecoderResblockKernels = [3],
        DecoderResblockDilations = [[1]]
    };

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Xtts_GptLatentsMatchCoquiIncludingBothFinalNormalizations(bool fileImport)
    {
        using var document = Read("xtts_gpt_reference.json"); var f = document.RootElement;
        var model = new XttsLanguageModel<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), XttsOptions(), new Random(1));
        model.LoadTorchWeights(fileImport ? FileWeights("xtts_wrapped_gpt_reference.pth", true) : Weights(f), includeConditioning: false);
        var condition = new Tensor<double>(new[] { 2, 32 }, new Vector<double>(Values(f.GetProperty("condition"))));
        var text = f.GetProperty("text").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        var audio = f.GetProperty("audio").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        var actual = model.Forward(text, audio, condition);
        Close(actual.Text, Values(f.GetProperty("text_hidden")), 2e-6);
        Close(actual.Audio, Values(f.GetProperty("audio_hidden")), 2e-6);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Xtts_ReferenceMelConditioningMatchesCoquiAttentionAndPerceiver(bool fileImport)
    {
        using var document = Read("xtts_gpt_reference.json"); var f = document.RootElement;
        var model = new XttsLanguageModel<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), XttsOptions(), new Random(1));
        model.LoadTorchWeights(fileImport ? FileWeights("xtts_wrapped_gpt_reference.pth", true) : Weights(f));
        var mel = new Tensor<double>(new[] { 80, 5 }, new Vector<double>(Values(f.GetProperty("mel"))));
        Close(model.Condition(mel), Values(f.GetProperty("conditioning")), 2e-5);
    }

    [Fact]
    public void Xtts_LatentInterpolationMatchesBothPyTorchScaleFactorPasses()
    {
        using var document = Read("xtts_gpt_reference.json"); var f = document.RootElement;
        var decoder = new XttsWaveformDecoder<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), XttsOptions());
        var input = new Tensor<double>(new[] { 3, 2 }, new Vector<double>(Values(f.GetProperty("interpolation_input"))));
        var output = decoder.LinearScale(decoder.LinearScale(input, 1024.0 / 256), 24000.0 / 22050);
        Close(output, Values(f.GetProperty("interpolation_output")), 1e-12);
    }

    [Fact]
    public void Xtts_OriginalWrappedDecoderAndCoquiConditionNamesMatchFullWaveform()
    {
        using var document = Read("xtts_decoder_reference.json"); var f = document.RootElement;
        var options = XttsOptions(); options.LLMDim = 4; options.SpeakerEmbeddingDim = 4;
        options.DecoderResblockDilations = new[] { new[] { 1, 3, 5 } };
        var decoder = new XttsWaveformDecoder<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), options);
        var state = XTTSv2<double>.UnwrapAuthorState(new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(
            Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", "xtts_wrapped_decoder_reference.pth")));
        decoder.LoadTorchWeights(name => state.TryGetValue("hifigan_decoder.waveform_decoder." + name, out var tensor)
            ? tensor.ToArray().Select(value => (double)value).ToArray() : null);
        var latents = new Tensor<double>(new[] { 3, 4 }, new Vector<double>(Values(f.GetProperty("latents"))));
        var speaker = new Tensor<double>(new[] { 4 }, new Vector<double>(Values(f.GetProperty("speaker"))));
        Close(decoder.Forward(latents, speaker), Values(f.GetProperty("waveform")), 2e-9);
    }
}
