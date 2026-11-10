using AiDotNet.ModelLoading;
using AiDotNet.Attributes;
using AiDotNet.Audio.Resampling;
using AiDotNet.Audio.Watermarking;
using AiDotNet.TextToSpeech.FrontEnd;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using System.Runtime.CompilerServices;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>
/// Resemble AI's Chatterbox: an emotion-controllable zero-shot T3 model whose speech tokens are rendered by S3Gen.
/// </summary>
/// <remarks>
/// <para>This implementation follows the released English model: BPE text, voice-encoder and prompt-token conditioning,
/// the 520M Llama-based T3 decoder, S3Tokenizer-v2, causal conditional flow matching, CAM++ and HiFT.</para>
/// <para>Like the release, every synthesized waveform carries Resemble's Perth implicit watermark (a frozen PerthNet run
/// natively; see <see cref="ChatterboxOptions.Watermark"/> and <see cref="ChatterboxOptions.WatermarkModelPath"/>).</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Chatterbox: Open-Source Voice Cloning", "https://github.com/resemble-ai/Chatterbox")]
public partial class Chatterbox<T> : CosyVoiceBase<T>
{
    private readonly ChatterboxOptions _options;
    private readonly List<LayerBase<T>> _voiceEncoderLayers = new();
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _t3Optimizer;
    private ChatterboxT3<T>? _t3;
    private ChatterboxVoiceEncoder<T>? _voiceEncoder;
    private ChatterboxTokenizer? _tokenizer;
    private PerthWatermarker<T>? _watermarker;
    private ChatterboxMultilingualText? _multilingualText;
    private CosyVoiceFlowConfiguration? _flowConfiguration;
    private readonly ConditionalWeakTable<TtsVoice<T>, T3VoiceCondition> _t3VoiceConditions = new();
    private int _t3WeightsGeneration;

    public Chatterbox(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        ChatterboxOptions? options = null)
        : base(architecture, (options ?? new ChatterboxOptions()).SamplingSeed, modelPath,
            (options ?? new ChatterboxOptions()).OnnxOptions)
    {
        _options = options ?? new ChatterboxOptions();
        Options = _options;
        Validate();
        LoadConfiguredTokenizer();
        Initialize();
        InitializeReleasedFrontends();
        InitializeVoiceEncoder();
    }

    public Chatterbox(
        NeuralNetworkArchitecture<T> architecture,
        ChatterboxOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, (options ?? new ChatterboxOptions()).SamplingSeed, optimizer)
    {
        _options = options ?? new ChatterboxOptions();
        Options = _options;
        Validate();
        LoadConfiguredTokenizer();
        Initialize();
        InitializeReleasedFrontends();
        InitializeVoiceEncoder();
        _t3Optimizer = optimizer ?? CreateComponentOptimizer("language model", _options.WarmupSteps);
    }

    /// <summary>The Perth watermark model, read from <see cref="ChatterboxOptions.WatermarkModelPath"/> on first use.</summary>
    internal PerthWatermarker<T> Watermarker
    {
        get
        {
            if (_watermarker is not null) return _watermarker;
            var path = _options.WatermarkModelPath;
            if (string.IsNullOrWhiteSpace(path))
                throw new InvalidOperationException("Chatterbox watermarks its output as the release does: set ChatterboxOptions.WatermarkModelPath to "
                    + "perth_net_250000.pth.tar (from the resemble-perth package), or set ChatterboxOptions.Watermark to false.");
            return _watermarker = PerthWatermarker<T>.Load(Engine, path!);
        }
        set => _watermarker = value;
    }

    // The release's generate(): Perth's apply_watermark on S3Gen's (trim-faded) output.
    internal Tensor<T> Watermarked(Tensor<T> waveform)
    {
        if (!_options.Watermark || waveform.Length == 0) return waveform;
        var marked = Watermarker.Apply(waveform.ToArray().Select(v => NumOps.ToDouble(v)).ToArray(), _options.SampleRate);
        return new Tensor<T>(new[] { marked.Length }, new Vector<T>(marked.Select(NumOps.FromDouble).ToArray()));
    }

    protected override CosyVoiceFamilyOptions Settings => _options;
    protected override int TextVocabularySize => _options.VocabSize;
    protected override TtsSupervision RequiredVoice => TtsSupervision.ReferenceRecording;

    internal ChatterboxT3<T>? T3Model => _t3;
    internal ChatterboxVoiceEncoder<T>? VoiceEncoderModel => _voiceEncoder;
    internal IReadOnlyList<LayerBase<T>> VoiceEncoderLayers => _voiceEncoderLayers;

    public override int[] Tokenize(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        return IsMultilingual ? MultilingualText.Encode(text, _options.LanguageId.ToLowerInvariant()) : Tokenizer.Encode(text);
    }

    private bool IsMultilingual => _options.Variant != ChatterboxVariant.English;

    public override string Detokenize(IEnumerable<int> ids) => Tokenizer.Decode(ids);

    /// <summary>Applies the release's punctuation cleanup before BPE tokenization.</summary>
    public static string NormalizeText(string text) => NormalizeText(text, new[] { ".", "!", "?", "-", "," });

    // punc_norm with its language's sentence enders (the multilingual release also accepts 、，。？！).
    internal static string NormalizeText(string text, string[] sentenceEnders)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (text.Length == 0) return "You need to add some text for me to talk.";
        if (char.IsLower(text[0])) text = char.ToUpperInvariant(text[0]) + text.Substring(1);
        text = string.Join(" ", text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
        foreach (var replacement in new[]
        {
            ("...", ", "), ("…", ", "), (":", ","), (" - ", ", "), (";", ", "),
            ("—", "-"), ("–", "-"), (" ,", ","), ("“", "\""), ("”", "\""), ("‘", "'"), ("’", "'"),
        })
            text = text.Replace(replacement.Item1, replacement.Item2);
        text = text.TrimEnd(' ');
        if (!sentenceEnders.Any(ender => text.EndsWith(ender, StringComparison.Ordinal)))
            text += ".";
        return text;
    }

    /// <summary>
    /// Creates both sets of released conditionals: T3 reads the full voice-encoder recording and its first six seconds
    /// of S3 tokens; S3Gen reads at most ten seconds for its prompt mel, tokens and CAM++ x-vector.
    /// </summary>
    public override TtsVoice<T> CreateVoice(Tensor<T> recording, int sampleRate, string? transcript = null)
    {
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        var full = Resample(recording, sampleRate, _options.SampleRate);
        int decoderSamples = Math.Min(full.Length, _options.DecoderReferenceSeconds * _options.SampleRate);
        var decoderReference = Engine.TensorSlice(full, new[] { 0 }, new[] { decoderSamples });
        var s3Voice = base.CreateVoice(decoderReference, _options.SampleRate, transcript: null);
        var voice = new TtsVoice<T>
        {
            Reference = s3Voice.Reference,
            ReferenceCodecTokens = s3Voice.ReferenceCodecTokens,
            SpeakerEmbedding = s3Voice.SpeakerEmbedding,
            ReferenceAudio = full,
        };
        if (HasPaperLayers) _t3VoiceConditions.Add(voice, PrepareT3VoiceCondition(full));
        return voice;
    }

    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        string normalized = IsMultilingual ? ChatterboxMultilingualText.NormalizePunctuation(text) : NormalizeText(text);
        var input = Ids(Tokenize(normalized));
        // The native path watermarks in PredictCore.
        return IsOnnxMode && OnnxModel is not null ? Watermarked(OnnxModel.Run(input)) : Predict(input);
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        var waveform = base.PredictCore(input);
        if (!HasPaperLayers || waveform.Length == 0) return waveform;

        int trim = _options.SampleRate / 50;
        var envelope = new Tensor<T>(new[] { waveform.Length });
        for (int i = 0; i < envelope.Length; i++)
        {
            double value = i < trim
                ? 0.0
                : i < 2 * trim
                    ? (Math.Cos(Math.PI - Math.PI * (i - trim) / Math.Max(1, trim - 1)) + 1.0) / 2.0
                    : 1.0;
            envelope[i] = NumOps.FromDouble(value);
        }
        return Watermarked(Engine.TensorMultiply(waveform, envelope));
    }

    private protected override S3TokenizerConfiguration SpeechTokenizerConfiguration =>
        S3TokenizerConfiguration.V2Default with
        {
            State = _options.SpeechTokenizerState,
            Heads = _options.SpeechTokenizerHeads,
            Layers = _options.SpeechTokenizerLayers,
            CodebookSize = _options.CodebookSize,
        };

    private protected override CosyVoiceFlowConfiguration FlowConfiguration =>
        _flowConfiguration ??= new CosyVoiceFlowConfiguration(
            VocabSize: _options.CodebookSize,
            InputSize: _options.FlowDim,
            OutputSize: _options.MelChannels,
            SpeakerDim: _options.SpeakerEmbeddingDim,
            InputFrameRate: _options.CodecFrameRate,
            SampleRate: _options.SampleRate,
            HopSize: _options.HopSize,
            Encoder: new WenetEncoderConfiguration(
                _options.FlowDim,
                _options.FlowDim,
                _options.FlowHeads,
                _options.FlowFeedForwardDim,
                _options.FlowEncoderLayers,
                WenetInputLayer.Linear,
                Swish: true,
                Conformer: true,
                StaticChunkSize: _options.StaticChunkSize,
                Dropout: _options.DropoutRate,
                PositionalDropout: _options.DropoutRate,
                AttentionDropout: _options.FlowAttentionDropoutRate),
            RegulatorStages: 0,
            DecoderChannels: (int[])_options.DecoderChannels.Clone(),
            DecoderHeads: _options.DecoderHeads,
            DecoderHeadDim: _options.DecoderHeadDim,
            DecoderBlocksPerLevel: _options.DecoderBlocks,
            DecoderMidBlocks: _options.DecoderMidBlocks,
            SigmaMin: 1e-6,
            TrainingCfgRate: _options.TrainingCfgRate,
            InferenceCfgRate: _options.InferenceCfgRate,
            Timesteps: _options.FlowTimesteps,
            Loss: _options.FlowLoss,
            Causal: true,
            TokenMelRatio: _options.TokenMelRatio,
            Lookahead: _options.Lookahead);

    protected override void BuildLanguageModel(List<LayerBase<T>> layers, Random initialization) =>
        _t3 = new ChatterboxT3<T>(Engine, layers, _options, initialization);

    protected override int[] GenerateWithLanguageModel(
        IReadOnlyList<int> text,
        IReadOnlyList<int> promptText,
        IReadOnlyList<int> promptSpeech,
        Tensor<T>? xvector,
        Random random)
    {
        var voice = EnsurePreparedVoice();
        var reference = voice.ReferenceAudio
            ?? throw new InvalidOperationException("Chatterbox T3 needs the reference waveform.");
        if (!_t3VoiceConditions.TryGetValue(voice, out var condition)
            || condition.WeightsGeneration != _t3WeightsGeneration)
        {
            _t3VoiceConditions.Remove(voice);
            condition = PrepareT3VoiceCondition(reference);
            _t3VoiceConditions.Add(voice, condition);
        }
        return T3.Generate(text,
            new ChatterboxT3Condition<T>(condition.SpeakerEmbedding, condition.PromptSpeechTokens,
                _options.Exaggeration), random);
    }

    protected override Tensor<T> LanguageModelLoss(
        IReadOnlyList<int> text,
        IReadOnlyList<int> speech,
        Tensor<T> xvector,
        bool training,
        Random random)
    {
        throw new InvalidOperationException(
            "Chatterbox's T3 loss needs the voice-encoder recording as well as CAM++; use Train(TtsTrainingSample).");
    }

    private protected override void LoadLanguageModel(TorchTensorSource source) => T3.LoadTorchWeights(source);

    /// <inheritdoc />
    /// <remarks>The release provides inference checkpoints, not a training recipe. Native training uses configured
    /// AdamW on the selected component only; this is a configurable training policy, not an attributed release optimizer.</remarks>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateComponentOptimizer(string component, int warmupSteps)
    {
        var layers = component switch
        {
            "language model" => LanguageModelLayers,
            "flow" => FlowLayers,
            "vocoder" => VocoderLayers,
            "discriminator" => DiscriminatorLayers,
            _ => throw new ArgumentOutOfRangeException(nameof(component)),
        };
        var options = new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.LearningRate,
            WeightDecay = _options.WeightDecay,
            MaskParameters = Training.TapeTrainingStep<T>.CollectParameters(layers.Cast<ILayer<T>>().ToList(), -1).ToList(),
            SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
        };
        if (warmupSteps > 0)
            options.LearningRateScheduler = new AiDotNet.LearningRateSchedulers.LinearWarmupScheduler(
                _options.LearningRate, warmupSteps, warmupSteps, 0.0,
                AiDotNet.LearningRateSchedulers.LinearWarmupScheduler.DecayMode.Constant);
        return new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, options);
    }


    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        if (CurrentStage != CosyVoiceTrainingStage.LanguageModel)
            return base.TrainOnSample(sample);
        if (_t3Optimizer is null) throw new NotSupportedException("Training is not supported in ONNX mode.");
        return TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => BuildT3Objective(sample), _t3Optimizer);
    }

    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        if (CurrentStage != CosyVoiceTrainingStage.LanguageModel)
            return base.EvaluateTrainingObjective(sample);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return BuildT3Objective(sample)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    private Tensor<T> BuildT3Objective(TtsTrainingSample<T> sample)
    {
        if (!HasPaperLayers)
            throw new NotSupportedException("The T3 objective needs the release's native layers.");
        var text = IdsOf(sample.Tokens);
        int[] speech;
        if (sample.CodecTokens is { } supplied)
        {
            speech = IdsOf(supplied);
        }
        else
        {
            var audio = sample.Audio
                ?? throw new ArgumentException("T3 training needs CodecTokens or Audio.", nameof(sample));
            using var _ = new NoGradScope<T>();
            speech = IdsOf(EncodeToTokens(audio));
        }

        var reference = sample.SpeakerReference ?? sample.Audio
            ?? throw new ArgumentException("T3 training needs a speaker recording in SpeakerReference or Audio.", nameof(sample));
        var condition = PrepareT3VoiceCondition(reference);
        return T3.Loss(text, speech,
            new ChatterboxT3Condition<T>(condition.SpeakerEmbedding, condition.PromptSpeechTokens,
                _options.Exaggeration));
    }

    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        foreach (var layer in _voiceEncoderLayers) layer.SetTrainingMode(false);
    }

    /// <inheritdoc />
    /// <remarks>S3Gen's <c>hifigan.py</c> has only the original <c>SineGen</c>, which it runs at 24 kHz.</remarks>
    private protected override bool HiftUsesOriginalSineGen => true;

    /// <summary>
    /// Loads the unmodified release directory: for the English model <c>t3_cfg.safetensors</c>, <c>s3gen.safetensors</c>,
    /// <c>ve.safetensors</c> and <c>tokenizer.json</c>; for a multilingual one <c>t3_mtl23ls_v2</c> (or <c>_v3</c>)
    /// <c>.safetensors</c>, <c>s3gen.pt</c>, <c>ve.pt</c>, <c>grapheme_mtl_merged_expanded_v1.json</c> and
    /// <c>Cangjie5_TC.json</c>. A <c>perth_net_250000.pth.tar</c> beside them becomes the watermark model unless
    /// <see cref="ChatterboxOptions.WatermarkModelPath"/> is set.
    /// </summary>
    public void LoadPretrainedWeights(string modelDirectory)
    {
        if (!IsNative) throw new NotSupportedException("Native checkpoints cannot be loaded into an ONNX model.");
        if (string.IsNullOrWhiteSpace(modelDirectory))
            throw new ArgumentException("The model directory is required.", nameof(modelDirectory));

        string tokenizerPath = Path.Combine(modelDirectory, IsMultilingual ? "grapheme_mtl_merged_expanded_v1.json" : "tokenizer.json");
        _tokenizer = ChatterboxTokenizer.Load(tokenizerPath);
        _options.TokenizerPath = tokenizerPath;
        if (IsMultilingual)
        {
            _options.CangjieMappingPath = Path.Combine(modelDirectory, "Cangjie5_TC.json");
            _multilingualText = null;
        }
        string perth = Path.Combine(modelDirectory, "perth_net_250000.pth.tar");
        if (_options.WatermarkModelPath is null && File.Exists(perth))
        {
            _options.WatermarkModelPath = perth;
            _watermarker = null;
        }

        // Each file is streamed one tensor at a time and closed after its components load.
        string t3File = _options.Variant switch
        {
            ChatterboxVariant.MultilingualV2 => "t3_mtl23ls_v2.safetensors",
            ChatterboxVariant.MultilingualV3 => "t3_mtl23ls_v3.safetensors",
            _ => "t3_cfg.safetensors",
        };
        // The multilingual release ships its voice encoder and S3Gen as PyTorch state dicts.
        string extension = IsMultilingual ? ".pt" : ".safetensors";
        using (var t3 = TorchCheckpointFile.Open(Path.Combine(modelDirectory, t3File))) T3.LoadTorchWeights(t3);
        using (var ve = TorchCheckpointFile.Open(Path.Combine(modelDirectory, "ve" + extension))) VoiceEncoder.LoadTorchWeights(ve);
        using (var s3 = TorchCheckpointFile.Open(Path.Combine(modelDirectory, "s3gen" + extension)))
        {
            Flow.LoadTorchWeights(s3.WithPrefix("flow."), "");
            Hift.LoadTorchWeights(s3.WithPrefix("mel2wav."));
            SpeechTokenizer.LoadTorchWeights(s3.WithPrefix("tokenizer."));
            SpeakerEncoder.LoadTorchWeights(s3.WithPrefix("speaker_encoder."));
        }
        _t3WeightsGeneration++;
        InvalidateWeightCachesAfterSuccessfulWeightUpdate();
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = IsNative ? "Chatterbox-Native" : "Chatterbox-ONNX",
            Description = "Resemble AI Chatterbox: voice-encoder and prompt-conditioned T3 with S3Gen.",
            FeatureCount = _options.LLMDim,
            Complexity = _options.NumLLMLayers,
        };
        metadata.AdditionalInfo["Architecture"] = "T3 + S3Gen";
        metadata.AdditionalInfo["Mode"] = IsNative ? "Native" : "ONNX";
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate;
        metadata.AdditionalInfo["SpeechVocabulary"] = _options.SpeechVocabularySize;
        metadata.AdditionalInfo["Watermark"] = _options.Watermark ? "Perth" : "None";
        return metadata;
    }

    private void InitializeVoiceEncoder()
    {
        if (!IsNative || !HasPaperLayers) return;
        var scope = LayerInitializationSeedScope.CaptureScope();
        try
        {
            if (_options.Seed is int seed) LayerInitializationSeedScope.ResetForModelConstruction(seed);
            _voiceEncoder = new ChatterboxVoiceEncoder<T>(Engine, _voiceEncoderLayers, _options);
        }
        finally
        {
            if (_options.Seed.HasValue) LayerInitializationSeedScope.RestoreScope(scope);
        }
        ComponentLayers.AddRange(_voiceEncoderLayers);
        foreach (var layer in _voiceEncoderLayers) layer.SetTrainingMode(false);
    }


    private TtsVoice<T> EnsurePreparedVoice()
    {
        var voice = RequireVoice();
        if (voice.Reference is not null && voice.ReferenceCodecTokens is not null && voice.SpeakerEmbedding is not null)
            return voice;
        Voice = CreateVoice(voice.ReferenceAudio
            ?? throw new InvalidOperationException("Chatterbox needs a reference recording."), _options.SampleRate);
        return Voice;
    }

    private T3VoiceCondition PrepareT3VoiceCondition(Tensor<T> reference)
    {
        // prepare_conditionals first resamples the complete 24 kHz reference with librosa's default soxr_hq.
        // This deliberately differs from S3Gen's torchaudio resampler in base.CreateVoice.
        var reference16k = LibrosaResample(reference, _options.SampleRate, 16000);
        int samples = Math.Min(reference16k.Length, _options.EncoderReferenceSeconds * 16000);
        var prefix = Engine.TensorSlice(reference16k, new[] { 0 }, new[] { samples });
        int[] prompt = SpeechTokens(prefix, 16000).Take(_options.SpeechConditionPromptTokens).ToArray();
        Tensor<T> speaker;
        using (new NoGradScope<T>()) speaker = VoiceEncoder.Embed(reference16k, 16000);
        return new T3VoiceCondition(speaker, prompt, _t3WeightsGeneration);
    }

    private Tensor<T> Resample(Tensor<T> audio, int from, int to)
    {
        var flat = Engine.Reshape(audio, new[] { audio.Length });
        return from == to ? flat : LibrosaResample(flat, from, to);
    }

    private Tensor<T> LibrosaResample(Tensor<T> audio, int from, int to)
    {
        var values = new double[audio.Length];
        for (int i = 0; i < values.Length; i++) values[i] = NumOps.ToDouble(audio[i]);
        var resampled = SoxrResampler.LibrosaResample(values, from, to);
        var tensor = new Tensor<T>(new[] { resampled.Length });
        for (int i = 0; i < resampled.Length; i++) tensor[i] = NumOps.FromDouble(resampled[i]);
        return tensor;
    }

    private void Validate()
    {
        if (_options.NumCodebooks != 1)
            throw new ArgumentException("Chatterbox uses one S3 speech-token stream.");
        if (_options.VocabSize <= Math.Max(_options.TextStartToken, _options.TextStopToken))
            throw new ArgumentException("VocabSize must contain the text start and stop tokens.");
        if (_options.CodebookSize <= 0
            || _options.SpeechStartToken < _options.CodebookSize
            || _options.SpeechStopToken < _options.CodebookSize
            || _options.SpeechVocabularySize <= Math.Max(_options.SpeechStartToken, _options.SpeechStopToken))
            throw new ArgumentException("The speech vocabulary must contain the S3 codes and the T3 start and stop tokens.");
        if (_options.LLMDim <= 0 || _options.NumHeads <= 0 || _options.KeyValueHeads <= 0
            || _options.LLMDim % _options.NumHeads != 0
            || _options.NumHeads % _options.KeyValueHeads != 0)
            throw new ArgumentException("LLMDim must divide by the attention heads, and heads by key-value heads.");
        if (_options.PerceiverQueries <= 0 || _options.PerceiverHeads <= 0
            || _options.LLMDim % _options.PerceiverHeads != 0)
            throw new ArgumentException("The Perceiver needs positive query/head counts and heads that divide LLMDim.");
        if (_options.MaxTextLength <= 0 || _options.MaxSpeechTokens <= 0
            || _options.MaxGeneratedSpeechTokens <= 0 || _options.SpeechConditionPromptTokens <= 0)
            throw new ArgumentException("Text, speech, generation and prompt limits must be positive.");
        if (_options.VoiceEncoderLayers <= 0 || _options.VoiceEncoderHiddenDim <= 0
            || _options.T3SpeakerEmbeddingDim <= 0 || _options.VoiceEncoderPartialFrames <= 0
            || _options.VoiceEncoderPartialRate <= 0)
            throw new ArgumentException("Voice-encoder dimensions and partial settings must be positive.");
        if (_options.TokenMelRatio <= 0 || _options.Lookahead < 0 || _options.StaticChunkSize < 0)
            throw new ArgumentException("TokenMelRatio must be positive; Lookahead and StaticChunkSize cannot be negative.");
        if (_options.Exaggeration < 0 || _options.Exaggeration > 1
            || _options.MinP < 0 || _options.MinP > 1
            || _options.GenerationTopP <= 0 || _options.GenerationTopP > 1)
            throw new ArgumentException("Exaggeration, MinP and GenerationTopP must be probabilities.");
    }

    private void LoadConfiguredTokenizer()
    {
        string? tokenizerPath = _options.TokenizerPath;
        if (!string.IsNullOrWhiteSpace(tokenizerPath))
            _tokenizer = ChatterboxTokenizer.Load(tokenizerPath!);
    }

    private ChatterboxTokenizer Tokenizer => _tokenizer ?? throw new InvalidOperationException(
        "Chatterbox needs its released tokenizer.json. Set ChatterboxOptions.TokenizerPath or call LoadPretrainedWeights.");

    // The multilingual front end, built on first use from the tokenizer and the Cangjie mapping; pkuseg, Dicta and the
    // Russian stresser load only for text in their languages.
    private ChatterboxMultilingualText MultilingualText
    {
        get
        {
            if (_multilingualText is not null) return _multilingualText;
            var cangjie = _options.CangjieMappingPath;
            if (string.IsNullOrWhiteSpace(cangjie))
                throw new InvalidOperationException("A multilingual Chatterbox needs the release's Cangjie5_TC.json: set "
                    + "ChatterboxOptions.CangjieMappingPath or call LoadPretrainedWeights.");
            var segmenter = new Lazy<Pkuseg>(() => Pkuseg.Load(_options.ChineseSegmenterPath ?? Pkuseg.DefaultModelDirectory()));
            var hebrew = new Lazy<DictaDiacritizer>(() => DictaDiacritizer.Load(_options.HebrewDiacritizerPath ?? DictaDiacritizer.DefaultModelDirectory()));
            var russian = new Lazy<RussianTextStresser>(() => RussianTextStresser.Load(_options.RussianStresserPath ?? RussianTextStresser.DefaultModelDirectory()));
            return _multilingualText = new ChatterboxMultilingualText(Tokenizer, cangjie!, () => segmenter.Value,
                text => hebrew.Value.AddDiacritics(text), text => russian.Value.StressText(text));
        }
    }

    private ChatterboxT3<T> T3 => _t3
        ?? throw new InvalidOperationException("Chatterbox T3 exists in native release mode only.");

    private ChatterboxVoiceEncoder<T> VoiceEncoder => _voiceEncoder
        ?? throw new InvalidOperationException("Chatterbox's voice encoder exists in native release mode only.");

    private sealed record T3VoiceCondition(
        Tensor<T> SpeakerEmbedding,
        int[] PromptSpeechTokens,
        int WeightsGeneration);
}
