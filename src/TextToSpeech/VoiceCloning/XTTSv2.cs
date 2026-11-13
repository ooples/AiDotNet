using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.EndToEnd;
using AiDotNet.TextToSpeech.FrontEnd;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>XTTS-v2: multilingual zero-shot TTS with a dVAE tokenizer, GPT-2 and speaker-conditioned HiFi-GAN.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// This is the architecture released by Coqui for Casanova et al. (2024): 80-band 22.05 kHz log-power mels feed a
/// two-stride, 1024-code dVAE tokenizer; six Tortoise attention blocks and a 32-latent Perceiver encode a reference;
/// a 30-layer GPT-2 predicts acoustic ids; its teacher-forced hidden states—not reconstructed dVAE codes—condition a
/// 24 kHz HiFi-GAN. Speaker identity is the frozen 512-wide H/ASP SE-ResNet embedding added before and after every
/// HiFi-GAN upsampler. The public training path implements the released GPT fine-tuning objective (text CE × .01 plus
/// acoustic CE) while the dVAE, speaker encoder and waveform decoder act as pretrained feature modules.
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Model",
    "https://arxiv.org/abs/2406.04904",
    Year = 2024,
    Authors = "Casanova et al.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-5, Beta1 = 0.9, Beta2 = 0.96,
    WeightDecay = 0.01, ReferenceBatchSize = 4, Provenance = RecipeProvenance.Stated,
    Source = "Casanova et al. 2024 Sec. 4.2: AdamW 5e-5, betas (.9,.96), decay .01, batch 4, "
        + "16-step accumulation on four GPUs; the public GPT fine-tuning recipe uses 5e-6.")]
public partial class XTTSv2<T> : TtsModelBase<T>, ICodecTts<T>
{
    private readonly XTTSv2Options _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly Random _samplingRandom;
    private ByteLevelBpeTokenizer? _tokenizer;
    private double[] _melStats;
    private readonly List<LayerBase<T>> _componentLayers = new();
    private bool _useNativeMode;
    private bool _disposed;

    private XttsDiscreteVae<T>? _dvae;
    private XttsLanguageModel<T>? _languageModel;
    private HaspSpeakerEncoder<T>? _speakerEncoder;
    private XttsWaveformDecoder<T>? _decoder;

    /// <summary>Creates an ONNX inference wrapper.</summary>
    public XTTSv2(NeuralNetworkArchitecture<T> architecture, string modelPath, XTTSv2Options? options = null)
        : base(architecture)
    {
        _options = options ?? new XTTSv2Options();
        ValidateOptions(_options);
        Options = _options;
        _samplingRandom = new Random(_options.SamplingSeed);
        _melStats = Enumerable.Repeat(1.0, _options.MelChannels).ToArray();
        ConfigureBaseProperties();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        LoadConfiguredAssets();
    }

    /// <summary>Creates the trainable native XTTS-v2 graph.</summary>
    public XTTSv2(
        NeuralNetworkArchitecture<T> architecture,
        XTTSv2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new XTTSv2Options();
        ValidateOptions(_options);
        Options = _options;
        _useNativeMode = true;
        _samplingRandom = new Random(_options.SamplingSeed);
        _melStats = Enumerable.Repeat(1.0, _options.MelChannels).ToArray();
        _optimizer = optimizer ?? CreateDefaultOptimizer();
        ConfigureBaseProperties();
        InitializeLayers();
        LoadConfiguredAssets();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;
    /// <inheritdoc />
    public int NumCodebooks => _options.NumCodebooks;
    /// <inheritdoc />
    public int CodebookSize => _options.CodebookSize;
    /// <inheritdoc />
    public int CodecFrameRate => _options.CodecFrameRate;
    public override int CodecTokenCodebooks => 1;
    public override int CodecTokenVocabulary => _options.CodebookSize;
    protected override TtsSupervision RequiredVoice => TtsSupervision.ReferenceRecording;
    protected override TtsSupervision RequiredSupervision =>
        TtsSupervision.CodecTokens | TtsSupervision.ReferenceRecording;
    public override bool TrainsOnSampleObjective => true;

    /// <inheritdoc />
    /// <remarks>The native graph accepts already-tokenized rows from the released XTTS BPE vocabulary.</remarks>
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperPipeline ? LayerInputDomain.Indices(_options.TextVocabularySize) : base.GetInputDomain(inputShape);

    /// <summary>Whether the native dVAE, conditioning/GPT, H/ASP and HiFi-GAN components are present.</summary>
    public bool HasPaperPipeline => _dvae is not null && _languageModel is not null
        && _speakerEncoder is not null && _decoder is not null;
    public int TransformerLayerCount => _languageModel?.LayerCount ?? 0;
    public int ConditioningAttentionBlocks => _languageModel?.ConditioningBlocks ?? 0;
    public int ConditioningLatentCount => _languageModel?.ConditioningLatents ?? 0;
    public int DvaeDownsampleFactor => _dvae?.DownsampleFactor ?? 0;
    public bool TokenizerLoaded => _tokenizer is not null;

    /// <summary>Synthesizes in <see cref="TtsModelBase{T}.Voice"/>'s reference voice.</summary>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        var voice = RequireVoice();
        return Synthesize(text, _options.Language, voice.ReferenceAudio!);
    }

    /// <summary>Runs the released zero-shot path for a language and 24 kHz reference recording.</summary>
    public Tensor<T> Synthesize(string text, string language, Tensor<T> referenceAudio)
    {
        ThrowIfDisposed();
        RequirePipeline();
        ValidateReference(referenceAudio);
        int[] textIds = TokenizeText(text, language);
        var condition = ExtractConditioningLatents(referenceAudio);
        var speaker = ExtractSpeakerEmbedding(referenceAudio);
        SetTrainingMode(false);
        using var _ = new NoGradScope<T>();
        var generated = _languageModel!.Generate(textIds, condition, _samplingRandom);
        return _decoder!.Forward(generated.Latents, speaker);
    }

    /// <summary>Runs the native path from already-tokenized XTTS BPE ids (useful for dataset and converted-tokenizer integrations).</summary>
    public Tensor<T> SynthesizeTokens(Tensor<T> textTokens, Tensor<T> referenceAudio)
    {
        ThrowIfDisposed();
        RequirePipeline();
        ValidateReference(referenceAudio);
        int[] ids = XttsOps<T>.IdsOf(Flatten(textTokens));
        if (ids.Length == 0 || ids.Length >= _options.MaxTextLength
            || ids.Any(id => id < 0 || id >= _options.TextVocabularySize))
            throw new ArgumentException("Text tokens must be nonempty, in vocabulary, and shorter than MaxTextLength.", nameof(textTokens));
        var condition = ExtractConditioningLatents(referenceAudio);
        var speaker = ExtractSpeakerEmbedding(referenceAudio);
        SetTrainingMode(false);
        using var _ = new NoGradScope<T>();
        var generated = _languageModel!.Generate(ids, condition, _samplingRandom);
        return _decoder!.Forward(generated.Latents, speaker);
    }

    /// <summary>Creates the voice value consumed by <see cref="Synthesize(string)"/>.</summary>
    public TtsVoice<T> CreateVoice(Tensor<T> referenceAudio) => new() { ReferenceAudio = referenceAudio };

    /// <summary>Loads the exact BPE tokenizer shipped by <c>coqui/XTTS-v2</c>.</summary>
    public void LoadTokenizer(string path)
    {
        if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("Tokenizer path required.", nameof(path));
        if (!File.Exists(path)) throw new FileNotFoundException($"Tokenizer not found: {path}", path);
        var tokenizer = ByteLevelBpeTokenizer.FromJson(File.ReadAllText(path));
        if (tokenizer.VocabularySize != _options.TextVocabularySize)
            throw new InvalidDataException(
                $"XTTS-v2 expects {_options.TextVocabularySize} tokenizer rows, but this tokenizer exposes {tokenizer.VocabularySize}.");
        _options.StartTextToken = tokenizer.TokenId("[START]");
        _options.StopTextToken = tokenizer.TokenId("[STOP]");
        if (_options.StartTextToken < 0 || _options.StopTextToken < 0)
            throw new InvalidDataException("The XTTS tokenizer must contain [START] and [STOP].");
        _tokenizer = tokenizer;
        _options.TokenizerPath = path;
    }

    /// <summary>Returns the released language-prefixed BPE ids.</summary>
    public int[] TokenizeText(string text, string? language = null)
    {
        if (_tokenizer is null)
            throw new InvalidOperationException("XTTS-v2 text requires the released vocab.json; call LoadTokenizer first.");
        string lang = NormalizeLanguage(language ?? _options.Language);
        string normalized = NormalizeText(text, lang);
        string marked = $"[{(lang == "zh" ? "zh-cn" : lang)}]{normalized.Replace(" ", "[SPACE]")}";
        int[] ids = _tokenizer.Encode(marked);
        if (ids.Length >= _options.MaxTextLength)
            throw new ArgumentException($"XTTS-v2 accepts fewer than {_options.MaxTextLength} text tokens.", nameof(text));
        return ids;
    }

    /// <summary>Applies the release's lowercasing/space cleanup and East-Asian romanization boundary.</summary>
    public static string NormalizeText(string text, string language)
    {
        if (string.IsNullOrWhiteSpace(text)) throw new ArgumentException("Text must not be empty.", nameof(text));
        string lang = NormalizeLanguage(language);
        if (lang == "ja") return Cutlet.Default.Romanize(text);
        // The released Hindi branch uses basic_cleaners: lowercase and whitespace only, preserving quotes/digits.
        if (lang == "hi") return Regex.Replace(text.ToLowerInvariant(), @"\s+", " ");
        string cleaned = text.Replace("\"", string.Empty);
        if (lang == "tr") cleaned = cleaned.Replace('İ', 'i').Replace('Ö', 'ö').Replace('Ü', 'ü');
        cleaned = cleaned.ToLowerInvariant();
        if (lang == "zh")
            cleaned = XttsChineseText.Normalize(cleaned);
        else if (lang == "en") cleaned = CleanEnglish(cleaned);
        else if (lang == "fr") cleaned = XttsLatinCleaner.French(cleaned);
        else if (lang == "es") cleaned = XttsLatinCleaner.Spanish(cleaned);
        else if (lang == "de") cleaned = XttsLatinCleaner.German(cleaned);
        else if (lang == "nl") cleaned = XttsLatinCleaner.Dutch(cleaned);
        else if (lang == "ko") cleaned = XttsLatinCleaner.Korean(cleaned);
        else if (lang == "pt") cleaned = XttsLatinCleaner.Portuguese(cleaned);
        else if (lang == "it") cleaned = XttsLatinCleaner.Italian(cleaned);
        else if (lang == "ru") cleaned = XttsLatinCleaner.Russian(cleaned);
        else if (lang == "pl") cleaned = XttsLatinCleaner.Polish(cleaned);
        else if (lang == "cs") cleaned = XttsLatinCleaner.Czech(cleaned);
        else if (lang == "hu") cleaned = XttsLatinCleaner.Hungarian(cleaned);
        else if (lang == "tr") cleaned = XttsLatinCleaner.Turkish(cleaned);
        else if (lang == "ar") cleaned = XttsLatinCleaner.Arabic(cleaned);
        else cleaned = XttsLatinCleaner.ApplyLexicalRules(cleaned, lang);
        if (lang == "ko") cleaned = RomanizeKorean(cleaned);
        return Regex.Replace(cleaned, @"\s+", " ");
    }

    private static string CleanEnglish(string text) => XttsLatinCleaner.English(text);

    /// <summary>Computes the averaged 32×1024 conditioning sequence from six-second chunks.</summary>
    public Tensor<T> ExtractConditioningLatents(Tensor<T> referenceAudio)
    {
        RequirePipeline();
        ValidateReference(referenceAudio);
        var input = HaspSpeakerEncoder<T>.Resample(Engine, Flatten(referenceAudio), _options.SampleRate, _options.InputSampleRate);
        int maximum = Math.Min(input.Length, (int)Math.Round(_options.MaximumReferenceSeconds * _options.InputSampleRate));
        int conditioning = Math.Min(maximum, (int)Math.Round(_options.ConditioningSeconds * _options.InputSampleRate));
        input = Engine.TensorSlice(input, new[] { 0 }, new[] { conditioning });
        int chunk = Math.Max(1, (int)Math.Round(_options.ConditioningChunkSeconds * _options.InputSampleRate));
        int minimum = Math.Max(1, (int)Math.Round(_options.MinimumConditioningChunkSeconds * _options.InputSampleRate));
        Tensor<T>? sum = null;
        int count = 0;
        for (int start = 0; start < input.Length; start += chunk)
        {
            int length = Math.Min(chunk, input.Length - start);
            if (length < minimum) continue;
            var audio = Engine.TensorSlice(input, new[] { start }, new[] { length });
            var mel = PowerMel(audio, _options.ConditioningFftSize, _options.ConditioningHopSize,
                _options.ConditioningWindowSize);
            var latent = _languageModel!.Condition(mel);
            sum = sum is null ? latent : Engine.TensorAdd(sum, latent);
            count++;
        }
        if (sum is null) throw new ArgumentException("Reference audio has no conditioning chunk of at least 0.33 seconds.", nameof(referenceAudio));
        return Engine.TensorMultiplyScalar(sum, NumOps.FromDouble(1.0 / count));
    }

    /// <summary>Returns the released H/ASP speaker embedding, L2-normalized to 512 dimensions.</summary>
    public Tensor<T> ExtractSpeakerEmbedding(Tensor<T> referenceAudio)
    {
        RequirePipeline();
        ValidateReference(referenceAudio);
        var input = HaspSpeakerEncoder<T>.Resample(Engine, Flatten(referenceAudio), _options.SampleRate,
            HaspSpeakerEncoder<T>.SampleRate);
        return _speakerEncoder!.Embed(input);
    }

    /// <summary>Encodes a 24 kHz waveform with the released 22.05 kHz dVAE tokenizer.</summary>
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        RequirePipeline();
        if (audio is null || audio.Length == 0) throw new ArgumentException("Audio must not be empty.", nameof(audio));
        var input = HaspSpeakerEncoder<T>.Resample(Engine, Flatten(audio), _options.SampleRate, _options.InputSampleRate);
        return _dvae!.Encode(PowerMel(input, _options.DvaeFftSize, _options.DvaeHopSize, _options.DvaeWindowSize));
    }

    /// <summary>Decodes dVAE ids to their normalized log-power mel reconstruction.</summary>
    public Tensor<T> DecodeTokensToMel(Tensor<T> tokens)
    {
        RequirePipeline();
        if (tokens is null || tokens.Length == 0) throw new ArgumentException("Tokens must not be empty.", nameof(tokens));
        return _dvae!.Decode(Flatten(tokens));
    }

    /// <summary>
    /// Reconstructs diagnostic audio from the dVAE mel decoder with mel-filter inversion and Griffin-Lim. XTTS speech
    /// synthesis itself does not use this method: the released HiFi-GAN consumes GPT hidden states directly.
    /// </summary>
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        var mel = DecodeTokensToMel(tokens);
        int frames = mel.Shape[0], bins = _options.DvaeFftSize / 2 + 1;
        var basis = PowerMelTransform(_options.DvaeFftSize, _options.DvaeHopSize, _options.DvaeWindowSize).MelFilterbank;
        // |S| = sqrt(max(1e-10, sum_m w[bin, m] exp(mel[frame, m] * stats[m]) / max(1e-12, sum_m w[bin, m]))), in double
        int bands = _options.MelChannels;
        var weights = Engine.TensorNarrow(Engine.TensorNarrow(basis.Cast<double>(), 0, 0, bins), 1, 0, bands);   // [bins, bands]
        var logMel = Engine.TensorNarrow(Engine.TensorNarrow(mel.Cast<double>(), 0, 0, frames), 1, 0, bands);
        var stats = Engine.TensorBroadcastTo(
            new Tensor<double>((double[])_melStats.Clone(), new[] { 1, bands }), new[] { frames, bands });
        var weighted = Engine.TensorMatMul(Engine.TensorExp(Engine.TensorMultiply(logMel, stats)), Engine.TensorTranspose(weights));
        var denominator = Engine.TensorClampMin(Engine.ReduceSum(weights, new[] { 1 }, false), 1e-12);
        var ratio = Engine.TensorDivide(weighted, Engine.TensorBroadcastTo(Engine.Reshape(denominator, new[] { 1, bins }), new[] { frames, bins }));
        var magnitude = Engine.TensorSqrt(Engine.TensorClampMin(ratio, 1e-10)).Cast<T>();
        var griffinLim = new AiDotNet.Diffusion.Audio.GriffinLim<T>(_options.DvaeFftSize, _options.DvaeHopSize,
            new XttsHannWindow(_options.DvaeWindowSize), iterations: 16, momentum: 0, seed: _options.SamplingSeed);
        var wave22050 = griffinLim.Reconstruct(magnitude);
        return HaspSpeakerEncoder<T>.Resample(Engine, wave22050, _options.InputSampleRate, _options.SampleRate);
    }

    /// <inheritdoc />
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        RequirePipeline();
        if (_optimizer is null) throw new NotSupportedException("Native optimizer is unavailable.");
        return TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => BuildObjective(sample), _optimizer);
    }

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        RequirePipeline();
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return BuildObjective(sample)[0];
        }
        finally { SetTrainingMode(wasTraining); }
    }

    protected override Tensor<T> PreprocessText(string text) => XttsOps<T>.Ids(TokenizeText(text));
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is { Count: > 0 })
        {
            Layers.AddRange(Architecture.Layers);
            return;
        }
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        _dvae = new XttsDiscreteVae<T>(Engine, _componentLayers, _options);
        _languageModel = new XttsLanguageModel<T>(Engine, _componentLayers, _options, random);
        _speakerEncoder = new HaspSpeakerEncoder<T>(Engine, _options.SpeakerEmbeddingDim, _options.SpeakerEncoderFilters);
        _componentLayers.AddRange(_speakerEncoder.Layers);
        _decoder = new XttsWaveformDecoder<T>(Engine, _componentLayers, _options);
        Layers.AddRange(_componentLayers);
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperPipeline)
        {
            var output = input;
            foreach (var layer in Layers) output = layer.Forward(output);
            return output;
        }
        var voice = RequireVoice();
        var condition = ExtractConditioningLatents(voice.ReferenceAudio!);
        var speaker = ExtractSpeakerEmbedding(voice.ReferenceAudio!);
        var generated = _languageModel!.Generate(XttsOps<T>.IdsOf(input), condition, _samplingRandom);
        return _decoder!.Forward(generated.Latents, speaker);
    }

    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "XTTS-v2 Native" : "XTTS-v2 ONNX",
            Description = "Casanova et al.'s multilingual zero-shot GPT/dVAE/HiFi-GAN system",
            FeatureCount = _options.LLMDim
        };
        metadata.AdditionalInfo["Architecture"] = "dVAE tokenizer + conditioning encoder/Perceiver + GPT-2 + H/ASP-conditioned HiFi-GAN";
        metadata.AdditionalInfo["Languages"] = XTTSv2Options.ReleasedLanguages.Length;
        metadata.AdditionalInfo["InputSampleRate"] = _options.InputSampleRate;
        metadata.AdditionalInfo["OutputSampleRate"] = _options.SampleRate;
        metadata.AdditionalInfo["AcousticTokenRate"] = _options.AcousticTokenRate;
        metadata.AdditionalInfo["OfficialWeightsAvailable"] = true;
        metadata.AdditionalInfo["TrainingPath"] = "Released GPT fine-tuning objective";
        return metadata;
    }

    private Tensor<T> BuildObjective(TtsTrainingSample<T> sample)
    {
        Validation.Guard.NotNull(sample);
        var audio = sample.Audio;
        if (audio is null && sample.CodecTokens is null)
            throw new ArgumentException("XTTS-v2 GPT training needs target audio or its frozen-dVAE codec tokens.", nameof(sample));
        var reference = sample.SpeakerReference ?? audio
            ?? throw new ArgumentException("XTTS-v2 GPT training needs a speaker reference recording.", nameof(sample));
        var condition = ExtractConditioningLatents(reference);
        int[] codes = sample.CodecTokens is not null
            ? XttsOps<T>.IdsOf(sample.CodecTokens)
            : XttsOps<T>.IdsOf(EncodeToTokens(audio!));
        int[] text = XttsOps<T>.IdsOf(sample.Tokens).Select(x => MathPolyfill.Clamp(x, 0, _options.TextVocabularySize - 1)).ToArray();
        return _languageModel!.Objective(text, codes, condition);
    }

    private AiDotNet.TextToSpeech.Vocoders.CenteredLogMel<T> PowerMelTransform(int fft, int hop, int window)
        => new(Engine, _options.InputSampleRate, fft, hop, window, _options.MelChannels, 0,
            _options.MelMaxFrequency, 1e-5, htkScale: true, naturalLog: true,
            spectrumPower: 2, htkAreaNormalization: true);

    private Tensor<T> PowerMel(Tensor<T> audio, int fft, int hop, int window)
    {
        // Coqui sets norm="slaney" but leaves torchaudio's default mel_scale="htk" unchanged.
        var mel = PowerMelTransform(fft, hop, window).Forward(audio, mean: new double[_options.MelChannels], scale: _melStats);
        return Engine.Reshape(mel, new[] { _options.MelChannels, mel.Shape[2] });
    }

    private void LoadConfiguredAssets()
    {
        if (_options.TokenizerPath is string tokenizerPath && !string.IsNullOrWhiteSpace(tokenizerPath))
            LoadTokenizer(tokenizerPath);
        if (_options.MelStatsPath is string melStatsPath && !string.IsNullOrWhiteSpace(melStatsPath))
            LoadMelStats(melStatsPath);
    }

    private void ConfigureBaseProperties()
    {
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.LLMDim;
    }

    private AdamWOptimizer<T, Tensor<T>, Tensor<T>> CreateDefaultOptimizer() => new(this,
        new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.LearningRate,
            WeightDecay = _options.WeightDecay,
            Beta1 = _options.AdamBeta1,
            Beta2 = _options.AdamBeta2,
            Epsilon = _options.AdamEpsilon,
            UseAdaptiveLearningRate = false
        });

    private void RequirePipeline()
    {
        ThrowIfDisposed();
        if (!HasPaperPipeline)
            throw new InvalidOperationException("This operation requires the native XTTS-v2 paper pipeline.");
    }

    private void ValidateReference(Tensor<T> audio)
    {
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        int minimum = (int)Math.Ceiling(_options.MinimumConditioningChunkSeconds * _options.SampleRate);
        if (audio.Length < minimum)
            throw new ArgumentException($"XTTS-v2 needs at least {_options.MinimumConditioningChunkSeconds:0.##} seconds ({minimum} samples) of reference audio.", nameof(audio));
    }

    private Tensor<T> Flatten(Tensor<T> input) => input.Shape.Length == 1 ? input : Engine.Reshape(input, new[] { input.Length });

    private static string NormalizeLanguage(string language)
    {
        if (string.IsNullOrWhiteSpace(language)) throw new ArgumentException("Language is required.", nameof(language));
        string value = language.ToLowerInvariant();
        if (value == "zh-cn") value = "zh";
        else value = value.Split('-')[0];
        if (!XTTSv2Options.ReleasedLanguages.Contains(value == "zh" ? "zh-cn" : value, StringComparer.Ordinal))
            throw new NotSupportedException($"XTTS-v2 does not support language '{language}'.");
        return value;
    }
    private static string RomanizeKorean(string text) => HangulAcademicRomanizer.Romanize(text);

    private static void ValidateOptions(XTTSv2Options options)
    {
        if (options.NumEncoderLayers != 0)
            throw new ArgumentException("XTTS GPT is decoder-only; NumEncoderLayers must be zero. Use ConditioningBlocks for its reference encoder.", nameof(options));
        if (options.NumCodebooks != 1) throw new ArgumentException("XTTS-v2 uses one dVAE codebook.", nameof(options));
        if (options.CodebookSize + 2 != options.AudioVocabularySize
            || options.StartAudioToken != options.CodebookSize || options.StopAudioToken != options.CodebookSize + 1)
            throw new ArgumentException("XTTS-v2 audio ids are 1024 codes followed by start and stop.", nameof(options));
        if (options.LLMDim <= 0 || options.LLMDim % options.NumHeads != 0 || options.LLMDim % options.ConditioningHeads != 0)
            throw new ArgumentException("XTTS dimensions must divide the GPT and conditioning head counts.", nameof(options));
        if (options.PerceiverHeadDim != 64) throw new ArgumentException("The released Perceiver uses 64-wide heads.", nameof(options));
        if (options.DvaeLayers != 2) throw new ArgumentException("The released XTTS-v2 dVAE has two stride-2 layers.", nameof(options));
        if (options.DecoderUpsampleRates.Length != options.DecoderUpsampleKernels.Length)
            throw new ArgumentException("Every decoder upsampling rate needs a kernel.", nameof(options));
        if (options.Temperature <= 0 || options.TopP < 0 || options.TopP > 1 || options.Speed <= 0)
            throw new ArgumentException("Sampling temperature and speed must be positive; top-p must be between zero and one.", nameof(options));
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(XTTSv2<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }

    private sealed class XttsHannWindow : IWindowFunction<T>
    {
        private readonly int _length;
        public XttsHannWindow(int length) => _length = length;
        public Vector<T> Create(int windowSize)
        {
            // A periodic Hann window of the configured length, centred and zero-padded to windowSize
            var window = new double[windowSize];
            int length = Math.Min(_length, windowSize), offset = (windowSize - length) / 2;
            for (int i = 0; i < length; i++)
                window[offset + i] = 0.5 - 0.5 * Math.Cos(2 * Math.PI * i / length);
            return new Tensor<double>(window, new[] { windowSize }).Cast<T>().ToVector();
        }
        public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Hanning;
    }
}
