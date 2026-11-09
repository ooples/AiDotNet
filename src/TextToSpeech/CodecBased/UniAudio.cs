using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.TextToSpeech.FrontEnd;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>UniAudio: a universal neural codec and multi-scale autoregressive audio Transformer.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// The native path implements the released zero-shot TTS sequence, the 16 kHz universal SEANet/RVQ codec, and the
/// paper's two-scale Transformer. Every condition is serialized as three-token patches. The global Transformer models
/// frames; its previous-frame state conditions a local Transformer that generates the three RVQ levels within a frame.
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "UniAudio: An Audio Foundation Model Toward Universal Audio Generation",
    "https://arxiv.org/abs/2310.00704",
    Year = 2024,
    Authors = "Yang et al.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, Beta1 = 0.9, Beta2 = 0.95, Epsilon = 1e-8,
    WeightDecay = 1e-3, WarmupSteps = 10000, Schedule = LearningRateSchedulerType.Noam,
    Provenance = RecipeProvenance.ReleasedCode,
    Source = "Yang et al. 2024, App. A.3 gives peak LR 1e-4 and 10k warmup under Noam decay; the released train.py "
        + "uses AdamW betas (0.9, 0.95), epsilon 1e-8 and weight decay 1e-3.")]
public partial class UniAudio<T> : TtsModelBase<T>, ICodecTts<T>
{
    private readonly UniAudioOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private UniAudioCodec<T>? _codec;
    private UniAudioMultiScaleTransformer<T>? _languageModel;
    private bool _useNativeMode;
    private bool _disposed;

    /// <summary>Creates an ONNX inference wrapper.</summary>
    public UniAudio(NeuralNetworkArchitecture<T> architecture, string modelPath, UniAudioOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new UniAudioOptions();
        ValidateOptions(_options);
        Options = _options;
        ConfigureBaseProperties();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
    }

    /// <summary>Creates the native codec and paper/release multi-scale Transformer.</summary>
    public UniAudio(
        NeuralNetworkArchitecture<T> architecture,
        UniAudioOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new UniAudioOptions();
        ValidateOptions(_options);
        Options = _options;
        _useNativeMode = true;
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this,
                learningRateOverride: _options.LearningRate, weightDecayOverride: _options.WeightDecay)
            ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        ConfigureBaseProperties();
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => _options.MaxTextLength;
    public int NumCodebooks => _options.NumCodebooks;
    public int CodebookSize => _options.CodebookSize;
    public int CodecFrameRate => _options.CodecFrameRate;
    public override int CodecTokenCodebooks => _options.NumCodebooks;
    public override int CodecTokenVocabulary => _options.CodebookSize;
    protected override int OutputFeatureWidth => _options.JointVocabularySize;
    protected override TtsSupervision RequiredVoice => TtsSupervision.ReferenceRecording;
    protected override TtsSupervision RequiredSupervision => _options.TrainingComponent == UniAudioTrainingComponent.UniversalCodec
        ? TtsSupervision.Recording : TtsSupervision.Text | TtsSupervision.Recording;
    public override bool TrainsOnSampleObjective => true;

    /// <summary>Whether this instance owns the released universal codec and both Transformer scales.</summary>
    public bool HasPaperPipeline => _codec is not null && _languageModel is not null;

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) => HasPaperPipeline
        ? LayerInputDomain.Indices(_options.PhoneVocabularySize) : base.GetInputDomain(inputShape);

    /// <summary>The global Transformer's layer count.</summary>
    public int GlobalLayerCount => _languageModel?.GlobalLayerCount ?? 0;

    /// <summary>The local Transformer's layer count.</summary>
    public int LocalLayerCount => _languageModel?.LocalLayerCount ?? 0;

    /// <summary>Whether all universal-codec codebooks have been initialized or restored.</summary>
    public bool CodecCodebooksInitialized => _codec?.Quantizer.CodebooksInitialized ?? false;

    /// <summary>Runs the configured raw-text frontend. The released v1 preset uses its Kaldi lexicon and alignment-phone IDs.</summary>
    public Tensor<T> TokenizePhones(string text)
    {
        ThrowIfDisposed();
        if (string.IsNullOrWhiteSpace(text)) throw new ArgumentException("Text must not be empty.", nameof(text));
        return UniAudioOps<T>.Ids(TokenizePhoneIds(text));
    }

    /// <summary>Encodes a whitespace-separated alignment-phone sequence using the released inventory and TTS duplicate removal.</summary>
    public Tensor<T> TokenizeReleasedPhoneSequence(string phones)
    {
        ThrowIfDisposed();
        if (phones is null) throw new ArgumentNullException(nameof(phones));
        return UniAudioOps<T>.Ids(UniAudioReleasedTokenizer.TokenizePhones(phones, _options.MaxTextLength));
    }

    private int[] TokenizePhoneIds(string text) => _options.UseReleasedTextFrontend
        ? UniAudioReleasedTokenizer.Tokenize(text, _options.MaxTextLength)
        : UniAudioPhoneTokenizer.Tokenize(text, _options.MaxTextLength);

    /// <summary>Creates a three-second-compatible zero-shot speaker prompt from a recording.</summary>
    public TtsVoice<T> CreateVoice(Tensor<T> recording, int sampleRate = 16000)
    {
        ThrowIfDisposed();
        Validation.Guard.NotNull(recording);
        var audio = Resample(recording, sampleRate, _options.SampleRate);
        int maximum = Math.Max(1, (int)Math.Round(_options.PromptSeconds * _options.SampleRate));
        if (audio.Length > maximum) audio = Engine.TensorSlice(audio, new[] { 0 }, new[] { maximum });
        return new TtsVoice<T> { ReferenceAudio = audio };
    }

    /// <summary>Synthesizes with a recording without changing the configured voice.</summary>
    public Tensor<T> SynthesizeWithVoice(string text, Tensor<T> recording, int sampleRate = 16000)
    {
        var previous = Voice;
        try
        {
            Voice = CreateVoice(recording, sampleRate);
            return Synthesize(text);
        }
        finally { Voice = previous; }
    }

    /// <summary>Serializes phonemes and speaker prompt, autoregressively generates three RVQ streams, then decodes them.</summary>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        RequirePipeline();
        using var _ = new NoGradScope<T>();
        return SynthesizePhones(TokenizePhoneIds(text),
            RequireVoice().ReferenceAudio!);
    }

    /// <summary>Encodes 16 kHz audio into flattened, codebook-offset UniAudio tokens.</summary>
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        ThrowIfDisposed();
        Validation.Guard.NotNull(audio);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(audio);
        RequirePipeline();
        using var _ = new NoGradScope<T>();
        return UniAudioOps<T>.Ids(_codec!.Flatten(_codec.Encode(audio)));
    }

    /// <summary>Encodes audio after band-limited resampling to the release's 16 kHz codec rate.</summary>
    public Tensor<T> EncodeToTokens(Tensor<T> audio, int sampleRate)
        => EncodeToTokens(Resample(audio, sampleRate, _options.SampleRate));

    /// <summary>Decodes flattened, codebook-offset tokens through the universal SEANet decoder.</summary>
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        ThrowIfDisposed();
        Validation.Guard.NotNull(tokens);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(tokens);
        RequirePipeline();
        using var _ = new NoGradScope<T>();
        return _codec!.Decode(_codec.Unflatten(UniAudioOps<T>.IdsOf(tokens)));
    }

    protected override Tensor<T> PreprocessText(string text) => TokenizePhones(text);

    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        RequirePipeline();
        if (_options.TrainingComponent == UniAudioTrainingComponent.UniversalCodec) return TrainCodecSample(sample);
        if (_optimizer is null) throw new NotSupportedException("Native optimizer is unavailable.");
        return TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => BuildObjective(sample), _optimizer);
    }

    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        RequirePipeline();
        if (_options.TrainingComponent == UniAudioTrainingComponent.UniversalCodec) EnsureCodecGan();
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return (_options.TrainingComponent == UniAudioTrainingComponent.UniversalCodec
                ? CodecGeneratorObjective(PrepareCodecAudio(sample.Audio!, randomCrop: false), _options.NumCodebooks)
                : BuildObjective(sample))[0];
        }
        finally { SetTrainingMode(wasTraining); }
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is { Count: > 0 })
        {
            Layers.AddRange(Architecture.Layers);
            return;
        }
        _codec = new UniAudioCodec<T>(Engine, _codecLayers, _options);
        _languageModel = new UniAudioMultiScaleTransformer<T>(Engine, _languageLayers, _options);
        ComponentLayers.AddRange(_codecLayers);
        ComponentLayers.AddRange(_languageLayers);
        if (_options.TrainingComponent == UniAudioTrainingComponent.UniversalCodec || _options.IncludeCodecDiscriminators) EnsureCodecGan();
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (HasPaperPipeline)
        {
            using var _ = new NoGradScope<T>();
            return SynthesizePhones(NormalizePhoneIds(UniAudioOps<T>.IdsOf(input)), RequireVoice().ReferenceAudio!);
        }
        var current = input;
        foreach (var layer in Layers) current = layer.Forward(current);
        return current;
    }

    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    protected override bool SupportsParameterMutation => _useNativeMode;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "UniAudio-Native" : "UniAudio-ONNX",
            Description = "UniAudio universal SEANet/RVQ codec and global/local multi-scale Transformer",
            FeatureCount = _options.LLMDim,
        };
        metadata.AdditionalInfo["Architecture"] = "16 kHz SEANet/RVQ -> 24-layer global Transformer -> 8-layer local Transformer";
        metadata.AdditionalInfo["GlobalLayers"] = _options.NumEncoderLayers;
        metadata.AdditionalInfo["LocalLayers"] = _options.NumLLMLayers;
        metadata.AdditionalInfo["GlobalContextPatches"] = _options.MaxContextPatches;
        metadata.AdditionalInfo["JointVocabularySize"] = _options.JointVocabularySize;
        metadata.AdditionalInfo["CodecQuantizersAvailable"] = _options.CodecTotalQuantizers;
        metadata.AdditionalInfo["CodecQuantizersUsed"] = _options.NumCodebooks;
        metadata.AdditionalInfo["CodecFrameRate"] = _options.CodecFrameRate;
        metadata.AdditionalInfo["Sampling"] = $"top-{_options.TopK}, temperature {_options.Temperature}";
        metadata.AdditionalInfo["SpeakerPromptSeconds"] = _options.PromptSeconds;
        metadata.AdditionalInfo["OfficialWeightsAvailable"] = true;
        metadata.AdditionalInfo["Mode"] = _useNativeMode ? "Native" : "ONNX";
        return metadata;
    }

    private Tensor<T> SynthesizePhones(int[] phoneIds, Tensor<T> promptAudio)
    {
        var prompt = PromptTokens(promptAudio);
        var prefix = BuildPrefix(phoneIds, prompt);
        int prefixPatches = prefix.Count / _options.NumCodebooks;
        int remaining = Math.Max(1, _options.MaxContextPatches - prefixPatches - 1);
        int maximum = Math.Min(remaining, Math.Min(_options.MaximumGeneratedFrames,
            Math.Max(1, (int)Math.Floor(prefixPatches * _options.MaximumLengthRatio))));
        int minimum = Math.Min(maximum, Math.Max(_options.MinimumGeneratedFrames,
            // The author masks EOS while g_idx < minlen - 1.
            (int)Math.Floor(prefixPatches * _options.MinimumLengthRatio) - 1));
        var generatedGlobal = _languageModel!.Generate(prefix, minimum, maximum, new Random(_options.SamplingSeed));
        int complete = generatedGlobal.Length / _options.NumCodebooks * _options.NumCodebooks;
        if (complete == 0) return new Tensor<T>(new[] { 0 });
        var local = new int[complete];
        for (int i = 0; i < complete; i++) local[i] = generatedGlobal[i] - _options.AudioTokenOffset;
        return _codec!.Decode(_codec.Unflatten(local));
    }

    private Tensor<T> BuildObjective(TtsTrainingSample<T> sample)
    {
        var audio = sample.Audio ?? throw new ArgumentException("UniAudio training requires target audio.", nameof(sample));
        int[] phones = NormalizePhoneIds(UniAudioOps<T>.IdsOf(sample.Tokens));
        int[] target = sample.CodecTokens is not null
            ? NormalizeCodecTokens(UniAudioOps<T>.IdsOf(sample.CodecTokens))
            : UniAudioOps<T>.IdsOf(EncodeToTokens(audio));
        var reference = sample.SpeakerReference ?? audio;
        int[] prompt = PromptTokens(reference);
        var prefix = BuildPrefix(phones, prompt);
        int firstTarget = prefix.Count;
        var sequence = new List<int>(prefix);
        sequence.AddRange(ToJointAudio(target));
        AddPatch(sequence, _options.AudioEndToken);
        AddPatch(sequence, _options.EndToken);
        if (sequence.Count / _options.NumCodebooks > _options.MaxContextPatches)
            throw new ArgumentException("The UniAudio training sequence exceeds the global context window.", nameof(sample));
        var logits = _languageModel!.Forward(sequence, firstTarget);
        return UniAudioOps<T>.TeacherForcedLoss(Engine, logits, sequence, firstTarget,
            _options.PrefixLanguageModel, _options.PadToken);
    }

    private List<int> BuildPrefix(IReadOnlyList<int> phoneIds, IReadOnlyList<int> promptTokens)
    {
        var sequence = new List<int>();
        AddPatch(sequence, _options.StartToken);
        AddPatch(sequence, _options.TtsTaskToken);
        AddPatch(sequence, _options.PhoneStartToken);
        foreach (int phone in phoneIds.Take(_options.MaxTextLength))
            AddPatch(sequence, _options.PhoneTokenOffset + MathPolyfill.Clamp(phone, 0, _options.PhoneVocabularySize - 1));
        AddPatch(sequence, _options.PhoneEndToken);
        AddPatch(sequence, _options.AudioPromptStartToken);
        sequence.AddRange(ToJointAudio(promptTokens));
        AddPatch(sequence, _options.AudioPromptEndToken);
        AddPatch(sequence, _options.AudioStartToken);
        if (sequence.Count / _options.NumCodebooks >= _options.MaxContextPatches)
            throw new ArgumentException("The UniAudio condition sequence leaves no room for generated audio.");
        return sequence;
    }

    private int[] PromptTokens(Tensor<T> recording)
    {
        var audio = Engine.Reshape(recording, new[] { recording.Length });
        int maximum = Math.Max(_codec!.HopLength, (int)Math.Round(_options.PromptSeconds * _options.SampleRate));
        if (audio.Length > maximum) audio = Engine.TensorSlice(audio, new[] { 0 }, new[] { maximum });
        return _codec.Flatten(_codec.Encode(audio));
    }

    private int[] ToJointAudio(IReadOnlyList<int> local)
    {
        int complete = local.Count / _options.NumCodebooks * _options.NumCodebooks;
        var joint = new int[complete];
        for (int i = 0; i < complete; i++) joint[i] = _options.AudioTokenOffset + local[i];
        return joint;
    }

    private int[] NormalizeCodecTokens(IReadOnlyList<int> tokens)
    {
        int complete = tokens.Count / _options.NumCodebooks * _options.NumCodebooks;
        var result = new int[complete];
        for (int i = 0; i < complete; i++)
        {
            int q = i % _options.NumCodebooks;
            int token = tokens[i];
            if (token >= _options.AudioTokenOffset) token -= _options.AudioTokenOffset;
            if (token < q * _options.CodebookSize) token += q * _options.CodebookSize;
            result[i] = MathPolyfill.Clamp(token, q * _options.CodebookSize, (q + 1) * _options.CodebookSize - 1);
        }
        return result;
    }

    private int[] NormalizePhoneIds(IReadOnlyList<int> phones)
    {
        var result = new int[Math.Min(phones.Count, _options.MaxTextLength)];
        for (int i = 0; i < result.Length; i++)
        {
            int id = phones[i];
            if (id >= _options.PhoneTokenOffset) id -= _options.PhoneTokenOffset;
            result[i] = MathPolyfill.Clamp(id, 0, _options.PhoneVocabularySize - 1);
        }
        return result.Length == 0 ? new[] { 0 } : result;
    }

    private void AddPatch(List<int> sequence, int token)
    {
        for (int q = 0; q < _options.NumCodebooks; q++) sequence.Add(token);
    }

    private Tensor<T> Resample(Tensor<T> audio, int from, int to)
    {
        var flat = Engine.Reshape(audio, new[] { audio.Length });
        return from == to ? flat : AudioHelper<T>.ResampleBandLimited(Engine, flat, from, to);
    }

    private void ConfigureBaseProperties()
    {
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.LLMDim;
    }

    private static void ValidateOptions(UniAudioOptions options)
    {
        if (options.DropoutRate != 0)
            throw new ArgumentException("UniAudio's released global/local Transformers have no dropout; DropoutRate must be zero.");
        // The release uses three codes, while the author's MegaByteModel supports
        // a configurable na_repeat that divides its Transformer width.
        if (options.NumCodebooks <= 0)
            throw new ArgumentException("UniAudio needs a positive number of RVQ codes per patch.");
        if (options.CodebookSize <= 0 || options.CodecTotalQuantizers < options.NumCodebooks)
            throw new ArgumentException("The codec must provide every configured positive-sized codebook.");
        if (options.CodecRatios is null || options.CodecRatios.Length == 0 || options.CodecRatios.Any(ratio => ratio <= 0)
            || options.SampleRate <= 0 || options.HopSize <= 0 || options.SampleRate % options.HopSize != 0)
            throw new ArgumentException("UniAudio's codec strides and sample rate must define a positive integral frame rate.");
        if (options.LLMDim <= 0 || options.LLMDim % options.NumCodebooks != 0
            || options.NumHeads <= 0 || options.LLMDim % options.NumHeads != 0)
            throw new ArgumentException("The Transformer width must divide evenly across codebooks and attention heads.");
        if (options.NumEncoderLayers <= 0 || options.NumLLMLayers <= 0
            || options.GlobalFeedForwardDim <= 0 || options.LocalFeedForwardDim <= 0)
            throw new ArgumentException("Both Transformer scales need positive depth and feed-forward width.");
        if (options.JointVocabularySize != options.VocabSize
            || options.AudioTokenOffset + options.NumCodebooks * options.CodebookSize >= options.JointVocabularySize
            || options.PhoneTokenOffset + options.PhoneVocabularySize > options.JointVocabularySize)
            throw new ArgumentException("The joint vocabulary does not contain the configured audio and phoneme ranges.");
        if (options.MaxContextPatches <= 0 || options.MaximumGeneratedFrames <= 0 || options.MinimumGeneratedFrames < 0)
            throw new ArgumentException("Context and generation lengths must be positive.");
        if (options.TopK <= 0 || options.Temperature <= 0 || options.PromptSeconds <= 0
            || options.MinimumLengthRatio < 0 || options.MaximumLengthRatio <= 0)
            throw new ArgumentException("Sampling and prompt settings must be positive.");
    }

    private void RequirePipeline()
    {
        if (!HasPaperPipeline)
            throw new NotSupportedException("This operation requires UniAudio's native codec and multi-scale Transformer.");
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(UniAudio<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
