using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Audio.Codecs;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// Seed-TTS's disclosed autoregressive pipeline: speech tokenizer, causal token LM, token diffusion, and acoustic vocoder.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This implementation follows Figure 1 and Section 2 of the technical report. The LM predicts speech tokens while
/// masking text positions from its loss; diffusion restores continuous acoustic detail in a coarse-to-fine process;
/// and a separately trainable neural vocoder produces the waveform. A reference recording supplies both the speech
/// token prefix and the timbre condition used by the paper's self-distillation extension.
/// </para>
/// <para>
/// ByteDance explicitly did not release model code or weights. The native graph therefore exposes all undisclosed
/// dimensions as options and loads only clearly labelled AiDotNet-converted flat checkpoints; it does not claim that
/// its configurable defaults are proprietary production settings.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Seed-TTS: A Family of High-Quality Versatile Speech Generation Models",
    "https://arxiv.org/abs/2406.02430",
    Year = 2024,
    Authors = "Anastassiou et al."
)]
public partial class SeedTTS<T> : TtsModelBase<T>, ICodecTts<T>
{
    private readonly SeedTTSOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _discriminatorOptimizer;
    private CodecWaveformGan<T>? _codecGan;
    private bool _discriminatorTurn;
    private readonly List<LayerBase<T>> _languageLayers = new(), _diffusionLayers = new();
    private SeedTTSSpeechTokenizer<T>? _speechTokenizer;
    private SeedTTSTokenLanguageModel<T>? _tokenLanguageModel;
    private SeedTTSTokenDiffusion<T>? _tokenDiffusion;
    private SeedTTSAcousticVocoder<T>? _acousticVocoder;
    private FullyConnectedLayer<T>? _timbreEncoder;
    private bool _useNativeMode;
    private bool _disposed;

    /// <summary>Creates an ONNX inference wrapper.</summary>
    public SeedTTS(NeuralNetworkArchitecture<T> architecture, string modelPath, SeedTTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new SeedTTSOptions();
        ValidateOptions(_options);
        Options = _options;
        _useNativeMode = false;
        ConfigureBaseProperties();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates the native four-block Seed-TTS graph.</summary>
    public SeedTTS(
        NeuralNetworkArchitecture<T> architecture,
        SeedTTSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new SeedTTSOptions();
        ValidateOptions(_options);
        Options = _options;
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                WeightDecay = _options.WeightDecay,
            });
        ConfigureBaseProperties();
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;
    /// <inheritdoc />
    public int NumCodebooks => _options.NumCodebooks;
    /// <inheritdoc />
    public int CodebookSize => _options.CodebookSize;
    public int CodecFrameRate => _options.CodecFrameRate;
    public override int CodecTokenCodebooks => 1;
    public override int CodecTokenVocabulary => _options.CodebookSize;
    protected override int OutputFeatureWidth => _options.CodebookSize + 2;
    public SeedTTSTrainingComponent TrainingComponent => _options.TrainingComponent;

    /// <summary>Whether the native graph contains the four paper blocks rather than caller-supplied sequential layers.</summary>
    public bool HasPaperPipeline => _speechTokenizer is not null && _tokenLanguageModel is not null
        && _tokenDiffusion is not null && _acousticVocoder is not null;

    /// <summary>Synthesizes with the configured voice prompt when one is present, otherwise from the learned prior.</summary>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        if (!HasPaperPipeline) return PostprocessAudio(Predict(PreprocessText(text)));
        return SynthesizeWithPrompt(text, string.Empty, Voice?.ReferenceAudio);
    }

    /// <summary>
    /// Performs the paper's in-context continuation from a text/audio prompt. Prompt text may be empty for the
    /// cross-lingual, transcript-free convenience path used by <c>IVoiceCloner</c>.
    /// </summary>
    public Tensor<T> SynthesizeWithPrompt(string text, string promptText, Tensor<T>? referenceAudio)
    {
        ThrowIfDisposed();
        if (string.IsNullOrWhiteSpace(text)) throw new ArgumentException("Target text is required.", nameof(text));
        if (!HasPaperPipeline)
            throw new NotSupportedException("Prompt-conditioned synthesis requires the native Seed-TTS paper pipeline.");

        int[] targetText = IdsOf(PreprocessText(text), _options.VocabSize);
        int[] prefixText = string.IsNullOrWhiteSpace(promptText)
            ? Array.Empty<int>()
            : IdsOf(PreprocessText(promptText), _options.VocabSize);
        int[] combinedText = prefixText.Concat(targetText).Take(_options.MaxTextLength).ToArray();
        Tensor<T>? timbre = null;
        int[] promptSpeech = Array.Empty<int>();
        if (referenceAudio is not null)
        {
            promptSpeech = IdsOf(_speechTokenizer!.Encode(referenceAudio), _options.CodebookSize);
            timbre = ExtractTimbreEmbedding(referenceAudio);
        }

        using var _ = new NoGradScope<T>();
        var generated = _tokenLanguageModel!.Generate(combinedText, promptSpeech, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.GenerationSeed));
        var tokenTensor = Tokens(generated);
        var acoustic = _tokenDiffusion!.Refine(tokenTensor, timbre);
        return PostprocessAudio(_acousticVocoder!.Decode(acoustic));
    }

    /// <inheritdoc />
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(audio);
        if (_speechTokenizer is null)
            throw new NotSupportedException("Speech tokenization requires the native Seed-TTS paper pipeline.");
        using var _ = new NoGradScope<T>();
        return _speechTokenizer.Encode(audio);
    }

    /// <inheritdoc />
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(tokens);
        if (_tokenDiffusion is null || _acousticVocoder is null)
            throw new NotSupportedException("Speech-token decoding requires the native Seed-TTS paper pipeline.");
        using var _ = new NoGradScope<T>();
        return _acousticVocoder.Decode(_tokenDiffusion.Refine(NormalizeTokens(tokens), null));
    }

    /// <summary>Extracts the timbre reference integrated into the self-distilled diffusion model.</summary>
    public Tensor<T> ExtractTimbreEmbedding(Tensor<T> referenceAudio)
    {
        ThrowIfDisposed();
        if (_speechTokenizer is null || _timbreEncoder is null)
            throw new NotSupportedException("Timbre extraction requires the native Seed-TTS paper pipeline.");
        using var _ = new NoGradScope<T>();
        var features = _speechTokenizer.Analyze(referenceAudio);
        // Time-average of the first AcousticDimension feature channels.
        var pooled = Engine.TensorDivideScalar(
            Engine.ReduceSum(Engine.TensorSlice(features, new[] { 0, 0 }, new[] { features.Shape[0], _options.AcousticDimension }),
                new[] { 0 }, keepDims: false),
            NumOps.FromDouble(features.Shape[0]));
        var embedding = _timbreEncoder.Forward(pooled);
        // Unit L2 norm when the norm exceeds 1e-12; the norm is taken after scaling by the peak magnitude so its sum of
        // squares cannot overflow a narrow T.
        if (embedding.Length == 0) return embedding;
        var peak = Engine.TensorMaxValue(Engine.TensorAbs(embedding));
        if (NumOps.Equals(peak, NumOps.Zero)) return embedding;
        var scaled = Engine.TensorDivideScalar(embedding, peak);
        var scaledNorm = NumOps.Sqrt(Engine.TensorSum(Engine.TensorSquare(scaled)));
        return NumOps.GreaterThan(NumOps.Multiply(peak, scaledNorm), NumOps.FromDouble(1e-12))
            ? Engine.TensorDivideScalar(scaled, scaledNorm)
            : embedding;
    }

    protected override TtsSupervision RequiredSupervision => _options.TrainingComponent switch
    {
        SeedTTSTrainingComponent.TokenLanguageModel => TtsSupervision.Text | TtsSupervision.CodecTokens,
        SeedTTSTrainingComponent.TokenDiffusion => TtsSupervision.CodecTokens | TtsSupervision.Recording,
        SeedTTSTrainingComponent.AcousticVocoder => TtsSupervision.Recording,
        SeedTTSTrainingComponent.SpeechTokenizer => TtsSupervision.Recording,
        SeedTTSTrainingComponent.TimbreSelfDistillation => TtsSupervision.PromptCodecTokens
            | TtsSupervision.SpeakerReference | TtsSupervision.Recording,
        _ => throw new InvalidOperationException("Unknown Seed-TTS training component."),
    };

    public override bool TrainsOnSampleObjective => true;

    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        if (!HasPaperPipeline || _optimizer is null)
            throw new NotSupportedException("Sample-level Seed-TTS training requires the native paper pipeline.");
        if (_options.TrainingComponent == SeedTTSTrainingComponent.SpeechTokenizer || _options.TrainingComponent == SeedTTSTrainingComponent.AcousticVocoder)
        {
            EnsureCodecGan();
            _discriminatorTurn = true;
            try
            {
                TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) =>
                {
                    Tensor<T> generated;
                    using (new NoGradScope<T>()) generated = CodecTrainingWaveform(sample.Audio!);
                    var real = Engine.TensorNarrow(Engine.Reshape(sample.Audio!, new[] { sample.Audio!.Length }), 0, 0, generated.Length);
                    return _codecGan!.DiscriminatorLoss(real, generated);
                }, _discriminatorOptimizer!);
            }
            finally { _discriminatorTurn = false; }
        }
        return TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => BuildObjective(sample, training: true), _optimizer);
    }

    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        if (!HasPaperPipeline) throw new NotSupportedException("The objective requires the native paper pipeline.");
        if (_options.TrainingComponent == SeedTTSTrainingComponent.SpeechTokenizer || _options.TrainingComponent == SeedTTSTrainingComponent.AcousticVocoder) EnsureCodecGan();
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return BuildObjective(sample)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is { Count: > 0 })
        {
            Layers.AddRange(Architecture.Layers);
            return;
        }

        _speechTokenizer = new SeedTTSSpeechTokenizer<T>(Engine, ComponentLayers, _options);
        _tokenLanguageModel = new SeedTTSTokenLanguageModel<T>(Engine, _languageLayers, _options);
        ComponentLayers.AddRange(_languageLayers);
        _timbreEncoder = new FullyConnectedLayer<T>(_options.AcousticDimension, _options.SpeakerEmbeddingDim,
            new TanhActivation<T>());
        _diffusionLayers.Add(_timbreEncoder);
        _tokenDiffusion = new SeedTTSTokenDiffusion<T>(Engine, _diffusionLayers, _options);
        ComponentLayers.AddRange(_diffusionLayers);
        _acousticVocoder = new SeedTTSAcousticVocoder<T>(Engine, ComponentLayers, _options);
    }

    private void EnsureCodecGan()
    {
        if (_codecGan is not null) return;
        _codecGan = new CodecWaveformGan<T>(Engine, _options.SampleRate, _options.TrainingMelWindows, _options.TrainingMelBins,
            _options.DiscriminatorWindows, new[] { 0.0, 0.1, 0.25, 0.5, 0.75, 1.0 }, _options.DiscriminatorChannels, _options.DiscriminatorWidthDivisor);
        ComponentLayers.AddRange(_codecGan.Layers);
        _discriminatorOptimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = _options.LearningRate, WeightDecay = _options.WeightDecay });
    }

    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperPipeline) return parameters;
        IEnumerable<LayerBase<T>> layers = _options.TrainingComponent switch
        {
            SeedTTSTrainingComponent.TokenLanguageModel => _languageLayers,
            SeedTTSTrainingComponent.TokenDiffusion or SeedTTSTrainingComponent.TimbreSelfDistillation => _diffusionLayers,
            _ when _discriminatorTurn => _codecGan!.Layers,
            SeedTTSTrainingComponent.SpeechTokenizer => _speechTokenizer!.Layers.Concat(_acousticVocoder!.Layers),
            _ => _acousticVocoder!.Layers,
        };
        var selected = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(layers.Cast<ILayer<T>>().ToList(), -1),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(selected.Contains).ToList();
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (_tokenLanguageModel is not null)
            return _tokenLanguageModel.TextConditionLogits(IdsOf(input, _options.VocabSize));
        var current = input;
        foreach (var layer in Layers) current = layer.Forward(current);
        return current;
    }

    /// <summary>The report does not publish its text tokenizer. This native frontend uses reversible UTF-8 byte IDs;
    /// it does not claim equivalence to ByteDance's undisclosed tokenizer. Numeric training inputs may supply the
    /// actual tokenizer's IDs when training a correspondingly configured vocabulary.</summary>
    protected override Tensor<T> PreprocessText(string text)
    {
        Validation.Guard.NotNull(text);
        if (text.Length == 0) throw new ArgumentException("Seed-TTS text cannot be empty.", nameof(text));
        var bytes = new System.Text.UTF8Encoding(false, true).GetBytes(text);
        var ids = bytes.Take(_options.MaxTextLength).Select(value => (int)value).ToArray();
        if (ids.Any(value => value >= _options.VocabSize))
            throw new ArgumentException("The native Seed-TTS UTF-8 frontend needs a vocabulary containing every input byte (256 entries support all text).", nameof(text));
        return Tokens(ids);
    }

    public override LayerInputDomain GetInputDomain(int[]? inputShape) => HasPaperPipeline
        ? LayerInputDomain.Indices(_options.VocabSize) : base.GetInputDomain(inputShape);

    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    protected override bool SupportsParameterMutation => _useNativeMode;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "Seed-TTS-Native" : "Seed-TTS-ONNX",
            Description = "Seed-TTS four-block autoregressive speech-generation pipeline (Anastassiou et al., 2024)",
            FeatureCount = _options.LLMDim,
        };
        metadata.AdditionalInfo["Architecture"] = "SpeechTokenizer -> CausalTokenLM -> TokenDiffusion -> AcousticVocoder";
        metadata.AdditionalInfo["TextLossMasked"] = true;
        metadata.AdditionalInfo["TextTokenizer"] = "Native UTF-8 bytes; author tokenizer undisclosed";
        metadata.AdditionalInfo["TrainingComponent"] = _options.TrainingComponent.ToString();
        metadata.AdditionalInfo["OfficialWeightsAvailable"] = false;
        metadata.AdditionalInfo["Mode"] = _useNativeMode ? "Native" : "ONNX";
        return metadata;
    }

    private Tensor<T> BuildObjective(TtsTrainingSample<T> sample, bool training = false)
    {
        return _options.TrainingComponent switch
        {
            SeedTTSTrainingComponent.TokenLanguageModel => LanguageModelObjective(sample),
            SeedTTSTrainingComponent.TokenDiffusion => DiffusionObjective(sample, selfDistillation: false, training: training),
            SeedTTSTrainingComponent.AcousticVocoder => VocoderObjective(sample),
            SeedTTSTrainingComponent.SpeechTokenizer => VocoderObjective(sample),
            SeedTTSTrainingComponent.TimbreSelfDistillation => DiffusionObjective(sample, selfDistillation: true, training: training),
            _ => throw new InvalidOperationException("Unknown Seed-TTS training component."),
        };
    }

    private Tensor<T> LanguageModelObjective(TtsTrainingSample<T> sample)
    {
        int[] text = IdsOf(sample.Tokens, _options.VocabSize);
        int[] speech = TrainingSpeechTokens(sample);
        int[]? prompt = sample.PromptCodecTokens is null
            ? null
            : IdsOf(sample.PromptCodecTokens, _options.CodebookSize);
        return _tokenLanguageModel!.Loss(text, speech, prompt);
    }

    private Tensor<T> DiffusionObjective(TtsTrainingSample<T> sample, bool selfDistillation, bool training)
    {
        var audio = sample.Audio ?? throw new ArgumentException("Diffusion training requires Audio.", nameof(sample));
        var target = Detached(_speechTokenizer!.Analyze(audio));
        Tensor<T> tokenSource;
        Tensor<T>? timbre = null;
        if (selfDistillation)
        {
            tokenSource = sample.PromptCodecTokens
                ?? throw new ArgumentException("Self-distillation requires alternate-timbre PromptCodecTokens.", nameof(sample));
            timbre = TrainingTimbreEmbedding(sample.SpeakerReference
                ?? throw new ArgumentException("Self-distillation requires the original-timbre SpeakerReference.", nameof(sample)));
        }
        else
        {
            tokenSource = sample.CodecTokens ?? _speechTokenizer.Encode(audio);
            if (sample.SpeakerReference is not null) timbre = TrainingTimbreEmbedding(sample.SpeakerReference);
        }
        return _tokenDiffusion!.DenoisingLoss(AlignTokens(tokenSource, target.Shape[0]), target, timbre,
            training ? null : AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.GenerationSeed + 17));
    }

    private Tensor<T> TrainingTimbreEmbedding(Tensor<T> reference)
    {
        var features = Detached(_speechTokenizer!.Analyze(reference));
        var pooled = Engine.ReduceMean(features, new[] { 0 }, false);
        var embedding = _timbreEncoder!.Forward(pooled);
        var norm = Engine.TensorClampMin(Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(embedding, embedding),
            new[] { 0 }, true)), NumOps.FromDouble(1e-12));
        return Engine.TensorDivide(embedding, Engine.TensorBroadcastTo(norm, embedding.Shape.ToArray()));
    }

    private Tensor<T> VocoderObjective(TtsTrainingSample<T> sample)
    {
        var audio = sample.Audio ?? throw new ArgumentException("Vocoder training requires Audio.", nameof(sample));
        var generated = CodecTrainingWaveform(audio);
        var real = Engine.TensorNarrow(Engine.Reshape(audio, new[] { audio.Length }), 0, 0, generated.Length);
        var loss = _codecGan!.GeneratorLoss(real, generated);
        if (_options.TrainingComponent == SeedTTSTrainingComponent.SpeechTokenizer)
        {
            if (_speechTokenizer!.Quantizer.CodebookLoss is { } codebook) loss = Engine.TensorAdd(loss, Engine.Reshape(codebook, new[] { 1 }));
            if (_speechTokenizer.Quantizer.CommitmentLoss is { } commitment)
                loss = Engine.TensorAdd(loss, Engine.Reshape(Engine.TensorMultiplyScalar(commitment, NumOps.FromDouble(0.25)), new[] { 1 }));
        }
        return loss;
    }

    private Tensor<T> CodecTrainingWaveform(Tensor<T> audio)
    {
        var acoustic = _options.TrainingComponent == SeedTTSTrainingComponent.SpeechTokenizer
            ? _speechTokenizer!.QuantizeForTraining(audio) : Detached(_speechTokenizer!.Analyze(audio));
        var generated = _acousticVocoder!.Decode(acoustic);
        return generated.Length > audio.Length ? Engine.TensorNarrow(generated, 0, 0, audio.Length) : generated;
    }

    private int[] TrainingSpeechTokens(TtsTrainingSample<T> sample)
    {
        if (sample.CodecTokens is not null) return IdsOf(sample.CodecTokens, _options.CodebookSize);
        if (sample.Audio is null) throw new ArgumentException("Supply CodecTokens or Audio.", nameof(sample));
        using var _ = new NoGradScope<T>();
        return IdsOf(_speechTokenizer!.Encode(sample.Audio), _options.CodebookSize);
    }

    private Tensor<T> AlignTokens(Tensor<T> source, int frames)
    {
        int[] ids = IdsOf(source, _options.CodebookSize);
        var aligned = new int[frames];
        for (int i = 0; i < frames; i++) aligned[i] = ids[Math.Min(i, ids.Length - 1)];
        return IdTensor(aligned);
    }

    private Tensor<T> NormalizeTokens(Tensor<T> source) => Tokens(IdsOf(source, _options.CodebookSize));

    private Tensor<T> Tokens(IReadOnlyList<int> ids)
    {
        return IdTensor(ids);
    }

    private static Tensor<T> Detached(Tensor<T> tensor) => new(tensor._shape, tensor.ToVector());

    private int[] IdsOf(Tensor<T> tensor, int vocabulary)
        => FrontEnd.TtsTokenIds.Read(tensor, vocabulary, int.MaxValue, "Seed-TTS");

    private void ConfigureBaseProperties()
    {
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.SamplesPerCodecFrame;
        base.HiddenDim = _options.LLMDim;
    }

    private static void ValidateOptions(SeedTTSOptions options)
    {
        if ((double.IsNaN(options.DropoutRate) || double.IsInfinity(options.DropoutRate)) || options.DropoutRate is < 0 or >= 1)
            throw new ArgumentOutOfRangeException(nameof(options.DropoutRate), "DropoutRate must be finite and lie in [0, 1).");
        if (options.SampleRate <= 0 || options.CodecFrameRate <= 0
            || options.SampleRate % options.CodecFrameRate != 0)
            throw new ArgumentException("SampleRate must be a positive integer multiple of CodecFrameRate.", nameof(options));
        if (options.NumCodebooks != 1)
            throw new ArgumentException("The report does not disclose an RVQ layout; this native graph represents one token stream.", nameof(options));
        if (options.CodecRates.Length == 0 || options.CodecRates.Any(r => r <= 1)
            || options.CodecRates.Aggregate(1, (a, b) => checked(a * b)) != options.SamplesPerCodecFrame)
            throw new ArgumentException("CodecRates must multiply to SampleRate / CodecFrameRate.", nameof(options));
        if (options.TokenizerChannels <= 0 || options.TokenizerCodeDimension <= 0 || options.VocoderHiddenDimension < (1 << options.CodecRates.Length))
            throw new ArgumentException("Tokenizer/vocoder widths must support every convolutional stage.", nameof(options));
        if (options.CodebookSize < 2 || options.VocabSize < 2 || options.AcousticDimension <= 0
            || options.LLMDim <= 0 || options.NumHeads <= 0 || options.LLMDim % options.NumHeads != 0
            || options.AcousticDimension % options.NumHeads != 0)
            throw new ArgumentException("Seed-TTS dimensions and vocabularies are invalid or not divisible by NumHeads.", nameof(options));
        if (options.NumLLMLayers <= 0 || options.DiffusionLayers <= 0 || options.DiffusionSteps <= 0
            || options.MaxTextLength <= 0
            || options.MaxCodecFrames <= 0 || options.MinimumGeneratedCodecFrames <= 0
            || options.FramesPerTextToken <= 0 || options.SpeakerEmbeddingDim <= 0)
            throw new ArgumentException("Seed-TTS layer counts, generation limits, and embedding dimensions must be positive.", nameof(options));
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SeedTTS<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
