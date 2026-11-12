using AiDotNet.Attributes;
using AiDotNet.Audio.Codecs;
using AiDotNet.Enums;
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
/// Spark-TTS: Qwen2.5 generates a single semantic speech-token stream which BiCodec renders with decoupled speaker tokens.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// The native graph follows the official 0.5B release: XLSR hidden-state mixing, a Vocos semantic encoder and
/// 8,192-way factorized VQ, an ECAPA/Perceiver/FSQ speaker tokenizer, Qwen2.5-0.5B, a d-vector-conditioned Vocos
/// prenet, and the released four-stage DAC WaveGenerator. Voice cloning uses the official 32-global-token prompt
/// grammar; controllable synthesis asks the LM to generate those global tokens from gender, pitch and speed labels.
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens",
    "https://arxiv.org/abs/2503.01710",
    Year = 2025,
    Authors = "Wang et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, Beta1 = 0.8, Beta2 = 0.9,
    Source = "Wang et al. 2025, Sec. 6.1: BiCodec AdamW with beta1=0.8 and beta2=0.9; the paper reports a "
        + "614.4-second speech batch and approximately 800k steps rather than an example-count batch size.")]
public partial class SparkTTS<T> : TtsModelBase<T>, ICodecTts<T>
{
    private readonly SparkTTSOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private SparkTokenLanguageModel<T>? _languageModel;
    private SparkBiCodec<T>? _biCodec;
    private CodecWaveformGan<T>? _codecTraining;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _discriminatorOptimizer;
    private readonly List<LayerBase<T>> _languageLayers = new();
    private bool _discriminatorTurn;
    private bool _speakerEncoderInitialized;
    private int _codecTrainingSteps;
    private bool _useNativeMode;
    private bool _disposed;

    /// <summary>Creates an ONNX inference wrapper.</summary>
    public SparkTTS(NeuralNetworkArchitecture<T> architecture, string modelPath, SparkTTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new SparkTTSOptions();
        ValidateOptions(_options);
        Options = _options;
        _useNativeMode = false;
        ConfigureBaseProperties();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(_options.ModelPath!, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates the native official-release graph.</summary>
    public SparkTTS(
        NeuralNetworkArchitecture<T> architecture,
        SparkTTSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new SparkTTSOptions();
        ValidateOptions(_options);
        Options = _options;
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                WeightDecay = _options.WeightDecay,
                Beta1 = _options.TrainingComponent == SparkTTSTrainingComponent.LanguageModel ? 0.9 : 0.8,
                Beta2 = _options.TrainingComponent == SparkTTSTrainingComponent.LanguageModel ? 0.96 : 0.9,
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
    /// <inheritdoc />
    public int CodecFrameRate => _options.CodecFrameRate;
    public override int CodecTokenCodebooks => 1;
    public override int CodecTokenVocabulary => _options.CodebookSize;
    protected override int OutputFeatureWidth => _options.VocabSize;

    /// <summary>Whether this instance owns the official Qwen/BiCodec component graph.</summary>
    public bool HasPaperPipeline => _languageModel is not null && _biCodec is not null;

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) => HasPaperPipeline
        ? LayerInputDomain.Indices(_options.VocabSize) : base.GetInputDomain(inputShape);

    /// <summary>Synthesizes with a configured voice prompt, or with the paper's neutral controllable-speaker path.</summary>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        if (!HasPaperPipeline) return PostprocessAudio(Predict(PreprocessText(text)));
        if (Voice?.ReferenceAudio is { } reference)
            return SynthesizeWithPrompt(text, string.Empty, reference);
        return SynthesizeControlled(text, female: true, SparkTTSControlLevel.Moderate, SparkTTSControlLevel.Moderate);
    }

    /// <summary>Runs the released zero-shot voice-cloning prompt format.</summary>
    public Tensor<T> SynthesizeWithPrompt(string text, string promptText, Tensor<T> referenceAudio)
    {
        ThrowIfDisposed();
        RequirePipeline();
        if (string.IsNullOrWhiteSpace(text)) throw new ArgumentException("Target text is required.", nameof(text));
        Validation.Guard.NotNull(referenceAudio);
        using var _ = new NoGradScope<T>();
        var encoded = _biCodec!.Tokenize(referenceAudio);
        var prompt = new List<int> { Special(_options.TaskTtsToken), Special(_options.StartContentToken) };
        prompt.AddRange(TextIds((promptText ?? string.Empty) + text));
        prompt.Add(Special(_options.EndContentToken));
        prompt.Add(Special(_options.StartGlobalToken));
        prompt.AddRange(TokenIds(encoded.Global, _options.GlobalTokenOffset, _options.GlobalCodebookSize));
        prompt.Add(Special(_options.EndGlobalToken));
        if (!string.IsNullOrWhiteSpace(promptText))
        {
            prompt.Add(Special(_options.StartSemanticToken));
            prompt.AddRange(TokenIds(encoded.Semantic, _options.SemanticTokenOffset, _options.CodebookSize));
        }
        else if (_options.ConstrainedSpeechDecoding) prompt.Add(Special(_options.StartSemanticToken));
        var semantic = _languageModel!.GenerateSemantic(prompt, new Random(_options.GenerationSeed));
        return PostprocessAudio(_biCodec.Decode(TensorOf(semantic), encoded.Global));
    }

    /// <summary>Runs the official gender/pitch/speed controllable-speaker prompt and its global-token chain of thought.</summary>
    public Tensor<T> SynthesizeControlled(string text, bool female, SparkTTSControlLevel pitch, SparkTTSControlLevel speed)
    {
        ThrowIfDisposed();
        RequirePipeline();
        if (string.IsNullOrWhiteSpace(text)) throw new ArgumentException("Target text is required.", nameof(text));
        using var _ = new NoGradScope<T>();
        var prompt = new List<int>
        {
            Special(_options.TaskControllableTtsToken), Special(_options.StartContentToken),
        };
        prompt.AddRange(TextIds(text));
        prompt.AddRange(new[]
        {
            Special(_options.EndContentToken), Special(_options.StartStyleLabelToken),
            Special(_options.GenderTokenOffset + (female ? 0 : 1)),
            Special(_options.PitchLabelOffset + (int)pitch), Special(_options.SpeedLabelOffset + (int)speed),
            Special(_options.EndStyleLabelToken),
        });
        var random = new Random(_options.GenerationSeed);
        if (!_options.ConstrainedSpeechDecoding)
        {
            int[] generated = _languageModel!.GenerateAll(prompt, random);
            int[] globals = generated.Where(id => id >= _options.GlobalTokenOffset
                && id < _options.GlobalTokenOffset + _options.GlobalCodebookSize).Select(id => id - _options.GlobalTokenOffset).ToArray();
            int[] semantics = generated.Where(id => id >= _options.SemanticTokenOffset
                && id < _options.SemanticTokenOffset + _options.CodebookSize).Select(id => id - _options.SemanticTokenOffset).ToArray();
            if (globals.Length != _options.GlobalTokenCount || semantics.Length == 0)
                throw new InvalidOperationException("Spark did not generate a complete global/semantic token sequence.");
            return PostprocessAudio(_biCodec!.Decode(TensorOf(semantics), TensorOf(globals)));
        }
        prompt.Add(Special(_options.StartGlobalToken));
        int[] global = _languageModel!.GenerateGlobal(prompt, random);
        prompt.AddRange(global.Select(id => LanguageToken(id, _options.GlobalTokenOffset, _options.GlobalCodebookSize)));
        prompt.Add(Special(_options.EndGlobalToken));
        prompt.Add(Special(_options.StartSemanticToken));
        int[] semantic = _languageModel.GenerateSemantic(prompt, random);
        return PostprocessAudio(_biCodec!.Decode(TensorOf(semantic), TensorOf(global)));
    }

    /// <inheritdoc />
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(audio);
        RequirePipeline();
        return _biCodec!.EncodeSemantic(audio);
    }

    /// <summary>Extracts the 32 released BiCodec global speaker-token ids.</summary>
    public Tensor<T> EncodeGlobalTokens(Tensor<T> referenceAudio)
    {
        ThrowIfDisposed();
        RequirePipeline();
        return _biCodec!.EncodeGlobal(referenceAudio);
    }

    /// <inheritdoc />
    /// <remarks>Without an explicit voice prompt the neutral FSQ code is used. Use the two-argument overload to clone a voice.</remarks>
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        var global = new Tensor<T>(new[] { _options.GlobalTokenCount });
        return DecodeFromTokens(tokens, global);
    }

    /// <summary>Decodes semantic speech tokens with an explicit 32-token speaker prompt.</summary>
    public Tensor<T> DecodeFromTokens(Tensor<T> semanticTokens, Tensor<T> globalTokens)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(semanticTokens);
        RequirePipeline();
        using var _ = new NoGradScope<T>();
        return _biCodec!.Decode(semanticTokens, globalTokens);
    }

    protected override TtsSupervision RequiredSupervision => _options.TrainingComponent == SparkTTSTrainingComponent.BiCodec
        ? TtsSupervision.Recording : TtsSupervision.Text | TtsSupervision.Recording;
    public override bool TrainsOnSampleObjective => true;

    /// <summary>Trains both zero-shot and controllable samples for one annotated recording (paper Eq. 2/3).</summary>
    /// <param name="sample">Text tokens and the corresponding recording.</param>
    /// <param name="female">The recording's gender annotation.</param>
    /// <param name="pitch">Gender-conditioned coarse pitch annotation.</param>
    /// <param name="speed">Coarse syllable-rate annotation.</param>
    /// <param name="pitchMel">Fine pitch value on the Mel scale, as used by the released attribute tokenizer.</param>
    /// <param name="syllablesPerSecond">Fine syllable-rate value.</param>
    /// <remarks>Annotations must come from the dataset; they cannot be inferred from text. The two suffix likelihoods
    /// are normalized per output token and averaged, with no prediction loss on either conditioning prefix.</remarks>
    public T TrainAnnotatedSample(TtsTrainingSample<T> sample, bool female, SparkTTSControlLevel pitch,
        SparkTTSControlLevel speed, int pitchMel, int syllablesPerSecond)
    {
        ThrowIfDisposed(); RequirePipeline();
        if (_optimizer is null || _options.TrainingComponent != SparkTTSTrainingComponent.LanguageModel)
            throw new NotSupportedException("Annotated mixed-objective training requires the language-model stage.");
        if ((int)pitch < 0 || (int)pitch > 4 || (int)speed < 0 || (int)speed > 4)
            throw new ArgumentOutOfRangeException(nameof(pitch));
        Validation.Guard.NotNull(sample);
        EnsureSpeakerEncoderInitialized(sample);
        var annotation = new ControlAnnotation(female, pitch, speed, MathPolyfill.Clamp(pitchMel, 0, 1000),
            MathPolyfill.Clamp(syllablesPerSecond, 0, 10));
        return TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => BuildObjective(sample, annotation), _optimizer);
    }

    private sealed record ControlAnnotation(bool Female, SparkTTSControlLevel Pitch, SparkTTSControlLevel Speed,
        int PitchMel, int SyllablesPerSecond);

    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        RequirePipeline();
        if (_optimizer is null) throw new NotSupportedException("Native optimizer is unavailable.");
        EnsureSpeakerEncoderInitialized(sample);
        if (_options.TrainingComponent == SparkTTSTrainingComponent.BiCodec)
        {
            EnsureCodecTraining();
            _discriminatorTurn = true;
            try
            {
                TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) =>
                {
                    Tensor<T> generated;
                    using (new NoGradScope<T>()) generated = _biCodec!.TrainingForward(sample.Audio!, _codecTrainingSteps < _options.GlobalTokenizerWarmupSteps).Waveform;
                    var real = Engine.TensorNarrow(Engine.Reshape(sample.Audio!, new[] { sample.Audio!.Length }), 0, 0, generated.Length);
                    return _codecTraining!.DiscriminatorLoss(real, generated);
                }, _discriminatorOptimizer!);
            }
            finally { _discriminatorTurn = false; }
        }
        T loss = TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => BuildObjective(sample), _optimizer);
        if (_options.TrainingComponent == SparkTTSTrainingComponent.BiCodec) _codecTrainingSteps++;
        return loss;
    }

    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        RequirePipeline();
        if (_options.TrainingComponent == SparkTTSTrainingComponent.BiCodec) EnsureCodecTraining();
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return BuildObjective(sample)[0];
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
        var random = new Random(_options.Seed ?? _options.GenerationSeed);
        _languageModel = new SparkTokenLanguageModel<T>(Engine, _languageLayers, _options);
        ComponentLayers.AddRange(_languageLayers);
        _biCodec = new SparkBiCodec<T>(Engine, ComponentLayers, _options, random);
        // Finalize the model's layer RNGs before parameters are inspected or original weights are imported.
        // Deferring this until the first training step can reinitialize a frozen DenseLayer after checkpoint loading.
        EnsureLayerRandomSeedsWired();
        if (_options.TrainingComponent == SparkTTSTrainingComponent.BiCodec) EnsureCodecTraining();
    }

    private void EnsureSpeakerEncoderInitialized(TtsTrainingSample<T> sample)
    {
        if (_speakerEncoderInitialized) return;
        var audio = sample.Audio ?? throw new ArgumentException("Spark-TTS training requires Audio.", nameof(sample));
        // ECAPA's first convolution learns its mel input width on its first forward.
        // Resolve that width before the training step captures parameter references.
        using (new NoGradScope<T>()) _biCodec!.EncodeGlobal(audio);
        _speakerEncoderInitialized = true;
    }

    private void EnsureCodecTraining()
    {
        if (_codecTraining is not null) return;
        _codecTraining = new CodecWaveformGan<T>(Engine, _options.SampleRate, _options.TrainingMelWindows, _options.TrainingMelBins,
            _options.DiscriminatorWindows, _options.DiscriminatorBands, _options.DiscriminatorChannels,
            _options.DiscriminatorWidthDivisor, _options.MelLossWeight, _options.FeatureMatchingWeight);
        ComponentLayers.AddRange(_codecTraining.Layers);
        _discriminatorOptimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = _options.LearningRate,
                WeightDecay = _options.WeightDecay, Beta1 = 0.8, Beta2 = 0.9 });
    }

    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (_biCodec is null) return parameters;
        var layers = _options.TrainingComponent == SparkTTSTrainingComponent.LanguageModel
            ? (IReadOnlyList<LayerBase<T>>)_languageLayers
            : _discriminatorTurn ? _codecTraining!.Layers : _biCodec.TrainableLayers;
        var selected = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(layers.Cast<ILayer<T>>().ToList(), -1),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(selected.Contains).ToList();
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (_languageModel is not null) return _languageModel.Logits(IdsOf(input, _options.VocabSize));
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
            Name = _useNativeMode ? "Spark-TTS-0.5B-Native" : "Spark-TTS-ONNX",
            Description = "Spark-TTS Qwen2.5 plus decoupled semantic/global-token BiCodec (Wang et al., 2025)",
            FeatureCount = _options.LLMDim,
        };
        metadata.AdditionalInfo["Architecture"] = "Qwen2.5-0.5B -> BiCodec(Vocos+FVQ, ECAPA+Perceiver+FSQ, WaveGenerator)";
        metadata.AdditionalInfo["SemanticVocabulary"] = _options.CodebookSize;
        metadata.AdditionalInfo["GlobalVocabulary"] = _options.GlobalCodebookSize;
        metadata.AdditionalInfo["GlobalTokenCount"] = _options.GlobalTokenCount;
        metadata.AdditionalInfo["CodecFrameRate"] = _options.CodecFrameRate;
        metadata.AdditionalInfo["TrainingComponent"] = _options.TrainingComponent.ToString();
        metadata.AdditionalInfo["OfficialWeightsAvailable"] = true;
        metadata.AdditionalInfo["Mode"] = _useNativeMode ? "Native" : "ONNX";
        return metadata;
    }

    private Tensor<T> BuildObjective(TtsTrainingSample<T> sample, ControlAnnotation? annotation = null)
    {
        var audio = sample.Audio ?? throw new ArgumentException("Spark-TTS training requires Audio.", nameof(sample));
        if (_options.TrainingComponent == SparkTTSTrainingComponent.BiCodec)
        {
            var output = _biCodec!.TrainingForward(audio, _codecTrainingSteps < _options.GlobalTokenizerWarmupSteps);
            var real = Engine.TensorNarrow(Engine.Reshape(audio, new[] { audio.Length }), 0, 0, output.Waveform.Length);
            return Engine.TensorAdd(Engine.Reshape(output.AuxiliaryLoss, new[] { 1 }), _codecTraining!.GeneratorLoss(real, output.Waveform));
        }
        var encoded = _biCodec!.Tokenize(audio);

        var sequence = new List<int> { Special(_options.TaskTtsToken), Special(_options.StartContentToken) };
        sequence.AddRange(IdsOf(sample.Tokens, _options.VocabSize));
        sequence.Add(Special(_options.EndContentToken));
        sequence.Add(Special(_options.StartGlobalToken));
        sequence.AddRange(TokenIds(encoded.Global, _options.GlobalTokenOffset, _options.GlobalCodebookSize));
        sequence.Add(Special(_options.EndGlobalToken));
        int firstTarget = sequence.Count;
        sequence.Add(Special(_options.StartSemanticToken));
        sequence.AddRange(TokenIds(encoded.Semantic, _options.SemanticTokenOffset, _options.CodebookSize));
        sequence.Add(Special(_options.EndSemanticToken));
        sequence.Add(Special(_options.EosToken));
        var zeroShot = _languageModel!.Loss(sequence, firstTarget);
        if (annotation is null) return zeroShot;
        var control = new List<int> { Special(_options.TaskControllableTtsToken), Special(_options.StartContentToken) };
        control.AddRange(IdsOf(sample.Tokens, _options.VocabSize));
        control.AddRange(new[] { Special(_options.EndContentToken), Special(_options.StartStyleLabelToken),
            Special(_options.GenderTokenOffset + (annotation.Female ? 0 : 1)),
            Special(_options.PitchLabelOffset + (int)annotation.Pitch), Special(_options.SpeedLabelOffset + (int)annotation.Speed),
            Special(_options.EndStyleLabelToken) });
        int controlTarget = control.Count;
        control.AddRange(new[] { Special(_options.StartStylePromptToken), Special(_options.PitchValueOffset + annotation.PitchMel),
            Special(_options.SpeedValueOffset + annotation.SyllablesPerSecond), Special(_options.EndStylePromptToken),
            Special(_options.StartGlobalToken) });
        control.AddRange(TokenIds(encoded.Global, _options.GlobalTokenOffset, _options.GlobalCodebookSize));
        control.Add(Special(_options.EndGlobalToken));
        control.AddRange(sequence.Skip(firstTarget));
        return Engine.TensorMultiplyScalar(Engine.TensorAdd(zeroShot, _languageModel.Loss(control, controlTarget)), NumOps.FromDouble(.5));
    }

    /// <summary>Uses the released multilingual Qwen BPE and Spark's complete added-token vocabulary.</summary>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) throw new ArgumentException("Text is required.", nameof(text));
        int[] ids;
        if (_options.TextSymbols is { } symbols)
        {
            ids = text.Select(c => Array.IndexOf(symbols, c.ToString())).ToArray();
            if (ids.Any(id => id < 0)) throw new ArgumentException("Text contains a character outside TextSymbols.", nameof(text));
        }
        else ids = AiDotNet.Tokenization.Algorithms.QwenTokenizer.Spark.Encode(text).ToArray();
        if (ids.Length > _options.MaxTextLength) throw new ArgumentException("Text exceeds MaxTextLength.", nameof(text));
        foreach (int id in ids) Special(id);
        return TensorOf(ids);
    }

    private int[] TextIds(string text) => IdsOf(PreprocessText(text), _options.VocabSize);

    private int[] TokenIds(Tensor<T> tokens, int preferredOffset, int count)
        => IdsOf(tokens, count).Select(id => LanguageToken(id, preferredOffset, count)).ToArray();

    private int LanguageToken(int raw, int preferredOffset, int count)
    {
        if (raw < 0 || raw >= count) throw new ArgumentOutOfRangeException(nameof(raw));
        return Special(preferredOffset + raw);
    }

    private int Special(int id) => id >= 0 && id < _options.VocabSize ? id
        : throw new ArgumentOutOfRangeException(nameof(id), id, "Token is outside the Spark vocabulary.");

    private Tensor<T> TensorOf(IReadOnlyList<int> ids)
    {
        var tensor = new Tensor<T>(new[] { ids.Count });
        for (int i = 0; i < ids.Count; i++) tensor[i] = NumOps.FromDouble(ids[i]);
        return tensor;
    }

    private int[] IdsOf(Tensor<T> tensor, int vocabulary)
    {
        if (tensor.Length == 0) throw new ArgumentException("Token tensor cannot be empty.", nameof(tensor));
        var ids = new int[tensor.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            double value = NumOps.ToDouble(tensor[i]);
            if (double.IsNaN(value) || value < 0 || value >= vocabulary || value != Math.Truncate(value))
                throw new ArgumentException("Token IDs must be integers inside the configured vocabulary.", nameof(tensor));
            ids[i] = (int)value;
        }
        return ids;
    }

    private void ConfigureBaseProperties()
    {
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.LLMDim;
    }

    private static void ValidateOptions(SparkTTSOptions options)
    {
        if (options.DiscriminatorBands is null || options.DiscriminatorBands.Length < 2
            || options.DiscriminatorBands.Any(edge => double.IsNaN(edge) || double.IsInfinity(edge) || edge < 0 || edge > 1)
            || options.DiscriminatorBands.Zip(options.DiscriminatorBands.Skip(1), (left, right) => left >= right).Any(invalid => invalid))
            throw new ArgumentException("Discriminator bands must be strictly increasing finite edges in [0, 1].", nameof(options));
        if (options.SampleRate <= 0 || options.CodecFrameRate <= 0 || options.SampleRate % options.CodecFrameRate != 0)
            throw new ArgumentException("SampleRate must be a positive integer multiple of CodecFrameRate.", nameof(options));
        if (options.NumCodebooks != 1 || options.CodebookSize < 2 || options.GlobalCodebookSize < 2
            || options.GlobalTokenCount <= 0 || options.VocabSize < Math.Max(options.CodebookSize, options.GlobalCodebookSize))
            throw new ArgumentException("Spark uses one semantic stream and valid semantic/global vocabularies.", nameof(options));
        if (options.LLMDim <= 0 || options.NumHeads <= 0 || options.LLMDim % options.NumHeads != 0
            || options.NumHeads % options.KeyValueHeads != 0 || options.NumLLMLayers <= 0 || options.LmIntermediateDim <= 0)
            throw new ArgumentException("Qwen dimensions must be positive and compatible with grouped-query attention.", nameof(options));
        if (options.Wav2VecHiddenDim <= 0 || options.Wav2VecHeads <= 0
            || options.Wav2VecHiddenDim % options.Wav2VecHeads != 0 || options.Wav2VecLayers <= 0)
            throw new ArgumentException("XLSR dimensions must be positive and heads must divide its width.", nameof(options));
        if (options.Wav2VecMixedLayers.Length == 0 || options.FsqLevels.Length == 0
            || options.FsqLevels.Aggregate(1, (product, level) => product * level) != options.GlobalCodebookSize)
            throw new ArgumentException("FSQ levels must multiply to GlobalCodebookSize and XLSR needs mixed layers.", nameof(options));
        if (options.SpeakerEcapaChannels <= 0 || options.SpeakerEcapaChannels % 8 != 0
            || options.SpeakerLatentDim <= 0 || options.PerceiverHeads <= 0 || options.PerceiverDepth <= 0)
            throw new ArgumentException("The ECAPA/Perceiver speaker tokenizer dimensions are invalid.", nameof(options));
        if (options.WaveDecoderRates.Length == 0 || options.WaveDecoderRates.Length != options.WaveDecoderKernels.Length
            || options.WaveDecoderRates.Aggregate(1, (product, rate) => product * rate) != options.SampleRate / options.CodecFrameRate)
            throw new ArgumentException("WaveGenerator rates/kernels must match and their rate product must equal the codec hop.", nameof(options));
    }

    private void RequirePipeline()
    {
        if (!HasPaperPipeline)
            throw new NotSupportedException("This operation requires the native Spark-TTS Qwen/BiCodec pipeline.");
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SparkTTS<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
