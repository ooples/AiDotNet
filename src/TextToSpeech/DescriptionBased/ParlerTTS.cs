using AiDotNet.Attributes;
using AiDotNet.Audio.Effects;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.Tokenization.Specialized;

namespace AiDotNet.TextToSpeech.DescriptionBased;

/// <summary>
/// Parler-TTS: speech in the voice a plain-language description gives (Lyth &amp; King 2024), as the open reproduction
/// huggingface/parler-tts builds it.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The paper's model (§3): a decoder-only Transformer predicts the codes of DAC (44.1 kHz, nine codebooks) in MusicGen's
/// delay pattern, codebook <c>q</c> delayed by <c>q</c> steps. The transcript's token embeddings are prepended to the
/// codes, and the description is read by cross-attention from a frozen Flan-T5 encoder. The rest follows the
/// reproduction (<c>ParlerTTSForConditionalGeneration</c>): per-codebook code embeddings summed, sinusoidal positions,
/// pre-norm layers with bias-free projections and the exact GELU, a bias-free head per codebook, the cross-entropy of
/// each codebook averaged over its positions and the codebooks, and generation that lets codebook <c>q</c> end only after
/// codebook <c>q − 1</c> has.
/// </para>
/// <para>
/// Both texts are tokenized with T5's SentencePiece vocabulary, the transcript into the model's input
/// (<see cref="TtsModelBase{T}.Synthesize(string)"/>) and the description into the voice (<see cref="CreateVoice"/>).
/// Training reads the transcript (<see cref="TtsTrainingSample{T}.Tokens"/>), the description
/// (<see cref="TtsTrainingSample{T}.Description"/>) and the recording's DAC codes
/// (<see cref="TtsTrainingSample{T}.CodecTokens"/>, or the recording, which the model encodes).
/// </para>
/// <para><b>For Beginners:</b> Instead of a recording of a speaker, you describe the voice you want in plain English
/// ("A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch") and the
/// model speaks the text in such a voice.</para>
/// <example>
/// <code>
/// var model = new ParlerTTS&lt;float&gt;(new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional,
///     NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1));
/// model.LoadReferenceWeights("parler-tts-mini-v1.safetensors");
/// model.Codec.LoadPretrainedWeights("dac_44khz.safetensors");
/// model.Voice = model.CreateVoice("A female speaker delivers a slightly expressive speech with a moderate pitch.");
/// var audio = model.Synthesize("Hey, how are you doing today?");
/// </code>
/// </example>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Natural Language Guidance of High-Fidelity Text-to-Speech with Synthetic Annotations",
    "https://arxiv.org/abs/2402.01912",
    Year = 2024,
    Authors = "Lyth and King"
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 9.5e-4, Beta1 = 0.9, Beta2 = 0.99, Epsilon = 1e-8, WeightDecay = 0.01,
    MaxGradientNorm = 1.0, Schedule = LearningRateSchedulerType.LinearWarmup, WarmupSteps = 20_000,
    PostWarmupDecay = LinearWarmupScheduler.DecayMode.Constant, Provenance = RecipeProvenance.NamedReproduction,
    Source = "The paper states no optimizer settings; huggingface/parler-tts trains parler-tts-mini-v1 with "
             + "helpers/training_configs/starting_point_v1.json: torch.optim.AdamW at 9.5e-4 with betas (0.9, 0.99), "
             + "weight decay 0.01 and the Trainer's epsilon 1e-8, the constant_with_warmup schedule over 20,000 steps, and "
             + "the default max_grad_norm 1.0.")]
public partial class ParlerTTS<T> : TtsModelBase<T>, ICodecTts<T>
{
    private readonly ParlerTTSOptions _options;
    private readonly bool _useNativeMode;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _suppliedOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly Random _trainingRandom;
    private readonly T5TextTokenizer _tokenizer = new();
    private bool _disposed;

    private ParlerCore<T>? _core;
    private T5Seq2Seq<T>? _encoder;
    private readonly List<LayerBase<T>> _decoderLayers = new();
    private readonly List<LayerBase<T>> _encoderLayers = new();
    private AudioCodecLayer<T>? _codec;

    /// <summary>Creates an ONNX-backed Parler-TTS for inference.</summary>
    public ParlerTTS(NeuralNetworkArchitecture<T> architecture, string modelPath, ParlerTTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new ParlerTTSOptions();
        Options = _options;
        _useNativeMode = false;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new AiDotNet.Onnx.OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable Parler-TTS.</summary>
    public ParlerTTS(NeuralNetworkArchitecture<T> architecture, ParlerTTSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new ParlerTTSOptions();
        Options = _options;
        _useNativeMode = true;
        _suppliedOptimizer = optimizer;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        ConfigureBase();
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        var o = _options;
        base.SampleRate = o.SampleRate;
        base.HopSize = o.HopSize;
        base.HiddenDim = o.HiddenDim;
        o.VocabSize = o.PromptVocabulary;
        if (o.AudioVocabulary < o.CodebookSize + 2)
            throw new ArgumentException($"Parler-TTS's {o.AudioVocabulary} logits per codebook must hold the {o.CodebookSize} codes, " +
                "the end token and the start token.");
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    public int NumCodebooks => _options.NumCodebooks;

    /// <inheritdoc />
    public int CodebookSize => _options.CodebookSize;

    /// <inheritdoc />
    public int CodecFrameRate => (int)Math.Round((double)_options.SampleRate / _options.HopSize);

    /// <summary>The DAC whose codes the model predicts (load its released weights with
    /// <see cref="DAC{T}.LoadPretrainedWeights"/>).</summary>
    public DAC<T> Codec => (DAC<T>?)_codec?.Codec ?? throw new InvalidOperationException("Parler-TTS's codec exists in native mode only.");

    /// <summary>The end token every codebook closes with (the codebook size).</summary>
    private int EndToken => _options.CodebookSize;

    /// <summary>The start token every codebook opens with (one past the end token).</summary>
    private int BeginToken => _options.CodebookSize + 1;

    /// <inheritdoc />
    protected override TtsSupervision RequiredSupervision => TtsSupervision.CodecTokens | TtsSupervision.Description;

    /// <inheritdoc />
    protected override TtsSupervision RequiredVoice => TtsSupervision.Description;

    /// <inheritdoc />
    public override int CodecTokenCodebooks => _options.NumCodebooks;

    /// <inheritdoc />
    public override int CodecTokenVocabulary => _options.CodebookSize;

    private bool HasPaperLayers => _core is not null;

    private ParlerCore<T> Core => _core ?? throw new InvalidOperationException("Parler-TTS's networks exist in native mode only.");

    private T5Seq2Seq<T> DescriptionEncoder =>
        _encoder ?? throw new InvalidOperationException("Parler-TTS's description encoder exists in native mode only.");

    // The networks' layers, for tests.
    internal IReadOnlyList<LayerBase<T>> DecoderLayers => _decoderLayers;
    internal IReadOnlyList<LayerBase<T>> DescriptionEncoderLayers => _encoderLayers;
    internal ParlerCore<T>? PaperCore => _core;

    /// <inheritdoc />
    /// <remarks>The transcript's T5 token ids.</remarks>
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperLayers ? LayerInputDomain.Indices(_options.PromptVocabulary) : base.GetInputDomain(inputShape);

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            AddUnsplitLayers(Architecture.Layers);
            return;
        }

        var o = _options;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? o.SamplingSeed);
        int keyValueHeads = o.KeyValueHeads > 0 ? o.KeyValueHeads : o.NumHeads;
        _core = new ParlerCore<T>(Engine, _decoderLayers, new ParlerConfiguration(
            AudioVocabulary: o.AudioVocabulary, Codebooks: o.NumCodebooks, ModelDim: o.HiddenDim, Heads: o.NumHeads,
            KeyValueHeads: keyValueHeads,
            CrossKeyValueHeads: o.CrossAttentionKeyValueHeads > 0 ? o.CrossAttentionKeyValueHeads : keyValueHeads,
            Layers: o.NumDecoderLayers, FeedForwardDim: o.FeedForwardDim, PromptVocabulary: o.PromptVocabulary,
            EncoderDim: o.TextEncoderDim, Dropout: o.DropoutRate, AttentionDropout: o.AttentionDropout,
            ActivationDropout: o.ActivationDropout, LayerDrop: o.LayerDrop, RotaryEmbeddings: o.RotaryEmbeddings,
            RopeTheta: o.RopeTheta, PromptCrossAttention: o.PromptCrossAttention, BeginToken: BeginToken, EndToken: EndToken,
            CodebookWeights: o.CodebookWeights));
        _core.InitializeLikeReference(random, o.InitializerFactor);

        // Flan-T5's encoder (T5 v1.1: gated GELU, untied), pretrained and frozen in practice.
        _encoder = new T5Seq2Seq<T>(Engine, _encoderLayers, new T5Configuration(
            VocabularySize: o.TextEncoderVocabulary, ModelDim: o.TextEncoderDim, FeedForwardDim: o.TextEncoderFeedForwardDim,
            KeyValueDim: o.TextEncoderKeyValueDim, Heads: o.TextEncoderHeads, EncoderLayers: o.TextEncoderLayers,
            DecoderLayers: 0, Dropout: o.TextEncoderDropout, EndTokenId: T5TextTokenizer.EosId, GatedGelu: true,
            TieWordEmbeddings: false, PadTokenId: T5TextTokenizer.PadId));
        _encoder.InitializeLikeHuggingFace(random);

        // The codec (pretrained, frozen) only encodes and decodes.
        var codecOptions = (DACOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(o.Codec);
        codecOptions.IncludeDiscriminators = false;
        var codec = new DAC<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1) { RandomSeed = Architecture.RandomSeed }, codecOptions);
        int codecRate = ((IAudioCodec<T>)codec).SampleRate;
        if (codec.HopLength != o.HopSize || codecRate != o.SampleRate)
            throw new ArgumentException($"Parler-TTS's {o.SampleRate} Hz audio and {o.HopSize}-sample frames must be its codec's " +
                $"({codecRate} Hz, {codec.HopLength}-sample hop); set SampleRate and HopSize to match the codec options.");
        if (codec.CodebookSize != o.CodebookSize)
            throw new ArgumentException($"Parler-TTS's {o.CodebookSize} codes per codebook must be the codec's {codec.CodebookSize}.");
        int codebooks = codec.QuantizersForBandwidth(codecOptions.TargetBandwidthKbps);
        if (codebooks != o.NumCodebooks)
            throw new ArgumentException($"Parler-TTS models {o.NumCodebooks} codebooks; the codec's bandwidth gives {codebooks}.");
        _codec = new AudioCodecLayer<T>(codec);

        AddEncoderDecoderLayers(_decoderLayers.Cast<ILayer<T>>().ToList(), Array.Empty<ILayer<T>>());
        ComponentLayers.AddRange(_encoderLayers);
        ComponentLayers.Add(_codec);
    }

    // ---------------------------------------------------------------- front end

    /// <summary>The T5 token ids of <paramref name="text"/>, ending with <c>&lt;/s&gt;</c>, as the reference's tokenizer
    /// gives them.</summary>
    public int[] Tokenize(string text) => _tokenizer.Encode(text ?? throw new ArgumentNullException(nameof(text)));

    /// <summary>A voice for <see cref="TtsModelBase{T}.Voice"/>: a plain-language description of the speaker, the
    /// recording conditions and the delivery.</summary>
    public TtsVoice<T> CreateVoice(string description)
    {
        if (string.IsNullOrWhiteSpace(description)) throw new ArgumentException("Describe the voice.", nameof(description));
        return new TtsVoice<T> { Description = ToTensor(Tokenize(description)) };
    }

    /// <inheritdoc />
    /// <remarks>The transcript's T5 token ids (at most <see cref="MaxTextLength"/>).</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        var ids = Tokenize(text);
        if (ids.Length > _options.MaxTextLength) ids = ids.Take(_options.MaxTextLength - 1).Append(T5TextTokenizer.EosId).ToArray();
        return ToTensor(ids);
    }

    private Tensor<T> ToTensor(IReadOnlyList<int> ids)
    {
        var tensor = new Tensor<T>(new[] { ids.Count });
        for (int i = 0; i < ids.Count; i++) tensor[i] = NumOps.FromDouble(ids[i]);
        return tensor;
    }

    private int[] ToIds(Tensor<T> tensor, int vocabulary, string name)
    {
        if (tensor.Length == 0) throw new ArgumentException("Expected at least one token id.", name);
        var ids = new int[tensor.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = (int)Math.Round(NumOps.ToDouble(tensor[i]));
            if (ids[i] < 0 || ids[i] >= vocabulary)
                throw new ArgumentOutOfRangeException(name, $"{GetType().Name}'s token ids here lie in [0, {vocabulary}).");
        }
        return ids;
    }

    /// <summary>The codes <c>[frames, codebooks]</c> of a codec-token tensor or, without one, of a recording.</summary>
    private int[,] CodesOf(Tensor<T>? tokens, Tensor<T>? audio, string name)
    {
        int codebooks = _options.NumCodebooks;
        if (tokens is not null)
        {
            if (tokens.Rank != 2 || tokens.Shape[1] != codebooks)
                throw new ArgumentException($"Expected codec tokens [frames, {codebooks}].", name);
            var codes = new int[tokens.Shape[0], codebooks];
            for (int f = 0; f < tokens.Shape[0]; f++)
                for (int q = 0; q < codebooks; q++)
                {
                    codes[f, q] = (int)Math.Round(NumOps.ToDouble(tokens[f, q]));
                    if (codes[f, q] < 0 || codes[f, q] >= _options.CodebookSize)
                        throw new ArgumentOutOfRangeException(name, $"Codec tokens lie in [0, {_options.CodebookSize}).");
                }
            return codes;
        }
        if (audio is null) throw new ArgumentException($"{GetType().Name} trains on codec tokens or the recording.", name);
        return FramesFirst(Codec.Encode(audio));
    }

    private static int[,] FramesFirst(int[,] codes)
    {
        int codebooks = codes.GetLength(0), frames = codes.GetLength(1);
        var transposed = new int[frames, codebooks];
        for (int q = 0; q < codebooks; q++)
            for (int f = 0; f < frames; f++) transposed[f, q] = codes[q, f];
        return transposed;
    }

    /// <summary>The decoder's memory for a description: Flan-T5's states (frozen: outside the tape), projected, and the
    /// transcript's embeddings when it is cross-attended.</summary>
    private Tensor<T> Memory(int[] description, int[] transcript, bool training, Random random)
    {
        var ids = ToTensor(description);
        Tensor<T> states;
        if (_options.FreezeTextEncoder)
        {
            using (new NoGradScope<T>()) states = DescriptionEncoder.Encode(ids, training, random);
        }
        else
        {
            states = DescriptionEncoder.Encode(ids, training, random);
        }
        return Core.Memory(states, transcript);
    }

    // ---------------------------------------------------------------- training

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // A token/mel pair does not carry the supervision this model trains on; Train(TtsTrainingSample) does.
        ThrowIfTokenMelTrainingUnsupported();
        return false;
    }

    /// <inheritdoc />
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        var objective = BuildObjective(sample, _trainingRandom, training: true);
        return TrainWithCustomObjective(sample.Tokens, sample.Tokens, objective, Optimizer());
    }

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        var objective = BuildObjective(sample, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed), training: false);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return objective(sample.Tokens, sample.Tokens)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>The loss and each codebook's loss on one utterance (the reference's <c>loss</c> and
    /// <c>per_codebook_losses</c>), without dropout and without updating the model.</summary>
    internal (T Loss, T[] PerCodebook) EvaluateLosses(int[] transcript, int[] description, int[,] codes)
    {
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
            var (loss, perCodebook) = Core.Loss(codes, transcript, Memory(description, transcript, false, random), false, random);
            return (loss[0], perCodebook.Select(l => l[0]).ToArray());
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>The logits of each codebook <c>[length, vocabulary]</c> for decoder input codes
    /// <c>[codebooks, length]</c>, without dropout.</summary>
    internal IReadOnlyList<Tensor<T>> EvaluateLogits(int[] transcript, int[] description, int[,] decoderInput)
    {
        using var _ = new NoGradScope<T>();
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        return Core.Logits(decoderInput, transcript, Memory(description, transcript, false, random), false, random);
    }

    private Func<Tensor<T>, Tensor<T>, Tensor<T>> BuildObjective(TtsTrainingSample<T> sample, Random random, bool training)
    {
        Guard.NotNull(sample);
        if (!HasPaperLayers)
            throw new NotSupportedException("The paper objective needs the paper's layers; this model was built from caller-supplied layers.");
        var transcript = ToIds(sample.Tokens, _options.PromptVocabulary, nameof(sample));
        var description = ToIds(sample.Description ?? throw new ArgumentException("Parler-TTS trains on a description; set Description.", nameof(sample)),
            _options.TextEncoderVocabulary, nameof(sample));
        var codes = CodesOf(sample.CodecTokens, sample.Audio, nameof(sample));
        if (codes.GetLength(0) < 1)
            throw new ArgumentException($"{GetType().Name} trains on at least one frame.", nameof(sample));
        return (_, _) => Core.Loss(codes, transcript, Memory(description, transcript, training, random), training, random).Loss;
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? Optimizer()
    {
        if (_suppliedOptimizer is not null)
            return _suppliedOptimizer;
        if (_optimizer is null)
        {
            var o = _options;
            int warmup = o.WarmupSteps;
            // constant_with_warmup: linear from 0 over the warm-up, then constant.
            var schedule = new LambdaLRScheduler(o.LearningRate, step => step < warmup ? (double)step / Math.Max(1, warmup) : 1.0);
            var options = new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = o.LearningRate,
                Beta1 = o.Beta1,
                Beta2 = o.Beta2,
                Epsilon = o.Epsilon,
                WeightDecay = o.WeightDecay,
                EnableGradientClipping = o.MaxGradientNorm > 0,
                MaxGradientNorm = o.MaxGradientNorm > 0 ? o.MaxGradientNorm : 1.0,
                LearningRateScheduler = schedule,
            };
            _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, options));
        }
        return _optimizer;
    }

    /// <inheritdoc />
    /// <remarks>The decoder learns; the description encoder learns too only when it is not frozen.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var layers = _options.FreezeTextEncoder ? _decoderLayers : _decoderLayers.Concat(_encoderLayers).ToList();
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(layers.Cast<ILayer<T>>().ToList(), -1),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    /// <remarks>
    /// <paramref name="input"/> is the transcript's T5 token ids (<see cref="PreprocessText"/>), spoken in the voice
    /// <see cref="TtsVoice{T}.Description"/> describes; the output is the 44.1 kHz waveform <c>[samples]</c> (one zero
    /// sample when no frame holds codes in every codebook, as the reference returns).
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        if (!HasPaperLayers) return base.PredictCore(input);
        var codes = GenerateCodes(input);
        if (codes.GetLength(0) == 0) return new Tensor<T>(new[] { 1 });
        var audio = Codec.Decode(ByCodebook(codes));
        return audio.Reshape(new[] { audio.Length });
    }

    /// <summary>The codes <c>[frames, codebooks]</c> generated for the transcript ids <paramref name="input"/> in the
    /// voice's description.</summary>
    internal int[,] GenerateCodes(Tensor<T> input)
    {
        var voice = RequireVoice();
        var o = _options;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var transcript = ToIds(input, o.PromptVocabulary, nameof(input));
            var description = ToIds(voice.Description!, o.TextEncoderVocabulary, nameof(voice));
            var memory = Memory(description, transcript, training: false, random);
            return Core.Generate(transcript, memory, o.MaxLength, o.MinNewTokens, o.DoSample, o.Temperature, o.TopK, o.CodebookSize, random);
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    private static int[,] ByCodebook(int[,] codes)
    {
        int frames = codes.GetLength(0), codebooks = codes.GetLength(1);
        var transposed = new int[codebooks, frames];
        for (int f = 0; f < frames; f++)
            for (int q = 0; q < codebooks; q++) transposed[q, f] = codes[f, q];
        return transposed;
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    // ---------------------------------------------------------------- ICodecTts

    /// <inheritdoc />
    /// <remarks>DAC's codes of <paramref name="audio"/>, <c>[frames, codebooks]</c>.</remarks>
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        var codes = FramesFirst(Codec.Encode(audio ?? throw new ArgumentNullException(nameof(audio))));
        var tensor = new Tensor<T>(new[] { codes.GetLength(0), codes.GetLength(1) });
        for (int f = 0; f < codes.GetLength(0); f++)
            for (int q = 0; q < codes.GetLength(1); q++) tensor[f, q] = NumOps.FromDouble(codes[f, q]);
        return tensor;
    }

    /// <inheritdoc />
    /// <remarks>The waveform of codes <c>[frames, codebooks]</c>.</remarks>
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        var codes = CodesOf(tokens ?? throw new ArgumentNullException(nameof(tokens)), null, nameof(tokens));
        var audio = Codec.Decode(ByCodebook(codes));
        return audio.Reshape(new[] { audio.Length });
    }

    // ---------------------------------------------------------------- pretrained weights

    /// <summary>
    /// Loads a <c>ParlerTTSForConditionalGeneration</c> state dict (safetensors or <c>.pt</c>): the decoder, the
    /// transcript embedding, the projection and the Flan-T5 encoder (<c>text_encoder.*</c>). The codec's weights load
    /// separately (<see cref="Codec"/>). Build the model with the checkpoint's configuration.
    /// </summary>
    public void LoadReferenceWeights(string path)
    {
        var file = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
        LoadReferenceWeights((name, shape) =>
        {
            if (!file.TryGetValue(name, out var tensor))
                throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
            var actual = tensor.Shape.ToArray();
            if (!actual.SequenceEqual(shape))
                throw new InvalidDataException($"'{name}' is [{string.Join(", ", actual)}] in the checkpoint but [{string.Join(", ", shape)}] " +
                    "in this model; build the model with the checkpoint's configuration.");
            return tensor.ToVector().Select(v => (double)v).ToArray();
        }, file.ContainsKey);
    }

    /// <summary>Loads the weights from a reader of named tensors.</summary>
    internal void LoadReferenceWeights(Func<string, int[], double[]> read, Func<string, bool> contains)
    {
        Core.LoadTorchWeights(read, contains);
        DescriptionEncoder.LoadHuggingFaceEncoderWeights((name, shape) => read("text_encoder." + name, shape), _options.TextEncoderVocabulary);
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "ParlerTTS-Native" : "ParlerTTS-ONNX",
            Description = "Parler-TTS: Natural Language Guidance of High-Fidelity Text-to-Speech (Lyth & King, 2024)",
            FeatureCount = _options.HiddenDim,
            Complexity = _options.NumDecoderLayers + _options.TextEncoderLayers,
        };
        metadata.AdditionalInfo["Architecture"] = "ParlerTTS";
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(ParlerTTS<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        if (disposing) (_codec?.Codec as IDisposable)?.Dispose();
        base.Dispose(disposing);
    }
}
