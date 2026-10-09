using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Which of a VALL-E-family model's two codec language models a training step updates.</summary>
public enum VallETrainingStage
{
    /// <summary>The autoregressive model of the first codebook.</summary>
    AutoRegressive,

    /// <summary>The non-autoregressive model of the other codebooks.</summary>
    NonAutoRegressive,
}

/// <summary>
/// What the models of the VALL-E family share: EnCodec, the AR and NAR codec language models (<see cref="VallECore{T}"/>),
/// a phoneme vocabulary, training each model on its own, and prompted synthesis.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// A model supplies its phoneme table and front end, how its NAR model is prompted in training, and (for a multilingual
/// model) its languages; this base owns the codec, the vocabulary <c>[&lt;pad&gt;, &lt;bos&gt;, &lt;eos&gt;] +
/// sorted(table)</c> (the reference's <c>TextTokenCollater</c>), the optimizers, synthesis and the weights.
/// </remarks>
public abstract partial class VallEModelBase<T> : TtsModelBase<T>, ICodecTts<T>
{
    /// <summary>The frame tokens.</summary>
    protected const int PadToken = 0, BeginToken = 1, EndToken = 2;

    private readonly VALLEOptions _options;
    private readonly bool _useNativeMode;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _suppliedOptimizer;
    private readonly Dictionary<VallETrainingStage, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _stageOptimizers = new();
    private readonly Random _trainingRandom;
    private bool _disposed;

    private readonly Dictionary<string, int> _tokenIds = new(StringComparer.Ordinal);
    private readonly string[] _tokens;
    private readonly int _wordSeparator;

    private VallECore<T>? _core;
    private readonly List<LayerBase<T>> _arLayers = new();
    private readonly List<LayerBase<T>> _narLayers = new();
    private AudioCodecLayer<T>? _codec;

    /// <summary>Creates an ONNX-backed model for inference.</summary>
    protected VallEModelBase(NeuralNetworkArchitecture<T> architecture, string modelPath, VALLEOptions options)
        : base(architecture)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _useNativeMode = false;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        (_tokens, _wordSeparator) = BuildVocabulary(_tokenIds);
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new AiDotNet.Onnx.OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable model.</summary>
    protected VallEModelBase(NeuralNetworkArchitecture<T> architecture, VALLEOptions options,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer)
        : base(architecture)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _useNativeMode = true;
        _suppliedOptimizer = optimizer;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        (_tokens, _wordSeparator) = BuildVocabulary(_tokenIds);
        ConfigureBase();
        InitializeLayers();
    }

    /// <summary>The model's options.</summary>
    protected VALLEOptions Settings => _options;

    /// <summary>The model's phoneme table (sorted into the vocabulary after the three frame tokens).</summary>
    protected abstract IReadOnlyList<string> PhonemeTable { get; }

    /// <summary>Rows of the language-ID table (0 for a monolingual model).</summary>
    protected virtual int LanguageCount => 0;

    /// <summary>Where the language embedding is added (multilingual models).</summary>
    protected virtual VallELanguagePlacement LanguagePlacement => VallELanguagePlacement.AcousticTokens;

    /// <summary>Whether a symbol outside the table is an error (VALL-E's reference asserts) rather than skipped.</summary>
    protected virtual bool RejectsUnknownSymbols => true;

    /// <summary>The phoneme symbols of <paramref name="text"/>.</summary>
    protected abstract IReadOnlyList<string> PhonemizeText(string text);

    /// <summary>The paper component name of each training stage's optimizer.</summary>
    protected virtual string OptimizerComponent(VallETrainingStage stage) =>
        stage == VallETrainingStage.AutoRegressive ? "autoregressive" : "non-autoregressive";

    private void ConfigureBase()
    {
        base.SampleRate = _options.SampleRate;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        _options.VocabSize = _tokens.Length;
        if (_options.VocabSize > _options.TextTokens)
            throw new ArgumentException($"{GetType().Name}'s {_options.VocabSize} phoneme tokens need TextTokens ≥ {_options.VocabSize}.");
    }

    private (string[] Tokens, int WordSeparator) BuildVocabulary(Dictionary<string, int> ids)
    {
        var symbols = PhonemeTable.Distinct(StringComparer.Ordinal).ToList();
        symbols.Sort(StringComparer.Ordinal);                      // Python's sorted(): code-point order
        var tokens = new List<string> { "<pad>", "<bos>", "<eos>" };
        tokens.AddRange(symbols);
        for (int i = 0; i < tokens.Count; i++) ids[tokens[i]] = i;
        return (tokens.ToArray(), ids["_"]);
    }

    /// <summary>The vocabulary, in id order.</summary>
    protected IReadOnlyList<string> Tokens => _tokens;

    int ITtsModel<T>.SampleRate => _options.SampleRate;

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    public int NumCodebooks => _options.NumCodebooks;

    /// <inheritdoc />
    public int CodebookSize => _options.CodebookSize;

    /// <inheritdoc />
    public int CodecFrameRate => _options.SampleRate / _options.HopSize;

    /// <summary>The network a training step updates.</summary>
    public VallETrainingStage CurrentStage { get; set; } = VallETrainingStage.AutoRegressive;

    /// <summary>The codec whose codes the model predicts.</summary>
    public EnCodec<T> Codec => (EnCodec<T>?)_codec?.Codec ?? throw new InvalidOperationException($"{GetType().Name}'s codec exists in native mode only.");

    /// <inheritdoc />
    protected override TtsSupervision RequiredSupervision => TtsSupervision.CodecTokens;

    /// <inheritdoc />
    protected override TtsSupervision RequiredVoice => TtsSupervision.ReferenceRecording;

    /// <inheritdoc />
    public override int CodecTokenCodebooks => _options.NumCodebooks;

    /// <inheritdoc />
    public override int CodecTokenVocabulary => _options.CodebookSize;

    /// <summary>Whether the model was built with its paper layers (not caller-supplied ones).</summary>
    protected bool HasPaperLayers => _core is not null;

    /// <inheritdoc />
    /// <remarks>The phoneme ids: the vocabulary, not the embedding table (whose rows can exceed it, as the reference's
    /// <c>NUM_TEXT_TOKENS</c> does).</remarks>
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperLayers ? LayerInputDomain.Indices(_options.VocabSize) : base.GetInputDomain(inputShape);

    // The networks' layers, for tests that check a stage updates its own network only.
    internal IReadOnlyList<LayerBase<T>> AutoRegressiveLayers => _arLayers;
    internal IReadOnlyList<LayerBase<T>> NonAutoRegressiveLayers => _narLayers;
    internal VallECore<T>? Core => _core;

    /// <summary>The core, which exists in native mode.</summary>
    private protected VallECore<T> PaperCore => _core ?? throw new InvalidOperationException($"{GetType().Name}'s networks exist in native mode only.");

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
        if (o.NumEncoderLayers != o.NumDecoderLayers)
            throw new ArgumentException($"{GetType().Name}'s AR and NAR models have the same depth; set NumEncoderLayers = NumDecoderLayers.");
        _core = new VallECore<T>(Engine, _arLayers, _narLayers, new VallEConfiguration(
            TextTokens: o.TextTokens, AudioTokens: o.CodebookSize, Codebooks: o.NumCodebooks, ModelDim: o.HiddenDim,
            Heads: o.NumHeads, Layers: o.NumDecoderLayers, FeedForwardDim: o.FeedForwardDim, Dropout: o.DropoutRate,
            Languages: LanguageCount, LanguagePlacement: LanguagePlacement));
        _core.InitializeLikePyTorch(AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? o.SamplingSeed));

        // The codec (pretrained, frozen) only encodes and decodes.
        var codecOptions = (EnCodecOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(o.Codec);
        codecOptions.IncludeDiscriminators = false;
        var codec = new EnCodec<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1) { RandomSeed = Architecture.RandomSeed }, codecOptions);
        int codecRate = ((IAudioCodec<T>)codec).SampleRate;
        if (codec.HopLength != o.HopSize || codecRate != o.SampleRate)
            throw new ArgumentException($"{GetType().Name}'s {o.SampleRate} Hz audio and {o.HopSize}-sample frames must be its codec's " +
                $"({codecRate} Hz, {codec.HopLength}-sample hop); set SampleRate and HopSize to match the codec options.");
        if (codec.CodebookSize != o.CodebookSize)
            throw new ArgumentException($"{GetType().Name}'s {o.CodebookSize} codes per codebook must be the codec's {codec.CodebookSize}.");
        int codebooks = codec.QuantizersForBandwidth(codecOptions.TargetBandwidthKbps);
        if (codebooks != o.NumCodebooks)
            throw new ArgumentException($"{GetType().Name} models {o.NumCodebooks} codebooks; the codec's bandwidth gives {codebooks}.");
        _codec = new AudioCodecLayer<T>(codec);

        AddEncoderDecoderLayers(_arLayers.Cast<ILayer<T>>().ToList(), Array.Empty<ILayer<T>>());
        ComponentLayers.AddRange(_narLayers);
        ComponentLayers.Add(_codec);
    }

    // ---------------------------------------------------------------- vocabulary and front end

    /// <summary>The token id of each phoneme symbol, without the <c>&lt;bos&gt; … &lt;eos&gt;</c> frame.</summary>
    public int[] EncodePhonemes(IEnumerable<string> symbols)
    {
        var ids = new List<int>();
        foreach (var symbol in symbols)
        {
            if (_tokenIds.TryGetValue(symbol, out int id) && id >= 3)
                ids.Add(id);
            else if (RejectsUnknownSymbols)
                throw new ArgumentException($"'{symbol}' is not in {GetType().Name}'s phoneme table.", nameof(symbols));
        }
        return ids.ToArray();
    }

    /// <summary>A voice for <see cref="TtsModelBase{T}.Voice"/>: a prompt recording at the model's sample rate (about 3
    /// seconds) and its transcript.</summary>
    public virtual TtsVoice<T> CreateVoice(Tensor<T> recording, string transcript)
    {
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        var ids = EncodePhonemes(PhonemizeText(transcript ?? throw new ArgumentNullException(nameof(transcript))));
        return new TtsVoice<T> { ReferenceAudio = recording, ReferenceTokens = ToTensor(ids) };
    }

    /// <summary>A tensor of token ids.</summary>
    protected Tensor<T> ToTensor(IReadOnlyList<int> ids)
    {
        var tensor = new Tensor<T>(new[] { ids.Count });
        for (int i = 0; i < ids.Count; i++) tensor[i] = NumOps.FromDouble(ids[i]);
        return tensor;
    }

    /// <summary>The token ids of a tensor, checked against the vocabulary.</summary>
    protected int[] ToIds(Tensor<T> tensor, string name)
    {
        var ids = new int[tensor.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = (int)Math.Round(NumOps.ToDouble(tensor[i]));
            if (ids[i] < 0 || ids[i] >= _options.VocabSize)
                throw new ArgumentOutOfRangeException(name, $"{GetType().Name}'s token ids lie in [0, {_options.VocabSize}).");
        }
        return ids;
    }

    /// <inheritdoc />
    /// <remarks>The text's phonemes as token ids, framed <c>&lt;bos&gt; … &lt;eos&gt;</c>.</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        var ids = EncodePhonemes(PhonemizeText(text));
        if (ids.Length == 0) throw new ArgumentException("The text has no pronounceable content.", nameof(text));
        return ToTensor(Frame(ids.Take(_options.MaxTextLength).ToArray()));
    }

    private static int[] Frame(IReadOnlyList<int> ids)
    {
        var framed = new int[ids.Count + 2];
        framed[0] = BeginToken;
        for (int i = 0; i < ids.Count; i++) framed[i + 1] = ids[i];
        framed[ids.Count + 1] = EndToken;
        return framed;
    }

    /// <summary>Phoneme ids with or without their <c>&lt;bos&gt; … &lt;eos&gt;</c> frame, as framed ids.</summary>
    protected static int[] Framed(int[] ids) =>
        ids.Length >= 2 && ids[0] == BeginToken && ids[ids.Length - 1] == EndToken ? ids : Frame(ids);

    /// <summary>The symbol of token id <paramref name="id"/>.</summary>
    protected string Symbol(int id) => _tokens[id];

    // ---------------------------------------------------------------- codes

    /// <summary>The codes <c>[frames, codebooks]</c> of a codec-token tensor or, without one, of a recording.</summary>
    protected int[,] CodesOf(Tensor<T>? tokens, Tensor<T>? audio, string name)
    {
        int codebooks = _options.NumCodebooks;
        if (tokens is not null)
        {
            if (tokens.Rank != 2 || tokens.Shape[1] != codebooks)
                throw new ArgumentException($"Expected codec tokens [frames, {codebooks}].", name);
            int frames = tokens.Shape[0];
            var codes = new int[frames, codebooks];
            for (int f = 0; f < frames; f++)
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

    /// <summary>The codec's <c>[codebooks, frames]</c> as <c>[frames, codebooks]</c>.</summary>
    protected static int[,] FramesFirst(int[,] codes)
    {
        int codebooks = codes.GetLength(0), frames = codes.GetLength(1);
        var transposed = new int[frames, codebooks];
        for (int q = 0; q < codebooks; q++)
            for (int f = 0; f < frames; f++) transposed[f, q] = codes[q, f];
        return transposed;
    }

    /// <summary>The first codebook of codes <c>[frames, codebooks]</c>.</summary>
    protected static int[] FirstCodebook(int[,] codes) =>
        Enumerable.Range(0, codes.GetLength(0)).Select(t => codes[t, 0]).ToArray();

    // ---------------------------------------------------------------- training

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        ThrowIfTokenMelTrainingUnsupported();
    }

    /// <inheritdoc />
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        var objective = BuildObjective(sample, _trainingRandom, training: true);
        return TrainWithCustomObjective(sample.Tokens, sample.Tokens, objective, StageOptimizer());
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

    private Func<Tensor<T>, Tensor<T>, Tensor<T>> BuildObjective(TtsTrainingSample<T> sample, Random random, bool training)
    {
        Guard.NotNull(sample);
        if (!HasPaperLayers)
            throw new NotSupportedException("The paper objective needs the paper's layers; this model was built from caller-supplied layers.");
        var text = Framed(ToIds(sample.Tokens, nameof(sample)));
        var codes = CodesOf(sample.CodecTokens, sample.Audio, nameof(sample));
        if (codes.GetLength(0) < 2)
            throw new ArgumentException($"{GetType().Name} trains on at least two frames.", nameof(sample));
        return CurrentStage == VallETrainingStage.AutoRegressive
            ? AutoRegressiveObjective(sample, text, codes, training, random)
            : NonAutoRegressiveObjective(sample, text, codes, training, random);
    }

    /// <summary>The AR objective on one utterance (by default the core's teacher-forced loss of its first codebook).</summary>
    protected virtual Func<Tensor<T>, Tensor<T>, Tensor<T>> AutoRegressiveObjective(TtsTrainingSample<T> sample, int[] text,
        int[,] codes, bool training, Random random)
    {
        var first = FirstCodebook(codes);
        return (_, _) => PaperCore.ArLoss(text, first, training, random);
    }

    /// <summary>The NAR objective on one utterance.</summary>
    protected abstract Func<Tensor<T>, Tensor<T>, Tensor<T>> NonAutoRegressiveObjective(TtsTrainingSample<T> sample, int[] text,
        int[,] codes, bool training, Random random);

    /// <summary>The learning rate's linear warm-up, in updates.</summary>
    protected virtual int WarmupSteps => _options.WarmupSteps;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? StageOptimizer()
    {
        if (_suppliedOptimizer is not null)
            return _suppliedOptimizer;
        if (!_stageOptimizers.TryGetValue(CurrentStage, out var optimizer))
        {
            var o = _options;
            int warmup = WarmupSteps, total = o.TrainingSteps;
            var schedule = new LambdaLRScheduler(o.LearningRate, step =>
            {
                if (step < warmup) return (double)step / Math.Max(1, warmup);
                return Math.Max(0.0, (double)(total - step) / Math.Max(1, total - warmup));
            });
            var options = new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = o.LearningRate,
                Beta1 = 0.9,
                Beta2 = 0.999,
                Epsilon = 1e-8,
                WeightDecay = o.WeightDecay,
                LearningRateScheduler = schedule,
            };
            optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, options),
                OptimizerComponent(CurrentStage));
            _stageOptimizers[CurrentStage] = optimizer;
        }
        return optimizer;
    }

    /// <inheritdoc />
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var stageLayers = CurrentStage == VallETrainingStage.AutoRegressive ? _arLayers : _narLayers;
        var stage = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(stageLayers.Cast<ILayer<T>>().ToList(), -1),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(stage.Contains).ToList();
    }

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    /// <remarks>
    /// <paramref name="input"/> is the framed phoneme sequence (<see cref="PreprocessText"/>); the output is the
    /// waveform <c>[samples]</c> of the new frames.
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        if (!HasPaperLayers) return base.PredictCore(input);
        var voice = RequireVoice();
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var prompt = FramesFirst(Codec.Encode(voice.ReferenceAudio!));                                   // [P, Q]
            var enrolled = voice.ReferenceTokens is { } tokens ? ToIds(tokens, nameof(voice)) : Array.Empty<int>();
            var codes = Generate(Framed(ToIds(input, nameof(input))), enrolled, prompt, voice, random);     // [frames, Q]
            int frames = codes.GetLength(0), codebooks = codes.GetLength(1);
            var byCodebook = new int[codebooks, frames];
            for (int f = 0; f < frames; f++)
                for (int q = 0; q < codebooks; q++) byCodebook[q, f] = codes[f, q];
            var audio = Codec.Decode(byCodebook);
            return audio.Reshape(new[] { audio.Length });
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>
    /// The phonemes the AR model reads in synthesis: <c>&lt;bos&gt;</c>, the prompt's transcript and a word separator
    /// (when there is a transcript), the text, <c>&lt;eos&gt;</c> — the reference phonemizes "{prompt} {text}" as one
    /// sentence.
    /// </summary>
    protected int[] JoinPrompt(int[] enrolled, int[] target)
    {
        var full = new List<int> { BeginToken };
        if (enrolled.Length > 0)
        {
            full.AddRange(enrolled);
            full.Add(_wordSeparator);
        }
        full.AddRange(target.Skip(1));
        return full.ToArray();
    }

    /// <summary>The codes <c>[frames, codebooks]</c> generated for the framed phonemes <paramref name="target"/> in the
    /// voice of the prompt (its transcript <paramref name="enrolled"/>, codes <paramref name="prompt"/>).</summary>
    protected abstract int[,] Generate(int[] target, int[] enrolled, int[,] prompt, TtsVoice<T> voice, Random random);

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    // ---------------------------------------------------------------- ICodecTts

    /// <inheritdoc />
    /// <remarks>EnCodec's codes of <paramref name="audio"/>, <c>[frames, codebooks]</c>.</remarks>
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
        if (tokens is null) throw new ArgumentNullException(nameof(tokens));
        if (tokens.Rank != 2) throw new ArgumentException("Expected codes [frames, codebooks].", nameof(tokens));
        var codes = new int[tokens.Shape[1], tokens.Shape[0]];
        for (int f = 0; f < tokens.Shape[0]; f++)
            for (int q = 0; q < tokens.Shape[1]; q++) codes[q, f] = (int)Math.Round(NumOps.ToDouble(tokens[f, q]));
        var audio = Codec.Decode(codes);
        return audio.Reshape(new[] { audio.Length });
    }

    // ---------------------------------------------------------------- pretrained weights

    /// <summary>Loads a reference state dictionary (safetensors, or a <c>.pt</c> of the checkpoint's <c>model</c>
    /// entry). Build the model with the checkpoint's sizes.</summary>
    public void LoadReferenceWeights(string path)
    {
        var file = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
        PaperCore.LoadTorchWeights((name, shape) =>
        {
            if (!file.TryGetValue(name, out var tensor) && !file.TryGetValue("model." + name, out tensor))
                throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
            var actual = tensor.Shape.ToArray();
            if (!actual.SequenceEqual(shape))
                throw new InvalidDataException($"'{name}' is [{string.Join(", ", actual)}] in the checkpoint but [{string.Join(", ", shape)}] " +
                    "in this model; build the model with the checkpoint's configuration.");
            return tensor.ToVector().Select(v => (double)v).ToArray();
        });
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <summary>Whether this model was built native (not ONNX).</summary>
    protected bool IsNative => _useNativeMode;

    /// <summary>Throws when the model has been disposed.</summary>
    protected void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? GetType().Name);
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
