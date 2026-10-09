using AiDotNet.Attributes;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.FrontEnd;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Which of VALL-E's two codec language models a training step updates.</summary>
public enum VallETrainingStage
{
    /// <summary>The autoregressive model of the first codebook (§4.2.1).</summary>
    AutoRegressive,

    /// <summary>The non-autoregressive model of codebooks 2 … 8 (§4.2.2).</summary>
    NonAutoRegressive,
}

/// <summary>
/// VALL-E: a neural codec language model for zero-shot text-to-speech, which continues a 3-second recording of an unseen
/// speaker by predicting EnCodec codes from phonemes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Reference: "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers" (Wang et al., 2023). Microsoft
/// released no code; what the paper leaves open follows the reproduction lifeiteng/vall-e, against which this model is
/// tested (see <see cref="VALLEOptions"/>).
/// </para>
/// <para>
/// <b>Model</b> (§4): EnCodec turns 24 kHz speech into eight codebooks of codes at 75 frames a second. An
/// autoregressive Transformer predicts the first codebook from the phonemes and the codes before it; a
/// non-autoregressive Transformer, told the stage through adaptive layer norm, predicts each later codebook from the
/// phonemes, the codebooks below it and an acoustic prompt. Each is trained on its own (<see cref="CurrentStage"/>).
/// </para>
/// <para>
/// <b>Synthesis</b> (§4.3, "VALL-E"): the prompt's transcript precedes the text, the prompt's first-codebook codes
/// start the AR decoding (sampling until the end token, or 16 codes per phoneme token), the NAR fills the other
/// codebooks greedily after the prompt (its text without the prompt's transcript, as the reference does), and EnCodec
/// decodes the new frames. A voice without a transcript is an empty enrolled transcript.
/// </para>
/// <para>
/// <b>Training data</b>: <see cref="TtsTrainingSample{T}.Tokens"/> are the phoneme ids (<see cref="EncodePhonemes"/>),
/// <see cref="TtsTrainingSample{T}.CodecTokens"/> the EnCodec codes <c>[frames, 8]</c> (or
/// <see cref="TtsTrainingSample{T}.Audio"/> at 24 kHz, which the model encodes). The paper crops each utterance to a
/// random 10–20 seconds together with its aligned phonemes; that needs the alignment, so it is the caller's.
/// </para>
/// <para><b>For Beginners:</b> VALL-E treats speech as text-like tokens. Given a few seconds of someone's voice and a
/// sentence, it writes the tokens of that person saying the sentence, and a codec turns them into audio.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional,
///     NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var valle = new VALLE&lt;float&gt;(architecture, new VALLEOptions());
/// valle.Voice = valle.CreateVoice(promptAudio24kHz, "the prompt's transcript");
/// var audio = valle.Synthesize("Hello there.");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers",
    "https://arxiv.org/abs/2301.02111",
    Year = 2023,
    Authors = "Wang et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-4, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = 0.01,
    Schedule = LearningRateSchedulerType.LinearWarmup, WarmupSteps = 32_000,
    PostWarmupDecay = LinearWarmupScheduler.DecayMode.Linear,
    Component = "autoregressive", Provenance = RecipeProvenance.Stated,
    Source = "Section 5.1: AdamW, the learning rate warmed up over the first 32k updates to a peak of 5e-4, then "
             + "decayed linearly, for 800k steps. The paper names no betas or weight decay; these are PyTorch's AdamW "
             + "defaults.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-4, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = 0.01,
    Schedule = LearningRateSchedulerType.LinearWarmup, WarmupSteps = 32_000,
    PostWarmupDecay = LinearWarmupScheduler.DecayMode.Linear,
    Component = "non-autoregressive", Provenance = RecipeProvenance.Stated,
    Source = "Section 5.1: AdamW, the learning rate warmed up over the first 32k updates to a peak of 5e-4, then "
             + "decayed linearly, for 800k steps. The paper names no betas or weight decay; these are PyTorch's AdamW "
             + "defaults.")]
public partial class VALLE<T> : TtsModelBase<T>, ICodecTts<T>
{
    private const int PadToken = 0, BeginToken = 1, EndToken = 2;

    private readonly VALLEOptions _options;
    private readonly bool _useNativeMode;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _suppliedOptimizer;
    private readonly Dictionary<VallETrainingStage, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _stageOptimizers = new();
    private readonly Random _trainingRandom;
    private bool _disposed;

    // Vocabulary: [<pad>, <bos>, <eos>] + sorted(phoneme table) (the reference's TextTokenCollater).
    private readonly Dictionary<string, int> _tokenIds = new(StringComparer.Ordinal);
    private readonly int _wordSeparator;

    private VallECore<T>? _core;
    private readonly List<LayerBase<T>> _arLayers = new();
    private readonly List<LayerBase<T>> _narLayers = new();
    private AudioCodecLayer<T>? _codec;

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Creates an ONNX-backed VALL-E for inference.</summary>
    public VALLE(NeuralNetworkArchitecture<T> architecture, string modelPath, VALLEOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new VALLEOptions();
        _useNativeMode = false;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        _wordSeparator = BuildVocabulary(_tokenIds);
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new AiDotNet.Onnx.OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable VALL-E.</summary>
    public VALLE(NeuralNetworkArchitecture<T> architecture, VALLEOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new VALLEOptions();
        _useNativeMode = true;
        _suppliedOptimizer = optimizer;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        _wordSeparator = BuildVocabulary(_tokenIds);
        ConfigureBase();
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        base.SampleRate = _options.SampleRate;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        _options.VocabSize = 3 + LibriTtsPhonemeTable.Symbols.Length;
        if (_options.VocabSize > _options.TextTokens)
            throw new ArgumentException($"VALL-E's {_options.VocabSize} phoneme tokens need TextTokens ≥ {_options.VocabSize}.");
    }

    private static int BuildVocabulary(Dictionary<string, int> ids)
    {
        var symbols = new List<string>(LibriTtsPhonemeTable.Symbols);
        symbols.Sort(StringComparer.Ordinal);                      // Python's sorted(): code-point order
        var tokens = new List<string> { "<pad>", "<bos>", "<eos>" };
        tokens.AddRange(symbols);
        for (int i = 0; i < tokens.Count; i++) ids[tokens[i]] = i;
        return ids["_"];
    }

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

    /// <summary>The codec whose codes VALL-E models.</summary>
    public EnCodec<T> Codec => (EnCodec<T>?)_codec?.Codec ?? throw new InvalidOperationException("VALL-E's codec exists in native mode only.");

    /// <inheritdoc />
    protected override TtsSupervision RequiredSupervision => TtsSupervision.CodecTokens;

    /// <inheritdoc />
    protected override TtsSupervision RequiredVoice => TtsSupervision.ReferenceRecording;

    /// <inheritdoc />
    public override int CodecTokenCodebooks => _options.NumCodebooks;

    /// <inheritdoc />
    public override int CodecTokenVocabulary => _options.CodebookSize;

    private bool HasPaperLayers => _core is not null;

    /// <inheritdoc />
    /// <remarks>The phoneme ids: the vocabulary, not the embedding table (whose 512 rows are the reference's
    /// <c>NUM_TEXT_TOKENS</c>, more than the vocabulary uses).</remarks>
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperLayers ? LayerInputDomain.Indices(_options.VocabSize) : base.GetInputDomain(inputShape);

    // The networks' layers, for tests that check a stage updates its own network only.
    internal IReadOnlyList<LayerBase<T>> AutoRegressiveLayers => _arLayers;
    internal IReadOnlyList<LayerBase<T>> NonAutoRegressiveLayers => _narLayers;
    internal VallECore<T>? Core => _core;

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
        _core = new VallECore<T>(Engine, _arLayers, _narLayers, new VallEConfiguration(
            TextTokens: o.TextTokens, AudioTokens: o.CodebookSize, Codebooks: o.NumCodebooks, ModelDim: o.HiddenDim,
            Heads: o.NumHeads, Layers: o.NumDecoderLayers, FeedForwardDim: o.FeedForwardDim, Dropout: o.DropoutRate));
        if (o.NumEncoderLayers != o.NumDecoderLayers)
            throw new ArgumentException("VALL-E's AR and NAR models have the same depth (§5.1); set NumEncoderLayers = NumDecoderLayers.");
        _core.InitializeLikePyTorch(AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? o.SamplingSeed));

        // VALL-E only encodes and decodes with its (pretrained, frozen) codec.
        var codecOptions = (EnCodecOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(o.Codec);
        codecOptions.IncludeDiscriminators = false;
        var codec = new EnCodec<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1) { RandomSeed = Architecture.RandomSeed }, codecOptions);
        int codecRate = ((IAudioCodec<T>)codec).SampleRate;
        if (codec.HopLength != o.HopSize || codecRate != o.SampleRate)
            throw new ArgumentException($"VALL-E's {o.SampleRate} Hz audio and {o.HopSize}-sample frames must be its codec's " +
                $"({codecRate} Hz, {codec.HopLength}-sample hop); set SampleRate and HopSize to match the codec options.");
        if (codec.CodebookSize != o.CodebookSize)
            throw new ArgumentException($"VALL-E's {o.CodebookSize} codes per codebook must be the codec's {codec.CodebookSize}.");
        int codebooks = codec.QuantizersForBandwidth(codecOptions.TargetBandwidthKbps);
        if (codebooks != o.NumCodebooks)
            throw new ArgumentException($"VALL-E models {o.NumCodebooks} codebooks; the codec's bandwidth gives {codebooks}.");
        _codec = new AudioCodecLayer<T>(codec);

        AddEncoderDecoderLayers(_arLayers.Cast<ILayer<T>>().ToList(), Array.Empty<ILayer<T>>());
        ComponentLayers.AddRange(_narLayers);
        ComponentLayers.Add(_codec);
    }

    // ---------------------------------------------------------------- vocabulary and front end

    /// <summary>VALL-E's token id of each phoneme symbol, without the <c>&lt;bos&gt; … &lt;eos&gt;</c> frame. Every symbol
    /// must be in the phoneme table (the reference asserts it).</summary>
    public int[] EncodePhonemes(IEnumerable<string> symbols)
    {
        var ids = new List<int>();
        foreach (var symbol in symbols)
        {
            if (!_tokenIds.TryGetValue(symbol, out int id) || id < 3)
                throw new ArgumentException($"'{symbol}' is not in VALL-E's phoneme table.", nameof(symbols));
            ids.Add(id);
        }
        return ids.ToArray();
    }

    /// <summary>A voice for <see cref="TtsModelBase{T}.Voice"/>: a 24 kHz prompt recording (about 3 seconds) and its
    /// transcript.</summary>
    public TtsVoice<T> CreateVoice(Tensor<T> recording, string transcript)
    {
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        var ids = EncodePhonemes(EnglishG2P.Default.Phonemize(transcript ?? throw new ArgumentNullException(nameof(transcript))));
        return new TtsVoice<T> { ReferenceAudio = recording, ReferenceTokens = ToTensor(ids) };
    }

    private Tensor<T> ToTensor(IReadOnlyList<int> ids)
    {
        var tensor = new Tensor<T>(new[] { ids.Count });
        for (int i = 0; i < ids.Count; i++) tensor[i] = NumOps.FromDouble(ids[i]);
        return tensor;
    }

    private int[] ToIds(Tensor<T> tensor, string name)
    {
        var ids = new int[tensor.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = (int)Math.Round(NumOps.ToDouble(tensor[i]));
            if (ids[i] < 0 || ids[i] >= _options.VocabSize)
                throw new ArgumentOutOfRangeException(name, $"VALL-E's token ids lie in [0, {_options.VocabSize}).");
        }
        return ids;
    }

    /// <inheritdoc />
    /// <remarks>The text's phonemes (<see cref="EnglishG2P"/>) as VALL-E's token ids, framed
    /// <c>&lt;bos&gt; … &lt;eos&gt;</c>.</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        var ids = EncodePhonemes(EnglishG2P.Default.Phonemize(text));
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

    // Phoneme ids with or without their <bos> … <eos> frame, as framed ids.
    private static int[] Framed(int[] ids) =>
        ids.Length >= 2 && ids[0] == BeginToken && ids[ids.Length - 1] == EndToken ? ids : Frame(ids);

    // ---------------------------------------------------------------- codes

    private int[,] CodesOf(TtsTrainingSample<T> sample)
    {
        int codebooks = _options.NumCodebooks;
        if (sample.CodecTokens is { } tokens)
        {
            if (tokens.Rank != 2 || tokens.Shape[1] != codebooks)
                throw new ArgumentException($"Expected codec tokens [frames, {codebooks}].", nameof(sample));
            int frames = tokens.Shape[0];
            var codes = new int[frames, codebooks];
            for (int f = 0; f < frames; f++)
                for (int q = 0; q < codebooks; q++)
                {
                    codes[f, q] = (int)Math.Round(NumOps.ToDouble(tokens[f, q]));
                    if (codes[f, q] < 0 || codes[f, q] >= _options.CodebookSize)
                        throw new ArgumentOutOfRangeException(nameof(sample), $"Codec tokens lie in [0, {_options.CodebookSize}).");
                }
            return codes;
        }
        var audio = sample.Audio ?? throw new ArgumentException("VALL-E trains on codec tokens or the recording.", nameof(sample));
        return FramesFirst(Codec.Encode(audio));
    }

    // The codec's [codebooks, frames] as [frames, codebooks].
    private static int[,] FramesFirst(int[,] codes)
    {
        int codebooks = codes.GetLength(0), frames = codes.GetLength(1);
        var transposed = new int[frames, codebooks];
        for (int q = 0; q < codebooks; q++)
            for (int f = 0; f < frames; f++) transposed[f, q] = codes[q, f];
        return transposed;
    }

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
        var codes = CodesOf(sample);
        int frames = codes.GetLength(0);
        if (frames < 2)
            throw new ArgumentException("VALL-E trains on at least two frames.", nameof(sample));
        var core = _core!;
        if (CurrentStage == VallETrainingStage.AutoRegressive)
        {
            var first = Enumerable.Range(0, frames).Select(t => codes[t, 0]).ToArray();
            return (_, _) => core.ArLoss(text, first, training, random);
        }
        int promptFrames = (int)Math.Round(_options.PromptSeconds * CodecFrameRate);
        return (_, _) => core.NarLoss(text, codes, promptFrames, training, random);
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? StageOptimizer()
    {
        if (_suppliedOptimizer is not null)
            return _suppliedOptimizer;
        if (!_stageOptimizers.TryGetValue(CurrentStage, out var optimizer))
        {
            var o = _options;
            int warmup = o.WarmupSteps, total = o.TrainingSteps;
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
                CurrentStage == VallETrainingStage.AutoRegressive ? "autoregressive" : "non-autoregressive");
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
    /// <paramref name="input"/> is VALL-E's framed phoneme sequence (<see cref="PreprocessText"/>); the output is the
    /// 24 kHz waveform <c>[samples]</c> of the new frames.
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
            var codes = Generate(ToIds(input, nameof(input)), voice, random);                                  // [frames, 8]
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

    /// <summary>The codes <c>[frames, codebooks]</c> VALL-E generates for phoneme ids in the voice of
    /// <paramref name="voice"/> (§4.3).</summary>
    private int[,] Generate(int[] textIds, TtsVoice<T> voice, Random random)
    {
        var core = _core!;
        var prompt = FramesFirst(Codec.Encode(voice.ReferenceAudio!));                                    // [P, 8]
        var target = Framed(textIds);
        var enrolled = voice.ReferenceTokens is { } tokens ? ToIds(tokens, nameof(voice)) : Array.Empty<int>();
        // "{prompt} {text}" phonemized as one sentence: the prompt's phonemes, a word separator, then the text's.
        var full = new List<int> { BeginToken };
        if (enrolled.Length > 0)
        {
            full.AddRange(enrolled);
            full.Add(_wordSeparator);
        }
        full.AddRange(target.Skip(1));
        var promptFirst = Enumerable.Range(0, prompt.GetLength(0)).Select(t => prompt[t, 0]).ToArray();
        var first = core.ArGenerate(full, promptFirst, _options.Temperature, _options.TopK,
            _options.MaxCodesPerTextToken * full.Count + 1, random);
        // prefix_mode 2: the NAR reads <bos> + the text from the separator before it (without the enrolled phonemes).
        int enrolledLength = enrolled.Length + 2;
        var narText = enrolled.Length > 0 ? new[] { BeginToken }.Concat(full.Skip(enrolledLength - 1)).ToArray() : full.ToArray();
        return core.NarGenerate(narText, prompt, first.ToArray(), random);
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    // ---------------------------------------------------------------- ICodecTts

    /// <inheritdoc />
    /// <remarks>EnCodec's codes of <paramref name="audio"/> (24 kHz), <c>[frames, codebooks]</c>.</remarks>
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        var codes = FramesFirst(Codec.Encode(audio ?? throw new ArgumentNullException(nameof(audio))));
        var tensor = new Tensor<T>(new[] { codes.GetLength(0), codes.GetLength(1) });
        for (int f = 0; f < codes.GetLength(0); f++)
            for (int q = 0; q < codes.GetLength(1); q++) tensor[f, q] = NumOps.FromDouble(codes[f, q]);
        return tensor;
    }

    /// <inheritdoc />
    /// <remarks>The 24 kHz waveform of codes <c>[frames, codebooks]</c>.</remarks>
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

    /// <summary>Loads the reference's <c>VALLE</c> state dictionary (safetensors, or a <c>.pt</c> of the
    /// checkpoint's <c>model</c> entry). Build the model with the checkpoint's sizes.</summary>
    public void LoadReferenceWeights(string path)
    {
        var file = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
        (_core ?? throw new InvalidOperationException("VALL-E's networks exist in native mode only.")).LoadTorchWeights((name, shape) =>
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

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "VALL-E-Native" : "VALL-E-ONNX",
            Description = "VALL-E: Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (Wang et al., 2023)",
            FeatureCount = _options.HiddenDim,
            Complexity = _options.NumEncoderLayers + _options.NumDecoderLayers,
        };
        metadata.AdditionalInfo["Architecture"] = "VALL-E";
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(VALLE<T>));
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
