using AiDotNet.Attributes;
using AiDotNet.Audio.Foundations;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.FrontEnd;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Which SPEAR-TTS network, and which of its training tasks, a training step runs.</summary>
public enum SPEARTTSTrainingStage
{
    /// <summary>S1's pretraining (§4.1): reconstruct semantic tokens from a copy with tokens deleted at random.</summary>
    Pretraining,

    /// <summary>S1's finetuning (§4.1, §7.2): text to semantic tokens, updating the encoder's embeddings and lower layers.</summary>
    Reading,

    /// <summary>The backtranslation model's finetuning (§4.2): semantic tokens to text, updating the decoder's top and
    /// bottom layers.</summary>
    Backtranslation,

    /// <summary>S2 (§5): semantic to acoustic tokens, prompted by another window of the same utterance.</summary>
    Speaking,
}

/// <summary>
/// SPEAR-TTS: multi-speaker text-to-speech trained with minimal supervision (Kharitonov et al. 2023), in two stages:
/// "reading" text into semantic tokens and "speaking" them as acoustic tokens.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>Reading</b>, S1 (§4): a T5-Large encoder–decoder from phonemes to w2v-BERT semantic tokens (512 clusters, 25 Hz,
/// consecutive repetitions removed). It is pretrained on audio alone to reconstruct each semantic sequence from a copy
/// with every token deleted with probability 0.6 (<see cref="SPEARTTSTrainingStage.Pretraining"/>), then finetuned on a
/// little parallel data (<see cref="SPEARTTSTrainingStage.Reading"/>): only the encoder's embeddings and its lower
/// layers learn, the upper encoder and the whole decoder stay as pretrained. Backtranslation (§4.2) finetunes a copy of
/// the pretrained model the other way, semantic tokens to text (<see cref="SPEARTTSTrainingStage.Backtranslation"/>);
/// <see cref="Backtranscribe"/> then turns untranscribed audio into synthetic parallel data for the reading stage.
/// Synthesis reads with beam search (beam 10).
/// </para>
/// <para>
/// <b>Speaking</b>, S2 (§5): a decoder-only Transformer from semantic to acoustic tokens, the first 3 levels of a
/// SoundStream codec interleaved per frame. Training takes two non-overlapping windows of each utterance: the prompt's
/// semantic tokens, the target's semantic tokens and the prompt's acoustic tokens are the prefix, and the model learns
/// the target's acoustic tokens, so the voice and recording conditions of a 3-second prompt carry over to what it
/// speaks. Synthesis samples at temperature 0.75 with <see cref="TtsModelBase{T}.Voice"/> as the prompt.
/// </para>
/// <para>
/// No official code exists. Gaps in S1's procedure follow lucidrains/spear-tts-pytorch and S2's sequence layout
/// follows AudioLM's coarse stage (<see cref="AudioLmCore{T}"/>); each choice is listed in
/// <c>docs/text-to-speech/paper-fidelity-decisions.md</c>.
/// </para>
/// <para><b>For Beginners:</b> SPEAR-TTS first learns how speech is built from hours of recordings with no
/// transcripts, then needs only a few minutes of transcribed speech to learn to read, and speaks in any voice it hears
/// for three seconds.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional,
///     NeuralNetworkTaskType.Generative, inputSize: 1, outputSize: 1);
/// var model = new SPEARTTS&lt;float&gt;(architecture, new SPEARTTSOptions(), semanticTokenizer: w2vBert);
/// model.Voice = model.CreateVoice(threeSecondRecording);
/// Tensor&lt;float&gt; audio = model.Synthesize("Hello there.");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Speak, Read and Prompt: High-Fidelity Text-to-Speech with Minimal Supervision",
    "https://arxiv.org/abs/2302.03540",
    Year = 2023,
    Authors = "Kharitonov et al."
)]
[PaperOptimizer(OptimizerKind.Adafactor, Component = "reading",
    Source = "Kharitonov et al. 2023, Sec. 7.2: every S1 experiment uses the Adafactor optimizer with inverse square-root "
             + "learning rate decay (Adafactor's relative step size, min(1e-2, 1/sqrt(step))). No rate is declared because "
             + "the paper states none and Adafactor derives its own. Pretraining, reading and backtranslation share it.")]
[PaperOptimizer(OptimizerKind.Adafactor, Component = "speaking",
    Source = "Kharitonov et al. 2023, Sec. 7.3: the optimizer and the learning rate schedule of S2 are the same as for S1.")]
public partial class SPEARTTS<T> : TtsModelBase<T>, ICodecTts<T>
{
    // S1's vocabulary: <pad> 0 (also the decoder's start), </s> 1, the phonemes, then the semantic tokens.
    private const int PadToken = 0, EndToken = 1, PhonemeOffset = 2;

    private readonly SPEARTTSOptions _options;
    private readonly bool _useNativeMode;
    private readonly W2vBert<T>? _semanticTokenizer;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _suppliedOptimizer;
    private readonly Dictionary<SPEARTTSTrainingStage, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _stageOptimizers = new();
    private readonly Dictionary<string, int> _phonemeIds = new(StringComparer.Ordinal);
    private readonly string[] _phonemes = LibriTtsPhonemeTable.Symbols.ToArray();
    private readonly Random _trainingRandom;
    private bool _disposed;

    private readonly List<LayerBase<T>> _readingLayers = new();
    private readonly List<LayerBase<T>> _backtranslationLayers = new();
    private readonly List<LayerBase<T>> _speakingLayers = new();
    private T5Seq2Seq<T>? _reading;
    private T5Seq2Seq<T>? _backtranslation;
    private T5DecoderOnly<T>? _speaking;
    private AudioCodecLayer<T>? _codec;

    /// <summary>Creates an ONNX-backed SPEAR-TTS for inference.</summary>
    public SPEARTTS(NeuralNetworkArchitecture<T> architecture, string modelPath, SPEARTTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new SPEARTTSOptions();
        Options = _options;
        _useNativeMode = false;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable SPEAR-TTS.</summary>
    /// <param name="architecture">The architecture (its layers are built here; custom layers replace the paper's).</param>
    /// <param name="options">The options; the paper's when null.</param>
    /// <param name="semanticTokenizer">A pre-trained w2v-BERT with a fitted semantic tokenizer, for voice prompts and
    /// training utterances given as audio. Optional: without it, pass semantic tokens yourself.</param>
    /// <param name="optimizer">An optimizer for every stage instead of the per-stage Adafactor.</param>
    public SPEARTTS(NeuralNetworkArchitecture<T> architecture, SPEARTTSOptions? options = null,
        W2vBert<T>? semanticTokenizer = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new SPEARTTSOptions();
        Options = _options;
        _useNativeMode = true;
        _semanticTokenizer = semanticTokenizer;
        _suppliedOptimizer = optimizer;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        ConfigureBase();
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        var o = _options;
        if (o.SemanticCodes < 1 || o.NumCodebooks < 1 || o.CodebookSize < 1)
            throw new ArgumentException("SPEAR-TTS needs semantic codes, codec levels and codes per level.");
        if (o.PretrainingDeletionProbability < 0 || o.PretrainingDeletionProbability >= 1)
            throw new ArgumentOutOfRangeException(nameof(o.PretrainingDeletionProbability), "The deletion probability lies in [0, 1).");
        if (o.ReadingTrainableEncoderLayers < 0 || o.ReadingTrainableEncoderLayers > o.NumEncoderLayers)
            throw new ArgumentOutOfRangeException(nameof(o.ReadingTrainableEncoderLayers), $"S1 has {o.NumEncoderLayers} encoder layers.");
        if (o.BacktranslationTrainableDecoderLayers < 0 || o.BacktranslationTrainableDecoderLayers > o.NumDecoderLayers)
            throw new ArgumentOutOfRangeException(nameof(o.BacktranslationTrainableDecoderLayers), $"S1 has {o.NumDecoderLayers} decoder layers.");
        base.SampleRate = o.SampleRate;
        base.HopSize = o.HopSize;
        base.HiddenDim = o.HiddenDim;
        o.VocabSize = SemanticOffset + o.SemanticCodes;
        for (int i = 0; i < _phonemes.Length; i++) _phonemeIds[_phonemes[i]] = PhonemeOffset + i;
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

    /// <summary>The network and task a training step runs; <see cref="SPEARTTSTrainingStage.Reading"/>, training on
    /// parallel data, until set. The paper's order is pretraining, backtranslation, then reading and speaking.</summary>
    public SPEARTTSTrainingStage CurrentStage { get; set; } = SPEARTTSTrainingStage.Reading;

    /// <summary>The SoundStream codec whose codes SPEAR-TTS speaks.</summary>
    public SoundStream<T> Codec => (SoundStream<T>?)_codec?.Codec ?? throw new InvalidOperationException("SPEAR-TTS's codec exists in native mode only.");

    /// <inheritdoc />
    protected override TtsSupervision RequiredSupervision => TtsSupervision.CodecTokens;

    /// <inheritdoc />
    /// <remarks>The semantic token at the codec's frame rate, then the acoustic levels.</remarks>
    public override int CodecTokenCodebooks => 1 + _options.NumCodebooks;

    /// <inheritdoc />
    public override int CodecTokenVocabulary => Math.Min(_options.SemanticCodes, _options.CodebookSize);

    private int SemanticOffset => PhonemeOffset + _phonemes.Length;

    private bool HasPaperLayers => _speaking is not null;

    /// <inheritdoc />
    /// <remarks>S1's text ids: the special tokens and the phonemes.</remarks>
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperLayers ? LayerInputDomain.Indices(SemanticOffset) : base.GetInputDomain(inputShape);

    private T5Seq2Seq<T> Reading =>
        _reading ?? throw new InvalidOperationException("SPEAR-TTS's reading network exists in native mode only.");

    private T5Seq2Seq<T> BacktranslationModel =>
        _backtranslation ?? throw new InvalidOperationException("SPEAR-TTS's backtranslation network exists in native mode only.");

    private T5DecoderOnly<T> Speaking =>
        _speaking ?? throw new InvalidOperationException("SPEAR-TTS's speaking network exists in native mode only.");

    // The networks, for tests that check a stage updates what the paper says and nothing else.
    internal T5Seq2Seq<T>? ReadingModel => _reading;
    internal T5Seq2Seq<T>? BacktranslationNetwork => _backtranslation;
    internal T5DecoderOnly<T>? SpeakingModel => _speaking;
    internal IReadOnlyList<LayerBase<T>> ReadingLayers => _readingLayers;
    internal IReadOnlyList<LayerBase<T>> BacktranslationLayers => _backtranslationLayers;
    internal IReadOnlyList<LayerBase<T>> SpeakingLayers => _speakingLayers;

    // S2's vocabulary (AudioLM's coarse stage): the semantic codes, the Q levels offset by q·N, then BOS, SEP and EOS.
    private int AcousticOffset => _options.SemanticCodes;
    private int SpeakingBegin => _options.SemanticCodes + _options.NumCodebooks * _options.CodebookSize;
    private int SpeakingSeparator => SpeakingBegin + 1;
    private int SpeakingEnd => SpeakingBegin + 2;

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
        // T5-Large as T5 v1.0 defines it: ReLU feed-forward and the decoder's embedding tied to its output. The encoder
        // reads its own table, since finetuning updates "the encoder embeddings" with "the entire decoder" frozen.
        var reading = new T5Configuration(
            VocabularySize: o.VocabSize, ModelDim: o.HiddenDim, FeedForwardDim: o.ReadingFeedForwardDim,
            KeyValueDim: o.ReadingKeyValueDim, Heads: o.NumHeads, EncoderLayers: o.NumEncoderLayers,
            DecoderLayers: o.NumDecoderLayers, Dropout: o.DropoutRate, DecoderStartTokenId: PadToken,
            EndTokenId: EndToken, GatedGelu: false, TieWordEmbeddings: true, SeparateEncoderEmbedding: true);
        _reading = new T5Seq2Seq<T>(Engine, _readingLayers, reading);
        _reading.InitializeLikeHuggingFace(random);
        _backtranslation = new T5Seq2Seq<T>(Engine, _backtranslationLayers, reading);
        _backtranslation.InitializeLikeHuggingFace(random);

        _speaking = new T5DecoderOnly<T>(Engine, _speakingLayers, new T5Configuration(
            VocabularySize: SpeakingEnd + 1, ModelDim: o.SpeakingModelDim, FeedForwardDim: o.SpeakingFeedForwardDim,
            KeyValueDim: o.SpeakingKeyValueDim, Heads: o.SpeakingHeads, EncoderLayers: 0, DecoderLayers: o.SpeakingLayers,
            Dropout: o.SpeakingDropoutRate, TieWordEmbeddings: false));
        _speaking.InitializeLikeHuggingFace(random);

        // SPEAR-TTS only encodes and decodes with its (pretrained, frozen) codec, so the codec's discriminators are
        // never built.
        var codecOptions = (SoundStreamOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(o.SoundStream);
        codecOptions.IncludeDiscriminators = false;
        var codec = new SoundStream<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1) { RandomSeed = Architecture.RandomSeed }, codecOptions);
        if (codec.CodebookSize != o.CodebookSize)
            throw new ArgumentException($"SPEAR-TTS's {o.CodebookSize} codes per level must equal the codec's {codec.CodebookSize}.");
        int codecRate = ((IAudioCodec<T>)codec).SampleRate;
        if (codec.HopLength != o.HopSize || codecRate != o.SampleRate)
            throw new ArgumentException($"SPEAR-TTS's {o.SampleRate} Hz audio and {o.HopSize}-sample frames must be its codec's " +
                $"({codecRate} Hz, {codec.HopLength}-sample hop); set SampleRate and HopSize to match the SoundStream options.");
        if (codec.QuantizersForBandwidth(o.SoundStream.TargetBandwidthKbps) != o.NumCodebooks)
            throw new ArgumentException($"SPEAR-TTS speaks {o.NumCodebooks} levels; the codec's bandwidth gives " +
                $"{codec.QuantizersForBandwidth(o.SoundStream.TargetBandwidthKbps)}.");
        if (CodecFrameRate % o.SemanticFrameRate != 0)
            throw new ArgumentException($"The codec's {CodecFrameRate} frames per second must be a multiple of the semantic tokens' {o.SemanticFrameRate}.");
        _codec = new AudioCodecLayer<T>(codec);

        AddEncoderDecoderLayers(_readingLayers.Cast<ILayer<T>>().ToList(), Array.Empty<ILayer<T>>());
        ComponentLayers.AddRange(_backtranslationLayers);
        ComponentLayers.AddRange(_speakingLayers);
        ComponentLayers.Add(_codec);
    }

    // ---------------------------------------------------------------- front end

    /// <summary>S1's token id of each phoneme symbol (espeak-ng's, as <see cref="EnglishG2P"/> emits them); symbols
    /// outside the table are dropped.</summary>
    public int[] EncodePhonemes(IEnumerable<string> symbols)
    {
        if (symbols is null) throw new ArgumentNullException(nameof(symbols));
        var ids = new List<int>();
        foreach (var symbol in symbols)
            if (_phonemeIds.TryGetValue(symbol, out int id)) ids.Add(id);
        return ids.ToArray();
    }

    /// <summary>The phoneme symbols of S1 token ids (special and semantic ids are skipped).</summary>
    public string[] DecodePhonemes(IEnumerable<int> ids)
    {
        if (ids is null) throw new ArgumentNullException(nameof(ids));
        return ids.Where(id => id >= PhonemeOffset && id < SemanticOffset).Select(id => _phonemes[id - PhonemeOffset]).ToArray();
    }

    /// <inheritdoc />
    /// <remarks>The text's phonemes (<see cref="EnglishG2P"/>) as S1's ids, truncated to
    /// <see cref="MaxTextLength"/>, then <c>&lt;/s&gt;</c>. The paper reads phonemes (§6.3).</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        var ids = EncodePhonemes(EnglishG2P.Default.Phonemize(text));
        if (ids.Length == 0) throw new ArgumentException("The text has no pronounceable content.", nameof(text));
        int length = Math.Min(ids.Length, _options.MaxTextLength - 1);
        var tokens = new Tensor<T>(new[] { length + 1 });
        for (int i = 0; i < length; i++) tokens[i] = NumOps.FromDouble(ids[i]);
        tokens[length] = NumOps.FromDouble(EndToken);
        return tokens;
    }

    /// <summary>A voice for <see cref="TtsModelBase{T}.Voice"/>: a recording of the speaker, of which the first
    /// <see cref="SPEARTTSOptions.PromptSeconds"/> are the prompt, with its semantic tokens at 25 Hz (from the semantic
    /// tokenizer when <paramref name="semanticTokens"/> is null). The paper needs no transcript of the prompt.</summary>
    public TtsVoice<T> CreateVoice(Tensor<T> recording, int[]? semanticTokens = null)
    {
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        int samples = Math.Min(recording.Length, (int)Math.Round(_options.PromptSeconds * _options.SampleRate));
        var prompt = new Tensor<T>(new[] { samples });
        for (int i = 0; i < samples; i++) prompt[i] = recording[i];
        var tokens = semanticTokens ?? (_semanticTokenizer ?? throw new InvalidOperationException(
            "A voice needs its semantic tokens; pass them, or give SPEAR-TTS a semantic tokenizer (a W2vBert with a " +
            "fitted tokenizer).")).SemanticTokens(prompt);
        var tensor = new Tensor<T>(new[] { tokens.Length });
        for (int i = 0; i < tokens.Length; i++) tensor[i] = NumOps.FromDouble(tokens[i]);
        return new TtsVoice<T> { ReferenceAudio = prompt, ReferenceTokens = tensor };
    }

    private int UpsampleFactor => CodecFrameRate / _options.SemanticFrameRate;

    // ---------------------------------------------------------------- training

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // A token/mel pair does not carry the supervision this model trains on; Train(TtsTrainingSample) does.
        ThrowIfTokenMelTrainingUnsupported();
        return false;
    }

    /// <summary>
    /// Copies the pretrained S1 into the backtranslation model (§4.2: "the same pretrained checkpoint P is finetuned
    /// for different purposes"). Call it after <see cref="SPEARTTSTrainingStage.Pretraining"/> and before training
    /// <see cref="SPEARTTSTrainingStage.Backtranslation"/> or <see cref="SPEARTTSTrainingStage.Reading"/>, which would
    /// otherwise change S1 first.
    /// </summary>
    public void StartBacktranslationFromPretrained()
    {
        ThrowIfDisposed();
        var source = Training.TapeTrainingStep<T>.CollectParameters(_readingLayers.Cast<ILayer<T>>().ToList(), -1);
        var target = Training.TapeTrainingStep<T>.CollectParameters(_backtranslationLayers.Cast<ILayer<T>>().ToList(), -1);
        if (source.Count != target.Count)
            throw new InvalidOperationException("S1 and the backtranslation model must have the same parameters.");
        for (int i = 0; i < source.Count; i++)
        {
            if (!source[i].Shape.ToArray().SequenceEqual(target[i].Shape.ToArray()))
                throw new InvalidOperationException("S1 and the backtranslation model must have the same parameter shapes.");
            source[i].AsSpan().CopyTo(target[i].AsWritableSpan());
            Engine.InvalidatePersistentTensor(target[i]);
        }
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

    // The utterance's codes [1 + Q, frames]: the semantic token at the codec's frame rate, then the acoustic levels.
    private int[,] CodesOf(TtsTrainingSample<T> sample, bool needAcoustic)
    {
        int codebooks = CodecTokenCodebooks;
        var tokens = sample.CodecTokens ?? throw new ArgumentException(
            "SPEAR-TTS trains on codec tokens [frames, 1 + Q]: the semantic token at the codec's frame rate, then the Q codes " +
            "(or [frames, 1] of semantic tokens, with the recording when the stage needs acoustic tokens).", nameof(sample));
        if (tokens.Rank != 2 || (tokens.Shape[1] != codebooks && tokens.Shape[1] != 1))
            throw new ArgumentException($"Expected codec tokens [frames, {codebooks}], or [frames, 1] of semantic tokens.", nameof(sample));
        int frames = tokens.Shape[0];
        var codes = new int[codebooks, frames];
        for (int f = 0; f < frames; f++)
        {
            codes[0, f] = (int)Math.Round(NumOps.ToDouble(tokens[f, 0]));
            if (codes[0, f] < 0 || codes[0, f] >= _options.SemanticCodes)
                throw new ArgumentOutOfRangeException(nameof(sample), $"Semantic tokens lie in [0, {_options.SemanticCodes}).");
        }
        if (tokens.Shape[1] == codebooks)
        {
            for (int f = 0; f < frames; f++)
                for (int q = 1; q < codebooks; q++)
                {
                    codes[q, f] = (int)Math.Round(NumOps.ToDouble(tokens[f, q]));
                    if (codes[q, f] < 0 || codes[q, f] >= _options.CodebookSize)
                        throw new ArgumentOutOfRangeException(nameof(sample), $"Codes lie in [0, {_options.CodebookSize}).");
                }
            return codes;
        }
        if (!needAcoustic) return codes;
        var audio = sample.Audio ?? throw new ArgumentException("Semantic tokens alone need the recording for the acoustic codes.", nameof(sample));
        var acoustic = Codec.Encode(audio);
        int levels = acoustic.GetLength(0), common = Math.Min(frames, acoustic.GetLength(1));
        if (levels < _options.NumCodebooks)
            throw new ArgumentException($"The codec gave {levels} levels; SPEAR-TTS speaks {_options.NumCodebooks}.", nameof(sample));
        var aligned = new int[codebooks, common];
        for (int f = 0; f < common; f++)
        {
            aligned[0, f] = codes[0, f];
            for (int q = 0; q < _options.NumCodebooks; q++) aligned[q + 1, f] = acoustic[q, f];
        }
        return aligned;
    }

    // The semantic tokens at their own rate (every UpsampleFactor-th codec frame) with consecutive repetitions removed.
    private int[] SemanticOf(int[,] codes)
    {
        int frames = codes.GetLength(1), factor = UpsampleFactor;
        var semantic = new List<int>();
        for (int f = 0; f < frames; f += factor) semantic.Add(codes[0, f]);
        return AudioLmCore<T>.Deduplicate(semantic);
    }

    // The text's ids without </s> and padding: Tokens is PreprocessText's output, or phoneme ids alone.
    private int[] TextOf(TtsTrainingSample<T> sample)
    {
        var ids = new List<int>();
        for (int i = 0; i < sample.Tokens.Length; i++)
        {
            int id = (int)Math.Round(NumOps.ToDouble(sample.Tokens[i]));
            if (id == PadToken || id == EndToken) continue;
            if (id < PhonemeOffset || id >= SemanticOffset)
                throw new ArgumentOutOfRangeException(nameof(sample), $"Text ids lie in [{PhonemeOffset}, {SemanticOffset}).");
            ids.Add(id);
        }
        if (ids.Count == 0) throw new ArgumentException("The sample's text has no phonemes.", nameof(sample));
        return ids.ToArray();
    }

    private Tensor<T> Ids(IReadOnlyList<int> ids)
    {
        var tensor = new Tensor<T>(new[] { ids.Count });
        for (int i = 0; i < ids.Count; i++) tensor[i] = NumOps.FromDouble(ids[i]);
        return tensor;
    }

    private int[] WithEnd(IEnumerable<int> ids) => ids.Concat(new[] { EndToken }).ToArray();

    /// <summary>
    /// The pretraining task's input (§4.1, §7.2): the semantic ids with every token deleted independently with
    /// probability <see cref="SPEARTTSOptions.PretrainingDeletionProbability"/>, then <c>&lt;/s&gt;</c>. The target is
    /// the whole uncorrupted sequence.
    /// </summary>
    internal int[] CorruptForPretraining(IReadOnlyList<int> semanticIds, Random random)
    {
        var kept = new List<int>(semanticIds.Count);
        foreach (int id in semanticIds)
            if (random.NextDouble() >= _options.PretrainingDeletionProbability) kept.Add(id);
        return WithEnd(kept);
    }

    private Func<Tensor<T>, Tensor<T>, Tensor<T>> BuildObjective(TtsTrainingSample<T> sample, Random random, bool training)
    {
        Guard.NotNull(sample);
        if (!HasPaperLayers)
            throw new NotSupportedException("The paper objective needs the paper's layers; this model was built from caller-supplied layers.");
        var o = _options;
        var codes = CodesOf(sample, needAcoustic: CurrentStage == SPEARTTSTrainingStage.Speaking);
        var semanticIds = SemanticOf(codes).Select(code => SemanticOffset + code).ToArray();

        switch (CurrentStage)
        {
            case SPEARTTSTrainingStage.Pretraining:
            {
                Reading.SetDropout(o.PretrainingDropoutRate);
                var input = Ids(CorruptForPretraining(semanticIds, random));
                var labels = WithEnd(semanticIds);
                // Label smoothing is used "except in the case of pretraining" (§7.2).
                return (_, _) => Reading.Loss(input, labels, training, random);
            }
            case SPEARTTSTrainingStage.Reading:
            {
                Reading.SetDropout(o.DropoutRate);
                var input = Ids(WithEnd(TextOf(sample)));
                var labels = WithEnd(semanticIds);
                return (_, _) => Reading.Loss(input, labels, training, random, labelSmoothing: o.LabelSmoothing);
            }
            case SPEARTTSTrainingStage.Backtranslation:
            {
                BacktranslationModel.SetDropout(o.DropoutRate);
                var input = Ids(WithEnd(semanticIds));
                var labels = WithEnd(TextOf(sample));
                return (_, _) => BacktranslationModel.Loss(input, labels, training, random, labelSmoothing: o.LabelSmoothing);
            }
            default:
            {
                var (ids, firstTarget) = SpeakingSequence(codes, random);
                return (_, _) => Speaking.Loss(ids, firstTarget, training, random);
            }
        }
    }

    /// <summary>
    /// S2's training sequence (§5, Figure 3) and its first target position: a prompt window of
    /// <see cref="SPEARTTSOptions.PromptSeconds"/> (at most half the utterance) at a random place, and as the target
    /// the longer stretch of the utterance on either side of it; then
    /// <c>BOS z_prompt z_target SEP y_prompt y_target EOS</c>, the semantic tokens deduplicated per window and the
    /// acoustic tokens interleaved per frame. The model predicts <c>y_target</c> and the end.
    /// </summary>
    internal (List<int> Ids, int FirstTarget) SpeakingSequence(int[,] codes, Random random)
    {
        int frames = codes.GetLength(1);
        if (frames < 2)
            throw new ArgumentException("S2 trains on two non-overlapping windows; the utterance needs at least two frames.", nameof(codes));
        int promptFrames = Math.Max(1, Math.Min((int)Math.Round(_options.PromptSeconds * CodecFrameRate), frames / 2));
        int promptStart = random.Next(frames - promptFrames + 1);
        int leftFrames = promptStart, rightFrames = frames - promptStart - promptFrames;
        int targetStart = rightFrames >= leftFrames ? promptStart + promptFrames : 0;
        int targetFrames = Math.Max(leftFrames, rightFrames);

        var ids = new List<int> { SpeakingBegin };
        ids.AddRange(WindowSemantic(codes, promptStart, promptFrames));
        ids.AddRange(WindowSemantic(codes, targetStart, targetFrames));
        ids.Add(SpeakingSeparator);
        AppendAcoustic(ids, codes, 1, promptStart, promptFrames);
        int firstTarget = ids.Count;
        AppendAcoustic(ids, codes, 1, targetStart, targetFrames);
        ids.Add(SpeakingEnd);
        return (ids, firstTarget);
    }

    // A window's semantic tokens (from codec-rate semantic row 0) at their own rate, deduplicated.
    private int[] WindowSemantic(int[,] codes, int start, int frames)
    {
        var semantic = new List<int>();
        for (int f = start; f < start + frames; f += UpsampleFactor) semantic.Add(codes[0, f]);
        return AudioLmCore<T>.Deduplicate(semantic);
    }

    // The window's acoustic tokens, frame by frame and level by level, level q offset by q·N; rows firstRow.. hold levels.
    private void AppendAcoustic(List<int> ids, int[,] codes, int firstRow, int start, int frames)
    {
        int n = _options.CodebookSize;
        for (int f = start; f < start + frames; f++)
            for (int q = 0; q < _options.NumCodebooks; q++) ids.Add(AcousticOffset + q * n + codes[firstRow + q, f]);
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? StageOptimizer()
    {
        if (_suppliedOptimizer is not null)
            return _suppliedOptimizer;
        if (!_stageOptimizers.TryGetValue(CurrentStage, out var optimizer))
        {
            // Adafactor (Shazeer & Stern 2018) with its relative step size min(1e-2, 1/sqrt(step)): the inverse
            // square-root decay of §7.2. Each stage starts its own, as each finetuning starts from the checkpoint.
            optimizer = PaperOptimizerFactory.VerifyHandBuilt(this,
                new AdafactorOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdafactorOptimizerOptions<T, Tensor<T>, Tensor<T>>()),
                CurrentStage == SPEARTTSTrainingStage.Speaking ? "speaking" : "reading");
            _stageOptimizers[CurrentStage] = optimizer;
        }
        return optimizer;
    }

    /// <inheritdoc />
    /// <remarks>
    /// Pretraining updates all of S1. Reading updates the encoder's embeddings and its lower
    /// <see cref="SPEARTTSOptions.ReadingTrainableEncoderLayers"/> layers, freezing the upper encoder and the decoder
    /// (§7.2). Backtranslation updates the backtranslation model's top and bottom
    /// <see cref="SPEARTTSOptions.BacktranslationTrainableDecoderLayers"/> decoder layers and the decoder's embeddings,
    /// whose text rows the semantic-only pretraining never trained. A stack's relative position bias table is its first
    /// block's (as T5 stores it), so it trains with that block. Speaking updates S2.
    /// </remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var o = _options;
        var trainable = new List<LayerBase<T>>();
        switch (CurrentStage)
        {
            case SPEARTTSTrainingStage.Pretraining:
                trainable.AddRange(_readingLayers);
                break;
            case SPEARTTSTrainingStage.Reading:
                trainable.Add(Reading.EncoderEmbedding ?? throw new InvalidOperationException("S1's encoder reads its own embeddings."));
                foreach (var block in Reading.Encoder.Take(o.ReadingTrainableEncoderLayers)) trainable.AddRange(BlockLayers(block));
                // T5 keeps the relative position bias table in its first block, so the lower layers include it.
                if (o.ReadingTrainableEncoderLayers > 0) trainable.Add(Reading.EncoderBias);
                break;
            case SPEARTTSTrainingStage.Backtranslation:
            {
                var model = BacktranslationModel;
                var decoder = model.Decoder;
                int n = o.BacktranslationTrainableDecoderLayers;
                trainable.Add(model.Shared);
                for (int i = 0; i < decoder.Count; i++)
                    if (i < n || i >= decoder.Count - n) trainable.AddRange(BlockLayers(decoder[i]));
                // The decoder's bias table belongs to its first block, a bottom layer.
                if (n > 0 && model.DecoderBias is { } bias) trainable.Add(bias);
                break;
            }
            default:
                trainable.AddRange(_speakingLayers);
                break;
        }
        var selected = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(trainable.Cast<ILayer<T>>().ToList(), -1),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(selected.Contains).ToList();
    }

    private static IEnumerable<LayerBase<T>> BlockLayers(T5Block<T> block)
    {
        yield return block.SelfNorm;
        foreach (var layer in AttentionLayers(block.SelfAttention)) yield return layer;
        if (block.CrossNorm is not null) yield return block.CrossNorm;
        if (block.CrossAttention is not null)
            foreach (var layer in AttentionLayers(block.CrossAttention)) yield return layer;
        yield return block.FeedForwardNorm;
        yield return block.Wi;
        if (block.WiLinear is not null) yield return block.WiLinear;
        yield return block.Wo;
    }

    private static IEnumerable<LayerBase<T>> AttentionLayers(T5Attention<T> attention)
    {
        yield return attention.Query;
        yield return attention.Key;
        yield return attention.Value;
        yield return attention.Output;
    }

    // ---------------------------------------------------------------- reading and backtranscription

    /// <summary>S1's semantic tokens (deduplicated, 25 Hz) for text ids, by beam search (§7.4).</summary>
    public int[] ReadText(Tensor<T> textIds)
    {
        ThrowIfDisposed();
        if (textIds is null) throw new ArgumentNullException(nameof(textIds));
        using var _ = new NoGradScope<T>();
        var o = _options;
        var memory = Reading.Encode(textIds, training: false, new Random(0));
        int offset = SemanticOffset;
        // S1 emits semantic tokens and </s>, and </s> only after the first token.
        return Reading.BeamSearch(memory, null, Array.Empty<int>(), o.BeamSize, o.MaxSemanticTokens,
                (token, step) => token >= offset || (token == EndToken && step > 0))
            .Where(id => id >= offset).Select(id => id - offset).ToArray();
    }

    /// <summary>
    /// Backtranslation (§4.2): the phonemes the backtranslation model reads out of an utterance's semantic tokens
    /// (25 Hz; repetitions are removed here), by the same beam search as reading. Pairs of these and the semantic
    /// tokens are the synthetic parallel data for <see cref="SPEARTTSTrainingStage.Reading"/>.
    /// </summary>
    public string[] Backtranscribe(IReadOnlyList<int> semanticTokens)
    {
        ThrowIfDisposed();
        if (semanticTokens is null) throw new ArgumentNullException(nameof(semanticTokens));
        if (semanticTokens.Any(code => code < 0 || code >= _options.SemanticCodes))
            throw new ArgumentOutOfRangeException(nameof(semanticTokens), $"Semantic tokens lie in [0, {_options.SemanticCodes}).");
        using var _ = new NoGradScope<T>();
        var input = Ids(WithEnd(AudioLmCore<T>.Deduplicate(semanticTokens).Select(code => SemanticOffset + code)));
        var memory = BacktranslationModel.Encode(input, training: false, new Random(0));
        int offset = SemanticOffset;
        var ids = BacktranslationModel.BeamSearch(memory, null, Array.Empty<int>(), _options.BeamSize, _options.MaxTextLength,
            (token, step) => (token >= PhonemeOffset && token < offset) || (token == EndToken && step > 0));
        return DecodePhonemes(ids);
    }

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    /// <remarks>
    /// <paramref name="input"/> is the text as S1's phoneme ids (<see cref="PreprocessText"/>); the output is the
    /// waveform <c>[samples]</c>. S1 reads the text with beam search; S2 samples the acoustic tokens at
    /// <see cref="SPEARTTSOptions.Temperature"/>, prompted by <see cref="TtsModelBase{T}.Voice"/> (its first
    /// <see cref="SPEARTTSOptions.PromptSeconds"/> and their semantic tokens). Without a voice the prompt is empty and the
    /// voice is sampled (§5: a prefix of semantic tokens alone gives a random voice).
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        if (!HasPaperLayers) return base.PredictCore(input);
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var o = _options;
            var semantic = ReadText(input);
            if (semantic.Length == 0)
                throw new InvalidOperationException("S1 read no semantic tokens from the text.");

            int[] promptSemantic = Array.Empty<int>();
            int[,] promptCodes = new int[o.NumCodebooks, 0];
            if (Voice is { } voice)
            {
                var recording = voice.ReferenceAudio ?? throw new ArgumentException("A SPEAR-TTS voice needs its recording.");
                var tokens = voice.ReferenceTokens ?? throw new ArgumentException("A SPEAR-TTS voice needs its semantic tokens at 25 Hz.");
                promptSemantic = AudioLmCore<T>.Deduplicate(Enumerable.Range(0, tokens.Length)
                    .Select(i => (int)Math.Round(NumOps.ToDouble(tokens[i]))).ToArray());
                if (promptSemantic.Any(code => code < 0 || code >= o.SemanticCodes))
                    throw new ArgumentOutOfRangeException(nameof(input), $"Semantic tokens lie in [0, {o.SemanticCodes}).");
                var codes = Codec.Encode(recording);
                int frames = Math.Min(codes.GetLength(1), (int)Math.Round(o.PromptSeconds * CodecFrameRate));
                promptCodes = new int[o.NumCodebooks, frames];
                for (int q = 0; q < o.NumCodebooks; q++)
                    for (int f = 0; f < frames; f++) promptCodes[q, f] = codes[q, f];
            }

            var prefix = new List<int> { SpeakingBegin };
            prefix.AddRange(promptSemantic);
            prefix.AddRange(semantic);
            prefix.Add(SpeakingSeparator);
            AppendAcoustic(prefix, promptCodes, 0, 0, promptCodes.GetLength(1));

            int levels = o.NumCodebooks, n = o.CodebookSize, offset = AcousticOffset, end = SpeakingEnd;
            // Each step emits the next level's code; the end may close a frame, but not before the first (ending at once
            // would render no audio).
            var generated = Speaking.Generate(prefix, o.Temperature, o.MaxAcousticFrames * levels, (step, v) =>
            {
                int level = step % levels;
                return (v >= offset + level * n && v < offset + (level + 1) * n) || (level == 0 && step > 0 && v == end);
            }, end, random);
            int count = generated.Count / levels;
            if (count == 0) throw new InvalidOperationException("S2 generated no acoustic frames.");
            var target = new int[levels, count];
            for (int f = 0; f < count; f++)
                for (int q = 0; q < levels; q++) target[q, f] = generated[f * levels + q] - offset - q * n;
            var audio = Codec.Decode(target);
            return audio.Reshape(new[] { audio.Length });
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    /// <summary>The codec's first <see cref="NumCodebooks"/> levels of a recording, <c>[frames, Q]</c>.</summary>
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(audio);
        var codes = Codec.Encode(audio);
        int levels = Math.Min(codes.GetLength(0), _options.NumCodebooks), frames = codes.GetLength(1);
        var tokens = new Tensor<T>(new[] { frames, levels });
        for (int f = 0; f < frames; f++)
            for (int q = 0; q < levels; q++) tokens[f, q] = NumOps.FromDouble(codes[q, f]);
        return tokens;
    }

    /// <summary>The waveform of codec codes <c>[frames, Q]</c>.</summary>
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(tokens);
        if (tokens is null) throw new ArgumentNullException(nameof(tokens));
        if (tokens.Rank != 2)
            throw new ArgumentException("Expected codec tokens [frames, levels].", nameof(tokens));
        int frames = tokens.Shape[0], levels = tokens.Shape[1];
        var codes = new int[levels, frames];
        for (int f = 0; f < frames; f++)
            for (int q = 0; q < levels; q++) codes[q, f] = (int)Math.Round(NumOps.ToDouble(tokens[f, q]));
        var audio = Codec.Decode(codes);
        return audio.Reshape(new[] { audio.Length });
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "SPEARTTS-Native" : "SPEARTTS-ONNX",
            Description = "Speak, Read and Prompt: High-Fidelity Text-to-Speech with Minimal Supervision (Kharitonov et al., 2023)",
            FeatureCount = _options.HiddenDim,
            Complexity = _options.NumEncoderLayers + _options.NumDecoderLayers + _options.SpeakingLayers,
        };
        metadata.AdditionalInfo["Architecture"] = "SPEAR-TTS";
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(SPEARTTS<T>));
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
