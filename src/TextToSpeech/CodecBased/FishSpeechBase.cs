using System.Runtime.CompilerServices;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.FrontEnd;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// The Fish Speech models (Liao et al. 2024): a Dual-AR transformer that speaks Firefly-GAN's codes after the text, and
/// the codec that turns them into 44.1 kHz audio. The releases differ in their tokenizer, in how a frame's codes appear in
/// the sequence (1.5: the frame's <c>&lt;|semantic:c0|&gt;</c> token; 1.4: one <c>&lt;|semantic|&gt;</c> token, codes plus
/// one), in the system prompt (1.5's "Speak out the provided text.") and in their text cleaning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><see cref="Synthesize"/> runs the release's <c>generate_long</c>: the text cleaned and split into segments of at
/// most <see cref="FishSpeechFamilyOptions.ChunkLength"/> bytes, each generated after the references and the earlier
/// segments that fit the context, its codes decoded by Firefly-GAN and the audio concatenated. A voice from
/// <see cref="CreateVoice(Tensor{T}, int, string)"/> clones the speaker of one or more transcribed recordings.</para>
/// <para>Without a tokenizer file (<see cref="FishSpeechFamilyOptions.TokenizerPath"/> or
/// <see cref="LoadReferenceWeights"/>'s directory) the model reads text as UTF-8 bytes, one token each, with the release's
/// special tokens beside them, a vocabulary for training from scratch.</para>
/// </remarks>
public abstract partial class FishSpeechBase<T> : TtsModelBase<T>, ICodecTts<T>, IPaperOptimizerVariant
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    private readonly FishSpeechFamilyOptions _options;
    private readonly FishSpeechVersion _version;
    private readonly bool _useNativeMode;
    private bool _disposed;

    private FishDualAr<T>? _net;
    private FishFireflyCodec<T>? _codec;
    private readonly List<LayerBase<T>> _netLayers = new();
    private readonly List<LayerBase<T>> _codecLayers = new();
    private FishTextTokenizer? _tokenizer;
    private FishSpeechFormat? _format;

    private protected FishSpeechBase(NeuralNetworkArchitecture<T> architecture, FishSpeechFamilyOptions options, FishSpeechVersion version)
        : base(architecture)
    {
        _options = options;
        _version = version;
        Options = options;
        _useNativeMode = true;
        ConfigureBase();
        InitializeLayers();
    }

    private protected FishSpeechBase(NeuralNetworkArchitecture<T> architecture, string modelPath, FishSpeechFamilyOptions options, FishSpeechVersion version)
        : base(architecture)
    {
        _options = options;
        _version = version;
        Options = options;
        _useNativeMode = false;
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        var o = _options;
        if (o.HiddenDim % o.NumHeads != 0 || o.NumHeads % o.KeyValueHeads != 0)
            throw new ArgumentException("HiddenDim must divide into NumHeads and NumHeads into KeyValueHeads.");
        // The codec's codes must fit the transformer's codebooks (1.4 stores them plus one).
        int codes = o.CodecLevels.Aggregate(1, (a, b) => a * b) + (_version == FishSpeechVersion.V14 ? 1 : 0);
        if (codes > o.CodebookSize)
            throw new ArgumentException($"The quantizer's levels give {codes} codebook entries, more than CodebookSize ({o.CodebookSize}).");
        if (o.ChunkLength < 0 || o.MaxNewTokens < 0) throw new ArgumentException("ChunkLength and MaxNewTokens are non-negative.");
        base.SampleRate = o.SampleRate;
        base.MelChannels = o.MelChannels;
        base.HopSize = o.HopSize;
        base.HiddenDim = o.HiddenDim;
    }

    /// <inheritdoc />
    /// <remarks>The training recipe's name (<see cref="FishSpeechFamilyOptions.Recipe"/>).</remarks>
    public string? PaperOptimizerVariant => _options.Recipe.ToString();

    private bool HasPaperLayers => _net is not null;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");

    internal FishDualAr<T> Transformer => _net ?? throw NotNative();
    internal FishFireflyCodec<T> Codec => _codec ?? throw NotNative();

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperLayers ? LayerInputDomain.Indices(Tokenizer.Size) : base.GetInputDomain(inputShape);

    internal FishDualArConfiguration TransformerConfiguration()
    {
        var o = _options;
        return new FishDualArConfiguration(o.VocabSize, o.HiddenDim, o.NumDecoderLayers, o.NumHeads, o.KeyValueHeads, o.IntermediateSize, o.FastLayers,
            o.CodebookSize, o.NumCodebooks, o.MaxSequenceLength, o.RopeBase, o.NormEpsilon, o.DropoutRate);
    }

    internal FishFireflyConfiguration CodecConfiguration()
    {
        var o = _options;
        return new FishFireflyConfiguration(o.SampleRate, o.FftSize, o.HopSize, o.MelChannels, o.CodecEncoderDepths, o.CodecEncoderDims,
            o.CodecQuantizerDim, o.NumCodebooks, o.CodecLevels, o.CodecDownsampleFactors, o.CodecUpsampleRates, o.CodecUpsampleKernels,
            o.CodecResBlockKernels, o.CodecResBlockDilations, o.CodecChannels, o.CodecPreKernel, o.CodecPostKernel);
    }

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            AddUnsplitLayers(Architecture.Layers);
            return;
        }
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? _options.TrainingSeed);
        _net = new FishDualAr<T>(Engine, _netLayers, TransformerConfiguration(), random);
        _codec = new FishFireflyCodec<T>(Engine, _codecLayers, CodecConfiguration(), random);
        // The Dual-AR transformer trains; Firefly-GAN is trained on its own (a GAN stage) and is frozen here.
        AddUnsplitLayers(_netLayers.Cast<ILayer<T>>().ToList());
        ComponentLayers.AddRange(_codecLayers);
    }

    // ---------------------------------------------------------------- text

    internal FishTextTokenizer Tokenizer
    {
        get
        {
            if (_tokenizer is null)
            {
                var tokenizer = _options.TokenizerPath is { } path
                    ? FishTextTokenizer.Release(_version, path, _options.CodebookSize)
                    : FishTextTokenizer.Bytes(_version, _options.CodebookSize);
                if (tokenizer.Size > _options.VocabSize)
                    throw new InvalidOperationException($"The tokenizer uses {tokenizer.Size} ids but VocabSize is {_options.VocabSize}.");
                _tokenizer = tokenizer;
            }
            return _tokenizer;
        }
    }

    internal FishSpeechFormat Format => _format ??= new FishSpeechFormat(_version, Tokenizer.Encode, Tokenizer.TokenId, _options.NumCodebooks, _options.CodebookSize);

    private void ResetTokenizer()
    {
        _tokenizer = null;
        _format = null;
    }

    /// <summary>The release's <c>clean_text</c> for this version.</summary>
    public string CleanText(string text) => FishSpeechText.Clean(text ?? throw new ArgumentNullException(nameof(text)), _version);

    /// <summary>The segments <see cref="Synthesize"/> speaks in turn (<c>split_text</c>, after the Chinese number
    /// normalizer when <see cref="FishSpeechFamilyOptions.NormalizeChineseText"/> is set).</summary>
    public IReadOnlyList<string> SplitText(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (_options.NormalizeChineseText) text = ChineseNswNormalizer.Normalize(text, fishSpeech: true);
        return _options.ChunkLength > 0 ? FishSpeechText.Split(text, _options.ChunkLength, _version) : new List<string> { text };
    }

    /// <summary>The token ids of text as the user's turn reads it (cleaned, then tokenized).</summary>
    public int[] Tokenize(string text) => Tokenizer.Encode(CleanText(text));

    /// <summary>Decodes token ids to text.</summary>
    public string Detokenize(IEnumerable<int> ids) => Tokenizer.Decode(ids ?? throw new ArgumentNullException(nameof(ids)));

    /// <inheritdoc />
    /// <remarks>The cleaned text's token ids.</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        var ids = Tokenize(text);
        if (ids.Length == 0) throw new ArgumentException("The text is empty once cleaned.", nameof(text));
        return IndexTtsOps.Ids<T>(ids);
    }

    private int[] TextIds(Tensor<T> tensor, string name)
    {
        var ids = new int[tensor.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = (int)Math.Round(Ops.ToDouble(tensor.GetFlat(i)));
            if (ids[i] < 0 || ids[i] >= Tokenizer.Size)
                throw new ArgumentOutOfRangeException(name, $"Token ids lie in [0, {Tokenizer.Size}).");
        }
        return ids;
    }

    // ---------------------------------------------------------------- audio and codes

    /// <summary>A recording at the codec's rate (<c>load_audio</c>: torchaudio's resampling).</summary>
    internal Tensor<T> AtModelRate(Tensor<T> recording, int sampleRate)
    {
        var mono = Engine.Reshape(recording, new[] { recording.Length });
        return sampleRate == _options.SampleRate ? mono : AudioHelper<T>.ResampleBandLimited(Engine, mono, sampleRate, _options.SampleRate);
    }

    /// <summary>Firefly-GAN's codes <c>[codebooks, frames]</c> of audio at the model's rate.</summary>
    public int[,] EncodeAudio(Tensor<T> audio)
    {
        ThrowIfDisposed();
        using var _ = new NoGradScope<T>();
        foreach (var layer in _codecLayers) layer.SetTrainingMode(false);
        return Codec.Encode(Engine.Reshape(audio ?? throw new ArgumentNullException(nameof(audio)), new[] { audio.Length }));
    }

    /// <summary>The audio Firefly-GAN decodes from codes <c>[codebooks, frames]</c>.</summary>
    public Tensor<T> DecodeAudio(int[,] codes)
    {
        ThrowIfDisposed();
        if (codes is null) throw new ArgumentNullException(nameof(codes));
        if (codes.GetLength(1) == 0) return new Tensor<T>(new[] { 0 });
        using var _ = new NoGradScope<T>();
        foreach (var layer in _codecLayers) layer.SetTrainingMode(false);
        return Codec.Decode(codes);
    }

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    public int NumCodebooks => _options.NumCodebooks;

    /// <inheritdoc />
    public int CodebookSize => _options.CodebookSize;

    /// <inheritdoc />
    /// <remarks>Rounded: one frame per hop times the quantizer's downsampling (44,100 / 2,048 ≈ 21.5 Hz in the
    /// release).</remarks>
    public int CodecFrameRate =>
        (int)Math.Round(_options.SampleRate / (double)(_options.HopSize * _options.CodecDownsampleFactors.Aggregate(1, (a, b) => a * b)));

    /// <inheritdoc />
    /// <remarks>Firefly-GAN's codes <c>[frames, codebooks]</c> of audio at the model's rate.</remarks>
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        var codes = EncodeAudio(audio);
        var tokens = new Tensor<T>(new[] { codes.GetLength(1), codes.GetLength(0) });
        for (int t = 0; t < codes.GetLength(1); t++)
            for (int b = 0; b < codes.GetLength(0); b++) tokens[t, b] = Ops.FromDouble(codes[b, t]);
        return tokens;
    }

    /// <inheritdoc />
    /// <remarks>The audio Firefly-GAN decodes from codes <c>[frames, codebooks]</c>.</remarks>
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        if (tokens is null) throw new ArgumentNullException(nameof(tokens));
        if (tokens.Rank != 2 || tokens.Shape[1] != _options.NumCodebooks)
            throw new ArgumentException($"Codec tokens are [frames, {_options.NumCodebooks}].", nameof(tokens));
        var codes = new int[tokens.Shape[1], tokens.Shape[0]];
        for (int t = 0; t < tokens.Shape[0]; t++)
            for (int b = 0; b < tokens.Shape[1]; b++) codes[b, t] = (int)Math.Round(Ops.ToDouble(tokens[t, b]));
        return DecodeAudio(codes);
    }

    // ---------------------------------------------------------------- voices

    private sealed class References
    {
        public readonly List<(object Text, int[,] Codes)> Items = new();
    }

    private readonly ConditionalWeakTable<TtsVoice<T>, References> _voices = new();

    /// <summary>A voice cloned from a recording of the speaker at <paramref name="sampleRate"/> and what it says.</summary>
    public TtsVoice<T> CreateVoice(Tensor<T> recording, int sampleRate, string transcript) =>
        CreateVoice(new[] { (recording, sampleRate, transcript) });

    /// <summary>A voice cloned from several transcribed recordings of the speaker (the release's references: each one's
    /// text and codes precede the text to speak).</summary>
    public TtsVoice<T> CreateVoice(IEnumerable<(Tensor<T> Recording, int SampleRate, string Transcript)> references)
    {
        ThrowIfDisposed();
        if (references is null) throw new ArgumentNullException(nameof(references));
        if (!HasPaperLayers) throw NotNative();
        var entries = new References();
        Tensor<T>? first = null;
        foreach (var (recording, sampleRate, transcript) in references)
        {
            if (recording is null) throw new ArgumentException("A reference has no recording.", nameof(references));
            if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(references), "A reference's sample rate is positive.");
            if (transcript is null) throw new ArgumentException("A reference has no transcript.", nameof(references));
            var audio = AtModelRate(recording, sampleRate);
            first ??= audio;
            entries.Items.Add((transcript, EncodeAudio(audio)));
        }
        if (first is null) throw new ArgumentException("There are no references.", nameof(references));
        var voice = new TtsVoice<T> { ReferenceAudio = first };
        _voices.Add(voice, entries);
        return voice;
    }

    // The voice's references: those CreateVoice made, or a voice's own recording (at the model's rate) with its
    // transcript's token ids (ReferenceTokens).
    private List<(object Text, int[,] Codes)> ReferencesOf(TtsVoice<T>? voice)
    {
        if (voice is null) return new List<(object, int[,])>();
        if (_voices.TryGetValue(voice, out var entries)) return entries.Items;
        var audio = voice.ReferenceAudio ?? throw new InvalidOperationException($"{GetType().Name} clones a recording; build the voice with CreateVoice.");
        var tokens = voice.ReferenceTokens ?? throw new InvalidOperationException("The voice's recording needs its transcript; build the voice with CreateVoice or set ReferenceTokens.");
        return new List<(object, int[,])> { (TextIds(tokens, nameof(voice)), EncodeAudio(audio)) };
    }

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    /// <remarks>Speech at 44.1 kHz in the voice's speaker (<see cref="TtsModelBase{T}.Voice"/>), or in a voice of the
    /// model's choosing without one.</remarks>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        if (!HasPaperLayers) return base.Synthesize(text);
        var segments = SplitText(text);
        if (segments.Count == 0) throw new ArgumentException("The text has nothing to speak.", nameof(text));
        return Inference(random => GenerateLong(segments.Cast<object>().ToList(), random));
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is the text's token ids, spoken as one segment.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        var ids = TextIds(input, nameof(input));
        if (ids.Length == 0) throw new ArgumentException("There are no tokens.", nameof(input));
        return Inference(random => GenerateLong(new List<object> { ids }, random));
    }

    private T2 Inference<T2>(Func<Random, T2> run)
    {
        var o = _options;
        if (!(o.TopP > 0 && o.TopP <= 1)) throw new InvalidOperationException("TopP must be in (0, 1].");
        if (!(o.RepetitionPenalty > 0 && o.RepetitionPenalty < 2)) throw new InvalidOperationException("RepetitionPenalty must be in (0, 2).");
        if (!(o.Temperature > 0 && o.Temperature < 2)) throw new InvalidOperationException("Temperature must be in (0, 2).");
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return run(AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed));
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    private int[,] SegmentOf(object text, int[,]? codes) =>
        text is string s ? Format.Segment(CleanText(s), codes) : Format.Segment((int[])text, codes);

    private static int[,] Concatenate(IReadOnlyList<int[,]> parts)
    {
        int rows = parts[0].GetLength(0), total = parts.Sum(p => p.GetLength(1)), at = 0;
        var joined = new int[rows, total];
        foreach (var p in parts)
        {
            for (int r = 0; r < rows; r++)
                for (int t = 0; t < p.GetLength(1); t++) joined[r, at + t] = p[r, t];
            at += p.GetLength(1);
        }
        return joined;
    }

    private static int[,] Columns(int[,] rows, int from, int to)
    {
        var part = new int[rows.GetLength(0), Math.Max(0, to - from)];
        for (int r = 0; r < rows.GetLength(0); r++)
            for (int t = from; t < to; t++) part[r, t - from] = rows[r, t];
        return part;
    }

    /// <summary>
    /// <c>generate_long</c>: each segment is generated after the references (1.5: and the system prompt) and as many earlier
    /// segments and their generated frames as the context budget allows, always keeping the first segment and its frames.
    /// </summary>
    internal Tensor<T> GenerateLong(IReadOnlyList<object> texts, Random random)
    {
        var o = _options;
        var format = Format;
        var references = ReferencesOf(Voice);
        var prompts = new List<int[,]>();
        if (format.SystemPrompt() is { } system) prompts.Add(system);
        foreach (var (text, codes) in references) prompts.Add(SegmentOf(text, codes));
        bool usePrompt = references.Count > 0;
        int budget = o.ContextLength - 1024 - prompts.Sum(p => p.GetLength(1));
        var history = new List<int[,]>();
        var audio = new List<Tensor<T>>();
        foreach (var text in texts)
        {
            history.Add(SegmentOf(text, null));
            // "Pick last 2000 tokens": count is advanced twice per segment, as the release does.
            int count = 0, i = 0;
            for (int k = history.Count - 1, step = 0; k >= 0; k--, step++)
            {
                i = step;
                int length = history[k].GetLength(1);
                count += length;
                if (count + length > budget) break;
            }
            if (i != 0 && i % 2 == 0) i--;
            // Python's global_encoded[-i:] with i = 0 is the whole list.
            var partial = i < history.Count - 2
                ? history.Take(2).Concat(i == 0 ? history : history.Skip(history.Count - i)).ToList()
                : history.ToList();
            if (usePrompt) partial = prompts.Concat(partial).ToList();
            var prompt = Concatenate(partial);
            int promptLength = prompt.GetLength(1);
            var y = FishSpeechGenerator<T>.Generate(Engine, Transformer, format, prompt, o.MaxNewTokens, o.Temperature, o.TopP, o.RepetitionPenalty, random,
                o.ConstrainToSemanticTokens);
            var kept = format.KeptCodes(y, promptLength);
            int end = _version == FishSpeechVersion.V15 ? y.GetLength(1) : y.GetLength(1) - 1;
            history.Add(Columns(y, promptLength, end));
            audio.Add(DecodeAudio(kept));
        }
        var waves = audio.Where(a => a.Length > 0).ToArray();
        if (waves.Length == 0) throw new InvalidOperationException("No audio was generated.");
        return waves.Length == 1 ? waves[0] : Engine.TensorConcatenate(waves, 0);
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <summary>The model's name in its metadata.</summary>
    private protected abstract string ModelName { get; }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = ModelName + (_useNativeMode ? "-Native" : "-ONNX"),
            Description = $"{ModelName}: a Dual-AR transformer over Firefly-GAN codes (Liao et al. 2024)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = ModelName;
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(FishSpeechBase<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
