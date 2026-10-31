using AiDotNet.Attributes;
using AiDotNet.Audio.Foundations;
using AiDotNet.Audio.Pitch;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.TextToSpeech.Vocoders;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>A piece of SpiritLM's input or output: text, or 16 kHz speech.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public sealed class SpiritLmContent<T>
{
    /// <summary>Text content.</summary>
    public static SpiritLmContent<T> FromText(string text) => new() { Text = text ?? throw new ArgumentNullException(nameof(text)) };

    /// <summary>Speech content (16 kHz).</summary>
    public static SpiritLmContent<T> FromSpeech(Tensor<T> audio) => new() { Audio = audio ?? throw new ArgumentNullException(nameof(audio)) };

    /// <summary>The text, when this is text.</summary>
    public string? Text { get; init; }

    /// <summary>The speech, when this is speech.</summary>
    public Tensor<T>? Audio { get; init; }

    /// <summary>The speech tokens (<c>[Hu…]</c>, with <c>[Pi…]</c> and <c>[St…]</c> for Expressive) of generated speech.</summary>
    public string? SpeechTokens { get; init; }

    /// <summary>Whether this is speech.</summary>
    public bool IsSpeech => Audio is not null || SpeechTokens is not null;
}

/// <summary>
/// SpiritLM (Nguyen et al., TACL 2025): an interleaved spoken and written language model. LLaMA 2 7B, continuously
/// pre-trained on text, speech and word-level interleaved speech-text sequences, reads and writes both modalities, each
/// span marked <c>[Text]</c> or <c>[Speech]</c>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>Speech tokens</b> (§3.1): the 25 Hz mHuBERT's layer-11 units, consecutive repeats merged, <c>[Hu0]</c> …
/// <c>[Hu500]</c>. SpiritLM Expressive (§3.2) adds pitch tokens (a VQ-VAE over FCPE's F0, 12.5 a second) and style tokens
/// (one a second), deduplicated and interleaved with the units by time.</para>
/// <para><b>Speech</b> is voiced by a unit HiFi-GAN trained on Expresso with a duration predictor (Base: 4 speakers and 34
/// styles; Expressive: the pitch and style tokens too).</para>
/// <para><b>Training</b> (§3.3, App. A): next-token prediction over text-only (<c>[Text]…</c>), speech-only
/// (<c>[Speech]…</c>) and aligned sequences whose modality switches at word boundaries, text spans of 10–30 words and
/// speech spans of 5–15, at a constant 3e-5 with RoPE's base at 100,000.</para>
/// <para><b>For Beginners:</b> SpiritLM writes text and speech in one stream: it can continue a recording as speech or as
/// text, or read text aloud, because it learned both "languages" at once.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var model = new SpiritLM&lt;float&gt;(architecture, new SpiritLMOptions());
/// model.LoadReleasedWeights("spiritlm_model/spirit-lm-base-7b", "speech_tokenizer");
/// var speech = model.Synthesize("The largest country in the world is");
/// var continuation = model.Generate(new[] { SpiritLmContent&lt;float&gt;.FromSpeech(recording16kHz) }, SpiritLmModality.Text);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "SpiRit-LM: Interleaved Spoken and Written Language Model",
    "https://arxiv.org/abs/2402.05755",
    Year = 2024,
    Authors = "Tu Anh Nguyen, Benjamin Muller, Bokai Yu, Marta R. Costa-jussa, Maha Elbayad, Sravya Popuri, Christophe Ropers, Paul-Ambroise Duquenne, Robin Algayres, Ruslan Mavlyutov, Itai Gat, Mary Williamson, Gabriel Synnaeve, Juan Pino, Benoit Sagot, Emmanuel Dupoux"
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 3e-5, Beta1 = 0.9, Beta2 = 0.95, WeightDecay = 0.1, MaxGradientNorm = 1.0,
                Source = "Nguyen et al. 2024 App. A: LLaMA 2 7B continued at its constant final learning rate of 3e-5. The optimizer "
                        + "is LLaMA 2's (Touvron et al. 2023): AdamW with beta2 0.95, epsilon 1e-5, weight decay 0.1 and clipping at 1.0.")]
[PaperOptimizer(OptimizerKind.AdamW, Component = "vocoder", LearningRate = 2e-4, Beta1 = 0.8, Beta2 = 0.99, WeightDecay = 0.01,
                Provenance = RecipeProvenance.ReleasedCode,
                Source = "The released vocoder configs (hifigan_spiritlm_base, hifigan_spiritlm_expressive_w2v2): speech-resynthesis' "
                        + "AdamW (0.8, 0.99) at 2e-4, decayed by 0.999 per epoch, for the vocoder and the pitch VQ-VAE.")]
public partial class SpiritLM<T> : TtsModelBase<T>
{
    private readonly SpiritLMOptions _options;
    private readonly bool _useNativeMode;
    private bool _disposed;
    private LlamaNetwork<T>? _lm;
    private HubertUnitExtractor<T>? _units;
    private FcpePitchEstimator<T>? _fcpe;
    private SpiritLmPitchQuantizer<T>? _pitchQuantizer;
    private SpiritLmStyleEncoder<T>? _styleEncoder;
    private CodeHiFiGan<T>? _vocoder;
    private HiFiGanDiscriminators<T>? _discriminators;
    private DifferentiableMel<T>? _lossMel;
    private SentencePieceModel? _tokenizer;
    private readonly List<LayerBase<T>> _lmLayers = new();
    private readonly List<LayerBase<T>> _hubertLayers = new();
    private readonly List<LayerBase<T>> _fcpeLayers = new();
    private readonly List<LayerBase<T>> _pitchLayers = new();
    private readonly List<LayerBase<T>> _styleLayers = new();
    private readonly List<LayerBase<T>> _vocoderLayers = new();
    private readonly List<LayerBase<T>> _discriminatorLayers = new();
    private Random? _samplingRandom;

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>Creates an ONNX-backed SpiritLM for inference.</summary>
    public SpiritLM(NeuralNetworkArchitecture<T> architecture, string modelPath, SpiritLMOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new SpiritLMOptions();
        Options = _options;
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        Validate(_options);
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable SpiritLM.</summary>
    public SpiritLM(NeuralNetworkArchitecture<T> architecture, SpiritLMOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new SpiritLMOptions();
        Options = _options;
        _useNativeMode = true;
        base.SampleRate = _options.SampleRate;
        base.HiddenDim = _options.HiddenDim;
        Validate(_options);
        InitializeLayers();
    }

    private static void Validate(SpiritLMOptions o)
    {
        if (o.HiddenDim <= 0 || o.Layers <= 0 || o.NumHeads <= 0 || o.HiddenDim % o.NumHeads != 0)
            throw new ArgumentException("The language model's width must divide into its heads.");
        if (o.Temperature < 0 || o.TopP <= 0 || o.TopP > 1) throw new ArgumentException("Sampling needs a temperature ≥ 0 and top-p in (0, 1].");
        if (o.MaxNewTextTokens <= 0 || o.MaxNewSpeechTokensFromText <= 0 || o.MaxNewSpeechTokensFromSpeech <= 0)
            throw new ArgumentException("The generation limits must be positive.");
        if (o.Speaker < 0 || o.Speaker >= o.VocoderSpeakers) throw new ArgumentException($"Speaker must lie in [0, {o.VocoderSpeakers}).");
        if (o.Variant == SpiritLmVariant.Base && o.VocoderStyles > 0 && (o.VocoderStyle < 0 || o.VocoderStyle >= o.VocoderStyles))
            throw new ArgumentException($"VocoderStyle must lie in [0, {o.VocoderStyles}).");
        if (o.VocoderUpsampleRates.Length != o.VocoderUpsampleKernels.Length) throw new ArgumentException("The vocoder needs one kernel per upsampling rate.");
        if (o.MinTextSpanWords < 1 || o.MaxTextSpanWords < o.MinTextSpanWords || o.MinSpeechSpanWords < 1 || o.MaxSpeechSpanWords < o.MinSpeechSpanWords)
            throw new ArgumentException("The interleaving spans need 1 ≤ minimum ≤ maximum words.");
    }

    /// <summary>The longest sequence the language model takes (4096).</summary>
    public int MaxTextLength => _options.MaxTextLength;

    /// <summary>Whether this is SpiritLM Expressive.</summary>
    public bool IsExpressive => _options.Variant == SpiritLmVariant.Expressive;

    /// <inheritdoc />
    /// <remarks>The vocoder speaks as <see cref="SpiritLMOptions.Speaker"/>; no recording is needed.</remarks>
    protected override TtsSupervision RequiredVoice => TtsSupervision.None;

    private bool HasPaperLayers => _lm is not null;
    private int SeedValue => _options.Seed ?? 42;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");
    internal LlamaNetwork<T> LanguageModel => _lm ?? throw NotNative();
    internal HubertUnitExtractor<T> UnitExtractor => _units ?? throw NotNative();
    internal CodeHiFiGan<T> Vocoder => _vocoder ?? throw NotNative();
    internal SpiritLmPitchQuantizer<T> PitchQuantizer => _pitchQuantizer ?? throw new InvalidOperationException("Pitch tokens are SpiritLM Expressive's.");
    internal FcpePitchEstimator<T>? Fcpe => _fcpe;
    internal SpiritLmStyleEncoder<T> StyleEncoder => _styleEncoder ?? throw new InvalidOperationException("Style tokens are SpiritLM Expressive's.");

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperLayers ? LayerInputDomain.Indices(_options.VocabSize) : base.GetInputDomain(inputShape);

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            AddUnsplitLayers(Architecture.Layers);
            return;
        }
        var o = _options;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? SeedValue);
        _lm = new LlamaNetwork<T>(Engine, _lmLayers, new LlamaConfiguration
        {
            VocabSize = o.VocabSize, Hidden = o.HiddenDim, Layers = o.Layers, Heads = o.NumHeads, KeyValueHeads = o.NumHeads,
            Intermediate = o.Intermediate, RmsEpsilon = o.RmsEpsilon, RopeTheta = o.RopeTheta,
        });
        // App. A: the new rows start random; a loaded checkpoint overwrites them all.
        _lm.InitializeLikeReference(random);

        // The 25 Hz mHuBERT: a fifth 2-stride convolution (640 samples a frame) and a batch-normed position convolution.
        var hubert = HubertUnitExtractor<T>.MHubertBase(o.HubertLayer) with
        {
            HiddenSize = o.HubertWidth, Heads = o.HubertHeads, IntermediateSize = o.HubertIntermediate,
            ConvDims = Enumerable.Repeat(o.HubertConvChannels, 8).ToArray(), ConvKernels = new[] { 10, 3, 3, 3, 3, 2, 2, 2 },
            ConvStrides = new[] { 5, 2, 2, 2, 2, 2, 2, 2 }, PositionalConvBatchNorm = true,
        };
        _units = new HubertUnitExtractor<T>(Engine, _hubertLayers, hubert, o.HubertLayer, o.Units) { MinChunk = 1280 };
        _units.Network.InitializeLikeReference(random);
        if (o.UnitQuantizer == SpiritLmUnitQuantizer.LinearCtc)
        {
            _units.Quantizer = new HubertLinearQuantizer<T>(Engine, _hubertLayers, o.HubertWidth, o.HubertWidth, o.Units);
            _units.Quantizer.InitializeLikeReference(random);
        }

        if (IsExpressive)
        {
            if (o.PitchExtractor == SpiritLmPitchExtractor.Fcpe)
                _fcpe = new FcpePitchEstimator<T>(Engine, _fcpeLayers, o.FcpeMels, o.FcpeWidth, o.FcpeLayers, o.FcpeBins);
            _pitchQuantizer = new SpiritLmPitchQuantizer<T>(Engine, _pitchLayers, new SpiritLmPitchQuantizerConfiguration
            {
                Codes = o.PitchTokens, Width = o.PitchQuantizerWidth, LatentDim = o.PitchQuantizerLatent,
            }, random);
            if (o.StyleTokenizer == SpiritLmStyleTokenizer.SpeechProp) BuildSpeechProp(random);
            else
            {
                _styleEncoder = new SpiritLmStyleEncoder<T>(Engine, _styleLayers, SpiritLmStyleEncoder<T>.Wav2Vec2Base() with
                {
                    HiddenSize = o.StyleEncoderWidth, Layers = o.StyleEncoderLayers, Heads = o.StyleEncoderHeads, IntermediateSize = o.StyleEncoderIntermediate,
                    ConvDims = Enumerable.Repeat(o.StyleEncoderConvChannels, 7).ToArray(),
                }, o.StyleEncoderProjection, o.StyleTokens);
                _styleEncoder.Network.InitializeLikeReference(random);
            }
        }

        _vocoder = new CodeHiFiGan<T>(Engine, _vocoderLayers, new CodeHiFiGanConfiguration
        {
            Units = o.Units + 1, EmbeddingDim = o.VocoderEmbeddingDim, InitialChannels = o.VocoderChannels,
            UpsampleRates = (int[])o.VocoderUpsampleRates.Clone(), UpsampleKernels = (int[])o.VocoderUpsampleKernels.Clone(),
            DurationHidden = o.VocoderEmbeddingDim, Speakers = o.VocoderSpeakers,
            Styles = IsExpressive ? 0 : o.VocoderStyles, PitchTokens = IsExpressive ? o.PitchTokens : 0, StyleTokens = IsExpressive ? o.StyleTokens : 0,
        }, random);
        if (o.IncludeVocoderDiscriminators)
        {
            _discriminators = new HiFiGanDiscriminators<T>(Engine, new[] { 2, 3, 5, 7, 11 }, 3);
            _discriminatorLayers.AddRange(_discriminators.Layers);
            // speech-resynthesis' mel loss: n_fft 1024, hop 256, window 1024, 80 bands, 0–8 kHz (the configs' fmax).
            _lossMel = new DifferentiableMel<T>(Engine, o.SampleRate, 1024, 256, 1024, 80, 0.0, 8000.0);
        }
        AddUnsplitLayers(_lmLayers.Cast<ILayer<T>>().ToList());
        // The speech tokenizers, the vocoder and its discriminators, so clones and saved models keep them.
        ComponentLayers.AddRange(_hubertLayers);
        ComponentLayers.AddRange(_fcpeLayers);
        ComponentLayers.AddRange(_pitchLayers);
        ComponentLayers.AddRange(_styleLayers);
        ComponentLayers.AddRange(_speechPropLayers);
        ComponentLayers.AddRange(_sonarDecoderLayers);
        ComponentLayers.AddRange(_sonarTextLayers);
        if (_speechPropCodec is not null) ComponentLayers.Add(new AudioCodecLayer<T>(_speechPropCodec));
        ComponentLayers.AddRange(_vocoderLayers);
        ComponentLayers.AddRange(_discriminatorLayers);
    }

    // ---------------------------------------------------------------- tokens

    /// <summary>The release's SentencePiece model (its user-defined pieces are the markers and speech tokens).</summary>
    internal SentencePieceModel Tokenizer
    {
        get
        {
            if (_tokenizer is not null) return _tokenizer;
            string path = _options.TokenizerPath ?? throw new InvalidOperationException("Text needs SpiritLM's tokenizer: set SpiritLMOptions.TokenizerPath.");
            if (!File.Exists(path)) throw new FileNotFoundException($"Tokenizer not found: {path}", path);
            _tokenizer = new SentencePieceModel(File.ReadAllBytes(path));
            if (_tokenizer.Count > _options.VocabSize)
                throw new InvalidOperationException($"The tokenizer has {_tokenizer.Count} pieces; the language model's vocabulary is {_options.VocabSize}.");
            return _tokenizer;
        }
    }

    /// <summary>The ids of a string as the release's <c>LlamaTokenizer</c> (legacy, <c>add_bos_token</c>) encodes it.</summary>
    public int[] Encode(string text, bool addBos = true)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        var ids = Tokenizer.Encode(text);
        return addBos ? new[] { Tokenizer.BosId }.Concat(ids).ToArray() : ids;
    }

    /// <summary>The string of ids (<c>&lt;s&gt;</c> and <c>&lt;/s&gt;</c> skipped).</summary>
    public string Decode(IEnumerable<int> ids) =>
        Tokenizer.Decode(ids.Where(id => id != Tokenizer.BosId && id != Tokenizer.EosId && id != Tokenizer.UnknownId));

    /// <summary>The token ids of <paramref name="text"/> (no <c>&lt;s&gt;</c>).</summary>
    public int[] TokenizeText(string text) => Encode(text, addBos: false);

    private int Piece(string piece)
    {
        int id = Tokenizer.PieceToId(piece);
        if (id == Tokenizer.UnknownId) throw new InvalidOperationException($"The tokenizer has no piece '{piece}'.");
        return id;
    }

    /// <summary>The ids the release forbids when generating <paramref name="modality"/> (<c>get_forbidden_tokens</c>): the
    /// markers, and every id below <c>[Text]</c> for speech, or every speech token (with pitch and style for Expressive)
    /// for text.</summary>
    internal HashSet<int> Forbidden(SpiritLmModality modality)
    {
        if (modality == SpiritLmModality.Arbitrary) return new HashSet<int>();
        int text = Piece(SpiritLmTokens.TextPrefix);
        var forbidden = SpiritLmTokens.ForbiddenTokens(modality == SpiritLmModality.Speech, modality == SpiritLmModality.Text, IsExpressive,
            text, _options.Units + 1, _options.PitchTokens, _options.StyleTokens);
        return forbidden;
    }

    // ---------------------------------------------------------------- speech tokens

    private static double[] Samples(Tensor<T> recording)
    {
        var audio = new double[recording.Length];
        for (int i = 0; i < audio.Length; i++) audio[i] = Ops.ToDouble(recording.GetFlat(i));
        return audio;
    }

    /// <summary>
    /// The speech tokens of 16 kHz audio (the release's <c>encode_units</c>): per chunk of at most 100 seconds (chunks under
    /// 1280 samples skipped), the HuBERT units and, for Expressive, the pitch and style tokens.
    /// </summary>
    public (int[] Hubert, int[]? Pitch, int[]? Style) SpeechUnits(Tensor<T> recording)
    {
        ThrowIfDisposed();
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        const int maxChunk = 100 * 16000, minChunk = 1280;
        var audio = Samples(recording);
        var hubert = new List<int>();
        var pitch = IsExpressive ? new List<int>() : null;
        var style = IsExpressive ? new List<int>() : null;
        using var _ = new NoGradScope<T>();
        for (int start = 0; start < audio.Length; start += maxChunk)
        {
            int count = Math.Min(maxChunk, audio.Length - start);
            if (count < minChunk) continue;
            var chunk = new ArraySegment<double>(audio, start, count);
            hubert.AddRange(UnitExtractor.Units(chunk, deduplicate: false));
            if (IsExpressive)
            {
                pitch!.AddRange(PitchTokensOf(chunk));
                style!.AddRange(StyleTokensOf(chunk));
            }
        }
        return (hubert.ToArray(), pitch?.ToArray(), style?.ToArray());
    }

    /// <summary>The F0 and voicing SpiritLM's pitch tokenizer reads (FCPE or pYAAPT, interpolated over unvoiced frames, at
    /// 5 ms).</summary>
    internal (double[] F0, double[] Voiced) F0Of(IReadOnlyList<double> audio, SpiritLmPitchExtractor extractor)
    {
        if (extractor == SpiritLmPitchExtractor.Fcpe)
            return (_fcpe ?? throw new InvalidOperationException("FCPE was not built: set PitchExtractor = Fcpe.")).SpiritLmF0(audio);
        return YaaptPitchTracker.SpiritLmF0(audio);
    }

    /// <summary><c>norm_f0_interp</c>: the mean of the voiced frames taken from the nonzero F0 (the release's
    /// <c>quantize_vqvae</c> without speaker statistics).</summary>
    internal static double[] NormalizeF0(double[] f0, double[] voiced)
    {
        double sum = 0;
        int count = 0;
        for (int i = 0; i < f0.Length; i++) if (voiced[i] != 0) { sum += f0[i]; count++; }
        double mean = count > 0 ? sum / count : double.NaN;
        return f0.Select(v => v != 0 ? v - mean : v).ToArray();
    }

    private int[] PitchTokensOf(IReadOnlyList<double> audio)
    {
        var (f0, voiced) = F0Of(audio, _options.PitchExtractor);
        return PitchQuantizer.Codes(NormalizeF0(f0, voiced), voiced);
    }

    /// <summary>The speech-token string of 16 kHz audio (the release's <c>encode_string</c>).</summary>
    public string SpeechString(Tensor<T> recording)
    {
        var (hubert, pitch, style) = SpeechUnits(recording);
        return SpiritLmTokens.UnitsToString(hubert, pitch, style);
    }

    /// <summary>
    /// The speech of a token string (the release's <c>decode</c>): its HuBERT units, with the pitch and style each carries,
    /// voiced with duration prediction as <paramref name="speaker"/> (<see cref="SpiritLMOptions.Speaker"/> by default) and,
    /// for Base, <see cref="SpiritLMOptions.VocoderStyle"/>.
    /// </summary>
    public Tensor<T> Speak(string speechTokens, int? speaker = null)
    {
        ThrowIfDisposed();
        if (speechTokens is null) throw new ArgumentNullException(nameof(speechTokens));
        var (hubert, pitch, style) = SpiritLmTokens.StringToUnits(speechTokens, SamplingRandom());
        if (hubert.Length == 0) return new Tensor<T>(new[] { 0 });
        if (IsExpressive && (pitch is null || style is null))
            throw new ArgumentException("SpiritLM Expressive's vocoder reads pitch and style tokens; the string has none.", nameof(speechTokens));
        using var _ = new NoGradScope<T>();
        return Vocoder.Forward(hubert, durationPrediction: true, speaker ?? _options.Speaker,
            IsExpressive || _options.VocoderStyles == 0 ? null : _options.VocoderStyle, IsExpressive ? pitch : null, IsExpressive ? style : null);
    }

    private Random SamplingRandom() => _samplingRandom ?? AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(SeedValue);
}
