using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>
/// WhisperSpeech (Collabora 2024): text-to-speech by inverting Whisper, after SPEAR-TTS (Kharitonov et al. 2023). A
/// text-to-semantic transformer writes the semantic tokens Whisper's quantized encoder would hear for the text, a
/// semantic-to-acoustic transformer writes EnCodec's tokens for them in a speaker's voice (an ECAPA-TDNN embedding), and
/// Vocos turns the EnCodec tokens into 24 kHz speech.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>Built as the release (collabora/WhisperSpeech) runs it; see <see cref="WhisperSpeechOptions"/> and
/// <c>docs/text-to-speech/paper-fidelity-decisions.md</c>. Text goes in as UTF-8 bytes; speech comes out at 24 kHz. Without
/// a <see cref="SpeakerEmbedding"/> the model speaks in the release's default voice; <see cref="CreateSpeaker"/> makes one
/// from a recording. All three trained parts train here: the T2S and S2A models and the semantic tokenizer
/// (<see cref="WhisperSpeechOptions.Stage"/>).</para>
/// <para><b>For Beginners:</b> WhisperSpeech first decides "what is said" as the tokens a speech recognizer would hear,
/// then "how it sounds" as audio-codec tokens in a chosen voice, and finally turns those into a waveform.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Speak, Read and Prompt: High-Fidelity Text-to-Speech with Minimal Supervision",
    "https://arxiv.org/abs/2302.03540",
    Year = 2023,
    Authors = "Eugene Kharitonov, Damien Vincent, Zalán Borsos, Raphaël Marinier, Sertan Girgin, Olivier Pietquin, Matt Sharifi, Marco Tagliasacchi, Neil Zeghidour"
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1.5e-3, Beta1 = 0.9, Beta2 = 0.95, WeightDecay = 0.1,
                Schedule = LearningRateSchedulerType.OneCycle,
                Source = "collabora/WhisperSpeech train_multi.py: AdamW β (0.9, 0.95) with per-module groups (embeddings and norms "
                        + "undecayed, linear layers at base width / fan-in), OneCycleLR (cosine, pct_start = min(0.3, warmup / steps), "
                        + "div and final div 25, β1 cycled 0.95 → 0.85 → 0.95), gradient-norm clipping; each model's tunables: "
                        + "T2S 1.5e-3 / clip 0.2 / wd 0.1 / warmup 4000, S2A 3e-3 / 2 / 1e-3 / 2000, VQ 9e-4 / 2 / 1e-3 / 850.")]
public partial class WhisperSpeech<T> : TtsModelBase<T>, ICodecTts<T>
{
    private readonly WhisperSpeechOptions _options;
    private readonly bool _useNativeMode;
    private bool _disposed;
    private WhisperSpeechCore<T>? _core;
    private readonly WhisperSpeechLayers<T> _layers = new();

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>Creates WhisperSpeech running an ONNX export.</summary>
    public WhisperSpeech(NeuralNetworkArchitecture<T> architecture, string modelPath, WhisperSpeechOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new WhisperSpeechOptions();
        Options = _options;
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates WhisperSpeech with its networks (randomly initialized).</summary>
    public WhisperSpeech(NeuralNetworkArchitecture<T> architecture, WhisperSpeechOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new WhisperSpeechOptions();
        Options = _options;
        _useNativeMode = true;
        base.SampleRate = _options.SampleRate;
        base.HiddenDim = _options.T2SHeads * _options.T2SHeadWidth;
        InitializeLayers();
    }

    /// <summary>The longest text, in UTF-8 bytes (one position is the start token).</summary>
    public int MaxTextLength => _options.MaxTextLength - 1;

    private bool HasPaperLayers => _core is not null;
    private int SeedValue => _options.Seed ?? 42;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");
    internal WhisperSpeechCore<T> Core => _core ?? throw NotNative();
    internal WhisperSpeechLayers<T> CoreLayers => _layers;

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperLayers ? LayerInputDomain.Indices(256) : base.GetInputDomain(inputShape);

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            AddUnsplitLayers(Architecture.Layers);
            return;
        }
        _core = new WhisperSpeechCore<T>(Engine, _options, _layers, Architecture.RandomSeed ?? SeedValue);
        AddUnsplitLayers(_layers.TextToSemantic.Cast<ILayer<T>>().ToList());
        // The S2A model, the semantic tokenizer, Whisper, the speaker extractor, the codec and Vocos, so clones and saved
        // models keep them.
        ComponentLayers.AddRange(_layers.SemanticToAcoustic);
        ComponentLayers.AddRange(_layers.Tokenizer);
        ComponentLayers.AddRange(_layers.Whisper);
        ComponentLayers.AddRange(_layers.Speaker);
        ComponentLayers.Add(new AiDotNet.TextToSpeech.CodecBased.AudioCodecLayer<T>(_core.Codec));
        ComponentLayers.AddRange(_core.Vocos.Layers);
    }

    // ---------------------------------------------------------------- speakers and tokens

    /// <summary>The speaker embedding the S2A model speaks with; null speaks in the release's default voice.</summary>
    public Tensor<T>? SpeakerEmbedding { get; set; }

    /// <summary>The conditioned S2A variant's language, SNR and C50 (null: the language from the options, the rest
    /// missing).</summary>
    public WhisperSpeechConditions? Conditions { get; set; }

    /// <summary>A speaker embedding from a recording at any rate (<c>extract_spk_emb</c>: its first 30 seconds).</summary>
    public Tensor<T> CreateSpeaker(Tensor<T> recording, int sampleRate)
    {
        ThrowIfDisposed();
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        using var _ = Inference();
        return Core.SpeakerEmbedding(recording, sampleRate);
    }

    /// <summary>The semantic tokens of a recording at any rate (the Whisper VQ tokenizer, at most 30 seconds).</summary>
    public int[] SemanticTokens(Tensor<T> recording, int sampleRate)
    {
        ThrowIfDisposed();
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        using var _ = Inference();
        return Core.SemanticTokens(Core.Resampled(recording, sampleRate, 16000));
    }

    /// <summary>The semantic tokens the T2S model writes for text.</summary>
    public int[] TextToSemanticTokens(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        using var _ = Inference();
        return Core.SemanticFor(text.Replace("\n", " "), _options.Language, _options.CharactersPerSecond, SamplingRandom());
    }

    /// <summary>The EnCodec tokens <c>[quantizers][frames]</c> the S2A model writes for semantic tokens in
    /// <see cref="SpeakerEmbedding"/>.</summary>
    public int[][] SemanticToAcousticTokens(IReadOnlyList<int> semantic)
    {
        ThrowIfDisposed();
        if (semantic is null) throw new ArgumentNullException(nameof(semantic));
        using var _ = Inference();
        return Core.SemanticToAcoustic.Generate(semantic, Speaker(), null, _options.Temperature, _options.TopK, SamplingRandom(), CurrentConditions());
    }

    private Tensor<T> Speaker() => SpeakerEmbedding ?? Core.DefaultSpeakerEmbedding();

    private WhisperSpeechConditions? CurrentConditions() =>
        !_options.S2AConditioned ? null : Conditions ?? new WhisperSpeechConditions(WhisperSpeechCore<T>.LanguageId(_options.Language));

    // ---------------------------------------------------------------- codec tokens (ICodecTts)

    /// <inheritdoc />
    public int NumCodebooks => HasPaperLayers ? Core.SemanticToAcoustic.Configuration.Quantizers : 0;

    /// <inheritdoc />
    public int CodebookSize => _options.Codec.CodebookSize;

    /// <inheritdoc />
    public int CodecFrameRate => _options.SampleRate / _options.HopSize;

    /// <inheritdoc />
    /// <remarks>EnCodec's codes <c>[quantizers, frames]</c> of audio at the model's rate.</remarks>
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        var codes = Core.AcousticTokens(audio);
        var tokens = new Tensor<T>(new[] { codes.Length, codes[0].Length });
        for (int q = 0; q < codes.Length; q++)
            for (int f = 0; f < codes[q].Length; f++) tokens[q, f] = Ops.FromDouble(codes[q][f]);
        return tokens;
    }

    /// <inheritdoc />
    /// <remarks>Vocos's waveform of codes <c>[quantizers, frames]</c>.</remarks>
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        ThrowIfDisposed();
        if (tokens is null) throw new ArgumentNullException(nameof(tokens));
        if (tokens.Rank != 2) throw new ArgumentException("Expected codes [quantizers, frames].", nameof(tokens));
        var codes = Enumerable.Range(0, tokens.Shape[0])
            .Select(q => Enumerable.Range(0, tokens.Shape[1]).Select(f => (int)Math.Round(Ops.ToDouble(tokens[q, f]))).ToArray()).ToArray();
        using var _ = Inference();
        return Core.Vocode(codes);
    }

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    /// <remarks>The release's <c>Pipeline.generate</c> in <see cref="WhisperSpeechOptions.Language"/> at
    /// <see cref="WhisperSpeechOptions.CharactersPerSecond"/>, in <see cref="SpeakerEmbedding"/>.</remarks>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        if (!HasPaperLayers) return base.Synthesize(text);
        using var _ = Inference();
        return Core.Speak(text, Speaker(), _options.Language, _options.CharactersPerSecond, SamplingRandom(), CurrentConditions());
    }

    /// <inheritdoc />
    /// <remarks>The text's UTF-8 bytes (stripped, newlines as spaces), as the T2S model reads them.</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        var bytes = System.Text.Encoding.UTF8.GetBytes(text.Replace("\n", " ").Trim());
        return IndexTtsOps.Ids<T>(bytes.Length == 0 ? new[] { 32 } : bytes.Select(b => (int)b).ToArray());
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is the text's UTF-8 bytes; the result is their speech.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        using var _ = Inference();
        return Core.Speak(TextOf(input), Speaker(), _options.Language, _options.CharactersPerSecond, SamplingRandom(), CurrentConditions());
    }

    // The text of UTF-8 byte ids (invalid sequences decode to U+FFFD, as Python's errors="replace").
    private string TextOf(Tensor<T> input)
    {
        var bytes = new byte[input.Length];
        for (int i = 0; i < bytes.Length; i++)
        {
            int b = (int)Math.Round(Ops.ToDouble(input.GetFlat(i)));
            if (b < 0 || b > 255) throw new ArgumentOutOfRangeException(nameof(input), "Text ids are UTF-8 bytes in [0, 256).");
            bytes[i] = (byte)b;
        }
        if (bytes.Length > MaxTextLength) throw new ArgumentException($"The text is longer than {MaxTextLength} bytes.", nameof(input));
        return System.Text.Encoding.UTF8.GetString(bytes);
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    // A fresh generator per call: the same input gives the same output.
    private Random SamplingRandom() => AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(SeedValue);

    private IDisposable Inference()
    {
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        var scope = new NoGradScope<T>();
        return new Restore(() =>
        {
            scope.Dispose();
            SetTrainingMode(wasTraining);
        });
    }

    private sealed class Restore : IDisposable
    {
        private Action? _action;
        public Restore(Action action) => _action = action;
        public void Dispose()
        {
            _action?.Invoke();
            _action = null;
        }
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "WhisperSpeech-Native" : "WhisperSpeech-ONNX",
            Description = "WhisperSpeech: text-to-speech by inverting Whisper (Collabora, after SPEAR-TTS, Kharitonov et al., 2023)",
            FeatureCount = _options.T2SHeads * _options.T2SHeadWidth,
        };
        metadata.AdditionalInfo["Architecture"] = "WhisperSpeech";
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(WhisperSpeech<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
