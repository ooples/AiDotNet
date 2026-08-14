using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.SpeechRecognition.ConformerFamily;

/// <summary>
/// ContextNet: CNN encoder with squeeze-and-excitation and global context for ASR.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "ContextNet: Improving Convolutional Neural Networks for ASR with Global Context" (Han et al., 2020)</item></list></para>
/// <para><b>For Beginners:</b> A purely convolutional encoder that uses squeeze-and-excitation (SE) blocks to capture global context. Each block contains depthwise separable convolutions with SE modules that adaptively reweight channel features based on the entire sequence. Typ...</para>
/// <para>
/// A purely convolutional encoder that uses squeeze-and-excitation (SE) blocks to capture
/// global context. Each block contains depthwise separable convolutions with SE modules
/// that adaptively reweight channel features based on the entire sequence. Typically paired
/// with an RNN-T decoder for streaming ASR. Achieves WER 1.9%/3.9% on LibriSpeech.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a ContextNet model with CNN squeeze-and-excitation for ASR
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 16000, inputWidth: 1, inputDepth: 1, outputSize: 5000);
/// var model = new ContextNet&lt;double&gt;(architecture);
///
/// // Or load a pre-trained ONNX model for convolutional ASR inference
/// var onnxModel = new ContextNet&lt;double&gt;(architecture, "contextnet.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("ContextNet: Improving Convolutional Neural Networks for Automatic Speech Recognition with Global Context", "https://arxiv.org/abs/2005.03191", Year = 2020, Authors = "Han et al.")]
public partial class ContextNet<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    private readonly ContextNetOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    public override ModelOptions GetOptions() => _options;
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => true;
    public bool SupportsWordTimestamps => false;

    public ContextNet(NeuralNetworkArchitecture<T> architecture, string modelPath, ContextNetOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new ContextNetOptions();
        _options.Validate();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        base.NumMels = _options.NumMels;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = new[] { _options.Language };
        InitializeLayers();
    }

    public ContextNet(NeuralNetworkArchitecture<T> architecture, ContextNetOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new ContextNetOptions();
        _options.Validate();
        _useNativeMode = true;
        // Global-norm gradient clipping. The default-constructed optimizer applied no
        // clipping at all, so a single update could move a deep CTC stack far enough to
        // swing the output by orders of magnitude. Same remedy, and the same 1.0 bound,
        // that the video models here already use (MoG, VideoCLIP) and that fixed the
        // N-BEATS blow-up. Paired with the residual connections restored in
        // CreateDefaultDeepCNNCTCLayers -- the architecture bounds the gain, this bounds
        // the step.
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            { EnableGradientClipping = true, MaxGradientNorm = 1.0 });
        base.SampleRate = _options.SampleRate;
        base.NumMels = _options.NumMels;
        SupportedLanguages = new[] { _options.Language };
        InitializeLayers();
    }

    /// <summary>
    /// Transcribes audio using ContextNet's CNN encoder with squeeze-and-excitation.
    /// Per the paper: 23 convolutional blocks, each with depthwise separable convolutions
    /// and SE modules that pool global context to reweight channel features.
    /// Paired with CTC or RNN-T decoding.
    /// </summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool includeTimestamps = false)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        var logits = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);
        var (tokens, confidence) = CTCGreedyDecodeWithConfidence(logits); var text = TokensToText(tokens);
        double duration = audio.Length > 0 ? (double)audio.Shape[0] / SampleRate : 0;
        if (includeTimestamps) throw new NotSupportedException("Word-level timestamps are not supported for ContextNet.");
        return new TranscriptionResult<T> { Text = text, Language = language ?? _options.Language, Confidence = NumOps.FromDouble(confidence), DurationSeconds = duration, Segments = Array.Empty<TranscriptionSegment<T>>() };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool includeTimestamps = false, CancellationToken cancellationToken = default) => Task.Run(() => { cancellationToken.ThrowIfCancellationRequested(); return Transcribe(audio, language, includeTimestamps); }, cancellationToken);
    public string DetectLanguage(Tensor<T> audio)
    {
        ThrowIfDisposed();
        return _options.Language; // Monolingual model - returns configured language
    }
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio)
    {
        ThrowIfDisposed();
        // ContextNet is monolingual; return full confidence for the configured language.
        var result = new Dictionary<string, T>();
        foreach (var lang in SupportedLanguages)
            result[lang] = NumOps.FromDouble(lang == _options.Language ? 1.0 : 0.0);
        return result;
    }
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => new ContextNetStreamingSession(this, language ?? _options.Language);

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultContextNetLayers(
                numBlocks: _options.NumBlocks,
                numSubBlocks: _options.NumSubBlocks,
                numMels: _options.NumMels,
                vocabSize: _options.VocabSize,
                kernelSize: _options.KernelSize,
                squeezeExcitationRatio: _options.SqueezeExcitationRatio,
                widthScaling: _options.WidthScaling,
                dropoutRate: _options.DropoutRate));
        }
    }
    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            // Honor the optimizer supplied to the native constructor. The prior
            // null call silently selected NeuralNetworkBase's fallback optimizer,
            // so ContextNet's declared AdamW configuration was never used and its
            // first update could move uphill on the deterministic fitting probe.
            TrainWithTape(input, expected, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }
    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) { if (MelSpec is not null) return MelSpec.Forward(rawAudio); return rawAudio; }
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        Name = _useNativeMode ? "ContextNet-Native" : "ContextNet-ONNX",
        Description = "ContextNet: CNN with Squeeze-and-Excitation (Han et al., 2020)",
        FeatureCount = _options.NumMels,
        Complexity = _options.NumBlocks,
        AdditionalInfo = new Dictionary<string, object>
        {
            ["Mode"] = _useNativeMode ? "Native" : "ONNX",
            ["EncoderDim"] = _options.EncoderDim,
            ["NumBlocks"] = _options.NumBlocks,
            ["SqueezeExcitationRatio"] = _options.SqueezeExcitationRatio,
            ["NumMels"] = _options.NumMels,
            ["VocabSize"] = _options.VocabSize,
            ["SampleRate"] = _options.SampleRate,
            ["MaxAudioLengthSeconds"] = _options.MaxAudioLengthSeconds,
            ["DropoutRate"] = _options.DropoutRate,
            ["Language"] = _options.Language
        }
    };
    /// <summary>
    /// Marks a payload as carrying a version number, distinguishing it from the unversioned layout.
    /// </summary>
    /// <remarks>
    /// <b>0xFF is a value the first byte of a v1 payload cannot hold.</b> That payload began with a
    /// <see cref="bool"/>, which <see cref="BinaryWriter"/> writes as exactly 0x00 or 0x01, so a leading
    /// 0xFF is an unambiguous discriminator rather than a guess. Without one there is no way to tell the
    /// two layouts apart at all: both are opaque byte streams that begin with a plausible value.
    /// </remarks>
    private const byte SerializationVersionMarker = 0xFF;

    /// <summary>
    /// Version 2 inserted <c>NumSubBlocks</c>, <c>KernelSize</c> and <c>WidthScaling</c>.
    /// </summary>
    private const int SerializationVersion = 2;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>The three v2 fields were inserted MID-STREAM, not appended, which is why this needs a version
    /// rather than a length check.</b> A v1 payload runs ... EncoderDim, NumBlocks,
    /// SqueezeExcitationRatio, NumMels ...; reading the v2 layout against it consumes
    /// SqueezeExcitationRatio as NumSubBlocks and then misaligns every remaining field, including the
    /// <see cref="double"/> and the length-prefixed string. The result is not an exception -- it is a
    /// model that loads successfully with silently wrong architecture options.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(SerializationVersionMarker);
        w.Write(SerializationVersion);

        w.Write(_useNativeMode);
        w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate);
        w.Write(_options.MaxAudioLengthSeconds);
        w.Write(_options.EncoderDim);
        w.Write(_options.NumBlocks);
        w.Write(_options.NumSubBlocks);
        w.Write(_options.KernelSize);
        w.Write(_options.WidthScaling);
        w.Write(_options.SqueezeExcitationRatio);
        w.Write(_options.NumMels);
        w.Write(_options.VocabSize);
        w.Write(_options.DropoutRate);
        w.Write(_options.Language);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        // The first byte decides the layout. 0xFF means a versioned payload; 0x00 or 0x01 is the v1
        // bool that used to lead, and is consumed as that bool rather than re-read.
        byte lead = r.ReadByte();
        int version;
        if (lead == SerializationVersionMarker)
        {
            version = r.ReadInt32();
            if (version > SerializationVersion)
            {
                throw new InvalidOperationException(
                    $"This ContextNet payload was written by a newer AiDotNet (serialization version " +
                    $"{version}); this build reads up to version {SerializationVersion}. Upgrade AiDotNet " +
                    $"to load it. Refusing rather than reading it as version {SerializationVersion}, which " +
                    $"would load a model configured with whatever the extra bytes happened to decode to.");
            }
            _useNativeMode = r.ReadBoolean();
        }
        else
        {
            version = 1;
            _useNativeMode = lead != 0;
        }

        string mp = r.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;

        _options.SampleRate = r.ReadInt32();
        _options.MaxAudioLengthSeconds = r.ReadInt32();
        _options.EncoderDim = r.ReadInt32();
        _options.NumBlocks = r.ReadInt32();

        // Absent from v1. Left at their defaults there, which is the closest thing to the truth
        // available: the payload was written by a build for which these were not configurable.
        if (version >= 2)
        {
            _options.NumSubBlocks = r.ReadInt32();
            _options.KernelSize = r.ReadInt32();
            _options.WidthScaling = r.ReadDouble();
        }

        _options.SqueezeExcitationRatio = r.ReadInt32();
        _options.NumMels = r.ReadInt32();
        _options.VocabSize = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble();
        _options.Language = r.ReadString();

        base.SampleRate = _options.SampleRate;
        base.NumMels = _options.NumMels;

        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
        {
            OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
        }
    }
    private (List<int> tokens, double confidence) CTCGreedyDecodeWithConfidence(Tensor<T> logits) { var tokens = new List<int>(); double totalConf = 0; int confCount = 0; int prevToken = -1; int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1; int vocabSize = logits.Rank >= 2 ? logits.Shape[^1] : logits.Shape[0]; for (int t = 0; t < numFrames; t++) { int maxIdx = 0; double maxVal = double.NegativeInfinity; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); if (val > maxVal) { maxVal = val; maxIdx = v; } } double sumExp = 0; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); sumExp += Math.Exp(val - maxVal); } double frameConf = 1.0 / sumExp; if (maxIdx != prevToken && maxIdx > 0) { tokens.Add(maxIdx); totalConf += frameConf; confCount++; } prevToken = maxIdx; } return (tokens, confCount > 0 ? totalConf / confCount : 0.0); }
    /// <summary>
    /// Maps token IDs to text. Without a loaded vocabulary, uses Unicode codepoint mapping
    /// as a best-effort fallback for models with Unicode-based token vocabularies.
    /// ONNX models typically include their own tokenizer; this path is for native mode.
    /// </summary>
    /// <summary>Renders CTC token ids as text using the configured vocabulary.</summary>
    /// <remarks>
    /// THROUGH THE VOCABULARY, not as Unicode code points. This used to cast each id straight to a
    /// char, which ignored <c>ContextNetOptions.Vocabulary</c> entirely: token 6 rendered as the
    /// control character U+0006 rather than as "a". Every transcript the native path produced was
    /// mojibake, and nothing threw. Index 0 is the CTC blank and is skipped; the word separator "|"
    /// becomes a space; an id outside the vocabulary is skipped rather than guessed at.
    /// </remarks>
    private string TokensToText(List<int> tokens)
    {
        var vocabulary = _options.Vocabulary;
        var sb = new System.Text.StringBuilder();
        foreach (var t in tokens)
        {
            if (t <= 0 || t >= vocabulary.Length) continue;

            string piece = vocabulary[t];
            if (piece == "|") { sb.Append(' '); continue; }

            // Remaining specials (<pad>, <s>, </s>, <unk>) carry no text.
            if (piece.Length > 1 && piece[0] == '<' && piece[piece.Length - 1] == '>') continue;

            sb.Append(piece);
        }

        return sb.ToString().Trim();
    }
    private IReadOnlyList<TranscriptionSegment<T>> ExtractSegments(string text, double duration, double confidence) { if (string.IsNullOrWhiteSpace(text)) return Array.Empty<TranscriptionSegment<T>>(); return new[] { new TranscriptionSegment<T> { Text = text, StartTime = 0.0, EndTime = duration, Confidence = NumOps.FromDouble(confidence) } }; }
    private string ClassifyLanguageFromTokens(List<int> _) => _options.Language;
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(ContextNet<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    private sealed class ContextNetStreamingSession : IStreamingTranscriptionSession<T>
    {
        private readonly ContextNet<T> _model;
        private readonly string _language;
        private readonly List<Tensor<T>> _chunks = new();
        private readonly object _lock = new();
        private bool _disposed;

        public ContextNetStreamingSession(ContextNet<T> model, string language) { _model = model; _language = language; }

        public void FeedAudio(Tensor<T> audioChunk)
        {
            lock (_lock)
            {
                if (_disposed) throw new ObjectDisposedException(nameof(ContextNetStreamingSession));
                _chunks.Add(audioChunk);
            }
        }

        public TranscriptionResult<T> GetPartialResult()
        {
            List<Tensor<T>> snapshot;
            lock (_lock)
            {
                if (_disposed) throw new ObjectDisposedException(nameof(ContextNetStreamingSession));
                if (_chunks.Count == 0) return new TranscriptionResult<T> { Language = _language };
                snapshot = new List<Tensor<T>>(_chunks);
            }
            return TranscribeSnapshot(snapshot);
        }

        public TranscriptionResult<T> Finalize()
        {
            List<Tensor<T>> snapshot;
            lock (_lock)
            {
                if (_disposed) throw new ObjectDisposedException(nameof(ContextNetStreamingSession));
                snapshot = new List<Tensor<T>>(_chunks);
                _disposed = true;
            }
            if (snapshot.Count == 0) return new TranscriptionResult<T> { Language = _language };
            return TranscribeSnapshot(snapshot);
        }

        private TranscriptionResult<T> TranscribeSnapshot(List<Tensor<T>> snapshot)
        {
            int totalLen = 0;
            foreach (var ch in snapshot) totalLen += ch.Length;
            var combined = new Tensor<T>(new[] { totalLen });
            int offset = 0;
            foreach (var ch in snapshot) { for (int i = 0; i < ch.Length; i++) combined[offset + i] = ch[i]; offset += ch.Length; }
            return _model.Transcribe(combined, _language);
        }

        public void Dispose() { lock (_lock) { _disposed = true; } }
    }
}
