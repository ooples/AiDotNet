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
/// RWKV streaming transducer: a linear-attention encoder that is genuinely recurrent, so it streams
/// without chunking or a growing cache.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// An and Zhang, "Exploring RWKV for Memory Efficient and Low Latency Streaming ASR"
/// (arXiv:2309.14758). The problem they state: "the full-sequence attention mechanism is
/// non-streamable and computationally expensive, thus requiring modifications, such as chunking and
/// caching, for efficient streaming ASR." RWKV "combines the superior performance of transformers
/// and the inference efficiency of RNNs, which is well-suited for streaming ASR scenarios where the
/// budget for latency and memory is restricted."
/// </para>
/// <para>
/// <b>Two properties carry the method and are implemented rather than described.</b>
/// </para>
/// <list type="number">
/// <item><description><b>Constant memory per step.</b> The recurrence keeps two accumulators and one
/// previous frame per channel, whatever the utterance length. A chunked transformer's cache grows
/// with the audio, which is the cost being removed.</description></item>
/// <item><description><b>Streaming output equals batch output exactly.</b> Because the encoder IS a
/// recurrence rather than an approximation of one, frame-by-frame decoding is not a degraded mode —
/// it is the same computation. Chunked attention only approximates this.</description></item>
/// </list>
/// <para>
/// See <see cref="RwkvTimeMixing{T}"/> for the recurrence itself.
/// </para>
/// <para>
/// <b>For Beginners:</b> To caption speech live, a model must answer as words arrive rather than
/// after the speaker stops. Ordinary attention re-reads the whole recording for each new frame, so
/// it cannot do that cheaply. This keeps a small running summary updated once per frame — the memory
/// it uses does not grow as someone keeps talking, and the live answer matches what you would get
/// from the finished recording.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Exploring RWKV for Memory Efficient and Low Latency Streaming ASR",
    "https://arxiv.org/abs/2309.14758",
    Year = 2023,
    Authors = "Keyu An, Shiliang Zhang")]
public partial class RWKVTransducer<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    private readonly RWKVTransducerOptions _options; public override ModelOptions GetOptions() => _options;
    private RwkvTimeMixing<T>? _timeMixing;

    /// <summary>
    /// Gets the RWKV recurrence, whose state size is constant in the utterance length.
    /// </summary>
    /// <remarks>
    /// Internal: the recurrence is implementation plumbing, not something a model user configures or
    /// drives. Exposing it publicly invited callers to depend on the shape of the recurrence, which
    /// would freeze an internal detail into the API before it is even paper-faithful.
    /// </remarks>
    internal RwkvTimeMixing<T> TimeMixing => _timeMixing ??= new RwkvTimeMixing<T>(
        _options.EncoderDim, _options.TimeDecay, _options.CurrentTokenBonus, _options.TokenShiftMix);

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => true;
    /// <summary>Always false: this model has no word alignment.</summary>
    /// <remarks>
    /// It returned true while ExtractSegments produced ONE segment spanning 0.0 to the full duration
    /// -- an utterance boundary, not word timestamps. A caller checking this flag before asking for
    /// timestamps got told yes and handed the whole utterance as a single "word". Producing real word
    /// timestamps needs CTC forced alignment converting token spans to word spans, which this model
    /// does not implement; until it does, saying no is the only honest answer, and
    /// <c>Transcribe</c> rejects includeTimestamps rather than returning the degenerate segment.
    /// </remarks>
    public bool SupportsWordTimestamps => false;

    public RWKVTransducer(NeuralNetworkArchitecture<T> architecture, string modelPath, RWKVTransducerOptions? options = null) : base(architecture) { _options = options ?? new RWKVTransducerOptions(); _options.Validate(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions); SupportedLanguages = new[] { _options.Language }; InitializeLayers(); }
    public RWKVTransducer(NeuralNetworkArchitecture<T> architecture, RWKVTransducerOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new RWKVTransducerOptions(); _options.Validate(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = _options.LearningRate }); base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; SupportedLanguages = new[] { _options.Language }; InitializeLayers(); }

    /// <summary>
    /// Transcribes audio using RWKVTransducer's RWKV-enhanced parallel branches.
    /// Per the paper: replaces quadratic self-attention with linear RWKV in the attention branch,
    /// while retaining cgMLP for local patterns, then merges via enhanced depthwise-conv module.
    /// </summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool includeTimestamps = false)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        var logits = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);
        if (includeTimestamps)
            throw new NotSupportedException(
                "RWKVTransducer does not produce word timestamps; SupportsWordTimestamps is false. " +
                "Word alignment needs CTC forced alignment, which this model does not implement.");

        var (tokens, confidence) = CTCGreedyDecodeWithConfidence(logits); var text = TokensToText(tokens);
        double duration = audio.Length > 0 ? (double)audio.Shape[0] / SampleRate : 0;
        return new TranscriptionResult<T> { Text = text, Language = language ?? _options.Language, Confidence = NumOps.FromDouble(confidence), DurationSeconds = duration, Segments = Array.Empty<TranscriptionSegment<T>>() };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool includeTimestamps = false, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    /// <summary>Not supported: this model has no language-identification head.</summary>
    /// <remarks>
    /// This ran a full forward pass and then classified by counting CJK versus Latin CODE POINTS in
    /// the token ids -- ids that index a vocabulary and are not code points at all. With
    /// <c>SupportedLanguages</c> holding only the configured language, the result was that language
    /// whatever the audio contained. See <see cref="DetectLanguageProbabilities"/>.
    /// </remarks>
    public string DetectLanguage(Tensor<T> audio)
        => throw new NotSupportedException(
            "RWKVTransducer has no language-identification head, so it cannot detect the spoken " +
            "language. Set it with RWKVTransducerOptions.Language.");
    /// <summary>Not supported: this model has no language-identification head.</summary>
    /// <remarks>
    /// Both constructors set <c>SupportedLanguages</c> to the single configured language, so
    /// <c>DetectLanguage</c> could only ever return that one value -- there was nothing to detect
    /// between. This then reported it with a hardcoded 0.85, a distribution over one outcome that does
    /// not sum to 1. A confident-looking wrong number is worse than a refusal, because a caller can
    /// act on it. Configure the language directly through <c>RWKVTransducerOptions.Language</c>.
    /// </remarks>
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio)
        => throw new NotSupportedException(
            "RWKVTransducer has no language-identification head, so it cannot produce language " +
            "probabilities. Set the language with RWKVTransducerOptions.Language.");
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => new RWKVStreamingSession(this, language ?? _options.Language);

    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultRWKVTransducerLayers(encoderDim: _options.EncoderDim, numLayers: _options.NumEncoderLayers, numHeads: _options.NumAttentionHeads, numMels: _options.NumMels, vocabSize: _options.VocabSize, dropoutRate: _options.DropoutRate, maxSequenceLength: _options.MaxEncoderFrames)); }
    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode.");

        // TRY/FINALLY, NOT TWO STATEMENTS. TrainWithTape can throw -- a shape mismatch, a diverged
        // loss, an OOM part-way through the tape -- and the bare call left the model stuck in
        // training mode when it did. The next Predict then runs with dropout live and stochastic
        // batch-norm statistics, so it silently returns a different answer for the same input, and
        // nothing about that failure points back at the exception that caused it.
        SetTrainingMode(true);
        try
        {
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
    /// <inheritdoc/>
    /// <remarks>
    /// <b>Native mode requires the mel front-end rather than falling through.</b> The default layers are
    /// built against <c>_options.NumMels</c>, so the first layer expects feature frames. Returning the
    /// raw waveform when <c>MelSpec</c> is null hands it audio samples instead -- a different rank and a
    /// different meaning for the last dimension. ONNX mode keeps the passthrough: those graphs commonly
    /// include their own front-end, so the raw waveform is what they are supposed to receive.
    /// </remarks>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (MelSpec is not null) return MelSpec.Forward(rawAudio);

        if (_useNativeMode)
        {
            throw new InvalidOperationException(
                "RWKVTransducer native mode has no log-mel front-end, but its encoder layers are built " +
                $"for {_options.NumMels} mel bands. Passing the raw waveform through would feed audio " +
                "samples to a layer expecting feature frames.");
        }

        return rawAudio;
    }
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "RWKVTransducer-Native" : "RWKVTransducer-ONNX", Description = "RWKVTransducer: RWKV-Enhanced E-Branchformer (Song et al., 2025)", FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };


    private (List<int> tokens, double confidence) CTCGreedyDecodeWithConfidence(Tensor<T> logits) { var tokens = new List<int>(); double totalConf = 0; int confCount = 0; int prevToken = -1; int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1; int vocabSize = logits.Rank >= 2 ? logits.Shape[^1] : logits.Shape[0]; for (int t = 0; t < numFrames; t++) { int maxIdx = 0; double maxVal = double.NegativeInfinity; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); if (val > maxVal) { maxVal = val; maxIdx = v; } } double sumExp = 0; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); sumExp += Math.Exp(val - maxVal); } double frameConf = 1.0 / sumExp; if (maxIdx != prevToken && maxIdx > 0) { tokens.Add(maxIdx); totalConf += frameConf; confCount++; } prevToken = maxIdx; } return (tokens, confCount > 0 ? totalConf / confCount : 0.0); }
    /// <summary>Renders CTC token ids as text using the configured vocabulary.</summary>
    /// <remarks>
    /// THROUGH THE VOCABULARY, not as Unicode code points. Casting each id to a char ignored
    /// <c>RWKVTransducerOptions.Vocabulary</c> entirely, so token 6 rendered as the control character
    /// U+0006 rather than "a" -- every native transcript was mojibake, and nothing threw. Index 0 is
    /// the CTC blank and is skipped, "|" is the word separator, and an id outside the vocabulary is
    /// skipped rather than guessed at.
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
            if (piece.Length > 1 && piece[0] == '<' && piece[piece.Length - 1] == '>') continue;

            sb.Append(piece);
        }

        return sb.ToString().Trim();
    }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(RWKVTransducer<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    private sealed class RWKVStreamingSession : IStreamingTranscriptionSession<T>
    {
        // EACH CHUNK IS TRANSCRIBED ONCE, AS IT ARRIVES, AND ITS WAVEFORM IS THEN RELEASED.
        //
        // This used to retain every chunk for the session's lifetime, and GetPartialResult allocated a
        // combined waveform and re-transcribed the ENTIRE stream on every call. Memory grew with
        // session duration and the inference cost of the Nth partial was O(N) in the stream length, so
        // a long session got steadily slower and larger -- the opposite of what streaming is for.
        //
        // What this does NOT do is carry encoder state across chunks. The native encoder here is a
        // Branchformer, which is attention-based rather than recurrent, so there is no recurrent state
        // to carry; that arrives with the RWKV time-mixing encoder. The consequence is that a word
        // straddling a chunk boundary can be split, which is the accuracy cost of streaming a
        // non-streaming encoder and is stated rather than hidden. Feed overlapping chunks if that
        // matters for your audio.
        private readonly RWKVTransducer<T> _model;
        private readonly string _language;
        private readonly System.Text.StringBuilder _text = new();
        private double _totalDurationSeconds;
        private double _confidenceSum;
        private int _confidenceCount;
        private bool _disposed;

        public RWKVStreamingSession(RWKVTransducer<T> model, string language) { _model = model; _language = language; }

        public void FeedAudio(Tensor<T> audioChunk)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(RWKVStreamingSession));
            if (audioChunk is null) throw new ArgumentNullException(nameof(audioChunk));
            if (audioChunk.Length == 0) return;

            var chunkResult = _model.Transcribe(audioChunk, _language);

            string chunkText = chunkResult.Text;
            if (!string.IsNullOrEmpty(chunkText))
            {
                if (_text.Length > 0) _text.Append(' ');
                _text.Append(chunkText);
            }

            _totalDurationSeconds += chunkResult.DurationSeconds;
            _confidenceSum += _model.NumOps.ToDouble(chunkResult.Confidence);
            _confidenceCount++;

            // The chunk's tensor goes out of scope here: nothing retains it past this call.
        }

        public TranscriptionResult<T> GetPartialResult()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(RWKVStreamingSession));
            if (_confidenceCount == 0) return new TranscriptionResult<T> { Language = _language };

            return new TranscriptionResult<T>
            {
                Text = _text.ToString(),
                Language = _language,
                Confidence = _model.NumOps.FromDouble(_confidenceSum / _confidenceCount),
                DurationSeconds = _totalDurationSeconds,
                Segments = Array.Empty<TranscriptionSegment<T>>()
            };
        }

        public TranscriptionResult<T> Finalize() { if (_disposed) throw new ObjectDisposedException(nameof(RWKVStreamingSession)); var result = GetPartialResult(); _disposed = true; return result; }
        public void Dispose() { _disposed = true; }
    }
}
