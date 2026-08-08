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
public class RWKVTransducer<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
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
    public bool SupportsWordTimestamps => true;

    public RWKVTransducer(NeuralNetworkArchitecture<T> architecture, string modelPath, RWKVTransducerOptions? options = null) : base(architecture) { _options = options ?? new RWKVTransducerOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions); SupportedLanguages = new[] { _options.Language }; InitializeLayers(); }
    public RWKVTransducer(NeuralNetworkArchitecture<T> architecture, RWKVTransducerOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new RWKVTransducerOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>> { LearningRate = _options.LearningRate }); base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; SupportedLanguages = new[] { _options.Language }; InitializeLayers(); }

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
        var (tokens, confidence) = CTCGreedyDecodeWithConfidence(logits); var text = TokensToText(tokens);
        double duration = audio.Length > 0 ? (double)audio.Shape[0] / SampleRate : 0;
        return new TranscriptionResult<T> { Text = text, Language = language ?? _options.Language, Confidence = NumOps.FromDouble(confidence), DurationSeconds = duration, Segments = includeTimestamps ? ExtractSegments(text, duration, confidence) : Array.Empty<TranscriptionSegment<T>>() };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool includeTimestamps = false, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    public string DetectLanguage(Tensor<T> audio) { var features = PreprocessAudio(audio); Tensor<T> logits; if (IsOnnxMode && OnnxEncoder is not null) logits = OnnxEncoder.Run(features); else { logits = features; foreach (var l in Layers) logits = l.Forward(logits); } var (tokens, _) = CTCGreedyDecodeWithConfidence(logits); return ClassifyLanguageFromTokens(tokens); }
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) { var detected = DetectLanguage(audio); var result = new Dictionary<string, T>(); double primaryProb = 0.85; double otherProb = SupportedLanguages.Count > 1 ? (1.0 - primaryProb) / (SupportedLanguages.Count - 1) : 0.0; foreach (var lang in SupportedLanguages) result[lang] = NumOps.FromDouble(lang == detected ? primaryProb : otherProb); return result; }
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => new RWKVStreamingSession(this, language ?? _options.Language);

    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultBranchformerLayers(encoderDim: _options.EncoderDim, numLayers: _options.NumEncoderLayers, numAttentionHeads: _options.NumAttentionHeads, cgmlpDim: _options.CgmlpDim, numMels: _options.NumMels, vocabSize: _options.VocabSize, dropoutRate: _options.DropoutRate)); }
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
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = (int)l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
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
    protected override void SerializeNetworkSpecificData(BinaryWriter w) { w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty); w.Write(_options.SampleRate); w.Write(_options.MaxAudioLengthSeconds); w.Write(_options.EncoderDim); w.Write(_options.NumEncoderLayers); w.Write(_options.NumAttentionHeads); w.Write(_options.CgmlpDim); w.Write(_options.NumMels); w.Write(_options.VocabSize); w.Write(_options.DropoutRate); w.Write(_options.Language); w.Write(_options.TimeDecay); w.Write(_options.CurrentTokenBonus); w.Write(_options.TokenShiftMix); w.Write(_options.BoundaryAware); w.Write(_options.LearningRate); }
    protected override void DeserializeNetworkSpecificData(BinaryReader r) { _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = r.ReadInt32(); _options.MaxAudioLengthSeconds = r.ReadInt32(); _options.EncoderDim = r.ReadInt32(); _options.NumEncoderLayers = r.ReadInt32(); _options.NumAttentionHeads = r.ReadInt32(); _options.CgmlpDim = r.ReadInt32(); _options.NumMels = r.ReadInt32(); _options.VocabSize = r.ReadInt32(); _options.DropoutRate = r.ReadDouble(); _options.Language = r.ReadString(); if (r.BaseStream.Position < r.BaseStream.Length) _options.TimeDecay = r.ReadDouble(); if (r.BaseStream.Position < r.BaseStream.Length) _options.CurrentTokenBonus = r.ReadDouble(); if (r.BaseStream.Position < r.BaseStream.Length) _options.TokenShiftMix = r.ReadDouble(); if (r.BaseStream.Position < r.BaseStream.Length) _options.BoundaryAware = r.ReadBoolean(); if (r.BaseStream.Position < r.BaseStream.Length) _options.LearningRate = r.ReadDouble(); _timeMixing = null; base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new RWKVTransducer<T>(Architecture, mp, _options); return new RWKVTransducer<T>(Architecture, _options); }
    private (List<int> tokens, double confidence) CTCGreedyDecodeWithConfidence(Tensor<T> logits) { var tokens = new List<int>(); double totalConf = 0; int confCount = 0; int prevToken = -1; int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1; int vocabSize = logits.Rank >= 2 ? logits.Shape[^1] : logits.Shape[0]; for (int t = 0; t < numFrames; t++) { int maxIdx = 0; double maxVal = double.NegativeInfinity; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); if (val > maxVal) { maxVal = val; maxIdx = v; } } double sumExp = 0; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); sumExp += Math.Exp(val - maxVal); } double frameConf = 1.0 / sumExp; if (maxIdx != prevToken && maxIdx > 0) { tokens.Add(maxIdx); totalConf += frameConf; confCount++; } prevToken = maxIdx; } return (tokens, confCount > 0 ? totalConf / confCount : 0.0); }
    private static string TokensToText(List<int> tokens) { var sb = new System.Text.StringBuilder(); foreach (var t in tokens) { if (t > 0 && t <= char.MaxValue) sb.Append((char)t); else if (t > char.MaxValue && t <= 0x10FFFF) sb.Append(char.ConvertFromUtf32(t)); } return sb.ToString().Trim(); }
    private IReadOnlyList<TranscriptionSegment<T>> ExtractSegments(string text, double duration, double confidence) { if (string.IsNullOrWhiteSpace(text)) return Array.Empty<TranscriptionSegment<T>>(); return new[] { new TranscriptionSegment<T> { Text = text, StartTime = 0.0, EndTime = duration, Confidence = NumOps.FromDouble(confidence) } }; }
    private string ClassifyLanguageFromTokens(List<int> tokens) { if (tokens.Count == 0) return _options.Language; int cjkCount = 0, latinCount = 0; foreach (var t in tokens) { if (t >= 0x4E00 && t <= 0x9FFF) cjkCount++; else if (t >= 0x41 && t <= 0x7A) latinCount++; } if (cjkCount > latinCount && SupportedLanguages.Contains("zh")) return "zh"; return _options.Language; }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(RWKVTransducer<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    private sealed class RWKVStreamingSession : IStreamingTranscriptionSession<T>
    {
        private readonly RWKVTransducer<T> _model; private readonly string _language; private readonly List<Tensor<T>> _chunks = new(); private bool _disposed;
        public RWKVStreamingSession(RWKVTransducer<T> model, string language) { _model = model; _language = language; }
        public void FeedAudio(Tensor<T> audioChunk) { if (_disposed) throw new ObjectDisposedException(nameof(RWKVStreamingSession)); _chunks.Add(audioChunk); }
        public TranscriptionResult<T> GetPartialResult() { if (_disposed) throw new ObjectDisposedException(nameof(RWKVStreamingSession)); if (_chunks.Count == 0) return new TranscriptionResult<T> { Language = _language }; int totalLen = 0; foreach (var ch in _chunks) totalLen += ch.Length; var combined = new Tensor<T>(new[] { totalLen }); int offset = 0; foreach (var ch in _chunks) { for (int i = 0; i < ch.Length; i++) combined[offset + i] = ch[i]; offset += ch.Length; } return _model.Transcribe(combined, _language); }
        public TranscriptionResult<T> Finalize() { if (_disposed) throw new ObjectDisposedException(nameof(RWKVStreamingSession)); var result = GetPartialResult(); _disposed = true; return result; }
        public void Dispose() { _disposed = true; }
    }
}
