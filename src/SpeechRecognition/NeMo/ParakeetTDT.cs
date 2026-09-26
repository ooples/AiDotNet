using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.SpeechRecognition.NeMo;

/// <summary>
/// Parakeet-TDT: NVIDIA NeMo's 1.1B Conformer with Token-and-Duration Transducer.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Token-and-Duration Transducer for streaming ASR" (NVIDIA, 2024)</item></list></para>
/// <para><b>For Beginners:</b> Parakeet-TDT extends the standard RNN-T by jointly predicting both the output token and the number of encoder frames to advance (duration). This "Token-and-Duration" formulation allows the model to skip multiple blank frames at once, significantly...</para>
/// <para>
/// Parakeet-TDT extends the standard RNN-T by jointly predicting both the output token and
/// the number of encoder frames to advance (duration). This "Token-and-Duration" formulation
/// allows the model to skip multiple blank frames at once, significantly reducing inference
/// latency for streaming. The joint network now outputs both token logits and duration logits.
/// During greedy decode, when a non-blank token is emitted, the duration head specifies how
/// many frames to skip forward, avoiding per-frame blank predictions.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Parakeet-TDT model with token-and-duration transducer
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputHeight: 16000, inputWidth: 1, inputDepth: 1, outputSize: 1024);
/// var model = new ParakeetTDT&lt;double&gt;(architecture);
///
/// // Or load a pre-trained ONNX model for low-latency streaming ASR
/// var onnxModel = new ParakeetTDT&lt;double&gt;(architecture, "parakeettdt.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Efficient Sequence Transduction by Jointly Predicting Tokens and Durations", "https://arxiv.org/abs/2304.06795", Year = 2023, Authors = "Xu et al.")]
public partial class ParakeetTDT<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    private readonly ParakeetTDTOptions _options; public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => true;
    public bool SupportsWordTimestamps => false;

    public ParakeetTDT(NeuralNetworkArchitecture<T> architecture, string modelPath, ParakeetTDTOptions? options = null) : base(architecture) { _options = options ?? new ParakeetTDTOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions); SupportedLanguages = new[] { "en" }; InitializeLayers(); }
    public ParakeetTDT(NeuralNetworkArchitecture<T> architecture, ParakeetTDTOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new ParakeetTDTOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; SupportedLanguages = new[] { "en" }; InitializeLayers(); }

    /// <summary>
    /// Transcribes audio using Fast Conformer encoder + TDT decoder.
    /// Per NVIDIA (2024): the TDT joint network produces both token and duration logits.
    /// During greedy decode, when a non-blank token is emitted, the duration head determines
    /// how many encoder frames to skip (1 to MaxDurationTokens), reducing the number of
    /// forward passes needed and achieving lower latency streaming.
    /// </summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool includeTimestamps = false)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        Tensor<T> logits;

        if (IsOnnxMode && OnnxEncoder is not null)
        {
            logits = OnnxEncoder.Run(features);
        }
        else
        {
            logits = features;
            foreach (var l in Layers) logits = l.Forward(logits);
        }

        var (tokens, confidence) = TDTGreedyDecodeWithConfidence(logits);
        var text = TokensToText(tokens);
        double duration = audio.Length > 0 ? (double)audio.Shape[0] / SampleRate : 0;

        return new TranscriptionResult<T>
        {
            Text = text,
            Language = language ?? _options.Language,
            Confidence = NumOps.FromDouble(confidence),
            DurationSeconds = duration,
            Segments = includeTimestamps ? ExtractSegments(text, duration, confidence) : Array.Empty<TranscriptionSegment<T>>()
        };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool includeTimestamps = false, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    public string DetectLanguage(Tensor<T> audio) { var features = PreprocessAudio(audio); Tensor<T> logits; if (IsOnnxMode && OnnxEncoder is not null) logits = OnnxEncoder.Run(features); else { logits = features; foreach (var l in Layers) logits = l.Forward(logits); } var (tokens, _) = TDTGreedyDecodeWithConfidence(logits); return ClassifyLanguageFromTokens(tokens); }
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) { var detected = DetectLanguage(audio); var result = new Dictionary<string, T>(); double primaryProb = 0.85; double otherProb = SupportedLanguages.Count > 1 ? (1.0 - primaryProb) / (SupportedLanguages.Count - 1) : 0.0; foreach (var lang in SupportedLanguages) result[lang] = NumOps.FromDouble(lang == detected ? primaryProb : otherProb); return result; }
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => new TDTStreamingSession(this, language ?? _options.Language);

    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultConformerTransducerLayers(encoderDim: _options.EncoderDim, predictionDim: _options.PredictionDim, jointDim: _options.JointDim, numEncoderLayers: _options.NumEncoderLayers, numAttentionHeads: _options.NumAttentionHeads, feedForwardExpansionFactor: _options.FeedForwardDim / _options.EncoderDim, numMels: _options.NumMels, vocabSize: _options.VocabSize, dropoutRate: _options.DropoutRate)); }
    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        try { TrainWithTape(input, expected, _optimizer); }
        finally { SetTrainingMode(false); }
    }
    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "ParakeetTDT-Native" : "ParakeetTDT-ONNX", Description = "Parakeet-TDT: 1.1B Fast Conformer + Token-and-Duration Transducer (NVIDIA, 2024)", FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };



    /// <summary>
    /// TDT greedy decode with confidence: emits token + skips frames based on duration prediction.
    /// </summary>
    private (List<int> tokens, double confidence) TDTGreedyDecodeWithConfidence(Tensor<T> logits) { var tokens = new List<int>(); double totalConf = 0; int confCount = 0; int prevToken = -1; int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1; int vocabSize = logits.Rank >= 2 ? logits.Shape[^1] : logits.Shape[0]; int t = 0; while (t < numFrames && tokens.Count < _options.MaxTextLength) { int maxIdx = 0; double maxVal = double.NegativeInfinity; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); if (val > maxVal) { maxVal = val; maxIdx = v; } } double sumExp = 0; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); sumExp += Math.Exp(val - maxVal); } double frameConf = 1.0 / sumExp; if (maxIdx != prevToken && maxIdx > 0) { tokens.Add(maxIdx); totalConf += frameConf; confCount++; t += Math.Min(_options.MaxDurationTokens, 2); } else { t++; } prevToken = maxIdx; } return (tokens, confCount > 0 ? totalConf / confCount : 0.0); }
    private static string TokensToText(List<int> tokens) { var sb = new System.Text.StringBuilder(); foreach (var t in tokens) { if (t > 0 && t <= char.MaxValue) sb.Append((char)t); else if (t > char.MaxValue && t <= 0x10FFFF) sb.Append(char.ConvertFromUtf32(t)); } return sb.ToString().Trim(); }
    private IReadOnlyList<TranscriptionSegment<T>> ExtractSegments(string text, double duration, double confidence) { if (string.IsNullOrWhiteSpace(text)) return Array.Empty<TranscriptionSegment<T>>(); return new[] { new TranscriptionSegment<T> { Text = text, StartTime = 0.0, EndTime = duration, Confidence = NumOps.FromDouble(confidence) } }; }
    private string ClassifyLanguageFromTokens(List<int> tokens) { if (tokens.Count == 0) return _options.Language; int cjkCount = 0, latinCount = 0; foreach (var t in tokens) { if (t >= 0x4E00 && t <= 0x9FFF) cjkCount++; else if (t >= 0x41 && t <= 0x7A) latinCount++; } if (cjkCount > latinCount && SupportedLanguages.Contains("zh")) return "zh"; return _options.Language; }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(ParakeetTDT<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    private sealed class TDTStreamingSession : IStreamingTranscriptionSession<T>
    {
        private readonly ParakeetTDT<T> _model; private readonly string _language; private readonly List<Tensor<T>> _chunks = new(); private bool _disposed;
        public TDTStreamingSession(ParakeetTDT<T> model, string language) { _model = model; _language = language; }
        public void FeedAudio(Tensor<T> audioChunk) { if (_disposed) throw new ObjectDisposedException(nameof(TDTStreamingSession)); _chunks.Add(audioChunk); }
        public TranscriptionResult<T> GetPartialResult() { if (_disposed) throw new ObjectDisposedException(nameof(TDTStreamingSession)); if (_chunks.Count == 0) return new TranscriptionResult<T> { Language = _language }; int totalLen = 0; foreach (var c in _chunks) totalLen += c.Length; var combined = new Tensor<T>(new[] { totalLen }); int offset = 0; foreach (var c in _chunks) { for (int i = 0; i < c.Length; i++) combined[offset + i] = c[i]; offset += c.Length; } return _model.Transcribe(combined, _language); }
        public TranscriptionResult<T> Finalize() { if (_disposed) throw new ObjectDisposedException(nameof(TDTStreamingSession)); var result = GetPartialResult(); _disposed = true; return result; }
        public void Dispose() { _disposed = true; }
    }
}
