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
/// Conformer-Transducer: Conformer encoder with RNN-T/TDT decoder for streaming ASR.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Conformer: Convolution-augmented Transformer" (Gulati et al., 2020) + RNN-T (Graves, 2012)</item></list></para>
/// <para><b>For Beginners:</b> Combines the Conformer encoder's strong acoustic modeling with the RNN-T decoder's streaming capability. The prediction network maintains output history, the joint network combines encoder and prediction states, and the model autoregressively emit...</para>
/// <para>
/// Combines the Conformer encoder's strong acoustic modeling with the RNN-T decoder's
/// streaming capability. The prediction network maintains output history, the joint network
/// combines encoder and prediction states, and the model autoregressively emits tokens
/// frame by frame. Used in production at Google (Pixel phones) and NVIDIA Riva.
/// </para>
/// </remarks>
public class ConformerTransducer<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    private readonly ConformerTransducerOptions _options; public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    public IReadOnlyList<string> SupportedLanguages { get; private set; }
    public bool SupportsStreaming => true;
    public bool SupportsWordTimestamps => false;

    public ConformerTransducer(NeuralNetworkArchitecture<T> architecture, string modelPath, ConformerTransducerOptions? options = null) : base(architecture) { _options = options ?? new ConformerTransducerOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions); SupportedLanguages = new[] { _options.Language }; InitializeLayers(); }
    public ConformerTransducer(NeuralNetworkArchitecture<T> architecture, ConformerTransducerOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new ConformerTransducerOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; SupportedLanguages = new[] { _options.Language }; InitializeLayers(); }

    /// <summary>
    /// Transcribes audio using Conformer encoder + RNN-T decoder.
    /// Per the transducer framework: encoder produces acoustic features per frame,
    /// prediction network maintains label history, and the joint network combines both
    /// to produce output probability at each (time, label) position.
    /// Greedy decoding emits the most likely non-blank token at each step.
    /// </summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool includeTimestamps = false)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);

        if (IsOnnxMode && OnnxEncoder is not null)
        {
            var logits = OnnxEncoder.Run(features);
            var (tokens, conf) = GreedyDecodeWithConfidence(logits); var text = TokensToText(tokens);
            double dur = audio.Length > 0 ? (double)audio.Shape[0] / SampleRate : 0;
            return new TranscriptionResult<T> { Text = text, Language = language ?? _options.Language, Confidence = NumOps.FromDouble(conf), DurationSeconds = dur, Segments = includeTimestamps ? ExtractSegments(text, dur, conf) : Array.Empty<TranscriptionSegment<T>>() };
        }

        // Native: run through all layers (encoder + prediction + joint)
        var output = features;
        foreach (var l in Layers) output = l.Forward(output);

        var (decodedTokens, decodedConf) = GreedyDecodeWithConfidence(output);
        var decodedText = TokensToText(decodedTokens);
        double duration = audio.Length > 0 ? (double)audio.Shape[0] / SampleRate : 0;
        return new TranscriptionResult<T> { Text = decodedText, Language = language ?? _options.Language, Confidence = NumOps.FromDouble(decodedConf), DurationSeconds = duration, Segments = includeTimestamps ? ExtractSegments(decodedText, duration, decodedConf) : Array.Empty<TranscriptionSegment<T>>() };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool includeTimestamps = false, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    public string DetectLanguage(Tensor<T> audio) { var features = PreprocessAudio(audio); Tensor<T> logits; if (IsOnnxMode && OnnxEncoder is not null) logits = OnnxEncoder.Run(features); else { logits = features; foreach (var l in Layers) logits = l.Forward(logits); } var (tokens, _) = GreedyDecodeWithConfidence(logits); return ClassifyLanguageFromTokens(tokens); }
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio)
    {
        var detected = DetectLanguage(audio);
        var result = new Dictionary<string, T>();
        foreach (var lang in SupportedLanguages)
            result[lang] = NumOps.FromDouble(lang == detected ? 1.0 : 0.0);
        return result;
    }
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => new TransducerStreamingSession(this, language ?? _options.Language);

    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultConformerTransducerLayers(encoderDim: _options.EncoderDim, numEncoderLayers: _options.NumEncoderLayers, numAttentionHeads: _options.NumAttentionHeads, feedForwardExpansionFactor: _options.FeedForwardExpansionFactor, predictionDim: _options.PredictionDim, jointDim: _options.JointDim, numMels: _options.NumMels, vocabSize: _options.VocabSize, dropoutRate: _options.DropoutRate)); }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) { if (MelSpec is not null) return MelSpec.Forward(rawAudio); return rawAudio; }
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "ConformerTransducer-Native" : "ConformerTransducer-ONNX", Description = "Conformer-Transducer: Streaming ASR (Gulati+Graves)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers };
    protected override void SerializeNetworkSpecificData(BinaryWriter w) { w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty); w.Write(_options.SampleRate); w.Write(_options.MaxAudioLengthSeconds); w.Write(_options.EncoderDim); w.Write(_options.NumEncoderLayers); w.Write(_options.NumAttentionHeads); w.Write(_options.FeedForwardExpansionFactor); w.Write(_options.PredictionDim); w.Write(_options.JointDim); w.Write(_options.NumMels); w.Write(_options.VocabSize); w.Write(_options.DropoutRate); w.Write(_options.Language); }
    protected override void DeserializeNetworkSpecificData(BinaryReader r) { _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = r.ReadInt32(); _options.MaxAudioLengthSeconds = r.ReadInt32(); _options.EncoderDim = r.ReadInt32(); _options.NumEncoderLayers = r.ReadInt32(); _options.NumAttentionHeads = r.ReadInt32(); _options.FeedForwardExpansionFactor = r.ReadInt32(); _options.PredictionDim = r.ReadInt32(); _options.JointDim = r.ReadInt32(); _options.NumMels = r.ReadInt32(); _options.VocabSize = r.ReadInt32(); _options.DropoutRate = r.ReadDouble(); _options.Language = r.ReadString(); base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; SupportedLanguages = new[] { _options.Language }; if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new ConformerTransducer<T>(Architecture, mp, _options); return new ConformerTransducer<T>(Architecture, _options); }
    /// <summary>
    /// Greedy decoding on output logits. For ONNX models the prediction+joint networks are
    /// internal to the ONNX graph, so the output is already joint logits. For native mode,
    /// the layer stack includes prediction and joint layers.
    /// </summary>
    private (List<int> tokens, double confidence) GreedyDecodeWithConfidence(Tensor<T> logits) { var tokens = new List<int>(); double totalConf = 0; int confCount = 0; int prevToken = -1; int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1; int vocabSize = logits.Rank >= 2 ? logits.Shape[^1] : logits.Shape[0]; for (int t = 0; t < numFrames; t++) { int maxIdx = 0; double maxVal = double.NegativeInfinity; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); if (val > maxVal) { maxVal = val; maxIdx = v; } } double sumExp = 0; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); sumExp += Math.Exp(val - maxVal); } double frameConf = 1.0 / sumExp; if (maxIdx != prevToken && maxIdx > 0) { tokens.Add(maxIdx); totalConf += frameConf; confCount++; } prevToken = maxIdx; } return (tokens, confCount > 0 ? totalConf / confCount : 0.0); }
    private static string TokensToText(List<int> tokens) { var sb = new System.Text.StringBuilder(); foreach (var t in tokens) { if (t > 0 && t <= char.MaxValue) sb.Append((char)t); else if (t > char.MaxValue && t <= 0x10FFFF) sb.Append(char.ConvertFromUtf32(t)); } return sb.ToString().Trim(); }
    private IReadOnlyList<TranscriptionSegment<T>> ExtractSegments(string text, double duration, double confidence) { if (string.IsNullOrWhiteSpace(text)) return Array.Empty<TranscriptionSegment<T>>(); return new[] { new TranscriptionSegment<T> { Text = text, StartTime = 0.0, EndTime = duration, Confidence = NumOps.FromDouble(confidence) } }; }
    private string ClassifyLanguageFromTokens(List<int> tokens) { if (tokens.Count == 0) return _options.Language; int cjkCount = 0, latinCount = 0; foreach (var t in tokens) { if (t >= 0x4E00 && t <= 0x9FFF) cjkCount++; else if (t >= 0x41 && t <= 0x7A) latinCount++; } if (cjkCount > latinCount && SupportedLanguages.Contains("zh")) return "zh"; return _options.Language; }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(ConformerTransducer<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    private sealed class TransducerStreamingSession : IStreamingTranscriptionSession<T>
    {
        private readonly ConformerTransducer<T> _model;
        private readonly string _language;
        private readonly List<Tensor<T>> _chunks = new();
        private readonly object _lock = new();
        private bool _disposed;

        public TransducerStreamingSession(ConformerTransducer<T> model, string language) { _model = model; _language = language; }

        public void FeedAudio(Tensor<T> audioChunk)
        {
            lock (_lock)
            {
                if (_disposed) throw new ObjectDisposedException(nameof(TransducerStreamingSession));
                _chunks.Add(audioChunk);
            }
        }

        public TranscriptionResult<T> GetPartialResult()
        {
            List<Tensor<T>> snapshot;
            lock (_lock)
            {
                if (_disposed) throw new ObjectDisposedException(nameof(TransducerStreamingSession));
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
                if (_disposed) throw new ObjectDisposedException(nameof(TransducerStreamingSession));
                snapshot = new List<Tensor<T>>(_chunks);
                _disposed = true;
            }
            if (snapshot.Count == 0) return new TranscriptionResult<T> { Language = _language };
            return TranscribeSnapshot(snapshot);
        }

        public void Dispose() { lock (_lock) { _disposed = true; } }

        private TranscriptionResult<T> TranscribeSnapshot(List<Tensor<T>> snapshot)
        {
            int totalLen = 0;
            foreach (var ch in snapshot) totalLen += ch.Length;
            var combined = new Tensor<T>(new[] { totalLen });
            int offset = 0;
            foreach (var ch in snapshot) { for (int i = 0; i < ch.Length; i++) combined[offset + i] = ch[i]; offset += ch.Length; }
            return _model.Transcribe(combined, _language);
        }
    }
}
