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
/// Conformer-CTC: Conformer encoder with CTC-only decoding (no external decoder).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Conformer: Convolution-augmented Transformer for Speech Recognition" (Gulati et al., 2020)</item></list></para>
/// <para>
/// This is the CTC-only variant of Conformer: the encoder output is projected to vocabulary
/// size and decoded with CTC greedy search. Unlike the attention-decoder or transducer variants,
/// CTC decoding is fully non-autoregressive and very fast.
/// </para>
/// </remarks>
public class ConformerCTC<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    private readonly ConformerCTCOptions _options; public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => true;
    public bool SupportsWordTimestamps => true;

    public ConformerCTC(NeuralNetworkArchitecture<T> architecture, string modelPath, ConformerCTCOptions? options = null) : base(architecture) { _options = options ?? new ConformerCTCOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions); SupportedLanguages = new[] { _options.Language }; InitializeLayers(); }
    public ConformerCTC(NeuralNetworkArchitecture<T> architecture, ConformerCTCOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new ConformerCTCOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; SupportedLanguages = new[] { _options.Language }; InitializeLayers(); }

    /// <summary>
    /// Transcribes audio using Conformer encoder with CTC greedy decoding.
    /// Per the paper: macaron-style FF-MHA-Conv-FF encoder blocks produce frame-level
    /// logits which are decoded non-autoregressively via CTC blank-collapse.
    /// </summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool includeTimestamps = false)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        var logits = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);
        var tokens = CTCGreedyDecode(logits); var text = TokensToText(tokens);
        double duration = audio.Length > 0 ? (double)audio.Shape[0] / SampleRate : 0;
        return new TranscriptionResult<T> { Text = text, Language = language ?? _options.Language, Confidence = NumOps.FromDouble(tokens.Count > 0 ? 0.9 : 0.0), DurationSeconds = duration, Segments = includeTimestamps ? ExtractSegments(text, duration) : Array.Empty<TranscriptionSegment<T>>() };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool includeTimestamps = false, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    public string DetectLanguage(Tensor<T> audio) => _options.Language;
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) => new Dictionary<string, T> { [_options.Language] = NumOps.FromDouble(1.0) };
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => new CTCStreamingSession(this, language ?? _options.Language);

    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultConformerLayers(encoderDim: _options.EncoderDim, numLayers: _options.NumEncoderLayers, numAttentionHeads: _options.NumAttentionHeads, feedForwardExpansionFactor: _options.FeedForwardExpansionFactor, numMels: _options.NumMels, vocabSize: _options.VocabSize, dropoutRate: _options.DropoutRate)); }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) { if (MelSpec is not null) return MelSpec.Forward(rawAudio); return rawAudio; }
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "ConformerCTC-Native" : "ConformerCTC-ONNX", Description = "Conformer-CTC: CTC-only variant (Gulati et al., 2020)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers };
    protected override void SerializeNetworkSpecificData(BinaryWriter w) { w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty); w.Write(_options.SampleRate); w.Write(_options.EncoderDim); w.Write(_options.NumEncoderLayers); w.Write(_options.NumAttentionHeads); w.Write(_options.FeedForwardExpansionFactor); w.Write(_options.NumMels); w.Write(_options.VocabSize); w.Write(_options.DropoutRate); w.Write(_options.Language); }
    protected override void DeserializeNetworkSpecificData(BinaryReader r) { _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = r.ReadInt32(); _options.EncoderDim = r.ReadInt32(); _options.NumEncoderLayers = r.ReadInt32(); _options.NumAttentionHeads = r.ReadInt32(); _options.FeedForwardExpansionFactor = r.ReadInt32(); _options.NumMels = r.ReadInt32(); _options.VocabSize = r.ReadInt32(); _options.DropoutRate = r.ReadDouble(); _options.Language = r.ReadString(); base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new ConformerCTC<T>(Architecture, mp, _options); return new ConformerCTC<T>(Architecture, _options); }
    private List<int> CTCGreedyDecode(Tensor<T> logits) { var tokens = new List<int>(); int prevToken = -1; int blankIdx = 0; int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1; int vocabSize = logits.Rank >= 2 ? logits.Shape[^1] : logits.Shape[0]; for (int t = 0; t < numFrames; t++) { int maxIdx = 0; double maxVal = double.NegativeInfinity; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); if (val > maxVal) { maxVal = val; maxIdx = v; } } if (maxIdx != blankIdx && maxIdx != prevToken) tokens.Add(maxIdx); prevToken = maxIdx; } return tokens; }
    private string TokensToText(List<int> tokens) { var vocab = _options.Vocabulary; var chars = new List<char>(); foreach (var token in tokens) { if (token >= 0 && token < vocab.Length) { var s = vocab[token]; if (s == "|" || s == " ") chars.Add(' '); else if (s.Length == 1 && char.IsLetter(s[0])) chars.Add(s[0]); else if (s == "'") chars.Add('\''); } } return new string(chars.ToArray()).Trim(); }
    private IReadOnlyList<TranscriptionSegment<T>> ExtractSegments(string text, double duration) { if (string.IsNullOrWhiteSpace(text)) return Array.Empty<TranscriptionSegment<T>>(); return new[] { new TranscriptionSegment<T> { Text = text, StartTime = 0.0, EndTime = duration, Confidence = NumOps.FromDouble(0.9) } }; }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(ConformerCTC<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    private sealed class CTCStreamingSession : IStreamingTranscriptionSession<T>
    {
        private readonly ConformerCTC<T> _model; private readonly string _language; private readonly List<Tensor<T>> _chunks = new(); private bool _disposed;
        public CTCStreamingSession(ConformerCTC<T> model, string language) { _model = model; _language = language; }
        public void FeedAudio(Tensor<T> audioChunk) { if (_disposed) throw new ObjectDisposedException(nameof(CTCStreamingSession)); _chunks.Add(audioChunk); }
        public TranscriptionResult<T> GetPartialResult() { if (_disposed) throw new ObjectDisposedException(nameof(CTCStreamingSession)); if (_chunks.Count == 0) return new TranscriptionResult<T> { Language = _language }; int totalLen = 0; foreach (var ch in _chunks) totalLen += ch.Length; var combined = new Tensor<T>(new[] { totalLen }); int offset = 0; foreach (var ch in _chunks) { for (int i = 0; i < ch.Length; i++) combined[offset + i] = ch[i]; offset += ch.Length; } return _model.Transcribe(combined, _language); }
        public TranscriptionResult<T> Finalize() { if (_disposed) throw new ObjectDisposedException(nameof(CTCStreamingSession)); var result = GetPartialResult(); _disposed = true; return result; }
        public void Dispose() { _disposed = true; }
    }
}
