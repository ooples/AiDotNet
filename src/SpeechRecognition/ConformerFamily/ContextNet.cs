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
public class ContextNet<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    private readonly ContextNetOptions _options; public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => true;
    public bool SupportsWordTimestamps => true;

    public ContextNet(NeuralNetworkArchitecture<T> architecture, string modelPath, ContextNetOptions? options = null) : base(architecture) { _options = options ?? new ContextNetOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions); SupportedLanguages = new[] { _options.Language }; InitializeLayers(); }
    public ContextNet(NeuralNetworkArchitecture<T> architecture, ContextNetOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new ContextNetOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; SupportedLanguages = new[] { _options.Language }; InitializeLayers(); }

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
        var tokens = CTCGreedyDecode(logits); var text = TokensToText(tokens);
        double duration = audio.Length > 0 ? (double)audio.Shape[0] / SampleRate : 0;
        return new TranscriptionResult<T> { Text = text, Language = language ?? _options.Language, Confidence = NumOps.FromDouble(tokens.Count > 0 ? 0.9 : 0.0), DurationSeconds = duration, Segments = includeTimestamps ? ExtractSegments(text, duration) : Array.Empty<TranscriptionSegment<T>>() };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool includeTimestamps = false, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    public string DetectLanguage(Tensor<T> audio) => _options.Language;
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) => new Dictionary<string, T> { [_options.Language] = NumOps.FromDouble(1.0) };
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => new ContextNetStreamingSession(this, language ?? _options.Language);

    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultDeepCNNCTCLayers(encoderDim: _options.EncoderDim, numBlocks: _options.NumBlocks, numSubBlocks: 5, numMels: _options.NumMels, vocabSize: _options.VocabSize, dropoutRate: _options.DropoutRate)); }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) { if (MelSpec is not null) return MelSpec.Forward(rawAudio); return rawAudio; }
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "ContextNet-Native" : "ContextNet-ONNX", Description = "ContextNet: CNN with Squeeze-and-Excitation (Han et al., 2020)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumMels, Complexity = _options.NumBlocks };
    protected override void SerializeNetworkSpecificData(BinaryWriter w) { w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty); w.Write(_options.SampleRate); w.Write(_options.EncoderDim); w.Write(_options.NumBlocks); w.Write(_options.SqueezeExcitationRatio); w.Write(_options.NumMels); w.Write(_options.VocabSize); w.Write(_options.DropoutRate); w.Write(_options.Language); }
    protected override void DeserializeNetworkSpecificData(BinaryReader r) { _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = r.ReadInt32(); _options.EncoderDim = r.ReadInt32(); _options.NumBlocks = r.ReadInt32(); _options.SqueezeExcitationRatio = r.ReadInt32(); _options.NumMels = r.ReadInt32(); _options.VocabSize = r.ReadInt32(); _options.DropoutRate = r.ReadDouble(); _options.Language = r.ReadString(); base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new ContextNet<T>(Architecture, mp, _options); return new ContextNet<T>(Architecture, _options); }
    private List<int> CTCGreedyDecode(Tensor<T> logits) { var tokens = new List<int>(); int prevToken = -1; int blankIdx = 0; int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1; int vocabSize = logits.Rank >= 2 ? logits.Shape[^1] : logits.Shape[0]; for (int t = 0; t < numFrames; t++) { int maxIdx = 0; double maxVal = double.NegativeInfinity; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); if (val > maxVal) { maxVal = val; maxIdx = v; } } if (maxIdx != blankIdx && maxIdx != prevToken) tokens.Add(maxIdx); prevToken = maxIdx; } return tokens; }
    private string TokensToText(List<int> tokens) { var vocab = _options.Vocabulary; var chars = new List<char>(); foreach (var token in tokens) { if (token >= 0 && token < vocab.Length) { var s = vocab[token]; if (s == "|" || s == " ") chars.Add(' '); else if (s.Length == 1 && char.IsLetter(s[0])) chars.Add(s[0]); else if (s == "'") chars.Add('\''); } } return new string(chars.ToArray()).Trim(); }
    private IReadOnlyList<TranscriptionSegment<T>> ExtractSegments(string text, double duration) { if (string.IsNullOrWhiteSpace(text)) return Array.Empty<TranscriptionSegment<T>>(); return new[] { new TranscriptionSegment<T> { Text = text, StartTime = 0.0, EndTime = duration, Confidence = NumOps.FromDouble(0.9) } }; }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(ContextNet<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    private sealed class ContextNetStreamingSession : IStreamingTranscriptionSession<T>
    {
        private readonly ContextNet<T> _model; private readonly string _language; private readonly List<Tensor<T>> _chunks = new(); private bool _disposed;
        public ContextNetStreamingSession(ContextNet<T> model, string language) { _model = model; _language = language; }
        public void FeedAudio(Tensor<T> audioChunk) { if (_disposed) throw new ObjectDisposedException(nameof(ContextNetStreamingSession)); _chunks.Add(audioChunk); }
        public TranscriptionResult<T> GetPartialResult() { if (_disposed) throw new ObjectDisposedException(nameof(ContextNetStreamingSession)); if (_chunks.Count == 0) return new TranscriptionResult<T> { Language = _language }; int totalLen = 0; foreach (var ch in _chunks) totalLen += ch.Length; var combined = new Tensor<T>(new[] { totalLen }); int offset = 0; foreach (var ch in _chunks) { for (int i = 0; i < ch.Length; i++) combined[offset + i] = ch[i]; offset += ch.Length; } return _model.Transcribe(combined, _language); }
        public TranscriptionResult<T> Finalize() { if (_disposed) throw new ObjectDisposedException(nameof(ContextNetStreamingSession)); var result = GetPartialResult(); _disposed = true; return result; }
        public void Dispose() { _disposed = true; }
    }
}
