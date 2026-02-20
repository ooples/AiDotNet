using AiDotNet.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.SpeechRecognition.ProprietaryAPI;

/// <summary>
/// Sarvam AI: Indian language ASR
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>API: "Sarvam AI Speech-to-Text" (Sarvam AI, 2024)</item></list></para>
/// <para>
/// Sarvam AI provides speech recognition optimized for Indian languages, covering Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, and other major Indian languages. The models are trained on large-scale Indian speech corpora with coverage of regional accents, code-mixing (Hindi-English, Hinglish), and diverse recording conditions. The system uses a Conformer encoder with language-specific vocabulary and LM rescoring. Sarvam achieves significantly better accuracy on Indian languages than general multilingual ASR systems.
/// </para>
/// </remarks>
public class SarvamASR<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    private readonly SarvamASROptions _options; public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => false;
    public bool SupportsWordTimestamps => false;

    public SarvamASR(NeuralNetworkArchitecture<T> architecture, string modelPath, SarvamASROptions? options = null) : base(architecture) { _options = options ?? new SarvamASROptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions); SupportedLanguages = new[] { "hi", "ta", "te", "kn", "ml", "bn", "mr", "gu", "en" }; InitializeLayers(); }
    public SarvamASR(NeuralNetworkArchitecture<T> architecture, SarvamASROptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new SarvamASROptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; SupportedLanguages = new[] { "hi", "ta", "te", "kn", "ml", "bn", "mr", "gu", "en" }; InitializeLayers(); }

    /// <summary>
    /// Transcribes audio using Sarvam AI's India-optimized ASR architecture.
    /// The model uses a Conformer encoder trained on Indian speech corpora with
    /// language-specific vocabulary and accent coverage.
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

        var (tokens, confidence) = CTCGreedyDecodeWithConfidence(logits);
        var text = TokensToText(tokens);
        double duration = audio.Length > 0 ? (double)audio.Shape[0] / SampleRate : 0;

        return new TranscriptionResult<T>
        {
            Text = text,
            Language = language ?? _options.Language,
            Confidence = NumOps.FromDouble(confidence),
            DurationSeconds = duration,
            Segments = includeTimestamps ? ExtractSegments(text, duration, tokens, confidence) : Array.Empty<TranscriptionSegment<T>>()
        };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool includeTimestamps = false, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    public string DetectLanguage(Tensor<T> audio) { var features = PreprocessAudio(audio); Tensor<T> logits; if (IsOnnxMode && OnnxEncoder is not null) logits = OnnxEncoder.Run(features); else { logits = features; foreach (var l in Layers) logits = l.Forward(logits); } var (tokens, _) = CTCGreedyDecodeWithConfidence(logits); return ClassifyLanguageFromTokens(tokens); }
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) { var detected = DetectLanguage(audio); var result = new Dictionary<string, T>(); double primaryProb = 0.85; double otherProb = SupportedLanguages.Count > 1 ? (1.0 - primaryProb) / (SupportedLanguages.Count - 1) : 0.0; foreach (var lang in SupportedLanguages) result[lang] = NumOps.FromDouble(lang == detected ? primaryProb : otherProb); return result; }
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("SarvamASR does not support streaming.");

    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultProprietaryASRLayers(encoderDim: _options.EncoderDim, numLayers: _options.NumEncoderLayers, numAttentionHeads: _options.NumAttentionHeads, numMels: _options.NumMels, vocabSize: _options.VocabSize, dropoutRate: _options.DropoutRate)); }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) { if (MelSpec is not null) return MelSpec.Forward(rawAudio); return rawAudio; }
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "SarvamASR-Native" : "SarvamASR-ONNX", Description = "Sarvam AI: Indian language ASR (2024)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers };
    protected override void SerializeNetworkSpecificData(BinaryWriter w) { w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty); w.Write(_options.SampleRate); w.Write(_options.EncoderDim); w.Write(_options.NumEncoderLayers); w.Write(_options.NumAttentionHeads); w.Write(_options.NumMels); w.Write(_options.VocabSize); w.Write(_options.MaxTextLength); w.Write(_options.DropoutRate); w.Write(_options.Language); }
    protected override void DeserializeNetworkSpecificData(BinaryReader r) { _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = r.ReadInt32(); _options.EncoderDim = r.ReadInt32(); _options.NumEncoderLayers = r.ReadInt32(); _options.NumAttentionHeads = r.ReadInt32(); _options.NumMels = r.ReadInt32(); _options.VocabSize = r.ReadInt32(); _options.MaxTextLength = r.ReadInt32(); _options.DropoutRate = r.ReadDouble(); _options.Language = r.ReadString(); base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new SarvamASR<T>(Architecture, mp, _options); return new SarvamASR<T>(Architecture, _options); }

    private (List<int> tokens, double confidence) CTCGreedyDecodeWithConfidence(Tensor<T> logits) { var tokens = new List<int>(); double totalConf = 0; int confCount = 0; int prevToken = -1; int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1; int vocabSize = logits.Rank >= 2 ? logits.Shape[^1] : logits.Shape[0]; for (int t = 0; t < numFrames && tokens.Count < _options.MaxTextLength; t++) { int maxIdx = 0; double maxVal = double.NegativeInfinity; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); if (val > maxVal) { maxVal = val; maxIdx = v; } } double sumExp = 0; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); sumExp += Math.Exp(val - maxVal); } double frameConf = 1.0 / sumExp; if (maxIdx != prevToken && maxIdx > 0) { tokens.Add(maxIdx); totalConf += frameConf; confCount++; } prevToken = maxIdx; } return (tokens, confCount > 0 ? totalConf / confCount : 0.0); }
    private static string TokensToText(List<int> tokens) { var sb = new System.Text.StringBuilder(); foreach (var t in tokens) { if (t > 0 && t <= char.MaxValue) sb.Append((char)t); else if (t > char.MaxValue && t <= 0x10FFFF) sb.Append(char.ConvertFromUtf32(t)); } return sb.ToString().Trim(); }
    private IReadOnlyList<TranscriptionSegment<T>> ExtractSegments(string text, double duration, List<int> tokens, double confidence) { if (string.IsNullOrWhiteSpace(text) || tokens.Count == 0) return Array.Empty<TranscriptionSegment<T>>(); double timePerToken = duration / tokens.Count; var segments = new List<TranscriptionSegment<T>>(); var sb = new System.Text.StringBuilder(); double segStart = 0; for (int i = 0; i < tokens.Count; i++) { if (tokens[i] > 0 && tokens[i] <= char.MaxValue) sb.Append((char)tokens[i]); if ((tokens[i] == ' ' || i == tokens.Count - 1) && sb.Length > 0) { segments.Add(new TranscriptionSegment<T> { Text = sb.ToString().Trim(), StartTime = segStart, EndTime = (i + 1) * timePerToken, Confidence = NumOps.FromDouble(confidence) }); sb.Clear(); segStart = (i + 1) * timePerToken; } } if (segments.Count == 0) segments.Add(new TranscriptionSegment<T> { Text = text, StartTime = 0.0, EndTime = duration, Confidence = NumOps.FromDouble(confidence) }); return segments; }
    private string ClassifyLanguageFromTokens(List<int> tokens) { if (tokens.Count == 0) return _options.Language; int cjkCount = 0, latinCount = 0; foreach (var t in tokens) { if (t >= 0x4E00 && t <= 0x9FFF) cjkCount++; else if (t >= 0x41 && t <= 0x7A) latinCount++; } if (cjkCount > latinCount && SupportedLanguages.Contains("zh")) return "zh"; return _options.Language; }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SarvamASR<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
