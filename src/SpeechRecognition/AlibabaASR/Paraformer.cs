using AiDotNet.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.SpeechRecognition.AlibabaASR;

/// <summary>
/// Paraformer: fast and accurate parallel Transformer for non-autoregressive ASR
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition" (Gao et al., Alibaba DAMO, 2022)</item></list></para>
/// <para><b>For Beginners:</b> Paraformer uses Continuous Integrate-and-Fire (CIF) to predict token counts and extract acoustic embeddings in a single forward pass, enabling non-autoregressive parallel decoding. The CIF module accumulates encoder hidden states weighted by learn...</para>
/// <para>
/// Paraformer uses Continuous Integrate-and-Fire (CIF) to predict token counts and extract acoustic embeddings in a single forward pass, enabling non-autoregressive parallel decoding. The CIF module accumulates encoder hidden states weighted by learned firing probabilities. When cumulative weight exceeds a threshold, an acoustic embedding is emitted. A glancing language model (GLM) decoder then generates all tokens in parallel from these embeddings. This achieves comparable accuracy to autoregressive models with much lower latency.
/// </para>
/// </remarks>
public class Paraformer<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    private readonly ParaformerOptions _options; public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => false;
    public bool SupportsWordTimestamps => false;

    public Paraformer(NeuralNetworkArchitecture<T> architecture, string modelPath, ParaformerOptions? options = null) : base(architecture) { _options = options ?? new ParaformerOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions); SupportedLanguages = new[] { "en" }; InitializeLayers(); }
    public Paraformer(NeuralNetworkArchitecture<T> architecture, ParaformerOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new ParaformerOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; SupportedLanguages = new[] { "en" }; InitializeLayers(); }

    /// <summary>
    /// Transcribes audio using CIF-based parallel decoding.
    /// Per Gao et al. (2022): the Conformer encoder processes mel features, CIF predicts
    /// token count and extracts acoustic embeddings, then the GLM decoder generates all
    /// tokens in a single parallel forward pass.
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

        var (tokens, confidence) = GreedyDecodeWithConfidence(logits);
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

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool includeTimestamps = false, CancellationToken cancellationToken = default) => Task.Run(() => { cancellationToken.ThrowIfCancellationRequested(); return Transcribe(audio, language, includeTimestamps); }, cancellationToken);
    public string DetectLanguage(Tensor<T> audio) { var features = PreprocessAudio(audio); Tensor<T> logits; if (IsOnnxMode && OnnxEncoder is not null) logits = OnnxEncoder.Run(features); else { logits = features; foreach (var l in Layers) logits = l.Forward(logits); } var (tokens, _) = GreedyDecodeWithConfidence(logits); return ClassifyLanguageFromTokens(tokens); }
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio)
    {
        var detected = DetectLanguage(audio);
        var result = new Dictionary<string, T>();
        foreach (var lang in SupportedLanguages)
            result[lang] = NumOps.FromDouble(lang == detected ? 1.0 : 0.0);
        return result;
    }
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("Paraformer does not support streaming.");

    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultParaformerLayers(encoderDim: _options.EncoderDim, numEncoderLayers: _options.NumEncoderLayers, numAttentionHeads: _options.NumAttentionHeads, numMels: _options.NumMels, vocabSize: _options.VocabSize, dropoutRate: _options.DropoutRate)); }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) { if (MelSpec is not null) return MelSpec.Forward(rawAudio); return rawAudio; }
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "Paraformer-Native" : "Paraformer-ONNX", Description = "Paraformer: CIF + parallel Transformer (Alibaba DAMO, 2022)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers };
    protected override void SerializeNetworkSpecificData(BinaryWriter w) { w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty); w.Write(_options.SampleRate); w.Write(_options.MaxAudioLengthSeconds); w.Write(_options.EncoderDim); w.Write(_options.NumEncoderLayers); w.Write(_options.NumAttentionHeads); w.Write(_options.NumMels); w.Write(_options.VocabSize); w.Write(_options.MaxTextLength); w.Write(_options.DropoutRate); w.Write(_options.Language); }
    protected override void DeserializeNetworkSpecificData(BinaryReader r) { _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = r.ReadInt32(); _options.MaxAudioLengthSeconds = r.ReadInt32(); _options.EncoderDim = r.ReadInt32(); _options.NumEncoderLayers = r.ReadInt32(); _options.NumAttentionHeads = r.ReadInt32(); _options.NumMels = r.ReadInt32(); _options.VocabSize = r.ReadInt32(); _options.MaxTextLength = r.ReadInt32(); _options.DropoutRate = r.ReadDouble(); _options.Language = r.ReadString(); base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new Paraformer<T>(Architecture, mp, _options); return new Paraformer<T>(Architecture, _options); }

    private (List<int> tokens, double confidence) GreedyDecodeWithConfidence(Tensor<T> logits) { var tokens = new List<int>(); double totalConf = 0; int confCount = 0; int prevToken = -1; int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1; int vocabSize = logits.Rank >= 2 ? logits.Shape[^1] : logits.Shape[0]; for (int t = 0; t < numFrames && tokens.Count < _options.MaxTextLength; t++) { int maxIdx = 0; double maxVal = double.NegativeInfinity; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); if (val > maxVal) { maxVal = val; maxIdx = v; } } double sumExp = 0; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); sumExp += Math.Exp(val - maxVal); } double frameConf = 1.0 / sumExp; if (maxIdx != prevToken && maxIdx > 0) { tokens.Add(maxIdx); totalConf += frameConf; confCount++; } prevToken = maxIdx; } return (tokens, confCount > 0 ? totalConf / confCount : 0.0); }
    private string TokensToText(List<int> tokens)
    {
        var vocab = _options.Vocabulary;
        if (vocab is not null && vocab.Length > 0)
        {
            var sb = new System.Text.StringBuilder();
            foreach (var t in tokens)
            {
                if (t > 0 && t < vocab.Length) sb.Append(vocab[t]);
            }
            return sb.ToString().Trim();
        }
        // Fallback: Unicode codepoint mapping for models that use Unicode-based tokens
        var fb = new System.Text.StringBuilder();
        foreach (var t in tokens)
        {
            if (t > 0 && t <= char.MaxValue) fb.Append((char)t);
            else if (t > char.MaxValue && t <= 0x10FFFF) fb.Append(char.ConvertFromUtf32(t));
        }
        return fb.ToString().Trim();
    }
    private IReadOnlyList<TranscriptionSegment<T>> ExtractSegments(string text, double duration, double confidence) { if (string.IsNullOrWhiteSpace(text)) return Array.Empty<TranscriptionSegment<T>>(); return new[] { new TranscriptionSegment<T> { Text = text, StartTime = 0.0, EndTime = duration, Confidence = NumOps.FromDouble(confidence) } }; }
    private string ClassifyLanguageFromTokens(List<int> _) => _options.Language;
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Paraformer<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
