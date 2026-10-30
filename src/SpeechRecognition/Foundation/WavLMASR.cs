using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.SpeechRecognition.Foundation;

/// <summary>
/// WavLM ASR: speech pre-training with denoising and attention
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing" (Chen et al., Microsoft, 2022)</item></list></para>
/// <para><b>For Beginners:</b> WavLM extends HuBERT with: (1) gated relative position bias in self-attention for better sequence modeling; (2) denoising pre-training using overlapping utterances to learn robust representations in noisy conditions. The model is pre-trained on 94...</para>
/// <para>
/// WavLM extends HuBERT with: (1) gated relative position bias in self-attention for better sequence modeling; (2) denoising pre-training using overlapping utterances to learn robust representations in noisy conditions. The model is pre-trained on 94k hours of speech data. WavLM achieves state-of-the-art on the SUPERB benchmark across ASR, speaker verification, separation, and other tasks.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a WavLM ASR model for denoising-augmented speech pre-training
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputHeight: 16000, inputWidth: 1, inputDepth: 1, outputSize: 5000);
/// var model = new WavLMASR&lt;double&gt;(architecture);
///
/// // Or load a pre-trained ONNX model for robust speech recognition
/// var onnxModel = new WavLMASR&lt;double&gt;(architecture, "wavlmasr.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing", "https://arxiv.org/abs/2110.13900", Year = 2022, Authors = "Chen et al.")]
public partial class WavLMASR<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    private readonly WavLMASROptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode;
    private bool _disposed;
    // The paper network on the native path; null in ONNX mode or with custom layers.
    private Audio.Foundations.WavLmCtcStack<T>? _stack;
    private readonly Random _random;
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => false;
    public bool SupportsWordTimestamps => false;

    public WavLMASR(NeuralNetworkArchitecture<T> architecture, string modelPath, WavLMASROptions? options = null) : base(architecture) { _options = options ?? new WavLMASROptions(); Options = _options; _useNativeMode = false; base.SampleRate = _options.SampleRate; _random = Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1); base.NumMels = 0; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions); SupportedLanguages = new[] { "en" }; InitializeLayers(); }
    public WavLMASR(NeuralNetworkArchitecture<T> architecture, WavLMASROptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new WavLMASROptions(); Options = _options; _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; _random = Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1); base.NumMels = 0; SupportedLanguages = new[] { "en" }; InitializeLayers(); }

    /// <summary>
    /// Transcribes audio using WavLM's denoised SSL encoder with CTC decoding.
    /// Per Chen et al. (2022): audio is encoded with gated relative position bias attention
    /// and denoising pre-training objectives. CTC head produces per-frame token logits.
    /// </summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        Tensor<T> logits;

        logits = Logits(features, training: false);

        var (tokens, confidence) = CTCGreedyDecodeWithConfidence(logits);
        var text = TokensToText(tokens);
        double duration = audio.Length > 0 ? (double)audio.Shape[0] / SampleRate : 0;

        return new TranscriptionResult<T>
        {
            Text = text,
            Language = language ?? _options.Language,
            Confidence = NumOps.FromDouble(confidence),
            DurationSeconds = duration,
            Segments = ResolveReturnTimestamps(includeTimestamps) ? ExtractSegments(text, duration, tokens, confidence) : Array.Empty<TranscriptionSegment<T>>()
        };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    public string DetectLanguage(Tensor<T> audio) { var logits = Logits(PreprocessAudio(audio), training: false); var (tokens, _) = CTCGreedyDecodeWithConfidence(logits); return ClassifyLanguageFromTokens(tokens); }
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) { var detected = DetectLanguage(audio); var result = new Dictionary<string, T>(); double primaryProb = 0.85; double otherProb = SupportedLanguages.Count > 1 ? (1.0 - primaryProb) / (SupportedLanguages.Count - 1) : 0.0; foreach (var lang in SupportedLanguages) result[lang] = NumOps.FromDouble(lang == detected ? primaryProb : otherProb); return result; }
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("WavLMASR does not support streaming.");

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        _stack = null;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); return; }
        // The paper network: WavLM (convolutional feature encoder over the raw waveform, convolutional positional embedding, Transformer with the gated relative position bias) with a CTC head, as transformers' WavLMForCTC builds it. It used to be a generic encoder over mel features with the feature encoder approximated with dense (#2321).
        _stack = new Audio.Foundations.WavLmCtcStack<T>(Engine, _options.ToConfiguration(), _options.VocabSize, _options.Seed ?? 42);
        Layers.AddRange(_stack.Layers);
    }

    /// <summary>Loads a transformers <c>WavLMForCTC</c> state dict (<c>wavlm.*</c> and <c>lm_head.*</c>).</summary>
    /// <param name="read">Returns the named tensor's values, row-major, for the shape given.</param>
    /// <param name="has">Whether the state dict holds the named tensor (the positional convolution's weight norm comes
    /// under its old or its parametrized names).</param>
    public void LoadTorchWeights(Func<string, int[], double[]> read, Func<string, bool> has)
    {
        if (_stack is null) throw new InvalidOperationException("Weights load into the native network.");
        _stack.LoadTorchWeights(read, has);
    }

    // The network reads the raw waveform, truncated to the longest window a transcription covers.
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (_stack is null) return base.PreprocessAudio(rawAudio);
        int length = Math.Min(rawAudio.Length, SampleRate * _options.MaxAudioLengthSeconds);
        var waveform = new Tensor<T>(new[] { length });
        rawAudio.Data.Span.Slice(0, length).CopyTo(waveform.Data.Span);
        return waveform;
    }

    private Tensor<T> Logits(Tensor<T> input, bool training)
    {
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        if (_stack is not null)
            return _stack.Run(input, training ? _options.ToRegularization(_random) : null, _options.FinalDropout);
        var c = input; foreach (var l in Layers) c = l.Forward(c); return c;
    }

    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); return Logits(input, training: false); }

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => _stack is null ? base.ForwardForTraining(input) : Logits(input, training: true);

    protected override AiDotNet.Models.Parameters.ParameterSlotRole ResolveLayerParameterRole(int layerIndex, AiDotNet.Models.Parameters.ParameterSlotRole declared)
        => declared == AiDotNet.Models.Parameters.ParameterSlotRole.Trainable && _options.FreezeFeatureEncoder
            && _stack is not null && _stack.IsFeatureEncoderLayer(Layers[layerIndex])
            ? AiDotNet.Models.Parameters.ParameterSlotRole.Frozen
            : declared;
    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "WavLMASR-Native" : "WavLMASR-ONNX", Description = "WavLM: denoising SSL + CTC (Microsoft, 2022)", FeatureCount = 1, Complexity = _options.NumEncoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };



    private (List<int> tokens, double confidence) CTCGreedyDecodeWithConfidence(Tensor<T> logits) { var tokens = new List<int>(); double totalConf = 0; int confCount = 0; int prevToken = -1; int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1; int vocabSize = logits.Rank >= 2 ? logits.Shape[^1] : logits.Shape[0]; for (int t = 0; t < numFrames && tokens.Count < _options.MaxTextLength; t++) { int maxIdx = 0; double maxVal = double.NegativeInfinity; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); if (val > maxVal) { maxVal = val; maxIdx = v; } } double sumExp = 0; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); sumExp += Math.Exp(val - maxVal); } double frameConf = 1.0 / sumExp; if (maxIdx != prevToken && maxIdx > 0) { tokens.Add(maxIdx); totalConf += frameConf; confCount++; } prevToken = maxIdx; } return (tokens, confCount > 0 ? totalConf / confCount : 0.0); }
    private static string TokensToText(List<int> tokens) { var sb = new System.Text.StringBuilder(); foreach (var t in tokens) { if (t > 0 && t <= char.MaxValue) sb.Append((char)t); else if (t > char.MaxValue && t <= 0x10FFFF) sb.Append(char.ConvertFromUtf32(t)); } return sb.ToString().Trim(); }
    private IReadOnlyList<TranscriptionSegment<T>> ExtractSegments(string text, double duration, List<int> tokens, double confidence) { if (string.IsNullOrWhiteSpace(text) || tokens.Count == 0) return Array.Empty<TranscriptionSegment<T>>(); double timePerToken = duration / tokens.Count; var segments = new List<TranscriptionSegment<T>>(); var sb = new System.Text.StringBuilder(); double segStart = 0; for (int i = 0; i < tokens.Count; i++) { if (tokens[i] > 0 && tokens[i] <= char.MaxValue) sb.Append((char)tokens[i]); if ((tokens[i] == ' ' || i == tokens.Count - 1) && sb.Length > 0) { segments.Add(new TranscriptionSegment<T> { Text = sb.ToString().Trim(), StartTime = segStart, EndTime = (i + 1) * timePerToken, Confidence = NumOps.FromDouble(confidence) }); sb.Clear(); segStart = (i + 1) * timePerToken; } } if (segments.Count == 0) segments.Add(new TranscriptionSegment<T> { Text = text, StartTime = 0.0, EndTime = duration, Confidence = NumOps.FromDouble(confidence) }); return segments; }
    private string ClassifyLanguageFromTokens(List<int> tokens) { if (tokens.Count == 0) return _options.Language; int cjkCount = 0, latinCount = 0; foreach (var t in tokens) { if (t >= 0x4E00 && t <= 0x9FFF) cjkCount++; else if (t >= 0x41 && t <= 0x7A) latinCount++; } if (cjkCount > latinCount && SupportedLanguages.Contains("zh")) return "zh"; return _options.Language; }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(WavLMASR<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
