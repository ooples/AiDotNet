using AiDotNet.LearningRateSchedulers;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.SpeechRecognition.Multilingual;

/// <summary>
/// Omnilingual ASR's CTC recognizer: a wav2vec 2.0 encoder over the raw waveform with a single linear CTC layer.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Omnilingual ASR: Open-Source Multilingual Speech Recognition for 1600+ Languages"
/// (Omnilingual ASR team, Meta, 2025)</item></list></para>
/// <para><b>For Beginners:</b> The model listens to raw 16 kHz audio and outputs, for every 20 ms frame, a probability
/// for each character (plus a "blank"). Collapsing repeats and dropping blanks gives the transcript.</para>
/// <para>
/// Omnilingual ASR pre-trains wav2vec 2.0 encoders of 0.3B, 1B, 3B and 7B parameters on 4.3M hours of unlabeled speech in
/// over 1,600 languages (section 4.1), then builds two recognizers on them (section 4.2): a CTC model that adds a single
/// linear layer and fine-tunes the whole network with a CTC loss, and an LLM-ASR model with a Transformer decoder. This
/// class is the CTC model. Its defaults are the 300M encoder: 24 layers, 1024 wide, 16 heads, 4096 feed-forward (Table 4).
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an Omnilingual ASR model for universal multilingual recognition
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputHeight: 16000, inputWidth: 1, inputDepth: 1, outputSize: 9812);
/// var model = new OmnilangualASR&lt;double&gt;(architecture);
///
/// // Or load an exported ONNX model
/// var onnxModel = new OmnilangualASR&lt;double&gt;(architecture, "omnilangual.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Omnilingual ASR: Open-Source Multilingual Speech Recognition for 1600+ Languages", "https://arxiv.org/abs/2511.09690", Year = 2025, Authors = "Omnilingual ASR team")]
[ArchitectureFromPaper("https://arxiv.org/abs/2006.11477",
    "Omnilingual ASR extends wav2vec 2.0 to 0.3B-7B parameters (Omnilingual ASR team 2025, section 4.1) and builds its CTC models on those encoders, so it runs on the wav2vec 2.0 network.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, WarmupSteps = 32000, MinLearningRate = 0,
                DecayRate = 1.0, Schedule = LearningRateSchedulerType.Polynomial,
                Phase = TrainingPhase.PreTraining,
                Source = "Omnilingual ASR team 2025, Sec. 4.1: all pre-trained models were optimized with Adam at a "
                        + "learning rate of 1e-4, warmed up for the first 32K steps followed by polynomial decay to zero "
                        + "for the remainder of training, one million updates in total. The paper states no separate "
                        + "CTC fine-tuning recipe, so the pre-training one is the declared recipe.")]
public partial class OmnilangualASR<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    private readonly OmnilangualASROptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode;
    private bool _disposed;
    // The paper network on the native path; null in ONNX mode or with custom layers.
    private Audio.Foundations.Wav2Vec2CtcStack<T>? _stack;
    private readonly Random _random;
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => false;
    public bool SupportsWordTimestamps => false;

    public OmnilangualASR(NeuralNetworkArchitecture<T> architecture, string modelPath, OmnilangualASROptions? options = null) : base(architecture) { _options = options ?? new OmnilangualASROptions(); Options = _options; _useNativeMode = false; base.SampleRate = _options.SampleRate; _random = Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1); base.NumMels = 0; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions); SupportedLanguages = new[] { "en", "zh", "de", "es", "fr", "ja", "ko", "pt", "ru", "ar", "hi", "sw" }; InitializeLayers(); }
    public OmnilangualASR(NeuralNetworkArchitecture<T> architecture, OmnilangualASROptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new OmnilangualASROptions(); Options = _options; _useNativeMode = true; _optimizer = optimizer
        ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
        ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; _random = Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1); base.NumMels = 0; SupportedLanguages = new[] { "en", "zh", "de", "es", "fr", "ja", "ko", "pt", "ru", "ar", "hi", "sw" }; InitializeLayers(); }

    /// <summary>
    /// Transcribes audio: the encoder and CTC layer give per-frame distributions, decoded greedily.
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
            Segments = ResolveReturnTimestamps(includeTimestamps) ? ExtractSegments(text, duration, confidence) : Array.Empty<TranscriptionSegment<T>>()
        };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    public string DetectLanguage(Tensor<T> audio) { var logits = Logits(PreprocessAudio(audio), training: false); var (tokens, _) = CTCGreedyDecodeWithConfidence(logits); return ClassifyLanguageFromTokens(tokens); }
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) { var detected = DetectLanguage(audio); var result = new Dictionary<string, T>(); double primaryProb = 0.85; double otherProb = SupportedLanguages.Count > 1 ? (1.0 - primaryProb) / (SupportedLanguages.Count - 1) : 0.0; foreach (var lang in SupportedLanguages) result[lang] = NumOps.FromDouble(lang == detected ? primaryProb : otherProb); return result; }
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("OmnilangualASR does not support streaming.");

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        _stack = null;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); return; }
        // The paper network: a wav2vec 2.0 encoder (layer-norm feature encoder over the raw waveform, stable pre-norm Transformer) with a single linear CTC layer (Omnilingual ASR team 2025, section 4.2). It used to be a generic encoder over mel features (#2321).
        _stack = new Audio.Foundations.Wav2Vec2CtcStack<T>(Engine, _options.ToConfiguration(), _options.VocabSize, _options.Seed ?? 42);
        Layers.AddRange(_stack.Layers);
    }

    /// <summary>Loads a transformers <c>Wav2Vec2ForCTC</c> state dict (<c>wav2vec2.*</c> and <c>lm_head.*</c>).</summary>
    /// <param name="read">Returns the named tensor's values, row-major, for the shape given.</param>
    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        if (_stack is null) throw new InvalidOperationException("Weights load into the native network.");
        _stack.LoadTorchWeights(read, "wav2vec2.");
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
            return _stack.Run(input, training, _random, freezeFeatureEncoder: training && _options.FreezeFeatureEncoder, _options.FinalDropout);
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
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "OmnilangualASR-Native" : "OmnilangualASR-ONNX", Description = "Omnilingual ASR: universal multilingual recognition (Meta, 2024)", FeatureCount = 1, Complexity = _options.NumEncoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };



    private (List<int> tokens, double confidence) CTCGreedyDecodeWithConfidence(Tensor<T> logits) { var tokens = new List<int>(); double totalConf = 0; int confCount = 0; int prevToken = -1; int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1; int vocabSize = logits.Rank >= 2 ? logits.Shape[^1] : logits.Shape[0]; for (int t = 0; t < numFrames && tokens.Count < _options.MaxTextLength; t++) { int maxIdx = 0; double maxVal = double.NegativeInfinity; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); if (val > maxVal) { maxVal = val; maxIdx = v; } } double sumExp = 0; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); sumExp += Math.Exp(val - maxVal); } double frameConf = 1.0 / sumExp; if (maxIdx != prevToken && maxIdx > 0) { tokens.Add(maxIdx); totalConf += frameConf; confCount++; } prevToken = maxIdx; } return (tokens, confCount > 0 ? totalConf / confCount : 0.0); }
    private static string TokensToText(List<int> tokens) { var sb = new System.Text.StringBuilder(); foreach (var t in tokens) { if (t > 0 && t <= char.MaxValue) sb.Append((char)t); else if (t > char.MaxValue && t <= 0x10FFFF) sb.Append(char.ConvertFromUtf32(t)); } return sb.ToString().Trim(); }
    private IReadOnlyList<TranscriptionSegment<T>> ExtractSegments(string text, double duration, double confidence) { if (string.IsNullOrWhiteSpace(text)) return Array.Empty<TranscriptionSegment<T>>(); return new[] { new TranscriptionSegment<T> { Text = text, StartTime = 0.0, EndTime = duration, Confidence = NumOps.FromDouble(confidence) } }; }
    private string ClassifyLanguageFromTokens(List<int> tokens) { if (tokens.Count == 0) return _options.Language; int cjkCount = 0, latinCount = 0; foreach (var t in tokens) { if (t >= 0x4E00 && t <= 0x9FFF) cjkCount++; else if (t >= 0x41 && t <= 0x7A) latinCount++; } if (cjkCount > latinCount && SupportedLanguages.Contains("zh")) return "zh"; return _options.Language; }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(OmnilangualASR<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
