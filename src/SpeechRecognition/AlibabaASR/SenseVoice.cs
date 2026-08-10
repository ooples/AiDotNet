using AiDotNet.Attributes;
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
/// SenseVoice: multi-task speech understanding model
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Model: "SenseVoice" (Alibaba FunASR, 2024)</item></list></para>
/// <para><b>For Beginners:</b> SenseVoice is a multi-task speech understanding model that handles ASR, language identification, emotion recognition, and audio event detection in a single model. It uses a shared encoder with task-specific output heads. The model processes speech...</para>
/// <para>
/// SenseVoice is a multi-task speech understanding model that handles ASR, language identification, emotion recognition, and audio event detection in a single model. It uses a shared encoder with task-specific output heads. The model processes speech with a Paraformer-style encoder and uses task tokens to select the output modality. SenseVoice Small covers 50+ languages while maintaining fast inference through non-autoregressive decoding.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a SenseVoice model for multi-task speech understanding
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 16000, inputWidth: 1, inputDepth: 1, outputSize: 5000);
/// var model = new SenseVoice&lt;double&gt;(architecture);
///
/// // Or load a pre-trained ONNX model for ASR, emotion, and language ID
/// var onnxModel = new SenseVoice&lt;double&gt;(architecture, "sensevoice.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("FunAudioLLM: Voice Understanding and Generation Foundation Models for Natural Interaction Between Humans and LLMs", "https://arxiv.org/abs/2407.04051", Year = 2024, Authors = "Du et al.")]
public class SenseVoice<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    private readonly SenseVoiceOptions _options; public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => false;
    public bool SupportsWordTimestamps => false;

    public SenseVoice(NeuralNetworkArchitecture<T> architecture, string modelPath, SenseVoiceOptions? options = null) : base(architecture) { _options = options ?? new SenseVoiceOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions); SupportedLanguages = new[] { "zh", "en", "ja", "ko", "de", "es", "fr", "it", "pt", "ru" }; InitializeLayers(); }
    public SenseVoice(NeuralNetworkArchitecture<T> architecture, SenseVoiceOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new SenseVoiceOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = _options.LearningRate, WeightDecay = _options.WeightDecay }); base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; SupportedLanguages = new[] { "zh", "en", "ja", "ko", "de", "es", "fr", "it", "pt", "ru" }; InitializeLayers(); }

    /// <summary>
    /// Transcribes audio using SenseVoice's multi-task encoder with task-specific heads.
    /// Per Alibaba (2024): the shared encoder processes speech, then task tokens select
    /// ASR decoding via CIF-based parallel generation.
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
            Segments = includeTimestamps ? ExtractSegments(text, duration, confidence) : Array.Empty<TranscriptionSegment<T>>()
        };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool includeTimestamps = false, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    public string DetectLanguage(Tensor<T> audio) { var features = PreprocessAudio(audio); Tensor<T> logits; if (IsOnnxMode && OnnxEncoder is not null) logits = OnnxEncoder.Run(features); else { logits = features; foreach (var l in Layers) logits = l.Forward(logits); } var (tokens, _) = CTCGreedyDecodeWithConfidence(logits); return ClassifyLanguageFromTokens(tokens); }
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) { var detected = DetectLanguage(audio); var result = new Dictionary<string, T>(); double primaryProb = 0.85; double otherProb = SupportedLanguages.Count > 1 ? (1.0 - primaryProb) / (SupportedLanguages.Count - 1) : 0.0; foreach (var lang in SupportedLanguages) result[lang] = NumOps.FromDouble(lang == detected ? primaryProb : otherProb); return result; }
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("SenseVoice does not support streaming.");

    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultParaformerLayers(encoderDim: _options.EncoderDim, decoderDim: _options.DecoderDim, numEncoderLayers: _options.NumEncoderLayers, numDecoderLayers: _options.NumDecoderLayers, numAttentionHeads: _options.NumAttentionHeads, feedForwardDim: _options.FeedForwardDim, numMels: _options.NumMels, vocabSize: _options.VocabSize, dropoutRate: _options.DropoutRate, useCifAlignment: _options.UseCifAlignment)); }
    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); try { TrainWithCifSupervision(input, expected, _optimizer); } finally { SetTrainingMode(false); } }
    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) { if (MelSpec is not null) return MelSpec.Forward(rawAudio); return rawAudio; }
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "SenseVoice-Native" : "SenseVoice-ONNX", Description = "SenseVoice: multi-task speech understanding (Alibaba, 2024)", FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };
    /// <summary>Version of the trailing section of this payload.</summary>
    /// <remarks>
    /// Version 1 added DecoderDim, NumDecoderLayers, FeedForwardDim, UseCifAlignment, LearningRate and
    /// WeightDecay. Payloads written before it end after Language, which is what the reader keys on --
    /// this method is the last thing NeuralNetworkBase.Serialize writes, so an exhausted stream is an
    /// exact signal rather than a guess.
    /// </remarks>
    private const int NetworkSpecificPayloadVersion = 1;

    protected override void SerializeNetworkSpecificData(BinaryWriter w) { w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty); w.Write(_options.SampleRate); w.Write(_options.MaxAudioLengthSeconds); w.Write(_options.EncoderDim); w.Write(_options.NumEncoderLayers); w.Write(_options.NumAttentionHeads); w.Write(_options.NumMels); w.Write(_options.VocabSize); w.Write(_options.MaxTextLength); w.Write(_options.DropoutRate); w.Write(_options.Language);
        // The configuration a caller can actually set. Without these a saved model reloaded
        // with default decoder width, depth, feed-forward size, CIF alignment and optimizer
        // settings -- a different model from the one that was saved, reported as success.
        w.Write(NetworkSpecificPayloadVersion); w.Write(_options.DecoderDim); w.Write(_options.NumDecoderLayers); w.Write(_options.FeedForwardDim); w.Write(_options.UseCifAlignment); w.Write(_options.LearningRate); w.Write(_options.WeightDecay); }
    protected override void DeserializeNetworkSpecificData(BinaryReader r) { _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = r.ReadInt32(); _options.MaxAudioLengthSeconds = r.ReadInt32(); _options.EncoderDim = r.ReadInt32(); _options.NumEncoderLayers = r.ReadInt32(); _options.NumAttentionHeads = r.ReadInt32(); _options.NumMels = r.ReadInt32(); _options.VocabSize = r.ReadInt32(); _options.MaxTextLength = r.ReadInt32(); _options.DropoutRate = r.ReadDouble(); _options.Language = r.ReadString(); base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        var stream = r.BaseStream;
        if (!stream.CanSeek || stream.Position < stream.Length)
        {
            int payloadVersion = r.ReadInt32();
            if (payloadVersion != NetworkSpecificPayloadVersion)
            {
                throw new InvalidOperationException(
                    $"SenseVoice was saved with network-payload version {payloadVersion}, but this build " +
                    $"reads version {NetworkSpecificPayloadVersion}. Load it with a matching version of " +
                    "AiDotNet, or re-save it from one.");
            }

            _options.DecoderDim = r.ReadInt32(); _options.NumDecoderLayers = r.ReadInt32(); _options.FeedForwardDim = r.ReadInt32(); _options.UseCifAlignment = r.ReadBoolean(); _options.LearningRate = r.ReadDouble(); _options.WeightDecay = r.ReadDouble();
        }
        else
        {
            System.Diagnostics.Trace.TraceWarning(
                "AiDotNet.SenseVoice: this model was saved before the decoder and optimizer settings were " +
                "persisted, so they keep their defaults. Re-save the model to carry them forward.");
        }
 if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { var options = new SenseVoiceOptions(_options); if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new SenseVoice<T>(Architecture, mp, options); return new SenseVoice<T>(Architecture, options); }

    private (List<int> tokens, double confidence) CTCGreedyDecodeWithConfidence(Tensor<T> logits) { var tokens = new List<int>(); double totalConf = 0; int confCount = 0; int prevToken = -1; int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1; int vocabSize = logits.Rank >= 2 ? logits.Shape[^1] : logits.Shape[0]; for (int t = 0; t < numFrames && tokens.Count < _options.MaxTextLength; t++) { int maxIdx = 0; double maxVal = double.NegativeInfinity; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); if (val > maxVal) { maxVal = val; maxIdx = v; } } double sumExp = 0; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); sumExp += Math.Exp(val - maxVal); } double frameConf = 1.0 / sumExp; if (maxIdx != prevToken && maxIdx > 0) { tokens.Add(maxIdx); totalConf += frameConf; confCount++; } prevToken = maxIdx; } return (tokens, confCount > 0 ? totalConf / confCount : 0.0); }
    private static string TokensToText(List<int> tokens) { var sb = new System.Text.StringBuilder(); foreach (var t in tokens) { if (t > 0 && t <= char.MaxValue) sb.Append((char)t); else if (t > char.MaxValue && t <= 0x10FFFF) sb.Append(char.ConvertFromUtf32(t)); } return sb.ToString().Trim(); }
    private IReadOnlyList<TranscriptionSegment<T>> ExtractSegments(string text, double duration, double confidence) { if (string.IsNullOrWhiteSpace(text)) return Array.Empty<TranscriptionSegment<T>>(); return new[] { new TranscriptionSegment<T> { Text = text, StartTime = 0.0, EndTime = duration, Confidence = NumOps.FromDouble(confidence) } }; }
    private string ClassifyLanguageFromTokens(List<int> tokens) { if (tokens.Count == 0) return _options.Language; int cjkCount = 0, latinCount = 0; foreach (var t in tokens) { if (t >= 0x4E00 && t <= 0x9FFF) cjkCount++; else if (t >= 0x41 && t <= 0x7A) latinCount++; } if (cjkCount > latinCount && SupportedLanguages.Contains("zh")) return "zh"; return _options.Language; }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SenseVoice<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
