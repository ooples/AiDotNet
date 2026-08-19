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
/// Convolution-augmented Transformer for ASR (pre-Conformer architecture).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Convolution-Augmented Transformer for Speech Recognition" (2019)</item></list></para>
/// <para><b>For Beginners:</b> A precursor to the Conformer that adds convolution blocks before or after each Transformer layer. Unlike Conformer's interleaved design, ConvTransformer uses convolution as a separate preprocessing stage, which provides local feature enhancement b...</para>
/// <para>
/// A precursor to the Conformer that adds convolution blocks before or after each
/// Transformer layer. Unlike Conformer's interleaved design, ConvTransformer uses
/// convolution as a separate preprocessing stage, which provides local feature enhancement
/// before the global self-attention. This architecture demonstrated the importance of
/// combining convolution with attention for speech.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a ConvTransformer model (convolution-augmented Transformer for ASR)
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 16000, inputWidth: 1, inputDepth: 1, outputSize: 5000);
/// var model = new ConvTransformer&lt;double&gt;(architecture);
///
/// // Or load a pre-trained ONNX model for conv-augmented ASR inference
/// var onnxModel = new ConvTransformer&lt;double&gt;(architecture, "convtransformer.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Conformer: Convolution-augmented Transformer for End-to-End Speech Recognition", "https://arxiv.org/abs/2005.08100", Year = 2020, Authors = "Gulati et al.")]
public partial class ConvTransformer<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    /// <inheritdoc />
    /// <remarks>
    /// Measured from this model's own output head. <c>InitializeLayers</c> builds
    /// <c>LayerHelper&lt;T&gt;.CreateDefaultCTCDecoderLayers(..., vocabSize: _options.VocabSize, ...)</c>,
    /// whose LAST emitted layer is the CTC head <c>new DenseLayer&lt;T&gt;(vocabSize, identity)</c> — the
    /// per-block <c>DenseLayer&lt;T&gt;(encoderDim)</c> that precedes it is not the final axis.
    /// <c>PredictCore</c> delegates to the base fold and <c>PostprocessOutput</c> is the identity.
    /// </remarks>
    protected override int OutputFeatureWidth => _options.VocabSize;

    private readonly ConvTransformerOptions _options; public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => false;
    public bool SupportsWordTimestamps => false;

    public ConvTransformer(NeuralNetworkArchitecture<T> architecture, string modelPath, ConvTransformerOptions? options = null) : base(architecture) { _options = options ?? new ConvTransformerOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions); SupportedLanguages = new[] { _options.Language }; InitializeLayers(); }
    public ConvTransformer(NeuralNetworkArchitecture<T> architecture, ConvTransformerOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new ConvTransformerOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; SupportedLanguages = new[] { _options.Language }; InitializeLayers(); }

    /// <summary>
    /// Transcribes audio using convolution-augmented Transformer encoder.
    /// Convolution blocks preprocess acoustic features before self-attention layers,
    /// providing local feature enhancement before global context modeling.
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

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool includeTimestamps = false, CancellationToken cancellationToken = default) => Task.Run(() => { cancellationToken.ThrowIfCancellationRequested(); return Transcribe(audio, language, includeTimestamps); }, cancellationToken);
    public string DetectLanguage(Tensor<T> audio)
    {
        ThrowIfDisposed();
        // ConvTransformer is monolingual; return the configured language.
        return _options.Language;
    }
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio)
    {
        ThrowIfDisposed();
        // ConvTransformer is monolingual; return full confidence for the configured language.
        var result = new Dictionary<string, T>();
        foreach (var lang in SupportedLanguages)
            result[lang] = NumOps.FromDouble(lang == _options.Language ? 1.0 : 0.0);
        return result;
    }
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("ConvTransformer does not support streaming.");

    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultCTCDecoderLayers(encoderDim: _options.EncoderDim, numLayers: _options.NumEncoderLayers, numAttentionHeads: _options.NumAttentionHeads, feedForwardDim: _options.FeedForwardDim, numMels: _options.NumMels, vocabSize: _options.VocabSize, dropoutRate: _options.DropoutRate)); }
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null)
            return OnnxEncoder.Run(input);
        return base.PredictCore(input);
    }
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        ThrowIfDisposed();
        if (IsOnnxMode)
            throw new NotSupportedException("Training not supported in ONNX mode.");
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
    /// <summary>
    /// Refuses parameter work on a disposed model, on every entry point rather than one.
    /// </summary>
    /// <remarks>
    /// This check used to live inside UpdateParameters, which meant ParameterCount, GetParameters
    /// and SetParameters reached a disposed model unguarded. The base calls this hook from all of
    /// them, so moving it here widens the guard and lets the hand-written UpdateParameters -- whose
    /// only other content was a walk the base already performs -- be deleted.
    /// </remarks>
    protected override void EnsureParametersReady()
    {
        ThrowIfDisposed();
        base.EnsureParametersReady();
    }

    // UpdateParameters folded one enumeration the base already folds. Removed under AIDN082.
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "ConvTransformer-Native" : "ConvTransformer-ONNX", Description = "Convolution-Augmented Transformer for ASR (2019)", FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };


    private (List<int> tokens, double confidence) CTCGreedyDecodeWithConfidence(Tensor<T> logits) { var tokens = new List<int>(); double totalConf = 0; int confCount = 0; int prevToken = -1; int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1; int vocabSize = logits.Rank >= 2 ? logits.Shape[^1] : logits.Shape[0]; for (int t = 0; t < numFrames; t++) { int maxIdx = 0; double maxVal = double.NegativeInfinity; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); if (val > maxVal) { maxVal = val; maxIdx = v; } } double sumExp = 0; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); sumExp += Math.Exp(val - maxVal); } double frameConf = 1.0 / sumExp; if (maxIdx != prevToken && maxIdx > 0) { tokens.Add(maxIdx); totalConf += frameConf; confCount++; } prevToken = maxIdx; } return (tokens, confCount > 0 ? totalConf / confCount : 0.0); }
    private static string TokensToText(List<int> tokens) { var sb = new System.Text.StringBuilder(); foreach (var t in tokens) { if (t > 0 && t <= char.MaxValue) sb.Append((char)t); else if (t > char.MaxValue && t <= 0x10FFFF) sb.Append(char.ConvertFromUtf32(t)); } return sb.ToString().Trim(); }
    private IReadOnlyList<TranscriptionSegment<T>> ExtractSegments(string text, double duration, double confidence) { if (string.IsNullOrWhiteSpace(text)) return Array.Empty<TranscriptionSegment<T>>(); return new[] { new TranscriptionSegment<T> { Text = text, StartTime = 0.0, EndTime = duration, Confidence = NumOps.FromDouble(confidence) } }; }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(ConvTransformer<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
