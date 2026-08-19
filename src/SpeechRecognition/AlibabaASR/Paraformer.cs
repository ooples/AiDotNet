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
/// <example>
/// <code>
/// // Create a Paraformer model for non-autoregressive parallel ASR
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 16000, inputWidth: 1, inputDepth: 1, outputSize: 5000);
/// var model = new Paraformer&lt;double&gt;(architecture);
///
/// // Or load a pre-trained ONNX model for CIF-based parallel decoding
/// var onnxModel = new Paraformer&lt;double&gt;(architecture, "paraformer.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition", "https://arxiv.org/abs/2206.08317", Year = 2022, Authors = "Gao et al.")]
public partial class Paraformer<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    private readonly ParaformerOptions _options; public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => false;
    public bool SupportsWordTimestamps => false;

    public Paraformer(NeuralNetworkArchitecture<T> architecture, string modelPath, ParaformerOptions? options = null) : base(architecture) { _options = options ?? new ParaformerOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions); SupportedLanguages = new[] { "en" }; InitializeLayers(); }
    public Paraformer(NeuralNetworkArchitecture<T> architecture, ParaformerOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new ParaformerOptions();
        if (_options.LearningRate <= 0.0 || double.IsNaN(_options.LearningRate) || double.IsInfinity(_options.LearningRate))
            throw new ArgumentOutOfRangeException(nameof(options), "LearningRate must be finite and greater than zero.");
        if (_options.WeightDecay < 0.0 || double.IsNaN(_options.WeightDecay) || double.IsInfinity(_options.WeightDecay))
            throw new ArgumentOutOfRangeException(nameof(options), "WeightDecay must be finite and non-negative.");

        _useNativeMode = true;
        _optimizerIsDefault = optimizer is null;
        _optimizer = optimizer ?? CreateDefaultOptimizer();
        base.SampleRate = _options.SampleRate;
        base.NumMels = _options.NumMels;
        SupportedLanguages = new[] { "en" };
        InitializeLayers();
    }

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

    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultParaformerLayers(encoderDim: _options.EncoderDim, decoderDim: _options.DecoderDim, numEncoderLayers: _options.NumEncoderLayers, numDecoderLayers: _options.NumDecoderLayers, numAttentionHeads: _options.NumAttentionHeads, feedForwardDim: _options.FeedForwardDim, numMels: _options.NumMels, vocabSize: _options.VocabSize, dropoutRate: _options.DropoutRate)); }
    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            TrainWithCifSupervision(input, expected, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }
    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "Paraformer-Native" : "Paraformer-ONNX", Description = "Paraformer: CIF + parallel Transformer (Alibaba DAMO, 2022)", FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };
    /// <summary>Whether <c>_optimizer</c> is the one this class built rather than the caller's.</summary>
    private bool _optimizerIsDefault;

    /// <summary>
    /// Builds the default optimizer from the CURRENT options.
    /// </summary>
    /// <remarks>
    /// Must run again after deserialization. LearningRate and WeightDecay are baked into the optimizer
    /// at construction, so restoring a checkpoint that saved different values left the optimizer on the
    /// constructor's -- the model then trained at settings the checkpoint did not choose, with the
    /// options object reading correctly the whole time.
    /// </remarks>
    private AdamWOptimizer<T, Tensor<T>, Tensor<T>> CreateDefaultOptimizer()
        => new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                WeightDecay = _options.WeightDecay
            });


    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>Every field appended after the original layout is read only if bytes remain.</b>
    /// <c>LearningRate</c> and <c>WeightDecay</c> were appended alongside the three decoder fields but
    /// read unconditionally, so a checkpoint written before they existed -- which ends after
    /// <c>Language</c> -- threw <see cref="EndOfStreamException"/> on the first <c>ReadDouble</c> and
    /// could not be loaded at all. Anything absent keeps the option's default, which is the value the
    /// build that wrote the payload was using.
    /// </para>
    /// <para>
    /// The remaining-bytes test is sound HERE specifically because this payload owns its stream end:
    /// <c>NeuralNetworkBase.Serialize</c> writes it last, and the clone path hands it a dedicated
    /// <see cref="MemoryStream"/>. It would NOT be sound for a field inserted mid-stream, where there
    /// is no shortage of bytes -- only a misalignment -- and a version marker is required instead.
    /// </para>
    /// </remarks>


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
