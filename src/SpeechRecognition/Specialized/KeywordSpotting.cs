using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.SpeechRecognition.Specialized;

/// <summary>
/// Keyword Spotting: lightweight wake-word and command detection
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Small-footprint Keyword Spotting Using Deep Neural Networks" (2014)</item></list></para>
/// <para><b>For Beginners:</b> Keyword Spotting provides lightweight, always-on detection of specific keywords and voice commands using a small feed-forward neural network.</para>
/// <para>
/// The cited Deep KWS model stacks log-filterbank frames, passes them through fully connected
/// ReLU layers, and produces frame-level keyword-label posteriors with a softmax output layer.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Keyword Spotting model for wake-word detection
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 16000, inputWidth: 1, inputDepth: 1, outputSize: 32);
/// var model = new KeywordSpotting&lt;double&gt;(architecture);
///
/// // Or load a pre-trained ONNX model for edge-device keyword detection
/// var onnxModel = new KeywordSpotting&lt;double&gt;(architecture, "keywordspotting.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Small-footprint Keyword Spotting Using Deep Neural Networks", "https://doi.org/10.1109/ICASSP.2014.6854370")]
public partial class KeywordSpotting<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    protected override int OutputFeatureWidth => Architecture.OutputSize > 0
        ? Architecture.OutputSize
        : (_options.Vocabulary.Length > 0 ? _options.Vocabulary.Length : _options.VocabSize);

    private readonly KeywordSpottingOptions _options; public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => true;
    public bool SupportsWordTimestamps => false;

    public KeywordSpotting(NeuralNetworkArchitecture<T> architecture, string modelPath, KeywordSpottingOptions? options = null) : base(architecture) { _options = options ?? new KeywordSpottingOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions); SupportedLanguages = new[] { "en" }; InitializeLayers(); }
    public KeywordSpotting(NeuralNetworkArchitecture<T> architecture, KeywordSpottingOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new KeywordSpottingOptions(); _useNativeMode = true; _optimizer = optimizer ?? CreateExponentialSgdOptimizer(_options.LearningRate, _options.LearningRateDecay); base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; SupportedLanguages = new[] { "en" }; InitializeLayers(); }

    /// <summary>
    /// Detects keywords using the paper's compact feed-forward Deep KWS network.
    /// The model processes stacked acoustic frames and outputs softmax posterior scores
    /// for the configured keyword labels.
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

        var (tokens, confidence) = DecodeKeywordPosteriors(logits);
        var text = LabelsToText(tokens);
        double duration = audio.Length > 0 ? (double)audio.Shape[0] / SampleRate : 0;

        return new TranscriptionResult<T>
        {
            Text = text,
            Language = language ?? _options.Language,
            Confidence = NumOps.FromDouble(confidence),
            DurationSeconds = duration,
            Segments = includeTimestamps ? ExtractSegment(text, duration, confidence) : Array.Empty<TranscriptionSegment<T>>()
        };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool includeTimestamps = false, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    public string DetectLanguage(Tensor<T> audio) { ThrowIfDisposed(); if (audio is null) throw new ArgumentNullException(nameof(audio)); return _options.Language; }
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) { var detected = DetectLanguage(audio); var result = new Dictionary<string, T>(); double primaryProb = 0.85; double otherProb = SupportedLanguages.Count > 1 ? (1.0 - primaryProb) / (SupportedLanguages.Count - 1) : 0.0; foreach (var lang in SupportedLanguages) result[lang] = NumOps.FromDouble(lang == detected ? primaryProb : otherProb); return result; }
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => new KeywordSpottingStreamingSession(this, language ?? _options.Language);

    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultKeywordSpottingLayers(Architecture, hiddenLayerCount: _options.NumEncoderLayers, hiddenLayerSize: _options.EncoderDim, outputLabelCount: OutputFeatureWidth)); }
    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); try { TrainWithTape(input, expected, _optimizer); } finally { SetTrainingMode(false); } }
    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "KeywordSpotting-Native" : "KeywordSpotting-ONNX", Description = "Small-footprint Deep KWS (Chen, Parada, and Heigold, 2014)", FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };

    private (List<int> tokens, double confidence) DecodeKeywordPosteriors(Tensor<T> logits)
    {
        if (logits.Length == 0)
            return (new List<int>(), 0.0);

        int labelCount = logits.Rank > 0 ? logits.Shape[^1] : logits.Length;
        if (labelCount <= 0 || logits.Length % labelCount != 0)
            throw new ArgumentException("Keyword posterior tensor must have a positive trailing label axis.", nameof(logits));

        int frameCount = logits.Length / labelCount;
        var probabilities = new double[frameCount, labelCount];

        for (int frame = 0; frame < frameCount; frame++)
        {
            double maximum = double.NegativeInfinity;
            double rawSum = 0.0;
            bool bounded = true;
            for (int label = 0; label < labelCount; label++)
            {
                double value = NumOps.ToDouble(logits[frame * labelCount + label]);
                probabilities[frame, label] = value;
                maximum = Math.Max(maximum, value);
                rawSum += value;
                bounded &= value >= 0.0 && value <= 1.0;
            }

            bool alreadyProbabilities = bounded && Math.Abs(rawSum - 1.0) <= 1e-4;
            if (alreadyProbabilities)
                continue;

            double exponentialSum = 0.0;
            for (int label = 0; label < labelCount; label++)
            {
                double probability = Math.Exp(probabilities[frame, label] - maximum);
                probabilities[frame, label] = probability;
                exponentialSum += probability;
            }
            for (int label = 0; label < labelCount; label++)
                probabilities[frame, label] /= exponentialSum;
        }

        // Chen et al. Eq. (2): smooth each frame posterior over the latest 30 frames.
        const int SmoothingWindow = 30;
        var smoothed = new double[frameCount, labelCount];
        for (int label = 0; label < labelCount; label++)
        {
            double runningSum = 0.0;
            for (int frame = 0; frame < frameCount; frame++)
            {
                runningSum += probabilities[frame, label];
                if (frame >= SmoothingWindow)
                    runningSum -= probabilities[frame - SmoothingWindow, label];
                smoothed[frame, label] = runningSum / Math.Min(frame + 1, SmoothingWindow);
            }
        }

        var tokens = new List<int>();
        int previousLabel = -1;
        for (int frame = 0; frame < frameCount && tokens.Count < _options.MaxTextLength; frame++)
        {
            int bestLabel = 0;
            double bestPosterior = smoothed[frame, 0];
            for (int label = 1; label < labelCount; label++)
            {
                if (smoothed[frame, label] > bestPosterior)
                {
                    bestPosterior = smoothed[frame, label];
                    bestLabel = label;
                }
            }

            // Label zero is the paper's non-keyword/filler label. Collapse repeated frame
            // decisions only for presentation; confidence below still uses every frame.
            if (bestLabel > 0 && bestLabel != previousLabel)
                tokens.Add(bestLabel);
            previousLabel = bestLabel;
        }

        // Chen et al. Eq. (3): geometric mean of each keyword label's maximum
        // smoothed posterior over the latest 100-frame confidence window.
        if (labelCount <= 1)
            return (tokens, 0.0);

        const int ConfidenceWindow = 100;
        int firstConfidenceFrame = Math.Max(0, frameCount - ConfidenceWindow);
        double sumLogPeak = 0.0;
        for (int label = 1; label < labelCount; label++)
        {
            double peak = 0.0;
            for (int frame = firstConfidenceFrame; frame < frameCount; frame++)
                peak = Math.Max(peak, smoothed[frame, label]);
            sumLogPeak += Math.Log(Math.Max(peak, 1e-300));
        }

        double confidence = Math.Exp(sumLogPeak / (labelCount - 1));
        return (tokens, confidence);
    }

    private string LabelsToText(IEnumerable<int> tokens)
    {
        return string.Join(
            " ",
            tokens.Select(label =>
                label >= 0 && label < _options.Vocabulary.Length &&
                !string.IsNullOrWhiteSpace(_options.Vocabulary[label])
                    ? _options.Vocabulary[label]
                    : label.ToString(System.Globalization.CultureInfo.InvariantCulture)));
    }

    private IReadOnlyList<TranscriptionSegment<T>> ExtractSegment(
        string text,
        double duration,
        double confidence)
    {
        if (string.IsNullOrWhiteSpace(text))
            return Array.Empty<TranscriptionSegment<T>>();

        return new[]
        {
            new TranscriptionSegment<T>
            {
                Text = text,
                StartTime = 0.0,
                EndTime = duration,
                Confidence = NumOps.FromDouble(confidence),
            },
        };
    }

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(KeywordSpotting<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    private sealed class KeywordSpottingStreamingSession : IStreamingTranscriptionSession<T>
    {
        private readonly KeywordSpotting<T> _model; private readonly string _language; private readonly List<Tensor<T>> _chunks = new(); private bool _disposed;
        public KeywordSpottingStreamingSession(KeywordSpotting<T> model, string language) { _model = model; _language = language; }
        public void FeedAudio(Tensor<T> audioChunk) { if (_disposed) throw new ObjectDisposedException(nameof(KeywordSpottingStreamingSession)); _chunks.Add(audioChunk); }
        public TranscriptionResult<T> GetPartialResult() { if (_disposed) throw new ObjectDisposedException(nameof(KeywordSpottingStreamingSession)); if (_chunks.Count == 0) return new TranscriptionResult<T> { Language = _language }; int totalLen = 0; foreach (var c in _chunks) totalLen += c.Length; var combined = new Tensor<T>(new[] { totalLen }); int offset = 0; foreach (var c in _chunks) { for (int i = 0; i < c.Length; i++) combined[offset + i] = c[i]; offset += c.Length; } return _model.Transcribe(combined, _language); }
        public TranscriptionResult<T> Finalize() { if (_disposed) throw new ObjectDisposedException(nameof(KeywordSpottingStreamingSession)); var result = GetPartialResult(); _disposed = true; return result; }
        public void Dispose() { _disposed = true; }
    }
}
