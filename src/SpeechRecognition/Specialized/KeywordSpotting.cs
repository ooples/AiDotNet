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
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
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
    /// <inheritdoc />
    protected override int OutputFeatureWidth => Architecture.OutputSize > 0
        ? Architecture.OutputSize
        : (_options.Vocabulary.Length > 0 ? _options.Vocabulary.Length : _options.VocabSize);

    private readonly KeywordSpottingOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    /// <inheritdoc />
    public IReadOnlyList<string> SupportedLanguages { get; }
    /// <inheritdoc />
    public bool SupportsStreaming => true;
    /// <inheritdoc />
    public bool SupportsWordTimestamps => false;

    public KeywordSpotting(NeuralNetworkArchitecture<T> architecture, string modelPath, KeywordSpottingOptions? options = null) : base(architecture) { _options = options ?? new KeywordSpottingOptions(); Options = _options; _useNativeMode = false; base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions); SupportedLanguages = new[] { "en" }; InitializeLayers(); }
    public KeywordSpotting(NeuralNetworkArchitecture<T> architecture, KeywordSpottingOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new KeywordSpottingOptions(); Options = _options; _useNativeMode = true; _optimizer = optimizer ?? CreateExponentialSgdOptimizer(_options.LearningRate, _options.LearningRateDecay); base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; SupportedLanguages = new[] { "en" }; InitializeLayers(); }

    /// <summary>
    /// Detects keywords using the paper's compact feed-forward Deep KWS network.
    /// The model processes stacked acoustic frames and outputs softmax posterior scores
    /// for the configured keyword labels.
    /// </summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
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
            Segments = ResolveReturnTimestamps(includeTimestamps) ? ExtractSegment(text, duration, confidence) : Array.Empty<TranscriptionSegment<T>>()
        };
    }

    /// <inheritdoc />
    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    /// <inheritdoc />
    public string DetectLanguage(Tensor<T> audio) { ThrowIfDisposed(); if (audio is null) throw new ArgumentNullException(nameof(audio)); return _options.Language; }
    /// <inheritdoc />
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) { var detected = DetectLanguage(audio); var result = new Dictionary<string, T>(); double primaryProb = 0.85; double otherProb = SupportedLanguages.Count > 1 ? (1.0 - primaryProb) / (SupportedLanguages.Count - 1) : 0.0; T primaryValue = NumOps.FromDouble(primaryProb); T otherValue = NumOps.FromDouble(otherProb); foreach (var lang in SupportedLanguages) result[lang] = lang == detected ? primaryValue : otherValue; return result; }
    /// <inheritdoc />
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => new KeywordSpottingStreamingSession(this, language ?? _options.Language);

    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultKeywordSpottingLayers(Architecture, hiddenLayerCount: _options.NumEncoderLayers, hiddenLayerSize: _options.EncoderDim, outputLabelCount: OutputFeatureWidth)); }
    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "KeywordSpotting-Native" : "KeywordSpotting-ONNX", Description = "Small-footprint Deep KWS (Chen, Parada, and Heigold, 2014)", FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };

    private (List<int> tokens, double confidence) DecodeKeywordPosteriors(Tensor<T> logits)
    {
        if (logits.Length == 0)
            return (new List<int>(), 0.0);

        int labelCount = logits.Rank > 0 ? logits.Shape[^1] : logits.Length;
        if (labelCount <= 0 || logits.Length % labelCount != 0)
            throw new ArgumentException("Keyword posterior tensor must have a positive trailing label axis.", nameof(logits));

        int frameCount = logits.Length / labelCount;
        var engine = AiDotNetEngine.Current;
        var shape = new[] { frameCount, labelCount };

        // Decoding only reads the posteriors; nothing here belongs on a gradient tape.
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            // Frame f, label l is flat element f * labelCount + l.
            var values = logits.Reshape(frameCount, labelCount);
            var zeros = engine.TensorZerosLike(values);
            var ones = engine.TensorOnesLike(values);

            // A frame that is already a distribution (every value in [0, 1], summing to 1 within 1e-4) is kept
            // as is; any other frame is softmax-normalized. NaN is outside [0, 1].
            var outsideUnit = engine.TensorAdd(
                engine.TensorAdd(engine.TensorLessThan(values, NumOps.Zero), engine.TensorGreaterThan(values, NumOps.One)),
                engine.TensorWhere(engine.TensorIsNan(values), ones, zeros));
            var bounded = engine.TensorEquals(engine.ReduceMax(outsideUnit, new[] { 1 }, true), NumOps.Zero);
            var rawSum = engine.ReduceSum(values, new[] { 1 }, true);
            var sumOff = engine.TensorGreaterThan(
                engine.TensorAbs(engine.TensorSubtractScalar(rawSum, NumOps.One)), NumOps.FromDouble(1e-4));
            var alreadyProbabilities = engine.TensorBroadcastTo(
                engine.TensorMultiply(bounded, engine.ScalarMinusTensor(NumOps.One, sumOff)), shape);

            var maximum = engine.TensorBroadcastTo(engine.ReduceMax(values, new[] { 1 }, true), shape);
            var exponentials = engine.TensorExp(engine.TensorSubtract(values, maximum));
            var softmax = engine.TensorDivide(
                exponentials, engine.TensorBroadcastTo(engine.ReduceSum(exponentials, new[] { 1 }, true), shape));
            var probabilities = engine.TensorWhere(alreadyProbabilities, values, softmax);

            // Chen et al. Eq. (2): smooth each frame posterior over the latest 30 frames, as a difference of
            // running totals divided by the number of frames actually in the window.
            const int SmoothingWindow = 30;
            var runningTotal = engine.TensorCumSum(probabilities, 0);
            var windowTotal = frameCount > SmoothingWindow
                ? engine.TensorSubtract(
                    runningTotal,
                    engine.TensorConcatenate(
                        new[]
                        {
                            new Tensor<T>(new[] { SmoothingWindow, labelCount }),
                            engine.TensorNarrow(runningTotal, 0, 0, frameCount - SmoothingWindow),
                        },
                        0))
                : runningTotal;
            var windowLength = engine.TensorMin(engine.TensorArange<T>(1, frameCount + 1), NumOps.FromDouble(SmoothingWindow));
            var smoothed = engine.TensorDivide(
                windowTotal, engine.TensorBroadcastTo(windowLength.Reshape(frameCount, 1), shape));

            // Each frame's decision is its first maximal smoothed posterior, scanning from label 0; a NaN at
            // label 0 is never beaten, and NaN elsewhere never wins.
            var firstMax = CtcGreedyDecoding.FirstMaxIndexPerRow(smoothed);
            var bestLabel = engine.TensorWhere(
                engine.TensorIsNan(engine.TensorNarrow(smoothed, 1, 0, 1).Reshape(frameCount)),
                new Tensor<int>(new[] { frameCount }),
                firstMax);

            // Label zero is the paper's non-keyword/filler label. Collapse repeated frame
            // decisions only for presentation; confidence below still uses every frame.
            var keywordFrames = CtcGreedyDecoding.EmittedFrames(bestLabel, _options.MaxTextLength);
            var tokens = keywordFrames.Length == 0
                ? new List<int>()
                : new List<int>(engine.TensorIndexSelect(
                    bestLabel, new Tensor<int>(keywordFrames, new[] { keywordFrames.Length }), 0).ToArray());

            // Chen et al. Eq. (3): geometric mean of each keyword label's maximum
            // smoothed posterior over the latest 100-frame confidence window.
            if (labelCount <= 1)
                return (tokens, 0.0);

            const int ConfidenceWindow = 100;
            int firstConfidenceFrame = Math.Max(0, frameCount - ConfidenceWindow);
            var window = engine.TensorNarrow(
                engine.TensorNarrow(smoothed, 0, firstConfidenceFrame, frameCount - firstConfidenceFrame),
                1, 1, labelCount - 1);

            // peak = max(0, window values), and NaN as soon as the window holds one (as Math.Max does). The
            // smoothed posteriors are finite or NaN, so window * 0 sums to 0 or NaN and carries that NaN
            // past the floor.
            var isNaN = engine.TensorIsNan(window);
            var cleanWindow = engine.TensorWhere(isNaN, engine.TensorZerosLike(window), window);
            var nanCarrier = engine.ReduceSum(engine.TensorMultiplyScalar(window, NumOps.Zero), new[] { 0 }, false);
            var peak = engine.TensorMax(engine.ReduceMax(cleanWindow, new[] { 0 }, false), NumOps.Zero);
            var logPeak = engine.TensorAdd(
                engine.TensorLog(engine.TensorMax(peak, NumOps.FromDouble(1e-300))), nanCarrier);
            double sumLogPeak = NumOps.ToDouble(engine.TensorSum(logPeak));

            double confidence = Math.Exp(sumLogPeak / (labelCount - 1));
            return (tokens, confidence);
        }
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
    /// <inheritdoc />
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    private sealed class KeywordSpottingStreamingSession : IStreamingTranscriptionSession<T>
    {
        private readonly KeywordSpotting<T> _model; private readonly string _language; private readonly List<Tensor<T>> _chunks = new(); private bool _disposed;
        public KeywordSpottingStreamingSession(KeywordSpotting<T> model, string language) { _model = model; _language = language; }
        public void FeedAudio(Tensor<T> audioChunk) { if (_disposed) throw new ObjectDisposedException(nameof(KeywordSpottingStreamingSession)); _chunks.Add(audioChunk); }
        public TranscriptionResult<T> GetPartialResult() { if (_disposed) throw new ObjectDisposedException(nameof(KeywordSpottingStreamingSession)); if (_chunks.Count == 0) return new TranscriptionResult<T> { Language = _language }; Tensor<T> combined; using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>()) { combined = AiDotNetEngine.Current.TensorConcatenate(_chunks.Select(c => c.Reshape(c.Length)).ToArray(), 0); } return _model.Transcribe(combined, _language); }
        public TranscriptionResult<T> Finalize() { if (_disposed) throw new ObjectDisposedException(nameof(KeywordSpottingStreamingSession)); var result = GetPartialResult(); _disposed = true; return result; }
        public void Dispose() { _disposed = true; }
    }
}
