using AiDotNet.Attributes;
using AiDotNet.Audio.Features;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// PANNs (Pre-trained Audio Neural Networks) CNN14 model for audio classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PANNs (Kong et al., IEEE/ACM TASLP 2020) provides pre-trained CNN-based audio classification
/// models. The flagship CNN14 achieves 43.1% mAP on AudioSet-2M and is widely used as a feature
/// extractor for downstream audio tasks.
/// </para>
/// <para>
/// <b>Architecture:</b> CNN14 is a 14-layer convolutional neural network:
/// <list type="number">
/// <item><b>Input</b>: 64-bin log-mel spectrogram at 32 kHz</item>
/// <item><b>6 CNN blocks</b>: Each with two 3x3 conv layers, batch norm, ReLU, and 2x2 avg pooling</item>
/// <item><b>Channel progression</b>: 64 -> 128 -> 256 -> 512 -> 1024 -> 2048</item>
/// <item><b>Global pooling</b>: Average and max pooling combined</item>
/// <item><b>Classification head</b>: 2048-dim FC -> 527-class sigmoid output</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> PANNs is like having increasingly specialized detectors stacked on top
/// of each other. The first layers detect simple patterns (edges, tones), middle layers detect
/// intermediate patterns (harmonics, rhythms), and final layers detect complete sounds (speech,
/// music, dog bark). Being CNN-based, it is faster than Transformer models but slightly less accurate.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 2048, outputSize: 527);
/// var panns = new PANNs&lt;float&gt;(arch, "cnn14_audioset.onnx");
/// var result = panns.Detect(audioTensor);
/// </code>
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "PANNs: Large-Scale Pretrained Audio Neural Networks" (Kong et al., 2020)</item>
/// <item>Repository: https://github.com/qiuqiangkong/audioset_tagging_cnn</item>
/// </list>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition", "https://arxiv.org/abs/1912.10211", Year = 2020, Authors = "Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, Mark D. Plumbley")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 0.001, ReferenceBatchSize = 32,
                Source = "Kong et al. 2020, Sec. IV: batch size 32 and an Adam optimizer with a learning rate of 0.001. Built here rather than by the factory because the paper does not clip gradients and the options default to clipping; the declaration verifies the hand-built values instead of replacing them.")]
public partial class PANNs<T> : AudioClassifierBase<T>, IAudioEventDetector<T>
{
    #region Fields

    private readonly PANNsOptions _options;
    private MelSpectrogram<T>? _melSpectrogram;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    public static readonly string[] AudioSetLabels = BEATs<T>.AudioSetLabels;

    #endregion

    #region Constructors

    public PANNs(NeuralNetworkArchitecture<T> architecture, string modelPath, PANNsOptions? options = null)
        : base(architecture, new BinaryCrossEntropyWithLogitsLoss<T>())
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options = options ?? new PANNsOptions(); Options = _options; _useNativeMode = false;
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        _melSpectrogram = new MelSpectrogram<T>(_options.SampleRate, _options.NumMels, _options.FftSize, _options.HopLength, _options.FMin, _options.FMax, logMel: true);
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        ClassLabels = _options.CustomLabels ?? AudioSetLabels;
        InitializeLayers();
    }

    public PANNs(
        NeuralNetworkArchitecture<T> architecture,
        PANNsOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new BinaryCrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new PANNsOptions(); Options = _options; _useNativeMode = true;
        if (_options.LearningRate <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(options), "PANNs learning rate must be positive.");
        _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this,
            optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
                this,
                new Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    // Kong et al.'s released AudioSet trainer uses ordinary Adam at 1e-3 and does not
                    // apply a global-norm clipping pass. AdamOptimizerOptions enables clipping as a
                    // library-wide transformer safety default, so disable that extra policy here.
                    InitialLearningRate = _options.LearningRate,
                    UseAdaptiveLearningRate = false,
                    EnableGradientClipping = false,
                }));
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        ClassLabels = _options.CustomLabels ?? AudioSetLabels;
        _melSpectrogram = new MelSpectrogram<T>(_options.SampleRate, _options.NumMels, _options.FftSize, _options.HopLength, _options.FMin, _options.FMax, logMel: true);
        InitializeLayers();
    }

    internal static async Task<PANNs<T>> CreateAsync(PANNsOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new PANNsOptions(); string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp)) { var dl = new OnnxModelDownloader(); mp = await dl.DownloadAsync("panns", "cnn14_audioset.onnx", progress: progress, cancellationToken); options.ModelPath = mp; }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.EmbeddingDim, outputSize: (options.CustomLabels ?? AudioSetLabels).Length);
        return new PANNs<T>(arch, mp, options);
    }

    #endregion

    #region IAudioEventDetector

    /// <inheritdoc />
    public IReadOnlyList<string> SupportedEvents => ClassLabels;
    public IReadOnlyList<string> EventLabels => ClassLabels;
    /// <inheritdoc />
    public double TimeResolution => _options.WindowSize * (1 - _options.WindowOverlap);

    /// <inheritdoc />
    public AudioEventResult<T> Detect(Tensor<T> audio) { ThrowIfDisposed(); return Detect(audio, NumOps.FromDouble(_options.Threshold)); }

    /// <inheritdoc />
    public AudioEventResult<T> Detect(Tensor<T> audio, T threshold)
    {
        ThrowIfDisposed(); double dur = audio.Length / (double)_options.SampleRate;
        var wins = SplitIntoWindows(audio); var all = new List<AudioEvent<T>>();
        for (int w = 0; w < wins.Count; w++) { double st = w * TimeResolution; var mel = _melSpectrogram?.Forward(wins[w]) ?? throw new InvalidOperationException("MelSpectrogram not initialized."); var sc = ClassifyWindow(mel); foreach (int i in AudioEventDetectionOps.IndicesAtOrAbove(Engine, sc, ClassLabels.Count, threshold)) all.Add(new AudioEvent<T> { EventType = ClassLabels[i], Confidence = sc[i], StartTime = st, EndTime = Math.Min(st + _options.WindowSize, dur), PeakTime = st + _options.WindowSize / 2 }); }
        var merged = MergeEvents(all);
        return new AudioEventResult<T> { Events = merged, TotalDuration = dur, DetectedEventTypes = merged.Select(e => e.EventType).Distinct().ToList(), EventStats = ComputeEventStatistics(merged) };
    }

    /// <inheritdoc />
    public Task<AudioEventResult<T>> DetectAsync(Tensor<T> audio, CancellationToken ct = default) => Task.Run(() => Detect(audio), ct);
    /// <inheritdoc />
    public AudioEventResult<T> DetectSpecific(Tensor<T> audio, IReadOnlyList<string> types) => DetectSpecific(audio, types, NumOps.FromDouble(_options.Threshold));
    /// <inheritdoc />
    public AudioEventResult<T> DetectSpecific(Tensor<T> audio, IReadOnlyList<string> types, T threshold) { var r = Detect(audio, threshold); var s = new HashSet<string>(types, StringComparer.OrdinalIgnoreCase); var f = r.Events.Where(e => s.Contains(e.EventType)).ToList(); return new AudioEventResult<T> { Events = f, TotalDuration = r.TotalDuration, DetectedEventTypes = f.Select(e => e.EventType).Distinct().ToList(), EventStats = ComputeEventStatistics(f) }; }

    /// <inheritdoc />
    public Tensor<T> GetEventProbabilities(Tensor<T> audio)
    {
        ThrowIfDisposed(); var wins = SplitIntoWindows(audio); var rows = new List<Tensor<T>>(wins.Count);
        for (int w = 0; w < wins.Count; w++) { var mel = _melSpectrogram?.Forward(wins[w]) ?? throw new InvalidOperationException(); rows.Add(AudioEventDetectionOps.ScoreRow(ClassifyWindow(mel), ClassLabels.Count)); }
        return AudioEventDetectionOps.StackScoreRows(Engine, rows, ClassLabels.Count);
    }

    /// <inheritdoc />
    public IStreamingEventDetectionSession<T> StartStreamingSession() => StartStreamingSession(_options.SampleRate, NumOps.FromDouble(_options.Threshold));
    /// <inheritdoc />
    public IStreamingEventDetectionSession<T> StartStreamingSession(int sr, T thr)
    {
        if (sr <= 0)
            throw new ArgumentOutOfRangeException(nameof(sr), "Sample rate must be positive.");
        return new PANNsStreamingSession(this, sr, thr);
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultPANNsLayers(numMels: _options.NumMels, baseChannels: _options.BaseChannels, numBlocks: _options.NumBlocks, embeddingDim: _options.EmbeddingDim, numClasses: ClassLabels.Count, dropoutRate: _options.DropoutRate));
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        var modelInput = NormalizeModelInput(input);
        if (IsOnnxMode && OnnxEncoder is not null) return PostprocessOutput(OnnxEncoder.Run(modelInput));
        // PANNs (Kong et al., 2020) emits SIGMOID clip-wise probabilities, one per AudioSet tag —
        // multi-label scores in [0,1], never negative. Route the raw head through PostprocessOutput
        // (the sigmoid) so the public prediction is a probability regardless of whether the head is
        // the sigmoid-terminated CNN14 factory or a linear-headed architecture the caller supplied.
        // Force inference mode for the forward so the probabilities are deterministic (dropout off).
        bool wasTraining = IsTrainingMode;
        if (wasTraining) SetTrainingMode(false);
        try
        {
            var c = modelInput;
            foreach (var l in Layers) c = l.Forward(c);
            return PostprocessOutput(c);
        }
        finally
        {
            if (wasTraining) SetTrainingMode(true);
        }
    }

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
        => base.ForwardForTraining(NormalizeModelInput(input));

    /// <inheritdoc />
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
        => base.GetNamedLayerActivations(NormalizeModelInput(input));

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <inheritdoc />
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => _melSpectrogram?.Forward(rawAudio) ?? throw new InvalidOperationException("MelSpectrogram not initialized.");
    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => AudioEventDetectionOps.SigmoidDetached(Engine, o);

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "PANNs-Native" : "PANNs-ONNX", Description = "PANNs CNN14 (Kong et al., 2020)", FeatureCount = ClassLabels.Count, Complexity = _options.NumBlocks };
        m.AdditionalInfo["Architecture"] = "PANNs-CNN14"; m.AdditionalInfo["EmbeddingDim"] = _options.EmbeddingDim.ToString();
        m.AdditionalInfo["NumBlocks"] = _options.NumBlocks.ToString(); m.AdditionalInfo["NumClasses"] = ClassLabels.Count.ToString();
        return m;
    }
    #endregion

    #region Helpers

    private T[] ClassifyWindow(Tensor<T> melSpec)
    {
        Tensor<T> output;
        if (IsOnnxMode && OnnxEncoder is not null)
        {
            var input = NormalizeModelInput(melSpec);
            output = PostprocessOutput(OnnxEncoder.Run(input));
        }
        else if (_useNativeMode)
        {
            output = Predict(melSpec);
        }
        else
        {
            throw new InvalidOperationException(
                "No model available for classification. Provide an ONNX model path or use native training mode.");
        }

        return AudioEventDetectionOps.HeadScores(Engine, output, ClassLabels.Count);
    }

    private Tensor<T> NormalizeModelInput(Tensor<T> input)
    {
        if (input.Rank == 4)
            return input;

        if (input.Rank == 1)
            return NormalizeModelInput(PreprocessAudio(input));

        if (input.Rank == 2)
            return Engine.Reshape(input, [1, 1, input.Shape[0], input.Shape[1]]);

        if (input.Rank == 3)
            return Engine.Reshape(input, [input.Shape[0], 1, input.Shape[1], input.Shape[2]]);

        throw new ArgumentException(
            $"PANNs expects raw rank-1 audio or rank-2/3/4 mel features, got rank {input.Rank}.",
            nameof(input));
    }
    private List<Tensor<T>> SplitIntoWindows(Tensor<T> audio) { int ws = (int)(_options.WindowSize * _options.SampleRate), hs = (int)(ws * (1 - _options.WindowOverlap)); if (hs <= 0) hs = 1; return AudioEventDetectionOps.SplitIntoWindows(Engine, audio, ws, hs, ws / 10); }
    private List<AudioEvent<T>> MergeEvents(List<AudioEvent<T>> events) => AudioEventDetectionOps.MergeEvents(Engine, events, 0.1);
    private Dictionary<string, EventStatistics<T>> ComputeEventStatistics(IReadOnlyList<AudioEvent<T>> events) => AudioEventDetectionOps.ComputeEventStatistics(Engine, events);

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(PANNs<T>)); }
    /// <inheritdoc />
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    #endregion

    #region Streaming Session

    private sealed class PANNsStreamingSession : IStreamingEventDetectionSession<T>
    {
        private readonly PANNs<T> _det; private readonly T _thr; private readonly List<T> _buf; private readonly List<AudioEvent<T>> _new; private readonly Dictionary<string, T> _state; private readonly int _ws; private double _pt; private volatile bool _disp; private readonly object _lk = new();
        public event EventHandler<AudioEvent<T>>? EventDetected;
        public PANNsStreamingSession(PANNs<T> det, int sr, T thr) { _det = det; _thr = thr; _buf = []; _new = []; _state = new Dictionary<string, T>(); _ws = (int)(det._options.WindowSize * sr); foreach (var l in det.ClassLabels) _state[l] = det.NumOps.Zero; }
        public void FeedAudio(Tensor<T> chunk) { if (_disp) throw new ObjectDisposedException(nameof(PANNsStreamingSession)); List<AudioEvent<T>>? raise = null; lock (_lk) { if (_disp) throw new ObjectDisposedException(nameof(PANNsStreamingSession)); _buf.AddRange(chunk.ToArray()); while (_buf.Count >= _ws) { var w = new Tensor<T>(_buf.GetRange(0, _ws).ToArray(), [_ws]); var mel = _det._melSpectrogram?.Forward(w) ?? throw new InvalidOperationException(); var scores = _det.ClassifyWindow(mel); int nc = Math.Min(scores.Length, _det.ClassLabels.Count); for (int i = 0; i < nc; i++) _state[_det.ClassLabels[i]] = scores[i]; foreach (int i in AudioEventDetectionOps.IndicesAtOrAbove(_det.Engine, scores, nc, _thr)) { var e = new AudioEvent<T> { EventType = _det.ClassLabels[i], Confidence = scores[i], StartTime = _pt, EndTime = _pt + _det._options.WindowSize, PeakTime = _pt + _det._options.WindowSize / 2 }; _new.Add(e); raise ??= []; raise.Add(e); } int hs = (int)(_ws * (1 - _det._options.WindowOverlap)); if (hs <= 0) hs = 1; _buf.RemoveRange(0, hs); _pt += hs / (double)_det._options.SampleRate; } } if (raise is not null) foreach (var e in raise) EventDetected?.Invoke(this, e); }
        public IReadOnlyList<AudioEvent<T>> GetNewEvents() { lock (_lk) { var e = _new.ToList(); _new.Clear(); return e; } }
        public IReadOnlyDictionary<string, T> GetCurrentState() { lock (_lk) { return new Dictionary<string, T>(_state); } }
        public void Dispose() { if (_disp) return; lock (_lk) { if (_disp) return; _disp = true; _buf.Clear(); _new.Clear(); _state.Clear(); } }
    }

    #endregion
}
