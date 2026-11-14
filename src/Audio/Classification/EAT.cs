using AiDotNet.LearningRateSchedulers;
using AiDotNet.Attributes;
using AiDotNet.Audio.Features;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// EAT (Efficient Audio Transformer) model for efficient audio event detection and classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// EAT (Chen et al., 2024) achieves competitive performance with significantly less compute
/// than BEATs through an efficient self-supervised pre-training approach using teacher-student
/// distillation. It reaches 49.7% mAP on AudioSet-2M using only 10% of BEATs' compute.
/// </para>
/// <para>
/// <b>Architecture:</b> EAT uses a standard ViT-Base encoder with an EMA teacher:
/// <list type="number">
/// <item><b>Student encoder</b>: 12-layer Transformer that processes visible (unmasked) patches</item>
/// <item><b>Teacher encoder</b>: EMA copy of student that processes all patches, providing targets</item>
/// <item><b>Masked prediction</b>: Student predicts teacher's representations for masked patches</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> EAT is like having a wise teacher and a student. The teacher sees
/// the full spectrogram and the student only sees 25% of it. The student must predict what
/// the teacher sees. As the student improves, the teacher slowly updates to match, creating
/// a virtuous learning cycle. This is more efficient than BEATs because the student processes
/// fewer patches.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 768, outputSize: 527);
/// var eat = new EAT&lt;float&gt;(arch, "eat_audioset.onnx");
/// var result = eat.Detect(audioTensor);
/// </code>
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "EAT: Self-Supervised Pre-Training with Efficient Audio Transformer" (Chen et al., 2024)</item>
/// </list>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("EAT: Self-Supervised Pre-Training with Efficient Audio Transformer", "https://arxiv.org/abs/2401.03497", Year = 2024, Authors = "Wenxi Chen, Yuzhe Liang, Ziyang Ma, Zhisheng Zheng, Xie Chen")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-4, Beta1 = 0.9, Beta2 = 0.95,
                WeightDecay = 0.05, ReferenceBatchSize = 12, MinLearningRate = 1e-6,
                Schedule = LearningRateSchedulerType.CosineAnnealing,
                Phase = TrainingPhase.PreTraining,
                Source = "Chen et al. 2024, Appendix A.1 Table 4: AdamW with beta1 0.9 and beta2 0.95, "
                        + "weight decay 0.05, a cosine schedule, a peak learning rate of 0.0005 decaying "
                        + "to a minimum of 0.000001, and a batch size of 12 over 10 epochs of AS-2M. The "
                        + "body text calls it Adam while citing Loshchilov and Hutter 2017, which is the "
                        + "AdamW paper, and the appendix table names AdamW outright. The step and "
                        + "warm-up rows of that table cannot be assigned to their columns from the "
                        + "extracted text, so no warm-up length is declared.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-5,
                Phase = TrainingPhase.FineTuning,
                Provenance = RecipeProvenance.PerDataset,
                Source = "Chen et al. 2024, Appendix A.1 Table 4: fine-tuning uses a peak learning rate "
                        + "of 0.00005 on AS-2M, AS-20K and ESC-50, and 0.0002 on SPC-2. The value "
                        + "declared here is the one shared by three of the four datasets.")]
public partial class EAT<T> : AudioClassifierBase<T>, IAudioEventDetector<T>
{
    #region Fields

    private readonly EATOptions _options;
    private MelSpectrogram<T>? _melSpectrogram;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    public static readonly string[] AudioSetLabels = BEATs<T>.AudioSetLabels;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates an EAT model for ONNX inference mode.
    /// </summary>
    public EAT(NeuralNetworkArchitecture<T> architecture, string modelPath, EATOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options = options ?? new EATOptions();
        Options = _options;
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        _melSpectrogram = new MelSpectrogram<T>(_options.SampleRate, _options.NumMels, _options.FftSize, _options.HopLength, _options.FMin, _options.FMax, logMel: true);
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        ClassLabels = _options.CustomLabels ?? AudioSetLabels;
        InitializeLayers();
    }

    /// <summary>
    /// Creates an EAT model for native training mode.
    /// </summary>
    public EAT(NeuralNetworkArchitecture<T> architecture, EATOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new EATOptions();
        Options = _options;
        _useNativeMode = true;
        // The rate this model publishes on its own options. Built bare, the optimizer
        // would use its own default instead and LearningRate would be configuration that
        // nothing reads — the defect that diverged MusicFlamingo's training.
        _optimizer = optimizer
    ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
    ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
        new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        { InitialLearningRate = _options.LearningRate });
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        ClassLabels = _options.CustomLabels ?? AudioSetLabels;
        _melSpectrogram = new MelSpectrogram<T>(_options.SampleRate, _options.NumMels, _options.FftSize, _options.HopLength, _options.FMin, _options.FMax, logMel: true);
        InitializeLayers();
    }

    internal static async Task<EAT<T>> CreateAsync(EATOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new EATOptions();
        string modelPath = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(modelPath))
        {
            var downloader = new OnnxModelDownloader();
            modelPath = await downloader.DownloadAsync("eat", "eat_audioset.onnx", progress: progress, cancellationToken);
            options.ModelPath = modelPath;
        }
        var architecture = new NeuralNetworkArchitecture<T>(inputFeatures: options.EmbeddingDim, outputSize: (options.CustomLabels ?? AudioSetLabels).Length);
        return new EAT<T>(architecture, modelPath, options);
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
    public Task<AudioEventResult<T>> DetectAsync(Tensor<T> audio, CancellationToken cancellationToken = default) => Task.Run(() => Detect(audio), cancellationToken);
    /// <inheritdoc />
    public AudioEventResult<T> DetectSpecific(Tensor<T> audio, IReadOnlyList<string> eventTypes) => DetectSpecific(audio, eventTypes, NumOps.FromDouble(_options.Threshold));

    /// <inheritdoc />
    public AudioEventResult<T> DetectSpecific(Tensor<T> audio, IReadOnlyList<string> eventTypes, T threshold)
    {
        var result = Detect(audio, threshold);
        var set = new HashSet<string>(eventTypes, StringComparer.OrdinalIgnoreCase);
        var filtered = result.Events.Where(e => set.Contains(e.EventType)).ToList();
        return new AudioEventResult<T> { Events = filtered, TotalDuration = result.TotalDuration, DetectedEventTypes = filtered.Select(e => e.EventType).Distinct().ToList(), EventStats = ComputeEventStatistics(filtered) };
    }

    /// <inheritdoc />
    public Tensor<T> GetEventProbabilities(Tensor<T> audio)
    {
        ThrowIfDisposed(); var wins = SplitIntoWindows(audio); var rows = new List<Tensor<T>>(wins.Count);
        for (int w = 0; w < wins.Count; w++) { var mel = _melSpectrogram?.Forward(wins[w]) ?? throw new InvalidOperationException("MelSpectrogram not initialized."); rows.Add(AudioEventDetectionOps.ScoreRow(ClassifyWindow(mel), ClassLabels.Count)); }
        return AudioEventDetectionOps.StackScoreRows(Engine, rows, ClassLabels.Count);
    }

    /// <inheritdoc />
    public IStreamingEventDetectionSession<T> StartStreamingSession() => StartStreamingSession(_options.SampleRate, NumOps.FromDouble(_options.Threshold));
    /// <inheritdoc />
    public IStreamingEventDetectionSession<T> StartStreamingSession(int sampleRate, T threshold) => new EATStreamingSession(this, sampleRate, threshold);

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultEATLayers(
                patchFeatureSize: _options.PatchSize * _options.PatchSize, embeddingDim: _options.EmbeddingDim,
                numEncoderLayers: _options.NumEncoderLayers, numAttentionHeads: _options.NumAttentionHeads,
                feedForwardDim: _options.FeedForwardDim, numClasses: ClassLabels.Count, dropoutRate: _options.DropoutRate));
        }
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        var current = input;
        foreach (var layer in Layers) current = layer.Forward(current);
        return current;
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <inheritdoc />
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => _melSpectrogram?.Forward(rawAudio) ?? throw new InvalidOperationException("MelSpectrogram not initialized.");

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => AudioEventDetectionOps.SigmoidDetached(Engine, modelOutput);

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "EAT-Native" : "EAT-ONNX", Description = "EAT: Efficient Audio Transformer (Chen et al., 2024)", FeatureCount = ClassLabels.Count, Complexity = _options.NumEncoderLayers };
        m.AdditionalInfo["Architecture"] = "EAT"; m.AdditionalInfo["EmbeddingDim"] = _options.EmbeddingDim.ToString();
        m.AdditionalInfo["NumEncoderLayers"] = _options.NumEncoderLayers.ToString(); m.AdditionalInfo["NumClasses"] = ClassLabels.Count.ToString();
        return m;
    }





    #endregion

    #region Helpers

    private T[] ClassifyWindow(Tensor<T> melSpec)
    {
        Tensor<T> output;
        if (IsOnnxMode && OnnxEncoder is not null) { output = OnnxEncoder.Run(Engine.Reshape(melSpec, [1, 1, melSpec.Shape[0], melSpec.Shape[1]])); }
        else if (_useNativeMode) { output = Predict(AudioEventDetectionOps.Flatten(Engine, melSpec)); }
        else { throw new InvalidOperationException("No model available for classification. Provide an ONNX model path or use native training mode."); }
        return AudioEventDetectionOps.SigmoidScores(Engine, output, ClassLabels.Count);
    }

    private List<Tensor<T>> SplitIntoWindows(Tensor<T> audio) { int ws = (int)(_options.WindowSize * _options.SampleRate), hs = (int)(ws * (1 - _options.WindowOverlap)); if (hs <= 0) hs = 1; return AudioEventDetectionOps.SplitIntoWindows(Engine, audio, ws, hs, ws / 10); }

    private List<AudioEvent<T>> MergeEvents(List<AudioEvent<T>> events) => AudioEventDetectionOps.MergeEvents(Engine, events, 0.1);

    private Dictionary<string, EventStatistics<T>> ComputeEventStatistics(IReadOnlyList<AudioEvent<T>> events) => AudioEventDetectionOps.ComputeEventStatistics(Engine, events);

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(EAT<T>)); }
    /// <inheritdoc />
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    #endregion

    #region Streaming Session

    private sealed class EATStreamingSession : IStreamingEventDetectionSession<T>
    {
        private readonly EAT<T> _det; private readonly T _thr; private readonly List<T> _buf; private readonly List<AudioEvent<T>> _new;
        private readonly Dictionary<string, T> _state; private readonly int _ws; private double _pt; private volatile bool _disp; private readonly object _lk = new();
        public event EventHandler<AudioEvent<T>>? EventDetected;

        public EATStreamingSession(EAT<T> det, int sr, T thr)
        {
            _det = det; _thr = thr; _buf = []; _new = []; _state = new Dictionary<string, T>();
            _ws = (int)(det._options.WindowSize * sr);
            foreach (var l in det.ClassLabels) _state[l] = det.NumOps.Zero;
        }

        public void FeedAudio(Tensor<T> chunk) { if (_disp) throw new ObjectDisposedException(nameof(EATStreamingSession)); List<AudioEvent<T>>? raise = null; lock (_lk) { if (_disp) throw new ObjectDisposedException(nameof(EATStreamingSession)); _buf.AddRange(chunk.ToArray()); while (_buf.Count >= _ws) { var w = new Tensor<T>(_buf.GetRange(0, _ws).ToArray(), [_ws]); var mel = _det._melSpectrogram?.Forward(w) ?? throw new InvalidOperationException("MelSpectrogram not initialized."); var scores = _det.ClassifyWindow(mel); int nc = Math.Min(scores.Length, _det.ClassLabels.Count); for (int i = 0; i < nc; i++) _state[_det.ClassLabels[i]] = scores[i]; foreach (int i in AudioEventDetectionOps.IndicesAtOrAbove(_det.Engine, scores, nc, _thr)) { var e = new AudioEvent<T> { EventType = _det.ClassLabels[i], Confidence = scores[i], StartTime = _pt, EndTime = _pt + _det._options.WindowSize, PeakTime = _pt + _det._options.WindowSize / 2 }; _new.Add(e); raise ??= []; raise.Add(e); } int hs = (int)(_ws * (1 - _det._options.WindowOverlap)); if (hs <= 0) hs = 1; _buf.RemoveRange(0, hs); _pt += hs / (double)_det._options.SampleRate; } } if (raise is not null) foreach (var e in raise) EventDetected?.Invoke(this, e); }

        public IReadOnlyList<AudioEvent<T>> GetNewEvents() { lock (_lk) { var e = _new.ToList(); _new.Clear(); return e; } }
        public IReadOnlyDictionary<string, T> GetCurrentState() { lock (_lk) { return new Dictionary<string, T>(_state); } }
        public void Dispose() { if (_disp) return; lock (_lk) { if (_disp) return; _disp = true; _buf.Clear(); _new.Clear(); _state.Clear(); } }
    }

    #endregion
}
