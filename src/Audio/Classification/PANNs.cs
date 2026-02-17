using AiDotNet.Audio.Features;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
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
public class PANNs<T> : AudioClassifierBase<T>, IAudioEventDetector<T>
{
    #region Fields

    private readonly PANNsOptions _options;
    public override ModelOptions GetOptions() => _options;
    private MelSpectrogram<T>? _melSpectrogram;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    public static readonly string[] AudioSetLabels = BEATs<T>.AudioSetLabels;

    #endregion

    #region Constructors

    public PANNs(NeuralNetworkArchitecture<T> architecture, string modelPath, PANNsOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new PANNsOptions(); _useNativeMode = false;
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        _melSpectrogram = new MelSpectrogram<T>(_options.SampleRate, _options.NumMels, _options.FftSize, _options.HopLength, _options.FMin, _options.FMax, logMel: true);
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        ClassLabels = _options.CustomLabels ?? AudioSetLabels;
        InitializeLayers();
    }

    public PANNs(NeuralNetworkArchitecture<T> architecture, PANNsOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new PANNsOptions(); _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
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

    public IReadOnlyList<string> SupportedEvents => ClassLabels;
    public IReadOnlyList<string> EventLabels => ClassLabels;
    public double TimeResolution => _options.WindowSize * (1 - _options.WindowOverlap);

    public AudioEventResult<T> Detect(Tensor<T> audio) { ThrowIfDisposed(); return Detect(audio, NumOps.FromDouble(_options.Threshold)); }

    public AudioEventResult<T> Detect(Tensor<T> audio, T threshold)
    {
        ThrowIfDisposed(); double tv = NumOps.ToDouble(threshold), dur = audio.Length / (double)_options.SampleRate;
        var wins = SplitIntoWindows(audio); var all = new List<AudioEvent<T>>();
        for (int w = 0; w < wins.Count; w++) { double st = w * TimeResolution; var mel = _melSpectrogram?.Forward(wins[w]) ?? throw new InvalidOperationException("MelSpectrogram not initialized."); var sc = ClassifyWindow(mel); for (int i = 0; i < sc.Length && i < ClassLabels.Count; i++) if (NumOps.ToDouble(sc[i]) >= tv) all.Add(new AudioEvent<T> { EventType = ClassLabels[i], Confidence = sc[i], StartTime = st, EndTime = Math.Min(st + _options.WindowSize, dur), PeakTime = st + _options.WindowSize / 2 }); }
        var merged = MergeEvents(all);
        return new AudioEventResult<T> { Events = merged, TotalDuration = dur, DetectedEventTypes = merged.Select(e => e.EventType).Distinct().ToList(), EventStats = ComputeEventStatistics(merged) };
    }

    public Task<AudioEventResult<T>> DetectAsync(Tensor<T> audio, CancellationToken ct = default) => Task.Run(() => Detect(audio), ct);
    public AudioEventResult<T> DetectSpecific(Tensor<T> audio, IReadOnlyList<string> types) => DetectSpecific(audio, types, NumOps.FromDouble(_options.Threshold));
    public AudioEventResult<T> DetectSpecific(Tensor<T> audio, IReadOnlyList<string> types, T threshold) { var r = Detect(audio, threshold); var s = new HashSet<string>(types, StringComparer.OrdinalIgnoreCase); var f = r.Events.Where(e => s.Contains(e.EventType)).ToList(); return new AudioEventResult<T> { Events = f, TotalDuration = r.TotalDuration, DetectedEventTypes = f.Select(e => e.EventType).Distinct().ToList(), EventStats = ComputeEventStatistics(f) }; }

    public Tensor<T> GetEventProbabilities(Tensor<T> audio)
    {
        ThrowIfDisposed(); var wins = SplitIntoWindows(audio); var p = new Tensor<T>([wins.Count, ClassLabels.Count]);
        for (int w = 0; w < wins.Count; w++) { var mel = _melSpectrogram?.Forward(wins[w]) ?? throw new InvalidOperationException(); var sc = ClassifyWindow(mel); for (int i = 0; i < ClassLabels.Count && i < sc.Length; i++) p[w, i] = sc[i]; }
        return p;
    }

    public IStreamingEventDetectionSession<T> StartStreamingSession() => StartStreamingSession(_options.SampleRate, NumOps.FromDouble(_options.Threshold));
    public IStreamingEventDetectionSession<T> StartStreamingSession(int sr, T thr) => new PANNsStreamingSession(this, sr, thr);

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultPANNsLayers(numMels: _options.NumMels, baseChannels: _options.BaseChannels, numBlocks: _options.NumBlocks, embeddingDim: _options.EmbeddingDim, numClasses: ClassLabels.Count, dropoutRate: _options.DropoutRate));
    }

    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true); var output = Predict(input); var grad = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(grad); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false);
    }

    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => _melSpectrogram?.Forward(rawAudio) ?? throw new InvalidOperationException("MelSpectrogram not initialized.");
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) { var r = new Tensor<T>(o.Shape); for (int i = 0; i < o.Length; i++) r[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-NumOps.ToDouble(o[i])))); return r; }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "PANNs-Native" : "PANNs-ONNX", Description = "PANNs CNN14 (Kong et al., 2020)", ModelType = ModelType.NeuralNetwork, FeatureCount = ClassLabels.Count, Complexity = _options.NumBlocks };
        m.AdditionalInfo["Architecture"] = "PANNs-CNN14"; m.AdditionalInfo["EmbeddingDim"] = _options.EmbeddingDim.ToString();
        m.AdditionalInfo["NumBlocks"] = _options.NumBlocks.ToString(); m.AdditionalInfo["NumClasses"] = ClassLabels.Count.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w) { w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty); w.Write(_options.SampleRate); w.Write(_options.NumMels); w.Write(_options.FftSize); w.Write(_options.HopLength); w.Write(_options.EmbeddingDim); w.Write(_options.NumBlocks); w.Write(_options.BaseChannels); w.Write(_options.Threshold); w.Write(_options.WindowSize); w.Write(_options.WindowOverlap); w.Write(_options.DropoutRate); w.Write(ClassLabels.Count); foreach (var l in ClassLabels) w.Write(l); }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.NumMels = r.ReadInt32(); _options.FftSize = r.ReadInt32(); _options.HopLength = r.ReadInt32();
        _options.EmbeddingDim = r.ReadInt32(); _options.NumBlocks = r.ReadInt32(); _options.BaseChannels = r.ReadInt32();
        _options.Threshold = r.ReadDouble(); _options.WindowSize = r.ReadDouble(); _options.WindowOverlap = r.ReadDouble(); _options.DropoutRate = r.ReadDouble();
        int n = r.ReadInt32(); var labels = new string[n]; for (int i = 0; i < n; i++) labels[i] = r.ReadString(); ClassLabels = labels;
        _melSpectrogram = new MelSpectrogram<T>(_options.SampleRate, _options.NumMels, _options.FftSize, _options.HopLength, _options.FMin, _options.FMax, logMel: true);
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new PANNs<T>(Architecture, _options);

    #endregion

    #region Helpers

    private T[] ClassifyWindow(Tensor<T> melSpec) { Tensor<T> output; if (IsOnnxMode && OnnxEncoder is not null) { var inp = new Tensor<T>([1, 1, melSpec.Shape[0], melSpec.Shape[1]]); for (int t = 0; t < melSpec.Shape[0]; t++) for (int f = 0; f < melSpec.Shape[1]; f++) inp[0, 0, t, f] = melSpec[t, f]; output = OnnxEncoder.Run(inp); } else if (_useNativeMode) { var inp = new Tensor<T>([melSpec.Length]); int idx = 0; for (int t = 0; t < melSpec.Shape[0]; t++) for (int f = 0; f < melSpec.Shape[1]; f++) inp[idx++] = melSpec[t, f]; output = Predict(inp); } else { throw new InvalidOperationException("No model available for classification. Provide an ONNX model path or use native training mode."); } var scores = new T[ClassLabels.Count]; for (int i = 0; i < Math.Min(output.Length, scores.Length); i++) scores[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-NumOps.ToDouble(output[i])))); return scores; }
    private List<Tensor<T>> SplitIntoWindows(Tensor<T> audio) { var w = new List<Tensor<T>>(); int ws = (int)(_options.WindowSize * _options.SampleRate), hs = (int)(ws * (1 - _options.WindowOverlap)); if (hs <= 0) hs = 1; int ls = 0; for (int s = 0; s + ws <= audio.Length; s += hs) { var t = new Tensor<T>([ws]); for (int i = 0; i < ws; i++) t[i] = audio[s + i]; w.Add(t); ls = s + hs; } int rs = w.Count > 0 ? ls : 0, rem = audio.Length - rs; if (rem > ws / 10) { var t = new Tensor<T>([ws]); for (int i = 0; i < rem && i < ws; i++) t[i] = audio[rs + i]; w.Add(t); } else if (w.Count == 0 && audio.Length > 0) { var t = new Tensor<T>([ws]); for (int i = 0; i < audio.Length; i++) t[i] = audio[i]; w.Add(t); } return w; }
    private List<AudioEvent<T>> MergeEvents(List<AudioEvent<T>> events) { if (events.Count == 0) return events; var m = new List<AudioEvent<T>>(); foreach (var g in events.GroupBy(e => e.EventType)) { var sorted = g.OrderBy(e => e.StartTime).ToList(); var cur = sorted[0]; for (int i = 1; i < sorted.Count; i++) { var next = sorted[i]; if (next.StartTime <= cur.EndTime + 0.1) { double cc = NumOps.ToDouble(cur.Confidence), nc = NumOps.ToDouble(next.Confidence); cur = new AudioEvent<T> { EventType = cur.EventType, StartTime = cur.StartTime, EndTime = Math.Max(cur.EndTime, next.EndTime), Confidence = cc > nc ? cur.Confidence : next.Confidence, PeakTime = cc > nc ? cur.PeakTime : next.PeakTime }; } else { m.Add(cur); cur = next; } } m.Add(cur); } return m.OrderBy(e => e.StartTime).ToList(); }
    private Dictionary<string, EventStatistics<T>> ComputeEventStatistics(IReadOnlyList<AudioEvent<T>> events) { var s = new Dictionary<string, EventStatistics<T>>(); foreach (var g in events.GroupBy(e => e.EventType)) { var l = g.ToList(); s[g.Key] = new EventStatistics<T> { Count = l.Count, TotalDuration = l.Sum(e => e.Duration), AverageConfidence = NumOps.FromDouble(l.Average(e => NumOps.ToDouble(e.Confidence))), MaxConfidence = NumOps.FromDouble(l.Max(e => NumOps.ToDouble(e.Confidence))) }; } return s; }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(PANNs<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    #endregion

    #region Streaming Session

    private sealed class PANNsStreamingSession : IStreamingEventDetectionSession<T>
    {
        private readonly PANNs<T> _det; private readonly T _thr; private readonly List<T> _buf; private readonly List<AudioEvent<T>> _new; private readonly Dictionary<string, T> _state; private readonly int _ws; private double _pt; private volatile bool _disp; private readonly object _lk = new();
        public event EventHandler<AudioEvent<T>>? EventDetected;
        public PANNsStreamingSession(PANNs<T> det, int sr, T thr) { _det = det; _thr = thr; _buf = []; _new = []; _state = new Dictionary<string, T>(); _ws = (int)(det._options.WindowSize * sr); foreach (var l in det.ClassLabels) _state[l] = det.NumOps.Zero; }
        public void FeedAudio(Tensor<T> chunk) { if (_disp) throw new ObjectDisposedException(nameof(PANNsStreamingSession)); List<AudioEvent<T>>? raise = null; lock (_lk) { if (_disp) throw new ObjectDisposedException(nameof(PANNsStreamingSession)); for (int i = 0; i < chunk.Length; i++) _buf.Add(chunk[i]); while (_buf.Count >= _ws) { var w = new Tensor<T>([_ws]); for (int i = 0; i < _ws; i++) w[i] = _buf[i]; var mel = _det._melSpectrogram?.Forward(w) ?? throw new InvalidOperationException(); var scores = _det.ClassifyWindow(mel); double tv = _det.NumOps.ToDouble(_thr); for (int i = 0; i < scores.Length && i < _det.ClassLabels.Count; i++) { _state[_det.ClassLabels[i]] = scores[i]; if (_det.NumOps.ToDouble(scores[i]) >= tv) { var e = new AudioEvent<T> { EventType = _det.ClassLabels[i], Confidence = scores[i], StartTime = _pt, EndTime = _pt + _det._options.WindowSize, PeakTime = _pt + _det._options.WindowSize / 2 }; _new.Add(e); raise ??= []; raise.Add(e); } } int hs = (int)(_ws * (1 - _det._options.WindowOverlap)); if (hs <= 0) hs = 1; _buf.RemoveRange(0, hs); _pt += hs / (double)_det._options.SampleRate; } } if (raise is not null) foreach (var e in raise) EventDetected?.Invoke(this, e); }
        public IReadOnlyList<AudioEvent<T>> GetNewEvents() { lock (_lk) { var e = _new.ToList(); _new.Clear(); return e; } }
        public IReadOnlyDictionary<string, T> GetCurrentState() { lock (_lk) { return new Dictionary<string, T>(_state); } }
        public void Dispose() { if (_disp) return; lock (_lk) { if (_disp) return; _disp = true; _buf.Clear(); _new.Clear(); _state.Clear(); } }
    }

    #endregion
}
