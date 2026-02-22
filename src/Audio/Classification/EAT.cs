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
public class EAT<T> : AudioClassifierBase<T>, IAudioEventDetector<T>
{
    #region Fields

    private readonly EATOptions _options;
    public override ModelOptions GetOptions() => _options;
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
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
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

    public IReadOnlyList<string> SupportedEvents => ClassLabels;
    public IReadOnlyList<string> EventLabels => ClassLabels;
    public double TimeResolution => _options.WindowSize * (1 - _options.WindowOverlap);

    public AudioEventResult<T> Detect(Tensor<T> audio) { ThrowIfDisposed(); return Detect(audio, NumOps.FromDouble(_options.Threshold)); }

    public AudioEventResult<T> Detect(Tensor<T> audio, T threshold)
    {
        ThrowIfDisposed();
        double tv = NumOps.ToDouble(threshold);
        double totalDuration = audio.Length / (double)_options.SampleRate;
        var windows = SplitIntoWindows(audio);
        var allEvents = new List<AudioEvent<T>>();
        for (int w = 0; w < windows.Count; w++)
        {
            double startTime = w * TimeResolution;
            var melSpec = _melSpectrogram?.Forward(windows[w]) ?? throw new InvalidOperationException("MelSpectrogram not initialized.");
            var scores = ClassifyWindow(melSpec);
            for (int i = 0; i < scores.Length && i < ClassLabels.Count; i++)
            {
                if (NumOps.ToDouble(scores[i]) >= tv)
                    allEvents.Add(new AudioEvent<T> { EventType = ClassLabels[i], Confidence = scores[i], StartTime = startTime, EndTime = Math.Min(startTime + _options.WindowSize, totalDuration), PeakTime = startTime + _options.WindowSize / 2 });
            }
        }
        var merged = MergeEvents(allEvents);
        return new AudioEventResult<T> { Events = merged, TotalDuration = totalDuration, DetectedEventTypes = merged.Select(e => e.EventType).Distinct().ToList(), EventStats = ComputeEventStatistics(merged) };
    }

    public Task<AudioEventResult<T>> DetectAsync(Tensor<T> audio, CancellationToken cancellationToken = default) => Task.Run(() => Detect(audio), cancellationToken);
    public AudioEventResult<T> DetectSpecific(Tensor<T> audio, IReadOnlyList<string> eventTypes) => DetectSpecific(audio, eventTypes, NumOps.FromDouble(_options.Threshold));

    public AudioEventResult<T> DetectSpecific(Tensor<T> audio, IReadOnlyList<string> eventTypes, T threshold)
    {
        var result = Detect(audio, threshold);
        var set = new HashSet<string>(eventTypes, StringComparer.OrdinalIgnoreCase);
        var filtered = result.Events.Where(e => set.Contains(e.EventType)).ToList();
        return new AudioEventResult<T> { Events = filtered, TotalDuration = result.TotalDuration, DetectedEventTypes = filtered.Select(e => e.EventType).Distinct().ToList(), EventStats = ComputeEventStatistics(filtered) };
    }

    public Tensor<T> GetEventProbabilities(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var windows = SplitIntoWindows(audio);
        var probs = new Tensor<T>([windows.Count, ClassLabels.Count]);
        for (int w = 0; w < windows.Count; w++)
        {
            var melSpec = _melSpectrogram?.Forward(windows[w]) ?? throw new InvalidOperationException("MelSpectrogram not initialized.");
            var scores = ClassifyWindow(melSpec);
            for (int i = 0; i < ClassLabels.Count && i < scores.Length; i++) probs[w, i] = scores[i];
        }
        return probs;
    }

    public IStreamingEventDetectionSession<T> StartStreamingSession() => StartStreamingSession(_options.SampleRate, NumOps.FromDouble(_options.Threshold));
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

    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        var current = input;
        foreach (var layer in Layers) current = layer.Forward(current);
        return current;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        var output = Predict(input);
        var gradient = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());
        var gradTensor = Tensor<T>.FromVector(gradient);
        for (int i = Layers.Count - 1; i >= 0; i--) gradTensor = Layers[i].Backward(gradTensor);
        _optimizer?.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("UpdateParameters is not supported in ONNX mode.");
        int idx = 0;
        foreach (var layer in Layers) { int c = layer.ParameterCount; layer.UpdateParameters(parameters.Slice(idx, c)); idx += c; }
    }

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => _melSpectrogram?.Forward(rawAudio) ?? throw new InvalidOperationException("MelSpectrogram not initialized.");

    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        var result = new Tensor<T>(modelOutput.Shape);
        for (int i = 0; i < modelOutput.Length; i++) { double l = NumOps.ToDouble(modelOutput[i]); result[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-l))); }
        return result;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "EAT-Native" : "EAT-ONNX", Description = "EAT: Efficient Audio Transformer (Chen et al., 2024)", ModelType = ModelType.NeuralNetwork, FeatureCount = ClassLabels.Count, Complexity = _options.NumEncoderLayers };
        m.AdditionalInfo["Architecture"] = "EAT"; m.AdditionalInfo["EmbeddingDim"] = _options.EmbeddingDim.ToString();
        m.AdditionalInfo["NumEncoderLayers"] = _options.NumEncoderLayers.ToString(); m.AdditionalInfo["NumClasses"] = ClassLabels.Count.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write(_options.SampleRate); writer.Write(_options.NumMels); writer.Write(_options.FftSize); writer.Write(_options.HopLength);
        writer.Write(_options.EmbeddingDim); writer.Write(_options.NumEncoderLayers); writer.Write(_options.NumAttentionHeads); writer.Write(_options.FeedForwardDim);
        writer.Write(_options.PatchSize); writer.Write(_options.PatchStride); writer.Write(_options.Threshold); writer.Write(_options.WindowSize); writer.Write(_options.WindowOverlap); writer.Write(_options.DropoutRate);
        writer.Write((int)_options.FMin); writer.Write((int)_options.FMax);
        writer.Write(ClassLabels.Count); foreach (var label in ClassLabels) writer.Write(label);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = reader.ReadInt32(); _options.NumMels = reader.ReadInt32(); _options.FftSize = reader.ReadInt32(); _options.HopLength = reader.ReadInt32();
        _options.EmbeddingDim = reader.ReadInt32(); _options.NumEncoderLayers = reader.ReadInt32(); _options.NumAttentionHeads = reader.ReadInt32(); _options.FeedForwardDim = reader.ReadInt32();
        _options.PatchSize = reader.ReadInt32(); _options.PatchStride = reader.ReadInt32(); _options.Threshold = reader.ReadDouble(); _options.WindowSize = reader.ReadDouble(); _options.WindowOverlap = reader.ReadDouble(); _options.DropoutRate = reader.ReadDouble();
        _options.FMin = reader.ReadInt32(); _options.FMax = reader.ReadInt32();
        int n = reader.ReadInt32(); var labels = new string[n]; for (int i = 0; i < n; i++) labels[i] = reader.ReadString(); ClassLabels = labels;
        _melSpectrogram = new MelSpectrogram<T>(_options.SampleRate, _options.NumMels, _options.FftSize, _options.HopLength, _options.FMin, _options.FMax, logMel: true);
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new EAT<T>(Architecture, mp, _options);
        return new EAT<T>(Architecture, _options);
    }

    #endregion

    #region Helpers

    private T[] ClassifyWindow(Tensor<T> melSpec)
    {
        Tensor<T> output;
        if (IsOnnxMode && OnnxEncoder is not null)
        {
            var input = new Tensor<T>([1, 1, melSpec.Shape[0], melSpec.Shape[1]]);
            for (int t = 0; t < melSpec.Shape[0]; t++) for (int f = 0; f < melSpec.Shape[1]; f++) input[0, 0, t, f] = melSpec[t, f];
            output = OnnxEncoder.Run(input);
        }
        else if (_useNativeMode)
        {
            var input = new Tensor<T>([melSpec.Length]); int idx = 0;
            for (int t = 0; t < melSpec.Shape[0]; t++) for (int f = 0; f < melSpec.Shape[1]; f++) input[idx++] = melSpec[t, f];
            output = Predict(input);
        }
        else { throw new InvalidOperationException("No model available for classification. Provide an ONNX model path or use native training mode."); }
        var scores = new T[ClassLabels.Count];
        for (int i = 0; i < Math.Min(output.Length, scores.Length); i++) { double l = NumOps.ToDouble(output[i]); scores[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-l))); }
        return scores;
    }

    private List<Tensor<T>> SplitIntoWindows(Tensor<T> audio)
    {
        var windows = new List<Tensor<T>>();
        int ws = (int)(_options.WindowSize * _options.SampleRate), hs = (int)(ws * (1 - _options.WindowOverlap));
        if (hs <= 0) hs = 1; int lastStart = 0;
        for (int s = 0; s + ws <= audio.Length; s += hs) { var w = new Tensor<T>([ws]); for (int i = 0; i < ws; i++) w[i] = audio[s + i]; windows.Add(w); lastStart = s + hs; }
        int rs = windows.Count > 0 ? lastStart : 0, rem = audio.Length - rs;
        if (rem > ws / 10) { var w = new Tensor<T>([ws]); for (int i = 0; i < rem && i < ws; i++) w[i] = audio[rs + i]; windows.Add(w); }
        else if (windows.Count == 0 && audio.Length > 0) { var w = new Tensor<T>([ws]); for (int i = 0; i < audio.Length; i++) w[i] = audio[i]; windows.Add(w); }
        return windows;
    }

    private List<AudioEvent<T>> MergeEvents(List<AudioEvent<T>> events)
    {
        if (events.Count == 0) return events;
        var merged = new List<AudioEvent<T>>();
        foreach (var g in events.GroupBy(e => e.EventType))
        {
            var sorted = g.OrderBy(e => e.StartTime).ToList(); var cur = sorted[0];
            for (int i = 1; i < sorted.Count; i++)
            {
                var next = sorted[i];
                if (next.StartTime <= cur.EndTime + 0.1)
                {
                    double cc = NumOps.ToDouble(cur.Confidence), nc = NumOps.ToDouble(next.Confidence);
                    cur = new AudioEvent<T> { EventType = cur.EventType, StartTime = cur.StartTime, EndTime = Math.Max(cur.EndTime, next.EndTime), Confidence = cc > nc ? cur.Confidence : next.Confidence, PeakTime = cc > nc ? cur.PeakTime : next.PeakTime };
                }
                else { merged.Add(cur); cur = next; }
            }
            merged.Add(cur);
        }
        return merged.OrderBy(e => e.StartTime).ToList();
    }

    private Dictionary<string, EventStatistics<T>> ComputeEventStatistics(IReadOnlyList<AudioEvent<T>> events)
    {
        var s = new Dictionary<string, EventStatistics<T>>();
        foreach (var g in events.GroupBy(e => e.EventType))
        {
            var l = g.ToList();
            s[g.Key] = new EventStatistics<T> { Count = l.Count, TotalDuration = l.Sum(e => e.Duration), AverageConfidence = NumOps.FromDouble(l.Average(e => NumOps.ToDouble(e.Confidence))), MaxConfidence = NumOps.FromDouble(l.Max(e => NumOps.ToDouble(e.Confidence))) };
        }
        return s;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(EAT<T>)); }
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

        public void FeedAudio(Tensor<T> chunk)
        {
            if (_disp) throw new ObjectDisposedException(nameof(EATStreamingSession));
            List<AudioEvent<T>>? raise = null;
            lock (_lk)
            {
                if (_disp) throw new ObjectDisposedException(nameof(EATStreamingSession));
                for (int i = 0; i < chunk.Length; i++) _buf.Add(chunk[i]);
                while (_buf.Count >= _ws)
                {
                    var w = new Tensor<T>([_ws]); for (int i = 0; i < _ws; i++) w[i] = _buf[i];
                    var mel = _det._melSpectrogram?.Forward(w) ?? throw new InvalidOperationException("MelSpectrogram not initialized.");
                    var scores = _det.ClassifyWindow(mel); double tv = _det.NumOps.ToDouble(_thr);
                    for (int i = 0; i < scores.Length && i < _det.ClassLabels.Count; i++)
                    {
                        _state[_det.ClassLabels[i]] = scores[i];
                        if (_det.NumOps.ToDouble(scores[i]) >= tv)
                        {
                            var e = new AudioEvent<T> { EventType = _det.ClassLabels[i], Confidence = scores[i], StartTime = _pt, EndTime = _pt + _det._options.WindowSize, PeakTime = _pt + _det._options.WindowSize / 2 };
                            _new.Add(e); raise ??= []; raise.Add(e);
                        }
                    }
                    int hs = (int)(_ws * (1 - _det._options.WindowOverlap)); if (hs <= 0) hs = 1;
                    _buf.RemoveRange(0, hs); _pt += hs / (double)_det._options.SampleRate;
                }
            }
            if (raise is not null) foreach (var e in raise) EventDetected?.Invoke(this, e);
        }

        public IReadOnlyList<AudioEvent<T>> GetNewEvents() { lock (_lk) { var e = _new.ToList(); _new.Clear(); return e; } }
        public IReadOnlyDictionary<string, T> GetCurrentState() { lock (_lk) { return new Dictionary<string, T>(_state); } }
        public void Dispose() { if (_disp) return; lock (_lk) { if (_disp) return; _disp = true; _buf.Clear(); _new.Clear(); _state.Clear(); } }
    }

    #endregion
}
