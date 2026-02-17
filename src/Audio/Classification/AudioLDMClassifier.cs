using AiDotNet.Audio.Classification;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// AudioLDM Classifier that repurposes AudioLDM's latent representations for audio event detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// AudioLDM Classifier (Liu et al., 2023) extracts intermediate features from the AudioLDM
/// diffusion U-Net to build a strong audio classifier. Since AudioLDM was trained to generate
/// audio from text, its internal representations capture rich, semantically meaningful audio
/// features that transfer well to classification tasks.
/// </para>
/// <para>
/// <b>For Beginners:</b> AudioLDM was originally built to create audio from text descriptions.
/// But it turns out the "understanding" it developed during training is also great for
/// recognizing sounds. This classifier reuses that understanding: instead of generating audio,
/// it identifies what sounds are present in a recording.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 64, outputSize: 527);
/// var model = new AudioLDMClassifier&lt;float&gt;(arch, "audioldm_classifier.onnx");
/// var events = model.Detect(audio);
/// foreach (var e in events) Console.WriteLine($"{e.Label}: {e.Confidence}");
/// </code>
/// </para>
/// </remarks>
public class AudioLDMClassifier<T> : AudioClassifierBase<T>, IAudioEventDetector<T>
{
    #region Fields

    private readonly AudioLDMClassifierOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    /// <summary>Standard AudioSet labels for audio event detection.</summary>
    public static IReadOnlyList<string> StandardLabels => BEATs<T>.AudioSetLabels;

    #endregion

    #region Constructors

    /// <summary>Creates an AudioLDM Classifier in ONNX inference mode.</summary>
    public AudioLDMClassifier(NeuralNetworkArchitecture<T> architecture, string modelPath, AudioLDMClassifierOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new AudioLDMClassifierOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        ClassLabels = _options.CustomLabels ?? BEATs<T>.AudioSetLabels;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <summary>Creates an AudioLDM Classifier in native training mode.</summary>
    public AudioLDMClassifier(NeuralNetworkArchitecture<T> architecture, AudioLDMClassifierOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new AudioLDMClassifierOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        ClassLabels = _options.CustomLabels ?? BEATs<T>.AudioSetLabels;
        InitializeLayers();
    }

    internal static async Task<AudioLDMClassifier<T>> CreateAsync(AudioLDMClassifierOptions? options = null,
        IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new AudioLDMClassifierOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("audioldm_classifier", "audioldm_classifier.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        int numLabels = options.CustomLabels is not null ? options.CustomLabels.Length : BEATs<T>.AudioSetLabels.Count();
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumMels, outputSize: numLabels);
        return new AudioLDMClassifier<T>(arch, mp, options);
    }

    #endregion

    #region IAudioEventDetector Properties

    /// <inheritdoc />
    public IReadOnlyList<string> SupportedEvents => ClassLabels;

    /// <inheritdoc />
    public double TimeResolution => _options.DetectionWindowSize;

    /// <inheritdoc />
    public new bool IsOnnxMode => !_useNativeMode && OnnxEncoder is not null;

    #endregion

    #region IAudioEventDetector Methods

    /// <inheritdoc />
    public AudioEventResult<T> Detect(Tensor<T> audio)
        => Detect(audio, NumOps.FromDouble(_options.Threshold));

    /// <inheritdoc />
    public AudioEventResult<T> Detect(Tensor<T> audio, T threshold)
    {
        ThrowIfDisposed();
        double thresholdValue = NumOps.ToDouble(threshold);
        var windows = SplitIntoWindows(audio);
        var allEvents = new List<AudioEvent<T>>();
        foreach (var (windowData, startSample) in windows)
        {
            var probs = ClassifyWindow(windowData);
            double startTime = (double)startSample / _options.SampleRate;
            double endTime = startTime + _options.DetectionWindowSize;
            for (int i = 0; i < probs.Length && i < ClassLabels.Count; i++)
            {
                double p = NumOps.ToDouble(probs[i]);
                if (p >= thresholdValue)
                    allEvents.Add(new AudioEvent<T>
                    {
                        EventType = ClassLabels[i], Confidence = probs[i],
                        StartTime = startTime, EndTime = endTime, PeakTime = (startTime + endTime) / 2.0
                    });
            }
        }
        var merged = MergeEvents(allEvents);
        double totalDuration = (double)audio.Length / _options.SampleRate;
        var eventStats = ComputeEventStatistics(merged);
        return new AudioEventResult<T>
        {
            Events = merged, TotalDuration = totalDuration,
            DetectedEventTypes = merged.Select(e => e.EventType).Distinct().ToArray(),
            EventStats = eventStats
        };
    }

    /// <inheritdoc />
    public Task<AudioEventResult<T>> DetectAsync(Tensor<T> audio, CancellationToken cancellationToken = default)
        => Task.Run(() => Detect(audio), cancellationToken);

    /// <inheritdoc />
    public AudioEventResult<T> DetectSpecific(Tensor<T> audio, IReadOnlyList<string> eventTypes)
        => DetectSpecific(audio, eventTypes, NumOps.FromDouble(_options.Threshold));

    /// <inheritdoc />
    public AudioEventResult<T> DetectSpecific(Tensor<T> audio, IReadOnlyList<string> eventTypes, T threshold)
    {
        var result = Detect(audio, threshold);
        var eventTypeSet = new HashSet<string>(eventTypes, StringComparer.OrdinalIgnoreCase);
        var filtered = result.Events.Where(e => eventTypeSet.Contains(e.EventType)).ToList();
        return new AudioEventResult<T>
        {
            Events = filtered, TotalDuration = result.TotalDuration,
            DetectedEventTypes = filtered.Select(e => e.EventType).Distinct().ToArray(),
            EventStats = ComputeEventStatistics(filtered)
        };
    }

    /// <inheritdoc />
    public Tensor<T> GetEventProbabilities(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var windows = SplitIntoWindows(audio);
        var probabilities = new Tensor<T>([windows.Count, ClassLabels.Count]);
        for (int w = 0; w < windows.Count; w++)
        {
            var probs = ClassifyWindow(windows[w].Data);
            for (int i = 0; i < probs.Length && i < ClassLabels.Count; i++)
                probabilities[w, i] = probs[i];
        }
        return probabilities;
    }

    /// <inheritdoc />
    public IStreamingEventDetectionSession<T> StartStreamingSession()
        => StartStreamingSession(_options.SampleRate, NumOps.FromDouble(_options.Threshold));

    /// <inheritdoc />
    public IStreamingEventDetectionSession<T> StartStreamingSession(int sampleRate, T threshold)
        => new AudioLDMClassifierStreamingSession(this, sampleRate, threshold);

    #endregion

    #region NeuralNetworkBase Implementation

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultAudioLDMClassifierLayers(
            numMels: _options.NumMels, latentDim: _options.LatentDim,
            classifierDim: _options.ClassifierDim, numClassifierLayers: _options.NumClassifierLayers,
            numClasses: ClassLabels.Count, dropoutRate: _options.DropoutRate));
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        var c = input; foreach (var l in Layers) c = l.Forward(c); return c;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        var output = Predict(input);
        var grad = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());
        var gt = Tensor<T>.FromVector(grad);
        for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt);
        _optimizer.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("ONNX mode.");
        int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; }
    }

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (MelSpec is not null) return MelSpec.Forward(rawAudio);
        return rawAudio;
    }

    protected override Tensor<T> PostprocessOutput(Tensor<T> o)
    {
        // Apply softmax returning tensor (base ApplySoftmax returns Dictionary)
        T maxVal = o[0];
        for (int i = 1; i < o.Length; i++)
            if (NumOps.GreaterThan(o[i], maxVal)) maxVal = o[i];
        var result = new Tensor<T>(o.Shape);
        T sum = NumOps.Zero;
        for (int i = 0; i < o.Length; i++)
        {
            result[i] = NumOps.Exp(NumOps.Subtract(o[i], maxVal));
            sum = NumOps.Add(sum, result[i]);
        }
        for (int i = 0; i < o.Length; i++)
            result[i] = NumOps.Divide(result[i], sum);
        return result;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "AudioLDMClassifier-Native" : "AudioLDMClassifier-ONNX",
            Description = "AudioLDM Classifier - diffusion features for audio classification (Liu et al., 2023)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumMels,
            Complexity = _options.NumClassifierLayers
        };
        m.AdditionalInfo["LatentDim"] = _options.LatentDim.ToString();
        m.AdditionalInfo["ClassifierDim"] = _options.ClassifierDim.ToString();
        m.AdditionalInfo["NumClasses"] = ClassLabels.Count.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.NumMels); w.Write(_options.FftSize); w.Write(_options.HopLength);
        w.Write(_options.LatentDim); w.Write(_options.ClassifierDim); w.Write(_options.NumClassifierLayers);
        w.Write(_options.Threshold); w.Write(_options.DetectionWindowSize); w.Write(_options.WindowOverlap);
        w.Write(_options.DropoutRate);
        w.Write(ClassLabels.Count);
        foreach (var label in ClassLabels) w.Write(label);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.NumMels = r.ReadInt32(); _options.FftSize = r.ReadInt32(); _options.HopLength = r.ReadInt32();
        _options.LatentDim = r.ReadInt32(); _options.ClassifierDim = r.ReadInt32(); _options.NumClassifierLayers = r.ReadInt32();
        _options.Threshold = r.ReadDouble(); _options.DetectionWindowSize = r.ReadDouble(); _options.WindowOverlap = r.ReadDouble();
        _options.DropoutRate = r.ReadDouble();
        int numLabels = r.ReadInt32();
        var labels = new string[numLabels]; for (int i = 0; i < numLabels; i++) labels[i] = r.ReadString();
        ClassLabels = labels;
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new AudioLDMClassifier<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private Tensor<T> ClassifyWindow(Tensor<T> windowAudio)
    {
        var features = PreprocessAudio(windowAudio);
        var output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);
        return PostprocessOutput(output);
    }

    private List<(Tensor<T> Data, int StartSample)> SplitIntoWindows(Tensor<T> audio)
    {
        var windows = new List<(Tensor<T>, int)>();
        int windowSamples = (int)(_options.DetectionWindowSize * _options.SampleRate);
        int hopSamples = (int)(windowSamples * (1.0 - _options.WindowOverlap));
        if (hopSamples <= 0) hopSamples = windowSamples;
        for (int start = 0; start < audio.Length; start += hopSamples)
        {
            int end = Math.Min(start + windowSamples, audio.Length);
            var window = new Tensor<T>([windowSamples]);
            for (int i = 0; i < end - start; i++) window[i] = audio[start + i];
            windows.Add((window, start));
        }
        return windows;
    }

    private List<AudioEvent<T>> MergeEvents(List<AudioEvent<T>> events)
    {
        if (events.Count == 0) return events;
        var grouped = events.GroupBy(e => e.EventType);
        var merged = new List<AudioEvent<T>>();
        foreach (var group in grouped)
        {
            var sorted = group.OrderBy(e => e.StartTime).ToList();
            var current = sorted[0];
            for (int i = 1; i < sorted.Count; i++)
            {
                if (sorted[i].StartTime <= current.EndTime)
                {
                    double currentConf = NumOps.ToDouble(current.Confidence);
                    double nextConf = NumOps.ToDouble(sorted[i].Confidence);
                    current = new AudioEvent<T>
                    {
                        EventType = current.EventType,
                        StartTime = current.StartTime,
                        EndTime = Math.Max(current.EndTime, sorted[i].EndTime),
                        Confidence = nextConf > currentConf ? sorted[i].Confidence : current.Confidence,
                        PeakTime = nextConf > currentConf ? sorted[i].PeakTime : current.PeakTime
                    };
                }
                else { merged.Add(current); current = sorted[i]; }
            }
            merged.Add(current);
        }
        return merged.OrderBy(e => e.StartTime).ToList();
    }

    private Dictionary<string, EventStatistics<T>> ComputeEventStatistics(IReadOnlyList<AudioEvent<T>> events)
    {
        var stats = new Dictionary<string, EventStatistics<T>>();
        foreach (var group in events.GroupBy(e => e.EventType))
        {
            var list = group.ToList();
            T maxConf = NumOps.MinValue;
            foreach (var e in list)
                if (NumOps.GreaterThan(e.Confidence, maxConf)) maxConf = e.Confidence;
            stats[group.Key] = new EventStatistics<T>
            {
                Count = list.Count,
                TotalDuration = list.Sum(e => e.Duration),
                AverageConfidence = NumOps.FromDouble(list.Average(e => NumOps.ToDouble(e.Confidence))),
                MaxConfidence = maxConf
            };
        }
        return stats;
    }

    #endregion

    #region Streaming Session

    private sealed class AudioLDMClassifierStreamingSession : IStreamingEventDetectionSession<T>
    {
        private readonly AudioLDMClassifier<T> _model;
        private readonly int _sampleRate;
        private readonly T _threshold;
        private readonly List<T> _buffer = [];
        private readonly List<AudioEvent<T>> _newEvents = [];
        private readonly Dictionary<string, T> _currentState = new();
        private readonly int _windowSamples;
        private double _processedTime;
        private bool _disposed;

        public event EventHandler<AudioEvent<T>>? EventDetected;

        public AudioLDMClassifierStreamingSession(AudioLDMClassifier<T> model, int sampleRate, T threshold)
        {
            _model = model; _sampleRate = sampleRate; _threshold = threshold;
            _windowSamples = (int)(_model._options.DetectionWindowSize * sampleRate);
        }

        public void FeedAudio(Tensor<T> audioChunk)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(AudioLDMClassifierStreamingSession));
            for (int i = 0; i < audioChunk.Length; i++) _buffer.Add(audioChunk[i]);
            double thresholdValue = _model.NumOps.ToDouble(_threshold);
            while (_buffer.Count >= _windowSamples)
            {
                var frame = new Tensor<T>([_windowSamples]);
                for (int i = 0; i < _windowSamples; i++) frame[i] = _buffer[i];
                _buffer.RemoveRange(0, _windowSamples / 2);
                var probs = _model.ClassifyWindow(frame);
                double startTime = _processedTime;
                double endTime = startTime + _model._options.DetectionWindowSize;
                for (int i = 0; i < probs.Length && i < _model.ClassLabels.Count; i++)
                {
                    double p = _model.NumOps.ToDouble(probs[i]);
                    _currentState[_model.ClassLabels[i]] = probs[i];
                    if (p >= thresholdValue)
                    {
                        var ev = new AudioEvent<T>
                        {
                            EventType = _model.ClassLabels[i], Confidence = probs[i],
                            StartTime = startTime, EndTime = endTime,
                            PeakTime = (startTime + endTime) / 2.0
                        };
                        _newEvents.Add(ev);
                        EventDetected?.Invoke(this, ev);
                    }
                }
                _processedTime += _model._options.DetectionWindowSize / 2.0;
            }
        }

        public IReadOnlyList<AudioEvent<T>> GetNewEvents()
        {
            var events = new List<AudioEvent<T>>(_newEvents);
            _newEvents.Clear();
            return events;
        }

        public IReadOnlyDictionary<string, T> GetCurrentState() => _currentState;

        public void Dispose()
        {
            if (_disposed) return;
            _buffer.Clear(); _newEvents.Clear();
            _disposed = true;
        }
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(AudioLDMClassifier<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
