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
/// CRNN (Convolutional Recurrent Neural Network) model for Sound Event Detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CRNN for SED (Cakir et al., 2017) is the standard baseline architecture for the DCASE
/// Sound Event Detection challenge. It combines CNN layers for spectral feature extraction
/// with bidirectional GRU/LSTM layers for temporal modeling, producing frame-level event
/// probabilities. Despite its simplicity compared to Transformer-based models, CRNN remains
/// competitive and is widely used as a strong baseline.
/// </para>
/// <para>
/// <b>Architecture:</b>
/// <list type="number">
/// <item><b>CNN blocks</b>: 3 convolutional blocks (64/128/256 channels) with batch normalization,
/// ReLU activation, and frequency-axis pooling. Extracts local spectro-temporal features.</item>
/// <item><b>RNN layers</b>: 2 bidirectional GRU layers (128 hidden units each) that model
/// temporal dependencies across frames. Bidirectional processing captures both past and future context.</item>
/// <item><b>Classification head</b>: Frame-level linear projection with sigmoid activation
/// for multi-label detection (multiple events can occur simultaneously).</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> CRNN is like a two-stage sound detective:
///
/// Stage 1 (CNN): Looks at short moments of the spectrogram and identifies spectral patterns.
/// "This frequency pattern looks like a door slam" or "These harmonics look like speech."
///
/// Stage 2 (RNN): Reads those patterns over time like a story.
/// "The door slam pattern appeared briefly at 2.3 seconds and ended at 2.5 seconds."
///
/// Together, they can tell you WHAT sounds happened and WHEN they happened.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 256, outputSize: 527);
/// var model = new CRNNEventDetector&lt;float&gt;(arch, "crnn_sed.onnx");
/// var result = model.Detect(audioTensor);
/// </code>
/// </para>
/// </remarks>
public class CRNNEventDetector<T> : AudioClassifierBase<T>, IAudioEventDetector<T>
{
    #region Fields

    private readonly CRNNEventDetectorOptions _options;
    public override ModelOptions GetOptions() => _options;
    private MelSpectrogram<T>? _melSpectrogram;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    /// <summary>AudioSet-527 standard event labels.</summary>
    public static readonly string[] AudioSetLabels = BEATs<T>.AudioSetLabels;

    #endregion

    #region Constructors

    /// <summary>Creates a CRNN SED model for ONNX inference mode.</summary>
    public CRNNEventDetector(NeuralNetworkArchitecture<T> architecture, string modelPath, CRNNEventDetectorOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new CRNNEventDetectorOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        base.NumMels = _options.NumMels;
        _melSpectrogram = new MelSpectrogram<T>(sampleRate: _options.SampleRate, nMels: _options.NumMels,
            nFft: _options.FftSize, hopLength: _options.HopLength, fMin: _options.FMin, fMax: _options.FMax, logMel: true);
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        ClassLabels = _options.CustomLabels ?? AudioSetLabels;
        InitializeLayers();
    }

    /// <summary>Creates a CRNN SED model for native training mode.</summary>
    public CRNNEventDetector(NeuralNetworkArchitecture<T> architecture, CRNNEventDetectorOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new CRNNEventDetectorOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        base.NumMels = _options.NumMels;
        ClassLabels = _options.CustomLabels ?? AudioSetLabels;
        _melSpectrogram = new MelSpectrogram<T>(sampleRate: _options.SampleRate, nMels: _options.NumMels,
            nFft: _options.FftSize, hopLength: _options.HopLength, fMin: _options.FMin, fMax: _options.FMax, logMel: true);
        InitializeLayers();
    }

    internal static async Task<CRNNEventDetector<T>> CreateAsync(CRNNEventDetectorOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new CRNNEventDetectorOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("crnn_sed", $"crnn_sed_{options.Variant}.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        int outputSize = (options.CustomLabels ?? AudioSetLabels).Length;
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.CNNChannels[^1], outputSize: outputSize);
        return new CRNNEventDetector<T>(arch, mp, options);
    }

    #endregion

    #region IAudioEventDetector Properties

    /// <inheritdoc/>
    public IReadOnlyList<string> SupportedEvents => ClassLabels;
    /// <inheritdoc/>
    public IReadOnlyList<string> EventLabels => ClassLabels;
    /// <inheritdoc/>
    public double TimeResolution => _options.DetectionWindowSize * (1 - _options.WindowOverlap);

    #endregion

    #region IAudioEventDetector Methods

    /// <inheritdoc/>
    public AudioEventResult<T> Detect(Tensor<T> audio)
    {
        ThrowIfDisposed();
        return Detect(audio, NumOps.FromDouble(_options.Threshold));
    }

    /// <inheritdoc/>
    public AudioEventResult<T> Detect(Tensor<T> audio, T threshold)
    {
        ThrowIfDisposed();
        double thresholdValue = NumOps.ToDouble(threshold);
        double totalDuration = audio.Length / (double)_options.SampleRate;
        var windows = SplitIntoWindows(audio);
        var allEvents = new List<AudioEvent<T>>();

        for (int windowIdx = 0; windowIdx < windows.Count; windowIdx++)
        {
            double startTime = windowIdx * TimeResolution;
            var melSpec = _melSpectrogram?.Forward(windows[windowIdx]) ??
                throw new InvalidOperationException("MelSpectrogram not initialized.");
            var scores = ClassifyWindow(melSpec);

            for (int i = 0; i < scores.Length && i < ClassLabels.Count; i++)
            {
                double score = NumOps.ToDouble(scores[i]);
                if (score >= thresholdValue)
                {
                    allEvents.Add(new AudioEvent<T>
                    {
                        EventType = ClassLabels[i], Confidence = scores[i],
                        StartTime = startTime,
                        EndTime = Math.Min(startTime + _options.DetectionWindowSize, totalDuration),
                        PeakTime = startTime + _options.DetectionWindowSize / 2
                    });
                }
            }
        }

        var mergedEvents = MergeEvents(allEvents);
        var eventStats = ComputeEventStatistics(mergedEvents);

        return new AudioEventResult<T>
        {
            Events = mergedEvents, TotalDuration = totalDuration,
            DetectedEventTypes = mergedEvents.Select(e => e.EventType).Distinct().ToList(),
            EventStats = eventStats
        };
    }

    /// <inheritdoc/>
    public Task<AudioEventResult<T>> DetectAsync(Tensor<T> audio, CancellationToken cancellationToken = default)
        => Task.Run(() => Detect(audio), cancellationToken);

    /// <inheritdoc/>
    public AudioEventResult<T> DetectSpecific(Tensor<T> audio, IReadOnlyList<string> eventTypes)
        => DetectSpecific(audio, eventTypes, NumOps.FromDouble(_options.Threshold));

    /// <inheritdoc/>
    public AudioEventResult<T> DetectSpecific(Tensor<T> audio, IReadOnlyList<string> eventTypes, T threshold)
    {
        var result = Detect(audio, threshold);
        var eventTypeSet = new HashSet<string>(eventTypes, StringComparer.OrdinalIgnoreCase);
        var filteredEvents = result.Events.Where(e => eventTypeSet.Contains(e.EventType)).ToList();
        return new AudioEventResult<T>
        {
            Events = filteredEvents, TotalDuration = result.TotalDuration,
            DetectedEventTypes = filteredEvents.Select(e => e.EventType).Distinct().ToList(),
            EventStats = ComputeEventStatistics(filteredEvents)
        };
    }

    /// <inheritdoc/>
    public Tensor<T> GetEventProbabilities(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var windows = SplitIntoWindows(audio);
        var probabilities = new Tensor<T>([windows.Count, ClassLabels.Count]);
        for (int w = 0; w < windows.Count; w++)
        {
            var melSpec = _melSpectrogram?.Forward(windows[w]) ??
                throw new InvalidOperationException("MelSpectrogram not initialized.");
            var scores = ClassifyWindow(melSpec);
            for (int i = 0; i < ClassLabels.Count && i < scores.Length; i++)
                probabilities[w, i] = scores[i];
        }
        return probabilities;
    }

    /// <inheritdoc/>
    public IStreamingEventDetectionSession<T> StartStreamingSession()
        => StartStreamingSession(_options.SampleRate, NumOps.FromDouble(_options.Threshold));

    /// <inheritdoc/>
    public IStreamingEventDetectionSession<T> StartStreamingSession(int sampleRate, T threshold)
        => new CRNNStreamingSession(this, sampleRate, threshold);

    #endregion

    #region NeuralNetworkBase Implementation

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultCRNNEventDetectorLayers(
            cnnChannels: _options.CNNChannels, rnnHiddenSize: _options.RNNHiddenSize,
            numRNNLayers: _options.NumRNNLayers, numClasses: ClassLabels.Count,
            numMels: _options.NumMels, dropoutRate: _options.DropoutRate));
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        var current = input; foreach (var layer in Layers) current = layer.Forward(current); return current;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        var output = Predict(input);
        var gradient = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());
        var gt = Tensor<T>.FromVector(gradient);
        for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt);
        _optimizer?.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("UpdateParameters is not supported in ONNX mode.");
        int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; }
    }

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (_melSpectrogram is null) throw new InvalidOperationException("MelSpectrogram not initialized.");
        return _melSpectrogram.Forward(rawAudio);
    }

    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        var result = new Tensor<T>(modelOutput.Shape);
        for (int i = 0; i < modelOutput.Length; i++)
        {
            double logit = NumOps.ToDouble(modelOutput[i]);
            result[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-logit)));
        }
        return result;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "CRNNEventDetector-Native" : "CRNNEventDetector-ONNX",
            Description = $"CRNN for Sound Event Detection {_options.Variant} (Cakir et al., 2017)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = ClassLabels.Count,
            Complexity = _options.CNNChannels.Length + _options.NumRNNLayers
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["CNNChannels"] = string.Join(",", _options.CNNChannels);
        m.AdditionalInfo["RNNHiddenSize"] = _options.RNNHiddenSize.ToString();
        m.AdditionalInfo["NumClasses"] = ClassLabels.Count.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.NumMels);
        w.Write(_options.FftSize); w.Write(_options.HopLength);
        w.Write(_options.CNNChannels.Length);
        foreach (int ch in _options.CNNChannels) w.Write(ch);
        w.Write(_options.RNNHiddenSize); w.Write(_options.NumRNNLayers);
        w.Write(_options.Threshold); w.Write(_options.DetectionWindowSize);
        w.Write(_options.WindowOverlap); w.Write(_options.DropoutRate);
        w.Write(ClassLabels.Count);
        foreach (var label in ClassLabels) w.Write(label);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.NumMels = r.ReadInt32();
        _options.FftSize = r.ReadInt32(); _options.HopLength = r.ReadInt32();
        int nch = r.ReadInt32(); _options.CNNChannels = new int[nch];
        for (int i = 0; i < nch; i++) _options.CNNChannels[i] = r.ReadInt32();
        _options.RNNHiddenSize = r.ReadInt32(); _options.NumRNNLayers = r.ReadInt32();
        _options.Threshold = r.ReadDouble(); _options.DetectionWindowSize = r.ReadDouble();
        _options.WindowOverlap = r.ReadDouble(); _options.DropoutRate = r.ReadDouble();
        int numLabels = r.ReadInt32(); var labels = new string[numLabels];
        for (int i = 0; i < numLabels; i++) labels[i] = r.ReadString();
        ClassLabels = labels;
        _melSpectrogram = new MelSpectrogram<T>(sampleRate: _options.SampleRate, nMels: _options.NumMels,
            nFft: _options.FftSize, hopLength: _options.HopLength, fMin: _options.FMin, fMax: _options.FMax, logMel: true);
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new CRNNEventDetector<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private T[] ClassifyWindow(Tensor<T> melSpec)
    {
        Tensor<T> output;
        if (IsOnnxMode && OnnxEncoder is not null)
        {
            var input = new Tensor<T>([1, 1, melSpec.Shape[0], melSpec.Shape[1]]);
            for (int t = 0; t < melSpec.Shape[0]; t++)
                for (int f = 0; f < melSpec.Shape[1]; f++)
                    input[0, 0, t, f] = melSpec[t, f];
            output = OnnxEncoder.Run(input);
        }
        else if (_useNativeMode)
        {
            var input = new Tensor<T>([melSpec.Length]);
            int idx = 0;
            for (int t = 0; t < melSpec.Shape[0]; t++)
                for (int f = 0; f < melSpec.Shape[1]; f++)
                    input[idx++] = melSpec[t, f];
            output = Predict(input);
        }
        else
        {
            throw new InvalidOperationException(
                "No model available for classification. Provide an ONNX model path or use native training mode.");
        }

        var scores = new T[ClassLabels.Count];
        for (int i = 0; i < Math.Min(output.Length, scores.Length); i++)
        {
            double logit = NumOps.ToDouble(output[i]);
            scores[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-logit)));
        }
        return scores;
    }

    private List<Tensor<T>> SplitIntoWindows(Tensor<T> audio)
    {
        var windows = new List<Tensor<T>>();
        int windowSamples = (int)(_options.DetectionWindowSize * _options.SampleRate);
        int hopSamples = (int)(windowSamples * (1 - _options.WindowOverlap));
        if (hopSamples <= 0) hopSamples = 1;

        int lastStart = 0;
        for (int start = 0; start + windowSamples <= audio.Length; start += hopSamples)
        {
            var window = new Tensor<T>([windowSamples]);
            for (int i = 0; i < windowSamples; i++) window[i] = audio[start + i];
            windows.Add(window);
            lastStart = start + hopSamples;
        }

        int remainingStart = windows.Count > 0 ? lastStart : 0;
        int remainingSamples = audio.Length - remainingStart;
        if (remainingSamples > windowSamples / 10)
        {
            var window = new Tensor<T>([windowSamples]);
            for (int i = 0; i < remainingSamples && i < windowSamples; i++) window[i] = audio[remainingStart + i];
            windows.Add(window);
        }
        else if (windows.Count == 0 && audio.Length > 0)
        {
            var window = new Tensor<T>([windowSamples]);
            for (int i = 0; i < audio.Length; i++) window[i] = audio[i];
            windows.Add(window);
        }
        return windows;
    }

    private List<AudioEvent<T>> MergeEvents(List<AudioEvent<T>> events)
    {
        if (events.Count == 0) return events;
        var merged = new List<AudioEvent<T>>();
        foreach (var group in events.GroupBy(e => e.EventType))
        {
            var sorted = group.OrderBy(e => e.StartTime).ToList();
            var current = sorted[0];
            for (int i = 1; i < sorted.Count; i++)
            {
                var next = sorted[i];
                if (next.StartTime <= current.EndTime + 0.1)
                {
                    double cc = NumOps.ToDouble(current.Confidence), nc = NumOps.ToDouble(next.Confidence);
                    current = new AudioEvent<T>
                    {
                        EventType = current.EventType, StartTime = current.StartTime,
                        EndTime = Math.Max(current.EndTime, next.EndTime),
                        Confidence = cc > nc ? current.Confidence : next.Confidence,
                        PeakTime = cc > nc ? current.PeakTime : next.PeakTime
                    };
                }
                else { merged.Add(current); current = next; }
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
            stats[group.Key] = new EventStatistics<T>
            {
                Count = list.Count, TotalDuration = list.Sum(e => e.Duration),
                AverageConfidence = NumOps.FromDouble(list.Average(e => NumOps.ToDouble(e.Confidence))),
                MaxConfidence = NumOps.FromDouble(list.Max(e => NumOps.ToDouble(e.Confidence)))
            };
        }
        return stats;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(CRNNEventDetector<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion

    #region Streaming Session

    private sealed class CRNNStreamingSession : IStreamingEventDetectionSession<T>
    {
        private readonly CRNNEventDetector<T> _detector;
        private readonly int _sampleRate;
        private readonly T _threshold;
        private readonly List<T> _buffer;
        private readonly List<AudioEvent<T>> _newEvents;
        private readonly Dictionary<string, T> _currentState;
        private readonly int _windowSamples;
        private double _processedTime;
        private volatile bool _disposed;
        private readonly object _lock = new object();
        public event EventHandler<AudioEvent<T>>? EventDetected;

        public CRNNStreamingSession(CRNNEventDetector<T> detector, int sampleRate, T threshold)
        {
            _detector = detector; _sampleRate = sampleRate; _threshold = threshold;
            _buffer = []; _newEvents = []; _currentState = new Dictionary<string, T>();
            _windowSamples = (int)(detector._options.DetectionWindowSize * sampleRate);
            foreach (var label in detector.ClassLabels) _currentState[label] = detector.NumOps.Zero;
        }

        public void FeedAudio(Tensor<T> audioChunk)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(CRNNStreamingSession));
            List<AudioEvent<T>>? eventsToRaise = null;
            lock (_lock)
            {
                if (_disposed) throw new ObjectDisposedException(nameof(CRNNStreamingSession));
                for (int i = 0; i < audioChunk.Length; i++) _buffer.Add(audioChunk[i]);
                while (_buffer.Count >= _windowSamples)
                {
                    var window = new Tensor<T>([_windowSamples]);
                    for (int i = 0; i < _windowSamples; i++) window[i] = _buffer[i];
                    var melSpec = _detector._melSpectrogram?.Forward(window) ??
                        throw new InvalidOperationException("MelSpectrogram not initialized.");
                    var scores = _detector.ClassifyWindow(melSpec);
                    double tv = _detector.NumOps.ToDouble(_threshold);
                    for (int i = 0; i < scores.Length && i < _detector.ClassLabels.Count; i++)
                    {
                        _currentState[_detector.ClassLabels[i]] = scores[i];
                        if (_detector.NumOps.ToDouble(scores[i]) >= tv)
                        {
                            var evt = new AudioEvent<T>
                            {
                                EventType = _detector.ClassLabels[i], Confidence = scores[i],
                                StartTime = _processedTime, EndTime = _processedTime + _detector._options.DetectionWindowSize,
                                PeakTime = _processedTime + _detector._options.DetectionWindowSize / 2
                            };
                            _newEvents.Add(evt);
                            eventsToRaise ??= new List<AudioEvent<T>>();
                            eventsToRaise.Add(evt);
                        }
                    }
                    int hopSamples = (int)(_windowSamples * (1 - _detector._options.WindowOverlap));
                    if (hopSamples <= 0) hopSamples = 1;
                    _buffer.RemoveRange(0, hopSamples);
                    _processedTime += hopSamples / (double)_sampleRate;
                }
            }
            if (eventsToRaise is not null) foreach (var evt in eventsToRaise) EventDetected?.Invoke(this, evt);
        }

        public IReadOnlyList<AudioEvent<T>> GetNewEvents()
        {
            lock (_lock) { var events = _newEvents.ToList(); _newEvents.Clear(); return events; }
        }

        public IReadOnlyDictionary<string, T> GetCurrentState()
        {
            lock (_lock) { return new Dictionary<string, T>(_currentState); }
        }

        public void Dispose()
        {
            if (_disposed) return;
            lock (_lock) { if (_disposed) return; _disposed = true; _buffer.Clear(); _newEvents.Clear(); _currentState.Clear(); }
        }
    }

    #endregion
}
