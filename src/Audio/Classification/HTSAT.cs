using AiDotNet.Audio.Features;
using AiDotNet.Diffusion;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// HTS-AT (Hierarchical Token-Semantic Audio Transformer) model for efficient audio classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// HTS-AT (Chen et al., ICASSP 2022) is a hierarchical Transformer architecture that uses
/// Swin Transformer blocks with a token-semantic module for efficient audio classification.
/// It achieves 47.1% mAP on AudioSet-2M with only 30M parameters, making it more parameter-efficient
/// than models like AST (87M parameters) while achieving higher accuracy.
/// </para>
/// <para>
/// <b>Architecture:</b> HTS-AT processes audio spectrograms hierarchically through four stages:
/// <list type="number">
/// <item><b>Patch embedding</b>: 4x4 patch embedding of the mel spectrogram</item>
/// <item><b>Stage 1</b>: 2 Swin blocks at 96-dim with local window attention</item>
/// <item><b>Stage 2</b>: 2 Swin blocks at 192-dim (patch merging doubles channels, halves resolution)</item>
/// <item><b>Stage 3</b>: 6 Swin blocks at 384-dim (the deepest processing stage)</item>
/// <item><b>Stage 4</b>: 2 Swin blocks at 768-dim</item>
/// <item><b>Token-semantic module</b>: Groups tokens by semantic meaning for global context</item>
/// <item><b>Classification head</b>: Projects to class logits with sigmoid for multi-label</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> HTS-AT is like reading a book at different zoom levels. First it
/// reads individual words (small patches), then sentences (merged patches), then paragraphs
/// (further merged), and finally understands the whole story. At each level, it uses "window
/// attention" - looking at nearby patches first, which is much faster than looking at everything.
///
/// <b>Usage with ONNX (recommended):</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 96, outputSize: 527);
/// var htsat = new HTSAT&lt;float&gt;(arch, "htsat_audioset.onnx");
/// var result = htsat.Detect(audioTensor);
/// </code>
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "HTS-AT: A Hierarchical Token-Semantic Audio Transformer" (Chen et al., ICASSP 2022)</item>
/// <item>Repository: https://github.com/RetroCirce/HTS-Audio-Transformer</item>
/// </list>
/// </para>
/// </remarks>
public class HTSAT<T> : AudioClassifierBase<T>, IAudioEventDetector<T>
{
    #region Fields

    private readonly HTSATOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private MelSpectrogram<T>? _melSpectrogram;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    /// <summary>
    /// AudioSet-527 standard event labels.
    /// </summary>
    public static readonly string[] AudioSetLabels = BEATs<T>.AudioSetLabels;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates an HTS-AT model for ONNX inference mode.
    /// </summary>
    public HTSAT(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        HTSATOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new HTSATOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        base.NumMels = _options.NumMels;

        _melSpectrogram = new MelSpectrogram<T>(
            sampleRate: _options.SampleRate,
            nMels: _options.NumMels,
            nFft: _options.FftSize,
            hopLength: _options.HopLength,
            fMin: _options.FMin,
            fMax: _options.FMax,
            logMel: true);

        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        ClassLabels = _options.CustomLabels ?? AudioSetLabels;
        InitializeLayers();
    }

    /// <summary>
    /// Creates an HTS-AT model for native training mode.
    /// </summary>
    public HTSAT(
        NeuralNetworkArchitecture<T> architecture,
        HTSATOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new HTSATOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        base.NumMels = _options.NumMels;
        ClassLabels = _options.CustomLabels ?? AudioSetLabels;

        _melSpectrogram = new MelSpectrogram<T>(
            sampleRate: _options.SampleRate,
            nMels: _options.NumMels,
            nFft: _options.FftSize,
            hopLength: _options.HopLength,
            fMin: _options.FMin,
            fMax: _options.FMax,
            logMel: true);

        InitializeLayers();
    }

    /// <summary>
    /// Creates an HTS-AT model asynchronously with optional model download.
    /// </summary>
    internal static async Task<HTSAT<T>> CreateAsync(
        HTSATOptions? options = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new HTSATOptions();
        string modelPath = options.ModelPath ?? string.Empty;

        if (string.IsNullOrEmpty(modelPath))
        {
            var downloader = new OnnxModelDownloader();
            modelPath = await downloader.DownloadAsync("htsat", "htsat_audioset.onnx",
                progress: progress, cancellationToken);
            options.ModelPath = modelPath;
        }

        var architecture = new NeuralNetworkArchitecture<T>(
            inputFeatures: options.EmbeddingDim,
            outputSize: (options.CustomLabels ?? AudioSetLabels).Length);

        return new HTSAT<T>(architecture, modelPath, options);
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
    {
        if (sampleRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(sampleRate), "Sample rate must be positive.");
        if (sampleRate != _options.SampleRate)
            throw new ArgumentException($"Sample rate {sampleRate} does not match model's configured sample rate {_options.SampleRate}. Resample audio before streaming.", nameof(sampleRate));
        return new HTSATStreamingSession(this, sampleRate, threshold);
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultHTSATLayers(
                patchFeatureSize: _options.PatchSize * _options.PatchSize,
                embeddingDim: _options.EmbeddingDim,
                numStages: _options.NumLayersPerStage.Length,
                numLayersPerStage: _options.NumLayersPerStage,
                numClasses: ClassLabels.Count,
                maxSequenceLength: 1024,
                dropoutRate: _options.DropoutRate));
        }
    }

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        var current = input;
        foreach (var layer in Layers) current = layer.Forward(current);
        return current;
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        var output = Predict(input);
        var gradient = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());
        var gradientTensor = Tensor<T>.FromVector(gradient);
        for (int i = Layers.Count - 1; i >= 0; i--) gradientTensor = Layers[i].Backward(gradientTensor);
        _optimizer?.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("UpdateParameters is not supported in ONNX mode.");
        int index = 0;
        foreach (var layer in Layers)
        {
            int count = layer.ParameterCount;
            layer.UpdateParameters(parameters.Slice(index, count));
            index += count;
        }
    }

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (_melSpectrogram is null) throw new InvalidOperationException("MelSpectrogram not initialized.");
        return _melSpectrogram.Forward(rawAudio);
    }

    /// <inheritdoc/>
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

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "HTS-AT-Native" : "HTS-AT-ONNX",
            Description = "HTS-AT: Hierarchical Token-Semantic Audio Transformer (Chen et al., ICASSP 2022)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = ClassLabels.Count,
            Complexity = _options.NumLayersPerStage.Sum()
        };
        metadata.AdditionalInfo["Architecture"] = "HTS-AT";
        metadata.AdditionalInfo["EmbeddingDim"] = _options.EmbeddingDim.ToString();
        metadata.AdditionalInfo["NumStages"] = _options.NumLayersPerStage.Length.ToString();
        metadata.AdditionalInfo["WindowSize"] = _options.WindowSize.ToString();
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        metadata.AdditionalInfo["NumMels"] = _options.NumMels.ToString();
        metadata.AdditionalInfo["NumClasses"] = ClassLabels.Count.ToString();
        return metadata;
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write(_options.SampleRate); writer.Write(_options.NumMels);
        writer.Write(_options.FftSize); writer.Write(_options.HopLength);
        writer.Write(_options.EmbeddingDim); writer.Write(_options.WindowSize);
        writer.Write(_options.PatchSize); writer.Write(_options.Threshold);
        writer.Write(_options.DetectionWindowSize); writer.Write(_options.WindowOverlap);
        writer.Write(_options.DropoutRate);
        writer.Write((int)_options.FMin); writer.Write((int)_options.FMax);
        writer.Write(ClassLabels.Count);
        foreach (var label in ClassLabels) writer.Write(label);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string modelPath = reader.ReadString();
        if (!string.IsNullOrEmpty(modelPath)) _options.ModelPath = modelPath;
        _options.SampleRate = reader.ReadInt32(); _options.NumMels = reader.ReadInt32();
        _options.FftSize = reader.ReadInt32(); _options.HopLength = reader.ReadInt32();
        _options.EmbeddingDim = reader.ReadInt32(); _options.WindowSize = reader.ReadInt32();
        _options.PatchSize = reader.ReadInt32(); _options.Threshold = reader.ReadDouble();
        _options.DetectionWindowSize = reader.ReadDouble(); _options.WindowOverlap = reader.ReadDouble();
        _options.DropoutRate = reader.ReadDouble();
        _options.FMin = reader.ReadInt32(); _options.FMax = reader.ReadInt32();
        int numLabels = reader.ReadInt32();
        var labels = new string[numLabels];
        for (int i = 0; i < numLabels; i++) labels[i] = reader.ReadString();
        ClassLabels = labels;

        _melSpectrogram = new MelSpectrogram<T>(
            sampleRate: _options.SampleRate, nMels: _options.NumMels,
            nFft: _options.FftSize, hopLength: _options.HopLength,
            fMin: _options.FMin, fMax: _options.FMax, logMel: true);

        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new HTSAT<T>(Architecture, mp, _options);
        return new HTSAT<T>(Architecture, _options);
    }

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

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(HTSAT<T>)); }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion

    #region Streaming Session

    private sealed class HTSATStreamingSession : IStreamingEventDetectionSession<T>
    {
        private readonly HTSAT<T> _detector;
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

        public HTSATStreamingSession(HTSAT<T> detector, int sampleRate, T threshold)
        {
            _detector = detector; _sampleRate = sampleRate; _threshold = threshold;
            _buffer = []; _newEvents = []; _currentState = new Dictionary<string, T>();
            _windowSamples = (int)(detector._options.DetectionWindowSize * sampleRate);
            foreach (var label in detector.ClassLabels) _currentState[label] = detector.NumOps.Zero;
        }

        public void FeedAudio(Tensor<T> audioChunk)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(HTSATStreamingSession));
            List<AudioEvent<T>>? eventsToRaise = null;
            lock (_lock)
            {
                if (_disposed) throw new ObjectDisposedException(nameof(HTSATStreamingSession));
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
