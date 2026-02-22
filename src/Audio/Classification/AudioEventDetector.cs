using AiDotNet.Audio.Features;
using AiDotNet.Diffusion;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// Audio event detection model for identifying sounds in audio (AudioSet-style).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Detects various audio events like speech, music, environmental sounds, and more.
/// Based on AudioSet ontology with 527+ event classes organized hierarchically.
/// </para>
/// <para>
/// <b>Architecture:</b> This model extends <see cref="AudioClassifierBase{T}"/>
/// and implements <see cref="IAudioEventDetector{T}"/> for multi-label event detection.
/// Unlike single-label classification, event detection identifies multiple overlapping
/// events with their temporal boundaries.
/// </para>
/// <para><b>For Beginners:</b> Audio event detection answers "What sounds are in this audio?":
/// <list type="bullet">
/// <item>Human sounds: speech, laughter, coughing, footsteps</item>
/// <item>Animal sounds: dog barking, bird singing, cat meowing</item>
/// <item>Music: instruments, genres, singing</item>
/// <item>Environmental: traffic, rain, wind, construction</item>
/// </list>
///
/// Usage with ONNX:
/// <code>
/// var options = new AudioEventDetectorOptions { ModelPath = "audio-events.onnx" };
/// var detector = new AudioEventDetector&lt;float&gt;(options);
/// var result = detector.Detect(audio);
/// foreach (var evt in result.Events)
/// {
///     Console.WriteLine($"{evt.EventType}: {evt.Confidence} at {evt.StartTime:F2}s");
/// }
/// </code>
///
/// Usage with training:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 64, outputSize: 50);
/// var detector = new AudioEventDetector&lt;float&gt;(architecture, new AudioEventDetectorOptions());
/// detector.Train(features, labels);
/// </code>
/// </para>
/// </remarks>
public class AudioEventDetector<T> : AudioClassifierBase<T>, IAudioEventDetector<T>
{
    #region Fields

    private readonly AudioEventDetectorOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private MelSpectrogram<T>? _melSpectrogram;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    /// <summary>
    /// Common audio event categories from AudioSet.
    /// </summary>
    public static readonly string[] CommonEventLabels =
    [
        // Human sounds
        "Speech", "Male speech", "Female speech", "Child speech",
        "Conversation", "Narration", "Laughter", "Crying", "Cough",
        "Sneeze", "Breathing", "Sigh", "Yawn", "Snoring",

        // Animal sounds
        "Dog", "Dog barking", "Dog howl", "Cat", "Cat meowing",
        "Bird", "Bird song", "Chirp", "Crow", "Rooster",

        // Music
        "Music", "Singing", "Guitar", "Piano", "Drums",
        "Bass guitar", "Electric guitar", "Violin", "Flute",

        // Environmental
        "Rain", "Thunder", "Wind", "Water", "Fire",
        "Traffic", "Car", "Car horn", "Siren", "Train",
        "Airplane", "Helicopter", "Engine",

        // Household
        "Door", "Knock", "Bell", "Telephone", "Alarm",
        "Keyboard typing", "Mouse click", "Printer",

        // Other
        "Silence", "Noise", "Static", "White noise", "Pink noise"
    ];

    #endregion

    #region Constructors

    /// <summary>
    /// Creates an AudioEventDetector for ONNX inference mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture.</param>
    /// <param name="modelPath">Path to ONNX model file.</param>
    /// <param name="options">Detection options.</param>
    public AudioEventDetector(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        AudioEventDetectorOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new AudioEventDetectorOptions();
        _useNativeMode = false;

        base.SampleRate = _options.SampleRate;

        // Initialize mel spectrogram for feature extraction
        _melSpectrogram = new MelSpectrogram<T>(
            sampleRate: _options.SampleRate,
            nMels: _options.NumMels,
            nFft: _options.FftSize,
            hopLength: _options.HopLength,
            fMin: _options.FMin,
            fMax: _options.FMax,
            logMel: true);

        // Load ONNX model
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);

        // Initialize class labels
        ClassLabels = _options.CustomLabels ?? CommonEventLabels;

        _options.ModelPath = modelPath;

        InitializeLayers();
    }

    /// <summary>
    /// Creates an AudioEventDetector for native training mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="options">Detection options.</param>
    /// <param name="optimizer">Optional custom optimizer (defaults to AdamW).</param>
    public AudioEventDetector(
        NeuralNetworkArchitecture<T> architecture,
        AudioEventDetectorOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new AudioEventDetectorOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);

        base.SampleRate = _options.SampleRate;

        // Initialize class labels
        ClassLabels = _options.CustomLabels ?? CommonEventLabels;

        // Initialize mel spectrogram
        _melSpectrogram = new MelSpectrogram<T>(
            sampleRate: _options.SampleRate,
            nMels: _options.NumMels,
            nFft: _options.FftSize,
            hopLength: _options.HopLength,
            fMin: _options.FMin,
            fMax: _options.FMax,
            logMel: true);

        // Create feature extractors

        InitializeLayers();
    }

    /// <summary>
    /// Creates an AudioEventDetector with legacy options only (native mode).
    /// </summary>
    /// <param name="options">Detection options.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a native mode detector. For ONNX inference mode,
    /// use <see cref="CreateAsync"/> or the constructor that accepts a model path.
    /// </para>
    /// <para>
    /// Example for ONNX mode:
    /// <code>
    /// var detector = await AudioEventDetector&lt;float&gt;.CreateAsync(options);
    /// </code>
    /// </para>
    /// </remarks>
    public AudioEventDetector(AudioEventDetectorOptions? options = null)
        : this(
            new NeuralNetworkArchitecture<T>(
                inputFeatures: (options ?? new AudioEventDetectorOptions()).NumMels,
                outputSize: (options?.CustomLabels ?? CommonEventLabels).Length),
            options)
    {
    }

    private MfccExtractor<T> CreateMfccExtractor()
    {
        return new MfccExtractor<T>(new MfccOptions
        {
            SampleRate = _options.SampleRate,
            NumCoefficients = 13,
            FftSize = _options.FftSize,
            HopLength = _options.HopLength,
            FMin = _options.FMin,
            FMax = _options.FMax
        });
    }

    /// <summary>
    /// Creates an AudioEventDetector asynchronously with model download.
    /// </summary>
    internal static async Task<AudioEventDetector<T>> CreateAsync(
        AudioEventDetectorOptions? options = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new AudioEventDetectorOptions();
        string modelPath = options.ModelPath ?? string.Empty;

        if (string.IsNullOrEmpty(modelPath))
        {
            var downloader = new OnnxModelDownloader();
            modelPath = await downloader.DownloadAsync(
                "audio-event-detector",
                "model.onnx",
                progress: progress,
                cancellationToken);
            options.ModelPath = modelPath;
        }

        var architecture = new NeuralNetworkArchitecture<T>(
            inputFeatures: options.NumMels,
            outputSize: (options.CustomLabels ?? CommonEventLabels).Length);

        return new AudioEventDetector<T>(architecture, modelPath, options);
    }

    #endregion

    #region IAudioEventDetector Properties

    /// <summary>
    /// Gets the list of event types this model can detect.
    /// </summary>
    public IReadOnlyList<string> SupportedEvents => ClassLabels;

    /// <summary>
    /// Gets the event labels (alias for SupportedEvents for legacy API compatibility).
    /// </summary>
    public IReadOnlyList<string> EventLabels => ClassLabels;

    /// <summary>
    /// Gets the time resolution for event detection in seconds.
    /// </summary>
    public double TimeResolution => _options.WindowSize * (1 - _options.WindowOverlap);

    #endregion

    #region IAudioEventDetector Methods

    /// <summary>
    /// Detects audio events in the audio stream.
    /// </summary>
    public AudioEventResult<T> Detect(Tensor<T> audio)
    {
        ThrowIfDisposed();
        return Detect(audio, NumOps.FromDouble(_options.Threshold));
    }

    /// <summary>
    /// Detects audio events with custom threshold.
    /// </summary>
    public AudioEventResult<T> Detect(Tensor<T> audio, T threshold)
    {
        ThrowIfDisposed();

        double thresholdValue = NumOps.ToDouble(threshold);
        double totalDuration = audio.Length / (double)_options.SampleRate;

        // Split audio into windows
        var windows = SplitIntoWindows(audio);
        var allEvents = new List<AudioEvent<T>>();

        for (int windowIdx = 0; windowIdx < windows.Count; windowIdx++)
        {
            var window = windows[windowIdx];
            double startTime = windowIdx * TimeResolution;

            // Extract features
            var melSpec = _melSpectrogram?.Forward(window) ??
                throw new InvalidOperationException("MelSpectrogram not initialized.");

            // Classify
            var scores = ClassifyWindow(melSpec, window);

            // Get events above threshold
            for (int i = 0; i < scores.Length && i < ClassLabels.Count; i++)
            {
                double score = NumOps.ToDouble(scores[i]);
                if (score >= thresholdValue)
                {
                    allEvents.Add(new AudioEvent<T>
                    {
                        EventType = ClassLabels[i],
                        Confidence = scores[i],
                        StartTime = startTime,
                        EndTime = Math.Min(startTime + _options.WindowSize, totalDuration),
                        PeakTime = startTime + _options.WindowSize / 2
                    });
                }
            }
        }

        // Merge overlapping events of same class
        var mergedEvents = MergeEvents(allEvents);

        // Compute statistics
        var eventStats = ComputeEventStatistics(mergedEvents);

        return new AudioEventResult<T>
        {
            Events = mergedEvents,
            TotalDuration = totalDuration,
            DetectedEventTypes = mergedEvents.Select(e => e.EventType).Distinct().ToList(),
            EventStats = eventStats
        };
    }

    /// <summary>
    /// Detects audio events asynchronously.
    /// </summary>
    public Task<AudioEventResult<T>> DetectAsync(
        Tensor<T> audio,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Detect(audio), cancellationToken);
    }

    /// <summary>
    /// Detects specific events only.
    /// </summary>
    public AudioEventResult<T> DetectSpecific(Tensor<T> audio, IReadOnlyList<string> eventTypes)
    {
        return DetectSpecific(audio, eventTypes, NumOps.FromDouble(_options.Threshold));
    }

    /// <summary>
    /// Detects specific events only with custom threshold.
    /// </summary>
    public AudioEventResult<T> DetectSpecific(Tensor<T> audio, IReadOnlyList<string> eventTypes, T threshold)
    {
        var result = Detect(audio, threshold);

        // Filter to only requested event types
        var eventTypeSet = new HashSet<string>(eventTypes, StringComparer.OrdinalIgnoreCase);
        var filteredEvents = result.Events.Where(e => eventTypeSet.Contains(e.EventType)).ToList();

        return new AudioEventResult<T>
        {
            Events = filteredEvents,
            TotalDuration = result.TotalDuration,
            DetectedEventTypes = filteredEvents.Select(e => e.EventType).Distinct().ToList(),
            EventStats = ComputeEventStatistics(filteredEvents)
        };
    }

    /// <summary>
    /// Gets frame-level event probabilities.
    /// </summary>
    public Tensor<T> GetEventProbabilities(Tensor<T> audio)
    {
        ThrowIfDisposed();

        var windows = SplitIntoWindows(audio);
        var probabilities = new Tensor<T>([windows.Count, ClassLabels.Count]);

        for (int windowIdx = 0; windowIdx < windows.Count; windowIdx++)
        {
            var window = windows[windowIdx];
            var melSpec = _melSpectrogram?.Forward(window) ??
                throw new InvalidOperationException("MelSpectrogram not initialized.");

            var scores = ClassifyWindow(melSpec, window);

            for (int i = 0; i < ClassLabels.Count && i < scores.Length; i++)
            {
                probabilities[windowIdx, i] = scores[i];
            }
        }

        return probabilities;
    }

    /// <summary>
    /// Starts a streaming event detection session.
    /// </summary>
    public IStreamingEventDetectionSession<T> StartStreamingSession()
    {
        return StartStreamingSession(_options.SampleRate, NumOps.FromDouble(_options.Threshold));
    }

    /// <summary>
    /// Starts a streaming event detection session with custom settings.
    /// </summary>
    public IStreamingEventDetectionSession<T> StartStreamingSession(int sampleRate, T threshold)
    {
        return new StreamingEventDetectionSession(this, sampleRate, threshold);
    }

    #endregion

    #region Legacy API Support

    /// <summary>
    /// Detects audio events in the given audio (legacy API).
    /// </summary>
    /// <param name="audio">Audio waveform.</param>
    /// <returns>List of detected audio events.</returns>
    public List<AudioEvent> DetectLegacy(Tensor<T> audio)
    {
        var result = Detect(audio);

        return result.Events.Select(e => new AudioEvent
        {
            Label = e.EventType,
            Confidence = NumOps.ToDouble(e.Confidence),
            StartTime = e.StartTime,
            EndTime = e.EndTime
        }).ToList();
    }

    /// <summary>
    /// Detects audio events asynchronously (legacy API).
    /// </summary>
    public async Task<List<AudioEvent>> DetectLegacyAsync(
        Tensor<T> audio,
        CancellationToken cancellationToken = default)
    {
        var result = await DetectAsync(audio, cancellationToken);

        return result.Events.Select(e => new AudioEvent
        {
            Label = e.EventType,
            Confidence = NumOps.ToDouble(e.Confidence),
            StartTime = e.StartTime,
            EndTime = e.EndTime
        }).ToList();
    }

    /// <summary>
    /// Detects a single frame (no windowing) - legacy API.
    /// </summary>
    public Dictionary<string, double> DetectFrame(Tensor<T> audio)
    {
        ThrowIfDisposed();

        var melSpec = _melSpectrogram?.Forward(audio) ??
            throw new InvalidOperationException("MelSpectrogram not initialized.");

        var scores = ClassifyWindow(melSpec, audio);

        var result = new Dictionary<string, double>();
        for (int i = 0; i < scores.Length && i < ClassLabels.Count; i++)
        {
            result[ClassLabels[i]] = NumOps.ToDouble(scores[i]);
        }

        return result;
    }

    /// <summary>
    /// Gets the top K events for a single frame (legacy API).
    /// </summary>
    public List<(string Label, double Confidence)> DetectTopK(Tensor<T> audio, int topK = 5)
    {
        var allScores = DetectFrame(audio);
        return allScores
            .OrderByDescending(x => x.Value)
            .Take(topK)
            .Select(x => (x.Key, x.Value))
            .ToList();
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <summary>
    /// Initializes the neural network layers.
    /// </summary>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            // Use same architecture as genre classifier for event detection
            Layers.AddRange(LayerHelper<T>.CreateDefaultGenreClassifierLayers(
                numMels: _options.NumMels,
                numClasses: ClassLabels.Count,
                hiddenDim: 256,
                dropoutRate: 0.3));
        }
    }

    /// <summary>
    /// Predicts output for the given input.
    /// </summary>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();

        if (IsOnnxMode && OnnxEncoder is not null)
        {
            return OnnxEncoder.Run(input);
        }

        // Native mode: run through layers
        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }
        return current;
    }

    /// <summary>
    /// Trains the model on a single example.
    /// </summary>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
        {
            throw new NotSupportedException(
                "Training is not supported in ONNX mode. Create a new AudioEventDetector " +
                "without modelPath parameter to train natively.");
        }

        // Set training mode
        SetTrainingMode(true);

        // Forward pass
        var output = Predict(input);

        // Compute loss and gradients
        var loss = LossFunction.CalculateLoss(output.ToVector(), expected.ToVector());
        var gradient = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());

        // Backward pass
        var gradientTensor = Tensor<T>.FromVector(gradient);

        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradientTensor = Layers[i].Backward(gradientTensor);
        }

        // Update parameters using optimizer
        _optimizer?.UpdateParameters(Layers);

        // Set inference mode
        SetTrainingMode(false);
    }

    /// <summary>
    /// Updates network parameters from a flattened vector.
    /// </summary>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("UpdateParameters is not supported in ONNX mode.");

        int index = 0;
        foreach (var layer in Layers)
        {
            int count = layer.ParameterCount;
            var layerParams = parameters.Slice(index, count);
            layer.UpdateParameters(layerParams);
            index += count;
        }
    }

    /// <summary>
    /// Preprocesses audio for the model.
    /// </summary>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        // Extract mel spectrogram features
        if (_melSpectrogram is null)
            throw new InvalidOperationException("MelSpectrogram not initialized.");

        return _melSpectrogram.Forward(rawAudio);
    }

    /// <summary>
    /// Post-processes model output.
    /// </summary>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        // Apply sigmoid for multi-label classification
        var result = new Tensor<T>(modelOutput.Shape);
        for (int i = 0; i < modelOutput.Length; i++)
        {
            double logit = NumOps.ToDouble(modelOutput[i]);
            double sigmoid = 1.0 / (1.0 + Math.Exp(-logit));
            result[i] = NumOps.FromDouble(sigmoid);
        }
        return result;
    }

    /// <summary>
    /// Gets model metadata.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "AudioEventDetector-Native" : "AudioEventDetector-ONNX",
            Description = "Audio event detection model for identifying sounds (AudioSet-style)",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = ClassLabels.Count,
            Complexity = 1
        };
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        metadata.AdditionalInfo["NumMels"] = _options.NumMels.ToString();
        metadata.AdditionalInfo["NumEvents"] = ClassLabels.Count.ToString();
        metadata.AdditionalInfo["WindowSize"] = _options.WindowSize.ToString("F2");
        metadata.AdditionalInfo["TimeResolution"] = TimeResolution.ToString("F3");
        return metadata;
    }

    /// <summary>
    /// Serializes network-specific data.
    /// </summary>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Write ONNX mode state
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);

        // Write options
        writer.Write(_options.SampleRate);
        writer.Write(_options.NumMels);
        writer.Write(_options.FftSize);
        writer.Write(_options.HopLength);
        writer.Write(_options.WindowSize);
        writer.Write(_options.Threshold);
        writer.Write(ClassLabels.Count);
        foreach (var label in ClassLabels)
        {
            writer.Write(label);
        }
    }

    /// <summary>
    /// Deserializes network-specific data.
    /// </summary>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Restore ONNX mode state
        _useNativeMode = reader.ReadBoolean();
        string modelPath = reader.ReadString();
        if (!string.IsNullOrEmpty(modelPath))
        {
            _options.ModelPath = modelPath;
        }

        // Restore options properties
        _options.SampleRate = reader.ReadInt32();
        _options.NumMels = reader.ReadInt32();
        _options.FftSize = reader.ReadInt32();
        _options.HopLength = reader.ReadInt32();
        _options.WindowSize = reader.ReadDouble();
        _options.Threshold = reader.ReadDouble();

        // Read class labels
        int numLabels = reader.ReadInt32();
        var labels = new string[numLabels];
        for (int i = 0; i < numLabels; i++)
        {
            labels[i] = reader.ReadString();
        }
        ClassLabels = labels;

        // Reinitialize mel spectrogram with deserialized options
        _melSpectrogram = new MelSpectrogram<T>(
            sampleRate: _options.SampleRate,
            nMels: _options.NumMels,
            nFft: _options.FftSize,
            hopLength: _options.HopLength,
            fMin: _options.FMin,
            fMax: _options.FMax,
            logMel: true);

        // Restore ONNX model if in ONNX inference mode
        if (!_useNativeMode && _options.ModelPath is { } onnxModelPath && !string.IsNullOrEmpty(onnxModelPath))
        {
            OnnxEncoder = new OnnxModel<T>(onnxModelPath, _options.OnnxOptions);
        }
    }

    /// <summary>
    /// Creates a new instance for deserialization.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new AudioEventDetector<T>(Architecture, _options);
    }

    #endregion

    #region Helper Methods

    private List<Tensor<T>> SplitIntoWindows(Tensor<T> audio)
    {
        var windows = new List<Tensor<T>>();
        int windowSamples = (int)(_options.WindowSize * _options.SampleRate);
        int hopSamples = (int)(windowSamples * (1 - _options.WindowOverlap));

        // Ensure we don't divide by zero
        if (hopSamples <= 0) hopSamples = 1;

        int lastStart = 0;
        for (int start = 0; start + windowSamples <= audio.Length; start += hopSamples)
        {
            var window = new Tensor<T>([windowSamples]);
            for (int i = 0; i < windowSamples; i++)
            {
                window[i] = audio[start + i];
            }
            windows.Add(window);
            lastStart = start + hopSamples;
        }

        // Handle last partial window - include remaining samples that weren't captured
        int remainingStart = windows.Count > 0 ? lastStart : 0;
        int remainingSamples = audio.Length - remainingStart;

        // Add partial window if there's significant remaining audio (at least 10% of window size)
        if (remainingSamples > windowSamples / 10)
        {
            // Pad partial window to full window size for consistent processing
            var window = new Tensor<T>([windowSamples]);
            for (int i = 0; i < remainingSamples && i < windowSamples; i++)
            {
                window[i] = audio[remainingStart + i];
            }
            // Remaining values stay at default (zero-padded)
            windows.Add(window);
        }
        else if (windows.Count == 0 && audio.Length > 0)
        {
            // No full windows and not enough for partial - still process what we have
            var window = new Tensor<T>([windowSamples]);
            for (int i = 0; i < audio.Length; i++)
            {
                window[i] = audio[i];
            }
            windows.Add(window);
        }

        return windows;
    }

    private T[] ClassifyWindow(Tensor<T> melSpec, Tensor<T> audio)
    {
        if (IsOnnxMode && OnnxEncoder is not null)
        {
            return ClassifyWithModel(melSpec);
        }
        else if (_useNativeMode)
        {
            return ClassifyWithNative(melSpec);
        }
        else
        {
            // Fallback to rule-based classification when neither ONNX nor native mode is available
            // This can occur after deserialization if the ONNX model file is missing
            return ClassifyWithRules(melSpec, audio);
        }
    }

    private T[] ClassifyWithModel(Tensor<T> melSpec)
    {
        if (OnnxEncoder is null)
            throw new InvalidOperationException("Model not loaded.");

        // Prepare input (add batch and channel dimensions)
        var input = new Tensor<T>([1, 1, melSpec.Shape[0], melSpec.Shape[1]]);
        for (int t = 0; t < melSpec.Shape[0]; t++)
        {
            for (int f = 0; f < melSpec.Shape[1]; f++)
            {
                input[0, 0, t, f] = melSpec[t, f];
            }
        }

        var output = OnnxEncoder.Run(input);

        // Apply sigmoid to get probabilities (multi-label)
        var scores = new T[ClassLabels.Count];
        for (int i = 0; i < Math.Min(output.Length, scores.Length); i++)
        {
            double logit = NumOps.ToDouble(output[i]);
            scores[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-logit)));
        }

        return scores;
    }

    private T[] ClassifyWithNative(Tensor<T> melSpec)
    {
        // Flatten mel spectrogram for neural network
        var input = new Tensor<T>([melSpec.Length]);
        int idx = 0;
        for (int t = 0; t < melSpec.Shape[0]; t++)
        {
            for (int f = 0; f < melSpec.Shape[1]; f++)
            {
                input[idx++] = melSpec[t, f];
            }
        }

        // Run through network
        var output = Predict(input);

        // Apply sigmoid for multi-label
        var scores = new T[ClassLabels.Count];
        for (int i = 0; i < Math.Min(output.Length, scores.Length); i++)
        {
            double logit = NumOps.ToDouble(output[i]);
            scores[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-logit)));
        }

        return scores;
    }

    /// <summary>
    /// Rule-based fallback classification when neither ONNX nor native neural network mode is available.
    /// </summary>
    /// <remarks>
    /// This method is invoked when:
    /// - ONNX mode is not active (no OnnxEncoder loaded or IsOnnxMode is false)
    /// - Native neural network mode is not active (_useNativeMode is false)
    ///
    /// This typically occurs during:
    /// - Deserialization when the original ONNX model file is missing or inaccessible
    /// - Recovery scenarios where model loading failed but detection should continue
    ///
    /// Note: Normal constructor paths do not reach this method because:
    /// - ONNX constructor sets _useNativeMode=false and loads OnnxEncoder (takes ONNX path)
    /// - Native/legacy constructors set _useNativeMode=true (takes native path)
    ///
    /// The method uses spectral features (energy, zero-crossing rate, spectral centroid,
    /// spectral flatness, band energies) to classify audio events using predefined heuristics.
    /// While less accurate than neural network classification, it provides a functional
    /// fallback that doesn't require external model files.
    /// </remarks>
    private T[] ClassifyWithRules(Tensor<T> melSpec, Tensor<T> audio)
    {
        var scores = new T[ClassLabels.Count];

        // Compute basic features
        double energy = ComputeEnergy(audio);
        double zcr = ComputeZeroCrossingRate(audio);
        double spectralCentroid = ComputeSpectralCentroid(melSpec);
        double spectralFlatness = ComputeSpectralFlatness(melSpec);
        double lowFreqEnergy = ComputeBandEnergy(melSpec, 0, 10);
        double highFreqEnergy = ComputeBandEnergy(melSpec, melSpec.Shape[1] - 20, melSpec.Shape[1]);

        for (int i = 0; i < ClassLabels.Count; i++)
        {
            double score = ComputeEventScore(ClassLabels[i], energy, zcr, spectralCentroid,
                spectralFlatness, lowFreqEnergy, highFreqEnergy);
            scores[i] = NumOps.FromDouble(score);
        }

        return scores;
    }

    private double ComputeEnergy(Tensor<T> audio)
    {
        double sum = 0;
        for (int i = 0; i < audio.Length; i++)
        {
            double val = NumOps.ToDouble(audio[i]);
            sum += val * val;
        }
        return sum / audio.Length;
    }

    private double ComputeZeroCrossingRate(Tensor<T> audio)
    {
        int crossings = 0;
        for (int i = 1; i < audio.Length; i++)
        {
            double prev = NumOps.ToDouble(audio[i - 1]);
            double curr = NumOps.ToDouble(audio[i]);
            if ((prev >= 0 && curr < 0) || (prev < 0 && curr >= 0))
                crossings++;
        }
        return (double)crossings / audio.Length;
    }

    private double ComputeSpectralCentroid(Tensor<T> melSpec)
    {
        double weightedSum = 0;
        double totalSum = 0;

        for (int t = 0; t < melSpec.Shape[0]; t++)
        {
            for (int f = 0; f < melSpec.Shape[1]; f++)
            {
                double mag = NumOps.ToDouble(melSpec[t, f]);
                weightedSum += f * mag;
                totalSum += mag;
            }
        }

        return totalSum > 0 ? weightedSum / totalSum : 0;
    }

    private double ComputeSpectralFlatness(Tensor<T> melSpec)
    {
        double logSum = 0;
        double sum = 0;
        int count = 0;

        for (int t = 0; t < melSpec.Shape[0]; t++)
        {
            for (int f = 0; f < melSpec.Shape[1]; f++)
            {
                double mag = Math.Max(NumOps.ToDouble(melSpec[t, f]), 1e-10);
                logSum += Math.Log(mag);
                sum += mag;
                count++;
            }
        }

        if (count == 0) return 0;

        double geometricMean = Math.Exp(logSum / count);
        double arithmeticMean = sum / count;

        return arithmeticMean > 0 ? geometricMean / arithmeticMean : 0;
    }

    private double ComputeBandEnergy(Tensor<T> melSpec, int startBin, int endBin)
    {
        double sum = 0;
        int count = 0;

        for (int t = 0; t < melSpec.Shape[0]; t++)
        {
            for (int f = startBin; f < endBin && f < melSpec.Shape[1]; f++)
            {
                sum += NumOps.ToDouble(melSpec[t, f]);
                count++;
            }
        }

        return count > 0 ? sum / count : 0;
    }

    private static double ComputeEventScore(
        string label,
        double energy,
        double zcr,
        double centroid,
        double flatness,
        double lowEnergy,
        double highEnergy)
    {
        double score = 0;
        string lowerLabel = label.ToLowerInvariant();

        // Silence detection
        if (lowerLabel == "silence")
        {
            return energy < 0.0001 ? 0.9 : 0.1;
        }

        // Noise detection
        if (lowerLabel.Contains("noise"))
        {
            return flatness > 0.8 ? 0.7 : 0.2;
        }

        // Speech detection (moderate ZCR, moderate centroid)
        if (lowerLabel.Contains("speech") || lowerLabel.Contains("conversation") ||
            lowerLabel.Contains("narration"))
        {
            if (energy > 0.001 && zcr > 0.02 && zcr < 0.15 && centroid > 30 && centroid < 80)
            {
                score = 0.6;
            }
        }

        // Music detection (wider spectrum, energy patterns)
        if (lowerLabel.Contains("music") || lowerLabel.Contains("singing") ||
            lowerLabel.Contains("guitar") || lowerLabel.Contains("piano"))
        {
            if (energy > 0.005 && flatness < 0.5)
            {
                score = 0.5;
            }
        }

        // Animal sounds
        if (lowerLabel.Contains("dog") || lowerLabel.Contains("bark"))
        {
            if (energy > 0.01 && zcr > 0.05 && zcr < 0.2)
            {
                score = 0.4;
            }
        }

        if (lowerLabel.Contains("bird") || lowerLabel.Contains("chirp"))
        {
            if (highEnergy > lowEnergy * 2 && zcr > 0.1)
            {
                score = 0.4;
            }
        }

        // Environmental
        if (lowerLabel.Contains("rain") || lowerLabel.Contains("water"))
        {
            if (flatness > 0.6 && energy > 0.001)
            {
                score = 0.4;
            }
        }

        if (lowerLabel.Contains("traffic") || lowerLabel.Contains("car") ||
            lowerLabel.Contains("engine"))
        {
            if (lowEnergy > highEnergy && energy > 0.005)
            {
                score = 0.4;
            }
        }

        return Math.Max(score, 0.05); // Minimum baseline
    }

    private List<AudioEvent<T>> MergeEvents(List<AudioEvent<T>> events)
    {
        if (events.Count == 0) return events;

        // Group by event type
        var grouped = events.GroupBy(e => e.EventType);
        var merged = new List<AudioEvent<T>>();

        foreach (var group in grouped)
        {
            var sortedEvents = group.OrderBy(e => e.StartTime).ToList();
            var currentEvent = sortedEvents[0];

            for (int i = 1; i < sortedEvents.Count; i++)
            {
                var next = sortedEvents[i];

                // Check if events overlap or are adjacent
                if (next.StartTime <= currentEvent.EndTime + 0.1)
                {
                    // Merge - keep higher confidence
                    double currentConf = NumOps.ToDouble(currentEvent.Confidence);
                    double nextConf = NumOps.ToDouble(next.Confidence);

                    currentEvent = new AudioEvent<T>
                    {
                        EventType = currentEvent.EventType,
                        StartTime = currentEvent.StartTime,
                        EndTime = Math.Max(currentEvent.EndTime, next.EndTime),
                        Confidence = currentConf > nextConf ? currentEvent.Confidence : next.Confidence,
                        PeakTime = currentConf > nextConf ? currentEvent.PeakTime : next.PeakTime
                    };
                }
                else
                {
                    merged.Add(currentEvent);
                    currentEvent = next;
                }
            }

            merged.Add(currentEvent);
        }

        return merged.OrderBy(e => e.StartTime).ToList();
    }

    private Dictionary<string, EventStatistics<T>> ComputeEventStatistics(IReadOnlyList<AudioEvent<T>> events)
    {
        var stats = new Dictionary<string, EventStatistics<T>>();

        var grouped = events.GroupBy(e => e.EventType);
        foreach (var group in grouped)
        {
            var eventList = group.ToList();
            double totalDuration = eventList.Sum(e => e.Duration);
            double avgConfidence = eventList.Average(e => NumOps.ToDouble(e.Confidence));
            double maxConfidence = eventList.Max(e => NumOps.ToDouble(e.Confidence));

            stats[group.Key] = new EventStatistics<T>
            {
                Count = eventList.Count,
                TotalDuration = totalDuration,
                AverageConfidence = NumOps.FromDouble(avgConfidence),
                MaxConfidence = NumOps.FromDouble(maxConfidence)
            };
        }

        return stats;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(GetType().FullName ?? nameof(AudioEventDetector<T>));
        }
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Disposes managed resources.
    /// </summary>
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;

        if (disposing)
        {
            OnnxEncoder?.Dispose();
        }

        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion

    #region Nested Types

    /// <summary>
    /// Implementation of streaming event detection session.
    /// </summary>
    private sealed class StreamingEventDetectionSession : IStreamingEventDetectionSession<T>
    {
        private readonly AudioEventDetector<T> _detector;
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

        public StreamingEventDetectionSession(
            AudioEventDetector<T> detector,
            int sampleRate,
            T threshold)
        {
            _detector = detector;
            _sampleRate = sampleRate;
            _threshold = threshold;
            _buffer = [];
            _newEvents = [];
            _currentState = new Dictionary<string, T>();
            _windowSamples = (int)(detector._options.WindowSize * sampleRate);
            _processedTime = 0;

            // Initialize state
            foreach (var label in detector.ClassLabels)
            {
                _currentState[label] = detector.NumOps.Zero;
            }
        }

        public void FeedAudio(Tensor<T> audioChunk)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(StreamingEventDetectionSession));
            }

            // Collect events to raise outside the lock to prevent deadlock
            // (event handlers might acquire other locks or call back into this session)
            List<AudioEvent<T>>? eventsToRaise = null;

            lock (_lock)
            {
                if (_disposed)
                {
                    throw new ObjectDisposedException(nameof(StreamingEventDetectionSession));
                }

                // Add to buffer
                for (int i = 0; i < audioChunk.Length; i++)
                {
                    _buffer.Add(audioChunk[i]);
                }

                // Process complete windows (inside lock for thread-safety)
                while (_buffer.Count >= _windowSamples)
                {
                    // Extract window
                    var window = new Tensor<T>([_windowSamples]);
                    for (int i = 0; i < _windowSamples; i++)
                    {
                        window[i] = _buffer[i];
                    }

                    // Process window
                    var melSpec = _detector._melSpectrogram?.Forward(window) ??
                        throw new InvalidOperationException("MelSpectrogram not initialized.");
                    var scores = _detector.ClassifyWindow(melSpec, window);

                    // Update state and check for events
                    double thresholdValue = _detector.NumOps.ToDouble(_threshold);
                    for (int i = 0; i < scores.Length && i < _detector.ClassLabels.Count; i++)
                    {
                        _currentState[_detector.ClassLabels[i]] = scores[i];

                        if (_detector.NumOps.ToDouble(scores[i]) >= thresholdValue)
                        {
                            var evt = new AudioEvent<T>
                            {
                                EventType = _detector.ClassLabels[i],
                                Confidence = scores[i],
                                StartTime = _processedTime,
                                EndTime = _processedTime + _detector._options.WindowSize,
                                PeakTime = _processedTime + _detector._options.WindowSize / 2
                            };

                            _newEvents.Add(evt);

                            // Collect event for raising outside lock
                            eventsToRaise ??= new List<AudioEvent<T>>();
                            eventsToRaise.Add(evt);
                        }
                    }

                    // Advance buffer (with overlap)
                    int hopSamples = (int)(_windowSamples * (1 - _detector._options.WindowOverlap));
                    _buffer.RemoveRange(0, hopSamples);
                    _processedTime += hopSamples / (double)_sampleRate;
                }
            }

            // Raise events outside lock to prevent deadlock
            if (eventsToRaise is not null)
            {
                foreach (var evt in eventsToRaise)
                {
                    EventDetected?.Invoke(this, evt);
                }
            }
        }

        public IReadOnlyList<AudioEvent<T>> GetNewEvents()
        {
            lock (_lock)
            {
                var events = _newEvents.ToList();
                _newEvents.Clear();
                return events;
            }
        }

        public IReadOnlyDictionary<string, T> GetCurrentState()
        {
            lock (_lock)
            {
                return new Dictionary<string, T>(_currentState);
            }
        }

        public void Dispose()
        {
            if (_disposed) return;

            lock (_lock)
            {
                if (_disposed) return;
                _disposed = true;
                _buffer.Clear();
                _newEvents.Clear();
                _currentState.Clear();
            }
        }
    }

    #endregion
}
