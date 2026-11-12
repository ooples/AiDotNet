using AiDotNet.Attributes;
using AiDotNet.Audio.Features;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Enums;
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
/// <b>Architecture:</b> Hershey et al. 2017's best classifier, ResNet-50 adapted to audio (section 3.3.5): the
/// stride of 2 is removed from the first 7x7 convolution and the final average pool covers the 6 x 4 activations of a
/// 96 x 64 log-mel patch (960 ms of 25 ms windows every 10 ms, 64 mel bins, log after a small offset). Batch
/// normalization follows every convolution and one sigmoid unit per label replaces the softmax, trained with
/// cross-entropy and Adam (section 3.1). Audio is classified in non-overlapping 960 ms windows.
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
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.TwoDimensional,
///     NeuralNetworkTaskType.MultiLabelClassification, inputHeight: 96, inputWidth: 64, outputSize: 50);
/// var detector = new AudioEventDetector&lt;float&gt;(architecture, new AudioEventDetectorOptions());
/// detector.Train(features, labels);
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Detection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
// arXiv 1609.09430 is Hershey et al.'s CNN-architectures study, which is what this CNN-based
// detector implements. The previous entry paired that id with the title and author list of the
// companion Audio Set dataset paper (Gemmeke et al., ICASSP 2017) — two different papers.
[ResearchPaper("CNN Architectures for Large-Scale Audio Classification", "https://arxiv.org/abs/1609.09430", Year = 2017, Authors = "Shawn Hershey, Sourish Chaudhuri, Daniel P. W. Ellis, Jort F. Gemmeke, Aren Jansen, R. Channing Moore, Manoj Plakal, Devin Platt, Rif A. Saurous, Bryan Seybold, Malcolm Slaney, Ron J. Weiss, Kevin Wilson")]
public partial class AudioEventDetector<T> : AudioClassifierBase<T>, IAudioEventDetector<T>
{
    #region Fields

    private readonly AudioEventDetectorOptions _options;

    private MelSpectrogram<T>? _melSpectrogram;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    // The stem's 3x3 max pooling pads by one on each side (ResNet's stem), and the global average pooling ends the
    // convolutional trunk; both -1 with custom layers.
    private int _stemPoolIndex = -1;
    private int _globalPoolIndex = -1;
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
        Options = _options;
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
            logMel: true,
            // Section 3.1: "the magnitude of each bin is log-transformed after adding a small offset"; 0.01 in the
            // authors' released front end.
            logOffset: 0.01);

        // Load ONNX model
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);

        // Initialize class labels
        ClassLabels = _options.CustomLabels ?? CommonEventLabels;

        _options.ModelPath = modelPath;

        InitializeLayers();
    }

    /// <summary>Creates an AudioEventDetector for native training mode with the default objective.</summary>
    /// <remarks>
    /// Kept as its own three-parameter constructor rather than relying on the defaulted
    /// parameter below: a defaulted parameter is a compile-time convenience but only ONE CLR
    /// signature, so adding it removed the arity already-compiled callers bind to and they
    /// would fail with MissingMethodException.
    /// </remarks>
    public AudioEventDetector(
        NeuralNetworkArchitecture<T> architecture,
        AudioEventDetectorOptions? options,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer)
        : this(architecture, options, optimizer, null)
    {
    }

    /// <summary>
    /// Creates an AudioEventDetector for native training mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="options">Detection options.</param>
    /// <param name="optimizer">Optional custom optimizer (defaults to Adam, the paper's optimizer).</param>
    /// <param name="lossFunction">
    /// Optional custom training loss; defaults to binary cross-entropy with logits for multi-label detection.
    /// </param>
    public AudioEventDetector(
        NeuralNetworkArchitecture<T> architecture,
        AudioEventDetectorOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new BinaryCrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new AudioEventDetectorOptions();
        Options = _options;
        _useNativeMode = true;
        // Section 3.1: trained with Adam (learning rate by grid search), no dropout or weight decay.
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

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
            logMel: true,
            // Section 3.1: "the magnitude of each bin is log-transformed after adding a small offset"; 0.01 in the
            // authors' released front end.
            logOffset: 0.01);

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
                inputType: InputType.TwoDimensional,
                taskType: NeuralNetworkTaskType.MultiLabelClassification,
                inputHeight: (options ?? new AudioEventDetectorOptions()).PatchFrames,
                inputWidth: (options ?? new AudioEventDetectorOptions()).NumMels,
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
            foreach (int i in AudioEventDetectionOps.IndicesAtOrAbove(Engine, scores, ClassLabels.Count, threshold))
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
        var rows = new List<Tensor<T>>(windows.Count);

        for (int windowIdx = 0; windowIdx < windows.Count; windowIdx++)
        {
            var window = windows[windowIdx];
            var melSpec = _melSpectrogram?.Forward(window) ??
                throw new InvalidOperationException("MelSpectrogram not initialized.");

            rows.Add(AudioEventDetectionOps.ScoreRow(ClassifyWindow(melSpec, window), ClassLabels.Count));
        }

        return AudioEventDetectionOps.StackScoreRows(Engine, rows, ClassLabels.Count);
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
        return ToLegacyEvents(result.Events);
    }

    /// <summary>
    /// Converts generic events to the legacy double-confidence events, converting all confidences in
    /// one tensor cast (the legacy type's contract is double).
    /// </summary>
    private static List<AudioEvent> ToLegacyEvents(IReadOnlyList<AudioEvent<T>> events)
    {
        if (events.Count == 0) return new List<AudioEvent>();
        double[] confidences = new Tensor<T>(events.Select(e => e.Confidence).ToArray(), [events.Count]).Cast<double>().ToArray();
        return events.Select((e, k) => new AudioEvent
        {
            Label = e.EventType,
            Confidence = confidences[k],
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

        return ToLegacyEvents(result.Events);
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
        int count = Math.Min(scores.Length, ClassLabels.Count);
        if (count == 0) return result;
        // The legacy contract is double: convert the scores with one tensor cast.
        double[] values = new Tensor<T>(scores.AsSpan(0, count).ToArray(), [count]).Cast<double>().ToArray();
        for (int i = 0; i < count; i++)
        {
            result[ClassLabels[i]] = values[i];
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

        _stemPoolIndex = -1;
        _globalPoolIndex = -1;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            // Hershey et al.'s audio ResNet-50 (section 3.3.5): ResNet-50 with the stride of 2 removed from the first
            // 7x7 convolution, batch normalization after every convolution, and an average pool over the 6 x 4 final
            // activations of a 96 x 64 patch, under one sigmoid unit per label (section 3.1).
            Layers.Add(new ConvolutionalLayer<T>(_options.BaseChannels, kernelSize: 7, stride: 1, padding: 3,
                activationFunction: new IdentityActivation<T>()));
            Layers.Add(new BatchNormalizationLayer<T>());
            Layers.Add(new ActivationLayer<T>((IActivationFunction<T>)new ReLUActivation<T>()));
            _stemPoolIndex = Layers.Count;
            Layers.Add(new MaxPoolingLayer<T>(poolSize: 3, stride: 2));
            for (int stage = 0; stage < _options.StageDepths.Length; stage++)
            {
                for (int block = 0; block < _options.StageDepths[stage]; block++)
                    Layers.Add(new BottleneckBlock<T>(_options.BaseChannels << stage, stride: stage > 0 && block == 0 ? 2 : 1));
            }
            _globalPoolIndex = Layers.Count;
            Layers.Add(AdaptiveAveragePoolingLayer<T>.GlobalPool());
            Layers.Add(new FlattenLayer<T>());
            Layers.Add(new DenseLayer<T>(ClassLabels.Count, (IActivationFunction<T>)new IdentityActivation<T>()));
        }
    }

    /// <summary>
    /// Predicts output for the given input.
    /// </summary>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();

        if (IsOnnxMode && OnnxEncoder is not null)
        {
            return OnnxEncoder.Run(input);
        }

        // Native mode: per-label logits, then a per-label SIGMOID. The paper's examples carry several labels at once,
        // so "all models used a final sigmoid layer rather than a softmax layer" (section 3.1). Training runs on the
        // raw logits (ForwardForTraining) with cross-entropy, the standard train-on-logits / predict-with-sigmoid
        // pattern.
        return Engine.Sigmoid(Logits(input));
    }

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => Logits(input);

    /// <summary>
    /// Per-label logits of log-mel patches: <c>[frames, mels]</c> gives <c>[labels]</c>, <c>[batch, frames, mels]</c> gives
    /// <c>[batch, labels]</c>.
    /// </summary>
    private Tensor<T> Logits(Tensor<T> input)
    {
        var current = Run(input, Layers.Count);
        return input.Rank == 2 && current.Rank == 2 && current.Shape[0] == 1 ? Engine.Reshape(current, new[] { current.Shape[1] }) : current;
    }

    /// <summary>The convolutional trunk's output <c>[batch, channels, height, width]</c> for log-mel patches, the
    /// activations the global average pooling reduces (6 x 4 for a 96 x 64 patch).</summary>
    internal Tensor<T> TrunkActivations(Tensor<T> input) =>
        Run(input, _globalPoolIndex >= 0 ? _globalPoolIndex : Layers.Count);

    private Tensor<T> Run(Tensor<T> input, int end)
    {
        var current = input.Rank switch
        {
            2 => Engine.Reshape(input, new[] { 1, 1, input.Shape[0], input.Shape[1] }),
            3 => Engine.Reshape(input, new[] { input.Shape[0], 1, input.Shape[1], input.Shape[2] }),
            _ => input,
        };
        for (int i = 0; i < end; i++)
        {
            // The stem pooling pads by one; after the ReLU every value is non-negative, so zeros never win a maximum.
            if (i == _stemPoolIndex) current = Engine.TensorConstantPad(current, new[] { 1, 1, 1, 1 }, NumOps.Zero);
            current = Layers[i].Forward(current);
        }
        return current;
    }

    /// <summary>
    /// The classifier input of one window's log-mel spectrogram: its first <see cref="AudioEventDetectorOptions.PatchFrames"/>
    /// frames, zero-padded when shorter (a 960 ms window gives 97 frames with centred framing).
    /// </summary>
    internal Tensor<T> Patch(Tensor<T> melSpec)
    {
        int frames = _options.PatchFrames, bins = melSpec.Shape[melSpec.Rank - 1], available = melSpec.Length / bins;
        var patch = new Tensor<T>(new[] { frames, bins });
        int copy = Math.Min(frames, available) * bins;
        melSpec.Data.Span.Slice(0, copy).CopyTo(patch.Data.Span);
        return patch;
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
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
        return AudioEventDetectionOps.SigmoidDetached(Engine, modelOutput);
    }

    /// <summary>
    /// Gets model metadata.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "AudioEventDetector-Native" : "AudioEventDetector-ONNX",
            Description = "Audio event detection with the audio ResNet-50 of Hershey et al. 2017",
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


    /// <summary>
    /// Deserializes network-specific data.
    /// </summary>


    #endregion

    #region Helper Methods

    private List<Tensor<T>> SplitIntoWindows(Tensor<T> audio)
    {
        int windowSamples = (int)(_options.WindowSize * _options.SampleRate);
        int hopSamples = (int)(windowSamples * (1 - _options.WindowOverlap));

        // Ensure we don't divide by zero
        if (hopSamples <= 0) hopSamples = 1;

        // Full windows every hop, then a zero-padded partial window if at least 10% of a window
        // remains; audio too short for either is still zero-padded into one window.
        return AudioEventDetectionOps.SplitIntoWindows(Engine, audio, windowSamples, hopSamples, windowSamples / 10);
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
        var patch = Patch(melSpec);
        var input = Engine.Reshape(patch, new[] { 1, 1, patch.Shape[0], patch.Shape[1] });

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
        // One 96 x 64 patch per 960 ms window; PredictCore already applies the per-label sigmoid, so these are the
        // presence probabilities.
        var output = Predict(Patch(melSpec));
        var scores = new T[ClassLabels.Count];
        for (int i = 0; i < Math.Min(output.Length, scores.Length); i++)
        {
            scores[i] = output[i];
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
        if (audio.Length == 0) return 0.0 / audio.Length;
        return NumOps.ToDouble(Engine.TensorSum(Engine.TensorSquare(audio))) / audio.Length;
    }

    private double ComputeZeroCrossingRate(Tensor<T> audio)
        => AudioEngineOps.ZeroCrossingCount(Engine, audio) / (double)audio.Length;

    private double ComputeSpectralCentroid(Tensor<T> melSpec)
    {
        int frames = melSpec.Shape[0], bins = melSpec.Shape[1];
        if (frames == 0 || bins == 0) return 0;

        // weightedSum = sum_f f * (sum_t mag[t, f]); totalSum = sum of all magnitudes.
        var binTotals = Engine.ReduceSum(melSpec, [0], keepDims: false);
        double weightedSum = NumOps.ToDouble(Engine.TensorSum(Engine.TensorMultiply(binTotals, Engine.TensorArange<T>(0, bins))));
        double totalSum = NumOps.ToDouble(Engine.TensorSum(binTotals));

        return totalSum > 0 ? weightedSum / totalSum : 0;
    }

    private double ComputeSpectralFlatness(Tensor<T> melSpec)
    {
        int count = melSpec.Shape[0] * melSpec.Shape[1];
        if (count == 0) return 0;

        var mag = Engine.TensorMax(melSpec, NumOps.FromDouble(1e-10));
        double logSum = NumOps.ToDouble(Engine.TensorSum(Engine.TensorLog(mag)));
        double sum = NumOps.ToDouble(Engine.TensorSum(mag));

        double geometricMean = Math.Exp(logSum / count);
        double arithmeticMean = sum / count;

        return arithmeticMean > 0 ? geometricMean / arithmeticMean : 0;
    }

    private double ComputeBandEnergy(Tensor<T> melSpec, int startBin, int endBin)
    {
        int frames = melSpec.Shape[0];
        int stop = Math.Min(endBin, melSpec.Shape[1]);
        int width = stop - startBin;
        if (frames == 0 || width <= 0) return 0;

        var band = Engine.TensorSlice(melSpec, [0, startBin], [frames, width]);
        return NumOps.ToDouble(Engine.TensorSum(band)) / (frames * width);
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

    // Overlapping or adjacent (within 100ms) detections merge, keeping the higher confidence.
    private List<AudioEvent<T>> MergeEvents(List<AudioEvent<T>> events)
        => AudioEventDetectionOps.MergeEvents(Engine, events, 0.1);

    private Dictionary<string, EventStatistics<T>> ComputeEventStatistics(IReadOnlyList<AudioEvent<T>> events)
        => AudioEventDetectionOps.ComputeEventStatistics(Engine, events);

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
                _buffer.AddRange(audioChunk.ToArray());

                // Process complete windows (inside lock for thread-safety)
                while (_buffer.Count >= _windowSamples)
                {
                    // Extract window
                    var window = new Tensor<T>(_buffer.GetRange(0, _windowSamples).ToArray(), [_windowSamples]);

                    // Process window
                    var melSpec = _detector._melSpectrogram?.Forward(window) ??
                        throw new InvalidOperationException("MelSpectrogram not initialized.");
                    var scores = _detector.ClassifyWindow(melSpec, window);

                    // Update state and check for events
                    int classCount = Math.Min(scores.Length, _detector.ClassLabels.Count);
                    for (int i = 0; i < classCount; i++)
                    {
                        _currentState[_detector.ClassLabels[i]] = scores[i];
                    }

                    foreach (int i in AudioEventDetectionOps.IndicesAtOrAbove(_detector.Engine, scores, classCount, _threshold))
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
