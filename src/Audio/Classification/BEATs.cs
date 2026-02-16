using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Features;
using AiDotNet.Diffusion.Audio;
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
/// BEATs (Audio Pre-Training with Acoustic Tokenizers) model for state-of-the-art audio
/// event detection and classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// BEATs (Chen et al., ICML 2023) achieves state-of-the-art results on multiple audio benchmarks:
/// <list type="bullet">
/// <item><b>AudioSet-2M</b>: 50.6% mAP (mean Average Precision)</item>
/// <item><b>ESC-50</b>: 98.1% accuracy</item>
/// <item><b>AudioSet-Balanced</b>: 47.5% mAP</item>
/// </list>
/// </para>
/// <para>
/// <b>Architecture:</b> BEATs follows a Vision Transformer (ViT)-inspired architecture adapted
/// for audio:
/// <list type="number">
/// <item><b>Audio preprocessing</b>: Waveform to 128-bin mel spectrogram</item>
/// <item><b>Patch embedding</b>: Non-overlapping 16x16 patches linearly projected to embedding space</item>
/// <item><b>Positional encoding</b>: Learnable positional embeddings added to patch embeddings</item>
/// <item><b>Transformer encoder</b>: 12-layer encoder with multi-head self-attention and GELU FFN</item>
/// <item><b>Classification head</b>: Mean pooling + linear projection to class logits</item>
/// </list>
/// </para>
/// <para>
/// <b>Training Strategy:</b> BEATs uses an iterative self-distillation framework:
/// <list type="number">
/// <item>An acoustic tokenizer creates discrete labels for audio patches</item>
/// <item>The audio model is pre-trained via masked patch prediction against these labels</item>
/// <item>The improved audio model is used to train a better tokenizer</item>
/// <item>This cycle repeats for 3 iterations (BEATs_iter3 = best variant)</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> BEATs is one of the best models for identifying sounds in audio.
/// It can detect hundreds of different sounds (speech, music, animals, machinery, etc.)
/// simultaneously in the same audio clip.
///
/// Think of how it works:
/// <list type="bullet">
/// <item>Audio is converted to a visual representation (spectrogram, like a sound fingerprint)</item>
/// <item>The spectrogram is cut into small tiles (patches, like puzzle pieces)</item>
/// <item>A Transformer (the architecture behind modern AI) learns how the patches relate to each other</item>
/// <item>The model determines what sounds are present and when they occur</item>
/// </list>
///
/// Usage with a pre-trained ONNX model:
/// <code>
/// var options = new BEATsOptions { ModelPath = "beats_iter3.onnx" };
/// var beats = new BEATs&lt;float&gt;(options);
/// var result = beats.Detect(audioWaveform);
/// foreach (var evt in result.Events)
/// {
///     Console.WriteLine($"{evt.EventType}: {evt.Confidence:P1} at {evt.StartTime:F2}s-{evt.EndTime:F2}s");
/// }
/// </code>
///
/// Usage with native training:
/// <code>
/// var options = new BEATsOptions
/// {
///     EmbeddingDim = 768,
///     NumEncoderLayers = 12,
///     CustomLabels = new[] { "speech", "music", "noise" }
/// };
/// var beats = new BEATs&lt;float&gt;(options);
/// beats.Train(features, labels);
/// var result = beats.Detect(testAudio);
/// </code>
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "BEATs: Audio Pre-Training with Acoustic Tokenizers" (Chen et al., ICML 2023)</item>
/// <item>Repository: https://github.com/microsoft/unilm/tree/master/beats</item>
/// </list>
/// </para>
/// </remarks>
public class BEATs<T> : AudioClassifierBase<T>, IAudioEventDetector<T>
{
    #region Fields

    private readonly BEATsOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private MelSpectrogram<T>? _melSpectrogram;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    // BEATs-specific layers for native training mode
    private FullyConnectedLayer<T>? _patchEmbedding;
    private Tensor<T>? _positionalEmbeddings;
    private Tensor<T>? _clsToken;
    private LayerNormalizationLayer<T>? _preNorm;
    private List<TransformerEncoderLayer<T>>? _encoderLayers;
    private LayerNormalizationLayer<T>? _postNorm;
    private FullyConnectedLayer<T>? _classificationHead;

    /// <summary>
    /// AudioSet-527 standard event labels used by BEATs pre-trained models.
    /// </summary>
    /// <remarks>
    /// <para>This is a representative subset of AudioSet's 527 classes. The full AudioSet
    /// ontology covers a comprehensive taxonomy of everyday sounds.</para>
    /// </remarks>
    public static readonly string[] AudioSetLabels =
    [
        // Human sounds
        "Speech", "Male speech", "Female speech", "Child speech",
        "Conversation", "Narration", "Laughter", "Baby laughter",
        "Crying", "Baby cry", "Whimper", "Cough", "Sneeze",
        "Breathing", "Sigh", "Yawn", "Snoring", "Gasp", "Hiccup",
        "Clapping", "Cheering", "Crowd", "Footsteps", "Chewing",

        // Musical instruments
        "Music", "Musical instrument", "Guitar", "Electric guitar",
        "Bass guitar", "Acoustic guitar", "Piano", "Keyboard",
        "Organ", "Synthesizer", "Drums", "Drum kit", "Snare drum",
        "Bass drum", "Hi-hat", "Cymbal", "Tambourine",
        "Violin", "Cello", "Harp", "Flute", "Clarinet", "Saxophone",
        "Trumpet", "Trombone", "French horn", "Harmonica", "Accordion",

        // Singing and vocal music
        "Singing", "Choir", "Rapping", "Humming", "Whistling",
        "Yodeling",

        // Animal sounds
        "Dog", "Dog barking", "Dog howl", "Dog whimper",
        "Cat", "Cat meowing", "Cat purring", "Cat hissing",
        "Bird", "Bird song", "Bird call", "Chirp", "Tweet",
        "Crow", "Rooster", "Duck", "Goose", "Owl",
        "Horse", "Cow", "Pig", "Sheep", "Goat",
        "Frog", "Cricket", "Insect", "Bee", "Mosquito",

        // Nature and environment
        "Rain", "Raindrop", "Thunder", "Thunderstorm",
        "Wind", "Howling wind", "Rustling leaves",
        "Water", "Stream", "Waterfall", "Ocean", "Waves",
        "Fire", "Crackling fire",

        // Vehicles and transport
        "Car", "Car engine", "Car horn", "Tire squeal",
        "Truck", "Bus", "Motorcycle", "Bicycle",
        "Train", "Train horn", "Train wheels",
        "Airplane", "Helicopter", "Jet engine",
        "Boat", "Ship horn", "Siren", "Ambulance siren",
        "Police siren", "Fire truck siren",

        // Household sounds
        "Door", "Doorbell", "Door knock", "Door slam",
        "Clock", "Clock tick", "Clock alarm", "Alarm",
        "Telephone", "Telephone ring", "Telephone dialing",
        "Television", "Radio",
        "Microwave", "Blender", "Vacuum cleaner", "Washing machine",
        "Toilet flush", "Water tap", "Dishes", "Cutlery",

        // Office and technology
        "Keyboard typing", "Mouse click", "Printer",
        "Computer fan", "Camera shutter",

        // Tools and machinery
        "Hammer", "Saw", "Drill", "Chainsaw",
        "Engine", "Motor", "Power tool",
        "Construction", "Jackhammer",

        // Impact and action sounds
        "Knock", "Tap", "Bang", "Crash", "Thud",
        "Glass breaking", "Wood cracking",
        "Explosion", "Gunshot", "Fireworks",
        "Splash", "Drip",

        // Signals and alarms
        "Bell", "Church bell", "Chime",
        "Buzzer", "Beep", "Ring",
        "Whistle", "Horn", "Foghorn",

        // Background and noise
        "Silence", "Noise", "Static", "White noise",
        "Pink noise", "Hum", "Buzz", "Hiss",
        "Echo", "Reverberation",

        // Music genres (for classification fine-tuning)
        "Rock music", "Pop music", "Hip hop music",
        "Jazz", "Classical music", "Electronic music",
        "Country music", "Blues", "Reggae", "Folk music"
    ];

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a BEATs model for ONNX inference mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input/output dimensions.</param>
    /// <param name="modelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="options">BEATs configuration options. If null, uses defaults.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you have a pre-trained BEATs model
    /// in ONNX format. The model will be ready for inference immediately without any training.
    /// </para>
    /// </remarks>
    public BEATs(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        BEATsOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new BEATsOptions();
        _useNativeMode = false;

        base.SampleRate = _options.SampleRate;
        base.NumMels = _options.NumMels;

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
        ClassLabels = _options.CustomLabels ?? AudioSetLabels;

        // Optimizer not used in ONNX mode but required by base infrastructure
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a BEATs model for native training mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="options">BEATs configuration options. If null, uses defaults.</param>
    /// <param name="optimizer">Optional custom optimizer. Defaults to AdamW.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you want to train a BEATs model
    /// from scratch on your own audio data. The model will need to be trained before
    /// it can make predictions.
    /// </para>
    /// </remarks>
    public BEATs(
        NeuralNetworkArchitecture<T> architecture,
        BEATsOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new BEATsOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);

        base.SampleRate = _options.SampleRate;
        base.NumMels = _options.NumMels;

        // Initialize class labels
        ClassLabels = _options.CustomLabels ?? AudioSetLabels;

        // Initialize mel spectrogram
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
    /// Creates a BEATs model with default architecture (native training mode).
    /// </summary>
    /// <param name="options">BEATs configuration options. If null, uses defaults.</param>
    /// <remarks>
    /// <para>
    /// This constructor automatically creates an appropriate architecture based on the options.
    /// The input features correspond to the embedding dimension and the output size to the
    /// number of class labels.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The simplest way to create a BEATs model.
    /// Just provide options (or use defaults) and the model configures itself.
    /// <code>
    /// var beats = new BEATs&lt;float&gt;(); // Default AudioSet-527 model
    /// var beats = new BEATs&lt;float&gt;(new BEATsOptions
    /// {
    ///     CustomLabels = new[] { "speech", "music", "silence" },
    ///     EmbeddingDim = 512,
    ///     NumEncoderLayers = 6
    /// });
    /// </code>
    /// </para>
    /// </remarks>
    public BEATs(BEATsOptions? options = null)
        : this(
            new NeuralNetworkArchitecture<T>(
                inputFeatures: (options ?? new BEATsOptions()).EmbeddingDim,
                outputSize: (options?.CustomLabels ?? AudioSetLabels).Length),
            options)
    {
    }

    /// <summary>
    /// Creates a BEATs model asynchronously with optional model download.
    /// </summary>
    /// <param name="options">BEATs configuration options.</param>
    /// <param name="progress">Optional progress reporter for model download.</param>
    /// <param name="cancellationToken">Cancellation token for the async operation.</param>
    /// <returns>A configured BEATs model ready for inference.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the easiest way to get started with a pre-trained BEATs model.
    /// If you provide a model path in options, it loads that file. Otherwise, it downloads
    /// a pre-trained model automatically.
    /// <code>
    /// var beats = await BEATs&lt;float&gt;.CreateAsync();
    /// var result = beats.Detect(audioTensor);
    /// </code>
    /// </para>
    /// </remarks>
    public static async Task<BEATs<T>> CreateAsync(
        BEATsOptions? options = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new BEATsOptions();
        string modelPath = options.ModelPath ?? string.Empty;

        if (string.IsNullOrEmpty(modelPath))
        {
            var downloader = new OnnxModelDownloader();
            modelPath = await downloader.DownloadAsync(
                "beats",
                "beats_iter3.onnx",
                progress: progress,
                cancellationToken);
            options.ModelPath = modelPath;
        }

        var architecture = new NeuralNetworkArchitecture<T>(
            inputFeatures: options.EmbeddingDim,
            outputSize: (options.CustomLabels ?? AudioSetLabels).Length);

        return new BEATs<T>(architecture, modelPath, options);
    }

    #endregion

    #region IAudioEventDetector Properties

    /// <summary>
    /// Gets the list of event types this model can detect.
    /// </summary>
    public IReadOnlyList<string> SupportedEvents => ClassLabels;

    /// <summary>
    /// Gets the event labels (alias for backwards compatibility).
    /// </summary>
    public IReadOnlyList<string> EventLabels => ClassLabels;

    /// <summary>
    /// Gets the time resolution for event detection in seconds.
    /// </summary>
    public double TimeResolution => _options.WindowSize * (1 - _options.WindowOverlap);

    #endregion

    #region IAudioEventDetector Methods

    /// <summary>
    /// Detects audio events in the audio stream using default threshold.
    /// </summary>
    /// <param name="audio">Audio waveform tensor [samples].</param>
    /// <returns>Detection result containing all events above the confidence threshold.</returns>
    public AudioEventResult<T> Detect(Tensor<T> audio)
    {
        ThrowIfDisposed();
        return Detect(audio, NumOps.FromDouble(_options.Threshold));
    }

    /// <summary>
    /// Detects audio events with a custom confidence threshold.
    /// </summary>
    /// <param name="audio">Audio waveform tensor [samples].</param>
    /// <param name="threshold">Confidence threshold (0.0 to 1.0).</param>
    /// <returns>Detection result containing events above the threshold.</returns>
    public AudioEventResult<T> Detect(Tensor<T> audio, T threshold)
    {
        ThrowIfDisposed();

        double thresholdValue = NumOps.ToDouble(threshold);
        double totalDuration = audio.Length / (double)_options.SampleRate;

        // Split audio into overlapping windows
        var windows = SplitIntoWindows(audio);
        var allEvents = new List<AudioEvent<T>>();

        for (int windowIdx = 0; windowIdx < windows.Count; windowIdx++)
        {
            var window = windows[windowIdx];
            double startTime = windowIdx * TimeResolution;

            // Extract mel spectrogram features
            var melSpec = _melSpectrogram?.Forward(window) ??
                throw new InvalidOperationException("MelSpectrogram not initialized.");

            // Classify the window
            var scores = ClassifyWindow(melSpec);

            // Collect events above threshold
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

        // Merge overlapping events of the same class
        var mergedEvents = MergeEvents(allEvents);

        // Compute per-class statistics
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
    /// <param name="audio">Audio waveform tensor.</param>
    /// <param name="cancellationToken">Cancellation token for the async operation.</param>
    /// <returns>Detection result.</returns>
    public Task<AudioEventResult<T>> DetectAsync(
        Tensor<T> audio,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Detect(audio), cancellationToken);
    }

    /// <summary>
    /// Detects specific event types only (filters out all other events).
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <param name="eventTypes">Event types to detect (case-insensitive match).</param>
    /// <returns>Detection result filtered to specified event types.</returns>
    public AudioEventResult<T> DetectSpecific(Tensor<T> audio, IReadOnlyList<string> eventTypes)
    {
        return DetectSpecific(audio, eventTypes, NumOps.FromDouble(_options.Threshold));
    }

    /// <summary>
    /// Detects specific event types with a custom confidence threshold.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <param name="eventTypes">Event types to detect (case-insensitive match).</param>
    /// <param name="threshold">Confidence threshold (0.0 to 1.0).</param>
    /// <returns>Detection result filtered to specified event types.</returns>
    public AudioEventResult<T> DetectSpecific(Tensor<T> audio, IReadOnlyList<string> eventTypes, T threshold)
    {
        var result = Detect(audio, threshold);

        // Filter to only requested event types (case-insensitive)
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
    /// Gets frame-level event probabilities for all classes.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <returns>Tensor of shape [time_frames, num_events] with probabilities.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This returns the full probability matrix instead of just
    /// the detected events. Useful for visualization, custom thresholding, or analysis.
    /// </para>
    /// </remarks>
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

            var scores = ClassifyWindow(melSpec);

            for (int i = 0; i < ClassLabels.Count && i < scores.Length; i++)
            {
                probabilities[windowIdx, i] = scores[i];
            }
        }

        return probabilities;
    }

    /// <summary>
    /// Starts a streaming event detection session with default settings.
    /// </summary>
    /// <returns>A streaming session that can process audio in real-time.</returns>
    public IStreamingEventDetectionSession<T> StartStreamingSession()
    {
        return StartStreamingSession(_options.SampleRate, NumOps.FromDouble(_options.Threshold));
    }

    /// <summary>
    /// Starts a streaming event detection session with custom settings.
    /// </summary>
    /// <param name="sampleRate">Sample rate of incoming audio.</param>
    /// <param name="threshold">Confidence threshold for event detection.</param>
    /// <returns>A streaming session that can process audio in real-time.</returns>
    public IStreamingEventDetectionSession<T> StartStreamingSession(int sampleRate, T threshold)
    {
        return new BEATsStreamingSession(this, sampleRate, threshold);
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <summary>
    /// Initializes the BEATs neural network layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// In native mode, creates the full BEATs architecture:
    /// <list type="number">
    /// <item>Patch embedding: Linear projection from patch features to embedding space</item>
    /// <item>CLS token: Learnable classification token prepended to the sequence</item>
    /// <item>Positional embeddings: Learnable embeddings for each patch position</item>
    /// <item>Pre-norm: Layer normalization before the Transformer encoder</item>
    /// <item>Transformer encoder: Stack of self-attention + FFN layers</item>
    /// <item>Post-norm: Layer normalization after the encoder</item>
    /// <item>Classification head: Linear projection from embeddings to class logits</item>
    /// </list>
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;

        // Use custom layers if provided via architecture
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            return;
        }

        int embDim = _options.EmbeddingDim;
        int patchFeatures = _options.PatchSize * _options.PatchSize;

        // Patch embedding: project flattened patch pixels to embedding dimension
        // Input: [patchSize * patchSize] (flattened mel spectrogram patch)
        // Output: [embeddingDim]
        _patchEmbedding = new FullyConnectedLayer<T>(
            patchFeatures,
            embDim,
            activationFunction: null);

        // CLS token: learnable vector prepended to the patch sequence for classification
        _clsToken = new Tensor<T>([embDim]);
        InitializeTensorXavier(_clsToken, embDim);

        // Positional embeddings: learnable embeddings for CLS + max patches
        // Max number of patches depends on audio length and patch size
        int maxPatches = 600; // Generous upper bound for 10s audio at 16kHz with 128 mel / 16 patch
        _positionalEmbeddings = new Tensor<T>([maxPatches + 1, embDim]); // +1 for CLS token
        InitializeTensorXavier(_positionalEmbeddings, embDim);

        // Pre-normalization before encoder
        _preNorm = new LayerNormalizationLayer<T>(embDim);

        // Transformer encoder layers
        _encoderLayers = new List<TransformerEncoderLayer<T>>();
        for (int i = 0; i < _options.NumEncoderLayers; i++)
        {
            _encoderLayers.Add(new TransformerEncoderLayer<T>(
                embDim,
                _options.NumAttentionHeads,
                _options.FeedForwardDim));
        }

        // Post-normalization after encoder
        _postNorm = new LayerNormalizationLayer<T>(embDim);

        // Classification head: project from embedding dim to number of classes
        _classificationHead = new FullyConnectedLayer<T>(
            embDim,
            ClassLabels.Count,
            activationFunction: null); // Sigmoid applied later for multi-label

        // Register all layers for parameter management
        Layers.Add(_patchEmbedding);
        Layers.Add(_preNorm);
        foreach (var encoderLayer in _encoderLayers)
        {
            Layers.Add(encoderLayer);
        }
        Layers.Add(_postNorm);
        Layers.Add(_classificationHead);
    }

    /// <summary>
    /// Runs inference on the given input tensor.
    /// </summary>
    /// <param name="input">Preprocessed input features.</param>
    /// <returns>Model output tensor (logits or probabilities).</returns>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();

        if (IsOnnxMode && OnnxEncoder is not null)
        {
            return OnnxEncoder.Run(input);
        }

        // Native mode: BEATs forward pass
        return ForwardBEATs(input);
    }

    /// <summary>
    /// Trains the model on a single input-target pair.
    /// </summary>
    /// <param name="input">Input audio features (mel spectrogram).</param>
    /// <param name="expected">Expected output labels (multi-hot encoding).</param>
    /// <exception cref="NotSupportedException">Thrown when called in ONNX mode.</exception>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
        {
            throw new NotSupportedException(
                "Training is not supported in ONNX mode. Create a BEATs model " +
                "without a modelPath parameter to train natively.");
        }

        // Enable training mode (enables dropout, batch norm in training mode, etc.)
        SetTrainingMode(true);

        // Forward pass
        var output = Predict(input);

        // Compute loss and gradient (BCE for multi-label classification)
        var loss = LossFunction.CalculateLoss(output.ToVector(), expected.ToVector());
        var gradient = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());

        // Backward pass through layers
        var gradientTensor = Tensor<T>.FromVector(gradient);

        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradientTensor = Layers[i].Backward(gradientTensor);
        }

        // Update parameters via optimizer
        _optimizer.UpdateParameters(Layers);

        // Switch back to inference mode
        SetTrainingMode(false);
    }

    /// <summary>
    /// Updates all network parameters from a flattened parameter vector.
    /// </summary>
    /// <param name="parameters">Flattened parameter vector.</param>
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
    /// Preprocesses raw audio into mel spectrogram features.
    /// </summary>
    /// <param name="rawAudio">Raw audio waveform tensor.</param>
    /// <returns>Log-mel spectrogram features.</returns>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (_melSpectrogram is null)
            throw new InvalidOperationException("MelSpectrogram not initialized.");

        return _melSpectrogram.Forward(rawAudio);
    }

    /// <summary>
    /// Post-processes model output by applying sigmoid activation for multi-label classification.
    /// </summary>
    /// <param name="modelOutput">Raw logits from the model.</param>
    /// <returns>Probabilities in [0, 1] for each class.</returns>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
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
    /// Gets metadata about this BEATs model instance.
    /// </summary>
    /// <returns>Model metadata including architecture details.</returns>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "BEATs-Native" : "BEATs-ONNX",
            Description = "BEATs: Audio Pre-Training with Acoustic Tokenizers (Chen et al., ICML 2023)",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = ClassLabels.Count,
            Complexity = _options.NumEncoderLayers
        };
        metadata.AdditionalInfo["Architecture"] = "BEATs";
        metadata.AdditionalInfo["EmbeddingDim"] = _options.EmbeddingDim.ToString();
        metadata.AdditionalInfo["NumEncoderLayers"] = _options.NumEncoderLayers.ToString();
        metadata.AdditionalInfo["NumAttentionHeads"] = _options.NumAttentionHeads.ToString();
        metadata.AdditionalInfo["FeedForwardDim"] = _options.FeedForwardDim.ToString();
        metadata.AdditionalInfo["PatchSize"] = _options.PatchSize.ToString();
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        metadata.AdditionalInfo["NumMels"] = _options.NumMels.ToString();
        metadata.AdditionalInfo["NumClasses"] = ClassLabels.Count.ToString();
        metadata.AdditionalInfo["TimeResolution"] = TimeResolution.ToString("F3");
        return metadata;
    }

    /// <summary>
    /// Serializes BEATs-specific model data for persistence.
    /// </summary>
    /// <param name="writer">Binary writer for serialization.</param>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Write mode and model path
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);

        // Write architecture options
        writer.Write(_options.SampleRate);
        writer.Write(_options.NumMels);
        writer.Write(_options.FftSize);
        writer.Write(_options.HopLength);
        writer.Write(_options.EmbeddingDim);
        writer.Write(_options.NumEncoderLayers);
        writer.Write(_options.NumAttentionHeads);
        writer.Write(_options.FeedForwardDim);
        writer.Write(_options.PatchSize);
        writer.Write(_options.PatchStride);
        writer.Write(_options.Threshold);
        writer.Write(_options.WindowSize);
        writer.Write(_options.WindowOverlap);
        writer.Write(_options.DropoutRate);
        writer.Write(_options.MaskProbability);
        writer.Write(_options.CodebookSize);

        // Write class labels
        writer.Write(ClassLabels.Count);
        foreach (var label in ClassLabels)
        {
            writer.Write(label);
        }
    }

    /// <summary>
    /// Deserializes BEATs-specific model data from persistent storage.
    /// </summary>
    /// <param name="reader">Binary reader for deserialization.</param>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Restore mode and model path
        _useNativeMode = reader.ReadBoolean();
        string modelPath = reader.ReadString();
        if (!string.IsNullOrEmpty(modelPath))
        {
            _options.ModelPath = modelPath;
        }

        // Restore architecture options
        _options.SampleRate = reader.ReadInt32();
        _options.NumMels = reader.ReadInt32();
        _options.FftSize = reader.ReadInt32();
        _options.HopLength = reader.ReadInt32();
        _options.EmbeddingDim = reader.ReadInt32();
        _options.NumEncoderLayers = reader.ReadInt32();
        _options.NumAttentionHeads = reader.ReadInt32();
        _options.FeedForwardDim = reader.ReadInt32();
        _options.PatchSize = reader.ReadInt32();
        _options.PatchStride = reader.ReadInt32();
        _options.Threshold = reader.ReadDouble();
        _options.WindowSize = reader.ReadDouble();
        _options.WindowOverlap = reader.ReadDouble();
        _options.DropoutRate = reader.ReadDouble();
        _options.MaskProbability = reader.ReadDouble();
        _options.CodebookSize = reader.ReadInt32();

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
    /// Creates a new BEATs instance for deserialization.
    /// </summary>
    /// <returns>A new uninitialized BEATs instance.</returns>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new BEATs<T>(Architecture, _options);
    }

    #endregion

    #region BEATs-Specific Methods

    /// <summary>
    /// Performs the full BEATs forward pass in native mode.
    /// </summary>
    /// <param name="melSpec">Mel spectrogram features [time_frames, num_mels].</param>
    /// <returns>Classification logits [num_classes].</returns>
    private Tensor<T> ForwardBEATs(Tensor<T> melSpec)
    {
        if (_patchEmbedding is null || _clsToken is null || _positionalEmbeddings is null ||
            _preNorm is null || _encoderLayers is null || _postNorm is null ||
            _classificationHead is null)
        {
            throw new InvalidOperationException(
                "BEATs layers not initialized. Ensure InitializeLayers() was called.");
        }

        // Step 1: Extract and embed patches from the mel spectrogram
        var patchEmbeddings = ExtractAndEmbedPatches(melSpec);
        int numPatches = patchEmbeddings.Count;

        // Step 2: Prepend CLS token and add positional embeddings
        // Sequence layout: [CLS, patch_1, patch_2, ..., patch_N]
        int seqLen = numPatches + 1; // +1 for CLS
        int embDim = _options.EmbeddingDim;
        var sequence = new Tensor<T>([seqLen * embDim]);

        // Add CLS token with positional embedding
        for (int d = 0; d < embDim; d++)
        {
            sequence[d] = NumOps.Add(_clsToken[d], _positionalEmbeddings[0, d]);
        }

        // Add patch embeddings with positional embeddings
        for (int p = 0; p < numPatches; p++)
        {
            int seqOffset = (p + 1) * embDim;
            int posIdx = Math.Min(p + 1, _positionalEmbeddings.Shape[0] - 1);
            for (int d = 0; d < embDim; d++)
            {
                sequence[seqOffset + d] = NumOps.Add(
                    patchEmbeddings[p][d],
                    _positionalEmbeddings[posIdx, d]);
            }
        }

        // Step 3: Pre-normalization
        var current = _preNorm.Forward(sequence);

        // Step 4: Pass through Transformer encoder layers
        foreach (var encoderLayer in _encoderLayers)
        {
            current = encoderLayer.Forward(current);
        }

        // Step 5: Post-normalization
        current = _postNorm.Forward(current);

        // Step 6: Extract CLS token representation (mean pooling as alternative)
        // BEATs uses the CLS token output for classification
        var clsOutput = new Tensor<T>([embDim]);
        for (int d = 0; d < embDim; d++)
        {
            clsOutput[d] = current[d]; // First embDim values = CLS token
        }

        // Step 7: Classification head
        var logits = _classificationHead.Forward(clsOutput);

        return logits;
    }

    /// <summary>
    /// Extracts non-overlapping patches from the mel spectrogram and projects them to embeddings.
    /// </summary>
    /// <param name="melSpec">Mel spectrogram [time_frames, num_mels] or flattened.</param>
    /// <returns>List of patch embeddings, each of shape [embedding_dim].</returns>
    private List<Tensor<T>> ExtractAndEmbedPatches(Tensor<T> melSpec)
    {
        if (_patchEmbedding is null)
            throw new InvalidOperationException("Patch embedding layer not initialized.");

        int numMels = _options.NumMels;
        int patchSize = _options.PatchSize;
        int patchStride = _options.PatchStride;
        int patchFeatures = patchSize * patchSize;

        // Determine spectrogram dimensions
        int timeFrames;
        if (melSpec.Shape.Length == 2)
        {
            timeFrames = melSpec.Shape[0];
        }
        else
        {
            // Flattened: assume [time_frames * num_mels]
            timeFrames = melSpec.Length / numMels;
        }

        // Calculate number of patches along each axis
        int patchesFreq = Math.Max(1, (numMels - patchSize) / patchStride + 1);
        int patchesTime = Math.Max(1, (timeFrames - patchSize) / patchStride + 1);

        var patches = new List<Tensor<T>>();

        for (int pt = 0; pt < patchesTime; pt++)
        {
            for (int pf = 0; pf < patchesFreq; pf++)
            {
                int startTime = pt * patchStride;
                int startFreq = pf * patchStride;

                // Extract and flatten the patch
                var patchFlat = new Tensor<T>([patchFeatures]);
                int idx = 0;

                for (int t = 0; t < patchSize && (startTime + t) < timeFrames; t++)
                {
                    for (int f = 0; f < patchSize && (startFreq + f) < numMels; f++)
                    {
                        if (melSpec.Shape.Length == 2)
                        {
                            patchFlat[idx] = melSpec[startTime + t, startFreq + f];
                        }
                        else
                        {
                            int flatIdx = (startTime + t) * numMels + (startFreq + f);
                            if (flatIdx < melSpec.Length)
                            {
                                patchFlat[idx] = melSpec[flatIdx];
                            }
                        }
                        idx++;
                    }
                }

                // Project to embedding dimension
                var embedding = _patchEmbedding.Forward(patchFlat);
                patches.Add(embedding);
            }
        }

        // Ensure at least one patch
        if (patches.Count == 0)
        {
            var zeroPatch = new Tensor<T>([patchFeatures]);
            for (int i = 0; i < Math.Min(melSpec.Length, patchFeatures); i++)
            {
                zeroPatch[i] = melSpec[i];
            }
            patches.Add(_patchEmbedding.Forward(zeroPatch));
        }

        return patches;
    }

    /// <summary>
    /// Initializes a tensor with Xavier/Glorot uniform initialization.
    /// </summary>
    /// <param name="tensor">The tensor to initialize.</param>
    /// <param name="fanIn">Number of input units.</param>
    private void InitializeTensorXavier(Tensor<T> tensor, int fanIn)
    {
        var rand = RandomHelper.CreateSecureRandom();
        double scale = Math.Sqrt(2.0 / fanIn);

        for (int i = 0; i < tensor.Length; i++)
        {
            double value = (rand.NextDouble() * 2.0 - 1.0) * scale;
            tensor[i] = NumOps.FromDouble(value);
        }
    }

    #endregion

    #region Classification Helpers

    /// <summary>
    /// Classifies a mel spectrogram window and returns per-class probabilities.
    /// </summary>
    /// <param name="melSpec">Mel spectrogram features for one window.</param>
    /// <returns>Array of per-class probabilities.</returns>
    private T[] ClassifyWindow(Tensor<T> melSpec)
    {
        Tensor<T> output;

        if (IsOnnxMode && OnnxEncoder is not null)
        {
            // ONNX mode: prepare input with batch and channel dimensions [1, 1, T, F]
            var input = new Tensor<T>([1, 1, melSpec.Shape[0], melSpec.Shape[1]]);
            for (int t = 0; t < melSpec.Shape[0]; t++)
            {
                for (int f = 0; f < melSpec.Shape[1]; f++)
                {
                    input[0, 0, t, f] = melSpec[t, f];
                }
            }
            output = OnnxEncoder.Run(input);
        }
        else if (_useNativeMode)
        {
            // Native mode: run full BEATs forward pass
            output = ForwardBEATs(melSpec);
        }
        else
        {
            // Fallback: return uniform low probabilities
            var fallback = new T[ClassLabels.Count];
            for (int i = 0; i < fallback.Length; i++)
            {
                fallback[i] = NumOps.FromDouble(0.01);
            }
            return fallback;
        }

        // Apply sigmoid for multi-label probabilities
        var scores = new T[ClassLabels.Count];
        for (int i = 0; i < Math.Min(output.Length, scores.Length); i++)
        {
            double logit = NumOps.ToDouble(output[i]);
            scores[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-logit)));
        }

        return scores;
    }

    #endregion

    #region Audio Windowing Helpers

    /// <summary>
    /// Splits audio into overlapping windows for frame-level processing.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <returns>List of audio windows.</returns>
    private List<Tensor<T>> SplitIntoWindows(Tensor<T> audio)
    {
        var windows = new List<Tensor<T>>();
        int windowSamples = (int)(_options.WindowSize * _options.SampleRate);
        int hopSamples = (int)(windowSamples * (1 - _options.WindowOverlap));

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

        // Handle remaining audio that doesn't fill a complete window
        int remainingStart = windows.Count > 0 ? lastStart : 0;
        int remainingSamples = audio.Length - remainingStart;

        // Include partial window if at least 10% of window size remains
        if (remainingSamples > windowSamples / 10)
        {
            var window = new Tensor<T>([windowSamples]);
            for (int i = 0; i < remainingSamples && i < windowSamples; i++)
            {
                window[i] = audio[remainingStart + i];
            }
            // Remaining values are zero-padded
            windows.Add(window);
        }
        else if (windows.Count == 0 && audio.Length > 0)
        {
            // Short audio: zero-pad to window size
            var window = new Tensor<T>([windowSamples]);
            for (int i = 0; i < audio.Length; i++)
            {
                window[i] = audio[i];
            }
            windows.Add(window);
        }

        return windows;
    }

    #endregion

    #region Event Processing Helpers

    /// <summary>
    /// Merges overlapping or adjacent events of the same type into single continuous events.
    /// </summary>
    /// <param name="events">Raw detected events with potential overlaps.</param>
    /// <returns>Merged events with no overlaps within the same class.</returns>
    private List<AudioEvent<T>> MergeEvents(List<AudioEvent<T>> events)
    {
        if (events.Count == 0) return events;

        var grouped = events.GroupBy(e => e.EventType);
        var merged = new List<AudioEvent<T>>();

        foreach (var group in grouped)
        {
            var sortedEvents = group.OrderBy(e => e.StartTime).ToList();
            var currentEvent = sortedEvents[0];

            for (int i = 1; i < sortedEvents.Count; i++)
            {
                var next = sortedEvents[i];

                // Merge if overlapping or within 100ms gap
                if (next.StartTime <= currentEvent.EndTime + 0.1)
                {
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

    /// <summary>
    /// Computes per-class event statistics from a list of detected events.
    /// </summary>
    /// <param name="events">Detected events to compute statistics for.</param>
    /// <returns>Dictionary of event type to statistics.</returns>
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

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(GetType().FullName ?? nameof(BEATs<T>));
        }
    }

    /// <summary>
    /// Disposes managed and unmanaged resources.
    /// </summary>
    /// <param name="disposing">True if disposing managed resources.</param>
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

    #region Streaming Session

    /// <summary>
    /// Streaming event detection session for real-time BEATs inference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This allows you to feed audio data chunk by chunk
    /// (like from a microphone) and get event detections in real-time.
    /// <code>
    /// using var session = beats.StartStreamingSession();
    /// session.EventDetected += (sender, evt) =>
    /// {
    ///     Console.WriteLine($"Detected: {evt.EventType} ({evt.Confidence:P1})");
    /// };
    /// session.FeedAudio(microphoneChunk);
    /// </code>
    /// </para>
    /// </remarks>
    private sealed class BEATsStreamingSession : IStreamingEventDetectionSession<T>
    {
        private readonly BEATs<T> _detector;
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

        public BEATsStreamingSession(
            BEATs<T> detector,
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

            // Initialize state for all event types
            foreach (var label in detector.ClassLabels)
            {
                _currentState[label] = detector.NumOps.Zero;
            }
        }

        public void FeedAudio(Tensor<T> audioChunk)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(BEATsStreamingSession));
            }

            // Collect events to raise outside the lock to prevent deadlock
            List<AudioEvent<T>>? eventsToRaise = null;

            lock (_lock)
            {
                if (_disposed)
                {
                    throw new ObjectDisposedException(nameof(BEATsStreamingSession));
                }

                // Accumulate audio samples
                for (int i = 0; i < audioChunk.Length; i++)
                {
                    _buffer.Add(audioChunk[i]);
                }

                // Process complete windows
                while (_buffer.Count >= _windowSamples)
                {
                    var window = new Tensor<T>([_windowSamples]);
                    for (int i = 0; i < _windowSamples; i++)
                    {
                        window[i] = _buffer[i];
                    }

                    // Extract features and classify
                    var melSpec = _detector._melSpectrogram?.Forward(window) ??
                        throw new InvalidOperationException("MelSpectrogram not initialized.");
                    var scores = _detector.ClassifyWindow(melSpec);

                    // Update state and detect events
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
                            eventsToRaise ??= new List<AudioEvent<T>>();
                            eventsToRaise.Add(evt);
                        }
                    }

                    // Advance buffer with overlap
                    int hopSamples = (int)(_windowSamples * (1 - _detector._options.WindowOverlap));
                    if (hopSamples <= 0) hopSamples = 1;
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
