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
/// BEATs (Audio Pre-Training with Acoustic Tokenizers) model for state-of-the-art audio
/// event detection and classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// BEATs (Chen et al., ICML 2023) achieves state-of-the-art results on multiple audio benchmarks:
/// <list type="bullet">
/// <item><b>AudioSet-2M</b>: 50.6% mAP (mean Average Precision) - the largest audio benchmark
/// with 2 million YouTube clips spanning 527 sound event classes</item>
/// <item><b>ESC-50</b>: 98.1% accuracy - a dataset of 2000 environmental sound clips
/// in 50 categories (rain, dog bark, clock tick, etc.)</item>
/// <item><b>AudioSet-Balanced</b>: 47.5% mAP - the class-balanced evaluation subset</item>
/// </list>
/// </para>
/// <para>
/// <b>Architecture:</b> BEATs adapts the Vision Transformer (ViT) architecture for audio spectrograms.
/// The key insight is treating a mel spectrogram like an image and processing it with patches:
/// <list type="number">
/// <item><b>Audio preprocessing</b>: Raw waveform (e.g., 16kHz mono) is converted to a 128-bin
/// log-mel spectrogram, creating a 2D time-frequency representation of the sound</item>
/// <item><b>Patch embedding</b>: The spectrogram is divided into non-overlapping 16x16 patches,
/// and each patch is linearly projected to a 768-dimensional embedding vector</item>
/// <item><b>Positional encoding</b>: Learnable positional embeddings are added so the Transformer
/// knows the spatial ordering of patches (which part of the spectrogram each patch came from)</item>
/// <item><b>Transformer encoder</b>: A 12-layer encoder with 12-head self-attention and GELU
/// feed-forward networks processes the patch sequence, learning relationships between different
/// time-frequency regions (e.g., a bark onset followed by harmonics = dog barking)</item>
/// <item><b>Classification head</b>: The encoded representations are pooled and projected through
/// a linear layer to produce per-class logits, with sigmoid activation for multi-label detection
/// (multiple sounds can occur simultaneously in real audio)</item>
/// </list>
/// </para>
/// <para>
/// <b>Training Strategy:</b> BEATs uses a novel iterative self-distillation framework that
/// alternates between two components:
/// <list type="number">
/// <item><b>Acoustic Tokenizer</b>: Converts each audio patch into a discrete token (one of 8192 codes).
/// This is similar to how text is tokenized into words, but for audio. The tokenizer is trained
/// using the audio model's representations from the previous iteration.</item>
/// <item><b>Masked Patch Prediction</b>: 75% of spectrogram patches are randomly masked, and the model
/// must predict the acoustic tokens assigned by the tokenizer. This forces the model to learn rich
/// audio representations by filling in the missing pieces, similar to BERT's masked language modeling.</item>
/// <item><b>Iteration</b>: The improved audio model produces better representations, which train a better
/// tokenizer, which provides better labels for the next round of pre-training. BEATs_iter3 (3 iterations)
/// achieves the best results.</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> BEATs is one of the most accurate models for identifying sounds in audio.
/// It can detect hundreds of different sounds simultaneously - speech, music, animals, machinery,
/// weather, and more - all from the same audio clip.
///
/// Here is how BEATs processes audio, step by step:
///
/// <b>Step 1 - Sound to picture:</b> Audio waves are converted to a "mel spectrogram" - think
/// of it as a heat map where the x-axis is time, the y-axis is pitch (frequency), and the
/// brightness shows how loud each pitch is at each moment. A dog bark would show short bright
/// bursts in the mid-frequency range, while a whistle would show a thin bright line at high frequency.
///
/// <b>Step 2 - Cut into tiles:</b> The spectrogram image is cut into small 16x16 pixel tiles
/// called "patches" (like puzzle pieces). Each patch captures about 0.16 seconds of a narrow
/// frequency band. A 10-second clip produces roughly 500 patches.
///
/// <b>Step 3 - Describe each tile:</b> Each small tile is converted into a list of 768 numbers
/// (a "vector") that describes its contents. This is done by a linear projection layer - essentially
/// a learned formula that extracts the most important features from each tile.
///
/// <b>Step 4 - Understand context:</b> This is where the magic happens. A Transformer encoder
/// (the same architecture behind ChatGPT and modern AI) looks at ALL tiles simultaneously and
/// figures out how they relate to each other. For example, it learns that:
/// - A particular frequency pattern repeated rhythmically = drums
/// - Formant patterns in the 200-4000 Hz range with pauses = speech
/// - Broadband energy with no harmonic structure = wind or rain
/// The Transformer has 12 layers, each adding deeper understanding.
///
/// <b>Step 5 - Make predictions:</b> The Transformer's output is averaged across all tiles and
/// fed through a classification head that outputs a probability for each of the 527 possible
/// sounds. Multiple sounds can be detected at once (e.g., "speech: 95%, music: 40%, rain: 15%").
///
/// <b>Why BEATs is special:</b> Most previous audio models learned by being told what sounds are
/// in training clips. BEATs instead learns by playing a game with itself: it masks (hides) 75%
/// of the spectrogram tiles and tries to guess what was hidden. This "self-supervised" approach
/// means BEATs can learn from millions of unlabeled audio clips, making it much more accurate.
///
/// <b>Usage with a pre-trained ONNX model (recommended for most users):</b>
/// <code>
/// // Load a pre-trained BEATs model for instant sound detection
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 768, outputSize: 527);
/// var beats = new BEATs&lt;float&gt;(architecture, "beats_iter3.onnx");
///
/// // Detect sounds in an audio clip
/// var result = beats.Detect(audioWaveform);
/// foreach (var evt in result.Events)
/// {
///     Console.WriteLine($"{evt.EventType}: {evt.Confidence:P1} at {evt.StartTime:F2}s-{evt.EndTime:F2}s");
/// }
/// // Output: "Speech: 95.2% at 0.00s-3.50s"
/// //         "Music: 42.1% at 2.00s-8.00s"
/// //         "Dog barking: 87.3% at 5.20s-5.80s"
/// </code>
///
/// <b>Usage with native training (for researchers and custom datasets):</b>
/// <code>
/// // Train a custom BEATs model on your own audio categories
/// var options = new BEATsOptions
/// {
///     EmbeddingDim = 768,
///     NumEncoderLayers = 12,
///     CustomLabels = new[] { "machine_normal", "machine_anomaly", "silence" }
/// };
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 768, outputSize: 3);
/// var beats = new BEATs&lt;float&gt;(architecture, options);
///
/// // Train on your labeled data
/// beats.Train(spectrogramFeatures, multiHotLabels);
///
/// // Detect on new audio
/// var result = beats.Detect(testAudio);
/// </code>
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "BEATs: Audio Pre-Training with Acoustic Tokenizers" (Chen et al., ICML 2023)</item>
/// <item>Repository: https://github.com/microsoft/unilm/tree/master/beats</item>
/// <item>AudioSet: https://research.google.com/audioset/ (the primary evaluation benchmark)</item>
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
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    /// <summary>
    /// AudioSet-527 standard event labels used by BEATs pre-trained models.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is a representative subset of AudioSet's 527 classes. The full AudioSet ontology
    /// (developed by Google Research) covers a comprehensive taxonomy of everyday sounds,
    /// organized hierarchically. AudioSet was created by manually labeling 2 million 10-second
    /// YouTube video clips, making it the largest public audio classification dataset.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> These labels are the "vocabulary" of sounds that BEATs can recognize.
    /// When BEATs analyzes audio, it outputs a confidence score (0-100%) for each of these labels.
    /// Multiple labels can be active simultaneously (e.g., "Speech" + "Music" + "Rain" if someone
    /// is talking over music during a rainstorm). You can also provide your own custom labels
    /// via <see cref="BEATsOptions.CustomLabels"/> when training on a specialized dataset.
    /// </para>
    /// </remarks>
    public static readonly string[] AudioSetLabels =
    [
        // Human sounds - speech, vocalizations, and body sounds
        "Speech", "Male speech", "Female speech", "Child speech",
        "Conversation", "Narration", "Laughter", "Baby laughter",
        "Crying", "Baby cry", "Whimper", "Cough", "Sneeze",
        "Breathing", "Sigh", "Yawn", "Snoring", "Gasp", "Hiccup",
        "Clapping", "Cheering", "Crowd", "Footsteps", "Chewing",

        // Musical instruments - acoustic, electric, and electronic
        "Music", "Musical instrument", "Guitar", "Electric guitar",
        "Bass guitar", "Acoustic guitar", "Piano", "Keyboard",
        "Organ", "Synthesizer", "Drums", "Drum kit", "Snare drum",
        "Bass drum", "Hi-hat", "Cymbal", "Tambourine",
        "Violin", "Cello", "Harp", "Flute", "Clarinet", "Saxophone",
        "Trumpet", "Trombone", "French horn", "Harmonica", "Accordion",

        // Singing and vocal music
        "Singing", "Choir", "Rapping", "Humming", "Whistling",
        "Yodeling",

        // Animal sounds - domestic, wild, and insects
        "Dog", "Dog barking", "Dog howl", "Dog whimper",
        "Cat", "Cat meowing", "Cat purring", "Cat hissing",
        "Bird", "Bird song", "Bird call", "Chirp", "Tweet",
        "Crow", "Rooster", "Duck", "Goose", "Owl",
        "Horse", "Cow", "Pig", "Sheep", "Goat",
        "Frog", "Cricket", "Insect", "Bee", "Mosquito",

        // Nature and environment - weather and natural phenomena
        "Rain", "Raindrop", "Thunder", "Thunderstorm",
        "Wind", "Howling wind", "Rustling leaves",
        "Water", "Stream", "Waterfall", "Ocean", "Waves",
        "Fire", "Crackling fire",

        // Vehicles and transport - cars, trains, aircraft
        "Car", "Car engine", "Car horn", "Tire squeal",
        "Truck", "Bus", "Motorcycle", "Bicycle",
        "Train", "Train horn", "Train wheels",
        "Airplane", "Helicopter", "Jet engine",
        "Boat", "Ship horn", "Siren", "Ambulance siren",
        "Police siren", "Fire truck siren",

        // Household sounds - appliances and everyday objects
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
    /// <param name="architecture">Neural network architecture defining input/output dimensions.
    /// For BEATs, inputFeatures should match the embedding dimension (768 for BEATs_iter3)
    /// and outputSize should match the number of class labels (527 for AudioSet).</param>
    /// <param name="modelPath">Path to the pre-trained ONNX model file. BEATs models are
    /// available from the official Microsoft repository. The ONNX file contains all the
    /// pre-trained weights from iterative self-distillation training.</param>
    /// <param name="options">BEATs configuration options controlling audio preprocessing
    /// (sample rate, mel bands, FFT settings), detection behavior (threshold, window size),
    /// and custom labels. If null, uses BEATs_iter3 paper defaults.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you have a pre-trained BEATs model
    /// file in ONNX format. ONNX (Open Neural Network Exchange) is a standard format for
    /// storing trained neural network weights that works across different frameworks.
    /// The model will be ready for inference immediately - no training needed.
    ///
    /// This is the recommended approach for most users because:
    /// <list type="bullet">
    /// <item>Pre-trained BEATs models already achieve state-of-the-art accuracy</item>
    /// <item>Training from scratch requires millions of audio clips and significant compute</item>
    /// <item>ONNX inference is optimized and fast (often with GPU acceleration)</item>
    /// </list>
    ///
    /// Example:
    /// <code>
    /// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 768, outputSize: 527);
    /// var beats = new BEATs&lt;float&gt;(arch, "path/to/beats_iter3.onnx");
    /// var result = beats.Detect(audioTensor);
    /// </code>
    /// </para>
    /// </remarks>
    public BEATs(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        BEATsOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options = options ?? new BEATsOptions();
        _options.ModelPath = modelPath;
        _useNativeMode = false;

        base.SampleRate = _options.SampleRate;
        base.NumMels = _options.NumMels;

        // Initialize mel spectrogram for converting raw audio waveforms to time-frequency
        // representations that BEATs expects as input. Uses 128 mel bands (finer frequency
        // resolution than typical 64/80 bands) to capture subtle spectral differences.
        _melSpectrogram = new MelSpectrogram<T>(
            sampleRate: _options.SampleRate,
            nMels: _options.NumMels,
            nFft: _options.FftSize,
            hopLength: _options.HopLength,
            fMin: _options.FMin,
            fMax: _options.FMax,
            logMel: true);

        // Load the pre-trained ONNX model containing all BEATs weights
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);

        _options.ModelPath = modelPath;

        // Set class labels: either user-provided custom labels or the standard AudioSet-527 set
        ClassLabels = _options.CustomLabels ?? AudioSetLabels;

        InitializeLayers();
    }

    /// <summary>
    /// Creates a BEATs model for native training mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture. For BEATs, inputFeatures
    /// should be set to the embedding dimension (768 for the standard model) and outputSize
    /// to the number of target classes. You can also pass custom layers via
    /// <see cref="NeuralNetworkArchitecture{T}.Layers"/> to override the default BEATs architecture.</param>
    /// <param name="options">BEATs configuration options. Controls all hyperparameters
    /// including the Transformer dimensions, dropout rates, and audio preprocessing.
    /// If null, uses BEATs_iter3 paper defaults (768-dim, 12 layers, 12 heads).</param>
    /// <param name="optimizer">Optional custom optimizer for training. BEATs uses AdamW
    /// (Adam with weight decay) with a peak learning rate of 5e-4, linear warm-up over
    /// 32000 steps, and cosine decay. If null, defaults to AdamW with standard settings.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you want to train a BEATs model from
    /// scratch on your own audio dataset. Training from scratch is useful when:
    /// <list type="bullet">
    /// <item>You have a specialized domain (industrial sounds, medical auscultation, etc.)</item>
    /// <item>Your target classes are very different from AudioSet's 527 categories</item>
    /// <item>You want to experiment with different model sizes or architectures</item>
    /// </list>
    ///
    /// Note: Training requires labeled audio data. Each training example consists of:
    /// <list type="bullet">
    /// <item><b>Input</b>: A mel spectrogram tensor (from audio preprocessing)</item>
    /// <item><b>Labels</b>: A multi-hot vector where 1 = sound is present, 0 = absent</item>
    /// </list>
    ///
    /// Example:
    /// <code>
    /// var options = new BEATsOptions
    /// {
    ///     EmbeddingDim = 768,
    ///     NumEncoderLayers = 12,
    ///     CustomLabels = new[] { "normal_operation", "bearing_fault", "motor_overload" }
    /// };
    /// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 768, outputSize: 3);
    /// var beats = new BEATs&lt;float&gt;(arch, options);
    /// beats.Train(spectrogramFeatures, labels);
    /// </code>
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

        // Set class labels: custom labels for specialized tasks, or AudioSet-527 for general use
        ClassLabels = _options.CustomLabels ?? AudioSetLabels;

        // Initialize mel spectrogram for audio-to-spectrogram conversion during training
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
    /// Creates a BEATs model asynchronously with optional model download.
    /// </summary>
    /// <param name="options">BEATs configuration options. If <see cref="BEATsOptions.ModelPath"/>
    /// is set, loads that file directly. Otherwise, downloads a pre-trained BEATs model
    /// from the model registry.</param>
    /// <param name="progress">Optional progress reporter for tracking download progress.
    /// Reports values from 0.0 (starting) to 1.0 (complete).</param>
    /// <param name="cancellationToken">Cancellation token to abort the download operation.</param>
    /// <returns>A configured BEATs model ready for inference.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the easiest way to get started with BEATs. It handles
    /// downloading the pre-trained model file and setting everything up automatically.
    /// After calling this method, the model is ready to detect sounds in audio.
    ///
    /// Example:
    /// <code>
    /// // Download and load a pre-trained BEATs model
    /// var beats = await BEATs&lt;float&gt;.CreateAsync(
    ///     progress: new Progress&lt;double&gt;(p => Console.Write($"\rDownloading: {p:P0}")));
    ///
    /// // Detect sounds in audio
    /// var result = beats.Detect(audioWaveform);
    /// Console.WriteLine($"Detected {result.Events.Count} events in {result.TotalDuration:F1}s");
    /// </code>
    ///
    /// The downloaded model is cached locally, so subsequent calls will be fast.
    /// </para>
    /// </remarks>
    internal static async Task<BEATs<T>> CreateAsync(
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
    /// <remarks>
    /// <para>
    /// For a default BEATs model, this returns the AudioSet-527 label set covering categories
    /// like human speech, musical instruments, animal sounds, vehicles, household sounds, and more.
    /// When <see cref="BEATsOptions.CustomLabels"/> is provided, returns those custom labels instead.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This property tells you what sounds the model knows about. Each string
    /// in the list is a sound category. When you call <see cref="Detect"/>, the model checks for
    /// every one of these sounds and reports which ones it hears with their confidence scores.
    /// </para>
    /// </remarks>
    public IReadOnlyList<string> SupportedEvents => ClassLabels;

    /// <summary>
    /// Gets the event labels (alias for <see cref="SupportedEvents"/> for API compatibility).
    /// </summary>
    public IReadOnlyList<string> EventLabels => ClassLabels;

    /// <summary>
    /// Gets the time resolution for event detection in seconds.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The time resolution determines the granularity of event boundaries. It equals
    /// <c>WindowSize * (1 - WindowOverlap)</c>. With default BEATs settings (10s window,
    /// 0.5 overlap), the time resolution is 5.0 seconds, meaning event boundaries are
    /// determined at 5-second intervals.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells you how precisely the model can pinpoint when a sound
    /// starts and stops. A smaller value means more precise timing but requires more computation.
    /// For many applications (smart home, security monitoring), 5-second resolution is sufficient.
    /// For precise onset detection (music transcription), you may want smaller windows.
    /// </para>
    /// </remarks>
    public double TimeResolution => _options.WindowSize * (1 - _options.WindowOverlap);

    #endregion

    #region IAudioEventDetector Methods

    /// <summary>
    /// Detects all audio events in the audio stream using the default confidence threshold.
    /// </summary>
    /// <param name="audio">Raw audio waveform tensor with shape [samples]. The audio should be
    /// mono (single channel) at the sample rate specified in <see cref="BEATsOptions.SampleRate"/>
    /// (default: 16000 Hz). Values should be normalized to the range [-1.0, 1.0].</param>
    /// <returns>
    /// An <see cref="AudioEventResult{T}"/> containing all detected events that exceed the
    /// confidence threshold (default: 0.3), along with timing information, event statistics,
    /// and the total audio duration.
    /// </returns>
    /// <remarks>
    /// <para>
    /// <b>How BEATs detection works internally:</b>
    /// <list type="number">
    /// <item>The audio is split into overlapping windows (default: 10s windows with 50% overlap)</item>
    /// <item>Each window is converted to a 128-bin log-mel spectrogram</item>
    /// <item>The spectrogram is fed through the BEATs Transformer encoder</item>
    /// <item>Sigmoid activation converts logits to per-class probabilities [0, 1]</item>
    /// <item>Events above the threshold are collected with their time positions</item>
    /// <item>Overlapping detections of the same class are merged into continuous events</item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the main method for detecting sounds. Pass in audio data
    /// and get back a list of all sounds the model detected, including what they are, how
    /// confident the model is, and when they occurred.
    ///
    /// Example:
    /// <code>
    /// var result = beats.Detect(audioTensor);
    ///
    /// // Print all detected events
    /// foreach (var evt in result.Events)
    /// {
    ///     Console.WriteLine(
    ///         $"  {evt.EventType}: {evt.Confidence:P1} " +
    ///         $"({evt.StartTime:F1}s - {evt.EndTime:F1}s)");
    /// }
    ///
    /// // Check for specific sounds
    /// var dogEvents = result.GetEventsByType("Dog barking");
    /// if (dogEvents.Any())
    /// {
    ///     Console.WriteLine("A dog was detected barking!");
    /// }
    /// </code>
    ///
    /// The threshold (default 0.3 = 30% confidence) controls sensitivity:
    /// <list type="bullet">
    /// <item>Lower threshold (e.g., 0.1): Detects more sounds but with more false positives</item>
    /// <item>Higher threshold (e.g., 0.7): Only reports sounds the model is very confident about</item>
    /// </list>
    /// </para>
    /// </remarks>
    public AudioEventResult<T> Detect(Tensor<T> audio)
    {
        ThrowIfDisposed();
        return Detect(audio, NumOps.FromDouble(_options.Threshold));
    }

    /// <summary>
    /// Detects audio events with a custom confidence threshold.
    /// </summary>
    /// <param name="audio">Raw audio waveform tensor [samples], mono, at the configured sample rate.</param>
    /// <param name="threshold">Confidence threshold in range [0, 1]. Only events with confidence
    /// at or above this value are included in the results. BEATs uses sigmoid output, so
    /// each class independently produces a probability. A value of 0.5 means "more likely
    /// than not" for that class.</param>
    /// <returns>Detection result containing events above the threshold.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This overload lets you control how sensitive the detection is.
    /// BEATs performs multi-label classification using sigmoid activation, which means each
    /// sound class gets an independent probability score between 0% and 100%. The threshold
    /// determines the minimum score required for a sound to be reported.
    ///
    /// Guidelines for choosing a threshold:
    /// <list type="bullet">
    /// <item><b>0.1-0.2</b>: Very sensitive - catches most sounds but may report sounds that aren't there</item>
    /// <item><b>0.3</b>: Balanced (default) - good trade-off between catching real sounds and avoiding false alarms</item>
    /// <item><b>0.5-0.7</b>: Conservative - only reports sounds the model is quite confident about</item>
    /// <item><b>0.8-0.9</b>: Very conservative - only reports near-certain detections</item>
    /// </list>
    /// </para>
    /// </remarks>
    public AudioEventResult<T> Detect(Tensor<T> audio, T threshold)
    {
        ThrowIfDisposed();

        double thresholdValue = NumOps.ToDouble(threshold);
        double totalDuration = audio.Length / (double)_options.SampleRate;

        // Split audio into overlapping windows for frame-level analysis.
        // BEATs processes fixed-length segments (default 10s), so longer audio
        // is split into overlapping windows to ensure temporal coverage.
        var windows = SplitIntoWindows(audio);
        var allEvents = new List<AudioEvent<T>>();

        for (int windowIdx = 0; windowIdx < windows.Count; windowIdx++)
        {
            var window = windows[windowIdx];
            double startTime = windowIdx * TimeResolution;

            // Convert raw audio window to mel spectrogram.
            // This transforms the waveform into a 2D time-frequency representation
            // that BEATs' Transformer encoder expects as input.
            var melSpec = _melSpectrogram?.Forward(window) ??
                throw new InvalidOperationException("MelSpectrogram not initialized.");

            // Run the BEATs model (ONNX or native) to get per-class probabilities.
            // The model applies its full pipeline: patch embedding -> Transformer -> classification head.
            var scores = ClassifyWindow(melSpec);

            // Collect all events above the confidence threshold for this window.
            // BEATs is multi-label, so multiple classes can be active simultaneously.
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

        // Merge overlapping events of the same class into continuous segments.
        // Due to window overlap, the same sound may be detected in adjacent windows.
        // Merging combines these into a single event with the highest confidence.
        var mergedEvents = MergeEvents(allEvents);

        // Compute per-class statistics (count, total duration, avg/max confidence)
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
    /// Detects audio events asynchronously using a background thread.
    /// </summary>
    /// <param name="audio">Raw audio waveform tensor [samples].</param>
    /// <param name="cancellationToken">Cancellation token to abort the detection.</param>
    /// <returns>Detection result (same as <see cref="Detect(Tensor{T})"/> but non-blocking).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the async version of <see cref="Detect(Tensor{T})"/>.
    /// Use it when you don't want to block the main thread while BEATs processes audio.
    /// This is especially useful in UI applications (WPF, WinForms, MAUI) where blocking
    /// the main thread would freeze the interface, or in web APIs where you need to handle
    /// multiple requests concurrently.
    ///
    /// Example:
    /// <code>
    /// // Non-blocking detection - UI stays responsive
    /// var result = await beats.DetectAsync(audioTensor);
    /// UpdateUI(result.Events); // Update UI on the main thread
    /// </code>
    /// </para>
    /// </remarks>
    public Task<AudioEventResult<T>> DetectAsync(
        Tensor<T> audio,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Detect(audio), cancellationToken);
    }

    /// <summary>
    /// Detects only specific event types, filtering out all others.
    /// </summary>
    /// <param name="audio">Raw audio waveform tensor [samples].</param>
    /// <param name="eventTypes">Event types to detect. Matching is case-insensitive.
    /// Only events whose <see cref="AudioEvent{T}.EventType"/> matches one of these
    /// strings will be included in the results.</param>
    /// <returns>Detection result filtered to only the specified event types.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you only care about specific sounds. The model
    /// still analyzes all sounds internally (BEATs always produces scores for all classes),
    /// but the results are filtered to only include the sounds you asked for. This is useful
    /// when you have a specific use case like:
    /// <list type="bullet">
    /// <item><b>Smart home</b>: Only detect "Doorbell", "Dog barking", "Glass breaking"</item>
    /// <item><b>Wildlife monitoring</b>: Only detect "Bird song", "Frog", "Cricket"</item>
    /// <item><b>Safety</b>: Only detect "Gunshot", "Explosion", "Siren"</item>
    /// </list>
    ///
    /// Example:
    /// <code>
    /// var safetyEvents = new[] { "Gunshot", "Explosion", "Glass breaking", "Siren" };
    /// var result = beats.DetectSpecific(audio, safetyEvents);
    /// if (result.Events.Count > 0)
    /// {
    ///     TriggerAlert(result.Events.First().EventType);
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public AudioEventResult<T> DetectSpecific(Tensor<T> audio, IReadOnlyList<string> eventTypes)
    {
        return DetectSpecific(audio, eventTypes, NumOps.FromDouble(_options.Threshold));
    }

    /// <summary>
    /// Detects specific event types with a custom confidence threshold.
    /// </summary>
    /// <param name="audio">Raw audio waveform tensor [samples].</param>
    /// <param name="eventTypes">Event types to detect (case-insensitive).</param>
    /// <param name="threshold">Custom confidence threshold [0, 1].</param>
    /// <returns>Detection result filtered to specified event types.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Combines targeted detection with custom sensitivity. For example,
    /// you might want to be very sensitive to glass breaking (low threshold = 0.1) but only
    /// report speech when very confident (high threshold = 0.7). In that case, call this
    /// method twice with different thresholds and combine the results.
    /// </para>
    /// </remarks>
    public AudioEventResult<T> DetectSpecific(Tensor<T> audio, IReadOnlyList<string> eventTypes, T threshold)
    {
        var result = Detect(audio, threshold);

        // Filter to only the requested event types using case-insensitive matching
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
    /// Gets the raw frame-level event probabilities for all classes without applying a threshold.
    /// </summary>
    /// <param name="audio">Raw audio waveform tensor [samples].</param>
    /// <returns>
    /// A 2D tensor of shape [time_frames, num_events] where each value is a sigmoid probability
    /// in [0, 1]. Row i contains the probabilities for time frame i, and column j contains the
    /// probability of event class j being active. The number of time frames depends on the audio
    /// length and window settings.
    /// </returns>
    /// <remarks>
    /// <para>
    /// <b>How this works in BEATs:</b> Each time frame corresponds to one analysis window.
    /// For each window, BEATs computes a mel spectrogram, runs it through the Transformer
    /// encoder, and applies sigmoid to the classification head output. The result is a matrix
    /// showing how likely each sound is at each point in time.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This gives you the "full picture" of what BEATs detected,
    /// without any filtering. It returns a matrix where:
    /// <list type="bullet">
    /// <item>Each row represents a point in time</item>
    /// <item>Each column represents a sound category</item>
    /// <item>Each value (0.0 to 1.0) represents how confident the model is that the sound is present</item>
    /// </list>
    ///
    /// This is useful for:
    /// <list type="bullet">
    /// <item><b>Visualization</b>: Create heatmaps showing which sounds are present over time</item>
    /// <item><b>Custom thresholding</b>: Apply different thresholds per class</item>
    /// <item><b>Analysis</b>: Study temporal patterns in audio (e.g., speech-music alternation)</item>
    /// <item><b>Post-processing</b>: Apply your own smoothing or median filtering to the probabilities</item>
    /// </list>
    ///
    /// Example:
    /// <code>
    /// var probMatrix = beats.GetEventProbabilities(audioTensor);
    /// // probMatrix[0, 5] = probability of ClassLabels[5] in the first window
    /// // probMatrix[3, 10] = probability of ClassLabels[10] in the fourth window
    ///
    /// // Find which sounds exceed 50% confidence in frame 0:
    /// for (int c = 0; c &lt; beats.SupportedEvents.Count; c++)
    /// {
    ///     double prob = NumOps.ToDouble(probMatrix[0, c]);
    ///     if (prob > 0.5)
    ///         Console.WriteLine($"{beats.SupportedEvents[c]}: {prob:P1}");
    /// }
    /// </code>
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
    /// <returns>A streaming session that processes audio chunks in real-time and emits
    /// events as they are detected.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Streaming mode is for real-time audio monitoring. Instead of
    /// analyzing a complete audio file at once, you feed audio chunk by chunk (e.g., from a
    /// microphone) and get detections as they happen. This is essential for applications like:
    /// <list type="bullet">
    /// <item><b>Live security monitoring</b>: Detect gunshots or glass breaking as they happen</item>
    /// <item><b>Smart speakers</b>: Continuously listen for specific wake words or sounds</item>
    /// <item><b>Wildlife cameras</b>: Alert when a specific animal call is detected</item>
    /// <item><b>Industrial monitoring</b>: Detect machine malfunction sounds in real-time</item>
    /// </list>
    ///
    /// The session maintains an internal buffer. As you feed audio chunks, it accumulates
    /// samples until a complete analysis window (10s by default) is available, runs BEATs
    /// detection, and emits events through the <see cref="IStreamingEventDetectionSession{T}.EventDetected"/>
    /// event handler.
    ///
    /// Example:
    /// <code>
    /// using var session = beats.StartStreamingSession();
    ///
    /// // Subscribe to real-time detections
    /// session.EventDetected += (sender, evt) =>
    /// {
    ///     Console.WriteLine($"[{evt.StartTime:F1}s] {evt.EventType}: {evt.Confidence:P1}");
    /// };
    ///
    /// // Feed audio chunks as they arrive from the microphone
    /// while (recording)
    /// {
    ///     var chunk = microphone.ReadSamples(4096);
    ///     session.FeedAudio(chunk);
    /// }
    ///
    /// // Or poll for events manually
    /// var recentEvents = session.GetNewEvents();
    /// </code>
    /// </para>
    /// </remarks>
    public IStreamingEventDetectionSession<T> StartStreamingSession()
    {
        return StartStreamingSession(_options.SampleRate, NumOps.FromDouble(_options.Threshold));
    }

    /// <summary>
    /// Starts a streaming event detection session with custom sample rate and threshold.
    /// </summary>
    /// <param name="sampleRate">Sample rate of incoming audio in Hz. Must match the actual
    /// sample rate of the audio chunks you will feed to the session.</param>
    /// <param name="threshold">Confidence threshold for event detection [0, 1].</param>
    /// <returns>A streaming session configured with the specified settings.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this overload when your audio source has a different sample
    /// rate than the default (16000 Hz), or when you want to use a different confidence
    /// threshold than the default (0.3). For example, if your microphone captures at 44100 Hz
    /// and you want to be very sensitive:
    /// <code>
    /// using var session = beats.StartStreamingSession(
    ///     sampleRate: 44100,
    ///     threshold: NumOps.FromDouble(0.15));
    /// </code>
    /// </para>
    /// </remarks>
    public IStreamingEventDetectionSession<T> StartStreamingSession(int sampleRate, T threshold)
    {
        if (sampleRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(sampleRate), "Sample rate must be positive.");
        return new BEATsStreamingSession(this, sampleRate, threshold);
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <summary>
    /// Initializes the BEATs neural network layers using either custom user-provided layers
    /// or the paper-standard BEATs architecture from <see cref="LayerHelper{T}.CreateDefaultBEATsLayers"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Layer initialization follows this priority:</b>
    /// <list type="number">
    /// <item>If the model is in ONNX mode, no native layers are needed (inference uses ONNX runtime)</item>
    /// <item>If the user provided custom layers via <see cref="NeuralNetworkArchitecture{T}.Layers"/>,
    /// those layers are used directly, allowing full architectural customization</item>
    /// <item>Otherwise, the standard BEATs_iter3 architecture is created using
    /// <see cref="LayerHelper{T}.CreateDefaultBEATsLayers"/> with parameters from the options:
    ///   <list type="bullet">
    ///   <item>Patch projection: patchSize^2 (256) -> embeddingDim (768)</item>
    ///   <item>Pre-layer normalization</item>
    ///   <item>Positional encoding for patch sequence ordering</item>
    ///   <item>12 Transformer encoder layers (multi-head attention + FFN)</item>
    ///   <item>Post-layer normalization</item>
    ///   <item>Two-layer classification MLP: embeddingDim -> embeddingDim -> numClasses</item>
    ///   </list>
    /// </item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method builds the neural network's internal structure.
    /// You normally don't call this directly - it's called automatically by the constructor.
    /// The network has these main parts:
    /// <list type="bullet">
    /// <item><b>Patch projection</b>: Converts each spectrogram tile into a 768-number description</item>
    /// <item><b>Transformer layers</b>: 12 layers that help the model understand how different
    /// parts of the audio relate to each other (like how a bark onset relates to bark harmonics)</item>
    /// <item><b>Classification head</b>: Converts the model's understanding into probabilities
    /// for each sound category</item>
    /// </list>
    /// If you want to use a different architecture, pass custom layers when creating
    /// the <see cref="NeuralNetworkArchitecture{T}"/>.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;

        // Priority 1: Use custom layers if the user provided them via architecture
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            // Priority 2: Create the standard BEATs_iter3 architecture from the paper
            // All parameters are taken from the options, which default to paper values
            Layers.AddRange(LayerHelper<T>.CreateDefaultBEATsLayers(
                patchFeatureSize: _options.PatchSize * _options.PatchSize,
                embeddingDim: _options.EmbeddingDim,
                numEncoderLayers: _options.NumEncoderLayers,
                numAttentionHeads: _options.NumAttentionHeads,
                feedForwardDim: _options.FeedForwardDim,
                numClasses: ClassLabels.Count,
                maxSequenceLength: 600,
                dropoutRate: _options.DropoutRate));
        }
    }

    /// <summary>
    /// Runs BEATs inference on the given input tensor.
    /// </summary>
    /// <param name="input">Preprocessed input features. In ONNX mode, this should match the
    /// ONNX model's expected input format. In native mode, this is a flattened mel spectrogram
    /// that flows through the BEATs layer stack (patch projection -> Transformer -> classification).</param>
    /// <returns>Model output tensor containing per-class logits (before sigmoid activation).
    /// The length equals the number of class labels.</returns>
    /// <remarks>
    /// <para>
    /// <b>How BEATs prediction works:</b>
    /// <list type="bullet">
    /// <item><b>ONNX mode</b>: The input is passed directly to the ONNX runtime, which executes
    /// the pre-trained BEATs model using optimized operators (potentially with GPU acceleration).
    /// This is the fastest inference path.</item>
    /// <item><b>Native mode</b>: The input flows through the layer stack created by
    /// <see cref="InitializeLayers"/>. Each layer transforms the data sequentially:
    /// patch projection embeds features, Transformer layers add contextual understanding,
    /// and the classification head produces per-class scores.</item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the core inference method. It takes preprocessed audio
    /// features and outputs a prediction. You typically don't call this directly - use
    /// <see cref="Detect(Tensor{T})"/> instead, which handles audio preprocessing, windowing,
    /// sigmoid activation, thresholding, and event merging automatically.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();

        if (IsOnnxMode && OnnxEncoder is not null)
        {
            return OnnxEncoder.Run(input);
        }

        // Native mode: run through the BEATs layer stack
        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }
        return current;
    }

    /// <summary>
    /// Trains the BEATs model on a single input-target pair using backpropagation.
    /// </summary>
    /// <param name="input">Input audio features (mel spectrogram), typically flattened.
    /// The features should already be preprocessed (e.g., via <see cref="PreprocessAudio"/>).</param>
    /// <param name="expected">Expected output labels as a multi-hot encoded vector. Each position
    /// corresponds to a class label: 1.0 means the sound is present, 0.0 means absent.
    /// For example, if classes are ["speech", "music", "noise"] and the audio contains speech
    /// and music, the expected vector would be [1.0, 1.0, 0.0].</param>
    /// <exception cref="NotSupportedException">Thrown when called in ONNX mode, since pre-trained
    /// ONNX models cannot be fine-tuned through this interface.</exception>
    /// <remarks>
    /// <para>
    /// <b>BEATs training procedure:</b>
    /// <list type="number">
    /// <item><b>Training mode enabled</b>: Activates dropout layers (randomly disabling 10% of neurons
    /// to prevent overfitting) and sets batch normalization to training statistics</item>
    /// <item><b>Forward pass</b>: Input flows through the full BEATs stack (patch projection ->
    /// Transformer encoder -> classification head) to produce per-class logits</item>
    /// <item><b>Loss computation</b>: Binary cross-entropy loss measures how far the predicted
    /// probabilities are from the true labels. For multi-label classification, each class is
    /// treated as an independent binary prediction</item>
    /// <item><b>Backward pass</b>: Gradients are computed and propagated back through every layer,
    /// from the classification head all the way to the patch projection</item>
    /// <item><b>Parameter update</b>: The AdamW optimizer adjusts all weights based on the gradients,
    /// using momentum and adaptive learning rates for stable convergence</item>
    /// <item><b>Inference mode restored</b>: Dropout is disabled and batch norm uses running statistics</item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Training teaches the model to recognize sounds by showing it examples.
    /// For each example:
    /// <list type="bullet">
    /// <item>The model makes a prediction (forward pass)</item>
    /// <item>The prediction is compared to the correct answer (loss computation)</item>
    /// <item>The model figures out how to improve (backward pass / gradient computation)</item>
    /// <item>The model's internal numbers are adjusted slightly (parameter update)</item>
    /// </list>
    ///
    /// This process is repeated thousands of times with different audio examples.
    /// After sufficient training, the model learns to generalize and detect sounds it hasn't
    /// seen before. BEATs typically needs hundreds of thousands of training examples for good
    /// performance, which is why using a pre-trained model (ONNX mode) is recommended for most users.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
        {
            throw new NotSupportedException(
                "Training is not supported in ONNX mode. Create a BEATs model " +
                "without a modelPath parameter to train natively.");
        }

        // Enable training mode: activates dropout and sets batch norm to use batch statistics
        SetTrainingMode(true);

        // Forward pass through the full BEATs layer stack
        var output = Predict(input);

        // Compute gradient for backward pass
        var gradient = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());

        // Backward pass: propagate gradients from classification head back through
        // Transformer encoder to patch projection, computing parameter gradients
        var gradientTensor = Tensor<T>.FromVector(gradient);
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradientTensor = Layers[i].Backward(gradientTensor);
        }

        // Update all parameters using AdamW optimizer
        _optimizer?.UpdateParameters(Layers);

        // Restore inference mode: disables dropout, uses running statistics for batch norm
        SetTrainingMode(false);
    }

    /// <summary>
    /// Updates all network parameters from a flattened parameter vector.
    /// </summary>
    /// <param name="parameters">A flat vector containing all model parameters concatenated.
    /// The order matches the layer stack: patch projection weights first, then each Transformer
    /// layer's attention and FFN weights, then the classification head weights.</param>
    /// <exception cref="NotSupportedException">Thrown in ONNX mode since parameters are stored
    /// in the ONNX runtime and cannot be modified through this interface.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A neural network's "knowledge" is stored as millions of numbers
    /// (parameters/weights). This method replaces all those numbers at once from a single flat
    /// list. It's used by advanced optimizers that operate on the full parameter vector rather
    /// than individual layers, and for restoring model state from a checkpoint.
    /// </para>
    /// </remarks>
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
    /// Preprocesses raw audio into a log-mel spectrogram suitable for BEATs input.
    /// </summary>
    /// <param name="rawAudio">Raw audio waveform tensor [samples]. Expected to be mono audio
    /// at the configured sample rate (default: 16000 Hz) with values in [-1.0, 1.0].</param>
    /// <returns>A log-mel spectrogram tensor [time_frames, num_mels] where num_mels is 128
    /// by default. Each frame represents a short time slice of the audio, and each mel bin
    /// represents a perceptual frequency band. Values are in log scale for better dynamic range.</returns>
    /// <remarks>
    /// <para>
    /// <b>What happens during preprocessing:</b>
    /// <list type="number">
    /// <item>The audio waveform is divided into overlapping frames using a window function</item>
    /// <item>Each frame undergoes Short-Time Fourier Transform (STFT) to get frequency content</item>
    /// <item>The frequency magnitudes are mapped to 128 mel-spaced filter banks, which model
    /// how humans perceive pitch (we are more sensitive to differences at low frequencies)</item>
    /// <item>Log compression is applied (similar to how human ears perceive loudness logarithmically)</item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This converts raw sound waves into the "picture" format that BEATs
    /// understands. Think of it like converting a song into sheet music - the raw audio is like
    /// hearing the music, and the mel spectrogram is like seeing it written down with pitch on
    /// one axis and time on the other. BEATs then "reads" this picture to identify what sounds
    /// are present.
    /// </para>
    /// </remarks>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (_melSpectrogram is null)
            throw new InvalidOperationException("MelSpectrogram not initialized.");

        return _melSpectrogram.Forward(rawAudio);
    }

    /// <summary>
    /// Post-processes BEATs model output by applying element-wise sigmoid activation
    /// for multi-label classification.
    /// </summary>
    /// <param name="modelOutput">Raw logits from the classification head. These are unbounded
    /// real numbers where higher values indicate greater confidence. In the BEATs architecture,
    /// these come from the final linear projection layer.</param>
    /// <returns>A tensor of the same shape with values squashed to the [0, 1] range via
    /// the sigmoid function: sigma(x) = 1 / (1 + e^(-x)). Each value represents the
    /// independent probability that the corresponding sound class is present.</returns>
    /// <remarks>
    /// <para>
    /// <b>Why sigmoid and not softmax?</b> BEATs uses sigmoid (not softmax) because audio event
    /// detection is a <b>multi-label</b> problem: multiple sounds can occur simultaneously.
    /// Softmax forces probabilities to sum to 1.0, which would incorrectly assume only one sound
    /// can be present at a time. Sigmoid treats each class independently, allowing the model to
    /// output "speech: 95%, music: 60%, rain: 30%" for audio that contains all three sounds.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The model's raw output numbers (logits) can be any value from
    /// negative infinity to positive infinity. The sigmoid function squashes them into the
    /// range 0% to 100%, making them interpretable as probabilities:
    /// <list type="bullet">
    /// <item>A logit of 0 becomes 50% (uncertain)</item>
    /// <item>A logit of +5 becomes ~99% (very confident the sound is present)</item>
    /// <item>A logit of -5 becomes ~1% (very confident the sound is absent)</item>
    /// </list>
    /// </para>
    /// </remarks>
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
    /// Gets metadata about this BEATs model instance, including architecture details and configuration.
    /// </summary>
    /// <returns>A <see cref="ModelMetadata{T}"/> containing the model name, description, type,
    /// feature count, complexity level, and additional BEATs-specific information like embedding
    /// dimensions, number of Transformer layers, and audio processing parameters.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Model metadata is a description of the model's configuration.
    /// It's useful for logging, debugging, and understanding what model you're working with.
    /// For example, you can check whether you loaded the right model:
    /// <code>
    /// var meta = beats.GetModelMetadata();
    /// Console.WriteLine($"Model: {meta.Name}");
    /// Console.WriteLine($"Classes: {meta.FeatureCount}");
    /// Console.WriteLine($"Encoder layers: {meta.AdditionalInfo["NumEncoderLayers"]}");
    /// </code>
    /// </para>
    /// </remarks>
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
    /// Serializes all BEATs-specific model data into a binary stream for persistence.
    /// </summary>
    /// <param name="writer">Binary writer to serialize data into. The data is written in a
    /// fixed order that must match <see cref="DeserializeNetworkSpecificData"/>.</param>
    /// <remarks>
    /// <para>
    /// <b>What is serialized:</b> The mode (ONNX vs native), ONNX model path, all architecture
    /// hyperparameters (embedding dim, layer count, attention heads, FFN dim, patch size, etc.),
    /// detection settings (threshold, window size, overlap), training settings (dropout, mask
    /// probability, codebook size), and class labels. Native layer weights are serialized
    /// separately by the base class.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Serialization saves the model to a file so you can reload it later
    /// without retraining. Think of it like saving a game - all the model's learned knowledge
    /// and configuration is written to disk. You can then load it with
    /// <see cref="DeserializeNetworkSpecificData"/> to continue where you left off.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Write mode and model path
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);

        // Write architecture hyperparameters
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
        writer.Write(_options.FMin);
        writer.Write(_options.FMax);

        // Write class labels
        writer.Write(ClassLabels.Count);
        foreach (var label in ClassLabels)
        {
            writer.Write(label);
        }
    }

    /// <summary>
    /// Deserializes BEATs-specific model data from a binary stream.
    /// </summary>
    /// <param name="reader">Binary reader to deserialize data from. The read order must match
    /// the write order in <see cref="SerializeNetworkSpecificData"/>.</param>
    /// <remarks>
    /// <para>
    /// <b>What is restored:</b> All configuration written by <see cref="SerializeNetworkSpecificData"/>
    /// is read back and applied to the model options. The mel spectrogram extractor is re-created
    /// with the deserialized parameters, and if in ONNX mode, the ONNX model is reloaded from
    /// the saved path.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the counterpart to serialization - it loads a previously
    /// saved model from a file. After deserialization, the model is ready to use for inference
    /// just as it was before saving. Note that for ONNX models, the .onnx file must still exist
    /// at the saved path.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Restore mode and model path
        _useNativeMode = reader.ReadBoolean();
        string modelPath = reader.ReadString();
        if (!string.IsNullOrEmpty(modelPath))
        {
            _options.ModelPath = modelPath;
        }

        // Restore architecture hyperparameters
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
        _options.FMin = reader.ReadInt32();
        _options.FMax = reader.ReadInt32();

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
    /// Creates a new BEATs instance for the deserialization framework.
    /// </summary>
    /// <returns>A new BEATs instance configured with the same architecture and options.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is used internally by the serialization system. When loading
    /// a saved model, the framework first creates a blank instance using this method, then
    /// fills in the saved weights and configuration. You don't need to call this directly.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new BEATs<T>(Architecture, mp, _options);
        return new BEATs<T>(Architecture, _options);
    }

    #endregion

    #region Classification Helpers

    /// <summary>
    /// Classifies a single mel spectrogram window and returns per-class sigmoid probabilities.
    /// </summary>
    /// <param name="melSpec">Mel spectrogram features for one analysis window.
    /// Shape: [time_frames, num_mels] for a single window of audio.</param>
    /// <returns>Array of per-class probabilities after sigmoid activation.
    /// Length equals <see cref="AudioClassifierBase{T}.ClassLabels"/> count.</returns>
    /// <remarks>
    /// <para>
    /// <b>How classification works per window:</b>
    /// <list type="bullet">
    /// <item><b>ONNX mode</b>: The mel spectrogram is reshaped to [1, 1, T, F] (adding batch and
    /// channel dimensions as required by most ONNX audio models), then run through the ONNX
    /// encoder. The output logits are passed through sigmoid.</item>
    /// <item><b>Native mode</b>: The mel spectrogram is flattened and passed through the
    /// full BEATs layer stack (patch projection, Transformer encoder, classification head).
    /// Output logits are passed through sigmoid.</item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the core classification method that BEATs uses internally.
    /// For each window of audio, it converts the spectrogram into probabilities for every sound
    /// class. The sigmoid function ensures each probability is between 0 and 1, and multiple
    /// classes can have high probabilities simultaneously (multi-label classification).
    /// </para>
    /// </remarks>
    private T[] ClassifyWindow(Tensor<T> melSpec)
    {
        Tensor<T> output;

        if (IsOnnxMode && OnnxEncoder is not null)
        {
            // ONNX mode: reshape to [batch=1, channels=1, time, frequency] format
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
            // Native mode: flatten spectrogram and run through layer stack
            var input = new Tensor<T>([melSpec.Length]);
            int idx = 0;
            for (int t = 0; t < melSpec.Shape[0]; t++)
            {
                for (int f = 0; f < melSpec.Shape[1]; f++)
                {
                    input[idx++] = melSpec[t, f];
                }
            }
            output = Predict(input);
        }
        else
        {
            throw new InvalidOperationException(
                "No model available for classification. Provide an ONNX model path or use native training mode.");
        }

        // Apply sigmoid activation for multi-label classification.
        // Each class gets an independent probability in [0, 1].
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
    /// Splits raw audio into overlapping windows for frame-level BEATs analysis.
    /// </summary>
    /// <param name="audio">Raw audio waveform tensor [samples].</param>
    /// <returns>List of audio window tensors, each of length <c>WindowSize * SampleRate</c>.
    /// Windows overlap by the configured <see cref="BEATsOptions.WindowOverlap"/> ratio.
    /// Short audio clips are zero-padded to the full window size.</returns>
    /// <remarks>
    /// <para>
    /// <b>Why windowing is needed:</b> BEATs processes fixed-length audio segments (default 10s,
    /// matching AudioSet clip length). For longer audio, the waveform is split into overlapping
    /// windows. The overlap (default 50%) ensures that events near window boundaries are not
    /// missed - each region of audio is analyzed by at least two windows. Events detected in
    /// multiple overlapping windows are later merged by <see cref="MergeEvents"/>.
    /// </para>
    /// <para>
    /// <b>Edge cases handled:</b>
    /// <list type="bullet">
    /// <item>Audio shorter than one window: zero-padded to full window size</item>
    /// <item>Remaining audio after last full window: included as a partial window (zero-padded)
    /// if at least 10% of window size remains, ensuring no significant audio is lost</item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Imagine you have a 25-second recording and BEATs processes 10 seconds
    /// at a time with 50% overlap. The windows would be:
    /// <list type="bullet">
    /// <item>Window 1: 0s - 10s</item>
    /// <item>Window 2: 5s - 15s</item>
    /// <item>Window 3: 10s - 20s</item>
    /// <item>Window 4: 15s - 25s</item>
    /// </list>
    /// Notice that seconds 5-20 are analyzed by two windows each, which improves detection
    /// reliability, especially for events that span window boundaries.
    /// </para>
    /// </remarks>
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
            // Remaining values default to zero (zero-padded)
            windows.Add(window);
        }
        else if (windows.Count == 0 && audio.Length > 0)
        {
            // Very short audio: zero-pad to full window size so BEATs can still process it
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
    /// Merges overlapping or adjacent events of the same type into continuous event segments.
    /// </summary>
    /// <param name="events">Raw detected events, potentially with overlaps from adjacent windows.
    /// Due to the 50% window overlap, the same sound may be detected in two consecutive windows.</param>
    /// <returns>Merged events with no temporal overlaps within the same class. Events within
    /// 100ms of each other are merged, keeping the higher confidence score.</returns>
    /// <remarks>
    /// <para>
    /// <b>Why merging is needed:</b> Because BEATs processes overlapping windows (e.g., window 1
    /// covers 0-10s and window 2 covers 5-15s), a continuous dog bark from 4s to 8s would be
    /// detected in both windows. Without merging, you would get two separate "Dog barking" events.
    /// Merging combines them into a single event from 4s to 8s with the higher confidence score.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method cleans up the raw detection results by combining
    /// duplicate detections into single, clean events. Think of it like a spell-checker
    /// removing duplicate words - if "dog bark" was detected in window 1 (0-10s) and
    /// window 2 (5-15s), it combines them into one "dog bark" event spanning the full duration.
    /// </para>
    /// </remarks>
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
    /// Computes per-class aggregate statistics from a list of detected events.
    /// </summary>
    /// <param name="events">Detected (and merged) events to compute statistics for.</param>
    /// <returns>A dictionary mapping event type names to their statistics, including
    /// occurrence count, total duration, average confidence, and maximum confidence.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After detection, this method summarizes what was found. For example:
    /// <list type="bullet">
    /// <item>"Speech": detected 3 times, total 15.2s, avg confidence 87%, max confidence 95%</item>
    /// <item>"Music": detected 1 time, total 30.0s, avg confidence 62%, max confidence 62%</item>
    /// </list>
    /// This is useful for understanding the overall audio content without looking at individual events.
    /// </para>
    /// </remarks>
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
    /// Disposes managed and unmanaged resources held by this BEATs instance.
    /// </summary>
    /// <param name="disposing">True when called from <see cref="IDisposable.Dispose"/>,
    /// false when called from the finalizer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> BEATs may hold large resources (ONNX model in memory, GPU memory).
    /// Always use the <c>using</c> pattern or call <c>Dispose()</c> when done:
    /// <code>
    /// using var beats = new BEATs&lt;float&gt;(architecture, "model.onnx");
    /// var result = beats.Detect(audio);
    /// // Resources automatically freed when 'using' block ends
    /// </code>
    /// </para>
    /// </remarks>
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
    /// Streaming event detection session for real-time BEATs inference on continuous audio.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This nested class implements <see cref="IStreamingEventDetectionSession{T}"/> to enable
    /// real-time audio event detection. It maintains an internal buffer that accumulates audio
    /// samples as they arrive, processes complete windows through BEATs when enough samples
    /// are available, and emits detected events through the <see cref="IStreamingEventDetectionSession{T}.EventDetected"/>
    /// event handler.
    /// </para>
    /// <para>
    /// <b>Thread safety:</b> This class is thread-safe. All buffer operations are protected by
    /// a lock, and events are raised outside the lock to prevent deadlocks from event handlers
    /// that might call back into the session.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This allows you to process audio in real-time, like from a microphone.
    /// Instead of waiting for a complete audio file, you feed chunks of audio as they arrive:
    /// <code>
    /// using var session = beats.StartStreamingSession();
    ///
    /// // Option 1: Event-driven - get notified immediately when a sound is detected
    /// session.EventDetected += (sender, evt) =>
    /// {
    ///     Console.WriteLine($"[LIVE] Detected: {evt.EventType} ({evt.Confidence:P1})");
    /// };
    ///
    /// // Feed audio chunks from your microphone
    /// while (isRecording)
    /// {
    ///     var chunk = microphone.ReadSamples(bufferSize);
    ///     session.FeedAudio(chunk);
    /// }
    ///
    /// // Option 2: Polling - check for events periodically
    /// session.FeedAudio(audioChunk);
    /// var events = session.GetNewEvents(); // Returns events since last call
    /// var state = session.GetCurrentState(); // Returns current probabilities for all classes
    /// </code>
    ///
    /// The session automatically handles buffering, windowing, and overlap - you just feed
    /// raw audio samples and receive event notifications.
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

        /// <inheritdoc/>
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

            // Initialize state: all event probabilities start at zero
            foreach (var label in detector.ClassLabels)
            {
                _currentState[label] = detector.NumOps.Zero;
            }
        }

        /// <inheritdoc/>
        public void FeedAudio(Tensor<T> audioChunk)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(BEATsStreamingSession));
            }

            // Collect events to raise outside the lock to prevent deadlock.
            // Event handlers might acquire other locks or call back into this session.
            List<AudioEvent<T>>? eventsToRaise = null;

            lock (_lock)
            {
                if (_disposed)
                {
                    throw new ObjectDisposedException(nameof(BEATsStreamingSession));
                }

                // Accumulate incoming audio samples into the buffer
                for (int i = 0; i < audioChunk.Length; i++)
                {
                    _buffer.Add(audioChunk[i]);
                }

                // Process all complete windows available in the buffer
                while (_buffer.Count >= _windowSamples)
                {
                    // Extract one window from the front of the buffer
                    var window = new Tensor<T>([_windowSamples]);
                    for (int i = 0; i < _windowSamples; i++)
                    {
                        window[i] = _buffer[i];
                    }

                    // Run BEATs: mel spectrogram extraction + classification
                    var melSpec = _detector._melSpectrogram?.Forward(window) ??
                        throw new InvalidOperationException("MelSpectrogram not initialized.");
                    var scores = _detector.ClassifyWindow(melSpec);

                    // Check each class against the threshold
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

                    // Advance buffer by hop size (keeping overlap samples for next window)
                    int hopSamples = (int)(_windowSamples * (1 - _detector._options.WindowOverlap));
                    if (hopSamples <= 0) hopSamples = 1;
                    _buffer.RemoveRange(0, hopSamples);
                    _processedTime += hopSamples / (double)_sampleRate;
                }
            }

            // Raise events OUTSIDE the lock to prevent deadlock from event handlers
            if (eventsToRaise is not null)
            {
                foreach (var evt in eventsToRaise)
                {
                    EventDetected?.Invoke(this, evt);
                }
            }
        }

        /// <inheritdoc/>
        public IReadOnlyList<AudioEvent<T>> GetNewEvents()
        {
            lock (_lock)
            {
                var events = _newEvents.ToList();
                _newEvents.Clear();
                return events;
            }
        }

        /// <inheritdoc/>
        public IReadOnlyDictionary<string, T> GetCurrentState()
        {
            lock (_lock)
            {
                return new Dictionary<string, T>(_currentState);
            }
        }

        /// <inheritdoc/>
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
