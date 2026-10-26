using AiDotNet.Attributes;
using AiDotNet.Audio.Features;
using AiDotNet.LossFunctions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// AST (Audio Spectrogram Transformer) model for audio event detection and classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// AST (Gong et al., Interspeech 2021) is the first purely attention-based model for audio
/// classification. It directly applies a Vision Transformer (ViT) architecture to audio spectrograms,
/// achieving strong results on multiple benchmarks:
/// <list type="bullet">
/// <item><b>AudioSet</b>: 45.9% mAP - competitive with CNN-based models like PANNs</item>
/// <item><b>ESC-50</b>: 95.6% accuracy - environmental sound classification</item>
/// <item><b>Speech Commands V2</b>: 98.1% accuracy - keyword spotting</item>
/// </list>
/// </para>
/// <para>
/// <b>Architecture:</b> AST treats the audio spectrogram as an image and processes it with
/// a standard Vision Transformer:
/// <list type="number">
/// <item><b>Audio preprocessing</b>: Raw waveform is converted to a 128-bin Kaldi log-mel filterbank (25 ms
/// frames every 10 ms), padded or cut to 1024 frames and normalized to mean 0 and standard deviation 0.5</item>
/// <item><b>Patch embedding</b>: The spectrogram is split into overlapping 16x16 patches (stride 10),
/// each linearly projected to a 768-dimensional embedding</item>
/// <item><b>Tokens</b>: DeiT's class and distillation tokens are prepended to the patch sequence</item>
/// <item><b>Positional encoding</b>: Learnable positional embeddings encode spatial position</item>
/// <item><b>Transformer encoder</b>: 12 pre-norm blocks with 12-head self-attention</item>
/// <item><b>Classification</b>: The average of the two tokens' outputs goes through LayerNorm and a linear head,
/// and a sigmoid scores each label</item>
/// </list>
/// </para>
/// <para>
/// <b>Key Innovation:</b> AST demonstrates that ImageNet-pretrained ViT weights transfer effectively
/// to audio spectrograms. By initializing from DeiT (Data-efficient Image Transformer) and fine-tuning
/// on AudioSet, AST achieves competitive results without any audio-specific architectural modifications.
/// </para>
/// <para>
/// <b>For Beginners:</b> AST is one of the simplest yet effective audio classification models.
/// It works by treating sound spectrograms exactly like images and using a powerful image model
/// (Vision Transformer) to classify them.
///
/// Here is how AST processes audio, step by step:
///
/// <b>Step 1 - Sound to picture:</b> Audio is converted to a spectrogram (a 2D image showing
/// frequency vs time). This is the same mel spectrogram used by BEATs and other audio models.
///
/// <b>Step 2 - Cut into overlapping tiles:</b> The spectrogram is cut into small 16x16 patches
/// with overlap (stride 10), which means each patch shares some pixels with its neighbors.
/// This overlap improves accuracy at the cost of more patches to process.
///
/// <b>Step 3 - Add summary tokens:</b> Two special tokens (DeiT's class and distillation tokens) are added.
/// They collect information from all patches and serve as the overall summary.
///
/// <b>Step 4 - Understand context:</b> A 12-layer Transformer encoder lets every patch attend
/// to every other patch. The two summary tokens gather global information.
///
/// <b>Step 5 - Classify:</b> The average of the two tokens goes through a linear layer to produce
/// probabilities for each sound class. Sigmoid activation enables multi-label detection.
///
/// <b>Usage with a pre-trained ONNX model (recommended):</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 160000, outputSize: 527);
/// var ast = new AST&lt;float&gt;(arch, "ast_audioset.onnx");
/// var result = ast.Detect(audioTensor);
/// foreach (var evt in result.Events)
/// {
///     Console.WriteLine($"{evt.EventType}: {evt.Confidence:P1}");
/// }
/// </code>
///
/// <b>Usage with native training:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 160000, outputSize: 527);
/// var ast = new AST&lt;float&gt;(arch, new ASTOptions());
/// ast.Train(waveforms, multiHotLabels);   // [batch, samples] at 16 kHz, [batch, 527]
/// </code>
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "AST: Audio Spectrogram Transformer" (Gong et al., Interspeech 2021)</item>
/// <item>Repository: https://github.com/YuanGongND/ast</item>
/// </list>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("AST: Audio Spectrogram Transformer", "https://arxiv.org/abs/2104.01778", Year = 2021, Authors = "Yuan Gong, Yu-An Chung, James Glass")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-5, Beta1 = 0.95, Beta2 = 0.999, WeightDecay = 5e-7, ReferenceBatchSize = 12,
                WarmupSteps = 1000, Schedule = LearningRateSchedulerType.MultiStep, DecayRate = 0.5, MilestoneFractions = [0.4, 0.6, 0.8],
                Source = "Gong et al. 2021, Sec. 3.1.1 (full AudioSet): batch 12, Adam, binary cross-entropy, initial rate 1e-5 for 5 epochs, halved every epoch after the 2nd (milestones at 2/5, 3/5 and 4/5 of the run). The official recipe (YuanGongND/ast egs/audioset/run.sh, traintest.py) fills in betas (0.95, 0.999), weight decay 5e-7 and a linear warmup over the first 1000 steps.")]
public partial class AST<T> : AudioClassifierBase<T>, IAudioEventDetector<T>
{
    #region Fields

    private readonly ASTOptions _options;


    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly bool _useNativeMode;
    private bool _disposed;
    private AstPipeline<T>? _pipeline;
    private Random _trainingRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom(43);

    /// <summary>
    /// AudioSet-527 standard event labels used by AST pre-trained models.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These labels are the sounds that AST can recognize when using
    /// a pre-trained AudioSet model. You can provide your own labels via
    /// <see cref="ASTOptions.CustomLabels"/> for specialized tasks.
    /// </para>
    /// </remarks>
    public static readonly string[] AudioSetLabels = BEATs<T>.AudioSetLabels;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates an AST model for ONNX inference mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input/output dimensions.</param>
    /// <param name="modelPath">Path to a pre-trained ONNX graph that takes the official input
    /// <c>[batch, TargetLength, NumMels]</c> and returns label logits.</param>
    /// <param name="options">AST configuration options. If null, uses paper defaults.</param>
    public AST(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        ASTOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);

        _options = options ?? new ASTOptions(); Options = _options;
        _useNativeMode = false;
        _options.ModelPath = modelPath;
        base.SampleRate = _options.SampleRate;
        base.NumMels = _options.NumMels;
        ClassLabels = _options.CustomLabels ?? AudioSetLabels;
        _pipeline = new AstPipeline<T>(Engine, _options, ClassLabels.Count, layers: null);
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
    }

    /// <summary>
    /// Creates an AST model for native training and inference: the paper's network, initialized as the official code
    /// initializes it when training from scratch.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input/output dimensions.</param>
    /// <param name="options">AST configuration options. If null, uses paper defaults.</param>
    /// <param name="optimizer">Optimizer for training; the paper's recipe by default.</param>
    public AST(
        NeuralNetworkArchitecture<T> architecture,
        ASTOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new BinaryCrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new ASTOptions(); Options = _options;
        _useNativeMode = true;
        base.SampleRate = _options.SampleRate;
        base.NumMels = _options.NumMels;
        ClassLabels = _options.CustomLabels ?? AudioSetLabels;
        InitializeLayers();
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? throw new InvalidOperationException("AST declares its paper recipe.");
    }

    /// <summary>
    /// Creates an AST model asynchronously with optional model download.
    /// </summary>
    internal static async Task<AST<T>> CreateAsync(
        ASTOptions? options = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new ASTOptions();
        string modelPath = options.ModelPath ?? string.Empty;

        if (string.IsNullOrEmpty(modelPath))
        {
            var downloader = new OnnxModelDownloader();
            modelPath = await downloader.DownloadAsync(
                "ast",
                "ast_audioset.onnx",
                progress: progress,
                cancellationToken);
            options.ModelPath = modelPath;
        }

        var architecture = new NeuralNetworkArchitecture<T>(
            inputFeatures: options.EmbeddingDim,
            outputSize: (options.CustomLabels ?? AudioSetLabels).Length);

        return new AST<T>(architecture, modelPath, options);
    }

    #endregion

    /// <summary>The network, the same one <see cref="Fingerprinting.ASTModel{T}"/> runs.</summary>
    internal AstNetwork<T> Network => Pipeline.Network;

    private AstPipeline<T> Pipeline => _pipeline ?? throw new InvalidOperationException("The AST front end is not initialized.");

    /// <summary>Loads an official AST checkpoint's state dict.</summary>
    internal void LoadTorchWeights(Func<string, int[], double[]> read) => Network.LoadTorchWeights(read);

    /// <summary>Initializes from an ImageNet DeiT checkpoint with the paper's adaptation (§2.2).</summary>
    internal void LoadImageNetDeiT(Func<string, int[], double[]> read, int imageGrid)
        => Network.LoadImageNetDeiT(read, imageGrid, Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 2));

    #region IAudioEventDetector Properties

    /// <inheritdoc/>
    public IReadOnlyList<string> SupportedEvents => ClassLabels;

    /// <inheritdoc/>
    public IReadOnlyList<string> EventLabels => ClassLabels;

    /// <inheritdoc/>
    public double TimeResolution => _options.WindowSize * (1 - _options.WindowOverlap);

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
            var scores = ClassifyWindow(windows[windowIdx]);
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

        var mergedEvents = MergeEvents(allEvents);
        return new AudioEventResult<T>
        {
            Events = mergedEvents,
            TotalDuration = totalDuration,
            DetectedEventTypes = mergedEvents.Select(e => e.EventType).Distinct().ToList(),
            EventStats = ComputeEventStatistics(mergedEvents)
        };
    }

    /// <inheritdoc/>
    public Task<AudioEventResult<T>> DetectAsync(
        Tensor<T> audio,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Detect(audio), cancellationToken);
    }

    /// <inheritdoc/>
    public AudioEventResult<T> DetectSpecific(Tensor<T> audio, IReadOnlyList<string> eventTypes)
    {
        return DetectSpecific(audio, eventTypes, NumOps.FromDouble(_options.Threshold));
    }

    /// <inheritdoc/>
    public AudioEventResult<T> DetectSpecific(Tensor<T> audio, IReadOnlyList<string> eventTypes, T threshold)
    {
        var result = Detect(audio, threshold);
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

    /// <inheritdoc/>
    public Tensor<T> GetEventProbabilities(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var windows = SplitIntoWindows(audio);
        var rows = windows.Select(w => Engine.Reshape(Predict(w), new[] { 1, ClassLabels.Count })).ToArray();
        return rows.Length == 1 ? rows[0] : Engine.TensorConcatenate(rows, 0);
    }

    /// <inheritdoc/>
    public IStreamingEventDetectionSession<T> StartStreamingSession()
    {
        return StartStreamingSession(_options.SampleRate, NumOps.FromDouble(_options.Threshold));
    }

    /// <inheritdoc/>
    public IStreamingEventDetectionSession<T> StartStreamingSession(int sampleRate, T threshold)
    {
        if (sampleRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(sampleRate), "Sample rate must be positive.");
        return new ASTStreamingSession(this, sampleRate, threshold);
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode || _options is null || _pipeline is not null) return;
        if (Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException("AST builds the paper's network; custom layers are not supported.");
        var layers = new List<LayerBase<T>>();
        _pipeline = new AstPipeline<T>(Engine, _options, ClassLabels.Count, layers);
        int seed = _options.Seed ?? 42;
        _pipeline.Network.InitializeLikeReference(Tensors.Helpers.RandomHelper.CreateSeededRandom(seed));
        _trainingRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom(seed + 1);
        Layers.AddRange(layers);
    }

    /// <summary>
    /// The per-label probabilities <c>[batch, labels]</c> (sigmoid, multi-label) of waveforms <c>[samples]</c> or
    /// <c>[batch, samples]</c> at <see cref="ASTOptions.SampleRate"/>.
    /// </summary>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(input);
        if (IsOnnxMode && OnnxEncoder is not null)
            return PostprocessOutput(OnnxEncoder.Run(features));
        bool wasTraining = IsTrainingMode;
        if (wasTraining) SetTrainingMode(false);
        try
        {
            return PostprocessOutput(Network.Forward(features));
        }
        finally
        {
            if (wasTraining) SetTrainingMode(true);
        }
    }

    /// <inheritdoc />
    /// <remarks>The label logits of the waveforms.</remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => Network.Forward(PreprocessAudio(input));

    /// <summary>
    /// One training step on waveforms <c>[batch, samples]</c> and multi-hot labels <c>[batch, labels]</c>, as the
    /// paper trains on AudioSet: mixup, SpecAugment and binary cross-entropy (<see cref="AstPipeline{T}"/>).
    /// </summary>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
        {
            throw new NotSupportedException(
                "Training is not supported in ONNX mode. Create an AST model " +
                "without a modelPath parameter to train natively.");
        }
        var (features, targets) = Pipeline.TrainingBatch(input, expected, _trainingRandom);
        TrainWithCustomObjective(features, targets, (x, y) => Pipeline.Loss(Network.Forward(x), y), _optimizer);
        return true;
    }

    /// <inheritdoc />
    /// <remarks>Every layer is sized at construction and <see cref="NeuralNetworkBase{T}.Layers"/> is not a chain.</remarks>
    protected override void ResolveLazyLayerShapes()
    {
    }

    /// <inheritdoc />
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input) => ObservedLayerActivations(input);

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => Pipeline.Features(rawAudio);

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => Engine.Sigmoid(modelOutput);

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "AST-Native" : "AST-ONNX",
            Description = "AST: Audio Spectrogram Transformer (Gong et al., Interspeech 2021)",
            FeatureCount = ClassLabels.Count,
            Complexity = _options.NumEncoderLayers
        };
        metadata.AdditionalInfo["Architecture"] = "AST";
        metadata.AdditionalInfo["EmbeddingDim"] = _options.EmbeddingDim.ToString();
        metadata.AdditionalInfo["NumEncoderLayers"] = _options.NumEncoderLayers.ToString();
        metadata.AdditionalInfo["NumAttentionHeads"] = _options.NumAttentionHeads.ToString();
        metadata.AdditionalInfo["FeedForwardDim"] = _options.FeedForwardDim.ToString();
        metadata.AdditionalInfo["PatchSize"] = _options.PatchSize.ToString();
        metadata.AdditionalInfo["FrequencyStride"] = _options.FrequencyStride.ToString();
        metadata.AdditionalInfo["TimeStride"] = _options.TimeStride.ToString();
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        metadata.AdditionalInfo["NumMels"] = _options.NumMels.ToString();
        metadata.AdditionalInfo["TargetLength"] = _options.TargetLength.ToString();
        metadata.AdditionalInfo["NumClasses"] = ClassLabels.Count.ToString();
        return metadata;
    }

    #endregion

    #region Classification Helpers

    /// <summary>The label probabilities of one waveform window.</summary>
    private T[] ClassifyWindow(Tensor<T> window) => Predict(window).ToArray();

    #endregion

    #region Audio Windowing Helpers

    /// <summary>
    /// Splits audio into windows of <see cref="ASTOptions.WindowSize"/> seconds every
    /// <see cref="TimeResolution"/> seconds; a remainder longer than a tenth of a window becomes a last, zero-padded
    /// window, and audio shorter than one window is one zero-padded window.
    /// </summary>
    private List<Tensor<T>> SplitIntoWindows(Tensor<T> audio)
    {
        var flat = Engine.Reshape(audio, new[] { audio.Length });
        int length = flat.Length;
        var windows = new List<Tensor<T>>();
        int windowSamples = (int)(_options.WindowSize * _options.SampleRate);
        int hopSamples = Math.Max(1, (int)(windowSamples * (1 - _options.WindowOverlap)));
        Tensor<T> Window(int start)
        {
            int available = Math.Min(windowSamples, length - start);
            var part = Engine.TensorSlice(flat, new[] { start }, new[] { available });
            return available == windowSamples ? part : Engine.TensorConcatenate(new[] { part, new Tensor<T>(new[] { windowSamples - available }) }, 0);
        }

        int next = 0;
        for (; next + windowSamples <= length; next += hopSamples) windows.Add(Window(next));
        int remainder = length - next;
        if (windows.Count == 0 ? length > 0 : remainder > windowSamples / 10) windows.Add(Window(next));
        return windows;
    }

    #endregion

    #region Event Processing Helpers

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
            throw new ObjectDisposedException(GetType().FullName ?? nameof(AST<T>));
        }
    }

    /// <inheritdoc/>
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
    /// Streaming event detection session for real-time AST inference on continuous audio.
    /// </summary>
    private sealed class ASTStreamingSession : IStreamingEventDetectionSession<T>
    {
        private readonly AST<T> _detector;
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

        public ASTStreamingSession(AST<T> detector, int sampleRate, T threshold)
        {
            _detector = detector;
            _sampleRate = sampleRate;
            _threshold = threshold;
            _buffer = [];
            _newEvents = [];
            _currentState = new Dictionary<string, T>();
            _windowSamples = (int)(detector._options.WindowSize * sampleRate);
            _processedTime = 0;

            foreach (var label in detector.ClassLabels)
            {
                _currentState[label] = detector.NumOps.Zero;
            }
        }

        /// <inheritdoc/>
        public void FeedAudio(Tensor<T> audioChunk)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(ASTStreamingSession));

            List<AudioEvent<T>>? eventsToRaise = null;

            lock (_lock)
            {
                if (_disposed) throw new ObjectDisposedException(nameof(ASTStreamingSession));

                for (int i = 0; i < audioChunk.Length; i++)
                {
                    _buffer.Add(audioChunk[i]);
                }

                while (_buffer.Count >= _windowSamples)
                {
                    var window = new Tensor<T>(new[] { _windowSamples }, new Vector<T>(_buffer.GetRange(0, _windowSamples).ToArray()));
                    var scores = _detector.ClassifyWindow(window);

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

                    int hopSamples = (int)(_windowSamples * (1 - _detector._options.WindowOverlap));
                    if (hopSamples <= 0) hopSamples = 1;
                    _buffer.RemoveRange(0, hopSamples);
                    _processedTime += hopSamples / (double)_sampleRate;
                }
            }

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
