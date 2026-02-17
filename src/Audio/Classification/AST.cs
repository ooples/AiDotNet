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
/// <item><b>Audio preprocessing</b>: Raw waveform is converted to a 128-bin log-mel spectrogram</item>
/// <item><b>Patch embedding</b>: The spectrogram is split into overlapping 16x16 patches (stride 10),
/// each linearly projected to a 768-dimensional embedding</item>
/// <item><b>[CLS] token</b>: A learnable classification token is prepended to the patch sequence</item>
/// <item><b>Positional encoding</b>: Learnable positional embeddings encode spatial position</item>
/// <item><b>Transformer encoder</b>: 12-layer encoder with 12-head self-attention processes patches</item>
/// <item><b>Classification</b>: The [CLS] token output is projected to class logits via a linear head</item>
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
/// <b>Step 3 - Add a special token:</b> A special [CLS] (classification) token is added.
/// This token will collect information from all patches and serve as the overall summary.
///
/// <b>Step 4 - Understand context:</b> A 12-layer Transformer encoder lets every patch attend
/// to every other patch. The [CLS] token gathers global information.
///
/// <b>Step 5 - Classify:</b> The [CLS] token output goes through a linear layer to produce
/// probabilities for each sound class. Sigmoid activation enables multi-label detection.
///
/// <b>Usage with a pre-trained ONNX model (recommended):</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 768, outputSize: 527);
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
/// var options = new ASTOptions { EmbeddingDim = 768, NumEncoderLayers = 12 };
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 768, outputSize: 527);
/// var ast = new AST&lt;float&gt;(arch, options);
/// ast.Train(spectrogramFeatures, labels);
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
public class AST<T> : AudioClassifierBase<T>, IAudioEventDetector<T>
{
    #region Fields

    private readonly ASTOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private MelSpectrogram<T>? _melSpectrogram;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

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
    /// <param name="modelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="options">AST configuration options. If null, uses paper defaults.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you have a pre-trained AST model file.
    /// The model will be ready for inference immediately without training.
    ///
    /// Example:
    /// <code>
    /// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 768, outputSize: 527);
    /// var ast = new AST&lt;float&gt;(arch, "ast_audioset.onnx");
    /// var result = ast.Detect(audioTensor);
    /// </code>
    /// </para>
    /// </remarks>
    public AST(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        ASTOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new ASTOptions();
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

        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        ClassLabels = _options.CustomLabels ?? AudioSetLabels;
        InitializeLayers();
    }

    /// <summary>
    /// Creates an AST model for native training mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="options">AST configuration options. If null, uses paper defaults.</param>
    /// <param name="optimizer">Optional custom optimizer. Defaults to AdamW.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor to train an AST model from scratch.
    ///
    /// Example:
    /// <code>
    /// var options = new ASTOptions { EmbeddingDim = 768, CustomLabels = new[] { "speech", "music", "noise" } };
    /// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 768, outputSize: 3);
    /// var ast = new AST&lt;float&gt;(arch, options);
    /// ast.Train(features, labels);
    /// </code>
    /// </para>
    /// </remarks>
    public AST(
        NeuralNetworkArchitecture<T> architecture,
        ASTOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new ASTOptions();
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
    /// Creates an AST model asynchronously with optional model download.
    /// </summary>
    /// <param name="options">AST configuration options.</param>
    /// <param name="progress">Optional progress reporter for download tracking.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A configured AST model ready for inference.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The easiest way to get started. Downloads and sets up everything:
    /// <code>
    /// var ast = await AST&lt;float&gt;.CreateAsync(
    ///     progress: new Progress&lt;double&gt;(p => Console.Write($"\rDownloading: {p:P0}")));
    /// var result = ast.Detect(audioTensor);
    /// </code>
    /// </para>
    /// </remarks>
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

    #region IAudioEventDetector Properties

    /// <summary>
    /// Gets the list of event types this model can detect.
    /// </summary>
    public IReadOnlyList<string> SupportedEvents => ClassLabels;

    /// <summary>
    /// Gets the event labels.
    /// </summary>
    public IReadOnlyList<string> EventLabels => ClassLabels;

    /// <summary>
    /// Gets the time resolution for event detection in seconds.
    /// </summary>
    public double TimeResolution => _options.WindowSize * (1 - _options.WindowOverlap);

    #endregion

    #region IAudioEventDetector Methods

    /// <summary>
    /// Detects all audio events using the default confidence threshold.
    /// </summary>
    /// <param name="audio">Raw audio waveform tensor [samples].</param>
    /// <returns>Detection result with events above the threshold.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pass in audio and get back detected sounds with timing and confidence.
    /// <code>
    /// var result = ast.Detect(audioTensor);
    /// foreach (var evt in result.Events)
    ///     Console.WriteLine($"{evt.EventType}: {evt.Confidence:P1}");
    /// </code>
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
    /// <param name="audio">Raw audio waveform tensor [samples].</param>
    /// <param name="threshold">Confidence threshold [0, 1].</param>
    /// <returns>Detection result with events above threshold.</returns>
    public AudioEventResult<T> Detect(Tensor<T> audio, T threshold)
    {
        ThrowIfDisposed();

        double thresholdValue = NumOps.ToDouble(threshold);
        double totalDuration = audio.Length / (double)_options.SampleRate;

        var windows = SplitIntoWindows(audio);
        var allEvents = new List<AudioEvent<T>>();

        for (int windowIdx = 0; windowIdx < windows.Count; windowIdx++)
        {
            var window = windows[windowIdx];
            double startTime = windowIdx * TimeResolution;

            var melSpec = _melSpectrogram?.Forward(window) ??
                throw new InvalidOperationException("MelSpectrogram not initialized.");

            var scores = ClassifyWindow(melSpec);

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
        var eventStats = ComputeEventStatistics(mergedEvents);

        return new AudioEventResult<T>
        {
            Events = mergedEvents,
            TotalDuration = totalDuration,
            DetectedEventTypes = mergedEvents.Select(e => e.EventType).Distinct().ToList(),
            EventStats = eventStats
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

    /// <inheritdoc/>
    public IStreamingEventDetectionSession<T> StartStreamingSession()
    {
        return StartStreamingSession(_options.SampleRate, NumOps.FromDouble(_options.Threshold));
    }

    /// <inheritdoc/>
    public IStreamingEventDetectionSession<T> StartStreamingSession(int sampleRate, T threshold)
    {
        return new ASTStreamingSession(this, sampleRate, threshold);
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultASTLayers(
                patchFeatureSize: _options.PatchSize * _options.PatchSize,
                embeddingDim: _options.EmbeddingDim,
                numEncoderLayers: _options.NumEncoderLayers,
                numAttentionHeads: _options.NumAttentionHeads,
                feedForwardDim: _options.FeedForwardDim,
                numClasses: ClassLabels.Count,
                maxSequenceLength: 1214,
                dropoutRate: _options.DropoutRate));
        }
    }

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();

        if (IsOnnxMode && OnnxEncoder is not null)
        {
            return OnnxEncoder.Run(input);
        }

        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }
        return current;
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
        {
            throw new NotSupportedException(
                "Training is not supported in ONNX mode. Create an AST model " +
                "without a modelPath parameter to train natively.");
        }

        SetTrainingMode(true);

        var output = Predict(input);
        var gradient = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());

        var gradientTensor = Tensor<T>.FromVector(gradient);
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradientTensor = Layers[i].Backward(gradientTensor);
        }

        _optimizer?.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    /// <inheritdoc/>
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

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (_melSpectrogram is null)
            throw new InvalidOperationException("MelSpectrogram not initialized.");

        return _melSpectrogram.Forward(rawAudio);
    }

    /// <inheritdoc/>
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

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "AST-Native" : "AST-ONNX",
            Description = "AST: Audio Spectrogram Transformer (Gong et al., Interspeech 2021)",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = ClassLabels.Count,
            Complexity = _options.NumEncoderLayers
        };
        metadata.AdditionalInfo["Architecture"] = "AST";
        metadata.AdditionalInfo["EmbeddingDim"] = _options.EmbeddingDim.ToString();
        metadata.AdditionalInfo["NumEncoderLayers"] = _options.NumEncoderLayers.ToString();
        metadata.AdditionalInfo["NumAttentionHeads"] = _options.NumAttentionHeads.ToString();
        metadata.AdditionalInfo["FeedForwardDim"] = _options.FeedForwardDim.ToString();
        metadata.AdditionalInfo["PatchSize"] = _options.PatchSize.ToString();
        metadata.AdditionalInfo["PatchStride"] = _options.PatchStride.ToString();
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

        writer.Write(ClassLabels.Count);
        foreach (var label in ClassLabels)
        {
            writer.Write(label);
        }
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string modelPath = reader.ReadString();
        if (!string.IsNullOrEmpty(modelPath))
        {
            _options.ModelPath = modelPath;
        }

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

        int numLabels = reader.ReadInt32();
        var labels = new string[numLabels];
        for (int i = 0; i < numLabels; i++)
        {
            labels[i] = reader.ReadString();
        }
        ClassLabels = labels;

        _melSpectrogram = new MelSpectrogram<T>(
            sampleRate: _options.SampleRate,
            nMels: _options.NumMels,
            nFft: _options.FftSize,
            hopLength: _options.HopLength,
            fMin: _options.FMin,
            fMax: _options.FMax,
            logMel: true);

        if (!_useNativeMode && _options.ModelPath is { } onnxModelPath && !string.IsNullOrEmpty(onnxModelPath))
        {
            OnnxEncoder = new OnnxModel<T>(onnxModelPath, _options.OnnxOptions);
        }
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new AST<T>(Architecture, mp, _options);
        return new AST<T>(Architecture, _options);
    }

    #endregion

    #region Classification Helpers

    private T[] ClassifyWindow(Tensor<T> melSpec)
    {
        Tensor<T> output;

        if (IsOnnxMode && OnnxEncoder is not null)
        {
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

        int remainingStart = windows.Count > 0 ? lastStart : 0;
        int remainingSamples = audio.Length - remainingStart;

        if (remainingSamples > windowSamples / 10)
        {
            var window = new Tensor<T>([windowSamples]);
            for (int i = 0; i < remainingSamples && i < windowSamples; i++)
            {
                window[i] = audio[remainingStart + i];
            }
            windows.Add(window);
        }
        else if (windows.Count == 0 && audio.Length > 0)
        {
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
                    var window = new Tensor<T>([_windowSamples]);
                    for (int i = 0; i < _windowSamples; i++)
                    {
                        window[i] = _buffer[i];
                    }

                    var melSpec = _detector._melSpectrogram?.Forward(window) ??
                        throw new InvalidOperationException("MelSpectrogram not initialized.");
                    var scores = _detector.ClassifyWindow(melSpec);

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
