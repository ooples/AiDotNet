using AiDotNet.Attributes;
using AiDotNet.Audio.Features;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Speaker;

/// <summary>
/// Extracts speaker embeddings (d-vectors) from audio for speaker recognition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Speaker embeddings are compact vector representations that capture the
/// unique characteristics of a speaker's voice. These can be used for
/// speaker verification (is this the same person?) and speaker identification
/// (who is speaking?).
/// </para>
/// <para><b>For Beginners:</b> Each person's voice has unique characteristics
/// like pitch, rhythm, and timbre (tone color). This class converts audio into
/// a numerical "fingerprint" of the speaker's voice.
///
/// These embeddings are vectors (lists of numbers) that are:
/// - Close together for the same speaker
/// - Far apart for different speakers
///
/// Usage (ONNX Mode):
/// <code>
/// var extractor = new SpeakerEmbeddingExtractor&lt;float&gt;(
///     architecture,
///     modelPath: "speaker_model.onnx");
/// var embedding = extractor.ExtractEmbedding(audio);
/// </code>
///
/// Usage (Native Training Mode):
/// <code>
/// var extractor = new SpeakerEmbeddingExtractor&lt;float&gt;(architecture);
/// extractor.Train(audioInput, expectedEmbedding);
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Embedding)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("X-Vectors: Robust DNN Embeddings for Speaker Recognition", "https://doi.org/10.1109/ICASSP.2018.8461375")]
public partial class SpeakerEmbeddingExtractor<T> : SpeakerRecognitionBase<T>, ISpeakerEmbeddingExtractor<T>
{
    #region Execution Mode

    /// <summary>
    /// Whether the model is operating in native training mode.
    /// </summary>
    private readonly bool _useNativeMode;

    #endregion

    #region ONNX Mode Fields

    /// <summary>
    /// Path to the speaker model ONNX file.
    /// </summary>
    private readonly string? _modelPath;

    /// <summary>
    /// ONNX speaker embedding model.
    /// </summary>
    private readonly OnnxModel<T>? _onnxModel;

    #endregion

    #region Shared Fields

    /// <summary>
    /// Optimizer for training.
    /// </summary>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    /// <summary>
    /// Loss function for training.
    /// </summary>
    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// Speaker embedding options.
    /// </summary>
    private readonly SpeakerEmbeddingOptions _options;

    /// <summary>
    /// Whether the model has been disposed.
    /// </summary>
    private bool _disposed;

    #endregion

    #region Model Architecture Parameters

    /// <summary>
    /// Hidden dimension for encoder layers.
    /// </summary>
    private readonly int _hiddenDim;

    /// <summary>
    /// Number of encoder layers.
    /// </summary>
    private readonly int _numEncoderLayers;

    // The paper network on the native path; null in ONNX mode or with custom layers.
    private XVectorNetwork<T>? _network;

    internal XVectorNetwork<T>? Network => _network;


    #endregion

    #region Public Properties

    /// <summary>
    /// Gets the minimum audio duration required for reliable embedding extraction.
    /// </summary>
    public double MinimumDurationSeconds { get; }

    /// <summary>
    /// Gets whether the model is in ONNX inference mode.
    /// </summary>
    public new bool IsOnnxMode => _onnxModel is not null;

    /// <summary>
    /// Gets whether a neural model is loaded.
    /// </summary>
    public bool HasNeuralModel => _onnxModel?.IsLoaded == true || (!_useNativeMode && Layers.Count > 0);

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a SpeakerEmbeddingExtractor with default settings for native training mode.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the simplest way to create a speaker embedding extractor.
    /// It uses default settings suitable for most use cases.
    /// </para>
    /// </remarks>
    public SpeakerEmbeddingExtractor()
        : this(new SpeakerEmbeddingOptions())
    {
    }

    /// <summary>
    /// Creates a SpeakerEmbeddingExtractor with custom options.
    /// </summary>
    /// <param name="options">Configuration options for the extractor.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to customize sample rate, embedding dimension, etc.
    /// </para>
    /// </remarks>
    public SpeakerEmbeddingExtractor(SpeakerEmbeddingOptions options)
        : this(
            new NeuralNetworkArchitecture<T>(
                inputFeatures: (options ?? throw new ArgumentNullException(nameof(options))).EmbeddingDimension,
                outputSize: options.EmbeddingDimension),
            options: options)
    {
    }

    /// <summary>
    /// Creates a SpeakerEmbeddingExtractor for ONNX inference with a pretrained model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="modelPath">Required path to speaker embedding ONNX model.</param>
    /// <param name="options">Configuration options; null uses the published defaults.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor when you have a pretrained speaker embedding model.
    ///
    /// Example:
    /// <code>
    /// var extractor = new SpeakerEmbeddingExtractor&lt;float&gt;(
    ///     architecture,
    ///     modelPath: "ecapa_tdnn.onnx");
    /// </code>
    /// </para>
    /// </remarks>
    public SpeakerEmbeddingExtractor(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        SpeakerEmbeddingOptions? options = null)
        : base(architecture)
    {
        if (modelPath is null)
            throw new ArgumentNullException(nameof(modelPath));

        // Copied rather than held by reference so nothing here can mutate an options object the
        // caller still holds.
        _options = options is null ? new SpeakerEmbeddingOptions() : new SpeakerEmbeddingOptions(options);
        Options = _options;
        _options.Validate();

        _useNativeMode = false;
        _modelPath = modelPath;

        // Store parameters
        SampleRate = _options.SampleRate;
        EmbeddingDimension = _options.EmbeddingDimension;
        MinimumDurationSeconds = _options.MinimumDurationSeconds;
        _hiddenDim = _options.HiddenDim;
        _numEncoderLayers = _options.NumEncoderLayers;

        // Create MFCC extractor
        MfccExtractor = CreateMfccExtractor(_options);

        // Load ONNX model
        _onnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        OnnxModel = _onnxModel;

        // Default loss function (MSE is standard for embedding extraction)
        _lossFunction = new MeanSquaredErrorLoss<T>();

        // Initialize layers (empty for ONNX mode)
        InitializeLayers();
    }

    /// <summary>
    /// Creates a SpeakerEmbeddingExtractor for native training mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="optimizer">Optimizer for training. If null, AdamW is used.</param>
    /// <param name="lossFunction">Loss function for training. If null, MSE loss is used.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to train your own speaker embedding model.
    ///
    /// Example:
    /// <code>
    /// var extractor = new SpeakerEmbeddingExtractor&lt;float&gt;(architecture);
    /// extractor.Train(audioInput, expectedEmbedding);
    /// </code>
    /// </para>
    /// </remarks>
    public SpeakerEmbeddingExtractor(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        SpeakerEmbeddingOptions? options = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new SpeakerEmbeddingOptions();
        Options = _options;
        _options.Validate();

        _useNativeMode = true;
        _modelPath = null;

        // Store parameters
        SampleRate = _options.SampleRate;
        EmbeddingDimension = _options.EmbeddingDimension;
        MinimumDurationSeconds = _options.MinimumDurationSeconds;
        _hiddenDim = _options.HiddenDim;
        _numEncoderLayers = _options.NumEncoderLayers;

        // Create MFCC extractor
        MfccExtractor = CreateMfccExtractor(_options);

        // Initialize optimizer and loss function
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        SetBaseTrainOptimizer(_optimizer);

        // Initialize layers
        InitializeLayers();
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the layers for the speaker embedding model.
    /// </summary>
    /// <remarks>
    /// Follows the golden standard pattern:
    /// 1. Check if in native mode (ONNX mode returns early)
    /// 2. Use Architecture.Layers if provided by user
    /// 3. Fall back to LayerHelper.CreateDefaultSpeakerEmbeddingLayers() otherwise
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        _network = null;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            return;
        }
        // The x-vector network (Snyder et al. 2018, Table 1). It used to be a generic dense stack over MFCCs.
        var layers = new List<LayerBase<T>>();
        _network = new XVectorNetwork<T>(Engine, layers, _options.SampleRate, _options.NumMfcc, _options.FftSize, _options.HopLength,
            _hiddenDim, _options.PoolingDim, EmbeddingDimension, _options.NumTrainingSpeakers, _options.NormalizationWindowFrames);
        Layers.AddRange(layers);
    }

    // Features [frames, bins] (or [B, frames, bins]) of a waveform [S] (or of features passed through).
    private Tensor<T> NetworkFeatures(Tensor<T> input)
    {
        var network = _network ?? throw new InvalidOperationException("No native x-vector network.");
        if (input.Rank == 1)
        {
            var features = network.Features(input);
            return Engine.Reshape(features, new[] { 1, features.Shape[0], features.Shape[1] });
        }
        return input.Rank == 2 ? Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1] }) : input;
    }

    private const int SegmentLevelTrailingLayers = 3;

    /// <summary>
    /// X-Vector forward pass: run the frame-level TDNN/attention stack, pool
    /// across the time dimension, then run the segment-level FC layers and
    /// embedding projection. The default
    /// <see cref="LayerHelper{T}.CreateDefaultSpeakerEmbeddingLayers"/> chain
    /// labelled its trailing dense layers as "attentive statistics pooling
    /// (simplified)" but those layers operate on the last (feature) dim only —
    /// they do NOT reduce the time axis, so the model returned
    /// <c>[batch, frames, embed_dim]</c> instead of the
    /// <c>[batch, embed_dim]</c> the Predict / Train contract expects.
    /// The MSE loss in TrainWithTape then threw "Tensor shapes must match
    /// [1, 64, 256] and [256]". Inserting <c>ReduceMean</c> across the time
    /// axis between the frame-level and segment-level halves restores the
    /// paper's pooling contract; mean-pool is the simplified form of the
    /// paper's mean+std pool, sufficient for the segment-level FC chain
    /// already present in <c>CreateDefaultSpeakerEmbeddingLayers</c>.
    /// </summary>
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        return ForwardWithTemporalPool(input);
    }

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        return ForwardWithTemporalPool(input);
    }

    private Tensor<T> ForwardWithTemporalPool(Tensor<T> input)
    {
        if (_network is not null)
        {
            var xvectors = _network.Embed(NetworkFeatures(input), IsTrainingMode);
            return input.Rank == 3 ? xvectors : Engine.Reshape(xvectors, new[] { EmbeddingDimension });
        }
        Tensor<T> x = input;
        int segCount = System.Math.Min(SegmentLevelTrailingLayers, Layers.Count);
        int frameCount = Layers.Count - segCount;

        // Frame-level TDNN / attention stack
        for (int i = 0; i < frameCount; i++)
        {
            x = Layers[i].Forward(x);
        }

        // Temporal pooling: reduce the time dim (rank-3 [B, T, D] → [B, D];
        // rank-2 [T, D] → [D] for unbatched inputs). No-op for already-pooled
        // representations (rank <= 1).
        if (x.Rank >= 2)
        {
            int timeAxis = x.Rank - 2;
            x = Engine.ReduceMean(x, new[] { timeAxis }, keepDims: false);
        }

        // Segment-level FC + embedding projection
        for (int i = frameCount; i < Layers.Count; i++)
        {
            x = Layers[i].Forward(x);
        }

        return x;
    }

    #endregion

    #region ISpeakerEmbeddingExtractor Implementation

    /// <summary>
    /// Extracts a speaker embedding from audio.
    /// </summary>
    /// <param name="audio">Audio waveform tensor [samples] or [batch, samples].</param>
    /// <returns>Speaker embedding tensor [embedding_dim] or [batch, embedding_dim].</returns>
    public Tensor<T> ExtractEmbedding(Tensor<T> audio)
    {
        ThrowIfDisposed();
        return Predict(audio);
    }

    /// <summary>
    /// Extracts a speaker embedding from audio asynchronously.
    /// </summary>
    public Task<Tensor<T>> ExtractEmbeddingAsync(Tensor<T> audio, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => ExtractEmbedding(audio), cancellationToken);
    }

    /// <summary>
    /// Extracts embeddings from multiple audio segments.
    /// </summary>
    public IReadOnlyList<Tensor<T>> ExtractEmbeddings(IReadOnlyList<Tensor<T>> audioSegments)
    {
        return audioSegments.Select(ExtractEmbedding).ToList();
    }

    /// <summary>
    /// Computes similarity between two speaker embeddings.
    /// </summary>
    public T ComputeSimilarity(Tensor<T> embedding1, Tensor<T> embedding2)
    {
        return ComputeCosineSimilarity(embedding1, embedding2);
    }

    /// <summary>
    /// Aggregates multiple embeddings into a single representative embedding.
    /// </summary>
    Tensor<T> ISpeakerEmbeddingExtractor<T>.AggregateEmbeddings(IReadOnlyList<Tensor<T>> embeddings)
    {
        return AggregateEmbeddings(embeddings);
    }

    /// <summary>
    /// Normalizes an embedding to unit length.
    /// </summary>
    Tensor<T> ISpeakerEmbeddingExtractor<T>.NormalizeEmbedding(Tensor<T> embedding)
    {
        return NormalizeEmbedding(embedding);
    }

    #endregion

    #region AudioNeuralNetworkBase Implementation

    /// <summary>
    /// Preprocesses raw audio for model input.
    /// </summary>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        // Extract MFCCs from raw audio
        if (MfccExtractor is not null)
        {
            return MfccExtractor.Extract(rawAudio);
        }
        return rawAudio;
    }

    /// <summary>
    /// Postprocesses model output into the final result format.
    /// </summary>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        // Validate and handle output dimension mismatch
        if (modelOutput.Length != EmbeddingDimension)
        {
            // If output is shorter, pad with zeros; if longer, truncate
            // Log warning for diagnostic purposes via debug output
            System.Diagnostics.Debug.WriteLine(
                $"SpeakerEmbeddingExtractor: Model output size ({modelOutput.Length}) does not match " +
                $"expected EmbeddingDimension ({EmbeddingDimension}). Adjusting embedding.");

            var adjusted = new Tensor<T>([EmbeddingDimension]);
            int copyLen = Math.Min(modelOutput.Length, EmbeddingDimension);
            for (int i = 0; i < copyLen; i++)
            {
                adjusted[i] = modelOutput[i];
            }
            // Remaining elements are default (zero) if output was shorter
            modelOutput = adjusted;
        }

        // L2 normalize the embedding
        return NormalizeEmbedding(modelOutput);
    }

    /// <summary>
    /// Makes a prediction using the model.
    /// </summary>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        if (!_useNativeMode)
        {
            // ONNX inference
            if (_onnxModel is null)
                throw new InvalidOperationException("ONNX model not loaded.");

            // Preprocess audio to MFCCs
            var mfccs = PreprocessAudio(input);

            // Run through ONNX model
            var output = _onnxModel.Run(mfccs);

            // Postprocess (normalize)
            return PostprocessOutput(output);
        }
        else
        {
            if (_network is not null)
            {
                bool wasTraining = IsTrainingMode;
                if (wasTraining) SetTrainingMode(false);
                try { return PostprocessOutput(ForwardWithTemporalPool(input)); }
                finally { if (wasTraining) SetTrainingMode(true); }
            }
            // Native forward pass
            var preprocessed = PreprocessAudio(input);
            Tensor<T> output = preprocessed;
            foreach (var layer in Layers)
            {
                output = layer.Forward(output);
            }
            return PostprocessOutput(output);
        }
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    /// <remarks>
    /// With a speaker classifier (<see cref="SpeakerEmbeddingOptions.NumTrainingSpeakers"/>), a target that is one
    /// distribution over the training speakers per input trains the x-vector network as the paper does (section 2.2):
    /// the cross-entropy of the softmax over speakers. Any other target falls back to the standard regression step.
    /// </remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> target)
    {
        var network = _network;
        if (network?.Classifier is null || target.Shape[^1] != _options.NumTrainingSpeakers) return false;
        int speakers = _options.NumTrainingSpeakers;
        var weights = Engine.TensorMultiplyScalar(Engine.Reshape(target, new[] { target.Length / speakers, speakers }), NumOps.FromDouble(-1.0));
        TrainWithCustomLoss(input, _ =>
        {
            var logits = network.SpeakerLogits(network.Embed(NetworkFeatures(input), training: true));
            var loss = Engine.ReduceSum(Engine.TensorMultiply(Engine.TensorLogSoftmax(logits, axis: -1), weights), new[] { 0, 1 }, keepDims: false);
            return Engine.TensorMultiplyScalar(loss, NumOps.FromDouble(1.0 / logits.Shape[0]));
        }, _optimizer);
        return true;
    }
    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "SpeakerEmbeddingExtractor-Native" : "SpeakerEmbeddingExtractor-ONNX",
            Description = "Speaker embedding extraction model for voice recognition",
            FeatureCount = _options.NumMfcc * 3, // MFCC + delta + delta-delta
            Complexity = 1
        };
        metadata.AdditionalInfo["InputFormat"] = $"Audio ({SampleRate}Hz)";
        metadata.AdditionalInfo["OutputFormat"] = $"Embedding ({EmbeddingDimension}D)";
        metadata.AdditionalInfo["Mode"] = _useNativeMode ? "Native Training" : "ONNX Inference";
        return metadata;
    }

    /// <summary>
    /// Serializes network-specific data.
    /// </summary>


    /// <summary>
    /// Deserializes network-specific data.
    /// </summary>


    #endregion

    #region Legacy API Support

    /// <summary>
    /// Extracts a speaker embedding from audio (legacy API).
    /// </summary>
    /// <param name="audio">Audio samples as a tensor.</param>
    /// <returns>Speaker embedding result.</returns>
    public SpeakerEmbedding<T> Extract(Tensor<T> audio)
    {
        var embedding = ExtractEmbedding(audio);
        return new SpeakerEmbedding<T>
        {
            Vector = embedding.ToArray(),
            Duration = (double)audio.Length / SampleRate,
            NumFrames = MfccExtractor is not null ? MfccExtractor.Extract(audio).Shape[0] : audio.Length
        };
    }

    /// <summary>
    /// Extracts a speaker embedding from audio (legacy API).
    /// </summary>
    /// <param name="audio">Audio samples as a vector.</param>
    /// <returns>Speaker embedding result.</returns>
    public SpeakerEmbedding<T> Extract(Vector<T> audio)
    {
        var tensor = new Tensor<T>([audio.Length]);
        for (int i = 0; i < audio.Length; i++)
        {
            tensor[i] = audio[i];
        }
        return Extract(tensor);
    }

    /// <summary>
    /// Extracts speaker embedding from audio as a Tensor.
    /// </summary>
    public Tensor<T> ExtractTensor(Tensor<T> audio)
    {
        return ExtractEmbedding(audio);
    }

    /// <summary>
    /// Extracts embeddings from multiple audio segments (legacy API).
    /// </summary>
    public List<SpeakerEmbedding<T>> ExtractBatch(IEnumerable<Tensor<T>> segments)
    {
        return segments.Select(Extract).ToList();
    }

    /// <summary>
    /// Computes cosine similarity between two speaker embeddings (legacy API).
    /// </summary>
    public T ComputeSimilarity(SpeakerEmbedding<T> embedding1, SpeakerEmbedding<T> embedding2)
    {
        return NumOps.FromDouble(embedding1.CosineSimilarity(embedding2));
    }

    #endregion

    #region Private Methods

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName);
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Disposes the model and releases resources.
    /// </summary>
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;

        if (disposing)
        {
            _onnxModel?.Dispose();
        }

        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
