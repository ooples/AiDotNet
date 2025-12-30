using AiDotNet.Audio.Features;
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
public class SpeakerEmbeddingExtractor<T> : SpeakerRecognitionBase<T>, ISpeakerEmbeddingExtractor<T>
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
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// Loss function for training.
    /// </summary>
    private readonly ILossFunction<T> _lossFunction;

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

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    private readonly int _numHeads;

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
    /// Creates a SpeakerEmbeddingExtractor for ONNX inference with a pretrained model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="modelPath">Required path to speaker embedding ONNX model.</param>
    /// <param name="sampleRate">Expected sample rate for input audio. Default is 16000.</param>
    /// <param name="embeddingDimension">Dimension of output embeddings. Default is 256.</param>
    /// <param name="minimumDurationSeconds">Minimum audio duration for reliable embedding. Default is 0.5.</param>
    /// <param name="onnxOptions">ONNX runtime options.</param>
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
        int sampleRate = 16000,
        int embeddingDimension = 256,
        double minimumDurationSeconds = 0.5,
        OnnxModelOptions? onnxOptions = null)
        : base(architecture)
    {
        if (modelPath is null)
            throw new ArgumentNullException(nameof(modelPath));

        _useNativeMode = false;
        _modelPath = modelPath;

        // Store parameters
        SampleRate = sampleRate;
        EmbeddingDimension = embeddingDimension;
        MinimumDurationSeconds = minimumDurationSeconds;
        _hiddenDim = 256;
        _numEncoderLayers = 3;
        _numHeads = 4;

        // Initialize options
        _options = new SpeakerEmbeddingOptions
        {
            SampleRate = sampleRate,
            EmbeddingDimension = embeddingDimension,
            ModelPath = modelPath,
            OnnxOptions = onnxOptions ?? new OnnxModelOptions()
        };

        // Create MFCC extractor
        MfccExtractor = CreateMfccExtractor(sampleRate);

        // Load ONNX model
        _onnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        OnnxModel = _onnxModel;

        // Initialize optimizer and loss function (not used in ONNX mode, but required for readonly fields)
        _lossFunction = new MeanSquaredErrorLoss<T>();
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);

        // Initialize layers (empty for ONNX mode)
        InitializeLayers();
    }

    /// <summary>
    /// Creates a SpeakerEmbeddingExtractor for native training mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sampleRate">Expected sample rate for input audio. Default is 16000.</param>
    /// <param name="embeddingDimension">Dimension of output embeddings. Default is 256.</param>
    /// <param name="minimumDurationSeconds">Minimum audio duration for reliable embedding. Default is 0.5.</param>
    /// <param name="hiddenDim">Hidden dimension for encoder layers. Default is 256.</param>
    /// <param name="numEncoderLayers">Number of encoder layers. Default is 3.</param>
    /// <param name="numHeads">Number of attention heads. Default is 4.</param>
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
        int sampleRate = 16000,
        int embeddingDimension = 256,
        double minimumDurationSeconds = 0.5,
        int hiddenDim = 256,
        int numEncoderLayers = 3,
        int numHeads = 4,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        _useNativeMode = true;
        _modelPath = null;

        // Store parameters
        SampleRate = sampleRate;
        EmbeddingDimension = embeddingDimension;
        MinimumDurationSeconds = minimumDurationSeconds;
        _hiddenDim = hiddenDim;
        _numEncoderLayers = numEncoderLayers;
        _numHeads = numHeads;

        // Initialize options
        _options = new SpeakerEmbeddingOptions
        {
            SampleRate = sampleRate,
            EmbeddingDimension = embeddingDimension
        };

        // Create MFCC extractor
        MfccExtractor = CreateMfccExtractor(sampleRate);

        // Initialize optimizer and loss function
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);

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

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            int numMels = _options.NumMfcc > 0 ? _options.NumMfcc : 80;
            Layers.AddRange(LayerHelper<T>.CreateDefaultSpeakerEmbeddingLayers(
                numMels: numMels,
                hiddenDim: _hiddenDim,
                embeddingDim: EmbeddingDimension,
                numLayers: _numEncoderLayers,
                maxFrames: 1000,
                dropoutRate: 0.0));
        }
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
        // L2 normalize the embedding
        return NormalizeEmbedding(modelOutput);
    }

    /// <summary>
    /// Makes a prediction using the model.
    /// </summary>
    public override Tensor<T> Predict(Tensor<T> input)
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

    /// <summary>
    /// Updates model parameters.
    /// </summary>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException("Cannot update parameters in ONNX inference mode.");
        }

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
    /// Trains the model on input data.
    /// </summary>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException("Cannot train in ONNX inference mode. Use the native training constructor.");
        }

        // 1. Set training mode
        SetTrainingMode(true);

        // 2. Forward pass
        var prediction = Predict(input);

        // 3. Convert tensors to vectors for loss calculation
        var flatPrediction = prediction.ToVector();
        var flatExpected = expectedOutput.ToVector();

        // 4. Compute loss
        LastLoss = _lossFunction.CalculateLoss(flatPrediction, flatExpected);

        // 5. Compute gradients via backpropagation
        var lossGradient = _lossFunction.CalculateDerivative(flatPrediction, flatExpected);
        Backpropagate(Tensor<T>.FromVector(lossGradient));

        // 6. Update parameters using optimizer
        _optimizer.UpdateParameters(Layers);

        // 7. Exit training mode
        SetTrainingMode(false);
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
            ModelType = ModelType.NeuralNetwork,
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
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(SampleRate);
        writer.Write(EmbeddingDimension);
        writer.Write(MinimumDurationSeconds);
        writer.Write(_useNativeMode);
        writer.Write(_hiddenDim);
        writer.Write(_numEncoderLayers);
        writer.Write(_numHeads);
    }

    /// <summary>
    /// Deserializes network-specific data.
    /// </summary>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        SampleRate = reader.ReadInt32();
        EmbeddingDimension = reader.ReadInt32();
        _ = reader.ReadDouble(); // MinimumDurationSeconds
        _ = reader.ReadBoolean(); // useNativeMode
        _ = reader.ReadInt32(); // hiddenDim
        _ = reader.ReadInt32(); // numEncoderLayers
        _ = reader.ReadInt32(); // numHeads
    }

    /// <summary>
    /// Creates a new instance of this model for cloning.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _modelPath is not null)
        {
            return new SpeakerEmbeddingExtractor<T>(
                Architecture,
                _modelPath,
                SampleRate,
                EmbeddingDimension,
                MinimumDurationSeconds,
                _options.OnnxOptions);
        }
        else
        {
            return new SpeakerEmbeddingExtractor<T>(
                Architecture,
                SampleRate,
                EmbeddingDimension,
                MinimumDurationSeconds,
                _hiddenDim,
                _numEncoderLayers,
                _numHeads,
                lossFunction: _lossFunction);
        }
    }

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
