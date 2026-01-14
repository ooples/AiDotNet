using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Video.Understanding;

/// <summary>
/// InternVideo2: Scaling Video Foundation Models for Multimodal Video Understanding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// InternVideo2 is a state-of-the-art video understanding model that combines:
/// - Video-text contrastive learning
/// - Masked video modeling
/// - Video-text generative learning
/// </para>
/// <para>
/// <b>For Beginners:</b> InternVideo2 understands video content by analyzing frames
/// and learning relationships between visual content and language. It can:
/// - Classify videos (what's happening?)
/// - Find videos matching text descriptions
/// - Answer questions about video content
/// - Generate video captions
///
/// Example usage (native mode for training):
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3);
/// var model = new InternVideo2&lt;double&gt;(arch);
/// model.Train(videoFrames, expectedEmbedding);
/// var embedding = model.EncodeVideo(videoFrames);
/// </code>
///
/// Example usage (ONNX mode for inference only):
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3);
/// var model = new InternVideo2&lt;double&gt;(arch, "internvideo2.onnx");
/// var embedding = model.EncodeVideo(videoFrames);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "InternVideo2: Scaling Video Foundation Models for Multimodal Video Understanding"
/// https://arxiv.org/abs/2403.15377
/// </para>
/// </remarks>
public class InternVideo2<T> : NeuralNetworkBase<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether this model uses native layers (true) or ONNX model (false).
    /// </summary>
    private readonly bool _useNativeMode;

    #endregion

    #region ONNX Mode Fields

    /// <summary>
    /// The ONNX inference session for the model.
    /// </summary>
    private readonly InferenceSession? _onnxSession;

    /// <summary>
    /// Path to the ONNX model file.
    /// </summary>
    private readonly string? _onnxModelPath;

    #endregion

    #region Native Mode Fields

    /// <summary>
    /// The optimizer used for training.
    /// </summary>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    /// <summary>
    /// The loss function for training.
    /// </summary>
    private readonly ILossFunction<T> _lossFunction;

    /// <summary>
    /// Embedding dimension for the model.
    /// </summary>
    private readonly int _embedDim;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    private readonly int _numHeads;

    /// <summary>
    /// Number of encoder layers.
    /// </summary>
    private readonly int _numEncoderLayers;

    /// <summary>
    /// Number of video frames to process.
    /// </summary>
    private readonly int _numFrames;

    /// <summary>
    /// Patch size for tokenization.
    /// </summary>
    private readonly int _patchSize;

    /// <summary>
    /// Input image size.
    /// </summary>
    private readonly int _imageSize;

    #endregion

    #region Properties

    /// <summary>
    /// Gets whether this model uses native mode (true) or ONNX mode (false).
    /// </summary>
    internal bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets whether training is supported (only in native mode).
    /// </summary>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// Gets the embedding dimension.
    /// </summary>
    internal int EmbedDim => _embedDim;

    /// <summary>
    /// Gets the number of frames processed.
    /// </summary>
    internal int NumFrames => _numFrames;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates an InternVideo2 model using native layers for training and inference.
    /// </summary>
    /// <param name="architecture">Architecture for the video encoder.</param>
    /// <param name="optimizer">Optional optimizer for training. Default: Adam.</param>
    /// <param name="lossFunction">Optional loss function. Default: MSE.</param>
    /// <param name="embedDim">Embedding dimension (default: 768).</param>
    /// <param name="numHeads">Number of attention heads (default: 12).</param>
    /// <param name="numEncoderLayers">Number of encoder layers (default: 12).</param>
    /// <param name="numFrames">Number of frames to process (default: 8).</param>
    /// <param name="patchSize">Patch size for tokenization (default: 14).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create a trainable InternVideo2 model:
    /// <code>
    /// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     inputType: InputType.ThreeDimensional,
    ///     inputHeight: 224, inputWidth: 224, inputDepth: 3);
    /// var model = new InternVideo2&lt;double&gt;(arch);
    /// </code>
    /// </para>
    /// </remarks>
    public InternVideo2(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int embedDim = 768,
        int numHeads = 12,
        int numEncoderLayers = 12,
        int numFrames = 8,
        int patchSize = 14)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        if (embedDim < 1)
            throw new ArgumentOutOfRangeException(nameof(embedDim), embedDim, "Embedding dimension must be at least 1.");
        if (numEncoderLayers < 1)
            throw new ArgumentOutOfRangeException(nameof(numEncoderLayers), numEncoderLayers, "Number of encoder layers must be at least 1.");
        if (patchSize < 1)
            throw new ArgumentOutOfRangeException(nameof(patchSize), patchSize, "Patch size must be at least 1.");

        _useNativeMode = true;
        _embedDim = embedDim;
        _numHeads = numHeads;
        _numEncoderLayers = numEncoderLayers;
        _numFrames = numFrames;
        _patchSize = patchSize;
        _imageSize = architecture.InputHeight > 0 ? architecture.InputHeight : 224;

        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Creates an InternVideo2 model using a pretrained ONNX model for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the pretrained ONNX model.</param>
    /// <param name="embedDim">Embedding dimension of the model (default: 768).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you have a pretrained model
    /// in ONNX format. Training is not supported in ONNX mode.
    /// <code>
    /// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     inputType: InputType.ThreeDimensional,
    ///     inputHeight: 224, inputWidth: 224, inputDepth: 3);
    /// var model = new InternVideo2&lt;double&gt;(arch, "internvideo2.onnx");
    /// var embedding = model.EncodeVideo(videoFrames);
    /// </code>
    /// </para>
    /// </remarks>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    public InternVideo2(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int embedDim = 768)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"InternVideo2 ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _embedDim = embedDim;
        _numHeads = 12;
        _numEncoderLayers = 12;
        _numFrames = 8;
        _patchSize = 14;
        _imageSize = architecture.InputHeight > 0 ? architecture.InputHeight : 224;
        _lossFunction = new MeanSquaredErrorLoss<T>();

        try
        {
            _onnxSession = new InferenceSession(onnxModelPath);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load ONNX model: {ex.Message}", ex);
        }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Encodes video frames into an embedding vector.
    /// </summary>
    /// <param name="videoFrames">Video frames tensor [B, C, H, W] or [C, H, W].</param>
    /// <returns>Video embedding tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method converts video frames into a fixed-size
    /// vector that represents the video content. Similar videos will have similar embeddings.
    /// </para>
    /// </remarks>
    public Tensor<T> EncodeVideo(Tensor<T> videoFrames)
    {
        if (videoFrames is null)
            throw new ArgumentNullException(nameof(videoFrames));

        if (_useNativeMode)
        {
            return Forward(videoFrames);
        }
        else
        {
            return PredictOnnx(videoFrames);
        }
    }

    /// <summary>
    /// Computes similarity between video and text embeddings.
    /// </summary>
    /// <param name="videoEmbedding">Video embedding from EncodeVideo.</param>
    /// <param name="textEmbedding">Text embedding from a text encoder.</param>
    /// <returns>Cosine similarity score.</returns>
    public T ComputeSimilarity(Tensor<T> videoEmbedding, Tensor<T> textEmbedding)
    {
        if (videoEmbedding is null)
            throw new ArgumentNullException(nameof(videoEmbedding));
        if (textEmbedding is null)
            throw new ArgumentNullException(nameof(textEmbedding));

        return ComputeCosineSimilarity(videoEmbedding, textEmbedding);
    }

    #endregion

    #region Inference

    /// <summary>
    /// Performs a forward pass through the network.
    /// </summary>
    private Tensor<T> Forward(Tensor<T> input)
    {
        var result = input;
        foreach (var layer in Layers)
        {
            result = layer.Forward(result);
        }
        return result;
    }

    /// <summary>
    /// Performs inference using the ONNX model.
    /// </summary>
    private Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            inputData[i] = Convert.ToSingle(input.Data.Span[i]);
        }

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);
        var inputMeta = _onnxSession.InputMetadata;
        string inputName = inputMeta.Keys.First();

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, onnxInput)
        };

        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputShape = outputTensor.Dimensions.ToArray();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
        {
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        }

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return EncodeVideo(input);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is not supported in ONNX mode. Use native mode for training.");

        var prediction = Predict(input);
        var loss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());
        LastLoss = loss;

        var outputGradient = _lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
        var outputGradientTensor = new Tensor<T>(prediction.Shape, outputGradient);

        var currentGradient = outputGradientTensor;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            currentGradient = Layers[i].Backward(currentGradient);
        }

        if (_optimizer != null)
        {
            _optimizer.UpdateParameters(Layers);
        }
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Computes cosine similarity between two tensors.
    /// </summary>
    private T ComputeCosineSimilarity(Tensor<T> a, Tensor<T> b)
    {
        double dotProduct = 0;
        double normA = 0;
        double normB = 0;

        int minLen = Math.Min(a.Data.Length, b.Data.Length);
        for (int i = 0; i < minLen; i++)
        {
            double valA = NumOps.ToDouble(a.Data.Span[i]);
            double valB = NumOps.ToDouble(b.Data.Span[i]);
            dotProduct += valA * valB;
            normA += valA * valA;
            normB += valB * valB;
        }

        double similarity = dotProduct / (Math.Sqrt(normA) * Math.Sqrt(normB) + 1e-8);
        return NumOps.FromDouble(similarity);
    }

    #endregion

    #region Layer Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
        {
            ClearLayers();
            return;
        }

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            int inputChannels = Architecture.InputDepth > 0 ? Architecture.InputDepth : 3;
            int inputHeight = Architecture.InputHeight > 0 ? Architecture.InputHeight : 224;
            int inputWidth = Architecture.InputWidth > 0 ? Architecture.InputWidth : 224;

            Layers.AddRange(LayerHelper<T>.CreateDefaultInternVideo2Layers(
                inputChannels: inputChannels,
                inputHeight: inputHeight,
                inputWidth: inputWidth,
                embedDim: _embedDim,
                numEncoderLayers: _numEncoderLayers,
                patchSize: _patchSize));
        }
    }

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Parameter updates are not supported in ONNX mode.");

        int index = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            var layerParameters = parameters.Slice(index, layerParameterCount);
            layer.UpdateParameters(layerParameters);
            index += layerParameterCount;
        }
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var additionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "InternVideo2" },
            { "EmbedDim", _embedDim },
            { "NumHeads", _numHeads },
            { "NumEncoderLayers", _numEncoderLayers },
            { "NumFrames", _numFrames },
            { "PatchSize", _patchSize },
            { "ImageSize", _imageSize },
            { "UseNativeMode", _useNativeMode }
        };

        if (!_useNativeMode && _onnxModelPath != null)
        {
            additionalInfo["OnnxModelPath"] = _onnxModelPath;
        }

        return new ModelMetadata<T>
        {
            ModelType = ModelType.VideoActionRecognition,
            AdditionalInfo = additionalInfo,
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Serialization is not supported in ONNX mode.");

        writer.Write(_embedDim);
        writer.Write(_numHeads);
        writer.Write(_numEncoderLayers);
        writer.Write(_numFrames);
        writer.Write(_patchSize);
        writer.Write(_imageSize);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Deserialization is not supported in ONNX mode.");

        _ = reader.ReadInt32(); // embedDim
        _ = reader.ReadInt32(); // numHeads
        _ = reader.ReadInt32(); // numEncoderLayers
        _ = reader.ReadInt32(); // numFrames
        _ = reader.ReadInt32(); // patchSize
        _ = reader.ReadInt32(); // imageSize
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new InternVideo2<T>(
            Architecture,
            _optimizer,
            _lossFunction,
            _embedDim,
            _numHeads,
            _numEncoderLayers,
            _numFrames,
            _patchSize);
    }

    #endregion
}
