using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Options;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements the Vision Transformer (ViT) architecture for image classification tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for computations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// The Vision Transformer applies transformer architecture, originally designed for natural language processing,
/// to computer vision tasks. It divides images into fixed-size patches, linearly embeds them, adds positional
/// embeddings, and processes the sequence through transformer encoder layers.
/// </para>
/// <para>
/// <b>For Beginners:</b> The Vision Transformer (ViT) is a modern approach to understanding images using transformers.
///
/// Unlike traditional neural networks that process images pixel by pixel or with sliding windows (convolutions),
/// ViT treats an image like a sentence of words:
/// - First, it cuts the image into small square patches (like breaking a sentence into words)
/// - Each patch gets converted to a numerical representation (like word embeddings)
/// - Position information is added so the model knows where each patch came from
/// - A special classification token is added to gather information about the whole image
/// - Transformer layers process all patches together, learning relationships between them
/// - Finally, the classification token's output is used to predict the image class
///
/// This approach has been very successful and often outperforms traditional convolutional neural networks,
/// especially when trained on large datasets.
/// </para>
/// </remarks>
public class VisionTransformer<T> : NeuralNetworkBase<T>
{
    private readonly VisionTransformerOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// The size of each square patch.
    /// </summary>
    private readonly int _patchSize;

    /// <summary>
    /// The number of transformer encoder layers.
    /// </summary>
    private readonly int _numLayers;

    /// <summary>
    /// The number of attention heads in each transformer layer.
    /// </summary>
    private readonly int _numHeads;

    /// <summary>
    /// The dimension of the embedding vectors.
    /// </summary>
    private readonly int _hiddenDim;

    /// <summary>
    /// The dimension of the feed-forward network in each transformer layer.
    /// </summary>
    private readonly int _mlpDim;

    /// <summary>
    /// The height of input images.
    /// </summary>
    private readonly int _imageHeight;

    /// <summary>
    /// The width of input images.
    /// </summary>
    private readonly int _imageWidth;

    /// <summary>
    /// The number of color channels in input images.
    /// </summary>
    private readonly int _channels;

    /// <summary>
    /// The number of output classes.
    /// </summary>
    private readonly int _numClasses;

    /// <summary>
    /// The total number of patches.
    /// </summary>
    private readonly int _numPatches;

    /// <summary>
    /// The classification token embedding.
    /// </summary>
    private Vector<T> _clsToken;

    /// <summary>
    /// The positional embeddings for all patches plus the classification token.
    /// </summary>
    private Matrix<T> _positionalEmbeddings;

    /// <summary>
    /// The final classification head (MLP).
    /// </summary>
    private DenseLayer<T> _classificationHead = default!;

    /// <summary>
    /// Indicates whether this network supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Creates a new Vision Transformer with the specified configuration.
    /// </summary>
    /// <param name="architecture">The architecture defining the network structure.</param>
    /// <param name="imageHeight">The height of input images.</param>
    /// <param name="imageWidth">The width of input images.</param>
    /// <param name="channels">The number of color channels (e.g., 3 for RGB).</param>
    /// <param name="patchSize">The size of each square patch.</param>
    /// <param name="numClasses">The number of output classes.</param>
    /// <param name="hiddenDim">The dimension of embeddings (default: 768).</param>
    /// <param name="numLayers">The number of transformer encoder layers (default: 12).</param>
    /// <param name="numHeads">The number of attention heads (default: 12).</param>
    /// <param name="mlpDim">The dimension of the feed-forward network (default: 3072).</param>
    /// <param name="lossFunction">The loss function to use (defaults to categorical cross-entropy if null).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a Vision Transformer with your specifications.
    ///
    /// The parameters let you customize:
    /// - Image dimensions: Size and channels of input images
    /// - Patch size: How to divide images (common values: 16 or 32)
    /// - Architecture depth: More layers = more capacity but slower and needs more data
    /// - Hidden dimensions: Larger = more expressive but more parameters
    /// - Attention heads: More heads = more diverse attention patterns
    ///
    /// Common configurations:
    /// - ViT-Base: hiddenDim=768, numLayers=12, numHeads=12, mlpDim=3072
    /// - ViT-Large: hiddenDim=1024, numLayers=24, numHeads=16, mlpDim=4096
    /// - ViT-Huge: hiddenDim=1280, numLayers=32, numHeads=16, mlpDim=5120
    /// </para>
    /// </remarks>
    public VisionTransformer(
        NeuralNetworkArchitecture<T> architecture,
        int imageHeight,
        int imageWidth,
        int channels,
        int patchSize,
        int numClasses,
        int hiddenDim = 768,
        int numLayers = 12,
        int numHeads = 12,
        int mlpDim = 3072,
        ILossFunction<T>? lossFunction = null,
        VisionTransformerOptions? options = null)
        : base(architecture, lossFunction ?? new CategoricalCrossEntropyLoss<T>())
    {
        _options = options ?? new VisionTransformerOptions();
        Options = _options;

        // Validate all parameters are positive
        if (imageHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(imageHeight), imageHeight, "Image height must be greater than 0.");
        if (imageWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(imageWidth), imageWidth, "Image width must be greater than 0.");
        if (channels <= 0)
            throw new ArgumentOutOfRangeException(nameof(channels), channels, "Number of channels must be greater than 0.");
        if (patchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(patchSize), patchSize, "Patch size must be greater than 0.");
        if (numClasses <= 0)
            throw new ArgumentOutOfRangeException(nameof(numClasses), numClasses, "Number of classes must be greater than 0.");
        if (hiddenDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(hiddenDim), hiddenDim, "Hidden dimension must be greater than 0.");
        if (numLayers <= 0)
            throw new ArgumentOutOfRangeException(nameof(numLayers), numLayers, "Number of layers must be greater than 0.");
        if (numHeads <= 0)
            throw new ArgumentOutOfRangeException(nameof(numHeads), numHeads, "Number of heads must be greater than 0.");
        if (mlpDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(mlpDim), mlpDim, "MLP dimension must be greater than 0.");

        // Validate image dimensions are divisible by patch size (prevents silent truncation)
        if (imageHeight % patchSize != 0)
            throw new ArgumentException($"Image height ({imageHeight}) must be divisible by patch size ({patchSize}).", nameof(imageHeight));
        if (imageWidth % patchSize != 0)
            throw new ArgumentException($"Image width ({imageWidth}) must be divisible by patch size ({patchSize}).", nameof(imageWidth));

        // Validate multi-head attention requirement
        if (hiddenDim % numHeads != 0)
            throw new ArgumentException($"Hidden dimension ({hiddenDim}) must be divisible by number of heads ({numHeads}) for multi-head attention.", nameof(hiddenDim));

        _imageHeight = imageHeight;
        _imageWidth = imageWidth;
        _channels = channels;
        _patchSize = patchSize;
        _numClasses = numClasses;
        _hiddenDim = hiddenDim;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _mlpDim = mlpDim;

        _numPatches = (imageHeight / patchSize) * (imageWidth / patchSize);

        _clsToken = new Vector<T>(hiddenDim);
        _positionalEmbeddings = new Matrix<T>(_numPatches + 1, hiddenDim);

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the Vision Transformer.
    /// </summary>
    protected override void InitializeLayers()
    {
        ClearLayers();

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateVisionTransformerLayers(
                _imageHeight, _imageWidth, _channels, _patchSize,
                _hiddenDim, _numLayers, _numHeads, _mlpDim, _numClasses));
        }

        // Last layer is the classification head
        _classificationHead = (DenseLayer<T>)Layers[^1];

        InitializeClassificationToken();
        InitializePositionalEmbeddings();
    }

    /// <summary>
    /// Initializes the classification token with random values.
    /// </summary>
    private void InitializeClassificationToken()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(1.0 / _hiddenDim));
        for (int i = 0; i < _hiddenDim; i++)
        {
            _clsToken[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
        }
    }

    /// <summary>
    /// Initializes the positional embeddings with random values.
    /// </summary>
    private void InitializePositionalEmbeddings()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(1.0 / _hiddenDim));
        for (int i = 0; i < _positionalEmbeddings.Rows; i++)
        {
            for (int j = 0; j < _positionalEmbeddings.Columns; j++)
            {
                _positionalEmbeddings[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
            }
        }
    }

    /// <summary>
    /// Makes a prediction using the Vision Transformer.
    /// </summary>
    /// <param name="input">The input image tensor with shape [batch, channels, height, width].</param>
    /// <returns>The predicted class probabilities with shape [batch, num_classes].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method processes an image to predict its class.
    ///
    /// The prediction process:
    /// 1. Convert the image into patches and embed them
    /// 2. Add a classification token to the beginning of the sequence
    /// 3. Add positional embeddings so the model knows where each patch came from
    /// 4. Process through transformer encoder layers
    /// 5. Extract the classification token's representation
    /// 6. Pass through the classification head to get class probabilities
    ///
    /// The output is a probability distribution over classes (values sum to 1),
    /// where higher values indicate more confidence in that class.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        Tensor<T> input4D;

        // Support both 3D [C, H, W] and 4D [B, C, H, W] input
        if (input.Shape.Length == 3)
        {
            // Add batch dimension: [C, H, W] -> [1, C, H, W]
            input4D = input.Reshape(1, input.Shape[0], input.Shape[1], input.Shape[2]);
        }
        else if (input.Shape.Length == 4)
        {
            input4D = input;
        }
        else
        {
            throw new ArgumentException(
                $"Input must be a 3D tensor [C, H, W] or 4D tensor [B, C, H, W], but got rank {input.Shape.Length}.",
                nameof(input));
        }

        if (input4D.Shape[1] != _channels || input4D.Shape[2] != _imageHeight || input4D.Shape[3] != _imageWidth)
        {
            throw new ArgumentException(
                $"Input shape {string.Join("x", input4D.Shape)} does not match expected " +
                $"[batch, {_channels}, {_imageHeight}, {_imageWidth}].",
                nameof(input));
        }

        int batchSize = input4D.Shape[0];

        // Use input4D for the forward pass
        input = input4D;

        var patchEmbeddings = Layers[0].Forward(input);

        var sequenceWithCls = new Tensor<T>([batchSize, _numPatches + 1, _hiddenDim]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < _hiddenDim; d++)
            {
                sequenceWithCls[b, 0, d] = _clsToken[d];
            }

            for (int p = 0; p < _numPatches; p++)
            {
                for (int d = 0; d < _hiddenDim; d++)
                {
                    sequenceWithCls[b, p + 1, d] = patchEmbeddings[b, p, d];
                }
            }
        }

        for (int b = 0; b < batchSize; b++)
        {
            for (int p = 0; p < _numPatches + 1; p++)
            {
                for (int d = 0; d < _hiddenDim; d++)
                {
                    sequenceWithCls[b, p, d] = NumOps.Add(
                        sequenceWithCls[b, p, d],
                        _positionalEmbeddings[p, d]);
                }
            }
        }

        var current = sequenceWithCls;
        for (int i = 1; i < Layers.Count - 1; i++)
        {
            current = Layers[i].Forward(current);
        }

        var clsOutput = new Tensor<T>([batchSize, _hiddenDim]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < _hiddenDim; d++)
            {
                clsOutput[b, d] = current[b, 0, d];
            }
        }

        var reshapedForClassification = new Tensor<T>([batchSize, 1, _hiddenDim]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < _hiddenDim; d++)
            {
                reshapedForClassification[b, 0, d] = clsOutput[b, d];
            }
        }

        var output = _classificationHead.Forward(reshapedForClassification);

        var finalOutput = new Tensor<T>([batchSize, _numClasses]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < _numClasses; c++)
            {
                finalOutput[b, c] = output[b, 0, c];
            }
        }

        return finalOutput;
    }

    /// <summary>
    /// Trains the Vision Transformer on a single input-output pair.
    /// </summary>
    /// <param name="input">The input image tensor.</param>
    /// <param name="expectedOutput">The expected output (class labels or probabilities).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method trains the Vision Transformer on one example.
    ///
    /// During training:
    /// 1. Forward pass: Make a prediction using the current parameters
    /// 2. Calculate loss: Measure how wrong the prediction is
    /// 3. Backward pass: Calculate gradients for all parameters
    /// 4. Update parameters: Adjust weights and biases to improve performance
    ///
    /// This is typically called many times with different images to train the model.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);

        var prediction = ForwardWithMemory(input);

        LastLoss = LossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

        var lossGradient = LossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());

        Backpropagate(Tensor<T>.FromVector(lossGradient));
    }

    /// <summary>
    /// Updates the network's parameters with new values.
    /// </summary>
    /// <param name="parameters">The new parameter values.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method sets all the network's internal values at once.
    /// This is typically used when loading a saved model or when an optimizer
    /// computes improved parameter values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int totalExpected = ParameterCount;
        if (parameters.Length != totalExpected)
        {
            throw new ArgumentException($"Expected {totalExpected} parameters, but got {parameters.Length}", nameof(parameters));
        }

        int currentIndex = 0;

        for (int i = 0; i < _clsToken.Length; i++)
        {
            _clsToken[i] = parameters[currentIndex++];
        }

        for (int i = 0; i < _positionalEmbeddings.Rows; i++)
        {
            for (int j = 0; j < _positionalEmbeddings.Columns; j++)
            {
                _positionalEmbeddings[i, j] = parameters[currentIndex++];
            }
        }

        foreach (var layer in Layers)
        {
            int layerParamCount = layer.ParameterCount;
            if (layerParamCount > 0)
            {
                var layerParams = new Vector<T>(layerParamCount);
                for (int i = 0; i < layerParamCount; i++)
                {
                    layerParams[i] = parameters[currentIndex++];
                }
                layer.SetParameters(layerParams);
            }
        }
    }

    /// <summary>
    /// Gets the model metadata.
    /// </summary>
    /// <returns>Metadata describing the Vision Transformer model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method returns information about the model like its type,
    /// parameter count, and configuration. This is useful for documentation and debugging.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "VisionTransformer",
            ModelType = ModelType.Transformer,
            FeatureCount = _imageHeight * _imageWidth * _channels,
            Complexity = ParameterCount,
            Description = $"Vision Transformer with {_numLayers} layers, {_numHeads} attention heads, and {_numPatches} patches",
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ImageHeight", _imageHeight },
                { "ImageWidth", _imageWidth },
                { "Channels", _channels },
                { "PatchSize", _patchSize },
                { "NumClasses", _numClasses },
                { "HiddenDim", _hiddenDim },
                { "NumLayers", _numLayers },
                { "NumHeads", _numHeads },
                { "MlpDim", _mlpDim },
                { "NumPatches", _numPatches }
            }
        };
        return metadata;
    }

    /// <summary>
    /// Serializes Vision Transformer-specific data.
    /// </summary>
    /// <param name="writer">The binary writer to write data to.</param>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_imageHeight);
        writer.Write(_imageWidth);
        writer.Write(_channels);
        writer.Write(_patchSize);
        writer.Write(_numClasses);
        writer.Write(_hiddenDim);
        writer.Write(_numLayers);
        writer.Write(_numHeads);
        writer.Write(_mlpDim);

        for (int i = 0; i < _clsToken.Length; i++)
        {
            writer.Write(Convert.ToDouble(_clsToken[i]));
        }

        for (int i = 0; i < _positionalEmbeddings.Rows; i++)
        {
            for (int j = 0; j < _positionalEmbeddings.Columns; j++)
            {
                writer.Write(Convert.ToDouble(_positionalEmbeddings[i, j]));
            }
        }
    }

    /// <summary>
    /// Deserializes Vision Transformer-specific data.
    /// </summary>
    /// <param name="reader">The binary reader to read data from.</param>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        int imageHeight = reader.ReadInt32();
        int imageWidth = reader.ReadInt32();
        int channels = reader.ReadInt32();
        int patchSize = reader.ReadInt32();
        int numClasses = reader.ReadInt32();
        int hiddenDim = reader.ReadInt32();
        int numLayers = reader.ReadInt32();
        int numHeads = reader.ReadInt32();
        int mlpDim = reader.ReadInt32();

        if (imageHeight != _imageHeight || imageWidth != _imageWidth ||
            channels != _channels || patchSize != _patchSize ||
            numClasses != _numClasses || hiddenDim != _hiddenDim ||
            numLayers != _numLayers || numHeads != _numHeads ||
            mlpDim != _mlpDim)
        {
            throw new InvalidOperationException(
                $"Serialized model configuration does not match current instance. " +
                $"Expected: {_imageHeight}x{_imageWidth}x{_channels}, patch={_patchSize}, " +
                $"classes={_numClasses}, hidden={_hiddenDim}, layers={_numLayers}, " +
                $"heads={_numHeads}, mlp={_mlpDim}. " +
                $"Got: {imageHeight}x{imageWidth}x{channels}, patch={patchSize}, " +
                $"classes={numClasses}, hidden={hiddenDim}, layers={numLayers}, " +
                $"heads={numHeads}, mlp={mlpDim}.");
        }

        for (int i = 0; i < _clsToken.Length; i++)
        {
            _clsToken[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        for (int i = 0; i < _positionalEmbeddings.Rows; i++)
        {
            for (int j = 0; j < _positionalEmbeddings.Columns; j++)
            {
                _positionalEmbeddings[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }
    }

    /// <summary>
    /// Creates a new instance of the Vision Transformer.
    /// </summary>
    /// <returns>A new Vision Transformer instance with the same configuration.</returns>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new VisionTransformer<T>(
            Architecture,
            _imageHeight,
            _imageWidth,
            _channels,
            _patchSize,
            _numClasses,
            _hiddenDim,
            _numLayers,
            _numHeads,
            _mlpDim,
            LossFunction);
    }

    /// <summary>
    /// Gets all model parameters in a single vector.
    /// </summary>
    /// <returns>A vector containing CLS token, positional embeddings, and all layer parameters in sequence.</returns>
    /// <remarks>
    /// <para>
    /// This method returns parameters in the exact order expected by UpdateParameters:
    /// 1. CLS token vector
    /// 2. Positional embeddings (flattened row-major)
    /// 3. Parameters from each transformer layer in sequence
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        int totalCount = ParameterCount;
        var parameters = new Vector<T>(totalCount);
        int currentIndex = 0;

        // Pack CLS token
        for (int i = 0; i < _clsToken.Length; i++)
        {
            parameters[currentIndex++] = _clsToken[i];
        }

        // Pack positional embeddings (row-major)
        for (int i = 0; i < _positionalEmbeddings.Rows; i++)
        {
            for (int j = 0; j < _positionalEmbeddings.Columns; j++)
            {
                parameters[currentIndex++] = _positionalEmbeddings[i, j];
            }
        }

        // Pack layer parameters
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            for (int i = 0; i < layerParams.Length; i++)
            {
                parameters[currentIndex++] = layerParams[i];
            }
        }

        return parameters;
    }

    /// <summary>
    /// Gets the total number of parameters in the model.
    /// </summary>
    public override int ParameterCount
    {
        get
        {
            int count = _clsToken.Length;
            count += _positionalEmbeddings.Rows * _positionalEmbeddings.Columns;
            count += Layers.Sum(layer => layer.ParameterCount);
            return count;
        }
    }
}

