using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements the SpiralNet++ architecture for mesh-based deep learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// SpiralNet++ processes 3D meshes by applying convolutions along spiral sequences
/// of vertex neighbors. This creates translation-equivariant operations on
/// irregular mesh structures without requiring mesh registration.
/// </para>
/// <para><b>For Beginners:</b> SpiralNet++ is designed for learning from 3D mesh data.
/// 
/// Key concepts:
/// - Mesh: A 3D surface made of vertices connected by edges/triangles
/// - Spiral ordering: A consistent way to visit vertex neighbors (like a clock hand)
/// - Spiral convolution: Apply weights to neighbors in spiral order
/// 
/// How it works:
/// 1. For each vertex, define a spiral ordering of its neighbors
/// 2. Gather neighbor features in spiral order
/// 3. Apply learned weights to the gathered features
/// 4. Pool vertices to create hierarchical representations
/// 5. Classify or segment the mesh
/// 
/// Applications:
/// - 3D face reconstruction and expression recognition
/// - Human body shape analysis
/// - Medical surface analysis (organs, bones)
/// - CAD model classification
/// </para>
/// <para>
/// Reference: "SpiralNet++: A Fast and Highly Efficient Mesh Convolution Operator" by Gong et al.
/// </para>
/// </remarks>
public class SpiralNet<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// The loss function used to compute training loss.
    /// </summary>
    private readonly ILossFunction<T> _lossFunction;

    /// <summary>
    /// The optimizer used to update network parameters.
    /// </summary>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// The configuration options for this SpiralNet instance.
    /// </summary>
    private SpiralNetOptions _options;

    /// <summary>
    /// Cached spiral indices for each resolution level.
    /// </summary>
    private readonly List<int[,]> _spiralIndicesPerLevel;

    /// <summary>
    /// Gets the number of output classes for classification.
    /// </summary>
    public int NumClasses => _options.NumClasses;

    /// <summary>
    /// Gets the number of input features per vertex.
    /// </summary>
    public int InputFeatures => _options.InputFeatures;

    /// <summary>
    /// Gets the spiral sequence length.
    /// </summary>
    public int SpiralLength => _options.SpiralLength;

    /// <summary>
    /// Gets the channel configuration for spiral convolution layers.
    /// </summary>
    public int[] ConvChannels => _options.ConvChannels;

    /// <summary>
    /// Initializes a new instance of the <see cref="SpiralNet{T}"/> class with default options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Creates a SpiralNet with default configuration suitable for common mesh tasks.
    /// </para>
    /// </remarks>
    public SpiralNet()
        : this(new SpiralNetOptions(), null, null)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="SpiralNet{T}"/> class with specified options.
    /// </summary>
    /// <param name="options">Configuration options for the SpiralNet.</param>
    /// <param name="optimizer">The optimizer for training. Defaults to Adam if null.</param>
    /// <param name="lossFunction">The loss function. Defaults based on task type if null.</param>
    /// <exception cref="ArgumentNullException">Thrown when options is null.</exception>
    public SpiralNet(
        SpiralNetOptions options,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(
            CreateArchitecture(options),
            lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            1.0)
    {
        if (options == null)
            throw new ArgumentNullException(nameof(options));

        ValidateOptions(options);

        _options = options;
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _spiralIndicesPerLevel = [];

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="SpiralNet{T}"/> class with simple parameters.
    /// </summary>
    /// <param name="numClasses">Number of output classes for classification.</param>
    /// <param name="inputFeatures">Number of input features per vertex. Default is 3.</param>
    /// <param name="spiralLength">Length of spiral sequences. Default is 9.</param>
    /// <param name="lossFunction">The loss function. Defaults based on task type if null.</param>
    public SpiralNet(
        int numClasses,
        int inputFeatures = 3,
        int spiralLength = 9,
        ILossFunction<T>? lossFunction = null)
        : this(
            new SpiralNetOptions
            {
                NumClasses = numClasses,
                InputFeatures = inputFeatures,
                SpiralLength = spiralLength
            },
            null,
            lossFunction)
    {
    }

    /// <summary>
    /// Creates the neural network architecture for SpiralNet.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    /// <returns>The architecture specification.</returns>
    private static NeuralNetworkArchitecture<T> CreateArchitecture(SpiralNetOptions options)
    {
        return new NeuralNetworkArchitecture<T>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Medium,
            inputHeight: 1,
            inputWidth: 1,
            inputDepth: options.InputFeatures,
            outputSize: options.NumClasses);
    }

    /// <summary>
    /// Validates configuration options.
    /// </summary>
    /// <param name="options">Options to validate.</param>
    /// <exception cref="ArgumentException">Thrown when options are invalid.</exception>
    private static void ValidateOptions(SpiralNetOptions options)
    {
        if (options.NumClasses <= 0)
            throw new ArgumentException("NumClasses must be positive.", nameof(options));
        if (options.InputFeatures <= 0)
            throw new ArgumentException("InputFeatures must be positive.", nameof(options));
        if (options.SpiralLength <= 0)
            throw new ArgumentException("SpiralLength must be positive.", nameof(options));
        if (options.ConvChannels == null || options.ConvChannels.Length == 0)
            throw new ArgumentException("ConvChannels must not be empty.", nameof(options));
        if (options.PoolRatios == null)
            throw new ArgumentException("PoolRatios must not be null.", nameof(options));
        if (options.DropoutRate < 0 || options.DropoutRate >= 1)
            throw new ArgumentException("DropoutRate must be in [0, 1).", nameof(options));
    }

    /// <summary>
    /// Initializes the layers of the SpiralNet network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// If the architecture provides custom layers, those are used. Otherwise,
    /// default layers are created using <see cref="LayerHelper{T}.CreateDefaultSpiralNetLayers"/>.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultSpiralNetLayers(
                Architecture,
                _options.InputFeatures,
                _options.SpiralLength,
                _options.ConvChannels,
                _options.PoolRatios,
                _options.FullyConnectedSizes,
                _options.UseBatchNorm,
                _options.DropoutRate,
                _options.UseGlobalAveragePooling));
        }
    }

    /// <summary>
    /// Validates custom layers for compatibility with SpiralNet architecture.
    /// </summary>
    /// <param name="layers">Layers to validate.</param>
    /// <exception cref="ArgumentException">Thrown when layers are invalid.</exception>
    private void ValidateCustomLayers(IList<ILayer<T>> layers)
    {
        if (layers.Count == 0)
            throw new ArgumentException("Layer list cannot be empty.", nameof(layers));

        var lastLayer = layers[^1];
        var outputShape = lastLayer.GetOutputShape();
        if (outputShape == null || outputShape.Length == 0)
            throw new ArgumentException("Last layer must have defined output shape.", nameof(layers));
    }

    /// <summary>
    /// Sets the spiral indices for the current mesh being processed.
    /// </summary>
    /// <param name="spiralIndices">
    /// A 2D array of shape [numVertices, SpiralLength] containing neighbor vertex indices
    /// in spiral order for each vertex.
    /// </param>
    /// <exception cref="ArgumentNullException">Thrown when spiralIndices is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Before processing a mesh, you must define how
    /// vertices are connected in spiral order. This method sets that connectivity.</para>
    /// </remarks>
    public void SetSpiralIndices(int[,] spiralIndices)
    {
        if (spiralIndices == null)
            throw new ArgumentNullException(nameof(spiralIndices));

        _spiralIndicesPerLevel.Clear();
        _spiralIndicesPerLevel.Add(spiralIndices);

        PropagateSpiralIndicesToLayers();
    }

    /// <summary>
    /// Sets spiral indices for multiple resolution levels (for hierarchical processing).
    /// </summary>
    /// <param name="spiralIndicesPerLevel">List of spiral indices for each resolution level.</param>
    /// <exception cref="ArgumentNullException">Thrown when list is null.</exception>
    /// <exception cref="ArgumentException">Thrown when list is empty.</exception>
    public void SetMultiResolutionSpiralIndices(List<int[,]> spiralIndicesPerLevel)
    {
        if (spiralIndicesPerLevel == null)
            throw new ArgumentNullException(nameof(spiralIndicesPerLevel));
        if (spiralIndicesPerLevel.Count == 0)
            throw new ArgumentException("Must provide at least one level of spiral indices.", nameof(spiralIndicesPerLevel));

        _spiralIndicesPerLevel.Clear();
        _spiralIndicesPerLevel.AddRange(spiralIndicesPerLevel);

        PropagateSpiralIndicesToLayers();
    }

    /// <summary>
    /// Propagates spiral indices to all SpiralConv layers.
    /// </summary>
    private void PropagateSpiralIndicesToLayers()
    {
        if (_spiralIndicesPerLevel.Count == 0) return;

        int levelIndex = 0;

        foreach (var layer in Layers)
        {
            if (layer is SpiralConvLayer<T> convLayer)
            {
                int idx = Math.Min(levelIndex, _spiralIndicesPerLevel.Count - 1);
                convLayer.SetSpiralIndices(_spiralIndicesPerLevel[idx]);
            }
            else if (layer is GlobalPoolingLayer<T>)
            {
                levelIndex++;
            }
        }
    }

    /// <summary>
    /// Performs a forward pass through the network.
    /// </summary>
    /// <param name="input">
    /// Vertex features tensor with shape [numVertices, InputFeatures].
    /// </param>
    /// <returns>Classification logits with shape [NumClasses].</returns>
    /// <exception cref="InvalidOperationException">Thrown when spiral indices are not set.</exception>
    public Tensor<T> Forward(Tensor<T> input)
    {
        if (_spiralIndicesPerLevel.Count == 0)
        {
            throw new InvalidOperationException(
                "Spiral indices must be set via SetSpiralIndices before calling Forward.");
        }

        PropagateSpiralIndicesToLayers();

        Tensor<T> output = input;

        foreach (var layer in Layers)
        {
            output = layer.Forward(output);
        }

        return output;
    }

    /// <summary>
    /// Performs a backward pass to compute gradients.
    /// </summary>
    /// <param name="lossGradient">Gradient of the loss with respect to network output.</param>
    /// <returns>Gradient with respect to input.</returns>
    public Tensor<T> Backward(Tensor<T> lossGradient)
    {
        Tensor<T> gradient = lossGradient;

        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradient = Layers[i].Backward(gradient);
        }

        return gradient;
    }

    /// <summary>
    /// Updates network parameters using the optimizer.
    /// </summary>
    /// <param name="learningRate">Learning rate for parameter updates.</param>
    public void UpdateParameters(T learningRate)
    {
        foreach (var layer in Layers)
        {
            if (layer.SupportsTraining)
            {
                layer.UpdateParameters(learningRate);
            }
        }
    }

    /// <summary>
    /// Trains the network on mesh data.
    /// </summary>
    /// <param name="meshFeatures">List of vertex feature tensors for training meshes.</param>
    /// <param name="spiralIndices">List of spiral indices for each training mesh.</param>
    /// <param name="labels">List of class labels for each mesh.</param>
    /// <param name="epochs">Number of training epochs.</param>
    /// <param name="learningRate">Learning rate for optimization.</param>
    /// <returns>Training loss history.</returns>
    /// <exception cref="ArgumentException">Thrown when input lists have mismatched lengths.</exception>
    public List<double> Train(
        List<Tensor<T>> meshFeatures,
        List<int[,]> spiralIndices,
        List<int> labels,
        int epochs,
        T learningRate)
    {
        if (meshFeatures.Count != spiralIndices.Count || meshFeatures.Count != labels.Count)
        {
            throw new ArgumentException("Input lists must have the same length.");
        }

        var lossHistory = new List<double>();

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double epochLoss = 0.0;

            for (int i = 0; i < meshFeatures.Count; i++)
            {
                SetSpiralIndices(spiralIndices[i]);

                var output = Forward(meshFeatures[i]);

                var target = CreateOneHotTarget(labels[i], NumClasses);
                var loss = _lossFunction.CalculateLoss(output.ToVector(), target);
                epochLoss += NumOps.ToDouble(loss);

                var lossGrad = _lossFunction.CalculateDerivative(output.ToVector(), target);
                var lossGradTensor = new Tensor<T>(lossGrad.ToArray(), output.Shape);

                Backward(lossGradTensor);
                UpdateParameters(learningRate);
            }

            lossHistory.Add(epochLoss / meshFeatures.Count);
        }

        return lossHistory;
    }

    /// <summary>
    /// Creates a one-hot encoded target vector.
    /// </summary>
    /// <param name="classIndex">The class index.</param>
    /// <param name="numClasses">Total number of classes.</param>
    /// <returns>One-hot encoded vector.</returns>
    private Vector<T> CreateOneHotTarget(int classIndex, int numClasses)
    {
        var target = new T[numClasses];
        for (int i = 0; i < numClasses; i++)
        {
            target[i] = i == classIndex ? NumOps.One : NumOps.Zero;
        }
        return new Vector<T>(target);
    }

    /// <summary>
    /// Predicts the class for a single mesh.
    /// </summary>
    /// <param name="meshFeatures">Vertex features tensor.</param>
    /// <param name="meshSpiralIndices">Spiral indices for the mesh.</param>
    /// <returns>Predicted class index.</returns>
    public int PredictClass(Tensor<T> meshFeatures, int[,] meshSpiralIndices)
    {
        SetSpiralIndices(meshSpiralIndices);
        var output = Forward(meshFeatures);

        var outputArray = output.ToArray();
        int predictedClass = 0;
        T maxValue = outputArray[0];

        for (int i = 1; i < outputArray.Length; i++)
        {
            if (NumOps.GreaterThan(outputArray[i], maxValue))
            {
                maxValue = outputArray[i];
                predictedClass = i;
            }
        }

        return predictedClass;
    }

    /// <summary>
    /// Generates predictions for the given input.
    /// </summary>
    /// <param name="input">Vertex features tensor.</param>
    /// <returns>Classification logits.</returns>
    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        if (_spiralIndicesPerLevel.Count == 0)
        {
            throw new InvalidOperationException(
                "Spiral indices must be set via SetSpiralIndices before calling Predict.");
        }
        return Forward(input);
    }

    /// <summary>
    /// Trains the network on a single batch.
    /// </summary>
    /// <param name="input">Vertex features tensor.</param>
    /// <param name="expectedOutput">Ground truth labels.</param>
    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        var prediction = Forward(input);

        var loss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());
        LastLoss = loss;

        var outputGradient = _lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
        var outputGradientTensor = new Tensor<T>(prediction.Shape, outputGradient);

        var gradients = new List<Tensor<T>>();
        var currentGradient = outputGradientTensor;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            currentGradient = Layers[i].Backward(currentGradient);
            gradients.Insert(0, currentGradient);
        }

        ClipGradients(gradients);
        _optimizer.UpdateParameters(Layers);
    }

    /// <summary>
    /// Updates network parameters using a flat parameter vector.
    /// </summary>
    /// <param name="parameters">Vector containing all parameters.</param>
    /// <inheritdoc />
    public override void UpdateParameters(Vector<T> parameters)
    {
        int index = 0;
        foreach (var layer in Layers)
        {
            int layerParams = layer.ParameterCount;
            var layerParameters = parameters.Slice(index, layerParams);
            layer.UpdateParameters(layerParameters);
            index += layerParams;
        }
    }

    /// <summary>
    /// Gets metadata about this model.
    /// </summary>
    /// <returns>Model metadata.</returns>
    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.SpiralNet,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumClasses", _options.NumClasses },
                { "InputFeatures", _options.InputFeatures },
                { "SpiralLength", _options.SpiralLength },
                { "ConvChannels", _options.ConvChannels },
                { "PoolRatios", _options.PoolRatios },
                { "UseBatchNorm", _options.UseBatchNorm },
                { "DropoutRate", _options.DropoutRate },
                { "LayerCount", Layers.Count }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes network-specific data.
    /// </summary>
    /// <param name="writer">Binary writer.</param>
    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_options.NumClasses);
        writer.Write(_options.InputFeatures);
        writer.Write(_options.SpiralLength);
        writer.Write(_options.UseBatchNorm);
        writer.Write(_options.DropoutRate);
        writer.Write(_options.UseGlobalAveragePooling);

        writer.Write(_options.ConvChannels.Length);
        foreach (var ch in _options.ConvChannels)
            writer.Write(ch);

        writer.Write(_options.PoolRatios.Length);
        foreach (var pr in _options.PoolRatios)
            writer.Write(pr);

        writer.Write(_options.FullyConnectedSizes.Length);
        foreach (var fc in _options.FullyConnectedSizes)
            writer.Write(fc);
    }

    /// <summary>
    /// Deserializes network-specific data.
    /// </summary>
    /// <param name="reader">Binary reader.</param>
    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _options.NumClasses = reader.ReadInt32();
        _options.InputFeatures = reader.ReadInt32();
        _options.SpiralLength = reader.ReadInt32();
        _options.UseBatchNorm = reader.ReadBoolean();
        _options.DropoutRate = reader.ReadDouble();
        _options.UseGlobalAveragePooling = reader.ReadBoolean();

        int convLen = reader.ReadInt32();
        _options.ConvChannels = new int[convLen];
        for (int i = 0; i < convLen; i++)
            _options.ConvChannels[i] = reader.ReadInt32();

        int poolLen = reader.ReadInt32();
        _options.PoolRatios = new double[poolLen];
        for (int i = 0; i < poolLen; i++)
            _options.PoolRatios[i] = reader.ReadDouble();

        int fcLen = reader.ReadInt32();
        _options.FullyConnectedSizes = new int[fcLen];
        for (int i = 0; i < fcLen; i++)
            _options.FullyConnectedSizes[i] = reader.ReadInt32();
    }

    /// <summary>
    /// Creates a new instance for cloning.
    /// </summary>
    /// <returns>New SpiralNet instance.</returns>
    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new SpiralNet<T>(_options, _optimizer, _lossFunction);
    }

    /// <summary>
    /// Computes class probabilities for a single mesh using softmax.
    /// </summary>
    /// <param name="meshFeatures">Vertex features tensor.</param>
    /// <param name="meshSpiralIndices">Spiral indices for the mesh.</param>
    /// <returns>Probability distribution over classes.</returns>
    public Vector<T> PredictProbabilities(Tensor<T> meshFeatures, int[,] meshSpiralIndices)
    {
        SetSpiralIndices(meshSpiralIndices);
        var output = Forward(meshFeatures);

        var softmax = new SoftmaxActivation<T>();
        return softmax.Activate(output.ToVector());
    }
}
