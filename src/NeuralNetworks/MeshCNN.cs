using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements the MeshCNN architecture for processing 3D triangle meshes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// MeshCNN is a deep learning architecture that operates directly on 3D mesh data
/// by treating edges as the fundamental unit of computation. This enables learning
/// from the mesh structure itself rather than converting to voxels or point clouds.
/// </para>
/// <para><b>For Beginners:</b> MeshCNN processes 3D shapes represented as triangle meshes.
/// 
/// Key concepts:
/// - Mesh: A 3D surface made of connected triangles (vertices + faces)
/// - Edge: A line segment connecting two vertices, shared by up to 2 faces
/// - Edge features: Properties like dihedral angle, edge ratios, face angles
/// 
/// How it works:
/// 1. Extract edge features from the mesh (5 features per edge by default)
/// 2. Apply edge convolutions to learn patterns in edge neighborhoods
/// 3. Pool edges by removing less important ones (simplifies the mesh)
/// 4. Repeat conv + pool to build hierarchical features
/// 5. Aggregate edge features for classification/segmentation
/// 
/// Applications:
/// - 3D shape classification (e.g., recognize chair vs table)
/// - Mesh segmentation (label each part of a 3D model)
/// - Shape retrieval (find similar 3D models)
/// </para>
/// <para>
/// Reference: "MeshCNN: A Network with an Edge" by Hanocka et al., SIGGRAPH 2019
/// </para>
/// </remarks>
public class MeshCNN<T> : NeuralNetworkBase<T>
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
    /// The configuration options for this MeshCNN instance.
    /// </summary>
    private readonly MeshCNNOptions _options;

    /// <summary>
    /// Cached edge adjacency for the current mesh being processed.
    /// </summary>
    private int[,]? _currentEdgeAdjacency;

    /// <summary>
    /// Gets the number of output classes for classification.
    /// </summary>
    public int NumClasses => _options.NumClasses;

    /// <summary>
    /// Gets the number of input features per edge.
    /// </summary>
    public int InputFeatures => _options.InputFeatures;

    /// <summary>
    /// Gets the channel configuration for edge convolution layers.
    /// </summary>
    public int[] ConvChannels => _options.ConvChannels;

    /// <summary>
    /// Gets the pooling targets for mesh simplification.
    /// </summary>
    public int[] PoolTargets => _options.PoolTargets;

    /// <summary>
    /// Initializes a new instance of the <see cref="MeshCNN{T}"/> class with default options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Creates a MeshCNN with default configuration suitable for ModelNet40 classification.
    /// </para>
    /// </remarks>
    public MeshCNN()
        : this(new MeshCNNOptions(), null, null)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="MeshCNN{T}"/> class with specified options.
    /// </summary>
    /// <param name="options">Configuration options for the MeshCNN.</param>
    /// <param name="optimizer">The optimizer for training. Defaults to Adam if null.</param>
    /// <param name="lossFunction">The loss function. Defaults based on task type if null.</param>
    /// <exception cref="ArgumentNullException">Thrown when options is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a MeshCNN with custom configuration.</para>
    /// </remarks>
    public MeshCNN(
        MeshCNNOptions options,
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

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="MeshCNN{T}"/> class with simple parameters.
    /// </summary>
    /// <param name="numClasses">Number of output classes for classification.</param>
    /// <param name="inputFeatures">Number of input features per edge. Default is 5.</param>
    /// <param name="lossFunction">The loss function. Defaults based on task type if null.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a MeshCNN with default architecture settings.</para>
    /// </remarks>
    public MeshCNN(
        int numClasses,
        int inputFeatures = 5,
        ILossFunction<T>? lossFunction = null)
        : this(
            new MeshCNNOptions
            {
                NumClasses = numClasses,
                InputFeatures = inputFeatures
            },
            null,
            lossFunction)
    {
    }

    /// <summary>
    /// Creates the neural network architecture for MeshCNN.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    /// <returns>The architecture specification.</returns>
    private static NeuralNetworkArchitecture<T> CreateArchitecture(MeshCNNOptions options)
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
    private static void ValidateOptions(MeshCNNOptions options)
    {
        if (options.NumClasses <= 0)
            throw new ArgumentException("NumClasses must be positive.", nameof(options));
        if (options.InputFeatures <= 0)
            throw new ArgumentException("InputFeatures must be positive.", nameof(options));
        if (options.ConvChannels == null || options.ConvChannels.Length == 0)
            throw new ArgumentException("ConvChannels must not be empty.", nameof(options));
        if (options.PoolTargets == null)
            throw new ArgumentException("PoolTargets must not be null.", nameof(options));
        if (options.PoolTargets.Length > options.ConvChannels.Length)
            throw new ArgumentException("PoolTargets must have at most ConvChannels.Length elements.", nameof(options));
        if (options.NumNeighbors <= 0)
            throw new ArgumentException("NumNeighbors must be positive.", nameof(options));
        if (options.DropoutRate < 0 || options.DropoutRate >= 1)
            throw new ArgumentException("DropoutRate must be in [0, 1).", nameof(options));
    }

    /// <summary>
    /// Initializes the layers of the MeshCNN network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// If the architecture provides custom layers, those are used. Otherwise,
    /// default layers are created using <see cref="LayerHelper{T}.CreateDefaultMeshCNNLayers"/>.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultMeshCNNLayers(
                Architecture,
                _options.InputFeatures,
                _options.ConvChannels,
                _options.PoolTargets,
                _options.FullyConnectedSizes,
                _options.NumNeighbors,
                _options.UseBatchNorm,
                _options.DropoutRate,
                _options.UseGlobalAveragePooling));
        }
    }

    /// <summary>
    /// Sets the edge adjacency for the current mesh being processed.
    /// </summary>
    /// <param name="edgeAdjacency">
    /// A 2D array of shape [numEdges, NumNeighbors] containing neighbor edge indices.
    /// </param>
    /// <exception cref="ArgumentNullException">Thrown when edgeAdjacency is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Before processing a mesh, you must tell the network
    /// how edges are connected. This method sets that connectivity information.</para>
    /// <para>
    /// Call this method before each Forward pass with a new mesh.
    /// </para>
    /// </remarks>
    public void SetEdgeAdjacency(int[,] edgeAdjacency)
    {
        if (edgeAdjacency == null)
            throw new ArgumentNullException(nameof(edgeAdjacency));

        _currentEdgeAdjacency = edgeAdjacency;
        PropagateAdjacencyToLayers();
    }

    /// <summary>
    /// Propagates edge adjacency to all MeshEdgeConv and MeshPool layers.
    /// </summary>
    private void PropagateAdjacencyToLayers()
    {
        if (_currentEdgeAdjacency == null) return;

        var currentAdjacency = _currentEdgeAdjacency;

        foreach (var layer in Layers)
        {
            if (layer is MeshEdgeConvLayer<T> convLayer)
            {
                convLayer.SetEdgeAdjacency(currentAdjacency);
            }
            else if (layer is MeshPoolLayer<T> poolLayer)
            {
                poolLayer.SetEdgeAdjacency(currentAdjacency);
            }
        }
    }

    /// <summary>
    /// Performs a forward pass through the network.
    /// </summary>
    /// <param name="input">
    /// Edge features tensor with shape [numEdges, InputFeatures].
    /// </param>
    /// <returns>Classification logits with shape [NumClasses].</returns>
    /// <exception cref="InvalidOperationException">Thrown when edge adjacency is not set.</exception>
    /// <remarks>
    /// <para>
    /// Call <see cref="SetEdgeAdjacency"/> before calling this method.
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for 10-50x speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;


        if (_currentEdgeAdjacency == null)
        {
            throw new InvalidOperationException(
                "Edge adjacency must be set via SetEdgeAdjacency before calling Forward.");
        }

        PropagateAdjacencyToLayers();

        Tensor<T> output = input;
        var currentAdjacency = _currentEdgeAdjacency;

        foreach (var layer in Layers)
        {
            output = layer.Forward(output);

            if (layer is MeshPoolLayer<T> poolLayer && poolLayer.UpdatedAdjacency != null)
            {
                currentAdjacency = poolLayer.UpdatedAdjacency;
                UpdateSubsequentLayerAdjacency(layer, currentAdjacency);
            }
        }

        return output;
    }

    /// <summary>
    /// Updates adjacency for layers following a pooling operation.
    /// </summary>
    /// <param name="poolLayer">The pooling layer that was just executed.</param>
    /// <param name="newAdjacency">The updated adjacency after pooling.</param>
    private void UpdateSubsequentLayerAdjacency(ILayer<T> poolLayer, int[,] newAdjacency)
    {
        bool foundPool = false;
        foreach (var layer in Layers)
        {
            if (layer == poolLayer)
            {
                foundPool = true;
                continue;
            }

            if (foundPool)
            {
                if (layer is MeshEdgeConvLayer<T> convLayer)
                {
                    convLayer.SetEdgeAdjacency(newAdjacency);
                }
                else if (layer is MeshPoolLayer<T> nextPoolLayer)
                {
                    nextPoolLayer.SetEdgeAdjacency(newAdjacency);
                    break;
                }
            }
        }
    }

    /// <summary>
    /// Performs backward pass through the network.
    /// </summary>
    /// <param name="outputGradient">Gradient of loss with respect to output.</param>
    /// <returns>Gradient with respect to input.</returns>
    public Tensor<T> Backward(Tensor<T> outputGradient)
    {
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            outputGradient = Layers[i].Backward(outputGradient);
        }
        return outputGradient;
    }

    /// <summary>
    /// Generates predictions for the given input.
    /// </summary>
    /// <param name="input">Edge features tensor.</param>
    /// <returns>Classification logits.</returns>
    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return Forward(input);
    }

    /// <summary>
    /// Trains the network on a single batch.
    /// </summary>
    /// <param name="input">Edge features tensor.</param>
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
            ModelType = ModelType.MeshCNN,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumClasses", _options.NumClasses },
                { "InputFeatures", _options.InputFeatures },
                { "ConvChannels", _options.ConvChannels },
                { "PoolTargets", _options.PoolTargets },
                { "NumNeighbors", _options.NumNeighbors },
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
        writer.Write(_options.NumNeighbors);
        writer.Write(_options.UseBatchNorm);
        writer.Write(_options.DropoutRate);
        writer.Write(_options.UseGlobalAveragePooling);

        writer.Write(_options.ConvChannels.Length);
        foreach (var ch in _options.ConvChannels)
            writer.Write(ch);

        writer.Write(_options.PoolTargets.Length);
        foreach (var pt in _options.PoolTargets)
            writer.Write(pt);

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
        _options.NumNeighbors = reader.ReadInt32();
        _options.UseBatchNorm = reader.ReadBoolean();
        _options.DropoutRate = reader.ReadDouble();
        _options.UseGlobalAveragePooling = reader.ReadBoolean();

        int convLen = reader.ReadInt32();
        _options.ConvChannels = new int[convLen];
        for (int i = 0; i < convLen; i++)
            _options.ConvChannels[i] = reader.ReadInt32();

        int poolLen = reader.ReadInt32();
        _options.PoolTargets = new int[poolLen];
        for (int i = 0; i < poolLen; i++)
            _options.PoolTargets[i] = reader.ReadInt32();

        int fcLen = reader.ReadInt32();
        _options.FullyConnectedSizes = new int[fcLen];
        for (int i = 0; i < fcLen; i++)
            _options.FullyConnectedSizes[i] = reader.ReadInt32();
    }

    /// <summary>
    /// Creates a new instance for cloning.
    /// </summary>
    /// <returns>New MeshCNN instance.</returns>
    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new MeshCNN<T>(_options, _optimizer, _lossFunction);
    }
}
