using AiDotNet.Attributes;
using AiDotNet.Enums;
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
/// <example>
/// <code>
/// var options = new MeshCNNOptions { InputEdgeFeatures = 5, HiddenSize = 64, NumLayers = 4 };
/// var model = new MeshCNN&lt;float&gt;(options);
/// var edgeFeatures = Tensor&lt;float&gt;.Random(new[] { 1, 500, 5 });
/// var output = model.Predict(edgeFeatures);
/// </code>
/// </example>
[ModelDomain(ModelDomain.ThreeD)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("MeshCNN: A Network with an Edge", "https://arxiv.org/abs/1809.05910", Year = 2019, Authors = "Rana Hanocka, Amir Hertz, Noa Fish, Raja Giryes, Shachar Fleishman, Daniel Cohen-Or")]
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

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

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
        Options = _options;
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

        return WalkLayersWithMeshReshape(input, activations: null);
    }

    /// <summary>
    /// Walks <see cref="NeuralNetworkBase{T}.Layers"/> applying the MeshCNN-specific
    /// rank promotion required by <see cref="GlobalPoolingLayer{T}"/>. MeshCNN data is
    /// shaped <c>[numEdges, channels]</c> (rank-2), but <see cref="GlobalPoolingLayer{T}"/>
    /// requires rank-3+. Rather than touching the pooling layer's contract — which
    /// would impact every other consumer — we reshape locally to
    /// <c>[1, numEdges, channels]</c> right before the pooling layer runs.
    /// </summary>
    /// <param name="input">Edge-feature tensor.</param>
    /// <param name="activations">Optional dictionary that receives a clone of every
    /// layer's output (keyed <c>Layer_{i}_{TypeName}</c>) — used by
    /// <see cref="GetNamedLayerActivations"/> so its walk goes through the same
    /// reshape path. Pass <c>null</c> to skip recording.</param>
    private Tensor<T> WalkLayersWithMeshReshape(Tensor<T> input, Dictionary<string, Tensor<T>>? activations)
    {
        if (_currentEdgeAdjacency == null)
        {
            throw new InvalidOperationException(
                "Edge adjacency must be set via SetEdgeAdjacency before calling Forward.");
        }

        // Guard the empty-mesh case up front: MeshCNN data is [numEdges, channels], and an
        // edge count of 0 produces a degenerate [1, 0, channels] tensor at the GlobalPooling
        // reshape that fails downstream with an opaque pooling error. Fail fast with a clear
        // message instead.
        if (input.Rank >= 1 && input.Shape[0] == 0)
            throw new ArgumentException("Cannot pool over empty edge dimension (0 edges).", nameof(input));

        PropagateAdjacencyToLayers();

        Tensor<T> output = input;
        var currentAdjacency = _currentEdgeAdjacency;

        for (int i = 0; i < Layers.Count; i++)
        {
            var layer = Layers[i];

            // GlobalPoolingLayer expects 3D+ input. MeshCNN data is [edges, channels] (2D).
            // Reshape to [1, edges, channels] so GlobalPoolingLayer processes it normally
            // and caches the input for backward pass gradient computation.
            if (layer is GlobalPoolingLayer<T> && output.Rank == 2)
            {
                output = output.Reshape([1, output.Shape[0], output.Shape[1]]);
            }

            output = layer.Forward(output);

            if (activations is not null)
                activations[$"Layer_{i}_{layer.GetType().Name}"] = output.Clone();

            if (layer is MeshPoolLayer<T> poolLayer && poolLayer.UpdatedAdjacency != null)
            {
                currentAdjacency = poolLayer.UpdatedAdjacency;
                UpdateSubsequentLayerAdjacency(layer, currentAdjacency);
            }
        }

        return output;
    }

    /// <inheritdoc />
    /// <remarks>
    /// Override to route through <see cref="WalkLayersWithMeshReshape"/> so the tape
    /// path sees the same rank-2 → rank-3 promotion before <see cref="GlobalPoolingLayer{T}"/>
    /// that the eager <see cref="Forward"/> applies. The base implementation walks
    /// <c>Layers</c> directly and would fail with "GlobalPoolingLayer requires rank-3+".
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        return WalkLayersWithMeshReshape(input, activations: null);
    }

    /// <inheritdoc />
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        var activations = new Dictionary<string, Tensor<T>>();
        WalkLayersWithMeshReshape(input, activations);
        return activations;
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
    /// Generates predictions for the given input.
    /// </summary>
    /// <param name="input">Edge features tensor.</param>
    /// <returns>Classification logits.</returns>
    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        bool wasTraining = IsTrainingMode;

        // Disable training mode on all layers (disables dropout)
        foreach (var layer in Layers)
            layer.SetTrainingMode(false);
        IsTrainingMode = false;

        try
        {
            return Forward(input);
        }
        finally
        {
            // Restore prior training state even if Forward throws
            foreach (var layer in Layers)
                layer.SetTrainingMode(wasTraining);
            IsTrainingMode = wasTraining;
        }
    }

    /// <summary>
    /// Trains the network on a single batch.
    /// </summary>
    /// <param name="input">Edge features tensor.</param>
    /// <param name="expectedOutput">Ground truth labels.</param>
    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expectedOutput, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
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
            int layerParams = checked((int)layer.ParameterCount);
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
            ModelData = SerializeForMetadata()
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

        // Per-mesh adjacency is NOT model state — write empty marker for backward compat
        writer.Write(0);
        writer.Write(0);
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

        // Skip adjacency data from older serialized models (per-mesh adjacency is NOT model state)
        // Callers must call SetEdgeAdjacency() for each new mesh sample
        _currentEdgeAdjacency = null;
        if (reader.BaseStream.Position < reader.BaseStream.Length)
        {
            int adjRows = reader.ReadInt32();
            int adjCols = reader.ReadInt32();
            if (adjRows > 0 && adjCols > 0)
            {
                // Skip the adjacency data but don't restore it
                for (int r = 0; r < adjRows; r++)
                    for (int c = 0; c < adjCols; c++)
                        reader.ReadInt32();
            }
        }
    }

    /// <inheritdoc />
    /// <remarks>
    /// MeshCNN's per-mesh edge adjacency is intentionally NOT serialized as model
    /// state (see <see cref="SerializeNetworkSpecificData"/>) because real workloads
    /// supply a fresh adjacency per mesh sample via <see cref="SetEdgeAdjacency"/>.
    /// However, <see cref="Clone"/> consumers expect to call <c>Predict</c> on the
    /// clone with the same input the original was using, so we propagate the live
    /// adjacency to the clone alongside the serialized weights. This preserves
    /// "clone reproduces original on the same input" without changing the on-disk
    /// model format.
    /// </remarks>
    public override IFullModel<T, Tensor<T>, Tensor<T>> Clone()
    {
        var clone = base.Clone();
        if (clone is MeshCNN<T> meshClone && _currentEdgeAdjacency is not null)
        {
            meshClone.SetEdgeAdjacency(_currentEdgeAdjacency);
        }
        return clone;
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
