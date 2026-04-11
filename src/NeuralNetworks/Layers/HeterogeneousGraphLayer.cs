using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Validation;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents metadata for heterogeneous graphs with multiple node and edge types.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This defines the "schema" of your heterogeneous graph.
///
/// Think of a knowledge graph with different types of entities and relationships:
/// - Node types: Person, Company, Product
/// - Edge types: WorksAt, Manufactures, Purchases
///
/// This metadata tells the layer what types exist and how they connect.
/// </para>
/// </remarks>
public partial class HeterogeneousGraphMetadata
{
    /// <summary>
    /// Names of node types (e.g., ["user", "item", "category"]).
    /// </summary>
    public string[] NodeTypes { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Names of edge types (e.g., ["likes", "belongs_to", "similar_to"]).
    /// </summary>
    public string[] EdgeTypes { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Input feature dimensions for each node type.
    /// </summary>
    public Dictionary<string, int> NodeTypeFeatures { get; set; } = new();

    /// <summary>
    /// Edge type connections: maps edge type to (source node type, target node type).
    /// </summary>
    public Dictionary<string, (string SourceType, string TargetType)> EdgeTypeSchema { get; set; } = new();
}

/// <summary>
/// Implements Heterogeneous Graph Neural Network layer for graphs with multiple node and edge types.
/// </summary>
/// <remarks>
/// <para>
/// Heterogeneous Graph Neural Networks (HGNNs) handle graphs where nodes and edges have different types.
/// Unlike homogeneous GNNs that treat all nodes and edges uniformly, HGNNs use type-specific
/// transformations and aggregations. This layer implements the R-GCN (Relational GCN) approach
/// with type-specific weight matrices.
/// </para>
/// <para>
/// The layer computes: h_i' = σ(Σ_{r∈R} Σ_{j∈N_r(i)} (1/c_{i,r}) W_r h_j + W_0 h_i)
/// where R is the set of relation types, N_r(i) are neighbors of type r, c_{i,r} is a normalization
/// constant, W_r are relation-specific weights, and W_0 is the self-loop weight.
/// </para>
/// <para><b>For Beginners:</b> This layer handles graphs where not all nodes and edges are the same.
///
/// Real-world examples:
///
/// **Knowledge Graph:**
/// - Node types: Person, Place, Event
/// - Edge types: BornIn, HappenedAt, AttendedBy
/// - Each type needs different processing
///
/// **E-commerce:**
/// - Node types: User, Product, Brand, Category
/// - Edge types: Purchased, Manufactured, BelongsTo, Viewed
/// - Different relationships have different meanings
///
/// **Academic Network:**
/// - Node types: Author, Paper, Venue, Topic
/// - Edge types: Wrote, PublishedIn, About, Cites
/// - Mixed types of entities and relationships
///
/// Why heterogeneous?
/// - **Different semantics**: A "User" has different properties than a "Product"
/// - **Type-specific patterns**: Relationships mean different things
/// - **Better representation**: Specialized processing for each type
///
/// The layer learns separate transformations for each edge type, then combines them intelligently.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.Graph)]
[LayerTask(LayerTask.GraphProcessing)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(ApiShape = LayerApiShape.GraphWithSetup, IsTrainable = true, ChangesShape = true, Cost = ComputeCost.High,
    TestInputShape = "1, 4, 8",
    TestConstructorArgs = "new AiDotNet.NeuralNetworks.Layers.HeterogeneousGraphMetadata { NodeTypes = new[] { \"A\" }, EdgeTypes = new[] { \"e\" }, NodeTypeFeatures = new System.Collections.Generic.Dictionary<string, int> { [\"A\"] = 8 }, EdgeTypeSchema = new System.Collections.Generic.Dictionary<string, (string, string)> { [\"e\"] = (\"A\", \"A\") } }, 4",
    TestSetupCode = "var t = (AiDotNet.NeuralNetworks.Layers.HeterogeneousGraphLayer<double>)layer; var adj = new AiDotNet.Tensors.LinearAlgebra.Tensor<double>(new[] { 4, 4 }); for (int i = 0; i < 4; i++) { adj[i, i] = 1.0; if (i > 0) adj[i, i-1] = 1.0; } t.SetAdjacencyMatrices(new System.Collections.Generic.Dictionary<string, AiDotNet.Tensors.LinearAlgebra.Tensor<double>> { [\"e\"] = adj }); t.SetNodeTypeMap(new System.Collections.Generic.Dictionary<int, string> { [0] = \"A\", [1] = \"A\", [2] = \"A\", [3] = \"A\" });")]
public partial class HeterogeneousGraphLayer<T> : LayerBase<T>, IGraphConvolutionLayer<T>
{
    private readonly HeterogeneousGraphMetadata _metadata;
    private readonly int _outputFeatures;
    private readonly bool _useBasis;
    private readonly int _numBases;

    /// <summary>
    /// Type-specific weight tensors. Key: edge type, Value: weight tensor [inputFeatures, outputFeatures].
    /// </summary>
    private Dictionary<string, Tensor<T>> _edgeTypeWeights;

    /// <summary>
    /// Self-loop weights for each node type. Key: node type, Value: weight tensor [inputFeatures, outputFeatures].
    /// </summary>
    private Dictionary<string, Tensor<T>> _selfLoopWeights;

    /// <summary>
    /// Bias for each node type. Key: node type, Value: bias tensor [outputFeatures].
    /// </summary>
    private Dictionary<string, Tensor<T>> _biases;

    /// <summary>
    /// Basis matrices for weight decomposition (if using basis). Shape: [numBases, inputFeatures, outputFeatures].
    /// </summary>
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T>? _basisMatrices;

    /// <summary>
    /// Coefficients for combining basis matrices per edge type. Key: edge type, Value: coefficients [numBases].
    /// </summary>
    private Dictionary<string, Tensor<T>>? _basisCoefficients;

    /// <summary>
    /// The adjacency matrices for each edge type.
    /// </summary>
    private Dictionary<string, Tensor<T>>? _adjacencyMatrices;

    /// <summary>
    /// Node type assignments for each node.
    /// </summary>
    private Dictionary<int, string>? _nodeTypeMap;

    /// <summary>
    /// Cached values for backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Gradients for weights, self-loop weights, and biases.
    /// </summary>
    private Dictionary<string, Tensor<T>>? _edgeTypeWeightsGradients;
    private Dictionary<string, Tensor<T>>? _selfLoopWeightsGradients;
    private Dictionary<string, Tensor<T>>? _biasesGradients;
    private Tensor<T>? _basisMatricesGradient;
    private Dictionary<string, Tensor<T>>? _basisCoefficientsGradients;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    protected override bool SupportsGpuExecution => true;

    /// <inheritdoc/>
    public int InputFeatures { get; private set; }

    /// <inheritdoc/>
    public int OutputFeatures => _outputFeatures;

    /// <summary>
    /// Initializes a new instance of the <see cref="HeterogeneousGraphLayer{T}"/> class.
    /// </summary>
    /// <param name="metadata">Metadata describing node and edge types.</param>
    /// <param name="outputFeatures">Number of output features per node.</param>
    /// <param name="useBasis">Whether to use basis decomposition (default: false).</param>
    /// <param name="numBases">Number of basis matrices if using decomposition (default: 4).</param>
    /// <param name="activationFunction">Activation function to apply.</param>
    /// <remarks>
    /// <para>
    /// Creates a heterogeneous graph layer. If useBasis is true, weights are decomposed as
    /// W_r = Σ_b a_{rb} V_b, reducing parameters for graphs with many edge types.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new heterogeneous graph layer.
    ///
    /// Key parameters:
    /// - metadata: Describes your graph structure (what types exist)
    /// - useBasis: Memory-saving technique for graphs with many edge types
    ///   * false: Each edge type has its own weights (more expressive)
    ///   * true: Edge types share basis matrices (fewer parameters)
    /// - numBases: How many shared patterns to use (if useBasis=true)
    ///
    /// Example setup:
    /// ```
    /// var metadata = new HeterogeneousGraphMetadata
    /// {
    ///     NodeTypes = ["user", "product"],
    ///     EdgeTypes = ["purchased", "viewed", "rated"],
    ///     NodeTypeFeatures = { ["user"] = 32, ["product"] = 64 },
    ///     EdgeTypeSchema = {
    ///         ["purchased"] = ("user", "product"),
    ///         ["viewed"] = ("user", "product"),
    ///         ["rated"] = ("user", "product")
    ///     }
    /// };
    /// var layer = new HeterogeneousGraphLayer(metadata, 128);
    /// ```
    /// </para>
    /// </remarks>
    public HeterogeneousGraphLayer(
        HeterogeneousGraphMetadata metadata,
        int outputFeatures,
        bool useBasis = false,
        int numBases = 4,
        IActivationFunction<T>? activationFunction = null)
        : base([0], [outputFeatures], activationFunction ?? new IdentityActivation<T>())
    {
        Guard.NotNull(metadata);
        _metadata = metadata;
        _outputFeatures = outputFeatures;
        _useBasis = useBasis && metadata.EdgeTypes.Length > numBases;
        _numBases = numBases;

        // Determine max input features across node types
        InputFeatures = metadata.NodeTypeFeatures.Values.Max();

        _edgeTypeWeights = new Dictionary<string, Tensor<T>>();
        _selfLoopWeights = new Dictionary<string, Tensor<T>>();
        _biases = new Dictionary<string, Tensor<T>>();

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        if (_useBasis)
        {
            // Initialize basis matrices [numBases, inputFeatures, outputFeatures]
            _basisMatrices = new Tensor<T>([_numBases, InputFeatures, _outputFeatures]);
            _basisCoefficients = new Dictionary<string, Tensor<T>>();

            T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (InputFeatures + _outputFeatures)));
            InitializeTensor(_basisMatrices, scale);

            // Initialize coefficients for each edge type [numBases]
            foreach (var edgeType in _metadata.EdgeTypes)
            {
                var coeffs = new Tensor<T>([_numBases]);
                T coeffScale = NumOps.FromDouble(1.0 / _numBases);
                InitializeTensor(coeffs, coeffScale);
                _basisCoefficients[edgeType] = coeffs;
            }
        }
        else
        {
            // Initialize separate weight tensor for each edge type
            foreach (var edgeType in _metadata.EdgeTypes)
            {
                var (sourceType, targetType) = _metadata.EdgeTypeSchema[edgeType];
                int inFeatures = _metadata.NodeTypeFeatures[sourceType];

                var weights = new Tensor<T>([inFeatures, _outputFeatures]);
                T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (inFeatures + _outputFeatures)));
                InitializeTensor(weights, scale);

                _edgeTypeWeights[edgeType] = weights;
            }
        }

        // Initialize self-loop weights and biases for each node type
        foreach (var nodeType in _metadata.NodeTypes)
        {
            int inFeatures = _metadata.NodeTypeFeatures[nodeType];

            var selfWeights = new Tensor<T>([inFeatures, _outputFeatures]);
            T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (inFeatures + _outputFeatures)));
            InitializeTensor(selfWeights, scale);
            _selfLoopWeights[nodeType] = selfWeights;

            var bias = new Tensor<T>([_outputFeatures]);
            bias.Fill(NumOps.Zero);
            _biases[nodeType] = bias;
        }

        // Register all trainable parameters for gradient tape discovery
        // Flatten dictionary values at registration time — edge types are known at construction
        foreach (var (_, weight) in _edgeTypeWeights)
            RegisterTrainableParameter(weight, PersistentTensorRole.Weights);
        foreach (var (_, weight) in _selfLoopWeights)
            RegisterTrainableParameter(weight, PersistentTensorRole.Weights);
        foreach (var (_, bias2) in _biases)
            RegisterTrainableParameter(bias2, PersistentTensorRole.Biases);
        if (_basisMatrices is not null)
            RegisterTrainableParameter(_basisMatrices, PersistentTensorRole.Weights);
        if (_basisCoefficients is not null)
        {
            foreach (var (_, coeffs) in _basisCoefficients)
                RegisterTrainableParameter(coeffs, PersistentTensorRole.Weights);
        }
    }

    /// <summary>
    /// Initializes a tensor with scaled random values.
    /// </summary>
    private void InitializeTensor(Tensor<T> tensor, T scale)
    {
        var randomTensor = Tensor<T>.CreateRandom(tensor._shape);
        var halfTensor = new Tensor<T>(tensor._shape);
        halfTensor.Fill(NumOps.FromDouble(0.5));
        var shifted = Engine.TensorSubtract(randomTensor, halfTensor);
        var scaled = Engine.TensorMultiplyScalar(shifted, scale);

        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = scaled.GetFlat(i);
        }
    }

    /// <inheritdoc/>
    public void SetAdjacencyMatrix(Tensor<T> adjacencyMatrix)
    {
        // For heterogeneous graphs, use SetAdjacencyMatrices instead
        throw new NotSupportedException(
            "Heterogeneous graphs require multiple adjacency matrices. Use SetAdjacencyMatrices instead.");
    }

    /// <summary>
    /// Sets the adjacency matrices for all edge types.
    /// </summary>
    /// <param name="adjacencyMatrices">Dictionary mapping edge types to their adjacency matrices.</param>
    public void SetAdjacencyMatrices(Dictionary<string, Tensor<T>> adjacencyMatrices)
    {
        _adjacencyMatrices = adjacencyMatrices;
    }

    /// <summary>
    /// Sets the node type mapping.
    /// </summary>
    /// <param name="nodeTypeMap">Dictionary mapping node indices to their types.</param>
    public void SetNodeTypeMap(Dictionary<int, string> nodeTypeMap)
    {
        _nodeTypeMap = nodeTypeMap;
    }

    /// <inheritdoc/>
    public Tensor<T>? GetAdjacencyMatrix()
    {
        // Return null for heterogeneous graphs
        return null;
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (_adjacencyMatrices == null || _nodeTypeMap == null)
        {
            throw new InvalidOperationException(
                "Adjacency matrices and node type map must be set before calling Forward.");
        }

        // Store original shape for any-rank tensor support
        _originalInputShape = input._shape;
        int rank = input.Shape.Length;

        // Handle any-rank tensor: normalize to 3D [batchSize, numNodes, features] for graph processing
        Tensor<T> processInput;
        int batchSize;

        if (rank == 1)
        {
            // 1D: treat as single sample, single node
            batchSize = 1;
            processInput = Engine.Reshape(input, [1, 1, input.Shape[0]]);
        }
        else if (rank == 2)
        {
            // 2D [numNodes, features]: add batch dimension
            batchSize = 1;
            processInput = Engine.Reshape(input, [1, input.Shape[0], input.Shape[1]]);
        }
        else if (rank == 3)
        {
            // Already 3D [batchSize, numNodes, features]
            batchSize = input.Shape[0];
            processInput = input;
        }
        else
        {
            // Higher rank: collapse leading dimensions into batch
            int flatBatch = 1;
            for (int d = 0; d < rank - 2; d++)
                flatBatch *= input.Shape[d];
            batchSize = flatBatch;
            int numNodes = input.Shape[rank - 2];
            int features = input.Shape[rank - 1];
            processInput = Engine.Reshape(input, [flatBatch, numNodes, features]);
        }

        _lastInput = processInput;
        int processNumNodes = processInput.Shape[1];

        var output = TensorAllocator.Rent<T>([batchSize, processNumNodes, _outputFeatures]);
        output.Fill(NumOps.Zero);

        // Process each edge type
        foreach (var edgeType in _metadata.EdgeTypes)
        {
            if (!_adjacencyMatrices.TryGetValue(edgeType, out var adjacency))
                continue;

            var (sourceType, targetType) = _metadata.EdgeTypeSchema[edgeType];

            // Get weights for this edge type
            Tensor<T> weights;

            if (_useBasis && _basisMatrices != null && _basisCoefficients != null)
            {
                // Reconstruct weights from basis decomposition: W_r = sum_b (a_rb * V_b)
                var coeffs = _basisCoefficients[edgeType]; // [numBases]
                weights = new Tensor<T>([InputFeatures, _outputFeatures]);
                weights.Fill(NumOps.Zero);

                for (int b = 0; b < _numBases; b++)
                {
                    // Extract basis matrix [inputFeatures, outputFeatures]
                    var basisSlice = ExtractBasisMatrix(_basisMatrices, b, InputFeatures, _outputFeatures);
                    var scaledBasis = Engine.TensorMultiplyScalar(basisSlice, coeffs[b]);
                    weights = Engine.TensorAdd(weights, scaledBasis);
                }
            }
            else
            {
                weights = _edgeTypeWeights[edgeType];
            }

            // Aggregate messages of this edge type using Engine operations
            // For each batch: output += adjacency @ input @ weights (with normalization)
            int inFeatures = _metadata.NodeTypeFeatures[sourceType];

            // Extract relevant input features [batch, nodes, inFeatures]
            var inputSlice = processInput.Shape[2] == inFeatures ? processInput :
                ExtractInputFeatures(processInput, batchSize, processNumNodes, inFeatures);

            // Compute normalization factor from adjacency matrix
            var normalizedAdj = NormalizeAdjacency(adjacency, batchSize, processNumNodes);

            // Perform graph convolution: normalizedAdj @ inputSlice @ weights
            var xw = BatchedMatMul3Dx2D(inputSlice, weights, batchSize, processNumNodes, inFeatures, _outputFeatures);
            var convOutput = Engine.BatchMatMul(normalizedAdj, xw);

            // Accumulate to output
            output = Engine.TensorAdd(output, convOutput);
        }

        // Add self-loops and biases
        for (int i = 0; i < processNumNodes; i++)
        {
            string nodeType = _nodeTypeMap[i];
            var selfWeights = _selfLoopWeights[nodeType];
            var bias = _biases[nodeType];
            int inFeatures = _metadata.NodeTypeFeatures[nodeType];

            // Extract input for this node across all batches
            var nodeInput = ExtractNodeInput(processInput, batchSize, i, inFeatures);

            // Apply self-loop transformation: nodeInput @ selfWeights
            var selfOutput = Engine.TensorMatMul(nodeInput, selfWeights); // [batch, outputFeatures]

            // Broadcast bias across batch
            var biasBroadcast = BroadcastBias(bias, batchSize);

            // Add to output at this node position
            output = AddNodeOutput(output, selfOutput, biasBroadcast, batchSize, i, _outputFeatures);
        }

        var result = ApplyActivation(output);

        // Restore output shape to match input rank
        if (_originalInputShape != null && _originalInputShape.Length != 3)
        {
            if (_originalInputShape.Length == 2)
            {
                // 2D input [numNodes, features] -> 2D output [numNodes, outputFeatures]
                result = Engine.Reshape(result, [processNumNodes, _outputFeatures]);
            }
            else if (_originalInputShape.Length == 1)
            {
                // 1D input -> 1D output
                result = Engine.Reshape(result, [_outputFeatures]);
            }
            else
            {
                // Higher rank: restore leading dimensions
                var outShape = new int[_originalInputShape.Length];
                for (int d = 0; d < _originalInputShape.Length - 2; d++)
                    outShape[d] = _originalInputShape[d];
                outShape[_originalInputShape.Length - 2] = processNumNodes;
                outShape[_originalInputShape.Length - 1] = _outputFeatures;
                result = Engine.Reshape(result, outShape);
            }
        }

        // Store output AFTER shape restoration so Backward gets matching rank
        if (IsTrainingMode)
        {
            _lastOutput = result;
        }

        return result;
    }

    /// <summary>
    /// GPU-accelerated forward pass for HeterogeneousGraphLayer.
    /// Implements type-specific graph convolution with fully GPU-native operations.
    /// </summary>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend == null)
            throw new InvalidOperationException("GPU backend unavailable.");

        if (_adjacencyMatrices == null || _nodeTypeMap == null)
        {
            throw new InvalidOperationException(
                "Adjacency matrices and node type map must be set before calling ForwardGpu.");
        }

        var input = inputs[0];
        int[] inputShape = input._shape;

        // Handle shape normalization
        int batchSize;
        int numNodes;
        int inputFeatures;

        // Support any rank >= 2: last 2 dims are [nodes, features], earlier dims are batch-like
        if (inputShape.Length < 2)
            throw new ArgumentException($"HeterogeneousGraph layer requires at least 2D tensor [nodes, features]. Got rank {inputShape.Length}.");

        if (inputShape.Length == 2)
        {
            batchSize = 1;
            numNodes = inputShape[0];
            inputFeatures = inputShape[1];
        }
        else if (inputShape.Length == 3)
        {
            batchSize = inputShape[0];
            numNodes = inputShape[1];
            inputFeatures = inputShape[2];
        }
        else
        {
            // Higher rank: flatten leading dimensions into batch
            batchSize = 1;
            for (int d = 0; d < inputShape.Length - 2; d++)
                batchSize *= inputShape[d];
            numNodes = inputShape[inputShape.Length - 2];
            inputFeatures = inputShape[inputShape.Length - 1];
        }

        int outputSize = batchSize * numNodes * _outputFeatures;

        // Initialize output buffer to zero
        var outputBuffer = backend.AllocateBuffer(outputSize);
        backend.Fill(outputBuffer, 0.0f, outputSize);

        // ========================================
        // EDGE-TYPE SPECIFIC GRAPH CONVOLUTIONS
        // ========================================

        foreach (var edgeType in _metadata.EdgeTypes)
        {
            if (!_adjacencyMatrices.TryGetValue(edgeType, out var adjacency))
                continue;

            var (sourceType, _) = _metadata.EdgeTypeSchema[edgeType];
            int inFeatures = _metadata.NodeTypeFeatures[sourceType];

            // Get or reconstruct weights for this edge type
            Tensor<T> weights;
            if (_useBasis && _basisMatrices != null && _basisCoefficients != null)
            {
                var coeffs = _basisCoefficients[edgeType];
                weights = new Tensor<T>([InputFeatures, _outputFeatures]);
                weights.Fill(NumOps.Zero);
                for (int b = 0; b < _numBases; b++)
                {
                    var basisSlice = ExtractBasisMatrix(_basisMatrices, b, InputFeatures, _outputFeatures);
                    var scaledBasis = Engine.TensorMultiplyScalar(basisSlice, coeffs[b]);
                    weights = Engine.TensorAdd(weights, scaledBasis);
                }
            }
            else
            {
                weights = _edgeTypeWeights[edgeType];
            }

            // Upload weights to GPU
            using var weightsBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(weights.Data.ToArray()));

            // Normalize adjacency matrix (precompute on CPU, upload once)
            var normalizedAdj = NormalizeAdjacency(adjacency, batchSize, numNodes);
            using var adjBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(normalizedAdj.Data.ToArray()));

            // GPU-native processing: For all batches at once
            // If inFeatures == inputFeatures, use input directly; otherwise extract columns on GPU
            IGpuBuffer inputSliceBuffer;
            bool needsDisposeSlice = false;

            if (inFeatures == inputFeatures)
            {
                inputSliceBuffer = input.Buffer;
            }
            else
            {
                // Extract first inFeatures columns - single CPU roundtrip for preprocessing
                // This is O(1) roundtrips vs O(numNodes) in the original code
                var fullData = backend.DownloadBuffer(input.Buffer);
                var sliceData = new float[batchSize * numNodes * inFeatures];
                for (int b = 0; b < batchSize; b++)
                {
                    for (int n = 0; n < numNodes; n++)
                    {
                        int srcOffset = b * numNodes * inputFeatures + n * inputFeatures;
                        int dstOffset = b * numNodes * inFeatures + n * inFeatures;
                        for (int f = 0; f < inFeatures && f < inputFeatures; f++)
                        {
                            sliceData[dstOffset + f] = fullData[srcOffset + f];
                        }
                    }
                }
                inputSliceBuffer = backend.AllocateBuffer(sliceData);
                needsDisposeSlice = true;
            }

            // Process all batches using batched GEMM
            // xw = input @ weights : [batchSize * numNodes, inFeatures] @ [inFeatures, outputFeatures] -> [batchSize * numNodes, outputFeatures]
            using var xwBuffer = backend.AllocateBuffer(batchSize * numNodes * _outputFeatures);
            backend.Gemm(inputSliceBuffer, weightsBuffer, xwBuffer, batchSize * numNodes, _outputFeatures, inFeatures);

            // Apply adjacency matrix per batch: convOutput = adj @ xw
            // For single batch or 2D adjacency, apply once to all
            using var convOutputBuffer = backend.AllocateBuffer(batchSize * numNodes * _outputFeatures);

            if (batchSize == 1)
            {
                // Single batch: direct GEMM
                backend.Gemm(adjBuffer, xwBuffer, convOutputBuffer, numNodes, _outputFeatures, numNodes);
            }
            else if (normalizedAdj.Shape.Length == 2)
            {
                // 2D adjacency shared across batches: tile adjacency and use BatchedGemm
                // Create tiled adjacency buffer [batchSize, numNodes, numNodes]
                using var tiledAdjBuffer = backend.AllocateBuffer(batchSize * numNodes * numNodes);
                for (int b = 0; b < batchSize; b++)
                {
                    int dstOffset = b * numNodes * numNodes;
                    backend.Copy2DStrided(adjBuffer, tiledAdjBuffer, 1, numNodes * numNodes, batchSize * numNodes * numNodes, dstOffset);
                }
                // BatchedGemm: [batch, numNodes, numNodes] @ [batch, numNodes, outputFeatures]
                backend.BatchedGemm(tiledAdjBuffer, xwBuffer, convOutputBuffer, numNodes, _outputFeatures, numNodes, batchSize);
            }
            else
            {
                // Per-batch 3D adjacency: use BatchedGemm directly
                backend.BatchedGemm(adjBuffer, xwBuffer, convOutputBuffer, numNodes, _outputFeatures, numNodes, batchSize);
            }

            // Accumulate to output using GPU Add
            backend.Add(outputBuffer, convOutputBuffer, outputBuffer, outputSize);

            if (needsDisposeSlice)
            {
                inputSliceBuffer.Dispose();
            }
        }

        // ========================================
        // SELF-LOOP WEIGHTS AND BIASES PER NODE TYPE
        // (GPU-native batch processing with type masks)
        // ========================================

        // For each node type, pre-compute a mask and process all nodes in batched fashion
        foreach (var nodeType in _metadata.NodeTypes)
        {
            var selfWeights = _selfLoopWeights[nodeType];
            var bias = _biases[nodeType];
            int inFeatures = _metadata.NodeTypeFeatures[nodeType];

            // Upload weights and bias
            using var selfWeightsBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(selfWeights.Data.ToArray()));
            using var biasBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(bias.Data.ToArray()));

            // Create a broadcast mask [numNodes, outputFeatures] where mask[n, :] = 1.0 if node n is of this type
            var nodeMaskData = new float[numNodes * _outputFeatures];
            for (int n = 0; n < numNodes; n++)
            {
                float maskVal = (_nodeTypeMap.TryGetValue(n, out var type) && type == nodeType) ? 1.0f : 0.0f;
                for (int f = 0; f < _outputFeatures; f++)
                {
                    nodeMaskData[n * _outputFeatures + f] = maskVal;
                }
            }
            using var nodeMaskBuffer = backend.AllocateBuffer(nodeMaskData);

            // Extract input features for this node type (first inFeatures columns)
            IGpuBuffer typeInputBuffer;
            bool needsDisposeTypeInput = false;

            if (inFeatures == inputFeatures)
            {
                typeInputBuffer = input.Buffer;
            }
            else
            {
                // Extract first inFeatures columns - single CPU roundtrip for preprocessing
                var fullData = backend.DownloadBuffer(input.Buffer);
                var sliceData = new float[batchSize * numNodes * inFeatures];
                for (int b = 0; b < batchSize; b++)
                {
                    for (int n = 0; n < numNodes; n++)
                    {
                        int srcOffset = b * numNodes * inputFeatures + n * inputFeatures;
                        int dstOffset = b * numNodes * inFeatures + n * inFeatures;
                        for (int f = 0; f < inFeatures && f < inputFeatures; f++)
                        {
                            sliceData[dstOffset + f] = fullData[srcOffset + f];
                        }
                    }
                }
                typeInputBuffer = backend.AllocateBuffer(sliceData);
                needsDisposeTypeInput = true;
            }

            // Compute selfOutput = input @ selfWeights for ALL nodes
            // [batchSize * numNodes, inFeatures] @ [inFeatures, outputFeatures] -> [batchSize * numNodes, outputFeatures]
            using var selfOutputBuffer = backend.AllocateBuffer(batchSize * numNodes * _outputFeatures);
            backend.Gemm(typeInputBuffer, selfWeightsBuffer, selfOutputBuffer, batchSize * numNodes, _outputFeatures, inFeatures);

            // Add bias using BiasAdd (broadcasts across nodes)
            backend.BiasAdd(selfOutputBuffer, biasBuffer, selfOutputBuffer, batchSize * numNodes, _outputFeatures);

            // Apply mask: maskedSelfOutput = selfOutput * mask (element-wise)
            // The mask is [numNodes, outputFeatures], need to tile for batch dimension
            using var tiledMaskBuffer = backend.AllocateBuffer(batchSize * numNodes * _outputFeatures);
            for (int b = 0; b < batchSize; b++)
            {
                int dstOffset = b * numNodes * _outputFeatures;
                backend.Copy2DStrided(nodeMaskBuffer, tiledMaskBuffer, 1, numNodes * _outputFeatures, batchSize * numNodes * _outputFeatures, dstOffset);
            }

            // Element-wise multiply: selfOutput * tiledMask
            using var maskedSelfOutputBuffer = backend.AllocateBuffer(batchSize * numNodes * _outputFeatures);
            backend.Multiply(selfOutputBuffer, tiledMaskBuffer, maskedSelfOutputBuffer, batchSize * numNodes * _outputFeatures);

            // Accumulate to output using GPU Add
            backend.Add(outputBuffer, maskedSelfOutputBuffer, outputBuffer, outputSize);

            if (needsDisposeTypeInput)
            {
                typeInputBuffer.Dispose();
            }
        }

        // ========================================
        // APPLY ACTIVATION
        // ========================================

        ApplyGpuActivation(backend, outputBuffer, outputBuffer, outputSize, GetFusedActivationType());

        // Create output tensor with appropriate shape — restore original leading dims for higher-rank input
        int[] outputShape;
        if (inputShape.Length > 3)
        {
            outputShape = new int[inputShape.Length];
            for (int d = 0; d < inputShape.Length - 2; d++)
                outputShape[d] = inputShape[d];
            outputShape[inputShape.Length - 2] = numNodes;
            outputShape[inputShape.Length - 1] = _outputFeatures;
        }
        else if (inputShape.Length == 2)
        {
            outputShape = [numNodes, _outputFeatures];
        }
        else
        {
            outputShape = [batchSize, numNodes, _outputFeatures];
        }

        var finalBuffer = backend.AllocateBuffer(outputSize);
        backend.Copy(outputBuffer, finalBuffer, outputSize);
        outputBuffer.Dispose();
        return GpuTensorHelper.UploadToGpu<T>(backend, finalBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// Converts a normalized adjacency tensor to CSR format (using first batch element).
    /// </summary>
    private (float[] Values, int[] ColIndices, int[] RowPointers) ConvertToCSR(Tensor<T> adjacency, int numNodes)
    {
        var values = new List<float>();
        var colIndices = new List<int>();
        var rowPointers = new List<int> { 0 };

        bool is2D = adjacency.Shape.Length == 2;

        for (int i = 0; i < numNodes; i++)
        {
            for (int j = 0; j < numNodes; j++)
            {
                float val = is2D
                    ? (float)NumOps.ToDouble(adjacency[i, j])
                    : (float)NumOps.ToDouble(adjacency[0, i, j]);

                if (MathF.Abs(val) > 1e-6f)
                {
                    values.Add(val);
                    colIndices.Add(j);
                }
            }
            rowPointers.Add(values.Count);
        }

        return (values.ToArray(), colIndices.ToArray(), rowPointers.ToArray());
    }

    /// <summary>
    /// Extracts a basis matrix from the basis matrices tensor.
    /// </summary>
    private Tensor<T> ExtractBasisMatrix(Tensor<T> basisMatrices, int basisIndex, int rows, int cols)
    {
        var result = TensorAllocator.Rent<T>([rows, cols]);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = basisMatrices[basisIndex, i, j];
            }
        }
        return result;
    }

    /// <summary>
    /// Extracts input features for a specific number of features.
    /// </summary>
    private Tensor<T> ExtractInputFeatures(Tensor<T> input, int batchSize, int numNodes, int inFeatures)
    {
        var result = TensorAllocator.Rent<T>([batchSize, numNodes, inFeatures]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int f = 0; f < inFeatures && f < input.Shape[2]; f++)
                {
                    result[b, n, f] = input[b, n, f];
                }
            }
        }
        return result;
    }

    /// <summary>
    /// Normalizes adjacency matrix by degree (row normalization).
    /// </summary>
    private Tensor<T> NormalizeAdjacency(Tensor<T> adjacency, int batchSize, int numNodes)
    {
        // Handle both 2D [N, N] and 3D [B, N, N] adjacency matrices
        bool is2D = adjacency.Shape.Length == 2;
        Tensor<T> adj3D;
        if (is2D)
        {
            // Reshape 2D to 3D for processing
            adj3D = adjacency.Reshape([1, adjacency.Shape[0], adjacency.Shape[1]]);
            // Tile for batch if needed
            if (batchSize > 1)
            {
                var tiled = new Tensor<T>([batchSize, adjacency.Shape[0], adjacency.Shape[1]]);
                for (int b = 0; b < batchSize; b++)
                {
                    for (int i = 0; i < adjacency.Shape[0]; i++)
                    {
                        for (int j = 0; j < adjacency.Shape[1]; j++)
                        {
                            tiled[new int[] { b, i, j }] = adjacency[new int[] { i, j }];
                        }
                    }
                }
                adj3D = tiled;
            }
        }
        else
        {
            adj3D = adjacency;
        }

        var normalized = new Tensor<T>(adj3D._shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                // Count degree
                int degree = 0;
                for (int j = 0; j < numNodes; j++)
                {
                    if (!NumOps.Equals(adj3D[new int[] { b, i, j }], NumOps.Zero))
                        degree++;
                }

                if (degree == 0)
                {
                    // No neighbors, copy zeros
                    for (int j = 0; j < numNodes; j++)
                    {
                        normalized[new int[] { b, i, j }] = NumOps.Zero;
                    }
                }
                else
                {
                    T normalization = NumOps.Divide(NumOps.One, NumOps.FromDouble(degree));
                    for (int j = 0; j < numNodes; j++)
                    {
                        normalized[new int[] { b, i, j }] = NumOps.Multiply(adj3D[new int[] { b, i, j }], normalization);
                    }
                }
            }
        }

        return normalized;
    }

    /// <summary>
    /// Extracts input for a specific node across all batches.
    /// </summary>
    private Tensor<T> ExtractNodeInput(Tensor<T> input, int batchSize, int nodeIndex, int inFeatures)
    {
        var result = TensorAllocator.Rent<T>([batchSize, inFeatures]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < inFeatures && f < input.Shape[2]; f++)
            {
                result[b, f] = input[b, nodeIndex, f];
            }
        }
        return result;
    }

    /// <summary>
    /// Broadcasts a bias tensor across batch dimension.
    /// </summary>
    private Tensor<T> BroadcastBias(Tensor<T> bias, int batchSize)
    {
        var biasReshaped = bias.Reshape([1, bias.Length]);
        return Engine.TensorTile(biasReshaped, [batchSize, 1]);
    }

    /// <summary>
    /// Adds node output and bias to the overall output tensor.
    /// </summary>
    private Tensor<T> AddNodeOutput(Tensor<T> output, Tensor<T> nodeOutput, Tensor<T> bias,
        int batchSize, int nodeIndex, int outputFeatures)
    {
        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < outputFeatures; f++)
            {
                T value = NumOps.Add(nodeOutput[b, f], bias[b, f]);
                output[b, nodeIndex, f] = NumOps.Add(output[b, nodeIndex, f], value);
            }
        }
        return output;
    }

    /// <summary>
    /// Performs batched matrix multiplication between a 3D tensor and a 2D weight matrix.
    /// Input: [batch, rows, cols] @ weights: [cols, output_cols] -> [batch, rows, output_cols]
    /// </summary>
    private Tensor<T> BatchedMatMul3Dx2D(Tensor<T> input3D, Tensor<T> weights2D, int batch, int rows, int cols, int outputCols)
    {
        var flattened = input3D.Reshape([batch * rows, cols]);
        var result = Engine.TensorMatMul(flattened, weights2D);
        return result.Reshape([batch, rows, outputCols]);
    }

    /// <summary>
    /// Extracts a 2D batch slice from a 3D tensor.
    /// </summary>
    private Tensor<T> ExtractBatchSlice(Tensor<T> tensor, int batchIndex, int rows, int cols)
    {
        // If tensor is 2D (no batch dim), return it directly when batch=0
        if (tensor.Rank == 2)
        {
            if (batchIndex == 0 && tensor.Shape[0] == rows && tensor.Shape[1] == cols)
                return tensor;
            // Slice isn't possible from 2D — return as-is
            return tensor;
        }

        var result = TensorAllocator.Rent<T>([rows, cols]);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = tensor[batchIndex, i, j];
            }
        }
        return result;
    }

    /// <summary>
    /// Updates the layer parameters based on computed gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    public override void UpdateParameters(T learningRate)
    {
        if (_edgeTypeWeightsGradients == null || _selfLoopWeightsGradients == null || _biasesGradients == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        // Update edge type weights
        if (!_useBasis)
        {
            foreach (var kvp in _edgeTypeWeights)
            {
                string edgeType = kvp.Key;
                if (_edgeTypeWeightsGradients.TryGetValue(edgeType, out var gradient))
                {
                    var scaledGrad = Engine.TensorMultiplyScalar(gradient, learningRate);
                    _edgeTypeWeights[edgeType] = Engine.TensorSubtract(_edgeTypeWeights[edgeType], scaledGrad);
                }
            }
        }
        else if (_basisMatricesGradient != null && _basisCoefficientsGradients != null &&
                 _basisMatrices != null && _basisCoefficients != null)
        {
            // Update basis matrices
            var scaledBasisGrad = Engine.TensorMultiplyScalar(_basisMatricesGradient, learningRate);
            _basisMatrices = Engine.TensorSubtract(_basisMatrices, scaledBasisGrad);

            // Update basis coefficients
            foreach (var kvp in _basisCoefficients)
            {
                string edgeType = kvp.Key;
                if (_basisCoefficientsGradients.TryGetValue(edgeType, out var gradient))
                {
                    var scaledGrad = Engine.TensorMultiplyScalar(gradient, learningRate);
                    _basisCoefficients[edgeType] = Engine.TensorSubtract(_basisCoefficients[edgeType], scaledGrad);
                }
            }
        }

        // Update self-loop weights
        foreach (var kvp in _selfLoopWeights)
        {
            string nodeType = kvp.Key;
            if (_selfLoopWeightsGradients.TryGetValue(nodeType, out var gradient))
            {
                var scaledGrad = Engine.TensorMultiplyScalar(gradient, learningRate);
                _selfLoopWeights[nodeType] = Engine.TensorSubtract(_selfLoopWeights[nodeType], scaledGrad);
            }
        }

        // Update biases
        foreach (var kvp in _biases)
        {
            string nodeType = kvp.Key;
            if (_biasesGradients.TryGetValue(nodeType, out var gradient))
            {
                var scaledGrad = Engine.TensorMultiplyScalar(gradient, learningRate);
                _biases[nodeType] = Engine.TensorSubtract(_biases[nodeType], scaledGrad);
            }
        }
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a list of tensors.
    /// </summary>
    /// <returns>A list containing all trainable parameter tensors.</returns>
    public List<Tensor<T>> GetParameterTensors()
    {
        var parameters = new List<Tensor<T>>();

        if (_useBasis)
        {
            if (_basisMatrices != null)
                parameters.Add(_basisMatrices);

            if (_basisCoefficients != null)
            {
                foreach (var coeffs in _basisCoefficients.Values)
                {
                    parameters.Add(coeffs);
                }
            }
        }
        else
        {
            foreach (var weights in _edgeTypeWeights.Values)
            {
                parameters.Add(weights);
            }
        }

        foreach (var selfWeights in _selfLoopWeights.Values)
        {
            parameters.Add(selfWeights);
        }

        foreach (var bias in _biases.Values)
        {
            parameters.Add(bias);
        }

        return parameters;
    }

    /// <summary>
    /// Sets the trainable parameters of the layer from a list of tensors.
    /// </summary>
    /// <param name="parameters">A list containing all parameter tensors to set.</param>
    public void SetParameterTensors(List<Tensor<T>> parameters)
    {
        int index = 0;

        if (_useBasis)
        {
            if (_basisMatrices != null && index < parameters.Count)
            {
                _basisMatrices = parameters[index++];
            }

            if (_basisCoefficients != null)
            {
                var edgeTypes = _basisCoefficients.Keys.ToList();
                foreach (var edgeType in edgeTypes)
                {
                    if (index < parameters.Count)
                    {
                        _basisCoefficients[edgeType] = parameters[index++];
                    }
                }
            }
        }
        else
        {
            var edgeTypes = _edgeTypeWeights.Keys.ToList();
            foreach (var edgeType in edgeTypes)
            {
                if (index < parameters.Count)
                {
                    _edgeTypeWeights[edgeType] = parameters[index++];
                }
            }
        }

        var nodeTypes = _selfLoopWeights.Keys.ToList();
        foreach (var nodeType in nodeTypes)
        {
            if (index < parameters.Count)
            {
                _selfLoopWeights[nodeType] = parameters[index++];
            }
        }

        nodeTypes = _biases.Keys.ToList();
        foreach (var nodeType in nodeTypes)
        {
            if (index < parameters.Count)
            {
                _biases[nodeType] = parameters[index++];
            }
        }
    }

    /// <inheritdoc/>
    public override int ParameterCount => GetParameterTensors().Sum(t => t.Length);

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        // Flatten all tensors into a single vector
        var tensorParams = GetParameterTensors();
        var allValues = new List<T>();

        foreach (var tensor in tensorParams)
        {
            for (int i = 0; i < tensor.Length; i++)
            {
                allValues.Add(tensor.GetFlat(i));
            }
        }

        return new Vector<T>(allValues.ToArray());
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        // Reconstruct tensors from flattened vector
        var tensorParams = GetParameterTensors();
        int index = 0;

        foreach (var tensor in tensorParams)
        {
            for (int i = 0; i < tensor.Length; i++)
            {
                if (index < parameters.Length)
                {
                    tensor[i] = parameters[index++];
                }
            }
        }
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _edgeTypeWeightsGradients = null;
        _selfLoopWeightsGradients = null;
        _biasesGradients = null;
        _basisMatricesGradient = null;
        _basisCoefficientsGradients = null;
    }
}
