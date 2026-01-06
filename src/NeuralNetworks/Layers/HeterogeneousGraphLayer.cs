using AiDotNet.ActivationFunctions;
using AiDotNet.Autodiff;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

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
public class HeterogeneousGraphMetadata
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
public class HeterogeneousGraphLayer<T> : LayerBase<T>, IGraphConvolutionLayer<T>
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
        _metadata = metadata ?? throw new ArgumentNullException(nameof(metadata));
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
    }

    /// <summary>
    /// Initializes a tensor with scaled random values.
    /// </summary>
    private void InitializeTensor(Tensor<T> tensor, T scale)
    {
        var randomTensor = Tensor<T>.CreateRandom(tensor.Shape);
        var halfTensor = new Tensor<T>(tensor.Shape);
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
        _originalInputShape = input.Shape;
        int rank = input.Shape.Length;

        // Handle any-rank tensor: normalize to 3D [batchSize, numNodes, features] for graph processing
        Tensor<T> processInput;
        int batchSize;

        if (rank == 1)
        {
            // 1D: treat as single sample, single node
            batchSize = 1;
            processInput = input.Reshape([1, 1, input.Shape[0]]);
        }
        else if (rank == 2)
        {
            // 2D [numNodes, features]: add batch dimension
            batchSize = 1;
            processInput = input.Reshape([1, input.Shape[0], input.Shape[1]]);
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
            processInput = input.Reshape([flatBatch, numNodes, features]);
        }

        _lastInput = processInput;
        int processNumNodes = processInput.Shape[1];

        var output = new Tensor<T>([batchSize, processNumNodes, _outputFeatures]);
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

        // Only store for backward pass during training - skip during inference
        if (IsTrainingMode)
        {
            _lastOutput = result;
        }

        // Restore output shape to match input rank
        if (_originalInputShape != null && _originalInputShape.Length != 3)
        {
            if (_originalInputShape.Length == 2)
            {
                // 2D input [numNodes, features] -> 2D output [numNodes, outputFeatures]
                return result.Reshape([processNumNodes, _outputFeatures]);
            }
            else if (_originalInputShape.Length == 1)
            {
                // 1D input -> 1D output
                return result.Reshape([_outputFeatures]);
            }
            else
            {
                // Higher rank: restore leading dimensions
                var outShape = new int[_originalInputShape.Length];
                for (int d = 0; d < _originalInputShape.Length - 2; d++)
                    outShape[d] = _originalInputShape[d];
                outShape[_originalInputShape.Length - 2] = processNumNodes;
                outShape[_originalInputShape.Length - 1] = _outputFeatures;
                return result.Reshape(outShape);
            }
        }

        return result;
    }

    /// <summary>
    /// GPU-accelerated forward pass for HeterogeneousGraphLayer.
    /// Implements type-specific graph convolution with GPU operations where possible.
    /// </summary>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
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
        int[] inputShape = input.Shape;

        // Handle shape normalization
        int batchSize;
        int numNodes;
        int inputFeatures;

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
            throw new ArgumentException($"Input must be 2D or 3D, got {inputShape.Length}D");
        }

        int outputSize = batchSize * numNodes * _outputFeatures;

        // Initialize output buffer to zero
        var outputData = new float[outputSize];
        using var outputBuffer = backend.AllocateBuffer(outputData);

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
            using var weightsBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(weights.Data));

            // Normalize adjacency matrix (precompute on CPU, upload once)
            var normalizedAdj = NormalizeAdjacency(adjacency, batchSize, numNodes);
            using var adjBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(normalizedAdj.Data));

            // Process each batch
            for (int b = 0; b < batchSize; b++)
            {
                int batchInputOffset = b * numNodes * inputFeatures;
                int batchOutputOffset = b * numNodes * _outputFeatures;

                // Get view of input for this batch
                var batchInputView = input.CreateView(batchInputOffset, [numNodes, inputFeatures]);

                // If inFeatures != inputFeatures, we need to extract a slice
                IGpuBuffer inputSliceBuffer;
                bool needsDispose = false;

                if (inFeatures == inputFeatures)
                {
                    inputSliceBuffer = batchInputView.Buffer;
                }
                else
                {
                    // Extract only the first inFeatures columns - requires CPU roundtrip
                    var fullData = backend.DownloadBuffer(batchInputView.Buffer);
                    var sliceData = new float[numNodes * inFeatures];
                    for (int n = 0; n < numNodes; n++)
                    {
                        for (int f = 0; f < inFeatures && f < inputFeatures; f++)
                        {
                            sliceData[n * inFeatures + f] = fullData[n * inputFeatures + f];
                        }
                    }
                    inputSliceBuffer = backend.AllocateBuffer(sliceData);
                    needsDispose = true;
                }

                // Step 1: xw = inputSlice @ weights : [numNodes, inFeatures] @ [inFeatures, outputFeatures] -> [numNodes, outputFeatures]
                using var xwBuffer = backend.AllocateBuffer(numNodes * _outputFeatures);
                backend.Gemm(inputSliceBuffer, weightsBuffer, xwBuffer, numNodes, _outputFeatures, inFeatures);

                // Step 2: convOutput = adj @ xw : [numNodes, numNodes] @ [numNodes, outputFeatures] -> [numNodes, outputFeatures]
                // Get adjacency slice for this batch
                int adjBatchOffset = (normalizedAdj.Shape.Length == 3) ? b * numNodes * numNodes : 0;
                IGpuBuffer adjSliceBuffer;
                if (normalizedAdj.Shape.Length == 2 || batchSize == 1)
                {
                    adjSliceBuffer = adjBuffer;
                }
                else
                {
                    // Create view for this batch's adjacency
                    var adjData = backend.DownloadBuffer(adjBuffer);
                    var adjSliceData = new float[numNodes * numNodes];
                    Array.Copy(adjData, adjBatchOffset, adjSliceData, 0, numNodes * numNodes);
                    adjSliceBuffer = backend.AllocateBuffer(adjSliceData);
                }

                using var convOutputBuffer = backend.AllocateBuffer(numNodes * _outputFeatures);
                backend.Gemm(adjSliceBuffer, xwBuffer, convOutputBuffer, numNodes, _outputFeatures, numNodes);

                // Accumulate to output at correct batch offset
                // Download current output for this batch, add convOutput, re-upload
                var currentOutputSlice = new float[numNodes * _outputFeatures];
                var downloadedOutput = backend.DownloadBuffer(outputBuffer);
                Array.Copy(downloadedOutput, batchOutputOffset, currentOutputSlice, 0, numNodes * _outputFeatures);
                using var currentOutputBuffer = backend.AllocateBuffer(currentOutputSlice);

                using var accumulatedBuffer = backend.AllocateBuffer(numNodes * _outputFeatures);
                backend.Add(currentOutputBuffer, convOutputBuffer, accumulatedBuffer, numNodes * _outputFeatures);

                // Copy back to output buffer at correct offset
                backend.Copy2DStrided(accumulatedBuffer, outputBuffer, 1, numNodes * _outputFeatures, outputSize, batchOutputOffset);

                if (needsDispose)
                {
                    inputSliceBuffer.Dispose();
                }

                if (normalizedAdj.Shape.Length == 3 && batchSize > 1 && adjSliceBuffer != adjBuffer)
                {
                    adjSliceBuffer.Dispose();
                }
            }
        }

        // ========================================
        // SELF-LOOP WEIGHTS AND BIASES PER NODE TYPE
        // ========================================

        // Precompute self-loop contributions for all node types
        foreach (var nodeType in _metadata.NodeTypes)
        {
            var selfWeights = _selfLoopWeights[nodeType];
            var bias = _biases[nodeType];
            int inFeatures = _metadata.NodeTypeFeatures[nodeType];

            // Upload weights and bias
            using var selfWeightsBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(selfWeights.Data));
            using var biasBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(bias.Data));

            // Create a mask for nodes of this type
            var nodeMask = new float[numNodes];
            for (int n = 0; n < numNodes; n++)
            {
                nodeMask[n] = (_nodeTypeMap.TryGetValue(n, out var type) && type == nodeType) ? 1.0f : 0.0f;
            }

            // For each batch, apply self-loop transformation to nodes of this type
            for (int b = 0; b < batchSize; b++)
            {
                int batchInputOffset = b * numNodes * inputFeatures;
                int batchOutputOffset = b * numNodes * _outputFeatures;

                // For each node of this type in the batch
                for (int n = 0; n < numNodes; n++)
                {
                    if (nodeMask[n] < 0.5f)
                        continue;

                    // Extract this node's input [1, inFeatures]
                    var nodeInputData = new float[inFeatures];
                    var fullInputData = backend.DownloadBuffer(input.CreateView(batchInputOffset, [numNodes, inputFeatures]).Buffer);
                    for (int f = 0; f < inFeatures && f < inputFeatures; f++)
                    {
                        nodeInputData[f] = fullInputData[n * inputFeatures + f];
                    }
                    using var nodeInputBuffer = backend.AllocateBuffer(nodeInputData);

                    // Compute selfOutput = nodeInput @ selfWeights : [1, inFeatures] @ [inFeatures, outputFeatures] -> [1, outputFeatures]
                    using var selfOutputBuffer = backend.AllocateBuffer(_outputFeatures);
                    backend.Gemm(nodeInputBuffer, selfWeightsBuffer, selfOutputBuffer, 1, _outputFeatures, inFeatures);

                    // Add bias using BiasAdd
                    backend.BiasAdd(selfOutputBuffer, biasBuffer, selfOutputBuffer, 1, _outputFeatures);

                    // Add to output at this node's position
                    // Get current output for this node
                    var fullOutputData = backend.DownloadBuffer(outputBuffer);
                    int nodeOutputOffset = batchOutputOffset + n * _outputFeatures;

                    var nodeOutputData = new float[_outputFeatures];
                    Array.Copy(fullOutputData, nodeOutputOffset, nodeOutputData, 0, _outputFeatures);
                    using var nodeOutputBuffer = backend.AllocateBuffer(nodeOutputData);

                    // Add self-loop output
                    using var accumulatedNodeBuffer = backend.AllocateBuffer(_outputFeatures);
                    backend.Add(nodeOutputBuffer, selfOutputBuffer, accumulatedNodeBuffer, _outputFeatures);

                    // Copy back using Copy2DStrided
                    backend.Copy2DStrided(accumulatedNodeBuffer, outputBuffer, 1, _outputFeatures, outputSize, nodeOutputOffset);
                }
            }
        }

        // ========================================
        // APPLY ACTIVATION
        // ========================================

        ApplyGpuActivation(backend, outputBuffer, outputBuffer, outputSize, GetFusedActivationType());

        // Create output tensor with appropriate shape
        int[] outputShape = batchSize == 1
            ? [numNodes, _outputFeatures]
            : [batchSize, numNodes, _outputFeatures];

        var finalBuffer = backend.AllocateBuffer(outputSize);
        backend.Copy(outputBuffer, finalBuffer, outputSize);
        return new GpuTensor<T>(backend, finalBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: true);
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
        var result = new Tensor<T>([rows, cols]);
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
        var result = new Tensor<T>([batchSize, numNodes, inFeatures]);
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

        var normalized = new Tensor<T>(adj3D.Shape);

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
        var result = new Tensor<T>([batchSize, inFeatures]);
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
    /// Computes the backward pass for this Heterogeneous Graph layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to this layer's output.</param>
    /// <returns>The gradient of the loss with respect to this layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method computes gradients for all type-specific parameters including edge type weights,
    /// self-loop weights, biases, and basis decomposition parameters if enabled.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _adjacencyMatrices == null || _nodeTypeMap == null)
        {
            throw new InvalidOperationException("Forward pass must be called before Backward.");
        }

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);

        int batchSize = _lastInput.Shape[0];
        int numNodes = _lastInput.Shape[1];
        int inputFeatures = _lastInput.Shape[2];

        // Initialize gradient accumulators
        _edgeTypeWeightsGradients = new Dictionary<string, Tensor<T>>();
        _selfLoopWeightsGradients = new Dictionary<string, Tensor<T>>();
        _biasesGradients = new Dictionary<string, Tensor<T>>();

        var inputGradient = new Tensor<T>(_lastInput.Shape);
        inputGradient.Fill(NumOps.Zero);

        // Compute gradients for edge type weights
        // Forward: output = normalizedAdj @ input @ weights
        // Backward: dL/dweights = sum_batch(input^T @ normalizedAdj^T @ grad)
        //           dL/dinput = normalizedAdj^T @ grad @ weights^T
        foreach (var edgeType in _metadata.EdgeTypes)
        {
            if (!_adjacencyMatrices.TryGetValue(edgeType, out var adjacency))
                continue;

            var (sourceType, targetType) = _metadata.EdgeTypeSchema[edgeType];
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

            // Normalize adjacency for this edge type
            var normalizedAdj = NormalizeAdjacency(adjacency, batchSize, numNodes);

            // Extract input features for source type
            var inputSlice = _lastInput.Shape[2] == inFeatures ? _lastInput :
                ExtractInputFeatures(_lastInput, batchSize, numNodes, inFeatures);

            // Compute weight gradient: dL/dW = sum_batch(input^T @ adj^T @ grad)
            var weightGradient = new Tensor<T>([inFeatures, _outputFeatures]);
            weightGradient.Fill(NumOps.Zero);

            // For each batch: adj^T @ grad -> [numNodes, outputFeatures]
            // Then: input^T @ (adj^T @ grad) -> [inFeatures, outputFeatures]
            for (int b = 0; b < batchSize; b++)
            {
                // Extract batch slices
                var adjBatch = ExtractBatchSlice(normalizedAdj, b, numNodes, numNodes);
                var gradBatch = ExtractBatchSlice(activationGradient, b, numNodes, _outputFeatures);
                var inputBatch = ExtractBatchSlice(inputSlice, b, numNodes, inFeatures);

                // adj^T @ grad: transpose adjacency and multiply
                var adjT = Engine.TensorTranspose(adjBatch);
                var adjTGrad = Engine.TensorMatMul(adjT, gradBatch); // [numNodes, outputFeatures]

                // input^T @ (adj^T @ grad)
                var inputT = Engine.TensorTranspose(inputBatch);
                var batchWeightGrad = Engine.TensorMatMul(inputT, adjTGrad); // [inFeatures, outputFeatures]

                weightGradient = Engine.TensorAdd(weightGradient, batchWeightGrad);
            }

            _edgeTypeWeightsGradients[edgeType] = weightGradient;

            // Compute input gradient: dL/dinput = adj^T @ grad @ weights^T
            var weightsT = Engine.TensorTranspose(weights);
            for (int b = 0; b < batchSize; b++)
            {
                var adjBatch = ExtractBatchSlice(normalizedAdj, b, numNodes, numNodes);
                var gradBatch = ExtractBatchSlice(activationGradient, b, numNodes, _outputFeatures);

                // adj^T @ grad
                var adjT = Engine.TensorTranspose(adjBatch);
                var adjTGrad = Engine.TensorMatMul(adjT, gradBatch);

                // (adj^T @ grad) @ weights^T
                var inputGradBatch = Engine.TensorMatMul(adjTGrad, weightsT);

                // Add to input gradient (only first inFeatures)
                for (int n = 0; n < numNodes; n++)
                {
                    for (int f = 0; f < Math.Min(inFeatures, inputFeatures); f++)
                    {
                        inputGradient[b, n, f] = NumOps.Add(inputGradient[b, n, f], inputGradBatch[n, f]);
                    }
                }
            }
        }

        // Compute gradients for self-loop weights and biases
        // Forward: output += input @ selfWeights + bias (for each node of the type)
        // Backward: dL/dselfWeights = sum_batch(input^T @ grad) for nodes of this type
        //           dL/dbias = sum(grad) for nodes of this type
        //           dL/dinput += grad @ selfWeights^T
        foreach (var nodeType in _metadata.NodeTypes)
        {
            int inFeatures = _metadata.NodeTypeFeatures[nodeType];

            var selfWeightGradient = new Tensor<T>([inFeatures, _outputFeatures]);
            selfWeightGradient.Fill(NumOps.Zero);

            var biasGradient = new Tensor<T>([_outputFeatures]);
            biasGradient.Fill(NumOps.Zero);

            var selfWeights = _selfLoopWeights[nodeType];
            var selfWeightsT = Engine.TensorTranspose(selfWeights);

            // For each node of this type
            for (int i = 0; i < numNodes; i++)
            {
                if (_nodeTypeMap[i] != nodeType)
                    continue;

                for (int b = 0; b < batchSize; b++)
                {
                    // Accumulate bias gradient
                    for (int f = 0; f < _outputFeatures; f++)
                    {
                        biasGradient[f] = NumOps.Add(biasGradient[f], activationGradient[b, i, f]);
                    }

                    // Accumulate self-weight gradient: input^T @ grad for this node
                    for (int inF = 0; inF < inFeatures && inF < inputFeatures; inF++)
                    {
                        for (int outF = 0; outF < _outputFeatures; outF++)
                        {
                            T contribution = NumOps.Multiply(_lastInput[b, i, inF], activationGradient[b, i, outF]);
                            selfWeightGradient[inF, outF] = NumOps.Add(selfWeightGradient[inF, outF], contribution);
                        }
                    }

                    // Accumulate input gradient: grad @ selfWeights^T
                    for (int inF = 0; inF < inFeatures && inF < inputFeatures; inF++)
                    {
                        T gradSum = NumOps.Zero;
                        for (int outF = 0; outF < _outputFeatures; outF++)
                        {
                            gradSum = NumOps.Add(gradSum, NumOps.Multiply(activationGradient[b, i, outF], selfWeights[inF, outF]));
                        }
                        inputGradient[b, i, inF] = NumOps.Add(inputGradient[b, i, inF], gradSum);
                    }
                }
            }

            _selfLoopWeightsGradients[nodeType] = selfWeightGradient;
            _biasesGradients[nodeType] = biasGradient;
        }

        // Restore gradient shape to match original input shape
        if (_originalInputShape != null && _originalInputShape.Length != 3)
        {
            return inputGradient.Reshape(_originalInputShape);
        }

        return inputGradient;
    }

    /// <summary>
    /// Extracts a 2D batch slice from a 3D tensor.
    /// </summary>
    private Tensor<T> ExtractBatchSlice(Tensor<T> tensor, int batchIndex, int rows, int cols)
    {
        var result = new Tensor<T>([rows, cols]);
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

    /// <inheritdoc/>
    public override bool SupportsJitCompilation => true;

    /// <inheritdoc/>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null || inputNodes.Count < 1)
        {
            throw new ArgumentException("HeterogeneousGraphLayer requires at least 1 input node (node features).");
        }

        if (_adjacencyMatrices == null || _nodeTypeMap == null)
        {
            throw new InvalidOperationException(
                "Adjacency matrices and node type map must be set before exporting computation graph. " +
                "Call SetAdjacencyMatrices() and SetNodeTypeMap() first.");
        }

        var inputNode = inputNodes[0]; // Node features [batch, numNodes, inputFeatures]
        var inputShape = inputNode.Value.Shape;
        int batchSize = inputShape[0];
        int numNodes = inputShape[1];

        // Create output accumulator initialized to zero
        var outputTensor = new Tensor<T>(new int[] { batchSize, numNodes, _outputFeatures });
        outputTensor.Fill(NumOps.Zero);
        var outputNode = TensorOperations<T>.Constant(outputTensor, "hgnn_output_init");

        // Process each edge type
        foreach (var edgeType in _metadata.EdgeTypes)
        {
            if (!_adjacencyMatrices.TryGetValue(edgeType, out var adjacency))
                continue;

            var (sourceType, _) = _metadata.EdgeTypeSchema[edgeType];
            int inFeatures = _metadata.NodeTypeFeatures[sourceType];

            // Get weights for this edge type
            Tensor<T> weights;
            if (_useBasis && _basisMatrices != null && _basisCoefficients != null)
            {
                // Reconstruct weights from basis decomposition
                var coeffs = _basisCoefficients[edgeType];
                weights = new Tensor<T>(new int[] { InputFeatures, _outputFeatures });
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

            // Create constant nodes for weights and normalized adjacency
            var weightsNode = TensorOperations<T>.Constant(weights, $"edge_weights_{edgeType}");

            // Normalize and store adjacency as constant
            var normalizedAdj = NormalizeAdjacency(adjacency, batchSize, numNodes);
            var adjNode = TensorOperations<T>.Constant(normalizedAdj, $"adj_{edgeType}");

            // Extract input features if needed (or use full input if dimensions match)
            ComputationNode<T> inputSlice;
            if (inputShape[2] == inFeatures)
            {
                inputSlice = inputNode;
            }
            else
            {
                // Extract relevant input features at export time by slicing
                var extractedFeatures = ExtractInputFeatures(inputNode.Value, batchSize, numNodes, inFeatures);
                inputSlice = TensorOperations<T>.Constant(extractedFeatures, $"input_slice_{edgeType}");
            }

            // Compute: normalizedAdj @ inputSlice @ weights
            // First: inputSlice @ weights (batched matrix multiply across last dimensions)
            var xw = TensorOperations<T>.BatchMatrixMultiply(inputSlice, weightsNode);

            // Then: adj @ xw for message passing
            var convOutput = TensorOperations<T>.BatchMatrixMultiply(adjNode, xw);

            // Accumulate to output
            outputNode = TensorOperations<T>.Add(outputNode, convOutput);
        }

        // Add self-loops and biases per node type
        // For JIT compilation, we precompute the self-loop contribution for all nodes
        foreach (var nodeType in _metadata.NodeTypes)
        {
            var selfWeights = _selfLoopWeights[nodeType];
            var bias = _biases[nodeType];
            int inFeatures = _metadata.NodeTypeFeatures[nodeType];

            // Create constant nodes
            var selfWeightsNode = TensorOperations<T>.Constant(selfWeights, $"self_weights_{nodeType}");
            var biasNode = TensorOperations<T>.Constant(bias, $"bias_{nodeType}");

            // Create a mask tensor for nodes of this type
            var nodeMask = new Tensor<T>(new int[] { batchSize, numNodes, 1 });
            nodeMask.Fill(NumOps.Zero);
            for (int n = 0; n < numNodes; n++)
            {
                if (_nodeTypeMap.TryGetValue(n, out var type) && type == nodeType)
                {
                    for (int b = 0; b < batchSize; b++)
                    {
                        nodeMask[b, n, 0] = NumOps.One;
                    }
                }
            }
            var maskNode = TensorOperations<T>.Constant(nodeMask, $"node_mask_{nodeType}");

            // For nodes of this type: compute input @ selfWeights and add bias
            // Since we need type-specific extraction, compute it for all nodes
            // then mask to keep only relevant nodes

            // Extract features for this node type
            ComputationNode<T> typeInput;
            if (inputShape[2] == inFeatures)
            {
                typeInput = inputNode;
            }
            else
            {
                var extractedFeatures = ExtractInputFeatures(inputNode.Value, batchSize, numNodes, inFeatures);
                typeInput = TensorOperations<T>.Constant(extractedFeatures, $"type_input_{nodeType}");
            }

            // Compute self-loop transformation: input @ selfWeights
            var selfOutput = TensorOperations<T>.BatchMatrixMultiply(typeInput, selfWeightsNode);

            // Broadcast and add bias
            var biasBroadcast = new Tensor<T>(new int[] { batchSize, numNodes, _outputFeatures });
            for (int b = 0; b < batchSize; b++)
            {
                for (int n = 0; n < numNodes; n++)
                {
                    for (int f = 0; f < _outputFeatures; f++)
                    {
                        biasBroadcast[b, n, f] = bias[f];
                    }
                }
            }
            var biasBroadcastNode = TensorOperations<T>.Constant(biasBroadcast, $"bias_broadcast_{nodeType}");
            selfOutput = TensorOperations<T>.Add(selfOutput, biasBroadcastNode);

            // Mask to only keep contribution from nodes of this type
            // Broadcast mask to match output features
            var maskBroadcast = new Tensor<T>(new int[] { batchSize, numNodes, _outputFeatures });
            for (int b = 0; b < batchSize; b++)
            {
                for (int n = 0; n < numNodes; n++)
                {
                    T maskVal = nodeMask[b, n, 0];
                    for (int f = 0; f < _outputFeatures; f++)
                    {
                        maskBroadcast[b, n, f] = maskVal;
                    }
                }
            }
            var maskBroadcastNode = TensorOperations<T>.Constant(maskBroadcast, $"mask_broadcast_{nodeType}");

            // Apply mask: selfOutput * mask (element-wise)
            var maskedSelfOutput = TensorOperations<T>.ElementwiseMultiply(selfOutput, maskBroadcastNode);

            // Accumulate to output
            outputNode = TensorOperations<T>.Add(outputNode, maskedSelfOutput);
        }

        // Apply activation function
        if (ScalarActivation is not null && ScalarActivation is not IdentityActivation<T>)
        {
            if (ScalarActivation.SupportsJitCompilation)
            {
                outputNode = ScalarActivation.ApplyToGraph(outputNode);
            }
            else
            {
                var activated = ScalarActivation.Activate(outputNode.Value);
                outputNode = TensorOperations<T>.Constant(activated, "activated_output");
            }
        }

        return outputNode;
    }
}
