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

        // Handle any-rank tensor: collapse to 2D for processing
        Tensor<T> processInput;
        int batchSize;

        if (rank == 1)
        {
            batchSize = 1;
            processInput = input.Reshape([1, input.Shape[0]]);
        }
        else if (rank == 2)
        {
            batchSize = input.Shape[0];
            processInput = input;
        }
        else
        {
            int flatBatch = 1;
            for (int d = 0; d < rank - 1; d++)
                flatBatch *= input.Shape[d];
            batchSize = flatBatch;
            processInput = input.Reshape([flatBatch, input.Shape[rank - 1]]);
        }

        _lastInput = processInput;
        int numNodes = input.Shape[1];

        var output = new Tensor<T>([batchSize, numNodes, _outputFeatures]);
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
            var inputSlice = input.Shape[2] == inFeatures ? input :
                ExtractInputFeatures(input, batchSize, numNodes, inFeatures);

            // Compute normalization factor from adjacency matrix
            var normalizedAdj = NormalizeAdjacency(adjacency, batchSize, numNodes);

            // Perform graph convolution: normalizedAdj @ inputSlice @ weights
            var xw = BatchedMatMul3Dx2D(inputSlice, weights, batchSize, numNodes, inFeatures, _outputFeatures);
            var convOutput = Engine.BatchMatMul(normalizedAdj, xw);

            // Accumulate to output
            output = Engine.TensorAdd(output, convOutput);
        }

        // Add self-loops and biases
        for (int i = 0; i < numNodes; i++)
        {
            string nodeType = _nodeTypeMap[i];
            var selfWeights = _selfLoopWeights[nodeType];
            var bias = _biases[nodeType];
            int inFeatures = _metadata.NodeTypeFeatures[nodeType];

            // Extract input for this node across all batches
            var nodeInput = ExtractNodeInput(input, batchSize, i, inFeatures);

            // Apply self-loop transformation: nodeInput @ selfWeights
            var selfOutput = Engine.TensorMatMul(nodeInput, selfWeights); // [batch, outputFeatures]

            // Broadcast bias across batch
            var biasBroadcast = BroadcastBias(bias, batchSize);

            // Add to output at this node position
            output = AddNodeOutput(output, selfOutput, biasBroadcast, batchSize, i, _outputFeatures);
        }

        _lastOutput = ApplyActivation(output);
        return _lastOutput;
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
        var normalized = new Tensor<T>(adjacency.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                // Count degree
                int degree = 0;
                for (int j = 0; j < numNodes; j++)
                {
                    if (!NumOps.Equals(adjacency[b, i, j], NumOps.Zero))
                        degree++;
                }

                if (degree == 0)
                {
                    // No neighbors, copy zeros
                    for (int j = 0; j < numNodes; j++)
                    {
                        normalized[b, i, j] = NumOps.Zero;
                    }
                }
                else
                {
                    T normalization = NumOps.Divide(NumOps.One, NumOps.FromDouble(degree));
                    for (int j = 0; j < numNodes; j++)
                    {
                        normalized[b, i, j] = NumOps.Multiply(adjacency[b, i, j], normalization);
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
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc/>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException(
            "HeterogeneousGraphLayer does not support computation graph export due to type-specific transformations.");
    }
}
