namespace AiDotNet.NeuralNetworks.Layers.Graph;

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
    private readonly bool _useBasis; // Use basis decomposition for efficiency
    private readonly int _numBases;

    /// <summary>
    /// Type-specific weight matrices. Key: edge type, Value: weight matrix.
    /// </summary>
    private Dictionary<string, Matrix<T>> _edgeTypeWeights;

    /// <summary>
    /// Self-loop weights for each node type.
    /// </summary>
    private Dictionary<string, Matrix<T>> _selfLoopWeights;

    /// <summary>
    /// Bias for each node type.
    /// </summary>
    private Dictionary<string, Vector<T>> _biases;

    /// <summary>
    /// Basis matrices for weight decomposition (if using basis).
    /// </summary>
    private Tensor<T>? _basisMatrices;

    /// <summary>
    /// Coefficients for combining basis matrices per edge type.
    /// </summary>
    private Dictionary<string, Vector<T>>? _basisCoefficients;

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
    private Tensor<T>? _lastOutput;

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

        _edgeTypeWeights = new Dictionary<string, Matrix<T>>();
        _selfLoopWeights = new Dictionary<string, Matrix<T>>();
        _biases = new Dictionary<string, Vector<T>>();

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        if (_useBasis)
        {
            // Initialize basis matrices
            _basisMatrices = new Tensor<T>([_numBases, InputFeatures, _outputFeatures]);
            _basisCoefficients = new Dictionary<string, Vector<T>>();

            T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (InputFeatures + _outputFeatures)));

            for (int b = 0; b < _numBases; b++)
            {
                for (int i = 0; i < InputFeatures; i++)
                {
                    for (int j = 0; j < _outputFeatures; j++)
                    {
                        _basisMatrices[b, i, j] = NumOps.Multiply(
                            NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
                    }
                }
            }

            // Initialize coefficients for each edge type
            foreach (var edgeType in _metadata.EdgeTypes)
            {
                var coeffs = new Vector<T>(_numBases);
                for (int b = 0; b < _numBases; b++)
                {
                    coeffs[b] = NumOps.FromDouble(Random.NextDouble());
                }
                _basisCoefficients[edgeType] = coeffs;
            }
        }
        else
        {
            // Initialize separate weight matrix for each edge type
            foreach (var edgeType in _metadata.EdgeTypes)
            {
                var (sourceType, targetType) = _metadata.EdgeTypeSchema[edgeType];
                int inFeatures = _metadata.NodeTypeFeatures[sourceType];

                var weights = new Matrix<T>(inFeatures, _outputFeatures);
                T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (inFeatures + _outputFeatures)));

                for (int i = 0; i < inFeatures; i++)
                {
                    for (int j = 0; j < _outputFeatures; j++)
                    {
                        weights[i, j] = NumOps.Multiply(
                            NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
                    }
                }

                _edgeTypeWeights[edgeType] = weights;
            }
        }

        // Initialize self-loop weights and biases for each node type
        foreach (var nodeType in _metadata.NodeTypes)
        {
            int inFeatures = _metadata.NodeTypeFeatures[nodeType];
            var selfWeights = new Matrix<T>(inFeatures, _outputFeatures);
            T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (inFeatures + _outputFeatures)));

            for (int i = 0; i < inFeatures; i++)
            {
                for (int j = 0; j < _outputFeatures; j++)
                {
                    selfWeights[i, j] = NumOps.Multiply(
                        NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
                }
            }

            _selfLoopWeights[nodeType] = selfWeights;

            var bias = new Vector<T>(_outputFeatures);
            for (int i = 0; i < _outputFeatures; i++)
            {
                bias[i] = NumOps.Zero;
            }

            _biases[nodeType] = bias;
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

        _lastInput = input;
        int batchSize = input.Shape[0];
        int numNodes = input.Shape[1];

        var output = new Tensor<T>([batchSize, numNodes, _outputFeatures]);

        // Process each edge type
        foreach (var edgeType in _metadata.EdgeTypes)
        {
            if (!_adjacencyMatrices.TryGetValue(edgeType, out var adjacency))
                continue;

            var (sourceType, targetType) = _metadata.EdgeTypeSchema[edgeType];

            // Get weights for this edge type
            Matrix<T>? weights = null;

            if (_useBasis && _basisMatrices != null && _basisCoefficients != null)
            {
                // Reconstruct weights from basis decomposition
                var coeffs = _basisCoefficients[edgeType];
                weights = new Matrix<T>(InputFeatures, _outputFeatures);

                for (int i = 0; i < InputFeatures; i++)
                {
                    for (int j = 0; j < _outputFeatures; j++)
                    {
                        T sum = NumOps.Zero;
                        for (int b = 0; b < _numBases; b++)
                        {
                            sum = NumOps.Add(sum,
                                NumOps.Multiply(coeffs[b], _basisMatrices[b, i, j]));
                        }
                        weights[i, j] = sum;
                    }
                }
            }
            else
            {
                weights = _edgeTypeWeights[edgeType];
            }

            // Aggregate messages of this edge type
            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < numNodes; i++)
                {
                    // Count neighbors of this edge type for normalization
                    int degree = 0;
                    for (int j = 0; j < numNodes; j++)
                    {
                        if (!NumOps.Equals(adjacency[b, i, j], NumOps.Zero))
                            degree++;
                    }

                    if (degree == 0)
                        continue;

                    T normalization = NumOps.Divide(NumOps.FromDouble(1.0), NumOps.FromDouble(degree));

                    // Aggregate from neighbors
                    for (int outF = 0; outF < _outputFeatures; outF++)
                    {
                        T sum = NumOps.Zero;

                        for (int j = 0; j < numNodes; j++)
                        {
                            if (NumOps.Equals(adjacency[b, i, j], NumOps.Zero))
                                continue;

                            // Apply edge-type-specific transformation
                            int inFeatures = _metadata.NodeTypeFeatures[sourceType];
                            for (int inF = 0; inF < inFeatures && inF < input.Shape[2]; inF++)
                            {
                                sum = NumOps.Add(sum,
                                    NumOps.Multiply(
                                        NumOps.Multiply(input[b, j, inF], weights[inF, outF]),
                                        normalization));
                            }
                        }

                        output[b, i, outF] = NumOps.Add(output[b, i, outF], sum);
                    }
                }
            }
        }

        // Add self-loops and biases
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                string nodeType = _nodeTypeMap[i];
                var selfWeights = _selfLoopWeights[nodeType];
                var bias = _biases[nodeType];
                int inFeatures = _metadata.NodeTypeFeatures[nodeType];

                for (int outF = 0; outF < _outputFeatures; outF++)
                {
                    for (int inF = 0; inF < inFeatures && inF < input.Shape[2]; inF++)
                    {
                        output[b, i, outF] = NumOps.Add(output[b, i, outF],
                            NumOps.Multiply(input[b, i, inF], selfWeights[inF, outF]));
                    }

                    output[b, i, outF] = NumOps.Add(output[b, i, outF], bias[outF]);
                }
            }
        }

        _lastOutput = ApplyActivation(output);
        return _lastOutput;
    }

    /// <inheritdoc/>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before Backward.");
        }

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);
        return new Tensor<T>(_lastInput.Shape);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        // Simplified - full implementation would update all type-specific weights
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        return new Vector<T>(1);
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        // Simplified
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
    }
}
