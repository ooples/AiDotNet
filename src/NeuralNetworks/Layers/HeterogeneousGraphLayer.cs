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
public partial class HeterogeneousGraphMetadata : AiDotNet.Serialization.ILayerStatePersistable
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

    /// <inheritdoc />
    /// <remarks>
    /// Written by hand rather than routed through JSON because EdgeTypeSchema's value is a
    /// ValueTuple, whose members are FIELDS: System.Text.Json emits {} for it and the edge schema
    /// would come back empty with nothing reporting a problem. Here SourceType and TargetType are
    /// still named, so they survive.
    /// </remarks>
    public string SaveState()
    {
        var sb = new System.Text.StringBuilder();
        sb.Append(string.Join(",", NodeTypes)).Append('|');
        sb.Append(string.Join(",", EdgeTypes)).Append('|');
        sb.Append(string.Join(",", System.Linq.Enumerable.Select(
            NodeTypeFeatures, kv => kv.Key + "=" + kv.Value.ToString(
                System.Globalization.CultureInfo.InvariantCulture)))).Append('|');
        sb.Append(string.Join(",", System.Linq.Enumerable.Select(
            EdgeTypeSchema, kv => kv.Key + "=" + kv.Value.SourceType + ">" + kv.Value.TargetType)));
        return sb.ToString();
    }

    /// <inheritdoc />
    public void LoadState(string text)
    {
        var parts = (text ?? string.Empty).Split('|');
        string Part(int n) => n < parts.Length ? parts[n] : string.Empty;

        NodeTypes = Part(0).Length == 0 ? [] : Part(0).Split(',');
        EdgeTypes = Part(1).Length == 0 ? [] : Part(1).Split(',');

        NodeTypeFeatures = new Dictionary<string, int>();
        foreach (var entry in Part(2).Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries))
        {
            int eq = entry.LastIndexOf('=');
            if (eq <= 0) continue;
            NodeTypeFeatures[entry.Substring(0, eq)] = int.Parse(
                entry.Substring(eq + 1), System.Globalization.CultureInfo.InvariantCulture);
        }

        EdgeTypeSchema = new Dictionary<string, (string SourceType, string TargetType)>();
        foreach (var entry in Part(3).Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries))
        {
            int eq = entry.LastIndexOf('=');
            if (eq <= 0) continue;
            var pair = entry.Substring(eq + 1).Split('>');
            if (pair.Length != 2) continue;
            EdgeTypeSchema[entry.Substring(0, eq)] = (pair[0], pair[1]);
        }
    }
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
// Rank 3 - "Already 3D [batchSize, numNodes, features]", the branch that needs no reshape at all - is
// both what the forward treats as canonical and the shape [LayerProperty(TestInputShape = "1, 4, 8")]
// exercises. The rank-1 and rank-2 forms are handled by adding axes, and rank > 3 flattens leading axes
// into an anonymous batch; none of those node counts can be checked against the separately-installed
// per-edge-type adjacency matrices, so they are left undeclared.
//
// The node axis is TensorAxis.Other: nodes here are typed graph vertices, not a sequence or a spatial
// extent, and their ordering is meaningful only as an index into the installed node-type map.
[TensorLayout(TensorAxis.Batch, TensorAxis.Other, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input,
    Note = "Typed node features; the node axis is indexed by the map installed via SetNodeTypeMap.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Other, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class HeterogeneousGraphLayer<T> : LayerBase<T>, IGraphConvolutionLayer<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// HAND-WRITTEN because the emitted width is one constructor argument and NOT a sum over edge types.
    /// That is worth stating rather than assuming: a relational layer running a separate projection per
    /// edge type looks as though it ought to widen, but the accumulator is allocated once as
    /// <c>[batchSize, processNumNodes, _outputFeatures]</c> and every edge type is ADDED into that same
    /// buffer, so the width stays <c>_outputFeatures</c> however many edge types the metadata declares.
    /// </para>
    /// <para>
    /// The node axis is Same: message passing rewrites what each node holds, never how many there are.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 3 || _outputFeatures <= 0) return null;

        return new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Other, AxisRelation.Same(TensorAxis.Other)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_outputFeatures)),
        };
    }

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
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// Gradients for weights, self-loop weights, and biases.
    /// </summary>
    private Dictionary<string, Tensor<T>>? _edgeTypeWeightsGradients;
    private Dictionary<string, Tensor<T>>? _selfLoopWeightsGradients;
    private Dictionary<string, Tensor<T>>? _biasesGradients;
    [Scratch]
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
            T coeffScale = NumOps.FromDouble(1.0 / _numBases);
            foreach (var edgeType in _metadata.EdgeTypes)
            {
                var coeffs = new Tensor<T>([_numBases]);
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
                InitializeGlorotTensor(weights, 2.0 / (inFeatures + _outputFeatures));

                _edgeTypeWeights[edgeType] = weights;
            }
        }

        // Initialize self-loop weights and biases for each node type
        foreach (var nodeType in _metadata.NodeTypes)
        {
            int inFeatures = _metadata.NodeTypeFeatures[nodeType];

            var selfWeights = new Tensor<T>([inFeatures, _outputFeatures]);
            InitializeGlorotTensor(selfWeights, 2.0 / (inFeatures + _outputFeatures));
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
        var randomTensor = Tensor<T>.CreateRandom(Random, tensor._shape);
        var halfTensor = new Tensor<T>(tensor._shape);
        halfTensor.Fill(NumOps.FromDouble(0.5));
        var shifted = Engine.TensorSubtract(randomTensor, halfTensor);
        var scaled = Engine.TensorMultiplyScalar(shifted, scale);

        scaled.CopyTo(tensor.AsWritableSpan());
    }

    /// <summary>Initializes a tensor with scale sqrt(variance), the scale computed in T as before.</summary>
    private void InitializeGlorotTensor(Tensor<T> tensor, double variance)
    {
        InitializeTensor(tensor, NumOps.Sqrt(NumOps.FromDouble(variance)));
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
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
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
                weights = ReconstructBasisWeights(_basisMatrices, coeffs);
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

        return result;
    }

    /// <summary>
    /// Converts a normalized adjacency tensor to CSR format (using first batch element).
    /// </summary>
    private (float[] Values, int[] ColIndices, int[] RowPointers) ConvertToCSR(Tensor<T> adjacency, int numNodes)
    {
        if (numNodes == 0)
            return (Array.Empty<float>(), Array.Empty<int>(), new[] { 0 });

        bool is2D = adjacency.Shape.Length == 2;
        var graph = is2D
            ? Engine.TensorSlice(adjacency, [0, 0], [numNodes, numNodes])
            : Engine.Reshape(Engine.TensorSlice(adjacency, [0, 0, 0], [1, numNodes, numNodes]), [numNodes, numNodes]);

        // Entries with |value| > 1e-6 in float (NaN is not), row-major, as CSR.
        var floats = Engine.Reshape(graph.Cast<float>(), [numNodes * numNodes]);
        var kept = Engine.TensorGreaterThan(Engine.TensorAbs(floats), 1e-6f);
        var nonzero = Engine.TensorNonzero(kept);   // [count, 1]
        int count = nonzero.Shape[0];
        var flatIndices = Engine.Reshape(nonzero, [count]);
        var values = count == 0 ? Array.Empty<float>() : Engine.TensorGather(floats, flatIndices, axis: 0).ToVector().ToArray();
        var colIndices = count == 0 ? Array.Empty<int>() : flatIndices.ToVector().ToArray().Select(flat => flat % numNodes).ToArray();
        var rowCounts = Engine.ReduceSum(Engine.Reshape(kept, [numNodes, numNodes]), new[] { 1 }, keepDims: false).Cast<int>();
        var rowPointers = new[] { 0 }.Concat(Engine.TensorCumSum(rowCounts, 0).ToVector().ToArray()).ToArray();

        return (values, colIndices, rowPointers);
    }

    /// <summary>
    /// Extracts a basis matrix from the basis matrices tensor.
    /// </summary>
    private Tensor<T> ExtractBasisMatrix(Tensor<T> basisMatrices, int basisIndex, int rows, int cols)
    {
        if (rows == 0 || cols == 0) return new Tensor<T>([rows, cols]);

        // A plain copy of basis basisIndex (off the tape, as the element-wise copy was).
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            return Engine.Reshape(
                Engine.TensorSlice(basisMatrices, [basisIndex, 0, 0], [1, rows, cols]), [rows, cols]).Clone();
        }
    }

    /// <summary>
    /// W_r = sum_b a_rb * V_b over the [numBases, InputFeatures, outputFeatures] basis, as a plain value
    /// (off the tape), as the copied-basis accumulation was.
    /// </summary>
    private Tensor<T> ReconstructBasisWeights(Tensor<T> basisMatrices, Tensor<T> coefficients)
    {
        int width = InputFeatures * _outputFeatures;
        if (_numBases == 0 || width == 0) return new Tensor<T>([InputFeatures, _outputFeatures]);

        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            var bases = Engine.Reshape(
                Engine.TensorSlice(basisMatrices, [0, 0, 0], [_numBases, InputFeatures, _outputFeatures]),
                [_numBases, width]);
            var coefficientColumn = Engine.Reshape(
                Engine.TensorSlice(Engine.Reshape(coefficients, [coefficients.Length]), [0], [_numBases]), [_numBases, 1]);
            return Engine.Reshape(
                Engine.ReduceSum(Engine.TensorMultiply(bases, coefficientColumn), new[] { 0 }, keepDims: false),
                [InputFeatures, _outputFeatures]);
        }
    }

    /// <summary>
    /// Extracts input features for a specific number of features.
    /// </summary>
    private Tensor<T> ExtractInputFeatures(Tensor<T> input, int batchSize, int numNodes, int inFeatures)
    {
        // The first min(inFeatures, available) features, zero-padded to inFeatures; a plain copy (off the tape).
        int kept = Math.Min(inFeatures, input.Shape[2]);
        if (batchSize == 0 || numNodes == 0 || kept <= 0) return new Tensor<T>([batchSize, numNodes, inFeatures]);

        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            var head = Engine.TensorSlice(input, [0, 0, 0], [batchSize, numNodes, kept]);
            return kept == inFeatures
                ? head.Clone()
                : Engine.TensorConcatenate(new[] { head, new Tensor<T>([batchSize, numNodes, inFeatures - kept]) }, 2);
        }
    }

    /// <summary>
    /// Normalizes adjacency matrix by degree (row normalization).
    /// </summary>
    private Tensor<T> NormalizeAdjacency(Tensor<T> adjacency, int batchSize, int numNodes)
    {
        // Handle both 2D [N, N] and 3D [B, N, N] adjacency matrices (2D is tiled over the batch).
        bool is2D = adjacency.Shape.Length == 2;
        int[] adjacencyShape = is2D
            ? [batchSize > 1 ? batchSize : 1, adjacency.Shape[0], adjacency.Shape[1]]
            : adjacency.Shape.ToArray();
        if (adjacencyShape.Length != 3)
            throw new ArgumentOutOfRangeException(nameof(adjacency), "Adjacency matrix must be rank 2 or 3.");
        if ((batchSize > 0 && numNodes > 0)
            && (adjacencyShape[0] < batchSize || adjacencyShape[1] < numNodes || adjacencyShape[2] < numNodes))
            throw new ArgumentOutOfRangeException(nameof(adjacency),
                $"Adjacency matrix of shape [{string.Join(", ", adjacencyShape)}] does not cover {batchSize} samples of {numNodes} nodes.");
        if (batchSize == 0 || numNodes == 0)
            return new Tensor<T>(adjacencyShape);

        // Row normalisation by degree (the count of non-zero entries; NaN counts), zero rows stay zero.
        // Entries outside [batchSize, numNodes, numNodes] stay zero. A plain value, off the tape.
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            var adj3D = is2D
                ? Engine.TensorBroadcastTo(
                    Engine.Reshape(adjacency, [1, adjacency.Shape[0], adjacency.Shape[1]]), adjacencyShape)
                : adjacency;
            var core = Engine.TensorSlice(adj3D, [0, 0, 0], [batchSize, numNodes, numNodes]);
            var degree = Engine.ReduceSum(Engine.TensorNotEquals(core, NumOps.Zero), new[] { 2 }, keepDims: true);
            var isolated = Engine.TensorEquals(degree, NumOps.Zero);
            var ones = new Tensor<T>([batchSize, numNodes, 1]);
            ones.Fill(NumOps.One);
            var normalization = Engine.TensorDivide(ones, Engine.TensorWhere(isolated, ones, degree));
            var normalized = Engine.TensorWhere(
                Engine.TensorBroadcastTo(isolated, [batchSize, numNodes, numNodes]),
                new Tensor<T>([batchSize, numNodes, numNodes]),
                Engine.TensorMultiply(core, normalization));

            if (adjacencyShape[2] > numNodes)
                normalized = Engine.TensorConcatenate(
                    new[] { normalized, new Tensor<T>([batchSize, numNodes, adjacencyShape[2] - numNodes]) }, 2);
            if (adjacencyShape[1] > numNodes)
                normalized = Engine.TensorConcatenate(
                    new[] { normalized, new Tensor<T>([batchSize, adjacencyShape[1] - numNodes, adjacencyShape[2]]) }, 1);
            if (adjacencyShape[0] > batchSize)
                normalized = Engine.TensorConcatenate(
                    new[] { normalized, new Tensor<T>([adjacencyShape[0] - batchSize, adjacencyShape[1], adjacencyShape[2]]) }, 0);
            return normalized.Clone();
        }
    }

    /// <summary>
    /// Extracts input for a specific node across all batches.
    /// </summary>
    private Tensor<T> ExtractNodeInput(Tensor<T> input, int batchSize, int nodeIndex, int inFeatures)
    {
        // Node nodeIndex's first min(inFeatures, available) features per sample, zero-padded; a plain copy.
        int kept = Math.Min(inFeatures, input.Shape[2]);
        if (batchSize == 0 || kept <= 0) return new Tensor<T>([batchSize, inFeatures]);

        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            var head = Engine.Reshape(
                Engine.TensorSlice(input, [0, nodeIndex, 0], [batchSize, 1, kept]), [batchSize, kept]);
            return kept == inFeatures
                ? head.Clone()
                : Engine.TensorConcatenate(new[] { head, new Tensor<T>([batchSize, inFeatures - kept]) }, 1);
        }
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
        if (batchSize == 0 || outputFeatures == 0) return output;

        // output[:, nodeIndex, :] += nodeOutput + bias, written in place (not recorded on the tape),
        // as the element-wise update was.
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            var value = Engine.TensorAdd(
                Engine.TensorSlice(nodeOutput, [0, 0], [batchSize, outputFeatures]),
                Engine.TensorSlice(bias, [0, 0], [batchSize, outputFeatures]));
            var current = Engine.Reshape(
                Engine.TensorSlice(output, [0, nodeIndex, 0], [batchSize, 1, outputFeatures]), [batchSize, outputFeatures]);
            output.SetSlice(1, nodeIndex, Engine.TensorAdd(current, value));
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

        if (rows == 0 || cols == 0) return new Tensor<T>([rows, cols]);

        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            return Engine.Reshape(
                Engine.TensorSlice(tensor, [batchIndex, 0, 0], [1, rows, cols]), [rows, cols]).Clone();
        }
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
                    Engine.TensorSubtractInPlace(_edgeTypeWeights[edgeType], scaledGrad);
                    Engine.InvalidatePersistentTensor(_edgeTypeWeights[edgeType]);
                }
            }
        }
        else if (_basisMatricesGradient != null && _basisCoefficientsGradients != null &&
                 _basisMatrices != null && _basisCoefficients != null)
        {
            // Update basis matrices
            var scaledBasisGrad = Engine.TensorMultiplyScalar(_basisMatricesGradient, learningRate);
            Engine.TensorSubtractInPlace(_basisMatrices, scaledBasisGrad);
            Engine.InvalidatePersistentTensor(_basisMatrices);

            // Update basis coefficients
            foreach (var kvp in _basisCoefficients)
            {
                string edgeType = kvp.Key;
                if (_basisCoefficientsGradients.TryGetValue(edgeType, out var gradient))
                {
                    var scaledGrad = Engine.TensorMultiplyScalar(gradient, learningRate);
                    Engine.TensorSubtractInPlace(_basisCoefficients[edgeType], scaledGrad);
                    Engine.InvalidatePersistentTensor(_basisCoefficients[edgeType]);
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
                Engine.TensorSubtractInPlace(_selfLoopWeights[nodeType], scaledGrad);
                Engine.InvalidatePersistentTensor(_selfLoopWeights[nodeType]);
            }
        }

        // Update biases
        foreach (var kvp in _biases)
        {
            string nodeType = kvp.Key;
            if (_biasesGradients.TryGetValue(nodeType, out var gradient))
            {
                var scaledGrad = Engine.TensorMultiplyScalar(gradient, learningRate);
                Engine.TensorSubtractInPlace(_biases[nodeType], scaledGrad);
                Engine.InvalidatePersistentTensor(_biases[nodeType]);
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

    /// <inheritdoc />
    /// <remarks>
    /// Relation- and node-specific tensors live in dictionaries, so their complete surface cannot
    /// be expressed by one attribute per field. Publishing the live ordered manifest here lets the
    /// base checkpoint and copy-on-write machinery validate and adopt the entire runtime registry.
    /// </remarks>
    protected override IReadOnlyList<(Tensor<T>? Tensor, TensorShape Expected, PersistentTensorRole Role)>
        DeclaredParameterShapes()
    {
        var tensors = GetParameterTensors();
        int biasStart = tensors.Count - _biases.Count;
        var declared = new (Tensor<T>?, TensorShape, PersistentTensorRole)[tensors.Count];
        for (int i = 0; i < tensors.Count; i++)
        {
            var tensor = tensors[i];
            declared[i] = (
                tensor,
                ShapeOf(tensor.Shape.ToArray()),
                i >= biasStart ? PersistentTensorRole.Biases : PersistentTensorRole.Weights);
        }
        return declared;
    }

    /// <inheritdoc />
    protected override void AdoptTrainableParameterTensors(IReadOnlyList<Tensor<T>> parameters)
    {
        SetParameterTensors(parameters.ToList());
    }

    /// <summary>
    /// Sets the trainable parameters of the layer from a list of tensors.
    /// </summary>
    /// <param name="parameters">A list containing all parameter tensors to set.</param>
    public void SetParameterTensors(List<Tensor<T>> parameters)
    {
        int expected = GetParameterTensors().Count;
        if (parameters.Count != expected)
        {
            throw new ArgumentException(
                $"{nameof(HeterogeneousGraphLayer<T>)} requires exactly {expected} parameter tensors, " +
                $"but received {parameters.Count}.",
                nameof(parameters));
        }

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
    public override void ResetState()
    {
        _edgeTypeWeightsGradients = null;
        _selfLoopWeightsGradients = null;
        _biasesGradients = null;
        _basisMatricesGradient = null;
        _basisCoefficientsGradients = null;
    }
}
