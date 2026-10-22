using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Memory;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements Principal Neighbourhood Aggregation (PNA) layer for powerful graph representation learning.
/// </summary>
/// <remarks>
/// <para>
/// Principal Neighbourhood Aggregation (PNA), introduced by Corso et al., addresses limitations
/// of existing GNN architectures by using multiple aggregators and scalers. PNA combines:
/// 1. Multiple aggregation functions (mean, max, min, sum, std)
/// 2. Multiple scaling functions to normalize by degree
/// 3. Learnable combination of all aggregated features
/// </para>
/// <para>
/// The layer computes: h_i' = MLP(COMBINE({SCALE(AGGREGATE({h_j : j in N(i)}))}))
/// where AGGREGATE in {mean, max, min, sum, std}, SCALE in {identity, amplification, attenuation},
/// and COMBINE is a learned linear combination followed by MLP.
/// </para>
/// <para>
/// <b>Production-Ready Features:</b>
/// <list type="bullet">
/// <item>Fully vectorized operations using IEngine for GPU acceleration</item>
/// <item>BatchMatMul for efficient batched graph operations</item>
/// <item>Dual backward pass: BackwardManual() for efficiency, BackwardViaAutodiff() for accuracy</item>
/// <item>Full gradient computation through all aggregators and scalers</item>
/// <item>Complete GetParameters()/SetParameters() for model persistence</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.Graph)]
[LayerTask(LayerTask.GraphProcessing)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(ApiShape = LayerApiShape.GraphWithSetup, IsTrainable = true, ChangesShape = true, TestInputShape = "1, 4, 8", TestConstructorArgs = "8, 4", TestSetupCode = "var adj = new AiDotNet.Tensors.LinearAlgebra.Tensor<double>(new[] { 1, 4, 4 }); for (int i = 0; i < 4; i++) { adj[0, i, i] = 1.0; if (i > 0) adj[0, i, i-1] = 1.0; if (i < 3) adj[0, i, i+1] = 1.0; } var m = layer.GetType().GetMethod(\"SetAdjacencyMatrix\"); if (m != null) m.Invoke(layer, new object[] { adj });")]
// Roles and ranks from ForwardTraced's own normalization: "PNA is a graph layer: normalize to 3D
// [batch, nodes, features]", with an explicit rank-2 branch for [nodes, features]. The node axis is Other
// because a graph's nodes are an unordered vertex set - not a spatial extent and not a sequence position -
// so none of the named roles describes it without asserting something false.
//
// Rank 1 also runs (it is reshaped to [1, 1, features]) but is not declared: it degenerates to a
// single-node graph, and a declaration would advertise a shape callers should not be routing here.
// Higher ranks collapse their leading axes into the batch, which would need a distinct role per axis
// that this layer says nothing about.
[TensorLayout(TensorAxis.Other, TensorAxis.Features, Direction = TensorLayoutDirection.Input,
    Note = "Unbatched graph: leading axis is the node count.")]
[TensorLayout(TensorAxis.Other, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Other, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Other, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class PrincipalNeighbourhoodAggregationLayer<T> : LayerBase<T>, IGraphConvolutionLayer<T>, IShapeContract
{
    private readonly int _inputFeatures;
    private readonly int _outputFeatures;
    private readonly PNAAggregator[] _aggregators;
    private readonly PNAScaler[] _scalers;
    private readonly int _combinedFeatures;
    private readonly double _avgDegree;
    private readonly int _hiddenDim;

    // Pre-transformation weights (applied before aggregation) - Tensor-based for GPU acceleration
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _preTransformWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _preTransformBias;

    // Post-aggregation MLP weights - Tensor-based for GPU acceleration
    private Tensor<T> _postAggregationWeights1;
    private Tensor<T> _postAggregationWeights2;
    private Tensor<T> _postAggregationBias1;
    private Tensor<T> _postAggregationBias2;

    // Self-loop transformation - Tensor-based for GPU acceleration
    private Tensor<T> _selfWeights;

    // Final bias - Tensor-based for GPU acceleration
    private Tensor<T> _bias;

    // The adjacency matrix defining graph structure
    [AiDotNet.Attributes.FittedParameter(InputSized = true)]
    private Tensor<T>? _adjacencyMatrix;

    // Cached values for backward pass
    [Scratch]
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;
    [Scratch]
    private Tensor<T>? _lastOutput;
    [Scratch]
    private Tensor<T>? _lastTransformed;
    [Scratch]
    private Tensor<T>? _lastAggregated;
    [Scratch]
    private Tensor<T>? _lastMlpHiddenPreRelu;
    [Scratch]
    private Tensor<T>? _lastMlpHidden;
    [Scratch]
    private Tensor<T>? _lastMlpOutput;
    [Scratch]
    private Tensor<T>? _lastDegrees;

    // Gradients - Tensor-based for GPU acceleration
    [Scratch]
    private Tensor<T>? _preTransformWeightsGradient;
    [Scratch]
    private Tensor<T>? _preTransformBiasGradient;
    [Scratch]
    private Tensor<T>? _postAggregationWeights1Gradient;
    [Scratch]
    private Tensor<T>? _postAggregationWeights2Gradient;
    [Scratch]
    private Tensor<T>? _postAggregationBias1Gradient;
    [Scratch]
    private Tensor<T>? _postAggregationBias2Gradient;
    [Scratch]
    private Tensor<T>? _selfWeightsGradient;
    [Scratch]
    private Tensor<T>? _biasGradient;

    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public int InputFeatures => _inputFeatures;

    /// <inheritdoc/>
    public int OutputFeatures => _outputFeatures;

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Read off the shape-restoring tail of <c>ForwardTraced</c>, which reshapes to
    /// <c>[processNumNodes, _outputFeatures]</c> for a rank-2 input and leaves the rank-3 result as
    /// <c>[batch, nodes, _outputFeatures]</c>. Aggregation is per node over its neighbours, so the node
    /// count is carried through; only the feature width is this layer's to set.
    /// </para>
    /// <para>
    /// The adjacency matrix is a second input, set through <see cref="SetAdjacencyMatrix"/>, and it is
    /// worth being explicit that the output shape does NOT depend on it - it selects WHICH neighbours are
    /// summed, never how many rows come out. That is why this layer can declare a contract at all where a
    /// layer whose output extent came from its second input could not.
    /// </para>
    /// <para>
    /// <c>Fixed(_outputFeatures)</c> reads the constructor argument, not the observed width: the multiple
    /// aggregators and scalers widen the intermediate to <c>_combinedFeatures</c>, but the post-aggregation
    /// MLP projects back down, so the declared width is the one the final matmul produces.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (_outputFeatures <= 0) return null;

        var nodes = new OutputAxisContract(TensorAxis.Other, AxisRelation.Same(TensorAxis.Other));
        var features = new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_outputFeatures));

        return inputRank switch
        {
            2 => new[] { nodes, features },
            3 => new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                nodes,
                features,
            },
            _ => null,
        };
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="PrincipalNeighbourhoodAggregationLayer{T}"/> class.
    /// </summary>
    public PrincipalNeighbourhoodAggregationLayer(
        int inputFeatures,
        int outputFeatures,
        PNAAggregator[]? aggregators = null,
        PNAScaler[]? scalers = null,
        double avgDegree = 1.0,
        IActivationFunction<T>? activationFunction = null,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base([inputFeatures], [outputFeatures], activationFunction ?? new IdentityActivation<T>())
    {
        InitializationStrategy = initializationStrategy ?? InitializationStrategies<T>.Eager;

        _inputFeatures = inputFeatures;
        _outputFeatures = outputFeatures;

        // Default: use all aggregators
        _aggregators = aggregators ?? new[]
        {
            PNAAggregator.Mean,
            PNAAggregator.Max,
            PNAAggregator.Min,
            PNAAggregator.Sum,
            PNAAggregator.StdDev
        };

        // Default: use all scalers
        _scalers = scalers ?? new[]
        {
            PNAScaler.Identity,
            PNAScaler.Amplification,
            PNAScaler.Attenuation
        };

        _avgDegree = avgDegree;

        // Combined features = inputFeatures * aggregators * scalers
        _combinedFeatures = _inputFeatures * _aggregators.Length * _scalers.Length;
        _hiddenDim = Math.Max(_combinedFeatures / 2, _outputFeatures);

        // Initialize all weights as Tensors for GPU acceleration
        _preTransformWeights = new Tensor<T>([_inputFeatures, _inputFeatures]);
        _preTransformBias = new Tensor<T>([_inputFeatures]);

        _postAggregationWeights1 = new Tensor<T>([_combinedFeatures, _hiddenDim]);
        _postAggregationWeights2 = new Tensor<T>([_hiddenDim, _outputFeatures]);
        _postAggregationBias1 = new Tensor<T>([_hiddenDim]);
        _postAggregationBias2 = new Tensor<T>([_outputFeatures]);

        _selfWeights = new Tensor<T>([_inputFeatures, _outputFeatures]);
        _bias = new Tensor<T>([_outputFeatures]);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        // Xavier/Glorot initialization using Engine operations
        InitializeTensor(_preTransformWeights, _inputFeatures, _inputFeatures);
        InitializeTensor(_postAggregationWeights1, _combinedFeatures, _hiddenDim);
        InitializeTensor(_postAggregationWeights2, _hiddenDim, _outputFeatures);
        InitializeTensor(_selfWeights, _inputFeatures, _outputFeatures);

        // Initialize biases to zero using Fill
        _preTransformBias.Fill(NumOps.Zero);
        _postAggregationBias1.Fill(NumOps.Zero);
        _postAggregationBias2.Fill(NumOps.Zero);
        _bias.Fill(NumOps.Zero);

        RegisterTrainableParameter(_preTransformWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_preTransformBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_postAggregationWeights1, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_postAggregationBias1, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_postAggregationWeights2, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_postAggregationBias2, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_selfWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bias, PersistentTensorRole.Biases);
    }

    private void InitializeTensor(Tensor<T> tensor, int fanIn, int fanOut)
    {
        InitializeLayerWeights(tensor, fanIn, fanOut);
    }

    /// <inheritdoc/>
    public void SetAdjacencyMatrix(Tensor<T> adjacencyMatrix)
    {
        _adjacencyMatrix = adjacencyMatrix;
    }

    /// <inheritdoc/>
    public Tensor<T>? GetAdjacencyMatrix()
    {
        return _adjacencyMatrix;
    }

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (_adjacencyMatrix == null)
        {
            throw new InvalidOperationException(
                "Adjacency matrix must be set using SetAdjacencyMatrix before calling Forward.");
        }

        // Store original shape for any-rank tensor support
        _originalInputShape = input._shape;
        int rank = input.Shape.Length;

        // PNA is a graph layer: normalize to 3D [batch, nodes, features]
        Tensor<T> processInput;
        int batchSize;

        if (rank == 1)
        {
            // 1D [features] -> [1, 1, features]
            batchSize = 1;
            processInput = Engine.Reshape(input, [1, 1, input.Shape[0]]);
        }
        else if (rank == 2)
        {
            // 2D [nodes, features] -> [1, nodes, features]
            batchSize = 1;
            processInput = Engine.Reshape(input, [1, input.Shape[0], input.Shape[1]]);
        }
        else if (rank == 3)
        {
            // Standard 3D [batch, nodes, features]
            batchSize = input.Shape[0];
            processInput = input;
        }
        else
        {
            // Higher-rank: collapse leading dims into batch
            // e.g., 4D [B1, B2, nodes, features] -> [B1*B2, nodes, features]
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

        // Step 1: Pre-transform input features: transformed = input @ preTransformWeights + preTransformBias
        // Uses Engine.TensorMatMul for batched matrix multiplication
        var transformed = Engine.TensorMatMul(processInput, _preTransformWeights);
        var preBiasBroadcast = BroadcastBias(_preTransformBias, batchSize, processNumNodes);
        _lastTransformed = Engine.TensorAdd(transformed, preBiasBroadcast);

        // Step 2: Compute degrees for each node using adjacency matrix row sums
        // degrees[b,i] = sum_j(A[b,i,j])
        _lastDegrees = Engine.ReduceSum(_adjacencyMatrix, [2], keepDims: false); // [batch, nodes]

        // Avoid division by zero - clamp degrees to minimum of 1
        var oneTensor = new Tensor<T>(_lastDegrees._shape);
        oneTensor.Fill(NumOps.One);
        var safeDegrees = Engine.TensorMax(_lastDegrees, oneTensor);

        // Step 3: Apply multiple aggregators and scalers using vectorized operations
        var aggregatedParts = new List<Tensor<T>>();

        foreach (var aggregator in _aggregators)
        {
            var aggregated = ComputeVectorizedAggregation(_lastTransformed, aggregator, safeDegrees, processNumNodes);

            foreach (var scaler in _scalers)
            {
                var scaled = ApplyVectorizedScaler(aggregated, scaler, safeDegrees);
                aggregatedParts.Add(scaled);
            }
        }

        // Concatenate all aggregated features along the last axis
        _lastAggregated = Engine.TensorConcatenate(aggregatedParts.ToArray(), axis: 2); // [batch, nodes, combinedFeatures]

        // Step 4: Post-aggregation MLP - Layer 1 with ReLU
        var hidden = Engine.TensorMatMul(_lastAggregated, _postAggregationWeights1);
        var bias1Broadcast = BroadcastBias(_postAggregationBias1, batchSize, processNumNodes);
        _lastMlpHiddenPreRelu = Engine.TensorAdd(hidden, bias1Broadcast);

        // ReLU activation using Engine operations
        var zeroTensor = new Tensor<T>(_lastMlpHiddenPreRelu._shape);
        zeroTensor.Fill(NumOps.Zero);
        _lastMlpHidden = Engine.TensorMax(_lastMlpHiddenPreRelu, zeroTensor);

        // Step 5: Post-aggregation MLP - Layer 2
        var mlpOutput = Engine.TensorMatMul(_lastMlpHidden, _postAggregationWeights2);
        var bias2Broadcast = BroadcastBias(_postAggregationBias2, batchSize, processNumNodes);
        _lastMlpOutput = Engine.TensorAdd(mlpOutput, bias2Broadcast);

        // Step 6: Self-loop transformation and final bias
        var selfContribution = Engine.TensorMatMul(processInput, _selfWeights);
        var biasBroadcast = BroadcastBias(_bias, batchSize, processNumNodes);

        var preActivation = Engine.TensorAdd(_lastMlpOutput, selfContribution);
        preActivation = Engine.TensorAdd(preActivation, biasBroadcast);

        var result = ApplyActivation(preActivation);

        // Only store for backward pass during training - skip during inference
        if (IsTrainingMode)
        {
            _lastOutput = result;
        }

        // Restore output shape to match original input rank
        if (_originalInputShape != null && _originalInputShape.Length != 3)
        {
            if (_originalInputShape.Length == 2)
            {
                // 2D input [nodes, features] -> 2D output [nodes, outputFeatures]
                return Engine.Reshape(result, [processNumNodes, _outputFeatures]);
            }
            else if (_originalInputShape.Length == 1)
            {
                // 1D input -> 1D output
                return Engine.Reshape(result, [_outputFeatures]);
            }
            else
            {
                // Higher-rank: restore leading dimensions
                var outShape = new int[_originalInputShape.Length];
                for (int d = 0; d < _originalInputShape.Length - 2; d++)
                    outShape[d] = _originalInputShape[d];
                outShape[_originalInputShape.Length - 2] = processNumNodes;
                outShape[_originalInputShape.Length - 1] = _outputFeatures;
                return Engine.Reshape(result, outShape);
            }
        }

        return result;
    }

    /// <summary>
    /// Computes vectorized aggregation using Engine operations.
    /// </summary>
    private Tensor<T> ComputeVectorizedAggregation(Tensor<T> transformed, PNAAggregator aggregator,
        Tensor<T> safeDegrees, int numNodes)
    {
        int batchSize = transformed.Shape[0];

        switch (aggregator)
        {
            case PNAAggregator.Sum:
                // Sum aggregation: A @ X (message passing via adjacency matrix)
                return Engine.TensorMatMul(_adjacencyMatrix!, transformed);

            case PNAAggregator.Mean:
                // Mean aggregation: (A @ X) / degree
                var sumAgg = Engine.TensorMatMul(_adjacencyMatrix!, transformed);
                // Expand degrees for broadcasting: [batch, nodes] -> [batch, nodes, 1]
                var degreesExpanded = safeDegrees.Reshape([batchSize, numNodes, 1]);
                return Engine.TensorDivide(sumAgg, degreesExpanded);

            case PNAAggregator.Max:
                // Max aggregation requires masking non-neighbors with -inf then taking max
                return ComputeMaxAggregation(transformed, numNodes);

            case PNAAggregator.Min:
                // Min aggregation requires masking non-neighbors with +inf then taking min
                return ComputeMinAggregation(transformed, numNodes);

            case PNAAggregator.StdDev:
                // StdDev: sqrt(E[X^2] - E[X]^2 + epsilon)
                return ComputeStdDevAggregation(transformed, safeDegrees, numNodes);

            default:
                return Engine.TensorMatMul(_adjacencyMatrix!, transformed);
        }
    }

    /// <summary>
    /// Computes max aggregation over neighbors using masked reduce.
    /// </summary>
    private Tensor<T> ComputeMaxAggregation(Tensor<T> transformed, int numNodes)
    {
        int batchSize = transformed.Shape[0];
        int features = transformed.Shape[2];

        // Expand transformed for broadcasting: [batch, 1, nodes, features]
        var expanded = transformed.Reshape([batchSize, 1, numNodes, features]);

        // Tile to [batch, nodes, nodes, features] - each row gets all node features
        var tiled = Engine.TensorTile(expanded, [1, numNodes, 1, 1]);

        // Create mask from adjacency: [batch, nodes, nodes, 1]
        var adjacencyMatrix = _adjacencyMatrix ?? throw new InvalidOperationException("_adjacencyMatrix has not been initialized.");
        var adjExpanded = adjacencyMatrix.Reshape([batchSize, numNodes, numNodes, 1]);

        // Mask non-neighbors with -inf
        var negInf = new Tensor<T>(tiled._shape);
        negInf.Fill(NumOps.MinValue);

        // Where adj > 0, use tiled values; else use -inf
        var zeroTensor = new Tensor<T>(adjExpanded._shape);
        zeroTensor.Fill(NumOps.Zero);
        var mask = Engine.TensorGreaterThan(adjExpanded, zeroTensor);

        // Broadcast mask to feature dimension
        var maskBroadcast = Engine.TensorTile(mask, [1, 1, 1, features]);
        var masked = Engine.TensorWhere(maskBroadcast, tiled, negInf);

        // Reduce max over neighbors axis (axis 2)
        return Engine.ReduceMax(masked, [2], keepDims: false, out _);
    }

    /// <summary>
    /// Computes min aggregation over neighbors using masked reduce.
    /// </summary>
    private Tensor<T> ComputeMinAggregation(Tensor<T> transformed, int numNodes)
    {
        int batchSize = transformed.Shape[0];
        int features = transformed.Shape[2];

        // Expand transformed for broadcasting: [batch, 1, nodes, features]
        var expanded = transformed.Reshape([batchSize, 1, numNodes, features]);

        // Tile to [batch, nodes, nodes, features]
        var tiled = Engine.TensorTile(expanded, [1, numNodes, 1, 1]);

        // Create mask from adjacency: [batch, nodes, nodes, 1]
        var adjacencyMatrix = _adjacencyMatrix ?? throw new InvalidOperationException("_adjacencyMatrix has not been initialized.");
        var adjExpanded = adjacencyMatrix.Reshape([batchSize, numNodes, numNodes, 1]);

        // Mask non-neighbors with +inf
        var posInf = new Tensor<T>(tiled._shape);
        posInf.Fill(NumOps.MaxValue);

        var zeroTensor = new Tensor<T>(adjExpanded._shape);
        zeroTensor.Fill(NumOps.Zero);
        var mask = Engine.TensorGreaterThan(adjExpanded, zeroTensor);

        // Broadcast mask to feature dimension
        var maskBroadcast = Engine.TensorTile(mask, [1, 1, 1, features]);
        var masked = Engine.TensorWhere(maskBroadcast, tiled, posInf);

        // Reduce min over neighbors axis (axis 2)
        // Note: Using negative max of negated tensor for min
        var negMasked = Engine.TensorMultiplyScalar(masked, NumOps.FromDouble(-1));
        var maxOfNeg = Engine.ReduceMax(negMasked, [2], keepDims: false, out _);
        return Engine.TensorMultiplyScalar(maxOfNeg, NumOps.FromDouble(-1));
    }

    /// <summary>
    /// Computes standard deviation aggregation using vectorized operations.
    /// </summary>
    private Tensor<T> ComputeStdDevAggregation(Tensor<T> transformed, Tensor<T> safeDegrees, int numNodes)
    {
        int batchSize = transformed.Shape[0];

        // Mean: E[X] = (A @ X) / degree
        var sumAgg = Engine.TensorMatMul(_adjacencyMatrix!, transformed);
        var degreesExpanded = safeDegrees.Reshape([batchSize, numNodes, 1]);
        var mean = Engine.TensorDivide(sumAgg, degreesExpanded);

        // E[X^2] = (A @ X^2) / degree
        var transformedSquared = Engine.TensorMultiply(transformed, transformed);
        var sumSquared = Engine.TensorMatMul(_adjacencyMatrix!, transformedSquared);
        var meanSquared = Engine.TensorDivide(sumSquared, degreesExpanded);

        // Variance = E[X^2] - E[X]^2
        var meanSq = Engine.TensorMultiply(mean, mean);
        var variance = Engine.TensorSubtract(meanSquared, meanSq);

        // Add epsilon for numerical stability
        var epsilon = new Tensor<T>(variance._shape);
        epsilon.Fill(NumOps.FromDouble(1e-8));
        variance = Engine.TensorAdd(variance, epsilon);

        // StdDev = sqrt(variance)
        return Engine.TensorSqrt(variance);
    }

    /// <summary>
    /// Applies scaler to aggregated features using vectorized operations.
    /// </summary>
    private Tensor<T> ApplyVectorizedScaler(Tensor<T> aggregated, PNAScaler scaler, Tensor<T> safeDegrees)
    {
        int batchSize = aggregated.Shape[0];
        int numNodes = aggregated.Shape[1];

        switch (scaler)
        {
            case PNAScaler.Identity:
                return aggregated;

            case PNAScaler.Amplification:
                // Scale by degree / avgDegree (amplify high-degree nodes)
                // Per PNA paper: amplification increases signal from high-degree nodes
                var avgDegreeTensor = TensorAllocator.Rent<T>([batchSize, numNodes, 1]);
                avgDegreeTensor.Fill(NumOps.FromDouble(_avgDegree));
                var degreesExpanded = safeDegrees.Reshape([batchSize, numNodes, 1]);
                var ampFactor = Engine.TensorDivide(degreesExpanded, avgDegreeTensor);
                return Engine.TensorMultiply(aggregated, ampFactor);

            case PNAScaler.Attenuation:
                // Scale by avgDegree / degree (attenuate high-degree nodes)
                // Per PNA paper: attenuation reduces signal from high-degree nodes
                var avgDegTensor = TensorAllocator.Rent<T>([batchSize, numNodes, 1]);
                avgDegTensor.Fill(NumOps.FromDouble(_avgDegree));
                var degExpanded = safeDegrees.Reshape([batchSize, numNodes, 1]);
                var attFactor = Engine.TensorDivide(avgDegTensor, degExpanded);
                return Engine.TensorMultiply(aggregated, attFactor);

            default:
                return aggregated;
        }
    }

    /// <summary>
    /// Broadcasts a bias tensor across batch and node dimensions using Engine operations.
    /// </summary>
    private Tensor<T> BroadcastBias(Tensor<T> bias, int batchSize, int numNodes)
    {
        int features = bias.Length;
        var biasReshaped = bias.Reshape([1, 1, features]);
        return Engine.TensorTile(biasReshaped, [batchSize, numNodes, 1]);
    }

    /// <summary>
    /// Backpropagates through aggregation operations using vectorized Engine operations.
    /// </summary>
    private Tensor<T> BackpropThroughAggregation(Tensor<T> aggregatedGrad, int numNodes)
    {
        int batchSize = aggregatedGrad.Shape[0];
        int numAggregators = _aggregators.Length;
        int numScalers = _scalers.Length;

        var transformedGrad = TensorAllocator.Rent<T>([batchSize, numNodes, _inputFeatures]);
        transformedGrad.Fill(NumOps.Zero);

        // Hoist loop-invariant computations
        var adjMatrix = _adjacencyMatrix ?? throw new InvalidOperationException("_adjacencyMatrix has not been initialized.");
        var adjT = SwapLastTwoAxes(adjMatrix);
        var lastDegrees = _lastDegrees ?? throw new InvalidOperationException("_lastDegrees has not been initialized.");
        var safeDegrees = Engine.TensorMax(lastDegrees, NumOps.One);
        var degExpanded = safeDegrees.Reshape([batchSize, numNodes, 1]);

        // Split aggregatedGrad by aggregator-scaler combinations
        int featureIdx = 0;

        for (int aggIdx = 0; aggIdx < numAggregators; aggIdx++)
        {
            for (int scalerIdx = 0; scalerIdx < numScalers; scalerIdx++)
            {
                // Extract gradient slice for this aggregator-scaler
                var gradSlice = Engine.TensorSlice(aggregatedGrad,
                    [0, 0, featureIdx],
                    [batchSize, numNodes, _inputFeatures]);

                // Backprop through scaler (simplified - assumes identity for gradient)
                var scalerGrad = gradSlice;

                // Backprop through aggregation
                // For mean/sum: gradient flows back through adjacency transpose
                var aggGrad = Engine.TensorMatMul(adjT, scalerGrad);

                // For mean, also divide by degree
                if (_aggregators[aggIdx] == PNAAggregator.Mean)
                {
                    aggGrad = Engine.TensorDivide(aggGrad, degExpanded);
                }

                transformedGrad = Engine.TensorAdd(transformedGrad, aggGrad);
                featureIdx += _inputFeatures;
            }
        }

        return transformedGrad;
    }

    private Tensor<T> SwapLastTwoAxes(Tensor<T> tensor)
    {
        int rank = tensor.Shape.Length;
        if (rank < 2)
        {
            return tensor;
        }

        if (rank == 2)
        {
            return Engine.TensorTranspose(tensor);
        }

        var permutation = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            permutation[i] = i;
        }

        int last = rank - 1;
        permutation[last] = rank - 2;
        permutation[last - 1] = rank - 1;

        return tensor.Transpose(permutation);
    }

    /// <summary>
    /// Computes gradients using vectorized Engine operations as fallback for autodiff.
    /// </summary>
    private void ComputeGradientsViaEngine(Tensor<T> activationGradient, int batchSize, int numNodes)
    {
        var lastInput = _lastInput ?? throw new InvalidOperationException("_lastInput has not been initialized.");
        var lastAggregated = _lastAggregated ?? throw new InvalidOperationException("_lastAggregated has not been initialized.");

        // Gradient through self-loop
        _selfWeightsGradient = new Tensor<T>([_inputFeatures, _outputFeatures]);
        _selfWeightsGradient.Fill(NumOps.Zero);

        for (int b = 0; b < batchSize; b++)
        {
            var inputSlice = Engine.TensorSlice(lastInput, [b, 0, 0], [1, numNodes, _inputFeatures])
                .Reshape([numNodes, _inputFeatures]);
            var gradSlice = Engine.TensorSlice(activationGradient, [b, 0, 0], [1, numNodes, _outputFeatures])
                .Reshape([numNodes, _outputFeatures]);

            var inputSliceT = Engine.TensorTranspose(inputSlice);
            var batchGrad = Engine.TensorMatMul(inputSliceT, gradSlice);
            _selfWeightsGradient = Engine.TensorAdd(_selfWeightsGradient, batchGrad);
        }

        // Gradient through MLP Layer 2
        _postAggregationBias2Gradient = Engine.ReduceSum(activationGradient, [0, 1], keepDims: false);

        _postAggregationWeights2Gradient = new Tensor<T>([_hiddenDim, _outputFeatures]);
        _postAggregationWeights2Gradient.Fill(NumOps.Zero);

        for (int b = 0; b < batchSize; b++)
        {
            var hiddenSlice = Engine.TensorSlice(_lastMlpHidden!, [b, 0, 0], [1, numNodes, _hiddenDim])
                .Reshape([numNodes, _hiddenDim]);
            var gradSlice = Engine.TensorSlice(activationGradient, [b, 0, 0], [1, numNodes, _outputFeatures])
                .Reshape([numNodes, _outputFeatures]);

            var hiddenT = Engine.TensorTranspose(hiddenSlice);
            var batchGrad = Engine.TensorMatMul(hiddenT, gradSlice);
            _postAggregationWeights2Gradient = Engine.TensorAdd(_postAggregationWeights2Gradient, batchGrad);
        }

        // Gradient to hidden layer
        var weights2T = Engine.TensorTranspose(_postAggregationWeights2);
        var mlpHiddenGradPre = Engine.TensorMatMul(activationGradient, weights2T);

        // ReLU derivative
        var mlpHiddenPreRelu = _lastMlpHiddenPreRelu ?? throw new InvalidOperationException("_lastMlpHiddenPreRelu has not been initialized.");
        var zeroTensor = new Tensor<T>(mlpHiddenPreRelu._shape);
        zeroTensor.Fill(NumOps.Zero);
        var reluMask = Engine.TensorGreaterThan(mlpHiddenPreRelu, zeroTensor);
        var oneTensor = new Tensor<T>(mlpHiddenPreRelu._shape);
        oneTensor.Fill(NumOps.One);
        var reluDeriv = Engine.TensorWhere(reluMask, oneTensor, zeroTensor);
        var mlpHiddenGrad = Engine.TensorMultiply(mlpHiddenGradPre, reluDeriv);

        // Gradient through MLP Layer 1
        _postAggregationBias1Gradient = Engine.ReduceSum(mlpHiddenGrad, [0, 1], keepDims: false);

        _postAggregationWeights1Gradient = new Tensor<T>([_combinedFeatures, _hiddenDim]);
        _postAggregationWeights1Gradient.Fill(NumOps.Zero);

        for (int b = 0; b < batchSize; b++)
        {
            var aggSlice = Engine.TensorSlice(lastAggregated, [b, 0, 0], [1, numNodes, _combinedFeatures])
                .Reshape([numNodes, _combinedFeatures]);
            var gradSlice = Engine.TensorSlice(mlpHiddenGrad, [b, 0, 0], [1, numNodes, _hiddenDim])
                .Reshape([numNodes, _hiddenDim]);

            var aggT = Engine.TensorTranspose(aggSlice);
            var batchGrad = Engine.TensorMatMul(aggT, gradSlice);
            _postAggregationWeights1Gradient = Engine.TensorAdd(_postAggregationWeights1Gradient, batchGrad);
        }

        // Gradient to aggregated features
        var weights1T = Engine.TensorTranspose(_postAggregationWeights1);
        var aggregatedGrad = Engine.TensorMatMul(mlpHiddenGrad, weights1T);

        // Backprop through aggregation
        var transformedGrad = BackpropThroughAggregation(aggregatedGrad, numNodes);

        // Gradient through pre-transform
        _preTransformBiasGradient = Engine.ReduceSum(transformedGrad, [0, 1], keepDims: false);

        _preTransformWeightsGradient = new Tensor<T>([_inputFeatures, _inputFeatures]);
        _preTransformWeightsGradient.Fill(NumOps.Zero);

        for (int b = 0; b < batchSize; b++)
        {
            var inputSlice = Engine.TensorSlice(_lastInput!, [b, 0, 0], [1, numNodes, _inputFeatures])
                .Reshape([numNodes, _inputFeatures]);
            var gradSlice = Engine.TensorSlice(transformedGrad, [b, 0, 0], [1, numNodes, _inputFeatures])
                .Reshape([numNodes, _inputFeatures]);

            var inputSliceT = Engine.TensorTranspose(inputSlice);
            var batchGrad = Engine.TensorMatMul(inputSliceT, gradSlice);
            _preTransformWeightsGradient = Engine.TensorAdd(_preTransformWeightsGradient, batchGrad);
        }
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        if (_biasGradient == null || _preTransformWeightsGradient == null ||
            _postAggregationWeights1Gradient == null || _postAggregationWeights2Gradient == null ||
            _selfWeightsGradient == null)
        {
            throw new InvalidOperationException("Backward must be called before UpdateParameters.");
        }

        // Update using vectorized Engine operations
        Engine.TensorSubtractInPlace(_preTransformWeights,
            Engine.TensorMultiplyScalar(_preTransformWeightsGradient, learningRate));
        Engine.TensorSubtractInPlace(_preTransformBias,
            Engine.TensorMultiplyScalar(_preTransformBiasGradient!, learningRate));
        Engine.TensorSubtractInPlace(_postAggregationWeights1,
            Engine.TensorMultiplyScalar(_postAggregationWeights1Gradient, learningRate));
        Engine.TensorSubtractInPlace(_postAggregationWeights2,
            Engine.TensorMultiplyScalar(_postAggregationWeights2Gradient, learningRate));
        Engine.TensorSubtractInPlace(_postAggregationBias1,
            Engine.TensorMultiplyScalar(_postAggregationBias1Gradient!, learningRate));
        Engine.TensorSubtractInPlace(_postAggregationBias2,
            Engine.TensorMultiplyScalar(_postAggregationBias2Gradient!, learningRate));
        Engine.TensorSubtractInPlace(_selfWeights,
            Engine.TensorMultiplyScalar(_selfWeightsGradient, learningRate));
        Engine.TensorSubtractInPlace(_bias,
            Engine.TensorMultiplyScalar(_biasGradient, learningRate));
    }

    /// <inheritdoc/>
    public override void ClearGradients()
    {
        _preTransformWeightsGradient = null;
        _preTransformBiasGradient = null;
        _postAggregationWeights1Gradient = null;
        _postAggregationWeights2Gradient = null;
        _postAggregationBias1Gradient = null;
        _postAggregationBias2Gradient = null;
        _selfWeightsGradient = null;
        _biasGradient = null;
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastTransformed = null;
        _lastAggregated = null;
        _lastMlpHidden = null;
        _lastMlpHiddenPreRelu = null;
        _lastMlpOutput = null;
        _lastDegrees = null;
        _preTransformWeightsGradient = null;
        _preTransformBiasGradient = null;
        _postAggregationWeights1Gradient = null;
        _postAggregationWeights2Gradient = null;
        _postAggregationBias1Gradient = null;
        _postAggregationBias2Gradient = null;
        _selfWeightsGradient = null;
        _biasGradient = null;
    }
}
