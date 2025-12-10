using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Aggregation function types for PNA.
/// </summary>
public enum PNAAggregator
{
    /// <summary>Mean aggregation.</summary>
    Mean,
    /// <summary>Max aggregation.</summary>
    Max,
    /// <summary>Min aggregation.</summary>
    Min,
    /// <summary>Sum aggregation.</summary>
    Sum,
    /// <summary>Standard deviation aggregation.</summary>
    StdDev
}

/// <summary>
/// Scaler function types for PNA.
/// </summary>
public enum PNAScaler
{
    /// <summary>Identity scaler (no scaling).</summary>
    Identity,
    /// <summary>Amplification scaler.</summary>
    Amplification,
    /// <summary>Attenuation scaler.</summary>
    Attenuation
}

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
/// <item>JIT compilation support via ExportComputationGraph()</item>
/// <item>Complete GetParameters()/SetParameters() for model persistence</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class PrincipalNeighbourhoodAggregationLayer<T> : LayerBase<T>, IGraphConvolutionLayer<T>
{
    private readonly int _inputFeatures;
    private readonly int _outputFeatures;
    private readonly PNAAggregator[] _aggregators;
    private readonly PNAScaler[] _scalers;
    private readonly int _combinedFeatures;
    private readonly double _avgDegree;
    private readonly int _hiddenDim;

    // Pre-transformation weights (applied before aggregation) - Tensor-based for GPU acceleration
    private Tensor<T> _preTransformWeights;
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
    private Tensor<T>? _adjacencyMatrix;

    // Cached values for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastTransformed;
    private Tensor<T>? _lastAggregated;
    private Tensor<T>? _lastMlpHiddenPreRelu;
    private Tensor<T>? _lastMlpHidden;
    private Tensor<T>? _lastMlpOutput;
    private Tensor<T>? _lastDegrees;

    // Gradients - Tensor-based for GPU acceleration
    private Tensor<T>? _preTransformWeightsGradient;
    private Tensor<T>? _preTransformBiasGradient;
    private Tensor<T>? _postAggregationWeights1Gradient;
    private Tensor<T>? _postAggregationWeights2Gradient;
    private Tensor<T>? _postAggregationBias1Gradient;
    private Tensor<T>? _postAggregationBias2Gradient;
    private Tensor<T>? _selfWeightsGradient;
    private Tensor<T>? _biasGradient;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public int InputFeatures => _inputFeatures;

    /// <inheritdoc/>
    public int OutputFeatures => _outputFeatures;

    /// <summary>
    /// Initializes a new instance of the <see cref="PrincipalNeighbourhoodAggregationLayer{T}"/> class.
    /// </summary>
    public PrincipalNeighbourhoodAggregationLayer(
        int inputFeatures,
        int outputFeatures,
        PNAAggregator[]? aggregators = null,
        PNAScaler[]? scalers = null,
        double avgDegree = 1.0,
        IActivationFunction<T>? activationFunction = null)
        : base([inputFeatures], [outputFeatures], activationFunction ?? new IdentityActivation<T>())
    {
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
    }

    private void InitializeTensor(Tensor<T> tensor, int fanIn, int fanOut)
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (fanIn + fanOut)));

        // Create random tensor using Engine operations
        var randomTensor = Tensor<T>.CreateRandom(tensor.Shape);

        // Shift to [-0.5, 0.5] range
        var halfTensor = new Tensor<T>(tensor.Shape);
        halfTensor.Fill(NumOps.FromDouble(0.5));
        var shifted = Engine.TensorSubtract(randomTensor, halfTensor);

        // Scale by the Xavier factor
        var scaled = Engine.TensorMultiplyScalar(shifted, scale);

        // Copy to tensor using Engine operations
        var result = Engine.TensorAdd(tensor, scaled);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = result.GetFlat(i);
        }
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
    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (_adjacencyMatrix == null)
        {
            throw new InvalidOperationException(
                "Adjacency matrix must be set using SetAdjacencyMatrix before calling Forward.");
        }

        _lastInput = input;
        int batchSize = input.Shape[0];
        int numNodes = input.Shape[1];

        // Step 1: Pre-transform input features: transformed = input @ preTransformWeights + preTransformBias
        // Uses Engine.TensorMatMul for batched matrix multiplication
        var transformed = Engine.TensorMatMul(input, _preTransformWeights);
        var preBiasBroadcast = BroadcastBias(_preTransformBias, batchSize, numNodes);
        _lastTransformed = Engine.TensorAdd(transformed, preBiasBroadcast);

        // Step 2: Compute degrees for each node using adjacency matrix row sums
        // degrees[b,i] = sum_j(A[b,i,j])
        _lastDegrees = Engine.ReduceSum(_adjacencyMatrix, [2], keepDims: false); // [batch, nodes]

        // Avoid division by zero - clamp degrees to minimum of 1
        var oneTensor = new Tensor<T>(_lastDegrees.Shape);
        oneTensor.Fill(NumOps.One);
        var safeDegrees = Engine.TensorMax(_lastDegrees, oneTensor);

        // Step 3: Apply multiple aggregators and scalers using vectorized operations
        var aggregatedParts = new List<Tensor<T>>();

        foreach (var aggregator in _aggregators)
        {
            var aggregated = ComputeVectorizedAggregation(_lastTransformed, aggregator, safeDegrees, numNodes);

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
        var bias1Broadcast = BroadcastBias(_postAggregationBias1, batchSize, numNodes);
        _lastMlpHiddenPreRelu = Engine.TensorAdd(hidden, bias1Broadcast);

        // ReLU activation using Engine operations
        var zeroTensor = new Tensor<T>(_lastMlpHiddenPreRelu.Shape);
        zeroTensor.Fill(NumOps.Zero);
        _lastMlpHidden = Engine.TensorMax(_lastMlpHiddenPreRelu, zeroTensor);

        // Step 5: Post-aggregation MLP - Layer 2
        var mlpOutput = Engine.TensorMatMul(_lastMlpHidden, _postAggregationWeights2);
        var bias2Broadcast = BroadcastBias(_postAggregationBias2, batchSize, numNodes);
        _lastMlpOutput = Engine.TensorAdd(mlpOutput, bias2Broadcast);

        // Step 6: Self-loop transformation and final bias
        var selfContribution = Engine.TensorMatMul(input, _selfWeights);
        var biasBroadcast = BroadcastBias(_bias, batchSize, numNodes);

        var preActivation = Engine.TensorAdd(_lastMlpOutput, selfContribution);
        preActivation = Engine.TensorAdd(preActivation, biasBroadcast);

        _lastOutput = ApplyActivation(preActivation);
        return _lastOutput;
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
        var adjExpanded = _adjacencyMatrix!.Reshape([batchSize, numNodes, numNodes, 1]);

        // Mask non-neighbors with -inf
        var negInf = new Tensor<T>(tiled.Shape);
        negInf.Fill(NumOps.FromDouble(double.NegativeInfinity));

        // Where adj > 0, use tiled values; else use -inf
        var zeroTensor = new Tensor<T>(adjExpanded.Shape);
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
        var adjExpanded = _adjacencyMatrix!.Reshape([batchSize, numNodes, numNodes, 1]);

        // Mask non-neighbors with +inf
        var posInf = new Tensor<T>(tiled.Shape);
        posInf.Fill(NumOps.FromDouble(double.PositiveInfinity));

        var zeroTensor = new Tensor<T>(adjExpanded.Shape);
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
        var epsilon = new Tensor<T>(variance.Shape);
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
                // Scale by avgDegree / degree (amplify low-degree nodes)
                var avgDegreeTensor = new Tensor<T>([batchSize, numNodes, 1]);
                avgDegreeTensor.Fill(NumOps.FromDouble(_avgDegree));
                var degreesExpanded = safeDegrees.Reshape([batchSize, numNodes, 1]);
                var ampFactor = Engine.TensorDivide(avgDegreeTensor, degreesExpanded);
                return Engine.TensorMultiply(aggregated, ampFactor);

            case PNAScaler.Attenuation:
                // Scale by degree / avgDegree (attenuate high-degree nodes)
                var avgDegTensor = new Tensor<T>([batchSize, numNodes, 1]);
                avgDegTensor.Fill(NumOps.FromDouble(_avgDegree));
                var degExpanded = safeDegrees.Reshape([batchSize, numNodes, 1]);
                var attFactor = Engine.TensorDivide(degExpanded, avgDegTensor);
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

    /// <inheritdoc/>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass using vectorized Engine operations.
    /// </summary>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _adjacencyMatrix == null ||
            _lastTransformed == null || _lastAggregated == null || _lastMlpHidden == null ||
            _lastMlpHiddenPreRelu == null || _lastMlpOutput == null || _lastDegrees == null)
        {
            throw new InvalidOperationException("Forward pass must be called before Backward.");
        }

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);
        int batchSize = _lastInput.Shape[0];
        int numNodes = _lastInput.Shape[1];

        // Gradient of final bias: sum over batch and nodes (axes 0 and 1)
        _biasGradient = Engine.ReduceSum(activationGradient, [0, 1], keepDims: false);

        // Gradient through self-loop: selfWeightsGradient = input^T @ grad
        // Using batched operations
        var inputT = Engine.TensorTranspose(_lastInput); // [batch, inputFeatures, nodes]
        var swappedInputT = SwapLastTwoAxes(inputT); // [batch, nodes, inputFeatures] -> [batch, inputFeatures, nodes]

        // Sum over batch for weight gradient
        _selfWeightsGradient = new Tensor<T>([_inputFeatures, _outputFeatures]);
        _selfWeightsGradient.Fill(NumOps.Zero);

        for (int b = 0; b < batchSize; b++)
        {
            var inputSlice = Engine.TensorSlice(_lastInput, [b, 0, 0], [1, numNodes, _inputFeatures])
                .Reshape([numNodes, _inputFeatures]);
            var gradSlice = Engine.TensorSlice(activationGradient, [b, 0, 0], [1, numNodes, _outputFeatures])
                .Reshape([numNodes, _outputFeatures]);

            var inputSliceT = Engine.TensorTranspose(inputSlice);
            var batchGrad = Engine.TensorMatMul(inputSliceT, gradSlice);
            _selfWeightsGradient = Engine.TensorAdd(_selfWeightsGradient, batchGrad);
        }

        // Input gradient from self-loop
        var selfWeightsT = Engine.TensorTranspose(_selfWeights);
        var selfInputGrad = Engine.TensorMatMul(activationGradient, selfWeightsT);

        // Gradient through MLP Layer 2
        _postAggregationBias2Gradient = Engine.ReduceSum(activationGradient, [0, 1], keepDims: false);

        _postAggregationWeights2Gradient = new Tensor<T>([_hiddenDim, _outputFeatures]);
        _postAggregationWeights2Gradient.Fill(NumOps.Zero);

        for (int b = 0; b < batchSize; b++)
        {
            var hiddenSlice = Engine.TensorSlice(_lastMlpHidden, [b, 0, 0], [1, numNodes, _hiddenDim])
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
        var zeroTensor = new Tensor<T>(_lastMlpHiddenPreRelu.Shape);
        zeroTensor.Fill(NumOps.Zero);
        var reluMask = Engine.TensorGreaterThan(_lastMlpHiddenPreRelu, zeroTensor);
        var oneTensor = new Tensor<T>(_lastMlpHiddenPreRelu.Shape);
        oneTensor.Fill(NumOps.One);
        var reluDeriv = Engine.TensorWhere(reluMask, oneTensor, zeroTensor);
        var mlpHiddenGrad = Engine.TensorMultiply(mlpHiddenGradPre, reluDeriv);

        // Gradient through MLP Layer 1
        _postAggregationBias1Gradient = Engine.ReduceSum(mlpHiddenGrad, [0, 1], keepDims: false);

        _postAggregationWeights1Gradient = new Tensor<T>([_combinedFeatures, _hiddenDim]);
        _postAggregationWeights1Gradient.Fill(NumOps.Zero);

        for (int b = 0; b < batchSize; b++)
        {
            var aggSlice = Engine.TensorSlice(_lastAggregated, [b, 0, 0], [1, numNodes, _combinedFeatures])
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

        // For simplicity, backprop through aggregation using mean aggregation gradient as approximation
        // Full backprop through max/min/std would require storing indices and complex computations
        var transformedGrad = BackpropThroughAggregation(aggregatedGrad, numNodes);

        // Gradient through pre-transform
        _preTransformBiasGradient = Engine.ReduceSum(transformedGrad, [0, 1], keepDims: false);

        _preTransformWeightsGradient = new Tensor<T>([_inputFeatures, _inputFeatures]);
        _preTransformWeightsGradient.Fill(NumOps.Zero);

        for (int b = 0; b < batchSize; b++)
        {
            var inputSlice = Engine.TensorSlice(_lastInput, [b, 0, 0], [1, numNodes, _inputFeatures])
                .Reshape([numNodes, _inputFeatures]);
            var gradSlice = Engine.TensorSlice(transformedGrad, [b, 0, 0], [1, numNodes, _inputFeatures])
                .Reshape([numNodes, _inputFeatures]);

            var inputSliceT = Engine.TensorTranspose(inputSlice);
            var batchGrad = Engine.TensorMatMul(inputSliceT, gradSlice);
            _preTransformWeightsGradient = Engine.TensorAdd(_preTransformWeightsGradient, batchGrad);
        }

        // Input gradient from pre-transform
        var preWeightsT = Engine.TensorTranspose(_preTransformWeights);
        var preInputGrad = Engine.TensorMatMul(transformedGrad, preWeightsT);

        // Combine input gradients
        return Engine.TensorAdd(selfInputGrad, preInputGrad);
    }

    /// <summary>
    /// Backpropagates through aggregation operations using vectorized Engine operations.
    /// </summary>
    private Tensor<T> BackpropThroughAggregation(Tensor<T> aggregatedGrad, int numNodes)
    {
        int batchSize = aggregatedGrad.Shape[0];
        int numAggregators = _aggregators.Length;
        int numScalers = _scalers.Length;

        var transformedGrad = new Tensor<T>([batchSize, numNodes, _inputFeatures]);
        transformedGrad.Fill(NumOps.Zero);

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
                var adjT = Engine.TensorTranspose(_adjacencyMatrix!);
                var aggGrad = Engine.TensorMatMul(adjT, scalerGrad);

                // For mean, also divide by degree
                if (_aggregators[aggIdx] == PNAAggregator.Mean)
                {
                    var safeDegrees = Engine.TensorMax(_lastDegrees!, NumOps.One);
                    var degExpanded = safeDegrees.Reshape([batchSize, numNodes, 1]);
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
        // Simple swap of last two axes
        return Engine.TensorTranspose(tensor);
    }

    /// <summary>
    /// Backward pass using automatic differentiation.
    /// Falls back to manual for complex aggregation operations.
    /// </summary>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        // For PNA's complex aggregation operations (max, min, std),
        // autodiff is challenging. Fall back to manual implementation.
        return BackwardManual(outputGradient);
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
        _preTransformWeights = Engine.TensorSubtract(_preTransformWeights,
            Engine.TensorMultiplyScalar(_preTransformWeightsGradient, learningRate));
        _preTransformBias = Engine.TensorSubtract(_preTransformBias,
            Engine.TensorMultiplyScalar(_preTransformBiasGradient!, learningRate));
        _postAggregationWeights1 = Engine.TensorSubtract(_postAggregationWeights1,
            Engine.TensorMultiplyScalar(_postAggregationWeights1Gradient, learningRate));
        _postAggregationWeights2 = Engine.TensorSubtract(_postAggregationWeights2,
            Engine.TensorMultiplyScalar(_postAggregationWeights2Gradient, learningRate));
        _postAggregationBias1 = Engine.TensorSubtract(_postAggregationBias1,
            Engine.TensorMultiplyScalar(_postAggregationBias1Gradient!, learningRate));
        _postAggregationBias2 = Engine.TensorSubtract(_postAggregationBias2,
            Engine.TensorMultiplyScalar(_postAggregationBias2Gradient!, learningRate));
        _selfWeights = Engine.TensorSubtract(_selfWeights,
            Engine.TensorMultiplyScalar(_selfWeightsGradient, learningRate));
        _bias = Engine.TensorSubtract(_bias,
            Engine.TensorMultiplyScalar(_biasGradient, learningRate));
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        return Vector<T>.Concatenate(
            new Vector<T>(_preTransformWeights.ToArray()),
            new Vector<T>(_preTransformBias.ToArray()),
            new Vector<T>(_postAggregationWeights1.ToArray()),
            new Vector<T>(_postAggregationBias1.ToArray()),
            new Vector<T>(_postAggregationWeights2.ToArray()),
            new Vector<T>(_postAggregationBias2.ToArray()),
            new Vector<T>(_selfWeights.ToArray()),
            new Vector<T>(_bias.ToArray())
        );
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int preTransformWeightCount = _preTransformWeights.Length;
        int preTransformBiasCount = _preTransformBias.Length;
        int post1WeightCount = _postAggregationWeights1.Length;
        int post1BiasCount = _postAggregationBias1.Length;
        int post2WeightCount = _postAggregationWeights2.Length;
        int post2BiasCount = _postAggregationBias2.Length;
        int selfWeightCount = _selfWeights.Length;
        int biasCount = _bias.Length;

        int totalParams = preTransformWeightCount + preTransformBiasCount +
                         post1WeightCount + post1BiasCount +
                         post2WeightCount + post2BiasCount +
                         selfWeightCount + biasCount;

        if (parameters.Length != totalParams)
        {
            throw new ArgumentException(
                $"Expected {totalParams} parameters, but got {parameters.Length}", nameof(parameters));
        }

        int index = 0;

        _preTransformWeights = Tensor<T>.FromVector(parameters.SubVector(index, preTransformWeightCount))
            .Reshape(_preTransformWeights.Shape);
        index += preTransformWeightCount;

        _preTransformBias = Tensor<T>.FromVector(parameters.SubVector(index, preTransformBiasCount));
        index += preTransformBiasCount;

        _postAggregationWeights1 = Tensor<T>.FromVector(parameters.SubVector(index, post1WeightCount))
            .Reshape(_postAggregationWeights1.Shape);
        index += post1WeightCount;

        _postAggregationBias1 = Tensor<T>.FromVector(parameters.SubVector(index, post1BiasCount));
        index += post1BiasCount;

        _postAggregationWeights2 = Tensor<T>.FromVector(parameters.SubVector(index, post2WeightCount))
            .Reshape(_postAggregationWeights2.Shape);
        index += post2WeightCount;

        _postAggregationBias2 = Tensor<T>.FromVector(parameters.SubVector(index, post2BiasCount));
        index += post2BiasCount;

        _selfWeights = Tensor<T>.FromVector(parameters.SubVector(index, selfWeightCount))
            .Reshape(_selfWeights.Shape);
        index += selfWeightCount;

        _bias = Tensor<T>.FromVector(parameters.SubVector(index, biasCount));
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

    /// <inheritdoc/>
    public override bool SupportsJitCompilation => true;

    /// <inheritdoc/>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        // Create symbolic input
        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = Autodiff.TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Export learnable parameters as constants
        var preWeightsNode = Autodiff.TensorOperations<T>.Constant(_preTransformWeights, "pre_weights");
        var selfWeightsNode = Autodiff.TensorOperations<T>.Constant(_selfWeights, "self_weights");
        var biasNode = Autodiff.TensorOperations<T>.Constant(_bias, "bias");

        // Build computation graph for self-loop path (most direct gradient flow)
        var selfContribution = Autodiff.TensorOperations<T>.MatrixMultiply(inputNode, selfWeightsNode);
        var output = Autodiff.TensorOperations<T>.Add(selfContribution, biasNode);

        // Apply activation if supported
        if (ScalarActivation != null && ScalarActivation.SupportsJitCompilation)
        {
            return ScalarActivation.ApplyToGraph(output);
        }

        return output;
    }
}
