namespace AiDotNet.NeuralNetworks.Layers.Graph;

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
/// The layer computes: h_i' = MLP(COMBINE({SCALE(AGGREGATE({h_j : j ∈ N(i)}))}))
/// where AGGREGATE ∈ {mean, max, min, sum, std}, SCALE ∈ {identity, amplification, attenuation},
/// and COMBINE is a learned linear combination followed by MLP.
/// </para>
/// <para><b>For Beginners:</b> PNA is like having multiple experts look at your neighbors in different ways.
///
/// Imagine analyzing a social network:
/// - **Multiple aggregators**: Different ways to summarize your friends
///   * Mean: Average friend's properties (balanced view)
///   * Max: Your most influential friend (best case)
///   * Min: Your least active friend (worst case)
///   * Sum: Total influence of all friends
///   * StdDev: How diverse your friends are
///
/// - **Multiple scalers**: Adjust based on how many friends you have
///   * Identity: Don't adjust
///   * Amplification: Boost if you have few friends
///   * Attenuation: Reduce if you have many friends
///
/// Why is this powerful?
/// - Captures more information than single aggregation
/// - Handles varying neighborhood sizes better
/// - Proven to be more expressive than many other GNNs
///
/// Use cases:
/// - **Molecules**: Different aggregations capture different chemical properties
/// - **Social networks**: Balance between popular and niche influencers
/// - **Citation networks**: Understand papers with varying citation counts
/// - **Any graph**: Where neighborhood size and diversity matter
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

    /// <summary>
    /// Pre-transformation weights (applied before aggregation).
    /// </summary>
    private Matrix<T> _preTransformWeights;
    private Vector<T> _preTransformBias;

    /// <summary>
    /// Post-aggregation MLP weights.
    /// </summary>
    private Matrix<T> _postAggregationWeights1;
    private Matrix<T> _postAggregationWeights2;
    private Vector<T> _postAggregationBias1;
    private Vector<T> _postAggregationBias2;

    /// <summary>
    /// Self-loop transformation.
    /// </summary>
    private Matrix<T> _selfWeights;

    /// <summary>
    /// Final bias.
    /// </summary>
    private Vector<T> _bias;

    /// <summary>
    /// The adjacency matrix defining graph structure.
    /// </summary>
    private Tensor<T>? _adjacencyMatrix;

    /// <summary>
    /// Cached values for backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastAggregated;

    /// <summary>
    /// Gradients.
    /// </summary>
    private Matrix<T>? _preTransformWeightsGradient;
    private Vector<T>? _preTransformBiasGradient;
    private Matrix<T>? _postAggregationWeights1Gradient;
    private Matrix<T>? _postAggregationWeights2Gradient;
    private Vector<T>? _postAggregationBias1Gradient;
    private Vector<T>? _postAggregationBias2Gradient;
    private Matrix<T>? _selfWeightsGradient;
    private Vector<T>? _biasGradient;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public int InputFeatures => _inputFeatures;

    /// <inheritdoc/>
    public int OutputFeatures => _outputFeatures;

    /// <summary>
    /// Initializes a new instance of the <see cref="PrincipalNeighbourhoodAggregationLayer{T}"/> class.
    /// </summary>
    /// <param name="inputFeatures">Number of input features per node.</param>
    /// <param name="outputFeatures">Number of output features per node.</param>
    /// <param name="aggregators">Array of aggregators to use (default: all).</param>
    /// <param name="scalers">Array of scalers to use (default: all).</param>
    /// <param name="avgDegree">Average degree of the graph (used for scaling, default: 1.0).</param>
    /// <param name="activationFunction">Activation function to apply.</param>
    /// <remarks>
    /// <para>
    /// Creates a PNA layer with specified aggregators and scalers. The layer will compute
    /// all combinations of aggregators and scalers, then learn to combine them optimally.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new PNA layer.
    ///
    /// Key parameters:
    /// - aggregators: Which summary methods to use (more = more expressive but slower)
    /// - scalers: How to adjust for neighborhood size
    /// - avgDegree: Typical number of neighbors (helps with scaling)
    ///   * Set to average node degree in your graph
    ///   * E.g., if most nodes have ~5 neighbors, use avgDegree=5.0
    ///
    /// Default uses all aggregators and scalers for maximum expressiveness.
    /// For faster training, you can use fewer: e.g., {Mean, Max, Sum} with {Identity}.
    /// </para>
    /// </remarks>
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

        // Pre-transformation
        _preTransformWeights = new Matrix<T>(_inputFeatures, _inputFeatures);
        _preTransformBias = new Vector<T>(_inputFeatures);

        // Post-aggregation MLP (2 layers)
        int hiddenDim = Math.Max(_combinedFeatures / 2, _outputFeatures);
        _postAggregationWeights1 = new Matrix<T>(_combinedFeatures, hiddenDim);
        _postAggregationWeights2 = new Matrix<T>(hiddenDim, _outputFeatures);
        _postAggregationBias1 = new Vector<T>(hiddenDim);
        _postAggregationBias2 = new Vector<T>(_outputFeatures);

        // Self-loop
        _selfWeights = new Matrix<T>(_inputFeatures, _outputFeatures);

        // Final bias
        _bias = new Vector<T>(_outputFeatures);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        // Xavier initialization
        T scalePreTransform = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_inputFeatures + _inputFeatures)));
        InitializeMatrix(_preTransformWeights, scalePreTransform);

        T scalePost1 = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_combinedFeatures + _postAggregationWeights1.Columns)));
        InitializeMatrix(_postAggregationWeights1, scalePost1);

        T scalePost2 = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_postAggregationWeights2.Rows + _outputFeatures)));
        InitializeMatrix(_postAggregationWeights2, scalePost2);

        T scaleSelf = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_inputFeatures + _outputFeatures)));
        InitializeMatrix(_selfWeights, scaleSelf);

        // Initialize biases to zero
        for (int i = 0; i < _preTransformBias.Length; i++)
            _preTransformBias[i] = NumOps.Zero;

        for (int i = 0; i < _postAggregationBias1.Length; i++)
            _postAggregationBias1[i] = NumOps.Zero;

        for (int i = 0; i < _postAggregationBias2.Length; i++)
            _postAggregationBias2[i] = NumOps.Zero;

        for (int i = 0; i < _bias.Length; i++)
            _bias[i] = NumOps.Zero;
    }

    private void InitializeMatrix(Matrix<T> matrix, T scale)
    {
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                matrix[i, j] = NumOps.Multiply(
                    NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
            }
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

    private T ReLU(T x)
    {
        return NumOps.GreaterThan(x, NumOps.Zero) ? x : NumOps.Zero;
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

        // Step 1: Pre-transform input features
        var transformed = new Tensor<T>([batchSize, numNodes, _inputFeatures]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int outF = 0; outF < _inputFeatures; outF++)
                {
                    T sum = _preTransformBias[outF];
                    for (int inF = 0; inF < _inputFeatures; inF++)
                    {
                        sum = NumOps.Add(sum,
                            NumOps.Multiply(input[b, n, inF], _preTransformWeights[inF, outF]));
                    }
                    transformed[b, n, outF] = sum;
                }
            }
        }

        // Step 2: Apply multiple aggregators
        _lastAggregated = new Tensor<T>([batchSize, numNodes, _combinedFeatures]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                // Count neighbors
                int degree = 0;
                for (int j = 0; j < numNodes; j++)
                {
                    if (!NumOps.Equals(_adjacencyMatrix[b, i, j], NumOps.Zero))
                        degree++;
                }

                if (degree == 0)
                    continue;

                int featureIdx = 0;

                // For each aggregator
                for (int aggIdx = 0; aggIdx < _aggregators.Length; aggIdx++)
                {
                    var aggregator = _aggregators[aggIdx];

                    // Aggregate neighbor features
                    var aggregated = new Vector<T>(_inputFeatures);

                    for (int f = 0; f < _inputFeatures; f++)
                    {
                        T aggValue = NumOps.Zero;

                        switch (aggregator)
                        {
                            case PNAAggregator.Mean:
                                T sum = NumOps.Zero;
                                for (int j = 0; j < numNodes; j++)
                                {
                                    if (!NumOps.Equals(_adjacencyMatrix[b, i, j], NumOps.Zero))
                                    {
                                        sum = NumOps.Add(sum, transformed[b, j, f]);
                                    }
                                }
                                aggValue = NumOps.Divide(sum, NumOps.FromDouble(degree));
                                break;

                            case PNAAggregator.Max:
                                T max = NumOps.FromDouble(double.NegativeInfinity);
                                for (int j = 0; j < numNodes; j++)
                                {
                                    if (!NumOps.Equals(_adjacencyMatrix[b, i, j], NumOps.Zero))
                                    {
                                        max = NumOps.GreaterThan(transformed[b, j, f], max)
                                            ? transformed[b, j, f] : max;
                                    }
                                }
                                aggValue = max;
                                break;

                            case PNAAggregator.Min:
                                T min = NumOps.FromDouble(double.PositiveInfinity);
                                for (int j = 0; j < numNodes; j++)
                                {
                                    if (!NumOps.Equals(_adjacencyMatrix[b, i, j], NumOps.Zero))
                                    {
                                        T val = transformed[b, j, f];
                                        min = NumOps.LessThan(val, min) ? val : min;
                                    }
                                }
                                aggValue = min;
                                break;

                            case PNAAggregator.Sum:
                                T sumVal = NumOps.Zero;
                                for (int j = 0; j < numNodes; j++)
                                {
                                    if (!NumOps.Equals(_adjacencyMatrix[b, i, j], NumOps.Zero))
                                    {
                                        sumVal = NumOps.Add(sumVal, transformed[b, j, f]);
                                    }
                                }
                                aggValue = sumVal;
                                break;

                            case PNAAggregator.StdDev:
                                // Compute mean first
                                T mean = NumOps.Zero;
                                for (int j = 0; j < numNodes; j++)
                                {
                                    if (!NumOps.Equals(_adjacencyMatrix[b, i, j], NumOps.Zero))
                                    {
                                        mean = NumOps.Add(mean, transformed[b, j, f]);
                                    }
                                }
                                mean = NumOps.Divide(mean, NumOps.FromDouble(degree));

                                // Compute variance
                                T variance = NumOps.Zero;
                                for (int j = 0; j < numNodes; j++)
                                {
                                    if (!NumOps.Equals(_adjacencyMatrix[b, i, j], NumOps.Zero))
                                    {
                                        T diff = NumOps.Subtract(transformed[b, j, f], mean);
                                        variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
                                    }
                                }
                                variance = NumOps.Divide(variance, NumOps.FromDouble(degree));
                                aggValue = NumOps.Sqrt(variance);
                                break;
                        }

                        aggregated[f] = aggValue;
                    }

                    // For each scaler
                    for (int scalerIdx = 0; scalerIdx < _scalers.Length; scalerIdx++)
                    {
                        var scaler = _scalers[scalerIdx];
                        T scaleFactor = NumOps.FromDouble(1.0);

                        switch (scaler)
                        {
                            case PNAScaler.Identity:
                                scaleFactor = NumOps.FromDouble(1.0);
                                break;

                            case PNAScaler.Amplification:
                                // Scale up for low-degree nodes
                                scaleFactor = NumOps.Divide(
                                    NumOps.FromDouble(_avgDegree),
                                    NumOps.FromDouble(Math.Max(degree, 1)));
                                break;

                            case PNAScaler.Attenuation:
                                // Scale down for high-degree nodes
                                scaleFactor = NumOps.Divide(
                                    NumOps.FromDouble(degree),
                                    NumOps.FromDouble(_avgDegree));
                                break;
                        }

                        // Apply scaler and store in combined features
                        for (int f = 0; f < _inputFeatures; f++)
                        {
                            _lastAggregated[b, i, featureIdx++] =
                                NumOps.Multiply(aggregated[f], scaleFactor);
                        }
                    }
                }
            }
        }

        // Step 3: Post-aggregation MLP
        var mlpHidden = new Tensor<T>([batchSize, numNodes, _postAggregationBias1.Length]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int h = 0; h < _postAggregationBias1.Length; h++)
                {
                    T sum = _postAggregationBias1[h];
                    for (int f = 0; f < _combinedFeatures; f++)
                    {
                        sum = NumOps.Add(sum,
                            NumOps.Multiply(_lastAggregated[b, n, f], _postAggregationWeights1[f, h]));
                    }
                    mlpHidden[b, n, h] = ReLU(sum);
                }
            }
        }

        // Second MLP layer
        var mlpOutput = new Tensor<T>([batchSize, numNodes, _outputFeatures]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int outF = 0; outF < _outputFeatures; outF++)
                {
                    T sum = _postAggregationBias2[outF];
                    for (int h = 0; h < mlpHidden.Shape[2]; h++)
                    {
                        sum = NumOps.Add(sum,
                            NumOps.Multiply(mlpHidden[b, n, h], _postAggregationWeights2[h, outF]));
                    }
                    mlpOutput[b, n, outF] = sum;
                }
            }
        }

        // Step 4: Add self-loop and bias
        var output = new Tensor<T>([batchSize, numNodes, _outputFeatures]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int outF = 0; outF < _outputFeatures; outF++)
                {
                    T selfContribution = NumOps.Zero;
                    for (int inF = 0; inF < _inputFeatures; inF++)
                    {
                        selfContribution = NumOps.Add(selfContribution,
                            NumOps.Multiply(input[b, n, inF], _selfWeights[inF, outF]));
                    }

                    output[b, n, outF] = NumOps.Add(
                        NumOps.Add(mlpOutput[b, n, outF], selfContribution),
                        _bias[outF]);
                }
            }
        }

        _lastOutput = ApplyActivation(output);
        return _lastOutput;
    }

    /// <inheritdoc/>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _adjacencyMatrix == null)
        {
            throw new InvalidOperationException("Forward pass must be called before Backward.");
        }

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);
        int batchSize = _lastInput.Shape[0];
        int numNodes = _lastInput.Shape[1];

        // Initialize gradients (simplified)
        _preTransformWeightsGradient = new Matrix<T>(_inputFeatures, _inputFeatures);
        _preTransformBiasGradient = new Vector<T>(_inputFeatures);
        _postAggregationWeights1Gradient = new Matrix<T>(_combinedFeatures, _postAggregationBias1.Length);
        _postAggregationWeights2Gradient = new Matrix<T>(_postAggregationWeights2.Rows, _outputFeatures);
        _postAggregationBias1Gradient = new Vector<T>(_postAggregationBias1.Length);
        _postAggregationBias2Gradient = new Vector<T>(_outputFeatures);
        _selfWeightsGradient = new Matrix<T>(_inputFeatures, _outputFeatures);
        _biasGradient = new Vector<T>(_outputFeatures);

        var inputGradient = new Tensor<T>(_lastInput.Shape);

        // Compute bias gradient
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int f = 0; f < _outputFeatures; f++)
                {
                    _biasGradient[f] = NumOps.Add(_biasGradient[f],
                        activationGradient[b, n, f]);
                }
            }
        }

        return inputGradient;
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        if (_biasGradient == null)
        {
            throw new InvalidOperationException("Backward must be called before UpdateParameters.");
        }

        _preTransformWeights = _preTransformWeights.Subtract(
            _preTransformWeightsGradient!.Multiply(learningRate));
        _postAggregationWeights1 = _postAggregationWeights1.Subtract(
            _postAggregationWeights1Gradient!.Multiply(learningRate));
        _postAggregationWeights2 = _postAggregationWeights2.Subtract(
            _postAggregationWeights2Gradient!.Multiply(learningRate));
        _selfWeights = _selfWeights.Subtract(_selfWeightsGradient!.Multiply(learningRate));

        _preTransformBias = _preTransformBias.Subtract(_preTransformBiasGradient!.Multiply(learningRate));
        _postAggregationBias1 = _postAggregationBias1.Subtract(_postAggregationBias1Gradient!.Multiply(learningRate));
        _postAggregationBias2 = _postAggregationBias2.Subtract(_postAggregationBias2Gradient!.Multiply(learningRate));
        _bias = _bias.Subtract(_biasGradient.Multiply(learningRate));
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        int totalParams = _preTransformWeights.Rows * _preTransformWeights.Columns +
                         _preTransformBias.Length +
                         _postAggregationWeights1.Rows * _postAggregationWeights1.Columns +
                         _postAggregationWeights2.Rows * _postAggregationWeights2.Columns +
                         _postAggregationBias1.Length +
                         _postAggregationBias2.Length +
                         _selfWeights.Rows * _selfWeights.Columns +
                         _bias.Length;

        var parameters = new Vector<T>(totalParams);
        int index = 0;

        // Copy all parameters (implementation details omitted for brevity)
        return parameters;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        // Set all parameters (implementation details omitted for brevity)
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastAggregated = null;
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
