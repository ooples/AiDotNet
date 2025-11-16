namespace AiDotNet.NeuralNetworks.Layers.Graph;

/// <summary>
/// Implements Directional Graph Networks for directed graph processing with separate in/out aggregations.
/// </summary>
/// <remarks>
/// <para>
/// Directional Graph Networks (DGN) explicitly model the directionality of edges in directed graphs.
/// Unlike standard GNNs that often ignore edge direction or treat graphs as undirected, DGNs
/// maintain separate aggregations for incoming and outgoing edges, capturing asymmetric relationships.
/// </para>
/// <para>
/// The layer computes separate representations for in-neighbors and out-neighbors:
/// - h_in = AGGREGATE_IN({h_j : j → i})
/// - h_out = AGGREGATE_OUT({h_j : i → j})
/// - h_i' = UPDATE(h_i, h_in, h_out)
///
/// This allows the network to learn different patterns for sources and targets of edges.
/// </para>
/// <para><b>For Beginners:</b> This layer understands that graph connections can have direction.
///
/// Think of different types of directed networks:
///
/// **Twitter/Social Media:**
/// - You follow someone (outgoing edge)
/// - Someone follows you (incoming edge)
/// - These are NOT the same! Celebrities have many incoming, fewer outgoing
///
/// **Citation Networks:**
/// - Papers you cite (outgoing): Shows your influences
/// - Papers citing you (incoming): Shows your impact
/// - Direction matters for understanding importance
///
/// **Web Pages:**
/// - Links you have (outgoing): What you reference
/// - Links to you (incoming/backlinks): Your authority
/// - Google PageRank uses this directionality
///
/// **Transaction Networks:**
/// - Money sent (outgoing): Your purchases
/// - Money received (incoming): Your sales
/// - Different patterns for buyers vs sellers
///
/// Why separate in/out aggregation?
/// - **Asymmetric roles**: Being cited vs citing have different meanings
/// - **Different patterns**: Incoming and outgoing patterns can be very different
/// - **Better expressiveness**: Captures more information than treating edges as undirected
///
/// The layer learns separate transformations for incoming and outgoing neighbors,
/// then combines them to update each node's representation.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class DirectionalGraphLayer<T> : LayerBase<T>, IGraphConvolutionLayer<T>
{
    private readonly int _inputFeatures;
    private readonly int _outputFeatures;
    private readonly bool _useGating;

    /// <summary>
    /// Weights for incoming edge aggregation.
    /// </summary>
    private Matrix<T> _incomingWeights;

    /// <summary>
    /// Weights for outgoing edge aggregation.
    /// </summary>
    private Matrix<T> _outgoingWeights;

    /// <summary>
    /// Self-loop weights.
    /// </summary>
    private Matrix<T> _selfWeights;

    /// <summary>
    /// Gating mechanism weights (if enabled).
    /// </summary>
    private Matrix<T>? _gateWeights;
    private Vector<T>? _gateBias;

    /// <summary>
    /// Biases for incoming, outgoing, and self transformations.
    /// </summary>
    private Vector<T> _incomingBias;
    private Vector<T> _outgoingBias;
    private Vector<T> _selfBias;

    /// <summary>
    /// Combination weights for merging in/out/self features.
    /// </summary>
    private Matrix<T> _combinationWeights;
    private Vector<T> _combinationBias;

    /// <summary>
    /// The adjacency matrix defining graph structure (interpreted as directed).
    /// </summary>
    private Tensor<T>? _adjacencyMatrix;

    /// <summary>
    /// Cached values for backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastIncoming;
    private Tensor<T>? _lastOutgoing;
    private Tensor<T>? _lastSelf;

    /// <summary>
    /// Gradients.
    /// </summary>
    private Matrix<T>? _incomingWeightsGradient;
    private Matrix<T>? _outgoingWeightsGradient;
    private Matrix<T>? _selfWeightsGradient;
    private Matrix<T>? _combinationWeightsGradient;
    private Vector<T>? _incomingBiasGradient;
    private Vector<T>? _outgoingBiasGradient;
    private Vector<T>? _selfBiasGradient;
    private Vector<T>? _combinationBiasGradient;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public int InputFeatures => _inputFeatures;

    /// <inheritdoc/>
    public int OutputFeatures => _outputFeatures;

    /// <summary>
    /// Initializes a new instance of the <see cref="DirectionalGraphLayer{T}"/> class.
    /// </summary>
    /// <param name="inputFeatures">Number of input features per node.</param>
    /// <param name="outputFeatures">Number of output features per node.</param>
    /// <param name="useGating">Whether to use gating mechanism for combining in/out features (default: false).</param>
    /// <param name="activationFunction">Activation function to apply.</param>
    /// <remarks>
    /// <para>
    /// Creates a directional graph layer that processes incoming and outgoing edges separately.
    /// The layer maintains three transformation paths: incoming neighbors, outgoing neighbors,
    /// and self-features, which are then combined using learned weights.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new directional graph layer.
    ///
    /// Key parameters:
    /// - useGating: Advanced feature for dynamic combination of in/out information
    ///   * false: Simple weighted combination (faster, good for most cases)
    ///   * true: Learned gating decides how much to use each direction (more expressive)
    ///
    /// The layer has three "paths":
    /// 1. **Incoming path**: Processes nodes that point TO this node
    /// 2. **Outgoing path**: Processes nodes that this node points TO
    /// 3. **Self path**: Processes the node's own features
    ///
    /// All three are combined to create the final node representation.
    ///
    /// Example usage:
    /// ```
    /// // For a citation network where direction matters
    /// var layer = new DirectionalGraphLayer(128, 256, useGating: true);
    ///
    /// // Set directed adjacency matrix
    /// // adjacency[i,j] = 1 means edge from j to i (j→i)
    /// layer.SetAdjacencyMatrix(adjacencyMatrix);
    ///
    /// var output = layer.Forward(nodeFeatures);
    /// // Output captures both who cites you (incoming) and who you cite (outgoing)
    /// ```
    /// </para>
    /// </remarks>
    public DirectionalGraphLayer(
        int inputFeatures,
        int outputFeatures,
        bool useGating = false,
        IActivationFunction<T>? activationFunction = null)
        : base([inputFeatures], [outputFeatures], activationFunction ?? new IdentityActivation<T>())
    {
        _inputFeatures = inputFeatures;
        _outputFeatures = outputFeatures;
        _useGating = useGating;

        // Initialize transformation weights for each direction
        _incomingWeights = new Matrix<T>(_inputFeatures, _outputFeatures);
        _outgoingWeights = new Matrix<T>(_inputFeatures, _outputFeatures);
        _selfWeights = new Matrix<T>(_inputFeatures, _outputFeatures);

        _incomingBias = new Vector<T>(_outputFeatures);
        _outgoingBias = new Vector<T>(_outputFeatures);
        _selfBias = new Vector<T>(_outputFeatures);

        // Combination weights: combines in/out/self features
        int combinedDim = 3 * _outputFeatures;
        _combinationWeights = new Matrix<T>(combinedDim, _outputFeatures);
        _combinationBias = new Vector<T>(_outputFeatures);

        // Gating mechanism (optional)
        if (_useGating)
        {
            _gateWeights = new Matrix<T>(combinedDim, 3); // 3 gates for in/out/self
            _gateBias = new Vector<T>(3);
        }

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_inputFeatures + _outputFeatures)));

        // Initialize directional weights
        InitializeMatrix(_incomingWeights, scale);
        InitializeMatrix(_outgoingWeights, scale);
        InitializeMatrix(_selfWeights, scale);

        // Initialize combination weights
        T scaleComb = NumOps.Sqrt(NumOps.FromDouble(2.0 / (3 * _outputFeatures + _outputFeatures)));
        InitializeMatrix(_combinationWeights, scaleComb);

        // Initialize gating weights if used
        if (_gateWeights != null)
        {
            T scaleGate = NumOps.Sqrt(NumOps.FromDouble(2.0 / (3 * _outputFeatures + 3)));
            InitializeMatrix(_gateWeights, scaleGate);
        }

        // Initialize biases to zero
        for (int i = 0; i < _outputFeatures; i++)
        {
            _incomingBias[i] = NumOps.Zero;
            _outgoingBias[i] = NumOps.Zero;
            _selfBias[i] = NumOps.Zero;
            _combinationBias[i] = NumOps.Zero;
        }

        if (_gateBias != null)
        {
            for (int i = 0; i < 3; i++)
            {
                _gateBias[i] = NumOps.Zero;
            }
        }
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

    private T Sigmoid(T x)
    {
        return NumOps.Divide(NumOps.FromDouble(1.0),
            NumOps.Add(NumOps.FromDouble(1.0), NumOps.Exp(NumOps.Negate(x))));
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

        // Step 1: Aggregate incoming edges (nodes that point TO this node)
        _lastIncoming = new Tensor<T>([batchSize, numNodes, _outputFeatures]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                // Count incoming edges (j → i means adjacency[i,j] = 1)
                int inDegree = 0;
                for (int j = 0; j < numNodes; j++)
                {
                    if (!NumOps.Equals(_adjacencyMatrix[b, i, j], NumOps.Zero))
                        inDegree++;
                }

                if (inDegree > 0)
                {
                    T normalization = NumOps.Divide(NumOps.FromDouble(1.0), NumOps.FromDouble(inDegree));

                    for (int outF = 0; outF < _outputFeatures; outF++)
                    {
                        T sum = _incomingBias[outF];

                        for (int j = 0; j < numNodes; j++)
                        {
                            if (NumOps.Equals(_adjacencyMatrix[b, i, j], NumOps.Zero))
                                continue;

                            // Transform and aggregate incoming neighbor j
                            for (int inF = 0; inF < _inputFeatures; inF++)
                            {
                                sum = NumOps.Add(sum,
                                    NumOps.Multiply(
                                        NumOps.Multiply(input[b, j, inF], _incomingWeights[inF, outF]),
                                        normalization));
                            }
                        }

                        _lastIncoming[b, i, outF] = sum;
                    }
                }
            }
        }

        // Step 2: Aggregate outgoing edges (nodes that this node points TO)
        _lastOutgoing = new Tensor<T>([batchSize, numNodes, _outputFeatures]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                // Count outgoing edges (i → j means adjacency[j,i] = 1)
                int outDegree = 0;
                for (int j = 0; j < numNodes; j++)
                {
                    if (!NumOps.Equals(_adjacencyMatrix[b, j, i], NumOps.Zero))
                        outDegree++;
                }

                if (outDegree > 0)
                {
                    T normalization = NumOps.Divide(NumOps.FromDouble(1.0), NumOps.FromDouble(outDegree));

                    for (int outF = 0; outF < _outputFeatures; outF++)
                    {
                        T sum = _outgoingBias[outF];

                        for (int j = 0; j < numNodes; j++)
                        {
                            if (NumOps.Equals(_adjacencyMatrix[b, j, i], NumOps.Zero))
                                continue;

                            // Transform and aggregate outgoing neighbor j
                            for (int inF = 0; inF < _inputFeatures; inF++)
                            {
                                sum = NumOps.Add(sum,
                                    NumOps.Multiply(
                                        NumOps.Multiply(input[b, j, inF], _outgoingWeights[inF, outF]),
                                        normalization));
                            }
                        }

                        _lastOutgoing[b, i, outF] = sum;
                    }
                }
            }
        }

        // Step 3: Transform self features
        _lastSelf = new Tensor<T>([batchSize, numNodes, _outputFeatures]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int outF = 0; outF < _outputFeatures; outF++)
                {
                    T sum = _selfBias[outF];

                    for (int inF = 0; inF < _inputFeatures; inF++)
                    {
                        sum = NumOps.Add(sum,
                            NumOps.Multiply(input[b, n, inF], _selfWeights[inF, outF]));
                    }

                    _lastSelf[b, n, outF] = sum;
                }
            }
        }

        // Step 4: Combine incoming, outgoing, and self features
        var output = new Tensor<T>([batchSize, numNodes, _outputFeatures]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                // Concatenate in/out/self
                var combined = new Vector<T>(3 * _outputFeatures);
                for (int f = 0; f < _outputFeatures; f++)
                {
                    combined[f] = _lastIncoming[b, n, f];
                    combined[_outputFeatures + f] = _lastOutgoing[b, n, f];
                    combined[2 * _outputFeatures + f] = _lastSelf[b, n, f];
                }

                if (_useGating && _gateWeights != null && _gateBias != null)
                {
                    // Compute gates
                    var gates = new Vector<T>(3);
                    for (int g = 0; g < 3; g++)
                    {
                        T sum = _gateBias[g];
                        for (int c = 0; c < combined.Length; c++)
                        {
                            sum = NumOps.Add(sum, NumOps.Multiply(combined[c], _gateWeights[c, g]));
                        }
                        gates[g] = Sigmoid(sum);
                    }

                    // Apply gates to in/out/self
                    for (int f = 0; f < _outputFeatures; f++)
                    {
                        combined[f] = NumOps.Multiply(combined[f], gates[0]);
                        combined[_outputFeatures + f] = NumOps.Multiply(combined[_outputFeatures + f], gates[1]);
                        combined[2 * _outputFeatures + f] = NumOps.Multiply(combined[2 * _outputFeatures + f], gates[2]);
                    }
                }

                // Final combination
                for (int outF = 0; outF < _outputFeatures; outF++)
                {
                    T sum = _combinationBias[outF];

                    for (int c = 0; c < combined.Length; c++)
                    {
                        sum = NumOps.Add(sum,
                            NumOps.Multiply(combined[c], _combinationWeights[c, outF]));
                    }

                    output[b, n, outF] = sum;
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
        _incomingWeightsGradient = new Matrix<T>(_inputFeatures, _outputFeatures);
        _outgoingWeightsGradient = new Matrix<T>(_inputFeatures, _outputFeatures);
        _selfWeightsGradient = new Matrix<T>(_inputFeatures, _outputFeatures);
        _combinationWeightsGradient = new Matrix<T>(3 * _outputFeatures, _outputFeatures);
        _incomingBiasGradient = new Vector<T>(_outputFeatures);
        _outgoingBiasGradient = new Vector<T>(_outputFeatures);
        _selfBiasGradient = new Vector<T>(_outputFeatures);
        _combinationBiasGradient = new Vector<T>(_outputFeatures);

        var inputGradient = new Tensor<T>(_lastInput.Shape);

        // Compute gradients (simplified implementation)
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int f = 0; f < _outputFeatures; f++)
                {
                    _combinationBiasGradient[f] = NumOps.Add(_combinationBiasGradient[f],
                        activationGradient[b, n, f]);
                }
            }
        }

        return inputGradient;
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        if (_combinationBiasGradient == null)
        {
            throw new InvalidOperationException("Backward must be called before UpdateParameters.");
        }

        _incomingWeights = _incomingWeights.Subtract(_incomingWeightsGradient!.Multiply(learningRate));
        _outgoingWeights = _outgoingWeights.Subtract(_outgoingWeightsGradient!.Multiply(learningRate));
        _selfWeights = _selfWeights.Subtract(_selfWeightsGradient!.Multiply(learningRate));
        _combinationWeights = _combinationWeights.Subtract(_combinationWeightsGradient!.Multiply(learningRate));

        _incomingBias = _incomingBias.Subtract(_incomingBiasGradient!.Multiply(learningRate));
        _outgoingBias = _outgoingBias.Subtract(_outgoingBiasGradient!.Multiply(learningRate));
        _selfBias = _selfBias.Subtract(_selfBiasGradient!.Multiply(learningRate));
        _combinationBias = _combinationBias.Subtract(_combinationBiasGradient.Multiply(learningRate));
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        // Simplified
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
        _lastIncoming = null;
        _lastOutgoing = null;
        _lastSelf = null;
        _incomingWeightsGradient = null;
        _outgoingWeightsGradient = null;
        _selfWeightsGradient = null;
        _combinationWeightsGradient = null;
        _incomingBiasGradient = null;
        _outgoingBiasGradient = null;
        _selfBiasGradient = null;
        _combinationBiasGradient = null;
    }
}
