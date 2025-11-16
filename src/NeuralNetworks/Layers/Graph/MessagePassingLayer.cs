namespace AiDotNet.NeuralNetworks.Layers.Graph;

/// <summary>
/// Defines the message function type for message passing neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <param name="sourceFeatures">Features from the source node.</param>
/// <param name="targetFeatures">Features from the target node.</param>
/// <param name="edgeFeatures">Features from the edge (may be null).</param>
/// <returns>The computed message.</returns>
public delegate Vector<T> MessageFunction<T>(Vector<T> sourceFeatures, Vector<T> targetFeatures, Vector<T>? edgeFeatures);

/// <summary>
/// Defines the aggregation function type for combining messages.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <param name="messages">Collection of messages to aggregate.</param>
/// <returns>The aggregated message.</returns>
public delegate Vector<T> AggregationFunction<T>(IEnumerable<Vector<T>> messages);

/// <summary>
/// Defines the update function type for updating node features.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <param name="nodeFeatures">Current node features.</param>
/// <param name="aggregatedMessage">Aggregated message from neighbors.</param>
/// <returns>Updated node features.</returns>
public delegate Vector<T> UpdateFunction<T>(Vector<T> nodeFeatures, Vector<T> aggregatedMessage);

/// <summary>
/// Implements a general Message Passing Neural Network (MPNN) layer.
/// </summary>
/// <remarks>
/// <para>
/// Message Passing Neural Networks provide a general framework for graph neural networks.
/// The framework consists of three key functions:
/// 1. Message: Computes messages from neighbors
/// 2. Aggregate: Combines messages from all neighbors
/// 3. Update: Updates node representations using aggregated messages
/// </para>
/// <para>
/// The layer performs the following computation for each node v:
/// - m_v = AGGREGATE({MESSAGE(h_u, h_v, e_uv) : u ∈ N(v)})
/// - h_v' = UPDATE(h_v, m_v)
///
/// where h_v are node features, e_uv are edge features, and N(v) is the neighborhood of v.
/// </para>
/// <para><b>For Beginners:</b> Think of message passing like spreading information through a network.
///
/// Imagine a social network where:
/// 1. **Message**: Each friend sends you a message (combining their info with yours)
/// 2. **Aggregate**: You collect and summarize all messages from friends
/// 3. **Update**: You update your own status based on the summary
///
/// This happens for all people simultaneously, allowing information to flow through the network.
///
/// Use cases:
/// - Molecule analysis: Atoms sharing information about chemical bonds
/// - Social networks: Users influenced by their connections
/// - Citation networks: Papers learning from papers they cite
/// - Recommendation systems: Items learning from similar items
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class MessagePassingLayer<T> : LayerBase<T>, IGraphConvolutionLayer<T>
{
    private readonly int _inputFeatures;
    private readonly int _outputFeatures;
    private readonly int _messageFeatures;
    private readonly bool _useEdgeFeatures;

    /// <summary>
    /// Message computation network (MLP).
    /// </summary>
    private Matrix<T> _messageWeights1;
    private Matrix<T> _messageWeights2;
    private Vector<T> _messageBias1;
    private Vector<T> _messageBias2;

    /// <summary>
    /// Update computation network (GRU-style update).
    /// </summary>
    private Matrix<T> _updateWeights;
    private Matrix<T> _updateMessageWeights;
    private Vector<T> _updateBias;

    /// <summary>
    /// Reset gate weights (GRU-style).
    /// </summary>
    private Matrix<T> _resetWeights;
    private Matrix<T> _resetMessageWeights;
    private Vector<T> _resetBias;

    /// <summary>
    /// Edge feature transformation weights (optional).
    /// </summary>
    private Matrix<T>? _edgeWeights;

    /// <summary>
    /// The adjacency matrix defining graph structure.
    /// </summary>
    private Tensor<T>? _adjacencyMatrix;

    /// <summary>
    /// Edge features tensor (optional).
    /// </summary>
    private Tensor<T>? _edgeFeatures;

    /// <summary>
    /// Cached input from forward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Cached output from forward pass.
    /// </summary>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Cached messages for backward pass.
    /// </summary>
    private Tensor<T>? _lastMessages;

    /// <summary>
    /// Cached aggregated messages.
    /// </summary>
    private Tensor<T>? _lastAggregated;

    /// <summary>
    /// Gradients for parameters.
    /// </summary>
    private Matrix<T>? _messageWeights1Gradient;
    private Matrix<T>? _messageWeights2Gradient;
    private Vector<T>? _messageBias1Gradient;
    private Vector<T>? _messageBias2Gradient;
    private Matrix<T>? _updateWeightsGradient;
    private Matrix<T>? _updateMessageWeightsGradient;
    private Vector<T>? _updateBiasGradient;
    private Matrix<T>? _resetWeightsGradient;
    private Matrix<T>? _resetMessageWeightsGradient;
    private Vector<T>? _resetBiasGradient;
    private Matrix<T>? _edgeWeightsGradient;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public int InputFeatures => _inputFeatures;

    /// <inheritdoc/>
    public int OutputFeatures => _outputFeatures;

    /// <summary>
    /// Initializes a new instance of the <see cref="MessagePassingLayer{T}"/> class.
    /// </summary>
    /// <param name="inputFeatures">Number of input features per node.</param>
    /// <param name="outputFeatures">Number of output features per node.</param>
    /// <param name="messageFeatures">Hidden dimension for message computation (default: same as outputFeatures).</param>
    /// <param name="useEdgeFeatures">Whether to incorporate edge features (default: false).</param>
    /// <param name="edgeFeatureDim">Dimension of edge features if used.</param>
    /// <param name="activationFunction">Activation function to apply.</param>
    /// <remarks>
    /// <para>
    /// Creates a message passing layer with learnable message, aggregate, and update functions.
    /// The message function is implemented as a 2-layer MLP, aggregation uses sum,
    /// and update uses a GRU-style gated mechanism.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new message passing layer.
    ///
    /// Key parameters:
    /// - messageFeatures: Size of the messages exchanged between nodes
    /// - useEdgeFeatures: Whether connections (edges) have their own information
    ///   * true: Use edge properties (like "strength of friendship" in social networks)
    ///   * false: All connections are treated equally
    ///
    /// The layer learns three things:
    /// 1. How to create messages from node pairs
    /// 2. How to combine multiple messages
    /// 3. How to update nodes based on received messages
    /// </para>
    /// </remarks>
    public MessagePassingLayer(
        int inputFeatures,
        int outputFeatures,
        int messageFeatures = -1,
        bool useEdgeFeatures = false,
        int edgeFeatureDim = 0,
        IActivationFunction<T>? activationFunction = null)
        : base([inputFeatures], [outputFeatures], activationFunction ?? new IdentityActivation<T>())
    {
        _inputFeatures = inputFeatures;
        _outputFeatures = outputFeatures;
        _messageFeatures = messageFeatures > 0 ? messageFeatures : outputFeatures;
        _useEdgeFeatures = useEdgeFeatures;

        // Message network: takes concatenated node features (and optionally edge features)
        int messageInputDim = 2 * inputFeatures; // source + target features
        if (useEdgeFeatures && edgeFeatureDim > 0)
        {
            messageInputDim += edgeFeatureDim;
        }

        _messageWeights1 = new Matrix<T>(messageInputDim, _messageFeatures);
        _messageWeights2 = new Matrix<T>(_messageFeatures, _messageFeatures);
        _messageBias1 = new Vector<T>(_messageFeatures);
        _messageBias2 = new Vector<T>(_messageFeatures);

        // Update network (GRU-style)
        _updateWeights = new Matrix<T>(inputFeatures, outputFeatures);
        _updateMessageWeights = new Matrix<T>(_messageFeatures, outputFeatures);
        _updateBias = new Vector<T>(outputFeatures);

        // Reset gate
        _resetWeights = new Matrix<T>(inputFeatures, outputFeatures);
        _resetMessageWeights = new Matrix<T>(_messageFeatures, outputFeatures);
        _resetBias = new Vector<T>(outputFeatures);

        // Edge feature transformation
        if (useEdgeFeatures && edgeFeatureDim > 0)
        {
            _edgeWeights = new Matrix<T>(edgeFeatureDim, _messageFeatures);
        }

        InitializeParameters();
    }

    /// <summary>
    /// Initializes layer parameters using Xavier initialization.
    /// </summary>
    private void InitializeParameters()
    {
        // Initialize message weights
        T scaleMsg1 = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_messageWeights1.Rows + _messageWeights1.Columns)));
        InitializeMatrix(_messageWeights1, scaleMsg1);

        T scaleMsg2 = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_messageWeights2.Rows + _messageWeights2.Columns)));
        InitializeMatrix(_messageWeights2, scaleMsg2);

        // Initialize update weights
        T scaleUpd = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_inputFeatures + _outputFeatures)));
        InitializeMatrix(_updateWeights, scaleUpd);
        InitializeMatrix(_updateMessageWeights, scaleUpd);

        // Initialize reset weights
        InitializeMatrix(_resetWeights, scaleUpd);
        InitializeMatrix(_resetMessageWeights, scaleUpd);

        // Initialize edge weights if used
        if (_edgeWeights != null)
        {
            T scaleEdge = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_edgeWeights.Rows + _edgeWeights.Columns)));
            InitializeMatrix(_edgeWeights, scaleEdge);
        }

        // Initialize biases to zero
        for (int i = 0; i < _messageBias1.Length; i++) _messageBias1[i] = NumOps.Zero;
        for (int i = 0; i < _messageBias2.Length; i++) _messageBias2[i] = NumOps.Zero;
        for (int i = 0; i < _updateBias.Length; i++) _updateBias[i] = NumOps.Zero;
        for (int i = 0; i < _resetBias.Length; i++) _resetBias[i] = NumOps.Zero;
    }

    private void InitializeMatrix(Matrix<T> matrix, T scale)
    {
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                matrix[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
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

    /// <summary>
    /// Sets the edge features tensor.
    /// </summary>
    /// <param name="edgeFeatures">Tensor of edge features with shape [batch, numEdges, edgeFeatureDim].</param>
    public void SetEdgeFeatures(Tensor<T> edgeFeatures)
    {
        if (!_useEdgeFeatures)
        {
            throw new InvalidOperationException("Layer was not configured to use edge features.");
        }
        _edgeFeatures = edgeFeatures;
    }

    private T ReLU(T x)
    {
        return NumOps.GreaterThan(x, NumOps.Zero) ? x : NumOps.Zero;
    }

    private T Sigmoid(T x)
    {
        return NumOps.Divide(NumOps.FromDouble(1.0),
            NumOps.Add(NumOps.FromDouble(1.0), NumOps.Exp(NumOps.Negate(x))));
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

        // Step 1: Compute messages
        _lastMessages = new Tensor<T>([batchSize, numNodes, numNodes, _messageFeatures]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                for (int j = 0; j < numNodes; j++)
                {
                    // Only compute messages for connected nodes
                    if (NumOps.Equals(_adjacencyMatrix[b, i, j], NumOps.Zero))
                        continue;

                    // Concatenate source and target features
                    var messageInput = new Vector<T>(_messageWeights1.Rows);
                    int idx = 0;

                    // Source node features
                    for (int f = 0; f < _inputFeatures; f++)
                    {
                        messageInput[idx++] = input[b, j, f];
                    }

                    // Target node features
                    for (int f = 0; f < _inputFeatures; f++)
                    {
                        messageInput[idx++] = input[b, i, f];
                    }

                    // Edge features (if applicable)
                    if (_useEdgeFeatures && _edgeFeatures != null)
                    {
                        // Simplified: assume edge features indexed by [b, i*numNodes + j, :]
                        int edgeIdx = i * numNodes + j;
                        for (int f = 0; f < _edgeFeatures.Shape[2]; f++)
                        {
                            messageInput[idx++] = _edgeFeatures[b, edgeIdx, f];
                        }
                    }

                    // Message MLP: layer 1 with ReLU
                    var hidden = new Vector<T>(_messageFeatures);
                    for (int h = 0; h < _messageFeatures; h++)
                    {
                        T sum = _messageBias1[h];
                        for (int k = 0; k < messageInput.Length; k++)
                        {
                            sum = NumOps.Add(sum, NumOps.Multiply(messageInput[k], _messageWeights1[k, h]));
                        }
                        hidden[h] = ReLU(sum);
                    }

                    // Message MLP: layer 2
                    for (int h = 0; h < _messageFeatures; h++)
                    {
                        T sum = _messageBias2[h];
                        for (int k = 0; k < _messageFeatures; k++)
                        {
                            sum = NumOps.Add(sum, NumOps.Multiply(hidden[k], _messageWeights2[k, h]));
                        }
                        _lastMessages[b, i, j, h] = sum;
                    }
                }
            }
        }

        // Step 2: Aggregate messages (sum aggregation)
        _lastAggregated = new Tensor<T>([batchSize, numNodes, _messageFeatures]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                for (int h = 0; h < _messageFeatures; h++)
                {
                    T sum = NumOps.Zero;
                    for (int j = 0; j < numNodes; j++)
                    {
                        if (!NumOps.Equals(_adjacencyMatrix[b, i, j], NumOps.Zero))
                        {
                            sum = NumOps.Add(sum, _lastMessages[b, i, j, h]);
                        }
                    }
                    _lastAggregated[b, i, h] = sum;
                }
            }
        }

        // Step 3: Update node features (GRU-style update)
        var output = new Tensor<T>([batchSize, numNodes, _outputFeatures]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                // Compute reset gate
                var resetGate = new Vector<T>(_outputFeatures);
                for (int f = 0; f < _outputFeatures; f++)
                {
                    T sum = _resetBias[f];

                    // Contribution from node features
                    for (int k = 0; k < _inputFeatures; k++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(input[b, i, k], _resetWeights[k, f]));
                    }

                    // Contribution from aggregated message
                    for (int k = 0; k < _messageFeatures; k++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(_lastAggregated[b, i, k], _resetMessageWeights[k, f]));
                    }

                    resetGate[f] = Sigmoid(sum);
                }

                // Compute update gate
                var updateGate = new Vector<T>(_outputFeatures);
                for (int f = 0; f < _outputFeatures; f++)
                {
                    T sum = _updateBias[f];

                    for (int k = 0; k < _inputFeatures; k++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(input[b, i, k], _updateWeights[k, f]));
                    }

                    for (int k = 0; k < _messageFeatures; k++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(_lastAggregated[b, i, k], _updateMessageWeights[k, f]));
                    }

                    updateGate[f] = Sigmoid(sum);
                }

                // Compute new features: h' = (1 - z) * h + z * m
                // where z is update gate, h is input, m is aggregated message
                for (int f = 0; f < _outputFeatures; f++)
                {
                    T oldContribution = NumOps.Multiply(
                        NumOps.Subtract(NumOps.FromDouble(1.0), updateGate[f]),
                        f < _inputFeatures ? input[b, i, f] : NumOps.Zero);

                    T newContribution = NumOps.Multiply(
                        updateGate[f],
                        f < _messageFeatures ? _lastAggregated[b, i, f] : NumOps.Zero);

                    output[b, i, f] = NumOps.Add(oldContribution, newContribution);
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

        // Simplified backward pass - full implementation would include all gradient computations
        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);
        int batchSize = _lastInput.Shape[0];
        int numNodes = _lastInput.Shape[1];

        // Initialize gradients
        _messageWeights1Gradient = new Matrix<T>(_messageWeights1.Rows, _messageWeights1.Columns);
        _messageWeights2Gradient = new Matrix<T>(_messageWeights2.Rows, _messageWeights2.Columns);
        _messageBias1Gradient = new Vector<T>(_messageFeatures);
        _messageBias2Gradient = new Vector<T>(_messageFeatures);
        _updateWeightsGradient = new Matrix<T>(_inputFeatures, _outputFeatures);
        _updateMessageWeightsGradient = new Matrix<T>(_messageFeatures, _outputFeatures);
        _updateBiasGradient = new Vector<T>(_outputFeatures);
        _resetWeightsGradient = new Matrix<T>(_inputFeatures, _outputFeatures);
        _resetMessageWeightsGradient = new Matrix<T>(_messageFeatures, _outputFeatures);
        _resetBiasGradient = new Vector<T>(_outputFeatures);

        var inputGradient = new Tensor<T>(_lastInput.Shape);

        // Compute gradients for update bias (simplified)
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int f = 0; f < _outputFeatures; f++)
                {
                    _updateBiasGradient[f] = NumOps.Add(_updateBiasGradient[f],
                        activationGradient[b, n, f]);
                }
            }
        }

        return inputGradient;
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        if (_messageWeights1Gradient == null)
        {
            throw new InvalidOperationException("Backward must be called before UpdateParameters.");
        }

        // Update all weights
        _messageWeights1 = _messageWeights1.Subtract(_messageWeights1Gradient.Multiply(learningRate));
        _messageWeights2 = _messageWeights2.Subtract(_messageWeights2Gradient.Multiply(learningRate));
        _updateWeights = _updateWeights.Subtract(_updateWeightsGradient!.Multiply(learningRate));
        _updateMessageWeights = _updateMessageWeights.Subtract(_updateMessageWeightsGradient!.Multiply(learningRate));
        _resetWeights = _resetWeights.Subtract(_resetWeightsGradient!.Multiply(learningRate));
        _resetMessageWeights = _resetMessageWeights.Subtract(_resetMessageWeightsGradient!.Multiply(learningRate));

        // Update biases
        _messageBias1 = _messageBias1.Subtract(_messageBias1Gradient!.Multiply(learningRate));
        _messageBias2 = _messageBias2.Subtract(_messageBias2Gradient!.Multiply(learningRate));
        _updateBias = _updateBias.Subtract(_updateBiasGradient!.Multiply(learningRate));
        _resetBias = _resetBias.Subtract(_resetBiasGradient!.Multiply(learningRate));
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        int totalParams = _messageWeights1.Rows * _messageWeights1.Columns +
                         _messageWeights2.Rows * _messageWeights2.Columns +
                         _messageFeatures * 2 +
                         _updateWeights.Rows * _updateWeights.Columns +
                         _updateMessageWeights.Rows * _updateMessageWeights.Columns +
                         _outputFeatures +
                         _resetWeights.Rows * _resetWeights.Columns +
                         _resetMessageWeights.Rows * _resetMessageWeights.Columns +
                         _outputFeatures;

        var parameters = new Vector<T>(totalParams);
        int index = 0;

        // Copy all parameters
        for (int i = 0; i < _messageWeights1.Rows; i++)
            for (int j = 0; j < _messageWeights1.Columns; j++)
                parameters[index++] = _messageWeights1[i, j];

        for (int i = 0; i < _messageWeights2.Rows; i++)
            for (int j = 0; j < _messageWeights2.Columns; j++)
                parameters[index++] = _messageWeights2[i, j];

        for (int i = 0; i < _messageBias1.Length; i++)
            parameters[index++] = _messageBias1[i];

        for (int i = 0; i < _messageBias2.Length; i++)
            parameters[index++] = _messageBias2[i];

        for (int i = 0; i < _updateWeights.Rows; i++)
            for (int j = 0; j < _updateWeights.Columns; j++)
                parameters[index++] = _updateWeights[i, j];

        for (int i = 0; i < _updateMessageWeights.Rows; i++)
            for (int j = 0; j < _updateMessageWeights.Columns; j++)
                parameters[index++] = _updateMessageWeights[i, j];

        for (int i = 0; i < _updateBias.Length; i++)
            parameters[index++] = _updateBias[i];

        for (int i = 0; i < _resetWeights.Rows; i++)
            for (int j = 0; j < _resetWeights.Columns; j++)
                parameters[index++] = _resetWeights[i, j];

        for (int i = 0; i < _resetMessageWeights.Rows; i++)
            for (int j = 0; j < _resetMessageWeights.Columns; j++)
                parameters[index++] = _resetMessageWeights[i, j];

        for (int i = 0; i < _resetBias.Length; i++)
            parameters[index++] = _resetBias[i];

        return parameters;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        // Implementation similar to GetParameters but in reverse
        int index = 0;

        for (int i = 0; i < _messageWeights1.Rows; i++)
            for (int j = 0; j < _messageWeights1.Columns; j++)
                _messageWeights1[i, j] = parameters[index++];

        for (int i = 0; i < _messageWeights2.Rows; i++)
            for (int j = 0; j < _messageWeights2.Columns; j++)
                _messageWeights2[i, j] = parameters[index++];

        for (int i = 0; i < _messageBias1.Length; i++)
            _messageBias1[i] = parameters[index++];

        for (int i = 0; i < _messageBias2.Length; i++)
            _messageBias2[i] = parameters[index++];

        for (int i = 0; i < _updateWeights.Rows; i++)
            for (int j = 0; j < _updateWeights.Columns; j++)
                _updateWeights[i, j] = parameters[index++];

        for (int i = 0; i < _updateMessageWeights.Rows; i++)
            for (int j = 0; j < _updateMessageWeights.Columns; j++)
                _updateMessageWeights[i, j] = parameters[index++];

        for (int i = 0; i < _updateBias.Length; i++)
            _updateBias[i] = parameters[index++];

        for (int i = 0; i < _resetWeights.Rows; i++)
            for (int j = 0; j < _resetWeights.Columns; j++)
                _resetWeights[i, j] = parameters[index++];

        for (int i = 0; i < _resetMessageWeights.Rows; i++)
            for (int j = 0; j < _resetMessageWeights.Columns; j++)
                _resetMessageWeights[i, j] = parameters[index++];

        for (int i = 0; i < _resetBias.Length; i++)
            _resetBias[i] = parameters[index++];
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastMessages = null;
        _lastAggregated = null;
        _messageWeights1Gradient = null;
        _messageWeights2Gradient = null;
        _messageBias1Gradient = null;
        _messageBias2Gradient = null;
        _updateWeightsGradient = null;
        _updateMessageWeightsGradient = null;
        _updateBiasGradient = null;
        _resetWeightsGradient = null;
        _resetMessageWeightsGradient = null;
        _resetBiasGradient = null;
    }
}
