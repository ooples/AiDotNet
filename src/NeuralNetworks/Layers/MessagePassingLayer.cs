using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

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
    private readonly Random _random;

    /// <summary>
    /// Message computation network (MLP).
    /// Shape: [messageInputDim, messageFeatures] and [messageFeatures, messageFeatures]
    /// </summary>
    private Tensor<T> _messageWeights1;
    private Tensor<T> _messageWeights2;
    private Tensor<T> _messageBias1;
    private Tensor<T> _messageBias2;

    /// <summary>
    /// Update computation network (GRU-style update).
    /// Shape: [inputFeatures, outputFeatures] and [messageFeatures, outputFeatures]
    /// </summary>
    private Tensor<T> _updateWeights;
    private Tensor<T> _updateMessageWeights;
    private Tensor<T> _updateBias;

    /// <summary>
    /// Reset gate weights (GRU-style).
    /// Shape: [inputFeatures, outputFeatures] and [messageFeatures, outputFeatures]
    /// </summary>
    private Tensor<T> _resetWeights;
    private Tensor<T> _resetMessageWeights;
    private Tensor<T> _resetBias;

    /// <summary>
    /// Edge feature transformation weights (optional).
    /// Shape: [edgeFeatureDim, messageFeatures]
    /// </summary>
    private Tensor<T>? _edgeWeights;

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
    /// Cached hidden activations from message MLP layer 1.
    /// </summary>
    private Tensor<T>? _lastMessageHidden;

    /// <summary>
    /// Cached reset gates.
    /// </summary>
    private Tensor<T>? _lastResetGate;

    /// <summary>
    /// Cached update gates.
    /// </summary>
    private Tensor<T>? _lastUpdateGate;

    /// <summary>
    /// Gradients for parameters.
    /// </summary>
    private Tensor<T>? _messageWeights1Gradient;
    private Tensor<T>? _messageWeights2Gradient;
    private Tensor<T>? _messageBias1Gradient;
    private Tensor<T>? _messageBias2Gradient;
    private Tensor<T>? _updateWeightsGradient;
    private Tensor<T>? _updateMessageWeightsGradient;
    private Tensor<T>? _updateBiasGradient;
    private Tensor<T>? _resetWeightsGradient;
    private Tensor<T>? _resetMessageWeightsGradient;
    private Tensor<T>? _resetBiasGradient;
#pragma warning disable CS0169 // Field is never used - reserved for future edge weight gradient computation
    private Tensor<T>? _edgeWeightsGradient;
#pragma warning restore CS0169

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
        _random = RandomHelper.CreateSecureRandom();

        // Message network: takes concatenated node features (and optionally edge features)
        int messageInputDim = 2 * inputFeatures; // source + target features
        if (useEdgeFeatures && edgeFeatureDim > 0)
        {
            messageInputDim += edgeFeatureDim;
        }

        _messageWeights1 = new Tensor<T>([messageInputDim, _messageFeatures]);
        _messageWeights2 = new Tensor<T>([_messageFeatures, _messageFeatures]);
        _messageBias1 = new Tensor<T>([_messageFeatures]);
        _messageBias2 = new Tensor<T>([_messageFeatures]);

        // Update network (GRU-style)
        _updateWeights = new Tensor<T>([inputFeatures, outputFeatures]);
        _updateMessageWeights = new Tensor<T>([_messageFeatures, outputFeatures]);
        _updateBias = new Tensor<T>([outputFeatures]);

        // Reset gate
        _resetWeights = new Tensor<T>([inputFeatures, outputFeatures]);
        _resetMessageWeights = new Tensor<T>([_messageFeatures, outputFeatures]);
        _resetBias = new Tensor<T>([outputFeatures]);

        // Edge feature transformation
        if (useEdgeFeatures && edgeFeatureDim > 0)
        {
            _edgeWeights = new Tensor<T>([edgeFeatureDim, _messageFeatures]);
        }

        InitializeParameters();
    }

    /// <summary>
    /// Initializes layer parameters using Xavier initialization.
    /// </summary>
    private void InitializeParameters()
    {
        // Initialize message weights
        T scaleMsg1 = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_messageWeights1.Shape[0] + _messageWeights1.Shape[1])));
        InitializeTensor(_messageWeights1, scaleMsg1);

        T scaleMsg2 = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_messageWeights2.Shape[0] + _messageWeights2.Shape[1])));
        InitializeTensor(_messageWeights2, scaleMsg2);

        // Initialize update weights
        T scaleUpd = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_inputFeatures + _outputFeatures)));
        InitializeTensor(_updateWeights, scaleUpd);
        InitializeTensor(_updateMessageWeights, scaleUpd);

        // Initialize reset weights
        InitializeTensor(_resetWeights, scaleUpd);
        InitializeTensor(_resetMessageWeights, scaleUpd);

        // Initialize edge weights if used
        if (_edgeWeights != null)
        {
            T scaleEdge = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_edgeWeights.Shape[0] + _edgeWeights.Shape[1])));
            InitializeTensor(_edgeWeights, scaleEdge);
        }

        // Initialize biases to zero
        _messageBias1.Fill(NumOps.Zero);
        _messageBias2.Fill(NumOps.Zero);
        _updateBias.Fill(NumOps.Zero);
        _resetBias.Fill(NumOps.Zero);
    }

    /// <summary>
    /// Initializes a tensor with scaled random values.
    /// </summary>
    /// <param name="tensor">The tensor to initialize.</param>
    /// <param name="scale">The scale factor for the random values.</param>
    private void InitializeTensor(Tensor<T> tensor, T scale)
    {
        // Create random tensor using Engine operations
        var randomTensor = Tensor<T>.CreateRandom(tensor.Shape);

        // Shift to [-0.5, 0.5] range: randomTensor - 0.5
        var halfTensor = new Tensor<T>(tensor.Shape);
        halfTensor.Fill(NumOps.FromDouble(0.5));
        var shifted = Engine.TensorSubtract(randomTensor, halfTensor);

        // Scale by the scale factor
        var scaled = Engine.TensorMultiplyScalar(shifted, scale);

        // Copy to tensor
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = scaled.GetFlat(i);
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
    /// <param name="edgeFeatures">Tensor of edge features with shape [batch, numNodes * numNodes, edgeFeatureDim].
    /// Edge features are indexed by flattened source-target node indices: edgeIdx = sourceNode * numNodes + targetNode.</param>
    /// <remarks>
    /// <para><b>Shape Contract:</b> The tensor must have shape [batch, numNodes * numNodes, edgeFeatureDim]
    /// where numNodes is the number of nodes in the graph. Edge (i, j) is accessed at index i * numNodes + j.
    /// This dense representation includes slots for all possible edges; non-existent edges are ignored
    /// based on the adjacency matrix during forward computation.</para>
    /// <para><b>Design Note:</b> Dense edge feature storage is used for efficient random access during
    /// message computation. For sparse graphs, consider using attention-based layers (GAT) instead,
    /// which do not require explicit edge features.</para>
    /// </remarks>
    public void SetEdgeFeatures(Tensor<T> edgeFeatures)
    {
        if (!_useEdgeFeatures)
        {
            throw new InvalidOperationException("Layer was not configured to use edge features.");
        }
        if (edgeFeatures.Shape.Length != 3)
        {
            throw new ArgumentException(
                $"Edge features must be a 3D tensor [batch, numEdgeSlots, edgeFeatureDim], but got shape [{string.Join(", ", edgeFeatures.Shape)}].");
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
        _lastMessageHidden = new Tensor<T>([batchSize, numNodes, numNodes, _messageFeatures]);

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
                    var messageInput = new Vector<T>(_messageWeights1.Shape[0]);
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
                    // Dense edge storage: edge (i, j) is stored at index i * numNodes + j
                    // This provides O(1) access for any edge during message computation
                    if (_useEdgeFeatures && _edgeFeatures != null)
                    {
                        int edgeIdx = i * numNodes + j;
                        if (edgeIdx < _edgeFeatures.Shape[1])
                        {
                            for (int f = 0; f < _edgeFeatures.Shape[2]; f++)
                            {
                                messageInput[idx++] = _edgeFeatures[b, edgeIdx, f];
                            }
                        }
                        else
                        {
                            // Edge features tensor is smaller than expected - use zeros
                            for (int f = 0; f < _edgeFeatures.Shape[2]; f++)
                            {
                                messageInput[idx++] = NumOps.Zero;
                            }
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
                        _lastMessageHidden[b, i, j, h] = hidden[h];
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
        _lastResetGate = new Tensor<T>([batchSize, numNodes, _outputFeatures]);
        _lastUpdateGate = new Tensor<T>([batchSize, numNodes, _outputFeatures]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                // Compute reset gate
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

                    _lastResetGate[b, i, f] = Sigmoid(sum);
                }

                // Compute update gate
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

                    _lastUpdateGate[b, i, f] = Sigmoid(sum);
                }

                // Compute new features: h' = (1 - z) * h + z * m
                // where z is update gate, h is input, m is aggregated message
                for (int f = 0; f < _outputFeatures; f++)
                {
                    T oldContribution = NumOps.Multiply(
                        NumOps.Subtract(NumOps.FromDouble(1.0), _lastUpdateGate[b, i, f]),
                        f < _inputFeatures ? input[b, i, f] : NumOps.Zero);

                    T newContribution = NumOps.Multiply(
                        _lastUpdateGate[b, i, f],
                        f < _messageFeatures ? _lastAggregated[b, i, f] : NumOps.Zero);

                    output[b, i, f] = NumOps.Add(oldContribution, newContribution);
                }
            }
        }

        _lastOutput = ApplyActivation(output);
        return _lastOutput;
    }

    /// <summary>
    /// Computes the backward pass for this Message Passing layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to this layer's output.</param>
    /// <returns>The gradient of the loss with respect to this layer's input.</returns>
    /// <remarks>
    /// <para><b>Implementation Note:</b> This backward pass computes actual gradients for all parameters
    /// through the full message passing, aggregation, and GRU-style update operations. The implementation
    /// properly handles gradient flow through:</para>
    /// <list type="bullet">
    /// <item><description>Message network weights and biases (2-layer MLP with ReLU)</description></item>
    /// <item><description>Update gate weights and biases (GRU-style gating)</description></item>
    /// <item><description>Reset gate weights and biases (GRU-style gating)</description></item>
    /// <item><description>Input gradients for proper backpropagation to upstream layers</description></item>
    /// </list>
    /// <para>This enables effective training of the message passing layer with full gradient-based optimization.</para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _adjacencyMatrix == null)
        {
            throw new InvalidOperationException("Forward pass must be called before Backward.");
        }

        if (_lastAggregated == null || _lastMessages == null || _lastMessageHidden == null)
        {
            throw new InvalidOperationException("Forward pass state incomplete.");
        }

        if (_lastResetGate == null || _lastUpdateGate == null)
        {
            throw new InvalidOperationException("Forward pass gate state incomplete.");
        }

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);
        int batchSize = _lastInput.Shape[0];
        int numNodes = _lastInput.Shape[1];

        // Initialize gradients
        _messageWeights1Gradient = new Tensor<T>(_messageWeights1.Shape);
        _messageWeights2Gradient = new Tensor<T>(_messageWeights2.Shape);
        _messageBias1Gradient = new Tensor<T>([_messageFeatures]);
        _messageBias2Gradient = new Tensor<T>([_messageFeatures]);
        _updateWeightsGradient = new Tensor<T>(_updateWeights.Shape);
        _updateMessageWeightsGradient = new Tensor<T>(_updateMessageWeights.Shape);
        _updateBiasGradient = new Tensor<T>([_outputFeatures]);
        _resetWeightsGradient = new Tensor<T>(_resetWeights.Shape);
        _resetMessageWeightsGradient = new Tensor<T>(_resetMessageWeights.Shape);
        _resetBiasGradient = new Tensor<T>([_outputFeatures]);

        _messageWeights1Gradient.Fill(NumOps.Zero);
        _messageWeights2Gradient.Fill(NumOps.Zero);
        _messageBias1Gradient.Fill(NumOps.Zero);
        _messageBias2Gradient.Fill(NumOps.Zero);
        _updateWeightsGradient.Fill(NumOps.Zero);
        _updateMessageWeightsGradient.Fill(NumOps.Zero);
        _updateBiasGradient.Fill(NumOps.Zero);
        _resetWeightsGradient.Fill(NumOps.Zero);
        _resetMessageWeightsGradient.Fill(NumOps.Zero);
        _resetBiasGradient.Fill(NumOps.Zero);

        var inputGradient = new Tensor<T>(_lastInput.Shape);
        inputGradient.Fill(NumOps.Zero);

        // Gradient through aggregated messages
        var aggregatedGradient = new Tensor<T>([batchSize, numNodes, _messageFeatures]);
        aggregatedGradient.Fill(NumOps.Zero);

        // Backward through update gate and feature combination
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                for (int f = 0; f < _outputFeatures; f++)
                {
                    T dOutput = activationGradient[b, i, f];
                    T updateGate = _lastUpdateGate[b, i, f];
                    T oldFeature = f < _inputFeatures ? _lastInput[b, i, f] : NumOps.Zero;
                    T newFeature = f < _messageFeatures ? _lastAggregated[b, i, f] : NumOps.Zero;

                    // Gradient w.r.t. update gate: (newFeature - oldFeature) * dOutput
                    T dUpdateGate = NumOps.Multiply(NumOps.Subtract(newFeature, oldFeature), dOutput);

                    // Gradient of sigmoid: sigmoid * (1 - sigmoid)
                    T sigmoidDerivative = NumOps.Multiply(updateGate, NumOps.Subtract(NumOps.One, updateGate));
                    T dUpdatePreActivation = NumOps.Multiply(dUpdateGate, sigmoidDerivative);

                    // Accumulate gradients for update weights and biases
                    _updateBiasGradient[f] = NumOps.Add(_updateBiasGradient[f], dUpdatePreActivation);

                    for (int k = 0; k < _inputFeatures; k++)
                    {
                        T grad = NumOps.Multiply(_lastInput[b, i, k], dUpdatePreActivation);
                        _updateWeightsGradient[k, f] = NumOps.Add(_updateWeightsGradient[k, f], grad);
                    }

                    for (int k = 0; k < _messageFeatures; k++)
                    {
                        T grad = NumOps.Multiply(_lastAggregated[b, i, k], dUpdatePreActivation);
                        _updateMessageWeightsGradient[k, f] = NumOps.Add(_updateMessageWeightsGradient[k, f], grad);
                    }

                    // Gradient w.r.t. old feature: (1 - updateGate) * dOutput
                    if (f < _inputFeatures)
                    {
                        T dOldFeature = NumOps.Multiply(NumOps.Subtract(NumOps.One, updateGate), dOutput);
                        inputGradient[b, i, f] = NumOps.Add(inputGradient[b, i, f], dOldFeature);
                    }

                    // Gradient w.r.t. new feature (aggregated message): updateGate * dOutput
                    if (f < _messageFeatures)
                    {
                        T dNewFeature = NumOps.Multiply(updateGate, dOutput);
                        aggregatedGradient[b, i, f] = NumOps.Add(aggregatedGradient[b, i, f], dNewFeature);
                    }
                }
            }
        }

        // Backward through aggregation (sum)
        var messageGradient = new Tensor<T>([batchSize, numNodes, numNodes, _messageFeatures]);
        messageGradient.Fill(NumOps.Zero);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                for (int j = 0; j < numNodes; j++)
                {
                    if (!NumOps.Equals(_adjacencyMatrix[b, i, j], NumOps.Zero))
                    {
                        for (int h = 0; h < _messageFeatures; h++)
                        {
                            messageGradient[b, i, j, h] = aggregatedGradient[b, i, h];
                        }
                    }
                }
            }
        }

        // Backward through message MLP layer 2
        var hiddenGradient = new Tensor<T>([batchSize, numNodes, numNodes, _messageFeatures]);
        hiddenGradient.Fill(NumOps.Zero);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                for (int j = 0; j < numNodes; j++)
                {
                    if (NumOps.Equals(_adjacencyMatrix[b, i, j], NumOps.Zero))
                        continue;

                    for (int h = 0; h < _messageFeatures; h++)
                    {
                        T dMessage = messageGradient[b, i, j, h];

                        // Accumulate bias gradient
                        _messageBias2Gradient[h] = NumOps.Add(_messageBias2Gradient[h], dMessage);

                        // Accumulate weight gradients and propagate to hidden
                        for (int k = 0; k < _messageFeatures; k++)
                        {
                            T hidden = _lastMessageHidden[b, i, j, k];
                            T grad = NumOps.Multiply(hidden, dMessage);
                            _messageWeights2Gradient[k, h] = NumOps.Add(_messageWeights2Gradient[k, h], grad);

                            T dHidden = NumOps.Multiply(_messageWeights2[k, h], dMessage);
                            hiddenGradient[b, i, j, k] = NumOps.Add(hiddenGradient[b, i, j, k], dHidden);
                        }
                    }
                }
            }
        }

        // Backward through ReLU and message MLP layer 1
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                for (int j = 0; j < numNodes; j++)
                {
                    if (NumOps.Equals(_adjacencyMatrix[b, i, j], NumOps.Zero))
                        continue;

                    // Build message input for this edge
                    var messageInput = new Vector<T>(_messageWeights1.Shape[0]);
                    int idx = 0;

                    for (int f = 0; f < _inputFeatures; f++)
                    {
                        messageInput[idx++] = _lastInput[b, j, f]; // source
                    }
                    for (int f = 0; f < _inputFeatures; f++)
                    {
                        messageInput[idx++] = _lastInput[b, i, f]; // target
                    }
                    if (_useEdgeFeatures && _edgeFeatures != null)
                    {
                        int edgeIdx = i * numNodes + j;
                        if (edgeIdx < _edgeFeatures.Shape[1])
                        {
                            for (int f = 0; f < _edgeFeatures.Shape[2]; f++)
                            {
                                messageInput[idx++] = _edgeFeatures[b, edgeIdx, f];
                            }
                        }
                        else
                        {
                            for (int f = 0; f < _edgeFeatures.Shape[2]; f++)
                            {
                                messageInput[idx++] = NumOps.Zero;
                            }
                        }
                    }

                    for (int h = 0; h < _messageFeatures; h++)
                    {
                        T dHidden = hiddenGradient[b, i, j, h];
                        T hidden = _lastMessageHidden[b, i, j, h];

                        // ReLU derivative
                        T dPreReLU = NumOps.GreaterThan(hidden, NumOps.Zero) ? dHidden : NumOps.Zero;

                        // Accumulate bias gradient
                        _messageBias1Gradient[h] = NumOps.Add(_messageBias1Gradient[h], dPreReLU);

                        // Accumulate weight gradients and propagate to input
                        for (int k = 0; k < messageInput.Length; k++)
                        {
                            T grad = NumOps.Multiply(messageInput[k], dPreReLU);
                            _messageWeights1Gradient[k, h] = NumOps.Add(_messageWeights1Gradient[k, h], grad);

                            T dInput = NumOps.Multiply(_messageWeights1[k, h], dPreReLU);

                            // Route gradient to appropriate input feature
                            if (k < _inputFeatures)
                            {
                                // Source node gradient
                                inputGradient[b, j, k] = NumOps.Add(inputGradient[b, j, k], dInput);
                            }
                            else if (k < 2 * _inputFeatures)
                            {
                                // Target node gradient
                                inputGradient[b, i, k - _inputFeatures] = NumOps.Add(inputGradient[b, i, k - _inputFeatures], dInput);
                            }
                        }
                    }
                }
            }
        }

        return inputGradient;
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        if (_messageWeights1Gradient == null || _messageWeights2Gradient == null ||
            _updateWeightsGradient == null || _updateMessageWeightsGradient == null ||
            _resetWeightsGradient == null || _resetMessageWeightsGradient == null ||
            _messageBias1Gradient == null || _messageBias2Gradient == null ||
            _updateBiasGradient == null || _resetBiasGradient == null)
        {
            throw new InvalidOperationException("Backward must be called before UpdateParameters.");
        }

        // Update all weights using Engine operations
        var scaledGrad = Engine.TensorMultiplyScalar(_messageWeights1Gradient, learningRate);
        _messageWeights1 = Engine.TensorSubtract(_messageWeights1, scaledGrad);

        scaledGrad = Engine.TensorMultiplyScalar(_messageWeights2Gradient, learningRate);
        _messageWeights2 = Engine.TensorSubtract(_messageWeights2, scaledGrad);

        scaledGrad = Engine.TensorMultiplyScalar(_updateWeightsGradient, learningRate);
        _updateWeights = Engine.TensorSubtract(_updateWeights, scaledGrad);

        scaledGrad = Engine.TensorMultiplyScalar(_updateMessageWeightsGradient, learningRate);
        _updateMessageWeights = Engine.TensorSubtract(_updateMessageWeights, scaledGrad);

        scaledGrad = Engine.TensorMultiplyScalar(_resetWeightsGradient, learningRate);
        _resetWeights = Engine.TensorSubtract(_resetWeights, scaledGrad);

        scaledGrad = Engine.TensorMultiplyScalar(_resetMessageWeightsGradient, learningRate);
        _resetMessageWeights = Engine.TensorSubtract(_resetMessageWeights, scaledGrad);

        // Update biases
        scaledGrad = Engine.TensorMultiplyScalar(_messageBias1Gradient, learningRate);
        _messageBias1 = Engine.TensorSubtract(_messageBias1, scaledGrad);

        scaledGrad = Engine.TensorMultiplyScalar(_messageBias2Gradient, learningRate);
        _messageBias2 = Engine.TensorSubtract(_messageBias2, scaledGrad);

        scaledGrad = Engine.TensorMultiplyScalar(_updateBiasGradient, learningRate);
        _updateBias = Engine.TensorSubtract(_updateBias, scaledGrad);

        scaledGrad = Engine.TensorMultiplyScalar(_resetBiasGradient, learningRate);
        _resetBias = Engine.TensorSubtract(_resetBias, scaledGrad);
    }

    /// <summary>
    /// Gets all trainable parameters as a list of tensors.
    /// </summary>
    /// <returns>A list containing all trainable parameter tensors.</returns>
    public List<Tensor<T>> GetParameterTensors()
    {
        var parameters = new List<Tensor<T>>
        {
            _messageWeights1,
            _messageWeights2,
            _messageBias1,
            _messageBias2,
            _updateWeights,
            _updateMessageWeights,
            _updateBias,
            _resetWeights,
            _resetMessageWeights,
            _resetBias
        };

        if (_edgeWeights != null)
        {
            parameters.Add(_edgeWeights);
        }

        return parameters;
    }

    /// <summary>
    /// Sets all trainable parameters from a list of tensors.
    /// </summary>
    /// <param name="parameters">The list of parameter tensors to set.</param>
    public void SetParameterTensors(List<Tensor<T>> parameters)
    {
        int expectedCount = _edgeWeights != null ? 11 : 10;
        if (parameters.Count != expectedCount)
        {
            throw new ArgumentException($"Expected {expectedCount} parameter tensors, but got {parameters.Count}");
        }

        int idx = 0;
        _messageWeights1 = parameters[idx++];
        _messageWeights2 = parameters[idx++];
        _messageBias1 = parameters[idx++];
        _messageBias2 = parameters[idx++];
        _updateWeights = parameters[idx++];
        _updateMessageWeights = parameters[idx++];
        _updateBias = parameters[idx++];
        _resetWeights = parameters[idx++];
        _resetMessageWeights = parameters[idx++];
        _resetBias = parameters[idx++];

        if (_edgeWeights != null && idx < parameters.Count)
        {
            _edgeWeights = parameters[idx++];
        }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var tensors = GetParameterTensors();
        var arrays = tensors.Select(t => t.ToArray()).ToList();
        int totalLength = arrays.Sum(a => a.Length);

        var result = new Vector<T>(totalLength);
        int index = 0;

        foreach (var array in arrays)
        {
            for (int i = 0; i < array.Length; i++)
            {
                result[index++] = array[i];
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        var tensors = GetParameterTensors();
        int totalLength = tensors.Sum(t => t.Length);

        if (parameters.Length != totalLength)
        {
            throw new ArgumentException($"Expected {totalLength} parameters, but got {parameters.Length}");
        }

        int index = 0;
        var updatedTensors = new List<Tensor<T>>();

        foreach (var tensor in tensors)
        {
            var array = new T[tensor.Length];
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = parameters[index++];
            }

            var newTensor = new Tensor<T>(tensor.Shape);
            for (int i = 0; i < array.Length; i++)
            {
                newTensor[i] = array[i];
            }
            updatedTensors.Add(newTensor);
        }

        SetParameterTensors(updatedTensors);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastMessages = null;
        _lastAggregated = null;
        _lastMessageHidden = null;
        _lastResetGate = null;
        _lastUpdateGate = null;
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

    /// <inheritdoc/>
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc/>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException(
            "MessagePassingLayer does not support computation graph export due to dynamic message computation.");
    }
}
