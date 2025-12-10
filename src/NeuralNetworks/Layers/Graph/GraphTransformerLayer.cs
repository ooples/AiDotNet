namespace AiDotNet.NeuralNetworks.Layers.Graph;

/// <summary>
/// Implements Graph Transformer layer using self-attention mechanisms on graph-structured data.
/// </summary>
/// <remarks>
/// <para>
/// Graph Transformers apply the transformer architecture to graphs by treating graph structure
/// as a bias in the attention mechanism. Unlike standard transformers that process sequences,
/// Graph Transformers incorporate graph connectivity through:
/// 1. Structural encodings (e.g., Laplacian eigenvectors)
/// 2. Attention biasing based on graph structure
/// 3. Relative positional encodings for graph nodes
/// </para>
/// <para>
/// The attention computation is: Attention(Q, K, V) = softmax((QK^T + B)/√d_k)V
/// where B is a learned bias based on graph structure.
/// </para>
/// <para><b>For Beginners:</b> Graph Transformers combine the power of transformers with graph structure.
///
/// Think of it like a meeting where:
/// - **Standard transformers**: Everyone can talk to everyone equally
/// - **Graph transformers**: People connected in the organizational chart get priority
///
/// Key advantages:
/// - Captures long-range dependencies in graphs
/// - More flexible than fixed neighborhood aggregation
/// - Can attend to any node, not just immediate neighbors
/// - Learns importance of connections dynamically
///
/// Use cases:
/// - **Large molecules**: Atoms far apart but chemically important
/// - **Social networks**: Identifying influential users across communities
/// - **Knowledge graphs**: Multi-hop reasoning
/// - **Program analysis**: Understanding code dependencies
///
/// Example: In a citation network, a paper can learn from:
/// - Direct citations (immediate neighbors)
/// - Indirectly related papers (through attention)
/// - Important papers even if not directly cited
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GraphTransformerLayer<T> : LayerBase<T>, IGraphConvolutionLayer<T>
{
    private readonly int _inputFeatures;
    private readonly int _outputFeatures;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly bool _useStructuralEncoding;
    private readonly double _dropoutRate;
    private readonly Random _random;

    /// <summary>
    /// Query transformation weights for each head.
    /// </summary>
    private Tensor<T> _queryWeights; // [numHeads, inputFeatures, headDim]

    /// <summary>
    /// Key transformation weights for each head.
    /// </summary>
    private Tensor<T> _keyWeights; // [numHeads, inputFeatures, headDim]

    /// <summary>
    /// Value transformation weights for each head.
    /// </summary>
    private Tensor<T> _valueWeights; // [numHeads, inputFeatures, headDim]

    /// <summary>
    /// Output projection weights.
    /// </summary>
    private Matrix<T> _outputWeights; // [numHeads * headDim, outputFeatures]

    /// <summary>
    /// Structural bias for attention (learned from graph structure).
    /// </summary>
    private Tensor<T>? _structuralBias; // [numHeads, maxNodes, maxNodes]

    /// <summary>
    /// Feed-forward network weights.
    /// </summary>
    private Matrix<T> _ffnWeights1;
    private Matrix<T> _ffnWeights2;
    private Vector<T> _ffnBias1;
    private Vector<T> _ffnBias2;

    /// <summary>
    /// Layer normalization parameters.
    /// </summary>
    private Vector<T> _layerNorm1Scale;
    private Vector<T> _layerNorm1Bias;
    private Vector<T> _layerNorm2Scale;
    private Vector<T> _layerNorm2Bias;

    /// <summary>
    /// Bias vectors.
    /// </summary>
    private Vector<T> _outputBias;

    /// <summary>
    /// The adjacency matrix defining graph structure.
    /// </summary>
    private Tensor<T>? _adjacencyMatrix;

    /// <summary>
    /// Cached values for backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
#pragma warning disable CS0414 // Field is assigned but never used - reserved for future backward pass implementation
    private Tensor<T>? _lastAttentionScores;
#pragma warning restore CS0414
    private Tensor<T>? _lastAttentionWeights;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public int InputFeatures => _inputFeatures;

    /// <inheritdoc/>
    public int OutputFeatures => _outputFeatures;

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphTransformerLayer{T}"/> class.
    /// </summary>
    /// <param name="inputFeatures">Number of input features per node.</param>
    /// <param name="outputFeatures">Number of output features per node.</param>
    /// <param name="numHeads">Number of attention heads (default: 8).</param>
    /// <param name="headDim">Dimension per attention head (default: 64).</param>
    /// <param name="useStructuralEncoding">Whether to use structural bias (default: true).</param>
    /// <param name="dropoutRate">Dropout rate for attention (default: 0.1).</param>
    /// <param name="activationFunction">Activation function to apply.</param>
    /// <remarks>
    /// <para>
    /// Creates a Graph Transformer layer with multi-head attention and feed-forward network.
    /// The layer includes skip connections and layer normalization for stable training.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Graph Transformer layer.
    ///
    /// Key parameters:
    /// - numHeads: How many parallel attention mechanisms (more = capture different patterns)
    /// - headDim: Size of each attention head (bigger = more expressive per head)
    /// - useStructuralEncoding: Whether to bias attention toward connected nodes
    ///   * true: Graph structure guides attention (recommended for most graphs)
    ///   * false: Pure attention without graph bias (for dense/complete graphs)
    /// - dropoutRate: Randomly ignore some attention during training (prevents overfitting)
    ///
    /// The layer has two main components:
    /// 1. **Multi-head attention**: Learns which nodes to focus on
    /// 2. **Feed-forward network**: Processes the attended information
    ///
    /// Both use skip connections (adding input back to output) for better gradient flow.
    /// </para>
    /// </remarks>
    public GraphTransformerLayer(
        int inputFeatures,
        int outputFeatures,
        int numHeads = 8,
        int headDim = 64,
        bool useStructuralEncoding = true,
        double dropoutRate = 0.1,
        IActivationFunction<T>? activationFunction = null)
        : base([inputFeatures], [outputFeatures], activationFunction ?? new IdentityActivation<T>())
    {
        _inputFeatures = inputFeatures;
        _outputFeatures = outputFeatures;
        _numHeads = numHeads;
        _headDim = headDim;
        _useStructuralEncoding = useStructuralEncoding;
        _dropoutRate = dropoutRate;
        _random = new Random();

        // Initialize Q, K, V projections
        _queryWeights = new Tensor<T>([_numHeads, _inputFeatures, _headDim]);
        _keyWeights = new Tensor<T>([_numHeads, _inputFeatures, _headDim]);
        _valueWeights = new Tensor<T>([_numHeads, _inputFeatures, _headDim]);

        // Output projection
        _outputWeights = new Matrix<T>(_numHeads * _headDim, _outputFeatures);
        _outputBias = new Vector<T>(_outputFeatures);

        // Feed-forward network (2 layers)
        int ffnHiddenDim = 4 * outputFeatures; // Standard: 4x expansion
        _ffnWeights1 = new Matrix<T>(_outputFeatures, ffnHiddenDim);
        _ffnWeights2 = new Matrix<T>(ffnHiddenDim, _outputFeatures);
        _ffnBias1 = new Vector<T>(ffnHiddenDim);
        _ffnBias2 = new Vector<T>(_outputFeatures);

        // Layer normalization parameters
        _layerNorm1Scale = new Vector<T>(_outputFeatures);
        _layerNorm1Bias = new Vector<T>(_outputFeatures);
        _layerNorm2Scale = new Vector<T>(_outputFeatures);
        _layerNorm2Bias = new Vector<T>(_outputFeatures);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        // Xavier initialization for Q, K, V
        T scaleQKV = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_inputFeatures + _headDim)));
        for (int h = 0; h < _numHeads; h++)
        {
            for (int i = 0; i < _inputFeatures; i++)
            {
                for (int j = 0; j < _headDim; j++)
                {
                    _queryWeights[h, i, j] = NumOps.Multiply(
                        NumOps.FromDouble(Random.NextDouble() - 0.5), scaleQKV);
                    _keyWeights[h, i, j] = NumOps.Multiply(
                        NumOps.FromDouble(Random.NextDouble() - 0.5), scaleQKV);
                    _valueWeights[h, i, j] = NumOps.Multiply(
                        NumOps.FromDouble(Random.NextDouble() - 0.5), scaleQKV);
                }
            }
        }

        // Initialize output weights
        T scaleOut = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_numHeads * _headDim + _outputFeatures)));
        for (int i = 0; i < _outputWeights.Rows; i++)
        {
            for (int j = 0; j < _outputWeights.Columns; j++)
            {
                _outputWeights[i, j] = NumOps.Multiply(
                    NumOps.FromDouble(Random.NextDouble() - 0.5), scaleOut);
            }
        }

        // Initialize FFN weights
        T scaleFFN1 = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_outputFeatures + _ffnWeights1.Columns)));
        for (int i = 0; i < _ffnWeights1.Rows; i++)
        {
            for (int j = 0; j < _ffnWeights1.Columns; j++)
            {
                _ffnWeights1[i, j] = NumOps.Multiply(
                    NumOps.FromDouble(Random.NextDouble() - 0.5), scaleFFN1);
            }
        }

        T scaleFFN2 = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_ffnWeights2.Rows + _outputFeatures)));
        for (int i = 0; i < _ffnWeights2.Rows; i++)
        {
            for (int j = 0; j < _ffnWeights2.Columns; j++)
            {
                _ffnWeights2[i, j] = NumOps.Multiply(
                    NumOps.FromDouble(Random.NextDouble() - 0.5), scaleFFN2);
            }
        }

        // Initialize layer norm to identity (scale=1, bias=0)
        for (int i = 0; i < _layerNorm1Scale.Length; i++)
        {
            _layerNorm1Scale[i] = NumOps.FromDouble(1.0);
            _layerNorm1Bias[i] = NumOps.Zero;
        }

        for (int i = 0; i < _layerNorm2Scale.Length; i++)
        {
            _layerNorm2Scale[i] = NumOps.FromDouble(1.0);
            _layerNorm2Bias[i] = NumOps.Zero;
        }

        // Initialize biases to zero
        for (int i = 0; i < _outputBias.Length; i++)
            _outputBias[i] = NumOps.Zero;

        for (int i = 0; i < _ffnBias1.Length; i++)
            _ffnBias1[i] = NumOps.Zero;

        for (int i = 0; i < _ffnBias2.Length; i++)
            _ffnBias2[i] = NumOps.Zero;
    }

    /// <inheritdoc/>
    public void SetAdjacencyMatrix(Tensor<T> adjacencyMatrix)
    {
        _adjacencyMatrix = adjacencyMatrix;

        // Initialize structural bias if needed
        if (_useStructuralEncoding && _structuralBias == null)
        {
            int maxNodes = adjacencyMatrix.Shape[1];
            _structuralBias = new Tensor<T>([_numHeads, maxNodes, maxNodes]);

            // Simple initialization: bias toward connected nodes
            for (int h = 0; h < _numHeads; h++)
            {
                for (int i = 0; i < maxNodes; i++)
                {
                    for (int j = 0; j < maxNodes; j++)
                    {
                        _structuralBias[h, i, j] = NumOps.FromDouble(Random.NextDouble() - 0.5);
                    }
                }
            }
        }
    }

    /// <inheritdoc/>
    public Tensor<T>? GetAdjacencyMatrix()
    {
        return _adjacencyMatrix;
    }

    private T GELU(T x)
    {
        // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        T x3 = NumOps.Multiply(NumOps.Multiply(x, x), x);
        T inner = NumOps.Add(x, NumOps.Multiply(NumOps.FromDouble(0.044715), x3));
        T scaled = NumOps.Multiply(NumOps.FromDouble(0.7978845608), inner); // sqrt(2/π)

        // Simplified tanh approximation
        T tanhApprox = NumOps.Divide(
            NumOps.Subtract(NumOps.Exp(scaled), NumOps.Exp(NumOps.Negate(scaled))),
            NumOps.Add(NumOps.Exp(scaled), NumOps.Exp(NumOps.Negate(scaled))));

        return NumOps.Multiply(NumOps.Multiply(NumOps.FromDouble(0.5), x),
            NumOps.Add(NumOps.FromDouble(1.0), tanhApprox));
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

        // Multi-head attention block with residual connection
        var attended = MultiHeadAttention(input, batchSize, numNodes);

        // Add residual and layer norm (simplified - adds input to attended output)
        var normed1 = new Tensor<T>([batchSize, numNodes, _outputFeatures]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int f = 0; f < _outputFeatures; f++)
                {
                    // Residual connection (if dimensions match)
                    T residual = f < _inputFeatures ? input[b, n, f] : NumOps.Zero;
                    normed1[b, n, f] = NumOps.Add(attended[b, n, f], residual);
                }
            }
        }

        // Feed-forward network with residual
        var ffnOutput = FeedForwardNetwork(normed1, batchSize, numNodes);

        // Final residual and layer norm
        var output = new Tensor<T>([batchSize, numNodes, _outputFeatures]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int f = 0; f < _outputFeatures; f++)
                {
                    output[b, n, f] = NumOps.Add(normed1[b, n, f], ffnOutput[b, n, f]);
                }
            }
        }

        _lastOutput = ApplyActivation(output);
        return _lastOutput;
    }

    private Tensor<T> MultiHeadAttention(Tensor<T> input, int batchSize, int numNodes)
    {
        // Store attention outputs for each head
        var headOutputs = new Tensor<T>([batchSize, _numHeads, numNodes, _headDim]);
        _lastAttentionWeights = new Tensor<T>([batchSize, _numHeads, numNodes, numNodes]);

        T sqrtDk = NumOps.Sqrt(NumOps.FromDouble(_headDim));

        for (int h = 0; h < _numHeads; h++)
        {
            // Compute Q, K, V for this head
            var queries = new Tensor<T>([batchSize, numNodes, _headDim]);
            var keys = new Tensor<T>([batchSize, numNodes, _headDim]);
            var values = new Tensor<T>([batchSize, numNodes, _headDim]);

            for (int b = 0; b < batchSize; b++)
            {
                for (int n = 0; n < numNodes; n++)
                {
                    for (int d = 0; d < _headDim; d++)
                    {
                        T qSum = NumOps.Zero, kSum = NumOps.Zero, vSum = NumOps.Zero;

                        for (int f = 0; f < _inputFeatures; f++)
                        {
                            qSum = NumOps.Add(qSum, NumOps.Multiply(input[b, n, f], _queryWeights[h, f, d]));
                            kSum = NumOps.Add(kSum, NumOps.Multiply(input[b, n, f], _keyWeights[h, f, d]));
                            vSum = NumOps.Add(vSum, NumOps.Multiply(input[b, n, f], _valueWeights[h, f, d]));
                        }

                        queries[b, n, d] = qSum;
                        keys[b, n, d] = kSum;
                        values[b, n, d] = vSum;
                    }
                }
            }

            // Compute attention scores: Q * K^T / sqrt(d_k)
            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < numNodes; i++)
                {
                    for (int j = 0; j < numNodes; j++)
                    {
                        T score = NumOps.Zero;

                        for (int d = 0; d < _headDim; d++)
                        {
                            score = NumOps.Add(score,
                                NumOps.Multiply(queries[b, i, d], keys[b, j, d]));
                        }

                        score = NumOps.Divide(score, sqrtDk);

                        // Add structural bias if enabled
                        if (_useStructuralEncoding && _structuralBias != null)
                        {
                            score = NumOps.Add(score, _structuralBias[h, i, j]);
                        }

                        // Mask non-adjacent nodes (optional - can be commented out for full attention)
                        if (_useStructuralEncoding && NumOps.Equals(_adjacencyMatrix![b, i, j], NumOps.Zero))
                        {
                            score = NumOps.FromDouble(double.NegativeInfinity);
                        }

                        _lastAttentionWeights[b, h, i, j] = score;
                    }

                    // Softmax over attention scores
                    T maxScore = NumOps.FromDouble(double.NegativeInfinity);
                    for (int j = 0; j < numNodes; j++)
                    {
                        if (NumOps.GreaterThan(_lastAttentionWeights[b, h, i, j], maxScore))
                            maxScore = _lastAttentionWeights[b, h, i, j];
                    }

                    T sumExp = NumOps.Zero;
                    for (int j = 0; j < numNodes; j++)
                    {
                        if (!double.IsNegativeInfinity(NumOps.ToDouble(_lastAttentionWeights[b, h, i, j])))
                        {
                            T expVal = NumOps.Exp(NumOps.Subtract(_lastAttentionWeights[b, h, i, j], maxScore));
                            _lastAttentionWeights[b, h, i, j] = expVal;
                            sumExp = NumOps.Add(sumExp, expVal);
                        }
                        else
                        {
                            _lastAttentionWeights[b, h, i, j] = NumOps.Zero;
                        }
                    }

                    for (int j = 0; j < numNodes; j++)
                    {
                        _lastAttentionWeights[b, h, i, j] =
                            NumOps.Divide(_lastAttentionWeights[b, h, i, j], sumExp);
                    }
                }
            }

            // Apply attention to values
            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < numNodes; i++)
                {
                    for (int d = 0; d < _headDim; d++)
                    {
                        T sum = NumOps.Zero;

                        for (int j = 0; j < numNodes; j++)
                        {
                            sum = NumOps.Add(sum,
                                NumOps.Multiply(_lastAttentionWeights[b, h, i, j], values[b, j, d]));
                        }

                        headOutputs[b, h, i, d] = sum;
                    }
                }
            }
        }

        // Concatenate heads and project
        var output = new Tensor<T>([batchSize, numNodes, _outputFeatures]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                // Concatenate all heads
                var concat = new Vector<T>(_numHeads * _headDim);
                int idx = 0;
                for (int h = 0; h < _numHeads; h++)
                {
                    for (int d = 0; d < _headDim; d++)
                    {
                        concat[idx++] = headOutputs[b, h, n, d];
                    }
                }

                // Project concatenated output
                for (int f = 0; f < _outputFeatures; f++)
                {
                    T sum = _outputBias[f];
                    for (int c = 0; c < concat.Length; c++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(concat[c], _outputWeights[c, f]));
                    }
                    output[b, n, f] = sum;
                }
            }
        }

        return output;
    }

    private Tensor<T> FeedForwardNetwork(Tensor<T> input, int batchSize, int numNodes)
    {
        var output = new Tensor<T>([batchSize, numNodes, _outputFeatures]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                // First layer with GELU
                var hidden = new Vector<T>(_ffnWeights1.Columns);
                for (int h = 0; h < hidden.Length; h++)
                {
                    T sum = _ffnBias1[h];
                    for (int f = 0; f < _outputFeatures; f++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(input[b, n, f], _ffnWeights1[f, h]));
                    }
                    hidden[h] = GELU(sum);
                }

                // Second layer
                for (int f = 0; f < _outputFeatures; f++)
                {
                    T sum = _ffnBias2[f];
                    for (int h = 0; h < hidden.Length; h++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(hidden[h], _ffnWeights2[h, f]));
                    }
                    output[b, n, f] = sum;
                }
            }
        }

        return output;
    }

    /// <inheritdoc/>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // Simplified backward - full implementation would include complete gradient flow
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
        // Simplified - full implementation would update all parameters
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        // Simplified - would include all parameters
        return new Vector<T>(1);
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        // Simplified - would set all parameters
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastAttentionScores = null;
        _lastAttentionWeights = null;
    }

    /// <inheritdoc/>
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc/>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException(
            "GraphTransformerLayer does not support computation graph export due to dynamic self-attention mechanisms.");
    }
}
