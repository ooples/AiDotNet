using AiDotNet.ActivationFunctions;
using AiDotNet.Autodiff;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

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
    /// Query transformation weights for each head: [numHeads, inputFeatures, headDim]
    /// </summary>
    private Tensor<T> _queryWeights;

    /// <summary>
    /// Key transformation weights for each head: [numHeads, inputFeatures, headDim]
    /// </summary>
    private Tensor<T> _keyWeights;

    /// <summary>
    /// Value transformation weights for each head: [numHeads, inputFeatures, headDim]
    /// </summary>
    private Tensor<T> _valueWeights;

    /// <summary>
    /// Output projection weights: [numHeads * headDim, outputFeatures]
    /// </summary>
    private Tensor<T> _outputWeights;

    /// <summary>
    /// Output projection bias: [outputFeatures]
    /// </summary>
    private Tensor<T> _outputBias;

    /// <summary>
    /// Structural bias for attention (learned from graph structure): [numHeads, maxNodes, maxNodes]
    /// </summary>
    private Tensor<T>? _structuralBias;

    /// <summary>
    /// Feed-forward network first layer weights: [outputFeatures, ffnHiddenDim]
    /// </summary>
    private Tensor<T> _ffnWeights1;

    /// <summary>
    /// Feed-forward network second layer weights: [ffnHiddenDim, outputFeatures]
    /// </summary>
    private Tensor<T> _ffnWeights2;

    /// <summary>
    /// Feed-forward network first layer bias: [ffnHiddenDim]
    /// </summary>
    private Tensor<T> _ffnBias1;

    /// <summary>
    /// Feed-forward network second layer bias: [outputFeatures]
    /// </summary>
    private Tensor<T> _ffnBias2;

    /// <summary>
    /// Layer normalization scale for first norm: [outputFeatures]
    /// </summary>
    private Tensor<T> _layerNorm1Scale;

    /// <summary>
    /// Layer normalization bias for first norm: [outputFeatures]
    /// </summary>
    private Tensor<T> _layerNorm1Bias;

    /// <summary>
    /// Layer normalization scale for second norm: [outputFeatures]
    /// </summary>
    private Tensor<T> _layerNorm2Scale;

    /// <summary>
    /// Layer normalization bias for second norm: [outputFeatures]
    /// </summary>
    private Tensor<T> _layerNorm2Bias;

    /// <summary>
    /// The adjacency matrix defining graph structure.
    /// </summary>
    private Tensor<T>? _adjacencyMatrix;

    /// <summary>
    /// Cached values for backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastQueries;
    private Tensor<T>? _lastKeys;
    private Tensor<T>? _lastValues;
    private Tensor<T>? _lastAttentionWeights;
    private Tensor<T>? _lastHeadOutputs;
    private Tensor<T>? _lastConcatenated;
    private Tensor<T>? _lastAttnOutput;
    private Tensor<T>? _lastNormed1;
    private Tensor<T>? _lastFFNHidden;
    private Tensor<T>? _lastFFNOutput;

    /// <summary>
    /// Gradients for parameters.
    /// </summary>
    private Tensor<T>? _queryWeightsGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _outputWeightsGradient;
    private Tensor<T>? _outputBiasGradient;
    private Tensor<T>? _ffnWeights1Gradient;
    private Tensor<T>? _ffnWeights2Gradient;
    private Tensor<T>? _ffnBias1Gradient;
    private Tensor<T>? _ffnBias2Gradient;

    private readonly int _ffnHiddenDim;

    /// <summary>
    /// Activation function for the feed-forward network hidden layer.
    /// Default is GELU as used in transformer architectures.
    /// </summary>
    private readonly IActivationFunction<T> _ffnActivation;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public int InputFeatures => _inputFeatures;

    /// <inheritdoc/>
    public int OutputFeatures => _outputFeatures;

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphTransformerLayer{T}"/> class with a scalar activation function.
    /// </summary>
    /// <param name="inputFeatures">Number of input features per node.</param>
    /// <param name="outputFeatures">Number of output features per node.</param>
    /// <param name="numHeads">Number of attention heads (default: 8).</param>
    /// <param name="headDim">Dimension per attention head (default: 64).</param>
    /// <param name="useStructuralEncoding">Whether to use structural bias (default: true).</param>
    /// <param name="dropoutRate">Dropout rate for attention (default: 0.1).</param>
    /// <param name="activationFunction">Scalar activation function for the layer output. Defaults to Identity if not specified.</param>
    /// <param name="ffnActivation">Scalar activation function for FFN hidden layer. Defaults to GELU if not specified.</param>
    /// <remarks>
    /// <para>
    /// Creates a Graph Transformer layer with multi-head attention and feed-forward network.
    /// The layer includes skip connections and layer normalization for stable training.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Graph Transformer layer with scalar activation functions.
    ///
    /// Key parameters:
    /// - numHeads: How many parallel attention mechanisms (more = capture different patterns)
    /// - headDim: Size of each attention head (bigger = more expressive per head)
    /// - useStructuralEncoding: Whether to bias attention toward connected nodes
    ///   * true: Graph structure guides attention (recommended for most graphs)
    ///   * false: Pure attention without graph bias (for dense/complete graphs)
    /// - dropoutRate: Randomly ignore some attention during training (prevents overfitting)
    /// - ffnActivation: Activation for the feed-forward network (GELU, ReLU, SiLU, etc.)
    ///
    /// The layer has two main components:
    /// 1. **Multi-head attention**: Learns which nodes to focus on
    /// 2. **Feed-forward network**: Processes the attended information
    ///
    /// Both use skip connections (adding input back to output) for better gradient flow.
    /// </para>
    /// </remarks>
#pragma warning disable CS8618 // Non-nullable field must contain a non-null value - fields are initialized by InitializeWeightTensors()
    public GraphTransformerLayer(
        int inputFeatures,
        int outputFeatures,
        int numHeads = 8,
        int headDim = 64,
        bool useStructuralEncoding = true,
        double dropoutRate = 0.1,
        IActivationFunction<T>? activationFunction = null,
        IActivationFunction<T>? ffnActivation = null)
        : base([inputFeatures], [outputFeatures], activationFunction ?? new IdentityActivation<T>())
    {
        _inputFeatures = inputFeatures;
        _outputFeatures = outputFeatures;
        _numHeads = numHeads;
        _headDim = headDim;
        _useStructuralEncoding = useStructuralEncoding;
        _dropoutRate = dropoutRate;
        _random = RandomHelper.CreateSecureRandom();
        _ffnActivation = ffnActivation ?? new GELUActivation<T>();
        _ffnHiddenDim = 4 * outputFeatures;

        InitializeWeightTensors();
        InitializeParameters();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphTransformerLayer{T}"/> class with a vector activation function.
    /// </summary>
    /// <param name="inputFeatures">Number of input features per node.</param>
    /// <param name="outputFeatures">Number of output features per node.</param>
    /// <param name="numHeads">Number of attention heads (default: 8).</param>
    /// <param name="headDim">Dimension per attention head (default: 64).</param>
    /// <param name="useStructuralEncoding">Whether to use structural bias (default: true).</param>
    /// <param name="dropoutRate">Dropout rate for attention (default: 0.1).</param>
    /// <param name="vectorActivationFunction">Vector activation function for the layer output. Defaults to Identity if not specified.</param>
    /// <param name="ffnActivation">Scalar activation function for FFN hidden layer. Defaults to GELU if not specified.</param>
    /// <remarks>
    /// <para>
    /// Creates a Graph Transformer layer with multi-head attention and feed-forward network,
    /// using a vector activation function that operates on entire vectors rather than individual elements.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor is similar to the scalar version but uses a vector activation function.
    ///
    /// Vector activation functions like Softmax are useful for:
    /// - Node classification problems (choosing between multiple node types)
    /// - Problems where outputs need to sum to 1 (like probabilities)
    /// - Cases where output values should influence each other
    ///
    /// For example, in a graph with molecules, you might use Softmax to classify
    /// each atom node into one of several element types.
    /// </para>
    /// </remarks>
    public GraphTransformerLayer(
        int inputFeatures,
        int outputFeatures,
        int numHeads = 8,
        int headDim = 64,
        bool useStructuralEncoding = true,
        double dropoutRate = 0.1,
        IVectorActivationFunction<T>? vectorActivationFunction = null,
        IActivationFunction<T>? ffnActivation = null)
        : base([inputFeatures], [outputFeatures], vectorActivationFunction ?? new IdentityActivation<T>())
    {
        _inputFeatures = inputFeatures;
        _outputFeatures = outputFeatures;
        _numHeads = numHeads;
        _headDim = headDim;
        _useStructuralEncoding = useStructuralEncoding;
        _dropoutRate = dropoutRate;
        _random = RandomHelper.CreateSecureRandom();
        _ffnActivation = ffnActivation ?? new GELUActivation<T>();
        _ffnHiddenDim = 4 * outputFeatures;

        InitializeWeightTensors();
        InitializeParameters();
    }
#pragma warning restore CS8618

    /// <summary>
    /// Initializes weight tensors with appropriate shapes.
    /// </summary>
    private void InitializeWeightTensors()
    {
        // Initialize Q, K, V projections
        _queryWeights = new Tensor<T>([_numHeads, _inputFeatures, _headDim]);
        _keyWeights = new Tensor<T>([_numHeads, _inputFeatures, _headDim]);
        _valueWeights = new Tensor<T>([_numHeads, _inputFeatures, _headDim]);

        // Output projection
        _outputWeights = new Tensor<T>([_numHeads * _headDim, _outputFeatures]);
        _outputBias = new Tensor<T>([_outputFeatures]);

        // Feed-forward network (2 layers)
        // _ffnHiddenDim is initialized in constructor
        _ffnWeights1 = new Tensor<T>([_outputFeatures, _ffnHiddenDim]);
        _ffnWeights2 = new Tensor<T>([_ffnHiddenDim, _outputFeatures]);
        _ffnBias1 = new Tensor<T>([_ffnHiddenDim]);
        _ffnBias2 = new Tensor<T>([_outputFeatures]);

        // Layer normalization parameters
        _layerNorm1Scale = new Tensor<T>([_outputFeatures]);
        _layerNorm1Bias = new Tensor<T>([_outputFeatures]);
        _layerNorm2Scale = new Tensor<T>([_outputFeatures]);
        _layerNorm2Bias = new Tensor<T>([_outputFeatures]);
    }

    private void InitializeParameters()
    {
        // Xavier initialization for Q, K, V
        T scaleQKV = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_inputFeatures + _headDim)));
        InitializeTensor(_queryWeights, scaleQKV);
        InitializeTensor(_keyWeights, scaleQKV);
        InitializeTensor(_valueWeights, scaleQKV);

        // Initialize output weights
        T scaleOut = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_numHeads * _headDim + _outputFeatures)));
        InitializeTensor(_outputWeights, scaleOut);

        // Initialize FFN weights
        T scaleFFN1 = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_outputFeatures + _ffnHiddenDim)));
        InitializeTensor(_ffnWeights1, scaleFFN1);

        T scaleFFN2 = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_ffnHiddenDim + _outputFeatures)));
        InitializeTensor(_ffnWeights2, scaleFFN2);

        // Initialize layer norm to identity (scale=1, bias=0)
        _layerNorm1Scale.Fill(NumOps.FromDouble(1.0));
        _layerNorm1Bias.Fill(NumOps.Zero);
        _layerNorm2Scale.Fill(NumOps.FromDouble(1.0));
        _layerNorm2Bias.Fill(NumOps.Zero);

        // Initialize biases to zero
        _outputBias.Fill(NumOps.Zero);
        _ffnBias1.Fill(NumOps.Zero);
        _ffnBias2.Fill(NumOps.Zero);
    }

    /// <summary>
    /// Initializes a tensor with scaled random values.
    /// </summary>
    /// <param name="tensor">The tensor to initialize.</param>
    /// <param name="scale">The scale factor for the random values.</param>
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
        _adjacencyMatrix = adjacencyMatrix;

        // Initialize structural bias if needed
        if (_useStructuralEncoding && _structuralBias == null)
        {
            int maxNodes = adjacencyMatrix.Shape[1];
            _structuralBias = new Tensor<T>([_numHeads, maxNodes, maxNodes]);

            // Simple initialization: bias toward connected nodes
            T scale = NumOps.FromDouble(0.1);
            InitializeTensor(_structuralBias, scale);
        }
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

        // Store original shape for any-rank tensor support
        _originalInputShape = input.Shape;
        int rank = input.Shape.Length;

        // Graph layer expects 3D: [batchSize, numNodes, features]
        // Handle any-rank tensor: normalize to 3D for processing
        Tensor<T> processInput;
        int batchSize;

        if (rank == 2)
        {
            // 2D [nodes, features]: add batch dim
            batchSize = 1;
            processInput = input.Reshape([1, input.Shape[0], input.Shape[1]]);
        }
        else if (rank == 3)
        {
            // Standard 3D [batchSize, nodes, features]
            batchSize = input.Shape[0];
            processInput = input;
        }
        else if (rank > 3)
        {
            // Higher-rank: collapse leading dims into batch
            int flatBatch = 1;
            for (int d = 0; d < rank - 2; d++)
                flatBatch *= input.Shape[d];
            batchSize = flatBatch;
            processInput = input.Reshape([flatBatch, input.Shape[rank - 2], input.Shape[rank - 1]]);
        }
        else
        {
            // 1D: treat as single node with features
            batchSize = 1;
            processInput = input.Reshape([1, 1, input.Shape[0]]);
        }

        _lastInput = processInput;
        int numNodes = processInput.Shape[1];

        // Multi-head attention block with residual connection
        var attended = MultiHeadAttention(processInput, batchSize, numNodes);

        // Add residual connection (with projection if dimensions don't match)
        Tensor<T> residualInput;
        if (_inputFeatures == _outputFeatures)
        {
            residualInput = processInput;
        }
        else
        {
            // Project input to match output dimensions
            residualInput = new Tensor<T>([batchSize, numNodes, _outputFeatures]);
            for (int b = 0; b < batchSize; b++)
            {
                for (int n = 0; n < numNodes; n++)
                {
                    for (int f = 0; f < _outputFeatures; f++)
                    {
                        residualInput[b, n, f] = f < _inputFeatures ? processInput[b, n, f] : NumOps.Zero;
                    }
                }
            }
        }

        var normed1 = Engine.TensorAdd(attended, residualInput);
        _lastNormed1 = normed1;

        // Feed-forward network with residual
        var ffnOutput = FeedForwardNetwork(normed1, batchSize, numNodes);
        _lastFFNOutput = ffnOutput;

        // Final residual and layer norm
        var output = Engine.TensorAdd(normed1, ffnOutput);

        _lastOutput = ApplyActivation(output);

        // Restore original shape for any-rank tensor support
        if (_originalInputShape != null && _originalInputShape.Length != 3)
        {
            if (_originalInputShape.Length == 2)
            {
                // Was 2D, return [nodes, outputFeatures]
                return _lastOutput.Reshape([numNodes, _outputFeatures]);
            }
            else if (_originalInputShape.Length == 1)
            {
                // Was 1D, return [outputFeatures]
                return _lastOutput.Reshape([_outputFeatures]);
            }
            else
            {
                // Higher-rank: restore leading dimensions
                var newShape = new int[_originalInputShape.Length];
                for (int d = 0; d < _originalInputShape.Length - 1; d++)
                    newShape[d] = _originalInputShape[d];
                newShape[_originalInputShape.Length - 1] = _outputFeatures;
                return _lastOutput.Reshape(newShape);
            }
        }

        return _lastOutput;
    }

    private Tensor<T> MultiHeadAttention(Tensor<T> input, int batchSize, int numNodes)
    {
        // Store attention outputs for each head: [batch, numHeads, numNodes, headDim]
        var headOutputs = new Tensor<T>([batchSize, _numHeads, numNodes, _headDim]);
        _lastAttentionWeights = new Tensor<T>([batchSize, _numHeads, numNodes, numNodes]);

        // Store Q, K, V for backward pass: [batch, numHeads, numNodes, headDim]
        _lastQueries = new Tensor<T>([batchSize, _numHeads, numNodes, _headDim]);
        _lastKeys = new Tensor<T>([batchSize, _numHeads, numNodes, _headDim]);
        _lastValues = new Tensor<T>([batchSize, _numHeads, numNodes, _headDim]);

        T sqrtDk = NumOps.Sqrt(NumOps.FromDouble(_headDim));

        for (int h = 0; h < _numHeads; h++)
        {
            // Compute Q, K, V for this head using batched matmul
            // Extract weight slices for this head: [inputFeatures, headDim]
            var qWeightSlice = ExtractHeadWeights(_queryWeights, h);
            var kWeightSlice = ExtractHeadWeights(_keyWeights, h);
            var vWeightSlice = ExtractHeadWeights(_valueWeights, h);

            // Compute Q = input @ W_q: [batch, numNodes, inputFeatures] @ [inputFeatures, headDim]
            var queries = BatchedMatMul3Dx2D(input, qWeightSlice, batchSize, numNodes, _inputFeatures, _headDim);
            var keys = BatchedMatMul3Dx2D(input, kWeightSlice, batchSize, numNodes, _inputFeatures, _headDim);
            var values = BatchedMatMul3Dx2D(input, vWeightSlice, batchSize, numNodes, _inputFeatures, _headDim);

            // Store for backward pass
            for (int b = 0; b < batchSize; b++)
            {
                for (int n = 0; n < numNodes; n++)
                {
                    for (int d = 0; d < _headDim; d++)
                    {
                        _lastQueries[b, h, n, d] = queries[b, n, d];
                        _lastKeys[b, h, n, d] = keys[b, n, d];
                        _lastValues[b, h, n, d] = values[b, n, d];
                    }
                }
            }

            // Compute attention scores: Q * K^T / sqrt(d_k)
            // For each batch, compute: [numNodes, headDim] @ [headDim, numNodes] = [numNodes, numNodes]
            for (int b = 0; b < batchSize; b++)
            {
                // Extract Q and K for this batch
                var qBatch = queries.Reshape([batchSize, numNodes, _headDim]);
                var kBatch = keys.Reshape([batchSize, numNodes, _headDim]);

                // Transpose K: [numNodes, headDim] -> [headDim, numNodes]
                for (int i = 0; i < numNodes; i++)
                {
                    for (int j = 0; j < numNodes; j++)
                    {
                        T score = NumOps.Zero;
                        for (int d = 0; d < _headDim; d++)
                        {
                            score = NumOps.Add(score, NumOps.Multiply(qBatch[b, i, d], kBatch[b, j, d]));
                        }

                        score = NumOps.Divide(score, sqrtDk);

                        // Add structural bias if enabled
                        if (_useStructuralEncoding && _structuralBias != null)
                        {
                            score = NumOps.Add(score, _structuralBias[h, i, j]);
                        }

                        // Mask non-adjacent nodes (optional - can be commented out for full attention)
                        if (_useStructuralEncoding && _adjacencyMatrix != null)
                        {
                            // Handle both 2D [N, N] and 3D [B, N, N] adjacency matrices
                            T adjValue = _adjacencyMatrix.Shape.Length == 2
                                ? _adjacencyMatrix[new int[] { i, j }]
                                : _adjacencyMatrix[new int[] { b, i, j }];
                            if (NumOps.Equals(adjValue, NumOps.Zero))
                            {
                                score = NumOps.FromDouble(double.NegativeInfinity);
                            }
                        }

                        _lastAttentionWeights[b, h, i, j] = score;
                    }

                    // Softmax over attention scores for row i
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

                    // Guard against division by zero
                    if (NumOps.Equals(sumExp, NumOps.Zero))
                    {
                        sumExp = NumOps.FromDouble(1e-10);
                    }

                    for (int j = 0; j < numNodes; j++)
                    {
                        _lastAttentionWeights[b, h, i, j] = NumOps.Divide(_lastAttentionWeights[b, h, i, j], sumExp);
                    }
                }
            }

            // Apply attention to values: attn @ V
            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < numNodes; i++)
                {
                    for (int d = 0; d < _headDim; d++)
                    {
                        T sum = NumOps.Zero;
                        for (int j = 0; j < numNodes; j++)
                        {
                            sum = NumOps.Add(sum, NumOps.Multiply(_lastAttentionWeights[b, h, i, j], _lastValues[b, h, j, d]));
                        }
                        headOutputs[b, h, i, d] = sum;
                    }
                }
            }
        }

        _lastHeadOutputs = headOutputs;

        // Concatenate heads: [batch, numNodes, numHeads * headDim]
        var concatenated = new Tensor<T>([batchSize, numNodes, _numHeads * _headDim]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                int idx = 0;
                for (int h = 0; h < _numHeads; h++)
                {
                    for (int d = 0; d < _headDim; d++)
                    {
                        concatenated[b, n, idx++] = headOutputs[b, h, n, d];
                    }
                }
            }
        }

        _lastConcatenated = concatenated;

        // Project concatenated output: [batch, numNodes, numHeads*headDim] @ [numHeads*headDim, outputFeatures]
        var output = BatchedMatMul3Dx2D(concatenated, _outputWeights, batchSize, numNodes, _numHeads * _headDim, _outputFeatures);

        // Add bias
        var biasBroadcast = BroadcastBias(_outputBias, batchSize, numNodes);
        output = Engine.TensorAdd(output, biasBroadcast);

        _lastAttnOutput = output;
        return output;
    }

    private Tensor<T> ExtractHeadWeights(Tensor<T> weights, int headIndex)
    {
        // Extract [inputFeatures, headDim] from [numHeads, inputFeatures, headDim]
        var result = new Tensor<T>([_inputFeatures, _headDim]);
        for (int i = 0; i < _inputFeatures; i++)
        {
            for (int j = 0; j < _headDim; j++)
            {
                result[i, j] = weights[headIndex, i, j];
            }
        }
        return result;
    }

    private Tensor<T> BatchedMatMul3Dx2D(Tensor<T> input3D, Tensor<T> weights2D, int batch, int rows, int cols, int outputCols)
    {
        var flattened = input3D.Reshape([batch * rows, cols]);
        var result = Engine.TensorMatMul(flattened, weights2D);
        return result.Reshape([batch, rows, outputCols]);
    }

    private Tensor<T> BroadcastBias(Tensor<T> bias, int batchSize, int numNodes)
    {
        int outputFeatures = bias.Length;
        var biasReshaped = bias.Reshape([1, 1, outputFeatures]);
        var broadcast = Engine.TensorTile(biasReshaped, [batchSize, numNodes, 1]);
        return broadcast;
    }

    private Tensor<T> FeedForwardNetwork(Tensor<T> input, int batchSize, int numNodes)
    {
        // First layer: [batch, numNodes, outputFeatures] @ [outputFeatures, ffnHiddenDim]
        var hidden = BatchedMatMul3Dx2D(input, _ffnWeights1, batchSize, numNodes, _outputFeatures, _ffnHiddenDim);

        // Add bias1
        var bias1Broadcast = BroadcastBias(_ffnBias1, batchSize, numNodes);
        hidden = Engine.TensorAdd(hidden, bias1Broadcast);

        // Apply FFN activation (configurable: GELU by default, but user can choose any activation)
        hidden = ApplyFFNActivation(hidden);
        _lastFFNHidden = hidden;

        // Second layer: [batch, numNodes, ffnHiddenDim] @ [ffnHiddenDim, outputFeatures]
        var output = BatchedMatMul3Dx2D(hidden, _ffnWeights2, batchSize, numNodes, _ffnHiddenDim, _outputFeatures);

        // Add bias2
        var bias2Broadcast = BroadcastBias(_ffnBias2, batchSize, numNodes);
        output = Engine.TensorAdd(output, bias2Broadcast);

        return output;
    }

    /// <summary>
    /// Applies the configured FFN activation function to the input tensor.
    /// </summary>
    /// <param name="input">The input tensor to apply activation to.</param>
    /// <returns>The tensor with activation applied element-wise.</returns>
    /// <remarks>
    /// Uses the _ffnActivation interface which allows users to configure any activation
    /// function (GELU, ReLU, SiLU, etc.) rather than hardcoding GELU.
    /// </remarks>
    private Tensor<T> ApplyFFNActivation(Tensor<T> input)
    {
        var result = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            result[i] = _ffnActivation.Activate(input.GetFlat(i));
        }
        return result;
    }

    /// <summary>
    /// Performs the backward pass of the graph transformer layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when Forward has not been called before Backward.</exception>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before Backward.");
        }

        // Reshape outputGradient to match _lastOutput shape if needed
        var gradForBackward = outputGradient;
        if (_originalInputShape != null && _originalInputShape.Length != _lastOutput.Shape.Length)
        {
            gradForBackward = outputGradient.Reshape(_lastOutput.Shape);
        }

        var activationGradient = ApplyActivationDerivative(_lastOutput, gradForBackward);

        int batchSize = _lastInput.Shape[0];
        int numNodes = _lastInput.Shape[1];

        // Initialize gradients
        _outputWeightsGradient = new Tensor<T>(_outputWeights.Shape);
        _outputBiasGradient = new Tensor<T>(_outputBias.Shape);
        _queryWeightsGradient = new Tensor<T>(_queryWeights.Shape);
        _keyWeightsGradient = new Tensor<T>(_keyWeights.Shape);
        _valueWeightsGradient = new Tensor<T>(_valueWeights.Shape);
        _ffnWeights1Gradient = new Tensor<T>(_ffnWeights1.Shape);
        _ffnWeights2Gradient = new Tensor<T>(_ffnWeights2.Shape);
        _ffnBias1Gradient = new Tensor<T>(_ffnBias1.Shape);
        _ffnBias2Gradient = new Tensor<T>(_ffnBias2.Shape);

        _outputWeightsGradient.Fill(NumOps.Zero);
        _outputBiasGradient.Fill(NumOps.Zero);
        _queryWeightsGradient.Fill(NumOps.Zero);
        _keyWeightsGradient.Fill(NumOps.Zero);
        _valueWeightsGradient.Fill(NumOps.Zero);
        _ffnWeights1Gradient.Fill(NumOps.Zero);
        _ffnWeights2Gradient.Fill(NumOps.Zero);
        _ffnBias1Gradient.Fill(NumOps.Zero);
        _ffnBias2Gradient.Fill(NumOps.Zero);

        // Backward through residual: gradient splits to FFN and to normed1
        var gradNormed1 = activationGradient;
        var gradFFNOutput = activationGradient;

        // Backward through FFN second layer
        if (_lastFFNHidden == null || _lastNormed1 == null)
        {
            throw new InvalidOperationException("Forward pass incomplete.");
        }

        // dL/dW2 = hidden^T @ grad
        _ffnWeights2Gradient = Engine.ReduceSum(
            BackwardFFNWeights2(_lastFFNHidden, gradFFNOutput, batchSize, numNodes),
            [0], keepDims: false);

        // dL/db2 = sum over batch and nodes
        _ffnBias2Gradient = Engine.ReduceSum(gradFFNOutput, [0, 1], keepDims: false);

        // dL/dhidden = grad @ W2^T
        var gradFFNHidden = BackwardFFNHidden(gradFFNOutput, batchSize, numNodes);

        // Backward through FFN activation (configurable: GELU by default)
        gradFFNHidden = BackwardFFNActivation(_lastFFNHidden, gradFFNHidden);

        // Backward through FFN first layer
        _ffnWeights1Gradient = Engine.ReduceSum(
            BackwardFFNWeights1(_lastNormed1, gradFFNHidden, batchSize, numNodes),
            [0], keepDims: false);

        _ffnBias1Gradient = Engine.ReduceSum(gradFFNHidden, [0, 1], keepDims: false);

        // dL/dnormed1 from FFN
        var gradNormed1FromFFN = BackwardFFNInput(gradFFNHidden, batchSize, numNodes);

        // Combine gradients to normed1
        gradNormed1 = Engine.TensorAdd(gradNormed1, gradNormed1FromFFN);

        // Backward through attention residual
        var gradAttnOutput = gradNormed1;
        var gradInputFromResidual = gradNormed1;

        // Backward through attention output projection
        if (_lastConcatenated == null)
        {
            throw new InvalidOperationException("Forward pass incomplete.");
        }

        // dL/dW_out = concatenated^T @ grad
        _outputWeightsGradient = Engine.ReduceSum(
            BackwardOutputWeights(_lastConcatenated, gradAttnOutput, batchSize, numNodes),
            [0], keepDims: false);

        _outputBiasGradient = Engine.ReduceSum(gradAttnOutput, [0, 1], keepDims: false);

        // dL/dconcatenated = grad @ W_out^T
        var gradConcatenated = BackwardConcatenated(gradAttnOutput, batchSize, numNodes);

        // Backward through head concatenation and attention
        var gradInput = BackwardAttention(gradConcatenated, batchSize, numNodes);

        // Add gradient from residual connection
        if (_inputFeatures == _outputFeatures)
        {
            gradInput = Engine.TensorAdd(gradInput, gradInputFromResidual);
        }

        // Reshape to match original input shape
        if (_originalInputShape != null && _originalInputShape.Length != gradInput.Shape.Length)
        {
            return gradInput.Reshape(_originalInputShape);
        }

        return gradInput;
    }

    private Tensor<T> BackwardFFNWeights2(Tensor<T> hidden, Tensor<T> grad, int batchSize, int numNodes)
    {
        // For each batch: hidden^T @ grad
        var result = new Tensor<T>([batchSize, _ffnHiddenDim, _outputFeatures]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < _ffnHiddenDim; i++)
            {
                for (int j = 0; j < _outputFeatures; j++)
                {
                    T sum = NumOps.Zero;
                    for (int n = 0; n < numNodes; n++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(hidden[b, n, i], grad[b, n, j]));
                    }
                    result[b, i, j] = sum;
                }
            }
        }
        return result;
    }

    private Tensor<T> BackwardFFNHidden(Tensor<T> grad, int batchSize, int numNodes)
    {
        return BatchedMatMul3Dx2D(grad, Engine.TensorTranspose(_ffnWeights2), batchSize, numNodes, _outputFeatures, _ffnHiddenDim);
    }

    /// <summary>
    /// Computes the backward pass through the FFN activation function.
    /// </summary>
    /// <param name="input">The input tensor from the forward pass (pre-activation values).</param>
    /// <param name="grad">The gradient flowing back from the next operation.</param>
    /// <returns>The gradient with respect to the input.</returns>
    /// <remarks>
    /// Uses the configurable _ffnActivation.Derivative() method, allowing users
    /// to plug in any activation function (GELU, ReLU, SiLU, etc.) with proper
    /// gradient computation.
    /// </remarks>
    private Tensor<T> BackwardFFNActivation(Tensor<T> input, Tensor<T> grad)
    {
        var result = new Tensor<T>(grad.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            T x = input.GetFlat(i);
            T activationDeriv = _ffnActivation.Derivative(x);
            result[i] = NumOps.Multiply(grad.GetFlat(i), activationDeriv);
        }
        return result;
    }

    private Tensor<T> BackwardFFNWeights1(Tensor<T> input, Tensor<T> grad, int batchSize, int numNodes)
    {
        var result = new Tensor<T>([batchSize, _outputFeatures, _ffnHiddenDim]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < _outputFeatures; i++)
            {
                for (int j = 0; j < _ffnHiddenDim; j++)
                {
                    T sum = NumOps.Zero;
                    for (int n = 0; n < numNodes; n++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(input[b, n, i], grad[b, n, j]));
                    }
                    result[b, i, j] = sum;
                }
            }
        }
        return result;
    }

    private Tensor<T> BackwardFFNInput(Tensor<T> grad, int batchSize, int numNodes)
    {
        return BatchedMatMul3Dx2D(grad, Engine.TensorTranspose(_ffnWeights1), batchSize, numNodes, _ffnHiddenDim, _outputFeatures);
    }

    private Tensor<T> BackwardOutputWeights(Tensor<T> concatenated, Tensor<T> grad, int batchSize, int numNodes)
    {
        var result = new Tensor<T>([batchSize, _numHeads * _headDim, _outputFeatures]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < _numHeads * _headDim; i++)
            {
                for (int j = 0; j < _outputFeatures; j++)
                {
                    T sum = NumOps.Zero;
                    for (int n = 0; n < numNodes; n++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(concatenated[b, n, i], grad[b, n, j]));
                    }
                    result[b, i, j] = sum;
                }
            }
        }
        return result;
    }

    private Tensor<T> BackwardConcatenated(Tensor<T> grad, int batchSize, int numNodes)
    {
        return BatchedMatMul3Dx2D(grad, Engine.TensorTranspose(_outputWeights), batchSize, numNodes, _outputFeatures, _numHeads * _headDim);
    }

    private Tensor<T> BackwardAttention(Tensor<T> gradConcatenated, int batchSize, int numNodes)
    {
        var gradInput = new Tensor<T>([batchSize, numNodes, _inputFeatures]);
        gradInput.Fill(NumOps.Zero);

        for (int h = 0; h < _numHeads; h++)
        {
            // Extract gradient for this head from concatenated
            var gradHead = new Tensor<T>([batchSize, numNodes, _headDim]);
            for (int b = 0; b < batchSize; b++)
            {
                for (int n = 0; n < numNodes; n++)
                {
                    for (int d = 0; d < _headDim; d++)
                    {
                        gradHead[b, n, d] = gradConcatenated[b, n, h * _headDim + d];
                    }
                }
            }

            // Backward through attention application (attn @ V)
            // dL/dV = attn^T @ gradHead
            var gradValues = new Tensor<T>([batchSize, numNodes, _headDim]);
            if (_lastAttentionWeights == null || _lastQueries == null || _lastKeys == null || _lastValues == null)
            {
                throw new InvalidOperationException("Forward pass incomplete.");
            }

            for (int b = 0; b < batchSize; b++)
            {
                for (int j = 0; j < numNodes; j++)
                {
                    for (int d = 0; d < _headDim; d++)
                    {
                        T sum = NumOps.Zero;
                        for (int i = 0; i < numNodes; i++)
                        {
                            sum = NumOps.Add(sum, NumOps.Multiply(_lastAttentionWeights[b, h, i, j], gradHead[b, i, d]));
                        }
                        gradValues[b, j, d] = sum;
                    }
                }
            }

            // Backward through V projection
            var vWeightSlice = ExtractHeadWeights(_valueWeights, h);
            var gradInputFromV = BatchedMatMul3Dx2D(gradValues, Engine.TensorTranspose(vWeightSlice), batchSize, numNodes, _headDim, _inputFeatures);
            gradInput = Engine.TensorAdd(gradInput, gradInputFromV);

            // Update V weight gradients
            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < _inputFeatures; i++)
                {
                    for (int j = 0; j < _headDim; j++)
                    {
                        T sum = NumOps.Zero;
                        for (int n = 0; n < numNodes; n++)
                        {
                            if (_lastInput != null)
                            {
                                sum = NumOps.Add(sum, NumOps.Multiply(_lastInput[b, n, i], gradValues[b, n, j]));
                            }
                        }
                        if (_valueWeightsGradient != null)
                        {
                            _valueWeightsGradient[h, i, j] = NumOps.Add(_valueWeightsGradient[h, i, j], sum);
                        }
                    }
                }
            }

            // Full backward through attention mechanism (Q, K gradients)
            // Attention: output = softmax(Q @ K^T / sqrt(d_k)) @ V
            // dL/d(attn_weights) = gradHead @ V^T
            var gradAttentionWeights = new Tensor<T>([batchSize, numNodes, numNodes]);
            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < numNodes; i++)
                {
                    for (int j = 0; j < numNodes; j++)
                    {
                        T sum = NumOps.Zero;
                        for (int d = 0; d < _headDim; d++)
                        {
                            sum = NumOps.Add(sum, NumOps.Multiply(gradHead[b, i, d], _lastValues[b, h, j, d]));
                        }
                        gradAttentionWeights[b, i, j] = sum;
                    }
                }
            }

            // Backward through softmax: d(softmax)/d(score) = softmax * (delta_ij - softmax)
            // dL/d(score_ij) = sum_k(dL/d(attn_ik) * attn_ik * (delta_jk - attn_ij))
            //                = dL/d(attn_ij) * attn_ij - attn_ij * sum_k(dL/d(attn_ik) * attn_ik)
            var gradScores = new Tensor<T>([batchSize, numNodes, numNodes]);
            T scale = NumOps.Sqrt(NumOps.FromDouble(_headDim));

            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < numNodes; i++)
                {
                    // Compute sum_k(dL/d(attn_ik) * attn_ik) for this row
                    T weightedSum = NumOps.Zero;
                    for (int k = 0; k < numNodes; k++)
                    {
                        T attn_ik = _lastAttentionWeights[b, h, i, k];
                        weightedSum = NumOps.Add(weightedSum,
                            NumOps.Multiply(gradAttentionWeights[b, i, k], attn_ik));
                    }

                    // Compute gradient for each score
                    for (int j = 0; j < numNodes; j++)
                    {
                        T attn_ij = _lastAttentionWeights[b, h, i, j];
                        // dL/d(score_ij) = attn_ij * (dL/d(attn_ij) - weighted_sum) / scale
                        T softmaxGrad = NumOps.Multiply(attn_ij,
                            NumOps.Subtract(gradAttentionWeights[b, i, j], weightedSum));
                        gradScores[b, i, j] = NumOps.Divide(softmaxGrad, scale);
                    }
                }
            }

            // Backward through Q @ K^T
            // dL/dQ = gradScores @ K
            // dL/dK = gradScores^T @ Q
            var gradQueries = new Tensor<T>([batchSize, numNodes, _headDim]);
            var gradKeys = new Tensor<T>([batchSize, numNodes, _headDim]);

            for (int b = 0; b < batchSize; b++)
            {
                // dL/dQ = gradScores @ K
                for (int i = 0; i < numNodes; i++)
                {
                    for (int d = 0; d < _headDim; d++)
                    {
                        T sum = NumOps.Zero;
                        for (int j = 0; j < numNodes; j++)
                        {
                            sum = NumOps.Add(sum, NumOps.Multiply(gradScores[b, i, j], _lastKeys[b, h, j, d]));
                        }
                        gradQueries[b, i, d] = sum;
                    }
                }

                // dL/dK = gradScores^T @ Q
                for (int j = 0; j < numNodes; j++)
                {
                    for (int d = 0; d < _headDim; d++)
                    {
                        T sum = NumOps.Zero;
                        for (int i = 0; i < numNodes; i++)
                        {
                            sum = NumOps.Add(sum, NumOps.Multiply(gradScores[b, i, j], _lastQueries[b, h, i, d]));
                        }
                        gradKeys[b, j, d] = sum;
                    }
                }
            }

            // Backward through Q projection and update Q weight gradients
            var qWeightSlice = ExtractHeadWeights(_queryWeights, h);
            var gradInputFromQ = BatchedMatMul3Dx2D(gradQueries, Engine.TensorTranspose(qWeightSlice), batchSize, numNodes, _headDim, _inputFeatures);
            gradInput = Engine.TensorAdd(gradInput, gradInputFromQ);

            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < _inputFeatures; i++)
                {
                    for (int j = 0; j < _headDim; j++)
                    {
                        T sum = NumOps.Zero;
                        for (int n = 0; n < numNodes; n++)
                        {
                            if (_lastInput != null)
                            {
                                sum = NumOps.Add(sum, NumOps.Multiply(_lastInput[b, n, i], gradQueries[b, n, j]));
                            }
                        }
                        if (_queryWeightsGradient != null)
                        {
                            _queryWeightsGradient[h, i, j] = NumOps.Add(_queryWeightsGradient[h, i, j], sum);
                        }
                    }
                }
            }

            // Backward through K projection and update K weight gradients
            var kWeightSlice = ExtractHeadWeights(_keyWeights, h);
            var gradInputFromK = BatchedMatMul3Dx2D(gradKeys, Engine.TensorTranspose(kWeightSlice), batchSize, numNodes, _headDim, _inputFeatures);
            gradInput = Engine.TensorAdd(gradInput, gradInputFromK);

            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < _inputFeatures; i++)
                {
                    for (int j = 0; j < _headDim; j++)
                    {
                        T sum = NumOps.Zero;
                        for (int n = 0; n < numNodes; n++)
                        {
                            if (_lastInput != null)
                            {
                                sum = NumOps.Add(sum, NumOps.Multiply(_lastInput[b, n, i], gradKeys[b, n, j]));
                            }
                        }
                        if (_keyWeightsGradient != null)
                        {
                            _keyWeightsGradient[h, i, j] = NumOps.Add(_keyWeightsGradient[h, i, j], sum);
                        }
                    }
                }
            }
        }

        return gradInput;
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        if (_queryWeightsGradient == null || _keyWeightsGradient == null || _valueWeightsGradient == null ||
            _outputWeightsGradient == null || _outputBiasGradient == null ||
            _ffnWeights1Gradient == null || _ffnWeights2Gradient == null ||
            _ffnBias1Gradient == null || _ffnBias2Gradient == null)
        {
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        }

        // Update all parameters using Engine operations
        var scaledQueryGrad = Engine.TensorMultiplyScalar(_queryWeightsGradient, learningRate);
        _queryWeights = Engine.TensorSubtract(_queryWeights, scaledQueryGrad);

        var scaledKeyGrad = Engine.TensorMultiplyScalar(_keyWeightsGradient, learningRate);
        _keyWeights = Engine.TensorSubtract(_keyWeights, scaledKeyGrad);

        var scaledValueGrad = Engine.TensorMultiplyScalar(_valueWeightsGradient, learningRate);
        _valueWeights = Engine.TensorSubtract(_valueWeights, scaledValueGrad);

        var scaledOutputWeightsGrad = Engine.TensorMultiplyScalar(_outputWeightsGradient, learningRate);
        _outputWeights = Engine.TensorSubtract(_outputWeights, scaledOutputWeightsGrad);

        var scaledOutputBiasGrad = Engine.TensorMultiplyScalar(_outputBiasGradient, learningRate);
        _outputBias = Engine.TensorSubtract(_outputBias, scaledOutputBiasGrad);

        var scaledFFN1Grad = Engine.TensorMultiplyScalar(_ffnWeights1Gradient, learningRate);
        _ffnWeights1 = Engine.TensorSubtract(_ffnWeights1, scaledFFN1Grad);

        var scaledFFN2Grad = Engine.TensorMultiplyScalar(_ffnWeights2Gradient, learningRate);
        _ffnWeights2 = Engine.TensorSubtract(_ffnWeights2, scaledFFN2Grad);

        var scaledFFNBias1Grad = Engine.TensorMultiplyScalar(_ffnBias1Gradient, learningRate);
        _ffnBias1 = Engine.TensorSubtract(_ffnBias1, scaledFFNBias1Grad);

        var scaledFFNBias2Grad = Engine.TensorMultiplyScalar(_ffnBias2Gradient, learningRate);
        _ffnBias2 = Engine.TensorSubtract(_ffnBias2, scaledFFNBias2Grad);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        return Vector<T>.Concatenate(
            new Vector<T>(_queryWeights.ToArray()),
            new Vector<T>(_keyWeights.ToArray()),
            new Vector<T>(_valueWeights.ToArray()),
            new Vector<T>(_outputWeights.ToArray()),
            new Vector<T>(_outputBias.ToArray()),
            new Vector<T>(_ffnWeights1.ToArray()),
            new Vector<T>(_ffnWeights2.ToArray()),
            new Vector<T>(_ffnBias1.ToArray()),
            new Vector<T>(_ffnBias2.ToArray()),
            new Vector<T>(_layerNorm1Scale.ToArray()),
            new Vector<T>(_layerNorm1Bias.ToArray()),
            new Vector<T>(_layerNorm2Scale.ToArray()),
            new Vector<T>(_layerNorm2Bias.ToArray())
        );
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int querySize = _queryWeights.Length;
        int keySize = _keyWeights.Length;
        int valueSize = _valueWeights.Length;
        int outputWeightsSize = _outputWeights.Length;
        int outputBiasSize = _outputBias.Length;
        int ffn1Size = _ffnWeights1.Length;
        int ffn2Size = _ffnWeights2.Length;
        int ffnBias1Size = _ffnBias1.Length;
        int ffnBias2Size = _ffnBias2.Length;
        int ln1ScaleSize = _layerNorm1Scale.Length;
        int ln1BiasSize = _layerNorm1Bias.Length;
        int ln2ScaleSize = _layerNorm2Scale.Length;
        int ln2BiasSize = _layerNorm2Bias.Length;

        int totalParams = querySize + keySize + valueSize + outputWeightsSize + outputBiasSize +
                          ffn1Size + ffn2Size + ffnBias1Size + ffnBias2Size +
                          ln1ScaleSize + ln1BiasSize + ln2ScaleSize + ln2BiasSize;

        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }

        int index = 0;

        var queryParams = parameters.SubVector(index, querySize);
        _queryWeights = Tensor<T>.FromVector(queryParams).Reshape(_queryWeights.Shape);
        index += querySize;

        var keyParams = parameters.SubVector(index, keySize);
        _keyWeights = Tensor<T>.FromVector(keyParams).Reshape(_keyWeights.Shape);
        index += keySize;

        var valueParams = parameters.SubVector(index, valueSize);
        _valueWeights = Tensor<T>.FromVector(valueParams).Reshape(_valueWeights.Shape);
        index += valueSize;

        var outputWeightsParams = parameters.SubVector(index, outputWeightsSize);
        _outputWeights = Tensor<T>.FromVector(outputWeightsParams).Reshape(_outputWeights.Shape);
        index += outputWeightsSize;

        var outputBiasParams = parameters.SubVector(index, outputBiasSize);
        _outputBias = Tensor<T>.FromVector(outputBiasParams);
        index += outputBiasSize;

        var ffn1Params = parameters.SubVector(index, ffn1Size);
        _ffnWeights1 = Tensor<T>.FromVector(ffn1Params).Reshape(_ffnWeights1.Shape);
        index += ffn1Size;

        var ffn2Params = parameters.SubVector(index, ffn2Size);
        _ffnWeights2 = Tensor<T>.FromVector(ffn2Params).Reshape(_ffnWeights2.Shape);
        index += ffn2Size;

        var ffnBias1Params = parameters.SubVector(index, ffnBias1Size);
        _ffnBias1 = Tensor<T>.FromVector(ffnBias1Params);
        index += ffnBias1Size;

        var ffnBias2Params = parameters.SubVector(index, ffnBias2Size);
        _ffnBias2 = Tensor<T>.FromVector(ffnBias2Params);
        index += ffnBias2Size;

        var ln1ScaleParams = parameters.SubVector(index, ln1ScaleSize);
        _layerNorm1Scale = Tensor<T>.FromVector(ln1ScaleParams);
        index += ln1ScaleSize;

        var ln1BiasParams = parameters.SubVector(index, ln1BiasSize);
        _layerNorm1Bias = Tensor<T>.FromVector(ln1BiasParams);
        index += ln1BiasSize;

        var ln2ScaleParams = parameters.SubVector(index, ln2ScaleSize);
        _layerNorm2Scale = Tensor<T>.FromVector(ln2ScaleParams);
        index += ln2ScaleSize;

        var ln2BiasParams = parameters.SubVector(index, ln2BiasSize);
        _layerNorm2Bias = Tensor<T>.FromVector(ln2BiasParams);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastQueries = null;
        _lastKeys = null;
        _lastValues = null;
        _lastAttentionWeights = null;
        _lastHeadOutputs = null;
        _lastConcatenated = null;
        _lastAttnOutput = null;
        _lastNormed1 = null;
        _lastFFNHidden = null;
        _lastFFNOutput = null;
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _outputWeightsGradient = null;
        _outputBiasGradient = null;
        _ffnWeights1Gradient = null;
        _ffnWeights2Gradient = null;
        _ffnBias1Gradient = null;
        _ffnBias2Gradient = null;
    }

    /// <inheritdoc/>
    public override bool SupportsJitCompilation => true;

    /// <inheritdoc/>
    /// <summary>
    /// Exports the layer's forward pass as a JIT-compilable computation graph.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the graph transformer layer.</returns>
    /// <remarks>
    /// <para>
    /// The computation graph implements a simplified Graph Transformer with:
    /// 1. Multi-head self-attention with Q, K, V projections
    /// 2. Output projection and residual connection
    /// 3. Feed-forward network with residual connection
    /// 4. Final activation function
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        // Create symbolic input for node features [batch, nodes, features]
        int numNodes = InputShape[0];
        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "node_features");
        inputNodes.Add(inputNode);

        // Export learnable parameters as constants
        var outputWeightsNode = TensorOperations<T>.Constant(_outputWeights, "output_weights");
        var outputBiasNode = TensorOperations<T>.Constant(_outputBias, "output_bias");
        var ffnWeights1Node = TensorOperations<T>.Constant(_ffnWeights1, "ffn_weights1");
        var ffnWeights2Node = TensorOperations<T>.Constant(_ffnWeights2, "ffn_weights2");
        var ffnBias1Node = TensorOperations<T>.Constant(_ffnBias1, "ffn_bias1");
        var ffnBias2Node = TensorOperations<T>.Constant(_ffnBias2, "ffn_bias2");

        // Build multi-head attention computation graph
        var headOutputNodes = new List<ComputationNode<T>>();

        for (int h = 0; h < _numHeads; h++)
        {
            // Extract weight matrices for this head
            var qWeightSlice = ExtractHeadWeights(_queryWeights, h);
            var kWeightSlice = ExtractHeadWeights(_keyWeights, h);
            var vWeightSlice = ExtractHeadWeights(_valueWeights, h);

            var qWeightNode = TensorOperations<T>.Constant(qWeightSlice, $"query_weights_{h}");
            var kWeightNode = TensorOperations<T>.Constant(kWeightSlice, $"key_weights_{h}");
            var vWeightNode = TensorOperations<T>.Constant(vWeightSlice, $"value_weights_{h}");

            // Q = input @ W_q, K = input @ W_k, V = input @ W_v
            var queries = TensorOperations<T>.MatrixMultiply(inputNode, qWeightNode);
            var keys = TensorOperations<T>.MatrixMultiply(inputNode, kWeightNode);
            var values = TensorOperations<T>.MatrixMultiply(inputNode, vWeightNode);

            // Transpose keys for attention score computation
            var keysT = TensorOperations<T>.Transpose(keys);

            // Attention scores = Q @ K^T / sqrt(d_k)
            var scores = TensorOperations<T>.MatrixMultiply(queries, keysT);
            var scaleFactor = NumOps.Sqrt(NumOps.FromDouble(_headDim));
            var scaleNode = TensorOperations<T>.Constant(new Tensor<T>(new T[] { scaleFactor }, new int[] { 1 }), $"scale_{h}");
            scores = TensorOperations<T>.Divide(scores, scaleNode);

            // Apply softmax to get attention weights
            var attentionWeights = TensorOperations<T>.Softmax(scores, axis: -1);

            // Apply attention to values
            var headOutput = TensorOperations<T>.MatrixMultiply(attentionWeights, values);
            headOutputNodes.Add(headOutput);
        }

        // Concatenate head outputs
        ComputationNode<T> concatenated;
        if (_numHeads == 1)
        {
            concatenated = headOutputNodes[0];
        }
        else
        {
            concatenated = TensorOperations<T>.Concat(headOutputNodes, axis: -1);
        }

        // Output projection: concatenated @ W_out + b_out
        var attnOutput = TensorOperations<T>.MatrixMultiply(concatenated, outputWeightsNode);
        attnOutput = TensorOperations<T>.Add(attnOutput, outputBiasNode);

        // Residual connection (if dimensions match)
        ComputationNode<T> residual1;
        if (_inputFeatures == _outputFeatures)
        {
            residual1 = TensorOperations<T>.Add(attnOutput, inputNode);
        }
        else
        {
            residual1 = attnOutput;
        }

        // Feed-forward network: FFN(x) = W2 * activation(W1 * x + b1) + b2
        var ffnHidden = TensorOperations<T>.MatrixMultiply(residual1, ffnWeights1Node);
        ffnHidden = TensorOperations<T>.Add(ffnHidden, ffnBias1Node);

        // Apply FFN activation (GELU by default)
        if (_ffnActivation.SupportsJitCompilation)
        {
            ffnHidden = _ffnActivation.ApplyToGraph(ffnHidden);
        }
        else
        {
            ffnHidden = TensorOperations<T>.GELU(ffnHidden);
        }

        var ffnOutput = TensorOperations<T>.MatrixMultiply(ffnHidden, ffnWeights2Node);
        ffnOutput = TensorOperations<T>.Add(ffnOutput, ffnBias2Node);

        // Second residual connection
        var output = TensorOperations<T>.Add(residual1, ffnOutput);

        // Apply output activation function if needed
        if (ScalarActivation is not null && ScalarActivation is not IdentityActivation<T>)
        {
            if (ScalarActivation.SupportsJitCompilation)
            {
                output = ScalarActivation.ApplyToGraph(output);
            }
            else
            {
                var activated = ScalarActivation.Activate(output.Value);
                output = TensorOperations<T>.Constant(activated, "activated_output");
            }
        }

        return output;
    }
}
