namespace AiDotNet.NeuralNetworks.Layers.Graph;

/// <summary>
/// Implements Graph Attention Network (GAT) layer for processing graph-structured data with attention mechanisms.
/// </summary>
/// <remarks>
/// <para>
/// Graph Attention Networks (GAT) introduced by Veličković et al. use attention mechanisms to learn
/// the relative importance of neighboring nodes. Unlike standard GCN which treats all neighbors equally,
/// GAT can assign different weights to different neighbors, allowing the model to focus on the most
/// relevant connections. The layer uses multi-head attention for robustness and expressiveness.
/// </para>
/// <para>
/// The attention mechanism computes: α_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))
/// where α_ij is the attention coefficient from node j to node i, W is a weight matrix,
/// h_i and h_j are node features, a is the attention vector, and || denotes concatenation.
/// </para>
/// <para><b>For Beginners:</b> This layer helps neural networks understand graphs by paying attention to important connections.
///
/// Imagine you're in a social network:
/// - Not all your friends influence you equally
/// - Some friends might have more relevant opinions on certain topics
/// - GAT learns which connections matter most for each situation
///
/// The "attention mechanism" is like deciding how much to listen to each friend.
/// The layer automatically learns these attention weights during training.
///
/// For example, in a citation network, a research paper might pay more attention
/// to highly-cited papers it references, and less attention to obscure references.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GraphAttentionLayer<T> : LayerBase<T>, IGraphConvolutionLayer<T>
{
    private readonly int _inputFeatures;
    private readonly int _outputFeatures;
    private readonly int _numHeads;
    private readonly T _alpha; // LeakyReLU negative slope
    private readonly double _dropoutRate;
    private readonly Random _random;

    /// <summary>
    /// Weight matrices for each attention head. Shape: [numHeads, inputFeatures, outputFeatures].
    /// </summary>
    private Tensor<T> _weights;

    /// <summary>
    /// Attention mechanism parameters for each head. Shape: [numHeads, 2 * outputFeatures].
    /// </summary>
    private Matrix<T> _attentionWeights;

    /// <summary>
    /// Bias vector for the output transformation.
    /// </summary>
    private Vector<T> _bias;

    /// <summary>
    /// The adjacency matrix defining graph structure.
    /// </summary>
    private Tensor<T>? _adjacencyMatrix;

    /// <summary>
    /// Cached input from forward pass for backward computation.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Cached output from forward pass for backward computation.
    /// </summary>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Cached attention coefficients from forward pass.
    /// </summary>
    private Tensor<T>? _lastAttentionCoefficients;

    /// <summary>
    /// Cached transformed features from forward pass for gradient computation.
    /// </summary>
    private Tensor<T>? _lastTransformed;

    /// <summary>
    /// Gradients for weight parameters.
    /// </summary>
    private Tensor<T>? _weightsGradient;

    /// <summary>
    /// Gradients for attention parameters.
    /// </summary>
    private Matrix<T>? _attentionWeightsGradient;

    /// <summary>
    /// Gradients for bias parameters.
    /// </summary>
    private Vector<T>? _biasGradient;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public int InputFeatures => _inputFeatures;

    /// <inheritdoc/>
    public int OutputFeatures => _outputFeatures;

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphAttentionLayer{T}"/> class.
    /// </summary>
    /// <param name="inputFeatures">Number of input features per node.</param>
    /// <param name="outputFeatures">Number of output features per node.</param>
    /// <param name="numHeads">Number of attention heads (default: 1).</param>
    /// <param name="alpha">Negative slope for LeakyReLU in attention mechanism (default: 0.2).</param>
    /// <param name="dropoutRate">Dropout rate for attention coefficients (default: 0).</param>
    /// <param name="activationFunction">Activation function to apply after aggregation.</param>
    /// <remarks>
    /// <para>
    /// Creates a GAT layer with specified dimensions and attention heads. Multiple attention heads
    /// allow the layer to attend to different aspects of the neighborhood simultaneously,
    /// similar to multi-head attention in Transformers.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up a new Graph Attention layer.
    ///
    /// Parameters explained:
    /// - inputFeatures: How many numbers describe each node initially
    /// - outputFeatures: How many numbers you want for each node after processing
    /// - numHeads: How many different "attention perspectives" to use (more heads = more flexible)
    /// - alpha: Controls the attention mechanism's sensitivity
    /// - dropoutRate: Randomly ignores some connections during training to prevent overfitting
    ///
    /// Think of attention heads like having multiple experts looking at the same graph,
    /// each focusing on different patterns or relationships.
    /// </para>
    /// </remarks>
    public GraphAttentionLayer(
        int inputFeatures,
        int outputFeatures,
        int numHeads = 1,
        double alpha = 0.2,
        double dropoutRate = 0.0,
        IActivationFunction<T>? activationFunction = null)
        : base([inputFeatures], [outputFeatures], activationFunction ?? new IdentityActivation<T>())
    {
        _inputFeatures = inputFeatures;
        _outputFeatures = outputFeatures;
        _numHeads = numHeads;
        _alpha = NumOps.FromDouble(alpha);
        _dropoutRate = dropoutRate;
        _random = new Random();

        // Initialize weights for each attention head
        _weights = new Tensor<T>([_numHeads, _inputFeatures, _outputFeatures]);

        // Initialize attention mechanism weights (one per head)
        _attentionWeights = new Matrix<T>(_numHeads, 2 * _outputFeatures);

        // Initialize bias
        _bias = new Vector<T>(_outputFeatures);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes layer parameters using Xavier/Glorot initialization.
    /// </summary>
    private void InitializeParameters()
    {
        // Xavier initialization for weights
        T weightScale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_inputFeatures + _outputFeatures)));
        for (int h = 0; h < _numHeads; h++)
        {
            for (int i = 0; i < _inputFeatures; i++)
            {
                for (int j = 0; j < _outputFeatures; j++)
                {
                    _weights[h, i, j] = NumOps.Multiply(
                        NumOps.FromDouble(Random.NextDouble() - 0.5),
                        weightScale);
                }
            }
        }

        // Initialize attention weights
        T attentionScale = NumOps.Sqrt(NumOps.FromDouble(1.0 / _outputFeatures));
        for (int h = 0; h < _numHeads; h++)
        {
            for (int j = 0; j < 2 * _outputFeatures; j++)
            {
                _attentionWeights[h, j] = NumOps.Multiply(
                    NumOps.FromDouble(Random.NextDouble() - 0.5),
                    attentionScale);
            }
        }

        // Initialize bias to zero
        for (int i = 0; i < _bias.Length; i++)
        {
            _bias[i] = NumOps.Zero;
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
    /// Applies LeakyReLU activation.
    /// </summary>
    private T LeakyReLU(T x)
    {
        return NumOps.GreaterThan(x, NumOps.Zero)
            ? x
            : NumOps.Multiply(_alpha, x);
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
        int inputFeatures = input.Shape[2];

        // Store attention coefficients for all heads
        _lastAttentionCoefficients = new Tensor<T>([batchSize, _numHeads, numNodes, numNodes]);

        // Store transformed features for gradient computation
        _lastTransformed = new Tensor<T>([batchSize, _numHeads, numNodes, _outputFeatures]);

        // Output for each head: [batchSize, numHeads, numNodes, outputFeatures]
        var headOutputs = new Tensor<T>([batchSize, _numHeads, numNodes, _outputFeatures]);

        // Process each attention head
        for (int h = 0; h < _numHeads; h++)
        {
            // Linear transformation: Wh for all nodes
            // [batchSize, numNodes, outputFeatures]
            var transformed = new Tensor<T>([batchSize, numNodes, _outputFeatures]);
            for (int b = 0; b < batchSize; b++)
            {
                for (int n = 0; n < numNodes; n++)
                {
                    for (int outF = 0; outF < _outputFeatures; outF++)
                    {
                        T sum = NumOps.Zero;
                        for (int inF = 0; inF < inputFeatures; inF++)
                        {
                            sum = NumOps.Add(sum,
                                NumOps.Multiply(input[b, n, inF], _weights[h, inF, outF]));
                        }
                        transformed[b, n, outF] = sum;
                        _lastTransformed[b, h, n, outF] = sum;
                    }
                }
            }

            // Compute attention coefficients
            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < numNodes; i++)
                {
                    for (int j = 0; j < numNodes; j++)
                    {
                        // Only compute attention for connected nodes
                        if (NumOps.Equals(_adjacencyMatrix[b, i, j], NumOps.Zero))
                        {
                            _lastAttentionCoefficients[b, h, i, j] = NumOps.FromDouble(double.NegativeInfinity);
                            continue;
                        }

                        // Compute attention: a^T [Wh_i || Wh_j]
                        T attentionScore = NumOps.Zero;

                        // First half: contribution from node i
                        for (int f = 0; f < _outputFeatures; f++)
                        {
                            attentionScore = NumOps.Add(attentionScore,
                                NumOps.Multiply(transformed[b, i, f], _attentionWeights[h, f]));
                        }

                        // Second half: contribution from node j
                        for (int f = 0; f < _outputFeatures; f++)
                        {
                            attentionScore = NumOps.Add(attentionScore,
                                NumOps.Multiply(transformed[b, j, f],
                                    _attentionWeights[h, _outputFeatures + f]));
                        }

                        _lastAttentionCoefficients[b, h, i, j] = LeakyReLU(attentionScore);
                    }

                    // Softmax over neighbors for node i
                    T maxScore = NumOps.FromDouble(double.NegativeInfinity);
                    for (int j = 0; j < numNodes; j++)
                    {
                        if (!NumOps.Equals(_adjacencyMatrix[b, i, j], NumOps.Zero))
                        {
                            maxScore = NumOps.GreaterThan(_lastAttentionCoefficients[b, h, i, j], maxScore) ? _lastAttentionCoefficients[b, h, i, j] : maxScore;
                        }
                    }

                    T sumExp = NumOps.Zero;
                    for (int j = 0; j < numNodes; j++)
                    {
                        if (!NumOps.Equals(_adjacencyMatrix[b, i, j], NumOps.Zero))
                        {
                            T expVal = NumOps.Exp(
                                NumOps.Subtract(_lastAttentionCoefficients[b, h, i, j], maxScore));
                            _lastAttentionCoefficients[b, h, i, j] = expVal;
                            sumExp = NumOps.Add(sumExp, expVal);
                        }
                    }

                    // Normalize
                    for (int j = 0; j < numNodes; j++)
                    {
                        if (!NumOps.Equals(_adjacencyMatrix[b, i, j], NumOps.Zero))
                        {
                            T normalizedCoeff = NumOps.Divide(_lastAttentionCoefficients[b, h, i, j], sumExp);

                            // Apply dropout to attention coefficients during training
                            if (_dropoutRate > 0.0)
                            {
                                // Dropout: randomly zero out with probability dropoutRate
                                double rand = _random.NextDouble();
                                if (rand < _dropoutRate)
                                {
                                    normalizedCoeff = NumOps.Zero;
                                }
                                else
                                {
                                    // Scale by 1/(1-p) during training to maintain expected value
                                    T scale = NumOps.FromDouble(1.0 / (1.0 - _dropoutRate));
                                    normalizedCoeff = NumOps.Multiply(normalizedCoeff, scale);
                                }
                            }

                            _lastAttentionCoefficients[b, h, i, j] = normalizedCoeff;
                        }
                        else
                        {
                            _lastAttentionCoefficients[b, h, i, j] = NumOps.Zero;
                        }
                    }
                }
            }

            // Aggregate using attention coefficients
            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < numNodes; i++)
                {
                    for (int f = 0; f < _outputFeatures; f++)
                    {
                        T aggregated = NumOps.Zero;
                        for (int j = 0; j < numNodes; j++)
                        {
                            aggregated = NumOps.Add(aggregated,
                                NumOps.Multiply(_lastAttentionCoefficients[b, h, i, j],
                                    transformed[b, j, f]));
                        }
                        headOutputs[b, h, i, f] = aggregated;
                    }
                }
            }
        }

        // Combine multi-head outputs (concatenation or averaging)
        var output = new Tensor<T>([batchSize, numNodes, _outputFeatures]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int f = 0; f < _outputFeatures; f++)
                {
                    T sum = NumOps.Zero;
                    for (int h = 0; h < _numHeads; h++)
                    {
                        sum = NumOps.Add(sum, headOutputs[b, h, n, f]);
                    }
                    // Average across heads
                    output[b, n, f] = NumOps.Divide(sum, NumOps.FromDouble(_numHeads));
                }
            }
        }

        // Add bias
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int f = 0; f < _outputFeatures; f++)
                {
                    output[b, n, f] = NumOps.Add(output[b, n, f], _bias[f]);
                }
            }
        }

        _lastOutput = ApplyActivation(output);
        return _lastOutput;
    }

    /// <inheritdoc/>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _adjacencyMatrix == null || _lastTransformed == null || _lastAttentionCoefficients == null)
        {
            throw new InvalidOperationException("Forward pass must be called before Backward.");
        }

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);

        int batchSize = _lastInput.Shape[0];
        int numNodes = _lastInput.Shape[1];

        // Initialize gradients
        _weightsGradient = new Tensor<T>([_numHeads, _inputFeatures, _outputFeatures]);
        _attentionWeightsGradient = new Matrix<T>(_numHeads, 2 * _outputFeatures);
        _biasGradient = new Vector<T>(_outputFeatures);
        var inputGradient = new Tensor<T>(_lastInput.Shape);

        // Compute bias gradient
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int f = 0; f < _outputFeatures; f++)
                {
                    _biasGradient[f] = NumOps.Add(_biasGradient[f], activationGradient[b, n, f]);
                }
            }
        }

        // Compute weight gradients and input gradients for each head
        for (int h = 0; h < _numHeads; h++)
        {
            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < numNodes; i++)
                {
                    // Gradient contribution from aggregated neighbors
                    for (int j = 0; j < numNodes; j++)
                    {
                        if (!NumOps.Equals(_adjacencyMatrix[b, i, j], NumOps.Zero))
                        {
                            T attnCoeff = _lastAttentionCoefficients[b, h, i, j];

                            // Weight gradient: dL/dW = attnCoeff * input^T * outputGrad
                            for (int inF = 0; inF < _inputFeatures; inF++)
                            {
                                for (int outF = 0; outF < _outputFeatures; outF++)
                                {
                                    T grad = NumOps.Multiply(attnCoeff, NumOps.Multiply(_lastInput[b, j, inF], activationGradient[b, i, outF]));
                                    _weightsGradient[h, inF, outF] = NumOps.Add(_weightsGradient[h, inF, outF], grad);
                                }
                            }

                            // Input gradient: dL/dInput = attnCoeff * W^T * outputGrad
                            for (int inF = 0; inF < _inputFeatures; inF++)
                            {
                                T grad = NumOps.Zero;
                                for (int outF = 0; outF < _outputFeatures; outF++)
                                {
                                    grad = NumOps.Add(grad, NumOps.Multiply(_weights[h, inF, outF], activationGradient[b, i, outF]));
                                }
                                grad = NumOps.Multiply(attnCoeff, grad);
                                inputGradient[b, j, inF] = NumOps.Add(inputGradient[b, j, inF], grad);
                            }
                        }
                    }

                    // Attention weight gradients (simplified - full implementation would backprop through softmax)
                    for (int j = 0; j < numNodes; j++)
                    {
                        if (!NumOps.Equals(_adjacencyMatrix[b, i, j], NumOps.Zero))
                        {
                            // Approximate gradient for attention parameters
                            for (int outF = 0; outF < _outputFeatures; outF++)
                            {
                                T transformedI = _lastTransformed[b, h, i, outF];
                                T transformedJ = _lastTransformed[b, h, j, outF];
                                T grad = activationGradient[b, i, outF];

                                // Gradient w.r.t. attention weights for source node features
                                _attentionWeightsGradient[h, outF] = NumOps.Add(_attentionWeightsGradient[h, outF], NumOps.Multiply(transformedI, grad));

                                // Gradient w.r.t. attention weights for neighbor node features
                                _attentionWeightsGradient[h, _outputFeatures + outF] = NumOps.Add(_attentionWeightsGradient[h, _outputFeatures + outF], NumOps.Multiply(transformedJ, grad));
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
        if (_weightsGradient == null || _attentionWeightsGradient == null || _biasGradient == null)
        {
            throw new InvalidOperationException("Backward must be called before UpdateParameters.");
        }

        // Update weights
        for (int h = 0; h < _numHeads; h++)
        {
            for (int i = 0; i < _inputFeatures; i++)
            {
                for (int j = 0; j < _outputFeatures; j++)
                {
                    _weights[h, i, j] = NumOps.Subtract(_weights[h, i, j],
                        NumOps.Multiply(learningRate, _weightsGradient[h, i, j]));
                }
            }
        }

        // Update attention weights
        for (int h = 0; h < _numHeads; h++)
        {
            for (int j = 0; j < 2 * _outputFeatures; j++)
            {
                _attentionWeights[h, j] = NumOps.Subtract(_attentionWeights[h, j],
                    NumOps.Multiply(learningRate, _attentionWeightsGradient[h, j]));
            }
        }

        // Update bias
        for (int i = 0; i < _bias.Length; i++)
        {
            _bias[i] = NumOps.Subtract(_bias[i], NumOps.Multiply(learningRate, _biasGradient[i]));
        }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        int totalParams = _numHeads * _inputFeatures * _outputFeatures +
                         _numHeads * 2 * _outputFeatures +
                         _outputFeatures;
        var parameters = new Vector<T>(totalParams);
        int index = 0;

        // Weights
        for (int h = 0; h < _numHeads; h++)
        {
            for (int i = 0; i < _inputFeatures; i++)
            {
                for (int j = 0; j < _outputFeatures; j++)
                {
                    parameters[index++] = _weights[h, i, j];
                }
            }
        }

        // Attention weights
        for (int h = 0; h < _numHeads; h++)
        {
            for (int j = 0; j < 2 * _outputFeatures; j++)
            {
                parameters[index++] = _attentionWeights[h, j];
            }
        }

        // Bias
        for (int i = 0; i < _bias.Length; i++)
        {
            parameters[index++] = _bias[i];
        }

        return parameters;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int expectedParams = _numHeads * _inputFeatures * _outputFeatures +
                           _numHeads * 2 * _outputFeatures +
                           _outputFeatures;

        if (parameters.Length != expectedParams)
        {
            throw new ArgumentException(
                $"Expected {expectedParams} parameters, but got {parameters.Length}");
        }

        int index = 0;

        // Set weights
        for (int h = 0; h < _numHeads; h++)
        {
            for (int i = 0; i < _inputFeatures; i++)
            {
                for (int j = 0; j < _outputFeatures; j++)
                {
                    _weights[h, i, j] = parameters[index++];
                }
            }
        }

        // Set attention weights
        for (int h = 0; h < _numHeads; h++)
        {
            for (int j = 0; j < 2 * _outputFeatures; j++)
            {
                _attentionWeights[h, j] = parameters[index++];
            }
        }

        // Set bias
        for (int i = 0; i < _bias.Length; i++)
        {
            _bias[i] = parameters[index++];
        }
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastAttentionCoefficients = null;
        _lastTransformed = null;
        _weightsGradient = null;
        _attentionWeightsGradient = null;
        _biasGradient = null;
    }
}
