namespace AiDotNet.NeuralNetworks.Layers.Graph;

/// <summary>
/// Implements Graph Isomorphism Network (GIN) layer for powerful graph representation learning.
/// </summary>
/// <remarks>
/// <para>
/// Graph Isomorphism Networks (GIN), introduced by Xu et al., are provably as powerful as the
/// Weisfeiler-Lehman (WL) graph isomorphism test for distinguishing graph structures. GIN uses
/// a sum aggregation with a learnable epsilon parameter and applies a multi-layer perceptron (MLP)
/// to the aggregated features.
/// </para>
/// <para>
/// The layer computes: h_v^(k) = MLP^(k)((1 + ε^(k)) · h_v^(k-1) + Σ_{u∈N(v)} h_u^(k-1))
/// where h_v is the representation of node v, N(v) is the neighborhood of v,
/// ε is a learnable parameter (or fixed), and MLP is a multi-layer perceptron.
/// </para>
/// <para><b>For Beginners:</b> GIN is designed to be maximally expressive for graph structures.
///
/// Think of it like a very careful observer of patterns:
/// - It can distinguish between different graph structures better than most other methods
/// - It combines information from neighbors in a mathematically optimal way
/// - It's particularly good at tasks where the exact structure of the graph matters
///
/// The key insight is using sum aggregation (not mean or max) and a learnable MLP,
/// which together can capture subtle differences in graph topology.
///
/// Use cases:
/// - Molecular property prediction (where exact molecular structure is critical)
/// - Graph classification (determining if two graphs are structurally different)
/// - Chemical reaction prediction
/// - Any task requiring fine-grained structural understanding
///
/// Real-world example: In drug discovery, GIN can distinguish between molecules that
/// have the same atoms but different structural arrangements (isomers), which may have
/// completely different biological effects.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GraphIsomorphismLayer<T> : LayerBase<T>, IGraphConvolutionLayer<T>
{
    private readonly int _inputFeatures;
    private readonly int _outputFeatures;
    private readonly int _mlpHiddenDim;
    private readonly bool _learnEpsilon;

    /// <summary>
    /// Epsilon parameter for weighting self vs neighbor features.
    /// </summary>
    private T _epsilon;

    /// <summary>
    /// First layer of the MLP: [inputFeatures, mlpHiddenDim].
    /// </summary>
    private Matrix<T> _mlpWeights1;

    /// <summary>
    /// Second layer of the MLP: [mlpHiddenDim, outputFeatures].
    /// </summary>
    private Matrix<T> _mlpWeights2;

    /// <summary>
    /// Bias for first MLP layer.
    /// </summary>
    private Vector<T> _mlpBias1;

    /// <summary>
    /// Bias for second MLP layer.
    /// </summary>
    private Vector<T> _mlpBias2;

    /// <summary>
    /// The adjacency matrix defining graph structure.
    /// </summary>
    private Tensor<T>? _adjacencyMatrix;

    /// <summary>
    /// Cached input from forward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Cached output from forward pass.
    /// </summary>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Cached aggregated features (before MLP).
    /// </summary>
    private Tensor<T>? _lastAggregated;

    /// <summary>
    /// Cached hidden layer output from MLP.
    /// </summary>
    private Tensor<T>? _lastMlpHidden;

    /// <summary>
    /// Gradients for epsilon.
    /// </summary>
    private T _epsilonGradient;

    /// <summary>
    /// Gradients for MLP weights.
    /// </summary>
    private Matrix<T>? _mlpWeights1Gradient;
    private Matrix<T>? _mlpWeights2Gradient;
    private Vector<T>? _mlpBias1Gradient;
    private Vector<T>? _mlpBias2Gradient;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public int InputFeatures => _inputFeatures;

    /// <inheritdoc/>
    public int OutputFeatures => _outputFeatures;

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphIsomorphismLayer{T}"/> class.
    /// </summary>
    /// <param name="inputFeatures">Number of input features per node.</param>
    /// <param name="outputFeatures">Number of output features per node.</param>
    /// <param name="mlpHiddenDim">Hidden dimension for the MLP (default: same as outputFeatures).</param>
    /// <param name="learnEpsilon">Whether to learn epsilon parameter (default: true).</param>
    /// <param name="epsilon">Initial value for epsilon (default: 0.0).</param>
    /// <param name="activationFunction">Activation function for MLP hidden layer.</param>
    /// <remarks>
    /// <para>
    /// Creates a GIN layer with a two-layer MLP. The MLP hidden dimension can be adjusted
    /// to control the expressiveness and computational cost of the layer.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Graph Isomorphism Network layer.
    ///
    /// Key parameters:
    /// - inputFeatures/outputFeatures: Input and output dimensions per node
    /// - mlpHiddenDim: Size of the hidden layer in the MLP (bigger = more expressive but slower)
    /// - learnEpsilon: Whether the network should learn how much to weight self vs neighbors
    ///   * true: Let the network figure out the best balance (usually better)
    ///   * false: Use a fixed epsilon value
    /// - epsilon: Starting value for the self-weighting parameter
    ///
    /// The MLP (Multi-Layer Perceptron) is like a mini neural network inside this layer
    /// that learns complex transformations of the aggregated features.
    /// </para>
    /// </remarks>
    public GraphIsomorphismLayer(
        int inputFeatures,
        int outputFeatures,
        int mlpHiddenDim = -1,
        bool learnEpsilon = true,
        double epsilon = 0.0,
        IActivationFunction<T>? activationFunction = null)
        : base([inputFeatures], [outputFeatures], activationFunction ?? new IdentityActivation<T>())
    {
        _inputFeatures = inputFeatures;
        _outputFeatures = outputFeatures;
        _mlpHiddenDim = mlpHiddenDim > 0 ? mlpHiddenDim : outputFeatures;
        _learnEpsilon = learnEpsilon;
        _epsilon = NumOps.FromDouble(epsilon);

        _mlpWeights1 = new Matrix<T>(_inputFeatures, _mlpHiddenDim);
        _mlpWeights2 = new Matrix<T>(_mlpHiddenDim, _outputFeatures);
        _mlpBias1 = new Vector<T>(_mlpHiddenDim);
        _mlpBias2 = new Vector<T>(_outputFeatures);
        _epsilonGradient = NumOps.Zero;

        InitializeParameters();
    }

    /// <summary>
    /// Initializes layer parameters using Xavier initialization.
    /// </summary>
    private void InitializeParameters()
    {
        // Xavier initialization for first MLP layer
        T scale1 = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_inputFeatures + _mlpHiddenDim)));
        for (int i = 0; i < _mlpWeights1.Rows; i++)
        {
            for (int j = 0; j < _mlpWeights1.Columns; j++)
            {
                _mlpWeights1[i, j] = NumOps.Multiply(
                    NumOps.FromDouble(Random.NextDouble() - 0.5), scale1);
            }
        }

        // Xavier initialization for second MLP layer
        T scale2 = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_mlpHiddenDim + _outputFeatures)));
        for (int i = 0; i < _mlpWeights2.Rows; i++)
        {
            for (int j = 0; j < _mlpWeights2.Columns; j++)
            {
                _mlpWeights2[i, j] = NumOps.Multiply(
                    NumOps.FromDouble(Random.NextDouble() - 0.5), scale2);
            }
        }

        // Initialize biases to zero
        for (int i = 0; i < _mlpBias1.Length; i++)
        {
            _mlpBias1[i] = NumOps.Zero;
        }

        for (int i = 0; i < _mlpBias2.Length; i++)
        {
            _mlpBias2[i] = NumOps.Zero;
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
    /// Applies ReLU activation.
    /// </summary>
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

        // Step 1: Aggregate neighbor features using sum
        var neighborSum = new Tensor<T>([batchSize, numNodes, _inputFeatures]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                for (int f = 0; f < _inputFeatures; f++)
                {
                    T sum = NumOps.Zero;
                    for (int j = 0; j < numNodes; j++)
                    {
                        if (!NumOps.Equals(_adjacencyMatrix[b, i, j], NumOps.Zero))
                        {
                            sum = NumOps.Add(sum, input[b, j, f]);
                        }
                    }
                    neighborSum[b, i, f] = sum;
                }
            }
        }

        // Step 2: Combine with self features: (1 + ε) * h_v + Σ h_u
        _lastAggregated = new Tensor<T>([batchSize, numNodes, _inputFeatures]);
        T onePlusEpsilon = NumOps.Add(NumOps.FromDouble(1.0), _epsilon);

        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int f = 0; f < _inputFeatures; f++)
                {
                    _lastAggregated[b, n, f] = NumOps.Add(
                        NumOps.Multiply(onePlusEpsilon, input[b, n, f]),
                        neighborSum[b, n, f]);
                }
            }
        }

        // Step 3: Apply MLP - First layer with ReLU
        _lastMlpHidden = new Tensor<T>([batchSize, numNodes, _mlpHiddenDim]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int h = 0; h < _mlpHiddenDim; h++)
                {
                    T sum = _mlpBias1[h];
                    for (int f = 0; f < _inputFeatures; f++)
                    {
                        sum = NumOps.Add(sum,
                            NumOps.Multiply(_lastAggregated[b, n, f], _mlpWeights1[f, h]));
                    }
                    _lastMlpHidden[b, n, h] = ReLU(sum);
                }
            }
        }

        // Step 4: Apply MLP - Second layer
        var mlpOutput = new Tensor<T>([batchSize, numNodes, _outputFeatures]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int outF = 0; outF < _outputFeatures; outF++)
                {
                    T sum = _mlpBias2[outF];
                    for (int h = 0; h < _mlpHiddenDim; h++)
                    {
                        sum = NumOps.Add(sum,
                            NumOps.Multiply(_lastMlpHidden[b, n, h], _mlpWeights2[h, outF]));
                    }
                    mlpOutput[b, n, outF] = sum;
                }
            }
        }

        _lastOutput = ApplyActivation(mlpOutput);
        return _lastOutput;
    }

    /// <inheritdoc/>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _adjacencyMatrix == null ||
            _lastAggregated == null || _lastMlpHidden == null)
        {
            throw new InvalidOperationException("Forward pass must be called before Backward.");
        }

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);
        int batchSize = _lastInput.Shape[0];
        int numNodes = _lastInput.Shape[1];

        // Initialize gradients
        _mlpWeights1Gradient = new Matrix<T>(_inputFeatures, _mlpHiddenDim);
        _mlpWeights2Gradient = new Matrix<T>(_mlpHiddenDim, _outputFeatures);
        _mlpBias1Gradient = new Vector<T>(_mlpHiddenDim);
        _mlpBias2Gradient = new Vector<T>(_outputFeatures);
        _epsilonGradient = NumOps.Zero;

        // Backprop through second MLP layer
        var hiddenGradient = new Tensor<T>([batchSize, numNodes, _mlpHiddenDim]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int outF = 0; outF < _outputFeatures; outF++)
                {
                    T outGrad = activationGradient[b, n, outF];

                    // Bias gradient
                    _mlpBias2Gradient[outF] = NumOps.Add(_mlpBias2Gradient[outF], outGrad);

                    // Weight gradient and backprop to hidden
                    for (int h = 0; h < _mlpHiddenDim; h++)
                    {
                        _mlpWeights2Gradient[h, outF] = NumOps.Add(
                            _mlpWeights2Gradient[h, outF],
                            NumOps.Multiply(_lastMlpHidden[b, n, h], outGrad));

                        hiddenGradient[b, n, h] = NumOps.Add(
                            hiddenGradient[b, n, h],
                            NumOps.Multiply(_mlpWeights2[h, outF], outGrad));
                    }
                }
            }
        }

        // Backprop through ReLU and first MLP layer
        var aggregatedGradient = new Tensor<T>([batchSize, numNodes, _inputFeatures]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int h = 0; h < _mlpHiddenDim; h++)
                {
                    // ReLU derivative
                    T reluGrad = NumOps.GreaterThan(_lastMlpHidden[b, n, h], NumOps.Zero)
                        ? hiddenGradient[b, n, h]
                        : NumOps.Zero;

                    // Bias gradient
                    _mlpBias1Gradient[h] = NumOps.Add(_mlpBias1Gradient[h], reluGrad);

                    // Weight gradient and backprop to aggregated
                    for (int f = 0; f < _inputFeatures; f++)
                    {
                        _mlpWeights1Gradient[f, h] = NumOps.Add(
                            _mlpWeights1Gradient[f, h],
                            NumOps.Multiply(_lastAggregated[b, n, f], reluGrad));

                        aggregatedGradient[b, n, f] = NumOps.Add(
                            aggregatedGradient[b, n, f],
                            NumOps.Multiply(_mlpWeights1[f, h], reluGrad));
                    }
                }
            }
        }

        // Backprop through aggregation
        var inputGradient = new Tensor<T>(_lastInput.Shape);
        T onePlusEpsilon = NumOps.Add(NumOps.FromDouble(1.0), _epsilon);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                for (int f = 0; f < _inputFeatures; f++)
                {
                    // Gradient from self connection (1 + ε)
                    T selfGrad = NumOps.Multiply(onePlusEpsilon, aggregatedGradient[b, i, f]);
                    inputGradient[b, i, f] = NumOps.Add(inputGradient[b, i, f], selfGrad);

                    // Epsilon gradient (if learning)
                    if (_learnEpsilon)
                    {
                        _epsilonGradient = NumOps.Add(_epsilonGradient,
                            NumOps.Multiply(_lastInput[b, i, f], aggregatedGradient[b, i, f]));
                    }

                    // Gradient from neighbor aggregation
                    for (int j = 0; j < numNodes; j++)
                    {
                        if (!NumOps.Equals(_adjacencyMatrix[b, j, i], NumOps.Zero))
                        {
                            inputGradient[b, i, f] = NumOps.Add(
                                inputGradient[b, i, f],
                                aggregatedGradient[b, j, f]);
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
        if (_mlpWeights1Gradient == null || _mlpWeights2Gradient == null ||
            _mlpBias1Gradient == null || _mlpBias2Gradient == null)
        {
            throw new InvalidOperationException("Backward must be called before UpdateParameters.");
        }

        // Update MLP weights and biases
        _mlpWeights1 = _mlpWeights1.Subtract(_mlpWeights1Gradient.Multiply(learningRate));
        _mlpWeights2 = _mlpWeights2.Subtract(_mlpWeights2Gradient.Multiply(learningRate));
        _mlpBias1 = _mlpBias1.Subtract(_mlpBias1Gradient.Multiply(learningRate));
        _mlpBias2 = _mlpBias2.Subtract(_mlpBias2Gradient.Multiply(learningRate));

        // Update epsilon if learnable
        if (_learnEpsilon)
        {
            _epsilon = NumOps.Subtract(_epsilon, NumOps.Multiply(learningRate, _epsilonGradient));
        }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        int mlpParams = _inputFeatures * _mlpHiddenDim + _mlpHiddenDim +
                       _mlpHiddenDim * _outputFeatures + _outputFeatures;
        int totalParams = mlpParams + (_learnEpsilon ? 1 : 0);

        var parameters = new Vector<T>(totalParams);
        int index = 0;

        // MLP weights 1
        for (int i = 0; i < _mlpWeights1.Rows; i++)
        {
            for (int j = 0; j < _mlpWeights1.Columns; j++)
            {
                parameters[index++] = _mlpWeights1[i, j];
            }
        }

        // MLP bias 1
        for (int i = 0; i < _mlpBias1.Length; i++)
        {
            parameters[index++] = _mlpBias1[i];
        }

        // MLP weights 2
        for (int i = 0; i < _mlpWeights2.Rows; i++)
        {
            for (int j = 0; j < _mlpWeights2.Columns; j++)
            {
                parameters[index++] = _mlpWeights2[i, j];
            }
        }

        // MLP bias 2
        for (int i = 0; i < _mlpBias2.Length; i++)
        {
            parameters[index++] = _mlpBias2[i];
        }

        // Epsilon (if learnable)
        if (_learnEpsilon)
        {
            parameters[index] = _epsilon;
        }

        return parameters;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int mlpParams = _inputFeatures * _mlpHiddenDim + _mlpHiddenDim +
                       _mlpHiddenDim * _outputFeatures + _outputFeatures;
        int expectedParams = mlpParams + (_learnEpsilon ? 1 : 0);

        if (parameters.Length != expectedParams)
        {
            throw new ArgumentException(
                $"Expected {expectedParams} parameters, but got {parameters.Length}");
        }

        int index = 0;

        // Set MLP weights 1
        for (int i = 0; i < _mlpWeights1.Rows; i++)
        {
            for (int j = 0; j < _mlpWeights1.Columns; j++)
            {
                _mlpWeights1[i, j] = parameters[index++];
            }
        }

        // Set MLP bias 1
        for (int i = 0; i < _mlpBias1.Length; i++)
        {
            _mlpBias1[i] = parameters[index++];
        }

        // Set MLP weights 2
        for (int i = 0; i < _mlpWeights2.Rows; i++)
        {
            for (int j = 0; j < _mlpWeights2.Columns; j++)
            {
                _mlpWeights2[i, j] = parameters[index++];
            }
        }

        // Set MLP bias 2
        for (int i = 0; i < _mlpBias2.Length; i++)
        {
            _mlpBias2[i] = parameters[index++];
        }

        // Set epsilon (if learnable)
        if (_learnEpsilon)
        {
            _epsilon = parameters[index];
        }
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastAggregated = null;
        _lastMlpHidden = null;
        _mlpWeights1Gradient = null;
        _mlpWeights2Gradient = null;
        _mlpBias1Gradient = null;
        _mlpBias2Gradient = null;
        _epsilonGradient = NumOps.Zero;
    }

    /// <inheritdoc/>
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc/>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException(
            "GraphIsomorphismLayer does not support computation graph export due to dynamic graph-based aggregation.");
    }
}
