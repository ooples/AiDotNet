namespace AiDotNet.NeuralNetworks.Layers.Graph;

/// <summary>
/// Aggregation function type for GraphSAGE.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These are different ways to combine information from neighbors.
///
/// - Mean: Average all neighbor features (balanced, smooth)
/// - Max: Take the maximum value from neighbors (emphasizes outliers)
/// - Sum: Add up all neighbor features (sensitive to number of neighbors)
/// - LSTM: Use a recurrent network to aggregate (most expressive but slowest)
/// </para>
/// </remarks>
public enum SAGEAggregatorType
{
    /// <summary>
    /// Mean aggregation: averages neighbor features.
    /// </summary>
    Mean,

    /// <summary>
    /// Max pooling aggregation: takes maximum of neighbor features.
    /// </summary>
    MaxPool,

    /// <summary>
    /// Sum aggregation: sums neighbor features.
    /// </summary>
    Sum
}

/// <summary>
/// Implements GraphSAGE (Graph Sample and Aggregate) layer for inductive learning on graphs.
/// </summary>
/// <remarks>
/// <para>
/// GraphSAGE, introduced by Hamilton et al., is designed for inductive learning on graphs,
/// meaning it can generalize to unseen nodes and graphs. Instead of learning embeddings for
/// each node directly, it learns aggregator functions that generate embeddings by sampling
/// and aggregating features from a node's local neighborhood.
/// </para>
/// <para>
/// The layer performs: h_v = σ(W · CONCAT(h_v, AGGREGATE({h_u : u ∈ N(v)})))
/// where h_v is the representation of node v, N(v) is the neighborhood of v,
/// AGGREGATE is an aggregation function (mean, max, LSTM), and σ is an activation function.
/// </para>
/// <para><b>For Beginners:</b> GraphSAGE is like learning a recipe for combining neighbor information.
///
/// Think of it like getting advice from friends:
/// - You have your own opinion (your node features)
/// - You ask your friends for their opinions (neighbor features)
/// - You combine everyone's input in a smart way (aggregation)
/// - You form your final opinion (updated node features)
///
/// The key advantage is that this "recipe" can work on new people (nodes) you haven't seen before,
/// as long as they have the same types of features.
///
/// Use cases:
/// - Social networks: Predict properties of new users based on their connections
/// - Recommendation systems: Suggest items to new users
/// - Molecular graphs: Predict properties of new molecules
/// - Knowledge graphs: Infer facts about new entities
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GraphSAGELayer<T> : LayerBase<T>, IGraphConvolutionLayer<T>
{
    private readonly int _inputFeatures;
    private readonly int _outputFeatures;
    private readonly SAGEAggregatorType _aggregatorType;
    private readonly bool _normalize;

    /// <summary>
    /// Weight matrix for self features.
    /// </summary>
    private Matrix<T> _selfWeights;

    /// <summary>
    /// Weight matrix for neighbor features.
    /// </summary>
    private Matrix<T> _neighborWeights;

    /// <summary>
    /// Bias vector.
    /// </summary>
    private Vector<T> _bias;

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
    /// Cached aggregated neighbor features.
    /// </summary>
    private Tensor<T>? _lastAggregated;

    /// <summary>
    /// Gradients for self weights.
    /// </summary>
    private Matrix<T>? _selfWeightsGradient;

    /// <summary>
    /// Gradients for neighbor weights.
    /// </summary>
    private Matrix<T>? _neighborWeightsGradient;

    /// <summary>
    /// Gradients for bias.
    /// </summary>
    private Vector<T>? _biasGradient;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public int InputFeatures => _inputFeatures;

    /// <inheritdoc/>
    public int OutputFeatures => _outputFeatures;

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphSAGELayer{T}"/> class.
    /// </summary>
    /// <param name="inputFeatures">Number of input features per node.</param>
    /// <param name="outputFeatures">Number of output features per node.</param>
    /// <param name="aggregatorType">Type of aggregation function to use.</param>
    /// <param name="normalize">Whether to L2-normalize output features.</param>
    /// <param name="activationFunction">Activation function to apply.</param>
    /// <remarks>
    /// <para>
    /// Creates a GraphSAGE layer with the specified aggregator function. The aggregator
    /// determines how information from neighbors is combined.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new GraphSAGE layer.
    ///
    /// Key parameters:
    /// - aggregatorType: How to combine neighbor information
    ///   * Mean: Average everyone's opinion (most common)
    ///   * MaxPool: Take the strongest signal from any neighbor
    ///   * Sum: Add up all neighbor contributions
    /// - normalize: Whether to normalize the output (helps with stability)
    ///
    /// Example: For a social network with 64 features per user, you might use:
    /// new GraphSAGELayer(64, 128, SAGEAggregatorType.Mean, normalize: true)
    /// </para>
    /// </remarks>
    public GraphSAGELayer(
        int inputFeatures,
        int outputFeatures,
        SAGEAggregatorType aggregatorType = SAGEAggregatorType.Mean,
        bool normalize = true,
        IActivationFunction<T>? activationFunction = null)
        : base([inputFeatures], [outputFeatures], activationFunction ?? new IdentityActivation<T>())
    {
        _inputFeatures = inputFeatures;
        _outputFeatures = outputFeatures;
        _aggregatorType = aggregatorType;
        _normalize = normalize;

        _selfWeights = new Matrix<T>(_inputFeatures, _outputFeatures);
        _neighborWeights = new Matrix<T>(_inputFeatures, _outputFeatures);
        _bias = new Vector<T>(_outputFeatures);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes layer parameters using Xavier initialization.
    /// </summary>
    private void InitializeParameters()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_inputFeatures + _outputFeatures)));

        // Initialize self weights
        for (int i = 0; i < _selfWeights.Rows; i++)
        {
            for (int j = 0; j < _selfWeights.Columns; j++)
            {
                _selfWeights[i, j] = NumOps.Multiply(
                    NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
            }
        }

        // Initialize neighbor weights
        for (int i = 0; i < _neighborWeights.Rows; i++)
        {
            for (int j = 0; j < _neighborWeights.Columns; j++)
            {
                _neighborWeights[i, j] = NumOps.Multiply(
                    NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
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
    /// Aggregates neighbor features according to the specified aggregator type.
    /// </summary>
    private Tensor<T> AggregateNeighbors(Tensor<T> input, int batchSize, int numNodes)
    {
        if (_adjacencyMatrix == null)
        {
            throw new InvalidOperationException("Adjacency matrix not set.");
        }

        var aggregated = new Tensor<T>([batchSize, numNodes, _inputFeatures]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                // Count neighbors
                int neighborCount = 0;
                for (int j = 0; j < numNodes; j++)
                {
                    if (!NumOps.Equals(_adjacencyMatrix[b, i, j], NumOps.Zero))
                    {
                        neighborCount++;
                    }
                }

                if (neighborCount == 0)
                {
                    // No neighbors, use zeros
                    continue;
                }

                // Aggregate based on type
                switch (_aggregatorType)
                {
                    case SAGEAggregatorType.Mean:
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
                            aggregated[b, i, f] = NumOps.Divide(sum,
                                NumOps.FromDouble(neighborCount));
                        }
                        break;

                    case SAGEAggregatorType.MaxPool:
                        for (int f = 0; f < _inputFeatures; f++)
                        {
                            T max = NumOps.FromDouble(double.NegativeInfinity);
                            for (int j = 0; j < numNodes; j++)
                            {
                                if (!NumOps.Equals(_adjacencyMatrix[b, i, j], NumOps.Zero))
                                {
                                    max = NumOps.Max(max, input[b, j, f]);
                                }
                            }
                            aggregated[b, i, f] = max;
                        }
                        break;

                    case SAGEAggregatorType.Sum:
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
                            aggregated[b, i, f] = sum;
                        }
                        break;
                }
            }
        }

        return aggregated;
    }

    /// <summary>
    /// Applies L2 normalization to node features.
    /// </summary>
    private void L2Normalize(Tensor<T> features, int batchSize, int numNodes)
    {
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                // Compute L2 norm
                T normSquared = NumOps.Zero;
                for (int f = 0; f < _outputFeatures; f++)
                {
                    T val = features[b, n, f];
                    normSquared = NumOps.Add(normSquared, NumOps.Multiply(val, val));
                }

                T norm = NumOps.Sqrt(normSquared);

                // Avoid division by zero
                if (NumOps.GreaterThan(norm, NumOps.FromDouble(1e-12)))
                {
                    // Normalize
                    for (int f = 0; f < _outputFeatures; f++)
                    {
                        features[b, n, f] = NumOps.Divide(features[b, n, f], norm);
                    }
                }
            }
        }
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

        // Aggregate neighbor features
        _lastAggregated = AggregateNeighbors(input, batchSize, numNodes);

        // Transform self features: input * selfWeights
        var selfTransformed = new Tensor<T>([batchSize, numNodes, _outputFeatures]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int outF = 0; outF < _outputFeatures; outF++)
                {
                    T sum = NumOps.Zero;
                    for (int inF = 0; inF < _inputFeatures; inF++)
                    {
                        sum = NumOps.Add(sum,
                            NumOps.Multiply(input[b, n, inF], _selfWeights[inF, outF]));
                    }
                    selfTransformed[b, n, outF] = sum;
                }
            }
        }

        // Transform neighbor features: aggregated * neighborWeights
        var neighborTransformed = new Tensor<T>([batchSize, numNodes, _outputFeatures]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int outF = 0; outF < _outputFeatures; outF++)
                {
                    T sum = NumOps.Zero;
                    for (int inF = 0; inF < _inputFeatures; inF++)
                    {
                        sum = NumOps.Add(sum,
                            NumOps.Multiply(_lastAggregated[b, n, inF],
                                _neighborWeights[inF, outF]));
                    }
                    neighborTransformed[b, n, outF] = sum;
                }
            }
        }

        // Combine: self + neighbor + bias
        var output = new Tensor<T>([batchSize, numNodes, _outputFeatures]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int f = 0; f < _outputFeatures; f++)
                {
                    output[b, n, f] = NumOps.Add(
                        NumOps.Add(selfTransformed[b, n, f], neighborTransformed[b, n, f]),
                        _bias[f]);
                }
            }
        }

        // Apply normalization if enabled
        if (_normalize)
        {
            L2Normalize(output, batchSize, numNodes);
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

        // Initialize gradients
        _selfWeightsGradient = new Matrix<T>(_inputFeatures, _outputFeatures);
        _neighborWeightsGradient = new Matrix<T>(_inputFeatures, _outputFeatures);
        _biasGradient = new Vector<T>(_outputFeatures);
        var inputGradient = new Tensor<T>(_lastInput.Shape);

        // Compute weight gradients
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int inF = 0; inF < _inputFeatures; inF++)
                {
                    for (int outF = 0; outF < _outputFeatures; outF++)
                    {
                        // Self weight gradient
                        T selfGrad = NumOps.Multiply(
                            _lastInput![b, n, inF],
                            activationGradient[b, n, outF]);
                        _selfWeightsGradient[inF, outF] =
                            NumOps.Add(_selfWeightsGradient[inF, outF], selfGrad);

                        // Neighbor weight gradient
                        T neighborGrad = NumOps.Multiply(
                            _lastAggregated![b, n, inF],
                            activationGradient[b, n, outF]);
                        _neighborWeightsGradient[inF, outF] =
                            NumOps.Add(_neighborWeightsGradient[inF, outF], neighborGrad);
                    }
                }

                // Bias gradient
                for (int f = 0; f < _outputFeatures; f++)
                {
                    _biasGradient[f] = NumOps.Add(_biasGradient[f],
                        activationGradient[b, n, f]);
                }
            }
        }

        // Compute input gradient (simplified - full version would backprop through aggregation)
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int inF = 0; inF < _inputFeatures; inF++)
                {
                    T grad = NumOps.Zero;
                    for (int outF = 0; outF < _outputFeatures; outF++)
                    {
                        grad = NumOps.Add(grad,
                            NumOps.Multiply(activationGradient[b, n, outF],
                                _selfWeights[inF, outF]));
                    }
                    inputGradient[b, n, inF] = grad;
                }
            }
        }

        return inputGradient;
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        if (_selfWeightsGradient == null || _neighborWeightsGradient == null || _biasGradient == null)
        {
            throw new InvalidOperationException("Backward must be called before UpdateParameters.");
        }

        // Update self weights
        _selfWeights = _selfWeights.Subtract(_selfWeightsGradient.Multiply(learningRate));

        // Update neighbor weights
        _neighborWeights = _neighborWeights.Subtract(_neighborWeightsGradient.Multiply(learningRate));

        // Update bias
        _bias = _bias.Subtract(_biasGradient.Multiply(learningRate));
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        int totalParams = 2 * _inputFeatures * _outputFeatures + _outputFeatures;
        var parameters = new Vector<T>(totalParams);
        int index = 0;

        // Self weights
        for (int i = 0; i < _selfWeights.Rows; i++)
        {
            for (int j = 0; j < _selfWeights.Columns; j++)
            {
                parameters[index++] = _selfWeights[i, j];
            }
        }

        // Neighbor weights
        for (int i = 0; i < _neighborWeights.Rows; i++)
        {
            for (int j = 0; j < _neighborWeights.Columns; j++)
            {
                parameters[index++] = _neighborWeights[i, j];
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
        int expectedParams = 2 * _inputFeatures * _outputFeatures + _outputFeatures;
        if (parameters.Length != expectedParams)
        {
            throw new ArgumentException(
                $"Expected {expectedParams} parameters, but got {parameters.Length}");
        }

        int index = 0;

        // Set self weights
        for (int i = 0; i < _selfWeights.Rows; i++)
        {
            for (int j = 0; j < _selfWeights.Columns; j++)
            {
                _selfWeights[i, j] = parameters[index++];
            }
        }

        // Set neighbor weights
        for (int i = 0; i < _neighborWeights.Rows; i++)
        {
            for (int j = 0; j < _neighborWeights.Columns; j++)
            {
                _neighborWeights[i, j] = parameters[index++];
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
        _lastAggregated = null;
        _selfWeightsGradient = null;
        _neighborWeightsGradient = null;
        _biasGradient = null;
    }
}
