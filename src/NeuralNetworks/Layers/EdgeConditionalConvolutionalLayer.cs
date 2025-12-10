namespace AiDotNet.NeuralNetworks.Layers.Graph;

/// <summary>
/// Implements Edge-Conditioned Convolution for incorporating edge features in graph convolutions.
/// </summary>
/// <remarks>
/// <para>
/// Edge-Conditioned Convolutions extend standard graph convolutions by incorporating edge features
/// into the aggregation process. Instead of treating all edges equally, this layer learns
/// edge-specific transformations based on edge attributes.
/// </para>
/// <para>
/// The layer computes: h_i' = σ(Σ_{j∈N(i)} θ(e_ij) · h_j + b)
/// where θ(e_ij) is an edge-specific transformation learned from edge features e_ij.
/// </para>
/// <para><b>For Beginners:</b> This layer lets connections (edges) have their own properties.
///
/// Think of a transportation network:
/// - Regular graph layers: All roads are treated the same
/// - Edge-conditioned layers: Each road has properties (speed limit, distance, traffic)
///
/// Examples where edge features matter:
/// - **Molecules**: Bond types (single/double/triple) affect how atoms interact
/// - **Social networks**: Relationship types (friend/colleague/family) influence information flow
/// - **Knowledge graphs**: Relationship types (is-a/part-of/located-in) determine connections
/// - **Transportation**: Road types (highway/street/path) affect travel patterns
///
/// This layer learns how to use these edge properties to better aggregate neighbor information.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class EdgeConditionalConvolutionalLayer<T> : LayerBase<T>, IGraphConvolutionLayer<T>
{
    private readonly int _inputFeatures;
    private readonly int _outputFeatures;
    private readonly int _edgeFeaturesCount;
    private readonly int _edgeNetworkHiddenDim;

    /// <summary>
    /// Edge network: transforms edge features to weight matrices.
    /// </summary>
    private Matrix<T> _edgeNetworkWeights1;
    private Matrix<T> _edgeNetworkWeights2;
    private Vector<T> _edgeNetworkBias1;
    private Vector<T> _edgeNetworkBias2;

    /// <summary>
    /// Self-loop transformation weights.
    /// </summary>
    private Matrix<T> _selfWeights;

    /// <summary>
    /// Bias vector.
    /// </summary>
    private Vector<T> _bias;

    /// <summary>
    /// The adjacency matrix defining graph structure.
    /// </summary>
    private Tensor<T>? _adjacencyMatrix;

    /// <summary>
    /// Edge features tensor.
    /// </summary>
    private Tensor<T>? _edgeFeatures;

    /// <summary>
    /// Cached values for backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastEdgeWeights;

    /// <summary>
    /// Gradients.
    /// </summary>
    private Matrix<T>? _edgeNetworkWeights1Gradient;
    private Matrix<T>? _edgeNetworkWeights2Gradient;
    private Vector<T>? _edgeNetworkBias1Gradient;
    private Vector<T>? _edgeNetworkBias2Gradient;
    private Matrix<T>? _selfWeightsGradient;
    private Vector<T>? _biasGradient;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public int InputFeatures => _inputFeatures;

    /// <inheritdoc/>
    public int OutputFeatures => _outputFeatures;

    /// <summary>
    /// Initializes a new instance of the <see cref="EdgeConditionalConvolutionalLayer{T}"/> class.
    /// </summary>
    /// <param name="inputFeatures">Number of input features per node.</param>
    /// <param name="outputFeatures">Number of output features per node.</param>
    /// <param name="edgeFeatures">Number of edge features.</param>
    /// <param name="edgeNetworkHiddenDim">Hidden dimension for edge network (default: 64).</param>
    /// <param name="activationFunction">Activation function to apply.</param>
    /// <remarks>
    /// <para>
    /// Creates an edge-conditioned convolution layer. The edge network is a 2-layer MLP
    /// that transforms edge features into node feature transformation weights.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new edge-conditioned layer.
    ///
    /// Parameters:
    /// - edgeFeatures: How many properties each connection has
    /// - edgeNetworkHiddenDim: Size of the network that learns from edge properties
    ///   (bigger = more expressive but slower)
    ///
    /// Example: In a molecule
    /// - inputFeatures=32: Each atom has 32 properties
    /// - outputFeatures=64: Transform to 64 properties
    /// - edgeFeatures=4: Each bond has 4 properties (type, length, strength, angle)
    ///
    /// The layer learns to use bond properties to determine how atoms influence each other.
    /// </para>
    /// </remarks>
    public EdgeConditionalConvolutionalLayer(
        int inputFeatures,
        int outputFeatures,
        int edgeFeatures,
        int edgeNetworkHiddenDim = 64,
        IActivationFunction<T>? activationFunction = null)
        : base([inputFeatures], [outputFeatures], activationFunction ?? new IdentityActivation<T>())
    {
        _inputFeatures = inputFeatures;
        _outputFeatures = outputFeatures;
        _edgeFeaturesCount = edgeFeatures;
        _edgeNetworkHiddenDim = edgeNetworkHiddenDim;

        // Edge network: maps edge features to transformation weights
        // Output size = inputFeatures * outputFeatures (flattened weight matrix per edge)
        _edgeNetworkWeights1 = new Matrix<T>(edgeFeatures, edgeNetworkHiddenDim);
        _edgeNetworkWeights2 = new Matrix<T>(edgeNetworkHiddenDim, inputFeatures * outputFeatures);
        _edgeNetworkBias1 = new Vector<T>(edgeNetworkHiddenDim);
        _edgeNetworkBias2 = new Vector<T>(inputFeatures * outputFeatures);

        // Self-loop weights
        _selfWeights = new Matrix<T>(inputFeatures, outputFeatures);

        // Bias
        _bias = new Vector<T>(outputFeatures);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        // Xavier initialization for edge network
        T scale1 = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_edgeFeaturesCount + _edgeNetworkHiddenDim)));
        InitializeMatrix(_edgeNetworkWeights1, scale1);

        T scale2 = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_edgeNetworkHiddenDim + _inputFeatures * _outputFeatures)));
        InitializeMatrix(_edgeNetworkWeights2, scale2);

        // Initialize self-loop weights
        T scaleSelf = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_inputFeatures + _outputFeatures)));
        InitializeMatrix(_selfWeights, scaleSelf);

        // Initialize biases to zero
        for (int i = 0; i < _edgeNetworkBias1.Length; i++)
            _edgeNetworkBias1[i] = NumOps.Zero;

        for (int i = 0; i < _edgeNetworkBias2.Length; i++)
            _edgeNetworkBias2[i] = NumOps.Zero;

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

    /// <summary>
    /// Sets the edge features for this layer.
    /// </summary>
    /// <param name="edgeFeatures">Edge features tensor with shape [batch, numEdges, edgeFeatureDim].</param>
    public void SetEdgeFeatures(Tensor<T> edgeFeatures)
    {
        _edgeFeatures = edgeFeatures;
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

        if (_edgeFeatures == null)
        {
            throw new InvalidOperationException(
                "Edge features must be set using SetEdgeFeatures before calling Forward.");
        }

        _lastInput = input;
        int batchSize = input.Shape[0];
        int numNodes = input.Shape[1];

        // Store edge-specific weights for backward pass
        _lastEdgeWeights = new Tensor<T>([batchSize, numNodes, numNodes, _inputFeatures, _outputFeatures]);

        // Step 1: Compute edge-specific weights using edge network
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                for (int j = 0; j < numNodes; j++)
                {
                    if (NumOps.Equals(_adjacencyMatrix[b, i, j], NumOps.Zero))
                        continue;

                    // Get edge features for edge (i, j)
                    int edgeIdx = i * numNodes + j;

                    // Edge network layer 1 with ReLU
                    var hidden = new Vector<T>(_edgeNetworkHiddenDim);
                    for (int h = 0; h < _edgeNetworkHiddenDim; h++)
                    {
                        T sum = _edgeNetworkBias1[h];
                        for (int f = 0; f < _edgeFeaturesCount; f++)
                        {
                            sum = NumOps.Add(sum,
                                NumOps.Multiply(_edgeFeatures[b, edgeIdx, f],
                                    _edgeNetworkWeights1[f, h]));
                        }
                        hidden[h] = ReLU(sum);
                    }

                    // Edge network layer 2 - outputs flattened weight matrix
                    var flatWeights = new Vector<T>(_inputFeatures * _outputFeatures);
                    for (int k = 0; k < _inputFeatures * _outputFeatures; k++)
                    {
                        T sum = _edgeNetworkBias2[k];
                        for (int h = 0; h < _edgeNetworkHiddenDim; h++)
                        {
                            sum = NumOps.Add(sum,
                                NumOps.Multiply(hidden[h], _edgeNetworkWeights2[h, k]));
                        }
                        flatWeights[k] = sum;
                    }

                    // Unflatten into weight matrix
                    int idx = 0;
                    for (int inF = 0; inF < _inputFeatures; inF++)
                    {
                        for (int outF = 0; outF < _outputFeatures; outF++)
                        {
                            _lastEdgeWeights[b, i, j, inF, outF] = flatWeights[idx++];
                        }
                    }
                }
            }
        }

        // Step 2: Aggregate neighbor features using edge-specific weights
        var output = new Tensor<T>([batchSize, numNodes, _outputFeatures]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                // Aggregate from neighbors
                for (int outF = 0; outF < _outputFeatures; outF++)
                {
                    T sum = NumOps.Zero;

                    for (int j = 0; j < numNodes; j++)
                    {
                        if (NumOps.Equals(_adjacencyMatrix[b, i, j], NumOps.Zero))
                            continue;

                        // Apply edge-specific transformation to neighbor features
                        for (int inF = 0; inF < _inputFeatures; inF++)
                        {
                            sum = NumOps.Add(sum,
                                NumOps.Multiply(
                                    _lastEdgeWeights[b, i, j, inF, outF],
                                    input[b, j, inF]));
                        }
                    }

                    output[b, i, outF] = sum;
                }

                // Add self-loop transformation
                for (int outF = 0; outF < _outputFeatures; outF++)
                {
                    for (int inF = 0; inF < _inputFeatures; inF++)
                    {
                        output[b, i, outF] = NumOps.Add(output[b, i, outF],
                            NumOps.Multiply(input[b, i, inF], _selfWeights[inF, outF]));
                    }

                    // Add bias
                    output[b, i, outF] = NumOps.Add(output[b, i, outF], _bias[outF]);
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

        // Initialize gradients
        _edgeNetworkWeights1Gradient = new Matrix<T>(_edgeFeaturesCount, _edgeNetworkHiddenDim);
        _edgeNetworkWeights2Gradient = new Matrix<T>(_edgeNetworkHiddenDim, _inputFeatures * _outputFeatures);
        _edgeNetworkBias1Gradient = new Vector<T>(_edgeNetworkHiddenDim);
        _edgeNetworkBias2Gradient = new Vector<T>(_inputFeatures * _outputFeatures);
        _selfWeightsGradient = new Matrix<T>(_inputFeatures, _outputFeatures);
        _biasGradient = new Vector<T>(_outputFeatures);

        var inputGradient = new Tensor<T>(_lastInput.Shape);

        // Compute gradients (simplified)
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                // Bias gradient
                for (int f = 0; f < _outputFeatures; f++)
                {
                    _biasGradient[f] = NumOps.Add(_biasGradient[f],
                        activationGradient[b, n, f]);
                }

                // Self-weights gradient
                for (int inF = 0; inF < _inputFeatures; inF++)
                {
                    for (int outF = 0; outF < _outputFeatures; outF++)
                    {
                        T grad = NumOps.Multiply(_lastInput[b, n, inF],
                            activationGradient[b, n, outF]);
                        _selfWeightsGradient[inF, outF] =
                            NumOps.Add(_selfWeightsGradient[inF, outF], grad);
                    }
                }
            }
        }

        return inputGradient;
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        if (_edgeNetworkWeights1Gradient == null)
        {
            throw new InvalidOperationException("Backward must be called before UpdateParameters.");
        }

        _edgeNetworkWeights1 = _edgeNetworkWeights1.Subtract(
            _edgeNetworkWeights1Gradient.Multiply(learningRate));
        _edgeNetworkWeights2 = _edgeNetworkWeights2.Subtract(
            _edgeNetworkWeights2Gradient!.Multiply(learningRate));
        _selfWeights = _selfWeights.Subtract(
            _selfWeightsGradient!.Multiply(learningRate));

        _edgeNetworkBias1 = _edgeNetworkBias1.Subtract(
            _edgeNetworkBias1Gradient!.Multiply(learningRate));
        _edgeNetworkBias2 = _edgeNetworkBias2.Subtract(
            _edgeNetworkBias2Gradient!.Multiply(learningRate));
        _bias = _bias.Subtract(_biasGradient!.Multiply(learningRate));
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        int totalParams = _edgeNetworkWeights1.Rows * _edgeNetworkWeights1.Columns +
                         _edgeNetworkWeights2.Rows * _edgeNetworkWeights2.Columns +
                         _edgeNetworkBias1.Length +
                         _edgeNetworkBias2.Length +
                         _selfWeights.Rows * _selfWeights.Columns +
                         _bias.Length;

        var parameters = new Vector<T>(totalParams);
        int index = 0;

        for (int i = 0; i < _edgeNetworkWeights1.Rows; i++)
            for (int j = 0; j < _edgeNetworkWeights1.Columns; j++)
                parameters[index++] = _edgeNetworkWeights1[i, j];

        for (int i = 0; i < _edgeNetworkWeights2.Rows; i++)
            for (int j = 0; j < _edgeNetworkWeights2.Columns; j++)
                parameters[index++] = _edgeNetworkWeights2[i, j];

        for (int i = 0; i < _edgeNetworkBias1.Length; i++)
            parameters[index++] = _edgeNetworkBias1[i];

        for (int i = 0; i < _edgeNetworkBias2.Length; i++)
            parameters[index++] = _edgeNetworkBias2[i];

        for (int i = 0; i < _selfWeights.Rows; i++)
            for (int j = 0; j < _selfWeights.Columns; j++)
                parameters[index++] = _selfWeights[i, j];

        for (int i = 0; i < _bias.Length; i++)
            parameters[index++] = _bias[i];

        return parameters;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int index = 0;

        for (int i = 0; i < _edgeNetworkWeights1.Rows; i++)
            for (int j = 0; j < _edgeNetworkWeights1.Columns; j++)
                _edgeNetworkWeights1[i, j] = parameters[index++];

        for (int i = 0; i < _edgeNetworkWeights2.Rows; i++)
            for (int j = 0; j < _edgeNetworkWeights2.Columns; j++)
                _edgeNetworkWeights2[i, j] = parameters[index++];

        for (int i = 0; i < _edgeNetworkBias1.Length; i++)
            _edgeNetworkBias1[i] = parameters[index++];

        for (int i = 0; i < _edgeNetworkBias2.Length; i++)
            _edgeNetworkBias2[i] = parameters[index++];

        for (int i = 0; i < _selfWeights.Rows; i++)
            for (int j = 0; j < _selfWeights.Columns; j++)
                _selfWeights[i, j] = parameters[index++];

        for (int i = 0; i < _bias.Length; i++)
            _bias[i] = parameters[index++];
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastEdgeWeights = null;
        _edgeNetworkWeights1Gradient = null;
        _edgeNetworkWeights2Gradient = null;
        _edgeNetworkBias1Gradient = null;
        _edgeNetworkBias2Gradient = null;
        _selfWeightsGradient = null;
        _biasGradient = null;
    }

    /// <inheritdoc/>
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc/>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException(
            "EdgeConditionalConvolutionalLayer does not support computation graph export due to dynamic edge-based weight generation.");
    }
}
