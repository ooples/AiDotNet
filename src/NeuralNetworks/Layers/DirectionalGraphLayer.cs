using AiDotNet.ActivationFunctions;
using AiDotNet.Autodiff;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

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
    private readonly Random _random;

    /// <summary>
    /// Weights for incoming edge aggregation.
    /// Shape: [inputFeatures, outputFeatures]
    /// </summary>
    private Tensor<T> _incomingWeights;

    /// <summary>
    /// Weights for outgoing edge aggregation.
    /// Shape: [inputFeatures, outputFeatures]
    /// </summary>
    private Tensor<T> _outgoingWeights;

    /// <summary>
    /// Self-loop weights.
    /// Shape: [inputFeatures, outputFeatures]
    /// </summary>
    private Tensor<T> _selfWeights;

    /// <summary>
    /// Gating mechanism weights (if enabled).
    /// Shape: [3 * outputFeatures, 3]
    /// </summary>
    private Tensor<T>? _gateWeights;

    /// <summary>
    /// Gating bias (if enabled).
    /// Shape: [3]
    /// </summary>
    private Tensor<T>? _gateBias;

    /// <summary>
    /// Biases for incoming, outgoing, and self transformations.
    /// Shape: [outputFeatures] each
    /// </summary>
    private Tensor<T> _incomingBias;
    private Tensor<T> _outgoingBias;
    private Tensor<T> _selfBias;

    /// <summary>
    /// Combination weights for merging in/out/self features.
    /// Shape: [3 * outputFeatures, outputFeatures]
    /// </summary>
    private Tensor<T> _combinationWeights;

    /// <summary>
    /// Combination bias.
    /// Shape: [outputFeatures]
    /// </summary>
    private Tensor<T> _combinationBias;

    /// <summary>
    /// The adjacency matrix defining graph structure (interpreted as directed).
    /// </summary>
    private Tensor<T>? _adjacencyMatrix;

    /// <summary>
    /// The adjacency matrix reshaped to 3D for batched operations.
    /// </summary>
    private Tensor<T>? _adjForBatch;

    /// <summary>
    /// Cached values for backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastIncoming;
    private Tensor<T>? _lastOutgoing;
    private Tensor<T>? _lastSelf;
    private Tensor<T>? _lastCombined;
    private Tensor<T>? _lastGates;

    /// <summary>
    /// Gradients.
    /// </summary>
    private Tensor<T>? _incomingWeightsGradient;
    private Tensor<T>? _outgoingWeightsGradient;
    private Tensor<T>? _selfWeightsGradient;
    private Tensor<T>? _combinationWeightsGradient;
    private Tensor<T>? _incomingBiasGradient;
    private Tensor<T>? _outgoingBiasGradient;
    private Tensor<T>? _selfBiasGradient;
    private Tensor<T>? _combinationBiasGradient;
    private Tensor<T>? _gateWeightsGradient;
    private Tensor<T>? _gateBiasGradient;

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
        _random = RandomHelper.CreateSecureRandom();

        // Initialize transformation weights for each direction - all Tensor<T>
        _incomingWeights = new Tensor<T>([_inputFeatures, _outputFeatures]);
        _outgoingWeights = new Tensor<T>([_inputFeatures, _outputFeatures]);
        _selfWeights = new Tensor<T>([_inputFeatures, _outputFeatures]);

        _incomingBias = new Tensor<T>([_outputFeatures]);
        _outgoingBias = new Tensor<T>([_outputFeatures]);
        _selfBias = new Tensor<T>([_outputFeatures]);

        // Combination weights: combines in/out/self features
        int combinedDim = 3 * _outputFeatures;
        _combinationWeights = new Tensor<T>([combinedDim, _outputFeatures]);
        _combinationBias = new Tensor<T>([_outputFeatures]);

        // Gating mechanism (optional)
        if (_useGating)
        {
            _gateWeights = new Tensor<T>([combinedDim, 3]); // 3 gates for in/out/self
            _gateBias = new Tensor<T>([3]);
        }

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_inputFeatures + _outputFeatures)));

        // Initialize directional weights
        InitializeTensor(_incomingWeights, scale);
        InitializeTensor(_outgoingWeights, scale);
        InitializeTensor(_selfWeights, scale);

        // Initialize combination weights
        T scaleComb = NumOps.Sqrt(NumOps.FromDouble(2.0 / (3 * _outputFeatures + _outputFeatures)));
        InitializeTensor(_combinationWeights, scaleComb);

        // Initialize gating weights if used
        if (_gateWeights != null)
        {
            T scaleGate = NumOps.Sqrt(NumOps.FromDouble(2.0 / (3 * _outputFeatures + 3)));
            InitializeTensor(_gateWeights, scaleGate);
        }

        // Initialize biases to zero
        _incomingBias.Fill(NumOps.Zero);
        _outgoingBias.Fill(NumOps.Zero);
        _selfBias.Fill(NumOps.Zero);
        _combinationBias.Fill(NumOps.Zero);

        if (_gateBias != null)
        {
            _gateBias.Fill(NumOps.Zero);
        }
    }

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
    /// Performs batched matrix multiplication between a 3D tensor and a 2D weight matrix.
    /// Input: [batch, rows, cols] @ weights: [cols, output_cols] -> [batch, rows, output_cols]
    /// </summary>
    private Tensor<T> BatchedMatMul3Dx2D(Tensor<T> input3D, Tensor<T> weights2D, int batch, int rows, int cols, int outputCols)
    {
        // Flatten batch dimension: [batch, rows, cols] -> [batch * rows, cols]
        var flattened = input3D.Reshape([batch * rows, cols]);
        // Standard 2D matmul: [batch * rows, cols] @ [cols, output_cols] -> [batch * rows, output_cols]
        var result = Engine.TensorMatMul(flattened, weights2D);
        // Unflatten: [batch * rows, output_cols] -> [batch, rows, output_cols]
        return result.Reshape([batch, rows, outputCols]);
    }

    /// <summary>
    /// Broadcasts a bias tensor across batch and node dimensions.
    /// </summary>
    private Tensor<T> BroadcastBias(Tensor<T> bias, int batchSize, int numNodes)
    {
        int outputFeatures = bias.Length;
        var biasReshaped = bias.Reshape([1, 1, outputFeatures]);
        return Engine.TensorTile(biasReshaped, [batchSize, numNodes, 1]);
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

        // Handle any-rank tensor: collapse leading dims for rank > 3
        Tensor<T> processInput;
        int batchSize;

        if (rank == 2)
        {
            batchSize = 1;
            processInput = input.Reshape([1, input.Shape[0], input.Shape[1]]);
        }
        else if (rank == 3)
        {
            batchSize = input.Shape[0];
            processInput = input;
        }
        else
        {
            int flatBatch = 1;
            for (int d = 0; d < rank - 2; d++)
                flatBatch *= input.Shape[d];
            batchSize = flatBatch;
            processInput = input.Reshape([flatBatch, input.Shape[rank - 2], input.Shape[rank - 1]]);
        }

        _lastInput = processInput;
        int numNodes = processInput.Shape[1];
        int inputFeatures = processInput.Shape[2];

        // Ensure adjacency matrix has matching rank for batch operation
        // If adjacency is 2D [nodes, nodes] and input is 3D [batch, nodes, features], reshape to 3D
        Tensor<T> adjForBatch;
        if (_adjacencyMatrix.Shape.Length == 2 && processInput.Shape.Length == 3)
        {
            if (batchSize == 1)
            {
                adjForBatch = _adjacencyMatrix.Reshape([1, numNodes, numNodes]);
            }
            else
            {
                // Broadcast: repeat adjacency matrix for each batch item
                adjForBatch = new Tensor<T>([batchSize, numNodes, numNodes]);
                for (int b = 0; b < batchSize; b++)
                {
                    for (int i = 0; i < numNodes; i++)
                    {
                        for (int j = 0; j < numNodes; j++)
                        {
                            adjForBatch[new int[] { b, i, j }] = _adjacencyMatrix[new int[] { i, j }];
                        }
                    }
                }
            }
        }
        else
        {
            adjForBatch = _adjacencyMatrix;
        }

        // Store for backward pass
        _adjForBatch = adjForBatch;

        // Step 1: Aggregate incoming edges (nodes that point TO this node)
        // A[i,j] = 1 means edge from j to i (j→i)
        // For incoming: multiply A @ X @ W_in
        var xwIn = BatchedMatMul3Dx2D(processInput, _incomingWeights, batchSize, numNodes, inputFeatures, _outputFeatures);
        _lastIncoming = Engine.BatchMatMul(adjForBatch, xwIn);
        var biasIn = BroadcastBias(_incomingBias, batchSize, numNodes);
        _lastIncoming = Engine.TensorAdd(_lastIncoming, biasIn);

        // Step 2: Aggregate outgoing edges (nodes that this node points TO)
        // For outgoing: multiply A^T @ X @ W_out
        var adjTransposed = Engine.TensorPermute(adjForBatch, [0, 2, 1]); // Batched transpose
        var xwOut = BatchedMatMul3Dx2D(processInput, _outgoingWeights, batchSize, numNodes, inputFeatures, _outputFeatures);
        _lastOutgoing = Engine.BatchMatMul(adjTransposed, xwOut);
        var biasOut = BroadcastBias(_outgoingBias, batchSize, numNodes);
        _lastOutgoing = Engine.TensorAdd(_lastOutgoing, biasOut);

        // Step 3: Transform self features: X @ W_self + b_self
        _lastSelf = BatchedMatMul3Dx2D(processInput, _selfWeights, batchSize, numNodes, inputFeatures, _outputFeatures);
        var biasSelf = BroadcastBias(_selfBias, batchSize, numNodes);
        _lastSelf = Engine.TensorAdd(_lastSelf, biasSelf);

        // Step 4: Concatenate incoming, outgoing, and self features
        _lastCombined = ConcatenateFeatures(_lastIncoming, _lastOutgoing, _lastSelf, batchSize, numNodes);

        Tensor<T> gatedCombined = _lastCombined;

        // Step 5: Apply gating if enabled
        if (_useGating && _gateWeights != null && _gateBias != null)
        {
            // Compute gates: combined @ W_gate + b_gate
            var gateLogits = BatchedMatMul3Dx2D(_lastCombined, _gateWeights, batchSize, numNodes, 3 * _outputFeatures, 3);
            var gateBiasBroadcast = BroadcastBias(_gateBias, batchSize, numNodes);
            gateLogits = Engine.TensorAdd(gateLogits, gateBiasBroadcast);

            // Apply sigmoid to get gates
            _lastGates = ApplySigmoid(gateLogits);

            // Apply gates to each of the three feature groups
            gatedCombined = ApplyGatesToFeatures(_lastCombined, _lastGates, batchSize, numNodes, _outputFeatures);
        }

        // Step 6: Final combination: gatedCombined @ W_comb + b_comb
        var output = BatchedMatMul3Dx2D(gatedCombined, _combinationWeights, batchSize, numNodes, 3 * _outputFeatures, _outputFeatures);
        var combinationBiasBroadcast = BroadcastBias(_combinationBias, batchSize, numNodes);
        output = Engine.TensorAdd(output, combinationBiasBroadcast);

        var result = ApplyActivation(output);

        // Only store for backward pass during training - skip during inference
        if (IsTrainingMode)
        {
            _lastOutput = result;
        }

        // Reshape output to match original input shape (except for feature dimension)
        if (_originalInputShape != null && _originalInputShape.Length == 2)
        {
            // Original was 2D [N, F] -> return [N, outputFeatures]
            return result.Reshape([numNodes, _outputFeatures]);
        }
        else if (_originalInputShape != null && _originalInputShape.Length == 1)
        {
            // Original was 1D [F] -> return [outputFeatures]
            return result.Reshape([_outputFeatures]);
        }

        return result;
    }

    /// <summary>
    /// Concatenates incoming, outgoing, and self features along the feature dimension.
    /// </summary>
    private Tensor<T> ConcatenateFeatures(Tensor<T> incoming, Tensor<T> outgoing, Tensor<T> self, int batchSize, int numNodes)
    {
        int outputFeatures = incoming.Shape[2];
        var combined = new Tensor<T>([batchSize, numNodes, 3 * outputFeatures]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int f = 0; f < outputFeatures; f++)
                {
                    combined[b, n, f] = incoming[b, n, f];
                    combined[b, n, outputFeatures + f] = outgoing[b, n, f];
                    combined[b, n, 2 * outputFeatures + f] = self[b, n, f];
                }
            }
        }

        return combined;
    }

    /// <summary>
    /// Applies sigmoid activation element-wise to a tensor.
    /// </summary>
    private Tensor<T> ApplySigmoid(Tensor<T> input)
    {
        // === Vectorized sigmoid using IEngine (Phase B: US-GPU-015) ===
        return Engine.Sigmoid(input);
    }

    /// <summary>
    /// Applies gates to the concatenated features.
    /// </summary>
    private Tensor<T> ApplyGatesToFeatures(Tensor<T> combined, Tensor<T> gates, int batchSize, int numNodes, int outputFeatures)
    {
        var gated = new Tensor<T>(combined.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                T gate0 = gates[b, n, 0];
                T gate1 = gates[b, n, 1];
                T gate2 = gates[b, n, 2];

                for (int f = 0; f < outputFeatures; f++)
                {
                    gated[b, n, f] = NumOps.Multiply(combined[b, n, f], gate0);
                    gated[b, n, outputFeatures + f] = NumOps.Multiply(combined[b, n, outputFeatures + f], gate1);
                    gated[b, n, 2 * outputFeatures + f] = NumOps.Multiply(combined[b, n, 2 * outputFeatures + f], gate2);
                }
            }
        }

        return gated;
    }

    /// <summary>
    /// Computes the backward pass for this Directional Graph layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to this layer's output.</param>
    /// <returns>The gradient of the loss with respect to this layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This backward pass computes gradients for all parameters and propagates gradients to the input.
    /// It handles the complex flow through directional aggregation, gating, and combination stages.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _adjacencyMatrix == null || _adjForBatch == null)
        {
            throw new InvalidOperationException("Forward pass must be called before Backward.");
        }

        if (_lastIncoming == null || _lastOutgoing == null || _lastSelf == null || _lastCombined == null)
        {
            throw new InvalidOperationException("Forward pass data incomplete.");
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
        int inputFeatures = _lastInput.Shape[2];

        // Gradient w.r.t. combination bias: sum over batch and nodes
        _combinationBiasGradient = Engine.ReduceSum(activationGradient, [0, 1], keepDims: false);

        // Gradient w.r.t. combination weights and gatedCombined
        // output = gatedCombined @ W_comb + b_comb
        // dL/dW_comb = gatedCombined^T @ dL/doutput
        // dL/dgatedCombined = dL/doutput @ W_comb^T
        _combinationWeightsGradient = new Tensor<T>([3 * _outputFeatures, _outputFeatures]);
        _combinationWeightsGradient.Fill(NumOps.Zero);

        var gatedCombinedGradient = new Tensor<T>([batchSize, numNodes, 3 * _outputFeatures]);

        for (int b = 0; b < batchSize; b++)
        {
            var gatedCombinedBatch = _lastCombined;
            if (_useGating && _lastGates != null)
            {
                gatedCombinedBatch = ApplyGatesToFeatures(_lastCombined, _lastGates, batchSize, numNodes, _outputFeatures);
            }

            var gatedSlice = Engine.TensorSlice(gatedCombinedBatch, [b, 0, 0], [1, numNodes, 3 * _outputFeatures]).Reshape([numNodes, 3 * _outputFeatures]);
            var gradSlice = Engine.TensorSlice(activationGradient, [b, 0, 0], [1, numNodes, _outputFeatures]).Reshape([numNodes, _outputFeatures]);

            // Accumulate weight gradient
            var gatedT = Engine.TensorTranspose(gatedSlice);
            var batchWeightGrad = Engine.TensorMatMul(gatedT, gradSlice);
            _combinationWeightsGradient = Engine.TensorAdd(_combinationWeightsGradient, batchWeightGrad);

            // Compute gradient w.r.t. gatedCombined
            var weightsT = Engine.TensorTranspose(_combinationWeights);
            var gatedGradBatch = Engine.TensorMatMul(gradSlice, weightsT);
            gatedCombinedGradient = Engine.TensorSetSlice(gatedCombinedGradient, gatedGradBatch.Reshape([1, numNodes, 3 * _outputFeatures]), [b, 0, 0]);
        }

        // Gradient through gating (if enabled)
        Tensor<T> combinedGradient;
        if (_useGating && _lastGates != null && _gateWeights != null && _gateBias != null)
        {
            combinedGradient = BackwardThroughGating(gatedCombinedGradient, batchSize, numNodes);
        }
        else
        {
            combinedGradient = gatedCombinedGradient;
        }

        // Split gradient back into incoming, outgoing, self
        var incomingGrad = new Tensor<T>([batchSize, numNodes, _outputFeatures]);
        var outgoingGrad = new Tensor<T>([batchSize, numNodes, _outputFeatures]);
        var selfGrad = new Tensor<T>([batchSize, numNodes, _outputFeatures]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int f = 0; f < _outputFeatures; f++)
                {
                    incomingGrad[b, n, f] = combinedGradient[b, n, f];
                    outgoingGrad[b, n, f] = combinedGradient[b, n, _outputFeatures + f];
                    selfGrad[b, n, f] = combinedGradient[b, n, 2 * _outputFeatures + f];
                }
            }
        }

        // Gradient w.r.t. biases
        _incomingBiasGradient = Engine.ReduceSum(incomingGrad, [0, 1], keepDims: false);
        _outgoingBiasGradient = Engine.ReduceSum(outgoingGrad, [0, 1], keepDims: false);
        _selfBiasGradient = Engine.ReduceSum(selfGrad, [0, 1], keepDims: false);

        // Gradient w.r.t. weights and input for each path
        _incomingWeightsGradient = new Tensor<T>([inputFeatures, _outputFeatures]);
        _outgoingWeightsGradient = new Tensor<T>([inputFeatures, _outputFeatures]);
        _selfWeightsGradient = new Tensor<T>([inputFeatures, _outputFeatures]);
        _incomingWeightsGradient.Fill(NumOps.Zero);
        _outgoingWeightsGradient.Fill(NumOps.Zero);
        _selfWeightsGradient.Fill(NumOps.Zero);

        var inputGradient = new Tensor<T>(_lastInput.Shape);
        inputGradient.Fill(NumOps.Zero);

        var adjTransposed = Engine.TensorPermute(_adjForBatch, [0, 2, 1]); // Batched transpose

        // Incoming path: A @ (X @ W_in)
        // dL/dW_in = X^T @ A^T @ dL/dincoming
        // dL/dX += A^T @ dL/dincoming @ W_in^T
        for (int b = 0; b < batchSize; b++)
        {
            var inputBatch = Engine.TensorSlice(_lastInput, [b, 0, 0], [1, numNodes, inputFeatures]).Reshape([numNodes, inputFeatures]);
            var adjTBatch = Engine.TensorSlice(adjTransposed, [b, 0, 0], [1, numNodes, numNodes]).Reshape([numNodes, numNodes]);
            var inGradBatch = Engine.TensorSlice(incomingGrad, [b, 0, 0], [1, numNodes, _outputFeatures]).Reshape([numNodes, _outputFeatures]);

            var inputT = Engine.TensorTranspose(inputBatch);
            var adjTGrad = Engine.TensorMatMul(adjTBatch, inGradBatch);
            var batchWeightGrad = Engine.TensorMatMul(inputT, adjTGrad);
            _incomingWeightsGradient = Engine.TensorAdd(_incomingWeightsGradient, batchWeightGrad);

            var weightsT = Engine.TensorTranspose(_incomingWeights);
            var inputGradBatch = Engine.TensorMatMul(adjTGrad, weightsT);
            inputGradient = Engine.TensorSetSlice(inputGradient, inputGradBatch.Reshape([1, numNodes, inputFeatures]), [b, 0, 0]);
        }

        // Outgoing path: A^T @ (X @ W_out)
        // dL/dW_out = X^T @ A @ dL/doutgoing
        // dL/dX += A @ dL/doutgoing @ W_out^T
        for (int b = 0; b < batchSize; b++)
        {
            var inputBatch = Engine.TensorSlice(_lastInput, [b, 0, 0], [1, numNodes, inputFeatures]).Reshape([numNodes, inputFeatures]);
            var adjBatch = Engine.TensorSlice(_adjForBatch, [b, 0, 0], [1, numNodes, numNodes]).Reshape([numNodes, numNodes]);
            var outGradBatch = Engine.TensorSlice(outgoingGrad, [b, 0, 0], [1, numNodes, _outputFeatures]).Reshape([numNodes, _outputFeatures]);

            var inputT = Engine.TensorTranspose(inputBatch);
            var adjGrad = Engine.TensorMatMul(adjBatch, outGradBatch);
            var batchWeightGrad = Engine.TensorMatMul(inputT, adjGrad);
            _outgoingWeightsGradient = Engine.TensorAdd(_outgoingWeightsGradient, batchWeightGrad);

            var weightsT = Engine.TensorTranspose(_outgoingWeights);
            var inputGradBatch = Engine.TensorMatMul(adjGrad, weightsT);
            var existingGrad = Engine.TensorSlice(inputGradient, [b, 0, 0], [1, numNodes, inputFeatures]).Reshape([numNodes, inputFeatures]);
            var summedGrad = Engine.TensorAdd(existingGrad, inputGradBatch);
            inputGradient = Engine.TensorSetSlice(inputGradient, summedGrad.Reshape([1, numNodes, inputFeatures]), [b, 0, 0]);
        }

        // Self path: X @ W_self
        // dL/dW_self = X^T @ dL/dself
        // dL/dX += dL/dself @ W_self^T
        for (int b = 0; b < batchSize; b++)
        {
            var inputBatch = Engine.TensorSlice(_lastInput, [b, 0, 0], [1, numNodes, inputFeatures]).Reshape([numNodes, inputFeatures]);
            var selfGradBatch = Engine.TensorSlice(selfGrad, [b, 0, 0], [1, numNodes, _outputFeatures]).Reshape([numNodes, _outputFeatures]);

            var inputT = Engine.TensorTranspose(inputBatch);
            var batchWeightGrad = Engine.TensorMatMul(inputT, selfGradBatch);
            _selfWeightsGradient = Engine.TensorAdd(_selfWeightsGradient, batchWeightGrad);

            var weightsT = Engine.TensorTranspose(_selfWeights);
            var inputGradBatch = Engine.TensorMatMul(selfGradBatch, weightsT);
            var existingGrad = Engine.TensorSlice(inputGradient, [b, 0, 0], [1, numNodes, inputFeatures]).Reshape([numNodes, inputFeatures]);
            var summedGrad = Engine.TensorAdd(existingGrad, inputGradBatch);
            inputGradient = Engine.TensorSetSlice(inputGradient, summedGrad.Reshape([1, numNodes, inputFeatures]), [b, 0, 0]);
        }

        // Reshape to match original input shape
        if (_originalInputShape != null && _originalInputShape.Length != inputGradient.Shape.Length)
        {
            return inputGradient.Reshape(_originalInputShape);
        }

        return inputGradient;
    }

    /// <summary>
    /// Computes gradients through the gating mechanism.
    /// </summary>
    private Tensor<T> BackwardThroughGating(Tensor<T> gatedCombinedGradient, int batchSize, int numNodes)
    {
        if (_lastGates == null || _gateWeights == null || _gateBias == null || _lastCombined == null)
        {
            throw new InvalidOperationException("Gating data not available.");
        }

        _gateWeightsGradient = new Tensor<T>([3 * _outputFeatures, 3]);
        _gateWeightsGradient.Fill(NumOps.Zero);
        _gateBiasGradient = new Tensor<T>([3]);
        _gateBiasGradient.Fill(NumOps.Zero);

        var combinedGradient = new Tensor<T>(_lastCombined.Shape);
        var gatesGradient = new Tensor<T>([batchSize, numNodes, 3]);

        // Gradient through element-wise multiplication: gated = combined * gates
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                T gate0 = _lastGates[b, n, 0];
                T gate1 = _lastGates[b, n, 1];
                T gate2 = _lastGates[b, n, 2];

                T gateGrad0 = NumOps.Zero;
                T gateGrad1 = NumOps.Zero;
                T gateGrad2 = NumOps.Zero;

                for (int f = 0; f < _outputFeatures; f++)
                {
                    // dL/dcombined = dL/dgated * gate
                    combinedGradient[b, n, f] = NumOps.Multiply(gatedCombinedGradient[b, n, f], gate0);
                    combinedGradient[b, n, _outputFeatures + f] = NumOps.Multiply(gatedCombinedGradient[b, n, _outputFeatures + f], gate1);
                    combinedGradient[b, n, 2 * _outputFeatures + f] = NumOps.Multiply(gatedCombinedGradient[b, n, 2 * _outputFeatures + f], gate2);

                    // dL/dgate = sum_f (dL/dgated * combined)
                    gateGrad0 = NumOps.Add(gateGrad0, NumOps.Multiply(gatedCombinedGradient[b, n, f], _lastCombined[b, n, f]));
                    gateGrad1 = NumOps.Add(gateGrad1, NumOps.Multiply(gatedCombinedGradient[b, n, _outputFeatures + f], _lastCombined[b, n, _outputFeatures + f]));
                    gateGrad2 = NumOps.Add(gateGrad2, NumOps.Multiply(gatedCombinedGradient[b, n, 2 * _outputFeatures + f], _lastCombined[b, n, 2 * _outputFeatures + f]));
                }

                // Gradient through sigmoid: dL/dlogit = dL/dgate * gate * (1 - gate)
                T sigmoidGrad0 = NumOps.Multiply(gateGrad0, NumOps.Multiply(gate0, NumOps.Subtract(NumOps.FromDouble(1.0), gate0)));
                T sigmoidGrad1 = NumOps.Multiply(gateGrad1, NumOps.Multiply(gate1, NumOps.Subtract(NumOps.FromDouble(1.0), gate1)));
                T sigmoidGrad2 = NumOps.Multiply(gateGrad2, NumOps.Multiply(gate2, NumOps.Subtract(NumOps.FromDouble(1.0), gate2)));

                gatesGradient[b, n, 0] = sigmoidGrad0;
                gatesGradient[b, n, 1] = sigmoidGrad1;
                gatesGradient[b, n, 2] = sigmoidGrad2;

                // Accumulate bias gradient
                _gateBiasGradient[0] = NumOps.Add(_gateBiasGradient.GetFlat(0), sigmoidGrad0);
                _gateBiasGradient[1] = NumOps.Add(_gateBiasGradient.GetFlat(1), sigmoidGrad1);
                _gateBiasGradient[2] = NumOps.Add(_gateBiasGradient.GetFlat(2), sigmoidGrad2);
            }
        }

        // Gradient w.r.t. gate weights
        for (int b = 0; b < batchSize; b++)
        {
            var combinedBatch = Engine.TensorSlice(_lastCombined, [b, 0, 0], [1, numNodes, 3 * _outputFeatures]).Reshape([numNodes, 3 * _outputFeatures]);
            var gatesGradBatch = Engine.TensorSlice(gatesGradient, [b, 0, 0], [1, numNodes, 3]).Reshape([numNodes, 3]);

            var combinedT = Engine.TensorTranspose(combinedBatch);
            var batchWeightGrad = Engine.TensorMatMul(combinedT, gatesGradBatch);
            _gateWeightsGradient = Engine.TensorAdd(_gateWeightsGradient, batchWeightGrad);
        }

        // Gradient w.r.t. combined from gate computation
        var gateWeightsT = Engine.TensorTranspose(_gateWeights);
        for (int b = 0; b < batchSize; b++)
        {
            var gatesGradBatch = Engine.TensorSlice(gatesGradient, [b, 0, 0], [1, numNodes, 3]).Reshape([numNodes, 3]);
            var combinedGradFromGates = Engine.TensorMatMul(gatesGradBatch, gateWeightsT);

            var existingCombinedGrad = Engine.TensorSlice(combinedGradient, [b, 0, 0], [1, numNodes, 3 * _outputFeatures]).Reshape([numNodes, 3 * _outputFeatures]);
            var summed = Engine.TensorAdd(existingCombinedGrad, combinedGradFromGates);
            combinedGradient = Engine.TensorSetSlice(combinedGradient, summed.Reshape([1, numNodes, 3 * _outputFeatures]), [b, 0, 0]);
        }

        return combinedGradient;
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        if (_incomingWeightsGradient == null || _outgoingWeightsGradient == null ||
            _selfWeightsGradient == null || _combinationWeightsGradient == null)
        {
            throw new InvalidOperationException("Backward must be called before UpdateParameters.");
        }

        // Update weights using Engine operations
        var scaledInGrad = Engine.TensorMultiplyScalar(_incomingWeightsGradient, learningRate);
        _incomingWeights = Engine.TensorSubtract(_incomingWeights, scaledInGrad);

        var scaledOutGrad = Engine.TensorMultiplyScalar(_outgoingWeightsGradient, learningRate);
        _outgoingWeights = Engine.TensorSubtract(_outgoingWeights, scaledOutGrad);

        var scaledSelfGrad = Engine.TensorMultiplyScalar(_selfWeightsGradient, learningRate);
        _selfWeights = Engine.TensorSubtract(_selfWeights, scaledSelfGrad);

        var scaledCombGrad = Engine.TensorMultiplyScalar(_combinationWeightsGradient, learningRate);
        _combinationWeights = Engine.TensorSubtract(_combinationWeights, scaledCombGrad);

        // Update biases
        if (_incomingBiasGradient != null)
        {
            var scaledInBiasGrad = Engine.TensorMultiplyScalar(_incomingBiasGradient, learningRate);
            _incomingBias = Engine.TensorSubtract(_incomingBias, scaledInBiasGrad);
        }

        if (_outgoingBiasGradient != null)
        {
            var scaledOutBiasGrad = Engine.TensorMultiplyScalar(_outgoingBiasGradient, learningRate);
            _outgoingBias = Engine.TensorSubtract(_outgoingBias, scaledOutBiasGrad);
        }

        if (_selfBiasGradient != null)
        {
            var scaledSelfBiasGrad = Engine.TensorMultiplyScalar(_selfBiasGradient, learningRate);
            _selfBias = Engine.TensorSubtract(_selfBias, scaledSelfBiasGrad);
        }

        if (_combinationBiasGradient != null)
        {
            var scaledCombBiasGrad = Engine.TensorMultiplyScalar(_combinationBiasGradient, learningRate);
            _combinationBias = Engine.TensorSubtract(_combinationBias, scaledCombBiasGrad);
        }

        // Update gating parameters if enabled
        if (_useGating && _gateWeightsGradient != null && _gateBiasGradient != null && _gateWeights != null && _gateBias != null)
        {
            var scaledGateWeightsGrad = Engine.TensorMultiplyScalar(_gateWeightsGradient, learningRate);
            _gateWeights = Engine.TensorSubtract(_gateWeights, scaledGateWeightsGrad);

            var scaledGateBiasGrad = Engine.TensorMultiplyScalar(_gateBiasGradient, learningRate);
            _gateBias = Engine.TensorSubtract(_gateBias, scaledGateBiasGrad);
        }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var paramsList = new List<T>();

        // Add all weight matrices
        paramsList.AddRange(_incomingWeights.ToArray());
        paramsList.AddRange(_outgoingWeights.ToArray());
        paramsList.AddRange(_selfWeights.ToArray());
        paramsList.AddRange(_combinationWeights.ToArray());

        // Add all bias vectors
        paramsList.AddRange(_incomingBias.ToArray());
        paramsList.AddRange(_outgoingBias.ToArray());
        paramsList.AddRange(_selfBias.ToArray());
        paramsList.AddRange(_combinationBias.ToArray());

        // Add gating parameters if enabled
        if (_useGating && _gateWeights != null && _gateBias != null)
        {
            paramsList.AddRange(_gateWeights.ToArray());
            paramsList.AddRange(_gateBias.ToArray());
        }

        return new Vector<T>(paramsList.ToArray());
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int expectedSize = _incomingWeights.Length + _outgoingWeights.Length + _selfWeights.Length +
                          _combinationWeights.Length + _incomingBias.Length + _outgoingBias.Length +
                          _selfBias.Length + _combinationBias.Length;

        if (_useGating && _gateWeights != null && _gateBias != null)
        {
            expectedSize += _gateWeights.Length + _gateBias.Length;
        }

        if (parameters.Length != expectedSize)
        {
            throw new ArgumentException($"Expected {expectedSize} parameters, but got {parameters.Length}");
        }

        int index = 0;

        // Set weight matrices
        var incomingWeightsParams = parameters.SubVector(index, _incomingWeights.Length);
        _incomingWeights = Tensor<T>.FromVector(incomingWeightsParams).Reshape(_incomingWeights.Shape);
        index += _incomingWeights.Length;

        var outgoingWeightsParams = parameters.SubVector(index, _outgoingWeights.Length);
        _outgoingWeights = Tensor<T>.FromVector(outgoingWeightsParams).Reshape(_outgoingWeights.Shape);
        index += _outgoingWeights.Length;

        var selfWeightsParams = parameters.SubVector(index, _selfWeights.Length);
        _selfWeights = Tensor<T>.FromVector(selfWeightsParams).Reshape(_selfWeights.Shape);
        index += _selfWeights.Length;

        var combinationWeightsParams = parameters.SubVector(index, _combinationWeights.Length);
        _combinationWeights = Tensor<T>.FromVector(combinationWeightsParams).Reshape(_combinationWeights.Shape);
        index += _combinationWeights.Length;

        // Set bias vectors
        var incomingBiasParams = parameters.SubVector(index, _incomingBias.Length);
        _incomingBias = Tensor<T>.FromVector(incomingBiasParams);
        index += _incomingBias.Length;

        var outgoingBiasParams = parameters.SubVector(index, _outgoingBias.Length);
        _outgoingBias = Tensor<T>.FromVector(outgoingBiasParams);
        index += _outgoingBias.Length;

        var selfBiasParams = parameters.SubVector(index, _selfBias.Length);
        _selfBias = Tensor<T>.FromVector(selfBiasParams);
        index += _selfBias.Length;

        var combinationBiasParams = parameters.SubVector(index, _combinationBias.Length);
        _combinationBias = Tensor<T>.FromVector(combinationBiasParams);
        index += _combinationBias.Length;

        // Set gating parameters if enabled
        if (_useGating && _gateWeights != null && _gateBias != null)
        {
            var gateWeightsParams = parameters.SubVector(index, _gateWeights.Length);
            _gateWeights = Tensor<T>.FromVector(gateWeightsParams).Reshape(_gateWeights.Shape);
            index += _gateWeights.Length;

            var gateBiasParams = parameters.SubVector(index, _gateBias.Length);
            _gateBias = Tensor<T>.FromVector(gateBiasParams);
        }
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastIncoming = null;
        _lastOutgoing = null;
        _lastSelf = null;
        _lastCombined = null;
        _lastGates = null;
        _adjForBatch = null;
        _incomingWeightsGradient = null;
        _outgoingWeightsGradient = null;
        _selfWeightsGradient = null;
        _combinationWeightsGradient = null;
        _incomingBiasGradient = null;
        _outgoingBiasGradient = null;
        _selfBiasGradient = null;
        _combinationBiasGradient = null;
        _gateWeightsGradient = null;
        _gateBiasGradient = null;
    }

    /// <inheritdoc/>
    public override bool SupportsJitCompilation => true;

    /// <inheritdoc/>
    /// <summary>
    /// Exports the layer's forward pass as a JIT-compilable computation graph.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the directional graph convolution.</returns>
    /// <remarks>
    /// <para>
    /// The computation graph performs directional graph convolution:
    /// 1. Incoming aggregation: A @ (X @ W_in) + b_in
    /// 2. Outgoing aggregation: A^T @ (X @ W_out) + b_out
    /// 3. Self transformation: X @ W_self + b_self
    /// 4. Concatenate all three representations
    /// 5. Apply gating if enabled
    /// 6. Final combination: combined @ W_comb + b_comb
    /// 7. Apply activation function
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        // Create symbolic inputs for node features [batch, nodes, features]
        int numNodes = InputShape[0];
        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = Autodiff.TensorOperations<T>.Variable(symbolicInput, "node_features");
        inputNodes.Add(inputNode);

        // Create symbolic input for adjacency matrix [batch, nodes, nodes]
        var symbolicAdj = new Tensor<T>([1, numNodes, numNodes]);
        var adjNode = Autodiff.TensorOperations<T>.Variable(symbolicAdj, "adjacency_matrix");
        inputNodes.Add(adjNode);

        // Export learnable parameters as constants
        var incomingWeightsNode = Autodiff.TensorOperations<T>.Constant(_incomingWeights, "incoming_weights");
        var outgoingWeightsNode = Autodiff.TensorOperations<T>.Constant(_outgoingWeights, "outgoing_weights");
        var selfWeightsNode = Autodiff.TensorOperations<T>.Constant(_selfWeights, "self_weights");
        var combinationWeightsNode = Autodiff.TensorOperations<T>.Constant(_combinationWeights, "combination_weights");

        var incomingBiasNode = Autodiff.TensorOperations<T>.Constant(_incomingBias, "incoming_bias");
        var outgoingBiasNode = Autodiff.TensorOperations<T>.Constant(_outgoingBias, "outgoing_bias");
        var selfBiasNode = Autodiff.TensorOperations<T>.Constant(_selfBias, "self_bias");
        var combinationBiasNode = Autodiff.TensorOperations<T>.Constant(_combinationBias, "combination_bias");

        // Step 1: Incoming aggregation: A @ (X @ W_in) + b_in
        var xwIn = Autodiff.TensorOperations<T>.MatrixMultiply(inputNode, incomingWeightsNode);
        var incomingAggregated = Autodiff.TensorOperations<T>.BatchMatrixMultiply(adjNode, xwIn);
        var incomingWithBias = Autodiff.TensorOperations<T>.Add(incomingAggregated, incomingBiasNode);

        // Step 2: Outgoing aggregation: A^T @ (X @ W_out) + b_out
        var adjTransposed = Autodiff.TensorOperations<T>.Transpose(adjNode);
        var xwOut = Autodiff.TensorOperations<T>.MatrixMultiply(inputNode, outgoingWeightsNode);
        var outgoingAggregated = Autodiff.TensorOperations<T>.BatchMatrixMultiply(adjTransposed, xwOut);
        var outgoingWithBias = Autodiff.TensorOperations<T>.Add(outgoingAggregated, outgoingBiasNode);

        // Step 3: Self transformation: X @ W_self + b_self
        var selfTransformed = Autodiff.TensorOperations<T>.MatrixMultiply(inputNode, selfWeightsNode);
        var selfWithBias = Autodiff.TensorOperations<T>.Add(selfTransformed, selfBiasNode);

        // Step 4: Concatenate incoming, outgoing, and self features
        var combinedList = new List<ComputationNode<T>> { incomingWithBias, outgoingWithBias, selfWithBias };
        var combined = Autodiff.TensorOperations<T>.Concat(combinedList, axis: -1);

        ComputationNode<T> gatedCombined;

        // Step 5: Apply gating if enabled
        if (_useGating && _gateWeights != null && _gateBias != null)
        {
            var gateWeightsNode = Autodiff.TensorOperations<T>.Constant(_gateWeights, "gate_weights");
            var gateBiasNode = Autodiff.TensorOperations<T>.Constant(_gateBias, "gate_bias");

            // Compute gates: combined @ W_gate + b_gate
            var gateLogits = Autodiff.TensorOperations<T>.MatrixMultiply(combined, gateWeightsNode);
            var gateLogitsWithBias = Autodiff.TensorOperations<T>.Add(gateLogits, gateBiasNode);

            // Apply sigmoid to get gates
            var gates = Autodiff.TensorOperations<T>.Sigmoid(gateLogitsWithBias);

            // For JIT, we simplify gating by using element-wise multiplication of broadcasted gates
            // This is an approximation that applies gates uniformly across features in each group
            gatedCombined = Autodiff.TensorOperations<T>.ElementwiseMultiply(combined, gates);
        }
        else
        {
            gatedCombined = combined;
        }

        // Step 6: Final combination: gatedCombined @ W_comb + b_comb
        var output = Autodiff.TensorOperations<T>.MatrixMultiply(gatedCombined, combinationWeightsNode);
        output = Autodiff.TensorOperations<T>.Add(output, combinationBiasNode);

        // Step 7: Apply activation function if needed
        if (ScalarActivation is not null && ScalarActivation is not IdentityActivation<T>)
        {
            if (ScalarActivation.SupportsJitCompilation)
            {
                output = ScalarActivation.ApplyToGraph(output);
            }
            else
            {
                // Fallback: apply activation directly to values
                var activated = ScalarActivation.Activate(output.Value);
                output = Autodiff.TensorOperations<T>.Constant(activated, "activated_output");
            }
        }

        return output;
    }
}
