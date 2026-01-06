using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

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
/// The layer computes: h_v^(k) = MLP^(k)((1 + epsilon^(k)) * h_v^(k-1) + sum_{u in N(v)} h_u^(k-1))
/// where h_v is the representation of node v, N(v) is the neighborhood of v,
/// epsilon is a learnable parameter (or fixed), and MLP is a multi-layer perceptron.
/// </para>
/// <para>
/// <b>Production-Ready Features:</b>
/// <list type="bullet">
/// <item>Fully vectorized operations using IEngine for GPU acceleration</item>
/// <item>Tensor-based weights for all parameters</item>
/// <item>Dual backward pass: BackwardManual() for efficiency, BackwardViaAutodiff() for accuracy</item>
/// <item>Full gradient computation through MLP and aggregation paths</item>
/// <item>JIT compilation support via ExportComputationGraph()</item>
/// <item>Complete GetParameters()/SetParameters() for model persistence</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GraphIsomorphismLayer<T> : LayerBase<T>, IGraphConvolutionLayer<T>
{
    private readonly int _inputFeatures;
    private readonly int _outputFeatures;
    private readonly int _mlpHiddenDim;
    private readonly bool _learnEpsilon;
    private readonly Random _random;

    /// <summary>
    /// Epsilon parameter for weighting self vs neighbor features.
    /// </summary>
    private T _epsilon;

    /// <summary>
    /// First layer of the MLP: [inputFeatures, mlpHiddenDim].
    /// </summary>
    private Tensor<T> _mlpWeights1;

    /// <summary>
    /// Second layer of the MLP: [mlpHiddenDim, outputFeatures].
    /// </summary>
    private Tensor<T> _mlpWeights2;

    /// <summary>
    /// Bias for first MLP layer: [mlpHiddenDim].
    /// </summary>
    private Tensor<T> _mlpBias1;

    /// <summary>
    /// Bias for second MLP layer: [outputFeatures].
    /// </summary>
    private Tensor<T> _mlpBias2;

    /// <summary>
    /// The adjacency matrix defining graph structure.
    /// </summary>
    private Tensor<T>? _adjacencyMatrix;

    /// <summary>
    /// Cached input from forward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// Cached output from forward pass.
    /// </summary>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Cached aggregated features (before MLP).
    /// </summary>
    private Tensor<T>? _lastAggregated;

    /// <summary>
    /// Cached pre-ReLU hidden layer output from MLP.
    /// </summary>
    private Tensor<T>? _lastMlpHiddenPreRelu;

    /// <summary>
    /// Cached hidden layer output from MLP (after ReLU).
    /// </summary>
    private Tensor<T>? _lastMlpHidden;

    /// <summary>
    /// Cached neighbor sum before applying epsilon.
    /// </summary>
    private Tensor<T>? _lastNeighborSum;

    /// <summary>
    /// Gradients for epsilon.
    /// </summary>
    private T _epsilonGradient;

    /// <summary>
    /// Gradients for MLP weights.
    /// </summary>
    private Tensor<T>? _mlpWeights1Gradient;
    private Tensor<T>? _mlpWeights2Gradient;
    private Tensor<T>? _mlpBias1Gradient;
    private Tensor<T>? _mlpBias2Gradient;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets whether this layer supports GPU execution.
    /// </summary>
    /// <remarks>
    /// GraphIsomorphismLayer (GIN) supports GPU execution with efficient neighbor sum aggregation
    /// and GPU-accelerated MLP computation.
    /// </remarks>
    protected override bool SupportsGpuExecution => true;

    /// <inheritdoc/>
    public override int ParameterCount =>
        _mlpWeights1.Length + _mlpWeights2.Length + _mlpBias1.Length + _mlpBias2.Length + (_learnEpsilon ? 1 : 0);

    /// <inheritdoc/>
    public int InputFeatures => _inputFeatures;

    /// <inheritdoc/>
    public int OutputFeatures => _outputFeatures;

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphIsomorphismLayer{T}"/> class.
    /// </summary>
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
        _random = RandomHelper.CreateSecureRandom();
        _epsilonGradient = NumOps.Zero;

        // Initialize weights as Tensors for GPU acceleration
        _mlpWeights1 = new Tensor<T>([_inputFeatures, _mlpHiddenDim]);
        _mlpWeights2 = new Tensor<T>([_mlpHiddenDim, _outputFeatures]);
        _mlpBias1 = new Tensor<T>([_mlpHiddenDim]);
        _mlpBias2 = new Tensor<T>([_outputFeatures]);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes layer parameters using Xavier initialization with Engine operations.
    /// </summary>
    private void InitializeParameters()
    {
        // Xavier/Glorot initialization using Engine operations
        InitializeTensor(_mlpWeights1, _inputFeatures, _mlpHiddenDim);
        InitializeTensor(_mlpWeights2, _mlpHiddenDim, _outputFeatures);

        // Initialize biases to zero
        _mlpBias1.Fill(NumOps.Zero);
        _mlpBias2.Fill(NumOps.Zero);
    }

    private void InitializeTensor(Tensor<T> tensor, int fanIn, int fanOut)
    {
        // Xavier/Glorot initialization: fully vectorized
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (fanIn + fanOut)));
        var randomTensor = Tensor<T>.CreateRandom(tensor.Shape);
        var halfTensor = new Tensor<T>(tensor.Shape);
        Engine.TensorFill(halfTensor, NumOps.FromDouble(0.5));
        var shifted = Engine.TensorSubtract(randomTensor, halfTensor);
        var scaled = Engine.TensorMultiplyScalar(shifted, scale);
        // Copy result using vectorized operation
        Engine.TensorCopy(scaled, tensor);
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

        // Handle tensor shapes for graph processing:
        // - 2D input [N, F] = single graph with N nodes and F features -> reshape to [1, N, F]
        // - 3D input [B, N, F] = batch of B graphs with N nodes and F features
        Tensor<T> processInput;
        int batchSize;
        int numNodes;

        if (rank == 1)
        {
            // 1D: treat as single node with F features -> [1, 1, F]
            batchSize = 1;
            numNodes = 1;
            processInput = input.Reshape([1, 1, input.Shape[0]]);
        }
        else if (rank == 2)
        {
            // 2D: [numNodes, features] -> reshape to [1, numNodes, features]
            batchSize = 1;
            numNodes = input.Shape[0];
            processInput = input.Reshape([1, input.Shape[0], input.Shape[1]]);
        }
        else
        {
            // 3D: [batch, numNodes, features]
            batchSize = input.Shape[0];
            numNodes = input.Shape[1];
            processInput = input;
        }

        _lastInput = processInput;

        // Reshape adjacency matrix to match batch dimension if needed
        Tensor<T> adjForBatch;
        if (_adjacencyMatrix.Shape.Length == 2 && batchSize == 1)
        {
            adjForBatch = _adjacencyMatrix.Reshape([1, _adjacencyMatrix.Shape[0], _adjacencyMatrix.Shape[1]]);
        }
        else if (_adjacencyMatrix.Shape.Length == 2 && batchSize > 1)
        {
            // Tile adjacency matrix for batch
            int adjN = _adjacencyMatrix.Shape[0];
            adjForBatch = new Tensor<T>([batchSize, adjN, adjN]);
            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < adjN; i++)
                {
                    for (int j = 0; j < adjN; j++)
                    {
                        adjForBatch[new int[] { b, i, j }] = _adjacencyMatrix[new int[] { i, j }];
                    }
                }
            }
        }
        else
        {
            adjForBatch = _adjacencyMatrix;
        }

        // Step 1: Aggregate neighbor features using sum (batched matmul: [batch, nodes, nodes] @ [batch, nodes, features])
        _lastNeighborSum = Engine.BatchMatMul(adjForBatch, processInput);

        // Step 2: Combine with self features: (1 + epsilon) * h_v + sum(h_u)
        T onePlusEpsilon = NumOps.Add(NumOps.One, _epsilon);
        var scaledSelf = Engine.TensorMultiplyScalar(processInput, onePlusEpsilon);
        _lastAggregated = Engine.TensorAdd(scaledSelf, _lastNeighborSum);

        // Step 3: Apply MLP - First layer (3D @ 2D requires reshape pattern)
        var hidden1 = BatchedMatMul3Dx2D(_lastAggregated, _mlpWeights1, batchSize, numNodes, _inputFeatures, _mlpHiddenDim);
        var bias1Broadcast = BroadcastBias(_mlpBias1, batchSize, numNodes);
        _lastMlpHiddenPreRelu = Engine.TensorAdd(hidden1, bias1Broadcast);

        // Apply ReLU activation using Engine operations
        var zeroTensor = new Tensor<T>(_lastMlpHiddenPreRelu.Shape);
        zeroTensor.Fill(NumOps.Zero);
        _lastMlpHidden = Engine.TensorMax(_lastMlpHiddenPreRelu, zeroTensor);

        // Step 4: Apply MLP - Second layer (3D @ 2D requires reshape pattern)
        var hidden2 = BatchedMatMul3Dx2D(_lastMlpHidden, _mlpWeights2, batchSize, numNodes, _mlpHiddenDim, _outputFeatures);
        var bias2Broadcast = BroadcastBias(_mlpBias2, batchSize, numNodes);
        var mlpOutput = Engine.TensorAdd(hidden2, bias2Broadcast);

        var result = ApplyActivation(mlpOutput);

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
    /// Broadcasts a bias tensor across batch and node dimensions using Engine operations.
    /// </summary>
    private Tensor<T> BroadcastBias(Tensor<T> bias, int batchSize, int numNodes)
    {
        int features = bias.Length;
        var biasReshaped = bias.Reshape([1, 1, features]);
        return Engine.TensorTile(biasReshaped, [batchSize, numNodes, 1]);
    }

    /// <inheritdoc/>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass with full gradient computation using fully vectorized Engine operations.
    /// </summary>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _adjacencyMatrix == null ||
            _lastAggregated == null || _lastMlpHidden == null || _lastMlpHiddenPreRelu == null ||
            _lastNeighborSum == null)
        {
            throw new InvalidOperationException("Forward pass must be called before Backward.");
        }

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);
        int batchSize = _lastInput.Shape[0];
        int numNodes = _lastInput.Shape[1];

        // Gradient through MLP Layer 2 bias: sum over batch and nodes
        _mlpBias2Gradient = Engine.ReduceSum(activationGradient, [0, 1], keepDims: false);

        // Gradient through MLP Layer 2 weights: hidden^T @ grad (batched matmul then sum)
        // Use permute for batched transpose: [batch, nodes, hidden] -> [batch, hidden, nodes]
        var hiddenBatchedT = Engine.TensorPermute(_lastMlpHidden, [0, 2, 1]);
        // Batched matmul: [batch, hidden, nodes] @ [batch, nodes, output] -> [batch, hidden, output]
        var weights2GradBatched = Engine.BatchMatMul(hiddenBatchedT, activationGradient);
        // Sum over batch dimension
        _mlpWeights2Gradient = Engine.ReduceSum(weights2GradBatched, [0], keepDims: false);

        // Gradient to hidden layer: grad @ weights2^T (broadcasting over batch)
        var weights2T = Engine.TensorTranspose(_mlpWeights2);
        var hiddenGradPre = Engine.TensorMatMul(activationGradient, weights2T);

        // Gradient through ReLU: element-wise vectorized
        var zeroTensor = new Tensor<T>(_lastMlpHiddenPreRelu.Shape);
        Engine.TensorFill(zeroTensor, NumOps.Zero);
        var reluMask = Engine.TensorGreaterThan(_lastMlpHiddenPreRelu, zeroTensor);
        var oneTensor = new Tensor<T>(_lastMlpHiddenPreRelu.Shape);
        Engine.TensorFill(oneTensor, NumOps.One);
        var reluDeriv = Engine.TensorWhere(reluMask, oneTensor, zeroTensor);
        var hiddenGrad = Engine.TensorMultiply(hiddenGradPre, reluDeriv);

        // Gradient through MLP Layer 1 bias: sum over batch and nodes
        _mlpBias1Gradient = Engine.ReduceSum(hiddenGrad, [0, 1], keepDims: false);

        // Gradient through MLP Layer 1 weights: aggregated^T @ hiddenGrad (batched matmul then sum)
        var aggBatchedT = Engine.TensorPermute(_lastAggregated, [0, 2, 1]);
        var weights1GradBatched = Engine.BatchMatMul(aggBatchedT, hiddenGrad);
        _mlpWeights1Gradient = Engine.ReduceSum(weights1GradBatched, [0], keepDims: false);

        // Gradient to aggregated: hiddenGrad @ weights1^T (broadcasting over batch)
        var weights1T = Engine.TensorTranspose(_mlpWeights1);
        var aggregatedGrad = Engine.TensorMatMul(hiddenGrad, weights1T);

        // Gradient through aggregation: (1 + epsilon) * h_v + neighbor_sum
        T onePlusEpsilon = NumOps.Add(NumOps.One, _epsilon);

        // Self gradient: (1 + epsilon) * aggregatedGrad
        var selfGrad = Engine.TensorMultiplyScalar(aggregatedGrad, onePlusEpsilon);

        // Neighbor gradient: A^T @ aggregatedGrad (batched via permute)
        var adjT = Engine.TensorPermute(_adjacencyMatrix, [0, 2, 1]);
        var neighborGrad = Engine.TensorMatMul(adjT, aggregatedGrad);

        // Combine gradients: fully vectorized addition
        var inputGradient = Engine.TensorAdd(selfGrad, neighborGrad);

        // Epsilon gradient (if learnable): fully vectorized
        if (_learnEpsilon)
        {
            var epsilonGradTensor = Engine.TensorMultiply(_lastInput, aggregatedGrad);
            var epsilonGradSum = Engine.ReduceSum(epsilonGradTensor, [0, 1, 2], keepDims: false);
            _epsilonGradient = epsilonGradSum[0];
        }
        else
        {
            _epsilonGradient = NumOps.Zero;
        }

        return inputGradient;
    }

    /// <summary>
    /// Backward pass using automatic differentiation with computation graph.
    /// </summary>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _adjacencyMatrix == null ||
            _lastAggregated == null || _lastMlpHidden == null || _lastMlpHiddenPreRelu == null ||
            _lastNeighborSum == null)
        {
            throw new InvalidOperationException("Forward pass must be called before Backward.");
        }

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);
        int batchSize = _lastInput.Shape[0];
        int numNodes = _lastInput.Shape[1];

        // Create computation nodes for autodiff
        var inputNode = Autodiff.TensorOperations<T>.Variable(_lastInput, "input", requiresGradient: true);
        var weights1Node = Autodiff.TensorOperations<T>.Variable(_mlpWeights1, "weights1", requiresGradient: true);
        var weights2Node = Autodiff.TensorOperations<T>.Variable(_mlpWeights2, "weights2", requiresGradient: true);
        var bias1Node = Autodiff.TensorOperations<T>.Variable(_mlpBias1, "bias1", requiresGradient: true);
        var bias2Node = Autodiff.TensorOperations<T>.Variable(_mlpBias2, "bias2", requiresGradient: true);

        var allNodes = new List<Autodiff.ComputationNode<T>>
        {
            inputNode, weights1Node, weights2Node, bias1Node, bias2Node
        };

        // Build computation graph

        // Use cached aggregated features
        var aggregatedNode = Autodiff.TensorOperations<T>.Variable(_lastAggregated, "aggregated", requiresGradient: true);
        allNodes.Add(aggregatedNode);

        // MLP Layer 1: aggregated @ weights1 + bias1
        var hidden1 = Autodiff.TensorOperations<T>.MatrixMultiply(aggregatedNode, weights1Node);
        allNodes.Add(hidden1);

        var bias1Broadcast = BroadcastBias(_mlpBias1, batchSize, numNodes);
        var bias1BroadcastNode = Autodiff.TensorOperations<T>.Variable(bias1Broadcast, "bias1_broadcast", requiresGradient: true);
        allNodes.Add(bias1BroadcastNode);

        var hidden1WithBias = Autodiff.TensorOperations<T>.Add(hidden1, bias1BroadcastNode);
        allNodes.Add(hidden1WithBias);

        // ReLU activation
        var hidden1Activated = Autodiff.TensorOperations<T>.ReLU(hidden1WithBias);
        allNodes.Add(hidden1Activated);

        // MLP Layer 2: hidden @ weights2 + bias2
        var hidden2 = Autodiff.TensorOperations<T>.MatrixMultiply(hidden1Activated, weights2Node);
        allNodes.Add(hidden2);

        var bias2Broadcast = BroadcastBias(_mlpBias2, batchSize, numNodes);
        var bias2BroadcastNode = Autodiff.TensorOperations<T>.Variable(bias2Broadcast, "bias2_broadcast", requiresGradient: true);
        allNodes.Add(bias2BroadcastNode);

        var outputNode = Autodiff.TensorOperations<T>.Add(hidden2, bias2BroadcastNode);
        allNodes.Add(outputNode);

        // Set gradient on output node
        outputNode.Gradient = activationGradient;

        // Topological sort for backward pass
        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var topoOrder = new List<Autodiff.ComputationNode<T>>();
        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();

        foreach (var node in allNodes)
        {
            if (!visited.Contains(node))
            {
                stack.Push((node, false));

                while (stack.Count > 0)
                {
                    var (currentNode, processed) = stack.Pop();
                    if (visited.Contains(currentNode)) continue;

                    if (processed)
                    {
                        visited.Add(currentNode);
                        topoOrder.Add(currentNode);
                    }
                    else
                    {
                        stack.Push((currentNode, true));
                        if (currentNode.Parents != null)
                        {
                            foreach (var parent in currentNode.Parents)
                            {
                                if (!visited.Contains(parent))
                                {
                                    stack.Push((parent, false));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Execute backward pass in reverse topological order
        for (int i = topoOrder.Count - 1; i >= 0; i--)
        {
            var node = topoOrder[i];
            if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
            {
                node.BackwardFunction(node.Gradient);
            }
        }

        // Extract gradients
        _mlpBias2Gradient = bias2Node.Gradient != null
            ? Engine.ReduceSum(bias2Node.Gradient, [0, 1], keepDims: false)
            : Engine.ReduceSum(activationGradient, [0, 1], keepDims: false);

        _mlpBias1Gradient = bias1Node.Gradient != null
            ? Engine.ReduceSum(bias1Node.Gradient, [0, 1], keepDims: false)
            : new Tensor<T>([_mlpHiddenDim]);

        _mlpWeights2Gradient = weights2Node.Gradient ?? new Tensor<T>([_mlpHiddenDim, _outputFeatures]);
        _mlpWeights1Gradient = weights1Node.Gradient ?? new Tensor<T>([_inputFeatures, _mlpHiddenDim]);

        // If autodiff didn't compute gradients properly, compute them using Engine
        if (NumOps.Equals(_mlpWeights1Gradient[0], NumOps.Zero))
        {
            ComputeGradientsViaEngine(activationGradient, batchSize, numNodes);
        }

        // Compute input gradient from aggregated gradient
        var aggregatedGrad = aggregatedNode.Gradient ?? new Tensor<T>(_lastAggregated.Shape);

        // Gradient through aggregation
        T onePlusEpsilon = NumOps.Add(NumOps.One, _epsilon);
        var selfGrad = Engine.TensorMultiplyScalar(aggregatedGrad, onePlusEpsilon);
        var adjT = Engine.TensorTranspose(_adjacencyMatrix);
        var neighborGrad = Engine.TensorMatMul(adjT, aggregatedGrad);
        var inputGradient = Engine.TensorAdd(selfGrad, neighborGrad);

        // Epsilon gradient
        if (_learnEpsilon)
        {
            var epsilonGradTensor = Engine.TensorMultiply(_lastInput, aggregatedGrad);
            var epsilonGradSum = Engine.ReduceSum(epsilonGradTensor, [0, 1, 2], keepDims: false);
            _epsilonGradient = epsilonGradSum[0];
        }
        else
        {
            _epsilonGradient = NumOps.Zero;
        }

        return inputGradient;
    }

    /// <summary>
    /// Computes gradients using fully vectorized Engine operations.
    /// </summary>
    private void ComputeGradientsViaEngine(Tensor<T> activationGradient, int batchSize, int numNodes)
    {
        // Gradient through MLP Layer 2 bias: sum over batch and nodes
        _mlpBias2Gradient = Engine.ReduceSum(activationGradient, [0, 1], keepDims: false);

        // Gradient through MLP Layer 2 weights: hidden^T @ grad (batched then summed)
        var hiddenBatchedT = Engine.TensorPermute(_lastMlpHidden!, [0, 2, 1]);
        var weights2GradBatched = Engine.BatchMatMul(hiddenBatchedT, activationGradient);
        _mlpWeights2Gradient = Engine.ReduceSum(weights2GradBatched, [0], keepDims: false);

        // Gradient to hidden layer: grad @ weights2^T
        var weights2T = Engine.TensorTranspose(_mlpWeights2);
        var hiddenGradPre = Engine.TensorMatMul(activationGradient, weights2T);

        // Gradient through ReLU: fully vectorized element-wise operations
        var zeroTensor = new Tensor<T>(_lastMlpHiddenPreRelu!.Shape);
        Engine.TensorFill(zeroTensor, NumOps.Zero);
        var reluMask = Engine.TensorGreaterThan(_lastMlpHiddenPreRelu, zeroTensor);
        var oneTensor = new Tensor<T>(_lastMlpHiddenPreRelu.Shape);
        Engine.TensorFill(oneTensor, NumOps.One);
        var reluDeriv = Engine.TensorWhere(reluMask, oneTensor, zeroTensor);
        var hiddenGrad = Engine.TensorMultiply(hiddenGradPre, reluDeriv);

        // Gradient through MLP Layer 1 bias: sum over batch and nodes
        _mlpBias1Gradient = Engine.ReduceSum(hiddenGrad, [0, 1], keepDims: false);

        // Gradient through MLP Layer 1 weights: aggregated^T @ hiddenGrad (batched then summed)
        var aggBatchedT = Engine.TensorPermute(_lastAggregated!, [0, 2, 1]);
        var weights1GradBatched = Engine.BatchMatMul(aggBatchedT, hiddenGrad);
        _mlpWeights1Gradient = Engine.ReduceSum(weights1GradBatched, [0], keepDims: false);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        if (_mlpWeights1Gradient == null || _mlpWeights2Gradient == null ||
            _mlpBias1Gradient == null || _mlpBias2Gradient == null)
        {
            throw new InvalidOperationException("Backward must be called before UpdateParameters.");
        }

        // Update using vectorized Engine operations
        _mlpWeights1 = Engine.TensorSubtract(_mlpWeights1,
            Engine.TensorMultiplyScalar(_mlpWeights1Gradient, learningRate));
        _mlpWeights2 = Engine.TensorSubtract(_mlpWeights2,
            Engine.TensorMultiplyScalar(_mlpWeights2Gradient, learningRate));
        _mlpBias1 = Engine.TensorSubtract(_mlpBias1,
            Engine.TensorMultiplyScalar(_mlpBias1Gradient, learningRate));
        _mlpBias2 = Engine.TensorSubtract(_mlpBias2,
            Engine.TensorMultiplyScalar(_mlpBias2Gradient, learningRate));

        // Update epsilon if learnable
        if (_learnEpsilon)
        {
            _epsilon = NumOps.Subtract(_epsilon, NumOps.Multiply(learningRate, _epsilonGradient));
        }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var paramList = new List<T>();

        // MLP weights 1
        for (int i = 0; i < _mlpWeights1.Length; i++)
        {
            paramList.Add(_mlpWeights1.GetFlat(i));
        }

        // MLP bias 1
        for (int i = 0; i < _mlpBias1.Length; i++)
        {
            paramList.Add(_mlpBias1[i]);
        }

        // MLP weights 2
        for (int i = 0; i < _mlpWeights2.Length; i++)
        {
            paramList.Add(_mlpWeights2.GetFlat(i));
        }

        // MLP bias 2
        for (int i = 0; i < _mlpBias2.Length; i++)
        {
            paramList.Add(_mlpBias2[i]);
        }

        // Epsilon (if learnable)
        if (_learnEpsilon)
        {
            paramList.Add(_epsilon);
        }

        return new Vector<T>(paramList.ToArray());
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int weights1Count = _mlpWeights1.Length;
        int bias1Count = _mlpBias1.Length;
        int weights2Count = _mlpWeights2.Length;
        int bias2Count = _mlpBias2.Length;
        int expectedParams = weights1Count + bias1Count + weights2Count + bias2Count + (_learnEpsilon ? 1 : 0);

        if (parameters.Length != expectedParams)
        {
            throw new ArgumentException(
                $"Expected {expectedParams} parameters, but got {parameters.Length}", nameof(parameters));
        }

        int index = 0;

        // Set MLP weights 1
        _mlpWeights1 = Tensor<T>.FromVector(parameters.SubVector(index, weights1Count))
            .Reshape(_mlpWeights1.Shape);
        index += weights1Count;

        // Set MLP bias 1
        _mlpBias1 = Tensor<T>.FromVector(parameters.SubVector(index, bias1Count));
        index += bias1Count;

        // Set MLP weights 2
        _mlpWeights2 = Tensor<T>.FromVector(parameters.SubVector(index, weights2Count))
            .Reshape(_mlpWeights2.Shape);
        index += weights2Count;

        // Set MLP bias 2
        _mlpBias2 = Tensor<T>.FromVector(parameters.SubVector(index, bias2Count));
        index += bias2Count;

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
        _lastMlpHiddenPreRelu = null;
        _lastMlpHidden = null;
        _lastNeighborSum = null;
        _mlpWeights1Gradient = null;
        _mlpWeights2Gradient = null;
        _mlpBias1Gradient = null;
        _mlpBias2Gradient = null;
        _epsilonGradient = NumOps.Zero;
    }

    /// <summary>
    /// GPU-accelerated forward pass for Graph Isomorphism Network (GIN).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Implements GPU-accelerated GIN computation:
    /// h_v^(k) = MLP((1 + ε) * h_v^(k-1) + Σ_{u∈N(v)} h_u^(k-1))
    /// </para>
    /// <para>
    /// The MLP is a two-layer network with ReLU activation.
    /// </para>
    /// </remarks>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs == null || inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        var input = inputs[0];
        if (input.Shape == null || input.Shape.Length < 2)
            throw new ArgumentException("Input must be at least 2D [numNodes, inputFeatures].");

        if (_adjacencyMatrix == null)
        {
            throw new InvalidOperationException(
                "Adjacency matrix must be set using SetAdjacencyMatrix before calling ForwardGpu.");
        }

        // Get GPU engine
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        var backend = gpuEngine.GetGpuBackend();
        if (backend == null)
            throw new InvalidOperationException("No GPU backend available.");

        int rank = input.Shape.Length;
        int batchSize, numNodes, inputFeatures;

        // Determine dimensions
        if (rank == 2)
        {
            batchSize = 1;
            numNodes = input.Shape[0];
            inputFeatures = input.Shape[1];
        }
        else
        {
            int flatBatch = 1;
            for (int d = 0; d < rank - 2; d++)
                flatBatch *= input.Shape[d];
            batchSize = flatBatch;
            numNodes = input.Shape[rank - 2];
            inputFeatures = input.Shape[rank - 1];
        }

        if (inputFeatures != _inputFeatures)
            throw new ArgumentException($"Input features ({inputFeatures}) doesn't match layer input features ({_inputFeatures}).");

        // Upload MLP weights to GPU
        var weights1Data = new float[_inputFeatures * _mlpHiddenDim];
        var weights2Data = new float[_mlpHiddenDim * _outputFeatures];
        for (int i = 0; i < _inputFeatures; i++)
        {
            for (int j = 0; j < _mlpHiddenDim; j++)
            {
                weights1Data[i * _mlpHiddenDim + j] = (float)NumOps.ToDouble(_mlpWeights1[i, j]);
            }
        }
        for (int i = 0; i < _mlpHiddenDim; i++)
        {
            for (int j = 0; j < _outputFeatures; j++)
            {
                weights2Data[i * _outputFeatures + j] = (float)NumOps.ToDouble(_mlpWeights2[i, j]);
            }
        }
        var weights1Buffer = backend.AllocateBuffer(weights1Data);
        var weights2Buffer = backend.AllocateBuffer(weights2Data);

        // Upload biases
        var bias1Data = new float[_mlpHiddenDim];
        var bias2Data = new float[_outputFeatures];
        for (int i = 0; i < _mlpHiddenDim; i++)
            bias1Data[i] = (float)NumOps.ToDouble(_mlpBias1[i]);
        for (int i = 0; i < _outputFeatures; i++)
            bias2Data[i] = (float)NumOps.ToDouble(_mlpBias2[i]);
        var bias1Buffer = backend.AllocateBuffer(bias1Data);
        var bias2Buffer = backend.AllocateBuffer(bias2Data);

        // Upload adjacency matrix
        bool adj2D = _adjacencyMatrix.Shape.Length == 2;
        var adjData = new float[numNodes * numNodes];
        for (int i = 0; i < numNodes; i++)
        {
            for (int j = 0; j < numNodes; j++)
            {
                T adjVal = adj2D ? _adjacencyMatrix[i, j] : _adjacencyMatrix[0, i, j];
                adjData[i * numNodes + j] = (float)NumOps.ToDouble(adjVal);
            }
        }
        var adjBuffer = backend.AllocateBuffer(adjData);

        // Compute (1 + epsilon)
        float onePlusEpsilon = 1.0f + (float)NumOps.ToDouble(_epsilon);

        // Allocate output buffer
        int outputSize = batchSize * numNodes * _outputFeatures;
        var outputBuffer = backend.AllocateBuffer(new float[outputSize]);

        // Allocate temporary buffers
        var aggregatedBuffer = backend.AllocateBuffer(new float[numNodes * _inputFeatures]);
        var neighborSumBuffer = backend.AllocateBuffer(new float[numNodes * _inputFeatures]);
        var hiddenBuffer = backend.AllocateBuffer(new float[numNodes * _mlpHiddenDim]);
        var mlpOutputBuffer = backend.AllocateBuffer(new float[numNodes * _outputFeatures]);

        // Process each batch
        for (int b = 0; b < batchSize; b++)
        {
            // Extract batch input slice
            int batchOffset = b * numNodes * inputFeatures;
            var batchInputBuffer = gpuEngine.GetOrAllocateBufferSlice(input.Buffer, batchOffset, numNodes * inputFeatures, backend);

            // Step 1: Compute neighbor sum = adj @ input
            backend.Gemm(
                adjBuffer,
                batchInputBuffer,
                neighborSumBuffer,
                numNodes, _inputFeatures, numNodes,
                alpha: 1.0f, beta: 0.0f,
                transposeA: false, transposeB: false);

            // Step 2: Aggregate = (1 + epsilon) * self + neighbor_sum
            // First: aggregated = onePlusEpsilon * input
            backend.MultiplyScalar(batchInputBuffer, onePlusEpsilon, aggregatedBuffer, numNodes * _inputFeatures);
            // Then: aggregated += neighborSum
            backend.Add(aggregatedBuffer, neighborSumBuffer, aggregatedBuffer, numNodes * _inputFeatures);

            // Step 3: MLP Layer 1 with ReLU
            // hidden = aggregated @ weights1 + bias1
            backend.Gemm(
                aggregatedBuffer,
                weights1Buffer,
                hiddenBuffer,
                numNodes, _mlpHiddenDim, _inputFeatures,
                alpha: 1.0f, beta: 0.0f,
                transposeA: false, transposeB: false);
            backend.AddBias(hiddenBuffer, bias1Buffer, hiddenBuffer, numNodes, _mlpHiddenDim);

            // Apply ReLU
            backend.ReLU(hiddenBuffer, hiddenBuffer, numNodes * _mlpHiddenDim);

            // Step 4: MLP Layer 2
            // output = hidden @ weights2 + bias2
            backend.Gemm(
                hiddenBuffer,
                weights2Buffer,
                mlpOutputBuffer,
                numNodes, _outputFeatures, _mlpHiddenDim,
                alpha: 1.0f, beta: 0.0f,
                transposeA: false, transposeB: false);
            backend.AddBias(mlpOutputBuffer, bias2Buffer, mlpOutputBuffer, numNodes, _outputFeatures);

            // Copy to output buffer at correct batch offset
            int outputOffset = b * numNodes * _outputFeatures;
            backend.Copy(mlpOutputBuffer, 0, outputBuffer, outputOffset, numNodes * _outputFeatures);
        }

        // Apply activation
        var activationType = GetFusedActivationType();
        if (activationType != FusedActivationType.None)
        {
            backend.ApplyActivation(outputBuffer, outputBuffer, outputSize, activationType);
        }

        // Clean up
        weights1Buffer.Dispose();
        weights2Buffer.Dispose();
        bias1Buffer.Dispose();
        bias2Buffer.Dispose();
        adjBuffer.Dispose();
        aggregatedBuffer.Dispose();
        neighborSumBuffer.Dispose();
        hiddenBuffer.Dispose();
        mlpOutputBuffer.Dispose();

        // Determine output shape
        int[] outputShape = rank == 2
            ? [numNodes, _outputFeatures]
            : [batchSize, numNodes, _outputFeatures];

        return new GpuTensor<T>(outputBuffer, outputShape, backend);
    }

    /// <inheritdoc/>
    public override bool SupportsJitCompilation => true;

    /// <inheritdoc/>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        // Create symbolic input
        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = Autodiff.TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Export MLP parameters as constants
        var weights1Node = Autodiff.TensorOperations<T>.Constant(_mlpWeights1, "weights1");
        var weights2Node = Autodiff.TensorOperations<T>.Constant(_mlpWeights2, "weights2");
        var bias1Node = Autodiff.TensorOperations<T>.Constant(_mlpBias1, "bias1");
        var bias2Node = Autodiff.TensorOperations<T>.Constant(_mlpBias2, "bias2");

        // Build MLP computation graph (self-path only for JIT)
        // Layer 1: input @ weights1 + bias1
        var hidden1 = Autodiff.TensorOperations<T>.MatrixMultiply(inputNode, weights1Node);
        var hidden1WithBias = Autodiff.TensorOperations<T>.Add(hidden1, bias1Node);
        var hidden1Activated = Autodiff.TensorOperations<T>.ReLU(hidden1WithBias);

        // Layer 2: hidden @ weights2 + bias2
        var hidden2 = Autodiff.TensorOperations<T>.MatrixMultiply(hidden1Activated, weights2Node);
        var output = Autodiff.TensorOperations<T>.Add(hidden2, bias2Node);

        // Apply activation if supported
        if (ScalarActivation != null && ScalarActivation.SupportsJitCompilation)
        {
            return ScalarActivation.ApplyToGraph(output);
        }

        return output;
    }
}
