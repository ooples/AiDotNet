using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

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
/// The layer performs: h_v = sigma(W_self * h_v + W_neigh * AGGREGATE({h_u : u in N(v)}) + b)
/// where h_v is the representation of node v, N(v) is the neighborhood of v,
/// AGGREGATE is an aggregation function (mean, max, sum), and sigma is an activation function.
/// </para>
/// <para>
/// <b>Production-Ready Features:</b>
/// <list type="bullet">
/// <item>Fully vectorized operations using IEngine for GPU acceleration</item>
/// <item>Tensor-based weights for all parameters</item>
/// <item>Dual backward pass: BackwardManual() for efficiency, BackwardViaAutodiff() for accuracy</item>
/// <item>Full gradient computation through aggregation paths</item>
/// <item>JIT compilation support via ExportComputationGraph()</item>
/// <item>Complete GetParameters()/SetParameters() for model persistence</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GraphSAGELayer<T> : LayerBase<T>, IGraphConvolutionLayer<T>
{
    private readonly int _inputFeatures;
    private readonly int _outputFeatures;
    private readonly SAGEAggregatorType _aggregatorType;
    private readonly bool _normalize;
    private readonly Random _random;

    /// <summary>
    /// Weight tensor for self features. Shape: [inputFeatures, outputFeatures].
    /// </summary>
    private Tensor<T> _selfWeights;

    /// <summary>
    /// Weight tensor for neighbor features. Shape: [inputFeatures, outputFeatures].
    /// </summary>
    private Tensor<T> _neighborWeights;

    /// <summary>
    /// Bias tensor. Shape: [outputFeatures].
    /// </summary>
    private Tensor<T> _bias;

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
    /// Cached aggregated neighbor features.
    /// </summary>
    private Tensor<T>? _lastAggregated;

    /// <summary>
    /// Cached pre-normalization output for gradient computation.
    /// </summary>
    private Tensor<T>? _lastPreNorm;

    /// <summary>
    /// Cached degrees for each node.
    /// </summary>
    private Tensor<T>? _lastDegrees;

    /// <summary>
    /// Gradients for self weights.
    /// </summary>
    private Tensor<T>? _selfWeightsGradient;

    /// <summary>
    /// Gradients for neighbor weights.
    /// </summary>
    private Tensor<T>? _neighborWeightsGradient;

    /// <summary>
    /// Gradients for bias.
    /// </summary>
    private Tensor<T>? _biasGradient;

    /// <summary>
    /// Cached max indices from MaxPool aggregation for proper backward pass.
    /// </summary>
    private int[]? _lastMaxIndices;

    /// <summary>
    /// Cached input shape before max pooling for backward computation.
    /// </summary>
    private int[]? _lastMaxInputShape;

    /// <summary>
    /// Cached reshaped adjacency matrix for backward pass.
    /// </summary>
    private Tensor<T>? _adjForBatch;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets whether this layer supports GPU execution.
    /// </summary>
    /// <remarks>
    /// GraphSAGELayer supports GPU execution with efficient sparse aggregation when using
    /// Sum or Mean aggregators. MaxPool aggregation uses a hybrid approach.
    /// </remarks>
    protected override bool SupportsGpuExecution => true;

    /// <inheritdoc/>
    public override int ParameterCount => _selfWeights.Length + _neighborWeights.Length + _bias.Length;

    /// <inheritdoc/>
    public int InputFeatures => _inputFeatures;

    /// <inheritdoc/>
    public int OutputFeatures => _outputFeatures;

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphSAGELayer{T}"/> class.
    /// </summary>
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
        _random = RandomHelper.CreateSecureRandom();

        // Initialize weights as Tensors for GPU acceleration
        _selfWeights = new Tensor<T>([_inputFeatures, _outputFeatures]);
        _neighborWeights = new Tensor<T>([_inputFeatures, _outputFeatures]);
        _bias = new Tensor<T>([_outputFeatures]);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes layer parameters using Xavier initialization with Engine operations.
    /// </summary>
    private void InitializeParameters()
    {
        // Xavier/Glorot initialization using Engine operations
        InitializeTensor(_selfWeights, _inputFeatures, _outputFeatures);
        InitializeTensor(_neighborWeights, _inputFeatures, _outputFeatures);

        // Initialize bias to zero
        _bias.Fill(NumOps.Zero);
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

        // Ensure adjacency matrix has matching rank for batch operation
        // If adjacency is 2D [nodes, nodes] and input is 3D [batch, nodes, features], reshape to 3D
        Tensor<T> adjForBatch;
        if (_adjacencyMatrix.Shape.Length == 2 && processInput.Shape.Length == 3)
        {
            // Reshape 2D adjacency [nodes, nodes] to 3D [1, nodes, nodes] and broadcast to [batchSize, nodes, nodes]
            if (batchSize == 1)
            {
                adjForBatch = _adjacencyMatrix.Reshape([1, numNodes, numNodes]);
            }
            else
            {
                // Broadcast: repeat adjacency matrix for each batch item using Engine.TensorTile
                var adjReshaped = _adjacencyMatrix.Reshape([1, numNodes, numNodes]);
                adjForBatch = Engine.TensorTile(adjReshaped, [batchSize, 1, 1]);
            }
        }
        else
        {
            adjForBatch = _adjacencyMatrix;
        }

        // Store for backward pass
        _adjForBatch = adjForBatch;

        // Step 1: Compute degrees for normalization
        _lastDegrees = Engine.ReduceSum(adjForBatch, [2], keepDims: false); // [batch, nodes]

        // Clamp degrees to minimum of 1 to avoid division by zero
        var oneTensor = new Tensor<T>(_lastDegrees.Shape);
        oneTensor.Fill(NumOps.One);
        var safeDegrees = Engine.TensorMax(_lastDegrees, oneTensor);

        // Step 2: Aggregate neighbor features using vectorized operations
        _lastAggregated = ComputeVectorizedAggregation(processInput, safeDegrees, batchSize, numNodes, adjForBatch);

        // Step 3: Transform self features (3D @ 2D requires reshape pattern)
        var selfTransformed = BatchedMatMul3Dx2D(processInput, _selfWeights, batchSize, numNodes, _inputFeatures, _outputFeatures);

        // Step 4: Transform neighbor features (3D @ 2D requires reshape pattern)
        var neighborTransformed = BatchedMatMul3Dx2D(_lastAggregated, _neighborWeights, batchSize, numNodes, _inputFeatures, _outputFeatures);

        // Step 5: Combine: self + neighbor + bias
        var combined = Engine.TensorAdd(selfTransformed, neighborTransformed);
        var biasBroadcast = BroadcastBias(_bias, batchSize, numNodes);
        _lastPreNorm = Engine.TensorAdd(combined, biasBroadcast);

        // Step 6: Apply L2 normalization if enabled
        Tensor<T> output;
        if (_normalize)
        {
            output = L2NormalizeVectorized(_lastPreNorm, batchSize, numNodes);
        }
        else
        {
            output = _lastPreNorm;
        }

        var result = ApplyActivation(output);

        // Only store for backward pass during training - skip during inference
        if (IsTrainingMode)
        {
            _lastOutput = result;
        }

        // Restore original shape for any-rank tensor support
        if (_originalInputShape != null && _originalInputShape.Length != 3)
        {
            if (_originalInputShape.Length == 2)
            {
                // Was 2D, return [nodes, outputFeatures]
                return result.Reshape([numNodes, _outputFeatures]);
            }
            else if (_originalInputShape.Length == 1)
            {
                // Was 1D, return [outputFeatures]
                return result.Reshape([_outputFeatures]);
            }
            else
            {
                // Higher-rank: restore leading dimensions
                var newShape = new int[_originalInputShape.Length];
                for (int d = 0; d < _originalInputShape.Length - 1; d++)
                    newShape[d] = _originalInputShape[d];
                newShape[_originalInputShape.Length - 1] = _outputFeatures;
                return result.Reshape(newShape);
            }
        }

        return result;
    }

    /// <summary>
    /// Computes vectorized aggregation using Engine operations.
    /// </summary>
    private Tensor<T> ComputeVectorizedAggregation(Tensor<T> input, Tensor<T> safeDegrees, int batchSize, int numNodes, Tensor<T> adjForBatch)
    {
        switch (_aggregatorType)
        {
            case SAGEAggregatorType.Sum:
                // Sum aggregation: A @ X (batched matmul: [batch, nodes, nodes] @ [batch, nodes, features])
                return Engine.BatchMatMul(adjForBatch, input);

            case SAGEAggregatorType.Mean:
                // Mean aggregation: (A @ X) / degree (batched matmul then divide)
                var sumAgg = Engine.BatchMatMul(adjForBatch, input);
                int features = input.Shape[2];
                var degreesExpanded = safeDegrees.Reshape([batchSize, numNodes, 1]);
                // Broadcast degrees to match feature dimension: [batch, nodes, 1] -> [batch, nodes, features]
                var degreesBroadcast = Engine.TensorTile(degreesExpanded, [1, 1, features]);
                return Engine.TensorDivide(sumAgg, degreesBroadcast);

            case SAGEAggregatorType.MaxPool:
                // Max aggregation requires masking non-neighbors with -inf then taking max
                return ComputeMaxAggregation(input, batchSize, numNodes, adjForBatch);

            default:
                return Engine.BatchMatMul(adjForBatch, input);
        }
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
    /// Computes max aggregation over neighbors using masked reduce.
    /// Stores max indices for proper backward pass gradient routing.
    /// </summary>
    private Tensor<T> ComputeMaxAggregation(Tensor<T> input, int batchSize, int numNodes, Tensor<T> adjForBatch)
    {
        int features = input.Shape[2];

        // Expand input for broadcasting: [batch, 1, nodes, features]
        var expanded = input.Reshape([batchSize, 1, numNodes, features]);

        // Tile to [batch, nodes, nodes, features]
        var tiled = Engine.TensorTile(expanded, [1, numNodes, 1, 1]);

        // Create mask from adjacency: [batch, nodes, nodes, 1]
        var adjExpanded = adjForBatch.Reshape([batchSize, numNodes, numNodes, 1]);

        // Mask non-neighbors with -inf
        var negInf = new Tensor<T>(tiled.Shape);
        negInf.Fill(NumOps.FromDouble(double.NegativeInfinity));

        var zeroTensor = new Tensor<T>(adjExpanded.Shape);
        zeroTensor.Fill(NumOps.Zero);
        var mask = Engine.TensorGreaterThan(adjExpanded, zeroTensor);

        // Broadcast mask to feature dimension
        var maskBroadcast = Engine.TensorTile(mask, [1, 1, 1, features]);
        var masked = Engine.TensorWhere(maskBroadcast, tiled, negInf);

        // Store input shape for backward pass
        _lastMaxInputShape = masked.Shape;

        // Reduce max over neighbors axis (axis 2) and store indices for backward
        var result = Engine.ReduceMax(masked, [2], keepDims: false, out int[] maxIndices);
        _lastMaxIndices = maxIndices;

        return result;
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

    /// <summary>
    /// Applies L2 normalization using vectorized Engine operations.
    /// </summary>
    private Tensor<T> L2NormalizeVectorized(Tensor<T> features, int batchSize, int numNodes)
    {
        // Compute squared values
        var squared = Engine.TensorMultiply(features, features);

        // Sum over feature dimension to get squared norms
        var normSquared = Engine.ReduceSum(squared, [2], keepDims: true);

        // Add epsilon for numerical stability
        var epsilon = new Tensor<T>(normSquared.Shape);
        epsilon.Fill(NumOps.FromDouble(1e-12));
        normSquared = Engine.TensorAdd(normSquared, epsilon);

        // Take square root
        var norm = Engine.TensorSqrt(normSquared);

        // Broadcast norm to match feature dimension: [batch, nodes, 1] -> [batch, nodes, features]
        int featureDim = features.Shape[2];
        var normBroadcast = Engine.TensorTile(norm, [1, 1, featureDim]);

        // Divide features by norm
        return Engine.TensorDivide(features, normBroadcast);
    }

    /// <inheritdoc/>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass with full gradient computation.
    /// </summary>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _adjacencyMatrix == null ||
            _lastAggregated == null || _lastPreNorm == null || _lastDegrees == null)
        {
            throw new InvalidOperationException("Forward pass must be called before Backward.");
        }

        // Reshape outputGradient to match _lastOutput's 3D shape if needed
        var gradForBackward = outputGradient;
        if (_originalInputShape != null && _originalInputShape.Length != 3 && outputGradient.Shape.Length != _lastOutput.Shape.Length)
        {
            // Reshape gradient to 3D [batch, nodes, features] to match _lastOutput
            gradForBackward = outputGradient.Reshape(_lastOutput.Shape);
        }

        var activationGradient = ApplyActivationDerivative(_lastOutput, gradForBackward);
        int batchSize = _lastInput.Shape[0];
        int numNodes = _lastInput.Shape[1];

        // Backprop through L2 normalization if enabled
        Tensor<T> preNormGradient;
        if (_normalize)
        {
            preNormGradient = BackpropThroughL2Norm(activationGradient, _lastPreNorm, batchSize, numNodes);
        }
        else
        {
            preNormGradient = activationGradient;
        }

        // Bias gradient: sum over batch and nodes
        _biasGradient = Engine.ReduceSum(preNormGradient, [0, 1], keepDims: false);

        // Self weights gradient: input^T @ preNormGradient (batched matmul then sum)
        // Use permute for batched transpose: [batch, nodes, features] -> [batch, features, nodes]
        var inputBatchedT = Engine.TensorPermute(_lastInput, [0, 2, 1]);
        // Batched matmul: [batch, features, nodes] @ [batch, nodes, output] -> [batch, features, output]
        var selfWeightsGradBatched = Engine.BatchMatMul(inputBatchedT, preNormGradient);
        // Sum over batch dimension
        _selfWeightsGradient = Engine.ReduceSum(selfWeightsGradBatched, [0], keepDims: false);

        // Neighbor weights gradient: aggregated^T @ preNormGradient (batched matmul then sum)
        // Use permute for batched transpose: [batch, nodes, features] -> [batch, features, nodes]
        var aggBatchedT = Engine.TensorPermute(_lastAggregated, [0, 2, 1]);
        // Batched matmul: [batch, features, nodes] @ [batch, nodes, output] -> [batch, features, output]
        var neighborWeightsGradBatched = Engine.BatchMatMul(aggBatchedT, preNormGradient);
        // Sum over batch dimension
        _neighborWeightsGradient = Engine.ReduceSum(neighborWeightsGradBatched, [0], keepDims: false);

        // Input gradient from self path: preNormGradient @ selfWeights^T
        var selfWeightsT = Engine.TensorTranspose(_selfWeights);
        var inputGradientSelf = Engine.TensorMatMul(preNormGradient, selfWeightsT);

        // Input gradient from neighbor path (through aggregation)
        var neighborWeightsT = Engine.TensorTranspose(_neighborWeights);
        var aggGradient = Engine.TensorMatMul(preNormGradient, neighborWeightsT);
        var inputGradientNeighbor = BackpropThroughAggregation(aggGradient, batchSize, numNodes);

        // Combine input gradients
        var inputGradient = Engine.TensorAdd(inputGradientSelf, inputGradientNeighbor);

        // Reshape back to original input shape if it was not 3D
        if (_originalInputShape != null && _originalInputShape.Length != 3)
        {
            return inputGradient.Reshape(_originalInputShape);
        }

        return inputGradient;
    }

    /// <summary>
    /// Backpropagates through L2 normalization.
    /// </summary>
    private Tensor<T> BackpropThroughL2Norm(Tensor<T> gradient, Tensor<T> preNorm, int batchSize, int numNodes)
    {
        // For L2 norm: y = x / ||x||
        // Gradient: dy/dx = (I - y * y^T) / ||x||

        int featureDim = preNorm.Shape[2];

        // Compute squared values and norms
        var squared = Engine.TensorMultiply(preNorm, preNorm);
        var normSquared = Engine.ReduceSum(squared, [2], keepDims: true);
        var epsilon = new Tensor<T>(normSquared.Shape);
        epsilon.Fill(NumOps.FromDouble(1e-12));
        normSquared = Engine.TensorAdd(normSquared, epsilon);
        var norm = Engine.TensorSqrt(normSquared);

        // Broadcast norm to match feature dimension: [batch, nodes, 1] -> [batch, nodes, features]
        var normBroadcast = Engine.TensorTile(norm, [1, 1, featureDim]);

        // Normalized output
        var normalized = Engine.TensorDivide(preNorm, normBroadcast);

        // Compute dot product of gradient and normalized output
        var dotProduct = Engine.TensorMultiply(gradient, normalized);
        var dotSum = Engine.ReduceSum(dotProduct, [2], keepDims: true);

        // Broadcast dotSum to match feature dimension
        var dotSumBroadcast = Engine.TensorTile(dotSum, [1, 1, featureDim]);

        // Gradient: (gradient - normalized * dot_sum) / norm
        var scaled = Engine.TensorMultiply(normalized, dotSumBroadcast);
        var diff = Engine.TensorSubtract(gradient, scaled);
        return Engine.TensorDivide(diff, normBroadcast);
    }

    /// <summary>
    /// Backpropagates through aggregation operation.
    /// </summary>
    private Tensor<T> BackpropThroughAggregation(Tensor<T> aggGradient, int batchSize, int numNodes)
    {
        // Use the stored batched adjacency matrix for backward pass
        var adjBatched = _adjForBatch ?? _adjacencyMatrix!;

        switch (_aggregatorType)
        {
            case SAGEAggregatorType.Sum:
                // For sum: gradient flows back through A^T (batched transpose)
                var adjT = Engine.TensorPermute(adjBatched, [0, 2, 1]); // [batch, nodes, nodes] -> transpose last 2 dims
                return Engine.BatchMatMul(adjT, aggGradient);

            case SAGEAggregatorType.Mean:
                // For mean: gradient flows back through A^T and is divided by degree
                var adjTMean = Engine.TensorPermute(adjBatched, [0, 2, 1]);
                var gradThroughAdj = Engine.BatchMatMul(adjTMean, aggGradient);
                var safeDegrees = Engine.TensorMax(_lastDegrees!, NumOps.One);
                var degExpanded = safeDegrees.Reshape([batchSize, numNodes, 1]);
                // Broadcast degree to match feature dimension
                int featureDim = aggGradient.Shape[2];
                var degBroadcast = Engine.TensorTile(degExpanded, [1, 1, featureDim]);
                return Engine.TensorDivide(gradThroughAdj, degBroadcast);

            case SAGEAggregatorType.MaxPool:
                // For max pooling: gradient only flows to the max elements
                // Use stored indices from forward pass for proper gradient routing
                if (_lastMaxIndices != null && _lastMaxInputShape != null)
                {
                    // Use ReduceMaxBackward to route gradients to correct positions
                    var maxGradExpanded = Engine.ReduceMaxBackward(aggGradient, _lastMaxIndices, _lastMaxInputShape);

                    // The expanded gradient has shape [batch, nodes, nodes, features]
                    // We need to sum over the target node dimension to get [batch, nodes, features]
                    return Engine.ReduceSum(maxGradExpanded, [2], keepDims: false);
                }
                else
                {
                    // Fallback if indices not available (shouldn't happen in normal flow)
                    var adjTMax = Engine.TensorPermute(adjBatched, [0, 2, 1]);
                    return Engine.BatchMatMul(adjTMax, aggGradient);
                }

            default:
                var adjTDefault = Engine.TensorPermute(adjBatched, [0, 2, 1]);
                return Engine.BatchMatMul(adjTDefault, aggGradient);
        }
    }

    /// <summary>
    /// Backward pass using automatic differentiation with computation graph.
    /// </summary>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _adjacencyMatrix == null ||
            _lastAggregated == null || _lastPreNorm == null || _lastDegrees == null)
        {
            throw new InvalidOperationException("Forward pass must be called before Backward.");
        }

        // Reshape outputGradient to match _lastOutput's 3D shape if needed
        var gradForBackward = outputGradient;
        if (_originalInputShape != null && _originalInputShape.Length != 3 && outputGradient.Shape.Length != _lastOutput.Shape.Length)
        {
            gradForBackward = outputGradient.Reshape(_lastOutput.Shape);
        }

        var activationGradient = ApplyActivationDerivative(_lastOutput, gradForBackward);
        int batchSize = _lastInput.Shape[0];
        int numNodes = _lastInput.Shape[1];

        // Create computation nodes for autodiff
        var inputNode = Autodiff.TensorOperations<T>.Variable(_lastInput, "input", requiresGradient: true);
        var selfWeightsNode = Autodiff.TensorOperations<T>.Variable(_selfWeights, "self_weights", requiresGradient: true);
        var neighborWeightsNode = Autodiff.TensorOperations<T>.Variable(_neighborWeights, "neighbor_weights", requiresGradient: true);
        var biasNode = Autodiff.TensorOperations<T>.Variable(_bias, "bias", requiresGradient: true);

        var allNodes = new List<Autodiff.ComputationNode<T>>
        {
            inputNode, selfWeightsNode, neighborWeightsNode, biasNode
        };

        // Build computation graph

        // Self transformation: input @ selfWeights
        var selfTransformed = Autodiff.TensorOperations<T>.MatrixMultiply(inputNode, selfWeightsNode);
        allNodes.Add(selfTransformed);

        // Use cached aggregated features
        var aggregatedNode = Autodiff.TensorOperations<T>.Variable(_lastAggregated, "aggregated", requiresGradient: true);
        allNodes.Add(aggregatedNode);

        // Neighbor transformation: aggregated @ neighborWeights
        var neighborTransformed = Autodiff.TensorOperations<T>.MatrixMultiply(aggregatedNode, neighborWeightsNode);
        allNodes.Add(neighborTransformed);

        // Combine: self + neighbor
        var combined = Autodiff.TensorOperations<T>.Add(selfTransformed, neighborTransformed);
        allNodes.Add(combined);

        // Add bias
        var biasBroadcast = BroadcastBias(_bias, batchSize, numNodes);
        var biasBroadcastNode = Autodiff.TensorOperations<T>.Variable(biasBroadcast, "bias_broadcast", requiresGradient: true);
        allNodes.Add(biasBroadcastNode);

        var withBias = Autodiff.TensorOperations<T>.Add(combined, biasBroadcastNode);
        allNodes.Add(withBias);

        // Use cached pre-norm output
        var outputNode = Autodiff.TensorOperations<T>.Variable(_lastPreNorm, "output", requiresGradient: true);
        allNodes.Add(outputNode);

        // Set gradient on output node (after handling normalization)
        Tensor<T> gradientToPropagate;
        if (_normalize)
        {
            gradientToPropagate = BackpropThroughL2Norm(activationGradient, _lastPreNorm, batchSize, numNodes);
        }
        else
        {
            gradientToPropagate = activationGradient;
        }
        outputNode.Gradient = gradientToPropagate;

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
        _biasGradient = biasNode.Gradient != null
            ? Engine.ReduceSum(biasNode.Gradient, [0, 1], keepDims: false)
            : Engine.ReduceSum(gradientToPropagate, [0, 1], keepDims: false);

        _selfWeightsGradient = selfWeightsNode.Gradient ?? new Tensor<T>([_inputFeatures, _outputFeatures]);
        _neighborWeightsGradient = neighborWeightsNode.Gradient ?? new Tensor<T>([_inputFeatures, _outputFeatures]);

        // If autodiff didn't compute gradients properly, compute them using Engine
        if (NumOps.Equals(_selfWeightsGradient[0], NumOps.Zero))
        {
            ComputeGradientsViaEngine(gradientToPropagate, batchSize, numNodes);
        }

        // Extract input gradient
        var inputGradient = inputNode.Gradient ?? new Tensor<T>(_lastInput.Shape);

        // Reshape back to original input shape if it was not 3D
        if (_originalInputShape != null && _originalInputShape.Length != 3)
        {
            return inputGradient.Reshape(_originalInputShape);
        }

        return inputGradient;
    }

    /// <summary>
    /// Computes gradients using fully vectorized Engine operations as fallback.
    /// </summary>
    private void ComputeGradientsViaEngine(Tensor<T> gradient, int batchSize, int numNodes)
    {
        // Self weights gradient: input^T @ gradient (batched matmul then sum)
        // Use permute for batched transpose: [batch, nodes, features] -> [batch, features, nodes]
        var inputBatchedT = Engine.TensorPermute(_lastInput!, [0, 2, 1]);
        // Batched matmul: [batch, features, nodes] @ [batch, nodes, output] -> [batch, features, output]
        var selfWeightsGradBatched = Engine.BatchMatMul(inputBatchedT, gradient);
        // Sum over batch dimension
        _selfWeightsGradient = Engine.ReduceSum(selfWeightsGradBatched, [0], keepDims: false);

        // Neighbor weights gradient: aggregated^T @ gradient (batched matmul then sum)
        // Use permute for batched transpose: [batch, nodes, features] -> [batch, features, nodes]
        var aggBatchedT = Engine.TensorPermute(_lastAggregated!, [0, 2, 1]);
        // Batched matmul: [batch, features, nodes] @ [batch, nodes, output] -> [batch, features, output]
        var neighborWeightsGradBatched = Engine.BatchMatMul(aggBatchedT, gradient);
        // Sum over batch dimension
        _neighborWeightsGradient = Engine.ReduceSum(neighborWeightsGradBatched, [0], keepDims: false);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        if (_selfWeightsGradient == null || _neighborWeightsGradient == null || _biasGradient == null)
        {
            throw new InvalidOperationException("Backward must be called before UpdateParameters.");
        }

        // Update using vectorized Engine operations
        _selfWeights = Engine.TensorSubtract(_selfWeights,
            Engine.TensorMultiplyScalar(_selfWeightsGradient, learningRate));
        _neighborWeights = Engine.TensorSubtract(_neighborWeights,
            Engine.TensorMultiplyScalar(_neighborWeightsGradient, learningRate));
        _bias = Engine.TensorSubtract(_bias,
            Engine.TensorMultiplyScalar(_biasGradient, learningRate));
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        return Vector<T>.Concatenate(
            new Vector<T>(_selfWeights.ToArray()),
            new Vector<T>(_neighborWeights.ToArray()),
            new Vector<T>(_bias.ToArray())
        );
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int selfCount = _selfWeights.Length;
        int neighborCount = _neighborWeights.Length;
        int biasCount = _bias.Length;
        int totalParams = selfCount + neighborCount + biasCount;

        if (parameters.Length != totalParams)
        {
            throw new ArgumentException(
                $"Expected {totalParams} parameters, but got {parameters.Length}", nameof(parameters));
        }

        int index = 0;

        _selfWeights = Tensor<T>.FromVector(parameters.SubVector(index, selfCount))
            .Reshape(_selfWeights.Shape);
        index += selfCount;

        _neighborWeights = Tensor<T>.FromVector(parameters.SubVector(index, neighborCount))
            .Reshape(_neighborWeights.Shape);
        index += neighborCount;

        _bias = Tensor<T>.FromVector(parameters.SubVector(index, biasCount));
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastAggregated = null;
        _lastPreNorm = null;
        _lastDegrees = null;
        _lastMaxIndices = null;
        _lastMaxInputShape = null;
        _adjForBatch = null;
        _selfWeightsGradient = null;
        _neighborWeightsGradient = null;
        _biasGradient = null;
    }

    /// <summary>
    /// GPU-accelerated forward pass for GraphSAGE layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Implements GPU-accelerated GraphSAGE aggregation:
    /// h_v = σ(W_self * h_v + W_neigh * AGG({h_u : u ∈ N(v)}) + b)
    /// </para>
    /// <para>
    /// Supports Sum, Mean, and MaxPool aggregators on GPU.
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

        var backend = gpuEngine.GetBackend();
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

        // Upload weights to GPU
        var selfWeightData = new float[_inputFeatures * _outputFeatures];
        var neighborWeightData = new float[_inputFeatures * _outputFeatures];
        for (int i = 0; i < _inputFeatures; i++)
        {
            for (int j = 0; j < _outputFeatures; j++)
            {
                selfWeightData[i * _outputFeatures + j] = (float)NumOps.ToDouble(_selfWeights[i, j]);
                neighborWeightData[i * _outputFeatures + j] = (float)NumOps.ToDouble(_neighborWeights[i, j]);
            }
        }
        var selfWeightBuffer = backend.AllocateBuffer(selfWeightData);
        var neighborWeightBuffer = backend.AllocateBuffer(neighborWeightData);

        // Upload bias
        var biasData = new float[_outputFeatures];
        for (int f = 0; f < _outputFeatures; f++)
            biasData[f] = (float)NumOps.ToDouble(_bias[f]);
        var biasBuffer = backend.AllocateBuffer(biasData);

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

        // Compute degrees for mean aggregation
        var degreeData = new float[numNodes];
        for (int i = 0; i < numNodes; i++)
        {
            float deg = 0;
            for (int j = 0; j < numNodes; j++)
            {
                deg += adjData[i * numNodes + j];
            }
            degreeData[i] = Math.Max(deg, 1.0f); // Avoid division by zero
        }
        var degreeBuffer = backend.AllocateBuffer(degreeData);

        // Allocate output buffer
        int outputSize = batchSize * numNodes * _outputFeatures;
        var outputBuffer = backend.AllocateBuffer(new float[outputSize]);

        // Allocate temporary buffers
        var selfTransformedBuffer = backend.AllocateBuffer(new float[numNodes * _outputFeatures]);
        var aggregatedBuffer = backend.AllocateBuffer(new float[numNodes * _inputFeatures]);
        var neighborTransformedBuffer = backend.AllocateBuffer(new float[numNodes * _outputFeatures]);

        // Process each batch
        for (int b = 0; b < batchSize; b++)
        {
            // Extract batch input slice using GPU-native view
            int batchOffset = b * numNodes * inputFeatures;
            var batchView = input.CreateView(batchOffset, [numNodes, inputFeatures]);
            var batchInputBuffer = batchView.Buffer;

            // Step 1: Transform self features
            // selfTransformed = input @ selfWeights
            backend.Gemm(
                batchInputBuffer,
                selfWeightBuffer,
                selfTransformedBuffer,
                numNodes, _outputFeatures, _inputFeatures);

            // Step 2: Aggregate neighbor features
            switch (_aggregatorType)
            {
                case SAGEAggregatorType.Sum:
                    // aggregated = adj @ input
                    backend.Gemm(
                        adjBuffer,
                        batchInputBuffer,
                        aggregatedBuffer,
                        numNodes, _inputFeatures, numNodes);
                    break;

                case SAGEAggregatorType.Mean:
                    // aggregated = adj @ input, then divide by degree
                    backend.Gemm(
                        adjBuffer,
                        batchInputBuffer,
                        aggregatedBuffer,
                        numNodes, _inputFeatures, numNodes);

                    // Divide each row by its degree (CPU fallback for this operation)
                    DivideByRowDegreeCpu(backend, aggregatedBuffer, degreeBuffer, numNodes, _inputFeatures);
                    break;

                case SAGEAggregatorType.MaxPool:
                    // For MaxPool, compute max over neighbors for each feature (CPU fallback)
                    MaxPoolNeighborsCpu(backend, batchInputBuffer, adjBuffer, aggregatedBuffer, numNodes, _inputFeatures);
                    break;
            }

            // Step 3: Transform aggregated features
            // neighborTransformed = aggregated @ neighborWeights
            backend.Gemm(
                aggregatedBuffer,
                neighborWeightBuffer,
                neighborTransformedBuffer,
                numNodes, _outputFeatures, _inputFeatures);

            // Step 4: Combine: self + neighbor + bias
            backend.Add(selfTransformedBuffer, neighborTransformedBuffer, selfTransformedBuffer, numNodes * _outputFeatures);
            backend.BiasAdd(selfTransformedBuffer, biasBuffer, selfTransformedBuffer, numNodes, _outputFeatures);

            // Step 5: Apply L2 normalization if enabled
            if (_normalize)
            {
                L2NormalizeRowsCpu(backend, selfTransformedBuffer, numNodes, _outputFeatures);
            }

            // Copy to output buffer at correct batch offset using GPU-native strided copy
            int outputOffset = b * numNodes * _outputFeatures;
            int copySize = numNodes * _outputFeatures;
            backend.Copy2DStrided(selfTransformedBuffer, outputBuffer, 1, copySize, outputSize, outputOffset);
            // Note: batchInputBuffer is a view and doesn't need disposal
        }

        // Apply activation using GPU-native base class method
        var activationType = GetFusedActivationType();
        if (activationType != FusedActivationType.None)
        {
            ApplyGpuActivation(backend, outputBuffer, outputBuffer, outputSize, activationType);
        }

        // Clean up
        selfWeightBuffer.Dispose();
        neighborWeightBuffer.Dispose();
        biasBuffer.Dispose();
        adjBuffer.Dispose();
        degreeBuffer.Dispose();
        selfTransformedBuffer.Dispose();
        aggregatedBuffer.Dispose();
        neighborTransformedBuffer.Dispose();

        // Determine output shape
        int[] outputShape = rank == 2
            ? [numNodes, _outputFeatures]
            : [batchSize, numNodes, _outputFeatures];

        return new GpuTensor<T>(backend, outputBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: false);
    }

    #region GPU Helper Methods

    /// <summary>
    /// CPU fallback for dividing each row by its degree.
    /// </summary>
    private static void DivideByRowDegreeCpu(IDirectGpuBackend backend, IGpuBuffer dataBuffer, IGpuBuffer degreeBuffer, int numNodes, int features)
    {
        // GPU-native implementation using broadcast multiply with reciprocal degrees
        // Step 1: Clamp degrees to minimum 1.0 using Max operation
        using var onesBuffer = backend.AllocateBuffer(numNodes);
        backend.Fill(onesBuffer, 1.0f, numNodes);
        using var clampedDegreesBuffer = backend.AllocateBuffer(numNodes);
        backend.Max(degreeBuffer, onesBuffer, clampedDegreesBuffer, numNodes);

        // Step 2: Compute reciprocal: 1/degrees
        using var reciprocalBuffer = backend.AllocateBuffer(numNodes);
        backend.Reciprocal(clampedDegreesBuffer, reciprocalBuffer, numNodes);

        // Step 3: Broadcast multiply each row by its reciprocal degree
        // data[i, f] *= reciprocal[i] for all f
        backend.BroadcastMultiplyFirstAxis(dataBuffer, reciprocalBuffer, dataBuffer, numNodes, features);
    }

    /// <summary>
    /// CPU fallback for max pooling over neighbors.
    /// </summary>
    private static void MaxPoolNeighborsCpu(IDirectGpuBackend backend, IGpuBuffer inputBuffer, IGpuBuffer adjBuffer, IGpuBuffer outputBuffer, int numNodes, int features)
    {
        float[] input = backend.DownloadBuffer(inputBuffer);
        float[] adj = backend.DownloadBuffer(adjBuffer);
        float[] result = new float[numNodes * features];

        for (int i = 0; i < numNodes; i++)
        {
            for (int f = 0; f < features; f++)
            {
                float maxVal = float.NegativeInfinity;
                bool hasNeighbor = false;
                for (int j = 0; j < numNodes; j++)
                {
                    if (adj[i * numNodes + j] > 0)
                    {
                        hasNeighbor = true;
                        maxVal = MathF.Max(maxVal, input[j * features + f]);
                    }
                }
                result[i * features + f] = hasNeighbor ? maxVal : 0;
            }
        }

        using var tempBuffer = backend.AllocateBuffer(result);
        backend.Copy(tempBuffer, outputBuffer, result.Length);
    }

    /// <summary>
    /// CPU fallback for L2 row normalization.
    /// </summary>
    private static void L2NormalizeRowsCpu(IDirectGpuBackend backend, IGpuBuffer buffer, int numNodes, int features)
    {
        float[] data = backend.DownloadBuffer(buffer);

        for (int i = 0; i < numNodes; i++)
        {
            float sumSq = 0;
            for (int f = 0; f < features; f++)
            {
                float val = data[i * features + f];
                sumSq += val * val;
            }
            float norm = MathF.Sqrt(sumSq + 1e-12f);
            for (int f = 0; f < features; f++)
            {
                data[i * features + f] /= norm;
            }
        }

        using var tempBuffer = backend.AllocateBuffer(data);
        backend.Copy(tempBuffer, buffer, data.Length);
    }

    /// <summary>
    /// CPU fallback for copying data to an offset in a destination buffer.
    /// </summary>
    private static void CopyToOffsetCpu(IDirectGpuBackend backend, IGpuBuffer sourceBuffer, IGpuBuffer destBuffer, int destOffset, int size)
    {
        float[] source = backend.DownloadBuffer(sourceBuffer);
        float[] dest = backend.DownloadBuffer(destBuffer);
        Array.Copy(source, 0, dest, destOffset, size);
        using var tempBuffer = backend.AllocateBuffer(dest);
        backend.Copy(tempBuffer, destBuffer, dest.Length);
    }

    /// <summary>
    /// CPU fallback for applying activation function.
    /// </summary>
    private static void ApplyActivationCpu(IDirectGpuBackend backend, IGpuBuffer buffer, int size, FusedActivationType activationType)
    {
        switch (activationType)
        {
            case FusedActivationType.ReLU:
                backend.Relu(buffer, buffer, size);
                break;
            case FusedActivationType.Sigmoid:
                backend.Sigmoid(buffer, buffer, size);
                break;
            case FusedActivationType.Tanh:
                backend.Tanh(buffer, buffer, size);
                break;
            case FusedActivationType.GELU:
                backend.Gelu(buffer, buffer, size);
                break;
                // None/Identity does nothing
        }
    }

    #endregion

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

        // Export learnable parameters as constants
        var selfWeightsNode = Autodiff.TensorOperations<T>.Constant(_selfWeights, "self_weights");
        var biasNode = Autodiff.TensorOperations<T>.Constant(_bias, "bias");

        // Build computation graph for self-loop path (most direct gradient flow)
        var selfContribution = Autodiff.TensorOperations<T>.MatrixMultiply(inputNode, selfWeightsNode);
        var output = Autodiff.TensorOperations<T>.Add(selfContribution, biasNode);

        // Apply activation if supported
        if (ScalarActivation != null && ScalarActivation.SupportsJitCompilation)
        {
            return ScalarActivation.ApplyToGraph(output);
        }

        return output;
    }
}
