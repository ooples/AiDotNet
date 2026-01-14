using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

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
    private Tensor<T> _edgeNetworkWeights1;
    private Tensor<T> _edgeNetworkWeights2;
    private Tensor<T> _edgeNetworkBias1;
    private Tensor<T> _edgeNetworkBias2;

    /// <summary>
    /// Self-loop transformation weights.
    /// </summary>
    private Tensor<T> _selfWeights;

    /// <summary>
    /// Bias vector.
    /// </summary>
    private Tensor<T> _bias;

    /// <summary>
    /// The adjacency matrix defining graph structure.
    /// </summary>
    private Tensor<T>? _adjacencyMatrix;

    /// <summary>
    /// Edge features tensor.
    /// </summary>
    private Tensor<T>? _edgeFeatures;
    private Tensor<T>? _normalizedAdjacencyMatrix;
    private Tensor<T>? _normalizedEdgeFeatures;

    /// <summary>
    /// Cached values for backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastEdgeWeights;
    private Tensor<T>? _lastHidden;

    /// <summary>
    /// Gradients.
    /// </summary>
    private Tensor<T>? _edgeNetworkWeights1Gradient;
    private Tensor<T>? _edgeNetworkWeights2Gradient;
    private Tensor<T>? _edgeNetworkBias1Gradient;
    private Tensor<T>? _edgeNetworkBias2Gradient;
    private Tensor<T>? _selfWeightsGradient;
    private Tensor<T>? _biasGradient;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    protected override bool SupportsGpuExecution => true;

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
        _edgeNetworkWeights1 = new Tensor<T>([edgeFeatures, edgeNetworkHiddenDim]);
        _edgeNetworkWeights2 = new Tensor<T>([edgeNetworkHiddenDim, inputFeatures * outputFeatures]);
        _edgeNetworkBias1 = new Tensor<T>([edgeNetworkHiddenDim]);
        _edgeNetworkBias2 = new Tensor<T>([inputFeatures * outputFeatures]);

        // Self-loop weights
        _selfWeights = new Tensor<T>([inputFeatures, outputFeatures]);

        // Bias
        _bias = new Tensor<T>([outputFeatures]);

        InitializeParameters();

        // Register trainable parameters for GPU memory persistence
        RegisterTrainableParameter(_edgeNetworkWeights1, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_edgeNetworkWeights2, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_edgeNetworkBias1, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_edgeNetworkBias2, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_selfWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bias, PersistentTensorRole.Biases);
    }

    private void InitializeParameters()
    {
        // Xavier initialization for edge network
        T scale1 = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_edgeFeaturesCount + _edgeNetworkHiddenDim)));
        InitializeTensor(_edgeNetworkWeights1, scale1);

        T scale2 = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_edgeNetworkHiddenDim + _inputFeatures * _outputFeatures)));
        InitializeTensor(_edgeNetworkWeights2, scale2);

        // Initialize self-loop weights
        T scaleSelf = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_inputFeatures + _outputFeatures)));
        InitializeTensor(_selfWeights, scaleSelf);

        // Initialize biases to zero
        _edgeNetworkBias1.Fill(NumOps.Zero);
        _edgeNetworkBias2.Fill(NumOps.Zero);
        _bias.Fill(NumOps.Zero);
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
    /// Sets the edge features for this layer.
    /// </summary>
    /// <param name="edgeFeatures">Edge features tensor with shape [batch, numEdges, edgeFeatureDim].</param>
    public void SetEdgeFeatures(Tensor<T> edgeFeatures)
    {
        _edgeFeatures = edgeFeatures;
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
        _normalizedAdjacencyMatrix = NormalizeAdjacency(_adjacencyMatrix, batchSize, numNodes);
        _normalizedEdgeFeatures = NormalizeEdgeFeatures(_edgeFeatures, batchSize);
        int numEdges = _normalizedEdgeFeatures.Shape[1];

        // Step 1: Compute edge-specific weights using edge network
        // Edge network layer 1: [batch, numEdges, edgeFeatures] @ [edgeFeatures, hiddenDim] = [batch, numEdges, hiddenDim]
        var hidden = BatchedMatMul3Dx2D(_normalizedEdgeFeatures, _edgeNetworkWeights1, batchSize, numEdges, _edgeFeaturesCount, _edgeNetworkHiddenDim);

        // Add bias: broadcast [hiddenDim] to [batch, numEdges, hiddenDim]
        var bias1Broadcast = BroadcastBias(_edgeNetworkBias1, batchSize, numEdges);
        hidden = Engine.TensorAdd(hidden, bias1Broadcast);

        // Apply ReLU activation
        hidden = ApplyReLU(hidden);
        _lastHidden = hidden;

        // Edge network layer 2: [batch, numEdges, hiddenDim] @ [hiddenDim, inF*outF] = [batch, numEdges, inF*outF]
        var flatWeights = BatchedMatMul3Dx2D(hidden, _edgeNetworkWeights2, batchSize, numEdges, _edgeNetworkHiddenDim, _inputFeatures * _outputFeatures);

        // Add bias: broadcast [inF*outF] to [batch, numEdges, inF*outF]
        var bias2Broadcast = BroadcastBias(_edgeNetworkBias2, batchSize, numEdges);
        flatWeights = Engine.TensorAdd(flatWeights, bias2Broadcast);

        // Reshape to [batch, numEdges, inputFeatures, outputFeatures]
        // Store as [batch, numNodes, numNodes, inputFeatures, outputFeatures] for backward compatibility
        _lastEdgeWeights = new Tensor<T>([batchSize, numNodes, numNodes, _inputFeatures, _outputFeatures]);

        // Map flat edge weights to node pairs
        for (int b = 0; b < batchSize; b++)
        {
            int edgeIdx = 0;
            for (int i = 0; i < numNodes; i++)
            {
                for (int j = 0; j < numNodes; j++)
                {
                    if (!NumOps.Equals(_normalizedAdjacencyMatrix[b, i, j], NumOps.Zero))
                    {
                        for (int inF = 0; inF < _inputFeatures; inF++)
                        {
                            for (int outF = 0; outF < _outputFeatures; outF++)
                            {
                                int flatIdx = inF * _outputFeatures + outF;
                                _lastEdgeWeights[b, i, j, inF, outF] = flatWeights[b, edgeIdx, flatIdx];
                            }
                        }
                        edgeIdx++;
                    }
                }
            }
        }

        // Step 2: Aggregate neighbor features using edge-specific weights
        var output = new Tensor<T>([batchSize, numNodes, _outputFeatures]);
        output.Fill(NumOps.Zero);

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
                        if (NumOps.Equals(_normalizedAdjacencyMatrix[b, i, j], NumOps.Zero))
                            continue;

                        // Apply edge-specific transformation to neighbor features
                        for (int inF = 0; inF < _inputFeatures; inF++)
                        {
                            sum = NumOps.Add(sum,
                                NumOps.Multiply(
                                    _lastEdgeWeights[b, i, j, inF, outF],
                                    processInput[b, j, inF]));
                        }
                    }

                    output[b, i, outF] = sum;
                }
            }
        }

        // Step 3: Add self-loop transformation
        var selfTransform = BatchedMatMul3Dx2D(processInput, _selfWeights, batchSize, numNodes, _inputFeatures, _outputFeatures);
        output = Engine.TensorAdd(output, selfTransform);

        // Step 4: Add bias
        var biasBroadcast = BroadcastBias(_bias, batchSize, numNodes);
        output = Engine.TensorAdd(output, biasBroadcast);

        var result = ApplyActivation(output);

        // Only store for backward pass during training - skip during inference
        if (IsTrainingMode)
        {
            _lastOutput = result;
        }

        if (_originalInputShape != null && _originalInputShape.Length != 3)
        {
            if (_originalInputShape.Length == 2)
            {
                return result.Reshape([numNodes, _outputFeatures]);
            }

            if (_originalInputShape.Length == 1)
            {
                return result.Reshape([_outputFeatures]);
            }

            var newShape = new int[_originalInputShape.Length];
            for (int d = 0; d < _originalInputShape.Length - 1; d++)
            {
                newShape[d] = _originalInputShape[d];
            }
            newShape[_originalInputShape.Length - 1] = _outputFeatures;
            return result.Reshape(newShape);
        }

        return result;
    }

    /// <inheritdoc/>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine tensorEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine");

        var backend = tensorEngine.GetBackend();
        if (backend == null)
            throw new InvalidOperationException("GPU backend unavailable");

        if (_adjacencyMatrix == null)
            throw new InvalidOperationException("Adjacency matrix must be set using SetAdjacencyMatrix before calling ForwardGpu.");

        if (_edgeFeatures == null)
            throw new InvalidOperationException("Edge features must be set using SetEdgeFeatures before calling ForwardGpu.");

        var input = inputs[0];

        // Get input dimensions - expect [batchSize, numNodes, inputFeatures]
        int[] inputShape = input.Shape;
        int batchSize = inputShape.Length >= 3 ? inputShape[0] : 1;
        int numNodes = inputShape.Length >= 3 ? inputShape[1] : inputShape[0];
        int inputFeatures = inputShape.Length >= 3 ? inputShape[2] : inputShape[1];

        // Normalize adjacency and edge features on CPU (one-time preprocessing)
        var normalizedAdj = NormalizeAdjacency(_adjacencyMatrix, batchSize, numNodes);
        var normalizedEdge = NormalizeEdgeFeatures(_edgeFeatures, batchSize);
        int numEdges = normalizedEdge.Shape[1];

        // Upload edge network weights and biases
        using var w1Buffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_edgeNetworkWeights1.Data.ToArray()));
        using var w2Buffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_edgeNetworkWeights2.Data.ToArray()));
        using var b1Buffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_edgeNetworkBias1.Data.ToArray()));
        using var b2Buffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_edgeNetworkBias2.Data.ToArray()));
        using var selfBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_selfWeights.Data.ToArray()));
        using var biasBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_bias.Data.ToArray()));

        // Upload edge features: [batch, numEdges, edgeFeatures]
        using var edgeBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(normalizedEdge.Data.ToArray()));

        // ===== Edge Network Layer 1: [batch*numEdges, edgeFeatures] @ [edgeFeatures, hiddenDim] =====
        // Uses fused GemmBiasRelu for optimal GPU performance
        int M1 = batchSize * numEdges;
        int K1 = _edgeFeaturesCount;
        int N1 = _edgeNetworkHiddenDim;
        using var hidden1Buffer = backend.GemmBiasRelu(edgeBuffer, w1Buffer, b1Buffer, M1, N1, K1);

        // ===== Edge Network Layer 2: [batch*numEdges, hiddenDim] @ [hiddenDim, inF*outF] =====
        int M2 = batchSize * numEdges;
        int K2 = _edgeNetworkHiddenDim;
        int N2 = _inputFeatures * _outputFeatures;
        using var flatWeightsBuffer = backend.AllocateBuffer(M2 * N2);

        // GEMM + bias broadcast (no activation) - use BiasAdd for efficient broadcast
        backend.Gemm(hidden1Buffer, w2Buffer, flatWeightsBuffer, M2, N2, K2);
        backend.BiasAdd(flatWeightsBuffer, b2Buffer, flatWeightsBuffer, M2, N2);

        // ===== Edge-Conditioned Aggregation: h_i' = Σ_{j∈N(i)} θ(e_ij) · h_j =====
        // GPU-native implementation using Gather, BatchedGemm, and ScatterAdd

        // Step 1: Pre-compute edge source/target indices from normalized adjacency (one-time preprocessing)
        var (sourceIndices, targetIndices) = BuildEdgeIndicesFromAdjacency(normalizedAdj, batchSize, numNodes, numEdges);

        // Upload edge indices to GPU (use AllocateIntBuffer for int arrays)
        using var sourceIdxBuffer = backend.AllocateIntBuffer(sourceIndices);
        using var targetIdxBuffer = backend.AllocateIntBuffer(targetIndices);

        // Step 2: Gather neighbor features for all edges: [batchSize * numEdges, inputFeatures]
        // For each edge e at batch b, get input[b, sourceIdx[e], :]
        using var neighborFeaturesBuffer = backend.AllocateBuffer(batchSize * numEdges * _inputFeatures);

        // Gather with batch handling: each batch has numEdges edges
        for (int b = 0; b < batchSize; b++)
        {
            int inputBatchOffset = b * numNodes * _inputFeatures;
            int edgeBatchOffset = b * numEdges;
            int outputBatchOffset = b * numEdges * _inputFeatures;

            // Gather neighbor features for this batch's edges
            // sourceIndices[edgeBatchOffset...edgeBatchOffset+numEdges] contains source node indices
            // We need a custom gather that handles the batch offset
            var batchSourceIndices = new int[numEdges];
            for (int e = 0; e < numEdges; e++)
            {
                // Compute absolute index into flattened input
                int relativeNodeIdx = sourceIndices[edgeBatchOffset + e];
                batchSourceIndices[e] = inputBatchOffset / _inputFeatures + relativeNodeIdx;
            }
            using var batchSourceIdxBuffer = backend.AllocateIntBuffer(batchSourceIndices);

            // Create a view for this batch's output
            var batchNeighborData = new float[numEdges * _inputFeatures];
            using var batchNeighborBuffer = backend.AllocateBuffer(batchNeighborData);

            // Gather: output[e, :] = input[batchSourceIndices[e], :]
            backend.Gather(input.Buffer, batchSourceIdxBuffer, batchNeighborBuffer, numEdges, _inputFeatures);

            // Copy to the correct batch position in neighborFeaturesBuffer
            backend.Copy2DStrided(batchNeighborBuffer, neighborFeaturesBuffer, 1, numEdges * _inputFeatures,
                batchSize * numEdges * _inputFeatures, outputBatchOffset);
        }

        // Step 3: Apply per-edge transformations using BatchedGemm
        // For each edge e, compute: transformed[e] = neighborFeatures[e] @ edgeWeights[e]
        // where edgeWeights[e] is reshaped from flat [inF*outF] to [inF, outF]
        //
        // Memory layout analysis for BatchedGemm(A, B, C, M, N, K, batchCount):
        // - A: [batchCount, M, K] with stride M*K per batch -> [numEdges, 1, inF], stride inF
        // - B: [batchCount, K, N] with stride K*N per batch -> [numEdges, inF, outF], stride inF*outF
        // - C: [batchCount, M, N] with stride M*N per batch -> [numEdges, 1, outF], stride outF
        //
        // Our buffers:
        // - neighborFeaturesBuffer: [batchSize*numEdges, inF] = contiguous rows of inF elements ✓
        // - flatWeightsBuffer: [batchSize*numEdges, inF*outF] = contiguous rows of inF*outF elements ✓
        // - transformedBuffer: [batchSize*numEdges, outF] = contiguous rows of outF elements ✓
        //
        // The strides match exactly, so BatchedGemm works correctly with our layout.
        using var transformedBuffer = backend.AllocateBuffer(batchSize * numEdges * _outputFeatures);
        backend.BatchedGemm(neighborFeaturesBuffer, flatWeightsBuffer, transformedBuffer,
            1, _outputFeatures, _inputFeatures, batchSize * numEdges);

        // Step 4: Scatter-add transformed features to target nodes
        // Download transformed buffer once, scatter on CPU per batch, upload aggregated result once
        // This is O(1) roundtrips vs O(numNodes * numEdges) in the original implementation
        var transformedData = backend.DownloadBuffer(transformedBuffer);
        var aggregatedOutput = new float[batchSize * numNodes * _outputFeatures];

        for (int b = 0; b < batchSize; b++)
        {
            int edgeBatchOffset = b * numEdges;
            int outputBatchOffset = b * numNodes * _outputFeatures;
            int transformedBatchOffset = b * numEdges * _outputFeatures;

            // Scatter-add: for each edge, accumulate transformed features to target node
            for (int e = 0; e < numEdges; e++)
            {
                int targetNode = targetIndices[edgeBatchOffset + e];
                for (int f = 0; f < _outputFeatures; f++)
                {
                    int srcIdx = transformedBatchOffset + e * _outputFeatures + f;
                    int dstIdx = outputBatchOffset + targetNode * _outputFeatures + f;
                    aggregatedOutput[dstIdx] += transformedData[srcIdx];
                }
            }
        }

        using var aggBuffer = backend.AllocateBuffer(aggregatedOutput);

        // ===== Self-Loop Transformation: input @ selfWeights =====
        // Use input buffer directly (already on GPU) for GEMM
        int MSelf = batchSize * numNodes;
        using var selfOutputBuffer = backend.AllocateBuffer(MSelf * _outputFeatures);
        backend.Gemm(input.Buffer, selfBuffer, selfOutputBuffer, MSelf, _outputFeatures, _inputFeatures);

        // ===== Combine: aggregated + selfLoop =====
        int totalOutput = batchSize * numNodes * _outputFeatures;
        using var combinedBuffer = backend.AllocateBuffer(totalOutput);
        backend.Add(aggBuffer, selfOutputBuffer, combinedBuffer, totalOutput);

        // Add bias using BiasAdd (efficient GPU broadcast) - treats output as [batch*numNodes, outputFeatures]
        backend.BiasAdd(combinedBuffer, biasBuffer, combinedBuffer, MSelf, _outputFeatures);

        // ===== Apply Activation =====
        var finalBuffer = backend.AllocateBuffer(totalOutput);
        ApplyGpuActivation(backend, combinedBuffer, finalBuffer, totalOutput, GetFusedActivationType());

        return new GpuTensor<T>(backend, finalBuffer, [batchSize, numNodes, _outputFeatures], GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// Computes edge-conditioned aggregation on CPU due to edge-specific indexing.
    /// </summary>
    private float[] ComputeEdgeConditionedAggregationCpu(
        float[] input, float[] flatEdgeWeights, Tensor<T> normalizedAdj,
        int batchSize, int numNodes, int numEdges, int inF, int outF)
    {
        var output = new float[batchSize * numNodes * outF];

        for (int b = 0; b < batchSize; b++)
        {
            int edgeIdx = 0;
            for (int i = 0; i < numNodes; i++)
            {
                for (int j = 0; j < numNodes; j++)
                {
                    if (NumOps.Equals(normalizedAdj[b, i, j], NumOps.Zero))
                        continue;

                    // Get edge-specific transformation θ(e_ij) from flatEdgeWeights
                    int edgeOffset = (b * numEdges + edgeIdx) * (inF * outF);

                    // Apply transformation: θ(e_ij) · h_j and accumulate to h_i
                    for (int oF = 0; oF < outF; oF++)
                    {
                        float sum = 0f;
                        for (int iF = 0; iF < inF; iF++)
                        {
                            // Edge weight at [inF, outF]
                            float edgeWeight = flatEdgeWeights[edgeOffset + iF * outF + oF];
                            // Neighbor feature h_j[iF]
                            float neighborFeature = input[(b * numNodes + j) * inF + iF];
                            sum += edgeWeight * neighborFeature;
                        }
                        output[(b * numNodes + i) * outF + oF] += sum;
                    }

                    edgeIdx++;
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Builds source and target node indices from adjacency matrix for GPU edge processing.
    /// Returns (sourceIndices, targetIndices) where each index array has length batchSize * numEdges.
    /// </summary>
    /// <param name="normalizedAdj">Normalized adjacency matrix [batchSize, numNodes, numNodes].</param>
    /// <param name="batchSize">Batch size.</param>
    /// <param name="numNodes">Number of nodes.</param>
    /// <param name="numEdges">Expected number of edges per batch.</param>
    /// <returns>Tuple of (sourceIndices, targetIndices) as int arrays.</returns>
    private (int[] sourceIndices, int[] targetIndices) BuildEdgeIndicesFromAdjacency(
        Tensor<T> normalizedAdj, int batchSize, int numNodes, int numEdges)
    {
        var sourceIndices = new int[batchSize * numEdges];
        var targetIndices = new int[batchSize * numEdges];

        for (int b = 0; b < batchSize; b++)
        {
            int edgeIdx = 0;
            for (int i = 0; i < numNodes; i++)
            {
                for (int j = 0; j < numNodes; j++)
                {
                    if (NumOps.Equals(normalizedAdj[b, i, j], NumOps.Zero))
                        continue;

                    int globalEdgeIdx = b * numEdges + edgeIdx;
                    sourceIndices[globalEdgeIdx] = j;  // Source node (neighbor)
                    targetIndices[globalEdgeIdx] = i;  // Target node (current)
                    edgeIdx++;
                }
            }

            // Verify edge count matches expected
            if (edgeIdx != numEdges)
            {
                throw new InvalidOperationException(
                    $"Edge count mismatch at batch {b}: expected {numEdges}, found {edgeIdx}");
            }
        }

        return (sourceIndices, targetIndices);
    }

    /// <summary>
    /// Normalizes adjacency matrix by degree (row normalization) and ensures batch dimension.
    /// </summary>
    private Tensor<T> NormalizeAdjacency(Tensor<T> adjacency, int batchSize, int numNodes)
    {
        bool is2D = adjacency.Shape.Length == 2;
        Tensor<T> adj3D;
        if (is2D)
        {
            adj3D = adjacency.Reshape([1, adjacency.Shape[0], adjacency.Shape[1]]);
            if (batchSize > 1)
            {
                var tiled = new Tensor<T>([batchSize, adjacency.Shape[0], adjacency.Shape[1]]);
                for (int b = 0; b < batchSize; b++)
                {
                    for (int i = 0; i < adjacency.Shape[0]; i++)
                    {
                        for (int j = 0; j < adjacency.Shape[1]; j++)
                        {
                            tiled[b, i, j] = adjacency[i, j];
                        }
                    }
                }
                adj3D = tiled;
            }
        }
        else
        {
            adj3D = adjacency;
        }

        var normalized = new Tensor<T>(adj3D.Shape);
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                int degree = 0;
                for (int j = 0; j < numNodes; j++)
                {
                    if (!NumOps.Equals(adj3D[b, i, j], NumOps.Zero))
                    {
                        degree++;
                    }
                }

                if (degree == 0)
                {
                    for (int j = 0; j < numNodes; j++)
                    {
                        normalized[b, i, j] = NumOps.Zero;
                    }
                }
                else
                {
                    T scale = NumOps.Divide(NumOps.One, NumOps.FromDouble(degree));
                    for (int j = 0; j < numNodes; j++)
                    {
                        normalized[b, i, j] = NumOps.Multiply(adj3D[b, i, j], scale);
                    }
                }
            }
        }

        return normalized;
    }

    /// <summary>
    /// Normalizes edge features to include batch dimension.
    /// </summary>
    private Tensor<T> NormalizeEdgeFeatures(Tensor<T> edgeFeatures, int batchSize)
    {
        if (edgeFeatures.Shape.Length == 3)
        {
            if (edgeFeatures.Shape[0] == batchSize)
            {
                return edgeFeatures;
            }

            if (edgeFeatures.Shape[0] == 1 && batchSize > 1)
            {
                var tiled = new Tensor<T>([batchSize, edgeFeatures.Shape[1], edgeFeatures.Shape[2]]);
                for (int b = 0; b < batchSize; b++)
                {
                    for (int i = 0; i < edgeFeatures.Shape[1]; i++)
                    {
                        for (int j = 0; j < edgeFeatures.Shape[2]; j++)
                        {
                            tiled[b, i, j] = edgeFeatures[0, i, j];
                        }
                    }
                }
                return tiled;
            }

            return edgeFeatures;
        }

        if (edgeFeatures.Shape.Length == 2)
        {
            var reshaped = edgeFeatures.Reshape([1, edgeFeatures.Shape[0], edgeFeatures.Shape[1]]);
            if (batchSize == 1)
            {
                return reshaped;
            }

            var tiled = new Tensor<T>([batchSize, edgeFeatures.Shape[0], edgeFeatures.Shape[1]]);
            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < edgeFeatures.Shape[0]; i++)
                {
                    for (int j = 0; j < edgeFeatures.Shape[1]; j++)
                    {
                        tiled[b, i, j] = edgeFeatures[i, j];
                    }
                }
            }
            return tiled;
        }

        throw new ArgumentException("Edge features must be rank 2 or 3.");
    }

    /// <summary>
    /// Applies ReLU activation element-wise to a tensor.
    /// </summary>
    private Tensor<T> ApplyReLU(Tensor<T> input)
    {
        var output = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            T val = input.GetFlat(i);
            output[i] = NumOps.GreaterThan(val, NumOps.Zero) ? val : NumOps.Zero;
        }
        return output;
    }

    /// <summary>
    /// Broadcasts a bias tensor across batch and node dimensions.
    /// </summary>
    private Tensor<T> BroadcastBias(Tensor<T> bias, int batchSize, int numNodes)
    {
        int outputFeatures = bias.Length;

        // Reshape bias from [outputFeatures] to [1, 1, outputFeatures]
        var biasReshaped = bias.Reshape([1, 1, outputFeatures]);

        // Tile across batch and node dimensions: [batchSize, numNodes, outputFeatures]
        var broadcast = Engine.TensorTile(biasReshaped, [batchSize, numNodes, 1]);

        return broadcast;
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
    /// Computes the backward pass for this edge-conditional layer.
    /// </summary>
    /// <param name="outputGradient">The gradient from the next layer.</param>
    /// <returns>The gradient to propagate to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// This backward pass computes gradients for all parameters including edge network weights,
    /// self-weights, biases, and propagates gradients to the input.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _normalizedAdjacencyMatrix == null || _normalizedEdgeFeatures == null || _lastEdgeWeights == null || _lastHidden == null)
        {
            throw new InvalidOperationException("Forward pass must be called before Backward.");
        }

        var outputGradient3D = NormalizeOutputGradient(outputGradient);
        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient3D);
        int batchSize = _lastInput.Shape[0];
        int numNodes = _lastInput.Shape[1];
        int numEdges = _normalizedEdgeFeatures.Shape[1];

        // Initialize gradients
        _edgeNetworkWeights1Gradient = new Tensor<T>([_edgeFeaturesCount, _edgeNetworkHiddenDim]);
        _edgeNetworkWeights2Gradient = new Tensor<T>([_edgeNetworkHiddenDim, _inputFeatures * _outputFeatures]);
        _edgeNetworkBias1Gradient = new Tensor<T>([_edgeNetworkHiddenDim]);
        _edgeNetworkBias2Gradient = new Tensor<T>([_inputFeatures * _outputFeatures]);
        _selfWeightsGradient = new Tensor<T>([_inputFeatures, _outputFeatures]);
        _biasGradient = new Tensor<T>([_outputFeatures]);

        _edgeNetworkWeights1Gradient.Fill(NumOps.Zero);
        _edgeNetworkWeights2Gradient.Fill(NumOps.Zero);
        _edgeNetworkBias1Gradient.Fill(NumOps.Zero);
        _edgeNetworkBias2Gradient.Fill(NumOps.Zero);
        _selfWeightsGradient.Fill(NumOps.Zero);
        _biasGradient.Fill(NumOps.Zero);

        var inputGradient = new Tensor<T>(_lastInput.Shape);
        inputGradient.Fill(NumOps.Zero);

        // Step 1: Bias gradient (sum over batch and nodes)
        _biasGradient = Engine.ReduceSum(activationGradient, [0, 1], keepDims: false);

        // Step 2: Self-weights gradient
        // dL/dSelfWeights = sum_b (input[b]^T @ activationGradient[b])
        for (int b = 0; b < batchSize; b++)
        {
            for (int n = 0; n < numNodes; n++)
            {
                for (int inF = 0; inF < _inputFeatures; inF++)
                {
                    for (int outF = 0; outF < _outputFeatures; outF++)
                    {
                        T grad = NumOps.Multiply(_lastInput[b, n, inF], activationGradient[b, n, outF]);
                        _selfWeightsGradient[inF, outF] = NumOps.Add(_selfWeightsGradient[inF, outF], grad);
                    }
                }
            }
        }

        // Step 3: Input gradient from self-loop
        // dL/dInput += activationGradient @ selfWeights^T
        var selfWeightsT = Engine.TensorTranspose(_selfWeights);
        var inputGradFromSelf = BatchedMatMul3Dx2D(activationGradient, selfWeightsT, batchSize, numNodes, _outputFeatures, _inputFeatures);
        inputGradient = Engine.TensorAdd(inputGradient, inputGradFromSelf);

        // Step 4: Gradients through edge-conditioned aggregation
        var edgeWeightsGradient = new Tensor<T>([batchSize, numNodes, numNodes, _inputFeatures, _outputFeatures]);
        edgeWeightsGradient.Fill(NumOps.Zero);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                for (int outF = 0; outF < _outputFeatures; outF++)
                {
                    T gradOut = activationGradient[b, i, outF];

                    for (int j = 0; j < numNodes; j++)
                    {
                        if (NumOps.Equals(_normalizedAdjacencyMatrix[b, i, j], NumOps.Zero))
                            continue;

                        for (int inF = 0; inF < _inputFeatures; inF++)
                        {
                            // Gradient w.r.t. edge weights
                            T inputVal = _lastInput[b, j, inF];
                            edgeWeightsGradient[b, i, j, inF, outF] = NumOps.Multiply(gradOut, inputVal);

                            // Gradient w.r.t. input
                            T edgeWeight = _lastEdgeWeights[b, i, j, inF, outF];
                            inputGradient[b, j, inF] = NumOps.Add(inputGradient[b, j, inF],
                                NumOps.Multiply(gradOut, edgeWeight));
                        }
                    }
                }
            }
        }

        // Step 5: Backpropagate through edge network
        // Map edge weights gradient back to flat format
        var flatWeightsGrad = new Tensor<T>([batchSize, numEdges, _inputFeatures * _outputFeatures]);
        flatWeightsGrad.Fill(NumOps.Zero);

        for (int b = 0; b < batchSize; b++)
        {
            int edgeIdx = 0;
            for (int i = 0; i < numNodes; i++)
            {
                for (int j = 0; j < numNodes; j++)
                {
                    if (!NumOps.Equals(_normalizedAdjacencyMatrix[b, i, j], NumOps.Zero))
                    {
                        for (int inF = 0; inF < _inputFeatures; inF++)
                        {
                            for (int outF = 0; outF < _outputFeatures; outF++)
                            {
                                int flatIdx = inF * _outputFeatures + outF;
                                flatWeightsGrad[b, edgeIdx, flatIdx] = edgeWeightsGradient[b, i, j, inF, outF];
                            }
                        }
                        edgeIdx++;
                    }
                }
            }
        }

        // Gradient w.r.t. edge network bias 2
        _edgeNetworkBias2Gradient = Engine.ReduceSum(flatWeightsGrad, [0, 1], keepDims: false);

        // Gradient w.r.t. edge network weights 2
        // dL/dW2 = hidden^T @ flatWeightsGrad
        var hiddenT = _lastHidden.Transpose([0, 2, 1]);
        for (int b = 0; b < batchSize; b++)
        {
            var hiddenBatchT = Engine.TensorSlice(hiddenT, [b, 0, 0], [1, _edgeNetworkHiddenDim, numEdges]).Reshape([_edgeNetworkHiddenDim, numEdges]);
            var flatGradBatch = Engine.TensorSlice(flatWeightsGrad, [b, 0, 0], [1, numEdges, _inputFeatures * _outputFeatures]).Reshape([numEdges, _inputFeatures * _outputFeatures]);
            var w2Grad = Engine.TensorMatMul(hiddenBatchT, flatGradBatch);
            _edgeNetworkWeights2Gradient = Engine.TensorAdd(_edgeNetworkWeights2Gradient, w2Grad);
        }

        // Gradient w.r.t. hidden layer
        var weights2T = Engine.TensorTranspose(_edgeNetworkWeights2);
        var hiddenGrad = BatchedMatMul3Dx2D(flatWeightsGrad, weights2T, batchSize, numEdges, _inputFeatures * _outputFeatures, _edgeNetworkHiddenDim);

        // Apply ReLU derivative
        var reluGrad = new Tensor<T>(hiddenGrad.Shape);
        for (int i = 0; i < hiddenGrad.Length; i++)
        {
            T hiddenVal = _lastHidden.GetFlat(i);
            T gradVal = hiddenGrad.GetFlat(i);
            reluGrad[i] = NumOps.GreaterThan(hiddenVal, NumOps.Zero) ? gradVal : NumOps.Zero;
        }

        // Gradient w.r.t. edge network bias 1
        _edgeNetworkBias1Gradient = Engine.ReduceSum(reluGrad, [0, 1], keepDims: false);

        // Gradient w.r.t. edge network weights 1
        // dL/dW1 = edgeFeatures^T @ reluGrad
        var edgeFeaturesT = _normalizedEdgeFeatures.Transpose([0, 2, 1]);
        for (int b = 0; b < batchSize; b++)
        {
            var edgesBatchT = Engine.TensorSlice(edgeFeaturesT, [b, 0, 0], [1, _edgeFeaturesCount, numEdges]).Reshape([_edgeFeaturesCount, numEdges]);
            var reluGradBatch = Engine.TensorSlice(reluGrad, [b, 0, 0], [1, numEdges, _edgeNetworkHiddenDim]).Reshape([numEdges, _edgeNetworkHiddenDim]);
            var w1Grad = Engine.TensorMatMul(edgesBatchT, reluGradBatch);
            _edgeNetworkWeights1Gradient = Engine.TensorAdd(_edgeNetworkWeights1Gradient, w1Grad);
        }

        if (_originalInputShape != null && _originalInputShape.Length != inputGradient.Shape.Length)
        {
            if (_originalInputShape.Length == 2)
            {
                return inputGradient.Reshape([numNodes, _inputFeatures]);
            }

            if (_originalInputShape.Length == 1)
            {
                return inputGradient.Reshape([_inputFeatures]);
            }

            return inputGradient.Reshape(_originalInputShape);
        }

        return inputGradient;
    }

    private Tensor<T> NormalizeOutputGradient(Tensor<T> outputGradient)
    {
        int rank = outputGradient.Shape.Length;
        if (rank == 3)
        {
            return outputGradient;
        }

        if (rank == 2)
        {
            return outputGradient.Reshape([1, outputGradient.Shape[0], outputGradient.Shape[1]]);
        }

        if (rank == 1)
        {
            return outputGradient.Reshape([1, 1, outputGradient.Shape[0]]);
        }

        int flatBatch = 1;
        for (int d = 0; d < rank - 2; d++)
        {
            flatBatch *= outputGradient.Shape[d];
        }

        int numNodes = outputGradient.Shape[rank - 2];
        int features = outputGradient.Shape[rank - 1];
        return outputGradient.Reshape([flatBatch, numNodes, features]);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        if (_edgeNetworkWeights1Gradient == null)
        {
            throw new InvalidOperationException("Backward must be called before UpdateParameters.");
        }

        // Use Engine operations for parameter updates
        var scaledW1Grad = Engine.TensorMultiplyScalar(_edgeNetworkWeights1Gradient, learningRate);
        _edgeNetworkWeights1 = Engine.TensorSubtract(_edgeNetworkWeights1, scaledW1Grad);

        var scaledW2Grad = Engine.TensorMultiplyScalar(_edgeNetworkWeights2Gradient!, learningRate);
        _edgeNetworkWeights2 = Engine.TensorSubtract(_edgeNetworkWeights2, scaledW2Grad);

        var scaledSelfGrad = Engine.TensorMultiplyScalar(_selfWeightsGradient!, learningRate);
        _selfWeights = Engine.TensorSubtract(_selfWeights, scaledSelfGrad);

        var scaledBias1Grad = Engine.TensorMultiplyScalar(_edgeNetworkBias1Gradient!, learningRate);
        _edgeNetworkBias1 = Engine.TensorSubtract(_edgeNetworkBias1, scaledBias1Grad);

        var scaledBias2Grad = Engine.TensorMultiplyScalar(_edgeNetworkBias2Gradient!, learningRate);
        _edgeNetworkBias2 = Engine.TensorSubtract(_edgeNetworkBias2, scaledBias2Grad);

        var scaledBiasGrad = Engine.TensorMultiplyScalar(_biasGradient!, learningRate);
        _bias = Engine.TensorSubtract(_bias, scaledBiasGrad);

        // Invalidate GPU cache after parameter updates
        Engine.InvalidatePersistentTensor(_edgeNetworkWeights1);
        Engine.InvalidatePersistentTensor(_edgeNetworkWeights2);
        Engine.InvalidatePersistentTensor(_edgeNetworkBias1);
        Engine.InvalidatePersistentTensor(_edgeNetworkBias2);
        Engine.InvalidatePersistentTensor(_selfWeights);
        Engine.InvalidatePersistentTensor(_bias);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        // Use Vector.Concatenate to efficiently combine all parameters
        return Vector<T>.Concatenate(
            new Vector<T>(_edgeNetworkWeights1.ToArray()),
            new Vector<T>(_edgeNetworkWeights2.ToArray()),
            new Vector<T>(_edgeNetworkBias1.ToArray()),
            new Vector<T>(_edgeNetworkBias2.ToArray()),
            new Vector<T>(_selfWeights.ToArray()),
            new Vector<T>(_bias.ToArray())
        );
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int w1Size = _edgeNetworkWeights1.Length;
        int w2Size = _edgeNetworkWeights2.Length;
        int b1Size = _edgeNetworkBias1.Length;
        int b2Size = _edgeNetworkBias2.Length;
        int selfSize = _selfWeights.Length;
        int biasSize = _bias.Length;
        int totalParams = w1Size + w2Size + b1Size + b2Size + selfSize + biasSize;

        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }

        int index = 0;

        // Set edge network weights 1
        var w1Params = parameters.SubVector(index, w1Size);
        _edgeNetworkWeights1 = Tensor<T>.FromVector(w1Params).Reshape(_edgeNetworkWeights1.Shape);
        index += w1Size;

        // Set edge network weights 2
        var w2Params = parameters.SubVector(index, w2Size);
        _edgeNetworkWeights2 = Tensor<T>.FromVector(w2Params).Reshape(_edgeNetworkWeights2.Shape);
        index += w2Size;

        // Set edge network bias 1
        var b1Params = parameters.SubVector(index, b1Size);
        _edgeNetworkBias1 = Tensor<T>.FromVector(b1Params);
        index += b1Size;

        // Set edge network bias 2
        var b2Params = parameters.SubVector(index, b2Size);
        _edgeNetworkBias2 = Tensor<T>.FromVector(b2Params);
        index += b2Size;

        // Set self weights
        var selfParams = parameters.SubVector(index, selfSize);
        _selfWeights = Tensor<T>.FromVector(selfParams).Reshape(_selfWeights.Shape);
        index += selfSize;

        // Set bias
        var biasParams = parameters.SubVector(index, biasSize);
        _bias = Tensor<T>.FromVector(biasParams);

        // Invalidate GPU cache after parameter updates
        Engine.InvalidatePersistentTensor(_edgeNetworkWeights1);
        Engine.InvalidatePersistentTensor(_edgeNetworkWeights2);
        Engine.InvalidatePersistentTensor(_edgeNetworkBias1);
        Engine.InvalidatePersistentTensor(_edgeNetworkBias2);
        Engine.InvalidatePersistentTensor(_selfWeights);
        Engine.InvalidatePersistentTensor(_bias);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastEdgeWeights = null;
        _lastHidden = null;
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
