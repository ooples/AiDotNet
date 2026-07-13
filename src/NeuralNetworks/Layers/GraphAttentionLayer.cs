using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

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
/// <para>
/// <b>Production-Ready Features:</b>
/// <list type="bullet">
/// <item>Fully vectorized operations using IEngine for GPU acceleration</item>
/// <item>Tensor-based weights for all parameters</item>
/// <item>Dual backward pass: BackwardManual() for efficiency, BackwardViaAutodiff() for accuracy</item>
/// <item>Full gradient computation through attention mechanism</item>
/// <item>Complete GetParameters()/SetParameters() for model persistence</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.Graph)]
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.GraphProcessing)]
[LayerTask(LayerTask.AttentionComputation)]
[LayerProperty(ApiShape = LayerApiShape.GraphWithSetup, IsTrainable = true, ChangesShape = true, Cost = ComputeCost.High, TestInputShape = "4, 8", TestConstructorArgs = "8, 4, 2", TestSetupCode = "var adj = new AiDotNet.Tensors.LinearAlgebra.Tensor<double>(new[] { 4, 4 }); for (int i = 0; i < 4; i++) { adj[i, i] = 1.0; if (i > 0) adj[i, i-1] = 1.0; if (i < 3) adj[i, i+1] = 1.0; } var m = layer.GetType().GetMethod(\"SetAdjacencyMatrix\"); if (m != null) m.Invoke(layer, new object[] { adj });")]
public partial class GraphAttentionLayer<T> : LayerBase<T>, IGraphConvolutionLayer<T>
{
    private readonly int _inputFeatures;
    private readonly int _outputFeatures;
    private readonly int _numHeads;
    private readonly T _alpha; // LeakyReLU negative slope
    private readonly double _dropoutRate;
    private readonly Random _random;

    /// <summary>
    /// Weight tensor for each attention head. Shape: [numHeads, inputFeatures, outputFeatures].
    /// </summary>
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _weights;

    /// <summary>
    /// Attention mechanism parameters tensor. Shape: [numHeads, 2 * outputFeatures].
    /// </summary>
    private Tensor<T> _attentionWeights;

    /// <summary>
    /// Bias tensor for the output transformation. Shape: [outputFeatures].
    /// </summary>
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _bias;

    /// <summary>
    /// The adjacency matrix defining graph structure.
    /// </summary>
    private Tensor<T>? _adjacencyMatrix;

    /// <summary>
    /// Helper to get adjacency value - supports both 2D [nodes, nodes] and 3D [batch, nodes, nodes].
    /// </summary>
    private T GetAdjacencyValue(int b, int i, int j)
    {
        if (_adjacencyMatrix == null)
            throw new InvalidOperationException("Adjacency matrix is not set.");
        return _adjacencyMatrix.Shape.Length == 3 ? _adjacencyMatrix[b, i, j] : _adjacencyMatrix[i, j];
    }

    /// <summary>
    /// Edge source node indices for sparse graph representation.
    /// </summary>
    private Tensor<int>? _edgeSourceIndices;

    /// <summary>
    /// Edge target node indices for sparse graph representation.
    /// </summary>
    private Tensor<int>? _edgeTargetIndices;

    /// <summary>
    /// Indicates whether to use sparse (edge-based) or dense (adjacency matrix) aggregation.
    /// </summary>
    private bool _useSparseAggregation = false;

    /// <summary>
    /// Cached input from forward pass for backward computation.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// Cached output from forward pass for backward computation.
    /// </summary>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Cached attention coefficients from forward pass.
    /// </summary>
    private Tensor<T>? _lastAttentionCoefficients;

    /// <summary>
    /// Cached pre-softmax attention scores for gradient computation.
    /// </summary>
    private Tensor<T>? _lastPreSoftmaxScores;



    /// <summary>
    /// Gradients for weight parameters.
    /// </summary>
    private Tensor<T>? _weightsGradient;

    /// <summary>
    /// Gradients for attention parameters.
    /// </summary>
    private Tensor<T>? _attentionWeightsGradient;

    /// <summary>
    /// Gradients for bias parameters.
    /// </summary>
    private Tensor<T>? _biasGradient;

    // GPU cache fields for backward pass
    private Tensor<T>? _gpuLastInput;
    private IGpuBuffer? _gpuTransformedCache;  // [numNodes * outputFeatures * numHeads]
    private IGpuBuffer? _gpuAttentionCache;    // [numNodes * numNodes * numHeads]
    private IGpuBuffer? _gpuPreActivationCache;  // [batchSize * numNodes * outputFeatures] - pre-activation output for backward
    private IGpuBuffer? _gpuPostActivationCache; // [batchSize * numNodes * outputFeatures] - post-activation output for backward
    private IGpuBuffer? _gpuPreLeakyReluCache;   // [batchSize * numHeads * numNodes * numNodes] - pre-LeakyReLU attention scores
    private int _gpuNumNodes;
    private int _gpuBatchSize;

    // GPU gradient fields
    private IGpuBuffer? _gpuWeightsGradient;
    private IGpuBuffer? _gpuAttentionWeightsGradient;
    private IGpuBuffer? _gpuBiasGradient;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets whether this layer supports GPU execution.
    /// </summary>
    /// <remarks>
    /// GraphAttentionLayer supports GPU execution with multi-head attention computed on GPU.
    /// When sparse aggregation is enabled via SetEdges(), the layer uses O(E) GPU operations
    /// for efficient attention computation on large graphs.
    /// </remarks>
    protected override bool SupportsGpuExecution => true;

    /// <inheritdoc/>
    public override long ParameterCount => _weights.Length + _attentionWeights.Length + _bias.Length;

    /// <inheritdoc/>
    public int InputFeatures => _inputFeatures;

    /// <inheritdoc/>
    public int OutputFeatures => _outputFeatures;

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphAttentionLayer{T}"/> class.
    /// </summary>
    public GraphAttentionLayer(
        int inputFeatures,
        int outputFeatures,
        int numHeads = 1,
        double alpha = 0.2,
        double dropoutRate = 0.0,
        IActivationFunction<T>? activationFunction = null,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base([inputFeatures], [outputFeatures], activationFunction ?? new IdentityActivation<T>())
    {
        InitializationStrategy = initializationStrategy ?? InitializationStrategies<T>.Eager;

        _inputFeatures = inputFeatures;
        _outputFeatures = outputFeatures;
        _numHeads = numHeads;
        _alpha = NumOps.FromDouble(alpha);
        _dropoutRate = dropoutRate;
        // Seed from the layer's RandomSeed (wired from architecture.RandomSeed via the
        // LayerInitializationSeedScope) so weight init AND the training-time dropout mask are
        // REPRODUCIBLE and order/platform-independent. The prior CreateSecureRandom() drew from a
        // non-deterministic source, so the attention weights (and dropout) differed every run —
        // which made training outcomes depend on the draw (some draws diverged to NaN) and
        // Clone/ScaledInput invariants pass on one machine but fail on another. Falls back to a
        // secure RNG only when no seed was requested (production default).
        _random = RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(RandomSeed.Value)
            : RandomHelper.CreateSecureRandom();

        // Initialize weights as Tensors for GPU acceleration
        _weights = new Tensor<T>([_numHeads, _inputFeatures, _outputFeatures]);
        _attentionWeights = new Tensor<T>([_numHeads, 2 * _outputFeatures]);
        _bias = new Tensor<T>([_outputFeatures]);

        InitializeParameters();

        // Register trainable parameters for GPU memory optimization
        RegisterTrainableParameter(_weights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_attentionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bias, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Initializes layer parameters using Xavier/Glorot initialization with Engine operations.
    /// </summary>
    private void InitializeParameters()
    {
        // Xavier initialization for weights
        InitializeTensor(_weights, _inputFeatures, _outputFeatures);

        // Initialize attention weights as (rand − 0.5)·scale, drawing the random tensor from the
        // SEEDED _random (CreateRandom(Random, …) — not the process-shared CreateRandom(shape)
        // overload that made this init order/platform-dependent). The vectorized Engine subtract +
        // scalar-multiply are kept as-is.
        T attentionScale = NumOps.Sqrt(NumOps.FromDouble(1.0 / _outputFeatures));
        var randomAttn = Tensor<T>.CreateRandom(_random, _attentionWeights._shape);
        var halfTensor = new Tensor<T>(_attentionWeights._shape);
        halfTensor.Fill(NumOps.FromDouble(0.5));
        var shiftedAttn = Engine.TensorSubtract(randomAttn, halfTensor);
        var scaledAttn = Engine.TensorMultiplyScalar(shiftedAttn, attentionScale);
        for (int i = 0; i < _attentionWeights.Length; i++)
        {
            _attentionWeights[i] = scaledAttn.GetFlat(i);
        }

        // Initialize bias to zero
        _bias.Fill(NumOps.Zero);
    }

    private void InitializeTensor(Tensor<T> tensor, int fanIn, int fanOut)
    {
        InitializeLayerWeights(tensor, fanIn, fanOut);
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
    /// Sets the edge list representation of the graph structure for sparse aggregation.
    /// </summary>
    /// <param name="sourceIndices">Tensor containing source node indices for each edge. Shape: [numEdges].</param>
    /// <param name="targetIndices">Tensor containing target node indices for each edge. Shape: [numEdges].</param>
    /// <remarks>
    /// <para>
    /// This method provides an edge-list representation of the graph, enabling memory-efficient
    /// sparse attention computation using the Engine's GraphAttention operations. This is the
    /// recommended approach for production GAT workloads with large sparse graphs.
    /// </para>
    /// </remarks>
    public void SetEdges(Tensor<int> sourceIndices, Tensor<int> targetIndices)
    {
        if (sourceIndices == null)
            throw new ArgumentNullException(nameof(sourceIndices));

        if (targetIndices == null)
            throw new ArgumentNullException(nameof(targetIndices));

        if (sourceIndices.Length != targetIndices.Length)
            throw new ArgumentException($"Source and target index tensors must have the same length. Got {sourceIndices.Length} and {targetIndices.Length}.");

        _edgeSourceIndices = sourceIndices;
        _edgeTargetIndices = targetIndices;
        _useSparseAggregation = true;
    }

    /// <summary>
    /// Gets whether sparse (edge-based) aggregation is currently enabled.
    /// </summary>
    public bool UsesSparseAggregation => _useSparseAggregation;

    /// <summary>
    /// Clears the edge list and switches back to dense adjacency matrix aggregation.
    /// </summary>
    public void ClearEdges()
    {
        _edgeSourceIndices = null;
        _edgeTargetIndices = null;
        _useSparseAggregation = false;
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Check that either adjacency matrix or edge indices are set
        if (_adjacencyMatrix == null && !_useSparseAggregation)
        {
            throw new InvalidOperationException(
                "Graph structure must be set using SetAdjacencyMatrix or SetEdges before calling Forward.");
        }

        // Store original shape for any-rank tensor support
        _originalInputShape = input._shape;
        int rank = input.Shape.Length;

        // Handle any-rank tensor: ensure 3D [batch, numNodes, inputFeatures]
        Tensor<T> processInput;
        int batchSize;
        int numNodes;

        if (rank == 1)
        {
            // [inputFeatures] -> [1, 1, inputFeatures]
            batchSize = 1;
            numNodes = 1;
            processInput = Engine.Reshape(input, [1, 1, input.Shape[0]]);
        }
        else if (rank == 2)
        {
            // [numNodes, inputFeatures] -> [1, numNodes, inputFeatures]
            batchSize = 1;
            numNodes = input.Shape[0];
            processInput = Engine.Reshape(input, [1, input.Shape[0], input.Shape[1]]);
        }
        else
        {
            // [batch, numNodes, inputFeatures] or higher-rank
            int flatBatch = 1;
            for (int d = 0; d < rank - 2; d++)
                flatBatch *= input.Shape[d];
            batchSize = flatBatch;
            numNodes = input.Shape[rank - 2];
            processInput = Engine.Reshape(input, [flatBatch, input.Shape[rank - 2], input.Shape[rank - 1]]);
        }

        // #1668: compute the backward-cache decision once up front so every backward-only cache
        // in both the sparse and dense paths (and _lastInput here) is gated/cleared consistently.
        // Clearing (not just skipping) is required: a prior step's arena-owned tensor must not
        // survive into the next denoise-loop Reset().
        bool cacheBwd = ShouldCacheForBackward;
        _lastInput = cacheBwd ? processInput : null;

        Tensor<T> output;
        Tensor<T> activatedOutput;

        if (_useSparseAggregation && _edgeSourceIndices != null && _edgeTargetIndices != null)
        {
            // Sparse aggregation using Engine.MultiHeadGraphAttention (production-recommended).
            // _attentionWeights is laid out as [numHeads, 2 * outputFeatures] with the
            // source-half occupying columns [0, outputFeatures) and the target-half
            // [outputFeatures, 2 * outputFeatures). Engine.TensorSlice returns each half
            // as a view-style operation in one call instead of the per-(h, f) copy loop.
            var attnWeightsSource = Engine.TensorSlice(_attentionWeights, [0, 0], [_numHeads, _outputFeatures]);
            var attnWeightsTarget = Engine.TensorSlice(_attentionWeights, [0, _outputFeatures], [_numHeads, _outputFeatures]);

            output = TensorAllocator.Rent<T>([batchSize, numNodes, _outputFeatures]);
            output.Fill(NumOps.Zero);

            for (int b = 0; b < batchSize; b++)
            {
                // Extract batch features using Slice: [numNodes, inputFeatures]
                var batchFeatures = processInput.Slice(b);

                // Call Engine.MultiHeadGraphAttention
                var batchOutput = Engine.MultiHeadGraphAttention(
                    batchFeatures,
                    _edgeSourceIndices,
                    _edgeTargetIndices,
                    _weights,
                    attnWeightsSource,
                    attnWeightsTarget,
                    NumOps.ToDouble(_alpha),
                    concatenate: false,  // Average heads
                    out var batchAttnCoeffs);

                // Add bias using broadcasting and set in output
                var biasBroadcast = Engine.Reshape(_bias, [1, _outputFeatures]);
                var biasedOutput = Engine.TensorBroadcastAdd(batchOutput, biasBroadcast);
                output.SetSlice(b, biasedOutput);
            }

            // Store cached values for backward pass
            _lastPreSoftmaxScores = null;
            _lastAttentionCoefficients = null;

            activatedOutput = ApplyActivation(output);

            // Reshape output to match original input rank
            Tensor<T> sparseReshaped;
            if (rank == 1)
            {
                sparseReshaped = Engine.Reshape(activatedOutput, [_outputFeatures]);
            }
            else if (rank == 2)
            {
                sparseReshaped = Engine.Reshape(activatedOutput, [numNodes, _outputFeatures]);
            }
            else
            {
                if (_originalInputShape == null)
                {
                    throw new InvalidOperationException("Original input shape was not captured.");
                }
                var originalShape = _originalInputShape;
                var outputShape = new int[rank];
                for (int d = 0; d < rank - 2; d++)
                    outputShape[d] = originalShape[d];
                outputShape[rank - 2] = numNodes;
                outputShape[rank - 1] = _outputFeatures;
                sparseReshaped = Engine.Reshape(activatedOutput, outputShape);
            }

            // #1668: _lastOutput is a backward-only cache; clear it in inference so the
            // arena-owned reshaped output is not retained across the next Reset().
            _lastOutput = cacheBwd ? sparseReshaped : null;
            return sparseReshaped;
        }

        // Dense aggregation path — FULLY ON-TAPE (autodiff-differentiable). The prior
        // implementation ran the per-head transform, attention scores, softmax and
        // aggregation through raw Tensor.GetCpuData()/SetFlat round-trips (ExtractHeadWeight,
        // Set3DSliceIn4DForHead, ComputeAttentionScores, Get2DSliceFrom4D). That disconnected
        // the entire per-head computation from the autodiff tape: _weights and _attentionWeights
        // received ZERO gradient (frozen attention that never learns) and the tape reading the
        // raw-written leaf tensors corrupted training to NaN over many iterations
        // (GraphAttentionNetwork MoreData / Clone). This version keeps the paper-faithful GAT
        // mechanism (Velickovic et al. 2018: e_ij = LeakyReLU(a_self·Wh_i + a_neigh·Wh_j);
        // softmax over neighbours; dropout on the normalized coefficients, §3.3) but expresses
        // every step with Engine ops that reference the ACTUAL parameter tensors, so autodiff
        // derives exact gradients for _weights, _attentionWeights and _bias.
        var adjacency = _adjacencyMatrix!;
        bool adjacency2D = adjacency.Shape.Length == 2;
        T maskNegInf = NumOps.FromDouble(-1e9);

        // These per-forward scratch caches are unused on the on-tape path (autodiff owns the
        // backward). Null them so ResetState / arena recycling see no stale references.
        _lastPreSoftmaxScores = null;
        _lastAttentionCoefficients = null;

        // The additive attention mask depends only on the batch element (adjacency[b]), not the
        // head, so build it once per batch element instead of recomputing the identical O(numNodes^2)
        // mask _numHeads times per element inside the (h, b) nest. It is a constant tensor (not a
        // tape parameter), so the same instance can be reused across heads.
        var perBatchMasks = new Tensor<T>[batchSize];
        for (int b = 0; b < batchSize; b++)
            perBatchMasks[b] = BuildAttentionMask(adjacency, adjacency2D, b, numNodes, maskNegInf);

        // Sum of the per-head aggregated outputs: [batchSize, numNodes, outputFeatures] (on-tape).
        Tensor<T>? denseHeadSum = null;
        for (int h = 0; h < _numHeads; h++)
        {
            // Head weight [inputFeatures, outputFeatures] sliced from the real _weights tensor
            // ([numHeads, inputFeatures, outputFeatures]) via Engine so the tape connects to it.
            var headWeight = Engine.Reshape(
                Engine.TensorSlice(_weights, [h, 0, 0], [1, _inputFeatures, _outputFeatures]),
                [_inputFeatures, _outputFeatures]);
            // Wh for every graph in the batch: [batchSize, numNodes, outputFeatures] (on-tape).
            var transformedHead = BatchedMatMul3Dx2D(processInput, headWeight, batchSize, numNodes, _inputFeatures, _outputFeatures);

            // Attention-weight halves for this head, [outputFeatures, 1] (on-tape).
            var attnSelf = Engine.Reshape(
                Engine.TensorSlice(_attentionWeights, [h, 0], [1, _outputFeatures]), [_outputFeatures, 1]);
            var attnNeigh = Engine.Reshape(
                Engine.TensorSlice(_attentionWeights, [h, _outputFeatures], [1, _outputFeatures]), [_outputFeatures, 1]);

            var perBatchOutputs = new Tensor<T>[batchSize];
            for (int b = 0; b < batchSize; b++)
            {
                // Wh for this graph: [numNodes, outputFeatures] (on-tape).
                var wh = Engine.Reshape(
                    Engine.TensorSlice(transformedHead, [b, 0, 0], [1, numNodes, _outputFeatures]),
                    [numNodes, _outputFeatures]);
                // self_i = Wh_i · a_self -> [numNodes, 1]; neigh_j = Wh_j · a_neigh -> [numNodes, 1].
                var selfScores = Engine.TensorMatMul(wh, attnSelf);
                var neighborScores = Engine.TensorMatMul(wh, attnNeigh);
                var neighborRow = Engine.Reshape(neighborScores, [1, numNodes]);
                // e_ij = LeakyReLU(self_i + neigh_j): broadcast [N,1] + [1,N] -> [N,N].
                var scores = Engine.LeakyReLU(Engine.TensorBroadcastAdd(selfScores, neighborRow), _alpha);
                // Additive mask: 0 where an edge exists, -1e9 where adj == 0, so masked
                // neighbours get ~0 softmax weight (constant tensor, not a tape parameter).
                // Precomputed once per batch element above (head-independent).
                var maskAdd = perBatchMasks[b];
                var maskedScores = Engine.TensorAdd(scores, maskAdd);
                // Softmax over the neighbour axis (axis 1); Engine.Softmax is max-subtracted.
                var coeff = Engine.Softmax(maskedScores, 1);
                // Dropout on the normalized attention coefficients (Velickovic 2018 §3.3):
                // inverted, seeded (reproducible), training only. A constant {0, 1/(1-p)} mask
                // multiplied on-tape keeps the aggregation differentiable.
                if (_dropoutRate > 0.0 && IsTrainingMode)
                {
                    var dropMask = BuildAttentionDropoutMask(numNodes, _dropoutRate);
                    coeff = Engine.TensorMultiply(coeff, dropMask);
                }
                // Aggregate neighbour features: [N,N] @ [N,F] -> [N,F] (on-tape).
                var aggregated = Engine.TensorMatMul(coeff, wh);
                perBatchOutputs[b] = Engine.Reshape(aggregated, [1, numNodes, _outputFeatures]);
            }

            var headOutput = batchSize == 1 ? perBatchOutputs[0] : Engine.Concat(perBatchOutputs, 0);
            denseHeadSum = denseHeadSum is null ? headOutput : Engine.TensorAdd(denseHeadSum, headOutput);
        }

        // Average across heads, then add the output bias (on-tape).
        var headAveraged = Engine.TensorDivideScalar(denseHeadSum!, NumOps.FromDouble(_numHeads));
        var denseBias = Engine.Reshape(_bias, [1, 1, _outputFeatures]);
        output = Engine.TensorBroadcastAdd(headAveraged, denseBias);

        activatedOutput = ApplyActivation(output);

        // Reshape output to match original input rank
        Tensor<T> denseReshaped;
        if (rank == 1)
        {
            // Original was [inputFeatures], output should be [outputFeatures]
            denseReshaped = Engine.Reshape(activatedOutput, [_outputFeatures]);
        }
        else if (rank == 2)
        {
            // Original was [numNodes, inputFeatures], output should be [numNodes, outputFeatures]
            denseReshaped = Engine.Reshape(activatedOutput, [numNodes, _outputFeatures]);
        }
        else
        {
            // Restore original batch dimensions
            if (_originalInputShape == null)
            {
                throw new InvalidOperationException("Original input shape was not captured.");
            }
            var originalShape = _originalInputShape;
            var outputShape = new int[rank];
            for (int d = 0; d < rank - 2; d++)
                outputShape[d] = originalShape[d];
            outputShape[rank - 2] = numNodes;
            outputShape[rank - 1] = _outputFeatures;
            denseReshaped = Engine.Reshape(activatedOutput, outputShape);
        }

        // #1668: _lastOutput is a backward-only cache; clear it in inference so the
        // arena-owned reshaped output is not retained across the next Reset().
        _lastOutput = cacheBwd ? denseReshaped : null;
        return denseReshaped;
    }

    private Tensor<T> ExtractHeadWeight(int h)
    {
        var headWeight = new Tensor<T>([_inputFeatures, _outputFeatures]);
        for (int i = 0; i < _inputFeatures; i++)
        {
            for (int j = 0; j < _outputFeatures; j++)
            {
                headWeight[i, j] = _weights[h, i, j];
            }
        }
        return headWeight;
    }

    /// <summary>
    /// Performs batched matrix multiplication for 3D × 2D tensors.
    /// Flattens the batch dimension, performs matmul, then reshapes.
    /// </summary>
    /// <param name="input3D">3D input tensor [batch, rows, cols]</param>
    /// <param name="weights2D">2D weight tensor [cols, outputCols]</param>
    /// <param name="batch">Batch size</param>
    /// <param name="rows">Number of rows per batch (nodes)</param>
    /// <param name="cols">Number of columns (input features)</param>
    /// <param name="outputCols">Number of output columns (output features)</param>
    /// <returns>3D output tensor [batch, rows, outputCols]</returns>
    private Tensor<T> BatchedMatMul3Dx2D(Tensor<T> input3D, Tensor<T> weights2D, int batch, int rows, int cols, int outputCols)
    {
        // Flatten batch dimension: [batch, rows, cols] -> [batch*rows, cols]
        var flattened = input3D.Reshape([batch * rows, cols]);
        // Standard 2D matmul: [batch*rows, cols] @ [cols, outputCols] -> [batch*rows, outputCols]
        var result = Engine.TensorMatMul(flattened, weights2D);
        // Reshape back: [batch*rows, outputCols] -> [batch, rows, outputCols]
        return result.Reshape([batch, rows, outputCols]);
    }

    /// <summary>
    /// Builds the additive attention mask for one graph: 0 where an edge exists in the
    /// adjacency matrix, -1e9 (<paramref name="negInf"/>) where there is no edge, so a
    /// subsequent softmax gives masked (non-neighbour) entries a ~0 coefficient. Constant
    /// data (not a tape parameter). Supports 2-D [nodes, nodes] adjacency (shared across the
    /// batch) and 3-D [batch, nodes, nodes] (per-graph).
    /// </summary>
    private Tensor<T> BuildAttentionMask(Tensor<T> adjacency, bool adjacency2D, int b, int numNodes, T negInf)
    {
        var mask = new Tensor<T>([numNodes, numNodes]);
        T zero = NumOps.Zero;
        for (int i = 0; i < numNodes; i++)
        {
            for (int j = 0; j < numNodes; j++)
            {
                T a = adjacency2D ? adjacency[i, j] : adjacency[b, i, j];
                mask[i, j] = NumOps.Equals(a, zero) ? negInf : zero;
            }
        }
        return mask;
    }

    /// <summary>
    /// Builds an inverted-dropout mask over the [numNodes, numNodes] attention coefficients:
    /// each entry is 0 with probability <paramref name="dropoutRate"/>, otherwise 1/(1-rate)
    /// (Velickovic et al. 2018 §3.3 applies dropout to the normalized coefficients). Uses the
    /// layer's seeded <see cref="_random"/> so training is reproducible. Constant data.
    /// </summary>
    private Tensor<T> BuildAttentionDropoutMask(int numNodes, double dropoutRate)
    {
        var mask = new Tensor<T>([numNodes, numNodes]);
        T zero = NumOps.Zero;
        T keepScale = NumOps.FromDouble(1.0 / (1.0 - dropoutRate));
        int total = numNodes * numNodes;
        for (int i = 0; i < total; i++)
        {
            mask.SetFlat(i, _random.NextDouble() < dropoutRate ? zero : keepScale);
        }
        return mask;
    }

    private void ComputeAttentionScores(int b, int h, int numNodes, Tensor<T> selfScores, Tensor<T> neighborScores)
    {
        // This method is only called from Forward after _adjacencyMatrix, _lastPreSoftmaxScores, and _lastAttentionCoefficients are validated
        if (_adjacencyMatrix == null || _lastPreSoftmaxScores == null || _lastAttentionCoefficients == null)
        {
            throw new InvalidOperationException("Adjacency matrix and score tensors must be set before computing attention scores.");
        }
        var adjacencyMatrix = _adjacencyMatrix;

        // Handle 2D or 3D adjacency matrix
        bool adj2D = adjacencyMatrix.Shape.Length == 2;

        // Raw-array flat indexing. The 4-D tensor indexer [b, h, i, j] resolves to the params-int[]
        // overload, which HEAP-ALLOCATES an int[4] on every access (plus a virtual call + stride
        // resolution). Across the three O(numNodes^2) passes per (b, h) that was on the order of
        // millions of int[4] allocations per forward. Work on the backing arrays with computed flat
        // offsets instead — identical math, no per-element allocation or dispatch.
        var pre = _lastPreSoftmaxScores.GetCpuData();        // [batch, numHeads, numNodes, numNodes]
        var coeff = _lastAttentionCoefficients.GetCpuData(); // same shape
        var adj = adjacencyMatrix.GetCpuData();
        var self = selfScores.GetCpuData();
        var neigh = neighborScores.GetCpuData();
        T zero = NumOps.Zero;
        T minVal = NumOps.MinValue;

        // Base offset of the [i, j] block for this (b, h) slice in the [batch, numHeads, N, N] tensor.
        int bhBase = ((b * _numHeads) + h) * numNodes * numNodes;
        // Adjacency: 2D is shared across the batch (base 0); 3D is offset by b.
        int adjBase = adj2D ? 0 : b * numNodes * numNodes;

        // Compute attention scores with LeakyReLU and softmax.
        var maxScores = new T[numNodes];

        // First pass: compute raw scores and find the per-node max for numerical stability.
        for (int i = 0; i < numNodes; i++)
        {
            int rowBase = bhBase + i * numNodes;
            int adjRow = adjBase + i * numNodes;
            T selfI = self[i];
            T maxI = minVal;
            for (int j = 0; j < numNodes; j++)
            {
                if (NumOps.Equals(adj[adjRow + j], zero))
                {
                    pre[rowBase + j] = minVal;
                    continue;
                }

                // e_ij = LeakyReLU(a_1^T * Wh_i + a_2^T * Wh_j)
                T score = LeakyReLU(NumOps.Add(selfI, neigh[j]));
                pre[rowBase + j] = score;
                if (NumOps.GreaterThan(score, maxI))
                {
                    maxI = score;
                }
            }
            maxScores[i] = maxI;
        }

        // Second pass: softmax over neighbors per node.
        for (int i = 0; i < numNodes; i++)
        {
            int rowBase = bhBase + i * numNodes;
            int adjRow = adjBase + i * numNodes;
            T maxI = maxScores[i];
            T sumExp = zero;

            // exp(score - max) for numerical stability.
            for (int j = 0; j < numNodes; j++)
            {
                if (!NumOps.Equals(adj[adjRow + j], zero))
                {
                    T expVal = NumOps.Exp(NumOps.Subtract(pre[rowBase + j], maxI));
                    coeff[rowBase + j] = expVal;
                    sumExp = NumOps.Add(sumExp, expVal);
                }
            }

            // Normalize and apply dropout.
            for (int j = 0; j < numNodes; j++)
            {
                if (!NumOps.Equals(adj[adjRow + j], zero))
                {
                    T c = NumOps.Divide(coeff[rowBase + j], sumExp);

                    if (_dropoutRate > 0.0 && IsTrainingMode && _random.NextDouble() < _dropoutRate)
                    {
                        c = zero;
                    }
                    else if (_dropoutRate > 0.0 && IsTrainingMode)
                    {
                        c = NumOps.Multiply(c, NumOps.FromDouble(1.0 / (1.0 - _dropoutRate)));
                    }

                    coeff[rowBase + j] = c;
                }
                else
                {
                    coeff[rowBase + j] = zero;
                }
            }
        }
    }

    private T LeakyReLU(T x)
    {
        return NumOps.GreaterThan(x, NumOps.Zero) ? x : NumOps.Multiply(_alpha, x);
    }

    /// <summary>
    /// Sets a 2D slice in a 4D tensor at position [batchIdx, headIdx, :, :] using direct memory copy.
    /// </summary>
    private void Set2DSliceIn4D(Tensor<T> tensor4D, int batchIdx, int headIdx, Tensor<T> slice2D)
    {
        int numHeads = tensor4D.Shape[1];
        int numNodes = tensor4D.Shape[2];
        int features = tensor4D.Shape[3];
        int sliceSize = numNodes * features;

        // Calculate flat offset: batch * (heads * nodes * features) + head * (nodes * features)
        int offset = batchIdx * (numHeads * sliceSize) + headIdx * sliceSize;

        // Copy data directly using indexer for cross-assembly compatibility
        for (int i = 0; i < sliceSize; i++)
        {
            tensor4D.SetFlat(offset + i, slice2D.GetFlat(i));
        }
    }

    /// <summary>
    /// Extracts a 2D slice from a 4D tensor at position [batchIdx, headIdx, :, :] using direct memory copy.
    /// </summary>
    private Tensor<T> Get2DSliceFrom4D(Tensor<T> tensor4D, int batchIdx, int headIdx)
    {
        int numHeads = tensor4D.Shape[1];
        int numNodes = tensor4D.Shape[2];
        int features = tensor4D.Shape[3];
        int sliceSize = numNodes * features;

        // Calculate flat offset
        int offset = batchIdx * (numHeads * sliceSize) + headIdx * sliceSize;

        // Create result tensor and copy using indexer for cross-assembly compatibility
        var result = TensorAllocator.Rent<T>([numNodes, features]);
        for (int i = 0; i < sliceSize; i++)
        {
            result.SetFlat(i, tensor4D.GetFlat(offset + i));
        }

        return result;
    }

    /// <summary>
    /// Sets a 3D slice [h, :, :] into a 4D tensor at [b, h, :, :] for all batches.
    /// </summary>
    private void Set3DSliceIn4DForHead(Tensor<T> tensor4D, int headIdx, Tensor<T> slice3D)
    {
        int batchSize = tensor4D.Shape[0];
        for (int b = 0; b < batchSize; b++)
        {
            var batch2D = slice3D.Slice(b);
            Set2DSliceIn4D(tensor4D, b, headIdx, batch2D);
        }
    }

    /// <summary>
    /// Adds a 2D tensor to a 3D tensor at position [batchIdx, :, :] in-place.
    /// </summary>
    private void Add2DSliceTo3D(Tensor<T> tensor3D, int batchIdx, Tensor<T> toAdd2D)
    {
        int numNodes = tensor3D.Shape[1];
        int features = tensor3D.Shape[2];
        int sliceSize = numNodes * features;
        int offset = batchIdx * sliceSize;

        for (int i = 0; i < sliceSize; i++)
        {
            T current = tensor3D.GetFlat(offset + i);
            T addVal = toAdd2D.GetFlat(i);
            tensor3D.SetFlat(offset + i, NumOps.Add(current, addVal));
        }
    }

    /// <summary>
    /// Adds a 2D tensor to a 3D tensor slice [headIdx, :, :] in-place.
    /// </summary>
    private void Add2DSliceTo3DHead(Tensor<T> tensor3D, int headIdx, Tensor<T> toAdd2D)
    {
        int inputFeatures = tensor3D.Shape[1];
        int outputFeatures = tensor3D.Shape[2];
        int sliceSize = inputFeatures * outputFeatures;
        int offset = headIdx * sliceSize;

        for (int i = 0; i < sliceSize; i++)
        {
            T current = tensor3D.GetFlat(offset + i);
            T addVal = toAdd2D.GetFlat(i);
            tensor3D.SetFlat(offset + i, NumOps.Add(current, addVal));
        }
    }

    /// <summary>
    /// Gets the adjacency matrix slice for a batch (handles both 2D and 3D cases).
    /// </summary>
    private Tensor<T> GetAdjacencySlice(Tensor<T> adjacency, int batchIdx, bool is2D)
    {
        if (is2D)
        {
            // 2D adjacency: return a copy (same for all batches)
            return adjacency;
        }
        else
        {
            // 3D adjacency: extract slice [b, :, :]
            return adjacency.Slice(batchIdx);
        }
    }

    /// <summary>
    /// Computes the LeakyReLU gradient matrix: 1 if preSoftmax > 0, else alpha.
    /// The result is masked by the adjacency matrix.
    /// </summary>
    private Tensor<T> ComputeLeakyReluGradientMatrix(Tensor<T> preSoftmax, Tensor<T> adjMask)
    {
        int numNodes = preSoftmax.Shape[0];
        var result = TensorAllocator.Rent<T>([numNodes, numNodes]);

        for (int i = 0; i < numNodes; i++)
        {
            for (int j = 0; j < numNodes; j++)
            {
                T adjVal = adjMask[i, j];
                if (!NumOps.Equals(adjVal, NumOps.Zero))
                {
                    T val = preSoftmax[i, j];
                    result[i, j] = NumOps.GreaterThan(val, NumOps.Zero) ? NumOps.One : _alpha;
                }
                else
                {
                    result[i, j] = NumOps.Zero;
                }
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        if (_weightsGradient == null || _attentionWeightsGradient == null || _biasGradient == null)
        {
            throw new InvalidOperationException("Backward must be called before UpdateParameters.");
        }

        // Update using Engine operations
        _weights = Engine.TensorSubtract(_weights, Engine.TensorMultiplyScalar(_weightsGradient, learningRate));
        _attentionWeights = Engine.TensorSubtract(_attentionWeights,
            Engine.TensorMultiplyScalar(_attentionWeightsGradient, learningRate));
        _bias = Engine.TensorSubtract(_bias, Engine.TensorMultiplyScalar(_biasGradient, learningRate));

        // Notify GPU that tensor data has changed
        Engine.InvalidatePersistentTensor(_weights);
        Engine.InvalidatePersistentTensor(_attentionWeights);
        Engine.InvalidatePersistentTensor(_bias);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
    {
        var weightsGrad = _weightsGradient != null
            ? new Vector<T>(_weightsGradient.ToArray())
            : new Vector<T>(_weights.Length);
        var attnGrad = _attentionWeightsGradient != null
            ? new Vector<T>(_attentionWeightsGradient.ToArray())
            : new Vector<T>(_attentionWeights.Length);
        var biasGrad = _biasGradient != null
            ? new Vector<T>(_biasGradient.ToArray())
            : new Vector<T>(_bias.Length);

        return Vector<T>.Concatenate(weightsGrad, attnGrad, biasGrad);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        return Vector<T>.Concatenate(
            new Vector<T>(_weights.ToArray()),
            new Vector<T>(_attentionWeights.ToArray()),
            new Vector<T>(_bias.ToArray())
        );
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int weightsCount = _weights.Length;
        int attnCount = _attentionWeights.Length;
        int biasCount = _bias.Length;
        int totalParams = weightsCount + attnCount + biasCount;

        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }

        int index = 0;

        _weights = Tensor<T>.FromVector(parameters.SubVector(index, weightsCount)).Reshape(_weights._shape);
        index += weightsCount;

        _attentionWeights = Tensor<T>.FromVector(parameters.SubVector(index, attnCount))
            .Reshape(_attentionWeights._shape);
        index += attnCount;

        _bias = Tensor<T>.FromVector(parameters.SubVector(index, biasCount));

        // Notify GPU that tensor data has changed
        Engine.InvalidatePersistentTensor(_weights);
        Engine.InvalidatePersistentTensor(_attentionWeights);
        Engine.InvalidatePersistentTensor(_bias);
    }

    /// <summary>
    /// Returns layer-specific metadata for serialization (numHeads, alpha, dropoutRate).
    /// </summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["NumHeads"] = _numHeads.ToString();
        metadata["Alpha"] = NumOps.ToDouble(_alpha).ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["DropoutRate"] = _dropoutRate.ToString(System.Globalization.CultureInfo.InvariantCulture);

        // Store LeakyReLU alpha for proper deserialization
        if (ScalarActivation is LeakyReLUActivation<T> leakyRelu)
        {
            metadata["LeakyReLUAlpha"] = NumOps.ToDouble(leakyRelu.Alpha)
                .ToString(System.Globalization.CultureInfo.InvariantCulture);
        }

        return metadata;
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _weightsGradient = null; _attentionWeightsGradient = null; _biasGradient = null;
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastAttentionCoefficients = null;
        _lastPreSoftmaxScores = null;
        _weightsGradient = null;
        _attentionWeightsGradient = null;
        _biasGradient = null;

        // Clear GPU cache
        ClearGpuCache();
    }

    /// <summary>
    /// Clears GPU cache tensors and gradients.
    /// </summary>
    private void ClearGpuCache()
    {
        _gpuLastInput = null;
        _gpuTransformedCache?.Dispose();
        _gpuTransformedCache = null;
        _gpuAttentionCache?.Dispose();
        _gpuAttentionCache = null;
        _gpuPreActivationCache?.Dispose();
        _gpuPreActivationCache = null;
        _gpuPostActivationCache?.Dispose();
        _gpuPostActivationCache = null;
        _gpuPreLeakyReluCache?.Dispose();
        _gpuPreLeakyReluCache = null;

        _gpuWeightsGradient?.Dispose();
        _gpuWeightsGradient = null;
        _gpuAttentionWeightsGradient?.Dispose();
        _gpuAttentionWeightsGradient = null;
        _gpuBiasGradient?.Dispose();
        _gpuBiasGradient = null;
    }

    /// <summary>
    /// GPU-accelerated forward pass for Graph Attention Networks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Implements multi-head graph attention with GPU acceleration. The computation involves:
    /// 1. Linear transformation for each attention head: H_h = X * W_h
    /// 2. Attention score computation: e_ij = LeakyReLU(a_source^T * H_hi + a_target^T * H_hj)
    /// 3. Softmax normalization over neighbors: α_ij = softmax_j(e_ij)
    /// 4. Weighted aggregation: output_i = Σ_j α_ij * H_hj
    /// 5. Head averaging and bias addition
    /// </para>
    /// <para>
    /// For sparse graphs, uses efficient O(E) edge-based computation instead of O(N²) dense operations.
    /// </para>
    /// </remarks>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs == null || inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        var input = inputs[0];
        if (input._shape == null || input.Shape.Length < 2)
            throw new ArgumentException("Input must be at least 2D [numNodes, inputFeatures].");

        // Check that either adjacency matrix or edge indices are set
        if (_adjacencyMatrix == null && !_useSparseAggregation)
        {
            throw new InvalidOperationException(
                "Graph structure must be set using SetAdjacencyMatrix or SetEdges before calling ForwardGpu.");
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
            // Handle 3D+ tensors - flatten leading dimensions into batch
            int flatBatch = 1;
            for (int d = 0; d < rank - 2; d++)
                flatBatch *= input.Shape[d];
            batchSize = flatBatch;
            numNodes = input.Shape[rank - 2];
            inputFeatures = input.Shape[rank - 1];
        }

        if (inputFeatures != _inputFeatures)
            throw new ArgumentException($"Input features ({inputFeatures}) doesn't match layer input features ({_inputFeatures}).");

        // Allocate output buffer on GPU: [batchSize, numNodes, outputFeatures]
        int outputSize = batchSize * numNodes * _outputFeatures;
        var outputBuffer = backend.AllocateBuffer(new float[outputSize]);

        // Upload weights to GPU for each head
        var headWeightBuffers = new IGpuBuffer[_numHeads];
        var attnSourceBuffers = new IGpuBuffer[_numHeads];
        var attnTargetBuffers = new IGpuBuffer[_numHeads];

        for (int h = 0; h < _numHeads; h++)
        {
            // Extract and upload head weights
            var headWeightData = new float[_inputFeatures * _outputFeatures];
            for (int i = 0; i < _inputFeatures; i++)
            {
                for (int j = 0; j < _outputFeatures; j++)
                {
                    headWeightData[i * _outputFeatures + j] = (float)NumOps.ToDouble(_weights[h, i, j]);
                }
            }
            headWeightBuffers[h] = backend.AllocateBuffer(headWeightData);

            // Extract attention source and target vectors
            var attnSourceData = new float[_outputFeatures];
            var attnTargetData = new float[_outputFeatures];
            for (int f = 0; f < _outputFeatures; f++)
            {
                attnSourceData[f] = (float)NumOps.ToDouble(_attentionWeights[h, f]);
                attnTargetData[f] = (float)NumOps.ToDouble(_attentionWeights[h, _outputFeatures + f]);
            }
            attnSourceBuffers[h] = backend.AllocateBuffer(attnSourceData);
            attnTargetBuffers[h] = backend.AllocateBuffer(attnTargetData);
        }

        // Upload bias
        var biasData = new float[_outputFeatures];
        for (int f = 0; f < _outputFeatures; f++)
            biasData[f] = (float)NumOps.ToDouble(_bias[f]);
        var biasBuffer = backend.AllocateBuffer(biasData);

        // Allocate temporary buffers for intermediate results
        int transformedSize = numNodes * _outputFeatures;
        var transformedBuffer = backend.AllocateBuffer(new float[transformedSize]);
        var attnScoreBuffer = backend.AllocateBuffer(new float[numNodes * numNodes]);
        var headOutputBuffer = backend.AllocateBuffer(new float[numNodes * _outputFeatures]);

        // Zero the output buffer using GPU Fill
        backend.Fill(outputBuffer, 0.0f, outputSize);

        float alphaValue = (float)NumOps.ToDouble(_alpha);

        // Pre-allocate cache for backward pass when training (MUST be before batch loop)
        if (IsTrainingMode)
        {
            ClearGpuCache();
            _gpuLastInput = input;
            _gpuNumNodes = numNodes;
            _gpuBatchSize = batchSize;

            // Transformed cache: [batchSize * numHeads * numNodes * outputFeatures]
            int transformedCacheSize = batchSize * _numHeads * numNodes * _outputFeatures;
            _gpuTransformedCache = backend.AllocateBuffer(transformedCacheSize);
            backend.Fill(_gpuTransformedCache, 0.0f, transformedCacheSize);

            // Attention cache: [batchSize * numHeads * numNodes * numNodes]
            int attentionCacheSize = batchSize * _numHeads * numNodes * numNodes;
            _gpuAttentionCache = backend.AllocateBuffer(attentionCacheSize);
            backend.Fill(_gpuAttentionCache, 0.0f, attentionCacheSize);

            // Pre-activation cache: [batchSize * numNodes * outputFeatures] - stores output before activation
            int preActivationCacheSize = batchSize * numNodes * _outputFeatures;
            _gpuPreActivationCache = backend.AllocateBuffer(preActivationCacheSize);

            // Post-activation cache: same size as pre-activation - stores output after activation
            _gpuPostActivationCache = backend.AllocateBuffer(preActivationCacheSize);

            // Pre-LeakyReLU cache: same size as attention cache - stores scores before LeakyReLU
            _gpuPreLeakyReluCache = backend.AllocateBuffer(attentionCacheSize);
            backend.Fill(_gpuPreLeakyReluCache, 0.0f, attentionCacheSize);
        }

        // Process each batch
        for (int b = 0; b < batchSize; b++)
        {
            // Extract batch input
            IGpuBuffer batchInputBuffer;
            bool ownsBatchBuffer = false;
            if (batchSize == 1)
            {
                batchInputBuffer = input.Buffer;
            }
            else
            {
                // Multi-batch: use GPU-native view for batch slice
                int batchOffset = b * numNodes * inputFeatures;
                var batchView = input.CreateView(batchOffset, [numNodes, inputFeatures]);
                batchInputBuffer = batchView.Buffer;
                ownsBatchBuffer = false; // View doesn't own the underlying buffer
            }

            // Zero the temporary head accumulator using GPU Fill
            backend.Fill(headOutputBuffer, 0.0f, numNodes * _outputFeatures);

            // Process each attention head
            for (int h = 0; h < _numHeads; h++)
            {
                // Step 1: Transform input - transformed = input @ headWeight
                // [numNodes, inputFeatures] @ [inputFeatures, outputFeatures] -> [numNodes, outputFeatures]
                backend.Gemm(batchInputBuffer, headWeightBuffers[h], transformedBuffer,
                    numNodes, _outputFeatures, inputFeatures);

                // Cache transformed features for backward pass
                if (IsTrainingMode && _gpuTransformedCache != null)
                {
                    int transformedOffset = (b * _numHeads + h) * numNodes * _outputFeatures;
                    backend.Copy2DStrided(transformedBuffer, _gpuTransformedCache,
                        1, numNodes * _outputFeatures,
                        batchSize * _numHeads * numNodes * _outputFeatures, transformedOffset);
                }

                // Step 2: Compute attention scores
                // For each node i, compute score_ij = LeakyReLU(source_i + target_j)
                // source_i = transformed[i, :] @ attnSource -> [numNodes]
                // target_j = transformed[j, :] @ attnTarget -> [numNodes]
                var sourceScoreBuffer = backend.AllocateBuffer(new float[numNodes]);
                var targetScoreBuffer = backend.AllocateBuffer(new float[numNodes]);

                // Compute source and target scores using matmul
                backend.Gemm(transformedBuffer, attnSourceBuffers[h], sourceScoreBuffer,
                    numNodes, 1, _outputFeatures);

                backend.Gemm(transformedBuffer, attnTargetBuffers[h], targetScoreBuffer,
                    numNodes, 1, _outputFeatures);

                // Compute pairwise attention scores with LeakyReLU and masking
                if (_useSparseAggregation && _edgeSourceIndices != null && _edgeTargetIndices != null)
                {
                    // Sparse attention using GPU operations
                    int numEdges = _edgeSourceIndices.Length;

                    // Upload edge indices to GPU (done once per batch, could be cached)
                    var sourceIndicesData = new int[numEdges];
                    var targetIndicesData = new int[numEdges];
                    for (int e = 0; e < numEdges; e++)
                    {
                        sourceIndicesData[e] = _edgeSourceIndices.GetFlat(e);
                        targetIndicesData[e] = _edgeTargetIndices.GetFlat(e);
                    }
                    var srcIdxBuffer = backend.AllocateIntBuffer(sourceIndicesData);
                    var tgtIdxBuffer = backend.AllocateIntBuffer(targetIndicesData);

                    // Gather source and target scores for each edge on GPU
                    var edgeSrcScoreBuffer = backend.AllocateBuffer(new float[numEdges]);
                    var edgeTgtScoreBuffer = backend.AllocateBuffer(new float[numEdges]);
                    backend.Gather(sourceScoreBuffer, srcIdxBuffer, edgeSrcScoreBuffer, numEdges, 1);
                    backend.Gather(targetScoreBuffer, tgtIdxBuffer, edgeTgtScoreBuffer, numEdges, 1);

                    // Add source and target scores on GPU: e_ij = source_i + target_j
                    var edgeScoreBuffer = backend.AllocateBuffer(new float[numEdges]);
                    backend.Add(edgeSrcScoreBuffer, edgeTgtScoreBuffer, edgeScoreBuffer, numEdges);

                    // Cache pre-LeakyReLU scores in dense format for backward pass
                    if (IsTrainingMode && _gpuPreLeakyReluCache != null)
                    {
                        // Create dense indices for scattering edge scores
                        var denseIndicesPreLeaky = new int[numEdges];
                        for (int e = 0; e < numEdges; e++)
                        {
                            int src = sourceIndicesData[e];
                            int tgt = targetIndicesData[e];
                            denseIndicesPreLeaky[e] = tgt * numNodes + src;
                        }
                        var denseIdxPreLeakyBuffer = backend.AllocateIntBuffer(denseIndicesPreLeaky);

                        // Create temp buffer for this batch/head, fill with 0 (non-edges stay 0)
                        using var preLeakyTemp = backend.AllocateBuffer(numNodes * numNodes);
                        backend.Fill(preLeakyTemp, 0.0f, numNodes * numNodes);

                        // Scatter pre-LeakyReLU edge scores to dense format
                        backend.ScatterAdd(edgeScoreBuffer, denseIdxPreLeakyBuffer, preLeakyTemp, numEdges, numNodes * numNodes);

                        // Copy to cache at correct offset
                        int preLeakyOffset = (b * _numHeads + h) * numNodes * numNodes;
                        backend.Copy2DStrided(preLeakyTemp, _gpuPreLeakyReluCache,
                            1, numNodes * numNodes,
                            batchSize * _numHeads * numNodes * numNodes, preLeakyOffset);

                        denseIdxPreLeakyBuffer.Dispose();
                    }

                    // Apply LeakyReLU to edge scores on GPU
                    backend.LeakyRelu(edgeScoreBuffer, edgeScoreBuffer, alphaValue, numEdges);

                    // Edge softmax on GPU using SegmentedSoftmax if available, otherwise fall back to
                    // building a sparse-to-dense attention matrix for nodes with edges
                    // Convert edges to dense attention matrix per target, apply softmax, then aggregate
                    var edgeAttnBuffer = backend.AllocateBuffer(new float[numNodes * numNodes]);
                    backend.Fill(edgeAttnBuffer, float.NegativeInfinity, numNodes * numNodes);

                    // Scatter edge scores into attention matrix: attn[target, source] = edgeScore
                    // Create scatter indices for dense matrix positions
                    var denseIndices = new int[numEdges];
                    for (int e = 0; e < numEdges; e++)
                    {
                        int src = sourceIndicesData[e];
                        int tgt = targetIndicesData[e];
                        denseIndices[e] = tgt * numNodes + src;
                    }
                    var denseIdxBuffer = backend.AllocateIntBuffer(denseIndices);

                    // Scatter edge scores to dense attention matrix
                    // First fill with -inf, then scatter actual scores
                    backend.Fill(edgeAttnBuffer, float.NegativeInfinity, numNodes * numNodes);
                    backend.ScatterAdd(edgeScoreBuffer, denseIdxBuffer, edgeAttnBuffer, numEdges, numNodes * numNodes);

                    // Apply row-wise softmax on GPU (handles -inf for non-edges)
                    backend.Softmax(edgeAttnBuffer, edgeAttnBuffer, numNodes, numNodes);

                    // Cache attention scores for backward pass
                    if (IsTrainingMode && _gpuAttentionCache != null)
                    {
                        int attnOffset = (b * _numHeads + h) * numNodes * numNodes;
                        backend.Copy2DStrided(edgeAttnBuffer, _gpuAttentionCache,
                            1, numNodes * numNodes,
                            batchSize * _numHeads * numNodes * numNodes, attnOffset);
                    }

                    // Aggregate: headResult = attention @ transformed
                    var headResultBuffer = backend.AllocateBuffer(new float[numNodes * _outputFeatures]);
                    backend.Gemm(edgeAttnBuffer, transformedBuffer, headResultBuffer,
                        numNodes, _outputFeatures, numNodes);

                    // Accumulate head result into headOutputBuffer on GPU
                    backend.Add(headOutputBuffer, headResultBuffer, headOutputBuffer, numNodes * _outputFeatures);

                    // Clean up
                    srcIdxBuffer.Dispose();
                    tgtIdxBuffer.Dispose();
                    edgeSrcScoreBuffer.Dispose();
                    edgeTgtScoreBuffer.Dispose();
                    edgeScoreBuffer.Dispose();
                    denseIdxBuffer.Dispose();
                    edgeAttnBuffer.Dispose();
                    headResultBuffer.Dispose();
                }
                else if (_adjacencyMatrix != null)
                {
                    // Dense attention entirely on GPU
                    // Upload adjacency matrix for this batch (could be cached)
                    var adjData = new float[numNodes * numNodes];
                    bool adj2D = _adjacencyMatrix.Shape.Length == 2;
                    for (int i = 0; i < numNodes; i++)
                    {
                        for (int j = 0; j < numNodes; j++)
                        {
                            T adjVal = adj2D ? _adjacencyMatrix[i, j] : _adjacencyMatrix[b, i, j];
                            adjData[i * numNodes + j] = (float)NumOps.ToDouble(adjVal);
                        }
                    }
                    var adjBuffer = backend.AllocateBuffer(adjData);

                    // Build pairwise score matrix on GPU: score[i,j] = source[i] + target[j]
                    // Use Gemm for outer sum: source[N,1] @ ones[1,N] broadcasts source across columns
                    // ones[N,1] @ target[1,N] broadcasts target across rows
                    var onesRowBuffer = backend.AllocateBuffer(numNodes);
                    backend.Fill(onesRowBuffer, 1.0f, numNodes);

                    // sourceBroadcast[i,j] = source[i] for all j
                    // Using Gemm: [N,1] @ [1,N] where the [1,N] is all 1s
                    var sourceBroadcastBuffer = backend.AllocateBuffer(new float[numNodes * numNodes]);
                    backend.Gemm(sourceScoreBuffer, onesRowBuffer, sourceBroadcastBuffer, numNodes, numNodes, 1);

                    // targetBroadcast[i,j] = target[j] for all i
                    // Using Gemm: [N,1] of 1s @ [1,N] of targets
                    var onesColBuffer = backend.AllocateBuffer(numNodes);
                    backend.Fill(onesColBuffer, 1.0f, numNodes);
                    var targetBroadcastBuffer = backend.AllocateBuffer(new float[numNodes * numNodes]);
                    backend.Gemm(onesColBuffer, targetScoreBuffer, targetBroadcastBuffer, numNodes, numNodes, 1);

                    // Add source and target broadcasts on GPU
                    backend.Add(sourceBroadcastBuffer, targetBroadcastBuffer, attnScoreBuffer, numNodes * numNodes);

                    // Cache pre-LeakyReLU scores for backward pass (before LeakyReLU is applied)
                    if (IsTrainingMode && _gpuPreLeakyReluCache != null)
                    {
                        int preLeakyOffset = (b * _numHeads + h) * numNodes * numNodes;
                        backend.Copy2DStrided(attnScoreBuffer, _gpuPreLeakyReluCache,
                            1, numNodes * numNodes,
                            batchSize * _numHeads * numNodes * numNodes, preLeakyOffset);
                    }

                    // Apply LeakyReLU on GPU
                    backend.LeakyRelu(attnScoreBuffer, attnScoreBuffer, alphaValue, numNodes * numNodes);

                    // Mask with adjacency: where adj==0, set to -inf for softmax
                    // Create mask: -inf where adj==0, 0 where adj!=0
                    var maskBuffer = backend.AllocateBuffer(new float[numNodes * numNodes]);
                    // mask = (1 - adj) * (-inf) = -inf where adj=0, 0 where adj=1
                    var onesMatrixBuffer = backend.AllocateBuffer(numNodes * numNodes);
                    backend.Fill(onesMatrixBuffer, 1.0f, numNodes * numNodes);
                    backend.Subtract(onesMatrixBuffer, adjBuffer, maskBuffer, numNodes * numNodes);  // 1-adj
                    backend.Scale(maskBuffer, maskBuffer, float.NegativeInfinity, numNodes * numNodes);  // (1-adj)*-inf

                    // Add mask to attention scores (adds -inf to non-edges)
                    backend.Add(attnScoreBuffer, maskBuffer, attnScoreBuffer, numNodes * numNodes);

                    // Apply row-wise softmax on GPU
                    backend.Softmax(attnScoreBuffer, attnScoreBuffer, numNodes, numNodes);

                    // Cache attention scores for backward pass
                    if (IsTrainingMode && _gpuAttentionCache != null)
                    {
                        int attnOffset = (b * _numHeads + h) * numNodes * numNodes;
                        backend.Copy2DStrided(attnScoreBuffer, _gpuAttentionCache,
                            1, numNodes * numNodes,
                            batchSize * _numHeads * numNodes * numNodes, attnOffset);
                    }

                    // Aggregate: output = attention @ transformed on GPU
                    var headResultBuffer = backend.AllocateBuffer(new float[numNodes * _outputFeatures]);
                    backend.Gemm(attnScoreBuffer, transformedBuffer, headResultBuffer,
                        numNodes, _outputFeatures, numNodes);

                    // Accumulate head result on GPU
                    backend.Add(headOutputBuffer, headResultBuffer, headOutputBuffer, numNodes * _outputFeatures);

                    // Clean up
                    adjBuffer.Dispose();
                    onesRowBuffer.Dispose();
                    onesColBuffer.Dispose();
                    sourceBroadcastBuffer.Dispose();
                    targetBroadcastBuffer.Dispose();
                    maskBuffer.Dispose();
                    onesMatrixBuffer.Dispose();
                    headResultBuffer.Dispose();
                }

                sourceScoreBuffer.Dispose();
                targetScoreBuffer.Dispose();
            }

            // Average across heads using GPU Scale
            float headScale = 1.0f / _numHeads;
            backend.Scale(headOutputBuffer, headOutputBuffer, headScale, numNodes * _outputFeatures);

            // Add bias (broadcast to all nodes) using GPU BiasAdd
            backend.BiasAdd(headOutputBuffer, biasBuffer, headOutputBuffer, numNodes, _outputFeatures);

            // Copy to output buffer at correct batch offset
            if (batchSize == 1)
            {
                backend.Copy(headOutputBuffer, outputBuffer, numNodes * _outputFeatures);
            }
            else
            {
                // Multi-batch: use GPU-native strided copy to write to offset
                int outputOffset = b * numNodes * _outputFeatures;
                int copySize = numNodes * _outputFeatures;
                // Copy2DStrided copies contiguous data to a position in the destination
                // Using numRows=1 for simple offset copy
                backend.Copy2DStrided(headOutputBuffer, outputBuffer, 1, copySize, outputSize, outputOffset);
            }

            if (ownsBatchBuffer)
            {
                batchInputBuffer.Dispose();
            }
        }

        // Apply activation using base class GPU activation method
        var activationType = GetFusedActivationType();
        if (activationType != FusedActivationType.None)
        {
            // Cache pre-activation output for backward pass (before activation is applied)
            if (IsTrainingMode && _gpuPreActivationCache != null)
            {
                backend.Copy(outputBuffer, _gpuPreActivationCache, outputSize);
            }

            ApplyGpuActivation(backend, outputBuffer, outputBuffer, outputSize, activationType);

            // Cache post-activation output for backward pass (after activation is applied)
            // Some activations (Sigmoid, Tanh) need the output for their backward pass
            if (IsTrainingMode && _gpuPostActivationCache != null)
            {
                backend.Copy(outputBuffer, _gpuPostActivationCache, outputSize);
            }
        }

        // Clean up weight buffers
        for (int h = 0; h < _numHeads; h++)
        {
            headWeightBuffers[h].Dispose();
            attnSourceBuffers[h].Dispose();
            attnTargetBuffers[h].Dispose();
        }
        biasBuffer.Dispose();
        transformedBuffer.Dispose();
        attnScoreBuffer.Dispose();
        headOutputBuffer.Dispose();

        // Determine output shape
        int[] outputShape = rank == 2
            ? [numNodes, _outputFeatures]
            : [batchSize, numNodes, _outputFeatures];

        return GpuTensorHelper.UploadToGpu<T>(backend, outputBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// Gets the number of attention heads used in multi-head attention.
    /// </summary>
    public int NumHeads => _numHeads;

    /// <summary>
    /// Gets the dropout rate applied to attention coefficients during training.
    /// </summary>
    public double DropoutRate => _dropoutRate;

    /// <summary>
    /// Gets the LeakyReLU negative-slope used inside the attention
    /// score computation. Exposed for deserialization round-trip
    /// verification.
    /// </summary>
    public double Alpha => NumOps.ToDouble(_alpha);
}
