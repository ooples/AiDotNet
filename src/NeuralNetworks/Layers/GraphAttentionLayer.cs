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
// Rank 2 - [numNodes, inputFeatures] - is the form both forward paths name in their own comments
// ("Original was [numNodes, inputFeatures], output should be [numNodes, outputFeatures]") and the only
// one [LayerProperty(TestInputShape = "4, 8")] exercises. Higher ranks ARE handled, but their node count
// has to agree with the separately-installed adjacency matrix, so a declaration at those ranks would be
// making a claim about a tensor this layer never sees on its own; they are left undeclared deliberately.
//
// The node axis is TensorAxis.Other because there is no shared role for it: graph nodes are neither a
// sequence nor a spatial extent, and calling them Time or Length would licence a causal or convolutional
// layer to consume this output as if the ordering meant something. Other participates in rank checks and
// still resolves Same(), which is all this relation needs.
[TensorLayout(TensorAxis.Other, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input,
    Note = "Node features: the leading axis is the graph's nodes, sized by the installed adjacency matrix.")]
[TensorLayout(TensorAxis.Other, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class GraphAttentionLayer<T> : LayerBase<T>, IGraphConvolutionLayer<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// HAND-WRITTEN because the emitted width is not <c>outputFeatures</c> but
    /// <c>_combinedOutputFeatures</c>, and that difference is the whole point of multi-head attention:
    /// <c>concatenateHeads ? checked(outputFeatures * numHeads) : outputFeatures</c>. Both forward paths
    /// end at that field - the sparse one allocates <c>[batchSize, numNodes, _combinedOutputFeatures]</c>,
    /// the dense one concatenates the heads on axis 2 - so declaring <c>Fixed(_outputFeatures)</c> here
    /// would be wrong by a factor of numHeads for every concatenating configuration, which is the default.
    /// </para>
    /// <para>
    /// The node axis is Same: attention re-weights neighbours, it never adds or removes a node.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 2 || _combinedOutputFeatures <= 0) return null;

        return new[]
        {
            new OutputAxisContract(TensorAxis.Other, AxisRelation.Same(TensorAxis.Other)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_combinedOutputFeatures)),
        };
    }

    private readonly int _inputFeatures;
    private readonly int _outputFeatures;
    private readonly int _numHeads;
    private readonly bool _concatenateHeads;
    private readonly int _combinedOutputFeatures;
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
    [AiDotNet.Attributes.FittedParameter(InputSized = true)]
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
    [Scratch]
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// Cached output from forward pass for backward computation.
    /// </summary>
    [Scratch]
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Cached attention coefficients from forward pass.
    /// </summary>
    [Scratch]
    private Tensor<T>? _lastAttentionCoefficients;

    /// <summary>
    /// Cached pre-softmax attention scores for gradient computation.
    /// </summary>
    [Scratch]
    private Tensor<T>? _lastPreSoftmaxScores;



    /// <summary>
    /// Gradients for weight parameters.
    /// </summary>
    [Scratch]
    private Tensor<T>? _weightsGradient;

    /// <summary>
    /// Gradients for attention parameters.
    /// </summary>
    [Scratch]
    private Tensor<T>? _attentionWeightsGradient;

    /// <summary>
    /// Gradients for bias parameters.
    /// </summary>
    [Scratch]
    private Tensor<T>? _biasGradient;
    private IGpuBuffer? _gpuTransformedCache;  // [numNodes * outputFeatures * numHeads]
    private IGpuBuffer? _gpuAttentionCache;    // [numNodes * numNodes * numHeads]
    private IGpuBuffer? _gpuPreActivationCache;  // [batchSize * numNodes * outputFeatures] - pre-activation output for backward
    private IGpuBuffer? _gpuPostActivationCache; // [batchSize * numNodes * outputFeatures] - post-activation output for backward
    private IGpuBuffer? _gpuPreLeakyReluCache;   // [batchSize * numHeads * numNodes * numNodes] - pre-LeakyReLU attention scores

    // GPU gradient fields
    private IGpuBuffer? _gpuWeightsGradient;
    private IGpuBuffer? _gpuAttentionWeightsGradient;
    private IGpuBuffer? _gpuBiasGradient;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public int InputFeatures => _inputFeatures;

    /// <inheritdoc/>
    public int OutputFeatures => _outputFeatures;

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphAttentionLayer{T}"/> class.
    /// </summary>
    public GraphAttentionLayer(
        [LayerState] int inputFeatures,
        [LayerState] int outputFeatures,
        [LayerState] int numHeads = 1,
        [LayerState] double alpha = 0.2,
        [LayerState] double dropoutRate = 0.0,
        IActivationFunction<T>? activationFunction = null,
        IInitializationStrategy<T>? initializationStrategy = null,
        bool concatenateHeads = false)
        : base(
            [inputFeatures],
            [concatenateHeads ? checked(outputFeatures * numHeads) : outputFeatures],
            activationFunction ?? new IdentityActivation<T>())
    {
        InitializationStrategy = initializationStrategy ?? InitializationStrategies<T>.Eager;

        _inputFeatures = inputFeatures;
        _outputFeatures = outputFeatures;
        _numHeads = numHeads;
        _concatenateHeads = concatenateHeads;
        _combinedOutputFeatures = concatenateHeads ? checked(outputFeatures * numHeads) : outputFeatures;
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
        _bias = new Tensor<T>([_combinedOutputFeatures]);

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
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
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

            output = TensorAllocator.Rent<T>([batchSize, numNodes, _combinedOutputFeatures]);
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
                    concatenate: _concatenateHeads,
                    out var batchAttnCoeffs);

                // Add bias using broadcasting and set in output
                var biasBroadcast = Engine.Reshape(_bias, [1, _combinedOutputFeatures]);
                var biasedOutput = Engine.TensorAdd(batchOutput, biasBroadcast);
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
                sparseReshaped = Engine.Reshape(activatedOutput, [_combinedOutputFeatures]);
            }
            else if (rank == 2)
            {
                sparseReshaped = Engine.Reshape(activatedOutput, [numNodes, _combinedOutputFeatures]);
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
                outputShape[rank - 1] = _combinedOutputFeatures;
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

        // Retain every head output until the paper's head-combination operation. Hidden
        // GAT layers concatenate heads; the final prediction layer averages them.
        var denseHeadOutputs = new Tensor<T>[_numHeads];
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
                var scores = Engine.LeakyReLU(Engine.TensorAdd(selfScores, neighborRow), _alpha);
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
            denseHeadOutputs[h] = headOutput;
            if (!_concatenateHeads)
                denseHeadSum = denseHeadSum is null ? headOutput : Engine.TensorAdd(denseHeadSum, headOutput);
        }

        var combinedHeads = _concatenateHeads
            ? Engine.Concat(denseHeadOutputs, 2)
            : Engine.TensorDivideScalar(denseHeadSum!, NumOps.FromDouble(_numHeads));
        var denseBias = Engine.Reshape(_bias, [1, 1, _combinedOutputFeatures]);
        output = Engine.TensorAdd(combinedHeads, denseBias);

        activatedOutput = ApplyActivation(output);

        // Reshape output to match original input rank
        Tensor<T> denseReshaped;
        if (rank == 1)
        {
            // Original was [inputFeatures], output should be [outputFeatures]
            denseReshaped = Engine.Reshape(activatedOutput, [_combinedOutputFeatures]);
        }
        else if (rank == 2)
        {
            // Original was [numNodes, inputFeatures], output should be [numNodes, outputFeatures]
            denseReshaped = Engine.Reshape(activatedOutput, [numNodes, _combinedOutputFeatures]);
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
            outputShape[rank - 1] = _combinedOutputFeatures;
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

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        if (_weightsGradient == null || _attentionWeightsGradient == null || _biasGradient == null)
        {
            throw new InvalidOperationException("Backward must be called before UpdateParameters.");
        }

        // Update using Engine operations
        Engine.TensorSubtractInPlace(_weights, Engine.TensorMultiplyScalar(_weightsGradient, learningRate));
        Engine.TensorSubtractInPlace(_attentionWeights,
            Engine.TensorMultiplyScalar(_attentionWeightsGradient, learningRate));
        Engine.TensorSubtractInPlace(_bias, Engine.TensorMultiplyScalar(_biasGradient, learningRate));

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
        metadata["HeadOutputFeatures"] = _outputFeatures.ToString();
        metadata["ConcatenateHeads"] = _concatenateHeads.ToString();
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
    /// Gets the number of attention heads used in multi-head attention.
    /// </summary>
    public int NumHeads => _numHeads;

    /// <summary>
    /// Gets whether the per-head outputs are concatenated (hidden-layer GAT rule)
    /// instead of averaged (prediction-layer GAT rule).
    /// </summary>
    public bool ConcatenateHeads => _concatenateHeads;

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
