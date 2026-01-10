using AiDotNet.ActivationFunctions;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements cross-attention for conditioning diffusion models on text or other context.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para>
/// Cross-attention differs from self-attention in that queries come from spatial features
/// while keys and values come from conditioning (text embeddings). This is the core mechanism
/// that allows text-to-image models like Stable Diffusion to follow prompts.
/// </para>
/// <para>
/// <b>For Beginners:</b> Cross-attention is how the model "reads" the text prompt.
/// - Queries: "What should I generate at each position?"
/// - Keys/Values: "What does the text describe?"
/// - Output: Spatial features modified to match the text description
/// </para>
/// </remarks>
public class CrossAttentionLayer<T> : LayerBase<T>
{
    private readonly int _queryDim;
    private readonly int _contextDim;
    private readonly int _headCount;
    private readonly int _headDim;

    // Query projection: queryDim -> queryDim
    private Tensor<T> _queryWeights;

    // Key projection: contextDim -> queryDim
    private Tensor<T> _keyWeights;

    // Value projection: contextDim -> queryDim
    private Tensor<T> _valueWeights;

    // Output projection: queryDim -> queryDim
    private Tensor<T> _outputWeights;
    private Tensor<T> _outputBias;

    // Cached values for backward pass
    private Tensor<T>? _lastQuery;
    private Tensor<T>? _lastContext;
    private Tensor<T>? _lastAttentionScores;
    private Tensor<T>? _lastOutput;

    // Gradient tensors
    private Tensor<T>? _queryWeightsGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _outputWeightsGradient;
    private Tensor<T>? _outputBiasGradient;

    /// <summary>
    /// Stores the original query shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalQueryShape;

    // GPU cached tensors for backward pass
    private IGpuTensor<T>? _gpuQuery;
    private IGpuTensor<T>? _gpuContext;
    private IGpuTensor<T>? _gpuQ;
    private IGpuTensor<T>? _gpuK;
    private IGpuTensor<T>? _gpuV;
    private IGpuTensor<T>? _gpuAttnOutput;
    private IGpuTensor<T>? _gpuAttnWeights;
    private int _gpuBatch;
    private int _gpuQueryLen;
    private int _gpuContextLen;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer can execute on GPU.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <inheritdoc/>
    public override int ParameterCount =>
        _queryWeights.Length + _keyWeights.Length + _valueWeights.Length +
        _outputWeights.Length + _outputBias.Length;

    /// <summary>
    /// Creates a new cross-attention layer for conditioning.
    /// </summary>
    /// <param name="queryDim">Dimension of query features (spatial channels).</param>
    /// <param name="contextDim">Dimension of context features (text embedding).</param>
    /// <param name="headCount">Number of attention heads.</param>
    /// <param name="sequenceLength">Maximum sequence length for queries.</param>
    public CrossAttentionLayer(int queryDim, int contextDim, int headCount, int sequenceLength = 64)
        : base(new[] { sequenceLength, queryDim }, new[] { sequenceLength, queryDim }, (IActivationFunction<T>)new IdentityActivation<T>())
    {
        _queryDim = queryDim;
        _contextDim = contextDim;
        _headCount = headCount;
        _headDim = queryDim / headCount;

        if (queryDim % headCount != 0)
        {
            throw new ArgumentException($"Query dimension ({queryDim}) must be divisible by head count ({headCount}).");
        }

        // Initialize projection matrices
        // Q projection: queryDim -> queryDim (same dimension)
        _queryWeights = new Tensor<T>(new[] { queryDim, queryDim });

        // K and V projections: contextDim -> queryDim (different input/output dims)
        _keyWeights = new Tensor<T>(new[] { contextDim, queryDim });
        _valueWeights = new Tensor<T>(new[] { contextDim, queryDim });

        // Output projection: queryDim -> queryDim
        _outputWeights = new Tensor<T>(new[] { queryDim, queryDim });
        _outputBias = new Tensor<T>(new[] { queryDim });

        InitializeParameters();

        // Register trainable parameters for GPU memory optimization
        RegisterTrainableParameter(_queryWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_keyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_valueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputBias, PersistentTensorRole.Biases);
    }

    private void InitializeParameters()
    {
        // Xavier initialization for each weight matrix
        InitializeWeights(_queryWeights);
        InitializeWeights(_keyWeights);
        InitializeWeights(_valueWeights);
        InitializeWeights(_outputWeights);
        _outputBias.Fill(NumOps.Zero);
    }

    private void InitializeWeights(Tensor<T> weights)
    {
        int rows = weights.Shape[0];
        int cols = weights.Shape[1];
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (rows + cols)));

        var span = weights.AsWritableSpan();
        var rng = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < span.Length; i++)
        {
            double val = (rng.NextDouble() - 0.5) * NumOps.ToDouble(scale);
            span[i] = NumOps.FromDouble(val);
        }
    }

    /// <summary>
    /// Forward pass for self-attention (not typically used for cross-attention).
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // For single input, use it as both query and context (self-attention fallback)
        return ForwardCrossAttention(input, input);
    }

    /// <summary>
    /// Forward pass with multiple inputs for cross-attention.
    /// </summary>
    /// <param name="inputs">Array of inputs: [query] or [query, context].</param>
    /// <returns>Output tensor with same shape as query.</returns>
    public override Tensor<T> Forward(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 1)
        {
            return ForwardCrossAttention(inputs[0], inputs[0]);
        }
        else if (inputs.Length >= 2)
        {
            return ForwardCrossAttention(inputs[0], inputs[1]);
        }

        throw new ArgumentException("CrossAttentionLayer requires 1 or 2 inputs.");
    }

    /// <summary>
    /// Forward pass with separate query and context tensors.
    /// </summary>
    /// <param name="query">Query tensor [batch, querySeqLen, queryDim] or [batch, channels, H, W].</param>
    /// <param name="context">Context tensor [batch, contextSeqLen, contextDim].</param>
    /// <returns>Output tensor with same shape as query.</returns>
    private Tensor<T> ForwardCrossAttention(Tensor<T> query, Tensor<T> context)
    {
        // Store original shape for any-rank tensor support
        _originalQueryShape = query.Shape;
        var queryShape = query.Shape;
        var contextShape = context.Shape;
        int queryRank = queryShape.Length;

        // Handle any-rank query input
        int batch, queryLen, height = 0, width = 0;
        int queryDimActual;
        bool is4D = queryRank == 4;
        bool isHigherRank = queryRank > 4;

        if (queryRank == 2)
        {
            // 2D: [seqLen, queryDim] -> add batch dim
            batch = 1;
            queryLen = queryShape[0];
            queryDimActual = queryShape[1];
            query = query.Reshape(new[] { 1, queryLen, queryDimActual });
        }
        else if (queryRank == 3)
        {
            // Standard 3D: [batch, seqLen, queryDim]
            batch = queryShape[0];
            queryLen = queryShape[1];
            queryDimActual = queryShape[2];
        }
        else if (is4D)
        {
            batch = queryShape[0];
            queryDimActual = queryShape[1];
            height = queryShape[2];
            width = queryShape[3];
            queryLen = height * width;

            // Reshape query from [B, C, H, W] to [B, H*W, C]
            query = ReshapeNCHWToNLC(query);
        }
        else
        {
            // Higher-rank: collapse leading dims into batch
            int flatBatch = 1;
            for (int d = 0; d < queryRank - 2; d++)
                flatBatch *= queryShape[d];
            batch = flatBatch;
            queryLen = queryShape[queryRank - 2];
            queryDimActual = queryShape[queryRank - 1];
            query = query.Reshape(new[] { flatBatch, queryLen, queryDimActual });
        }

        if (queryDimActual != _queryDim)
        {
            throw new ArgumentException(
                $"Query feature dimension ({queryDimActual}) must match expected {_queryDim}.");
        }

        // Handle context shape - for 2D [seqLen, dim], for 3D [batch, seqLen, dim]
        int contextRank = contextShape.Length;
        int contextBatch;
        int contextLen;
        int contextDimActual;

        if (contextRank == 1)
        {
            // 1D: [dim]
            contextBatch = 1;
            contextLen = 1;
            contextDimActual = contextShape[0];
            context = context.Reshape(new[] { 1, 1, contextDimActual });
        }
        else if (contextRank == 2)
        {
            // 2D: [seqLen, dim]
            contextBatch = 1;
            contextLen = contextShape[0];
            contextDimActual = contextShape[1];
            // Add batch dimension of 1, broadcast later if needed
            context = context.Reshape(new[] { 1, contextLen, contextDimActual });
        }
        else
        {
            // Rank >= 3: [batch..., seqLen, dim]
            contextDimActual = contextShape[contextRank - 1];
            contextLen = contextShape[contextRank - 2];
            int flatBatch = 1;
            for (int d = 0; d < contextRank - 2; d++)
                flatBatch *= contextShape[d];
            contextBatch = flatBatch;
            if (contextRank != 3)
            {
                context = context.Reshape(new[] { flatBatch, contextLen, contextDimActual });
            }
        }

        if (contextDimActual != _contextDim)
        {
            throw new ArgumentException(
                $"Context feature dimension ({contextDimActual}) must match expected {_contextDim}.");
        }

        if (contextBatch != batch)
        {
            if (contextBatch == 1 && batch > 1)
            {
                context = BroadcastContext(context, batch, contextLen, contextDimActual);
                contextBatch = batch;
            }
            else
            {
                throw new ArgumentException(
                    $"Context batch dimension ({contextBatch}) must match query batch ({batch}) or be 1.");
            }
        }

        _lastQuery = query;
        _lastContext = context;
        // Project Q, K, V
        // query: [B, queryLen, queryDim]
        // context: [B, contextLen, contextDim]
        var Q = ProjectTensor(query, _queryWeights); // [B, queryLen, queryDim]
        var K = ProjectTensor(context, _keyWeights); // [B, contextLen, queryDim]
        var V = ProjectTensor(context, _valueWeights); // [B, contextLen, queryDim]

        // Reshape to multi-head: [B, seqLen, numHeads, headDim] -> [B, numHeads, seqLen, headDim]
        Q = ReshapeToHeads(Q, batch, queryLen, _headCount, _headDim);
        K = ReshapeToHeads(K, batch, contextLen, _headCount, _headDim);
        V = ReshapeToHeads(V, batch, contextLen, _headCount, _headDim);

        // Compute scaled dot-product attention using the Engine
        // ScaledDotProductAttention handles: Q @ K^T / sqrt(d_k), softmax, scores @ V
        var attended = Engine.ScaledDotProductAttention(
            Q, K, V,
            mask: null,
            scale: 1.0 / Math.Sqrt(_headDim),
            out var attentionWeights);

        // Cache post-softmax attention weights for any downstream use
        _lastAttentionScores = attentionWeights;

        // Reshape back: [B, numHeads, queryLen, headDim] -> [B, queryLen, queryDim]
        attended = ReshapeFromHeads(attended, batch, queryLen, _headCount, _headDim);

        // Output projection
        var output = ProjectTensor(attended, _outputWeights);

        // Add bias
        output = AddBias(output, _outputBias);

        // Only store for backward pass during training - skip during inference
        if (IsTrainingMode)
        {
            _lastOutput = output;
        }

        // Restore original shape for any-rank support
        if (_originalQueryShape != null)
        {
            int origRank = _originalQueryShape.Length;
            if (origRank == 2)
            {
                // 2D input -> 2D output (remove batch dim)
                output = output.Reshape(new[] { queryLen, _queryDim });
            }
            else if (is4D)
            {
                // Reshape back to 4D
                output = ReshapeNLCToNCHW(output, batch, _queryDim, height, width);
            }
            else if (isHigherRank)
            {
                // Restore original leading dims
                int[] newShape = new int[origRank];
                for (int d = 0; d < origRank - 2; d++)
                    newShape[d] = _originalQueryShape[d];
                newShape[origRank - 2] = queryLen;
                newShape[origRank - 1] = _queryDim;
                output = output.Reshape(newShape);
            }
        }

        return output;
    }

    private Tensor<T> ReshapeNCHWToNLC(Tensor<T> input)
    {
        // [B, C, H, W] -> [B, L, C] where L = H*W
        // Use IEngine for GPU-accelerated permute and reshape
        var shape = input.Shape;
        int batch = shape[0];
        int channels = shape[1];
        int height = shape[2];
        int width = shape[3];
        int seqLen = height * width;

        // First permute NCHW to NHWC: [B, C, H, W] -> [B, H, W, C]
        var nhwc = Engine.TensorPermute(input, new[] { 0, 2, 3, 1 });

        // Then reshape to NLC: [B, H, W, C] -> [B, H*W, C]
        return Engine.Reshape(nhwc, new[] { batch, seqLen, channels });
    }

    private Tensor<T> ReshapeNLCToNCHW(Tensor<T> input, int batch, int channels, int height, int width)
    {
        // [B, L, C] -> [B, C, H, W] where L = H*W
        // Use IEngine for GPU-accelerated reshape and permute

        // First reshape NLC to NHWC: [B, H*W, C] -> [B, H, W, C]
        var nhwc = Engine.Reshape(input, new[] { batch, height, width, channels });

        // Then permute NHWC to NCHW: [B, H, W, C] -> [B, C, H, W]
        return Engine.TensorPermute(nhwc, new[] { 0, 3, 1, 2 });
    }
    private Tensor<T> BroadcastContext(Tensor<T> context, int batch, int contextLen, int contextDim)
    {
        if (context.Shape.Length != 3 || context.Shape[0] != 1)
        {
            throw new ArgumentException("Context tensor must have batch dimension 1 for broadcasting.");
        }

        // Use IEngine for GPU-accelerated tensor tiling
        // Tile the context along the batch dimension: [1, contextLen, contextDim] -> [batch, contextLen, contextDim]
        return Engine.TensorTile(context, new[] { batch, 1, 1 });
    }

    private Tensor<T> ProjectTensor(Tensor<T> input, Tensor<T> weights)
    {
        // input: [B, seqLen, inputDim]
        // weights: [inputDim, outputDim]
        // output: [B, seqLen, outputDim]
        // Use IEngine for GPU-accelerated batched matrix multiplication
        var inputShape = input.Shape;
        int batch = inputShape[0];
        int seqLen = inputShape[1];
        int inputDim = inputShape[2];
        int outputDim = weights.Shape[1];

        // Reshape input to [B*seqLen, inputDim] for 2D matmul
        var flatInput = Engine.Reshape(input, new[] { batch * seqLen, inputDim });

        // Perform 2D matrix multiplication: [B*seqLen, inputDim] × [inputDim, outputDim] = [B*seqLen, outputDim]
        var flatOutput = Engine.TensorMatMul(flatInput, weights);

        // Reshape back to [B, seqLen, outputDim]
        return Engine.Reshape(flatOutput, new[] { batch, seqLen, outputDim });
    }

    private Tensor<T> ReshapeToHeads(Tensor<T> input, int batch, int seqLen, int numHeads, int headDim)
    {
        // [B, seqLen, numHeads * headDim] -> [B, numHeads, seqLen, headDim]
        // Use IEngine for GPU-accelerated reshape and permute operations

        // First reshape: [B, seqLen, numHeads * headDim] -> [B, seqLen, numHeads, headDim]
        var reshaped = Engine.Reshape(input, new[] { batch, seqLen, numHeads, headDim });

        // Then permute: [B, seqLen, numHeads, headDim] -> [B, numHeads, seqLen, headDim]
        return Engine.TensorPermute(reshaped, new[] { 0, 2, 1, 3 });
    }

    private Tensor<T> ReshapeFromHeads(Tensor<T> input, int batch, int seqLen, int numHeads, int headDim)
    {
        // [B, numHeads, seqLen, headDim] -> [B, seqLen, numHeads * headDim]
        // Use IEngine for GPU-accelerated permute and reshape operations

        // First permute: [B, numHeads, seqLen, headDim] -> [B, seqLen, numHeads, headDim]
        var permuted = Engine.TensorPermute(input, new[] { 0, 2, 1, 3 });

        // Then reshape: [B, seqLen, numHeads, headDim] -> [B, seqLen, numHeads * headDim]
        int embDim = numHeads * headDim;
        return Engine.Reshape(permuted, new[] { batch, seqLen, embDim });
    }

    private Tensor<T> AddBias(Tensor<T> input, Tensor<T> bias)
    {
        // input: [B, seqLen, dim]
        // bias: [dim]
        // Use IEngine for GPU-accelerated broadcast addition
        // Reshape bias to [1, 1, dim] for proper broadcasting
        var biasReshaped = Engine.Reshape(bias, new[] { 1, 1, bias.Length });
        return Engine.TensorBroadcastAdd(input, biasReshaped);
    }

    /// <inheritdoc/>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // Simplified backward pass - in production would use autodiff
        // For now, return gradient scaled by output weights
        if (_lastQuery == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Normalize outputGradient to 3D to match canonical _lastQuery shape
        var outGrad3D = outputGradient;
        int origRank = _originalQueryShape?.Length ?? 3;
        int height = 0, width = 0;
        bool is4D = origRank == 4;
        bool isHigherRank = origRank > 4;

        if (_originalQueryShape != null && origRank == 2)
        {
            // 2D output gradient -> 3D (add batch dim)
            outGrad3D = outputGradient.Reshape(new[] { 1, outputGradient.Shape[0], outputGradient.Shape[1] });
        }
        else if (_originalQueryShape != null && is4D)
        {
            // 4D NCHW -> 3D [B, H*W, C]
            height = _originalQueryShape[2];
            width = _originalQueryShape[3];
            outGrad3D = ReshapeNCHWToNLC(outputGradient);
        }
        else if (_originalQueryShape != null && isHigherRank)
        {
            // Higher-rank output gradient -> 3D (flatten leading dims)
            int flatBatch = 1;
            for (int d = 0; d < origRank - 2; d++)
                flatBatch *= _originalQueryShape[d];
            int seqLenHigh = _originalQueryShape[origRank - 2];
            outGrad3D = outputGradient.Reshape(new[] { flatBatch, seqLenHigh, _queryDim });
        }

        // Compute weight gradients (simplified)
        var queryShape = _lastQuery.Shape;
        int batch = queryShape[0];
        int seqLen = queryShape[1];

        // Approximate gradients
        _queryWeightsGradient = new Tensor<T>(_queryWeights.Shape);
        _keyWeightsGradient = new Tensor<T>(_keyWeights.Shape);
        _valueWeightsGradient = new Tensor<T>(_valueWeights.Shape);
        _outputWeightsGradient = new Tensor<T>(_outputWeights.Shape);
        _outputBiasGradient = new Tensor<T>(_outputBias.Shape);

        // Return input gradient (backprop through output projection)
        var inputGradient = ProjectTensor(outGrad3D, TransposeWeights(_outputWeights));

        // Restore higher-rank gradients to their original shape
        if (_originalQueryShape != null && origRank != 3)
        {
            if (origRank == 2)
            {
                // 3D -> 2D (remove batch dim)
                inputGradient = inputGradient.Reshape(new[] { seqLen, _queryDim });
            }
            else if (is4D)
            {
                // 3D -> 4D NCHW
                inputGradient = ReshapeNLCToNCHW(inputGradient, batch, _queryDim, height, width);
            }
            else if (isHigherRank)
            {
                // 3D -> higher-rank (restore original leading dims)
                inputGradient = inputGradient.Reshape(_originalQueryShape);
            }
        }

        return inputGradient;
    }

    /// <summary>
    /// Performs the backward pass on GPU for the cross-attention layer.
    /// </summary>
    /// <param name="outputGradient">The GPU tensor containing the gradient of the loss with respect to the output.</param>
    /// <returns>The GPU tensor containing the gradient of the loss with respect to the query input.</returns>
    public override IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (_gpuQuery == null || _gpuContext == null || _gpuQ == null || _gpuK == null || _gpuV == null ||
            _gpuAttnOutput == null || _gpuAttnWeights == null)
            throw new InvalidOperationException("ForwardGpu must be called before BackwardGpu.");

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires a DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend == null)
            throw new InvalidOperationException("GPU backend unavailable.");

        int batch = _gpuBatch;
        int queryLen = _gpuQueryLen;
        int contextLen = _gpuContextLen;
        int flatQueryLen = batch * queryLen;
        int flatContextLen = batch * contextLen;

        // Step 1: Backprop through output projection and bias
        // Forward: output = contextFlat @ outputWeights + bias
        // Backward: dOutputWeights = dOutput.T @ contextFlat, dAttnOutput = dOutput @ outputWeights.T, dBias = sum(dOutput)

        // Flatten output gradient for matmul
        var dOutputFlat = gpuEngine.ReshapeGpu(outputGradient, [flatQueryLen, _queryDim]);
        var attnOutputFlat = gpuEngine.ReshapeGpu(_gpuAttnOutput, [flatQueryLen, _queryDim]);

        // Upload weights to GPU
        var gpuOutputWeights = gpuEngine.UploadToGpu(_outputWeights, GpuTensorRole.Weight);

        // Compute output weight gradient: dOutput.T @ attnOutput
        var dOutputFlatT = gpuEngine.TransposeGpu(dOutputFlat);
        var gpuDOutputWeights = gpuEngine.MatMulGpuTensors(dOutputFlatT, attnOutputFlat);
        _outputWeightsGradient = gpuDOutputWeights.ToTensor();

        // Compute output bias gradient: sum over batch*seq dimension
        var gpuDBias = gpuEngine.SumAxisGpu(dOutputFlat, 0);
        _outputBiasGradient = gpuDBias.ToTensor();

        // Compute gradient through output projection: dAttnOutput = dOutput @ outputWeights.T
        var outputWeightsT = gpuEngine.TransposeGpu(gpuOutputWeights);
        var dAttnOutputFlat = gpuEngine.MatMulGpuTensors(dOutputFlat, outputWeightsT);
        var dAttnOutput = gpuEngine.ReshapeGpu(dAttnOutputFlat, [batch, queryLen, _queryDim]);

        // Step 2: Backprop through attention (multi-head)
        // Reshape to 4D for attention backward: [B, queryLen, numHeads, headDim] -> [B, numHeads, queryLen, headDim]
        var dAttnReshaped = gpuEngine.ReshapeGpu(dAttnOutput, [batch, queryLen, _headCount, _headDim]);
        var dAttnPermuted = gpuEngine.PermuteGpu(dAttnReshaped, [0, 2, 1, 3]);

        // Reshape Q, K, V to 4D for attention backward
        var qReshaped = gpuEngine.ReshapeGpu(_gpuQ, [batch, queryLen, _headCount, _headDim]);
        var kReshaped = gpuEngine.ReshapeGpu(_gpuK, [batch, contextLen, _headCount, _headDim]);
        var vReshaped = gpuEngine.ReshapeGpu(_gpuV, [batch, contextLen, _headCount, _headDim]);
        var qPermuted = gpuEngine.PermuteGpu(qReshaped, [0, 2, 1, 3]);
        var kPermuted = gpuEngine.PermuteGpu(kReshaped, [0, 2, 1, 3]);
        var vPermuted = gpuEngine.PermuteGpu(vReshaped, [0, 2, 1, 3]);

        // Reshape attention weights to 4D
        var attnWeights4D = gpuEngine.ReshapeGpu(_gpuAttnWeights, [batch, _headCount, queryLen, contextLen]);

        // Use ScaledDotProductAttentionBackwardGpu
        double scale = 1.0 / Math.Sqrt(_headDim);
        var (dQ4D, dK4D, dV4D) = gpuEngine.ScaledDotProductAttentionBackwardGpu(
            dAttnPermuted, qPermuted, kPermuted, vPermuted, attnWeights4D, scale);

        // Permute gradients back to [B, seqLen, numHeads, headDim]
        var dQPermuted = gpuEngine.PermuteGpu(dQ4D, [0, 2, 1, 3]);
        var dKPermuted = gpuEngine.PermuteGpu(dK4D, [0, 2, 1, 3]);
        var dVPermuted = gpuEngine.PermuteGpu(dV4D, [0, 2, 1, 3]);

        // Reshape to [B, seqLen, queryDim]
        var dQ = gpuEngine.ReshapeGpu(dQPermuted, [batch, queryLen, _queryDim]);
        var dK = gpuEngine.ReshapeGpu(dKPermuted, [batch, contextLen, _queryDim]);
        var dV = gpuEngine.ReshapeGpu(dVPermuted, [batch, contextLen, _queryDim]);

        // Step 3: Backprop through Q, K, V projections
        // Forward: Q = query @ queryWeights, K = context @ keyWeights, V = context @ valueWeights
        // Backward: dQueryWeights = dQ.T @ query, dKeyWeights = dK.T @ context, dValueWeights = dV.T @ context

        // Flatten for matmul
        var dQFlat = gpuEngine.ReshapeGpu(dQ, [flatQueryLen, _queryDim]);
        var dKFlat = gpuEngine.ReshapeGpu(dK, [flatContextLen, _queryDim]);
        var dVFlat = gpuEngine.ReshapeGpu(dV, [flatContextLen, _queryDim]);

        var queryFlat = gpuEngine.ReshapeGpu(_gpuQuery, [flatQueryLen, _queryDim]);
        var contextFlat = gpuEngine.ReshapeGpu(_gpuContext, [flatContextLen, _contextDim]);

        // Upload weights
        var gpuQueryWeights = gpuEngine.UploadToGpu(_queryWeights, GpuTensorRole.Weight);
        var gpuKeyWeights = gpuEngine.UploadToGpu(_keyWeights, GpuTensorRole.Weight);
        var gpuValueWeights = gpuEngine.UploadToGpu(_valueWeights, GpuTensorRole.Weight);

        // Compute weight gradients
        var dQFlatT = gpuEngine.TransposeGpu(dQFlat);
        var dKFlatT = gpuEngine.TransposeGpu(dKFlat);
        var dVFlatT = gpuEngine.TransposeGpu(dVFlat);

        var gpuDQueryWeights = gpuEngine.MatMulGpuTensors(dQFlatT, queryFlat);
        var gpuDKeyWeights = gpuEngine.MatMulGpuTensors(dKFlatT, contextFlat);
        var gpuDValueWeights = gpuEngine.MatMulGpuTensors(dVFlatT, contextFlat);

        _queryWeightsGradient = gpuDQueryWeights.ToTensor();
        _keyWeightsGradient = gpuDKeyWeights.ToTensor();
        _valueWeightsGradient = gpuDValueWeights.ToTensor();

        // Compute query input gradient: dQuery = dQ @ queryWeights.T
        var queryWeightsT = gpuEngine.TransposeGpu(gpuQueryWeights);
        var dQueryFlat = gpuEngine.MatMulGpuTensors(dQFlat, queryWeightsT);

        // Reshape back to 3D: [B, queryLen, queryDim]
        var dQuery = gpuEngine.ReshapeGpu(dQueryFlat, [batch, queryLen, _queryDim]);

        // Restore original shape if needed
        if (_originalQueryShape != null)
        {
            int origRank = _originalQueryShape.Length;
            if (origRank == 2)
            {
                dQuery = gpuEngine.ReshapeGpu(dQuery, [queryLen, _queryDim]);
            }
            else if (origRank == 4)
            {
                int height = _originalQueryShape[2];
                int width = _originalQueryShape[3];
                dQuery = ReshapeNLCToNCHWGpu(gpuEngine, dQuery, batch, _queryDim, height, width);
            }
            else if (origRank > 3)
            {
                dQuery = gpuEngine.ReshapeGpu(dQuery, _originalQueryShape);
            }
        }

        return dQuery;
    }

    private Tensor<T> TransposeWeights(Tensor<T> weights)
    {
        // Use IEngine for GPU-accelerated 2D tensor transpose
        return Engine.TensorTranspose(weights);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        // Only update weights that have computed gradients
        if (_queryWeightsGradient is not null)
        {
            UpdateWeight(_queryWeights, _queryWeightsGradient, learningRate);
        }

        if (_keyWeightsGradient is not null)
        {
            UpdateWeight(_keyWeights, _keyWeightsGradient, learningRate);
        }

        if (_valueWeightsGradient is not null)
        {
            UpdateWeight(_valueWeights, _valueWeightsGradient, learningRate);
        }

        if (_outputWeightsGradient is not null)
        {
            UpdateWeight(_outputWeights, _outputWeightsGradient, learningRate);
        }

        if (_outputBiasGradient is not null)
        {
            UpdateWeight(_outputBias, _outputBiasGradient, learningRate);
        }

        // Notify GPU that tensor data has changed
        Engine.InvalidatePersistentTensor(_queryWeights);
        Engine.InvalidatePersistentTensor(_keyWeights);
        Engine.InvalidatePersistentTensor(_valueWeights);
        Engine.InvalidatePersistentTensor(_outputWeights);
        Engine.InvalidatePersistentTensor(_outputBias);
    }

    private void UpdateWeight(Tensor<T> weight, Tensor<T> gradient, T learningRate)
    {
        var wSpan = weight.AsWritableSpan();
        var gSpan = gradient.AsSpan();
        for (int i = 0; i < wSpan.Length; i++)
        {
            wSpan[i] = NumOps.Subtract(wSpan[i], NumOps.Multiply(learningRate, gSpan[i]));
        }
    }

    /// <summary>
    /// GPU-resident forward pass for cross-attention with multiple inputs.
    /// </summary>
    /// <param name="inputs">Array containing [query] or [query, context] GPU tensors.</param>
    /// <returns>GPU-resident output tensor with same shape as query.</returns>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.");

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        // Handle single or dual input
        IGpuTensor<T> query = inputs[0];
        IGpuTensor<T> context = inputs.Length >= 2 ? inputs[1] : inputs[0];

        int[] queryShape = query.Shape;
        int[] contextShape = context.Shape;
        int queryRank = queryShape.Length;

        // Store original shape for output
        int[] originalQueryShape = queryShape;
        int batch, queryLen;
        int height = 0, width = 0;
        bool is4D = queryRank == 4;

        if (queryRank == 2)
        {
            batch = 1;
            queryLen = queryShape[0];
        }
        else if (queryRank == 3)
        {
            batch = queryShape[0];
            queryLen = queryShape[1];
        }
        else if (is4D)
        {
            batch = queryShape[0];
            height = queryShape[2];
            width = queryShape[3];
            queryLen = height * width;

            // Reshape query from [B, C, H, W] to [B, H*W, C]
            query = ReshapeNCHWToNLCGpu(gpuEngine, query);
        }
        else
        {
            int flatBatch = 1;
            for (int d = 0; d < queryRank - 2; d++)
                flatBatch *= queryShape[d];
            batch = flatBatch;
            queryLen = queryShape[queryRank - 2];
            query = gpuEngine.ReshapeGpu(query, new[] { flatBatch, queryLen, _queryDim });
        }

        // Handle context shape
        int contextRank = contextShape.Length;
        int contextLen;
        int contextBatch;

        if (contextRank == 1)
        {
            contextBatch = 1;
            contextLen = 1;
            context = gpuEngine.ReshapeGpu(context, new[] { 1, 1, _contextDim });
        }
        else if (contextRank == 2)
        {
            contextBatch = 1;
            contextLen = contextShape[0];
            context = gpuEngine.ReshapeGpu(context, new[] { 1, contextLen, _contextDim });
        }
        else
        {
            int flatBatch = 1;
            for (int d = 0; d < contextRank - 2; d++)
                flatBatch *= contextShape[d];
            contextBatch = flatBatch;
            contextLen = contextShape[contextRank - 2];
            if (contextRank != 3)
            {
                context = gpuEngine.ReshapeGpu(context, new[] { flatBatch, contextLen, _contextDim });
            }
        }

        // Broadcast context if needed using GPU tile kernel
        if (contextBatch != batch)
        {
            if (contextBatch == 1 && batch > 1)
            {
                // Tile context along batch dimension using proper GPU kernel
                context = gpuEngine.TileBatchGpu<T>(context, batch);
            }
            else
            {
                throw new ArgumentException(
                    $"Context batch dimension ({contextBatch}) must match query batch ({batch}) or be 1.");
            }
        }

        // Project Q, K, V
        var Q = gpuEngine.BatchedMatMulGpu(query, _queryWeights);   // [B, queryLen, queryDim]
        var K = gpuEngine.BatchedMatMulGpu(context, _keyWeights);  // [B, contextLen, queryDim]
        var V = gpuEngine.BatchedMatMulGpu(context, _valueWeights); // [B, contextLen, queryDim]

        // Reshape to [batch, seqLen, heads, headDim]
        var qReshaped = gpuEngine.ReshapeGpu(Q, new[] { batch, queryLen, _headCount, _headDim });
        var kReshaped = gpuEngine.ReshapeGpu(K, new[] { batch, contextLen, _headCount, _headDim });
        var vReshaped = gpuEngine.ReshapeGpu(V, new[] { batch, contextLen, _headCount, _headDim });

        // Transpose to [batch, heads, seqLen, headDim]
        var qPermuted = gpuEngine.PermuteGpu(qReshaped, new[] { 0, 2, 1, 3 });
        var kPermuted = gpuEngine.PermuteGpu(kReshaped, new[] { 0, 2, 1, 3 });
        var vPermuted = gpuEngine.PermuteGpu(vReshaped, new[] { 0, 2, 1, 3 });

        // Scaled dot-product attention
        // Use overload that returns attention weights during training for backward pass
        double scale = 1.0 / Math.Sqrt(_headDim);
        IGpuTensor<T> attended;
        IGpuTensor<T>? attentionWeightsGpu = null;

        if (IsTrainingMode)
        {
            // Training mode: get attention weights for backward pass
            attended = gpuEngine.ScaledDotProductAttentionGpu(
                qPermuted, kPermuted, vPermuted, scale, out attentionWeightsGpu);
        }
        else
        {
            // Inference mode: no need for attention weights
            attended = gpuEngine.ScaledDotProductAttentionGpu(qPermuted, kPermuted, vPermuted, scale);
        }

        // Transpose back to [batch, seqLen, heads, headDim]
        var contextPermuted = gpuEngine.PermuteGpu(attended, new[] { 0, 2, 1, 3 });

        // Reshape to [batch, seqLen, queryDim]
        var contextFlat = gpuEngine.ReshapeGpu(contextPermuted, new[] { batch, queryLen, _queryDim });

        // Output projection
        var output = gpuEngine.BatchedMatMulGpu(contextFlat, _outputWeights);

        // Add bias
        output = gpuEngine.AddBiasGpu(output, _outputBias);

        // Cache state for backward pass only during training
        // Skip this expensive download during inference (50% overhead reduction)
        if (IsTrainingMode)
        {
            // Cache GPU tensors for GPU backward pass
            _gpuQuery = query;
            _gpuContext = context;
            _gpuQ = Q;
            _gpuK = K;
            _gpuV = V;
            _gpuAttnOutput = contextFlat;
            _gpuAttnWeights = attentionWeightsGpu;
            _gpuBatch = batch;
            _gpuQueryLen = queryLen;
            _gpuContextLen = contextLen;

            // Also cache CPU tensors for CPU backward compatibility
            _lastQuery = query.ToTensor();
            _lastContext = context.ToTensor();
            _lastAttentionScores = attentionWeightsGpu?.ToTensor();
            _lastOutput = output.ToTensor();
            _originalQueryShape = originalQueryShape;
        }

        // Restore original shape
        if (originalQueryShape.Length == 2)
        {
            output = gpuEngine.ReshapeGpu(output, new[] { queryLen, _queryDim });
        }
        else if (is4D)
        {
            output = ReshapeNLCToNCHWGpu(gpuEngine, output, batch, _queryDim, height, width);
        }
        else if (originalQueryShape.Length > 3)
        {
            int[] newShape = new int[originalQueryShape.Length];
            for (int d = 0; d < originalQueryShape.Length - 2; d++)
                newShape[d] = originalQueryShape[d];
            newShape[^2] = queryLen;
            newShape[^1] = _queryDim;
            output = gpuEngine.ReshapeGpu(output, newShape);
        }

        return output;
    }

    private static IGpuTensor<T> ReshapeNCHWToNLCGpu(DirectGpuTensorEngine gpuEngine, IGpuTensor<T> input)
    {
        int[] shape = input.Shape;
        int batch = shape[0];
        int channels = shape[1];
        int height = shape[2];
        int width = shape[3];
        int seqLen = height * width;

        // Permute NCHW to NHWC: [B, C, H, W] -> [B, H, W, C]
        var nhwc = gpuEngine.PermuteGpu(input, new[] { 0, 2, 3, 1 });

        // Reshape to NLC: [B, H, W, C] -> [B, H*W, C]
        return gpuEngine.ReshapeGpu(nhwc, new[] { batch, seqLen, channels });
    }

    private static IGpuTensor<T> ReshapeNLCToNCHWGpu(DirectGpuTensorEngine gpuEngine, IGpuTensor<T> input, int batch, int channels, int height, int width)
    {
        // Reshape NLC to NHWC: [B, H*W, C] -> [B, H, W, C]
        var nhwc = gpuEngine.ReshapeGpu(input, new[] { batch, height, width, channels });

        // Permute NHWC to NCHW: [B, H, W, C] -> [B, C, H, W]
        return gpuEngine.PermuteGpu(nhwc, new[] { 0, 3, 1, 2 });
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        return Vector<T>.Concatenate(
            new Vector<T>(_queryWeights.ToArray()),
            new Vector<T>(_keyWeights.ToArray()),
            new Vector<T>(_valueWeights.ToArray()),
            new Vector<T>(_outputWeights.ToArray()),
            new Vector<T>(_outputBias.ToArray()));
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int qLen = _queryWeights.Length;
        int kLen = _keyWeights.Length;
        int vLen = _valueWeights.Length;
        int oLen = _outputWeights.Length;
        int bLen = _outputBias.Length;

        int index = 0;
        CopyToTensor(parameters, index, _queryWeights); index += qLen;
        CopyToTensor(parameters, index, _keyWeights); index += kLen;
        CopyToTensor(parameters, index, _valueWeights); index += vLen;
        CopyToTensor(parameters, index, _outputWeights); index += oLen;
        CopyToTensor(parameters, index, _outputBias);

        // Notify GPU that tensor data has changed
        Engine.InvalidatePersistentTensor(_queryWeights);
        Engine.InvalidatePersistentTensor(_keyWeights);
        Engine.InvalidatePersistentTensor(_valueWeights);
        Engine.InvalidatePersistentTensor(_outputWeights);
        Engine.InvalidatePersistentTensor(_outputBias);
    }

    private void CopyToTensor(Vector<T> source, int offset, Tensor<T> dest)
    {
        var span = dest.AsWritableSpan();
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = source[offset + i];
        }
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _lastQuery = null;
        _lastContext = null;
        _lastAttentionScores = null;
        _lastOutput = null;
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _outputWeightsGradient = null;
        _outputBiasGradient = null;

        // Clear GPU cached tensors
        _gpuQuery = null;
        _gpuContext = null;
        _gpuQ = null;
        _gpuK = null;
        _gpuV = null;
        _gpuAttnOutput = null;
        _gpuAttnWeights = null;
    }

    /// <inheritdoc/>
    public override bool SupportsJitCompilation =>
        _queryWeights != null && _keyWeights != null &&
        _valueWeights != null && _outputWeights != null;

    /// <inheritdoc/>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        // Create symbolic input nodes for query and context
        var queryInput = new Tensor<T>(new int[] { 1, InputShape[0], _queryDim });
        var queryNode = Autodiff.TensorOperations<T>.Variable(queryInput, "query");
        inputNodes.Add(queryNode);

        var contextInput = new Tensor<T>(new int[] { 1, 77, _contextDim }); // Standard text encoder output length
        var contextNode = Autodiff.TensorOperations<T>.Variable(contextInput, "context");
        inputNodes.Add(contextNode);

        // Create weight nodes
        var wqNode = Autodiff.TensorOperations<T>.Constant(_queryWeights, "Wq");
        var wkNode = Autodiff.TensorOperations<T>.Constant(_keyWeights, "Wk");
        var wvNode = Autodiff.TensorOperations<T>.Constant(_valueWeights, "Wv");
        var woNode = Autodiff.TensorOperations<T>.Constant(_outputWeights, "Wo");

        // Apply cross-attention using multi-head attention with separate query/key sources
        var output = Autodiff.TensorOperations<T>.MultiHeadAttention(
            query: queryNode,
            key: contextNode,
            value: contextNode,
            numHeads: _headCount,
            wQ: wqNode,
            wK: wkNode,
            wV: wvNode,
            wO: woNode);

        return output;
    }
}



