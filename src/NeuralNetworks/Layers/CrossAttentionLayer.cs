using AiDotNet.ActivationFunctions;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
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

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

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

        _lastOutput = output;

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
        var shape = input.Shape;
        int batch = shape[0];
        int channels = shape[1];
        int height = shape[2];
        int width = shape[3];
        int seqLen = height * width;

        var output = new Tensor<T>(new[] { batch, seqLen, channels });
        var inSpan = input.AsSpan();
        var outSpan = output.AsWritableSpan();

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    int spatialIdx = h * width + w;
                    for (int c = 0; c < channels; c++)
                    {
                        int srcIdx = b * channels * height * width + c * height * width + h * width + w;
                        int dstIdx = b * seqLen * channels + spatialIdx * channels + c;
                        outSpan[dstIdx] = inSpan[srcIdx];
                    }
                }
            }
        }

        return output;
    }

    private Tensor<T> ReshapeNLCToNCHW(Tensor<T> input, int batch, int channels, int height, int width)
    {
        int seqLen = height * width;
        var output = new Tensor<T>(new[] { batch, channels, height, width });
        var inSpan = input.AsSpan();
        var outSpan = output.AsWritableSpan();

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    int spatialIdx = h * width + w;
                    for (int c = 0; c < channels; c++)
                    {
                        int srcIdx = b * seqLen * channels + spatialIdx * channels + c;
                        int dstIdx = b * channels * height * width + c * height * width + h * width + w;
                        outSpan[dstIdx] = inSpan[srcIdx];
                    }
                }
            }
        }

        return output;
    }
    private Tensor<T> BroadcastContext(Tensor<T> context, int batch, int contextLen, int contextDim)
    {
        if (context.Shape.Length != 3 || context.Shape[0] != 1)
        {
            throw new ArgumentException("Context tensor must have batch dimension 1 for broadcasting.");
        }

        var output = new Tensor<T>(new[] { batch, contextLen, contextDim });
        var inSpan = context.AsSpan();
        var outSpan = output.AsWritableSpan();
        int sliceSize = contextLen * contextDim;

        for (int b = 0; b < batch; b++)
        {
            int dstOffset = b * sliceSize;
            for (int i = 0; i < sliceSize; i++)
            {
                outSpan[dstOffset + i] = inSpan[i];
            }
        }

        return output;
    }

    private Tensor<T> ProjectTensor(Tensor<T> input, Tensor<T> weights)
    {
        // input: [B, seqLen, inputDim]
        // weights: [inputDim, outputDim]
        // output: [B, seqLen, outputDim]
        var inputShape = input.Shape;
        int batch = inputShape[0];
        int seqLen = inputShape[1];
        int inputDim = inputShape[2];
        int outputDim = weights.Shape[1];

        var output = new Tensor<T>(new[] { batch, seqLen, outputDim });
        var inSpan = input.AsSpan();
        var wSpan = weights.AsSpan();
        var outSpan = output.AsWritableSpan();

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int o = 0; o < outputDim; o++)
                {
                    double sum = 0.0;
                    for (int i = 0; i < inputDim; i++)
                    {
                        int inIdx = b * seqLen * inputDim + s * inputDim + i;
                        int wIdx = i * outputDim + o;
                        sum += NumOps.ToDouble(inSpan[inIdx]) * NumOps.ToDouble(wSpan[wIdx]);
                    }
                    int outIdx = b * seqLen * outputDim + s * outputDim + o;
                    outSpan[outIdx] = NumOps.FromDouble(sum);
                }
            }
        }

        return output;
    }

    private Tensor<T> ReshapeToHeads(Tensor<T> input, int batch, int seqLen, int numHeads, int headDim)
    {
        // [B, seqLen, numHeads * headDim] -> [B, numHeads, seqLen, headDim]
        var output = new Tensor<T>(new[] { batch, numHeads, seqLen, headDim });
        var inSpan = input.AsSpan();
        var outSpan = output.AsWritableSpan();

        int embDim = numHeads * headDim;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    for (int d = 0; d < headDim; d++)
                    {
                        int srcIdx = b * seqLen * embDim + s * embDim + h * headDim + d;
                        int dstIdx = b * numHeads * seqLen * headDim + h * seqLen * headDim + s * headDim + d;
                        outSpan[dstIdx] = inSpan[srcIdx];
                    }
                }
            }
        }

        return output;
    }

    private Tensor<T> ReshapeFromHeads(Tensor<T> input, int batch, int seqLen, int numHeads, int headDim)
    {
        // [B, numHeads, seqLen, headDim] -> [B, seqLen, numHeads * headDim]
        int embDim = numHeads * headDim;
        var output = new Tensor<T>(new[] { batch, seqLen, embDim });
        var inSpan = input.AsSpan();
        var outSpan = output.AsWritableSpan();

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    for (int d = 0; d < headDim; d++)
                    {
                        int srcIdx = b * numHeads * seqLen * headDim + h * seqLen * headDim + s * headDim + d;
                        int dstIdx = b * seqLen * embDim + s * embDim + h * headDim + d;
                        outSpan[dstIdx] = inSpan[srcIdx];
                    }
                }
            }
        }

        return output;
    }

    private Tensor<T> ComputeAttentionScores(Tensor<T> Q, Tensor<T> K, int batch, int queryLen, int keyLen)
    {
        // Q: [B, numHeads, queryLen, headDim]
        // K: [B, numHeads, keyLen, headDim]
        // scores: [B, numHeads, queryLen, keyLen]
        var scores = new Tensor<T>(new[] { batch, _headCount, queryLen, keyLen });
        var qSpan = Q.AsSpan();
        var kSpan = K.AsSpan();
        var scoresSpan = scores.AsWritableSpan();

        double scale = 1.0 / Math.Sqrt(_headDim);

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < _headCount; h++)
            {
                for (int q = 0; q < queryLen; q++)
                {
                    for (int k = 0; k < keyLen; k++)
                    {
                        double dot = 0.0;
                        for (int d = 0; d < _headDim; d++)
                        {
                            int qIdx = b * _headCount * queryLen * _headDim + h * queryLen * _headDim + q * _headDim + d;
                            int kIdx = b * _headCount * keyLen * _headDim + h * keyLen * _headDim + k * _headDim + d;
                            dot += NumOps.ToDouble(qSpan[qIdx]) * NumOps.ToDouble(kSpan[kIdx]);
                        }
                        int scoreIdx = b * _headCount * queryLen * keyLen + h * queryLen * keyLen + q * keyLen + k;
                        scoresSpan[scoreIdx] = NumOps.FromDouble(dot * scale);
                    }
                }
            }
        }

        return scores;
    }

    private Tensor<T> ApplySoftmax(Tensor<T> scores, int batch, int queryLen, int keyLen)
    {
        // Apply softmax along the last dimension (keyLen)
        var output = new Tensor<T>(scores.Shape);
        var inSpan = scores.AsSpan();
        var outSpan = output.AsWritableSpan();

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < _headCount; h++)
            {
                for (int q = 0; q < queryLen; q++)
                {
                    // Find max for numerical stability
                    double maxVal = double.MinValue;
                    int baseIdx = b * _headCount * queryLen * keyLen + h * queryLen * keyLen + q * keyLen;
                    for (int k = 0; k < keyLen; k++)
                    {
                        double val = NumOps.ToDouble(inSpan[baseIdx + k]);
                        if (val > maxVal) maxVal = val;
                    }

                    // Compute exp and sum
                    double expSum = 0.0;
                    for (int k = 0; k < keyLen; k++)
                    {
                        double exp = Math.Exp(NumOps.ToDouble(inSpan[baseIdx + k]) - maxVal);
                        outSpan[baseIdx + k] = NumOps.FromDouble(exp);
                        expSum += exp;
                    }

                    // Normalize
                    for (int k = 0; k < keyLen; k++)
                    {
                        outSpan[baseIdx + k] = NumOps.FromDouble(NumOps.ToDouble(outSpan[baseIdx + k]) / expSum);
                    }
                }
            }
        }

        return output;
    }

    private Tensor<T> ApplyAttentionToValues(Tensor<T> scores, Tensor<T> V, int batch, int queryLen)
    {
        // scores: [B, numHeads, queryLen, keyLen]
        // V: [B, numHeads, keyLen, headDim]
        // output: [B, numHeads, queryLen, headDim]
        int keyLen = V.Shape[2];
        var output = new Tensor<T>(new[] { batch, _headCount, queryLen, _headDim });
        var scoresSpan = scores.AsSpan();
        var vSpan = V.AsSpan();
        var outSpan = output.AsWritableSpan();

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < _headCount; h++)
            {
                for (int q = 0; q < queryLen; q++)
                {
                    for (int d = 0; d < _headDim; d++)
                    {
                        double sum = 0.0;
                        for (int k = 0; k < keyLen; k++)
                        {
                            int scoreIdx = b * _headCount * queryLen * keyLen + h * queryLen * keyLen + q * keyLen + k;
                            int vIdx = b * _headCount * keyLen * _headDim + h * keyLen * _headDim + k * _headDim + d;
                            sum += NumOps.ToDouble(scoresSpan[scoreIdx]) * NumOps.ToDouble(vSpan[vIdx]);
                        }
                        int outIdx = b * _headCount * queryLen * _headDim + h * queryLen * _headDim + q * _headDim + d;
                        outSpan[outIdx] = NumOps.FromDouble(sum);
                    }
                }
            }
        }

        return output;
    }

    private Tensor<T> AddBias(Tensor<T> input, Tensor<T> bias)
    {
        var shape = input.Shape;
        int batch = shape[0];
        int seqLen = shape[1];
        int dim = shape[2];

        var output = new Tensor<T>(shape);
        var inSpan = input.AsSpan();
        var biasSpan = bias.AsSpan();
        var outSpan = output.AsWritableSpan();

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int d = 0; d < dim; d++)
                {
                    int idx = b * seqLen * dim + s * dim + d;
                    outSpan[idx] = NumOps.Add(inSpan[idx], biasSpan[d]);
                }
            }
        }

        return output;
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

    private Tensor<T> TransposeWeights(Tensor<T> weights)
    {
        int rows = weights.Shape[0];
        int cols = weights.Shape[1];
        var output = new Tensor<T>(new[] { cols, rows });
        var inSpan = weights.AsSpan();
        var outSpan = output.AsWritableSpan();

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                outSpan[c * rows + r] = inSpan[r * cols + c];
            }
        }

        return output;
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



