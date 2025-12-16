using AiDotNet.Inference.PagedAttention;
using AiDotNet.NeuralNetworks.Attention;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Inference;

/// <summary>
/// Multi-head attention layer backed by PagedKVCache for efficient multi-sequence inference.
/// </summary>
/// <remarks>
/// This layer is intended for inference-time usage. When <see cref="InferenceMode"/> is enabled
/// and a <see cref="Kernel"/> is attached, it uses PagedKVCache to avoid reallocations and
/// allow many independent sequences to grow efficiently.
/// </remarks>
internal class PagedCachedMultiHeadAttention<T> : LayerBase<T>, AiDotNet.NeuralNetworks.Layers.ILayerSerializationMetadata
{
    private readonly int _headCount;
    private readonly int _headDimension;
    private readonly int _embeddingDimension;
    private readonly bool _useCausalMask;

    private Matrix<T> _queryWeights;
    private Matrix<T> _keyWeights;
    private Matrix<T> _valueWeights;
    private Matrix<T> _outputWeights;
    private Vector<T> _outputBias;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private int _currentPosition;

    private readonly FlashAttentionConfig _flashConfig;

    /// <summary>
    /// Gets whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the number of attention heads.
    /// </summary>
    public int HeadCount => _headCount;

    /// <summary>
    /// Gets the dimension of each attention head.
    /// </summary>
    public int HeadDimension => _headDimension;

    /// <summary>
    /// Gets or sets the layer index for KV-cache addressing.
    /// </summary>
    public int LayerIndex { get; set; }

    /// <summary>
    /// Gets or sets whether the layer is in inference mode (uses paged cache).
    /// </summary>
    public bool InferenceMode { get; set; }

    /// <summary>
    /// Gets or sets the PagedAttention kernel (owns the paged cache).
    /// </summary>
    public PagedAttentionKernel<T>? Kernel { get; set; }

    /// <summary>
    /// Gets or sets the sequence ID used for this layer's cache operations.
    /// </summary>
    public long SequenceId { get; set; }

    public PagedCachedMultiHeadAttention(
        int sequenceLength,
        int embeddingDimension,
        int headCount,
        bool useCausalMask,
        IActivationFunction<T>? activationFunction = null)
        : base(
            [sequenceLength, embeddingDimension],
            [sequenceLength, embeddingDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
        if (embeddingDimension % headCount != 0)
        {
            throw new ArgumentException(
                $"Embedding dimension ({embeddingDimension}) must be divisible by head count ({headCount}).",
                nameof(headCount));
        }

        _embeddingDimension = embeddingDimension;
        _headCount = headCount;
        _headDimension = embeddingDimension / headCount;
        _useCausalMask = useCausalMask;

        _queryWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _keyWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _valueWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _outputWeights = new Matrix<T>(embeddingDimension, embeddingDimension);
        _outputBias = new Vector<T>(embeddingDimension);

        _flashConfig = FlashAttentionConfig.Default;
        _flashConfig.UseCausalMask = useCausalMask;
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        if (!InferenceMode || Kernel == null)
        {
            var statelessOutput = ForwardStateless(input);
            _lastOutput = statelessOutput;
            return statelessOutput;
        }

        // Inference mode: update cache and compute attention token-by-token.
        // This supports both prefill (seqLen>1) and decode (seqLen==1) by iterating tokens.
        if (input.Shape.Length < 3)
        {
            throw new ArgumentException("Expected input shape [batch, seqLen, embeddingDim].", nameof(input));
        }

        int batchSize = input.Shape[0];
        int seqLen = input.Shape[1];
        int embDim = input.Shape[2];

        if (embDim != _embeddingDimension)
        {
            throw new ArgumentException($"Expected embeddingDim={_embeddingDimension}, got {embDim}.", nameof(input));
        }

        if (batchSize != 1)
        {
            // PagedAttentionKernel supports batched attention, but this layer's state model is per-sequence.
            // Keep it strict for now to avoid cache mixing.
            throw new NotSupportedException("PagedCachedMultiHeadAttention currently supports batchSize==1 per sequence.");
        }

        var output = new Tensor<T>([batchSize, seqLen, embDim]);

        // Materialize weights to float spans for the paged kernel.
        // Note: This is intentionally conservative and prioritizes correctness.
        // PagedAttentionKernel's MatVecMul expects matrices stored as [outDim, inDim] row-major.
        // Our weights are stored as [inDim, outDim], so we pass a transposed layout.
        var wQ = MatrixToFloatForKernel(_queryWeights);
        var wK = MatrixToFloatForKernel(_keyWeights);
        var wV = MatrixToFloatForKernel(_valueWeights);
        var wO = MatrixToFloatForKernel(_outputWeights);

        // Process each token sequentially to ensure causal behavior during prefill.
        for (int t = 0; t < seqLen; t++)
        {
            var hidden = new float[embDim];
            for (int d = 0; d < embDim; d++)
            {
                hidden[d] = Convert.ToSingle(input[0, t, d]);
            }

            var tokenOut = new float[embDim];
            Kernel.Forward(
                hiddenStates: hidden.AsSpan(),
                wQ: wQ,
                wK: wK,
                wV: wV,
                wO: wO,
                sequenceId: SequenceId,
                position: _currentPosition,
                layer: LayerIndex,
                output: tokenOut.AsSpan());
            _currentPosition++;

            // Add bias and activation.
            for (int d = 0; d < embDim; d++)
            {
                T value = NumOps.FromDouble(tokenOut[d]);
                value = NumOps.Add(value, _outputBias[d]);
                output[0, t, d] = ScalarActivation!.Activate(value);
            }
        }

        _lastOutput = output;
        return output;
    }

    private Tensor<T> ForwardStateless(Tensor<T> input)
    {
        // Stateless fallback using FlashAttention.
        // Compute Q,K,V projections.
        var (q, k, v) = ComputeQkv(input);

        // FlashAttention expects [B, H, S, D]
        var qh = SplitHeads(q);
        var kh = SplitHeads(k);
        var vh = SplitHeads(v);

        var (attn, _) = FlashAttention<T>.Forward(qh, kh, vh, _flashConfig);

        // Merge heads back to [B, S, E]
        var merged = MergeHeads(attn);

        // Output projection + bias + activation
        int batch = merged.Shape[0];
        int seqLen = merged.Shape[1];
        var output = new Tensor<T>([batch, seqLen, _embeddingDimension]);

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int o = 0; o < _embeddingDimension; o++)
                {
                    T sum = NumOps.Zero;
                    for (int i = 0; i < _embeddingDimension; i++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(merged[b, s, i], _outputWeights[i, o]));
                    }

                    sum = NumOps.Add(sum, _outputBias[o]);
                    output[b, s, o] = ScalarActivation!.Activate(sum);
                }
            }
        }

        return output;
    }

    private (Tensor<T> Q, Tensor<T> K, Tensor<T> V) ComputeQkv(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int seqLen = input.Shape[1];
        int embDim = input.Shape[2];

        var q = new Tensor<T>([batchSize, seqLen, embDim]);
        var k = new Tensor<T>([batchSize, seqLen, embDim]);
        var v = new Tensor<T>([batchSize, seqLen, embDim]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int o = 0; o < embDim; o++)
                {
                    T sumQ = NumOps.Zero;
                    T sumK = NumOps.Zero;
                    T sumV = NumOps.Zero;
                    for (int i = 0; i < embDim; i++)
                    {
                        var x = input[b, s, i];
                        sumQ = NumOps.Add(sumQ, NumOps.Multiply(x, _queryWeights[i, o]));
                        sumK = NumOps.Add(sumK, NumOps.Multiply(x, _keyWeights[i, o]));
                        sumV = NumOps.Add(sumV, NumOps.Multiply(x, _valueWeights[i, o]));
                    }

                    q[b, s, o] = sumQ;
                    k[b, s, o] = sumK;
                    v[b, s, o] = sumV;
                }
            }
        }

        return (q, k, v);
    }

    private Tensor<T> SplitHeads(Tensor<T> x)
    {
        int batchSize = x.Shape[0];
        int seqLen = x.Shape[1];
        var reshaped = new Tensor<T>([batchSize, _headCount, seqLen, _headDimension]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < _headCount; h++)
                {
                    int baseOffset = h * _headDimension;
                    for (int d = 0; d < _headDimension; d++)
                    {
                        reshaped[b, h, s, d] = x[b, s, baseOffset + d];
                    }
                }
            }
        }

        return reshaped;
    }

    private Tensor<T> MergeHeads(Tensor<T> x)
    {
        int batchSize = x.Shape[0];
        int seqLen = x.Shape[2];
        var merged = new Tensor<T>([batchSize, seqLen, _embeddingDimension]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < _headCount; h++)
                {
                    int baseOffset = h * _headDimension;
                    for (int d = 0; d < _headDimension; d++)
                    {
                        merged[b, s, baseOffset + d] = x[b, h, s, d];
                    }
                }
            }
        }

        return merged;
    }

    private static float[] MatrixToFloatForKernel(Matrix<T> matrix)
    {
        int inDim = matrix.Rows;
        int outDim = matrix.Columns;
        var data = new float[outDim * inDim];

        for (int o = 0; o < outDim; o++)
        {
            int rowOffset = o * inDim;
            for (int i = 0; i < inDim; i++)
            {
                data[rowOffset + i] = Convert.ToSingle(matrix[i, o]);
            }
        }

        return data;
    }

    public override Vector<T> GetParameters()
    {
        int totalParams = _queryWeights.Rows * _queryWeights.Columns * 4 + _outputBias.Length;
        var parameters = new Vector<T>(totalParams);
        int index = 0;

        foreach (var matrix in new[] { _queryWeights, _keyWeights, _valueWeights, _outputWeights })
        {
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    parameters[index++] = matrix[i, j];
                }
            }
        }

        for (int i = 0; i < _outputBias.Length; i++)
        {
            parameters[index++] = _outputBias[i];
        }

        return parameters;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int expectedParams = _queryWeights.Rows * _queryWeights.Columns * 4 + _outputBias.Length;
        if (parameters.Length != expectedParams)
        {
            throw new ArgumentException($"Expected {expectedParams} parameters, got {parameters.Length}");
        }

        int index = 0;

        foreach (var matrix in new[] { _queryWeights, _keyWeights, _valueWeights, _outputWeights })
        {
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    matrix[i, j] = parameters[index++];
                }
            }
        }

        for (int i = 0; i < _outputBias.Length; i++)
        {
            _outputBias[i] = parameters[index++];
        }
    }

    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _currentPosition = 0;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        throw new NotSupportedException($"{nameof(PagedCachedMultiHeadAttention<T>)} is intended for inference-time usage only.");
    }

    public override void UpdateParameters(T learningRate)
    {
        throw new NotSupportedException($"{nameof(PagedCachedMultiHeadAttention<T>)} is intended for inference-time usage only.");
    }

    public override bool SupportsJitCompilation => false;

    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException($"{nameof(PagedCachedMultiHeadAttention<T>)} does not support JIT compilation.");
    }

    Dictionary<string, string> AiDotNet.NeuralNetworks.Layers.ILayerSerializationMetadata.GetSerializationMetadata()
    {
        return new Dictionary<string, string>
        {
            ["HeadCount"] = _headCount.ToString(),
            ["UseCausalMask"] = _useCausalMask.ToString()
        };
    }
}
