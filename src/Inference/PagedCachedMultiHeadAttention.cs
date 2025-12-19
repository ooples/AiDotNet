using AiDotNet.Inference.PagedAttention;
using AiDotNet.NeuralNetworks.Attention;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using System.Buffers;
using AiDotNet.Inference.Quantization;

namespace AiDotNet.Inference;

/// <summary>
/// Multi-head attention layer backed by PagedKVCache for efficient multi-sequence inference.
/// </summary>
/// <remarks>
/// This layer is intended for inference-time usage. When <see cref="InferenceMode"/> is enabled
/// and a <see cref="Kernel"/> is attached, it uses PagedKVCache to avoid reallocations and
/// allow many independent sequences to grow efficiently.
/// <para>
/// <b>Limitation:</b> This layer currently supports <c>batchSize == 1</c> per sequence to avoid cache mixing.
/// For concurrent serving, create one sequence per request (distinct <see cref="SequenceId"/> values).
/// </para>
/// </remarks>
internal class PagedCachedMultiHeadAttention<T> : LayerBase<T>
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

    private readonly object _kernelWeightsLock = new();
    private float[]? _cachedWQ;
    private float[]? _cachedWK;
    private float[]? _cachedWV;
    private float[]? _cachedWO;
    private Int8WeightOnlyQuantization.QuantizedWeights? _cachedWQInt8;
    private Int8WeightOnlyQuantization.QuantizedWeights? _cachedWKInt8;
    private Int8WeightOnlyQuantization.QuantizedWeights? _cachedWVInt8;
    private Int8WeightOnlyQuantization.QuantizedWeights? _cachedWOInt8;

    internal bool EnableWeightOnlyQuantization { get; set; }

    /// <summary>
    /// Gets whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => false;

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
        EnsureKernelWeightCache();
        var wQ = _cachedWQ!;
        var wK = _cachedWK!;
        var wV = _cachedWV!;
        var wO = _cachedWO!;

        // Process each token sequentially to ensure causal behavior during prefill.
        var pool = ArrayPool<float>.Shared;
        var hiddenBuffer = pool.Rent(embDim);
        var tokenOutBuffer = pool.Rent(embDim);

        try
        {
            var hidden = hiddenBuffer.AsSpan(0, embDim);
            var tokenOut = tokenOutBuffer.AsSpan(0, embDim);

            for (int t = 0; t < seqLen; t++)
            {
                for (int d = 0; d < embDim; d++)
                {
                    hidden[d] = Convert.ToSingle(input[0, t, d]);
                }

                if (EnableWeightOnlyQuantization &&
                    typeof(T) == typeof(float) &&
                    _cachedWQInt8.HasValue &&
                    _cachedWKInt8.HasValue &&
                    _cachedWVInt8.HasValue &&
                    _cachedWOInt8.HasValue)
                {
                    Kernel.ForwardQuantized(
                        hiddenStates: hidden,
                        wQ: _cachedWQInt8.Value,
                        wK: _cachedWKInt8.Value,
                        wV: _cachedWVInt8.Value,
                        wO: _cachedWOInt8.Value,
                        sequenceId: SequenceId,
                        position: _currentPosition,
                        layer: LayerIndex,
                        output: tokenOut);
                }
                else
                {
                    Kernel.Forward(
                        hiddenStates: hidden,
                        wQ: wQ,
                        wK: wK,
                        wV: wV,
                        wO: wO,
                        sequenceId: SequenceId,
                        position: _currentPosition,
                        layer: LayerIndex,
                        output: tokenOut);
                }

                // Add bias and activation.
                for (int d = 0; d < embDim; d++)
                {
                    T value = NumOps.FromDouble(tokenOut[d]);
                    value = NumOps.Add(value, _outputBias[d]);
                    output[0, t, d] = ScalarActivation!.Activate(value);
                }

                _currentPosition++;
            }
        }
        finally
        {
            pool.Return(hiddenBuffer);
            pool.Return(tokenOutBuffer);
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

        // Output projection + bias + activation.
        // Use the tensor/matrix multiply path to leverage optimized kernels.
        var projected = merged.Multiply(_outputWeights);

        int batch = projected.Shape[0];
        int seqLen = projected.Shape[1];
        var output = new Tensor<T>([batch, seqLen, _embeddingDimension]);

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int o = 0; o < _embeddingDimension; o++)
                {
                    T value = NumOps.Add(projected[b, s, o], _outputBias[o]);
                    output[b, s, o] = ScalarActivation!.Activate(value);
                }
            }
        }

        return output;
    }

    private (Tensor<T> Q, Tensor<T> K, Tensor<T> V) ComputeQkv(Tensor<T> input)
    {
        // Use the tensor/matrix multiply path to leverage optimized kernels.
        var q = input.Multiply(_queryWeights);
        var k = input.Multiply(_keyWeights);
        var v = input.Multiply(_valueWeights);
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

        InvalidateKernelWeightCache();
    }

    private void EnsureKernelWeightCache()
    {
        if (_cachedWQ != null && _cachedWK != null && _cachedWV != null && _cachedWO != null)
        {
            return;
        }

        lock (_kernelWeightsLock)
        {
            _cachedWQ ??= MatrixToFloatForKernel(_queryWeights);
            _cachedWK ??= MatrixToFloatForKernel(_keyWeights);
            _cachedWV ??= MatrixToFloatForKernel(_valueWeights);
            _cachedWO ??= MatrixToFloatForKernel(_outputWeights);

            if (EnableWeightOnlyQuantization && typeof(T) == typeof(float))
            {
                int projDim = _headCount * _headDimension;
                int hiddenDim = _embeddingDimension;

                _cachedWQInt8 = Int8WeightOnlyQuantization.QuantizePerRow(_cachedWQ, projDim, hiddenDim);
                _cachedWKInt8 = Int8WeightOnlyQuantization.QuantizePerRow(_cachedWK, projDim, hiddenDim);
                _cachedWVInt8 = Int8WeightOnlyQuantization.QuantizePerRow(_cachedWV, projDim, hiddenDim);
                _cachedWOInt8 = Int8WeightOnlyQuantization.QuantizePerRow(_cachedWO, hiddenDim, projDim);
            }
            else
            {
                _cachedWQInt8 = null;
                _cachedWKInt8 = null;
                _cachedWVInt8 = null;
                _cachedWOInt8 = null;
            }
        }
    }

    private void InvalidateKernelWeightCache()
    {
        lock (_kernelWeightsLock)
        {
            _cachedWQ = null;
            _cachedWK = null;
            _cachedWV = null;
            _cachedWO = null;
            _cachedWQInt8 = null;
            _cachedWKInt8 = null;
            _cachedWVInt8 = null;
            _cachedWOInt8 = null;
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

    internal override Dictionary<string, string> GetMetadata()
    {
        return new Dictionary<string, string>
        {
            ["HeadCount"] = _headCount.ToString(),
            ["UseCausalMask"] = _useCausalMask.ToString(),
            ["EnableWeightOnlyQuantization"] = EnableWeightOnlyQuantization.ToString()
        };
    }
}
