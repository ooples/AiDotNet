using System.Buffers;
using AiDotNet.Enums;
using AiDotNet.Inference.PagedAttention;
using AiDotNet.Inference.Quantization;
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

    // Positional encoding
    private RotaryPositionalEncodingLayer<T>? _ropeLayer;
    private ALiBiPositionalBiasLayer<T>? _alibiLayer;

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

    /// <summary>
    /// Gets the positional encoding type used by this attention layer.
    /// </summary>
    public PositionalEncodingType PositionalEncoding { get; private set; } = PositionalEncodingType.None;

    /// <summary>
    /// Gets the RoPE base frequency (theta) if RoPE is configured.
    /// </summary>
    public double RoPETheta => _ropeLayer?.Theta ?? 10000.0;

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

    /// <summary>
    /// Configures positional encoding for this paged cached attention layer.
    /// </summary>
    /// <param name="encodingType">The type of positional encoding to use.</param>
    /// <param name="ropeTheta">Base frequency for RoPE (default: 10000.0).</param>
    /// <param name="maxSequenceLength">Maximum sequence length for pre-computation.</param>
    public void ConfigurePositionalEncoding(
        PositionalEncodingType encodingType,
        double ropeTheta = 10000.0,
        int maxSequenceLength = 2048)
    {
        PositionalEncoding = encodingType;
        _ropeLayer = null;
        _alibiLayer = null;

        switch (encodingType)
        {
            case PositionalEncodingType.Rotary:
                _ropeLayer = new RotaryPositionalEncodingLayer<T>(
                    maxSequenceLength, _headDimension, ropeTheta);
                break;
            case PositionalEncodingType.ALiBi:
                _alibiLayer = new ALiBiPositionalBiasLayer<T>(_headCount, maxSequenceLength);
                break;
            case PositionalEncodingType.None:
                break;
            default:
                throw new ArgumentException(
                    $"Unsupported positional encoding type for PagedCachedMultiHeadAttention: {encodingType}.",
                    nameof(encodingType));
        }
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

            var wQInt8 = _cachedWQInt8;
            var wKInt8 = _cachedWKInt8;
            var wVInt8 = _cachedWVInt8;
            var wOInt8 = _cachedWOInt8;

            bool useQuantized = EnableWeightOnlyQuantization &&
                                typeof(T) == typeof(float) &&
                                wQInt8.HasValue &&
                                wKInt8.HasValue &&
                                wVInt8.HasValue &&
                                wOInt8.HasValue;

            // Pre-compute ALiBi slopes per head for the paged attention path.
            float[]? alibiSlopes = null;
            if (_alibiLayer != null)
            {
                alibiSlopes = new float[_headCount];
                for (int h = 0; h < _headCount; h++)
                {
                    // ALiBi slope for head h: 2^(-8h/H) where H is total heads
                    double exponent = -8.0 * (h + 1) / _headCount;
                    alibiSlopes[h] = (float)Math.Pow(2.0, exponent);
                }
            }

            int projDim = _headCount * _headDimension;
            var queryBuf = pool.Rent(projDim);
            var keyBuf = pool.Rent(projDim);
            var valueBuf = pool.Rent(projDim);
            var attnBuf = pool.Rent(projDim);

            try
            {
                var querySpan = queryBuf.AsSpan(0, projDim);
                var keySpan = keyBuf.AsSpan(0, projDim);
                var valueSpan = valueBuf.AsSpan(0, projDim);
                var attnOutput = attnBuf.AsSpan(0, projDim);

                for (int t = 0; t < seqLen; t++)
                {
                    for (int d = 0; d < embDim; d++)
                    {
                        hidden[d] = Convert.ToSingle(input[0, t, d]);
                    }

                    if (_ropeLayer != null || _alibiLayer != null)
                    {
                        // Decompose the kernel call to inject RoPE/ALiBi.
                        // Step 1: Project Q, K, V
                        if (useQuantized)
                        {
                            MatVecMulInt8(hidden, wQInt8!.Value, querySpan);
                            MatVecMulInt8(hidden, wKInt8!.Value, keySpan);
                            MatVecMulInt8(hidden, wVInt8!.Value, valueSpan);
                        }
                        else
                        {
                            MatVecMul(hidden, wQ, querySpan, embDim, projDim);
                            MatVecMul(hidden, wK, keySpan, embDim, projDim);
                            MatVecMul(hidden, wV, valueSpan, embDim, projDim);
                        }

                        // Step 2: Apply RoPE to Q and K
                        if (_ropeLayer != null)
                        {
                            ApplyRoPEToSpan(querySpan, _currentPosition);
                            ApplyRoPEToSpan(keySpan, _currentPosition);
                        }

                        // Step 3: Update cache and compute attention via kernel
                        Kernel.UpdateCache(keySpan, valueSpan, SequenceId, _currentPosition, LayerIndex);
                        Kernel.ComputeTiledPagedAttention(querySpan, SequenceId, LayerIndex, attnOutput,
                            1.0f / MathF.Sqrt(_headDimension));

                        // Step 4: Output projection
                        if (useQuantized)
                        {
                            MatVecMulInt8(attnOutput, wOInt8!.Value, tokenOut);
                        }
                        else
                        {
                            MatVecMul(attnOutput, wO, tokenOut, projDim, embDim);
                        }
                    }
                    else if (useQuantized)
                    {
                        Kernel.ForwardQuantized(
                            hiddenStates: hidden,
                            wQ: wQInt8!.Value,
                            wK: wKInt8!.Value,
                            wV: wVInt8!.Value,
                            wO: wOInt8!.Value,
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
                pool.Return(queryBuf);
                pool.Return(keyBuf);
                pool.Return(valueBuf);
                pool.Return(attnBuf);
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

        // Apply RoPE to Q and K if configured (position starts at 0 for standard forward)
        if (_ropeLayer != null)
        {
            (qh, kh) = _ropeLayer.ApplyRoPE(qh, kh, startPosition: 0);
        }

        // Compute ALiBi bias if configured
        int seqLen = input.Shape[1];
        Tensor<T>? aliBiBias = _alibiLayer?.ComputeBias(seqLen, seqLen, _useCausalMask);

        var (attn, _) = FlashAttention<T>.Forward(qh, kh, vh, _flashConfig, attentionBias: aliBiBias);

        // Merge heads back to [B, S, E]
        var merged = MergeHeads(attn);

        // Output projection + bias + activation.
        // Use the tensor/matrix multiply path to leverage optimized kernels.
        var projected = merged.Multiply(_outputWeights);

        int batch = projected.Shape[0];
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

    /// <summary>
    /// Applies RoPE rotation to a projected Q or K span in-place.
    /// The span contains [numHeads * headDim] floats, laid out as [head0_d0..head0_dH, head1_d0..head1_dH, ...].
    /// </summary>
    private void ApplyRoPEToSpan(Span<float> projected, int position)
    {
        if (_ropeLayer == null) return;

        double theta = _ropeLayer.Theta;

        for (int h = 0; h < _headCount; h++)
        {
            int offset = h * _headDimension;
            for (int d = 0; d < _headDimension / 2; d++)
            {
                double freq = 1.0 / Math.Pow(theta, 2.0 * d / _headDimension);
                double angle = position * freq;
                float cos = (float)Math.Cos(angle);
                float sin = (float)Math.Sin(angle);

                float x0 = projected[offset + 2 * d];
                float x1 = projected[offset + 2 * d + 1];
                projected[offset + 2 * d] = x0 * cos - x1 * sin;
                projected[offset + 2 * d + 1] = x0 * sin + x1 * cos;
            }
        }
    }

    private static void MatVecMul(ReadOnlySpan<float> vec, ReadOnlySpan<float> mat, Span<float> output, int inDim, int outDim)
    {
        output.Clear();
        for (int i = 0; i < outDim; i++)
        {
            float sum = 0;
            int rowOffset = i * inDim;
            for (int j = 0; j < inDim; j++)
            {
                sum += vec[j] * mat[rowOffset + j];
            }
            output[i] = sum;
        }
    }

    private static void MatVecMulInt8(ReadOnlySpan<float> vec, in Int8WeightOnlyQuantization.QuantizedWeights mat, Span<float> output)
    {
        int rows = mat.Rows;
        int cols = mat.Cols;
        var weights = mat.Weights;
        var scales = mat.Scales;

        for (int r = 0; r < rows; r++)
        {
            int baseIdx = r * cols;
            float sum = 0f;
            for (int c = 0; c < cols; c++)
            {
                sum += weights[baseIdx + c] * vec[c];
            }
            output[r] = sum * scales[r];
        }
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
        bool enableQuantization = EnableWeightOnlyQuantization && typeof(T) == typeof(float);

        bool hasDenseWeights = _cachedWQ != null && _cachedWK != null && _cachedWV != null && _cachedWO != null;
        bool hasQuantizedWeights = _cachedWQInt8.HasValue && _cachedWKInt8.HasValue && _cachedWVInt8.HasValue && _cachedWOInt8.HasValue;

        if (hasDenseWeights && (!enableQuantization || hasQuantizedWeights))
        {
            return;
        }

        float[]? localWQ = null;
        float[]? localWK = null;
        float[]? localWV = null;
        float[]? localWO = null;

        Int8WeightOnlyQuantization.QuantizedWeights? localWQInt8 = null;
        Int8WeightOnlyQuantization.QuantizedWeights? localWKInt8 = null;
        Int8WeightOnlyQuantization.QuantizedWeights? localWVInt8 = null;
        Int8WeightOnlyQuantization.QuantizedWeights? localWOInt8 = null;

        // First, determine what's missing and take a quick snapshot inside the lock.
        lock (_kernelWeightsLock)
        {
            hasDenseWeights = _cachedWQ != null && _cachedWK != null && _cachedWV != null && _cachedWO != null;
            hasQuantizedWeights = _cachedWQInt8.HasValue && _cachedWKInt8.HasValue && _cachedWVInt8.HasValue && _cachedWOInt8.HasValue;

            if (hasDenseWeights && (!enableQuantization || hasQuantizedWeights))
            {
                return;
            }

            if (_cachedWQ == null) localWQ = MatrixToFloatForKernel(_queryWeights);
            if (_cachedWK == null) localWK = MatrixToFloatForKernel(_keyWeights);
            if (_cachedWV == null) localWV = MatrixToFloatForKernel(_valueWeights);
            if (_cachedWO == null) localWO = MatrixToFloatForKernel(_outputWeights);

            if (!enableQuantization)
            {
                _cachedWQInt8 = null;
                _cachedWKInt8 = null;
                _cachedWVInt8 = null;
                _cachedWOInt8 = null;
            }
        }

        // Compute expensive quantization outside the lock to minimize contention.
        if (enableQuantization)
        {
            int projDim = _headCount * _headDimension;
            int hiddenDim = _embeddingDimension;

            var wq = _cachedWQ ?? localWQ;
            var wk = _cachedWK ?? localWK;
            var wv = _cachedWV ?? localWV;
            var wo = _cachedWO ?? localWO;

            if (wq != null && wk != null && wv != null && wo != null)
            {
                localWQInt8 = Int8WeightOnlyQuantization.QuantizePerRow(wq, projDim, hiddenDim);
                localWKInt8 = Int8WeightOnlyQuantization.QuantizePerRow(wk, projDim, hiddenDim);
                localWVInt8 = Int8WeightOnlyQuantization.QuantizePerRow(wv, projDim, hiddenDim);
                localWOInt8 = Int8WeightOnlyQuantization.QuantizePerRow(wo, hiddenDim, projDim);
            }
        }

        // Publish results under lock (double-checked to avoid overwriting).
        lock (_kernelWeightsLock)
        {
            _cachedWQ ??= localWQ;
            _cachedWK ??= localWK;
            _cachedWV ??= localWV;
            _cachedWO ??= localWO;

            if (enableQuantization)
            {
                _cachedWQInt8 ??= localWQInt8;
                _cachedWKInt8 ??= localWKInt8;
                _cachedWVInt8 ??= localWVInt8;
                _cachedWOInt8 ??= localWOInt8;
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
        _ropeLayer?.ResetState();
        _alibiLayer?.ResetState();
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
            ["EnableWeightOnlyQuantization"] = EnableWeightOnlyQuantization.ToString(),
            ["PositionalEncoding"] = PositionalEncoding.ToString()
        };
    }
}
