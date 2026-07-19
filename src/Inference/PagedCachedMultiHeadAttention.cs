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
internal class PagedCachedMultiHeadAttention<T> : LayerBase<T>, IContextAwareInferenceLayer<T>
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

    // Weight matrices as [inDim, outDim] tensors for the batched-GEMM projection path (Engine.TensorMatMul,
    // which routes to the optimized BLAS/GPU kernels). Built lazily from the Matrix weights and invalidated
    // alongside the float kernel-weight caches when the weights change.
    private Tensor<T>? _wqTensor;
    private Tensor<T>? _wkTensor;
    private Tensor<T>? _wvTensor;
    private Tensor<T>? _woTensor;

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

        // Single-sequence legacy path: use the instance's SequenceId/position and advance it.
        var legacyOutput = ComputePagedAttention(input, new[] { SequenceId }, new[] { _currentPosition }, rowLengths: null);
        _currentPosition += input.Shape[1];
        _lastOutput = legacyOutput;
        return legacyOutput;
    }

    /// <inheritdoc/>
    public Tensor<T> ForwardWithContext(Tensor<T> input, InferenceForwardContext ctx)
    {
        if (!InferenceMode || Kernel == null)
        {
            throw new InvalidOperationException(
                $"{nameof(PagedCachedMultiHeadAttention<T>)}.ForwardWithContext requires inference mode with a paged KV kernel.");
        }

        // Concurrency-safe: sequence id + start position come from the per-call context; no shared
        // instance fields are mutated (no _currentPosition/_lastInput/_lastOutput writes), so many
        // sequences can drive one shared layer + KV cache at once, isolated by sequence id.
        // Batched context: batch row b belongs to SequenceIds[b] at Positions[b] (one batched decode step
        // across many sequences); otherwise a single sequence occupies the whole (batch=1) input.
        if (ctx.SequenceIds is { } batchSeqIds && ctx.Positions is { } batchPositions)
        {
            return ComputePagedAttention(input, batchSeqIds, batchPositions, ctx.RowLengths);
        }
        return ComputePagedAttention(input, new[] { ctx.SequenceId }, new[] { ctx.Position }, rowLengths: null);
    }

    private Tensor<T> ComputePagedAttention(Tensor<T> input, long[] seqIds, int[] basePositions, int[]? rowLengths)
    {
        // Inference mode: update cache and compute attention token-by-token. Supports both prefill
        // (seqLen>1 for one sequence) and BATCHED decode: batch row b belongs to seqIds[b] starting at
        // basePositions[b], so ONE forward serves many independent sequences over the shared paged cache
        // (the continuous-batching throughput win). Each row is isolated by its own sequence id.
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

        if (seqIds.Length != batchSize || basePositions.Length != batchSize)
        {
            throw new ArgumentException(
                $"seqIds/basePositions length must equal batch size {batchSize} (got {seqIds.Length}/{basePositions.Length}).");
        }
        if (rowLengths is not null && rowLengths.Length != batchSize)
        {
            throw new ArgumentException(
                $"rowLengths length must equal batch size {batchSize} (got {rowLengths.Length}).");
        }

        var output = new Tensor<T>([batchSize, seqLen, embDim]);

        // Callers (Forward / ForwardWithContext) only invoke this in inference mode with a kernel,
        // but capture a non-null local so the per-token kernel calls are flow-checked.
        var kernel = Kernel
            ?? throw new InvalidOperationException(
                $"{nameof(PagedCachedMultiHeadAttention<T>)}: paged KV kernel is not initialized.");

        var activation = ScalarActivation
            ?? throw new InvalidOperationException(
                $"{nameof(PagedCachedMultiHeadAttention<T>)}: ScalarActivation not initialized.");

        // Non-quantized models take the batched-GEMM projection path: Q/K/V/O for the whole
        // [batch, seqLen] block are projected in ONE matmul each (routed to the optimized GEMM kernels)
        // instead of a per-token matrix x vector. This is the continuous-batching compute win — a batched
        // decode of N sequences becomes one [N, embDim] x [embDim, projDim] GEMM rather than N mat-vecs.
        // Int8 weight-only quantized models keep the per-token fused kernel path below (its dequant-on-read
        // mat-vec has no batched-GEMM equivalent here yet).
        EnsureKernelWeightCache();
        bool useQuantizedPath = EnableWeightOnlyQuantization && typeof(T) == typeof(float)
            && _cachedWQInt8.HasValue && _cachedWKInt8.HasValue && _cachedWVInt8.HasValue && _cachedWOInt8.HasValue;
        // Batched-GEMM projection pays off only with more than one row (batched decode of several sequences,
        // or a multi-token prefill). A single-token single-sequence decode has one row, where the GEMM
        // dispatch/reshape overhead exceeds a direct mat-vec, so that case keeps the per-token path below.
        if (!useQuantizedPath && batchSize * seqLen > 1)
        {
            return ComputePagedAttentionBatched(input, seqIds, basePositions, rowLengths, kernel, activation);
        }

        // Materialize weights to float spans for the paged kernel.
        // Note: This is intentionally conservative and prioritizes correctness.
        // PagedAttentionKernel's MatVecMul expects matrices stored as [outDim, inDim] row-major.
        // Our weights are stored as [inDim, outDim], so we pass a transposed layout.
        EnsureKernelWeightCache();
        if (_cachedWQ is null || _cachedWK is null || _cachedWV is null || _cachedWO is null)
        {
            throw new InvalidOperationException(
                $"{nameof(PagedCachedMultiHeadAttention<T>)}: Kernel weight cache initialization failed.");
        }

        var wQ = _cachedWQ;
        var wK = _cachedWK;
        var wV = _cachedWV;
        var wO = _cachedWO;

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

            // Extract quantized weights once (guaranteed non-null when useQuantized is true)
            var qWeightsQ = useQuantized ? wQInt8.GetValueOrDefault() : default;
            var qWeightsK = useQuantized ? wKInt8.GetValueOrDefault() : default;
            var qWeightsV = useQuantized ? wVInt8.GetValueOrDefault() : default;
            var qWeightsO = useQuantized ? wOInt8.GetValueOrDefault() : default;

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

                // Each batch row is an INDEPENDENT sequence (seqIds[b]) at its own start position
                // (basePositions[b]); position advances locally per row, and the instance _currentPosition
                // is NOT touched here, so concurrent sequences do not collide on position state.
                for (int b = 0; b < batchSize; b++)
                {
                long sequenceId = seqIds[b];
                int position = basePositions[b];
                // Right-padded batched prefill: process only this row's valid tokens; the padded tail
                // (t >= rowLen) is skipped — no KV written, output stays zero, and its logits are unread.
                int rowLen = rowLengths is not null ? rowLengths[b] : seqLen;
                for (int t = 0; t < rowLen; t++)
                {
                    for (int d = 0; d < embDim; d++)
                    {
                        hidden[d] = Convert.ToSingle(input[b, t, d]);
                    }

                    if (_ropeLayer != null || _alibiLayer != null)
                    {
                        // Decompose the kernel call to inject RoPE/ALiBi.
                        // Step 1: Project Q, K, V
                        if (useQuantized)
                        {
                            MatVecMulInt8(hidden, qWeightsQ, querySpan);
                            MatVecMulInt8(hidden, qWeightsK, keySpan);
                            MatVecMulInt8(hidden, qWeightsV, valueSpan);
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
                            ApplyRoPEToSpan(querySpan, position);
                            ApplyRoPEToSpan(keySpan, position);
                        }

                        // Step 3: Update cache and compute attention via kernel
                        kernel.UpdateCache(keySpan, valueSpan, sequenceId, position, LayerIndex);
                        kernel.ComputeTiledPagedAttention(querySpan, sequenceId, LayerIndex, attnOutput,
                            1.0f / MathF.Sqrt(_headDimension),
                            alibiSlopes: alibiSlopes,
                            queryPosition: position);

                        // Step 4: Output projection
                        if (useQuantized)
                        {
                            MatVecMulInt8(attnOutput, qWeightsO, tokenOut);
                        }
                        else
                        {
                            MatVecMul(attnOutput, wO, tokenOut, projDim, embDim);
                        }
                    }
                    else if (useQuantized)
                    {
                        kernel.ForwardQuantized(
                            hiddenStates: hidden,
                            wQ: qWeightsQ,
                            wK: qWeightsK,
                            wV: qWeightsV,
                            wO: qWeightsO,
                            sequenceId: sequenceId,
                            position: position,
                            layer: LayerIndex,
                            output: tokenOut);
                    }
                    else
                    {
                        kernel.Forward(
                            hiddenStates: hidden,
                            wQ: wQ,
                            wK: wK,
                            wV: wV,
                            wO: wO,
                            sequenceId: sequenceId,
                            position: position,
                            layer: LayerIndex,
                            output: tokenOut);
                    }

                    // Add bias and activation.
                    for (int d = 0; d < embDim; d++)
                    {
                        T value = NumOps.FromDouble(tokenOut[d]);
                        value = NumOps.Add(value, _outputBias[d]);
                        output[b, t, d] = activation.Activate(value);
                    }

                    // Advance the LOCAL position (the per-call context owns position now). The legacy
                    // Forward path advances the instance _currentPosition itself after this returns.
                    position++;
                }
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

        return output;
    }

    /// <summary>
    /// Batched-GEMM paged attention for non-quantized models: projects Q/K/V for the whole
    /// [batch, seqLen] block in one matmul each, then walks each row's valid tokens in causal order to
    /// update the paged KV cache and read paged attention, and finally projects the output in one matmul.
    /// Numerically equivalent to the per-token path (same weights) but replaces batchSize x seqLen
    /// per-token mat-vecs with four batched GEMMs — the continuous-batching throughput win.
    /// </summary>
    private Tensor<T> ComputePagedAttentionBatched(
        Tensor<T> input, long[] seqIds, int[] basePositions, int[]? rowLengths,
        PagedAttentionKernel<T> kernel, IActivationFunction<T> activation)
    {
        int batchSize = input.Shape[0];
        int seqLen = input.Shape[1];
        int embDim = input.Shape[2];
        int projDim = _headCount * _headDimension;
        int rows = batchSize * seqLen;
        float scale = 1.0f / MathF.Sqrt(_headDimension);

        // Batched projections: ONE Engine GEMM each over the flattened [rows, embDim] block, routed to the
        // optimized BLAS/GPU kernels (NOT the generic-T Tensor.Multiply, which is a per-element NumOps loop).
        EnsureProjectionWeightTensors();
        var input2D = Engine.Reshape(input, new[] { rows, embDim });
        var q2D = Engine.TensorMatMul(input2D, _wqTensor!); // [rows, projDim]
        var k2D = Engine.TensorMatMul(input2D, _wkTensor!);
        var v2D = Engine.TensorMatMul(input2D, _wvTensor!);
        // Materialize the projected Q/K/V to flat row-major spans once so the causal per-token loop reads
        // them without per-element Tensor indexing (which would round-trip a GPU-resident GEMM result).
        var qFlat = q2D.AsSpan();
        var kFlat = k2D.AsSpan();
        var vFlat = v2D.AsSpan();

        // Pre-compute ALiBi slopes per head (unchanged from the per-token path).
        float[]? alibiSlopes = null;
        if (_alibiLayer != null)
        {
            alibiSlopes = new float[_headCount];
            for (int h = 0; h < _headCount; h++)
            {
                double exponent = -8.0 * (h + 1) / _headCount;
                alibiSlopes[h] = (float)Math.Pow(2.0, exponent);
            }
        }

        // Attention output accumulates into a flat [rows, projDim] buffer fed to the batched output GEMM.
        var attnData = new T[rows * projDim];
        var pool = ArrayPool<float>.Shared;
        var qBuf = pool.Rent(projDim);
        var kBuf = pool.Rent(projDim);
        var vBuf = pool.Rent(projDim);
        var aBuf = pool.Rent(projDim);
        try
        {
            var q = qBuf.AsSpan(0, projDim);
            var k = kBuf.AsSpan(0, projDim);
            var v = vBuf.AsSpan(0, projDim);
            var a = aBuf.AsSpan(0, projDim);

            // Each batch row is an INDEPENDENT sequence (seqIds[b]) starting at basePositions[b]. Tokens are
            // walked in causal order so the KV cache grows before each token's attention read; the padded
            // tail (t >= rowLen) is skipped (no KV written, its attn output stays zero).
            for (int b = 0; b < batchSize; b++)
            {
                long sequenceId = seqIds[b];
                int position = basePositions[b];
                int rowLen = rowLengths is not null ? rowLengths[b] : seqLen;
                for (int t = 0; t < rowLen; t++)
                {
                    int rowBase = (b * seqLen + t) * projDim;
                    for (int d = 0; d < projDim; d++)
                    {
                        q[d] = Convert.ToSingle(qFlat[rowBase + d]);
                        k[d] = Convert.ToSingle(kFlat[rowBase + d]);
                        v[d] = Convert.ToSingle(vFlat[rowBase + d]);
                    }

                    if (_ropeLayer != null)
                    {
                        ApplyRoPEToSpan(q, position);
                        ApplyRoPEToSpan(k, position);
                    }

                    kernel.UpdateCache(k, v, sequenceId, position, LayerIndex);
                    kernel.ComputeTiledPagedAttention(q, sequenceId, LayerIndex, a, scale,
                        alibiSlopes: alibiSlopes, queryPosition: position);

                    for (int d = 0; d < projDim; d++)
                    {
                        attnData[rowBase + d] = NumOps.FromDouble(a[d]);
                    }

                    position++;
                }
            }
        }
        finally
        {
            pool.Return(qBuf);
            pool.Return(kBuf);
            pool.Return(vBuf);
            pool.Return(aBuf);
        }

        // Batched output projection (one Engine GEMM) + bias + activation.
        var attn2D = new Tensor<T>(attnData, new[] { rows, projDim });
        var o2D = Engine.TensorMatMul(attn2D, _woTensor!); // [rows, embDim]
        var oFlat = o2D.AsSpan();
        var output = new Tensor<T>([batchSize, seqLen, embDim]);
        for (int b = 0; b < batchSize; b++)
        {
            int rowLen = rowLengths is not null ? rowLengths[b] : seqLen;
            for (int t = 0; t < rowLen; t++)
            {
                int rowBase = (b * seqLen + t) * embDim;
                for (int d = 0; d < embDim; d++)
                {
                    T value = NumOps.Add(oFlat[rowBase + d], _outputBias[d]);
                    output[b, t, d] = activation.Activate(value);
                }
            }
        }

        return output;
    }

    // Builds the [inDim, outDim] weight tensors used by the batched-GEMM projection path, once, from the
    // Matrix weights. Invalidated with the float kernel-weight caches when the weights change.
    private void EnsureProjectionWeightTensors()
    {
        if (_wqTensor is not null && _wkTensor is not null && _wvTensor is not null && _woTensor is not null)
        {
            return;
        }
        lock (_kernelWeightsLock)
        {
            _wqTensor ??= MatrixToTensor(_queryWeights);
            _wkTensor ??= MatrixToTensor(_keyWeights);
            _wvTensor ??= MatrixToTensor(_valueWeights);
            _woTensor ??= MatrixToTensor(_outputWeights);
        }
    }

    private static Tensor<T> MatrixToTensor(Matrix<T> m)
    {
        var t = new Tensor<T>([m.Rows, m.Columns]);
        for (int i = 0; i < m.Rows; i++)
        {
            for (int j = 0; j < m.Columns; j++)
            {
                t[i, j] = m[i, j];
            }
        }
        return t;
    }

    private Tensor<T> ForwardStateless(Tensor<T> input)
    {
        var activation = ScalarActivation
            ?? throw new InvalidOperationException(
                $"{nameof(PagedCachedMultiHeadAttention<T>)}: ScalarActivation not initialized.");

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
                    output[b, s, o] = activation.Activate(value);
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
            _wqTensor = null;
            _wkTensor = null;
            _wvTensor = null;
            _woTensor = null;
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

    public override void UpdateParameters(T learningRate)
    {
        throw new NotSupportedException($"{nameof(PagedCachedMultiHeadAttention<T>)} is intended for inference-time usage only.");
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
