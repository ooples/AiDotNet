using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Attention;

/// <summary>
/// Memory-efficient scaled dot-product attention. Forward is a thin bridge over
/// the Tensors-side fused-attention engine (<see
/// cref="AiDotNet.Tensors.Engines.Autodiff.FusedAttention{T}"/>).
/// </summary>
/// <remarks>
/// <para>
/// Historically this class held a ~900-line hand-rolled tiled-online-softmax
/// implementation in parallel to the Tensors-side <c>FusedAttention&lt;T&gt;</c>.
/// The two diverged in micro-detail (precision handling, bias plumbing, KV-cache
/// query offset semantics) and the framework version was significantly slower —
/// it built its dot products via <c>Engine.DotProduct</c> in scalar inner loops
/// while the Tensors version dispatches through the vectorized SDPA / fused-QKV
/// kernels (Tensors #485 6.9× alloc reduction + 2.7× mean latency, #488 fused
/// QKV + transpose-fused SDPA, #472 <c>Float32Precision</c> opt-in for
/// <c>FusedAttention&lt;double&gt;</c> running the kernel internally in float32).
/// Replacing the body with a bridge eliminates the duplication, the confusion
/// over which path is "the real one," and the perf gap. All ~10 callers
/// (FlashAttentionLayer, MultiHeadAttentionLayer, GroupedQueryAttentionLayer,
/// QuantizedAttentionLayer, T5RelativeBiasAttentionLayer, CachedMultiHeadAttention,
/// CachedGroupedQueryAttention, PagedCachedMultiHeadAttention, ...) keep the
/// same Forward signature; the old Backward had zero callers and is dropped.
/// </para>
/// <para>
/// For <c>T == double</c> the bridge enables <c>FlashAttentionConfig.Float32Precision = true</c>
/// by default so the FP64 attention runs internally on float32 (Tensors #472) —
/// the SD-UNet FP64 attention bottleneck called out in #1305 / PR #1456.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for computations (typically float or double).</typeparam>
internal static class FlashAttention<T>
{
    /// <summary>
    /// Computes <c>softmax(Q · Kᵀ / √d) · V</c> without materializing the full
    /// attention matrix. Delegates to
    /// <c>AiDotNet.Tensors.Engines.Autodiff.FusedAttention&lt;T&gt;.Forward(...)</c>.
    /// </summary>
    /// <param name="query">[batch, seq, dim] or [batch, heads, seq, dim].</param>
    /// <param name="key">Same rank as <paramref name="query"/>.</param>
    /// <param name="value">Same rank as <paramref name="query"/>.</param>
    /// <param name="config">
    /// Framework <see cref="FlashAttentionConfig"/>. Mapped to the Tensors-side
    /// <see cref="AiDotNet.Tensors.Engines.Autodiff.FlashAttentionConfig"/> on the
    /// way through. Null means <see cref="FlashAttentionConfig.Default"/>.
    /// </param>
    /// <param name="queryOffset">
    /// KV-cache offset — when <paramref name="query"/> is a window into a longer
    /// KV sequence (autoregressive decode), the causal mask shifts by this much.
    /// Passed through to <see cref="AiDotNet.Tensors.Engines.Autodiff.FlashAttentionConfig.QueryOffset"/>.
    /// </param>
    /// <param name="attentionBias">
    /// Additive bias broadcast onto the attention scores (ALiBi, relative-pos
    /// encodings, custom masks). Passed through unchanged.
    /// </param>
    public static (Tensor<T> Output, Tensor<T>? AttentionWeights) Forward(
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        FlashAttentionConfig? config = null,
        int queryOffset = 0,
        Tensor<T>? attentionBias = null)
    {
        config ??= FlashAttentionConfig.Default;
        ValidateInputs(query, key, value);

        bool is4D = query.Shape.Length == 4;
        int seqLenQ = is4D ? query.Shape[2] : query.Shape[1];
        int seqLenKV = is4D ? key.Shape[2] : key.Shape[1];
        // A noncausal cross-attention query is not a window into the key sequence:
        // learned queries may outnumber memory tokens. Preserve window bounds for causal
        // attention or an explicit nonzero offset; subtraction prevents overflow.
        if (queryOffset < 0 || ((config.UseCausalMask || queryOffset != 0) && queryOffset > seqLenKV - seqLenQ))
        {
            throw new ArgumentOutOfRangeException(
                nameof(queryOffset),
                $"queryOffset ({queryOffset}) must be nonnegative and, for causal attention or a nonzero offset, queryOffset + seqLenQ ({seqLenQ}) must not exceed seqLenKV ({seqLenKV}).");
        }

        if (!config.UseCausalMask && queryOffset == 0 && seqLenQ > seqLenKV)
        {
            // The fused engine treats every query block as a WINDOW into the key sequence and rejects
            // seqQ > seqKV for every configuration, causal or not. Noncausal cross-attention whose
            // learned queries outnumber the memory tokens is a legitimate shape — the MGIE edit mapper
            // reads its query embeddings off a shorter joint context — so compute it exactly here
            // instead of failing.
            return RectangularAttention(query, key, value, config, attentionBias, is4D);
        }

        var tensorsConfig = MapConfig(config, queryOffset);
        return AiDotNet.Tensors.Engines.Autodiff.FusedAttention<T>.Forward(
            query, key, value, tensorsConfig, attentionBias);
    }

    /// <summary>
    /// Exact attention for the one shape the fused kernel refuses: more queries than keys, with no
    /// causal mask and no query offset. This materializes the [batch, heads, seqQ, seqKV] scores —
    /// the very cost the fused path exists to avoid — which is acceptable precisely because this
    /// branch only runs when the key axis is the SHORTER one.
    /// </summary>
    private static (Tensor<T> Output, Tensor<T>? AttentionWeights) RectangularAttention(
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        FlashAttentionConfig config,
        Tensor<T>? attentionBias,
        bool is4D)
    {
        var engine = AiDotNetEngine.Current;
        var queries4D = is4D ? query : PromoteToFourD(engine, query);
        var keys4D = is4D ? key : PromoteToFourD(engine, key);
        var values4D = is4D ? value : PromoteToFourD(engine, value);
        double scale = config.ScaleFactor ?? 1.0 / Math.Sqrt(queries4D.Shape[3]);

        Tensor<T> output;
        Tensor<T>? weights;
        if (attentionBias is null)
        {
            output = engine.ScaledDotProductAttention(
                queries4D, keys4D, values4D, mask: null, scale: scale, out var produced);
            weights = config.ReturnAttentionWeights ? produced : null;
        }
        else
        {
            // The engine's attention takes a boolean mask, not an additive bias, so the biased case
            // walks the three ops itself: scores, softmax, and the value average.
            var numOps = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            var scores = engine.TensorAdd(
                engine.TensorMultiplyScalar(
                    engine.BatchMatMul(queries4D, engine.TensorPermute(keys4D, new[] { 0, 1, 3, 2 })),
                    numOps.FromDouble(scale)),
                attentionBias);
            var probabilities = engine.TensorSoftmax(scores, 3);
            output = engine.BatchMatMul(probabilities, values4D);
            weights = config.ReturnAttentionWeights ? probabilities : null;
        }

        if (!is4D)
        {
            output = engine.Reshape(output, new[] { output.Shape[0], output.Shape[2], output.Shape[3] });
            if (weights is not null)
                weights = engine.Reshape(weights, new[] { weights.Shape[0], weights.Shape[2], weights.Shape[3] });
        }
        return (output, weights);
    }

    /// <summary>Adds the singleton head axis a 3D [batch, seq, dim] input leaves implicit.</summary>
    private static Tensor<T> PromoteToFourD(IEngine engine, Tensor<T> tensor)
        => engine.Reshape(tensor, new[] { tensor.Shape[0], 1, tensor.Shape[1], tensor.Shape[2] });

    /// <summary>
    /// Map the framework's <see cref="FlashAttentionConfig"/> onto the Tensors-side
    /// <see cref="AiDotNet.Tensors.Engines.Autodiff.FlashAttentionConfig"/>.
    /// <see cref="FlashAttentionConfig.UseGpuKernel"/> and
    /// <see cref="FlashAttentionConfig.RecomputeInBackward"/> have no counterpart
    /// — the Tensors engine selects the device-appropriate kernel automatically
    /// and the autograd tape handles backward (recompute vs. store is a future
    /// memory-mode option, not a correctness toggle).
    /// </summary>
    private static AiDotNet.Tensors.Engines.Autodiff.FlashAttentionConfig MapConfig(
        FlashAttentionConfig config, int queryOffset)
        => new()
        {
            // Mirror the documented fields.
            BlockSizeQ = config.BlockSizeQ,
            BlockSizeKV = config.BlockSizeKV,
            IsCausal = config.UseCausalMask,
            Scale = config.ScaleFactor,
            DropoutRate = config.DropoutProbability,
            ReturnAttentionWeights = config.ReturnAttentionWeights,
            QueryOffset = queryOffset,
            // For T == double, opt into the float32-internal kernel by default
            // (Tensors #472). Maps the framework's Precision enum:
            //   * Float32 / Float16 / Mixed -> Float32Precision = true  (float-grade compute)
            //   * (no enum value forces FP64 compute, but if one is added later
            //      this lets it opt out by setting Float32Precision = false explicitly)
            // The flag is a no-op when T == float (already at float compute).
            Float32Precision = typeof(T) == typeof(double),
        };

    private static void ValidateInputs(Tensor<T> query, Tensor<T> key, Tensor<T> value)
    {
        if (query.Shape.Length != key.Shape.Length || key.Shape.Length != value.Shape.Length)
            throw new ArgumentException("Query, Key, and Value must have the same number of dimensions.");

        if (query.Shape.Length < 3 || query.Shape.Length > 4)
            throw new ArgumentException("Query, Key, and Value must be 3D [batch, seq, dim] or 4D [batch, heads, seq, dim].");

        // Batch size must match.
        if (query.Shape[0] != key.Shape[0] || key.Shape[0] != value.Shape[0])
            throw new ArgumentException("Batch sizes must match across Query, Key, and Value.");

        if (query.Shape.Length == 4)
        {
            // Heads must match.
            if (query.Shape[1] != key.Shape[1] || key.Shape[1] != value.Shape[1])
                throw new ArgumentException("Number of heads must match across Query, Key, and Value.");
            // Head dimension must match between Q and K.
            if (query.Shape[3] != key.Shape[3])
                throw new ArgumentException("Head dimension must match between Query and Key.");
            // K and V sequence lengths must match.
            if (key.Shape[2] != value.Shape[2])
                throw new ArgumentException("Key and Value sequence lengths must match.");
        }
        else
        {
            // 3D: feature dim between Q and K must match; K and V seq must match.
            if (query.Shape[2] != key.Shape[2])
                throw new ArgumentException("Feature dimension must match between Query and Key.");
            if (key.Shape[1] != value.Shape[1])
                throw new ArgumentException("Key and Value sequence lengths must match.");
        }
    }
}
