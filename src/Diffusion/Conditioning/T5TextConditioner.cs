using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Attention;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Diffusion.Conditioning;

/// <summary>
/// T5 text encoder conditioning module for diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// T5 (Text-To-Text Transfer Transformer) text encoder provides high-dimensional sequence
/// embeddings for advanced diffusion models. T5-XXL is used in Imagen, SD3, and FLUX.1
/// for its superior text understanding compared to CLIP.
/// </para>
/// <para>
/// <b>For Beginners:</b> T5 is a more powerful text encoder than CLIP.
///
/// Why T5 in addition to CLIP:
/// - CLIP understands image-text relationships (good for visual concepts)
/// - T5 understands language deeply (good for complex prompts, text rendering)
/// - Together they give the best of both worlds
///
/// Key differences from CLIP:
/// - Much larger: T5-XXL has 4.7B parameters (vs CLIP's 123M-354M)
/// - Longer sequences: 256-512 tokens (vs CLIP's 77)
/// - Higher dimensional: 4096-dim (vs CLIP's 768-1280)
/// - No pooled output: Only produces sequence embeddings
/// - Better at: Complex prompts, counting, spatial relationships, text in images
///
/// T5 variants used in diffusion:
/// - T5-XXL: 4096-dim, 24 layers, used in Imagen, SD3, FLUX.1
/// - T5-XL: 2048-dim, 24 layers, sometimes used for memory-constrained setups
/// - T5-Large: 1024-dim, 24 layers
///
/// Important: T5 uses relative position encodings (not absolute like CLIP),
/// which helps with varying-length inputs.
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
/// JMLR 2020 — defines the T5 architecture: pre-norm RMSNorm, no attention scaling, ReLU FFN,
/// shared learned relative position bias with logarithmic bucketing.</item>
/// <item>Shazeer, "GLU Variants Improve Transformer", arXiv:2002.05202, 2020 — introduces the GeGLU
/// activation that replaces the original ReLU FFN block in T5.1.1 (Google's later T5 codebase
/// improvement). This implementation uses GeGLU because the empirical gains over ReLU are reported
/// across multiple downstream tasks while changing only the FFN block.</item>
/// </list>
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class T5TextConditioner<T> : TextConditioningBase<T>
{
    /// <summary>
    /// Relative position bias weights for self-attention.
    /// </summary>
    private readonly Vector<T> _relativePositionBias;

    /// <summary>
    /// Number of relative position buckets.
    /// </summary>
    private readonly int _numBuckets;

    /// <summary>
    /// The T5 variant name.
    /// </summary>
    private readonly string _variant;

    /// <summary>
    /// FFN inner dimension per T5.1.1 variant configs (d_ff in the Raffel/Shazeer notation).
    /// </summary>
    private readonly int _ffnDim;

    // ---- Per-layer lazy rent-and-return weight storage. ----
    // The previous implementation cached every layer's Q/K/V/O/FFN
    // tensors simultaneously in a single flat Vector<T>. That layout
    // overflows int32 for T5-XXL (24 layers × 4·H² + 3·H·F + 2·H =
    // ~4.6B elements; issue #1189) AND keeps ~37GB of weight memory
    // resident in a 32GB CI process even when it fits in int.
    //
    // Instead we allocate each layer's weights on demand inside the
    // forward pass via TensorAllocator.Rent<T>(), initialize them from
    // a per-layer deterministic seed, run the layer, and return the
    // rented buffers to the pool before the next layer's rent. Peak
    // working memory drops to a single layer (≈1.5GB for T5-XXL at
    // double) and the pool reuses the same backing arrays across the
    // 24 layers.
    //
    // Determinism: every EncodeText call re-seeds layer l from
    // (_baseSeed XOR l XOR matrixTag), so repeat calls with the same
    // input produce the same output without ever persisting weights.
    // When a caller eventually wants pretrained weights loaded from a
    // .safetensors checkpoint, that is a separate code path that
    // populates a _loadedWeights cache short-circuiting the rent — not
    // in this PR.
    //
    // Matrix tags (used to salt the per-matrix seed so Q weights are
    // distinct from K, V, O, etc. within the same layer).
    private const int MatrixTagQ       = 0;
    private const int MatrixTagK       = 1;
    private const int MatrixTagV       = 2;
    private const int MatrixTagO       = 3;
    private const int MatrixTagFfnGate = 4;
    private const int MatrixTagFfnVal  = 5;
    private const int MatrixTagFfnOut  = 6;

    /// <summary>
    /// The master seed this encoder was constructed with. Used to
    /// generate per-layer per-matrix deterministic weights during
    /// forward. Stored separately from <see cref="TextConditioningBase{T}.Rng"/>
    /// (which is shared with TokenEmbeddings / PositionEmbeddings
    /// initialization) so layer weight generation is reproducible
    /// independently of any other random draws.
    /// </summary>
    private readonly int _baseSeed;

    /// <summary>
    /// Bundle of rented tensors that make up one encoder layer's
    /// weight set. Populated by <see cref="RentAndInitLayerWeights"/>,
    /// returned to the pool by <see cref="ReturnLayerWeights"/>.
    /// </summary>
    private readonly struct LayerWeights
    {
        public readonly Tensor<T> Q;            // [H, H]
        public readonly Tensor<T> K;            // [H, H]
        public readonly Tensor<T> V;            // [H, H]
        public readonly Tensor<T> AttnOut;      // [H, H]
        public readonly Tensor<T> FfnGate;      // [H, F] — GeGLU gate (GELU-activated)
        public readonly Tensor<T> FfnValue;     // [H, F] — GeGLU value (linear)
        public readonly Tensor<T> FfnOut;       // [F, H] — FFN down-projection
        public readonly Vector<T> Norm1Gamma;   // [H]
        public readonly Vector<T> Norm2Gamma;   // [H]

        public LayerWeights(
            Tensor<T> q, Tensor<T> k, Tensor<T> v, Tensor<T> attnOut,
            Tensor<T> ffnGate, Tensor<T> ffnValue, Tensor<T> ffnOut,
            Vector<T> norm1Gamma, Vector<T> norm2Gamma)
        {
            Q = q; K = k; V = v; AttnOut = attnOut;
            FfnGate = ffnGate; FfnValue = ffnValue; FfnOut = ffnOut;
            Norm1Gamma = norm1Gamma; Norm2Gamma = norm2Gamma;
        }
    }

    /// <summary>
    /// Cached relative-position-bias broadcast for a specific seqLen, shape [NumHeads, seqLen, seqLen].
    /// Computed once per unique seqLen via T5 bucketing and reused on subsequent calls.
    /// Invalidated together with the weight tensors when <see cref="InvalidateCachedWeightTensors"/>
    /// is called (so a mutated <see cref="_relativePositionBias"/> is picked up correctly).
    /// Capacity is bounded by <see cref="MaxRpbCacheEntries"/> with simple LRU eviction —
    /// each tensor is O(NumHeads * seqLen^2) which can be very large for T5-XXL (64 heads,
    /// seqLen=256 → ~4M elements ≈ 16MB per entry in float32). An unbounded cache would
    /// retain hundreds of MB across workloads with variable sequence lengths.
    /// </summary>
    private readonly Dictionary<int, Tensor<T>> _rpbBroadcastCache = new();

    /// <summary>
    /// LRU eviction order for <see cref="_rpbBroadcastCache"/>: tracks most-recently-used
    /// seqLen at the tail. Sized small to bound peak memory; in practice 4 distinct
    /// sequence lengths covers most workloads (one common train length, one inference
    /// length, plus a couple of edge cases) without thrashing.
    /// </summary>
    private readonly LinkedList<int> _rpbCacheLruOrder = new();

    /// <summary>
    /// Maximum number of distinct seqLen entries to cache. With T5-XXL each entry can be
    /// tens of MB; this cap keeps the worst-case memory footprint bounded regardless of
    /// how many distinct sequence lengths a workload feeds the encoder.
    /// </summary>
    private const int MaxRpbCacheEntries = 4;

    /// <summary>
    /// Cached ones tensor [HiddenSize, 1] for RMS norm sum reduction.
    /// </summary>
    private Tensor<T>? _rmsOnesTensor;

    /// <summary>
    /// Cached epsilon tensor [1, 1] for RMS norm numerical stability.
    /// </summary>
    private Tensor<T>? _rmsEpsTensor;

    /// <summary>
    /// Shared all-ones Vector&lt;T&gt; of length HiddenSize, reused for every
    /// layer's pre-attention and pre-FFN RMSNorm gamma. T5.1.1 initializes
    /// every norm gamma to 1.0 and the forward pass never mutates them, so
    /// 24 layers × 2 gammas can safely alias a single immutable buffer
    /// instead of allocating 48 fresh Vector&lt;T&gt;(H) per encode (~1.5MB
    /// of gen0 garbage per call for T5-XXL).
    /// </summary>
    private Vector<T>? _onesGamma;

    /// <summary>
    /// Gets whether this module produces pooled output (T5 does not).
    /// </summary>
    public override bool ProducesPooledOutput => false;

    /// <summary>
    /// Initializes a new T5 text encoder conditioning module.
    /// </summary>
    /// <param name="variant">
    /// T5 variant to use:
    /// - "T5-XXL": 4096-dim, 24 layers (used in Imagen, SD3, FLUX.1)
    /// - "T5-XL": 2048-dim, 24 layers
    /// - "T5-Large": 1024-dim, 24 layers
    /// Default: "T5-XXL"
    /// </param>
    /// <param name="maxSequenceLength">Maximum sequence length. Default: 256.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <example>
    /// <code>
    /// // Create T5-XXL for SD3/FLUX.1
    /// var t5 = new T5TextConditioner&lt;float&gt;();
    ///
    /// // Create T5-XL for memory-constrained setups
    /// var t5xl = new T5TextConditioner&lt;float&gt;(variant: "T5-XL");
    /// </code>
    /// </example>
    public T5TextConditioner(string variant = "T5-XXL", int maxSequenceLength = 256, int? seed = null)
        : base(
            vocabSize: 32128, // T5 SentencePiece vocabulary size
            embeddingDimension: GetEmbeddingDim(variant),
            hiddenSize: GetHiddenSize(variant),
            numLayers: GetNumLayers(variant),
            numHeads: GetNumHeads(variant),
            maxSequenceLength: maxSequenceLength,
            seed: seed,
            // We manage per-layer weight storage ourselves via lazy
            // rent-and-return on TensorAllocator (see LayerWeights +
            // RentAndInitLayerWeights). The base class's single flat
            // Vector<T> layout cannot represent T5-XXL (4.6B params >
            // int.MaxValue), so opt out of that allocation entirely.
            skipBaseWeightAllocation: true)
    {
        _variant = variant;
        _numBuckets = 32;
        _ffnDim = GetFfnDim(variant);
        // Default seed when the caller passes null: 0 is fine because
        // the master seed salts per-layer, per-matrix below and the
        // hash-combine guarantees distinct streams for different
        // (layer, matrix) pairs even at _baseSeed == 0.
        _baseSeed = seed ?? 0;

        // Relative position bias: [numHeads, numBuckets] — shared across all encoder layers
        // per the original T5 reference implementation.
        _relativePositionBias = InitializeWeights(NumHeads * _numBuckets);

        // Sanity-check that a single layer's weight matrices fit in
        // int32 (needed for Vector<T> and the per-matrix element count
        // passed to TensorAllocator.Rent). For T5-XXL per-layer is
        // 4·H² + 3·H·F + 2·H = ~193M elements, well below int.MaxValue
        // (2.1B) — so this is a guard for future oversized variants,
        // not for T5-XXL itself. T5-XXL's per-layer size at double is
        // ~1.5GB of RAM, which our rent-and-return pattern keeps to
        // one layer resident at a time.
        long perLayerElements = 4L * HiddenSize * HiddenSize
                              + 3L * HiddenSize * _ffnDim
                              + 2L * HiddenSize;
        if (perLayerElements > int.MaxValue)
        {
            throw new InvalidOperationException(
                $"T5 variant '{variant}' has {perLayerElements:N0} elements per layer, " +
                $"which exceeds the int32 per-layer cap ({int.MaxValue:N0}). " +
                $"Per-layer lazy allocation only works when each individual " +
                $"matrix fits in a single Vector<T> / Tensor<T>. No known T5 " +
                $"variant trips this — if you hit it, you're on an unsupported " +
                $"config.");
        }
    }

    /// <summary>
    /// Rents the seven weight tensors and two norm gammas that make up one
    /// encoder layer, populating each tensor's backing buffer with a
    /// deterministic Xavier-normal draw seeded by
    /// <c>(_baseSeed, layerIdx, matrixTag)</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The tensors are drawn from <see cref="TensorAllocator"/>'s pool, so
    /// after the first forward pass warms the arena, every subsequent
    /// layer in every subsequent call reuses the same underlying arrays —
    /// zero GC allocations in steady state. Every rented buffer MUST be
    /// handed back via <see cref="ReturnLayerWeights"/> before the next
    /// layer is rented so the pool recycles them into the next layer's
    /// rents rather than growing unboundedly.
    /// </para>
    /// <para>
    /// Determinism across calls: the per-matrix seed is derived by
    /// <see cref="DeriveSeed"/>, a process-stable integer mix (deliberately
    /// NOT <c>HashCode.Combine</c>, which is randomized per process start).
    /// So a T5TextConditioner constructed with a given seed produces the
    /// same encoder output on every call — essential for checkpointing and
    /// reproducible training — without persisting any weight cache.
    /// </para>
    /// </remarks>
    private LayerWeights RentAndInitLayerWeights(int layerIdx)
    {
        int H = HiddenSize;
        int F = _ffnDim;

        Tensor<T>? q = null, k = null, v = null, attnOut = null;
        Tensor<T>? ffnGate = null, ffnValue = null, ffnOut = null;
        try
        {
            // RentUninitialized — every element is overwritten below by
            // FillXavier so zeroing the buffer first would be wasted work.
            q        = TensorAllocator.RentUninitialized<T>(new[] { H, H });
            k        = TensorAllocator.RentUninitialized<T>(new[] { H, H });
            v        = TensorAllocator.RentUninitialized<T>(new[] { H, H });
            attnOut  = TensorAllocator.RentUninitialized<T>(new[] { H, H });
            ffnGate  = TensorAllocator.RentUninitialized<T>(new[] { H, F });
            ffnValue = TensorAllocator.RentUninitialized<T>(new[] { H, F });
            ffnOut   = TensorAllocator.RentUninitialized<T>(new[] { F, H });

            FillXavier(q.AsWritableSpan(),        H * H, DeriveSeed(_baseSeed, layerIdx, MatrixTagQ));
            FillXavier(k.AsWritableSpan(),        H * H, DeriveSeed(_baseSeed, layerIdx, MatrixTagK));
            FillXavier(v.AsWritableSpan(),        H * H, DeriveSeed(_baseSeed, layerIdx, MatrixTagV));
            FillXavier(attnOut.AsWritableSpan(),  H * H, DeriveSeed(_baseSeed, layerIdx, MatrixTagO));
            FillXavier(ffnGate.AsWritableSpan(),  H * F, DeriveSeed(_baseSeed, layerIdx, MatrixTagFfnGate));
            FillXavier(ffnValue.AsWritableSpan(), H * F, DeriveSeed(_baseSeed, layerIdx, MatrixTagFfnVal));
            FillXavier(ffnOut.AsWritableSpan(),   F * H, DeriveSeed(_baseSeed, layerIdx, MatrixTagFfnOut));

            // RMSNorm gammas: T5.1.1 initializes every gamma to 1.0 and the
            // forward pass never mutates them, so all 48 per-layer gammas
            // (24 layers × 2 norms) safely alias a single shared all-ones
            // buffer cached on the instance. Saves ~1.5MB of per-call gen0
            // garbage for T5-XXL.
            var ones = _onesGamma;
            if (ones == null || ones.Length != H)
            {
                ones = new Vector<T>(H);
                T one = NumOps.One;
                for (int i = 0; i < H; i++) ones[i] = one;
                _onesGamma = ones;
            }

            return new LayerWeights(q, k, v, attnOut, ffnGate, ffnValue, ffnOut, ones, ones);
        }
        catch
        {
            // A failure mid-init (RentUninitialized OOM, FillXavier
            // throwing on a degenerate seed, etc.) must not strand the
            // earlier rents in the pool — for T5-XXL each layer's
            // rents total ~1.5GB and would never be reclaimed.
            // TryReturn swallows secondary failures so the original
            // exception still surfaces.
            Exception? cleanup = null;
            if (q        != null) TryReturn(q,        ref cleanup);
            if (k        != null) TryReturn(k,        ref cleanup);
            if (v        != null) TryReturn(v,        ref cleanup);
            if (attnOut  != null) TryReturn(attnOut,  ref cleanup);
            if (ffnGate  != null) TryReturn(ffnGate,  ref cleanup);
            if (ffnValue != null) TryReturn(ffnValue, ref cleanup);
            if (ffnOut   != null) TryReturn(ffnOut,   ref cleanup);
            throw;
        }
    }

    /// <summary>
    /// Returns every tensor in <paramref name="lw"/> to the shared pool so
    /// subsequent <see cref="RentAndInitLayerWeights"/> calls reuse the
    /// backing arrays. Must be called in a <c>finally</c> block so a failure
    /// inside the forward pass doesn't leak the layer's buffers.
    /// </summary>
    /// <remarks>
    /// Each return is independently guarded: if one tensor's <c>Return</c>
    /// throws (e.g., the allocator's double-return detection trips on a
    /// caller mistake), the remaining tensors still go back to the pool so
    /// a single bad return doesn't leak ~1.5GB of layer weights for T5-XXL.
    /// The first exception is rethrown after all returns have been
    /// attempted; subsequent exceptions are dropped because surfacing both
    /// the original failure and downstream failures-to-clean-up only adds
    /// noise to the stack trace without changing the diagnosis.
    /// </remarks>
    private static void ReturnLayerWeights(LayerWeights lw)
    {
        Exception? first = null;
        TryReturn(lw.Q, ref first);
        TryReturn(lw.K, ref first);
        TryReturn(lw.V, ref first);
        TryReturn(lw.AttnOut, ref first);
        TryReturn(lw.FfnGate, ref first);
        TryReturn(lw.FfnValue, ref first);
        TryReturn(lw.FfnOut, ref first);
        // Norm gammas are plain Vector<T>; the GC reclaims them.
        if (first != null) throw first;
    }

    private static void TryReturn(Tensor<T> t, ref Exception? first)
    {
        try
        {
            TensorAllocator.Return(t);
        }
        catch (Exception ex)
        {
            if (first == null) first = ex;
        }
    }

    /// <summary>
    /// Fills <paramref name="buf"/> with a Box-Muller Xavier-normal draw
    /// (stddev = sqrt(2/(size+1))) from a fresh <see cref="Random"/> seeded
    /// with <paramref name="seed"/>. Each pair of uniforms produces two
    /// independent normals (cos + sin branches) so RNG work is halved
    /// versus the single-branch implementation in
    /// <see cref="TextConditioningBase{T}.InitializeWeights"/> — which
    /// matters because T5-XXL fills ~193M elements per layer × 24 layers
    /// on every encode.
    /// </summary>
    private static void FillXavier(Span<T> buf, int size, int seed)
    {
        var rng = new Random(seed);
        double stddev = Math.Sqrt(2.0 / (size + 1));
        int i = 0;
        while (i < size)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double r = Math.Sqrt(-2.0 * Math.Log(u1));
            double theta = 2.0 * Math.PI * u2;
            buf[i++] = NumOps.FromDouble(r * Math.Cos(theta) * stddev);
            if (i < size)
                buf[i++] = NumOps.FromDouble(r * Math.Sin(theta) * stddev);
        }
    }

    /// <summary>
    /// Stable per-process integer mix of <c>(baseSeed, layerIdx, matrixTag)</c>.
    /// Deliberately does NOT use <see cref="System.HashCode.Combine(int, int, int)"/>
    /// because that API is randomized per-process by design, which would
    /// make encoder outputs non-reproducible across process starts.
    /// </summary>
    private static int DeriveSeed(int baseSeed, int layerIdx, int matrixTag)
    {
        unchecked
        {
            int h = baseSeed;
            h = h * 397 ^ layerIdx;
            h = h * 397 ^ matrixTag;
            return h;
        }
    }

    /// <summary>
    /// Drops the cached relative-position-bias broadcasts. Call this from
    /// any code path that mutates <see cref="_relativePositionBias"/>
    /// (custom <c>SetParameters</c>, checkpoint load, QAT update) so the
    /// next <see cref="EncodeText"/> rebuilds the per-seqLen broadcast
    /// from the new bias values.
    /// </summary>
    /// <remarks>
    /// Encoder-layer weights are no longer cached — they are rented fresh
    /// from <see cref="TensorAllocator"/> on every forward pass (see
    /// <see cref="RentAndInitLayerWeights"/>), so no weight invalidation
    /// is needed. The method's name is kept for API stability; only the
    /// RPB cache survived the rent-and-return refactor.
    /// </remarks>
    public void InvalidateCachedWeightTensors()
    {
        _rpbBroadcastCache.Clear();
        _rpbCacheLruOrder.Clear();
    }

    /// <inheritdoc />
    public override Tensor<T> Encode(Tensor<T> input)
    {
        return EncodeText(input);
    }

    /// <inheritdoc />
    public override Tensor<T> EncodeText(Tensor<T> tokenIds, Tensor<T>? attentionMask = null)
    {
        // NOTE: we DO NOT open a TensorArena scope here. A tempting
        // optimization would be to wrap the whole encode in
        // `using var arena = TensorArena.Create()` for zero GC across
        // calls, but the arena captures every rent and only recycles on
        // Dispose — which means mid-flight Return() calls are no-ops.
        // For T5-XXL each layer rents ~1.5GB of weight tensors, so 24
        // layers inside an arena would accumulate to ~37GB before
        // release (exactly the pre-refactor OOM pattern). Skipping the
        // arena lets Return() actually recycle each layer's buffers into
        // the next layer's rents, keeping peak working set at one layer.

        var shape = tokenIds._shape;
        int batchSize = shape[0];
        int seqLen = shape.Length > 1 ? shape[1] : MaxSequenceLength;
        int totalTokens = batchSize * (shape.Length > 1 ? shape[1] : 1);

        // Token embedding lookup: build hidden tensor [batchSize, seqLen, HiddenSize].
        // (Per-token gather is inherently scalar; this is the T5 input embedding layer.)
        var hiddenData = new Vector<T>(batchSize * seqLen * HiddenSize);
        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int flatIdx = b * seqLen + s;
                int tokenId = flatIdx < totalTokens
                    ? Math.Max(0, Math.Min((int)NumOps.ToDouble(tokenIds[flatIdx]), VocabSize - 1))
                    : 0;
                int srcOff = tokenId * HiddenSize;
                int dstOff = (b * seqLen + s) * HiddenSize;
                for (int d = 0; d < HiddenSize; d++)
                    hiddenData[dstOff + d] = TokenEmbeddings[srcOff + d];
            }
        }

        var hidden = new Tensor<T>(new[] { batchSize, seqLen, HiddenSize }, hiddenData);

        // Run all encoder layers fully batched over [B, S, H] — no per-batch Python-style loop.
        hidden = ApplyT5EncoderLayersBatched(hidden, batchSize, seqLen);

        // T5 final RMS norm applied across the trailing HiddenSize axis.
        hidden = RMSNormEngine(hidden, FinalLayerNormWeights, HiddenSize);

        // For all current variants EmbeddingDimension == HiddenSize, so the encoder output
        // is already the desired shape. Reshape (no-copy) only if a future variant decouples them.
        if (EmbeddingDimension == HiddenSize)
            return hidden;

        // Defensive path: if a variant ever sets EmbeddingDimension != HiddenSize, project / pad
        // by simple element copy. Avoids hiding a configuration mismatch at runtime.
        var outputData = new Vector<T>(batchSize * seqLen * EmbeddingDimension);
        int copyDim = Math.Min(HiddenSize, EmbeddingDimension);
        for (int b = 0; b < batchSize; b++)
        for (int s = 0; s < seqLen; s++)
        for (int d = 0; d < copyDim; d++)
            outputData[(b * seqLen + s) * EmbeddingDimension + d] = hidden[(b * seqLen + s) * HiddenSize + d];
        return new Tensor<T>(new[] { batchSize, seqLen, EmbeddingDimension }, outputData);
    }

    /// <inheritdoc />
    public override Tensor<T> GetPooledEmbedding(Tensor<T> sequenceEmbeddings)
    {
        // T5 doesn't produce pooled output - use mean pooling via Engine
        var shape = sequenceEmbeddings._shape;
        int batchSize = shape[0];
        int seqLen = shape[1];

        // Mean pool: sum across sequence dimension, then divide
        var pooledData = new Vector<T>(batchSize * EmbeddingDimension);

        // Hoist constant ones tensor outside the batch loop
        var onesForMean = new Tensor<T>(new[] { 1, seqLen });
        var onesSpan = onesForMean.AsWritableSpan();
        for (int i = 0; i < seqLen; i++) onesSpan[i] = NumOps.One;

        for (int b = 0; b < batchSize; b++)
        {
            // Build a [seqLen, embDim] tensor for this batch
            int batchOff = b * seqLen * EmbeddingDimension;
            var batchVec = new Vector<T>(sequenceEmbeddings.AsSpan().Slice(batchOff, seqLen * EmbeddingDimension).ToArray());
            var batchTensor = new Tensor<T>(new[] { seqLen, EmbeddingDimension }, batchVec);

            // [1, seqLen] @ [seqLen, embDim] -> [1, embDim]
            var summed = Engine.TensorMatMul(onesForMean, batchTensor);

            // Divide by seqLen
            var meanTensor = Engine.TensorDivideScalar(summed, NumOps.FromDouble(seqLen));

            // Copy result
            for (int d = 0; d < EmbeddingDimension; d++)
                pooledData[b * EmbeddingDimension + d] = meanTensor[d];
        }

        return new Tensor<T>(new[] { batchSize, EmbeddingDimension }, pooledData);
    }

    /// <inheritdoc />
    public override Tensor<T> GetUnconditionalEmbedding(int batchSize)
    {
        // Empty input: just the padding/EOS token
        var tokenIds = new Vector<T>(batchSize * MaxSequenceLength);
        for (int b = 0; b < batchSize; b++)
        {
            tokenIds[b * MaxSequenceLength] = NumOps.FromDouble(1); // EOS/pad token
        }

        var input = new Tensor<T>(new[] { batchSize, MaxSequenceLength }, tokenIds);
        return EncodeText(input);
    }

    /// <inheritdoc />
    public override Tensor<T> Tokenize(string text)
    {
        var tokens = SimpleTokenize(text, MaxSequenceLength);
        var tokenData = new Vector<T>(MaxSequenceLength);
        for (int i = 0; i < MaxSequenceLength; i++)
            tokenData[i] = NumOps.FromDouble(tokens[i]);

        return new Tensor<T>(new[] { 1, MaxSequenceLength }, tokenData);
    }

    /// <inheritdoc />
    public override Tensor<T> TokenizeBatch(string[] texts)
    {
        var tokenData = new Vector<T>(texts.Length * MaxSequenceLength);
        for (int b = 0; b < texts.Length; b++)
        {
            var tokens = SimpleTokenize(texts[b], MaxSequenceLength);
            for (int i = 0; i < MaxSequenceLength; i++)
                tokenData[b * MaxSequenceLength + i] = NumOps.FromDouble(tokens[i]);
        }

        return new Tensor<T>(new[] { texts.Length, MaxSequenceLength }, tokenData);
    }

    /// <summary>
    /// Paper-faithful T5 encoder forward pass, fully batched over <c>[B, S, H]</c>.
    /// </summary>
    /// <remarks>
    /// <para>Per-layer structure (Raffel et al. 2020, §3.1.1; Shazeer 2020 for GeGLU):</para>
    /// <list type="number">
    /// <item>Pre-norm self-attention: <c>RMSNorm → Q/K/V projections → multi-head attention with
    /// shared learned relative position bias added to scores → output projection → residual</c>.
    /// Attention is unscaled (T5 explicitly omits 1/√d_k) and bidirectional (encoder).</item>
    /// <item>Pre-norm GeGLU FFN: <c>RMSNorm → (GELU(x · W_i_0) ⊙ (x · W_i_1)) · W_o → residual</c>.
    /// GeGLU replaces the original ReLU FFN per Shazeer 2020.</item>
    /// </list>
    /// <para>Attention uses <see cref="FlashAttention{T}"/> with <c>ScaleFactor=1.0</c> and the T5
    /// relative position bias passed via <c>attentionBias</c>; FlashAttention adds it to scores
    /// before the (online) softmax — exactly where T5 places it.</para>
    /// </remarks>
    private Tensor<T> ApplyT5EncoderLayersBatched(Tensor<T> hidden, int batchSize, int seqLen)
    {
        int H = HiddenSize;
        if (H % NumHeads != 0)
        {
            throw new InvalidOperationException(
                $"HiddenSize ({H}) must be divisible by NumHeads ({NumHeads}). Variant '{_variant}' has an invalid configuration.");
        }
        int headDim = H / NumHeads;
        int totalTokens = batchSize * seqLen;

        // Build the relative-position-bias tensor once for this seqLen and reuse for every
        // layer — T5 shares one learned bias matrix [NumHeads, NumBuckets] across all layers.
        // Cached per seqLen; invalidated by InvalidateCachedWeightTensors when bias mutates.
        var rpb = GetOrBuildRelativePositionBias(seqLen);  // [NumHeads, S, S]

        // Configure FlashAttention for T5 encoder: no scaling, bidirectional, no dropout.
        var flashConfig = new FlashAttentionConfig
        {
            ScaleFactor = 1.0f,
            UseCausalMask = false,
            ReturnAttentionWeights = false,
            DropoutProbability = 0.0f,
        };

        for (int layer = 0; layer < NumLayers; layer++)
        {
            // Rent this layer's weights from the pool, use them for exactly
            // this layer's forward, and return them before the next layer's
            // rent. Peak working memory drops from all-layers-cached
            // (~37GB for T5-XXL at double) to one-layer-resident
            // (~1.5GB), and the pool reuses the same backing arrays across
            // all 24 layers — zero GC allocations after the first pass
            // warms the pool.
            var lw = RentAndInitLayerWeights(layer);
            Exception? bodyFailure = null;
            try
            {
                // ===== Self-attention block (pre-norm + residual) =====
                var residual = hidden;
                var normed = RMSNormEngine(hidden, lw.Norm1Gamma, H);

                // Project Q, K, V via 2D matmul (collapse batch and seq into one axis).
                var normed2D = Engine.Reshape(normed, new[] { totalTokens, H });
                var qFlat = Engine.TensorMatMul(normed2D, lw.Q);  // [B*S, H]
                var kFlat = Engine.TensorMatMul(normed2D, lw.K);
                var vFlat = Engine.TensorMatMul(normed2D, lw.V);

                // Split heads: [B*S, H] → [B, S, NumHeads, HeadDim] → [B, NumHeads, S, HeadDim]
                var qSplit = Engine.Reshape(qFlat, new[] { batchSize, seqLen, NumHeads, headDim });
                var kSplit = Engine.Reshape(kFlat, new[] { batchSize, seqLen, NumHeads, headDim });
                var vSplit = Engine.Reshape(vFlat, new[] { batchSize, seqLen, NumHeads, headDim });
                var q4D = Engine.TensorPermute(qSplit, new[] { 0, 2, 1, 3 });
                var k4D = Engine.TensorPermute(kSplit, new[] { 0, 2, 1, 3 });
                var v4D = Engine.TensorPermute(vSplit, new[] { 0, 2, 1, 3 });

                // FlashAttention: softmax(Q @ K^T + RPB) @ V, with our scale=1.0 override (T5).
                var (attnOut4D, _) = FlashAttention<T>.Forward(
                    q4D, k4D, v4D, flashConfig, queryOffset: 0, attentionBias: rpb);
                // attnOut4D: [B, NumHeads, S, HeadDim]

                // Combine heads: [B, NumHeads, S, HeadDim] → [B, S, NumHeads, HeadDim] → [B*S, H]
                var attnPerm = Engine.TensorPermute(attnOut4D, new[] { 0, 2, 1, 3 });
                var attnFlat = Engine.Reshape(attnPerm, new[] { totalTokens, H });

                // Output projection W_o.
                var attnProjFlat = Engine.TensorMatMul(attnFlat, lw.AttnOut);
                var attnProj = Engine.Reshape(attnProjFlat, new[] { batchSize, seqLen, H });

                hidden = Engine.TensorAdd(residual, attnProj);

                // ===== Feed-forward block (pre-norm + residual + GeGLU) =====
                residual = hidden;
                normed = RMSNormEngine(hidden, lw.Norm2Gamma, H);
                normed2D = Engine.Reshape(normed, new[] { totalTokens, H });

                // GeGLU: gate = GELU(x @ W_i_0), value = x @ W_i_1, fused = gate ⊙ value, out = fused @ W_o
                var gateProj = Engine.TensorMatMul(normed2D, lw.FfnGate);   // [B*S, F]
                var valueProj = Engine.TensorMatMul(normed2D, lw.FfnValue); // [B*S, F]
                var gateActivated = Engine.GELU(gateProj);                  // GELU per Shazeer
                var fused = Engine.TensorMultiply(gateActivated, valueProj); // elementwise gating
                var ffnOutFlat = Engine.TensorMatMul(fused, lw.FfnOut);      // [B*S, H]
                var ffnOut = Engine.Reshape(ffnOutFlat, new[] { batchSize, seqLen, H });

                hidden = Engine.TensorAdd(residual, ffnOut);
            }
            catch (Exception ex)
            {
                bodyFailure = ex;
                throw;
            }
            finally
            {
                // If the layer body already failed, swallow any cleanup
                // exception so the original matmul/attention error keeps
                // propagating — losing that diagnosis to a downstream
                // pool-return failure would defeat the whole point of
                // the rent/return pattern. When the body succeeded,
                // a Return failure should still surface (pool corruption
                // or double-return is a real bug worth seeing).
                try
                {
                    ReturnLayerWeights(lw);
                }
                catch when (bodyFailure != null)
                {
                    // Original failure wins; do nothing.
                }
            }
        }

        return hidden;
    }

    /// <summary>
    /// Returns (and caches per <paramref name="seqLen"/>) the T5 relative-position-bias tensor
    /// of shape <c>[NumHeads, seqLen, seqLen]</c>, ready to be added to attention scores.
    /// The underlying <see cref="_relativePositionBias"/> is shared across all layers per the
    /// T5 reference implementation.
    /// </summary>
    private Tensor<T> GetOrBuildRelativePositionBias(int seqLen)
    {
        if (_rpbBroadcastCache.TryGetValue(seqLen, out var cached))
        {
            // Hit — bump to MRU position so it survives the next eviction.
            _rpbCacheLruOrder.Remove(seqLen);
            _rpbCacheLruOrder.AddLast(seqLen);
            return cached;
        }

        const int maxDistance = 128; // T5 default
        var data = new Vector<T>(NumHeads * seqLen * seqLen);

        // Loop order: q outermost, k inner — same as the reference implementation.
        // For each (q, k) pair, the bucket is the same across heads, so compute once and reuse.
        for (int q = 0; q < seqLen; q++)
        {
            for (int k = 0; k < seqLen; k++)
            {
                int bucket = RelativePositionBucket(
                    relativePosition: k - q,
                    bidirectional: true,
                    numBuckets: _numBuckets,
                    maxDistance: maxDistance);

                for (int h = 0; h < NumHeads; h++)
                {
                    int rpbIdx = h * _numBuckets + bucket;
                    int outIdx = h * seqLen * seqLen + q * seqLen + k;
                    data[outIdx] = _relativePositionBias[rpbIdx];
                }
            }
        }

        var tensor = new Tensor<T>(new[] { NumHeads, seqLen, seqLen }, data);

        // LRU eviction before insert — evict the LEAST-recently-used seqLen entry when at
        // capacity. Without this, workloads that feed varying sequence lengths (e.g.,
        // dynamic-batch inference) would grow the cache unboundedly and retain potentially
        // hundreds of MB worth of RPB tensors.
        if (_rpbBroadcastCache.Count >= MaxRpbCacheEntries && _rpbCacheLruOrder.First is { } victim)
        {
            _rpbBroadcastCache.Remove(victim.Value);
            _rpbCacheLruOrder.RemoveFirst();
        }
        _rpbBroadcastCache[seqLen] = tensor;
        _rpbCacheLruOrder.AddLast(seqLen);
        return tensor;
    }

    /// <summary>
    /// T5 bidirectional relative-position bucketing function (Raffel et al. 2020).
    /// Half of the buckets handle exact positions in <c>[0, numBuckets/4)</c>, the other half
    /// log-spaced positions out to <paramref name="maxDistance"/>; for bidirectional encoders
    /// the full bucket range is split in two — one half for forward, one half for backward.
    /// </summary>
    /// <remarks>
    /// Mirrors the canonical implementation in the T5/HuggingFace codebase. Both directions
    /// are necessary because T5's encoder attention is bidirectional, so the model must
    /// distinguish "key is N tokens ahead" from "key is N tokens behind".
    /// </remarks>
    private static int RelativePositionBucket(int relativePosition, bool bidirectional, int numBuckets, int maxDistance)
    {
        int bucket = 0;
        int n = numBuckets;
        int relPos;
        if (bidirectional)
        {
            n = numBuckets / 2;
            if (relativePosition > 0) bucket += n; // direction bit: forward = upper half
            relPos = Math.Abs(relativePosition);
        }
        else
        {
            relPos = -Math.Min(relativePosition, 0); // decoder: only attend to past
        }

        int maxExact = n / 2;
        if (relPos < maxExact)
        {
            bucket += relPos;
        }
        else
        {
            // Logarithmic bucketing for distant positions:
            //   bucket = maxExact + log(relPos / maxExact) / log(maxDistance / maxExact) * (n - maxExact)
            double ratio = Math.Log((double)relPos / maxExact) / Math.Log((double)maxDistance / maxExact);
            int relPosIfLarge = maxExact + (int)(ratio * (n - maxExact));
            if (relPosIfLarge > n - 1) relPosIfLarge = n - 1;
            bucket += relPosIfLarge;
        }
        return bucket;
    }

    /// <summary>
    /// Engine-accelerated RMS normalization.
    /// Uses Engine.TensorMultiply for element-wise square, Engine.TensorMatMul for sum reduction,
    /// and Engine.TensorBroadcastDivide/Multiply for normalize+scale.
    /// </summary>
    private Tensor<T> RMSNormEngine(Tensor<T> input, Vector<T> gamma, int dim)
    {
        var shape = input._shape;
        int numVectors = input.Length / dim;

        // Element-wise square via Engine
        var squared = Engine.TensorMultiply(input, input);

        // Sum across last dimension: [numVectors, dim] @ [dim, 1] -> [numVectors, 1]
        // Cache the ones tensor to avoid reallocating on every call
        if (_rmsOnesTensor == null || _rmsOnesTensor.Length != dim)
        {
            var onesVec = new Vector<T>(dim);
            for (int i = 0; i < dim; i++) onesVec[i] = NumOps.One;
            _rmsOnesTensor = new Tensor<T>(new[] { dim, 1 }, onesVec);
        }
        var onesTensor = _rmsOnesTensor;

        var reshapedSquared = squared.Reshape(numVectors, dim);
        var sumSq = Engine.TensorMatMul(reshapedSquared, onesTensor); // [numVectors, 1]

        // mean = sumSq / dim
        var mean = Engine.TensorDivideScalar(sumSq, NumOps.FromDouble(dim));

        // rms = sqrt(mean + eps)
        _rmsEpsTensor ??= new Tensor<T>(new[] { 1, 1 }, new Vector<T>(new[] { NumOps.FromDouble(1e-6) }));
        var meanPlusEps = Engine.TensorBroadcastAdd(mean, _rmsEpsTensor);
        var rms = Engine.TensorSqrt(meanPlusEps); // [numVectors, 1]

        // normalized = input / rms (broadcast along last dim)
        var inputReshaped = input.Reshape(numVectors, dim);
        var normalized = Engine.TensorBroadcastDivide(inputReshaped, rms);

        // scaled = normalized * gamma (broadcast along first dim)
        var gammaTensor = new Tensor<T>(new[] { 1, dim }, gamma);
        var scaled = Engine.TensorBroadcastMultiply(normalized, gammaTensor);

        return scaled.Reshape(shape);
    }

    #region Variant Configuration

    private static int GetEmbeddingDim(string variant) => variant switch
    {
        "T5-XL" => 2048,
        "T5-Large" => 1024,
        _ => 4096 // T5-XXL default
    };

    private static int GetHiddenSize(string variant) => variant switch
    {
        "T5-XL" => 2048,
        "T5-Large" => 1024,
        _ => 4096 // T5-XXL
    };

    private static int GetNumLayers(string variant) => variant switch
    {
        "T5-Large" => 24,
        _ => 24 // T5-XXL and T5-XL both have 24 layers
    };

    private static int GetNumHeads(string variant) => variant switch
    {
        "T5-XL" => 32,
        "T5-Large" => 16,
        _ => 64 // T5-XXL
    };

    /// <summary>
    /// Returns the FFN inner dimension (d_ff) per T5.1.1 official configs.
    /// These ratios are not a clean multiple of HiddenSize because Google tuned them
    /// per-variant rather than as a fixed expansion factor.
    /// </summary>
    private static int GetFfnDim(string variant) => variant switch
    {
        "T5-XL" => 5120,    // T5.1.1-XL  (d_model=2048, d_ff=5120)
        "T5-Large" => 2816, // T5.1.1-Large (d_model=1024, d_ff=2816)
        _ => 10240          // T5.1.1-XXL (d_model=4096, d_ff=10240)
    };

    #endregion
}
