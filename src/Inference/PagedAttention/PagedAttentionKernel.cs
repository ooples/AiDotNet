using System.Buffers;
using System.Runtime.CompilerServices;
using AiDotNet.Inference.Quantization;
using AiDotNet.Validation;

namespace AiDotNet.Inference.PagedAttention;

/// <summary>
/// Paged attention kernel that computes attention with block-based KV cache.
/// </summary>
/// <remarks>
/// <para>
/// This kernel performs attention computation using the paged KV cache structure.
/// Instead of accessing KV tensors contiguously, it uses block tables to find
/// the physical locations of each token's KV data.
/// </para>
/// <para><b>For Beginners:</b> Normal attention reads KV cache like reading a book page by page.
///
/// Paged attention is like reading a book where pages are scattered:
/// 1. Look up where each page is stored (block table)
/// 2. Go to that location (physical block)
/// 3. Read the content (KV values)
/// 4. Continue with next page
///
/// The extra lookups add slight overhead, but the memory savings are huge!
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for tensor computations.</typeparam>
internal class PagedAttentionKernel<T>
{
    private readonly PagedKVCache<T> _kvCache;
    private readonly PagedAttentionConfig _config;

    /// <summary>
    /// Gets the configuration.
    /// </summary>
    public PagedAttentionConfig Config => _config;

    /// <summary>
    /// Creates a new paged attention kernel.
    /// </summary>
    public PagedAttentionKernel(PagedKVCache<T> kvCache, PagedAttentionConfig? config = null)
    {
        Guard.NotNull(kvCache);
        _kvCache = kvCache;
        _config = config ?? new PagedAttentionConfig
        {
            NumHeads = kvCache.Config.NumHeads,
            HeadDimension = kvCache.Config.HeadDimension,
            BlockSize = kvCache.Config.BlockSize
        };
    }

    /// <summary>
    /// Computes paged attention for a single query token.
    /// </summary>
    /// <param name="query">Query tensor [num_heads, head_dim].</param>
    /// <param name="sequenceId">Sequence ID for KV cache lookup.</param>
    /// <param name="layer">Layer index.</param>
    /// <param name="output">Output tensor [num_heads, head_dim].</param>
    /// <param name="scale">Attention scale factor (typically 1/sqrt(head_dim)).</param>
    /// <param name="causalMask">Whether to apply causal masking.</param>
    public void ComputeAttention(
        ReadOnlySpan<float> query,
        long sequenceId,
        int layer,
        Span<float> output,
        float scale,
        bool causalMask = true)
    {
        int numKVHeads = _config.NumHeads;
        int numQueryHeads = _config.NumQueryHeads > 0 ? _config.NumQueryHeads : numKVHeads;
        int group = numQueryHeads / numKVHeads; // GQA repeat factor; 1 for standard multi-head attention
        int headDim = _config.HeadDimension;
        int seqLen = _kvCache.GetSequenceLength(sequenceId);

        var blockTable = _kvCache.GetBlockTable(sequenceId);
        if (blockTable == null || seqLen == 0)
        {
            output.Clear();
            return;
        }

        // Sliding-window attention (Mistral-style): a decode query at the last cached position attends only
        // to the most recent WindowSize keys. windowStart clamps the lower bound of every per-position loop
        // below, so the softmax is normalized over the window only. 0 => full causal attention.
        int windowStart = _config.WindowSize > 0 ? Math.Max(0, seqLen - _config.WindowSize) : 0;

        // Allocate working memory. K/V buffers hold the (possibly fewer) KV heads the cache stores; the query
        // and output are laid out over the full query-head count.
        var scores = new float[seqLen];
        var keyBuffer = new T[numKVHeads * headDim];
        var valueBuffer = new T[numKVHeads * headDim];

        // Process each query head; under GQA it reads the KV head it shares (kvHead = head / group).
        for (int head = 0; head < numQueryHeads; head++)
        {
            int queryOffset = head * headDim;
            int kvOffset = (head / group) * headDim;

            // Compute attention scores for all positions
            float maxScore = float.NegativeInfinity;

            for (int pos = windowStart; pos < seqLen; pos++)
            {
                // Read key from paged cache
                _kvCache.ReadKey(sequenceId, pos, layer, keyBuffer.AsSpan());

                // Compute Q @ K^T for this head
                float score = 0;
                for (int d = 0; d < headDim; d++)
                {
                    score += query[queryOffset + d] * ToFloat(keyBuffer[kvOffset + d]);
                }
                score *= scale;

                // Apply causal mask
                if (causalMask && pos > seqLen - 1)
                {
                    score = float.NegativeInfinity;
                }

                scores[pos] = score;
                maxScore = Math.Max(maxScore, score);
            }

            // Softmax (over the window only)
            float sumExp = 0;
            for (int pos = windowStart; pos < seqLen; pos++)
            {
                scores[pos] = MathF.Exp(scores[pos] - maxScore);
                sumExp += scores[pos];
            }

            if (sumExp > 0)
            {
                for (int pos = windowStart; pos < seqLen; pos++)
                {
                    scores[pos] /= sumExp;
                }
            }

            // Compute weighted sum of values
            var headOutput = new float[headDim];
            for (int pos = windowStart; pos < seqLen; pos++)
            {
                if (scores[pos] < 1e-10f)
                    continue;

                _kvCache.ReadValue(sequenceId, pos, layer, valueBuffer.AsSpan());

                for (int d = 0; d < headDim; d++)
                {
                    headOutput[d] += scores[pos] * ToFloat(valueBuffer[kvOffset + d]);
                }
            }

            // Write to output
            int outputOffset = head * headDim;
            for (int d = 0; d < headDim; d++)
            {
                output[outputOffset + d] = headOutput[d];
            }
        }
    }

    /// <summary>
    /// Computes paged attention for a batch of queries.
    /// </summary>
    /// <param name="queries">Query tensors [batch, num_heads, head_dim].</param>
    /// <param name="sequenceIds">Sequence IDs for each batch item.</param>
    /// <param name="layer">Layer index.</param>
    /// <param name="outputs">Output tensors [batch, num_heads, head_dim].</param>
    /// <param name="scale">Attention scale factor.</param>
    public void ComputeBatchedAttention(
        ReadOnlySpan<float> queries,
        long[] sequenceIds,
        int layer,
        Span<float> outputs,
        float scale)
    {
        int batchSize = sequenceIds.Length;
        // Query/output are laid out over the full query-head count (GQA-aware); the cache still stores fewer
        // KV heads internally.
        int numQueryHeads = _config.NumQueryHeads > 0 ? _config.NumQueryHeads : _config.NumHeads;
        int headSize = numQueryHeads * _config.HeadDimension;

        // Process each batch item (could be parallelized)
        for (int b = 0; b < batchSize; b++)
        {
            var query = queries.Slice(b * headSize, headSize);
            var output = outputs.Slice(b * headSize, headSize);
            ComputeAttention(query, sequenceIds[b], layer, output, scale);
        }
    }

    /// <summary>
    /// Computes paged attention with Flash Attention-style tiling.
    /// </summary>
    /// <remarks>
    /// This implementation combines paged memory with tiled computation
    /// for better cache efficiency on CPU.
    /// </remarks>
    public void ComputeTiledPagedAttention(
        ReadOnlySpan<float> query,
        long sequenceId,
        int layer,
        Span<float> output,
        float scale,
        float[]? alibiSlopes = null,
        int queryPosition = -1)
    {
        int numKVHeads = _config.NumHeads;
        int numQueryHeads = _config.NumQueryHeads > 0 ? _config.NumQueryHeads : numKVHeads;
        int group = numQueryHeads / numKVHeads; // GQA repeat factor; 1 for standard multi-head attention
        int headDim = _config.HeadDimension;
        int blockSize = _config.BlockSize;
        int seqLen = _kvCache.GetSequenceLength(sequenceId);

        var blockTable = _kvCache.GetBlockTable(sequenceId);
        if (blockTable == null || seqLen == 0)
        {
            output.Clear();
            return;
        }

        // If queryPosition not specified, default to end of sequence (autoregressive decode)
        int qPos = queryPosition >= 0 ? queryPosition : seqLen - 1;

        // Sliding-window attention: the query attends only to the most recent WindowSize keys (relative to the
        // query position). 0 => full causal attention.
        int windowStart = _config.WindowSize > 0 ? Math.Max(0, (qPos + 1) - _config.WindowSize) : 0;

        int numBlocks = blockTable.Length;

        // Per-QUERY-head online-softmax scratch (softmax is independent per query head, even when several share
        // a KV head under GQA). Pooled: this method runs once per decode token, per layer, per sequence, so a
        // fresh new[] here was the top compute-path allocation site (dotnet-trace). The pooled arrays may be
        // longer than requested; only the used prefix is touched. accumulators is zero-initialized for the used
        // region below; keyBuffer/valueBuffer (sized to the KV heads the cache stores) are fully overwritten by
        // ReadKey/ReadValue before each read.
        var floatPool = ArrayPool<float>.Shared;
        var tPool = ArrayPool<T>.Shared;
        var maxScores = floatPool.Rent(numQueryHeads);
        var sumExps = floatPool.Rent(numQueryHeads);
        var accumulators = floatPool.Rent(numQueryHeads * headDim);
        // Bulk-read the WHOLE KV history for this layer under ONE lock instead of a lock per position: the
        // per-position ReadKey/ReadValue took O(seqLen) contended Monitor.Enter per layer per token — profiled
        // as ~51% of decode time. keyAll/valueAll are laid out [seqLen, numKVHeads*headDim].
        int perPos = numKVHeads * headDim;
        var keyAll = tPool.Rent(seqLen * perPos);
        var valueAll = tPool.Rent(seqLen * perPos);
        try
        {
        _kvCache.ReadKeyValueRange(sequenceId, 0, seqLen, layer,
            keyAll.AsSpan(0, seqLen * perPos), valueAll.AsSpan(0, seqLen * perPos));
        Array.Clear(accumulators, 0, numQueryHeads * headDim);
        for (int head = 0; head < numQueryHeads; head++)
        {
            maxScores[head] = float.NegativeInfinity;
            sumExps[head] = 0f;
        }

        // Process block by block (tiled computation)
        for (int blockIdx = 0; blockIdx < numBlocks; blockIdx++)
        {
            int blockStart = blockIdx * blockSize;
            int blockEnd = Math.Min(blockStart + blockSize, seqLen);
            int tokensInBlock = blockEnd - blockStart;

            // Sliding window: skip blocks entirely before the window start.
            if (blockEnd <= windowStart) continue;

            // Process tokens in this block
            for (int tokenOffset = 0; tokenOffset < tokensInBlock; tokenOffset++)
            {
                int pos = blockStart + tokenOffset;

                // Sliding window: skip positions before the window start.
                if (pos < windowStart) continue;

                // KV for this position was bulk-read into keyAll/valueAll under a single lock above.
                int posBase = pos * perPos;

                // Update each query head; under GQA it reads the KV head it shares (kvHead = head / group).
                for (int head = 0; head < numQueryHeads; head++)
                {
                    int offset = head * headDim;            // query / accumulator / output (per query head)
                    int kvOffset = posBase + (head / group) * headDim; // key / value (per shared KV head, this pos)

                    // Compute score = Q dot K * scale
                    float score = 0;
                    for (int d = 0; d < headDim; d++)
                    {
                        score += query[offset + d] * ToFloat(keyAll[kvOffset + d]);
                    }
                    score *= scale;

                    // Apply ALiBi bias: -slope[head] * |keyPos - queryPos|
                    // Consistent with ALiBiPositionalBiasLayer.ComputeBias which uses -slope * |i - j|
                    if (alibiSlopes != null)
                    {
                        score += -alibiSlopes[head] * Math.Abs(pos - qPos);
                    }

                    // Online softmax update
                    float oldMax = maxScores[head];
                    float newMax = Math.Max(oldMax, score);
                    float expOld = MathF.Exp(oldMax - newMax);
                    float expNew = MathF.Exp(score - newMax);

                    // Update accumulator
                    for (int d = 0; d < headDim; d++)
                    {
                        accumulators[offset + d] = accumulators[offset + d] * expOld + expNew * ToFloat(valueAll[kvOffset + d]);
                    }

                    // Update sum and max
                    sumExps[head] = sumExps[head] * expOld + expNew;
                    maxScores[head] = newMax;
                }
            }
        }

        // Normalize and write output (one entry per query head)
        for (int head = 0; head < numQueryHeads; head++)
        {
            int offset = head * headDim;
            float invSum = sumExps[head] > 0 ? 1.0f / sumExps[head] : 0;

            for (int d = 0; d < headDim; d++)
            {
                output[offset + d] = accumulators[offset + d] * invSum;
            }
        }

        // Sliding-window KV retention: attention for this step is done, so the KV blocks holding positions
        // entirely below the window are now unreachable (this query and every later one start at windowStart),
        // and can be released back to the pool. This makes the WindowSize mask deliver actually-bounded KV
        // memory instead of leaving old blocks allocated. Idempotent, so the per-layer calls within one step
        // are safe (all layers share the same windowStart, so a block freed after layer 0 is never read by a
        // later layer of the same step).
        if (_config.WindowSize > 0 && windowStart > 0)
        {
            _kvCache.EvictBlocksBelow(sequenceId, windowStart);
        }
        }
        finally
        {
            floatPool.Return(maxScores);
            floatPool.Return(sumExps);
            floatPool.Return(accumulators);
            tPool.Return(keyAll);
            tPool.Return(valueAll);
        }
    }

    /// <summary>
    /// Computes causal self-attention for a fresh prefill block whose Q/K/V are supplied directly as
    /// contiguous, already-projected (and, for RoPE models, already-rotated) buffers laid out
    /// [rowLen, numHeads*headDim] in position order.
    /// </summary>
    /// <remarks>
    /// This is the prefill counterpart to <see cref="ComputeTiledPagedAttention"/>: instead of round-tripping
    /// every query through the paged KV cache (block-table lookups + per-token allocations, O(seqLen^2) paged
    /// reads over a full prefill), it reads K/V straight from the contiguous buffers the projection GEMM just
    /// produced. Query position <paramref name="q"/> attends causally to keys [0..q] using the same
    /// online-softmax accumulation as the paged kernel, so results match the per-token paged path. It is
    /// cache-agnostic (the caller persists KV separately for the decode continuation) and does NOT support
    /// sliding-window eviction — callers must keep the per-token paged path when a window is configured.
    /// </remarks>
    /// <param name="queries">Query buffer [rowLen, numQueryHeads*headDim], RoPE-applied if the model uses RoPE.</param>
    /// <param name="keys">Key buffer [rowLen, numKVHeads*headDim] (numKVHeads &lt; numQueryHeads under GQA), RoPE-applied if the model uses RoPE.</param>
    /// <param name="values">Value buffer [rowLen, numKVHeads*headDim].</param>
    /// <param name="rowLen">Number of query/key positions in this prefill block.</param>
    /// <param name="output">Output buffer [rowLen, numQueryHeads*headDim].</param>
    /// <param name="scale">Attention scale factor (typically 1/sqrt(head_dim)).</param>
    /// <param name="alibiSlopes">Optional per-head ALiBi slopes; null for RoPE/no positional bias.</param>
    public void ComputeContiguousCausalPrefill(
        ReadOnlySpan<float> queries,
        ReadOnlySpan<float> keys,
        ReadOnlySpan<float> values,
        int rowLen,
        Span<float> output,
        float scale,
        float[]? alibiSlopes = null)
    {
        int numKVHeads = _config.NumHeads;
        int numQueryHeads = _config.NumQueryHeads > 0 ? _config.NumQueryHeads : numKVHeads;
        int group = numQueryHeads / numKVHeads; // GQA repeat factor; 1 for standard multi-head attention
        int headDim = _config.HeadDimension;
        int qProjDim = numQueryHeads * headDim;  // query/output row stride
        int kvProjDim = numKVHeads * headDim;    // key/value row stride (fewer heads under GQA)

        var pool = ArrayPool<float>.Shared;
        var accumulators = pool.Rent(numQueryHeads * headDim);
        var maxScores = pool.Rent(numQueryHeads);
        var sumExps = pool.Rent(numQueryHeads);
        try
        {
            for (int q = 0; q < rowLen; q++)
            {
                int qBase = q * qProjDim;

                // Reset per-query online-softmax accumulators.
                Array.Clear(accumulators, 0, numQueryHeads * headDim);
                for (int head = 0; head < numQueryHeads; head++)
                {
                    maxScores[head] = float.NegativeInfinity;
                    sumExps[head] = 0f;
                }

                // Causal: query position q attends only to key positions [0..q].
                for (int pos = 0; pos <= q; pos++)
                {
                    int kBase = pos * kvProjDim;
                    for (int head = 0; head < numQueryHeads; head++)
                    {
                        int offset = head * headDim;             // query / accumulator / output
                        int kvOffset = (head / group) * headDim; // shared KV head

                        float score = 0;
                        for (int d = 0; d < headDim; d++)
                        {
                            score += queries[qBase + offset + d] * keys[kBase + kvOffset + d];
                        }
                        score *= scale;

                        // ALiBi bias: -slope[head] * |keyPos - queryPos|; causal => q - pos.
                        if (alibiSlopes != null)
                        {
                            score += -alibiSlopes[head] * (q - pos);
                        }

                        // Online-softmax update (matches ComputeTiledPagedAttention).
                        float oldMax = maxScores[head];
                        float newMax = Math.Max(oldMax, score);
                        float expOld = MathF.Exp(oldMax - newMax);
                        float expNew = MathF.Exp(score - newMax);

                        for (int d = 0; d < headDim; d++)
                        {
                            accumulators[offset + d] =
                                accumulators[offset + d] * expOld + expNew * values[kBase + kvOffset + d];
                        }

                        sumExps[head] = sumExps[head] * expOld + expNew;
                        maxScores[head] = newMax;
                    }
                }

                // Normalize and write this query's output.
                for (int head = 0; head < numQueryHeads; head++)
                {
                    int offset = head * headDim;
                    float invSum = sumExps[head] > 0 ? 1.0f / sumExps[head] : 0;
                    for (int d = 0; d < headDim; d++)
                    {
                        output[qBase + offset + d] = accumulators[offset + d] * invSum;
                    }
                }
            }
        }
        finally
        {
            pool.Return(accumulators);
            pool.Return(maxScores);
            pool.Return(sumExps);
        }
    }

    /// <summary>
    /// Updates the KV cache with new key and value tensors.
    /// </summary>
    /// <param name="key">Key tensor [num_heads, head_dim].</param>
    /// <param name="value">Value tensor [num_heads, head_dim].</param>
    /// <param name="sequenceId">Sequence ID.</param>
    /// <param name="position">Token position.</param>
    /// <param name="layer">Layer index.</param>
    public void UpdateCache(
        ReadOnlySpan<float> key,
        ReadOnlySpan<float> value,
        long sequenceId,
        int position,
        int layer)
    {
        // Ensure logical length and capacity for this position.
        int requiredLength = position + 1;
        int currentLength = _kvCache.GetSequenceLength(sequenceId);
        if (requiredLength > currentLength)
        {
            int additionalTokens = requiredLength - currentLength;
            if (!_kvCache.ExtendSequence(sequenceId, additionalTokens))
            {
                throw new InvalidOperationException(
                    $"Failed to extend PagedKVCache sequence {sequenceId} to length {requiredLength}.");
            }
        }

        // Convert and write
        var keyT = ConvertArray(key);
        var valueT = ConvertArray(value);

        _kvCache.WriteKey(sequenceId, position, layer, keyT);
        _kvCache.WriteValue(sequenceId, position, layer, valueT);
    }

    /// <summary>
    /// Performs a full forward pass: projects QKV, updates cache, computes attention.
    /// </summary>
    /// <param name="hiddenStates">Input hidden states [hidden_dim].</param>
    /// <param name="wQ">Query weight matrix [hidden_dim, numQueryHeads * head_dim].</param>
    /// <param name="wK">Key weight matrix [hidden_dim, numKVHeads * head_dim] (numKVHeads &lt; numQueryHeads under GQA).</param>
    /// <param name="wV">Value weight matrix [hidden_dim, numKVHeads * head_dim].</param>
    /// <param name="wO">Output weight matrix [numQueryHeads * head_dim, hidden_dim].</param>
    /// <param name="sequenceId">Sequence ID.</param>
    /// <param name="position">Current token position.</param>
    /// <param name="layer">Layer index.</param>
    /// <param name="output">Output tensor [hidden_dim].</param>
    public void Forward(
        ReadOnlySpan<float> hiddenStates,
        ReadOnlySpan<float> wQ,
        ReadOnlySpan<float> wK,
        ReadOnlySpan<float> wV,
        ReadOnlySpan<float> wO,
        long sequenceId,
        int position,
        int layer,
        Span<float> output)
    {
        int hiddenDim = hiddenStates.Length;
        int numKVHeads = _config.NumHeads;
        int numQueryHeads = _config.NumQueryHeads > 0 ? _config.NumQueryHeads : numKVHeads;
        int headDim = _config.HeadDimension;
        // Q and O are laid out over the query heads; K/V over the (possibly fewer) KV heads the cache stores.
        // Under standard multi-head attention the two counts are equal. Mirrors HF q_proj vs k_proj/v_proj.
        int qProjDim = numQueryHeads * headDim;
        int kvProjDim = numKVHeads * headDim;
        float scale = 1.0f / MathF.Sqrt(headDim);

        var pool = ArrayPool<float>.Shared;
        var queryBuf = pool.Rent(qProjDim);
        var keyBuf = pool.Rent(kvProjDim);
        var valueBuf = pool.Rent(kvProjDim);
        var attnBuf = pool.Rent(qProjDim);

        try
        {
            var query = queryBuf.AsSpan(0, qProjDim);
            var key = keyBuf.AsSpan(0, kvProjDim);
            var value = valueBuf.AsSpan(0, kvProjDim);
            var attnOutput = attnBuf.AsSpan(0, qProjDim);

            // Q = hidden @ wQ (query heads)
            MatVecMul(hiddenStates, wQ, query, hiddenDim, qProjDim);
            // K = hidden @ wK (KV heads)
            MatVecMul(hiddenStates, wK, key, hiddenDim, kvProjDim);
            // V = hidden @ wV (KV heads)
            MatVecMul(hiddenStates, wV, value, hiddenDim, kvProjDim);

            // Update cache with new K, V
            UpdateCache(key, value, sequenceId, position, layer);

            // Compute attention (GQA-aware: repeats each KV head across its query-head group)
            ComputeTiledPagedAttention(query, sequenceId, layer, attnOutput, scale);

            // Project output: out = attn @ wO
            MatVecMul(attnOutput, wO, output, qProjDim, hiddenDim);
        }
        finally
        {
            pool.Return(queryBuf);
            pool.Return(keyBuf);
            pool.Return(valueBuf);
            pool.Return(attnBuf);
        }
    }

    public void ForwardQuantized(
        ReadOnlySpan<float> hiddenStates,
        in Int8WeightOnlyQuantization.QuantizedWeights wQ,
        in Int8WeightOnlyQuantization.QuantizedWeights wK,
        in Int8WeightOnlyQuantization.QuantizedWeights wV,
        in Int8WeightOnlyQuantization.QuantizedWeights wO,
        long sequenceId,
        int position,
        int layer,
        Span<float> output)
    {
        int hiddenDim = hiddenStates.Length;
        int numKVHeads = _config.NumHeads;
        int numQueryHeads = _config.NumQueryHeads > 0 ? _config.NumQueryHeads : numKVHeads;
        int headDim = _config.HeadDimension;
        // Q/O over query heads, K/V over the (possibly fewer) KV heads — same asymmetry as HF q_proj vs k_proj.
        int qProjDim = numQueryHeads * headDim;
        int kvProjDim = numKVHeads * headDim;
        float scale = 1.0f / MathF.Sqrt(headDim);

        if (wQ.Cols != hiddenDim || wK.Cols != hiddenDim || wV.Cols != hiddenDim || wO.Cols != qProjDim)
        {
            throw new ArgumentException("Quantized weight dimensions do not match expected shapes.");
        }

        var pool = ArrayPool<float>.Shared;
        var queryBuf = pool.Rent(qProjDim);
        var keyBuf = pool.Rent(kvProjDim);
        var valueBuf = pool.Rent(kvProjDim);
        var attnBuf = pool.Rent(qProjDim);

        try
        {
            var query = queryBuf.AsSpan(0, qProjDim);
            var key = keyBuf.AsSpan(0, kvProjDim);
            var value = valueBuf.AsSpan(0, kvProjDim);
            var attnOutput = attnBuf.AsSpan(0, qProjDim);

            MatVecMulInt8(hiddenStates, wQ, query);
            MatVecMulInt8(hiddenStates, wK, key);
            MatVecMulInt8(hiddenStates, wV, value);

            UpdateCache(key, value, sequenceId, position, layer);
            ComputeTiledPagedAttention(query, sequenceId, layer, attnOutput, scale);
            MatVecMulInt8(attnOutput, wO, output);
        }
        finally
        {
            pool.Return(queryBuf);
            pool.Return(keyBuf);
            pool.Return(valueBuf);
            pool.Return(attnBuf);
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

        if (vec.Length != cols)
            throw new ArgumentException("Input vector length must match quantized matrix column count.", nameof(vec));
        if (output.Length < rows)
            throw new ArgumentException("Output span too small for quantized matvec.", nameof(output));

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

    private static float ToFloat(T value)
    {
        object boxed = value ?? throw new InvalidOperationException("Unexpected null value in PagedAttention type conversion.");
        if (typeof(T) == typeof(float))
            return (float)boxed;
        if (typeof(T) == typeof(double))
            return (float)(double)boxed;
        if (typeof(T) == typeof(Half))
            return (float)(Half)boxed;

        return Convert.ToSingle(value);
    }

    private static T FromFloat(float value)
    {
        if (typeof(T) == typeof(float))
            return (T)(object)value;
        if (typeof(T) == typeof(double))
            return (T)(object)(double)value;
        if (typeof(T) == typeof(Half))
            return (T)(object)(Half)value;

        object converted = Convert.ChangeType(value, typeof(T))
            ?? throw new InvalidOperationException($"Failed to convert float value to {typeof(T).Name}.");
        return (T)converted;
    }

    private static T[] ConvertArray(ReadOnlySpan<float> source)
    {
        if (typeof(T) == typeof(float))
        {
            // Safe: runtime-verified T == float.
            // Return a rooted array so GC cannot collect it while spans are in use.
            return (T[])(object)source.ToArray();
        }

        var result = new T[source.Length];
        for (int i = 0; i < source.Length; i++)
        {
            result[i] = FromFloat(source[i]);
        }
        return result;
    }
}

/// <summary>
/// Configuration for paged attention kernel.
/// </summary>
internal class PagedAttentionConfig
{
    /// <summary>
    /// Number of KEY/VALUE attention heads. This is the head count the paged KV cache physically stores, so
    /// it is always the cache's head count. Under multi-head attention it equals the number of query heads;
    /// under grouped-query attention (GQA) it is smaller (see <see cref="NumQueryHeads"/>).
    /// </summary>
    public int NumHeads { get; set; } = 32;

    /// <summary>
    /// Number of QUERY attention heads. Under grouped-query attention the query heads outnumber the KV heads
    /// (<see cref="NumHeads"/>): each KV head is shared by <c>NumQueryHeads / NumHeads</c> query heads, and
    /// query head <c>q</c> reads KV head <c>q / (NumQueryHeads / NumHeads)</c>. 0 (the default) means "same as
    /// <see cref="NumHeads"/>" — standard multi-head attention with no KV sharing.
    /// </summary>
    public int NumQueryHeads { get; set; }

    /// <summary>Dimension of each head.</summary>
    public int HeadDimension { get; set; } = 128;

    /// <summary>Tokens per block.</summary>
    public int BlockSize { get; set; } = 16;

    /// <summary>Whether to use Flash Attention-style tiling.</summary>
    public bool UseTiling { get; set; } = true;

    /// <summary>Maximum batch size for batched attention.</summary>
    public int MaxBatchSize { get; set; } = 256;

    /// <summary>Whether to use parallel processing for batched attention.</summary>
    public bool UseParallel { get; set; } = true;

    /// <summary>
    /// Sliding-window attention size (Mistral/Mixtral-style SWA). When &gt; 0, each query attends only to the
    /// most recent <c>WindowSize</c> key positions, bounding both the attention span and (with paged-cache
    /// eviction) the KV memory. 0 disables the window (full causal attention).
    /// </summary>
    public int WindowSize { get; set; }
}

/// <summary>
/// Integrates PagedAttention with ContinuousBatcher for high-throughput serving.
/// </summary>
/// <typeparam name="T">Numeric type.</typeparam>
internal class PagedAttentionServer<T> : IDisposable
{
    private readonly PagedKVCache<T> _kvCache;
    private readonly PagedAttentionKernel<T> _kernel;
    private readonly object _lock = new();
    private bool _disposed;

    /// <summary>
    /// Gets the KV cache.
    /// </summary>
    public PagedKVCache<T> KVCache => _kvCache;

    /// <summary>
    /// Gets the attention kernel.
    /// </summary>
    public PagedAttentionKernel<T> Kernel => _kernel;

    /// <summary>
    /// Creates a new paged attention server.
    /// </summary>
    public PagedAttentionServer(PagedKVCacheConfig config)
    {
        _kvCache = new PagedKVCache<T>(config);
        _kernel = new PagedAttentionKernel<T>(_kvCache);
    }

    /// <summary>
    /// Creates a server for a specific model.
    /// </summary>
    public static PagedAttentionServer<T> ForModel(string modelName, long availableMemoryBytes)
    {
        var config = PagedKVCacheConfig.ForModel(modelName, availableMemoryBytes);
        return new PagedAttentionServer<T>(config);
    }

    /// <summary>
    /// Registers a new sequence.
    /// </summary>
    public bool RegisterSequence(long sequenceId, int promptLength)
    {
        lock (_lock)
        {
            return _kvCache.AllocateSequence(sequenceId, promptLength);
        }
    }

    /// <summary>
    /// Unregisters a sequence and frees its resources.
    /// </summary>
    public void UnregisterSequence(long sequenceId)
    {
        lock (_lock)
        {
            _kvCache.FreeSequence(sequenceId);
        }
    }

    /// <summary>
    /// Forks a sequence for beam search.
    /// </summary>
    public bool ForkSequence(long sourceId, long[] newIds)
    {
        lock (_lock)
        {
            foreach (var newId in newIds)
            {
                if (!_kvCache.ForkSequence(sourceId, newId))
                    return false;
            }
            return true;
        }
    }

    /// <summary>
    /// Processes a batch step for multiple sequences.
    /// </summary>
    public void ProcessBatchStep(
        ReadOnlySpan<float> queries,
        long[] sequenceIds,
        int layer,
        Span<float> outputs,
        float scale)
    {
        _kernel.ComputeBatchedAttention(queries, sequenceIds, layer, outputs, scale);
    }

    /// <summary>
    /// Gets server statistics.
    /// </summary>
    public PagedKVCacheStats GetStats() => _kvCache.GetStats();

    /// <summary>
    /// Releases resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _kvCache.Dispose();
    }
}
