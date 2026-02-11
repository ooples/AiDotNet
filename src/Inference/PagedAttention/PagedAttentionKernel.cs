using System.Buffers;
using System.Runtime.CompilerServices;
using AiDotNet.Inference.Quantization;

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
        _kvCache = kvCache ?? throw new ArgumentNullException(nameof(kvCache));
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
        int numHeads = _config.NumHeads;
        int headDim = _config.HeadDimension;
        int seqLen = _kvCache.GetSequenceLength(sequenceId);

        var blockTable = _kvCache.GetBlockTable(sequenceId);
        if (blockTable == null || seqLen == 0)
        {
            output.Clear();
            return;
        }

        // Allocate working memory
        var scores = new float[seqLen];
        var keyBuffer = new T[numHeads * headDim];
        var valueBuffer = new T[numHeads * headDim];

        // Process each head
        for (int head = 0; head < numHeads; head++)
        {
            int queryOffset = head * headDim;

            // Compute attention scores for all positions
            float maxScore = float.NegativeInfinity;

            for (int pos = 0; pos < seqLen; pos++)
            {
                // Read key from paged cache
                _kvCache.ReadKey(sequenceId, pos, layer, keyBuffer.AsSpan());

                // Compute Q @ K^T for this head
                float score = 0;
                int keyOffset = head * headDim;
                for (int d = 0; d < headDim; d++)
                {
                    score += query[queryOffset + d] * ToFloat(keyBuffer[keyOffset + d]);
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

            // Softmax
            float sumExp = 0;
            for (int pos = 0; pos < seqLen; pos++)
            {
                scores[pos] = MathF.Exp(scores[pos] - maxScore);
                sumExp += scores[pos];
            }

            if (sumExp > 0)
            {
                for (int pos = 0; pos < seqLen; pos++)
                {
                    scores[pos] /= sumExp;
                }
            }

            // Compute weighted sum of values
            var headOutput = new float[headDim];
            for (int pos = 0; pos < seqLen; pos++)
            {
                if (scores[pos] < 1e-10f)
                    continue;

                _kvCache.ReadValue(sequenceId, pos, layer, valueBuffer.AsSpan());

                int valueOffset = head * headDim;
                for (int d = 0; d < headDim; d++)
                {
                    headOutput[d] += scores[pos] * ToFloat(valueBuffer[valueOffset + d]);
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
        int headSize = _config.NumHeads * _config.HeadDimension;

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
        int numHeads = _config.NumHeads;
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

        int numBlocks = blockTable.Length;

        // Per-head accumulators for online softmax
        var maxScores = new float[numHeads];
        var sumExps = new float[numHeads];
        var accumulators = new float[numHeads * headDim];

#if NET5_0_OR_GREATER
        Array.Fill(maxScores, float.NegativeInfinity);
        Array.Fill(sumExps, 0f);
#else
        ArrayPolyfill.Fill(maxScores, float.NegativeInfinity);
        ArrayPolyfill.Fill(sumExps, 0f);
#endif

        var keyBuffer = new T[numHeads * headDim];
        var valueBuffer = new T[numHeads * headDim];

        // Process block by block (tiled computation)
        for (int blockIdx = 0; blockIdx < numBlocks; blockIdx++)
        {
            int blockStart = blockIdx * blockSize;
            int blockEnd = Math.Min(blockStart + blockSize, seqLen);
            int tokensInBlock = blockEnd - blockStart;

            // Process tokens in this block
            for (int tokenOffset = 0; tokenOffset < tokensInBlock; tokenOffset++)
            {
                int pos = blockStart + tokenOffset;

                // Read KV from this position
                _kvCache.ReadKey(sequenceId, pos, layer, keyBuffer.AsSpan());
                _kvCache.ReadValue(sequenceId, pos, layer, valueBuffer.AsSpan());

                // Update each head
                for (int head = 0; head < numHeads; head++)
                {
                    int offset = head * headDim;

                    // Compute score = Q dot K * scale
                    float score = 0;
                    for (int d = 0; d < headDim; d++)
                    {
                        score += query[offset + d] * ToFloat(keyBuffer[offset + d]);
                    }
                    score *= scale;

                    // Apply ALiBi bias: slope[head] * (keyPos - queryPos)
                    // This penalizes distant tokens with a linear bias per head
                    if (alibiSlopes != null)
                    {
                        score += alibiSlopes[head] * (pos - qPos);
                    }

                    // Online softmax update
                    float oldMax = maxScores[head];
                    float newMax = Math.Max(oldMax, score);
                    float expOld = MathF.Exp(oldMax - newMax);
                    float expNew = MathF.Exp(score - newMax);

                    // Update accumulator
                    for (int d = 0; d < headDim; d++)
                    {
                        accumulators[offset + d] = accumulators[offset + d] * expOld + expNew * ToFloat(valueBuffer[offset + d]);
                    }

                    // Update sum and max
                    sumExps[head] = sumExps[head] * expOld + expNew;
                    maxScores[head] = newMax;
                }
            }
        }

        // Normalize and write output
        for (int head = 0; head < numHeads; head++)
        {
            int offset = head * headDim;
            float invSum = sumExps[head] > 0 ? 1.0f / sumExps[head] : 0;

            for (int d = 0; d < headDim; d++)
            {
                output[offset + d] = accumulators[offset + d] * invSum;
            }
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
    /// <param name="wQ">Query weight matrix [hidden_dim, num_heads * head_dim].</param>
    /// <param name="wK">Key weight matrix [hidden_dim, num_heads * head_dim].</param>
    /// <param name="wV">Value weight matrix [hidden_dim, num_heads * head_dim].</param>
    /// <param name="wO">Output weight matrix [num_heads * head_dim, hidden_dim].</param>
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
        int numHeads = _config.NumHeads;
        int headDim = _config.HeadDimension;
        int projDim = numHeads * headDim;
        float scale = 1.0f / MathF.Sqrt(headDim);

        var pool = ArrayPool<float>.Shared;
        var queryBuf = pool.Rent(projDim);
        var keyBuf = pool.Rent(projDim);
        var valueBuf = pool.Rent(projDim);
        var attnBuf = pool.Rent(projDim);

        try
        {
            var query = queryBuf.AsSpan(0, projDim);
            var key = keyBuf.AsSpan(0, projDim);
            var value = valueBuf.AsSpan(0, projDim);
            var attnOutput = attnBuf.AsSpan(0, projDim);

            // Q = hidden @ wQ
            MatVecMul(hiddenStates, wQ, query, hiddenDim, projDim);
            // K = hidden @ wK
            MatVecMul(hiddenStates, wK, key, hiddenDim, projDim);
            // V = hidden @ wV
            MatVecMul(hiddenStates, wV, value, hiddenDim, projDim);

            // Update cache with new K, V
            UpdateCache(key, value, sequenceId, position, layer);

            // Compute attention
            ComputeTiledPagedAttention(query, sequenceId, layer, attnOutput, scale);

            // Project output: out = attn @ wO
            MatVecMul(attnOutput, wO, output, projDim, hiddenDim);
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
        int numHeads = _config.NumHeads;
        int headDim = _config.HeadDimension;
        int projDim = numHeads * headDim;
        float scale = 1.0f / MathF.Sqrt(headDim);

        if (wQ.Cols != hiddenDim || wK.Cols != hiddenDim || wV.Cols != hiddenDim || wO.Cols != projDim)
        {
            throw new ArgumentException("Quantized weight dimensions do not match expected shapes.");
        }

        var pool = ArrayPool<float>.Shared;
        var queryBuf = pool.Rent(projDim);
        var keyBuf = pool.Rent(projDim);
        var valueBuf = pool.Rent(projDim);
        var attnBuf = pool.Rent(projDim);

        try
        {
            var query = queryBuf.AsSpan(0, projDim);
            var key = keyBuf.AsSpan(0, projDim);
            var value = valueBuf.AsSpan(0, projDim);
            var attnOutput = attnBuf.AsSpan(0, projDim);

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
        if (typeof(T) == typeof(float))
            return (float)(object)value!;
        if (typeof(T) == typeof(double))
            return (float)(double)(object)value!;
        if (typeof(T) == typeof(Half))
            return (float)(Half)(object)value!;

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

        return (T)Convert.ChangeType(value, typeof(T))!;
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
    /// <summary>Number of attention heads.</summary>
    public int NumHeads { get; set; } = 32;

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
