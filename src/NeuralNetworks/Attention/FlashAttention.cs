

namespace AiDotNet.NeuralNetworks.Attention;

/// <summary>
/// Implements the Flash Attention algorithm for memory-efficient scaled dot-product attention.
/// </summary>
/// <remarks>
/// <para>
/// Flash Attention is a breakthrough algorithm that computes exact attention without materializing
/// the full N x N attention matrix. It achieves this through:
/// 1. Tiled computation that processes attention in blocks
/// 2. Online softmax algorithm that computes softmax incrementally
/// 3. Careful memory management to minimize HBM (GPU main memory) access
/// </para>
/// <para><b>For Beginners:</b> Flash Attention is a clever way to compute attention faster.
///
/// The problem with standard attention:
/// - Creates a huge N x N matrix (N = sequence length)
/// - For 4096 tokens, that's 16 million numbers to store!
/// - Reading/writing this matrix is slow (memory bandwidth limited)
///
/// Flash Attention's solution:
/// - Process in small blocks that fit in fast cache memory
/// - Compute softmax incrementally (online softmax)
/// - Never create the full attention matrix
///
/// Results:
/// - 2-4x faster than standard attention
/// - Uses O(N) memory instead of O(N^2)
/// - Enables much longer sequences
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for computations (typically float or double).</typeparam>
internal static class FlashAttention<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Computes Flash Attention: softmax(Q @ K^T / sqrt(d)) @ V without materializing the full attention matrix.
    /// </summary>
    /// <param name="query">Query tensor of shape [batch, seqLen, headDim] or [batch, heads, seqLen, headDim].</param>
    /// <param name="key">Key tensor of shape [batch, seqLen, headDim] or [batch, heads, seqLen, headDim].</param>
    /// <param name="value">Value tensor of shape [batch, seqLen, headDim] or [batch, heads, seqLen, headDim].</param>
    /// <param name="config">Flash Attention configuration.</param>
    /// <param name="queryOffset">
    /// Optional offset for causal masking when <paramref name="query"/> represents a window into a longer KV sequence.
    /// Use this for KV-cached decoding where Q is the newly appended tokens and K/V contain the full cached sequence.
    /// </param>
    /// <param name="attentionBias">
    /// Optional additive bias tensor added to attention scores before softmax.
    /// For 4D inputs: shape [heads, seqLenQ, seqLenKV] or [batch, heads, seqLenQ, seqLenKV].
    /// For 3D inputs: shape [seqLenQ, seqLenKV] or [batch, seqLenQ, seqLenKV].
    /// Used for ALiBi positional bias, relative position encodings, or custom attention masks.
    /// </param>
    /// <returns>Output tensor of same shape as query, and optionally attention weights if configured.</returns>
    public static (Tensor<T> Output, Tensor<T>? AttentionWeights) Forward(
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        FlashAttentionConfig? config = null,
        int queryOffset = 0,
        Tensor<T>? attentionBias = null)
    {
        config ??= FlashAttentionConfig.Default;

        // Validate input shapes
        ValidateInputs(query, key, value);

        // Determine if inputs are 3D [batch, seq, dim] or 4D [batch, heads, seq, dim]
        bool is4D = query.Shape.Length == 4;

        int seqLenQ = is4D ? query.Shape[2] : query.Shape[1];
        int seqLenKV = is4D ? key.Shape[2] : key.Shape[1];
        if (queryOffset < 0 || queryOffset + seqLenQ > seqLenKV)
        {
            throw new ArgumentOutOfRangeException(
                nameof(queryOffset),
                $"queryOffset ({queryOffset}) must satisfy 0 <= queryOffset and queryOffset + seqLenQ ({seqLenQ}) <= seqLenKV ({seqLenKV}).");
        }

        return is4D
            ? Forward4D(query, key, value, config, queryOffset, attentionBias)
            : Forward3D(query, key, value, config, queryOffset, attentionBias);
    }

    /// <summary>
    /// Flash Attention for 3D tensors [batch, seqLen, headDim].
    /// </summary>
    private static (Tensor<T> Output, Tensor<T>? AttentionWeights) Forward3D(
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        FlashAttentionConfig config,
        int queryOffset,
        Tensor<T>? attentionBias)
    {
        int batchSize = query.Shape[0];
        int seqLenQ = query.Shape[1];
        int seqLenKV = key.Shape[1];
        int headDim = query.Shape[2];

        // Compute scale factor
        T scale = config.ScaleFactor.HasValue
            ? NumOps.FromDouble(config.ScaleFactor.Value)
            : NumOps.FromDouble(1.0 / Math.Sqrt(headDim));

        // Initialize output tensor
        var output = new Tensor<T>(query.Shape);

        // Optional: materialize attention weights for debugging
        Tensor<T>? attentionWeights = config.ReturnAttentionWeights
            ? new Tensor<T>(new[] { batchSize, seqLenQ, seqLenKV })
            : null;

        // Process each batch
        for (int b = 0; b < batchSize; b++)
        {
            FlashAttentionCore(
                query, key, value, output, attentionWeights,
                b, 0, seqLenQ, seqLenKV, headDim, scale, config, queryOffset, attentionBias);
        }

        return (output, attentionWeights);
    }

    /// <summary>
    /// Flash Attention for 4D tensors [batch, heads, seqLen, headDim].
    /// </summary>
    private static (Tensor<T> Output, Tensor<T>? AttentionWeights) Forward4D(
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        FlashAttentionConfig config,
        int queryOffset,
        Tensor<T>? attentionBias)
    {
        int batchSize = query.Shape[0];
        int numHeads = query.Shape[1];
        int seqLenQ = query.Shape[2];
        int seqLenKV = key.Shape[2];
        int headDim = query.Shape[3];

        // Compute scale factor
        T scale = config.ScaleFactor.HasValue
            ? NumOps.FromDouble(config.ScaleFactor.Value)
            : NumOps.FromDouble(1.0 / Math.Sqrt(headDim));

        // Initialize output tensor
        var output = new Tensor<T>(query.Shape);

        // Optional: materialize attention weights for debugging
        Tensor<T>? attentionWeights = config.ReturnAttentionWeights
            ? new Tensor<T>(new[] { batchSize, numHeads, seqLenQ, seqLenKV })
            : null;

        // Process each batch and head
        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                FlashAttentionCore4D(
                    query, key, value, output, attentionWeights,
                    b, h, seqLenQ, seqLenKV, headDim, scale, config, queryOffset, attentionBias);
            }
        }

        return (output, attentionWeights);
    }

    /// <summary>
    /// Core Flash Attention algorithm using tiled computation and online softmax.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This implements Algorithm 1 from the Flash Attention paper:
    /// 1. Divide Q into blocks of size Br (BlockSizeQ)
    /// 2. Divide K, V into blocks of size Bc (BlockSizeKV)
    /// 3. For each Q block, iterate over all K,V blocks
    /// 4. Use online softmax to compute attention incrementally
    /// 5. Update output using rescaling trick
    /// </para>
    /// </remarks>
    private static void FlashAttentionCore(
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        Tensor<T> output,
        Tensor<T>? attentionWeights,
        int batch,
        int head,
        int seqLenQ,
        int seqLenKV,
        int headDim,
        T scale,
        FlashAttentionConfig config,
        int queryOffset,
        Tensor<T>? attentionBias = null)
    {
        int blockSizeQ = Math.Min(config.BlockSizeQ, seqLenQ);
        int blockSizeKV = Math.Min(config.BlockSizeKV, seqLenKV);

        // Number of blocks
        int numBlocksQ = (seqLenQ + blockSizeQ - 1) / blockSizeQ;
        int numBlocksKV = (seqLenKV + blockSizeKV - 1) / blockSizeKV;

        // Process each query block
        for (int qBlock = 0; qBlock < numBlocksQ; qBlock++)
        {
            int qStart = qBlock * blockSizeQ;
            int qEnd = Math.Min(qStart + blockSizeQ, seqLenQ);
            int qBlockSize = qEnd - qStart;

            // Initialize per-row statistics for online softmax
            // m_i = running maximum, l_i = running sum of exp(x - m)
            var rowMax = new T[qBlockSize];      // m_i
            var rowSum = new T[qBlockSize];      // l_i
            var outputAcc = new T[qBlockSize, headDim];  // O_i accumulator

            // Initialize to -infinity for max, 0 for sum
            T negInf = NumOps.FromDouble(double.NegativeInfinity);
            for (int i = 0; i < qBlockSize; i++)
            {
                rowMax[i] = negInf;
                rowSum[i] = NumOps.Zero;
            }

            // Iterate over key/value blocks
            for (int kvBlock = 0; kvBlock < numBlocksKV; kvBlock++)
            {
                int kvStart = kvBlock * blockSizeKV;
                int kvEnd = Math.Min(kvStart + blockSizeKV, seqLenKV);
                int kvBlockSize = kvEnd - kvStart;

                // Apply causal mask: skip blocks that are entirely masked
                if (config.UseCausalMask && kvStart > queryOffset + qEnd - 1)
                {
                    continue;
                }

                // Compute attention scores for this block: S_ij = Q_i @ K_j^T * scale
                var scores = new T[qBlockSize, kvBlockSize];

                for (int qi = 0; qi < qBlockSize; qi++)
                {
                    int qIdx = qStart + qi;

                    for (int kj = 0; kj < kvBlockSize; kj++)
                    {
                        int kIdx = kvStart + kj;

                        // Apply causal mask
                        if (config.UseCausalMask && kIdx > queryOffset + qIdx)
                        {
                            scores[qi, kj] = negInf;
                            continue;
                        }

                        // Dot product: Q[batch, qIdx, :] @ K[batch, kIdx, :]
                        T dotProduct = NumOps.Zero;
                        for (int d = 0; d < headDim; d++)
                        {
                            T qVal = query[new[] { batch, qIdx, d }];
                            T kVal = key[new[] { batch, kIdx, d }];
                            dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(qVal, kVal));
                        }

                        T score = NumOps.Multiply(dotProduct, scale);

                        // Add attention bias (e.g., ALiBi positional bias)
                        if (attentionBias != null)
                        {
                            T bias = attentionBias.Rank switch
                            {
                                3 => attentionBias[new[] { batch, qIdx, kIdx }],  // [batch, seqQ, seqKV]
                                2 => attentionBias[new[] { qIdx, kIdx }],          // [seqQ, seqKV]
                                _ => NumOps.Zero
                            };
                            score = NumOps.Add(score, bias);
                        }

                        scores[qi, kj] = score;
                    }
                }

                // Online softmax update for each row
                for (int qi = 0; qi < qBlockSize; qi++)
                {
                    // Find max of current block
                    T blockMax = negInf;
                    for (int kj = 0; kj < kvBlockSize; kj++)
                    {
                        if (NumOps.GreaterThan(scores[qi, kj], blockMax))
                        {
                            blockMax = scores[qi, kj];
                        }
                    }

                    // Update running max: m_new = max(m_old, blockMax)
                    T mOld = rowMax[qi];
                    T mNew = NumOps.GreaterThan(blockMax, mOld) ? blockMax : mOld;

                    // Compute correction factor: exp(m_old - m_new)
                    T correction = NumOps.Exp(NumOps.Subtract(mOld, mNew));

                    // Compute exp(scores - m_new) and sum
                    T blockSum = NumOps.Zero;
                    var expScores = new T[kvBlockSize];

                    for (int kj = 0; kj < kvBlockSize; kj++)
                    {
                        expScores[kj] = NumOps.Exp(NumOps.Subtract(scores[qi, kj], mNew));
                        blockSum = NumOps.Add(blockSum, expScores[kj]);
                    }

                    // Update running sum: l_new = l_old * correction + blockSum
                    T lOld = rowSum[qi];
                    T lNew = NumOps.Add(NumOps.Multiply(lOld, correction), blockSum);

                    // Update output accumulator: O_new = O_old * (l_old * correction / l_new) + (exp @ V) / l_new
                    // Simplified: O_new = O_old * correction * (l_old / l_new) + (exp @ V) / l_new
                    T outputScale = NumericalStabilityHelper.SafeDiv(
                        NumOps.Multiply(lOld, correction), lNew);

                    // Scale existing output
                    for (int d = 0; d < headDim; d++)
                    {
                        outputAcc[qi, d] = NumOps.Multiply(outputAcc[qi, d], outputScale);
                    }

                    // Add contribution from current block: (exp @ V) / l_new
                    T valueScale = NumericalStabilityHelper.SafeDiv(NumOps.One, lNew);

                    for (int kj = 0; kj < kvBlockSize; kj++)
                    {
                        int kIdx = kvStart + kj;
                        T weight = NumOps.Multiply(expScores[kj], valueScale);

                        for (int d = 0; d < headDim; d++)
                        {
                            T vVal = value[new[] { batch, kIdx, d }];
                            outputAcc[qi, d] = NumOps.Add(outputAcc[qi, d], NumOps.Multiply(weight, vVal));
                        }

                        // Optionally store attention weights
                        if (attentionWeights != null)
                        {
                            int qIdx = qStart + qi;
                            // Note: Final weights need rescaling after all blocks are processed
                            // For now, store unnormalized weights
                            attentionWeights[new[] { batch, qIdx, kIdx }] = weight;
                        }
                    }

                    // Update statistics
                    rowMax[qi] = mNew;
                    rowSum[qi] = lNew;
                }
            }

            // Write output block
            for (int qi = 0; qi < qBlockSize; qi++)
            {
                int qIdx = qStart + qi;
                for (int d = 0; d < headDim; d++)
                {
                    output[new[] { batch, qIdx, d }] = outputAcc[qi, d];
                }
            }
        }
    }

    /// <summary>
    /// Core Flash Attention algorithm for 4D tensors [batch, heads, seq, dim].
    /// </summary>
    private static void FlashAttentionCore4D(
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        Tensor<T> output,
        Tensor<T>? attentionWeights,
        int batch,
        int head,
        int seqLenQ,
        int seqLenKV,
        int headDim,
        T scale,
        FlashAttentionConfig config,
        int queryOffset,
        Tensor<T>? attentionBias = null)
    {
        int blockSizeQ = Math.Min(config.BlockSizeQ, seqLenQ);
        int blockSizeKV = Math.Min(config.BlockSizeKV, seqLenKV);

        int numBlocksQ = (seqLenQ + blockSizeQ - 1) / blockSizeQ;
        int numBlocksKV = (seqLenKV + blockSizeKV - 1) / blockSizeKV;

        T negInf = NumOps.FromDouble(double.NegativeInfinity);

        for (int qBlock = 0; qBlock < numBlocksQ; qBlock++)
        {
            int qStart = qBlock * blockSizeQ;
            int qEnd = Math.Min(qStart + blockSizeQ, seqLenQ);
            int qBlockSize = qEnd - qStart;

            var rowMax = new T[qBlockSize];
            var rowSum = new T[qBlockSize];
            var outputAcc = new T[qBlockSize, headDim];

            for (int i = 0; i < qBlockSize; i++)
            {
                rowMax[i] = negInf;
                rowSum[i] = NumOps.Zero;
            }

            for (int kvBlock = 0; kvBlock < numBlocksKV; kvBlock++)
            {
                int kvStart = kvBlock * blockSizeKV;
                int kvEnd = Math.Min(kvStart + blockSizeKV, seqLenKV);
                int kvBlockSize = kvEnd - kvStart;

                if (config.UseCausalMask && kvStart > queryOffset + qEnd - 1)
                {
                    continue;
                }

                var scores = new T[qBlockSize, kvBlockSize];

                // Compute attention scores
                for (int qi = 0; qi < qBlockSize; qi++)
                {
                    int qIdx = qStart + qi;

                    for (int kj = 0; kj < kvBlockSize; kj++)
                    {
                        int kIdx = kvStart + kj;

                        if (config.UseCausalMask && kIdx > queryOffset + qIdx)
                        {
                            scores[qi, kj] = negInf;
                            continue;
                        }

                        T dotProduct = NumOps.Zero;
                        for (int d = 0; d < headDim; d++)
                        {
                            T qVal = query[new[] { batch, head, qIdx, d }];
                            T kVal = key[new[] { batch, head, kIdx, d }];
                            dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(qVal, kVal));
                        }

                        T score = NumOps.Multiply(dotProduct, scale);

                        // Add attention bias (e.g., ALiBi positional bias)
                        if (attentionBias != null)
                        {
                            T bias = attentionBias.Rank switch
                            {
                                4 => attentionBias[new[] { batch, head, qIdx, kIdx }],  // [batch, heads, seqQ, seqKV]
                                3 => attentionBias[new[] { head, qIdx, kIdx }],          // [heads, seqQ, seqKV]
                                _ => NumOps.Zero
                            };
                            score = NumOps.Add(score, bias);
                        }

                        scores[qi, kj] = score;
                    }
                }

                // Online softmax and output update
                for (int qi = 0; qi < qBlockSize; qi++)
                {
                    T blockMax = negInf;
                    for (int kj = 0; kj < kvBlockSize; kj++)
                    {
                        if (NumOps.GreaterThan(scores[qi, kj], blockMax))
                        {
                            blockMax = scores[qi, kj];
                        }
                    }

                    T mOld = rowMax[qi];
                    T mNew = NumOps.GreaterThan(blockMax, mOld) ? blockMax : mOld;
                    T correction = NumOps.Exp(NumOps.Subtract(mOld, mNew));

                    T blockSum = NumOps.Zero;
                    var expScores = new T[kvBlockSize];

                    for (int kj = 0; kj < kvBlockSize; kj++)
                    {
                        expScores[kj] = NumOps.Exp(NumOps.Subtract(scores[qi, kj], mNew));
                        blockSum = NumOps.Add(blockSum, expScores[kj]);
                    }

                    T lOld = rowSum[qi];
                    T lNew = NumOps.Add(NumOps.Multiply(lOld, correction), blockSum);

                    T outputScale = NumericalStabilityHelper.SafeDiv(
                        NumOps.Multiply(lOld, correction), lNew);

                    for (int d = 0; d < headDim; d++)
                    {
                        outputAcc[qi, d] = NumOps.Multiply(outputAcc[qi, d], outputScale);
                    }

                    T valueScale = NumericalStabilityHelper.SafeDiv(NumOps.One, lNew);

                    for (int kj = 0; kj < kvBlockSize; kj++)
                    {
                        int kIdx = kvStart + kj;
                        T weight = NumOps.Multiply(expScores[kj], valueScale);

                        for (int d = 0; d < headDim; d++)
                        {
                            T vVal = value[new[] { batch, head, kIdx, d }];
                            outputAcc[qi, d] = NumOps.Add(outputAcc[qi, d], NumOps.Multiply(weight, vVal));
                        }

                        if (attentionWeights != null)
                        {
                            int qIdx = qStart + qi;
                            attentionWeights[new[] { batch, head, qIdx, kIdx }] = weight;
                        }
                    }

                    rowMax[qi] = mNew;
                    rowSum[qi] = lNew;
                }
            }

            // Write output
            for (int qi = 0; qi < qBlockSize; qi++)
            {
                int qIdx = qStart + qi;
                for (int d = 0; d < headDim; d++)
                {
                    output[new[] { batch, head, qIdx, d }] = outputAcc[qi, d];
                }
            }
        }
    }

    /// <summary>
    /// Computes the backward pass of Flash Attention using recomputation.
    /// </summary>
    /// <param name="gradOutput">Gradient of loss with respect to attention output.</param>
    /// <param name="query">Original query tensor.</param>
    /// <param name="key">Original key tensor.</param>
    /// <param name="value">Original value tensor.</param>
    /// <param name="output">Original output from forward pass.</param>
    /// <param name="config">Flash Attention configuration.</param>
    /// <returns>Gradients with respect to query, key, and value.</returns>
    public static (Tensor<T> GradQuery, Tensor<T> GradKey, Tensor<T> GradValue) Backward(
        Tensor<T> gradOutput,
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        Tensor<T> output,
        FlashAttentionConfig? config = null,
        Tensor<T>? attentionBias = null)
    {
        config ??= FlashAttentionConfig.Default;

        bool is4D = query.Shape.Length == 4;

        return is4D
            ? Backward4D(gradOutput, query, key, value, output, config, attentionBias)
            : Backward3D(gradOutput, query, key, value, output, config, attentionBias);
    }

    /// <summary>
    /// Backward pass for 3D tensors using recomputation strategy.
    /// </summary>
    private static (Tensor<T> GradQuery, Tensor<T> GradKey, Tensor<T> GradValue) Backward3D(
        Tensor<T> gradOutput,
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        Tensor<T> output,
        FlashAttentionConfig config,
        Tensor<T>? attentionBias)
    {
        int batchSize = query.Shape[0];
        int seqLenQ = query.Shape[1];
        int seqLenKV = key.Shape[1];
        int headDim = query.Shape[2];

        var gradQuery = new Tensor<T>(query.Shape);
        var gradKey = new Tensor<T>(key.Shape);
        var gradValue = new Tensor<T>(value.Shape);

        T scale = config.ScaleFactor.HasValue
            ? NumOps.FromDouble(config.ScaleFactor.Value)
            : NumOps.FromDouble(1.0 / Math.Sqrt(headDim));

        // Process each batch
        for (int b = 0; b < batchSize; b++)
        {
            BackwardCore3D(gradOutput, query, key, value, output,
                gradQuery, gradKey, gradValue,
                b, seqLenQ, seqLenKV, headDim, scale, config, attentionBias);
        }

        return (gradQuery, gradKey, gradValue);
    }

    /// <summary>
    /// Core backward computation with recomputation of attention weights.
    /// </summary>
    private static void BackwardCore3D(
        Tensor<T> gradOutput,
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        Tensor<T> output,
        Tensor<T> gradQuery,
        Tensor<T> gradKey,
        Tensor<T> gradValue,
        int batch,
        int seqLenQ,
        int seqLenKV,
        int headDim,
        T scale,
        FlashAttentionConfig config,
        Tensor<T>? attentionBias = null)
    {
        T negInf = NumOps.FromDouble(double.NegativeInfinity);

        // Recompute attention weights and compute gradients
        // This is the memory-efficient approach from Flash Attention 2

        // First, compute D = rowsum(dO * O) for each row
        var D = new T[seqLenQ];
        for (int i = 0; i < seqLenQ; i++)
        {
            T sum = NumOps.Zero;
            for (int d = 0; d < headDim; d++)
            {
                T dO = gradOutput[new[] { batch, i, d }];
                T O = output[new[] { batch, i, d }];
                sum = NumOps.Add(sum, NumOps.Multiply(dO, O));
            }
            D[i] = sum;
        }

        // Recompute attention and gradients row by row
        for (int i = 0; i < seqLenQ; i++)
        {
            // Compute attention scores for row i
            var scores = new T[seqLenKV];
            T maxScore = negInf;

            for (int j = 0; j < seqLenKV; j++)
            {
                if (config.UseCausalMask && j > i)
                {
                    scores[j] = negInf;
                    continue;
                }

                T dot = NumOps.Zero;
                for (int d = 0; d < headDim; d++)
                {
                    T qVal = query[new[] { batch, i, d }];
                    T kVal = key[new[] { batch, j, d }];
                    dot = NumOps.Add(dot, NumOps.Multiply(qVal, kVal));
                }
                scores[j] = NumOps.Multiply(dot, scale);

                // Add attention bias during recomputation (must match forward pass)
                if (attentionBias != null)
                {
                    T bias = attentionBias.Rank switch
                    {
                        3 => attentionBias[new[] { batch, i, j }],
                        2 => attentionBias[new[] { i, j }],
                        _ => NumOps.Zero
                    };
                    scores[j] = NumOps.Add(scores[j], bias);
                }

                if (NumOps.GreaterThan(scores[j], maxScore))
                {
                    maxScore = scores[j];
                }
            }

            // Compute softmax
            T sumExp = NumOps.Zero;
            var attnWeights = new T[seqLenKV];
            for (int j = 0; j < seqLenKV; j++)
            {
                attnWeights[j] = NumOps.Exp(NumOps.Subtract(scores[j], maxScore));
                sumExp = NumOps.Add(sumExp, attnWeights[j]);
            }
            for (int j = 0; j < seqLenKV; j++)
            {
                attnWeights[j] = NumericalStabilityHelper.SafeDiv(attnWeights[j], sumExp);
            }

            // Compute gradient of attention weights: dP = dO @ V^T
            var gradAttn = new T[seqLenKV];
            for (int j = 0; j < seqLenKV; j++)
            {
                T sum = NumOps.Zero;
                for (int d = 0; d < headDim; d++)
                {
                    T dO = gradOutput[new[] { batch, i, d }];
                    T vVal = value[new[] { batch, j, d }];
                    sum = NumOps.Add(sum, NumOps.Multiply(dO, vVal));
                }
                gradAttn[j] = sum;
            }

            // Compute gradient of scores: dS = P * (dP - D[i])
            var gradScores = new T[seqLenKV];
            for (int j = 0; j < seqLenKV; j++)
            {
                T diff = NumOps.Subtract(gradAttn[j], D[i]);
                gradScores[j] = NumOps.Multiply(attnWeights[j], diff);
            }

            // Update gradients
            // dQ[i] += scale * dS @ K
            for (int d = 0; d < headDim; d++)
            {
                T sum = NumOps.Zero;
                for (int j = 0; j < seqLenKV; j++)
                {
                    T kVal = key[new[] { batch, j, d }];
                    sum = NumOps.Add(sum, NumOps.Multiply(gradScores[j], kVal));
                }
                T current = gradQuery[new[] { batch, i, d }];
                gradQuery[new[] { batch, i, d }] = NumOps.Add(current, NumOps.Multiply(scale, sum));
            }

            // dK[j] += scale * dS[j] * Q[i] and dV[j] += P[j] * dO[i]
            for (int j = 0; j < seqLenKV; j++)
            {
                T scaledGradScore = NumOps.Multiply(scale, gradScores[j]);

                for (int d = 0; d < headDim; d++)
                {
                    // dK
                    T qVal = query[new[] { batch, i, d }];
                    T currentK = gradKey[new[] { batch, j, d }];
                    gradKey[new[] { batch, j, d }] = NumOps.Add(currentK, NumOps.Multiply(scaledGradScore, qVal));

                    // dV
                    T dO = gradOutput[new[] { batch, i, d }];
                    T currentV = gradValue[new[] { batch, j, d }];
                    gradValue[new[] { batch, j, d }] = NumOps.Add(currentV, NumOps.Multiply(attnWeights[j], dO));
                }
            }
        }
    }

    /// <summary>
    /// Backward pass for 4D tensors.
    /// </summary>
    private static (Tensor<T> GradQuery, Tensor<T> GradKey, Tensor<T> GradValue) Backward4D(
        Tensor<T> gradOutput,
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        Tensor<T> output,
        FlashAttentionConfig config,
        Tensor<T>? attentionBias)
    {
        int batchSize = query.Shape[0];
        int numHeads = query.Shape[1];
        int seqLenQ = query.Shape[2];
        int seqLenKV = key.Shape[2];
        int headDim = query.Shape[3];

        var gradQuery = new Tensor<T>(query.Shape);
        var gradKey = new Tensor<T>(key.Shape);
        var gradValue = new Tensor<T>(value.Shape);

        T scale = config.ScaleFactor.HasValue
            ? NumOps.FromDouble(config.ScaleFactor.Value)
            : NumOps.FromDouble(1.0 / Math.Sqrt(headDim));

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                BackwardCore4D(gradOutput, query, key, value, output,
                    gradQuery, gradKey, gradValue,
                    b, h, seqLenQ, seqLenKV, headDim, scale, config, attentionBias);
            }
        }

        return (gradQuery, gradKey, gradValue);
    }

    /// <summary>
    /// Core backward computation for 4D tensors.
    /// </summary>
    private static void BackwardCore4D(
        Tensor<T> gradOutput,
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        Tensor<T> output,
        Tensor<T> gradQuery,
        Tensor<T> gradKey,
        Tensor<T> gradValue,
        int batch,
        int head,
        int seqLenQ,
        int seqLenKV,
        int headDim,
        T scale,
        FlashAttentionConfig config,
        Tensor<T>? attentionBias = null)
    {
        T negInf = NumOps.FromDouble(double.NegativeInfinity);

        // Compute D = rowsum(dO * O)
        var D = new T[seqLenQ];
        for (int i = 0; i < seqLenQ; i++)
        {
            T sum = NumOps.Zero;
            for (int d = 0; d < headDim; d++)
            {
                T dO = gradOutput[new[] { batch, head, i, d }];
                T O = output[new[] { batch, head, i, d }];
                sum = NumOps.Add(sum, NumOps.Multiply(dO, O));
            }
            D[i] = sum;
        }

        for (int i = 0; i < seqLenQ; i++)
        {
            var scores = new T[seqLenKV];
            T maxScore = negInf;

            for (int j = 0; j < seqLenKV; j++)
            {
                if (config.UseCausalMask && j > i)
                {
                    scores[j] = negInf;
                    continue;
                }

                T dot = NumOps.Zero;
                for (int d = 0; d < headDim; d++)
                {
                    T qVal = query[new[] { batch, head, i, d }];
                    T kVal = key[new[] { batch, head, j, d }];
                    dot = NumOps.Add(dot, NumOps.Multiply(qVal, kVal));
                }
                scores[j] = NumOps.Multiply(dot, scale);

                // Add attention bias during recomputation (must match forward pass)
                if (attentionBias != null)
                {
                    T bias = attentionBias.Rank switch
                    {
                        4 => attentionBias[new[] { batch, head, i, j }],
                        3 => attentionBias[new[] { head, i, j }],
                        _ => NumOps.Zero
                    };
                    scores[j] = NumOps.Add(scores[j], bias);
                }

                if (NumOps.GreaterThan(scores[j], maxScore))
                {
                    maxScore = scores[j];
                }
            }

            T sumExp = NumOps.Zero;
            var attnWeights = new T[seqLenKV];
            for (int j = 0; j < seqLenKV; j++)
            {
                attnWeights[j] = NumOps.Exp(NumOps.Subtract(scores[j], maxScore));
                sumExp = NumOps.Add(sumExp, attnWeights[j]);
            }
            for (int j = 0; j < seqLenKV; j++)
            {
                attnWeights[j] = NumericalStabilityHelper.SafeDiv(attnWeights[j], sumExp);
            }

            var gradAttn = new T[seqLenKV];
            for (int j = 0; j < seqLenKV; j++)
            {
                T sum = NumOps.Zero;
                for (int d = 0; d < headDim; d++)
                {
                    T dO = gradOutput[new[] { batch, head, i, d }];
                    T vVal = value[new[] { batch, head, j, d }];
                    sum = NumOps.Add(sum, NumOps.Multiply(dO, vVal));
                }
                gradAttn[j] = sum;
            }

            var gradScores = new T[seqLenKV];
            for (int j = 0; j < seqLenKV; j++)
            {
                T diff = NumOps.Subtract(gradAttn[j], D[i]);
                gradScores[j] = NumOps.Multiply(attnWeights[j], diff);
            }

            for (int d = 0; d < headDim; d++)
            {
                T sum = NumOps.Zero;
                for (int j = 0; j < seqLenKV; j++)
                {
                    T kVal = key[new[] { batch, head, j, d }];
                    sum = NumOps.Add(sum, NumOps.Multiply(gradScores[j], kVal));
                }
                T current = gradQuery[new[] { batch, head, i, d }];
                gradQuery[new[] { batch, head, i, d }] = NumOps.Add(current, NumOps.Multiply(scale, sum));
            }

            for (int j = 0; j < seqLenKV; j++)
            {
                T scaledGradScore = NumOps.Multiply(scale, gradScores[j]);

                for (int d = 0; d < headDim; d++)
                {
                    T qVal = query[new[] { batch, head, i, d }];
                    T currentK = gradKey[new[] { batch, head, j, d }];
                    gradKey[new[] { batch, head, j, d }] = NumOps.Add(currentK, NumOps.Multiply(scaledGradScore, qVal));

                    T dO = gradOutput[new[] { batch, head, i, d }];
                    T currentV = gradValue[new[] { batch, head, j, d }];
                    gradValue[new[] { batch, head, j, d }] = NumOps.Add(currentV, NumOps.Multiply(attnWeights[j], dO));
                }
            }
        }
    }

    /// <summary>
    /// Validates input tensor shapes.
    /// </summary>
    private static void ValidateInputs(Tensor<T> query, Tensor<T> key, Tensor<T> value)
    {
        if (query.Shape.Length != key.Shape.Length || key.Shape.Length != value.Shape.Length)
        {
            throw new ArgumentException("Query, Key, and Value must have the same number of dimensions.");
        }

        if (query.Shape.Length < 3 || query.Shape.Length > 4)
        {
            throw new ArgumentException("Query, Key, and Value must be 3D [batch, seq, dim] or 4D [batch, heads, seq, dim].");
        }

        // Batch size must match
        if (query.Shape[0] != key.Shape[0] || key.Shape[0] != value.Shape[0])
        {
            throw new ArgumentException("Batch sizes must match across Query, Key, and Value.");
        }

        // For 4D tensors, heads must match
        if (query.Shape.Length == 4)
        {
            if (query.Shape[1] != key.Shape[1] || key.Shape[1] != value.Shape[1])
            {
                throw new ArgumentException("Number of heads must match across Query, Key, and Value.");
            }

            // Head dimension must match
            if (query.Shape[3] != key.Shape[3])
            {
                throw new ArgumentException("Head dimension must match between Query and Key.");
            }

            // Key and Value sequence lengths must match
            if (key.Shape[2] != value.Shape[2])
            {
                throw new ArgumentException("Key and Value sequence lengths must match.");
            }
        }
        else
        {
            // For 3D tensors
            if (query.Shape[2] != key.Shape[2])
            {
                throw new ArgumentException("Feature dimension must match between Query and Key.");
            }

            if (key.Shape[1] != value.Shape[1])
            {
                throw new ArgumentException("Key and Value sequence lengths must match.");
            }
        }
    }
}
