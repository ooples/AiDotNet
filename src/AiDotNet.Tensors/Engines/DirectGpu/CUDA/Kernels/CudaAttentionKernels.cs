// Copyright (c) AiDotNet. All rights reserved.
// CUDA kernels for attention operations - FlashAttention, GroupedQueryAttention, and standard attention.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    /// <summary>
    /// CUDA kernels for attention mechanisms used in transformer architectures.
    /// Includes ScaledDotProductAttention, FlashAttention v2, and GroupedQueryAttention.
    /// </summary>
    internal static class CudaAttentionKernels
    {
        public static string GetSource()
        {
            return @"
#include <math.h>

// Block size for tiling in FlashAttention
#define FLASH_BLOCK_SIZE 64
#define MAX_HEAD_DIM 128

// ===========================================================================
// SCALED DOT-PRODUCT ATTENTION
// ===========================================================================

extern ""C"" __global__ void scaled_dot_product_attention(
    const float* query,      // [batch * heads * seqQ * headDim]
    const float* key,        // [batch * heads * seqK * headDim]
    const float* value,      // [batch * heads * seqK * headDim]
    float* output,           // [batch * heads * seqQ * headDim]
    float* attentionWeights, // [batch * heads * seqQ * seqK] (optional)
    int batch,
    int numHeads,
    int seqQ,
    int seqK,
    int headDim,
    float scale,
    int isCausal,
    int storeWeights)
{
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    int bh = blockIdx.y * blockDim.y + threadIdx.y;

    if (bh >= batch * numHeads || qi >= seqQ) return;

    // Offsets
    int qOffset = bh * seqQ * headDim + qi * headDim;
    int kOffset = bh * seqK * headDim;
    int vOffset = bh * seqK * headDim;
    int oOffset = bh * seqQ * headDim + qi * headDim;
    int wOffset = bh * seqQ * seqK + qi * seqK;

    // Compute attention scores and find max for numerical stability
    float maxScore = -INFINITY;
    float scores[1024];

    for (int ki = 0; ki < seqK; ki++) {
        if (isCausal && ki > qi) {
            scores[ki] = -INFINITY;
            continue;
        }

        float score = 0.0f;
        for (int d = 0; d < headDim; d++) {
            score += query[qOffset + d] * key[kOffset + ki * headDim + d];
        }
        score *= scale;
        scores[ki] = score;
        maxScore = fmaxf(maxScore, score);
    }

    // Compute softmax
    float sumExp = 0.0f;
    for (int ki = 0; ki < seqK; ki++) {
        float expScore = expf(scores[ki] - maxScore);
        scores[ki] = expScore;
        sumExp += expScore;
    }

    // Normalize and compute output
    for (int d = 0; d < headDim; d++) {
        float val = 0.0f;
        for (int ki = 0; ki < seqK; ki++) {
            float weight = scores[ki] / sumExp;
            val += weight * value[vOffset + ki * headDim + d];
        }
        output[oOffset + d] = val;
    }

    // Store attention weights if requested
    if (storeWeights) {
        for (int ki = 0; ki < seqK; ki++) {
            attentionWeights[wOffset + ki] = scores[ki] / sumExp;
        }
    }
}

// ===========================================================================
// FLASH ATTENTION V2
// Memory-efficient attention using online softmax and tiling
// ===========================================================================

extern ""C"" __global__ void flash_attention_v2(
    const float* query,      // [batch * heads * seqQ * headDim]
    const float* key,        // [batch * heads * seqK * headDim]
    const float* value,      // [batch * heads * seqK * headDim]
    float* output,           // [batch * heads * seqQ * headDim]
    float* softmaxStats,     // [batch * heads * seqQ] (log-sum-exp)
    int batch,
    int numHeads,
    int seqQ,
    int seqK,
    int headDim,
    float scale,
    int isCausal)
{
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    int bh = blockIdx.y * blockDim.y + threadIdx.y;

    if (bh >= batch * numHeads || qi >= seqQ) return;

    // Offsets
    int qOffset = bh * seqQ * headDim + qi * headDim;
    int kOffset = bh * seqK * headDim;
    int vOffset = bh * seqK * headDim;
    int oOffset = bh * seqQ * headDim + qi * headDim;
    int sOffset = bh * seqQ + qi;

    // Initialize accumulators for online softmax
    float rowMax = -INFINITY;
    float rowSum = 0.0f;

    // Initialize output accumulator
    float outAcc[MAX_HEAD_DIM];
    for (int d = 0; d < headDim && d < MAX_HEAD_DIM; d++) {
        outAcc[d] = 0.0f;
    }

    // Process key-value pairs in blocks for memory efficiency
    for (int kvBlockStart = 0; kvBlockStart < seqK; kvBlockStart += FLASH_BLOCK_SIZE) {
        int kvBlockEnd = min(kvBlockStart + FLASH_BLOCK_SIZE, seqK);

        // Skip block if causal and all keys are after query
        if (isCausal && kvBlockStart > qi) continue;

        // Find max in this block
        float blockMax = -INFINITY;
        for (int ki = kvBlockStart; ki < kvBlockEnd; ki++) {
            if (isCausal && ki > qi) continue;

            float score = 0.0f;
            for (int d = 0; d < headDim; d++) {
                score += query[qOffset + d] * key[kOffset + ki * headDim + d];
            }
            score *= scale;
            blockMax = fmaxf(blockMax, score);
        }

        // New global max and rescale factor
        float newMax = fmaxf(rowMax, blockMax);
        float rescale = expf(rowMax - newMax);
        float newSum = rowSum * rescale;

        // Rescale output accumulator
        for (int d = 0; d < headDim && d < MAX_HEAD_DIM; d++) {
            outAcc[d] *= rescale;
        }

        // Add contributions from this block
        for (int ki = kvBlockStart; ki < kvBlockEnd; ki++) {
            if (isCausal && ki > qi) continue;

            float score = 0.0f;
            for (int d = 0; d < headDim; d++) {
                score += query[qOffset + d] * key[kOffset + ki * headDim + d];
            }
            score *= scale;

            float expScore = expf(score - newMax);
            newSum += expScore;

            // Accumulate weighted value
            for (int d = 0; d < headDim && d < MAX_HEAD_DIM; d++) {
                outAcc[d] += expScore * value[vOffset + ki * headDim + d];
            }
        }

        rowMax = newMax;
        rowSum = newSum;
    }

    // Final normalization and write output
    float invSum = 1.0f / rowSum;
    for (int d = 0; d < headDim && d < MAX_HEAD_DIM; d++) {
        output[oOffset + d] = outAcc[d] * invSum;
    }

    // Store log-sum-exp for backward pass
    softmaxStats[sOffset] = rowMax + logf(rowSum);
}

// ===========================================================================
// FLASH ATTENTION BACKWARD
// Recomputes attention weights during backward pass
// ===========================================================================

extern ""C"" __global__ void flash_attention_backward(
    const float* gradOutput,   // [batch * heads * seqQ * headDim]
    const float* query,        // [batch * heads * seqQ * headDim]
    const float* key,          // [batch * heads * seqK * headDim]
    const float* value,        // [batch * heads * seqK * headDim]
    const float* output,       // [batch * heads * seqQ * headDim]
    const float* softmaxStats, // [batch * heads * seqQ]
    float* gradQuery,          // [batch * heads * seqQ * headDim]
    float* gradKey,            // [batch * heads * seqK * headDim]
    float* gradValue,          // [batch * heads * seqK * headDim]
    int batch,
    int numHeads,
    int seqQ,
    int seqK,
    int headDim,
    float scale,
    int isCausal)
{
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    int bh = blockIdx.y * blockDim.y + threadIdx.y;

    if (bh >= batch * numHeads || qi >= seqQ) return;

    // Offsets
    int qOffset = bh * seqQ * headDim + qi * headDim;
    int kOffset = bh * seqK * headDim;
    int vOffset = bh * seqK * headDim;
    int gOffset = bh * seqQ * headDim + qi * headDim;
    int sOffset = bh * seqQ + qi;

    float logsumexp = softmaxStats[sOffset];

    // Compute dO @ O (for softmax backward)
    float doO = 0.0f;
    for (int d = 0; d < headDim; d++) {
        doO += gradOutput[gOffset + d] * output[qOffset + d];
    }

    // Process each key position
    for (int ki = 0; ki < seqK; ki++) {
        if (isCausal && ki > qi) continue;

        // Recompute attention score
        float score = 0.0f;
        for (int d = 0; d < headDim; d++) {
            score += query[qOffset + d] * key[kOffset + ki * headDim + d];
        }
        score *= scale;

        // Recompute attention weight
        float attnWeight = expf(score - logsumexp);

        // Gradient w.r.t. V: attnWeight * gradOutput
        for (int d = 0; d < headDim; d++) {
            atomicAdd(&gradValue[vOffset + ki * headDim + d],
                      attnWeight * gradOutput[gOffset + d]);
        }

        // Compute dO @ v
        float doV = 0.0f;
        for (int d = 0; d < headDim; d++) {
            doV += gradOutput[gOffset + d] * value[vOffset + ki * headDim + d];
        }

        // dS = attnWeight * (doV - doO) * scale
        float dS = attnWeight * (doV - doO) * scale;

        // Gradient w.r.t. Q: dS * K
        for (int d = 0; d < headDim; d++) {
            gradQuery[qOffset + d] += dS * key[kOffset + ki * headDim + d];
        }

        // Gradient w.r.t. K: dS * Q
        for (int d = 0; d < headDim; d++) {
            atomicAdd(&gradKey[kOffset + ki * headDim + d],
                      dS * query[qOffset + d]);
        }
    }
}

// ===========================================================================
// GROUPED QUERY ATTENTION (GQA)
// Multiple query heads share the same key-value head
// ===========================================================================

extern ""C"" __global__ void grouped_query_attention(
    const float* query,          // [batch * numQHeads * seqQ * headDim]
    const float* key,            // [batch * numKVHeads * seqK * headDim]
    const float* value,          // [batch * numKVHeads * seqK * headDim]
    float* output,               // [batch * numQHeads * seqQ * headDim]
    float* attentionWeights,     // [batch * numQHeads * seqQ * seqK] (optional)
    int batch,
    int numQHeads,
    int numKVHeads,
    int queriesPerKV,            // numQHeads / numKVHeads
    int seqQ,
    int seqK,
    int headDim,
    float scale,
    int isCausal,
    int storeWeights)
{
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    int bqh = blockIdx.y * blockDim.y + threadIdx.y;

    if (qi >= seqQ || bqh >= batch * numQHeads) return;

    int b = bqh / numQHeads;
    int qh = bqh % numQHeads;
    int kvh = qh / queriesPerKV;  // Which KV head this query uses

    // Offsets
    int qOffset = bqh * seqQ * headDim + qi * headDim;
    int kOffset = (b * numKVHeads + kvh) * seqK * headDim;
    int vOffset = (b * numKVHeads + kvh) * seqK * headDim;
    int oOffset = bqh * seqQ * headDim + qi * headDim;
    int wOffset = bqh * seqQ * seqK + qi * seqK;

    // Compute attention scores
    float maxScore = -INFINITY;
    float scores[1024];

    for (int ki = 0; ki < seqK; ki++) {
        if (isCausal && ki > qi) {
            scores[ki] = -INFINITY;
            continue;
        }

        float score = 0.0f;
        for (int d = 0; d < headDim; d++) {
            score += query[qOffset + d] * key[kOffset + ki * headDim + d];
        }
        score *= scale;
        scores[ki] = score;
        maxScore = fmaxf(maxScore, score);
    }

    // Softmax normalization
    float sumExp = 0.0f;
    for (int ki = 0; ki < seqK; ki++) {
        float expScore = expf(scores[ki] - maxScore);
        scores[ki] = expScore;
        sumExp += expScore;
    }

    // Compute output and optionally store weights
    for (int d = 0; d < headDim; d++) {
        float val = 0.0f;
        for (int ki = 0; ki < seqK; ki++) {
            float weight = scores[ki] / sumExp;
            val += weight * value[vOffset + ki * headDim + d];
        }
        output[oOffset + d] = val;
    }

    if (storeWeights) {
        for (int ki = 0; ki < seqK; ki++) {
            attentionWeights[wOffset + ki] = scores[ki] / sumExp;
        }
    }
}

// ===========================================================================
// GQA BACKWARD
// ===========================================================================

extern ""C"" __global__ void grouped_query_attention_backward(
    const float* gradOutput,       // [batch * numQHeads * seqQ * headDim]
    const float* query,            // [batch * numQHeads * seqQ * headDim]
    const float* key,              // [batch * numKVHeads * seqK * headDim]
    const float* value,            // [batch * numKVHeads * seqK * headDim]
    const float* attentionWeights, // [batch * numQHeads * seqQ * seqK]
    float* gradQuery,              // [batch * numQHeads * seqQ * headDim]
    float* gradKey,                // [batch * numKVHeads * seqK * headDim]
    float* gradValue,              // [batch * numKVHeads * seqK * headDim]
    int batch,
    int numQHeads,
    int numKVHeads,
    int queriesPerKV,
    int seqQ,
    int seqK,
    int headDim,
    float scale)
{
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    int bqh = blockIdx.y * blockDim.y + threadIdx.y;

    if (qi >= seqQ || bqh >= batch * numQHeads) return;

    int b = bqh / numQHeads;
    int qh = bqh % numQHeads;
    int kvh = qh / queriesPerKV;

    // Offsets
    int qOffset = bqh * seqQ * headDim + qi * headDim;
    int kOffset = (b * numKVHeads + kvh) * seqK * headDim;
    int vOffset = (b * numKVHeads + kvh) * seqK * headDim;
    int gOffset = bqh * seqQ * headDim + qi * headDim;
    int wOffset = bqh * seqQ * seqK + qi * seqK;

    // Compute gradients w.r.t. attention weights: gradOutput @ V^T
    float gradWeights[1024];
    for (int ki = 0; ki < seqK; ki++) {
        float sum = 0.0f;
        for (int d = 0; d < headDim; d++) {
            sum += gradOutput[gOffset + d] * value[vOffset + ki * headDim + d];
        }
        gradWeights[ki] = sum;
    }

    // Softmax backward: gradScores = weights * (gradWeights - dot(weights, gradWeights))
    float dotProduct = 0.0f;
    for (int ki = 0; ki < seqK; ki++) {
        dotProduct += attentionWeights[wOffset + ki] * gradWeights[ki];
    }

    // Compute gradients
    for (int ki = 0; ki < seqK; ki++) {
        float weight = attentionWeights[wOffset + ki];
        float gradScore = weight * (gradWeights[ki] - dotProduct) * scale;

        // Gradient w.r.t. V (accumulated across query heads using atomics)
        for (int d = 0; d < headDim; d++) {
            atomicAdd(&gradValue[vOffset + ki * headDim + d],
                      weight * gradOutput[gOffset + d]);
        }

        // Gradient w.r.t. Q
        for (int d = 0; d < headDim; d++) {
            gradQuery[qOffset + d] += gradScore * key[kOffset + ki * headDim + d];
        }

        // Gradient w.r.t. K (accumulated across query heads using atomics)
        for (int d = 0; d < headDim; d++) {
            atomicAdd(&gradKey[kOffset + ki * headDim + d],
                      gradScore * query[qOffset + d]);
        }
    }
}

// ===========================================================================
// FLASH ATTENTION FORWARD (Compatibility version without stats)
// ===========================================================================

extern ""C"" __global__ void flash_attention_forward(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    int batch,
    int numHeads,
    int seqLen,
    int headDim,
    float scale,
    int isCausal)
{
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    int bh = blockIdx.y * blockDim.y + threadIdx.y;

    if (bh >= batch * numHeads || qi >= seqLen) return;

    int qOffset = bh * seqLen * headDim + qi * headDim;
    int kOffset = bh * seqLen * headDim;
    int vOffset = bh * seqLen * headDim;
    int oOffset = bh * seqLen * headDim + qi * headDim;

    float rowMax = -INFINITY;
    float rowSum = 0.0f;
    float outAcc[MAX_HEAD_DIM];
    for (int d = 0; d < headDim && d < MAX_HEAD_DIM; d++) outAcc[d] = 0.0f;

    for (int ki = 0; ki < seqLen; ki++) {
        if (isCausal && ki > qi) continue;

        float score = 0.0f;
        for (int d = 0; d < headDim; d++) {
            score += query[qOffset + d] * key[kOffset + ki * headDim + d];
        }
        score *= scale;

        float newMax = fmaxf(rowMax, score);
        float rescale = expf(rowMax - newMax);
        rowSum = rowSum * rescale + expf(score - newMax);

        for (int d = 0; d < headDim && d < MAX_HEAD_DIM; d++) {
            outAcc[d] = outAcc[d] * rescale + expf(score - newMax) * value[vOffset + ki * headDim + d];
        }
        rowMax = newMax;
    }

    float invSum = 1.0f / rowSum;
    for (int d = 0; d < headDim && d < MAX_HEAD_DIM; d++) {
        output[oOffset + d] = outAcc[d] * invSum;
    }
}
";
        }

        public static string[] GetKernelNames()
        {
            return new[]
            {
                "scaled_dot_product_attention",
                "flash_attention_v2",
                "flash_attention_backward",
                "grouped_query_attention",
                "grouped_query_attention_backward",
                "flash_attention_forward"
            };
        }
    }
}
