// Copyright (c) AiDotNet. All rights reserved.
// GPU kernels for attention operations including standard attention, FlashAttention, and GQA.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    /// <summary>
    /// GPU kernels for attention mechanisms used in transformer architectures.
    /// Includes ScaledDotProductAttention, FlashAttention, and GroupedQueryAttention.
    /// </summary>
    internal static class AttentionKernels
    {
        /// <summary>
        /// Gets all attention kernel sources.
        /// </summary>
        public static string GetSource()
        {
            return @"
// ===========================================================================
// SCALED DOT-PRODUCT ATTENTION
// ===========================================================================

// Standard scaled dot-product attention
// Output: attention(Q, K, V) = softmax(Q @ K^T / scale) @ V
__kernel void scaled_dot_product_attention(
    __global const float* query,      // [batch * heads * seqQ * headDim]
    __global const float* key,        // [batch * heads * seqK * headDim]
    __global const float* value,      // [batch * heads * seqK * headDim]
    __global float* output,           // [batch * heads * seqQ * headDim]
    __global float* attentionWeights, // [batch * heads * seqQ * seqK] (optional)
    __global const int* mask,         // [seqQ * seqK] (optional, 0=masked, 1=valid)
    const int batch,
    const int numHeads,
    const int seqQ,
    const int seqK,
    const int headDim,
    const float scale,
    const int isCausal,
    const int hasMask,
    const int storeWeights)
{
    const int bh = get_global_id(1);  // batch * head index
    const int qi = get_global_id(0);  // query position

    if (bh >= batch * numHeads || qi >= seqQ) return;

    const int b = bh / numHeads;
    const int h = bh % numHeads;

    // Offsets
    const int qOffset = bh * seqQ * headDim + qi * headDim;
    const int kOffset = bh * seqK * headDim;
    const int vOffset = bh * seqK * headDim;
    const int oOffset = bh * seqQ * headDim + qi * headDim;
    const int wOffset = bh * seqQ * seqK + qi * seqK;

    // Compute attention scores and find max for numerical stability
    float maxScore = -INFINITY;
    for (int ki = 0; ki < seqK; ki++) {
        // Causal mask
        if (isCausal && ki > qi) continue;

        float score = 0.0f;
        for (int d = 0; d < headDim; d++) {
            score += query[qOffset + d] * key[kOffset + ki * headDim + d];
        }
        score *= scale;
        maxScore = fmax(maxScore, score);
    }

    // Compute softmax
    float sumExp = 0.0f;
    float scores[1024]; // Temporary storage (adjust size as needed)
    for (int ki = 0; ki < seqK; ki++) {
        if (isCausal && ki > qi) {
            scores[ki] = 0.0f;
            continue;
        }

        float score = 0.0f;
        for (int d = 0; d < headDim; d++) {
            score += query[qOffset + d] * key[kOffset + ki * headDim + d];
        }
        score *= scale;

        float expScore = exp(score - maxScore);
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

// Block size for tiling (adjust based on GPU shared memory)
#define FLASH_BLOCK_SIZE 64

__kernel void flash_attention_v2(
    __global const float* query,      // [batch * heads * seqQ * headDim]
    __global const float* key,        // [batch * heads * seqK * headDim]
    __global const float* value,      // [batch * heads * seqK * headDim]
    __global float* output,           // [batch * heads * seqQ * headDim]
    __global float* softmaxStats,     // [batch * heads * seqQ] (log-sum-exp)
    const int batch,
    const int numHeads,
    const int seqQ,
    const int seqK,
    const int headDim,
    const float scale,
    const int isCausal)
{
    const int bh = get_global_id(1);  // batch * head index
    const int qi = get_global_id(0);  // query position

    if (bh >= batch * numHeads || qi >= seqQ) return;

    // Offsets
    const int qOffset = bh * seqQ * headDim + qi * headDim;
    const int kOffset = bh * seqK * headDim;
    const int vOffset = bh * seqK * headDim;
    const int oOffset = bh * seqQ * headDim + qi * headDim;
    const int sOffset = bh * seqQ + qi;

    // Initialize accumulators for online softmax
    float rowMax = -INFINITY;
    float rowSum = 0.0f;

    // Initialize output to zero
    float outAcc[128]; // Max headDim supported
    for (int d = 0; d < headDim; d++) {
        outAcc[d] = 0.0f;
    }

    // Process key-value pairs in blocks
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
            blockMax = fmax(blockMax, score);
        }

        // New global max
        float newMax = fmax(rowMax, blockMax);

        // Rescale factor for previous accumulator
        float rescale = exp(rowMax - newMax);
        float newSum = rowSum * rescale;

        // Rescale output accumulator
        for (int d = 0; d < headDim; d++) {
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

            float expScore = exp(score - newMax);
            newSum += expScore;

            // Accumulate weighted value
            for (int d = 0; d < headDim; d++) {
                outAcc[d] += expScore * value[vOffset + ki * headDim + d];
            }
        }

        rowMax = newMax;
        rowSum = newSum;
    }

    // Final normalization and write output
    float invSum = 1.0f / rowSum;
    for (int d = 0; d < headDim; d++) {
        output[oOffset + d] = outAcc[d] * invSum;
    }

    // Store log-sum-exp for backward pass
    softmaxStats[sOffset] = rowMax + log(rowSum);
}

// FlashAttention backward pass with recomputation
__kernel void flash_attention_backward(
    __global const float* gradOutput,   // [batch * heads * seqQ * headDim]
    __global const float* query,        // [batch * heads * seqQ * headDim]
    __global const float* key,          // [batch * heads * seqK * headDim]
    __global const float* value,        // [batch * heads * seqK * headDim]
    __global const float* output,       // [batch * heads * seqQ * headDim]
    __global const float* softmaxStats, // [batch * heads * seqQ]
    __global float* gradQuery,          // [batch * heads * seqQ * headDim]
    __global float* gradKey,            // [batch * heads * seqK * headDim]
    __global float* gradValue,          // [batch * heads * seqK * headDim]
    const int batch,
    const int numHeads,
    const int seqQ,
    const int seqK,
    const int headDim,
    const float scale,
    const int isCausal)
{
    const int bh = get_global_id(1);
    const int qi = get_global_id(0);

    if (bh >= batch * numHeads || qi >= seqQ) return;

    // Offsets
    const int qOffset = bh * seqQ * headDim + qi * headDim;
    const int kOffset = bh * seqK * headDim;
    const int vOffset = bh * seqK * headDim;
    const int gOffset = bh * seqQ * headDim + qi * headDim;
    const int sOffset = bh * seqQ + qi;

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
        float attnWeight = exp(score - logsumexp);

        // Gradient w.r.t. V: attnWeight * gradOutput
        for (int d = 0; d < headDim; d++) {
            atomic_add(&gradValue[vOffset + ki * headDim + d],
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
            atomic_add(&gradKey[kOffset + ki * headDim + d],
                       dS * query[qOffset + d]);
        }
    }
}

// ===========================================================================
// GROUPED QUERY ATTENTION (GQA)
// Multiple query heads share the same key-value head
// ===========================================================================

__kernel void grouped_query_attention(
    __global const float* query,          // [batch * numQHeads * seqQ * headDim]
    __global const float* key,            // [batch * numKVHeads * seqK * headDim]
    __global const float* value,          // [batch * numKVHeads * seqK * headDim]
    __global float* output,               // [batch * numQHeads * seqQ * headDim]
    __global float* attentionWeights,     // [batch * numQHeads * seqQ * seqK] (optional)
    const int batch,
    const int numQHeads,
    const int numKVHeads,
    const int queriesPerKV,               // numQHeads / numKVHeads
    const int seqQ,
    const int seqK,
    const int headDim,
    const float scale,
    const int isCausal,
    const int storeWeights)
{
    const int d = get_global_id(0);       // head dimension
    const int qi = get_global_id(1);      // query position
    const int bqh = get_global_id(2);     // batch * query head

    if (d >= headDim || qi >= seqQ || bqh >= batch * numQHeads) return;

    const int b = bqh / numQHeads;
    const int qh = bqh % numQHeads;
    const int kvh = qh / queriesPerKV;    // Which KV head this query uses

    // Offsets
    const int qOffset = bqh * seqQ * headDim + qi * headDim;
    const int kOffset = (b * numKVHeads + kvh) * seqK * headDim;
    const int vOffset = (b * numKVHeads + kvh) * seqK * headDim;
    const int oOffset = bqh * seqQ * headDim + qi * headDim;
    const int wOffset = bqh * seqQ * seqK + qi * seqK;

    // Compute attention scores and softmax (only thread d=0 does this)
    if (d == 0) {
        float maxScore = -INFINITY;
        float scores[1024];

        // Compute scores and find max
        for (int ki = 0; ki < seqK; ki++) {
            if (isCausal && ki > qi) {
                scores[ki] = -INFINITY;
                continue;
            }

            float score = 0.0f;
            for (int dd = 0; dd < headDim; dd++) {
                score += query[qOffset + dd] * key[kOffset + ki * headDim + dd];
            }
            score *= scale;
            scores[ki] = score;
            maxScore = fmax(maxScore, score);
        }

        // Softmax
        float sumExp = 0.0f;
        for (int ki = 0; ki < seqK; ki++) {
            float expScore = exp(scores[ki] - maxScore);
            scores[ki] = expScore;
            sumExp += expScore;
        }

        // Compute output and optionally store weights
        for (int dd = 0; dd < headDim; dd++) {
            float val = 0.0f;
            for (int ki = 0; ki < seqK; ki++) {
                float weight = scores[ki] / sumExp;
                val += weight * value[vOffset + ki * headDim + dd];
                if (storeWeights && dd == 0) {
                    attentionWeights[wOffset + ki] = weight;
                }
            }
            output[oOffset + dd] = val;
        }
    }
}

__kernel void grouped_query_attention_backward(
    __global const float* gradOutput,     // [batch * numQHeads * seqQ * headDim]
    __global const float* query,          // [batch * numQHeads * seqQ * headDim]
    __global const float* key,            // [batch * numKVHeads * seqK * headDim]
    __global const float* value,          // [batch * numKVHeads * seqK * headDim]
    __global const float* attentionWeights, // [batch * numQHeads * seqQ * seqK]
    __global float* gradQuery,            // [batch * numQHeads * seqQ * headDim]
    __global float* gradKey,              // [batch * numKVHeads * seqK * headDim]
    __global float* gradValue,            // [batch * numKVHeads * seqK * headDim]
    const int batch,
    const int numQHeads,
    const int numKVHeads,
    const int queriesPerKV,
    const int seqQ,
    const int seqK,
    const int headDim,
    const float scale)
{
    const int d = get_global_id(0);
    const int qi = get_global_id(1);
    const int bqh = get_global_id(2);

    if (d >= headDim || qi >= seqQ || bqh >= batch * numQHeads) return;

    const int b = bqh / numQHeads;
    const int qh = bqh % numQHeads;
    const int kvh = qh / queriesPerKV;

    // Offsets
    const int qOffset = bqh * seqQ * headDim + qi * headDim;
    const int kOffset = (b * numKVHeads + kvh) * seqK * headDim;
    const int vOffset = (b * numKVHeads + kvh) * seqK * headDim;
    const int gOffset = bqh * seqQ * headDim + qi * headDim;
    const int wOffset = bqh * seqQ * seqK + qi * seqK;

    // Only thread d=0 computes gradients for this position
    if (d == 0) {
        // Compute gradients w.r.t. attention weights: gradOutput @ V^T
        float gradWeights[1024];
        for (int ki = 0; ki < seqK; ki++) {
            float sum = 0.0f;
            for (int dd = 0; dd < headDim; dd++) {
                sum += gradOutput[gOffset + dd] * value[vOffset + ki * headDim + dd];
            }
            gradWeights[ki] = sum;
        }

        // Softmax backward: gradScores = weights * (gradWeights - dot(weights, gradWeights))
        float dotProduct = 0.0f;
        for (int ki = 0; ki < seqK; ki++) {
            dotProduct += attentionWeights[wOffset + ki] * gradWeights[ki];
        }

        // Gradient w.r.t. V and compute gradScores
        for (int ki = 0; ki < seqK; ki++) {
            float weight = attentionWeights[wOffset + ki];
            float gradScore = weight * (gradWeights[ki] - dotProduct) * scale;

            // Gradient w.r.t. V (accumulated across query heads)
            for (int dd = 0; dd < headDim; dd++) {
                atomic_add(&gradValue[vOffset + ki * headDim + dd],
                           weight * gradOutput[gOffset + dd]);
            }

            // Gradient w.r.t. Q
            for (int dd = 0; dd < headDim; dd++) {
                gradQuery[qOffset + dd] += gradScore * key[kOffset + ki * headDim + dd];
            }

            // Gradient w.r.t. K (accumulated across query heads)
            for (int dd = 0; dd < headDim; dd++) {
                atomic_add(&gradKey[kOffset + ki * headDim + dd],
                           gradScore * query[qOffset + dd]);
            }
        }
    }
}

// Atomic add for float (not natively supported in all OpenCL versions)
inline void atomic_add_float(__global float* addr, float val) {
    union {
        unsigned int u32;
        float f32;
    } next, expected, current;
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atomic_cmpxchg((volatile __global unsigned int*)addr,
                                      expected.u32, next.u32);
    } while (current.u32 != expected.u32);
}

// ===========================================================================
// FLASH ATTENTION FORWARD (Original tiled version for compatibility)
// ===========================================================================

__kernel void flash_attention_forward(
    __global const float* query,
    __global const float* key,
    __global const float* value,
    __global float* output,
    __global const int* mask,
    const int batch,
    const int numHeads,
    const int seqLen,
    const int headDim,
    const float scale,
    const int isCausal,
    const int hasMask)
{
    const int bh = get_global_id(1);
    const int qi = get_global_id(0);

    if (bh >= batch * numHeads || qi >= seqLen) return;

    // Same as flash_attention_v2 but without storing stats
    const int qOffset = bh * seqLen * headDim + qi * headDim;
    const int kOffset = bh * seqLen * headDim;
    const int vOffset = bh * seqLen * headDim;
    const int oOffset = bh * seqLen * headDim + qi * headDim;

    float rowMax = -INFINITY;
    float rowSum = 0.0f;
    float outAcc[128];
    for (int d = 0; d < headDim; d++) outAcc[d] = 0.0f;

    for (int ki = 0; ki < seqLen; ki++) {
        if (isCausal && ki > qi) continue;

        float score = 0.0f;
        for (int d = 0; d < headDim; d++) {
            score += query[qOffset + d] * key[kOffset + ki * headDim + d];
        }
        score *= scale;

        float newMax = fmax(rowMax, score);
        float rescale = exp(rowMax - newMax);
        rowSum = rowSum * rescale + exp(score - newMax);

        for (int d = 0; d < headDim; d++) {
            outAcc[d] = outAcc[d] * rescale + exp(score - newMax) * value[vOffset + ki * headDim + d];
        }
        rowMax = newMax;
    }

    float invSum = 1.0f / rowSum;
    for (int d = 0; d < headDim; d++) {
        output[oOffset + d] = outAcc[d] * invSum;
    }
}
";
        }

        /// <summary>
        /// Gets the names of all kernels defined in this source.
        /// </summary>
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
