// Copyright (c) AiDotNet. All rights reserved.
// CUDA kernels for MeshPool (edge collapse pooling) neural network operations.
// Implements graph-based edge pooling with importance scoring.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// CUDA kernels for mesh pooling operations used in MeshCNN-style networks.
/// Implements forward importance scoring and backward gradient scatter operations.
/// </summary>
internal static class CudaMeshPoolKernels
{
    public static string GetSource()
    {
        return @"
#include <math.h>

#define EPSILON 1e-15f

// ===========================================================================
// MESH POOL FORWARD KERNELS
// ===========================================================================

// Compute importance scores for all edges
// input: [numEdges, inputChannels] - edge features
// importanceWeights: [inputChannels] - learnable weights
// scores: [numEdges] - computed importance scores
extern ""C"" __global__ void mesh_pool_compute_scores(
    const float* input,
    const float* importanceWeights,
    float* scores,
    int numEdges, int inputChannels)
{
    int edge = blockIdx.x * blockDim.x + threadIdx.x;

    if (edge >= numEdges) return;

    // Compute dot product of edge features with importance weights
    float score = 0.0f;
    for (int c = 0; c < inputChannels; c++) {
        score += input[edge * inputChannels + c] * importanceWeights[c];
    }

    scores[edge] = score;
}

// Gather pooled features based on kept indices
// input: [numEdges, inputChannels] - original edge features
// keptIndices: [numKept] - indices of edges to keep (must be valid indices in [0, numEdges))
// output: [numKept, inputChannels] - pooled features
extern ""C"" __global__ void mesh_pool_gather(
    const float* input,
    const int* keptIndices,
    float* output,
    int numKept, int numEdges, int inputChannels)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = numKept * inputChannels;

    if (gid >= totalElements) return;

    int keptIdx = gid / inputChannels;
    int channel = gid % inputChannels;

    int origIdx = keptIndices[keptIdx];
    // Bounds check: ensure origIdx is valid
    if (origIdx < 0 || origIdx >= numEdges) {
        output[keptIdx * inputChannels + channel] = 0.0f;
        return;
    }
    output[keptIdx * inputChannels + channel] = input[origIdx * inputChannels + channel];
}

// ===========================================================================
// MESH POOL BACKWARD KERNELS
// ===========================================================================

// Backward pass: scatter-add gradients back to original edge positions
// gradOutput: [numKept, inputChannels] - gradient from downstream
// keptIndices: [numKept] - indices of edges that were kept (must be valid indices in [0, numEdges))
// gradInput: [numEdges, inputChannels] - gradient to input (output, zero-initialized)
extern ""C"" __global__ void mesh_pool_backward(
    const float* gradOutput,
    const int* keptIndices,
    float* gradInput,
    int numKept, int numEdges, int inputChannels)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = numKept * inputChannels;

    if (gid >= totalElements) return;

    int keptIdx = gid / inputChannels;
    int channel = gid % inputChannels;

    int origIdx = keptIndices[keptIdx];
    // Bounds check: ensure origIdx is valid before writing to gradInput
    if (origIdx < 0 || origIdx >= numEdges) return;

    float gradVal = gradOutput[keptIdx * inputChannels + channel];

    // Atomic add to handle potential duplicate indices
    atomicAdd(&gradInput[origIdx * inputChannels + channel], gradVal);
}

// Backward pass: compute gradient for importance weights
// gradOutput: [numKept, inputChannels] - gradient from downstream
// input: [numEdges, inputChannels] - original edge features
// keptIndices: [numKept] - indices of edges that were kept (must be valid indices in [0, numEdges))
// gradImportanceWeights: [inputChannels] - gradient for importance weights (output, zero-initialized)
extern ""C"" __global__ void mesh_pool_importance_backward(
    const float* gradOutput,
    const float* input,
    const int* keptIndices,
    float* gradImportanceWeights,
    int numKept, int numEdges, int inputChannels)
{
    int channel = blockIdx.x * blockDim.x + threadIdx.x;

    if (channel >= inputChannels) return;

    float grad = 0.0f;

    // Sum over all kept edges: grad += sum(gradOutput_channel) * input_feature
    // The importance score affects which edges are kept, so the gradient flows
    // through the edge features of kept edges
    for (int k = 0; k < numKept; k++) {
        int origIdx = keptIndices[k];
        // Bounds check: skip invalid indices
        if (origIdx < 0 || origIdx >= numEdges) continue;

        // Sum gradients across channels for this edge
        float gradSum = 0.0f;
        for (int c = 0; c < inputChannels; c++) {
            gradSum += gradOutput[k * inputChannels + c];
        }

        // Multiply by the feature at this channel for the original edge
        grad += gradSum * input[origIdx * inputChannels + channel];
    }

    gradImportanceWeights[channel] = grad;
}

// ===========================================================================
// MESH POOL UTILITY KERNELS
// ===========================================================================

// Zero-initialize gradient tensor
extern ""C"" __global__ void mesh_pool_zero_grad(
    float* gradInput,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size) return;

    gradInput[idx] = 0.0f;
}

// ===========================================================================
// MESH POOL SOFTMAX - MULTI-KERNEL APPROACH
// Correct implementation using separate kernels for global synchronization
// ===========================================================================

// Step 1: Parallel reduction to find maximum score
// Uses block-level reduction with shared memory
// Caller must launch enough blocks to cover all edges
// partialMax output should have size = num_blocks
extern ""C"" __global__ void mesh_pool_softmax_find_max(
    const float* scores,
    float* partialMax,
    int numEdges)
{
    extern __shared__ float scratch[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x;

    // Each thread loads its value (or -INFINITY if out of bounds)
    float val = (gid < numEdges) ? scores[gid] : -1e30f;
    scratch[tid] = val;
    __syncthreads();

    // Parallel reduction within block
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }

    // Thread 0 writes the result for this block
    if (tid == 0) {
        partialMax[blockIdx.x] = scratch[0];
    }
}

// Step 2: Final max reduction (single block)
// Reduces partialMax array to a single global max value
extern ""C"" __global__ void mesh_pool_softmax_final_max(
    const float* partialMax,
    float* globalMax,
    int numPartials)
{
    extern __shared__ float scratch[];

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    // Load partial max values
    float val = (tid < numPartials) ? partialMax[tid] : -1e30f;
    scratch[tid] = val;
    __syncthreads();

    // Parallel reduction
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        globalMax[0] = scratch[0];
    }
}

// Step 3: Compute exp values and partial sums
// Each thread computes exp((score - max) / temperature)
// Blocks compute partial sums
extern ""C"" __global__ void mesh_pool_softmax_exp_sum(
    const float* scores,
    const float* globalMax,
    float* expValues,
    float* partialSum,
    float temperature,
    int numEdges)
{
    extern __shared__ float scratch[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x;

    float maxVal = globalMax[0];
    float expVal = 0.0f;

    if (gid < numEdges) {
        expVal = expf((scores[gid] - maxVal) / temperature);
        expValues[gid] = expVal;
    }

    scratch[tid] = expVal;
    __syncthreads();

    // Parallel sum reduction within block
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partialSum[blockIdx.x] = scratch[0];
    }
}

// Step 4: Final sum reduction (single block)
extern ""C"" __global__ void mesh_pool_softmax_final_sum(
    const float* partialSum,
    float* globalSum,
    int numPartials)
{
    extern __shared__ float scratch[];

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    float val = (tid < numPartials) ? partialSum[tid] : 0.0f;
    scratch[tid] = val;
    __syncthreads();

    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        globalSum[0] = scratch[0];
    }
}

// Step 5: Normalize exp values by the global sum
extern ""C"" __global__ void mesh_pool_softmax_normalize(
    const float* expValues,
    const float* globalSum,
    float* softmaxScores,
    int numEdges)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= numEdges) return;

    float sum = globalSum[0];
    softmaxScores[gid] = expValues[gid] / (sum + EPSILON);
}

// Legacy wrapper: Single block softmax for small numEdges
// IMPORTANT: Must be launched with exactly ONE block where blockDim.x >= numEdges
// For larger numEdges, use the multi-kernel approach above
extern ""C"" __global__ void mesh_pool_softmax_scores(
    const float* scores,
    float* softmaxScores,
    float temperature,
    int numEdges)
{
    extern __shared__ float scratch[];

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    // Load scores into shared memory (padding with -INFINITY for reduction)
    float val = (tid < numEdges) ? scores[tid] : -1e30f;
    scratch[tid] = val;
    __syncthreads();

    // Step 1: Find max using parallel reduction
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }
    float maxVal = scratch[0];
    __syncthreads();

    // Step 2: Compute exp and store, then reduce sum
    float expVal = 0.0f;
    if (tid < numEdges) {
        expVal = expf((scores[tid] - maxVal) / temperature);
        softmaxScores[tid] = expVal;
    }
    scratch[tid] = expVal;
    __syncthreads();

    // Sum reduction
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    float sumExp = scratch[0];
    __syncthreads();

    // Step 3: Normalize
    if (tid < numEdges) {
        softmaxScores[tid] = softmaxScores[tid] / (sumExp + EPSILON);
    }
}

// Weighted gather using importance scores
// input: [numEdges, inputChannels]
// scores: [numEdges] - importance weights
// keptIndices: [numKept] - must be valid indices in [0, numEdges)
// output: [numKept, inputChannels]
extern ""C"" __global__ void mesh_pool_weighted_gather(
    const float* input,
    const float* scores,
    const int* keptIndices,
    float* output,
    int numKept, int numEdges, int inputChannels)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = numKept * inputChannels;

    if (gid >= totalElements) return;

    int keptIdx = gid / inputChannels;
    int channel = gid % inputChannels;

    int origIdx = keptIndices[keptIdx];
    // Bounds check: ensure origIdx is valid
    if (origIdx < 0 || origIdx >= numEdges) {
        output[keptIdx * inputChannels + channel] = 0.0f;
        return;
    }
    float weight = scores[origIdx];
    output[keptIdx * inputChannels + channel] = input[origIdx * inputChannels + channel] * weight;
}

// Weighted scatter-add with importance scores for backward pass
// gradOutput: [numKept, inputChannels]
// scores: [numEdges]
// keptIndices: [numKept] - must be valid indices in [0, numEdges)
// gradInput: [numEdges, inputChannels]
extern ""C"" __global__ void mesh_pool_weighted_backward(
    const float* gradOutput,
    const float* scores,
    const int* keptIndices,
    float* gradInput,
    int numKept, int numEdges, int inputChannels)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = numKept * inputChannels;

    if (gid >= totalElements) return;

    int keptIdx = gid / inputChannels;
    int channel = gid % inputChannels;

    int origIdx = keptIndices[keptIdx];
    // Bounds check: ensure origIdx is valid before writing to gradInput
    if (origIdx < 0 || origIdx >= numEdges) return;

    float weight = scores[origIdx];
    float gradVal = gradOutput[keptIdx * inputChannels + channel] * weight;

    atomicAdd(&gradInput[origIdx * inputChannels + channel], gradVal);
}

// Backward pass for scores: compute gradient of importance scores
// gradOutput: [numKept, inputChannels]
// input: [numEdges, inputChannels]
// keptIndices: [numKept]
// gradScores: [numEdges]
extern ""C"" __global__ void mesh_pool_scores_backward(
    const float* gradOutput,
    const float* input,
    const int* keptIndices,
    float* gradScores,
    int numKept, int numEdges, int inputChannels)
{
    int edge = blockIdx.x * blockDim.x + threadIdx.x;

    if (edge >= numEdges) return;

    // Check if this edge was kept
    int keptIdx = -1;
    for (int k = 0; k < numKept; k++) {
        if (keptIndices[k] == edge) {
            keptIdx = k;
            break;
        }
    }

    if (keptIdx < 0) {
        gradScores[edge] = 0.0f;
        return;
    }

    // Gradient: sum over channels of (gradOutput * input)
    float grad = 0.0f;
    for (int c = 0; c < inputChannels; c++) {
        grad += gradOutput[keptIdx * inputChannels + c] * input[edge * inputChannels + c];
    }

    gradScores[edge] = grad;
}
";
    }

    /// <summary>
    /// Gets the list of kernel names provided by this source.
    /// </summary>
    public static string[] GetKernelNames()
    {
        return new[]
        {
            "mesh_pool_compute_scores",
            "mesh_pool_gather",
            "mesh_pool_backward",
            "mesh_pool_importance_backward",
            "mesh_pool_zero_grad",
            // Multi-kernel softmax approach (for large numEdges)
            "mesh_pool_softmax_find_max",
            "mesh_pool_softmax_final_max",
            "mesh_pool_softmax_exp_sum",
            "mesh_pool_softmax_final_sum",
            "mesh_pool_softmax_normalize",
            // Single block softmax (for small numEdges, legacy compatibility)
            "mesh_pool_softmax_scores",
            "mesh_pool_weighted_gather",
            "mesh_pool_weighted_backward",
            "mesh_pool_scores_backward"
        };
    }
}
