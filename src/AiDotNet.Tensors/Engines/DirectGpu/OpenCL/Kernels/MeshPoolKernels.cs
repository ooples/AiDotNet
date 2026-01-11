// Copyright (c) AiDotNet. All rights reserved.
// OpenCL kernels for MeshPool (edge collapse pooling) neural network operations.
// Implements graph-based edge pooling with importance scoring.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// OpenCL kernels for mesh pooling operations used in MeshCNN-style networks.
/// Implements forward importance scoring and backward gradient scatter operations.
/// </summary>
internal static class MeshPoolKernels
{
    public static string GetSource()
    {
        return @"
#define EPSILON 1e-15f

// ===========================================================================
// MESH POOL FORWARD KERNELS
// ===========================================================================

__kernel void mesh_pool_compute_scores(
    __global const float* input,
    __global const float* importanceWeights,
    __global float* scores,
    const int numEdges, const int inputChannels)
{
    int edge = get_global_id(0);

    if (edge >= numEdges) return;

    float score = 0.0f;
    for (int c = 0; c < inputChannels; c++) {
        score += input[edge * inputChannels + c] * importanceWeights[c];
    }

    scores[edge] = score;
}

// keptIndices: [numKept] - must be valid indices in [0, numEdges)
__kernel void mesh_pool_gather(
    __global const float* input,
    __global const int* keptIndices,
    __global float* output,
    const int numKept, const int numEdges, const int inputChannels)
{
    int gid = get_global_id(0);
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

// Note: OpenCL doesn't have built-in atomicAdd for floats in all versions.
// We use a manual implementation with atomic_cmpxchg.
inline void atomic_add_float(__global float* ptr, float val) {
    union {
        float f;
        unsigned int i;
    } old, new_val;
    do {
        old.f = *ptr;
        new_val.f = old.f + val;
    } while (atomic_cmpxchg((__global unsigned int*)ptr, old.i, new_val.i) != old.i);
}

// keptIndices: [numKept] - must be valid indices in [0, numEdges)
__kernel void mesh_pool_backward(
    __global const float* gradOutput,
    __global const int* keptIndices,
    __global float* gradInput,
    const int numKept, const int numEdges, const int inputChannels)
{
    int gid = get_global_id(0);
    int totalElements = numKept * inputChannels;

    if (gid >= totalElements) return;

    int keptIdx = gid / inputChannels;
    int channel = gid % inputChannels;

    int origIdx = keptIndices[keptIdx];
    // Bounds check: ensure origIdx is valid before writing to gradInput
    if (origIdx < 0 || origIdx >= numEdges) return;

    float gradVal = gradOutput[keptIdx * inputChannels + channel];

    atomic_add_float(&gradInput[origIdx * inputChannels + channel], gradVal);
}

// keptIndices: [numKept] - must be valid indices in [0, numEdges)
__kernel void mesh_pool_importance_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global const int* keptIndices,
    __global float* gradImportanceWeights,
    const int numKept, const int numEdges, const int inputChannels)
{
    int channel = get_global_id(0);

    if (channel >= inputChannels) return;

    float grad = 0.0f;

    for (int k = 0; k < numKept; k++) {
        int origIdx = keptIndices[k];
        // Bounds check: skip invalid indices
        if (origIdx < 0 || origIdx >= numEdges) continue;

        float gradSum = 0.0f;
        for (int c = 0; c < inputChannels; c++) {
            gradSum += gradOutput[k * inputChannels + c];
        }

        grad += gradSum * input[origIdx * inputChannels + channel];
    }

    gradImportanceWeights[channel] = grad;
}

// ===========================================================================
// MESH POOL UTILITY KERNELS
// ===========================================================================

__kernel void mesh_pool_zero_grad(
    __global float* gradInput,
    const int size)
{
    int idx = get_global_id(0);

    if (idx >= size) return;

    gradInput[idx] = 0.0f;
}

// ===========================================================================
// MESH POOL SOFTMAX - THREE-KERNEL APPROACH
// Correct implementation using separate kernels for global synchronization
// ===========================================================================

// Step 1: Parallel reduction to find maximum score
// Uses work-group reduction with local memory
// Caller must launch enough work-groups to cover all edges
// partialMax output should have size = num_work_groups
__kernel void mesh_pool_softmax_find_max(
    __global const float* scores,
    __global float* partialMax,
    __local float* scratch,
    const int numEdges)
{
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int groupId = get_group_id(0);
    int localSize = get_local_size(0);

    // Each work-item loads its value (or -INFINITY if out of bounds)
    float val = (gid < numEdges) ? scores[gid] : -1e30f;
    scratch[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Parallel reduction within work-group
    for (int stride = localSize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] = fmax(scratch[lid], scratch[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Work-item 0 writes the result for this work-group
    if (lid == 0) {
        partialMax[groupId] = scratch[0];
    }
}

// Step 2: Final max reduction (single work-group)
// Reduces partialMax array to a single global max value
__kernel void mesh_pool_softmax_final_max(
    __global const float* partialMax,
    __global float* globalMax,
    __local float* scratch,
    const int numPartials)
{
    int lid = get_local_id(0);
    int localSize = get_local_size(0);

    // Load partial max values
    float val = (lid < numPartials) ? partialMax[lid] : -1e30f;
    scratch[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Parallel reduction
    for (int stride = localSize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] = fmax(scratch[lid], scratch[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        globalMax[0] = scratch[0];
    }
}

// Step 3: Compute exp values and partial sums
// Each work-item computes exp((score - max) / temperature)
// Work-groups compute partial sums
__kernel void mesh_pool_softmax_exp_sum(
    __global const float* scores,
    __global const float* globalMax,
    __global float* expValues,
    __global float* partialSum,
    __local float* scratch,
    const float temperature,
    const int numEdges)
{
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int groupId = get_group_id(0);
    int localSize = get_local_size(0);

    float maxVal = globalMax[0];
    float expVal = 0.0f;

    if (gid < numEdges) {
        expVal = exp((scores[gid] - maxVal) / temperature);
        expValues[gid] = expVal;
    }

    scratch[lid] = expVal;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Parallel sum reduction within work-group
    for (int stride = localSize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        partialSum[groupId] = scratch[0];
    }
}

// Step 4: Final sum reduction (single work-group)
__kernel void mesh_pool_softmax_final_sum(
    __global const float* partialSum,
    __global float* globalSum,
    __local float* scratch,
    const int numPartials)
{
    int lid = get_local_id(0);
    int localSize = get_local_size(0);

    float val = (lid < numPartials) ? partialSum[lid] : 0.0f;
    scratch[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = localSize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        globalSum[0] = scratch[0];
    }
}

// Step 5: Normalize exp values by the global sum
__kernel void mesh_pool_softmax_normalize(
    __global const float* expValues,
    __global const float* globalSum,
    __global float* softmaxScores,
    const int numEdges)
{
    int gid = get_global_id(0);
    if (gid >= numEdges) return;

    float sum = globalSum[0];
    softmaxScores[gid] = expValues[gid] / (sum + EPSILON);
}

// Legacy wrapper: Single work-group softmax for small numEdges
// IMPORTANT: Must be launched with exactly ONE work-group where local_size >= numEdges
// For larger numEdges, use the multi-kernel approach above
__kernel void mesh_pool_softmax_scores(
    __global const float* scores,
    __global float* softmaxScores,
    __local float* scratch,
    const float temperature,
    const int numEdges)
{
    int lid = get_local_id(0);
    int localSize = get_local_size(0);

    // Load scores into local memory (padding with -INFINITY for reduction)
    float val = (lid < numEdges) ? scores[lid] : -1e30f;
    scratch[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 1: Find max using parallel reduction
    for (int stride = localSize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] = fmax(scratch[lid], scratch[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float maxVal = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 2: Compute exp and store, then reduce sum
    float expVal = 0.0f;
    if (lid < numEdges) {
        expVal = exp((scores[lid] - maxVal) / temperature);
        softmaxScores[lid] = expVal;
    }
    scratch[lid] = expVal;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Sum reduction
    for (int stride = localSize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float sumExp = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 3: Normalize
    if (lid < numEdges) {
        softmaxScores[lid] = softmaxScores[lid] / (sumExp + EPSILON);
    }
}

// keptIndices: [numKept] - must be valid indices in [0, numEdges)
__kernel void mesh_pool_weighted_gather(
    __global const float* input,
    __global const float* scores,
    __global const int* keptIndices,
    __global float* output,
    const int numKept, const int numEdges, const int inputChannels)
{
    int gid = get_global_id(0);
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

// keptIndices: [numKept] - must be valid indices in [0, numEdges)
__kernel void mesh_pool_weighted_backward(
    __global const float* gradOutput,
    __global const float* scores,
    __global const int* keptIndices,
    __global float* gradInput,
    const int numKept, const int numEdges, const int inputChannels)
{
    int gid = get_global_id(0);
    int totalElements = numKept * inputChannels;

    if (gid >= totalElements) return;

    int keptIdx = gid / inputChannels;
    int channel = gid % inputChannels;

    int origIdx = keptIndices[keptIdx];
    // Bounds check: ensure origIdx is valid before writing to gradInput
    if (origIdx < 0 || origIdx >= numEdges) return;

    float weight = scores[origIdx];
    float gradVal = gradOutput[keptIdx * inputChannels + channel] * weight;

    atomic_add_float(&gradInput[origIdx * inputChannels + channel], gradVal);
}

// OPTIMIZED: O(k) instead of O(n*k)
// Caller MUST zero gradScores before launching using mesh_pool_zero_grad kernel.
// Launch with global_size = numKept (NOT numEdges).
// Each work-item handles one kept edge, directly writing its gradient.
// Since each edge can appear at most once in keptIndices, no atomics needed.
__kernel void mesh_pool_scores_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global const int* keptIndices,
    __global float* gradScores,
    const int numKept, const int numEdges, const int inputChannels)
{
    int keptIdx = get_global_id(0);

    if (keptIdx >= numKept) return;

    int edge = keptIndices[keptIdx];
    // Bounds check: ensure edge index is valid
    if (edge < 0 || edge >= numEdges) return;

    float grad = 0.0f;
    for (int c = 0; c < inputChannels; c++) {
        grad += gradOutput[keptIdx * inputChannels + c] * input[edge * inputChannels + c];
    }

    gradScores[edge] = grad;
}

// Legacy version for backward compatibility - O(n*k) complexity
// Use mesh_pool_scores_backward instead for better performance
__kernel void mesh_pool_scores_backward_legacy(
    __global const float* gradOutput,
    __global const float* input,
    __global const int* keptIndices,
    __global float* gradScores,
    const int numKept, const int numEdges, const int inputChannels)
{
    int edge = get_global_id(0);

    if (edge >= numEdges) return;

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
            // Single work-group softmax (for small numEdges, legacy compatibility)
            "mesh_pool_softmax_scores",
            "mesh_pool_weighted_gather",
            "mesh_pool_weighted_backward",
            "mesh_pool_scores_backward",
            "mesh_pool_scores_backward_legacy"
        };
    }
}
