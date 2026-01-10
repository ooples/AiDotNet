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

__kernel void mesh_pool_gather(
    __global const float* input,
    __global const int* keptIndices,
    __global float* output,
    const int numKept, const int inputChannels)
{
    int gid = get_global_id(0);
    int totalElements = numKept * inputChannels;

    if (gid >= totalElements) return;

    int keptIdx = gid / inputChannels;
    int channel = gid % inputChannels;

    int origIdx = keptIndices[keptIdx];
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
    float gradVal = gradOutput[keptIdx * inputChannels + channel];

    atomic_add_float(&gradInput[origIdx * inputChannels + channel], gradVal);
}

__kernel void mesh_pool_importance_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global const int* keptIndices,
    __global float* gradImportanceWeights,
    const int numKept, const int inputChannels)
{
    int channel = get_global_id(0);

    if (channel >= inputChannels) return;

    float grad = 0.0f;

    for (int k = 0; k < numKept; k++) {
        int origIdx = keptIndices[k];

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

__kernel void mesh_pool_softmax_scores(
    __global const float* scores,
    __global float* softmaxScores,
    const float temperature,
    const int numEdges)
{
    // Find max for numerical stability (single work-item reduction)
    int idx = get_global_id(0);

    if (idx >= numEdges) return;

    // Each work-item computes its own exp value
    // Note: For production, a proper parallel reduction should be used
    float maxScore = -1e30f;
    for (int i = 0; i < numEdges; i++) {
        if (scores[i] > maxScore) maxScore = scores[i];
    }

    float expVal = exp((scores[idx] - maxScore) / temperature);
    softmaxScores[idx] = expVal;

    barrier(CLK_GLOBAL_MEM_FENCE);

    float sumExp = 0.0f;
    for (int i = 0; i < numEdges; i++) {
        sumExp += softmaxScores[i];
    }

    softmaxScores[idx] = softmaxScores[idx] / (sumExp + EPSILON);
}

__kernel void mesh_pool_weighted_gather(
    __global const float* input,
    __global const float* scores,
    __global const int* keptIndices,
    __global float* output,
    const int numKept, const int inputChannels)
{
    int gid = get_global_id(0);
    int totalElements = numKept * inputChannels;

    if (gid >= totalElements) return;

    int keptIdx = gid / inputChannels;
    int channel = gid % inputChannels;

    int origIdx = keptIndices[keptIdx];
    float weight = scores[origIdx];
    output[keptIdx * inputChannels + channel] = input[origIdx * inputChannels + channel] * weight;
}

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
    float weight = scores[origIdx];
    float gradVal = gradOutput[keptIdx * inputChannels + channel] * weight;

    atomic_add_float(&gradInput[origIdx * inputChannels + channel], gradVal);
}

__kernel void mesh_pool_scores_backward(
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
            "mesh_pool_softmax_scores",
            "mesh_pool_weighted_gather",
            "mesh_pool_weighted_backward",
            "mesh_pool_scores_backward"
        };
    }
}
