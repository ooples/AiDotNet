// Copyright (c) AiDotNet. All rights reserved.
// HIP kernels for MeshPool (edge collapse pooling) neural network operations.
// Implements graph-based edge pooling with importance scoring.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// HIP kernels for mesh pooling operations used in MeshCNN-style networks.
/// Implements forward importance scoring and backward gradient scatter operations.
/// </summary>
internal static class HipMeshPoolKernels
{
    public static string GetSource()
    {
        return @"
#include <hip/hip_runtime.h>
#include <math.h>

#define EPSILON 1e-15f

// ===========================================================================
// MESH POOL FORWARD KERNELS
// ===========================================================================

extern ""C"" __global__ void mesh_pool_compute_scores(
    const float* input,
    const float* importanceWeights,
    float* scores,
    int numEdges, int inputChannels)
{
    int edge = blockIdx.x * blockDim.x + threadIdx.x;

    if (edge >= numEdges) return;

    float score = 0.0f;
    for (int c = 0; c < inputChannels; c++) {
        score += input[edge * inputChannels + c] * importanceWeights[c];
    }

    scores[edge] = score;
}

extern ""C"" __global__ void mesh_pool_gather(
    const float* input,
    const int* keptIndices,
    float* output,
    int numKept, int inputChannels)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
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
    float gradVal = gradOutput[keptIdx * inputChannels + channel];

    atomicAdd(&gradInput[origIdx * inputChannels + channel], gradVal);
}

extern ""C"" __global__ void mesh_pool_importance_backward(
    const float* gradOutput,
    const float* input,
    const int* keptIndices,
    float* gradImportanceWeights,
    int numKept, int inputChannels)
{
    int channel = blockIdx.x * blockDim.x + threadIdx.x;

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

extern ""C"" __global__ void mesh_pool_zero_grad(
    float* gradInput,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size) return;

    gradInput[idx] = 0.0f;
}

extern ""C"" __global__ void mesh_pool_softmax_scores(
    const float* scores,
    float* softmaxScores,
    float temperature,
    int numEdges)
{
    __shared__ float maxScore;
    if (threadIdx.x == 0) {
        maxScore = -1e30f;
        for (int i = 0; i < numEdges; i++) {
            if (scores[i] > maxScore) maxScore = scores[i];
        }
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numEdges) return;

    float expVal = expf((scores[idx] - maxScore) / temperature);
    softmaxScores[idx] = expVal;

    __syncthreads();

    __shared__ float sumExp;
    if (threadIdx.x == 0) {
        sumExp = 0.0f;
        for (int i = 0; i < numEdges; i++) {
            sumExp += softmaxScores[i];
        }
    }
    __syncthreads();

    softmaxScores[idx] = softmaxScores[idx] / (sumExp + EPSILON);
}

extern ""C"" __global__ void mesh_pool_weighted_gather(
    const float* input,
    const float* scores,
    const int* keptIndices,
    float* output,
    int numKept, int inputChannels)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = numKept * inputChannels;

    if (gid >= totalElements) return;

    int keptIdx = gid / inputChannels;
    int channel = gid % inputChannels;

    int origIdx = keptIndices[keptIdx];
    float weight = scores[origIdx];
    output[keptIdx * inputChannels + channel] = input[origIdx * inputChannels + channel] * weight;
}

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
    float weight = scores[origIdx];
    float gradVal = gradOutput[keptIdx * inputChannels + channel] * weight;

    atomicAdd(&gradInput[origIdx * inputChannels + channel], gradVal);
}

extern ""C"" __global__ void mesh_pool_scores_backward(
    const float* gradOutput,
    const float* input,
    const int* keptIndices,
    float* gradScores,
    int numKept, int numEdges, int inputChannels)
{
    int edge = blockIdx.x * blockDim.x + threadIdx.x;

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
