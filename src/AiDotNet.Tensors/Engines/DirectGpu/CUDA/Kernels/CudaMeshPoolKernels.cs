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
// keptIndices: [numKept] - indices of edges to keep
// output: [numKept, inputChannels] - pooled features
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

// Backward pass: scatter-add gradients back to original edge positions
// gradOutput: [numKept, inputChannels] - gradient from downstream
// keptIndices: [numKept] - indices of edges that were kept
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
    float gradVal = gradOutput[keptIdx * inputChannels + channel];

    // Atomic add to handle potential duplicate indices
    atomicAdd(&gradInput[origIdx * inputChannels + channel], gradVal);
}

// Backward pass: compute gradient for importance weights
// gradOutput: [numKept, inputChannels] - gradient from downstream
// input: [numEdges, inputChannels] - original edge features
// keptIndices: [numKept] - indices of edges that were kept
// gradImportanceWeights: [inputChannels] - gradient for importance weights (output, zero-initialized)
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

    // Sum over all kept edges: grad += sum(gradOutput_channel) * input_feature
    // The importance score affects which edges are kept, so the gradient flows
    // through the edge features of kept edges
    for (int k = 0; k < numKept; k++) {
        int origIdx = keptIndices[k];

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

// Apply softmax to importance scores for differentiable edge selection
// scores: [numEdges] - raw importance scores
// softmaxScores: [numEdges] - softmax normalized scores
// temperature: temperature parameter for softmax sharpness
extern ""C"" __global__ void mesh_pool_softmax_scores(
    const float* scores,
    float* softmaxScores,
    float temperature,
    int numEdges)
{
    // First pass: find max for numerical stability
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

    // Compute exp((score - max) / temperature)
    float expVal = expf((scores[idx] - maxScore) / temperature);
    softmaxScores[idx] = expVal;

    __syncthreads();

    // Second pass: compute sum
    __shared__ float sumExp;
    if (threadIdx.x == 0) {
        sumExp = 0.0f;
        for (int i = 0; i < numEdges; i++) {
            sumExp += softmaxScores[i];
        }
    }
    __syncthreads();

    // Normalize
    softmaxScores[idx] = softmaxScores[idx] / (sumExp + EPSILON);
}

// Weighted gather using importance scores
// input: [numEdges, inputChannels]
// scores: [numEdges] - importance weights
// keptIndices: [numKept]
// output: [numKept, inputChannels]
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

// Weighted scatter-add with importance scores for backward pass
// gradOutput: [numKept, inputChannels]
// scores: [numEdges]
// keptIndices: [numKept]
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
            "mesh_pool_softmax_scores",
            "mesh_pool_weighted_gather",
            "mesh_pool_weighted_backward",
            "mesh_pool_scores_backward"
        };
    }
}
