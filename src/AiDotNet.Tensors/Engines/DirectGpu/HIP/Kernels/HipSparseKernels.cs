// Copyright (c) AiDotNet. All rights reserved.
// HIP sparse matrix kernels - CSR SpMM and GNN message passing operations.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// HIP kernels for sparse matrix operations including CSR SpMM (sparse matrix - dense matrix
/// multiplication) and edge-based message passing for Graph Neural Networks.
/// </summary>
internal static class HipSparseKernels
{
    public static string GetSource()
    {
        return @"
#include <hip/hip_runtime.h>
#include <math.h>

// ===========================================================================
// CSR SPARSE MATRIX - DENSE MATRIX MULTIPLICATION (SpMM)
// ===========================================================================

extern ""C"" __global__ void csr_spmm(
    const float* __restrict__ csrValues,
    const int* __restrict__ csrColIndices,
    const int* __restrict__ csrRowPointers,
    const float* __restrict__ denseB,
    float* __restrict__ output,
    int M, int K, int N, int nnz)
{
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    int rowStart = csrRowPointers[row];
    int rowEnd = csrRowPointers[row + 1];

    float sum = 0.0f;
    for (int i = rowStart; i < rowEnd; i++)
    {
        int colA = csrColIndices[i];
        float valA = csrValues[i];
        sum += valA * denseB[colA * N + col];
    }

    output[row * N + col] = sum;
}

extern ""C"" __global__ void csr_spmm_warp(
    const float* __restrict__ csrValues,
    const int* __restrict__ csrColIndices,
    const int* __restrict__ csrRowPointers,
    const float* __restrict__ denseB,
    float* __restrict__ output,
    int M, int K, int N, int nnz)
{
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 64;
    int lane = threadIdx.x % 64;
    int row = warpId;

    if (row >= M) return;

    int rowStart = csrRowPointers[row];
    int rowEnd = csrRowPointers[row + 1];

    for (int col = lane; col < N; col += 64)
    {
        float sum = 0.0f;
        for (int i = rowStart; i < rowEnd; i++)
        {
            int colA = csrColIndices[i];
            float valA = csrValues[i];
            sum += valA * denseB[colA * N + col];
        }
        output[row * N + col] = sum;
    }
}

extern ""C"" __global__ void csr_spmm_bias(
    const float* __restrict__ csrValues,
    const int* __restrict__ csrColIndices,
    const int* __restrict__ csrRowPointers,
    const float* __restrict__ denseB,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int M, int K, int N, int nnz)
{
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    int rowStart = csrRowPointers[row];
    int rowEnd = csrRowPointers[row + 1];

    float sum = bias[col];
    for (int i = rowStart; i < rowEnd; i++)
    {
        int colA = csrColIndices[i];
        float valA = csrValues[i];
        sum += valA * denseB[colA * N + col];
    }

    output[row * N + col] = sum;
}

extern ""C"" __global__ void csr_spmm_bias_relu(
    const float* __restrict__ csrValues,
    const int* __restrict__ csrColIndices,
    const int* __restrict__ csrRowPointers,
    const float* __restrict__ denseB,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int M, int K, int N, int nnz)
{
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    int rowStart = csrRowPointers[row];
    int rowEnd = csrRowPointers[row + 1];

    float sum = bias[col];
    for (int i = rowStart; i < rowEnd; i++)
    {
        int colA = csrColIndices[i];
        float valA = csrValues[i];
        sum += valA * denseB[colA * N + col];
    }

    output[row * N + col] = fmaxf(sum, 0.0f);
}

// ===========================================================================
// GRAPH NEURAL NETWORK MESSAGE PASSING OPERATIONS
// ===========================================================================

extern ""C"" __global__ void scatter_add_edges(
    const float* __restrict__ input,
    const int* __restrict__ sourceIndices,
    const int* __restrict__ targetIndices,
    const float* __restrict__ edgeValues,
    float* __restrict__ output,
    int numNodes, int numEdges, int features,
    int hasEdgeValues)
{
    int edge = blockIdx.x;
    int feat = blockIdx.y * blockDim.x + threadIdx.x;

    if (edge >= numEdges || feat >= features) return;

    int src = sourceIndices[edge];
    int tgt = targetIndices[edge];

    float value = input[src * features + feat];
    if (hasEdgeValues)
    {
        value *= edgeValues[edge];
    }

    atomicAdd(&output[tgt * features + feat], value);
}

extern ""C"" __global__ void gather_source_features(
    const float* __restrict__ input,
    const int* __restrict__ sourceIndices,
    float* __restrict__ output,
    int numEdges, int features)
{
    int edge = blockIdx.x;
    int feat = blockIdx.y * blockDim.x + threadIdx.x;

    if (edge >= numEdges || feat >= features) return;

    int src = sourceIndices[edge];
    output[edge * features + feat] = input[src * features + feat];
}

extern ""C"" __global__ void gather_target_features(
    const float* __restrict__ input,
    const int* __restrict__ targetIndices,
    float* __restrict__ output,
    int numEdges, int features)
{
    int edge = blockIdx.x;
    int feat = blockIdx.y * blockDim.x + threadIdx.x;

    if (edge >= numEdges || feat >= features) return;

    int tgt = targetIndices[edge];
    output[edge * features + feat] = input[tgt * features + feat];
}

extern ""C"" __global__ void segment_sum(
    const float* __restrict__ input,
    const int* __restrict__ segmentIds,
    float* __restrict__ output,
    int numItems, int features)
{
    int item = blockIdx.x;
    int feat = blockIdx.y * blockDim.x + threadIdx.x;

    if (item >= numItems || feat >= features) return;

    int segment = segmentIds[item];
    atomicAdd(&output[segment * features + feat], input[item * features + feat]);
}

extern ""C"" __global__ void segment_mean(
    const float* __restrict__ input,
    const int* __restrict__ segmentIds,
    const int* __restrict__ segmentSizes,
    float* __restrict__ output,
    int numItems, int features)
{
    int item = blockIdx.x;
    int feat = blockIdx.y * blockDim.x + threadIdx.x;

    if (item >= numItems || feat >= features) return;

    int segment = segmentIds[item];
    int size = segmentSizes[segment];
    if (size > 0)
    {
        atomicAdd(&output[segment * features + feat], input[item * features + feat] / (float)size);
    }
}

extern ""C"" __global__ void segment_max(
    const float* __restrict__ input,
    const int* __restrict__ segmentIds,
    float* __restrict__ output,
    int numItems, int features)
{
    int item = blockIdx.x;
    int feat = blockIdx.y * blockDim.x + threadIdx.x;

    if (item >= numItems || feat >= features) return;

    int segment = segmentIds[item];
    float val = input[item * features + feat];

    int* address = (int*)&output[segment * features + feat];
    int old = *address, assumed;
    do
    {
        assumed = old;
        float assumedF = __int_as_float(assumed);
        old = atomicCAS(address, assumed, __float_as_int(fmaxf(val, assumedF)));
    } while (assumed != old);
}

// ===========================================================================
// SPARSE MATRIX BACKWARD OPERATIONS
// ===========================================================================

extern ""C"" __global__ void csr_spmm_backward_b(
    const float* __restrict__ csrValues,
    const int* __restrict__ csrColIndices,
    const int* __restrict__ csrRowPointers,
    const float* __restrict__ gradOutput,
    float* __restrict__ gradB,
    int M, int K, int N, int nnz)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;

    for (int row = 0; row < M; row++)
    {
        float gradVal = gradOutput[row * N + col];
        if (gradVal == 0.0f) continue;

        int rowStart = csrRowPointers[row];
        int rowEnd = csrRowPointers[row + 1];

        for (int i = rowStart; i < rowEnd; i++)
        {
            int colA = csrColIndices[i];
            float valA = csrValues[i];
            atomicAdd(&gradB[colA * N + col], valA * gradVal);
        }
    }
}

extern ""C"" __global__ void csr_spmm_backward_values(
    const float* __restrict__ csrColIndices_float,
    const int* __restrict__ csrColIndices,
    const int* __restrict__ csrRowPointers,
    const float* __restrict__ denseB,
    const float* __restrict__ gradOutput,
    float* __restrict__ gradValues,
    int M, int K, int N, int nnz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;

    int row = 0;
    for (int r = 0; r < M; r++)
    {
        if (csrRowPointers[r + 1] > i)
        {
            row = r;
            break;
        }
    }

    int colA = csrColIndices[i];

    float grad = 0.0f;
    for (int col = 0; col < N; col++)
    {
        grad += gradOutput[row * N + col] * denseB[colA * N + col];
    }

    gradValues[i] = grad;
}

// ===========================================================================
// SPARSE UTILITY OPERATIONS
// ===========================================================================

extern ""C"" __global__ void zero_buffer(float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = 0.0f;
}

extern ""C"" __global__ void init_neg_inf(float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = -3.402823466e+38f;
}

extern ""C"" __global__ void degree_normalize(
    const float* __restrict__ input,
    const float* __restrict__ degrees,
    float* __restrict__ output,
    int numNodes, int features, float epsilon)
{
    int node = blockIdx.x;
    int feat = blockIdx.y * blockDim.x + threadIdx.x;

    if (node >= numNodes || feat >= features) return;

    float deg = degrees[node];
    float normFactor = rsqrtf(deg + epsilon);
    output[node * features + feat] = input[node * features + feat] * normFactor;
}

extern ""C"" __global__ void symmetric_degree_normalize(
    const float* __restrict__ edgeValues,
    const int* __restrict__ sourceIndices,
    const int* __restrict__ targetIndices,
    const float* __restrict__ srcDegrees,
    const float* __restrict__ tgtDegrees,
    float* __restrict__ output,
    int numEdges, float epsilon)
{
    int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge >= numEdges) return;

    int src = sourceIndices[edge];
    int tgt = targetIndices[edge];

    float srcDeg = srcDegrees[src];
    float tgtDeg = tgtDegrees[tgt];

    float normFactor = rsqrtf(srcDeg + epsilon) * rsqrtf(tgtDeg + epsilon);
    output[edge] = edgeValues[edge] * normFactor;
}
";
    }

    public static string[] GetKernelNames()
    {
        return
        [
            "csr_spmm",
            "csr_spmm_warp",
            "csr_spmm_bias",
            "csr_spmm_bias_relu",
            "scatter_add_edges",
            "gather_source_features",
            "gather_target_features",
            "segment_sum",
            "segment_mean",
            "segment_max",
            "csr_spmm_backward_b",
            "csr_spmm_backward_values",
            "zero_buffer",
            "init_neg_inf",
            "degree_normalize",
            "symmetric_degree_normalize"
        ];
    }
}
