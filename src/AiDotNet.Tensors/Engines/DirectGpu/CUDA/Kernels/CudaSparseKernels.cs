// Copyright (c) AiDotNet. All rights reserved.
// CUDA sparse matrix kernels - CSR SpMM and GNN message passing operations.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// CUDA kernels for sparse matrix operations including CSR SpMM (sparse matrix - dense matrix
/// multiplication) and edge-based message passing for Graph Neural Networks.
/// </summary>
/// <remarks>
/// <para><b>CSR Format (Compressed Sparse Row):</b></para>
/// <para>
/// A sparse matrix of shape [M, K] is stored as three arrays:
/// - Values: Non-zero values in row-major order [nnz elements]
/// - ColumnIndices: Column index for each value [nnz elements]
/// - RowPointers: Start of each row in values array [M+1 elements]
/// </para>
/// <para>
/// This format enables O(nnz) SpMM operations instead of O(M*K) for dense matrices,
/// providing significant speedup for sparse graphs (typically 90%+ sparse).
/// </para>
/// </remarks>
internal static class CudaSparseKernels
{
    public static string GetSource()
    {
        return @"
#include <math.h>

// ===========================================================================
// CSR SPARSE MATRIX - DENSE MATRIX MULTIPLICATION (SpMM)
// ===========================================================================

// CSR SpMM: C[M,N] = A[M,K] * B[K,N] where A is sparse (CSR format)
// Each thread computes one element of the output matrix.
// Row parallelism: each thread block handles a row of the output.
extern ""C"" __global__ void csr_spmm(
    const float* __restrict__ csrValues,       // [nnz] - non-zero values of A
    const int* __restrict__ csrColIndices,     // [nnz] - column indices
    const int* __restrict__ csrRowPointers,    // [M+1] - row pointers
    const float* __restrict__ denseB,          // [K,N] - dense matrix B
    float* __restrict__ output,                // [M,N] - output matrix C
    int M, int K, int N, int nnz)
{
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    // Get row bounds in CSR values array
    int rowStart = csrRowPointers[row];
    int rowEnd = csrRowPointers[row + 1];

    // Compute dot product of sparse row A[row,:] with dense column B[:,col]
    float sum = 0.0f;
    for (int i = rowStart; i < rowEnd; i++)
    {
        int colA = csrColIndices[i];
        float valA = csrValues[i];
        sum += valA * denseB[colA * N + col];
    }

    output[row * N + col] = sum;
}

// Warp-based CSR SpMM for better memory coalescing
// Each warp processes one row, with threads collaborating on the reduction
extern ""C"" __global__ void csr_spmm_warp(
    const float* __restrict__ csrValues,
    const int* __restrict__ csrColIndices,
    const int* __restrict__ csrRowPointers,
    const float* __restrict__ denseB,
    float* __restrict__ output,
    int M, int K, int N, int nnz)
{
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int row = warpId;

    if (row >= M) return;

    int rowStart = csrRowPointers[row];
    int rowEnd = csrRowPointers[row + 1];

    // Each thread in warp handles different output columns
    for (int col = lane; col < N; col += 32)
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

// CSR SpMM with fused bias addition: C[M,N] = A[M,K] * B[K,N] + bias[N]
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

    float sum = bias[col];  // Start with bias
    for (int i = rowStart; i < rowEnd; i++)
    {
        int colA = csrColIndices[i];
        float valA = csrValues[i];
        sum += valA * denseB[colA * N + col];
    }

    output[row * N + col] = sum;
}

// CSR SpMM with fused bias and ReLU: C = ReLU(A * B + bias)
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

// Scatter-add for edge-based message passing (GCN, GAT aggregation)
// For each edge (src -> tgt), adds input[src, :] * edgeValue to output[tgt, :]
// This implements the aggregation: output[tgt] = sum_{src in neighbors(tgt)} input[src] * edge_weight
extern ""C"" __global__ void scatter_add_edges(
    const float* __restrict__ input,           // [numNodes, features]
    const int* __restrict__ sourceIndices,     // [numEdges] - source node for each edge
    const int* __restrict__ targetIndices,     // [numEdges] - target node for each edge
    const float* __restrict__ edgeValues,      // [numEdges] or nullptr for unweighted
    float* __restrict__ output,                // [numNodes, features]
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

// Message passing: gather features from source nodes for each edge
// output[edge, :] = input[source[edge], :]
extern ""C"" __global__ void gather_source_features(
    const float* __restrict__ input,           // [numNodes, features]
    const int* __restrict__ sourceIndices,     // [numEdges]
    float* __restrict__ output,                // [numEdges, features]
    int numEdges, int features)
{
    int edge = blockIdx.x;
    int feat = blockIdx.y * blockDim.x + threadIdx.x;

    if (edge >= numEdges || feat >= features) return;

    int src = sourceIndices[edge];
    output[edge * features + feat] = input[src * features + feat];
}

// Message passing: gather features from target nodes for each edge
// output[edge, :] = input[target[edge], :]
extern ""C"" __global__ void gather_target_features(
    const float* __restrict__ input,           // [numNodes, features]
    const int* __restrict__ targetIndices,     // [numEdges]
    float* __restrict__ output,                // [numEdges, features]
    int numEdges, int features)
{
    int edge = blockIdx.x;
    int feat = blockIdx.y * blockDim.x + threadIdx.x;

    if (edge >= numEdges || feat >= features) return;

    int tgt = targetIndices[edge];
    output[edge * features + feat] = input[tgt * features + feat];
}

// Segment sum: sum features within each segment (node)
// Used for aggregating edge messages back to nodes
// output[node, :] = sum_{edge in edges_to_node} input[edge, :]
extern ""C"" __global__ void segment_sum(
    const float* __restrict__ input,           // [numItems, features]
    const int* __restrict__ segmentIds,        // [numItems] - segment ID for each item
    float* __restrict__ output,                // [numSegments, features]
    int numItems, int features)
{
    int item = blockIdx.x;
    int feat = blockIdx.y * blockDim.x + threadIdx.x;

    if (item >= numItems || feat >= features) return;

    int segment = segmentIds[item];
    atomicAdd(&output[segment * features + feat], input[item * features + feat]);
}

// Segment mean: compute mean features within each segment
// Requires segment sizes as input
extern ""C"" __global__ void segment_mean(
    const float* __restrict__ input,           // [numItems, features]
    const int* __restrict__ segmentIds,        // [numItems]
    const int* __restrict__ segmentSizes,      // [numSegments]
    float* __restrict__ output,                // [numSegments, features]
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

// Segment max: compute max features within each segment
extern ""C"" __global__ void segment_max(
    const float* __restrict__ input,           // [numItems, features]
    const int* __restrict__ segmentIds,        // [numItems]
    float* __restrict__ output,                // [numSegments, features] - initialized to -FLT_MAX
    int numItems, int features)
{
    int item = blockIdx.x;
    int feat = blockIdx.y * blockDim.x + threadIdx.x;

    if (item >= numItems || feat >= features) return;

    int segment = segmentIds[item];
    float val = input[item * features + feat];

    // Use atomicMax for floats (requires casting to int for atomicMax)
    // This is a common workaround for float atomicMax
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

// CSR SpMM backward for gradient w.r.t. dense matrix B
// Given dL/dC, compute dL/dB = A^T * dL/dC (A is sparse CSR)
// This is also SpMM but with transposed access pattern
extern ""C"" __global__ void csr_spmm_backward_b(
    const float* __restrict__ csrValues,       // [nnz] - values of A
    const int* __restrict__ csrColIndices,     // [nnz] - column indices
    const int* __restrict__ csrRowPointers,    // [M+1] - row pointers
    const float* __restrict__ gradOutput,      // [M,N] - dL/dC
    float* __restrict__ gradB,                 // [K,N] - dL/dB (must be zero-initialized)
    int M, int K, int N, int nnz)
{
    // Each thread handles one column of B
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;

    // For each row of A (and gradOutput)
    for (int row = 0; row < M; row++)
    {
        float gradVal = gradOutput[row * N + col];
        if (gradVal == 0.0f) continue;

        int rowStart = csrRowPointers[row];
        int rowEnd = csrRowPointers[row + 1];

        // For each non-zero in this row of A
        for (int i = rowStart; i < rowEnd; i++)
        {
            int colA = csrColIndices[i];  // This is the row index in B
            float valA = csrValues[i];
            // dB[colA, col] += A[row, colA] * dC[row, col]
            atomicAdd(&gradB[colA * N + col], valA * gradVal);
        }
    }
}

// CSR SpMM backward for gradient w.r.t. sparse values (for trainable adjacency)
// Given dL/dC, compute dL/d(csrValues)
extern ""C"" __global__ void csr_spmm_backward_values(
    const float* __restrict__ csrColIndices_float,  // Use float for column indices
    const int* __restrict__ csrColIndices,
    const int* __restrict__ csrRowPointers,
    const float* __restrict__ denseB,
    const float* __restrict__ gradOutput,
    float* __restrict__ gradValues,            // [nnz] - dL/d(csrValues)
    int M, int K, int N, int nnz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;

    // Find which row this value belongs to
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

    // dL/d(A[row,colA]) = sum_col dL/dC[row,col] * B[colA,col]
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

// Zero initialize buffer
extern ""C"" __global__ void zero_buffer(float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = 0.0f;
}

// Initialize buffer to negative infinity (for max operations)
extern ""C"" __global__ void init_neg_inf(float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = -3.402823466e+38f;  // -FLT_MAX
}

// Degree normalization for GCN: out[i] = in[i] / sqrt(degree[i])
extern ""C"" __global__ void degree_normalize(
    const float* __restrict__ input,
    const float* __restrict__ degrees,         // [numNodes]
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

// Symmetric degree normalization: out[i] = in[i] / (sqrt(degree[src]) * sqrt(degree[tgt]))
// Applied per-edge
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
            // CSR SpMM operations
            "csr_spmm",
            "csr_spmm_warp",
            "csr_spmm_bias",
            "csr_spmm_bias_relu",
            // GNN message passing
            "scatter_add_edges",
            "gather_source_features",
            "gather_target_features",
            "segment_sum",
            "segment_mean",
            "segment_max",
            // Backward operations
            "csr_spmm_backward_b",
            "csr_spmm_backward_values",
            // Utilities
            "zero_buffer",
            "init_neg_inf",
            "degree_normalize",
            "symmetric_degree_normalize"
        ];
    }
}
