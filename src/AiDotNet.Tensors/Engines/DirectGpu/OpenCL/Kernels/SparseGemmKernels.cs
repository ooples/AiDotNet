// Copyright (c) AiDotNet. All rights reserved.
// Sparse GEMM kernels with 2:4 structured sparsity support.
// Provides 2x compression and up to 2x speedup on compatible hardware.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// Sparse GEMM GPU kernels implementing 2:4 structured sparsity.
/// </summary>
/// <remarks>
/// <para><b>2:4 Structured Sparsity:</b></para>
/// <para>
/// In 2:4 sparsity, every group of 4 consecutive elements has exactly 2 zeros.
/// This provides:
/// - 2x memory compression (store only non-zero values + metadata)
/// - 2x compute reduction (skip zero multiplications)
/// - Hardware support on AMD MI200+, NVIDIA Ampere+
/// </para>
/// <para><b>Storage Format:</b></para>
/// <para>
/// For a sparse matrix A with 2:4 pattern:
/// - Values: Dense array of non-zero values (50% of original)
/// - Indices: 2-bit indices indicating positions within each group of 4
/// </para>
/// <para><b>Performance Target:</b> 2x speedup over dense GEMM for 50% sparse matrices</para>
/// </remarks>
internal static class SparseGemmKernels
{
    /// <summary>
    /// Gets the OpenCL kernel source for sparse GEMM operations.
    /// </summary>
    public static string GetSource()
    {
        return @"
// ===========================================================================
// SPARSE GEMM KERNELS - 2:4 Structured Sparsity
// Every group of 4 elements has exactly 2 non-zeros.
// Provides 2x compression and 2x compute speedup.
// ===========================================================================

#define TILE_SIZE 16
#define SPARSE_GROUP_SIZE 4
#define SPARSE_NONZEROS 2

// Unpack 2-bit index from packed byte
// indices: 2 positions packed into 4 bits (2 bits each)
inline int2 unpack_indices(uchar packed) {
    int2 idx;
    idx.x = packed & 0x3;        // First 2 bits
    idx.y = (packed >> 2) & 0x3; // Next 2 bits
    return idx;
}

// ===========================================================================
// Sparse GEMM: C = A_sparse * B_dense
// A is stored in 2:4 compressed format
// B is dense
// ===========================================================================
__kernel void sparse_gemm_2_4(
    __global const float* A_values,      // Compressed values (M * K/2)
    __global const uchar* A_indices,     // 2-bit packed indices (M * K/4)
    __global const float* B,             // Dense B matrix (K x N)
    __global float* C,                   // Output C matrix (M x N)
    const int M,
    const int N,
    const int K,                         // Original K (dense)
    const float alpha,
    const float beta)
{
    __local float Bs[TILE_SIZE][TILE_SIZE];

    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);

    float sum = 0.0f;

    // K_compressed = K / 2 (2 non-zeros per 4 elements)
    const int K_compressed = K / 2;
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Load B tile into shared memory
        int bRow = t * TILE_SIZE + localRow;
        int bCol = col;

        if (bRow < K && bCol < N) {
            Bs[localRow][localCol] = B[bRow * N + bCol];
        } else {
            Bs[localRow][localCol] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Process sparse A elements for this tile
        if (row < M) {
            int kStart = t * TILE_SIZE;
            int kEnd = min(kStart + TILE_SIZE, K);

            // Iterate through groups of 4 in K dimension
            for (int kGroup = kStart / SPARSE_GROUP_SIZE;
                 kGroup < (kEnd + SPARSE_GROUP_SIZE - 1) / SPARSE_GROUP_SIZE;
                 kGroup++) {

                int kBase = kGroup * SPARSE_GROUP_SIZE;
                if (kBase >= K) break;

                // Get compressed index for this row and K-group
                int compressedIdx = row * (K / SPARSE_GROUP_SIZE) + kGroup;
                uchar packedIndices = A_indices[compressedIdx];
                int2 indices = unpack_indices(packedIndices);

                // Get the two non-zero values
                int valueIdx = row * K_compressed + kGroup * SPARSE_NONZEROS;
                float val0 = A_values[valueIdx];
                float val1 = A_values[valueIdx + 1];

                // Actual K positions
                int k0 = kBase + indices.x;
                int k1 = kBase + indices.y;

                // Accumulate if within tile range
                if (k0 >= kStart && k0 < kEnd) {
                    int localK = k0 - kStart;
                    if (localK < TILE_SIZE) {
                        sum += val0 * Bs[localK][localCol];
                    }
                }

                if (k1 >= kStart && k1 < kEnd) {
                    int localK = k1 - kStart;
                    if (localK < TILE_SIZE) {
                        sum += val1 * Bs[localK][localCol];
                    }
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result
    if (row < M && col < N) {
        int idx = row * N + col;
        C[idx] = alpha * sum + beta * C[idx];
    }
}

// ===========================================================================
// Sparse GEMM with fused bias and ReLU
// C = ReLU(A_sparse * B_dense + bias)
// ===========================================================================
__kernel void sparse_gemm_bias_relu(
    __global const float* A_values,
    __global const uchar* A_indices,
    __global const float* B,
    __global const float* bias,
    __global float* C,
    const int M,
    const int N,
    const int K)
{
    __local float Bs[TILE_SIZE][TILE_SIZE];

    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);

    float sum = 0.0f;
    const int K_compressed = K / 2;
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int bRow = t * TILE_SIZE + localRow;
        int bCol = col;

        if (bRow < K && bCol < N) {
            Bs[localRow][localCol] = B[bRow * N + bCol];
        } else {
            Bs[localRow][localCol] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (row < M) {
            int kStart = t * TILE_SIZE;
            int kEnd = min(kStart + TILE_SIZE, K);

            for (int kGroup = kStart / SPARSE_GROUP_SIZE;
                 kGroup < (kEnd + SPARSE_GROUP_SIZE - 1) / SPARSE_GROUP_SIZE;
                 kGroup++) {

                int kBase = kGroup * SPARSE_GROUP_SIZE;
                if (kBase >= K) break;

                int compressedIdx = row * (K / SPARSE_GROUP_SIZE) + kGroup;
                uchar packedIndices = A_indices[compressedIdx];
                int2 indices = unpack_indices(packedIndices);

                int valueIdx = row * K_compressed + kGroup * SPARSE_NONZEROS;
                float val0 = A_values[valueIdx];
                float val1 = A_values[valueIdx + 1];

                int k0 = kBase + indices.x;
                int k1 = kBase + indices.y;

                if (k0 >= kStart && k0 < kEnd) {
                    int localK = k0 - kStart;
                    if (localK < TILE_SIZE) {
                        sum += val0 * Bs[localK][localCol];
                    }
                }

                if (k1 >= kStart && k1 < kEnd) {
                    int localK = k1 - kStart;
                    if (localK < TILE_SIZE) {
                        sum += val1 * Bs[localK][localCol];
                    }
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Fused bias + ReLU
    if (row < M && col < N) {
        float result = sum + bias[col];
        C[row * N + col] = fmax(0.0f, result);
    }
}

// ===========================================================================
// Dense to 2:4 Sparse Conversion
// Enforces 2:4 sparsity pattern by zeroing smallest 2 values per group
// ===========================================================================
__kernel void enforce_2_4_sparsity(
    __global const float* dense_input,   // M x K dense matrix
    __global float* sparse_values,       // M x K/2 compressed values
    __global uchar* sparse_indices,      // M x K/4 packed indices
    const int M,
    const int K)
{
    const int row = get_global_id(0);
    const int group = get_global_id(1);  // K group index

    if (row >= M || group >= K / SPARSE_GROUP_SIZE) return;

    int kBase = group * SPARSE_GROUP_SIZE;

    // Load 4 values
    float vals[4];
    for (int i = 0; i < 4; i++) {
        int k = kBase + i;
        vals[i] = (k < K) ? dense_input[row * K + k] : 0.0f;
    }

    // Find indices of 2 largest absolute values
    float absVals[4];
    for (int i = 0; i < 4; i++) {
        absVals[i] = fabs(vals[i]);
    }

    // Simple selection of top 2
    int idx0 = 0, idx1 = 1;
    float max0 = absVals[0], max1 = absVals[1];

    if (max1 > max0) {
        int tmpIdx = idx0; idx0 = idx1; idx1 = tmpIdx;
        float tmpVal = max0; max0 = max1; max1 = tmpVal;
    }

    for (int i = 2; i < 4; i++) {
        if (absVals[i] > max0) {
            idx1 = idx0;
            max1 = max0;
            idx0 = i;
            max0 = absVals[i];
        } else if (absVals[i] > max1) {
            idx1 = i;
            max1 = absVals[i];
        }
    }

    // Ensure idx0 < idx1 for consistent ordering
    if (idx0 > idx1) {
        int tmp = idx0; idx0 = idx1; idx1 = tmp;
    }

    // Store compressed values
    int valueIdx = row * (K / 2) + group * 2;
    sparse_values[valueIdx] = vals[idx0];
    sparse_values[valueIdx + 1] = vals[idx1];

    // Pack indices into byte (2 bits each)
    uchar packed = (uchar)((idx1 << 2) | idx0);
    sparse_indices[row * (K / SPARSE_GROUP_SIZE) + group] = packed;
}

// ===========================================================================
// Sparse to Dense Decompression
// ===========================================================================
__kernel void decompress_2_4_sparse(
    __global const float* sparse_values,
    __global const uchar* sparse_indices,
    __global float* dense_output,
    const int M,
    const int K)
{
    const int row = get_global_id(0);
    const int group = get_global_id(1);

    if (row >= M || group >= K / SPARSE_GROUP_SIZE) return;

    int kBase = group * SPARSE_GROUP_SIZE;

    // Zero initialize
    for (int i = 0; i < 4; i++) {
        int k = kBase + i;
        if (k < K) {
            dense_output[row * K + k] = 0.0f;
        }
    }

    // Get packed indices
    uchar packed = sparse_indices[row * (K / SPARSE_GROUP_SIZE) + group];
    int idx0 = packed & 0x3;
    int idx1 = (packed >> 2) & 0x3;

    // Get values
    int valueIdx = row * (K / 2) + group * 2;
    float val0 = sparse_values[valueIdx];
    float val1 = sparse_values[valueIdx + 1];

    // Write to dense
    dense_output[row * K + kBase + idx0] = val0;
    dense_output[row * K + kBase + idx1] = val1;
}

// ===========================================================================
// Compute sparsity ratio of a matrix
// ===========================================================================
__kernel void compute_sparsity_ratio(
    __global const float* input,
    __global int* zero_count,
    const int size,
    const float threshold)
{
    const int idx = get_global_id(0);

    if (idx >= size) return;

    if (fabs(input[idx]) < threshold) {
        atomic_add(zero_count, 1);
    }
}
";
    }

    /// <summary>
    /// Gets kernel names for compilation.
    /// </summary>
    public static string[] GetKernelNames()
    {
        return new[]
        {
            "sparse_gemm_2_4",
            "sparse_gemm_bias_relu",
            "enforce_2_4_sparsity",
            "decompress_2_4_sparse",
            "compute_sparsity_ratio"
        };
    }
}
