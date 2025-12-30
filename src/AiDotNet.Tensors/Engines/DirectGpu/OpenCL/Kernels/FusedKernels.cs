// Copyright (c) AiDotNet. All rights reserved.
// Fused operation kernels - GEMM + Bias + Activation in single pass.
// Eliminates memory round-trips for 20-50% performance gain.
// Works on ALL .NET versions including .NET Framework 4.6.2.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    /// <summary>
    /// Fused GPU kernels that combine multiple operations to eliminate memory round-trips.
    /// These are key optimizations missing from CLBlast.
    /// </summary>
    /// <remarks>
    /// <para><b>Performance Benefit:</b></para>
    /// <para>
    /// For DenseLayer forward pass, fusing GEMM + Bias + Activation:
    /// - Eliminates 2 global memory round-trips
    /// - Expected gain: 20-50% for memory-bound workloads
    /// </para>
    /// </remarks>
    internal static class FusedKernels
    {
        /// <summary>
        /// Gets all fused kernel sources.
        /// </summary>
        public static string GetSource()
        {
            return @"
// ===========================================================================
// FUSED KERNELS: GEMM + BIAS + ACTIVATION
// Single kernel for entire DenseLayer forward pass.
// Eliminates memory round-trips between operations.
// ===========================================================================

#define TILE_SIZE 16

// Fused GEMM + Bias + ReLU
// output = ReLU(A * B + bias)
__kernel void gemm_bias_relu(
    __global const float* A,
    __global const float* B,
    __global const float* bias,
    __global float* C,
    const int M,
    const int N,
    const int K)
{
    // Tile-based GEMM with fused bias and ReLU
    __local float As[TILE_SIZE][TILE_SIZE];
    __local float Bs[TILE_SIZE][TILE_SIZE];

    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);

    float sum = 0.0f;
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Load tiles into shared memory
        int aCol = t * TILE_SIZE + localCol;
        int bRow = t * TILE_SIZE + localRow;

        As[localRow][localCol] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[localRow][localCol] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = fma(As[localRow][k], Bs[k][localCol], sum);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Fused bias add and ReLU activation
    if (row < M && col < N) {
        float result = sum + bias[col];
        C[row * N + col] = fmax(0.0f, result);  // ReLU
    }
}

// Fused GEMM + Bias + GELU (for Transformer FFN)
// output = GELU(A * B + bias)
__kernel void gemm_bias_gelu(
    __global const float* A,
    __global const float* B,
    __global const float* bias,
    __global float* C,
    const int M,
    const int N,
    const int K)
{
    __local float As[TILE_SIZE][TILE_SIZE];
    __local float Bs[TILE_SIZE][TILE_SIZE];

    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);

    float sum = 0.0f;
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + localCol;
        int bRow = t * TILE_SIZE + localRow;

        As[localRow][localCol] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[localRow][localCol] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = fma(As[localRow][k], Bs[k][localCol], sum);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Fused bias add and GELU activation
    if (row < M && col < N) {
        float x = sum + bias[col];

        // GELU approximation
        const float SQRT_2_OVER_PI = 0.7978845608f;
        const float COEFF = 0.044715f;
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
        C[row * N + col] = 0.5f * x * (1.0f + tanh(inner));
    }
}

// Fused GEMM + Bias + Sigmoid
// output = Sigmoid(A * B + bias)
__kernel void gemm_bias_sigmoid(
    __global const float* A,
    __global const float* B,
    __global const float* bias,
    __global float* C,
    const int M,
    const int N,
    const int K)
{
    __local float As[TILE_SIZE][TILE_SIZE];
    __local float Bs[TILE_SIZE][TILE_SIZE];

    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);

    float sum = 0.0f;
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + localCol;
        int bRow = t * TILE_SIZE + localRow;

        As[localRow][localCol] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[localRow][localCol] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = fma(As[localRow][k], Bs[k][localCol], sum);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Fused bias add and Sigmoid activation
    if (row < M && col < N) {
        float x = sum + bias[col];
        C[row * N + col] = 1.0f / (1.0f + exp(-x));
    }
}

// Fused GEMM + Bias + Tanh
// output = Tanh(A * B + bias)
__kernel void gemm_bias_tanh(
    __global const float* A,
    __global const float* B,
    __global const float* bias,
    __global float* C,
    const int M,
    const int N,
    const int K)
{
    __local float As[TILE_SIZE][TILE_SIZE];
    __local float Bs[TILE_SIZE][TILE_SIZE];

    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);

    float sum = 0.0f;
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + localCol;
        int bRow = t * TILE_SIZE + localRow;

        As[localRow][localCol] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[localRow][localCol] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = fma(As[localRow][k], Bs[k][localCol], sum);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Fused bias add and Tanh activation
    if (row < M && col < N) {
        float x = sum + bias[col];
        C[row * N + col] = tanh(x);
    }
}

// Fused GEMM + Bias (no activation, but saves a kernel launch)
// output = A * B + bias
__kernel void gemm_bias(
    __global const float* A,
    __global const float* B,
    __global const float* bias,
    __global float* C,
    const int M,
    const int N,
    const int K)
{
    __local float As[TILE_SIZE][TILE_SIZE];
    __local float Bs[TILE_SIZE][TILE_SIZE];

    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);

    float sum = 0.0f;
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + localCol;
        int bRow = t * TILE_SIZE + localRow;

        As[localRow][localCol] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[localRow][localCol] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = fma(As[localRow][k], Bs[k][localCol], sum);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Just bias add, no activation
    if (row < M && col < N) {
        C[row * N + col] = sum + bias[col];
    }
}
";
        }

        /// <summary>
        /// Gets kernel names for compilation.
        /// </summary>
        public static string[] GetKernelNames()
        {
            return new string[]
            {
                "gemm_bias_relu", "gemm_bias_gelu",
                "gemm_bias_sigmoid", "gemm_bias_tanh", "gemm_bias"
            };
        }
    }
}
