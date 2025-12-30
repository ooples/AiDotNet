// Copyright (c) AiDotNet. All rights reserved.
// CLBlast-style optimized GEMM kernel with proper register blocking and cooperative loading.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    /// <summary>
    /// High-performance GEMM kernels using CLBlast-proven techniques:
    /// - 16x16 work groups (256 threads, universally compatible)
    /// - 8x8 register blocking per thread (128x128 output tile per work group)
    /// - Cooperative tile loading to shared memory
    /// - Bank-conflict-free shared memory layout
    /// - Vectorized float4 loads
    /// </summary>
    internal static class GemmKernel
    {
        // Kernel configuration constants (must match kernel #defines)
        // Using 128x128 tiles with 8x8 register blocking - high arithmetic intensity
        public const int WG_SIZE_M = 16;
        public const int WG_SIZE_N = 16;
        public const int TILE_M = 128;
        public const int TILE_N = 128;
        public const int TILE_K = 16;
        public const int OUTPUTS_M = 8;
        public const int OUTPUTS_N = 8;

        /// <summary>
        /// Gets the optimized GEMM kernel source.
        /// </summary>
        public static string GetSource()
        {
            return @"
// ===========================================================================
// CLBlast-Style High-Performance GEMM with TRUE Double Buffering
// Target: 2500+ GFLOPS on AMD GPUs
// Key optimization: Overlap memory loads with computation using 2 buffers
// ===========================================================================

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Configuration - 16x16 work group, 8x8 register blocking = 128x128 tile
// High arithmetic intensity for maximum GFLOPS
#define WG_SIZE_M 16
#define WG_SIZE_N 16
#define TILE_M 128
#define TILE_N 128
#define TILE_K 16     // K tile size
#define OUTPUTS_M 8   // 8x8 = 64 outputs per thread
#define OUTPUTS_N 8

// Padding for bank conflict avoidance (AMD RDNA: 32 banks, need offset=8 for stride access)
#define PAD 8

// ===========================================================================
// Helper macros for double buffering
// ===========================================================================
#define LOAD_A_TILE(As, kBase, wgRowStart, M, K) \
    _Pragma(""unroll"") \
    for (int i = 0; i < 8; i++) { \
        int loadIdx = tid + i * 256; \
        int loadRow = loadIdx / TILE_K; \
        int loadCol = loadIdx % TILE_K; \
        int globalRow = wgRowStart + loadRow; \
        int globalCol = kBase + loadCol; \
        if (globalRow < M && globalCol < K) { \
            As[loadRow][loadCol] = A[globalRow * K + globalCol]; \
        } else { \
            As[loadRow][loadCol] = 0.0f; \
        } \
    }

#define LOAD_B_TILE(Bs, kBase, wgColStart, K, N) \
    _Pragma(""unroll"") \
    for (int i = 0; i < 2; i++) { \
        int loadIdx = tid + i * 256; \
        int loadRow = loadIdx / 32; \
        int loadCol4 = loadIdx % 32; \
        int globalRow = kBase + loadRow; \
        int globalCol = wgColStart + loadCol4 * 4; \
        if (globalRow < K && globalCol + 3 < N) { \
            float4 vec = vload4(0, &B[globalRow * N + globalCol]); \
            Bs[loadRow][loadCol4 * 4 + 0] = vec.x; \
            Bs[loadRow][loadCol4 * 4 + 1] = vec.y; \
            Bs[loadRow][loadCol4 * 4 + 2] = vec.z; \
            Bs[loadRow][loadCol4 * 4 + 3] = vec.w; \
        } else if (globalRow < K) { \
            for (int c = 0; c < 4; c++) { \
                int col = globalCol + c; \
                Bs[loadRow][loadCol4 * 4 + c] = (col < N) ? B[globalRow * N + col] : 0.0f; \
            } \
        } else { \
            Bs[loadRow][loadCol4 * 4 + 0] = 0.0f; \
            Bs[loadRow][loadCol4 * 4 + 1] = 0.0f; \
            Bs[loadRow][loadCol4 * 4 + 2] = 0.0f; \
            Bs[loadRow][loadCol4 * 4 + 3] = 0.0f; \
        } \
    }

#define COMPUTE_TILE(As, Bs, acc) \
    _Pragma(""unroll"") \
    for (int kk = 0; kk < TILE_K; kk++) { \
        float aVals[OUTPUTS_M]; \
        _Pragma(""unroll"") \
        for (int mi = 0; mi < OUTPUTS_M; mi++) { \
            aVals[mi] = As[tidM * OUTPUTS_M + mi][kk]; \
        } \
        float bVals[OUTPUTS_N]; \
        _Pragma(""unroll"") \
        for (int ni = 0; ni < OUTPUTS_N; ni++) { \
            bVals[ni] = Bs[kk][tidN * OUTPUTS_N + ni]; \
        } \
        _Pragma(""unroll"") \
        for (int mi = 0; mi < OUTPUTS_M; mi++) { \
            _Pragma(""unroll"") \
            for (int ni = 0; ni < OUTPUTS_N; ni++) { \
                acc[mi][ni] = fma(aVals[mi], bVals[ni], acc[mi][ni]); \
            } \
        } \
    }

// ===========================================================================
// Main GEMM Kernel - TRUE Double Buffering
// C = alpha * A * B + beta * C
// Uses ping-pong buffers to overlap memory loads with computation
// ===========================================================================
__kernel __attribute__((reqd_work_group_size(WG_SIZE_M, WG_SIZE_N, 1)))
void gemm_double_buffered(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float beta)
{
    // Work group and thread indices
    const int wgRow = get_group_id(0);
    const int wgCol = get_group_id(1);
    const int tidM = get_local_id(0);
    const int tidN = get_local_id(1);
    const int tid = tidM * WG_SIZE_N + tidN;  // Linear thread ID (0-255)

    // Double-buffered shared memory: two sets of tiles for ping-pong
    // Buffer 0 and Buffer 1 for A and B tiles
    __local float As0[TILE_M][TILE_K + PAD];
    __local float As1[TILE_M][TILE_K + PAD];
    __local float Bs0[TILE_K][TILE_N + PAD];
    __local float Bs1[TILE_K][TILE_N + PAD];

    // Register accumulators - each thread computes 8x8 = 64 outputs
    float acc[OUTPUTS_M][OUTPUTS_N];
    #pragma unroll
    for (int i = 0; i < OUTPUTS_M; i++) {
        #pragma unroll
        for (int j = 0; j < OUTPUTS_N; j++) {
            acc[i][j] = 0.0f;
        }
    }

    // Base row/col for this thread's output block
    const int outRowBase = wgRow * TILE_M + tidM * OUTPUTS_M;
    const int outColBase = wgCol * TILE_N + tidN * OUTPUTS_N;

    // Number of K tiles
    const int numKTiles = (K + TILE_K - 1) / TILE_K;

    // Precompute base addresses for this work group
    const int wgRowStart = wgRow * TILE_M;
    const int wgColStart = wgCol * TILE_N;

    // Early exit for trivial case
    if (numKTiles == 0) {
        // Just handle beta * C
        #pragma unroll
        for (int mi = 0; mi < OUTPUTS_M; mi++) {
            int outRow = outRowBase + mi;
            if (outRow < M) {
                #pragma unroll
                for (int ni = 0; ni < OUTPUTS_N; ni++) {
                    int outCol = outColBase + ni;
                    if (outCol < N) {
                        int idx = outRow * N + outCol;
                        C[idx] = beta * C[idx];
                    }
                }
            }
        }
        return;
    }

    // ===== Load first tile into buffer 0 =====
    LOAD_A_TILE(As0, 0, wgRowStart, M, K)
    LOAD_B_TILE(Bs0, 0, wgColStart, K, N)
    barrier(CLK_LOCAL_MEM_FENCE);

    // Main K-loop with double buffering
    // Pattern: Load(buf1) -> Compute(buf0) -> swap -> Load(buf0) -> Compute(buf1) -> swap
    for (int kt = 0; kt < numKTiles - 1; kt++) {
        const int kBaseNext = (kt + 1) * TILE_K;

        if ((kt & 1) == 0) {
            // kt is even: compute on buffer 0, load into buffer 1
            // Start loading next tile into buffer 1 (async)
            LOAD_A_TILE(As1, kBaseNext, wgRowStart, M, K)
            LOAD_B_TILE(Bs1, kBaseNext, wgColStart, K, N)

            // Compute on current tile in buffer 0
            COMPUTE_TILE(As0, Bs0, acc)
        } else {
            // kt is odd: compute on buffer 1, load into buffer 0
            // Start loading next tile into buffer 0 (async)
            LOAD_A_TILE(As0, kBaseNext, wgRowStart, M, K)
            LOAD_B_TILE(Bs0, kBaseNext, wgColStart, K, N)

            // Compute on current tile in buffer 1
            COMPUTE_TILE(As1, Bs1, acc)
        }

        // Synchronize before next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Process last tile (no next tile to load)
    if ((numKTiles - 1) & 1) {
        // Last tile is in buffer 1
        COMPUTE_TILE(As1, Bs1, acc)
    } else {
        // Last tile is in buffer 0
        COMPUTE_TILE(As0, Bs0, acc)
    }

    // ===== Write results to global memory using vectorized stores =====
    #pragma unroll
    for (int mi = 0; mi < OUTPUTS_M; mi++) {
        int outRow = outRowBase + mi;
        if (outRow < M) {
            // Store 8 columns as 2 float4s
            #pragma unroll
            for (int ni4 = 0; ni4 < 2; ni4++) {
                int outCol = outColBase + ni4 * 4;
                if (outCol + 3 < N) {
                    int idx = outRow * N + outCol;
                    float4 result;
                    result.x = alpha * acc[mi][ni4 * 4 + 0];
                    result.y = alpha * acc[mi][ni4 * 4 + 1];
                    result.z = alpha * acc[mi][ni4 * 4 + 2];
                    result.w = alpha * acc[mi][ni4 * 4 + 3];

                    if (beta != 0.0f) {
                        float4 cOld = vload4(0, &C[idx]);
                        result.x += beta * cOld.x;
                        result.y += beta * cOld.y;
                        result.z += beta * cOld.z;
                        result.w += beta * cOld.w;
                    }
                    vstore4(result, 0, &C[idx]);
                } else {
                    // Scalar fallback for edge
                    for (int ni = ni4 * 4; ni < OUTPUTS_N && outColBase + ni < N; ni++) {
                        int idx = outRow * N + outColBase + ni;
                        if (beta != 0.0f) {
                            C[idx] = alpha * acc[mi][ni] + beta * C[idx];
                        } else {
                            C[idx] = alpha * acc[mi][ni];
                        }
                    }
                }
            }
        }
    }
}

// ===========================================================================
// Simple GEMM for Small Matrices (< 128x128)
// Uses direct computation without complex tiling
// ===========================================================================
__kernel void gemm_small(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float beta)
{
    const int row = get_global_id(0);
    const int col = get_global_id(1);

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum = fma(A[row * K + k], B[k * N + col], sum);
    }

    const int idx = row * N + col;
    if (beta != 0.0f) {
        C[idx] = alpha * sum + beta * C[idx];
    } else {
        C[idx] = alpha * sum;
    }
}

// ===========================================================================
// Vectorized GEMM using float4 for 4x memory bandwidth
// ===========================================================================
__kernel void gemm_vectorized(
    __global const float* restrict A,
    __global const float4* restrict B,  // B is accessed as float4 columns
    __global float4* restrict C,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float beta)
{
    const int row = get_global_id(0);
    const int col4 = get_global_id(1);  // Column index / 4

    if (row >= M || col4 >= (N / 4)) return;

    float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    const int N4 = N / 4;

    for (int k = 0; k < K; k++) {
        float aVal = A[row * K + k];
        float4 bVec = B[k * N4 + col4];
        sum = fma((float4)(aVal), bVec, sum);
    }

    const int idx = row * N4 + col4;
    if (beta != 0.0f) {
        C[idx] = alpha * sum + beta * C[idx];
    } else {
        C[idx] = alpha * sum;
    }
}

// ===========================================================================
// Batched Persistent GEMM with Work-Stealing
// Processes multiple small matrices efficiently
// ===========================================================================
__kernel void gemm_batched_persistent(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    const int M,
    const int N,
    const int K,
    const int batchCount,
    __global volatile int* batchCounter,
    const float alpha,
    const float beta)
{
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);
    const int localId = localRow * get_local_size(1) + localCol;
    const int wgSize = get_local_size(0) * get_local_size(1);

    // Use local memory for batch index coordination
    __local int sharedBatchIdx;

    // Each work group processes batches via work-stealing
    while (true) {
        // Work-steal: atomically get next batch index
        if (localId == 0) {
            sharedBatchIdx = atomic_inc(batchCounter);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int batchIdx = sharedBatchIdx;
        if (batchIdx >= batchCount) break;

        // Pointers to this batch's matrices
        __global const float* batchA = A + batchIdx * M * K;
        __global const float* batchB = B + batchIdx * K * N;
        __global float* batchC = C + batchIdx * M * N;

        // Simple per-element computation for batched case
        int elementsPerBatch = M * N;
        int elementsPerThread = (elementsPerBatch + wgSize - 1) / wgSize;

        for (int e = 0; e < elementsPerThread; e++) {
            int elemIdx = localId + e * wgSize;
            if (elemIdx >= elementsPerBatch) break;

            int row = elemIdx / N;
            int col = elemIdx % N;

            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum = fma(batchA[row * K + k], batchB[k * N + col], sum);
            }

            int outIdx = row * N + col;
            if (beta != 0.0f) {
                batchC[outIdx] = alpha * sum + beta * batchC[outIdx];
            } else {
                batchC[outIdx] = alpha * sum;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
";
        }

        /// <summary>
        /// Gets the list of kernel names provided by this source.
        /// </summary>
        public static string[] GetKernelNames()
        {
            return new string[]
            {
                "gemm_double_buffered",
                "gemm_small",
                "gemm_vectorized",
                "gemm_batched_persistent"
            };
        }
    }
}
