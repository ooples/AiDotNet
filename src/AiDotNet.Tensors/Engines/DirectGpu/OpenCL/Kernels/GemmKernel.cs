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
        // Optimized for RDNA1 (RX 5500 XT): 64x64 tiles with 4x4 register blocking
        // Lower register pressure = higher occupancy = better performance on limited CUs
        public const int WG_SIZE_M = 16;
        public const int WG_SIZE_N = 16;
        public const int TILE_M = 64;
        public const int TILE_N = 64;
        public const int TILE_K = 16;
        public const int OUTPUTS_M = 4;
        public const int OUTPUTS_N = 4;

        /// <summary>
        /// Gets the optimized GEMM kernel source.
        /// </summary>
        public static string GetSource()
        {
            return @"
// ===========================================================================
// RDNA1-Optimized High-Performance GEMM with TRUE Double Buffering
// Target: 2500+ GFLOPS on AMD GPUs
// Optimized for low register pressure and high occupancy on RDNA1 (RX 5500 XT)
// ===========================================================================

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Configuration - 16x16 work group, 4x4 register blocking = 64x64 tile
// Lower register pressure = higher occupancy on limited CUs
#define WG_SIZE_M 16
#define WG_SIZE_N 16
#define TILE_M 64
#define TILE_N 64
#define TILE_K 16     // K tile size
#define OUTPUTS_M 4   // 4x4 = 16 outputs per thread (vs 8x8 = 64)
#define OUTPUTS_N 4

// Padding for bank conflict avoidance (AMD RDNA: 32 banks, need offset=4)
#define PAD 4

// ===========================================================================
// Helper macros for double buffering (64x64 tiles)
// ===========================================================================
// Vectorized A loading: 64 rows * 16 cols = 1024 elements = 256 float4s
// 256 threads * 1 iteration = 256 float4 loads (each thread loads 1 float4)
#define LOAD_A_TILE(As, kBase, wgRowStart, M, K) \
    { \
        int loadRow = tid / 4;   /* 4 = TILE_K/4 float4s per row, tid 0-255 -> row 0-63 */ \
        int loadCol4 = tid % 4;  /* float4 index within row (0-3) */ \
        int globalRow = wgRowStart + loadRow; \
        int globalCol = kBase + loadCol4 * 4; \
        if (globalRow < M && globalCol + 3 < K) { \
            float4 vec = vload4(0, &A[globalRow * K + globalCol]); \
            As[loadRow][loadCol4 * 4 + 0] = vec.x; \
            As[loadRow][loadCol4 * 4 + 1] = vec.y; \
            As[loadRow][loadCol4 * 4 + 2] = vec.z; \
            As[loadRow][loadCol4 * 4 + 3] = vec.w; \
        } else if (globalRow < M) { \
            for (int c = 0; c < 4; c++) { \
                int col = globalCol + c; \
                As[loadRow][loadCol4 * 4 + c] = (col < K) ? A[globalRow * K + col] : 0.0f; \
            } \
        } else { \
            As[loadRow][loadCol4 * 4 + 0] = 0.0f; \
            As[loadRow][loadCol4 * 4 + 1] = 0.0f; \
            As[loadRow][loadCol4 * 4 + 2] = 0.0f; \
            As[loadRow][loadCol4 * 4 + 3] = 0.0f; \
        } \
    }

// Vectorized B loading: 16 rows * 64 cols = 1024 elements = 256 float4s
#define LOAD_B_TILE(Bs, kBase, wgColStart, K, N) \
    { \
        int loadRow = tid / 16;  /* 16 = TILE_N/4 float4s per row, tid 0-255 -> row 0-15 */ \
        int loadCol4 = tid % 16; /* float4 index within row (0-15) */ \
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

// KREG = 4: Process 4 K values per loop iteration for better register utilization
#define KREG 4

#define COMPUTE_TILE(As, Bs, acc) \
    _Pragma(""unroll"") \
    for (int kk = 0; kk < TILE_K; kk += KREG) { \
        /* Load KREG A values for each of OUTPUTS_M rows */ \
        float aVals0[OUTPUTS_M], aVals1[OUTPUTS_M], aVals2[OUTPUTS_M], aVals3[OUTPUTS_M]; \
        _Pragma(""unroll"") \
        for (int mi = 0; mi < OUTPUTS_M; mi++) { \
            aVals0[mi] = As[tidM * OUTPUTS_M + mi][kk + 0]; \
            aVals1[mi] = As[tidM * OUTPUTS_M + mi][kk + 1]; \
            aVals2[mi] = As[tidM * OUTPUTS_M + mi][kk + 2]; \
            aVals3[mi] = As[tidM * OUTPUTS_M + mi][kk + 3]; \
        } \
        /* Load KREG B values for each of OUTPUTS_N columns */ \
        float bVals0[OUTPUTS_N], bVals1[OUTPUTS_N], bVals2[OUTPUTS_N], bVals3[OUTPUTS_N]; \
        _Pragma(""unroll"") \
        for (int ni = 0; ni < OUTPUTS_N; ni++) { \
            bVals0[ni] = Bs[kk + 0][tidN * OUTPUTS_N + ni]; \
            bVals1[ni] = Bs[kk + 1][tidN * OUTPUTS_N + ni]; \
            bVals2[ni] = Bs[kk + 2][tidN * OUTPUTS_N + ni]; \
            bVals3[ni] = Bs[kk + 3][tidN * OUTPUTS_N + ni]; \
        } \
        /* 4 outer products accumulated */ \
        _Pragma(""unroll"") \
        for (int mi = 0; mi < OUTPUTS_M; mi++) { \
            _Pragma(""unroll"") \
            for (int ni = 0; ni < OUTPUTS_N; ni++) { \
                acc[mi][ni] = fma(aVals0[mi], bVals0[ni], acc[mi][ni]); \
                acc[mi][ni] = fma(aVals1[mi], bVals1[ni], acc[mi][ni]); \
                acc[mi][ni] = fma(aVals2[mi], bVals2[ni], acc[mi][ni]); \
                acc[mi][ni] = fma(aVals3[mi], bVals3[ni], acc[mi][ni]); \
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

    // Register accumulators - each thread computes 4x4 = 16 outputs
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
            // Store 4 columns as 1 float4 (OUTPUTS_N = 4)
            int outCol = outColBase;
            if (outCol + 3 < N) {
                int idx = outRow * N + outCol;
                float4 result;
                result.x = alpha * acc[mi][0];
                result.y = alpha * acc[mi][1];
                result.z = alpha * acc[mi][2];
                result.w = alpha * acc[mi][3];

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
                #pragma unroll
                for (int ni = 0; ni < OUTPUTS_N; ni++) {
                    int outColScalar = outColBase + ni;
                    if (outColScalar < N) {
                        int idx = outRow * N + outColScalar;
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
// Low-Register GEMM - 2x2 outputs per thread for higher occupancy
// Uses 32x32 tiles with 16x16 work group, each thread computes 2x2 outputs
// Target: Higher occupancy on register-limited RDNA1 GPUs
// ===========================================================================
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void gemm_low_register(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float beta)
{
    // 32x32 tile, 16x16 work group, 2x2 outputs per thread
    const int TILE = 32;
    const int OUT = 2;

    const int wgRow = get_group_id(0);
    const int wgCol = get_group_id(1);
    const int tidM = get_local_id(0);
    const int tidN = get_local_id(1);
    const int tid = tidM * 16 + tidN;

    // Shared memory with padding
    __local float As[32][33];  // 32x32 + padding
    __local float Bs[32][33];

    // Only 4 accumulators - minimal register pressure
    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;

    const int baseRow = wgRow * TILE + tidM * OUT;
    const int baseCol = wgCol * TILE + tidN * OUT;

    // Main K loop
    for (int kt = 0; kt < K; kt += TILE) {
        // Cooperative loading: 256 threads load 32x32 = 1024 elements
        // Each thread loads 4 elements (spread across rows)
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int loadRow = (tid + i * 256) / 32;
            int loadCol = (tid + i * 256) % 32;
            if (loadRow < 32) {
                int gRow = wgRow * TILE + loadRow;
                int gCol = kt + loadCol;
                As[loadRow][loadCol] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0f;

                int bRow = kt + loadRow;
                int bCol = wgCol * TILE + loadCol;
                Bs[loadRow][loadCol] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute 2x2 output tile
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            float a0 = As[tidM * OUT + 0][k];
            float a1 = As[tidM * OUT + 1][k];
            float b0 = Bs[k][tidN * OUT + 0];
            float b1 = Bs[k][tidN * OUT + 1];

            acc00 = fma(a0, b0, acc00);
            acc01 = fma(a0, b1, acc01);
            acc10 = fma(a1, b0, acc10);
            acc11 = fma(a1, b1, acc11);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write results
    if (baseRow < M && baseCol < N) {
        int idx = baseRow * N + baseCol;
        C[idx] = (beta != 0.0f) ? alpha * acc00 + beta * C[idx] : alpha * acc00;
    }
    if (baseRow < M && baseCol + 1 < N) {
        int idx = baseRow * N + baseCol + 1;
        C[idx] = (beta != 0.0f) ? alpha * acc01 + beta * C[idx] : alpha * acc01;
    }
    if (baseRow + 1 < M && baseCol < N) {
        int idx = (baseRow + 1) * N + baseCol;
        C[idx] = (beta != 0.0f) ? alpha * acc10 + beta * C[idx] : alpha * acc10;
    }
    if (baseRow + 1 < M && baseCol + 1 < N) {
        int idx = (baseRow + 1) * N + baseCol + 1;
        C[idx] = (beta != 0.0f) ? alpha * acc11 + beta * C[idx] : alpha * acc11;
    }
}

// ===========================================================================
// Simple Tiled GEMM - for comparing against double-buffered version
// Uses shared memory tiling without double buffering
// ===========================================================================
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void gemm_tiled_simple(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float beta)
{
    const int TILE = 16;  // Simple 16x16 tiles

    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);

    // Shared memory tiles
    __local float As[16][17];  // +1 padding for bank conflicts
    __local float Bs[16][17];

    float sum = 0.0f;

    // Process tiles of K
    for (int t = 0; t < K; t += TILE) {
        // Load A tile - each thread loads one element
        if (row < M && (t + localCol) < K) {
            As[localRow][localCol] = A[row * K + t + localCol];
        } else {
            As[localRow][localCol] = 0.0f;
        }

        // Load B tile - each thread loads one element
        if ((t + localRow) < K && col < N) {
            Bs[localRow][localCol] = B[(t + localRow) * N + col];
        } else {
            Bs[localRow][localCol] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum = fma(As[localRow][k], Bs[k][localCol], sum);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result
    if (row < M && col < N) {
        int idx = row * N + col;
        if (beta != 0.0f) {
            C[idx] = alpha * sum + beta * C[idx];
        } else {
            C[idx] = alpha * sum;
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

// ===========================================================================
// GEMM Variation 1: Small Tile (16x16 tiles, 1x1 output per thread)
// Hypothesis: Minimal register pressure for higher occupancy baseline
// Each thread computes exactly 1 output element
// ===========================================================================
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void gemm_small_tile(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float beta)
{
    // 16x16 tile, 1 output per thread = minimal register pressure
    const int TILE = 16;

    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);

    // Shared memory tiles with padding for bank conflicts
    __local float As[16][17];
    __local float Bs[16][17];

    float sum = 0.0f;

    // Number of tiles in K dimension
    int numKTiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < numKTiles; t++) {
        int kBase = t * TILE;

        // Cooperative loading: each thread loads one element
        int aRow = row;
        int aCol = kBase + localCol;
        As[localRow][localCol] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;

        int bRow = kBase + localRow;
        int bCol = col;
        Bs[localRow][localCol] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute partial dot product - fully unrolled for 16 iterations
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum = fma(As[localRow][k], Bs[k][localCol], sum);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result
    if (row < M && col < N) {
        int idx = row * N + col;
        if (beta != 0.0f) {
            C[idx] = alpha * sum + beta * C[idx];
        } else {
            C[idx] = alpha * sum;
        }
    }
}

// ===========================================================================
// GEMM Variation 2: Medium Tile (32x32 tiles, 2x2 output per thread)
// Hypothesis: Better balance between register usage and shared memory loads
// Each thread computes 2x2 outputs = 4 registers for accumulators
// ===========================================================================
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void gemm_medium_tile(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float beta)
{
    // 32x32 tile, 16x16 work group, 2x2 outputs per thread
    const int TILE = 32;
    const int OUT = 2;

    const int wgRow = get_group_id(0);
    const int wgCol = get_group_id(1);
    const int tidM = get_local_id(0);
    const int tidN = get_local_id(1);
    const int tid = tidM * 16 + tidN;

    // Shared memory with padding
    __local float As[32][33];
    __local float Bs[32][33];

    // 4 accumulators (2x2 output block)
    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;

    // Base indices for this thread's output block
    const int baseRow = wgRow * TILE + tidM * OUT;
    const int baseCol = wgCol * TILE + tidN * OUT;

    // Main K loop
    int numKTiles = (K + TILE - 1) / TILE;
    for (int kt = 0; kt < numKTiles; kt++) {
        int kBase = kt * TILE;

        // Cooperative loading: 256 threads load 32x32 = 1024 elements
        // Each thread loads 4 elements in a strided pattern
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int flatIdx = tid + i * 256;
            if (flatIdx < TILE * TILE) {
                int loadRow = flatIdx / TILE;
                int loadCol = flatIdx % TILE;

                int gRowA = wgRow * TILE + loadRow;
                int gColA = kBase + loadCol;
                As[loadRow][loadCol] = (gRowA < M && gColA < K) ? A[gRowA * K + gColA] : 0.0f;

                int gRowB = kBase + loadRow;
                int gColB = wgCol * TILE + loadCol;
                Bs[loadRow][loadCol] = (gRowB < K && gColB < N) ? B[gRowB * N + gColB] : 0.0f;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute 2x2 output tile
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            float a0 = As[tidM * OUT + 0][k];
            float a1 = As[tidM * OUT + 1][k];
            float b0 = Bs[k][tidN * OUT + 0];
            float b1 = Bs[k][tidN * OUT + 1];

            acc00 = fma(a0, b0, acc00);
            acc01 = fma(a0, b1, acc01);
            acc10 = fma(a1, b0, acc10);
            acc11 = fma(a1, b1, acc11);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write results (2x2 block)
    if (baseRow < M && baseCol < N) {
        int idx = baseRow * N + baseCol;
        C[idx] = (beta != 0.0f) ? alpha * acc00 + beta * C[idx] : alpha * acc00;
    }
    if (baseRow < M && baseCol + 1 < N) {
        int idx = baseRow * N + baseCol + 1;
        C[idx] = (beta != 0.0f) ? alpha * acc01 + beta * C[idx] : alpha * acc01;
    }
    if (baseRow + 1 < M && baseCol < N) {
        int idx = (baseRow + 1) * N + baseCol;
        C[idx] = (beta != 0.0f) ? alpha * acc10 + beta * C[idx] : alpha * acc10;
    }
    if (baseRow + 1 < M && baseCol + 1 < N) {
        int idx = (baseRow + 1) * N + baseCol + 1;
        C[idx] = (beta != 0.0f) ? alpha * acc11 + beta * C[idx] : alpha * acc11;
    }
}

// ===========================================================================
// GEMM Variation 3: Coalesced Memory Access
// Hypothesis: Focus on memory coalescing - threads in a wave access consecutive addresses
// Uses optimized loading pattern for best memory access coalescence
// ===========================================================================
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void gemm_coalesced(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float beta)
{
    // 16x16 tile with focus on coalesced access patterns
    const int TILE = 16;

    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);

    // Shared memory with padding to avoid bank conflicts
    __local float As[16][17];
    __local float Bs[16][17];

    float sum = 0.0f;

    // Work group base indices
    const int wgRowStart = get_group_id(0) * TILE;
    const int wgColStart = get_group_id(1) * TILE;

    int numKTiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < numKTiles; t++) {
        int kBase = t * TILE;

        // COALESCED loading pattern:
        // For A: threads with consecutive localCol access consecutive memory
        // Row i, columns 0-15 -> consecutive threads access A[row][kBase..kBase+15]
        int aRow = wgRowStart + localRow;
        int aCol = kBase + localCol;
        if (aRow < M && aCol < K) {
            As[localRow][localCol] = A[aRow * K + aCol];
        } else {
            As[localRow][localCol] = 0.0f;
        }

        // For B: transpose during load for better compute access
        // Thread (r,c) loads B[kBase+r][wgColStart+c]
        // Threads in same row (same r, different c) access consecutive B columns
        int bRow = kBase + localRow;
        int bCol = wgColStart + localCol;
        if (bRow < K && bCol < N) {
            Bs[localRow][localCol] = B[bRow * N + bCol];
        } else {
            Bs[localRow][localCol] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute with good register reuse
        // Manually unroll by 4 for better ILP
        #pragma unroll 4
        for (int k = 0; k < TILE; k++) {
            sum = fma(As[localRow][k], Bs[k][localCol], sum);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // COALESCED write: threads in same row write to consecutive columns
    if (row < M && col < N) {
        int idx = row * N + col;
        if (beta != 0.0f) {
            C[idx] = alpha * sum + beta * C[idx];
        } else {
            C[idx] = alpha * sum;
        }
    }
}

// ===========================================================================
// GEMM Variation 4: Vectorized Tile (float4 for all loads and stores)
// Hypothesis: Maximize memory bandwidth using wide vector loads
// Each thread processes 4 columns at once using float4
// ===========================================================================
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void gemm_vectorized_tile(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float beta)
{
    // 16x64 tile: 16 rows, 64 columns (each thread handles 4 columns via float4)
    // Work group: 16x16, each thread computes 1 row x 4 cols = 4 outputs
    // Note: using VEC_K_SIZE instead of TILE_K to avoid macro conflict
    const int VEC_TILE_ROWS = 16;
    const int VEC_K_SIZE = 16;

    const int wgRow = get_group_id(0);
    const int wgCol = get_group_id(1);
    const int tidM = get_local_id(0);  // Row within work group (0-15)
    const int tidN = get_local_id(1);  // Column group within work group (0-15)

    // Each thread handles 4 consecutive columns
    const int row = wgRow * VEC_TILE_ROWS + tidM;
    const int col4 = wgCol * 64 + tidN * 4;  // Base column (multiple of 4)

    // Shared memory for A tile (16 x 16 + padding)
    __local float As[16][17];

    // Accumulators for 4 output columns
    float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    int numKTiles = (K + VEC_K_SIZE - 1) / VEC_K_SIZE;

    for (int t = 0; t < numKTiles; t++) {
        int kBase = t * VEC_K_SIZE;

        // Load A tile cooperatively (16x16 elements, one per thread)
        int aRow = wgRow * VEC_TILE_ROWS + tidM;
        int aCol = kBase + tidN;
        if (aRow < M && aCol < K) {
            As[tidM][tidN] = A[aRow * K + aCol];
        } else {
            As[tidM][tidN] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute: for each k, load A from shared, B as float4 from global
        #pragma unroll
        for (int k = 0; k < VEC_K_SIZE; k++) {
            float aVal = As[tidM][k];

            // Vectorized B load: 4 consecutive columns
            int bRow = kBase + k;
            int bCol = col4;

            if (bRow < K && bCol + 3 < N) {
                // Fast path: aligned float4 load
                float4 bVec = vload4(0, &B[bRow * N + bCol]);
                sum = fma((float4)(aVal), bVec, sum);
            } else if (bRow < K) {
                // Slow path: scalar loads for edge case
                float b0 = (bCol + 0 < N) ? B[bRow * N + bCol + 0] : 0.0f;
                float b1 = (bCol + 1 < N) ? B[bRow * N + bCol + 1] : 0.0f;
                float b2 = (bCol + 2 < N) ? B[bRow * N + bCol + 2] : 0.0f;
                float b3 = (bCol + 3 < N) ? B[bRow * N + bCol + 3] : 0.0f;
                sum.x = fma(aVal, b0, sum.x);
                sum.y = fma(aVal, b1, sum.y);
                sum.z = fma(aVal, b2, sum.z);
                sum.w = fma(aVal, b3, sum.w);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write results using vectorized store
    if (row < M) {
        if (col4 + 3 < N) {
            // Fast path: aligned float4 store
            int idx = row * N + col4;
            float4 result = alpha * sum;

            if (beta != 0.0f) {
                float4 cOld = vload4(0, &C[idx]);
                result = fma((float4)(beta), cOld, result);
            }
            vstore4(result, 0, &C[idx]);
        } else {
            // Slow path: scalar stores for edge case
            if (col4 + 0 < N) {
                int idx = row * N + col4 + 0;
                C[idx] = (beta != 0.0f) ? alpha * sum.x + beta * C[idx] : alpha * sum.x;
            }
            if (col4 + 1 < N) {
                int idx = row * N + col4 + 1;
                C[idx] = (beta != 0.0f) ? alpha * sum.y + beta * C[idx] : alpha * sum.y;
            }
            if (col4 + 2 < N) {
                int idx = row * N + col4 + 2;
                C[idx] = (beta != 0.0f) ? alpha * sum.z + beta * C[idx] : alpha * sum.z;
            }
            if (col4 + 3 < N) {
                int idx = row * N + col4 + 3;
                C[idx] = (beta != 0.0f) ? alpha * sum.w + beta * C[idx] : alpha * sum.w;
            }
        }
    }
}

// ===========================================================================
// GEMM Variation 5: K-Register Unrolling (KREG=4)
// Hypothesis: Process 4 K iterations per loop to reduce stalls and improve ILP
// Based on CLBlast KREG parameter - load 4 A and 4 B values, do 4 FMAs per iteration
// ===========================================================================
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void gemm_kreg4(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float beta)
{
    // 32x32 tile, 16x16 work group, 2x2 outputs per thread, KREG=4
    const int TILE = 32;
    const int OUT = 2;
    const int KREG_UNROLL = 4;  // Process 4 K values per iteration

    const int wgRow = get_group_id(0);
    const int wgCol = get_group_id(1);
    const int tidM = get_local_id(0);
    const int tidN = get_local_id(1);
    const int tid = tidM * 16 + tidN;

    // Shared memory with padding
    __local float As[32][33];
    __local float Bs[32][33];

    // 4 accumulators (2x2 output block)
    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;

    // Base indices for this thread's output block
    const int baseRow = wgRow * TILE + tidM * OUT;
    const int baseCol = wgCol * TILE + tidN * OUT;

    // Main K loop
    int numKTiles = (K + TILE - 1) / TILE;
    for (int kt = 0; kt < numKTiles; kt++) {
        int kBase = kt * TILE;

        // Cooperative loading: 256 threads load 32x32 = 1024 elements
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int flatIdx = tid + i * 256;
            if (flatIdx < TILE * TILE) {
                int loadRow = flatIdx / TILE;
                int loadCol = flatIdx % TILE;

                int gRowA = wgRow * TILE + loadRow;
                int gColA = kBase + loadCol;
                As[loadRow][loadCol] = (gRowA < M && gColA < K) ? A[gRowA * K + gColA] : 0.0f;

                int gRowB = kBase + loadRow;
                int gColB = wgCol * TILE + loadCol;
                Bs[loadRow][loadCol] = (gRowB < K && gColB < N) ? B[gRowB * N + gColB] : 0.0f;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute 2x2 output tile with KREG=4 unrolling
        // Process 4 K values per iteration for better ILP
        #pragma unroll
        for (int k = 0; k < TILE; k += KREG_UNROLL) {
            // Load 4 A values for each row
            float a0_0 = As[tidM * OUT + 0][k + 0];
            float a0_1 = As[tidM * OUT + 0][k + 1];
            float a0_2 = As[tidM * OUT + 0][k + 2];
            float a0_3 = As[tidM * OUT + 0][k + 3];

            float a1_0 = As[tidM * OUT + 1][k + 0];
            float a1_1 = As[tidM * OUT + 1][k + 1];
            float a1_2 = As[tidM * OUT + 1][k + 2];
            float a1_3 = As[tidM * OUT + 1][k + 3];

            // Load 4 B values for each column
            float b0_0 = Bs[k + 0][tidN * OUT + 0];
            float b0_1 = Bs[k + 1][tidN * OUT + 0];
            float b0_2 = Bs[k + 2][tidN * OUT + 0];
            float b0_3 = Bs[k + 3][tidN * OUT + 0];

            float b1_0 = Bs[k + 0][tidN * OUT + 1];
            float b1_1 = Bs[k + 1][tidN * OUT + 1];
            float b1_2 = Bs[k + 2][tidN * OUT + 1];
            float b1_3 = Bs[k + 3][tidN * OUT + 1];

            // 4 FMAs per accumulator (4 K values)
            acc00 = fma(a0_0, b0_0, acc00);
            acc00 = fma(a0_1, b0_1, acc00);
            acc00 = fma(a0_2, b0_2, acc00);
            acc00 = fma(a0_3, b0_3, acc00);

            acc01 = fma(a0_0, b1_0, acc01);
            acc01 = fma(a0_1, b1_1, acc01);
            acc01 = fma(a0_2, b1_2, acc01);
            acc01 = fma(a0_3, b1_3, acc01);

            acc10 = fma(a1_0, b0_0, acc10);
            acc10 = fma(a1_1, b0_1, acc10);
            acc10 = fma(a1_2, b0_2, acc10);
            acc10 = fma(a1_3, b0_3, acc10);

            acc11 = fma(a1_0, b1_0, acc11);
            acc11 = fma(a1_1, b1_1, acc11);
            acc11 = fma(a1_2, b1_2, acc11);
            acc11 = fma(a1_3, b1_3, acc11);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write results (2x2 block)
    if (baseRow < M && baseCol < N) {
        int idx = baseRow * N + baseCol;
        C[idx] = (beta != 0.0f) ? alpha * acc00 + beta * C[idx] : alpha * acc00;
    }
    if (baseRow < M && baseCol + 1 < N) {
        int idx = baseRow * N + baseCol + 1;
        C[idx] = (beta != 0.0f) ? alpha * acc01 + beta * C[idx] : alpha * acc01;
    }
    if (baseRow + 1 < M && baseCol < N) {
        int idx = (baseRow + 1) * N + baseCol;
        C[idx] = (beta != 0.0f) ? alpha * acc10 + beta * C[idx] : alpha * acc10;
    }
    if (baseRow + 1 < M && baseCol + 1 < N) {
        int idx = (baseRow + 1) * N + baseCol + 1;
        C[idx] = (beta != 0.0f) ? alpha * acc11 + beta * C[idx] : alpha * acc11;
    }
}

// ===========================================================================
// GEMM Variation 6: Software Prefetch
// Hypothesis: Load next K-tile while computing current tile to hide memory latency
// Uses explicit prefetch hints and async pattern for better memory/compute overlap
// ===========================================================================
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void gemm_prefetch(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float beta)
{
    // 32x32 tile with prefetching, 16x16 work group, 2x2 outputs per thread
    const int TILE = 32;
    const int OUT = 2;

    const int wgRow = get_group_id(0);
    const int wgCol = get_group_id(1);
    const int tidM = get_local_id(0);
    const int tidN = get_local_id(1);
    const int tid = tidM * 16 + tidN;

    // Double-buffered shared memory for ping-pong prefetching
    __local float As0[32][33];
    __local float As1[32][33];
    __local float Bs0[32][33];
    __local float Bs1[32][33];

    // 4 accumulators (2x2 output block)
    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;

    // Base indices
    const int baseRow = wgRow * TILE + tidM * OUT;
    const int baseCol = wgCol * TILE + tidN * OUT;
    const int wgRowStart = wgRow * TILE;
    const int wgColStart = wgCol * TILE;

    int numKTiles = (K + TILE - 1) / TILE;
    if (numKTiles == 0) {
        // Handle beta * C case
        if (baseRow < M && baseCol < N && beta != 0.0f) {
            C[baseRow * N + baseCol] = beta * C[baseRow * N + baseCol];
        }
        return;
    }

    // Load first tile into buffer 0
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int flatIdx = tid + i * 256;
        if (flatIdx < TILE * TILE) {
            int loadRow = flatIdx / TILE;
            int loadCol = flatIdx % TILE;

            int gRowA = wgRowStart + loadRow;
            int gColA = loadCol;
            As0[loadRow][loadCol] = (gRowA < M && gColA < K) ? A[gRowA * K + gColA] : 0.0f;

            int gRowB = loadRow;
            int gColB = wgColStart + loadCol;
            Bs0[loadRow][loadCol] = (gRowB < K && gColB < N) ? B[gRowB * N + gColB] : 0.0f;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Main K loop with prefetching
    for (int kt = 0; kt < numKTiles; kt++) {
        int kBaseNext = (kt + 1) * TILE;
        int kBaseCur = kt * TILE;

        if ((kt & 1) == 0) {
            // Even iteration: compute on buffer 0, prefetch to buffer 1
            if (kt + 1 < numKTiles) {
                // Prefetch next tile into buffer 1
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int flatIdx = tid + i * 256;
                    if (flatIdx < TILE * TILE) {
                        int loadRow = flatIdx / TILE;
                        int loadCol = flatIdx % TILE;

                        int gRowA = wgRowStart + loadRow;
                        int gColA = kBaseNext + loadCol;
                        As1[loadRow][loadCol] = (gRowA < M && gColA < K) ? A[gRowA * K + gColA] : 0.0f;

                        int gRowB = kBaseNext + loadRow;
                        int gColB = wgColStart + loadCol;
                        Bs1[loadRow][loadCol] = (gRowB < K && gColB < N) ? B[gRowB * N + gColB] : 0.0f;
                    }
                }
            }

            // Compute on buffer 0
            #pragma unroll
            for (int k = 0; k < TILE; k++) {
                float a0 = As0[tidM * OUT + 0][k];
                float a1 = As0[tidM * OUT + 1][k];
                float b0 = Bs0[k][tidN * OUT + 0];
                float b1 = Bs0[k][tidN * OUT + 1];

                acc00 = fma(a0, b0, acc00);
                acc01 = fma(a0, b1, acc01);
                acc10 = fma(a1, b0, acc10);
                acc11 = fma(a1, b1, acc11);
            }
        } else {
            // Odd iteration: compute on buffer 1, prefetch to buffer 0
            if (kt + 1 < numKTiles) {
                // Prefetch next tile into buffer 0
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int flatIdx = tid + i * 256;
                    if (flatIdx < TILE * TILE) {
                        int loadRow = flatIdx / TILE;
                        int loadCol = flatIdx % TILE;

                        int gRowA = wgRowStart + loadRow;
                        int gColA = kBaseNext + loadCol;
                        As0[loadRow][loadCol] = (gRowA < M && gColA < K) ? A[gRowA * K + gColA] : 0.0f;

                        int gRowB = kBaseNext + loadRow;
                        int gColB = wgColStart + loadCol;
                        Bs0[loadRow][loadCol] = (gRowB < K && gColB < N) ? B[gRowB * N + gColB] : 0.0f;
                    }
                }
            }

            // Compute on buffer 1
            #pragma unroll
            for (int k = 0; k < TILE; k++) {
                float a0 = As1[tidM * OUT + 0][k];
                float a1 = As1[tidM * OUT + 1][k];
                float b0 = Bs1[k][tidN * OUT + 0];
                float b1 = Bs1[k][tidN * OUT + 1];

                acc00 = fma(a0, b0, acc00);
                acc01 = fma(a0, b1, acc01);
                acc10 = fma(a1, b0, acc10);
                acc11 = fma(a1, b1, acc11);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write results (2x2 block)
    if (baseRow < M && baseCol < N) {
        int idx = baseRow * N + baseCol;
        C[idx] = (beta != 0.0f) ? alpha * acc00 + beta * C[idx] : alpha * acc00;
    }
    if (baseRow < M && baseCol + 1 < N) {
        int idx = baseRow * N + baseCol + 1;
        C[idx] = (beta != 0.0f) ? alpha * acc01 + beta * C[idx] : alpha * acc01;
    }
    if (baseRow + 1 < M && baseCol < N) {
        int idx = (baseRow + 1) * N + baseCol;
        C[idx] = (beta != 0.0f) ? alpha * acc10 + beta * C[idx] : alpha * acc10;
    }
    if (baseRow + 1 < M && baseCol + 1 < N) {
        int idx = (baseRow + 1) * N + baseCol + 1;
        C[idx] = (beta != 0.0f) ? alpha * acc11 + beta * C[idx] : alpha * acc11;
    }
}

// ===========================================================================
// GEMM Variation 7: Wide Vector (float4 for both A and B)
// Hypothesis: Use float4 vectorization for both A and B loads to maximize bandwidth
// Wider tiles (32x128) with float4, each thread loads float4 values
// ===========================================================================
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void gemm_wide_vec(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float beta)
{
    // 32x128 tile: 32 rows, 128 columns
    // Work group: 16x16 = 256 threads
    // Each thread computes 2x8 outputs (2 rows, 8 cols via 2 float4)
    const int TILE_ROWS = 32;
    const int TILE_COLS = 128;
    const int WIDE_TILE_K = 16;
    const int OUT_ROWS = 2;
    const int OUT_COLS = 8;

    const int wgRow = get_group_id(0);
    const int wgCol = get_group_id(1);
    const int tidM = get_local_id(0);  // Row within work group (0-15)
    const int tidN = get_local_id(1);  // Column group within work group (0-15)
    const int tid = tidM * 16 + tidN;

    // Shared memory for A tile (32 x 16 + padding)
    __local float As[32][17];

    // Accumulators: 2 rows x 8 cols = 16 values = 2 float4 per row
    float4 acc0_lo = (float4)(0.0f);  // Row 0, cols 0-3
    float4 acc0_hi = (float4)(0.0f);  // Row 0, cols 4-7
    float4 acc1_lo = (float4)(0.0f);  // Row 1, cols 0-3
    float4 acc1_hi = (float4)(0.0f);  // Row 1, cols 4-7

    // Base output positions
    const int outRow = wgRow * TILE_ROWS + tidM * OUT_ROWS;
    const int outCol = wgCol * TILE_COLS + tidN * OUT_COLS;
    const int wgRowStart = wgRow * TILE_ROWS;
    const int wgColStart = wgCol * TILE_COLS;

    int numKTiles = (K + WIDE_TILE_K - 1) / WIDE_TILE_K;

    for (int kt = 0; kt < numKTiles; kt++) {
        int kBase = kt * WIDE_TILE_K;

        // Cooperative loading of A tile (32x16 = 512 elements, 256 threads load 2 each)
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int flatIdx = tid + i * 256;
            if (flatIdx < TILE_ROWS * WIDE_TILE_K) {
                int loadRow = flatIdx / WIDE_TILE_K;
                int loadCol = flatIdx % WIDE_TILE_K;
                int gRow = wgRowStart + loadRow;
                int gCol = kBase + loadCol;
                As[loadRow][loadCol] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0f;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute: for each k, load A from shared, B as float4 from global
        #pragma unroll
        for (int k = 0; k < WIDE_TILE_K; k++) {
            // Load 2 A values (one per output row)
            float a0 = As[tidM * OUT_ROWS + 0][k];
            float a1 = As[tidM * OUT_ROWS + 1][k];

            // Load B as 2 float4 vectors (8 consecutive columns)
            int bRow = kBase + k;
            int bCol = outCol;

            float4 bVec_lo, bVec_hi;

            if (bRow < K && bCol + 7 < N) {
                // Fast path: aligned float4 loads for 8 columns
                bVec_lo = vload4(0, &B[bRow * N + bCol]);
                bVec_hi = vload4(0, &B[bRow * N + bCol + 4]);
            } else if (bRow < K) {
                // Slow path: scalar loads for edge cases
                bVec_lo.x = (bCol + 0 < N) ? B[bRow * N + bCol + 0] : 0.0f;
                bVec_lo.y = (bCol + 1 < N) ? B[bRow * N + bCol + 1] : 0.0f;
                bVec_lo.z = (bCol + 2 < N) ? B[bRow * N + bCol + 2] : 0.0f;
                bVec_lo.w = (bCol + 3 < N) ? B[bRow * N + bCol + 3] : 0.0f;
                bVec_hi.x = (bCol + 4 < N) ? B[bRow * N + bCol + 4] : 0.0f;
                bVec_hi.y = (bCol + 5 < N) ? B[bRow * N + bCol + 5] : 0.0f;
                bVec_hi.z = (bCol + 6 < N) ? B[bRow * N + bCol + 6] : 0.0f;
                bVec_hi.w = (bCol + 7 < N) ? B[bRow * N + bCol + 7] : 0.0f;
            } else {
                bVec_lo = (float4)(0.0f);
                bVec_hi = (float4)(0.0f);
            }

            // Accumulate: 2 rows x 8 cols
            acc0_lo = fma((float4)(a0), bVec_lo, acc0_lo);
            acc0_hi = fma((float4)(a0), bVec_hi, acc0_hi);
            acc1_lo = fma((float4)(a1), bVec_lo, acc1_lo);
            acc1_hi = fma((float4)(a1), bVec_hi, acc1_hi);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write results using vectorized stores
    // Row 0
    if (outRow < M) {
        if (outCol + 3 < N) {
            int idx = outRow * N + outCol;
            float4 result = alpha * acc0_lo;
            if (beta != 0.0f) {
                float4 cOld = vload4(0, &C[idx]);
                result = fma((float4)(beta), cOld, result);
            }
            vstore4(result, 0, &C[idx]);
        } else {
            // Scalar fallback for edge
            if (outCol + 0 < N) C[outRow * N + outCol + 0] = (beta != 0.0f) ? alpha * acc0_lo.x + beta * C[outRow * N + outCol + 0] : alpha * acc0_lo.x;
            if (outCol + 1 < N) C[outRow * N + outCol + 1] = (beta != 0.0f) ? alpha * acc0_lo.y + beta * C[outRow * N + outCol + 1] : alpha * acc0_lo.y;
            if (outCol + 2 < N) C[outRow * N + outCol + 2] = (beta != 0.0f) ? alpha * acc0_lo.z + beta * C[outRow * N + outCol + 2] : alpha * acc0_lo.z;
            if (outCol + 3 < N) C[outRow * N + outCol + 3] = (beta != 0.0f) ? alpha * acc0_lo.w + beta * C[outRow * N + outCol + 3] : alpha * acc0_lo.w;
        }
        if (outCol + 7 < N) {
            int idx = outRow * N + outCol + 4;
            float4 result = alpha * acc0_hi;
            if (beta != 0.0f) {
                float4 cOld = vload4(0, &C[idx]);
                result = fma((float4)(beta), cOld, result);
            }
            vstore4(result, 0, &C[idx]);
        } else {
            if (outCol + 4 < N) C[outRow * N + outCol + 4] = (beta != 0.0f) ? alpha * acc0_hi.x + beta * C[outRow * N + outCol + 4] : alpha * acc0_hi.x;
            if (outCol + 5 < N) C[outRow * N + outCol + 5] = (beta != 0.0f) ? alpha * acc0_hi.y + beta * C[outRow * N + outCol + 5] : alpha * acc0_hi.y;
            if (outCol + 6 < N) C[outRow * N + outCol + 6] = (beta != 0.0f) ? alpha * acc0_hi.z + beta * C[outRow * N + outCol + 6] : alpha * acc0_hi.z;
            if (outCol + 7 < N) C[outRow * N + outCol + 7] = (beta != 0.0f) ? alpha * acc0_hi.w + beta * C[outRow * N + outCol + 7] : alpha * acc0_hi.w;
        }
    }

    // Row 1
    if (outRow + 1 < M) {
        if (outCol + 3 < N) {
            int idx = (outRow + 1) * N + outCol;
            float4 result = alpha * acc1_lo;
            if (beta != 0.0f) {
                float4 cOld = vload4(0, &C[idx]);
                result = fma((float4)(beta), cOld, result);
            }
            vstore4(result, 0, &C[idx]);
        } else {
            if (outCol + 0 < N) C[(outRow + 1) * N + outCol + 0] = (beta != 0.0f) ? alpha * acc1_lo.x + beta * C[(outRow + 1) * N + outCol + 0] : alpha * acc1_lo.x;
            if (outCol + 1 < N) C[(outRow + 1) * N + outCol + 1] = (beta != 0.0f) ? alpha * acc1_lo.y + beta * C[(outRow + 1) * N + outCol + 1] : alpha * acc1_lo.y;
            if (outCol + 2 < N) C[(outRow + 1) * N + outCol + 2] = (beta != 0.0f) ? alpha * acc1_lo.z + beta * C[(outRow + 1) * N + outCol + 2] : alpha * acc1_lo.z;
            if (outCol + 3 < N) C[(outRow + 1) * N + outCol + 3] = (beta != 0.0f) ? alpha * acc1_lo.w + beta * C[(outRow + 1) * N + outCol + 3] : alpha * acc1_lo.w;
        }
        if (outCol + 7 < N) {
            int idx = (outRow + 1) * N + outCol + 4;
            float4 result = alpha * acc1_hi;
            if (beta != 0.0f) {
                float4 cOld = vload4(0, &C[idx]);
                result = fma((float4)(beta), cOld, result);
            }
            vstore4(result, 0, &C[idx]);
        } else {
            if (outCol + 4 < N) C[(outRow + 1) * N + outCol + 4] = (beta != 0.0f) ? alpha * acc1_hi.x + beta * C[(outRow + 1) * N + outCol + 4] : alpha * acc1_hi.x;
            if (outCol + 5 < N) C[(outRow + 1) * N + outCol + 5] = (beta != 0.0f) ? alpha * acc1_hi.y + beta * C[(outRow + 1) * N + outCol + 5] : alpha * acc1_hi.y;
            if (outCol + 6 < N) C[(outRow + 1) * N + outCol + 6] = (beta != 0.0f) ? alpha * acc1_hi.z + beta * C[(outRow + 1) * N + outCol + 6] : alpha * acc1_hi.z;
            if (outCol + 7 < N) C[(outRow + 1) * N + outCol + 7] = (beta != 0.0f) ? alpha * acc1_hi.w + beta * C[(outRow + 1) * N + outCol + 7] : alpha * acc1_hi.w;
        }
    }
}

// ===========================================================================
// gemm_clblast_rdna1: Exact CLBlast parameters for RDNA1 (RX 5500 XT)
// Based on CLBlast tuning database for gfx1010 (RX 5700 XT - closest match)
// Key insight: Use 8x8 work group (64 threads) with 8x8 outputs per thread
// This maps better to RDNA1's wave32 architecture!
// ===========================================================================

// CLBlast parameters for RDNA1:
// MWG=64, NWG=64, KWG=16 (tile sizes)
// MDIMC=8, NDIMC=8 (work group = 64 threads)
// MWI=8, NWI=8 (each thread computes 8x8=64 outputs)
// VWM=2, VWN=2 (vector widths)
// SA=1, SB=1 (use local memory for both)
// STRM=0, STRN=1 (strided B access)
// KWI=2 (K-loop unroll factor)

#define CLBLAST_MWG 64
#define CLBLAST_NWG 64
#define CLBLAST_KWG 16
#define CLBLAST_MDIMC 8
#define CLBLAST_NDIMC 8
#define CLBLAST_MWI (CLBLAST_MWG / CLBLAST_MDIMC)  // = 8
#define CLBLAST_NWI (CLBLAST_NWG / CLBLAST_NDIMC)  // = 8
#define CLBLAST_VWM 2
#define CLBLAST_VWN 2
#define CLBLAST_KWI 2

__kernel __attribute__((reqd_work_group_size(CLBLAST_MDIMC, CLBLAST_NDIMC, 1)))
void gemm_clblast_rdna1(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* C,
    const int M, const int N, const int K,
    const float alpha, const float beta)
{
    // Local memory for A and B tiles (with padding for bank conflict avoidance)
    __local float Als[CLBLAST_KWG][CLBLAST_MWG + 1];  // +1 padding
    __local float Bls[CLBLAST_KWG][CLBLAST_NWG + 1];  // +1 padding

    // Thread indices within work group
    const int tidM = get_local_id(0);  // 0-7
    const int tidN = get_local_id(1);  // 0-7
    const int tid = tidN * CLBLAST_MDIMC + tidM;  // Linear thread ID (0-63)

    // Work group indices (with staggered access for partition camping avoidance)
    const int numGroupsN = (N + CLBLAST_NWG - 1) / CLBLAST_NWG;
    const int flatGroupId = get_group_id(0) + get_num_groups(0) * get_group_id(1);
    const int wgN = flatGroupId % numGroupsN;
    const int wgM = ((flatGroupId / numGroupsN) + wgN) % get_num_groups(0);

    // Global starting positions
    const int wgRowStart = wgM * CLBLAST_MWG;
    const int wgColStart = wgN * CLBLAST_NWG;

    // Register accumulators: 8x8 = 64 outputs per thread
    // Using float2 vectors for VWM=2, VWN=2
    float2 acc[CLBLAST_NWI][CLBLAST_MWI / CLBLAST_VWM];  // [8][4] = 32 float2 = 64 floats

    // Initialize accumulators
    #pragma unroll
    for (int ni = 0; ni < CLBLAST_NWI; ni++) {
        #pragma unroll
        for (int mi = 0; mi < CLBLAST_MWI / CLBLAST_VWM; mi++) {
            acc[ni][mi] = (float2)(0.0f);
        }
    }

    // Main K-loop
    for (int kBase = 0; kBase < K; kBase += CLBLAST_KWG) {

        // Load A tile: 64x16 elements, 64 threads, each thread loads 16 elements
        // Pattern: each thread loads a row of A tile
        #pragma unroll
        for (int loadIter = 0; loadIter < (CLBLAST_MWG * CLBLAST_KWG) / 64; loadIter++) {
            int loadIdx = tid + loadIter * 64;
            int loadRow = loadIdx / CLBLAST_KWG;  // M dimension (0-63)
            int loadCol = loadIdx % CLBLAST_KWG;  // K dimension (0-15)

            int globalRow = wgRowStart + loadRow;
            int globalCol = kBase + loadCol;

            Als[loadCol][loadRow] = (globalRow < M && globalCol < K) ?
                                     A[globalRow * K + globalCol] : 0.0f;
        }

        // Load B tile: 16x64 elements, 64 threads, each thread loads 16 elements
        // Use strided access pattern (STRN=1) for better coalescing
        #pragma unroll
        for (int loadIter = 0; loadIter < (CLBLAST_KWG * CLBLAST_NWG) / 64; loadIter++) {
            int loadIdx = tid + loadIter * 64;
            int loadRow = loadIdx / CLBLAST_NWG;  // K dimension (0-15)
            int loadCol = loadIdx % CLBLAST_NWG;  // N dimension (0-63)

            int globalRow = kBase + loadRow;
            int globalCol = wgColStart + loadCol;

            Bls[loadRow][loadCol] = (globalRow < K && globalCol < N) ?
                                     B[globalRow * N + globalCol] : 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute: each thread processes its 8x8 output tile
        // Unroll K by KWI=2
        #pragma unroll
        for (int k = 0; k < CLBLAST_KWG; k += CLBLAST_KWI) {
            // Load A values from local memory (8 values per thread)
            // Each thread reads from rows: tidM*8 to tidM*8+7
            float2 aReg[CLBLAST_MWI / CLBLAST_VWM];  // [4] float2 = 8 floats
            #pragma unroll
            for (int mi = 0; mi < CLBLAST_MWI / CLBLAST_VWM; mi++) {
                int mIdx = tidM * CLBLAST_MWI + mi * CLBLAST_VWM;
                aReg[mi].x = Als[k][mIdx];
                aReg[mi].y = Als[k][mIdx + 1];
            }

            // Load B values from local memory (8 values per thread)
            // Each thread reads from cols: tidN*8 to tidN*8+7
            float2 bReg[CLBLAST_NWI / CLBLAST_VWN];  // [4] float2 = 8 floats
            #pragma unroll
            for (int ni = 0; ni < CLBLAST_NWI / CLBLAST_VWN; ni++) {
                int nIdx = tidN * CLBLAST_NWI + ni * CLBLAST_VWN;
                bReg[ni].x = Bls[k][nIdx];
                bReg[ni].y = Bls[k][nIdx + 1];
            }

            // Outer product accumulation (K unroll iteration 0)
            #pragma unroll
            for (int ni = 0; ni < CLBLAST_NWI / CLBLAST_VWN; ni++) {
                #pragma unroll
                for (int mi = 0; mi < CLBLAST_MWI / CLBLAST_VWM; mi++) {
                    // acc[ni*2][mi] += aReg[mi] * bReg[ni].x
                    acc[ni * 2][mi] = fma(aReg[mi], (float2)(bReg[ni].x), acc[ni * 2][mi]);
                    // acc[ni*2+1][mi] += aReg[mi] * bReg[ni].y
                    acc[ni * 2 + 1][mi] = fma(aReg[mi], (float2)(bReg[ni].y), acc[ni * 2 + 1][mi]);
                }
            }

            // K unroll iteration 1 (k+1)
            if (k + 1 < CLBLAST_KWG) {
                #pragma unroll
                for (int mi = 0; mi < CLBLAST_MWI / CLBLAST_VWM; mi++) {
                    int mIdx = tidM * CLBLAST_MWI + mi * CLBLAST_VWM;
                    aReg[mi].x = Als[k + 1][mIdx];
                    aReg[mi].y = Als[k + 1][mIdx + 1];
                }

                #pragma unroll
                for (int ni = 0; ni < CLBLAST_NWI / CLBLAST_VWN; ni++) {
                    int nIdx = tidN * CLBLAST_NWI + ni * CLBLAST_VWN;
                    bReg[ni].x = Bls[k + 1][nIdx];
                    bReg[ni].y = Bls[k + 1][nIdx + 1];
                }

                #pragma unroll
                for (int ni = 0; ni < CLBLAST_NWI / CLBLAST_VWN; ni++) {
                    #pragma unroll
                    for (int mi = 0; mi < CLBLAST_MWI / CLBLAST_VWM; mi++) {
                        acc[ni * 2][mi] = fma(aReg[mi], (float2)(bReg[ni].x), acc[ni * 2][mi]);
                        acc[ni * 2 + 1][mi] = fma(aReg[mi], (float2)(bReg[ni].y), acc[ni * 2 + 1][mi]);
                    }
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results: each thread writes 8x8 elements
    int outRowBase = wgRowStart + tidM * CLBLAST_MWI;
    int outColBase = wgColStart + tidN * CLBLAST_NWI;

    #pragma unroll
    for (int ni = 0; ni < CLBLAST_NWI; ni++) {
        int outCol = outColBase + ni;
        if (outCol >= N) continue;

        #pragma unroll
        for (int mi = 0; mi < CLBLAST_MWI / CLBLAST_VWM; mi++) {
            int outRow = outRowBase + mi * CLBLAST_VWM;

            float2 result = alpha * acc[ni][mi];

            // Write with float2 vectorization where possible
            if (outRow + 1 < M) {
                if (beta != 0.0f) {
                    float2 cOld;
                    cOld.x = C[outRow * N + outCol];
                    cOld.y = C[(outRow + 1) * N + outCol];
                    result = fma((float2)(beta), cOld, result);
                }
                C[outRow * N + outCol] = result.x;
                C[(outRow + 1) * N + outCol] = result.y;
            } else if (outRow < M) {
                if (beta != 0.0f) {
                    result.x = fma(beta, C[outRow * N + outCol], result.x);
                }
                C[outRow * N + outCol] = result.x;
            }
        }
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
                "gemm_low_register",
                "gemm_tiled_simple",
                "gemm_small",
                "gemm_vectorized",
                "gemm_batched_persistent",
                "gemm_small_tile",
                "gemm_medium_tile",
                "gemm_coalesced",
                "gemm_vectorized_tile",
                "gemm_kreg4",
                "gemm_prefetch",
                "gemm_wide_vec",
                "gemm_clblast_rdna1"
            };
        }
    }
}
