// Copyright (c) AiDotNet. All rights reserved.
// HIP kernels for spatial transformer operations including TopK selection.
// HIP is source-compatible with CUDA device code.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// HIP kernels for spatial transformer and selection operations.
/// Includes TopK selection, affine grid generation, and bilinear grid sampling.
/// </summary>
internal static class HipSpatialTransformerKernels
{
    public static string[] GetKernelNames()
    {
        return new[]
        {
            "topk",
            "affine_grid",
            "grid_sample",
            "grid_sample_backward"
        };
    }

    public static string GetSource()
    {
        return @"
// HIP RTC Compatibility - device intrinsics are built-in
#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

// ===========================================================================
// ATOMIC FLOAT ADD HELPER
// ===========================================================================

// Emulate atomic float add using CAS loop (for compatibility with older HIP/CUDA)
__device__ inline float atomicAddFloat(float* addr, float val) {
    unsigned int* addr_as_ui = (unsigned int*)addr;
    unsigned int old = *addr_as_ui;
    unsigned int assumed;
    do {
        assumed = old;
        float old_f = __int_as_float(assumed);
        float new_f = old_f + val;
        old = atomicCAS(addr_as_ui, assumed, __float_as_int(new_f));
    } while (assumed != old);
    return __int_as_float(old);
}

// ===========================================================================
// TOP-K SELECTION KERNEL
// ===========================================================================

// TopK selection - each work group handles one row
// Uses heap-based selection for efficient O(n log k) complexity
extern ""C"" __global__ void topk(
    const float* input,
    float* values,
    int* indices,
    int outerSize,
    int reduceSize,
    int k,
    int sorted)
{
    int row = blockIdx.x;
    if (row >= outerSize) return;

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    const float* rowData = input + row * reduceSize;
    float* outValues = values + row * k;
    int* outIndices = indices + row * k;

    // Shared memory for parallel reduction
    extern __shared__ char sharedMem[];
    float* sharedValues = (float*)sharedMem;
    int* sharedIndices = (int*)(sharedMem + k * sizeof(float));

    // Initialize shared top-k with -INFINITY
    if (tid < k) {
        sharedValues[tid] = -INFINITY;
        sharedIndices[tid] = -1;
    }
    __syncthreads();

    // Thread-local top-k (up to 8 elements to avoid register pressure)
    float localTop[8];
    int localIdx[8];
    int localK = (k < 8) ? k : 8;
    for (int i = 0; i < localK; i++) {
        localTop[i] = -INFINITY;
        localIdx[i] = -1;
    }

    // Scan through the row - each thread handles strided elements
    for (int i = tid; i < reduceSize; i += blockSize) {
        float val = rowData[i];

        // Check if val should be in thread-local top-k
        if (val > localTop[localK - 1]) {
            // Insert val into sorted local array
            int insertPos = localK - 1;
            while (insertPos > 0 && val > localTop[insertPos - 1]) {
                localTop[insertPos] = localTop[insertPos - 1];
                localIdx[insertPos] = localIdx[insertPos - 1];
                insertPos--;
            }
            localTop[insertPos] = val;
            localIdx[insertPos] = i;
        }
    }
    __syncthreads();

    // Thread 0 performs final merge of all thread-local results
    // For simplicity, we re-scan the data - this is still efficient for moderate k
    if (tid == 0) {
        float finalTop[32];  // Support k up to 32 (can be extended)
        int finalIdx[32];
        int actualK = (k < 32) ? k : 32;

        for (int i = 0; i < actualK; i++) {
            finalTop[i] = -INFINITY;
            finalIdx[i] = -1;
        }

        // Scan entire row for final results
        for (int i = 0; i < reduceSize; i++) {
            float val = rowData[i];
            if (val > finalTop[actualK - 1]) {
                int insertPos = actualK - 1;
                while (insertPos > 0 && val > finalTop[insertPos - 1]) {
                    finalTop[insertPos] = finalTop[insertPos - 1];
                    finalIdx[insertPos] = finalIdx[insertPos - 1];
                    insertPos--;
                }
                finalTop[insertPos] = val;
                finalIdx[insertPos] = i;
            }
        }

        // Write output (already sorted by construction)
        for (int i = 0; i < k && i < 32; i++) {
            outValues[i] = finalTop[i];
            outIndices[i] = finalIdx[i];
        }
    }
}

// ===========================================================================
// AFFINE GRID GENERATION KERNEL
// ===========================================================================

// Generate affine sampling grid from transformation matrices
// theta: [batch, 2, 3] affine transformation matrices (row-major)
// grid: [batch, outH, outW, 2] output sampling coordinates in [-1, 1]
extern ""C"" __global__ void affine_grid(
    const float* theta,
    float* grid,
    int batch,
    int outHeight,
    int outWidth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (x >= outWidth || y >= outHeight || b >= batch) return;

    // Normalized coordinates in [-1, 1]
    float nx = outWidth > 1 ? 2.0f * x / (outWidth - 1) - 1.0f : 0.0f;
    float ny = outHeight > 1 ? 2.0f * y / (outHeight - 1) - 1.0f : 0.0f;

    // Affine transformation: [x', y'] = theta * [x, y, 1]^T
    // theta is [2, 3] per batch: [[a00, a01, a02], [a10, a11, a12]]
    const float* t = theta + b * 6;
    float outX = t[0] * nx + t[1] * ny + t[2];
    float outY = t[3] * nx + t[4] * ny + t[5];

    // Output grid location [batch, outH, outW, 2]
    int gridIdx = ((b * outHeight + y) * outWidth + x) * 2;
    grid[gridIdx] = outX;
    grid[gridIdx + 1] = outY;
}

// ===========================================================================
// GRID SAMPLE KERNEL (BILINEAR INTERPOLATION)
// ===========================================================================

// Sample from input using grid with bilinear interpolation
// input: [batch, channels, inH, inW] NCHW format
// grid: [batch, outH, outW, 2] sampling coordinates in [-1, 1]
// output: [batch, channels, outH, outW]
// paddingMode: 0 = zeros, 1 = border
// alignCorners: if true, treat [-1,1] as exact corner coordinates
extern ""C"" __global__ void grid_sample(
    const float* input,
    const float* grid,
    float* output,
    int batch,
    int channels,
    int inHeight,
    int inWidth,
    int outHeight,
    int outWidth,
    int paddingMode,
    int alignCorners)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int bc = blockIdx.z;
    int b = bc / channels;
    int c = bc % channels;

    if (x >= outWidth || y >= outHeight || b >= batch) return;

    // Get sampling location from grid
    int gridIdx = ((b * outHeight + y) * outWidth + x) * 2;
    float gx = grid[gridIdx];
    float gy = grid[gridIdx + 1];

    // Convert from [-1, 1] to pixel coordinates
    float ix, iy;
    if (alignCorners) {
        ix = (gx + 1.0f) * 0.5f * (inWidth - 1);
        iy = (gy + 1.0f) * 0.5f * (inHeight - 1);
    } else {
        ix = ((gx + 1.0f) * inWidth - 1.0f) * 0.5f;
        iy = ((gy + 1.0f) * inHeight - 1.0f) * 0.5f;
    }

    // Get the four nearest pixel coordinates
    int ix0 = (int)floorf(ix);
    int iy0 = (int)floorf(iy);
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;

    // Interpolation weights
    float wx1 = ix - ix0;
    float wy1 = iy - iy0;
    float wx0 = 1.0f - wx1;
    float wy0 = 1.0f - wy1;

    // Inline clamp function
    #define CLAMP(val, minVal, maxVal) ((val) < (minVal) ? (minVal) : ((val) > (maxVal) ? (maxVal) : (val)))

    // Get pixel values with padding
    float v00 = 0.0f, v01 = 0.0f, v10 = 0.0f, v11 = 0.0f;
    int inHmax = inHeight - 1;
    int inWmax = inWidth - 1;

    if (ix0 >= 0 && ix0 < inWidth && iy0 >= 0 && iy0 < inHeight)
        v00 = input[((b * channels + c) * inHeight + iy0) * inWidth + ix0];
    else if (paddingMode == 1 && inWidth > 0 && inHeight > 0)
        v00 = input[((b * channels + c) * inHeight + CLAMP(iy0, 0, inHmax)) * inWidth + CLAMP(ix0, 0, inWmax)];

    if (ix1 >= 0 && ix1 < inWidth && iy0 >= 0 && iy0 < inHeight)
        v01 = input[((b * channels + c) * inHeight + iy0) * inWidth + ix1];
    else if (paddingMode == 1 && inWidth > 0 && inHeight > 0)
        v01 = input[((b * channels + c) * inHeight + CLAMP(iy0, 0, inHmax)) * inWidth + CLAMP(ix1, 0, inWmax)];

    if (ix0 >= 0 && ix0 < inWidth && iy1 >= 0 && iy1 < inHeight)
        v10 = input[((b * channels + c) * inHeight + iy1) * inWidth + ix0];
    else if (paddingMode == 1 && inWidth > 0 && inHeight > 0)
        v10 = input[((b * channels + c) * inHeight + CLAMP(iy1, 0, inHmax)) * inWidth + CLAMP(ix0, 0, inWmax)];

    if (ix1 >= 0 && ix1 < inWidth && iy1 >= 0 && iy1 < inHeight)
        v11 = input[((b * channels + c) * inHeight + iy1) * inWidth + ix1];
    else if (paddingMode == 1 && inWidth > 0 && inHeight > 0)
        v11 = input[((b * channels + c) * inHeight + CLAMP(iy1, 0, inHmax)) * inWidth + CLAMP(ix1, 0, inWmax)];

    #undef CLAMP

    // Bilinear interpolation
    float result = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);

    // Write output
    int outIdx = ((b * channels + c) * outHeight + y) * outWidth + x;
    output[outIdx] = result;
}

// ===========================================================================
// GRID SAMPLE BACKWARD KERNEL
// ===========================================================================

// Backward pass for grid sampling - computes gradients for input and grid
extern ""C"" __global__ void grid_sample_backward(
    const float* gradOutput,
    const float* input,
    const float* grid,
    float* gradInput,
    float* gradGrid,
    int batch,
    int channels,
    int inHeight,
    int inWidth,
    int outHeight,
    int outWidth,
    int paddingMode,
    int alignCorners)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (x >= outWidth || y >= outHeight || b >= batch) return;

    // Get sampling location from grid
    int gridIdx = ((b * outHeight + y) * outWidth + x) * 2;
    float gx = grid[gridIdx];
    float gy = grid[gridIdx + 1];

    // Convert from [-1, 1] to pixel coordinates
    float ix, iy;
    float gradMultX, gradMultY;
    if (alignCorners) {
        ix = (gx + 1.0f) * 0.5f * (inWidth - 1);
        iy = (gy + 1.0f) * 0.5f * (inHeight - 1);
        gradMultX = 0.5f * (inWidth - 1);
        gradMultY = 0.5f * (inHeight - 1);
    } else {
        ix = ((gx + 1.0f) * inWidth - 1.0f) * 0.5f;
        iy = ((gy + 1.0f) * inHeight - 1.0f) * 0.5f;
        gradMultX = 0.5f * inWidth;
        gradMultY = 0.5f * inHeight;
    }

    int ix0 = (int)floorf(ix);
    int iy0 = (int)floorf(iy);
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;

    float wx1 = ix - ix0;
    float wy1 = iy - iy0;
    float wx0 = 1.0f - wx1;
    float wy0 = 1.0f - wy1;

    float gradGridX = 0.0f;
    float gradGridY = 0.0f;

    // Accumulate gradients for each channel
    for (int c = 0; c < channels; c++) {
        int outIdx = ((b * channels + c) * outHeight + y) * outWidth + x;
        float go = gradOutput[outIdx];

        // Get input values for gradient computation
        float v00 = 0.0f, v01 = 0.0f, v10 = 0.0f, v11 = 0.0f;
        if (ix0 >= 0 && ix0 < inWidth && iy0 >= 0 && iy0 < inHeight)
            v00 = input[((b * channels + c) * inHeight + iy0) * inWidth + ix0];
        if (ix1 >= 0 && ix1 < inWidth && iy0 >= 0 && iy0 < inHeight)
            v01 = input[((b * channels + c) * inHeight + iy0) * inWidth + ix1];
        if (ix0 >= 0 && ix0 < inWidth && iy1 >= 0 && iy1 < inHeight)
            v10 = input[((b * channels + c) * inHeight + iy1) * inWidth + ix0];
        if (ix1 >= 0 && ix1 < inWidth && iy1 >= 0 && iy1 < inHeight)
            v11 = input[((b * channels + c) * inHeight + iy1) * inWidth + ix1];

        // Gradient with respect to grid coordinates
        gradGridX += go * (wy0 * (v01 - v00) + wy1 * (v11 - v10)) * gradMultX;
        gradGridY += go * (wx0 * (v10 - v00) + wx1 * (v11 - v01)) * gradMultY;

        // Gradient with respect to input (atomic add for overlapping writes)
        if (ix0 >= 0 && ix0 < inWidth && iy0 >= 0 && iy0 < inHeight) {
            int idx = ((b * channels + c) * inHeight + iy0) * inWidth + ix0;
            atomicAddFloat(&gradInput[idx], go * wy0 * wx0);
        }
        if (ix1 >= 0 && ix1 < inWidth && iy0 >= 0 && iy0 < inHeight) {
            int idx = ((b * channels + c) * inHeight + iy0) * inWidth + ix1;
            atomicAddFloat(&gradInput[idx], go * wy0 * wx1);
        }
        if (ix0 >= 0 && ix0 < inWidth && iy1 >= 0 && iy1 < inHeight) {
            int idx = ((b * channels + c) * inHeight + iy1) * inWidth + ix0;
            atomicAddFloat(&gradInput[idx], go * wy1 * wx0);
        }
        if (ix1 >= 0 && ix1 < inWidth && iy1 >= 0 && iy1 < inHeight) {
            int idx = ((b * channels + c) * inHeight + iy1) * inWidth + ix1;
            atomicAddFloat(&gradInput[idx], go * wy1 * wx1);
        }
    }

    // Write grid gradient
    gradGrid[gridIdx] = gradGridX;
    gradGrid[gridIdx + 1] = gradGridY;
}
";
    }
}
