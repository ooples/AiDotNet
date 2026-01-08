// Copyright (c) AiDotNet. All rights reserved.
// CUDA kernels for spatial transformer operations including TopK selection.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels
{
    /// <summary>
    /// CUDA kernels for spatial transformer and selection operations.
    /// </summary>
    internal static class CudaSpatialTransformerKernels
    {
        public static string GetSource()
        {
            return @"
#include <math.h>

// ===========================================================================
// TOP-K SELECTION KERNEL
// ===========================================================================

// TopK selection using partial sort (efficient for small K)
// Each thread block handles one row of the input
extern ""C"" __global__ void topk(
    const float* input, float* values, int* indices,
    int outerSize, int reduceSize, int k, int sorted)
{
    int row = blockIdx.x;
    if (row >= outerSize) return;

    // Shared memory for top-k candidates
    extern __shared__ float shared[];
    float* topValues = shared;
    int* topIndices = (int*)(shared + k);

    const float* rowData = input + row * reduceSize;

    // Initialize with -infinity
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        topValues[i] = -INFINITY;
        topIndices[i] = -1;
    }
    __syncthreads();

    // Each thread processes a portion of the row
    for (int i = threadIdx.x; i < reduceSize; i += blockDim.x) {
        float val = rowData[i];

        // Check if this value should be in top-k
        // Find the minimum in current top-k
        int minIdx = 0;
        float minVal = topValues[0];
        for (int j = 1; j < k; j++) {
            if (topValues[j] < minVal) {
                minVal = topValues[j];
                minIdx = j;
            }
        }

        // If current value is larger than minimum in top-k, replace it
        if (val > minVal) {
            // Use atomic operations for thread safety
            atomicMax((int*)&topValues[minIdx], __float_as_int(val));
            atomicExch(&topIndices[minIdx], i);
        }
    }
    __syncthreads();

    // Sort the top-k if requested (simple bubble sort for small k)
    if (sorted && threadIdx.x == 0) {
        for (int i = 0; i < k - 1; i++) {
            for (int j = 0; j < k - i - 1; j++) {
                if (topValues[j] < topValues[j + 1]) {
                    float tmpVal = topValues[j];
                    topValues[j] = topValues[j + 1];
                    topValues[j + 1] = tmpVal;

                    int tmpIdx = topIndices[j];
                    topIndices[j] = topIndices[j + 1];
                    topIndices[j + 1] = tmpIdx;
                }
            }
        }
    }
    __syncthreads();

    // Write output
    float* outValues = values + row * k;
    int* outIndices = indices + row * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        outValues[i] = topValues[i];
        outIndices[i] = topIndices[i];
    }
}

// Optimized TopK for small K using warp-level operations
extern ""C"" __global__ void topk_small(
    const float* input, float* values, int* indices,
    int outerSize, int reduceSize, int k)
{
    int row = blockIdx.x;
    if (row >= outerSize) return;

    const float* rowData = input + row * reduceSize;
    float* outValues = values + row * k;
    int* outIndices = indices + row * k;

    // Local top-k storage per thread
    float localTop[8];  // Max k=8 per thread
    int localIdx[8];

    for (int i = 0; i < k && i < 8; i++) {
        localTop[i] = -INFINITY;
        localIdx[i] = -1;
    }

    // Scan through data
    for (int i = threadIdx.x; i < reduceSize; i += blockDim.x) {
        float val = rowData[i];

        // Insert into local top-k
        for (int j = 0; j < k && j < 8; j++) {
            if (val > localTop[j]) {
                // Shift down
                for (int m = k - 1; m > j && m < 8; m--) {
                    localTop[m] = localTop[m - 1];
                    localIdx[m] = localIdx[m - 1];
                }
                localTop[j] = val;
                localIdx[j] = i;
                break;
            }
        }
    }

    // Reduce across threads (simplified - first thread writes)
    __shared__ float sharedTop[256 * 8];
    __shared__ int sharedIdx[256 * 8];

    int tid = threadIdx.x;
    for (int i = 0; i < k && i < 8; i++) {
        sharedTop[tid * 8 + i] = localTop[i];
        sharedIdx[tid * 8 + i] = localIdx[i];
    }
    __syncthreads();

    // Thread 0 merges all results
    if (tid == 0) {
        float finalTop[8];
        int finalIdx[8];
        for (int i = 0; i < k && i < 8; i++) {
            finalTop[i] = -INFINITY;
            finalIdx[i] = -1;
        }

        for (int t = 0; t < blockDim.x; t++) {
            for (int i = 0; i < k && i < 8; i++) {
                float val = sharedTop[t * 8 + i];
                int idx = sharedIdx[t * 8 + i];

                // Insert into final top-k
                for (int j = 0; j < k && j < 8; j++) {
                    if (val > finalTop[j]) {
                        for (int m = k - 1; m > j && m < 8; m--) {
                            finalTop[m] = finalTop[m - 1];
                            finalIdx[m] = finalIdx[m - 1];
                        }
                        finalTop[j] = val;
                        finalIdx[j] = idx;
                        break;
                    }
                }
            }
        }

        // Write output
        for (int i = 0; i < k && i < 8; i++) {
            outValues[i] = finalTop[i];
            outIndices[i] = finalIdx[i];
        }
    }
}

// ===========================================================================
// AFFINE GRID GENERATION KERNEL
// ===========================================================================

// Generate affine sampling grid
// theta: [batch, 2, 3] affine transformation matrices
// grid: [batch, outH, outW, 2] output sampling coordinates
extern ""C"" __global__ void affine_grid(
    const float* theta, float* grid,
    int batch, int outHeight, int outWidth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (x >= outWidth || y >= outHeight || b >= batch) return;

    // Normalized coordinates in [-1, 1]
    float nx = 2.0f * x / (outWidth - 1) - 1.0f;
    float ny = 2.0f * y / (outHeight - 1) - 1.0f;

    // Handle edge case for dimension 1
    if (outWidth == 1) nx = 0.0f;
    if (outHeight == 1) ny = 0.0f;

    // Affine transformation: [x', y'] = theta * [x, y, 1]^T
    const float* t = theta + b * 6;
    float outX = t[0] * nx + t[1] * ny + t[2];
    float outY = t[3] * nx + t[4] * ny + t[5];

    // Output grid location
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
extern ""C"" __global__ void grid_sample(
    const float* input, const float* grid, float* output,
    int batch, int channels, int inHeight, int inWidth,
    int outHeight, int outWidth, int paddingMode, int alignCorners)
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

    // Helper function to get pixel value with padding
    #define GET_PIXEL(px, py) \
        ((px >= 0 && px < inWidth && py >= 0 && py < inHeight) ? \
            input[((b * channels + c) * inHeight + py) * inWidth + px] : \
            (paddingMode == 1 ? input[((b * channels + c) * inHeight + \
                max(0, min(py, inHeight - 1))) * inWidth + max(0, min(px, inWidth - 1))] : 0.0f))

    // Bilinear interpolation
    float v00 = GET_PIXEL(ix0, iy0);
    float v01 = GET_PIXEL(ix1, iy0);
    float v10 = GET_PIXEL(ix0, iy1);
    float v11 = GET_PIXEL(ix1, iy1);

    float result = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);

    #undef GET_PIXEL

    // Write output
    int outIdx = ((b * channels + c) * outHeight + y) * outWidth + x;
    output[outIdx] = result;
}

// ===========================================================================
// GRID SAMPLE BACKWARD KERNEL
// ===========================================================================

extern ""C"" __global__ void grid_sample_backward(
    const float* gradOutput, const float* input, const float* grid,
    float* gradInput, float* gradGrid,
    int batch, int channels, int inHeight, int inWidth,
    int outHeight, int outWidth, int paddingMode, int alignCorners)
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

        // Get input values
        #define GET_PIXEL(px, py) \
            ((px >= 0 && px < inWidth && py >= 0 && py < inHeight) ? \
                input[((b * channels + c) * inHeight + py) * inWidth + px] : 0.0f)

        float v00 = GET_PIXEL(ix0, iy0);
        float v01 = GET_PIXEL(ix1, iy0);
        float v10 = GET_PIXEL(ix0, iy1);
        float v11 = GET_PIXEL(ix1, iy1);

        #undef GET_PIXEL

        // Gradient with respect to grid
        // d(result)/d(ix) = wy0 * (v01 - v00) + wy1 * (v11 - v10)
        // d(result)/d(iy) = wx0 * (v10 - v00) + wx1 * (v11 - v01)
        gradGridX += go * (wy0 * (v01 - v00) + wy1 * (v11 - v10)) * gradMultX;
        gradGridY += go * (wx0 * (v10 - v00) + wx1 * (v11 - v01)) * gradMultY;

        // Gradient with respect to input (scatter add)
        if (ix0 >= 0 && ix0 < inWidth && iy0 >= 0 && iy0 < inHeight)
            atomicAdd(&gradInput[((b * channels + c) * inHeight + iy0) * inWidth + ix0], go * wy0 * wx0);
        if (ix1 >= 0 && ix1 < inWidth && iy0 >= 0 && iy0 < inHeight)
            atomicAdd(&gradInput[((b * channels + c) * inHeight + iy0) * inWidth + ix1], go * wy0 * wx1);
        if (ix0 >= 0 && ix0 < inWidth && iy1 >= 0 && iy1 < inHeight)
            atomicAdd(&gradInput[((b * channels + c) * inHeight + iy1) * inWidth + ix0], go * wy1 * wx0);
        if (ix1 >= 0 && ix1 < inWidth && iy1 >= 0 && iy1 < inHeight)
            atomicAdd(&gradInput[((b * channels + c) * inHeight + iy1) * inWidth + ix1], go * wy1 * wx1);
    }

    // Write grid gradient
    gradGrid[gridIdx] = gradGridX;
    gradGrid[gridIdx + 1] = gradGridY;
}
";
        }

        public static string[] GetKernelNames()
        {
            return new[]
            {
                "topk",
                "topk_small",
                "affine_grid",
                "grid_sample",
                "grid_sample_backward"
            };
        }
    }
}
