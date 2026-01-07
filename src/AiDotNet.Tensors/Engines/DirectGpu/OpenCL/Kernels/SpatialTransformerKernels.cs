// Copyright (c) AiDotNet. All rights reserved.
// OpenCL kernels for spatial transformer operations including TopK selection.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    /// <summary>
    /// OpenCL kernels for spatial transformer and selection operations.
    /// </summary>
    internal static class SpatialTransformerKernels
    {
        public static string GetSource()
        {
            return @"
// ===========================================================================
// TOP-K SELECTION KERNEL
// ===========================================================================

// TopK selection - each work group handles one row
__kernel void topk(
    __global const float* input,
    __global float* values,
    __global int* indices,
    const int outerSize,
    const int reduceSize,
    const int k,
    const int sorted,
    __local float* localTop,
    __local int* localIdx)
{
    int row = get_group_id(0);
    if (row >= outerSize) return;

    int lid = get_local_id(0);
    int localSize = get_local_size(0);

    __global const float* rowData = input + row * reduceSize;

    // Initialize local top-k (only first k threads)
    if (lid < k) {
        localTop[lid] = -INFINITY;
        localIdx[lid] = -1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Each thread scans its portion and maintains local top-k
    float myTop[8];
    int myIdx[8];
    for (int i = 0; i < k && i < 8; i++) {
        myTop[i] = -INFINITY;
        myIdx[i] = -1;
    }

    // Scan through the row
    for (int i = lid; i < reduceSize; i += localSize) {
        float val = rowData[i];

        // Insert into thread-local top-k
        for (int j = 0; j < k && j < 8; j++) {
            if (val > myTop[j]) {
                // Shift down
                for (int m = min(k, 8) - 1; m > j; m--) {
                    myTop[m] = myTop[m - 1];
                    myIdx[m] = myIdx[m - 1];
                }
                myTop[j] = val;
                myIdx[j] = i;
                break;
            }
        }
    }

    // Write thread-local results to local memory for reduction
    barrier(CLK_LOCAL_MEM_FENCE);

    // Thread 0 merges all results
    if (lid == 0) {
        float finalTop[8];
        int finalIdx[8];
        for (int i = 0; i < k && i < 8; i++) {
            finalTop[i] = -INFINITY;
            finalIdx[i] = -1;
        }

        // Simple merge - collect all candidates from all threads
        // (In practice, would need shared memory staging)
        for (int i = 0; i < reduceSize; i++) {
            float val = rowData[i];
            for (int j = 0; j < k && j < 8; j++) {
                if (val > finalTop[j]) {
                    for (int m = min(k, 8) - 1; m > j; m--) {
                        finalTop[m] = finalTop[m - 1];
                        finalIdx[m] = finalIdx[m - 1];
                    }
                    finalTop[j] = val;
                    finalIdx[j] = i;
                    break;
                }
            }
        }

        // Write output
        __global float* outValues = values + row * k;
        __global int* outIndices = indices + row * k;
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
__kernel void affine_grid(
    __global const float* theta,
    __global float* grid,
    const int batch,
    const int outHeight,
    const int outWidth)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int b = get_global_id(2);

    if (x >= outWidth || y >= outHeight || b >= batch) return;

    // Normalized coordinates in [-1, 1]
    float nx = outWidth > 1 ? 2.0f * x / (outWidth - 1) - 1.0f : 0.0f;
    float ny = outHeight > 1 ? 2.0f * y / (outHeight - 1) - 1.0f : 0.0f;

    // Affine transformation: [x', y'] = theta * [x, y, 1]^T
    __global const float* t = theta + b * 6;
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
__kernel void grid_sample(
    __global const float* input,
    __global const float* grid,
    __global float* output,
    const int batch,
    const int channels,
    const int inHeight,
    const int inWidth,
    const int outHeight,
    const int outWidth,
    const int paddingMode,
    const int alignCorners)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int bc = get_global_id(2);
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
    int ix0 = (int)floor(ix);
    int iy0 = (int)floor(iy);
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;

    // Interpolation weights
    float wx1 = ix - ix0;
    float wy1 = iy - iy0;
    float wx0 = 1.0f - wx1;
    float wy0 = 1.0f - wy1;

    // Get pixel values with padding
    float v00 = 0.0f, v01 = 0.0f, v10 = 0.0f, v11 = 0.0f;

    if (ix0 >= 0 && ix0 < inWidth && iy0 >= 0 && iy0 < inHeight)
        v00 = input[((b * channels + c) * inHeight + iy0) * inWidth + ix0];
    else if (paddingMode == 1 && inWidth > 0 && inHeight > 0)
        v00 = input[((b * channels + c) * inHeight + clamp(iy0, 0, inHeight-1)) * inWidth + clamp(ix0, 0, inWidth-1)];

    if (ix1 >= 0 && ix1 < inWidth && iy0 >= 0 && iy0 < inHeight)
        v01 = input[((b * channels + c) * inHeight + iy0) * inWidth + ix1];
    else if (paddingMode == 1 && inWidth > 0 && inHeight > 0)
        v01 = input[((b * channels + c) * inHeight + clamp(iy0, 0, inHeight-1)) * inWidth + clamp(ix1, 0, inWidth-1)];

    if (ix0 >= 0 && ix0 < inWidth && iy1 >= 0 && iy1 < inHeight)
        v10 = input[((b * channels + c) * inHeight + iy1) * inWidth + ix0];
    else if (paddingMode == 1 && inWidth > 0 && inHeight > 0)
        v10 = input[((b * channels + c) * inHeight + clamp(iy1, 0, inHeight-1)) * inWidth + clamp(ix0, 0, inWidth-1)];

    if (ix1 >= 0 && ix1 < inWidth && iy1 >= 0 && iy1 < inHeight)
        v11 = input[((b * channels + c) * inHeight + iy1) * inWidth + ix1];
    else if (paddingMode == 1 && inWidth > 0 && inHeight > 0)
        v11 = input[((b * channels + c) * inHeight + clamp(iy1, 0, inHeight-1)) * inWidth + clamp(ix1, 0, inWidth-1)];

    // Bilinear interpolation
    float result = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);

    // Write output
    int outIdx = ((b * channels + c) * outHeight + y) * outWidth + x;
    output[outIdx] = result;
}

// ===========================================================================
// GRID SAMPLE BACKWARD KERNEL
// ===========================================================================

__kernel void grid_sample_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global const float* grid,
    __global float* gradInput,
    __global float* gradGrid,
    const int batch,
    const int channels,
    const int inHeight,
    const int inWidth,
    const int outHeight,
    const int outWidth,
    const int paddingMode,
    const int alignCorners)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int b = get_global_id(2);

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

    int ix0 = (int)floor(ix);
    int iy0 = (int)floor(iy);
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
        float v00 = 0.0f, v01 = 0.0f, v10 = 0.0f, v11 = 0.0f;
        if (ix0 >= 0 && ix0 < inWidth && iy0 >= 0 && iy0 < inHeight)
            v00 = input[((b * channels + c) * inHeight + iy0) * inWidth + ix0];
        if (ix1 >= 0 && ix1 < inWidth && iy0 >= 0 && iy0 < inHeight)
            v01 = input[((b * channels + c) * inHeight + iy0) * inWidth + ix1];
        if (ix0 >= 0 && ix0 < inWidth && iy1 >= 0 && iy1 < inHeight)
            v10 = input[((b * channels + c) * inHeight + iy1) * inWidth + ix0];
        if (ix1 >= 0 && ix1 < inWidth && iy1 >= 0 && iy1 < inHeight)
            v11 = input[((b * channels + c) * inHeight + iy1) * inWidth + ix1];

        // Gradient with respect to grid
        gradGridX += go * (wy0 * (v01 - v00) + wy1 * (v11 - v10)) * gradMultX;
        gradGridY += go * (wx0 * (v10 - v00) + wx1 * (v11 - v01)) * gradMultY;

        // Gradient with respect to input (atomic add)
        if (ix0 >= 0 && ix0 < inWidth && iy0 >= 0 && iy0 < inHeight) {
            int idx = ((b * channels + c) * inHeight + iy0) * inWidth + ix0;
            // Note: OpenCL 1.x doesn't have atomic_add for float, need extension or emulation
            // For now, we'll use a simple add (not thread-safe for overlapping writes)
            gradInput[idx] += go * wy0 * wx0;
        }
        if (ix1 >= 0 && ix1 < inWidth && iy0 >= 0 && iy0 < inHeight) {
            int idx = ((b * channels + c) * inHeight + iy0) * inWidth + ix1;
            gradInput[idx] += go * wy0 * wx1;
        }
        if (ix0 >= 0 && ix0 < inWidth && iy1 >= 0 && iy1 < inHeight) {
            int idx = ((b * channels + c) * inHeight + iy1) * inWidth + ix0;
            gradInput[idx] += go * wy1 * wx0;
        }
        if (ix1 >= 0 && ix1 < inWidth && iy1 >= 0 && iy1 < inHeight) {
            int idx = ((b * channels + c) * inHeight + iy1) * inWidth + ix1;
            gradInput[idx] += go * wy1 * wx1;
        }
    }

    // Write grid gradient
    gradGrid[gridIdx] = gradGridX;
    gradGrid[gridIdx + 1] = gradGridY;
}
";
        }
    }
}
