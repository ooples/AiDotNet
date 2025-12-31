// Copyright (c) AiDotNet. All rights reserved.
// Padding and copy kernels for packed/indirect GEMM paths.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    internal static class PackingKernels
    {
        public static string GetSource()
        {
            return @"
// ===========================================================================
// PACKING / PAD COPY KERNELS
// ===========================================================================

// Copies src into dst with zero-padding for out-of-bounds regions.
// Layout: row-major with srcCols/dstCols as the leading dimensions.
__kernel void pad_copy(
    __global const float* src,
    __global float* dst,
    const int srcRows,
    const int srcCols,
    const int dstRows,
    const int dstCols)
{
    const int row = get_global_id(0);
    const int col = get_global_id(1);

    if (row >= dstRows || col >= dstCols) return;

    if (row < srcRows && col < srcCols) {
        dst[row * dstCols + col] = src[row * srcCols + col];
    } else {
        dst[row * dstCols + col] = 0.0f;
    }
}

// Copies a submatrix from src to dst (row-major, configurable strides).
__kernel void copy_submatrix(
    __global const float* src,
    __global float* dst,
    const int rows,
    const int cols,
    const int srcStride,
    const int dstStride)
{
    const int row = get_global_id(0);
    const int col = get_global_id(1);

    if (row >= rows || col >= cols) return;

    dst[row * dstStride + col] = src[row * srcStride + col];
}
";
        }

        public static string[] GetKernelNames()
        {
            return new[]
            {
                "pad_copy",
                "copy_submatrix"
            };
        }
    }
}
