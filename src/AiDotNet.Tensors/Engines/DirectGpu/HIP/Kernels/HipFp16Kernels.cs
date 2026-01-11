// Copyright (c) AiDotNet. All rights reserved.
// HIP kernels for FP16 (half-precision) conversion operations.
// HIP is source-compatible with CUDA for FP16 intrinsics.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// HIP kernels for FP16 (half-precision) conversion operations.
/// These kernels convert between FP32 (float) and FP16 (half) precision.
/// </summary>
public static class HipFp16Kernels
{
    public static string GetSource()
    {
        // HIP provides FP16 support via hip_fp16.h which is automatically available in hiprtc
        // The __half type and conversion functions are available when targeting AMD GPUs with FP16 support
        return @"
#include <hip/hip_fp16.h>

// ============================================================================
// FP16 CONVERSION KERNELS
// ============================================================================

// Convert FP32 (float) array to FP16 (half) array
// input: float array of size 'size'
// output: half array of size 'size' (stored as unsigned short for compatibility)
extern ""C"" __global__ void convert_fp32_to_fp16(
    const float* input, unsigned short* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    __half h = __float2half(input[idx]);
    output[idx] = *reinterpret_cast<unsigned short*>(&h);
}

// Convert FP16 (half) array to FP32 (float) array
// input: half array of size 'size' (stored as unsigned short for compatibility)
// output: float array of size 'size'
extern ""C"" __global__ void convert_fp16_to_fp32(
    const unsigned short* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    __half h = *reinterpret_cast<const __half*>(&input[idx]);
    output[idx] = __half2float(h);
}

// Convert FP32 to FP16 with rounding mode control
// roundMode: 0 = round to nearest even (default), 1 = round toward zero, 2 = round down, 3 = round up
extern ""C"" __global__ void convert_fp32_to_fp16_rounding(
    const float* input, unsigned short* output, int size, int roundMode)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    __half h;
    switch (roundMode) {
        case 1: h = __float2half_rz(input[idx]); break;  // Round toward zero
        case 2: h = __float2half_rd(input[idx]); break;  // Round down
        case 3: h = __float2half_ru(input[idx]); break;  // Round up
        default: h = __float2half_rn(input[idx]); break; // Round to nearest even
    }
    output[idx] = *reinterpret_cast<unsigned short*>(&h);
}

// Vectorized FP32 to FP16 conversion (processes 2 elements per thread for better performance)
extern ""C"" __global__ void convert_fp32_to_fp16_vec2(
    const float2* input, unsigned int* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numPairs = size / 2;
    if (idx >= numPairs) return;

    float2 f2 = input[idx];
    __half2 h2 = __floats2half2_rn(f2.x, f2.y);
    output[idx] = *reinterpret_cast<unsigned int*>(&h2);
}

// Vectorized FP16 to FP32 conversion (processes 2 elements per thread for better performance)
extern ""C"" __global__ void convert_fp16_to_fp32_vec2(
    const unsigned int* input, float2* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numPairs = size / 2;
    if (idx >= numPairs) return;

    __half2 h2 = *reinterpret_cast<const __half2*>(&input[idx]);
    float2 f2 = __half22float2(h2);
    output[idx] = f2;
}
";
    }

    /// <summary>
    /// Gets the list of kernel names for compilation.
    /// </summary>
    public static string[] GetKernelNames()
    {
        return new[]
        {
            "convert_fp32_to_fp16",
            "convert_fp16_to_fp32",
            "convert_fp32_to_fp16_rounding",
            "convert_fp32_to_fp16_vec2",
            "convert_fp16_to_fp32_vec2"
        };
    }
}
