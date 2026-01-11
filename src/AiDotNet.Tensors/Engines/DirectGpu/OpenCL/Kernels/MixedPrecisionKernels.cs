// Copyright (c) AiDotNet. All rights reserved.
// Mixed precision training kernels for GPU-resident training.
// Provides FP32/FP16 conversion and mixed precision operations.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// GPU kernels for mixed precision training operations.
/// Uses 16-bit representation stored in ushort for compatibility with all OpenCL devices.
/// </summary>
internal static class MixedPrecisionKernels
{
    /// <summary>
    /// Kernel to convert FP32 to FP16 (stored as ushort).
    /// Uses IEEE 754 half-precision format.
    /// </summary>
    public static string ConvertFp32ToFp16 => @"
// ---------------------------------------------------------------------------
// FP32 to FP16 conversion
// Uses software conversion to IEEE 754 half-precision format
// ---------------------------------------------------------------------------

// Helper function to convert float to half (stored as ushort)
ushort float_to_half(float f) {
    uint x = as_uint(f);
    uint sign = (x >> 16) & 0x8000;
    int exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint mantissa = x & 0x007FFFFF;

    if (exp <= 0) {
        // Denormalized number or zero
        if (exp < -10) {
            return (ushort)sign;
        }
        mantissa = (mantissa | 0x00800000) >> (1 - exp);
        return (ushort)(sign | (mantissa >> 13));
    } else if (exp >= 31) {
        // Infinity or NaN
        if (exp == 128 && mantissa != 0) {
            // NaN - preserve some mantissa bits
            return (ushort)(sign | 0x7E00 | (mantissa >> 13));
        }
        return (ushort)(sign | 0x7C00);
    }

    return (ushort)(sign | (exp << 10) | (mantissa >> 13));
}

__kernel void convert_fp32_to_fp16(
    __global const float* input,
    __global ushort* output,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    output[idx] = float_to_half(input[idx]);
}
";

    /// <summary>
    /// Kernel to convert FP16 (stored as ushort) to FP32.
    /// </summary>
    public static string ConvertFp16ToFp32 => @"
// ---------------------------------------------------------------------------
// FP16 to FP32 conversion
// Uses software conversion from IEEE 754 half-precision format
// ---------------------------------------------------------------------------

// Helper function to convert half (stored as ushort) to float
float half_to_float(ushort h) {
    uint sign = (h & 0x8000) << 16;
    uint exp = (h >> 10) & 0x1F;
    uint mantissa = h & 0x03FF;

    if (exp == 0) {
        // Zero or denormalized number
        if (mantissa == 0) {
            return as_float(sign);
        }
        // Denormalized - normalize it
        while ((mantissa & 0x0400) == 0) {
            mantissa <<= 1;
            exp--;
        }
        exp++;
        mantissa &= ~0x0400;
        exp = exp + (127 - 15);
        return as_float(sign | (exp << 23) | (mantissa << 13));
    } else if (exp == 31) {
        // Infinity or NaN
        return as_float(sign | 0x7F800000 | (mantissa << 13));
    }

    exp = exp + (127 - 15);
    return as_float(sign | (exp << 23) | (mantissa << 13));
}

__kernel void convert_fp16_to_fp32(
    __global const ushort* input,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    output[idx] = half_to_float(input[idx]);
}
";

    /// <summary>
    /// Mixed precision forward pass kernel.
    /// Converts FP32 input to FP16, performs computation, converts result to FP32.
    /// </summary>
    public static string MixedPrecisionForward => @"
// ---------------------------------------------------------------------------
// Mixed precision forward pass
// Performs element-wise scaling in mixed precision
// ---------------------------------------------------------------------------

ushort float_to_half_fwd(float f) {
    uint x = as_uint(f);
    uint sign = (x >> 16) & 0x8000;
    int exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint mantissa = x & 0x007FFFFF;

    if (exp <= 0) {
        if (exp < -10) return (ushort)sign;
        mantissa = (mantissa | 0x00800000) >> (1 - exp);
        return (ushort)(sign | (mantissa >> 13));
    } else if (exp >= 31) {
        if (exp == 128 && mantissa != 0) return (ushort)(sign | 0x7E00 | (mantissa >> 13));
        return (ushort)(sign | 0x7C00);
    }
    return (ushort)(sign | (exp << 10) | (mantissa >> 13));
}

float half_to_float_fwd(ushort h) {
    uint sign = (h & 0x8000) << 16;
    uint exp = (h >> 10) & 0x1F;
    uint mantissa = h & 0x03FF;

    if (exp == 0) {
        if (mantissa == 0) return as_float(sign);
        while ((mantissa & 0x0400) == 0) { mantissa <<= 1; exp--; }
        exp++;
        mantissa &= ~0x0400;
        exp = exp + (127 - 15);
        return as_float(sign | (exp << 23) | (mantissa << 13));
    } else if (exp == 31) {
        return as_float(sign | 0x7F800000 | (mantissa << 13));
    }
    exp = exp + (127 - 15);
    return as_float(sign | (exp << 23) | (mantissa << 13));
}

__kernel void mixed_precision_forward(
    __global const float* input,
    __global float* output,
    __global const float* weights,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    // Convert to FP16, compute, convert back to FP32
    ushort input_fp16 = float_to_half_fwd(input[idx]);
    ushort weight_fp16 = float_to_half_fwd(weights[idx]);

    // Compute in FP32 after conversion (for numerical stability)
    float input_f = half_to_float_fwd(input_fp16);
    float weight_f = half_to_float_fwd(weight_fp16);

    output[idx] = input_f * weight_f;
}
";

    /// <summary>
    /// Mixed precision backward pass kernel.
    /// Computes gradients in mixed precision.
    /// </summary>
    public static string MixedPrecisionBackward => @"
// ---------------------------------------------------------------------------
// Mixed precision backward pass
// Computes gradients with mixed precision for memory efficiency
// ---------------------------------------------------------------------------

ushort float_to_half_bwd(float f) {
    uint x = as_uint(f);
    uint sign = (x >> 16) & 0x8000;
    int exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint mantissa = x & 0x007FFFFF;

    if (exp <= 0) {
        if (exp < -10) return (ushort)sign;
        mantissa = (mantissa | 0x00800000) >> (1 - exp);
        return (ushort)(sign | (mantissa >> 13));
    } else if (exp >= 31) {
        if (exp == 128 && mantissa != 0) return (ushort)(sign | 0x7E00 | (mantissa >> 13));
        return (ushort)(sign | 0x7C00);
    }
    return (ushort)(sign | (exp << 10) | (mantissa >> 13));
}

float half_to_float_bwd(ushort h) {
    uint sign = (h & 0x8000) << 16;
    uint exp = (h >> 10) & 0x1F;
    uint mantissa = h & 0x03FF;

    if (exp == 0) {
        if (mantissa == 0) return as_float(sign);
        while ((mantissa & 0x0400) == 0) { mantissa <<= 1; exp--; }
        exp++;
        mantissa &= ~0x0400;
        exp = exp + (127 - 15);
        return as_float(sign | (exp << 23) | (mantissa << 13));
    } else if (exp == 31) {
        return as_float(sign | 0x7F800000 | (mantissa << 13));
    }
    exp = exp + (127 - 15);
    return as_float(sign | (exp << 23) | (mantissa << 13));
}

__kernel void mixed_precision_backward(
    __global const float* output_grad,
    __global const float* input,
    __global float* input_grad,
    __global const float* weights,
    __global float* weight_grad,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    // Convert to FP16 for storage efficiency
    ushort og_fp16 = float_to_half_bwd(output_grad[idx]);
    ushort in_fp16 = float_to_half_bwd(input[idx]);
    ushort w_fp16 = float_to_half_bwd(weights[idx]);

    // Compute gradients in FP32
    float og = half_to_float_bwd(og_fp16);
    float in_val = half_to_float_bwd(in_fp16);
    float w = half_to_float_bwd(w_fp16);

    // d(input) = output_grad * weight
    input_grad[idx] = og * w;
    // d(weight) = output_grad * input
    weight_grad[idx] = og * in_val;
}
";

    /// <summary>
    /// Accumulate gradients in FP32 for numerical stability.
    /// Used with loss scaling for mixed precision training.
    /// </summary>
    public static string AccumulateGradientFp32 => @"
// ---------------------------------------------------------------------------
// Accumulate gradients in FP32
// Adds scaled gradients to accumulator for numerical stability
// ---------------------------------------------------------------------------

__kernel void accumulate_gradient_fp32(
    __global float* accumulator,
    __global const float* gradient,
    const float scale,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    // Accumulate with scaling (for loss scaling in mixed precision)
    accumulator[idx] += gradient[idx] * scale;
}
";

    /// <summary>
    /// Gets all kernel names for registration.
    /// </summary>
    public static string[] GetKernelNames()
    {
        return new[]
        {
            "convert_fp32_to_fp16",
            "convert_fp16_to_fp32",
            "mixed_precision_forward",
            "mixed_precision_backward",
            "accumulate_gradient_fp32"
        };
    }
}
