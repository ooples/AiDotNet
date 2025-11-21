using System;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics.Arm;
#endif
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Provides core dispatch logic for SIMD-accelerated tensor operations.
/// </summary>
/// <remarks>
/// <para>
/// This class implements Microsoft's operator pattern for hardware-accelerated operations.
/// It automatically detects available SIMD instruction sets and dispatches to the widest
/// available vector width for maximum performance.
/// </para>
/// <para>
/// <b>SIMD Hierarchy (fastest to slowest):</b>
/// - AVX-512: 512-bit vectors (8 doubles or 16 floats) - Intel Skylake-X 2017+, AMD Zen 4 2022+
/// - AVX2: 256-bit vectors (4 doubles or 8 floats) - Intel Haswell 2013+, AMD Excavator 2015+
/// - SSE/NEON: 128-bit vectors (2 doubles or 4 floats) - All modern x64/ARM64 CPUs
/// - Scalar: Fallback for unsupported hardware or remainder elements
/// </para>
/// <para>
/// <b>Performance Impact:</b>
/// AVX-512 can process 8x more doubles (16x more floats) per instruction than scalar code,
/// achieving 8-12x speedups for transcendental functions like Sin/Cos.
/// </para>
/// </remarks>
public static class TensorPrimitivesCore
{
    /// <summary>
    /// Applies a unary operator to a span of float values using the best available SIMD instructions.
    /// </summary>
    /// <typeparam name="TOperator">The unary operator to apply.</typeparam>
    /// <param name="x">The input span.</param>
    /// <param name="destination">The destination span to write results to.</param>
    /// <remarks>
    /// Automatically dispatches to AVX-512, AVX2, SSE, or scalar based on hardware support.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void InvokeSpanIntoSpan<TOperator>(ReadOnlySpan<float> x, Span<float> destination)
        where TOperator : struct, IUnaryOperator<float, float>
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Input and destination spans must have the same length.");

        TOperator op = default;
        int i = 0;

#if NET5_0_OR_GREATER
        // AVX-512: Process 16 floats at a time
        if (Vector512.IsHardwareAccelerated && x.Length >= Vector512<float>.Count)
        {
            int vector512Count = x.Length - (x.Length % Vector512<float>.Count);
            for (; i < vector512Count; i += Vector512<float>.Count)
            {
                Vector512<float> vec = Vector512.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                Vector512<float> result = op.Invoke(vec);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
        // AVX2: Process 8 floats at a time
        else if (Vector256.IsHardwareAccelerated && x.Length >= Vector256<float>.Count)
        {
            int vector256Count = x.Length - (x.Length % Vector256<float>.Count);
            for (; i < vector256Count; i += Vector256<float>.Count)
            {
                Vector256<float> vec = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                Vector256<float> result = op.Invoke(vec);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
        // SSE/NEON: Process 4 floats at a time
        else if (Vector128.IsHardwareAccelerated && x.Length >= Vector128<float>.Count)
        {
            int vector128Count = x.Length - (x.Length % Vector128<float>.Count);
            for (; i < vector128Count; i += Vector128<float>.Count)
            {
                Vector128<float> vec = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                Vector128<float> result = op.Invoke(vec);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
#endif

        // Scalar fallback for remaining elements
        for (; i < x.Length; i++)
        {
            destination[i] = op.Invoke(x[i]);
        }
    }

    /// <summary>
    /// Applies a unary operator to a span of double values using the best available SIMD instructions.
    /// </summary>
    /// <typeparam name="TOperator">The unary operator to apply.</typeparam>
    /// <param name="x">The input span.</param>
    /// <param name="destination">The destination span to write results to.</param>
    /// <remarks>
    /// Automatically dispatches to AVX-512, AVX2, SSE, or scalar based on hardware support.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void InvokeSpanIntoSpan<TOperator>(ReadOnlySpan<double> x, Span<double> destination)
        where TOperator : struct, IUnaryOperator<double, double>
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Input and destination spans must have the same length.");

        TOperator op = default;
        int i = 0;

#if NET5_0_OR_GREATER
        // AVX-512: Process 8 doubles at a time
        if (Vector512.IsHardwareAccelerated && x.Length >= Vector512<double>.Count)
        {
            int vector512Count = x.Length - (x.Length % Vector512<double>.Count);
            for (; i < vector512Count; i += Vector512<double>.Count)
            {
                Vector512<double> vec = Vector512.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                Vector512<double> result = op.Invoke(vec);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
        // AVX2: Process 4 doubles at a time
        else if (Vector256.IsHardwareAccelerated && x.Length >= Vector256<double>.Count)
        {
            int vector256Count = x.Length - (x.Length % Vector256<double>.Count);
            for (; i < vector256Count; i += Vector256<double>.Count)
            {
                Vector256<double> vec = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                Vector256<double> result = op.Invoke(vec);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
        // SSE/NEON: Process 2 doubles at a time
        else if (Vector128.IsHardwareAccelerated && x.Length >= Vector128<double>.Count)
        {
            int vector128Count = x.Length - (x.Length % Vector128<double>.Count);
            for (; i < vector128Count; i += Vector128<double>.Count)
            {
                Vector128<double> vec = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                Vector128<double> result = op.Invoke(vec);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
#endif

        // Scalar fallback for remaining elements
        for (; i < x.Length; i++)
        {
            destination[i] = op.Invoke(x[i]);
        }
    }

    /// <summary>
    /// Gets diagnostic information about available SIMD instruction sets.
    /// </summary>
    /// <returns>A string describing available hardware acceleration.</returns>
    public static string GetHardwareAccelerationInfo()
    {
        var info = new System.Text.StringBuilder();
        info.AppendLine("SIMD Hardware Acceleration Status:");
#if NET5_0_OR_GREATER
        info.AppendLine($"  Vector512 (AVX-512): {Vector512.IsHardwareAccelerated}");
        info.AppendLine($"  Vector256 (AVX2): {Vector256.IsHardwareAccelerated}");
        info.AppendLine($"  Vector128 (SSE/NEON): {Vector128.IsHardwareAccelerated}");

        if (Avx512F.IsSupported)
            info.AppendLine($"  AVX-512F: Supported");
        if (Avx2.IsSupported)
            info.AppendLine($"  AVX2: Supported");
        if (Sse2.IsSupported)
            info.AppendLine($"  SSE2: Supported");
        if (AdvSimd.IsSupported)
            info.AppendLine($"  ARM NEON: Supported");
#else
        info.AppendLine("  SIMD not available on .NET Framework (requires .NET 5+)");
        info.AppendLine("  Using scalar operations only");
#endif

        return info.ToString();
    }
}
