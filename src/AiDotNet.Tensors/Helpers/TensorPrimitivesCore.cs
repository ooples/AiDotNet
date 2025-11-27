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

    #region Binary Operations

    /// <summary>
    /// Applies a binary operator to two spans of double values using the best available SIMD instructions.
    /// </summary>
    /// <typeparam name="TOperator">The binary operator to apply.</typeparam>
    /// <param name="x">The first input span.</param>
    /// <param name="y">The second input span.</param>
    /// <param name="destination">The destination span to write results to.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void InvokeSpanSpanIntoSpan<TOperator>(ReadOnlySpan<double> x, ReadOnlySpan<double> y, Span<double> destination)
        where TOperator : struct, IBinaryOperator<double, double>
    {
        if (x.Length != y.Length || x.Length != destination.Length)
            throw new ArgumentException("All spans must have the same length.");

        TOperator op = default;
        int i = 0;

#if NET5_0_OR_GREATER
        // AVX-512: Process 8 doubles at a time
        if (Vector512.IsHardwareAccelerated && x.Length >= Vector512<double>.Count)
        {
            int vectorCount = x.Length - (x.Length % Vector512<double>.Count);
            for (; i < vectorCount; i += Vector512<double>.Count)
            {
                var vecX = Vector512.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector512.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                var result = op.Invoke(vecX, vecY);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
        // AVX2: Process 4 doubles at a time
        else if (Vector256.IsHardwareAccelerated && x.Length >= Vector256<double>.Count)
        {
            int vectorCount = x.Length - (x.Length % Vector256<double>.Count);
            for (; i < vectorCount; i += Vector256<double>.Count)
            {
                var vecX = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                var result = op.Invoke(vecX, vecY);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
        // SSE/NEON: Process 2 doubles at a time
        else if (Vector128.IsHardwareAccelerated && x.Length >= Vector128<double>.Count)
        {
            int vectorCount = x.Length - (x.Length % Vector128<double>.Count);
            for (; i < vectorCount; i += Vector128<double>.Count)
            {
                var vecX = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                var result = op.Invoke(vecX, vecY);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
#endif

        // Scalar fallback for remaining elements
        for (; i < x.Length; i++)
        {
            destination[i] = op.Invoke(x[i], y[i]);
        }
    }

    /// <summary>
    /// Applies a binary operator to two spans of float values using the best available SIMD instructions.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void InvokeSpanSpanIntoSpan<TOperator>(ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<float> destination)
        where TOperator : struct, IBinaryOperator<float, float>
    {
        if (x.Length != y.Length || x.Length != destination.Length)
            throw new ArgumentException("All spans must have the same length.");

        TOperator op = default;
        int i = 0;

#if NET5_0_OR_GREATER
        if (Vector512.IsHardwareAccelerated && x.Length >= Vector512<float>.Count)
        {
            int vectorCount = x.Length - (x.Length % Vector512<float>.Count);
            for (; i < vectorCount; i += Vector512<float>.Count)
            {
                var vecX = Vector512.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector512.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                var result = op.Invoke(vecX, vecY);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated && x.Length >= Vector256<float>.Count)
        {
            int vectorCount = x.Length - (x.Length % Vector256<float>.Count);
            for (; i < vectorCount; i += Vector256<float>.Count)
            {
                var vecX = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                var result = op.Invoke(vecX, vecY);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated && x.Length >= Vector128<float>.Count)
        {
            int vectorCount = x.Length - (x.Length % Vector128<float>.Count);
            for (; i < vectorCount; i += Vector128<float>.Count)
            {
                var vecX = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                var result = op.Invoke(vecX, vecY);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
#endif

        for (; i < x.Length; i++)
        {
            destination[i] = op.Invoke(x[i], y[i]);
        }
    }

    /// <summary>
    /// Applies a binary operator to two spans of int values using the best available SIMD instructions.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void InvokeSpanSpanIntoSpan<TOperator>(ReadOnlySpan<int> x, ReadOnlySpan<int> y, Span<int> destination)
        where TOperator : struct, IBinaryOperator<int, int>
    {
        if (x.Length != y.Length || x.Length != destination.Length)
            throw new ArgumentException("All spans must have the same length.");

        TOperator op = default;
        int i = 0;

#if NET5_0_OR_GREATER
        if (Vector512.IsHardwareAccelerated && x.Length >= Vector512<int>.Count)
        {
            int vectorCount = x.Length - (x.Length % Vector512<int>.Count);
            for (; i < vectorCount; i += Vector512<int>.Count)
            {
                var vecX = Vector512.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector512.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                var result = op.Invoke(vecX, vecY);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated && x.Length >= Vector256<int>.Count)
        {
            int vectorCount = x.Length - (x.Length % Vector256<int>.Count);
            for (; i < vectorCount; i += Vector256<int>.Count)
            {
                var vecX = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                var result = op.Invoke(vecX, vecY);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated && x.Length >= Vector128<int>.Count)
        {
            int vectorCount = x.Length - (x.Length % Vector128<int>.Count);
            for (; i < vectorCount; i += Vector128<int>.Count)
            {
                var vecX = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                var result = op.Invoke(vecX, vecY);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
#endif

        for (; i < x.Length; i++)
        {
            destination[i] = op.Invoke(x[i], y[i]);
        }
    }

    /// <summary>
    /// Applies a binary operator to two spans of long values using the best available SIMD instructions.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void InvokeSpanSpanIntoSpan<TOperator>(ReadOnlySpan<long> x, ReadOnlySpan<long> y, Span<long> destination)
        where TOperator : struct, IBinaryOperator<long, long>
    {
        if (x.Length != y.Length || x.Length != destination.Length)
            throw new ArgumentException("All spans must have the same length.");

        TOperator op = default;
        int i = 0;

#if NET5_0_OR_GREATER
        if (Vector512.IsHardwareAccelerated && x.Length >= Vector512<long>.Count)
        {
            int vectorCount = x.Length - (x.Length % Vector512<long>.Count);
            for (; i < vectorCount; i += Vector512<long>.Count)
            {
                var vecX = Vector512.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector512.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                var result = op.Invoke(vecX, vecY);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
        else if (Vector256.IsHardwareAccelerated && x.Length >= Vector256<long>.Count)
        {
            int vectorCount = x.Length - (x.Length % Vector256<long>.Count);
            for (; i < vectorCount; i += Vector256<long>.Count)
            {
                var vecX = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                var result = op.Invoke(vecX, vecY);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated && x.Length >= Vector128<long>.Count)
        {
            int vectorCount = x.Length - (x.Length % Vector128<long>.Count);
            for (; i < vectorCount; i += Vector128<long>.Count)
            {
                var vecX = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                var result = op.Invoke(vecX, vecY);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
#endif

        for (; i < x.Length; i++)
        {
            destination[i] = op.Invoke(x[i], y[i]);
        }
    }

    /// <summary>
    /// Applies a binary operator to two spans of short values using the best available SIMD instructions.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void InvokeSpanSpanIntoSpan<TOperator>(ReadOnlySpan<short> x, ReadOnlySpan<short> y, Span<short> destination)
        where TOperator : struct, IBinaryOperator<short, short>
    {
        if (x.Length != y.Length || x.Length != destination.Length)
            throw new ArgumentException("All spans must have the same length.");

        TOperator op = default;
        int i = 0;

#if NET5_0_OR_GREATER
        if (Vector256.IsHardwareAccelerated && x.Length >= Vector256<short>.Count)
        {
            int vectorCount = x.Length - (x.Length % Vector256<short>.Count);
            for (; i < vectorCount; i += Vector256<short>.Count)
            {
                var vecX = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                var result = op.Invoke(vecX, vecY);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated && x.Length >= Vector128<short>.Count)
        {
            int vectorCount = x.Length - (x.Length % Vector128<short>.Count);
            for (; i < vectorCount; i += Vector128<short>.Count)
            {
                var vecX = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                var result = op.Invoke(vecX, vecY);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
#endif

        for (; i < x.Length; i++)
        {
            destination[i] = op.Invoke(x[i], y[i]);
        }
    }

    /// <summary>
    /// Applies a binary operator to two spans of ushort values using the best available SIMD instructions.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void InvokeSpanSpanIntoSpan<TOperator>(ReadOnlySpan<ushort> x, ReadOnlySpan<ushort> y, Span<ushort> destination)
        where TOperator : struct, IBinaryOperator<ushort, ushort>
    {
        if (x.Length != y.Length || x.Length != destination.Length)
            throw new ArgumentException("All spans must have the same length.");

        TOperator op = default;
        int i = 0;

#if NET5_0_OR_GREATER
        if (Vector256.IsHardwareAccelerated && x.Length >= Vector256<ushort>.Count)
        {
            int vectorCount = x.Length - (x.Length % Vector256<ushort>.Count);
            for (; i < vectorCount; i += Vector256<ushort>.Count)
            {
                var vecX = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                var result = op.Invoke(vecX, vecY);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated && x.Length >= Vector128<ushort>.Count)
        {
            int vectorCount = x.Length - (x.Length % Vector128<ushort>.Count);
            for (; i < vectorCount; i += Vector128<ushort>.Count)
            {
                var vecX = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                var result = op.Invoke(vecX, vecY);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
#endif

        for (; i < x.Length; i++)
        {
            destination[i] = op.Invoke(x[i], y[i]);
        }
    }

    /// <summary>
    /// Applies a binary operator to two spans of uint values using the best available SIMD instructions.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void InvokeSpanSpanIntoSpan<TOperator>(ReadOnlySpan<uint> x, ReadOnlySpan<uint> y, Span<uint> destination)
        where TOperator : struct, IBinaryOperator<uint, uint>
    {
        if (x.Length != y.Length || x.Length != destination.Length)
            throw new ArgumentException("All spans must have the same length.");

        TOperator op = default;
        int i = 0;

#if NET5_0_OR_GREATER
        if (Vector256.IsHardwareAccelerated && x.Length >= Vector256<uint>.Count)
        {
            int vectorCount = x.Length - (x.Length % Vector256<uint>.Count);
            for (; i < vectorCount; i += Vector256<uint>.Count)
            {
                var vecX = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                var result = op.Invoke(vecX, vecY);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated && x.Length >= Vector128<uint>.Count)
        {
            int vectorCount = x.Length - (x.Length % Vector128<uint>.Count);
            for (; i < vectorCount; i += Vector128<uint>.Count)
            {
                var vecX = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                var result = op.Invoke(vecX, vecY);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
#endif

        for (; i < x.Length; i++)
        {
            destination[i] = op.Invoke(x[i], y[i]);
        }
    }

    /// <summary>
    /// Applies a binary operator to two spans of ulong values using the best available SIMD instructions.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void InvokeSpanSpanIntoSpan<TOperator>(ReadOnlySpan<ulong> x, ReadOnlySpan<ulong> y, Span<ulong> destination)
        where TOperator : struct, IBinaryOperator<ulong, ulong>
    {
        if (x.Length != y.Length || x.Length != destination.Length)
            throw new ArgumentException("All spans must have the same length.");

        TOperator op = default;
        int i = 0;

#if NET5_0_OR_GREATER
        if (Vector256.IsHardwareAccelerated && x.Length >= Vector256<ulong>.Count)
        {
            int vectorCount = x.Length - (x.Length % Vector256<ulong>.Count);
            for (; i < vectorCount; i += Vector256<ulong>.Count)
            {
                var vecX = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                var result = op.Invoke(vecX, vecY);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated && x.Length >= Vector128<ulong>.Count)
        {
            int vectorCount = x.Length - (x.Length % Vector128<ulong>.Count);
            for (; i < vectorCount; i += Vector128<ulong>.Count)
            {
                var vecX = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                var result = op.Invoke(vecX, vecY);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
#endif

        for (; i < x.Length; i++)
        {
            destination[i] = op.Invoke(x[i], y[i]);
        }
    }

    /// <summary>
    /// Applies a binary operator to two spans of byte values using the best available SIMD instructions.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void InvokeSpanSpanIntoSpan<TOperator>(ReadOnlySpan<byte> x, ReadOnlySpan<byte> y, Span<byte> destination)
        where TOperator : struct, IBinaryOperator<byte, byte>
    {
        if (x.Length != y.Length || x.Length != destination.Length)
            throw new ArgumentException("All spans must have the same length.");

        TOperator op = default;
        int i = 0;

#if NET5_0_OR_GREATER
        if (Vector256.IsHardwareAccelerated && x.Length >= Vector256<byte>.Count)
        {
            int vectorCount = x.Length - (x.Length % Vector256<byte>.Count);
            for (; i < vectorCount; i += Vector256<byte>.Count)
            {
                var vecX = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                var result = op.Invoke(vecX, vecY);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated && x.Length >= Vector128<byte>.Count)
        {
            int vectorCount = x.Length - (x.Length % Vector128<byte>.Count);
            for (; i < vectorCount; i += Vector128<byte>.Count)
            {
                var vecX = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                var result = op.Invoke(vecX, vecY);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
#endif

        for (; i < x.Length; i++)
        {
            destination[i] = op.Invoke(x[i], y[i]);
        }
    }

    /// <summary>
    /// Applies a binary operator to two spans of sbyte values using the best available SIMD instructions.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void InvokeSpanSpanIntoSpan<TOperator>(ReadOnlySpan<sbyte> x, ReadOnlySpan<sbyte> y, Span<sbyte> destination)
        where TOperator : struct, IBinaryOperator<sbyte, sbyte>
    {
        if (x.Length != y.Length || x.Length != destination.Length)
            throw new ArgumentException("All spans must have the same length.");

        TOperator op = default;
        int i = 0;

#if NET5_0_OR_GREATER
        if (Vector256.IsHardwareAccelerated && x.Length >= Vector256<sbyte>.Count)
        {
            int vectorCount = x.Length - (x.Length % Vector256<sbyte>.Count);
            for (; i < vectorCount; i += Vector256<sbyte>.Count)
            {
                var vecX = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                var result = op.Invoke(vecX, vecY);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated && x.Length >= Vector128<sbyte>.Count)
        {
            int vectorCount = x.Length - (x.Length % Vector128<sbyte>.Count);
            for (; i < vectorCount; i += Vector128<sbyte>.Count)
            {
                var vecX = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                var result = op.Invoke(vecX, vecY);
                result.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(destination), (nuint)i);
            }
        }
#endif

        for (; i < x.Length; i++)
        {
            destination[i] = op.Invoke(x[i], y[i]);
        }
    }

    #endregion

    #region Reduction Operations

    /// <summary>
    /// Computes the sum of all elements in a span of doubles using SIMD acceleration.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Sum(ReadOnlySpan<double> x)
    {
        double sum = 0;
        int i = 0;

#if NET5_0_OR_GREATER
        if (Vector256.IsHardwareAccelerated && x.Length >= Vector256<double>.Count)
        {
            var vSum = Vector256<double>.Zero;
            int vectorCount = x.Length - (x.Length % Vector256<double>.Count);
            for (; i < vectorCount; i += Vector256<double>.Count)
            {
                var vec = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                vSum = Vector256.Add(vSum, vec);
            }
            sum = Vector256.Sum(vSum);
        }
        else if (Vector128.IsHardwareAccelerated && x.Length >= Vector128<double>.Count)
        {
            var vSum = Vector128<double>.Zero;
            int vectorCount = x.Length - (x.Length % Vector128<double>.Count);
            for (; i < vectorCount; i += Vector128<double>.Count)
            {
                var vec = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                vSum = Vector128.Add(vSum, vec);
            }
            sum = Vector128.Sum(vSum);
        }
#endif

        for (; i < x.Length; i++)
            sum += x[i];

        return sum;
    }

    /// <summary>
    /// Computes the sum of all elements in a span of floats using SIMD acceleration.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sum(ReadOnlySpan<float> x)
    {
        float sum = 0;
        int i = 0;

#if NET5_0_OR_GREATER
        if (Vector256.IsHardwareAccelerated && x.Length >= Vector256<float>.Count)
        {
            var vSum = Vector256<float>.Zero;
            int vectorCount = x.Length - (x.Length % Vector256<float>.Count);
            for (; i < vectorCount; i += Vector256<float>.Count)
            {
                var vec = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                vSum = Vector256.Add(vSum, vec);
            }
            sum = Vector256.Sum(vSum);
        }
        else if (Vector128.IsHardwareAccelerated && x.Length >= Vector128<float>.Count)
        {
            var vSum = Vector128<float>.Zero;
            int vectorCount = x.Length - (x.Length % Vector128<float>.Count);
            for (; i < vectorCount; i += Vector128<float>.Count)
            {
                var vec = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                vSum = Vector128.Add(vSum, vec);
            }
            sum = Vector128.Sum(vSum);
        }
#endif

        for (; i < x.Length; i++)
            sum += x[i];

        return sum;
    }

    /// <summary>
    /// Computes the dot product of two spans of doubles using SIMD acceleration.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Dot(ReadOnlySpan<double> x, ReadOnlySpan<double> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Spans must have the same length.");

        double sum = 0;
        int i = 0;

#if NET5_0_OR_GREATER
        if (Vector256.IsHardwareAccelerated && x.Length >= Vector256<double>.Count)
        {
            var vSum = Vector256<double>.Zero;
            int vectorCount = x.Length - (x.Length % Vector256<double>.Count);
            for (; i < vectorCount; i += Vector256<double>.Count)
            {
                var vecX = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                vSum = Vector256.Add(vSum, Vector256.Multiply(vecX, vecY));
            }
            sum = Vector256.Sum(vSum);
        }
        else if (Vector128.IsHardwareAccelerated && x.Length >= Vector128<double>.Count)
        {
            var vSum = Vector128<double>.Zero;
            int vectorCount = x.Length - (x.Length % Vector128<double>.Count);
            for (; i < vectorCount; i += Vector128<double>.Count)
            {
                var vecX = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                vSum = Vector128.Add(vSum, Vector128.Multiply(vecX, vecY));
            }
            sum = Vector128.Sum(vSum);
        }
#endif

        for (; i < x.Length; i++)
            sum += x[i] * y[i];

        return sum;
    }

    /// <summary>
    /// Computes the dot product of two spans of floats using SIMD acceleration.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Dot(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Spans must have the same length.");

        float sum = 0;
        int i = 0;

#if NET5_0_OR_GREATER
        if (Vector256.IsHardwareAccelerated && x.Length >= Vector256<float>.Count)
        {
            var vSum = Vector256<float>.Zero;
            int vectorCount = x.Length - (x.Length % Vector256<float>.Count);
            for (; i < vectorCount; i += Vector256<float>.Count)
            {
                var vecX = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                vSum = Vector256.Add(vSum, Vector256.Multiply(vecX, vecY));
            }
            sum = Vector256.Sum(vSum);
        }
        else if (Vector128.IsHardwareAccelerated && x.Length >= Vector128<float>.Count)
        {
            var vSum = Vector128<float>.Zero;
            int vectorCount = x.Length - (x.Length % Vector128<float>.Count);
            for (; i < vectorCount; i += Vector128<float>.Count)
            {
                var vecX = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                vSum = Vector128.Add(vSum, Vector128.Multiply(vecX, vecY));
            }
            sum = Vector128.Sum(vSum);
        }
#endif

        for (; i < x.Length; i++)
            sum += x[i] * y[i];

        return sum;
    }

    /// <summary>
    /// Finds the maximum value in a span of doubles using SIMD acceleration.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Max(ReadOnlySpan<double> x)
    {
        if (x.Length == 0)
            throw new ArgumentException("Span cannot be empty.");

        double max = x[0];
        int i = 1;

#if NET5_0_OR_GREATER
        if (Vector256.IsHardwareAccelerated && x.Length >= Vector256<double>.Count)
        {
            var vMax = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), 0);
            int vectorCount = x.Length - (x.Length % Vector256<double>.Count);
            for (i = Vector256<double>.Count; i < vectorCount; i += Vector256<double>.Count)
            {
                var vec = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                vMax = Vector256.Max(vMax, vec);
            }
            // Reduce vector to scalar
            Span<double> temp = stackalloc double[Vector256<double>.Count];
            vMax.CopyTo(temp);
            max = temp[0];
            for (int j = 1; j < temp.Length; j++)
                if (temp[j] > max) max = temp[j];
        }
#endif

        for (; i < x.Length; i++)
            if (x[i] > max) max = x[i];

        return max;
    }

    /// <summary>
    /// Finds the maximum value in a span of floats using SIMD acceleration.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Max(ReadOnlySpan<float> x)
    {
        if (x.Length == 0)
            throw new ArgumentException("Span cannot be empty.");

        float max = x[0];
        int i = 1;

#if NET5_0_OR_GREATER
        if (Vector256.IsHardwareAccelerated && x.Length >= Vector256<float>.Count)
        {
            var vMax = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), 0);
            int vectorCount = x.Length - (x.Length % Vector256<float>.Count);
            for (i = Vector256<float>.Count; i < vectorCount; i += Vector256<float>.Count)
            {
                var vec = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                vMax = Vector256.Max(vMax, vec);
            }
            Span<float> temp = stackalloc float[Vector256<float>.Count];
            vMax.CopyTo(temp);
            max = temp[0];
            for (int j = 1; j < temp.Length; j++)
                if (temp[j] > max) max = temp[j];
        }
#endif

        for (; i < x.Length; i++)
            if (x[i] > max) max = x[i];

        return max;
    }

    /// <summary>
    /// Finds the minimum value in a span of doubles using SIMD acceleration.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Min(ReadOnlySpan<double> x)
    {
        if (x.Length == 0)
            throw new ArgumentException("Span cannot be empty.");

        double min = x[0];
        int i = 1;

#if NET5_0_OR_GREATER
        if (Vector256.IsHardwareAccelerated && x.Length >= Vector256<double>.Count)
        {
            var vMin = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), 0);
            int vectorCount = x.Length - (x.Length % Vector256<double>.Count);
            for (i = Vector256<double>.Count; i < vectorCount; i += Vector256<double>.Count)
            {
                var vec = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                vMin = Vector256.Min(vMin, vec);
            }
            Span<double> temp = stackalloc double[Vector256<double>.Count];
            vMin.CopyTo(temp);
            min = temp[0];
            for (int j = 1; j < temp.Length; j++)
                if (temp[j] < min) min = temp[j];
        }
#endif

        for (; i < x.Length; i++)
            if (x[i] < min) min = x[i];

        return min;
    }

    /// <summary>
    /// Finds the minimum value in a span of floats using SIMD acceleration.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Min(ReadOnlySpan<float> x)
    {
        if (x.Length == 0)
            throw new ArgumentException("Span cannot be empty.");

        float min = x[0];
        int i = 1;

#if NET5_0_OR_GREATER
        if (Vector256.IsHardwareAccelerated && x.Length >= Vector256<float>.Count)
        {
            var vMin = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), 0);
            int vectorCount = x.Length - (x.Length % Vector256<float>.Count);
            for (i = Vector256<float>.Count; i < vectorCount; i += Vector256<float>.Count)
            {
                var vec = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                vMin = Vector256.Min(vMin, vec);
            }
            Span<float> temp = stackalloc float[Vector256<float>.Count];
            vMin.CopyTo(temp);
            min = temp[0];
            for (int j = 1; j < temp.Length; j++)
                if (temp[j] < min) min = temp[j];
        }
#endif

        for (; i < x.Length; i++)
            if (x[i] < min) min = x[i];

        return min;
    }

    /// <summary>
    /// Computes the sum of all elements in a span of ints using SIMD acceleration.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Sum(ReadOnlySpan<int> x)
    {
        int sum = 0;
        int i = 0;

#if NET5_0_OR_GREATER
        if (Vector256.IsHardwareAccelerated && x.Length >= Vector256<int>.Count)
        {
            var vSum = Vector256<int>.Zero;
            int vectorCount = x.Length - (x.Length % Vector256<int>.Count);
            for (; i < vectorCount; i += Vector256<int>.Count)
            {
                var vec = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                vSum = Vector256.Add(vSum, vec);
            }
            sum = Vector256.Sum(vSum);
        }
        else if (Vector128.IsHardwareAccelerated && x.Length >= Vector128<int>.Count)
        {
            var vSum = Vector128<int>.Zero;
            int vectorCount = x.Length - (x.Length % Vector128<int>.Count);
            for (; i < vectorCount; i += Vector128<int>.Count)
            {
                var vec = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                vSum = Vector128.Add(vSum, vec);
            }
            sum = Vector128.Sum(vSum);
        }
#endif

        for (; i < x.Length; i++)
            sum += x[i];

        return sum;
    }

    /// <summary>
    /// Computes the sum of all elements in a span of longs using SIMD acceleration.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static long Sum(ReadOnlySpan<long> x)
    {
        long sum = 0;
        int i = 0;

#if NET5_0_OR_GREATER
        if (Vector256.IsHardwareAccelerated && x.Length >= Vector256<long>.Count)
        {
            var vSum = Vector256<long>.Zero;
            int vectorCount = x.Length - (x.Length % Vector256<long>.Count);
            for (; i < vectorCount; i += Vector256<long>.Count)
            {
                var vec = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                vSum = Vector256.Add(vSum, vec);
            }
            sum = Vector256.Sum(vSum);
        }
        else if (Vector128.IsHardwareAccelerated && x.Length >= Vector128<long>.Count)
        {
            var vSum = Vector128<long>.Zero;
            int vectorCount = x.Length - (x.Length % Vector128<long>.Count);
            for (; i < vectorCount; i += Vector128<long>.Count)
            {
                var vec = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                vSum = Vector128.Add(vSum, vec);
            }
            sum = Vector128.Sum(vSum);
        }
#endif

        for (; i < x.Length; i++)
            sum += x[i];

        return sum;
    }

    /// <summary>
    /// Computes the dot product of two spans of ints using SIMD acceleration.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Dot(ReadOnlySpan<int> x, ReadOnlySpan<int> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Spans must have the same length.");

        int sum = 0;
        int i = 0;

#if NET5_0_OR_GREATER
        if (Vector256.IsHardwareAccelerated && x.Length >= Vector256<int>.Count)
        {
            var vSum = Vector256<int>.Zero;
            int vectorCount = x.Length - (x.Length % Vector256<int>.Count);
            for (; i < vectorCount; i += Vector256<int>.Count)
            {
                var vecX = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                vSum = Vector256.Add(vSum, Vector256.Multiply(vecX, vecY));
            }
            sum = Vector256.Sum(vSum);
        }
        else if (Vector128.IsHardwareAccelerated && x.Length >= Vector128<int>.Count)
        {
            var vSum = Vector128<int>.Zero;
            int vectorCount = x.Length - (x.Length % Vector128<int>.Count);
            for (; i < vectorCount; i += Vector128<int>.Count)
            {
                var vecX = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                vSum = Vector128.Add(vSum, Vector128.Multiply(vecX, vecY));
            }
            sum = Vector128.Sum(vSum);
        }
#endif

        for (; i < x.Length; i++)
            sum += x[i] * y[i];

        return sum;
    }

    /// <summary>
    /// Computes the dot product of two spans of longs using SIMD acceleration.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static long Dot(ReadOnlySpan<long> x, ReadOnlySpan<long> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Spans must have the same length.");

        long sum = 0;
        int i = 0;

#if NET5_0_OR_GREATER
        if (Vector256.IsHardwareAccelerated && x.Length >= Vector256<long>.Count)
        {
            var vSum = Vector256<long>.Zero;
            int vectorCount = x.Length - (x.Length % Vector256<long>.Count);
            for (; i < vectorCount; i += Vector256<long>.Count)
            {
                var vecX = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                vSum = Vector256.Add(vSum, Vector256.Multiply(vecX, vecY));
            }
            sum = Vector256.Sum(vSum);
        }
        else if (Vector128.IsHardwareAccelerated && x.Length >= Vector128<long>.Count)
        {
            var vSum = Vector128<long>.Zero;
            int vectorCount = x.Length - (x.Length % Vector128<long>.Count);
            for (; i < vectorCount; i += Vector128<long>.Count)
            {
                var vecX = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                var vecY = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(y), (nuint)i);
                vSum = Vector128.Add(vSum, Vector128.Multiply(vecX, vecY));
            }
            sum = Vector128.Sum(vSum);
        }
#endif

        for (; i < x.Length; i++)
            sum += x[i] * y[i];

        return sum;
    }

    /// <summary>
    /// Finds the maximum value in a span of ints using SIMD acceleration.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Max(ReadOnlySpan<int> x)
    {
        if (x.Length == 0)
            throw new ArgumentException("Span cannot be empty.");

        int max = x[0];
        int i = 1;

#if NET5_0_OR_GREATER
        if (Vector256.IsHardwareAccelerated && x.Length >= Vector256<int>.Count)
        {
            var vMax = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), 0);
            int vectorCount = x.Length - (x.Length % Vector256<int>.Count);
            for (i = Vector256<int>.Count; i < vectorCount; i += Vector256<int>.Count)
            {
                var vec = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                vMax = Vector256.Max(vMax, vec);
            }
            Span<int> temp = stackalloc int[Vector256<int>.Count];
            vMax.CopyTo(temp);
            max = temp[0];
            for (int j = 1; j < temp.Length; j++)
                if (temp[j] > max) max = temp[j];
        }
#endif

        for (; i < x.Length; i++)
            if (x[i] > max) max = x[i];

        return max;
    }

    /// <summary>
    /// Finds the maximum value in a span of longs using SIMD acceleration.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static long Max(ReadOnlySpan<long> x)
    {
        if (x.Length == 0)
            throw new ArgumentException("Span cannot be empty.");

        long max = x[0];
        int i = 1;

#if NET5_0_OR_GREATER
        if (Vector256.IsHardwareAccelerated && x.Length >= Vector256<long>.Count)
        {
            var vMax = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), 0);
            int vectorCount = x.Length - (x.Length % Vector256<long>.Count);
            for (i = Vector256<long>.Count; i < vectorCount; i += Vector256<long>.Count)
            {
                var vec = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                vMax = Vector256.Max(vMax, vec);
            }
            Span<long> temp = stackalloc long[Vector256<long>.Count];
            vMax.CopyTo(temp);
            max = temp[0];
            for (int j = 1; j < temp.Length; j++)
                if (temp[j] > max) max = temp[j];
        }
#endif

        for (; i < x.Length; i++)
            if (x[i] > max) max = x[i];

        return max;
    }

    /// <summary>
    /// Finds the minimum value in a span of ints using SIMD acceleration.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Min(ReadOnlySpan<int> x)
    {
        if (x.Length == 0)
            throw new ArgumentException("Span cannot be empty.");

        int min = x[0];
        int i = 1;

#if NET5_0_OR_GREATER
        if (Vector256.IsHardwareAccelerated && x.Length >= Vector256<int>.Count)
        {
            var vMin = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), 0);
            int vectorCount = x.Length - (x.Length % Vector256<int>.Count);
            for (i = Vector256<int>.Count; i < vectorCount; i += Vector256<int>.Count)
            {
                var vec = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                vMin = Vector256.Min(vMin, vec);
            }
            Span<int> temp = stackalloc int[Vector256<int>.Count];
            vMin.CopyTo(temp);
            min = temp[0];
            for (int j = 1; j < temp.Length; j++)
                if (temp[j] < min) min = temp[j];
        }
#endif

        for (; i < x.Length; i++)
            if (x[i] < min) min = x[i];

        return min;
    }

    /// <summary>
    /// Finds the minimum value in a span of longs using SIMD acceleration.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static long Min(ReadOnlySpan<long> x)
    {
        if (x.Length == 0)
            throw new ArgumentException("Span cannot be empty.");

        long min = x[0];
        int i = 1;

#if NET5_0_OR_GREATER
        if (Vector256.IsHardwareAccelerated && x.Length >= Vector256<long>.Count)
        {
            var vMin = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), 0);
            int vectorCount = x.Length - (x.Length % Vector256<long>.Count);
            for (i = Vector256<long>.Count; i < vectorCount; i += Vector256<long>.Count)
            {
                var vec = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x), (nuint)i);
                vMin = Vector256.Min(vMin, vec);
            }
            Span<long> temp = stackalloc long[Vector256<long>.Count];
            vMin.CopyTo(temp);
            min = temp[0];
            for (int j = 1; j < temp.Length; j++)
                if (temp[j] < min) min = temp[j];
        }
#endif

        for (; i < x.Length; i++)
            if (x[i] < min) min = x[i];

        return min;
    }

    #endregion

    #region Diagnostics

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
