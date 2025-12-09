using System.Runtime.CompilerServices;
#if NET6_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics.Arm;
#endif

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Provides SIMD vector operations using hardware intrinsics.
/// </summary>
/// <remarks>
/// <para>
/// This class replaces System.Numerics.Vector with direct intrinsics usage for better
/// control and performance. It automatically selects the best available instruction set
/// (AVX-512, AVX, SSE, or ARM NEON).
/// </para>
/// <para><b>For Beginners:</b> SIMD (Single Instruction Multiple Data) allows processing
/// multiple numbers with a single CPU instruction. This class provides methods to:
/// - Load multiple numbers from memory into a SIMD register
/// - Perform arithmetic on all numbers at once
/// - Store the results back to memory
///
/// For example, with AVX you can add 8 floats in one instruction instead of 8 separate adds.
/// </para>
/// </remarks>
public static class SimdVector
{
    #region Hardware Detection

    /// <summary>
    /// Gets whether any SIMD acceleration is available.
    /// </summary>
    public static bool IsHardwareAccelerated => Sse.IsSupported || AdvSimd.IsSupported;

    /// <summary>
    /// Gets the number of float elements that fit in a SIMD register.
    /// </summary>
    public static int FloatCount
    {
        get
        {
            if (Avx512F.IsSupported) return 16;
            if (Avx.IsSupported) return 8;
            if (Sse.IsSupported || AdvSimd.IsSupported) return 4;
            return 1;
        }
    }

    /// <summary>
    /// Gets the number of double elements that fit in a SIMD register.
    /// </summary>
    public static int DoubleCount
    {
        get
        {
            if (Avx512F.IsSupported) return 8;
            if (Avx.IsSupported) return 4;
            if (Sse2.IsSupported || AdvSimd.IsSupported) return 2;
            return 1;
        }
    }

    #endregion

    #region Float Operations

    /// <summary>
    /// Loads floats from a span into a Vector256.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> LoadVector256(ReadOnlySpan<float> source)
    {
        if (Avx.IsSupported && source.Length >= 8)
        {
            unsafe
            {
                fixed (float* ptr = source)
                {
                    return Avx.LoadVector256(ptr);
                }
            }
        }
        // Fallback: create from values, zero-padding if source is shorter than 8 elements.
        // Zero-padding is intentional for SIMD operations - callers should handle the mask
        // if they need to ignore padded elements (e.g., for reductions like sum or dot product).
        return Vector256.Create(
            source.Length > 0 ? source[0] : 0f,
            source.Length > 1 ? source[1] : 0f,
            source.Length > 2 ? source[2] : 0f,
            source.Length > 3 ? source[3] : 0f,
            source.Length > 4 ? source[4] : 0f,
            source.Length > 5 ? source[5] : 0f,
            source.Length > 6 ? source[6] : 0f,
            source.Length > 7 ? source[7] : 0f);
    }

    /// <summary>
    /// Loads floats from a span into a Vector128.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> LoadVector128(ReadOnlySpan<float> source)
    {
        if (Sse.IsSupported && source.Length >= 4)
        {
            unsafe
            {
                fixed (float* ptr = source)
                {
                    return Sse.LoadVector128(ptr);
                }
            }
        }
        if (AdvSimd.IsSupported && source.Length >= 4)
        {
            unsafe
            {
                fixed (float* ptr = source)
                {
                    return AdvSimd.LoadVector128(ptr);
                }
            }
        }
        return Vector128.Create(
            source.Length > 0 ? source[0] : 0f,
            source.Length > 1 ? source[1] : 0f,
            source.Length > 2 ? source[2] : 0f,
            source.Length > 3 ? source[3] : 0f);
    }

    /// <summary>
    /// Creates a Vector256 with all elements set to the same value.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> BroadcastFloat256(float value)
    {
        return Vector256.Create(value);
    }

    /// <summary>
    /// Creates a Vector128 with all elements set to the same value.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> BroadcastFloat128(float value)
    {
        return Vector128.Create(value);
    }

    /// <summary>
    /// Stores a Vector256 to a span.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void StoreVector256(Vector256<float> vector, Span<float> destination)
    {
        if (Avx.IsSupported && destination.Length >= 8)
        {
            unsafe
            {
                fixed (float* ptr = destination)
                {
                    Avx.Store(ptr, vector);
                }
            }
        }
        else
        {
            for (int i = 0; i < Math.Min(8, destination.Length); i++)
            {
                destination[i] = vector.GetElement(i);
            }
        }
    }

    /// <summary>
    /// Stores a Vector128 to a span.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void StoreVector128(Vector128<float> vector, Span<float> destination)
    {
        if (Sse.IsSupported && destination.Length >= 4)
        {
            unsafe
            {
                fixed (float* ptr = destination)
                {
                    Sse.Store(ptr, vector);
                }
            }
        }
        else if (AdvSimd.IsSupported && destination.Length >= 4)
        {
            unsafe
            {
                fixed (float* ptr = destination)
                {
                    AdvSimd.Store(ptr, vector);
                }
            }
        }
        else
        {
            for (int i = 0; i < Math.Min(4, destination.Length); i++)
            {
                destination[i] = vector.GetElement(i);
            }
        }
    }

    /// <summary>
    /// Adds two Vector256 floats.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Add(Vector256<float> left, Vector256<float> right)
    {
        if (Avx.IsSupported)
            return Avx.Add(left, right);
        return Vector256.Add(left, right);
    }

    /// <summary>
    /// Adds two Vector128 floats.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Add(Vector128<float> left, Vector128<float> right)
    {
        if (Sse.IsSupported)
            return Sse.Add(left, right);
        if (AdvSimd.IsSupported)
            return AdvSimd.Add(left, right);
        return Vector128.Add(left, right);
    }

    /// <summary>
    /// Multiplies two Vector256 floats.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Multiply(Vector256<float> left, Vector256<float> right)
    {
        if (Avx.IsSupported)
            return Avx.Multiply(left, right);
        return Vector256.Multiply(left, right);
    }

    /// <summary>
    /// Multiplies two Vector128 floats.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Multiply(Vector128<float> left, Vector128<float> right)
    {
        if (Sse.IsSupported)
            return Sse.Multiply(left, right);
        if (AdvSimd.IsSupported)
            return AdvSimd.Multiply(left, right);
        return Vector128.Multiply(left, right);
    }

    /// <summary>
    /// Performs fused multiply-add: (a * b) + c.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> MultiplyAdd(Vector256<float> a, Vector256<float> b, Vector256<float> c)
    {
        if (Fma.IsSupported)
            return Fma.MultiplyAdd(a, b, c);
        return Add(Multiply(a, b), c);
    }

    /// <summary>
    /// Performs fused multiply-add: (a * b) + c.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> MultiplyAdd(Vector128<float> a, Vector128<float> b, Vector128<float> c)
    {
        if (AdvSimd.IsSupported)
            return AdvSimd.FusedMultiplyAdd(c, a, b);
        return Add(Multiply(a, b), c);
    }

    #endregion

    #region Double Operations

    /// <summary>
    /// Loads doubles from a span into a Vector256.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<double> LoadVector256(ReadOnlySpan<double> source)
    {
        if (Avx.IsSupported && source.Length >= 4)
        {
            unsafe
            {
                fixed (double* ptr = source)
                {
                    return Avx.LoadVector256(ptr);
                }
            }
        }
        return Vector256.Create(
            source.Length > 0 ? source[0] : 0d,
            source.Length > 1 ? source[1] : 0d,
            source.Length > 2 ? source[2] : 0d,
            source.Length > 3 ? source[3] : 0d);
    }

    /// <summary>
    /// Loads doubles from a span into a Vector128.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<double> LoadVector128(ReadOnlySpan<double> source)
    {
        if (Sse2.IsSupported && source.Length >= 2)
        {
            unsafe
            {
                fixed (double* ptr = source)
                {
                    return Sse2.LoadVector128(ptr);
                }
            }
        }
        if (AdvSimd.Arm64.IsSupported && source.Length >= 2)
        {
            unsafe
            {
                fixed (double* ptr = source)
                {
                    return AdvSimd.LoadVector128(ptr);
                }
            }
        }
        return Vector128.Create(
            source.Length > 0 ? source[0] : 0d,
            source.Length > 1 ? source[1] : 0d);
    }

    /// <summary>
    /// Creates a Vector256 with all elements set to the same value.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<double> BroadcastDouble256(double value)
    {
        return Vector256.Create(value);
    }

    /// <summary>
    /// Creates a Vector128 with all elements set to the same value.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<double> BroadcastDouble128(double value)
    {
        return Vector128.Create(value);
    }

    /// <summary>
    /// Stores a Vector256 to a span.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void StoreVector256(Vector256<double> vector, Span<double> destination)
    {
        if (Avx.IsSupported && destination.Length >= 4)
        {
            unsafe
            {
                fixed (double* ptr = destination)
                {
                    Avx.Store(ptr, vector);
                }
            }
        }
        else
        {
            for (int i = 0; i < Math.Min(4, destination.Length); i++)
            {
                destination[i] = vector.GetElement(i);
            }
        }
    }

    /// <summary>
    /// Stores a Vector128 to a span.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void StoreVector128(Vector128<double> vector, Span<double> destination)
    {
        if (Sse2.IsSupported && destination.Length >= 2)
        {
            unsafe
            {
                fixed (double* ptr = destination)
                {
                    Sse2.Store(ptr, vector);
                }
            }
        }
        else if (AdvSimd.Arm64.IsSupported && destination.Length >= 2)
        {
            unsafe
            {
                fixed (double* ptr = destination)
                {
                    AdvSimd.Store(ptr, vector);
                }
            }
        }
        else
        {
            for (int i = 0; i < Math.Min(2, destination.Length); i++)
            {
                destination[i] = vector.GetElement(i);
            }
        }
    }

    /// <summary>
    /// Adds two Vector256 doubles.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<double> Add(Vector256<double> left, Vector256<double> right)
    {
        if (Avx.IsSupported)
            return Avx.Add(left, right);
        return Vector256.Add(left, right);
    }

    /// <summary>
    /// Adds two Vector128 doubles.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<double> Add(Vector128<double> left, Vector128<double> right)
    {
#if NET6_0_OR_GREATER
        if (Sse2.IsSupported)
            return Sse2.Add(left, right);
        if (AdvSimd.Arm64.IsSupported)
            return AdvSimd.Arm64.Add(left, right);
#endif
        return Vector128.Add(left, right);
    }

    /// <summary>
    /// Multiplies two Vector256 doubles.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<double> Multiply(Vector256<double> left, Vector256<double> right)
    {
        if (Avx.IsSupported)
            return Avx.Multiply(left, right);
        return Vector256.Multiply(left, right);
    }

    /// <summary>
    /// Multiplies two Vector128 doubles.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<double> Multiply(Vector128<double> left, Vector128<double> right)
    {
#if NET6_0_OR_GREATER
        if (Sse2.IsSupported)
            return Sse2.Multiply(left, right);
        if (AdvSimd.Arm64.IsSupported)
            return AdvSimd.Arm64.Multiply(left, right);
#endif
        return Vector128.Multiply(left, right);
    }

    /// <summary>
    /// Performs fused multiply-add: (a * b) + c.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<double> MultiplyAdd(Vector256<double> a, Vector256<double> b, Vector256<double> c)
    {
        if (Fma.IsSupported)
            return Fma.MultiplyAdd(a, b, c);
        return Add(Multiply(a, b), c);
    }

    /// <summary>
    /// Performs fused multiply-add: (a * b) + c.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<double> MultiplyAdd(Vector128<double> a, Vector128<double> b, Vector128<double> c)
    {
        if (AdvSimd.Arm64.IsSupported)
            return AdvSimd.Arm64.FusedMultiplyAdd(c, a, b);
        return Add(Multiply(a, b), c);
    }

    #endregion

    #region Adaptive Width Operations

    /// <summary>
    /// Performs SIMD matrix multiplication inner loop for floats.
    /// Automatically selects AVX (8-wide) or SSE (4-wide) based on hardware.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void MatMulInnerLoopFloat(
        float aik,
        ReadOnlySpan<float> B,
        Span<float> C,
        int jStart,
        int jEnd)
    {
        int j = jStart;

        if (Avx.IsSupported)
        {
            var aVec = BroadcastFloat256(aik);
            int jVecEnd = jStart + ((jEnd - jStart) / 8) * 8;

            for (; j < jVecEnd; j += 8)
            {
                var bVec = LoadVector256(B.Slice(j, 8));
                var cVec = LoadVector256(C.Slice(j, 8));
                var result = MultiplyAdd(aVec, bVec, cVec);
                StoreVector256(result, C.Slice(j, 8));
            }
        }
        else if (Sse.IsSupported || AdvSimd.IsSupported)
        {
            var aVec = BroadcastFloat128(aik);
            int jVecEnd = jStart + ((jEnd - jStart) / 4) * 4;

            for (; j < jVecEnd; j += 4)
            {
                var bVec = LoadVector128(B.Slice(j, 4));
                var cVec = LoadVector128(C.Slice(j, 4));
                var result = MultiplyAdd(aVec, bVec, cVec);
                StoreVector128(result, C.Slice(j, 4));
            }
        }

        // Scalar remainder
        for (; j < jEnd; j++)
        {
            C[j] += aik * B[j];
        }
    }

    /// <summary>
    /// Performs SIMD matrix multiplication inner loop for doubles.
    /// Automatically selects AVX (4-wide) or SSE2 (2-wide) based on hardware.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void MatMulInnerLoopDouble(
        double aik,
        ReadOnlySpan<double> B,
        Span<double> C,
        int jStart,
        int jEnd)
    {
        int j = jStart;

        if (Avx.IsSupported)
        {
            var aVec = BroadcastDouble256(aik);
            int jVecEnd = jStart + ((jEnd - jStart) / 4) * 4;

            for (; j < jVecEnd; j += 4)
            {
                var bVec = LoadVector256(B.Slice(j, 4));
                var cVec = LoadVector256(C.Slice(j, 4));
                var result = MultiplyAdd(aVec, bVec, cVec);
                StoreVector256(result, C.Slice(j, 4));
            }
        }
        else if (Sse2.IsSupported || AdvSimd.Arm64.IsSupported)
        {
            var aVec = BroadcastDouble128(aik);
            int jVecEnd = jStart + ((jEnd - jStart) / 2) * 2;

            for (; j < jVecEnd; j += 2)
            {
                var bVec = LoadVector128(B.Slice(j, 2));
                var cVec = LoadVector128(C.Slice(j, 2));
                var result = MultiplyAdd(aVec, bVec, cVec);
                StoreVector128(result, C.Slice(j, 2));
            }
        }

        // Scalar remainder
        for (; j < jEnd; j++)
        {
            C[j] += aik * B[j];
        }
    }

    #endregion
}
