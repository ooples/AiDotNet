#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
#endif

namespace AiDotNet.Tensors.Interfaces;

/// <summary>
/// Represents a unary operator that can be applied to scalar values and SIMD vectors.
/// </summary>
/// <typeparam name="T">The numeric type for scalar operations.</typeparam>
/// <typeparam name="TVector">The numeric type for SIMD vector operations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// This interface defines the operator pattern used by Microsoft's TensorPrimitives implementation.
/// It allows a single operator implementation to work across:
/// - Scalar values (for fallback and small arrays)
/// - Vector128 (SSE/NEON - 128-bit SIMD) [.NET 5+ only]
/// - Vector256 (AVX2 - 256-bit SIMD) [.NET 5+ only]
/// - Vector512 (AVX-512 - 512-bit SIMD) [.NET 5+ only]
/// </para>
/// <para>
/// The dispatch logic in TensorPrimitivesCore automatically selects the best available
/// SIMD width at runtime based on hardware capabilities.
/// </para>
/// <para>
/// <b>Framework Support:</b>
/// - .NET Framework 4.6.2/4.7.1: Scalar operations only
/// - .NET 5+/.NET 8.0: Full SIMD support (Vector128/256/512)
/// </para>
/// </remarks>
public interface IUnaryOperator<T, TVector>
{
    /// <summary>
    /// Performs the operation on a single scalar value.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The result of the operation.</returns>
    /// <remarks>
    /// Used for fallback when SIMD is not available or for the remaining elements
    /// that don't fill a complete SIMD vector.
    /// </remarks>
    T Invoke(T x);

#if NET5_0_OR_GREATER
    /// <summary>
    /// Performs the operation on a 128-bit SIMD vector (SSE/NEON).
    /// </summary>
    /// <param name="x">The input vector containing 2 doubles or 4 floats.</param>
    /// <returns>The result vector with the operation applied element-wise.</returns>
    /// <remarks>
    /// Available on all modern x64 CPUs (SSE) and ARM64 CPUs (NEON).
    /// Processes 2 double values or 4 float values simultaneously.
    /// Only available on .NET 5+ / .NET 8.0.
    /// </remarks>
    Vector128<TVector> Invoke(Vector128<TVector> x);

    /// <summary>
    /// Performs the operation on a 256-bit SIMD vector (AVX2).
    /// </summary>
    /// <param name="x">The input vector containing 4 doubles or 8 floats.</param>
    /// <returns>The result vector with the operation applied element-wise.</returns>
    /// <remarks>
    /// Available on CPUs with AVX2 support (Intel Haswell 2013+, AMD Excavator 2015+).
    /// Processes 4 double values or 8 float values simultaneously.
    /// Only available on .NET 5+ / .NET 8.0.
    /// </remarks>
    Vector256<TVector> Invoke(Vector256<TVector> x);

    /// <summary>
    /// Performs the operation on a 512-bit SIMD vector (AVX-512).
    /// </summary>
    /// <param name="x">The input vector containing 8 doubles or 16 floats.</param>
    /// <returns>The result vector with the operation applied element-wise.</returns>
    /// <remarks>
    /// Available on CPUs with AVX-512 support (Intel Skylake-X 2017+, AMD Zen 4 2022+).
    /// Processes 8 double values or 16 float values simultaneously.
    /// Provides the highest throughput when available.
    /// Only available on .NET 5+ / .NET 8.0.
    /// </remarks>
    Vector512<TVector> Invoke(Vector512<TVector> x);
#endif
}
