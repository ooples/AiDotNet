using System;
using System.Runtime.Intrinsics;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.Operators;

/// <summary>
/// Implements the inverse tangent function (arctan(x)) using hardware-accelerated SIMD instructions.
/// </summary>
/// <remarks>
/// <para>
/// This operator provides optimized implementations of atan(x) for:
/// - Scalar float/double (using Math.Atan for accuracy)
/// - Vector128 (SSE/NEON): 2 doubles or 4 floats
/// - Vector256 (AVX2): 4 doubles or 8 floats
/// - Vector512 (AVX-512): 8 doubles or 16 floats
/// </para>
/// <para>
/// <b>Input Domain:</b> (-Inf, Inf)
/// <b>Output Range:</b> (-π/2, π/2) radians
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// Uses scalar Math.Atan for accurate computation. SIMD implementations use
/// scalar fallback with per-element Math.Atan for reliability and numerical stability.
/// </para>
/// </remarks>
public readonly struct AtanOperatorDouble : IUnaryOperator<double, double>
{
    /// <summary>
    /// Computes atan(x) for a single double value.
    /// </summary>
    public double Invoke(double x) => Math.Atan(x);

#if NET5_0_OR_GREATER
    /// <summary>
    /// Computes atan(x) for a Vector128 of doubles (2 values).
    /// </summary>
    public Vector128<double> Invoke(Vector128<double> x)
    {
        // Scalar fallback for numerical accuracy - inverse trig functions
        // are best computed with scalar Math functions
        Span<double> values = stackalloc double[Vector128<double>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = Math.Atan(values[i]);
        }

        return Vector128.Create(values);
    }

    /// <summary>
    /// Computes atan(x) for a Vector256 of doubles (4 values).
    /// </summary>
    public Vector256<double> Invoke(Vector256<double> x)
    {
        // Scalar fallback for numerical accuracy
        Span<double> values = stackalloc double[Vector256<double>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = Math.Atan(values[i]);
        }

        return Vector256.Create(values);
    }

    /// <summary>
    /// Computes atan(x) for a Vector512 of doubles (8 values).
    /// </summary>
    public Vector512<double> Invoke(Vector512<double> x)
    {
        // Scalar fallback for numerical accuracy
        Span<double> values = stackalloc double[Vector512<double>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = Math.Atan(values[i]);
        }

        return Vector512.Create(values);
    }
#endif
}

/// <summary>
/// Implements the inverse tangent function (arctan(x)) for single-precision floats using hardware-accelerated SIMD instructions.
/// </summary>
/// <remarks>
/// <para>
/// Provides the same functionality as <see cref="AtanOperatorDouble"/> but optimized for single-precision (float) values.
/// Float operations are twice as wide as double operations at the same SIMD width:
/// - Vector128: 4 floats vs 2 doubles
/// - Vector256: 8 floats vs 4 doubles
/// - Vector512: 16 floats vs 8 doubles
/// </para>
/// </remarks>
public readonly struct AtanOperatorFloat : IUnaryOperator<float, float>
{
    /// <summary>
    /// Computes atan(x) for a single float value.
    /// </summary>
    public float Invoke(float x) => MathF.Atan(x);

#if NET5_0_OR_GREATER
    /// <summary>
    /// Computes atan(x) for a Vector128 of floats (4 values).
    /// </summary>
    public Vector128<float> Invoke(Vector128<float> x)
    {
        // Scalar fallback for numerical accuracy
        Span<float> values = stackalloc float[Vector128<float>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = MathF.Atan(values[i]);
        }

        return Vector128.Create(values);
    }

    /// <summary>
    /// Computes atan(x) for a Vector256 of floats (8 values).
    /// </summary>
    public Vector256<float> Invoke(Vector256<float> x)
    {
        // Scalar fallback for numerical accuracy
        Span<float> values = stackalloc float[Vector256<float>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = MathF.Atan(values[i]);
        }

        return Vector256.Create(values);
    }

    /// <summary>
    /// Computes atan(x) for a Vector512 of floats (16 values).
    /// </summary>
    public Vector512<float> Invoke(Vector512<float> x)
    {
        // Scalar fallback for numerical accuracy
        Span<float> values = stackalloc float[Vector512<float>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = MathF.Atan(values[i]);
        }

        return Vector512.Create(values);
    }
#endif
}
