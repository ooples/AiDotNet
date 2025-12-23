using System;
using System.Runtime.Intrinsics;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.Operators;

/// <summary>
/// Implements the inverse hyperbolic tangent function (atanh(x)) using hardware-accelerated SIMD instructions.
/// atanh(x) = 0.5 * log((1 + x) / (1 - x)), domain: (-1, 1)
/// </summary>
public readonly struct AtanhOperatorDouble : IUnaryOperator<double, double>
{
    /// <summary>
    /// Computes atanh(x) for a single double value.
    /// </summary>
    public double Invoke(double x)
    {
        return MathHelper.Atanh(x);
    }

    /// <summary>
    /// Computes atanh(x) for a Vector128 of doubles (2 values).
    /// </summary>
    public Vector128<double> Invoke(Vector128<double> x)
    {
        // Scalar fallback - can be optimized with 0.5 * log((1 + x) / (1 - x))
        Span<double> values = stackalloc double[Vector128<double>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = Invoke(values[i]);
        }

        return Vector128.Create(values[0], values[1]);
    }

    /// <summary>
    /// Computes atanh(x) for a Vector256 of doubles (4 values).
    /// </summary>
    public Vector256<double> Invoke(Vector256<double> x)
    {
        // Scalar fallback - can be optimized with 0.5 * log((1 + x) / (1 - x))
        Span<double> values = stackalloc double[Vector256<double>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = Invoke(values[i]);
        }

        return Vector256.Create(values[0], values[1], values[2], values[3]);
    }

    /// <summary>
    /// Computes atanh(x) for a Vector512 of doubles (8 values).
    /// </summary>
    public Vector512<double> Invoke(Vector512<double> x)
    {
        // Scalar fallback - can be optimized with 0.5 * log((1 + x) / (1 - x))
        Span<double> values = stackalloc double[Vector512<double>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = Invoke(values[i]);
        }

        return Vector512.Create(values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]);
    }
}

/// <summary>
/// Implements the inverse hyperbolic tangent function (atanh(x)) for single-precision floats using hardware-accelerated SIMD instructions.
/// atanh(x) = 0.5 * log((1 + x) / (1 - x)), domain: (-1, 1)
/// </summary>
public readonly struct AtanhOperatorFloat : IUnaryOperator<float, float>
{
    /// <summary>
    /// Computes atanh(x) for a single float value.
    /// </summary>
    public float Invoke(float x)
    {
        return (float)MathHelper.Atanh(x);
    }

#if NET5_0_OR_GREATER
    /// <summary>
    /// Computes atanh(x) for a Vector128 of floats (4 values).
    /// </summary>
    public Vector128<float> Invoke(Vector128<float> x)
    {
        // Scalar fallback - can be optimized with 0.5 * log((1 + x) / (1 - x))
        Span<float> values = stackalloc float[Vector128<float>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = Invoke(values[i]);
        }

        return Vector128.Create(values[0], values[1], values[2], values[3]);
    }

    /// <summary>
    /// Computes atanh(x) for a Vector256 of floats (8 values).
    /// </summary>
    public Vector256<float> Invoke(Vector256<float> x)
    {
        // Scalar fallback - can be optimized with 0.5 * log((1 + x) / (1 - x))
        Span<float> values = stackalloc float[Vector256<float>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = Invoke(values[i]);
        }

        return Vector256.Create(values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]);
    }

    /// <summary>
    /// Computes atanh(x) for a Vector512 of floats (16 values).
    /// </summary>
    public Vector512<float> Invoke(Vector512<float> x)
    {
        // Scalar fallback - can be optimized with 0.5 * log((1 + x) / (1 - x))
        Span<float> values = stackalloc float[Vector512<float>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = Invoke(values[i]);
        }

        return Vector512.Create(values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8], values[9], values[10], values[11], values[12], values[13], values[14], values[15]);
    }
#endif
}
