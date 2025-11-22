using System;
using System.Runtime.Intrinsics;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.Operators;

/// <summary>
/// Implements the inverse hyperbolic sine function (asinh(x)) using hardware-accelerated SIMD instructions.
/// asinh(x) = log(x + sqrt(x^2 + 1))
/// </summary>
public readonly struct AsinhOperatorDouble : IUnaryOperator<double, double>
{
    /// <summary>
    /// Computes asinh(x) for a single double value.
    /// </summary>
    public double Invoke(double x)
    {
#if NET5_0_OR_GREATER
        return Math.Asinh(x);
#else
        // asinh(x) = log(x + sqrt(x^2 + 1))
        return Math.Log(x + Math.Sqrt(x * x + 1.0));
#endif
    }

    /// <summary>
    /// Computes asinh(x) for a Vector128 of doubles (2 values).
    /// </summary>
    public Vector128<double> Invoke(Vector128<double> x)
    {
        // Scalar fallback - can be optimized with log(x + sqrt(x^2 + 1))
        Span<double> values = stackalloc double[Vector128<double>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = Invoke(values[i]);
        }

        return Vector128.Create(values);
    }

    /// <summary>
    /// Computes asinh(x) for a Vector256 of doubles (4 values).
    /// </summary>
    public Vector256<double> Invoke(Vector256<double> x)
    {
        // Scalar fallback - can be optimized with log(x + sqrt(x^2 + 1))
        Span<double> values = stackalloc double[Vector256<double>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = Invoke(values[i]);
        }

        return Vector256.Create(values);
    }

    /// <summary>
    /// Computes asinh(x) for a Vector512 of doubles (8 values).
    /// </summary>
    public Vector512<double> Invoke(Vector512<double> x)
    {
        // Scalar fallback - can be optimized with log(x + sqrt(x^2 + 1))
        Span<double> values = stackalloc double[Vector512<double>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = Invoke(values[i]);
        }

        return Vector512.Create(values);
    }
}

/// <summary>
/// Implements the inverse hyperbolic sine function (asinh(x)) for single-precision floats using hardware-accelerated SIMD instructions.
/// asinh(x) = log(x + sqrt(x^2 + 1))
/// </summary>
public readonly struct AsinhOperatorFloat : IUnaryOperator<float, float>
{
    /// <summary>
    /// Computes asinh(x) for a single float value.
    /// </summary>
    public float Invoke(float x)
    {
#if NET5_0_OR_GREATER
        return MathF.Asinh(x);
#else
        // asinh(x) = log(x + sqrt(x^2 + 1))
        return (float)Math.Log(x + Math.Sqrt(x * x + 1f));
#endif
    }

#if NET5_0_OR_GREATER
    /// <summary>
    /// Computes asinh(x) for a Vector128 of floats (4 values).
    /// </summary>
    public Vector128<float> Invoke(Vector128<float> x)
    {
        // Scalar fallback - can be optimized with log(x + sqrt(x^2 + 1))
        Span<float> values = stackalloc float[Vector128<float>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = Invoke(values[i]);
        }

        return Vector128.Create(values);
    }

    /// <summary>
    /// Computes asinh(x) for a Vector256 of floats (8 values).
    /// </summary>
    public Vector256<float> Invoke(Vector256<float> x)
    {
        // Scalar fallback - can be optimized with log(x + sqrt(x^2 + 1))
        Span<float> values = stackalloc float[Vector256<float>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = Invoke(values[i]);
        }

        return Vector256.Create(values);
    }

    /// <summary>
    /// Computes asinh(x) for a Vector512 of floats (16 values).
    /// </summary>
    public Vector512<float> Invoke(Vector512<float> x)
    {
        // Scalar fallback - can be optimized with log(x + sqrt(x^2 + 1))
        Span<float> values = stackalloc float[Vector512<float>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = Invoke(values[i]);
        }

        return Vector512.Create(values);
    }
#endif
}
