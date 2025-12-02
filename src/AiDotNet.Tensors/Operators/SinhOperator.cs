using System;
using System.Runtime.Intrinsics;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.Operators;

/// <summary>
/// Implements the hyperbolic sine function (sinh(x)) using hardware-accelerated SIMD instructions.
/// </summary>
public readonly struct SinhOperatorDouble : IUnaryOperator<double, double>
{
    /// <summary>
    /// Computes sinh(x) for a single double value.
    /// </summary>
    public double Invoke(double x) => Math.Sinh(x);

#if NET5_0_OR_GREATER
    /// <summary>
    /// Computes sinh(x) for a Vector128 of doubles (2 values).
    /// </summary>
    public Vector128<double> Invoke(Vector128<double> x)
    {
        // Scalar fallback for now - can be optimized with (exp(x) - exp(-x)) / 2
        Span<double> values = stackalloc double[Vector128<double>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = Math.Sinh(values[i]);
        }

        return Vector128.Create(values);
    }

    /// <summary>
    /// Computes sinh(x) for a Vector256 of doubles (4 values).
    /// </summary>
    public Vector256<double> Invoke(Vector256<double> x)
    {
        // Scalar fallback for now - can be optimized with (exp(x) - exp(-x)) / 2
        Span<double> values = stackalloc double[Vector256<double>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = Math.Sinh(values[i]);
        }

        return Vector256.Create(values);
    }

    /// <summary>
    /// Computes sinh(x) for a Vector512 of doubles (8 values).
    /// </summary>
    public Vector512<double> Invoke(Vector512<double> x)
    {
        // Scalar fallback for now - can be optimized with (exp(x) - exp(-x)) / 2
        Span<double> values = stackalloc double[Vector512<double>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = Math.Sinh(values[i]);
        }

        return Vector512.Create(values);
    }
#endif
}

/// <summary>
/// Implements the hyperbolic sine function (sinh(x)) for single-precision floats using hardware-accelerated SIMD instructions.
/// </summary>
public readonly struct SinhOperatorFloat : IUnaryOperator<float, float>
{
    /// <summary>
    /// Computes sinh(x) for a single float value.
    /// </summary>
    public float Invoke(float x) => MathF.Sinh(x);

#if NET5_0_OR_GREATER
    /// <summary>
    /// Computes sinh(x) for a Vector128 of floats (4 values).
    /// </summary>
    public Vector128<float> Invoke(Vector128<float> x)
    {
        // Scalar fallback for now - can be optimized with (exp(x) - exp(-x)) / 2
        Span<float> values = stackalloc float[Vector128<float>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = MathF.Sinh(values[i]);
        }

        return Vector128.Create(values);
    }

    /// <summary>
    /// Computes sinh(x) for a Vector256 of floats (8 values).
    /// </summary>
    public Vector256<float> Invoke(Vector256<float> x)
    {
        // Scalar fallback for now - can be optimized with (exp(x) - exp(-x)) / 2
        Span<float> values = stackalloc float[Vector256<float>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = MathF.Sinh(values[i]);
        }

        return Vector256.Create(values);
    }

    /// <summary>
    /// Computes sinh(x) for a Vector512 of floats (16 values).
    /// </summary>
    public Vector512<float> Invoke(Vector512<float> x)
    {
        // Scalar fallback for now - can be optimized with (exp(x) - exp(-x)) / 2
        Span<float> values = stackalloc float[Vector512<float>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = MathF.Sinh(values[i]);
        }

        return Vector512.Create(values);
    }
#endif
}
