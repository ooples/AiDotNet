using System;
using System.Runtime.Intrinsics;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.Operators;

/// <summary>
/// Implements the reciprocal function (1/x) using hardware-accelerated SIMD instructions.
/// </summary>
public readonly struct ReciprocalOperatorDouble : IUnaryOperator<double, double>
{
    /// <summary>
    /// Computes 1/x for a single double value.
    /// </summary>
    public double Invoke(double x) => 1.0 / x;

#if NET5_0_OR_GREATER
    /// <summary>
    /// Computes 1/x for a Vector128 of doubles (2 values).
    /// </summary>
    public Vector128<double> Invoke(Vector128<double> x)
    {
        Vector128<double> one = Vector128.Create(1.0);
        return Vector128.Divide(one, x);
    }

    /// <summary>
    /// Computes 1/x for a Vector256 of doubles (4 values).
    /// </summary>
    public Vector256<double> Invoke(Vector256<double> x)
    {
        Vector256<double> one = Vector256.Create(1.0);
        return Vector256.Divide(one, x);
    }

    /// <summary>
    /// Computes 1/x for a Vector512 of doubles (8 values).
    /// </summary>
    public Vector512<double> Invoke(Vector512<double> x)
    {
        Vector512<double> one = Vector512.Create(1.0);
        return Vector512.Divide(one, x);
    }
#endif
}

/// <summary>
/// Implements the reciprocal function (1/x) for single-precision floats using hardware-accelerated SIMD instructions.
/// </summary>
public readonly struct ReciprocalOperatorFloat : IUnaryOperator<float, float>
{
    /// <summary>
    /// Computes 1/x for a single float value.
    /// </summary>
    public float Invoke(float x) => 1.0f / x;

#if NET5_0_OR_GREATER
    /// <summary>
    /// Computes 1/x for a Vector128 of floats (4 values).
    /// </summary>
    public Vector128<float> Invoke(Vector128<float> x)
    {
        Vector128<float> one = Vector128.Create(1.0f);
        return Vector128.Divide(one, x);
    }

    /// <summary>
    /// Computes 1/x for a Vector256 of floats (8 values).
    /// </summary>
    public Vector256<float> Invoke(Vector256<float> x)
    {
        Vector256<float> one = Vector256.Create(1.0f);
        return Vector256.Divide(one, x);
    }

    /// <summary>
    /// Computes 1/x for a Vector512 of floats (16 values).
    /// </summary>
    public Vector512<float> Invoke(Vector512<float> x)
    {
        Vector512<float> one = Vector512.Create(1.0f);
        return Vector512.Divide(one, x);
    }
#endif
}
