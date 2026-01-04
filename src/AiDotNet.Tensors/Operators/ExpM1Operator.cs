using System;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
#endif
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.Operators;

/// <summary>
/// Implements the exp(x) - 1 function with numerical stability for small x values.
/// </summary>
/// <remarks>
/// <para>
/// This operator provides optimized implementations of exp(x) - 1 for:
/// - Scalar float/double (using Math.Exp for accuracy)
/// - Vector128 (SSE/NEON): 2 doubles or 4 floats
/// - Vector256 (AVX2): 4 doubles or 8 floats
/// - Vector512 (AVX-512): 8 doubles or 16 floats
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// For .NET 7.0+, uses double.ExpM1/float.ExpM1 which provides better numerical stability
/// for values near zero compared to Math.Exp(x) - 1.
/// For older frameworks, falls back to Math.Exp(x) - 1.
/// </para>
/// <para>
/// <b>Numerical Stability:</b>
/// When x is very small, exp(x) is close to 1, and subtracting 1 causes
/// catastrophic cancellation. ExpM1 avoids this by computing the result
/// more accurately using Taylor series expansion for small values.
/// </para>
/// <para>
/// <b>Use Cases:</b>
/// - Loss functions and probability computations where x is near zero
/// - Gradient calculations in neural networks
/// - Statistical distributions requiring accurate near-zero computation
/// </para>
/// </remarks>
public readonly struct ExpM1OperatorDouble : IUnaryOperator<double, double>
{
    /// <summary>
    /// Computes exp(x) - 1 for a single double value with numerical stability.
    /// </summary>
    /// <param name="x">Input value</param>
    /// <returns>exp(x) - 1 computed with numerical stability for small x</returns>
#if NET7_0_OR_GREATER
    public double Invoke(double x) => double.ExpM1(x);
#else
    public double Invoke(double x) => Math.Exp(x) - 1.0;
#endif

#if NET5_0_OR_GREATER
    /// <summary>
    /// Computes exp(x) - 1 for a Vector128 of doubles (2 values).
    /// </summary>
    /// <remarks>
    /// Uses scalar fallback. Future optimization can add polynomial approximation.
    /// </remarks>
    public Vector128<double> Invoke(Vector128<double> x)
    {
        Span<double> values = stackalloc double[Vector128<double>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
#if NET7_0_OR_GREATER
            values[i] = double.ExpM1(values[i]);
#else
            values[i] = Math.Exp(values[i]) - 1.0;
#endif
        }

        return Vector128.Create(values);
    }

    /// <summary>
    /// Computes exp(x) - 1 for a Vector256 of doubles (4 values).
    /// </summary>
    public Vector256<double> Invoke(Vector256<double> x)
    {
        Span<double> values = stackalloc double[Vector256<double>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
#if NET7_0_OR_GREATER
            values[i] = double.ExpM1(values[i]);
#else
            values[i] = Math.Exp(values[i]) - 1.0;
#endif
        }

        return Vector256.Create(values);
    }

    /// <summary>
    /// Computes exp(x) - 1 for a Vector512 of doubles (8 values).
    /// </summary>
    public Vector512<double> Invoke(Vector512<double> x)
    {
        Span<double> values = stackalloc double[Vector512<double>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
#if NET7_0_OR_GREATER
            values[i] = double.ExpM1(values[i]);
#else
            values[i] = Math.Exp(values[i]) - 1.0;
#endif
        }

        return Vector512.Create(values);
    }
#endif
}

/// <summary>
/// Implements the exp(x) - 1 function for single-precision floats with numerical stability for small x values.
/// </summary>
public readonly struct ExpM1OperatorFloat : IUnaryOperator<float, float>
{
    /// <summary>
    /// Computes exp(x) - 1 for a single float value with numerical stability.
    /// </summary>
    /// <param name="x">Input value</param>
    /// <returns>exp(x) - 1 computed with numerical stability for small x</returns>
#if NET7_0_OR_GREATER
    public float Invoke(float x) => float.ExpM1(x);
#else
    public float Invoke(float x) => MathF.Exp(x) - 1.0f;
#endif

#if NET5_0_OR_GREATER
    /// <summary>
    /// Computes exp(x) - 1 for a Vector128 of floats (4 values).
    /// </summary>
    public Vector128<float> Invoke(Vector128<float> x)
    {
        Span<float> values = stackalloc float[Vector128<float>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
#if NET7_0_OR_GREATER
            values[i] = float.ExpM1(values[i]);
#else
            values[i] = MathF.Exp(values[i]) - 1.0f;
#endif
        }

        return Vector128.Create(values);
    }

    /// <summary>
    /// Computes exp(x) - 1 for a Vector256 of floats (8 values).
    /// </summary>
    public Vector256<float> Invoke(Vector256<float> x)
    {
        Span<float> values = stackalloc float[Vector256<float>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
#if NET7_0_OR_GREATER
            values[i] = float.ExpM1(values[i]);
#else
            values[i] = MathF.Exp(values[i]) - 1.0f;
#endif
        }

        return Vector256.Create(values);
    }

    /// <summary>
    /// Computes exp(x) - 1 for a Vector512 of floats (16 values).
    /// </summary>
    public Vector512<float> Invoke(Vector512<float> x)
    {
        Span<float> values = stackalloc float[Vector512<float>.Count];
        x.CopyTo(values);

        for (int i = 0; i < values.Length; i++)
        {
#if NET7_0_OR_GREATER
            values[i] = float.ExpM1(values[i]);
#else
            values[i] = MathF.Exp(values[i]) - 1.0f;
#endif
        }

        return Vector512.Create(values);
    }
#endif
}
