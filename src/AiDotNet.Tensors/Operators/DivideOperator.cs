using System;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
#endif
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.Operators;

/// <summary>
/// Implements element-wise division using hardware-accelerated SIMD instructions for double precision.
/// </summary>
public readonly struct DivideOperatorDouble : IBinaryOperator<double, double>
{
    public double Invoke(double x, double y) => x / y;

#if NET5_0_OR_GREATER
    public Vector128<double> Invoke(Vector128<double> x, Vector128<double> y)
        => Vector128.Divide(x, y);

    public Vector256<double> Invoke(Vector256<double> x, Vector256<double> y)
        => Vector256.Divide(x, y);

    public Vector512<double> Invoke(Vector512<double> x, Vector512<double> y)
        => Vector512.Divide(x, y);
#endif
}

/// <summary>
/// Implements element-wise division using hardware-accelerated SIMD instructions for single precision.
/// </summary>
public readonly struct DivideOperatorFloat : IBinaryOperator<float, float>
{
    public float Invoke(float x, float y) => x / y;

#if NET5_0_OR_GREATER
    public Vector128<float> Invoke(Vector128<float> x, Vector128<float> y)
        => Vector128.Divide(x, y);

    public Vector256<float> Invoke(Vector256<float> x, Vector256<float> y)
        => Vector256.Divide(x, y);

    public Vector512<float> Invoke(Vector512<float> x, Vector512<float> y)
        => Vector512.Divide(x, y);
#endif
}

/// <summary>
/// Implements element-wise division for integers.
/// </summary>
/// <remarks>
/// Integer division doesn't have direct SIMD support, so this falls back to scalar operations
/// within the vector processing loop for optimal cache utilization.
/// </remarks>
public readonly struct DivideOperatorInt : IBinaryOperator<int, int>
{
    public int Invoke(int x, int y) => x / y;

#if NET5_0_OR_GREATER
    public Vector128<int> Invoke(Vector128<int> x, Vector128<int> y)
    {
        // Integer division doesn't have direct SIMD support
        Span<int> xValues = stackalloc int[Vector128<int>.Count];
        Span<int> yValues = stackalloc int[Vector128<int>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] /= yValues[i];

        return Vector128.Create(xValues);
    }

    public Vector256<int> Invoke(Vector256<int> x, Vector256<int> y)
    {
        Span<int> xValues = stackalloc int[Vector256<int>.Count];
        Span<int> yValues = stackalloc int[Vector256<int>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] /= yValues[i];

        return Vector256.Create(xValues);
    }

    public Vector512<int> Invoke(Vector512<int> x, Vector512<int> y)
    {
        Span<int> xValues = stackalloc int[Vector512<int>.Count];
        Span<int> yValues = stackalloc int[Vector512<int>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] /= yValues[i];

        return Vector512.Create(xValues);
    }
#endif
}

/// <summary>
/// Implements element-wise division for long integers.
/// </summary>
/// <remarks>
/// Integer division doesn't have direct SIMD support, so this falls back to scalar operations
/// within the vector processing loop for optimal cache utilization.
/// </remarks>
public readonly struct DivideOperatorLong : IBinaryOperator<long, long>
{
    public long Invoke(long x, long y) => x / y;

#if NET5_0_OR_GREATER
    public Vector128<long> Invoke(Vector128<long> x, Vector128<long> y)
    {
        Span<long> xValues = stackalloc long[Vector128<long>.Count];
        Span<long> yValues = stackalloc long[Vector128<long>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] /= yValues[i];

        return Vector128.Create(xValues);
    }

    public Vector256<long> Invoke(Vector256<long> x, Vector256<long> y)
    {
        Span<long> xValues = stackalloc long[Vector256<long>.Count];
        Span<long> yValues = stackalloc long[Vector256<long>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] /= yValues[i];

        return Vector256.Create(xValues);
    }

    public Vector512<long> Invoke(Vector512<long> x, Vector512<long> y)
    {
        Span<long> xValues = stackalloc long[Vector512<long>.Count];
        Span<long> yValues = stackalloc long[Vector512<long>.Count];
        x.CopyTo(xValues);
        y.CopyTo(yValues);

        for (int i = 0; i < xValues.Length; i++)
            xValues[i] /= yValues[i];

        return Vector512.Create(xValues);
    }
#endif
}
