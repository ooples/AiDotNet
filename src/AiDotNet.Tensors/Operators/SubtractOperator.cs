#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
#endif
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.Operators;

/// <summary>
/// Implements element-wise subtraction using hardware-accelerated SIMD instructions for double precision.
/// </summary>
public readonly struct SubtractOperatorDouble : IBinaryOperator<double, double>
{
    public double Invoke(double x, double y) => x - y;

#if NET5_0_OR_GREATER
    public Vector128<double> Invoke(Vector128<double> x, Vector128<double> y)
        => Vector128.Subtract(x, y);

    public Vector256<double> Invoke(Vector256<double> x, Vector256<double> y)
        => Vector256.Subtract(x, y);

    public Vector512<double> Invoke(Vector512<double> x, Vector512<double> y)
        => Vector512.Subtract(x, y);
#endif
}

/// <summary>
/// Implements element-wise subtraction using hardware-accelerated SIMD instructions for single precision.
/// </summary>
public readonly struct SubtractOperatorFloat : IBinaryOperator<float, float>
{
    public float Invoke(float x, float y) => x - y;

#if NET5_0_OR_GREATER
    public Vector128<float> Invoke(Vector128<float> x, Vector128<float> y)
        => Vector128.Subtract(x, y);

    public Vector256<float> Invoke(Vector256<float> x, Vector256<float> y)
        => Vector256.Subtract(x, y);

    public Vector512<float> Invoke(Vector512<float> x, Vector512<float> y)
        => Vector512.Subtract(x, y);
#endif
}

/// <summary>
/// Implements element-wise subtraction using hardware-accelerated SIMD instructions for integers.
/// </summary>
public readonly struct SubtractOperatorInt : IBinaryOperator<int, int>
{
    public int Invoke(int x, int y) => x - y;

#if NET5_0_OR_GREATER
    public Vector128<int> Invoke(Vector128<int> x, Vector128<int> y)
        => Vector128.Subtract(x, y);

    public Vector256<int> Invoke(Vector256<int> x, Vector256<int> y)
        => Vector256.Subtract(x, y);

    public Vector512<int> Invoke(Vector512<int> x, Vector512<int> y)
        => Vector512.Subtract(x, y);
#endif
}

/// <summary>
/// Implements element-wise subtraction using hardware-accelerated SIMD instructions for long integers.
/// </summary>
public readonly struct SubtractOperatorLong : IBinaryOperator<long, long>
{
    public long Invoke(long x, long y) => x - y;

#if NET5_0_OR_GREATER
    public Vector128<long> Invoke(Vector128<long> x, Vector128<long> y)
        => Vector128.Subtract(x, y);

    public Vector256<long> Invoke(Vector256<long> x, Vector256<long> y)
        => Vector256.Subtract(x, y);

    public Vector512<long> Invoke(Vector512<long> x, Vector512<long> y)
        => Vector512.Subtract(x, y);
#endif
}
