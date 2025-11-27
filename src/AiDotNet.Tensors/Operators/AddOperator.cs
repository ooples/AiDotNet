#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
#endif
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.Operators;

/// <summary>
/// Implements element-wise addition using hardware-accelerated SIMD instructions for double precision.
/// </summary>
/// <remarks>
/// <para>
/// This operator provides optimized implementations of x + y for:
/// - Scalar double (direct addition)
/// - Vector128 (SSE/NEON): 2 doubles
/// - Vector256 (AVX2): 4 doubles
/// - Vector512 (AVX-512): 8 doubles
/// </para>
/// <para>
/// <b>Performance:</b>
/// SIMD implementations provide 2-8x speedup over scalar addition for large arrays.
/// </para>
/// </remarks>
public readonly struct AddOperatorDouble : IBinaryOperator<double, double>
{
    /// <summary>
    /// Adds two double values.
    /// </summary>
    public double Invoke(double x, double y) => x + y;

#if NET5_0_OR_GREATER
    /// <summary>
    /// Adds two Vector128 of doubles (2 values each).
    /// </summary>
    public Vector128<double> Invoke(Vector128<double> x, Vector128<double> y)
        => Vector128.Add(x, y);

    /// <summary>
    /// Adds two Vector256 of doubles (4 values each).
    /// </summary>
    public Vector256<double> Invoke(Vector256<double> x, Vector256<double> y)
        => Vector256.Add(x, y);

    /// <summary>
    /// Adds two Vector512 of doubles (8 values each).
    /// </summary>
    public Vector512<double> Invoke(Vector512<double> x, Vector512<double> y)
        => Vector512.Add(x, y);
#endif
}

/// <summary>
/// Implements element-wise addition using hardware-accelerated SIMD instructions for single precision.
/// </summary>
public readonly struct AddOperatorFloat : IBinaryOperator<float, float>
{
    /// <summary>
    /// Adds two float values.
    /// </summary>
    public float Invoke(float x, float y) => x + y;

#if NET5_0_OR_GREATER
    /// <summary>
    /// Adds two Vector128 of floats (4 values each).
    /// </summary>
    public Vector128<float> Invoke(Vector128<float> x, Vector128<float> y)
        => Vector128.Add(x, y);

    /// <summary>
    /// Adds two Vector256 of floats (8 values each).
    /// </summary>
    public Vector256<float> Invoke(Vector256<float> x, Vector256<float> y)
        => Vector256.Add(x, y);

    /// <summary>
    /// Adds two Vector512 of floats (16 values each).
    /// </summary>
    public Vector512<float> Invoke(Vector512<float> x, Vector512<float> y)
        => Vector512.Add(x, y);
#endif
}

/// <summary>
/// Implements element-wise addition using hardware-accelerated SIMD instructions for integers.
/// </summary>
public readonly struct AddOperatorInt : IBinaryOperator<int, int>
{
    /// <summary>
    /// Adds two int values.
    /// </summary>
    public int Invoke(int x, int y) => x + y;

#if NET5_0_OR_GREATER
    /// <summary>
    /// Adds two Vector128 of ints (4 values each).
    /// </summary>
    public Vector128<int> Invoke(Vector128<int> x, Vector128<int> y)
        => Vector128.Add(x, y);

    /// <summary>
    /// Adds two Vector256 of ints (8 values each).
    /// </summary>
    public Vector256<int> Invoke(Vector256<int> x, Vector256<int> y)
        => Vector256.Add(x, y);

    /// <summary>
    /// Adds two Vector512 of ints (16 values each).
    /// </summary>
    public Vector512<int> Invoke(Vector512<int> x, Vector512<int> y)
        => Vector512.Add(x, y);
#endif
}

/// <summary>
/// Implements element-wise addition using hardware-accelerated SIMD instructions for long integers.
/// </summary>
public readonly struct AddOperatorLong : IBinaryOperator<long, long>
{
    /// <summary>
    /// Adds two long values.
    /// </summary>
    public long Invoke(long x, long y) => x + y;

#if NET5_0_OR_GREATER
    /// <summary>
    /// Adds two Vector128 of longs (2 values each).
    /// </summary>
    public Vector128<long> Invoke(Vector128<long> x, Vector128<long> y)
        => Vector128.Add(x, y);

    /// <summary>
    /// Adds two Vector256 of longs (4 values each).
    /// </summary>
    public Vector256<long> Invoke(Vector256<long> x, Vector256<long> y)
        => Vector256.Add(x, y);

    /// <summary>
    /// Adds two Vector512 of longs (8 values each).
    /// </summary>
    public Vector512<long> Invoke(Vector512<long> x, Vector512<long> y)
        => Vector512.Add(x, y);
#endif
}
