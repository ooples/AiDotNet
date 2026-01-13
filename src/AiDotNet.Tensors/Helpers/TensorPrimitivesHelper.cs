using System;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using static AiDotNet.Tensors.ErrorMessages;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Provides type-safe wrappers around vectorized operations for generic type T.
/// Uses SIMD-optimized implementations when available (float, double), falls back to sequential loops otherwise.
/// </summary>
/// <typeparam name="T">The numeric type for tensor operations.</typeparam>
/// <remarks>
/// <para>
/// This helper class leverages the polymorphic IVectorizedOperations interface to provide
/// hardware-accelerated operations. Float and double types use TensorPrimitives for SIMD
/// acceleration (SSE, AVX, AVX2, AVX-512), while other types use sequential fallback implementations.
/// </para>
/// <para><b>Performance Characteristics:</b>
/// - float/double: 5-15x speedup via SIMD (TensorPrimitives)
/// - Other types: Sequential loops (no SIMD)
///
/// <b>Design:</b>
/// The dispatch is handled via polymorphism through INumericOperations, which extends
/// IVectorizedOperations. Each numeric type implementation provides its own optimized
/// vectorized operations, following the Open/Closed principle.
/// </para>
/// </remarks>
public static class TensorPrimitivesHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    #region Vector Operations

    /// <summary>
    /// Performs element-wise addition.
    /// </summary>
    public static Vector<T> Add(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException(VectorsSameLength);

        var xArray = x.ToArray();
        var yArray = y.ToArray();
        var result = new T[xArray.Length];

        NumOps.Add(xArray, yArray, result);

        return new Vector<T>(result);
    }

    /// <summary>
    /// Performs element-wise subtraction.
    /// </summary>
    public static Vector<T> Subtract(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException(VectorsSameLength);

        var xArray = x.ToArray();
        var yArray = y.ToArray();
        var result = new T[xArray.Length];

        NumOps.Subtract(xArray, yArray, result);

        return new Vector<T>(result);
    }

    /// <summary>
    /// Performs element-wise multiplication.
    /// </summary>
    public static Vector<T> Multiply(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException(VectorsSameLength);

        var xArray = x.ToArray();
        var yArray = y.ToArray();
        var result = new T[xArray.Length];

        NumOps.Multiply(xArray, yArray, result);

        return new Vector<T>(result);
    }

    /// <summary>
    /// Performs element-wise division.
    /// </summary>
    public static Vector<T> Divide(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException(VectorsSameLength);

        var xArray = x.ToArray();
        var yArray = y.ToArray();
        var result = new T[xArray.Length];

        NumOps.Divide(xArray, yArray, result);

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes dot product: sum(x[i] * y[i]).
    /// </summary>
    public static T Dot(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException(VectorsSameLength);

        var xArray = x.ToArray();
        var yArray = y.ToArray();

        return NumOps.Dot(xArray, yArray);
    }

    /// <summary>
    /// Computes sum of all elements.
    /// </summary>
    public static T Sum(Vector<T> x)
    {
        var xArray = x.ToArray();
        return NumOps.Sum(xArray);
    }

    /// <summary>
    /// Finds maximum value.
    /// </summary>
    public static T Max(Vector<T> x)
    {
        if (x.Length == 0)
            throw new ArgumentException(VectorCannotBeEmpty);

        var xArray = x.ToArray();
        return NumOps.Max(xArray);
    }

    /// <summary>
    /// Finds minimum value.
    /// </summary>
    public static T Min(Vector<T> x)
    {
        if (x.Length == 0)
            throw new ArgumentException(VectorCannotBeEmpty);

        var xArray = x.ToArray();
        return NumOps.Min(xArray);
    }

    /// <summary>
    /// Computes exponential element-wise: exp(x).
    /// </summary>
    public static Vector<T> Exp(Vector<T> x)
    {
        var xArray = x.ToArray();
        var result = new T[xArray.Length];

        NumOps.Exp(xArray, result);

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes natural logarithm element-wise: log(x).
    /// </summary>
    public static Vector<T> Log(Vector<T> x)
    {
        var xArray = x.ToArray();
        var result = new T[xArray.Length];

        NumOps.Log(xArray, result);

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes square root element-wise: sqrt(x).
    /// </summary>
    /// <remarks>
    /// Uses SIMD-optimized vectorized operations for float/double types via TensorPrimitives.
    /// </remarks>
    public static Vector<T> Sqrt(Vector<T> x)
    {
        var xArray = x.ToArray();
        var result = new T[xArray.Length];

        NumOps.Sqrt(xArray, result);

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes hyperbolic tangent element-wise: tanh(x).
    /// </summary>
    public static Vector<T> Tanh(Vector<T> x)
    {
        var xArray = x.ToArray();
        var result = new T[xArray.Length];

        NumOps.Tanh(xArray, result);

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes sigmoid element-wise: 1 / (1 + exp(-x)).
    /// </summary>
    public static Vector<T> Sigmoid(Vector<T> x)
    {
        var xArray = x.ToArray();
        var result = new T[xArray.Length];

        NumOps.Sigmoid(xArray, result);

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes LeakyReLU element-wise: x if x > 0, alpha * x otherwise.
    /// </summary>
    /// <remarks>
    /// Uses SIMD-optimized vectorized operations for float/double types via hardware intrinsics.
    /// </remarks>
    /// <param name="x">Input vector.</param>
    /// <param name="alpha">Negative slope coefficient (typically 0.01).</param>
    public static Vector<T> LeakyReLU(Vector<T> x, double alpha = 0.01)
    {
        var xArray = x.ToArray();
        var result = new T[xArray.Length];
        T alphaT = NumOps.FromDouble(alpha);

        NumOps.LeakyReLU(xArray, alphaT, result);

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes GELU (Gaussian Error Linear Unit) element-wise.
    /// Uses approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    /// </summary>
    /// <remarks>
    /// Uses SIMD-optimized vectorized operations for float/double types.
    /// </remarks>
    /// <param name="x">Input vector.</param>
    public static Vector<T> GELU(Vector<T> x)
    {
        var xArray = x.ToArray();
        var result = new T[xArray.Length];

        NumOps.GELU(xArray, result);

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes Mish activation element-wise: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x))).
    /// </summary>
    /// <remarks>
    /// Uses SIMD-optimized vectorized operations for float/double types.
    /// </remarks>
    /// <param name="x">Input vector.</param>
    public static Vector<T> Mish(Vector<T> x)
    {
        var xArray = x.ToArray();
        var result = new T[xArray.Length];

        NumOps.Mish(xArray, result);

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes Swish/SiLU activation element-wise: x * sigmoid(x) = x / (1 + exp(-x)).
    /// </summary>
    /// <remarks>
    /// Uses SIMD-optimized vectorized operations for float/double types.
    /// </remarks>
    /// <param name="x">Input vector.</param>
    public static Vector<T> Swish(Vector<T> x)
    {
        var xArray = x.ToArray();
        var result = new T[xArray.Length];

        NumOps.Swish(xArray, result);

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes ELU (Exponential Linear Unit) element-wise: x if x > 0, alpha * (exp(x) - 1) otherwise.
    /// </summary>
    /// <remarks>
    /// Uses SIMD-optimized vectorized operations for float/double types via hardware intrinsics.
    /// </remarks>
    /// <param name="x">Input vector.</param>
    /// <param name="alpha">Scale factor for negative values (typically 1.0).</param>
    public static Vector<T> ELU(Vector<T> x, double alpha = 1.0)
    {
        var xArray = x.ToArray();
        var result = new T[xArray.Length];
        T alphaT = NumOps.FromDouble(alpha);

        NumOps.ELU(xArray, alphaT, result);

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes base-2 logarithm element-wise: log2(x).
    /// </summary>
    public static Vector<T> Log2(Vector<T> x)
    {
        var xArray = x.ToArray();
        var result = new T[xArray.Length];

        NumOps.Log2(xArray, result);

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes softmax: exp(x - max) / sum(exp(x - max)).
    /// </summary>
    public static Vector<T> Softmax(Vector<T> x)
    {
        var xArray = x.ToArray();
        if (xArray.Length == 0)
            throw new ArgumentException("Vector cannot be empty", nameof(x));

        var result = new T[xArray.Length];

        NumOps.SoftMax(xArray, result);

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes cosine similarity: dot(a, b) / (norm(a) * norm(b)).
    /// </summary>
    public static T CosineSimilarity(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException(VectorsSameLength);

        var aArray = a.ToArray();
        var bArray = b.ToArray();

        return NumOps.CosineSimilarity(aArray, bArray);
    }

    /// <summary>
    /// Checks if any element in the vector is NaN or Infinity.
    /// </summary>
    /// <param name="x">The source vector.</param>
    /// <param name="badIndex">The index of the first non-finite value found, or -1 if all are finite.</param>
    /// <returns>True if any element is non-finite, false otherwise.</returns>
    public static bool IsAnyNonFinite(Vector<T> x, out int badIndex)
    {
        return NumOps.IsAnyNonFinite(x.AsSpan(), out badIndex);
    }

    #endregion
}
