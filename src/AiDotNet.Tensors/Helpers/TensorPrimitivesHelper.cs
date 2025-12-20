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
    /// Falls back to scalar implementation using INumericOperations.Sqrt.
    /// </remarks>
    public static Vector<T> Sqrt(Vector<T> x)
    {
        var xArray = x.ToArray();
        var result = new T[xArray.Length];

        // Use scalar Sqrt - no vectorized version available in the interface
        for (int i = 0; i < xArray.Length; i++)
            result[i] = NumOps.Sqrt(xArray[i]);

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
    /// <param name="x">Input vector.</param>
    /// <param name="alpha">Negative slope coefficient (typically 0.01).</param>
    public static Vector<T> LeakyReLU(Vector<T> x, double alpha = 0.01)
    {
        var xArray = x.ToArray();
        var result = new T[xArray.Length];
        T alphaT = NumOps.FromDouble(alpha);

        for (int i = 0; i < xArray.Length; i++)
        {
            result[i] = NumOps.GreaterThan(xArray[i], NumOps.Zero)
                ? xArray[i]
                : NumOps.Multiply(alphaT, xArray[i]);
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes GELU (Gaussian Error Linear Unit) element-wise.
    /// Uses approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    /// </summary>
    /// <param name="x">Input vector.</param>
    public static Vector<T> GELU(Vector<T> x)
    {
        var xArray = x.ToArray();
        var result = new T[xArray.Length];

        T sqrt2OverPi = NumOps.FromDouble(0.7978845608028654); // sqrt(2/pi)
        T coeff = NumOps.FromDouble(0.044715);
        T half = NumOps.FromDouble(0.5);
        T two = NumOps.FromDouble(2.0);

        for (int i = 0; i < xArray.Length; i++)
        {
            T x_val = xArray[i];
            T x_cubed = NumOps.Multiply(NumOps.Multiply(x_val, x_val), x_val);
            T inner = NumOps.Add(x_val, NumOps.Multiply(coeff, x_cubed));
            T tanh_arg = NumOps.Multiply(sqrt2OverPi, inner);

            // tanh(tanh_arg)
            T two_tanh_arg = NumOps.Multiply(two, tanh_arg);
            T exp_val = NumOps.Exp(two_tanh_arg);
            T tanh_val = NumOps.Divide(
                NumOps.Subtract(exp_val, NumOps.One),
                NumOps.Add(exp_val, NumOps.One)
            );

            T one_plus_tanh = NumOps.Add(NumOps.One, tanh_val);
            result[i] = NumOps.Multiply(NumOps.Multiply(half, x_val), one_plus_tanh);
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes Mish activation element-wise: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x))).
    /// </summary>
    /// <param name="x">Input vector.</param>
    public static Vector<T> Mish(Vector<T> x)
    {
        var xArray = x.ToArray();
        var result = new T[xArray.Length];
        T two = NumOps.FromDouble(2.0);

        for (int i = 0; i < xArray.Length; i++)
        {
            // softplus(x) = ln(1 + exp(x))
            T exp_x = NumOps.Exp(xArray[i]);
            T one_plus_exp = NumOps.Add(NumOps.One, exp_x);
            T softplus = NumOps.Log(one_plus_exp);

            // tanh(softplus)
            T two_softplus = NumOps.Multiply(two, softplus);
            T exp_2softplus = NumOps.Exp(two_softplus);
            T tanh_softplus = NumOps.Divide(
                NumOps.Subtract(exp_2softplus, NumOps.One),
                NumOps.Add(exp_2softplus, NumOps.One)
            );

            // x * tanh(softplus(x))
            result[i] = NumOps.Multiply(xArray[i], tanh_softplus);
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes Swish/SiLU activation element-wise: x * sigmoid(x) = x / (1 + exp(-x)).
    /// </summary>
    /// <param name="x">Input vector.</param>
    public static Vector<T> Swish(Vector<T> x)
    {
        var xArray = x.ToArray();
        var result = new T[xArray.Length];

        for (int i = 0; i < xArray.Length; i++)
        {
            T neg_x = NumOps.Negate(xArray[i]);
            T exp_neg_x = NumOps.Exp(neg_x);
            T sigmoid = NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, exp_neg_x));
            result[i] = NumOps.Multiply(xArray[i], sigmoid);
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes ELU (Exponential Linear Unit) element-wise: x if x > 0, alpha * (exp(x) - 1) otherwise.
    /// </summary>
    /// <param name="x">Input vector.</param>
    /// <param name="alpha">Scale factor for negative values (typically 1.0).</param>
    public static Vector<T> ELU(Vector<T> x, double alpha = 1.0)
    {
        var xArray = x.ToArray();
        var result = new T[xArray.Length];
        T alphaT = NumOps.FromDouble(alpha);

        for (int i = 0; i < xArray.Length; i++)
        {
            if (NumOps.GreaterThan(xArray[i], NumOps.Zero))
            {
                result[i] = xArray[i];
            }
            else
            {
                T exp_x = NumOps.Exp(xArray[i]);
                T exp_minus_one = NumOps.Subtract(exp_x, NumOps.One);
                result[i] = NumOps.Multiply(alphaT, exp_minus_one);
            }
        }

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

    #endregion
}
