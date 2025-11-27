using System;
using SystemVector = System.Numerics.Vector;
using System.Numerics.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Provides type-safe wrappers around TensorPrimitives for generic type T operations.
/// Uses SIMD-optimized implementations when available, falls back to manual loops otherwise.
/// </summary>
/// <typeparam name="T">The numeric type for tensor operations.</typeparam>
/// <remarks>
/// <para>
/// TensorPrimitives provides hardware-accelerated SIMD operations (SSE, AVX, AVX2, AVX-512) for
/// high-performance tensor computations.
/// </para>
/// <para><b>Performance Characteristics (float only):</b>
/// - Element-wise operations: 5-10ÃƒÆ’Ã¢â‚¬â€ speedup with AVX2
/// - Reductions (Sum, Max, Min): 8-12ÃƒÆ’Ã¢â‚¬â€ speedup
/// - Transcendentals (Exp, Log, Tanh): 3-6ÃƒÆ’Ã¢â‚¬â€ speedup
/// - Dot product: 10-15ÃƒÆ’Ã¢â‚¬â€ speedup on large vectors
///
/// <b>Threshold Recommendations:</b>
/// - Arrays &lt; 16 elements: Manual loops may be faster (overhead dominates)
/// - Arrays 16-10000: TensorPrimitives on CPU (optimal for float)
/// - Arrays &gt; 10000: Consider GPU (ILGPU) for maximum throughput
///
/// <b>Type Support:</b>
/// - float: Full SIMD optimization via TensorPrimitives
/// - double, other types: Fallback to INumericOperations (no SIMD)
/// </para>
/// </remarks>
public static class TensorPrimitivesHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Minimum array size threshold for using TensorPrimitives (below this, manual loops may be faster).
    /// </summary>
    private const int MinSizeForVectorization = 16;

    /// <summary>
    /// Cached flag indicating whether TensorPrimitives supports type T (float or double).
    /// TensorPrimitives provides SIMD-optimized operations for float and double only.
    /// </summary>
    private static readonly bool UseGenericTensorPrimitives = MathHelper.IsTensorPrimitivesSupported<T>();

    #region Vector Operations

    /// <summary>
    /// Performs element-wise addition.
    /// </summary>
    public static Vector<T> Add(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Vectors must have the same length");

        var xArray = x.ToArray();
        var yArray = y.ToArray();
        var result = new T[xArray.Length];

        if (xArray.Length >= MinSizeForVectorization && UseGenericTensorPrimitives)
        {
            TensorPrimitivesDispatcher.Add(xArray, yArray, result);
        }
        else
        {
            for (int i = 0; i < xArray.Length; i++)
                result[i] = NumOps.Add(xArray[i], yArray[i]);
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Performs element-wise subtraction.
    /// </summary>
    public static Vector<T> Subtract(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Vectors must have the same length");

        var xArray = x.ToArray();
        var yArray = y.ToArray();
        var result = new T[xArray.Length];

        if (xArray.Length >= MinSizeForVectorization && UseGenericTensorPrimitives)
        {
            TensorPrimitivesDispatcher.Subtract(xArray, yArray, result);
        }
        else
        {
            for (int i = 0; i < xArray.Length; i++)
                result[i] = NumOps.Subtract(xArray[i], yArray[i]);
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Performs element-wise multiplication.
    /// </summary>
    public static Vector<T> Multiply(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Vectors must have the same length");

        var xArray = x.ToArray();
        var yArray = y.ToArray();
        var result = new T[xArray.Length];

        if (xArray.Length >= MinSizeForVectorization && UseGenericTensorPrimitives)
        {
            TensorPrimitivesDispatcher.Multiply(xArray, yArray, result);
        }
        else
        {
            for (int i = 0; i < xArray.Length; i++)
                result[i] = NumOps.Multiply(xArray[i], yArray[i]);
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Performs element-wise division.
    /// </summary>
    public static Vector<T> Divide(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Vectors must have the same length");

        var xArray = x.ToArray();
        var yArray = y.ToArray();
        var result = new T[xArray.Length];

        if (xArray.Length >= MinSizeForVectorization && UseGenericTensorPrimitives)
        {
            TensorPrimitivesDispatcher.Divide(xArray, yArray, result);
        }
        else
        {
            for (int i = 0; i < xArray.Length; i++)
                result[i] = NumOps.Divide(xArray[i], yArray[i]);
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes dot product: sum(x[i] * y[i]).
    /// </summary>
    public static T Dot(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Vectors must have the same length");

        var xArray = x.ToArray();
        var yArray = y.ToArray();

        if (xArray.Length >= MinSizeForVectorization && UseGenericTensorPrimitives)
        {
            return TensorPrimitivesDispatcher.Dot(xArray, yArray);
        }

        // Fallback for non-accelerated types
        T fallbackResult = NumOps.Zero;
        for (int i = 0; i < xArray.Length; i++)
            fallbackResult = NumOps.Add(fallbackResult, NumOps.Multiply(xArray[i], yArray[i]));
        return fallbackResult;
    }

    /// <summary>
    /// Computes sum of all elements.
    /// </summary>
    public static T Sum(Vector<T> x)
    {
        var xArray = x.ToArray();

        if (xArray.Length >= MinSizeForVectorization && UseGenericTensorPrimitives)
        {
            return TensorPrimitivesDispatcher.Sum(xArray);
        }

        // Fallback for non-accelerated types
        T fallbackResult = NumOps.Zero;
        for (int i = 0; i < xArray.Length; i++)
            fallbackResult = NumOps.Add(fallbackResult, xArray[i]);
        return fallbackResult;
    }

    /// <summary>
    /// Finds maximum value.
    /// </summary>
    public static T Max(Vector<T> x)
    {
        if (x.Length == 0)
            throw new ArgumentException("Vector cannot be empty");

        var xArray = x.ToArray();

        if (xArray.Length >= MinSizeForVectorization && UseGenericTensorPrimitives)
        {
            return TensorPrimitivesDispatcher.Max(xArray);
        }

        // Fallback for non-accelerated types
        T max = xArray[0];
        for (int i = 1; i < xArray.Length; i++)
            if (NumOps.GreaterThan(xArray[i], max))
                max = xArray[i];
        return max;
    }

    /// <summary>
    /// Finds minimum value.
    /// </summary>
    public static T Min(Vector<T> x)
    {
        if (x.Length == 0)
            throw new ArgumentException("Vector cannot be empty");

        var xArray = x.ToArray();

        if (xArray.Length >= MinSizeForVectorization && UseGenericTensorPrimitives)
        {
            return TensorPrimitivesDispatcher.Min(xArray);
        }

        // Fallback for non-accelerated types
        T min = xArray[0];
        for (int i = 1; i < xArray.Length; i++)
            if (NumOps.LessThan(xArray[i], min))
                min = xArray[i];
        return min;
    }

    /// <summary>
    /// Computes exponential element-wise: exp(x).
    /// </summary>
    public static Vector<T> Exp(Vector<T> x)
    {
        var xArray = x.ToArray();
        var result = new T[xArray.Length];

        if (xArray.Length >= MinSizeForVectorization && UseGenericTensorPrimitives)
        {
            TensorPrimitivesDispatcher.Exp(xArray, result);
        }
        else
        {
            for (int i = 0; i < xArray.Length; i++)
                result[i] = NumOps.Exp(xArray[i]);
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes natural logarithm element-wise: log(x).
    /// </summary>
    public static Vector<T> Log(Vector<T> x)
    {
        var xArray = x.ToArray();
        var result = new T[xArray.Length];

        if (xArray.Length >= MinSizeForVectorization && UseGenericTensorPrimitives)
        {
            TensorPrimitivesDispatcher.Log(xArray, result);
        }
        else
        {
            for (int i = 0; i < xArray.Length; i++)
                result[i] = NumOps.Log(xArray[i]);
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes square root element-wise: sqrt(x).
    /// </summary>
    /// <remarks>
    /// TensorPrimitives.Sqrt is not available in all target frameworks (net462, net471, net472).
    /// Falls back to manual implementation using INumericOperations.
    /// </remarks>
    public static Vector<T> Sqrt(Vector<T> x)
    {
        var xArray = x.ToArray();
        var result = new T[xArray.Length];

        // TensorPrimitives.Sqrt not available in older frameworks
        // Use manual implementation for all types
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

        if (xArray.Length >= MinSizeForVectorization && UseGenericTensorPrimitives)
        {
            TensorPrimitivesDispatcher.Tanh(xArray, result);
        }
        else
        {
            // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
            for (int i = 0; i < xArray.Length; i++)
            {
                T twoX = NumOps.Multiply(NumOps.FromDouble(2.0), xArray[i]);
                T exp2x = NumOps.Exp(twoX);
                T numerator = NumOps.Subtract(exp2x, NumOps.One);
                T denominator = NumOps.Add(exp2x, NumOps.One);
                result[i] = NumOps.Divide(numerator, denominator);
            }
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes sigmoid element-wise: 1 / (1 + exp(-x)).
    /// </summary>
    public static Vector<T> Sigmoid(Vector<T> x)
    {
        var xArray = x.ToArray();
        var result = new T[xArray.Length];

        if (xArray.Length >= MinSizeForVectorization && UseGenericTensorPrimitives)
        {
            TensorPrimitivesDispatcher.Sigmoid(xArray, result);
        }
        else
        {
            for (int i = 0; i < xArray.Length; i++)
            {
                T negX = NumOps.Negate(xArray[i]);
                T expNegX = NumOps.Exp(negX);
                T onePlusExp = NumOps.Add(NumOps.One, expNegX);
                result[i] = NumOps.Divide(NumOps.One, onePlusExp);
            }
        }

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

        // Manual implementation (TensorPrimitives.LeakyReLU not available in 10.0.0)
        for (int i = 0; i < xArray.Length; i++)
        {
            result[i] = NumOps.GreaterThan(xArray[i], NumOps.Zero)
                ? xArray[i]
                : NumOps.Multiply(alphaT, xArray[i]);
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes GELU (Gaussian Error Linear Unit) element-wise: x * ÃƒÅ½Ã‚Â¦(x).
    /// Uses approximation: 0.5 * x * (1 + tanh(ÃƒÂ¢Ã‹â€ Ã…Â¡(2/ÃƒÂÃ¢â€šÂ¬) * (x + 0.044715 * xÃƒâ€šÃ‚Â³)))
    /// </summary>
    /// <param name="x">Input vector.</param>
    public static Vector<T> GELU(Vector<T> x)
    {
        var xArray = x.ToArray();
        var result = new T[xArray.Length];

        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        T sqrt2OverPi = NumOps.FromDouble(0.7978845608028654); // sqrt(2/pi)
        T coeff = NumOps.FromDouble(0.044715);
        T half = NumOps.FromDouble(0.5);

        for (int i = 0; i < xArray.Length; i++)
        {
            T x_val = xArray[i];
            T x_cubed = NumOps.Multiply(NumOps.Multiply(x_val, x_val), x_val);
            T inner = NumOps.Add(x_val, NumOps.Multiply(coeff, x_cubed));
            T tanh_arg = NumOps.Multiply(sqrt2OverPi, inner);

            // tanh(tanh_arg) = (exp(2*tanh_arg) - 1) / (exp(2*tanh_arg) + 1)
            T two_tanh_arg = NumOps.Multiply(NumOps.FromDouble(2.0), tanh_arg);
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

        for (int i = 0; i < xArray.Length; i++)
        {
            // softplus(x) = ln(1 + exp(x))
            T exp_x = NumOps.Exp(xArray[i]);
            T one_plus_exp = NumOps.Add(NumOps.One, exp_x);
            T softplus = NumOps.Log(one_plus_exp);

            // tanh(softplus)
            T two_softplus = NumOps.Multiply(NumOps.FromDouble(2.0), softplus);
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

        // Swish = x * sigmoid(x), compute sigmoid first then multiply
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

        if (xArray.Length >= MinSizeForVectorization && UseGenericTensorPrimitives)
        {
            TensorPrimitivesDispatcher.Log2(xArray, result);
        }
        else
        {
            // log2(x) = log(x) / log(2)
            T log2 = NumOps.Log(NumOps.FromDouble(2.0));
            for (int i = 0; i < xArray.Length; i++)
                result[i] = NumOps.Divide(NumOps.Log(xArray[i]), log2);
        }

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

        if (xArray.Length >= MinSizeForVectorization && UseGenericTensorPrimitives)
        {
            TensorPrimitivesDispatcher.SoftMax(xArray, result);
        }
        else
        {
            // Find max for numerical stability
            T max = xArray[0];
            for (int i = 1; i < xArray.Length; i++)
                if (NumOps.GreaterThan(xArray[i], max))
                    max = xArray[i];

            // Compute exp(x - max)
            T sum = NumOps.Zero;
            for (int i = 0; i < xArray.Length; i++)
            {
                T shifted = NumOps.Subtract(xArray[i], max);
                result[i] = NumOps.Exp(shifted);
                sum = NumOps.Add(sum, result[i]);
            }

            // Normalize
            for (int i = 0; i < xArray.Length; i++)
                result[i] = NumOps.Divide(result[i], sum);
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes cosine similarity: dot(a, b) / (norm(a) * norm(b)).
    /// </summary>
    public static T CosineSimilarity(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have the same length");

        var aArray = a.ToArray();
        var bArray = b.ToArray();

        if (aArray.Length >= MinSizeForVectorization && UseGenericTensorPrimitives)
        {
            return TensorPrimitivesDispatcher.CosineSimilarity(aArray, bArray);
        }
        else
        {
            // Compute dot product
            T dotProduct = NumOps.Zero;
            for (int i = 0; i < aArray.Length; i++)
                dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(aArray[i], bArray[i]));

            // Compute norms
            T normA = NumOps.Zero;
            T normB = NumOps.Zero;
            for (int i = 0; i < aArray.Length; i++)
            {
                normA = NumOps.Add(normA, NumOps.Multiply(aArray[i], aArray[i]));
                normB = NumOps.Add(normB, NumOps.Multiply(bArray[i], bArray[i]));
            }
            normA = NumOps.Sqrt(normA);
            normB = NumOps.Sqrt(normB);

            T denominator = NumOps.Multiply(normA, normB);
            if (NumOps.Equals(denominator, NumOps.Zero))
                return NumOps.Zero;

            return NumOps.Divide(dotProduct, denominator);
        }
    }

    #endregion
}
