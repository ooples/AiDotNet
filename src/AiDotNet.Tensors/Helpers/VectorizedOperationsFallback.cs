using System;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Provides generic fallback implementations for vectorized operations using sequential loops.
/// Used by numeric types that don't have SIMD-optimized implementations via TensorPrimitives.
/// </summary>
/// <remarks>
/// <para>
/// This helper class provides loop-based implementations of all IVectorizedOperations methods.
/// These implementations work for any numeric type T that has an INumericOperations implementation,
/// but they don't benefit from SIMD acceleration.
/// </para>
/// <para>
/// <b>Performance Note:</b> These fallback implementations are significantly slower than
/// SIMD-optimized versions (5-15x for typical operations). Use them only when TensorPrimitives
/// doesn't support the numeric type (e.g., Half, decimal, Complex).
/// </para>
/// </remarks>
internal static class VectorizedOperationsFallback
{
    /// <summary>
    /// Performs element-wise addition using sequential loops.
    /// </summary>
    public static void Add<T>(INumericOperations<T> ops, ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
    {
        if (x.Length != y.Length || x.Length != destination.Length)
            throw new ArgumentException("All spans must have the same length");

        for (int i = 0; i < x.Length; i++)
            destination[i] = ops.Add(x[i], y[i]);
    }

    /// <summary>
    /// Performs element-wise subtraction using sequential loops.
    /// </summary>
    public static void Subtract<T>(INumericOperations<T> ops, ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
    {
        if (x.Length != y.Length || x.Length != destination.Length)
            throw new ArgumentException("All spans must have the same length");

        for (int i = 0; i < x.Length; i++)
            destination[i] = ops.Subtract(x[i], y[i]);
    }

    /// <summary>
    /// Performs element-wise multiplication using sequential loops.
    /// </summary>
    public static void Multiply<T>(INumericOperations<T> ops, ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
    {
        if (x.Length != y.Length || x.Length != destination.Length)
            throw new ArgumentException("All spans must have the same length");

        for (int i = 0; i < x.Length; i++)
            destination[i] = ops.Multiply(x[i], y[i]);
    }

    /// <summary>
    /// Performs element-wise division using sequential loops.
    /// </summary>
    public static void Divide<T>(INumericOperations<T> ops, ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
    {
        if (x.Length != y.Length || x.Length != destination.Length)
            throw new ArgumentException("All spans must have the same length");

        for (int i = 0; i < x.Length; i++)
            destination[i] = ops.Divide(x[i], y[i]);
    }

    /// <summary>
    /// Computes dot product using sequential loops.
    /// </summary>
    public static T Dot<T>(INumericOperations<T> ops, ReadOnlySpan<T> x, ReadOnlySpan<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Spans must have the same length");

        T result = ops.Zero;
        for (int i = 0; i < x.Length; i++)
            result = ops.Add(result, ops.Multiply(x[i], y[i]));
        return result;
    }

    /// <summary>
    /// Computes sum using sequential loops.
    /// </summary>
    public static T Sum<T>(INumericOperations<T> ops, ReadOnlySpan<T> x)
    {
        T result = ops.Zero;
        for (int i = 0; i < x.Length; i++)
            result = ops.Add(result, x[i]);
        return result;
    }

    /// <summary>
    /// Finds maximum using sequential loops.
    /// </summary>
    public static T Max<T>(INumericOperations<T> ops, ReadOnlySpan<T> x)
    {
        if (x.Length == 0)
            throw new ArgumentException("Span cannot be empty");

        T max = x[0];
        for (int i = 1; i < x.Length; i++)
            if (ops.GreaterThan(x[i], max))
                max = x[i];
        return max;
    }

    /// <summary>
    /// Finds minimum using sequential loops.
    /// </summary>
    public static T Min<T>(INumericOperations<T> ops, ReadOnlySpan<T> x)
    {
        if (x.Length == 0)
            throw new ArgumentException("Span cannot be empty");

        T min = x[0];
        for (int i = 1; i < x.Length; i++)
            if (ops.LessThan(x[i], min))
                min = x[i];
        return min;
    }

    /// <summary>
    /// Computes exponential using sequential loops.
    /// </summary>
    public static void Exp<T>(INumericOperations<T> ops, ReadOnlySpan<T> x, Span<T> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        for (int i = 0; i < x.Length; i++)
            destination[i] = ops.Exp(x[i]);
    }

    /// <summary>
    /// Computes natural logarithm using sequential loops.
    /// </summary>
    public static void Log<T>(INumericOperations<T> ops, ReadOnlySpan<T> x, Span<T> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        for (int i = 0; i < x.Length; i++)
            destination[i] = ops.Log(x[i]);
    }

    /// <summary>
    /// Computes hyperbolic tangent using sequential loops.
    /// </summary>
    public static void Tanh<T>(INumericOperations<T> ops, ReadOnlySpan<T> x, Span<T> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        T two = ops.FromDouble(2.0);
        for (int i = 0; i < x.Length; i++)
        {
            T twoX = ops.Multiply(two, x[i]);
            T exp2x = ops.Exp(twoX);
            T numerator = ops.Subtract(exp2x, ops.One);
            T denominator = ops.Add(exp2x, ops.One);
            destination[i] = ops.Divide(numerator, denominator);
        }
    }

    /// <summary>
    /// Computes sigmoid using sequential loops.
    /// </summary>
    public static void Sigmoid<T>(INumericOperations<T> ops, ReadOnlySpan<T> x, Span<T> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        // sigmoid(x) = 1 / (1 + exp(-x))
        for (int i = 0; i < x.Length; i++)
        {
            T negX = ops.Negate(x[i]);
            T expNegX = ops.Exp(negX);
            T onePlusExp = ops.Add(ops.One, expNegX);
            destination[i] = ops.Divide(ops.One, onePlusExp);
        }
    }

    /// <summary>
    /// Computes base-2 logarithm using sequential loops.
    /// </summary>
    public static void Log2<T>(INumericOperations<T> ops, ReadOnlySpan<T> x, Span<T> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        // log2(x) = log(x) / log(2)
        T log2 = ops.Log(ops.FromDouble(2.0));
        for (int i = 0; i < x.Length; i++)
            destination[i] = ops.Divide(ops.Log(x[i]), log2);
    }

    /// <summary>
    /// Computes softmax using sequential loops.
    /// </summary>
    public static void SoftMax<T>(INumericOperations<T> ops, ReadOnlySpan<T> x, Span<T> destination)
    {
        if (x.Length == 0)
            throw new ArgumentException("Span cannot be empty");
        if (x.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        // Find max for numerical stability
        T max = x[0];
        for (int i = 1; i < x.Length; i++)
            if (ops.GreaterThan(x[i], max))
                max = x[i];

        // Compute exp(x - max) and sum
        T sum = ops.Zero;
        for (int i = 0; i < x.Length; i++)
        {
            T shifted = ops.Subtract(x[i], max);
            destination[i] = ops.Exp(shifted);
            sum = ops.Add(sum, destination[i]);
        }

        // Normalize
        for (int i = 0; i < x.Length; i++)
            destination[i] = ops.Divide(destination[i], sum);
    }

    /// <summary>
    /// Computes cosine similarity using sequential loops.
    /// </summary>
    public static T CosineSimilarity<T>(INumericOperations<T> ops, ReadOnlySpan<T> x, ReadOnlySpan<T> y)
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Spans must have the same length");

        // Compute dot product
        T dotProduct = ops.Zero;
        for (int i = 0; i < x.Length; i++)
            dotProduct = ops.Add(dotProduct, ops.Multiply(x[i], y[i]));

        // Compute norms
        T normX = ops.Zero;
        T normY = ops.Zero;
        for (int i = 0; i < x.Length; i++)
        {
            normX = ops.Add(normX, ops.Multiply(x[i], x[i]));
            normY = ops.Add(normY, ops.Multiply(y[i], y[i]));
        }
        normX = ops.Sqrt(normX);
        normY = ops.Sqrt(normY);

        T denominator = ops.Multiply(normX, normY);
        if (ops.Equals(denominator, ops.Zero))
            return ops.Zero;

        return ops.Divide(dotProduct, denominator);
    }

    /// <summary>
    /// Fills the destination span with a constant value using sequential loops.
    /// </summary>
    public static void Fill<T>(INumericOperations<T> ops, Span<T> destination, T value)
    {
        for (int i = 0; i < destination.Length; i++)
            destination[i] = value;
    }

    /// <summary>
    /// Multiplies each element by a scalar using sequential loops.
    /// </summary>
    public static void MultiplyScalar<T>(INumericOperations<T> ops, ReadOnlySpan<T> x, T scalar, Span<T> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        for (int i = 0; i < x.Length; i++)
            destination[i] = ops.Multiply(x[i], scalar);
    }

    /// <summary>
    /// Divides each element by a scalar using sequential loops.
    /// </summary>
    public static void DivideScalar<T>(INumericOperations<T> ops, ReadOnlySpan<T> x, T scalar, Span<T> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        for (int i = 0; i < x.Length; i++)
            destination[i] = ops.Divide(x[i], scalar);
    }

    /// <summary>
    /// Adds a scalar to each element using sequential loops.
    /// </summary>
    public static void AddScalar<T>(INumericOperations<T> ops, ReadOnlySpan<T> x, T scalar, Span<T> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        for (int i = 0; i < x.Length; i++)
            destination[i] = ops.Add(x[i], scalar);
    }

    /// <summary>
    /// Subtracts a scalar from each element using sequential loops.
    /// </summary>
    public static void SubtractScalar<T>(INumericOperations<T> ops, ReadOnlySpan<T> x, T scalar, Span<T> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        for (int i = 0; i < x.Length; i++)
            destination[i] = ops.Subtract(x[i], scalar);
    }

    /// <summary>
    /// Computes the square root of each element using sequential loops.
    /// </summary>
    public static void Sqrt<T>(INumericOperations<T> ops, ReadOnlySpan<T> x, Span<T> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        for (int i = 0; i < x.Length; i++)
            destination[i] = ops.Sqrt(x[i]);
    }

    /// <summary>
    /// Computes the absolute value of each element using sequential loops.
    /// </summary>
    public static void Abs<T>(INumericOperations<T> ops, ReadOnlySpan<T> x, Span<T> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        for (int i = 0; i < x.Length; i++)
            destination[i] = ops.Abs(x[i]);
    }

    /// <summary>
    /// Negates each element using sequential loops.
    /// </summary>
    public static void Negate<T>(INumericOperations<T> ops, ReadOnlySpan<T> x, Span<T> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        for (int i = 0; i < x.Length; i++)
            destination[i] = ops.Negate(x[i]);
    }

    /// <summary>
    /// Clips each element to a range using sequential loops.
    /// </summary>
    public static void Clip<T>(INumericOperations<T> ops, ReadOnlySpan<T> x, T min, T max, Span<T> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        for (int i = 0; i < x.Length; i++)
        {
            T val = x[i];
            if (ops.LessThan(val, min))
                destination[i] = min;
            else if (ops.GreaterThan(val, max))
                destination[i] = max;
            else
                destination[i] = val;
        }
    }

    /// <summary>
    /// Computes the power of each element using sequential loops.
    /// </summary>
    public static void Pow<T>(INumericOperations<T> ops, ReadOnlySpan<T> x, T power, Span<T> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        for (int i = 0; i < x.Length; i++)
            destination[i] = ops.Power(x[i], power);
    }

    /// <summary>
    /// Copies elements from source to destination using sequential loops.
    /// </summary>
    public static void Copy<T>(INumericOperations<T> ops, ReadOnlySpan<T> source, Span<T> destination)
    {
        if (source.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        source.CopyTo(destination);
    }

    /// <summary>
    /// Computes floor of each element using sequential loops.
    /// </summary>
    public static void Floor<T>(INumericOperations<T> ops, ReadOnlySpan<T> x, Span<T> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        for (int i = 0; i < x.Length; i++)
            destination[i] = ops.Floor(x[i]);
    }

    /// <summary>
    /// Computes ceiling of each element using sequential loops.
    /// </summary>
    public static void Ceiling<T>(INumericOperations<T> ops, ReadOnlySpan<T> x, Span<T> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        for (int i = 0; i < x.Length; i++)
            destination[i] = ops.Ceiling(x[i]);
    }

    /// <summary>
    /// Computes fractional part of each element using sequential loops.
    /// </summary>
    public static void Frac<T>(INumericOperations<T> ops, ReadOnlySpan<T> x, Span<T> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        for (int i = 0; i < x.Length; i++)
            destination[i] = ops.Frac(x[i]);
    }

    /// <summary>
    /// Computes sine of each element using sequential loops.
    /// </summary>
    public static void Sin<T>(INumericOperations<T> ops, ReadOnlySpan<T> x, Span<T> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        for (int i = 0; i < x.Length; i++)
            destination[i] = ops.Sin(x[i]);
    }

    /// <summary>
    /// Computes cosine of each element using sequential loops.
    /// </summary>
    public static void Cos<T>(INumericOperations<T> ops, ReadOnlySpan<T> x, Span<T> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        for (int i = 0; i < x.Length; i++)
            destination[i] = ops.Cos(x[i]);
    }

    /// <summary>
    /// Computes fused multiply-add: destination[i] = x[i] + y[i] * scalar.
    /// </summary>
    public static void MultiplyAdd<T>(INumericOperations<T> ops, ReadOnlySpan<T> x, ReadOnlySpan<T> y, T scalar, Span<T> destination)
    {
        if (x.Length != y.Length || x.Length != destination.Length)
            throw new ArgumentException("All spans must have the same length");

        for (int i = 0; i < x.Length; i++)
            destination[i] = ops.Add(x[i], ops.Multiply(y[i], scalar));
    }

    /// <summary>
    /// Converts elements from type T to float using sequential loops.
    /// </summary>
    public static void ToFloatSpan<T>(INumericOperations<T> ops, ReadOnlySpan<T> source, Span<float> destination)
    {
        if (source.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        for (int i = 0; i < source.Length; i++)
            destination[i] = ops.ToFloat(source[i]);
    }

    /// <summary>
    /// Converts elements from float to type T using sequential loops.
    /// </summary>
    public static void FromFloatSpan<T>(INumericOperations<T> ops, ReadOnlySpan<float> source, Span<T> destination)
    {
        if (source.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        for (int i = 0; i < source.Length; i++)
            destination[i] = ops.FromFloat(source[i]);
    }

    /// <summary>
    /// Converts elements from type T to Half (FP16) using sequential loops.
    /// </summary>
    /// <remarks>
    /// Conversion path: T -> float -> Half. For SIMD-optimized version on float arrays,
    /// use TensorPrimitivesHelper.ConvertToHalf when available.
    /// </remarks>
    public static void ToHalfSpan<T>(INumericOperations<T> ops, ReadOnlySpan<T> source, Span<Half> destination)
    {
        if (source.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        for (int i = 0; i < source.Length; i++)
            destination[i] = (Half)ops.ToFloat(source[i]);
    }

    /// <summary>
    /// Converts elements from Half (FP16) to type T using sequential loops.
    /// </summary>
    /// <remarks>
    /// Conversion path: Half -> float -> T. For SIMD-optimized version on float arrays,
    /// use TensorPrimitivesHelper.ConvertFromHalf when available.
    /// </remarks>
    public static void FromHalfSpan<T>(INumericOperations<T> ops, ReadOnlySpan<Half> source, Span<T> destination)
    {
        if (source.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        for (int i = 0; i < source.Length; i++)
            destination[i] = ops.FromFloat((float)source[i]);
    }
}
