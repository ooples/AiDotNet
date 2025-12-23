using System;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Operators;
using TensorPrimitives = System.Numerics.Tensors.TensorPrimitives;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// CPU-based execution engine using INumericOperations for type-generic operations.
/// </summary>
/// <remarks>
/// <para>
/// CpuEngine provides the default execution backend for AiDotNet. It works with
/// any numeric type that implements INumericOperations{T}, including decimal,
/// BigInteger, and custom numeric types.
/// </para>
/// <para><b>For Beginners:</b> This is the standard, "always works" mode.
///
/// CpuEngine characteristics:
/// - Works with ANY numeric type (float, double, decimal, BigInteger, custom types)
/// - No special hardware required
/// - Good performance for small-to-medium datasets
/// - Single-threaded by default (can be parallelized in future versions)
///
/// When to use:
/// - You need decimal or high-precision arithmetic
/// - You don't have a GPU
/// - Your datasets are small (< 100K parameters)
/// - You're using custom numeric types
/// </para>
/// </remarks>
public class CpuEngine : IEngine
{
    /// <inheritdoc/>
    public string Name => "CPU Engine";

    /// <inheritdoc/>
    public bool SupportsGpu => false;

    /// <inheritdoc/>
    public Vector<T> Add<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Vector lengths must match. Got {a.Length} and {b.Length}");
        }

        // Use SIMD-optimized TensorPrimitivesHelper (5-10ÃƒÆ’Ã¢â‚¬â€ speedup for float)
        return TensorPrimitivesHelper<T>.Add(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Subtract<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Vector lengths must match. Got {a.Length} and {b.Length}");
        }

        // Use SIMD-optimized TensorPrimitivesHelper (5-10ÃƒÆ’Ã¢â‚¬â€ speedup for float)
        return TensorPrimitivesHelper<T>.Subtract(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Multiply<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Vector lengths must match. Got {a.Length} and {b.Length}");
        }

        // Use SIMD-optimized TensorPrimitivesHelper (5-10ÃƒÆ’Ã¢â‚¬â€ speedup for float)
        return TensorPrimitivesHelper<T>.Multiply(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Multiply<T>(Vector<T> vector, T scalar)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        // Create scalar vector and use SIMD-optimized multiplication
        var scalarVector = Vector<T>.CreateDefault(vector.Length, scalar);
        return TensorPrimitivesHelper<T>.Multiply(vector, scalarVector);
    }

    /// <inheritdoc/>
    public Vector<T> Divide<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Vector lengths must match. Got {a.Length} and {b.Length}");
        }

        // Check for division by zero before calling TensorPrimitivesHelper
        var numOps = MathHelper.GetNumericOperations<T>();
        var bArray = b.ToArray();
        for (int i = 0; i < bArray.Length; i++)
        {
            if (numOps.Equals(bArray[i], numOps.Zero))
            {
                throw new DivideByZeroException($"Division by zero at index {i}");
            }
        }

        // Use SIMD-optimized TensorPrimitivesHelper (5-10ÃƒÆ’Ã¢â‚¬â€ speedup for float)
        return TensorPrimitivesHelper<T>.Divide(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Divide<T>(Vector<T> vector, T scalar)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        var numOps = MathHelper.GetNumericOperations<T>();

        // Check for division by zero
        if (numOps.Equals(scalar, numOps.Zero))
        {
            throw new DivideByZeroException("Cannot divide by zero");
        }

        // Create scalar vector and use SIMD-optimized division
        var scalarVector = Vector<T>.CreateDefault(vector.Length, scalar);
        return TensorPrimitivesHelper<T>.Divide(vector, scalarVector);
    }

    /// <inheritdoc/>
    public Vector<T> Sqrt<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        // Use SIMD-optimized TensorPrimitivesHelper (5-10ÃƒÆ’Ã¢â‚¬â€ speedup for float)
        return TensorPrimitivesHelper<T>.Sqrt(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Power<T>(Vector<T> vector, T exponent)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = numOps.Power(vector[i], exponent);
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Max<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Vector lengths must match. Got {a.Length} and {b.Length}");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(a.Length);

        for (int i = 0; i < a.Length; i++)
        {
            result[i] = numOps.GreaterThan(a[i], b[i]) ? a[i] : b[i];
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Min<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Vector lengths must match. Got {a.Length} and {b.Length}");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(a.Length);

        for (int i = 0; i < a.Length; i++)
        {
            result[i] = numOps.LessThan(a[i], b[i]) ? a[i] : b[i];
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Abs<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = numOps.Abs(vector[i]);
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Exp<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        // Use SIMD-optimized TensorPrimitivesHelper (3-6ÃƒÆ’Ã¢â‚¬â€ speedup for float)
        return TensorPrimitivesHelper<T>.Exp(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Log<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        // Use SIMD-optimized TensorPrimitivesHelper (3-6ÃƒÆ’Ã¢â‚¬â€ speedup for float)
        return TensorPrimitivesHelper<T>.Log(vector);
    }

    /// <inheritdoc/>
    /// <inheritdoc/>
    public Vector<T> Exp2<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        // For float and double, use optimized span operations
        if (typeof(T) == typeof(float) && vector is Vector<float> floatVec)
        {
            var result = new Vector<float>(floatVec.Length);
            Exp2(floatVec.AsSpan(), result.AsWritableSpan());
            if (result is Vector<T> typedResult)
            {
                return typedResult;
            }
        }
        else if (typeof(T) == typeof(double) && vector is Vector<double> doubleVec)
        {
            var result = new Vector<double>(doubleVec.Length);
            Exp2(doubleVec.AsSpan(), result.AsWritableSpan());
            if (result is Vector<T> typedResult)
            {
                return typedResult;
            }
        }

        // Generic fallback: 2^x
        var numOps = MathHelper.GetNumericOperations<T>();
        var genericResult = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
            genericResult[i] = numOps.FromDouble(Math.Pow(2.0, val));
        }
        return genericResult;
    }

    /// <inheritdoc/>
    public Vector<T> Exp10<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        // For float and double, use optimized span operations
        if (typeof(T) == typeof(float) && vector is Vector<float> floatVec)
        {
            var result = new Vector<float>(floatVec.Length);
            Exp10(floatVec.AsSpan(), result.AsWritableSpan());
            if (result is Vector<T> typedResult)
            {
                return typedResult;
            }
        }
        else if (typeof(T) == typeof(double) && vector is Vector<double> doubleVec)
        {
            var result = new Vector<double>(doubleVec.Length);
            Exp10(doubleVec.AsSpan(), result.AsWritableSpan());
            if (result is Vector<T> typedResult)
            {
                return typedResult;
            }
        }

        // Generic fallback: 10^x
        var numOps = MathHelper.GetNumericOperations<T>();
        var genericResult = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
            genericResult[i] = numOps.FromDouble(Math.Pow(10.0, val));
        }
        return genericResult;
    }

    public Vector<T> Sign<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        for (int i = 0; i < vector.Length; i++)
        {
            // Sign returns -1, 0, or +1
            if (numOps.GreaterThan(vector[i], numOps.Zero))
            {
                result[i] = numOps.One;
            }
            else if (numOps.LessThan(vector[i], numOps.Zero))
            {
                result[i] = numOps.Negate(numOps.One);
            }
            else
            {
                result[i] = numOps.Zero;
            }
        }

        return result;
    }

    #region Reduction Operations

    /// <inheritdoc/>
    public T Sum<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        var numOps = MathHelper.GetNumericOperations<T>();
        T sum = numOps.Zero;

        for (int i = 0; i < vector.Length; i++)
        {
            sum = numOps.Add(sum, vector[i]);
        }

        return sum;
    }

    /// <inheritdoc/>
    public T DotProduct<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Vectors must have the same length for dot product. Got lengths {a.Length} and {b.Length}.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        T result = numOps.Zero;

        for (int i = 0; i < a.Length; i++)
        {
            result = numOps.Add(result, numOps.Multiply(a[i], b[i]));
        }

        return result;
    }

    /// <inheritdoc/>
    public T Mean<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        if (vector.Length == 0) throw new ArgumentException("Cannot compute mean of empty vector.");

        var numOps = MathHelper.GetNumericOperations<T>();
        T sum = Sum(vector);
        T length = numOps.FromDouble(vector.Length);
        return numOps.Divide(sum, length);
    }

    /// <inheritdoc/>
    public Vector<T> Softmax<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        if (vector.Length == 0) throw new ArgumentException("Cannot compute softmax of empty vector.");

        // Use SIMD-optimized TensorPrimitivesHelper (3-6ÃƒÆ’Ã¢â‚¬â€ speedup for float)
        return TensorPrimitivesHelper<T>.Softmax(vector);
    }

    /// <inheritdoc/>
    public T CosineSimilarity<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have the same length.");

        // Use SIMD-optimized TensorPrimitivesHelper (10-15ÃƒÆ’Ã¢â‚¬â€ speedup for float)
        return TensorPrimitivesHelper<T>.CosineSimilarity(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Log2<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        // Use SIMD-optimized TensorPrimitivesHelper (3-6ÃƒÆ’Ã¢â‚¬â€ speedup for float)
        return TensorPrimitivesHelper<T>.Log2(vector);
    }

    /// <inheritdoc/>
    public Vector<T> ExpM1<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = numOps.Subtract(numOps.Exp(vector[i]), numOps.One);
        }
        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Log1P<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = numOps.Log(numOps.Add(vector[i], numOps.One));
        }
        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Negate<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = numOps.Negate(vector[i]);
        }
        return result;
    }

    /// <inheritdoc/>
    public T Product<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        if (vector.Length == 0) throw new ArgumentException("Cannot compute product of empty vector.");
        var numOps = MathHelper.GetNumericOperations<T>();
        T product = numOps.One;
        for (int i = 0; i < vector.Length; i++)
        {
            product = numOps.Multiply(product, vector[i]);
        }
        return product;
    }

    /// <inheritdoc/>
    public T StdDev<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        if (vector.Length == 0) throw new ArgumentException("Cannot compute standard deviation of empty vector.");
        var numOps = MathHelper.GetNumericOperations<T>();

        T mean = Mean(vector);
        T sumSquaredDiff = numOps.Zero;

        for (int i = 0; i < vector.Length; i++)
        {
            T diff = numOps.Subtract(vector[i], mean);
            sumSquaredDiff = numOps.Add(sumSquaredDiff, numOps.Square(diff));
        }

        T variance = numOps.Divide(sumSquaredDiff, numOps.FromDouble(vector.Length));
        return numOps.Sqrt(variance);
    }

    /// <inheritdoc/>
    public T Norm<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        T sumSquares = numOps.Zero;
        for (int i = 0; i < vector.Length; i++)
        {
            sumSquares = numOps.Add(sumSquares, numOps.Square(vector[i]));
        }
        return numOps.Sqrt(sumSquares);
    }

    /// <inheritdoc/>
    public T Distance<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length) throw new ArgumentException("Vectors must have the same length.");

        var numOps = MathHelper.GetNumericOperations<T>();
        T sumSquaredDiff = numOps.Zero;

        for (int i = 0; i < a.Length; i++)
        {
            T diff = numOps.Subtract(a[i], b[i]);
            sumSquaredDiff = numOps.Add(sumSquaredDiff, numOps.Square(diff));
        }

        return numOps.Sqrt(sumSquaredDiff);
    }

    /// <inheritdoc/>
    public Vector<T> MinMagnitude<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length) throw new ArgumentException("Vectors must have the same length.");

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(a.Length);

        for (int i = 0; i < a.Length; i++)
        {
            T absA = numOps.Abs(a[i]);
            T absB = numOps.Abs(b[i]);
            result[i] = numOps.LessThan(absA, absB) ? a[i] : b[i];
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> MaxMagnitude<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length) throw new ArgumentException("Vectors must have the same length.");

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(a.Length);

        for (int i = 0; i < a.Length; i++)
        {
            T absA = numOps.Abs(a[i]);
            T absB = numOps.Abs(b[i]);
            result[i] = numOps.GreaterThan(absA, absB) ? a[i] : b[i];
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Clamp<T>(Vector<T> vector, T min, T max)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        for (int i = 0; i < vector.Length; i++)
        {
            if (numOps.LessThan(vector[i], min))
                result[i] = min;
            else if (numOps.GreaterThan(vector[i], max))
                result[i] = max;
            else
                result[i] = vector[i];
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Lerp<T>(Vector<T> a, Vector<T> b, T t)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length) throw new ArgumentException("Vectors must have the same length.");

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(a.Length);

        for (int i = 0; i < a.Length; i++)
        {
            T diff = numOps.Subtract(b[i], a[i]);
            T scaled = numOps.Multiply(t, diff);
            result[i] = numOps.Add(a[i], scaled);
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Reciprocal<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = numOps.Divide(numOps.One, vector[i]);
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> ReciprocalSqrt<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = numOps.Divide(numOps.One, numOps.Sqrt(vector[i]));
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Sin<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        // For float and double, use SIMD-accelerated operators
        if (typeof(T) == typeof(float))
        {
            T[] inputData = vector.ToArray();
            T[] outputData = new T[vector.Length];

            // Reinterpret T[] as float[] since we know T is float
            float[] floatInput = Unsafe.As<float[]>(inputData);
            float[] floatOutput = Unsafe.As<float[]>(outputData);

            TensorPrimitivesCore.InvokeSpanIntoSpan<SinOperatorFloat>(
                floatInput.AsSpan(),
                floatOutput.AsSpan());

            var result = new Vector<T>(vector.Length);
            for (int i = 0; i < outputData.Length; i++)
            {
                result[i] = outputData[i];
            }
            return result;
        }
        else if (typeof(T) == typeof(double))
        {
            T[] inputData = vector.ToArray();
            T[] outputData = new T[vector.Length];

            // Reinterpret T[] as double[] since we know T is double
            double[] doubleInput = Unsafe.As<double[]>(inputData);
            double[] doubleOutput = Unsafe.As<double[]>(outputData);

            TensorPrimitivesCore.InvokeSpanIntoSpan<SinOperatorDouble>(
                doubleInput.AsSpan(),
                doubleOutput.AsSpan());

            var result = new Vector<T>(vector.Length);
            for (int i = 0; i < outputData.Length; i++)
            {
                result[i] = outputData[i];
            }
            return result;
        }
        else
        {
            // Fallback for other types using INumericOperations
            var numOps = MathHelper.GetNumericOperations<T>();
            var result = new Vector<T>(vector.Length);
            for (int i = 0; i < vector.Length; i++)
            {
                double val = Convert.ToDouble(vector[i]);
                result[i] = numOps.FromDouble(Math.Sin(val));
            }
            return result;
        }
    }

    /// <inheritdoc/>
    public Vector<T> Cos<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        // For float and double, use SIMD-accelerated operators
        if (typeof(T) == typeof(float))
        {
            T[] inputData = vector.ToArray();
            T[] outputData = new T[vector.Length];

            // Reinterpret T[] as float[] since we know T is float
            float[] floatInput = Unsafe.As<float[]>(inputData);
            float[] floatOutput = Unsafe.As<float[]>(outputData);

            TensorPrimitivesCore.InvokeSpanIntoSpan<CosOperatorFloat>(
                floatInput.AsSpan(),
                floatOutput.AsSpan());

            var result = new Vector<T>(vector.Length);
            for (int i = 0; i < outputData.Length; i++)
            {
                result[i] = outputData[i];
            }
            return result;
        }
        else if (typeof(T) == typeof(double))
        {
            T[] inputData = vector.ToArray();
            T[] outputData = new T[vector.Length];

            // Reinterpret T[] as double[] since we know T is double
            double[] doubleInput = Unsafe.As<double[]>(inputData);
            double[] doubleOutput = Unsafe.As<double[]>(outputData);

            TensorPrimitivesCore.InvokeSpanIntoSpan<CosOperatorDouble>(
                doubleInput.AsSpan(),
                doubleOutput.AsSpan());

            var result = new Vector<T>(vector.Length);
            for (int i = 0; i < outputData.Length; i++)
            {
                result[i] = outputData[i];
            }
            return result;
        }
        else
        {
            // Fallback for other types using INumericOperations
            var numOps = MathHelper.GetNumericOperations<T>();
            var result = new Vector<T>(vector.Length);
            for (int i = 0; i < vector.Length; i++)
            {
                double val = Convert.ToDouble(vector[i]);
                result[i] = numOps.FromDouble(Math.Cos(val));
            }
            return result;
        }
    }

    /// <inheritdoc/>
    public void SinCos<T>(Vector<T> vector, out Vector<T> sinResult, out Vector<T> cosResult)
    {
        // For now, compute separately (can be optimized later with simultaneous computation)
        sinResult = Sin(vector);
        cosResult = Cos(vector);
    }

    /// <inheritdoc/>
    public void Sin(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<SinOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Sin(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<SinOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Cos(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<CosOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Cos(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<CosOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Exp(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<ExpOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Exp(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<ExpOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Log(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<LogOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Log(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<LogOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Exp2(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<Exp2OperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Exp2(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<Exp2OperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Exp10(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<Exp10OperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Exp10(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<Exp10OperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void ExpM1(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<ExpM1OperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void ExpM1(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<ExpM1OperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Log1P(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<Log1POperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Log1P(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<Log1POperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Tan(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<TanOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Tan(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<TanOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public Vector<T> Asin<T>(Vector<T> vector)
    {
        if (vector is null)
        {
            throw new ArgumentNullException(nameof(vector));
        }

        // For float and double, use optimized span operations
        if (typeof(T) == typeof(float) && vector is Vector<float> floatVec)
        {
            var result = new Vector<float>(floatVec.Length);
            Asin(floatVec.AsSpan(), result.AsWritableSpan());
            if (result is Vector<T> typedResult)
            {
                return typedResult;
            }
        }
        else if (typeof(T) == typeof(double) && vector is Vector<double> doubleVec)
        {
            var result = new Vector<double>(doubleVec.Length);
            Asin(doubleVec.AsSpan(), result.AsWritableSpan());
            if (result is Vector<T> typedResult)
            {
                return typedResult;
            }
        }

        // Generic fallback
        var numOps = MathHelper.GetNumericOperations<T>();
        var genericResult = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
            genericResult[i] = numOps.FromDouble(Math.Asin(val));
        }
        return genericResult;
    }

    /// <inheritdoc/>
    public void Asin(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AsinOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Asin(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AsinOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public Vector<T> Acos<T>(Vector<T> vector)
    {
        if (vector is null)
        {
            throw new ArgumentNullException(nameof(vector));
        }

        // For float and double, use optimized span operations
        if (typeof(T) == typeof(float) && vector is Vector<float> floatVec)
        {
            var result = new Vector<float>(floatVec.Length);
            Acos(floatVec.AsSpan(), result.AsWritableSpan());
            if (result is Vector<T> typedResult)
            {
                return typedResult;
            }
        }
        else if (typeof(T) == typeof(double) && vector is Vector<double> doubleVec)
        {
            var result = new Vector<double>(doubleVec.Length);
            Acos(doubleVec.AsSpan(), result.AsWritableSpan());
            if (result is Vector<T> typedResult)
            {
                return typedResult;
            }
        }

        // Generic fallback
        var numOps = MathHelper.GetNumericOperations<T>();
        var genericResult = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
            genericResult[i] = numOps.FromDouble(Math.Acos(val));
        }
        return genericResult;
    }

    /// <inheritdoc/>
    public void Acos(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AcosOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Acos(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AcosOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public Vector<T> Atan<T>(Vector<T> vector)
    {
        if (vector is null)
        {
            throw new ArgumentNullException(nameof(vector));
        }

        // For float and double, use optimized span operations
        if (typeof(T) == typeof(float) && vector is Vector<float> floatVec)
        {
            var result = new Vector<float>(floatVec.Length);
            Atan(floatVec.AsSpan(), result.AsWritableSpan());
            if (result is Vector<T> typedResult)
            {
                return typedResult;
            }
        }
        else if (typeof(T) == typeof(double) && vector is Vector<double> doubleVec)
        {
            var result = new Vector<double>(doubleVec.Length);
            Atan(doubleVec.AsSpan(), result.AsWritableSpan());
            if (result is Vector<T> typedResult)
            {
                return typedResult;
            }
        }

        // Generic fallback
        var numOps = MathHelper.GetNumericOperations<T>();
        var genericResult = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
            genericResult[i] = numOps.FromDouble(Math.Atan(val));
        }
        return genericResult;
    }

    /// <inheritdoc/>
    public void Atan(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AtanOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Atan(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AtanOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Sqrt(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<SqrtOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Sqrt(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<SqrtOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Abs(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AbsOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Abs(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AbsOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Sinh(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<SinhOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Sinh(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<SinhOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Cosh(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<CoshOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Cosh(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<CoshOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Tanh(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<TanhOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Tanh(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<TanhOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Asinh(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AsinhOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Asinh(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AsinhOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Acosh(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AcoshOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Acosh(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AcoshOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Atanh(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AtanhOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Atanh(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AtanhOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Reciprocal(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<ReciprocalOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Reciprocal(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<ReciprocalOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Cbrt(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<CbrtOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Cbrt(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<CbrtOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Log2(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<Log2OperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Log2(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<Log2OperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Log10(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<Log10OperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Log10(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<Log10OperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public Vector<T> Sinh<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        // TensorPrimitives.Sinh not available - use Math.Sinh
        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
            result[i] = numOps.FromDouble(Math.Sinh(val));
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Cosh<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        // TensorPrimitives.Cosh not available - use Math.Cosh
        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
            result[i] = numOps.FromDouble(Math.Cosh(val));
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Asinh<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        // Use asinh(x) = log(x + sqrt(x^2 + 1))
        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
#if NET5_0_OR_GREATER
            result[i] = numOps.FromDouble(Math.Asinh(val));
#else
            result[i] = numOps.FromDouble(Math.Log(val + Math.Sqrt(val * val + 1.0)));
#endif
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Acosh<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        // Use acosh(x) = log(x + sqrt(x^2 - 1))
        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
#if NET5_0_OR_GREATER
            result[i] = numOps.FromDouble(Math.Acosh(val));
#else
            result[i] = numOps.FromDouble(Math.Log(val + Math.Sqrt(val * val - 1.0)));
#endif
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Atanh<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        // Use atanh(x) = 0.5 * log((1 + x) / (1 - x))
        for (int i = 0; i < vector.Length; i++)
        {
            double val = numOps.ToDouble(vector[i]);
            result[i] = numOps.FromDouble(MathHelper.Atanh(val));
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Round<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
            result[i] = numOps.FromDouble(Math.Round(val));
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Floor<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
            result[i] = numOps.FromDouble(Math.Floor(val));
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Ceiling<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
            result[i] = numOps.FromDouble(Math.Ceiling(val));
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Truncate<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
            result[i] = numOps.FromDouble(Math.Truncate(val));
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Fill<T>(int length, T value)
    {
        if (length < 0) throw new ArgumentException("Length must be non-negative.", nameof(length));
        var result = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            result[i] = value;
        }
        return result;
    }

    /// <inheritdoc/>
    public Vector<T> FillZero<T>(int length)
    {
        if (length < 0) throw new ArgumentException("Length must be non-negative.", nameof(length));
        return new Vector<T>(length); // Vector constructor already initializes to zero
    }

    /// <inheritdoc/>
    public Vector<T> GenerateDropoutMask<T>(int length, T dropoutRate, T scale, int? seed = null)
    {
        if (length < 0) throw new ArgumentException("Length must be non-negative.", nameof(length));
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        var numOps = MathHelper.GetNumericOperations<T>();
        double dropoutRateDouble = Convert.ToDouble(dropoutRate);
        var mask = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            mask[i] = random.NextDouble() > dropoutRateDouble ? scale : numOps.Zero;
        }
        return mask;
    }

    /// <inheritdoc/>
    public void CopyVectorToTensor<T>(Vector<T> source, Tensor<T> destination)
    {
        if (source == null) throw new ArgumentNullException(nameof(source));
        if (destination == null) throw new ArgumentNullException(nameof(destination));
        if (source.Length != destination.Length)
        {
            throw new ArgumentException(
                $"Vector length ({source.Length}) must equal tensor total elements ({destination.Length}).");
        }
        for (int i = 0; i < source.Length; i++)
        {
            destination[i] = source[i];
        }
    }
    /// <inheritdoc/>
    public Vector<T> GenerateGaussianNoise<T>(int length, T mean, T standardDeviation, int? seed = null)
    {
        if (length < 0) throw new ArgumentException("Length must be non-negative.", nameof(length));
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        var numOps = MathHelper.GetNumericOperations<T>();
        var noise = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            // Box-Muller transform to generate Gaussian random numbers
            T u1 = numOps.FromDouble(random.NextDouble());
            T u2 = numOps.FromDouble(random.NextDouble());
            T z = numOps.Multiply(
                numOps.Sqrt(numOps.Multiply(numOps.FromDouble(-2.0), numOps.Log(u1))),
                numOps.FromDouble(Math.Cos(2.0 * Math.PI * Convert.ToDouble(u2))));
            noise[i] = numOps.Add(mean, numOps.Multiply(standardDeviation, z));
        }
        return noise;
    }

    #endregion

    #region Matrix Operations (Phase B: Epic 2)

    /// <inheritdoc/>
    public Matrix<T> MatrixMultiply<T>(Matrix<T> a, Matrix<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Columns != b.Rows)
        {
            throw new ArgumentException(
                $"Matrix dimensions incompatible for multiplication. " +
                $"First matrix is {a.Rows}x{a.Columns}, second is {b.Rows}x{b.Columns}. " +
                $"First matrix columns ({a.Columns}) must equal second matrix rows ({b.Rows}).");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Matrix<T>(a.Rows, b.Columns);

        // Standard O(nÃƒâ€šÃ‚Â³) matrix multiplication
        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < b.Columns; j++)
            {
                T sum = numOps.Zero;
                for (int k = 0; k < a.Columns; k++)
                {
                    sum = numOps.Add(sum, numOps.Multiply(a[i, k], b[k, j]));
                }
                result[i, j] = sum;
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> MatrixVectorMultiply<T>(Matrix<T> matrix, Vector<T> vector)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        if (matrix.Columns != vector.Length)
        {
            throw new ArgumentException(
                $"Matrix-vector dimensions incompatible. " +
                $"Matrix is {matrix.Rows}x{matrix.Columns}, vector has {vector.Length} elements. " +
                $"Matrix columns ({matrix.Columns}) must equal vector length ({vector.Length}).");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(matrix.Rows);

        for (int i = 0; i < matrix.Rows; i++)
        {
            T sum = numOps.Zero;
            for (int j = 0; j < matrix.Columns; j++)
            {
                sum = numOps.Add(sum, numOps.Multiply(matrix[i, j], vector[j]));
            }
            result[i] = sum;
        }

        return result;
    }

    /// <inheritdoc/>
    public Matrix<T> MatrixTranspose<T>(Matrix<T> matrix)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));

        var result = new Matrix<T>(matrix.Columns, matrix.Rows);

        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                result[j, i] = matrix[i, j];
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Matrix<T> MatrixAdd<T>(Matrix<T> a, Matrix<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Rows != b.Rows || a.Columns != b.Columns)
        {
            throw new ArgumentException(
                $"Matrix dimensions must match for addition. " +
                $"First matrix is {a.Rows}x{a.Columns}, second is {b.Rows}x{b.Columns}.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Matrix<T>(a.Rows, a.Columns);

        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < a.Columns; j++)
            {
                result[i, j] = numOps.Add(a[i, j], b[i, j]);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Matrix<T> MatrixMultiplyScalar<T>(Matrix<T> matrix, T scalar)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Matrix<T>(matrix.Rows, matrix.Columns);

        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                result[i, j] = numOps.Multiply(matrix[i, j], scalar);
            }
        }

        return result;
    }

    public Matrix<T> MatrixSubtract<T>(Matrix<T> a, Matrix<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Rows != b.Rows || a.Columns != b.Columns)
            throw new ArgumentException("Matrix dimensions must match for subtraction");

        var result = new Matrix<T>(a.Rows, a.Columns);

        // VECTORIZED: Use existing Vector Subtract operation on each row
        for (int i = 0; i < a.Rows; i++)
        {
            var rowA = a.GetRow(i);
            var rowB = b.GetRow(i);
            var diffRow = Subtract(rowA, rowB); // Reuse vectorized Vector Subtract
            result.SetRow(i, diffRow);
        }

        return result;
    }

    public T MatrixSumOfSquares<T>(Matrix<T> matrix)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));

        var numOps = MathHelper.GetNumericOperations<T>();
        T sum = numOps.Zero;

        // VECTORIZED: Use existing DotProduct operation on each row
        for (int i = 0; i < matrix.Rows; i++)
        {
            var row = matrix.GetRow(i);
            T rowSumSquares = DotProduct(row, row); // row Ãƒâ€šÃ‚Â· row = sum of squares for row
            sum = numOps.Add(sum, rowSumSquares);
        }

        return sum;
    }

    public void SwapColumns<T>(Matrix<T> matrix, int col1, int col2)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));

        // Direct element swap - no vectorization benefit for column swaps due to strided access
        for (int i = 0; i < matrix.Rows; i++)
        {
            T temp = matrix[i, col1];
            matrix[i, col1] = matrix[i, col2];
            matrix[i, col2] = temp;
        }
    }

    public void SwapRows<T>(Matrix<T> matrix, int row1, int row2)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));

        // Use vectorized operations for row swapping
        var tempRow1 = GetRow(matrix, row1);
        var tempRow2 = GetRow(matrix, row2);

        SetRow(matrix, row1, tempRow2);
        SetRow(matrix, row2, tempRow1);
    }

    public Matrix<T> OuterProduct<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        var result = new Matrix<T>(a.Length, b.Length);
        var aArray = a.ToArray();
        var bArray = b.ToArray();

        // Use SIMD-optimized TensorPrimitives for float type
        if (typeof(T) == typeof(float) && bArray.Length >= 16)
        {
            var bFloat = (float[])(object)bArray;
            var aFloat = (float[])(object)aArray;

            for (int i = 0; i < aFloat.Length; i++)
            {
                var rowData = new float[bFloat.Length];
                // SIMD vectorized: multiply vector b by scalar a[i]
                TensorPrimitives.Multiply(bFloat, aFloat[i], rowData);

                // Copy result to matrix
                for (int j = 0; j < bFloat.Length; j++)
                {
                    result[i, j] = (T)(object)rowData[j];
                }
            }
        }
        else
        {
            // Fallback using NumOps
            var numOps = MathHelper.GetNumericOperations<T>();
            for (int i = 0; i < aArray.Length; i++)
            {
                for (int j = 0; j < bArray.Length; j++)
                {
                    result[i, j] = numOps.Multiply(aArray[i], bArray[j]);
                }
            }
        }

        return result;
    }

    public Vector<T> GetColumn<T>(Matrix<T> matrix, int columnIndex)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));
        if (columnIndex < 0 || columnIndex >= matrix.Columns)
            throw new ArgumentOutOfRangeException(nameof(columnIndex),
                $"Column index {columnIndex} is out of range. Valid range is 0 to {matrix.Columns - 1}.");

        // No vectorization benefit - column access is strided
        var result = new T[matrix.Rows];
        for (int i = 0; i < matrix.Rows; i++)
        {
            result[i] = matrix[i, columnIndex];
        }
        return new Vector<T>(result);
    }

    public Vector<T> GetRow<T>(Matrix<T> matrix, int rowIndex)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));
        if (rowIndex < 0 || rowIndex >= matrix.Rows)
            throw new ArgumentOutOfRangeException(nameof(rowIndex),
                $"Row index {rowIndex} is out of range. Valid range is 0 to {matrix.Rows - 1}.");

        // Row access is contiguous - can use direct array copy
        var result = new T[matrix.Columns];
        for (int j = 0; j < matrix.Columns; j++)
        {
            result[j] = matrix[rowIndex, j];
        }
        return new Vector<T>(result);
    }

    public void SetColumn<T>(Matrix<T> matrix, int columnIndex, Vector<T> values)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));
        if (values == null) throw new ArgumentNullException(nameof(values));
        if (columnIndex < 0 || columnIndex >= matrix.Columns)
            throw new ArgumentOutOfRangeException(nameof(columnIndex),
                $"Column index {columnIndex} is out of range. Valid range is 0 to {matrix.Columns - 1}.");
        if (values.Length != matrix.Rows)
            throw new ArgumentException(
                $"Values vector length ({values.Length}) must match matrix rows ({matrix.Rows}).",
                nameof(values));

        // No vectorization benefit - column access is strided
        var valuesArray = values.ToArray();
        for (int i = 0; i < matrix.Rows; i++)
        {
            matrix[i, columnIndex] = valuesArray[i];
        }
    }

    public void SetRow<T>(Matrix<T> matrix, int rowIndex, Vector<T> values)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));
        if (values == null) throw new ArgumentNullException(nameof(values));
        if (rowIndex < 0 || rowIndex >= matrix.Rows)
            throw new ArgumentOutOfRangeException(nameof(rowIndex),
                $"Row index {rowIndex} is out of range. Valid range is 0 to {matrix.Rows - 1}.");
        if (values.Length != matrix.Columns)
            throw new ArgumentException(
                $"Values vector length ({values.Length}) must match matrix columns ({matrix.Columns}).",
                nameof(values));

        // Row access is contiguous - direct assignment
        var valuesArray = values.ToArray();
        for (int j = 0; j < matrix.Columns; j++)
        {
            matrix[rowIndex, j] = valuesArray[j];
        }
    }

    #endregion

    #region Tensor Operations (Phase B: Epic 3)

    /// <inheritdoc/>
    public Tensor<T> Reshape<T>(Tensor<T> tensor, int[] newShape)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (newShape == null) throw new ArgumentNullException(nameof(newShape));

        return tensor.Reshape(newShape);
    }

    /// <inheritdoc/>
    public Tensor<T> BatchMatMul<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Rank != 3 || b.Rank != 3)
        {
            throw new ArgumentException(
                $"BatchMatMul requires 3D tensors. Got ranks {a.Rank} and {b.Rank}.");
        }

        int batchSize = a.Shape[0];
        int m = a.Shape[1];
        int k = a.Shape[2];
        int k2 = b.Shape[1];
        int n = b.Shape[2];

        if (b.Shape[0] != batchSize)
        {
            throw new ArgumentException(
                $"Batch sizes must match. Got {batchSize} and {b.Shape[0]}.");
        }
        if (k != k2)
        {
            throw new ArgumentException(
                $"Matrix dimensions incompatible for multiplication. " +
                $"First tensor has shape [{batchSize}, {m}, {k}], " +
                $"second has shape [{b.Shape[0]}, {k2}, {n}]. " +
                $"Inner dimensions must match ({k} != {k2}).");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(new[] { batchSize, m, n });

        // Process each batch
        for (int batch = 0; batch < batchSize; batch++)
        {
            // Standard matrix multiplication for this batch: C[batch] = A[batch] @ B[batch]
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    T sum = numOps.Zero;
                    for (int p = 0; p < k; p++)
                    {
                        sum = numOps.Add(sum, numOps.Multiply(
                            a[batch, i, p],
                            b[batch, p, j]));
                    }
                    result[batch, i, j] = sum;
                }
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorAdd<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!ShapesMatch(a.Shape, b.Shape))
        {
            throw new ArgumentException(
                $"Tensor shapes must match. Got {FormatShape(a.Shape)} and {FormatShape(b.Shape)}.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(a.Shape);

        for (int i = 0; i < a.Length; i++)
        {
            result.SetFlat(i, numOps.Add(a.GetFlat(i), b.GetFlat(i)));
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorBroadcastAdd<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        // Use optimized Tensor.BroadcastAdd which handles broadcasting logic
        return a.BroadcastAdd(b);
    }

    /// <inheritdoc/>
    public Tensor<T> TensorAddMany<T>(params Tensor<T>[] tensors)
    {
        if (tensors == null) throw new ArgumentNullException(nameof(tensors));
        if (tensors.Length < 2)
            throw new ArgumentException("TensorAddMany requires at least 2 tensors.", nameof(tensors));

        // Validate all shapes match the first tensor
        var referenceShape = tensors[0].Shape;
        for (int t = 1; t < tensors.Length; t++)
        {
            if (!ShapesMatch(referenceShape, tensors[t].Shape))
            {
                throw new ArgumentException(
                    $"All tensor shapes must match. Tensor 0 has shape {FormatShape(referenceShape)}, " +
                    $"but tensor {t} has shape {FormatShape(tensors[t].Shape)}.");
            }
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(referenceShape);
        int length = tensors[0].Length;

        // Single-pass addition: accumulate all tensors element by element
        // This avoids n-1 intermediate allocations from chained binary additions
        if (length > 10000)
        {
            // Parallel execution for large tensors
            Parallel.For(0, length, i =>
            {
                T sum = numOps.Zero;
                for (int t = 0; t < tensors.Length; t++)
                {
                    sum = numOps.Add(sum, tensors[t].GetFlat(i));
                }
                result.SetFlat(i, sum);
            });
        }
        else
        {
            // Sequential execution for smaller tensors (avoids parallel overhead)
            for (int i = 0; i < length; i++)
            {
                T sum = numOps.Zero;
                for (int t = 0; t < tensors.Length; t++)
                {
                    sum = numOps.Add(sum, tensors[t].GetFlat(i));
                }
                result.SetFlat(i, sum);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorSubtract<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!ShapesMatch(a.Shape, b.Shape))
        {
            throw new ArgumentException(
                $"Tensor shapes must match. Got {FormatShape(a.Shape)} and {FormatShape(b.Shape)}.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(a.Shape);

        for (int i = 0; i < a.Length; i++)
        {
            result.SetFlat(i, numOps.Subtract(a.GetFlat(i), b.GetFlat(i)));
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorMultiply<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!ShapesMatch(a.Shape, b.Shape))
        {
            throw new ArgumentException(
                $"Tensor shapes must match. Got {FormatShape(a.Shape)} and {FormatShape(b.Shape)}.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(a.Shape);

        for (int i = 0; i < a.Length; i++)
        {
            result.SetFlat(i, numOps.Multiply(a.GetFlat(i), b.GetFlat(i)));
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorMultiplyMany<T>(params Tensor<T>[] tensors)
    {
        if (tensors == null) throw new ArgumentNullException(nameof(tensors));
        if (tensors.Length < 2)
            throw new ArgumentException("TensorMultiplyMany requires at least 2 tensors.", nameof(tensors));

        // Validate all shapes match the first tensor
        var referenceShape = tensors[0].Shape;
        for (int t = 1; t < tensors.Length; t++)
        {
            if (!ShapesMatch(referenceShape, tensors[t].Shape))
            {
                throw new ArgumentException(
                    $"All tensor shapes must match. Tensor 0 has shape {FormatShape(referenceShape)}, " +
                    $"but tensor {t} has shape {FormatShape(tensors[t].Shape)}.");
            }
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(referenceShape);
        int length = tensors[0].Length;

        // Single-pass multiplication: accumulate all tensors element by element
        // This avoids n-1 intermediate allocations from chained binary multiplications
        if (length > 10000)
        {
            // Parallel execution for large tensors
            Parallel.For(0, length, i =>
            {
                T product = numOps.One;
                for (int t = 0; t < tensors.Length; t++)
                {
                    product = numOps.Multiply(product, tensors[t].GetFlat(i));
                }
                result.SetFlat(i, product);
            });
        }
        else
        {
            // Sequential execution for smaller tensors (avoids parallel overhead)
            for (int i = 0; i < length; i++)
            {
                T product = numOps.One;
                for (int t = 0; t < tensors.Length; t++)
                {
                    product = numOps.Multiply(product, tensors[t].GetFlat(i));
                }
                result.SetFlat(i, product);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorMultiplyScalar<T>(Tensor<T> tensor, T scalar)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(tensor.Shape);

        for (int i = 0; i < tensor.Length; i++)
        {
            result.SetFlat(i, numOps.Multiply(tensor.GetFlat(i), scalar));
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorDivide<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!ShapesMatch(a.Shape, b.Shape))
        {
            throw new ArgumentException(
                $"Tensor shapes must match. Got {FormatShape(a.Shape)} and {FormatShape(b.Shape)}.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(a.Shape);

        for (int i = 0; i < a.Length; i++)
        {
            // Check for division by zero
            if (numOps.Equals(b.GetFlat(i), numOps.Zero))
            {
                throw new DivideByZeroException($"Division by zero at index {i}");
            }

            result.SetFlat(i, numOps.Divide(a.GetFlat(i), b.GetFlat(i)));
        }

        return result;
    }

    #region Tensor Comparison Operations

    /// <inheritdoc/>
    public Tensor<T> TensorEquals<T>(Tensor<T> tensor, T value)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(tensor.Shape);

        // Use SIMD-friendly loop with parallel execution for large tensors
        if (tensor.Length > 10000)
        {
            Parallel.For(0, tensor.Length, i =>
            {
                result.SetFlat(i, numOps.Equals(tensor.GetFlat(i), value) ? numOps.One : numOps.Zero);
            });
        }
        else
        {
            for (int i = 0; i < tensor.Length; i++)
            {
                result.SetFlat(i, numOps.Equals(tensor.GetFlat(i), value) ? numOps.One : numOps.Zero);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorEquals<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!ShapesMatch(a.Shape, b.Shape))
        {
            throw new ArgumentException(
                $"Tensor shapes must match. Got {FormatShape(a.Shape)} and {FormatShape(b.Shape)}.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(a.Shape);

        if (a.Length > 10000)
        {
            Parallel.For(0, a.Length, i =>
            {
                result.SetFlat(i, numOps.Equals(a.GetFlat(i), b.GetFlat(i)) ? numOps.One : numOps.Zero);
            });
        }
        else
        {
            for (int i = 0; i < a.Length; i++)
            {
                result.SetFlat(i, numOps.Equals(a.GetFlat(i), b.GetFlat(i)) ? numOps.One : numOps.Zero);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorNotEquals<T>(Tensor<T> tensor, T value)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(tensor.Shape);

        if (tensor.Length > 10000)
        {
            Parallel.For(0, tensor.Length, i =>
            {
                result.SetFlat(i, !numOps.Equals(tensor.GetFlat(i), value) ? numOps.One : numOps.Zero);
            });
        }
        else
        {
            for (int i = 0; i < tensor.Length; i++)
            {
                result.SetFlat(i, !numOps.Equals(tensor.GetFlat(i), value) ? numOps.One : numOps.Zero);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorNotEquals<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!ShapesMatch(a.Shape, b.Shape))
        {
            throw new ArgumentException(
                $"Tensor shapes must match. Got {FormatShape(a.Shape)} and {FormatShape(b.Shape)}.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(a.Shape);

        if (a.Length > 10000)
        {
            Parallel.For(0, a.Length, i =>
            {
                result.SetFlat(i, !numOps.Equals(a.GetFlat(i), b.GetFlat(i)) ? numOps.One : numOps.Zero);
            });
        }
        else
        {
            for (int i = 0; i < a.Length; i++)
            {
                result.SetFlat(i, !numOps.Equals(a.GetFlat(i), b.GetFlat(i)) ? numOps.One : numOps.Zero);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorGreaterThan<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!ShapesMatch(a.Shape, b.Shape))
        {
            throw new ArgumentException(
                $"Tensor shapes must match. Got {FormatShape(a.Shape)} and {FormatShape(b.Shape)}.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(a.Shape);

        if (a.Length > 10000)
        {
            Parallel.For(0, a.Length, i =>
            {
                result.SetFlat(i, numOps.GreaterThan(a.GetFlat(i), b.GetFlat(i)) ? numOps.One : numOps.Zero);
            });
        }
        else
        {
            for (int i = 0; i < a.Length; i++)
            {
                result.SetFlat(i, numOps.GreaterThan(a.GetFlat(i), b.GetFlat(i)) ? numOps.One : numOps.Zero);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorGreaterThan<T>(Tensor<T> tensor, T value)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(tensor.Shape);

        if (tensor.Length > 10000)
        {
            Parallel.For(0, tensor.Length, i =>
            {
                result.SetFlat(i, numOps.GreaterThan(tensor.GetFlat(i), value) ? numOps.One : numOps.Zero);
            });
        }
        else
        {
            for (int i = 0; i < tensor.Length; i++)
            {
                result.SetFlat(i, numOps.GreaterThan(tensor.GetFlat(i), value) ? numOps.One : numOps.Zero);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorLessThan<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!ShapesMatch(a.Shape, b.Shape))
        {
            throw new ArgumentException(
                $"Tensor shapes must match. Got {FormatShape(a.Shape)} and {FormatShape(b.Shape)}.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(a.Shape);

        if (a.Length > 10000)
        {
            Parallel.For(0, a.Length, i =>
            {
                result.SetFlat(i, numOps.LessThan(a.GetFlat(i), b.GetFlat(i)) ? numOps.One : numOps.Zero);
            });
        }
        else
        {
            for (int i = 0; i < a.Length; i++)
            {
                result.SetFlat(i, numOps.LessThan(a.GetFlat(i), b.GetFlat(i)) ? numOps.One : numOps.Zero);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorLessThan<T>(Tensor<T> tensor, T value)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(tensor.Shape);

        if (tensor.Length > 10000)
        {
            Parallel.For(0, tensor.Length, i =>
            {
                result.SetFlat(i, numOps.LessThan(tensor.GetFlat(i), value) ? numOps.One : numOps.Zero);
            });
        }
        else
        {
            for (int i = 0; i < tensor.Length; i++)
            {
                result.SetFlat(i, numOps.LessThan(tensor.GetFlat(i), value) ? numOps.One : numOps.Zero);
            }
        }

        return result;
    }

    #endregion

    #region Tensor Element-wise Math Operations

    /// <inheritdoc/>
    public Tensor<T> TensorLog<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(tensor.Shape);

        if (tensor.Length > 10000)
        {
            Parallel.For(0, tensor.Length, i =>
            {
                result.SetFlat(i, numOps.Log(tensor.GetFlat(i)));
            });
        }
        else
        {
            for (int i = 0; i < tensor.Length; i++)
            {
                result.SetFlat(i, numOps.Log(tensor.GetFlat(i)));
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorExp<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(tensor.Shape);

        if (tensor.Length > 10000)
        {
            Parallel.For(0, tensor.Length, i =>
            {
                result.SetFlat(i, numOps.Exp(tensor.GetFlat(i)));
            });
        }
        else
        {
            for (int i = 0; i < tensor.Length; i++)
            {
                result.SetFlat(i, numOps.Exp(tensor.GetFlat(i)));
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorSqrt<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(tensor.Shape);

        if (tensor.Length > 10000)
        {
            Parallel.For(0, tensor.Length, i =>
            {
                result.SetFlat(i, numOps.Sqrt(tensor.GetFlat(i)));
            });
        }
        else
        {
            for (int i = 0; i < tensor.Length; i++)
            {
                result.SetFlat(i, numOps.Sqrt(tensor.GetFlat(i)));
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorAbs<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(tensor.Shape);

        if (tensor.Length > 10000)
        {
            Parallel.For(0, tensor.Length, i =>
            {
                result.SetFlat(i, numOps.Abs(tensor.GetFlat(i)));
            });
        }
        else
        {
            for (int i = 0; i < tensor.Length; i++)
            {
                result.SetFlat(i, numOps.Abs(tensor.GetFlat(i)));
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorNegate<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(tensor.Shape);

        if (tensor.Length > 10000)
        {
            Parallel.For(0, tensor.Length, i =>
            {
                result.SetFlat(i, numOps.Negate(tensor.GetFlat(i)));
            });
        }
        else
        {
            for (int i = 0; i < tensor.Length; i++)
            {
                result.SetFlat(i, numOps.Negate(tensor.GetFlat(i)));
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorPow<T>(Tensor<T> tensor, T exponent)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(tensor.Shape);

        if (tensor.Length > 10000)
        {
            Parallel.For(0, tensor.Length, i =>
            {
                result.SetFlat(i, numOps.Power(tensor.GetFlat(i), exponent));
            });
        }
        else
        {
            for (int i = 0; i < tensor.Length; i++)
            {
                result.SetFlat(i, numOps.Power(tensor.GetFlat(i), exponent));
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorMax<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!ShapesMatch(a.Shape, b.Shape))
            throw new ArgumentException($"Tensor shapes must match. Got {FormatShape(a.Shape)} and {FormatShape(b.Shape)}.");

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(a.Shape);

        if (a.Length > 10000)
        {
            Parallel.For(0, a.Length, i =>
            {
                var aVal = a.GetFlat(i);
                var bVal = b.GetFlat(i);
                result.SetFlat(i, numOps.GreaterThan(aVal, bVal) ? aVal : bVal);
            });
        }
        else
        {
            for (int i = 0; i < a.Length; i++)
            {
                var aVal = a.GetFlat(i);
                var bVal = b.GetFlat(i);
                result.SetFlat(i, numOps.GreaterThan(aVal, bVal) ? aVal : bVal);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorMax<T>(Tensor<T> tensor, T value)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(tensor.Shape);

        if (tensor.Length > 10000)
        {
            Parallel.For(0, tensor.Length, i =>
            {
                var tVal = tensor.GetFlat(i);
                result.SetFlat(i, numOps.GreaterThan(tVal, value) ? tVal : value);
            });
        }
        else
        {
            for (int i = 0; i < tensor.Length; i++)
            {
                var tVal = tensor.GetFlat(i);
                result.SetFlat(i, numOps.GreaterThan(tVal, value) ? tVal : value);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorMin<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!ShapesMatch(a.Shape, b.Shape))
            throw new ArgumentException($"Tensor shapes must match. Got {FormatShape(a.Shape)} and {FormatShape(b.Shape)}.");

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(a.Shape);

        if (a.Length > 10000)
        {
            Parallel.For(0, a.Length, i =>
            {
                var aVal = a.GetFlat(i);
                var bVal = b.GetFlat(i);
                result.SetFlat(i, numOps.LessThan(aVal, bVal) ? aVal : bVal);
            });
        }
        else
        {
            for (int i = 0; i < a.Length; i++)
            {
                var aVal = a.GetFlat(i);
                var bVal = b.GetFlat(i);
                result.SetFlat(i, numOps.LessThan(aVal, bVal) ? aVal : bVal);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorMin<T>(Tensor<T> tensor, T value)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(tensor.Shape);

        if (tensor.Length > 10000)
        {
            Parallel.For(0, tensor.Length, i =>
            {
                var tVal = tensor.GetFlat(i);
                result.SetFlat(i, numOps.LessThan(tVal, value) ? tVal : value);
            });
        }
        else
        {
            for (int i = 0; i < tensor.Length; i++)
            {
                var tVal = tensor.GetFlat(i);
                result.SetFlat(i, numOps.LessThan(tVal, value) ? tVal : value);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorClamp<T>(Tensor<T> tensor, T min, T max)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(tensor.Shape);

        if (tensor.Length > 10000)
        {
            Parallel.For(0, tensor.Length, i =>
            {
                var val = tensor.GetFlat(i);
                if (numOps.LessThan(val, min)) val = min;
                else if (numOps.GreaterThan(val, max)) val = max;
                result.SetFlat(i, val);
            });
        }
        else
        {
            for (int i = 0; i < tensor.Length; i++)
            {
                var val = tensor.GetFlat(i);
                if (numOps.LessThan(val, min)) val = min;
                else if (numOps.GreaterThan(val, max)) val = max;
                result.SetFlat(i, val);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public T TensorSum<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        T sum = numOps.Zero;

        for (int i = 0; i < tensor.Length; i++)
        {
            sum = numOps.Add(sum, tensor.GetFlat(i));
        }

        return sum;
    }

    /// <inheritdoc/>
    public Tensor<T> ReduceSum<T>(Tensor<T> tensor, int[]? axes = null, bool keepDims = false)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        // Full reduction - sum all elements
        if (axes == null || axes.Length == 0)
        {
            T sum = TensorSum(tensor);
            if (keepDims)
            {
                var shape = new int[tensor.Rank];
                for (int i = 0; i < tensor.Rank; i++) shape[i] = 1;
                var result = new Tensor<T>(shape);
                result.SetFlat(0, sum);
                return result;
            }
            return new Tensor<T>([1], new Vector<T>([sum]));
        }

        // Validate and normalize axes consistently with other reducers
        var normalizedAxes = ValidateAndNormalizeAxes(axes, tensor.Rank);

        // Calculate output shape
        var outputShape = new List<int>();
        for (int i = 0; i < tensor.Rank; i++)
        {
            if (normalizedAxes.Contains(i))
            {
                if (keepDims) outputShape.Add(1);
            }
            else
            {
                outputShape.Add(tensor.Shape[i]);
            }
        }

        var result2 = new Tensor<T>(outputShape.ToArray());

        // Use tensor's built-in Sum which is already optimized
        var summed = tensor.Sum(normalizedAxes);

        // Copy to result with correct shape
        if (keepDims && summed.Rank != result2.Rank)
        {
            // Need to reshape
            for (int i = 0; i < summed.Length; i++)
            {
                result2.SetFlat(i, summed.GetFlat(i));
            }
            return result2;
        }

        return summed;
    }

    /// <inheritdoc/>
    public T TensorMaxValue<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Length == 0) throw new ArgumentException("Cannot compute max of empty tensor.", nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        T maxVal = tensor.GetFlat(0);

        // Parallel reduction for large tensors
        if (tensor.Length > 10000)
        {
            int workerCount = Environment.ProcessorCount;
            var localMaxes = new T[workerCount];
            var hasValue = new bool[workerCount];
            int chunkSize = (tensor.Length + workerCount - 1) / workerCount;

            Parallel.For(0, workerCount, threadIdx =>
            {
                int start = threadIdx * chunkSize;
                int end = Math.Min(start + chunkSize, tensor.Length);
                if (start >= tensor.Length) return;

                T localMax = tensor.GetFlat(start);
                for (int i = start + 1; i < end; i++)
                {
                    var val = tensor.GetFlat(i);
                    if (numOps.GreaterThan(val, localMax))
                        localMax = val;
                }
                localMaxes[threadIdx] = localMax;
                hasValue[threadIdx] = true;
            });

            // Combine only populated slots
            bool first = true;
            for (int i = 0; i < workerCount; i++)
            {
                if (!hasValue[i]) continue;
                if (first)
                {
                    maxVal = localMaxes[i];
                    first = false;
                }
                else if (numOps.GreaterThan(localMaxes[i], maxVal))
                {
                    maxVal = localMaxes[i];
                }
            }
        }
        else
        {
            for (int i = 1; i < tensor.Length; i++)
            {
                var val = tensor.GetFlat(i);
                if (numOps.GreaterThan(val, maxVal))
                    maxVal = val;
            }
        }

        return maxVal;
    }

    /// <inheritdoc/>
    public T TensorMinValue<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Length == 0) throw new ArgumentException("Cannot compute min of empty tensor.", nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        T minVal = tensor.GetFlat(0);

        // Parallel reduction for large tensors
        if (tensor.Length > 10000)
        {
            int workerCount = Environment.ProcessorCount;
            var localMins = new T[workerCount];
            var hasValue = new bool[workerCount];
            int chunkSize = (tensor.Length + workerCount - 1) / workerCount;

            Parallel.For(0, workerCount, threadIdx =>
            {
                int start = threadIdx * chunkSize;
                int end = Math.Min(start + chunkSize, tensor.Length);
                if (start >= tensor.Length) return;

                T localMin = tensor.GetFlat(start);
                for (int i = start + 1; i < end; i++)
                {
                    var val = tensor.GetFlat(i);
                    if (numOps.LessThan(val, localMin))
                        localMin = val;
                }
                localMins[threadIdx] = localMin;
                hasValue[threadIdx] = true;
            });

            // Combine only populated slots
            bool first = true;
            for (int i = 0; i < workerCount; i++)
            {
                if (!hasValue[i]) continue;
                if (first)
                {
                    minVal = localMins[i];
                    first = false;
                }
                else if (numOps.LessThan(localMins[i], minVal))
                {
                    minVal = localMins[i];
                }
            }
        }
        else
        {
            for (int i = 1; i < tensor.Length; i++)
            {
                var val = tensor.GetFlat(i);
                if (numOps.LessThan(val, minVal))
                    minVal = val;
            }
        }

        return minVal;
    }

    /// <inheritdoc/>
    public T TensorMean<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Length == 0) throw new ArgumentException("Cannot compute mean of empty tensor.", nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        T sum = TensorSum(tensor);
        return numOps.Divide(sum, numOps.FromDouble(tensor.Length));
    }

    #endregion

    /// <summary>
    /// Helper method to check if two shapes match.
    /// </summary>
    private bool ShapesMatch(int[] shape1, int[] shape2)
    {
        if (shape1.Length != shape2.Length)
            return false;

        for (int i = 0; i < shape1.Length; i++)
        {
            if (shape1[i] != shape2[i])
                return false;
        }

        return true;
    }

    /// <summary>
    /// Helper method to format a shape for error messages.
    /// </summary>
    private string FormatShape(int[] shape)
    {
        return "[" + string.Join(", ", shape) + "]";
    }

    /// <inheritdoc/>
    public Tensor<T> MaxPool2D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 4)
        {
            throw new ArgumentException($"MaxPool2D requires a 4D tensor [batch, channels, height, width]. Got rank {input.Rank}.");
        }
        if (poolSize <= 0) throw new ArgumentException("Pool size must be positive.");

        if (stride == 0) stride = poolSize; // Default stride equals pool size

        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int outputHeight = (height + 2 * padding - poolSize) / stride + 1;
        int outputWidth = (width + 2 * padding - poolSize) / stride + 1;

        if (outputHeight <= 0 || outputWidth <= 0)
        {
            throw new ArgumentException(
                $"Invalid pooling parameters. Output dimensions would be {outputHeight}x{outputWidth}. " +
                $"Ensure poolSize={poolSize}, stride={stride}, padding={padding} are compatible with input size {height}x{width}.");
        }

        var result = new Tensor<T>(new[] { batch, channels, outputHeight, outputWidth });

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        // Use MinValue for type-safe initialization (works for all numeric types)
                        T maxValue = numOps.MinValue;

                        for (int kh = 0; kh < poolSize; kh++)
                        {
                            for (int kw = 0; kw < poolSize; kw++)
                            {
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;

                                // Check bounds (handle padding)
                                if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                {
                                    T value = input[b, c, ih, iw];
                                    if (numOps.GreaterThan(value, maxValue))
                                    {
                                        maxValue = value;
                                    }
                                }
                            }
                        }

                        result[b, c, oh, ow] = maxValue;
                    }
                }
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> AvgPool2D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 4)
        {
            throw new ArgumentException($"AvgPool2D requires a 4D tensor [batch, channels, height, width]. Got rank {input.Rank}.");
        }
        if (poolSize <= 0) throw new ArgumentException("Pool size must be positive.");

        if (stride == 0) stride = poolSize; // Default stride equals pool size

        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int outputHeight = (height + 2 * padding - poolSize) / stride + 1;
        int outputWidth = (width + 2 * padding - poolSize) / stride + 1;

        if (outputHeight <= 0 || outputWidth <= 0)
        {
            throw new ArgumentException(
                $"Invalid pooling parameters. Output dimensions would be {outputHeight}x{outputWidth}. " +
                $"Ensure poolSize={poolSize}, stride={stride}, padding={padding} are compatible with input size {height}x{width}.");
        }

        var result = new Tensor<T>(new[] { batch, channels, outputHeight, outputWidth });

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        T sum = numOps.Zero;
                        int count = 0;

                        for (int kh = 0; kh < poolSize; kh++)
                        {
                            for (int kw = 0; kw < poolSize; kw++)
                            {
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;

                                // Check bounds (handle padding)
                                if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                {
                                    sum = numOps.Add(sum, input[b, c, ih, iw]);
                                    count++;
                                }
                            }
                        }

                        // Calculate average
                        if (count > 0)
                        {
                            var countValue = numOps.FromDouble(count);
                            result[b, c, oh, ow] = numOps.Divide(sum, countValue);
                        }
                        else
                        {
                            result[b, c, oh, ow] = numOps.Zero;
                        }
                    }
                }
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> Conv2D<T>(Tensor<T> input, Tensor<T> kernel, int stride = 1, int padding = 0, int dilation = 1)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (input.Rank != 4)
        {
            throw new ArgumentException($"Conv2D input requires a 4D tensor [batch, in_channels, height, width]. Got rank {input.Rank}.");
        }
        if (kernel.Rank != 4)
        {
            throw new ArgumentException($"Conv2D kernel requires a 4D tensor [out_channels, in_channels, kernel_height, kernel_width]. Got rank {kernel.Rank}.");
        }
        if (stride <= 0) throw new ArgumentException("Stride must be positive.");
        if (dilation <= 0) throw new ArgumentException("Dilation must be positive.");

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int outChannels = kernel.Shape[0];
        int kernelInChannels = kernel.Shape[1];
        int kernelHeight = kernel.Shape[2];
        int kernelWidth = kernel.Shape[3];

        if (inChannels != kernelInChannels)
        {
            throw new ArgumentException(
                $"Input channels ({inChannels}) must match kernel input channels ({kernelInChannels}).");
        }

        int effectiveKernelHeight = dilation * (kernelHeight - 1) + 1;
        int effectiveKernelWidth = dilation * (kernelWidth - 1) + 1;

        int outputHeight = (height + 2 * padding - effectiveKernelHeight) / stride + 1;
        int outputWidth = (width + 2 * padding - effectiveKernelWidth) / stride + 1;

        if (outputHeight <= 0 || outputWidth <= 0)
        {
            throw new ArgumentException(
                $"Invalid convolution parameters. Output dimensions would be {outputHeight}x{outputWidth}. " +
                $"Ensure stride={stride}, padding={padding}, dilation={dilation} are compatible with input size {height}x{width} and kernel size {kernelHeight}x{kernelWidth}.");
        }

        var result = new Tensor<T>(new[] { batch, outChannels, outputHeight, outputWidth });

        // Perform convolution
        for (int b = 0; b < batch; b++)
        {
            for (int oc = 0; oc < outChannels; oc++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        T sum = numOps.Zero;

                        // Sum over all input channels
                        for (int ic = 0; ic < inChannels; ic++)
                        {
                            // Sum over kernel window
                            for (int kh = 0; kh < kernelHeight; kh++)
                            {
                                for (int kw = 0; kw < kernelWidth; kw++)
                                {
                                    int ih = oh * stride + kh * dilation - padding;
                                    int iw = ow * stride + kw * dilation - padding;

                                    // Check bounds (handle padding)
                                    if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                    {
                                        T inputVal = input[b, ic, ih, iw];
                                        T kernelVal = kernel[oc, ic, kh, kw];
                                        sum = numOps.Add(sum, numOps.Multiply(inputVal, kernelVal));
                                    }
                                }
                            }
                        }

                        result[b, oc, oh, ow] = sum;
                    }
                }
            }
        }

        return result;
    }

    #endregion

    #region Activation Functions

    public Vector<T> Tanh<T>(Vector<T> vector)
    {
        if (vector == null)
            throw new ArgumentNullException(nameof(vector));

        // Use SIMD-optimized Tanh (3-6ÃƒÆ’Ã¢â‚¬â€ speedup for float)
        return TensorPrimitivesHelper<T>.Tanh(vector);
    }

    public Vector<T> Sigmoid<T>(Vector<T> vector)
    {
        if (vector == null)
            throw new ArgumentNullException(nameof(vector));

        // Use SIMD-optimized Sigmoid (3-6ÃƒÆ’Ã¢â‚¬â€ speedup for float)
        return TensorPrimitivesHelper<T>.Sigmoid(vector);
    }

    public Vector<T> ReLU<T>(Vector<T> vector)
    {
        if (vector == null)
            throw new ArgumentNullException(nameof(vector));

        // ReLU(x) = max(0, x)
        // TensorPrimitives doesn't have ReLU directly, but has Max
        // For now, use element-wise max with zero
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputArray = vector.ToArray();
        var outputArray = new T[inputArray.Length];

        // For float, we could use TensorPrimitives.Max with scalar zero
        // For now, manual implementation that works for all types
        for (int i = 0; i < inputArray.Length; i++)
        {
            outputArray[i] = numOps.GreaterThan(inputArray[i], numOps.Zero)
                ? inputArray[i]
                : numOps.Zero;
        }

        return new Vector<T>(outputArray);
    }

    public Tensor<T> Tanh<T>(Tensor<T> tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        // Convert tensor to vector, apply SIMD-optimized Tanh, convert back
        var flatVector = tensor.ToVector();
        var resultVector = TensorPrimitivesHelper<T>.Tanh(flatVector);
        return new Tensor<T>(tensor.Shape, resultVector);
    }

    public Tensor<T> Sigmoid<T>(Tensor<T> tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        // Convert tensor to vector, apply SIMD-optimized Sigmoid, convert back
        var flatVector = tensor.ToVector();
        var resultVector = TensorPrimitivesHelper<T>.Sigmoid(flatVector);
        return new Tensor<T>(tensor.Shape, resultVector);
    }

    public Tensor<T> ReLU<T>(Tensor<T> tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        // ReLU(x) = max(0, x)
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputArray = tensor.ToArray();
        var outputArray = new T[inputArray.Length];

        // Manual implementation that works for all types
        for (int i = 0; i < inputArray.Length; i++)
        {
            outputArray[i] = numOps.GreaterThan(inputArray[i], numOps.Zero)
                ? inputArray[i]
                : numOps.Zero;
        }

        return new Tensor<T>(tensor.Shape, new Vector<T>(outputArray));
    }

    public Vector<T> GELU<T>(Vector<T> vector)
    {
        if (vector == null)
            throw new ArgumentNullException(nameof(vector));

        return TensorPrimitivesHelper<T>.GELU(vector);
    }

    public Vector<T> Mish<T>(Vector<T> vector)
    {
        if (vector == null)
            throw new ArgumentNullException(nameof(vector));

        return TensorPrimitivesHelper<T>.Mish(vector);
    }

    public Vector<T> Swish<T>(Vector<T> vector)
    {
        if (vector == null)
            throw new ArgumentNullException(nameof(vector));

        return TensorPrimitivesHelper<T>.Swish(vector);
    }

    public Vector<T> ELU<T>(Vector<T> vector, double alpha = 1.0)
    {
        if (vector == null)
            throw new ArgumentNullException(nameof(vector));

        return TensorPrimitivesHelper<T>.ELU(vector, alpha);
    }

    public Tensor<T> GELU<T>(Tensor<T> tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var flatVector = tensor.ToVector();
        var resultVector = TensorPrimitivesHelper<T>.GELU(flatVector);
        return new Tensor<T>(tensor.Shape, resultVector);
    }

    public Tensor<T> Mish<T>(Tensor<T> tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var flatVector = tensor.ToVector();
        var resultVector = TensorPrimitivesHelper<T>.Mish(flatVector);
        return new Tensor<T>(tensor.Shape, resultVector);
    }

    public Tensor<T> Swish<T>(Tensor<T> tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var flatVector = tensor.ToVector();
        var resultVector = TensorPrimitivesHelper<T>.Swish(flatVector);
        return new Tensor<T>(tensor.Shape, resultVector);
    }

    public Tensor<T> ELU<T>(Tensor<T> tensor, double alpha = 1.0)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var flatVector = tensor.ToVector();
        var resultVector = TensorPrimitivesHelper<T>.ELU(flatVector, alpha);
        return new Tensor<T>(tensor.Shape, resultVector);
    }

    #endregion

    #region Extended Tensor Operations

    /// <inheritdoc/>
    public Tensor<T> TensorTranspose<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Rank != 2)
            throw new ArgumentException($"TensorTranspose requires a 2D tensor. Got rank {tensor.Rank}.");

        int rows = tensor.Shape[0];
        int cols = tensor.Shape[1];
        var result = new Tensor<T>([cols, rows]);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[j, i] = tensor[i, j];
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorMatMul<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Rank != 2 || b.Rank != 2)
            throw new ArgumentException($"TensorMatMul requires 2D tensors. Got ranks {a.Rank} and {b.Rank}.");

        int m = a.Shape[0];
        int n = a.Shape[1];
        int p = b.Shape[1];

        if (n != b.Shape[0])
            throw new ArgumentException($"Matrix dimensions incompatible: [{m},{n}] x [{b.Shape[0]},{p}]");

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>([m, p]);

        Parallel.For(0, m, i =>
        {
            for (int j = 0; j < p; j++)
            {
                T sum = numOps.Zero;
                for (int k = 0; k < n; k++)
                {
                    sum = numOps.Add(sum, numOps.Multiply(a[i, k], b[k, j]));
                }
                result[i, j] = sum;
            }
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> Conv2D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding, int[] dilation)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (input.Rank != 4) throw new ArgumentException($"Conv2D requires 4D input tensor. Got rank {input.Rank}.", nameof(input));
        if (kernel.Rank != 4) throw new ArgumentException($"Conv2D requires 4D kernel tensor. Got rank {kernel.Rank}.", nameof(kernel));
        if (stride == null || stride.Length != 2) throw new ArgumentException("Stride must be array of 2 elements", nameof(stride));
        if (stride[0] <= 0 || stride[1] <= 0) throw new ArgumentException("Stride elements must be positive", nameof(stride));
        if (padding == null || padding.Length != 2) throw new ArgumentException("Padding must be array of 2 elements", nameof(padding));
        if (dilation == null || dilation.Length != 2) throw new ArgumentException("Dilation must be array of 2 elements", nameof(dilation));
        if (dilation[0] <= 0 || dilation[1] <= 0) throw new ArgumentException("Dilation elements must be positive", nameof(dilation));
        if (input.Shape[1] != kernel.Shape[1]) throw new ArgumentException($"Input channels ({input.Shape[1]}) must match kernel in_channels ({kernel.Shape[1]})");

        int strideH = stride[0], strideW = stride[1];
        int padH = padding[0], padW = padding[1];
        int dilationH = dilation[0], dilationW = dilation[1];

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int outChannels = kernel.Shape[0];
        int kernelHeight = kernel.Shape[2];
        int kernelWidth = kernel.Shape[3];

        int effectiveKernelH = dilationH * (kernelHeight - 1) + 1;
        int effectiveKernelW = dilationW * (kernelWidth - 1) + 1;

        int outputHeight = (height + 2 * padH - effectiveKernelH) / strideH + 1;
        int outputWidth = (width + 2 * padW - effectiveKernelW) / strideW + 1;

        if (outputHeight <= 0 || outputWidth <= 0)
            throw new ArgumentException($"Invalid output dimensions ({outputHeight}x{outputWidth}). Check kernel size, stride, padding, and dilation parameters.");

        var result = new Tensor<T>([batch, outChannels, outputHeight, outputWidth]);
        var inputData = input.ToArray();
        var kernelData = kernel.ToArray();
        var outputData = result.ToArray();

        Parallel.For(0, batch * outChannels, idx =>
        {
            int b = idx / outChannels;
            int oc = idx % outChannels;

            for (int oh = 0; oh < outputHeight; oh++)
            {
                for (int ow = 0; ow < outputWidth; ow++)
                {
                    T sum = numOps.Zero;

                    for (int ic = 0; ic < inChannels; ic++)
                    {
                        for (int kh = 0; kh < kernelHeight; kh++)
                        {
                            for (int kw = 0; kw < kernelWidth; kw++)
                            {
                                int ih = oh * strideH + kh * dilationH - padH;
                                int iw = ow * strideW + kw * dilationW - padW;

                                if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                {
                                    int inputIdx = ((b * inChannels + ic) * height + ih) * width + iw;
                                    int kernelIdx = ((oc * inChannels + ic) * kernelHeight + kh) * kernelWidth + kw;
                                    sum = numOps.Add(sum, numOps.Multiply(inputData[inputIdx], kernelData[kernelIdx]));
                                }
                            }
                        }
                    }

                    int outputIdx = ((b * outChannels + oc) * outputHeight + oh) * outputWidth + ow;
                    outputData[outputIdx] = sum;
                }
            }
        });

        return new Tensor<T>([batch, outChannels, outputHeight, outputWidth], new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public Tensor<T> Conv2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding, int[] dilation)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (inputShape == null || inputShape.Length != 4) throw new ArgumentException("inputShape must be array of 4 elements [batch, inChannels, height, width]", nameof(inputShape));
        if (gradOutput.Rank != 4) throw new ArgumentException($"Conv2DBackwardInput requires 4D gradOutput tensor. Got rank {gradOutput.Rank}.", nameof(gradOutput));
        if (kernel.Rank != 4) throw new ArgumentException($"Conv2DBackwardInput requires 4D kernel tensor. Got rank {kernel.Rank}.", nameof(kernel));
        if (stride == null || stride.Length != 2) throw new ArgumentException("Stride must be array of 2 elements", nameof(stride));
        if (stride[0] <= 0 || stride[1] <= 0) throw new ArgumentException("Stride elements must be positive", nameof(stride));
        if (padding == null || padding.Length != 2) throw new ArgumentException("Padding must be array of 2 elements", nameof(padding));
        if (dilation == null || dilation.Length != 2) throw new ArgumentException("Dilation must be array of 2 elements", nameof(dilation));
        if (dilation[0] <= 0 || dilation[1] <= 0) throw new ArgumentException("Dilation elements must be positive", nameof(dilation));
        if (gradOutput.Shape[0] != inputShape[0]) throw new ArgumentException($"gradOutput batch size ({gradOutput.Shape[0]}) must match inputShape batch size ({inputShape[0]})");
        if (gradOutput.Shape[1] != kernel.Shape[0]) throw new ArgumentException($"gradOutput outChannels ({gradOutput.Shape[1]}) must match kernel outChannels ({kernel.Shape[0]})");
        if (inputShape[1] != kernel.Shape[1]) throw new ArgumentException($"inputShape inChannels ({inputShape[1]}) must match kernel inChannels ({kernel.Shape[1]})");

        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int height = inputShape[2];
        int width = inputShape[3];

        int outChannels = kernel.Shape[0];
        int kernelHeight = kernel.Shape[2];
        int kernelWidth = kernel.Shape[3];

        int strideH = stride[0], strideW = stride[1];
        int padH = padding[0], padW = padding[1];
        int dilationH = dilation[0], dilationW = dilation[1];

        int outputHeight = gradOutput.Shape[2];
        int outputWidth = gradOutput.Shape[3];

        var gradInput = new T[batch * inChannels * height * width];
        var gradOutputData = gradOutput.ToArray();
        var kernelData = kernel.ToArray();

        // Initialize to zero
        for (int i = 0; i < gradInput.Length; i++)
            gradInput[i] = numOps.Zero;

        Parallel.For(0, batch * inChannels, idx =>
        {
            int b = idx / inChannels;
            int ic = idx % inChannels;

            for (int oh = 0; oh < outputHeight; oh++)
            {
                for (int ow = 0; ow < outputWidth; ow++)
                {
                    for (int oc = 0; oc < outChannels; oc++)
                    {
                        int gradOutIdx = ((b * outChannels + oc) * outputHeight + oh) * outputWidth + ow;
                        T gradVal = gradOutputData[gradOutIdx];

                        for (int kh = 0; kh < kernelHeight; kh++)
                        {
                            for (int kw = 0; kw < kernelWidth; kw++)
                            {
                                int ih = oh * strideH + kh * dilationH - padH;
                                int iw = ow * strideW + kw * dilationW - padW;

                                if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                {
                                    int gradInputIdx = ((b * inChannels + ic) * height + ih) * width + iw;
                                    int kernelIdx = ((oc * inChannels + ic) * kernelHeight + kh) * kernelWidth + kw;
                                    // No lock needed - each (batch, inChannel) partition owns disjoint gradInput slices
                                    gradInput[gradInputIdx] = numOps.Add(gradInput[gradInputIdx], numOps.Multiply(gradVal, kernelData[kernelIdx]));
                                }
                            }
                        }
                    }
                }
            }
        });

        return new Tensor<T>(inputShape, new Vector<T>(gradInput));
    }

    /// <inheritdoc/>
    public Tensor<T> Conv2DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding, int[] dilation)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernelShape == null || kernelShape.Length != 4) throw new ArgumentException("kernelShape must be array of 4 elements [outChannels, inChannels, kernelHeight, kernelWidth]", nameof(kernelShape));
        if (gradOutput.Rank != 4) throw new ArgumentException($"Conv2DBackwardKernel requires 4D gradOutput tensor. Got rank {gradOutput.Rank}.", nameof(gradOutput));
        if (input.Rank != 4) throw new ArgumentException($"Conv2DBackwardKernel requires 4D input tensor. Got rank {input.Rank}.", nameof(input));
        if (stride == null || stride.Length != 2) throw new ArgumentException("Stride must be array of 2 elements", nameof(stride));
        if (stride[0] <= 0 || stride[1] <= 0) throw new ArgumentException("Stride elements must be positive", nameof(stride));
        if (padding == null || padding.Length != 2) throw new ArgumentException("Padding must be array of 2 elements", nameof(padding));
        if (dilation == null || dilation.Length != 2) throw new ArgumentException("Dilation must be array of 2 elements", nameof(dilation));
        if (dilation[0] <= 0 || dilation[1] <= 0) throw new ArgumentException("Dilation elements must be positive", nameof(dilation));
        if (gradOutput.Shape[0] != input.Shape[0]) throw new ArgumentException($"gradOutput batch size ({gradOutput.Shape[0]}) must match input batch size ({input.Shape[0]})");
        if (gradOutput.Shape[1] != kernelShape[0]) throw new ArgumentException($"gradOutput outChannels ({gradOutput.Shape[1]}) must match kernelShape outChannels ({kernelShape[0]})");
        if (input.Shape[1] != kernelShape[1]) throw new ArgumentException($"input inChannels ({input.Shape[1]}) must match kernelShape inChannels ({kernelShape[1]})");

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int outChannels = kernelShape[0];
        int kernelHeight = kernelShape[2];
        int kernelWidth = kernelShape[3];

        int strideH = stride[0], strideW = stride[1];
        int padH = padding[0], padW = padding[1];
        int dilationH = dilation[0], dilationW = dilation[1];

        int outputHeight = gradOutput.Shape[2];
        int outputWidth = gradOutput.Shape[3];

        var gradKernel = new T[outChannels * inChannels * kernelHeight * kernelWidth];
        var gradOutputData = gradOutput.ToArray();
        var inputData = input.ToArray();

        for (int i = 0; i < gradKernel.Length; i++)
            gradKernel[i] = numOps.Zero;

        Parallel.For(0, outChannels * inChannels, idx =>
        {
            int oc = idx / inChannels;
            int ic = idx % inChannels;

            for (int kh = 0; kh < kernelHeight; kh++)
            {
                for (int kw = 0; kw < kernelWidth; kw++)
                {
                    T sum = numOps.Zero;

                    for (int b = 0; b < batch; b++)
                    {
                        for (int oh = 0; oh < outputHeight; oh++)
                        {
                            for (int ow = 0; ow < outputWidth; ow++)
                            {
                                int ih = oh * strideH + kh * dilationH - padH;
                                int iw = ow * strideW + kw * dilationW - padW;

                                if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                {
                                    int gradOutIdx = ((b * outChannels + oc) * outputHeight + oh) * outputWidth + ow;
                                    int inputIdx = ((b * inChannels + ic) * height + ih) * width + iw;
                                    sum = numOps.Add(sum, numOps.Multiply(gradOutputData[gradOutIdx], inputData[inputIdx]));
                                }
                            }
                        }
                    }

                    int kernelIdx = ((oc * inChannels + ic) * kernelHeight + kh) * kernelWidth + kw;
                    gradKernel[kernelIdx] = sum;
                }
            }
        });

        return new Tensor<T>(kernelShape, new Vector<T>(gradKernel));
    }

    /// <inheritdoc/>
    public Tensor<T> MaxPool2DWithIndices<T>(Tensor<T> input, int[] poolSize, int[] stride, out int[,,,,] maxIndices)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int poolH = poolSize[0], poolW = poolSize[1];
        int strideH = stride[0], strideW = stride[1];

        if (poolH > height || poolW > width)
            throw new ArgumentException($"Pool size ({poolH}x{poolW}) cannot exceed input spatial dimensions ({height}x{width})");

        int outputHeight = (height - poolH) / strideH + 1;
        int outputWidth = (width - poolW) / strideW + 1;

        if (outputHeight <= 0 || outputWidth <= 0)
            throw new ArgumentException($"Invalid output dimensions ({outputHeight}x{outputWidth}). Check pool size and stride.");

        var result = new Tensor<T>([batch, channels, outputHeight, outputWidth]);
        // Use local variable to avoid capturing out parameter in lambda
        var indices = new int[batch, channels, outputHeight, outputWidth, 2];

        var inputData = input.ToArray();
        var outputData = result.ToArray();

        Parallel.For(0, batch * channels, idx =>
        {
            int b = idx / channels;
            int c = idx % channels;

            for (int oh = 0; oh < outputHeight; oh++)
            {
                for (int ow = 0; ow < outputWidth; ow++)
                {
                    T maxVal = numOps.MinValue;
                    int maxH = 0, maxW = 0;

                    for (int kh = 0; kh < poolH; kh++)
                    {
                        for (int kw = 0; kw < poolW; kw++)
                        {
                            int ih = oh * strideH + kh;
                            int iw = ow * strideW + kw;

                            int inputIdx = ((b * channels + c) * height + ih) * width + iw;
                            T val = inputData[inputIdx];

                            if (numOps.GreaterThan(val, maxVal))
                            {
                                maxVal = val;
                                maxH = ih;
                                maxW = iw;
                            }
                        }
                    }

                    int outputIdx = ((b * channels + c) * outputHeight + oh) * outputWidth + ow;
                    outputData[outputIdx] = maxVal;
                    indices[b, c, oh, ow, 0] = maxH;
                    indices[b, c, oh, ow, 1] = maxW;
                }
            }
        });

        // Assign local variable to out parameter after parallel section
        maxIndices = indices;
        return new Tensor<T>([batch, channels, outputHeight, outputWidth], new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public Tensor<T> MaxPool2DBackward<T>(Tensor<T> gradOutput, int[,,,,] maxIndices, int[] inputShape, int[] poolSize, int[] stride)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = inputShape[0];
        int channels = inputShape[1];
        int height = inputShape[2];
        int width = inputShape[3];

        int outputHeight = gradOutput.Shape[2];
        int outputWidth = gradOutput.Shape[3];

        var gradInput = new T[batch * channels * height * width];
        var gradOutputData = gradOutput.ToArray();

        for (int i = 0; i < gradInput.Length; i++)
            gradInput[i] = numOps.Zero;

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        int maxH = maxIndices[b, c, oh, ow, 0];
                        int maxW = maxIndices[b, c, oh, ow, 1];

                        int gradOutIdx = ((b * channels + c) * outputHeight + oh) * outputWidth + ow;
                        int gradInIdx = ((b * channels + c) * height + maxH) * width + maxW;

                        gradInput[gradInIdx] = numOps.Add(gradInput[gradInIdx], gradOutputData[gradOutIdx]);
                    }
                }
            }
        }

        return new Tensor<T>(inputShape, new Vector<T>(gradInput));
    }

    /// <inheritdoc/>
    public Tensor<T> AvgPool2D<T>(Tensor<T> input, int[] poolSize, int[] stride)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int poolH = poolSize[0], poolW = poolSize[1];
        int strideH = stride[0], strideW = stride[1];

        if (poolH > height || poolW > width)
            throw new ArgumentException($"Pool size ({poolH}x{poolW}) cannot exceed input spatial dimensions ({height}x{width})");

        int outputHeight = (height - poolH) / strideH + 1;
        int outputWidth = (width - poolW) / strideW + 1;

        if (outputHeight <= 0 || outputWidth <= 0)
            throw new ArgumentException($"Invalid output dimensions ({outputHeight}x{outputWidth}). Check pool size and stride.");

        var inputData = input.ToArray();
        var outputData = new T[batch * channels * outputHeight * outputWidth];
        T poolArea = numOps.FromDouble(poolH * poolW);

        Parallel.For(0, batch * channels, idx =>
        {
            int b = idx / channels;
            int c = idx % channels;

            for (int oh = 0; oh < outputHeight; oh++)
            {
                for (int ow = 0; ow < outputWidth; ow++)
                {
                    T sum = numOps.Zero;

                    for (int kh = 0; kh < poolH; kh++)
                    {
                        for (int kw = 0; kw < poolW; kw++)
                        {
                            int ih = oh * strideH + kh;
                            int iw = ow * strideW + kw;
                            int inputIdx = ((b * channels + c) * height + ih) * width + iw;
                            sum = numOps.Add(sum, inputData[inputIdx]);
                        }
                    }

                    int outputIdx = ((b * channels + c) * outputHeight + oh) * outputWidth + ow;
                    outputData[outputIdx] = numOps.Divide(sum, poolArea);
                }
            }
        });

        return new Tensor<T>([batch, channels, outputHeight, outputWidth], new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public Tensor<T> AvgPool2DBackward<T>(Tensor<T> gradOutput, int[] inputShape, int[] poolSize, int[] stride)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = inputShape[0];
        int channels = inputShape[1];
        int height = inputShape[2];
        int width = inputShape[3];

        int poolH = poolSize[0], poolW = poolSize[1];
        int strideH = stride[0], strideW = stride[1];

        int outputHeight = gradOutput.Shape[2];
        int outputWidth = gradOutput.Shape[3];

        var gradInput = new T[batch * channels * height * width];
        var gradOutputData = gradOutput.ToArray();
        T poolArea = numOps.FromDouble(poolH * poolW);

        for (int i = 0; i < gradInput.Length; i++)
            gradInput[i] = numOps.Zero;

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        int gradOutIdx = ((b * channels + c) * outputHeight + oh) * outputWidth + ow;
                        T grad = numOps.Divide(gradOutputData[gradOutIdx], poolArea);

                        for (int kh = 0; kh < poolH; kh++)
                        {
                            for (int kw = 0; kw < poolW; kw++)
                            {
                                int ih = oh * strideH + kh;
                                int iw = ow * strideW + kw;
                                int gradInIdx = ((b * channels + c) * height + ih) * width + iw;
                                gradInput[gradInIdx] = numOps.Add(gradInput[gradInIdx], grad);
                            }
                        }
                    }
                }
            }
        }

        return new Tensor<T>(inputShape, new Vector<T>(gradInput));
    }

    /// <inheritdoc/>
    public Tensor<T> DepthwiseConv2D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int multiplier = kernel.Shape[1];
        int kernelHeight = kernel.Shape[2];
        int kernelWidth = kernel.Shape[3];

        int strideH = stride[0], strideW = stride[1];
        int padH = padding[0], padW = padding[1];

        int outputHeight = (height + 2 * padH - kernelHeight) / strideH + 1;
        int outputWidth = (width + 2 * padW - kernelWidth) / strideW + 1;
        int outChannels = inChannels * multiplier;

        var inputData = input.ToArray();
        var kernelData = kernel.ToArray();
        var outputData = new T[batch * outChannels * outputHeight * outputWidth];

        Parallel.For(0, batch * outChannels, idx =>
        {
            int b = idx / outChannels;
            int oc = idx % outChannels;
            int ic = oc / multiplier;
            int m = oc % multiplier;

            for (int oh = 0; oh < outputHeight; oh++)
            {
                for (int ow = 0; ow < outputWidth; ow++)
                {
                    T sum = numOps.Zero;

                    for (int kh = 0; kh < kernelHeight; kh++)
                    {
                        for (int kw = 0; kw < kernelWidth; kw++)
                        {
                            int ih = oh * strideH + kh - padH;
                            int iw = ow * strideW + kw - padW;

                            if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                            {
                                int inputIdx = ((b * inChannels + ic) * height + ih) * width + iw;
                                int kernelIdx = ((ic * multiplier + m) * kernelHeight + kh) * kernelWidth + kw;
                                sum = numOps.Add(sum, numOps.Multiply(inputData[inputIdx], kernelData[kernelIdx]));
                            }
                        }
                    }

                    int outputIdx = ((b * outChannels + oc) * outputHeight + oh) * outputWidth + ow;
                    outputData[outputIdx] = sum;
                }
            }
        });

        return new Tensor<T>([batch, outChannels, outputHeight, outputWidth], new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public Tensor<T> DepthwiseConv2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int height = inputShape[2];
        int width = inputShape[3];

        int multiplier = kernel.Shape[1];
        int kernelHeight = kernel.Shape[2];
        int kernelWidth = kernel.Shape[3];

        int strideH = stride[0], strideW = stride[1];
        int padH = padding[0], padW = padding[1];

        int outputHeight = gradOutput.Shape[2];
        int outputWidth = gradOutput.Shape[3];
        int outChannels = inChannels * multiplier;

        var gradInput = new T[batch * inChannels * height * width];
        var gradOutputData = gradOutput.ToArray();
        var kernelData = kernel.ToArray();

        for (int i = 0; i < gradInput.Length; i++)
            gradInput[i] = numOps.Zero;

        for (int b = 0; b < batch; b++)
        {
            for (int oc = 0; oc < outChannels; oc++)
            {
                int ic = oc / multiplier;
                int m = oc % multiplier;

                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        int gradOutIdx = ((b * outChannels + oc) * outputHeight + oh) * outputWidth + ow;
                        T gradVal = gradOutputData[gradOutIdx];

                        for (int kh = 0; kh < kernelHeight; kh++)
                        {
                            for (int kw = 0; kw < kernelWidth; kw++)
                            {
                                int ih = oh * strideH + kh - padH;
                                int iw = ow * strideW + kw - padW;

                                if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                {
                                    int gradInIdx = ((b * inChannels + ic) * height + ih) * width + iw;
                                    int kernelIdx = ((ic * multiplier + m) * kernelHeight + kh) * kernelWidth + kw;
                                    gradInput[gradInIdx] = numOps.Add(gradInput[gradInIdx], numOps.Multiply(gradVal, kernelData[kernelIdx]));
                                }
                            }
                        }
                    }
                }
            }
        }

        return new Tensor<T>(inputShape, new Vector<T>(gradInput));
    }

    /// <inheritdoc/>
    public Tensor<T> DepthwiseConv2DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int multiplier = kernelShape[1];
        int kernelHeight = kernelShape[2];
        int kernelWidth = kernelShape[3];

        int strideH = stride[0], strideW = stride[1];
        int padH = padding[0], padW = padding[1];

        int outputHeight = gradOutput.Shape[2];
        int outputWidth = gradOutput.Shape[3];

        var gradKernel = new T[inChannels * multiplier * kernelHeight * kernelWidth];
        var gradOutputData = gradOutput.ToArray();
        var inputData = input.ToArray();

        for (int i = 0; i < gradKernel.Length; i++)
            gradKernel[i] = numOps.Zero;

        for (int ic = 0; ic < inChannels; ic++)
        {
            for (int m = 0; m < multiplier; m++)
            {
                int oc = ic * multiplier + m;

                for (int kh = 0; kh < kernelHeight; kh++)
                {
                    for (int kw = 0; kw < kernelWidth; kw++)
                    {
                        T sum = numOps.Zero;

                        for (int b = 0; b < batch; b++)
                        {
                            for (int oh = 0; oh < outputHeight; oh++)
                            {
                                for (int ow = 0; ow < outputWidth; ow++)
                                {
                                    int ih = oh * strideH + kh - padH;
                                    int iw = ow * strideW + kw - padW;

                                    if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                    {
                                        int gradOutIdx = ((b * (inChannels * multiplier) + oc) * outputHeight + oh) * outputWidth + ow;
                                        int inputIdx = ((b * inChannels + ic) * height + ih) * width + iw;
                                        sum = numOps.Add(sum, numOps.Multiply(gradOutputData[gradOutIdx], inputData[inputIdx]));
                                    }
                                }
                            }
                        }

                        int kernelIdx = ((ic * multiplier + m) * kernelHeight + kh) * kernelWidth + kw;
                        gradKernel[kernelIdx] = sum;
                    }
                }
            }
        }

        return new Tensor<T>(kernelShape, new Vector<T>(gradKernel));
    }

    /// <inheritdoc/>
    public Tensor<T> ConvTranspose2D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding, int[] outputPadding)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (input.Rank != 4) throw new ArgumentException($"ConvTranspose2D requires 4D input tensor. Got rank {input.Rank}.", nameof(input));
        if (kernel.Rank != 4) throw new ArgumentException($"ConvTranspose2D requires 4D kernel tensor. Got rank {kernel.Rank}.", nameof(kernel));
        if (stride == null || stride.Length != 2) throw new ArgumentException("Stride must be array of 2 elements", nameof(stride));
        if (stride[0] <= 0 || stride[1] <= 0) throw new ArgumentException("Stride elements must be positive", nameof(stride));
        if (padding == null || padding.Length != 2) throw new ArgumentException("Padding must be array of 2 elements", nameof(padding));
        if (padding[0] < 0 || padding[1] < 0) throw new ArgumentException("Padding elements must be non-negative", nameof(padding));
        if (outputPadding == null || outputPadding.Length != 2) throw new ArgumentException("OutputPadding must be array of 2 elements", nameof(outputPadding));
        if (outputPadding[0] < 0 || outputPadding[1] < 0) throw new ArgumentException("OutputPadding elements must be non-negative", nameof(outputPadding));
        if (input.Shape[1] != kernel.Shape[0]) throw new ArgumentException($"Input inChannels ({input.Shape[1]}) must match kernel inChannels ({kernel.Shape[0]})");

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int outChannels = kernel.Shape[1];
        int kernelHeight = kernel.Shape[2];
        int kernelWidth = kernel.Shape[3];

        int strideH = stride[0], strideW = stride[1];
        int padH = padding[0], padW = padding[1];
        int outPadH = outputPadding[0], outPadW = outputPadding[1];

        int outputHeight = (height - 1) * strideH - 2 * padH + kernelHeight + outPadH;
        int outputWidth = (width - 1) * strideW - 2 * padW + kernelWidth + outPadW;

        var inputData = input.ToArray();
        var kernelData = kernel.ToArray();
        var outputData = new T[batch * outChannels * outputHeight * outputWidth];

        for (int i = 0; i < outputData.Length; i++)
            outputData[i] = numOps.Zero;

        // Use thread-local accumulation to avoid lock contention
        var lockObj = new object();
        Parallel.For(0, batch * inChannels,
            // Initialize thread-local storage
            () => new T[batch * outChannels * outputHeight * outputWidth],
            // Body
            (idx, state, localOutput) =>
            {
                int b = idx / inChannels;
                int ic = idx % inChannels;

                for (int ih = 0; ih < height; ih++)
                {
                    for (int iw = 0; iw < width; iw++)
                    {
                        int inputIdx = ((b * inChannels + ic) * height + ih) * width + iw;
                        T inputVal = inputData[inputIdx];

                        for (int oc = 0; oc < outChannels; oc++)
                        {
                            for (int kh = 0; kh < kernelHeight; kh++)
                            {
                                for (int kw = 0; kw < kernelWidth; kw++)
                                {
                                    int oh = ih * strideH - padH + kh;
                                    int ow = iw * strideW - padW + kw;

                                    if (oh >= 0 && oh < outputHeight && ow >= 0 && ow < outputWidth)
                                    {
                                        int outputIdx = ((b * outChannels + oc) * outputHeight + oh) * outputWidth + ow;
                                        int kernelIdx = ((ic * outChannels + oc) * kernelHeight + kh) * kernelWidth + kw;
                                        localOutput[outputIdx] = numOps.Add(localOutput[outputIdx], numOps.Multiply(inputVal, kernelData[kernelIdx]));
                                    }
                                }
                            }
                        }
                    }
                }
                return localOutput;
            },
            // Merge thread-local results
            (localOutput) =>
            {
                lock (lockObj)
                {
                    for (int i = 0; i < outputData.Length; i++)
                    {
                        outputData[i] = numOps.Add(outputData[i], localOutput[i]);
                    }
                }
            });

        return new Tensor<T>([batch, outChannels, outputHeight, outputWidth], new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public Tensor<T> ConvTranspose2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding)
    {
        // ConvTranspose2D backward w.r.t. input is equivalent to Conv2D forward
        // Note: This implementation assumes unit dilation. For non-unit dilation, the gradient requires
        // more complex handling (e.g., dilated convolution with flipped kernel).
        var result = Conv2D(gradOutput, kernel, stride, padding, [1, 1]);

        // Validate that the result matches expected input shape
        if (result.Shape[0] != inputShape[0] || result.Shape[1] != inputShape[1] ||
            result.Shape[2] != inputShape[2] || result.Shape[3] != inputShape[3])
        {
            throw new InvalidOperationException(
                $"ConvTranspose2DBackwardInput result shape [{string.Join(",", result.Shape)}] " +
                $"does not match expected inputShape [{string.Join(",", inputShape)}]. " +
                "This may occur with non-standard stride/padding configurations.");
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> ConvTranspose2DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int outChannels = kernelShape[1];
        int kernelHeight = kernelShape[2];
        int kernelWidth = kernelShape[3];

        int strideH = stride[0], strideW = stride[1];
        int padH = padding[0], padW = padding[1];

        int outputHeight = gradOutput.Shape[2];
        int outputWidth = gradOutput.Shape[3];

        var gradKernel = new T[inChannels * outChannels * kernelHeight * kernelWidth];
        var gradOutputData = gradOutput.ToArray();
        var inputData = input.ToArray();

        for (int i = 0; i < gradKernel.Length; i++)
            gradKernel[i] = numOps.Zero;

        for (int ic = 0; ic < inChannels; ic++)
        {
            for (int oc = 0; oc < outChannels; oc++)
            {
                for (int kh = 0; kh < kernelHeight; kh++)
                {
                    for (int kw = 0; kw < kernelWidth; kw++)
                    {
                        T sum = numOps.Zero;

                        for (int b = 0; b < batch; b++)
                        {
                            for (int ih = 0; ih < height; ih++)
                            {
                                for (int iw = 0; iw < width; iw++)
                                {
                                    int oh = ih * strideH - padH + kh;
                                    int ow = iw * strideW - padW + kw;

                                    if (oh >= 0 && oh < outputHeight && ow >= 0 && ow < outputWidth)
                                    {
                                        int gradOutIdx = ((b * outChannels + oc) * outputHeight + oh) * outputWidth + ow;
                                        int inputIdx = ((b * inChannels + ic) * height + ih) * width + iw;
                                        sum = numOps.Add(sum, numOps.Multiply(gradOutputData[gradOutIdx], inputData[inputIdx]));
                                    }
                                }
                            }
                        }

                        int kernelIdx = ((ic * outChannels + oc) * kernelHeight + kh) * kernelWidth + kw;
                        gradKernel[kernelIdx] = sum;
                    }
                }
            }
        }

        return new Tensor<T>(kernelShape, new Vector<T>(gradKernel));
    }

    #region 3D Convolution and Pooling Operations

    /// <inheritdoc/>
    public Tensor<T> Conv3D<T>(Tensor<T> input, Tensor<T> kernel, int stride = 1, int padding = 0, int dilation = 1)
    {
        return Conv3D(input, kernel, [stride, stride, stride], [padding, padding, padding], [dilation, dilation, dilation]);
    }

    /// <inheritdoc/>
    public Tensor<T> Conv3D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding, int[] dilation)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (input.Rank != 5) throw new ArgumentException($"Conv3D requires 5D input tensor [batch, in_channels, depth, height, width]. Got rank {input.Rank}.", nameof(input));
        if (kernel.Rank != 5) throw new ArgumentException($"Conv3D requires 5D kernel tensor [out_channels, in_channels, kernel_depth, kernel_height, kernel_width]. Got rank {kernel.Rank}.", nameof(kernel));
        if (stride == null || stride.Length != 3) throw new ArgumentException("Stride must be array of 3 elements [strideD, strideH, strideW].", nameof(stride));
        if (stride[0] <= 0 || stride[1] <= 0 || stride[2] <= 0) throw new ArgumentException("Stride elements must be positive.", nameof(stride));
        if (padding == null || padding.Length != 3) throw new ArgumentException("Padding must be array of 3 elements [padD, padH, padW].", nameof(padding));
        if (dilation == null || dilation.Length != 3) throw new ArgumentException("Dilation must be array of 3 elements [dilationD, dilationH, dilationW].", nameof(dilation));
        if (dilation[0] <= 0 || dilation[1] <= 0 || dilation[2] <= 0) throw new ArgumentException("Dilation elements must be positive.", nameof(dilation));
        if (input.Shape[1] != kernel.Shape[1]) throw new ArgumentException($"Input channels ({input.Shape[1]}) must match kernel in_channels ({kernel.Shape[1]}).");

        int strideD = stride[0], strideH = stride[1], strideW = stride[2];
        int padD = padding[0], padH = padding[1], padW = padding[2];
        int dilationD = dilation[0], dilationH = dilation[1], dilationW = dilation[2];

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int depth = input.Shape[2];
        int height = input.Shape[3];
        int width = input.Shape[4];

        int outChannels = kernel.Shape[0];
        int kernelDepth = kernel.Shape[2];
        int kernelHeight = kernel.Shape[3];
        int kernelWidth = kernel.Shape[4];

        int effectiveKernelD = dilationD * (kernelDepth - 1) + 1;
        int effectiveKernelH = dilationH * (kernelHeight - 1) + 1;
        int effectiveKernelW = dilationW * (kernelWidth - 1) + 1;

        int outputDepth = (depth + 2 * padD - effectiveKernelD) / strideD + 1;
        int outputHeight = (height + 2 * padH - effectiveKernelH) / strideH + 1;
        int outputWidth = (width + 2 * padW - effectiveKernelW) / strideW + 1;

        if (outputDepth <= 0 || outputHeight <= 0 || outputWidth <= 0)
        {
            throw new ArgumentException(
                $"Invalid output dimensions ({outputDepth}x{outputHeight}x{outputWidth}). " +
                $"Check kernel size, stride, padding, and dilation parameters for input size {depth}x{height}x{width}.");
        }

        var result = new Tensor<T>([batch, outChannels, outputDepth, outputHeight, outputWidth]);
        var inputData = input.ToArray();
        var kernelData = kernel.ToArray();
        var outputData = result.ToArray();

        // Parallel over batch * outChannels for maximum parallelism
        Parallel.For(0, batch * outChannels, idx =>
        {
            int b = idx / outChannels;
            int oc = idx % outChannels;

            for (int od = 0; od < outputDepth; od++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        T sum = numOps.Zero;

                        for (int ic = 0; ic < inChannels; ic++)
                        {
                            for (int kd = 0; kd < kernelDepth; kd++)
                            {
                                int id = od * strideD + kd * dilationD - padD;
                                if (id < 0 || id >= depth) continue;

                                for (int kh = 0; kh < kernelHeight; kh++)
                                {
                                    int ih = oh * strideH + kh * dilationH - padH;
                                    if (ih < 0 || ih >= height) continue;

                                    for (int kw = 0; kw < kernelWidth; kw++)
                                    {
                                        int iw = ow * strideW + kw * dilationW - padW;
                                        if (iw < 0 || iw >= width) continue;

                                        int inputIdx = (((b * inChannels + ic) * depth + id) * height + ih) * width + iw;
                                        int kernelIdx = (((oc * inChannels + ic) * kernelDepth + kd) * kernelHeight + kh) * kernelWidth + kw;
                                        sum = numOps.Add(sum, numOps.Multiply(inputData[inputIdx], kernelData[kernelIdx]));
                                    }
                                }
                            }
                        }

                        int outputIdx = (((b * outChannels + oc) * outputDepth + od) * outputHeight + oh) * outputWidth + ow;
                        outputData[outputIdx] = sum;
                    }
                }
            }
        });

        return new Tensor<T>([batch, outChannels, outputDepth, outputHeight, outputWidth], new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public Tensor<T> Conv3DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding, int[] dilation)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (inputShape == null || inputShape.Length != 5) throw new ArgumentException("Input shape must be array of 5 elements [batch, in_channels, depth, height, width].", nameof(inputShape));
        if (stride == null || stride.Length != 3) throw new ArgumentException("Stride must be array of 3 elements.", nameof(stride));
        if (padding == null || padding.Length != 3) throw new ArgumentException("Padding must be array of 3 elements.", nameof(padding));
        if (dilation == null || dilation.Length != 3) throw new ArgumentException("Dilation must be array of 3 elements.", nameof(dilation));

        // Rank and shape validation
        if (gradOutput.Rank != 5) throw new ArgumentException($"Conv3DBackwardInput requires 5D gradOutput tensor. Got rank {gradOutput.Rank}.", nameof(gradOutput));
        if (kernel.Rank != 5) throw new ArgumentException($"Conv3DBackwardInput requires 5D kernel tensor. Got rank {kernel.Rank}.", nameof(kernel));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int depth = inputShape[2];
        int height = inputShape[3];
        int width = inputShape[4];

        int outChannels = kernel.Shape[0];
        int kernelDepth = kernel.Shape[2];
        int kernelHeight = kernel.Shape[3];
        int kernelWidth = kernel.Shape[4];

        // Validate shape consistency
        if (gradOutput.Shape[0] != batch)
            throw new ArgumentException($"gradOutput batch size ({gradOutput.Shape[0]}) must match inputShape batch size ({batch}).", nameof(gradOutput));
        if (gradOutput.Shape[1] != outChannels)
            throw new ArgumentException($"gradOutput outChannels ({gradOutput.Shape[1]}) must match kernel out_channels ({outChannels}).", nameof(gradOutput));
        if (inputShape[1] != kernel.Shape[1])
            throw new ArgumentException($"inputShape in_channels ({inputShape[1]}) must match kernel in_channels ({kernel.Shape[1]}).", nameof(inputShape));

        int strideD = stride[0], strideH = stride[1], strideW = stride[2];
        int padD = padding[0], padH = padding[1], padW = padding[2];
        int dilationD = dilation[0], dilationH = dilation[1], dilationW = dilation[2];

        int outputDepth = gradOutput.Shape[2];
        int outputHeight = gradOutput.Shape[3];
        int outputWidth = gradOutput.Shape[4];

        var gradInputData = new T[batch * inChannels * depth * height * width];
        var gradOutputData = gradOutput.ToArray();
        var kernelData = kernel.ToArray();

        // Initialize to zero
        for (int i = 0; i < gradInputData.Length; i++)
            gradInputData[i] = numOps.Zero;

        // Parallel over (batch, inChannels) - each pair owns disjoint gradInput slices
        // so direct writes are race-free without thread-local buffers
        Parallel.For(0, batch * inChannels, idx =>
        {
            int b = idx / inChannels;
            int ic = idx % inChannels;

            for (int oc = 0; oc < outChannels; oc++)
            {
                for (int od = 0; od < outputDepth; od++)
                {
                    for (int oh = 0; oh < outputHeight; oh++)
                    {
                        for (int ow = 0; ow < outputWidth; ow++)
                        {
                            int gradOutIdx = (((b * outChannels + oc) * outputDepth + od) * outputHeight + oh) * outputWidth + ow;
                            T gradVal = gradOutputData[gradOutIdx];

                            for (int kd = 0; kd < kernelDepth; kd++)
                            {
                                int id = od * strideD + kd * dilationD - padD;
                                if (id < 0 || id >= depth) continue;

                                for (int kh = 0; kh < kernelHeight; kh++)
                                {
                                    int ih = oh * strideH + kh * dilationH - padH;
                                    if (ih < 0 || ih >= height) continue;

                                    for (int kw = 0; kw < kernelWidth; kw++)
                                    {
                                        int iw = ow * strideW + kw * dilationW - padW;
                                        if (iw < 0 || iw >= width) continue;

                                        int inputIdx = (((b * inChannels + ic) * depth + id) * height + ih) * width + iw;
                                        int kernelIdx = (((oc * inChannels + ic) * kernelDepth + kd) * kernelHeight + kh) * kernelWidth + kw;
                                        gradInputData[inputIdx] = numOps.Add(gradInputData[inputIdx], numOps.Multiply(gradVal, kernelData[kernelIdx]));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

        return new Tensor<T>(inputShape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public Tensor<T> Conv3DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding, int[] dilation)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernelShape == null || kernelShape.Length != 5) throw new ArgumentException("Kernel shape must be array of 5 elements [out_channels, in_channels, kernel_depth, kernel_height, kernel_width].", nameof(kernelShape));
        if (stride == null || stride.Length != 3) throw new ArgumentException("Stride must be array of 3 elements.", nameof(stride));
        if (padding == null || padding.Length != 3) throw new ArgumentException("Padding must be array of 3 elements.", nameof(padding));
        if (dilation == null || dilation.Length != 3) throw new ArgumentException("Dilation must be array of 3 elements.", nameof(dilation));

        // Rank and shape validation
        if (gradOutput.Rank != 5) throw new ArgumentException($"Conv3DBackwardKernel requires 5D gradOutput tensor. Got rank {gradOutput.Rank}.", nameof(gradOutput));
        if (input.Rank != 5) throw new ArgumentException($"Conv3DBackwardKernel requires 5D input tensor. Got rank {input.Rank}.", nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int depth = input.Shape[2];
        int height = input.Shape[3];
        int width = input.Shape[4];

        int outChannels = kernelShape[0];
        int kernelDepth = kernelShape[2];
        int kernelHeight = kernelShape[3];
        int kernelWidth = kernelShape[4];

        // Validate shape consistency
        if (gradOutput.Shape[0] != batch)
            throw new ArgumentException($"gradOutput batch size ({gradOutput.Shape[0]}) must match input batch size ({batch}).", nameof(gradOutput));
        if (gradOutput.Shape[1] != outChannels)
            throw new ArgumentException($"gradOutput outChannels ({gradOutput.Shape[1]}) must match kernelShape out_channels ({outChannels}).", nameof(gradOutput));
        if (input.Shape[1] != kernelShape[1])
            throw new ArgumentException($"input in_channels ({input.Shape[1]}) must match kernelShape in_channels ({kernelShape[1]}).", nameof(input));

        int strideD = stride[0], strideH = stride[1], strideW = stride[2];
        int padD = padding[0], padH = padding[1], padW = padding[2];
        int dilationD = dilation[0], dilationH = dilation[1], dilationW = dilation[2];

        int outputDepth = gradOutput.Shape[2];
        int outputHeight = gradOutput.Shape[3];
        int outputWidth = gradOutput.Shape[4];

        var gradKernelData = new T[outChannels * inChannels * kernelDepth * kernelHeight * kernelWidth];
        var gradOutputData = gradOutput.ToArray();
        var inputData = input.ToArray();

        // Initialize to zero
        for (int i = 0; i < gradKernelData.Length; i++)
            gradKernelData[i] = numOps.Zero;

        // Parallel over outChannels * inChannels for kernel gradient computation
        Parallel.For(0, outChannels * inChannels, idx =>
        {
            int oc = idx / inChannels;
            int ic = idx % inChannels;

            for (int kd = 0; kd < kernelDepth; kd++)
            {
                for (int kh = 0; kh < kernelHeight; kh++)
                {
                    for (int kw = 0; kw < kernelWidth; kw++)
                    {
                        T sum = numOps.Zero;

                        for (int b = 0; b < batch; b++)
                        {
                            for (int od = 0; od < outputDepth; od++)
                            {
                                int id = od * strideD + kd * dilationD - padD;
                                if (id < 0 || id >= depth) continue;

                                for (int oh = 0; oh < outputHeight; oh++)
                                {
                                    int ih = oh * strideH + kh * dilationH - padH;
                                    if (ih < 0 || ih >= height) continue;

                                    for (int ow = 0; ow < outputWidth; ow++)
                                    {
                                        int iw = ow * strideW + kw * dilationW - padW;
                                        if (iw < 0 || iw >= width) continue;

                                        int gradOutIdx = (((b * outChannels + oc) * outputDepth + od) * outputHeight + oh) * outputWidth + ow;
                                        int inputIdx = (((b * inChannels + ic) * depth + id) * height + ih) * width + iw;
                                        sum = numOps.Add(sum, numOps.Multiply(gradOutputData[gradOutIdx], inputData[inputIdx]));
                                    }
                                }
                            }
                        }

                        int kernelIdx = (((oc * inChannels + ic) * kernelDepth + kd) * kernelHeight + kh) * kernelWidth + kw;
                        gradKernelData[kernelIdx] = sum;
                    }
                }
            }
        });

        return new Tensor<T>(kernelShape, new Vector<T>(gradKernelData));
    }

    /// <inheritdoc/>
    public Tensor<T> MaxPool3D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0)
    {
        if (stride == 0) stride = poolSize;
        return MaxPool3D(input, [poolSize, poolSize, poolSize], [stride, stride, stride], [padding, padding, padding]);
    }

    /// <inheritdoc/>
    public Tensor<T> MaxPool3D<T>(Tensor<T> input, int[] poolSize, int[] stride, int[] padding)
    {
        // Use the core implementation directly - we don't need indices for simple forward
        return MaxPool3DCore(input, poolSize, stride, padding);
    }

    /// <summary>
    /// Core implementation of MaxPool3D without index tracking for simple forward pass.
    /// </summary>
    private Tensor<T> MaxPool3DCore<T>(Tensor<T> input, int[] poolSize, int[] stride, int[] padding)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 5) throw new ArgumentException($"MaxPool3D requires 5D input tensor [batch, channels, depth, height, width]. Got rank {input.Rank}.", nameof(input));
        if (poolSize == null || poolSize.Length != 3) throw new ArgumentException("Pool size must be array of 3 elements [poolD, poolH, poolW].", nameof(poolSize));
        if (stride == null || stride.Length != 3) throw new ArgumentException("Stride must be array of 3 elements [strideD, strideH, strideW].", nameof(stride));
        if (padding == null || padding.Length != 3) throw new ArgumentException("Padding must be array of 3 elements [padD, padH, padW].", nameof(padding));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int depth = input.Shape[2];
        int height = input.Shape[3];
        int width = input.Shape[4];

        int poolD = poolSize[0], poolH = poolSize[1], poolW = poolSize[2];
        int strideD = stride[0], strideH = stride[1], strideW = stride[2];
        int padD = padding[0], padH = padding[1], padW = padding[2];

        int outputDepth = (depth + 2 * padD - poolD) / strideD + 1;
        int outputHeight = (height + 2 * padH - poolH) / strideH + 1;
        int outputWidth = (width + 2 * padW - poolW) / strideW + 1;

        if (outputDepth <= 0 || outputHeight <= 0 || outputWidth <= 0)
        {
            throw new ArgumentException(
                $"Invalid output dimensions ({outputDepth}x{outputHeight}x{outputWidth}). " +
                $"Check pool size, stride, and padding parameters for input size {depth}x{height}x{width}.");
        }

        var result = new Tensor<T>([batch, channels, outputDepth, outputHeight, outputWidth]);
        var inputData = input.ToArray();
        var outputData = result.ToArray();

        // Parallel over batch * channels
        Parallel.For(0, batch * channels, idx =>
        {
            int b = idx / channels;
            int c = idx % channels;

            for (int od = 0; od < outputDepth; od++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        T maxVal = numOps.MinValue;
                        bool foundValid = false;

                        for (int pd = 0; pd < poolD; pd++)
                        {
                            int id = od * strideD + pd - padD;
                            if (id < 0 || id >= depth) continue;

                            for (int ph = 0; ph < poolH; ph++)
                            {
                                int ih = oh * strideH + ph - padH;
                                if (ih < 0 || ih >= height) continue;

                                for (int pw = 0; pw < poolW; pw++)
                                {
                                    int iw = ow * strideW + pw - padW;
                                    if (iw < 0 || iw >= width) continue;

                                    int inputIdx = (((b * channels + c) * depth + id) * height + ih) * width + iw;
                                    T val = inputData[inputIdx];
                                    if (!foundValid || numOps.GreaterThan(val, maxVal))
                                    {
                                        maxVal = val;
                                        foundValid = true;
                                    }
                                }
                            }
                        }

                        int outputIdx = (((b * channels + c) * outputDepth + od) * outputHeight + oh) * outputWidth + ow;
                        outputData[outputIdx] = foundValid ? maxVal : numOps.Zero;
                    }
                }
            }
        });

        return new Tensor<T>([batch, channels, outputDepth, outputHeight, outputWidth], new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public Tensor<T> MaxPool3DWithIndices<T>(Tensor<T> input, int[] poolSize, int[] stride, out int[,,,,,] maxIndices)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 5) throw new ArgumentException($"MaxPool3D requires 5D input tensor [batch, channels, depth, height, width]. Got rank {input.Rank}.", nameof(input));
        if (poolSize == null || poolSize.Length != 3) throw new ArgumentException("Pool size must be array of 3 elements [poolD, poolH, poolW].", nameof(poolSize));
        if (stride == null || stride.Length != 3) throw new ArgumentException("Stride must be array of 3 elements [strideD, strideH, strideW].", nameof(stride));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int depth = input.Shape[2];
        int height = input.Shape[3];
        int width = input.Shape[4];

        int poolD = poolSize[0], poolH = poolSize[1], poolW = poolSize[2];
        int strideD = stride[0], strideH = stride[1], strideW = stride[2];

        int outputDepth = (depth - poolD) / strideD + 1;
        int outputHeight = (height - poolH) / strideH + 1;
        int outputWidth = (width - poolW) / strideW + 1;

        if (outputDepth <= 0 || outputHeight <= 0 || outputWidth <= 0)
        {
            throw new ArgumentException(
                $"Invalid output dimensions ({outputDepth}x{outputHeight}x{outputWidth}). " +
                $"Check pool size and stride parameters for input size {depth}x{height}x{width}.");
        }

        var result = new Tensor<T>([batch, channels, outputDepth, outputHeight, outputWidth]);
        var inputData = input.ToArray();
        var outputData = result.ToArray();
        var localMaxIndices = new int[batch, channels, outputDepth, outputHeight, outputWidth, 3];

        // Parallel over batch * channels
        Parallel.For(0, batch * channels, idx =>
        {
            int b = idx / channels;
            int c = idx % channels;

            for (int od = 0; od < outputDepth; od++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        T maxVal = numOps.MinValue;
                        int maxId = 0, maxIh = 0, maxIw = 0;
                        bool foundValid = false;

                        for (int pd = 0; pd < poolD; pd++)
                        {
                            int id = od * strideD + pd;
                            if (id >= depth) continue;

                            for (int ph = 0; ph < poolH; ph++)
                            {
                                int ih = oh * strideH + ph;
                                if (ih >= height) continue;

                                for (int pw = 0; pw < poolW; pw++)
                                {
                                    int iw = ow * strideW + pw;
                                    if (iw >= width) continue;

                                    int inputIdx = (((b * channels + c) * depth + id) * height + ih) * width + iw;
                                    T val = inputData[inputIdx];
                                    if (!foundValid || numOps.GreaterThan(val, maxVal))
                                    {
                                        maxVal = val;
                                        maxId = id;
                                        maxIh = ih;
                                        maxIw = iw;
                                        foundValid = true;
                                    }
                                }
                            }
                        }

                        int outputIdx = (((b * channels + c) * outputDepth + od) * outputHeight + oh) * outputWidth + ow;
                        outputData[outputIdx] = foundValid ? maxVal : numOps.Zero;
                        localMaxIndices[b, c, od, oh, ow, 0] = maxId;
                        localMaxIndices[b, c, od, oh, ow, 1] = maxIh;
                        localMaxIndices[b, c, od, oh, ow, 2] = maxIw;
                    }
                }
            }
        });

        maxIndices = localMaxIndices;
        return new Tensor<T>([batch, channels, outputDepth, outputHeight, outputWidth], new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public Tensor<T> MaxPool3DBackward<T>(Tensor<T> gradOutput, int[,,,,,] maxIndices, int[] inputShape, int[] poolSize, int[] stride)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (maxIndices == null) throw new ArgumentNullException(nameof(maxIndices));
        if (inputShape == null || inputShape.Length != 5) throw new ArgumentException("Input shape must be array of 5 elements [batch, channels, depth, height, width].", nameof(inputShape));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = inputShape[0];
        int channels = inputShape[1];
        int depth = inputShape[2];
        int height = inputShape[3];
        int width = inputShape[4];

        int outputDepth = gradOutput.Shape[2];
        int outputHeight = gradOutput.Shape[3];
        int outputWidth = gradOutput.Shape[4];

        var gradInputData = new T[batch * channels * depth * height * width];
        var gradOutputData = gradOutput.ToArray();

        // Initialize to zero
        for (int i = 0; i < gradInputData.Length; i++)
            gradInputData[i] = numOps.Zero;

        // Use thread-local accumulators for thread safety
        int numThreads = Environment.ProcessorCount;
        var localGradInputs = new T[numThreads][];
        for (int t = 0; t < numThreads; t++)
        {
            localGradInputs[t] = new T[gradInputData.Length];
            for (int i = 0; i < gradInputData.Length; i++)
                localGradInputs[t][i] = numOps.Zero;
        }

        // Parallel over batch * channels
        Parallel.For(0, batch * channels, idx =>
        {
            int threadId = Environment.CurrentManagedThreadId % numThreads;
            var localGrad = localGradInputs[threadId];

            int b = idx / channels;
            int c = idx % channels;

            for (int od = 0; od < outputDepth; od++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        int gradOutIdx = (((b * channels + c) * outputDepth + od) * outputHeight + oh) * outputWidth + ow;
                        T gradVal = gradOutputData[gradOutIdx];

                        int id = maxIndices[b, c, od, oh, ow, 0];
                        int ih = maxIndices[b, c, od, oh, ow, 1];
                        int iw = maxIndices[b, c, od, oh, ow, 2];

                        int inputIdx = (((b * channels + c) * depth + id) * height + ih) * width + iw;
                        localGrad[inputIdx] = numOps.Add(localGrad[inputIdx], gradVal);
                    }
                }
            }
        });

        // Merge thread-local results
        for (int t = 0; t < numThreads; t++)
        {
            for (int i = 0; i < gradInputData.Length; i++)
            {
                gradInputData[i] = numOps.Add(gradInputData[i], localGradInputs[t][i]);
            }
        }

        return new Tensor<T>(inputShape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public Tensor<T> AvgPool3D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0)
    {
        if (stride == 0) stride = poolSize;
        return AvgPool3D(input, [poolSize, poolSize, poolSize], [stride, stride, stride], [padding, padding, padding]);
    }

    /// <inheritdoc/>
    public Tensor<T> AvgPool3D<T>(Tensor<T> input, int[] poolSize, int[] stride, int[] padding)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 5) throw new ArgumentException($"AvgPool3D requires 5D input tensor [batch, channels, depth, height, width]. Got rank {input.Rank}.", nameof(input));
        if (poolSize == null || poolSize.Length != 3) throw new ArgumentException("Pool size must be array of 3 elements [poolD, poolH, poolW].", nameof(poolSize));
        if (stride == null || stride.Length != 3) throw new ArgumentException("Stride must be array of 3 elements [strideD, strideH, strideW].", nameof(stride));
        if (padding == null || padding.Length != 3) throw new ArgumentException("Padding must be array of 3 elements [padD, padH, padW].", nameof(padding));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int depth = input.Shape[2];
        int height = input.Shape[3];
        int width = input.Shape[4];

        int poolD = poolSize[0], poolH = poolSize[1], poolW = poolSize[2];
        int strideD = stride[0], strideH = stride[1], strideW = stride[2];
        int padD = padding[0], padH = padding[1], padW = padding[2];

        int outputDepth = (depth + 2 * padD - poolD) / strideD + 1;
        int outputHeight = (height + 2 * padH - poolH) / strideH + 1;
        int outputWidth = (width + 2 * padW - poolW) / strideW + 1;

        if (outputDepth <= 0 || outputHeight <= 0 || outputWidth <= 0)
        {
            throw new ArgumentException(
                $"Invalid output dimensions ({outputDepth}x{outputHeight}x{outputWidth}). " +
                $"Check pool size, stride, and padding parameters for input size {depth}x{height}x{width}.");
        }

        var result = new Tensor<T>([batch, channels, outputDepth, outputHeight, outputWidth]);
        var inputData = input.ToArray();
        var outputData = result.ToArray();

        // Parallel over batch * channels
        Parallel.For(0, batch * channels, idx =>
        {
            int b = idx / channels;
            int c = idx % channels;

            for (int od = 0; od < outputDepth; od++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        T sum = numOps.Zero;
                        int count = 0;

                        for (int pd = 0; pd < poolD; pd++)
                        {
                            int id = od * strideD + pd - padD;
                            if (id < 0 || id >= depth) continue;

                            for (int ph = 0; ph < poolH; ph++)
                            {
                                int ih = oh * strideH + ph - padH;
                                if (ih < 0 || ih >= height) continue;

                                for (int pw = 0; pw < poolW; pw++)
                                {
                                    int iw = ow * strideW + pw - padW;
                                    if (iw < 0 || iw >= width) continue;

                                    int inputIdx = (((b * channels + c) * depth + id) * height + ih) * width + iw;
                                    sum = numOps.Add(sum, inputData[inputIdx]);
                                    count++;
                                }
                            }
                        }

                        int outputIdx = (((b * channels + c) * outputDepth + od) * outputHeight + oh) * outputWidth + ow;
                        outputData[outputIdx] = count > 0 ? numOps.Divide(sum, numOps.FromDouble(count)) : numOps.Zero;
                    }
                }
            }
        });

        return new Tensor<T>([batch, channels, outputDepth, outputHeight, outputWidth], new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public Tensor<T> AvgPool3DBackward<T>(Tensor<T> gradOutput, int[] inputShape, int[] poolSize, int[] stride, int[] padding)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (inputShape == null || inputShape.Length != 5) throw new ArgumentException("Input shape must be array of 5 elements [batch, channels, depth, height, width].", nameof(inputShape));
        if (poolSize == null || poolSize.Length != 3) throw new ArgumentException("Pool size must be array of 3 elements.", nameof(poolSize));
        if (stride == null || stride.Length != 3) throw new ArgumentException("Stride must be array of 3 elements.", nameof(stride));
        if (padding == null || padding.Length != 3) throw new ArgumentException("Padding must be array of 3 elements.", nameof(padding));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = inputShape[0];
        int channels = inputShape[1];
        int depth = inputShape[2];
        int height = inputShape[3];
        int width = inputShape[4];

        int poolD = poolSize[0], poolH = poolSize[1], poolW = poolSize[2];
        int strideD = stride[0], strideH = stride[1], strideW = stride[2];
        int padD = padding[0], padH = padding[1], padW = padding[2];

        int outputDepth = gradOutput.Shape[2];
        int outputHeight = gradOutput.Shape[3];
        int outputWidth = gradOutput.Shape[4];

        var gradInputData = new T[batch * channels * depth * height * width];
        var gradOutputData = gradOutput.ToArray();

        // Initialize to zero
        for (int i = 0; i < gradInputData.Length; i++)
            gradInputData[i] = numOps.Zero;

        // Use thread-local accumulators for thread safety
        int numThreads = Environment.ProcessorCount;
        var localGradInputs = new T[numThreads][];
        for (int t = 0; t < numThreads; t++)
        {
            localGradInputs[t] = new T[gradInputData.Length];
            for (int i = 0; i < gradInputData.Length; i++)
                localGradInputs[t][i] = numOps.Zero;
        }

        // Parallel over batch * channels
        Parallel.For(0, batch * channels, idx =>
        {
            int threadId = Environment.CurrentManagedThreadId % numThreads;
            var localGrad = localGradInputs[threadId];

            int b = idx / channels;
            int c = idx % channels;

            for (int od = 0; od < outputDepth; od++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        // Count valid positions in this pool window
                        int count = 0;
                        for (int pd = 0; pd < poolD; pd++)
                        {
                            int id = od * strideD + pd - padD;
                            if (id < 0 || id >= depth) continue;
                            for (int ph = 0; ph < poolH; ph++)
                            {
                                int ih = oh * strideH + ph - padH;
                                if (ih < 0 || ih >= height) continue;
                                for (int pw = 0; pw < poolW; pw++)
                                {
                                    int iw = ow * strideW + pw - padW;
                                    if (iw < 0 || iw >= width) continue;
                                    count++;
                                }
                            }
                        }

                        if (count == 0) continue;

                        int gradOutIdx = (((b * channels + c) * outputDepth + od) * outputHeight + oh) * outputWidth + ow;
                        T gradVal = numOps.Divide(gradOutputData[gradOutIdx], numOps.FromDouble(count));

                        // Distribute gradient equally to all contributing positions
                        for (int pd = 0; pd < poolD; pd++)
                        {
                            int id = od * strideD + pd - padD;
                            if (id < 0 || id >= depth) continue;

                            for (int ph = 0; ph < poolH; ph++)
                            {
                                int ih = oh * strideH + ph - padH;
                                if (ih < 0 || ih >= height) continue;

                                for (int pw = 0; pw < poolW; pw++)
                                {
                                    int iw = ow * strideW + pw - padW;
                                    if (iw < 0 || iw >= width) continue;

                                    int inputIdx = (((b * channels + c) * depth + id) * height + ih) * width + iw;
                                    localGrad[inputIdx] = numOps.Add(localGrad[inputIdx], gradVal);
                                }
                            }
                        }
                    }
                }
            }
        });

        // Merge thread-local results
        for (int t = 0; t < numThreads; t++)
        {
            for (int i = 0; i < gradInputData.Length; i++)
            {
                gradInputData[i] = numOps.Add(gradInputData[i], localGradInputs[t][i]);
            }
        }

        return new Tensor<T>(inputShape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public Tensor<T> Upsample3D<T>(Tensor<T> input, int scaleD, int scaleH, int scaleW)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 5) throw new ArgumentException($"Upsample3D requires 5D input tensor [batch, channels, depth, height, width]. Got rank {input.Rank}.", nameof(input));
        if (scaleD <= 0 || scaleH <= 0 || scaleW <= 0) throw new ArgumentException("Scale factors must be positive.");

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int depth = input.Shape[2];
        int height = input.Shape[3];
        int width = input.Shape[4];

        int outDepth = depth * scaleD;
        int outHeight = height * scaleH;
        int outWidth = width * scaleW;

        var outputData = new T[batch * channels * outDepth * outHeight * outWidth];
        var inputData = input.ToArray();

        // Use parallel processing over batch and channels
        Parallel.For(0, batch * channels, bc =>
        {
            int b = bc / channels;
            int c = bc % channels;

            for (int od = 0; od < outDepth; od++)
            {
                int id = od / scaleD;
                for (int oh = 0; oh < outHeight; oh++)
                {
                    int ih = oh / scaleH;
                    for (int ow = 0; ow < outWidth; ow++)
                    {
                        int iw = ow / scaleW;

                        int inputIdx = (((b * channels + c) * depth + id) * height + ih) * width + iw;
                        int outputIdx = (((b * channels + c) * outDepth + od) * outHeight + oh) * outWidth + ow;
                        outputData[outputIdx] = inputData[inputIdx];
                    }
                }
            }
        });

        return new Tensor<T>([batch, channels, outDepth, outHeight, outWidth], new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public Tensor<T> Upsample3DBackward<T>(Tensor<T> gradOutput, int[] inputShape, int scaleD, int scaleH, int scaleW)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (inputShape == null || inputShape.Length != 5) throw new ArgumentException("Input shape must be array of 5 elements [batch, channels, depth, height, width].", nameof(inputShape));
        if (scaleD <= 0 || scaleH <= 0 || scaleW <= 0) throw new ArgumentException("Scale factors must be positive.");

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = inputShape[0];
        int channels = inputShape[1];
        int depth = inputShape[2];
        int height = inputShape[3];
        int width = inputShape[4];

        int outDepth = depth * scaleD;
        int outHeight = height * scaleH;
        int outWidth = width * scaleW;

        var gradInputData = new T[batch * channels * depth * height * width];
        var gradOutputData = gradOutput.ToArray();

        // Use thread-local accumulators to avoid contention
        int numThreads = Environment.ProcessorCount;
        var localGradInputs = new T[numThreads][];
        for (int t = 0; t < numThreads; t++)
        {
            localGradInputs[t] = new T[gradInputData.Length];
        }

        Parallel.For(0, batch * channels, bc =>
        {
            int threadId = Environment.CurrentManagedThreadId % numThreads;
            var localGrad = localGradInputs[threadId];
            int b = bc / channels;
            int c = bc % channels;

            for (int od = 0; od < outDepth; od++)
            {
                int id = od / scaleD;
                for (int oh = 0; oh < outHeight; oh++)
                {
                    int ih = oh / scaleH;
                    for (int ow = 0; ow < outWidth; ow++)
                    {
                        int iw = ow / scaleW;

                        int inputIdx = (((b * channels + c) * depth + id) * height + ih) * width + iw;
                        int outputIdx = (((b * channels + c) * outDepth + od) * outHeight + oh) * outWidth + ow;
                        localGrad[inputIdx] = numOps.Add(localGrad[inputIdx], gradOutputData[outputIdx]);
                    }
                }
            }
        });

        // Merge thread-local results
        for (int t = 0; t < numThreads; t++)
        {
            for (int i = 0; i < gradInputData.Length; i++)
            {
                gradInputData[i] = numOps.Add(gradInputData[i], localGradInputs[t][i]);
            }
        }

        return new Tensor<T>(inputShape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public Tensor<T> ConvTranspose3D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding, int[] outputPadding)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (input.Rank != 5) throw new ArgumentException($"ConvTranspose3D input requires 5D tensor [batch, in_channels, depth, height, width]. Got rank {input.Rank}.", nameof(input));
        if (kernel.Rank != 5) throw new ArgumentException($"ConvTranspose3D kernel requires 5D tensor [in_channels, out_channels, kD, kH, kW]. Got rank {kernel.Rank}.", nameof(kernel));
        if (stride == null || stride.Length != 3) throw new ArgumentException("Stride must be array of 3 elements.", nameof(stride));
        if (padding == null || padding.Length != 3) throw new ArgumentException("Padding must be array of 3 elements.", nameof(padding));
        if (outputPadding == null || outputPadding.Length != 3) throw new ArgumentException("Output padding must be array of 3 elements.", nameof(outputPadding));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int inDepth = input.Shape[2];
        int inHeight = input.Shape[3];
        int inWidth = input.Shape[4];

        int kernelInChannels = kernel.Shape[0];
        int outChannels = kernel.Shape[1];
        int kD = kernel.Shape[2];
        int kH = kernel.Shape[3];
        int kW = kernel.Shape[4];

        if (inChannels != kernelInChannels)
            throw new ArgumentException($"Kernel's input channels ({kernelInChannels}) must match input tensor's channels ({inChannels}).");

        int strideD = stride[0], strideH = stride[1], strideW = stride[2];
        int padD = padding[0], padH = padding[1], padW = padding[2];
        int outPadD = outputPadding[0], outPadH = outputPadding[1], outPadW = outputPadding[2];

        // Calculate output dimensions for transposed convolution
        int outDepth = (inDepth - 1) * strideD - 2 * padD + kD + outPadD;
        int outHeight = (inHeight - 1) * strideH - 2 * padH + kH + outPadH;
        int outWidth = (inWidth - 1) * strideW - 2 * padW + kW + outPadW;

        var outputData = new T[batch * outChannels * outDepth * outHeight * outWidth];
        var inputData = input.ToArray();
        var kernelData = kernel.ToArray();

        // Use thread-local accumulators
        int numThreads = Environment.ProcessorCount;
        var localOutputs = new T[numThreads][];
        for (int t = 0; t < numThreads; t++)
        {
            localOutputs[t] = new T[outputData.Length];
        }

        Parallel.For(0, batch * inChannels, bic =>
        {
            int threadId = Environment.CurrentManagedThreadId % numThreads;
            var localOutput = localOutputs[threadId];
            int b = bic / inChannels;
            int ic = bic % inChannels;

            for (int id = 0; id < inDepth; id++)
            {
                for (int ih = 0; ih < inHeight; ih++)
                {
                    for (int iw = 0; iw < inWidth; iw++)
                    {
                        int inputIdx = (((b * inChannels + ic) * inDepth + id) * inHeight + ih) * inWidth + iw;
                        T inputVal = inputData[inputIdx];

                        for (int oc = 0; oc < outChannels; oc++)
                        {
                            for (int kd = 0; kd < kD; kd++)
                            {
                                int od = id * strideD - padD + kd;
                                if (od < 0 || od >= outDepth) continue;

                                for (int kh = 0; kh < kH; kh++)
                                {
                                    int oh = ih * strideH - padH + kh;
                                    if (oh < 0 || oh >= outHeight) continue;

                                    for (int kw = 0; kw < kW; kw++)
                                    {
                                        int ow = iw * strideW - padW + kw;
                                        if (ow < 0 || ow >= outWidth) continue;

                                        int kernelIdx = (((ic * outChannels + oc) * kD + kd) * kH + kh) * kW + kw;
                                        int outputIdx = (((b * outChannels + oc) * outDepth + od) * outHeight + oh) * outWidth + ow;

                                        localOutput[outputIdx] = numOps.Add(localOutput[outputIdx],
                                            numOps.Multiply(inputVal, kernelData[kernelIdx]));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

        // Merge thread-local results
        for (int t = 0; t < numThreads; t++)
        {
            for (int i = 0; i < outputData.Length; i++)
            {
                outputData[i] = numOps.Add(outputData[i], localOutputs[t][i]);
            }
        }

        return new Tensor<T>([batch, outChannels, outDepth, outHeight, outWidth], new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public Tensor<T> ConvTranspose3DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (inputShape == null || inputShape.Length != 5) throw new ArgumentException("Input shape must be array of 5 elements.", nameof(inputShape));

        // The backward pass for transposed convolution input is equivalent to a regular Conv3D
        // with the kernel applied in the normal direction
        return Conv3D(gradOutput, kernel, stride, padding, [1, 1, 1]);
    }

    /// <inheritdoc/>
    public Tensor<T> ConvTranspose3DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernelShape == null || kernelShape.Length != 5) throw new ArgumentException("Kernel shape must be array of 5 elements.", nameof(kernelShape));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int inDepth = input.Shape[2];
        int inHeight = input.Shape[3];
        int inWidth = input.Shape[4];

        int outChannels = gradOutput.Shape[1];
        int outDepth = gradOutput.Shape[2];
        int outHeight = gradOutput.Shape[3];
        int outWidth = gradOutput.Shape[4];

        int kD = kernelShape[2], kH = kernelShape[3], kW = kernelShape[4];
        int strideD = stride[0], strideH = stride[1], strideW = stride[2];
        int padD = padding[0], padH = padding[1], padW = padding[2];

        var gradKernelData = new T[inChannels * outChannels * kD * kH * kW];
        var inputData = input.ToArray();
        var gradOutputData = gradOutput.ToArray();

        // Use thread-local accumulators
        int numThreads = Environment.ProcessorCount;
        var localGradKernels = new T[numThreads][];
        for (int t = 0; t < numThreads; t++)
        {
            localGradKernels[t] = new T[gradKernelData.Length];
        }

        Parallel.For(0, batch * inChannels, bic =>
        {
            int threadId = Environment.CurrentManagedThreadId % numThreads;
            var localGradKernel = localGradKernels[threadId];
            int b = bic / inChannels;
            int ic = bic % inChannels;

            for (int id = 0; id < inDepth; id++)
            {
                for (int ih = 0; ih < inHeight; ih++)
                {
                    for (int iw = 0; iw < inWidth; iw++)
                    {
                        int inputIdx = (((b * inChannels + ic) * inDepth + id) * inHeight + ih) * inWidth + iw;
                        T inputVal = inputData[inputIdx];

                        for (int oc = 0; oc < outChannels; oc++)
                        {
                            for (int kd = 0; kd < kD; kd++)
                            {
                                int od = id * strideD - padD + kd;
                                if (od < 0 || od >= outDepth) continue;

                                for (int kh = 0; kh < kH; kh++)
                                {
                                    int oh = ih * strideH - padH + kh;
                                    if (oh < 0 || oh >= outHeight) continue;

                                    for (int kw = 0; kw < kW; kw++)
                                    {
                                        int ow = iw * strideW - padW + kw;
                                        if (ow < 0 || ow >= outWidth) continue;

                                        int gradOutputIdx = (((b * outChannels + oc) * outDepth + od) * outHeight + oh) * outWidth + ow;
                                        int kernelIdx = (((ic * outChannels + oc) * kD + kd) * kH + kh) * kW + kw;

                                        localGradKernel[kernelIdx] = numOps.Add(localGradKernel[kernelIdx],
                                            numOps.Multiply(inputVal, gradOutputData[gradOutputIdx]));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

        // Merge thread-local results
        for (int t = 0; t < numThreads; t++)
        {
            for (int i = 0; i < gradKernelData.Length; i++)
            {
                gradKernelData[i] = numOps.Add(gradKernelData[i], localGradKernels[t][i]);
            }
        }

        return new Tensor<T>(kernelShape, new Vector<T>(gradKernelData));
    }

    #endregion

    /// <inheritdoc/>
    public Tensor<T> LocallyConnectedConv2D<T>(Tensor<T> input, Tensor<T> weights, Tensor<T>? bias, int[] stride)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (weights == null) throw new ArgumentNullException(nameof(weights));
        if (input.Rank != 4) throw new ArgumentException($"LocallyConnectedConv2D input requires a 4D tensor [batch, in_channels, height, width]. Got rank {input.Rank}.");
        // weights shape: [output_height, output_width, out_channels, in_channels, kernel_height, kernel_width]
        if (weights.Rank != 6) throw new ArgumentException($"LocallyConnectedConv2D weights require a 6D tensor. Got rank {weights.Rank}.");
        if (stride == null || stride.Length != 2) throw new ArgumentException("Stride must be an array of 2 elements", nameof(stride));
        if (stride[0] <= 0 || stride[1] <= 0) throw new ArgumentException("Stride elements must be positive", nameof(stride));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int inputHeight = input.Shape[2];
        int inputWidth = input.Shape[3];

        int outputHeight = weights.Shape[0];
        int outputWidth = weights.Shape[1];
        int outChannels = weights.Shape[2];
        int kernelInChannels = weights.Shape[3]; // in_channels in the weights tensor definition
        int kernelHeight = weights.Shape[4];
        int kernelWidth = weights.Shape[5];

        int strideH = stride[0], strideW = stride[1];

        // Validate kernel in_channels matches input in_channels
        if (kernelInChannels != inChannels)
            throw new ArgumentException($"Weight's input channels ({kernelInChannels}) must match input tensor's in_channels ({inChannels}).");

        // Ensure output shape derived from input/kernel/stride matches weights
        int expectedOutputH = (inputHeight - kernelHeight) / strideH + 1;
        int expectedOutputW = (inputWidth - kernelWidth) / strideW + 1;

        if (outputHeight != expectedOutputH || outputWidth != expectedOutputW)
            throw new ArgumentException($"Calculated output dimensions ({expectedOutputH}x{expectedOutputW}) do not match weights dimensions ({outputHeight}x{outputWidth}). Check input, kernel, and stride parameters.");

        var result = new Tensor<T>(new[] { batch, outChannels, outputHeight, outputWidth });
        var inputData = input.ToArray();
        var weightsData = weights.ToArray();
        var biasData = bias?.ToArray();

        Parallel.For(0, batch, b => // Process each batch independently
        {
            for (int oc = 0; oc < outChannels; oc++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        T sum = numOps.Zero;

                        for (int ic = 0; ic < inChannels; ic++)
                        {
                            for (int kh = 0; kh < kernelHeight; kh++)
                            {
                                for (int kw = 0; kw < kernelWidth; kw++)
                                {
                                    int ih = oh * strideH + kh;
                                    int iw = ow * strideW + kw;

                                    // Check bounds (LocallyConnected typically doesn't use padding in direct impl, but can be added)
                                    if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth)
                                    {
                                        T inputVal = inputData[((b * inChannels + ic) * inputHeight + ih) * inputWidth + iw];
                                        // weightsData index: [oh, ow, oc, ic, kh, kw]
                                        T weightVal = weightsData[((((oh * outputWidth + ow) * outChannels + oc) * kernelInChannels + ic) * kernelHeight + kh) * kernelWidth + kw];
                                        sum = numOps.Add(sum, numOps.Multiply(inputVal, weightVal));
                                    }
                                }
                            }
                        }

                        // Add bias if provided
                        if (biasData != null)
                        {
                            sum = numOps.Add(sum, biasData[oc]);
                        }

                        result[b, oc, oh, ow] = sum;
                    }
                }
            }
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> LocallyConnectedConv2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> weights, int[] inputShape, int[] stride)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (weights == null) throw new ArgumentNullException(nameof(weights));
        if (inputShape == null || inputShape.Length != 4) throw new ArgumentException("inputShape must be array of 4 elements [batch, inChannels, height, width]", nameof(inputShape));
        if (gradOutput.Rank != 4) throw new ArgumentException($"LocallyConnectedConv2DBackwardInput gradOutput requires 4D tensor. Got rank {gradOutput.Rank}.");
        if (weights.Rank != 6) throw new ArgumentException($"LocallyConnectedConv2DBackwardInput weights require a 6D tensor. Got rank {weights.Rank}.");
        if (stride == null || stride.Length != 2) throw new ArgumentException("Stride must be an array of 2 elements", nameof(stride));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int inputHeight = inputShape[2];
        int inputWidth = inputShape[3];

        int outputHeight = weights.Shape[0];
        int outputWidth = weights.Shape[1];
        int outChannels = weights.Shape[2];
        int kernelInChannels = weights.Shape[3];
        int kernelHeight = weights.Shape[4];
        int kernelWidth = weights.Shape[5];

        int strideH = stride[0], strideW = stride[1];

        var finalGradInput = new T[batch * inChannels * inputHeight * inputWidth];
        var gradOutputData = gradOutput.ToArray();
        var weightsData = weights.ToArray();

        Parallel.For(0, batch * inChannels * inputHeight * inputWidth, idx =>
        {
            int b = idx / (inChannels * inputHeight * inputWidth);
            int ic = (idx / (inputHeight * inputWidth)) % inChannels;
            int ih = (idx / inputWidth) % inputHeight;
            int iw = idx % inputWidth;

            T sumGrad = numOps.Zero;
            // Iterate over all possible output positions this input pixel contributes to
            for (int oh_candidate = 0; oh_candidate < outputHeight; oh_candidate++)
            {
                for (int ow_candidate = 0; ow_candidate < outputWidth; ow_candidate++)
                {
                    // Check if input pixel (ih, iw) is covered by kernel at (oh_candidate, ow_candidate)
                    // (ih, iw) should be in the kernel window
                    int kh_relative = ih - oh_candidate * strideH;
                    int kw_relative = iw - ow_candidate * strideW;

                    if (kh_relative >= 0 && kh_relative < kernelHeight && kw_relative >= 0 && kw_relative < kernelWidth)
                    {
                        for (int oc = 0; oc < outChannels; oc++)
                        {
                            T gradOutVal = gradOutputData[((b * outChannels + oc) * outputHeight + oh_candidate) * outputWidth + ow_candidate];
                            T weightVal = weightsData[((((oh_candidate * outputWidth + ow_candidate) * outChannels + oc) * kernelInChannels + ic) * kernelHeight + kh_relative) * kernelWidth + kw_relative];
                            sumGrad = numOps.Add(sumGrad, numOps.Multiply(gradOutVal, weightVal));
                        }
                    }
                }
            }
            finalGradInput[idx] = sumGrad;
        });

        return new Tensor<T>(inputShape, new Vector<T>(finalGradInput));
    }

    /// <inheritdoc/>
    public Tensor<T> LocallyConnectedConv2DBackwardWeights<T>(Tensor<T> gradOutput, Tensor<T> input, int[] weightsShape, int[] stride)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (weightsShape == null || weightsShape.Length != 6) throw new ArgumentException("weightsShape must be array of 6 elements", nameof(weightsShape));
        if (gradOutput.Rank != 4) throw new ArgumentException($"LocallyConnectedConv2DBackwardWeights gradOutput requires 4D tensor. Got rank {gradOutput.Rank}.");
        if (input.Rank != 4) throw new ArgumentException($"LocallyConnectedConv2DBackwardWeights input requires 4D tensor. Got rank {input.Rank}.");
        if (stride == null || stride.Length != 2) throw new ArgumentException("Stride must be array of 2 elements", nameof(stride));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int inputHeight = input.Shape[2];
        int inputWidth = input.Shape[3];

        int outputHeight = weightsShape[0];
        int outputWidth = weightsShape[1];
        int outChannels = weightsShape[2];
        int kernelInChannels = weightsShape[3];
        int kernelHeight = weightsShape[4];
        int kernelWidth = weightsShape[5];

        int strideH = stride[0], strideW = stride[1];

        var gradWeights = new T[weightsShape.Aggregate(1, (acc, val) => acc * val)];
        var gradOutputData = gradOutput.ToArray();
        var inputData = input.ToArray();

        for (int i = 0; i < gradWeights.Length; i++)
            gradWeights[i] = numOps.Zero;

        Parallel.For(0, weightsShape.Aggregate(1, (acc, val) => acc * val), idx => // Iterate over all weight elements
        {
            // Deconstruct idx to weights indices (oh, ow, oc, ic, kh, kw)
            int flatIdx = idx;
            int kw_w = kernelWidth;
            int kh_w = kernelHeight;
            int ic_w = kernelInChannels; // This is the 3rd dim in weights (index 3)
            int oc_w = outChannels;
            int ow_w = outputWidth;

            int kw = flatIdx % kw_w; flatIdx /= kw_w;
            int kh = flatIdx % kh_w; flatIdx /= kh_w;
            int ic = flatIdx % ic_w; flatIdx /= ic_w;
            int oc = flatIdx % oc_w; flatIdx /= oc_w;
            int ow = flatIdx % ow_w; flatIdx /= ow_w;
            int oh = flatIdx;

            T sum = numOps.Zero;
            for (int b = 0; b < batch; b++)
            {
                int ih = oh * strideH + kh;
                int iw = ow * strideW + kw;

                // Check bounds for input
                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth)
                {
                    T gradOutVal = gradOutputData[((b * outChannels + oc) * outputHeight + oh) * outputWidth + ow];
                    T inputVal = inputData[((b * inChannels + ic) * inputHeight + ih) * inputWidth + iw];
                    sum = numOps.Add(sum, numOps.Multiply(gradOutVal, inputVal));
                }
            }
            gradWeights[idx] = sum;
        });

        return new Tensor<T>(weightsShape, new Vector<T>(gradWeights));
    }

    /// <inheritdoc/>
    public Tensor<T> LocallyConnectedConv2DBackwardBias<T>(Tensor<T> gradOutput)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (gradOutput.Rank != 4) throw new ArgumentException($"LocallyConnectedConv2DBackwardBias gradOutput requires 4D tensor. Got rank {gradOutput.Rank}.");

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = gradOutput.Shape[0];
        int outChannels = gradOutput.Shape[1];
        int outputHeight = gradOutput.Shape[2];
        int outputWidth = gradOutput.Shape[3];

        var gradBias = new T[outChannels]; // Bias gradient is 1D [outChannels]
        var gradOutputData = gradOutput.ToArray();

        for (int i = 0; i < gradBias.Length; i++)
            gradBias[i] = numOps.Zero;

        Parallel.For(0, outChannels, oc => // Iterate over output channels
        {
            T sum = numOps.Zero;
            for (int b = 0; b < batch; b++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        sum = numOps.Add(sum, gradOutputData[((b * outChannels + oc) * outputHeight + oh) * outputWidth + ow]);
                    }
                }
            }
            gradBias[oc] = sum;
        });

        return new Tensor<T>(new[] { outChannels }, new Vector<T>(gradBias));
    }

    #endregion

    #region Normalization and Activation Operations

    /// <inheritdoc/>
    public Tensor<T> Softmax<T>(Tensor<T> input, int axis = -1)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();
        int rank = input.Rank;
        if (axis < 0) axis = rank + axis;
        if (axis < 0 || axis >= rank)
            throw new ArgumentException($"Invalid axis {axis} for tensor with {rank} dimensions");

        var inputData = input.ToArray();
        var outputData = new T[inputData.Length];

        // Compute outer and inner sizes
        int outerSize = 1, axisSize = input.Shape[axis], innerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= input.Shape[i];
        for (int i = axis + 1; i < rank; i++) innerSize *= input.Shape[i];

        Parallel.For(0, outerSize * innerSize, idx =>
        {
            int outer = idx / innerSize;
            int inner = idx % innerSize;

            // Find max for numerical stability
            T maxVal = numOps.MinValue;
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                if (numOps.GreaterThan(inputData[flatIdx], maxVal))
                    maxVal = inputData[flatIdx];
            }

            // Compute exp and sum
            T sumExp = numOps.Zero;
            var expVals = new T[axisSize];
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                expVals[i] = numOps.Exp(numOps.Subtract(inputData[flatIdx], maxVal));
                sumExp = numOps.Add(sumExp, expVals[i]);
            }

            // Normalize
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                outputData[flatIdx] = numOps.Divide(expVals[i], sumExp);
            }
        });

        return new Tensor<T>(input.Shape, new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public Tensor<T> SoftmaxBackward<T>(Tensor<T> gradOutput, Tensor<T> output, int axis = -1)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (output == null) throw new ArgumentNullException(nameof(output));

        var numOps = MathHelper.GetNumericOperations<T>();
        int rank = output.Rank;
        if (axis < 0) axis = rank + axis;

        var gradOutputData = gradOutput.ToArray();
        var outputData = output.ToArray();
        var gradInputData = new T[outputData.Length];

        int outerSize = 1, axisSize = output.Shape[axis], innerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= output.Shape[i];
        for (int i = axis + 1; i < rank; i++) innerSize *= output.Shape[i];

        Parallel.For(0, outerSize * innerSize, idx =>
        {
            int outer = idx / innerSize;
            int inner = idx % innerSize;

            // Compute dot product of grad and output along axis
            T dotProduct = numOps.Zero;
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                dotProduct = numOps.Add(dotProduct, numOps.Multiply(gradOutputData[flatIdx], outputData[flatIdx]));
            }

            // Compute gradient: grad_input = output * (grad_output - dot_product)
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                gradInputData[flatIdx] = numOps.Multiply(outputData[flatIdx], numOps.Subtract(gradOutputData[flatIdx], dotProduct));
            }
        });

        return new Tensor<T>(output.Shape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public Tensor<T> GumbelSoftmax<T>(Tensor<T> input, double temperature = 1.0, bool hard = false, int axis = -1)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (temperature <= 0)
            throw new ArgumentOutOfRangeException(nameof(temperature), temperature, "Temperature must be positive.");
        if (double.IsNaN(temperature) || double.IsInfinity(temperature))
            throw new ArgumentOutOfRangeException(nameof(temperature), temperature, "Temperature must be a finite number.");

        var numOps = MathHelper.GetNumericOperations<T>();
        int rank = input.Rank;
        if (axis < 0) axis = rank + axis;

        var inputData = input.ToArray();
        var shape = input.Shape;
        const double eps = 1e-10;

        // Add Gumbel noise: -log(-log(U)) where U ~ Uniform(0, 1)
        var random = RandomHelper.ThreadSafeRandom;
        var perturbedData = new T[inputData.Length];
        for (int i = 0; i < inputData.Length; i++)
        {
            var u = random.NextDouble();
            u = Math.Max(u, eps);
            u = Math.Min(u, 1 - eps);
            var gumbel = numOps.FromDouble(-Math.Log(-Math.Log(u)));
            var val = numOps.Add(inputData[i], gumbel);
            perturbedData[i] = numOps.Divide(val, numOps.FromDouble(temperature));
        }

        // Apply softmax
        var perturbedTensor = new Tensor<T>(shape, new Vector<T>(perturbedData));
        var softResult = Softmax(perturbedTensor, axis);

        if (!hard)
            return softResult;

        // Hard mode: create one-hot and use straight-through estimator
        var softData = softResult.ToArray();
        var hardData = new T[softData.Length];
        int outerSize = 1, axisSize = shape[axis], innerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= shape[i];
        for (int i = axis + 1; i < rank; i++) innerSize *= shape[i];

        Parallel.For(0, outerSize * innerSize, idx =>
        {
            int outer = idx / innerSize;
            int inner = idx % innerSize;

            // Find argmax
            int maxIdx = 0;
            T maxVal = softData[(outer * axisSize) * innerSize + inner];
            for (int i = 1; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                if (numOps.GreaterThan(softData[flatIdx], maxVal))
                {
                    maxVal = softData[flatIdx];
                    maxIdx = i;
                }
            }

            // Create one-hot
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                hardData[flatIdx] = i == maxIdx ? numOps.One : numOps.Zero;
            }
        });

        return new Tensor<T>(shape, new Vector<T>(hardData));
    }

    /// <inheritdoc/>
    public Tensor<T> GumbelSoftmaxBackward<T>(Tensor<T> gradOutput, Tensor<T> output, double temperature, int axis = -1)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (output == null) throw new ArgumentNullException(nameof(output));
        if (temperature <= 0)
            throw new ArgumentOutOfRangeException(nameof(temperature), temperature, "Temperature must be positive.");

        // Gradient flows through softmax, scaled by 1/temperature
        var softmaxGrad = SoftmaxBackward(gradOutput, output, axis);
        var numOps = MathHelper.GetNumericOperations<T>();
        var gradData = softmaxGrad.ToArray();
        var scale = numOps.FromDouble(1.0 / temperature);

        for (int i = 0; i < gradData.Length; i++)
        {
            gradData[i] = numOps.Multiply(gradData[i], scale);
        }

        return new Tensor<T>(output.Shape, new Vector<T>(gradData));
    }

    /// <inheritdoc/>
    public Tensor<T> TaylorSoftmax<T>(Tensor<T> input, int order = 2, int axis = -1)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (order < 1)
            throw new ArgumentOutOfRangeException(nameof(order), order, "Order must be at least 1.");

        var numOps = MathHelper.GetNumericOperations<T>();
        int rank = input.Rank;
        if (axis < 0) axis = rank + axis;

        var inputData = input.ToArray();
        var shape = input.Shape;
        var outputData = new T[inputData.Length];

        // Precompute factorials
        var factorials = new double[order + 1];
        factorials[0] = 1;
        for (int i = 1; i <= order; i++)
            factorials[i] = factorials[i - 1] * i;

        int outerSize = 1, axisSize = shape[axis], innerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= shape[i];
        for (int i = axis + 1; i < rank; i++) innerSize *= shape[i];

        Parallel.For(0, outerSize * innerSize, idx =>
        {
            int outer = idx / innerSize;
            int inner = idx % innerSize;

            // Find max for numerical stability (similar to standard softmax)
            var maxVal = inputData[(outer * axisSize) * innerSize + inner];
            for (int i = 1; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                if (numOps.GreaterThan(inputData[flatIdx], maxVal))
                    maxVal = inputData[flatIdx];
            }

            // Compute Taylor approximation of exp for each position along axis
            var expApprox = new T[axisSize];
            T sumExp = numOps.Zero;

            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                // Subtract max for numerical stability
                var x = numOps.Subtract(inputData[flatIdx], maxVal);

                // Taylor: 1 + x + x^2/2! + x^3/3! + ...
                var taylorExp = numOps.One;
                var xPower = numOps.One;
                for (int n = 1; n <= order; n++)
                {
                    xPower = numOps.Multiply(xPower, x);
                    taylorExp = numOps.Add(taylorExp, numOps.Divide(xPower, numOps.FromDouble(factorials[n])));
                }

                // Ensure non-negative for numerical stability
                if (numOps.LessThan(taylorExp, numOps.Zero))
                    taylorExp = numOps.FromDouble(1e-10);

                expApprox[i] = taylorExp;
                sumExp = numOps.Add(sumExp, taylorExp);
            }

            // Guard against zero sum (shouldn't happen with proper max subtraction, but just in case)
            if (numOps.Equals(sumExp, numOps.Zero))
                sumExp = numOps.FromDouble(1e-10);

            // Normalize
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                outputData[flatIdx] = numOps.Divide(expApprox[i], sumExp);
            }
        });

        return new Tensor<T>(shape, new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public Tensor<T> TaylorSoftmaxBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> output, int order, int axis = -1)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (output == null) throw new ArgumentNullException(nameof(output));

        var numOps = MathHelper.GetNumericOperations<T>();
        int rank = output.Rank;
        if (axis < 0) axis = rank + axis;

        var gradOutputData = gradOutput.ToArray();
        var inputData = input.ToArray();
        var outputData = output.ToArray();
        var shape = output.Shape;
        var gradInputData = new T[outputData.Length];

        // Precompute factorials for derivative
        var factorials = new double[order + 1];
        factorials[0] = 1;
        for (int i = 1; i <= order; i++)
            factorials[i] = factorials[i - 1] * i;

        int outerSize = 1, axisSize = shape[axis], innerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= shape[i];
        for (int i = axis + 1; i < rank; i++) innerSize *= shape[i];

        Parallel.For(0, outerSize * innerSize, idx =>
        {
            int outer = idx / innerSize;
            int inner = idx % innerSize;

            // Compute g(x) and g'(x) for each position
            var gValues = new T[axisSize];
            var gPrimeValues = new T[axisSize];
            T sumG = numOps.Zero;

            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                var x = inputData[flatIdx];

                // g(x) = Taylor approximation
                var g = numOps.One;
                var xPower = numOps.One;
                for (int n = 1; n <= order; n++)
                {
                    xPower = numOps.Multiply(xPower, x);
                    g = numOps.Add(g, numOps.Divide(xPower, numOps.FromDouble(factorials[n])));
                }

                // g'(x) = derivative of Taylor = 1 + x + x^2/2! + ... (shifted)
                var gPrime = numOps.One;
                xPower = numOps.One;
                for (int n = 1; n < order; n++)
                {
                    xPower = numOps.Multiply(xPower, x);
                    gPrime = numOps.Add(gPrime, numOps.Divide(xPower, numOps.FromDouble(factorials[n])));
                }

                gValues[i] = g;
                gPrimeValues[i] = gPrime;
                sumG = numOps.Add(sumG, g);
            }

            // Compute gradient using chain rule: grad = softmaxGrad * g'(x) / g(x)
            T dotProduct = numOps.Zero;
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                dotProduct = numOps.Add(dotProduct, numOps.Multiply(gradOutputData[flatIdx], outputData[flatIdx]));
            }

            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                var softmaxGrad = numOps.Multiply(outputData[flatIdx], numOps.Subtract(gradOutputData[flatIdx], dotProduct));
                var gPrimeOverG = numOps.Divide(gPrimeValues[i], gValues[i]);
                gradInputData[flatIdx] = numOps.Multiply(softmaxGrad, gPrimeOverG);
            }
        });

        return new Tensor<T>(shape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public Tensor<T> Sparsemax<T>(Tensor<T> input, int axis = -1)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();
        int rank = input.Rank;
        if (axis < 0) axis = rank + axis;

        var inputData = input.ToArray();
        var shape = input.Shape;
        var outputData = new T[inputData.Length];

        int outerSize = 1, axisSize = shape[axis], innerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= shape[i];
        for (int i = axis + 1; i < rank; i++) innerSize *= shape[i];

        Parallel.For(0, outerSize * innerSize, idx =>
        {
            int outer = idx / innerSize;
            int inner = idx % innerSize;

            // Extract values along axis and sort by value (descending)
            var indexed = new List<(T value, int idx)>();
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                indexed.Add((inputData[flatIdx], i));
            }

            indexed.Sort((a, b) =>
            {
                if (numOps.GreaterThan(a.value, b.value)) return -1;
                if (numOps.LessThan(a.value, b.value)) return 1;
                return 0;
            });

            // Find threshold tau using the sparsemax algorithm
            T cumSum = numOps.Zero;
            int k = 0;
            T threshold = numOps.Zero;

            for (int i = 0; i < axisSize; i++)
            {
                cumSum = numOps.Add(cumSum, indexed[i].value);
                // Check if z[i] > (cumSum - 1) / (i + 1)
                var kPlusOne = numOps.FromDouble(i + 1);
                var testThreshold = numOps.Divide(numOps.Subtract(cumSum, numOps.One), kPlusOne);
                if (numOps.GreaterThan(indexed[i].value, testThreshold))
                {
                    k = i + 1;
                    threshold = testThreshold;
                }
            }

            // Compute output: max(0, z - tau)
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                var val = numOps.Subtract(inputData[flatIdx], threshold);
                outputData[flatIdx] = numOps.GreaterThan(val, numOps.Zero) ? val : numOps.Zero;
            }
        });

        return new Tensor<T>(shape, new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public Tensor<T> SparsemaxBackward<T>(Tensor<T> gradOutput, Tensor<T> output, int axis = -1)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (output == null) throw new ArgumentNullException(nameof(output));

        var numOps = MathHelper.GetNumericOperations<T>();
        int rank = output.Rank;
        if (axis < 0) axis = rank + axis;

        var gradOutputData = gradOutput.ToArray();
        var outputData = output.ToArray();
        var shape = output.Shape;
        var gradInputData = new T[outputData.Length];

        int outerSize = 1, axisSize = shape[axis], innerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= shape[i];
        for (int i = axis + 1; i < rank; i++) innerSize *= shape[i];

        Parallel.For(0, outerSize * innerSize, idx =>
        {
            int outer = idx / innerSize;
            int inner = idx % innerSize;

            // Find support set (non-zero outputs) and compute mean of gradients in support
            T sumGradSupport = numOps.Zero;
            int supportSize = 0;

            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                if (numOps.GreaterThan(outputData[flatIdx], numOps.Zero))
                {
                    sumGradSupport = numOps.Add(sumGradSupport, gradOutputData[flatIdx]);
                    supportSize++;
                }
            }

            T meanGradSupport = supportSize > 0
                ? numOps.Divide(sumGradSupport, numOps.FromDouble(supportSize))
                : numOps.Zero;

            // Gradient: grad_input = grad_output - mean(grad_output[support]) for support, 0 otherwise
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                gradInputData[flatIdx] = numOps.GreaterThan(outputData[flatIdx], numOps.Zero)
                    ? numOps.Subtract(gradOutputData[flatIdx], meanGradSupport)
                    : numOps.Zero;
            }
        });

        return new Tensor<T>(shape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public Tensor<T> SphericalSoftmax<T>(Tensor<T> input, int axis = -1)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();
        int rank = input.Rank;
        if (axis < 0) axis = rank + axis;

        var inputData = input.ToArray();
        var shape = input.Shape;
        var normalizedData = new T[inputData.Length];

        int outerSize = 1, axisSize = shape[axis], innerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= shape[i];
        for (int i = axis + 1; i < rank; i++) innerSize *= shape[i];

        Parallel.For(0, outerSize * innerSize, idx =>
        {
            int outer = idx / innerSize;
            int inner = idx % innerSize;

            // Compute L2 norm along axis
            T sumSquares = numOps.Zero;
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                var val = inputData[flatIdx];
                sumSquares = numOps.Add(sumSquares, numOps.Multiply(val, val));
            }
            var norm = numOps.Sqrt(sumSquares);

            // Avoid division by zero
            if (numOps.Equals(norm, numOps.Zero))
                norm = numOps.FromDouble(1e-10);

            // Normalize by L2 norm
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                normalizedData[flatIdx] = numOps.Divide(inputData[flatIdx], norm);
            }
        });

        // Apply softmax to normalized data
        var normalizedTensor = new Tensor<T>(shape, new Vector<T>(normalizedData));
        return Softmax(normalizedTensor, axis);
    }

    /// <inheritdoc/>
    public Tensor<T> SphericalSoftmaxBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> output, int axis = -1)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (output == null) throw new ArgumentNullException(nameof(output));

        var numOps = MathHelper.GetNumericOperations<T>();
        int rank = input.Rank;
        if (axis < 0) axis = rank + axis;

        var inputData = input.ToArray();
        var shape = input.Shape;

        int outerSize = 1, axisSize = shape[axis], innerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= shape[i];
        for (int i = axis + 1; i < rank; i++) innerSize *= shape[i];

        // First compute the normalized input
        var normalizedData = new T[inputData.Length];
        var norms = new T[outerSize * innerSize];

        Parallel.For(0, outerSize * innerSize, idx =>
        {
            int outer = idx / innerSize;
            int inner = idx % innerSize;

            // Compute L2 norm
            T sumSquares = numOps.Zero;
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                var val = inputData[flatIdx];
                sumSquares = numOps.Add(sumSquares, numOps.Multiply(val, val));
            }
            var norm = numOps.Sqrt(sumSquares);
            if (numOps.Equals(norm, numOps.Zero))
                norm = numOps.FromDouble(1e-10);
            norms[idx] = norm;

            // Normalize
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                normalizedData[flatIdx] = numOps.Divide(inputData[flatIdx], norm);
            }
        });

        // Get softmax gradient with respect to normalized input
        var normalizedTensor = new Tensor<T>(shape, new Vector<T>(normalizedData));
        var softmaxGrad = SoftmaxBackward(gradOutput, output, axis);
        var softmaxGradData = softmaxGrad.ToArray();

        // Chain rule through L2 normalization
        var gradInputData = new T[inputData.Length];

        Parallel.For(0, outerSize * innerSize, idx =>
        {
            int outer = idx / innerSize;
            int inner = idx % innerSize;
            var norm = norms[idx];
            var normCubed = numOps.Multiply(norm, numOps.Multiply(norm, norm));

            // Compute dot product of x and grad_normalized
            T dotProduct = numOps.Zero;
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                dotProduct = numOps.Add(dotProduct, numOps.Multiply(inputData[flatIdx], softmaxGradData[flatIdx]));
            }

            // grad_x = (grad_normalized - normalized * dot_product) / norm
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                var term = numOps.Multiply(normalizedData[flatIdx], dotProduct);
                gradInputData[flatIdx] = numOps.Divide(numOps.Subtract(softmaxGradData[flatIdx], term), norm);
            }
        });

        return new Tensor<T>(shape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public Tensor<T> BatchNorm<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (gamma == null) throw new ArgumentNullException(nameof(gamma));
        if (beta == null) throw new ArgumentNullException(nameof(beta));

        var numOps = MathHelper.GetNumericOperations<T>();
        T eps = numOps.FromDouble(epsilon);

        int batch = input.Shape[0];
        int features = input.Shape[1];

        var inputData = input.ToArray();
        var gammaData = gamma.ToArray();
        var betaData = beta.ToArray();

        var meanData = new T[features];
        var varData = new T[features];
        var outputData = new T[batch * features];

        // Compute mean per feature
        for (int f = 0; f < features; f++)
        {
            T sum = numOps.Zero;
            for (int b = 0; b < batch; b++)
            {
                sum = numOps.Add(sum, inputData[b * features + f]);
            }
            meanData[f] = numOps.Divide(sum, numOps.FromDouble(batch));
        }

        // Compute variance per feature
        for (int f = 0; f < features; f++)
        {
            T sumSq = numOps.Zero;
            for (int b = 0; b < batch; b++)
            {
                T diff = numOps.Subtract(inputData[b * features + f], meanData[f]);
                sumSq = numOps.Add(sumSq, numOps.Multiply(diff, diff));
            }
            varData[f] = numOps.Divide(sumSq, numOps.FromDouble(batch));
        }

        // Normalize and scale
        Parallel.For(0, batch, b =>
        {
            for (int f = 0; f < features; f++)
            {
                T normalized = numOps.Divide(
                    numOps.Subtract(inputData[b * features + f], meanData[f]),
                    numOps.Sqrt(numOps.Add(varData[f], eps)));
                outputData[b * features + f] = numOps.Add(numOps.Multiply(gammaData[f], normalized), betaData[f]);
            }
        });

        mean = new Tensor<T>([features], new Vector<T>(meanData));
        variance = new Tensor<T>([features], new Vector<T>(varData));
        return new Tensor<T>(input.Shape, new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public Tensor<T> BatchNormBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> gamma, Tensor<T> mean, Tensor<T> variance, double epsilon, out Tensor<T> gradGamma, out Tensor<T> gradBeta)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();
        T eps = numOps.FromDouble(epsilon);

        int batch = input.Shape[0];
        int features = input.Shape[1];
        T batchT = numOps.FromDouble(batch);

        var gradOutputData = gradOutput.ToArray();
        var inputData = input.ToArray();
        var gammaData = gamma.ToArray();
        var meanData = mean.ToArray();
        var varData = variance.ToArray();

        var gradGammaData = new T[features];
        var gradBetaData = new T[features];
        var gradInputData = new T[batch * features];

        // Compute gradGamma and gradBeta
        for (int f = 0; f < features; f++)
        {
            T gGamma = numOps.Zero;
            T gBeta = numOps.Zero;
            T invStd = numOps.Divide(numOps.One, numOps.Sqrt(numOps.Add(varData[f], eps)));

            for (int b = 0; b < batch; b++)
            {
                int idx = b * features + f;
                T normalized = numOps.Multiply(numOps.Subtract(inputData[idx], meanData[f]), invStd);
                gGamma = numOps.Add(gGamma, numOps.Multiply(gradOutputData[idx], normalized));
                gBeta = numOps.Add(gBeta, gradOutputData[idx]);
            }

            gradGammaData[f] = gGamma;
            gradBetaData[f] = gBeta;
        }

        // Compute gradInput
        // Standard batch norm backward formula:
        // dx = (gamma / sqrt(var + eps) / N) * (N * dy - sum(dy) - (x - mean) / (var + eps) * sum(dy * (x - mean)))
        // All terms must be scaled by gamma for correctness
        Parallel.For(0, features, f =>
        {
            T invStd = numOps.Divide(numOps.One, numOps.Sqrt(numOps.Add(varData[f], eps)));
            T gamma = gammaData[f];
            T sumGrad = numOps.Zero;
            T sumGradX = numOps.Zero;

            // Accumulate sums over batch dimension
            for (int b = 0; b < batch; b++)
            {
                int idx = b * features + f;
                sumGrad = numOps.Add(sumGrad, gradOutputData[idx]);
                sumGradX = numOps.Add(sumGradX, numOps.Multiply(gradOutputData[idx], numOps.Subtract(inputData[idx], meanData[f])));
            }

            // Apply gamma scaling to accumulated sums
            T gammaSumGrad = numOps.Multiply(gamma, sumGrad);
            T gammaSumGradX = numOps.Multiply(gamma, sumGradX);

            for (int b = 0; b < batch; b++)
            {
                int idx = b * features + f;
                T normalized = numOps.Multiply(numOps.Subtract(inputData[idx], meanData[f]), invStd);
                T gradNorm = numOps.Multiply(gamma, gradOutputData[idx]);
                T term1 = numOps.Multiply(batchT, gradNorm);
                T term2 = gammaSumGrad;
                T term3 = numOps.Multiply(normalized, numOps.Multiply(invStd, gammaSumGradX));
                gradInputData[idx] = numOps.Multiply(numOps.Divide(invStd, batchT), numOps.Subtract(numOps.Subtract(term1, term2), term3));
            }
        });

        gradGamma = new Tensor<T>([features], new Vector<T>(gradGammaData));
        gradBeta = new Tensor<T>([features], new Vector<T>(gradBetaData));
        return new Tensor<T>(input.Shape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public Tensor<T> LayerNorm<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (gamma == null) throw new ArgumentNullException(nameof(gamma));
        if (beta == null) throw new ArgumentNullException(nameof(beta));

        var numOps = MathHelper.GetNumericOperations<T>();
        T eps = numOps.FromDouble(epsilon);

        int batch = input.Shape[0];
        int features = input.Shape[1];

        var inputData = input.ToArray();
        var gammaData = gamma.ToArray();
        var betaData = beta.ToArray();

        var meanData = new T[batch];
        var varData = new T[batch];
        var outputData = new T[batch * features];

        // Compute mean per sample
        for (int b = 0; b < batch; b++)
        {
            T sum = numOps.Zero;
            for (int f = 0; f < features; f++)
            {
                sum = numOps.Add(sum, inputData[b * features + f]);
            }
            meanData[b] = numOps.Divide(sum, numOps.FromDouble(features));
        }

        // Compute variance per sample
        for (int b = 0; b < batch; b++)
        {
            T sumSq = numOps.Zero;
            for (int f = 0; f < features; f++)
            {
                T diff = numOps.Subtract(inputData[b * features + f], meanData[b]);
                sumSq = numOps.Add(sumSq, numOps.Multiply(diff, diff));
            }
            varData[b] = numOps.Divide(sumSq, numOps.FromDouble(features));
        }

        // Normalize and scale
        Parallel.For(0, batch, b =>
        {
            T invStd = numOps.Divide(numOps.One, numOps.Sqrt(numOps.Add(varData[b], eps)));
            for (int f = 0; f < features; f++)
            {
                T normalized = numOps.Multiply(numOps.Subtract(inputData[b * features + f], meanData[b]), invStd);
                outputData[b * features + f] = numOps.Add(numOps.Multiply(gammaData[f], normalized), betaData[f]);
            }
        });

        mean = new Tensor<T>([batch], new Vector<T>(meanData));
        variance = new Tensor<T>([batch], new Vector<T>(varData));
        return new Tensor<T>(input.Shape, new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public Tensor<T> LayerNormBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> gamma, Tensor<T> mean, Tensor<T> variance, double epsilon, out Tensor<T> gradGamma, out Tensor<T> gradBeta)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();
        T eps = numOps.FromDouble(epsilon);

        int batch = input.Shape[0];
        int features = input.Shape[1];
        T featuresT = numOps.FromDouble(features);

        var gradOutputData = gradOutput.ToArray();
        var inputData = input.ToArray();
        var gammaData = gamma.ToArray();
        var meanData = mean.ToArray();
        var varData = variance.ToArray();

        var gradGammaData = new T[features];
        var gradBetaData = new T[features];
        var gradInputData = new T[batch * features];

        // Initialize gradGamma and gradBeta to zero
        for (int f = 0; f < features; f++)
        {
            gradGammaData[f] = numOps.Zero;
            gradBetaData[f] = numOps.Zero;
        }

        // Compute gradGamma and gradBeta
        for (int b = 0; b < batch; b++)
        {
            T invStd = numOps.Divide(numOps.One, numOps.Sqrt(numOps.Add(varData[b], eps)));
            for (int f = 0; f < features; f++)
            {
                int idx = b * features + f;
                T normalized = numOps.Multiply(numOps.Subtract(inputData[idx], meanData[b]), invStd);
                gradGammaData[f] = numOps.Add(gradGammaData[f], numOps.Multiply(gradOutputData[idx], normalized));
                gradBetaData[f] = numOps.Add(gradBetaData[f], gradOutputData[idx]);
            }
        }

        // Compute gradInput
        Parallel.For(0, batch, b =>
        {
            T invStd = numOps.Divide(numOps.One, numOps.Sqrt(numOps.Add(varData[b], eps)));
            T sumGrad = numOps.Zero;
            T sumGradX = numOps.Zero;

            for (int f = 0; f < features; f++)
            {
                int idx = b * features + f;
                T scaledGrad = numOps.Multiply(gammaData[f], gradOutputData[idx]);
                sumGrad = numOps.Add(sumGrad, scaledGrad);
                sumGradX = numOps.Add(sumGradX, numOps.Multiply(scaledGrad, numOps.Subtract(inputData[idx], meanData[b])));
            }

            for (int f = 0; f < features; f++)
            {
                int idx = b * features + f;
                T normalized = numOps.Multiply(numOps.Subtract(inputData[idx], meanData[b]), invStd);
                T gradNorm = numOps.Multiply(gammaData[f], gradOutputData[idx]);
                T term1 = numOps.Multiply(featuresT, gradNorm);
                T term2 = sumGrad;
                T term3 = numOps.Multiply(normalized, numOps.Multiply(invStd, sumGradX));
                gradInputData[idx] = numOps.Multiply(numOps.Divide(invStd, featuresT), numOps.Subtract(numOps.Subtract(term1, term2), term3));
            }
        });

        gradGamma = new Tensor<T>([features], new Vector<T>(gradGammaData));
        gradBeta = new Tensor<T>([features], new Vector<T>(gradBetaData));
        return new Tensor<T>(input.Shape, new Vector<T>(gradInputData));
    }

    #endregion

    #region Tensor Reduction Operations

    /// <summary>
    /// Validates and normalizes reduction axes.
    /// </summary>
    /// <param name="axes">The axes to validate</param>
    /// <param name="rank">The tensor rank</param>
    /// <returns>Normalized, validated, and sorted unique axes</returns>
    private static int[] ValidateAndNormalizeAxes(int[] axes, int rank)
    {
        if (axes == null)
            throw new ArgumentNullException(nameof(axes), "Axes cannot be null");

        if (axes.Length == 0)
            throw new ArgumentException("Axes array cannot be empty", nameof(axes));

        var normalizedAxes = new int[axes.Length];
        for (int i = 0; i < axes.Length; i++)
        {
            int axis = axes[i];
            // Normalize negative indices
            int normalized = axis < 0 ? rank + axis : axis;

            if (normalized < 0 || normalized >= rank)
                throw new ArgumentOutOfRangeException(nameof(axes), $"Axis {axis} is out of range for tensor with rank {rank}. Valid range is [{-rank}, {rank - 1}].");

            normalizedAxes[i] = normalized;
        }

        // Check for duplicates
        var uniqueAxes = normalizedAxes.Distinct().ToArray();
        if (uniqueAxes.Length != axes.Length)
            throw new ArgumentException("Duplicate axes are not allowed", nameof(axes));

        return uniqueAxes.OrderBy(a => a).ToArray();
    }

    /// <inheritdoc/>
    public Tensor<T> ReduceMax<T>(Tensor<T> input, int[] axes, bool keepDims, out int[] maxIndices)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputShape = input.Shape;
        var inputData = input.ToArray();

        // Validate and normalize axes
        var normalizedAxes = ValidateAndNormalizeAxes(axes, inputShape.Length);

        // Compute output shape
        var outputShapeList = new List<int>();
        for (int i = 0; i < inputShape.Length; i++)
        {
            if (normalizedAxes.Contains(i))
            {
                if (keepDims) outputShapeList.Add(1);
            }
            else
            {
                outputShapeList.Add(inputShape[i]);
            }
        }
        var outputShape = outputShapeList.Count > 0 ? outputShapeList.ToArray() : [1];

        int outputSize = outputShape.Aggregate(1, (a, b) => a * b);
        var outputData = new T[outputSize];
        maxIndices = new int[outputSize];

        // Initialize with minimum values
        T minVal = numOps.MinValue;
        for (int i = 0; i < outputSize; i++)
        {
            outputData[i] = minVal;
            maxIndices[i] = -1;
        }

        var inputStrides = ComputeStrides(inputShape);
        var outputStrides = ComputeStrides(outputShape);

        for (int i = 0; i < input.Length; i++)
        {
            var multiIndex = FlatToMultiIndex(i, inputShape, inputStrides);

            var outputMultiIndex = new List<int>();
            for (int d = 0; d < inputShape.Length; d++)
            {
                if (normalizedAxes.Contains(d))
                {
                    if (keepDims) outputMultiIndex.Add(0);
                }
                else
                {
                    outputMultiIndex.Add(multiIndex[d]);
                }
            }
            if (outputMultiIndex.Count == 0) outputMultiIndex.Add(0);

            int outputIdx = MultiToFlatIndex([.. outputMultiIndex], outputShape, outputStrides);

            if (numOps.GreaterThan(inputData[i], outputData[outputIdx]))
            {
                outputData[outputIdx] = inputData[i];
                maxIndices[outputIdx] = i;
            }
        }

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public Tensor<T> ReduceMaxBackward<T>(Tensor<T> gradOutput, int[] maxIndices, int[] inputShape)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int inputSize = inputShape.Aggregate(1, (a, b) => a * b);
        var gradInputData = new T[inputSize];

        for (int i = 0; i < inputSize; i++)
            gradInputData[i] = numOps.Zero;

        var gradOutputData = gradOutput.ToArray();

        for (int i = 0; i < maxIndices.Length; i++)
        {
            if (maxIndices[i] >= 0 && maxIndices[i] < inputSize)
            {
                gradInputData[maxIndices[i]] = numOps.Add(gradInputData[maxIndices[i]], gradOutputData[i]);
            }
        }

        return new Tensor<T>(inputShape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public Tensor<T> ReduceMean<T>(Tensor<T> input, int[] axes, bool keepDims)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputShape = input.Shape;
        var inputData = input.ToArray();

        // Validate and normalize axes
        var normalizedAxes = ValidateAndNormalizeAxes(axes, inputShape.Length);

        var outputShapeList = new List<int>();
        for (int i = 0; i < inputShape.Length; i++)
        {
            if (normalizedAxes.Contains(i))
            {
                if (keepDims) outputShapeList.Add(1);
            }
            else
            {
                outputShapeList.Add(inputShape[i]);
            }
        }
        var outputShape = outputShapeList.Count > 0 ? outputShapeList.ToArray() : [1];

        int outputSize = outputShape.Aggregate(1, (a, b) => a * b);
        var outputData = new T[outputSize];
        var counts = new int[outputSize];

        for (int i = 0; i < outputSize; i++)
        {
            outputData[i] = numOps.Zero;
            counts[i] = 0;
        }

        var inputStrides = ComputeStrides(inputShape);
        var outputStrides = ComputeStrides(outputShape);

        for (int i = 0; i < input.Length; i++)
        {
            var multiIndex = FlatToMultiIndex(i, inputShape, inputStrides);

            var outputMultiIndex = new List<int>();
            for (int d = 0; d < inputShape.Length; d++)
            {
                if (normalizedAxes.Contains(d))
                {
                    if (keepDims) outputMultiIndex.Add(0);
                }
                else
                {
                    outputMultiIndex.Add(multiIndex[d]);
                }
            }
            if (outputMultiIndex.Count == 0) outputMultiIndex.Add(0);

            int outputIdx = MultiToFlatIndex([.. outputMultiIndex], outputShape, outputStrides);
            outputData[outputIdx] = numOps.Add(outputData[outputIdx], inputData[i]);
            counts[outputIdx]++;
        }

        for (int i = 0; i < outputSize; i++)
        {
            if (counts[i] > 0)
            {
                outputData[i] = numOps.Divide(outputData[i], numOps.FromDouble(counts[i]));
            }
        }

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public Tensor<T> ReduceMeanBackward<T>(Tensor<T> gradOutput, int[] inputShape, int[] axes)
    {
        if (inputShape == null || inputShape.Length == 0)
            throw new ArgumentNullException(nameof(inputShape), "inputShape cannot be null or empty");

        var numOps = MathHelper.GetNumericOperations<T>();
        int inputSize = inputShape.Aggregate(1, (a, b) => a * b);
        var gradInputData = new T[inputSize];

        // Validate and normalize axes
        var normalizedAxes = ValidateAndNormalizeAxes(axes, inputShape.Length);

        int reduceCount = 1;
        foreach (var ax in normalizedAxes)
        {
            reduceCount *= inputShape[ax];
        }
        T scale = numOps.Divide(numOps.One, numOps.FromDouble(reduceCount));

        var gradOutputData = gradOutput.ToArray();
        var gradOutputShape = gradOutput.Shape;
        var inputStrides = ComputeStrides(inputShape);
        var outputStrides = ComputeStrides(gradOutputShape);

        for (int i = 0; i < inputSize; i++)
        {
            var multiIndex = FlatToMultiIndex(i, inputShape, inputStrides);

            var outputMultiIndex = new List<int>();
            int d2 = 0;
            for (int d = 0; d < inputShape.Length; d++)
            {
                if (normalizedAxes.Contains(d))
                {
                    if (d2 < gradOutputShape.Length && gradOutputShape[d2] == 1)
                    {
                        outputMultiIndex.Add(0);
                        d2++;
                    }
                }
                else
                {
                    if (d2 < gradOutputShape.Length)
                    {
                        outputMultiIndex.Add(multiIndex[d]);
                        d2++;
                    }
                }
            }
            if (outputMultiIndex.Count == 0) outputMultiIndex.Add(0);

            while (outputMultiIndex.Count < gradOutputShape.Length)
                outputMultiIndex.Add(0);
            while (outputMultiIndex.Count > gradOutputShape.Length)
                outputMultiIndex.RemoveAt(outputMultiIndex.Count - 1);

            int outputIdx = MultiToFlatIndex([.. outputMultiIndex], gradOutputShape, outputStrides);
            if (outputIdx < 0 || outputIdx >= gradOutputData.Length)
                throw new InvalidOperationException($"Output index {outputIdx} out of range [0, {gradOutputData.Length}). This indicates a shape mismatch between gradOutput and the expected shape.");
            gradInputData[i] = numOps.Multiply(gradOutputData[outputIdx], scale);
        }

        return new Tensor<T>(inputShape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public Tensor<T> ReduceVariance<T>(Tensor<T> input, int[] axes, bool keepDims)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();
        var inputData = input.ToArray();
        var inputShape = input.Shape;

        // First compute the mean
        var mean = ReduceMean(input, axes, keepDims: true);
        var meanData = mean.ToArray();
        var meanShape = mean.Shape;

        // Normalize axes
        var normalizedAxes = ValidateAndNormalizeAxes(axes, inputShape.Length);

        // Compute output shape
        var outputShapeList = new List<int>();
        for (int d = 0; d < inputShape.Length; d++)
        {
            if (normalizedAxes.Contains(d))
            {
                if (keepDims) outputShapeList.Add(1);
            }
            else
            {
                outputShapeList.Add(inputShape[d]);
            }
        }
        if (outputShapeList.Count == 0) outputShapeList.Add(1);
        var outputShape = outputShapeList.ToArray();

        int outputSize = outputShape.Aggregate(1, (a, b) => a * b);
        var outputData = new T[outputSize];

        // Compute reduction count (number of elements being reduced)
        int reduceCount = 1;
        foreach (var ax in normalizedAxes)
        {
            reduceCount *= inputShape[ax];
        }
        T scale = numOps.Divide(numOps.One, numOps.FromDouble(reduceCount));

        var inputStrides = ComputeStrides(inputShape);
        var outputStrides = ComputeStrides(outputShape);
        var meanStrides = ComputeStrides(meanShape);

        // Accumulate squared differences from mean
        int inputSize = inputData.Length;
        for (int i = 0; i < inputSize; i++)
        {
            var multiIndex = FlatToMultiIndex(i, inputShape, inputStrides);

            var outputMultiIndex = new List<int>();
            for (int d = 0; d < inputShape.Length; d++)
            {
                if (normalizedAxes.Contains(d))
                {
                    if (keepDims) outputMultiIndex.Add(0);
                }
                else
                {
                    outputMultiIndex.Add(multiIndex[d]);
                }
            }
            if (outputMultiIndex.Count == 0) outputMultiIndex.Add(0);

            int outputIdx = MultiToFlatIndex([.. outputMultiIndex], outputShape, outputStrides);

            // Get mean value (mean tensor has keepDims=true shape)
            var meanMultiIndex = new List<int>();
            for (int d = 0; d < inputShape.Length; d++)
            {
                if (normalizedAxes.Contains(d))
                    meanMultiIndex.Add(0);
                else
                    meanMultiIndex.Add(multiIndex[d]);
            }
            int meanIdx = MultiToFlatIndex([.. meanMultiIndex], meanShape, meanStrides);

            T diff = numOps.Subtract(inputData[i], meanData[meanIdx]);
            T squaredDiff = numOps.Multiply(diff, diff);
            outputData[outputIdx] = numOps.Add(outputData[outputIdx], squaredDiff);
        }

        // Divide by count to get variance
        for (int i = 0; i < outputSize; i++)
        {
            outputData[i] = numOps.Multiply(outputData[i], scale);
        }

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public Tensor<T> ReduceVarianceBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> mean, int[] axes)
    {
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (mean == null)
            throw new ArgumentNullException(nameof(mean));

        var numOps = MathHelper.GetNumericOperations<T>();
        var inputData = input.ToArray();
        var inputShape = input.Shape;
        var meanData = mean.ToArray();
        var meanShape = mean.Shape;
        var gradOutputData = gradOutput.ToArray();
        var gradOutputShape = gradOutput.Shape;

        int inputSize = inputData.Length;
        var gradInputData = new T[inputSize];

        var normalizedAxes = ValidateAndNormalizeAxes(axes, inputShape.Length);

        int reduceCount = 1;
        foreach (var ax in normalizedAxes)
        {
            reduceCount *= inputShape[ax];
        }
        T scale = numOps.Divide(numOps.FromDouble(2.0), numOps.FromDouble(reduceCount));

        var inputStrides = ComputeStrides(inputShape);
        var outputStrides = ComputeStrides(gradOutputShape);
        var meanStrides = ComputeStrides(meanShape);

        for (int i = 0; i < inputSize; i++)
        {
            var multiIndex = FlatToMultiIndex(i, inputShape, inputStrides);

            // Map to output and mean indices using helper methods
            int outputIdx = MapToReducedIndex(multiIndex, inputShape, gradOutputShape, normalizedAxes, outputStrides);
            int meanIdx = MapToMeanIndex(multiIndex, inputShape, meanShape, normalizedAxes, meanStrides);

            // gradient = 2 * (x - mean) * gradOutput / N
            T diff = numOps.Subtract(inputData[i], meanData[meanIdx]);
            gradInputData[i] = numOps.Multiply(numOps.Multiply(diff, scale), gradOutputData[outputIdx]);
        }

        return new Tensor<T>(inputShape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public Tensor<T> ReduceLogVariance<T>(Tensor<T> input, int[] axes, bool keepDims, double epsilon = 1e-8)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();

        // Compute variance first
        var variance = ReduceVariance(input, axes, keepDims);
        var varianceData = variance.ToArray();

        // Apply log(variance + epsilon)
        T eps = numOps.FromDouble(epsilon);
        for (int i = 0; i < varianceData.Length; i++)
        {
            varianceData[i] = numOps.Log(numOps.Add(varianceData[i], eps));
        }

        return new Tensor<T>(variance.Shape, new Vector<T>(varianceData));
    }

    /// <inheritdoc/>
    public Tensor<T> ReduceLogVarianceBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> mean, Tensor<T> variance, int[] axes)
    {
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (mean == null)
            throw new ArgumentNullException(nameof(mean));
        if (variance == null)
            throw new ArgumentNullException(nameof(variance));

        var numOps = MathHelper.GetNumericOperations<T>();
        var inputData = input.ToArray();
        var inputShape = input.Shape;
        var meanData = mean.ToArray();
        var meanShape = mean.Shape;
        var varianceData = variance.ToArray();
        var varianceShape = variance.Shape;
        var gradOutputData = gradOutput.ToArray();
        var gradOutputShape = gradOutput.Shape;

        int inputSize = inputData.Length;
        var gradInputData = new T[inputSize];

        var normalizedAxes = ValidateAndNormalizeAxes(axes, inputShape.Length);

        int reduceCount = 1;
        foreach (var ax in normalizedAxes)
        {
            reduceCount *= inputShape[ax];
        }
        T scale = numOps.Divide(numOps.FromDouble(2.0), numOps.FromDouble(reduceCount));

        var inputStrides = ComputeStrides(inputShape);
        var outputStrides = ComputeStrides(gradOutputShape);
        var meanStrides = ComputeStrides(meanShape);
        var varianceStrides = ComputeStrides(varianceShape);

        for (int i = 0; i < inputSize; i++)
        {
            var multiIndex = FlatToMultiIndex(i, inputShape, inputStrides);

            // Map to output/variance index
            var outputMultiIndex = new List<int>();
            int d2 = 0;
            for (int d = 0; d < inputShape.Length; d++)
            {
                if (normalizedAxes.Contains(d))
                {
                    if (d2 < gradOutputShape.Length && gradOutputShape[d2] == 1)
                    {
                        outputMultiIndex.Add(0);
                        d2++;
                    }
                }
                else
                {
                    if (d2 < gradOutputShape.Length)
                    {
                        outputMultiIndex.Add(multiIndex[d]);
                        d2++;
                    }
                }
            }
            if (outputMultiIndex.Count == 0) outputMultiIndex.Add(0);
            while (outputMultiIndex.Count < gradOutputShape.Length) outputMultiIndex.Add(0);
            while (outputMultiIndex.Count > gradOutputShape.Length) outputMultiIndex.RemoveAt(outputMultiIndex.Count - 1);

            int outputIdx = MultiToFlatIndex([.. outputMultiIndex], gradOutputShape, outputStrides);

            // Map to mean index
            var meanMultiIndex = new List<int>();
            for (int d = 0; d < inputShape.Length; d++)
            {
                if (normalizedAxes.Contains(d))
                    meanMultiIndex.Add(0);
                else
                    meanMultiIndex.Add(multiIndex[d]);
            }
            int meanIdx = MultiToFlatIndex([.. meanMultiIndex], meanShape, meanStrides);
            int varianceIdx = MultiToFlatIndex([.. outputMultiIndex], varianceShape, varianceStrides);

            // gradient = 2 * (x - mean) / (N * variance) * gradOutput
            // = (x - mean) * scale / variance * gradOutput
            T diff = numOps.Subtract(inputData[i], meanData[meanIdx]);
            T varianceVal = varianceData[varianceIdx];
            // Avoid division by zero
            if (numOps.LessThanOrEquals(varianceVal, numOps.Zero))
                varianceVal = numOps.FromDouble(1e-8);
            T gradScale = numOps.Divide(scale, varianceVal);
            gradInputData[i] = numOps.Multiply(numOps.Multiply(diff, gradScale), gradOutputData[outputIdx]);
        }

        return new Tensor<T>(inputShape, new Vector<T>(gradInputData));
    }

    // Helper methods for reduction operations
    private static int[] ComputeStrides(int[] shape)
    {
        var strides = new int[shape.Length];
        int stride = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    private static int[] FlatToMultiIndex(int flatIndex, int[] shape, int[] strides)
    {
        var multiIndex = new int[shape.Length];
        for (int i = 0; i < shape.Length; i++)
        {
            multiIndex[i] = flatIndex / strides[i];
            flatIndex %= strides[i];
        }
        return multiIndex;
    }

    private static int MultiToFlatIndex(int[] multiIndex, int[] shape, int[] strides)
    {
        int flatIndex = 0;
        for (int i = 0; i < multiIndex.Length; i++)
        {
            flatIndex += multiIndex[i] * strides[i];
        }
        return flatIndex;
    }

    /// <summary>
    /// Maps a multi-index from input space to reduced output space for variance backward pass.
    /// </summary>
    private static int MapToReducedIndex(int[] multiIndex, int[] inputShape, int[] outputShape, int[] normalizedAxes, int[] outputStrides)
    {
        var outputMultiIndex = new List<int>();
        int d2 = 0;
        for (int d = 0; d < inputShape.Length; d++)
        {
            if (Array.IndexOf(normalizedAxes, d) >= 0)
            {
                if (d2 < outputShape.Length && outputShape[d2] == 1)
                {
                    outputMultiIndex.Add(0);
                    d2++;
                }
            }
            else
            {
                if (d2 < outputShape.Length)
                {
                    outputMultiIndex.Add(multiIndex[d]);
                    d2++;
                }
            }
        }
        if (outputMultiIndex.Count == 0) outputMultiIndex.Add(0);
        while (outputMultiIndex.Count < outputShape.Length) outputMultiIndex.Add(0);
        while (outputMultiIndex.Count > outputShape.Length) outputMultiIndex.RemoveAt(outputMultiIndex.Count - 1);

        return MultiToFlatIndex([.. outputMultiIndex], outputShape, outputStrides);
    }

    /// <summary>
    /// Maps a multi-index from input space to mean tensor space for variance backward pass.
    /// </summary>
    private static int MapToMeanIndex(int[] multiIndex, int[] inputShape, int[] meanShape, int[] normalizedAxes, int[] meanStrides)
    {
        var meanMultiIndex = new List<int>();
        for (int d = 0; d < inputShape.Length; d++)
        {
            if (Array.IndexOf(normalizedAxes, d) >= 0)
                meanMultiIndex.Add(0);
            else
                meanMultiIndex.Add(multiIndex[d]);
        }
        return MultiToFlatIndex([.. meanMultiIndex], meanShape, meanStrides);
    }

    #endregion

    #region Spatial Operations

    /// <inheritdoc/>
    public Tensor<T> Upsample<T>(Tensor<T> input, int scaleH, int scaleW)
    {
        var shape = input.Shape;
        if (shape.Length != 4)
            throw new ArgumentException("Upsample expects 4D tensor [batch, channels, height, width]");

        int batch = shape[0];
        int channels = shape[1];
        int height = shape[2];
        int width = shape[3];

        int newHeight = height * scaleH;
        int newWidth = width * scaleW;

        var inputData = input.ToArray();
        var outputData = new T[batch * channels * newHeight * newWidth];

        Parallel.For(0, batch * channels, bc =>
        {
            int b = bc / channels;
            int c = bc % channels;

            for (int oh = 0; oh < newHeight; oh++)
            {
                int ih = oh / scaleH;
                for (int ow = 0; ow < newWidth; ow++)
                {
                    int iw = ow / scaleW;
                    int inputIdx = ((b * channels + c) * height + ih) * width + iw;
                    int outputIdx = ((b * channels + c) * newHeight + oh) * newWidth + ow;
                    outputData[outputIdx] = inputData[inputIdx];
                }
            }
        });

        return new Tensor<T>([batch, channels, newHeight, newWidth], new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public Tensor<T> UpsampleBackward<T>(Tensor<T> gradOutput, int[] inputShape, int scaleH, int scaleW)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = inputShape[0];
        int channels = inputShape[1];
        int height = inputShape[2];
        int width = inputShape[3];

        int newHeight = height * scaleH;
        int newWidth = width * scaleW;

        var gradOutputData = gradOutput.ToArray();
        var gradInputData = new T[batch * channels * height * width];

        for (int i = 0; i < gradInputData.Length; i++)
            gradInputData[i] = numOps.Zero;

        Parallel.For(0, batch * channels, bc =>
        {
            int b = bc / channels;
            int c = bc % channels;

            for (int oh = 0; oh < newHeight; oh++)
            {
                int ih = oh / scaleH;
                for (int ow = 0; ow < newWidth; ow++)
                {
                    int iw = ow / scaleW;
                    int gradOutputIdx = ((b * channels + c) * newHeight + oh) * newWidth + ow;
                    int gradInputIdx = ((b * channels + c) * height + ih) * width + iw;
                    // No lock needed - each (batch, channel) partition owns disjoint gradInput slices
                    gradInputData[gradInputIdx] = numOps.Add(gradInputData[gradInputIdx], gradOutputData[gradOutputIdx]);
                }
            }
        });

        return new Tensor<T>(inputShape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public Tensor<T> PixelShuffle<T>(Tensor<T> input, int upscaleFactor)
    {
        var shape = input.Shape;
        if (shape.Length != 4)
            throw new ArgumentException("PixelShuffle expects 4D tensor [batch, channels, height, width]");

        int batch = shape[0];
        int channels = shape[1];
        int height = shape[2];
        int width = shape[3];

        int r = upscaleFactor;
        if (channels % (r * r) != 0)
            throw new ArgumentException($"Number of channels ({channels}) must be divisible by r^2 ({r * r})");

        int newChannels = channels / (r * r);
        int newHeight = height * r;
        int newWidth = width * r;

        var inputData = input.ToArray();
        var outputData = new T[batch * newChannels * newHeight * newWidth];

        Parallel.For(0, batch, b =>
        {
            for (int oc = 0; oc < newChannels; oc++)
            {
                for (int oh = 0; oh < newHeight; oh++)
                {
                    for (int ow = 0; ow < newWidth; ow++)
                    {
                        int ih = oh / r;
                        int iw = ow / r;
                        int subH = oh % r;
                        int subW = ow % r;
                        int ic = oc * r * r + subH * r + subW;

                        int inputIdx = ((b * channels + ic) * height + ih) * width + iw;
                        int outputIdx = ((b * newChannels + oc) * newHeight + oh) * newWidth + ow;
                        outputData[outputIdx] = inputData[inputIdx];
                    }
                }
            }
        });

        return new Tensor<T>([batch, newChannels, newHeight, newWidth], new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public Tensor<T> PixelShuffleBackward<T>(Tensor<T> gradOutput, int[] inputShape, int upscaleFactor)
    {
        int batch = inputShape[0];
        int channels = inputShape[1];
        int height = inputShape[2];
        int width = inputShape[3];

        int r = upscaleFactor;
        int newChannels = channels / (r * r);
        int newHeight = height * r;
        int newWidth = width * r;

        var gradOutputData = gradOutput.ToArray();
        var gradInputData = new T[batch * channels * height * width];

        Parallel.For(0, batch, b =>
        {
            for (int oc = 0; oc < newChannels; oc++)
            {
                for (int oh = 0; oh < newHeight; oh++)
                {
                    for (int ow = 0; ow < newWidth; ow++)
                    {
                        int ih = oh / r;
                        int iw = ow / r;
                        int subH = oh % r;
                        int subW = ow % r;
                        int ic = oc * r * r + subH * r + subW;

                        int gradInputIdx = ((b * channels + ic) * height + ih) * width + iw;
                        int gradOutputIdx = ((b * newChannels + oc) * newHeight + oh) * newWidth + ow;
                        gradInputData[gradInputIdx] = gradOutputData[gradOutputIdx];
                    }
                }
            }
        });

        return new Tensor<T>(inputShape, new Vector<T>(gradInputData));
    }

    public Tensor<T> AffineGrid<T>(Tensor<T> theta, int outputHeight, int outputWidth)
    {
        if (theta == null) throw new ArgumentNullException(nameof(theta));
        if (theta.Shape.Length != 3 || theta.Shape[1] != 2 || theta.Shape[2] != 3)
            throw new ArgumentException("AffineGrid expects theta shape [batch, 2, 3]");

        int batchSize = theta.Shape[0];
        var grid = new Tensor<T>([batchSize, outputHeight, outputWidth, 2]);
        var numOps = MathHelper.GetNumericOperations<T>();

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < outputHeight; h++)
            {
                T yNorm = outputHeight == 1
                    ? numOps.Zero
                    : numOps.FromDouble((double)h / (outputHeight - 1) * 2.0 - 1.0);

                for (int w = 0; w < outputWidth; w++)
                {
                    T xNorm = outputWidth == 1
                        ? numOps.Zero
                        : numOps.FromDouble((double)w / (outputWidth - 1) * 2.0 - 1.0);

                    T xPrime = numOps.Add(
                        numOps.Add(
                            numOps.Multiply(theta[b, 0, 0], xNorm),
                            numOps.Multiply(theta[b, 0, 1], yNorm)),
                        theta[b, 0, 2]);

                    T yPrime = numOps.Add(
                        numOps.Add(
                            numOps.Multiply(theta[b, 1, 0], xNorm),
                            numOps.Multiply(theta[b, 1, 1], yNorm)),
                        theta[b, 1, 2]);

                    grid[b, h, w, 0] = xPrime;
                    grid[b, h, w, 1] = yPrime;
                }
            }
        }

        return grid;
    }

    public Tensor<T> GridSample<T>(Tensor<T> input, Tensor<T> grid)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (grid == null) throw new ArgumentNullException(nameof(grid));

        if (input.Shape.Length != 4)
            throw new ArgumentException("GridSample expects input shape [batch, height, width, channels]");
        if (grid.Shape.Length != 4 || grid.Shape[3] != 2)
            throw new ArgumentException("GridSample expects grid shape [batch, outH, outW, 2]");
        if (input.Shape[0] != grid.Shape[0])
            throw new ArgumentException("GridSample batch size mismatch between input and grid");

        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = input.Shape[0];
        int inH = input.Shape[1];
        int inW = input.Shape[2];
        int channels = input.Shape[3];
        int outH = grid.Shape[1];
        int outW = grid.Shape[2];

        var output = new Tensor<T>([batch, outH, outW, channels]);

        T heightScale = numOps.FromDouble((inH - 1) / 2.0);
        T widthScale = numOps.FromDouble((inW - 1) / 2.0);

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < outH; h++)
            {
                for (int w = 0; w < outW; w++)
                {
                    T gridX = grid[b, h, w, 0];
                    T gridY = grid[b, h, w, 1];

                    T srcX = numOps.Multiply(numOps.Add(gridX, numOps.One), widthScale);
                    T srcY = numOps.Multiply(numOps.Add(gridY, numOps.One), heightScale);

                    double srcXDouble = Convert.ToDouble(srcX);
                    double srcYDouble = Convert.ToDouble(srcY);
                    int x0 = Math.Max(0, Math.Min((int)Math.Floor(srcXDouble), inW - 1));
                    int x1 = Math.Max(0, Math.Min(x0 + 1, inW - 1));
                    int y0 = Math.Max(0, Math.Min((int)Math.Floor(srcYDouble), inH - 1));
                    int y1 = Math.Max(0, Math.Min(y0 + 1, inH - 1));

                    T wx1 = numOps.Subtract(srcX, numOps.FromDouble(x0));
                    T wx0 = numOps.Subtract(numOps.One, wx1);
                    T wy1 = numOps.Subtract(srcY, numOps.FromDouble(y0));
                    T wy0 = numOps.Subtract(numOps.One, wy1);

                    for (int c = 0; c < channels; c++)
                    {
                        T v00 = input[b, y0, x0, c];
                        T v01 = input[b, y0, x1, c];
                        T v10 = input[b, y1, x0, c];
                        T v11 = input[b, y1, x1, c];

                        T interp = numOps.Add(
                            numOps.Add(
                                numOps.Multiply(numOps.Multiply(v00, wx0), wy0),
                                numOps.Multiply(numOps.Multiply(v01, wx1), wy0)),
                            numOps.Add(
                                numOps.Multiply(numOps.Multiply(v10, wx0), wy1),
                                numOps.Multiply(numOps.Multiply(v11, wx1), wy1)));

                        output[b, h, w, c] = interp;
                    }
                }
            }
        }

        return output;
    }

    public (Tensor<T> real, Tensor<T> imag) ComplexMatMul<T>(Tensor<T> aReal, Tensor<T> aImag, Tensor<T> bReal, Tensor<T> bImag)
    {
        if (aReal == null || aImag == null || bReal == null || bImag == null)
            throw new ArgumentNullException("ComplexMatMul inputs cannot be null");
        var aShape = aReal.Shape;
        var bShape = bReal.Shape;
        if (aShape.Length != 2 || bShape.Length != 2 || aShape[1] != bShape[0])
            throw new ArgumentException("ComplexMatMul expects shapes [M,K] x [K,N]");

        int m = aShape[0];
        int k = aShape[1];
        int n = bShape[1];

        var realOut = new Tensor<T>([m, n]);
        var imagOut = new Tensor<T>([m, n]);

        var numOps = MathHelper.GetNumericOperations<T>();

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                T realSum = numOps.Zero;
                T imagSum = numOps.Zero;
                for (int kk = 0; kk < k; kk++)
                {
                    var ar = aReal[i, kk];
                    var ai = aImag[i, kk];
                    var br = bReal[kk, j];
                    var bi = bImag[kk, j];
                    // (ar + i ai) * (br + i bi) = (ar*br - ai*bi) + i(ar*bi + ai*br)
                    realSum = numOps.Add(realSum, numOps.Subtract(numOps.Multiply(ar, br), numOps.Multiply(ai, bi)));
                    imagSum = numOps.Add(imagSum, numOps.Add(numOps.Multiply(ar, bi), numOps.Multiply(ai, br)));
                }
                realOut[i, j] = realSum;
                imagOut[i, j] = imagSum;
            }
        }

        return (realOut, imagOut);
    }

    public Tensor<T> ComplexMagnitudeSquared<T>(Tensor<T> real, Tensor<T> imag)
    {
        if (real == null || imag == null)
            throw new ArgumentNullException("ComplexMagnitudeSquared inputs cannot be null");
        if (!real.Shape.SequenceEqual(imag.Shape))
            throw new ArgumentException("Real and imaginary parts must have the same shape");

        var numOps = MathHelper.GetNumericOperations<T>();
        var output = new Tensor<T>(real.Shape);
        for (int idx = 0; idx < real.Length; idx++)
        {
            var r = real[idx];
            var i = imag[idx];
            output[idx] = numOps.Add(numOps.Multiply(r, r), numOps.Multiply(i, i));
        }
        return output;
    }

    public (Tensor<T> real, Tensor<T> imag) ComplexNormalize<T>(Tensor<T> real, Tensor<T> imag)
    {
        var magSq = ComplexMagnitudeSquared(real, imag);
        var total = TensorSum(magSq);
        var numOps = MathHelper.GetNumericOperations<T>();
        if (numOps.Equals(total, numOps.Zero))
            return (real.Clone(), imag.Clone());
        var denom = numOps.Sqrt(total);
        var denomTensor = new Tensor<T>(magSq.Shape);
        denomTensor.Fill(denom);
        var realNorm = TensorDivide(real, denomTensor);
        var imagNorm = TensorDivide(imag, denomTensor);
        return (realNorm, imagNorm);
    }

    /// <inheritdoc/>
    public Tensor<T> Crop<T>(Tensor<T> input, int top, int left, int height, int width)
    {
        var shape = input.Shape;
        if (shape.Length != 4)
            throw new ArgumentException("Crop expects 4D tensor [batch, channels, height, width]");

        int batch = shape[0];
        int channels = shape[1];
        int inputHeight = shape[2];
        int inputWidth = shape[3];

        if (top < 0 || left < 0 || top + height > inputHeight || left + width > inputWidth)
            throw new ArgumentException("Crop region is out of bounds");

        var inputData = input.ToArray();
        var outputData = new T[batch * channels * height * width];

        Parallel.For(0, batch * channels, bc =>
        {
            int b = bc / channels;
            int c = bc % channels;

            for (int oh = 0; oh < height; oh++)
            {
                int ih = top + oh;
                for (int ow = 0; ow < width; ow++)
                {
                    int iw = left + ow;
                    int inputIdx = ((b * channels + c) * inputHeight + ih) * inputWidth + iw;
                    int outputIdx = ((b * channels + c) * height + oh) * width + ow;
                    outputData[outputIdx] = inputData[inputIdx];
                }
            }
        });

        return new Tensor<T>([batch, channels, height, width], new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public Tensor<T> CropBackward<T>(Tensor<T> gradOutput, int[] inputShape, int top, int left)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = inputShape[0];
        int channels = inputShape[1];
        int inputHeight = inputShape[2];
        int inputWidth = inputShape[3];

        var gradOutputShape = gradOutput.Shape;
        int cropHeight = gradOutputShape[2];
        int cropWidth = gradOutputShape[3];

        var gradOutputData = gradOutput.ToArray();
        var gradInputData = new T[batch * channels * inputHeight * inputWidth];

        for (int i = 0; i < gradInputData.Length; i++)
            gradInputData[i] = numOps.Zero;

        Parallel.For(0, batch * channels, bc =>
        {
            int b = bc / channels;
            int c = bc % channels;

            for (int oh = 0; oh < cropHeight; oh++)
            {
                int ih = top + oh;
                for (int ow = 0; ow < cropWidth; ow++)
                {
                    int iw = left + ow;
                    int gradOutputIdx = ((b * channels + c) * cropHeight + oh) * cropWidth + ow;
                    int gradInputIdx = ((b * channels + c) * inputHeight + ih) * inputWidth + iw;
                    gradInputData[gradInputIdx] = gradOutputData[gradOutputIdx];
                }
            }
        });

        return new Tensor<T>(inputShape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public Tensor<T> Pad<T>(Tensor<T> input, int padTop, int padBottom, int padLeft, int padRight, T padValue)
    {
        var shape = input.Shape;
        if (shape.Length < 2)
            throw new ArgumentException("Pad expects at least 2D tensor");

        int rank = shape.Length;
        int height = shape[rank - 2];
        int width = shape[rank - 1];

        int newHeight = height + padTop + padBottom;
        int newWidth = width + padLeft + padRight;

        int batchSize = 1;
        for (int i = 0; i < rank - 2; i++)
            batchSize *= shape[i];

        var inputData = input.ToArray();
        var outputData = new T[batchSize * newHeight * newWidth];

        for (int i = 0; i < outputData.Length; i++)
            outputData[i] = padValue;

        Parallel.For(0, batchSize, b =>
        {
            for (int ih = 0; ih < height; ih++)
            {
                int oh = ih + padTop;
                for (int iw = 0; iw < width; iw++)
                {
                    int ow = iw + padLeft;
                    int inputIdx = b * height * width + ih * width + iw;
                    int outputIdx = b * newHeight * newWidth + oh * newWidth + ow;
                    outputData[outputIdx] = inputData[inputIdx];
                }
            }
        });

        var newShape = (int[])shape.Clone();
        newShape[rank - 2] = newHeight;
        newShape[rank - 1] = newWidth;

        return new Tensor<T>(newShape, new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public Tensor<T> PadBackward<T>(Tensor<T> gradOutput, int padTop, int padLeft, int[] inputShape)
    {
        int rank = inputShape.Length;
        int height = inputShape[rank - 2];
        int width = inputShape[rank - 1];

        int batchSize = 1;
        for (int i = 0; i < rank - 2; i++)
            batchSize *= inputShape[i];

        var gradOutputShape = gradOutput.Shape;
        int paddedHeight = gradOutputShape[rank - 2];
        int paddedWidth = gradOutputShape[rank - 1];

        var gradOutputData = gradOutput.ToArray();
        var gradInputData = new T[batchSize * height * width];

        Parallel.For(0, batchSize, b =>
        {
            for (int ih = 0; ih < height; ih++)
            {
                int oh = ih + padTop;
                for (int iw = 0; iw < width; iw++)
                {
                    int ow = iw + padLeft;
                    int gradOutputIdx = b * paddedHeight * paddedWidth + oh * paddedWidth + ow;
                    int gradInputIdx = b * height * width + ih * width + iw;
                    gradInputData[gradInputIdx] = gradOutputData[gradOutputIdx];
                }
            }
        });

        return new Tensor<T>(inputShape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public Tensor<T> Concat<T>(IReadOnlyList<Tensor<T>> tensors, int axis)
    {
        if (tensors == null || tensors.Count == 0)
            throw new ArgumentException("At least one tensor required for concatenation");

        var firstShape = tensors[0].Shape;
        int rank = firstShape.Length;

        if (axis < 0) axis = rank + axis;
        if (axis < 0 || axis >= rank)
            throw new ArgumentException($"Invalid axis {axis} for tensor with {rank} dimensions");

        int totalAxisSize = 0;
        foreach (var tensor in tensors)
        {
            if (tensor.Shape.Length != rank)
                throw new ArgumentException("All tensors must have the same number of dimensions");

            for (int i = 0; i < rank; i++)
            {
                if (i != axis && tensor.Shape[i] != firstShape[i])
                    throw new ArgumentException($"All tensors must have the same shape except along axis {axis}");
            }

            totalAxisSize += tensor.Shape[axis];
        }

        var outputShape = (int[])firstShape.Clone();
        outputShape[axis] = totalAxisSize;

        int outputSize = outputShape.Aggregate(1, (a, b) => a * b);
        var outputData = new T[outputSize];

        var outputStrides = ComputeStrides(outputShape);

        int axisOffset = 0;
        foreach (var tensor in tensors)
        {
            var tensorData = tensor.ToArray();
            var tensorShape = tensor.Shape;
            var tensorStrides = ComputeStrides(tensorShape);

            for (int i = 0; i < tensor.Length; i++)
            {
                var multiIndex = FlatToMultiIndex(i, tensorShape, tensorStrides);
                multiIndex[axis] += axisOffset;
                int outputIdx = MultiToFlatIndex(multiIndex, outputShape, outputStrides);
                outputData[outputIdx] = tensorData[i];
            }

            axisOffset += tensor.Shape[axis];
        }

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public T TensorSumOfSquares<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        T sum = numOps.Zero;
        int length = tensor.Length;

        // Use SIMD-friendly sequential access pattern
        var data = tensor.ToArray();
        for (int i = 0; i < length; i++)
        {
            T val = data[i];
            sum = numOps.Add(sum, numOps.Multiply(val, val));
        }

        return sum;
    }

    /// <inheritdoc/>
    public Tensor<TValue> TensorEmbeddingLookup<TValue, TIndex>(Tensor<TValue> embeddings, Tensor<TIndex> indices)
        where TIndex : unmanaged
    {
        if (embeddings == null) throw new ArgumentNullException(nameof(embeddings));
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        if (embeddings.Rank != 2)
            throw new ArgumentException($"Embeddings must be a 2D tensor [vocab_size, embedding_dim]. Got rank {embeddings.Rank}.");

        int vocabSize = embeddings.Shape[0];
        int embeddingDim = embeddings.Shape[1];
        int numIndices = indices.Length;

        // Output shape is [*indices.shape, embedding_dim]
        var outputShape = new int[indices.Rank + 1];
        for (int i = 0; i < indices.Rank; i++)
        {
            outputShape[i] = indices.Shape[i];
        }
        outputShape[indices.Rank] = embeddingDim;

        var result = new Tensor<TValue>(outputShape);
        var embData = embeddings.ToArray();
        var idxData = indices.ToArray();

        // For each index, copy the entire embedding row
        for (int i = 0; i < numIndices; i++)
        {
            int tokenIdx = Convert.ToInt32(idxData[i]);

            // Bounds check for index
            if (tokenIdx < 0 || tokenIdx >= vocabSize)
                throw new ArgumentOutOfRangeException(nameof(indices),
                    $"Index {tokenIdx} at position {i} is out of bounds for embedding table with vocabulary size {vocabSize}.");

            int srcOffset = tokenIdx * embeddingDim;
            int dstOffset = i * embeddingDim;

            // Vectorized copy of embedding row
            for (int d = 0; d < embeddingDim; d++)
            {
                result.SetFlat(dstOffset + d, embData[srcOffset + d]);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<TValue> TensorEmbeddingLookupBackward<TValue, TIndex>(Tensor<TValue> gradOutput, Tensor<TIndex> indices, int vocabSize, int embeddingDim)
        where TIndex : unmanaged
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        if (vocabSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(vocabSize), "Vocabulary size must be positive.");
        if (embeddingDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(embeddingDim), "Embedding dimension must be positive.");

        var numOps = MathHelper.GetNumericOperations<TValue>();
        var gradEmbeddings = new Tensor<TValue>(new[] { vocabSize, embeddingDim });

        var gradData = gradOutput.ToArray();
        var idxData = indices.ToArray();
        int numIndices = indices.Length;

        // Scatter-add: for each index, accumulate gradients to the embedding row
        for (int i = 0; i < numIndices; i++)
        {
            int tokenIdx = Convert.ToInt32(idxData[i]);

            // Bounds check for index
            if (tokenIdx < 0 || tokenIdx >= vocabSize)
                throw new ArgumentOutOfRangeException(nameof(indices),
                    $"Index {tokenIdx} at position {i} is out of bounds for vocabulary size {vocabSize}.");

            int srcOffset = i * embeddingDim;
            int dstOffset = tokenIdx * embeddingDim;

            // Accumulate gradient for this embedding row
            for (int d = 0; d < embeddingDim; d++)
            {
                TValue current = gradEmbeddings.GetFlat(dstOffset + d);
                TValue grad = gradData[srcOffset + d];
                gradEmbeddings.SetFlat(dstOffset + d, numOps.Add(current, grad));
            }
        }

        return gradEmbeddings;
    }

    /// <inheritdoc/>
    public Tensor<T> RBFKernel<T>(Tensor<T> input, Tensor<T> centers, Tensor<T> epsilons)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (centers == null) throw new ArgumentNullException(nameof(centers));
        if (epsilons == null) throw new ArgumentNullException(nameof(epsilons));

        if (input.Rank != 2)
            throw new ArgumentException($"input must be 2D [batch, features], got rank {input.Rank}", nameof(input));
        if (centers.Rank != 2)
            throw new ArgumentException($"centers must be 2D [numCenters, features], got rank {centers.Rank}", nameof(centers));
        if (epsilons.Rank != 1)
            throw new ArgumentException($"epsilons must be 1D [numCenters], got rank {epsilons.Rank}", nameof(epsilons));

        var numOps = MathHelper.GetNumericOperations<T>();
        int batchSize = input.Shape[0];
        int features = input.Shape[1];
        int numCenters = centers.Shape[0];

        if (centers.Shape[1] != features)
            throw new ArgumentException($"centers features dimension ({centers.Shape[1]}) must match input features ({features})", nameof(centers));
        if (epsilons.Shape[0] != numCenters)
            throw new ArgumentException($"epsilons length ({epsilons.Shape[0]}) must match number of centers ({numCenters})", nameof(epsilons));

        var output = new Tensor<T>([batchSize, numCenters]);
        var inputData = input.ToArray();
        var centersData = centers.ToArray();
        var epsilonsData = epsilons.ToArray();

        // Compute exp(-epsilon * ||x - center||²) for each (sample, center) pair
        for (int b = 0; b < batchSize; b++)
        {
            int inputOffset = b * features;
            for (int c = 0; c < numCenters; c++)
            {
                int centerOffset = c * features;
                T distSquared = numOps.Zero;

                // Compute ||x - center||²
                for (int f = 0; f < features; f++)
                {
                    T diff = numOps.Subtract(inputData[inputOffset + f], centersData[centerOffset + f]);
                    distSquared = numOps.Add(distSquared, numOps.Multiply(diff, diff));
                }

                // Compute exp(-epsilon * distSquared)
                T negEpsDist = numOps.Negate(numOps.Multiply(epsilonsData[c], distSquared));
                output[b, c] = numOps.Exp(negEpsDist);
            }
        }

        return output;
    }

    /// <inheritdoc/>
    public (Tensor<T> gradInput, Tensor<T> gradCenters, Tensor<T> gradEpsilons) RBFKernelBackward<T>(
        Tensor<T> gradOutput, Tensor<T> input, Tensor<T> centers, Tensor<T> epsilons, Tensor<T> output)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (centers == null) throw new ArgumentNullException(nameof(centers));
        if (epsilons == null) throw new ArgumentNullException(nameof(epsilons));
        if (output == null) throw new ArgumentNullException(nameof(output));

        if (input.Rank != 2)
            throw new ArgumentException($"input must be 2D [batch, features], got rank {input.Rank}", nameof(input));
        if (centers.Rank != 2)
            throw new ArgumentException($"centers must be 2D [numCenters, features], got rank {centers.Rank}", nameof(centers));
        if (epsilons.Rank != 1)
            throw new ArgumentException($"epsilons must be 1D [numCenters], got rank {epsilons.Rank}", nameof(epsilons));
        if (gradOutput.Rank != 2)
            throw new ArgumentException($"gradOutput must be 2D [batch, numCenters], got rank {gradOutput.Rank}", nameof(gradOutput));
        if (output.Rank != 2)
            throw new ArgumentException($"output must be 2D [batch, numCenters], got rank {output.Rank}", nameof(output));

        var numOps = MathHelper.GetNumericOperations<T>();
        int batchSize = input.Shape[0];
        int features = input.Shape[1];
        int numCenters = centers.Shape[0];

        if (centers.Shape[1] != features)
            throw new ArgumentException($"centers features dimension ({centers.Shape[1]}) must match input features ({features})", nameof(centers));
        if (epsilons.Shape[0] != numCenters)
            throw new ArgumentException($"epsilons length ({epsilons.Shape[0]}) must match number of centers ({numCenters})", nameof(epsilons));
        if (gradOutput.Shape[0] != batchSize || gradOutput.Shape[1] != numCenters)
            throw new ArgumentException($"gradOutput shape [{gradOutput.Shape[0]}, {gradOutput.Shape[1]}] must be [{batchSize}, {numCenters}]", nameof(gradOutput));
        if (output.Shape[0] != batchSize || output.Shape[1] != numCenters)
            throw new ArgumentException($"output shape [{output.Shape[0]}, {output.Shape[1]}] must be [{batchSize}, {numCenters}]", nameof(output));

        var gradInput = new Tensor<T>(input.Shape);
        var gradCenters = new Tensor<T>(centers.Shape);
        var gradEpsilons = new Tensor<T>(epsilons.Shape);

        var inputData = input.ToArray();
        var centersData = centers.ToArray();
        var epsilonsData = epsilons.ToArray();
        var outputData = output.ToArray();
        var gradOutputData = gradOutput.ToArray();

        // For RBF: K = exp(-epsilon * ||x - c||²)
        // dK/dx = K * (-epsilon) * 2 * (x - c) = -2 * epsilon * K * (x - c)
        // dK/dc = K * (-epsilon) * (-2) * (x - c) = 2 * epsilon * K * (x - c) = -dK/dx
        // dK/depsilon = K * (-||x - c||²)

        for (int b = 0; b < batchSize; b++)
        {
            int inputOffset = b * features;
            for (int c = 0; c < numCenters; c++)
            {
                int centerOffset = c * features;
                int outIdx = b * numCenters + c;
                T K = outputData[outIdx];
                T eps = epsilonsData[c];
                T dL_dK = gradOutputData[outIdx];

                // Compute ||x - c||² for epsilon gradient
                T distSquared = numOps.Zero;
                for (int f = 0; f < features; f++)
                {
                    T diff = numOps.Subtract(inputData[inputOffset + f], centersData[centerOffset + f]);
                    distSquared = numOps.Add(distSquared, numOps.Multiply(diff, diff));
                }

                // dL/depsilon += dL/dK * dK/depsilon = dL/dK * K * (-distSquared)
                T gradEps = numOps.Multiply(dL_dK, numOps.Multiply(K, numOps.Negate(distSquared)));
                gradEpsilons[c] = numOps.Add(gradEpsilons[c], gradEps);

                // Common factor: -2 * epsilon * K * dL/dK
                T commonFactor = numOps.Multiply(
                    numOps.Multiply(numOps.FromDouble(-2.0), eps),
                    numOps.Multiply(K, dL_dK));

                for (int f = 0; f < features; f++)
                {
                    T diff = numOps.Subtract(inputData[inputOffset + f], centersData[centerOffset + f]);
                    T grad = numOps.Multiply(commonFactor, diff);

                    // dL/dx = commonFactor * (x - c)
                    int inputIdx = b * features + f;
                    gradInput.SetFlat(inputIdx, numOps.Add(gradInput.GetFlat(inputIdx), grad));

                    // dL/dc = -dL/dx = -commonFactor * (x - c)
                    int centerIdx = c * features + f;
                    gradCenters.SetFlat(centerIdx, numOps.Subtract(gradCenters.GetFlat(centerIdx), grad));
                }
            }
        }

        return (gradInput, gradCenters, gradEpsilons);
    }

    #endregion

    #region Tensor Shape Operations

    /// <inheritdoc/>
    public Tensor<T> TensorRepeatElements<T>(Tensor<T> tensor, int repeats, int axis = 0)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (repeats < 1) throw new ArgumentOutOfRangeException(nameof(repeats), "Repeats must be at least 1");
        if (axis < 0 || axis >= tensor.Shape.Length)
            throw new ArgumentOutOfRangeException(nameof(axis), $"Axis {axis} out of range for tensor with {tensor.Shape.Length} dimensions");

        // Calculate output shape
        var outputShape = new int[tensor.Shape.Length];
        Array.Copy(tensor.Shape, outputShape, tensor.Shape.Length);
        outputShape[axis] *= repeats;

        var result = new Tensor<T>(outputShape);

        // Calculate strides for the tensor
        int outerSize = 1;
        for (int i = 0; i < axis; i++)
            outerSize *= tensor.Shape[i];

        int axisSize = tensor.Shape[axis];
        int innerSize = 1;
        for (int i = axis + 1; i < tensor.Shape.Length; i++)
            innerSize *= tensor.Shape[i];

        // Perform the repeat operation
        Parallel.For(0, outerSize, outer =>
        {
            for (int a = 0; a < axisSize; a++)
            {
                int srcBase = (outer * axisSize + a) * innerSize;
                int dstBase = (outer * axisSize * repeats + a * repeats) * innerSize;

                for (int r = 0; r < repeats; r++)
                {
                    int dstOffset = dstBase + r * innerSize;
                    for (int inner = 0; inner < innerSize; inner++)
                    {
                        result.SetFlat(dstOffset + inner, tensor.GetFlat(srcBase + inner));
                    }
                }
            }
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorTile<T>(Tensor<T> tensor, int[] multiples)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (multiples == null) throw new ArgumentNullException(nameof(multiples));
        if (multiples.Length != tensor.Shape.Length)
            throw new ArgumentException($"Multiples length ({multiples.Length}) must match tensor dimensions ({tensor.Shape.Length})");

        // Calculate output shape
        var outputShape = new int[tensor.Shape.Length];
        for (int i = 0; i < tensor.Shape.Length; i++)
        {
            if (multiples[i] < 1)
                throw new ArgumentOutOfRangeException(nameof(multiples), $"Multiple at index {i} must be at least 1");
            outputShape[i] = tensor.Shape[i] * multiples[i];
        }

        var result = new Tensor<T>(outputShape);
        int totalElements = result.Shape.Aggregate(1, (a, b) => a * b);

        // For each output element, find the corresponding input element
        Parallel.For(0, totalElements, flatIdx =>
        {
            // Convert flat index to multi-dimensional indices
            var outputIndices = new int[outputShape.Length];
            int remaining = flatIdx;
            for (int d = outputShape.Length - 1; d >= 0; d--)
            {
                outputIndices[d] = remaining % outputShape[d];
                remaining /= outputShape[d];
            }

            // Map to input indices (modulo original size)
            var inputIndices = new int[tensor.Shape.Length];
            for (int d = 0; d < tensor.Shape.Length; d++)
            {
                inputIndices[d] = outputIndices[d] % tensor.Shape[d];
            }

            // Convert input indices to flat index
            int inputFlat = 0;
            int stride = 1;
            for (int d = tensor.Shape.Length - 1; d >= 0; d--)
            {
                inputFlat += inputIndices[d] * stride;
                stride *= tensor.Shape[d];
            }

            result.SetFlat(flatIdx, tensor.GetFlat(inputFlat));
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorSlice<T>(Tensor<T> tensor, int[] start, int[] length)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (start == null) throw new ArgumentNullException(nameof(start));
        if (length == null) throw new ArgumentNullException(nameof(length));
        if (start.Length != tensor.Shape.Length)
            throw new ArgumentException($"Start length ({start.Length}) must match tensor dimensions ({tensor.Shape.Length})");
        if (length.Length != tensor.Shape.Length)
            throw new ArgumentException($"Length length ({length.Length}) must match tensor dimensions ({tensor.Shape.Length})");

        // Validate bounds
        for (int i = 0; i < tensor.Shape.Length; i++)
        {
            if (start[i] < 0 || start[i] >= tensor.Shape[i])
                throw new ArgumentOutOfRangeException(nameof(start), $"Start index {start[i]} out of range for axis {i} with size {tensor.Shape[i]}");
            if (length[i] < 1 || start[i] + length[i] > tensor.Shape[i])
                throw new ArgumentOutOfRangeException(nameof(length), $"Slice length {length[i]} starting at {start[i]} exceeds axis {i} size {tensor.Shape[i]}");
        }

        var result = new Tensor<T>(length);
        int totalElements = length.Aggregate(1, (a, b) => a * b);

        // For each output element, find the corresponding input element
        Parallel.For(0, totalElements, flatIdx =>
        {
            // Convert flat index to output indices
            var outputIndices = new int[length.Length];
            int remaining = flatIdx;
            for (int d = length.Length - 1; d >= 0; d--)
            {
                outputIndices[d] = remaining % length[d];
                remaining /= length[d];
            }

            // Map to input indices
            var inputIndices = new int[tensor.Shape.Length];
            for (int d = 0; d < tensor.Shape.Length; d++)
            {
                inputIndices[d] = start[d] + outputIndices[d];
            }

            // Convert input indices to flat index
            int inputFlat = 0;
            int stride = 1;
            for (int d = tensor.Shape.Length - 1; d >= 0; d--)
            {
                inputFlat += inputIndices[d] * stride;
                stride *= tensor.Shape[d];
            }

            result.SetFlat(flatIdx, tensor.GetFlat(inputFlat));
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorSetSlice<T>(Tensor<T> destination, Tensor<T> source, int[] start)
    {
        if (destination == null) throw new ArgumentNullException(nameof(destination));
        if (source == null) throw new ArgumentNullException(nameof(source));
        if (start == null) throw new ArgumentNullException(nameof(start));
        if (start.Length != destination.Shape.Length)
            throw new ArgumentException($"Start length ({start.Length}) must match destination dimensions ({destination.Shape.Length})");
        if (source.Shape.Length != destination.Shape.Length)
            throw new ArgumentException($"Source dimensions ({source.Shape.Length}) must match destination dimensions ({destination.Shape.Length})");

        // Validate bounds
        for (int i = 0; i < destination.Shape.Length; i++)
        {
            if (start[i] < 0 || start[i] + source.Shape[i] > destination.Shape[i])
                throw new ArgumentOutOfRangeException(nameof(start), $"Slice starting at {start[i]} with size {source.Shape[i]} exceeds destination axis {i} size {destination.Shape[i]}");
        }

        // Create a copy of destination to avoid modifying the original
        var result = new Tensor<T>(destination.Shape);
        int destTotal = destination.Shape.Aggregate(1, (a, b) => a * b);
        for (int i = 0; i < destTotal; i++)
        {
            result.SetFlat(i, destination.GetFlat(i));
        }

        int sourceTotal = source.Shape.Aggregate(1, (a, b) => a * b);

        // Set the slice values
        Parallel.For(0, sourceTotal, flatIdx =>
        {
            // Convert flat index to source indices
            var sourceIndices = new int[source.Shape.Length];
            int remaining = flatIdx;
            for (int d = source.Shape.Length - 1; d >= 0; d--)
            {
                sourceIndices[d] = remaining % source.Shape[d];
                remaining /= source.Shape[d];
            }

            // Map to destination indices
            var destIndices = new int[destination.Shape.Length];
            for (int d = 0; d < destination.Shape.Length; d++)
            {
                destIndices[d] = start[d] + sourceIndices[d];
            }

            // Convert destination indices to flat index
            int destFlat = 0;
            int stride = 1;
            for (int d = destination.Shape.Length - 1; d >= 0; d--)
            {
                destFlat += destIndices[d] * stride;
                stride *= destination.Shape[d];
            }

            result.SetFlat(destFlat, source.GetFlat(flatIdx));
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorWhere<T>(Tensor<T> condition, Tensor<T> x, Tensor<T> y)
    {
        if (condition == null) throw new ArgumentNullException(nameof(condition));
        if (x == null) throw new ArgumentNullException(nameof(x));
        if (y == null) throw new ArgumentNullException(nameof(y));

        // All tensors must have the same shape (or be broadcastable, but we'll require same shape for simplicity)
        if (!condition.Shape.SequenceEqual(x.Shape) || !condition.Shape.SequenceEqual(y.Shape))
            throw new ArgumentException("All tensors must have the same shape");

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(condition.Shape);
        int totalElements = condition.Shape.Aggregate(1, (a, b) => a * b);

        Parallel.For(0, totalElements, i =>
        {
            T condVal = condition.GetFlat(i);
            // Condition is true if not equal to zero
            bool isTrue = !numOps.Equals(condVal, numOps.Zero);
            result.SetFlat(i, isTrue ? x.GetFlat(i) : y.GetFlat(i));
        });

        return result;
    }

    #endregion

    #region Loop Elimination Operations

    /// <inheritdoc/>
    public void TensorCopy<T>(Tensor<T> source, Tensor<T> destination)
    {
        if (source == null) throw new ArgumentNullException(nameof(source));
        if (destination == null) throw new ArgumentNullException(nameof(destination));
        if (source.Length != destination.Length)
            throw new ArgumentException($"Tensor lengths must match. Got {source.Length} and {destination.Length}");

        var sourceArray = source.ToArray();
        var destArray = destination.ToArray();
        Array.Copy(sourceArray, destArray, sourceArray.Length);
    }

    /// <inheritdoc/>
    public void TensorFill<T>(Tensor<T> tensor, T value)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        tensor.Fill(value);
    }

    /// <inheritdoc/>
    public Tensor<T> TensorOuterProduct<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        // Flatten both tensors to 1D
        int n = a.Length;
        int m = b.Length;
        var result = new Tensor<T>([n, m]);
        var numOps = MathHelper.GetNumericOperations<T>();

        Parallel.For(0, n, i =>
        {
            T ai = a.GetFlat(i);
            for (int j = 0; j < m; j++)
            {
                result[i, j] = numOps.Multiply(ai, b.GetFlat(j));
            }
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorBatchOuterProduct<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Shape.Length != 2 || b.Shape.Length != 2)
            throw new ArgumentException("Both tensors must be 2D [batch, features]");
        if (a.Shape[0] != b.Shape[0])
            throw new ArgumentException("Batch sizes must match");

        int batch = a.Shape[0];
        int n = a.Shape[1];
        int m = b.Shape[1];
        var result = new Tensor<T>([batch, n, m]);
        var numOps = MathHelper.GetNumericOperations<T>();

        Parallel.For(0, batch, bIdx =>
        {
            for (int i = 0; i < n; i++)
            {
                T ai = a[bIdx, i];
                for (int j = 0; j < m; j++)
                {
                    result[bIdx, i, j] = numOps.Multiply(ai, b[bIdx, j]);
                }
            }
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorPermute<T>(Tensor<T> tensor, int[] axes)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (axes == null) throw new ArgumentNullException(nameof(axes));
        if (axes.Length != tensor.Shape.Length)
            throw new ArgumentException("Axes length must match tensor rank");

        // Use tensor's built-in Transpose method
        return tensor.Transpose(axes);
    }

    /// <inheritdoc/>
    public Tensor<T> TensorExpandDims<T>(Tensor<T> tensor, int axis)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        // Normalize negative axis
        int rank = tensor.Shape.Length;
        if (axis < 0) axis = rank + 1 + axis;
        if (axis < 0 || axis > rank)
            throw new ArgumentOutOfRangeException(nameof(axis), "Axis out of range");

        // Build new shape with 1 inserted at axis
        var newShape = new int[rank + 1];
        for (int i = 0; i < axis; i++)
            newShape[i] = tensor.Shape[i];
        newShape[axis] = 1;
        for (int i = axis; i < rank; i++)
            newShape[i + 1] = tensor.Shape[i];

        return tensor.Reshape(newShape);
    }

    /// <inheritdoc/>
    public Tensor<T> TensorSqueeze<T>(Tensor<T> tensor, int axis = -1)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var shape = tensor.Shape.ToList();

        if (axis == -1)
        {
            // Remove all singleton dimensions
            shape = shape.Where(s => s != 1).ToList();
            if (shape.Count == 0) shape.Add(1); // Keep at least one dimension
        }
        else
        {
            // Normalize axis
            if (axis < 0) axis = shape.Count + axis;
            if (axis < 0 || axis >= shape.Count)
                throw new ArgumentOutOfRangeException(nameof(axis));
            if (shape[axis] != 1)
                throw new ArgumentException($"Cannot squeeze axis {axis} with size {shape[axis]} (must be 1)");
            shape.RemoveAt(axis);
            if (shape.Count == 0) shape.Add(1);
        }

        return tensor.Reshape(shape.ToArray());
    }

    /// <inheritdoc/>
    public Tensor<T> TensorScatterAdd<T>(Tensor<T> destination, Tensor<int> indices, Tensor<T> updates, int axis = 0)
    {
        if (destination == null) throw new ArgumentNullException(nameof(destination));
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        if (updates == null) throw new ArgumentNullException(nameof(updates));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(destination.Shape);
        TensorCopy(destination, result);

        // Simple 1D scatter-add for now (most common use case: embeddings)
        if (axis == 0 && destination.Shape.Length == 2)
        {
            int embeddingDim = destination.Shape[1];
            for (int i = 0; i < indices.Length; i++)
            {
                int idx = indices.GetFlat(i);
                if (idx >= 0 && idx < destination.Shape[0])
                {
                    for (int j = 0; j < embeddingDim; j++)
                    {
                        result[idx, j] = numOps.Add(result[idx, j], updates[i, j]);
                    }
                }
            }
        }
        else
        {
            throw new NotImplementedException("Scatter-add only implemented for axis=0 with 2D destination");
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorGather<T>(Tensor<T> source, Tensor<int> indices, int axis = 0)
    {
        if (source == null) throw new ArgumentNullException(nameof(source));
        if (indices == null) throw new ArgumentNullException(nameof(indices));

        var numOps = MathHelper.GetNumericOperations<T>();

        // Simple 1D gather for embedding lookups
        if (axis == 0 && source.Shape.Length == 2)
        {
            int embeddingDim = source.Shape[1];
            int numIndices = indices.Length;
            var result = new Tensor<T>([numIndices, embeddingDim]);

            Parallel.For(0, numIndices, i =>
            {
                int idx = indices.GetFlat(i);
                if (idx >= 0 && idx < source.Shape[0])
                {
                    for (int j = 0; j < embeddingDim; j++)
                    {
                        result[i, j] = source[idx, j];
                    }
                }
            });

            return result;
        }
        else
        {
            throw new NotImplementedException("Gather only implemented for axis=0 with 2D source");
        }
    }

    /// <inheritdoc/>
    public Tensor<T> TensorCumSum<T>(Tensor<T> tensor, int axis)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(tensor.Shape);

        // Normalize axis
        if (axis < 0) axis = tensor.Shape.Length + axis;

        int outerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= tensor.Shape[i];

        int axisSize = tensor.Shape[axis];

        int innerSize = 1;
        for (int i = axis + 1; i < tensor.Shape.Length; i++) innerSize *= tensor.Shape[i];

        Parallel.For(0, outerSize * innerSize, flatIdx =>
        {
            int outer = flatIdx / innerSize;
            int inner = flatIdx % innerSize;

            T cumSum = numOps.Zero;
            for (int a = 0; a < axisSize; a++)
            {
                int srcIdx = (outer * axisSize + a) * innerSize + inner;
                cumSum = numOps.Add(cumSum, tensor.GetFlat(srcIdx));
                result.SetFlat(srcIdx, cumSum);
            }
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorLogSumExp<T>(Tensor<T> tensor, int axis, bool keepDims = false)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();

        // Normalize axis
        if (axis < 0) axis = tensor.Shape.Length + axis;

        // Compute max along axis for numerical stability
        var maxVals = ReduceMax(tensor, new[] { axis }, keepDims: true, out _);

        // Compute exp(x - max)
        var shifted = TensorSubtract(tensor, maxVals);
        var expShifted = TensorExp(shifted);

        // Sum along axis
        var sumExp = ReduceSum(expShifted, new[] { axis }, keepDims: keepDims);

        // log(sum) + max
        var logSum = TensorLog(sumExp);

        if (keepDims)
        {
            return TensorAdd(logSum, maxVals);
        }
        else
        {
            var maxSqueezed = TensorSqueeze(maxVals, axis);
            return TensorAdd(logSum, maxSqueezed);
        }
    }

    /// <inheritdoc/>
    public Tensor<T> TensorRandomUniform<T>(int[] shape)
    {
        if (shape == null) throw new ArgumentNullException(nameof(shape));
        return Tensor<T>.CreateRandom(shape);
    }

    /// <inheritdoc/>
    public Tensor<T> TensorRandomNormal<T>(int[] shape, T mean, T stddev)
    {
        if (shape == null) throw new ArgumentNullException(nameof(shape));

        var numOps = MathHelper.GetNumericOperations<T>();
        var random = RandomHelper.ThreadSafeRandom;
        var result = new Tensor<T>(shape);
        int totalElements = shape.Aggregate(1, (a, b) => a * b);

        // Box-Muller transform for normal distribution
        for (int i = 0; i < totalElements; i += 2)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double mag = Math.Sqrt(-2.0 * Math.Log(u1));
            double z0 = mag * Math.Cos(2.0 * Math.PI * u2);
            double z1 = mag * Math.Sin(2.0 * Math.PI * u2);

            double meanD = numOps.ToDouble(mean);
            double stdD = numOps.ToDouble(stddev);

            result.SetFlat(i, numOps.FromDouble(z0 * stdD + meanD));
            if (i + 1 < totalElements)
                result.SetFlat(i + 1, numOps.FromDouble(z1 * stdD + meanD));
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorEye<T>(int size)
    {
        if (size <= 0) throw new ArgumentException("Size must be positive", nameof(size));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>([size, size]);
        result.Fill(numOps.Zero);

        for (int i = 0; i < size; i++)
        {
            result[i, i] = numOps.One;
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorDiag<T>(Tensor<T> diagonal)
    {
        if (diagonal == null) throw new ArgumentNullException(nameof(diagonal));

        var numOps = MathHelper.GetNumericOperations<T>();
        int n = diagonal.Length;
        var result = new Tensor<T>([n, n]);
        result.Fill(numOps.Zero);

        for (int i = 0; i < n; i++)
        {
            result[i, i] = diagonal.GetFlat(i);
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorDiagonal<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Shape.Length != 2)
            throw new ArgumentException("Tensor must be 2D");

        int n = Math.Min(tensor.Shape[0], tensor.Shape[1]);
        var result = new Tensor<T>([n]);

        for (int i = 0; i < n; i++)
        {
            result[i] = tensor[i, i];
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorEinsum<T>(string subscripts, params Tensor<T>[] tensors)
    {
        if (subscripts == null) throw new ArgumentNullException(nameof(subscripts));
        if (tensors == null || tensors.Length == 0) throw new ArgumentNullException(nameof(tensors));

        var numOps = MathHelper.GetNumericOperations<T>();

        // Parse einsum notation
        var parts = subscripts.Replace(" ", "").Split(new[] { "->" }, StringSplitOptions.None);
        if (parts.Length != 2)
            throw new ArgumentException("Einsum subscripts must contain '->'");

        var inputSubscripts = parts[0].Split(new[] { ',' }, StringSplitOptions.None);
        var outputSubscripts = parts[1];

        if (inputSubscripts.Length != tensors.Length)
            throw new ArgumentException($"Expected {inputSubscripts.Length} tensors but got {tensors.Length}");

        // Handle common cases directly for efficiency
        if (tensors.Length == 2)
        {
            // Batched matrix multiplication: bij,bjk->bik
            if (subscripts == "bij,bjk->bik")
            {
                return BatchMatMul(tensors[0], tensors[1]);
            }
            // Matrix multiplication: ij,jk->ik
            if (subscripts == "ij,jk->ik")
            {
                return TensorMatMul(tensors[0], tensors[1]);
            }
            // Batched outer product: bi,bj->bij
            if (subscripts == "bi,bj->bij")
            {
                return TensorBatchOuterProduct(tensors[0], tensors[1]);
            }
            // Outer product: i,j->ij
            if (subscripts == "i,j->ij")
            {
                return TensorOuterProduct(tensors[0], tensors[1]);
            }
            // Batched dot: bi,bi->b
            if (subscripts == "bi,bi->b")
            {
                int batch = tensors[0].Shape[0];
                int n = tensors[0].Shape[1];
                var result = new Tensor<T>([batch]);
                Parallel.For(0, batch, b =>
                {
                    T sum = numOps.Zero;
                    for (int i = 0; i < n; i++)
                    {
                        sum = numOps.Add(sum, numOps.Multiply(tensors[0][b, i], tensors[1][b, i]));
                    }
                    result[b] = sum;
                });
                return result;
            }
        }

        // General einsum implementation is complex - throw for unsupported patterns
        throw new NotImplementedException($"Einsum pattern '{subscripts}' not implemented. Use specific tensor operations instead.");
    }

    /// <inheritdoc/>
    public Tensor<T> TensorAddScalar<T>(Tensor<T> tensor, T scalar)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(tensor.Shape);
        int totalElements = tensor.Length;

        Parallel.For(0, totalElements, i =>
        {
            result.SetFlat(i, numOps.Add(tensor.GetFlat(i), scalar));
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorSubtractScalar<T>(Tensor<T> tensor, T scalar)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(tensor.Shape);
        int totalElements = tensor.Length;

        Parallel.For(0, totalElements, i =>
        {
            result.SetFlat(i, numOps.Subtract(tensor.GetFlat(i), scalar));
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorDivideScalar<T>(Tensor<T> tensor, T scalar)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(tensor.Shape);
        int totalElements = tensor.Length;

        Parallel.For(0, totalElements, i =>
        {
            result.SetFlat(i, numOps.Divide(tensor.GetFlat(i), scalar));
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TanhDerivative<T>(Tensor<T> tanhOutput)
    {
        if (tanhOutput == null) throw new ArgumentNullException(nameof(tanhOutput));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(tanhOutput.Shape);
        int totalElements = tanhOutput.Length;

        // d/dx tanh(x) = 1 - tanh(x)^2, and we have tanhOutput = tanh(x)
        Parallel.For(0, totalElements, i =>
        {
            T y = tanhOutput.GetFlat(i);
            T y2 = numOps.Multiply(y, y);
            result.SetFlat(i, numOps.Subtract(numOps.One, y2));
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> SigmoidDerivative<T>(Tensor<T> sigmoidOutput)
    {
        if (sigmoidOutput == null) throw new ArgumentNullException(nameof(sigmoidOutput));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(sigmoidOutput.Shape);
        int totalElements = sigmoidOutput.Length;

        // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)), and we have sigmoidOutput = sigmoid(x)
        Parallel.For(0, totalElements, i =>
        {
            T y = sigmoidOutput.GetFlat(i);
            T oneMinusY = numOps.Subtract(numOps.One, y);
            result.SetFlat(i, numOps.Multiply(y, oneMinusY));
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> ReLUDerivative<T>(Tensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(input.Shape);
        int totalElements = input.Length;

        Parallel.For(0, totalElements, i =>
        {
            T val = input.GetFlat(i);
            bool positive = numOps.ToDouble(val) > 0;
            result.SetFlat(i, positive ? numOps.One : numOps.Zero);
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorTriangularMask<T>(int size, bool upper = false, int diagonal = 0)
    {
        if (size <= 0) throw new ArgumentException("Size must be positive", nameof(size));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>([size, size]);

        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                bool inMask = upper ? (j >= i + diagonal) : (j <= i + diagonal);
                result[i, j] = inMask ? numOps.One : numOps.Zero;
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorSquash<T>(Tensor<T> tensor, int axis = -1)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();

        // Normalize axis
        if (axis < 0) axis = tensor.Shape.Length + axis;

        // Compute squared norm along axis
        var squared = TensorMultiply(tensor, tensor);
        var normSquared = ReduceSum(squared, new[] { axis }, keepDims: true);

        // Compute scale factor: ||v||^2 / (1 + ||v||^2)
        var one = new Tensor<T>(normSquared.Shape);
        one.Fill(numOps.One);
        var denom = TensorAdd(one, normSquared);
        var scale = TensorDivide(normSquared, denom);

        // Compute ||v||
        var norm = TensorSqrt(normSquared);
        var epsilon = new Tensor<T>(norm.Shape);
        epsilon.Fill(numOps.FromDouble(1e-8));
        norm = TensorAdd(norm, epsilon);

        // Normalize: v / ||v||
        var normalized = TensorDivide(tensor, norm);

        // Apply scale
        return TensorMultiply(scale, normalized);
    }

    /// <inheritdoc/>
    public Tensor<T> TensorSquashBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> output, int axis = -1)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (output == null) throw new ArgumentNullException(nameof(output));

        var numOps = MathHelper.GetNumericOperations<T>();

        // Normalize axis
        if (axis < 0) axis = input.Shape.Length + axis;

        // This is a simplified gradient - full implementation would require proper Jacobian
        // For now, approximate with element-wise gradient scaling
        var squared = TensorMultiply(input, input);
        var normSquared = ReduceSum(squared, new[] { axis }, keepDims: true);
        var one = new Tensor<T>(normSquared.Shape);
        one.Fill(numOps.One);
        var denom = TensorAdd(one, normSquared);
        var scale = TensorDivide(one, denom);

        return TensorMultiply(gradOutput, scale);
    }

    /// <inheritdoc/>
    public Tensor<T> TensorNorm<T>(Tensor<T> tensor, int axis, bool keepDims = false)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        // Compute squared values
        var squared = TensorMultiply(tensor, tensor);
        // Sum along axis
        var sumSquared = ReduceSum(squared, new[] { axis }, keepDims: keepDims);
        // Square root
        return TensorSqrt(sumSquared);
    }

    /// <inheritdoc/>
    public Tensor<T> TensorNormalize<T>(Tensor<T> tensor, int axis, T epsilon)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        // Compute norm
        var norm = TensorNorm(tensor, axis, keepDims: true);

        // Add epsilon for numerical stability
        var epsArray = new Tensor<T>(norm.Shape);
        epsArray.Fill(epsilon);
        norm = TensorAdd(norm, epsArray);

        // Divide
        return TensorDivide(tensor, norm);
    }

    /// <inheritdoc/>
    public Tensor<T> TensorClip<T>(Tensor<T> tensor, T minValue, T maxValue)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        return TensorClamp(tensor, minValue, maxValue);
    }

    /// <inheritdoc/>
    public Tensor<T> TensorConcatenate<T>(Tensor<T>[] tensors, int axis = 0)
    {
        if (tensors == null || tensors.Length == 0)
            throw new ArgumentNullException(nameof(tensors));

        return Tensor<T>.Concatenate(tensors, axis);
    }

    /// <inheritdoc/>
    public Tensor<T>[] TensorSplit<T>(Tensor<T> tensor, int numSplits, int axis = 0)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (numSplits <= 0) throw new ArgumentException("Number of splits must be positive", nameof(numSplits));

        // Normalize axis
        if (axis < 0) axis = tensor.Shape.Length + axis;

        int axisSize = tensor.Shape[axis];
        if (axisSize % numSplits != 0)
            throw new ArgumentException($"Cannot split axis of size {axisSize} into {numSplits} equal parts");

        int splitSize = axisSize / numSplits;
        var results = new Tensor<T>[numSplits];

        for (int i = 0; i < numSplits; i++)
        {
            int start = i * splitSize;
            results[i] = tensor.Slice(axis, start, start + splitSize);
        }

        return results;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorOneHot<T>(Tensor<int> indices, int depth)
    {
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        if (depth <= 0) throw new ArgumentException("Depth must be positive", nameof(depth));

        var numOps = MathHelper.GetNumericOperations<T>();
        int numIndices = indices.Length;
        var result = new Tensor<T>([numIndices, depth]);
        result.Fill(numOps.Zero);

        for (int i = 0; i < numIndices; i++)
        {
            int idx = indices.GetFlat(i);
            if (idx >= 0 && idx < depth)
            {
                result[i, idx] = numOps.One;
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<int> TensorArgMax<T>(Tensor<T> tensor, int axis)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();

        // Normalize axis
        if (axis < 0) axis = tensor.Shape.Length + axis;

        // Build output shape (remove axis dimension)
        var outputShape = tensor.Shape.Where((_, i) => i != axis).ToArray();
        if (outputShape.Length == 0) outputShape = new[] { 1 };
        var result = new Tensor<int>(outputShape);

        int outerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= tensor.Shape[i];

        int axisSize = tensor.Shape[axis];

        int innerSize = 1;
        for (int i = axis + 1; i < tensor.Shape.Length; i++) innerSize *= tensor.Shape[i];

        Parallel.For(0, outerSize * innerSize, flatIdx =>
        {
            int outer = flatIdx / innerSize;
            int inner = flatIdx % innerSize;

            T maxVal = tensor.GetFlat((outer * axisSize) * innerSize + inner);
            int maxIdx = 0;

            for (int a = 1; a < axisSize; a++)
            {
                int srcIdx = (outer * axisSize + a) * innerSize + inner;
                T val = tensor.GetFlat(srcIdx);
                if (numOps.ToDouble(val) > numOps.ToDouble(maxVal))
                {
                    maxVal = val;
                    maxIdx = a;
                }
            }

            result.SetFlat(flatIdx, maxIdx);
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<int> TensorArgMin<T>(Tensor<T> tensor, int axis)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();

        // Normalize axis
        if (axis < 0) axis = tensor.Shape.Length + axis;

        // Build output shape (remove axis dimension)
        var outputShape = tensor.Shape.Where((_, i) => i != axis).ToArray();
        if (outputShape.Length == 0) outputShape = new[] { 1 };
        var result = new Tensor<int>(outputShape);

        int outerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= tensor.Shape[i];

        int axisSize = tensor.Shape[axis];

        int innerSize = 1;
        for (int i = axis + 1; i < tensor.Shape.Length; i++) innerSize *= tensor.Shape[i];

        Parallel.For(0, outerSize * innerSize, flatIdx =>
        {
            int outer = flatIdx / innerSize;
            int inner = flatIdx % innerSize;

            T minVal = tensor.GetFlat((outer * axisSize) * innerSize + inner);
            int minIdx = 0;

            for (int a = 1; a < axisSize; a++)
            {
                int srcIdx = (outer * axisSize + a) * innerSize + inner;
                T val = tensor.GetFlat(srcIdx);
                if (numOps.ToDouble(val) < numOps.ToDouble(minVal))
                {
                    minVal = val;
                    minIdx = a;
                }
            }

            result.SetFlat(flatIdx, minIdx);
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorBinaryCrossEntropy<T>(Tensor<T> predictions, Tensor<T> targets, T epsilon)
    {
        if (predictions == null) throw new ArgumentNullException(nameof(predictions));
        if (targets == null) throw new ArgumentNullException(nameof(targets));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(predictions.Shape);
        int totalElements = predictions.Length;

        Parallel.For(0, totalElements, i =>
        {
            T p = predictions.GetFlat(i);
            T t = targets.GetFlat(i);

            // Clip for numerical stability using comparisons
            T upperBound = numOps.Subtract(numOps.One, epsilon);
            double pVal = numOps.ToDouble(p);
            double upperVal = numOps.ToDouble(upperBound);
            double epsVal = numOps.ToDouble(epsilon);
            T clippedUpper = pVal < upperVal ? p : upperBound;
            double clippedUpperVal = numOps.ToDouble(clippedUpper);
            T pClipped = clippedUpperVal > epsVal ? clippedUpper : epsilon;

            // -[t * log(p) + (1-t) * log(1-p)]
            T logP = numOps.Log(pClipped);
            T log1MinusP = numOps.Log(numOps.Subtract(numOps.One, pClipped));
            T oneMinusT = numOps.Subtract(numOps.One, t);

            T loss = numOps.Negate(numOps.Add(
                numOps.Multiply(t, logP),
                numOps.Multiply(oneMinusT, log1MinusP)));

            result.SetFlat(i, loss);
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorBinaryCrossEntropyBackward<T>(Tensor<T> predictions, Tensor<T> targets, T epsilon)
    {
        if (predictions == null) throw new ArgumentNullException(nameof(predictions));
        if (targets == null) throw new ArgumentNullException(nameof(targets));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(predictions.Shape);
        int totalElements = predictions.Length;

        Parallel.For(0, totalElements, i =>
        {
            T p = predictions.GetFlat(i);
            T t = targets.GetFlat(i);

            // Clip for numerical stability using comparisons
            T upperBound = numOps.Subtract(numOps.One, epsilon);
            double pVal = numOps.ToDouble(p);
            double upperVal = numOps.ToDouble(upperBound);
            double epsVal = numOps.ToDouble(epsilon);
            T clippedUpper = pVal < upperVal ? p : upperBound;
            double clippedUpperVal = numOps.ToDouble(clippedUpper);
            T pClipped = clippedUpperVal > epsVal ? clippedUpper : epsilon;

            // Gradient: -t/p + (1-t)/(1-p)
            T termA = numOps.Divide(t, pClipped);
            T oneMinusP = numOps.Subtract(numOps.One, pClipped);
            T oneMinusT = numOps.Subtract(numOps.One, t);
            T termB = numOps.Divide(oneMinusT, oneMinusP);

            T grad = numOps.Subtract(termB, termA);
            result.SetFlat(i, grad);
        });

        return result;
    }

    /// <inheritdoc/>
    public (Tensor<T> X, Tensor<T> Y) TensorMeshgrid<T>(Tensor<T> x, Tensor<T> y)
    {
        if (x == null) throw new ArgumentNullException(nameof(x));
        if (y == null) throw new ArgumentNullException(nameof(y));

        int width = x.Length;
        int height = y.Length;

        var X = new Tensor<T>([height, width]);
        var Y = new Tensor<T>([height, width]);

        Parallel.For(0, height, row =>
        {
            T yVal = y.GetFlat(row);
            for (int col = 0; col < width; col++)
            {
                X[row, col] = x.GetFlat(col);
                Y[row, col] = yVal;
            }
        });

        return (X, Y);
    }

    /// <inheritdoc/>
    public Tensor<T> TensorSliceAxis<T>(Tensor<T> tensor, int axis, int index)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Rank != 3) throw new ArgumentException("TensorSliceAxis currently only supports 3D tensors.");

        int dim0 = tensor.Shape[0];
        int dim1 = tensor.Shape[1];
        int dim2 = tensor.Shape[2];

        Tensor<T> result;

        switch (axis)
        {
            case 0:
                // Slice along first axis: result[j,k] = tensor[index, j, k]
                result = new Tensor<T>([dim1, dim2]);
                Parallel.For(0, dim1, j =>
                {
                    for (int k = 0; k < dim2; k++)
                    {
                        result[j, k] = tensor[index, j, k];
                    }
                });
                break;

            case 1:
                // Slice along second axis: result[i,k] = tensor[i, index, k]
                result = new Tensor<T>([dim0, dim2]);
                Parallel.For(0, dim0, i =>
                {
                    for (int k = 0; k < dim2; k++)
                    {
                        result[i, k] = tensor[i, index, k];
                    }
                });
                break;

            case 2:
                // Slice along third axis: result[i,j] = tensor[i, j, index]
                result = new Tensor<T>([dim0, dim1]);
                Parallel.For(0, dim0, i =>
                {
                    for (int j = 0; j < dim1; j++)
                    {
                        result[i, j] = tensor[i, j, index];
                    }
                });
                break;

            default:
                throw new ArgumentOutOfRangeException(nameof(axis), "Axis must be 0, 1, or 2 for 3D tensors.");
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorLinspace<T>(T start, T end, int count)
    {
        if (count < 2) throw new ArgumentException("Count must be at least 2.", nameof(count));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>([count]);

        T range = numOps.Subtract(end, start);
        T divisor = numOps.FromDouble(count - 1);
        T step = numOps.Divide(range, divisor);

        Parallel.For(0, count, i =>
        {
            T value = numOps.Add(start, numOps.Multiply(numOps.FromDouble(i), step));
            result.SetFlat(i, value);
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorBatchMatMul<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        // If both tensors are 3D, delegate to BatchMatMul
        if (a.Rank == 3 && b.Rank == 3)
        {
            return BatchMatMul(a, b);
        }

        // Handle broadcasting case where b is 2D [K, N]
        if (a.Rank != 3 || b.Rank != 2)
        {
            throw new ArgumentException(
                $"TensorBatchMatMul requires a to be 3D and b to be 2D or 3D. Got ranks {a.Rank} and {b.Rank}.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();

        // a: [batch, M, K], b: [K, N]
        int batch = a.Shape[0];
        int M = a.Shape[1];
        int K = a.Shape[2];
        int N = b.Shape[1];

        if (b.Shape[0] != K)
        {
            throw new ArgumentException(
                $"Matrix dimensions incompatible. a has shape [{batch}, {M}, {K}], b has shape [{b.Shape[0]}, {N}]. Inner dimensions must match.");
        }

        var result = new Tensor<T>([batch, M, N]);

        Parallel.For(0, batch, batchIdx =>
        {
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    T sum = numOps.Zero;
                    for (int k = 0; k < K; k++)
                    {
                        sum = numOps.Add(sum, numOps.Multiply(a[batchIdx, i, k], b[k, j]));
                    }
                    result[batchIdx, i, j] = sum;
                }
            }
        });

        return result;
    }

    /// <inheritdoc/>
    public void TensorSetSliceAxis<T>(Tensor<T> destination, Tensor<T> source, int axis, int index)
    {
        if (destination == null) throw new ArgumentNullException(nameof(destination));
        if (source == null) throw new ArgumentNullException(nameof(source));

        // For 3D tensors
        if (destination.Rank == 3)
        {
            int dim0 = destination.Shape[0];
            int dim1 = destination.Shape[1];
            int dim2 = destination.Shape[2];

            switch (axis)
            {
                case 0:
                    // Set destination[index, :, :] = source
                    Parallel.For(0, dim1, j =>
                    {
                        for (int k = 0; k < dim2; k++)
                        {
                            destination[index, j, k] = source[j, k];
                        }
                    });
                    break;

                case 1:
                    Parallel.For(0, dim0, i =>
                    {
                        for (int k = 0; k < dim2; k++)
                        {
                            destination[i, index, k] = source[i, k];
                        }
                    });
                    break;

                case 2:
                    Parallel.For(0, dim0, i =>
                    {
                        for (int j = 0; j < dim1; j++)
                        {
                            destination[i, j, index] = source[i, j];
                        }
                    });
                    break;

                default:
                    throw new ArgumentOutOfRangeException(nameof(axis));
            }
        }
        else
        {
            throw new NotSupportedException("TensorSetSliceAxis currently only supports 3D tensors.");
        }
    }

    /// <inheritdoc/>
    public Tensor<T> TensorSoftmax<T>(Tensor<T> tensor, int axis)
    {
        // Delegate to Softmax which has the same functionality
        return Softmax(tensor, axis);
    }

    /// <inheritdoc/>
    public Tensor<T> TensorSoftmaxBackward<T>(Tensor<T> softmaxOutput, Tensor<T> outputGradient, int axis)
    {
        // Delegate to SoftmaxBackward with reordered parameters (grad first, then output)
        return SoftmaxBackward(outputGradient, softmaxOutput, axis);
    }

    /// <inheritdoc/>
    public Tensor<T> TensorLogSoftmax<T>(Tensor<T> tensor, int axis)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        // log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
        // More numerically stable than log(softmax(x))

        if (axis < 0) axis = tensor.Rank + axis;

        var maxValues = ReduceMax(tensor, new[] { axis }, keepDims: true, out _);
        var shifted = TensorSubtract(tensor, maxValues);
        var expValues = TensorExp(shifted);
        var sumExp = ReduceSum(expValues, new[] { axis }, keepDims: true);
        var logSumExp = TensorLog(sumExp);

        return TensorSubtract(shifted, logSumExp);
    }

    /// <inheritdoc/>
    public Tensor<T> TensorTopK<T>(Tensor<T> tensor, int k, int axis, out Tensor<int> indices)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (k <= 0) throw new ArgumentException("k must be positive.", nameof(k));

        var numOps = MathHelper.GetNumericOperations<T>();
        if (axis < 0) axis = tensor.Rank + axis;

        // For simplicity, handle 2D case: [batch, features]
        if (tensor.Rank == 2 && axis == 1)
        {
            int batch = tensor.Shape[0];
            int features = tensor.Shape[1];
            k = Math.Min(k, features);

            var resultValues = new Tensor<T>([batch, k]);
            var indicesResult = new Tensor<int>([batch, k]);

            Parallel.For(0, batch, b =>
            {
                // Extract row and sort with indices
                var rowWithIndices = new (T value, int index)[features];
                for (int i = 0; i < features; i++)
                {
                    rowWithIndices[i] = (tensor[b, i], i);
                }

                // Sort descending by value using GreaterThan
                Array.Sort(rowWithIndices, (a, x) =>
                {
                    if (numOps.GreaterThan(x.value, a.value)) return 1;
                    if (numOps.LessThan(x.value, a.value)) return -1;
                    return 0;
                });

                // Take top-k
                for (int i = 0; i < k; i++)
                {
                    resultValues[b, i] = rowWithIndices[i].value;
                    indicesResult[b, i] = rowWithIndices[i].index;
                }
            });

            indices = indicesResult;
            return resultValues;
        }
        else
        {
            throw new NotSupportedException("TensorTopK currently only supports 2D tensors with axis=1.");
        }
    }

    /// <inheritdoc/>
    public Tensor<T> TensorScatter<T>(Tensor<T> destination, Tensor<int> indices, Tensor<T> source, int axis)
    {
        if (destination == null) throw new ArgumentNullException(nameof(destination));
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        if (source == null) throw new ArgumentNullException(nameof(source));

        var result = destination.Clone();

        // For 2D with axis=1: result[b, indices[b, i]] = source[b, i]
        if (result.Rank == 2 && axis == 1)
        {
            int batch = result.Shape[0];
            int numIndices = indices.Shape[1];

            Parallel.For(0, batch, b =>
            {
                for (int i = 0; i < numIndices; i++)
                {
                    int idx = indices[b, i];
                    result[b, idx] = source[b, i];
                }
            });
        }
        else
        {
            throw new NotSupportedException("TensorScatter currently only supports 2D tensors with axis=1.");
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorIndexSelect<T>(Tensor<T> tensor, Tensor<int> indices, int axis)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (indices == null) throw new ArgumentNullException(nameof(indices));

        if (axis < 0) axis = tensor.Rank + axis;

        // For 2D tensor with axis=0: select rows
        if (tensor.Rank == 2 && axis == 0)
        {
            int numIndices = indices.Length;
            int cols = tensor.Shape[1];
            var result = new Tensor<T>([numIndices, cols]);

            Parallel.For(0, numIndices, i =>
            {
                int idx = indices.GetFlat(i);
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = tensor[idx, j];
                }
            });

            return result;
        }
        else if (tensor.Rank == 2 && axis == 1)
        {
            int rows = tensor.Shape[0];
            int numIndices = indices.Length;
            var result = new Tensor<T>([rows, numIndices]);

            Parallel.For(0, rows, i =>
            {
                for (int j = 0; j < numIndices; j++)
                {
                    int idx = indices.GetFlat(j);
                    result[i, j] = tensor[i, idx];
                }
            });

            return result;
        }
        else
        {
            throw new NotSupportedException("TensorIndexSelect currently only supports 2D tensors.");
        }
    }

    /// <inheritdoc/>
    public Tensor<T> TensorStack<T>(Tensor<T>[] tensors, int axis)
    {
        if (tensors == null || tensors.Length == 0)
            throw new ArgumentException("Tensors array must not be empty.", nameof(tensors));

        int numTensors = tensors.Length;
        var firstShape = tensors[0].Shape;

        // Validate all tensors have same shape
        for (int i = 1; i < numTensors; i++)
        {
            if (!tensors[i].Shape.SequenceEqual(firstShape))
                throw new ArgumentException("All tensors must have the same shape.");
        }

        if (axis < 0) axis = firstShape.Length + 1 + axis;

        // New shape: insert numTensors at axis position
        var newShape = new int[firstShape.Length + 1];
        for (int i = 0; i < axis; i++) newShape[i] = firstShape[i];
        newShape[axis] = numTensors;
        for (int i = axis; i < firstShape.Length; i++) newShape[i + 1] = firstShape[i];

        var result = new Tensor<T>(newShape);

        // Copy each tensor
        Parallel.For(0, numTensors, t =>
        {
            var tensor = tensors[t];
            int tensorSize = tensor.Length;
            int sliceSize = 1;
            for (int i = axis + 1; i < newShape.Length; i++) sliceSize *= newShape[i];

            int outerSize = 1;
            for (int i = 0; i < axis; i++) outerSize *= newShape[i];

            for (int outer = 0; outer < outerSize; outer++)
            {
                int srcOffset = outer * sliceSize;
                int dstOffset = (outer * numTensors + t) * sliceSize;
                for (int inner = 0; inner < sliceSize; inner++)
                {
                    result.SetFlat(dstOffset + inner, tensor.GetFlat(srcOffset + inner));
                }
            }
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T>[] TensorUnstack<T>(Tensor<T> tensor, int axis)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        if (axis < 0) axis = tensor.Rank + axis;

        int numSlices = tensor.Shape[axis];
        var result = new Tensor<T>[numSlices];

        // New shape: remove the axis dimension
        var newShape = new int[tensor.Rank - 1];
        for (int i = 0, j = 0; i < tensor.Rank; i++)
        {
            if (i != axis) newShape[j++] = tensor.Shape[i];
        }

        Parallel.For(0, numSlices, i =>
        {
            result[i] = TensorSliceAxis(tensor, axis, i);
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorMap<T>(Tensor<T> tensor, Func<T, T> func)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (func == null) throw new ArgumentNullException(nameof(func));

        var result = new Tensor<T>(tensor.Shape);

        Parallel.For(0, tensor.Length, i =>
        {
            result.SetFlat(i, func(tensor.GetFlat(i)));
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorMaskedFill<T>(Tensor<T> tensor, Tensor<bool> mask, T value)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (mask == null) throw new ArgumentNullException(nameof(mask));

        var result = tensor.Clone();

        Parallel.For(0, tensor.Length, i =>
        {
            if (mask.GetFlat(i))
            {
                result.SetFlat(i, value);
            }
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorWhere<T>(Tensor<bool> condition, Tensor<T> x, Tensor<T> y)
    {
        if (condition == null) throw new ArgumentNullException(nameof(condition));
        if (x == null) throw new ArgumentNullException(nameof(x));
        if (y == null) throw new ArgumentNullException(nameof(y));

        var result = new Tensor<T>(x.Shape);

        Parallel.For(0, x.Length, i =>
        {
            result.SetFlat(i, condition.GetFlat(i) ? x.GetFlat(i) : y.GetFlat(i));
        });

        return result;
    }

    #endregion

    #region Neural Radiance Fields Operations

    /// <inheritdoc/>
    public Tensor<T> PositionalEncoding<T>(Tensor<T> positions, int numFrequencies)
    {
        return NeRFOperations.PositionalEncoding(positions, numFrequencies);
    }

    /// <inheritdoc/>
    public Tensor<T> PositionalEncodingBackward<T>(Tensor<T> positions, Tensor<T> encodedGradient, int numFrequencies)
    {
        return NeRFOperations.PositionalEncodingBackward(positions, encodedGradient, numFrequencies);
    }

    /// <inheritdoc/>
    public Tensor<T> VolumeRendering<T>(Tensor<T> rgbSamples, Tensor<T> densitySamples, Tensor<T> tValues)
    {
        return NeRFOperations.VolumeRendering(rgbSamples, densitySamples, tValues);
    }

    /// <inheritdoc/>
    public void VolumeRenderingBackward<T>(
        Tensor<T> rgbSamples,
        Tensor<T> densitySamples,
        Tensor<T> tValues,
        Tensor<T> outputGradient,
        out Tensor<T> rgbGradient,
        out Tensor<T> densityGradient)
    {
        NeRFOperations.VolumeRenderingBackward(
            rgbSamples, densitySamples, tValues, outputGradient,
            out rgbGradient, out densityGradient);
    }

    /// <inheritdoc/>
    public (Tensor<T> positions, Tensor<T> directions, Tensor<T> tValues) SampleRayPoints<T>(
        Tensor<T> rayOrigins,
        Tensor<T> rayDirections,
        T nearBound,
        T farBound,
        int numSamples,
        bool stratified = true)
    {
        return NeRFOperations.SampleRayPoints(
            rayOrigins, rayDirections, nearBound, farBound, numSamples, stratified);
    }

    /// <inheritdoc/>
    public Tensor<T> ImportanceSampling<T>(Tensor<T> tValuesCoarse, Tensor<T> weightsCoarse, int numFineSamples)
    {
        return NeRFOperations.ImportanceSampling(tValuesCoarse, weightsCoarse, numFineSamples);
    }

    /// <inheritdoc/>
    public (Tensor<T> origins, Tensor<T> directions) GenerateCameraRays<T>(
        Vector<T> cameraPosition,
        Matrix<T> cameraRotation,
        int imageWidth,
        int imageHeight,
        T focalLength)
    {
        return NeRFOperations.GenerateCameraRays(
            cameraPosition, cameraRotation, imageWidth, imageHeight, focalLength);
    }

    #endregion

    #region Gaussian Splatting Operations

    /// <inheritdoc/>
    public void ProjectGaussians3DTo2D<T>(
        Tensor<T> means3D,
        Tensor<T> covariances3D,
        Matrix<T> viewMatrix,
        Matrix<T> projMatrix,
        int imageWidth,
        int imageHeight,
        out Tensor<T> means2D,
        out Tensor<T> covariances2D,
        out Tensor<T> depths,
        out Tensor<bool> visible)
    {
        GaussianSplattingOperations.ProjectGaussians3DTo2D(
            means3D, covariances3D, viewMatrix, projMatrix,
            imageWidth, imageHeight,
            out means2D, out covariances2D, out depths, out visible);
    }

    /// <inheritdoc/>
    public Tensor<T> RasterizeGaussians<T>(
        Tensor<T> means2D,
        Tensor<T> covariances2D,
        Tensor<T> colors,
        Tensor<T> opacities,
        Tensor<T> depths,
        int imageWidth,
        int imageHeight,
        int tileSize = 16)
    {
        return GaussianSplattingOperations.RasterizeGaussians(
            means2D, covariances2D, colors, opacities, depths,
            imageWidth, imageHeight, tileSize);
    }

    /// <inheritdoc/>
    public void RasterizeGaussiansBackward<T>(
        Tensor<T> means2D,
        Tensor<T> covariances2D,
        Tensor<T> colors,
        Tensor<T> opacities,
        Tensor<T> depths,
        int imageWidth,
        int imageHeight,
        Tensor<T> outputGradient,
        int tileSize,
        out Tensor<T> means2DGrad,
        out Tensor<T> covariances2DGrad,
        out Tensor<T> colorsGrad,
        out Tensor<T> opacitiesGrad)
    {
        GaussianSplattingOperations.RasterizeGaussiansBackward(
            means2D, covariances2D, colors, opacities, depths,
            imageWidth, imageHeight, outputGradient, tileSize,
            out means2DGrad, out covariances2DGrad, out colorsGrad, out opacitiesGrad);
    }

    /// <inheritdoc/>
    public Tensor<T> EvaluateSphericalHarmonics<T>(Tensor<T> shCoefficients, Tensor<T> viewDirections, int degree)
    {
        return GaussianSplattingOperations.EvaluateSphericalHarmonics(shCoefficients, viewDirections, degree);
    }

    /// <inheritdoc/>
    public Tensor<T> EvaluateSphericalHarmonicsBackward<T>(
        Tensor<T> shCoefficients,
        Tensor<T> viewDirections,
        int degree,
        Tensor<T> outputGradient)
    {
        return GaussianSplattingOperations.EvaluateSphericalHarmonicsBackward(
            shCoefficients, viewDirections, degree, outputGradient);
    }

    /// <inheritdoc/>
    public Tensor<T> ComputeGaussianCovariance<T>(Tensor<T> rotations, Tensor<T> scales)
    {
        return GaussianSplattingOperations.ComputeGaussianCovariance(rotations, scales);
    }

    /// <inheritdoc/>
    public void ComputeGaussianCovarianceBackward<T>(
        Tensor<T> rotations,
        Tensor<T> scales,
        Tensor<T> covarianceGradient,
        out Tensor<T> rotationsGrad,
        out Tensor<T> scalesGrad)
    {
        GaussianSplattingOperations.ComputeGaussianCovarianceBackward(
            rotations, scales, covarianceGradient,
            out rotationsGrad, out scalesGrad);
    }

    #endregion

    #region Instant-NGP Operations

    /// <inheritdoc/>
    public Tensor<T> MultiresolutionHashEncoding<T>(
        Tensor<T> positions,
        Tensor<T>[] hashTables,
        int[] resolutions,
        int featuresPerLevel)
    {
        return InstantNGPOperations.MultiresolutionHashEncoding(
            positions, hashTables, resolutions, featuresPerLevel);
    }

    /// <inheritdoc/>
    public Tensor<T>[] MultiresolutionHashEncodingBackward<T>(
        Tensor<T> positions,
        Tensor<T>[] hashTables,
        int[] resolutions,
        int featuresPerLevel,
        Tensor<T> outputGradient)
    {
        return InstantNGPOperations.MultiresolutionHashEncodingBackward(
            positions, hashTables, resolutions, featuresPerLevel, outputGradient);
    }

    /// <inheritdoc/>
    public Tensor<T> UpdateOccupancyGrid<T>(
        Tensor<T> occupancyGrid,
        Tensor<T> densities,
        Tensor<T> positions,
        int gridSize,
        T threshold,
        T decayFactor)
    {
        return InstantNGPOperations.UpdateOccupancyGrid(
            occupancyGrid, densities, positions, gridSize, threshold, decayFactor);
    }

    /// <inheritdoc/>
    public (Tensor<T> positions, Tensor<T> directions, Tensor<bool> validMask, Tensor<T> tValues) SampleRaysWithOccupancy<T>(
        Tensor<T> rayOrigins,
        Tensor<T> rayDirections,
        uint[] occupancyBitfield,
        int gridSize,
        Vector<T> sceneBoundsMin,
        Vector<T> sceneBoundsMax,
        T nearBound,
        T farBound,
        int maxSamples)
    {
        return InstantNGPOperations.SampleRaysWithOccupancy(
            rayOrigins, rayDirections, occupancyBitfield, gridSize,
            sceneBoundsMin, sceneBoundsMax, nearBound, farBound, maxSamples);
    }

    #endregion
}
