using System;
using System.Runtime.CompilerServices;
using AiDotNet.Helpers;
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
            double val = Convert.ToDouble(vector[i]);
#if NET5_0_OR_GREATER
            result[i] = numOps.FromDouble(Math.Atanh(val));
#else
            result[i] = numOps.FromDouble(0.5 * Math.Log((1.0 + val) / (1.0 - val)));
#endif
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
            result[i] = numOps.Add(a[i], b[i]);
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
            result[i] = numOps.Subtract(a[i], b[i]);
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
            result[i] = numOps.Multiply(a[i], b[i]);
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
            result[i] = numOps.Multiply(tensor[i], scalar);
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
            if (numOps.Equals(b[i], numOps.Zero))
            {
                throw new DivideByZeroException($"Division by zero at index {i}");
            }

            result[i] = numOps.Divide(a[i], b[i]);
        }

        return result;
    }

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
        if (stride == null || stride.Length != 2) throw new ArgumentException("Stride must be array of 2 elements");
        if (padding == null || padding.Length != 2) throw new ArgumentException("Padding must be array of 2 elements");
        if (dilation == null || dilation.Length != 2) throw new ArgumentException("Dilation must be array of 2 elements");

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
                                    lock (gradInput)
                                    {
                                        gradInput[gradInputIdx] = numOps.Add(gradInput[gradInputIdx], numOps.Multiply(gradVal, kernelData[kernelIdx]));
                                    }
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

        int outputHeight = (height - poolH) / strideH + 1;
        int outputWidth = (width - poolW) / strideW + 1;

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

        int outputHeight = (height - poolH) / strideH + 1;
        int outputWidth = (width - poolW) / strideW + 1;

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

        Parallel.For(0, batch * inChannels, idx =>
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
                                    lock (outputData)
                                    {
                                        outputData[outputIdx] = numOps.Add(outputData[outputIdx], numOps.Multiply(inputVal, kernelData[kernelIdx]));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

        return new Tensor<T>([batch, outChannels, outputHeight, outputWidth], new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public Tensor<T> ConvTranspose2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding)
    {
        // ConvTranspose2D backward w.r.t. input is equivalent to Conv2D forward
        return Conv2D(gradOutput, kernel, stride, padding, [1, 1]);
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
        Parallel.For(0, features, f =>
        {
            T invStd = numOps.Divide(numOps.One, numOps.Sqrt(numOps.Add(varData[f], eps)));
            T sumGrad = numOps.Zero;
            T sumGradX = numOps.Zero;

            for (int b = 0; b < batch; b++)
            {
                int idx = b * features + f;
                sumGrad = numOps.Add(sumGrad, gradOutputData[idx]);
                sumGradX = numOps.Add(sumGradX, numOps.Multiply(gradOutputData[idx], numOps.Subtract(inputData[idx], meanData[f])));
            }

            for (int b = 0; b < batch; b++)
            {
                int idx = b * features + f;
                T normalized = numOps.Multiply(numOps.Subtract(inputData[idx], meanData[f]), invStd);
                T gradNorm = numOps.Multiply(gammaData[f], gradOutputData[idx]);
                T term1 = numOps.Multiply(batchT, gradNorm);
                T term2 = sumGrad;
                T term3 = numOps.Multiply(normalized, numOps.Multiply(invStd, sumGradX));
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

    /// <inheritdoc/>
    public Tensor<T> ReduceMax<T>(Tensor<T> input, int[] axes, bool keepDims, out int[] maxIndices)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputShape = input.Shape;
        var inputData = input.ToArray();

        // Normalize axes
        var normalizedAxes = axes.Select(a => a < 0 ? inputShape.Length + a : a).OrderBy(a => a).ToArray();

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

        var normalizedAxes = axes.Select(a => a < 0 ? inputShape.Length + a : a).OrderBy(a => a).ToArray();

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
        var numOps = MathHelper.GetNumericOperations<T>();
        int inputSize = inputShape.Aggregate(1, (a, b) => a * b);
        var gradInputData = new T[inputSize];

        var normalizedAxes = axes.Select(a => a < 0 ? inputShape.Length + a : a).ToArray();

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

            int outputIdx = Math.Min(MultiToFlatIndex([.. outputMultiIndex], gradOutputShape, outputStrides), gradOutputData.Length - 1);
            gradInputData[i] = numOps.Multiply(gradOutputData[outputIdx], scale);
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

                    lock (gradInputData)
                    {
                        gradInputData[gradInputIdx] = numOps.Add(gradInputData[gradInputIdx], gradOutputData[gradOutputIdx]);
                    }
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

    #endregion
}
