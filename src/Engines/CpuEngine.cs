using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using TensorPrimitives = System.Numerics.Tensors.TensorPrimitives;

namespace AiDotNet.Engines;

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

        // Use SIMD-optimized TensorPrimitivesHelper (5-10× speedup for float)
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

        // Use SIMD-optimized TensorPrimitivesHelper (5-10× speedup for float)
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

        // Use SIMD-optimized TensorPrimitivesHelper (5-10× speedup for float)
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

        // Use SIMD-optimized TensorPrimitivesHelper (5-10× speedup for float)
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

        // Use SIMD-optimized TensorPrimitivesHelper (5-10× speedup for float)
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

        // Use SIMD-optimized TensorPrimitivesHelper (3-6× speedup for float)
        return TensorPrimitivesHelper<T>.Exp(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Log<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        // Use SIMD-optimized TensorPrimitivesHelper (3-6× speedup for float)
        return TensorPrimitivesHelper<T>.Log(vector);
    }

    /// <inheritdoc/>
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

        // Use SIMD-optimized TensorPrimitivesHelper (3-6× speedup for float)
        return TensorPrimitivesHelper<T>.Softmax(vector);
    }

    /// <inheritdoc/>
    public T CosineSimilarity<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have the same length.");

        // Use SIMD-optimized TensorPrimitivesHelper (10-15× speedup for float)
        return TensorPrimitivesHelper<T>.CosineSimilarity(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Log2<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        // Use SIMD-optimized TensorPrimitivesHelper (3-6× speedup for float)
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
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        // TensorPrimitives.Sin not available - use Math.Sin
        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
            result[i] = numOps.FromDouble(Math.Sin(val));
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Cos<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        // TensorPrimitives.Cos not available - use Math.Cos
        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
            result[i] = numOps.FromDouble(Math.Cos(val));
        }

        return result;
    }

    /// <inheritdoc/>
    public void SinCos<T>(Vector<T> vector, out Vector<T> sinResult, out Vector<T> cosResult)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();

        sinResult = new Vector<T>(vector.Length);
        cosResult = new Vector<T>(vector.Length);

        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
            sinResult[i] = numOps.FromDouble(Math.Sin(val));
            cosResult[i] = numOps.FromDouble(Math.Cos(val));
        }
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
        var random = seed.HasValue ? new Random(seed.Value) : new Random();
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
        var random = seed.HasValue ? new Random(seed.Value) : new Random();
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

        // Standard O(n³) matrix multiplication
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
            T rowSumSquares = DotProduct(row, row); // row · row = sum of squares for row
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

        // No vectorization benefit - column access is strided
        var valuesArray = values.ToArray();
        for (int i = 0; i < Math.Min(matrix.Rows, valuesArray.Length); i++)
        {
            matrix[i, columnIndex] = valuesArray[i];
        }
    }

    public void SetRow<T>(Matrix<T> matrix, int rowIndex, Vector<T> values)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));
        if (values == null) throw new ArgumentNullException(nameof(values));

        // Row access is contiguous - direct assignment
        var valuesArray = values.ToArray();
        for (int j = 0; j < Math.Min(matrix.Columns, valuesArray.Length); j++)
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
        // Use SIMD-optimized Tanh (3-6× speedup for float)
        return TensorPrimitivesHelper<T>.Tanh(vector);
    }

    public Vector<T> Sigmoid<T>(Vector<T> vector)
    {
        // Use SIMD-optimized Sigmoid (3-6× speedup for float)
        return TensorPrimitivesHelper<T>.Sigmoid(vector);
    }

    public Vector<T> ReLU<T>(Vector<T> vector)
    {
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
        // Convert tensor to vector, apply SIMD-optimized Tanh, convert back
        var flatVector = tensor.ToVector();
        var resultVector = TensorPrimitivesHelper<T>.Tanh(flatVector);
        return new Tensor<T>(tensor.Shape, resultVector);
    }

    public Tensor<T> Sigmoid<T>(Tensor<T> tensor)
    {
        // Convert tensor to vector, apply SIMD-optimized Sigmoid, convert back
        var flatVector = tensor.ToVector();
        var resultVector = TensorPrimitivesHelper<T>.Sigmoid(flatVector);
        return new Tensor<T>(tensor.Shape, resultVector);
    }

    public Tensor<T> ReLU<T>(Tensor<T> tensor)
    {
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
        return TensorPrimitivesHelper<T>.GELU(vector);
    }

    public Vector<T> Mish<T>(Vector<T> vector)
    {
        return TensorPrimitivesHelper<T>.Mish(vector);
    }

    public Vector<T> Swish<T>(Vector<T> vector)
    {
        return TensorPrimitivesHelper<T>.Swish(vector);
    }

    public Vector<T> ELU<T>(Vector<T> vector, double alpha = 1.0)
    {
        return TensorPrimitivesHelper<T>.ELU(vector, alpha);
    }

    public Tensor<T> GELU<T>(Tensor<T> tensor)
    {
        var flatVector = tensor.ToVector();
        var resultVector = TensorPrimitivesHelper<T>.GELU(flatVector);
        return new Tensor<T>(tensor.Shape, resultVector);
    }

    public Tensor<T> Mish<T>(Tensor<T> tensor)
    {
        var flatVector = tensor.ToVector();
        var resultVector = TensorPrimitivesHelper<T>.Mish(flatVector);
        return new Tensor<T>(tensor.Shape, resultVector);
    }

    public Tensor<T> Swish<T>(Tensor<T> tensor)
    {
        var flatVector = tensor.ToVector();
        var resultVector = TensorPrimitivesHelper<T>.Swish(flatVector);
        return new Tensor<T>(tensor.Shape, resultVector);
    }

    public Tensor<T> ELU<T>(Tensor<T> tensor, double alpha = 1.0)
    {
        var flatVector = tensor.ToVector();
        var resultVector = TensorPrimitivesHelper<T>.ELU(flatVector, alpha);
        return new Tensor<T>(tensor.Shape, resultVector);
    }

    #endregion
}
