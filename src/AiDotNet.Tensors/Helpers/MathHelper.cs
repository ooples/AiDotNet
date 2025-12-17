using System.Collections.Concurrent;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Provides mathematical utility methods for various numeric operations used in AI algorithms.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This helper class contains various mathematical functions that are commonly
/// used in AI and machine learning algorithms. These functions work with different numeric types
/// (like double, float, decimal) and handle the calculations in a consistent way.
///
/// Think of this class as a mathematical toolbox that provides specialized tools beyond what's
/// available in the standard Math class.
/// </para>
/// </remarks>
public static class MathHelper
{
    // Cache for numeric operations instances - avoids creating new objects on every call
    private static readonly ConcurrentDictionary<Type, object> _operationsCache = new();

    // Cache for acceleration support flags - avoids repeated type checks
    private static readonly ConcurrentDictionary<Type, (bool Cpu, bool Gpu)> _accelerationCache = new();

    /// <summary>
    /// Gets the appropriate numeric operations implementation for the specified type.
    /// </summary>
    /// <typeparam name="T">The numeric type to get operations for.</typeparam>
    /// <returns>An implementation of INumericOperations for the specified type.</returns>
    /// <exception cref="NotSupportedException">Thrown when the specified type is not supported.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method determines how to perform basic math operations (like addition,
    /// multiplication) based on what type of number you're working with.
    ///
    /// For example, adding two doubles is different from adding two integers at the computer level.
    /// This method returns the right "calculator" for your number type.
    /// </para>
    /// <para>
    /// <b>Performance:</b> This method caches the operations instances, so calling it multiple times
    /// for the same type T is very fast after the first call.
    /// </para>
    /// </remarks>
    public static INumericOperations<T> GetNumericOperations<T>()
    {
        return (INumericOperations<T>)_operationsCache.GetOrAdd(typeof(T), _ => CreateNumericOperations<T>());
    }

    /// <summary>
    /// Creates a new numeric operations instance for the specified type.
    /// </summary>
    private static object CreateNumericOperations<T>()
    {
        if (typeof(T) == typeof(double))
            return new DoubleOperations();
        if (typeof(T) == typeof(float))
            return new FloatOperations();
        if (typeof(T) == typeof(Half))
            return new HalfOperations();
        if (typeof(T) == typeof(decimal))
            return new DecimalOperations();
        if (typeof(T).IsGenericType && typeof(T).GetGenericTypeDefinition() == typeof(Complex<>))
        {
            var innerType = typeof(T).GetGenericArguments()[0];
            var complexOpsType = typeof(ComplexOperations<>).MakeGenericType(innerType);
            var instance = Activator.CreateInstance(complexOpsType);
            if (instance is null)
            {
                throw new InvalidOperationException($"Failed to create ComplexOperations instance for type {typeof(T)}");
            }
            return instance;
        }
        if (typeof(T) == typeof(byte))
            return new ByteOperations();
        if (typeof(T) == typeof(sbyte))
            return new SByteOperations();
        if (typeof(T) == typeof(short))
            return new ShortOperations();
        if (typeof(T) == typeof(ushort))
            return new UInt16Operations();
        if (typeof(T) == typeof(int))
            return new Int32Operations();
        if (typeof(T) == typeof(uint))
            return new UInt32Operations();
        if (typeof(T) == typeof(long))
            return new Int64Operations();
        if (typeof(T) == typeof(ulong))
            return new UInt64Operations();

        throw new NotSupportedException($"Numeric operations for type {typeof(T)} are not supported.");
    }

    /// <summary>
    /// Checks if the specified numeric type supports SIMD/CPU acceleration.
    /// </summary>
    /// <typeparam name="T">The numeric type to check.</typeparam>
    /// <returns>True if the type supports CPU acceleration; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SIMD (Single Instruction Multiple Data) allows the CPU to perform
    /// the same operation on multiple values at once, making vector operations much faster.
    /// Types like float, double, int, and long typically support SIMD acceleration.
    /// </para>
    /// <para>
    /// This method caches the result for performance - use it instead of checking
    /// typeof(T) == typeof(float) patterns in hot paths.
    /// </para>
    /// </remarks>
    public static bool SupportsCpuAcceleration<T>()
    {
        return GetAccelerationSupport<T>().Cpu;
    }

    /// <summary>
    /// Checks if the specified numeric type supports GPU acceleration.
    /// </summary>
    /// <typeparam name="T">The numeric type to check.</typeparam>
    /// <returns>True if the type supports GPU acceleration; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> GPU acceleration uses the graphics card to perform many calculations
    /// in parallel, which can be orders of magnitude faster for large datasets.
    /// Types like float and double are typically supported on GPUs, while decimal and
    /// complex types may only run on CPU.
    /// </para>
    /// <para>
    /// This method caches the result for performance - use it instead of checking
    /// typeof(T) == typeof(float) patterns in hot paths.
    /// </para>
    /// </remarks>
    public static bool SupportsGpuAcceleration<T>()
    {
        return GetAccelerationSupport<T>().Gpu;
    }

    /// <summary>
    /// Gets both CPU and GPU acceleration support for the specified numeric type.
    /// </summary>
    /// <typeparam name="T">The numeric type to check.</typeparam>
    /// <returns>A tuple containing (SupportsCpu, SupportsGpu) flags.</returns>
    public static (bool Cpu, bool Gpu) GetAccelerationSupport<T>()
    {
        return _accelerationCache.GetOrAdd(typeof(T), _ =>
        {
            var ops = GetNumericOperations<T>();
            return (ops.SupportsCpuAcceleration, ops.SupportsGpuAcceleration);
        });
    }

    /// <summary>
    /// Checks if the type T is float or double (the types that support TensorPrimitives operations).
    /// </summary>
    /// <typeparam name="T">The numeric type to check.</typeparam>
    /// <returns>True if T is float or double; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// Many SIMD-optimized operations in .NET's TensorPrimitives only support float and double.
    /// Use this method to check if you can use TensorPrimitives instead of generic fallback code.
    /// </para>
    /// </remarks>
    public static bool IsTensorPrimitivesSupported<T>()
    {
        return typeof(T) == typeof(float) || typeof(T) == typeof(double);
    }

    /// <summary>
    /// Checks if the type T is a floating-point type (float, double, or Half).
    /// </summary>
    /// <typeparam name="T">The numeric type to check.</typeparam>
    /// <returns>True if T is float, double, or Half; otherwise, false.</returns>
    public static bool IsFloatingPoint<T>()
    {
        return typeof(T) == typeof(float) || typeof(T) == typeof(double) || typeof(T) == typeof(Half);
    }

    /// <summary>
    /// Checks if the type T is an integer type.
    /// </summary>
    /// <typeparam name="T">The numeric type to check.</typeparam>
    /// <returns>True if T is an integer type; otherwise, false.</returns>
    public static bool IsIntegerType<T>()
    {
        return typeof(T) == typeof(byte) || typeof(T) == typeof(sbyte) ||
               typeof(T) == typeof(short) || typeof(T) == typeof(ushort) ||
               typeof(T) == typeof(int) || typeof(T) == typeof(uint) ||
               typeof(T) == typeof(long) || typeof(T) == typeof(ulong);
    }

    /// <summary>
    /// Restricts a value to be within a specified range.
    /// </summary>
    /// <typeparam name="T">The numeric type of the values.</typeparam>
    /// <param name="value">The value to clamp.</param>
    /// <param name="min">The minimum value of the range.</param>
    /// <param name="max">The maximum value of the range.</param>
    /// <returns>
    /// The value if it's within the range; otherwise, the nearest boundary value.
    /// </returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method ensures a number stays within a certain range.
    /// 
    /// For example, if you have a value of 15, but want to keep it between 0 and 10,
    /// Clamp(15, 0, 10) will return 10 (the maximum allowed).
    /// 
    /// Similarly, Clamp(-5, 0, 10) will return 0 (the minimum allowed).
    /// 
    /// This is useful in AI when you need to keep values within valid ranges,
    /// like probabilities between 0 and 1.
    /// </para>
    /// </remarks>
    public static T Clamp<T>(T value, T min, T max)
    {
        var numOps = GetNumericOperations<T>();
        if (numOps.LessThan(value, min))
            return min;
        if (numOps.GreaterThan(value, max))
            return max;

        return value;
    }

    /// <summary>
    /// Calculates the inverse hyperbolic tangent (atanh) of a value.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Inverse hyperbolic tangent: atanh(x) = 0.5 * log((1 + x) / (1 - x)).
    /// </para>
    /// <para>
    /// This method is provided for .NET Framework compatibility where <c>Math.Atanh</c> is not available.
    /// </para>
    /// </remarks>
    /// <param name="x">The input value.</param>
    /// <returns>The inverse hyperbolic tangent of <paramref name="x"/>.</returns>
    public static double Atanh(double x)
    {
        // Clamp to open interval (-1, 1) to avoid infinities from the identity near boundaries.
        const double eps = 1e-12;
        x = Clamp(x, -1.0 + eps, 1.0 - eps);

        // atanh(x) = 0.5 * ln((1+x)/(1-x))
        return 0.5 * Math.Log((1.0 + x) / (1.0 - x));
    }

    /// <summary>
    /// Calculates the modified Bessel function of the first kind of order 0.
    /// </summary>
    /// <typeparam name="T">The numeric type to use for calculations.</typeparam>
    /// <param name="x">The input value.</param>
    /// <returns>The value of the Bessel function I0(x).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Bessel functions are special mathematical functions that appear in many 
    /// AI and physics problems, especially those involving circular or cylindrical shapes.
    /// 
    /// The modified Bessel function I0(x) is used in probability distributions (like the 
    /// von Mises distribution) which are important in directional statistics and some 
    /// machine learning algorithms.
    /// 
    /// This method calculates an approximation of this function using a series expansion.
    /// </para>
    /// </remarks>
    public static T BesselI0<T>(T x)
    {
        var numOps = GetNumericOperations<T>();
        T sum = numOps.One;
        T y = numOps.Multiply(x, x);
        T term = numOps.One;

        for (int i = 1; i <= 50; i++)
        {
            term = numOps.Multiply(term, numOps.Divide(y, numOps.Multiply(numOps.FromDouble(4 * i * i), Factorial<T>(i))));
            sum = numOps.Add(sum, term);

            if (numOps.LessThan(term, numOps.FromDouble(1e-12)))
            {
                break;
            }
        }

        return sum;
    }

    /// <summary>
    /// Calculates the Gamma function for a given value.
    /// </summary>
    /// <typeparam name="T">The numeric type to use for calculations.</typeparam>
    /// <param name="x">The input value.</param>
    /// <returns>The Gamma function value G(x).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Gamma function is an extension of the factorial function to real numbers.
    /// While factorial (n!) is only defined for positive integers, the Gamma function works for 
    /// almost any real number.
    /// 
    /// For positive integers n: G(n) = (n-1)!
    /// 
    /// This function is important in many probability distributions used in machine learning,
    /// like the Beta and Dirichlet distributions, which are used in Bayesian methods.
    /// 
    /// This method uses the Lanczos approximation to calculate the Gamma function.
    /// </para>
    /// </remarks>
    public static T Gamma<T>(T x)
    {
        var numOps = GetNumericOperations<T>();

        // Lanczos approximation for Gamma function
        T[] p = { numOps.FromDouble(676.5203681218851),
                  numOps.FromDouble(-1259.1392167224028),
                  numOps.FromDouble(771.32342877765313),
                  numOps.FromDouble(-176.61502916214059),
                  numOps.FromDouble(12.507343278686905),
                  numOps.FromDouble(-0.13857109526572012),
                  numOps.FromDouble(9.9843695780195716e-6),
                  numOps.FromDouble(1.5056327351493116e-7) };

        if (numOps.LessThanOrEquals(x, numOps.Zero))
        {
            return numOps.Divide(Pi<T>(),
                numOps.Multiply(Sin(numOps.Multiply(Pi<T>(), x)),
                Gamma(numOps.Subtract(numOps.One, x))));
        }

        x = numOps.Subtract(x, numOps.One);
        T t = numOps.Add(x, numOps.FromDouble(7.5));
        T y = numOps.Exp(numOps.Subtract(
            numOps.Multiply(numOps.Add(x, numOps.FromDouble(0.5)), numOps.Log(t)),
            t));

        T sum = numOps.Zero;
        for (int i = 7; i >= 0; i--)
        {
            sum = numOps.Add(sum, numOps.Divide(p[i], numOps.Add(x, numOps.FromDouble(i))));
        }

        return numOps.Multiply(numOps.Multiply(numOps.Sqrt(numOps.FromDouble(2 * Math.PI)), sum), y);
    }

    /// <summary>
    /// Calculates the modified Bessel function of the second kind of order nu at point x.
    /// </summary>
    /// <typeparam name="T">The numeric type to use for calculations.</typeparam>
    /// <param name="nu">The order of the Bessel function.</param>
    /// <param name="x">The point at which to evaluate the function (must be positive).</param>
    /// <returns>The value of the modified Bessel function K_nu(x).</returns>
    /// <exception cref="ArgumentException">Thrown when x is not positive.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Bessel functions are special mathematical functions that appear in many 
    /// physics and engineering problems, especially those involving wave propagation, heat 
    /// conduction in cylindrical objects, and electromagnetic waves. This particular Bessel 
    /// function (K) is used when modeling damped oscillations or exponential decay in a system.
    /// </para>
    /// </remarks>
    public static T BesselK<T>(T nu, T x)
    {
        var numOps = GetNumericOperations<T>();

        // Approximation for modified Bessel function of the second kind
        if (numOps.LessThanOrEquals(x, numOps.Zero))
        {
            throw new ArgumentException("x must be positive");
        }

        T result;
        if (numOps.LessThan(x, numOps.FromDouble(2)))
        {
            T y = numOps.Multiply(numOps.FromDouble(0.25), numOps.Power(x, numOps.FromDouble(2)));
            result = numOps.Multiply(numOps.Power(numOps.FromDouble(0.5), nu),
                numOps.Divide(Gamma(numOps.Add(nu, numOps.FromDouble(1))),
                numOps.Power(x, nu)));

            T sum = numOps.One;
            T term = numOps.One;
            for (int k = 1; k <= 20; k++)
            {
                term = numOps.Multiply(term,
                    numOps.Divide(y,
                        numOps.Multiply(numOps.FromDouble(k),
                            numOps.Add(nu, numOps.FromDouble(k)))));
                sum = numOps.Add(sum, term);
                if (numOps.LessThan(numOps.Abs(term), numOps.Multiply(sum, numOps.FromDouble(1e-15))))
                {
                    break;
                }
            }
            result = numOps.Multiply(result, sum);
        }
        else
        {
            T y = numOps.Divide(numOps.FromDouble(2), x);
            result = numOps.Multiply(numOps.Exp(numOps.Multiply(x, numOps.FromDouble(-1))),
                numOps.Divide(numOps.Sqrt(numOps.Multiply(Pi<T>(), y)), numOps.FromDouble(2)));

            T sum = numOps.One;
            T term = numOps.One;
            for (int k = 1; k <= 20; k++)
            {
                term = numOps.Multiply(term,
                    numOps.Multiply(numOps.Add(numOps.Multiply(numOps.FromDouble(4),
                        numOps.Power(nu, numOps.FromDouble(2))),
                        numOps.Subtract(numOps.Power(numOps.FromDouble((2d * k) - 1d), numOps.FromDouble(2)),
                            numOps.One)),
                        numOps.Divide(y, numOps.FromDouble(k))));
                sum = numOps.Add(sum, term);
                if (numOps.LessThan(numOps.Abs(term), numOps.Multiply(sum, numOps.FromDouble(1e-15))))
                {
                    break;
                }
            }
            result = numOps.Multiply(result, sum);
        }

        return result;
    }

    /// <summary>
    /// Calculates the reciprocal (1/x) of a value.
    /// </summary>
    /// <typeparam name="T">The numeric type to use for calculations.</typeparam>
    /// <param name="value">The value to calculate the reciprocal of.</param>
    /// <returns>The reciprocal of the input value.</returns>
    /// <exception cref="DivideByZeroException">Thrown when the input value is zero.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The reciprocal of a number is simply 1 divided by that number. 
    /// For example, the reciprocal of 4 is 1/4 or 0.25, and the reciprocal of 0.5 is 1/0.5 or 2.
    /// Reciprocals are useful in many mathematical operations, especially when you need to 
    /// convert division into multiplication.
    /// </para>
    /// </remarks>
    public static T Reciprocal<T>(T value)
    {
        var numOps = GetNumericOperations<T>();
        if (numOps.Equals(value, numOps.Zero))
        {
            throw new DivideByZeroException("Cannot calculate reciprocal of zero.");
        }

        return numOps.Divide(numOps.One, value);
    }

    /// <summary>
    /// Calculates the sinc function (sin(px)/(px)) for a given value.
    /// </summary>
    /// <typeparam name="T">The numeric type to use for calculations.</typeparam>
    /// <param name="x">The input value.</param>
    /// <returns>The sinc of the input value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The sinc function is a mathematical function that appears frequently in 
    /// signal processing and Fourier analysis. It's defined as sin(px)/(px) for x ? 0, and 1 for x = 0.
    /// The sinc function creates a wave that gradually diminishes as you move away from the center,
    /// making it useful for filtering and interpolation in digital signal processing.
    /// </para>
    /// </remarks>
    public static T Sinc<T>(T x)
    {
        var numOps = GetNumericOperations<T>();
        if (numOps.Equals(x, numOps.Zero))
        {
            return numOps.One;
        }

        T piX = numOps.Multiply(numOps.FromDouble(Math.PI), x);
        return numOps.Divide(Sin(piX), piX);
    }

    /// <summary>
    /// Calculates the modulo (remainder after division) of x divided by y.
    /// </summary>
    /// <typeparam name="T">The numeric type to use for calculations.</typeparam>
    /// <param name="x">The dividend (number being divided).</param>
    /// <param name="y">The divisor (number dividing into x).</param>
    /// <returns>The remainder after dividing x by y.</returns>
    /// <exception cref="DivideByZeroException">Thrown when y is zero.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The modulo operation finds the remainder after division of one number by another.
    /// For example, 7 modulo 3 equals 1 because 7 divided by 3 is 2 with a remainder of 1.
    /// This is useful in many programming scenarios, like determining if a number is even or odd,
    /// or when you need to cycle through a range of values (like hours on a clock).
    /// </para>
    /// </remarks>
    public static T Modulo<T>(T x, T y)
    {
        var numOps = GetNumericOperations<T>();
        if (numOps.Equals(y, numOps.Zero))
        {
            throw new DivideByZeroException("Cannot perform modulo operation with zero divisor.");
        }

        T quotient = numOps.Divide(x, y);
        T flooredQuotient = numOps.FromDouble(Math.Floor(Convert.ToDouble(quotient)));

        return numOps.Subtract(x, numOps.Multiply(y, flooredQuotient));
    }

    /// <summary>
    /// Determines whether a numeric value is an integer (has no fractional part).
    /// </summary>
    /// <typeparam name="T">The numeric type to check.</typeparam>
    /// <param name="value">The value to check.</param>
    /// <returns>True if the value is an integer; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method checks if a number has any decimal/fractional part.
    /// For example, 5.0 is an integer (returns true), while 5.1 is not (returns false).
    /// This is useful when you need to ensure a value is a whole number before performing
    /// certain operations that only work with integers.
    /// </para>
    /// </remarks>
    public static bool IsInteger<T>(T value)
    {
        // If the value is equal to its rounded value, it's an integer
        var numOps = GetNumericOperations<T>();
        return numOps.Equals(value, numOps.Round(value));
    }

    /// <summary>
    /// Calculates the sigmoid function (1/(1+e^(-x))) for a given value.
    /// </summary>
    /// <typeparam name="T">The numeric type to use for calculations.</typeparam>
    /// <param name="x">The input value.</param>
    /// <returns>The sigmoid of the input value (between 0 and 1).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The sigmoid function is one of the most important functions in machine learning.
    /// It transforms any input value into a number between 0 and 1, creating an S-shaped curve.
    /// This is especially useful in neural networks and logistic regression where you need to 
    /// convert a raw score into a probability or make a binary decision (like yes/no classification).
    /// Large negative inputs produce values close to 0, while large positive inputs produce values close to 1.
    /// </para>
    /// </remarks>
    public static T Sigmoid<T>(T x)
    {
        var numOps = GetNumericOperations<T>();
        return numOps.Divide(numOps.One, numOps.Add(numOps.One, numOps.Exp(numOps.Negate(x))));
    }

    /// <summary>
    /// Determines if two numeric values are approximately equal within a specified tolerance.
    /// </summary>
    /// <typeparam name="T">The numeric type to compare.</typeparam>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <param name="tolerance">The maximum allowed difference between values to consider them equal.</param>
    /// <returns>True if the absolute difference between a and b is less than the tolerance; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When working with decimal numbers in computers, exact equality comparisons 
    /// can be problematic due to tiny rounding errors. This method allows you to check if two 
    /// numbers are "close enough" to be considered equal by specifying how much difference 
    /// you're willing to accept.
    /// </para>
    /// </remarks>
    public static bool AlmostEqual<T>(T a, T b, T tolerance)
    {
        var numOps = GetNumericOperations<T>();
        return numOps.LessThan(numOps.Abs(numOps.Subtract(a, b)), tolerance);
    }

    /// <summary>
    /// Determines if two numeric values are approximately equal using a default tolerance of 1e-8.
    /// </summary>
    /// <typeparam name="T">The numeric type to compare.</typeparam>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns>True if the values are approximately equal; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is a simplified version of the AlmostEqual method that uses a 
    /// pre-defined small tolerance value (0.00000001). Use this when you want to check if 
    /// two numbers are practically the same without specifying the exact tolerance.
    /// </para>
    /// </remarks>
    public static bool AlmostEqual<T>(T a, T b)
    {
        var numOps = GetNumericOperations<T>();
        return AlmostEqual(a, b, numOps.FromDouble(1e-8));
    }

    /// <summary>
    /// Generates a normally distributed random number using the Box-Muller transform.
    /// </summary>
    /// <typeparam name="T">The numeric type to return.</typeparam>
    /// <param name="mean">The mean of the normal distribution.</param>
    /// <param name="stdDev">The standard deviation of the normal distribution.</param>
    /// <param name="random">Optional Random instance to use. If null, creates a new unseeded Random instance.</param>
    /// <returns>A random number from the specified normal distribution.</returns>
    /// <remarks>
    /// <para>
    /// This method uses the Box-Muller transform to convert uniform random numbers into normally
    /// distributed random numbers. This is useful for initializing neural network weights.
    /// </para>
    /// <para><b>For Beginners:</b> Normal distribution (also called Gaussian distribution) is a
    /// bell-shaped probability distribution that is symmetric around its mean.
    ///
    /// This method generates random numbers that follow this distribution, which is important for
    /// neural network initialization. Using normally distributed values helps prevent issues during
    /// training and improves convergence.
    /// </para>
    /// <para>
    /// For reproducible results, pass in a seeded Random instance. Otherwise, a new unseeded
    /// Random will be created on each call, which breaks reproducibility.
    /// </para>
    /// <para><b>Type Requirements:</b> This method requires that type T can be converted to and from
    /// double via <see cref="Convert.ToDouble"/> and <see cref="INumericOperations{T}.FromDouble"/>.
    /// This works for all built-in numeric types (int, float, double, decimal, etc.) but may not
    /// work for custom numeric types that don't support double conversion.
    /// </para>
    /// </remarks>
    public static T GetNormalRandom<T>(T mean, T stdDev, Random? random = null)
    {
        var numOps = GetNumericOperations<T>();

        // Validate standard deviation is non-negative
        if (numOps.LessThan(stdDev, numOps.Zero))
            throw new ArgumentException("Standard deviation must be non-negative.", nameof(stdDev));

        var rng = random ?? RandomHelper.CreateSecureRandom();

        // Box-Muller transform
        double u1 = 1.0 - rng.NextDouble(); // Uniform(0,1] random numbers
        double u2 = 1.0 - rng.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);

        // Scale and shift to get desired mean and standard deviation
        double result = randStdNormal * Convert.ToDouble(stdDev) + Convert.ToDouble(mean);

        return numOps.FromDouble(result);
    }

    /// <summary>
    /// Calculates the Bessel function of the first kind of order nu at point x.
    /// </summary>
    /// <typeparam name="T">The numeric type to use for calculations.</typeparam>
    /// <param name="nu">The order of the Bessel function.</param>
    /// <param name="x">The point at which to evaluate the function.</param>
    /// <returns>The value of the Bessel function J_nu(x).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Bessel functions are special mathematical functions that appear in many 
    /// physics and engineering problems, especially those involving wave propagation, heat 
    /// conduction in cylindrical objects, and vibrations. This particular Bessel function (J) 
    /// is used when modeling oscillations or waves in a system. Think of it as a more complex 
    /// version of sine or cosine functions, but specifically designed for problems with 
    /// cylindrical symmetry.
    /// </para>
    /// </remarks>
    public static T BesselJ<T>(T nu, T x)
    {
        var numOps = GetNumericOperations<T>();

        // Handle special cases
        if (numOps.Equals(x, numOps.Zero))
        {
            return numOps.Equals(nu, numOps.Zero) ? numOps.One : numOps.Zero;
        }

        if (numOps.LessThan(x, numOps.Zero))
        {
            return numOps.Multiply(
                numOps.Power(numOps.FromDouble(-1), nu),
                BesselJ(nu, numOps.Abs(x))
            );
        }

        // Convert nu to double for comparisons
        double nuDouble = Convert.ToDouble(nu);
        double xDouble = Convert.ToDouble(x);

        // Use series expansion for small x
        if (xDouble <= 12)
        {
            return BesselJSeries(nu, x);
        }

        // Use asymptotic expansion for large x
        if (xDouble > 12 && xDouble > Math.Abs(nuDouble))
        {
            return BesselJAsymptotic(nu, x);
        }

        // Use recurrence relation for intermediate values
        return BesselJRecurrence(nu, x);
    }

    /// <summary>
    /// Calculates the Bessel function of the first kind using a series expansion method.
    /// </summary>
    /// <typeparam name="T">The numeric type to use for calculations.</typeparam>
    /// <param name="nu">The order of the Bessel function.</param>
    /// <param name="x">The point at which to evaluate the function.</param>
    /// <returns>The value of the Bessel function calculated using series expansion.</returns>
    private static T BesselJSeries<T>(T nu, T x)
    {
        var numOps = GetNumericOperations<T>();
        T _sum = numOps.Zero;
        T _factorial = numOps.One;
        T _xOver2 = numOps.Divide(x, numOps.FromDouble(2));
        T _xOver2Squared = numOps.Square(_xOver2);
        T _term = numOps.One;

        for (int m = 0; m <= 50; m++)  // Increased max terms for better accuracy
        {
            if (m > 0)
            {
                _factorial = numOps.Multiply(_factorial, numOps.FromDouble(m));
                _term = numOps.Divide(_term, _factorial);
                _term = numOps.Multiply(_term, _xOver2Squared);
            }

            T _numerator = numOps.Power(numOps.Negate(numOps.One), numOps.FromDouble(m));
            T _denominator = numOps.Multiply(_factorial, Gamma(numOps.Add(numOps.FromDouble(m), numOps.Add(nu, numOps.One))));

            T _summand = numOps.Multiply(_numerator, numOps.Divide(numOps.Power(_xOver2, numOps.Add(numOps.FromDouble(2 * m), nu)), _denominator));
            _sum = numOps.Add(_sum, _summand);

            if (numOps.LessThan(numOps.Abs(_summand), numOps.FromDouble(1e-15)))
            {
                break;
            }
        }

        return _sum;
    }

    /// <summary>
    /// Calculates the Bessel function of the first kind using an asymptotic expansion method.
    /// </summary>
    /// <typeparam name="T">The numeric type to use for calculations.</typeparam>
    /// <param name="nu">The order of the Bessel function.</param>
    /// <param name="x">The point at which to evaluate the function.</param>
    /// <returns>The value of the Bessel function calculated using asymptotic expansion.</returns>
    /// <remarks>
    /// <para>
    /// Uses the asymptotic expansion for large x:
    /// J_ν(x) ≈ sqrt(2/(πx)) * [P(ν,x) * cos(θ) - Q(ν,x) * sin(θ)]
    /// where θ = x - (2ν + 1)π/4
    /// </para>
    /// <para>
    /// For the first-order expansion:
    /// - P(ν,x) = 1 (leading term)
    /// - Q(ν,x) = μ / (8x) where μ = 4ν² - 1
    /// </para>
    /// </remarks>
    private static T BesselJAsymptotic<T>(T nu, T x)
    {
        var numOps = GetNumericOperations<T>();

        // Correct formula: μ = 4ν² - 1 (not ν² - 0.25)
        T _mu = numOps.Subtract(
            numOps.Multiply(numOps.FromDouble(4), numOps.Square(nu)),
            numOps.One
        );

        // Phase angle: θ = x - (2ν + 1)π/4
        T _theta = numOps.Subtract(
            x,
            numOps.Multiply(
                numOps.FromDouble(0.25 * Math.PI),
                numOps.Add(numOps.Multiply(numOps.FromDouble(2), nu), numOps.One)
            )
        );

        // First-order asymptotic expansion coefficients
        T _p = numOps.One;

        // Correct formula: Q = -μ / (8x) (note the negative sign)
        T _q = numOps.Negate(
            numOps.Divide(_mu, numOps.Multiply(numOps.FromDouble(8), x))
        );

        T _cosTheta = Cos(_theta);
        T _sinTheta = Sin(_theta);

        // Amplitude factor: sqrt(2/(πx))
        T _sqrtX = numOps.Sqrt(x);
        T _sqrtPi = numOps.Sqrt(numOps.FromDouble(Math.PI));
        T _factor = numOps.Divide(numOps.Sqrt(numOps.FromDouble(2)), numOps.Multiply(_sqrtPi, _sqrtX));

        // Combine: sqrt(2/(πx)) * [P*cos(θ) - Q*sin(θ)]
        // Since Q has the negative sign built in, we use Add instead of Subtract
        return numOps.Multiply(
            _factor,
            numOps.Add(numOps.Multiply(_p, _cosTheta), numOps.Multiply(_q, _sinTheta))
        );
    }

    /// <summary>
    /// Calculates the Bessel function of the first kind using a recurrence relation method.
    /// </summary>
    /// <typeparam name="T">The numeric type to use for calculations.</typeparam>
    /// <param name="nu">The order of the Bessel function.</param>
    /// <param name="x">The point at which to evaluate the function.</param>
    /// <returns>The value of the Bessel function calculated using recurrence relations.</returns>
    /// <remarks>
    /// <para>
    /// Uses the backward recurrence relation for Bessel functions:
    /// J_{k-1}(x) = (2k/x) * J_k(x) - J_{k+1}(x)
    /// </para>
    /// <para>
    /// Starting from high-order approximations (using asymptotic expansion), this method
    /// recursively computes lower-order Bessel functions using the stable backward recurrence.
    /// </para>
    /// </remarks>
    private static T BesselJRecurrence<T>(T nu, T x)
    {
        var numOps = GetNumericOperations<T>();
        int n = (int)Math.Ceiling(Convert.ToDouble(nu));
        T _nuInt = numOps.FromDouble(n);

        T _jn = BesselJAsymptotic(_nuInt, x);
        T _jnMinus1 = BesselJAsymptotic(numOps.Subtract(_nuInt, numOps.One), x);

        // Backward recurrence: J_{k-1}(x) = (2k/x) * J_k(x) - J_{k+1}(x)
        for (int k = n - 1; k >= 0; k--)
        {
            // Correct formula: use 2*k (not 2*k+2)
            T _jnMinus2 = numOps.Subtract(
                numOps.Multiply(numOps.FromDouble(2 * k), numOps.Divide(_jnMinus1, x)),
                _jn
            );
            _jn = _jnMinus1;
            _jnMinus1 = _jnMinus2;
        }

        if (numOps.Equals(nu, _nuInt))
        {
            return _jn;
        }

        // Interpolate for non-integer nu using linear interpolation
        T _jnPlus1 = numOps.Subtract(
            numOps.Multiply(numOps.FromDouble(2 * n), numOps.Divide(_jn, x)),
            _jnMinus1
        );
        T _t = numOps.Subtract(nu, numOps.Round(nu));
        return numOps.Add(numOps.Multiply(_jn, numOps.Subtract(numOps.One, _t)), numOps.Multiply(_jnPlus1, _t));
    }

    /// <summary>
    /// Calculates the factorial of a non-negative integer.
    /// </summary>
    /// <typeparam name="T">The numeric type to use for the result.</typeparam>
    /// <param name="n">The non-negative integer for which to calculate the factorial.</param>
    /// <returns>The factorial of n as type T.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The factorial of a number (written as n!) is the product of all positive 
    /// integers less than or equal to n. For example, 5! = 5 * 4 * 3 * 2 * 1 = 120.
    /// Factorials are used in many probability and statistics calculations.
    /// </para>
    /// </remarks>
    public static T Factorial<T>(int n)
    {
        if (n < 0)
            throw new ArgumentException("Factorial is not defined for negative numbers.", nameof(n));

        var ops = GetNumericOperations<T>();

        if (n == 0 || n == 1)
            return ops.One;

        T result = ops.One;
        for (int i = 2; i <= n; i++)
        {
            result = ops.Multiply(result, ops.FromDouble(i));
        }

        return result;
    }

    /// <summary>
    /// Returns the mathematical constant Pi (p) converted to the specified numeric type.
    /// </summary>
    /// <typeparam name="T">The numeric type to convert Pi to.</typeparam>
    /// <returns>The value of Pi as type T.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pi (p) is a fundamental mathematical constant representing the ratio of a 
    /// circle's circumference to its diameter, approximately equal to 3.14159. It appears in many 
    /// mathematical formulas, especially those involving circles, waves, and periodic functions.
    /// </para>
    /// </remarks>
    public static T Pi<T>()
    {
        return GetNumericOperations<T>().FromDouble(Math.PI);
    }

    /// <summary>
    /// Calculates the sine of an angle.
    /// </summary>
    /// <typeparam name="T">The numeric type to use for calculations.</typeparam>
    /// <param name="x">The angle in radians.</param>
    /// <returns>The sine of the specified angle as type T.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The sine function is a fundamental trigonometric function that relates the 
    /// angles of a right triangle to the ratios of the lengths of its sides. In the context of 
    /// a unit circle, sine represents the y-coordinate of a point on the circle at a given angle.
    /// The input angle must be in radians, not degrees (2p radians = 360 degrees).
    /// </para>
    /// </remarks>
    public static T Sin<T>(T x)
    {
        return GetNumericOperations<T>().FromDouble(Math.Sin(Convert.ToDouble(x)));
    }

    /// <summary>
    /// Calculates the cosine of an angle.
    /// </summary>
    /// <typeparam name="T">The numeric type to use for calculations.</typeparam>
    /// <param name="x">The angle in radians.</param>
    /// <returns>The cosine of the specified angle as type T.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The cosine function is a fundamental trigonometric function that relates the 
    /// angles of a right triangle to the ratios of the lengths of its sides. In the context of 
    /// a unit circle, cosine represents the x-coordinate of a point on the circle at a given angle.
    /// The input angle must be in radians, not degrees (2p radians = 360 degrees).
    /// </para>
    /// </remarks>
    public static T Cos<T>(T x)
    {
        return GetNumericOperations<T>().FromDouble(Math.Cos(Convert.ToDouble(x)));
    }

    /// <summary>
    /// Calculates the hyperbolic tangent of a value.
    /// </summary>
    /// <typeparam name="T">The numeric type to use for calculations.</typeparam>
    /// <param name="x">The value to calculate the hyperbolic tangent for.</param>
    /// <returns>The hyperbolic tangent of the specified value as type T.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The hyperbolic tangent (tanh) is a function commonly used in neural networks 
    /// as an activation function. Unlike the regular tangent function, which can grow infinitely large, 
    /// tanh always outputs values between -1 and 1. This makes it useful for creating models that need 
    /// to predict values within a specific range. The function has an S-shape (sigmoid) and maps any 
    /// input value to an output between -1 and 1.
    /// </para>
    /// </remarks>
    public static T Tanh<T>(T x)
    {
        var numOps = GetNumericOperations<T>();
        T exp2x = numOps.Exp(numOps.Multiply(numOps.FromDouble(2), x));
        return numOps.Divide(
            numOps.Subtract(exp2x, numOps.One),
            numOps.Add(exp2x, numOps.One)
        );
    }

    /// <summary>
    /// Calculates the base-2 logarithm of a number.
    /// </summary>
    /// <param name="x">The positive number to calculate the logarithm for.</param>
    /// <returns>The base-2 logarithm of the specified number.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when x is less than or equal to zero.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The base-2 logarithm (log2) tells you what power you need to raise 2 to in order 
    /// to get a specific number. For example, log2(8) = 3 because 2^3 = 8. Base-2 logarithms are commonly 
    /// used in computer science and information theory because computers use binary (base-2) number systems.
    /// </para>
    /// </remarks>
    public static double Log2(double x)
    {
        if (x <= 0)
            throw new ArgumentOutOfRangeException(nameof(x), "Logarithm is undefined for non-positive numbers.");
        return Math.Log(x) / Math.Log(2);
    }

    /// <summary>
    /// Returns the smaller of two values.
    /// </summary>
    /// <typeparam name="T">The numeric type of the values to compare.</typeparam>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns>The smaller of the two values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method simply compares two numbers and returns whichever one is smaller.
    /// For example, Min(5, 10) would return 5.
    /// </para>
    /// </remarks>
    public static T Min<T>(T a, T b)
    {
        return GetNumericOperations<T>().LessThan(a, b) ? a : b;
    }

    /// <summary>
    /// Returns the larger of two values.
    /// </summary>
    /// <typeparam name="T">The numeric type of the values to compare.</typeparam>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns>The larger of the two values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method simply compares two numbers and returns whichever one is larger.
    /// For example, Max(5, 10) would return 10.
    /// </para>
    /// </remarks>
    public static T Max<T>(T a, T b)
    {
        return GetNumericOperations<T>().GreaterThan(a, b) ? a : b;
    }

    /// <summary>
    /// Calculates the arc cosine (inverse cosine) of a value.
    /// </summary>
    /// <typeparam name="T">The numeric type to use for calculations.</typeparam>
    /// <param name="x">The value whose arc cosine is to be calculated. Must be between -1 and 1.</param>
    /// <returns>The arc cosine of the specified value, in radians.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The arc cosine function is the inverse of the cosine function. While cosine 
    /// takes an angle and returns a value between -1 and 1, arc cosine takes a value between -1 and 1 
    /// and returns the corresponding angle in radians. For example, since cos(0) = 1, arccos(1) = 0.
    /// This is useful when you know the cosine value and need to find the original angle.
    /// </para>
    /// </remarks>
    public static T ArcCos<T>(T x)
    {
        var numOps = GetNumericOperations<T>();

        // ArcCos(x) = p/2 - ArcSin(x)
        var arcSin = MathHelper.ArcSin(x);
        var halfPi = numOps.Divide(Pi<T>(), numOps.FromDouble(2.0));

        return numOps.Subtract(halfPi, arcSin);
    }

    /// <summary>
    /// Calculates the arc sine (inverse sine) of a value.
    /// </summary>
    /// <typeparam name="T">The numeric type to use for calculations.</typeparam>
    /// <param name="x">The value whose arc sine is to be calculated. Must be between -1 and 1.</param>
    /// <returns>The arc sine of the specified value, in radians.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when x is less than -1 or greater than 1.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The arc sine function is the inverse of the sine function. While sine 
    /// takes an angle and returns a value between -1 and 1, arc sine takes a value between -1 and 1 
    /// and returns the corresponding angle in radians. For example, since sin(p/2) = 1, arcsin(1) = p/2.
    /// This is useful when you know the sine value and need to find the original angle.
    /// </para>
    /// </remarks>
    public static T ArcSin<T>(T x)
    {
        var numOps = GetNumericOperations<T>();

        // Check if x is within the valid range [-1, 1]
        if (numOps.LessThan(x, numOps.FromDouble(-1)) || numOps.GreaterThan(x, numOps.One))
        {
            throw new ArgumentOutOfRangeException(nameof(x), "ArcSin is only defined for values between -1 and 1.");
        }

        // ArcSin(x) = ArcTan(x / sqrt(1 - x^2))
        var oneMinusXSquared = numOps.Subtract(numOps.One, numOps.Multiply(x, x));
        var denominator = numOps.Sqrt(oneMinusXSquared);
        var fraction = numOps.Divide(x, denominator);

        return ArcTan(fraction);
    }

    /// <summary>
    /// Calculates the arc tangent (inverse tangent) of a value.
    /// </summary>
    /// <typeparam name="T">The numeric type to use for calculations.</typeparam>
    /// <param name="x">The value whose arc tangent is to be calculated.</param>
    /// <returns>The arc tangent of the specified value, in radians.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The arc tangent function is the inverse of the tangent function. While tangent
    /// takes an angle and returns a ratio, arc tangent takes a ratio and returns the corresponding
    /// angle in radians. For example, since tan(0) = 0, arctan(0) = 0.
    /// </para>
    /// <para>
    /// This implementation uses a Taylor series approximation for |x| &lt;= 1, which is a mathematical technique
    /// that represents a function as an infinite sum of terms. For |x| &gt; 1, we use range reduction:
    /// atan(x) = sgn(x) * (π/2 - atan(1/|x|)) to transform the problem into the convergent range.
    /// This ensures accurate results for all input values.
    /// </para>
    /// <para>
    /// <b>Why range reduction is needed:</b>
    /// The Taylor series atan(x) = x - x³/3 + x⁵/5 - x⁷/7 + ... only converges for |x| ≤ 1.
    /// For larger values, we use the identity atan(x) + atan(1/x) = π/2 (for x &gt; 0) to transform
    /// the problem. This way, we compute atan(1/x) instead, which is guaranteed to be in the convergent
    /// range when |x| &gt; 1.
    /// </para>
    /// </remarks>
    public static T ArcTan<T>(T x)
    {
        var numOps = GetNumericOperations<T>();

        // Check if we need range reduction for |x| > 1
        T absX = numOps.Abs(x);
        bool needsRangeReduction = numOps.GreaterThan(absX, numOps.One);

        T xReduced;
        if (needsRangeReduction)
        {
            // For |x| > 1, use range reduction: atan(x) = sgn(x) * (π/2 - atan(1/|x|))
            // This transforms the problem to atan(1/|x|) where 1/|x| < 1 (convergent range)
            xReduced = numOps.Divide(numOps.One, absX);
        }
        else
        {
            // For |x| <= 1, use x directly (already in convergent range)
            xReduced = absX;
        }

        // Use Taylor series approximation for ArcTan in the convergent range |x| <= 1
        // Series: atan(x) = x - x³/3 + x⁵/5 - x⁷/7 + ...
        T _result = xReduced;
        T _xPower = xReduced;
        T _xSquared = numOps.Square(xReduced);
        int _sign = 1;

        for (int n = 3; n <= 15; n += 2)
        {
            _sign = -_sign;
            _xPower = numOps.Multiply(_xPower, _xSquared);
            T _term = numOps.Divide(_xPower, numOps.FromDouble(n));
            _result = numOps.Add(_result, numOps.Multiply(numOps.FromDouble(_sign), _term));
        }

        // If we used range reduction, apply the inverse transformation
        if (needsRangeReduction)
        {
            // atan(|x|) = π/2 - atan(1/|x|)
            T halfPi = numOps.Divide(Pi<T>(), numOps.FromDouble(2));
            _result = numOps.Subtract(halfPi, _result);
        }

        // Apply the original sign: atan(x) = sgn(x) * atan(|x|)
        if (numOps.LessThan(x, numOps.Zero))
        {
            _result = numOps.Negate(_result);
        }

        return _result;
    }

    /// <summary>
    /// Calculates the error function (erf) of a value.
    /// </summary>
    /// <typeparam name="T">The numeric type to use for calculations.</typeparam>
    /// <param name="x">The value for which to calculate the error function.</param>
    /// <returns>The error function value for the specified input.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The error function (erf) is a special mathematical function that appears in 
    /// probability, statistics, and partial differential equations. It describes the probability that 
    /// a random variable with normal distribution will fall within a certain range.
    /// </para>
    /// <para>
    /// This implementation uses Abramowitz and Stegun's numerical approximation, which provides 
    /// a good balance between accuracy and computational efficiency. The error function always 
    /// returns values between -1 and 1, and erf(0) = 0.
    /// </para>
    /// </remarks>
    public static T Erf<T>(T x)
    {
        var _numOps = GetNumericOperations<T>();
        T _sign = _numOps.GreaterThanOrEquals(x, _numOps.Zero) ? _numOps.FromDouble(1) : _numOps.FromDouble(-1);
        x = _numOps.Abs(x);

        // Constants for Abramowitz and Stegun approximation
        T _a1 = _numOps.FromDouble(0.254829592);
        T _a2 = _numOps.FromDouble(-0.284496736);
        T _a3 = _numOps.FromDouble(1.421413741);
        T _a4 = _numOps.FromDouble(-1.453152027);
        T _a5 = _numOps.FromDouble(1.061405429);
        T _p = _numOps.FromDouble(0.3275911);

        T _t = _numOps.Divide(_numOps.FromDouble(1), _numOps.Add(_numOps.FromDouble(1), _numOps.Multiply(_p, x)));
        T _y = _numOps.Subtract(_numOps.FromDouble(1),
            _numOps.Multiply(_t,
                _numOps.Multiply(
                    _numOps.Exp(_numOps.Negate(_numOps.Square(x))),
                    _numOps.Add(_a1,
                        _numOps.Multiply(_t,
                            _numOps.Add(_a2,
                                _numOps.Multiply(_t,
                                    _numOps.Add(_a3,
                                        _numOps.Multiply(_t,
                                            _numOps.Add(_a4,
                                                _numOps.Multiply(_a5, _t)))))))))));

        return _numOps.Multiply(_sign, _y);
    }

    /// <summary>
    /// Calculates the y-intercept for a linear regression model.
    /// </summary>
    /// <typeparam name="T">The numeric type to use for calculations.</typeparam>
    /// <param name="xMatrix">The matrix of independent variables (features).</param>
    /// <param name="y">The vector of dependent variables (target values).</param>
    /// <param name="coefficients">The vector of coefficients (slopes) for each feature.</param>
    /// <returns>The y-intercept value that makes the regression line pass through the mean of the data.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when the dimensions of xMatrix, y, and coefficients are not compatible.
    /// </exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In linear regression, the y-intercept is the value of the dependent variable 
    /// when all independent variables are zero. It represents the baseline value of your prediction 
    /// before considering any features.
    /// </para>
    /// <para>
    /// This method uses the formula: y-intercept = mean(y) - (coefficient1 * mean(x1) + coefficient2 * mean(x2) + ...)
    /// which ensures that the regression line passes through the point of means (the average of all data points).
    /// </para>
    /// <para>
    /// For example, in a house price prediction model, if you have features like square footage and 
    /// number of bedrooms, the y-intercept would be the baseline price of a house before considering 
    /// these features.
    /// </para>
    /// </remarks>
    public static T CalculateYIntercept<T>(AiDotNet.Tensors.LinearAlgebra.Matrix<T> xMatrix, AiDotNet.Tensors.LinearAlgebra.Vector<T> y, AiDotNet.Tensors.LinearAlgebra.Vector<T> coefficients)
    {
        var _numOps = GetNumericOperations<T>();

        if (xMatrix.Rows != y.Length || xMatrix.Columns != coefficients.Length)
            throw new ArgumentException("Dimensions of xMatrix, y, and coefficients must be compatible.");

        T _yMean = CalculateAverage(y);
        T _predictedSum = _numOps.Zero;

        for (int i = 0; i < xMatrix.Columns; i++)
        {
            T _xMean = CalculateAverage(xMatrix.GetColumn(i));
            _predictedSum = _numOps.Add(_predictedSum, _numOps.Multiply(coefficients[i], _xMean));
        }

        return _numOps.Subtract(_yMean, _predictedSum);
    }

    /// <summary>
    /// Calculates the average (arithmetic mean) of all elements in a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    /// <param name="vector">The vector to calculate the average of.</param>
    /// <returns>The average value of all elements in the vector.</returns>
    /// <exception cref="ArgumentException">Thrown when the vector is empty.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The average (or mean) is the sum of all values divided by the count of values.
    /// For example, the average of [2, 4, 6, 8] is (2+4+6+8)/4 = 5.
    /// </para>
    /// <para>
    /// This method works with any numeric type that implements INumericOperations.
    /// </para>
    /// </remarks>
    private static T CalculateAverage<T>(AiDotNet.Tensors.LinearAlgebra.Vector<T> vector)
    {
        if (vector.Length == 0)
            throw new ArgumentException("Cannot calculate average of an empty vector.", nameof(vector));

        return vector.Mean();
    }
}
