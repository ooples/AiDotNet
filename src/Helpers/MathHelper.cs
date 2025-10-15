namespace AiDotNet.Helpers;

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
    /// <summary>
    /// Gets a value indicating whether hardware acceleration (SIMD) is available for vector operations.
    /// </summary>
    /// <value>
    /// <c>true</c> if hardware acceleration is available; otherwise, <c>false</c>.
    /// </value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Hardware acceleration uses special CPU instructions to perform 
    /// multiple calculations at once, making vector and matrix operations much faster.
    /// 
    /// This property checks if your computer's processor supports these special instructions
    /// (called SIMD - Single Instruction, Multiple Data).
    /// </para>
    /// </remarks>
    public static bool IsHardwareAccelerated { get; } = System.Numerics.Vector.IsHardwareAccelerated;
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
    /// </remarks>
    public static INumericOperations<T> GetNumericOperations<T>()
    {
        if (typeof(T) == typeof(double))
            return (INumericOperations<T>)new DoubleOperations();
        else if (typeof(T) == typeof(float))
            return (INumericOperations<T>)new FloatOperations();
        else if (typeof(T) == typeof(decimal))
            return (INumericOperations<T>)new DecimalOperations();
        else if (typeof(T) == typeof(Complex<T>))
            return (INumericOperations<T>)new ComplexOperations<T>();
        else if (typeof(T) == typeof(byte))
            return (INumericOperations<T>)new ByteOperations();
        else if (typeof(T) == typeof(sbyte))
            return (INumericOperations<T>)new SByteOperations();
        else if (typeof(T) == typeof(short))
            return (INumericOperations<T>)new ShortOperations();
        else if (typeof(T) == typeof(ushort))
            return (INumericOperations<T>)new UInt16Operations();
        else if (typeof(T) == typeof(int))
            return (INumericOperations<T>)new Int32Operations();
        else if (typeof(T) == typeof(uint))
            return (INumericOperations<T>)new UInt32Operations();
        else if (typeof(T) == typeof(long))
            return (INumericOperations<T>)new Int64Operations();
        else if (typeof(T) == typeof(ulong))
            return (INumericOperations<T>)new UInt64Operations();
        else
            throw new NotSupportedException($"Numeric operations for type {typeof(T)} are not supported.");
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
    /// Calculates the modified Bessel function of the first kind of order 0.
    /// </summary>
    /// <typeparam name="T">The numeric type to use for calculations.</typeparam>
    /// <param name="x">The input value.</param>
    /// <returns>The value of the Bessel function I₀(x).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Bessel functions are special mathematical functions that appear in many 
    /// AI and physics problems, especially those involving circular or cylindrical shapes.
    /// 
    /// The modified Bessel function I₀(x) is used in probability distributions (like the 
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
    /// <returns>The Gamma function value Γ(x).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Gamma function is an extension of the factorial function to real numbers.
    /// While factorial (n!) is only defined for positive integers, the Gamma function works for 
    /// almost any real number.
    /// 
    /// For positive integers n: Γ(n) = (n-1)!
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
        T y = numOps.Exp(numOps.Multiply(numOps.Multiply(numOps.Add(x, numOps.FromDouble(0.5)), 
            numOps.Log(t)), numOps.FromDouble(-1)));

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
                        numOps.Subtract(numOps.Power(numOps.FromDouble(2 * k - 1), numOps.FromDouble(2)), 
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
    /// Calculates the sinc function (sin(πx)/(πx)) for a given value.
    /// </summary>
    /// <typeparam name="T">The numeric type to use for calculations.</typeparam>
    /// <param name="x">The input value.</param>
    /// <returns>The sinc of the input value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The sinc function is a mathematical function that appears frequently in 
    /// signal processing and Fourier analysis. It's defined as sin(πx)/(πx) for x ≠ 0, and 1 for x = 0.
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
    /// </remarks>
    public static T GetNormalRandom<T>(T mean, T stdDev)
    {
        var numOps = GetNumericOperations<T>();
        var random = new Random();

        // Box-Muller transform
        double u1 = 1.0 - random.NextDouble(); // Uniform(0,1] random numbers
        double u2 = 1.0 - random.NextDouble();
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
    private static T BesselJAsymptotic<T>(T nu, T x)
    {
        var numOps = GetNumericOperations<T>();
        T _mu = numOps.Subtract(numOps.Multiply(nu, nu), numOps.FromDouble(0.25));
        T _theta = numOps.Subtract(x, numOps.Multiply(numOps.FromDouble(0.25 * Math.PI), numOps.Add(numOps.Multiply(numOps.FromDouble(2), nu), numOps.One)));

        T _p = numOps.One;
        T _q = numOps.Divide(_mu, numOps.Multiply(numOps.FromDouble(8), x));

        T _cosTheta = Cos(_theta);
        T _sinTheta = Sin(_theta);

        T _sqrtX = numOps.Sqrt(x);
        T _sqrtPi = numOps.Sqrt(numOps.FromDouble(Math.PI));
        T _factor = numOps.Divide(numOps.Sqrt(numOps.FromDouble(2)), numOps.Multiply(_sqrtPi, _sqrtX));

        return numOps.Multiply(_factor, numOps.Add(numOps.Multiply(_p, _cosTheta), numOps.Multiply(_q, _sinTheta)));
    }

    /// <summary>
    /// Calculates the Bessel function of the first kind using a recurrence relation method.
    /// </summary>
    /// <typeparam name="T">The numeric type to use for calculations.</typeparam>
    /// <param name="nu">The order of the Bessel function.</param>
    /// <param name="x">The point at which to evaluate the function.</param>
    /// <returns>The value of the Bessel function calculated using recurrence relations.</returns>
    private static T BesselJRecurrence<T>(T nu, T x)
    {
        var numOps = GetNumericOperations<T>();
        int n = (int)Math.Ceiling(Convert.ToDouble(nu));
        T _nuInt = numOps.FromDouble(n);

        T _jn = BesselJAsymptotic(_nuInt, x);
        T _jnMinus1 = BesselJAsymptotic(numOps.Subtract(_nuInt, numOps.One), x);

        for (int k = n - 1; k >= 0; k--)
        {
            T _jnMinus2 = numOps.Subtract(
                numOps.Multiply(numOps.FromDouble(2 * k + 2), numOps.Divide(_jnMinus1, x)),
                _jn
            );
            _jn = _jnMinus1;
            _jnMinus1 = _jnMinus2;
        }

        if (numOps.Equals(nu, _nuInt))
        {
            return _jn;
        }

        // Interpolate for non-integer nu
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
    /// integers less than or equal to n. For example, 5! = 5 × 4 × 3 × 2 × 1 = 120.
    /// Factorials are used in many probability and statistics calculations.
    /// </para>
    /// </remarks>
    public static T Factorial<T>(int n)
    {
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
    /// Returns the mathematical constant Pi (π) converted to the specified numeric type.
    /// </summary>
    /// <typeparam name="T">The numeric type to convert Pi to.</typeparam>
    /// <returns>The value of Pi as type T.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pi (π) is a fundamental mathematical constant representing the ratio of a 
    /// circle's circumference to its diameter, approximately equal to 3.14159. It appears in many 
    /// mathematical formulas, especially those involving circles, waves, and periodic functions.
    /// </para>
    /// </remarks>
    public static T Pi<T>()
    {
        return GetNumericOperations<T>().FromDouble(Math.PI);
    }

    /// <summary>
    /// Returns positive infinity for the specified numeric type.
    /// </summary>
    /// <typeparam name="T">The numeric type for which to return positive infinity.</typeparam>
    /// <returns>A value representing positive infinity for the specified type.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Infinity represents a value that is larger than any finite number.
    /// This method provides a way to get the positive infinity value for different numeric types
    /// (like double, float, decimal) in a consistent way.
    /// 
    /// In mathematics and computing, infinity is used to represent unbounded values or the result
    /// of operations like division by zero.
    /// </para>
    /// </remarks>
    public static T PositiveInfinity<T>()
    {
        var numOps = GetNumericOperations<T>();

        // For floating-point types, return their specific infinity representation
        if (typeof(T) == typeof(double))
            return (T)(object)double.PositiveInfinity;
        if (typeof(T) == typeof(float))
            return (T)(object)float.PositiveInfinity;

        // For other types, return the maximum value as an approximation of infinity
        if (typeof(T) == typeof(decimal))
            return (T)(object)decimal.MaxValue;
        if (typeof(T) == typeof(int))
            return (T)(object)int.MaxValue;
        if (typeof(T) == typeof(long))
            return (T)(object)long.MaxValue;

        // Default fallback using the numeric operations
        return numOps.MaxValue;
    }

    /// <summary>
    /// Returns negative infinity for the specified numeric type.
    /// </summary>
    /// <typeparam name="T">The numeric type for which to return negative infinity.</typeparam>
    /// <returns>A value representing negative infinity for the specified type.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Negative infinity represents a value that is smaller than any finite number.
    /// This method provides a way to get the negative infinity value for different numeric types
    /// (like double, float, decimal) in a consistent way.
    /// 
    /// In mathematics and computing, negative infinity is used to represent unbounded negative values
    /// or the result of certain operations like dividing a negative number by zero.
    /// </para>
    /// </remarks>
    public static T NegativeInfinity<T>()
    {
        var numOps = GetNumericOperations<T>();

        // For floating-point types, return their specific negative infinity representation
        if (typeof(T) == typeof(double))
            return (T)(object)double.NegativeInfinity;
        if (typeof(T) == typeof(float))
            return (T)(object)float.NegativeInfinity;

        // For other types, return the minimum value as an approximation of negative infinity
        if (typeof(T) == typeof(decimal))
            return (T)(object)decimal.MinValue;
        if (typeof(T) == typeof(int))
            return (T)(object)int.MinValue;
        if (typeof(T) == typeof(long))
            return (T)(object)long.MinValue;

        // Default fallback using the numeric operations
        return numOps.MinValue;
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
    /// The input angle must be in radians, not degrees (2π radians = 360 degrees).
    /// </para>
    /// </remarks>
    public static T Sin<T>(T x)
    {
        return GetNumericOperations<T>().FromDouble(Math.Sin(Convert.ToDouble(x)));
    }

    /// <summary>
    /// Rounds a numeric value to the nearest integral value.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="value">The value to round.</param>
    /// <param name="midpointRounding">Specifies how to round when a value is midway between two other values. Default is ToEven.</param>
    /// <returns>The rounded value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method rounds a number to the nearest integer, with control over how midpoint values 
    /// (those exactly halfway between two integers) are handled. For example, 2.5 could round to either 2 or 3 
    /// depending on the midpoint rounding strategy. ToEven (also called "banker's rounding") rounds to the nearest 
    /// even number, while AwayFromZero always rounds up in magnitude.
    /// </para>
    /// </remarks>
    public static T Round<T>(T value, MidpointRounding midpointRounding = MidpointRounding.ToEven)
    {
        return GetNumericOperations<T>().FromDouble(Math.Round(Convert.ToDouble(value), midpointRounding));
    }

    /// <summary>
    /// Returns the largest integral value less than or equal to the specified number.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="value">The value to floor.</param>
    /// <returns>The largest integral value less than or equal to the specified number.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method rounds a number down to the nearest integer that's less than or equal to it.
    /// For example, Floor(3.7) returns 3, and Floor(-3.7) returns -4. This is useful when you need to ensure 
    /// a value doesn't exceed a certain threshold, or when working with indices that must stay within bounds.
    /// </para>
    /// </remarks>
    public static T Floor<T>(T value)
    {
        return GetNumericOperations<T>().FromDouble(Math.Floor(Convert.ToDouble(value)));
    }

    /// <summary>
    /// Returns the smallest integral value greater than or equal to the specified number.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="value">The value to ceiling.</param>
    /// <returns>The smallest integral value greater than or equal to the specified number.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method rounds a number up to the nearest integer that's greater than or equal to it.
    /// For example, Ceiling(3.2) returns 4, and Ceiling(-3.2) returns -3. This is useful when you need to ensure 
    /// you have enough resources to cover a fractional need, like calculating how many containers you need to store items.
    /// </para>
    /// </remarks>
    public static T Ceiling<T>(T value)
    {
        return GetNumericOperations<T>().FromDouble(Math.Ceiling(Convert.ToDouble(value)));
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
    /// The input angle must be in radians, not degrees (2π radians = 360 degrees).
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
    /// <b>For Beginners:</b> The base-2 logarithm (log₂) tells you what power you need to raise 2 to in order 
    /// to get a specific number. For example, log₂(8) = 3 because 2³ = 8. Base-2 logarithms are commonly 
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
    
        // ArcCos(x) = π/2 - ArcSin(x)
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
    /// and returns the corresponding angle in radians. For example, since sin(π/2) = 1, arcsin(1) = π/2.
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
    /// This implementation uses a Taylor series approximation, which is a mathematical technique 
    /// that represents a function as an infinite sum of terms. We use the first few terms to get 
    /// a good approximation of the arc tangent value.
    /// </para>
    /// </remarks>
    public static T ArcTan<T>(T x)
    {
        var numOps = GetNumericOperations<T>();
    
        // Use Taylor series approximation for ArcTan
        T _result = x;
        T _xPower = x;
        T _term = x;
        int _sign = 1;
    
        for (int n = 3; n <= 15; n += 2)
        {
            _sign = -_sign;
            _xPower = numOps.Multiply(_xPower, numOps.Multiply(x, x));
            _term = numOps.Divide(_xPower, numOps.FromDouble(n));
            _result = numOps.Add(_result, numOps.Multiply(numOps.FromDouble(_sign), _term));
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
            _numOps.Multiply(
                _numOps.Exp(_numOps.Negate(_numOps.Square(x))),
                _numOps.Add(_a1, 
                    _numOps.Multiply(_t, 
                        _numOps.Add(_a2, 
                            _numOps.Multiply(_t, 
                                _numOps.Add(_a3, 
                                    _numOps.Multiply(_t, 
                                        _numOps.Add(_a4, 
                                            _numOps.Multiply(_a5, _t))))))))));

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
    /// This method uses the formula: y-intercept = mean(y) - (coefficient₁ × mean(x₁) + coefficient₂ × mean(x₂) + ...)
    /// which ensures that the regression line passes through the point of means (the average of all data points).
    /// </para>
    /// <para>
    /// For example, in a house price prediction model, if you have features like square footage and 
    /// number of bedrooms, the y-intercept would be the baseline price of a house before considering 
    /// these features.
    /// </para>
    /// </remarks>
    public static T CalculateYIntercept<T>(Matrix<T> xMatrix, Vector<T> y, Vector<T> coefficients)
    {
        var _numOps = GetNumericOperations<T>();

        if (xMatrix.Rows != y.Length || xMatrix.Columns != coefficients.Length)
            throw new ArgumentException("Dimensions of xMatrix, y, and coefficients must be compatible.");

        T _yMean = y.Average();
        T _predictedSum = _numOps.Zero;

        for (int i = 0; i < xMatrix.Columns; i++)
        {
            T _xMean = xMatrix.GetColumn(i).Average();
            _predictedSum = _numOps.Add(_predictedSum, _numOps.Multiply(coefficients[i], _xMean));
        }

        return _numOps.Subtract(_yMean, _predictedSum);
    }
}