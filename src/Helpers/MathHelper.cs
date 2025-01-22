namespace AiDotNet.Helpers;

public static class MathHelper
{
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

    public static T Clamp<T>(T value, T min, T max)
    {
        var numOps = GetNumericOperations<T>();
        if (numOps.LessThan(value, min))
            return min;
        if (numOps.GreaterThan(value, max))
            return max;

        return value;
    }

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

    public static T Reciprocal<T>(T value)
    {
        var numOps = GetNumericOperations<T>();
        if (numOps.Equals(value, numOps.Zero))
        {
            throw new DivideByZeroException("Cannot calculate reciprocal of zero.");
        }

        return numOps.Divide(numOps.One, value);
    }

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

    public static bool IsInteger<T>(T value)
    {
        // If the value is equal to its rounded value, it's an integer
        var numOps = GetNumericOperations<T>();
        return numOps.Equals(value, numOps.Round(value));
    }

    public static bool AlmostEqual<T>(T a, T b, T tolerance)
    {
        var numOps = GetNumericOperations<T>();
        return numOps.LessThan(numOps.Abs(numOps.Subtract(a, b)), tolerance);
    }

    public static bool AlmostEqual<T>(T a, T b)
    {
        var numOps = GetNumericOperations<T>();
        return AlmostEqual(a, b, numOps.FromDouble(1e-8));
    }

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

    private static T BesselJSeries<T>(T nu, T x)
    {
        var numOps = GetNumericOperations<T>();
        T sum = numOps.Zero;
        T factorial = numOps.One;
        T xOver2 = numOps.Divide(x, numOps.FromDouble(2));
        T xOver2Squared = numOps.Square(xOver2);
        T term = numOps.One;

        for (int m = 0; m <= 50; m++)  // Increased max terms for better accuracy
        {
            if (m > 0)
            {
                factorial = numOps.Multiply(factorial, numOps.FromDouble(m));
                term = numOps.Divide(term, factorial);
                term = numOps.Multiply(term, xOver2Squared);
            }

            T numerator = numOps.Power(numOps.Negate(numOps.One), numOps.FromDouble(m));
            T denominator = numOps.Multiply(factorial, Gamma(numOps.Add(numOps.FromDouble(m), numOps.Add(nu, numOps.One))));
        
            T summand = numOps.Multiply(numerator, numOps.Divide(numOps.Power(xOver2, numOps.Add(numOps.FromDouble(2 * m), nu)), denominator));
            sum = numOps.Add(sum, summand);

            if (numOps.LessThan(numOps.Abs(summand), numOps.FromDouble(1e-15)))
            {
                break;
            }
        }

        return sum;
    }

    private static T BesselJAsymptotic<T>(T nu, T x)
    {
        var numOps = GetNumericOperations<T>();
        T mu = numOps.Subtract(numOps.Multiply(nu, nu), numOps.FromDouble(0.25));
        T theta = numOps.Subtract(x, numOps.Multiply(numOps.FromDouble(0.25 * Math.PI), numOps.Add(numOps.Multiply(numOps.FromDouble(2), nu), numOps.One)));

        T p = numOps.One;
        T q = numOps.Divide(mu, numOps.Multiply(numOps.FromDouble(8), x));

        T cosTheta = Cos(theta);
        T sinTheta = Sin(theta);

        T sqrtX = numOps.Sqrt(x);
        T sqrtPi = numOps.Sqrt(numOps.FromDouble(Math.PI));
        T factor = numOps.Divide(numOps.Sqrt(numOps.FromDouble(2)), numOps.Multiply(sqrtPi, sqrtX));

        return numOps.Multiply(factor, numOps.Add(numOps.Multiply(p, cosTheta), numOps.Multiply(q, sinTheta)));
    }

    private static T BesselJRecurrence<T>(T nu, T x)
    {
        var numOps = GetNumericOperations<T>();
        int n = (int)Math.Ceiling(Convert.ToDouble(nu));
        T nuInt = numOps.FromDouble(n);

        T jn = BesselJAsymptotic(nuInt, x);
        T jnMinus1 = BesselJAsymptotic(numOps.Subtract(nuInt, numOps.One), x);

        for (int k = n - 1; k >= 0; k--)
        {
            T jnMinus2 = numOps.Subtract(
                numOps.Multiply(numOps.FromDouble(2 * k + 2), numOps.Divide(jnMinus1, x)),
                jn
            );
            jn = jnMinus1;
            jnMinus1 = jnMinus2;
        }

        if (numOps.Equals(nu, nuInt))
        {
            return jn;
        }

        // Interpolate for non-integer nu
        T jnPlus1 = numOps.Subtract(
            numOps.Multiply(numOps.FromDouble(2 * n), numOps.Divide(jn, x)),
            jnMinus1
        );
        T t = numOps.Subtract(nu, numOps.Round(nu));
        return numOps.Add(numOps.Multiply(jn, numOps.Subtract(numOps.One, t)), numOps.Multiply(jnPlus1, t));
    }

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

    public static T Pi<T>()
    {
        return GetNumericOperations<T>().FromDouble(Math.PI);
    }

    public static T Sin<T>(T x)
    {
        return GetNumericOperations<T>().FromDouble(Math.Sin(Convert.ToDouble(x)));
    }

    public static T Cos<T>(T x)
    {
        return GetNumericOperations<T>().FromDouble(Math.Cos(Convert.ToDouble(x)));
    }

    public static T Tanh<T>(T x)
    {
        var numOps = GetNumericOperations<T>();
        T exp2x = numOps.Exp(numOps.Multiply(numOps.FromDouble(2), x));
        return numOps.Divide(
            numOps.Subtract(exp2x, numOps.One),
            numOps.Add(exp2x, numOps.One)
        );
    }

    public static double Log2(double x)
    {
        if (x <= 0)
            throw new ArgumentOutOfRangeException(nameof(x), "Logarithm is undefined for non-positive numbers.");
        return Math.Log(x) / Math.Log(2);
    }

    public static T Min<T>(T a, T b)
    {
        return GetNumericOperations<T>().LessThan(a, b) ? a : b;
    }

    public static T Max<T>(T a, T b)
    {
        return GetNumericOperations<T>().GreaterThan(a, b) ? a : b;
    }

    public static T ArcCos<T>(T x)
    {
        var numOps = GetNumericOperations<T>();
    
        // ArcCos(x) = π/2 - ArcSin(x)
        var arcSin = MathHelper.ArcSin(x);
        var halfPi = numOps.Divide(Pi<T>(), numOps.FromDouble(2.0));
    
        return numOps.Subtract(halfPi, arcSin);
    }

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

    public static T ArcTan<T>(T x)
    {
        var numOps = GetNumericOperations<T>();
    
        // Use Taylor series approximation for ArcTan
        T result = x;
        T xPower = x;
        T term = x;
        int sign = 1;
    
        for (int n = 3; n <= 15; n += 2)
        {
            sign = -sign;
            xPower = numOps.Multiply(xPower, numOps.Multiply(x, x));
            term = numOps.Divide(xPower, numOps.FromDouble(n));
            result = numOps.Add(result, numOps.Multiply(numOps.FromDouble(sign), term));
        }
    
        return result;
    }

    public static T Erf<T>(T x)
    {
        var NumOps = GetNumericOperations<T>();
        T sign = NumOps.GreaterThanOrEquals(x, NumOps.Zero) ? NumOps.FromDouble(1) : NumOps.FromDouble(-1);
        x = NumOps.Abs(x);

        T a1 = NumOps.FromDouble(0.254829592);
        T a2 = NumOps.FromDouble(-0.284496736);
        T a3 = NumOps.FromDouble(1.421413741);
        T a4 = NumOps.FromDouble(-1.453152027);
        T a5 = NumOps.FromDouble(1.061405429);
        T p = NumOps.FromDouble(0.3275911);

        T t = NumOps.Divide(NumOps.FromDouble(1), NumOps.Add(NumOps.FromDouble(1), NumOps.Multiply(p, x)));
        T y = NumOps.Subtract(NumOps.FromDouble(1), 
            NumOps.Multiply(
                NumOps.Exp(NumOps.Negate(NumOps.Square(x))),
                NumOps.Add(a1, 
                    NumOps.Multiply(t, 
                        NumOps.Add(a2, 
                            NumOps.Multiply(t, 
                                NumOps.Add(a3, 
                                    NumOps.Multiply(t, 
                                        NumOps.Add(a4, 
                                            NumOps.Multiply(a5, t))))))))));

        return NumOps.Multiply(sign, y);
    }

    public static T CalculateYIntercept<T>(Matrix<T> xMatrix, Vector<T> y, Vector<T> coefficients)
    {
        var numOps = GetNumericOperations<T>();

        if (xMatrix.Rows != y.Length || xMatrix.Columns != coefficients.Length)
            throw new ArgumentException("Dimensions of xMatrix, y, and coefficients must be compatible.");

        T yMean = y.Average();
        T predictedSum = numOps.Zero;

        for (int i = 0; i < xMatrix.Columns; i++)
        {
            T xMean = xMatrix.GetColumn(i).Average();
            predictedSum = numOps.Add(predictedSum, numOps.Multiply(coefficients[i], xMean));
        }

        return numOps.Subtract(yMean, predictedSum);
    }
}