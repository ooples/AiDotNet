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

    public static bool IsInteger<T>(T value, INumericOperations<T> numOps)
    {
        // If the value is equal to its rounded value, it's an integer
        return numOps.Equals(value, numOps.Round(value));
    }

    public static bool AlmostEqual<T>(T a, T b, T tolerance, INumericOperations<T> numOps)
    {
        return numOps.LessThan(numOps.Abs(numOps.Subtract(a, b)), tolerance);
    }

    public static bool AlmostEqual<T>(T a, T b, INumericOperations<T> numOps)
    {
        return AlmostEqual(a, b, numOps.FromDouble(1e-8), numOps);
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