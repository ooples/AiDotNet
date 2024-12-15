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