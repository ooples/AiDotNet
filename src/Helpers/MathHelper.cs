namespace AiDotNet.Helpers;

public static class MathHelper
{
    public static INumericOperations<T> GetNumericOperations<T>()
    {
        if (typeof(T) == typeof(double))
            return (INumericOperations<T>)new DoubleOperations();
        else if (typeof(T) == typeof(Complex))
            return (INumericOperations<T>)new ComplexOperations();
        else if (typeof(T) == typeof(byte))
            return (INumericOperations<T>)new ByteOperations();
        else if (typeof(T) == typeof(uint))
            return (INumericOperations<T>)new UIntOperations();
        else if (typeof(T) == typeof(int))
            return (INumericOperations<T>)new Int32Operations();
        else if (typeof(T) == typeof(long))
            return (INumericOperations<T>)new Int64Operations();
        else
            throw new NotSupportedException($"Numeric operations for type {typeof(T)} are not supported.");
    }
}