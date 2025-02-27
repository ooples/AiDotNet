namespace AiDotNet.Extensions;

public static class NumericTypeExtensions
{
    public static bool IsRealType<T>()
    {
        return typeof(T) == typeof(double) ||
               typeof(T) == typeof(float) ||
               typeof(T) == typeof(decimal) ||
               typeof(T) == typeof(int) ||
               typeof(T) == typeof(long) ||
               typeof(T) == typeof(short) ||
               typeof(T) == typeof(byte) ||
               typeof(T) == typeof(sbyte) ||
               typeof(T) == typeof(uint) ||
               typeof(T) == typeof(ulong) ||
               typeof(T) == typeof(ushort);
    }

    public static bool IsComplexType<T>()
    {
        return typeof(T).IsGenericType &&
               typeof(T).GetGenericTypeDefinition() == typeof(Complex<>);
    }

    public static T ToRealOrComplex<T>(this Complex<T> complex)
    {
        if (IsRealType<T>())
        {
            return complex.Real;
        }
        else if (IsComplexType<T>())
        {
            return (T)(object)complex;
        }
        else
        {
            throw new InvalidOperationException($"Unsupported type: {typeof(T)}");
        }
    }
}