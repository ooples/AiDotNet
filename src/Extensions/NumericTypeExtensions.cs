namespace AiDotNet.Extensions;

/// <summary>
/// Provides extension methods for working with numeric types in AI and machine learning contexts.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class contains helper methods that let you check what kind of numbers
/// you're working with and convert between different number formats. In AI and machine learning,
/// we often need to work with both regular numbers (like 1, 2.5, -3) and complex numbers
/// (which have both a real and imaginary part, like 3+2i).
/// </para>
/// </remarks>
public static class NumericTypeExtensions
{
    /// <summary>
    /// Determines whether a type represents a real number.
    /// </summary>
    /// <typeparam name="T">The type to check.</typeparam>
    /// <returns>True if the type is a real numeric type; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Real numbers are the kind of numbers you're familiar with from everyday life - 
    /// whole numbers (like 1, 2, 3), fractions (like 0.5), and negative numbers (like -10). This method
    /// checks if the data type you're using represents these kinds of numbers.
    /// </para>
    /// <para>
    /// In programming, there are several types that can represent real numbers with different ranges and precision:
    /// <list type="bullet">
    ///   <item><description>int, long, short: Whole numbers only</description></item>
    ///   <item><description>float, double, decimal: Can represent fractions</description></item>
    ///   <item><description>byte, sbyte, uint, ulong, ushort: Various specialized number types</description></item>
    /// </list>
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Determines whether a type represents a complex number.
    /// </summary>
    /// <typeparam name="T">The type to check.</typeparam>
    /// <returns>True if the type is a complex numeric type; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Complex numbers are numbers that have two parts: a real part and an imaginary part.
    /// The imaginary part involves the square root of -1, which is represented by the symbol 'i' in mathematics.
    /// </para>
    /// <para>
    /// For example, 3+2i is a complex number where 3 is the real part and 2i is the imaginary part.
    /// Complex numbers are important in many areas of science and engineering, including signal processing,
    /// electrical engineering, quantum physics, and certain AI algorithms.
    /// </para>
    /// <para>
    /// This method checks if the data type you're using represents a complex number in our library.
    /// </para>
    /// </remarks>
    public static bool IsComplexType<T>()
    {
        return typeof(T).IsGenericType &&
               typeof(T).GetGenericTypeDefinition() == typeof(Complex<>);
    }

    /// <summary>
    /// Converts a complex number to either its real part or keeps it as a complex number,
    /// depending on the target type.
    /// </summary>
    /// <typeparam name="T">The target type for conversion.</typeparam>
    /// <param name="complex">The complex number to convert.</param>
    /// <returns>
    /// If T is a real type, returns the real part of the complex number.
    /// If T is a complex type, returns the complex number itself.
    /// </returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when T is neither a real nor a complex type.
    /// </exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method helps you convert between complex numbers and regular (real) numbers.
    /// </para>
    /// <para>
    /// If you're trying to convert to a real number type (like double or int), this method will extract just
    /// the real part of the complex number and discard the imaginary part. For example, if you have the complex
    /// number 3+2i and convert it to a double, you'll get just 3.
    /// </para>
    /// <para>
    /// If you're converting to another complex number type, it will keep both the real and imaginary parts.
    /// </para>
    /// <para>
    /// This is useful when you have algorithms that can work with either real or complex numbers, and you
    /// need to ensure the output is in the correct format.
    /// </para>
    /// </remarks>
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
