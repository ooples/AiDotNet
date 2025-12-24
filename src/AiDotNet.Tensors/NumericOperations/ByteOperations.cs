using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Operators;

namespace AiDotNet.Tensors.NumericOperations;

/// <summary>
/// Provides mathematical operations for the byte data type.
/// </summary>
/// <remarks>
/// <para>
/// This class implements the INumericOperations interface for the byte data type, providing
/// basic arithmetic operations, comparison methods, and mathematical functions. Due to the limited
/// range of the byte type (0-255), some operations may result in overflow or underflow, which
/// will wrap around according to the byte data type's behavior.
/// </para>
/// <para><b>For Beginners:</b> This class lets you perform math operations on byte values.
///
/// A byte is a very small number type that can only hold values from 0 to 255.
///
/// Important things to know about bytes:
/// - When math operations result in values outside the 0-255 range, they "wrap around"
/// - For example, 255 + 10 = 9 (not 265) because it exceeds the maximum and wraps around
/// - This class handles all the math operations for bytes in AI.NET
///
/// Think of this like a car odometer with only 3 digits - after 999 miles, it rolls over to 000.
/// </para>
/// </remarks>
public class ByteOperations : INumericOperations<byte>
{
    /// <summary>
    /// Adds two byte values together.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    /// <returns>The sum of the two values, casted to a byte.</returns>
    /// <remarks>
    /// <para>
    /// If the sum exceeds the maximum value of a byte (255), the result will wrap around.
    /// For example, 200 + 100 = 44 (as a byte) because 300 exceeds 255 and wraps around.
    /// </para>
    /// <para><b>For Beginners:</b> This method adds two byte numbers together.
    ///
    /// Because bytes can only hold values up to 255:
    /// - Normal additions like 5 + 10 = 15 work as expected
    /// - But 250 + 10 = 4 (not 260) because it exceeds 255 and wraps around
    ///
    /// This wrapping behavior is important to understand when working with bytes.
    /// </para>
    /// </remarks>
    public byte Add(byte a, byte b) => (byte)(a + b);

    /// <summary>
    /// Subtracts the second byte value from the first.
    /// </summary>
    /// <param name="a">The value to subtract from.</param>
    /// <param name="b">The value to subtract.</param>
    /// <returns>The difference between the two values, casted to a byte.</returns>
    /// <remarks>
    /// <para>
    /// If the result is negative, the value will wrap around. For example, 10 - 20 = 246 (as a byte)
    /// because -10 wraps around to 246 in the byte range.
    /// </para>
    /// <para><b>For Beginners:</b> This method subtracts one byte number from another.
    /// 
    /// Because bytes can't hold negative values:
    /// - Normal subtractions like 20 - 10 = 10 work as expected
    /// - But 10 - 20 = 246 (not -10) because negative values wrap around from the other end
    /// 
    /// The formula for finding the wrapped value is: 256 + negative_result
    /// </para>
    /// </remarks>
    public byte Subtract(byte a, byte b) => (byte)(a - b);

    /// <summary>
    /// Multiplies two byte values together.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    /// <returns>The product of the two values, casted to a byte.</returns>
    /// <remarks>
    /// <para>
    /// If the product exceeds the maximum value of a byte (255), the result will wrap around.
    /// For example, 20 * 20 = 144 (as a byte) because 400 exceeds 255 and wraps around.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies two byte numbers together.
    /// 
    /// Because bytes have a small range:
    /// - Small multiplications like 2 * 3 = 6 work as expected
    /// - But larger ones like 16 * 16 = 0 (not 256) because it wraps around
    /// 
    /// Be careful with multiplication as it's easy to exceed the byte range.
    /// </para>
    /// </remarks>
    public byte Multiply(byte a, byte b) => (byte)(a * b);

    /// <summary>
    /// Divides the first byte value by the second.
    /// </summary>
    /// <param name="a">The dividend (value being divided).</param>
    /// <param name="b">The divisor (value to divide by).</param>
    /// <returns>The quotient of the division, casted to a byte.</returns>
    /// <remarks>
    /// <para>
    /// This performs integer division, so any fractional part of the result is truncated.
    /// For example, 5 / 2 = 2 (not 2.5). Division by zero will throw an exception.
    /// </para>
    /// <para><b>For Beginners:</b> This method divides one byte by another.
    /// 
    /// Important things to know:
    /// - This is integer division, so 5 / 2 = 2 (the decimal part is dropped)
    /// - Dividing by zero will cause an error
    /// - The result always fits within the byte range
    /// 
    /// This works like division with whole numbers in elementary math.
    /// </para>
    /// </remarks>
    public byte Divide(byte a, byte b) => (byte)(a / b);

    /// <summary>
    /// Negates the specified byte value.
    /// </summary>
    /// <param name="a">The value to negate.</param>
    /// <returns>The negated value, casted to a byte.</returns>
    /// <remarks>
    /// <para>
    /// Due to the unsigned nature of bytes, negating a byte value results in a wrap-around.
    /// For example, negating 1 results in 255, and negating 10 results in 246.
    /// </para>
    /// <para><b>For Beginners:</b> This method tries to reverse the sign of a byte.
    /// 
    /// Since bytes can't be negative:
    /// - Negating 5 gives 251 (not -5) due to wrapping
    /// - The formula is: 256 - value (when value > 0)
    /// 
    /// This mainly exists to fulfill the interface requirements and has specialized behavior for bytes.
    /// </para>
    /// </remarks>
    public byte Negate(byte a) => (byte)-a;

    /// <summary>
    /// Gets the byte representation of zero.
    /// </summary>
    /// <value>The value 0 as a byte.</value>
    /// <remarks>
    /// <para>
    /// This property returns the byte representation of the value zero, which is simply 0.
    /// It is often used as a neutral element for addition.
    /// </para>
    /// <para><b>For Beginners:</b> This property provides the value zero as a byte.
    /// 
    /// Zero is a special value in mathematics:
    /// - Adding zero to any number gives the same number
    /// - It's used as a starting point in many algorithms
    /// 
    /// This property gives you a zero that matches the byte type.
    /// </para>
    /// </remarks>
    public byte Zero => 0;

    /// <summary>
    /// Gets the byte representation of one.
    /// </summary>
    /// <value>The value 1 as a byte.</value>
    /// <remarks>
    /// <para>
    /// This property returns the byte representation of the value one, which is 1.
    /// It is often used as a neutral element for multiplication.
    /// </para>
    /// <para><b>For Beginners:</b> This property provides the value one as a byte.
    /// 
    /// One is a special value in mathematics:
    /// - Multiplying any number by one gives the same number
    /// - It's useful as a starting point or increment value
    /// 
    /// This property gives you a one that matches the byte type.
    /// </para>
    /// </remarks>
    public byte One => 1;

    /// <summary>
    /// Calculates the square root of a byte value.
    /// </summary>
    /// <param name="value">The value to calculate the square root of.</param>
    /// <returns>The square root of the specified value, casted to a byte.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the square root using Math.Sqrt and then casts the result to a byte.
    /// Since the result is cast to a byte, any fractional part is truncated.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the square root of a byte number.
    /// 
    /// For example:
    /// - The square root of 4 is 2
    /// - The square root of 9 is 3
    /// 
    /// Because bytes can't have decimal parts:
    /// - Square root of 5 becomes 2 (not 2.236...)
    /// - The decimal part is simply removed
    /// 
    /// This works for all values from 0 to 255.
    /// </para>
    /// </remarks>
    public byte Sqrt(byte value) => (byte)Math.Sqrt(value);

    /// <summary>
    /// Converts a double value to a byte.
    /// </summary>
    /// <param name="value">The double value to convert.</param>
    /// <returns>The double value converted to a byte.</returns>
    /// <remarks>
    /// <para>
    /// This method casts a double value to a byte. If the value is outside the range of a byte (0-255),
    /// it will be truncated. Fractional parts are also truncated.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts a decimal number to a byte.
    /// 
    /// When converting:
    /// - The decimal part is dropped (3.7 becomes 3)
    /// - Values below 0 become 0
    /// - Values above 255 become a wrapped value (usually unexpected)
    /// 
    /// For example, 300.5 would become 44 as a byte (300 - 256 = 44).
    /// </para>
    /// </remarks>
    public byte FromDouble(double value) => (byte)value;

    /// <summary>
    /// Determines whether the first byte value is greater than the second.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns>true if the first value is greater than the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two byte values and returns true if the first is greater than the second.
    /// Since bytes are unsigned, the comparison is straightforward.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is larger than the second.
    /// 
    /// For example:
    /// - 10 > 5 returns true
    /// - 5 > 10 returns false
    /// - 5 > 5 returns false
    /// 
    /// This is a simple comparison operation used in many algorithms.
    /// </para>
    /// </remarks>
    public bool GreaterThan(byte a, byte b) => a > b;

    /// <summary>
    /// Determines whether the first byte value is less than the second.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns>true if the first value is less than the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two byte values and returns true if the first is less than the second.
    /// Since bytes are unsigned, the comparison is straightforward.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is smaller than the second.
    /// 
    /// For example:
    /// - 5 < 10 returns true
    /// - 10 < 5 returns false
    /// - 5 < 5 returns false
    /// 
    /// This is a simple comparison operation used in many algorithms.
    /// </para>
    /// </remarks>
    public bool LessThan(byte a, byte b) => a < b;

    /// <summary>
    /// Returns the absolute value of a byte.
    /// </summary>
    /// <param name="value">The byte value.</param>
    /// <returns>The input value (since bytes are always non-negative).</returns>
    /// <remarks>
    /// <para>
    /// Since bytes are unsigned and can only hold positive values, this method simply returns the input value.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides the absolute (positive) value of a number.
    /// 
    /// For regular numbers:
    /// - The absolute value of 5 is 5
    /// - The absolute value of -5 is 5
    /// 
    /// For bytes, which can't be negative, this simply returns the same value.
    /// This method exists to maintain compatibility with other numeric types.
    /// </para>
    /// </remarks>
    public byte Abs(byte value) => value;

    /// <summary>
    /// Squares the specified byte value.
    /// </summary>
    /// <param name="value">The value to square.</param>
    /// <returns>The square of the specified value, casted to a byte.</returns>
    /// <remarks>
    /// <para>
    /// This method multiplies the value by itself. If the result exceeds the maximum value of a byte (255),
    /// the result will wrap around.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies a number by itself.
    /// 
    /// For example:
    /// - Square of 2 is 4 (2 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 2)
    /// - Square of 10 is 100 (10 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 10)
    /// 
    /// Because of byte limits:
    /// - Square of 16 is 0 (16 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 16 = 256, which wraps to 0)
    /// - Any value of 16 or higher will wrap around when squared
    /// 
    /// Be careful when squaring larger byte values.
    /// </para>
    /// </remarks>
    public byte Square(byte value) => Multiply(value, value);

    /// <summary>
    /// Calculates e raised to the specified power.
    /// </summary>
    /// <param name="value">The power to raise e to.</param>
    /// <returns>e raised to the specified power, casted to a byte.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the exponential function (e^value) and rounds the result to the nearest integer.
    /// If the result exceeds 255, it is capped at 255.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the mathematical constant e (ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â 2.718) raised to a power.
    /// 
    /// For example:
    /// - e^1 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  2.718 (rounded to 3 as a byte)
    /// - e^2 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  7.389 (rounded to 7 as a byte)
    /// - e^5 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  148.413 (rounded to 148 as a byte)
    /// 
    /// The result is limited to 255 (maximum byte value).
    /// This function grows very quickly, so even moderate input values will reach the maximum.
    /// </para>
    /// </remarks>
    public byte Exp(byte value) => (byte)Math.Min(255, Math.Round(Math.Exp(value)));

    /// <summary>
    /// Determines whether two byte values are equal.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns>true if the values are equal; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two byte values and returns true if they are equal.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if two numbers have exactly the same value.
    /// 
    /// For example:
    /// - 5 equals 5 returns true
    /// - 10 equals 5 returns false
    /// 
    /// This is a basic comparison operation used in many algorithms.
    /// </para>
    /// </remarks>
    public bool Equals(byte a, byte b) => a == b;

    public int Compare(byte a, byte b) => a.CompareTo(b);

    /// <summary>
    /// Raises a byte value to the specified power.
    /// </summary>
    /// <param name="baseValue">The base value.</param>
    /// <param name="exponent">The exponent.</param>
    /// <returns>The base value raised to the specified power, casted to a byte.</returns>
    /// <remarks>
    /// <para>
    /// This method uses Math.Pow to calculate the power and then casts the result to a byte.
    /// If the result exceeds the maximum value of a byte (255), it will wrap around.
    /// </para>
    /// <para><b>For Beginners:</b> This method raises one number to the power of another.
    /// 
    /// For example:
    /// - 2 raised to power 3 is 8 (2ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â³ = 2 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 2 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 2 = 8)
    /// - 3 raised to power 2 is 9 (3ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â² = 3 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 3 = 9)
    /// 
    /// Because of byte limits:
    /// - 2 raised to power 8 is 0 (28 = 256, which wraps to 0)
    /// - Results above 255 will wrap around
    /// 
    /// Powers grow very quickly, so be cautious with larger values.
    /// </para>
    /// </remarks>
    public byte Power(byte baseValue, byte exponent) => (byte)Math.Pow(baseValue, exponent);

    /// <summary>
    /// Calculates the natural logarithm of a byte value.
    /// </summary>
    /// <param name="value">The value to calculate the natural logarithm of.</param>
    /// <returns>The natural logarithm of the specified value, casted to a byte.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the natural logarithm (base e) of the specified value.
    /// The result is cast to a byte, so any fractional part is truncated. Log of 0 will result in an exception.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the natural logarithm of a number.
    /// 
    /// The natural logarithm answers the question: "To what power must e be raised to get this number?"
    /// 
    /// For example:
    /// - Log of 1 is 0 (e^0 = 1)
    /// - Log of 3 is approximately 1.099 (truncated to 1 as a byte)
    /// - Log of 7 is approximately 1.946 (truncated to 1 as a byte)
    /// 
    /// Important notes:
    /// - Log of 0 causes an error
    /// - Most logarithm results will be very small as bytes
    /// - The decimal part is removed when converting to byte
    /// </para>
    /// </remarks>
    public byte Log(byte value) => (byte)Math.Log(value);

    /// <summary>
    /// Determines whether the first byte value is greater than or equal to the second.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns>true if the first value is greater than or equal to the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two byte values and returns true if the first is greater than or equal to the second.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is larger than or the same as the second.
    /// 
    /// For example:
    /// - 10 >= 5 returns true
    /// - 5 >= 10 returns false
    /// - 5 >= 5 returns true
    /// 
    /// This is a simple comparison operation used in many algorithms.
    /// </para>
    /// </remarks>
    public bool GreaterThanOrEquals(byte a, byte b) => a >= b;

    /// <summary>
    /// Determines whether the first byte value is less than or equal to the second.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns>true if the first value is less than or equal to the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two byte values and returns true if the first is less than or equal to the second.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is smaller than or the same as the second.
    /// 
    /// For example:
    /// - 5 <= 10 returns true
    /// - 10 <= 5 returns false
    /// - 5 <= 5 returns true
    /// 
    /// This is a simple comparison operation used in many algorithms.
    /// </para>
    /// </remarks>
    public bool LessThanOrEquals(byte a, byte b) => a <= b;

    /// <summary>
    /// Converts a byte value to a 32-bit integer.
    /// </summary>
    /// <param name="value">The byte value to convert.</param>
    /// <returns>The byte value as a 32-bit integer.</returns>
    /// <remarks>
    /// <para>
    /// This method converts a byte to an Int32. Since a byte can only hold values from 0 to 255,
    /// the conversion is always safe and will never overflow.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts a byte to a regular integer.
    /// 
    /// The conversion is straightforward:
    /// - A byte like 5 becomes the integer 5
    /// - A byte like 255 becomes the integer 255
    /// 
    /// This is useful when you need to use a byte value with operations that expect a larger number type.
    /// </para>
    /// </remarks>
    public int ToInt32(byte value) => value;

    /// <summary>
    /// Rounds a byte value to the nearest integer (which is itself, since bytes are already integers).
    /// </summary>
    /// <param name="value">The byte value to round.</param>
    /// <returns>The input value (since bytes are already integers).</returns>
    /// <remarks>
    /// <para>
    /// Since bytes are already integer values, this method simply returns the input value.
    /// </para>
    /// <para><b>For Beginners:</b> This method rounds a number to the nearest whole number.
    /// 
    /// Since bytes are already whole numbers, this simply returns the original value.
    /// 
    /// This method exists to maintain compatibility with other numeric types that need rounding.
    /// </para>
    /// </remarks>
    public byte Round(byte value) => value;

    public byte Floor(byte value) => value;
    public byte Ceiling(byte value) => value;
    public byte Frac(byte value) => 0;

    /// <summary>
    /// Returns the sine of the specified value (truncated to integer).
    /// </summary>
    public byte Sin(byte value) => (byte)Math.Sin(value);

    /// <summary>
    /// Returns the cosine of the specified value (truncated to integer).
    /// </summary>
    public byte Cos(byte value) => (byte)Math.Cos(value);


    /// <summary>
    /// Gets the minimum value a byte can represent.
    /// </summary>
    /// <value>The minimum value of a byte, which is 0.</value>
    /// <remarks>
    /// <para>
    /// This property returns the minimum value that can be represented by a byte, which is 0.
    /// </para>
    /// <para><b>For Beginners:</b> This property gives you the smallest possible byte value.
    /// 
    /// For bytes, the minimum value is 0.
    /// 
    /// This is useful when you need to work with the full range of byte values
    /// or need to check against the minimum possible value.
    /// </para>
    /// </remarks>
    public byte MinValue => byte.MinValue;

    /// <summary>
    /// Gets the maximum value a byte can represent.
    /// </summary>
    /// <value>The maximum value of a byte, which is 255.</value>
    /// <remarks>
    /// <para>
    /// This property returns the maximum value that can be represented by a byte, which is 255.
    /// </para>
    /// <para><b>For Beginners:</b> This property gives you the largest possible byte value.
    /// 
    /// For bytes, the maximum value is 255.
    /// 
    /// This is useful when you need to work with the full range of byte values
    /// or need to check against the maximum possible value.
    /// </para>
    /// </remarks>
    public byte MaxValue => byte.MaxValue;

    /// <summary>
    /// Determines whether the specified byte value is NaN (Not a Number).
    /// </summary>
    /// <param name="value">The byte value to check.</param>
    /// <returns>Always returns false, as byte values cannot represent NaN.</returns>
    /// <remarks>
    /// <para>
    /// Bytes cannot represent special values like NaN, so this method always returns false.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if a value is "Not a Number" (NaN).
    /// 
    /// For floating-point types like float or double, certain operations can result in NaN.
    /// However, bytes cannot represent NaN, so this method always returns false.
    /// 
    /// This method exists to maintain compatibility with other numeric types.
    /// </para>
    /// </remarks>
    public bool IsNaN(byte value) => false;

    /// <summary>
    /// Determines whether the specified byte value is infinity.
    /// </summary>
    /// <param name="value">The byte value to check.</param>
    /// <returns>Always returns false, as byte values cannot represent infinity.</returns>
    /// <remarks>
    /// <para>
    /// Bytes cannot represent special values like infinity, so this method always returns false.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if a value is infinity.
    /// 
    /// For floating-point types like float or double, certain operations can result in infinity.
    /// However, bytes cannot represent infinity, so this method always returns false.
    /// 
    /// This method exists to maintain compatibility with other numeric types.
    /// </para>
    /// </remarks>
    public bool IsInfinity(byte value) => false;

    /// <summary>
    /// Returns the sign of the specified value as a byte.
    /// </summary>
    /// <param name="value">The value to check.</param>
    /// <returns>1 if the value is positive; 0 if the value is zero.</returns>
    /// <remarks>
    /// <para>
    /// Since bytes are unsigned, this method returns 1 for any positive value and 0 for zero.
    /// There is no negative representation in bytes.
    /// </para>
    /// <para><b>For Beginners:</b> This method determines the sign of a number.
    /// 
    /// For regular numbers, the sign function returns:
    /// - 1 for positive numbers
    /// - 0 for zero
    /// - -1 for negative numbers
    /// 
    /// But since bytes can't be negative, this method only returns:
    /// - 1 for values greater than 0
    /// - 0 for the value 0
    /// 
    /// This is useful in algorithms that need to know the direction or sign of a value.
    /// </para>
    /// </remarks>
    public byte SignOrZero(byte value)
    {
        if (value > 0) return 1;
        return 0;
    }


    /// <summary>
    /// Gets the number of bits used for precision in byte (8 bits).
    /// </summary>
    public int PrecisionBits => 8;

    /// <summary>
    /// Converts a byte value to float (FP32) precision.
    /// </summary>
    /// <param name="value">The byte value to convert.</param>
    /// <returns>The value as a float.</returns>
    public float ToFloat(byte value) => (float)value;

    /// <summary>
    /// Converts a float value to byte.
    /// </summary>
    /// <param name="value">The float value to convert.</param>
    /// <returns>The value as a byte.</returns>
    /// <remarks>
    /// This conversion will round the float to the nearest integer and clamp it to the byte range [0, 255].
    /// </remarks>
    public byte FromFloat(float value) => (byte)MathExtensions.Clamp((int)Math.Round(value), byte.MinValue, byte.MaxValue);

    /// <summary>
    /// Converts a byte value to Half (FP16) precision.
    /// </summary>
    /// <param name="value">The byte value to convert.</param>
    /// <returns>The value as a Half.</returns>
    public Half ToHalf(byte value) => (Half)value;

    /// <summary>
    /// Converts a Half value to byte.
    /// </summary>
    /// <param name="value">The Half value to convert.</param>
    /// <returns>The value as a byte.</returns>
    /// <remarks>
    /// This conversion will round the Half to the nearest integer and clamp it to the byte range [0, 255].
    /// </remarks>
    public byte FromHalf(Half value) => (byte)MathExtensions.Clamp((int)Math.Round((float)value), byte.MinValue, byte.MaxValue);

    /// <summary>
    /// Converts a byte value to double (FP64) precision.
    /// </summary>
    /// <param name="value">The byte value to convert.</param>
    /// <returns>The value as a double.</returns>
    public double ToDouble(byte value) => (double)value;

    /// <inheritdoc/>
    public bool SupportsCpuAcceleration => false;

    /// <inheritdoc/>
    public bool SupportsGpuAcceleration => false;

    #region IVectorizedOperations<byte> Implementation

    /// <summary>
    /// Performs element-wise addition using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public void Add(ReadOnlySpan<byte> x, ReadOnlySpan<byte> y, Span<byte> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<AddOperatorByte>(x, y, destination);

    /// <summary>
    /// Performs element-wise subtraction using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public void Subtract(ReadOnlySpan<byte> x, ReadOnlySpan<byte> y, Span<byte> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<SubtractOperatorByte>(x, y, destination);

    /// <summary>
    /// Performs element-wise multiplication using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public void Multiply(ReadOnlySpan<byte> x, ReadOnlySpan<byte> y, Span<byte> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<MultiplyOperatorByte>(x, y, destination);

    /// <summary>
    /// Performs element-wise division using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public void Divide(ReadOnlySpan<byte> x, ReadOnlySpan<byte> y, Span<byte> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<DivideOperatorByte>(x, y, destination);

    /// <summary>
    /// Computes dot product using sequential loops.
    /// </summary>
    public byte Dot(ReadOnlySpan<byte> x, ReadOnlySpan<byte> y)
        => VectorizedOperationsFallback.Dot(this, x, y);

    /// <summary>
    /// Computes sum using sequential loops.
    /// </summary>
    public byte Sum(ReadOnlySpan<byte> x)
        => VectorizedOperationsFallback.Sum(this, x);

    /// <summary>
    /// Finds maximum using sequential loops.
    /// </summary>
    public byte Max(ReadOnlySpan<byte> x)
        => VectorizedOperationsFallback.Max(this, x);

    /// <summary>
    /// Finds minimum using sequential loops.
    /// </summary>
    public byte Min(ReadOnlySpan<byte> x)
        => VectorizedOperationsFallback.Min(this, x);

    /// <summary>
    /// Transcendental operations are not supported for byte type.
    /// </summary>
    /// <exception cref="NotSupportedException">Always thrown. Exp produces misleading results for bytes (range 0-255).</exception>
    public void Exp(ReadOnlySpan<byte> x, Span<byte> destination)
        => throw new NotSupportedException("Transcendental operations (Exp) are not meaningful for byte type. Use float or double instead.");

    /// <summary>
    /// Transcendental operations are not supported for byte type.
    /// </summary>
    /// <exception cref="NotSupportedException">Always thrown. Log produces misleading results for bytes (only 0-7 possible).</exception>
    public void Log(ReadOnlySpan<byte> x, Span<byte> destination)
        => throw new NotSupportedException("Transcendental operations (Log) are not meaningful for byte type. Use float or double instead.");

    /// <summary>
    /// Transcendental operations are not supported for byte type.
    /// </summary>
    /// <exception cref="NotSupportedException">Always thrown. Tanh produces only 0 or 1 for bytes.</exception>
    public void Tanh(ReadOnlySpan<byte> x, Span<byte> destination)
        => throw new NotSupportedException("Transcendental operations (Tanh) are not meaningful for byte type. Use float or double instead.");

    /// <summary>
    /// Transcendental operations are not supported for byte type.
    /// </summary>
    /// <exception cref="NotSupportedException">Always thrown. Sigmoid saturates for byte inputs.</exception>
    public void Sigmoid(ReadOnlySpan<byte> x, Span<byte> destination)
        => throw new NotSupportedException("Transcendental operations (Sigmoid) are not meaningful for byte type. Use float or double instead.");

    /// <summary>
    /// Transcendental operations are not supported for byte type.
    /// </summary>
    /// <exception cref="NotSupportedException">Always thrown. Log2 produces only 0-7 for bytes.</exception>
    public void Log2(ReadOnlySpan<byte> x, Span<byte> destination)
        => throw new NotSupportedException("Transcendental operations (Log2) are not meaningful for byte type. Use float or double instead.");

    /// <summary>
    /// Transcendental operations are not supported for byte type.
    /// </summary>
    /// <exception cref="NotSupportedException">Always thrown. SoftMax requires floating-point for normalized probabilities.</exception>
    public void SoftMax(ReadOnlySpan<byte> x, Span<byte> destination)
        => throw new NotSupportedException("Transcendental operations (SoftMax) are not meaningful for byte type. Use float or double instead.");

    /// <summary>
    /// Computes cosine similarity using sequential loops.
    /// </summary>
    public byte CosineSimilarity(ReadOnlySpan<byte> x, ReadOnlySpan<byte> y)
        => VectorizedOperationsFallback.CosineSimilarity(this, x, y);

    private static readonly ByteOperations _instance = new();

    public void Fill(Span<byte> destination, byte value) => destination.Fill(value);
    public void MultiplyScalar(ReadOnlySpan<byte> x, byte scalar, Span<byte> destination) => VectorizedOperationsFallback.MultiplyScalar(_instance, x, scalar, destination);
    public void DivideScalar(ReadOnlySpan<byte> x, byte scalar, Span<byte> destination) => VectorizedOperationsFallback.DivideScalar(_instance, x, scalar, destination);
    public void AddScalar(ReadOnlySpan<byte> x, byte scalar, Span<byte> destination) => VectorizedOperationsFallback.AddScalar(_instance, x, scalar, destination);
    public void SubtractScalar(ReadOnlySpan<byte> x, byte scalar, Span<byte> destination) => VectorizedOperationsFallback.SubtractScalar(_instance, x, scalar, destination);
    public void Sqrt(ReadOnlySpan<byte> x, Span<byte> destination) => VectorizedOperationsFallback.Sqrt(_instance, x, destination);
    public void Abs(ReadOnlySpan<byte> x, Span<byte> destination) => VectorizedOperationsFallback.Abs(_instance, x, destination);
    public void Negate(ReadOnlySpan<byte> x, Span<byte> destination) => VectorizedOperationsFallback.Negate(_instance, x, destination);
    public void Clip(ReadOnlySpan<byte> x, byte min, byte max, Span<byte> destination) => VectorizedOperationsFallback.Clip(_instance, x, min, max, destination);
    public void Pow(ReadOnlySpan<byte> x, byte power, Span<byte> destination) => VectorizedOperationsFallback.Pow(_instance, x, power, destination);
    public void Copy(ReadOnlySpan<byte> source, Span<byte> destination) => source.CopyTo(destination);

    #endregion

    public void Floor(ReadOnlySpan<byte> x, Span<byte> destination) => x.CopyTo(destination);
    public void Ceiling(ReadOnlySpan<byte> x, Span<byte> destination) => x.CopyTo(destination);
    public void Frac(ReadOnlySpan<byte> x, Span<byte> destination) => destination.Fill(0);
    public void Sin(ReadOnlySpan<byte> x, Span<byte> destination)
    {
        for (int i = 0; i < x.Length; i++)
            destination[i] = (byte)Math.Sin(x[i]);
    }
    public void Cos(ReadOnlySpan<byte> x, Span<byte> destination)
    {
        for (int i = 0; i < x.Length; i++)
            destination[i] = (byte)Math.Cos(x[i]);
    }

}
