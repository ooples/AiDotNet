using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Operators;

namespace AiDotNet.Tensors.NumericOperations;

/// <summary>
/// Provides mathematical operations for the <see cref="short"/> data type.
/// </summary>
/// <remarks>
/// <para>
/// This class implements the <see cref="INumericOperations{T}"/> interface for the <see cref="short"/> type,
/// providing basic and advanced mathematical operations while handling the limitations of the short data type.
/// Since short values are limited to the range -32,768 to 32,767, operations that would result in values
/// outside this range will overflow and potentially produce unexpected results.
/// </para>
/// <para><b>For Beginners:</b> This class lets you perform math with short numbers (whole numbers between -32,768 and 32,767).
/// 
/// Think of it like a calculator that works specifically with short integer values. For example:
/// - You can add, subtract, multiply, and divide short numbers
/// - You can compare values (is one number greater than another?)
/// - You can perform more advanced operations like square roots or exponents
/// 
/// However, be careful! If your calculations produce a number outside the range -32,768 to 32,767,
/// the result will "wrap around" (overflow) and might give you an unexpected answer. This is like
/// a car odometer that rolls over to 0 after reaching its maximum value.
/// </para>
/// </remarks>
public class ShortOperations : INumericOperations<short>
{
    /// <summary>
    /// Adds two short values.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    /// <returns>The sum of <paramref name="a"/> and <paramref name="b"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method performs addition on two short values. If the result exceeds the maximum value of a short
    /// (32,767) or is less than the minimum value (-32,768), an overflow will occur, wrapping the result around.
    /// </para>
    /// <para><b>For Beginners:</b> This method adds two numbers together.
    /// 
    /// For example:
    /// - Add(5, 3) returns 8
    /// - Add(-10, 20) returns 10
    /// 
    /// Be careful with large numbers! If the result is too big for a short, it will wrap around:
    /// - Add(32000, 1000) might return a negative number because the true sum (33000) is too large
    /// </para>
    /// </remarks>
    public short Add(short a, short b) => (short)(a + b);

    /// <summary>
    /// Subtracts the second value from the first.
    /// </summary>
    /// <param name="a">The value to subtract from.</param>
    /// <param name="b">The value to subtract.</param>
    /// <returns>The difference between <paramref name="a"/> and <paramref name="b"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method performs subtraction of two short values. If the result exceeds the maximum value of a short
    /// (32,767) or is less than the minimum value (-32,768), an overflow will occur, wrapping the result around.
    /// </para>
    /// <para><b>For Beginners:</b> This method subtracts the second number from the first.
    /// 
    /// For example:
    /// - Subtract(10, 3) returns 7
    /// - Subtract(5, 8) returns -3
    /// 
    /// Be careful with very small numbers! If the result is too small for a short, it will wrap around:
    /// - Subtract(-30000, 5000) might return a positive number because the true difference (-35000) is too small
    /// </para>
    /// </remarks>
    public short Subtract(short a, short b) => (short)(a - b);

    /// <summary>
    /// Multiplies two short values.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    /// <returns>The product of <paramref name="a"/> and <paramref name="b"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method performs multiplication of two short values. The result of multiplying two short values can
    /// easily exceed the range of a short, causing overflow and potentially returning an unexpected value.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies two numbers together.
    /// 
    /// For example:
    /// - Multiply(4, 5) returns 20
    /// - Multiply(-3, 7) returns -21
    /// 
    /// Multiplication can easily produce numbers that are too large for a short:
    /// - Multiply(200, 200) would be 40,000, which is outside the short range, so the result will be incorrect
    /// </para>
    /// </remarks>
    public short Multiply(short a, short b) => (short)(a * b);

    /// <summary>
    /// Divides the first value by the second.
    /// </summary>
    /// <param name="a">The dividend (value to be divided).</param>
    /// <param name="b">The divisor (value to divide by).</param>
    /// <returns>The quotient of <paramref name="a"/> divided by <paramref name="b"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method performs integer division of two short values. Because short is an integer type, 
    /// the result will be truncated (rounded down). Division by zero will throw a DivideByZeroException.
    /// </para>
    /// <para><b>For Beginners:</b> This method divides the first number by the second.
    /// 
    /// For example:
    /// - Divide(10, 2) returns 5
    /// - Divide(7, 2) returns 3 (not 3.5, since short values are whole numbers only)
    /// 
    /// Important notes:
    /// - The result is always rounded down to the nearest whole number
    /// - Dividing by zero will cause your program to crash with an error
    /// </para>
    /// </remarks>
    public short Divide(short a, short b) => (short)(a / b);

    /// <summary>
    /// Negates a short value.
    /// </summary>
    /// <param name="a">The value to negate.</param>
    /// <returns>The negative of <paramref name="a"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the negative of the input value. Note that negating short.MinValue (-32,768)
    /// would result in 32,768, which exceeds short.MaxValue (32,767), causing an overflow. In this case,
    /// the result would be short.MinValue itself due to overflow.
    /// </para>
    /// <para><b>For Beginners:</b> This method changes the sign of a number.
    /// 
    /// For example:
    /// - Negate(5) returns -5
    /// - Negate(-10) returns 10
    /// 
    /// Special case:
    /// - Negate(-32768) will not work correctly because 32768 is too large for a short value
    /// </para>
    /// </remarks>
    public short Negate(short a) => (short)-a;

    /// <summary>
    /// Gets the value zero as a short.
    /// </summary>
    /// <value>The value 0 as a short.</value>
    /// <remarks>
    /// <para>
    /// This property returns the value zero (0) as a short. It is useful for operations that
    /// require a zero value, such as initializing variables or as a default value.
    /// </para>
    /// <para><b>For Beginners:</b> This property simply gives you the number zero (0) as a short.
    /// 
    /// This is useful when you need a known zero value in your code, for example:
    /// - When starting a counter
    /// - When you need to initialize a value before calculating
    /// - As a default or fallback value
    /// </para>
    /// </remarks>
    public short Zero => 0;

    /// <summary>
    /// Gets the value one as a short.
    /// </summary>
    /// <value>The value 1 as a short.</value>
    /// <remarks>
    /// <para>
    /// This property returns the value one (1) as a short. It is useful for operations that
    /// require a unit value, such as incrementing a counter or as an identity element in multiplication.
    /// </para>
    /// <para><b>For Beginners:</b> This property simply gives you the number one (1) as a short.
    /// 
    /// This is useful in many situations:
    /// - When incrementing a counter (adding 1)
    /// - In mathematical formulas that need the number 1
    /// - As a starting value for multiplication
    /// </para>
    /// </remarks>
    public short One => 1;

    /// <summary>
    /// Calculates the square root of a short value.
    /// </summary>
    /// <param name="value">The value to calculate the square root of.</param>
    /// <returns>The square root of <paramref name="value"/> as a short.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the square root of the input value and converts the result to a short.
    /// The calculation is performed using double-precision arithmetic and then cast to a short, which means
    /// the result will be truncated to an integer value. Negative inputs will result in NaN (Not a Number)
    /// which, when cast to short, will typically result in 0.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the square root of a number.
    /// 
    /// The square root of a number is another number that, when multiplied by itself, gives the original number.
    /// 
    /// For example:
    /// - Sqrt(4) returns 2 (because 2 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 2 = 4)
    /// - Sqrt(9) returns 3 (because 3 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 3 = 9)
    /// - Sqrt(10) returns 3 (because the true square root is approximately 3.16, but as a short it's rounded down to 3)
    /// 
    /// Note: Square roots of negative numbers aren't real numbers, so Sqrt(-4) will return 0.
    /// </para>
    /// </remarks>
    public short Sqrt(short value) => (short)Math.Sqrt(value);

    /// <summary>
    /// Converts a double value to a short.
    /// </summary>
    /// <param name="value">The double value to convert.</param>
    /// <returns>The double value converted to a short.</returns>
    /// <remarks>
    /// <para>
    /// This method converts a double-precision floating-point value to a short. The conversion truncates
    /// the fractional part of the double and may cause overflow if the double value is outside the range of short.
    /// Values outside the range of short (-32,768 to 32,767) will be clamped to short.MinValue or short.MaxValue.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts a decimal number to a whole short number.
    /// 
    /// When converting:
    /// - The decimal part is dropped (not rounded)
    /// - If the number is too large or too small for a short, you'll get unexpected results
    /// 
    /// For example:
    /// - FromDouble(5.7) returns 5 (decimal part is simply dropped)
    /// - FromDouble(3.2) returns 3
    /// - FromDouble(100000.0) will return a value that doesn't make sense because 100,000 is too large for a short
    /// </para>
    /// </remarks>
    public short FromDouble(double value) => (short)value;

    /// <summary>
    /// Determines if the first value is greater than the second.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns><c>true</c> if <paramref name="a"/> is greater than <paramref name="b"/>; otherwise, <c>false</c>.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two short values and returns true if the first value is greater than the second.
    /// The comparison uses the standard greater than operator for short values.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is bigger than the second.
    /// 
    /// For example:
    /// - GreaterThan(10, 5) returns true (because 10 is greater than 5)
    /// - GreaterThan(3, 7) returns false (because 3 is not greater than 7)
    /// - GreaterThan(4, 4) returns false (because 4 is equal to 4, not greater than it)
    /// </para>
    /// </remarks>
    public bool GreaterThan(short a, short b) => a > b;

    /// <summary>
    /// Determines if the first value is less than the second.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns><c>true</c> if <paramref name="a"/> is less than <paramref name="b"/>; otherwise, <c>false</c>.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two short values and returns true if the first value is less than the second.
    /// The comparison uses the standard less than operator for short values.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is smaller than the second.
    /// 
    /// For example:
    /// - LessThan(5, 10) returns true (because 5 is less than 10)
    /// - LessThan(7, 3) returns false (because 7 is not less than 3)
    /// - LessThan(4, 4) returns false (because 4 is equal to 4, not less than it)
    /// </para>
    /// </remarks>
    public bool LessThan(short a, short b) => a < b;

    /// <summary>
    /// Calculates the absolute value of a short.
    /// </summary>
    /// <param name="value">The value to calculate the absolute value for.</param>
    /// <returns>The absolute value of <paramref name="value"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the absolute value of the input, which is the value without its sign.
    /// Note that the absolute value of short.MinValue (-32,768) cannot be represented as a positive short,
    /// since short.MaxValue is 32,767. In this case, due to overflow, Abs(short.MinValue) will return short.MinValue itself.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives you the positive version of a number.
    /// 
    /// The absolute value of a number is how far it is from zero, regardless of direction (positive or negative).
    /// 
    /// For example:
    /// - Abs(5) returns 5 (a positive number stays positive)
    /// - Abs(-10) returns 10 (a negative number becomes positive)
    /// - Abs(0) returns 0
    /// 
    /// Special case:
    /// - Abs(-32768) will not work correctly because 32768 is too large for a short value
    /// </para>
    /// </remarks>
    public short Abs(short value) => Math.Abs(value);

    /// <summary>
    /// Squares a short value.
    /// </summary>
    /// <param name="value">The value to square.</param>
    /// <returns>The square of <paramref name="value"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the square of the input value (the value multiplied by itself).
    /// The result of squaring a short value can easily exceed the range of a short,
    /// causing overflow and potentially returning an unexpected value.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies a number by itself.
    /// 
    /// For example:
    /// - Square(4) returns 16 (because 4 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 4 = 16)
    /// - Square(-5) returns 25 (because -5 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  -5 = 25)
    /// 
    /// Be careful with larger numbers! Squaring even moderate values can easily exceed the short range:
    /// - Square(200) would be 40,000, which is outside the short range, so the result will be incorrect
    /// </para>
    /// </remarks>
    public short Square(short value) => Multiply(value, value);

    /// <summary>
    /// Calculates e raised to the specified power.
    /// </summary>
    /// <param name="value">The power to raise e to.</param>
    /// <returns>The value of e raised to the power of <paramref name="value"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the exponential function (e^value) for the input value, where e is Euler's number
    /// (approximately 2.71828). The calculation is performed using double-precision arithmetic and then
    /// rounded to the nearest integer and cast to a short. This may cause overflow for large input values,
    /// and the precision of the result is limited by the short data type.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates "e" raised to a power.
    /// 
    /// "e" is a special mathematical constant (approximately 2.71828) used in many calculations, especially
    /// those involving growth or decay.
    /// 
    /// For example:
    /// - Exp(1) returns 3 (because e^1 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 2.71828, rounded to 3 as a short)
    /// - Exp(2) returns 7 (because e^2 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 7.38906, rounded to 7 as a short)
    /// - Exp(10) will likely overflow since e^10 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 22,026.47, which is much larger than a short can hold
    /// 
    /// This function is useful in calculations involving:
    /// - Compound interest
    /// - Population growth
    /// - Radioactive decay
    /// </para>
    /// </remarks>
    public short Exp(short value) => (short)Math.Round(Math.Exp(value));

    /// <summary>
    /// Determines if two short values are equal.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns><c>true</c> if <paramref name="a"/> is equal to <paramref name="b"/>; otherwise, <c>false</c>.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two short values for equality. Two short values are considered equal
    /// if they represent the same numeric value.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if two numbers are exactly the same.
    /// 
    /// For example:
    /// - Equals(5, 5) returns true (because both numbers are 5)
    /// - Equals(10, 15) returns false (because 10 and 15 are different numbers)
    /// - Equals(-7, -7) returns true (because both numbers are -7)
    /// </para>
    /// </remarks>
    public bool Equals(short a, short b) => a == b;

    public int Compare(short a, short b) => a.CompareTo(b);

    /// <summary>
    /// Raises a value to the specified power.
    /// </summary>
    /// <param name="baseValue">The base value.</param>
    /// <param name="exponent">The exponent.</param>
    /// <returns>The base value raised to the specified power.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the base value raised to the power of the exponent. The calculation is
    /// performed using double-precision arithmetic and then cast to a short, which may cause
    /// overflow for large results and truncation of fractional parts.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies a number by itself a specified number of times.
    /// 
    /// For example:
    /// - Power(2, 3) returns 8 (because 2ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â² = 2 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 2 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  2 = 8)
    /// - Power(3, 2) returns 9 (because 3ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â² = 3 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 3 = 9)
    /// - Power(5, 0) returns 1 (any number raised to the power of 0 is 1)
    /// 
    /// Be careful with larger values! The result can quickly exceed the short range:
    /// - Power(10, 5) would be 100,000, which is outside the short range, so the result will be incorrect
    /// 
    /// Fractional results are truncated to whole numbers:
    /// - Power(2, -1) would mathematically be 0.5, but as a short it returns 0
    /// </para>
    /// </remarks>
    public short Power(short baseValue, short exponent) => (short)Math.Pow(baseValue, exponent);

    /// <summary>
    /// Calculates the natural logarithm (base e) of a value.
    /// </summary>
    /// <param name="value">The value to calculate the logarithm for.</param>
    /// <returns>The natural logarithm of <paramref name="value"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the natural logarithm (ln) of the input value. The calculation is
    /// performed using double-precision arithmetic and then cast to a short. The result is truncated
    /// to an integer, leading to loss of precision. If the input is less than or equal to zero,
    /// the result will be a mathematical error (NaN), which typically becomes 0 when cast to a short.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the natural logarithm of a number.
    /// 
    /// The natural logarithm (ln) is the reverse of the exponential function. It tells you what power
    /// you need to raise "e" to in order to get your input value.
    /// 
    /// For example:
    /// - Log(1) returns 0 (because e^0 = 1)
    /// - Log(3) returns 1 (because e^1 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 2.71828, and when cast to a short, the decimal part is dropped)
    /// - Log(10) returns 2 (because e^2.303 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 10, and when cast to a short, the decimal part is dropped)
    /// 
    /// Important notes:
    /// - The logarithm of a negative number or zero is not defined, so Log(-5) or Log(0) will return 0
    /// - Logarithm results are usually decimals, but they'll be converted to whole numbers when stored as shorts
    /// </para>
    /// </remarks>
    public short Log(short value) => (short)Math.Log(value);

    /// <summary>
    /// Determines if the first value is greater than or equal to the second.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns><c>true</c> if <paramref name="a"/> is greater than or equal to <paramref name="b"/>; otherwise, <c>false</c>.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two short values and returns true if the first value is greater than or equal to the second.
    /// The comparison uses the standard greater than or equal to operator for short values.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is bigger than or the same as the second.
    /// 
    /// For example:
    /// - GreaterThanOrEquals(10, 5) returns true (because 10 is greater than 5)
    /// - GreaterThanOrEquals(7, 7) returns true (because 7 is equal to 7)
    /// - GreaterThanOrEquals(3, 8) returns false (because 3 is less than 8)
    /// </para>
    /// </remarks>
    public bool GreaterThanOrEquals(short a, short b) => a >= b;

    /// <summary>
    /// Determines if the first value is less than or equal to the second.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns><c>true</c> if <paramref name="a"/> is less than or equal to <paramref name="b"/>; otherwise, <c>false</c>.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two short values and returns true if the first value is less than or equal to the second.
    /// The comparison uses the standard less than or equal to operator for short values.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is smaller than or the same as the second.
    /// 
    /// For example:
    /// - LessThanOrEquals(5, 10) returns true (because 5 is less than 10)
    /// - LessThanOrEquals(7, 7) returns true (because 7 is equal to 7)
    /// - LessThanOrEquals(9, 4) returns false (because 9 is greater than 4)
    /// </para>
    /// </remarks>
    public bool LessThanOrEquals(short a, short b) => a <= b;

    /// <summary>
    /// Converts a short value to a 32-bit integer.
    /// </summary>
    /// <param name="value">The short value to convert.</param>
    /// <returns>The short value as a 32-bit integer.</returns>
    /// <remarks>
    /// <para>
    /// This method converts a short (16-bit) value to an int (32-bit) value. The conversion will always succeed
    /// because all possible short values can be represented as int values.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts a short number to a regular integer (int).
    /// 
    /// A short can store numbers from -32,768 to 32,767.
    /// An int can store much larger numbers, from -2,147,483,648 to 2,147,483,647.
    /// 
    /// This conversion is always safe because any short value will fit within the int range.
    /// 
    /// For example:
    /// - ToInt32(5) returns 5 as an int
    /// - ToInt32(-10000) returns -10000 as an int
    /// - ToInt32(32767) returns 32767 as an int
    /// </para>
    /// </remarks>
    public int ToInt32(short value) => value;

    /// <summary>
    /// Rounds a short value.
    /// </summary>
    /// <param name="value">The value to round.</param>
    /// <returns>The rounded value.</returns>
    /// <remarks>
    /// <para>
    /// For short values, which are already integers, this method simply returns the value unchanged.
    /// Rounding only applies to floating-point values that have fractional parts.
    /// </para>
    /// <para><b>For Beginners:</b> This method rounds a number to the nearest whole number.
    /// 
    /// Since a short is already a whole number, this method simply returns the same number without any change.
    /// 
    /// For example:
    /// - Round(5) returns 5
    /// - Round(-10) returns -10
    /// 
    /// This method exists mainly for consistency with other numeric types like float or double,
    /// where rounding would actually change the value.
    /// </para>
    /// </remarks>
    public short Round(short value) => value;

    public short Floor(short value) => value;
    public short Ceiling(short value) => value;
    public short Frac(short value) => 0;


    /// <summary>
    /// Gets the minimum value that can be represented by a short.
    /// </summary>
    /// <value>The minimum value of a short, which is -32,768.</value>
    /// <remarks>
    /// <para>
    /// This property returns the smallest possible value that can be represented by the short data type.
    /// Attempting to store a value less than this in a short will result in overflow.
    /// </para>
    /// <para><b>For Beginners:</b> This property gives you the smallest possible number that a short can hold.
    /// 
    /// For short values, the minimum value is -32,768.
    /// If you try to create a short with a smaller value (like -40,000), the number will wrap around
    /// and give you an incorrect result.
    /// 
    /// This is useful when you need to:
    /// - Check if a value is too small to be stored as a short
    /// - Initialize a variable to the smallest possible value before comparing
    /// - Set boundaries for valid input values
    /// </para>
    /// </remarks>
    public short MinValue => short.MinValue;

    /// <summary>
    /// Gets the maximum value that can be represented by a short.
    /// </summary>
    /// <value>The maximum value of a short, which is 32,767.</value>
    /// <remarks>
    /// <para>
    /// This property returns the largest possible value that can be represented by the short data type.
    /// Attempting to store a value greater than this in a short will result in overflow.
    /// </para>
    /// <para><b>For Beginners:</b> This property gives you the largest possible number that a short can hold.
    /// 
    /// For short values, the maximum value is 32,767.
    /// If you try to create a short with a larger value (like 40,000), the number will wrap around
    /// and give you an incorrect result.
    /// 
    /// This is useful when you need to:
    /// - Check if a value is too large to be stored as a short
    /// - Initialize a variable to the largest possible value before comparing
    /// - Set boundaries for valid input values
    /// </para>
    /// </remarks>
    public short MaxValue => short.MaxValue;

    /// <summary>
    /// Determines if a short value is NaN (Not a Number).
    /// </summary>
    /// <param name="value">The value to check.</param>
    /// <returns>Always <c>false</c> for short values.</returns>
    /// <remarks>
    /// <para>
    /// This method always returns false because the short data type can only represent integers,
    /// and the concept of NaN (Not a Number) only applies to floating-point types like float and double.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if a number is "Not a Number" (NaN).
    /// 
    /// For short values, the result is always false because a short can only contain valid whole numbers.
    /// The concept of "Not a Number" applies only to floating-point types like float or double,
    /// which can represent special values like the result of divide-by-zero.
    /// 
    /// This method exists mainly for consistency with other numeric types where IsNaN is meaningful.
    /// </para>
    /// </remarks>
    public bool IsNaN(short value) => false;

    /// <summary>
    /// Determines if a short value is infinity.
    /// </summary>
    /// <param name="value">The value to check.</param>
    /// <returns>Always <c>false</c> for short values.</returns>
    /// <remarks>
    /// <para>
    /// This method always returns false because the short data type can only represent integers,
    /// and the concept of infinity only applies to floating-point types like float and double.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if a number is "infinity".
    /// 
    /// For short values, the result is always false because a short can only contain finite whole numbers.
    /// The concept of "infinity" applies only to floating-point types like float or double,
    /// which can represent special values like the result of divide-by-zero.
    /// 
    /// This method exists mainly for consistency with other numeric types where IsInfinity is meaningful.
    /// </para>
    /// </remarks>
    public bool IsInfinity(short value) => false;

    /// <summary>
    /// Returns the sign of a short value as -1, 0, or 1.
    /// </summary>
    /// <param name="value">The value to determine the sign of.</param>
    /// <returns>
    /// -1 if <paramref name="value"/> is negative;
    /// 0 if <paramref name="value"/> is zero;
    /// 1 if <paramref name="value"/> is positive.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method returns a value indicating the sign of the input value: -1 for negative values,
    /// 0 for zero, and 1 for positive values. This is useful for determining the direction or polarity
    /// of a value without considering its magnitude.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells you if a number is positive, negative, or zero.
    /// 
    /// It returns:
    /// - 1 if the number is positive (greater than zero)
    /// - 0 if the number is exactly zero
    /// - -1 if the number is negative (less than zero)
    /// 
    /// This is useful when you only care about the direction of a value, not how large it is.
    /// 
    /// For example:
    /// - SignOrZero(42) returns 1
    /// - SignOrZero(0) returns 0
    /// - SignOrZero(-15) returns -1
    /// 
    /// You might use this to determine which way something is moving, or to simplify comparisons.
    /// </para>
    /// </remarks>
    public short SignOrZero(short value)
    {
        if (value > 0) return 1;
        if (value < 0) return -1;
        return 0;
    }

    /// <summary>
    /// Gets the number of bits used for precision in short (16 bits).
    /// </summary>
    public int PrecisionBits => 16;

    /// <summary>
    /// Converts a short value to float (FP32) precision.
    /// </summary>
    public float ToFloat(short value) => (float)value;

    /// <summary>
    /// Converts a float value to short.
    /// </summary>
    public short FromFloat(float value) => (short)MathExtensions.Clamp((int)Math.Round(value), short.MinValue, short.MaxValue);

    /// <summary>
    /// Converts a short value to Half (FP16) precision.
    /// </summary>
    public Half ToHalf(short value) => (Half)value;

    /// <summary>
    /// Converts a Half value to short.
    /// </summary>
    public short FromHalf(Half value) => (short)MathExtensions.Clamp((int)Math.Round((float)value), short.MinValue, short.MaxValue);

    /// <summary>
    /// Converts a short value to double (FP64) precision.
    /// </summary>
    public double ToDouble(short value) => (double)value;

    /// <inheritdoc/>
    public bool SupportsCpuAcceleration => false;

    /// <inheritdoc/>
    public bool SupportsGpuAcceleration => false;

    #region IVectorizedOperations<short> Implementation

    /// <summary>
    /// Performs element-wise addition using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public void Add(ReadOnlySpan<short> x, ReadOnlySpan<short> y, Span<short> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<AddOperatorShort>(x, y, destination);

    /// <summary>
    /// Performs element-wise subtraction using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public void Subtract(ReadOnlySpan<short> x, ReadOnlySpan<short> y, Span<short> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<SubtractOperatorShort>(x, y, destination);

    /// <summary>
    /// Performs element-wise multiplication using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public void Multiply(ReadOnlySpan<short> x, ReadOnlySpan<short> y, Span<short> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<MultiplyOperatorShort>(x, y, destination);

    /// <summary>
    /// Performs element-wise division using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public void Divide(ReadOnlySpan<short> x, ReadOnlySpan<short> y, Span<short> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<DivideOperatorShort>(x, y, destination);

    /// <summary>
    /// Computes dot product using sequential loops.
    /// </summary>
    public short Dot(ReadOnlySpan<short> x, ReadOnlySpan<short> y)
        => VectorizedOperationsFallback.Dot(this, x, y);

    /// <summary>
    /// Computes sum using sequential loops.
    /// </summary>
    public short Sum(ReadOnlySpan<short> x)
        => VectorizedOperationsFallback.Sum(this, x);

    /// <summary>
    /// Finds maximum using sequential loops.
    /// </summary>
    public short Max(ReadOnlySpan<short> x)
        => VectorizedOperationsFallback.Max(this, x);

    /// <summary>
    /// Finds minimum using sequential loops.
    /// </summary>
    public short Min(ReadOnlySpan<short> x)
        => VectorizedOperationsFallback.Min(this, x);

    /// <summary>
    /// Transcendental operations are not supported for short type.
    /// </summary>
    /// <exception cref="NotSupportedException">Always thrown. Exp produces misleading results for short.</exception>
    public void Exp(ReadOnlySpan<short> x, Span<short> destination)
        => throw new NotSupportedException("Transcendental operations (Exp) are not meaningful for short type. Use float or double instead.");

    /// <summary>
    /// Transcendental operations are not supported for short type.
    /// </summary>
    /// <exception cref="NotSupportedException">Always thrown. Log produces misleading results for short.</exception>
    public void Log(ReadOnlySpan<short> x, Span<short> destination)
        => throw new NotSupportedException("Transcendental operations (Log) are not meaningful for short type. Use float or double instead.");

    /// <summary>
    /// Transcendental operations are not supported for short type.
    /// </summary>
    /// <exception cref="NotSupportedException">Always thrown. Tanh produces only -1, 0, or 1 for short.</exception>
    public void Tanh(ReadOnlySpan<short> x, Span<short> destination)
        => throw new NotSupportedException("Transcendental operations (Tanh) are not meaningful for short type. Use float or double instead.");

    /// <summary>
    /// Transcendental operations are not supported for short type.
    /// </summary>
    /// <exception cref="NotSupportedException">Always thrown. Sigmoid saturates for short inputs.</exception>
    public void Sigmoid(ReadOnlySpan<short> x, Span<short> destination)
        => throw new NotSupportedException("Transcendental operations (Sigmoid) are not meaningful for short type. Use float or double instead.");

    /// <summary>
    /// Transcendental operations are not supported for short type.
    /// </summary>
    /// <exception cref="NotSupportedException">Always thrown. Log2 produces misleading results for short.</exception>
    public void Log2(ReadOnlySpan<short> x, Span<short> destination)
        => throw new NotSupportedException("Transcendental operations (Log2) are not meaningful for short type. Use float or double instead.");

    /// <summary>
    /// Transcendental operations are not supported for short type.
    /// </summary>
    /// <exception cref="NotSupportedException">Always thrown. SoftMax requires floating-point for normalized probabilities.</exception>
    public void SoftMax(ReadOnlySpan<short> x, Span<short> destination)
        => throw new NotSupportedException("Transcendental operations (SoftMax) are not meaningful for short type. Use float or double instead.");

    /// <summary>
    /// Computes cosine similarity using sequential loops.
    /// </summary>
    public short CosineSimilarity(ReadOnlySpan<short> x, ReadOnlySpan<short> y)
        => VectorizedOperationsFallback.CosineSimilarity(this, x, y);

    private static readonly ShortOperations _instance = new();

    public void Fill(Span<short> destination, short value) => destination.Fill(value);
    public void MultiplyScalar(ReadOnlySpan<short> x, short scalar, Span<short> destination) => VectorizedOperationsFallback.MultiplyScalar(_instance, x, scalar, destination);
    public void DivideScalar(ReadOnlySpan<short> x, short scalar, Span<short> destination) => VectorizedOperationsFallback.DivideScalar(_instance, x, scalar, destination);
    public void AddScalar(ReadOnlySpan<short> x, short scalar, Span<short> destination) => VectorizedOperationsFallback.AddScalar(_instance, x, scalar, destination);
    public void SubtractScalar(ReadOnlySpan<short> x, short scalar, Span<short> destination) => VectorizedOperationsFallback.SubtractScalar(_instance, x, scalar, destination);
    public void Sqrt(ReadOnlySpan<short> x, Span<short> destination) => VectorizedOperationsFallback.Sqrt(_instance, x, destination);
    public void Abs(ReadOnlySpan<short> x, Span<short> destination) => VectorizedOperationsFallback.Abs(_instance, x, destination);
    public void Negate(ReadOnlySpan<short> x, Span<short> destination) => VectorizedOperationsFallback.Negate(_instance, x, destination);
    public void Clip(ReadOnlySpan<short> x, short min, short max, Span<short> destination) => VectorizedOperationsFallback.Clip(_instance, x, min, max, destination);
    public void Pow(ReadOnlySpan<short> x, short power, Span<short> destination) => VectorizedOperationsFallback.Pow(_instance, x, power, destination);
    public void Copy(ReadOnlySpan<short> source, Span<short> destination) => source.CopyTo(destination);

    #endregion

    public void Floor(ReadOnlySpan<short> x, Span<short> destination) => x.CopyTo(destination);
    public void Ceiling(ReadOnlySpan<short> x, Span<short> destination) => x.CopyTo(destination);
    public void Frac(ReadOnlySpan<short> x, Span<short> destination) => destination.Fill(0);

}
