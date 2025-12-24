using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Operators;

namespace AiDotNet.Tensors.NumericOperations;

/// <summary>
/// Provides mathematical operations for the <see cref="ushort"/> (UInt16) data type.
/// </summary>
/// <remarks>
/// <para>
/// This class implements the <see cref="INumericOperations{T}"/> interface for the <see cref="ushort"/> type,
/// providing basic and advanced mathematical operations while handling the limitations of the unsigned short data type.
/// Since ushort values are limited to the range 0 to 65,535, operations that would result in values
/// outside this range will overflow and potentially produce unexpected results.
/// </para>
/// <para><b>For Beginners:</b> This class lets you perform math with unsigned short numbers (whole numbers between 0 and 65,535).
/// 
/// Think of it like a calculator that works specifically with positive whole numbers up to 65,535. For example:
/// - You can add, subtract, multiply, and divide ushort numbers
/// - You can compare values (is one number greater than another?)
/// - You can perform more advanced operations like square roots or exponents
/// 
/// However, be careful! If your calculations produce a number larger than 65,535 or a negative number,
/// the result will "wrap around" (overflow) and might give you an unexpected answer. This is like
/// a car odometer that rolls over to 0 after reaching its maximum value.
/// 
/// The main difference between ushort and short is that ushort can only store positive numbers (and zero),
/// but it can store larger positive numbers (up to 65,535 instead of just 32,767).
/// </para>
/// </remarks>
public class UInt16Operations : INumericOperations<ushort>
{
    /// <summary>
    /// Adds two ushort values.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    /// <returns>The sum of <paramref name="a"/> and <paramref name="b"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method performs addition on two ushort values. If the result exceeds the maximum value of a ushort
    /// (65,535), an overflow will occur, wrapping the result around to start from zero again.
    /// </para>
    /// <para><b>For Beginners:</b> This method adds two numbers together.
    /// 
    /// For example:
    /// - Add(5, 3) returns 8
    /// - Add(10, 20) returns 30
    /// 
    /// Be careful with large numbers! If the result is too big for a ushort, it will wrap around:
    /// - Add(65000, 1000) might return a small number because the true sum (66000) is too large
    /// </para>
    /// </remarks>
    public ushort Add(ushort a, ushort b) => (ushort)(a + b);

    /// <summary>
    /// Subtracts the second value from the first.
    /// </summary>
    /// <param name="a">The value to subtract from.</param>
    /// <param name="b">The value to subtract.</param>
    /// <returns>The difference between <paramref name="a"/> and <paramref name="b"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method performs subtraction of two ushort values. If the result would be negative (when b > a),
    /// an overflow will occur, wrapping the result around to a large positive number. This is because ushort
    /// cannot represent negative values.
    /// </para>
    /// <para><b>For Beginners:</b> This method subtracts the second number from the first.
    /// 
    /// For example:
    /// - Subtract(10, 3) returns 7
    /// - Subtract(20, 5) returns 15
    /// 
    /// Be careful when the second number is larger than the first! Since a ushort can't be negative:
    /// - Subtract(5, 10) will not return -5. Instead, it will return 65,531 (which is 65,536 - 5)
    /// 
    /// This happens because the result wraps around from the end of the range to the beginning.
    /// </para>
    /// </remarks>
    public ushort Subtract(ushort a, ushort b) => (ushort)(a - b);

    /// <summary>
    /// Multiplies two ushort values.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    /// <returns>The product of <paramref name="a"/> and <paramref name="b"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method performs multiplication of two ushort values. The result of multiplying two ushort values can
    /// easily exceed the range of a ushort, causing overflow and potentially returning an unexpected value.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies two numbers together.
    /// 
    /// For example:
    /// - Multiply(4, 5) returns 20
    /// - Multiply(10, 3) returns 30
    /// 
    /// Multiplication can easily produce numbers that are too large for a ushort:
    /// - Multiply(300, 300) would be 90,000, which is outside the ushort range, so the result will be incorrect
    /// </para>
    /// </remarks>
    public ushort Multiply(ushort a, ushort b) => (ushort)(a * b);

    /// <summary>
    /// Divides the first value by the second.
    /// </summary>
    /// <param name="a">The dividend (value to be divided).</param>
    /// <param name="b">The divisor (value to divide by).</param>
    /// <returns>The quotient of <paramref name="a"/> divided by <paramref name="b"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method performs integer division of two ushort values. Because ushort is an integer type, 
    /// the result will be truncated (rounded down). Division by zero will throw a DivideByZeroException.
    /// </para>
    /// <para><b>For Beginners:</b> This method divides the first number by the second.
    /// 
    /// For example:
    /// - Divide(10, 2) returns 5
    /// - Divide(7, 2) returns 3 (not 3.5, since ushort values are whole numbers only)
    /// 
    /// Important notes:
    /// - The result is always rounded down to the nearest whole number
    /// - Dividing by zero will cause your program to crash with an error
    /// </para>
    /// </remarks>
    public ushort Divide(ushort a, ushort b) => (ushort)(a / b);

    /// <summary>
    /// Negates a ushort value.
    /// </summary>
    /// <param name="a">The value to negate.</param>
    /// <returns>The two's complement negation of <paramref name="a"/>.</returns>
    /// <remarks>
    /// <para>
    /// Since ushort cannot represent negative values, this method performs a two's complement negation.
    /// For a value 'a', it returns (ushort.MaxValue - a + 1), which is equivalent to (65536 - a) when 
    /// represented in the full 16-bit range.
    /// </para>
    /// <para><b>For Beginners:</b> This method attempts to find the "negative" of an unsigned number.
    /// 
    /// Since ushort can only store positive numbers, true negation isn't possible. Instead, this method
    /// uses a technique called "two's complement" to find the value that, when added to the original number,
    /// gives zero in the ushort range.
    /// 
    /// For example:
    /// - Negate(1) returns 65,535 (because 1 + 65,535 = 65,536, which overflows to 0 in ushort)
    /// - Negate(1000) returns 64,536 (because 1000 + 64,536 = 65,536, which overflows to 0 in ushort)
    /// 
    /// This operation is mostly used in specific bit manipulation contexts or when implementing
    /// certain algorithms that require a "wraparound" behavior.
    /// </para>
    /// </remarks>
    public ushort Negate(ushort a) => (ushort)(ushort.MaxValue - a + 1);

    /// <summary>
    /// Gets the value zero as a ushort.
    /// </summary>
    /// <value>The value 0 as a ushort.</value>
    /// <remarks>
    /// <para>
    /// This property returns the value zero (0) as a ushort. It is useful for operations that
    /// require a zero value, such as initializing variables or as a default value.
    /// </para>
    /// <para><b>For Beginners:</b> This property simply gives you the number zero (0) as a ushort.
    /// 
    /// This is useful when you need a known zero value in your code, for example:
    /// - When starting a counter
    /// - When you need to initialize a value before calculating
    /// - As a default or fallback value
    /// </para>
    /// </remarks>
    public ushort Zero => 0;

    /// <summary>
    /// Gets the value one as a ushort.
    /// </summary>
    /// <value>The value 1 as a ushort.</value>
    /// <remarks>
    /// <para>
    /// This property returns the value one (1) as a ushort. It is useful for operations that
    /// require a unit value, such as incrementing a counter or as an identity element in multiplication.
    /// </para>
    /// <para><b>For Beginners:</b> This property simply gives you the number one (1) as a ushort.
    /// 
    /// This is useful in many situations:
    /// - When incrementing a counter (adding 1)
    /// - In mathematical formulas that need the number 1
    /// - As a starting value for multiplication
    /// </para>
    /// </remarks>
    public ushort One => 1;

    /// <summary>
    /// Calculates the square root of a ushort value.
    /// </summary>
    /// <param name="value">The value to calculate the square root of.</param>
    /// <returns>The square root of <paramref name="value"/> as a ushort.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the square root of the input value and converts the result to a ushort.
    /// The calculation is performed using double-precision arithmetic and then cast to a ushort, which means
    /// the result will be truncated to an integer value.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the square root of a number.
    /// 
    /// The square root of a number is another number that, when multiplied by itself, gives the original number.
    /// 
    /// For example:
    /// - Sqrt(4) returns 2 (because 2 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 2 = 4)
    /// - Sqrt(9) returns 3 (because 3 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 3 = 9)
    /// - Sqrt(10) returns 3 (because the true square root is approximately 3.16, but as a ushort it's rounded down to 3)
    /// 
    /// Unlike with signed numbers, you don't need to worry about negative inputs since ushort values are always positive.
    /// </para>
    /// </remarks>
    public ushort Sqrt(ushort value) => (ushort)Math.Sqrt(value);

    /// <summary>
    /// Converts a double value to a ushort.
    /// </summary>
    /// <param name="value">The double value to convert.</param>
    /// <returns>The double value converted to a ushort.</returns>
    /// <remarks>
    /// <para>
    /// This method converts a double-precision floating-point value to a ushort. The conversion truncates
    /// the fractional part of the double. Negative values will underflow to a large positive value, and values
    /// greater than 65,535 will overflow.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts a decimal number to a whole ushort number.
    /// 
    /// When converting:
    /// - The decimal part is dropped (not rounded)
    /// - If the number is negative, you'll get an unexpected large positive number
    /// - If the number is too large for a ushort, you'll get an unexpected smaller result
    /// 
    /// For example:
    /// - FromDouble(5.7) returns 5 (decimal part is simply dropped)
    /// - FromDouble(3.2) returns 3
    /// - FromDouble(100000.0) will return a value that doesn't make sense because 100,000 is too large for a ushort
    /// - FromDouble(-5.0) will not return -5 (since ushort can't store negative numbers), but instead a large positive number
    /// </para>
    /// </remarks>
    public ushort FromDouble(double value) => (ushort)value;

    /// <summary>
    /// Determines if the first value is greater than the second.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns><c>true</c> if <paramref name="a"/> is greater than <paramref name="b"/>; otherwise, <c>false</c>.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two ushort values and returns true if the first value is greater than the second.
    /// The comparison uses the standard greater than operator for ushort values.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is bigger than the second.
    /// 
    /// For example:
    /// - GreaterThan(10, 5) returns true (because 10 is greater than 5)
    /// - GreaterThan(3, 7) returns false (because 3 is not greater than 7)
    /// - GreaterThan(4, 4) returns false (because 4 is equal to 4, not greater than it)
    /// </para>
    /// </remarks>
    public bool GreaterThan(ushort a, ushort b) => a > b;

    /// <summary>
    /// Determines if the first value is less than the second.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns><c>true</c> if <paramref name="a"/> is less than <paramref name="b"/>; otherwise, <c>false</c>.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two ushort values and returns true if the first value is less than the second.
    /// The comparison uses the standard less than operator for ushort values.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is smaller than the second.
    /// 
    /// For example:
    /// - LessThan(5, 10) returns true (because 5 is less than 10)
    /// - LessThan(7, 3) returns false (because 7 is not less than 3)
    /// - LessThan(4, 4) returns false (because 4 is equal to 4, not less than it)
    /// </para>
    /// </remarks>
    public bool LessThan(ushort a, ushort b) => a < b;

    /// <summary>
    /// Calculates the absolute value of a ushort.
    /// </summary>
    /// <param name="value">The value to calculate the absolute value for.</param>
    /// <returns>The input value unchanged.</returns>
    /// <remarks>
    /// <para>
    /// For ushort values, which are already non-negative, this method simply returns the input value unchanged.
    /// The absolute value function is traditionally used to get the non-negative version of a number, but
    /// since ushort values are always non-negative, no conversion is needed.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives you the positive version of a number.
    /// 
    /// The absolute value of a number is how far it is from zero, ignoring whether it's positive or negative.
    /// 
    /// For ushort values, which are always positive (or zero), this method simply returns the same number:
    /// - Abs(5) returns 5
    /// - Abs(0) returns 0
    /// 
    /// This method exists mainly for consistency with other numeric types where absolute value is meaningful.
    /// </para>
    /// </remarks>
    public ushort Abs(ushort value) => value;

    /// <summary>
    /// Squares a ushort value.
    /// </summary>
    /// <param name="value">The value to square.</param>
    /// <returns>The square of <paramref name="value"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the square of the input value (the value multiplied by itself).
    /// The result of squaring a ushort value can easily exceed the range of a ushort,
    /// causing overflow and potentially returning an unexpected value.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies a number by itself.
    /// 
    /// For example:
    /// - Square(4) returns 16 (because 4 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 4 = 16)
    /// - Square(10) returns 100 (because 10 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 10 = 100)
    /// 
    /// Be careful with larger numbers! Squaring even moderate values can easily exceed the ushort range:
    /// - Square(300) would be 90,000, which is outside the ushort range, so the result will be incorrect
    /// </para>
    /// </remarks>
    public ushort Square(ushort value) => Multiply(value, value);

    /// <summary>
    /// Calculates e raised to the specified power.
    /// </summary>
    /// <param name="value">The power to raise e to.</param>
    /// <returns>The value of e raised to the power of <paramref name="value"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the exponential function (e^value) for the input value, where e is Euler's number
    /// (approximately 2.71828). The calculation is performed using double-precision arithmetic, rounded to the
    /// nearest integer, and then clamped to the maximum ushort value before casting to a ushort. This prevents
    /// overflow for large input values, instead returning ushort.MaxValue (65,535).
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates "e" raised to a power.
    /// 
    /// "e" is a special mathematical constant (approximately 2.71828) used in many calculations, especially
    /// those involving growth or decay.
    /// 
    /// For example:
    /// - Exp(1) returns 3 (because e^1 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 2.71828, rounded to 3 as a ushort)
    /// - Exp(2) returns 7 (because e^2 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 7.38906, rounded to 7 as a ushort)
    /// 
    /// For larger input values, the result grows very quickly:
    /// - Exp(10) returns 22,026 (because e^10 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 22,026.47)
    /// - Exp(12) or higher will return 65,535 (the maximum ushort value) because the true result would be too large
    /// 
    /// This function is useful in calculations involving:
    /// - Compound interest
    /// - Population growth
    /// - Radioactive decay
    /// </para>
    /// </remarks>
    public ushort Exp(ushort value) => (ushort)Math.Min(ushort.MaxValue, Math.Round(Math.Exp(value)));

    /// <summary>
    /// Determines if two ushort values are equal.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns><c>true</c> if <paramref name="a"/> is equal to <paramref name="b"/>; otherwise, <c>false</c>.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two ushort values for equality. Two ushort values are considered equal
    /// if they represent the same numeric value.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if two numbers are exactly the same.
    /// 
    /// For example:
    /// - Equals(5, 5) returns true (because both numbers are 5)
    /// - Equals(10, 15) returns false (because 10 and 15 are different numbers)
    /// </para>
    /// </remarks>
    public bool Equals(ushort a, ushort b) => a == b;

    public int Compare(ushort a, ushort b) => a.CompareTo(b);

    /// <summary>
    /// Raises a value to the specified power.
    /// </summary>
    /// <param name="baseValue">The base value.</param>
    /// <param name="exponent">The exponent.</param>
    /// <returns>The base value raised to the specified power.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the base value raised to the power of the exponent. The calculation is
    /// performed using double-precision arithmetic and then cast to a ushort, which may cause
    /// overflow for large results. Negative exponents will result in fractional values that,
    /// when cast to ushort, will become 0.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies a number by itself a specified number of times.
    /// 
    /// For example:
    /// - Power(2, 3) returns 8 (because 2ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â² = 2 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 2 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  2 = 8)
    /// - Power(3, 2) returns 9 (because 3ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â² = 3 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 3 = 9)
    /// - Power(5, 0) returns 1 (any number raised to the power of 0 is 1)
    /// 
    /// Be careful with larger values! The result can quickly exceed the ushort range:
    /// - Power(10, 4) would be 10,000, which is within the ushort range
    /// - Power(10, 5) would be 100,000, which is outside the ushort range, so the result will be incorrect
    /// 
    /// Fractional results are truncated to whole numbers:
    /// - Power(2, -1) would mathematically be 0.5, but as a ushort it returns 0
    /// </para>
    /// </remarks>
    public ushort Power(ushort baseValue, ushort exponent) => (ushort)Math.Pow(baseValue, exponent);

    /// <summary>
    /// Calculates the natural logarithm (base e) of a value.
    /// </summary>
    /// <param name="value">The value to calculate the logarithm for.</param>
    /// <returns>The natural logarithm of <paramref name="value"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the natural logarithm (ln) of the input value. The calculation is
    /// performed using double-precision arithmetic and then cast to a ushort. The result is truncated
    /// to an integer, leading to loss of precision. If the input is 0, the result will be a mathematical error
    /// (negative infinity), which typically becomes 0 when cast to a ushort.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the natural logarithm of a number.
    /// 
    /// The natural logarithm (ln) is the reverse of the exponential function. It tells you what power
    /// you need to raise "e" to in order to get your input value.
    /// 
    /// For example:
    /// - Log(1) returns 0 (because e^0 = 1)
    /// - Log(3) returns 1 (because e^1 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 2.71828, and when cast to a ushort, the decimal part is dropped)
    /// - Log(10) returns 2 (because e^2.303 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 10, and when cast to a ushort, the decimal part is dropped)
    /// 
    /// Important notes:
    /// - The logarithm of zero is not defined mathematically, so Log(0) will return 0
    /// - Logarithm results are usually decimals, but they'll be converted to whole numbers when stored as ushorts
    /// </para>
    /// </remarks>
    public ushort Log(ushort value) => (ushort)Math.Log(value);

    /// <summary>
    /// Determines if the first value is greater than or equal to the second.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns><c>true</c> if <paramref name="a"/> is greater than or equal to <paramref name="b"/>; otherwise, <c>false</c>.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two ushort values and returns true if the first value is greater than or equal to the second.
    /// The comparison uses the standard greater than or equal to operator for ushort values.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is bigger than or the same as the second.
    /// 
    /// For example:
    /// - GreaterThanOrEquals(10, 5) returns true (because 10 is greater than 5)
    /// - GreaterThanOrEquals(7, 7) returns true (because 7 is equal to 7)
    /// - GreaterThanOrEquals(3, 8) returns false (because 3 is less than 8)
    /// </para>
    /// </remarks>
    public bool GreaterThanOrEquals(ushort a, ushort b) => a >= b;

    /// <summary>
    /// Determines if the first value is less than or equal to the second.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns><c>true</c> if <paramref name="a"/> is less than or equal to <paramref name="b"/>; otherwise, <c>false</c>.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two ushort values and returns true if the first value is less than or equal to the second.
    /// The comparison uses the standard less than or equal to operator for ushort values.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is smaller than or the same as the second.
    /// 
    /// For example:
    /// - LessThanOrEquals(5, 10) returns true (because 5 is less than 10)
    /// - LessThanOrEquals(7, 7) returns true (because 7 is equal to 7)
    /// - LessThanOrEquals(9, 4) returns false (because 9 is greater than 4)
    /// </para>
    /// </remarks>
    public bool LessThanOrEquals(ushort a, ushort b) => a <= b;

    /// <summary>
    /// Converts a ushort value to a 32-bit integer.
    /// </summary>
    /// <param name="value">The ushort value to convert.</param>
    /// <returns>The ushort value as a 32-bit integer.</returns>
    /// <remarks>
    /// <para>
    /// This method converts a ushort (16-bit) value to an int (32-bit) value. The conversion will always succeed
    /// because all possible ushort values (0 to 65,535) can be represented as int values.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts a ushort number to a regular integer (int).
    /// 
    /// A ushort can store numbers from 0 to 65,535.
    /// An int can store much larger numbers, from -2,147,483,648 to 2,147,483,647.
    /// 
    /// This conversion is always safe because any ushort value will fit within the int range.
    /// 
    /// For example:
    /// - ToInt32(5) returns 5 as an int
    /// - ToInt32(1000) returns 1000 as an int
    /// - ToInt32(65535) returns 65535 as an int
    /// </para>
    /// </remarks>
    public int ToInt32(ushort value) => value;

    /// <summary>
    /// Rounds a ushort value.
    /// </summary>
    /// <param name="value">The value to round.</param>
    /// <returns>The rounded value.</returns>
    /// <remarks>
    /// <para>
    /// For ushort values, which are already integers, this method simply returns the value unchanged.
    /// Rounding only applies to floating-point values that have fractional parts.
    /// </para>
    /// <para><b>For Beginners:</b> This method rounds a number to the nearest whole number.
    /// 
    /// Since a ushort is already a whole number, this method simply returns the same number without any change.
    /// 
    /// For example:
    /// - Round(5) returns 5
    /// - Round(10) returns 10
    /// 
    /// This method exists mainly for consistency with other numeric types like float or double,
    /// where rounding would actually change the value.
    /// </para>
    /// </remarks>
    public ushort Round(ushort value) => value;

    public ushort Floor(ushort value) => value;
    public ushort Ceiling(ushort value) => value;
    public ushort Frac(ushort value) => 0;


    /// <summary>
    /// Gets the minimum value that can be represented by a ushort.
    /// </summary>
    /// <value>The minimum value of a ushort, which is 0.</value>
    /// <remarks>
    /// <para>
    /// This property returns the smallest possible value that can be represented by the ushort data type,
    /// which is 0. Unlike signed types, ushort cannot represent negative values.
    /// </para>
    /// <para><b>For Beginners:</b> This property gives you the smallest possible number that a ushort can hold.
    /// 
    /// For ushort values, the minimum value is always 0, because ushort can only store positive whole numbers
    /// (and zero).
    /// 
    /// This is useful when you need to:
    /// - Check if a value is valid for a ushort
    /// - Initialize a variable to the smallest possible value
    /// - Set boundaries for valid input values
    /// </para>
    /// </remarks>
    public ushort MinValue => ushort.MinValue;

    /// <summary>
    /// Gets the maximum value that can be represented by a ushort.
    /// </summary>
    /// <value>The maximum value of a ushort, which is 65,535.</value>
    /// <remarks>
    /// <para>
    /// This property returns the largest possible value that can be represented by the ushort data type,
    /// which is 65,535. Attempting to store a value greater than this in a ushort will result in overflow.
    /// </para>
    /// <para><b>For Beginners:</b> This property gives you the largest possible number that a ushort can hold.
    /// 
    /// For ushort values, the maximum value is 65,535.
    /// If you try to create a ushort with a larger value (like 70,000), the number will wrap around
    /// and give you an incorrect result.
    /// 
    /// This is useful when you need to:
    /// - Check if a value is too large to be stored as a ushort
    /// - Initialize a variable to the largest possible value before comparing
    /// - Set boundaries for valid input values
    /// </para>
    /// </remarks>
    public ushort MaxValue => ushort.MaxValue;

    /// <summary>
    /// Determines if a ushort value is NaN (Not a Number).
    /// </summary>
    /// <param name="value">The value to check.</param>
    /// <returns>Always <c>false</c> for ushort values.</returns>
    /// <remarks>
    /// <para>
    /// This method always returns false because the ushort data type can only represent integers,
    /// and the concept of NaN (Not a Number) only applies to floating-point types like float and double.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if a number is "Not a Number" (NaN).
    /// 
    /// For ushort values, the result is always false because a ushort can only contain valid whole numbers.
    /// The concept of "Not a Number" applies only to floating-point types like float or double,
    /// which can represent special values like the result of divide-by-zero.
    /// 
    /// This method exists mainly for consistency with other numeric types where IsNaN is meaningful.
    /// </para>
    /// </remarks>
    public bool IsNaN(ushort value) => false;

    /// <summary>
    /// Determines if a ushort value is infinity.
    /// </summary>
    /// <param name="value">The value to check.</param>
    /// <returns>Always <c>false</c> for ushort values.</returns>
    /// <remarks>
    /// <para>
    /// This method always returns false because the ushort data type can only represent integers,
    /// and the concept of infinity only applies to floating-point types like float and double.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if a number is "infinity".
    /// 
    /// For ushort values, the result is always false because a ushort can only contain finite whole numbers.
    /// The concept of "infinity" applies only to floating-point types like float or double,
    /// which can represent special values like the result of divide-by-zero.
    /// 
    /// This method exists mainly for consistency with other numeric types where IsInfinity is meaningful.
    /// </para>
    /// </remarks>
    public bool IsInfinity(ushort value) => false;

    /// <summary>
    /// Returns the sign of a ushort value as 0 or 1.
    /// </summary>
    /// <param name="value">The value to determine the sign of.</param>
    /// <returns>
    /// 0 if <paramref name="value"/> is zero;
    /// 1 if <paramref name="value"/> is positive.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method returns a value indicating the sign of the input value. Since ushort can only
    /// represent non-negative values, the result will always be either 0 (for zero) or 1 (for positive values).
    /// This is different from signed numeric types where the result could also be -1 for negative values.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells you if a number is positive or zero.
    /// 
    /// It returns:
    /// - 0 if the number is exactly zero
    /// - 1 if the number is positive (greater than zero)
    /// 
    /// Since ushort can only store values that are zero or positive, you'll never get a -1 result
    /// (which would represent a negative number in other numeric types).
    /// 
    /// For example:
    /// - SignOrZero(0) returns 0
    /// - SignOrZero(42) returns 1
    /// - SignOrZero(65535) returns 1
    /// </para>
    /// </remarks>
    public ushort SignOrZero(ushort value) => value == 0 ? (ushort)0 : (ushort)1;

    /// <summary>
    /// Gets the number of bits used for precision in ushort (16 bits).
    /// </summary>
    public int PrecisionBits => 16;

    /// <summary>
    /// Converts a ushort value to float (FP32) precision.
    /// </summary>
    public float ToFloat(ushort value) => (float)value;

    /// <summary>
    /// Converts a float value to ushort.
    /// </summary>
    public ushort FromFloat(float value) => (ushort)MathExtensions.Clamp((int)Math.Round(value), ushort.MinValue, ushort.MaxValue);

    /// <summary>
    /// Converts a ushort value to Half (FP16) precision.
    /// </summary>
    public Half ToHalf(ushort value) => (Half)value;

    /// <summary>
    /// Converts a Half value to ushort.
    /// </summary>
    public ushort FromHalf(Half value) => (ushort)MathExtensions.Clamp((int)Math.Round((float)value), ushort.MinValue, ushort.MaxValue);

    /// <summary>
    /// Converts a ushort value to double (FP64) precision.
    /// </summary>
    public double ToDouble(ushort value) => (double)value;

    /// <inheritdoc/>
    public bool SupportsCpuAcceleration => false;

    /// <inheritdoc/>
    public bool SupportsGpuAcceleration => false;

    #region IVectorizedOperations<ushort> Implementation

    /// <summary>
    /// Performs element-wise addition using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public void Add(ReadOnlySpan<ushort> x, ReadOnlySpan<ushort> y, Span<ushort> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<AddOperatorUShort>(x, y, destination);

    /// <summary>
    /// Performs element-wise subtraction using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public void Subtract(ReadOnlySpan<ushort> x, ReadOnlySpan<ushort> y, Span<ushort> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<SubtractOperatorUShort>(x, y, destination);

    /// <summary>
    /// Performs element-wise multiplication using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public void Multiply(ReadOnlySpan<ushort> x, ReadOnlySpan<ushort> y, Span<ushort> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<MultiplyOperatorUShort>(x, y, destination);

    /// <summary>
    /// Performs element-wise division using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public void Divide(ReadOnlySpan<ushort> x, ReadOnlySpan<ushort> y, Span<ushort> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<DivideOperatorUShort>(x, y, destination);

    /// <summary>
    /// Computes dot product using sequential loops.
    /// </summary>
    public ushort Dot(ReadOnlySpan<ushort> x, ReadOnlySpan<ushort> y)
        => VectorizedOperationsFallback.Dot(this, x, y);

    /// <summary>
    /// Computes sum using sequential loops.
    /// </summary>
    public ushort Sum(ReadOnlySpan<ushort> x)
        => VectorizedOperationsFallback.Sum(this, x);

    /// <summary>
    /// Finds maximum using sequential loops.
    /// </summary>
    public ushort Max(ReadOnlySpan<ushort> x)
        => VectorizedOperationsFallback.Max(this, x);

    /// <summary>
    /// Finds minimum using sequential loops.
    /// </summary>
    public ushort Min(ReadOnlySpan<ushort> x)
        => VectorizedOperationsFallback.Min(this, x);

    /// <summary>
    /// Computes exponential using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Exp(ReadOnlySpan<ushort> x, Span<ushort> destination)
        => VectorizedOperationsFallback.Exp(this, x, destination);

    /// <summary>
    /// Computes natural logarithm using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Log(ReadOnlySpan<ushort> x, Span<ushort> destination)
        => VectorizedOperationsFallback.Log(this, x, destination);

    /// <summary>
    /// Computes hyperbolic tangent using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Tanh(ReadOnlySpan<ushort> x, Span<ushort> destination)
        => VectorizedOperationsFallback.Tanh(this, x, destination);

    /// <summary>
    /// Computes sigmoid using sequential loops (integers don't support transcendental SIMD).
    /// </summary>
    public void Sigmoid(ReadOnlySpan<ushort> x, Span<ushort> destination)
        => VectorizedOperationsFallback.Sigmoid(this, x, destination);

    /// <summary>
    /// Computes base-2 logarithm using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Log2(ReadOnlySpan<ushort> x, Span<ushort> destination)
        => VectorizedOperationsFallback.Log2(this, x, destination);

    /// <summary>
    /// Computes softmax using sequential loops (integers don't support transcendental SIMD).
    /// </summary>
    public void SoftMax(ReadOnlySpan<ushort> x, Span<ushort> destination)
        => VectorizedOperationsFallback.SoftMax(this, x, destination);

    /// <summary>
    /// Computes cosine similarity using sequential loops (integers don't support this SIMD operation).
    /// </summary>
    public ushort CosineSimilarity(ReadOnlySpan<ushort> x, ReadOnlySpan<ushort> y)
        => VectorizedOperationsFallback.CosineSimilarity(this, x, y);

    #endregion

    private static readonly UInt16Operations _instance = new UInt16Operations();

    public void Fill(Span<ushort> destination, ushort value) => destination.Fill(value);
    public void MultiplyScalar(ReadOnlySpan<ushort> x, ushort scalar, Span<ushort> destination) => VectorizedOperationsFallback.MultiplyScalar(_instance, x, scalar, destination);
    public void DivideScalar(ReadOnlySpan<ushort> x, ushort scalar, Span<ushort> destination) => VectorizedOperationsFallback.DivideScalar(_instance, x, scalar, destination);
    public void AddScalar(ReadOnlySpan<ushort> x, ushort scalar, Span<ushort> destination) => VectorizedOperationsFallback.AddScalar(_instance, x, scalar, destination);
    public void SubtractScalar(ReadOnlySpan<ushort> x, ushort scalar, Span<ushort> destination) => VectorizedOperationsFallback.SubtractScalar(_instance, x, scalar, destination);
    public void Sqrt(ReadOnlySpan<ushort> x, Span<ushort> destination) => VectorizedOperationsFallback.Sqrt(_instance, x, destination);
    public void Abs(ReadOnlySpan<ushort> x, Span<ushort> destination) => VectorizedOperationsFallback.Abs(_instance, x, destination);
    public void Negate(ReadOnlySpan<ushort> x, Span<ushort> destination) => VectorizedOperationsFallback.Negate(_instance, x, destination);
    public void Clip(ReadOnlySpan<ushort> x, ushort min, ushort max, Span<ushort> destination) => VectorizedOperationsFallback.Clip(_instance, x, min, max, destination);
    public void Pow(ReadOnlySpan<ushort> x, ushort power, Span<ushort> destination) => VectorizedOperationsFallback.Pow(_instance, x, power, destination);
    public void Copy(ReadOnlySpan<ushort> source, Span<ushort> destination) => source.CopyTo(destination);

    public void Floor(ReadOnlySpan<ushort> x, Span<ushort> destination) => x.CopyTo(destination);
    public void Ceiling(ReadOnlySpan<ushort> x, Span<ushort> destination) => x.CopyTo(destination);
    public void Frac(ReadOnlySpan<ushort> x, Span<ushort> destination) => destination.Fill(0);

}
