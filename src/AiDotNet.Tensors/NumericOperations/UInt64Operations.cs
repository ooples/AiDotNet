using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Operators;

namespace AiDotNet.Tensors.NumericOperations;

/// <summary>
/// Provides mathematical operations for the <see cref="ulong"/> (UInt64) data type.
/// </summary>
/// <remarks>
/// <para>
/// This class implements the <see cref="INumericOperations{T}"/> interface for the <see cref="ulong"/> type,
/// providing basic and advanced mathematical operations while handling the limitations of the unsigned long data type.
/// Since ulong values are limited to the range 0 to 18,446,744,073,709,551,615, operations that would result in values
/// outside this range will overflow and potentially produce unexpected results.
/// </para>
/// <para><b>For Beginners:</b> This class lets you perform math with unsigned long integers (whole numbers between 0 and approximately 18.4 quintillion).
/// 
/// Think of it like a calculator that works specifically with very large positive whole numbers. For example:
/// - You can add, subtract, multiply, and divide ulong numbers
/// - You can compare values (is one number greater than another?)
/// - You can perform more advanced operations like square roots or exponents
/// 
/// However, be careful! If your calculations produce a number larger than 18,446,744,073,709,551,615 or a negative number,
/// the result will "wrap around" (overflow) and might give you an unexpected answer. This is like
/// a car odometer that rolls over to 0 after reaching its maximum value.
/// 
/// The ulong type is useful when you need to work with very large positive numbers, such as:
/// - Unique identifiers in large databases
/// - Calculations involving extremely large counts
/// - Working with file sizes or memory addresses
/// </para>
/// </remarks>
public class UInt64Operations : INumericOperations<ulong>
{
    /// <summary>
    /// Adds two ulong values.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    /// <returns>The sum of <paramref name="a"/> and <paramref name="b"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method performs addition on two ulong values. If the result exceeds the maximum value of a ulong
    /// (18,446,744,073,709,551,615), an overflow will occur, wrapping the result around to start from zero again.
    /// </para>
    /// <para><b>For Beginners:</b> This method adds two numbers together.
    /// 
    /// For example:
    /// - Add(5, 3) returns 8
    /// - Add(1000000, 2000000) returns 3000000
    /// 
    /// Be careful with large numbers! If the result is too big for a ulong, it will wrap around:
    /// - Add(18446744073709551610, 10) would mathematically be 18446744073709551620, but since that's too large,
    ///   it will return 4 (the result after "wrapping around" from zero again)
    /// </para>
    /// </remarks>
    public ulong Add(ulong a, ulong b) => a + b;

    /// <summary>
    /// Subtracts the second value from the first.
    /// </summary>
    /// <param name="a">The value to subtract from.</param>
    /// <param name="b">The value to subtract.</param>
    /// <returns>The difference between <paramref name="a"/> and <paramref name="b"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method performs subtraction of two ulong values. If the result would be negative (when b > a),
    /// an overflow will occur, wrapping the result around to a large positive number. This is because ulong
    /// cannot represent negative values.
    /// </para>
    /// <para><b>For Beginners:</b> This method subtracts the second number from the first.
    /// 
    /// For example:
    /// - Subtract(10, 3) returns 7
    /// - Subtract(20, 5) returns 15
    /// 
    /// Be careful when the second number is larger than the first! Since a ulong can't be negative:
    /// - Subtract(5, 10) will not return -5. Instead, it will return 18,446,744,073,709,551,611
    ///   (which is 18,446,744,073,709,551,616 - 5)
    /// 
    /// This happens because the result wraps around from the end of the range to the beginning.
    /// </para>
    /// </remarks>
    public ulong Subtract(ulong a, ulong b) => a - b;

    /// <summary>
    /// Multiplies two ulong values.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    /// <returns>The product of <paramref name="a"/> and <paramref name="b"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method performs multiplication of two ulong values. The result of multiplying two ulong values can
    /// easily exceed the range of a ulong, causing overflow and potentially returning an unexpected value.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies two numbers together.
    /// 
    /// For example:
    /// - Multiply(4, 5) returns 20
    /// - Multiply(10, 3) returns 30
    /// 
    /// Multiplication can easily produce numbers that are too large for a ulong:
    /// - Multiply(10,000,000,000, 2,000) would be 20,000,000,000,000, which is within the ulong range
    /// - But Multiply(10,000,000,000,000, 2,000) would likely exceed the range and give an incorrect result
    /// </para>
    /// </remarks>
    public ulong Multiply(ulong a, ulong b) => a * b;

    /// <summary>
    /// Divides the first value by the second.
    /// </summary>
    /// <param name="a">The dividend (value to be divided).</param>
    /// <param name="b">The divisor (value to divide by).</param>
    /// <returns>The quotient of <paramref name="a"/> divided by <paramref name="b"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method performs integer division of two ulong values. Because ulong is an integer type, 
    /// the result will be truncated (rounded down). Division by zero will throw a DivideByZeroException.
    /// </para>
    /// <para><b>For Beginners:</b> This method divides the first number by the second.
    /// 
    /// For example:
    /// - Divide(10, 2) returns 5
    /// - Divide(7, 2) returns 3 (not 3.5, since ulong values are whole numbers only)
    /// 
    /// Important notes:
    /// - The result is always rounded down to the nearest whole number
    /// - Dividing by zero will cause your program to crash with an error
    /// </para>
    /// </remarks>
    public ulong Divide(ulong a, ulong b) => a / b;

    /// <summary>
    /// Negates a ulong value.
    /// </summary>
    /// <param name="a">The value to negate.</param>
    /// <returns>The two's complement negation of <paramref name="a"/>.</returns>
    /// <remarks>
    /// <para>
    /// Since ulong cannot represent negative values, this method performs a two's complement negation.
    /// For a value 'a', it returns (ulong.MaxValue - a + 1), which is equivalent to (2^64 - a) when 
    /// represented in the full 64-bit range. This operation has the property that a + Negate(a) = 0 
    /// (after overflow).
    /// </para>
    /// <para><b>For Beginners:</b> This method attempts to find the "negative" of an unsigned number.
    /// 
    /// Since ulong can only store positive numbers, true negation isn't possible. Instead, this method
    /// uses a technique called "two's complement" to find the value that, when added to the original number,
    /// gives zero in the ulong range.
    /// 
    /// For example:
    /// - Negate(1) returns 18,446,744,073,709,551,615 (because 1 + 18,446,744,073,709,551,615 = 18,446,744,073,709,551,616, which overflows to 0 in ulong)
    /// - Negate(1000) returns 18,446,744,073,709,550,616 (because 1000 + 18,446,744,073,709,550,616 = 18,446,744,073,709,551,616, which overflows to 0 in ulong)
    /// 
    /// This operation is mostly used in specific bit manipulation contexts or when implementing
    /// certain algorithms that require a "wraparound" behavior.
    /// </para>
    /// </remarks>
    public ulong Negate(ulong a) => ulong.MaxValue - a + 1;

    /// <summary>
    /// Gets the value zero as a ulong.
    /// </summary>
    /// <value>The value 0 as a ulong.</value>
    /// <remarks>
    /// <para>
    /// This property returns the value zero (0) as a ulong. It is useful for operations that
    /// require a zero value, such as initializing variables or as a default value.
    /// </para>
    /// <para><b>For Beginners:</b> This property simply gives you the number zero (0) as a ulong.
    /// 
    /// This is useful when you need a known zero value in your code, for example:
    /// - When starting a counter
    /// - When you need to initialize a value before calculating
    /// - As a default or fallback value
    /// </para>
    /// </remarks>
    public ulong Zero => 0;

    /// <summary>
    /// Gets the value one as a ulong.
    /// </summary>
    /// <value>The value 1 as a ulong.</value>
    /// <remarks>
    /// <para>
    /// This property returns the value one (1) as a ulong. It is useful for operations that
    /// require a unit value, such as incrementing a counter or as an identity element in multiplication.
    /// </para>
    /// <para><b>For Beginners:</b> This property simply gives you the number one (1) as a ulong.
    /// 
    /// This is useful in many situations:
    /// - When incrementing a counter (adding 1)
    /// - In mathematical formulas that need the number 1
    /// - As a starting value for multiplication
    /// </para>
    /// </remarks>
    public ulong One => 1;

    /// <summary>
    /// Calculates the square root of a ulong value.
    /// </summary>
    /// <param name="value">The value to calculate the square root of.</param>
    /// <returns>The square root of <paramref name="value"/> as a ulong.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the square root of the input value and converts the result to a ulong.
    /// The calculation is performed using double-precision arithmetic and then cast to a ulong, which means
    /// the result will be truncated to an integer value. For very large ulong values, there may be some
    /// precision loss in the intermediate double conversion.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the square root of a number.
    /// 
    /// The square root of a number is another number that, when multiplied by itself, gives the original number.
    /// 
    /// For example:
    /// - Sqrt(4) returns 2 (because 2 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 2 = 4)
    /// - Sqrt(9) returns 3 (because 3 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 3 = 9)
    /// - Sqrt(10) returns 3 (because the true square root is approximately 3.16, but as a ulong it's rounded down to 3)
    /// 
    /// For very large numbers, the result might not be perfectly accurate because of how the calculation
    /// is performed internally using double-precision floating-point.
    /// </para>
    /// </remarks>
    public ulong Sqrt(ulong value) => (ulong)Math.Sqrt(value);

    /// <summary>
    /// Converts a double value to a ulong.
    /// </summary>
    /// <param name="value">The double value to convert.</param>
    /// <returns>The double value converted to a ulong.</returns>
    /// <remarks>
    /// <para>
    /// This method converts a double-precision floating-point value to a ulong. The conversion truncates
    /// the fractional part of the double. Negative values will underflow to a large positive value, and values
    /// greater than 18,446,744,073,709,551,615 will overflow.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts a decimal number to a whole ulong number.
    /// 
    /// When converting:
    /// - The decimal part is dropped (not rounded)
    /// - If the number is negative, you'll get an unexpected large positive number
    /// - If the number is too large for a ulong, you'll get an unexpected smaller result
    /// 
    /// For example:
    /// - FromDouble(5.7) returns 5 (decimal part is simply dropped)
    /// - FromDouble(3.2) returns 3
    /// - FromDouble(-5.0) will not return -5 (since ulong can't store negative numbers),
    ///   but instead a very large positive number
    /// </para>
    /// </remarks>
    public ulong FromDouble(double value) => (ulong)value;

    /// <summary>
    /// Determines if the first value is greater than the second.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns><c>true</c> if <paramref name="a"/> is greater than <paramref name="b"/>; otherwise, <c>false</c>.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two ulong values and returns true if the first value is greater than the second.
    /// The comparison uses the standard greater than operator for ulong values.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is bigger than the second.
    /// 
    /// For example:
    /// - GreaterThan(10, 5) returns true (because 10 is greater than 5)
    /// - GreaterThan(3, 7) returns false (because 3 is not greater than 7)
    /// - GreaterThan(4, 4) returns false (because 4 is equal to 4, not greater than it)
    /// </para>
    /// </remarks>
    public bool GreaterThan(ulong a, ulong b) => a > b;

    /// <summary>
    /// Determines if the first value is less than the second.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns><c>true</c> if <paramref name="a"/> is less than <paramref name="b"/>; otherwise, <c>false</c>.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two ulong values and returns true if the first value is less than the second.
    /// The comparison uses the standard less than operator for ulong values.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is smaller than the second.
    /// 
    /// For example:
    /// - LessThan(5, 10) returns true (because 5 is less than 10)
    /// - LessThan(7, 3) returns false (because 7 is not less than 3)
    /// - LessThan(4, 4) returns false (because 4 is equal to 4, not less than it)
    /// </para>
    /// </remarks>
    public bool LessThan(ulong a, ulong b) => a < b;

    /// <summary>
    /// Calculates the absolute value of a ulong.
    /// </summary>
    /// <param name="value">The value to calculate the absolute value for.</param>
    /// <returns>The input value unchanged.</returns>
    /// <remarks>
    /// <para>
    /// For ulong values, which are already non-negative, this method simply returns the input value unchanged.
    /// The absolute value function is traditionally used to get the non-negative version of a number, but
    /// since ulong values are always non-negative, no conversion is needed.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives you the positive version of a number.
    /// 
    /// The absolute value of a number is how far it is from zero, ignoring whether it's positive or negative.
    /// 
    /// For ulong values, which are always positive (or zero), this method simply returns the same number:
    /// - Abs(5) returns 5
    /// - Abs(0) returns 0
    /// 
    /// This method exists mainly for consistency with other numeric types where absolute value is meaningful.
    /// </para>
    /// </remarks>
    public ulong Abs(ulong value) => value;

    /// <summary>
    /// Squares a ulong value.
    /// </summary>
    /// <param name="value">The value to square.</param>
    /// <returns>The square of <paramref name="value"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the square of the input value (the value multiplied by itself).
    /// The result of squaring a ulong value can easily exceed the range of a ulong,
    /// causing overflow and potentially returning an unexpected value.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies a number by itself.
    /// 
    /// For example:
    /// - Square(4) returns 16 (because 4 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 4 = 16)
    /// - Square(10) returns 100 (because 10 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 10 = 100)
    /// 
    /// Be careful with larger numbers! Squaring even moderate values can easily exceed the ulong range:
    /// - Square(4,294,967,296) would be 18,446,744,073,709,551,616, which is just outside the ulong range,
    ///   so the result will be incorrect (it will wrap around to 0)
    /// </para>
    /// </remarks>
    public ulong Square(ulong value) => Multiply(value, value);

    /// <summary>
    /// Calculates e raised to the specified power.
    /// </summary>
    /// <param name="value">The power to raise e to.</param>
    /// <returns>The value of e raised to the power of <paramref name="value"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the exponential function (e^value) for the input value, where e is Euler's number
    /// (approximately 2.71828). The calculation is performed using double-precision arithmetic, rounded to the
    /// nearest integer, and then clamped to the maximum ulong value before casting to a ulong. This prevents
    /// overflow for large input values, instead returning ulong.MaxValue (18,446,744,073,709,551,615).
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates "e" raised to a power.
    /// 
    /// "e" is a special mathematical constant (approximately 2.71828) used in many calculations, especially
    /// those involving growth or decay.
    /// 
    /// For example:
    /// - Exp(1) returns 3 (because e^1 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 2.71828, rounded to 3 as a ulong)
    /// - Exp(2) returns 7 (because e^2 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 7.38906, rounded to 7 as a ulong)
    /// 
    /// For larger input values, the result grows very quickly:
    /// - Exp(10) returns 22,026 (because e^10 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 22,026.47)
    /// - Exp(43) or higher will return 18,446,744,073,709,551,615 (the maximum ulong value)
    ///   because the true result would be too large
    /// 
    /// This function is useful in calculations involving:
    /// - Compound interest with very large balances
    /// - Population growth of large populations
    /// - Any exponential growth scenario with large numbers
    /// </para>
    /// </remarks>
    public ulong Exp(ulong value) => (ulong)Math.Min(ulong.MaxValue, Math.Round(Math.Exp((double)value)));

    /// <summary>
    /// Determines if two ulong values are equal.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns><c>true</c> if <paramref name="a"/> is equal to <paramref name="b"/>; otherwise, <c>false</c>.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two ulong values for equality. Two ulong values are considered equal
    /// if they represent the same numeric value.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if two numbers are exactly the same.
    /// 
    /// For example:
    /// - Equals(5, 5) returns true (because both numbers are 5)
    /// - Equals(10, 15) returns false (because 10 and 15 are different numbers)
    /// - Equals(18446744073709551615, 18446744073709551615) returns true (even with very large numbers)
    /// </para>
    /// </remarks>
    public bool Equals(ulong a, ulong b) => a == b;

    public int Compare(ulong a, ulong b) => a.CompareTo(b);

    /// <summary>
    /// Raises a value to the specified power.
    /// </summary>
    /// <param name="baseValue">The base value.</param>
    /// <param name="exponent">The exponent.</param>
    /// <returns>The base value raised to the specified power.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the base value raised to the power of the exponent. The calculation is
    /// performed using double-precision arithmetic and then cast to a ulong, which may cause
    /// overflow for large results. Due to limitations of the double type, this method may lose precision
    /// for very large base values. Negative exponents will result in fractional values that,
    /// when cast to ulong, will become 0.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies a number by itself a specified number of times.
    /// 
    /// For example:
    /// - Power(2, 3) returns 8 (because 2ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â² = 2 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 2 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  2 = 8)
    /// - Power(3, 2) returns 9 (because 3ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â² = 3 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 3 = 9)
    /// - Power(5, 0) returns 1 (any number raised to the power of 0 is 1)
    /// 
    /// Be careful with larger values! The result can quickly exceed the ulong range:
    /// - Power(10, 19) would exceed the range of ulong, resulting in an incorrect value
    /// 
    /// Also note that this method may not be perfectly accurate for very large base values due to
    /// how the math is calculated internally.
    /// 
    /// Fractional results are truncated to whole numbers:
    /// - Power(2, 18446744073709551615) would mathematically be a tiny fraction, but as a ulong it returns 0
    /// </para>
    /// </remarks>
    public ulong Power(ulong baseValue, ulong exponent) => (ulong)Math.Pow(baseValue, exponent);

    /// <summary>
    /// Calculates the natural logarithm (base e) of a value.
    /// </summary>
    /// <param name="value">The value to calculate the logarithm for.</param>
    /// <returns>The natural logarithm of <paramref name="value"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the natural logarithm (ln) of the input value. The calculation is
    /// performed using double-precision arithmetic and then cast to a ulong. The result is truncated
    /// to an integer, leading to loss of precision. If the input is 0, the result will be a mathematical error
    /// (negative infinity), which typically becomes 0 when cast to a ulong.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the natural logarithm of a number.
    /// 
    /// The natural logarithm (ln) is the reverse of the exponential function. It tells you what power
    /// you need to raise "e" to in order to get your input value.
    /// 
    /// For example:
    /// - Log(1) returns 0 (because e^0 = 1)
    /// - Log(3) returns 1 (because e^1 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 2.71828, and when cast to a ulong, the decimal part is dropped)
    /// - Log(10) returns 2 (because e^2.303 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 10, and when cast to a ulong, the decimal part is dropped)
    /// 
    /// Important notes:
    /// - The logarithm of zero is not defined mathematically, so Log(0) will return 0
    /// - Logarithm results are usually decimals, but they'll be converted to whole numbers when stored as ulongs
    /// - Even for very large inputs, the result is relatively small (e.g., Log(18446744073709551615) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  44)
    /// </para>
    /// </remarks>
    public ulong Log(ulong value) => (ulong)Math.Log(value);

    /// <summary>
    /// Determines if the first value is greater than or equal to the second.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns><c>true</c> if <paramref name="a"/> is greater than or equal to <paramref name="b"/>; otherwise, <c>false</c>.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two ulong values and returns true if the first value is greater than or equal to the second.
    /// The comparison uses the standard greater than or equal to operator for ulong values.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is bigger than or the same as the second.
    /// 
    /// For example:
    /// - GreaterThanOrEquals(10, 5) returns true (because 10 is greater than 5)
    /// - GreaterThanOrEquals(7, 7) returns true (because 7 is equal to 7)
    /// - GreaterThanOrEquals(3, 8) returns false (because 3 is less than 8)
    /// </para>
    /// </remarks>
    public bool GreaterThanOrEquals(ulong a, ulong b) => a >= b;

    /// <summary>
    /// Determines if the first value is less than or equal to the second.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns><c>true</c> if <paramref name="a"/> is less than or equal to <paramref name="b"/>; otherwise, <c>false</c>.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two ulong values and returns true if the first value is less than or equal to the second.
    /// The comparison uses the standard less than or equal to operator for ulong values.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is smaller than or the same as the second.
    /// 
    /// For example:
    /// - LessThanOrEquals(5, 10) returns true (because 5 is less than 10)
    /// - LessThanOrEquals(7, 7) returns true (because 7 is equal to 7)
    /// - LessThanOrEquals(9, 4) returns false (because 9 is greater than 4)
    /// </para>
    /// </remarks>
    public bool LessThanOrEquals(ulong a, ulong b) => a <= b;

    /// <summary>
    /// Converts a ulong value to a 32-bit integer.
    /// </summary>
    /// <param name="value">The ulong value to convert.</param>
    /// <returns>The ulong value as a 32-bit integer.</returns>
    /// <remarks>
    /// <para>
    /// This method converts a ulong (64-bit unsigned) value to an int (32-bit signed) value. The conversion may fail
    /// if the ulong value is greater than int.MaxValue (2,147,483,647), resulting in overflow. Values larger than
    /// int.MaxValue will be interpreted as negative values in the int type.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts a ulong number to a regular integer (int).
    /// 
    /// A ulong can store numbers from 0 to 18,446,744,073,709,551,615.
    /// An int can store numbers from -2,147,483,648 to 2,147,483,647.
    /// 
    /// This conversion is not always safe:
    /// - If the ulong value is less than or equal to 2,147,483,647, it converts correctly
    /// - If the ulong value is greater than 2,147,483,647, it will "wrap around" to a negative number
    /// 
    /// For example:
    /// - ToInt32(5) returns 5 as an int
    /// - ToInt32(1000) returns 1000 as an int
    /// - ToInt32(3,000,000,000) doesn't return 3,000,000,000 because that's too large for an int;
    ///   instead, it returns a negative number (-1,294,967,296)
    /// 
    /// Be very careful with this conversion, as it can easily produce unexpected results with larger values.
    /// </para>
    /// </remarks>
    public int ToInt32(ulong value) => (int)value;

    /// <summary>
    /// Rounds a ulong value.
    /// </summary>
    /// <param name="value">The value to round.</param>
    /// <returns>The rounded value.</returns>
    /// <remarks>
    /// <para>
    /// For ulong values, which are already integers, this method simply returns the value unchanged.
    /// Rounding only applies to floating-point values that have fractional parts.
    /// </para>
    /// <para><b>For Beginners:</b> This method rounds a number to the nearest whole number.
    /// 
    /// Since a ulong is already a whole number, this method simply returns the same number without any change.
    /// 
    /// For example:
    /// - Round(5) returns 5
    /// - Round(10) returns 10
    /// 
    /// This method exists mainly for consistency with other numeric types like float or double,
    /// where rounding would actually change the value.
    /// </para>
    /// </remarks>
    public ulong Round(ulong value) => value;

    public ulong Floor(ulong value) => value;
    public ulong Ceiling(ulong value) => value;
    public ulong Frac(ulong value) => 0;


    /// <summary>
    /// Gets the minimum value that can be represented by a ulong.
    /// </summary>
    /// <value>The minimum value of a ulong, which is 0.</value>
    /// <remarks>
    /// <para>
    /// This property returns the smallest possible value that can be represented by the ulong data type,
    /// which is 0. Unlike signed types, ulong cannot represent negative values.
    /// </para>
    /// <para><b>For Beginners:</b> This property gives you the smallest possible number that a ulong can hold.
    /// 
    /// For ulong values, the minimum value is always 0, because ulong can only store positive whole numbers
    /// (and zero).
    /// 
    /// This is useful when you need to:
    /// - Check if a value is valid for a ulong
    /// - Initialize a variable to the smallest possible value
    /// - Set boundaries for valid input values
    /// </para>
    /// </remarks>
    public ulong MinValue => ulong.MinValue;

    /// <summary>
    /// Gets the maximum value that can be represented by a ulong.
    /// </summary>
    /// <value>The maximum value of a ulong, which is 18,446,744,073,709,551,615.</value>
    /// <remarks>
    /// <para>
    /// This property returns the largest possible value that can be represented by the ulong data type,
    /// which is 18,446,744,073,709,551,615. Attempting to store a value greater than this in a ulong will result in overflow.
    /// </para>
    /// <para><b>For Beginners:</b> This property gives you the largest possible number that a ulong can hold.
    /// 
    /// For ulong values, the maximum value is 18,446,744,073,709,551,615 (over 18 quintillion).
    /// If you try to create a ulong with a larger value, the number will wrap around
    /// and give you an incorrect result.
    /// 
    /// This is useful when you need to:
    /// - Check if a value is too large to be stored as a ulong
    /// - Initialize a variable to the largest possible value before comparing
    /// - Set boundaries for valid input values
    /// 
    /// The ulong type can store much larger positive numbers than int, uint, or long, making it
    /// suitable for very large counts, IDs, or calculations that need to handle huge positive numbers.
    /// </para>
    /// </remarks>
    public ulong MaxValue => ulong.MaxValue;

    /// <summary>
    /// Determines if a ulong value is NaN (Not a Number).
    /// </summary>
    /// <param name="value">The value to check.</param>
    /// <returns>Always <c>false</c> for ulong values.</returns>
    /// <remarks>
    /// <para>
    /// This method always returns false because the ulong data type can only represent integers,
    /// and the concept of NaN (Not a Number) only applies to floating-point types like float and double.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if a number is "Not a Number" (NaN).
    /// 
    /// For ulong values, the result is always false because a ulong can only contain valid whole numbers.
    /// The concept of "Not a Number" applies only to floating-point types like float or double,
    /// which can represent special values like the result of divide-by-zero or the square root of a negative number.
    /// 
    /// This method exists mainly for consistency with other numeric types where IsNaN is meaningful.
    /// It allows code to be written that can work with different numeric types without needing special cases.
    /// </para>
    /// </remarks>
    public bool IsNaN(ulong value) => false;

    /// <summary>
    /// Determines if a ulong value is infinity.
    /// </summary>
    /// <param name="value">The value to check.</param>
    /// <returns>Always <c>false</c> for ulong values.</returns>
    /// <remarks>
    /// <para>
    /// This method always returns false because the ulong data type can only represent integers,
    /// and the concept of infinity only applies to floating-point types like float and double.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if a number is "infinity".
    /// 
    /// For ulong values, the result is always false because a ulong can only contain finite whole numbers.
    /// The concept of "infinity" applies only to floating-point types like float or double,
    /// which can represent special values like the result of divide-by-zero.
    /// 
    /// This method exists mainly for consistency with other numeric types where IsInfinity is meaningful.
    /// It allows generic algorithms to be written that can work with different numeric types.
    /// </para>
    /// </remarks>
    public bool IsInfinity(ulong value) => false;

    /// <summary>
    /// Returns the sign of a ulong value as 0 or 1.
    /// </summary>
    /// <param name="value">The value to determine the sign of.</param>
    /// <returns>
    /// 0 if <paramref name="value"/> is zero;
    /// 1 if <paramref name="value"/> is positive.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method returns a value indicating the sign of the input value. Since ulong can only
    /// represent non-negative values, the result will always be either 0 (for zero) or 1 (for positive values).
    /// This is different from signed numeric types where the result could also be -1 for negative values.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells you if a number is positive or zero.
    /// 
    /// It returns:
    /// - 0 if the number is exactly zero
    /// - 1 if the number is positive (greater than zero)
    /// 
    /// Since ulong can only store values that are zero or positive, you'll never get a -1 result
    /// (which would represent a negative number in other numeric types).
    /// 
    /// For example:
    /// - SignOrZero(0) returns 0
    /// - SignOrZero(42) returns 1
    /// - SignOrZero(18446744073709551615) returns 1
    /// 
    /// The suffix "ul" on the literals (0ul, 1ul) indicates that these are unsigned long integer values.
    /// </para>
    /// </remarks>
    public ulong SignOrZero(ulong value) => value == 0 ? 0ul : 1ul;

    /// <summary>
    /// Gets the number of bits used for precision in ulong (64 bits).
    /// </summary>
    public int PrecisionBits => 64;

    /// <summary>
    /// Converts a ulong value to float (FP32) precision.
    /// </summary>
    public float ToFloat(ulong value) => (float)value;

    /// <summary>
    /// Converts a float value to ulong with proper saturation.
    /// </summary>
    public ulong FromFloat(float value)
    {
        double rounded = Math.Round(value);

        if (double.IsNaN(rounded) || rounded <= 0d)
        {
            return 0ul;
        }

        if (rounded >= ulong.MaxValue)
        {
            return ulong.MaxValue;
        }

        return (ulong)rounded;
    }

    /// <summary>
    /// Converts a ulong value to Half (FP16) precision.
    /// </summary>
    public Half ToHalf(ulong value) => (Half)value;

    /// <summary>
    /// Converts a Half value to ulong with proper saturation.
    /// </summary>
    public ulong FromHalf(Half value)
    {
        double rounded = Math.Round((double)(float)value);

        if (double.IsNaN(rounded) || rounded <= 0d)
        {
            return 0ul;
        }

        if (rounded >= ulong.MaxValue)
        {
            return ulong.MaxValue;
        }

        return (ulong)rounded;
    }

    /// <summary>
    /// Converts a ulong value to double (FP64) precision.
    /// </summary>
    public double ToDouble(ulong value) => (double)value;

    /// <inheritdoc/>
    public bool SupportsCpuAcceleration => false;

    /// <inheritdoc/>
    public bool SupportsGpuAcceleration => false;

    #region IVectorizedOperations<ulong> Implementation

    /// <summary>
    /// Performs element-wise addition using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public void Add(ReadOnlySpan<ulong> x, ReadOnlySpan<ulong> y, Span<ulong> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<AddOperatorULong>(x, y, destination);

    /// <summary>
    /// Performs element-wise subtraction using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public void Subtract(ReadOnlySpan<ulong> x, ReadOnlySpan<ulong> y, Span<ulong> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<SubtractOperatorULong>(x, y, destination);

    /// <summary>
    /// Performs element-wise multiplication using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public void Multiply(ReadOnlySpan<ulong> x, ReadOnlySpan<ulong> y, Span<ulong> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<MultiplyOperatorULong>(x, y, destination);

    /// <summary>
    /// Performs element-wise division using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public void Divide(ReadOnlySpan<ulong> x, ReadOnlySpan<ulong> y, Span<ulong> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<DivideOperatorULong>(x, y, destination);

    /// <summary>
    /// Computes dot product using sequential loops.
    /// </summary>
    public ulong Dot(ReadOnlySpan<ulong> x, ReadOnlySpan<ulong> y)
        => VectorizedOperationsFallback.Dot(this, x, y);

    /// <summary>
    /// Computes sum using sequential loops.
    /// </summary>
    public ulong Sum(ReadOnlySpan<ulong> x)
        => VectorizedOperationsFallback.Sum(this, x);

    /// <summary>
    /// Finds maximum using sequential loops.
    /// </summary>
    public ulong Max(ReadOnlySpan<ulong> x)
        => VectorizedOperationsFallback.Max(this, x);

    /// <summary>
    /// Finds minimum using sequential loops.
    /// </summary>
    public ulong Min(ReadOnlySpan<ulong> x)
        => VectorizedOperationsFallback.Min(this, x);

    /// <summary>
    /// Transcendental operations are not supported for ulong type.
    /// </summary>
    /// <exception cref="NotSupportedException">Always thrown. Exp produces misleading results for ulong.</exception>
    public void Exp(ReadOnlySpan<ulong> x, Span<ulong> destination)
        => throw new NotSupportedException("Transcendental operations (Exp) are not meaningful for ulong type. Use float or double instead.");

    /// <summary>
    /// Transcendental operations are not supported for ulong type.
    /// </summary>
    /// <exception cref="NotSupportedException">Always thrown. Log produces misleading results for ulong.</exception>
    public void Log(ReadOnlySpan<ulong> x, Span<ulong> destination)
        => throw new NotSupportedException("Transcendental operations (Log) are not meaningful for ulong type. Use float or double instead.");

    /// <summary>
    /// Transcendental operations are not supported for ulong type.
    /// </summary>
    /// <exception cref="NotSupportedException">Always thrown. Tanh produces only 0 or 1 for ulong.</exception>
    public void Tanh(ReadOnlySpan<ulong> x, Span<ulong> destination)
        => throw new NotSupportedException("Transcendental operations (Tanh) are not meaningful for ulong type. Use float or double instead.");

    /// <summary>
    /// Transcendental operations are not supported for ulong type.
    /// </summary>
    /// <exception cref="NotSupportedException">Always thrown. Sigmoid saturates for ulong inputs.</exception>
    public void Sigmoid(ReadOnlySpan<ulong> x, Span<ulong> destination)
        => throw new NotSupportedException("Transcendental operations (Sigmoid) are not meaningful for ulong type. Use float or double instead.");

    /// <summary>
    /// Transcendental operations are not supported for ulong type.
    /// </summary>
    /// <exception cref="NotSupportedException">Always thrown. Log2 produces misleading results for ulong.</exception>
    public void Log2(ReadOnlySpan<ulong> x, Span<ulong> destination)
        => throw new NotSupportedException("Transcendental operations (Log2) are not meaningful for ulong type. Use float or double instead.");

    /// <summary>
    /// Transcendental operations are not supported for ulong type.
    /// </summary>
    /// <exception cref="NotSupportedException">Always thrown. SoftMax requires floating-point for normalized probabilities.</exception>
    public void SoftMax(ReadOnlySpan<ulong> x, Span<ulong> destination)
        => throw new NotSupportedException("Transcendental operations (SoftMax) are not meaningful for ulong type. Use float or double instead.");

    /// <summary>
    /// Computes cosine similarity using sequential loops.
    /// </summary>
    public ulong CosineSimilarity(ReadOnlySpan<ulong> x, ReadOnlySpan<ulong> y)
        => VectorizedOperationsFallback.CosineSimilarity(this, x, y);

    #endregion

    private static readonly UInt64Operations _instance = new UInt64Operations();

    public void Fill(Span<ulong> destination, ulong value) => destination.Fill(value);
    public void MultiplyScalar(ReadOnlySpan<ulong> x, ulong scalar, Span<ulong> destination) => VectorizedOperationsFallback.MultiplyScalar(_instance, x, scalar, destination);
    public void DivideScalar(ReadOnlySpan<ulong> x, ulong scalar, Span<ulong> destination) => VectorizedOperationsFallback.DivideScalar(_instance, x, scalar, destination);
    public void AddScalar(ReadOnlySpan<ulong> x, ulong scalar, Span<ulong> destination) => VectorizedOperationsFallback.AddScalar(_instance, x, scalar, destination);
    public void SubtractScalar(ReadOnlySpan<ulong> x, ulong scalar, Span<ulong> destination) => VectorizedOperationsFallback.SubtractScalar(_instance, x, scalar, destination);
    public void Sqrt(ReadOnlySpan<ulong> x, Span<ulong> destination) => VectorizedOperationsFallback.Sqrt(_instance, x, destination);
    public void Abs(ReadOnlySpan<ulong> x, Span<ulong> destination) => VectorizedOperationsFallback.Abs(_instance, x, destination);
    public void Negate(ReadOnlySpan<ulong> x, Span<ulong> destination) => VectorizedOperationsFallback.Negate(_instance, x, destination);
    public void Clip(ReadOnlySpan<ulong> x, ulong min, ulong max, Span<ulong> destination) => VectorizedOperationsFallback.Clip(_instance, x, min, max, destination);
    public void Pow(ReadOnlySpan<ulong> x, ulong power, Span<ulong> destination) => VectorizedOperationsFallback.Pow(_instance, x, power, destination);
    public void Copy(ReadOnlySpan<ulong> source, Span<ulong> destination) => source.CopyTo(destination);

    public void Floor(ReadOnlySpan<ulong> x, Span<ulong> destination) => x.CopyTo(destination);
    public void Ceiling(ReadOnlySpan<ulong> x, Span<ulong> destination) => x.CopyTo(destination);
    public void Frac(ReadOnlySpan<ulong> x, Span<ulong> destination) => destination.Fill(0);

}
