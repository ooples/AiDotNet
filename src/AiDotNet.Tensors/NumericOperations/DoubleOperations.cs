using System;
#if NET8_0_OR_GREATER
using System.Numerics.Tensors;
#endif
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Operators;

namespace AiDotNet.Tensors.NumericOperations;
/// <summary>
/// Provides mathematical operations for the double data type.
/// </summary>
/// <remarks>
/// <para>
/// This class implements the INumericOperations interface for the double data type, providing
/// basic arithmetic operations, comparison methods, and mathematical functions. The double
/// type is a 64-bit floating-point type that can represent a wide range of values with
/// high precision, making it suitable for scientific and engineering calculations.
/// </para>
/// <para><b>For Beginners:</b> This class handles math operations for the double number type.
///
/// The double type in C# is designed for general-purpose calculations:
/// - It can represent very large numbers (up to approximately 1.8 ÃƒÆ’Ã¢â‚¬â€ 10^308)
/// - It can represent very small numbers (down to approximately 5.0 ÃƒÆ’Ã¢â‚¬â€ 10^-324)
/// - It stores decimal numbers with about 15-17 significant digits of precision
/// - It can represent special values like infinity and NaN (Not a Number)
///
/// However, doubles have some limitations:
/// - They can't represent all decimal fractions exactly (e.g., 0.1 + 0.2 doesn't equal exactly 0.3)
/// - They may introduce small rounding errors in calculations
///
/// Doubles are best used for scientific, engineering, or graphics calculations where high range
/// is more important than exact decimal representation.
/// </para>
/// </remarks>
public class DoubleOperations : INumericOperations<double>
{
    /// <summary>
    /// Adds two double values together.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    /// <returns>The sum of the two values.</returns>
    /// <remarks>
    /// <para>
    /// This method performs standard double addition. Due to the nature of floating-point representation,
    /// the result may include small rounding errors for certain values.
    /// </para>
    /// <para><b>For Beginners:</b> This method adds two double numbers together.
    ///
    /// For example:
    /// - 5.25 + 3.75 = 9.0
    /// - 0.1 + 0.2 = 0.30000000000000004 (not exactly 0.3, due to how doubles represent numbers)
    ///
    /// Be aware that doubles can have small precision errors with certain decimal fractions.
    /// </para>
    /// </remarks>

    public double Add(double a, double b) => a + b;
    /// <summary>
    /// Subtracts the second double value from the first.
    /// </summary>
    /// <param name="a">The value to subtract from.</param>
    /// <param name="b">The value to subtract.</param>
    /// <returns>The difference between the two values.</returns>
    /// <remarks>
    /// <para>
    /// This method performs standard double subtraction. Due to the nature of floating-point representation,
    /// the result may include small rounding errors for certain values.
    /// </para>
    /// <para><b>For Beginners:</b> This method subtracts one double number from another.
    /// 
    /// For example:
    /// - 10.0 - 3.25 = 6.75
    /// - 0.3 - 0.2 = 0.09999999999999998 (not exactly 0.1, due to how doubles represent numbers)
    /// 
    /// The small imprecisions in double arithmetic are usually negligible for most applications,
    /// but can accumulate in complex calculations or when exact decimal representation is required.
    /// </para>
    /// </remarks>
    public double Subtract(double a, double b) => a - b;

    /// <summary>
    /// Multiplies two double values together.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    /// <returns>The product of the two values.</returns>
    /// <remarks>
    /// <para>
    /// This method performs standard double multiplication. Due to the nature of floating-point representation,
    /// the result may include small rounding errors for certain values.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies two double numbers together.
    /// 
    /// For example:
    /// - 5.5 * 2.0 = 11.0
    /// - 0.1 * 0.1 = 0.010000000000000002 (not exactly 0.01)
    /// 
    /// Double multiplication can handle a very wide range of values, from very small to very large.
    /// If the result is too large to represent, it will become positive or negative infinity.
    /// </para>
    /// </remarks>
    public double Multiply(double a, double b) => a * b;

    /// <summary>
    /// Divides the first double value by the second.
    /// </summary>
    /// <param name="a">The dividend (value being divided).</param>
    /// <param name="b">The divisor (value to divide by).</param>
    /// <returns>The quotient of the division.</returns>
    /// <remarks>
    /// <para>
    /// This method performs double division. Division by zero results in either positive infinity,
    /// negative infinity, or NaN (Not a Number), depending on the sign of the dividend and whether
    /// it is zero.
    /// </para>
    /// <para><b>For Beginners:</b> This method divides one double number by another.
    /// 
    /// For example:
    /// - 10.0 / 2.0 = 5.0
    /// - 1.0 / 3.0 = 0.3333333333333333 (approximately 1/3)
    /// 
    /// Special cases for division:
    /// - Dividing a non-zero number by zero gives infinity (positive or negative, depending on signs)
    /// - Dividing zero by zero gives NaN (Not a Number)
    /// 
    /// Unlike integer division, double division never throws an exception for division by zero.
    /// </para>
    /// </remarks>
    public double Divide(double a, double b) => a / b;

    /// <summary>
    /// Negates the specified double value.
    /// </summary>
    /// <param name="a">The value to negate.</param>
    /// <returns>The negated value.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the arithmetic negation of the input value, effectively changing its sign.
    /// If the input is positive, the result is negative, and vice versa. Zero remains zero when negated,
    /// though the distinction between positive and negative zero is preserved.
    /// </para>
    /// <para><b>For Beginners:</b> This method reverses the sign of a double number.
    /// 
    /// For example:
    /// - Negate(5.25) = -5.25
    /// - Negate(-10.5) = 10.5
    /// - Negate(0.0) = -0.0 (negative zero, which behaves like zero in most contexts)
    /// 
    /// This is the same as multiplying the number by -1.
    /// </para>
    /// </remarks>
    public double Negate(double a) => -a;

    /// <summary>
    /// Gets the double representation of zero.
    /// </summary>
    /// <value>The value 0 as a double.</value>
    /// <remarks>
    /// <para>
    /// This property returns the double representation of the value zero, which is 0.0.
    /// It is often used as a neutral element for addition.
    /// </para>
    /// <para><b>For Beginners:</b> This property provides the value zero as a double.
    /// 
    /// Zero is a special value in mathematics:
    /// - Adding zero to any number gives the same number
    /// - It's used as a starting point in many algorithms
    /// 
    /// In doubles, there's technically a positive zero and a negative zero (0.0 and -0.0),
    /// though they behave the same in most operations.
    /// </para>
    /// </remarks>
    public double Zero => 0;

    /// <summary>
    /// Gets the double representation of one.
    /// </summary>
    /// <value>The value 1 as a double.</value>
    /// <remarks>
    /// <para>
    /// This property returns the double representation of the value one, which is 1.0.
    /// It is often used as a neutral element for multiplication.
    /// </para>
    /// <para><b>For Beginners:</b> This property provides the value one as a double.
    /// 
    /// One is a special value in mathematics:
    /// - Multiplying any number by one gives the same number
    /// - It's useful as a starting point or increment value
    /// 
    /// This property gives you the value 1.0 as a double.
    /// </para>
    /// </remarks>
    public double One => 1;

    /// <summary>
    /// Calculates the square root of a double value.
    /// </summary>
    /// <param name="value">The value to calculate the square root of.</param>
    /// <returns>The square root of the specified value.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the square root using Math.Sqrt. If the input is negative,
    /// the result will be NaN (Not a Number).
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the square root of a double number.
    /// 
    /// For example:
    /// - Square root of 9.0 is 3.0
    /// - Square root of 2.0 is approximately 1.4142135623730951
    /// 
    /// If you try to take the square root of a negative number, the result is NaN (Not a Number),
    /// which represents an invalid mathematical operation.
    /// </para>
    /// </remarks>
    public double Sqrt(double value) => Math.Sqrt(value);

    /// <summary>
    /// Converts a double value to a double.
    /// </summary>
    /// <param name="value">The double value to convert.</param>
    /// <returns>The input value (unchanged).</returns>
    /// <remarks>
    /// <para>
    /// Since the input is already a double, this method simply returns the input value unchanged.
    /// It exists to fulfill the interface contract.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts a double to a double (no actual conversion happens).
    /// 
    /// Since the input is already a double, this method simply returns the same value.
    /// 
    /// This method exists to comply with the INumericOperations interface, which requires
    /// a method to convert from double to the specific numeric type.
    /// </para>
    /// </remarks>
    public double FromDouble(double value) => value;

    /// <summary>
    /// Determines whether the first double value is greater than the second.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns>true if the first value is greater than the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two double values and returns true if the first is greater than the second.
    /// Special values like NaN follow IEEE 754 comparison rules, where NaN is not greater than any value.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is larger than the second.
    /// 
    /// For example:
    /// - 10.5 > 5.2 returns true
    /// - 5.0 > 10.0 returns false
    /// - 5.0 > 5.0 returns false
    /// 
    /// Special cases:
    /// - Any comparison with NaN returns false, even NaN > NaN
    /// - Positive infinity is greater than any finite number
    /// - Negative infinity is less than any finite number
    /// </para>
    /// </remarks>
    public bool GreaterThan(double a, double b) => a > b;

    /// <summary>
    /// Determines whether the first double value is less than the second.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns>true if the first value is less than the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two double values and returns true if the first is less than the second.
    /// Special values like NaN follow IEEE 754 comparison rules, where NaN is not less than any value.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is smaller than the second.
    /// 
    /// For example:
    /// - 5.2 < 10.5 returns true
    /// - 10.0 < 5.0 returns false
    /// - 5.0 < 5.0 returns false
    /// 
    /// Special cases:
    /// - Any comparison with NaN returns false, even NaN < NaN
    /// - Positive infinity is greater than any finite number
    /// - Negative infinity is less than any finite number
    /// </para>
    /// </remarks>
    public bool LessThan(double a, double b) => a < b;

    /// <summary>
    /// Returns the absolute value of a double.
    /// </summary>
    /// <param name="value">The double value.</param>
    /// <returns>The absolute value of the specified double.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the absolute (positive) value of the specified double value.
    /// If the value is already positive or zero, it is returned unchanged. If the value is negative,
    /// its negation is returned. NaN values remain NaN.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides the positive version of a number.
    /// 
    /// For example:
    /// - Abs(5.25) = 5.25 (already positive, so unchanged)
    /// - Abs(-5.25) = 5.25 (negative becomes positive)
    /// - Abs(0.0) = 0.0 (zero remains zero)
    /// - Abs(NaN) = NaN (Not a Number remains Not a Number)
    /// 
    /// The absolute value is the distance of a number from zero, ignoring the direction.
    /// </para>
    /// </remarks>
    public double Abs(double value) => Math.Abs(value);

    /// <summary>
    /// Squares the specified double value.
    /// </summary>
    /// <param name="value">The value to square.</param>
    /// <returns>The square of the specified value.</returns>
    /// <remarks>
    /// <para>
    /// This method multiplies the value by itself to calculate its square.
    /// If the result is too large to represent as a double, it will become positive infinity.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies a number by itself.
    /// 
    /// For example:
    /// - Square of 4.0 is 16.0 (4.0 ÃƒÆ’Ã¢â‚¬â€ 4.0)
    /// - Square of 0.5 is 0.25 (0.5 ÃƒÆ’Ã¢â‚¬â€ 0.5)
    /// - Square of -3.0 is 9.0 (-3.0 ÃƒÂ¢Ã¢â‚¬Â°Ã‹â€  -3.0)
    /// 
    /// Squaring always produces a non-negative result (except for NaN, which remains NaN).
    /// If the result is too large to represent (over 1.8 ÃƒÆ’Ã¢â‚¬â€ 10^308), it becomes positive infinity.
    /// </para>
    /// </remarks>
    public double Square(double value) => Multiply(value, value);

    /// <summary>
    /// Calculates e raised to the specified power.
    /// </summary>
    /// <param name="value">The power to raise e to.</param>
    /// <returns>e raised to the specified power.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the exponential function (e^value) using Math.Exp.
    /// The constant e is approximately 2.71828. For large positive inputs, the result may become infinity.
    /// For large negative inputs, the result approaches zero.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the mathematical constant e (ÃƒÂ¢Ã¢â‚¬Â°Ã‹â€ 2.718) raised to a power.
    /// 
    /// For example:
    /// - e^1 ÃƒÆ’Ã¢â‚¬â€ 2.718
    /// - e^2 ÃƒÆ’Ã¢â‚¬â€ 7.389
    /// - e^0 = 1.0 exactly
    /// - e^-1 ÃƒÆ’Ã¢â‚¬â€ 0.368
    /// 
    /// The exponential function is used in many fields including finance (compound interest),
    /// science, and engineering. It grows very rapidly as the input increases.
    /// </para>
    /// </remarks>
    public double Exp(double value) => Math.Exp(value);

    /// <summary>
    /// Determines whether two double values are equal.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns>true if the values are equal; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two double values and returns true if they are exactly equal.
    /// Due to the nature of floating-point representation, comparing doubles for exact equality
    /// can be problematic. In many cases, it's better to check if the difference between two
    /// values is less than a small epsilon value.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if two numbers have exactly the same value.
    /// 
    /// For example:
    /// - 5.25 equals 5.25 returns true
    /// - 5.25 equals 5.250 returns true (same value, different representation)
    /// - 5.25 equals 5.24 returns false
    /// 
    /// Special cases:
    /// - NaN never equals anything, even itself
    /// - Positive and negative zero are considered equal
    /// 
    /// Be cautious when comparing doubles for equality, as rounding errors can make
    /// calculations that should be equal appear slightly different.
    /// For example, 0.1 + 0.2 does not exactly equal 0.3 in double arithmetic.
    /// </para>
    /// </remarks>
    public bool Equals(double a, double b) => a == b;

    /// <summary>
    /// Raises a double value to the specified power.
    /// </summary>
    /// <param name="baseValue">The base value.</param>
    /// <param name="exponent">The exponent.</param>
    /// <returns>The base value raised to the specified power.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates baseValue^exponent using Math.Pow. Special cases include:
    /// 0^0 = 1, x^0 = 1 for any x, 0^x = 0 for x > 0, 0^x = infinity for x < 0.
    /// Negative base values with non-integer exponents result in NaN.
    /// </para>
    /// <para><b>For Beginners:</b> This method raises one number to the power of another.
    /// 
    /// For example:
    /// - 2.0 raised to power 3.0 is 8.0 (2^3 = 2ÃƒÆ’Ã¢â‚¬â€2 ÃƒÆ’Ã¢â‚¬â€ 2 = 8)
    /// - 10.0 raised to power 2.0 is 100.0 (10^2 = 10ÃƒÆ’Ã¢â‚¬â€10 = 100)
    /// - Any number raised to power 0.0 is 1.0
    /// - Any number raised to power 1.0 is that number itself
    /// 
    /// Special cases:
    /// - 0.0 raised to a negative power gives positive infinity
    /// - Negative numbers raised to fractional powers give NaN (Not a Number)
    /// - Very large results may become infinity
    /// </para>
    /// </remarks>
    public double Power(double baseValue, double exponent) => Math.Pow(baseValue, exponent);

    /// <summary>
    /// Calculates the natural logarithm of a double value.
    /// </summary>
    /// <param name="value">The value to calculate the natural logarithm of.</param>
    /// <returns>The natural logarithm of the specified value.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the natural logarithm (base e) of the specified value using Math.Log.
    /// If the input is negative or zero, the result will be NaN or negative infinity, respectively.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the natural logarithm of a number.
    /// 
    /// The natural logarithm answers the question: "To what power must e be raised to get this number?"
    /// 
    /// For example:
    /// - Log of 1.0 is 0.0 (e^0 = 1)
    /// - Log of 2.718... is approximately 1.0 (e^1 ÃƒÆ’Ã¢â‚¬â€ 2.718)
    /// - Log of 7.389... is approximately 2.0 (e^2 ÃƒÆ’Ã¢â‚¬â€ 7.389)
    /// 
    /// Special cases:
    /// - Log of a negative number gives NaN (Not a Number)
    /// - Log of zero gives negative infinity
    /// - Log of infinity gives positive infinity
    /// 
    /// The logarithm function is the inverse of the exponential function.
    /// </para>
    /// </remarks>
    public double Log(double value) => Math.Log(value);

    /// <summary>
    /// Determines whether the first double value is greater than or equal to the second.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns>true if the first value is greater than or equal to the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two double values and returns true if the first is greater than or equal to the second.
    /// Special values like NaN follow IEEE 754 comparison rules, where NaN is not comparable to any value.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is larger than or the same as the second.
    /// 
    /// For example:
    /// - 10.5 >= 5.2 returns true
    /// - 5.0 >= 10.0 returns false
    /// - 5.0 >= 5.0 returns true
    /// 
    /// Special cases:
    /// - Any comparison with NaN returns false, even NaN >= NaN
    /// - Positive infinity is greater than any finite number
    /// - Negative infinity is less than any finite number
    /// </para>
    /// </remarks>
    public bool GreaterThanOrEquals(double a, double b) => a >= b;

    /// <summary>
    /// Determines whether the first double value is less than or equal to the second.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns>true if the first value is less than or equal to the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two double values and returns true if the first is less than or equal to the second.
    /// Special values like NaN follow IEEE 754 comparison rules, where NaN is not comparable to any value.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is smaller than or the same as the second.
    /// 
    /// For example:
    /// - 5.2 <= 10.5 returns true
    /// - 10.0 <= 5.0 returns false
    /// - 5.0 <= 5.0 returns true
    /// 
    /// Special cases:
    /// - Any comparison with NaN returns false, even NaN <= NaN
    /// - Positive infinity is greater than any finite number
    /// - Negative infinity is less than any finite number
    /// </para>
    /// </remarks>
    public bool LessThanOrEquals(double a, double b) => a <= b;

    /// <summary>
    /// Converts a double value to a 32-bit integer by rounding to the nearest integer.
    /// </summary>
    /// <param name="value">The double value to convert.</param>
    /// <returns>The double rounded to the nearest integer and converted to an Int32.</returns>
    /// <remarks>
    /// <para>
    /// This method rounds the double value to the nearest integer and then converts it to an Int32.
    /// If the result is outside the range of Int32, an OverflowException will be thrown.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts a double to a regular integer.
    /// 
    /// For example:
    /// - 5.7 becomes 6
    /// - 5.2 becomes 5
    /// - 5.5 becomes 6 (rounds to the nearest even number when exactly halfway)
    /// 
    /// This is useful when you need an integer result after performing floating-point calculations.
    /// 
    /// Note: If the double value is too large or too small to fit in an integer
    /// (outside the range of approximately Ãƒâ€šÃ‚Â±2.1 billion), this will cause an error.
    /// </para>
    /// </remarks>
    public int ToInt32(double value) => (int)Math.Round(value);

    /// <summary>
    /// Rounds a double value to the nearest integer.
    /// </summary>
    /// <param name="value">The double value to round.</param>
    /// <returns>The double value rounded to the nearest integer.</returns>
    /// <remarks>
    /// <para>
    /// This method rounds the double value to the nearest integer, following the "banker's rounding" rules.
    /// If the fractional part is exactly 0.5, it rounds to the nearest even number.
    /// </para>
    /// <para><b>For Beginners:</b> This method rounds a double to the nearest whole number.
    /// 
    /// For example:
    /// - Round(5.7) = 6.0
    /// - Round(5.2) = 5.0
    /// - Round(5.5) = 6.0
    /// - Round(4.5) = 4.0 (note this "banker's rounding" - it rounds to the nearest even number when exactly halfway)
    /// 
    /// Unlike ToInt32, this keeps the result as a double type, so it still has a decimal point.
    /// </para>
    /// </remarks>
    public double Round(double value) => Math.Round(value);

    /// <summary>
    /// Gets the minimum value that can be represented by a double.
    /// </summary>
    /// <value>The minimum value of a double, which is approximately -1.8 ÃƒÆ’Ã¢â‚¬â€ 10^308.</value>
    /// <remarks>
    /// <para>
    /// This property returns the minimum value that can be represented by a double,
    /// which is -1.7976931348623157E+308. Values smaller than this (more negative) become negative infinity.
    /// </para>
    /// <para><b>For Beginners:</b> This property gives you the smallest possible double value.
    /// 
    /// For doubles, the minimum value is approximately -1.8 ÃƒÆ’Ã¢â‚¬â€ 10^308, which is a very large
    /// negative number (about 1 with 308 zeros after it, with a negative sign).
    /// 
    /// This is useful when you need to work with the full range of double values or
    /// need to check against the minimum possible value. Values smaller than this
    /// become negative infinity.
    /// </para>
    /// </remarks>
    public double MinValue => double.MinValue;

    /// <summary>
    /// Gets the maximum value that can be represented by a double.
    /// </summary>
    /// <value>The maximum value of a double, which is approximately 1.8 ÃƒÆ’Ã¢â‚¬â€ 10^308.</value>
    /// <remarks>
    /// <para>
    /// This property returns the maximum value that can be represented by a double,
    /// which is 1.7976931348623157E+308. Values larger than this become positive infinity.
    /// </para>
    /// <para><b>For Beginners:</b> This property gives you the largest possible double value.
    /// 
    /// For doubles, the maximum value is approximately 1.8 ÃƒÆ’Ã¢â‚¬â€ 10^308, which is a very large
    /// positive number (about 1 with 308 zeros after it).
    /// 
    /// This is useful when you need to work with the full range of double values or
    /// need to check against the maximum possible value. Values larger than this
    /// become positive infinity.
    /// </para>
    /// </remarks>
    public double MaxValue => double.MaxValue;

    /// <summary>
    /// Determines whether the specified double value is NaN (Not a Number).
    /// </summary>
    /// <param name="value">The double value to check.</param>
    /// <returns>true if the value is NaN; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method checks whether a double value is NaN (Not a Number). NaN is a special value that
    /// represents the result of an invalid mathematical operation, such as the square root of a negative number.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if a value is "Not a Number" (NaN).
    /// 
    /// NaN is a special value that represents an invalid result, such as:
    /// - Square root of a negative number
    /// - Division of zero by zero
    /// - Logarithm of a negative number
    /// 
    /// For example:
    /// - IsNaN(5.25) returns false
    /// - IsNaN(double.NaN) returns true
    /// - IsNaN(0.0 / 0.0) returns true
    /// 
    /// NaN has special behavior:
    /// - It does not equal anything, even itself
    /// - Any arithmetic operation involving NaN results in NaN
    /// - Any comparison with NaN returns false
    /// </para>
    /// </remarks>
    public bool IsNaN(double value) => double.IsNaN(value);

    /// <summary>
    /// Determines whether the specified double value is infinity.
    /// </summary>
    /// <param name="value">The double value to check.</param>
    /// <returns>true if the value is positive or negative infinity; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method checks whether a double value is either positive infinity or negative infinity.
    /// Infinity represents a value that exceeds the representable range of a double.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if a value is infinity.
    /// 
    /// Infinity is a special value that represents a result too large to be represented
    /// as a normal double. It can be positive or negative.
    /// 
    /// Operations that can produce infinity include:
    /// - Division by zero (1.0 / 0.0 = positive infinity)
    /// - Very large calculations that exceed the double's range
    /// 
    /// For example:
    /// - IsInfinity(5.25) returns false
    /// - IsInfinity(double.PositiveInfinity) returns true
    /// - IsInfinity(double.NegativeInfinity) returns true
    /// - IsInfinity(1.0 / 0.0) returns true
    /// 
    /// Infinity has special behavior:
    /// - It's larger than any finite number (for positive infinity)
    /// - It's smaller than any finite number (for negative infinity)
    /// - Arithmetic operations involving infinity usually result in infinity
    /// </para>
    /// </remarks>
    public bool IsInfinity(double value) => double.IsInfinity(value);

    /// <summary>
    /// Returns the sign of the specified value as a double.
    /// </summary>
    /// <param name="value">The value to check.</param>
    /// <returns>1 if the value is positive; -1 if the value is negative; 0 if the value is zero.</returns>
    /// <remarks>
    /// <para>
    /// This method returns 1 for any positive value, -1 for any negative value, and 0 for zero.
    /// It is used to determine the sign or direction of a value without considering its magnitude.
    /// </para>
    /// <para><b>For Beginners:</b> This method determines the sign of a number.
    /// 
    /// For example:
    /// - SignOrZero(5.25) returns 1.0
    /// - SignOrZero(-10.5) returns -1.0
    /// - SignOrZero(0.0) returns 0.0
    /// 
    /// This is useful in algorithms that need to know the direction or sign of a value
    /// without caring about its magnitude. Think of it as an indicator showing which
    /// direction the number points on the number line.
    /// 
    /// Special cases:
    /// - SignOrZero(NaN) returns 0.0 (because NaN isn't greater than or less than 0)
    /// - SignOrZero(positive infinity) returns 1.0
    /// - SignOrZero(negative infinity) returns -1.0
    /// </para>
    /// </remarks>
    public double SignOrZero(double value)
    {
        if (value > 0) return 1;
        if (value < 0) return -1;

        return 0;
    }

    /// <summary>
    /// Gets the number of bits used for precision in double (64 bits).
    /// </summary>
    public int PrecisionBits => 64;

    /// <summary>
    /// Converts a double value to float (FP32) precision.
    /// </summary>
    public float ToFloat(double value) => (float)value;

    /// <summary>
    /// Converts a float value to double precision.
    /// </summary>
    public double FromFloat(float value) => (double)value;

    /// <summary>
    /// Converts a double value to Half (FP16) precision.
    /// </summary>
    /// <remarks>
    /// Warning: Double has a much larger range than Half. Values outside [-65504, 65504] will overflow to infinity.
    /// This conversion may also lose significant precision.
    /// </remarks>
    public Half ToHalf(double value) => (Half)value;

    /// <summary>
    /// Converts a Half value to double precision.
    /// </summary>
    public double FromHalf(Half value) => (double)value;

    /// <summary>
    /// Converts a double value to double (identity operation).
    /// </summary>
    public double ToDouble(double value) => value;

    /// <inheritdoc/>
    public bool SupportsCpuAcceleration => true;

    /// <inheritdoc/>
    public bool SupportsGpuAcceleration => true;

    #region IVectorizedOperations<double> Implementation - SIMD via TensorPrimitivesCore

    /// <summary>
    /// Performs element-wise addition using SIMD-optimized TensorPrimitivesCore.
    /// </summary>
    /// <remarks>
    /// Uses AVX-512/AVX2/SSE for hardware acceleration on .NET 5+, scalar fallback on .NET Framework.
    /// </remarks>
    public void Add(ReadOnlySpan<double> x, ReadOnlySpan<double> y, Span<double> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<AddOperatorDouble>(x, y, destination);

    /// <summary>
    /// Performs element-wise subtraction using SIMD-optimized TensorPrimitivesCore.
    /// </summary>
    public void Subtract(ReadOnlySpan<double> x, ReadOnlySpan<double> y, Span<double> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<SubtractOperatorDouble>(x, y, destination);

    /// <summary>
    /// Performs element-wise multiplication using SIMD-optimized TensorPrimitivesCore.
    /// </summary>
    public void Multiply(ReadOnlySpan<double> x, ReadOnlySpan<double> y, Span<double> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<MultiplyOperatorDouble>(x, y, destination);

    /// <summary>
    /// Performs element-wise division using SIMD-optimized TensorPrimitivesCore.
    /// </summary>
    public void Divide(ReadOnlySpan<double> x, ReadOnlySpan<double> y, Span<double> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<DivideOperatorDouble>(x, y, destination);

    /// <summary>
    /// Computes dot product using SIMD-optimized TensorPrimitivesCore.
    /// </summary>
    public double Dot(ReadOnlySpan<double> x, ReadOnlySpan<double> y)
        => TensorPrimitivesCore.Dot(x, y);

    /// <summary>
    /// Computes sum using SIMD-optimized TensorPrimitivesCore.
    /// </summary>
    public double Sum(ReadOnlySpan<double> x)
        => TensorPrimitivesCore.Sum(x);

    /// <summary>
    /// Finds maximum using SIMD-optimized TensorPrimitivesCore.
    /// </summary>
    public double Max(ReadOnlySpan<double> x)
        => TensorPrimitivesCore.Max(x);

    /// <summary>
    /// Finds minimum using SIMD-optimized TensorPrimitivesCore.
    /// </summary>
    public double Min(ReadOnlySpan<double> x)
        => TensorPrimitivesCore.Min(x);

    /// <summary>
    /// Computes exponential using SIMD-optimized TensorPrimitivesCore.
    /// </summary>
    public void Exp(ReadOnlySpan<double> x, Span<double> destination)
        => TensorPrimitivesCore.InvokeSpanIntoSpan<ExpOperatorDouble>(x, destination);

    /// <summary>
    /// Computes natural logarithm using SIMD-optimized TensorPrimitivesCore.
    /// </summary>
    public void Log(ReadOnlySpan<double> x, Span<double> destination)
        => TensorPrimitivesCore.InvokeSpanIntoSpan<LogOperatorDouble>(x, destination);

    /// <summary>
    /// Computes hyperbolic tangent using SIMD-optimized TensorPrimitivesCore.
    /// </summary>
    public void Tanh(ReadOnlySpan<double> x, Span<double> destination)
        => TensorPrimitivesCore.InvokeSpanIntoSpan<TanhOperatorDouble>(x, destination);

    /// <summary>
    /// Computes sigmoid using sequential loops (no SIMD operator yet).
    /// </summary>
    public void Sigmoid(ReadOnlySpan<double> x, Span<double> destination)
        => VectorizedOperationsFallback.Sigmoid(this, x, destination);

    /// <summary>
    /// Computes base-2 logarithm using SIMD-optimized TensorPrimitivesCore.
    /// </summary>
    public void Log2(ReadOnlySpan<double> x, Span<double> destination)
        => TensorPrimitivesCore.InvokeSpanIntoSpan<Log2OperatorDouble>(x, destination);

    /// <summary>
    /// Computes softmax using sequential loops (reduction operation).
    /// </summary>
    public void SoftMax(ReadOnlySpan<double> x, Span<double> destination)
        => VectorizedOperationsFallback.SoftMax(this, x, destination);

    /// <summary>
    /// Computes cosine similarity using sequential loops (complex reduction).
    /// </summary>
    public double CosineSimilarity(ReadOnlySpan<double> x, ReadOnlySpan<double> y)
        => VectorizedOperationsFallback.CosineSimilarity(this, x, y);

    /// <summary>
    /// Fills the destination span with a constant value.
    /// </summary>
    public void Fill(Span<double> destination, double value)
    {
        destination.Fill(value);
    }

    /// <summary>
    /// Multiplies each element by a scalar using SIMD-optimized TensorPrimitivesCore.
    /// </summary>
#if NET8_0_OR_GREATER
    public void MultiplyScalar(ReadOnlySpan<double> x, double scalar, Span<double> destination)
        => System.Numerics.Tensors.TensorPrimitives.Multiply(x, scalar, destination);
#else
    public void MultiplyScalar(ReadOnlySpan<double> x, double scalar, Span<double> destination)
        => VectorizedOperationsFallback.MultiplyScalar(this, x, scalar, destination);
#endif

    /// <summary>
    /// Divides each element by a scalar using SIMD-optimized TensorPrimitivesCore.
    /// </summary>
#if NET8_0_OR_GREATER
    public void DivideScalar(ReadOnlySpan<double> x, double scalar, Span<double> destination)
        => System.Numerics.Tensors.TensorPrimitives.Divide(x, scalar, destination);
#else
    public void DivideScalar(ReadOnlySpan<double> x, double scalar, Span<double> destination)
        => VectorizedOperationsFallback.DivideScalar(this, x, scalar, destination);
#endif

    /// <summary>
    /// Adds a scalar to each element using SIMD-optimized TensorPrimitivesCore.
    /// </summary>
#if NET8_0_OR_GREATER
    public void AddScalar(ReadOnlySpan<double> x, double scalar, Span<double> destination)
        => System.Numerics.Tensors.TensorPrimitives.Add(x, scalar, destination);
#else
    public void AddScalar(ReadOnlySpan<double> x, double scalar, Span<double> destination)
        => VectorizedOperationsFallback.AddScalar(this, x, scalar, destination);
#endif

    /// <summary>
    /// Subtracts a scalar from each element using SIMD-optimized TensorPrimitivesCore.
    /// </summary>
#if NET8_0_OR_GREATER
    public void SubtractScalar(ReadOnlySpan<double> x, double scalar, Span<double> destination)
        => System.Numerics.Tensors.TensorPrimitives.Subtract(x, scalar, destination);
#else
    public void SubtractScalar(ReadOnlySpan<double> x, double scalar, Span<double> destination)
        => VectorizedOperationsFallback.SubtractScalar(this, x, scalar, destination);
#endif

    /// <summary>
    /// Computes square root using SIMD-optimized TensorPrimitivesCore.
    /// </summary>
#if NET8_0_OR_GREATER
    public void Sqrt(ReadOnlySpan<double> x, Span<double> destination)
        => System.Numerics.Tensors.TensorPrimitives.Sqrt<double>(x, destination);
#else
    public void Sqrt(ReadOnlySpan<double> x, Span<double> destination)
        => VectorizedOperationsFallback.Sqrt(this, x, destination);
#endif

    /// <summary>
    /// Computes absolute value using SIMD-optimized TensorPrimitivesCore.
    /// </summary>
#if NET8_0_OR_GREATER
    public void Abs(ReadOnlySpan<double> x, Span<double> destination)
        => System.Numerics.Tensors.TensorPrimitives.Abs<double>(x, destination);
#else
    public void Abs(ReadOnlySpan<double> x, Span<double> destination)
        => VectorizedOperationsFallback.Abs(this, x, destination);
#endif

    /// <summary>
    /// Negates each element using SIMD-optimized TensorPrimitivesCore.
    /// </summary>
#if NET8_0_OR_GREATER
    public void Negate(ReadOnlySpan<double> x, Span<double> destination)
        => System.Numerics.Tensors.TensorPrimitives.Negate<double>(x, destination);
#else
    public void Negate(ReadOnlySpan<double> x, Span<double> destination)
        => VectorizedOperationsFallback.Negate(this, x, destination);
#endif

    /// <summary>
    /// Clips each element to a range.
    /// </summary>
    public void Clip(ReadOnlySpan<double> x, double min, double max, Span<double> destination)
    {
#if NET8_0_OR_GREATER
        TensorPrimitives.Clamp(x, min, max, destination);
#else
        VectorizedOperationsFallback.Clip(this, x, min, max, destination);
#endif
    }

    /// <summary>
    /// Computes the power of each element using SIMD-optimized TensorPrimitivesCore.
    /// </summary>
#if NET8_0_OR_GREATER
    public void Pow(ReadOnlySpan<double> x, double power, Span<double> destination)
        => System.Numerics.Tensors.TensorPrimitives.Pow<double>(x, power, destination);
#else
    public void Pow(ReadOnlySpan<double> x, double power, Span<double> destination)
        => VectorizedOperationsFallback.Pow(this, x, power, destination);
#endif

    /// <summary>
    /// Copies elements from source to destination.
    /// </summary>
    public void Copy(ReadOnlySpan<double> source, Span<double> destination)
    {
        source.CopyTo(destination);
    }

    #endregion
}
