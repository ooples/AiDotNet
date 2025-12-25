using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Operators;

namespace AiDotNet.Tensors.NumericOperations;

/// <summary>
/// Provides operations for signed byte numbers in neural network computations.
/// </summary>
/// <remarks>
/// <para>
/// The SByteOperations class implements the INumericOperations interface for the sbyte data type.
/// It provides essential mathematical operations needed for neural network computations, including
/// basic arithmetic, comparison, and mathematical functions adapted for signed byte values.
/// </para>
/// <para><b>For Beginners:</b> This class handles math operations for very small whole numbers.
/// 
/// Think of it as a calculator specifically designed for neural networks that:
/// - Performs basic operations like addition and multiplication with tiny whole numbers
/// - Handles special math functions adapted to work with small integers
/// - Manages number conversions and comparisons
/// 
/// The "sbyte" (signed byte) data type can only store numbers from -128 to 127, making it
/// useful when you need to save memory and know your values will always stay within this small range.
/// 
/// For example, if a neural network needs to store many small values (like simple flags or counts) 
/// in a very memory-efficient way, it might use the sbyte type instead of larger numeric types.
/// </para>
/// </remarks>
public class SByteOperations : INumericOperations<sbyte>
{
    /// <summary>
    /// Adds two signed byte numbers.
    /// </summary>
    /// <param name="a">The first number.</param>
    /// <param name="b">The second number.</param>
    /// <returns>The sum of the two numbers, cast to a signed byte.</returns>
    /// <remarks>
    /// <para>
    /// This method performs addition of two signed byte values and returns their sum, cast to a signed byte.
    /// Note that if the result exceeds the range of a signed byte (-128 to 127), overflow will occur,
    /// wrapping the result around to stay within the valid range.
    /// </para>
    /// <para><b>For Beginners:</b> This method adds two small numbers together, like 50 + 30 = 80.
    /// 
    /// Important: Because sbyte can only store numbers from -128 to 127, if the result is outside this range,
    /// you'll get unexpected results:
    /// - Add(100, 50) should be 150, but since that's outside the sbyte range, you get -106 instead
    /// - Add(-100, -50) should be -150, but since that's outside the sbyte range, you get 106 instead
    /// 
    /// This "wrapping around" happens because signed bytes can only represent 256 different values
    /// (from -128 to 127), so once you go beyond this range, it cycles back through the available values.
    /// </para>
    /// </remarks>
    public sbyte Add(sbyte a, sbyte b) => (sbyte)(a + b);

    /// <summary>
    /// Subtracts one signed byte from another.
    /// </summary>
    /// <param name="a">The number to subtract from.</param>
    /// <param name="b">The number to subtract.</param>
    /// <returns>The difference between the two numbers, cast to a signed byte.</returns>
    /// <remarks>
    /// <para>
    /// This method performs subtraction of two signed byte values and returns their difference, cast to a signed byte.
    /// Like with addition, if the result exceeds the range of a signed byte, overflow will occur.
    /// </para>
    /// <para><b>For Beginners:</b> This method subtracts the second number from the first, like 50 - 30 = 20.
    /// 
    /// As with addition, if the result goes outside the range of -128 to 127, you'll get unexpected results due to overflow:
    /// - Subtract(100, -50) should be 150, but you get -106 instead
    /// - Subtract(-100, 50) should be -150, but you get 106 instead
    /// 
    /// Be cautious when working near the limits of the sbyte range.
    /// </para>
    /// </remarks>
    public sbyte Subtract(sbyte a, sbyte b) => (sbyte)(a - b);

    /// <summary>
    /// Multiplies two signed byte numbers.
    /// </summary>
    /// <param name="a">The first number.</param>
    /// <param name="b">The second number.</param>
    /// <returns>The product of the two numbers, cast to a signed byte.</returns>
    /// <remarks>
    /// <para>
    /// This method performs multiplication of two signed byte values and returns their product, cast to a signed byte.
    /// If the product exceeds the range of a signed byte, overflow will occur.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies two numbers together, like 10 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 5 = 50.
    /// 
    /// Multiplication is especially prone to overflow with sbytes since numbers grow quickly when multiplied:
    /// - Multiply(20, 10) should be 200, but since that's outside the sbyte range, you get -56 instead
    /// - Multiply(20, -10) should be -200, but you get 56 instead
    /// 
    /// Because of these limitations, sbyte is typically used for very small values or flags in neural networks,
    /// rather than for values that will undergo extensive arithmetic operations.
    /// </para>
    /// </remarks>
    public sbyte Multiply(sbyte a, sbyte b) => (sbyte)(a * b);

    /// <summary>
    /// Divides one signed byte by another.
    /// </summary>
    /// <param name="a">The dividend (number being divided).</param>
    /// <param name="b">The divisor (number to divide by).</param>
    /// <returns>The quotient of the division, cast to a signed byte.</returns>
    /// <remarks>
    /// <para>
    /// This method performs integer division of two signed byte values, returning the result cast to a signed byte.
    /// This is integer division, so any fractional part is truncated. Care should be taken to ensure the divisor
    /// is not zero to avoid runtime exceptions.
    /// </para>
    /// <para><b>For Beginners:</b> This method divides the first number by the second, dropping any remainder.
    /// 
    /// For example:
    /// - Divide(10, 2) returns 5 (exact division, no remainder)
    /// - Divide(10, 3) returns 3 (not 3.33, because sbytes can't store decimals)
    /// - Divide(10, 11) returns 0 (less than 1, so the integer result is 0)
    /// 
    /// Unlike addition and multiplication, division is less likely to cause overflow issues since the result
    /// is always smaller in magnitude than the dividend (when dividing by values greater than 1).
    /// 
    /// Note: This method doesn't check if the second number is zero, which would cause an error
    /// (you can't divide by zero). Make sure the second number is not zero before using this method.
    /// </para>
    /// </remarks>
    public sbyte Divide(sbyte a, sbyte b) => (sbyte)(a / b);

    /// <summary>
    /// Negates a signed byte number.
    /// </summary>
    /// <param name="a">The number to negate.</param>
    /// <returns>The negated value, cast to a signed byte.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the negative of the input value, cast to a signed byte. If the input is positive,
    /// the output is negative, and vice versa. Zero remains zero when negated.
    /// </para>
    /// <para><b>For Beginners:</b> This method flips the sign of a number.
    /// 
    /// Examples:
    /// - Negate(50) returns -50
    /// - Negate(-30) returns 30
    /// - Negate(0) returns 0
    /// 
    /// Special case: Because of how signed bytes are stored, there's one value (-128) that can't be negated
    /// within the sbyte range. Attempting to negate -128 would give 128, which is outside the valid range,
    /// so it wraps around to -128 again.
    /// </para>
    /// </remarks>
    public sbyte Negate(sbyte a) => (sbyte)-a;

    /// <summary>
    /// Gets the zero value for the sbyte type.
    /// </summary>
    /// <value>The value 0.</value>
    /// <remarks>
    /// <para>
    /// This property returns the zero value for the sbyte type, which is 0.
    /// Zero is an important value in neural networks for initialization, comparison, and accumulation.
    /// </para>
    /// <para><b>For Beginners:</b> This property simply gives you the number zero (0) as a signed byte.
    /// 
    /// In neural networks, zero is commonly used for:
    /// - Initializing accumulators before adding values to them
    /// - Checking if a value is exactly zero
    /// - As a default or baseline value in many calculations
    /// </para>
    /// </remarks>
    public sbyte Zero => 0;

    /// <summary>
    /// Gets the one value for the sbyte type.
    /// </summary>
    /// <value>The value 1.</value>
    /// <remarks>
    /// <para>
    /// This property returns the one value for the sbyte type, which is 1.
    /// One is used in neural networks for initialization, identity operations, and counting.
    /// </para>
    /// <para><b>For Beginners:</b> This property simply gives you the number one (1) as a signed byte.
    /// 
    /// In neural networks, one is commonly used for:
    /// - Identity operations (multiplying by 1 leaves a value unchanged)
    /// - Initializing certain weights or biases
    /// - Incrementing counters
    /// </para>
    /// </remarks>
    public sbyte One => 1;

    /// <summary>
    /// Calculates the square root of a signed byte, truncated to a signed byte.
    /// </summary>
    /// <param name="value">The number to calculate the square root of.</param>
    /// <returns>The square root of the input value, cast to a signed byte.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the square root of the input value using the Math.Sqrt function
    /// and converts the result to a signed byte. The input should be non-negative;
    /// otherwise, the result will be undefined.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the square root of a number and gives you a small whole number result.
    /// 
    /// The square root of a number is a value that, when multiplied by itself, gives the original number.
    /// For example:
    /// - The square root of 9 is 3 (because 3 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 3 = 9)
    /// - The square root of 16 is 4 (because 4 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 4 = 16)
    /// - The square root of 125 would be approximately 11.18, but this method returns 11 (the whole number part only)
    /// 
    /// Since the square root of most numbers is not a whole number, and sbyte can only store whole numbers,
    /// this method loses precision. It also has a very limited useful range, since the square root of 127
    /// (the maximum sbyte value) is only about 11.27.
    /// 
    /// Note: You should only use this with positive numbers. If you try to calculate the square root
    /// of a negative number, you'll get an undefined result.
    /// </para>
    /// </remarks>
    public sbyte Sqrt(sbyte value) => (sbyte)Math.Sqrt(value);

    /// <summary>
    /// Converts a double-precision floating-point number to a signed byte.
    /// </summary>
    /// <param name="value">The double-precision value to convert.</param>
    /// <returns>The equivalent signed byte value, truncated toward zero.</returns>
    /// <remarks>
    /// <para>
    /// This method converts a double-precision floating-point value (double) to a signed byte (sbyte).
    /// The conversion truncates the value toward zero, discarding any fractional part, and then clamps
    /// the result to the valid sbyte range.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts a decimal number to a small whole number.
    /// 
    /// For example:
    /// - FromDouble(3.7) returns 3 (not 4, because it drops the decimal part instead of rounding)
    /// - FromDouble(-2.8) returns -2 (not -3, because it drops the decimal part)
    /// - FromDouble(200.0) returns 127 (the maximum sbyte value, since 200 is outside the valid range)
    /// - FromDouble(-200.0) returns -128 (the minimum sbyte value, since -200 is outside the valid range)
    /// 
    /// This conversion is used when:
    /// - You need a whole number result from a calculation that produces decimals
    /// - You're working with functions that use doubles but your neural network uses sbytes
    /// - You need to convert values to the most memory-efficient type
    /// </para>
    /// </remarks>
    public sbyte FromDouble(double value) => (sbyte)value;

    /// <summary>
    /// Checks if one signed byte is greater than another.
    /// </summary>
    /// <param name="a">The first number to compare.</param>
    /// <param name="b">The second number to compare.</param>
    /// <returns>True if the first number is greater than the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two signed byte values and returns true if the first value is greater than the second.
    /// Comparison operations are commonly used in neural networks for conditional logic and optimizations.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is larger than the second.
    /// 
    /// For example:
    /// - GreaterThan(50, 30) returns true because 50 is greater than 30
    /// - GreaterThan(20, 70) returns false because 20 is not greater than 70
    /// - GreaterThan(40, 40) returns false because the numbers are equal
    /// 
    /// In neural networks, comparisons like this are used for:
    /// - Finding maximum values
    /// - Implementing decision logic in algorithms
    /// - Detecting specific conditions during training
    /// </para>
    /// </remarks>
    public bool GreaterThan(sbyte a, sbyte b) => a > b;

    /// <summary>
    /// Checks if one signed byte is less than another.
    /// </summary>
    /// <param name="a">The first number to compare.</param>
    /// <param name="b">The second number to compare.</param>
    /// <returns>True if the first number is less than the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two signed byte values and returns true if the first value is less than the second.
    /// Like the GreaterThan method, this comparison is used in various conditional operations in neural networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is smaller than the second.
    /// 
    /// For example:
    /// - LessThan(30, 50) returns true because 30 is less than 50
    /// - LessThan(70, 20) returns false because 70 is not less than 20
    /// - LessThan(40, 40) returns false because the numbers are equal
    /// 
    /// In neural networks, this comparison is commonly used for:
    /// - Finding minimum values
    /// - Implementing thresholds in algorithms
    /// - Checking if values have fallen below certain limits during training
    /// </para>
    /// </remarks>
    public bool LessThan(sbyte a, sbyte b) => a < b;

    /// <summary>
    /// Calculates the absolute value of a signed byte.
    /// </summary>
    /// <param name="value">The number to find the absolute value of.</param>
    /// <returns>The absolute value of the input.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the absolute value of the input, which is its distance from zero
    /// regardless of sign. For positive numbers, the absolute value is the number itself;
    /// for negative numbers, it is the negation of the number.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives you the positive version of any number.
    /// 
    /// The absolute value is the distance from zero, ignoring the direction (sign):
    /// - Abs(50) returns 50 (already positive)
    /// - Abs(-30) returns 30 (converts negative to positive)
    /// - Abs(0) returns 0
    /// 
    /// Special case: Because the sbyte range is from -128 to 127, the absolute value of -128 cannot be
    /// represented as an sbyte (it would be 128, which exceeds the maximum value of 127). In this case,
    /// the result wraps around to -128 again.
    /// 
    /// In neural networks, absolute values are used for:
    /// - Measuring error magnitudes (how far predictions are from actual values)
    /// - Implementing certain activation functions
    /// - Checking if values are within certain tolerances, regardless of sign
    /// </para>
    /// </remarks>
    public sbyte Abs(sbyte value) => Math.Abs(value);

    /// <summary>
    /// Squares a signed byte number.
    /// </summary>
    /// <param name="value">The number to square.</param>
    /// <returns>The square of the input value, cast to a signed byte.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the square of the input value by multiplying it by itself,
    /// and then casts the result to a signed byte. If the square exceeds the range of a signed byte,
    /// overflow will occur.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies a number by itself.
    /// 
    /// For example:
    /// - Square(4) returns 16 (4 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 4 = 16)
    /// - Square(-3) returns 9 (-3 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  -3 = 9)
    /// - Square(12) should return 144, but since that's outside the range of sbyte, you get -112 instead
    /// 
    /// Due to the limited range of sbyte (-128 to 127), squaring even moderate values (like 12) can cause overflow.
    /// In fact, any number with an absolute value greater than 11 will cause overflow when squared.
    /// 
    /// Despite these limitations, squaring is useful for very small values, such as when implementing
    /// small-scale error calculations or when working with normalized values near zero.
    /// </para>
    /// </remarks>
    public sbyte Square(sbyte value) => Multiply(value, value);

    /// <summary>
    /// Calculates the exponential function (e raised to the power of the specified value), rounded and constrained to a signed byte.
    /// </summary>
    /// <param name="value">The exponent.</param>
    /// <returns>The value of e raised to the specified power, rounded and constrained to the signed byte range.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates e (approximately 2.71828) raised to the power of the input value
    /// using the Math.Exp function, rounds the result, and clamps it to the valid sbyte range
    /// if it exceeds the maximum value.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates "e" raised to a power and gives a small whole number result.
    /// 
    /// In mathematics, "e" is a special number (approximately 2.71828) that appears naturally in many calculations.
    /// This method computes e^value and rounds to the nearest whole number, capping at 127 (the maximum sbyte value):
    /// - Exp(1) returns 3 (e^ ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  2.71828, rounded to 3)
    /// - Exp(2) returns 7 (e^ ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  7.38906, rounded to 7)
    /// - Exp(0) returns 1 (e^ = 1)
    /// - Exp(5) returns 127 (e5 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 148.4, which exceeds 127, so it's capped at 127)
    /// 
    /// The exponential function grows very quickly, so it's only useful with sbyte for small input values.
    /// Any input value of 5 or greater will produce a result that exceeds the maximum sbyte value of 127,
    /// so the method caps the result to prevent overflow.
    /// </para>
    /// </remarks>
    public sbyte Exp(sbyte value) => (sbyte)Math.Min(sbyte.MaxValue, Math.Round(Math.Exp(value)));

    /// <summary>
    /// Checks if two signed bytes are equal.
    /// </summary>
    /// <param name="a">The first number to compare.</param>
    /// <param name="b">The second number to compare.</param>
    /// <returns>True if the numbers are equal; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two signed byte values for equality.
    /// Unlike floating-point equality, integer equality is exact and reliable.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if two numbers are exactly the same.
    /// 
    /// For example:
    /// - Equals(50, 50) returns true
    /// - Equals(30, 40) returns false
    /// 
    /// Unlike with decimal numbers (float/double), comparing integers for equality is straightforward
    /// and reliable because integers have exact representations in the computer.
    /// </para>
    /// </remarks>
    public bool Equals(sbyte a, sbyte b) => a == b;

    public int Compare(sbyte a, sbyte b) => a.CompareTo(b);

    /// <summary>
    /// Raises a signed byte to the specified power.
    /// </summary>
    /// <param name="baseValue">The base number.</param>
    /// <param name="exponent">The exponent.</param>
    /// <returns>The base raised to the power of the exponent, cast to a signed byte.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates baseValue raised to the power of exponent using the Math.Pow function
    /// and converts the result to a signed byte. If the result exceeds the range of a signed byte,
    /// overflow will occur.
    /// </para>
    /// <para><b>For Beginners:</b> This method raises a number to a power and gives a small whole number result.
    /// 
    /// For example:
    /// - Power(2, 3) returns 8 (2ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â² = 2 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 2ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â2 = 8)
    /// - Power(3, 2) returns 9 (3ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â² = 3 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 3 = 9)
    /// - Power(2, 7) should return 128, but since that's outside the range of sbyte, you'd get -128 instead
    /// 
    /// Due to the limited range of sbyte, even moderate powers can cause overflow:
    /// - Any base greater than 3 with an exponent greater than 3 will exceed the maximum value of 127
    /// - Any negative base with an odd exponent will produce a negative result
    /// 
    /// This method is primarily useful for very small numbers and low exponents.
    /// </para>
    /// </remarks>
    public sbyte Power(sbyte baseValue, sbyte exponent) => (sbyte)Math.Pow(baseValue, exponent);

    /// <summary>
    /// Calculates the natural logarithm (base e) of a signed byte, cast to a signed byte.
    /// </summary>
    /// <param name="value">The number to calculate the logarithm of.</param>
    /// <returns>The natural logarithm of the input value, cast to a signed byte.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the natural logarithm (base e) of the input value using the Math.Log function
    /// and converts the result to a signed byte. The input should be positive; otherwise, the result will be undefined.
    /// Since logarithm results are often not whole numbers, this conversion to signed byte loses precision.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the natural logarithm of a number and gives a small whole number result.
    /// 
    /// The natural logarithm tells you what power you need to raise "e" to get your number:
    /// - Log(3) returns 1 (because e^ ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  2.718, and the integer result of ln(3) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  1.099 is 1)
    /// - Log(10) returns 2 (because ln(10) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  2.303)
    /// - Log(125) returns 4 (because ln(125) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  4.828)
    /// - Log(1) returns 0 (because e^ = 1)
    /// 
    /// This integer version of logarithm loses a lot of precision compared to its floating-point
    /// equivalent. However, since the logarithm of small positive numbers is typically a small number,
    /// it works reasonably well within the limited sbyte range.
    /// 
    /// Note: You should only use this with positive numbers. If you try to calculate the logarithm
    /// of zero or a negative number, you'll get an undefined result.
    /// </para>
    /// </remarks>
    public sbyte Log(sbyte value) => (sbyte)Math.Log(value);

    /// <summary>
    /// Checks if one signed byte is greater than or equal to another.
    /// </summary>
    /// <param name="a">The first number to compare.</param>
    /// <param name="b">The second number to compare.</param>
    /// <returns>True if the first number is greater than or equal to the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two signed byte values and returns true if the first value is greater than or equal to the second.
    /// This comparison combines the functionality of GreaterThan and Equals methods.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is larger than or the same as the second.
    /// 
    /// For example:
    /// - GreaterThanOrEquals(50, 30) returns true because 50 is greater than 30
    /// - GreaterThanOrEquals(40, 40) returns true because the numbers are equal
    /// - GreaterThanOrEquals(20, 70) returns false because 20 is less than 70
    /// 
    /// In neural networks, this type of comparison is used for:
    /// - Implementing thresholds with inclusive boundaries
    /// - Checking if values have reached or exceeded certain levels
    /// - Decision logic in various algorithms
    /// </para>
    /// </remarks>
    public bool GreaterThanOrEquals(sbyte a, sbyte b) => a >= b;

    /// <summary>
    /// Checks if one signed byte is less than or equal to another.
    /// </summary>
    /// <param name="a">The first number to compare.</param>
    /// <param name="b">The second number to compare.</param>
    /// <returns>True if the first number is less than or equal to the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two signed byte values and returns true if the first value is less than or equal to the second.
    /// This comparison combines the functionality of LessThan and Equals methods.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is smaller than or the same as the second.
    /// 
    /// For example:
    /// - LessThanOrEquals(30, 50) returns true because 30 is less than 50
    /// - LessThanOrEquals(40, 40) returns true because the numbers are equal
    /// - LessThanOrEquals(70, 20) returns false because 70 is greater than 20
    /// 
    /// In neural networks, this type of comparison is used for:
    /// - Implementing thresholds with inclusive lower boundaries
    /// - Checking if values have reached or fallen below certain levels
    /// - Decision logic in various algorithms
    /// </para>
    /// </remarks>
    public bool LessThanOrEquals(sbyte a, sbyte b) => a <= b;

    /// <summary>
    /// Converts a signed byte to a 32-bit integer.
    /// </summary>
    /// <param name="value">The signed byte value to convert.</param>
    /// <returns>The equivalent 32-bit integer value.</returns>
    /// <remarks>
    /// <para>
    /// This method converts a signed byte (8-bit, range -128 to 127) to a standard 32-bit integer
    /// (range -2,147,483,648 to 2,147,483,647). Since the range of signed byte is much smaller than
    /// the range of int, this conversion never causes data loss.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts a very small whole number to a standard whole number.
    /// 
    /// For example:
    /// - ToInt32(50) returns 50 as an int instead of an sbyte
    /// - ToInt32(-30) returns -30 as an int instead of an sbyte
    /// 
    /// This conversion is always safe because all possible sbyte values (-128 to 127) fit easily
    /// within the much larger int range (-2,147,483,648 to 2,147,483,647).
    /// 
    /// In neural networks, this conversion might be needed when:
    /// - Interfacing with methods that require standard integers
    /// - Performing calculations that might exceed the sbyte range
    /// - Combining sbyte values with values of other types
    /// </para>
    /// </remarks>
    public int ToInt32(sbyte value) => value;

    /// <summary>
    /// Returns the same signed byte value (identity operation).
    /// </summary>
    /// <param name="value">The signed byte value.</param>
    /// <returns>The same signed byte value.</returns>
    /// <remarks>
    /// <para>
    /// This method simply returns the input value unchanged. It serves as an identity operation for signed bytes.
    /// For signed bytes, rounding is unnecessary since they are already whole numbers.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns the exact same number you give it.
    /// 
    /// For float or double types, the equivalent method would round the number to the nearest whole number,
    /// but since signed bytes are already whole numbers, no rounding is needed:
    /// - Round(50) returns 50
    /// - Round(-30) returns -30
    /// 
    /// This method exists to maintain consistency with the interface used for different numeric types.
    /// </para>
    /// </remarks>
    public sbyte Round(sbyte value) => value;

    public sbyte Floor(sbyte value) => value;
    public sbyte Ceiling(sbyte value) => value;
    public sbyte Frac(sbyte value) => 0;

    /// <summary>
    /// Returns the sine of the specified value (truncated to integer).
    /// </summary>
    public sbyte Sin(sbyte value) => (sbyte)Math.Sin(value);

    /// <summary>
    /// Returns the cosine of the specified value (truncated to integer).
    /// </summary>
    public sbyte Cos(sbyte value) => (sbyte)Math.Cos(value);


    /// <summary>
    /// Gets the minimum possible value for a signed byte.
    /// </summary>
    /// <value>The minimum value of sbyte, which is -128.</value>
    /// <remarks>
    /// <para>
    /// This property returns the smallest possible value for an 8-bit signed byte.
    /// This value represents the lower bound of the range of representable values for the sbyte type.
    /// </para>
    /// <para><b>For Beginners:</b> This property gives you the smallest possible value that a signed byte can store: -128.
    /// 
    /// Knowing the minimum value is important for:
    /// - Preventing underflow (when calculations produce results too small to represent)
    /// - Setting bounds for certain algorithms
    /// - Implementing special case handling for extreme values
    /// 
    /// Be careful when working with this minimum value: negating MinValue (-128) will cause an overflow
    /// because the positive equivalent (+128) is outside the representable range of a signed byte
    /// (which has a maximum value of +127).
    /// </para>
    /// </remarks>
    public sbyte MinValue => sbyte.MinValue;

    /// <summary>
    /// Gets the maximum possible value for a signed byte.
    /// </summary>
    /// <value>The maximum value of sbyte, which is 127.</value>
    /// <remarks>
    /// <para>
    /// This property returns the largest possible value for an 8-bit signed byte.
    /// This value represents the upper bound of the range of representable values for the sbyte type.
    /// </para>
    /// <para><b>For Beginners:</b> This property gives you the largest possible value that a signed byte can store: 127.
    /// 
    /// Knowing the maximum value is important for:
    /// - Preventing overflow (when calculations produce results too large to represent)
    /// - Setting bounds for certain algorithms
    /// - Implementing special case handling for extreme values
    /// 
    /// The sbyte type can only store 256 different values (from -128 to 127), making it very
    /// limited compared to larger integer types. However, it uses only a single byte of memory,
    /// which can be important when memory efficiency is critical.
    /// </para>
    /// </remarks>
    public sbyte MaxValue => sbyte.MaxValue;

    /// <summary>
    /// Determines whether the specified signed byte is not a number (NaN).
    /// </summary>
    /// <param name="value">The signed byte to test.</param>
    /// <returns>Always returns false because signed bytes cannot be NaN.</returns>
    /// <remarks>
    /// <para>
    /// This method always returns false because the concept of NaN (Not a Number) does not apply to integers.
    /// NaN is a special value that exists only for floating-point types like float and double.
    /// </para>
    /// <para><b>For Beginners:</b> This method always returns false because all signed bytes are valid numbers.
    /// 
    /// Unlike floating-point numbers (float/double) which can have special "Not a Number" values,
    /// every possible signed byte value represents a valid number. This method exists only to maintain
    /// consistency with the interface used for different numeric types.
    /// 
    /// In neural networks that can work with different numeric types, this consistent interface
    /// allows the same code to be used regardless of whether the network is using integers or
    /// floating-point numbers.
    /// </para>
    /// </remarks>
    public bool IsNaN(sbyte value) => false;

    /// <summary>
    /// Determines whether the specified signed byte is infinity.
    /// </summary>
    /// <param name="value">The signed byte to test.</param>
    /// <returns>Always returns false because signed bytes cannot be infinity.</returns>
    /// <remarks>
    /// <para>
    /// This method always returns false because the concept of infinity does not apply to integers.
    /// Infinity is a special value that exists only for floating-point types like float and double.
    /// </para>
    /// <para><b>For Beginners:</b> This method always returns false because signed bytes cannot represent infinity.
    /// 
    /// Unlike floating-point numbers (float/double) which can have special "Infinity" values,
    /// signed bytes have a fixed range and cannot represent concepts like infinity. This method exists
    /// only to maintain consistency with the interface used for different numeric types.
    /// 
    /// In neural networks that can work with different numeric types, this consistent interface
    /// allows the same code to be used regardless of whether the network is using integers or
    /// floating-point numbers.
    /// </para>
    /// </remarks>
    public bool IsInfinity(sbyte value) => false;

    /// <summary>
    /// Returns the sign of a signed byte, or zero if the number is zero.
    /// </summary>
    /// <param name="value">The signed byte to get the sign of.</param>
    /// <returns>1 if the number is positive, -1 if the number is negative, or 0 if the number is zero.</returns>
    /// <remarks>
    /// <para>
    /// This method determines the sign of the input value and returns 1 for positive numbers,
    /// -1 for negative numbers, and 0 for zero. This is similar to the Math.Sign function,
    /// but implemented specifically for the sbyte type.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells you if a number is positive, negative, or zero.
    /// 
    /// It returns:
    /// - 1 if the number is positive (greater than zero)
    /// - -1 if the number is negative (less than zero)
    /// - 0 if the number is exactly zero
    /// 
    /// For example:
    /// - SignOrZero(42) returns 1
    /// - SignOrZero(-3) returns -1
    /// - SignOrZero(0) returns 0
    /// 
    /// In neural networks, this function might be used for:
    /// - Implementing custom activation functions (like the sign function)
    /// - Thresholding operations that depend only on the sign of a value
    /// - Converting continuous values to discrete categories (-1, 0, +1)
    /// 
    /// Unlike some sign functions that return either -1 or 1, this method treats zero as its own category,
    /// which can be useful in certain neural network applications.
    /// </para>
    /// </remarks>
    public sbyte SignOrZero(sbyte value) => value == 0 ? (sbyte)0 : value > 0 ? (sbyte)1 : (sbyte)-1;

    /// <summary>
    /// Gets the number of bits used for precision in sbyte (8 bits).
    /// </summary>
    public int PrecisionBits => 8;

    /// <summary>
    /// Converts an sbyte value to float (FP32) precision.
    /// </summary>
    public float ToFloat(sbyte value) => (float)value;

    /// <summary>
    /// Converts a float value to sbyte.
    /// </summary>
    public sbyte FromFloat(float value) => (sbyte)MathExtensions.Clamp((int)Math.Round(value), sbyte.MinValue, sbyte.MaxValue);

    /// <summary>
    /// Converts an sbyte value to Half (FP16) precision.
    /// </summary>
    public Half ToHalf(sbyte value) => (Half)value;

    /// <summary>
    /// Converts a Half value to sbyte.
    /// </summary>
    public sbyte FromHalf(Half value) => (sbyte)MathExtensions.Clamp((int)Math.Round((float)value), sbyte.MinValue, sbyte.MaxValue);

    /// <summary>
    /// Converts an sbyte value to double (FP64) precision.
    /// </summary>
    public double ToDouble(sbyte value) => (double)value;

    /// <inheritdoc/>
    public bool SupportsCpuAcceleration => false;

    /// <inheritdoc/>
    public bool SupportsGpuAcceleration => false;

    #region IVectorizedOperations<sbyte> Implementation

    /// <summary>
    /// Performs element-wise addition using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public void Add(ReadOnlySpan<sbyte> x, ReadOnlySpan<sbyte> y, Span<sbyte> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<AddOperatorSByte>(x, y, destination);

    /// <summary>
    /// Performs element-wise subtraction using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public void Subtract(ReadOnlySpan<sbyte> x, ReadOnlySpan<sbyte> y, Span<sbyte> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<SubtractOperatorSByte>(x, y, destination);

    /// <summary>
    /// Performs element-wise multiplication using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public void Multiply(ReadOnlySpan<sbyte> x, ReadOnlySpan<sbyte> y, Span<sbyte> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<MultiplyOperatorSByte>(x, y, destination);

    /// <summary>
    /// Performs element-wise division using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public void Divide(ReadOnlySpan<sbyte> x, ReadOnlySpan<sbyte> y, Span<sbyte> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<DivideOperatorSByte>(x, y, destination);

    /// <summary>
    /// Computes dot product using sequential loops.
    /// </summary>
    public sbyte Dot(ReadOnlySpan<sbyte> x, ReadOnlySpan<sbyte> y)
        => VectorizedOperationsFallback.Dot(this, x, y);

    /// <summary>
    /// Computes sum using sequential loops.
    /// </summary>
    public sbyte Sum(ReadOnlySpan<sbyte> x)
        => VectorizedOperationsFallback.Sum(this, x);

    /// <summary>
    /// Finds maximum using sequential loops.
    /// </summary>
    public sbyte Max(ReadOnlySpan<sbyte> x)
        => VectorizedOperationsFallback.Max(this, x);

    /// <summary>
    /// Finds minimum using sequential loops.
    /// </summary>
    public sbyte Min(ReadOnlySpan<sbyte> x)
        => VectorizedOperationsFallback.Min(this, x);

    /// <summary>
    /// Transcendental operations are not supported for sbyte type.
    /// </summary>
    /// <exception cref="NotSupportedException">Always thrown. Exp produces misleading results for sbyte (range -128 to 127).</exception>
    public void Exp(ReadOnlySpan<sbyte> x, Span<sbyte> destination)
        => throw new NotSupportedException("Transcendental operations (Exp) are not meaningful for sbyte type. Use float or double instead.");

    /// <summary>
    /// Transcendental operations are not supported for sbyte type.
    /// </summary>
    /// <exception cref="NotSupportedException">Always thrown. Log produces misleading results for sbyte.</exception>
    public void Log(ReadOnlySpan<sbyte> x, Span<sbyte> destination)
        => throw new NotSupportedException("Transcendental operations (Log) are not meaningful for sbyte type. Use float or double instead.");

    /// <summary>
    /// Transcendental operations are not supported for sbyte type.
    /// </summary>
    /// <exception cref="NotSupportedException">Always thrown. Tanh produces only -1, 0, or 1 for sbyte.</exception>
    public void Tanh(ReadOnlySpan<sbyte> x, Span<sbyte> destination)
        => throw new NotSupportedException("Transcendental operations (Tanh) are not meaningful for sbyte type. Use float or double instead.");

    /// <summary>
    /// Transcendental operations are not supported for sbyte type.
    /// </summary>
    /// <exception cref="NotSupportedException">Always thrown. Sigmoid saturates for sbyte inputs.</exception>
    public void Sigmoid(ReadOnlySpan<sbyte> x, Span<sbyte> destination)
        => throw new NotSupportedException("Transcendental operations (Sigmoid) are not meaningful for sbyte type. Use float or double instead.");

    /// <summary>
    /// Transcendental operations are not supported for sbyte type.
    /// </summary>
    /// <exception cref="NotSupportedException">Always thrown. Log2 produces misleading results for sbyte.</exception>
    public void Log2(ReadOnlySpan<sbyte> x, Span<sbyte> destination)
        => throw new NotSupportedException("Transcendental operations (Log2) are not meaningful for sbyte type. Use float or double instead.");

    /// <summary>
    /// Transcendental operations are not supported for sbyte type.
    /// </summary>
    /// <exception cref="NotSupportedException">Always thrown. SoftMax requires floating-point for normalized probabilities.</exception>
    public void SoftMax(ReadOnlySpan<sbyte> x, Span<sbyte> destination)
        => throw new NotSupportedException("Transcendental operations (SoftMax) are not meaningful for sbyte type. Use float or double instead.");

    /// <summary>
    /// Computes cosine similarity using sequential loops.
    /// </summary>
    public sbyte CosineSimilarity(ReadOnlySpan<sbyte> x, ReadOnlySpan<sbyte> y)
        => VectorizedOperationsFallback.CosineSimilarity(this, x, y);

    private static readonly SByteOperations _instance = new();

    public void Fill(Span<sbyte> destination, sbyte value) => destination.Fill(value);
    public void MultiplyScalar(ReadOnlySpan<sbyte> x, sbyte scalar, Span<sbyte> destination) => VectorizedOperationsFallback.MultiplyScalar(_instance, x, scalar, destination);
    public void DivideScalar(ReadOnlySpan<sbyte> x, sbyte scalar, Span<sbyte> destination) => VectorizedOperationsFallback.DivideScalar(_instance, x, scalar, destination);
    public void AddScalar(ReadOnlySpan<sbyte> x, sbyte scalar, Span<sbyte> destination) => VectorizedOperationsFallback.AddScalar(_instance, x, scalar, destination);
    public void SubtractScalar(ReadOnlySpan<sbyte> x, sbyte scalar, Span<sbyte> destination) => VectorizedOperationsFallback.SubtractScalar(_instance, x, scalar, destination);
    public void Sqrt(ReadOnlySpan<sbyte> x, Span<sbyte> destination) => VectorizedOperationsFallback.Sqrt(_instance, x, destination);
    public void Abs(ReadOnlySpan<sbyte> x, Span<sbyte> destination) => VectorizedOperationsFallback.Abs(_instance, x, destination);
    public void Negate(ReadOnlySpan<sbyte> x, Span<sbyte> destination) => VectorizedOperationsFallback.Negate(_instance, x, destination);
    public void Clip(ReadOnlySpan<sbyte> x, sbyte min, sbyte max, Span<sbyte> destination) => VectorizedOperationsFallback.Clip(_instance, x, min, max, destination);
    public void Pow(ReadOnlySpan<sbyte> x, sbyte power, Span<sbyte> destination) => VectorizedOperationsFallback.Pow(_instance, x, power, destination);
    public void Copy(ReadOnlySpan<sbyte> source, Span<sbyte> destination) => source.CopyTo(destination);

    #endregion

    public void Floor(ReadOnlySpan<sbyte> x, Span<sbyte> destination) => x.CopyTo(destination);
    public void Ceiling(ReadOnlySpan<sbyte> x, Span<sbyte> destination) => x.CopyTo(destination);
    public void Frac(ReadOnlySpan<sbyte> x, Span<sbyte> destination) => destination.Fill(0);
    public void Sin(ReadOnlySpan<sbyte> x, Span<sbyte> destination)
    {
        for (int i = 0; i < x.Length; i++)
            destination[i] = (sbyte)Math.Sin(x[i]);
    }
    public void Cos(ReadOnlySpan<sbyte> x, Span<sbyte> destination)
    {
        for (int i = 0; i < x.Length; i++)
            destination[i] = (sbyte)Math.Cos(x[i]);
    }

}
