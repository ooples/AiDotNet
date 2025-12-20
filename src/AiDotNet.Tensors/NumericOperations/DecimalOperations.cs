using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.NumericOperations;
/// <summary>
/// Provides mathematical operations for the decimal data type.
/// </summary>
/// <remarks>
/// <para>
/// This class implements the INumericOperations interface for the decimal data type, providing
/// basic arithmetic operations, comparison methods, and mathematical functions. The decimal
/// type offers higher precision than floating-point types (float and double) and is particularly
/// suitable for financial and monetary calculations where precision is critical.
/// </para>
/// <para><b>For Beginners:</b> This class handles math operations for the decimal number type.
///
/// The decimal type in C# is designed for high-precision calculations, especially with money:
/// - It can store numbers with up to 28-29 significant digits
/// - It avoids many of the rounding errors common in float and double types
/// - It has a smaller range than float or double but much higher precision
///
/// Think of decimal as the type you'd want to use when every penny counts, like in
/// financial applications, banking, or any situation where exact calculations are required.
/// </para>
/// </remarks>
public class DecimalOperations : INumericOperations<decimal>
{
    /// <summary>
    /// Adds two decimal values together.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    /// <returns>The sum of the two values.</returns>
    /// <remarks>
    /// <para>
    /// This method performs standard decimal addition, which is exact within the range of the decimal type.
    /// Unlike floating-point addition, decimal addition does not introduce rounding errors for most values.
    /// </para>
    /// <para><b>For Beginners:</b> This method adds two decimal numbers together.
    ///
    /// For example:
    /// - 5.25m + 3.75m = 9.00m
    /// - 0.1m + 0.2m = 0.3m (exactly, unlike with float or double)
    ///
    /// The 'm' suffix is used to indicate decimal literals in C#.
    /// Unlike floating-point numbers, decimals can represent most decimal fractions exactly.
    /// </para>
    /// </remarks>
    public decimal Add(decimal a, decimal b) => a + b;

    /// <summary>
    /// Subtracts the second decimal value from the first.
    /// </summary>
    /// <param name="a">The value to subtract from.</param>
    /// <param name="b">The value to subtract.</param>
    /// <returns>The difference between the two values.</returns>
    /// <remarks>
    /// <para>
    /// This method performs standard decimal subtraction, which is exact within the range of the decimal type.
    /// Unlike floating-point subtraction, decimal subtraction does not introduce rounding errors for most values.
    /// </para>
    /// <para><b>For Beginners:</b> This method subtracts one decimal number from another.
    /// 
    /// For example:
    /// - 10.00m - 3.25m = 6.75m
    /// - 1.0m - 0.1m = 0.9m (exactly, unlike with float or double)
    /// 
    /// Decimals are particularly useful for financial calculations where exact subtraction is important.
    /// </para>
    /// </remarks>
    public decimal Subtract(decimal a, decimal b) => a - b;

    /// <summary>
    /// Multiplies two decimal values together.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    /// <returns>The product of the two values.</returns>
    /// <remarks>
    /// <para>
    /// This method performs standard decimal multiplication, which is exact within the range of the decimal type.
    /// If the result exceeds the range of the decimal type, an OverflowException will be thrown.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies two decimal numbers together.
    /// 
    /// For example:
    /// - 5.5m * 2.0m = 11.0m
    /// - 0.1m * 0.1m = 0.01m (exactly, unlike with float or double)
    /// 
    /// Decimals maintain high precision during multiplication, making them ideal
    /// for calculating prices, interest rates, and other financial values.
    /// </para>
    /// </remarks>
    public decimal Multiply(decimal a, decimal b) => a * b;

    /// <summary>
    /// Divides the first decimal value by the second.
    /// </summary>
    /// <param name="a">The dividend (value being divided).</param>
    /// <param name="b">The divisor (value to divide by).</param>
    /// <returns>The quotient of the division.</returns>
    /// <remarks>
    /// <para>
    /// This method performs decimal division. If the divisor is zero, a DivideByZeroException will be thrown.
    /// Decimal division may result in a value with more decimal places than either of the operands.
    /// </para>
    /// <para><b>For Beginners:</b> This method divides one decimal number by another.
    /// 
    /// For example:
    /// - 10.0m / 2.0m = 5.0m
    /// - 1.0m / 3.0m = 0.3333333333333333333333333333m
    /// 
    /// The result will have high precision but may eventually round if the division
    /// produces a repeating decimal that exceeds the precision of the decimal type.
    /// Division by zero will cause an error.
    /// </para>
    /// </remarks>
    public decimal Divide(decimal a, decimal b) => a / b;

    /// <summary>
    /// Negates the specified decimal value.
    /// </summary>
    /// <param name="a">The value to negate.</param>
    /// <returns>The negated value.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the arithmetic negation of the input value, effectively changing its sign.
    /// If the input is positive, the result is negative, and vice versa. Zero remains zero when negated.
    /// </para>
    /// <para><b>For Beginners:</b> This method reverses the sign of a decimal number.
    /// 
    /// For example:
    /// - Negate(5.25m) = -5.25m
    /// - Negate(-10.5m) = 10.5m
    /// - Negate(0.0m) = 0.0m
    /// 
    /// This is the same as multiplying the number by -1.
    /// </para>
    /// </remarks>
    public decimal Negate(decimal a) => -a;

    /// <summary>
    /// Gets the decimal representation of zero.
    /// </summary>
    /// <value>The value 0 as a decimal.</value>
    /// <remarks>
    /// <para>
    /// This property returns the decimal representation of the value zero, which is 0m.
    /// It is often used as a neutral element for addition.
    /// </para>
    /// <para><b>For Beginners:</b> This property provides the value zero as a decimal.
    /// 
    /// Zero is a special value in mathematics:
    /// - Adding zero to any number gives the same number
    /// - It's used as a starting point in many algorithms
    /// 
    /// This property gives you a zero that matches the decimal type, written as 0m in C#.
    /// </para>
    /// </remarks>
    public decimal Zero => 0m;

    /// <summary>
    /// Gets the decimal representation of one.
    /// </summary>
    /// <value>The value 1 as a decimal.</value>
    /// <remarks>
    /// <para>
    /// This property returns the decimal representation of the value one, which is 1m.
    /// It is often used as a neutral element for multiplication.
    /// </para>
    /// <para><b>For Beginners:</b> This property provides the value one as a decimal.
    /// 
    /// One is a special value in mathematics:
    /// - Multiplying any number by one gives the same number
    /// - It's useful as a starting point or increment value
    /// 
    /// This property gives you a one that matches the decimal type, written as 1m in C#.
    /// </para>
    /// </remarks>
    public decimal One => 1m;

    /// <summary>
    /// Calculates the square root of a decimal value.
    /// </summary>
    /// <param name="value">The value to calculate the square root of.</param>
    /// <returns>The square root of the specified value.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the square root by converting the decimal to a double,
    /// performing the square root operation, and then converting the result back to a decimal.
    /// This approach is used because there is no direct square root operation for decimals in .NET.
    /// Some precision may be lost in this conversion process.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the square root of a decimal number.
    /// 
    /// For example:
    /// - Square root of 9.0m is 3.0m
    /// - Square root of 2.0m is approximately 1.4142135623730950488016887242m
    /// 
    /// Note that this method first converts to double and then back to decimal,
    /// which may cause a slight loss of precision in some cases. This is because
    /// the .NET Framework doesn't provide a native square root operation for decimals.
    /// </para>
    /// </remarks>
    public decimal Sqrt(decimal value) => (decimal)Math.Sqrt((double)value);

    /// <summary>
    /// Converts a double value to a decimal.
    /// </summary>
    /// <param name="value">The double value to convert.</param>
    /// <returns>The double value converted to a decimal.</returns>
    /// <remarks>
    /// <para>
    /// This method converts a double value to a decimal. If the double value is outside the range
    /// that can be represented by a decimal, an OverflowException will be thrown.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts a double number to a decimal.
    /// 
    /// For example:
    /// - Converting 123.45 (double) to decimal gives 123.45m
    /// 
    /// Important notes:
    /// - Not all double values can be converted to decimal
    /// - Decimal has a smaller range but higher precision than double
    /// - If the double is too large or too small, this will cause an error
    /// 
    /// This conversion is useful when you need to use a double value in
    /// calculations that require the precision of decimal.
    /// </para>
    /// </remarks>
    public decimal FromDouble(double value) => (decimal)value;

    /// <summary>
    /// Determines whether the first decimal value is greater than the second.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns>true if the first value is greater than the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two decimal values and returns true if the first is greater than the second.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is larger than the second.
    /// 
    /// For example:
    /// - 10.5m > 5.2m returns true
    /// - 5.0m > 10.0m returns false
    /// - 5.0m > 5.0m returns false
    /// 
    /// This is a simple comparison operation used in many algorithms.
    /// </para>
    /// </remarks>
    public bool GreaterThan(decimal a, decimal b) => a > b;

    /// <summary>
    /// Determines whether the first decimal value is less than the second.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns>true if the first value is less than the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two decimal values and returns true if the first is less than the second.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is smaller than the second.
    /// 
    /// For example:
    /// - 5.2m < 10.5m returns true
    /// - 10.0m < 5.0m returns false
    /// - 5.0m < 5.0m returns false
    /// 
    /// This is a simple comparison operation used in many algorithms.
    /// </para>
    /// </remarks>
    public bool LessThan(decimal a, decimal b) => a < b;

    /// <summary>
    /// Returns the absolute value of a decimal.
    /// </summary>
    /// <param name="value">The decimal value.</param>
    /// <returns>The absolute value of the specified decimal.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the absolute (positive) value of the specified decimal value.
    /// If the value is already positive or zero, it is returned unchanged. If the value is negative,
    /// its negation is returned.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides the positive version of a number.
    /// 
    /// For example:
    /// - Abs(5.25m) = 5.25m (already positive, so unchanged)
    /// - Abs(-5.25m) = 5.25m (negative becomes positive)
    /// - Abs(0.0m) = 0.0m (zero remains zero)
    /// 
    /// The absolute value is the distance of a number from zero, ignoring the direction.
    /// </para>
    /// </remarks>
    public decimal Abs(decimal value) => Math.Abs(value);

    /// <summary>
    /// Squares the specified decimal value.
    /// </summary>
    /// <param name="value">The value to square.</param>
    /// <returns>The square of the specified value.</returns>
    /// <remarks>
    /// <para>
    /// This method multiplies the value by itself to calculate its square.
    /// If the result exceeds the range of the decimal type, an OverflowException will be thrown.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies a number by itself.
    /// 
    /// For example:
    /// - Square of 4.0m is 16.0m (4.0 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 4.0)
    /// - Square of 0.5m is 0.25m (0.5 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 0.5)
    /// - Square of -3.0m is 9.0m (-3.0 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  -3.0)
    /// 
    /// Squaring always produces a non-negative result (unless the number is NaN,
    /// which is not possible with decimals).
    /// </para>
    /// </remarks>
    public decimal Square(decimal value) => Multiply(value, value);

    /// <summary>
    /// Calculates e raised to the specified power.
    /// </summary>
    /// <param name="value">The power to raise e to.</param>
    /// <returns>e raised to the specified power.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the exponential function (e^value) by converting the decimal to a double,
    /// performing the operation, and then converting the result back to a decimal.
    /// Some precision may be lost in this conversion process.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the mathematical constant e (ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â 2.718) raised to a power.
    /// 
    /// For example:
    /// - e^1 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 2.718m
    /// - e^2 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 7.389m
    /// - e^0 = 1.0m exactly
    /// 
    /// The exponential function is used in many fields including finance (compound interest),
    /// science, and engineering. It grows very rapidly as the input increases.
    /// 
    /// Note that this method first converts to double and then back to decimal,
    /// which may cause a slight loss of precision in some cases.
    /// </para>
    /// </remarks>
    public decimal Exp(decimal value) => (decimal)Math.Exp((double)value);

    /// <summary>
    /// Determines whether two decimal values are equal.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns>true if the values are equal; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two decimal values and returns true if they are exactly equal.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if two numbers have exactly the same value.
    /// 
    /// For example:
    /// - 5.25m equals 5.25m returns true
    /// - 10.0m equals 10.00m returns true (the number of trailing zeros doesn't matter)
    /// - 5.25m equals 5.24m returns false
    /// 
    /// This is a basic comparison operation used in many algorithms.
    /// </para>
    /// </remarks>
    public bool Equals(decimal a, decimal b) => a == b;

    /// <summary>
    /// Raises a decimal value to the specified power.
    /// </summary>
    /// <param name="baseValue">The base value.</param>
    /// <param name="exponent">The exponent.</param>
    /// <returns>The base value raised to the specified power.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates baseValue^exponent by converting both values to doubles,
    /// performing the operation, and then converting the result back to a decimal.
    /// Some precision may be lost in this conversion process.
    /// </para>
    /// <para><b>For Beginners:</b> This method raises one number to the power of another.
    /// 
    /// For example:
    /// - 2.0m raised to power 3.0m is 8.0m (2^3 = 2ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â2 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 2 = 8)
    /// - 10.0m raised to power 2.0m is 100.0m (10^2 = 10ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â10 = 100)
    /// - Any number raised to power 0.0m is 1.0m
    /// - Any number raised to power 1.0m is that number itself
    /// 
    /// Note that this method first converts to double and then back to decimal,
    /// which may cause a slight loss of precision in some cases. This is because
    /// the .NET Framework doesn't provide a native power operation for decimals.
    /// </para>
    /// </remarks>
    public decimal Power(decimal baseValue, decimal exponent) => (decimal)Math.Pow((double)baseValue, (double)exponent);

    /// <summary>
    /// Calculates the natural logarithm of a decimal value.
    /// </summary>
    /// <param name="value">The value to calculate the natural logarithm of.</param>
    /// <returns>The natural logarithm of the specified value.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the natural logarithm (base e) by converting the decimal to a double,
    /// performing the operation, and then converting the result back to a decimal.
    /// Some precision may be lost in this conversion process.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the natural logarithm of a number.
    /// 
    /// The natural logarithm answers the question: "To what power must e be raised to get this number?"
    /// 
    /// For example:
    /// - Log of 1.0m is 0.0m (e^0 = 1)
    /// - Log of 2.718m is approximately 1.0m (e^1 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 2.718)
    /// - Log of 7.389m is approximately 2.0m (e^2 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 7.389)
    /// 
    /// Important notes:
    /// - Log of a negative number or zero will cause an error
    /// - This method first converts to double and then back to decimal,
    ///   which may cause a slight loss of precision in some cases
    /// 
    /// The logarithm function is the inverse of the exponential function.
    /// </para>
    /// </remarks>
    public decimal Log(decimal value) => (decimal)Math.Log((double)value);

    /// <summary>
    /// Determines whether the first decimal value is greater than or equal to the second.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns>true if the first value is greater than or equal to the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two decimal values and returns true if the first is greater than or equal to the second.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is larger than or the same as the second.
    /// 
    /// For example:
    /// - 10.5m >= 5.2m returns true
    /// - 5.0m >= 10.0m returns false
    /// - 5.0m >= 5.0m returns true
    /// 
    /// This is a simple comparison operation used in many algorithms.
    /// </para>
    /// </remarks>
    public bool GreaterThanOrEquals(decimal a, decimal b) => a >= b;

    /// <summary>
    /// Determines whether the first decimal value is less than or equal to the second.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns>true if the first value is less than or equal to the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two decimal values and returns true if the first is less than or equal to the second.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is smaller than or the same as the second.
    /// 
    /// For example:
    /// - 5.2m <= 10.5m returns true
    /// - 10.0m <= 5.0m returns false
    /// - 5.0m <= 5.0m returns true
    /// 
    /// This is a simple comparison operation used in many algorithms.
    /// </para>
    /// </remarks>
    public bool LessThanOrEquals(decimal a, decimal b) => a <= b;

    /// <summary>
    /// Converts a decimal value to a 32-bit integer by rounding to the nearest integer.
    /// </summary>
    /// <param name="value">The decimal value to convert.</param>
    /// <returns>The decimal rounded to the nearest integer and converted to an Int32.</returns>
    /// <remarks>
    /// <para>
    /// This method rounds the decimal value to the nearest integer and then converts it to an Int32.
    /// If the result is outside the range of Int32, an OverflowException will be thrown.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts a decimal to a regular integer.
    /// 
    /// For example:
    /// - 5.7m becomes 6
    /// - 5.2m becomes 5
    /// - 5.5m becomes 6 (rounds to the nearest even number when exactly halfway)
    /// 
    /// This is useful when you need an integer result after performing precise decimal calculations.
    /// 
    /// Note: If the decimal value is too large or too small to fit in an integer,
    /// this will cause an error.
    /// </para>
    /// </remarks>
    public int ToInt32(decimal value) => (int)Math.Round(value);

    /// <summary>
    /// Rounds a decimal value to the nearest integer.
    /// </summary>
    /// <param name="value">The decimal value to round.</param>
    /// <returns>The decimal value rounded to the nearest integer.</returns>
    /// <remarks>
    /// <para>
    /// This method rounds the decimal value to the nearest integer, following the standard rounding rules.
    /// If the fractional part is exactly 0.5, it rounds to the nearest even number (banker's rounding).
    /// </para>
    /// <para><b>For Beginners:</b> This method rounds a decimal to the nearest whole number.
    /// 
    /// For example:
    /// - Round(5.7m) = 6.0m
    /// - Round(5.2m) = 5.0m
    /// - Round(5.5m) = 6.0m
    /// - Round(4.5m) = 4.0m (note this "banker's rounding" - it rounds to the nearest even number when exactly halfway)
    /// 
    /// Unlike ToInt32, this keeps the result as a decimal type.
    /// </para>
    /// </remarks>
    public decimal Round(decimal value) => Math.Round(value);

    /// <summary>
    /// Gets the minimum value that can be represented by a decimal.
    /// </summary>
    /// <value>The minimum value of a decimal, which is approximately -7.9 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 10^28.</value>
    /// <remarks>
    /// <para>
    /// This property returns the minimum value that can be represented by a decimal,
    /// which is -79,228,162,514,264,337,593,543,950,335.
    /// </para>
    /// <para><b>For Beginners:</b> This property gives you the smallest possible decimal value.
    /// 
    /// For decimals, the minimum value is approximately -7.9 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 10^28
    /// (or -79,228,162,514,264,337,593,543,950,335 written out).
    /// 
    /// This is useful when you need to work with the full range of decimal values
    /// or need to check against the minimum possible value.
    /// 
    /// The minimum decimal value is much smaller than what int or long can represent,
    /// but larger than the minimum of float or double.
    /// </para>
    /// </remarks>
    public decimal MinValue => decimal.MinValue;

    /// <summary>
    /// Gets the maximum value that can be represented by a decimal.
    /// </summary>
    /// <value>The maximum value of a decimal, which is approximately 7.9 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 10^28.</value>
    /// <remarks>
    /// <para>
    /// This property returns the maximum value that can be represented by a decimal,
    /// which is 79,228,162,514,264,337,593,543,950,335.
    /// </para>
    /// <para><b>For Beginners:</b> This property gives you the largest possible decimal value.
    /// 
    /// For decimals, the maximum value is approximately 7.9 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 10^28
    /// (or 79,228,162,514,264,337,593,543,950,335 written out).
    /// 
    /// This is useful when you need to work with the full range of decimal values
    /// or need to check against the maximum possible value.
    /// 
    /// The maximum decimal value is much larger than what int or long can represent,
    /// but smaller than the maximum of float or double.
    /// </para>
    /// </remarks>
    public decimal MaxValue => decimal.MaxValue;

    /// <summary>
    /// Determines whether the specified decimal value is NaN (Not a Number).
    /// </summary>
    /// <param name="value">The decimal value to check.</param>
    /// <returns>Always returns false, as decimal values cannot represent NaN.</returns>
    /// <remarks>
    /// <para>
    /// This method always returns false because the decimal type does not support the concept of NaN.
    /// Unlike floating-point types (float and double), decimal can only represent actual numbers.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if a value is "Not a Number" (NaN).
    /// 
    /// Since decimals cannot represent NaN (unlike float or double),
    /// this method always returns false.
    /// 
    /// For example:
    /// - IsNaN(5.25m) returns false
    /// - IsNaN(-10.5m) returns false
    /// - IsNaN(0.0m) returns false
    /// 
    /// This method exists for compatibility with the INumericOperations interface,
    /// which is also used with other numeric types that can represent NaN.
    /// </para>
    /// </remarks>
    public bool IsNaN(decimal value) => false;

    /// <summary>
    /// Determines whether the specified decimal value is infinity.
    /// </summary>
    /// <param name="value">The decimal value to check.</param>
    /// <returns>Always returns false, as decimal values cannot represent infinity.</returns>
    /// <remarks>
    /// <para>
    /// This method always returns false because the decimal type does not support the concept of infinity.
    /// Unlike floating-point types (float and double), decimal can only represent actual numbers within its range.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if a value is infinity.
    /// 
    /// Since decimals cannot represent infinity (unlike float or double),
    /// this method always returns false.
    /// 
    /// For example:
    /// - IsInfinity(5.25m) returns false
    /// - IsInfinity(-10.5m) returns false
    /// - IsInfinity(0.0m) returns false
    /// 
    /// This method exists for compatibility with the INumericOperations interface,
    /// which is also used with other numeric types that can represent infinity.
    /// </para>
    /// </remarks>
    public bool IsInfinity(decimal value) => false;

    /// <summary>
    /// Returns the sign of the specified value as a decimal.
    /// </summary>
    /// <param name="value">The value to check.</param>
    /// <returns>1 if the value is positive; -1 if the value is negative; 0 if the value is zero.</returns>
    /// <remarks>
    /// <para>
    /// This method returns 1 for any positive value, -1 for any negative value, and 0 for zero.
    /// It is used to determine the sign or direction of a value.
    /// </para>
    /// <para><b>For Beginners:</b> This method determines the sign of a number.
    /// 
    /// For example:
    /// - SignOrZero(5.25m) returns 1.0m
    /// - SignOrZero(-10.5m) returns -1.0m
    /// - SignOrZero(0.0m) returns 0.0m
    /// 
    /// This is useful in algorithms that need to know the direction or sign of a value
    /// without caring about its magnitude.
    /// </para>
    /// </remarks>
    public decimal SignOrZero(decimal value)
    {
        if (value > 0) return 1m;
        if (value < 0) return -1m;

        return 0m;
    }

    /// <summary>
    /// Gets the number of bits used for precision in decimal (128 bits).
    /// </summary>
    public int PrecisionBits => 128;

    /// <summary>
    /// Converts a decimal value to float (FP32) precision.
    /// </summary>
    public float ToFloat(decimal value) => (float)value;

    /// <summary>
    /// Converts a float value to decimal precision.
    /// </summary>
    public decimal FromFloat(float value) => (decimal)value;

    /// <summary>
    /// Converts a decimal value to Half (FP16) precision.
    /// </summary>
    /// <remarks>
    /// Warning: Decimal has a much larger range than Half. Values outside [-65504, 65504] will overflow to infinity.
    /// This conversion may also lose significant precision.
    /// </remarks>
    public Half ToHalf(decimal value) => (Half)value;

    /// <summary>
    /// Converts a Half value to decimal precision.
    /// </summary>
    public decimal FromHalf(Half value) => (decimal)(float)value;

    /// <summary>
    /// Converts a decimal value to double precision.
    /// </summary>
    public double ToDouble(decimal value) => (double)value;

    /// <inheritdoc/>
    public bool SupportsCpuAcceleration => false;

    /// <inheritdoc/>
    public bool SupportsGpuAcceleration => false;

    #region IVectorizedOperations<decimal> Implementation - Fallback using sequential loops

    /// <summary>
    /// Performs element-wise addition using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Add(ReadOnlySpan<decimal> x, ReadOnlySpan<decimal> y, Span<decimal> destination)
        => VectorizedOperationsFallback.Add(this, x, y, destination);

    /// <summary>
    /// Performs element-wise subtraction using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Subtract(ReadOnlySpan<decimal> x, ReadOnlySpan<decimal> y, Span<decimal> destination)
        => VectorizedOperationsFallback.Subtract(this, x, y, destination);

    /// <summary>
    /// Performs element-wise multiplication using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Multiply(ReadOnlySpan<decimal> x, ReadOnlySpan<decimal> y, Span<decimal> destination)
        => VectorizedOperationsFallback.Multiply(this, x, y, destination);

    /// <summary>
    /// Performs element-wise division using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Divide(ReadOnlySpan<decimal> x, ReadOnlySpan<decimal> y, Span<decimal> destination)
        => VectorizedOperationsFallback.Divide(this, x, y, destination);

    /// <summary>
    /// Computes dot product using sequential loops (fallback, no SIMD).
    /// </summary>
    public decimal Dot(ReadOnlySpan<decimal> x, ReadOnlySpan<decimal> y)
        => VectorizedOperationsFallback.Dot(this, x, y);

    /// <summary>
    /// Computes sum using sequential loops (fallback, no SIMD).
    /// </summary>
    public decimal Sum(ReadOnlySpan<decimal> x)
        => VectorizedOperationsFallback.Sum(this, x);

    /// <summary>
    /// Finds maximum using sequential loops (fallback, no SIMD).
    /// </summary>
    public decimal Max(ReadOnlySpan<decimal> x)
        => VectorizedOperationsFallback.Max(this, x);

    /// <summary>
    /// Finds minimum using sequential loops (fallback, no SIMD).
    /// </summary>
    public decimal Min(ReadOnlySpan<decimal> x)
        => VectorizedOperationsFallback.Min(this, x);

    /// <summary>
    /// Computes exponential using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Exp(ReadOnlySpan<decimal> x, Span<decimal> destination)
        => VectorizedOperationsFallback.Exp(this, x, destination);

    /// <summary>
    /// Computes natural logarithm using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Log(ReadOnlySpan<decimal> x, Span<decimal> destination)
        => VectorizedOperationsFallback.Log(this, x, destination);

    /// <summary>
    /// Computes hyperbolic tangent using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Tanh(ReadOnlySpan<decimal> x, Span<decimal> destination)
        => VectorizedOperationsFallback.Tanh(this, x, destination);

    /// <summary>
    /// Computes sigmoid using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Sigmoid(ReadOnlySpan<decimal> x, Span<decimal> destination)
        => VectorizedOperationsFallback.Sigmoid(this, x, destination);

    /// <summary>
    /// Computes base-2 logarithm using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Log2(ReadOnlySpan<decimal> x, Span<decimal> destination)
        => VectorizedOperationsFallback.Log2(this, x, destination);

    /// <summary>
    /// Computes softmax using sequential loops (fallback, no SIMD).
    /// </summary>
    public void SoftMax(ReadOnlySpan<decimal> x, Span<decimal> destination)
        => VectorizedOperationsFallback.SoftMax(this, x, destination);

    /// <summary>
    /// Computes cosine similarity using sequential loops (fallback, no SIMD).
    /// </summary>
    public decimal CosineSimilarity(ReadOnlySpan<decimal> x, ReadOnlySpan<decimal> y)
        => VectorizedOperationsFallback.CosineSimilarity(this, x, y);

    private static readonly DecimalOperations _instance = new();

    /// <summary>
    /// Fills a span with a specified value.
    /// </summary>
    public void Fill(Span<decimal> destination, decimal value) => destination.Fill(value);

    /// <summary>
    /// Multiplies each element in a span by a scalar value.
    /// </summary>
    public void MultiplyScalar(ReadOnlySpan<decimal> x, decimal scalar, Span<decimal> destination)
        => VectorizedOperationsFallback.MultiplyScalar(_instance, x, scalar, destination);

    /// <summary>
    /// Divides each element in a span by a scalar value.
    /// </summary>
    public void DivideScalar(ReadOnlySpan<decimal> x, decimal scalar, Span<decimal> destination)
        => VectorizedOperationsFallback.DivideScalar(_instance, x, scalar, destination);

    /// <summary>
    /// Adds a scalar value to each element in a span.
    /// </summary>
    public void AddScalar(ReadOnlySpan<decimal> x, decimal scalar, Span<decimal> destination)
        => VectorizedOperationsFallback.AddScalar(_instance, x, scalar, destination);

    /// <summary>
    /// Subtracts a scalar value from each element in a span.
    /// </summary>
    public void SubtractScalar(ReadOnlySpan<decimal> x, decimal scalar, Span<decimal> destination)
        => VectorizedOperationsFallback.SubtractScalar(_instance, x, scalar, destination);

    /// <summary>
    /// Computes square root of each element using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Sqrt(ReadOnlySpan<decimal> x, Span<decimal> destination)
        => VectorizedOperationsFallback.Sqrt(_instance, x, destination);

    /// <summary>
    /// Computes absolute value of each element using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Abs(ReadOnlySpan<decimal> x, Span<decimal> destination)
        => VectorizedOperationsFallback.Abs(_instance, x, destination);

    /// <summary>
    /// Negates each element using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Negate(ReadOnlySpan<decimal> x, Span<decimal> destination)
        => VectorizedOperationsFallback.Negate(_instance, x, destination);

    /// <summary>
    /// Clips each element to the specified range using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Clip(ReadOnlySpan<decimal> x, decimal min, decimal max, Span<decimal> destination)
        => VectorizedOperationsFallback.Clip(_instance, x, min, max, destination);

    /// <summary>
    /// Raises each element to a specified power using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Pow(ReadOnlySpan<decimal> x, decimal power, Span<decimal> destination)
        => VectorizedOperationsFallback.Pow(_instance, x, power, destination);

    /// <summary>
    /// Copies elements from source to destination.
    /// </summary>
    public void Copy(ReadOnlySpan<decimal> source, Span<decimal> destination)
        => source.CopyTo(destination);

    #endregion
}
