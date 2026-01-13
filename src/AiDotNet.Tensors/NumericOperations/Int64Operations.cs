using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Operators;

namespace AiDotNet.Tensors.NumericOperations;

/// <summary>
/// Provides operations for long integer numbers in neural network computations.
/// </summary>
/// <remarks>
/// <para>
/// The Int64Operations class implements the INumericOperations interface for the long (Int64) data type.
/// It provides essential mathematical operations needed for neural network computations, including
/// basic arithmetic, comparison, and mathematical functions adapted for long integer values.
/// </para>
/// <para><b>For Beginners:</b> This class handles math operations for whole numbers that can be very large.
/// 
/// Think of it as a calculator specifically designed for neural networks that:
/// - Performs basic operations like addition and multiplication with large whole numbers
/// - Handles special math functions adapted to work with long integers
/// - Manages number conversions and comparisons
/// 
/// For example, when a neural network needs to work with very large numbers (like billions or trillions),
/// it can use this class instead of regular integers (which have a smaller range). The "long" data type 
/// can store numbers from -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807, which is much larger 
/// than the standard int type.
/// </para>
/// </remarks>
public class Int64Operations : INumericOperations<long>
{
    private static readonly Int64Operations _instance = new Int64Operations();

    /// <summary>
    /// Adds two long integer numbers.
    /// </summary>
    /// <param name="a">The first number.</param>
    /// <param name="b">The second number.</param>
    /// <returns>The sum of the two numbers.</returns>
    /// <remarks>
    /// <para>
    /// This method performs simple addition of two long integer values and returns their sum.
    /// It is a fundamental operation used throughout neural network computations.
    /// </para>
    /// <para><b>For Beginners:</b> This method adds two large whole numbers together, like 2000000000L + 3000000000L = 5000000000L.
    /// 
    /// The "L" suffix indicates that these are long integers, which can handle much larger values than regular integers.
    /// </para>
    /// </remarks>
    public long Add(long a, long b) => a + b;

    /// <summary>
    /// Subtracts one long integer number from another.
    /// </summary>
    /// <param name="a">The number to subtract from.</param>
    /// <param name="b">The number to subtract.</param>
    /// <returns>The difference between the two numbers.</returns>
    /// <remarks>
    /// <para>
    /// This method performs subtraction of two long integer values, computing a - b.
    /// Subtraction is essential for calculating errors and adjustments during neural network training.
    /// </para>
    /// <para><b>For Beginners:</b> This method subtracts the second number from the first, like 5000000000L - 2000000000L = 3000000000L.
    /// </para>
    /// </remarks>
    public long Subtract(long a, long b) => a - b;

    /// <summary>
    /// Multiplies two long integer numbers.
    /// </summary>
    /// <param name="a">The first number.</param>
    /// <param name="b">The second number.</param>
    /// <returns>The product of the two numbers.</returns>
    /// <remarks>
    /// <para>
    /// This method performs multiplication of two long integer values and returns their product.
    /// Multiplication is used extensively in neural networks, particularly for weight applications.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies two numbers together, like 2000000L ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  4000L = 8000000000L.
    /// 
    /// In neural networks, multiplication is often used when:
    /// - Applying weights to inputs
    /// - Scaling values during training
    /// - Computing repeated additions
    /// 
    /// The long data type is particularly useful when multiplying large numbers that might exceed
    /// the range of regular integers.
    /// </para>
    /// </remarks>
    public long Multiply(long a, long b) => a * b;

    /// <summary>
    /// Divides one long integer number by another.
    /// </summary>
    /// <param name="a">The dividend (number being divided).</param>
    /// <param name="b">The divisor (number to divide by).</param>
    /// <returns>The quotient of the division, truncated to a long integer.</returns>
    /// <remarks>
    /// <para>
    /// This method performs integer division of two long values, computing a / b. Note that this is integer
    /// division, which truncates the result to the nearest integer toward zero. For example, 5L / 2L equals 2L,
    /// not 2.5. Care should be taken to ensure the divisor is not zero to avoid runtime exceptions.
    /// </para>
    /// <para><b>For Beginners:</b> This method divides the first number by the second, but drops any remainder.
    /// 
    /// For example:
    /// - 10000000000L ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  2L = 5000000000L (exact division, no remainder)
    /// - 7000000000L ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  2L = 3500000000L (exact division, no remainder)
    /// - 5L ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  10L = 0L (less than 1, so the integer result is 0)
    /// 
    /// This is different from regular division you might do with a calculator because:
    /// - It only gives you the whole number part of the answer
    /// - Any remainder or decimal part is discarded
    /// 
    /// Note: This method doesn't check if the second number is zero, which would cause an error
    /// (you can't divide by zero). Make sure the second number is not zero before using this method.
    /// </para>
    /// </remarks>
    public long Divide(long a, long b) => a / b;

    /// <summary>
    /// Negates a long integer number.
    /// </summary>
    /// <param name="a">The number to negate.</param>
    /// <returns>The negated value.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the negative of the input value. If the input is positive, the output is negative,
    /// and vice versa. Zero remains as zero when negated.
    /// </para>
    /// <para><b>For Beginners:</b> This method flips the sign of a number.
    /// 
    /// Examples:
    /// - Negate(5000000000L) returns -5000000000L
    /// - Negate(-3000000000L) returns 3000000000L
    /// - Negate(0L) returns 0L
    /// 
    /// In neural networks, negation is often used when:
    /// - Computing negative gradients for gradient descent
    /// - Implementing certain activation functions
    /// - Reversing values for specific calculations
    /// 
    /// Note: Be careful when negating MinValue (-9,223,372,036,854,775,808), as its positive equivalent
    /// cannot be represented as a long integer.
    /// </para>
    /// </remarks>
    public long Negate(long a) => -a;

    /// <summary>
    /// Gets the zero value for the long type.
    /// </summary>
    /// <value>The value 0L.</value>
    /// <remarks>
    /// <para>
    /// This property returns the zero value for the long type, which is 0L.
    /// Zero is an important value in neural networks for initialization, comparison, and accumulation.
    /// </para>
    /// <para><b>For Beginners:</b> This property simply gives you the number zero (0) as a long integer.
    /// 
    /// The "L" suffix indicates that this is a long integer value.
    /// 
    /// In neural networks, zero is commonly used for:
    /// - Initializing accumulators before adding values to them
    /// - Checking if a value is exactly zero
    /// - As a default or baseline value in many calculations
    /// </para>
    /// </remarks>
    public long Zero => 0L;

    /// <summary>
    /// Gets the one value for the long type.
    /// </summary>
    /// <value>The value 1L.</value>
    /// <remarks>
    /// <para>
    /// This property returns the one value for the long type, which is 1L.
    /// One is used in neural networks for initialization, identity operations, and counting.
    /// </para>
    /// <para><b>For Beginners:</b> This property simply gives you the number one (1) as a long integer.
    /// 
    /// The "L" suffix indicates that this is a long integer value.
    /// 
    /// In neural networks, one is commonly used for:
    /// - Identity operations (multiplying by 1 leaves a value unchanged)
    /// - Initializing certain weights or biases
    /// - Incrementing counters
    /// </para>
    /// </remarks>
    public long One => 1L;

    /// <summary>
    /// Calculates the square root of a long integer number, truncated to a long integer.
    /// </summary>
    /// <param name="value">The number to calculate the square root of.</param>
    /// <returns>The square root of the input value, truncated to a long integer.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the square root of the input value using the Math.Sqrt function
    /// and converts the result to a long integer by truncation. The input should be non-negative;
    /// otherwise, the result will be undefined.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the square root of a number and gives you a whole number result.
    /// 
    /// The square root of a number is a value that, when multiplied by itself, gives the original number.
    /// For example:
    /// - The square root of 9 is 3 (because 3 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 3 = 9)
    /// - The square root of 16 is 4 (because 4 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 4 = 16)
    /// - The square root of 2 would be approximately 1.414, but this method returns 1 (the whole number part only)
    /// 
    /// This method drops any decimal part of the result, so:
    /// - Sqrt(9L) returns 3L
    /// - Sqrt(10L) returns 3L (not 3.162...)
    /// - Sqrt(100000000L) returns 10000L
    /// 
    /// Note: You should only use this with positive numbers. If you try to calculate the square root
    /// of a negative number, you'll get an undefined result.
    /// </para>
    /// </remarks>
    public long Sqrt(long value) => (long)Math.Sqrt(value);

    /// <summary>
    /// Converts a double-precision floating-point number to a long integer.
    /// </summary>
    /// <param name="value">The double-precision value to convert.</param>
    /// <returns>The equivalent long integer value, truncated toward zero.</returns>
    /// <remarks>
    /// <para>
    /// This method converts a double-precision floating-point value (double) to a long integer (long).
    /// The conversion truncates the value toward zero, discarding any fractional part.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts a decimal number to a large whole number by removing the decimal part.
    /// 
    /// For example:
    /// - FromDouble(3.7) returns 3L (not 4L, because it drops the decimal part instead of rounding)
    /// - FromDouble(-2.8) returns -2L (not -3L, because it drops the decimal part)
    /// - FromDouble(1000000000.9) returns 1000000000L
    /// 
    /// This is different from rounding because:
    /// - It always moves toward zero (cuts off the decimal part)
    /// - It doesn't look at whether the decimal part is closer to 0 or 1
    /// 
    /// This conversion is used when:
    /// - You need a whole number result from a calculation that produces decimals
    /// - You're working with functions that use doubles but your neural network uses long integers
    /// - Precision beyond whole numbers isn't needed for your calculations
    /// </para>
    /// </remarks>
    public long FromDouble(double value) => (long)value;

    /// <summary>
    /// Checks if one long integer is greater than another.
    /// </summary>
    /// <param name="a">The first number to compare.</param>
    /// <param name="b">The second number to compare.</param>
    /// <returns>True if the first number is greater than the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two long integer values and returns true if the first value is greater than the second.
    /// Comparison operations are commonly used in neural networks for conditional logic and optimizations.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is larger than the second.
    /// 
    /// For example:
    /// - GreaterThan(5000000000L, 3000000000L) returns true because 5000000000L is greater than 3000000000L
    /// - GreaterThan(2000000000L, 7000000000L) returns false because 2000000000L is not greater than 7000000000L
    /// - GreaterThan(4000000000L, 4000000000L) returns false because the numbers are equal
    /// 
    /// In neural networks, comparisons like this are used for:
    /// - Finding maximum values
    /// - Implementing decision logic in algorithms
    /// - Detecting specific conditions during training
    /// </para>
    /// </remarks>
    public bool GreaterThan(long a, long b) => a > b;

    /// <summary>
    /// Checks if one long integer is less than another.
    /// </summary>
    /// <param name="a">The first number to compare.</param>
    /// <param name="b">The second number to compare.</param>
    /// <returns>True if the first number is less than the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two long integer values and returns true if the first value is less than the second.
    /// Like the GreaterThan method, this comparison is used in various conditional operations in neural networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is smaller than the second.
    /// 
    /// For example:
    /// - LessThan(3000000000L, 5000000000L) returns true because 3000000000L is less than 5000000000L
    /// - LessThan(7000000000L, 2000000000L) returns false because 7000000000L is not less than 2000000000L
    /// - LessThan(4000000000L, 4000000000L) returns false because the numbers are equal
    /// 
    /// In neural networks, this comparison is commonly used for:
    /// - Finding minimum values
    /// - Implementing thresholds in algorithms
    /// - Checking if values have fallen below certain limits during training
    /// </para>
    /// </remarks>
    public bool LessThan(long a, long b) => a < b;

    /// <summary>
    /// Calculates the absolute value of a long integer.
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
    /// - Abs(5000000000L) returns 5000000000L (already positive)
    /// - Abs(-3000000000L) returns 3000000000L (converts negative to positive)
    /// - Abs(0L) returns 0L
    /// 
    /// In neural networks, absolute values are used for:
    /// - Measuring error magnitudes (how far predictions are from actual values)
    /// - Implementing certain activation functions
    /// - Checking if values are within certain tolerances, regardless of sign
    /// 
    /// Note: Be careful with the minimum value of long (-9,223,372,036,854,775,808L), as taking its
    /// absolute value could cause an overflow because the positive equivalent is outside the
    /// representable range of a long integer.
    /// </para>
    /// </remarks>
    public long Abs(long value) => Math.Abs(value);

    /// <summary>
    /// Squares a long integer number.
    /// </summary>
    /// <param name="value">The number to square.</param>
    /// <returns>The square of the input value.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the square of the input value by multiplying it by itself.
    /// Squaring is a common operation in neural networks, particularly in error calculations and regularization.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies a number by itself.
    /// 
    /// For example:
    /// - Square(4L) returns 16L (4L ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  4L = 16L)
    /// - Square(-3L) returns 9L (-3L ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  -3L = 9L)
    /// - Square(1000000L) returns 1000000000000L (1000000L ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  1000000L = 1000000000000L)
    /// 
    /// In neural networks, squaring is commonly used for:
    /// - Calculating squared errors (a measure of how far predictions are from actual values)
    /// - L2 regularization (a technique to prevent overfitting)
    /// - Computing variances and standard deviations
    /// 
    /// Note that squaring always produces a non-negative result, even when the input is negative.
    /// Also, be careful when squaring large values, as they might exceed the range of the long type.
    /// </para>
    /// </remarks>
    public long Square(long value) => Multiply(value, value);

    /// <summary>
    /// Calculates the exponential function (e raised to the power of the specified value), rounded to a long integer.
    /// </summary>
    /// <param name="value">The exponent.</param>
    /// <returns>The value of e raised to the specified power, rounded to the nearest long integer.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates e (approximately 2.71828) raised to the power of the input value
    /// using the Math.Exp function, rounds the result, and converts it to a long integer. The exponential function
    /// typically produces a floating-point result, so rounding is applied to convert to a long integer.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates "e" raised to a power and gives a whole number result.
    /// 
    /// In mathematics, "e" is a special number (approximately 2.71828) that appears naturally in many calculations.
    /// This method computes e^value and rounds to the nearest whole number:
    /// - Exp(1L) returns 3L (e^ ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  2.71828, rounded to 3)
    /// - Exp(2L) returns 7L (e^ ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  7.38906, rounded to 7)
    /// - Exp(0L) returns 1L (e^ = 1)
    /// - Exp(10L) returns 22026L (e^10 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  22026.4658)
    /// 
    /// Because long integers can't store decimal values, this operation loses precision compared to
    /// its floating-point equivalent. It's generally more common to use floating-point types
    /// for exponential calculations in neural networks.
    /// 
    /// Note that exponential functions grow very quickly, so even moderate input values
    /// can produce results that exceed the range of a long integer.
    /// </para>
    /// </remarks>
    public long Exp(long value) => (long)Math.Round(Math.Exp(value));

    /// <summary>
    /// Checks if two long integers are equal.
    /// </summary>
    /// <param name="a">The first number to compare.</param>
    /// <param name="b">The second number to compare.</param>
    /// <returns>True if the numbers are equal; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two long integer values for equality.
    /// Unlike floating-point equality, integer equality is exact and reliable.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if two numbers are exactly the same.
    /// 
    /// For example:
    /// - Equals(5000000000L, 5000000000L) returns true
    /// - Equals(3000000000L, 4000000000L) returns false
    /// 
    /// Unlike with decimal numbers (float/double), comparing integers for equality is straightforward
    /// and reliable because integers have exact representations in the computer.
    /// </para>
    /// </remarks>
    public bool Equals(long a, long b) => a == b;

    public int Compare(long a, long b) => a.CompareTo(b);

    /// <summary>
    /// Raises a long integer to the specified power.
    /// </summary>
    /// <param name="baseValue">The base number.</param>
    /// <param name="exponent">The exponent.</param>
    /// <returns>The base raised to the power of the exponent, converted to a long integer.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates baseValue raised to the power of exponent using the Math.Pow function
    /// and converts the result to a long integer. Power operations are useful for implementing various
    /// mathematical transformations in neural networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method raises a number to a power and gives a whole number result.
    /// 
    /// For example:
    /// - Power(2L, 3L) returns 8L (2ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â² = 2 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 2ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â2 = 8)
    /// - Power(3L, 2L) returns 9L (3ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â² = 3 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 3 = 9)
    /// - Power(10L, 9L) returns 1000000000L (10? = 1 billion)
    /// - Power(5L, 0L) returns 1L (any number raised to the power of 0 is 1)
    /// - Power(2L, -1L) returns 0L (2^-1 = 1/2 = 0.5, but as a long integer this becomes 0)
    /// 
    /// In neural networks, power functions with integer results might be used for:
    /// - Implementing certain discrete activation functions
    /// - Creating specific patterns of values
    /// - Scaling by powers of 10 or 2
    /// 
    /// Note that when the result isn't a whole number (like with negative exponents), the decimal
    /// part is discarded when converting to a long integer, which can lead to a loss of information.
    /// Also, be careful with large exponents, as they can easily produce results that exceed the
    /// range of a long integer.
    /// </para>
    /// </remarks>
    public long Power(long baseValue, long exponent) => (long)Math.Pow(baseValue, exponent);

    /// <summary>
    /// Calculates the natural logarithm (base e) of a long integer, converted to a long integer.
    /// </summary>
    /// <param name="value">The number to calculate the logarithm of.</param>
    /// <returns>The natural logarithm of the input value, converted to a long integer.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the natural logarithm (base e) of the input value using the Math.Log function
    /// and converts the result to a long integer. The input should be positive; otherwise, the result will be undefined.
    /// Since logarithm results are often not whole numbers, this conversion to long integer loses precision.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the natural logarithm of a number and gives a whole number result.
    /// 
    /// The natural logarithm tells you what power you need to raise "e" to get your number:
    /// - Log(3L) returns 1L (because e^ ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  2.718, and the long integer result of ln(3) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  1.099 is 1)
    /// - Log(10L) returns 2L (because ln(10) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  2.303)
    /// - Log(1000000000L) returns 20L (because ln(1000000000) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  20.723)
    /// - Log(1L) returns 0L (because e^ = 1)
    /// 
    /// This integer version of logarithm loses a lot of precision compared to its floating-point
    /// equivalent. In neural networks, it's generally better to use floating-point types for
    /// logarithmic calculations.
    /// 
    /// Note: You should only use this with positive numbers. If you try to calculate the logarithm
    /// of zero or a negative number, you'll get an undefined result.
    /// </para>
    /// </remarks>
    public long Log(long value) => (long)Math.Log(value);

    /// <summary>
    /// Checks if one long integer is greater than or equal to another.
    /// </summary>
    /// <param name="a">The first number to compare.</param>
    /// <param name="b">The second number to compare.</param>
    /// <returns>True if the first number is greater than or equal to the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two long integer values and returns true if the first value is greater than or equal to the second.
    /// This comparison combines the functionality of GreaterThan and Equals methods.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is larger than or the same as the second.
    /// 
    /// For example:
    /// - GreaterThanOrEquals(5000000000L, 3000000000L) returns true because 5000000000L is greater than 3000000000L
    /// - GreaterThanOrEquals(4000000000L, 4000000000L) returns true because the numbers are equal
    /// - GreaterThanOrEquals(2000000000L, 7000000000L) returns false because 2000000000L is less than 7000000000L
    /// 
    /// In neural networks, this type of comparison is used for:
    /// - Implementing thresholds with inclusive boundaries
    /// - Checking if values have reached or exceeded certain levels
    /// - Decision logic in various algorithms
    /// </para>
    /// </remarks>
    public bool GreaterThanOrEquals(long a, long b) => a >= b;

    /// <summary>
    /// Checks if one long integer is less than or equal to another.
    /// </summary>
    /// <param name="a">The first number to compare.</param>
    /// <param name="b">The second number to compare.</param>
    /// <returns>True if the first number is less than or equal to the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two long integer values and returns true if the first value is less than or equal to the second.
    /// This comparison combines the functionality of LessThan and Equals methods.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is smaller than or the same as the second.
    /// 
    /// For example:
    /// - LessThanOrEquals(3000000000L, 5000000000L) returns true because 3000000000L is less than 5000000000L
    /// - LessThanOrEquals(4000000000L, 4000000000L) returns true because the numbers are equal
    /// - LessThanOrEquals(7000000000L, 2000000000L) returns false because 7000000000L is greater than 2000000000L
    /// 
    /// In neural networks, this type of comparison is used for:
    /// - Implementing thresholds with inclusive lower boundaries
    /// - Checking if values have reached or fallen below certain levels
    /// - Decision logic in various algorithms
    /// </para>
    /// </remarks>
    public bool LessThanOrEquals(long a, long b) => a <= b;

    /// <summary>
    /// Converts a long integer to a 32-bit integer.
    /// </summary>
    /// <param name="value">The long integer value to convert.</param>
    /// <returns>The equivalent 32-bit integer value.</returns>
    /// <remarks>
    /// <para>
    /// This method converts a long integer (64-bit) to a standard 32-bit integer. If the long value
    /// is outside the range of a 32-bit integer, the result will be truncated, potentially leading to data loss.
    /// The valid range for 32-bit integers is from -2,147,483,648 to 2,147,483,647.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts a very large whole number to a smaller, standard whole number.
    /// 
    /// For example:
    /// - ToInt32(100L) returns 100 (fits within standard integer range)
    /// - ToInt32(2000000000L) returns 2000000000 (fits within standard integer range)
    /// - ToInt32(3000000000L) would cause truncation because 3000000000 is outside the range of standard integers
    /// 
    /// Be careful when using this method with large values. If the long integer is too large to fit
    /// in a standard integer (beyond roughly ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â±2.1 billion), the conversion will cause unexpected results
    /// due to truncation.
    /// 
    /// In neural networks, this conversion might be needed when:
    /// - Interfacing with methods that require standard integers
    /// - Calculating array indices (which are typically standard integers)
    /// - Reducing memory usage for values known to be within the standard integer range
    /// </para>
    /// </remarks>
    public int ToInt32(long value) => (int)value;

    /// <summary>
    /// Returns the same long integer value (identity operation).
    /// </summary>
    /// <param name="value">The long integer value.</param>
    /// <returns>The same long integer value.</returns>
    /// <remarks>
    /// <para>
    /// This method simply returns the input value unchanged. It serves as an identity operation for long integers.
    /// For long integers, rounding is unnecessary since they are already whole numbers.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns the exact same number you give it.
    /// 
    /// For float or double types, the equivalent method would round the number to the nearest whole number,
    /// but since long integers are already whole numbers, no rounding is needed:
    /// - Round(5000000000L) returns 5000000000L
    /// - Round(-3000000000L) returns -3000000000L
    /// 
    /// This method exists to maintain consistency with the interface used for different numeric types.
    /// </para>
    /// </remarks>
    public long Round(long value) => value;

    public long Floor(long value) => value;
    public long Ceiling(long value) => value;
    public long Frac(long value) => 0;

    /// <summary>
    /// Returns the sine of the specified value (truncated to integer).
    /// </summary>
    public long Sin(long value) => (long)Math.Sin(value);

    /// <summary>
    /// Returns the cosine of the specified value (truncated to integer).
    /// </summary>
    public long Cos(long value) => (long)Math.Cos(value);


    /// <summary>
    /// Gets the minimum possible value for a long integer.
    /// </summary>
    /// <value>The minimum value of long, which is -9,223,372,036,854,775,808.</value>
    /// <remarks>
    /// <para>
    /// This property returns the smallest possible value for a 64-bit signed long integer.
    /// This value represents the lower bound of the range of representable values for the long type.
    /// </para>
    /// <para><b>For Beginners:</b> This property gives you the smallest possible value that a long integer can store.
    /// 
    /// The minimum value for a 64-bit long integer is -9,223,372,036,854,775,808.
    /// That's approximately -9.2 quintillion, an extremely large negative number.
    /// 
    /// In neural networks, knowing the minimum value can be important for:
    /// - Preventing underflow (when calculations produce results too small to represent)
    /// - Setting bounds for certain algorithms
    /// - Implementing special case handling for extreme values
    /// 
    /// Be careful when working near this limit: subtracting from MinValue or negating it directly
    /// will cause an overflow because the positive equivalent (+9,223,372,036,854,775,808) is outside the
    /// representable range of a 64-bit signed long integer.
    /// </para>
    /// </remarks>
    public long MinValue => long.MinValue;

    /// <summary>
    /// Gets the maximum possible value for a long integer.
    /// </summary>
    /// <value>The maximum value of long, which is 9,223,372,036,854,775,807.</value>
    /// <remarks>
    /// <para>
    /// This property returns the largest possible value for a 64-bit signed long integer.
    /// This value represents the upper bound of the range of representable values for the long type.
    /// </para>
    /// <para><b>For Beginners:</b> This property gives you the largest possible value that a long integer can store.
    /// 
    /// The maximum value for a 64-bit long integer is 9,223,372,036,854,775,807.
    /// That's approximately 9.2 quintillion, an extremely large positive number.
    /// 
    /// In neural networks, knowing the maximum value can be important for:
    /// - Preventing overflow (when calculations produce results too large to represent)
    /// - Setting bounds for certain algorithms
    /// - Implementing special case handling for extreme values
    /// 
    /// Standard integers (int) can only go up to about 2.1 billion, so long integers are useful
    /// when dealing with very large counts, indices, or accumulations that might exceed this range.
    /// </para>
    /// </remarks>
    public long MaxValue => long.MaxValue;

    /// <summary>
    /// Determines whether the specified long integer is not a number (NaN).
    /// </summary>
    /// <param name="value">The long integer to test.</param>
    /// <returns>Always returns false because long integers cannot be NaN.</returns>
    /// <remarks>
    /// <para>
    /// This method always returns false because the concept of NaN (Not a Number) does not apply to integers.
    /// NaN is a special value that exists only for floating-point types like float and double.
    /// </para>
    /// <para><b>For Beginners:</b> This method always returns false because all long integers are valid numbers.
    /// 
    /// Unlike floating-point numbers (float/double) which can have special "Not a Number" values,
    /// every possible long integer value represents a valid number. This method exists only to maintain
    /// consistency with the interface used for different numeric types.
    /// 
    /// In neural networks that can work with different numeric types, this consistent interface
    /// allows the same code to be used regardless of whether the network is using integers or
    /// floating-point numbers.
    /// </para>
    /// </remarks>
    public bool IsNaN(long value) => false;

    /// <summary>
    /// Determines whether the specified long integer is infinity.
    /// </summary>
    /// <param name="value">The long integer to test.</param>
    /// <returns>Always returns false because long integers cannot be infinity.</returns>
    /// <remarks>
    /// <para>
    /// This method always returns false because the concept of infinity does not apply to integers.
    /// Infinity is a special value that exists only for floating-point types like float and double.
    /// </para>
    /// <para><b>For Beginners:</b> This method always returns false because long integers cannot represent infinity.
    /// 
    /// Unlike floating-point numbers (float/double) which can have special "Infinity" values,
    /// long integers have a fixed range and cannot represent concepts like infinity. This method exists
    /// only to maintain consistency with the interface used for different numeric types.
    /// 
    /// In neural networks that can work with different numeric types, this consistent interface
    /// allows the same code to be used regardless of whether the network is using integers or
    /// floating-point numbers.
    /// </para>
    /// </remarks>
    public bool IsInfinity(long value) => false;

    /// <summary>
    /// Returns the sign of a long integer, or zero if the number is zero.
    /// </summary>
    /// <param name="value">The long integer to get the sign of.</param>
    /// <returns>1L if the number is positive, -1L if the number is negative, or 0L if the number is zero.</returns>
    /// <remarks>
    /// <para>
    /// This method determines the sign of the input value and returns 1L for positive numbers,
    /// -1L for negative numbers, and 0L for zero. This is similar to the Math.Sign function,
    /// but it returns the sign values as long integers rather than using a different type.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells you if a number is positive, negative, or zero.
    /// 
    /// It returns:
    /// - 1L if the number is positive (greater than zero)
    /// - -1L if the number is negative (less than zero)
    /// - 0L if the number is exactly zero
    /// 
    /// For example:
    /// - SignOrZero(42000000000L) returns 1L
    /// - SignOrZero(-3000000000L) returns -1L
    /// - SignOrZero(0L) returns 0L
    /// 
    /// In neural networks, this function might be used for:
    /// - Implementing custom activation functions (like the sign function)
    /// - Thresholding operations that depend only on the sign of a value
    /// - Converting continuous values to discrete categories (-1, 0, +1)
    /// 
    /// Unlike some sign functions that return either -1 or 1, this method treats zero as its own category,
    /// which can be useful in certain neural network applications.
    /// 
    /// Note that the "L" suffix on values indicates they are long integers rather than standard integers.
    /// </para>
    /// </remarks>
    public long SignOrZero(long value)
    {
        if (value > 0) return 1L;
        if (value < 0) return -1L;

        return 0L;
    }

    /// <summary>
    /// Gets the number of bits used for precision in long (64 bits).
    /// </summary>
    public int PrecisionBits => 64;

    /// <summary>
    /// Converts a long value to float (FP32) precision.
    /// </summary>
    /// <param name="value">The long value to convert.</param>
    /// <returns>The value as a float.</returns>
    /// <remarks>
    /// Note: Large long values may lose precision when converted to float.
    /// </remarks>
    public float ToFloat(long value) => (float)value;

    /// <summary>
    /// Converts a float value to long.
    /// </summary>
    /// <param name="value">The float value to convert.</param>
    /// <returns>The value as a long.</returns>
    /// <remarks>
    /// This conversion will round the float to the nearest integer.
    /// Values outside the long range will be clamped.
    /// </remarks>
    public long FromFloat(float value) => (long)MathExtensions.Clamp(Math.Round(value), long.MinValue, long.MaxValue);

    /// <summary>
    /// Converts a long value to Half (FP16) precision.
    /// </summary>
    /// <param name="value">The long value to convert.</param>
    /// <returns>The value as a Half.</returns>
    /// <remarks>
    /// Note: Large long values will lose significant precision when converted to Half.
    /// </remarks>
    public Half ToHalf(long value) => (Half)value;

    /// <summary>
    /// Converts a Half value to long.
    /// </summary>
    /// <param name="value">The Half value to convert.</param>
    /// <returns>The value as a long.</returns>
    /// <remarks>
    /// This conversion will round the Half to the nearest integer.
    /// </remarks>
    public long FromHalf(Half value) => (long)Math.Round((float)value);

    /// <summary>
    /// Converts a long value to double (FP64) precision.
    /// </summary>
    /// <param name="value">The long value to convert.</param>
    /// <returns>The value as a double.</returns>
    public double ToDouble(long value) => (double)value;

    /// <inheritdoc/>
    public bool SupportsCpuAcceleration => true;

    /// <inheritdoc/>
    public bool SupportsGpuAcceleration => true;

    #region IVectorizedOperations<long> Implementation

    /// <summary>
    /// Performs element-wise addition using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public void Add(ReadOnlySpan<long> x, ReadOnlySpan<long> y, Span<long> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<AddOperatorLong>(x, y, destination);

    /// <summary>
    /// Performs element-wise subtraction using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public void Subtract(ReadOnlySpan<long> x, ReadOnlySpan<long> y, Span<long> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<SubtractOperatorLong>(x, y, destination);

    /// <summary>
    /// Performs element-wise multiplication using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public void Multiply(ReadOnlySpan<long> x, ReadOnlySpan<long> y, Span<long> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<MultiplyOperatorLong>(x, y, destination);

    /// <summary>
    /// Performs element-wise division using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    /// <remarks>
    /// Long integer division doesn't have direct SIMD support, so this uses a scalar fallback
    /// within the SIMD processing loop for optimal cache utilization.
    /// </remarks>
    public void Divide(ReadOnlySpan<long> x, ReadOnlySpan<long> y, Span<long> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<DivideOperatorLong>(x, y, destination);

    /// <summary>
    /// Computes dot product using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public long Dot(ReadOnlySpan<long> x, ReadOnlySpan<long> y)
        => TensorPrimitivesCore.Dot(x, y);

    /// <summary>
    /// Computes sum using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public long Sum(ReadOnlySpan<long> x)
        => TensorPrimitivesCore.Sum(x);

    /// <summary>
    /// Finds maximum using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public long Max(ReadOnlySpan<long> x)
        => TensorPrimitivesCore.Max(x);

    /// <summary>
    /// Finds minimum using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public long Min(ReadOnlySpan<long> x)
        => TensorPrimitivesCore.Min(x);

    /// <summary>
    /// Computes exponential using sequential loops (integers don't support transcendental SIMD).
    /// </summary>
    public void Exp(ReadOnlySpan<long> x, Span<long> destination)
        => VectorizedOperationsFallback.Exp(this, x, destination);

    /// <summary>
    /// Computes natural logarithm using sequential loops (integers don't support transcendental SIMD).
    /// </summary>
    public void Log(ReadOnlySpan<long> x, Span<long> destination)
        => VectorizedOperationsFallback.Log(this, x, destination);

    /// <summary>
    /// Computes hyperbolic tangent using sequential loops (integers don't support transcendental SIMD).
    /// </summary>
    public void Tanh(ReadOnlySpan<long> x, Span<long> destination)
        => VectorizedOperationsFallback.Tanh(this, x, destination);

    /// <summary>
    /// Computes sigmoid using sequential loops (integers don't support transcendental SIMD).
    /// </summary>
    public void Sigmoid(ReadOnlySpan<long> x, Span<long> destination)
        => VectorizedOperationsFallback.Sigmoid(this, x, destination);

    /// <summary>
    /// Computes base-2 logarithm using sequential loops (integers don't support transcendental SIMD).
    /// </summary>
    public void Log2(ReadOnlySpan<long> x, Span<long> destination)
        => VectorizedOperationsFallback.Log2(this, x, destination);

    /// <summary>
    /// Computes softmax using sequential loops (integers don't support transcendental SIMD).
    /// </summary>
    public void SoftMax(ReadOnlySpan<long> x, Span<long> destination)
        => VectorizedOperationsFallback.SoftMax(this, x, destination);

    /// <summary>
    /// Computes cosine similarity using sequential loops (integers don't support this SIMD operation).
    /// </summary>
    public long CosineSimilarity(ReadOnlySpan<long> x, ReadOnlySpan<long> y)
        => VectorizedOperationsFallback.CosineSimilarity(this, x, y);

    /// <summary>
    /// Fills the destination span with the specified value.
    /// </summary>
    public void Fill(Span<long> destination, long value) => destination.Fill(value);

    /// <summary>
    /// Multiplies each element in the span by a scalar value.
    /// </summary>
    public void MultiplyScalar(ReadOnlySpan<long> x, long scalar, Span<long> destination)
        => VectorizedOperationsFallback.MultiplyScalar(_instance, x, scalar, destination);

    /// <summary>
    /// Divides each element in the span by a scalar value.
    /// </summary>
    public void DivideScalar(ReadOnlySpan<long> x, long scalar, Span<long> destination)
        => VectorizedOperationsFallback.DivideScalar(_instance, x, scalar, destination);

    /// <summary>
    /// Adds a scalar value to each element in the span.
    /// </summary>
    public void AddScalar(ReadOnlySpan<long> x, long scalar, Span<long> destination)
        => VectorizedOperationsFallback.AddScalar(_instance, x, scalar, destination);

    /// <summary>
    /// Subtracts a scalar value from each element in the span.
    /// </summary>
    public void SubtractScalar(ReadOnlySpan<long> x, long scalar, Span<long> destination)
        => VectorizedOperationsFallback.SubtractScalar(_instance, x, scalar, destination);

    /// <summary>
    /// Computes the square root of each element in the span.
    /// </summary>
    public void Sqrt(ReadOnlySpan<long> x, Span<long> destination)
        => VectorizedOperationsFallback.Sqrt(_instance, x, destination);

    /// <summary>
    /// Computes the absolute value of each element in the span.
    /// </summary>
    public void Abs(ReadOnlySpan<long> x, Span<long> destination)
        => VectorizedOperationsFallback.Abs(_instance, x, destination);

    /// <summary>
    /// Negates each element in the span.
    /// </summary>
    public void Negate(ReadOnlySpan<long> x, Span<long> destination)
        => VectorizedOperationsFallback.Negate(_instance, x, destination);

    /// <summary>
    /// Clips each element in the span to the specified range.
    /// </summary>
    public void Clip(ReadOnlySpan<long> x, long min, long max, Span<long> destination)
        => VectorizedOperationsFallback.Clip(_instance, x, min, max, destination);

    /// <summary>
    /// Raises each element in the span to the specified power.
    /// </summary>
    public void Pow(ReadOnlySpan<long> x, long power, Span<long> destination)
        => VectorizedOperationsFallback.Pow(_instance, x, power, destination);

    /// <summary>
    /// Copies elements from the source span to the destination span.
    /// </summary>
    public void Copy(ReadOnlySpan<long> source, Span<long> destination) => source.CopyTo(destination);

    #endregion

    /// <summary>
    /// Copies the source values to destination (floor is identity for integers).
    /// </summary>
    public void Floor(ReadOnlySpan<long> x, Span<long> destination) => x.CopyTo(destination);

    /// <summary>
    /// Copies the source values to destination (ceiling is identity for integers).
    /// </summary>
    public void Ceiling(ReadOnlySpan<long> x, Span<long> destination) => x.CopyTo(destination);

    /// <summary>
    /// Fills destination with zeros (fractional part of integers is always zero).
    /// </summary>
    public void Frac(ReadOnlySpan<long> x, Span<long> destination)
    {
        if (destination.Length < x.Length)
        {
            throw new ArgumentException(
                $"Destination span length ({destination.Length}) must be at least as long as source span ({x.Length}).",
                nameof(destination));
        }
        destination.Slice(0, x.Length).Fill(0);
    }

    /// <summary>
    /// Computes sine of each element (truncated to long).
    /// </summary>
    public void Sin(ReadOnlySpan<long> x, Span<long> destination)
    {
        if (destination.Length < x.Length)
        {
            throw new ArgumentException(
                $"Destination span length ({destination.Length}) must be at least as long as source span ({x.Length}).",
                nameof(destination));
        }
        for (int i = 0; i < x.Length; i++)
            destination[i] = (long)Math.Sin(x[i]);
    }

    /// <summary>
    /// Computes cosine of each element (truncated to long).
    /// </summary>
    public void Cos(ReadOnlySpan<long> x, Span<long> destination)
    {
        if (destination.Length < x.Length)
        {
            throw new ArgumentException(
                $"Destination span length ({destination.Length}) must be at least as long as source span ({x.Length}).",
                nameof(destination));
        }
        for (int i = 0; i < x.Length; i++)
            destination[i] = (long)Math.Cos(x[i]);
    }

    public void MultiplyAdd(ReadOnlySpan<long> x, ReadOnlySpan<long> y, long scalar, Span<long> destination)
        => VectorizedOperationsFallback.MultiplyAdd(this, x, y, scalar, destination);

    public void ToFloatSpan(ReadOnlySpan<long> source, Span<float> destination)
        => VectorizedOperationsFallback.ToFloatSpan(this, source, destination);

    public void FromFloatSpan(ReadOnlySpan<float> source, Span<long> destination)
        => VectorizedOperationsFallback.FromFloatSpan(this, source, destination);

    public void ToHalfSpan(ReadOnlySpan<long> source, Span<Half> destination)
        => VectorizedOperationsFallback.ToHalfSpan(this, source, destination);

    public void FromHalfSpan(ReadOnlySpan<Half> source, Span<long> destination)
        => VectorizedOperationsFallback.FromHalfSpan(this, source, destination);

    public void LeakyReLU(ReadOnlySpan<long> x, long alpha, Span<long> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Input and destination spans must have the same length.");
        VectorizedOperationsFallback.LeakyReLU(this, x, alpha, destination);
    }

    public void GELU(ReadOnlySpan<long> x, Span<long> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Input and destination spans must have the same length.");
        VectorizedOperationsFallback.GELU(this, x, destination);
    }

    public void Mish(ReadOnlySpan<long> x, Span<long> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Input and destination spans must have the same length.");
        VectorizedOperationsFallback.Mish(this, x, destination);
    }

    public void Swish(ReadOnlySpan<long> x, Span<long> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Input and destination spans must have the same length.");
        VectorizedOperationsFallback.Swish(this, x, destination);
    }

    public void ELU(ReadOnlySpan<long> x, long alpha, Span<long> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Input and destination spans must have the same length.");
        VectorizedOperationsFallback.ELU(this, x, alpha, destination);
    }

    public void ReLU(ReadOnlySpan<long> x, Span<long> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Input and destination spans must have the same length.");
        VectorizedOperationsFallback.ReLU(this, x, destination);
    }
}
