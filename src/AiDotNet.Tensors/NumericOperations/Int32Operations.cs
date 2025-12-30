using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Operators;

namespace AiDotNet.Tensors.NumericOperations;

/// <summary>
/// Provides operations for integer numbers in neural network computations.
/// </summary>
/// <remarks>
/// <para>
/// The Int32Operations class implements the INumericOperations interface for the int data type.
/// It provides essential mathematical operations needed for neural network computations, including
/// basic arithmetic, comparison, and mathematical functions adapted for integer values.
/// </para>
/// <para><b>For Beginners:</b> This class handles math operations for whole numbers (like 1, 2, 3).
/// 
/// Think of it as a calculator specifically designed for neural networks that:
/// - Performs basic operations like addition and multiplication with whole numbers
/// - Handles special math functions adapted to work with integers
/// - Manages number conversions and comparisons
/// 
/// For example, when a neural network needs to multiply two whole numbers or calculate the
/// integer square root of a value, it uses the methods in this class. This approach allows
/// the neural network to work with different number types (like int or float) without changing
/// its core logic.
/// </para>
/// </remarks>
public class Int32Operations : INumericOperations<int>
{
    private static readonly Int32Operations _instance = new Int32Operations();

    /// <summary>
    /// Adds two integer numbers.
    /// </summary>
    /// <param name="a">The first number.</param>
    /// <param name="b">The second number.</param>
    /// <returns>The sum of the two numbers.</returns>
    /// <remarks>
    /// <para>
    /// This method performs simple addition of two integer values and returns their sum.
    /// It is a fundamental operation used throughout neural network computations.
    /// </para>
    /// <para><b>For Beginners:</b> This method adds two whole numbers together, like 2 + 3 = 5.
    /// </para>
    /// </remarks>
    public int Add(int a, int b) => a + b;

    /// <summary>
    /// Subtracts one integer number from another.
    /// </summary>
    /// <param name="a">The number to subtract from.</param>
    /// <param name="b">The number to subtract.</param>
    /// <returns>The difference between the two numbers.</returns>
    /// <remarks>
    /// <para>
    /// This method performs subtraction of two integer values, computing a - b.
    /// Subtraction is essential for calculating errors and adjustments during neural network training.
    /// </para>
    /// <para><b>For Beginners:</b> This method subtracts the second number from the first, like 5 - 2 = 3.
    /// </para>
    /// </remarks>
    public int Subtract(int a, int b) => a - b;

    /// <summary>
    /// Multiplies two integer numbers.
    /// </summary>
    /// <param name="a">The first number.</param>
    /// <param name="b">The second number.</param>
    /// <returns>The product of the two numbers.</returns>
    /// <remarks>
    /// <para>
    /// This method performs multiplication of two integer values and returns their product.
    /// Multiplication is used extensively in neural networks, particularly for weight applications.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies two numbers together, like 2 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 4 = 8.
    /// 
    /// In neural networks, multiplication is often used when:
    /// - Applying weights to inputs
    /// - Scaling values during training
    /// - Computing repeated additions
    /// </para>
    /// </remarks>
    public int Multiply(int a, int b) => a * b;

    /// <summary>
    /// Divides one integer number by another.
    /// </summary>
    /// <param name="a">The dividend (number being divided).</param>
    /// <param name="b">The divisor (number to divide by).</param>
    /// <returns>The quotient of the division, truncated to an integer.</returns>
    /// <remarks>
    /// <para>
    /// This method performs integer division of two values, computing a / b. Note that this is integer
    /// division, which truncates the result to the nearest integer toward zero. For example, 5 / 2 equals 2,
    /// not 2.5. Care should be taken to ensure the divisor is not zero to avoid runtime exceptions.
    /// </para>
    /// <para><b>For Beginners:</b> This method divides the first number by the second, but drops any remainder.
    /// 
    /// For example:
    /// - 10 / 2 = 5 (exact division, no remainder)
    /// - 7 / 2 = 3 (not 3.5, because integers can't store decimals)
    /// - 5 / 10 = 0 (less than 1, so the integer result is 0)
    /// 
    /// This is different from regular division you might do with a calculator because:
    /// - It only gives you the whole number part of the answer
    /// - Any remainder or decimal part is discarded
    /// 
    /// Note: This method doesn't check if the second number is zero, which would cause an error
    /// (you can't divide by zero). Make sure the second number is not zero before using this method.
    /// </para>
    /// </remarks>
    public int Divide(int a, int b) => a / b;

    /// <summary>
    /// Negates an integer number.
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
    /// - Negate(5) returns -5
    /// - Negate(-3) returns 3
    /// - Negate(0) returns 0
    /// 
    /// In neural networks, negation is often used when:
    /// - Computing negative gradients for gradient descent
    /// - Implementing certain activation functions
    /// - Reversing values for specific calculations
    /// </para>
    /// </remarks>
    public int Negate(int a) => -a;

    /// <summary>
    /// Gets the zero value for the int type.
    /// </summary>
    /// <value>The value 0.</value>
    /// <remarks>
    /// <para>
    /// This property returns the zero value for the int type, which is 0.
    /// Zero is an important value in neural networks for initialization, comparison, and accumulation.
    /// </para>
    /// <para><b>For Beginners:</b> This property simply gives you the number zero (0) as an integer.
    /// 
    /// In neural networks, zero is commonly used for:
    /// - Initializing accumulators before adding values to them
    /// - Checking if a value is exactly zero
    /// - As a default or baseline value in many calculations
    /// </para>
    /// </remarks>
    public int Zero => 0;

    /// <summary>
    /// Gets the one value for the int type.
    /// </summary>
    /// <value>The value 1.</value>
    /// <remarks>
    /// <para>
    /// This property returns the one value for the int type, which is 1.
    /// One is used in neural networks for initialization, identity operations, and counting.
    /// </para>
    /// <para><b>For Beginners:</b> This property simply gives you the number one (1) as an integer.
    /// 
    /// In neural networks, one is commonly used for:
    /// - Identity operations (multiplying by 1 leaves a value unchanged)
    /// - Initializing certain weights or biases
    /// - Incrementing counters
    /// </para>
    /// </remarks>
    public int One => 1;

    /// <summary>
    /// Calculates the square root of an integer number, truncated to an integer.
    /// </summary>
    /// <param name="value">The number to calculate the square root of.</param>
    /// <returns>The square root of the input value, truncated to an integer.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the square root of the input value using the Math.Sqrt function
    /// and converts the result to an integer by truncation. The input should be non-negative;
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
    /// - Sqrt(9) returns 3
    /// - Sqrt(10) returns 3 (not 3.162...)
    /// - Sqrt(15) returns 3 (not 3.873...)
    /// 
    /// Note: You should only use this with positive numbers. If you try to calculate the square root
    /// of a negative number, you'll get an undefined result.
    /// </para>
    /// </remarks>
    public int Sqrt(int value) => (int)Math.Sqrt(value);

    /// <summary>
    /// Converts a double-precision floating-point number to an integer.
    /// </summary>
    /// <param name="value">The double-precision value to convert.</param>
    /// <returns>The equivalent integer value, truncated toward zero.</returns>
    /// <remarks>
    /// <para>
    /// This method converts a double-precision floating-point value (double) to an integer (int).
    /// The conversion truncates the value toward zero, discarding any fractional part.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts a decimal number to a whole number by removing the decimal part.
    /// 
    /// For example:
    /// - FromDouble(3.7) returns 3 (not 4, because it drops the decimal part instead of rounding)
    /// - FromDouble(-2.8) returns -2 (not -3, because it drops the decimal part)
    /// 
    /// This is different from rounding because:
    /// - It always moves toward zero (cuts off the decimal part)
    /// - It doesn't look at whether the decimal part is closer to 0 or 1
    /// 
    /// This conversion is used when:
    /// - You need a whole number result from a calculation that produces decimals
    /// - You're working with functions that use doubles but your neural network uses integers
    /// - Precision beyond whole numbers isn't needed for your calculations
    /// </para>
    /// </remarks>
    public int FromDouble(double value) => (int)value;

    /// <summary>
    /// Checks if one integer is greater than another.
    /// </summary>
    /// <param name="a">The first number to compare.</param>
    /// <param name="b">The second number to compare.</param>
    /// <returns>True if the first number is greater than the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two integer values and returns true if the first value is greater than the second.
    /// Comparison operations are commonly used in neural networks for conditional logic and optimizations.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is larger than the second.
    /// 
    /// For example:
    /// - GreaterThan(5, 3) returns true because 5 is greater than 3
    /// - GreaterThan(2, 7) returns false because 2 is not greater than 7
    /// - GreaterThan(4, 4) returns false because the numbers are equal
    /// 
    /// In neural networks, comparisons like this are used for:
    /// - Finding maximum values
    /// - Implementing decision logic in algorithms
    /// - Detecting specific conditions during training
    /// </para>
    /// </remarks>
    public bool GreaterThan(int a, int b) => a > b;

    /// <summary>
    /// Checks if one integer is less than another.
    /// </summary>
    /// <param name="a">The first number to compare.</param>
    /// <param name="b">The second number to compare.</param>
    /// <returns>True if the first number is less than the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two integer values and returns true if the first value is less than the second.
    /// Like the GreaterThan method, this comparison is used in various conditional operations in neural networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is smaller than the second.
    /// 
    /// For example:
    /// - LessThan(3, 5) returns true because 3 is less than 5
    /// - LessThan(7, 2) returns false because 7 is not less than 2
    /// - LessThan(4, 4) returns false because the numbers are equal
    /// 
    /// In neural networks, this comparison is commonly used for:
    /// - Finding minimum values
    /// - Implementing thresholds in algorithms
    /// - Checking if values have fallen below certain limits during training
    /// </para>
    /// </remarks>
    public bool LessThan(int a, int b) => a < b;

    /// <summary>
    /// Calculates the absolute value of an integer.
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
    /// - Abs(5) returns 5 (already positive)
    /// - Abs(-3) returns 3 (converts negative to positive)
    /// - Abs(0) returns 0
    /// 
    /// In neural networks, absolute values are used for:
    /// - Measuring error magnitudes (how far predictions are from actual values)
    /// - Implementing certain activation functions
    /// - Checking if values are within certain tolerances, regardless of sign
    /// </para>
    /// </remarks>
    public int Abs(int value) => Math.Abs(value);

    /// <summary>
    /// Squares an integer number.
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
    /// - Square(4) returns 16 (4 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 4 = 16)
    /// - Square(-3) returns 9 (-3 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â -3 = 9)
    /// - Square(0) returns 0 (0 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 0 = 0)
    /// 
    /// In neural networks, squaring is commonly used for:
    /// - Calculating squared errors (a measure of how far predictions are from actual values)
    /// - L2 regularization (a technique to prevent overfitting)
    /// - Computing variances and standard deviations
    /// 
    /// Note that squaring always produces a non-negative result, even when the input is negative.
    /// </para>
    /// </remarks>
    public int Square(int value) => Multiply(value, value);

    /// <summary>
    /// Calculates the exponential function (e raised to the power of the specified value), rounded to an integer.
    /// </summary>
    /// <param name="value">The exponent.</param>
    /// <returns>The value of e raised to the specified power, rounded to the nearest integer.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates e (approximately 2.71828) raised to the power of the input value
    /// using the Math.Exp function, rounds the result, and converts it to an integer. The exponential function
    /// typically produces a floating-point result, so rounding is applied to convert to an integer.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates "e" raised to a power and gives a whole number result.
    /// 
    /// In mathematics, "e" is a special number (approximately 2.71828) that appears naturally in many calculations.
    /// This method computes e^value and rounds to the nearest whole number:
    /// - Exp(1) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  3 (e^1 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  2.71828, rounded to 3)
    /// - Exp(2) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  7 (e^2 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  7.38906, rounded to 7)
    /// - Exp(0) returns 1 (e^ = 1)
    /// - Exp(-1) returns 0 (e^-1 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  0.36788, rounded to 0)
    /// 
    /// Because integers can't store decimal values, this operation loses precision compared to
    /// its floating-point equivalent. It's generally more common to use floating-point types
    /// for exponential calculations in neural networks.
    /// </para>
    /// </remarks>
    public int Exp(int value) => (int)Math.Round(Math.Exp(value));

    /// <summary>
    /// Checks if two integers are equal.
    /// </summary>
    /// <param name="a">The first number to compare.</param>
    /// <param name="b">The second number to compare.</param>
    /// <returns>True if the numbers are equal; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two integer values for equality.
    /// Unlike floating-point equality, integer equality is exact and reliable.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if two numbers are exactly the same.
    /// 
    /// For example:
    /// - Equals(5, 5) returns true
    /// - Equals(3, 4) returns false
    /// 
    /// Unlike with decimal numbers (float/double), comparing integers for equality is straightforward
    /// and reliable because integers have exact representations in the computer.
    /// </para>
    /// </remarks>
    public bool Equals(int a, int b) => a == b;

    public int Compare(int a, int b) => a.CompareTo(b);

    /// <summary>
    /// Raises an integer to the specified power.
    /// </summary>
    /// <param name="baseValue">The base number.</param>
    /// <param name="exponent">The exponent.</param>
    /// <returns>The base raised to the power of the exponent, converted to an integer.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates baseValue raised to the power of exponent using the Math.Pow function
    /// and converts the result to an integer. Power operations are useful for implementing various
    /// mathematical transformations in neural networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method raises a number to a power and gives a whole number result.
    /// 
    /// For example:
    /// - Power(2, 3) returns 8 (2ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â² = 2 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 2ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â2 = 8)
    /// - Power(3, 2) returns 9 (3ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â² = 3 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 3 = 9)
    /// - Power(5, 0) returns 1 (any number raised to the power of 0 is 1)
    /// - Power(2, -1) returns 0 (2^-1 = 1/2 = 0.5, but as an integer this becomes 0)
    /// 
    /// In neural networks, power functions with integer results might be used for:
    /// - Implementing certain discrete activation functions
    /// - Creating specific patterns of values
    /// 
    /// Note that when the result isn't a whole number (like with negative exponents), the decimal
    /// part is discarded when converting to an integer, which can lead to a loss of information.
    /// </para>
    /// </remarks>
    public int Power(int baseValue, int exponent) => (int)Math.Pow(baseValue, exponent);

    /// <summary>
    /// Calculates the natural logarithm (base e) of an integer, converted to an integer.
    /// </summary>
    /// <param name="value">The number to calculate the logarithm of.</param>
    /// <returns>The natural logarithm of the input value, converted to an integer.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the natural logarithm (base e) of the input value using the Math.Log function
    /// and converts the result to an integer. The input should be positive; otherwise, the result will be undefined.
    /// Since logarithm results are often not whole numbers, this conversion to integer loses precision.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the natural logarithm of a number and gives a whole number result.
    /// 
    /// The natural logarithm tells you what power you need to raise "e" to get your number:
    /// - Log(3) returns 1 (because e^ ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  2.718, and the integer result of ln(3) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  1.099 is 1)
    /// - Log(10) returns 2 (because ln(10) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  2.303)
    /// - Log(1) returns 0 (because e^ = 1)
    /// 
    /// This integer version of logarithm loses a lot of precision compared to its floating-point
    /// equivalent. In neural networks, it's generally better to use floating-point types for
    /// logarithmic calculations.
    /// 
    /// Note: You should only use this with positive numbers. If you try to calculate the logarithm
    /// of zero or a negative number, you'll get an undefined result.
    /// </para>
    /// </remarks>
    public int Log(int value) => (int)Math.Log(value);

    /// <summary>
    /// Checks if one integer is greater than or equal to another.
    /// </summary>
    /// <param name="a">The first number to compare.</param>
    /// <param name="b">The second number to compare.</param>
    /// <returns>True if the first number is greater than or equal to the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two integer values and returns true if the first value is greater than or equal to the second.
    /// This comparison combines the functionality of GreaterThan and Equals methods.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is larger than or the same as the second.
    /// 
    /// For example:
    /// - GreaterThanOrEquals(5, 3) returns true because 5 is greater than 3
    /// - GreaterThanOrEquals(4, 4) returns true because the numbers are equal
    /// - GreaterThanOrEquals(2, 7) returns false because 2 is less than 7
    /// 
    /// In neural networks, this type of comparison is used for:
    /// - Implementing thresholds with inclusive boundaries
    /// - Checking if values have reached or exceeded certain levels
    /// - Decision logic in various algorithms
    /// </para>
    /// </remarks>
    public bool GreaterThanOrEquals(int a, int b) => a >= b;

    /// <summary>
    /// Checks if one integer is less than or equal to another.
    /// </summary>
    /// <param name="a">The first number to compare.</param>
    /// <param name="b">The second number to compare.</param>
    /// <returns>True if the first number is less than or equal to the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two integer values and returns true if the first value is less than or equal to the second.
    /// This comparison combines the functionality of LessThan and Equals methods.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is smaller than or the same as the second.
    /// 
    /// For example:
    /// - LessThanOrEquals(3, 5) returns true because 3 is less than 5
    /// - LessThanOrEquals(4, 4) returns true because the numbers are equal
    /// - LessThanOrEquals(7, 2) returns false because 7 is greater than 2
    /// 
    /// In neural networks, this type of comparison is used for:
    /// - Implementing thresholds with inclusive lower boundaries
    /// - Checking if values have reached or fallen below certain levels
    /// - Decision logic in various algorithms
    /// </para>
    /// </remarks>
    public bool LessThanOrEquals(int a, int b) => a <= b;

    /// <summary>
    /// Returns the same integer value (identity operation).
    /// </summary>
    /// <param name="value">The integer value.</param>
    /// <returns>The same integer value.</returns>
    /// <remarks>
    /// <para>
    /// This method simply returns the input value unchanged. It serves as an identity operation for integers.
    /// This is consistent with the INumericOperations interface but has no effect for integers.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns the exact same number you give it.
    /// 
    /// This is called an "identity operation" because it doesn't change the value:
    /// - ToInt32(5) returns 5
    /// - ToInt32(-3) returns -3
    /// 
    /// This method exists to maintain consistency with the interface. For other numeric types like
    /// float or double, the equivalent method would convert to an integer, but since we're already
    /// working with integers, no conversion is needed.
    /// </para>
    /// </remarks>
    public int ToInt32(int value) => value;

    /// <summary>
    /// Returns the same integer value (identity operation).
    /// </summary>
    /// <param name="value">The integer value.</param>
    /// <returns>The same integer value.</returns>
    /// <remarks>
    /// <para>
    /// This method simply returns the input value unchanged. It serves as an identity operation for integers.
    /// For integers, rounding is unnecessary since they are already whole numbers.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns the exact same number you give it.
    /// 
    /// For float or double types, the equivalent method would round the number to the nearest whole number,
    /// but since integers are already whole numbers, no rounding is needed:
    /// - Round(5) returns 5
    /// - Round(-3) returns -3
    /// 
    /// This method exists to maintain consistency with the interface used for different numeric types.
    /// </para>
    /// </remarks>
    public int Round(int value) => value;

    public int Floor(int value) => value;
    public int Ceiling(int value) => value;
    public int Frac(int value) => 0;

    /// <summary>
    /// Returns the sine of the specified value (truncated to integer).
    /// </summary>
    public int Sin(int value) => (int)Math.Sin(value);

    /// <summary>
    /// Returns the cosine of the specified value (truncated to integer).
    /// </summary>
    public int Cos(int value) => (int)Math.Cos(value);


    /// <summary>
    /// Gets the minimum possible value for an int.
    /// </summary>
    /// <value>The minimum value of int, which is -2,147,483,648.</value>
    /// <remarks>
    /// <para>
    /// This property returns the smallest possible value for a 32-bit signed integer.
    /// This value represents the lower bound of the range of representable values for the int type.
    /// </para>
    /// <para><b>For Beginners:</b> This property gives you the smallest possible value that an int can store.
    /// 
    /// The minimum value for a 32-bit integer is -2,147,483,648.
    /// 
    /// In neural networks, knowing the minimum value can be important for:
    /// - Preventing underflow (when calculations produce results too small to represent)
    /// - Setting bounds for certain algorithms
    /// - Implementing special case handling for extreme values
    /// 
    /// Be careful when working near this limit: subtracting from MinValue or negating it directly
    /// will cause an overflow because the positive equivalent (+2,147,483,648) is outside the
    /// representable range of a 32-bit signed integer.
    /// </para>
    /// </remarks>
    public int MinValue => int.MinValue;

    /// <summary>
    /// Gets the maximum possible value for an int.
    /// </summary>
    /// <value>The maximum value of int, which is 2,147,483,647.</value>
    /// <remarks>
    /// <para>
    /// This property returns the largest possible value for a 32-bit signed integer.
    /// This value represents the upper bound of the range of representable values for the int type.
    /// </para>
    /// <para><b>For Beginners:</b> This property gives you the largest possible value that an int can store.
    /// 
    /// The maximum value for a 32-bit integer is 2,147,483,647.
    /// 
    /// In neural networks, knowing the maximum value can be important for:
    /// - Preventing overflow (when calculations produce results too large to represent)
    /// - Setting bounds for certain algorithms
    /// - Implementing special case handling for extreme values
    /// </para>
    /// </remarks>
    public int MaxValue => int.MaxValue;

    /// <summary>
    /// Determines whether the specified integer is not a number (NaN).
    /// </summary>
    /// <param name="value">The integer to test.</param>
    /// <returns>Always returns false because integers cannot be NaN.</returns>
    /// <remarks>
    /// <para>
    /// This method always returns false because the concept of NaN (Not a Number) does not apply to integers.
    /// NaN is a special value that exists only for floating-point types like float and double.
    /// </para>
    /// <para><b>For Beginners:</b> This method always returns false because all integers are valid numbers.
    /// 
    /// Unlike floating-point numbers (float/double) which can have special "Not a Number" values,
    /// every possible integer value represents a valid number. This method exists only to maintain
    /// consistency with the interface used for different numeric types.
    /// 
    /// In neural networks that can work with different numeric types, this consistent interface
    /// allows the same code to be used regardless of whether the network is using integers or
    /// floating-point numbers.
    /// </para>
    /// </remarks>
    public bool IsNaN(int value) => false;

    /// <summary>
    /// Determines whether the specified integer is infinity.
    /// </summary>
    /// <param name="value">The integer to test.</param>
    /// <returns>Always returns false because integers cannot be infinity.</returns>
    /// <remarks>
    /// <para>
    /// This method always returns false because the concept of infinity does not apply to integers.
    /// Infinity is a special value that exists only for floating-point types like float and double.
    /// </para>
    /// <para><b>For Beginners:</b> This method always returns false because integers cannot represent infinity.
    /// 
    /// Unlike floating-point numbers (float/double) which can have special "Infinity" values,
    /// integers have a fixed range and cannot represent concepts like infinity. This method exists
    /// only to maintain consistency with the interface used for different numeric types.
    /// 
    /// In neural networks that can work with different numeric types, this consistent interface
    /// allows the same code to be used regardless of whether the network is using integers or
    /// floating-point numbers.
    /// </para>
    /// </remarks>
    public bool IsInfinity(int value) => false;

    /// <summary>
    /// Returns the sign of an integer, or zero if the number is zero.
    /// </summary>
    /// <param name="value">The integer to get the sign of.</param>
    /// <returns>1 if the number is positive, -1 if the number is negative, or 0 if the number is zero.</returns>
    /// <remarks>
    /// <para>
    /// This method determines the sign of the input value and returns 1 for positive numbers,
    /// -1 for negative numbers, and 0 for zero. This is similar to the Math.Sign function,
    /// but it returns the sign values as integers rather than using a different type.
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
    public int SignOrZero(int value)
    {
        if (value > 0) return 1;
        if (value < 0) return -1;

        return 0;
    }

    /// <summary>
    /// Gets the number of bits used for precision in int (32 bits).
    /// </summary>
    public int PrecisionBits => 32;

    /// <summary>
    /// Converts an int value to float (FP32) precision.
    /// </summary>
    /// <param name="value">The int value to convert.</param>
    /// <returns>The value as a float.</returns>
    public float ToFloat(int value) => (float)value;

    /// <summary>
    /// Converts a float value to int.
    /// </summary>
    /// <param name="value">The float value to convert.</param>
    /// <returns>The value as an int.</returns>
    /// <remarks>
    /// This conversion will round the float to the nearest integer and clamp it to the int range.
    /// </remarks>
    public int FromFloat(float value) => (int)MathExtensions.Clamp((long)Math.Round(value), int.MinValue, int.MaxValue);

    /// <summary>
    /// Converts an int value to Half (FP16) precision.
    /// </summary>
    /// <param name="value">The int value to convert.</param>
    /// <returns>The value as a Half.</returns>
    public Half ToHalf(int value) => (Half)value;

    /// <summary>
    /// Converts a Half value to int.
    /// </summary>
    /// <param name="value">The Half value to convert.</param>
    /// <returns>The value as an int.</returns>
    /// <remarks>
    /// This conversion will round the Half to the nearest integer.
    /// </remarks>
    public int FromHalf(Half value) => (int)Math.Round((float)value);

    /// <summary>
    /// Converts an int value to double (FP64) precision.
    /// </summary>
    /// <param name="value">The int value to convert.</param>
    /// <returns>The value as a double.</returns>
    public double ToDouble(int value) => (double)value;

    /// <inheritdoc/>
    public bool SupportsCpuAcceleration => true;

    /// <inheritdoc/>
    public bool SupportsGpuAcceleration => true;

    #region IVectorizedOperations<int> Implementation

    /// <summary>
    /// Performs element-wise addition using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public void Add(ReadOnlySpan<int> x, ReadOnlySpan<int> y, Span<int> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<AddOperatorInt>(x, y, destination);

    /// <summary>
    /// Performs element-wise subtraction using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public void Subtract(ReadOnlySpan<int> x, ReadOnlySpan<int> y, Span<int> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<SubtractOperatorInt>(x, y, destination);

    /// <summary>
    /// Performs element-wise multiplication using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public void Multiply(ReadOnlySpan<int> x, ReadOnlySpan<int> y, Span<int> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<MultiplyOperatorInt>(x, y, destination);

    /// <summary>
    /// Performs element-wise division using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    /// <remarks>
    /// Integer division doesn't have direct SIMD support, so this uses a scalar fallback
    /// within the SIMD processing loop for optimal cache utilization.
    /// </remarks>
    public void Divide(ReadOnlySpan<int> x, ReadOnlySpan<int> y, Span<int> destination)
        => TensorPrimitivesCore.InvokeSpanSpanIntoSpan<DivideOperatorInt>(x, y, destination);

    /// <summary>
    /// Computes dot product using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public int Dot(ReadOnlySpan<int> x, ReadOnlySpan<int> y)
        => TensorPrimitivesCore.Dot(x, y);

    /// <summary>
    /// Computes sum using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public int Sum(ReadOnlySpan<int> x)
        => TensorPrimitivesCore.Sum(x);

    /// <summary>
    /// Finds maximum using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public int Max(ReadOnlySpan<int> x)
        => TensorPrimitivesCore.Max(x);

    /// <summary>
    /// Finds minimum using SIMD-optimized operations via TensorPrimitivesCore.
    /// </summary>
    public int Min(ReadOnlySpan<int> x)
        => TensorPrimitivesCore.Min(x);

    /// <summary>
    /// Computes exponential using sequential loops (integers don't support transcendental SIMD).
    /// </summary>
    public void Exp(ReadOnlySpan<int> x, Span<int> destination)
        => VectorizedOperationsFallback.Exp(this, x, destination);

    /// <summary>
    /// Computes natural logarithm using sequential loops (integers don't support transcendental SIMD).
    /// </summary>
    public void Log(ReadOnlySpan<int> x, Span<int> destination)
        => VectorizedOperationsFallback.Log(this, x, destination);

    /// <summary>
    /// Computes hyperbolic tangent using sequential loops (integers don't support transcendental SIMD).
    /// </summary>
    public void Tanh(ReadOnlySpan<int> x, Span<int> destination)
        => VectorizedOperationsFallback.Tanh(this, x, destination);

    /// <summary>
    /// Computes sigmoid using sequential loops (integers don't support transcendental SIMD).
    /// </summary>
    public void Sigmoid(ReadOnlySpan<int> x, Span<int> destination)
        => VectorizedOperationsFallback.Sigmoid(this, x, destination);

    /// <summary>
    /// Computes base-2 logarithm using sequential loops (integers don't support transcendental SIMD).
    /// </summary>
    public void Log2(ReadOnlySpan<int> x, Span<int> destination)
        => VectorizedOperationsFallback.Log2(this, x, destination);

    /// <summary>
    /// Computes softmax using sequential loops (integers don't support transcendental SIMD).
    /// </summary>
    public void SoftMax(ReadOnlySpan<int> x, Span<int> destination)
        => VectorizedOperationsFallback.SoftMax(this, x, destination);

    /// <summary>
    /// Computes cosine similarity using sequential loops (integers don't support this SIMD operation).
    /// </summary>
    public int CosineSimilarity(ReadOnlySpan<int> x, ReadOnlySpan<int> y)
        => VectorizedOperationsFallback.CosineSimilarity(this, x, y);

    /// <summary>
    /// Fills the destination span with the specified value.
    /// </summary>
    public void Fill(Span<int> destination, int value) => destination.Fill(value);

    /// <summary>
    /// Multiplies each element in the span by a scalar value.
    /// </summary>
    public void MultiplyScalar(ReadOnlySpan<int> x, int scalar, Span<int> destination)
        => VectorizedOperationsFallback.MultiplyScalar(_instance, x, scalar, destination);

    /// <summary>
    /// Divides each element in the span by a scalar value.
    /// </summary>
    public void DivideScalar(ReadOnlySpan<int> x, int scalar, Span<int> destination)
        => VectorizedOperationsFallback.DivideScalar(_instance, x, scalar, destination);

    /// <summary>
    /// Adds a scalar value to each element in the span.
    /// </summary>
    public void AddScalar(ReadOnlySpan<int> x, int scalar, Span<int> destination)
        => VectorizedOperationsFallback.AddScalar(_instance, x, scalar, destination);

    /// <summary>
    /// Subtracts a scalar value from each element in the span.
    /// </summary>
    public void SubtractScalar(ReadOnlySpan<int> x, int scalar, Span<int> destination)
        => VectorizedOperationsFallback.SubtractScalar(_instance, x, scalar, destination);

    /// <summary>
    /// Computes the square root of each element in the span.
    /// </summary>
    public void Sqrt(ReadOnlySpan<int> x, Span<int> destination)
        => VectorizedOperationsFallback.Sqrt(_instance, x, destination);

    /// <summary>
    /// Computes the absolute value of each element in the span.
    /// </summary>
    public void Abs(ReadOnlySpan<int> x, Span<int> destination)
        => VectorizedOperationsFallback.Abs(_instance, x, destination);

    /// <summary>
    /// Negates each element in the span.
    /// </summary>
    public void Negate(ReadOnlySpan<int> x, Span<int> destination)
        => VectorizedOperationsFallback.Negate(_instance, x, destination);

    /// <summary>
    /// Clips each element in the span to the specified range.
    /// </summary>
    public void Clip(ReadOnlySpan<int> x, int min, int max, Span<int> destination)
        => VectorizedOperationsFallback.Clip(_instance, x, min, max, destination);

    /// <summary>
    /// Raises each element in the span to the specified power.
    /// </summary>
    public void Pow(ReadOnlySpan<int> x, int power, Span<int> destination)
        => VectorizedOperationsFallback.Pow(_instance, x, power, destination);

    /// <summary>
    /// Copies elements from the source span to the destination span.
    /// </summary>
    public void Copy(ReadOnlySpan<int> source, Span<int> destination) => source.CopyTo(destination);

    #endregion

    public void Floor(ReadOnlySpan<int> x, Span<int> destination) => x.CopyTo(destination);
    public void Ceiling(ReadOnlySpan<int> x, Span<int> destination) => x.CopyTo(destination);
    public void Frac(ReadOnlySpan<int> x, Span<int> destination) => destination.Fill(0);
    public void Sin(ReadOnlySpan<int> x, Span<int> destination)
    {
        for (int i = 0; i < x.Length; i++)
            destination[i] = (int)Math.Sin(x[i]);
    }
    public void Cos(ReadOnlySpan<int> x, Span<int> destination)
    {
        for (int i = 0; i < x.Length; i++)
            destination[i] = (int)Math.Cos(x[i]);
    }

    public void MultiplyAdd(ReadOnlySpan<int> x, ReadOnlySpan<int> y, int scalar, Span<int> destination)
        => VectorizedOperationsFallback.MultiplyAdd(this, x, y, scalar, destination);

    /// <summary>
    /// Converts int span to float span.
    /// </summary>
    public void ToFloatSpan(ReadOnlySpan<int> source, Span<float> destination)
        => VectorizedOperationsFallback.ToFloatSpan(this, source, destination);

    /// <summary>
    /// Converts float span to int span.
    /// </summary>
    public void FromFloatSpan(ReadOnlySpan<float> source, Span<int> destination)
        => VectorizedOperationsFallback.FromFloatSpan(this, source, destination);

    public void ToHalfSpan(ReadOnlySpan<int> source, Span<Half> destination)
        => VectorizedOperationsFallback.ToHalfSpan(this, source, destination);

    public void FromHalfSpan(ReadOnlySpan<Half> source, Span<int> destination)
        => VectorizedOperationsFallback.FromHalfSpan(this, source, destination);
}
