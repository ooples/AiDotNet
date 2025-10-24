namespace AiDotNet.Interfaces;

/// <summary>
/// Defines mathematical operations for numeric types used in machine learning algorithms.
/// </summary>
/// <remarks>
/// This interface provides a unified way to perform mathematical operations regardless of the
/// underlying numeric type (float, double, decimal, etc.), allowing algorithms to work with
/// different numeric types without changing their implementation.
/// 
/// <b>For Beginners:</b> This interface is like a translator that helps AI algorithms work with
/// different types of numbers.
/// 
/// Why is this needed?
/// - AI algorithms need to do math operations (add, multiply, etc.)
/// - Different applications might need different number types (float, double, decimal)
/// - This interface lets the same algorithm work with any number type
/// 
/// Real-world analogy:
/// Think of this interface like a universal calculator. Whether you're working with whole
/// numbers, decimals, or fractions, the calculator knows how to perform operations like
/// addition and multiplication for each type. Similarly, this interface knows how to perform
/// math operations for different numeric types used in AI.
/// 
/// When implementing AI algorithms:
/// - Instead of writing code that only works with one number type (like double)
/// - You can write code that works with this interface
/// - Then your algorithm can work with any number type that has an implementation of this interface
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public interface INumericOperations<T>
{
    /// <summary>
    /// Adds two values together.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    /// <returns>The sum of the two values.</returns>
    T Add(T a, T b);

    /// <summary>
    /// Subtracts the second value from the first value.
    /// </summary>
    /// <param name="a">The value to subtract from.</param>
    /// <param name="b">The value to subtract.</param>
    /// <returns>The result of subtracting b from a.</returns>
    T Subtract(T a, T b);

    /// <summary>
    /// Multiplies two values together.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    /// <returns>The product of the two values.</returns>
    T Multiply(T a, T b);

    /// <summary>
    /// Divides the first value by the second value.
    /// </summary>
    /// <param name="a">The dividend (value being divided).</param>
    /// <param name="b">The divisor (value to divide by).</param>
    /// <returns>The result of dividing a by b.</returns>
    T Divide(T a, T b);

    /// <summary>
    /// Negates a value (changes its sign).
    /// </summary>
    /// <param name="a">The value to negate.</param>
    /// <returns>The negated value (positive becomes negative, negative becomes positive).</returns>
    T Negate(T a);

    /// <summary>
    /// Gets the zero value for the numeric type.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This provides the value of zero in the current number type.
    /// For example, 0 for integers, 0.0 for floating-point numbers.
    /// </remarks>
    T Zero { get; }

    /// <summary>
    /// Gets the value of one for the numeric type.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This provides the value of one in the current number type.
    /// For example, 1 for integers, 1.0 for floating-point numbers.
    /// </remarks>
    T One { get; }

    /// <summary>
    /// Calculates the square root of a value.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The square root of a number is a value that, when multiplied by itself,
    /// gives the original number. For example, the square root of 9 is 3 because 3 � 3 = 9.
    /// </remarks>
    /// <param name="value">The value to calculate the square root of.</param>
    /// <returns>The square root of the value.</returns>
    T Sqrt(T value);

    /// <summary>
    /// Converts a double value to the numeric type T.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This converts a standard decimal number (double) to whatever
    /// number type this interface is working with.
    /// </remarks>
    /// <param name="value">The double value to convert.</param>
    /// <returns>The value converted to type T.</returns>
    T FromDouble(double value);

    /// <summary>
    /// Converts a value of type T to a 32-bit integer.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This converts the current number type to a whole number (integer).
    /// If the original number has a decimal part, it will be truncated (removed).
    /// </remarks>
    /// <param name="value">The value to convert.</param>
    /// <returns>The value converted to a 32-bit integer.</returns>
    int ToInt32(T value);

    /// <summary>
    /// Determines whether the first value is greater than the second value.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns>True if a is greater than b; otherwise, false.</returns>
    bool GreaterThan(T a, T b);

    /// <summary>
    /// Determines whether the first value is less than the second value.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns>True if a is less than b; otherwise, false.</returns>
    bool LessThan(T a, T b);

    /// <summary>
    /// Calculates the absolute value of a number.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The absolute value is the distance of a number from zero,
    /// without considering its sign. For example, the absolute value of both 5 and -5 is 5.
    /// </remarks>
    /// <param name="value">The value to calculate the absolute value of.</param>
    /// <returns>The absolute value.</returns>
    T Abs(T value);

    /// <summary>
    /// Calculates the square of a value.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The square of a number is the result of multiplying the number by itself.
    /// For example, the square of 4 is 16 because 4 � 4 = 16.
    /// </remarks>
    /// <param name="value">The value to square.</param>
    /// <returns>The square of the value.</returns>
    T Square(T value);

    /// <summary>
    /// Calculates the exponential function (e raised to the power of the value).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This calculates "e" (a special mathematical constant, approximately 2.71828)
    /// raised to the power of the given value. For example, Exp(2) is e� � 7.389.
    /// 
    /// The exponential function is commonly used in machine learning for:
    /// - Neural network activation functions
    /// - Probability calculations
    /// - Growth and decay models
    /// </remarks>
    /// <param name="value">The exponent value.</param>
    /// <returns>The value of e raised to the power of the specified value.</returns>
    T Exp(T value);

    /// <summary>
    /// Determines whether two values are equal.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns>True if the values are equal; otherwise, false.</returns>
    bool Equals(T a, T b);

    /// <summary>
    /// Raises a value to the power of an exponent.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This calculates the result of multiplying a number by itself a specific
    /// number of times. For example, Power(2, 3) means 2� = 2 � 2 � 2 = 8.
    /// </remarks>
    /// <param name="baseValue">The base value.</param>
    /// <param name="exponent">The exponent value.</param>
    /// <returns>The base value raised to the power of the exponent.</returns>
    T Power(T baseValue, T exponent);

    /// <summary>
    /// Calculates the natural logarithm (base e) of a value.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The natural logarithm is the inverse of the exponential function.
    /// It answers the question: "To what power must e be raised to get this value?"
    /// For example, Log(7.389) � 2 because e� � 7.389.
    /// 
    /// Natural logarithms are commonly used in machine learning for:
    /// - Converting multiplicative relationships to additive ones
    /// - Working with probabilities (log-likelihood)
    /// - Measuring information (entropy)
    /// </remarks>
    /// <param name="value">The value to calculate the natural logarithm of.</param>
    /// <returns>The natural logarithm of the value.</returns>
    T Log(T value);

    /// <summary>
    /// Determines whether the first value is greater than or equal to the second value.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns>True if a is greater than or equal to b; otherwise, false.</returns>
    bool GreaterThanOrEquals(T a, T b);

    /// <summary>
    /// Determines whether the first value is less than or equal to the second value.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns>True if a is less than or equal to b; otherwise, false.</returns>
    bool LessThanOrEquals(T a, T b);

    /// <summary>
    /// Rounds a value to the nearest integral value.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This converts a number with decimals to the nearest whole number.
    /// For example, Round(3.2) = 3 and Round(3.7) = 4.
    /// </remarks>
    /// <param name="value">The value to round.</param>
    /// <returns>The rounded value.</returns>
    T Round(T value);

    /// <summary>
    /// Gets the minimum possible value for the numeric type.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the smallest number that can be represented in the current number type.
    /// For example, for a 32-bit integer, this would be -2,147,483,648.
    /// </remarks>
    T MinValue { get; }

    /// <summary>
    /// Gets the maximum possible value for the numeric type.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the largest number that can be represented in the current number type.
    /// For example, for a 32-bit integer, this would be 2,147,483,647.
    /// </remarks>
    T MaxValue { get; }

    /// <summary>
    /// Determines whether the specified value is Not a Number (NaN).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> NaN (Not a Number) is a special value that represents an undefined or 
    /// unrepresentable mathematical result. It occurs in situations like:
    /// - Dividing zero by zero
    /// - Taking the square root of a negative number
    /// - Performing operations where the result cannot be expressed as a number
    /// 
    /// In machine learning, checking for NaN values is important because:
    /// - NaN values can cause algorithms to produce incorrect results
    /// - They can silently propagate through calculations (any operation with NaN results in NaN)
    /// - They often indicate a problem in your data or calculations that needs to be fixed
    /// </remarks>
    /// <param name="value">The value to check.</param>
    /// <returns>True if the value is NaN; otherwise, false.</returns>
    bool IsNaN(T value);

    /// <summary>
    /// Determines whether the specified value is positive or negative infinity.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Infinity represents a value that is larger than any finite number.
    /// In computing, infinity can occur when:
    /// - Dividing a number by zero
    /// - A calculation results in a number too large to be represented
    /// 
    /// There are two types of infinity:
    /// - Positive infinity: A value greater than any other number
    /// - Negative infinity: A value less than any other number
    /// 
    /// In machine learning, checking for infinity is important because:
    /// - Infinite values can cause algorithms to behave unexpectedly
    /// - They often indicate numerical overflow or division by zero
    /// - They can lead to incorrect predictions or model behavior
    /// </remarks>
    /// <param name="value">The value to check.</param>
    /// <returns>True if the value is positive or negative infinity; otherwise, false.</returns>
    bool IsInfinity(T value);

    /// <summary>
    /// Returns the sign of the value (1 for positive, -1 for negative) or zero if the value is zero.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This method tells you about the direction or sign of a number:
    /// - For positive numbers, it returns 1 (or the equivalent in type T)
    /// - For negative numbers, it returns -1 (or the equivalent in type T)
    /// - For zero, it returns 0 (or the equivalent in type T)
    /// 
    /// This is useful in machine learning when:
    /// - You need to know only the direction of a value, not its magnitude
    /// - Implementing algorithms that behave differently based on whether values are positive, negative, or zero
    /// - Normalizing or standardizing data
    /// </remarks>
    /// <param name="value">The value to get the sign of.</param>
    /// <returns>The sign of the value (1 for positive, -1 for negative) or zero if the value is zero.</returns>
    T SignOrZero(T value);
}