namespace AiDotNet.NumericOperations;

/// <summary>
/// Provides operations for floating-point numbers in neural network computations.
/// </summary>
/// <remarks>
/// <para>
/// The FloatOperations class implements the INumericOperations interface for the float data type.
/// It provides essential mathematical operations needed for neural network computations, including
/// basic arithmetic, comparison, and mathematical functions like square roots and exponentials.
/// </para>
/// <para><b>For Beginners:</b> This class handles math operations for decimal numbers (like 3.14).
/// 
/// Think of it as a calculator specifically designed for neural networks that:
/// - Performs basic operations like addition and multiplication
/// - Handles special math functions like square roots and exponents
/// - Manages number conversions and comparisons
/// 
/// For example, when a neural network needs to multiply two numbers or calculate the square root
/// of a value, it uses the methods in this class. This approach allows the neural network to work
/// with different number types (like float or double) without changing its core logic.
/// </para>
/// </remarks>
public class FloatOperations : INumericOperations<float>
{
    /// <summary>
    /// Adds two floating-point numbers.
    /// </summary>
    /// <param name="a">The first number.</param>
    /// <param name="b">The second number.</param>
    /// <returns>The sum of the two numbers.</returns>
    /// <remarks>
    /// <para>
    /// This method performs simple addition of two floating-point values and returns their sum.
    /// It is a fundamental operation used throughout neural network computations.
    /// </para>
    /// <para><b>For Beginners:</b> This method adds two numbers together, like 2.5 + 3.7 = 6.2.
    /// </para>
    /// </remarks>
    public float Add(float a, float b) => a + b;

    /// <summary>
    /// Subtracts one floating-point number from another.
    /// </summary>
    /// <param name="a">The number to subtract from.</param>
    /// <param name="b">The number to subtract.</param>
    /// <returns>The difference between the two numbers.</returns>
    /// <remarks>
    /// <para>
    /// This method performs subtraction of two floating-point values, computing a - b.
    /// Subtraction is essential for calculating errors and gradients during neural network training.
    /// </para>
    /// <para><b>For Beginners:</b> This method subtracts the second number from the first, like 5.0 - 2.3 = 2.7.
    /// </para>
    /// </remarks>
    public float Subtract(float a, float b) => a - b;

    /// <summary>
    /// Multiplies two floating-point numbers.
    /// </summary>
    /// <param name="a">The first number.</param>
    /// <param name="b">The second number.</param>
    /// <returns>The product of the two numbers.</returns>
    /// <remarks>
    /// <para>
    /// This method performs multiplication of two floating-point values and returns their product.
    /// Multiplication is used extensively in neural networks, particularly for weight applications.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies two numbers together, like 2.5 � 4.0 = 10.0.
    /// 
    /// In neural networks, multiplication is often used when:
    /// - Applying weights to inputs
    /// - Scaling values during training
    /// - Computing gradients for learning
    /// </para>
    /// </remarks>
    public float Multiply(float a, float b) => a * b;

    /// <summary>
    /// Divides one floating-point number by another.
    /// </summary>
    /// <param name="a">The dividend (number being divided).</param>
    /// <param name="b">The divisor (number to divide by).</param>
    /// <returns>The quotient of the division.</returns>
    /// <remarks>
    /// <para>
    /// This method performs division of two floating-point values, computing a / b.
    /// Care should be taken to ensure the divisor is not zero to avoid runtime exceptions.
    /// </para>
    /// <para><b>For Beginners:</b> This method divides the first number by the second, like 10.0 � 2.0 = 5.0.
    /// 
    /// In neural networks, division is commonly used for:
    /// - Normalizing values (making numbers fall within a certain range)
    /// - Computing averages
    /// - Applying certain learning rate adjustments
    /// 
    /// Note: This method doesn't check if the second number is zero, which would cause an error
    /// (you can't divide by zero). Make sure the second number is not zero before using this method.
    /// </para>
    /// </remarks>
    public float Divide(float a, float b) => a / b;

    /// <summary>
    /// Negates a floating-point number.
    /// </summary>
    /// <param name="a">The number to negate.</param>
    /// <returns>The negated value.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the negative of the input value. If the input is positive, the output is negative,
    /// and vice versa. Zero remains unchanged in terms of its absolute value but may change sign.
    /// </para>
    /// <para><b>For Beginners:</b> This method flips the sign of a number.
    /// 
    /// Examples:
    /// - Negate(5.0) returns -5.0
    /// - Negate(-3.2) returns 3.2
    /// - Negate(0.0) returns -0.0 (although this is functionally equivalent to 0.0)
    /// 
    /// In neural networks, negation is often used when:
    /// - Computing negative gradients for gradient descent
    /// - Implementing certain activation functions
    /// - Reversing values for specific calculations
    /// </para>
    /// </remarks>
    public float Negate(float a) => -a;

    /// <summary>
    /// Gets the zero value for the float type.
    /// </summary>
    /// <value>The value 0.0f.</value>
    /// <remarks>
    /// <para>
    /// This property returns the zero value for the float type, which is 0.0f.
    /// Zero is an important value in neural networks for initialization, comparison, and accumulation.
    /// </para>
    /// <para><b>For Beginners:</b> This property simply gives you the number zero (0.0) as a float.
    /// 
    /// In neural networks, zero is commonly used for:
    /// - Initializing accumulators before adding values to them
    /// - Checking if a value is exactly zero (although this is rare with floating-point due to precision issues)
    /// - As a default or baseline value in many calculations
    /// </para>
    /// </remarks>
    public float Zero => 0f;

    /// <summary>
    /// Gets the one value for the float type.
    /// </summary>
    /// <value>The value 1.0f.</value>
    /// <remarks>
    /// <para>
    /// This property returns the one value for the float type, which is 1.0f.
    /// One is used in neural networks for initialization, identity operations, and normalization.
    /// </para>
    /// <para><b>For Beginners:</b> This property simply gives you the number one (1.0) as a float.
    /// 
    /// In neural networks, one is commonly used for:
    /// - Identity operations (multiplying by 1 leaves a value unchanged)
    /// - Initializing certain weights or biases
    /// - Creating certain probability distributions
    /// </para>
    /// </remarks>
    public float One => 1f;

    /// <summary>
    /// Calculates the square root of a floating-point number.
    /// </summary>
    /// <param name="value">The number to calculate the square root of.</param>
    /// <returns>The square root of the input value.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the square root of the input value using the Math.Sqrt function
    /// and converts the result to a float. The input should be non-negative; otherwise, the result will be NaN.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the square root of a number.
    /// 
    /// The square root of a number is a value that, when multiplied by itself, gives the original number.
    /// For example:
    /// - The square root of 9 is 3 (because 3 � 3 = 9)
    /// - The square root of 2 is approximately 1.414
    /// 
    /// Square roots are used in neural networks for:
    /// - Normalizing vectors in certain algorithms
    /// - Implementing certain optimization techniques
    /// - Calculating distances or magnitudes
    /// 
    /// Note: You should only use this with positive numbers. If you try to calculate the square root
    /// of a negative number, you'll get a special value called NaN (Not a Number).
    /// </para>
    /// </remarks>
    public float Sqrt(float value) => (float)Math.Sqrt(value);

    /// <summary>
    /// Converts a double-precision floating-point number to a single-precision floating-point number.
    /// </summary>
    /// <param name="value">The double-precision value to convert.</param>
    /// <returns>The equivalent single-precision value.</returns>
    /// <remarks>
    /// <para>
    /// This method converts a double-precision floating-point value (double) to a single-precision
    /// floating-point value (float). This conversion may result in a loss of precision.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts a more precise decimal number (double) to a less precise decimal number (float).
    /// 
    /// In programming:
    /// - A "double" can store more decimal places than a "float"
    /// - When you convert from double to float, you might lose some precision
    /// 
    /// For example:
    /// - The double 3.141592653589793 might become the float 3.1415927
    /// 
    /// This conversion is used when:
    /// - You need to save memory (floats use less memory than doubles)
    /// - You're working with functions that use doubles but your neural network uses floats
    /// - Precision beyond 6-7 decimal places isn't needed for your calculations
    /// </para>
    /// </remarks>
    public float FromDouble(double value) => (float)value;

    /// <summary>
    /// Checks if one floating-point number is greater than another.
    /// </summary>
    /// <param name="a">The first number to compare.</param>
    /// <param name="b">The second number to compare.</param>
    /// <returns>True if the first number is greater than the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two floating-point values and returns true if the first value is greater than the second.
    /// Comparison operations are commonly used in neural networks for conditional logic and optimizations.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is larger than the second.
    /// 
    /// For example:
    /// - GreaterThan(5.0, 3.0) returns true because 5.0 is greater than 3.0
    /// - GreaterThan(2.0, 7.0) returns false because 2.0 is not greater than 7.0
    /// - GreaterThan(4.0, 4.0) returns false because the numbers are equal
    /// 
    /// In neural networks, comparisons like this are used for:
    /// - Finding maximum values (for example, in certain activation functions)
    /// - Implementing decision logic in algorithms
    /// - Detecting specific conditions during training
    /// </para>
    /// </remarks>
    public bool GreaterThan(float a, float b) => a > b;

    /// <summary>
    /// Checks if one floating-point number is less than another.
    /// </summary>
    /// <param name="a">The first number to compare.</param>
    /// <param name="b">The second number to compare.</param>
    /// <returns>True if the first number is less than the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two floating-point values and returns true if the first value is less than the second.
    /// Like the GreaterThan method, this comparison is used in various conditional operations in neural networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is smaller than the second.
    /// 
    /// For example:
    /// - LessThan(3.0, 5.0) returns true because 3.0 is less than 5.0
    /// - LessThan(7.0, 2.0) returns false because 7.0 is not less than 2.0
    /// - LessThan(4.0, 4.0) returns false because the numbers are equal
    /// 
    /// In neural networks, this comparison is commonly used for:
    /// - Finding minimum values
    /// - Implementing thresholds in algorithms
    /// - Checking if values have fallen below certain limits during training
    /// </para>
    /// </remarks>
    public bool LessThan(float a, float b) => a < b;

    /// <summary>
    /// Calculates the absolute value of a floating-point number.
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
    /// - Abs(5.0) returns 5.0 (already positive)
    /// - Abs(-3.2) returns 3.2 (converts negative to positive)
    /// - Abs(0.0) returns 0.0
    /// 
    /// In neural networks, absolute values are used for:
    /// - Measuring error magnitudes (how far predictions are from actual values)
    /// - Implementing certain activation functions
    /// - Checking if values are within certain tolerances, regardless of sign
    /// </para>
    /// </remarks>
    public float Abs(float value) => Math.Abs(value);

    /// <summary>
    /// Squares a floating-point number.
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
    /// - Square(4.0) returns 16.0 (4.0 � 4.0 = 16.0)
    /// - Square(-3.0) returns 9.0 (-3.0 � -3.0 = 9.0)
    /// - Square(0.5) returns 0.25 (0.5 � 0.5 = 0.25)
    /// 
    /// In neural networks, squaring is commonly used for:
    /// - Calculating squared errors (a measure of how far predictions are from actual values)
    /// - L2 regularization (a technique to prevent overfitting)
    /// - Computing variances and standard deviations
    /// 
    /// Note that squaring always produces a non-negative result, even when the input is negative.
    /// </para>
    /// </remarks>
    public float Square(float value) => Multiply(value, value);

    /// <summary>
    /// Calculates the exponential function (e raised to the power of the specified value).
    /// </summary>
    /// <param name="value">The exponent.</param>
    /// <returns>The value of e raised to the specified power.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates e (approximately 2.71828) raised to the power of the input value
    /// using the Math.Exp function and converts the result to a float. The exponential function
    /// is widely used in neural networks, particularly in activation functions like softmax.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates "e" raised to a power.
    /// 
    /// In mathematics, "e" is a special number (approximately 2.71828) that appears naturally in many calculations.
    /// This method computes e^value:
    /// - Exp(1.0) returns about 2.71828 (e�)
    /// - Exp(2.0) returns about 7.38906 (e�)
    /// - Exp(0.0) returns exactly 1.0 (e�)
    /// - Exp(-1.0) returns about 0.36788 (e?�)
    /// 
    /// The exponential function is fundamental in neural networks for:
    /// - Activation functions like sigmoid and softmax
    /// - Calculating probabilities in certain models
    /// - Transforming values in a way that emphasizes differences
    /// 
    /// It's especially useful because its derivative has a simple form, which makes
    /// training neural networks more efficient.
    /// </para>
    /// </remarks>
    public float Exp(float value) => (float)Math.Exp(value);

    /// <summary>
    /// Checks if two floating-point numbers are equal.
    /// </summary>
    /// <param name="a">The first number to compare.</param>
    /// <param name="b">The second number to compare.</param>
    /// <returns>True if the numbers are equal; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two floating-point values for equality. Due to the nature of floating-point
    /// representation, exact equality comparisons should be used with caution. For approximate equality,
    /// consider using a small epsilon value.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if two numbers are exactly the same.
    /// 
    /// For example:
    /// - Equals(5.0, 5.0) returns true
    /// - Equals(3.1, 3.2) returns false
    /// 
    /// Important note about floating-point numbers: Because of how computers store decimal numbers,
    /// sometimes numbers that should be equal might not be exactly equal. For example:
    /// - 0.1 + 0.2 might not be exactly equal to 0.3 in a computer
    /// 
    /// For this reason, when working with float values in neural networks,
    /// it's often better to check if two numbers are "close enough" rather than exactly equal.
    /// This method checks for exact equality, which may not always be what you want.
    /// </para>
    /// </remarks>
    public bool Equals(float a, float b) => a == b;

    /// <summary>
    /// Raises a floating-point number to the specified power.
    /// </summary>
    /// <param name="baseValue">The base number.</param>
    /// <param name="exponent">The exponent.</param>
    /// <returns>The base raised to the power of the exponent.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates baseValue raised to the power of exponent using the Math.Pow function
    /// and converts the result to a float. Power operations are useful for implementing various
    /// mathematical transformations in neural networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method raises a number to a power.
    /// 
    /// For example:
    /// - Power(2.0, 3.0) returns 8.0 (2� = 2�2�2 = 8)
    /// - Power(4.0, 0.5) returns 2.0 (4^(1/2) = v4 = 2)
    /// - Power(5.0, 0.0) returns 1.0 (any number raised to the power of 0 is 1)
    /// - Power(2.0, -1.0) returns 0.5 (2?� = 1/2 = 0.5)
    /// 
    /// In neural networks, power functions are used for:
    /// - Implementing certain activation functions
    /// - Applying specific mathematical transformations
    /// - Scaling values in a non-linear way
    /// </para>
    /// </remarks>
    public float Power(float baseValue, float exponent) => (float)Math.Pow(baseValue, exponent);

    /// <summary>
    /// Calculates the natural logarithm (base e) of a floating-point number.
    /// </summary>
    /// <param name="value">The number to calculate the logarithm of.</param>
    /// <returns>The natural logarithm of the input value.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the natural logarithm (base e) of the input value using the Math.Log function
    /// and converts the result to a float. The input should be positive; otherwise, the result will be NaN.
    /// Logarithms are used in various loss functions and information-theoretic calculations.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the natural logarithm of a number.
    /// 
    /// The natural logarithm (log base e) is the inverse of the exponential function:
    /// - Log(2.71828) returns about 1.0 (because e� � 2.71828)
    /// - Log(7.38906) returns about 2.0 (because e� � 7.38906)
    /// - Log(1.0) returns exactly 0.0 (because e� = 1)
    /// 
    /// In neural networks, logarithms are commonly used for:
    /// - Cross-entropy loss functions (used in classification problems)
    /// - Information theory calculations
    /// - Converting multiplicative relationships to additive ones
    /// 
    /// Note: You should only use this with positive numbers. If you try to calculate the logarithm
    /// of zero or a negative number, you'll get a special value (NaN or negative infinity).
    /// </para>
    /// </remarks>
    public float Log(float value) => (float)Math.Log(value);

    /// <summary>
    /// Checks if one floating-point number is greater than or equal to another.
    /// </summary>
    /// <param name="a">The first number to compare.</param>
    /// <param name="b">The second number to compare.</param>
    /// <returns>True if the first number is greater than or equal to the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two floating-point values and returns true if the first value is greater than or equal to the second.
    /// This comparison combines the functionality of GreaterThan and Equals methods.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is larger than or the same as the second.
    /// 
    /// For example:
    /// - GreaterThanOrEquals(5.0, 3.0) returns true because 5.0 is greater than 3.0
    /// - GreaterThanOrEquals(4.0, 4.0) returns true because the numbers are equal
    /// - GreaterThanOrEquals(2.0, 7.0) returns false because 2.0 is less than 7.0
    /// 
    /// In neural networks, this type of comparison is used for:
    /// - Implementing thresholds with inclusive boundaries
    /// - Checking if values have reached or exceeded certain levels
    /// - Decision logic in various algorithms
    /// </para>
    /// </remarks>
    public bool GreaterThanOrEquals(float a, float b) => a >= b;

    /// <summary>
    /// Checks if one floating-point number is less than or equal to another.
    /// </summary>
    /// <param name="a">The first number to compare.</param>
    /// <param name="b">The second number to compare.</param>
    /// <returns>True if the first number is less than or equal to the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two floating-point values and returns true if the first value is less than or equal to the second.
    /// This comparison combines the functionality of LessThan and Equals methods.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is smaller than or the same as the second.
    /// 
    /// For example:
    /// - LessThanOrEquals(3.0, 5.0) returns true because 3.0 is less than 5.0
    /// - LessThanOrEquals(4.0, 4.0) returns true because the numbers are equal
    /// - LessThanOrEquals(7.0, 2.0) returns false because 7.0 is greater than 2.0
    /// 
    /// In neural networks, this type of comparison is used for:
    /// - Implementing thresholds with inclusive lower boundaries
    /// - Checking if values have reached or fallen below certain levels
    /// - Decision logic in various algorithms
    /// </para>
    /// </remarks>
    public bool LessThanOrEquals(float a, float b) => a <= b;

    /// <summary>
    /// Converts a floating-point number to a 32-bit integer by rounding.
    /// </summary>
    /// <param name="value">The floating-point value to convert.</param>
    /// <returns>The rounded integer value.</returns>
    /// <remarks>
    /// <para>
    /// This method converts a floating-point value to a 32-bit integer by rounding to the nearest integer.
    /// It uses Math.Round to ensure proper rounding behavior rather than truncation.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts a decimal number to a whole number by rounding.
    /// 
    /// For example:
    /// - ToInt32(3.2) returns 3 (rounds down because 3.2 is closer to 3 than to 4)
    /// - ToInt32(3.7) returns 4 (rounds up because 3.7 is closer to 4 than to 3)
    /// - ToInt32(3.5) returns 4 (rounds to the nearest even number when exactly halfway)
    /// 
    /// In neural networks, this conversion might be used for:
    /// - Converting probabilities to binary decisions
    /// - Discretizing continuous values
    /// - Index calculations
    /// 
    /// Note that this uses proper rounding (to the nearest integer), not just cutting off
    /// the decimal part (truncation).
    /// </para>
    /// </remarks>
    public int ToInt32(float value) => (int)Math.Round(value);

    /// <summary>
    /// Rounds a floating-point number to the nearest integer value.
    /// </summary>
    /// <param name="value">The number to round.</param>
    /// <returns>The nearest integer value as a float.</returns>
    /// <remarks>
    /// <para>
    /// This method rounds the input value to the nearest integer using the Math.Round function,
    /// but returns the result as a float rather than an integer type. This preserves the data type
    /// while eliminating the fractional part.
    /// </para>
    /// <para><b>For Beginners:</b> This method rounds a decimal number to the nearest whole number, but keeps it as a float.
    /// 
    /// Unlike ToInt32 which changes the type to integer, this method keeps the result as a float:
    /// - Round(3.2) returns 3.0 (not 3)
    /// - Round(3.7) returns 4.0 (not 4)
    /// - Round(3.5) returns 4.0 (rounds to the nearest even number when exactly halfway)
    /// 
    /// In neural networks, rounding might be used for:
    /// - Simplifying values while maintaining the float data type
    /// - Preparing outputs for certain types of processing
    /// - Creating "stepped" or discretized activation functions
    /// </para>
    /// </remarks>
    public float Round(float value) => (float)Math.Round((double)value);

    /// <summary>
    /// Gets the minimum possible value for a float.
    /// </summary>
    /// <value>The minimum value of float, approximately -3.4 � 10^38.</value>
    /// <remarks>
    /// <para>
    /// This property returns the smallest possible value for a single-precision floating-point number.
    /// This value represents the lower bound of the range of representable values for the float type.
    /// </para>
    /// <para><b>For Beginners:</b> This property gives you the smallest possible value that a float can store.
    /// 
    /// The minimum value for a float is approximately -3.4 � 10^38, which is an extremely large negative number
    /// (about -340,000,000,000,000,000,000,000,000,000,000,000,000).
    /// 
    /// In neural networks, knowing the minimum value can be important for:
    /// - Preventing underflow (when values become too small for the computer to represent)
    /// - Setting bounds for certain algorithms
    /// - Implementing special case handling for extreme values
    /// </para>
    /// </remarks>
    public float MinValue => float.MinValue;

    /// <summary>
    /// Gets the maximum possible value for a float.
    /// </summary>
    /// <value>The maximum value of float, approximately 3.4 � 10^38.</value>
    /// <remarks>
    /// <para>
    /// This property returns the largest possible value for a single-precision floating-point number.
    /// This value represents the upper bound of the range of representable values for the float type.
    /// </para>
    /// <para><b>For Beginners:</b> This property gives you the largest possible value that a float can store.
    /// 
    /// The maximum value for a float is approximately 3.4 � 10^38, which is an extremely large positive number
    /// (about 340,000,000,000,000,000,000,000,000,000,000,000,000).
    /// 
    /// In neural networks, knowing the maximum value can be important for:
    /// - Preventing overflow (when values become too large for the computer to represent)
    /// - Setting bounds for certain algorithms
    /// - Implementing special case handling for extreme values
    /// </para>
    /// </remarks>
    public float MaxValue => float.MaxValue;

    /// <summary>
    /// Determines whether the specified floating-point number is not a number (NaN).
    /// </summary>
    /// <param name="value">The floating-point number to test.</param>
    /// <returns>True if the value is NaN; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method checks if the input value is NaN (Not a Number), which is a special floating-point value
    /// that represents an undefined or unrepresentable value. NaN can result from operations such as dividing
    /// zero by zero or taking the square root of a negative number.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if a number is "Not a Number" (NaN).
    /// 
    /// NaN is a special value that represents an undefined or impossible result:
    /// - IsNaN(0.0 / 0.0) returns true (dividing zero by zero is undefined)
    /// - IsNaN(Math.Sqrt(-1.0)) returns true (square root of a negative number is not a real number)
    /// - IsNaN(3.14) returns false (normal numbers are not NaN)
    /// 
    /// In neural networks, checking for NaN is important for:
    /// - Detecting calculation errors or numerical instability
    /// - Implementing "guard rails" to prevent propagating invalid values
    /// - Debugging training problems like exploding gradients
    /// 
    /// If your neural network produces NaN values, it typically indicates a problem that needs to be fixed.
    /// </para>
    /// </remarks>

    /// <summary>
    /// Determines whether the specified floating-point number is not a number (NaN).
    /// </summary>
    /// <param name="value">The floating-point number to test.</param>
    /// <returns>True if the value is NaN; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method checks if the input value is NaN (Not a Number), which is a special floating-point value
    /// that represents an undefined or unrepresentable value. NaN can result from operations such as dividing
    /// zero by zero or taking the square root of a negative number.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if a number is "Not a Number" (NaN).
    /// 
    /// NaN is a special value that represents an undefined or impossible result:
    /// - IsNaN(0.0 / 0.0) returns true (dividing zero by zero is undefined)
    /// - IsNaN(Math.Sqrt(-1.0)) returns true (square root of a negative number is not a real number)
    /// - IsNaN(3.14) returns false (normal numbers are not NaN)
    /// 
    /// In neural networks, checking for NaN is important for:
    /// - Detecting calculation errors or numerical instability
    /// - Implementing "guard rails" to prevent propagating invalid values
    /// - Debugging training problems like exploding gradients
    /// 
    /// If your neural network produces NaN values, it typically indicates a problem that needs to be fixed.
    /// </para>
    /// </remarks>
    public bool IsNaN(float value) => float.IsNaN(value);

    /// <summary>
    /// Determines whether the specified floating-point number is positive or negative infinity.
    /// </summary>
    /// <param name="value">The floating-point number to test.</param>
    /// <returns>True if the value is positive or negative infinity; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method checks if the input value is positive infinity or negative infinity, which are special
    /// floating-point values that represent values too large (in magnitude) to be represented by the float type.
    /// Infinity can result from operations such as dividing a non-zero number by zero.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if a number is infinity (either positive or negative).
    /// 
    /// Infinity represents a value that's too large to be stored as a normal float:
    /// - IsInfinity(1.0 / 0.0) returns true (dividing by zero gives positive infinity)
    /// - IsInfinity(-1.0 / 0.0) returns true (dividing a negative number by zero gives negative infinity)
    /// - IsInfinity(1000000.0) returns false (even large normal numbers are not infinity)
    /// 
    /// In neural networks, checking for infinity is important for:
    /// - Detecting overflow errors (when calculations produce values too large to represent)
    /// - Preventing further calculations with infinite values, which could lead to more errors
    /// - Debugging numerical stability issues during training
    /// 
    /// Like NaN, if your neural network produces infinity values, it typically indicates a problem that needs to be addressed.
    /// </para>
    /// </remarks>
    public bool IsInfinity(float value) => float.IsInfinity(value);

    /// <summary>
    /// Returns the sign of a floating-point number, or zero if the number is zero.
    /// </summary>
    /// <param name="value">The floating-point number to get the sign of.</param>
    /// <returns>1.0f if the number is positive, -1.0f if the number is negative, or 0.0f if the number is zero.</returns>
    /// <remarks>
    /// <para>
    /// This method determines the sign of the input value and returns 1.0f for positive numbers,
    /// -1.0f for negative numbers, and 0.0f for zero. This is similar to the Math.Sign function,
    /// but it returns a float value rather than an integer.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells you if a number is positive, negative, or zero.
    /// 
    /// It returns:
    /// - 1.0 if the number is positive (greater than zero)
    /// - -1.0 if the number is negative (less than zero)
    /// - 0.0 if the number is exactly zero
    /// 
    /// For example:
    /// - SignOrZero(42.5) returns 1.0
    /// - SignOrZero(-3.7) returns -1.0
    /// - SignOrZero(0.0) returns 0.0
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
    public float SignOrZero(float value)
    {
        if (value > 0) return 1f;
        if (value < 0) return -1f;

        return 0f;
    }
}