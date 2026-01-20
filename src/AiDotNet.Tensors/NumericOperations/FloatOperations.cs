using System;
using System.Buffers;
using System.Runtime.InteropServices;
#if NET8_0_OR_GREATER
using System.Numerics.Tensors;
#endif
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using static AiDotNet.Tensors.Helpers.CpuParallelSettings;

namespace AiDotNet.Tensors.NumericOperations;

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
    /// Threshold for parallel processing to maximize memory bandwidth.
    /// TorchSharp uses OpenMP to parallelize all operations including Add/Multiply/Sum.
    /// Set to 50000 to enable parallelism for 100K+ element arrays.
    /// </summary>
    private const int ParallelThreshold = 50000;

    /// <summary>
    /// Minimum chunk size per thread to ensure cache efficiency.
    /// </summary>
    private const int MinChunkSize = 8192;

    /// <summary>
    /// Tries to get the underlying array and offset from a Memory{T} without copying.
    /// Returns false if the memory is not backed by an array.
    /// </summary>
    private static bool TryGetArraySegment(Memory<float> memory, out float[] array, out int offset, out int length)
    {
        if (MemoryMarshal.TryGetArray<float>(memory, out var segment) && segment.Array is not null)
        {
            array = segment.Array;
            offset = segment.Offset;
            length = segment.Count;
            return true;
        }
        array = Array.Empty<float>();
        offset = 0;
        length = 0;
        return false;
    }

    /// <summary>
    /// Tries to get the underlying array and offset from a ReadOnlyMemory{T} without copying.
    /// Returns false if the memory is not backed by an array.
    /// </summary>
    private static bool TryGetArraySegment(ReadOnlyMemory<float> memory, out float[] array, out int offset, out int length)
    {
        if (MemoryMarshal.TryGetArray(memory, out var segment) && segment.Array is not null)
        {
            array = segment.Array;
            offset = segment.Offset;
            length = segment.Count;
            return true;
        }
        array = Array.Empty<float>();
        offset = 0;
        length = 0;
        return false;
    }

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
    /// <para><b>For Beginners:</b> This method multiplies two numbers together, like 2.5 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 4.0 = 10.0.
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
    /// <para><b>For Beginners:</b> This method divides the first number by the second, like 10.0 / 2.0 = 5.0.
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
    /// - The square root of 9 is 3 (because 3 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 3 = 9)
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
    /// - Square(4.0) returns 16.0 (4.0 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 4.0 = 16.0)
    /// - Square(-3.0) returns 9.0 (-3.0 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â -3.0 = 9.0)
    /// - Square(0.5) returns 0.25 (0.5 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 0.5 = 0.25)
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
    /// - Exp(1.0) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  2.71828 (e^1)
    /// - Exp(2.0) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  7.38906 (e^2)
    /// - Exp(0.0) = 1.0 (e^0)
    /// - Exp(-1.0) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  0.36788 (e^-1)
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

    public int Compare(float a, float b) => a.CompareTo(b);

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
    /// - Power(2.0, 3.0) returns 8.0 (2ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â³ = 2ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â2 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 2 = 8)
    /// - Power(4.0, 0.5) returns 2.0 (4^(1/2) = v4 = 2)
    /// - Power(5.0, 0.0) returns 1.0 (any number raised to the power of 0 is 1)
    /// - Power(2.0, -1.0) returns 0.5 (2^-1 = 1/2 = 0.5)
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
    /// - Log(2.71828) returns about 1.0 (because e^ ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  2.71828)
    /// - Log(7.38906) returns about 2.0 (because e^ ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  7.38906)
    /// - Log(1.0) returns exactly 0.0 (because e^ = 1)
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
    /// Returns the largest integer less than or equal to the specified value.
    /// </summary>
    /// <param name="value">The number to floor.</param>
    /// <returns>The largest integer less than or equal to value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Floor rounds a number down to the nearest whole number.
    /// For example, Floor(3.7) returns 3.0, Floor(-2.3) returns -3.0.
    /// </para>
    /// </remarks>
    public float Floor(float value) => (float)Math.Floor(value);

    /// <summary>
    /// Returns the smallest integer greater than or equal to the specified value.
    /// </summary>
    /// <param name="value">The number to ceiling.</param>
    /// <returns>The smallest integer greater than or equal to value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Ceiling rounds a number up to the nearest whole number.
    /// For example, Ceiling(3.2) returns 4.0, Ceiling(-2.7) returns -2.0.
    /// </para>
    /// </remarks>
    public float Ceiling(float value) => (float)Math.Ceiling(value);

    /// <summary>
    /// Returns the fractional part of the specified value.
    /// </summary>
    /// <param name="value">The number to get the fractional part of.</param>
    /// <returns>The fractional part (value - floor(value)).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Frac returns the portion after the decimal point.
    /// For example, Frac(3.7) returns 0.7. Note that Frac(-2.3) returns 0.7 (not -0.3),
    /// since frac is defined as x - floor(x).
    /// </para>
    /// </remarks>
    public float Frac(float value) => value - (float)Math.Floor(value);

    /// <summary>
    /// Returns the sine of the specified angle.
    /// </summary>
    /// <param name="value">An angle, measured in radians.</param>
    /// <returns>The sine of value.</returns>
    public float Sin(float value) => (float)Math.Sin(value);

    /// <summary>
    /// Returns the cosine of the specified angle.
    /// </summary>
    /// <param name="value">An angle, measured in radians.</param>
    /// <returns>The cosine of value.</returns>
    public float Cos(float value) => (float)Math.Cos(value);

    /// <summary>
    /// Gets the minimum possible value for a float.
    /// </summary>
    /// <value>The minimum value of float, approximately -3.4 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 10^38.</value>
    /// <remarks>
    /// <para>
    /// This property returns the smallest possible value for a single-precision floating-point number.
    /// This value represents the lower bound of the range of representable values for the float type.
    /// </para>
    /// <para><b>For Beginners:</b> This property gives you the smallest possible value that a float can store.
    /// 
    /// The minimum value for a float is approximately -3.4 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 10^38, which is an extremely large negative number
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
    /// <value>The maximum value of float, approximately 3.4 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 10^38.</value>
    /// <remarks>
    /// <para>
    /// This property returns the largest possible value for a single-precision floating-point number.
    /// This value represents the upper bound of the range of representable values for the float type.
    /// </para>
    /// <para><b>For Beginners:</b> This property gives you the largest possible value that a float can store.
    /// 
    /// The maximum value for a float is approximately 3.4 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 10^38, which is an extremely large positive number
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

    /// <summary>
    /// Gets the number of bits used for precision in float (32 bits).
    /// </summary>
    public int PrecisionBits => 32;

    /// <summary>
    /// Converts a float value to float (identity operation).
    /// </summary>
    /// <param name="value">The float value.</param>
    /// <returns>The same float value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method returns the same value since it's already a float.
    /// It's here for consistency with the interface, allowing generic code to work with multiple numeric types.
    /// </para>
    /// </remarks>
    public float ToFloat(float value) => value;

    /// <summary>
    /// Converts a float value to float (identity operation).
    /// </summary>
    /// <param name="value">The float value.</param>
    /// <returns>The same float value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method returns the same value since it's already a float.
    /// It's here for consistency with the interface, allowing generic code to work with multiple numeric types.
    /// </para>
    /// </remarks>
    public float FromFloat(float value) => value;

    /// <summary>
    /// Converts a float (FP32) value to Half (FP16) precision.
    /// </summary>
    /// <param name="value">The float value to convert.</param>
    /// <returns>The value converted to Half precision.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This converts a standard 32-bit float to a smaller 16-bit half-precision float.
    ///
    /// This conversion:
    /// - Reduces memory usage by 50%
    /// - Can be faster on modern GPUs with Tensor Cores
    /// - May lose precision (fewer decimal digits)
    /// - May overflow if value is outside Half's range [-65504, 65504]
    ///
    /// Used in mixed-precision training to reduce memory usage while maintaining acceptable accuracy.
    /// </para>
    /// </remarks>
    public Half ToHalf(float value) => (Half)value;

    /// <summary>
    /// Converts a Half (FP16) value to float (FP32) precision.
    /// </summary>
    /// <param name="value">The Half value to convert.</param>
    /// <returns>The value converted to float precision.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This converts a 16-bit half-precision float to a standard 32-bit float.
    ///
    /// This conversion:
    /// - Is lossless (no precision is lost)
    /// - Allows using the wider range of float
    /// - Used when accumulating gradients in mixed-precision training
    /// </para>
    /// </remarks>
    public float FromHalf(Half value) => (float)value;

    /// <summary>
    /// Converts a float (FP32) value to double (FP64) precision.
    /// </summary>
    /// <param name="value">The float value to convert.</param>
    /// <returns>The value converted to double precision.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This converts a 32-bit float to a 64-bit double.
    ///
    /// This conversion:
    /// - Is lossless (no precision is lost)
    /// - Provides more decimal digits of precision
    /// - Uses twice as much memory
    /// - Can represent much larger and smaller numbers
    /// </para>
    /// </remarks>
    public double ToDouble(float value) => (double)value;

    /// <summary>
    /// Indicates that float supports SIMD/CPU-accelerated operations.
    /// </summary>
    public bool SupportsCpuAcceleration => true;

    /// <summary>
    /// Indicates that float supports GPU-accelerated operations.
    /// </summary>
    public bool SupportsGpuAcceleration => true;

    #region IVectorizedOperations<float> Implementation - SIMD via TensorPrimitives

    private static readonly FloatOperations _instance = new();

    /// <summary>
    /// Performs element-wise addition using oneDNN when available, falling back to TensorPrimitives.
    /// Uses multi-threading for large arrays to maximize memory bandwidth.
    /// </summary>
    public void Add(ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<float> destination)
    {
        int length = x.Length;
#if NET8_0_OR_GREATER
        // Try oneDNN first for large arrays - it has highly optimized multi-threaded implementation
        // Validate all spans have sufficient length before unsafe operations
        if (length >= ParallelThreshold && OneDnnProvider.IsAvailable &&
            y.Length >= length && destination.Length >= length)
        {
            unsafe
            {
                fixed (float* xPtr = x, yPtr = y, destPtr = destination)
                {
                    if (OneDnnProvider.TryAdd(xPtr, yPtr, destPtr, length))
                        return;
                }
            }
        }

        // Fallback to TensorPrimitives (handles length validation internally)
        TensorPrimitives.Add(x, y, destination);
#else
        VectorizedOperationsFallback.Add(_instance, x, y, destination);
#endif
    }

    /// <summary>
    /// Performs element-wise subtraction using SIMD-optimized TensorPrimitives.
    /// Memory-bound operation - single-threaded SIMD saturates memory bandwidth.
    /// </summary>
    public void Subtract(ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<float> destination)
    {
#if NET8_0_OR_GREATER
        TensorPrimitives.Subtract(x, y, destination);
#else
        VectorizedOperationsFallback.Subtract(_instance, x, y, destination);
#endif
    }

    /// <summary>
    /// Performs element-wise multiplication using oneDNN when available, falling back to TensorPrimitives.
    /// Uses multi-threading for large arrays to maximize memory bandwidth.
    /// </summary>
    public void Multiply(ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<float> destination)
    {
        int length = x.Length;
#if NET8_0_OR_GREATER
        // Try oneDNN first for large arrays - it has highly optimized multi-threaded implementation
        // Validate all spans have sufficient length before unsafe operations
        if (length >= ParallelThreshold && OneDnnProvider.IsAvailable &&
            y.Length >= length && destination.Length >= length)
        {
            unsafe
            {
                fixed (float* xPtr = x, yPtr = y, destPtr = destination)
                {
                    if (OneDnnProvider.TryMultiply(xPtr, yPtr, destPtr, length))
                        return;
                }
            }
        }

        // Fallback to TensorPrimitives (handles length validation internally)
        TensorPrimitives.Multiply(x, y, destination);
#else
        VectorizedOperationsFallback.Multiply(_instance, x, y, destination);
#endif
    }

    /// <summary>
    /// Performs element-wise division using SIMD-optimized TensorPrimitives.
    /// Memory-bound operation - single-threaded SIMD saturates memory bandwidth.
    /// </summary>
    public void Divide(ReadOnlySpan<float> x, ReadOnlySpan<float> y, Span<float> destination)
    {
#if NET8_0_OR_GREATER
        TensorPrimitives.Divide(x, y, destination);
#else
        VectorizedOperationsFallback.Divide(_instance, x, y, destination);
#endif
    }

    /// <summary>
    /// Computes dot product using SIMD-optimized TensorPrimitives.
    /// </summary>
    public float Dot(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
#if NET8_0_OR_GREATER
        return TensorPrimitives.Dot(x, y);
#else
        return VectorizedOperationsFallback.Dot(_instance, x, y);
#endif
    }

    /// <summary>
    /// Computes sum using SIMD-optimized TensorPrimitives with multi-threading for large arrays.
    /// </summary>
    /// <remarks>
    /// Note: Due to floating-point non-associativity, parallel execution may produce slightly
    /// different results across runs depending on thread execution order. This is generally
    /// acceptable for neural network computations where exact bit-reproducibility is not required.
    /// </remarks>
    public float Sum(ReadOnlySpan<float> x)
    {
        int length = x.Length;
#if NET8_0_OR_GREATER
        if (length >= ParallelThreshold && MaxDegreeOfParallelism > 1)
        {
            // Lock-free parallel reduction: compute partial sums in array, then combine sequentially
            int maxDegree = MaxDegreeOfParallelism;
            int numChunks = Math.Min(maxDegree, (length + MinChunkSize - 1) / MinChunkSize);
            if (numChunks <= 1)
            {
                return TensorPrimitives.Sum(x);
            }

            // Allocate array for partial sums - each thread writes to its own slot
            var partialSums = new float[numChunks];
            int chunkSize = (length + numChunks - 1) / numChunks;

            unsafe
            {
                fixed (float* xPtr = x)
                {
                    var xp = xPtr;
                    Parallel.For(0, numChunks, new ParallelOptions { MaxDegreeOfParallelism = maxDegree }, i =>
                    {
                        int start = i * chunkSize;
                        int count = Math.Min(chunkSize, length - start);
                        if (count > 0)
                        {
                            partialSums[i] = TensorPrimitives.Sum(new ReadOnlySpan<float>(xp + start, count));
                        }
                    });
                }
            }

            // Combine partial sums sequentially (no lock contention)
            float totalSum = 0;
            for (int i = 0; i < numChunks; i++)
            {
                totalSum += partialSums[i];
            }
            return totalSum;
        }
        else
        {
            return TensorPrimitives.Sum(x);
        }
#else
        return VectorizedOperationsFallback.Sum(_instance, x);
#endif
    }

    /// <summary>
    /// Finds maximum using SIMD-optimized TensorPrimitives.
    /// </summary>
    public float Max(ReadOnlySpan<float> x)
    {
#if NET8_0_OR_GREATER
        return TensorPrimitives.Max(x);
#else
        return VectorizedOperationsFallback.Max(_instance, x);
#endif
    }

    /// <summary>
    /// Finds minimum using SIMD-optimized TensorPrimitives.
    /// </summary>
    public float Min(ReadOnlySpan<float> x)
    {
#if NET8_0_OR_GREATER
        return TensorPrimitives.Min(x);
#else
        return VectorizedOperationsFallback.Min(_instance, x);
#endif
    }

    /// <summary>
    /// Computes exponential using SIMD-optimized TensorPrimitives.
    /// </summary>
    public void Exp(ReadOnlySpan<float> x, Span<float> destination)
    {
#if NET8_0_OR_GREATER
        TensorPrimitives.Exp(x, destination);
#else
        VectorizedOperationsFallback.Exp(_instance, x, destination);
#endif
    }

    /// <summary>
    /// Computes natural logarithm using SIMD-optimized TensorPrimitives.
    /// </summary>
    public void Log(ReadOnlySpan<float> x, Span<float> destination)
    {
#if NET8_0_OR_GREATER
        TensorPrimitives.Log(x, destination);
#else
        VectorizedOperationsFallback.Log(_instance, x, destination);
#endif
    }

    /// <summary>
    /// Computes hyperbolic tangent using SIMD-optimized TensorPrimitives.
    /// </summary>
    public void Tanh(ReadOnlySpan<float> x, Span<float> destination)
    {
#if NET8_0_OR_GREATER
        TensorPrimitives.Tanh(x, destination);
#else
        VectorizedOperationsFallback.Tanh(_instance, x, destination);
#endif
    }

    /// <summary>
    /// Computes sigmoid using SIMD-optimized TensorPrimitives with parallel processing for large arrays.
    /// </summary>
    public void Sigmoid(ReadOnlySpan<float> x, Span<float> destination)
    {
        int length = x.Length;
        if (length >= ParallelThreshold && MaxDegreeOfParallelism > 1)
        {
            var pool = ArrayPool<float>.Shared;
            var xArray = pool.Rent(length);
            var destArray = pool.Rent(length);
            try
            {
                x.CopyTo(xArray);

                ParallelForChunks(length, MinChunkSize, (start, count) =>
                {
#if NET8_0_OR_GREATER
                    TensorPrimitives.Sigmoid(
                        new ReadOnlySpan<float>(xArray, start, count),
                        new Span<float>(destArray, start, count));
#else
                    for (int i = start; i < start + count; i++)
                        destArray[i] = 1.0f / (1.0f + (float)Math.Exp(-xArray[i]));
#endif
                });

                new ReadOnlySpan<float>(destArray, 0, length).CopyTo(destination);
            }
            finally
            {
                pool.Return(xArray);
                pool.Return(destArray);
            }
            return;
        }

#if NET8_0_OR_GREATER
        TensorPrimitives.Sigmoid(x, destination);
#else
        VectorizedOperationsFallback.Sigmoid(_instance, x, destination);
#endif
    }

    /// <summary>
    /// Computes base-2 logarithm using SIMD-optimized TensorPrimitives.
    /// </summary>
    public void Log2(ReadOnlySpan<float> x, Span<float> destination)
    {
#if NET8_0_OR_GREATER
        TensorPrimitives.Log2(x, destination);
#else
        VectorizedOperationsFallback.Log2(_instance, x, destination);
#endif
    }

    /// <summary>
    /// Computes softmax using SIMD-optimized TensorPrimitives.
    /// </summary>
    public void SoftMax(ReadOnlySpan<float> x, Span<float> destination)
    {
#if NET8_0_OR_GREATER
        TensorPrimitives.SoftMax(x, destination);
#else
        VectorizedOperationsFallback.SoftMax(_instance, x, destination);
#endif
    }

    /// <summary>
    /// Computes cosine similarity using SIMD-optimized TensorPrimitives.
    /// </summary>
    public float CosineSimilarity(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
#if NET8_0_OR_GREATER
        return TensorPrimitives.CosineSimilarity(x, y);
#else
        return VectorizedOperationsFallback.CosineSimilarity(_instance, x, y);
#endif
    }

    /// <summary>
    /// Fills the destination span with a constant value.
    /// </summary>
    public void Fill(Span<float> destination, float value)
    {
        destination.Fill(value);
    }

    /// <summary>
    /// Multiplies each element by a scalar using SIMD-optimized TensorPrimitives.
    /// </summary>
    public void MultiplyScalar(ReadOnlySpan<float> x, float scalar, Span<float> destination)
    {
#if NET8_0_OR_GREATER
        TensorPrimitives.Multiply(x, scalar, destination);
#else
        VectorizedOperationsFallback.MultiplyScalar(_instance, x, scalar, destination);
#endif
    }

    /// <summary>
    /// Divides each element by a scalar using SIMD-optimized TensorPrimitives.
    /// </summary>
    public void DivideScalar(ReadOnlySpan<float> x, float scalar, Span<float> destination)
    {
#if NET8_0_OR_GREATER
        TensorPrimitives.Divide(x, scalar, destination);
#else
        VectorizedOperationsFallback.DivideScalar(_instance, x, scalar, destination);
#endif
    }

    /// <summary>
    /// Adds a scalar to each element using SIMD-optimized TensorPrimitives.
    /// </summary>
    public void AddScalar(ReadOnlySpan<float> x, float scalar, Span<float> destination)
    {
#if NET8_0_OR_GREATER
        TensorPrimitives.Add(x, scalar, destination);
#else
        VectorizedOperationsFallback.AddScalar(_instance, x, scalar, destination);
#endif
    }

    /// <summary>
    /// Subtracts a scalar from each element using SIMD-optimized TensorPrimitives.
    /// </summary>
    public void SubtractScalar(ReadOnlySpan<float> x, float scalar, Span<float> destination)
    {
#if NET8_0_OR_GREATER
        TensorPrimitives.Subtract(x, scalar, destination);
#else
        VectorizedOperationsFallback.SubtractScalar(_instance, x, scalar, destination);
#endif
    }

    /// <summary>
    /// Computes square root using SIMD-optimized TensorPrimitives.
    /// </summary>
    public void Sqrt(ReadOnlySpan<float> x, Span<float> destination)
    {
#if NET8_0_OR_GREATER
        TensorPrimitives.Sqrt(x, destination);
#else
        VectorizedOperationsFallback.Sqrt(_instance, x, destination);
#endif
    }

    /// <summary>
    /// Computes absolute value using SIMD-optimized TensorPrimitives.
    /// </summary>
    public void Abs(ReadOnlySpan<float> x, Span<float> destination)
    {
#if NET8_0_OR_GREATER
        TensorPrimitives.Abs(x, destination);
#else
        VectorizedOperationsFallback.Abs(_instance, x, destination);
#endif
    }

    /// <summary>
    /// Negates each element using SIMD-optimized TensorPrimitives.
    /// </summary>
    public void Negate(ReadOnlySpan<float> x, Span<float> destination)
    {
#if NET8_0_OR_GREATER
        TensorPrimitives.Negate(x, destination);
#else
        VectorizedOperationsFallback.Negate(_instance, x, destination);
#endif
    }

    /// <summary>
    /// Clips each element to a range.
    /// </summary>
    public void Clip(ReadOnlySpan<float> x, float min, float max, Span<float> destination)
    {
#if NET8_0_OR_GREATER
        TensorPrimitives.Clamp(x, min, max, destination);
#else
        VectorizedOperationsFallback.Clip(_instance, x, min, max, destination);
#endif
    }

    /// <summary>
    /// Computes the power of each element.
    /// </summary>
    public void Pow(ReadOnlySpan<float> x, float power, Span<float> destination)
    {
#if NET8_0_OR_GREATER
        TensorPrimitives.Pow(x, power, destination);
#else
        VectorizedOperationsFallback.Pow(_instance, x, power, destination);
#endif
    }

    /// <summary>
    /// Copies elements from source to destination.
    /// </summary>
    public void Copy(ReadOnlySpan<float> source, Span<float> destination)
    {
        source.CopyTo(destination);
    }

    /// <summary>
    /// Computes floor of each element using SIMD-optimized operations.
    /// </summary>
    public void Floor(ReadOnlySpan<float> x, Span<float> destination)
    {
        Engines.Simd.SimdKernels.Floor(x, destination);
    }

    /// <summary>
    /// Computes ceiling of each element using SIMD-optimized operations.
    /// </summary>
    public void Ceiling(ReadOnlySpan<float> x, Span<float> destination)
    {
        Engines.Simd.SimdKernels.Ceiling(x, destination);
    }

    /// <summary>
    /// Computes fractional part of each element using SIMD-optimized operations.
    /// </summary>
    public void Frac(ReadOnlySpan<float> x, Span<float> destination)
    {
        Engines.Simd.SimdKernels.Frac(x, destination);
    }

    /// <summary>
    /// Computes sine of each element using SIMD-optimized operations.
    /// </summary>
    public void Sin(ReadOnlySpan<float> x, Span<float> destination)
    {
        Engines.Simd.SimdKernels.Sin(x, destination);
    }

    /// <summary>
    /// Computes cosine of each element using SIMD-optimized operations.
    /// </summary>
    public void Cos(ReadOnlySpan<float> x, Span<float> destination)
    {
        Engines.Simd.SimdKernels.Cos(x, destination);
    }

    /// <summary>
    /// Computes fused multiply-add: destination[i] = x[i] + y[i] * scalar.
    /// Uses SIMD with FMA intrinsics when available.
    /// </summary>
    public void MultiplyAdd(ReadOnlySpan<float> x, ReadOnlySpan<float> y, float scalar, Span<float> destination)
    {
        Engines.Simd.SimdKernels.ScalarMultiplyAdd(x, y, scalar, destination);
    }

    /// <summary>
    /// Converts float span to float span (identity operation - just copy).
    /// Uses SIMD-optimized copy for maximum performance.
    /// </summary>
    public void ToFloatSpan(ReadOnlySpan<float> source, Span<float> destination)
    {
        // Identity operation for float - just copy
        source.CopyTo(destination);
    }

    /// <summary>
    /// Converts float span to float span (identity operation - just copy).
    /// Uses SIMD-optimized copy for maximum performance.
    /// </summary>
    public void FromFloatSpan(ReadOnlySpan<float> source, Span<float> destination)
    {
        // Identity operation for float - just copy
        source.CopyTo(destination);
    }

    /// <summary>
    /// Converts float span to Half (FP16) span.
    /// SIMD-optimized on .NET 8+ using TensorPrimitives.ConvertToHalf.
    /// Critical for mixed-precision GPU operations (FP16 loads with FP32 accumulation).
    /// </summary>
    public void ToHalfSpan(ReadOnlySpan<float> source, Span<Half> destination)
    {
        if (source.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

#if NET8_0_OR_GREATER
        // SIMD-optimized conversion on .NET 8+
        System.Numerics.Tensors.TensorPrimitives.ConvertToHalf(source, destination);
#else
        // Sequential fallback for .NET Framework
        for (int i = 0; i < source.Length; i++)
            destination[i] = (Half)source[i];
#endif
    }

    /// <summary>
    /// Converts Half (FP16) span to float span.
    /// SIMD-optimized on .NET 8+ using TensorPrimitives.ConvertToSingle.
    /// Used when retrieving results from GPU operations using half precision.
    /// </summary>
    public void FromHalfSpan(ReadOnlySpan<Half> source, Span<float> destination)
    {
        if (source.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

#if NET8_0_OR_GREATER
        // SIMD-optimized conversion on .NET 8+
        System.Numerics.Tensors.TensorPrimitives.ConvertToSingle(source, destination);
#else
        // Sequential fallback for .NET Framework
        for (int i = 0; i < source.Length; i++)
            destination[i] = (float)source[i];
#endif
    }

    #region Vectorized Activation Functions

    /// <summary>
    /// Computes LeakyReLU element-wise using SIMD-optimized SimdKernels.
    /// Simple comparison operation - single-threaded SIMD saturates memory bandwidth.
    /// </summary>
    public void LeakyReLU(ReadOnlySpan<float> x, float alpha, Span<float> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        Engines.Simd.SimdKernels.LeakyReLU(x, alpha, destination);
    }

    /// <summary>
    /// Computes GELU (Gaussian Error Linear Unit) element-wise using SIMD-optimized SimdKernels with parallel processing for large arrays.
    /// Uses approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    /// </summary>
    public void GELU(ReadOnlySpan<float> x, Span<float> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        int length = x.Length;
        if (length >= ParallelThreshold && MaxDegreeOfParallelism > 1)
        {
            var pool = ArrayPool<float>.Shared;
            var xArray = pool.Rent(length);
            var destArray = pool.Rent(length);
            try
            {
                x.CopyTo(xArray);

                ParallelForChunks(length, MinChunkSize, (start, count) =>
                {
                    Engines.Simd.SimdKernels.GELU(
                        new ReadOnlySpan<float>(xArray, start, count),
                        new Span<float>(destArray, start, count));
                });

                new ReadOnlySpan<float>(destArray, 0, length).CopyTo(destination);
            }
            finally
            {
                pool.Return(xArray);
                pool.Return(destArray);
            }
            return;
        }

        Engines.Simd.SimdKernels.GELU(x, destination);
    }

    /// <summary>
    /// Computes Mish activation element-wise using SIMD-optimized SimdKernels with parallel processing for large arrays.
    /// x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    /// </summary>
    public void Mish(ReadOnlySpan<float> x, Span<float> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        int length = x.Length;
        if (length >= ParallelThreshold && MaxDegreeOfParallelism > 1)
        {
            var pool = ArrayPool<float>.Shared;
            var xArray = pool.Rent(length);
            var destArray = pool.Rent(length);
            try
            {
                x.CopyTo(xArray);

                ParallelForChunks(length, MinChunkSize, (start, count) =>
                {
                    Engines.Simd.SimdKernels.Mish(
                        new ReadOnlySpan<float>(xArray, start, count),
                        new Span<float>(destArray, start, count));
                });

                new ReadOnlySpan<float>(destArray, 0, length).CopyTo(destination);
            }
            finally
            {
                pool.Return(xArray);
                pool.Return(destArray);
            }
            return;
        }

        Engines.Simd.SimdKernels.Mish(x, destination);
    }

    /// <summary>
    /// Computes Swish/SiLU activation element-wise using SIMD-optimized SimdKernels with parallel processing for large arrays.
    /// x * sigmoid(x) = x / (1 + exp(-x))
    /// </summary>
    public void Swish(ReadOnlySpan<float> x, Span<float> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        int length = x.Length;
        if (length >= ParallelThreshold && MaxDegreeOfParallelism > 1)
        {
            var pool = ArrayPool<float>.Shared;
            var xArray = pool.Rent(length);
            var destArray = pool.Rent(length);
            try
            {
                x.CopyTo(xArray);

                ParallelForChunks(length, MinChunkSize, (start, count) =>
                {
                    Engines.Simd.SimdKernels.Swish(
                        new ReadOnlySpan<float>(xArray, start, count),
                        new Span<float>(destArray, start, count));
                });

                new ReadOnlySpan<float>(destArray, 0, length).CopyTo(destination);
            }
            finally
            {
                pool.Return(xArray);
                pool.Return(destArray);
            }
            return;
        }

        Engines.Simd.SimdKernels.Swish(x, destination);
    }

    /// <summary>
    /// Computes ELU (Exponential Linear Unit) element-wise using SIMD-optimized SimdKernels with parallel processing for large arrays.
    /// x if x > 0, alpha * (exp(x) - 1) otherwise
    /// </summary>
    public void ELU(ReadOnlySpan<float> x, float alpha, Span<float> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        int length = x.Length;
        if (length >= ParallelThreshold && MaxDegreeOfParallelism > 1)
        {
            var pool = ArrayPool<float>.Shared;
            var xArray = pool.Rent(length);
            var destArray = pool.Rent(length);
            try
            {
                x.CopyTo(xArray);

                ParallelForChunks(length, MinChunkSize, (start, count) =>
                {
                    Engines.Simd.SimdKernels.ELU(
                        new ReadOnlySpan<float>(xArray, start, count),
                        alpha,
                        new Span<float>(destArray, start, count));
                });

                new ReadOnlySpan<float>(destArray, 0, length).CopyTo(destination);
            }
            finally
            {
                pool.Return(xArray);
                pool.Return(destArray);
            }
            return;
        }

        Engines.Simd.SimdKernels.ELU(x, alpha, destination);
    }

    /// <summary>
    /// Computes ReLU (Rectified Linear Unit) element-wise using SIMD-optimized operations.
    /// max(0, x) - Memory-bound operation where single-threaded SIMD is optimal.
    /// Note: Parallelism adds copy overhead that hurts performance for simple ops like ReLU.
    /// </summary>
    public void ReLU(ReadOnlySpan<float> x, Span<float> destination)
    {
        if (x.Length != destination.Length)
            throw new ArgumentException("Spans must have the same length");

        Engines.Simd.SimdKernels.ReLU(x, destination);
    }

    #endregion

    #endregion
}
