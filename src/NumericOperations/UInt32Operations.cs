namespace AiDotNet.NumericOperations;

/// <summary>
/// Provides mathematical operations for the <see cref="uint"/> (UInt32) data type.
/// </summary>
/// <remarks>
/// <para>
/// This class implements the <see cref="INumericOperations{T}"/> interface for the <see cref="uint"/> type,
/// providing basic and advanced mathematical operations while handling the limitations of the unsigned integer data type.
/// Since uint values are limited to the range 0 to 4,294,967,295, operations that would result in values
/// outside this range will overflow and potentially produce unexpected results.
/// </para>
/// <para><b>For Beginners:</b> This class lets you perform math with unsigned integers (whole numbers between 0 and approximately 4.29 billion).
/// 
/// Think of it like a calculator that works specifically with positive whole numbers and zero. For example:
/// - You can add, subtract, multiply, and divide uint numbers
/// - You can compare values (is one number greater than another?)
/// - You can perform more advanced operations like square roots or exponents
/// 
/// However, be careful! If your calculations produce a number larger than 4,294,967,295 or a negative number,
/// the result will "wrap around" (overflow) and might give you an unexpected answer. This is like
/// a car odometer that rolls over to 0 after reaching its maximum value.
/// 
/// The main advantage of uint over other number types is that it can store large positive numbers
/// (up to about 4.29 billion) while using less memory than even larger number types like ulong.
/// </para>
/// </remarks>
public class UInt32Operations : INumericOperations<uint>
{
    /// <summary>
    /// Adds two uint values.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    /// <returns>The sum of <paramref name="a"/> and <paramref name="b"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method performs addition on two uint values. If the result exceeds the maximum value of a uint
    /// (4,294,967,295), an overflow will occur, wrapping the result around to start from zero again.
    /// </para>
    /// <para><b>For Beginners:</b> This method adds two numbers together.
    /// 
    /// For example:
    /// - Add(5, 3) returns 8
    /// - Add(10, 20) returns 30
    /// 
    /// Be careful with large numbers! If the result is too big for a uint, it will wrap around:
    /// - Add(4,294,967,290, 10) would mathematically be 4,294,967,300, but since that's too large,
    ///   it will return 4 (the result after "wrapping around" from zero again)
    /// </para>
    /// </remarks>
    public uint Add(uint a, uint b) => a + b;

    /// <summary>
    /// Subtracts the second value from the first.
    /// </summary>
    /// <param name="a">The value to subtract from.</param>
    /// <param name="b">The value to subtract.</param>
    /// <returns>The difference between <paramref name="a"/> and <paramref name="b"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method performs subtraction of two uint values. If the result would be negative (when b > a),
    /// an overflow will occur, wrapping the result around to a large positive number. This is because uint
    /// cannot represent negative values.
    /// </para>
    /// <para><b>For Beginners:</b> This method subtracts the second number from the first.
    /// 
    /// For example:
    /// - Subtract(10, 3) returns 7
    /// - Subtract(20, 5) returns 15
    /// 
    /// Be careful when the second number is larger than the first! Since a uint can't be negative:
    /// - Subtract(5, 10) will not return -5. Instead, it will return 4,294,967,291 (which is 4,294,967,296 - 5)
    /// 
    /// This happens because the result wraps around from the end of the range to the beginning.
    /// </para>
    /// </remarks>
    public uint Subtract(uint a, uint b) => a - b;

    /// <summary>
    /// Multiplies two uint values.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    /// <returns>The product of <paramref name="a"/> and <paramref name="b"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method performs multiplication of two uint values. The result of multiplying two uint values can
    /// easily exceed the range of a uint, causing overflow and potentially returning an unexpected value.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies two numbers together.
    /// 
    /// For example:
    /// - Multiply(4, 5) returns 20
    /// - Multiply(10, 3) returns 30
    /// 
    /// Multiplication can easily produce numbers that are too large for a uint:
    /// - Multiply(1,000,000, 5,000) would be 5,000,000,000, which is outside the uint range, 
    ///   so the result will wrap around and give you an incorrect answer (705,032,704)
    /// </para>
    /// </remarks>
    public uint Multiply(uint a, uint b) => a * b;

    /// <summary>
    /// Divides the first value by the second.
    /// </summary>
    /// <param name="a">The dividend (value to be divided).</param>
    /// <param name="b">The divisor (value to divide by).</param>
    /// <returns>The quotient of <paramref name="a"/> divided by <paramref name="b"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method performs integer division of two uint values. Because uint is an integer type, 
    /// the result will be truncated (rounded down). Division by zero will throw a DivideByZeroException.
    /// </para>
    /// <para><b>For Beginners:</b> This method divides the first number by the second.
    /// 
    /// For example:
    /// - Divide(10, 2) returns 5
    /// - Divide(7, 2) returns 3 (not 3.5, since uint values are whole numbers only)
    /// 
    /// Important notes:
    /// - The result is always rounded down to the nearest whole number
    /// - Dividing by zero will cause your program to crash with an error
    /// </para>
    /// </remarks>
    public uint Divide(uint a, uint b) => a / b;

    /// <summary>
    /// Negates a uint value.
    /// </summary>
    /// <param name="a">The value to negate.</param>
    /// <returns>The two's complement negation of <paramref name="a"/>.</returns>
    /// <remarks>
    /// <para>
    /// Since uint cannot represent negative values, this method performs a two's complement negation.
    /// For a value 'a', it returns (uint.MaxValue - a + 1), which is equivalent to (2^32 - a) when 
    /// represented in the full 32-bit range. This operation has the property that a + Negate(a) = 0 
    /// (after overflow).
    /// </para>
    /// <para><b>For Beginners:</b> This method attempts to find the "negative" of an unsigned number.
    /// 
    /// Since uint can only store positive numbers, true negation isn't possible. Instead, this method
    /// uses a technique called "two's complement" to find the value that, when added to the original number,
    /// gives zero in the uint range.
    /// 
    /// For example:
    /// - Negate(1) returns 4,294,967,295 (because 1 + 4,294,967,295 = 4,294,967,296, which overflows to 0 in uint)
    /// - Negate(1000) returns 4,294,966,296 (because 1000 + 4,294,966,296 = 4,294,967,296, which overflows to 0 in uint)
    /// 
    /// This operation is mostly used in specific bit manipulation contexts or when implementing
    /// certain algorithms that require a "wraparound" behavior.
    /// </para>
    /// </remarks>
    public uint Negate(uint a) => uint.MaxValue - a + 1;

    /// <summary>
    /// Gets the value zero as a uint.
    /// </summary>
    /// <value>The value 0 as a uint.</value>
    /// <remarks>
    /// <para>
    /// This property returns the value zero (0) as a uint. It is useful for operations that
    /// require a zero value, such as initializing variables or as a default value.
    /// </para>
    /// <para><b>For Beginners:</b> This property simply gives you the number zero (0) as a uint.
    /// 
    /// This is useful when you need a known zero value in your code, for example:
    /// - When starting a counter
    /// - When you need to initialize a value before calculating
    /// - As a default or fallback value
    /// </para>
    /// </remarks>
    public uint Zero => 0;

    /// <summary>
    /// Gets the value one as a uint.
    /// </summary>
    /// <value>The value 1 as a uint.</value>
    /// <remarks>
    /// <para>
    /// This property returns the value one (1) as a uint. It is useful for operations that
    /// require a unit value, such as incrementing a counter or as an identity element in multiplication.
    /// </para>
    /// <para><b>For Beginners:</b> This property simply gives you the number one (1) as a uint.
    /// 
    /// This is useful in many situations:
    /// - When incrementing a counter (adding 1)
    /// - In mathematical formulas that need the number 1
    /// - As a starting value for multiplication
    /// </para>
    /// </remarks>
    public uint One => 1;

    /// <summary>
    /// Calculates the square root of a uint value.
    /// </summary>
    /// <param name="value">The value to calculate the square root of.</param>
    /// <returns>The square root of <paramref name="value"/> as a uint.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the square root of the input value and converts the result to a uint.
    /// The calculation is performed using double-precision arithmetic and then cast to a uint, which means
    /// the result will be truncated to an integer value.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the square root of a number.
    /// 
    /// The square root of a number is another number that, when multiplied by itself, gives the original number.
    /// 
    /// For example:
    /// - Sqrt(4) returns 2 (because 2 � 2 = 4)
    /// - Sqrt(9) returns 3 (because 3 � 3 = 9)
    /// - Sqrt(10) returns 3 (because the true square root is approximately 3.16, but as a uint it's rounded down to 3)
    /// 
    /// Unlike with signed numbers, you don't need to worry about negative inputs since uint values are always positive.
    /// </para>
    /// </remarks>
    public uint Sqrt(uint value) => (uint)Math.Sqrt(value);

    /// <summary>
    /// Converts a double value to a uint.
    /// </summary>
    /// <param name="value">The double value to convert.</param>
    /// <returns>The double value converted to a uint.</returns>
    /// <remarks>
    /// <para>
    /// This method converts a double-precision floating-point value to a uint. The conversion truncates
    /// the fractional part of the double. Negative values will underflow to a large positive value, and values
    /// greater than 4,294,967,295 will overflow.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts a decimal number to a whole uint number.
    /// 
    /// When converting:
    /// - The decimal part is dropped (not rounded)
    /// - If the number is negative, you'll get an unexpected large positive number
    /// - If the number is too large for a uint, you'll get an unexpected smaller result
    /// 
    /// For example:
    /// - FromDouble(5.7) returns 5 (decimal part is simply dropped)
    /// - FromDouble(3.2) returns 3
    /// - FromDouble(5000000000.0) will return a value that doesn't make sense because 5 billion is too large for a uint
    /// - FromDouble(-5.0) will not return -5 (since uint can't store negative numbers), but instead a large positive number
    /// </para>
    /// </remarks>
    public uint FromDouble(double value) => (uint)value;

    /// <summary>
    /// Determines if the first value is greater than the second.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns><c>true</c> if <paramref name="a"/> is greater than <paramref name="b"/>; otherwise, <c>false</c>.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two uint values and returns true if the first value is greater than the second.
    /// The comparison uses the standard greater than operator for uint values.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is bigger than the second.
    /// 
    /// For example:
    /// - GreaterThan(10, 5) returns true (because 10 is greater than 5)
    /// - GreaterThan(3, 7) returns false (because 3 is not greater than 7)
    /// - GreaterThan(4, 4) returns false (because 4 is equal to 4, not greater than it)
    /// </para>
    /// </remarks>
    public bool GreaterThan(uint a, uint b) => a > b;

    /// <summary>
    /// Determines if the first value is less than the second.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns><c>true</c> if <paramref name="a"/> is less than <paramref name="b"/>; otherwise, <c>false</c>.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two uint values and returns true if the first value is less than the second.
    /// The comparison uses the standard less than operator for uint values.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is smaller than the second.
    /// 
    /// For example:
    /// - LessThan(5, 10) returns true (because 5 is less than 10)
    /// - LessThan(7, 3) returns false (because 7 is not less than 3)
    /// - LessThan(4, 4) returns false (because 4 is equal to 4, not less than it)
    /// </para>
    /// </remarks>
    public bool LessThan(uint a, uint b) => a < b;

    /// <summary>
    /// Calculates the absolute value of a uint.
    /// </summary>
    /// <param name="value">The value to calculate the absolute value for.</param>
    /// <returns>The input value unchanged.</returns>
    /// <remarks>
    /// <para>
    /// For uint values, which are already non-negative, this method simply returns the input value unchanged.
    /// The absolute value function is traditionally used to get the non-negative version of a number, but
    /// since uint values are always non-negative, no conversion is needed.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives you the positive version of a number.
    /// 
    /// The absolute value of a number is how far it is from zero, ignoring whether it's positive or negative.
    /// 
    /// For uint values, which are always positive (or zero), this method simply returns the same number:
    /// - Abs(5) returns 5
    /// - Abs(0) returns 0
    /// 
    /// This method exists mainly for consistency with other numeric types where absolute value is meaningful.
    /// </para>
    /// </remarks>
    public uint Abs(uint value) => value;

    /// <summary>
    /// Squares a uint value.
    /// </summary>
    /// <param name="value">The value to square.</param>
    /// <returns>The square of <paramref name="value"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the square of the input value (the value multiplied by itself).
    /// The result of squaring a uint value can easily exceed the range of a uint,
    /// causing overflow and potentially returning an unexpected value.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies a number by itself.
    /// 
    /// For example:
    /// - Square(4) returns 16 (because 4 � 4 = 16)
    /// - Square(10) returns 100 (because 10 � 10 = 100)
    /// 
    /// Be careful with larger numbers! Squaring even moderate values can easily exceed the uint range:
    /// - Square(100,000) would be 10,000,000,000, which is outside the uint range, so the result will be incorrect
    /// </para>
    /// </remarks>
    public uint Square(uint value) => Multiply(value, value);

    /// <summary>
    /// Calculates e raised to the specified power.
    /// </summary>
    /// <param name="value">The power to raise e to.</param>
    /// <returns>The value of e raised to the power of <paramref name="value"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the exponential function (e^value) for the input value, where e is Euler's number
    /// (approximately 2.71828). The calculation is performed using double-precision arithmetic, rounded to the
    /// nearest integer, and then clamped to the maximum uint value before casting to a uint. This prevents
    /// overflow for large input values, instead returning uint.MaxValue (4,294,967,295).
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates "e" raised to a power.
    /// 
    /// "e" is a special mathematical constant (approximately 2.71828) used in many calculations, especially
    /// those involving growth or decay.
    /// 
    /// For example:
    /// - Exp(1) returns 3 (because e^1 � 2.71828, rounded to 3 as a uint)
    /// - Exp(2) returns 7 (because e^2 � 7.38906, rounded to 7 as a uint)
    /// 
    /// For larger input values, the result grows very quickly:
    /// - Exp(10) returns 22,026 (because e^10 � 22,026.47)
    /// - Exp(30) or higher will return 4,294,967,295 (the maximum uint value) because the true result would be too large
    /// 
    /// This function is useful in calculations involving:
    /// - Compound interest
    /// - Population growth
    /// - Radioactive decay
    /// </para>
    /// </remarks>
    public uint Exp(uint value) => (uint)Math.Min(uint.MaxValue, Math.Round(Math.Exp(value)));

    /// <summary>
    /// Determines if two uint values are equal.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns><c>true</c> if <paramref name="a"/> is equal to <paramref name="b"/>; otherwise, <c>false</c>.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two uint values for equality. Two uint values are considered equal
    /// if they represent the same numeric value.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if two numbers are exactly the same.
    /// 
    /// For example:
    /// - Equals(5, 5) returns true (because both numbers are 5)
    /// - Equals(10, 15) returns false (because 10 and 15 are different numbers)
    /// </para>
    /// </remarks>
    public bool Equals(uint a, uint b) => a == b;

    /// <summary>
    /// Raises a value to the specified power.
    /// </summary>
    /// <param name="baseValue">The base value.</param>
    /// <param name="exponent">The exponent.</param>
    /// <returns>The base value raised to the specified power.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the base value raised to the power of the exponent. The calculation is
    /// performed using double-precision arithmetic and then cast to a uint, which may cause
    /// overflow for large results. Negative exponents will result in fractional values that,
    /// when cast to uint, will become 0.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies a number by itself a specified number of times.
    /// 
    /// For example:
    /// - Power(2, 3) returns 8 (because 2� = 2 � 2 � 2 = 8)
    /// - Power(3, 2) returns 9 (because 3� = 3 � 3 = 9)
    /// - Power(5, 0) returns 1 (any number raised to the power of 0 is 1)
    /// 
    /// Be careful with larger values! The result can quickly exceed the uint range:
    /// - Power(10, 9) would be 1,000,000,000, which is within the uint range
    /// - Power(10, 10) would be 10,000,000,000, which is outside the uint range, so the result will be incorrect
    /// 
    /// Fractional results are truncated to whole numbers:
    /// - Power(2, -1) would mathematically be 0.5, but as a uint it returns 0
    /// </para>
    /// </remarks>
    public uint Power(uint baseValue, uint exponent) => (uint)Math.Pow(baseValue, exponent);

    /// <summary>
    /// Calculates the natural logarithm (base e) of a value.
    /// </summary>
    /// <param name="value">The value to calculate the logarithm for.</param>
    /// <returns>The natural logarithm of <paramref name="value"/>.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the natural logarithm (ln) of the input value. The calculation is
    /// performed using double-precision arithmetic and then cast to a uint. The result is truncated
    /// to an integer, leading to loss of precision. If the input is 0, the result will be a mathematical error
    /// (negative infinity), which typically becomes 0 when cast to a uint.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the natural logarithm of a number.
    /// 
    /// The natural logarithm (ln) is the reverse of the exponential function. It tells you what power
    /// you need to raise "e" to in order to get your input value.
    /// 
    /// For example:
    /// - Log(1) returns 0 (because e^0 = 1)
    /// - Log(3) returns 1 (because e^1 � 2.71828, and when cast to a uint, the decimal part is dropped)
    /// - Log(10) returns 2 (because e^2.303 � 10, and when cast to a uint, the decimal part is dropped)
    /// 
    /// Important notes:
    /// - The logarithm of zero is not defined mathematically, so Log(0) will return 0
    /// - Logarithm results are usually decimals, but they'll be converted to whole numbers when stored as uints
    /// </para>
    /// </remarks>
    public uint Log(uint value) => (uint)Math.Log(value);

    /// <summary>
    /// Determines if the first value is greater than or equal to the second.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns><c>true</c> if <paramref name="a"/> is greater than or equal to <paramref name="b"/>; otherwise, <c>false</c>.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two uint values and returns true if the first value is greater than or equal to the second.
    /// The comparison uses the standard greater than or equal to operator for uint values.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is bigger than or the same as the second.
    /// 
    /// For example:
    /// - GreaterThanOrEquals(10, 5) returns true (because 10 is greater than 5)
    /// - GreaterThanOrEquals(7, 7) returns true (because 7 is equal to 7)
    /// - GreaterThanOrEquals(3, 8) returns false (because 3 is less than 8)
    /// </para>
    /// </remarks>
    public bool GreaterThanOrEquals(uint a, uint b) => a >= b;

    /// <summary>
    /// Determines if the first value is less than or equal to the second.
    /// </summary>
    /// <param name="a">The first value to compare.</param>
    /// <param name="b">The second value to compare.</param>
    /// <returns><c>true</c> if <paramref name="a"/> is less than or equal to <paramref name="b"/>; otherwise, <c>false</c>.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two uint values and returns true if the first value is less than or equal to the second.
    /// The comparison uses the standard less than or equal to operator for uint values.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first number is smaller than or the same as the second.
    /// 
    /// For example:
    /// - LessThanOrEquals(5, 10) returns true (because 5 is less than 10)
    /// - LessThanOrEquals(7, 7) returns true (because 7 is equal to 7)
    /// - LessThanOrEquals(9, 4) returns false (because 9 is greater than 4)
    /// </para>
    /// </remarks>
    public bool LessThanOrEquals(uint a, uint b) => a <= b;

    /// <summary>
    /// Converts a uint value to a 32-bit integer.
    /// </summary>
    /// <param name="value">The uint value to convert.</param>
    /// <returns>The uint value as a 32-bit integer.</returns>
    /// <remarks>
    /// <para>
    /// This method converts a uint (32-bit unsigned) value to an int (32-bit signed) value. The conversion may fail
    /// if the uint value is greater than int.MaxValue (2,147,483,647), resulting in overflow. Values larger than
    /// int.MaxValue will be interpreted as negative values in the int type.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts a uint number to a regular integer (int).
    /// 
    /// A uint can store numbers from 0 to 4,294,967,295.
    /// An int can store numbers from -2,147,483,648 to 2,147,483,647.
    /// 
    /// This conversion is not always safe:
    /// - If the uint value is less than or equal to 2,147,483,647, it converts correctly
    /// - If the uint value is greater than 2,147,483,647, it will "wrap around" to a negative number
    /// 
    /// For example:
    /// - ToInt32(5) returns 5 as an int
    /// - ToInt32(1000) returns 1000 as an int
    /// - ToInt32(3,000,000,000) doesn't return 3,000,000,000 because that's too large for an int;
    ///   instead, it returns a negative number (-1,294,967,296)
    /// </para>
    /// </remarks>
    public int ToInt32(uint value) => (int)value;

    /// <summary>
    /// Rounds a uint value.
    /// </summary>
    /// <param name="value">The value to round.</param>
    /// <returns>The rounded value.</returns>
    /// <remarks>
    /// <para>
    /// For uint values, which are already integers, this method simply returns the value unchanged.
    /// Rounding only applies to floating-point values that have fractional parts.
    /// </para>
    /// <para><b>For Beginners:</b> This method rounds a number to the nearest whole number.
    /// 
    /// Since a uint is already a whole number, this method simply returns the same number without any change.
    /// 
    /// For example:
    /// - Round(5) returns 5
    /// - Round(10) returns 10
    /// 
    /// This method exists mainly for consistency with other numeric types like float or double,
    /// where rounding would actually change the value.
    /// </para>
    /// </remarks>
    public uint Round(uint value) => value;

    /// <summary>
    /// Gets the minimum value that can be represented by a uint.
    /// </summary>
    /// <value>The minimum value of a uint, which is 0.</value>
    /// <remarks>
    /// <para>
    /// This property returns the smallest possible value that can be represented by the uint data type,
    /// which is 0. Unlike signed types, uint cannot represent negative values.
    /// </para>
    /// <para><b>For Beginners:</b> This property gives you the smallest possible number that a uint can hold.
    /// 
    /// For uint values, the minimum value is always 0, because uint can only store positive whole numbers
    /// (and zero).
    /// 
    /// This is useful when you need to:
    /// - Check if a value is valid for a uint
    /// - Initialize a variable to the smallest possible value
    /// - Set boundaries for valid input values
    /// </para>
    /// </remarks>
    public uint MinValue => uint.MinValue;

    /// <summary>
    /// Gets the maximum value that can be represented by a uint.
    /// </summary>
    /// <value>The maximum value of a uint, which is 4,294,967,295.</value>
    /// <remarks>
    /// <para>
    /// This property returns the largest possible value that can be represented by the uint data type,
    /// which is 4,294,967,295. Attempting to store a value greater than this in a uint will result in overflow.
    /// </para>
    /// <para><b>For Beginners:</b> This property gives you the largest possible number that a uint can hold.
    /// 
    /// For uint values, the maximum value is 4,294,967,295.
    /// If you try to create a uint with a larger value (like 5,000,000,000), the number will wrap around
    /// and give you an incorrect result.
    /// 
    /// This is useful when you need to:
    /// - Check if a value is too large to be stored as a uint
    /// - Initialize a variable to the largest possible value before comparing
    /// - Set boundaries for valid input values
    /// </para>
    /// </remarks>
    public uint MaxValue => uint.MaxValue;

    /// <summary>
    /// Determines if a uint value is NaN (Not a Number).
    /// </summary>
    /// <param name="value">The value to check.</param>
    /// <returns>Always <c>false</c> for uint values.</returns>
    /// <remarks>
    /// <para>
    /// This method always returns false because the uint data type can only represent integers,
    /// and the concept of NaN (Not a Number) only applies to floating-point types like float and double.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if a number is "Not a Number" (NaN).
    /// 
    /// For uint values, the result is always false because a uint can only contain valid whole numbers.
    /// The concept of "Not a Number" applies only to floating-point types like float or double,
    /// which can represent special values like the result of divide-by-zero.
    /// 
    /// This method exists mainly for consistency with other numeric types where IsNaN is meaningful.
    /// </para>
    /// </remarks>
    public bool IsNaN(uint value) => false;

    /// <summary>
    /// Determines if a uint value is infinity.
    /// </summary>
    /// <param name="value">The value to check.</param>
    /// <returns>Always <c>false</c> for uint values.</returns>
    /// <remarks>
    /// <para>
    /// This method always returns false because the uint data type can only represent integers,
    /// and the concept of infinity only applies to floating-point types like float and double.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if a number is "infinity".
    /// 
    /// For uint values, the result is always false because a uint can only contain finite whole numbers.
    /// The concept of "infinity" applies only to floating-point types like float or double,
    /// which can represent special values like the result of divide-by-zero.
    /// 
    /// This method exists mainly for consistency with other numeric types where IsInfinity is meaningful.
    /// </para>
    /// </remarks>
    public bool IsInfinity(uint value) => false;

    /// <summary>
    /// Returns the sign of a uint value as 0 or 1.
    /// </summary>
    /// <param name="value">The value to determine the sign of.</param>
    /// <returns>
    /// 0 if <paramref name="value"/> is zero;
    /// 1 if <paramref name="value"/> is positive.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method returns a value indicating the sign of the input value. Since uint can only
    /// represent non-negative values, the result will always be either 0 (for zero) or 1 (for positive values).
    /// This is different from signed numeric types where the result could also be -1 for negative values.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells you if a number is positive or zero.
    /// 
    /// It returns:
    /// - 0 if the number is exactly zero
    /// - 1 if the number is positive (greater than zero)
    /// 
    /// Since uint can only store values that are zero or positive, you'll never get a -1 result
    /// (which would represent a negative number in other numeric types).
    /// 
    /// For example:
    /// - SignOrZero(0) returns 0
    /// - SignOrZero(42) returns 1
    /// - SignOrZero(4294967295) returns 1
    /// 
    /// The suffix "u" on the literals (0u, 1u) simply indicates that these are unsigned integer values.
    /// </para>
    /// </remarks>
    public uint SignOrZero(uint value) => value == 0 ? 0u : 1u;
}