using System;

using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.NumericOperations;
/// <summary>
/// Provides mathematical operations for complex numbers.
/// </summary>
/// <remarks>
/// <para>
/// This class implements the INumericOperations interface for the Complex<T> type, enabling
/// arithmetic operations, comparisons, and mathematical functions on complex numbers.
/// It uses an underlying numeric operations implementation for the generic type T to perform
/// calculations on the real and imaginary components.
/// </para>
/// <para><b>For Beginners:</b> This class handles math with complex numbers.
///
/// Complex numbers have two parts:
/// - A real part (like regular numbers)
/// - An imaginary part (multiplied by i, where iÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â² = -1)
///
/// For example, 3 + 4i is a complex number where:
/// - 3 is the real part
/// - 4 is the imaginary part
///
/// Complex numbers are useful for:
/// - Advanced mathematical calculations
/// - Engineering problems (especially in electrical engineering)
/// - Physics and quantum mechanics
/// - Signal processing and control systems
///
/// This class lets you perform operations like addition, multiplication, and
/// more advanced functions on complex numbers.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for the real and imaginary parts, typically float or double.</typeparam>
public class ComplexOperations<T> : INumericOperations<Complex<T>>
{
    /// <summary>
    /// Provides operations for the underlying numeric type T.
    /// </summary>
    /// <remarks>
    /// This field holds the numeric operations for type T, which are used to perform
    /// calculations on the real and imaginary components of complex numbers.
    /// </remarks>
    private readonly INumericOperations<T> _ops;

    /// <summary>
    /// Initializes a new instance of the <see cref="ComplexOperations{T}"/> class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The constructor automatically obtains the appropriate numeric operations for the specified type T
    /// (such as float, double, or decimal) using the MathHelper utility.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new ComplexOperations object.
    /// 
    /// When you create this object:
    /// - It automatically sets up the correct math operations for the number type T
    /// - T is typically float or double (decimal number types)
    /// - No additional parameters are needed
    /// 
    /// For example: var complexOps = new ComplexOperations&lt;double&gt;();
    /// </para>
    /// </remarks>
    public ComplexOperations()
    {
        _ops = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Adds two complex numbers.
    /// </summary>
    /// <param name="a">The first complex number.</param>
    /// <param name="b">The second complex number.</param>
    /// <returns>The sum of the two complex numbers.</returns>
    /// <remarks>
    /// <para>
    /// This method adds two complex numbers by adding their real and imaginary parts separately.
    /// The result is a new complex number whose real part is the sum of the real parts and whose
    /// imaginary part is the sum of the imaginary parts.
    /// </para>
    /// <para><b>For Beginners:</b> This method adds two complex numbers together.
    /// 
    /// When adding complex numbers:
    /// - The real parts are added together
    /// - The imaginary parts are added together
    /// 
    /// For example:
    /// - (3 + 4i) + (2 + 5i) = 5 + 9i
    /// 
    /// This follows the same pattern as adding vectors.
    /// </para>
    /// </remarks>
    public Complex<T> Add(Complex<T> a, Complex<T> b) => a + b;

    /// <summary>
    /// Subtracts the second complex number from the first.
    /// </summary>
    /// <param name="a">The complex number to subtract from.</param>
    /// <param name="b">The complex number to subtract.</param>
    /// <returns>The difference between the two complex numbers.</returns>
    /// <remarks>
    /// <para>
    /// This method subtracts one complex number from another by subtracting their real and imaginary parts separately.
    /// The result is a new complex number whose real part is the difference of the real parts and whose
    /// imaginary part is the difference of the imaginary parts.
    /// </para>
    /// <para><b>For Beginners:</b> This method subtracts one complex number from another.
    /// 
    /// When subtracting complex numbers:
    /// - The real parts are subtracted
    /// - The imaginary parts are subtracted
    /// 
    /// For example:
    /// - (5 + 8i) - (2 + 3i) = 3 + 5i
    /// 
    /// This follows the same pattern as subtracting vectors.
    /// </para>
    /// </remarks>
    public Complex<T> Subtract(Complex<T> a, Complex<T> b) => a - b;

    /// <summary>
    /// Multiplies two complex numbers.
    /// </summary>
    /// <param name="a">The first complex number.</param>
    /// <param name="b">The second complex number.</param>
    /// <returns>The product of the two complex numbers.</returns>
    /// <remarks>
    /// <para>
    /// This method multiplies two complex numbers using the formula (a + bi)(c + di) = (ac - bd) + (ad + bc)i.
    /// The result is a new complex number.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies two complex numbers together.
    /// 
    /// Multiplying complex numbers is different from adding them:
    /// - It's not just multiplying the parts separately
    /// - It follows a specific formula: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    /// 
    /// For example:
    /// - (3 + 2i) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  (1 + 4i) = (3 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 1 - 2 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 4) + (3 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 4 + 2 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 1)i = -5 + 14i
    /// 
    /// This is because iÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â² = -1, which makes complex multiplication different from
    /// regular multiplication.
    /// </para>
    /// </remarks>
    public Complex<T> Multiply(Complex<T> a, Complex<T> b) => a * b;

    /// <summary>
    /// Divides the first complex number by the second.
    /// </summary>
    /// <param name="a">The complex number to divide (dividend).</param>
    /// <param name="b">The complex number to divide by (divisor).</param>
    /// <returns>The quotient of the division.</returns>
    /// <remarks>
    /// <para>
    /// This method divides one complex number by another. The division is implemented by multiplying
    /// the numerator by the complex conjugate of the denominator, then dividing by the square of the
    /// magnitude of the denominator.
    /// </para>
    /// <para><b>For Beginners:</b> This method divides one complex number by another.
    /// 
    /// Dividing complex numbers involves a special technique:
    /// 1. Multiply both top and bottom by the conjugate of the denominator
    /// 2. This makes the denominator a real number
    /// 3. Then divide each part of the numerator by this real number
    /// 
    /// For example, to calculate (3 + 2i) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  (1 + i):
    /// 1. Multiply top and bottom by (1 - i), the conjugate of (1 + i)
    /// 2. (3 + 2i)(1 - i) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  (1 + i)(1 - i)
    /// 3. (3 - 3i + 2i - 2iÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â²) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  (1ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â² - iÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â²)
    /// 4. (3 - 3i + 2i + 2) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  (1 + 1)
    /// 5. (5 - i) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  2
    /// 6. 2.5 - 0.5i
    /// 
    /// Complex division is one of the more challenging operations with complex numbers.
    /// </para>
    /// </remarks>
    public Complex<T> Divide(Complex<T> a, Complex<T> b) => a / b;

    /// <summary>
    /// Negates a complex number.
    /// </summary>
    /// <param name="a">The complex number to negate.</param>
    /// <returns>The negated complex number.</returns>
    /// <remarks>
    /// <para>
    /// This method negates a complex number by negating both its real and imaginary parts.
    /// The result is a new complex number with the opposite sign for both components.
    /// </para>
    /// <para><b>For Beginners:</b> This method reverses the sign of a complex number.
    /// 
    /// When negating a complex number:
    /// - The sign of the real part is reversed
    /// - The sign of the imaginary part is reversed
    /// 
    /// For example:
    /// - The negation of (3 + 4i) is (-3 - 4i)
    /// 
    /// This is the same as multiplying the number by -1.
    /// </para>
    /// </remarks>
    public Complex<T> Negate(Complex<T> a) => new(_ops.Negate(a.Real), _ops.Negate(a.Imaginary));

    /// <summary>
    /// Gets the complex representation of zero.
    /// </summary>
    /// <value>The complex number 0 + 0i.</value>
    /// <remarks>
    /// <para>
    /// This property returns the complex number zero, which has both real and imaginary parts set to zero.
    /// It is often used as a neutral element for addition.
    /// </para>
    /// <para><b>For Beginners:</b> This property provides the complex number zero.
    /// 
    /// The complex zero is 0 + 0i, where:
    /// - The real part is 0
    /// - The imaginary part is 0
    /// 
    /// This works like the number 0 in regular arithmetic:
    /// - Adding zero to any complex number gives the same number
    /// - It's often used as a starting point or default value
    /// </para>
    /// </remarks>
    public Complex<T> Zero => new(_ops.Zero, _ops.Zero);

    /// <summary>
    /// Gets the complex representation of one.
    /// </summary>
    /// <value>The complex number 1 + 0i.</value>
    /// <remarks>
    /// <para>
    /// This property returns the complex number one, which has a real part of one and an imaginary part of zero.
    /// It is often used as a neutral element for multiplication.
    /// </para>
    /// <para><b>For Beginners:</b> This property provides the complex number one.
    /// 
    /// The complex one is 1 + 0i, where:
    /// - The real part is 1
    /// - The imaginary part is 0
    /// 
    /// This works like the number 1 in regular arithmetic:
    /// - Multiplying any complex number by one gives the same number
    /// - It's used as a unit value in many calculations
    /// </para>
    /// </remarks>
    public Complex<T> One => new(_ops.One, _ops.Zero);

    /// <summary>
    /// Calculates the square root of a complex number.
    /// </summary>
    /// <param name="value">The complex number to calculate the square root of.</param>
    /// <returns>The square root of the complex number.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the square root of a complex number using polar form.
    /// It computes the square root of the magnitude and halves the phase angle,
    /// then converts back to rectangular form.
    /// </para>
    /// <para><b>For Beginners:</b> This method finds the square root of a complex number.
    /// 
    /// Finding the square root of a complex number involves:
    /// 1. Converting to polar form (magnitude and angle)
    /// 2. Taking the square root of the magnitude
    /// 3. Dividing the angle by 2
    /// 4. Converting back to the standard form
    /// 
    /// For example, the square root of -4 (which is 0 - 4i in complex form):
    /// - Has a magnitude of 4 and an angle of -90 degrees
    /// - The square root has magnitude v4 = 2 and angle -90/2 = -45 degrees
    /// - Converting back gives 2 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  (cos(-45ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â²) + i ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  sin(-45ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â²)) = v2 - v2i
    /// 
    /// This is one of the key advantages of complex numbers - they allow us to take
    /// square roots of negative numbers.
    /// </para>
    /// </remarks>
    public Complex<T> Sqrt(Complex<T> value)
    {
        var magnitude = _ops.Sqrt(_ops.Add(_ops.Square(value.Real), _ops.Square(value.Imaginary)));
        var r = _ops.Sqrt(magnitude);
        var theta = _ops.Divide(value.Phase, _ops.FromDouble(2));
        return new Complex<T>(
            _ops.Multiply(r, _ops.FromDouble(Math.Cos(Convert.ToDouble(theta)))),
            _ops.Multiply(r, _ops.FromDouble(Math.Sin(Convert.ToDouble(theta))))
        );
    }

    /// <summary>
    /// Converts a double value to a complex number.
    /// </summary>
    /// <param name="value">The double value to convert.</param>
    /// <returns>A complex number with the specified real part and an imaginary part of zero.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a complex number from a double value. The resulting complex number
    /// has a real part equal to the double value converted to type T, and an imaginary part of zero.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts a regular number to a complex number.
    /// 
    /// When converting a regular number to complex:
    /// - The regular number becomes the real part
    /// - The imaginary part is set to zero
    /// 
    /// For example:
    /// - The number 5 becomes the complex number 5 + 0i
    /// 
    /// This allows regular numbers to be used in calculations involving complex numbers.
    /// </para>
    /// </remarks>
    public Complex<T> FromDouble(double value) => new(_ops.FromDouble(value), _ops.Zero);

    /// <summary>
    /// Determines whether the first complex number has a greater magnitude than the second.
    /// </summary>
    /// <param name="a">The first complex number to compare.</param>
    /// <param name="b">The second complex number to compare.</param>
    /// <returns>true if the magnitude of the first complex number is greater than the magnitude of the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two complex numbers based on their magnitudes (absolute values).
    /// It returns true if the magnitude of the first complex number is greater than the magnitude of the second.
    /// </para>
    /// <para><b>For Beginners:</b> This method compares the sizes of two complex numbers.
    /// 
    /// Since complex numbers have two components, comparing them directly isn't straightforward.
    /// Instead, we compare their magnitudes (distances from zero).
    /// 
    /// The magnitude of a complex number a + bi is ÃƒÆ’Ã‚Â¢Ãƒâ€¹Ã¢â‚¬Â Ãƒâ€¦Ã‚Â¡(aÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â² + bÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â²).
    /// 
    /// For example:
    /// - The magnitude of 3 + 4i is ÃƒÆ’Ã‚Â¢Ãƒâ€¹Ã¢â‚¬Â Ãƒâ€¦Ã‚Â¡(3ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â² + 4ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â²) = v25 = 5
    /// - The magnitude of 1 + 2i is ÃƒÆ’Ã‚Â¢Ãƒâ€¹Ã¢â‚¬Â Ãƒâ€¦Ã‚Â¡(1ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â² + 2ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â²) = v5 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 2.24
    /// - So 3 + 4i is greater than 1 + 2i in terms of magnitude
    /// 
    /// This is similar to comparing the lengths of vectors.
    /// </para>
    /// </remarks>
    public bool GreaterThan(Complex<T> a, Complex<T> b) => _ops.GreaterThan(a.Magnitude, b.Magnitude);

    /// <summary>
    /// Determines whether the first complex number has a smaller magnitude than the second.
    /// </summary>
    /// <param name="a">The first complex number to compare.</param>
    /// <param name="b">The second complex number to compare.</param>
    /// <returns>true if the magnitude of the first complex number is less than the magnitude of the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two complex numbers based on their magnitudes (absolute values).
    /// It returns true if the magnitude of the first complex number is less than the magnitude of the second.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first complex number is smaller than the second.
    /// 
    /// Like with GreaterThan, this compares the magnitudes (distances from zero) of the complex numbers.
    /// 
    /// For example:
    /// - 1 + i has magnitude v2 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 1.41
    /// - 2 + 2i has magnitude v8 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 2.83
    /// - So 1 + i is less than 2 + 2i
    /// 
    /// This comparison ignores the direction and only considers the size of the complex numbers.
    /// </para>
    /// </remarks>
    public bool LessThan(Complex<T> a, Complex<T> b) => _ops.LessThan(a.Magnitude, b.Magnitude);

    /// <summary>
    /// Returns the absolute value (magnitude) of a complex number.
    /// </summary>
    /// <param name="value">The complex number.</param>
    /// <returns>A complex number with a real part equal to the magnitude of the input and an imaginary part of zero.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the absolute value (or magnitude) of a complex number.
    /// The result is a complex number with the magnitude as its real part and zero as its imaginary part.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the size of a complex number.
    /// 
    /// The absolute value (or magnitude) of a complex number a + bi is ÃƒÆ’Ã‚Â¢Ãƒâ€¹Ã¢â‚¬Â Ãƒâ€¦Ã‚Â¡(aÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â² + bÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â²).
    /// 
    /// For example:
    /// - The absolute value of 3 + 4i is ÃƒÆ’Ã‚Â¢Ãƒâ€¹Ã¢â‚¬Â Ãƒâ€¦Ã‚Â¡(3ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â² + 4ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â²) = v25 = 5
    /// - So Abs(3 + 4i) returns the complex number 5 + 0i
    /// 
    /// The magnitude represents the distance from the origin to the complex number
    /// when plotted on the complex plane.
    /// </para>
    /// </remarks>
    public Complex<T> Abs(Complex<T> value) => new(value.Magnitude, _ops.Zero);

    /// <summary>
    /// Squares a complex number.
    /// </summary>
    /// <param name="value">The complex number to square.</param>
    /// <returns>The square of the complex number.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the square of a complex number using the formula (a + bi)ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â² = (aÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â² - bÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â²) + 2abi.
    /// It calculates the real and imaginary parts separately and constructs a new complex number.
    /// </para>
    /// <para><b>For Beginners:</b> This method multiplies a complex number by itself.
    /// 
    /// When squaring a complex number (a + bi):
    /// - The real part of the result is aÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â² - bÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â²
    /// - The imaginary part is 2ab
    /// 
    /// For example:
    /// - (3 + 2i)ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â² = (3ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â² - 2ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â²) + 2 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 3ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â2i = (9 - 4) + 12i = 5 + 12i
    /// 
    /// This formula comes from applying the complex multiplication rule: (a + bi)(a + bi)
    /// </para>
    /// </remarks>
    public Complex<T> Square(Complex<T> value)
    {
        var a = value.Real;
        var b = value.Imaginary;
        return new Complex<T>(
            _ops.Subtract(_ops.Square(a), _ops.Square(b)),
            _ops.Multiply(_ops.FromDouble(2), _ops.Multiply(a, b))
        );
    }

    /// <summary>
    /// Calculates e raised to the power of a complex number.
    /// </summary>
    /// <param name="value">The complex exponent.</param>
    /// <returns>e raised to the power of the complex number.</returns>
    /// <remarks>
    /// <para>
    /// This method computes e^z for a complex number z using Euler's formula:
    /// e^(a + bi) = e^a * (cos(b) + i * sin(b))
    /// It calculates the components separately and constructs a new complex number.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates e raised to a complex power.
    /// 
    /// The constant e (ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â 2.718) is an important mathematical constant.
    /// Calculating e raised to a complex power follows Euler's formula:
    /// 
    /// e^(a + bi) = e^a ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  (cos(b) + i ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  sin(b))
    /// 
    /// For example:
    /// - e^(0 + pi) = e^0 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  (cos(p) + i ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  sin(p)) = 1 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  (-1 + 0i) = -1
    /// - This shows the famous equation: e^(pi) = -1
    /// 
    /// This function is fundamental in many areas of mathematics and engineering.
    /// </para>
    /// </remarks>
    public Complex<T> Exp(Complex<T> value)
    {
        var expReal = _ops.Exp(value.Real);
        return new Complex<T>(
            _ops.Multiply(expReal, _ops.FromDouble(Math.Cos(Convert.ToDouble(value.Imaginary)))),
            _ops.Multiply(expReal, _ops.FromDouble(Math.Sin(Convert.ToDouble(value.Imaginary))))
        );
    }

    /// <summary>
    /// Determines whether two complex numbers are equal.
    /// </summary>
    /// <param name="a">The first complex number to compare.</param>
    /// <param name="b">The second complex number to compare.</param>
    /// <returns>true if the complex numbers are equal; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two complex numbers for equality. Two complex numbers are equal
    /// if both their real and imaginary parts are equal.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if two complex numbers are exactly the same.
    /// 
    /// Two complex numbers are equal only if:
    /// - Their real parts are equal, AND
    /// - Their imaginary parts are equal
    /// 
    /// For example:
    /// - 3 + 4i equals 3 + 4i
    /// - 3 + 4i does not equal 3 + 5i
    /// - 3 + 4i does not equal 4 + 4i
    /// 
    /// This is stricter than comparing magnitudes - the numbers must match exactly.
    /// </para>
    /// </remarks>
    public bool Equals(Complex<T> a, Complex<T> b) => a == b;

    /// <summary>
    /// Raises a complex number to a complex power.
    /// </summary>
    /// <param name="baseValue">The complex base value.</param>
    /// <param name="exponent">The complex exponent.</param>
    /// <returns>The base value raised to the exponent.</returns>
    /// <remarks>
    /// <para>
    /// This method computes a complex number raised to a complex power. It uses the formula
    /// z^w = e^(w * ln(z)), converting the operation to exponential and logarithm operations.
    /// A special case is handled when both the base and exponent are zero, returning 1.
    /// </para>
    /// <para><b>For Beginners:</b> This method raises a complex number to a complex power.
    /// 
    /// For real numbers, 2ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â² means 2 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 2. For complex numbers, it's more complicated.
    /// 
    /// To calculate a complex number raised to a complex power:
    /// 1. Take the natural logarithm (ln) of the base
    /// 2. Multiply by the exponent
    /// 3. Raise e to that product
    /// 
    /// For example:
    /// - To calculate (2 + i)^(3 + 2i), we compute e^((3 + 2i) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  ln(2 + i))
    /// 
    /// A special case: if both base and exponent are zero, the result is 1.
    /// 
    /// This is one of the most complex operations you can do with complex numbers.
    /// </para>
    /// </remarks>
    public Complex<T> Power(Complex<T> baseValue, Complex<T> exponent)
    {
        if (baseValue == Zero && exponent == Zero)
            return One;
        return Exp(Multiply(Log(baseValue), exponent));
    }

    /// <summary>
    /// Calculates the natural logarithm of a complex number.
    /// </summary>
    /// <param name="value">The complex number.</param>
    /// <returns>The natural logarithm of the complex number.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the natural logarithm of a complex number using the formula
    /// ln(z) = ln|z| + i*arg(z), where |z| is the magnitude and arg(z) is the phase angle.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the natural logarithm of a complex number.
    /// 
    /// The natural logarithm of a complex number has:
    /// - A real part equal to the logarithm of its magnitude
    /// - An imaginary part equal to its phase angle
    /// 
    /// For example:
    /// - The natural logarithm of 1 + i has:
    ///   - Real part = ln(v2) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  0.347
    ///   - Imaginary part = p/4 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 0.785
    ///   - So ln(1 + i) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€¹Ã¢â‚¬Â  0.347 + 0.785i
    /// 
    /// One interesting result: ln(-1) = pi
    /// 
    /// This function is the inverse of the Exp function.
    /// </para>
    /// </remarks>
    public Complex<T> Log(Complex<T> value)
    {
        return new Complex<T>(_ops.Log(value.Magnitude), value.Phase);
    }

    /// <summary>
    /// Determines whether the first complex number has a magnitude greater than or equal to the second.
    /// </summary>
    /// <param name="a">The first complex number to compare.</param>
    /// <param name="b">The second complex number to compare.</param>
    /// <returns>true if the magnitude of the first complex number is greater than or equal to the magnitude of the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two complex numbers based on their magnitudes (absolute values).
    /// It returns true if the magnitude of the first complex number is greater than or equal to the magnitude of the second.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first complex number is larger than or equal to the second.
    /// 
    /// Like other comparison methods, this compares the magnitudes of the complex numbers.
    /// 
    /// For example:
    /// - 3 + 4i has magnitude 5
    /// - 5 + 0i has magnitude 5
    /// - So 3 + 4i >= 5 + 0i returns true because their magnitudes are equal
    /// 
    /// This is useful for comparing the size of complex numbers in algorithms.
    /// </para>
    /// </remarks>
    public bool GreaterThanOrEquals(Complex<T> a, Complex<T> b)
    {
        return _ops.GreaterThanOrEquals(a.Magnitude, b.Magnitude);
    }

    /// <summary>
    /// Determines whether the first complex number has a magnitude less than or equal to the second.
    /// </summary>
    /// <param name="a">The first complex number to compare.</param>
    /// <param name="b">The second complex number to compare.</param>
    /// <returns>true if the magnitude of the first complex number is less than or equal to the magnitude of the second; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method compares two complex numbers based on their magnitudes (absolute values).
    /// It returns true if the magnitude of the first complex number is less than or equal to the magnitude of the second.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the first complex number is smaller than or equal to the second.
    /// 
    /// This comparison is based on the magnitudes (distances from zero) of the complex numbers.
    /// 
    /// For example:
    /// - 3 + 0i has magnitude 3
    /// - 0 + 3i also has magnitude 3
    /// - So 3 + 0i <= 0 + 3i returns true because their magnitudes are equal
    /// 
    /// Even though these numbers point in different directions on the complex plane,
    /// they're considered equal in size.
    /// </para>
    /// </remarks>
    public bool LessThanOrEquals(Complex<T> a, Complex<T> b)
    {
        return _ops.LessThanOrEquals(a.Magnitude, b.Magnitude);
    }

    /// <summary>
    /// Rounds the real and imaginary parts of a complex number to the nearest integers.
    /// </summary>
    /// <param name="value">The complex number to round.</param>
    /// <returns>A new complex number with rounded components.</returns>
    /// <remarks>
    /// <para>
    /// This method rounds both the real and imaginary parts of a complex number to the nearest integers.
    /// It creates a new complex number with the rounded components.
    /// </para>
    /// <para><b>For Beginners:</b> This method rounds both parts of a complex number.
    /// 
    /// When rounding a complex number:
    /// - The real part is rounded to the nearest whole number
    /// - The imaginary part is rounded to the nearest whole number
    /// 
    /// For example:
    /// - Rounding 3.7 + 2.2i gives 4 + 2i
    /// 
    /// This is useful when you need to work with whole number approximations
    /// of complex values.
    /// </para>
    /// </remarks>
    public Complex<T> Round(Complex<T> value) => new(_ops.Round(value.Real), _ops.Round(value.Imaginary));

    /// <summary>
    /// Gets the minimum value that can be represented using complex numbers with the underlying type T.
    /// </summary>
    /// <value>A complex number with both real and imaginary parts set to the minimum value of type T.</value>
    /// <remarks>
    /// <para>
    /// This property returns a complex number with both real and imaginary parts set to the minimum value
    /// that can be represented by the underlying type T. This is useful for algorithms that need to work
    /// with the full range of possible complex values.
    /// </para>
    /// <para><b>For Beginners:</b> This property gives you the smallest possible complex number.
    /// 
    /// For complex numbers, "minimum" refers to having both parts at their minimum values.
    /// 
    /// For example, if T is double:
    /// - The minimum value would have both real and imaginary parts equal to double.MinValue
    /// - This represents the bottom-left corner of the representable complex plane
    /// 
    /// This is primarily used in algorithms that need to track the smallest possible value.
    /// </para>
    /// </remarks>
    public Complex<T> MinValue => new(_ops.MinValue, _ops.MinValue);

    /// <summary>
    /// Gets the maximum value that can be represented using complex numbers with the underlying type T.
    /// </summary>
    /// <value>A complex number with both real and imaginary parts set to the maximum value of type T.</value>
    /// <remarks>
    /// <para>
    /// This property returns a complex number with both real and imaginary parts set to the maximum value
    /// that can be represented by the underlying type T. This is useful for algorithms that need to work
    /// with the full range of possible complex values.
    /// </para>
    /// <para><b>For Beginners:</b> This property gives you the largest possible complex number.
    ///
    /// For complex numbers, "maximum" refers to having both parts at their maximum values.
    ///
    /// For example, if T is double:
    /// - The maximum value would have both real and imaginary parts equal to double.MaxValue
    /// - This represents the top-right corner of the representable complex plane
    ///
    /// This is primarily used in algorithms that need to track the largest possible value.
    /// </para>
    /// </remarks>
    public Complex<T> MaxValue => new(_ops.MaxValue, _ops.MaxValue);

    /// <summary>
    /// Determines whether a complex number has a NaN (Not a Number) component.
    /// </summary>
    /// <param name="value">The complex number to check.</param>
    /// <returns>true if either the real or imaginary part is NaN; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method checks whether either the real or imaginary component of a complex number is NaN.
    /// A complex number is considered NaN if either of its components is NaN.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if a complex number contains an invalid value.
    /// 
    /// NaN stands for "Not a Number" and occurs when a mathematical operation doesn't have a valid result.
    /// 
    /// A complex number is considered NaN if:
    /// - Its real part is NaN, OR
    /// - Its imaginary part is NaN
    /// 
    /// For example, operations like 0/0 or the square root of a negative number (in real arithmetic)
    /// result in NaN. This method helps detect these invalid results.
    /// 
    /// Note: Whether NaN can occur depends on the underlying type T. For instance, integers cannot
    /// represent NaN, but floating-point types like float and double can.
    /// </para>
    /// </remarks>
    public bool IsNaN(Complex<T> value) => _ops.IsNaN(value.Real) || _ops.IsNaN(value.Imaginary);

    /// <summary>
    /// Determines whether a complex number has an infinity component.
    /// </summary>
    /// <param name="value">The complex number to check.</param>
    /// <returns>true if either the real or imaginary part is infinity; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method checks whether either the real or imaginary component of a complex number is infinity.
    /// A complex number is considered infinite if either of its components is infinite.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if a complex number contains an infinite value.
    /// 
    /// Infinity represents a value that exceeds the representable range of the numeric type.
    /// 
    /// A complex number is considered infinite if:
    /// - Its real part is infinite, OR
    /// - Its imaginary part is infinite
    /// 
    /// For example, operations like 1/0 result in infinity. This method helps detect these
    /// special values in complex numbers.
    /// 
    /// Note: Whether infinity can occur depends on the underlying type T. For instance, integers
    /// cannot represent infinity, but floating-point types like float and double can.
    /// </para>
    /// </remarks>
    public bool IsInfinity(Complex<T> value) => _ops.IsInfinity(value.Real) || _ops.IsInfinity(value.Imaginary);

    /// <summary>
    /// Determines the sign of a complex number or returns zero if it is exactly zero.
    /// </summary>
    /// <param name="value">The complex number to check.</param>
    /// <returns>
    /// One if the complex number is in the right half-plane or positive imaginary axis;
    /// Negative one if the complex number is in the left half-plane or negative imaginary axis;
    /// Zero if both components are zero.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method determines the "sign" of a complex number, which is not as straightforward as with real numbers.
    /// It uses a convention where numbers with positive real part (or zero real and positive imaginary) return 1+0i,
    /// numbers with negative real part (or zero real and negative imaginary) return -1+0i,
    /// and zero (both components zero) returns 0+0i.
    /// </para>
    /// <para><b>For Beginners:</b> This method determines the "direction" of a complex number.
    /// 
    /// For real numbers, the sign is simple: positive, negative, or zero.
    /// For complex numbers, it's more complicated and follows these rules:
    /// 
    /// - If the real part is positive, the result is 1 + 0i
    /// - If the real part is zero but the imaginary part is positive, the result is 1 + 0i
    /// - If the real part is negative, the result is -1 + 0i
    /// - If the real part is zero but the imaginary part is negative, the result is -1 + 0i
    /// - If both parts are zero, the result is 0 + 0i
    /// 
    /// For example:
    /// - SignOrZero(3 + 4i) = 1 + 0i
    /// - SignOrZero(-2 + 7i) = -1 + 0i
    /// - SignOrZero(0 + 0i) = 0 + 0i
    /// 
    /// This is useful in algorithms that need to know the general direction of a complex number.
    /// </para>
    /// </remarks>
    public Complex<T> SignOrZero(Complex<T> value)
    {
        if (_ops.GreaterThan(value.Real, _ops.Zero) || (_ops.Equals(value.Real, _ops.Zero) && _ops.GreaterThan(value.Imaginary, _ops.Zero)))
            return One;
        if (_ops.LessThan(value.Real, _ops.Zero) || (_ops.Equals(value.Real, _ops.Zero) && _ops.LessThan(value.Imaginary, _ops.Zero)))
            return Negate(One);

        return Zero;
    }

    /// <summary>
    /// Converts a complex number to a 32-bit integer by rounding its magnitude.
    /// </summary>
    /// <param name="value">The complex number to convert.</param>
    /// <returns>The rounded magnitude of the complex number as an integer.</returns>
    /// <remarks>
    /// <para>
    /// This method converts a complex number to an integer by taking its magnitude (absolute value)
    /// and rounding it to the nearest integer. This discards all phase information and only
    /// considers the size of the complex number.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts a complex number to a regular integer.
    /// 
    /// Since complex numbers have two components, converting to a single integer isn't straightforward.
    /// This method:
    /// 
    /// 1. Calculates the magnitude (distance from zero) of the complex number
    /// 2. Rounds that magnitude to the nearest integer
    /// 
    /// For example:
    /// - 3 + 4i has magnitude 5, so it converts to 5
    /// - 1 + 1i has magnitude v2 ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 1.414, which rounds to 1
    /// 
    /// This conversion loses all information about direction, keeping only the size.
    /// </para>
    /// </remarks>
    public int ToInt32(Complex<T> value)
    {
        double magnitude = Convert.ToDouble(value.Magnitude);
        return (int)Math.Round(magnitude);
    }

    /// <summary>
    /// Gets the number of bits used for precision in the underlying type T.
    /// </summary>
    /// <remarks>
    /// For Complex<T>, this returns the precision bits of the underlying type T.
    /// Note that a complex number stores two values (real and imaginary), so the total
    /// storage is actually twice this value.
    /// </remarks>
    public int PrecisionBits => _ops.PrecisionBits;

    /// <summary>
    /// Converts a Complex<T> value to float by extracting the real part.
    /// </summary>
    /// <param name="value">The complex value to convert.</param>
    /// <returns>The real part of the complex number as a float.</returns>
    /// <remarks>
    /// This conversion only succeeds if the imaginary part is zero.
    /// If the imaginary part is non-zero, throws NotSupportedException to prevent silent data loss.
    /// </remarks>
    public float ToFloat(Complex<T> value)
    {
        if (!_ops.Equals(value.Imaginary, _ops.Zero))
        {
            throw new NotSupportedException(
                "Cannot convert Complex<T> with non-zero imaginary component to scalar float. " +
                "This would result in silent data loss. Extract Real property explicitly if this is intentional.");
        }
        return _ops.ToFloat(value.Real);
    }

    /// <summary>
    /// Converts a float value to Complex<T> with zero imaginary part.
    /// </summary>
    /// <param name="value">The float value to convert.</param>
    /// <returns>A complex number with the float as the real part and zero imaginary part.</returns>
    public Complex<T> FromFloat(float value) => new Complex<T>(_ops.FromFloat(value), _ops.Zero);

    /// <summary>
    /// Converts a Complex<T> value to Half by extracting the real part.
    /// </summary>
    /// <param name="value">The complex value to convert.</param>
    /// <returns>The real part of the complex number as a Half.</returns>
    /// <remarks>
    /// This conversion only succeeds if the imaginary part is zero.
    /// If the imaginary part is non-zero, throws NotSupportedException to prevent silent data loss.
    /// </remarks>
    public Half ToHalf(Complex<T> value)
    {
        if (!_ops.Equals(value.Imaginary, _ops.Zero))
        {
            throw new NotSupportedException(
                "Cannot convert Complex<T> with non-zero imaginary component to scalar Half. " +
                "This would result in silent data loss. Extract Real property explicitly if this is intentional.");
        }
        return _ops.ToHalf(value.Real);
    }

    /// <summary>
    /// Converts a Half value to Complex<T> with zero imaginary part.
    /// </summary>
    /// <param name="value">The Half value to convert.</param>
    /// <returns>A complex number with the Half as the real part and zero imaginary part.</returns>
    public Complex<T> FromHalf(Half value) => new Complex<T>(_ops.FromHalf(value), _ops.Zero);

    /// <summary>
    /// Converts a Complex<T> value to double by extracting the real part.
    /// </summary>
    /// <param name="value">The complex value to convert.</param>
    /// <returns>The real part of the complex number as a double.</returns>
    /// <remarks>
    /// This conversion only succeeds if the imaginary part is zero.
    /// If the imaginary part is non-zero, throws NotSupportedException to prevent silent data loss.
    /// </remarks>
    public double ToDouble(Complex<T> value)
    {
        if (!_ops.Equals(value.Imaginary, _ops.Zero))
        {
            throw new NotSupportedException(
                "Cannot convert Complex<T> with non-zero imaginary component to scalar double. " +
                "This would result in silent data loss. Extract Real property explicitly if this is intentional.");
        }
        return _ops.ToDouble(value.Real);
    }

    /// <inheritdoc/>
    public bool SupportsCpuAcceleration => false;

    /// <inheritdoc/>
    public bool SupportsGpuAcceleration => false;

    #region IVectorizedOperations<Complex<T>> Implementation - Fallback using sequential loops

    /// <summary>
    /// Performs element-wise addition using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Add(ReadOnlySpan<Complex<T>> x, ReadOnlySpan<Complex<T>> y, Span<Complex<T>> destination)
        => VectorizedOperationsFallback.Add(this, x, y, destination);

    /// <summary>
    /// Performs element-wise subtraction using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Subtract(ReadOnlySpan<Complex<T>> x, ReadOnlySpan<Complex<T>> y, Span<Complex<T>> destination)
        => VectorizedOperationsFallback.Subtract(this, x, y, destination);

    /// <summary>
    /// Performs element-wise multiplication using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Multiply(ReadOnlySpan<Complex<T>> x, ReadOnlySpan<Complex<T>> y, Span<Complex<T>> destination)
        => VectorizedOperationsFallback.Multiply(this, x, y, destination);

    /// <summary>
    /// Performs element-wise division using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Divide(ReadOnlySpan<Complex<T>> x, ReadOnlySpan<Complex<T>> y, Span<Complex<T>> destination)
        => VectorizedOperationsFallback.Divide(this, x, y, destination);

    /// <summary>
    /// Computes dot product using sequential loops (fallback, no SIMD).
    /// </summary>
    public Complex<T> Dot(ReadOnlySpan<Complex<T>> x, ReadOnlySpan<Complex<T>> y)
        => VectorizedOperationsFallback.Dot(this, x, y);

    /// <summary>
    /// Computes sum using sequential loops (fallback, no SIMD).
    /// </summary>
    public Complex<T> Sum(ReadOnlySpan<Complex<T>> x)
        => VectorizedOperationsFallback.Sum(this, x);

    /// <summary>
    /// Finds maximum using sequential loops (fallback, no SIMD).
    /// </summary>
    public Complex<T> Max(ReadOnlySpan<Complex<T>> x)
        => VectorizedOperationsFallback.Max(this, x);

    /// <summary>
    /// Finds minimum using sequential loops (fallback, no SIMD).
    /// </summary>
    public Complex<T> Min(ReadOnlySpan<Complex<T>> x)
        => VectorizedOperationsFallback.Min(this, x);

    /// <summary>
    /// Computes exponential using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Exp(ReadOnlySpan<Complex<T>> x, Span<Complex<T>> destination)
        => VectorizedOperationsFallback.Exp(this, x, destination);

    /// <summary>
    /// Computes natural logarithm using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Log(ReadOnlySpan<Complex<T>> x, Span<Complex<T>> destination)
        => VectorizedOperationsFallback.Log(this, x, destination);

    /// <summary>
    /// Computes hyperbolic tangent using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Tanh(ReadOnlySpan<Complex<T>> x, Span<Complex<T>> destination)
        => VectorizedOperationsFallback.Tanh(this, x, destination);

    /// <summary>
    /// Computes sigmoid using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Sigmoid(ReadOnlySpan<Complex<T>> x, Span<Complex<T>> destination)
        => VectorizedOperationsFallback.Sigmoid(this, x, destination);

    /// <summary>
    /// Computes base-2 logarithm using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Log2(ReadOnlySpan<Complex<T>> x, Span<Complex<T>> destination)
        => VectorizedOperationsFallback.Log2(this, x, destination);

    /// <summary>
    /// Computes softmax using sequential loops (fallback, no SIMD).
    /// </summary>
    public void SoftMax(ReadOnlySpan<Complex<T>> x, Span<Complex<T>> destination)
        => VectorizedOperationsFallback.SoftMax(this, x, destination);

    /// <summary>
    /// Computes cosine similarity using sequential loops (fallback, no SIMD).
    /// </summary>
    public Complex<T> CosineSimilarity(ReadOnlySpan<Complex<T>> x, ReadOnlySpan<Complex<T>> y)
        => VectorizedOperationsFallback.CosineSimilarity(this, x, y);

    /// <summary>
    /// Fills a span with a specified value.
    /// </summary>
    public void Fill(Span<Complex<T>> destination, Complex<T> value) => destination.Fill(value);

    /// <summary>
    /// Multiplies each element in a span by a scalar value.
    /// </summary>
    public void MultiplyScalar(ReadOnlySpan<Complex<T>> x, Complex<T> scalar, Span<Complex<T>> destination)
        => VectorizedOperationsFallback.MultiplyScalar(this, x, scalar, destination);

    /// <summary>
    /// Divides each element in a span by a scalar value.
    /// </summary>
    public void DivideScalar(ReadOnlySpan<Complex<T>> x, Complex<T> scalar, Span<Complex<T>> destination)
        => VectorizedOperationsFallback.DivideScalar(this, x, scalar, destination);

    /// <summary>
    /// Adds a scalar value to each element in a span.
    /// </summary>
    public void AddScalar(ReadOnlySpan<Complex<T>> x, Complex<T> scalar, Span<Complex<T>> destination)
        => VectorizedOperationsFallback.AddScalar(this, x, scalar, destination);

    /// <summary>
    /// Subtracts a scalar value from each element in a span.
    /// </summary>
    public void SubtractScalar(ReadOnlySpan<Complex<T>> x, Complex<T> scalar, Span<Complex<T>> destination)
        => VectorizedOperationsFallback.SubtractScalar(this, x, scalar, destination);

    /// <summary>
    /// Computes square root of each element using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Sqrt(ReadOnlySpan<Complex<T>> x, Span<Complex<T>> destination)
        => VectorizedOperationsFallback.Sqrt(this, x, destination);

    /// <summary>
    /// Computes absolute value of each element using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Abs(ReadOnlySpan<Complex<T>> x, Span<Complex<T>> destination)
        => VectorizedOperationsFallback.Abs(this, x, destination);

    /// <summary>
    /// Negates each element using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Negate(ReadOnlySpan<Complex<T>> x, Span<Complex<T>> destination)
        => VectorizedOperationsFallback.Negate(this, x, destination);

    /// <summary>
    /// Clips each element to the specified range using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Clip(ReadOnlySpan<Complex<T>> x, Complex<T> min, Complex<T> max, Span<Complex<T>> destination)
        => VectorizedOperationsFallback.Clip(this, x, min, max, destination);

    /// <summary>
    /// Raises each element to a specified power using sequential loops (fallback, no SIMD).
    /// </summary>
    public void Pow(ReadOnlySpan<Complex<T>> x, Complex<T> power, Span<Complex<T>> destination)
        => VectorizedOperationsFallback.Pow(this, x, power, destination);

    /// <summary>
    /// Copies elements from source to destination.
    /// </summary>
    public void Copy(ReadOnlySpan<Complex<T>> source, Span<Complex<T>> destination)
        => source.CopyTo(destination);

    #endregion
}
