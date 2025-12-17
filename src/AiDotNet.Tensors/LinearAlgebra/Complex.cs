using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Represents a complex number with real and imaginary parts.
/// </summary>
/// <typeparam name="T">The numeric type used for the real and imaginary parts.</typeparam>
/// <remarks>
/// <para>
/// Complex numbers extend the concept of real numbers by adding an imaginary component.
/// They are often used in advanced mathematical calculations, signal processing, and
/// certain machine learning algorithms.
/// </para>
/// <para>
/// <b>For Beginners:</b> A complex number has two parts - a real part and an imaginary part.
/// The real part is just like a regular number you're familiar with. The imaginary part
/// is multiplied by "i", which represents the square root of -1 (a number that doesn't
/// exist in the real number system).
/// </para>
/// <para>
/// For example, in the complex number "3 + 2i":
/// - 3 is the real part
/// - 2 is the imaginary part
/// </para>
/// <para>
/// Complex numbers are useful in many areas of science and engineering, including:
/// - Electrical engineering (analyzing circuits)
/// - Signal processing (analyzing sound waves)
/// - Quantum mechanics
/// - Some machine learning algorithms
/// </para>
/// </remarks>
public readonly struct Complex<T>
{
    /// <summary>
    /// Provides numeric operations for the type T.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is a helper object that allows us to perform math operations
    /// regardless of what numeric type (like double, float, decimal) we're using.
    /// You don't need to interact with this directly.
    /// </remarks>
    private readonly INumericOperations<T> _ops;

    /// <summary>
    /// Gets numeric operations, initializing if needed for default instances.
    /// </summary>
    private INumericOperations<T> Ops => _ops ?? MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the real part of the complex number.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the regular number part of a complex number.
    /// For example, in "3 + 2i", the real part is 3.
    /// </remarks>
    public T Real { get; }

    /// <summary>
    /// Gets the imaginary part of the complex number.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the part that's multiplied by "i" in a complex number.
    /// For example, in "3 + 2i", the imaginary part is 2.
    /// </remarks>
    public T Imaginary { get; }

    /// <summary>
    /// Initializes a new instance of the Complex struct with specified real and imaginary parts.
    /// </summary>
    /// <param name="real">The real part of the complex number.</param>
    /// <param name="imaginary">The imaginary part of the complex number.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This is how you create a new complex number. You provide the real part
    /// (a regular number) and the imaginary part (the coefficient of i).
    /// 
    /// Example: To create the complex number 3 + 2i, you would write:
    /// <code>
    /// var myComplex = new Complex&lt;double&gt;(3.0, 2.0);
    /// </code>
    /// </remarks>
    public Complex(T real, T imaginary)
    {
        Real = real;
        Imaginary = imaginary;
        _ops = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Gets the magnitude (or absolute value) of the complex number.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The magnitude represents the distance from the origin (0,0) to the complex number
    /// in the complex plane.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The magnitude is like the "size" of the complex number. It's calculated
    /// using the Pythagorean theorem: sqrt(real² + imaginary²).
    /// </para>
    /// <para>
    /// Think of a complex number as a point on a 2D graph, where the real part is the x-coordinate
    /// and the imaginary part is the y-coordinate. The magnitude is the straight-line distance
    /// from the origin (0,0) to that point.
    /// </para>
    /// <para>
    /// For example, the magnitude of 3 + 4i is sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5.
    /// </para>
    /// </remarks>
    public T Magnitude => Ops.Sqrt(Ops.Add(Ops.Square(Real), Ops.Square(Imaginary)));

    /// <summary>
    /// Gets the phase (or argument) of the complex number.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The phase represents the angle (in radians) between the positive real axis and the line
    /// connecting the origin to the complex number in the complex plane.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The phase is the angle that the complex number makes with the positive
    /// x-axis when plotted on a 2D graph. It's measured in radians (a full circle is 2p radians
    /// or about 6.28 radians).
    /// </para>
    /// <para>
    /// If you think of a complex number as a point on a 2D graph:
    /// - The real part is the x-coordinate
    /// - The imaginary part is the y-coordinate
    /// - The phase is the angle between the positive x-axis and the line from (0,0) to your point
    /// </para>
    /// <para>
    /// For example:
    /// - The phase of 1 + 0i is 0 radians (0 degrees)
    /// - The phase of 0 + 1i is p/2 radians (90 degrees)
    /// - The phase of -1 + 0i is p radians (180 degrees)
    /// - The phase of 0 - 1i is -p/2 radians (-90 degrees)
    /// </para>
    /// </remarks>
    public T Phase => Ops.FromDouble(Math.Atan2(Convert.ToDouble(Imaginary), Convert.ToDouble(Real)));

    /// <summary>
    /// Adds two complex numbers.
    /// </summary>
    /// <param name="a">The first complex number.</param>
    /// <param name="b">The second complex number.</param>
    /// <returns>A new complex number that is the sum of the two complex numbers.</returns>
    /// <remarks>
    /// <para>
    /// Addition of complex numbers is performed by adding their real parts together and
    /// adding their imaginary parts together.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> To add two complex numbers, you simply add their real parts together
    /// and their imaginary parts together.
    /// </para>
    /// <para>
    /// For example:
    /// (3 + 2i) + (4 + 5i) = (3 + 4) + (2 + 5)i = 7 + 7i
    /// </para>
    /// </remarks>
    public static Complex<T> operator +(Complex<T> a, Complex<T> b)
        => new(a.Ops.Add(a.Real, b.Real), a.Ops.Add(a.Imaginary, b.Imaginary));

    /// <summary>
    /// Subtracts one complex number from another.
    /// </summary>
    /// <param name="a">The complex number to subtract from.</param>
    /// <param name="b">The complex number to subtract.</param>
    /// <returns>A new complex number that is the difference of the two complex numbers.</returns>
    /// <remarks>
    /// <para>
    /// Subtraction of complex numbers is performed by subtracting their real parts and
    /// subtracting their imaginary parts.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> To subtract one complex number from another, you subtract the real parts
    /// and subtract the imaginary parts.
    /// </para>
    /// <para>
    /// For example:
    /// (7 + 3i) - (2 + 1i) = (7 - 2) + (3 - 1)i = 5 + 2i
    /// </para>
    /// </remarks>
    public static Complex<T> operator -(Complex<T> a, Complex<T> b)
        => new(a.Ops.Subtract(a.Real, b.Real), a.Ops.Subtract(a.Imaginary, b.Imaginary));

    /// <summary>
    /// Multiplies two complex numbers.
    /// </summary>
    /// <param name="a">The first complex number.</param>
    /// <param name="b">The second complex number.</param>
    /// <returns>A new complex number that is the product of the two complex numbers.</returns>
    /// <remarks>
    /// <para>
    /// Multiplication of complex numbers follows the distributive property and the rule that i² = -1.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Multiplying complex numbers is a bit more involved than addition or subtraction.
    /// The formula is:
    /// (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    /// </para>
    /// <para>
    /// For example:
    /// (3 + 2i) * (1 + 4i) = (3*1 - 2*4) + (3*4 + 2*1)i = (3 - 8) + (12 + 2)i = -5 + 14i
    /// </para>
    /// <para>
    /// This is similar to multiplying two binomials (a + b)(c + d), but with the special rule
    /// that i² = -1, which is why the term bd becomes negative.
    /// </para>
    /// </remarks>
    public static Complex<T> operator *(Complex<T> a, Complex<T> b)
        => new(
            a.Ops.Subtract(a.Ops.Multiply(a.Real, b.Real), a.Ops.Multiply(a.Imaginary, b.Imaginary)),
            a.Ops.Add(a.Ops.Multiply(a.Real, b.Imaginary), a.Ops.Multiply(a.Imaginary, b.Real))
        );

    /// <summary>
    /// Divides one complex number by another.
    /// </summary>
    /// <param name="a">The complex number to be divided (numerator).</param>
    /// <param name="b">The complex number to divide by (denominator).</param>
    /// <returns>A new complex number that is the quotient of the division.</returns>
    /// <remarks>
    /// <para>
    /// Division of complex numbers is performed by multiplying both numerator and denominator
    /// by the complex conjugate of the denominator, which converts the denominator to a real number.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Dividing complex numbers is one of the more complicated operations.
    /// We can't directly divide complex numbers, so we use a special technique:
    /// </para>
    /// <para>
    /// 1. We multiply both the top and bottom of the fraction by the conjugate of the denominator
    /// 2. This makes the denominator a real number (no imaginary part)
    /// 3. Then we can separate the real and imaginary parts of the result
    /// </para>
    /// <para>
    /// For example, to calculate (3 + 2i) / (1 + i):
    /// - First, we multiply both top and bottom by the conjugate of (1 + i), which is (1 - i)
    /// - This gives us: [(3 + 2i)(1 - i)] / [(1 + i)(1 - i)]
    /// - The denominator becomes (1² + 1²) = 2
    /// - The numerator becomes (3 + 2i)(1 - i) = 3 - 3i + 2i - 2i² = 3 - 3i + 2i + 2 = 5 - i
    /// - So the result is (5 - i) / 2 = 2.5 - 0.5i
    /// </para>
    /// </remarks>
    public static Complex<T> operator /(Complex<T> a, Complex<T> b)
    {
        T denominator = a.Ops.Add(a.Ops.Square(b.Real), a.Ops.Square(b.Imaginary));

        // Check for division by zero (both real and imaginary parts are zero)
        if (a.Ops.Equals(denominator, a.Ops.Zero))
        {
            throw new DivideByZeroException("Cannot divide by a complex number with both real and imaginary parts equal to zero.");
        }

        return new Complex<T>(
            a.Ops.Divide(a.Ops.Add(a.Ops.Multiply(a.Real, b.Real), a.Ops.Multiply(a.Imaginary, b.Imaginary)), denominator),
            a.Ops.Divide(a.Ops.Subtract(a.Ops.Multiply(a.Imaginary, b.Real), a.Ops.Multiply(a.Real, b.Imaginary)), denominator)
        );
    }

    /// <summary>
    /// Determines whether two complex numbers are equal.
    /// </summary>
    /// <param name="left">The first complex number to compare.</param>
    /// <param name="right">The second complex number to compare.</param>
    /// <returns>True if the complex numbers are equal; otherwise, false.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This checks if two complex numbers have the same real part and the same
    /// imaginary part. If both parts match, the complex numbers are considered equal.
    /// </remarks>
    public static bool operator ==(Complex<T> left, Complex<T> right)
        => left.Equals(right);

    /// <summary>
    /// Determines whether two complex numbers are not equal.
    /// </summary>
    /// <param name="left">The first complex number to compare.</param>
    /// <param name="right">The second complex number to compare.</param>
    /// <returns>True if the complex numbers are not equal; otherwise, false.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This checks if two complex numbers are different. If either the real part
    /// or the imaginary part (or both) are different, the complex numbers are not equal.
    /// </remarks>
    public static bool operator !=(Complex<T> left, Complex<T> right)
        => !left.Equals(right);

    /// <summary>
    /// Determines whether this complex number is equal to another object.
    /// </summary>
    /// <param name="obj">The object to compare with.</param>
    /// <returns>True if the objects are equal; otherwise, false.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method checks if this complex number equals another object.
    /// It first checks if the other object is a complex number, and if so, compares their values.
    /// </remarks>
    public override bool Equals(object? obj)
        => obj is Complex<T> complex && Equals(complex);

    /// <summary>
    /// Determines whether this complex number is equal to another complex number.
    /// </summary>
    /// <param name="other">The complex number to compare with.</param>
    /// <returns>True if the complex numbers are equal; otherwise, false.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method checks if two complex numbers have the same real and imaginary parts.
    /// If both parts match exactly, the complex numbers are considered equal.
    /// </remarks>
    public bool Equals(Complex<T> other)
        => Ops.Equals(Real, other.Real) && Ops.Equals(Imaginary, other.Imaginary);

    /// <summary>
    /// Returns a hash code for this complex number.
    /// </summary>
    /// <returns>A hash code for the current object.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> A hash code is a numeric value that is used to identify an object in collections
    /// like dictionaries and hash sets. You don't need to call this method directly in most cases.
    /// </remarks>
    public override int GetHashCode()
    {
        unchecked
        {
            int hash = 17;
            hash = hash * 23 + (Real?.GetHashCode() ?? 0);
            hash = hash * 23 + (Imaginary?.GetHashCode() ?? 0);
            return hash;
        }
    }

    /// <summary>
    /// Returns the complex conjugate of this complex number.
    /// </summary>
    /// <returns>A new complex number that is the conjugate of this complex number.</returns>
    /// <remarks>
    /// <para>
    /// The complex conjugate of a complex number (a + bi) is (a - bi).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The conjugate of a complex number keeps the real part the same but changes
    /// the sign of the imaginary part. For example, the conjugate of (3 + 2i) is (3 - 2i).
    /// </para>
    /// <para>
    /// Complex conjugates are useful in many calculations, especially when dividing complex numbers
    /// or finding absolute values. When you multiply a complex number by its conjugate, you get
    /// a real number (with no imaginary part).
    /// </para>
    /// </remarks>
    public Complex<T> Conjugate()
        => new(Real, Ops.Negate(Imaginary));

    /// <summary>
    /// Returns a string representation of this complex number.
    /// </summary>
    /// <returns>A string that represents the complex number in the format "a + bi" or "a - bi".</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method converts the complex number to a readable text format.
    /// For example, a complex number with Real = 3 and Imaginary = 2 would be displayed as "3 + 2i",
    /// and a complex number with Real = 3 and Imaginary = -2 would be displayed as "3 - 2i".
    /// </remarks>
    public override string ToString()
    {
        double imaginaryValue = Convert.ToDouble(Imaginary);
        if (imaginaryValue < 0)
        {
            return $"{Real} - {Ops.Negate(Imaginary)}i";
        }
        return $"{Real} + {Imaginary}i";
    }

    /// <summary>
    /// Creates a complex number from polar coordinates (magnitude and phase).
    /// </summary>
    /// <param name="magnitude">The magnitude (or absolute value) of the complex number.</param>
    /// <param name="phase">The phase (or argument) of the complex number in radians.</param>
    /// <returns>A new complex number with the specified magnitude and phase.</returns>
    /// <remarks>
    /// <para>
    /// Converts from polar form (magnitude and phase) to rectangular form (real and imaginary parts)
    /// using the formulas: real = magnitude * cos(phase) and imaginary = magnitude * sin(phase).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> There are two ways to represent complex numbers:
    /// </para>
    /// <para>
    /// 1. Rectangular form: a + bi (using real and imaginary parts)
    /// 2. Polar form: r?? (using magnitude and angle)
    /// </para>
    /// <para>
    /// This method converts from polar form to the standard rectangular form. The magnitude (r)
    /// represents the distance from the origin, and the phase (?) represents the angle from the
    /// positive x-axis (measured in radians).
    /// </para>
    /// <para>
    /// For example, to create the complex number 3 + 4i using polar coordinates:
    /// - First, calculate the magnitude: sqrt(3² + 4²) = 5
    /// - Then, calculate the phase: arctan(4/3) ≈ 0.9273 radians
    /// - Use FromPolarCoordinates(5, 0.9273)
    /// </para>
    /// </remarks>
    public static Complex<T> FromPolarCoordinates(T magnitude, T phase)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return new Complex<T>(
            ops.Multiply(magnitude, MathHelper.Cos(phase)),
            ops.Multiply(magnitude, MathHelper.Sin(phase))
        );
    }
}
