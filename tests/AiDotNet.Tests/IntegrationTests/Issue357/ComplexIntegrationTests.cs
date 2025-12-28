using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Issue357;

/// <summary>
/// Integration tests for the Complex<T> class covering arithmetic operations,
/// polar conversions, and mathematical properties.
/// </summary>
public class ComplexIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region Construction and Basic Properties

    [Fact]
    public void Complex_DefaultConstructor_CreatesZeroComplex()
    {
        // Arrange & Act
        var z = new Complex<double>();

        // Assert
        Assert.Equal(0.0, z.Real);
        Assert.Equal(0.0, z.Imaginary);
    }

    [Theory]
    [InlineData(3.0, 4.0)]
    [InlineData(-1.0, 2.0)]
    [InlineData(0.0, 0.0)]
    [InlineData(5.0, 0.0)]
    [InlineData(0.0, -3.0)]
    public void Complex_ParameterizedConstructor_SetsRealAndImaginary(double real, double imaginary)
    {
        // Arrange & Act
        var z = new Complex<double>(real, imaginary);

        // Assert
        Assert.Equal(real, z.Real);
        Assert.Equal(imaginary, z.Imaginary);
    }

    #endregion

    #region Arithmetic Operations

    [Theory]
    [InlineData(1, 2, 3, 4, 4, 6)]
    [InlineData(-1, -2, 1, 2, 0, 0)]
    [InlineData(0, 0, 5, 3, 5, 3)]
    public void Complex_Addition_ProducesCorrectResult(
        double r1, double i1, double r2, double i2, double expectedReal, double expectedImag)
    {
        // Arrange
        var z1 = new Complex<double>(r1, i1);
        var z2 = new Complex<double>(r2, i2);

        // Act
        var result = z1 + z2;

        // Assert
        Assert.Equal(expectedReal, result.Real, Tolerance);
        Assert.Equal(expectedImag, result.Imaginary, Tolerance);
    }

    [Theory]
    [InlineData(5, 3, 2, 1, 3, 2)]
    [InlineData(1, 1, 1, 1, 0, 0)]
    [InlineData(0, 0, 3, 4, -3, -4)]
    public void Complex_Subtraction_ProducesCorrectResult(
        double r1, double i1, double r2, double i2, double expectedReal, double expectedImag)
    {
        // Arrange
        var z1 = new Complex<double>(r1, i1);
        var z2 = new Complex<double>(r2, i2);

        // Act
        var result = z1 - z2;

        // Assert
        Assert.Equal(expectedReal, result.Real, Tolerance);
        Assert.Equal(expectedImag, result.Imaginary, Tolerance);
    }

    [Fact]
    public void Complex_Multiplication_FollowsFoilRule()
    {
        // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        // (3 + 2i)(1 + 4i) = (3*1 - 2*4) + (3*4 + 2*1)i = (3 - 8) + (12 + 2)i = -5 + 14i
        var z1 = new Complex<double>(3, 2);
        var z2 = new Complex<double>(1, 4);

        var result = z1 * z2;

        Assert.Equal(-5.0, result.Real, Tolerance);
        Assert.Equal(14.0, result.Imaginary, Tolerance);
    }

    [Fact]
    public void Complex_Division_ProducesCorrectResult()
    {
        // (a + bi)/(c + di) = ((ac + bd) + (bc - ad)i) / (c^2 + d^2)
        // (3 + 2i)/(1 + 1i) = ((3 + 2) + (2 - 3)i) / 2 = (5 - i) / 2 = 2.5 - 0.5i
        var z1 = new Complex<double>(3, 2);
        var z2 = new Complex<double>(1, 1);

        var result = z1 / z2;

        Assert.Equal(2.5, result.Real, Tolerance);
        Assert.Equal(-0.5, result.Imaginary, Tolerance);
    }

    [Fact]
    public void Complex_MultiplyByScalar_ScalesBothParts()
    {
        var z = new Complex<double>(3, 4);
        var scalar = new Complex<double>(2, 0);
        var result = z * scalar;

        Assert.Equal(6.0, result.Real, Tolerance);
        Assert.Equal(8.0, result.Imaginary, Tolerance);
    }

    #endregion

    #region Magnitude and Phase

    [Theory]
    [InlineData(3, 4, 5)]       // Classic 3-4-5 triangle
    [InlineData(0, 0, 0)]       // Zero complex
    [InlineData(1, 0, 1)]       // Real only
    [InlineData(0, 1, 1)]       // Imaginary only
    [InlineData(-3, -4, 5)]     // Negative components
    public void Complex_Magnitude_EqualsEuclideanNorm(double real, double imag, double expectedMagnitude)
    {
        var z = new Complex<double>(real, imag);

        double magnitude = z.Magnitude;

        Assert.Equal(expectedMagnitude, magnitude, Tolerance);
    }

    [Fact]
    public void Complex_Phase_ComputesCorrectAngle()
    {
        // 1 + i has phase pi/4
        var z = new Complex<double>(1, 1);
        Assert.Equal(Math.PI / 4, z.Phase, Tolerance);

        // 0 + i has phase pi/2
        var zPureImag = new Complex<double>(0, 1);
        Assert.Equal(Math.PI / 2, zPureImag.Phase, Tolerance);

        // -1 + 0i has phase pi
        var zNegReal = new Complex<double>(-1, 0);
        Assert.Equal(Math.PI, zNegReal.Phase, Tolerance);
    }

    #endregion

    #region Conjugate

    [Theory]
    [InlineData(3, 4, 3, -4)]
    [InlineData(-2, 5, -2, -5)]
    [InlineData(0, 0, 0, 0)]
    [InlineData(7, 0, 7, 0)]
    public void Complex_Conjugate_NegatesImaginaryPart(
        double real, double imag, double expectedReal, double expectedImag)
    {
        var z = new Complex<double>(real, imag);

        var conj = z.Conjugate();

        Assert.Equal(expectedReal, conj.Real, Tolerance);
        Assert.Equal(expectedImag, conj.Imaginary, Tolerance);
    }

    [Fact]
    public void Complex_MultiplyByConjugate_ProducesRealNumber()
    {
        // z * conj(z) = |z|^2 (a real number)
        var z = new Complex<double>(3, 4);
        var conj = z.Conjugate();

        var result = z * conj;

        Assert.Equal(25.0, result.Real, Tolerance); // 3^2 + 4^2 = 25
        Assert.Equal(0.0, result.Imaginary, Tolerance);
    }

    #endregion

    #region Polar Coordinates

    [Fact]
    public void Complex_FromPolarCoordinates_CreatesCorrectComplex()
    {
        // r=2, theta=pi/4 -> 2*(cos(pi/4) + i*sin(pi/4)) = sqrt(2) + sqrt(2)*i
        double r = 2.0;
        double theta = Math.PI / 4;

        var z = Complex<double>.FromPolarCoordinates(r, theta);

        Assert.Equal(Math.Sqrt(2), z.Real, Tolerance);
        Assert.Equal(Math.Sqrt(2), z.Imaginary, Tolerance);
    }

    [Fact]
    public void Complex_PolarRoundTrip_PreservesValue()
    {
        var original = new Complex<double>(3, 4);

        double r = original.Magnitude;
        double theta = original.Phase;
        var reconstructed = Complex<double>.FromPolarCoordinates(r, theta);

        Assert.Equal(original.Real, reconstructed.Real, Tolerance);
        Assert.Equal(original.Imaginary, reconstructed.Imaginary, Tolerance);
    }

    #endregion

    #region Mathematical Properties

    [Fact]
    public void Complex_Addition_IsCommutative()
    {
        var z1 = new Complex<double>(3, 4);
        var z2 = new Complex<double>(1, 2);

        var result1 = z1 + z2;
        var result2 = z2 + z1;

        Assert.Equal(result1.Real, result2.Real, Tolerance);
        Assert.Equal(result1.Imaginary, result2.Imaginary, Tolerance);
    }

    [Fact]
    public void Complex_Multiplication_IsCommutative()
    {
        var z1 = new Complex<double>(3, 4);
        var z2 = new Complex<double>(1, 2);

        var result1 = z1 * z2;
        var result2 = z2 * z1;

        Assert.Equal(result1.Real, result2.Real, Tolerance);
        Assert.Equal(result1.Imaginary, result2.Imaginary, Tolerance);
    }

    [Fact]
    public void Complex_Addition_IsAssociative()
    {
        var z1 = new Complex<double>(1, 2);
        var z2 = new Complex<double>(3, 4);
        var z3 = new Complex<double>(5, 6);

        var result1 = (z1 + z2) + z3;
        var result2 = z1 + (z2 + z3);

        Assert.Equal(result1.Real, result2.Real, Tolerance);
        Assert.Equal(result1.Imaginary, result2.Imaginary, Tolerance);
    }

    [Fact]
    public void Complex_Multiplication_IsAssociative()
    {
        var z1 = new Complex<double>(1, 2);
        var z2 = new Complex<double>(3, 4);
        var z3 = new Complex<double>(5, 6);

        var result1 = (z1 * z2) * z3;
        var result2 = z1 * (z2 * z3);

        Assert.Equal(result1.Real, result2.Real, Tolerance);
        Assert.Equal(result1.Imaginary, result2.Imaginary, Tolerance);
    }

    [Fact]
    public void Complex_Multiplication_DistributesOverAddition()
    {
        var z1 = new Complex<double>(1, 2);
        var z2 = new Complex<double>(3, 4);
        var z3 = new Complex<double>(5, 6);

        // z1 * (z2 + z3) = z1*z2 + z1*z3
        var left = z1 * (z2 + z3);
        var right = (z1 * z2) + (z1 * z3);

        Assert.Equal(left.Real, right.Real, Tolerance);
        Assert.Equal(left.Imaginary, right.Imaginary, Tolerance);
    }

    [Fact]
    public void Complex_DivisionThenMultiplication_ReturnsOriginal()
    {
        var z1 = new Complex<double>(6, 8);
        var z2 = new Complex<double>(2, 1);

        var divided = z1 / z2;
        var restored = divided * z2;

        Assert.Equal(z1.Real, restored.Real, Tolerance);
        Assert.Equal(z1.Imaginary, restored.Imaginary, Tolerance);
    }

    [Fact]
    public void Complex_ISquared_EqualsMinusOne()
    {
        // i^2 = -1
        var i = new Complex<double>(0, 1);
        var iSquared = i * i;

        Assert.Equal(-1.0, iSquared.Real, Tolerance);
        Assert.Equal(0.0, iSquared.Imaginary, Tolerance);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void Complex_AddZero_ReturnsOriginal()
    {
        var z = new Complex<double>(3, 4);
        var zero = new Complex<double>(0, 0);

        var result = z + zero;

        Assert.Equal(z.Real, result.Real, Tolerance);
        Assert.Equal(z.Imaginary, result.Imaginary, Tolerance);
    }

    [Fact]
    public void Complex_MultiplyByOne_ReturnsOriginal()
    {
        var z = new Complex<double>(3, 4);
        var one = new Complex<double>(1, 0);

        var result = z * one;

        Assert.Equal(z.Real, result.Real, Tolerance);
        Assert.Equal(z.Imaginary, result.Imaginary, Tolerance);
    }

    [Fact]
    public void Complex_MultiplyByI_RotatesBy90Degrees()
    {
        // Multiplying by i rotates by 90 degrees counterclockwise
        var z = new Complex<double>(3, 4);
        var i = new Complex<double>(0, 1);

        var result = z * i;

        // (3 + 4i) * i = 3i + 4i^2 = 3i - 4 = -4 + 3i
        Assert.Equal(-4.0, result.Real, Tolerance);
        Assert.Equal(3.0, result.Imaginary, Tolerance);
    }

    #endregion

    #region Equality

    [Fact]
    public void Complex_Equality_TrueForSameValues()
    {
        var z1 = new Complex<double>(3, 4);
        var z2 = new Complex<double>(3, 4);

        Assert.True(z1 == z2);
        Assert.False(z1 != z2);
    }

    [Fact]
    public void Complex_Equality_FalseForDifferentValues()
    {
        var z1 = new Complex<double>(3, 4);
        var z2 = new Complex<double>(3, 5);

        Assert.False(z1 == z2);
        Assert.True(z1 != z2);
    }

    [Fact]
    public void Complex_GetHashCode_SameForEqualValues()
    {
        var z1 = new Complex<double>(3, 4);
        var z2 = new Complex<double>(3, 4);

        Assert.Equal(z1.GetHashCode(), z2.GetHashCode());
    }

    #endregion

    #region ToString

    [Fact]
    public void Complex_ToString_PositiveImaginary()
    {
        var z = new Complex<double>(3, 4);

        Assert.Equal("3 + 4i", z.ToString());
    }

    [Fact]
    public void Complex_ToString_NegativeImaginary()
    {
        var z = new Complex<double>(3, -4);

        Assert.Equal("3 - 4i", z.ToString());
    }

    #endregion
}
