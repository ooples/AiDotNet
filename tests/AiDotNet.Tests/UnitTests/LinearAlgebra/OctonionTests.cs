using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.LinearAlgebra;

public class OctonionTests
{
    [Fact]
    public void Addition_AddsComponents()
    {
        var a = new Octonion<double>(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        var b = new Octonion<double>(0.5, -1.0, 0.0, 1.0, -2.0, 0.0, 0.0, 1.5);

        var sum = a + b;

        Assert.Equal(1.5, sum.Scalar, precision: 12);
        Assert.Equal(1.0, sum.E1, precision: 12);
        Assert.Equal(3.0, sum.E2, precision: 12);
        Assert.Equal(5.0, sum.E3, precision: 12);
        Assert.Equal(3.0, sum.E4, precision: 12);
        Assert.Equal(6.0, sum.E5, precision: 12);
        Assert.Equal(7.0, sum.E6, precision: 12);
        Assert.Equal(9.5, sum.E7, precision: 12);
    }

    [Fact]
    public void Multiplication_UsesStandardBasisRules()
    {
        var e1 = new Octonion<double>(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        var e2 = new Octonion<double>(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        var e3 = new Octonion<double>(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);

        AssertOctonionClose(e1 * e2, e3, 12);
        AssertOctonionClose(e2 * e1, new Octonion<double>(0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0), 12);
    }

    [Fact]
    public void Multiplication_IsNotAssociative()
    {
        var e1 = new Octonion<double>(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        var e2 = new Octonion<double>(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        var e4 = new Octonion<double>(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0);

        var left = (e1 * e2) * e4;
        var right = e1 * (e2 * e4);

        Assert.False(left.Equals(right));
    }

    [Fact]
    public void Conjugate_ProductIsScalarNormSquared()
    {
        var value = new Octonion<double>(1.5, -2.0, 0.5, 0.0, 1.0, -0.25, 0.0, 2.0);

        var product = value * value.Conjugate();

        Assert.Equal(value.NormSquared, product.Scalar, precision: 10);
        Assert.Equal(0.0, product.E1, precision: 10);
        Assert.Equal(0.0, product.E2, precision: 10);
        Assert.Equal(0.0, product.E3, precision: 10);
        Assert.Equal(0.0, product.E4, precision: 10);
        Assert.Equal(0.0, product.E5, precision: 10);
        Assert.Equal(0.0, product.E6, precision: 10);
        Assert.Equal(0.0, product.E7, precision: 10);
    }

    [Fact]
    public void Inverse_MultipliesToIdentity()
    {
        var value = new Octonion<double>(2.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0);

        var product = value * value.Inverse();

        Assert.Equal(1.0, product.Scalar, precision: 10);
        Assert.Equal(0.0, product.E1, precision: 10);
        Assert.Equal(0.0, product.E2, precision: 10);
        Assert.Equal(0.0, product.E3, precision: 10);
        Assert.Equal(0.0, product.E4, precision: 10);
        Assert.Equal(0.0, product.E5, precision: 10);
        Assert.Equal(0.0, product.E6, precision: 10);
        Assert.Equal(0.0, product.E7, precision: 10);
    }

    private static void AssertOctonionClose(Octonion<double> actual, Octonion<double> expected, int precision)
    {
        Assert.Equal(expected.Scalar, actual.Scalar, precision);
        Assert.Equal(expected.E1, actual.E1, precision);
        Assert.Equal(expected.E2, actual.E2, precision);
        Assert.Equal(expected.E3, actual.E3, precision);
        Assert.Equal(expected.E4, actual.E4, precision);
        Assert.Equal(expected.E5, actual.E5, precision);
        Assert.Equal(expected.E6, actual.E6, precision);
        Assert.Equal(expected.E7, actual.E7, precision);
    }
}
