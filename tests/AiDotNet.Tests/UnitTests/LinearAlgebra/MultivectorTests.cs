using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.LinearAlgebra;

public class MultivectorTests
{
    [Fact]
    public void GeometricProduct_BasisVectorsFollowAnticommutation()
    {
        var algebra = new CliffordAlgebra(3, 0, 0);
        var e1 = BasisVector(algebra, 0);
        var e2 = BasisVector(algebra, 1);

        var e12 = e1 * e2;
        AssertBlade(e12, 1 | 2, 1.0, 12);

        var e21 = e2 * e1;
        AssertBlade(e21, 1 | 2, -1.0, 12);

        var e1Squared = e1 * e1;
        AssertBlade(e1Squared, 0, 1.0, 12);
    }

    [Fact]
    public void OuterProduct_DisjointVectorsProduceBlade()
    {
        var algebra = new CliffordAlgebra(3, 0, 0);
        var e1 = BasisVector(algebra, 0);
        var e2 = BasisVector(algebra, 1);

        var wedge = e1.OuterProduct(e2);
        AssertBlade(wedge, 1 | 2, 1.0, 12);

        var zero = e1.OuterProduct(e1);
        Assert.True(zero.IsZero);
    }

    [Fact]
    public void InnerProduct_GradeSelectionMatchesLeftContraction()
    {
        var algebra = new CliffordAlgebra(3, 0, 0);
        var e1 = BasisVector(algebra, 0);
        var e12 = BasisBlade(algebra, 1 | 2);

        var result = e1.InnerProduct(e12);

        AssertBlade(result, 2, 1.0, 12);
    }

    [Fact]
    public void Reverse_NegatesBivector()
    {
        var algebra = new CliffordAlgebra(3, 0, 0);
        var e12 = BasisBlade(algebra, 1 | 2);

        var reversed = e12.Reverse();

        AssertBlade(reversed, 1 | 2, -1.0, 12);
    }

    [Fact]
    public void MetricSignature_ChangesSquareSign()
    {
        var algebra = new CliffordAlgebra(1, 1, 0);
        var e1 = BasisVector(algebra, 0);
        var e2 = BasisVector(algebra, 1);

        AssertBlade(e1 * e1, 0, 1.0, 12);
        AssertBlade(e2 * e2, 0, -1.0, 12);
    }

    [Fact]
    public void Inverse_VectorMultipliesToIdentity()
    {
        var algebra = new CliffordAlgebra(3, 0, 0);
        var e1 = BasisVector(algebra, 0);

        var product = e1 * e1.Inverse();

        AssertBlade(product, 0, 1.0, 12);
    }

    private static Multivector<double> BasisVector(CliffordAlgebra algebra, int basisIndex)
    {
        var coefficients = new Dictionary<int, double> { { 1 << basisIndex, 1.0 } };
        return new Multivector<double>(algebra, coefficients);
    }

    private static Multivector<double> BasisBlade(CliffordAlgebra algebra, int blade)
    {
        var coefficients = new Dictionary<int, double> { { blade, 1.0 } };
        return new Multivector<double>(algebra, coefficients);
    }

    private static void AssertBlade(Multivector<double> value, int blade, double expected, int precision)
    {
        for (int i = 0; i < value.BasisCount; i++)
        {
            double target = i == blade ? expected : 0.0;
            Assert.Equal(target, value[i], precision);
        }
    }
}
