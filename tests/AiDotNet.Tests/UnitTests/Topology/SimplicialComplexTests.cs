using AiDotNet.Tensors.Topology;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Topology;

public class SimplicialComplexTests
{
    [Fact]
    public void AddSimplex_IncludesFaces()
    {
        var complex = new SimplicialComplex();
        complex.AddSimplex(new Simplex(new[] { 0, 1, 2 }));

        Assert.Equal(3, complex.GetSimplices(0).Count);
        Assert.Equal(3, complex.GetSimplices(1).Count);
        Assert.Single(complex.GetSimplices(2));
    }

    [Fact]
    public void BoundaryOfBoundary_IsZero()
    {
        var complex = new SimplicialComplex();
        complex.AddSimplex(new Simplex(new[] { 0, 1, 2 }));

        var b1 = complex.BoundaryOperator<double>(1);
        var b2 = complex.BoundaryOperator<double>(2);
        var product = (AiDotNet.Tensors.LinearAlgebra.Matrix<double>)b1.Multiply(b2);

        for (int i = 0; i < product.Rows; i++)
        {
            for (int j = 0; j < product.Columns; j++)
            {
                Assert.Equal(0.0, product[i, j], precision: 12);
            }
        }
    }

    [Fact]
    public void HodgeLaplacian_HasExpectedShape()
    {
        var complex = new SimplicialComplex();
        complex.AddSimplex(new Simplex(new[] { 0, 1, 2 }));

        var laplacian = complex.HodgeLaplacian<double>(1);

        Assert.Equal(3, laplacian.Rows);
        Assert.Equal(3, laplacian.Columns);
    }
}
