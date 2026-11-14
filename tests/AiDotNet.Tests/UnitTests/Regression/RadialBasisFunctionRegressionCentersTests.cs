using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Regression;

public sealed class RadialBasisFunctionRegressionCentersTests
{
    [Fact]
    [Trait("category", "unit")]
    public void UpdateCenters_EmptyClusterBeforeNonEmpty_UsesMeansForReseeding()
    {
        // x = 0, 2, 10, 20; cluster 0 is empty, cluster 1 = {0, 2}, cluster 2 = {10, 20}.
        // Hand-derived means: cluster 1 = (0 + 2) / 2 = 1, cluster 2 = (10 + 20) / 2 = 15.
        // Distances of each point to the nearest of {1, 15}: 1, 1, 5, 5 -> first maximum is x = 10,
        // so the empty cluster 0 is reseeded with 10. (Reseeding against the unaveraged sum 30 of
        // cluster 2 would wrongly pick x = 20.)
        var x = new Matrix<double>(new double[,] { { 0.0 }, { 2.0 }, { 10.0 }, { 20.0 } });
        var model = new RadialBasisFunctionRegression<double>();

        var centers = model.UpdateCenters(x, new[] { 1, 1, 2, 2 }, 3);

        Assert.Equal(3, centers.Rows);
        Assert.Equal(1, centers.Columns);
        Assert.Equal(10.0, centers[0, 0], 12);
        Assert.Equal(1.0, centers[1, 0], 12);
        Assert.Equal(15.0, centers[2, 0], 12);
    }

    [Fact]
    [Trait("category", "unit")]
    public void UpdateCenters_NoEmptyCluster_ReturnsClusterMeans()
    {
        var x = new Matrix<double>(new double[,] { { 0.0, 4.0 }, { 2.0, 6.0 }, { 10.0, 1.0 } });
        var model = new RadialBasisFunctionRegression<double>();

        var centers = model.UpdateCenters(x, new[] { 0, 0, 1 }, 2);

        Assert.Equal(1.0, centers[0, 0], 12);
        Assert.Equal(5.0, centers[0, 1], 12);
        Assert.Equal(10.0, centers[1, 0], 12);
        Assert.Equal(1.0, centers[1, 1], 12);
    }
}
