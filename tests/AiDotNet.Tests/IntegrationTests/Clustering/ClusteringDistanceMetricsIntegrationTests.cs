using AiDotNet.Clustering.DistanceMetrics;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Clustering;

public class ClusteringDistanceMetricsIntegrationTests
{
    [Fact]
    public void ChebyshevDistance_ComputesMaxDifference()
    {
        var metric = new ChebyshevDistance<double>();
        var a = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var b = new Vector<double>(new[] { 3.0, 2.0, 1.0 });

        double distance = metric.Compute(a, b);

        Assert.Equal(2.0, distance, 1e-6);
    }

    [Fact]
    public void CosineDistance_HandlesOrthogonalAndZeroVectors()
    {
        var metric = new CosineDistance<double>();
        var a = new Vector<double>(new[] { 1.0, 0.0 });
        var b = new Vector<double>(new[] { 0.0, 1.0 });

        double orthogonal = metric.Compute(a, b);
        Assert.Equal(1.0, orthogonal, 1e-6);

        var zero = new Vector<double>(new[] { 0.0, 0.0 });
        double zeroDistance = metric.Compute(a, zero);
        Assert.Equal(1.0, zeroDistance, 1e-6);

        double similarity = metric.ComputeSimilarity(a, a);
        Assert.Equal(1.0, similarity, 1e-6);
    }

    [Fact]
    public void MinkowskiDistance_ComputesExpectedValuesAndValidatesP()
    {
        var a = new Vector<double>(new[] { 1.0, 2.0 });
        var b = new Vector<double>(new[] { 3.0, 4.0 });

        var manhattan = new MinkowskiDistance<double>(1.0);
        Assert.Equal(4.0, manhattan.Compute(a, b), 1e-6);

        var euclidean = new MinkowskiDistance<double>(2.0);
        Assert.Equal(Math.Sqrt(8.0), euclidean.Compute(a, b), 1e-6);

        var p3 = new MinkowskiDistance<double>(3.0);
        double p3Distance = p3.Compute(a, b);
        Assert.True(p3Distance > 0);
        Assert.Equal(3.0, p3.P, 1e-6);

        Assert.Throws<ArgumentException>(() => new MinkowskiDistance<double>(0.5));
    }

    [Fact]
    public void MahalanobisDistance_FallsBackToEuclideanAndFitsFromData()
    {
        var a = new Vector<double>(new[] { 1.0, 2.0 });
        var b = new Vector<double>(new[] { 4.0, 6.0 });

        var metric = new MahalanobisDistance<double>();
        double fallback = metric.Compute(a, b);
        Assert.Equal(5.0, fallback, 1e-6);

        var data = new Matrix<double>(4, 2);
        data[0, 0] = 0.0; data[0, 1] = 0.0;
        data[1, 0] = 1.0; data[1, 1] = 1.0;
        data[2, 0] = 2.0; data[2, 1] = 2.0;
        data[3, 0] = 3.0; data[3, 1] = 3.0;

        metric.FitFromData(data);
        double fitted = metric.Compute(a, b);
        Assert.True(fitted > 0);
    }

    [Fact]
    public void DistanceMetricBase_ComputeToAllAndPairwiseWork()
    {
        var metric = new EuclideanDistance<double>();
        var data = new Matrix<double>(3, 2);
        data[0, 0] = 0.0; data[0, 1] = 0.0;
        data[1, 0] = 1.0; data[1, 1] = 0.0;
        data[2, 0] = 0.0; data[2, 1] = 1.0;

        var point = new Vector<double>(new[] { 0.0, 0.0 });
        var distances = metric.ComputeToAll(point, data);

        Assert.Equal(3, distances.Length);
        Assert.Equal(0.0, distances[0], 1e-6);

        var pairwise = metric.ComputePairwise(data);
        Assert.Equal(3, pairwise.Rows);
        Assert.Equal(3, pairwise.Columns);
        Assert.Equal(0.0, pairwise[0, 0], 1e-6);
        Assert.Equal(pairwise[0, 1], pairwise[1, 0], 1e-6);

        var other = new Matrix<double>(2, 2);
        other[0, 0] = 2.0; other[0, 1] = 0.0;
        other[1, 0] = 0.0; other[1, 1] = 2.0;

        var cross = metric.ComputePairwise(data, other);
        Assert.Equal(3, cross.Rows);
        Assert.Equal(2, cross.Columns);

        Assert.Throws<ArgumentException>(() => metric.ComputePairwise(data, new Matrix<double>(1, 3)));
    }
}
