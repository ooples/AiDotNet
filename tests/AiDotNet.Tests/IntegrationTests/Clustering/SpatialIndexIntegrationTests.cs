using AiDotNet.Clustering.SpatialIndex;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Clustering;

public class SpatialIndexIntegrationTests
{
    [Fact]
    public void KDTree_BuildAndQueryNearest_ReturnsExpectedNeighbors()
    {
        var data = CreateSimpleData();
        var tree = new KDTree<double>();
        tree.Build(data);

        var query = CreateQueryPoint();
        var results = tree.QueryKNearest(query, 2);

        Assert.Equal(2, results.Length);
        Assert.Equal(0, results[0].Index);
        Assert.Equal(1, results[1].Index);
        Assert.True(results[0].Distance <= results[1].Distance);
    }

    [Fact]
    public void KDTree_QueryRadius_FindsExpectedPoints()
    {
        var data = CreateSimpleData();
        var tree = new KDTree<double>();
        tree.Build(data);

        var query = CreateQueryPoint();
        var results = tree.QueryRadius(query, 1.0);

        var indices = new HashSet<int>();
        foreach (var result in results)
        {
            indices.Add(result.Index);
        }

        Assert.Equal(3, indices.Count);
        Assert.Contains(0, indices);
        Assert.Contains(1, indices);
        Assert.Contains(2, indices);
    }

    [Fact]
    public void KDTree_QueryRadiusIndices_MatchesRadiusQuery()
    {
        var data = CreateSimpleData();
        var tree = new KDTree<double>();
        tree.Build(data);

        var query = CreateQueryPoint();
        var radiusResults = tree.QueryRadius(query, 1.0);
        var radiusIndices = tree.QueryRadiusIndices(query, 1.0);

        var radiusSet = new HashSet<int>();
        foreach (var result in radiusResults)
        {
            radiusSet.Add(result.Index);
        }

        var indexSet = new HashSet<int>(radiusIndices);
        Assert.Equal(radiusSet.Count, indexSet.Count);
        foreach (int idx in radiusSet)
        {
            Assert.Contains(idx, indexSet);
        }
    }

    [Fact]
    public void KDTree_QueryBeforeBuild_Throws()
    {
        var tree = new KDTree<double>();
        var query = CreateQueryPoint();

        Assert.Throws<InvalidOperationException>(() => tree.QueryKNearest(query, 1));
        Assert.Throws<InvalidOperationException>(() => tree.QueryRadius(query, 1.0));
    }

    [Fact]
    public void BallTree_BuildAndQueryNearest_ReturnsExpectedNeighbors()
    {
        var data = CreateSimpleData();
        var tree = new BallTree<double>();
        tree.Build(data);

        var query = CreateQueryPoint();
        var results = tree.QueryKNearest(query, 2);

        Assert.Equal(2, results.Length);
        Assert.Equal(0, results[0].Index);
        Assert.Equal(1, results[1].Index);
        Assert.True(results[0].Distance <= results[1].Distance);
    }

    [Fact]
    public void BallTree_QueryRadius_FindsExpectedPoints()
    {
        var data = CreateSimpleData();
        var tree = new BallTree<double>();
        tree.Build(data);

        var query = CreateQueryPoint();
        var results = tree.QueryRadius(query, 1.0);

        var indices = new HashSet<int>();
        foreach (var result in results)
        {
            indices.Add(result.Index);
        }

        Assert.Equal(3, indices.Count);
        Assert.Contains(0, indices);
        Assert.Contains(1, indices);
        Assert.Contains(2, indices);
    }

    [Fact]
    public void BallTree_QueryBeforeBuild_Throws()
    {
        var tree = new BallTree<double>();
        var query = CreateQueryPoint();

        Assert.Throws<InvalidOperationException>(() => tree.QueryKNearest(query, 1));
        Assert.Throws<InvalidOperationException>(() => tree.QueryRadius(query, 1.0));
    }

    private static Matrix<double> CreateSimpleData()
    {
        var data = new Matrix<double>(6, 2);
        data[0, 0] = 0.0; data[0, 1] = 0.0;
        data[1, 0] = 1.0; data[1, 1] = 0.0;
        data[2, 0] = 0.0; data[2, 1] = 1.0;
        data[3, 0] = 5.0; data[3, 1] = 5.0;
        data[4, 0] = 5.0; data[4, 1] = 6.0;
        data[5, 0] = 6.0; data[5, 1] = 5.0;
        return data;
    }

    private static Vector<double> CreateQueryPoint()
    {
        return new Vector<double>(new[] { 0.2, 0.1 });
    }
}
