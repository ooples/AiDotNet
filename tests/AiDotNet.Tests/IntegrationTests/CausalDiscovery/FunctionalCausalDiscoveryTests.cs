using AiDotNet.CausalDiscovery.Functional;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CausalDiscovery;

/// <summary>
/// Integration tests for functional causal discovery algorithms.
/// </summary>
public class FunctionalCausalDiscoveryTests
{
    private static Matrix<double> CreateSyntheticData()
    {
        int n = 30;
        var data = new double[n, 3];
        for (int i = 0; i < n; i++)
        {
            double x = i * 0.1;
            data[i, 0] = x;
            data[i, 1] = 2.0 * x + 0.5;
            data[i, 2] = x + data[i, 1] * 0.3;
        }

        return new Matrix<double>(data);
    }

    private static readonly string[] FeatureNames = ["X0", "X1", "X2"];

    [Fact]
    public void ANM_Construction_And_Discover()
    {
        var algo = new ANMAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
        Assert.Equal(3, graph.FeatureNames.Length);
        Assert.Equal(3, graph.AdjacencyMatrix.Rows);
        Assert.Equal(3, graph.AdjacencyMatrix.Columns);
    }

    [Fact]
    public void CAM_Construction_And_Discover()
    {
        var algo = new CAMAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
        Assert.Equal(3, graph.FeatureNames.Length);
        Assert.Equal(3, graph.AdjacencyMatrix.Rows);
        Assert.Equal(3, graph.AdjacencyMatrix.Columns);
    }

    [Fact]
    public void CAMUV_Construction_And_Discover()
    {
        var algo = new CAMUVAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
        Assert.Equal(3, graph.FeatureNames.Length);
        Assert.Equal(3, graph.AdjacencyMatrix.Rows);
        Assert.Equal(3, graph.AdjacencyMatrix.Columns);
    }

    [Fact]
    public void DirectLiNGAM_Construction_And_Discover()
    {
        var algo = new DirectLiNGAMAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
        Assert.Equal(3, graph.FeatureNames.Length);
        Assert.Equal(3, graph.AdjacencyMatrix.Rows);
        Assert.Equal(3, graph.AdjacencyMatrix.Columns);
    }

    [Fact]
    public void ICALiNGAM_Construction_And_Discover()
    {
        var algo = new ICALiNGAMAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
        Assert.Equal(3, graph.FeatureNames.Length);
        Assert.Equal(3, graph.AdjacencyMatrix.Rows);
        Assert.Equal(3, graph.AdjacencyMatrix.Columns);
    }

    [Fact]
    public void VARLiNGAM_Construction_And_Discover()
    {
        var algo = new VARLiNGAMAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
        Assert.Equal(3, graph.FeatureNames.Length);
        Assert.Equal(3, graph.AdjacencyMatrix.Rows);
        Assert.Equal(3, graph.AdjacencyMatrix.Columns);
    }

    [Fact]
    public void PNL_Construction_And_Discover()
    {
        var algo = new PNLAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
        Assert.Equal(3, graph.FeatureNames.Length);
        Assert.Equal(3, graph.AdjacencyMatrix.Rows);
        Assert.Equal(3, graph.AdjacencyMatrix.Columns);
    }

    [Fact]
    public void IGCI_Construction_And_Discover()
    {
        var algo = new IGCIAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
        Assert.Equal(3, graph.FeatureNames.Length);
        Assert.Equal(3, graph.AdjacencyMatrix.Rows);
        Assert.Equal(3, graph.AdjacencyMatrix.Columns);
    }

    [Fact]
    public void RCD_Construction_And_Discover()
    {
        var algo = new RCDAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
        Assert.Equal(3, graph.FeatureNames.Length);
        Assert.Equal(3, graph.AdjacencyMatrix.Rows);
        Assert.Equal(3, graph.AdjacencyMatrix.Columns);
    }

    [Fact]
    public void CCDr_Construction_And_Discover()
    {
        var algo = new CCDrAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
        Assert.Equal(3, graph.FeatureNames.Length);
        Assert.Equal(3, graph.AdjacencyMatrix.Rows);
        Assert.Equal(3, graph.AdjacencyMatrix.Columns);
    }
}
