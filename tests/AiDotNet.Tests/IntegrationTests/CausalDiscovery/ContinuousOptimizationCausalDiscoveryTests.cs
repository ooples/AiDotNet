using AiDotNet.CausalDiscovery.ContinuousOptimization;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CausalDiscovery;

/// <summary>
/// Integration tests for continuous optimization causal discovery algorithms.
/// </summary>
public class ContinuousOptimizationCausalDiscoveryTests
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

    private static void AssertValidGraph(AiDotNet.CausalDiscovery.CausalGraph<double> graph)
    {
        Assert.NotNull(graph);
        Assert.Equal(3, graph.FeatureNames.Length);
        Assert.Equal(3, graph.AdjacencyMatrix.Rows);
        Assert.Equal(3, graph.AdjacencyMatrix.Columns);
    }

    [Fact]
    public void NOTEARSLinear_Discover_ReturnsValidGraph()
    {
        var algo = new NOTEARSLinear<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertValidGraph(graph);
    }

    [Fact]
    public void NOTEARSNonlinear_Discover_ReturnsValidGraph()
    {
        var algo = new NOTEARSNonlinear<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertValidGraph(graph);
    }

    [Fact]
    public void NOTEARSLowRank_Discover_ReturnsValidGraph()
    {
        var algo = new NOTEARSLowRank<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertValidGraph(graph);
    }

    [Fact]
    public void NOTEARSSobolev_Discover_ReturnsValidGraph()
    {
        var algo = new NOTEARSSobolev<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertValidGraph(graph);
    }

    [Fact]
    public void DAGMALinear_Discover_ReturnsValidGraph()
    {
        var algo = new DAGMALinear<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertValidGraph(graph);
    }

    [Fact]
    public void DAGMANonlinear_Discover_ReturnsValidGraph()
    {
        var algo = new DAGMANonlinear<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertValidGraph(graph);
    }

    [Fact]
    public void GOLEM_Discover_ReturnsValidGraph()
    {
        var algo = new GOLEMAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertValidGraph(graph);
    }

    [Fact]
    public void MCSL_Discover_ReturnsValidGraph()
    {
        var algo = new MCSLAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertValidGraph(graph);
    }

    [Fact]
    public void NoCurl_Discover_ReturnsValidGraph()
    {
        var algo = new NoCurlAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertValidGraph(graph);
    }

    [Fact]
    public void CORL_Discover_ReturnsValidGraph()
    {
        var algo = new CORLAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertValidGraph(graph);
    }
}
