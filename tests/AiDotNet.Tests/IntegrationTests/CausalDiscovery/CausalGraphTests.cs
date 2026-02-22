using AiDotNet.CausalDiscovery;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CausalDiscovery;

/// <summary>
/// Integration tests for CausalGraph, CausalDiscoveryResult, CausalDiscoverySelector,
/// InterventionalDistribution, and CausalDiscoveryAlgorithmFactory.
/// </summary>
public class CausalGraphTests
{
    private static Matrix<double> CreateSmallAdjacency()
    {
        // 3x3 adjacency: X0 -> X1, X1 -> X2
        return new Matrix<double>(new double[,]
        {
            { 0.0, 1.0, 0.0 },
            { 0.0, 0.0, 1.0 },
            { 0.0, 0.0, 0.0 },
        });
    }

    private static Matrix<double> CreateSyntheticData()
    {
        int n = 20;
        var data = new double[n, 3];
        for (int i = 0; i < n; i++)
        {
            data[i, 0] = i * 0.1;
            data[i, 1] = i * 0.2 + 1.0;
            data[i, 2] = i * 0.3 + 0.5;
        }

        return new Matrix<double>(data);
    }

    #region CausalGraph Tests

    [Fact]
    public void CausalGraph_Construction_WithAdjacencyAndNames()
    {
        var adj = CreateSmallAdjacency();
        var names = new[] { "X0", "X1", "X2" };
        var graph = new CausalGraph<double>(adj, names);
        Assert.NotNull(graph);
        Assert.Equal(3, graph.FeatureNames.Length);
    }

    [Fact]
    public void CausalGraph_FeatureNames_PreservedCorrectly()
    {
        var adj = CreateSmallAdjacency();
        var names = new[] { "Age", "Income", "Score" };
        var graph = new CausalGraph<double>(adj, names);
        Assert.Equal("Age", graph.FeatureNames[0]);
        Assert.Equal("Income", graph.FeatureNames[1]);
        Assert.Equal("Score", graph.FeatureNames[2]);
    }

    #endregion

    #region CausalDiscoverySelector Tests

    [Fact]
    public void CausalDiscoverySelector_Construction()
    {
        var selector = new CausalDiscoverySelector<double>();
        Assert.NotNull(selector);
    }

    #endregion

    #region CausalDiscoveryAlgorithmFactory Tests

    [Fact]
    public void CausalDiscoveryAlgorithmFactory_CreatePC_ReturnsAlgorithm()
    {
        var algorithm = CausalDiscoveryAlgorithmFactory<double>.Create(
            AiDotNet.Enums.CausalDiscoveryAlgorithmType.PC);
        Assert.NotNull(algorithm);
    }

    [Fact]
    public void CausalDiscoveryAlgorithmFactory_CreateGES_ReturnsAlgorithm()
    {
        var algorithm = CausalDiscoveryAlgorithmFactory<double>.Create(
            AiDotNet.Enums.CausalDiscoveryAlgorithmType.GES);
        Assert.NotNull(algorithm);
    }

    [Fact]
    public void CausalDiscoveryAlgorithmFactory_Create_And_Discover()
    {
        var algorithm = CausalDiscoveryAlgorithmFactory<double>.Create(
            AiDotNet.Enums.CausalDiscoveryAlgorithmType.PC);
        var graph = algorithm.DiscoverStructure(CreateSyntheticData(), new[] { "X0", "X1", "X2" });
        Assert.NotNull(graph);
        Assert.Equal(3, graph.FeatureNames.Length);
    }

    #endregion
}
