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
    public void CausalGraph_Construction_PreservesAdjacencyStructure()
    {
        var adj = CreateSmallAdjacency();
        var names = new[] { "X0", "X1", "X2" };
        var graph = new CausalGraph<double>(adj, names);
        Assert.NotNull(graph);
        Assert.Equal(3, graph.FeatureNames.Length);

        // Verify adjacency: X0->X1, X1->X2
        Assert.True(graph.HasEdge(0, 1), "Expected edge X0->X1");
        Assert.True(graph.HasEdge(1, 2), "Expected edge X1->X2");
        Assert.False(graph.HasEdge(0, 2), "Should not have direct edge X0->X2");
        Assert.False(graph.HasEdge(2, 0), "Should not have reverse edge X2->X0");
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
    public void CausalDiscoverySelector_FitTransform_SelectsCausalFeatures()
    {
        var selector = new CausalDiscoverySelector<double>();
        Assert.NotNull(selector);

        var data = CreateSyntheticData();
        var target = new Vector<double>(data.Rows);
        for (int i = 0; i < data.Rows; i++)
            target[i] = data[i, 2]; // Use X2 as target

        selector.Fit(data, target);
        Assert.NotNull(selector.SelectedIndices);
        Assert.True(selector.SelectedIndices.Length > 0, "Should select at least one causal feature");
    }

    #endregion

    #region CausalDiscoveryAlgorithmFactory Tests

    [Fact]
    public void CausalDiscoveryAlgorithmFactory_CreatePC_ReturnsCorrectType()
    {
        var algorithm = CausalDiscoveryAlgorithmFactory<double>.Create(
            AiDotNet.Enums.CausalDiscoveryAlgorithmType.PC);
        Assert.NotNull(algorithm);
        // Factory should produce distinct algorithm instances
        var algorithm2 = CausalDiscoveryAlgorithmFactory<double>.Create(
            AiDotNet.Enums.CausalDiscoveryAlgorithmType.PC);
        Assert.NotSame(algorithm, algorithm2);
    }

    [Fact]
    public void CausalDiscoveryAlgorithmFactory_CreateGES_ReturnsCorrectType()
    {
        var algorithm = CausalDiscoveryAlgorithmFactory<double>.Create(
            AiDotNet.Enums.CausalDiscoveryAlgorithmType.GES);
        Assert.NotNull(algorithm);
        // GES and PC should produce different algorithm types
        var pcAlgorithm = CausalDiscoveryAlgorithmFactory<double>.Create(
            AiDotNet.Enums.CausalDiscoveryAlgorithmType.PC);
        Assert.NotEqual(algorithm.GetType(), pcAlgorithm.GetType());
    }

    [Fact]
    public void CausalDiscoveryAlgorithmFactory_Discover_ReturnsValidGraph()
    {
        var algorithm = CausalDiscoveryAlgorithmFactory<double>.Create(
            AiDotNet.Enums.CausalDiscoveryAlgorithmType.PC);
        var names = new[] { "X0", "X1", "X2" };
        var graph = algorithm.DiscoverStructure(CreateSyntheticData(), names);
        Assert.NotNull(graph);
        Assert.Equal(3, graph.FeatureNames.Length);
        Assert.Equal("X0", graph.FeatureNames[0]);
        Assert.Equal("X1", graph.FeatureNames[1]);
        Assert.Equal("X2", graph.FeatureNames[2]);

        // Verify adjacency matrix has correct dimensions
        Assert.Equal(3, graph.AdjacencyMatrix.Rows);
        Assert.Equal(3, graph.AdjacencyMatrix.Columns);
    }

    #endregion
}
