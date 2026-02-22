using AiDotNet.CausalDiscovery.ConstraintBased;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CausalDiscovery;

/// <summary>
/// Integration tests for constraint-based causal discovery algorithms.
/// </summary>
public class ConstraintBasedCausalDiscoveryTests
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
        Assert.Equal("X0", graph.FeatureNames[0]);
        Assert.Equal("X1", graph.FeatureNames[1]);
        Assert.Equal("X2", graph.FeatureNames[2]);
        Assert.Equal(3, graph.AdjacencyMatrix.Rows);
        Assert.Equal(3, graph.AdjacencyMatrix.Columns);
    }

    [Fact]
    public void PC_Discover_ReturnsValidGraph()
    {
        var algo = new PCAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertValidGraph(graph);
    }

    [Fact]
    public void FCI_Discover_ReturnsValidGraph()
    {
        var algo = new FCIAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertValidGraph(graph);
    }

    [Fact]
    public void RFCI_Discover_ReturnsValidGraph()
    {
        var algo = new RFCIAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertValidGraph(graph);
    }

    [Fact]
    public void CPC_Discover_ReturnsValidGraph()
    {
        var algo = new CPCAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertValidGraph(graph);
    }

    [Fact]
    public void MMPC_Discover_ReturnsValidGraph()
    {
        var algo = new MMPCAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertValidGraph(graph);
    }

    [Fact]
    public void IAMB_Discover_ReturnsValidGraph()
    {
        var algo = new IAMBAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertValidGraph(graph);
    }

    [Fact]
    public void FastIAMB_Discover_ReturnsValidGraph()
    {
        var algo = new FastIAMBAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertValidGraph(graph);
    }

    [Fact]
    public void MarkovBlanket_Discover_ReturnsValidGraph()
    {
        var algo = new MarkovBlanketAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertValidGraph(graph);
    }

    [Fact]
    public void CDNOD_Discover_ReturnsValidGraph()
    {
        var algo = new CDNODAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        AssertValidGraph(graph);
    }
}
