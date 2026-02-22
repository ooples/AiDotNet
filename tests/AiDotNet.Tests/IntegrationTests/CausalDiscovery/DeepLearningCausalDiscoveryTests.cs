using AiDotNet.CausalDiscovery.DeepLearning;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CausalDiscovery;

/// <summary>
/// Integration tests for deep learning causal discovery algorithms.
/// </summary>
public class DeepLearningCausalDiscoveryTests
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
    public void DAGGNN_Construction_And_Discover()
    {
        var algo = new DAGGNNAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }

    [Fact]
    public void GAE_Construction_And_Discover()
    {
        var algo = new GAEAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }

    [Fact]
    public void GraNDAG_Construction_And_Discover()
    {
        var algo = new GraNDAGAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }

    [Fact]
    public void CGNN_Construction_And_Discover()
    {
        var algo = new CGNNAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }

    [Fact]
    public void CASTLE_Construction_And_Discover()
    {
        var algo = new CASTLEAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }

    [Fact]
    public void DECI_Construction_And_Discover()
    {
        var algo = new DECIAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }

    [Fact]
    public void AVICI_Construction_And_Discover()
    {
        var algo = new AVICIAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }

    [Fact]
    public void AmortizedCD_Construction_And_Discover()
    {
        var algo = new AmortizedCDAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }

    [Fact]
    public void CausalVAE_Construction_And_Discover()
    {
        var algo = new CausalVAEAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }

    [Fact]
    public void TCDF_Construction_And_Discover()
    {
        var algo = new TCDFAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }
}
