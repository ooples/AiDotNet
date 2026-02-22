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

    [Fact]
    public void PC_Construction_And_Discover()
    {
        var algo = new PCAlgorithm<double>();
        Assert.NotNull(algo);
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
        Assert.Equal(3, graph.FeatureNames.Length);
    }

    [Fact]
    public void FCI_Construction_And_Discover()
    {
        var algo = new FCIAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }

    [Fact]
    public void RFCI_Construction_And_Discover()
    {
        var algo = new RFCIAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }

    [Fact]
    public void CPC_Construction_And_Discover()
    {
        var algo = new CPCAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }

    [Fact]
    public void MMPC_Construction_And_Discover()
    {
        var algo = new MMPCAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }

    [Fact]
    public void IAMB_Construction_And_Discover()
    {
        var algo = new IAMBAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }

    [Fact]
    public void FastIAMB_Construction_And_Discover()
    {
        var algo = new FastIAMBAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }

    [Fact]
    public void MarkovBlanket_Construction_And_Discover()
    {
        var algo = new MarkovBlanketAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }

    [Fact]
    public void CDNOD_Construction_And_Discover()
    {
        var algo = new CDNODAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }
}
