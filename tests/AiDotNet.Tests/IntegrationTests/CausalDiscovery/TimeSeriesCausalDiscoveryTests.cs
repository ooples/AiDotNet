using AiDotNet.CausalDiscovery.TimeSeries;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CausalDiscovery;

/// <summary>
/// Integration tests for time series causal discovery algorithms.
/// </summary>
public class TimeSeriesCausalDiscoveryTests
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
    public void GrangerCausality_Construction_And_Discover()
    {
        var algo = new GrangerCausalityAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }

    [Fact]
    public void PCMCI_Construction_And_Discover()
    {
        var algo = new PCMCIAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }

    [Fact]
    public void PCMCIPlus_Construction_And_Discover()
    {
        var algo = new PCMCIPlusAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }

    [Fact]
    public void DYNOTEARS_Construction_And_Discover()
    {
        var algo = new DYNOTEARSAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }

    [Fact]
    public void NTSNOTEARS_Construction_And_Discover()
    {
        var algo = new NTSNOTEARSAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }

    [Fact]
    public void CCM_Construction_And_Discover()
    {
        var algo = new CCMAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }

    [Fact]
    public void TSFCI_Construction_And_Discover()
    {
        var algo = new TSFCIAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }

    [Fact]
    public void LPCMCI_Construction_And_Discover()
    {
        var algo = new LPCMCIAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }

    [Fact]
    public void TiMINo_Construction_And_Discover()
    {
        var algo = new TiMINoAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }

    [Fact]
    public void NeuralGranger_Construction_And_Discover()
    {
        var algo = new NeuralGrangerAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }
}
