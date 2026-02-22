using AiDotNet.CausalDiscovery.Specialized;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CausalDiscovery;

/// <summary>
/// Integration tests for specialized causal discovery algorithms.
/// </summary>
public class SpecializedCausalDiscoveryTests
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
    public void GOBNILP_Construction_And_Discover()
    {
        var algo = new GOBNILPAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        Assert.NotNull(graph);
    }
}
