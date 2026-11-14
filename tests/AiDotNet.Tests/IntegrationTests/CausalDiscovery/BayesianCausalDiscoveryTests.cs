using AiDotNet.CausalDiscovery.Bayesian;
using AiDotNet.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.CausalDiscovery;

/// <summary>
/// Integration tests for Bayesian causal discovery algorithms.
/// Verifies each algorithm finds meaningful causal structure in strongly correlated data.
/// </summary>
public class BayesianCausalDiscoveryTests
{
    private static Matrix<double> CreateSyntheticData()
    {
        int n = 50;
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

    [Fact(Timeout = 120000)]
    public async Task OrderMCMC_FindsCausalStructure()
    {
        var algo = new OrderMCMCAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task PartitionMCMC_FindsCausalStructure()
    {
        var algo = new PartitionMCMCAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task IterativeMCMC_FindsCausalStructure()
    {
        var algo = new IterativeMCMCAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task BayesDAG_FindsCausalStructure()
    {
        var algo = new BayesDAGAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task DiBS_FindsCausalStructure()
    {
        var algo = new DiBSAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact]
    public void DiBS_SvgdKernelGradient_PushesParticlesApart()
    {
        // Two particles Z_0 = 0 and Z_1 = 1 in one dimension with kernel scale c = 2:
        // k(Z_1, Z_0) = exp(-1/2). The SVGD repulsive term for particle k is
        // sum_l grad_{Z_l} k(Z_l, Z_k) = sum_l k(Z_l, Z_k) * 2 (Z_k - Z_l) / c (Liu & Wang 2016, eq. 8),
        // which is -exp(-1/2) for particle 0 and +exp(-1/2) for particle 1: each moves away from the other.
        double e = System.Math.Exp(-0.5);
        var kernel = new AiDotNet.Tensors.LinearAlgebra.Tensor<double>(new[] { 1.0, e, e, 1.0 }, new[] { 2, 2 });
        var particles = new AiDotNet.Tensors.LinearAlgebra.Tensor<double>(new[] { 0.0, 1.0 }, new[] { 2, 1 });

        var repulsive = DiBSAlgorithm<double>.SvgdKernelGradient(kernel, particles, 2.0);

        Assert.Equal(-e, repulsive[0, 0], 12);
        Assert.Equal(e, repulsive[1, 0], 12);
    }

    [Fact(Timeout = 120000)]
    public async Task BCDNets_FindsCausalStructure()
    {
        var algo = new BCDNetsAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }
}
