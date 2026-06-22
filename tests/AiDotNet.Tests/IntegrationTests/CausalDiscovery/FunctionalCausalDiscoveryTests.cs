using AiDotNet.CausalDiscovery.Functional;
using AiDotNet.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.CausalDiscovery;

/// <summary>
/// Integration tests for functional causal discovery algorithms.
/// Verifies each algorithm finds meaningful causal structure in strongly correlated data.
/// </summary>
public class FunctionalCausalDiscoveryTests
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
    public async Task ANM_FindsCausalStructure()
    {
        var algo = new ANMAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task CAM_FindsCausalStructure()
    {
        var algo = new CAMAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task CAMUV_FindsCausalStructure()
    {
        var algo = new CAMUVAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task DirectLiNGAM_FindsCausalStructure()
    {
        var algo = new DirectLiNGAMAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task ICALiNGAM_FindsCausalStructure()
    {
        var algo = new ICALiNGAMAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task VARLiNGAM_FindsCausalStructure()
    {
        var algo = new VARLiNGAMAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task PNL_FindsCausalStructure()
    {
        var algo = new PNLAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    /// <summary>
    /// Deterministic NONLINEAR invertible chain X0 → X1 → X2. IGCI (Janzing
    /// et al. 2012, "Information-geometric approach to inferring causal
    /// directions", §2) infers direction from the correlation between log|f′|
    /// and the input density — for a LINEAR f the derivative is constant, the
    /// criterion is exactly zero in both directions, and the method is
    /// explicitly unidentifiable (the paper's stated boundary case). The
    /// shared ramp fixture is linear, so a paper-faithful IGCI correctly
    /// reports no direction there. Smooth nonlinear monotone maps are IGCI's
    /// designed regime.
    /// </summary>
    private static Matrix<double> CreateNonlinearDeterministicData()
    {
        int n = 100;
        var data = new double[n, 3];
        for (int i = 0; i < n; i++)
        {
            double x0 = 0.05 + 0.9 * i / (n - 1);   // uniform-ish on [0.05, 0.95]
            double x1 = x0 * x0 * x0;               // nonlinear invertible: f(x) = x^3
            double x2 = Math.Tanh(3.0 * x1);        // nonlinear invertible: g(x) = tanh(3x)

            data[i, 0] = x0;
            data[i, 1] = x1;
            data[i, 2] = x2;
        }

        return new Matrix<double>(data);
    }

    [Fact(Timeout = 120000)]
    public async Task IGCI_FindsCausalStructure()
    {
        var algo = new IGCIAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateNonlinearDeterministicData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task RCD_FindsCausalStructure()
    {
        var algo = new RCDAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task CCDr_FindsCausalStructure()
    {
        var algo = new CCDrAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }
}
