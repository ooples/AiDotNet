using AiDotNet.CausalDiscovery.TimeSeries;
using AiDotNet.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.CausalDiscovery;

/// <summary>
/// Integration tests for time series causal discovery algorithms.
/// Verifies each algorithm finds meaningful causal structure in strongly correlated data.
/// </summary>
public class TimeSeriesCausalDiscoveryTests
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
    public async Task GrangerCausality_FindsCausalStructure()
    {
        var algo = new GrangerCausalityAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task PCMCI_FindsCausalStructure()
    {
        var algo = new PCMCIAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task PCMCIPlus_FindsCausalStructure()
    {
        var algo = new PCMCIPlusAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task DYNOTEARS_FindsCausalStructure()
    {
        var algo = new DYNOTEARSAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task NTSNOTEARS_FindsCausalStructure()
    {
        var algo = new NTSNOTEARSAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    /// <summary>
    /// Coupled chaotic logistic maps with the chain X0 → X1 → X2 — the
    /// validation system from the CCM paper itself (Sugihara et al. 2012,
    /// "Detecting Causality in Complex Ecosystems", Science — Fig. 1 uses
    /// exactly this two-species coupled logistic model). CCM's causality
    /// criterion is CONVERGENCE: cross-map skill must INCREASE with library
    /// size, which requires a nontrivial attractor to reconstruct. On the
    /// shared noiseless-ramp fixture the shadow manifold is a degenerate 1-D
    /// line whose cross-map skill saturates at the smallest library, so no
    /// convergence can be observed and a paper-faithful CCM correctly reports
    /// no edges.
    /// </summary>
    private static Matrix<double> CreateCoupledLogisticData()
    {
        int n = 300;
        var data = new double[n, 3];
        double x0 = 0.4, x1 = 0.2, x2 = 0.3;
        for (int i = 0; i < n; i++)
        {
            double x0New = x0 * (3.8 - 3.8 * x0);                 // autonomous chaotic driver
            double x1New = x1 * (3.5 - 3.5 * x1 - 0.1 * x0);      // X0 --> X1
            double x2New = x2 * (3.7 - 3.7 * x2 - 0.1 * x1);      // X1 --> X2

            x0 = x0New; x1 = x1New; x2 = x2New;
            data[i, 0] = x0;
            data[i, 1] = x1;
            data[i, 2] = x2;
        }

        return new Matrix<double>(data);
    }

    /// <summary>
    /// Coupled stochastic AR system with lagged drive X0 → X1 → X2. tsFCI
    /// (Entner &amp; Hoyer 2010) runs conditional-independence tests on the
    /// time-lag-expanded variable set; on the noiseless collinear ramp every
    /// partial correlation is a degenerate 0/0 (each series is an exact linear
    /// function of any other), so the CI tests cannot certify any dependence
    /// and a paper-faithful tsFCI correctly returns an empty graph. Seeded
    /// noise + lagged coupling give genuine lagged dependence to detect.
    /// </summary>
    private static Matrix<double> CreateLaggedStochasticData()
    {
        int n = 200;
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        var data = new double[n, 3];
        double x0 = 0.0, x1 = 0.0, x2 = 0.0;
        for (int i = 0; i < n; i++)
        {
            double x0New = 0.6 * x0 + (rng.NextDouble() - 0.5);
            double x1New = 0.5 * x1 + 0.8 * x0 + 0.2 * (rng.NextDouble() - 0.5); // X0 --(lag 1)--> X1
            double x2New = 0.5 * x2 + 0.7 * x1 + 0.2 * (rng.NextDouble() - 0.5); // X1 --(lag 1)--> X2

            x0 = x0New; x1 = x1New; x2 = x2New;
            data[i, 0] = x0;
            data[i, 1] = x1;
            data[i, 2] = x2;
        }

        return new Matrix<double>(data);
    }

    [Fact(Timeout = 120000)]
    public async Task CCM_FindsCausalStructure()
    {
        var algo = new CCMAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateCoupledLogisticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task TSFCI_FindsCausalStructure()
    {
        var algo = new TSFCIAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateLaggedStochasticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task LPCMCI_FindsCausalStructure()
    {
        var algo = new LPCMCIAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task TiMINo_FindsCausalStructure()
    {
        var algo = new TiMINoAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task NeuralGranger_FindsCausalStructure()
    {
        var algo = new NeuralGrangerAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }
}
