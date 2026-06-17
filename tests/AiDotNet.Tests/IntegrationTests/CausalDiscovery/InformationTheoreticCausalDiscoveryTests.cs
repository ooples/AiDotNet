using AiDotNet.CausalDiscovery.InformationTheoretic;
using AiDotNet.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.CausalDiscovery;

/// <summary>
/// Integration tests for information-theoretic causal discovery algorithms.
/// Verifies each algorithm finds meaningful causal structure in strongly correlated data.
/// </summary>
public class InformationTheoreticCausalDiscoveryTests
{
    /// <summary>
    /// Coupled stochastic autoregressive system with lagged causal drive
    /// X0 → X1 → X2 — the canonical validation setup for information-flow
    /// estimators (Schreiber 2000 §IV; Sun et al. 2015 §5 use coupled
    /// stochastic processes). The previous fixture was a noiseless straight
    /// line (X0 = 0.1·i with exact linear functions of it), on which transfer
    /// entropy and causation entropy are EXACTLY ZERO by definition: a ramp is
    /// perfectly self-predictable (Y[t] = 2Y[t-1] − Y[t-2]) and its transitions
    /// are constant, so there is no uncertainty for a source to reduce. The
    /// paper-faithful algorithms correctly found no edges there, failing the
    /// ≥1-edge assertion. Seeded noise + lagged coupling gives genuine directed
    /// information flow while keeping the test deterministic.
    /// </summary>
    private static Matrix<double> CreateSyntheticData()
    {
        int n = 200;
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        var data = new double[n, 3];

        double x0 = 0.0, x1 = 0.0, x2 = 0.0;
        for (int i = 0; i < n; i++)
        {
            double e0 = rng.NextDouble() - 0.5;
            double e1 = 0.2 * (rng.NextDouble() - 0.5);
            double e2 = 0.2 * (rng.NextDouble() - 0.5);

            double x0New = 0.6 * x0 + e0;
            double x1New = 0.5 * x1 + 0.8 * x0 + e1; // X0 --(lag 1)--> X1
            double x2New = 0.5 * x2 + 0.7 * x1 + e2; // X1 --(lag 1)--> X2

            x0 = x0New; x1 = x1New; x2 = x2New;
            data[i, 0] = x0;
            data[i, 1] = x1;
            data[i, 2] = x2;
        }

        return new Matrix<double>(data);
    }

    private static readonly string[] FeatureNames = ["X0", "X1", "X2"];

    [Fact(Timeout = 120000)]
    public async Task TransferEntropy_FindsCausalStructure()
    {
        var algo = new TransferEntropyAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task KraskovMI_FindsCausalStructure()
    {
        var algo = new KraskovMIAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task OCSE_FindsCausalStructure()
    {
        var algo = new OCSEAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }
}
