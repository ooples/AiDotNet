using AiDotNet.CausalDiscovery.DeepLearning;
using AiDotNet.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.CausalDiscovery;

/// <summary>
/// Integration tests for deep learning causal discovery algorithms.
/// Verifies each algorithm finds meaningful causal structure in strongly correlated data.
/// </summary>
public class DeepLearningCausalDiscoveryTests
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

    private static Matrix<double> CreateCgnnDagRegressionData()
    {
        const int n = 200;
        var rng = new Random(42);
        var data = new Matrix<double>(n, 4);
        for (int i = 0; i < n; i++)
        {
            double x0 = rng.NextDouble() * 2.0 - 1.0;
            double x1 = 0.8 * x0 + (rng.NextDouble() * 0.2 - 0.1);
            double x2 = 0.6 * x1 + (rng.NextDouble() * 0.2 - 0.1);
            double x3 = -0.7 * x0 + (rng.NextDouble() * 0.2 - 0.1);

            data[i, 0] = x0;
            data[i, 1] = x1;
            data[i, 2] = x2;
            data[i, 3] = x3;
        }

        return data;
    }

    private static readonly string[] CgnnFeatureNames = ["X0", "X1", "X2", "X3"];

    [Fact(Timeout = 120000)]
    public async Task DAGGNN_FindsCausalStructure()
    {
        var algo = new DAGGNNAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task GAE_FindsCausalStructure()
    {
        var algo = new GAEAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    /// <summary>
    /// Noisy i.i.d. SEM with the same chain (X0 → X1, X0/X1 → X2). GraN-DAG
    /// (Lachapelle et al. 2020, §2) fits a Gaussian likelihood with per-node
    /// noise variances; the noiseless rank-1 ramp puts that likelihood in its
    /// degenerate zero-variance limit (density → ∞), so the score surface
    /// carries no usable signal and a paper-faithful implementation finds
    /// nothing. The paper's own experiments use stochastic SEM data.
    /// </summary>
    private static Matrix<double> CreateNoisySEMData()
    {
        int n = 200;
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        var data = new double[n, 3];
        for (int i = 0; i < n; i++)
        {
            double x0 = rng.NextDouble() * 2.0 - 1.0;
            double x1 = 2.0 * x0 + 0.5 + (rng.NextDouble() * 0.4 - 0.2);
            double x2 = x0 + 0.3 * x1 + (rng.NextDouble() * 0.4 - 0.2);

            data[i, 0] = x0;
            data[i, 1] = x1;
            data[i, 2] = x2;
        }

        return new Matrix<double>(data);
    }

    [Fact(Timeout = 120000)]
    public async Task GraNDAG_FindsCausalStructure()
    {
        var algo = new GraNDAGAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateNoisySEMData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task CGNN_FindsCausalStructure()
    {
        var algo = new CGNNAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Theory(Timeout = 120000)]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(42)]
    [InlineData(2028)]
    public async Task CGNN_PairwiseOrientations_AlwaysProduceADag(int seed)
    {
        await Task.Yield();
        var algo = new CGNNAlgorithm<double>(new AiDotNet.Models.Options.CausalDiscoveryOptions
        {
            Seed = seed,
            HiddenUnits = 16,
            MaxEpochs = 30
        });

        var graph = algo.DiscoverStructure(CreateCgnnDagRegressionData(), CgnnFeatureNames);

        Assert.True(graph.IsDAG(),
            $"CGNN returned a directed cycle for seed {seed}; pairwise MMD orientations must be projected onto a DAG.");
    }

    [Fact(Timeout = 120000)]
    public async Task CASTLE_FindsCausalStructure()
    {
        var algo = new CASTLEAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task DECI_FindsCausalStructure()
    {
        var algo = new DECIAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task AVICI_FindsCausalStructure()
    {
        var algo = new AVICIAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task AmortizedCD_FindsCausalStructure()
    {
        var algo = new AmortizedCDAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task CausalVAE_FindsCausalStructure()
    {
        var algo = new CausalVAEAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }

    [Fact(Timeout = 120000)]
    public async Task TCDF_FindsCausalStructure()
    {
        var algo = new TCDFAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateSyntheticData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }
}
