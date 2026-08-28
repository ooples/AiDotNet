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
    private sealed class CastleAdjacencyHarness : CASTLEAlgorithm<double>
    {
        public Matrix<double> BuildScaleAware(double[,] learned, Matrix<double> covariance, int dimensions)
            => BuildFinalAdjacency(learned, covariance, dimensions);

        public Matrix<double> Build(double[,] learned, Matrix<double> covariance, int dimensions, double threshold)
            => BuildFinalAdjacency(learned, covariance, dimensions, threshold);
    }

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

    [Fact]
    public void CASTLE_FinalAdjacency_UsesInclusiveThresholdDirectionAndCovarianceRatio()
    {
        var learned = new double[,]
        {
            { 0.0, 0.29, 0.30 },
            { 0.10, 0.0, 0.31 },
            { 0.10, 0.20, 0.0 },
        };
        var covariance = new Matrix<double>(new double[,]
        {
            { 2.0, 0.5, 1.0 },
            { 0.5, 4.0, 2.0 },
            { 1.0, 2.0, 8.0 },
        });

        Matrix<double> adjacency = new CastleAdjacencyHarness()
            .Build(learned, covariance, dimensions: 3, threshold: 0.3);

        Assert.Equal(0.0, adjacency[0, 1]); // 0.29 is below the inclusive 0.30 boundary.
        Assert.Equal(0.5, adjacency[0, 2], 12); // 0.30 retained; cov(0,2) / var(0) = 1 / 2.
        Assert.Equal(0.5, adjacency[1, 2], 12); // 0.31 retained; cov(1,2) / var(1) = 2 / 4.
        Assert.Equal(0.0, adjacency[2, 0]);
        Assert.Equal(0.0, adjacency[2, 1]);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    public void DeepCausal_FinalAdjacency_RejectsUniformInfluenceForSmallGraphs(int dimensions)
    {
        var learned = new double[dimensions, dimensions];
        var covariance = new Matrix<double>(dimensions, dimensions);
        for (int i = 0; i < dimensions; i++)
        {
            covariance[i, i] = 1.0;
            for (int j = 0; j < dimensions; j++)
            {
                if (i == j) continue;
                learned[i, j] = 1.0 / dimensions;
                covariance[i, j] = 0.5;
            }
        }

        Matrix<double> adjacency = new CastleAdjacencyHarness()
            .BuildScaleAware(learned, covariance, dimensions);

        for (int i = 0; i < dimensions; i++)
            for (int j = 0; j < dimensions; j++)
                Assert.Equal(0.0, adjacency[i, j]);
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
