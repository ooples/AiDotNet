using AiDotNet.CausalDiscovery.DeepLearning;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
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

    /// <summary>
    /// A linear structural equation model with the DAG X0 -> X1, X0 -> X2, X1 -> X2.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The previous fixture was <c>x = i*0.1</c> with <c>X0 = x</c>, <c>X1 = 2x + 0.5</c> and
    /// <c>X2 = x + 0.3*X1</c>, and no noise at all. Every column was then an exact affine function
    /// of the same <c>x</c> — <c>X2</c> reduces to <c>1.6x + 0.15</c> — so all three were perfectly
    /// collinear, every pairwise correlation was exactly 1 and the covariance matrix had rank 1.
    /// </para>
    /// <para>
    /// On rank-deficient noiseless data a structure learner has nothing to recover: infinitely many
    /// weight configurations reconstruct each variable from the others equally well, so which
    /// weights grow is decided by the optimization path rather than by the data. CASTLE thresholds
    /// raw weight-row norms at the reference implementation's 0.3, so "did any norm clear 0.3"
    /// became a coin toss — which is why CASTLE_FindsCausalStructure and
    /// AmortizedCD_FindsCausalStructure failed locally in Debug while passing CI in Release on the
    /// same commit. Nothing about the algorithms changed between those runs; floating-point
    /// differences moved an undetermined answer across a fixed threshold.
    /// </para>
    /// <para>
    /// Noise is what makes the structure identifiable, so the SEM below adds it. The coefficients
    /// are the ones the old fixture intended, the graph is a genuine DAG, the covariance is full
    /// rank, and the sample is large enough for the edges to be estimated rather than guessed.
    /// Seeded, so every run sees the same data.
    /// </para>
    /// </remarks>
    /// <summary>
    /// A linear structural equation model over X0 -> X1, X0 -> X2, X1 -> X2, driven by an
    /// AUTOREGRESSIVE exogenous series so both the contemporaneous and the temporal methods in this
    /// file have signal to find.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The original fixture was <c>x = i*0.1; X0 = x; X1 = 2x + 0.5; X2 = x + 0.3*X1</c> — noiseless
    /// and therefore RANK 1. Every variable was an exact affine function of the loop counter, so all
    /// pairwise correlations were exactly 1 and the covariance matrix was singular. Any threshold on
    /// a quantity estimated from that matrix sits on a knife edge, which is how
    /// <c>CASTLE_FindsCausalStructure</c> came to fail in Debug and pass in Release on the same sha:
    /// nothing about the model changed, only the last bits of the arithmetic.
    /// </para>
    /// <para>
    /// Two properties are needed at once, and getting only the first is a trap worth recording.
    /// Adding Gaussian exogenous noise fixes the degeneracy but, on its own, makes the rows i.i.d. —
    /// and <see cref="TCDFAlgorithm{T}"/> is a TEMPORAL method that predicts each variable from the
    /// others' histories through causal convolutions. On i.i.d. rows it has nothing to find, so an
    /// i.i.d. fixture does not test it, it starves it. The old ramp was accidentally supplying that
    /// temporal structure as a monotonic trend.
    /// </para>
    /// <para>
    /// So the driver X0 is an AR(1) series: autocorrelated, giving the temporal methods real lag
    /// structure, while the additive noise keeps the covariance full rank for everything else. The
    /// causal structure and coefficients are unchanged from the original fixture, so this is the
    /// same claim tested on data that can actually support it.
    /// </para>
    /// </remarks>
    private static Matrix<double> CreateSyntheticData()
    {
        const int n = 200;
        const double ar = 0.8;      // AR(1) coefficient on the exogenous driver
        const double noise = 0.2;   // exogenous noise on the structural equations
        var rng = RandomHelper.CreateSeededRandom(42);
        var data = new double[n, 3];

        // Box-Muller, so the exogenous noise is Gaussian rather than uniform.
        double Gaussian()
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }

        double prev = Gaussian();
        for (int i = 0; i < n; i++)
        {
            double x0 = ar * prev + Gaussian();
            double x1 = 2.0 * x0 + 0.5 + noise * Gaussian();
            double x2 = x0 + 0.3 * x1 + noise * Gaussian();

            data[i, 0] = x0;
            data[i, 1] = x1;
            data[i, 2] = x2;
            prev = x0;
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
