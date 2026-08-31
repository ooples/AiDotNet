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

    /// <summary>
    /// A purely LAGGED causal chain, for the temporal methods.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <see cref="CreateSyntheticData"/> cannot support a temporal claim, even though it has lag
    /// structure. Its driver is AR(1), so X0[t-1] predicts X0[t]; and X1 and X2 are built from the
    /// CONTEMPORANEOUS X0, so X1[t-1] and X2[t-1] both contain X0[t-1] too. A method that scores
    /// each variable against the others' histories therefore finds X1[t-1] and X2[t-1] predicting
    /// X0[t] about as well as X0's own past does, and the declared DAG has no incoming edge to X0
    /// at all. The signal is real in the data and wrong about the graph: it is the driver's
    /// autocorrelation reflected back through its own children.
    /// </para>
    /// <para>
    /// Here X0 is i.i.d., so nothing -- its own past included -- predicts X0[t], and the only
    /// incoming edges available to find are the true ones. The dependence is carried entirely by
    /// lag 1, which is what a causal-convolution model is built to detect:
    /// </para>
    /// <code>
    /// X0[t] = e0[t]                                  (i.i.d.)
    /// X1[t] = 1.5 * X0[t]   + 0.8 * X0[t-1] + e1[t]
    /// X2[t] = 1.0 * X0[t]   + 0.7 * X1[t-1] + e2[t]
    /// </code>
    /// <para>
    /// The contemporaneous terms are kept deliberately. They are not what created the reverse
    /// signal -- the AR(1) driver was. With an i.i.d. X0 a child's history carries no information
    /// about X0's future, so the edges stay strong enough to detect without becoming ambiguous.
    /// </para>
    /// <para>
    /// The graph is the same one the rest of the file asserts -- X0 to X1, X0 to X2, X1 to X2 --
    /// so this tests the same claim on data that can actually carry it.
    /// </para>
    /// </remarks>
    private static Matrix<double> CreateLaggedTemporalData()
    {
        const int n = 400;
        const double noise = 0.15;
        var rng = RandomHelper.CreateSeededRandom(1337);
        var data = new double[n, 3];

        double Gaussian()
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }

        double prevX0 = Gaussian();
        double prevX1 = Gaussian();
        for (int i = 0; i < n; i++)
        {
            // i.i.d. driver: X0[t-1] says nothing about X0[t], so no history -- its own or a
            // child's -- can predict X0, and the reverse edges the shared fixture invites do not
            // exist here. The children keep their strong CONTEMPORANEOUS coupling as well as a lag
            // term, so the dependence is easy to detect while the direction stays unambiguous.
            double x0 = 2.0 * Gaussian();
            double x1 = 1.5 * x0 + 0.8 * prevX0 + noise * Gaussian();
            double x2 = 1.0 * x0 + 0.7 * prevX1 + noise * Gaussian();

            data[i, 0] = x0;
            data[i, 1] = x1;
            data[i, 2] = x2;

            prevX0 = x0;
            prevX1 = x1;
        }

        return new Matrix<double>(data);
    }

    /// <summary>
    /// The same causal chain with variance DECREASING along it, for the variance-ordered methods.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <see cref="DAGGNNAlgorithm{T}"/> orients its final graph by descending raw column variance,
    /// so the fixture's spread decides the answer before the model does. In
    /// <see cref="CreateSyntheticData"/>, X1 = 2 * X0, which gives X1 roughly four times X0's
    /// variance: the algorithm can rank X1 above X0 and orient the edge backwards, and the test
    /// would still pass because it only asks for a meaningful graph. That makes it a test of the
    /// fixture's accidental scaling rather than of the method.
    /// </para>
    /// <para>
    /// Coefficients below one put the variance ordering where the topological order already is:
    /// with Var(X0) = 9, the chain gives Var(X1) ~ 8.1 and Var(X2) ~ 5.9, strictly decreasing with
    /// margins of roughly 10% and 27% -- comfortably wider than the sampling error on a variance at
    /// n = 500, so the ordering cannot invert by chance.
    /// </para>
    /// <para>
    /// They are close to one rather than small, which matters on the other side: BuildFinalAdjacency
    /// requires a learned influence of at least max(1/d + 0.15, 0.3), which is 0.483 for three
    /// variables. Shrinking the coefficients far enough to make the variance ordering obvious also
    /// shrinks the dependence until nothing clears that threshold and the graph comes back empty,
    /// so the fixture has to satisfy both constraints at once rather than either alone.
    /// </para>
    /// </remarks>
    private static Matrix<double> CreateVarianceOrderedSemData()
    {
        const int n = 500;
        const double noise = 0.05;
        var rng = RandomHelper.CreateSeededRandom(4242);
        var data = new double[n, 3];

        double Gaussian()
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }

        for (int i = 0; i < n; i++)
        {
            double x0 = 3.0 * Gaussian();
            double x1 = 0.95 * x0 + noise * Gaussian();
            double x2 = 0.10 * x0 + 0.75 * x1 + noise * Gaussian();

            data[i, 0] = x0;
            data[i, 1] = x1;
            data[i, 2] = x2;
        }

        return new Matrix<double>(data);
    }

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
        // Variance-ordered fixture: DAGGNN orients by descending column variance, so a fixture
        // whose spread contradicts its own topology tests the scaling, not the method.
        var algo = new DAGGNNAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateVarianceOrderedSemData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);

        // ASSERT THE STRUCTURE, not just that something was returned. Counting edges alone let this
        // test pass while every edge pointed the wrong way: on the shared fixture, whose variance
        // ordering contradicts its topology, the algorithm returned X1 -> X0 and X2 -> X0 and the
        // assertions above were satisfied. A discovery test that cannot tell a recovered DAG from
        // its transpose is not testing discovery.
        var a = graph.AdjacencyMatrix;
        Assert.True(a[0, 1] != 0.0, "X0 -> X1 should be recovered; got adjacency[0,1] = 0.");
        Assert.True(a[0, 2] != 0.0, "X0 -> X2 should be recovered; got adjacency[0,2] = 0.");
        Assert.True(a[1, 2] != 0.0, "X1 -> X2 should be recovered; got adjacency[1,2] = 0.");
        Assert.True(a[1, 0] == 0.0, $"X1 -> X0 is the reverse of a true edge; got {a[1, 0]}.");
        Assert.True(a[2, 0] == 0.0, $"X2 -> X0 is the reverse of a true edge; got {a[2, 0]}.");
        Assert.True(a[2, 1] == 0.0, $"X2 -> X1 is the reverse of a true edge; got {a[2, 1]}.");
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
        // Lagged fixture: the shared one carries the driver's autocorrelation into its children,
        // so every child's history predicts the driver and the reverse edges look real.
        var algo = new TCDFAlgorithm<double>();
        var graph = algo.DiscoverStructure(CreateLaggedTemporalData(), FeatureNames);
        CausalDiscoveryTestHelper.AssertMeaningfulGraph(graph);
        CausalDiscoveryTestHelper.AssertGraphAPIConsistency(graph);
    }
}
