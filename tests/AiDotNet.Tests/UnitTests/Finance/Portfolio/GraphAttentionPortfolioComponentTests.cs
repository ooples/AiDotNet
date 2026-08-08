using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Finance.Portfolio;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance.Portfolio;

/// <summary>
/// Verifies the GAT portfolio components against the paper (Kamesh Korangi, Christophe Mues and
/// Cristian Bravo, "Large-scale Time-Varying Portfolio Optimisation using Graph Attention Networks",
/// arXiv:2407.15532).
/// </summary>
/// <remarks>
/// Each test targets a way this method gets flattened: Pearson in place of distance correlation
/// (which misses non-linear dependence), skipping the double-centring, a softmax allocation layer
/// (which destroys the sparsity the paper explicitly chose its mechanism to obtain), averaging
/// attention heads instead of concatenating them, unmasked attention (which discards the filtered
/// graph), and an inverted Sharpe loss sign.
/// </remarks>
public class GraphAttentionPortfolioComponentTests
{
    private static Vector<double> Vec(params double[] values)
    {
        var v = new Vector<double>(values.Length);
        for (int i = 0; i < values.Length; i++) v[i] = values[i];
        return v;
    }

    private static Tensor<double> Panel(int steps, int assets, Func<int, int, double> value)
    {
        var t = new Tensor<double>(new[] { steps, assets });
        for (int s = 0; s < steps; s++)
            for (int a = 0; a < assets; a++)
                t[(s * assets) + a] = value(s, a);
        return t;
    }

    // ------------------------------------------------------------ distance correlation

    [Fact]
    public void DistanceCorrelationIsOneForAnIdenticalSeriesAndSymmetric()
    {
        var builder = new AssetGraphBuilder<double>();
        var x = Vec(0.1, 0.4, 0.2, 0.9, 0.3, 0.7);
        var y = Vec(0.5, 0.2, 0.8, 0.1, 0.6, 0.4);

        Assert.Equal(1.0, builder.DistanceCorrelation(x, x), 8);
        Assert.Equal(builder.DistanceCorrelation(x, y), builder.DistanceCorrelation(y, x), 10);
    }

    [Fact]
    public void DistanceCorrelationStaysWithinTheUnitInterval()
    {
        // Unlike Pearson it is never negative: it measures the STRENGTH of dependence, not direction.
        var builder = new AssetGraphBuilder<double>();
        var rng = new Random(17);

        for (int trial = 0; trial < 12; trial++)
        {
            var a = new Vector<double>(9);
            var b = new Vector<double>(9);
            for (int i = 0; i < 9; i++)
            {
                a[i] = (rng.NextDouble() * 4.0) - 2.0;
                b[i] = (rng.NextDouble() * 4.0) - 2.0;
            }

            double d = builder.DistanceCorrelation(a, b);
            Assert.InRange(d, 0.0, 1.0);
        }
    }

    [Fact]
    public void DistanceCorrelationDetectsANonLinearRelationshipThatPearsonMisses()
    {
        // THE reason the paper uses distance correlation. y = x^2 on a symmetric grid is a perfect
        // functional dependence, yet its Pearson correlation is ~0. A graph built on Pearson would
        // record these two firms as unrelated.
        var builder = new AssetGraphBuilder<double>();
        var x = Vec(-3, -2, -1, 0, 1, 2, 3);
        var y = Vec(9, 4, 1, 0, 1, 4, 9);

        double pearson = Pearson(x, y);
        double dcor = builder.DistanceCorrelation(x, y);

        Assert.True(Math.Abs(pearson) < 1e-9, $"Pearson should vanish here; got {pearson}.");
        Assert.True(dcor > 0.4,
            $"Distance correlation must detect the quadratic dependence Pearson misses; got {dcor}.");
    }

    [Fact]
    public void DistanceCorrelationIsNearZeroForAConstantSeries()
    {
        // A constant series has no variation and therefore no dependence to report. Without the
        // double-centring guard this path divides by zero.
        var builder = new AssetGraphBuilder<double>();
        var varying = Vec(1.0, 5.0, 2.0, 8.0);
        var constant = Vec(3.0, 3.0, 3.0, 3.0);

        Assert.Equal(0.0, builder.DistanceCorrelation(varying, constant), 10);
    }

    [Fact]
    public void DistanceCorrelationMatrixIsSymmetricWithAUnitDiagonal()
    {
        var builder = new AssetGraphBuilder<double>();
        var panel = Panel(10, 4, (s, a) => Math.Sin((s * 0.5) + a) + (0.1 * a * s));

        var m = builder.DistanceCorrelationMatrix(panel);
        Assert.Equal(new[] { 4, 4 }, m.Shape.ToArray());

        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(1.0, m[(i * 4) + i], 8);
            for (int j = 0; j < 4; j++)
                Assert.Equal(m[(i * 4) + j], m[(j * 4) + i], 10);
        }
    }

    // ------------------------------------------------------------ volatility series

    [Fact]
    public void VolatilitySeriesUsesARollingStandardDeviationNotRawReturns()
    {
        // The paper correlates return-VOLATILITY series, not returns, following Diebold and Yilmaz.
        // A constant return stream has zero volatility whatever its level, which is the property that
        // makes this a risk-co-movement measure rather than a price one.
        var builder = new AssetGraphBuilder<double>(volatilityLookback: 3);
        var returns = Panel(5, 2, (s, a) => a == 0 ? 0.01 : (s % 2 == 0 ? 0.05 : -0.05));

        var vol = builder.VolatilitySeries(returns);

        Assert.Equal(new[] { 3, 2 }, vol.Shape.ToArray());   // 5 - 3 + 1 rows
        for (int s = 0; s < 3; s++)
        {
            Assert.Equal(0.0, vol[(s * 2) + 0], 12);          // constant returns -> zero volatility
            Assert.True(vol[(s * 2) + 1] > 0.0, "The alternating series must show positive volatility.");
        }
    }

    [Fact]
    public void VolatilitySeriesRejectsTooShortAHistoryAndABadLookback()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new AssetGraphBuilder<double>(1));

        var builder = new AssetGraphBuilder<double>(volatilityLookback: 30);
        Assert.Throws<ArgumentException>(() => builder.VolatilitySeries(Panel(10, 3, (s, a) => 0.0)));
    }

    [Fact]
    public void VolatilityLookbackDefaultsToThePapersThirtyDays()
    {
        Assert.Equal(30, AssetGraphBuilder<double>.PaperVolatilityLookback);
        Assert.Equal(30, new AssetGraphBuilder<double>().VolatilityLookback);
    }

    // ------------------------------------------------------------ TMFG filtering

    [Theory]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(8)]
    [InlineData(12)]
    [InlineData(20)]
    public void TmfgProducesExactlyThreeTimesNMinusTwoEdges(int n)
    {
        // The planarity constraint's whole point: edges collapse from n(n-1)/2 to 3(n-2). A filter
        // that returned more than that is not planar and has not filtered.
        var builder = new AssetGraphBuilder<double>();
        var rng = new Random(n * 31);
        var dense = new Tensor<double>(new[] { n, n });
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
            {
                double w = rng.NextDouble();
                dense[(i * n) + j] = w;
                dense[(j * n) + i] = w;
            }

        var edges = builder.FilterTmfg(dense);

        Assert.Equal(AssetGraphBuilder<double>.MaxPlanarEdges(n), edges.Count);
        Assert.Equal(3 * (n - 2), edges.Count);

        // Far fewer than the dense graph once n is meaningful.
        if (n >= 8) Assert.True(edges.Count < n * (n - 1) / 2);
    }

    [Fact]
    public void TmfgEdgesAreUniqueAndUndirectedWithNoSelfLoops()
    {
        var builder = new AssetGraphBuilder<double>();
        const int n = 10;
        var rng = new Random(5);
        var dense = new Tensor<double>(new[] { n, n });
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
            {
                double w = rng.NextDouble();
                dense[(i * n) + j] = w;
                dense[(j * n) + i] = w;
            }

        var edges = builder.FilterTmfg(dense);
        var seen = new HashSet<(int, int)>();

        foreach (var e in edges)
        {
            Assert.True(e.Source < e.Target, $"Edge ({e.Source},{e.Target}) is not canonically ordered.");
            Assert.True(seen.Add((e.Source, e.Target)), $"Edge ({e.Source},{e.Target}) is duplicated.");
        }
    }

    [Fact]
    public void TmfgKeepsTheStrongEdgesOfAClearlyClusteredGraph()
    {
        // A filter that discarded the strongest relationships would defeat the purpose. Build two
        // tight clusters joined weakly and check the strong intra-cluster edges survive.
        var builder = new AssetGraphBuilder<double>();
        const int n = 8;
        var dense = new Tensor<double>(new[] { n, n });
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
            {
                bool sameCluster = (i < 4) == (j < 4);
                double w = sameCluster ? 0.9 : 0.05;
                dense[(i * n) + j] = w;
                dense[(j * n) + i] = w;
            }

        var edges = builder.FilterTmfg(dense);
        int strong = edges.Count(e => e.Weight > 0.5);

        Assert.True(strong >= edges.Count / 2,
            $"Most retained edges should be the strong ones; only {strong} of {edges.Count} were.");
    }

    [Fact]
    public void TmfgRejectsGraphsTooSmallToHaveAFace()
    {
        var builder = new AssetGraphBuilder<double>();
        Assert.Throws<ArgumentException>(() => builder.FilterTmfg(new Tensor<double>(new[] { 2, 2 })));
        Assert.Throws<ArgumentException>(() => builder.FilterTmfg(new Tensor<double>(new[] { 3, 4 })));
        Assert.Throws<ArgumentOutOfRangeException>(() => AssetGraphBuilder<double>.MaxPlanarEdges(2));
    }

    [Fact]
    public void AdjacencyMaskIncludesSelfLoops()
    {
        // A GAT node attends to itself as well as its neighbours; without the self-loop each asset's
        // own features are dropped from its updated representation.
        var builder = new AssetGraphBuilder<double>();
        var edges = new List<AssetGraphBuilder<double>.GraphEdge>
        {
            new(0, 1, 0.5),
        };

        var mask = builder.AdjacencyMask(edges, 3);

        Assert.Equal(1.0, mask[0], 10);          // (0,0)
        Assert.Equal(1.0, mask[(1 * 3) + 1], 10);
        Assert.Equal(1.0, mask[(2 * 3) + 2], 10);
        Assert.Equal(1.0, mask[1], 10);          // (0,1)
        Assert.Equal(1.0, mask[(1 * 3) + 0], 10); // symmetric
        Assert.Equal(0.0, mask[2], 10);          // (0,2) not connected
    }

    [Fact]
    public void AdjacencyMaskRejectsAnOutOfRangeEdge()
    {
        var builder = new AssetGraphBuilder<double>();
        var edges = new List<AssetGraphBuilder<double>.GraphEdge> { new(0, 5, 1.0) };
        Assert.Throws<ArgumentException>(() => builder.AdjacencyMask(edges, 3));
    }

    // ------------------------------------------------------------ graph attention

    [Fact]
    public void AttentionRowsSumToOneOverNeighboursOnly()
    {
        // The mask is what makes this a GRAPH attention: normalization happens WITHIN the
        // neighbourhood, so unconnected assets get exactly zero. Masking after a dense softmax would
        // leave the rows no longer summing to 1.
        var core = new GraphAttentionLayerCore<double>();
        const int nodes = 4, features = 2;

        var projected = new Tensor<double>(new[] { nodes, features });
        var rng = new Random(9);
        for (int i = 0; i < projected.Length; i++) projected[i] = (rng.NextDouble() * 2.0) - 1.0;

        var a = Vec(0.3, -0.7, 0.5, 0.2);

        // Path graph 0-1-2-3, with self-loops.
        var adjacency = new Tensor<double>(new[] { nodes, nodes });
        for (int i = 0; i < nodes; i++) adjacency[(i * nodes) + i] = 1.0;
        for (int i = 0; i < nodes - 1; i++)
        {
            adjacency[(i * nodes) + i + 1] = 1.0;
            adjacency[((i + 1) * nodes) + i] = 1.0;
        }

        var coeff = core.AttentionCoefficients(projected, a, adjacency);

        for (int u = 0; u < nodes; u++)
        {
            double sum = 0.0;
            for (int v = 0; v < nodes; v++)
            {
                double alpha = coeff[(u * nodes) + v];
                if (adjacency[(u * nodes) + v] == 0.0)
                    Assert.Equal(0.0, alpha, 12);   // never leaks onto a non-neighbour
                sum += alpha;
            }
            Assert.Equal(1.0, sum, 8);
        }
    }

    [Fact]
    public void AnIsolatedNodeAggregatesNothingRatherThanInventingUniformAttention()
    {
        // Node 2 has no edges and no self-loop here. A uniform fallback would have it attend to assets
        // it has no established relationship with, which is exactly what the graph filtering rejected.
        var core = new GraphAttentionLayerCore<double>();
        const int nodes = 3, features = 1;

        var projected = new Tensor<double>(new[] { nodes, features });
        projected[0] = 1.0; projected[1] = 2.0; projected[2] = 3.0;

        var adjacency = new Tensor<double>(new[] { nodes, nodes });
        adjacency[0] = 1.0; adjacency[1] = 1.0;
        adjacency[(1 * nodes) + 0] = 1.0; adjacency[(1 * nodes) + 1] = 1.0;

        var coeff = core.AttentionCoefficients(projected, Vec(0.5, 0.5), adjacency);

        for (int v = 0; v < nodes; v++)
            Assert.Equal(0.0, coeff[(2 * nodes) + v], 12);
    }

    [Fact]
    public void LeakyReLUKeepsAGradientPathForNegativelyScoredEdges()
    {
        // A plain ReLU (slope 0) would zero the gradient for every negatively scored pair, so those
        // edges could never recover. 0.2 is the GAT paper's value.
        Assert.Equal(0.2, GraphAttentionLayerCore<double>.LeakyReLUSlope, 12);

        var core = new GraphAttentionLayerCore<double>();
        Assert.Equal(3.0, core.LeakyReLU(3.0), 12);
        Assert.Equal(-0.4, core.LeakyReLU(-2.0), 12);
        Assert.Throws<ArgumentOutOfRangeException>(() => new GraphAttentionLayerCore<double>(-0.1));
    }

    [Fact]
    public void AggregationIsAConvexCombinationOfNeighboursThenRectified()
    {
        // With coefficients summing to 1 the aggregate must lie within the neighbours' range, and ReLU
        // then clamps at zero.
        var core = new GraphAttentionLayerCore<double>();
        const int nodes = 2, features = 1;

        var projected = new Tensor<double>(new[] { nodes, features });
        projected[0] = 2.0; projected[1] = 6.0;

        var coeff = new Tensor<double>(new[] { nodes, nodes });
        coeff[0] = 0.25; coeff[1] = 0.75;          // node 0 mixes both
        coeff[(1 * nodes) + 1] = 1.0;              // node 1 attends only to itself

        var aggregated = core.Aggregate(coeff, projected);

        Assert.Equal(5.0, aggregated[0], 10);      // 0.25*2 + 0.75*6
        Assert.Equal(6.0, aggregated[1], 10);
    }

    [Fact]
    public void AggregationRectifiesNegativeSums()
    {
        var core = new GraphAttentionLayerCore<double>();
        var projected = new Tensor<double>(new[] { 1, 1 });
        projected[0] = -4.0;
        var coeff = new Tensor<double>(new[] { 1, 1 });
        coeff[0] = 1.0;

        Assert.Equal(0.0, core.Aggregate(coeff, projected)[0], 12);
    }

    [Fact]
    public void HeadsAreConcatenatedNotAveraged()
    {
        // The paper's || operator. Averaging would collapse the heads into a single representation of
        // the original width and lose the distinct relational views they learn.
        var core = new GraphAttentionLayerCore<double>();

        var h0 = new Tensor<double>(new[] { 2, 2 });
        var h1 = new Tensor<double>(new[] { 2, 2 });
        for (int i = 0; i < 4; i++) { h0[i] = i + 1; h1[i] = 10 * (i + 1); }

        var joined = core.ConcatenateHeads(new[] { h0, h1 });

        Assert.Equal(new[] { 2, 4 }, joined.Shape.ToArray());   // width doubled, not preserved
        Assert.Equal(new[] { 1.0, 2.0, 10.0, 20.0 },
            new[] { joined[0], joined[1], joined[2], joined[3] });
    }

    [Fact]
    public void ConcatenateRejectsRaggedHeads()
    {
        var core = new GraphAttentionLayerCore<double>();
        Assert.Throws<ArgumentException>(() => core.ConcatenateHeads(
            new[] { new Tensor<double>(new[] { 2, 2 }), new Tensor<double>(new[] { 2, 3 }) }));
        Assert.Throws<ArgumentException>(() => core.ConcatenateHeads(Array.Empty<Tensor<double>>()));
    }

    // ------------------------------------------------------------ Sharpe loss + allocation

    [Fact]
    public void LossIsTheNegativeLogSharpeRatio()
    {
        // LF = -ln(mu) + ln(sigma) = -ln(mu/sigma). Minimising it maximises the Sharpe ratio; an
        // inverted sign would train for the WORST risk-adjusted return.
        var obj = new SharpeRatioPortfolioObjective<double>();
        var returns = new[] { 0.02, 0.01, 0.03, -0.005, 0.015 };

        double sharpe = obj.SharpeRatio(returns);
        Assert.Equal(-Math.Log(sharpe), obj.Loss(returns), 8);
    }

    [Fact]
    public void LossDecreasesAsTheSharpeRatioImproves()
    {
        var obj = new SharpeRatioPortfolioObjective<double>();
        var steady = new[] { 0.010, 0.011, 0.009, 0.010, 0.011 };   // same drift, low vol
        var choppy = new[] { 0.05, -0.04, 0.06, -0.03, 0.01 };      // similar drift, high vol

        Assert.True(obj.SharpeRatio(steady) > obj.SharpeRatio(choppy));
        Assert.True(obj.Loss(steady) < obj.Loss(choppy),
            "The better Sharpe ratio must give the lower loss.");
    }

    [Fact]
    public void LossStaysFiniteWhenTheMeanReturnIsNonPositive()
    {
        // -ln(mu) is undefined for mu <= 0, a case the paper does not discuss but training reaches
        // easily. It must yield a large finite penalty, not NaN, or one bad batch poisons every
        // subsequent gradient.
        var obj = new SharpeRatioPortfolioObjective<double>();
        var losing = new[] { -0.02, -0.01, -0.03, -0.015 };
        var flat = new[] { 0.0, 0.0, 0.0, 0.0 };

        double lossLosing = obj.Loss(losing);
        double lossFlat = obj.Loss(flat);

        Assert.False(double.IsNaN(lossLosing) || double.IsInfinity(lossLosing),
            $"A losing portfolio produced a non-finite loss: {lossLosing}.");
        Assert.False(double.IsNaN(lossFlat) || double.IsInfinity(lossFlat),
            $"A flat portfolio produced a non-finite loss: {lossFlat}.");

        // And it must be worse than any profitable portfolio.
        Assert.True(lossLosing > obj.Loss(new[] { 0.01, 0.012, 0.008, 0.011 }));
    }

    [Fact]
    public void AllocationSumNormalizesAndPreservesExactZeros()
    {
        // The paper's Importance Layer, and its defining property: a zero score yields a weight of
        // EXACTLY zero, so the firm leaves the portfolio. This is why softmax was rejected.
        var obj = new SharpeRatioPortfolioObjective<double>();
        var scores = Vec(3.0, 0.0, 1.0, 0.0);

        var w = obj.Allocate(scores);

        Assert.Equal(0.75, w[0], 10);
        Assert.Equal(0.0, w[1], 12);
        Assert.Equal(0.25, w[2], 10);
        Assert.Equal(0.0, w[3], 12);

        double sum = 0.0;
        for (int i = 0; i < w.Length; i++) sum += w[i];
        Assert.Equal(1.0, sum, 10);
        Assert.Equal(2, obj.ActivePositions(w));
    }

    [Fact]
    public void AllocationIsSparserThanASoftmaxWouldBe()
    {
        // Direct contrast with the alternative the paper rejected: over a large universe with mostly
        // zero scores, sum-normalisation holds a handful of positions while a softmax holds every one.
        var obj = new SharpeRatioPortfolioObjective<double>();
        const int assets = 50;
        var scores = new Vector<double>(assets);
        for (int i = 0; i < assets; i++) scores[i] = i < 3 ? 1.0 + i : 0.0;

        var w = obj.Allocate(scores);
        Assert.Equal(3, obj.ActivePositions(w));

        // A softmax over the same scores would give all 50 a positive share.
        int softmaxActive = 0;
        double denom = 0.0;
        for (int i = 0; i < assets; i++) denom += Math.Exp(scores[i]);
        for (int i = 0; i < assets; i++) if (Math.Exp(scores[i]) / denom > 0.0) softmaxActive++;
        Assert.Equal(assets, softmaxActive);
    }

    [Fact]
    public void AllocationClampsNegativeScoresRatherThanShifting()
    {
        // Long-only: a negative score means "do not hold". Shifting the vector to be non-negative
        // would give every previously-zero score a positive share and destroy the sparsity.
        var obj = new SharpeRatioPortfolioObjective<double>();
        var w = obj.Allocate(Vec(2.0, -5.0, 2.0));

        Assert.Equal(0.5, w[0], 10);
        Assert.Equal(0.0, w[1], 12);
        Assert.Equal(0.5, w[2], 10);
    }

    [Fact]
    public void AllocationFallsBackToEqualWeightWhenNoScoreIsPositive()
    {
        // Dividing by a zero sum would be undefined; an equal-weight portfolio is fully invested and
        // well defined.
        var obj = new SharpeRatioPortfolioObjective<double>();
        var w = obj.Allocate(Vec(-1.0, -2.0, 0.0, -0.5));

        double sum = 0.0;
        for (int i = 0; i < w.Length; i++) { Assert.Equal(0.25, w[i], 10); sum += w[i]; }
        Assert.Equal(1.0, sum, 10);
    }

    [Fact]
    public void PortfolioReturnsWeightTheAssetReturns()
    {
        var obj = new SharpeRatioPortfolioObjective<double>();
        var weights = Vec(0.5, 0.5);
        var returns = Panel(2, 2, (s, a) => s == 0 ? (a == 0 ? 0.10 : 0.00) : (a == 0 ? -0.02 : 0.06));

        var series = obj.PortfolioReturns(weights, returns);
        Assert.Equal(0.05, series[0], 10);
        Assert.Equal(0.02, series[1], 10);
    }

    [Fact]
    public void L1PenaltyIsTheSparsityDriver()
    {
        // Without the LASSO term the scores stay dense, the sum-normalisation spreads capital across
        // everything, and the allocation layer loses the property it was chosen for.
        var obj = new SharpeRatioPortfolioObjective<double>();
        Assert.Equal(6.0, obj.L1Penalty(Vec(1.0, -2.0, 3.0)), 10);
        Assert.True(obj.L1Penalty(Vec(1.0, 1.0, 1.0, 1.0)) > obj.L1Penalty(Vec(1.0, 0.0, 0.0, 0.0)));
    }

    [Fact]
    public void ObjectiveRejectsEmptyOrMismatchedInputs()
    {
        var obj = new SharpeRatioPortfolioObjective<double>();
        Assert.Throws<ArgumentException>(() => obj.Loss(Array.Empty<double>()));
        Assert.Throws<ArgumentException>(() => obj.Allocate(new Vector<double>(0)));
        Assert.Throws<ArgumentException>(() =>
            obj.PortfolioReturns(Vec(1.0, 0.0), Panel(2, 3, (s, a) => 0.0)));
    }

    private static double Pearson(Vector<double> x, Vector<double> y)
    {
        int n = x.Length;
        double mx = 0.0, my = 0.0;
        for (int i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
        mx /= n; my /= n;

        double cov = 0.0, vx = 0.0, vy = 0.0;
        for (int i = 0; i < n; i++)
        {
            double dx = x[i] - mx, dy = y[i] - my;
            cov += dx * dy; vx += dx * dx; vy += dy * dy;
        }

        return cov / Math.Sqrt(vx * vy);
    }
}
