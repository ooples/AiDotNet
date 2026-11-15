using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Preprocessing.FeatureSelection.Filter.Regression;
using AiDotNet.Preprocessing.FeatureSelection.Filter.Spectral;
using AiDotNet.Preprocessing.FeatureSelection.GameTheory;
using AiDotNet.Preprocessing.FeatureSelection.Genetic;
using AiDotNet.Preprocessing.FeatureSelection.Graph;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using SelectKBestScoreFunc = AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate.SelectKBestScoreFunc;
using UnivariateKBestHalf = AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate.SelectKBest<System.Half>;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Reference tests for the fifth review round of the feature selectors: Half overflow of raw sums of squares and
/// squared distances (chi2 totals, unit-variance scaling, scatters, Laplacian quadratic forms, heat kernels,
/// correlations, variances) and Half running sums that stop counting at 2048 (Banzhaf swings, Shapley samples and the
/// evolutionary importances). Each Half result is checked against the exact value or the double run.
/// </summary>
public class FeatureSelectionFilterRoundFiveTests
{
    private static Matrix<TN> Columns<TN>(int n, Func<double, TN> convert, params Func<int, double>[] columns)
    {
        var matrix = new Matrix<TN>(n, columns.Length);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < columns.Length; j++)
                matrix[i, j] = convert(columns[j](i));
        }

        return matrix;
    }

    private static Vector<TN> Values<TN>(int n, Func<double, TN> convert, Func<int, double> value)
    {
        var vector = new Vector<TN>(n);
        for (int i = 0; i < n; i++)
            vector[i] = convert(value(i));
        return vector;
    }

    private static Half ToHalf(double value) => (Half)value;

    private static double ToDouble(double value) => value;

    private static double[] AsDoubles(Vector<Half>? values)
    {
        Assert.NotNull(values);
        return values is null ? Array.Empty<double>() : Enumerable.Range(0, values.Length).Select(i => (double)values[i]).ToArray();
    }

    private static double[] AsDoubles(Vector<double>? values)
    {
        Assert.NotNull(values);
        return values is null ? Array.Empty<double>() : Enumerable.Range(0, values.Length).Select(i => values[i]).ToArray();
    }

    private static void AssertClose(double expected, double actual, double relative)
    {
        Assert.False(double.IsNaN(actual) || double.IsInfinity(actual), $"expected {expected}, got {actual}");
        Assert.InRange(actual, expected - relative * Math.Abs(expected), expected + relative * Math.Abs(expected));
    }

    // ---- raw sums of squares and squared distances in Half ----

    // 20 rows of 3000 and 20 of 1000: the feature total 80000 overflowed Half (O - E = inf - inf = NaN). Observed
    // 60000 / 20000 against 40000 expected each: chi2 = 2 * 20000^2 / 40000 = 20000.
    [Fact(Timeout = 60000)]
    public async Task Chi2_Half_TotalPastMaxValue_KeepsTheStatistic()
    {
        await Task.CompletedTask;
        const int n = 40;
        var data = Columns(n, ToHalf, i => i < 20 ? 3000 : 1000);
        var target = Values(n, ToHalf, i => i < 20 ? 0 : 1);

        var selector = new UnivariateKBestHalf(1, SelectKBestScoreFunc.Chi2);
        selector.Fit(data, target);

        AssertClose(20000, AsDoubles(selector.Scores)[0], 0.01);
    }

    // y = x = 100 i: the sum of squared deviations behind the unit-std scaling overflowed Half, std = inf turned
    // every value into 0 and the mutual information into 0; the Half estimate matches the double one.
    [Fact(Timeout = 60000)]
    public async Task MutualInfoRegression_Half_LargeDeviations_MatchesDouble()
    {
        await Task.CompletedTask;
        const int n = 50;
        var reference = new MutualInfoRegression<double>(1);
        reference.Fit(Columns(n, ToDouble, i => 100.0 * i), Values(n, ToDouble, i => 100.0 * i));
        var half = new MutualInfoRegression<Half>(1);
        half.Fit(Columns(n, ToHalf, i => 100.0 * i), Values(n, ToHalf, i => 100.0 * i));

        double expected = AsDoubles(reference.MIScores)[0];
        Assert.True(expected > 1, $"double MI {expected}");
        AssertClose(expected, AsDoubles(half.MIScores)[0], 0.05);
    }

    // classes [0, 200, 400, 600, 800] and [100, ..., 900]: between = 2 * 5 * 50^2 = 25000, within = 2 * 400000;
    // the within-class scatter overflowed Half and scored the feature 0. Fisher score = 0.03125.
    [Fact(Timeout = 60000)]
    public async Task FisherScoreAndTraceRatio_Half_LargeScatter_KeepTheirRatio()
    {
        await Task.CompletedTask;
        const int n = 10;
        Func<int, double> feature = i => 100.0 * i;
        Func<int, double> label = i => i % 2;

        var fisher = new FisherScore<Half>(1);
        fisher.Fit(Columns(n, ToHalf, feature), Values(n, ToHalf, label));
        AssertClose(0.03125, AsDoubles(fisher.FisherScores)[0], 0.01);

        var traceRatio = new TraceRatio<Half>(1);
        traceRatio.Fit(Columns(n, ToHalf, feature), Values(n, ToHalf, label));
        AssertClose(0.03125, AsDoubles(traceRatio.FeatureScores)[0], 0.01);
        AssertClose(0.03125, (double)traceRatio.TraceRatioValue, 0.01);
    }

    // x = 10 i over 20 rows (distances exact in Half, so the same neighbours): the degree-weighted sum of squared deviations overflowed Half (score NaN / inf)
    [Fact(Timeout = 60000)]
    public async Task LaplacianScore_Half_LargeDeviations_MatchesDouble()
    {
        await Task.CompletedTask;
        const int n = 20;
        var reference = new LaplacianScore<double>(1, 5, 1000.0);
        reference.Fit(Columns(n, ToDouble, i => 10.0 * i));
        var half = new LaplacianScore<Half>(1, 5, 1000.0);
        half.Fit(Columns(n, ToHalf, i => 10.0 * i));

        var expected = AsDoubles(reference.LaplacianScores);
        var actual = AsDoubles(half.LaplacianScores);
        for (int j = 0; j < expected.Length; j++)
            AssertClose(expected[j], actual[j], 0.02);
    }

    // x = 300 i: d^2 and sigma^2 of the heat kernel overflowed Half (inf / inf = NaN, kept by the 0 of the mask),
    // and so did the sum of squares of the regression; every score came out NaN
    [Fact(Timeout = 60000)]
    public async Task MCFS_Half_LargeDistances_ScoresStayFinite()
    {
        await Task.CompletedTask;
        const int n = 20;
        var half = new MCFS<Half>(1, 2, 3, 1.0, 7);
        half.Fit(Columns(n, ToHalf, i => 300.0 * i, i => (i * 7) % 20));

        foreach (double score in AsDoubles(half.MCFSScores))
            Assert.True(!double.IsNaN(score) && !double.IsInfinity(score) && score >= 0, $"score {score}");
    }

    // deviations of +-1350: the column sums of squares overflowed Half, norm = inf turned r into 0 and removed the
    // edge between two perfectly (anti-)correlated features
    [Fact(Timeout = 60000)]
    public async Task GraphBasedFS_Half_LargeDeviations_KeepsTheEdge()
    {
        await Task.CompletedTask;
        const int n = 10;
        var values = new Half[n * 2];
        for (int i = 0; i < n; i++)
        {
            values[i * 2] = (Half)(300.0 * (i - 4.5));
            values[i * 2 + 1] = (Half)(-300.0 * (i - 4.5));
        }

        var adjacency = GraphBasedFS<Half>.BuildAdjacencyMatrix(new Tensor<Half>(values, new[] { n, 2 }), 0.3);

        AssertClose(1.0, (double)adjacency[0, 1], 0.01);
        AssertClose(1.0, (double)adjacency[1, 0], 0.01);
        Assert.Equal(0.0, (double)adjacency[0, 0]);
    }

    // x = 50 i over 12 rows: variance 32500 fits Half, its sum of squared deviations 357500 did not (score inf)
    [Fact(Timeout = 60000)]
    public async Task SpectralClusteringFS_Half_LargeDeviations_MatchesDouble()
    {
        await Task.CompletedTask;
        const int n = 12;
        var reference = new SpectralClusteringFS<double>(2, 100.0);
        reference.Fit(Columns(n, ToDouble, i => 50.0 * i, i => 50.0 * i + 100, i => 25.0 * i));
        var half = new SpectralClusteringFS<Half>(2, 100.0);
        half.Fit(Columns(n, ToHalf, i => 50.0 * i, i => 50.0 * i + 100, i => 25.0 * i));

        var expected = AsDoubles(reference.FeatureScores);
        Assert.Equal(new[] { 32500.0, 32500.0, 8125.0 }, expected.Select(v => Math.Round(v, 6)).ToArray());
        var actual = AsDoubles(half.FeatureScores);
        for (int j = 0; j < expected.Length; j++)
            AssertClose(expected[j], actual[j], 0.01);
    }

    // ---- running sums in Half ----

    private static readonly double[] Uncorrelated = { 1, 0, 0, 1, 0, 1, 1, 0 };

    // feature 0 is the target, feature 1 uncorrelated with it: feature 0 swings every coalition (index 1), feature 1 none. The
    // Half 0/1 sum of swings stopped at 2048, which gave 2048 / 3000 = 0.683.
    [Fact(Timeout = 60000)]
    public async Task Banzhaf_Half_3000Samples_CountsEverySwing()
    {
        await Task.CompletedTask;
        const int n = 8;
        var selector = new BanzhafFeatureSelector<Half>(1, 3000, 0.25, 3);
        selector.Fit(Columns(n, ToHalf, i => i + 1, i => Uncorrelated[i]), Values(n, ToHalf, i => i + 1));

        var indices = AsDoubles(selector.BanzhafIndices);
        AssertClose(1.0, indices[0], 0.002);
        Assert.Equal(0.0, indices[1]);
    }

    // one feature equal to the target: every sampled marginal contribution is R^2 = 1, so the Shapley value is 1; the
    // Half sum of 3000 ones stopped at 2048 (0.683)
    [Fact(Timeout = 60000)]
    public async Task Shapley_Half_3000Samples_AveragesEverySample()
    {
        await Task.CompletedTask;
        const int n = 8;
        var selector = new ShapleyFeatureSelector<Half>(1, 3000, 3);
        selector.Fit(Columns(n, ToHalf, i => i + 1), Values(n, ToHalf, i => i + 1));

        AssertClose(1.0, AsDoubles(selector.ShapleyValues)[0], 0.01);
    }

    // Feature 0 is the target (|r| = 1), feature 1 has |r| = 3 / sqrt(84) = 0.327. The importances sum each
    // generation's fitness-weighted counts; the Half sums stopped growing once a contribution fell below half their
    // spacing (feature 0 near 32768 of about 48000), which skewed the normalised importance of feature 1.
    private static readonly double[] WeakFeature = { 0, 1, 0, 0, 1, 1, 0, 1 };

    private static Matrix<TN> EvolutionData<TN>(Func<double, TN> convert) =>
        Columns(8, convert, i => i + 1, i => WeakFeature[i]);

    private static Vector<TN> EvolutionTarget<TN>(Func<double, TN> convert) => Values(8, convert, i => i + 1);

    // the masks are booleans and the fitness ranks are the same in Half and double, so both runs take the same path
    [Fact(Timeout = 120000)]
    public async Task GeneticAlgorithm_Half_LongRun_MatchesDoubleImportances()
    {
        await Task.CompletedTask;
        var reference = new GeneticAlgorithmSelector<double>(1, 10, 6000, 0.1, 0.8, 5);
        reference.Fit(EvolutionData(ToDouble), EvolutionTarget(ToDouble));
        var half = new GeneticAlgorithmSelector<Half>(1, 10, 6000, 0.1, 0.8, 5);
        half.Fit(EvolutionData(ToHalf), EvolutionTarget(ToHalf));

        var expected = AsDoubles(reference.FeatureImportances);
        var actual = AsDoubles(half.FeatureImportances);
        Assert.Equal(1.0, expected[0]);
        Assert.True(expected[1] > 0.01, $"double importance {expected[1]}");
        for (int j = 0; j < expected.Length; j++)
            AssertClose(expected[j], actual[j], 0.02);
    }

    // Differential evolution keeps a trial only when it is fitter, so after the first generations every member keeps
    // feature 0: its importance is about 10 per generation over 3000 generations, while the Half sum (one member at
    // a time) stopped at 2048. Feature 1 only gathers importance in those first generations.
    [Fact(Timeout = 120000)]
    public async Task DifferentialEvolution_Half_LongRun_MatchesDoubleImportances()
    {
        await Task.CompletedTask;
        var reference = new DifferentialEvolutionSelector<double>(1, 10, 3000, 0.8, 0.7, 5);
        reference.Fit(EvolutionData(ToDouble), EvolutionTarget(ToDouble));
        var half = new DifferentialEvolutionSelector<Half>(1, 10, 3000, 0.8, 0.7, 5);
        half.Fit(EvolutionData(ToHalf), EvolutionTarget(ToHalf));

        var expected = AsDoubles(reference.FeatureImportances);
        var actual = AsDoubles(half.FeatureImportances);
        Assert.Equal(1.0, expected[0]);
        Assert.True(expected[1] > 0, $"double importance {expected[1]}");
        AssertClose(1.0, actual[0], 0.002);
        AssertClose(expected[1], actual[1], 0.1);
    }

    // The swarm settles on feature 0 (about 10 per iteration over 6000 iterations, 60000), while the Half sum of the
    // per-iteration contributions stopped at 32768; feature 1's normalised importance came out almost twice as large.
    [Fact(Timeout = 120000)]
    public async Task ParticleSwarm_Half_LongRun_MatchesDoubleImportances()
    {
        await Task.CompletedTask;
        var reference = new ParticleSwarmSelector<double>(1, 10, 6000, 0.7, 1.5, 1.5, 5);
        reference.Fit(EvolutionData(ToDouble), EvolutionTarget(ToDouble));
        var half = new ParticleSwarmSelector<Half>(1, 10, 6000, 0.7, 1.5, 1.5, 5);
        half.Fit(EvolutionData(ToHalf), EvolutionTarget(ToHalf));

        var expected = AsDoubles(reference.FeatureImportances);
        var actual = AsDoubles(half.FeatureImportances);
        Assert.Equal(1.0, expected[0]);
        Assert.True(expected[1] > 0, $"double importance {expected[1]}");
        AssertClose(1.0, actual[0], 0.002);
        AssertClose(expected[1], actual[1], 0.25);
    }
}
