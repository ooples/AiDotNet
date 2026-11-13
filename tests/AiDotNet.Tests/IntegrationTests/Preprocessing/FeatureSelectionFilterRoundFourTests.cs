using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Preprocessing;
using AiDotNet.Preprocessing.FeatureSelection;
using AiDotNet.Preprocessing.FeatureSelection.Filter.Multivariate;
using AiDotNet.Preprocessing.FeatureSelection.Filter.Regression;
using AiDotNet.Preprocessing.FeatureSelection.Filter.Spectral;
using AiDotNet.Preprocessing.FeatureSelection.Graph;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using FilterKendallTau = AiDotNet.Preprocessing.FeatureSelection.Filter.KendallTau<double>;
using FilterPearsonHalf = AiDotNet.Preprocessing.FeatureSelection.Filter.PearsonCorrelation<Half>;
using FilterSelectFdr = AiDotNet.Preprocessing.FeatureSelection.Filter.SelectFDR<double>;
using FilterSelectFpr = AiDotNet.Preprocessing.FeatureSelection.Filter.SelectFPR<double>;
using Knockoffs = AiDotNet.Preprocessing.FeatureSelection.HighDimensional.KnockoffFilter<double>;
using ReliefSelector = AiDotNet.Preprocessing.FeatureSelection.Filter.Relief.Relief<double>;
using SelectKBestScoreFunc = AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate.SelectKBestScoreFunc;
using StatisticalKruskalWallis = AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical.KruskalWallis<double>;
using StatisticalWilcoxon = AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical.WilcoxonSelector<double>;
using StatisticalWilcoxonHalf = AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical.WilcoxonSelector<Half>;
using SurfHalf = AiDotNet.Preprocessing.FeatureSelection.Filter.Relief.SURF<Half>;
using UnivariateGenericSelect = AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate.GenericUnivariateSelect<double>;
using UnivariateKBest = AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate.SelectKBest<double>;
using UnivariateKBestHalf = AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate.SelectKBest<Half>;
using UnivariateKruskalWallisH = AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate.KruskalWallisH<double>;
using UnivariateKruskalWallisTest = AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate.KruskalWallisTest<double>;
using UnivariateMannWhitneyU = AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate.MannWhitneyU<double>;
using UnivariateSelectMode = AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate.UnivariateSelectMode;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Reference tests for the fourth review round of the filter feature selectors: Half overflow in correlations,
/// variances, ranges, row counts, rank sums, DFT phases and corrections; scikit-learn's f_regression force_finite
/// rule and f_classif on continuous targets; input validation; the knockoff+ threshold and second-order knockoffs;
/// shortest-path counting; and the block-wise pair statistics checked against direct O(n^2) references.
/// </summary>
public class FeatureSelectionFilterRoundFourTests
{
    private const double EulerGamma = 0.57721566490153286060651209;

    private static Matrix<double> Columns(int n, params Func<int, double>[] columns)
    {
        var m = new Matrix<double>(n, columns.Length);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < columns.Length; j++)
                m[i, j] = columns[j](i);
        return m;
    }

    private static Vector<double> Values(int n, Func<int, double> value)
    {
        var v = new Vector<double>(n);
        for (int i = 0; i < n; i++)
            v[i] = value(i);
        return v;
    }

    private static double Correlation(double[] a, double[] b)
    {
        double ma = a.Average(), mb = b.Average(), sab = 0, saa = 0, sbb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            sab += (a[i] - ma) * (b[i] - mb);
            saa += (a[i] - ma) * (a[i] - ma);
            sbb += (b[i] - mb) * (b[i] - mb);
        }

        return sab / Math.Sqrt(saa * sbb);
    }

    // psi(m) = H(m - 1) - gamma for a positive integer m
    private static double Digamma(int m)
    {
        double h = 0;
        for (int i = 1; i < m; i++)
            h += 1.0 / i;
        return h - EulerGamma;
    }

    // ---- greedy selection, counts, ranges and row counts ----

    // Every candidate scores NaN: the -infinity mask on chosen features ranked above NaN, so feature 0 was chosen at
    // every step and Selected held duplicates.
    [Fact(Timeout = 60000)]
    public async Task Greedy_AllScoresNaN_NeverChoosesAFeatureTwice()
    {
        await Task.CompletedTask;
        var (selected, _) = DiscreteInformation<double>.Greedy(3, 3,
            _ => new Tensor<double>(new[] { double.NaN, double.NaN, double.NaN }, new[] { 3 }));

        Assert.Equal(new[] { 0, 1, 2 }, selected);
    }

    // The bincount tables equal the one-hot counts they replace, a row in no class counted nowhere.
    [Fact(Timeout = 60000)]
    public async Task CountTables_MatchDirectCounts()
    {
        await Task.CompletedTask;
        var codes = new Tensor<int>(new[] { 0, 1, 1, 1, 2, 0 }, new[] { 3, 2 });
        var counts = FeatureSelectionTensorOps.CodeCounts<double>(codes, 3);
        Assert.Equal(new[] { 1.0, 1.0, 1.0, 1.0, 2.0, 0.0 }, counts.ToArray());

        var single = new Tensor<int>(new[] { 0, 1, 1 }, new[] { 3, 1 });
        var membership = new Tensor<double>(new[] { 1.0, 0.0, 0.0, 0.0, 0.0, 1.0 }, new[] { 3, 2 });
        var joint = FeatureSelectionTensorOps.JointCounts(single, 2, membership);
        Assert.Equal(new[] { 1.0, 0.0, 0.0, 1.0 }, joint.ToArray());

        var paired = FeatureSelectionTensorOps.PairedJointCounts<double>(single, 2, single, 2);
        Assert.Equal(new[] { 1.0, 0.0, 0.0, 2.0 }, paired.ToArray());
    }

    // [-40000, 0, 40000] in Half: the range 80000 overflowed to infinity and every value fell in bin 0; the middle
    // value belongs in bin 5 of 10 and the maximum in the last bin.
    [Fact(Timeout = 60000)]
    public async Task Bins_Half_RangePastMaxValue_KeepTheirBins()
    {
        await Task.CompletedTask;
        var x = new Tensor<Half>(new[] { (Half)(-40000), (Half)0, (Half)40000 }, new[] { 3, 1 });

        var scaled = FeatureSelectionTensorOps.ScaledBins(x, 10);
        var equalWidth = FeatureSelectionTensorOps.EqualWidthBins(x, 10);

        Assert.Equal(new[] { 0, 5, 9 }, scaled.ToArray());
        Assert.Equal(new[] { 0, 5, 9 }, equalWidth.ToArray());
    }

    // 65536 rows in Half: the row count read as infinity, every share became 0 and so did every entropy. A feature
    // equal to a balanced binary target carries I = ln 2; an independent balanced one carries 0.
    [Fact(Timeout = 120000)]
    public async Task DiscreteInformation_Half_65536Rows_KeepsTheRelevance()
    {
        await Task.CompletedTask;
        const int n = 65536;
        var codes = new int[n * 2];
        var target = new Vector<Half>(n);
        for (int i = 0; i < n; i++)
        {
            codes[2 * i] = i % 2;
            codes[2 * i + 1] = (i / 2) % 2;
            target[i] = (Half)(i % 2);
        }

        var relevance = DiscreteInformation<Half>.ForTarget(new Tensor<int>(codes, new[] { n, 2 }), 2, target).Relevance();

        Assert.InRange((double)relevance[0], Math.Log(2) - 0.01, Math.Log(2) + 0.01);
        Assert.InRange((double)relevance[1], -0.01, 0.01);
    }

    // ---- correlations and variances in Half ----

    // Columns [-1, 1, -1, 1] and [-200, -200, 200, 200] with the second as the target: its centred sum of squares,
    // 160000, overflowed Half and r came out 0 or NaN instead of 1.
    [Fact(Timeout = 60000)]
    public async Task CfsAndPearson_Half_LargeDeviations_KeepTheirCorrelation()
    {
        await Task.CompletedTask;
        var data = new Matrix<Half>(4, 2);
        var target = new Vector<Half>(4);
        double[] a = { -1, 1, -1, 1 };
        double[] b = { -200, -200, 200, 200 };
        for (int i = 0; i < 4; i++)
        {
            data[i, 0] = (Half)a[i];
            data[i, 1] = (Half)b[i];
            target[i] = (Half)b[i];
        }

        var cfs = new CFS<Half>(nFeaturesToSelect: 1);
        cfs.Fit(data, target);
        Assert.NotNull(cfs.FeatureClassCorrelations);
        if (cfs.FeatureClassCorrelations is null)
            return;
        Assert.Equal(0.0, (double)cfs.FeatureClassCorrelations[0]);
        Assert.Equal(1.0, (double)cfs.FeatureClassCorrelations[1]);
        Assert.Equal(new[] { 1 }, cfs.SelectedIndices);

        var pearson = new FilterPearsonHalf(nFeaturesToSelect: 1);
        pearson.Fit(data, target);
        Assert.NotNull(pearson.Correlations);
        if (pearson.Correlations is null)
            return;
        Assert.Equal(0.0, (double)pearson.Correlations[0]);
        Assert.Equal(1.0, (double)pearson.Correlations[1]);
    }

    // Ten rows alternating [100, 200] and [-100, -200]: the sums of squared deviations (100000 and 400000) overflowed
    // Half although the sample variances, 11111.1 and 44444.4, are representable; both scores were infinite and the
    // tie went to column 0.
    [Fact(Timeout = 60000)]
    public async Task SelectKBest_Half_Variance_DoesNotOverflow()
    {
        await Task.CompletedTask;
        var data = new Matrix<Half>(10, 2);
        for (int i = 0; i < 10; i++)
        {
            double sign = i % 2 == 0 ? 1 : -1;
            data[i, 0] = (Half)(100 * sign);
            data[i, 1] = (Half)(200 * sign);
        }

        var selector = new UnivariateKBestHalf(k: 1);
        selector.Fit(data);

        Assert.Equal(new[] { 1 }, selector.SelectedFeatures);
        Assert.NotNull(selector.Scores);
        if (selector.Scores is null)
            return;
        Assert.InRange((double)selector.Scores[0], 100000.0 / 9 * 0.995, 100000.0 / 9 * 1.005);
        Assert.InRange((double)selector.Scores[1], 400000.0 / 9 * 0.995, 400000.0 / 9 * 1.005);
    }

    // ---- scikit-learn f_regression / f_classif ----

    // f_regression(force_finite=True) only replaces scores when some F is not finite. Both fits below are finite
    // (F about 4e5 and 1e7); the 1.001e-5 "perfect" test turned both into the largest double and the tie went to the
    // weaker column 0.
    [Fact(Timeout = 60000)]
    public async Task FRegression_NearPerfectFits_KeepFiniteScores()
    {
        await Task.CompletedTask;
        double[] y = { -3, -1, 1, 3 };
        double[] x0 = { -2.995, -1.005, 0.995, 3.005 };
        double[] x1 = { -2.999, -1.001, 0.999, 3.001 };
        var data = Columns(4, i => x0[i], i => x1[i]);

        var selector = new UnivariateKBest(k: 1, scoreFunc: SelectKBestScoreFunc.FRegression);
        selector.Fit(data, new Vector<double>(y));

        Assert.Equal(new[] { 1 }, selector.SelectedFeatures);
        Assert.NotNull(selector.Scores);
        if (selector.Scores is null)
            return;
        double r0 = Correlation(x0, y), r1 = Correlation(x1, y);
        Assert.Equal(1.0, selector.Scores[0] / (r0 * r0 / (1 - r0 * r0) * 2), 6);
        Assert.Equal(1.0, selector.Scores[1] / (r1 * r1 / (1 - r1 * r1) * 2), 6);
    }

    // A feature that varies at the 1e-8 scale and follows the target: its centred sum of squares (about 1e-12) fell
    // under the absolute 1e-10 cut-off and it scored 0 with p = 1. F is scale free.
    [Fact(Timeout = 60000)]
    public async Task FRegression_SmallScaleFeature_IsNotTreatedAsConstant()
    {
        await Task.CompletedTask;
        const int n = 50;
        var xs = Enumerable.Range(0, n).Select(i => 1e-8 * i + 1e-9 * ((7 * i) % 3)).ToArray();
        var ys = Enumerable.Range(0, n).Select(i => (double)i).ToArray();

        var (scores, pValues) = StatisticalTestHelper.ComputeScores(
            Columns(n, i => xs[i]), new Vector<double>(ys), SelectKBestScoreFunc.FRegression);

        double r = Correlation(xs, ys);
        Assert.Equal(1.0, scores[0] / (r * r / (1 - r * r) * (n - 2)), 6);
        Assert.NotNull(pValues);
        if (pValues is not null)
            Assert.True(pValues[0] < 1e-10);
    }

    // The same for f_classif: class means 1e-6 apart with spread at 1e-8 had a between-class mean square near 1e-11,
    // under the absolute cut-off, and scored 0. The reference is the one-way ANOVA F on the raw values.
    [Fact(Timeout = 60000)]
    public async Task FClassif_SmallScaleFeature_IsNotTreatedAsConstant()
    {
        await Task.CompletedTask;
        const int n = 50;
        var xs = Enumerable.Range(0, n).Select(i => 1e-6 * (i % 2) + 1e-8 * (i % 5)).ToArray();
        var classes = Enumerable.Range(0, n).Select(i => (double)(i % 2)).ToArray();

        var (scores, _) = StatisticalTestHelper.ComputeScores(
            Columns(n, _ => 5.0, i => xs[i]), new Vector<double>(classes), SelectKBestScoreFunc.FClassif);

        double mean = xs.Average();
        double ssBetween = 0, ssWithin = 0;
        for (int c = 0; c < 2; c++)
        {
            var group = xs.Where((_, i) => i % 2 == c).ToArray();
            double groupMean = group.Average();
            ssBetween += group.Length * (groupMean - mean) * (groupMean - mean);
            ssWithin += group.Sum(v => (v - groupMean) * (v - groupMean));
        }

        double expected = ssBetween / 1 / (ssWithin / (n - 2));
        Assert.Equal(0.0, scores[0]);
        Assert.Equal(1.0, scores[1] / expected, 6);
    }

    // Every distinct target value is a class for f_classif: a continuous target left no within-class degrees of
    // freedom and a constant one no second class, and both silently scored every feature 0 with p = 1.
    [Fact(Timeout = 60000)]
    public async Task FClassif_ContinuousOrConstantTarget_Throws()
    {
        await Task.CompletedTask;
        var data = Columns(6, i => i, i => i * i);

        Assert.Throws<ArgumentException>(() => new UnivariateKBest(k: 1).Fit(data, Values(6, i => 0.1 * i)));
        Assert.Throws<ArgumentException>(() => new UnivariateKBest(k: 1).Fit(data, Values(6, _ => 1.0)));
    }

    // ---- multiple testing ----

    // 65536 p-values in Half: Bonferroni multiplied by the count, which reads as infinity, so a zero p-value became
    // NaN and the perfectly predictive feature was rejected.
    [Fact(Timeout = 60000)]
    public async Task Corrections_Half_65536Tests_KeepAZeroPValue()
    {
        await Task.CompletedTask;
        const int n = 65536;
        var p = new Vector<Half>(n);
        for (int i = 1; i < n; i++)
            p[i] = (Half)0.5;

        var bonferroni = StatisticalTestHelper.BonferroniCorrection(p);
        var benjaminiHochberg = StatisticalTestHelper.BenjaminiHochbergCorrection(p);

        Assert.Equal(0.0, (double)bonferroni[0]);
        Assert.Equal(1.0, (double)bonferroni[1]);
        Assert.Equal(0.0, (double)benjaminiHochberg[0]);
    }

    // ---- input validation ----

    [Fact(Timeout = 60000)]
    public async Task PValueSelectors_RejectAScorerWithoutPValuesAtConstruction()
    {
        await Task.CompletedTask;
        Assert.Throws<ArgumentException>(() => new FilterSelectFdr(0.05, SelectKBestScoreFunc.MutualInfoRegression));
        Assert.Throws<ArgumentException>(() => new FilterSelectFpr(0.05, SelectKBestScoreFunc.MutualInfoRegression));
        Assert.Throws<ArgumentException>(() => new UnivariateGenericSelect(
            UnivariateSelectMode.Fdr, 0.05, SelectKBestScoreFunc.MutualInfoRegression));
    }

    // A refit whose scorer returns no p-values fails without touching the previous fit: the column count, scores and
    // selection used to change before the check, and the support mask then had the new width.
    [Fact(Timeout = 60000)]
    public async Task SelectFdr_FailedRefit_KeepsThePreviousFit()
    {
        await Task.CompletedTask;
        int calls = 0;
        var selector = new FilterSelectFdr(0.5, (data, target) =>
        {
            calls++;
            var scores = new Vector<double>(data.Columns);
            if (calls == 1)
                return (scores, new Vector<double>(data.Columns));
            return (scores, null);
        });
        var first = Columns(5, i => i, i => -i, i => i % 2, i => 1);
        selector.Fit(first, Values(5, i => i % 2));
        var selected = selector.SelectedIndices;
        var scoresBefore = selector.Scores;

        Assert.Throws<InvalidOperationException>(() => selector.Fit(Columns(5, i => i, i => -i, i => i % 2), Values(5, i => i % 2)));

        Assert.Same(scoresBefore, selector.Scores);
        Assert.Equal(selected, selector.SelectedIndices);
        Assert.Equal(4, selector.GetSupportMask().Length);
    }

    // scipy.stats.kruskal needs at least two groups: one class made H = 0 and p = 1 for every feature.
    [Fact(Timeout = 60000)]
    public async Task KruskalWallisSelectors_OneClass_Throw()
    {
        await Task.CompletedTask;
        var data = Columns(6, i => i, i => -i);
        var target = Values(6, _ => 1.0);

        Assert.Throws<ArgumentException>(() => new StatisticalKruskalWallis(nFeaturesToSelect: 1).Fit(data, target));
        Assert.Throws<ArgumentException>(() => new UnivariateKruskalWallisH(nFeaturesToSelect: 1).Fit(data, target));
        Assert.Throws<ArgumentException>(() => new UnivariateKruskalWallisTest(nFeaturesToSelect: 1).Fit(data, target));
    }

    // A third label was merged into class 1 without an error.
    [Fact(Timeout = 60000)]
    public async Task MannWhitneyU_ThreeClasses_Throws()
    {
        await Task.CompletedTask;
        Assert.Throws<ArgumentException>(() =>
            new UnivariateMannWhitneyU(nFeaturesToSelect: 1).Fit(Columns(6, i => i), Values(6, i => i % 3)));
    }

    // ---- rank sums ----

    // 50000 rows per class: n0 n1 = 2.5e9 overflowed int to a negative value, every score became 0 and the tie went to
    // the constant column 0 instead of the separating column 1.
    [Fact(Timeout = 120000)]
    public async Task Wilcoxon_50000RowsPerClass_ScoresTheSeparatingFeature()
    {
        await Task.CompletedTask;
        const int n = 100000;
        var selector = new StatisticalWilcoxon(nFeaturesToSelect: 1);
        selector.Fit(Columns(n, _ => 0.0, i => i % 2), Values(n, i => i % 2));

        Assert.Equal(new[] { 1 }, selector.SelectedIndices);
        Assert.NotNull(selector.UStatistics);
        if (selector.UStatistics is null)
            return;
        Assert.Equal(0.0, selector.UStatistics[0], 12);
        Assert.Equal(1.0, selector.UStatistics[1], 12);
    }

    // 400 rows per class in Half: the midpoint n0 n1 / 2 = 80000 overflowed and the normalised scores were NaN.
    [Fact(Timeout = 60000)]
    public async Task Wilcoxon_Half_400RowsPerClass_NormalisesInRange()
    {
        await Task.CompletedTask;
        const int n = 800;
        var data = new Matrix<Half>(n, 2);
        var target = new Vector<Half>(n);
        for (int i = 0; i < n; i++)
        {
            data[i, 1] = (Half)(i % 2);
            target[i] = (Half)(i % 2);
        }

        var selector = new StatisticalWilcoxonHalf(nFeaturesToSelect: 1);
        selector.Fit(data, target);

        Assert.NotNull(selector.UStatistics);
        if (selector.UStatistics is null)
            return;
        Assert.Equal(0.0, (double)selector.UStatistics[0]);
        Assert.Equal(1.0, (double)selector.UStatistics[1]);
    }

    // ---- spectral and argmax helpers ----

    // 0.1 cos(2 pi 3 i / n) over n = 1000 rows has |X(3)|^2 = (0.1 n / 2)^2 = 2500. The phase k i reached 5e5 in Half,
    // cos and sin of infinity were NaN, and so was every power value.
    [Fact(Timeout = 60000)]
    public async Task ColumnDftPower_Half_LongColumn_StaysFinite()
    {
        await Task.CompletedTask;
        const int n = 1000;
        var values = new Half[n];
        for (int i = 0; i < n; i++)
            values[i] = (Half)(0.1 * Math.Cos(2 * Math.PI * 3 * i / n));

        int kCount = FilterStatisticTensorOps.PositiveFrequencyCount(n);
        var power = FilterStatisticTensorOps.ColumnDftPower(new Tensor<Half>(values, new[] { n, 1 }), kCount);

        Assert.All(power.ToArray(), v => Assert.False(double.IsNaN((double)v) || double.IsInfinity((double)v)));
        Assert.InRange((double)power[2, 0], 2250.0, 2750.0);
    }

    // FirstArgMax ranks NaN below every number, like FilterStatisticTensorOps.FirstArgMax: the max-and-equals form
    // found no position when NaN was present and threw.
    [Fact(Timeout = 60000)]
    public async Task FirstArgMax_WithNaN_UsesTheStableRanking()
    {
        await Task.CompletedTask;
        Assert.Equal(0, PreprocessingTensorOps.FirstArgMax(new Tensor<double>(new[] { double.NaN, double.NaN }, new[] { 2 })));
        Assert.Equal(2, PreprocessingTensorOps.FirstArgMax(new Tensor<double>(new[] { double.NaN, 1.0, 3.0, 3.0 }, new[] { 4 })));
    }

    // A constant column of 1000s kept its raw values in the standardisation and scored |1000 sum(v)| / n, above every
    // informative feature; centred, it scores 0.
    [Fact(Timeout = 60000)]
    public async Task MultiClusterFS_ConstantColumn_ScoresZero()
    {
        await Task.CompletedTask;
        const int n = 30;
        var data = Columns(n, _ => 1000.0, i => i < 15 ? Math.Sin(i) : 5 + Math.Sin(i), i => Math.Cos(0.7 * i));

        var selector = new MultiClusterFS<double>(nFeaturesToSelect: 1, nClusters: 2, nNeighbors: 5, randomState: 3);
        selector.Fit(data);

        Assert.NotNull(selector.McfsScores);
        if (selector.McfsScores is null)
            return;
        Assert.Equal(0.0, selector.McfsScores[0]);
        Assert.DoesNotContain(0, selector.SelectedIndices ?? Array.Empty<int>());
    }

    // The head-row mean distance of SURF on 300 features: rows of opposite parity differ in every feature
    // (distance sqrt(300)), equal parity in none, so the mean is 2 50 50 sqrt(300) / (100 99). The plain sum,
    // 86603, overflowed Half and made the neighbour threshold infinite.
    [Fact(Timeout = 60000)]
    public async Task SurfMeanPairDistance_Half_DoesNotOverflow()
    {
        await Task.CompletedTask;
        const int s = 100;
        const int p = 300;
        var values = new Half[s * p];
        for (int i = 0; i < s; i++)
            for (int j = 0; j < p; j++)
                values[i * p + j] = (Half)((i + j) % 2);

        var mean = SurfHalf.MeanPairDistance(new Tensor<Half>(values, new[] { s, p }));

        double expected = 2.0 * 50 * 50 * Math.Sqrt(p) / (s * (s - 1));
        Assert.InRange((double)mean, expected * 0.99, expected * 1.01);
    }

    // ---- graph betweenness ----

    // A 250-clique with a 135-node path hanging off clique node 0. Every pair (left of q_k, right of q_k) routes
    // through path node q_k, so its betweenness is (250 + k - 1)(135 - k). Walk counts from the clique grow like
    // 249^step and overflowed double at step 130; inf * 0 = NaN then hid the pairs 136 steps apart.
    [Fact(Timeout = 300000)]
    public async Task Betweenness_DenseCliqueWithLongTail_ReachesEveryPair()
    {
        await Task.CompletedTask;
        const int clique = 250;
        const int tail = 135;
        int p = clique + tail;
        var adjacency = new double[p * p];
        for (int a = 0; a < clique; a++)
            for (int b = 0; b < clique; b++)
                if (a != b)
                    adjacency[a * p + b] = 1;
        for (int k = 0; k < tail; k++)
        {
            int node = clique + k;
            int previous = k == 0 ? 0 : node - 1;
            adjacency[node * p + previous] = 1;
            adjacency[previous * p + node] = 1;
        }

        var scores = GraphBasedFS<double>.BetweennessCentrality(new Tensor<double>(adjacency, new[] { p, p }));

        foreach (int k in new[] { 1, 67, 134 })
            Assert.Equal((double)(clique + k - 1) * (tail - k), scores[clique + k - 1], 6);
    }

    // ---- knockoffs ----

    // Barber and Candes (2015): T = min{t > 0 : (1 + #{W <= -t}) / max(1, #{W >= t}) <= q}. For W = [5, 4, 3, 2, 1, -0.5]
    // and q = 0.5, t = 0.5 gives 2 / 5 and passes, so the five positive features are selected; stopping at the
    // largest passing t (4, giving 1 / 2) selected two.
    [Fact(Timeout = 60000)]
    public async Task KnockoffPlus_UsesTheSmallestPassingThreshold()
    {
        await Task.CompletedTask;
        var w = new Tensor<double>(new[] { 5.0, 4.0, 3.0, 2.0, 1.0, -0.5 }, new[] { 6 });

        Assert.Equal(new[] { 0, 1, 2, 3, 4 }, Knockoffs.KnockoffPlusSelection(w, 0.5));
        Assert.Empty(Knockoffs.KnockoffPlusSelection(new Tensor<double>(new[] { 1.0, -1.0 }, new[] { 2 }), 0.5));
    }

    // Second-order knockoffs keep the correlation structure: corr(X~_0, X~_1) and corr(X_0, X~_1) both match
    // corr(X_0, X_1) (Candes et al. 2018). Independently permuted columns had correlations near 0.
    [Fact(Timeout = 60000)]
    public async Task GaussianKnockoffs_KeepTheCorrelationStructure()
    {
        await Task.CompletedTask;
        const int n = 4000;
        var x0 = Enumerable.Range(0, n).Select(i => Math.Sin(0.37 * i) + 0.5 * Math.Sin(1.91 * i)).ToArray();
        var x1 = Enumerable.Range(0, n).Select(i => x0[i] + 0.3 * Math.Cos(1.13 * i)).ToArray();
        var x2 = Enumerable.Range(0, n).Select(i => Math.Cos(0.71 * i)).ToArray();
        var x = Tensor<double>.FromMatrix(Columns(n, i => x0[i], i => x1[i], i => x2[i]));

        var knockoffs = Knockoffs.GaussianKnockoffs(x, 11);

        var k0 = Enumerable.Range(0, n).Select(i => knockoffs[i, 0]).ToArray();
        var k1 = Enumerable.Range(0, n).Select(i => knockoffs[i, 1]).ToArray();
        double target = Correlation(x0, x1);
        Assert.InRange(Correlation(k0, k1), target - 0.05, target + 0.05);
        Assert.InRange(Correlation(x0, k1), target - 0.05, target + 0.05);
    }

    // ---- block-wise pair statistics against direct references ----

    // Kendall tau-b over 3000 rows, which the block-wise sums cover in several blocks, against the direct pair loop.
    [Fact(Timeout = 120000)]
    public async Task KendallTau_ManyRows_MatchesTheDirectPairCount()
    {
        await Task.CompletedTask;
        const int n = 3000;
        var x0 = Enumerable.Range(0, n).Select(i => Math.Sin(0.7 * i)).ToArray();
        var x1 = Enumerable.Range(0, n).Select(i => (double)(i * 37 % 101)).ToArray();
        var y = Enumerable.Range(0, n).Select(i => Math.Cos(0.3 * i) + 0.001 * i).ToArray();

        var selector = new FilterKendallTau(nFeaturesToSelect: 1);
        selector.Fit(Columns(n, i => x0[i], i => x1[i]), new Vector<double>(y));

        Assert.NotNull(selector.Correlations);
        if (selector.Correlations is null)
            return;
        foreach (var (column, j) in new[] { (x0, 0), (x1, 1) })
        {
            double concordance = 0, xTies = 0, yTies = 0;
            for (int i = 0; i < n; i++)
                for (int k = i + 1; k < n; k++)
                {
                    double dx = column[k] - column[i], dy = y[k] - y[i];
                    bool tx = Math.Abs(dx) < 1e-10, ty = Math.Abs(dy) < 1e-10;
                    if (tx) xTies++;
                    if (ty) yTies++;
                    if (!tx && !ty) concordance += Math.Sign(dx) * Math.Sign(dy);
                }

            double pairs = (double)n * (n - 1) / 2;
            double tau = concordance / Math.Sqrt((pairs - xTies) * (pairs - yTies));
            Assert.Equal(tau, selector.Correlations[j], 10);
        }
    }

    // Relief over 3000 instances (every one sampled, several row blocks) against the direct nearest hit and miss
    // search on range-scaled features.
    [Fact(Timeout = 120000)]
    public async Task Relief_ManyRows_MatchesTheDirectNeighbourSearch()
    {
        await Task.CompletedTask;
        const int n = 3000;
        const int p = 3;
        var data = Columns(n, i => Math.Sin(0.31 * i), i => Math.Cos(0.17 * i) + 0.2 * (i % 2), i => Math.Sin(0.05 * i * i));
        var labels = Enumerable.Range(0, n).Select(i => (double)(i % 2)).ToArray();

        var selector = new ReliefSelector(nFeaturesToSelect: 1, randomState: 5);
        selector.Fit(data, new Vector<double>(labels));

        var scaled = new double[n, p];
        for (int j = 0; j < p; j++)
        {
            double min = double.MaxValue, max = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                min = Math.Min(min, data[i, j]);
                max = Math.Max(max, data[i, j]);
            }

            for (int i = 0; i < n; i++)
                scaled[i, j] = (data[i, j] - min) / (max - min);
        }

        var expected = new double[p];
        for (int i = 0; i < n; i++)
        {
            int hit = -1, miss = -1;
            double hitDistance = double.PositiveInfinity, missDistance = double.PositiveInfinity;
            for (int k = 0; k < n; k++)
            {
                if (k == i)
                    continue;
                double d = 0;
                for (int j = 0; j < p; j++)
                    d += (scaled[i, j] - scaled[k, j]) * (scaled[i, j] - scaled[k, j]);
                d = Math.Sqrt(d);
                if (labels[k] == labels[i] && d < hitDistance)
                {
                    hitDistance = d;
                    hit = k;
                }
                else if (labels[k] != labels[i] && d < missDistance)
                {
                    missDistance = d;
                    miss = k;
                }
            }

            for (int j = 0; j < p; j++)
                expected[j] += (Math.Abs(scaled[i, j] - scaled[miss, j]) - Math.Abs(scaled[i, j] - scaled[hit, j])) / n;
        }

        Assert.NotNull(selector.FeatureWeights);
        if (selector.FeatureWeights is null)
            return;
        for (int j = 0; j < p; j++)
            Assert.Equal(expected[j], selector.FeatureWeights[j], 9);
    }

    // The Kraskov estimator over 2500 rows (two row blocks, top-k radius) against the direct estimator 1 of Kraskov
    // et al. (2004) on unit-variance data.
    [Fact(Timeout = 120000)]
    public async Task MutualInfoRegression_ManyRows_MatchesTheDirectKraskovEstimate()
    {
        await Task.CompletedTask;
        const int n = 2500;
        const int k = 3;
        var x = Enumerable.Range(0, n).Select(i => Math.Sin(0.13 * i) + 0.01 * i).ToArray();
        var y = Enumerable.Range(0, n).Select(i => x[i] * x[i] + 0.3 * Math.Cos(0.91 * i)).ToArray();

        var mi = StatisticalTestHelper.MutualInfoRegression(Columns(n, i => x[i]), new Vector<double>(y), k);

        double Std(double[] v)
        {
            double m = v.Average();
            return Math.Sqrt(v.Sum(t => (t - m) * (t - m)) / v.Length);
        }

        double sx = Std(x), sy = Std(y);
        var xs = x.Select(v => v / sx).ToArray();
        var ys = y.Select(v => v / sy).ToArray();
        double psiSum = 0;
        var joint = new double[n - 1];
        for (int i = 0; i < n; i++)
        {
            int c = 0;
            for (int j = 0; j < n; j++)
                if (j != i)
                    joint[c++] = Math.Max(Math.Abs(xs[i] - xs[j]), Math.Abs(ys[i] - ys[j]));
            Array.Sort(joint);
            double radius = joint[k - 1];
            int nx = 0, ny = 0;
            for (int j = 0; j < n; j++)
            {
                if (j == i)
                    continue;
                if (Math.Abs(xs[i] - xs[j]) < radius) nx++;
                if (Math.Abs(ys[i] - ys[j]) < radius) ny++;
            }

            psiSum += Digamma(nx + 1) + Digamma(ny + 1);
        }

        double expected = Math.Max(0, Digamma(n) + Digamma(k) - psiSum / n);
        Assert.Equal(expected, mi[0], 9);
    }

    // The random state seeds scikit-learn's tie-breaking noise: the same seed gives the same scores, and on tied data
    // the noise changes them. The seed used to be stored and never read.
    [Fact(Timeout = 60000)]
    public async Task MutualInfoRegression_RandomState_SeedsTheTieBreakingNoise()
    {
        await Task.CompletedTask;
        const int n = 40;
        var data = Columns(n, i => i % 3);
        var target = Values(n, i => i % 3 + i % 2);

        MutualInfoRegression<double> Fit(int? seed)
        {
            var selector = new MutualInfoRegression<double>(nFeaturesToSelect: 1, nNeighbors: 3, randomState: seed);
            selector.Fit(data, target);
            return selector;
        }

        var first = Fit(7).MIScores;
        var second = Fit(7).MIScores;
        var unseeded = Fit(null).MIScores;
        Assert.NotNull(first);
        Assert.NotNull(second);
        Assert.NotNull(unseeded);
        if (first is null || second is null || unseeded is null)
            return;
        Assert.Equal(first[0], second[0]);
        Assert.NotEqual(unseeded[0], first[0]);
    }
}
