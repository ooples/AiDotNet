using AiDotNet.Preprocessing.FeatureSelection;
using AiDotNet.Preprocessing.FeatureSelection.Filter;
using AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical;
using AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;
using RegressionFRegression = AiDotNet.Preprocessing.FeatureSelection.Filter.Regression.FRegression<double>;
using StatisticalWelchTTest = AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical.WelchTTest<double>;
using UnivariateFRegression = AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate.FRegression<double>;
using UnivariateWelchTTest = AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate.WelchTTest<double>;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Reference values for the F, t and signal-to-noise feature scorers, computed with scipy 1.18 and
/// scikit-learn 1.9:
/// <list type="bullet">
/// <item>two-sample t-tests: <c>scipy.stats.ttest_ind(class0, class1)</c> (pooled) and
/// <c>ttest_ind(..., equal_var=False)</c> (Welch);</item>
/// <item>ANOVA F: <c>sklearn.feature_selection.f_classif</c>, p = <c>scipy.stats.f.sf</c>;</item>
/// <item>regression F: <c>sklearn.feature_selection.f_regression(..., force_finite=False)</c>.</item>
/// </list>
/// They pin the fixes made when these scorers moved to engine ops: exact F and t p-values in place of the
/// exponential, normal and unnormalised-beta approximations; +inf (not 0) for a feature that separates the
/// classes perfectly or fits the target exactly; and Welch-Satterthwaite degrees of freedom for small-scale data.
/// </summary>
public class FeatureSelectionStatisticReferenceTests
{
    private const double Tolerance = 1e-9;

    // Two classes (rows 0-4 target 0, rows 5-10 target 1).
    // Feature 0: overlapping noisy values; feature 1: 2 in class 0 and 5 in class 1 (no within-class spread,
    // perfect separation); feature 2: constant; feature 3: noisy with unequal class variances.
    private static readonly double[,] TwoClassData =
    {
        { 1.2, 2.0, 7.0, 10.5 }, { 2.9, 2.0, 7.0, 13.1 }, { 2.1, 2.0, 7.0, 9.4 }, { 3.4, 2.0, 7.0, 11.8 },
        { 1.7, 2.0, 7.0, 12.2 }, { 3.1, 5.0, 7.0, 14.9 }, { 4.4, 5.0, 7.0, 21.3 }, { 2.6, 5.0, 7.0, 17.0 },
        { 3.9, 5.0, 7.0, 12.7 }, { 5.2, 5.0, 7.0, 19.6 }, { 4.0, 5.0, 7.0, 16.1 }
    };

    private static readonly double[] TwoClassTarget = { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1 };

    // ttest_ind(class0, class1): feature 0 and feature 3
    private const double PooledT0 = -2.9172204537585156;
    private const double PooledP0 = 0.01710902245806547;
    private const double PooledT3 = -3.6173485848853932;
    private const double PooledP3 = 0.005594829394012269;

    // ttest_ind(class0, class1, equal_var=False): feature 0 and feature 3
    private const double WelchT0 = -2.928265663287489;
    private const double WelchP0 = 0.01728902907930985;
    private const double WelchDf0 = 8.759204540886433;
    private const double WelchT3 = -3.8589001685289532;
    private const double WelchP3 = 0.005703115846299661;

    // Three classes of four rows. Feature 0: noisy; feature 1: 1, 2, 3 per class (perfect separation);
    // feature 2: constant.
    private static readonly double[,] ThreeClassData =
    {
        { 1.0, 1.0, 4.0 }, { 2.5, 1.0, 4.0 }, { 1.8, 1.0, 4.0 }, { 2.2, 1.0, 4.0 },
        { 3.1, 2.0, 4.0 }, { 2.4, 2.0, 4.0 }, { 3.9, 2.0, 4.0 }, { 3.3, 2.0, 4.0 },
        { 4.2, 3.0, 4.0 }, { 5.0, 3.0, 4.0 }, { 3.6, 3.0, 4.0 }, { 4.4, 3.0, 4.0 }
    };

    private static readonly double[] ThreeClassTarget = { 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2 };

    // f_classif on ThreeClassData, feature 0: F(2, 9) and scipy.stats.f.sf(F, 2, 9)
    private const double AnovaF0 = 15.524890190336784;
    private const double AnovaP0 = 0.0012089008067610488;

    // Regression: feature 0 noisy, feature 1 = (y - 1) / 2 exactly (perfect linear fit), feature 2 constant.
    private static readonly double[,] RegressionData =
    {
        { 0.5, 0.5, 3.0 }, { 1.7, 1.5, 3.0 }, { 2.2, 2.5, 3.0 }, { 3.9, 3.0, 3.0 },
        { 4.1, 4.5, 3.0 }, { 5.6, 5.0, 3.0 }, { 6.3, 6.5, 3.0 }, { 7.8, 7.0, 3.0 }
    };

    private static readonly double[] RegressionTarget = { 2.0, 4.0, 6.0, 7.0, 10.0, 11.0, 14.0, 15.0 };

    // f_regression(RegressionData, RegressionTarget, force_finite=False), feature 0; R^2 from linregress
    private const double RegressionF0 = 141.876821879505;
    private const double RegressionP0 = 2.1199453328716933e-05;
    private const double RegressionR2 = 0.959425690086245;

    private static Matrix<double> M(double[,] data) => new Matrix<double>(data);

    private static Vector<double> V(double[] values) => new Vector<double>(values);

    private static Matrix<double> Scaled(double[,] data, double factor)
    {
        int rows = data.GetLength(0);
        int cols = data.GetLength(1);
        var scaled = new double[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                scaled[i, j] = data[i, j] * factor;
        return new Matrix<double>(scaled);
    }

    private static void AssertRelative(double expected, double actual, double relative = Tolerance)
    {
        Assert.True(Math.Abs(actual - expected) <= relative * Math.Abs(expected),
            $"Expected {expected:R}, got {actual:R}");
    }

    // ---------------------------------------------------------------- two-sample t-tests

    [Fact(Timeout = 120000)]
    public async Task UnivariateTTest_MatchesScipyPooledTTest()
    {
        var selector = new TTest<double>(nFeaturesToSelect: 1);
        selector.Fit(M(TwoClassData), V(TwoClassTarget));

        Assert.NotNull(selector.TStatistics);
        Assert.NotNull(selector.PValues);
        AssertRelative(PooledT0, selector.TStatistics[0]);
        AssertRelative(PooledP0, selector.PValues[0]);
        AssertRelative(PooledT3, selector.TStatistics[3]);
        AssertRelative(PooledP3, selector.PValues[3]);

        // perfect separation: t = -inf (mean0 < mean1), p = 0; constant feature: t = 0, p = 1
        Assert.True(double.IsNegativeInfinity(selector.TStatistics[1]), $"t = {selector.TStatistics[1]}");
        Assert.Equal(0.0, selector.PValues[1], Tolerance);
        Assert.Equal(0.0, selector.TStatistics[2], Tolerance);
        Assert.Equal(1.0, selector.PValues[2], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task UnivariateWelchTTest_MatchesScipyWelchTTest()
    {
        var selector = new UnivariateWelchTTest(nFeaturesToSelect: 1);
        selector.Fit(M(TwoClassData), V(TwoClassTarget));

        Assert.NotNull(selector.TStatistics);
        Assert.NotNull(selector.PValues);
        Assert.NotNull(selector.DegreesOfFreedom);
        AssertRelative(WelchT0, selector.TStatistics[0]);
        AssertRelative(WelchP0, selector.PValues[0]);
        AssertRelative(WelchDf0, selector.DegreesOfFreedom[0]);
        AssertRelative(WelchT3, selector.TStatistics[3]);
        AssertRelative(WelchP3, selector.PValues[3]);
        Assert.True(double.IsNegativeInfinity(selector.TStatistics[1]), $"t = {selector.TStatistics[1]}");
        Assert.Equal(0.0, selector.PValues[1], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task UnivariateWelchTTest_SmallScaleData_KeepsWelchSatterthwaiteDegreesOfFreedom()
    {
        // Scaling the data by 1e-3 leaves t, df and p unchanged (ttest_ind is scale invariant), but makes the
        // Welch denominator s0^4/(n0-1) + s1^4/(n1-1) ~ 1e-14. Master fell back to df = 1 below 1e-10, which
        // turned p = 0.0173 into p ~ 0.21.
        var selector = new UnivariateWelchTTest(nFeaturesToSelect: 1);
        selector.Fit(Scaled(TwoClassData, 1e-3), V(TwoClassTarget));

        Assert.NotNull(selector.DegreesOfFreedom);
        Assert.NotNull(selector.PValues);
        AssertRelative(WelchDf0, selector.DegreesOfFreedom[0], 1e-7);
        AssertRelative(WelchP0, selector.PValues[0], 1e-7);
    }

    [Fact(Timeout = 120000)]
    public async Task StatisticalStudentTTest_UsesStudentTDistribution()
    {
        // Master used a normal approximation that ignored the degrees of freedom.
        var pooled = new StudentTTest<double>(nFeaturesToSelect: 1, equalVariances: true);
        pooled.Fit(M(TwoClassData), V(TwoClassTarget));

        Assert.NotNull(pooled.TStatistics);
        Assert.NotNull(pooled.PValues);
        AssertRelative(PooledT0, pooled.TStatistics[0]);
        AssertRelative(PooledP0, pooled.PValues[0]);
        AssertRelative(PooledP3, pooled.PValues[3]);
        Assert.True(double.IsNegativeInfinity(pooled.TStatistics[1]), $"t = {pooled.TStatistics[1]}");

        var welch = new StudentTTest<double>(nFeaturesToSelect: 1, equalVariances: false);
        welch.Fit(M(TwoClassData), V(TwoClassTarget));

        Assert.NotNull(welch.TStatistics);
        Assert.NotNull(welch.PValues);
        AssertRelative(WelchT0, welch.TStatistics[0]);
        AssertRelative(WelchP0, welch.PValues[0]);
        AssertRelative(WelchP3, welch.PValues[3]);

        // smallest p-value first: the perfectly separating feature
        Assert.NotNull(pooled.SelectedIndices);
        Assert.Equal(new[] { 1 }, pooled.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task TTestSelector_ReportsWelchStatisticWithStudentTPValue()
    {
        var selector = new TTestSelector<double>(nFeaturesToSelect: 1);
        selector.Fit(M(TwoClassData), V(TwoClassTarget));

        Assert.NotNull(selector.TStatistics);
        Assert.NotNull(selector.PValues);
        AssertRelative(Math.Abs(WelchT0), selector.TStatistics[0]);
        AssertRelative(WelchP0, selector.PValues[0]);
        AssertRelative(WelchP3, selector.PValues[3]);
        Assert.True(double.IsPositiveInfinity(selector.TStatistics[1]), $"|t| = {selector.TStatistics[1]}");
        Assert.NotNull(selector.SelectedIndices);
        Assert.Equal(new[] { 1 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task StatisticalWelchTTest_MatchesScipyWelchTTest()
    {
        var selector = new StatisticalWelchTTest(nFeaturesToSelect: 1);
        selector.Fit(M(TwoClassData), V(TwoClassTarget));

        Assert.NotNull(selector.TStatistics);
        Assert.NotNull(selector.PValues);
        Assert.NotNull(selector.DegreesOfFreedom);
        AssertRelative(WelchT0, selector.TStatistics[0]);
        AssertRelative(WelchP0, selector.PValues[0]);
        AssertRelative(WelchDf0, selector.DegreesOfFreedom[0]);
        Assert.True(double.IsNegativeInfinity(selector.TStatistics[1]), $"t = {selector.TStatistics[1]}");
        Assert.NotNull(selector.SelectedIndices);
        Assert.Equal(new[] { 1 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task SignalToNoiseRatio_PerfectSeparationRanksFirst()
    {
        // SNR = |mean0 - mean1| / (std0 + std1) with population standard deviations (Golub et al. 1999).
        // Feature 0: |2.26 - 3.8667| / (0.7552 + 0.8826) = 0.97942104101...
        var selector = new SignalToNoiseRatio<double>(nFeaturesToSelect: 1);
        selector.Fit(M(TwoClassData), V(TwoClassTarget));

        Assert.NotNull(selector.SNRValues);
        AssertRelative(0.9794210410103084, selector.SNRValues[0]);
        AssertRelative(1.3302453008914197, selector.SNRValues[3]);
        Assert.True(double.IsPositiveInfinity(selector.SNRValues[1]), $"SNR = {selector.SNRValues[1]}");
        Assert.Equal(0.0, selector.SNRValues[2], Tolerance);
        Assert.NotNull(selector.SelectedIndices);
        Assert.Equal(new[] { 1 }, selector.SelectedIndices);
    }

    // ---------------------------------------------------------------- ANOVA F

    [Fact(Timeout = 120000)]
    public async Task UnivariateAnova_MatchesSklearnFClassif()
    {
        var selector = new ANOVA<double>(nFeaturesToSelect: 1);
        selector.Fit(M(ThreeClassData), V(ThreeClassTarget));

        Assert.NotNull(selector.FStatistics);
        Assert.NotNull(selector.PValues);
        AssertRelative(AnovaF0, selector.FStatistics[0]);
        AssertRelative(AnovaP0, selector.PValues[0]);

        // f_classif reports F = inf, p = 0 for a feature with no within-class scatter
        Assert.True(double.IsPositiveInfinity(selector.FStatistics[1]), $"F = {selector.FStatistics[1]}");
        Assert.Equal(0.0, selector.PValues[1], Tolerance);
        Assert.Equal(0.0, selector.FStatistics[2], Tolerance);
        Assert.Equal(1.0, selector.PValues[2], Tolerance);
        Assert.NotNull(selector.SelectedIndices);
        Assert.Equal(new[] { 1 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task AnovaFTest_PValueIsNormalizedFSurvival()
    {
        // Master's incomplete beta lacked the 1 / B(a, b) front factor. The F statistic keeps master's
        // 1e-10 stabiliser in the denominator, so it matches f_classif to about 1e-10 relative.
        var selector = new ANOVAFTest<double>(nFeaturesToSelect: 1);
        selector.Fit(M(ThreeClassData), V(ThreeClassTarget));

        Assert.NotNull(selector.FStatistics);
        Assert.NotNull(selector.PValues);
        AssertRelative(AnovaF0, selector.FStatistics[0], 1e-8);
        AssertRelative(AnovaP0, selector.PValues[0], 1e-8);
    }

    [Fact(Timeout = 120000)]
    public async Task AnovaSelector_PerfectSeparationScoresInfinity()
    {
        var selector = new ANOVASelector<double>(nFeaturesToSelect: 1);
        selector.Fit(M(ThreeClassData), V(ThreeClassTarget));

        Assert.NotNull(selector.FStatistics);
        AssertRelative(AnovaF0, selector.FStatistics[0]);
        Assert.True(double.IsPositiveInfinity(selector.FStatistics[1]), $"F = {selector.FStatistics[1]}");
        Assert.NotNull(selector.SelectedIndices);
        Assert.Equal(new[] { 1 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task GenericUnivariateSelect_FClassif_MatchesSklearn()
    {
        var selector = new GenericUnivariateSelect<double>(
            mode: SelectionMode.KBest, param: 1, defaultScoreFunc: "f_classif");
        selector.Fit(M(ThreeClassData), V(ThreeClassTarget));

        Assert.NotNull(selector.Scores);
        Assert.NotNull(selector.PValues);
        AssertRelative(AnovaF0, selector.Scores[0]);
        AssertRelative(AnovaP0, selector.PValues[0]);
        Assert.True(double.IsPositiveInfinity(selector.Scores[1]), $"F = {selector.Scores[1]}");
        Assert.Equal(0.0, selector.PValues[1], Tolerance);
        Assert.NotNull(selector.SelectedIndices);
        Assert.Equal(new[] { 1 }, selector.SelectedIndices);
    }

    // ---------------------------------------------------------------- regression F

    [Fact(Timeout = 120000)]
    public async Task UnivariateFRegression_MatchesSklearnFRegression()
    {
        var selector = new UnivariateFRegression(nFeaturesToSelect: 1);
        selector.Fit(M(RegressionData), V(RegressionTarget));

        Assert.NotNull(selector.FStatistics);
        Assert.NotNull(selector.PValues);
        Assert.NotNull(selector.RSquaredValues);
        AssertRelative(RegressionF0, selector.FStatistics[0], 1e-8);
        AssertRelative(RegressionP0, selector.PValues[0], 1e-7);
        AssertRelative(RegressionR2, selector.RSquaredValues[0], 1e-8);

        // perfect linear fit: F = inf, p = 0, R^2 = 1; constant feature: F = 0, p = 1
        Assert.True(double.IsPositiveInfinity(selector.FStatistics[1]), $"F = {selector.FStatistics[1]}");
        Assert.Equal(0.0, selector.PValues[1], Tolerance);
        Assert.Equal(1.0, selector.RSquaredValues[1], 1e-9);
        Assert.Equal(0.0, selector.FStatistics[2], Tolerance);
        Assert.Equal(1.0, selector.PValues[2], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task RegressionFRegression_MatchesSklearnFRegression()
    {
        var selector = new RegressionFRegression(nFeaturesToSelect: 1);
        selector.Fit(M(RegressionData), V(RegressionTarget));

        Assert.NotNull(selector.FScores);
        Assert.NotNull(selector.PValues);
        AssertRelative(RegressionF0, selector.FScores[0], 1e-8);
        AssertRelative(RegressionP0, selector.PValues[0], 1e-7);
        Assert.True(double.IsPositiveInfinity(selector.FScores[1]), $"F = {selector.FScores[1]}");
        Assert.Equal(0.0, selector.PValues[1], Tolerance);
        Assert.NotNull(selector.SelectedIndices);
        Assert.Equal(new[] { 1 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task FTest_MatchesSklearnFRegression()
    {
        var selector = new FTest<double>(nFeaturesToSelect: 1);
        selector.Fit(M(RegressionData), V(RegressionTarget));

        Assert.NotNull(selector.FScores);
        Assert.NotNull(selector.PValues);
        AssertRelative(RegressionF0, selector.FScores[0], 1e-8);
        AssertRelative(RegressionP0, selector.PValues[0], 1e-7);
        Assert.True(double.IsPositiveInfinity(selector.FScores[1]), $"F = {selector.FScores[1]}");
        Assert.NotNull(selector.SelectedIndices);
        Assert.Equal(new[] { 1 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task GenericUnivariateSelect_FRegression_MatchesSklearn()
    {
        var selector = new GenericUnivariateSelect<double>(
            mode: SelectionMode.KBest, param: 1, defaultScoreFunc: "f_regression");
        selector.Fit(M(RegressionData), V(RegressionTarget));

        Assert.NotNull(selector.Scores);
        Assert.NotNull(selector.PValues);
        AssertRelative(RegressionF0, selector.Scores[0], 1e-8);
        AssertRelative(RegressionP0, selector.PValues[0], 1e-7);
        Assert.True(double.IsPositiveInfinity(selector.Scores[1]), $"F = {selector.Scores[1]}");
        Assert.NotNull(selector.SelectedIndices);
        Assert.Equal(new[] { 1 }, selector.SelectedIndices);
    }

    // ---------------------------------------------------------------- FDR

    [Fact(Timeout = 120000)]
    public async Task GenericUnivariateSelect_Fdr_IsBenjaminiHochbergStepUp()
    {
        // p = [0.02, 0.06, 0.07, 0.5], alpha = 0.1, m = 4: BH thresholds 0.025, 0.05, 0.075, 0.1.
        // Sorted: 0.02 passes, 0.06 fails, 0.07 passes, 0.5 fails. The step-up rule keeps every feature up to
        // the largest passing rank (3), as sklearn.feature_selection.SelectFdr(alpha=0.1) does: {0, 1, 2}.
        // Master stopped at the first failure and kept only {0}.
        var data = M(new double[,] { { 1, 2, 3, 4 }, { 2, 3, 4, 5 } });
        var target = V(new double[] { 0, 1 });
        var pValues = new Vector<double>(new[] { 0.02, 0.06, 0.07, 0.5 });
        var scores = new Vector<double>(new[] { 4.0, 3.0, 2.0, 1.0 });

        var selector = new GenericUnivariateSelect<double>(
            mode: SelectionMode.FDR, param: 0.1, scoreFunc: (m, t) => (scores, pValues));
        selector.Fit(data, target);

        Assert.NotNull(selector.SelectedIndices);
        Assert.Equal(new[] { 0, 1, 2 }, selector.SelectedIndices);
    }
}
