using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using ANCOVA = AiDotNet.Preprocessing.FeatureSelection.Statistical.ANCOVASelector<double>;
using ANOVA = AiDotNet.Preprocessing.FeatureSelection.Statistical.ANOVASelector<double>;
using BackwardElimination = AiDotNet.Preprocessing.FeatureSelection.Wrapper.BackwardElimination<double>;
using Bhattacharyya = AiDotNet.Preprocessing.FeatureSelection.Supervised.BhattacharyyaDistanceSelector<double>;
using ChiSquareText = AiDotNet.Preprocessing.FeatureSelection.Text.ChiSquareText<double>;
using CorrelationFilter = AiDotNet.Preprocessing.FeatureSelection.Unsupervised.CorrelationFilter<double>;
using ElasticNet = AiDotNet.Preprocessing.FeatureSelection.Sparsity.ElasticNetSelector<double>;
using ExhaustiveFeatureSelector = AiDotNet.Preprocessing.FeatureSelection.Wrapper.ExhaustiveFeatureSelector<double>;
using ForwardSelection = AiDotNet.Preprocessing.FeatureSelection.Wrapper.ForwardSelection<double>;
using FusedLasso = AiDotNet.Preprocessing.FeatureSelection.Sparse.FusedLasso<double>;
using Granger = AiDotNet.Preprocessing.FeatureSelection.TimeSeries.GrangerCausalitySelector<double>;
using GroupLasso = AiDotNet.Preprocessing.FeatureSelection.Sparsity.GroupLassoSelector<double>;
using Hotelling = AiDotNet.Preprocessing.FeatureSelection.Statistical.HotellingT2Selector<double>;
using Jackknife = AiDotNet.Preprocessing.FeatureSelection.Stability.JackknifeSelector<double>;
using KendallTau = AiDotNet.Preprocessing.FeatureSelection.Statistical.KendallTauSelector<double>;
using KruskalWallis = AiDotNet.Preprocessing.FeatureSelection.Supervised.KruskalWallisSelector<double>;
using L1 = AiDotNet.Preprocessing.FeatureSelection.Sparsity.L1RegularizationSelector<double>;
using MaximalInformation = AiDotNet.Preprocessing.FeatureSelection.Unsupervised.MaximalInformation<double>;
using MinimalRedundancy = AiDotNet.Preprocessing.FeatureSelection.Unsupervised.MinimalRedundancy<double>;
using OnlineSelector = AiDotNet.Preprocessing.FeatureSelection.Streaming.OnlineFeatureSelector<double>;
using RandomizedLasso = AiDotNet.Preprocessing.FeatureSelection.Stability.RandomizedLassoSelector<double>;
using RFE = AiDotNet.Preprocessing.FeatureSelection.Wrapper.RFE<double>;
using RFECV = AiDotNet.Preprocessing.FeatureSelection.Wrapper.RFECV<double>;
using RFEImportanceMethod = AiDotNet.Preprocessing.FeatureSelection.Wrapper.RFEImportanceMethod;
using SparseGroupLasso = AiDotNet.Preprocessing.FeatureSelection.Sparse.SparseGroupLasso<double>;
using SparseGroupLassoSelector = AiDotNet.Preprocessing.FeatureSelection.Sparse.SparseGroupLassoSelector<double>;
using StatisticalTTest = AiDotNet.Preprocessing.FeatureSelection.Statistical.TTestSelector<double>;
using SupervisedTTest = AiDotNet.Preprocessing.FeatureSelection.Supervised.TTestSelector<double>;
using Wavelet = AiDotNet.Preprocessing.FeatureSelection.Wavelet.WaveletCoefficientSelector<double>;
using Wilcoxon = AiDotNet.Preprocessing.FeatureSelection.Supervised.WilcoxonSelector<double>;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Reference values for the feature selectors fixed in the element-wise sweep (plan 09b). Expected values come from
/// scikit-learn 1.9.1, scipy 1.18.1 and numpy, as quoted in each test.
/// </summary>
public class FeatureSelectionSweep09bReferenceTests
{
    private const double Tolerance = 1e-9;

    // Iterative solvers (FISTA, ADMM) stop on a coefficient-change tolerance, so their coefficients are compared
    // at the accuracy that stopping rule guarantees rather than to machine precision.
    private const double SolverTolerance = 1e-4;

    private static Matrix<double> TwoClassData() => new Matrix<double>(new double[,]
    {
        { 1, 2, 3.5 }, { 2, 2, 1.0 }, { 2, 5, 2.5 }, { 3, 1, 4.0 }, { 4, 2, 0.5 },
        { 5, 6, 3.0 }, { 5, 7, 1.5 }, { 6, 6, 2.0 }, { 7, 8, 5.0 }, { 9, 7, 0.0 }
    });

    private static Vector<double> TwoClassTarget() => new Vector<double>(new[] { 0.0, 0, 0, 0, 0, 1, 1, 1, 1, 1 });

    private static Matrix<double> RegressionData() => new Matrix<double>(new double[,]
    {
        { 1.0, 0.2, -1.0, 3.0 }, { 2.0, -0.4, 0.5, 1.0 }, { 0.5, 1.1, 2.0, -1.0 }, { -1.0, 0.7, 1.5, 0.0 },
        { 1.5, -1.3, -0.5, 2.0 }, { -0.5, 0.9, 1.0, -2.0 }, { 3.0, -0.2, 0.0, 1.5 }, { 0.0, 1.6, -1.5, 0.5 }
    });

    private static Vector<double> RegressionTarget() =>
        new Vector<double>(new[] { 2.1, 1.0, 0.3, -1.4, 2.6, -1.5, 3.9, -0.2 });

    private static void AssertValues(double[] expected, Vector<double>? actual, double tolerance)
    {
        Assert.NotNull(actual);
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task ANOVAAndHotelling_PerfectSeparator_ScoreInfinity()
    {
        // sklearn.feature_selection.f_classif(X, y)[0] -> [inf, 0.01250000000000029]
        var data = new Matrix<double>(new double[,]
        {
            { 1.0, 0.3 }, { 1.0, 0.9 }, { 1.0, 0.1 }, { 2.0, 0.4 }, { 2.0, 0.8 }, { 2.0, 0.2 }
        });
        var target = new Vector<double>(new[] { 0.0, 0, 0, 1, 1, 1 });

        var anova = new ANOVA(nFeaturesToSelect: 1);
        anova.Fit(data, target);
        Assert.NotNull(anova.FStatistics);
        Assert.True(double.IsPositiveInfinity(anova.FStatistics[0]));
        Assert.Equal(0.01250000000000029, anova.FStatistics[1], Tolerance);
        Assert.Equal(new[] { 0 }, anova.SelectedIndices);

        var hotelling = new Hotelling(nFeaturesToSelect: 1);
        hotelling.Fit(data, target);
        Assert.NotNull(hotelling.T2Contributions);
        Assert.True(double.IsPositiveInfinity(hotelling.T2Contributions[0]));
        Assert.Equal(new[] { 0 }, hotelling.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task ANCOVA_UsesCommonWithinClassSlope()
    {
        // numpy OLS: F = ((RSS(y ~ 1 + cov) - RSS(y ~ 1 + cov + class dummies)) / (k - 1)) / (RSS_full / (n - k - 1))
        //   -> [748.3854166666686, 1.3249795819595265] for features 1 and 2 (covariate 0)
        var data = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 0.5 }, { 2.0, 2.9, 1.5 }, { 3.0, 4.2, 0.0 }, { 1.5, 4.1, 2.0 }, { 2.5, 5.0, 1.0 },
            { 3.5, 6.2, 3.0 }, { 1.0, 6.0, 0.7 }, { 2.0, 7.1, 2.2 }, { 3.0, 7.9, 1.1 }
        });
        var target = new Vector<double>(new[] { 0.0, 0, 0, 1, 1, 1, 2, 2, 2 });

        var selector = new ANCOVA(nFeaturesToSelect: 1, covariateIndex: 0);
        selector.Fit(data, target);

        Assert.NotNull(selector.AdjustedFStatistics);
        Assert.Equal(0.0, selector.AdjustedFStatistics[0], Tolerance);
        Assert.Equal(748.3854166666686, selector.AdjustedFStatistics[1], 1e-7);
        Assert.Equal(1.3249795819595265, selector.AdjustedFStatistics[2], Tolerance);
        Assert.Equal(new[] { 1 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task KendallTau_JointTies_MatchScipyTauB()
    {
        // abs(scipy.stats.kendalltau(X[:, j], y).statistic) -> [0.7624928516630234, 0.7905694150420948, 0.0298142396999972]
        var selector = new KendallTau(nFeaturesToSelect: 2);
        selector.Fit(TwoClassData(), TwoClassTarget());

        AssertValues(new[] { 0.7624928516630234, 0.7905694150420948, 0.0298142396999972 }, selector.TauScores, Tolerance);
        Assert.Equal(new[] { 0, 1 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task StatisticalTTest_PValue_UsesWelchStudentT()
    {
        // scipy.stats.ttest_ind(a, b, equal_var=False): |t| = [4.417261042993862, 5.680375574437545, 0],
        // p = [0.003032744099460918, 0.0011286034321094532, 1]
        var selector = new StatisticalTTest(nFeaturesToSelect: 2);
        selector.Fit(TwoClassData(), TwoClassTarget());

        AssertValues(new[] { 4.417261042993862, 5.680375574437545, 0.0 }, selector.TStatistics, Tolerance);
        AssertValues(new[] { 0.003032744099460918, 0.0011286034321094532, 1.0 }, selector.PValues, Tolerance);
        Assert.Equal(new[] { 0, 1 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task SupervisedTTest_MatchesScipyTTest()
    {
        // scipy.stats.ttest_ind |t| = [4.417261042993862, 5.680375574437545, 0] (pooled and Welch agree for equal class
        // sizes); the selector divides by se + 1e-10: [4.417261042506057, 5.680375573704212, 0]
        var expected = new[] { 4.417261042506057, 5.680375573704212, 0.0 };
        foreach (bool equalVariance in new[] { false, true })
        {
            var selector = new SupervisedTTest(nFeaturesToSelect: 2, equalVariance: equalVariance);
            selector.Fit(TwoClassData(), TwoClassTarget());
            AssertValues(expected, selector.TStatistics, Tolerance);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task KruskalWallis_Ties_MatchScipyKruskal()
    {
        // scipy.stats.kruskal(a, b).statistic -> [6.90184049079754, 7.075471698113202, 0.010909090909095198]
        var selector = new KruskalWallis(nFeaturesToSelect: 2);
        selector.Fit(TwoClassData(), TwoClassTarget());

        AssertValues(new[] { 6.90184049079754, 7.075471698113202, 0.010909090909095198 }, selector.HStatistics, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task Wilcoxon_Ties_UseScipyTieCorrectedVariance()
    {
        // z = norm.isf(mannwhitneyu(b, a, use_continuity=False, method='asymptotic').pvalue / 2)
        //   -> [2.6271354153902204, 2.6599758829946576, 0.10444659357341869]
        var selector = new Wilcoxon(nFeaturesToSelect: 2);
        selector.Fit(TwoClassData(), TwoClassTarget());

        AssertValues(new[] { 2.6271354153902204, 2.6599758829946576, 0.10444659357341869 }, selector.TestStatistics, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task Bhattacharyya_MeanTerm_IsQuarterSquaredDifferenceOverVarianceSum()
    {
        // Bhattacharyya distance of two univariate Gaussians (sample variances):
        // 1/4 ln(1/4 (v1/v2 + v2/v1 + 2)) + 1/4 (m1 - m2)^2 / (v1 + v2)
        //   -> [1.011533732260557, 1.6970073426383228, 0.009671970403308926]
        var selector = new Bhattacharyya(nFeaturesToSelect: 2);
        selector.Fit(TwoClassData(), TwoClassTarget());

        AssertValues(new[] { 1.011533732260557, 1.6970073426383228, 0.009671970403308926 }, selector.DistanceScores, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task ChiSquareText_WeightsClassesByPrior()
    {
        // chi2_avg(t) = sum_c P(c) chi2(t, c) over the one-vs-rest 2x2 tables (Yang and Pedersen, 1997), each chi2 as
        // scipy.stats.chi2_contingency(correction=False) -> [0.11666666666666668, 2.770833333333333, 2.770833333333333,
        // 0.1166666666666667]
        var data = new Matrix<double>(new double[,]
        {
            { 1, 0, 2, 0 }, { 0, 0, 1, 1 }, { 3, 1, 0, 0 }, { 0, 2, 0, 1 }, { 1, 1, 1, 0 }, { 0, 0, 0, 2 }, { 2, 0, 1, 0 }
        });
        var target = new Vector<double>(new[] { 0.0, 0, 1, 1, 2, 2, 0 });

        var selector = new ChiSquareText(nFeaturesToSelect: 2, binarize: true);
        selector.Fit(data, target);

        AssertValues(new[] { 0.11666666666666668, 2.770833333333333, 2.770833333333333, 0.1166666666666667 },
            selector.ChiSquareScores, Tolerance);
        Assert.Equal(new[] { 1, 2 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task GrangerCausality_IncludesIntercept()
    {
        // statsmodels ssr_ftest via numpy.linalg.lstsq with a constant, maxLag 2:
        // F = ((RSS_r - RSS_u) / lag) / (RSS_u / (n - lag - 2 lag - 1)) -> [602.3253001160292, 3.01207862080417]
        var data = new Matrix<double>(new double[,]
        {
            { 0.3, 1.0 }, { 1.2, -0.5 }, { -0.7, 0.8 }, { 0.9, 0.2 }, { 1.5, -1.1 }, { -0.2, 0.6 }, { 0.4, -0.3 },
            { 1.1, 0.9 }, { -0.9, -0.4 }, { 0.6, 1.3 }, { 1.4, -0.8 }, { -0.5, 0.1 }
        });
        var target = new Vector<double>(new[] { 0.1, 0.5, 1.3, -0.4, 1.0, 1.7, 0.0, 0.6, 1.2, -0.6, 0.8, 1.5 });

        var selector = new Granger(nFeaturesToSelect: 1, maxLag: 2);
        selector.Fit(data, target);

        Assert.NotNull(selector.GrangerFScores);
        Assert.Equal(602.3253001160292, selector.GrangerFScores[0], 1e-6);
        Assert.Equal(3.01207862080417, selector.GrangerFScores[1], Tolerance);
        Assert.Equal(new[] { 0 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task L1AndGroupLasso_MatchSklearnLasso()
    {
        // X standardized (population std): sklearn.linear_model.Lasso(alpha=0.2).fit(Xs, y).coef_
        //   -> [1.1581360571760964, 0, 0, 0.49355907583947295]; singleton groups make the group lasso a lasso
        var expected = new[] { 1.1581360571760964, 0.0, 0.0, 0.49355907583947295 };

        var l1 = new L1(nFeaturesToSelect: 2, alpha: 0.2, maxIterations: 100000);
        l1.Fit(RegressionData(), RegressionTarget());
        AssertValues(expected, l1.Coefficients, SolverTolerance);
        Assert.Equal(new[] { 0, 3 }, l1.SelectedIndices);

        var groupLasso = new GroupLasso(nFeaturesToSelect: 2, alpha: 0.2, maxIterations: 100000);
        groupLasso.Fit(RegressionData(), RegressionTarget());
        AssertValues(expected, groupLasso.Coefficients, SolverTolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task ElasticNet_MatchesSklearnElasticNet()
    {
        // sklearn.linear_model.ElasticNet(alpha=0.2, l1_ratio=0.5).fit(Xs, y).coef_
        //   -> [1.0650138537186722, -0.08675147602232593, 0, 0.5419391683326337]
        var selector = new ElasticNet(nFeaturesToSelect: 2, alpha: 0.2, l1Ratio: 0.5, maxIterations: 100000);
        selector.Fit(RegressionData(), RegressionTarget());

        AssertValues(new[] { 1.0650138537186722, -0.08675147602232593, 0.0, 0.5419391683326337 },
            selector.Coefficients, SolverTolerance);
        Assert.Equal(new[] { 0, 3 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task FusedLasso_WithoutFusion_IsTheLasso()
    {
        // 1/2 ||y - X b||^2 + alpha ||b||_1 is sklearn's Lasso with alpha / n:
        // Lasso(alpha=0.3 / 8).fit(Xs, y).coef_ -> [1.2603077174952486, 0, 0, 0.5957307361586334]
        var selector = new FusedLasso(nFeaturesToSelect: 2, alpha: 0.3, fusedPenalty: 0.0, maxIterations: 200000,
            tolerance: 1e-12);
        selector.Fit(RegressionData(), RegressionTarget());

        AssertValues(new[] { 1.2603077174952486, 0.0, 0.0, 0.5957307361586334 }, selector.Coefficients, SolverTolerance);
        Assert.Equal(new[] { 0, 3 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task Jackknife_Variance_UsesEfronSteinFactor()
    {
        // theta_i = |corrcoef(X without row i, y without row i)|; mean and (m - 1) / m sum (theta_i - mean)^2
        var selector = new Jackknife(nFeaturesToSelect: 2);
        selector.Fit(RegressionData(), RegressionTarget());

        AssertValues(new[] { 0.9129729569853944, 0.6917919470795059, 0.4490219914773397, 0.7686878511469468 },
            selector.MeanImportance, Tolerance);
        AssertValues(new[] { 0.00684262602451494, 0.00363235056489894, 0.09398994690782841, 0.01436736460694113 },
            selector.ImportanceVariance, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task OnlineSelector_RunningScores_FollowWelfordCorrelations()
    {
        // r_k = Pearson r over the first k + 1 samples (0 while a variance is 0); S <- 0.7 S + 0.3 r_k from S = 0
        var selector = new OnlineSelector(nFeaturesToSelect: 2, learningRate: 0.3);
        selector.Fit(RegressionData(), RegressionTarget());

        AssertValues(new[] { 0.7068701779173281, -0.562519239091773, -0.6241073384934244, 0.7334045918824933 },
            selector.RunningScores, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task CorrelationFilter_DroppedFeature_NoLongerRemovesOthers()
    {
        // corr(a, b) = corr(a, c) = 0.7071 > 0.6, corr(b, c) = 0, var(b) > var(a) > var(c): a is dropped for b, and
        // c must then survive (the old loop kept comparing the dropped a and removed c too)
        var data = new Matrix<double>(new double[,]
        {
            { 1.2, 1, 0.8 }, { 0.0, -1, 0.8 }, { 0.0, 1, -0.8 }, { -1.2, -1, -0.8 }
        });

        var selector = new CorrelationFilter(threshold: 0.6);
        selector.Fit(data);

        Assert.NotNull(selector.CorrelationMatrix);
        Assert.Equal(Math.Sqrt(0.5), selector.CorrelationMatrix[0, 1], Tolerance);
        Assert.Equal(new[] { 1, 2 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task MaximalInformation_BinsLikeNumpyHistogram()
    {
        // scipy.stats.entropy(numpy.histogram(X[:, j], bins=4)[0]) -> [1.0789922078775833, 0.9556998911125343, 0]
        var data = new Matrix<double>(new double[,]
        {
            { 0.0, 1, 5 }, { 0.1, 1, 5 }, { 0.2, 2, 5 }, { 0.95, 2, 5 }, { 1.0, 3, 5 }, { 0.5, 4, 5 }, { 0.55, 9, 5 }
        });

        var selector = new MaximalInformation(nFeaturesToSelect: 1, nBins: 4);
        selector.Fit(data);

        AssertValues(new[] { 1.0789922078775833, 0.9556998911125343, 0.0 }, selector.Entropies, Tolerance);
        Assert.Equal(new[] { 0 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task MinimalRedundancy_ConstantFeature_HasZeroVariance()
    {
        // variances [0, 0.25, 0.75]: the constant feature used to score variance 1 and win
        var data = new Matrix<double>(new double[,] { { 5, 0, 0 }, { 5, 1, 0 }, { 5, 0, 2 }, { 5, 1, 0 } });

        var selector = new MinimalRedundancy(nFeaturesToSelect: 1);
        selector.Fit(data);

        Assert.Equal(new[] { 2 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task Wavelet_OddLength_RepeatsLastSampleLikePywt()
    {
        // Haar step with pywt's symmetric extension (odd length repeats the last sample); score = mean of |r| between
        // the feature and target details and approximations -> [0.9785336166153611, 0.9903042861630644]
        var data = new Matrix<double>(new double[,] { { 1.0, 0.5 }, { 3.0, -1.0 }, { 2.0, 2.5 }, { 5.0, 0.0 }, { 4.0, 1.5 } });
        var target = new Vector<double>(new[] { 0.5, 2.0, 1.0, 4.5, 3.0 });

        var selector = new Wavelet(nFeaturesToSelect: 1, decompositionLevel: 1);
        selector.Fit(data, target);

        AssertValues(new[] { 0.9785336166153611, 0.9903042861630644 }, selector.WaveletScores, Tolerance);
        Assert.Equal(new[] { 1 }, selector.SelectedIndices);
    }

    private static Matrix<double> SequentialData() => new Matrix<double>(new double[,]
    {
        { 1.0, 1.1, 0.3 }, { 2.0, 2.2, -0.4 }, { 3.0, 2.9, 0.8 }, { 4.0, 4.1, -0.1 }, { 5.0, 5.0, 0.5 },
        { 6.0, 6.2, -0.9 }, { 7.0, 6.8, 0.2 }, { 8.0, 8.1, -0.3 }
    });

    private static Vector<double> SequentialTarget()
    {
        var x = SequentialData();
        var noise = new[] { 0.05, -0.02, 0.01, 0.03, -0.04, 0.02, -0.01, 0.0 };
        return new Vector<double>(Enumerable.Range(0, 8).Select(i => x[i, 0] + 2.0 * x[i, 2] + noise[i]).ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task ForwardSelection_ScoresSubsetsByLeastSquaresRSquared()
    {
        // LinearRegression().fit(X[:, S], y).score(...): {0} 0.8058013414539595, then {0, 2} 0.999875369473426
        // (summing squared univariate correlations picked the redundant feature 1 instead)
        var selector = new ForwardSelection(nFeaturesToSelect: 2);
        selector.Fit(SequentialData(), SequentialTarget());

        Assert.Equal(new[] { 0, 2 }, selector.SelectedIndices);
        Assert.NotNull(selector.FeatureScores);
        Assert.Equal(0.8058013414539595, selector.FeatureScores[0], Tolerance);
        Assert.Equal(0.999875369473426, selector.FeatureScores[2], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task BackwardElimination_ScoresSubsetsByLeastSquaresRSquared()
    {
        // R^2 without feature 0, 1, 2: [0.9989725680876016, 0.999875369473426, 0.954034932635832] -> drop 1
        var selector = new BackwardElimination(nFeaturesToSelect: 2);
        selector.Fit(SequentialData(), SequentialTarget());

        Assert.Equal(new[] { 0, 2 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task ExhaustiveFeatureSelector_DefaultScore_IsCrossValidatedRSquared()
    {
        // cross_val_score(LinearRegression(), X, y, cv=KFold(3), scoring='r2').mean() -> 0.9616916388073768
        var data = new Matrix<double>(new double[,]
        {
            { 0.1, 1.0 }, { 0.9, 0.3 }, { 1.7, 2.2 }, { 2.4, 0.8 }, { 3.3, 1.9 }, { 4.1, 0.4 }, { 4.8, 2.7 }, { 6.0, 1.1 },
            { 6.6, 0.2 }, { 7.5, 2.0 }
        });
        var noise = new[] { 0.3, -0.2, 0.1, 0.4, -0.3, 0.2, -0.1, 0.0, 0.25, -0.15 };
        var target = new Vector<double>(Enumerable.Range(0, 10).Select(i => 1.5 * data[i, 0] - 0.8 * data[i, 1] + noise[i]).ToArray());

        var selector = new ExhaustiveFeatureSelector(minFeatures: 2, maxFeatures: 2, nFolds: 3);
        selector.Fit(data, target);

        Assert.Equal(0.9616916388073768, selector.BestScore, Tolerance);
        Assert.Equal(new[] { 0, 1 }, selector.SelectedIndices);
    }

    private static Matrix<double> RFEData() => new Matrix<double>(new double[,]
    {
        { 1.0, 0.5, -1.2, 2.0, 0.3 }, { 0.2, 1.5, 0.4, -1.0, 1.1 }, { -0.7, 0.3, 1.8, 0.6, -0.4 },
        { 1.4, -0.9, 0.1, 1.3, 0.8 }, { 0.6, 1.2, -0.5, -0.2, -1.3 }, { -1.1, -0.4, 0.9, 0.9, 0.5 },
        { 0.3, 0.8, 1.1, -1.5, -0.6 }, { 1.8, -1.2, -0.3, 0.4, 1.4 }, { -0.4, 0.1, -1.6, -0.8, 0.2 }
    });

    private static Vector<double> RFETarget() => new Vector<double>(new[] { 1.3, 0.2, -0.9, 2.2, 0.4, -1.0, -0.3, 2.9, 0.6 });

    [Fact(Timeout = 120000)]
    public async Task RFE_Ranking_MatchesSklearnRFE()
    {
        // sklearn.feature_selection.RFE(LinearRegression(), n_features_to_select=2, step=1).ranking_ -> [1, 1, 2, 4, 3]
        var single = new RFE(nFeaturesToSelect: 2, step: 1, importanceMethod: RFEImportanceMethod.LinearCoefficients);
        single.Fit(RFEData(), RFETarget());
        Assert.Equal(new[] { 1, 1, 2, 4, 3 }, single.Ranking);
        Assert.Equal(new[] { 0, 1 }, single.SelectedIndices);

        // n_features_to_select=1, step=2: features removed together share a rank -> [1, 2, 2, 3, 3]
        var paired = new RFE(nFeaturesToSelect: 1, step: 2, importanceMethod: RFEImportanceMethod.LinearCoefficients);
        paired.Fit(RFEData(), RFETarget());
        Assert.Equal(new[] { 1, 2, 2, 3, 3 }, paired.Ranking);
        Assert.Equal(new[] { 0 }, paired.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task RFECV_DefaultScore_UsesTheValidationFold()
    {
        // y is exactly 2 x0 - 3 x1 + 1, so a least-squares fit on any training folds predicts every validation fold
        // perfectly: the validation R^2 is 1 (the old score, the mean |r| on the training folds, was below 1)
        var x0 = new[] { 0.5, 1.2, -0.3, 2.0, 0.8, -1.1, 1.5, 0.1, -0.6, 0.9, 1.8, -0.2 };
        var x1 = new[] { 1.0, -0.5, 0.7, 0.2, -1.2, 0.4, 0.9, -0.8, 1.3, -0.1, 0.6, -0.9 };
        var x2 = new[] { 0.3, 0.1, -0.7, 0.5, 0.9, -0.2, 0.0, 0.6, -0.4, 0.8, -0.5, 0.2 };
        var values = new double[12, 3];
        for (int i = 0; i < 12; i++)
        {
            values[i, 0] = x0[i];
            values[i, 1] = x1[i];
            values[i, 2] = x2[i];
        }

        var target = new Vector<double>(Enumerable.Range(0, 12).Select(i => 2 * x0[i] - 3 * x1[i] + 1).ToArray());

        var selector = new RFECV(minFeatures: 1, nFolds: 3, randomState: 7);
        selector.Fit(new Matrix<double>(values), target);

        Assert.NotNull(selector.CVScores);
        Assert.Equal(1.0, selector.CVScores[0], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task SparseGroupLasso_AllL1_IsTheLasso()
    {
        // l1Ratio = 1 leaves only the lasso term: Lasso(alpha=0.2).fit(Xs, y).coef_ (population std)
        //   -> [1.1581360571760964, 0, 0, 0.49355907583947295]
        var selector = new SparseGroupLasso(nFeaturesToSelect: 2, alpha: 0.2, l1Ratio: 1.0, maxIterations: 100000,
            tolerance: 1e-10);
        selector.Fit(RegressionData(), RegressionTarget());

        AssertValues(new[] { 1.1581360571760964, 0.0, 0.0, 0.49355907583947295 }, selector.Coefficients, SolverTolerance);
        Assert.Equal(new[] { 0, 3 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task SparseGroupLassoSelector_NoGroupWeight_IsTheLasso()
    {
        // groupRatio = 0 leaves only the lasso term on X scaled by its sample std + 1e-10:
        // Lasso(alpha=0.2).fit(Xs, y).coef_ -> [1.2288176623268445, 0, 0, 0.518354984760389]
        var selector = new SparseGroupLassoSelector(nFeaturesToSelect: 2, alpha: 0.2, groupRatio: 0.0);
        selector.Fit(RegressionData(), RegressionTarget());

        AssertValues(new[] { 1.2288176623268445, 0.0, 0.0, 0.518354984760389 }, selector.Coefficients, SolverTolerance);
        Assert.Equal(new[] { 0, 3 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task RandomizedLasso_DominantFeature_IsAlwaysSelected()
    {
        // Meinshausen and Buhlmann (2010): the reweighted lasso keeps the dominant feature 0 (lasso coefficient 1.16
        // on the standardized data) in every subsample, so its selection probability is exactly 30 / 30 = 1. The
        // selector is seeded, so the subsamples and weights, and hence every probability, are reproducible.
        var selector = new RandomizedLasso(nFeaturesToSelect: 1, nIterations: 30, alpha: 0.01, randomState: 42);
        selector.Fit(RegressionData(), RegressionTarget());
        var repeat = new RandomizedLasso(nFeaturesToSelect: 1, nIterations: 30, alpha: 0.01, randomState: 42);
        repeat.Fit(RegressionData(), RegressionTarget());

        Assert.NotNull(selector.SelectionProbabilities);
        Assert.NotNull(repeat.SelectionProbabilities);
        Assert.Equal(1.0, selector.SelectionProbabilities[0]);
        Assert.Equal(new[] { 0 }, selector.SelectedIndices);
        Assert.Equal(selector.SelectionProbabilities.ToArray(), repeat.SelectionProbabilities.ToArray());
        Assert.Equal(selector.SelectedIndices, repeat.SelectedIndices);
    }
}
