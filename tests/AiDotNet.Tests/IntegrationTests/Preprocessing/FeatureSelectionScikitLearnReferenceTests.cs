using System;
using AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory;
using AiDotNet.Preprocessing.FeatureSelection.Filter.Relief;
using AiDotNet.Preprocessing.FeatureSelection.Filter.Spectral;
using AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;
using AiDotNet.Preprocessing.FeatureSelection.Graph;
using AiDotNet.Preprocessing.FeatureSelection.HighDimensional;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;
using FilterKBest = AiDotNet.Preprocessing.FeatureSelection.Filter.SelectKBest<double>;
using FilterPercentile = AiDotNet.Preprocessing.FeatureSelection.Filter.SelectPercentile<double>;
using KendallTauSelector = AiDotNet.Preprocessing.FeatureSelection.Filter.KendallTau<double>;
using SpearmanSelector = AiDotNet.Preprocessing.FeatureSelection.Filter.SpearmanCorrelation<double>;
using StatisticalKruskalWallis = AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical.KruskalWallis<double>;
using StatisticalMannWhitneyU = AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical.MannWhitneyU<double>;
using StatisticalWilcoxon = AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical.WilcoxonSignedRank<double>;
using UnivariateKBest = AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate.SelectKBest<double>;
using UnivariatePercentile = AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate.SelectPercentile<double>;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Feature selectors against reference values from scikit-learn 1.9.1, scipy 1.18.1 and networkx (generated
/// offline), and against the formulas of the papers the selectors implement.
/// </summary>
public class FeatureSelectionScikitLearnReferenceTests
{
    private const double Tolerance = 1e-9;

    // 30 samples, 4 continuous features; classes 0, 1, 2 in blocks of 10
    private static readonly double[,] X = new double[,]
        {
            { 1.37, -1.323, -0.133, -1.322 },
            { -1.757, 1.336, -0.35, -0.048 },
            { 0.968, 1.517, 0.295, 0.714 },
            { -0.319, 0.167, 0.453, 0.646 },
            { 0.218, -0.055, -1.311, -0.17 },
            { -0.269, -0.139, 1.578, -0.271 },
            { 1.69, -0.997, 1.52, -0.872 },
            { -0.094, 0.327, 0.921, 0.224 },
            { -0.015, 0.004, -0.024, 0.476 },
            { 0.042, 0.133, 0.088, -0.556 },
            { 1.795, 0.569, 1.0150000000000001, 0.201 },
            { 1.202, 1.478, -0.42800000000000005, 0.069 },
            { -0.04499999999999993, -2.061, -0.727, -0.153 },
            { 2.236, 0.287, 2.049, -0.646 },
            { 0.245, -0.218, -0.848, -0.121 },
            { 0.858, -0.039, 0.13199999999999998, 1.756 },
            { 0.904, 0.709, 3.08, 1.292 },
            { 0.684, -0.309, 1.4809999999999999, 0.701 },
            { 0.41100000000000003, 0.168, 0.282, -0.369 },
            { 3.3, 0.136, 0.0020000000000000018, -0.494 },
            { 1.487, 0.602, 0.9099999999999999, 0.914 },
            { 1.671, -1.07, 0.469, -0.365 },
            { 1.7870000000000001, -0.15, -1.055, 0.485 },
            { 3.415, -0.48, 1.335, 0.404 },
            { 2.549, 1.594, 1.536, -0.671 },
            { 3.032, -0.18, 0.129, -0.737 },
            { 2.648, 0.419, 1.0659999999999998, 0.398 },
            { 0.6440000000000001, -0.646, 0.16699999999999993, -0.212 },
            { 1.805, -1.626, -0.18800000000000006, 1.161 },
            { -0.19900000000000007, 1.422, -0.5010000000000001, 1.381 }
        };

    private static readonly double[] ClassTarget = new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 };

    private static readonly double[] BinaryTarget = new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

    private static readonly double[] RegressionTarget = new[] { -1.169, 1.984, 2.168, 0.782, -0.127, -0.086, -1.12, 0.345, 0.071, -0.567, 0.297, 2.06, -1.835, 1.287, 0.703, -0.429, -0.098, -0.584, -0.125, -1.543, 1.645, -2.172, -2.345, -0.202, 2.465, 1.241, 0.863, -0.693, -4.168, 0.794 };

    private static Matrix<double> Data(double[,] values) => new Matrix<double>(values);

    private static Vector<double> Target(double[] values) => new Vector<double>(values);

    private static void AssertClose(double[] expected, Vector<double>? actual, double tolerance = Tolerance)
    {
        Assert.NotNull(actual);
        Assert.Equal(expected.Length, actual.Length);
        for (int j = 0; j < expected.Length; j++)
        {
            if (double.IsPositiveInfinity(expected[j]))
                Assert.True(double.IsPositiveInfinity(actual[j]), $"entry {j}: expected +inf, got {actual[j]}");
            else
                Assert.True(Math.Abs(expected[j] - actual[j]) <= tolerance * Math.Max(1.0, Math.Abs(expected[j])),
                    $"entry {j}: expected {expected[j]:R}, got {actual[j]:R}");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task FilterSelectKBest_DefaultsToFClassif()
    {
        await Task.Yield();
        var selector = new FilterKBest(k: 2);
        Assert.Equal(SelectKBestScoreFunc.FClassif, selector.ScoreFunction);
        selector.Fit(Data(X), Target(ClassTarget));

        AssertClose(new[] { 6.823678055847624, 0.03634404305590805, 0.23193085293687768, 0.8415596511905195 }, selector.Scores);
        AssertClose(new[] { 0.0039946132130277605, 0.9643555651603974, 0.7945645199694538, 0.44203300296781406 }, selector.PValues);
        Assert.Equal(new[] { 0, 3 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task FilterSelectKBest_FRegressionOnContinuousTarget_MatchesScikitLearn()
    {
        await Task.Yield();
        var selector = new FilterKBest(k: 2, scoreFunction: SelectKBestScoreFunc.FRegression);
        selector.Fit(Data(X), Target(RegressionTarget));

        AssertClose(new[] { 0.40166655311326793, 54.492944387317294, 1.1808894840334392, 0.011203045063642357 }, selector.Scores);
        AssertClose(new[] { 0.5313734673721029, 4.872323942572125e-08, 0.28644180256737334, 0.9164604402108953 }, selector.PValues);
        Assert.Equal(new[] { 1, 2 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task FilterSelectKBest_DefaultDoesNotRerouteContinuousTargets()
    {
        await Task.Yield();

        // a continuous target is scored by f_classif (every distinct value a class) unless the caller asks for
        // f_regression, as scikit-learn does
        var target = Target(new[] { -2.0, 2.0, 2.0, 0.0, -0.0, -0.0, -2.0, 0.0, 0.0, -0.0, 0.0, 2.0, -2.0, 2.0, 0.0, -0.0, -0.0, -0.0, -0.0, -2.0, 2.0, -2.0, -2.0, -0.0, 2.0, 2.0, 0.0, -0.0, -4.0, 0.0 });
        var selector = new FilterKBest(k: 2);
        selector.Fit(Data(X), target);
        AssertClose(new[] { 1.255126143871024, 12.251360080711006, 0.609549977546493, 3.205920647436034 }, selector.Scores);
        AssertClose(new[] { 0.31024590406428804, 3.4784092569684426e-05, 0.6148335767062643, 0.03958640518743869 }, selector.PValues);
    }

    [Fact(Timeout = 120000)]
    public async Task FilterSelectKBest_CustomScorer_IsUsedAsGiven()
    {
        await Task.Yield();
        var selector = new FilterKBest(2, (data, target) =>
        {
            var scores = new Vector<double>(data.Columns);
            for (int j = 0; j < data.Columns; j++)
                scores[j] = -j;
            return (scores, null);
        });
        selector.Fit(Data(X), Target(ClassTarget));

        AssertClose(new[] { 0.0, -1.0, -2.0, -3.0 }, selector.Scores);
        Assert.Null(selector.PValues);
        Assert.Equal(new[] { 0, 1 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task FilterSelectPercentile_DefaultsMatchScikitLearn()
    {
        await Task.Yield();
        var selector = new FilterPercentile();
        Assert.Equal(10.0, selector.Percentile);
        Assert.Equal(SelectKBestScoreFunc.FClassif, selector.ScoreFunction);
    }

    [Fact(Timeout = 120000)]
    public async Task UnivariateChi2_MatchesScikitLearn()
    {
        await Task.Yield();
        var selector = new UnivariateKBest(k: 2, scoreFunc: SelectKBestScoreFunc.Chi2);
        selector.Fit(Data(new double[,]
        {
            { 1.37, 1.323, 0.133, 1.322 },
            { 1.757, 1.336, 0.35, 0.048 },
            { 0.968, 1.517, 0.295, 0.714 },
            { 0.319, 0.167, 0.453, 0.646 },
            { 0.218, 0.055, 1.311, 0.17 },
            { 0.269, 0.139, 1.578, 0.271 },
            { 1.69, 0.997, 1.52, 0.872 },
            { 0.094, 0.327, 0.921, 0.224 },
            { 0.015, 0.004, 0.024, 0.476 },
            { 0.042, 0.133, 0.088, 0.556 },
            { 1.795, 0.569, 1.0150000000000001, 0.201 },
            { 1.202, 1.478, 0.42800000000000005, 0.069 },
            { 0.04499999999999993, 2.061, 0.727, 0.153 },
            { 2.236, 0.287, 2.049, 0.646 },
            { 0.245, 0.218, 0.848, 0.121 },
            { 0.858, 0.039, 0.13199999999999998, 1.756 },
            { 0.904, 0.709, 3.08, 1.292 },
            { 0.684, 0.309, 1.4809999999999999, 0.701 },
            { 0.41100000000000003, 0.168, 0.282, 0.369 },
            { 3.3, 0.136, 0.0020000000000000018, 0.494 },
            { 1.487, 0.602, 0.9099999999999999, 0.914 },
            { 1.671, 1.07, 0.469, 0.365 },
            { 1.7870000000000001, 0.15, 1.055, 0.485 },
            { 3.415, 0.48, 1.335, 0.404 },
            { 2.549, 1.594, 1.536, 0.671 },
            { 3.032, 0.18, 0.129, 0.737 },
            { 2.648, 0.419, 1.0659999999999998, 0.398 },
            { 0.6440000000000001, 0.646, 0.16699999999999993, 0.212 },
            { 1.805, 1.626, 0.18800000000000006, 1.161 },
            { 0.19900000000000007, 1.422, 0.5010000000000001, 1.381 }
        }), Target(ClassTarget));

        AssertClose(new[] { 6.309703337847527, 0.4814881206289373, 0.7915703900635567, 0.17682012451623752 }, selector.Scores);
        AssertClose(new[] { 0.04264472606165181, 0.7860427801876029, 0.6731512766470917, 0.9153854347328758 }, selector.PValues);
    }

    [Fact(Timeout = 120000)]
    public async Task UnivariateMutualInfoRegression_MatchesKraskovEstimator()
    {
        await Task.Yield();
        var selector = new UnivariateKBest(k: 2, scoreFunc: SelectKBestScoreFunc.MutualInfoRegression);
        selector.Fit(Data(X), Target(RegressionTarget));

        // sklearn's _compute_mi_cc on unit-variance columns (mutual_info_regression without its jitter)
        AssertClose(new[] { 0.06875235218975195, 0.48536738863424067, 0.0, 0.0 }, selector.Scores, 1e-9);
    }

    [Fact(Timeout = 120000)]
    public async Task PValueSelectors_KeepTheScikitLearnSupport()
    {
        await Task.Yield();
        var fdr = new SelectFdr<double>(alpha: 0.01);
        fdr.Fit(Data(X), Target(ClassTarget));
        Assert.Equal(Array.Empty<int>(), fdr.SelectedIndices);

        var fpr = new SelectFpr<double>(alpha: 0.05);
        fpr.Fit(Data(X), Target(ClassTarget));
        Assert.Equal(new[] { 0 }, fpr.SelectedIndices);

        var fwe = new SelectFwe<double>(alpha: 0.05);
        fwe.Fit(Data(X), Target(ClassTarget));
        Assert.Equal(new[] { 0 }, fwe.SelectedIndices);

        var percentile = new UnivariatePercentile(percentile: 50);
        percentile.Fit(Data(X), Target(ClassTarget));
        Assert.Equal(new[] { 0, 3 }, percentile.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task RankTests_MatchScipy()
    {
        await Task.Yield();
        var kruskal = new StatisticalKruskalWallis(nFeaturesToSelect: 2);
        kruskal.Fit(Data(X), Target(ClassTarget));
        AssertClose(new[] { 9.726451612903219, 0.30193548387096314, 0.0800000000000125, 0.978064516129038 }, kruskal.HStatistics);
        AssertClose(new[] { 0.007725522611970483, 0.859875436129121, 0.9607894391523172, 0.6132195453992115 }, kruskal.PValues);

        var mannWhitney = new StatisticalMannWhitneyU(nFeaturesToSelect: 2);
        mannWhitney.Fit(Data(X), Target(BinaryTarget));
        AssertClose(new[] { 35.0, 97.0, 94.0, 78.0 }, mannWhitney.UStatistics);
        AssertClose(new[] { 0.00454510196108324, 0.9124210043205884, 0.8088051604661488, 0.3442129113242063 }, mannWhitney.PValues);

        var wilcoxon = new StatisticalWilcoxon(nFeaturesToSelect: 2);
        wilcoxon.Fit(Data(X), Target(new[] { 0.991, 0.123, -0.381, 0.073, -0.293, 0.182, -0.527, 0.079, -0.5, -0.439, -0.3, -0.116, -0.214, 0.168, 0.088, 0.461, -0.128, -0.473, -0.057, 0.188, -0.913, 0.394, 0.281, 0.445, -0.234, -0.584, 0.362, -0.548, -0.265, -0.29 }));
        AssertClose(new[] { 38.0, 191.0, 121.0, 179.0 }, wilcoxon.WStatistics);
        AssertClose(new[] { 6.319756764397932e-05, 0.3933343721992797, 0.02182672159602272, 0.27115519770883223 }, wilcoxon.PValues);
    }

    [Fact(Timeout = 120000)]
    public async Task CorrelationSelectors_MatchScipy()
    {
        await Task.Yield();
        var target = Target(new[] { -2.0, 2.0, 2.0, 0.0, -0.0, -0.0, -2.0, 0.0, 0.0, -0.0, 0.0, 2.0, -2.0, 2.0, 0.0, -0.0, -0.0, -0.0, -0.0, -2.0, 2.0, -2.0, -2.0, -0.0, 2.0, 2.0, 0.0, -0.0, -4.0, 0.0 });
        var kendall = new KendallTauSelector(nFeaturesToSelect: 2);
        kendall.Fit(Data(X), target);
        AssertClose(new[] { -0.04305707553701853, 0.5941876424108555, 0.1808397172554778, 0.03731613213208272 }, kendall.Correlations);

        var spearman = new SpearmanSelector(nFeaturesToSelect: 2);
        spearman.Fit(Data(X), target);
        AssertClose(new[] { -0.05181627354371766, 0.7041636041483518, 0.2304846507156875, 0.04668352946627393 }, spearman.Correlations);
    }

    [Fact(Timeout = 120000)]
    public async Task FisherExactTest_MatchesScipy()
    {
        await Task.Yield();
        var selector = new FisherExactTest<double>(nFeaturesToSelect: 2);
        selector.Fit(Data(X), Target(BinaryTarget));
        AssertClose(new[] { 0.057671164417791104, 0.7089455272363818, 1.0, 0.7089455272363818 }, selector.PValues, 1e-8);
        AssertClose(new[] { 6.0, 1.5, 0.8181818181818182, 1.5 }, selector.OddsRatios);
    }

    [Fact(Timeout = 120000)]
    public async Task BenjaminiHochberg_UsesExactTTestPValues()
    {
        await Task.Yield();
        var selector = new BenjaminiHochberg<double>(nFeaturesToSelect: 2, alpha: 0.05);
        selector.Fit(Data(X), Target(RegressionTarget));
        AssertClose(new[] { 0.5313734673721029, 4.872323942572125e-08, 0.28644180256737334, 0.9164604402108953 }, selector.PValues);
        AssertClose(new[] { 0.7084979564961372, 1.94892957702885e-07, 0.5728836051347467, 0.9164604402108953 }, selector.AdjustedPValues);
    }

    [Fact(Timeout = 120000)]
    public async Task ReliefF_MatchesKononenkoWithManhattanDistance()
    {
        await Task.Yield();
        var selector = new ReliefF<double>(nFeaturesToSelect: 2, nNeighbors: 3, nSamples: 30, randomState: 1);
        selector.Fit(Data(X), Target(ClassTarget));
        AssertClose(new[] { 0.0636665377674658, -0.028355373157014755, -0.039602469698119896, -0.011349361056963438 }, selector.FeatureWeights);
    }

    [Fact(Timeout = 120000)]
    public async Task RReliefF_MatchesRobnikSikonjaKononenko()
    {
        await Task.Yield();
        var selector = new RReliefF<double>(nFeaturesToSelect: 2, nNeighbors: 3, randomState: 1);
        selector.Fit(Data(X), Target(RegressionTarget));
        AssertClose(new[] { 0.012860987651322237, 0.045106741046975235, -0.015295393301337418, 0.00417017543241352 }, selector.FeatureWeights);
    }

    [Fact(Timeout = 120000)]
    public async Task FisherScore_PerfectSeparation_IsInfinite()
    {
        await Task.Yield();
        var data = new double[,] { { 0.0, 0.3 }, { 0.0, 0.1 }, { 1.0, 0.2 }, { 1.0, 0.4 } };
        var selector = new FisherScore<double>(nFeaturesToSelect: 1);
        selector.Fit(Data(data), Target(new[] { 0.0, 0.0, 1.0, 1.0 }));
        Assert.NotNull(selector.FisherScores);
        Assert.True(double.IsPositiveInfinity(selector.FisherScores[0]));
        Assert.Equal(new[] { 0 }, selector.SelectedIndices);
    }

    // integer codes 0..3 in every column (each column holds both 0 and 3, so 4 bins reproduce the codes)
    private static readonly double[,] Codes = new double[,]
        {
            { 0.0, 0.0, 0.0, 0.0, 0.0 },
            { 3.0, 3.0, 3.0, 3.0, 3.0 },
            { 0.0, 0.0, 3.0, 1.0, 1.0 },
            { 3.0, 0.0, 3.0, 0.0, 1.0 },
            { 0.0, 1.0, 1.0, 1.0, 2.0 },
            { 1.0, 1.0, 1.0, 1.0, 2.0 },
            { 0.0, 2.0, 2.0, 1.0, 3.0 },
            { 0.0, 2.0, 2.0, 1.0, 3.0 },
            { 0.0, 0.0, 3.0, 0.0, 3.0 },
            { 0.0, 0.0, 0.0, 1.0, 0.0 },
            { 0.0, 2.0, 3.0, 1.0, 3.0 },
            { 3.0, 3.0, 2.0, 1.0, 2.0 },
            { 1.0, 1.0, 1.0, 1.0, 3.0 },
            { 0.0, 0.0, 0.0, 3.0, 2.0 },
            { 0.0, 0.0, 2.0, 1.0, 1.0 },
            { 0.0, 0.0, 3.0, 1.0, 0.0 },
            { 0.0, 1.0, 3.0, 1.0, 0.0 },
            { 0.0, 3.0, 3.0, 2.0, 2.0 },
            { 0.0, 2.0, 1.0, 0.0, 0.0 },
            { 2.0, 2.0, 2.0, 2.0, 3.0 },
            { 1.0, 1.0, 2.0, 2.0, 0.0 },
            { 1.0, 2.0, 1.0, 2.0, 1.0 },
            { 1.0, 0.0, 1.0, 3.0, 1.0 },
            { 1.0, 2.0, 3.0, 1.0, 2.0 },
            { 1.0, 1.0, 1.0, 2.0, 2.0 },
            { 1.0, 1.0, 0.0, 2.0, 1.0 },
            { 1.0, 0.0, 0.0, 1.0, 3.0 },
            { 0.0, 0.0, 3.0, 3.0, 0.0 },
            { 0.0, 0.0, 0.0, 1.0, 1.0 },
            { 1.0, 0.0, 2.0, 2.0, 0.0 },
            { 3.0, 3.0, 1.0, 3.0, 2.0 },
            { 3.0, 3.0, 2.0, 2.0, 0.0 },
            { 2.0, 2.0, 3.0, 3.0, 0.0 },
            { 1.0, 0.0, 0.0, 1.0, 3.0 },
            { 3.0, 3.0, 2.0, 2.0, 0.0 },
            { 1.0, 1.0, 1.0, 2.0, 0.0 },
            { 1.0, 1.0, 3.0, 2.0, 2.0 },
            { 1.0, 3.0, 2.0, 2.0, 1.0 },
            { 1.0, 3.0, 0.0, 2.0, 2.0 },
            { 1.0, 1.0, 0.0, 3.0, 0.0 },
            { 2.0, 2.0, 0.0, 2.0, 2.0 },
            { 2.0, 2.0, 2.0, 0.0, 3.0 },
            { 2.0, 2.0, 0.0, 2.0, 1.0 },
            { 2.0, 2.0, 1.0, 3.0, 2.0 },
            { 2.0, 2.0, 3.0, 0.0, 3.0 },
            { 2.0, 2.0, 3.0, 1.0, 0.0 },
            { 2.0, 2.0, 2.0, 3.0, 2.0 },
            { 2.0, 3.0, 1.0, 0.0, 3.0 },
            { 3.0, 3.0, 3.0, 0.0, 1.0 },
            { 2.0, 2.0, 2.0, 0.0, 2.0 },
            { 0.0, 0.0, 0.0, 0.0, 3.0 },
            { 2.0, 0.0, 0.0, 0.0, 1.0 },
            { 2.0, 2.0, 3.0, 2.0, 3.0 },
            { 2.0, 2.0, 1.0, 0.0, 3.0 },
            { 2.0, 2.0, 0.0, 3.0, 0.0 },
            { 2.0, 2.0, 2.0, 0.0, 3.0 },
            { 3.0, 2.0, 2.0, 0.0, 3.0 },
            { 1.0, 1.0, 3.0, 1.0, 1.0 },
            { 2.0, 2.0, 0.0, 0.0, 0.0 },
            { 2.0, 0.0, 2.0, 0.0, 3.0 }
        };

    private static readonly double[] CodeTarget = new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 };

    [Fact(Timeout = 120000)]
    public async Task Cife_UsesConditionalRedundancy()
    {
        await Task.Yield();

        // Lin and Tang (2006): J = I(Xj;Y) - sum_s [I(Xj;Xs) - I(Xj;Xs|Y)]
        var selector = new CIFE<double>(nFeaturesToSelect: 3, nBins: 4);
        selector.Fit(Data(Codes), Target(CodeTarget));
        Assert.Equal(new[] { 0, 2, 3 }, selector.SelectedIndices);
        AssertClose(new[] { 0.4988473541172944, 0.0, 0.43366944910449146, 0.253941463139006, 0.0 }, selector.CIFEScores);
    }

    [Fact(Timeout = 120000)]
    public async Task Cmim_MinimumStartsAtRelevance()
    {
        await Task.Yield();

        // Fleuret (2004): J = min(I(Xj;Y), min_s I(Xj;Y|Xs))
        var selector = new CMIM<double>(nFeaturesToSelect: 3, nBins: 4);
        selector.Fit(Data(Codes), Target(CodeTarget));
        Assert.Equal(new[] { 0, 1, 3 }, selector.SelectedIndices);
        AssertClose(new[] { 0.4988473541172944, 0.13905479455446423, 0.028484129174978445, 0.30073440830890297, 0.06753249189451704 }, selector.MIScores);
        AssertClose(new[] { 0.4988473541172944, 0.11641172617167594, 0.0, 0.2539414631390069, 0.0 }, selector.CMIMScores);
    }

    [Fact(Timeout = 120000)]
    public async Task Jmi_SumsJointRelevance()
    {
        await Task.Yield();

        // Yang and Moody (1999): J = sum_s I(Xj, Xs; Y)
        var selector = new JMI<double>(nFeaturesToSelect: 3, nBins: 4);
        selector.Fit(Data(Codes), Target(CodeTarget));
        Assert.Equal(new[] { 0, 2, 3 }, selector.SelectedIndices);
        AssertClose(new[] { 0.4988473541172944, 0.0, 1.261735340705668, 0.7527888172563006, 0.0 }, selector.JMIScores);
    }

    private static readonly double[,] GraphData = new double[,]
        {
            { 1.59, 0.613, 1.504, -0.309, -0.844, 0.529, -0.045 },
            { -0.237, -0.122, 0.216, 1.877, 2.06, 1.372, 0.908 },
            { -0.387, -0.63, -1.731, 0.129, -0.336, -0.291, -0.838 },
            { 1.04, 1.99, 0.824, -1.234, -1.091, 0.084, -1.287 },
            { -1.389, -1.557, -2.01, -1.295, -1.443, -1.817, 0.031 },
            { -1.133, -1.373, -0.081, 0.598, 0.141, -0.403, 0.521 },
            { 0.621, 1.247, 0.302, -0.691, -0.074, 0.222, -0.078 },
            { -1.678, -2.77, -2.468, -0.276, -0.38, -1.216, 2.704 },
            { 1.617, 2.102, 0.862, 0.512, -0.702, 1.295, -0.866 },
            { 1.114, 2.617, 3.168, 0.105, 0.374, 0.923, -0.155 },
            { 0.304, -1.132, -1.916, -0.668, -1.561, -0.305, -0.12 },
            { 1.616, 1.568, 1.275, -0.805, 0.366, 0.123, 0.365 },
            { 1.625, 1.423, 1.339, -0.3, -0.84, 0.853, 0.21 },
            { 1.711, 2.93, 2.98, 0.512, 0.738, 1.233, 0.757 },
            { -1.103, -1.418, -2.157, -0.839, -0.987, -0.838, -1.107 },
            { 2.313, 1.722, 2.443, -1.046, -0.463, -0.176, 0.58 },
            { 0.457, 0.191, 0.792, -0.599, 0.008, 0.167, 1.072 },
            { 0.052, -0.221, 1.292, -0.205, 0.192, -0.554, -0.033 },
            { 0.235, -0.082, -1.638, 0.961, 1.941, 0.582, -0.532 },
            { -0.278, 0.702, 1.727, 0.696, 0.281, 0.076, -0.547 },
            { -0.144, -0.311, -0.074, 0.284, 0.137, -0.116, -1.788 },
            { -1.948, -2.57, -3.384, -0.725, -1.512, -1.321, -0.182 },
            { 0.751, 1.743, 3.173, -1.001, -1.034, -0.03, 0.895 },
            { 1.888, 2.329, 2.98, -1.303, -1.216, 0.105, 1.464 },
            { -1.59, -1.683, -1.862, -0.45, -0.869, -1.054, -0.488 },
            { 0.928, 1.241, 1.625, -2.262, -2.271, -1.169, -0.422 },
            { -0.019, 0.659, 1.536, -0.167, -0.241, -0.508, -0.509 },
            { 1.926, 2.684, 2.129, 0.037, -0.16, 1.116, -0.592 },
            { -0.945, -0.836, -0.516, -1.568, -1.791, -1.542, 0.325 },
            { -0.506, -1.657, -1.242, -0.018, -0.082, -0.812, -0.737 }
        };

    private static readonly double[] GraphTarget = new[] { 0.043, 1.456, -0.142, -1.025, -2.391, -1.009, 0.249, -1.774, 1.778, 1.061, -0.53, 1.124, 1.396, 2.068, -2.117, 0.609, -0.24, 0.538, 1.408, 0.867, 0.504, -2.525, 0.476, 0.549, -1.381, -1.012, 0.534, 1.774, -2.432, -0.795 };

    [Fact(Timeout = 120000)]
    public async Task GraphBasedFS_BetweennessMatchesNetworkx()
    {
        await Task.Yield();
        var selector = new GraphBasedFS<double>(nFeaturesToSelect: 2, edgeThreshold: 0.6, centralityMeasure: GraphCentralityMeasure.Betweenness);
        selector.Fit(Data(GraphData), Target(GraphTarget));
        AssertClose(new[] { 0.37028479137790954, 0.3732028346658071, 0.0, 0.0, 0.0, 0.0, 0.0 }, selector.CentralityScores, 1e-8);
    }

    [Fact(Timeout = 120000)]
    public async Task GraphBasedFS_PageRankSpreadsDanglingRank()
    {
        await Task.Yield();
        var selector = new GraphBasedFS<double>(nFeaturesToSelect: 2, edgeThreshold: 0.6, centralityMeasure: GraphCentralityMeasure.PageRank);
        selector.Fit(Data(GraphData), Target(GraphTarget));
        AssertClose(new[] { 0.14227080593945357, 0.14880857666576067, 0.09401285429021963, 0.08894505455065124, 0.10018188176346396, 0.10699183990745922, 0.0003343126018439846 }, selector.CentralityScores, 1e-6);
    }
}
