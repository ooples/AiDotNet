using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Preprocessing.FeatureSelection;
using AiDotNet.Preprocessing.FeatureSelection.Regression;
using AiDotNet.Preprocessing.FeatureSelection.Robust;
using AiDotNet.Preprocessing.FeatureSelection.Sequential;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Reference values for the feature selectors fixed in the element-wise sweep (plan 09). Expected values come from
/// scikit-learn 1.9.1 and scipy, as quoted in each test.
/// </summary>
public class FeatureSelectionSweep09ReferenceTests
{
    private const double Tolerance = 1e-9;

    [Fact(Timeout = 120000)]
    public async Task LARSSelector_EntryOrder_MatchesSklearnLarsPath()
    {
        // X standardized (population std), y centred:
        //   sklearn.linear_model.lars_path(Xs, yc, method='lar') -> active = [1, 2, 3, 0]
        // Orthogonal matching pursuit (the old behaviour) enters the features as [1, 3, 2, 0].
        var data = new Matrix<double>(new double[,]
        {
            { 0.9, -0.2, -0.6, 0.2 },
            { 0.7, 0.7, 2.0, 0.2 },
            { -0.6, -0.1, -0.1, 0.1 },
            { 0.0, 0.2, -1.7, 0.8 },
            { -0.6, -1.2, 0.6, 1.3 },
            { 0.5, 0.2, -0.9, 2.9 },
            { 0.9, -1.1, -0.8, 0.1 },
            { -1.6, 0.2, -0.5, 1.2 }
        });
        var target = new Vector<double>(new[] { 1.0, -2.7, 0.0, -1.6, 1.1, 0.2, 0.5, -1.1 });

        var selector = new LARSSelector<double>(nFeaturesToSelect: 3);
        selector.Fit(data, target);

        Assert.NotNull(selector.FeatureRanks);
        var ranks = selector.FeatureRanks;
        Assert.Equal(0.0, ranks[0], Tolerance);
        Assert.Equal(4.0, ranks[1], Tolerance);
        Assert.Equal(3.0, ranks[2], Tolerance);
        Assert.Equal(2.0, ranks[3], Tolerance);
        Assert.Equal(new[] { 1, 2, 3 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task RobustFeatureSelector_TiedValues_UseAverageRanks()
    {
        // scipy.stats.spearmanr(X[:, j], y) -> [0.991031208965115, 0.6671243849949912]
        var data = new Matrix<double>(new double[,]
        {
            { 1, 3 }, { 2, 1 }, { 2, 4 }, { 3, 2 }, { 4, 4 }, { 5, 5 }, { 6, 4 }
        });
        var target = new Vector<double>(new[] { 1.0, 2, 3, 4, 5, 6, 7 });

        var selector = new RobustFeatureSelector<double>(nFeaturesToSelect: 1);
        selector.Fit(data, target);

        Assert.NotNull(selector.RobustScores);
        Assert.Equal(0.991031208965115, selector.RobustScores[0], Tolerance);
        Assert.Equal(0.6671243849949912, selector.RobustScores[1], Tolerance);
        Assert.Equal(new[] { 0 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task RobustFeatureSelector_EvenRowCount_ScreensOutliersWithTrueMedian()
    {
        // y = [0, 1, 2, 3, 4, 10]: median 2.5, MAD 1.5, so |10 - 2.5| / 1.5 = 5 > 4.8 and the last row is an
        // outlier (the old upper median 3 and MAD 2 gave 3.5 and kept it).
        // On the five remaining rows: spearmanr([1,2,3,4,5], y) = 1, spearmanr([2,1,4,3,5], y) = 0.8
        var data = new Matrix<double>(new double[,]
        {
            { 1, 2 }, { 2, 1 }, { 3, 4 }, { 4, 3 }, { 5, 5 }, { 0, 6 }
        });
        var target = new Vector<double>(new[] { 0.0, 1, 2, 3, 4, 10 });

        var selector = new RobustFeatureSelector<double>(nFeaturesToSelect: 1, outlierThreshold: 4.8);
        selector.Fit(data, target);

        Assert.NotNull(selector.RobustScores);
        Assert.Equal(1.0, selector.RobustScores[0], Tolerance);
        Assert.Equal(0.8, selector.RobustScores[1], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task SelectKBest_FRegression_PerfectFit_RanksFirst()
    {
        // sklearn f_regression: F = [1.35e16 (perfect fit), 1.1095890410958904], p = [1.4e-24, 0.3695172283938317]
        var data = new Matrix<double>(new double[,]
        {
            { 1, 0.5 }, { 2, 2.5 }, { 3, 1.0 }, { 4, 4.0 }, { 5, 2.0 }
        });
        var target = new Vector<double>(new[] { 3.0, 5, 7, 9, 11 });

        var selector = new SelectKBest<double>(k: 1, defaultScoreFunc: UnivariateScoreFunction.FRegression);
        selector.Fit(data, target);

        Assert.NotNull(selector.Scores);
        Assert.NotNull(selector.PValues);
        Assert.True(selector.Scores[0] > 1e15, $"Expected a perfect-fit F, got {selector.Scores[0]}");
        Assert.Equal(1.1095890410958904, selector.Scores[1], Tolerance);
        Assert.Equal(0.0, selector.PValues[0], Tolerance);
        Assert.Equal(0.3695172283938317, selector.PValues[1], Tolerance);
        Assert.Equal(new[] { 0 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task SelectKBest_Chi2_NegativeInput_Throws()
    {
        // sklearn chi2 raises ValueError("Input X must be non-negative.")
        var data = new Matrix<double>(new double[,] { { 1, -1 }, { 2, 3 }, { 0, 1 }, { 4, 2 } });
        var target = new Vector<double>(new[] { 0.0, 0, 1, 1 });

        var selector = new SelectKBest<double>(k: 1, defaultScoreFunc: UnivariateScoreFunction.Chi2);

        Assert.Throws<ArgumentException>(() => selector.Fit(data, target));
    }

    [Fact(Timeout = 120000)]
    public async Task SelectKBest_MutualInfoClassif_MatchesSklearnKnnEstimator()
    {
        // X = values / 8 (exact in binary, so no distance rounding):
        //   sklearn.feature_selection._mutual_info._compute_mi_cd(X[:, j], y, n_neighbors=3)
        //   -> [0.12027417027417053, 0.015512265512265477]
        var raw = new double[,]
        {
            { 1, -1 }, { 5, 1 }, { -4, 3 }, { 10, 8 }, { -6, -10 }, { -5, 0 },
            { -6, 2 }, { 3, -2 }, { 8, 1 }, { 16, 12 }, { 12, 15 }, { 7, 7 }
        };
        var scaled = new double[12, 2];
        for (int i = 0; i < 12; i++)
        {
            scaled[i, 0] = raw[i, 0] / 8;
            scaled[i, 1] = raw[i, 1] / 8;
        }

        var target = new Vector<double>(new[] { 0.0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1 });

        var selector = new SelectKBest<double>(k: 1, defaultScoreFunc: UnivariateScoreFunction.MutualInfoClassif);
        selector.Fit(new Matrix<double>(scaled), target);

        Assert.NotNull(selector.Scores);
        Assert.Equal(0.12027417027417053, selector.Scores[0], Tolerance);
        Assert.Equal(0.015512265512265477, selector.Scores[1], Tolerance);
        Assert.Equal(new[] { 0 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task SelectKBest_TiesAtTheCut_KeepHigherIndices()
    {
        // sklearn SelectKBest(lambda X, y: [5, 5, 5, 1], k=2).get_support(indices=True) -> [1, 2]
        Func<Matrix<double>, Vector<double>, Vector<double>> scores =
            (m, t) => new Vector<double>(new[] { 5.0, 5.0, 5.0, 1.0 });
        var selector = new SelectKBest<double>(k: 2, scoreFunc: scores);

        selector.Fit(new Matrix<double>(4, 4), new Vector<double>(new[] { 0.0, 0, 1, 1 }));

        Assert.Equal(new[] { 1, 2 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task SelectPercentile_TiesAtThreshold_FollowSklearnRule()
    {
        // sklearn SelectPercentile(lambda X, y: [1, 3, 3, 3, 0], percentile=50).get_support(indices=True) -> [1, 2]
        Func<Matrix<double>, Vector<double>, Vector<double>> scores =
            (m, t) => new Vector<double>(new[] { 1.0, 3.0, 3.0, 3.0, 0.0 });
        var selector = new SelectPercentile<double>(percentile: 50.0, scoreFunc: scores);

        selector.Fit(new Matrix<double>(4, 5), new Vector<double>(new[] { 0.0, 0, 1, 1 }));

        Assert.Equal(new[] { 1, 2 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task SelectPercentile_FClassif_ReportsSklearnScoresAndPValues()
    {
        // sklearn f_classif -> F = [121.5, 0.375, 0], p = [0.00038506771136654104, 0.5733922538253555, 1]
        // SelectPercentile(f_classif, percentile=50).get_support(indices=True) -> [0]
        var data = new Matrix<double>(new double[,]
        {
            { 1, 10, 3 }, { 2, 12, 1 }, { 10, 11, 2 }, { 11, 13, 4 }, { 12, 9, 3 }, { 3, 14, 5 }
        });
        var target = new Vector<double>(new[] { 0.0, 0, 1, 1, 1, 0 });

        var selector = new SelectPercentile<double>(percentile: 50.0, defaultScoreFunc: UnivariateScoreFunction.FClassif);
        selector.Fit(data, target);

        Assert.NotNull(selector.Scores);
        Assert.NotNull(selector.PValues);
        Assert.Equal(121.5, selector.Scores[0], Tolerance);
        Assert.Equal(0.375, selector.Scores[1], Tolerance);
        Assert.Equal(0.0, selector.Scores[2], Tolerance);
        Assert.Equal(0.00038506771136654104, selector.PValues[0], Tolerance);
        Assert.Equal(0.5733922538253555, selector.PValues[1], Tolerance);
        Assert.Equal(1.0, selector.PValues[2], Tolerance);
        Assert.Equal(new[] { 0 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task FloatingSelectors_Backtracking_DoesNotModifyEnumeratedSet()
    {
        // y = x0 + 2 x1 exactly; x2 is unrelated. Both floating searches backtrack over their working set,
        // which used to throw InvalidOperationException (collection modified during enumeration).
        var data = new Matrix<double>(new double[,]
        {
            { 1, 2, 5 }, { 2, 0, 1 }, { 3, 3, 4 }, { 4, 1, 2 }, { 5, 4, 6 }, { 6, 2, 3 }, { 7, 5, 1 }, { 8, 1, 5 }
        });
        var target = new Vector<double>(Enumerable.Range(0, 8).Select(i => data[i, 0] + 2 * data[i, 1]).ToArray());

        var forward = new FloatingForwardSelector<double>(nFeaturesToSelect: 2);
        forward.Fit(data, target);
        Assert.Equal(new[] { 0, 1 }, forward.SelectedIndices);

        var backward = new FloatingBackwardSelector<double>(nFeaturesToSelect: 2);
        backward.Fit(data, target);
        Assert.Equal(new[] { 0, 1 }, backward.SelectedIndices);
    }
}
