using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Regression tests for defects found reviewing the feature-selection sweep (plan 09b). Expected values come from
/// scipy 1.18 and scikit-learn, as quoted in each test.
/// </summary>
public class FeatureSelectionSweep09bReviewTests
{
    // Welch's t-test on 400 rows has about 398 degrees of freedom, so the Student t tail needs ln Gamma(199.5):
    // Gamma itself overflows double past 171.6, the engine's lgamma returned infinity there and every p-value came
    // out NaN. scipy.stats.ttest_ind(a[y == 0], a[y == 1], equal_var=False) on the same columns gives
    // (-3.5553374162462905, 0.0004228870475972906) and (0.20709029570466897, 0.8360452580722552).
    [Fact(Timeout = 120000)]
    public async Task StatisticalTTest_Double_400Rows_PValuesMatchScipy()
    {
        await Task.CompletedTask;
        const int n = 400;
        var data = new double[n, 2];
        var target = new double[n];
        for (int i = 0; i < n; i++)
        {
            target[i] = i % 2;
            data[i, 0] = i * 0.618 % 1.0 + 0.1 * (i % 2);
            data[i, 1] = i * 0.414 % 1.0;
        }

        var selector = new AiDotNet.Preprocessing.FeatureSelection.Statistical.TTestSelector<double>(nFeaturesToSelect: 1);
        selector.Fit(new Matrix<double>(data), new Vector<double>(target));

        Assert.NotNull(selector.TStatistics);
        Assert.NotNull(selector.PValues);
        Assert.Equal(3.5553374162462905, selector.TStatistics[0], 1e-9);
        Assert.Equal(0.20709029570466897, selector.TStatistics[1], 1e-9);
        Assert.Equal(0.0004228870475972906, selector.PValues[0], 1e-9);
        Assert.Equal(0.8360452580722552, selector.PValues[1], 1e-9);
        Assert.Equal(new[] { 0 }, selector.SelectedIndices);
    }

    // 40 rows of x0 = i * 0.618 mod 1, x1 = i * 0.414 mod 1, x2 = i * 0.732 mod 1 and
    // y = 0.9 x0 - 0.4 x2 + 0.3 (i * 0.236 mod 1). Every reference test of the sweep fits fewer than 16 rows, and
    // the engine's TensorSumOfSquares is wrong for a Tensor<double> of 16 or more elements on an AVX2 CPU
    // (AiDotNet.Tensors 0.150.0), so nothing pinned the sums of squares the selectors took from it.
    private const int Rows = 40;

    private static double Feature(int row, int column) =>
        row * (column == 0 ? 0.618 : column == 1 ? 0.414 : 0.732) % 1.0;

    private static double Response(int row) =>
        0.9 * Feature(row, 0) - 0.4 * Feature(row, 2) + 0.3 * (row * 0.236 % 1.0);

    private static Matrix<double> Features()
    {
        var data = new double[Rows, 3];
        for (int i = 0; i < Rows; i++)
            for (int j = 0; j < 3; j++)
                data[i, j] = Feature(i, j);
        return new Matrix<double>(data);
    }

    private static Vector<double> Responses() =>
        new Vector<double>(Enumerable.Range(0, Rows).Select(Response).ToArray());

    [Fact(Timeout = 120000)]
    public async Task SumOfSquares_Double_SixteenOrMoreElements()
    {
        await Task.CompletedTask;
        // numpy: ((i * 0.618 % 1 - 0.5) ** 2).sum() over 40 rows -> 3.4509599999999967
        var values = new Tensor<double>(
            Enumerable.Range(0, Rows).Select(i => Feature(i, 0) - 0.5).ToArray(), new[] { Rows });

        Assert.Equal(3.4509599999999967, AiDotNet.Preprocessing.PreprocessingTensorOps.SumOfSquares(values), 1e-12);
        Assert.Equal(3.4509599999999967, AiDotNet.Preprocessing.PreprocessingTensorOps.SumOfSquares(
            new Tensor<double>(values.ToArray(), new[] { Rows, 1 })), 1e-12);
    }

    [Fact(Timeout = 120000)]
    public async Task PearsonCorrelationSelector_Double_40Rows_MatchesNumpy()
    {
        await Task.CompletedTask;
        // abs(numpy.corrcoef(x_j, y)[0, 1]); the target's standard deviation came from the wrong sum of squares,
        // which scaled every correlation by the same factor (0.9878 instead of 0.9229 on 400 rows).
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Similarity.PearsonCorrelationSelector<double>(
            nFeaturesToSelect: 1);
        selector.Fit(Features(), Responses());

        Assert.NotNull(selector.Correlations);
        Assert.Equal(0.9177535253367737, selector.Correlations[0], 1e-9);
        Assert.Equal(0.11948051482615907, selector.Correlations[1], 1e-9);
        Assert.Equal(0.33895104734738335, selector.Correlations[2], 1e-9);
        Assert.Equal(new[] { 0 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task GrangerCausality_Double_40Rows_MatchesLeastSquares()
    {
        await Task.CompletedTask;
        // numpy.linalg.lstsq with a constant, maxLag 2 (statsmodels ssr_ftest):
        // F = ((RSS_r - RSS_u) / 2) / (RSS_u / (38 - 5)). Each RSS is y'y minus the explained part, and y'y was the
        // wrong sum of squares.
        var selector = new AiDotNet.Preprocessing.FeatureSelection.TimeSeries.GrangerCausalitySelector<double>(
            nFeaturesToSelect: 1, maxLag: 2);
        selector.Fit(Features(), Responses());

        Assert.NotNull(selector.GrangerFScores);
        Assert.Equal(4.853082875804534, selector.GrangerFScores[0], 1e-6);
        Assert.Equal(1.8740066795377577, selector.GrangerFScores[1], 1e-6);
        Assert.Equal(2.211264641716058, selector.GrangerFScores[2], 1e-6);
        Assert.Equal(new[] { 0 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task WrapperRSquared_Double_40Rows_MatchesNumpy()
    {
        await Task.CompletedTask;
        // 1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum() for f = 0.8 x0 - 0.3 x2 + 0.1 -> 0.8809549812936678.
        // This is the score of forward selection, backward elimination, the exhaustive selector and RFECV.
        var y = Tensor<double>.FromVector(Responses());
        var predictions = new Tensor<double>(
            Enumerable.Range(0, Rows).Select(i => 0.8 * Feature(i, 0) - 0.3 * Feature(i, 2) + 0.1).ToArray(), new[] { Rows });

        double rSquared = AiDotNet.Preprocessing.FeatureSelection.Wrapper.WrapperScoringOps.RSquared(y, predictions);

        Assert.Equal(0.8809549812936678, rSquared, 1e-10);
    }

    // The loops this replaced scored a pair of columns 0 unless both sums of squares were above 1e-10, which a column
    // holding a NaN fails (NaN > 1e-10 is false). Multiplying |r| by the 0/1 guard instead put the NaN back
    // (0 * NaN): columns (1,2,3,4), (2,1,4,3) and (NaN,1,2,3) have r = 0.6 for the first pair and nothing countable
    // for the other two, so the mean over the three pairs is 0.2, not NaN.
    [Fact(Timeout = 120000)]
    public async Task MeanAbsPairCorrelation_ColumnWithNaN_CountsItsPairsAsZero()
    {
        await Task.CompletedTask;
        var data = new Matrix<double>(new double[,]
        {
            { 1, 2, double.NaN }, { 2, 1, 1 }, { 3, 4, 2 }, { 4, 3, 3 }
        });
        var features = new[] { 0, 1, 2 };

        Assert.Equal(0.2,
            AiDotNet.Preprocessing.FeatureSelection.Wrapper.WrapperScoringOps.MeanAbsPairCorrelation(data, features), 1e-12);
        Assert.Equal(0.2, AiDotNet.Preprocessing.FeatureSelection.Wrapper.WrapperScoringOps
            .MeanAbsPairCorrelationWithDenominatorGuard(data, features), 1e-12);
    }

    // With the NaN in every subset score no candidate ever compared above the running best, nothing was added and
    // the forward loop of the floating search never ended. Master selects all four features of this data.
    [Fact(Timeout = 120000)]
    public async Task FloatingSearch_DataWithNaNCell_TerminatesAndSelects()
    {
        const int n = 30;
        var data = new double[n, 4];
        var target = new double[n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < 4; j++)
                data[i, j] = i * (0.618 + 0.1 * j) % 1.0 + (j < 2 ? 0.8 * (i % 2) : 0.0);
            target[i] = i % 2;
        }

        data[3, 2] = double.NaN;
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Wrapper.FloatingSearch<double>(nFeaturesToSelect: 4);

        var fit = Task.Run(() => selector.Fit(new Matrix<double>(data), new Vector<double>(target)));
        var finished = await Task.WhenAny(fit, Task.Delay(TimeSpan.FromSeconds(30)));

        Assert.Same(fit, finished);
        await fit;
        Assert.Equal(new[] { 0, 1, 2, 3 }, selector.SelectedIndices);
    }

    // The same tail in float: Gamma overflows float past 35.04, so the p-values were NaN from about 70 rows.
    [Fact(Timeout = 120000)]
    public async Task StatisticalTTest_Float_400Rows_PValuesAreFinite()
    {
        await Task.CompletedTask;
        const int n = 400;
        var data = new float[n, 2];
        var target = new float[n];
        for (int i = 0; i < n; i++)
        {
            target[i] = i % 2;
            data[i, 0] = (float)(i * 0.618 % 1.0 + 0.1 * (i % 2));
            data[i, 1] = (float)(i * 0.414 % 1.0);
        }

        var selector = new AiDotNet.Preprocessing.FeatureSelection.Statistical.TTestSelector<float>(nFeaturesToSelect: 1);
        selector.Fit(new Matrix<float>(data), new Vector<float>(target));

        Assert.NotNull(selector.PValues);
        Assert.Equal(0.0004228870475972906, selector.PValues[0], 1e-5);
        Assert.Equal(0.8360452580722552, selector.PValues[1], 1e-4);
    }
}
