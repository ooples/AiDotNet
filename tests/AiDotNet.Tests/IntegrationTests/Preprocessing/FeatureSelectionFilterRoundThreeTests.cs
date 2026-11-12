using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Preprocessing.FeatureSelection;
using AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory;
using AiDotNet.Preprocessing.FeatureSelection.Filter.Spectral;
using AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Reference tests for the third review round of the information-theory, univariate and spectral filter
/// selectors: greedy criteria past 2048 steps in Half, products and quotients that left Half's range, the digamma
/// table of the KSG estimator, and the trace ratio and SPEC scores of their papers.
/// </summary>
public class FeatureSelectionFilterRoundThreeTests
{
    private const int WideP = 2100;

    // Every feature is [0, 0, 0, 1] against the target [0, 0, 1, 1] (two bins), so every greedy step sees the same
    // per-step term. Natural-log entropies of that table:
    private static readonly double HX = -(0.75 * Math.Log(0.75) + 0.25 * Math.Log(0.25));
    private static readonly double HY = Math.Log(2);
    private static readonly double HXY = -(0.5 * Math.Log(0.5) + 2 * 0.25 * Math.Log(0.25));
    private static readonly double Relevance = HX + HY - HXY;

    private static (Matrix<Half> X, Vector<Half> Y) IdenticalFeatures()
    {
        var x = new Matrix<Half>(4, WideP);
        for (int j = 0; j < WideP; j++)
            x[3, j] = (Half)1;
        var y = new Vector<Half>(new[] { (Half)0, (Half)0, (Half)1, (Half)1 });
        return (x, y);
    }

    // Feature j is picked at step j (all ties go to the lower index); from step 1 on its score is the mean of the
    // identical per-step terms. A step counter in Half stopped at 2048 and a growing sum stopped taking terms, so
    // the late scores drifted far from the term.
    private static void AssertLateScores(Vector<Half>? scores, double expected)
    {
        Assert.NotNull(scores);
        if (scores is null)
            return;
        for (int j = 1; j < WideP; j++)
            Assert.InRange((double)scores[j], expected - 0.01, expected + 0.01);
    }

    [Fact(Timeout = 120000)]
    public async Task MRMR_Half_MeanRedundancyStaysExactPast2048Selections()
    {
        await Task.CompletedTask;
        var (x, y) = IdenticalFeatures();
        var selector = new MRMR<Half>(nFeaturesToSelect: WideP, nBins: 2);
        selector.Fit(x, y);
        // relevance I(X; Y) minus the mean redundancy I(X; X) = H(X)
        AssertLateScores(selector.MRMRScores, Relevance - HX);
    }

    [Fact(Timeout = 120000)]
    public async Task JointMutualInformation_Half_MeanJointRelevanceStaysExactPast2048Selections()
    {
        await Task.CompletedTask;
        var (x, y) = IdenticalFeatures();
        var selector = new JointMutualInformation<Half>(nFeaturesToSelect: WideP, nBins: 2);
        selector.Fit(x, y);
        // I(Xj, Xs; Y) = I(X; Y) for identical features
        AssertLateScores(selector.JMIScores, Relevance);
    }

    [Fact(Timeout = 120000)]
    public async Task DISR_Half_MeanSymmetricRelevanceStaysExactPast2048Selections()
    {
        await Task.CompletedTask;
        var (x, y) = IdenticalFeatures();
        var selector = new DISR<Half>(nFeaturesToSelect: WideP, nBins: 2);
        selector.Fit(x, y);
        // I(Xj, Xs; Y) / H(Xj, Xs, Y) (Meyer et al. 2008)
        AssertLateScores(selector.DISRScores, Relevance / HXY);
    }

    [Fact(Timeout = 120000)]
    public async Task InteractionInformation_Half_MeanInteractionStaysExactPast2048Selections()
    {
        await Task.CompletedTask;
        var (x, y) = IdenticalFeatures();
        var selector = new InteractionInformation<Half>(nFeaturesToSelect: WideP, nBins: 2);
        selector.Fit(x, y);
        // I(X; Y) - mean of I(Xj; Y) + I(Xs; Y) - I(Xj, Xs; Y) = I(X; Y) - I(X; Y) = 0
        AssertLateScores(selector.InteractionScores, 0.0);
    }

    // scipy.stats.false_discovery_control: 3000 p-values of 1e-5 all adjust to 1e-5 (p_(r) n / r at the top rank is
    // p itself, and the running minimum carries it down). p / r underflowed to 0 in Half before the product with n,
    // which made every adjusted p-value 0.
    [Fact(Timeout = 60000)]
    public async Task BenjaminiHochberg_Half_SmallPValuesAtHighRanksDoNotUnderflow()
    {
        await Task.CompletedTask;
        const int n = 3000;
        var p = (Half)1e-5;
        var adjusted = StatisticalTestHelper.BenjaminiHochbergCorrection(
            new Vector<Half>(Enumerable.Repeat(p, n).ToArray()));

        for (int i = 0; i < n; i++)
            Assert.Equal((double)p, (double)adjusted[i], 7);
    }

    // sklearn.feature_selection.f_regression: x alternates -3, 3 and y = x + e with e = 1, 1, -1, -1, ... over 100
    // rows, so Sxx = 900, Sxy = 900, Syy = 1000, r^2 = 0.9 and F = 0.9 / 0.1 * 98 = 882. Sxy^2 and Sxx Syy overflowed
    // Half and the score came out NaN.
    [Fact(Timeout = 60000)]
    public async Task FRegression_Half_DoesNotOverflowTheCorrelation()
    {
        await Task.CompletedTask;
        const int n = 100;
        var x = new Matrix<Half>(n, 1);
        var y = new Vector<Half>(n);
        for (int i = 0; i < n; i++)
        {
            double xi = i % 2 == 0 ? -3 : 3;
            double ei = i % 4 < 2 ? 1 : -1;
            x[i, 0] = (Half)xi;
            y[i] = (Half)(xi + ei);
        }

        var (scores, pValues) = StatisticalTestHelper.ComputeScores(x, y, SelectKBestScoreFunc.FRegression);

        Assert.InRange((double)scores[0], 880.0, 884.0);
        Assert.NotNull(pValues);
    }

    // sklearn.feature_selection.chi2: a feature that is 0 on the 60 class-0 rows and 10 on the 60 class-1 rows has
    // observed [0, 600] against expected [300, 300], chi2 = 300 + 300 = 600. (O - E)^2 = 90000 overflowed Half.
    [Fact(Timeout = 60000)]
    public async Task Chi2_Half_DoesNotOverflowTheSquaredDeviation()
    {
        await Task.CompletedTask;
        const int n = 120;
        var x = new Matrix<Half>(n, 1);
        var y = new Vector<Half>(n);
        for (int i = 60; i < n; i++)
        {
            x[i, 0] = (Half)10;
            y[i] = (Half)1;
        }

        var (scores, _) = StatisticalTestHelper.ComputeScores(x, y, SelectKBestScoreFunc.Chi2);

        Assert.Equal(600.0, (double)scores[0], 0);
    }

    // scipy.stats.chi2_contingency(correction=False) of [[600, 0], [0, 600]]: every expected count is 300 and every
    // cell adds 300^2 / 300, chi2 = 1200. (O - E)^2 = 90000 overflowed Half.
    [Fact(Timeout = 60000)]
    public async Task ContingencyChiSquare_Half_DoesNotOverflowTheSquaredDeviation()
    {
        await Task.CompletedTask;
        var observed = new Tensor<Half>(new[] { (Half)600, (Half)0, (Half)0, (Half)600 }, new[] { 1, 2, 2 });

        var chi2 = FeatureSelectionTensorOps.ContingencyChiSquare(observed);

        Assert.Equal(1200.0, (double)chi2[0], 0);
    }

    // A feature equal to the class over 300 + 300 rows splits the classes exactly: the Gini reduction is the parent
    // impurity 1 - 0.5^2 - 0.5^2 = 0.5. count^2 = 90000 overflowed Half and the reduction came out infinite.
    [Fact(Timeout = 60000)]
    public async Task BestGiniReduction_Half_DoesNotOverflowTheSquaredCounts()
    {
        await Task.CompletedTask;
        const int n = 600;
        var values = new Half[n];
        var classIndex = new int[n];
        for (int i = 300; i < n; i++)
        {
            values[i] = (Half)1;
            classIndex[i] = 1;
        }

        var reduction = FeatureSelectionTensorOps.BestGiniReduction(
            new Tensor<Half>(values, new[] { n, 1 }), FilterStatisticTensorOps.ClassOneHot<Half>(classIndex, 2), 10);

        Assert.Equal(0.5, (double)reduction[0], 3);
    }

    // psi(3001) = H(3000) - gamma = 8.0065. Summed in Half the harmonic series stalls near 6.8 by m = 512.
    [Fact(Timeout = 60000)]
    public async Task DigammaTable_Half_MatchesDoublePast2048()
    {
        await Task.CompletedTask;
        double harmonic = 0;
        for (int m = 1; m <= 3000; m++)
            harmonic += 1.0 / m;
        double expected = harmonic - 0.57721566490153286060651209;

        var table = StatisticalTestHelper.DigammaTable<Half>(4000);

        Assert.InRange((double)table[3000], expected - 0.01, expected + 0.01);
    }

    // Two classes of two rows; the column with between-class scatter Sb = 4 a^2 and within-class scatter Sw = 4 s^2
    // holds -a - s, -a + s, a - s, a + s.
    private static (Matrix<double> X, Vector<double> Y) Scatter(double[] sb, double[] sw)
    {
        var x = new Matrix<double>(4, sb.Length);
        for (int j = 0; j < sb.Length; j++)
        {
            double a = Math.Sqrt(sb[j] / 4);
            double s = Math.Sqrt(sw[j] / 4);
            x[0, j] = -a - s;
            x[1, j] = -a + s;
            x[2, j] = a - s;
            x[3, j] = a + s;
        }

        return (x, new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 }));
    }

    // Nie et al. (2008): the k-subset with the largest sum(Sb) / sum(Sw). For Sb = [3, 2, 1, 6], Sw = [7, 6, 5, 1],
    // k = 3 the optimum (brute force over all subsets) is {0, 1, 3} with 11 / 14; the greedy forward search kept
    // {0, 2, 3} with 10 / 13.
    [Fact(Timeout = 60000)]
    public async Task TraceRatio_FindsTheGloballyBestSubset()
    {
        await Task.CompletedTask;
        var (x, y) = Scatter(new[] { 3.0, 2, 1, 6 }, new[] { 7.0, 6, 5, 1 });
        var selector = new TraceRatio<double>(nFeaturesToSelect: 3);
        selector.Fit(x, y);

        Assert.Equal(new[] { 0, 1, 3 }, selector.SelectedIndices);
        Assert.Equal(11.0 / 14.0, selector.TraceRatioValue, 10);
    }

    // Sb = [4, 6, 1, 5, 3], Sw = [6, 9, 2, 4, 5], k = 3: the three best single ratios {0, 1, 3} (15 / 19) are not the
    // optimum; one update of the iteration reaches {0, 2, 3} with 10 / 12, the brute-force optimum.
    [Fact(Timeout = 60000)]
    public async Task TraceRatio_IteratesFromTheBestSingleRatiosToTheOptimum()
    {
        await Task.CompletedTask;
        var (x, y) = Scatter(new[] { 4.0, 6, 1, 5, 3 }, new[] { 6.0, 9, 2, 4, 5 });
        var selector = new TraceRatio<double>(nFeaturesToSelect: 3);
        selector.Fit(x, y);

        Assert.Equal(new[] { 0, 2, 3 }, selector.SelectedIndices);
        Assert.Equal(10.0 / 12.0, selector.TraceRatioValue, 10);
    }

    // Zhao and Liu (2007) phi3 with k = 3 (skfeature's SPEC style = 3), computed with numpy.linalg.eigh on the same
    // 3-nearest-neighbour Gaussian graph (sigma = 1) of 8 rows: sum_{j=2..3} (2 - lambda_j) (f^ . xi_j)^2 with
    // f^ = D^1/2 f / ||D^1/2 f||.
    [Fact(Timeout = 60000)]
    public async Task SPEC_MatchesThePaperScoreFromNumpy()
    {
        await Task.CompletedTask;
        const int n = 8, p = 3;
        var x = new Matrix<double>(n, p);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                x[i, j] = Math.Sin(1.3 * i + 0.7 * j * j + 0.4) * (j + 1);

        var selector = new SPEC<double>(nFeaturesToSelect: 1, nNeighbors: 3, sigma: 1.0, nEigenvectors: 3);
        selector.Fit(x);

        var expected = new[] { 0.757224794727208, 1.523419748002370, 0.426953528790895 };
        Assert.NotNull(selector.SpecScores);
        if (selector.SpecScores is not null)
        {
            for (int j = 0; j < p; j++)
                Assert.Equal(expected[j], selector.SpecScores[j], 9);
        }

        Assert.Equal(new[] { 1 }, selector.SelectedIndices);
    }
}
