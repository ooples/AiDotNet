using AiDotNet.SpeechRecognition;
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Helpers;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.Helpers;

/// <summary>
/// Deep math-correctness integration tests for StatisticsHelper covering:
///   Median, MAD, Variance, StdDev, MSE, VarianceReduction, SplitScore,
///   T-Test, Mann-Whitney U Test, Chi-Square Test, F-Test
/// with hand-calculated expected values and mathematical properties.
/// </summary>
public class StatisticsHelperDeepMathIntegrationTests
{
    private const double Tol = 1e-6;

    // ═══════════════════════════════════════════════════════════════
    // MEDIAN
    // ═══════════════════════════════════════════════════════════════

    [Fact(Timeout = 120000)]
    public async Task Median_OddCount_ReturnsMiddle()
    {
        // [1, 3, 5, 7, 9] => sorted, middle (index 2) = 5
        var values = new double[] { 9, 1, 5, 3, 7 };
        double result = StatisticsHelper<double>.CalculateMedian(values);
        Assert.Equal(5.0, result, Tol);
    }

    [Fact(Timeout = 120000)]
    public async Task Median_EvenCount_ReturnsAverageOfMiddleTwo()
    {
        // [1, 3, 5, 7] => sorted, (3+5)/2 = 4
        var values = new double[] { 7, 1, 5, 3 };
        double result = StatisticsHelper<double>.CalculateMedian(values);
        Assert.Equal(4.0, result, Tol);
    }

    [Fact(Timeout = 120000)]
    public async Task Median_SingleElement_ReturnsThatElement()
    {
        var values = new double[] { 42.0 };
        double result = StatisticsHelper<double>.CalculateMedian(values);
        Assert.Equal(42.0, result, Tol);
    }

    [Fact(Timeout = 120000)]
    public async Task Median_AllSame_ReturnsThatValue()
    {
        var values = new double[] { 5, 5, 5, 5, 5 };
        double result = StatisticsHelper<double>.CalculateMedian(values);
        Assert.Equal(5.0, result, Tol);
    }

    [Fact(Timeout = 120000)]
    public async Task Median_TwoElements_ReturnsAverage()
    {
        var values = new double[] { 10, 20 };
        double result = StatisticsHelper<double>.CalculateMedian(values);
        Assert.Equal(15.0, result, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // MEAN ABSOLUTE DEVIATION (MAD)
    // ═══════════════════════════════════════════════════════════════

    [Fact(Timeout = 120000)]
    public async Task MAD_HandCalculated()
    {
        // values = [2, 4, 6, 8], median = 5
        // |2-5|=3, |4-5|=1, |6-5|=1, |8-5|=3
        // MAD = (3+1+1+3)/4 = 8/4 = 2.0
        var values = new Vector<double>(new double[] { 2, 4, 6, 8 });
        double median = 5.0;
        double result = StatisticsHelper<double>.CalculateMeanAbsoluteDeviation(values, median);
        Assert.Equal(2.0, result, Tol);
    }

    [Fact(Timeout = 120000)]
    public async Task MAD_AllSame_ReturnsZero()
    {
        var values = new Vector<double>(new double[] { 5, 5, 5, 5 });
        double result = StatisticsHelper<double>.CalculateMeanAbsoluteDeviation(values, 5.0);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact(Timeout = 120000)]
    public async Task MAD_IsNonNegative()
    {
        var values = new Vector<double>(new double[] { -10, 0, 10, 20, 30 });
        double median = StatisticsHelper<double>.CalculateMedian(new double[] { -10, 0, 10, 20, 30 });
        double result = StatisticsHelper<double>.CalculateMeanAbsoluteDeviation(values, median);
        Assert.True(result >= 0);
    }

    // ═══════════════════════════════════════════════════════════════
    // VARIANCE (SAMPLE VARIANCE, n-1 denominator)
    // ═══════════════════════════════════════════════════════════════

    [Fact(Timeout = 120000)]
    public async Task Variance_HandCalculated_WithMean()
    {
        // values = [2, 4, 6, 8, 10], mean = 6
        // sum((x-6)^2) = 16+4+0+4+16 = 40
        // Var = 40 / (5-1) = 10.0
        var values = new Vector<double>(new double[] { 2, 4, 6, 8, 10 });
        double result = StatisticsHelper<double>.CalculateVariance(values, 6.0);
        Assert.Equal(10.0, result, Tol);
    }

    [Fact(Timeout = 120000)]
    public async Task Variance_HandCalculated_NoMean()
    {
        // Same calculation but using the overload that computes mean internally
        var values = new double[] { 2, 4, 6, 8, 10 };
        double result = StatisticsHelper<double>.CalculateVariance(values);
        Assert.Equal(10.0, result, Tol);
    }

    [Fact(Timeout = 120000)]
    public async Task Variance_AllSame_ReturnsZero()
    {
        var values = new double[] { 5, 5, 5, 5, 5 };
        double result = StatisticsHelper<double>.CalculateVariance(values);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact(Timeout = 120000)]
    public async Task Variance_SingleElement_ReturnsZero()
    {
        var values = new Vector<double>(new double[] { 42.0 });
        double result = StatisticsHelper<double>.CalculateVariance(values, 42.0);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact(Timeout = 120000)]
    public async Task Variance_IsNonNegative()
    {
        var values = new double[] { -5, 0, 5, 10, 100 };
        double result = StatisticsHelper<double>.CalculateVariance(values);
        Assert.True(result >= 0);
    }

    [Fact(Timeout = 120000)]
    public async Task Variance_ScaleProperty()
    {
        // Var(c*X) = c^2 * Var(X)
        var values = new double[] { 1, 2, 3, 4, 5 };
        double varOriginal = StatisticsHelper<double>.CalculateVariance(values);
        var scaledValues = values.Select(x => x * 3.0).ToArray();
        double varScaled = StatisticsHelper<double>.CalculateVariance(scaledValues);
        Assert.Equal(9.0 * varOriginal, varScaled, Tol);
    }

    [Fact(Timeout = 120000)]
    public async Task Variance_ShiftInvariant()
    {
        // Var(X + c) = Var(X) for any constant c
        var values = new double[] { 1, 2, 3, 4, 5 };
        double varOriginal = StatisticsHelper<double>.CalculateVariance(values);
        var shiftedValues = values.Select(x => x + 100.0).ToArray();
        double varShifted = StatisticsHelper<double>.CalculateVariance(shiftedValues);
        Assert.Equal(varOriginal, varShifted, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // STANDARD DEVIATION
    // ═══════════════════════════════════════════════════════════════

    [Fact(Timeout = 120000)]
    public async Task StdDev_EqualsSqrtVariance()
    {
        var values = new double[] { 2, 4, 6, 8, 10 };
        double variance = StatisticsHelper<double>.CalculateVariance(values);
        double stdDev = StatisticsHelper<double>.CalculateStandardDeviation(values);
        Assert.Equal(Math.Sqrt(variance), stdDev, Tol);
    }

    [Fact(Timeout = 120000)]
    public async Task StdDev_AllSame_ReturnsZero()
    {
        var values = new double[] { 7, 7, 7, 7 };
        double result = StatisticsHelper<double>.CalculateStandardDeviation(values);
        Assert.Equal(0.0, result, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // MEAN SQUARED ERROR
    // ═══════════════════════════════════════════════════════════════

    [Fact(Timeout = 120000)]
    public async Task MSE_HandCalculated()
    {
        // actual=[1,2,3,4,5], pred=[1.5,2.5,3.5,4.5,5.5]
        // errors^2: 0.25, 0.25, 0.25, 0.25, 0.25
        // MSE = 1.25/5 = 0.25
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var pred = new double[] { 1.5, 2.5, 3.5, 4.5, 5.5 };
        double result = StatisticsHelper<double>.CalculateMeanSquaredError(actual, pred);
        Assert.Equal(0.25, result, Tol);
    }

    [Fact(Timeout = 120000)]
    public async Task MSE_PerfectPredictions_ShouldBeZero()
    {
        var actual = new double[] { 1, 2, 3 };
        var pred = new double[] { 1, 2, 3 };
        double result = StatisticsHelper<double>.CalculateMeanSquaredError(actual, pred);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact(Timeout = 120000)]
    public async Task MSE_IsNonNegative()
    {
        var actual = new double[] { -5, 0, 5 };
        var pred = new double[] { 10, -10, 0 };
        double result = StatisticsHelper<double>.CalculateMeanSquaredError(actual, pred);
        Assert.True(result >= 0);
    }

    // ═══════════════════════════════════════════════════════════════
    // VARIANCE REDUCTION (for decision trees)
    // ═══════════════════════════════════════════════════════════════

    [Fact(Timeout = 120000)]
    public async Task VarianceReduction_PerfectSplit_MaximizesReduction()
    {
        // y = [1, 1, 1, 10, 10, 10]
        // Perfect split: left=[0,1,2], right=[3,4,5]
        // Left variance: Var([1,1,1]) = 0, Right variance: Var([10,10,10]) = 0
        // Reduction = TotalPopVar - (0.5*0 + 0.5*0) = TotalPopVar
        // Variance reduction uses population variance (n denominator) for correct decomposition:
        // mean = 5.5, pop_var = ((1-5.5)^2*3 + (10-5.5)^2*3)/6 = 121.5/6 = 20.25
        var y = new Vector<double>(new double[] { 1, 1, 1, 10, 10, 10 });
        var leftIndices = new List<int> { 0, 1, 2 };
        var rightIndices = new List<int> { 3, 4, 5 };
        double reduction = StatisticsHelper<double>.CalculateVarianceReduction(y, leftIndices, rightIndices);
        Assert.Equal(20.25, reduction, Tol);
    }

    [Fact(Timeout = 120000)]
    public async Task VarianceReduction_NoSplitBenefit_ReturnsZero()
    {
        // When the split doesn't reduce variance at all, both subgroups must have
        // same mean as total AND same internal variance structure.
        // y = [1, 2, 3, 4], left=[0,3] => [1,4] (mean=2.5), right=[1,2] => [2,3] (mean=2.5)
        // Both subgroups have same mean as total (2.5), so between-group variance = 0
        // pop_var(total) = (2.25+0.25+0.25+2.25)/4 = 1.25
        // pop_var(left [1,4]) = (2.25+2.25)/2 = 2.25
        // pop_var(right [2,3]) = (0.25+0.25)/2 = 0.25
        // VR = 1.25 - (0.5*2.25 + 0.5*0.25) = 1.25 - 1.25 = 0
        var y = new Vector<double>(new double[] { 1, 2, 3, 4 });
        var leftIndices = new List<int> { 0, 3 };
        var rightIndices = new List<int> { 1, 2 };
        double reduction = StatisticsHelper<double>.CalculateVarianceReduction(y, leftIndices, rightIndices);
        Assert.Equal(0.0, reduction, 0.001);
    }

    [Fact(Timeout = 120000)]
    public async Task VarianceReduction_IsNonNegative()
    {
        // Variance reduction should always be >= 0 for any valid split
        var y = new Vector<double>(new double[] { 1, 5, 2, 8, 3, 7 });
        var leftIndices = new List<int> { 0, 3, 5 };
        var rightIndices = new List<int> { 1, 2, 4 };
        double reduction = StatisticsHelper<double>.CalculateVarianceReduction(y, leftIndices, rightIndices);
        Assert.True(reduction >= -Tol, $"Variance reduction ({reduction}) should be >= 0");
    }

    // ═══════════════════════════════════════════════════════════════
    // T-TEST
    // ═══════════════════════════════════════════════════════════════

    [Fact(Timeout = 120000)]
    public async Task TTest_IdenticalGroups_PValueOne()
    {
        // When both groups are identical, t-statistic = 0, p-value should be ≈ 1
        var group1 = new Vector<double>(new double[] { 5, 5, 5, 5, 5 });
        var group2 = new Vector<double>(new double[] { 5, 5, 5, 5, 5 });
        // Both have zero variance, so pooled SE = 0, which may cause division by zero
        // Let's use groups with same distribution instead
        var g1 = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var g2 = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var result = StatisticsHelper<double>.TTest(g1, g2);
        Assert.Equal(0.0, result.TStatistic, Tol);
        // p-value should be close to 1 for identical groups
        Assert.True(result.PValue > 0.9, $"P-value ({result.PValue}) should be > 0.9 for identical groups");
    }

    [Fact(Timeout = 120000)]
    public async Task TTest_VeryDifferentGroups_SmallPValue()
    {
        // Groups with clearly different means
        var group1 = new Vector<double>(new double[] { 1, 2, 3, 2, 1 });
        var group2 = new Vector<double>(new double[] { 100, 101, 102, 101, 100 });
        var result = StatisticsHelper<double>.TTest(group1, group2);
        // t-statistic should be large and negative (group1 mean << group2 mean)
        Assert.True(result.TStatistic < -10,
            $"T-statistic ({result.TStatistic}) should be strongly negative for very different groups");
        // p-value should be very small
        Assert.True(result.PValue < 0.01,
            $"P-value ({result.PValue}) should be < 0.01 for very different groups");
    }

    [Fact(Timeout = 120000)]
    public async Task TTest_HandCalculated_DegreesOfFreedom()
    {
        // df = n1 + n2 - 2
        var g1 = new Vector<double>(new double[] { 1, 2, 3 }); // n1 = 3
        var g2 = new Vector<double>(new double[] { 4, 5, 6, 7 }); // n2 = 4
        var result = StatisticsHelper<double>.TTest(g1, g2);
        Assert.Equal(5, result.DegreesOfFreedom); // 3 + 4 - 2 = 5
    }

    [Fact(Timeout = 120000)]
    public async Task TTest_TStatistic_HandCalculated()
    {
        // g1 = [2, 4, 6], mean = 4
        // g2 = [8, 10, 12], mean = 10
        // Var(g1) = ((2-4)^2 + (4-4)^2 + (6-4)^2) / 2 = 8/2 = 4
        // Var(g2) = ((8-10)^2 + (10-10)^2 + (12-10)^2) / 2 = 8/2 = 4
        // pooled SE = sqrt(4/3 + 4/3) = sqrt(8/3) = 1.6330
        // t = (4 - 10) / 1.6330 = -6 / 1.6330 = -3.674
        var g1 = new Vector<double>(new double[] { 2, 4, 6 });
        var g2 = new Vector<double>(new double[] { 8, 10, 12 });
        var result = StatisticsHelper<double>.TTest(g1, g2);
        double expectedT = -6.0 / Math.Sqrt(8.0 / 3.0);
        Assert.Equal(expectedT, result.TStatistic, 0.01);
    }

    // ═══════════════════════════════════════════════════════════════
    // MANN-WHITNEY U TEST
    // ═══════════════════════════════════════════════════════════════

    [Fact(Timeout = 120000)]
    public async Task MannWhitneyU_IdenticalGroups_HighPValue()
    {
        var g1 = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var g2 = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var result = StatisticsHelper<double>.MannWhitneyUTest(g1, g2);
        // P-value should be high for identical distributions
        Assert.True(result.PValue > 0.05,
            $"P-value ({result.PValue}) should be > 0.05 for identical groups");
    }

    [Fact(Timeout = 120000)]
    public async Task MannWhitneyU_CompleteSeparation_SmallPValue()
    {
        // All values in g1 are below all values in g2
        var g1 = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var g2 = new Vector<double>(new double[] { 100, 200, 300, 400, 500 });
        var result = StatisticsHelper<double>.MannWhitneyUTest(g1, g2);
        // U should be 0 (no values in g1 exceed values in g2)
        Assert.Equal(0.0, result.UStatistic, Tol);
    }

    [Fact(Timeout = 120000)]
    public async Task MannWhitneyU_Symmetry()
    {
        // U1 + U2 = n1 * n2
        var g1 = new Vector<double>(new double[] { 1, 3, 5, 7 });
        var g2 = new Vector<double>(new double[] { 2, 4, 6, 8 });
        var result = StatisticsHelper<double>.MannWhitneyUTest(g1, g2);
        // U is min(U1, U2), and U1 + U2 = n1*n2 = 16
        // So U <= 8
        Assert.True(result.UStatistic <= 16.0 + Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // CHI-SQUARE TEST
    // ═══════════════════════════════════════════════════════════════

    [Fact(Timeout = 120000)]
    public async Task ChiSquare_IdenticalDistributions_HighPValue()
    {
        // Both groups have same distribution => not significant
        var g1 = new Vector<double>(new double[] { 1, 1, 2, 2, 3, 3 });
        var g2 = new Vector<double>(new double[] { 1, 1, 2, 2, 3, 3 });
        var result = StatisticsHelper<double>.ChiSquareTest(g1, g2);
        Assert.Equal(0.0, result.ChiSquareStatistic, Tol);
        Assert.False(result.IsSignificant);
    }

    [Fact(Timeout = 120000)]
    public async Task ChiSquare_VeryDifferentDistributions_Significant()
    {
        // g1 all category 1, g2 all category 2 => very different
        var g1 = new Vector<double>(new double[] { 1, 1, 1, 1, 1 });
        var g2 = new Vector<double>(new double[] { 2, 2, 2, 2, 2 });
        var result = StatisticsHelper<double>.ChiSquareTest(g1, g2);
        Assert.True(result.ChiSquareStatistic > 0);
        Assert.True(result.IsSignificant,
            $"Chi-square test should be significant for completely different distributions (chi2={result.ChiSquareStatistic}, p={result.PValue})");
    }

    [Fact(Timeout = 120000)]
    public async Task ChiSquare_DegreesOfFreedom_IsCategoriesMinusOne()
    {
        // 3 unique categories => df = 3 - 1 = 2
        var g1 = new Vector<double>(new double[] { 1, 2, 3, 1, 2 });
        var g2 = new Vector<double>(new double[] { 1, 2, 3, 2, 3 });
        var result = StatisticsHelper<double>.ChiSquareTest(g1, g2);
        Assert.Equal(2, result.DegreesOfFreedom);
    }

    [Fact(Timeout = 120000)]
    public async Task ChiSquare_ChiStatistic_IsNonNegative()
    {
        var g1 = new Vector<double>(new double[] { 1, 1, 2, 3 });
        var g2 = new Vector<double>(new double[] { 2, 3, 3, 1 });
        var result = StatisticsHelper<double>.ChiSquareTest(g1, g2);
        Assert.True(result.ChiSquareStatistic >= 0);
    }

    [Theory(Timeout = 120000)]
    [InlineData(3.841458820694124, 1)]
    [InlineData(5.991464547107979, 2)]
    [InlineData(9.487729036781154, 4)]
    public async Task ChiSquarePValue_AtTheFivePercentCriticalValue_IsFivePercent(double chiSquare, int degreesOfFreedom)
    {
        // The arguments are scipy.stats.chi2.ppf(0.95, df); the survival function there is 0.05 by
        // definition. Closed forms: df=1 erfc(sqrt(x/2)), df=2 exp(-x/2), df=4 exp(-x/2)(1 + x/2).
        double p = StatisticsHelper<double>.ChiSquarePValue(chiSquare, degreesOfFreedom);
        Assert.Equal(0.05, p, 1e-9);
    }

    [Fact(Timeout = 120000)]
    public async Task ChiSquare_CriticalValue_IsTheChiSquareQuantile()
    {
        // Two categories => df = 1; at alpha = 0.05 the critical value is scipy.stats.chi2.ppf(0.95, 1).
        var g1 = new Vector<double>(new double[] { 1, 1, 2, 2 });
        var g2 = new Vector<double>(new double[] { 1, 2, 2, 2 });
        var result = StatisticsHelper<double>.ChiSquareTest(g1, g2);
        Assert.Equal(1, result.DegreesOfFreedom);
        Assert.Equal(3.841458820694124, result.CriticalValue, 1e-6);
    }

    [Theory(Timeout = 120000)]
    [InlineData(1.0, -0.57721566490153286)]   // psi(1) = -gamma (Euler-Mascheroni)
    [InlineData(0.5, -1.9635100260214235)]    // psi(1/2) = -gamma - 2 ln 2
    [InlineData(2.0, 0.42278433509846714)]    // psi(2) = 1 - gamma
    public async Task Digamma_MatchesClosedForms(double x, double expected)
    {
        Assert.Equal(expected, StatisticsHelper<double>.Digamma(x), 1e-9);
    }

    [Fact(Timeout = 120000)]
    public async Task IncompleteGamma_ShapeOne_IsExponentialCdf()
    {
        // P(1, x) = 1 - exp(-x)
        Assert.Equal(1.0 - Math.Exp(-2.0), StatisticsHelper<double>.IncompleteGamma(1.0, 2.0), 1e-9);
    }

    // ═══════════════════════════════════════════════════════════════
    // F-TEST
    // ═══════════════════════════════════════════════════════════════

    [Fact(Timeout = 120000)]
    public async Task FTest_EqualVariances_FStatisticNearOne()
    {
        // Groups with same variance => F ≈ 1
        var g1 = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var g2 = new Vector<double>(new double[] { 11, 12, 13, 14, 15 });
        var result = StatisticsHelper<double>.FTest(g1, g2);
        Assert.Equal(1.0, result.FStatistic, 0.01);
    }

    [Fact(Timeout = 120000)]
    public async Task FTest_VeryDifferentVariances_LargeFStatistic()
    {
        // g1 low variance, g2 high variance
        var g1 = new Vector<double>(new double[] { 10.0, 10.1, 9.9, 10.0, 10.2 });
        var g2 = new Vector<double>(new double[] { 1, 20, 5, 25, 10 });
        var result = StatisticsHelper<double>.FTest(g1, g2);
        Assert.True(result.FStatistic > 1.0,
            $"F-statistic ({result.FStatistic}) should be > 1 for different variances");
    }

    [Fact(Timeout = 120000)]
    public async Task FTest_DegreesOfFreedom_AreCorrect()
    {
        // Textbook two-sample F-test puts the LARGER sample variance in the
        // numerator (so F ≥ 1 by construction). The degrees of freedom must
        // follow whichever group's variance ended up in the numerator —
        // df_num = n_larger-var - 1, df_denom = n_smaller-var - 1.
        // For g1 = {1, 2, 3} (sample var ≈ 1.0, n=3) vs g2 = {4, 5, 6, 7}
        // (sample var ≈ 1.67, n=4), the LARGER variance is g2's, so g2
        // lands in the numerator with df_num = 4 - 1 = 3 and g1 in the
        // denominator with df_denom = 3 - 1 = 2.
        //
        // The earlier assertion (df_num = 2 from n_left - 1, df_denom = 3
        // from n_right - 1) was from the pre-fix implementation that
        // hard-wired numerator/denominator to leftY/rightY regardless of
        // which variance was larger — that produced the wrong F
        // distribution (and wrong CI quantiles) whenever rightVariance >
        // leftVariance. The source's current arithmetic is the
        // statistically correct one; see comment at
        // StatisticsHelper<T>.FTest line 913-920.
        var g1 = new Vector<double>(new double[] { 1, 2, 3 }); // var ≈ 1.0
        var g2 = new Vector<double>(new double[] { 4, 5, 6, 7 }); // var ≈ 1.67 (larger)
        var result = StatisticsHelper<double>.FTest(g1, g2);
        Assert.Equal(3, result.NumeratorDegreesOfFreedom); // g2's n - 1
        Assert.Equal(2, result.DenominatorDegreesOfFreedom); // g1's n - 1
    }

    [Fact(Timeout = 120000)]
    public async Task FTest_FStatistic_IsGreaterThanOrEqualOne()
    {
        // Implementation takes max(var1,var2)/min(var1,var2) so F >= 1
        var g1 = new Vector<double>(new double[] { 1, 3, 5, 7, 9 });
        var g2 = new Vector<double>(new double[] { 2, 4, 6 });
        var result = StatisticsHelper<double>.FTest(g1, g2);
        Assert.True(result.FStatistic >= 1.0 - Tol);
    }

    [Fact(Timeout = 120000)]
    public async Task FTest_BothZeroVariance_Throws()
    {
        var g1 = new Vector<double>(new double[] { 5, 5, 5 });
        var g2 = new Vector<double>(new double[] { 5, 5, 5 });
        Assert.Throws<InvalidOperationException>(() => StatisticsHelper<double>.FTest(g1, g2));
    }

    [Fact(Timeout = 120000)]
    public async Task FTest_TooFewElements_Throws()
    {
        var g1 = new Vector<double>(new double[] { 1 });
        var g2 = new Vector<double>(new double[] { 2, 3 });
        Assert.Throws<ArgumentException>(() => StatisticsHelper<double>.FTest(g1, g2));
    }

    // ═══════════════════════════════════════════════════════════════
    // CROSS-PROPERTY TESTS
    // ═══════════════════════════════════════════════════════════════

    [Fact(Timeout = 120000)]
    public async Task Variance_EqualsStdDev_Squared()
    {
        var values = new double[] { 3, 7, 1, 9, 5 };
        double variance = StatisticsHelper<double>.CalculateVariance(values);
        double stdDev = StatisticsHelper<double>.CalculateStandardDeviation(values);
        Assert.Equal(variance, stdDev * stdDev, Tol);
    }

    [Fact(Timeout = 120000)]
    public async Task MAD_LessThanOrEqual_StdDev()
    {
        // MAD <= StdDev for normal-like distributions
        // (more generally, MAD * sqrt(2/pi) ≈ StdDev for normal)
        var values = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        var vec = new Vector<double>(values);
        double median = StatisticsHelper<double>.CalculateMedian(values);
        double mad = StatisticsHelper<double>.CalculateMeanAbsoluteDeviation(vec, median);
        double stdDev = StatisticsHelper<double>.CalculateStandardDeviation(values);
        Assert.True(mad <= stdDev + Tol,
            $"MAD ({mad}) should be <= StdDev ({stdDev})");
    }

    [Fact(Timeout = 120000)]
    public async Task TTest_PValue_BetweenZeroAndOne()
    {
        var g1 = new Vector<double>(new double[] { 1, 3, 5, 7, 9 });
        var g2 = new Vector<double>(new double[] { 2, 4, 6, 8, 10 });
        var result = StatisticsHelper<double>.TTest(g1, g2);
        Assert.True(result.PValue >= 0 && result.PValue <= 1.0,
            $"P-value ({result.PValue}) should be in [0, 1]");
    }

    [Fact(Timeout = 120000)]
    public async Task ChiSquare_PValue_BetweenZeroAndOne()
    {
        var g1 = new Vector<double>(new double[] { 1, 1, 2, 3 });
        var g2 = new Vector<double>(new double[] { 2, 3, 3, 1 });
        var result = StatisticsHelper<double>.ChiSquareTest(g1, g2);
        Assert.True(result.PValue >= 0 && result.PValue <= 1.0,
            $"P-value ({result.PValue}) should be in [0, 1]");
    }

    [Fact(Timeout = 120000)]
    public async Task FTest_PValue_BetweenZeroAndOne()
    {
        var g1 = new Vector<double>(new double[] { 1, 3, 5, 7, 9 });
        var g2 = new Vector<double>(new double[] { 2, 4, 6 });
        var result = StatisticsHelper<double>.FTest(g1, g2);
        Assert.True(result.PValue >= 0 && result.PValue <= 1.0,
            $"P-value ({result.PValue}) should be in [0, 1]");
    }

    // ═══════════════════════════════════════════════════════════════
    // EDGE CASES
    // ═══════════════════════════════════════════════════════════════

    [Fact(Timeout = 120000)]
    public async Task Median_NegativeValues()
    {
        var values = new double[] { -5, -3, -1, 1, 3 };
        double result = StatisticsHelper<double>.CalculateMedian(values);
        Assert.Equal(-1.0, result, Tol);
    }

    [Fact(Timeout = 120000)]
    public async Task Variance_LargeValues_StillAccurate()
    {
        // Values centered around 1,000,000 with small variance
        var values = new double[] { 1000001, 1000002, 1000003, 1000004, 1000005 };
        double result = StatisticsHelper<double>.CalculateVariance(values);
        Assert.Equal(2.5, result, Tol);
    }

    [Fact(Timeout = 120000)]
    public async Task MSE_Symmetric_PredActual()
    {
        // MSE(actual, pred) = MSE(pred, actual)
        var a = new double[] { 1, 2, 3, 4, 5 };
        var b = new double[] { 1.5, 2.5, 3.5, 4.5, 5.5 };
        double mse1 = StatisticsHelper<double>.CalculateMeanSquaredError(a, b);
        double mse2 = StatisticsHelper<double>.CalculateMeanSquaredError(b, a);
        Assert.Equal(mse1, mse2, Tol);
    }

    [Fact(Timeout = 120000)]
    public async Task JackknifeInterval_OfMean_UsesStandardErrorOfMean()
    {
        // For the mean, the jackknife standard error sqrt((n - 1) / n * sum (theta_i - theta_bar)^2)
        // equals s / sqrt(n) (Efron & Tibshirani; astropy.stats.jackknife_stats). Here s / sqrt(5) = sqrt(0.5).
        var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        var (lower, upper) = StatisticsHelper<double>.CalculateJackknifeInterval(actual, predicted);
        double t = StatisticsHelper<double>.CalculateTValue(4, 0.95);

        Assert.Equal(3.0, (lower + upper) / 2.0, 1e-12);
        Assert.Equal(Math.Sqrt(0.5) * t, (upper - lower) / 2.0, 1e-12);
    }
}

public class ElementwiseSweepRegressionTests
{
    [Fact]
    public void HalfSnnReadout_ZeroGradientProducesAZeroFiniteUpdate()
    {
        using var model = CreateHalfSnn();
        var update = RunHalfReadoutStep(model, (Half)0, new double[1], new double[1], step: 1);
        Assert.Equal((Half)0, update[0]);
    }

    [Fact]
    public void HalfSnnReadout_TinyGradientAndLongHistoryKeepTheAdamVariance()
    {
        using var model = CreateHalfSnn();
        var firstMoment = new double[1];
        var secondMoment = new double[1];
        Half error = (Half)0.0001;
        var update = RunHalfReadoutStep(model, error, firstMoment, secondMoment, step: 1);
        double gradient = -(double)error;
        double expectedVariance = 0.001 * gradient * gradient;
        Assert.InRange(Math.Abs(firstMoment[0] - 0.1 * gradient), 0, 1e-15);
        Assert.InRange(Math.Abs(secondMoment[0] - expectedVariance), 0, 1e-20);
        double expectedUpdate = -0.001 * gradient / (Math.Abs(gradient) + 1e-8);
        Assert.InRange((double)update[0], expectedUpdate - 1e-6, expectedUpdate + 1e-6);
    }

    [Fact]
    public void HalfSnnCheckpoint_PreservesWideMomentsAndTheNextAdamUpdate()
    {
        using var source = CreateHalfSnn();
        using var restored = CreateHalfSnn();
        var sourceM = SnnMoments(source, "_adamMWide");
        var sourceV = SnnMoments(source, "_adamVWide");
        sourceM[0] = new double[1];
        sourceV[0] = new double[1];
        RunHalfReadoutStep(source, (Half)0.0001, sourceM[0], sourceV[0], step: 1);
        var counter = Required(typeof(SpikingNeuralNetwork<Half>).GetField("_adamStep",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic));
        counter.SetValue(source, 1);
        restored.Deserialize(source.Serialize());
        var restoredM = SnnMoments(restored, "_adamMWide");
        var restoredV = SnnMoments(restored, "_adamVWide");
        Assert.Equal(sourceM[0], restoredM[0]);
        Assert.Equal(sourceV[0], restoredV[0]);
        Assert.Equal(1, (int)Required(counter.GetValue(restored)));
        var expected = RunHalfReadoutStep(source, (Half)0.0002, sourceM[0], sourceV[0], step: 2);
        var actual = RunHalfReadoutStep(restored, (Half)0.0002, restoredM[0], restoredV[0], step: 2);
        Assert.Equal(expected.ToArray(), actual.ToArray());
        Assert.Equal(sourceM[0], restoredM[0]);
        Assert.Equal(sourceV[0], restoredV[0]);
    }

    [Fact]
    public void HalfSnnTraining_UsesWideMomentsAndKeepsFiniteParameters()
    {
        using var model = CreateHalfSnn();
        model.Train(new Tensor<Half>(new Half[] { (Half)1, (Half)0.5, (Half)0.25, (Half)0.75 }, new[] { 4 }),
            new Tensor<Half>(new Half[] { (Half)0.2 }, new[] { 1 }));
        Assert.NotEmpty(SnnMoments(model, "_adamMWide"));
        Assert.All(model.GetParameters().ToArray(), value => Assert.True(!Half.IsNaN(value) && !Half.IsInfinity(value)));
    }

    private static SpikingNeuralNetwork<Half> CreateHalfSnn() => new(
        new NeuralNetworkArchitecture<Half>(inputType: AiDotNet.Enums.InputType.OneDimensional,
            taskType: AiDotNet.Enums.NeuralNetworkTaskType.Regression, inputSize: 4, outputSize: 1,
            layers: new List<AiDotNet.Interfaces.ILayer<Half>>
            {
                new AiDotNet.NeuralNetworks.Layers.SpikingLayer<Half>(4, 2),
                new AiDotNet.NeuralNetworks.Layers.DenseLayer<Half>(1)
            }),
        vectorActivation: null,
        options: new AiDotNet.NeuralNetworks.Options.SpikingNeuralNetworkOptions { SimulationSteps = 2 });

    [Fact]
    public void HalfSnnTraining_ImportsLegacyMomentsBeforeTheNextUpdate()
    {
        using var legacy = CreateHalfSnn();
        using var wide = CreateHalfSnn();
        var input = new Tensor<Half>(new Half[] { (Half)1, (Half)0.5, (Half)0.25, (Half)0.75 }, new[] { 4 });
        var target = new Tensor<Half>(new Half[] { (Half)0.2 }, new[] { 1 });
        legacy.Train(input, target);
        wide.Deserialize(legacy.Serialize());
        var flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        var legacyM = (Dictionary<int, Vector<Half>>)Required(
            Required(typeof(SpikingNeuralNetwork<Half>).GetField("_adamM", flags)).GetValue(legacy));
        var legacyV = (Dictionary<int, Vector<Half>>)Required(
            Required(typeof(SpikingNeuralNetwork<Half>).GetField("_adamV", flags)).GetValue(legacy));
        foreach (var pair in SnnMoments(legacy, "_adamMWide"))
        {
            legacyM[pair.Key] = new Vector<Half>(Enumerable.Repeat((Half)0.002, pair.Value.Length).ToArray());
            legacyV[pair.Key] = new Vector<Half>(Enumerable.Repeat((Half)0.0002, pair.Value.Length).ToArray());
            SnnMoments(wide, "_adamMWide")[pair.Key] = legacyM[pair.Key].ToArray().Select(value => (double)value).ToArray();
            SnnMoments(wide, "_adamVWide")[pair.Key] = legacyV[pair.Key].ToArray().Select(value => (double)value).ToArray();
        }
        SnnMoments(legacy, "_adamMWide").Clear();
        SnnMoments(legacy, "_adamVWide").Clear();
        legacy.Train(input, target);
        wide.Train(input, target);
        Assert.Empty(legacyM);
        Assert.Empty(legacyV);
        Assert.Equal(wide.GetParameters().ToArray(), legacy.GetParameters().ToArray());
        foreach (var pair in SnnMoments(wide, "_adamVWide"))
            Assert.Equal(pair.Value, SnnMoments(legacy, "_adamVWide")[pair.Key]);
    }

    private static Dictionary<int, double[]> SnnMoments(SpikingNeuralNetwork<Half> model, string name) =>
        (Dictionary<int, double[]>)Required(Required(typeof(SpikingNeuralNetwork<Half>).GetField(name,
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)).GetValue(model));

    /// <summary>The value, after asserting that it is not null.</summary>
    internal static TValue Required<TValue>(TValue? value) where TValue : class
    {
        Assert.NotNull(value);
        return value ?? throw new InvalidOperationException("Assert.NotNull let a null through.");
    }

    private static Vector<Half> RunHalfReadoutStep(SpikingNeuralNetwork<Half> model, Half error,
        double[] firstMoment, double[] secondMoment, int step)
    {
        const int steps = 300; // Half overflows when the normalization denominator forms 300^2.
        var spikes = Enumerable.Range(0, steps).Select(_ => new Vector<Half>(new Half[] { (Half)1 })).ToList();
        var history = new List<List<Vector<Half>>> { spikes };
        var updates = new Vector<Half>(1);
        var method = Required(typeof(SpikingNeuralNetwork<Half>).GetMethod("ApplyReadoutAdamStep",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic));
        method.Invoke(model, new object[] { 0, 0, 1, 1, spikes, history,
            new Vector<Half>(new[] { error }), updates, firstMoment, secondMoment, 0.001,
            1 - Math.Pow(0.9, step), 1 - Math.Pow(0.999, step) });
        return updates;
    }

    [Fact]
    public void BandMultiply_SkippedPaddingDoesNotPropagateNonfiniteCoefficients()
    {
        var band = new Matrix<double>(new[,] { { double.NaN, 2.0, 3.0 }, { 4.0, 5.0, 6.0 }, { 7.0, 8.0, double.NaN } });
        var result = new Vector<double>(new[] { double.NaN, double.NaN, double.NaN });
        MatrixHelper<double>.BandDiagonalMultiply(1, 1, band, result, new Vector<double>(new[] { 1.0, 2.0, 3.0 }));
        Assert.Equal(new[] { 8.0, 32.0, 38.0 }, result.ToArray());
        Assert.True(double.IsNaN(band[0, 0]));
        Assert.True(double.IsNaN(band[2, 2]));
    }

    [Fact]
    public void HalfTabDDPM_ConstantNumericalColumnNormalizesToZero()
    {
        using var model = new TabDDPMGenerator<Half>(
            new NeuralNetworkArchitecture<Half>(inputType: AiDotNet.Enums.InputType.OneDimensional,
                taskType: AiDotNet.Enums.NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1),
            new TabDDPMOptions<Half> { NumTimesteps = 2, NumCategoricalDiffusionSteps = 2,
                MLPDimensions = new[] { 4 }, TimestepEmbeddingDimension = 4, Seed = 7 });
        var columns = new[] { new ColumnMetadata("constant", ColumnDataType.Continuous) };
        var flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        Required(typeof(TabDDPMGenerator<Half>).GetMethod("AnalyzeColumns", flags)).Invoke(model, new object[] { columns });
        var data = new Matrix<Half>(new[,] { { (Half)5 }, { (Half)5 }, { (Half)5 }, { (Half)5 } });
        var prepared = (ValueTuple<Matrix<Half>, Matrix<Half>>)Required(Required(typeof(TabDDPMGenerator<Half>)
            .GetMethod("PreprocessData", flags)).Invoke(model, new object[] { data, columns }));
        Assert.All(prepared.Item1.AsSpan().ToArray(), value => Assert.Equal((Half)0, value));
    }

    [Fact]
    public void HalfNormClipping_LargeFiniteVector_UsesWideAccumulation()
    {
        const int count = 70000;
        var input = new Vector<Half>(Enumerable.Repeat((Half)1, count).ToArray());
        var clipped = Required(GradientClippingHelper.ClipByNorm(input, 10));
        double expected = 10 / Math.Sqrt(count);
        Assert.All(clipped.ToArray(), value => Assert.InRange((double)value, expected - 3e-5, expected + 3e-5));
        Assert.Equal(1.0, (double)input[0]);
    }

    [Fact]
    public void DoubleNormClipping_DoesNotSquareExtremeFiniteMagnitudes()
    {
        var input = new Vector<double>(new[] { 1e300, -1e300 });
        double expected = 1 / Math.Sqrt(2);
        var clipped = Required(GradientClippingHelper.ClipByNorm(input, 1));
        Assert.Equal(expected, clipped[0], 12);
        Assert.Equal(-expected, clipped[1], 12);
        var global = Required(GradientClippingHelper.ClipByGlobalNorm(
            new List<Vector<double>> { new(new[] { 1e300 }), new(new[] { -1e300 }) }, 1));
        Assert.Equal(expected, global[0][0], 12);
        Assert.Equal(-expected, global[1][0], 12);
    }

    [Fact]
    public void NormClipping_PreservesARepresentableSubnormalDirection()
    {
        var clipped = Required(GradientClippingHelper.ClipByNorm(new Vector<float>(new[] { 1e-40f, -1e-40f }), 1e-42));
        Assert.InRange(clipped[0], 6.9e-43f, 7.2e-43f);
        Assert.InRange(clipped[1], -7.2e-43f, -6.9e-43f);
    }

    [Fact]
    public void HalfCausalMask_IsFiniteAndDoesNotReadFutureTokens()
    {
        using var decoder = new T5RelativeBiasAttentionLayer<Half>(4, 1, bidirectional: false, seed: 5);
        var input = new Tensor<Half>(
            Enumerable.Range(0, 12).Select(i => (Half)(0.01 * (i + 1))).ToArray(), new[] { 3, 4 });
        var full = decoder.Forward(input).ToArray();
        Assert.All(full, value => Assert.True(!Half.IsNaN(value) && !Half.IsInfinity(value)));
        var prefix = new Tensor<Half>(input.ToArray().Take(4).ToArray(), new[] { 1, 4 });
        Assert.Equal(decoder.Forward(prefix).ToArray(), full.Take(4).ToArray());
    }

    [Fact]
    public void MemoryUsagePenalty_UniformReadAndWriteWeightsHavePositiveShannonEntropy()
    {
        using var model = new NeuralTuringMachine<double> { UseAuxiliaryLoss = true };
        Assert.Equal(2 * Math.Log(64), model.ComputeAuxiliaryLoss(), 10);
    }

    [Fact]
    public void MemoryRead_SkippedLocationsDoNotPropagateNonfiniteValues()
    {
        using var model = new NeuralTuringMachine<double>();
        var memory = new Matrix<double>(64, 32);
        memory.AsWritableSpan().Fill(double.NaN);
        for (int column = 0; column < 32; column++) memory[0, column] = column + 1;
        var weights = new Vector<double>(64);
        weights[0] = 1;
        var read = Required(typeof(NeuralTuringMachine<double>).GetMethod("ReadFromMemory",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic));
        var result = (Vector<double>)Required(read.Invoke(model, new object[] { memory, weights }));
        Assert.Equal(Enumerable.Range(1, 32).Select(value => (double)value).ToArray(), result.ToArray());
    }

    [Fact]
    public void TabDDPMCheckpoint_RestoresFittedNormalizationAndReconstructionRange()
    {
        TabDDPMGenerator<double> Create(int seed) => new(
            new NeuralNetworkArchitecture<double>(inputType: AiDotNet.Enums.InputType.OneDimensional,
                taskType: AiDotNet.Enums.NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1),
            new TabDDPMOptions<double> { NumTimesteps = 2, NumCategoricalDiffusionSteps = 2,
                MLPDimensions = new[] { 4 }, TimestepEmbeddingDimension = 4, BatchSize = 4, Epochs = 1, Seed = seed });
        using var source = Create(7);
        using var restored = Create(19);
        var columns = new[] { new ColumnMetadata("value", ColumnDataType.Continuous) };
        source.Fit(new Matrix<double>(new[,] { { 10.0 }, { 20.0 }, { 30.0 }, { 40.0 } }), columns, epochs: 1);
        restored.Fit(new Matrix<double>(new[,] { { -40.0 }, { -30.0 }, { -20.0 }, { -10.0 } }), columns, epochs: 1);
        restored.Deserialize(source.Serialize());
        var generated = restored.Generate(3);
        Assert.Equal(3, generated.Rows);
        Assert.Equal(1, generated.Columns);
        Assert.All(generated.AsSpan().ToArray(), value => Assert.InRange(value, 10, 40));
    }

    [Fact]
    public void JackknifeMeanInterval_UsesOnlyThePairedPredictionPrefix()
    {
        var actual = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
        var expected = StatisticsHelper<double>.CalculateJackknifeInterval(actual, new Vector<double>(new[] { 1.0, 2.0, 3.0 }));
        var extra = StatisticsHelper<double>.CalculateJackknifeInterval(actual, new Vector<double>(new[] { 1.0, 2.0, 3.0, 1e6 }));
        Assert.Equal(expected, extra);
    }

    [Fact]
    public void JackknifeMeanInterval_DoesNotLoseSpreadAtALargeCommonOffset()
    {
        const double offset = 1e16;
        var values = new Vector<double>(new[] { offset + 2, offset + 4, offset + 6, offset + 8, offset + 10 });
        var interval = StatisticsHelper<double>.CalculateJackknifeInterval(values, values);
        // Mean = offset + 6; s/sqrt(5) = sqrt(2), t(4,.975) ~= 2.776: endpoints round to +2 and +10.
        Assert.Equal(offset + 2, interval.Lower);
        Assert.Equal(offset + 10, interval.Upper);
    }
}

public class CtcGreedyDecodingTests
{
    private static (List<int> tokens, double confidence) Reference(
        Tensor<double> logits, int maxTokens, int framesAfterEmission = 1)
    {
        var tokens = new List<int>();
        double totalConf = 0;
        int confCount = 0;
        int prevToken = -1;
        int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1;
        int vocabSize = logits.Rank >= 2 ? logits.Shape[logits.Rank - 1] : logits.Shape[0];
        int t = 0;
        while (t < numFrames && tokens.Count < maxTokens)
        {
            int maxIdx = 0;
            double maxVal = double.NegativeInfinity;
            for (int v = 0; v < vocabSize; v++)
            {
                double val = logits.Rank >= 2 ? logits[t, v] : logits[v];
                if (val > maxVal) { maxVal = val; maxIdx = v; }
            }

            double sumExp = 0;
            for (int v = 0; v < vocabSize; v++)
            {
                double val = logits.Rank >= 2 ? logits[t, v] : logits[v];
                sumExp += Math.Exp(val - maxVal);
            }

            double frameConf = 1.0 / sumExp;
            if (maxIdx != prevToken && maxIdx > 0)
            {
                tokens.Add(maxIdx);
                totalConf += frameConf;
                confCount++;
                t += framesAfterEmission;
            }
            else
            {
                t++;
            }

            prevToken = maxIdx;
        }

        return (tokens, confCount > 0 ? totalConf / confCount : 0.0);
    }

    /// <summary>Small integer logits, so equal maxima within a frame are common.</summary>
    private static Tensor<double> TiedLogits(int frames, int vocab, int seed)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var data = new double[frames * vocab];
        for (int i = 0; i < data.Length; i++) data[i] = rng.Next(0, 4);
        return new Tensor<double>(data, new[] { frames, vocab });
    }

    private static void AssertSame((List<int> tokens, double confidence) expected, (List<int> tokens, double confidence) actual)
    {
        Assert.Equal(expected.tokens, actual.tokens);
        if (double.IsNaN(expected.confidence))
            Assert.True(double.IsNaN(actual.confidence), $"Expected NaN confidence, got {actual.confidence}.");
        else
            Assert.Equal(expected.confidence, actual.confidence, 12);
    }

    [Theory]
    [InlineData(1, 5, int.MaxValue)]
    [InlineData(40, 6, int.MaxValue)]
    [InlineData(40, 6, 3)]
    [InlineData(25, 1, int.MaxValue)]
    [InlineData(25, 4, 0)]
    public void MatchesTheScalarScan_IncludingTies(int frames, int vocab, int maxTokens)
    {
        for (int seed = 0; seed < 5; seed++)
        {
            var logits = TiedLogits(frames, vocab, seed);
            AssertSame(Reference(logits, maxTokens), CtcGreedyDecoding.Decode(logits, maxTokens));
        }
    }

    [Fact]
    public void LowestIndexWinsATie()
    {
        var logits = new Tensor<double>(new double[] { 0, 2, 2, 1 }, new[] { 1, 4 });
        var (tokens, _) = CtcGreedyDecoding.Decode(logits, int.MaxValue);
        Assert.Equal(new List<int> { 1 }, tokens);
    }

    [Fact]
    public void NaNAndNegativeInfinityNeverWin()
    {
        double nan = double.NaN;
        double negInf = double.NegativeInfinity;
        var logits = new Tensor<double>(
            new[]
            {
                nan, 1.0, 3.0, nan,       // NaN skipped for the argmax, but poisons that frame's confidence
                negInf, negInf, negInf, 2.0,
                nan, nan, nan, nan,       // nothing can win: the blank
                negInf, negInf, negInf, negInf,
                0.5, 0.5, double.PositiveInfinity, 0.5,
            },
            new[] { 5, 4 });

        AssertSame(Reference(logits, int.MaxValue), CtcGreedyDecoding.Decode(logits, int.MaxValue));
    }

    [Fact]
    public void ASingleFrameVectorDecodesAsOneFrame()
    {
        var logits = new Tensor<double>(new[] { 0.1, 0.2, 0.9, 0.3 }, new[] { 4 });
        AssertSame(Reference(logits, int.MaxValue), CtcGreedyDecoding.Decode(logits, int.MaxValue));
    }

    [Fact]
    public void RankThreeLogitsAreRejectedLikeTheTwoIndexRead()
    {
        var logits = new Tensor<double>(new[] { 2, 3, 4 });
        Assert.Throws<ArgumentException>(() => CtcGreedyDecoding.Decode(logits, int.MaxValue));
    }

    [Fact]
    public void FlatDecodingReadsTheLeadingFramesTimesVocabElements()
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(11);
        var data = new double[2 * 3 * 4];
        for (int i = 0; i < data.Length; i++) data[i] = rng.Next(0, 3);
        var rankThree = new Tensor<double>(data, new[] { 2, 3, 4 });

        // frames = 2, vocab = 4: the first 8 flat elements, as two rows.
        var leadingData = new double[8];
        Array.Copy(data, leadingData, 8);
        var leading = new Tensor<double>(leadingData, new[] { 2, 4 });
        AssertSame(Reference(leading, int.MaxValue), CtcGreedyDecoding.DecodeFlat(rankThree, int.MaxValue));
    }

    [Theory]
    [InlineData(2, int.MaxValue)]
    [InlineData(1, 5)]
    [InlineData(0, int.MaxValue)]
    public void FrameSkippingMatchesTheScalarWalk(int framesAfterEmission, int maxTokens)
    {
        for (int seed = 0; seed < 5; seed++)
        {
            var logits = TiedLogits(30, 5, seed + 20);
            AssertSame(
                Reference(logits, maxTokens, framesAfterEmission),
                CtcGreedyDecoding.DecodeSkippingAfterEmission(logits, maxTokens, framesAfterEmission));
        }
    }
    [Fact]
    public void HalfWideVocabulary_ConfidenceDoesNotOverflowTheDenominator()
    {
        const int vocab = 70000;
        var values = new Half[vocab];
        values[1] = (Half)0.01;
        var logits = new Tensor<Half>(values, new[] { 1, vocab });
        var result = CtcGreedyDecoding.Decode(logits, 1);
        Assert.Equal(new[] { 1 }, result.tokens);
        double expected = 1.0 / (1.0 + (vocab - 1) * Math.Exp(-(double)values[1]));
        Assert.InRange(result.confidence, expected - 1e-10, expected + 1e-10);
        var flat = CtcGreedyDecoding.DecodeFlat(logits, 1);
        var skipped = CtcGreedyDecoding.DecodeSkippingAfterEmission(logits, 1, 1);
        Assert.Equal(result.tokens, flat.tokens);
        Assert.Equal(result.tokens, skipped.tokens);
        Assert.Equal(result.confidence, flat.confidence);
        Assert.Equal(result.confidence, skipped.confidence);
        var singlePrecision = CtcGreedyDecoding.Decode(logits.Cast<float>(), 1);
        Assert.InRange(singlePrecision.confidence, expected - 1e-10, expected + 1e-10);
    }

    [Fact]
    public void HalfLongUtterance_EmittedConfidenceSumDoesNotOverflow()
    {
        const int frames = 70000;
        var values = new Half[frames * 3];
        for (int frame = 0; frame < frames; frame++)
            values[frame * 3 + 1 + frame % 2] = (Half)100;
        var result = CtcGreedyDecoding.Decode(new Tensor<Half>(values, new[] { frames, 3 }), frames);
        Assert.Equal(frames, result.tokens.Count);
        Assert.Equal(1.0, result.confidence, 6);
    }

}
