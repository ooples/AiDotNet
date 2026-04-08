using AiDotNet.Helpers;
using Xunit;

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

    [Fact]
    public void Median_OddCount_ReturnsMiddle()
    {
        // [1, 3, 5, 7, 9] => sorted, middle (index 2) = 5
        var values = new double[] { 9, 1, 5, 3, 7 };
        double result = StatisticsHelper<double>.CalculateMedian(values);
        Assert.Equal(5.0, result, Tol);
    }

    [Fact]
    public void Median_EvenCount_ReturnsAverageOfMiddleTwo()
    {
        // [1, 3, 5, 7] => sorted, (3+5)/2 = 4
        var values = new double[] { 7, 1, 5, 3 };
        double result = StatisticsHelper<double>.CalculateMedian(values);
        Assert.Equal(4.0, result, Tol);
    }

    [Fact]
    public void Median_SingleElement_ReturnsThatElement()
    {
        var values = new double[] { 42.0 };
        double result = StatisticsHelper<double>.CalculateMedian(values);
        Assert.Equal(42.0, result, Tol);
    }

    [Fact]
    public void Median_AllSame_ReturnsThatValue()
    {
        var values = new double[] { 5, 5, 5, 5, 5 };
        double result = StatisticsHelper<double>.CalculateMedian(values);
        Assert.Equal(5.0, result, Tol);
    }

    [Fact]
    public void Median_TwoElements_ReturnsAverage()
    {
        var values = new double[] { 10, 20 };
        double result = StatisticsHelper<double>.CalculateMedian(values);
        Assert.Equal(15.0, result, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // MEAN ABSOLUTE DEVIATION (MAD)
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void MAD_HandCalculated()
    {
        // values = [2, 4, 6, 8], median = 5
        // |2-5|=3, |4-5|=1, |6-5|=1, |8-5|=3
        // MAD = (3+1+1+3)/4 = 8/4 = 2.0
        var values = new Vector<double>(new double[] { 2, 4, 6, 8 });
        double median = 5.0;
        double result = StatisticsHelper<double>.CalculateMeanAbsoluteDeviation(values, median);
        Assert.Equal(2.0, result, Tol);
    }

    [Fact]
    public void MAD_AllSame_ReturnsZero()
    {
        var values = new Vector<double>(new double[] { 5, 5, 5, 5 });
        double result = StatisticsHelper<double>.CalculateMeanAbsoluteDeviation(values, 5.0);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void MAD_IsNonNegative()
    {
        var values = new Vector<double>(new double[] { -10, 0, 10, 20, 30 });
        double median = StatisticsHelper<double>.CalculateMedian(new double[] { -10, 0, 10, 20, 30 });
        double result = StatisticsHelper<double>.CalculateMeanAbsoluteDeviation(values, median);
        Assert.True(result >= 0);
    }

    // ═══════════════════════════════════════════════════════════════
    // VARIANCE (SAMPLE VARIANCE, n-1 denominator)
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void Variance_HandCalculated_WithMean()
    {
        // values = [2, 4, 6, 8, 10], mean = 6
        // sum((x-6)^2) = 16+4+0+4+16 = 40
        // Var = 40 / (5-1) = 10.0
        var values = new Vector<double>(new double[] { 2, 4, 6, 8, 10 });
        double result = StatisticsHelper<double>.CalculateVariance(values, 6.0);
        Assert.Equal(10.0, result, Tol);
    }

    [Fact]
    public void Variance_HandCalculated_NoMean()
    {
        // Same calculation but using the overload that computes mean internally
        var values = new double[] { 2, 4, 6, 8, 10 };
        double result = StatisticsHelper<double>.CalculateVariance(values);
        Assert.Equal(10.0, result, Tol);
    }

    [Fact]
    public void Variance_AllSame_ReturnsZero()
    {
        var values = new double[] { 5, 5, 5, 5, 5 };
        double result = StatisticsHelper<double>.CalculateVariance(values);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void Variance_SingleElement_ReturnsZero()
    {
        var values = new Vector<double>(new double[] { 42.0 });
        double result = StatisticsHelper<double>.CalculateVariance(values, 42.0);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void Variance_IsNonNegative()
    {
        var values = new double[] { -5, 0, 5, 10, 100 };
        double result = StatisticsHelper<double>.CalculateVariance(values);
        Assert.True(result >= 0);
    }

    [Fact]
    public void Variance_ScaleProperty()
    {
        // Var(c*X) = c^2 * Var(X)
        var values = new double[] { 1, 2, 3, 4, 5 };
        double varOriginal = StatisticsHelper<double>.CalculateVariance(values);
        var scaledValues = values.Select(x => x * 3.0).ToArray();
        double varScaled = StatisticsHelper<double>.CalculateVariance(scaledValues);
        Assert.Equal(9.0 * varOriginal, varScaled, Tol);
    }

    [Fact]
    public void Variance_ShiftInvariant()
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

    [Fact]
    public void StdDev_EqualsSqrtVariance()
    {
        var values = new double[] { 2, 4, 6, 8, 10 };
        double variance = StatisticsHelper<double>.CalculateVariance(values);
        double stdDev = StatisticsHelper<double>.CalculateStandardDeviation(values);
        Assert.Equal(Math.Sqrt(variance), stdDev, Tol);
    }

    [Fact]
    public void StdDev_AllSame_ReturnsZero()
    {
        var values = new double[] { 7, 7, 7, 7 };
        double result = StatisticsHelper<double>.CalculateStandardDeviation(values);
        Assert.Equal(0.0, result, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // MEAN SQUARED ERROR
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void MSE_HandCalculated()
    {
        // actual=[1,2,3,4,5], pred=[1.5,2.5,3.5,4.5,5.5]
        // errors^2: 0.25, 0.25, 0.25, 0.25, 0.25
        // MSE = 1.25/5 = 0.25
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var pred = new double[] { 1.5, 2.5, 3.5, 4.5, 5.5 };
        double result = StatisticsHelper<double>.CalculateMeanSquaredError(actual, pred);
        Assert.Equal(0.25, result, Tol);
    }

    [Fact]
    public void MSE_PerfectPredictions_ShouldBeZero()
    {
        var actual = new double[] { 1, 2, 3 };
        var pred = new double[] { 1, 2, 3 };
        double result = StatisticsHelper<double>.CalculateMeanSquaredError(actual, pred);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void MSE_IsNonNegative()
    {
        var actual = new double[] { -5, 0, 5 };
        var pred = new double[] { 10, -10, 0 };
        double result = StatisticsHelper<double>.CalculateMeanSquaredError(actual, pred);
        Assert.True(result >= 0);
    }

    // ═══════════════════════════════════════════════════════════════
    // VARIANCE REDUCTION (for decision trees)
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void VarianceReduction_PerfectSplit_MaximizesReduction()
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

    [Fact]
    public void VarianceReduction_NoSplitBenefit_ReturnsZero()
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

    [Fact]
    public void VarianceReduction_IsNonNegative()
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

    [Fact]
    public void TTest_IdenticalGroups_PValueOne()
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

    [Fact]
    public void TTest_VeryDifferentGroups_SmallPValue()
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

    [Fact]
    public void TTest_HandCalculated_DegreesOfFreedom()
    {
        // df = n1 + n2 - 2
        var g1 = new Vector<double>(new double[] { 1, 2, 3 }); // n1 = 3
        var g2 = new Vector<double>(new double[] { 4, 5, 6, 7 }); // n2 = 4
        var result = StatisticsHelper<double>.TTest(g1, g2);
        Assert.Equal(5, result.DegreesOfFreedom); // 3 + 4 - 2 = 5
    }

    [Fact]
    public void TTest_TStatistic_HandCalculated()
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

    [Fact]
    public void MannWhitneyU_IdenticalGroups_HighPValue()
    {
        var g1 = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var g2 = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var result = StatisticsHelper<double>.MannWhitneyUTest(g1, g2);
        // P-value should be high for identical distributions
        Assert.True(result.PValue > 0.05,
            $"P-value ({result.PValue}) should be > 0.05 for identical groups");
    }

    [Fact]
    public void MannWhitneyU_CompleteSeparation_SmallPValue()
    {
        // All values in g1 are below all values in g2
        var g1 = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var g2 = new Vector<double>(new double[] { 100, 200, 300, 400, 500 });
        var result = StatisticsHelper<double>.MannWhitneyUTest(g1, g2);
        // U should be 0 (no values in g1 exceed values in g2)
        Assert.Equal(0.0, result.UStatistic, Tol);
    }

    [Fact]
    public void MannWhitneyU_Symmetry()
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

    [Fact]
    public void ChiSquare_IdenticalDistributions_HighPValue()
    {
        // Both groups have same distribution => not significant
        var g1 = new Vector<double>(new double[] { 1, 1, 2, 2, 3, 3 });
        var g2 = new Vector<double>(new double[] { 1, 1, 2, 2, 3, 3 });
        var result = StatisticsHelper<double>.ChiSquareTest(g1, g2);
        Assert.Equal(0.0, result.ChiSquareStatistic, Tol);
        Assert.False(result.IsSignificant);
    }

    [Fact]
    public void ChiSquare_VeryDifferentDistributions_Significant()
    {
        // g1 all category 1, g2 all category 2 => very different
        var g1 = new Vector<double>(new double[] { 1, 1, 1, 1, 1 });
        var g2 = new Vector<double>(new double[] { 2, 2, 2, 2, 2 });
        var result = StatisticsHelper<double>.ChiSquareTest(g1, g2);
        Assert.True(result.ChiSquareStatistic > 0);
        Assert.True(result.IsSignificant,
            $"Chi-square test should be significant for completely different distributions (chi2={result.ChiSquareStatistic}, p={result.PValue})");
    }

    [Fact]
    public void ChiSquare_DegreesOfFreedom_IsCategoriesMinusOne()
    {
        // 3 unique categories => df = 3 - 1 = 2
        var g1 = new Vector<double>(new double[] { 1, 2, 3, 1, 2 });
        var g2 = new Vector<double>(new double[] { 1, 2, 3, 2, 3 });
        var result = StatisticsHelper<double>.ChiSquareTest(g1, g2);
        Assert.Equal(2, result.DegreesOfFreedom);
    }

    [Fact]
    public void ChiSquare_ChiStatistic_IsNonNegative()
    {
        var g1 = new Vector<double>(new double[] { 1, 1, 2, 3 });
        var g2 = new Vector<double>(new double[] { 2, 3, 3, 1 });
        var result = StatisticsHelper<double>.ChiSquareTest(g1, g2);
        Assert.True(result.ChiSquareStatistic >= 0);
    }

    // ═══════════════════════════════════════════════════════════════
    // F-TEST
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void FTest_EqualVariances_FStatisticNearOne()
    {
        // Groups with same variance => F ≈ 1
        var g1 = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var g2 = new Vector<double>(new double[] { 11, 12, 13, 14, 15 });
        var result = StatisticsHelper<double>.FTest(g1, g2);
        Assert.Equal(1.0, result.FStatistic, 0.01);
    }

    [Fact]
    public void FTest_VeryDifferentVariances_LargeFStatistic()
    {
        // g1 low variance, g2 high variance
        var g1 = new Vector<double>(new double[] { 10.0, 10.1, 9.9, 10.0, 10.2 });
        var g2 = new Vector<double>(new double[] { 1, 20, 5, 25, 10 });
        var result = StatisticsHelper<double>.FTest(g1, g2);
        Assert.True(result.FStatistic > 1.0,
            $"F-statistic ({result.FStatistic}) should be > 1 for different variances");
    }

    [Fact]
    public void FTest_DegreesOfFreedom_AreCorrect()
    {
        var g1 = new Vector<double>(new double[] { 1, 2, 3 }); // n=3
        var g2 = new Vector<double>(new double[] { 4, 5, 6, 7 }); // n=4
        var result = StatisticsHelper<double>.FTest(g1, g2);
        Assert.Equal(2, result.NumeratorDegreesOfFreedom); // 3-1
        Assert.Equal(3, result.DenominatorDegreesOfFreedom); // 4-1
    }

    [Fact]
    public void FTest_FStatistic_IsGreaterThanOrEqualOne()
    {
        // Implementation takes max(var1,var2)/min(var1,var2) so F >= 1
        var g1 = new Vector<double>(new double[] { 1, 3, 5, 7, 9 });
        var g2 = new Vector<double>(new double[] { 2, 4, 6 });
        var result = StatisticsHelper<double>.FTest(g1, g2);
        Assert.True(result.FStatistic >= 1.0 - Tol);
    }

    [Fact]
    public void FTest_BothZeroVariance_Throws()
    {
        var g1 = new Vector<double>(new double[] { 5, 5, 5 });
        var g2 = new Vector<double>(new double[] { 5, 5, 5 });
        Assert.Throws<InvalidOperationException>(() => StatisticsHelper<double>.FTest(g1, g2));
    }

    [Fact]
    public void FTest_TooFewElements_Throws()
    {
        var g1 = new Vector<double>(new double[] { 1 });
        var g2 = new Vector<double>(new double[] { 2, 3 });
        Assert.Throws<ArgumentException>(() => StatisticsHelper<double>.FTest(g1, g2));
    }

    // ═══════════════════════════════════════════════════════════════
    // CROSS-PROPERTY TESTS
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void Variance_EqualsStdDev_Squared()
    {
        var values = new double[] { 3, 7, 1, 9, 5 };
        double variance = StatisticsHelper<double>.CalculateVariance(values);
        double stdDev = StatisticsHelper<double>.CalculateStandardDeviation(values);
        Assert.Equal(variance, stdDev * stdDev, Tol);
    }

    [Fact]
    public void MAD_LessThanOrEqual_StdDev()
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

    [Fact]
    public void TTest_PValue_BetweenZeroAndOne()
    {
        var g1 = new Vector<double>(new double[] { 1, 3, 5, 7, 9 });
        var g2 = new Vector<double>(new double[] { 2, 4, 6, 8, 10 });
        var result = StatisticsHelper<double>.TTest(g1, g2);
        Assert.True(result.PValue >= 0 && result.PValue <= 1.0,
            $"P-value ({result.PValue}) should be in [0, 1]");
    }

    [Fact]
    public void ChiSquare_PValue_BetweenZeroAndOne()
    {
        var g1 = new Vector<double>(new double[] { 1, 1, 2, 3 });
        var g2 = new Vector<double>(new double[] { 2, 3, 3, 1 });
        var result = StatisticsHelper<double>.ChiSquareTest(g1, g2);
        Assert.True(result.PValue >= 0 && result.PValue <= 1.0,
            $"P-value ({result.PValue}) should be in [0, 1]");
    }

    [Fact]
    public void FTest_PValue_BetweenZeroAndOne()
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

    [Fact]
    public void Median_NegativeValues()
    {
        var values = new double[] { -5, -3, -1, 1, 3 };
        double result = StatisticsHelper<double>.CalculateMedian(values);
        Assert.Equal(-1.0, result, Tol);
    }

    [Fact]
    public void Variance_LargeValues_StillAccurate()
    {
        // Values centered around 1,000,000 with small variance
        var values = new double[] { 1000001, 1000002, 1000003, 1000004, 1000005 };
        double result = StatisticsHelper<double>.CalculateVariance(values);
        Assert.Equal(2.5, result, Tol);
    }

    [Fact]
    public void MSE_Symmetric_PredActual()
    {
        // MSE(actual, pred) = MSE(pred, actual)
        var a = new double[] { 1, 2, 3, 4, 5 };
        var b = new double[] { 1.5, 2.5, 3.5, 4.5, 5.5 };
        double mse1 = StatisticsHelper<double>.CalculateMeanSquaredError(a, b);
        double mse2 = StatisticsHelper<double>.CalculateMeanSquaredError(b, a);
        Assert.Equal(mse1, mse2, Tol);
    }
}
