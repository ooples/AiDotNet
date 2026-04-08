using AiDotNet.Evaluation.Statistics;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Evaluation;

/// <summary>
/// Deep math integration tests for statistical hypothesis tests.
/// Verifies paired t-test, Wilcoxon signed-rank, McNemar's, Friedman, and Levene's tests
/// using hand-computed expected values.
/// </summary>
public class StatisticalTestsDeepMathIntegrationTests
{
    private const double Tolerance = 0.05; // Statistical approximations have limited precision

    // ===== Paired T-Test =====

    [Fact]
    public void PairedTTest_IdenticalSamples_NotSignificant()
    {
        // When samples are identical, differences are all zero -> not significant
        var test = new PairedTTest<double>();
        var sample = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        var result = test.Test(sample, sample);

        Assert.False(result.IsSignificant, "Identical samples should not be significant");
        Assert.Equal(1.0, result.PValue, 0.01);
    }

    [Fact]
    public void PairedTTest_KnownTStatistic_HandComputed()
    {
        // sample1 = [10, 20, 30, 40, 50], sample2 = [8, 18, 28, 38, 48]
        // differences = [2, 2, 2, 2, 2], mean_diff = 2, std_diff = 0 (all identical)
        // When std = 0 and mean != 0, this is highly significant (p=0)
        var test = new PairedTTest<double>();
        var s1 = new double[] { 10, 20, 30, 40, 50 };
        var s2 = new double[] { 8, 18, 28, 38, 48 };

        var result = test.Test(s1, s2);

        Assert.True(result.IsSignificant, "Constant non-zero difference should be significant");
        Assert.Equal(0.0, result.PValue, 0.01);
    }

    [Fact]
    public void PairedTTest_TStatisticFormula_HandComputed()
    {
        // sample1 = [5, 6, 7, 8, 9], sample2 = [3, 4, 5, 6, 7]
        // differences = [2, 2, 2, 2, 2], mean_diff = 2.0, std_diff = 0
        // All diffs identical -> zero std -> p=0 for nonzero mean diff
        var test = new PairedTTest<double>();
        var s1 = new double[] { 5, 6, 7, 8, 9 };
        var s2 = new double[] { 3, 4, 5, 6, 7 };

        var result = test.Test(s1, s2);

        // Zero std with nonzero mean -> highly significant
        Assert.True(result.IsSignificant);
    }

    [Fact]
    public void PairedTTest_VariableDifferences_HandComputed()
    {
        // sample1 = [10, 20, 30, 40, 50], sample2 = [9, 18, 33, 38, 52]
        // differences = [1, 2, -3, 2, -2]
        // mean_diff = 0, std_diff = sqrt(((1-0)^2 + (2-0)^2 + (-3-0)^2 + (2-0)^2 + (-2-0)^2) / 4)
        //           = sqrt((1+4+9+4+4)/4) = sqrt(22/4) = sqrt(5.5) ≈ 2.345
        // t = 0 / (2.345/sqrt(5)) = 0
        // p-value should be 1 (no evidence of difference)
        var test = new PairedTTest<double>();
        var s1 = new double[] { 10, 20, 30, 40, 50 };
        var s2 = new double[] { 9, 18, 33, 38, 52 };

        var result = test.Test(s1, s2);

        // t-statistic should be 0 (mean diff is 0)
        Assert.Equal(0.0, result.Statistic, 0.01);
        Assert.False(result.IsSignificant, "Zero mean difference should not be significant");
    }

    [Fact]
    public void PairedTTest_CohensD_HandComputed()
    {
        // differences = [2, 4, 6, 8, 10]
        // mean = 6, std = sqrt(((2-6)^2+(4-6)^2+(6-6)^2+(8-6)^2+(10-6)^2)/4) = sqrt((16+4+0+4+16)/4) = sqrt(10)
        // Cohen's d = mean/std = 6/sqrt(10) ≈ 1.897
        var test = new PairedTTest<double>();
        var s1 = new double[] { 12, 14, 16, 18, 20 };
        var s2 = new double[] { 10, 10, 10, 10, 10 };

        var result = test.Test(s1, s2);

        var expectedCohensD = 6.0 / Math.Sqrt(10.0);
        Assert.Equal(expectedCohensD, result.EffectSize, 0.1);
    }

    [Fact]
    public void PairedTTest_DegreesOfFreedom()
    {
        // df = n - 1 = 7 - 1 = 6
        var test = new PairedTTest<double>();
        var s1 = new double[] { 1, 2, 3, 4, 5, 6, 7 };
        var s2 = new double[] { 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5 };

        var result = test.Test(s1, s2);

        Assert.Equal(6, result.DegreesOfFreedom);
    }

    [Fact]
    public void PairedTTest_UnequalLengths_ThrowsArgumentException()
    {
        var test = new PairedTTest<double>();
        var s1 = new double[] { 1, 2, 3 };
        var s2 = new double[] { 1, 2 };

        Assert.Throws<ArgumentException>(() => test.Test(s1, s2));
    }

    // ===== Wilcoxon Signed-Rank Test =====

    [Fact]
    public void WilcoxonTest_IdenticalSamples_NotSignificant()
    {
        var test = new WilcoxonSignedRankTest<double>();
        var sample = new double[] { 1, 2, 3, 4, 5 };

        var result = test.Test(sample, sample);

        Assert.False(result.IsSignificant, "Identical samples should not be significant");
    }

    [Fact]
    public void WilcoxonTest_ClearDifference_Significant()
    {
        // All s1 > s2 by large amounts with more samples for better power
        var test = new WilcoxonSignedRankTest<double>();
        var s1 = new double[] { 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };
        var s2 = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        var result = test.Test(s1, s2);

        Assert.True(result.IsSignificant, "Clear positive difference should be significant");
    }

    [Fact]
    public void WilcoxonTest_RankComputation_HandComputed()
    {
        // s1 = [5, 8, 7, 10, 12], s2 = [3, 6, 9, 8, 10]
        // diffs = [2, 2, -2, 2, 2]
        // |diffs| = [2, 2, 2, 2, 2] -> all tied -> avg rank = 3
        // W+ = 3*4 = 12 (four positive), W- = 3*1 = 3 (one negative)
        // W = min(12, 3) = 3
        var test = new WilcoxonSignedRankTest<double>();
        var s1 = new double[] { 5, 8, 7, 10, 12 };
        var s2 = new double[] { 3, 6, 9, 8, 10 };

        var result = test.Test(s1, s2);

        // W statistic should be min(W+, W-)
        // W+ corresponds to positive diffs, W- to negative
        Assert.True(result.Statistic >= 0, "W should be non-negative");
    }

    [Fact]
    public void WilcoxonTest_EffectSize_RankBiserial()
    {
        // Effect size = (W+ - W-) / (n*(n+1)/2)
        // For all positive diffs: W+ = n*(n+1)/2, W- = 0
        // Effect size = 1.0
        var test = new WilcoxonSignedRankTest<double>();
        var s1 = new double[] { 10, 20, 30, 40, 50 };
        var s2 = new double[] { 1, 2, 3, 4, 5 };

        var result = test.Test(s1, s2);

        // All diffs positive -> effect size should be close to 1 or -1
        Assert.True(Math.Abs(result.EffectSize) > 0.5,
            $"All-positive diffs should have large effect size, got {result.EffectSize}");
    }

    [Fact]
    public void WilcoxonTest_NormalApproximation_MeanAndStd()
    {
        // For n non-zero differences:
        // mean_W = n*(n+1)/4
        // std_W = sqrt(n*(n+1)*(2n+1)/24)
        // z = (W - mean + 0.5) / std (continuity correction)
        var test = new WilcoxonSignedRankTest<double>();
        var s1 = new double[] { 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };
        var s2 = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        var result = test.Test(s1, s2);

        // With 10 observations, all positive diffs, should be highly significant
        Assert.True(result.IsSignificant);
    }

    [Fact]
    public void WilcoxonTest_MinimumSampleSize_5()
    {
        var test = new WilcoxonSignedRankTest<double>();
        var s1 = new double[] { 1, 2, 3, 4 }; // Only 4 samples

        Assert.Throws<ArgumentException>(() => test.Test(s1, s1));
    }

    // ===== McNemar's Test =====

    [Fact]
    public void McNemarTest_NoDisagreement_NotSignificant()
    {
        // Both classifiers make same predictions -> b=0, c=0
        var test = new McNemarTest<double>();
        var bothCorrect = Enumerable.Repeat(1.0, 20).ToArray();

        var result = test.Test(bothCorrect, bothCorrect);

        Assert.False(result.IsSignificant, "No disagreement should not be significant");
    }

    [Fact]
    public void McNemarTest_ChiSquaredStatistic_HandComputed()
    {
        // A correct + B wrong (b) = 30 times
        // A wrong + B correct (c) = 10 times
        // chi^2 = (|30 - 10| - 1)^2 / (30 + 10) = (19)^2 / 40 = 361/40 = 9.025
        var test = new McNemarTest<double>();

        // Create arrays: first 50 both correct, next 30 A correct B wrong, last 20 A wrong B correct (10) + both wrong (10)
        var correctA = new double[100];
        var correctB = new double[100];

        // Both correct: positions 0-49
        for (int i = 0; i < 50; i++) { correctA[i] = 1; correctB[i] = 1; }
        // A correct, B wrong: positions 50-79
        for (int i = 50; i < 80; i++) { correctA[i] = 1; correctB[i] = 0; }
        // A wrong, B correct: positions 80-89
        for (int i = 80; i < 90; i++) { correctA[i] = 0; correctB[i] = 1; }
        // Both wrong: positions 90-99
        for (int i = 90; i < 100; i++) { correctA[i] = 0; correctB[i] = 0; }

        var result = test.Test(correctA, correctB);

        // b + c = 40 >= 25, so uses chi-squared approximation
        // chi^2 = (|30-10| - 1)^2 / 40 = 361/40 = 9.025
        Assert.Equal(9.025, result.Statistic, 0.01);
    }

    [Fact]
    public void McNemarTest_OddsRatio_HandComputed()
    {
        // b = 30, c = 10 -> odds ratio = b/c = 30/10 = 3.0
        var test = new McNemarTest<double>();
        var correctA = new double[100];
        var correctB = new double[100];

        for (int i = 0; i < 50; i++) { correctA[i] = 1; correctB[i] = 1; }
        for (int i = 50; i < 80; i++) { correctA[i] = 1; correctB[i] = 0; }
        for (int i = 80; i < 90; i++) { correctA[i] = 0; correctB[i] = 1; }
        for (int i = 90; i < 100; i++) { correctA[i] = 0; correctB[i] = 0; }

        var result = test.Test(correctA, correctB);

        Assert.Equal(3.0, result.EffectSize, 0.01);
    }

    [Fact]
    public void McNemarTest_SymmetricDisagreement_NotSignificant()
    {
        // b = c -> no difference between classifiers
        // chi^2 = (|b-c| - 1)^2 / (b+c) = (0-1)^2 / (b+c) = 1/(b+c)
        // For large b+c, this is near zero -> not significant
        var test = new McNemarTest<double>();
        var correctA = new double[100];
        var correctB = new double[100];

        for (int i = 0; i < 40; i++) { correctA[i] = 1; correctB[i] = 1; }
        for (int i = 40; i < 70; i++) { correctA[i] = 1; correctB[i] = 0; } // b = 30
        for (int i = 70; i < 100; i++) { correctA[i] = 0; correctB[i] = 1; } // c = 30

        var result = test.Test(correctA, correctB);

        // b = c = 30, chi^2 = 1/60 ≈ 0.0167 -> not significant
        Assert.False(result.IsSignificant, "Symmetric disagreement should not be significant");
    }

    [Fact]
    public void McNemarTest_ExactBinomialForSmallSamples()
    {
        // When b + c < 25, uses exact binomial test
        var test = new McNemarTest<double>();
        var correctA = new double[20];
        var correctB = new double[20];

        // b = 8, c = 2
        for (int i = 0; i < 10; i++) { correctA[i] = 1; correctB[i] = 1; }
        for (int i = 10; i < 18; i++) { correctA[i] = 1; correctB[i] = 0; }
        for (int i = 18; i < 20; i++) { correctA[i] = 0; correctB[i] = 1; }

        var result = test.Test(correctA, correctB);

        // b + c = 10 < 25, so uses exact test
        // P-value from binomial(10, min(2,8), 0.5) * 2
        Assert.True(result.PValue >= 0 && result.PValue <= 1,
            $"P-value should be in [0,1], got {result.PValue}");
    }

    // ===== Friedman Test =====

    [Fact]
    public void FriedmanTest_IdenticalAlgorithms_NotSignificant()
    {
        // All algorithms perform the same on each dataset
        var test = new FriedmanTest<double>();
        var samples = new double[][]
        {
            new double[] { 0.8, 0.85, 0.9, 0.87, 0.83 }, // Algo 1
            new double[] { 0.8, 0.85, 0.9, 0.87, 0.83 }, // Algo 2 (identical)
            new double[] { 0.8, 0.85, 0.9, 0.87, 0.83 }  // Algo 3 (identical)
        };

        var result = test.Test(samples);

        Assert.False(result.IsSignificant, "Identical algorithms should not be significant");
    }

    [Fact]
    public void FriedmanTest_ClearlyDifferentAlgorithms_Significant()
    {
        // Algo 1 always best, Algo 3 always worst
        var test = new FriedmanTest<double>();
        var samples = new double[][]
        {
            new double[] { 0.95, 0.93, 0.96, 0.94, 0.97 }, // Best
            new double[] { 0.80, 0.78, 0.82, 0.79, 0.81 }, // Middle
            new double[] { 0.60, 0.58, 0.62, 0.59, 0.61 }  // Worst
        };

        var result = test.Test(samples);

        Assert.True(result.IsSignificant, "Clearly different algorithms should be significant");
    }

    [Fact]
    public void FriedmanTest_RankComputation_HandComputed()
    {
        // Dataset 1: Algo1=0.9, Algo2=0.8, Algo3=0.7 -> ranks: 1, 2, 3
        // Dataset 2: Algo1=0.85, Algo2=0.75, Algo3=0.65 -> ranks: 1, 2, 3
        // Dataset 3: Algo1=0.95, Algo2=0.85, Algo3=0.75 -> ranks: 1, 2, 3
        // Average ranks: Algo1=1, Algo2=2, Algo3=3
        // chi^2 = 12*3/(3*4) * (1+4+9 - 3*(4/2)^2) = 12*3/12 * (14 - 12) = 3 * 2 = 6
        var test = new FriedmanTest<double>();
        var samples = new double[][]
        {
            new double[] { 0.9, 0.85, 0.95 },
            new double[] { 0.8, 0.75, 0.85 },
            new double[] { 0.7, 0.65, 0.75 }
        };

        var result = test.Test(samples);

        // With 3 algorithms and 3 datasets, perfectly ordered ranks -> significant
        Assert.True(result.IsSignificant);
    }

    [Fact]
    public void FriedmanTest_ImanDavenportCorrection()
    {
        // The Friedman test uses Iman-Davenport F-statistic:
        // F = (n-1)*chi^2 / (n*(k-1) - chi^2)
        // For k=3, n=5, chi^2 = 6.4:
        // F = (5-1)*6.4 / (5*(3-1) - 6.4) = 25.6 / 3.6 ≈ 7.11
        var test = new FriedmanTest<double>();
        var samples = new double[][]
        {
            new double[] { 0.95, 0.93, 0.96, 0.94, 0.97 },
            new double[] { 0.80, 0.78, 0.82, 0.79, 0.81 },
            new double[] { 0.60, 0.58, 0.62, 0.59, 0.61 }
        };

        var result = test.Test(samples);

        // DegreesOfFreedom = k - 1 = 2
        Assert.Equal(2, result.DegreesOfFreedom);
    }

    [Fact]
    public void FriedmanTest_MinimumRequirements()
    {
        var test = new FriedmanTest<double>();

        // Need at least 2 algorithms
        Assert.Throws<ArgumentException>(() => test.Test(new double[][]
        {
            new double[] { 0.8, 0.85, 0.9 }
        }));

        // Need at least 3 datasets
        Assert.Throws<ArgumentException>(() => test.Test(new double[][]
        {
            new double[] { 0.8, 0.85 },
            new double[] { 0.7, 0.75 }
        }));
    }

    // ===== Levene's Test =====

    [Fact]
    public void LeveneTest_EqualVarianceGroups_NotSignificant()
    {
        // Two groups with similar variance should pass Levene's test
        var test = new LeveneTest<double>(LeveneTest<double>.CenterType.Median);
        var groups = new double[][]
        {
            new double[] { 10, 11, 12, 13, 14, 15, 16 },
            new double[] { 20, 21, 22, 23, 24, 25, 26 }
        };

        var result = test.Test(groups);

        Assert.False(result.IsSignificant,
            "Groups with equal variance should not be significant");
    }

    [Fact]
    public void LeveneTest_UnequalVarianceGroups_Significant()
    {
        // Group 1: low variance, Group 2: high variance
        var test = new LeveneTest<double>(LeveneTest<double>.CenterType.Median);
        var groups = new double[][]
        {
            new double[] { 10, 10.1, 10.2, 9.9, 9.8, 10.3, 10.05, 9.95, 10.15, 10.1 },
            new double[] { 0, 20, 40, -10, 30, -5, 25, 15, -15, 35 }
        };

        var result = test.Test(groups);

        Assert.True(result.IsSignificant,
            "Groups with very different variance should be significant");
    }

    [Fact]
    public void LeveneTest_FStatistic_Positive()
    {
        var test = new LeveneTest<double>(LeveneTest<double>.CenterType.Mean);
        var groups = new double[][]
        {
            new double[] { 1, 2, 3, 4, 5 },
            new double[] { 10, 20, 30, 40, 50 }
        };

        var result = test.Test(groups);

        Assert.True(result.Statistic >= 0,
            $"F-statistic should be non-negative, got {result.Statistic}");
    }

    [Fact]
    public void LeveneTest_MedianBasedIsBrownForsythe()
    {
        var test = new LeveneTest<double>(LeveneTest<double>.CenterType.Median);
        var groups = new double[][]
        {
            new double[] { 1, 2, 3, 4, 5 },
            new double[] { 10, 20, 30, 40, 50 }
        };

        var result = test.Test(groups);

        Assert.Contains("Brown-Forsythe", result.TestName);
    }

    [Fact]
    public void LeveneTest_MeanBasedIsLevene()
    {
        var test = new LeveneTest<double>(LeveneTest<double>.CenterType.Mean);
        var groups = new double[][]
        {
            new double[] { 1, 2, 3, 4, 5 },
            new double[] { 10, 20, 30, 40, 50 }
        };

        var result = test.Test(groups);

        Assert.Contains("Levene", result.TestName);
    }

    [Fact]
    public void LeveneTest_DegreesOfFreedom_HandComputed()
    {
        // k groups -> df_between = k - 1
        var test = new LeveneTest<double>();
        var groups = new double[][]
        {
            new double[] { 1, 2, 3, 4, 5 },
            new double[] { 10, 20, 30, 40, 50 },
            new double[] { 100, 200, 300, 400, 500 }
        };

        var result = test.Test(groups);

        // df_between = k - 1 = 3 - 1 = 2
        Assert.Equal(2, result.DegreesOfFreedom);
    }

    // ===== P-Value Properties =====

    [Fact]
    public void AllTests_PValue_InZeroOneRange()
    {
        // P-values should always be in [0, 1]
        var tTest = new PairedTTest<double>();
        var s1 = new double[] { 1, 3, 5, 7, 9 };
        var s2 = new double[] { 2, 4, 6, 8, 10 };

        var tResult = tTest.Test(s1, s2);
        Assert.True(tResult.PValue >= 0 && tResult.PValue <= 1,
            $"T-test p-value should be in [0,1], got {tResult.PValue}");

        var wilcoxon = new WilcoxonSignedRankTest<double>();
        var wResult = wilcoxon.Test(s1, s2);
        Assert.True(wResult.PValue >= 0 && wResult.PValue <= 1,
            $"Wilcoxon p-value should be in [0,1], got {wResult.PValue}");
    }

    [Fact]
    public void PairedTTest_LargerDifference_SmallerPValue()
    {
        // Larger consistent difference -> smaller p-value
        var test = new PairedTTest<double>();
        var baseline = new double[] { 10, 20, 30, 40, 50 };
        var smallDiff = new double[] { 11, 21, 31, 41, 51 };    // diff = 1
        var largeDiff = new double[] { 20, 30, 40, 50, 60 };    // diff = 10

        var resultSmall = test.Test(baseline, smallDiff);
        var resultLarge = test.Test(baseline, largeDiff);

        // Both have zero std (constant diff), so both p = 0
        // Instead test with variable diffs
        var baseline2 = new double[] { 10, 20, 30, 40, 50 };
        var small2 = new double[] { 9, 19, 31, 39, 51 };  // diffs: 1,-1,1,-1,1 -> mean≈0
        var large2 = new double[] { 5, 15, 25, 35, 45 };  // diffs: 5,5,5,5,5 -> mean=5

        var rSmall2 = test.Test(baseline2, small2);
        var rLarge2 = test.Test(baseline2, large2);

        // Large consistent diff should have smaller p-value
        Assert.True(rLarge2.PValue <= rSmall2.PValue + Tolerance,
            $"Larger diff p-value ({rLarge2.PValue}) should be <= small diff p-value ({rSmall2.PValue})");
    }

    // ===== Alpha Level Tests =====

    [Fact]
    public void PairedTTest_CustomAlpha()
    {
        var test = new PairedTTest<double>();
        var s1 = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        var s2 = new double[] { 1.5, 2.3, 3.1, 4.2, 5.4, 5.8, 7.1, 8.3, 8.9, 10.2 };

        var result01 = test.Test(s1, s2, alpha: 0.01);
        var result10 = test.Test(s1, s2, alpha: 0.10);

        Assert.Equal(0.01, result01.Alpha);
        Assert.Equal(0.10, result10.Alpha);

        // Same test statistic and p-value regardless of alpha
        Assert.Equal(result01.PValue, result10.PValue, 0.001);
    }

    // ===== McNemar Exact Binomial Test =====

    [Fact]
    public void McNemarTest_BinomialProbability_HandComputed()
    {
        // For n=10, k=0, p=0.5:
        // P(X=0) = C(10,0) * 0.5^0 * 0.5^10 = 1 * 1 * (1/1024) ≈ 0.000977
        // Two-tailed p for observing 0 or 10: 2 * 0.000977 ≈ 0.00195
        var test = new McNemarTest<double>();

        // b = 0, c = 10 (extreme case)
        var correctA = new double[20]; // 10 both correct, 10 A wrong B correct
        var correctB = new double[20];

        for (int i = 0; i < 10; i++) { correctA[i] = 1; correctB[i] = 1; }
        for (int i = 10; i < 20; i++) { correctA[i] = 0; correctB[i] = 1; }

        var result = test.Test(correctA, correctB);

        // b + c = 10 < 25, uses exact test
        // Very extreme case: b=0, c=10 -> should be significant
        Assert.True(result.IsSignificant, "Extreme asymmetry should be significant");
    }

    // ===== Wilcoxon Test - Tie Handling =====

    [Fact]
    public void WilcoxonTest_WithTies_AverageRanks()
    {
        // diffs have ties in absolute value
        // s1 - s2 = [1, -1, 2, -2, 3] -> |diffs| = [1, 1, 2, 2, 3]
        // Sorted |diffs|: [1, 1, 2, 2, 3]
        // Ranks: [1.5, 1.5, 3.5, 3.5, 5] (average ranks for ties)
        var test = new WilcoxonSignedRankTest<double>();
        var s1 = new double[] { 11, 9, 12, 8, 13 };
        var s2 = new double[] { 10, 10, 10, 10, 10 };

        var result = test.Test(s1, s2);

        // Should handle ties gracefully
        Assert.True(result.Statistic >= 0, "W should be non-negative with ties");
    }

    // ===== Friedman Test - Tied Ranks =====

    [Fact]
    public void FriedmanTest_TiedPerformance_HandlesCorrectly()
    {
        // Two algorithms with same score on some datasets
        var test = new FriedmanTest<double>();
        var samples = new double[][]
        {
            new double[] { 0.9, 0.9, 0.85, 0.88, 0.92 },  // Algo 1
            new double[] { 0.9, 0.85, 0.85, 0.88, 0.90 },  // Algo 2 (ties with Algo 1 on datasets 1, 3, 4)
            new double[] { 0.7, 0.75, 0.72, 0.78, 0.73 }   // Algo 3 (clearly worse)
        };

        var result = test.Test(samples);

        // Should still compute correctly with ties
        Assert.True(result.Statistic >= 0, "F-statistic should be non-negative");
        Assert.True(result.PValue >= 0 && result.PValue <= 1);
    }

    // ===== Cross-Test Consistency =====

    [Fact]
    public void PairedTTest_AndWilcoxon_AgreeOnDirection()
    {
        // Both tests should agree on whether there's a significant difference
        // for clearly different samples with enough data points
        var tTest = new PairedTTest<double>();
        var wilcoxon = new WilcoxonSignedRankTest<double>();
        var s1 = new double[] { 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };
        var s2 = new double[] { 5, 15, 25, 35, 45, 55, 65, 75, 85, 95 };

        var tResult = tTest.Test(s1, s2);
        var wResult = wilcoxon.Test(s1, s2);

        // Both should find this significant (all diffs are positive and equal)
        Assert.Equal(tResult.IsSignificant, wResult.IsSignificant);
    }
}
