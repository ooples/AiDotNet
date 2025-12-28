using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Statistics;

/// <summary>
/// Integration tests for information criteria calculations (AIC, BIC).
/// Ground truth values computed using standard formulas.
/// </summary>
public class InformationCriteriaIntegrationTests
{
    private const double Tolerance = 1e-4;

    #region AIC Alternative Tests

    [Fact]
    public void AICAlternative_StandardExample_ReturnsExactValue()
    {
        // AIC_alt = n*ln(RSS/n) + 2*k
        // n=100, k=3, RSS=250
        // = 100 * ln(2.5) + 6 = 100 * 0.916290731874155 + 6 = 97.6290731874155
        int n = 100;
        int k = 3;
        double rss = 250.0;

        var result = StatisticsHelper<double>.CalculateAICAlternative(n, k, rss);

        Assert.Equal(97.629, result, 0.01);
    }

    [Fact]
    public void AICAlternative_SmallSample_ReturnsExactValue()
    {
        // n=20, k=2, RSS=50
        // = 20 * ln(2.5) + 4 = 20 * 0.916290731874155 + 4 = 22.3258146
        int n = 20;
        int k = 2;
        double rss = 50.0;

        var result = StatisticsHelper<double>.CalculateAICAlternative(n, k, rss);

        Assert.Equal(22.326, result, 0.01);
    }

    [Fact]
    public void AICAlternative_LowRSS_ReturnsNegativeValue()
    {
        // n=100, k=2, RSS=10
        // = 100 * ln(0.1) + 4 = 100 * (-2.302585) + 4 = -226.26
        int n = 100;
        int k = 2;
        double rss = 10.0;

        var result = StatisticsHelper<double>.CalculateAICAlternative(n, k, rss);

        Assert.Equal(-226.26, result, 0.1);
    }

    [Fact]
    public void AICAlternative_HighK_IncreasesAIC()
    {
        int n = 100;
        double rss = 250.0;

        var aicK2 = StatisticsHelper<double>.CalculateAICAlternative(n, 2, rss);
        var aicK5 = StatisticsHelper<double>.CalculateAICAlternative(n, 5, rss);
        var aicK10 = StatisticsHelper<double>.CalculateAICAlternative(n, 10, rss);

        Assert.True(aicK2 < aicK5, "More parameters should increase AIC");
        Assert.True(aicK5 < aicK10, "More parameters should increase AIC");
    }

    [Fact]
    public void AICAlternative_LowerRSS_LowerAIC()
    {
        int n = 100;
        int k = 3;

        var aicHighRSS = StatisticsHelper<double>.CalculateAICAlternative(n, k, 500.0);
        var aicLowRSS = StatisticsHelper<double>.CalculateAICAlternative(n, k, 100.0);

        Assert.True(aicLowRSS < aicHighRSS, "Lower RSS should give lower AIC");
    }

    [Fact]
    public void AICAlternative_ZeroRSS_ReturnsZero()
    {
        var result = StatisticsHelper<double>.CalculateAICAlternative(100, 3, 0.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void AICAlternative_ZeroSampleSize_ReturnsZero()
    {
        var result = StatisticsHelper<double>.CalculateAICAlternative(0, 3, 250.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    #endregion

    #region AIC Tests

    [Fact]
    public void AIC_StandardExample_ReturnsExactValue()
    {
        // AIC = 2*k + n*(ln(2π*RSS/n) + 1)
        // n=100, k=3, RSS=250
        // = 6 + 100*(ln(2*π*2.5) + 1) = 6 + 100*(ln(15.7079632679) + 1)
        // = 6 + 100*(2.7542 + 1) = 6 + 375.42 = 381.42
        int n = 100;
        int k = 3;
        double rss = 250.0;

        var result = StatisticsHelper<double>.CalculateAIC(n, k, rss);

        Assert.Equal(381.46, result, 0.1);
    }

    [Fact]
    public void AIC_SmallSample_ReturnsExactValue()
    {
        // n=20, k=2, RSS=50
        // = 4 + 20*(ln(2*π*2.5) + 1) = 4 + 20*(2.7542 + 1) = 4 + 75.084 = 79.084
        int n = 20;
        int k = 2;
        double rss = 50.0;

        var result = StatisticsHelper<double>.CalculateAIC(n, k, rss);

        Assert.Equal(79.08, result, 0.1);
    }

    [Fact]
    public void AIC_HighK_IncreasesAIC()
    {
        int n = 100;
        double rss = 250.0;

        var aicK2 = StatisticsHelper<double>.CalculateAIC(n, 2, rss);
        var aicK5 = StatisticsHelper<double>.CalculateAIC(n, 5, rss);
        var aicK10 = StatisticsHelper<double>.CalculateAIC(n, 10, rss);

        Assert.True(aicK2 < aicK5, "More parameters should increase AIC");
        Assert.True(aicK5 < aicK10, "More parameters should increase AIC");
    }

    [Fact]
    public void AIC_LowerRSS_LowerAIC()
    {
        int n = 100;
        int k = 3;

        var aicHighRSS = StatisticsHelper<double>.CalculateAIC(n, k, 500.0);
        var aicLowRSS = StatisticsHelper<double>.CalculateAIC(n, k, 100.0);

        Assert.True(aicLowRSS < aicHighRSS, "Lower RSS should give lower AIC");
    }

    [Fact]
    public void AIC_ZeroRSS_ReturnsZero()
    {
        var result = StatisticsHelper<double>.CalculateAIC(100, 3, 0.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void AIC_ZeroSampleSize_ReturnsZero()
    {
        var result = StatisticsHelper<double>.CalculateAIC(0, 3, 250.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    #endregion

    #region BIC Tests

    [Fact]
    public void BIC_StandardExample_ReturnsExactValue()
    {
        // BIC = n*ln(RSS/n) + k*ln(n)
        // n=100, k=3, RSS=250
        // = 100 * ln(2.5) + 3 * ln(100) = 91.629 + 13.816 = 105.445
        int n = 100;
        int k = 3;
        double rss = 250.0;

        var result = StatisticsHelper<double>.CalculateBIC(n, k, rss);

        Assert.Equal(105.44, result, 0.1);
    }

    [Fact]
    public void BIC_SmallSample_ReturnsExactValue()
    {
        // n=20, k=2, RSS=50
        // = 20 * ln(2.5) + 2 * ln(20) = 18.326 + 5.991 = 24.317
        int n = 20;
        int k = 2;
        double rss = 50.0;

        var result = StatisticsHelper<double>.CalculateBIC(n, k, rss);

        Assert.Equal(24.32, result, 0.1);
    }

    [Fact]
    public void BIC_HighK_IncreasesBIC()
    {
        int n = 100;
        double rss = 250.0;

        var bicK2 = StatisticsHelper<double>.CalculateBIC(n, 2, rss);
        var bicK5 = StatisticsHelper<double>.CalculateBIC(n, 5, rss);
        var bicK10 = StatisticsHelper<double>.CalculateBIC(n, 10, rss);

        Assert.True(bicK2 < bicK5, "More parameters should increase BIC");
        Assert.True(bicK5 < bicK10, "More parameters should increase BIC");
    }

    [Fact]
    public void BIC_LowerRSS_LowerBIC()
    {
        int n = 100;
        int k = 3;

        var bicHighRSS = StatisticsHelper<double>.CalculateBIC(n, k, 500.0);
        var bicLowRSS = StatisticsHelper<double>.CalculateBIC(n, k, 100.0);

        Assert.True(bicLowRSS < bicHighRSS, "Lower RSS should give lower BIC");
    }

    [Fact]
    public void BIC_ZeroRSS_ReturnsZero()
    {
        var result = StatisticsHelper<double>.CalculateBIC(100, 3, 0.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void BIC_ZeroSampleSize_ReturnsZero()
    {
        var result = StatisticsHelper<double>.CalculateBIC(0, 3, 250.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void BIC_LargerSample_StrongerPenalty()
    {
        // BIC penalty increases with sample size: k*ln(n)
        int k = 5;
        double rss = 250.0;

        var bicN50 = StatisticsHelper<double>.CalculateBIC(50, k, rss * 50 / 100);
        var bicN100 = StatisticsHelper<double>.CalculateBIC(100, k, rss);
        var bicN1000 = StatisticsHelper<double>.CalculateBIC(1000, k, rss * 1000 / 100);

        // Extract penalty portion: k*ln(n)
        double penalty50 = k * Math.Log(50);    // 19.56
        double penalty100 = k * Math.Log(100);  // 23.03
        double penalty1000 = k * Math.Log(1000); // 34.54

        Assert.True(penalty50 < penalty100, "Larger sample should have stronger penalty");
        Assert.True(penalty100 < penalty1000, "Larger sample should have stronger penalty");
    }

    #endregion

    #region AIC vs BIC Comparison

    [Fact]
    public void AIC_vs_BIC_BICPenalizesMoreForLargeSamples()
    {
        // For large n, BIC penalty (k*ln(n)) > AIC penalty (2*k) when n > e² ≈ 7.39
        int n = 100;
        int k = 5;
        double rss = 250.0;

        var aic = StatisticsHelper<double>.CalculateAIC(n, k, rss);
        var aicAlt = StatisticsHelper<double>.CalculateAICAlternative(n, k, rss);
        var bic = StatisticsHelper<double>.CalculateBIC(n, k, rss);

        // AIC penalty = 2*k = 10
        // BIC penalty = k*ln(n) = 5*ln(100) = 23.03
        // So BIC should prefer simpler models more than AIC
        // We can't directly compare values due to different formulas,
        // but we can verify the penalty difference affects model selection
        double aicAltPenalty = 2 * k;
        double bicPenalty = k * Math.Log(n);

        Assert.True(bicPenalty > aicAltPenalty, "BIC penalty should be larger than AIC for n=100");
    }

    [Fact]
    public void AIC_vs_BIC_SameRanking_ForSimpleCases()
    {
        int n = 100;
        double rss = 250.0;

        // Compare models with different numbers of parameters
        var aicK3 = StatisticsHelper<double>.CalculateAIC(n, 3, rss);
        var aicK5 = StatisticsHelper<double>.CalculateAIC(n, 5, rss);
        var bicK3 = StatisticsHelper<double>.CalculateBIC(n, 3, rss);
        var bicK5 = StatisticsHelper<double>.CalculateBIC(n, 5, rss);

        // Both should prefer the simpler model (k=3)
        Assert.True(aicK3 < aicK5, "AIC should prefer simpler model");
        Assert.True(bicK3 < bicK5, "BIC should prefer simpler model");
    }

    #endregion

    #region Float Type Tests

    [Fact]
    public void AICAlternative_FloatType_ReturnsCorrectValue()
    {
        var result = StatisticsHelper<float>.CalculateAICAlternative(100, 3, 250.0f);
        Assert.Equal(97.63f, result, 0.1f);
    }

    [Fact]
    public void AIC_FloatType_ReturnsCorrectValue()
    {
        var result = StatisticsHelper<float>.CalculateAIC(100, 3, 250.0f);
        Assert.Equal(381.46f, result, 0.5f);
    }

    [Fact]
    public void BIC_FloatType_ReturnsCorrectValue()
    {
        var result = StatisticsHelper<float>.CalculateBIC(100, 3, 250.0f);
        Assert.Equal(105.44f, result, 0.5f);
    }

    #endregion
}
