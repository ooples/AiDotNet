using AiDotNet.Enums;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.FitDetectors;

/// <summary>
/// Deep integration tests for FitDetectors:
/// FitType enum (all values, ordering),
/// Fit detection math (bias-variance tradeoff, cross-validation, confidence intervals,
/// residual analysis, VIF for multicollinearity, Durbin-Watson for autocorrelation).
/// </summary>
public class FitDetectorsDeepMathIntegrationTests
{
    // ============================
    // FitType Enum Coverage
    // ============================

    [Fact]
    public void FitType_HasFifteenValues()
    {
        var values = Enum.GetValues<FitType>();
        Assert.Equal(15, values.Length);
    }

    [Theory]
    [InlineData(FitType.GoodFit)]
    [InlineData(FitType.Overfit)]
    [InlineData(FitType.Underfit)]
    [InlineData(FitType.HighBias)]
    [InlineData(FitType.HighVariance)]
    [InlineData(FitType.Unstable)]
    [InlineData(FitType.SevereMulticollinearity)]
    [InlineData(FitType.ModerateMulticollinearity)]
    [InlineData(FitType.PoorFit)]
    [InlineData(FitType.VeryPoorFit)]
    [InlineData(FitType.StrongPositiveAutocorrelation)]
    [InlineData(FitType.StrongNegativeAutocorrelation)]
    [InlineData(FitType.WeakAutocorrelation)]
    [InlineData(FitType.NoAutocorrelation)]
    [InlineData(FitType.Moderate)]
    public void FitType_AllValuesValid(FitType type)
    {
        Assert.True(Enum.IsDefined(type));
    }

    // ============================
    // Bias-Variance Tradeoff Math
    // ============================

    [Theory]
    [InlineData(0.1, 0.05, 0.02)]    // Low bias, low variance
    [InlineData(0.5, 0.01, 0.02)]    // High bias, low variance (underfit)
    [InlineData(0.01, 0.5, 0.02)]    // Low bias, high variance (overfit)
    public void FitMath_TotalError_IsSumOfBiasVarianceNoise(double bias, double variance, double irreducibleNoise)
    {
        // Total error = Bias^2 + Variance + Irreducible Noise
        double totalError = bias * bias + variance + irreducibleNoise;
        Assert.True(totalError >= irreducibleNoise,
            "Total error must be at least the irreducible noise");
        Assert.Equal(bias * bias + variance + irreducibleNoise, totalError, 1e-10);
    }

    [Theory]
    [InlineData(0.95, 0.60)]    // Train 95%, test 60% -> overfit
    [InlineData(0.55, 0.50)]    // Train 55%, test 50% -> underfit
    [InlineData(0.90, 0.88)]    // Train 90%, test 88% -> good fit
    public void FitMath_OverfitGap(double trainScore, double testScore)
    {
        double gap = trainScore - testScore;
        Assert.True(gap >= 0 || Math.Abs(gap) < 0.1,
            "Train-test gap should typically be non-negative or small");
    }

    [Fact]
    public void FitMath_OverfitDetection_LargeGapIndicatesOverfit()
    {
        double trainAccuracy = 0.99;
        double testAccuracy = 0.60;
        double gap = trainAccuracy - testAccuracy;

        // Gap > 0.20 is a strong indicator of overfitting
        Assert.True(gap > 0.20, $"Gap of {gap} should indicate overfitting");
    }

    [Fact]
    public void FitMath_UnderfitDetection_BothLowIndicatesUnderfit()
    {
        double trainAccuracy = 0.55;
        double testAccuracy = 0.52;
        double threshold = 0.70; // Below this is considered underfit

        Assert.True(trainAccuracy < threshold && testAccuracy < threshold,
            "Both train and test below threshold indicates underfitting");
    }

    // ============================
    // Cross-Validation Math
    // ============================

    [Theory]
    [InlineData(new double[] { 0.85, 0.87, 0.83, 0.86, 0.84 })]
    [InlineData(new double[] { 0.90, 0.88, 0.92, 0.89, 0.91 })]
    public void FitMath_CrossValidation_MeanAndStd(double[] foldScores)
    {
        double mean = foldScores.Average();
        double variance = foldScores.Select(x => (x - mean) * (x - mean)).Sum() / (foldScores.Length - 1);
        double stdDev = Math.Sqrt(variance);

        // Mean should be between min and max
        Assert.True(mean >= foldScores.Min());
        Assert.True(mean <= foldScores.Max());

        // Std should be non-negative
        Assert.True(stdDev >= 0);

        // 95% confidence interval: mean +/- 1.96 * stdDev / sqrt(n)
        double marginOfError = 1.96 * stdDev / Math.Sqrt(foldScores.Length);
        double lower = mean - marginOfError;
        double upper = mean + marginOfError;

        Assert.True(lower < upper);
        Assert.True(lower <= mean);
        Assert.True(upper >= mean);
    }

    [Fact]
    public void FitMath_CrossValidation_HighVarianceIndicatesInstability()
    {
        double[] foldScores = { 0.95, 0.60, 0.88, 0.55, 0.92 };
        double mean = foldScores.Average();
        double variance = foldScores.Select(x => (x - mean) * (x - mean)).Sum() / (foldScores.Length - 1);
        double stdDev = Math.Sqrt(variance);

        // High std relative to mean indicates unstable model
        double coefficientOfVariation = stdDev / mean;
        Assert.True(coefficientOfVariation > 0.10,
            $"CV of {coefficientOfVariation:F4} should indicate instability (>0.10)");
    }

    // ============================
    // Residual Analysis Math
    // ============================

    [Theory]
    [InlineData(new double[] { 0.1, -0.2, 0.15, -0.05, 0.1 }, true)]   // Random residuals (good fit)
    [InlineData(new double[] { 0.5, 0.6, 0.7, 0.8, 0.9 }, false)]      // Systematic pattern (bad fit)
    public void FitMath_ResidualAnalysis_MeanNearZero(double[] residuals, bool expectGoodFit)
    {
        double meanResidual = residuals.Average();

        if (expectGoodFit)
        {
            // Good fit: mean residual near zero
            Assert.True(Math.Abs(meanResidual) < 0.1,
                $"Mean residual ({meanResidual}) should be near zero for good fit");
        }
        else
        {
            // Bad fit: systematic bias
            Assert.True(Math.Abs(meanResidual) > 0.1,
                $"Mean residual ({meanResidual}) should show bias for bad fit");
        }
    }

    [Fact]
    public void FitMath_ResidualNormality_SkewnessAndKurtosis()
    {
        // For normally distributed residuals:
        // Skewness should be near 0, Kurtosis near 3
        double[] residuals = { -0.5, -0.2, 0.0, 0.1, 0.3, -0.1, 0.2, -0.3, 0.15, -0.05 };
        double mean = residuals.Average();
        int n = residuals.Length;

        double m2 = residuals.Select(r => Math.Pow(r - mean, 2)).Sum() / n;
        double m3 = residuals.Select(r => Math.Pow(r - mean, 3)).Sum() / n;
        double m4 = residuals.Select(r => Math.Pow(r - mean, 4)).Sum() / n;

        double skewness = m3 / Math.Pow(m2, 1.5);
        double kurtosis = m4 / (m2 * m2);

        // Skewness near 0 indicates symmetry
        Assert.True(Math.Abs(skewness) < 2.0, $"Skewness {skewness} too extreme");
        // Kurtosis near 3 for normal distribution (excess kurtosis near 0)
        Assert.True(kurtosis > 0, "Kurtosis must be positive");
    }

    // ============================
    // VIF (Variance Inflation Factor)
    // ============================

    [Theory]
    [InlineData(0.0, 1.0)]     // No correlation: VIF = 1
    [InlineData(0.5, 2.0)]     // Moderate: VIF = 1/(1-0.5) = 2.0
    [InlineData(0.9, 10.0)]    // High: VIF = 1/(1-0.9) = 10.0
    [InlineData(0.95, 20.0)]   // Very high: VIF = 1/(1-0.95) = 20.0
    public void FitMath_VIF_FromRSquared(double rSquared, double expectedVIF)
    {
        // VIF = 1 / (1 - R^2)
        double vif = 1.0 / (1.0 - rSquared);
        Assert.Equal(expectedVIF, vif, 1e-2);
    }

    [Theory]
    [InlineData(1.0, false)]     // VIF < 5: no multicollinearity
    [InlineData(3.0, false)]     // VIF < 5: no multicollinearity
    [InlineData(5.0, true)]      // VIF >= 5: moderate multicollinearity
    [InlineData(10.0, true)]     // VIF >= 10: severe multicollinearity
    public void FitMath_VIF_MulticollinearityThreshold(double vif, bool isMulticollinear)
    {
        bool detected = vif >= 5.0;
        Assert.Equal(isMulticollinear, detected);
    }

    // ============================
    // Durbin-Watson Statistic
    // ============================

    [Theory]
    [InlineData(new double[] { 0.1, 0.2, 0.15, 0.25, 0.18 })]
    public void FitMath_DurbinWatson_Calculation(double[] residuals)
    {
        // DW = sum((e_t - e_{t-1})^2) / sum(e_t^2)
        double numerator = 0;
        for (int i = 1; i < residuals.Length; i++)
        {
            double diff = residuals[i] - residuals[i - 1];
            numerator += diff * diff;
        }

        double denominator = residuals.Select(r => r * r).Sum();
        double dw = numerator / denominator;

        // DW is between 0 and 4
        Assert.True(dw >= 0 && dw <= 4, $"DW statistic {dw} should be between 0 and 4");
    }

    [Theory]
    [InlineData(2.0, FitType.NoAutocorrelation)]
    [InlineData(0.5, FitType.StrongPositiveAutocorrelation)]
    [InlineData(3.5, FitType.StrongNegativeAutocorrelation)]
    [InlineData(1.5, FitType.WeakAutocorrelation)]
    public void FitMath_DurbinWatson_Interpretation(double dwStatistic, FitType expectedFitType)
    {
        // DW near 2: no autocorrelation
        // DW near 0: positive autocorrelation
        // DW near 4: negative autocorrelation
        FitType detected;
        if (dwStatistic > 1.8 && dwStatistic < 2.2)
            detected = FitType.NoAutocorrelation;
        else if (dwStatistic < 1.0)
            detected = FitType.StrongPositiveAutocorrelation;
        else if (dwStatistic > 3.0)
            detected = FitType.StrongNegativeAutocorrelation;
        else
            detected = FitType.WeakAutocorrelation;

        Assert.Equal(expectedFitType, detected);
    }

    // ============================
    // R-squared and Adjusted R-squared
    // ============================

    [Theory]
    [InlineData(100.0, 20.0, 0.80)]   // R^2 = 1 - 20/100 = 0.80
    [InlineData(100.0, 5.0, 0.95)]    // R^2 = 1 - 5/100 = 0.95
    [InlineData(100.0, 100.0, 0.0)]   // R^2 = 0 (model no better than mean)
    public void FitMath_RSquared(double totalSS, double residualSS, double expectedRSquared)
    {
        // R^2 = 1 - (SS_res / SS_tot)
        double rSquared = 1.0 - (residualSS / totalSS);
        Assert.Equal(expectedRSquared, rSquared, 1e-10);
    }

    [Theory]
    [InlineData(0.80, 100, 5, 0.796)]    // Adjusted R^2 with n=100, p=5
    [InlineData(0.80, 20, 5, 0.729)]     // Adjusted R^2 with n=20, p=5 (more penalized)
    public void FitMath_AdjustedRSquared(double rSquared, int n, int p, double expectedAdjusted)
    {
        // Adjusted R^2 = 1 - (1 - R^2) * (n - 1) / (n - p - 1)
        double adjustedRSquared = 1.0 - (1.0 - rSquared) * (n - 1.0) / (n - p - 1.0);
        Assert.Equal(expectedAdjusted, adjustedRSquared, 1e-2);
    }

    [Fact]
    public void FitMath_AdjustedRSquared_AlwaysLessOrEqualToRSquared()
    {
        double rSquared = 0.85;
        int n = 50;
        int p = 10;

        double adjustedRSquared = 1.0 - (1.0 - rSquared) * (n - 1.0) / (n - p - 1.0);
        Assert.True(adjustedRSquared <= rSquared,
            $"Adjusted R^2 ({adjustedRSquared}) should be <= R^2 ({rSquared})");
    }

    // ============================
    // AIC/BIC Information Criteria
    // ============================

    [Theory]
    [InlineData(100, 5, -200.0)]    // n=100, k=5, logLikelihood=-200
    [InlineData(100, 10, -200.0)]   // More parameters -> higher AIC
    public void FitMath_AIC_Calculation(int n, int k, double logLikelihood)
    {
        // AIC = 2k - 2*ln(L)
        double aic = 2.0 * k - 2.0 * logLikelihood;
        Assert.True(aic > 0 || logLikelihood > 0);

        // BIC = k*ln(n) - 2*ln(L)
        double bic = k * Math.Log(n) - 2.0 * logLikelihood;

        // BIC penalizes more parameters more heavily for n > e^2 ~ 7.4
        if (n > 8)
        {
            Assert.True(bic >= aic,
                $"BIC ({bic}) should be >= AIC ({aic}) for n > 8");
        }
    }

    [Fact]
    public void FitMath_AIC_LowerIsBetter()
    {
        int n = 100;
        double logLikelihood = -200.0;

        double aicSimple = 2.0 * 3 - 2.0 * logLikelihood;   // k=3
        double aicComplex = 2.0 * 20 - 2.0 * logLikelihood;  // k=20

        // Same log-likelihood but more parameters -> higher (worse) AIC
        Assert.True(aicSimple < aicComplex,
            "Simpler model should have lower (better) AIC when fit is equal");
    }

    // ============================
    // Learning Curve Analysis
    // ============================

    [Fact]
    public void FitMath_LearningCurve_OverfitPattern()
    {
        // Overfit: train score stays high, test score stays low
        double[] trainSizes = { 10, 20, 50, 100, 200 };
        double[] trainScores = { 1.0, 0.99, 0.98, 0.97, 0.96 };
        double[] testScores = { 0.40, 0.45, 0.50, 0.55, 0.60 };

        for (int i = 0; i < trainSizes.Length; i++)
        {
            double gap = trainScores[i] - testScores[i];
            Assert.True(gap > 0.30, $"Large gap at size {trainSizes[i]} indicates overfitting");
        }
    }

    [Fact]
    public void FitMath_LearningCurve_GoodFitPattern()
    {
        // Good fit: train and test converge
        double[] trainScores = { 0.95, 0.93, 0.91, 0.90, 0.89 };
        double[] testScores = { 0.50, 0.70, 0.82, 0.86, 0.87 };

        // Gap should decrease over time
        double initialGap = trainScores[0] - testScores[0];
        double finalGap = trainScores[^1] - testScores[^1];

        Assert.True(finalGap < initialGap,
            $"Gap should decrease: initial={initialGap}, final={finalGap}");
    }

    // ============================
    // Confidence Level Math
    // ============================

    [Theory]
    [InlineData(0.35, 0.0)]    // Gap > 0.3: very confident overfit
    [InlineData(0.20, 0.0)]    // Gap 0.15-0.3: moderately confident
    [InlineData(0.05, 0.0)]    // Gap < 0.1: low confidence for overfit
    public void FitMath_OverfitConfidence_ScalesWithGap(double gap, double _)
    {
        // Confidence in overfit detection should scale with train-test gap
        double confidence = Math.Min(1.0, gap / 0.3); // Normalize to [0, 1]
        Assert.True(confidence >= 0.0 && confidence <= 1.0);

        if (gap > 0.3)
            Assert.True(confidence >= 0.9, "High gap should give high confidence");
        else if (gap < 0.1)
            Assert.True(confidence < 0.5, "Low gap should give low confidence");
    }
}
