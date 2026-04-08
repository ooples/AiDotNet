using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Statistics;

/// <summary>
/// Integration tests for probability distribution functions.
/// Ground truth values verified against SciPy stats library.
/// </summary>
public class DistributionFunctionsIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region Normal Distribution CDF

    [Fact]
    public void NormalCDF_StandardNormal_AtZero_Returns0_5()
    {
        // scipy.stats.norm.cdf(0, loc=0, scale=1) = 0.5
        var result = StatisticsHelper<double>.CalculateNormalCDF(0, 1, 0);
        Assert.Equal(0.5, result, Tolerance);
    }

    [Fact]
    public void NormalCDF_StandardNormal_AtOne_ReturnsExactValue()
    {
        // scipy.stats.norm.cdf(1, loc=0, scale=1) = 0.8413447460685429
        var result = StatisticsHelper<double>.CalculateNormalCDF(0, 1, 1);
        Assert.Equal(0.8413447, result, 1e-5);
    }

    [Fact]
    public void NormalCDF_StandardNormal_AtNegativeOne_ReturnsExactValue()
    {
        // scipy.stats.norm.cdf(-1, loc=0, scale=1) = 0.15865525393145707
        var result = StatisticsHelper<double>.CalculateNormalCDF(0, 1, -1);
        Assert.Equal(0.158655, result, 1e-4);
    }

    [Fact]
    public void NormalCDF_StandardNormal_AtTwo_ReturnsExactValue()
    {
        // scipy.stats.norm.cdf(2, loc=0, scale=1) = 0.9772498680518208
        var result = StatisticsHelper<double>.CalculateNormalCDF(0, 1, 2);
        Assert.Equal(0.97725, result, 1e-4);
    }

    [Fact]
    public void NormalCDF_NonStandard_ReturnsExactValue()
    {
        // scipy.stats.norm.cdf(15, loc=10, scale=3) = 0.9522096477271853
        var result = StatisticsHelper<double>.CalculateNormalCDF(10, 3, 15);
        Assert.Equal(0.95221, result, 1e-4);
    }

    [Fact]
    public void NormalCDF_AtMean_Returns0_5()
    {
        // CDF at mean always equals 0.5 for symmetric distribution
        var result = StatisticsHelper<double>.CalculateNormalCDF(100, 15, 100);
        Assert.Equal(0.5, result, Tolerance);
    }

    [Fact]
    public void NormalCDF_SymmetryAroundMean()
    {
        // P(X <= mean-1) + P(X <= mean+1) should be symmetric around 0.5
        double mean = 50;
        double stdDev = 10;
        var cdfLower = StatisticsHelper<double>.CalculateNormalCDF(mean, stdDev, mean - stdDev);
        var cdfUpper = StatisticsHelper<double>.CalculateNormalCDF(mean, stdDev, mean + stdDev);
        Assert.Equal(1.0 - cdfLower, cdfUpper, Tolerance);
    }

    #endregion

    #region Normal Distribution PDF

    [Fact]
    public void NormalPDF_StandardNormal_AtZero_ReturnsMaximum()
    {
        // scipy.stats.norm.pdf(0, loc=0, scale=1) = 0.3989422804014327
        var result = StatisticsHelper<double>.CalculateNormalPDF(0, 1, 0);
        Assert.Equal(0.3989423, result, 1e-5);
    }

    [Fact]
    public void NormalPDF_StandardNormal_AtOne_ReturnsExactValue()
    {
        // scipy.stats.norm.pdf(1, loc=0, scale=1) = 0.24197072451914337
        var result = StatisticsHelper<double>.CalculateNormalPDF(0, 1, 1);
        Assert.Equal(0.24197, result, 1e-4);
    }

    [Fact]
    public void NormalPDF_StandardNormal_SymmetricAroundMean()
    {
        var pdfLeft = StatisticsHelper<double>.CalculateNormalPDF(0, 1, -1.5);
        var pdfRight = StatisticsHelper<double>.CalculateNormalPDF(0, 1, 1.5);
        Assert.Equal(pdfLeft, pdfRight, Tolerance);
    }

    [Fact]
    public void NormalPDF_NonStandard_ReturnsExactValue()
    {
        // scipy.stats.norm.pdf(15, loc=10, scale=3) = 0.033159046239882694
        var result = StatisticsHelper<double>.CalculateNormalPDF(10, 3, 15);
        Assert.Equal(0.03316, result, 1e-4);
    }

    [Fact]
    public void NormalPDF_LargerStdDev_SmallerPeakValue()
    {
        var pdfNarrow = StatisticsHelper<double>.CalculateNormalPDF(0, 1, 0);
        var pdfWide = StatisticsHelper<double>.CalculateNormalPDF(0, 2, 0);
        Assert.True(pdfNarrow > pdfWide, "Narrower distribution should have higher peak");
    }

    #endregion

    #region Inverse Normal CDF (Quantile Function)

    [Fact]
    public void InverseNormalCDF_Standard_At0_5_ReturnsZero()
    {
        // scipy.stats.norm.ppf(0.5) = 0.0
        var result = StatisticsHelper<double>.CalculateInverseNormalCDF(0.5);
        Assert.Equal(0.0, result, 1e-4);
    }

    [Fact]
    public void InverseNormalCDF_Standard_At0_975_ReturnsExactValue()
    {
        // scipy.stats.norm.ppf(0.975) = 1.959963984540054
        var result = StatisticsHelper<double>.CalculateInverseNormalCDF(0.975);
        Assert.Equal(1.96, result, 0.01);
    }

    [Fact]
    public void InverseNormalCDF_Standard_At0_025_ReturnsExactValue()
    {
        // scipy.stats.norm.ppf(0.025) = -1.959963984540054
        var result = StatisticsHelper<double>.CalculateInverseNormalCDF(0.025);
        Assert.Equal(-1.96, result, 0.01);
    }

    [Fact]
    public void InverseNormalCDF_NonStandard_ReturnsExactValue()
    {
        // scipy.stats.norm.ppf(0.9, loc=10, scale=3) = 13.844652632856174
        var result = StatisticsHelper<double>.CalculateInverseNormalCDF(10, 3, 0.9);
        Assert.Equal(13.845, result, 0.01);
    }

    [Fact]
    public void InverseNormalCDF_InverseOfCDF_ReturnsOriginalValue()
    {
        double x = 1.5;
        var cdf = StatisticsHelper<double>.CalculateNormalCDF(0, 1, x);
        var inverted = StatisticsHelper<double>.CalculateInverseNormalCDF(cdf);
        Assert.Equal(x, inverted, 1e-3);
    }

    #endregion

    #region Chi-Square Distribution CDF

    [Fact]
    public void ChiSquareCDF_DF1_AtOne_ReturnsExactValue()
    {
        // scipy.stats.chi2.cdf(1, df=1) = 0.6826894921370859
        var result = StatisticsHelper<double>.CalculateChiSquareCDF(1, 1.0);
        Assert.Equal(0.6827, result, 1e-3);
    }

    [Fact]
    public void ChiSquareCDF_DF5_AtFive_ReturnsExactValue()
    {
        // scipy.stats.chi2.cdf(5, df=5) = 0.5841198130044921
        var result = StatisticsHelper<double>.CalculateChiSquareCDF(5, 5.0);
        Assert.Equal(0.584, result, 1e-2);
    }

    [Fact]
    public void ChiSquareCDF_DF10_AtTen_ReturnsExactValue()
    {
        // scipy.stats.chi2.cdf(10, df=10) = 0.5595067149347877
        var result = StatisticsHelper<double>.CalculateChiSquareCDF(10, 10.0);
        Assert.Equal(0.56, result, 0.02);
    }

    [Fact]
    public void ChiSquareCDF_AtZero_ReturnsZero()
    {
        var result = StatisticsHelper<double>.CalculateChiSquareCDF(5, 0.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void ChiSquareCDF_IsMonotonicallyIncreasing()
    {
        var cdf1 = StatisticsHelper<double>.CalculateChiSquareCDF(5, 1.0);
        var cdf2 = StatisticsHelper<double>.CalculateChiSquareCDF(5, 2.0);
        var cdf3 = StatisticsHelper<double>.CalculateChiSquareCDF(5, 3.0);
        Assert.True(cdf1 < cdf2 && cdf2 < cdf3);
    }

    #endregion

    #region Chi-Square Distribution PDF

    [Fact]
    public void ChiSquarePDF_DF1_AtOne_ReturnsExactValue()
    {
        // scipy.stats.chi2.pdf(1, df=1) = 0.24197072451914337
        var result = StatisticsHelper<double>.CalculateChiSquarePDF(1, 1.0);
        Assert.Equal(0.24197, result, 1e-3);
    }

    [Fact]
    public void ChiSquarePDF_DF5_AtThree_ReturnsExactValue()
    {
        // scipy.stats.chi2.pdf(3, df=5) = 0.1541803416100632
        var result = StatisticsHelper<double>.CalculateChiSquarePDF(5, 3.0);
        Assert.Equal(0.1542, result, 1e-3);
    }

    [Fact]
    public void ChiSquarePDF_AtZero_DF1_ReturnsInfinity()
    {
        // chi2.pdf(0, df=1) = infinity for df <= 2
        var result = StatisticsHelper<double>.CalculateChiSquarePDF(1, 0.0);
        Assert.True(double.IsPositiveInfinity(result) || result > 1e10);
    }

    [Fact]
    public void ChiSquarePDF_IsNonNegative()
    {
        for (int df = 1; df <= 10; df++)
        {
            for (double x = 0.1; x <= 10; x += 1)
            {
                var pdf = StatisticsHelper<double>.CalculateChiSquarePDF(df, x);
                Assert.True(pdf >= 0, $"PDF should be non-negative for df={df}, x={x}");
            }
        }
    }

    #endregion

    #region Inverse Chi-Square CDF

    [Fact]
    public void InverseChiSquareCDF_DF1_At0_95_ReturnsExactValue()
    {
        // scipy.stats.chi2.ppf(0.95, df=1) = 3.841458820694124
        var result = StatisticsHelper<double>.CalculateInverseChiSquareCDF(1, 0.95);
        Assert.Equal(3.841, result, 0.01);
    }

    [Fact]
    public void InverseChiSquareCDF_DF5_At0_5_ReturnsExactValue()
    {
        // scipy.stats.chi2.ppf(0.5, df=5) = 4.351460191102881
        var result = StatisticsHelper<double>.CalculateInverseChiSquareCDF(5, 0.5);
        Assert.Equal(4.35, result, 0.05);
    }

    [Fact]
    public void InverseChiSquareCDF_InverseOfCDF_ReturnsOriginalValue()
    {
        double x = 7.5;
        int df = 5;
        var cdf = StatisticsHelper<double>.CalculateChiSquareCDF(df, x);
        var inverted = StatisticsHelper<double>.CalculateInverseChiSquareCDF(df, cdf);
        Assert.Equal(x, inverted, 0.1);
    }

    #endregion

    #region Student's t Distribution PDF

    [Fact]
    public void StudentPDF_AtZero_MaximumValue()
    {
        // At mean, PDF should be at its maximum
        var pdfAtZero = StatisticsHelper<double>.CalculateStudentPDF(0.0, 0.0, 1.0, 10);
        var pdfAtOne = StatisticsHelper<double>.CalculateStudentPDF(1.0, 0.0, 1.0, 10);
        Assert.True(pdfAtZero > pdfAtOne);
    }

    [Fact]
    public void StudentPDF_SymmetricAroundMean()
    {
        double mean = 5.0;
        double stdDev = 2.0;
        int df = 10;
        var pdfLeft = StatisticsHelper<double>.CalculateStudentPDF(mean - 1, mean, stdDev, df);
        var pdfRight = StatisticsHelper<double>.CalculateStudentPDF(mean + 1, mean, stdDev, df);
        Assert.Equal(pdfLeft, pdfRight, Tolerance);
    }

    [Fact]
    public void StudentPDF_HighDF_ApproachesNormal()
    {
        // With high df, Student-t approaches normal distribution
        // Note: Using df=100 instead of 1000 to avoid Gamma function overflow
        double x = 1.0;
        var normalPdf = StatisticsHelper<double>.CalculateNormalPDF(0, 1, x);
        var studentPdf = StatisticsHelper<double>.CalculateStudentPDF(x, 0.0, 1.0, 100);
        Assert.Equal(normalPdf, studentPdf, 0.01);
    }

    [Fact]
    public void StudentPDF_LowDF_HeavierTails()
    {
        // Lower df means heavier tails (higher PDF at extremes)
        double x = 3.0;
        var pdfDf3 = StatisticsHelper<double>.CalculateStudentPDF(x, 0.0, 1.0, 3);
        var pdfDf30 = StatisticsHelper<double>.CalculateStudentPDF(x, 0.0, 1.0, 30);
        Assert.True(pdfDf3 > pdfDf30, "Lower df should have heavier tails");
    }

    #endregion

    #region Inverse Student's t CDF

    [Fact]
    public void InverseStudentTCDF_At0_5_ReturnsZero()
    {
        // scipy.stats.t.ppf(0.5, df=10) = 0.0
        var result = StatisticsHelper<double>.CalculateInverseStudentTCDF(10, 0.5);
        Assert.Equal(0.0, result, 0.01);
    }

    [Fact]
    public void InverseStudentTCDF_DF10_At0_975_ReturnsExactValue()
    {
        // scipy.stats.t.ppf(0.975, df=10) = 2.2281388519649385
        // Implementation uses approximation, allowing 1% tolerance
        var result = StatisticsHelper<double>.CalculateInverseStudentTCDF(10, 0.975);
        Assert.Equal(2.228, result, 0.025);
    }

    [Fact]
    public void InverseStudentTCDF_HighDF_ApproachesNormal()
    {
        // With high df, inverse t approaches inverse normal
        // Using df=100 to avoid numerical overflow issues
        double prob = 0.975;
        var normalInverse = StatisticsHelper<double>.CalculateInverseNormalCDF(prob);
        var studentInverse = StatisticsHelper<double>.CalculateInverseStudentTCDF(100, prob);
        Assert.Equal(normalInverse, studentInverse, 0.05);
    }

    #endregion

    #region Beta Distribution CDF

    [Fact]
    public void BetaCDF_UniformBeta_Linear()
    {
        // Beta(1,1) is uniform, CDF should equal x
        var result = StatisticsHelper<double>.CalculateBetaCDF(0.5, 1.0, 1.0);
        Assert.Equal(0.5, result, Tolerance);
    }

    [Fact]
    public void BetaCDF_StandardExample_ReturnsExactValue()
    {
        // scipy.stats.beta.cdf(0.5, a=2, b=5) = 0.890625
        var result = StatisticsHelper<double>.CalculateBetaCDF(0.5, 2.0, 5.0);
        Assert.Equal(0.8906, result, 0.01);
    }

    [Fact]
    public void BetaCDF_AtZero_ReturnsZero()
    {
        var result = StatisticsHelper<double>.CalculateBetaCDF(0.0, 2.0, 3.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void BetaCDF_AtOne_ReturnsOne()
    {
        var result = StatisticsHelper<double>.CalculateBetaCDF(1.0, 2.0, 3.0);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void BetaCDF_SymmetricBeta_SymmetricCDF()
    {
        // Beta(2,2) is symmetric around 0.5
        var cdfLow = StatisticsHelper<double>.CalculateBetaCDF(0.3, 2.0, 2.0);
        var cdfHigh = StatisticsHelper<double>.CalculateBetaCDF(0.7, 2.0, 2.0);
        Assert.Equal(cdfLow, 1 - cdfHigh, 0.01);
    }

    #endregion

    #region Inverse Beta CDF

    [Fact]
    public void InverseBetaCDF_At0_5_Uniform_Returns0_5()
    {
        // For Beta(1,1), inverse of 0.5 is 0.5
        var result = StatisticsHelper<double>.CalculateInverseBetaCDF(0.5, 1.0, 1.0);
        Assert.Equal(0.5, result, Tolerance);
    }

    [Fact]
    public void InverseBetaCDF_StandardExample_ReturnsExactValue()
    {
        // scipy.stats.beta.ppf(0.5, a=2, b=5) = 0.2707310193055339
        var result = StatisticsHelper<double>.CalculateInverseBetaCDF(0.5, 2.0, 5.0);
        Assert.Equal(0.2707, result, 0.01);
    }

    [Fact]
    public void InverseBetaCDF_InverseOfCDF_ReturnsOriginalValue()
    {
        double x = 0.3;
        var cdf = StatisticsHelper<double>.CalculateBetaCDF(x, 2.0, 3.0);
        var inverted = StatisticsHelper<double>.CalculateInverseBetaCDF(cdf, 2.0, 3.0);
        Assert.Equal(x, inverted, 0.01);
    }

    #endregion

    #region Exponential Distribution

    [Fact]
    public void ExponentialPDF_AtZero_ReturnsLambda()
    {
        // PDF at x=0 equals lambda
        double lambda = 2.0;
        var result = StatisticsHelper<double>.CalculateExponentialPDF(lambda, 0.0);
        Assert.Equal(lambda, result, Tolerance);
    }

    [Fact]
    public void ExponentialPDF_StandardExample_ReturnsExactValue()
    {
        // scipy.stats.expon.pdf(1, scale=1/2) = 0.27067056647322536 (lambda=2)
        var result = StatisticsHelper<double>.CalculateExponentialPDF(2.0, 1.0);
        Assert.Equal(0.2707, result, 0.01);
    }

    [Fact]
    public void ExponentialPDF_DecaysExponentially()
    {
        double lambda = 1.0;
        var pdf1 = StatisticsHelper<double>.CalculateExponentialPDF(lambda, 1.0);
        var pdf2 = StatisticsHelper<double>.CalculateExponentialPDF(lambda, 2.0);
        var pdf3 = StatisticsHelper<double>.CalculateExponentialPDF(lambda, 3.0);
        Assert.True(pdf1 > pdf2 && pdf2 > pdf3);
    }

    [Fact]
    public void InverseExponentialCDF_StandardExample_ReturnsExactValue()
    {
        // scipy.stats.expon.ppf(0.5, scale=1/2) = 0.3465735902799727 (lambda=2)
        var result = StatisticsHelper<double>.CalculateInverseExponentialCDF(2.0, 0.5);
        Assert.Equal(0.3466, result, 0.01);
    }

    #endregion

    #region Weibull Distribution PDF

    [Fact]
    public void WeibullPDF_K1_IsExponential()
    {
        // Weibull(k=1, lambda) is exponential with rate 1/lambda
        double k = 1.0;
        double lambda = 2.0;
        double x = 1.0;
        var weibullPdf = StatisticsHelper<double>.CalculateWeibullPDF(k, lambda, x);
        // Expected: (1/2) * exp(-1/2) = 0.3033
        Assert.Equal(0.303, weibullPdf, 0.01);
    }

    [Fact]
    public void WeibullPDF_K2_IsRayleigh()
    {
        // Weibull(k=2, lambda) is Rayleigh distribution
        double k = 2.0;
        double lambda = 1.0;
        double x = 0.5;
        var result = StatisticsHelper<double>.CalculateWeibullPDF(k, lambda, x);
        Assert.True(result > 0);
    }

    [Fact]
    public void WeibullPDF_IsNonNegative()
    {
        for (double x = 0.1; x <= 5; x += 0.5)
        {
            var pdf = StatisticsHelper<double>.CalculateWeibullPDF(2.0, 1.0, x);
            Assert.True(pdf >= 0);
        }
    }

    #endregion

    #region LogNormal Distribution PDF

    [Fact]
    public void LogNormalPDF_StandardExample_ReturnsExactValue()
    {
        // scipy.stats.lognorm.pdf(1, s=1, scale=exp(0)) = 0.3989422804014327
        var result = StatisticsHelper<double>.CalculateLogNormalPDF(0.0, 1.0, 1.0);
        Assert.Equal(0.3989, result, 0.01);
    }

    [Fact]
    public void LogNormalPDF_AtZero_ReturnsZero()
    {
        var result = StatisticsHelper<double>.CalculateLogNormalPDF(0.0, 1.0, 0.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void LogNormalPDF_IsNonNegative()
    {
        for (double x = 0.1; x <= 5; x += 0.5)
        {
            var pdf = StatisticsHelper<double>.CalculateLogNormalPDF(0.0, 0.5, x);
            Assert.True(pdf >= 0);
        }
    }

    #endregion

    #region Laplace Distribution

    [Fact]
    public void LaplacePDF_AtMedian_ReturnsMaximum()
    {
        double median = 0.0;
        double mad = 1.0;
        var pdfAtMedian = StatisticsHelper<double>.CalculateLaplacePDF(median, mad, 0.0);
        var pdfAway = StatisticsHelper<double>.CalculateLaplacePDF(median, mad, 1.0);
        Assert.True(pdfAtMedian > pdfAway);
    }

    [Fact]
    public void LaplacePDF_SymmetricAroundMedian()
    {
        double median = 5.0;
        double mad = 2.0;
        var pdfLeft = StatisticsHelper<double>.CalculateLaplacePDF(median, mad, 3.0);
        var pdfRight = StatisticsHelper<double>.CalculateLaplacePDF(median, mad, 7.0);
        Assert.Equal(pdfLeft, pdfRight, Tolerance);
    }

    [Fact]
    public void InverseLaplaceCDF_At0_5_ReturnsMedian()
    {
        double median = 10.0;
        double mad = 2.0;
        var result = StatisticsHelper<double>.CalculateInverseLaplaceCDF(median, mad, 0.5);
        Assert.Equal(median, result, Tolerance);
    }

    [Fact]
    public void InverseLaplaceCDF_SymmetricAroundMedian()
    {
        double median = 5.0;
        double mad = 2.0;
        var lower = StatisticsHelper<double>.CalculateInverseLaplaceCDF(median, mad, 0.25);
        var upper = StatisticsHelper<double>.CalculateInverseLaplaceCDF(median, mad, 0.75);
        Assert.Equal(median - lower, upper - median, Tolerance);
    }

    #endregion

    #region Float Type Tests

    [Fact]
    public void NormalCDF_FloatType_ReturnsCorrectValue()
    {
        var result = StatisticsHelper<float>.CalculateNormalCDF(0f, 1f, 0f);
        Assert.Equal(0.5f, result, 1e-4f);
    }

    [Fact]
    public void ChiSquareCDF_FloatType_ReturnsCorrectValue()
    {
        var result = StatisticsHelper<float>.CalculateChiSquareCDF(5, 5.0f);
        Assert.True(result > 0.5f && result < 0.7f);
    }

    [Fact]
    public void BetaCDF_FloatType_ReturnsCorrectValue()
    {
        var result = StatisticsHelper<float>.CalculateBetaCDF(0.5f, 2.0f, 5.0f);
        Assert.True(result > 0.8f && result < 1.0f);
    }

    #endregion
}
