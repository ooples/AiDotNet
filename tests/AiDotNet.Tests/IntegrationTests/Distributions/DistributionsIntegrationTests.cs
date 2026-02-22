using AiDotNet.Distributions;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Distributions;

/// <summary>
/// Integration tests for all distribution classes.
/// Tests PDF, CDF, InverseCDF, sampling, parameters, and validation.
/// </summary>
public class DistributionsIntegrationTests
{
    private const double Tolerance = 1e-4;

    #region NormalDistribution Tests

    [Fact]
    public void Normal_PdfAtMean_IsPeak()
    {
        var dist = new NormalDistribution<double>(0, 1);
        var pdfAtMean = dist.Pdf(0.0);
        var pdfAway = dist.Pdf(1.0);
        Assert.True(pdfAtMean > pdfAway);
    }

    [Fact]
    public void Normal_CdfAtMean_ReturnsHalf()
    {
        var dist = new NormalDistribution<double>(0, 1);
        Assert.Equal(0.5, dist.Cdf(0.0), Tolerance);
    }

    [Fact]
    public void Normal_CdfInverseCdf_RoundTrip()
    {
        var dist = new NormalDistribution<double>(0, 1);
        double[] probs = { 0.1, 0.25, 0.5, 0.75, 0.9 };
        foreach (var p in probs)
        {
            var roundTrip = dist.Cdf(dist.InverseCdf(p));
            Assert.Equal(p, roundTrip, Tolerance);
        }
    }

    [Fact]
    public void Normal_MeanAndVariance_MatchParameters()
    {
        var dist = new NormalDistribution<double>(5.0, 4.0);
        Assert.Equal(5.0, dist.Mean, Tolerance);
        Assert.Equal(4.0, dist.Variance, Tolerance);
    }

    [Fact]
    public void Normal_NumParameters_IsTwo()
    {
        var dist = new NormalDistribution<double>(0, 1);
        Assert.Equal(2, dist.NumParameters);
        Assert.Equal(2, dist.ParameterNames.Length);
    }

    [Fact]
    public void Normal_GradLogPdf_ReturnsCorrectLength()
    {
        var dist = new NormalDistribution<double>(0, 1);
        var grad = dist.GradLogPdf(0.5);
        Assert.Equal(2, grad.Length);
    }

    [Fact]
    public void Normal_FisherInformation_Is2x2()
    {
        var dist = new NormalDistribution<double>(0, 1);
        var fisher = dist.FisherInformation();
        Assert.Equal(2, fisher.Rows);
        Assert.Equal(2, fisher.Columns);
    }

    [Fact]
    public void Normal_Clone_ReturnsIndependentCopy()
    {
        var dist = new NormalDistribution<double>(3.0, 2.0);
        var clone = dist.Clone() as NormalDistribution<double>;
        Assert.NotNull(clone);
        Assert.Equal(dist.Mean, clone.Mean, Tolerance);
        Assert.Equal(dist.Variance, clone.Variance, Tolerance);
        // Verify independence: clone is a separate instance
        Assert.NotSame(dist, clone);
    }

    [Fact]
    public void Normal_FromMeanStdDev_SetsVarianceCorrectly()
    {
        var dist = NormalDistribution<double>.FromMeanStdDev(0.0, 2.0);
        Assert.Equal(0.0, dist.Mean, Tolerance);
        Assert.Equal(4.0, dist.Variance, Tolerance); // stddev=2 => variance=4
    }

    [Fact]
    public void Normal_InvalidVariance_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new NormalDistribution<double>(0, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NormalDistribution<double>(0, -1));
    }

    [Fact]
    public void Normal_Sample_ProducesValues()
    {
        var dist = new NormalDistribution<double>(0, 1);
        var rng = RandomHelper.CreateSeededRandom(42);
        var sample = dist.Sample(rng);
        Assert.False(double.IsNaN(sample));
        Assert.False(double.IsInfinity(sample));
    }

    [Fact]
    public void Normal_DefaultConstructor_IsStandardNormal()
    {
        var dist = new NormalDistribution<double>();
        Assert.Equal(0.0, dist.Mean, Tolerance);
        Assert.Equal(1.0, dist.Variance, Tolerance);
    }

    #endregion

    #region BetaDistribution Tests

    [Fact]
    public void Beta_PdfInUnitInterval_IsPositive()
    {
        var dist = new BetaDistribution<double>(2.0, 5.0);
        Assert.True(dist.Pdf(0.3) > 0);
    }

    [Fact]
    public void Beta_CdfBoundaries()
    {
        var dist = new BetaDistribution<double>(2.0, 2.0);
        Assert.Equal(0.0, dist.Cdf(0.0), Tolerance);
        Assert.Equal(1.0, dist.Cdf(1.0), Tolerance);
    }

    [Fact]
    public void Beta_Mean_EqualsAlphaOverAlphaPlusBeta()
    {
        double alpha = 3.0, beta = 7.0;
        var dist = new BetaDistribution<double>(alpha, beta);
        Assert.Equal(alpha / (alpha + beta), dist.Mean, Tolerance);
    }

    [Fact]
    public void Beta_InvalidParameters_Throw()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new BetaDistribution<double>(0, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new BetaDistribution<double>(1, 0));
    }

    [Fact]
    public void Beta_Clone_ReturnsIndependentCopy()
    {
        var dist = new BetaDistribution<double>(2.0, 3.0);
        var clone = dist.Clone();
        Assert.NotNull(clone);
        Assert.Equal(dist.Mean, clone.Mean, Tolerance);
    }

    #endregion

    #region GammaDistribution Tests

    [Fact]
    public void Gamma_PdfPositiveForPositiveX()
    {
        var dist = new GammaDistribution<double>(2.0, 1.0);
        Assert.True(dist.Pdf(1.0) > 0);
    }

    [Fact]
    public void Gamma_MeanEqualsShapeOverRate()
    {
        double shape = 3.0, rate = 2.0;
        var dist = new GammaDistribution<double>(shape, rate);
        Assert.Equal(shape / rate, dist.Mean, Tolerance);
    }

    [Fact]
    public void Gamma_VarianceEqualsShapeOverRateSquared()
    {
        double shape = 3.0, rate = 2.0;
        var dist = new GammaDistribution<double>(shape, rate);
        Assert.Equal(shape / (rate * rate), dist.Variance, Tolerance);
    }

    [Fact]
    public void Gamma_InvalidParameters_Throw()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new GammaDistribution<double>(0, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new GammaDistribution<double>(1, 0));
    }

    #endregion

    #region ExponentialDistribution Tests

    [Fact]
    public void Exponential_CdfAtZero_IsZero()
    {
        var dist = new ExponentialDistribution<double>(1.0);
        Assert.Equal(0.0, dist.Cdf(0.0), Tolerance);
    }

    [Fact]
    public void Exponential_MeanEqualsOneOverRate()
    {
        var dist = new ExponentialDistribution<double>(2.0);
        Assert.Equal(0.5, dist.Mean, Tolerance);
    }

    [Fact]
    public void Exponential_PdfAtZero_EqualsRate()
    {
        double rate = 3.0;
        var dist = new ExponentialDistribution<double>(rate);
        Assert.Equal(rate, dist.Pdf(0.0), Tolerance);
    }

    [Fact]
    public void Exponential_InvalidRate_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ExponentialDistribution<double>(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ExponentialDistribution<double>(-1));
    }

    #endregion

    #region PoissonDistribution Tests

    [Fact]
    public void Poisson_MeanEqualsVariance()
    {
        double lambda = 4.0;
        var dist = new PoissonDistribution<double>(lambda);
        Assert.Equal(lambda, dist.Mean, Tolerance);
        Assert.Equal(lambda, dist.Variance, Tolerance);
    }

    [Fact]
    public void Poisson_PmfAtKnownValues()
    {
        var dist = new PoissonDistribution<double>(1.0);
        // P(X=0) = e^(-1) ≈ 0.3679
        Assert.Equal(Math.Exp(-1), dist.Pdf(0.0), Tolerance);
    }

    [Fact]
    public void Poisson_InvalidLambda_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new PoissonDistribution<double>(0));
    }

    #endregion

    #region LaplaceDistribution Tests

    [Fact]
    public void Laplace_PdfSymmetric()
    {
        var dist = new LaplaceDistribution<double>(0.0, 1.0);
        var pdfPositive = dist.Pdf(1.0);
        var pdfNegative = dist.Pdf(-1.0);
        Assert.Equal(pdfPositive, pdfNegative, Tolerance);
    }

    [Fact]
    public void Laplace_CdfAtMedian_ReturnsHalf()
    {
        var dist = new LaplaceDistribution<double>(0.0, 1.0);
        Assert.Equal(0.5, dist.Cdf(0.0), Tolerance);
    }

    [Fact]
    public void Laplace_Mean_MatchesLocation()
    {
        var dist = new LaplaceDistribution<double>(3.0, 2.0);
        Assert.Equal(3.0, dist.Mean, Tolerance);
    }

    #endregion

    #region LogNormalDistribution Tests

    [Fact]
    public void LogNormal_PdfPositiveForPositiveX()
    {
        var dist = new LogNormalDistribution<double>(0.0, 1.0);
        Assert.True(dist.Pdf(1.0) > 0);
    }

    [Fact]
    public void LogNormal_CdfProperties()
    {
        var dist = new LogNormalDistribution<double>(0.0, 1.0);
        Assert.True(dist.Cdf(0.01) >= 0);
        Assert.True(dist.Cdf(100.0) <= 1.0);
    }

    [Fact]
    public void LogNormal_Clone_ReturnsIndependentCopy()
    {
        var dist = new LogNormalDistribution<double>(1.0, 0.5);
        var clone = dist.Clone();
        Assert.NotNull(clone);
        Assert.NotSame(dist, clone);
        Assert.Equal(dist.Mean, clone.Mean, Tolerance);
        Assert.Equal(dist.Variance, clone.Variance, Tolerance);
    }

    #endregion

    #region WeibullDistribution Tests

    [Fact]
    public void Weibull_ShapeParameterEffects()
    {
        // Shape=1 is Exponential
        var weibull1 = new WeibullDistribution<double>(1.0, 1.0);
        var exponential = new ExponentialDistribution<double>(1.0);
        Assert.Equal(exponential.Mean, weibull1.Mean, 0.1);
    }

    [Fact]
    public void Weibull_PdfPositive()
    {
        var dist = new WeibullDistribution<double>(2.0, 1.0);
        Assert.True(dist.Pdf(0.5) > 0);
    }

    [Fact]
    public void Weibull_InvalidParameters_Throw()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new WeibullDistribution<double>(0, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new WeibullDistribution<double>(1, 0));
    }

    #endregion

    #region StudentTDistribution Tests

    [Fact]
    public void StudentT_ApproachesNormalAsDfIncreases()
    {
        var normal = new NormalDistribution<double>(0, 1);
        var studentT = new StudentTDistribution<double>(1000); // High df
        // CDF values should be close
        Assert.Equal(normal.Cdf(0.0), studentT.Cdf(0.0), 0.01);
    }

    [Fact]
    public void StudentT_PdfSymmetric()
    {
        var dist = new StudentTDistribution<double>(5);
        Assert.Equal(dist.Pdf(1.0), dist.Pdf(-1.0), Tolerance);
    }

    [Fact]
    public void StudentT_InvalidDf_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new StudentTDistribution<double>(0));
    }

    #endregion

    #region NegativeBinomialDistribution Tests

    [Fact]
    public void NegativeBinomial_PmfAtKnownValues()
    {
        var dist = new NegativeBinomialDistribution<double>(5.0, 0.5);
        var pmf0 = dist.Pdf(0.0);
        Assert.True(pmf0 > 0);
        Assert.True(pmf0 < 1);
    }

    [Fact]
    public void NegativeBinomial_Mean_IsCorrect()
    {
        double r = 5.0, p = 0.5;
        var dist = new NegativeBinomialDistribution<double>(r, p);
        // Mean = r*(1-p)/p
        var expectedMean = r * (1 - p) / p;
        Assert.Equal(expectedMean, dist.Mean, Tolerance);
    }

    #endregion

    #region Cross-Distribution Tests

    [Fact]
    public void AllDistributions_Sample_ProducesFiniteValues()
    {
        var rng = RandomHelper.CreateSeededRandom(42);
        var distributions = new IParametricDistribution<double>[]
        {
            new NormalDistribution<double>(0, 1),
            new BetaDistribution<double>(2, 2),
            new GammaDistribution<double>(2, 1),
            new ExponentialDistribution<double>(1),
            new LaplaceDistribution<double>(0, 1),
            new WeibullDistribution<double>(2, 1),
            new StudentTDistribution<double>(5),
        };

        foreach (var dist in distributions)
        {
            if (dist is ISamplingDistribution<double> sampling)
            {
                var sample = sampling.Sample(rng);
                Assert.False(double.IsNaN(sample), $"{dist.GetType().Name} produced NaN");
                Assert.False(double.IsInfinity(sample), $"{dist.GetType().Name} produced Infinity");
            }
        }
    }

    #endregion
}
