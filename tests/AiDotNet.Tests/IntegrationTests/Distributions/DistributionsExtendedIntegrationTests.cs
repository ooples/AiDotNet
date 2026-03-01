using AiDotNet.Distributions;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Distributions;

/// <summary>
/// Extended integration tests for all distribution classes with deep mathematical verification.
/// Tests hand-calculated PDF/CDF values, GradLogPdf via finite difference, Fisher information
/// properties, LogPdf consistency, distribution-specific identities, and factory methods.
/// </summary>
public class DistributionsExtendedIntegrationTests
{
    private const double Tolerance = 1e-4;
    private const double GradTolerance = 1e-3;

    #region Normal Distribution - Deep Verification

    [Fact]
    public void Normal_PdfAtMean_HandCalculated()
    {
        // For N(0,1): PDF(0) = 1/sqrt(2*pi) ≈ 0.398942
        var dist = new NormalDistribution<double>(0, 1);
        double expected = 1.0 / Math.Sqrt(2 * Math.PI);
        Assert.Equal(expected, dist.Pdf(0.0), Tolerance);
    }

    [Fact]
    public void Normal_PdfAt1_HandCalculated()
    {
        // For N(0,1): PDF(1) = (1/sqrt(2*pi)) * exp(-0.5) ≈ 0.241971
        var dist = new NormalDistribution<double>(0, 1);
        double expected = Math.Exp(-0.5) / Math.Sqrt(2 * Math.PI);
        Assert.Equal(expected, dist.Pdf(1.0), Tolerance);
    }

    [Fact]
    public void Normal_CdfKnownValues()
    {
        // Standard normal CDF known values:
        // Phi(-1) ≈ 0.15866, Phi(0) = 0.5, Phi(1) ≈ 0.84134, Phi(1.96) ≈ 0.975
        var dist = new NormalDistribution<double>(0, 1);
        Assert.Equal(0.15866, dist.Cdf(-1.0), 1e-3);
        Assert.Equal(0.5, dist.Cdf(0.0), Tolerance);
        Assert.Equal(0.84134, dist.Cdf(1.0), 1e-3);
        Assert.Equal(0.975, dist.Cdf(1.96), 1e-2);
    }

    [Fact]
    public void Normal_LogPdf_ConsistentWithPdf()
    {
        var dist = new NormalDistribution<double>(2.0, 3.0);
        double[] testPoints = { -1.0, 0.0, 1.0, 2.0, 5.0 };
        foreach (var x in testPoints)
        {
            double logPdf = dist.LogPdf(x);
            double pdfLog = Math.Log(dist.Pdf(x));
            Assert.Equal(pdfLog, logPdf, Tolerance);
        }
    }

    [Fact]
    public void Normal_GradLogPdf_FiniteDifference()
    {
        var dist = new NormalDistribution<double>(2.0, 3.0);
        double x = 1.5;
        var analyticalGrad = dist.GradLogPdf(x);

        // Numerical gradient w.r.t. parameters
        double h = 1e-5;

        // Grad w.r.t. mean: d/d(mean) log(pdf) = (x - mean) / variance
        var distPlus = new NormalDistribution<double>(2.0 + h, 3.0);
        var distMinus = new NormalDistribution<double>(2.0 - h, 3.0);
        double numGradMean = (distPlus.LogPdf(x) - distMinus.LogPdf(x)) / (2 * h);
        Assert.Equal(numGradMean, analyticalGrad[0], GradTolerance);

        // Grad w.r.t. variance: d/d(var) log(pdf) = -1/(2*var) + (x-mean)^2/(2*var^2)
        var distPlusV = new NormalDistribution<double>(2.0, 3.0 + h);
        var distMinusV = new NormalDistribution<double>(2.0, 3.0 - h);
        double numGradVar = (distPlusV.LogPdf(x) - distMinusV.LogPdf(x)) / (2 * h);
        Assert.Equal(numGradVar, analyticalGrad[1], GradTolerance);
    }

    [Fact]
    public void Normal_FisherInformation_HandCalculated()
    {
        // For N(mean, variance):
        // I = [[1/variance, 0], [0, 1/(2*variance^2)]]
        var dist = new NormalDistribution<double>(0, 4.0);
        var fisher = dist.FisherInformation();

        Assert.Equal(1.0 / 4.0, fisher[0, 0], Tolerance);
        Assert.Equal(0.0, fisher[0, 1], Tolerance);
        Assert.Equal(0.0, fisher[1, 0], Tolerance);
        Assert.Equal(1.0 / (2.0 * 16.0), fisher[1, 1], Tolerance);
    }

    [Fact]
    public void Normal_SampleMeanConverges()
    {
        var dist = new NormalDistribution<double>(5.0, 2.0);
        var rng = RandomHelper.CreateSeededRandom(42);
        double sum = 0;
        int n = 10000;
        for (int i = 0; i < n; i++)
        {
            sum += dist.Sample(rng);
        }
        double sampleMean = sum / n;
        // Sample mean should be close to 5.0 with n=10000
        Assert.True(Math.Abs(sampleMean - 5.0) < 0.1,
            $"Sample mean {sampleMean} too far from theoretical 5.0");
    }

    [Fact]
    public void Normal_NonStandard_CdfRoundTrip()
    {
        // For N(3, 4): CDF(InverseCDF(p)) = p at various quantiles
        var dist = new NormalDistribution<double>(3.0, 4.0);
        double[] probs = { 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99 };
        foreach (var p in probs)
        {
            double roundTrip = dist.Cdf(dist.InverseCdf(p));
            Assert.Equal(p, roundTrip, 1e-3);
        }
    }

    #endregion

    #region Beta Distribution - Deep Verification

    [Fact]
    public void Beta_UniformCase_PdfIsOne()
    {
        // Beta(1,1) = Uniform(0,1), so PDF = 1 everywhere in (0,1)
        var dist = new BetaDistribution<double>(1.0, 1.0);
        Assert.Equal(1.0, dist.Pdf(0.3), Tolerance);
        Assert.Equal(1.0, dist.Pdf(0.5), Tolerance);
        Assert.Equal(1.0, dist.Pdf(0.9), Tolerance);
    }

    [Fact]
    public void Beta_SymmetricCase_CdfAtHalf()
    {
        // For symmetric Beta(a, a), CDF(0.5) = 0.5
        var dist = new BetaDistribution<double>(3.0, 3.0);
        Assert.Equal(0.5, dist.Cdf(0.5), Tolerance);
    }

    [Fact]
    public void Beta_Variance_HandCalculated()
    {
        // Var(Beta(a,b)) = ab / ((a+b)^2 * (a+b+1))
        double a = 2.0, b = 5.0;
        var dist = new BetaDistribution<double>(a, b);
        double expected = a * b / ((a + b) * (a + b) * (a + b + 1));
        Assert.Equal(expected, dist.Variance, Tolerance);
    }

    [Fact]
    public void Beta_CdfInverseCdf_RoundTrip()
    {
        var dist = new BetaDistribution<double>(2.0, 5.0);
        double[] probs = { 0.1, 0.25, 0.5, 0.75, 0.9 };
        foreach (var p in probs)
        {
            double roundTrip = dist.Cdf(dist.InverseCdf(p));
            Assert.Equal(p, roundTrip, 1e-3);
        }
    }

    [Fact]
    public void Beta_GradLogPdf_FiniteDifference()
    {
        var dist = new BetaDistribution<double>(3.0, 5.0);
        double x = 0.3;
        var analyticalGrad = dist.GradLogPdf(x);

        double h = 1e-5;

        // Grad w.r.t. alpha
        var distPlus = new BetaDistribution<double>(3.0 + h, 5.0);
        var distMinus = new BetaDistribution<double>(3.0 - h, 5.0);
        double numGradAlpha = (distPlus.LogPdf(x) - distMinus.LogPdf(x)) / (2 * h);
        Assert.Equal(numGradAlpha, analyticalGrad[0], GradTolerance);

        // Grad w.r.t. beta
        var distPlusB = new BetaDistribution<double>(3.0, 5.0 + h);
        var distMinusB = new BetaDistribution<double>(3.0, 5.0 - h);
        double numGradBeta = (distPlusB.LogPdf(x) - distMinusB.LogPdf(x)) / (2 * h);
        Assert.Equal(numGradBeta, analyticalGrad[1], GradTolerance);
    }

    [Fact]
    public void Beta_FromMeanVariance_RecoversParameters()
    {
        double a = 4.0, b = 6.0;
        var original = new BetaDistribution<double>(a, b);
        var reconstructed = BetaDistribution<double>.FromMeanVariance(
            original.Mean, original.Variance);
        Assert.Equal(original.Mean, reconstructed.Mean, Tolerance);
        Assert.Equal(original.Variance, reconstructed.Variance, Tolerance);
    }

    [Fact]
    public void Beta_LogPdf_ConsistentWithPdf()
    {
        var dist = new BetaDistribution<double>(2.0, 3.0);
        double[] testPoints = { 0.1, 0.3, 0.5, 0.7, 0.9 };
        foreach (var x in testPoints)
        {
            double logPdf = dist.LogPdf(x);
            double pdfLog = Math.Log(dist.Pdf(x));
            Assert.Equal(pdfLog, logPdf, Tolerance);
        }
    }

    #endregion

    #region Gamma Distribution - Deep Verification

    [Fact]
    public void Gamma_Shape1_IsExponential()
    {
        // Gamma(1, rate) = Exponential(rate)
        double rate = 2.0;
        var gamma = new GammaDistribution<double>(1.0, rate);
        var expo = new ExponentialDistribution<double>(rate);

        double[] testPoints = { 0.5, 1.0, 2.0, 3.0 };
        foreach (var x in testPoints)
        {
            Assert.Equal(expo.Pdf(x), gamma.Pdf(x), Tolerance);
            Assert.Equal(expo.Cdf(x), gamma.Cdf(x), Tolerance);
        }
    }

    [Fact]
    public void Gamma_PdfAtMode_HandCalculated()
    {
        // Mode of Gamma(alpha, beta) = (alpha-1)/beta for alpha >= 1
        // Gamma(3, 2): mode = 2/2 = 1.0
        var dist = new GammaDistribution<double>(3.0, 2.0);
        double mode = (3.0 - 1.0) / 2.0;

        // PDF at mode should be maximum
        double pdfAtMode = dist.Pdf(mode);
        Assert.True(pdfAtMode > dist.Pdf(mode - 0.1));
        Assert.True(pdfAtMode > dist.Pdf(mode + 0.1));
    }

    [Fact]
    public void Gamma_CdfRoundTrip()
    {
        var dist = new GammaDistribution<double>(3.0, 2.0);
        double[] probs = { 0.1, 0.25, 0.5, 0.75, 0.9 };
        foreach (var p in probs)
        {
            double roundTrip = dist.Cdf(dist.InverseCdf(p));
            Assert.Equal(p, roundTrip, 1e-3);
        }
    }

    [Fact]
    public void Gamma_GradLogPdf_FiniteDifference()
    {
        var dist = new GammaDistribution<double>(3.0, 2.0);
        double x = 1.5;
        var analyticalGrad = dist.GradLogPdf(x);

        double h = 1e-5;

        // Grad w.r.t. shape
        var distPlus = new GammaDistribution<double>(3.0 + h, 2.0);
        var distMinus = new GammaDistribution<double>(3.0 - h, 2.0);
        double numGradShape = (distPlus.LogPdf(x) - distMinus.LogPdf(x)) / (2 * h);
        Assert.Equal(numGradShape, analyticalGrad[0], GradTolerance);

        // Grad w.r.t. rate
        var distPlusR = new GammaDistribution<double>(3.0, 2.0 + h);
        var distMinusR = new GammaDistribution<double>(3.0, 2.0 - h);
        double numGradRate = (distPlusR.LogPdf(x) - distMinusR.LogPdf(x)) / (2 * h);
        Assert.Equal(numGradRate, analyticalGrad[1], GradTolerance);
    }

    [Fact]
    public void Gamma_FromMeanVariance_RecoversParameters()
    {
        double shape = 4.0, rate = 2.0;
        var original = new GammaDistribution<double>(shape, rate);
        var reconstructed = GammaDistribution<double>.FromMeanVariance(
            original.Mean, original.Variance);
        Assert.Equal(original.Mean, reconstructed.Mean, Tolerance);
        Assert.Equal(original.Variance, reconstructed.Variance, Tolerance);
    }

    [Fact]
    public void Gamma_LogPdf_ConsistentWithPdf()
    {
        var dist = new GammaDistribution<double>(2.0, 1.5);
        double[] testPoints = { 0.5, 1.0, 2.0, 4.0 };
        foreach (var x in testPoints)
        {
            double logPdf = dist.LogPdf(x);
            double pdfLog = Math.Log(dist.Pdf(x));
            Assert.Equal(pdfLog, logPdf, Tolerance);
        }
    }

    #endregion

    #region Exponential Distribution - Deep Verification

    [Fact]
    public void Exponential_CdfFormula_HandCalculated()
    {
        // CDF(x) = 1 - exp(-rate * x)
        double rate = 3.0;
        var dist = new ExponentialDistribution<double>(rate);
        double x = 0.5;
        double expected = 1 - Math.Exp(-rate * x);
        Assert.Equal(expected, dist.Cdf(x), Tolerance);
    }

    [Fact]
    public void Exponential_Variance_HandCalculated()
    {
        // Var(Exp(rate)) = 1/rate^2
        double rate = 2.5;
        var dist = new ExponentialDistribution<double>(rate);
        Assert.Equal(1.0 / (rate * rate), dist.Variance, Tolerance);
    }

    [Fact]
    public void Exponential_MemorylessProperty()
    {
        // P(X > s+t | X > s) = P(X > t)
        var dist = new ExponentialDistribution<double>(1.0);
        double s = 1.0, t = 2.0;

        double pGtSPlusT = 1 - dist.Cdf(s + t);
        double pGtS = 1 - dist.Cdf(s);
        double pGtT = 1 - dist.Cdf(t);

        double conditional = pGtSPlusT / pGtS;
        Assert.Equal(pGtT, conditional, Tolerance);
    }

    [Fact]
    public void Exponential_InverseCdfRoundTrip()
    {
        var dist = new ExponentialDistribution<double>(2.0);
        double[] probs = { 0.1, 0.25, 0.5, 0.75, 0.9 };
        foreach (var p in probs)
        {
            double roundTrip = dist.Cdf(dist.InverseCdf(p));
            Assert.Equal(p, roundTrip, Tolerance);
        }
    }

    #endregion

    #region Laplace Distribution - Deep Verification

    [Fact]
    public void Laplace_PdfAtLocation_HandCalculated()
    {
        // PDF at location = 1/(2*scale)
        double loc = 3.0, scale = 2.0;
        var dist = new LaplaceDistribution<double>(loc, scale);
        double expected = 1.0 / (2.0 * scale);
        Assert.Equal(expected, dist.Pdf(loc), Tolerance);
    }

    [Fact]
    public void Laplace_Variance_HandCalculated()
    {
        // Var(Laplace(loc, scale)) = 2 * scale^2
        double scale = 3.0;
        var dist = new LaplaceDistribution<double>(0.0, scale);
        double expected = 2.0 * scale * scale;
        Assert.Equal(expected, dist.Variance, Tolerance);
    }

    [Fact]
    public void Laplace_CdfKnownValues()
    {
        // CDF(x) = 0.5 * exp((x-loc)/scale) for x < loc
        // CDF(x) = 1 - 0.5 * exp(-(x-loc)/scale) for x >= loc
        var dist = new LaplaceDistribution<double>(0.0, 1.0);

        Assert.Equal(0.5 * Math.Exp(-1.0), dist.Cdf(-1.0), Tolerance);
        Assert.Equal(0.5, dist.Cdf(0.0), Tolerance);
        Assert.Equal(1.0 - 0.5 * Math.Exp(-1.0), dist.Cdf(1.0), Tolerance);
    }

    [Fact]
    public void Laplace_GradLogPdf_FiniteDifference()
    {
        var dist = new LaplaceDistribution<double>(2.0, 3.0);
        double x = 4.0; // x > location so sign is positive
        var analyticalGrad = dist.GradLogPdf(x);

        double h = 1e-5;

        // Grad w.r.t. location
        var distPlus = new LaplaceDistribution<double>(2.0 + h, 3.0);
        var distMinus = new LaplaceDistribution<double>(2.0 - h, 3.0);
        double numGradLoc = (distPlus.LogPdf(x) - distMinus.LogPdf(x)) / (2 * h);
        Assert.Equal(numGradLoc, analyticalGrad[0], GradTolerance);

        // Grad w.r.t. scale
        var distPlusS = new LaplaceDistribution<double>(2.0, 3.0 + h);
        var distMinusS = new LaplaceDistribution<double>(2.0, 3.0 - h);
        double numGradScale = (distPlusS.LogPdf(x) - distMinusS.LogPdf(x)) / (2 * h);
        Assert.Equal(numGradScale, analyticalGrad[1], GradTolerance);
    }

    [Fact]
    public void Laplace_InverseCdfRoundTrip()
    {
        var dist = new LaplaceDistribution<double>(1.0, 2.0);
        double[] probs = { 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95 };
        foreach (var p in probs)
        {
            double roundTrip = dist.Cdf(dist.InverseCdf(p));
            Assert.Equal(p, roundTrip, 1e-3);
        }
    }

    [Fact]
    public void Laplace_LogPdf_ConsistentWithPdf()
    {
        var dist = new LaplaceDistribution<double>(1.0, 2.0);
        double[] testPoints = { -2.0, 0.0, 1.0, 3.0, 5.0 };
        foreach (var x in testPoints)
        {
            double logPdf = dist.LogPdf(x);
            double pdfLog = Math.Log(dist.Pdf(x));
            Assert.Equal(pdfLog, logPdf, Tolerance);
        }
    }

    #endregion

    #region LogNormal Distribution - Deep Verification

    [Fact]
    public void LogNormal_MeanFormula_HandCalculated()
    {
        // E[X] = exp(mu + sigma^2/2)
        double mu = 1.0, sigma = 0.5;
        var dist = new LogNormalDistribution<double>(mu, sigma);
        double expected = Math.Exp(mu + sigma * sigma / 2);
        Assert.Equal(expected, dist.Mean, Tolerance);
    }

    [Fact]
    public void LogNormal_VarianceFormula_HandCalculated()
    {
        // Var[X] = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
        double mu = 1.0, sigma = 0.5;
        var dist = new LogNormalDistribution<double>(mu, sigma);
        double sigSq = sigma * sigma;
        double expected = (Math.Exp(sigSq) - 1) * Math.Exp(2 * mu + sigSq);
        Assert.Equal(expected, dist.Variance, Tolerance);
    }

    [Fact]
    public void LogNormal_MedianIsExpMu()
    {
        // Median = exp(mu)
        double mu = 2.0, sigma = 1.0;
        var dist = new LogNormalDistribution<double>(mu, sigma);
        double median = dist.InverseCdf(0.5);
        Assert.Equal(Math.Exp(mu), median, 1e-2);
    }

    [Fact]
    public void LogNormal_CdfRoundTrip()
    {
        var dist = new LogNormalDistribution<double>(0.0, 1.0);
        double[] probs = { 0.1, 0.25, 0.5, 0.75, 0.9 };
        foreach (var p in probs)
        {
            double roundTrip = dist.Cdf(dist.InverseCdf(p));
            Assert.Equal(p, roundTrip, 1e-3);
        }
    }

    [Fact]
    public void LogNormal_GradLogPdf_FiniteDifference()
    {
        var dist = new LogNormalDistribution<double>(1.0, 0.5);
        double x = 2.0;
        var analyticalGrad = dist.GradLogPdf(x);

        double h = 1e-5;

        // Grad w.r.t. mu
        var distPlus = new LogNormalDistribution<double>(1.0 + h, 0.5);
        var distMinus = new LogNormalDistribution<double>(1.0 - h, 0.5);
        double numGradMu = (distPlus.LogPdf(x) - distMinus.LogPdf(x)) / (2 * h);
        Assert.Equal(numGradMu, analyticalGrad[0], GradTolerance);

        // Grad w.r.t. sigma
        var distPlusS = new LogNormalDistribution<double>(1.0, 0.5 + h);
        var distMinusS = new LogNormalDistribution<double>(1.0, 0.5 - h);
        double numGradSigma = (distPlusS.LogPdf(x) - distMinusS.LogPdf(x)) / (2 * h);
        Assert.Equal(numGradSigma, analyticalGrad[1], GradTolerance);
    }

    [Fact]
    public void LogNormal_FromMeanVariance_RecoversDistribution()
    {
        double mu = 1.0, sigma = 0.5;
        var original = new LogNormalDistribution<double>(mu, sigma);
        var reconstructed = LogNormalDistribution<double>.FromMeanVariance(
            original.Mean, original.Variance);
        // The reconstructed should have the same mean and variance
        Assert.Equal(original.Mean, reconstructed.Mean, Tolerance);
        Assert.Equal(original.Variance, reconstructed.Variance, Tolerance);
    }

    [Fact]
    public void LogNormal_LogPdf_ConsistentWithPdf()
    {
        var dist = new LogNormalDistribution<double>(0.0, 1.0);
        double[] testPoints = { 0.5, 1.0, 2.0, 5.0 };
        foreach (var x in testPoints)
        {
            double logPdf = dist.LogPdf(x);
            double pdfLog = Math.Log(dist.Pdf(x));
            Assert.Equal(pdfLog, logPdf, Tolerance);
        }
    }

    #endregion

    #region Weibull Distribution - Deep Verification

    [Fact]
    public void Weibull_CdfFormula_HandCalculated()
    {
        // CDF(x) = 1 - exp(-(x/lambda)^k)
        double k = 2.0, lambda = 3.0;
        var dist = new WeibullDistribution<double>(k, lambda);
        double x = 2.0;
        double expected = 1 - Math.Exp(-Math.Pow(x / lambda, k));
        Assert.Equal(expected, dist.Cdf(x), Tolerance);
    }

    [Fact]
    public void Weibull_InverseCdfFormula_HandCalculated()
    {
        // InverseCDF(p) = lambda * (-ln(1-p))^(1/k)
        double k = 2.0, lambda = 3.0;
        var dist = new WeibullDistribution<double>(k, lambda);
        double p = 0.5;
        double expected = lambda * Math.Pow(-Math.Log(1 - p), 1 / k);
        Assert.Equal(expected, dist.InverseCdf(p), Tolerance);
    }

    [Fact]
    public void Weibull_CdfRoundTrip()
    {
        var dist = new WeibullDistribution<double>(2.0, 3.0);
        double[] probs = { 0.1, 0.25, 0.5, 0.75, 0.9 };
        foreach (var p in probs)
        {
            double roundTrip = dist.Cdf(dist.InverseCdf(p));
            Assert.Equal(p, roundTrip, Tolerance);
        }
    }

    [Fact]
    public void Weibull_MeanFormula_HandCalculated()
    {
        // Mean = lambda * Gamma(1 + 1/k)
        // For k=2, lambda=1: Mean = Gamma(1.5) = sqrt(pi)/2 ≈ 0.8862
        var dist = new WeibullDistribution<double>(2.0, 1.0);
        double expected = Math.Sqrt(Math.PI) / 2.0;
        Assert.Equal(expected, dist.Mean, 1e-3);
    }

    [Fact]
    public void Weibull_GradLogPdf_FiniteDifference()
    {
        var dist = new WeibullDistribution<double>(2.0, 3.0);
        double x = 2.0;
        var analyticalGrad = dist.GradLogPdf(x);

        double h = 1e-5;

        // Grad w.r.t. shape
        var distPlus = new WeibullDistribution<double>(2.0 + h, 3.0);
        var distMinus = new WeibullDistribution<double>(2.0 - h, 3.0);
        double numGradShape = (distPlus.LogPdf(x) - distMinus.LogPdf(x)) / (2 * h);
        Assert.Equal(numGradShape, analyticalGrad[0], GradTolerance);

        // Grad w.r.t. scale
        var distPlusS = new WeibullDistribution<double>(2.0, 3.0 + h);
        var distMinusS = new WeibullDistribution<double>(2.0, 3.0 - h);
        double numGradScale = (distPlusS.LogPdf(x) - distMinusS.LogPdf(x)) / (2 * h);
        Assert.Equal(numGradScale, analyticalGrad[1], GradTolerance);
    }

    [Fact]
    public void Weibull_LogPdf_ConsistentWithPdf()
    {
        var dist = new WeibullDistribution<double>(2.0, 3.0);
        double[] testPoints = { 0.5, 1.0, 2.0, 4.0 };
        foreach (var x in testPoints)
        {
            double logPdf = dist.LogPdf(x);
            double pdfLog = Math.Log(dist.Pdf(x));
            Assert.Equal(pdfLog, logPdf, Tolerance);
        }
    }

    #endregion

    #region Student-T Distribution - Deep Verification

    [Fact]
    public void StudentT_Df1_IsCauchy()
    {
        // Student-t with df=1 is the Cauchy distribution
        // Cauchy PDF(0) = 1/pi ≈ 0.31831
        var dist = new StudentTDistribution<double>(1);
        double expected = 1.0 / Math.PI;
        Assert.Equal(expected, dist.Pdf(0.0), 1e-3);
    }

    [Fact]
    public void StudentT_CdfAtZero_IsHalf()
    {
        // For any symmetric distribution, CDF(0) = 0.5
        foreach (int df in new[] { 1, 3, 5, 10, 30 })
        {
            var dist = new StudentTDistribution<double>(df);
            Assert.Equal(0.5, dist.Cdf(0.0), Tolerance);
        }
    }

    [Fact]
    public void StudentT_HighDf_ApproachesNormalPdf()
    {
        // Student-t with large df approaches standard normal
        var tDist = new StudentTDistribution<double>(1000);
        var normal = new NormalDistribution<double>(0, 1);

        double[] testPoints = { -2, -1, 0, 1, 2 };
        foreach (var x in testPoints)
        {
            Assert.Equal(normal.Pdf(x), tDist.Pdf(x), 1e-2);
        }
    }

    [Fact]
    public void StudentT_CdfRoundTrip()
    {
        var dist = new StudentTDistribution<double>(5);
        double[] probs = { 0.1, 0.25, 0.5, 0.75, 0.9 };
        foreach (var p in probs)
        {
            double roundTrip = dist.Cdf(dist.InverseCdf(p));
            Assert.Equal(p, roundTrip, 1e-2);
        }
    }

    #endregion

    #region Poisson Distribution - Deep Verification

    [Fact]
    public void Poisson_PmfKnownValues()
    {
        // P(X=k) = lambda^k * exp(-lambda) / k!
        // For lambda=2: P(0)=e^-2, P(1)=2*e^-2, P(2)=2*e^-2
        var dist = new PoissonDistribution<double>(2.0);
        double e2 = Math.Exp(-2);

        Assert.Equal(e2, dist.Pdf(0.0), Tolerance);          // lambda^0/0! * e^-lambda = e^-2
        Assert.Equal(2 * e2, dist.Pdf(1.0), Tolerance);      // lambda^1/1! * e^-lambda = 2*e^-2
        Assert.Equal(2 * e2, dist.Pdf(2.0), Tolerance);      // lambda^2/2! * e^-lambda = 4/2*e^-2 = 2*e^-2
    }

    [Fact]
    public void Poisson_PmfSumsToApproximatelyOne()
    {
        // Sum of PMF values from 0 to N should approach 1
        var dist = new PoissonDistribution<double>(3.0);
        double sum = 0;
        for (int k = 0; k <= 20; k++)
        {
            sum += dist.Pdf((double)k);
        }
        Assert.Equal(1.0, sum, 1e-6);
    }

    #endregion

    #region Cross-Distribution Tests

    [Fact]
    public void AllDistributions_PdfNonNegative()
    {
        var distributions = new IParametricDistribution<double>[]
        {
            new NormalDistribution<double>(0, 1),
            new BetaDistribution<double>(2, 3),
            new GammaDistribution<double>(2, 1),
            new ExponentialDistribution<double>(1),
            new LaplaceDistribution<double>(0, 1),
            new LogNormalDistribution<double>(0, 1),
            new WeibullDistribution<double>(2, 1),
            new StudentTDistribution<double>(5),
        };

        double[] testPoints = { 0.01, 0.1, 0.5, 1.0, 2.0, 5.0 };

        foreach (var dist in distributions)
        {
            foreach (var x in testPoints)
            {
                double pdf = dist.Pdf(x);
                Assert.True(pdf >= 0, $"{dist.GetType().Name}.Pdf({x}) = {pdf} is negative");
            }
        }
    }

    [Fact]
    public void AllDistributions_StdDev_IsSqrtVariance()
    {
        var distributions = new IParametricDistribution<double>[]
        {
            new NormalDistribution<double>(0, 4),
            new BetaDistribution<double>(2, 3),
            new GammaDistribution<double>(2, 1),
            new ExponentialDistribution<double>(2),
            new LaplaceDistribution<double>(0, 2),
            new LogNormalDistribution<double>(0, 1),
            new WeibullDistribution<double>(2, 1),
        };

        foreach (var dist in distributions)
        {
            if (dist is ISamplingDistribution<double> sampling)
            {
                double stdDev = sampling.StdDev;
                double variance = dist.Variance;
                Assert.Equal(Math.Sqrt(variance), stdDev, Tolerance);
            }
        }
    }

    [Fact]
    public void AllDistributions_FisherInformation_IsSymmetric()
    {
        var distributions = new IParametricDistribution<double>[]
        {
            new NormalDistribution<double>(2, 3),
            new GammaDistribution<double>(3, 2),
            new BetaDistribution<double>(2, 5),
            new LaplaceDistribution<double>(1, 2),
            new LogNormalDistribution<double>(1, 0.5),
            new WeibullDistribution<double>(2, 3),
        };

        foreach (var dist in distributions)
        {
            var fisher = dist.FisherInformation();
            // Fisher information matrix should be symmetric
            Assert.Equal(fisher[0, 1], fisher[1, 0], Tolerance);
        }
    }

    [Fact]
    public void AllDistributions_FisherInformation_DiagonalPositive()
    {
        var distributions = new IParametricDistribution<double>[]
        {
            new NormalDistribution<double>(2, 3),
            new GammaDistribution<double>(3, 2),
            new BetaDistribution<double>(2, 5),
            new LaplaceDistribution<double>(1, 2),
            new LogNormalDistribution<double>(1, 0.5),
            new WeibullDistribution<double>(2, 3),
        };

        foreach (var dist in distributions)
        {
            var fisher = dist.FisherInformation();
            // Diagonal elements of Fisher information should be positive
            Assert.True(fisher[0, 0] > 0,
                $"{dist.GetType().Name} Fisher I[0,0] = {fisher[0, 0]} should be positive");
            Assert.True(fisher[1, 1] > 0,
                $"{dist.GetType().Name} Fisher I[1,1] = {fisher[1, 1]} should be positive");
        }
    }

    [Fact]
    public void AllDistributions_Clone_ProducesSamePdf()
    {
        var distributions = new IParametricDistribution<double>[]
        {
            new NormalDistribution<double>(2, 3),
            new GammaDistribution<double>(3, 2),
            new BetaDistribution<double>(2, 5),
            new ExponentialDistribution<double>(2),
            new LaplaceDistribution<double>(1, 2),
            new LogNormalDistribution<double>(1, 0.5),
            new WeibullDistribution<double>(2, 3),
        };

        foreach (var dist in distributions)
        {
            var clone = dist.Clone();
            double x = 1.5;
            Assert.Equal(dist.Pdf(x), clone.Pdf(x), Tolerance);
            Assert.Equal(dist.Cdf(x), clone.Cdf(x), Tolerance);
            Assert.Equal(dist.Mean, clone.Mean, Tolerance);
            Assert.Equal(dist.Variance, clone.Variance, Tolerance);
        }
    }

    [Fact]
    public void AllDistributions_ParameterSetGet_RoundTrip()
    {
        var distributions = new IParametricDistribution<double>[]
        {
            new NormalDistribution<double>(2, 3),
            new GammaDistribution<double>(3, 2),
            new BetaDistribution<double>(2, 5),
            new LaplaceDistribution<double>(1, 2),
            new LogNormalDistribution<double>(1, 0.5),
            new WeibullDistribution<double>(2, 3),
        };

        foreach (var dist in distributions)
        {
            var original = dist.Parameters;
            var clone = dist.Clone();
            clone.Parameters = original;

            for (int i = 0; i < original.Length; i++)
            {
                Assert.Equal(original[i], clone.Parameters[i], Tolerance);
            }
        }
    }

    [Fact]
    public void NegativeBinomial_MeanFormula()
    {
        // Mean = r*(1-p)/p
        double r = 5.0, p = 0.4;
        var dist = new NegativeBinomialDistribution<double>(r, p);
        double expected = r * (1 - p) / p;
        Assert.Equal(expected, dist.Mean, Tolerance);
    }

    [Fact]
    public void NegativeBinomial_VarianceFormula()
    {
        // Variance = r*(1-p)/p^2
        double r = 5.0, p = 0.4;
        var dist = new NegativeBinomialDistribution<double>(r, p);
        double expected = r * (1 - p) / (p * p);
        Assert.Equal(expected, dist.Variance, Tolerance);
    }

    #endregion

    #region Edge Cases and Validation

    [Fact]
    public void Normal_StdDev_IsSqrtVariance()
    {
        var dist = new NormalDistribution<double>(0, 4.0);
        Assert.Equal(2.0, dist.StdDev, Tolerance);
    }

    [Fact]
    public void Gamma_Scale_IsOneOverRate()
    {
        var dist = new GammaDistribution<double>(2.0, 4.0);
        Assert.Equal(0.25, ((GammaDistribution<double>)dist).Scale, Tolerance);
    }

    [Fact]
    public void AllDistributions_SampleBatch_ReturnsCorrectCount()
    {
        var dist = new NormalDistribution<double>(0, 1);
        var rng = RandomHelper.CreateSeededRandom(42);
        var samples = dist.Sample(rng, 100);
        Assert.Equal(100, samples.Length);
    }

    [Fact]
    public void AllDistributions_SampleBatch_InvalidCount_Throws()
    {
        var dist = new NormalDistribution<double>(0, 1);
        var rng = RandomHelper.CreateSeededRandom(42);
        Assert.Throws<ArgumentOutOfRangeException>(() => dist.Sample(rng, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => dist.Sample(rng, -1));
    }

    [Fact]
    public void AllDistributions_ParameterCount_MatchesNames()
    {
        var distributions = new IParametricDistribution<double>[]
        {
            new NormalDistribution<double>(0, 1),
            new GammaDistribution<double>(2, 1),
            new BetaDistribution<double>(2, 3),
            new ExponentialDistribution<double>(1),
            new LaplaceDistribution<double>(0, 1),
            new LogNormalDistribution<double>(0, 1),
            new WeibullDistribution<double>(2, 1),
            new StudentTDistribution<double>(5),
        };

        foreach (var dist in distributions)
        {
            Assert.Equal(dist.NumParameters, dist.ParameterNames.Length);
            Assert.Equal(dist.NumParameters, dist.Parameters.Length);
        }
    }

    #endregion
}
