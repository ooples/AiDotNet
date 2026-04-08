using AiDotNet.Distributions;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Distributions;

/// <summary>
/// Deep mathematical correctness tests for probability distributions.
/// Verifies PDFs, CDFs, gradients, Fisher information, and cross-distribution identities
/// against hand-calculated values and numerical methods.
/// </summary>
public class DistributionsDeepMathIntegrationTests
{
    private const double Tolerance = 1e-6;
    private const double LooseTolerance = 1e-4;

    // Helper: numerical gradient of LogPdf w.r.t. parameters
    private static double[] NumericalGradLogPdf<T>(DistributionBase<T> dist, T x, double h = 1e-6)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var origParams = dist.Parameters;
        int n = dist.NumParameters;
        double[] grad = new double[n];

        for (int i = 0; i < n; i++)
        {
            // Forward
            var paramsPlus = new Vector<T>(origParams.Length);
            for (int j = 0; j < n; j++)
                paramsPlus[j] = origParams[j];
            paramsPlus[i] = numOps.Add(paramsPlus[i], numOps.FromDouble(h));

            // Backward
            var paramsMinus = new Vector<T>(origParams.Length);
            for (int j = 0; j < n; j++)
                paramsMinus[j] = origParams[j];
            paramsMinus[i] = numOps.Subtract(paramsMinus[i], numOps.FromDouble(h));

            try
            {
                dist.Parameters = paramsPlus;
                double fPlus = numOps.ToDouble(dist.LogPdf(x));

                dist.Parameters = paramsMinus;
                double fMinus = numOps.ToDouble(dist.LogPdf(x));

                grad[i] = (fPlus - fMinus) / (2 * h);
            }
            finally
            {
                dist.Parameters = origParams;
            }
        }

        return grad;
    }

    // ============================================================
    //  NORMAL DISTRIBUTION
    // ============================================================

    [Fact]
    public void Normal_Pdf_AtMean_EqualsExpectedPeak()
    {
        // PDF at mean = 1 / (sigma * sqrt(2*pi))
        var dist = new NormalDistribution<double>(3.0, 4.0); // mean=3, variance=4 => sigma=2
        double pdf = dist.Pdf(3.0);
        double expected = 1.0 / (2.0 * Math.Sqrt(2.0 * Math.PI)); // 0.19947...
        Assert.Equal(expected, pdf, Tolerance);
    }

    [Fact]
    public void Normal_Pdf_Symmetry()
    {
        var dist = new NormalDistribution<double>(5.0, 9.0); // mean=5, variance=9
        double left = dist.Pdf(2.0);  // 5 - 3
        double right = dist.Pdf(8.0); // 5 + 3
        Assert.Equal(left, right, Tolerance);
    }

    [Fact]
    public void Normal_Cdf_AtMean_IsHalf()
    {
        var dist = new NormalDistribution<double>(5.0, 2.0);
        double cdf = dist.Cdf(5.0);
        Assert.Equal(0.5, cdf, Tolerance);
    }

    [Fact]
    public void Normal_StandardCdf_At196_Approximately975()
    {
        var dist = new NormalDistribution<double>(0.0, 1.0);
        double cdf = dist.Cdf(1.96);
        Assert.Equal(0.975, cdf, 1e-3); // Erf approximation has ~1.5e-7 max error
    }

    [Fact]
    public void Normal_LogPdf_ConsistentWithPdf()
    {
        var dist = new NormalDistribution<double>(2.0, 3.0);
        double x = 4.5;
        double logPdf = dist.LogPdf(x);
        double logOfPdf = Math.Log(dist.Pdf(x));
        Assert.Equal(logOfPdf, logPdf, Tolerance);
    }

    [Fact]
    public void Normal_InverseCdf_Roundtrip()
    {
        var dist = new NormalDistribution<double>(1.0, 4.0);
        double[] probs = { 0.1, 0.25, 0.5, 0.75, 0.9 };
        foreach (double p in probs)
        {
            double x = dist.InverseCdf(p);
            double roundtrip = dist.Cdf(x);
            Assert.Equal(p, roundtrip, LooseTolerance);
        }
    }

    [Fact]
    public void Normal_GradLogPdf_MatchesNumerical()
    {
        var dist = new NormalDistribution<double>(2.0, 3.0);
        double x = 4.5;
        var analyticalGrad = dist.GradLogPdf(x);
        double[] numericalGrad = NumericalGradLogPdf(dist, x);

        Assert.Equal(numericalGrad[0], analyticalGrad[0], LooseTolerance); // grad w.r.t. mean
        Assert.Equal(numericalGrad[1], analyticalGrad[1], LooseTolerance); // grad w.r.t. variance
    }

    [Fact]
    public void Normal_FisherInformation_HandValues()
    {
        // Fisher Information for Normal(mean, variance):
        // I = [[1/variance, 0], [0, 1/(2*variance^2)]]
        var dist = new NormalDistribution<double>(0.0, 4.0); // variance=4
        var fisher = dist.FisherInformation();

        Assert.Equal(0.25, fisher[0, 0], Tolerance);  // 1/4
        Assert.Equal(0.0, fisher[0, 1], Tolerance);
        Assert.Equal(0.0, fisher[1, 0], Tolerance);
        Assert.Equal(0.03125, fisher[1, 1], Tolerance); // 1/(2*16) = 1/32
    }

    [Fact]
    public void Normal_Pdf_IntegratesToOne()
    {
        // Numerical integration via trapezoidal rule
        var dist = new NormalDistribution<double>(2.0, 3.0);
        double sum = 0;
        double lo = -20.0, hi = 24.0;
        int steps = 10000;
        double dx = (hi - lo) / steps;

        for (int i = 0; i < steps; i++)
        {
            double x0 = lo + i * dx;
            double x1 = x0 + dx;
            sum += 0.5 * (dist.Pdf(x0) + dist.Pdf(x1)) * dx;
        }

        Assert.Equal(1.0, sum, 1e-4);
    }

    [Fact]
    public void Normal_CdfDerivative_MatchesPdf()
    {
        // d/dx CDF(x) = PDF(x)
        var dist = new NormalDistribution<double>(1.0, 2.0);
        double x = 3.0;
        double h = 1e-6;
        double numericalDerivative = (dist.Cdf(x + h) - dist.Cdf(x - h)) / (2 * h);
        double pdf = dist.Pdf(x);
        Assert.Equal(pdf, numericalDerivative, 1e-4);
    }

    [Fact]
    public void Normal_Mean_And_Variance_MatchParameters()
    {
        var dist = new NormalDistribution<double>(5.0, 7.0);
        Assert.Equal(5.0, dist.Mean, Tolerance);
        Assert.Equal(7.0, dist.Variance, Tolerance);
    }

    // ============================================================
    //  EXPONENTIAL DISTRIBUTION
    // ============================================================

    [Fact]
    public void Exponential_Pdf_HandValues()
    {
        // PDF(x) = rate * exp(-rate * x)
        var dist = new ExponentialDistribution<double>(2.0);
        Assert.Equal(2.0, dist.Pdf(0.0), Tolerance);   // rate * e^0 = 2
        Assert.Equal(2.0 * Math.Exp(-2.0), dist.Pdf(1.0), Tolerance); // 2*e^(-2) = 0.27067...
    }

    [Fact]
    public void Exponential_Cdf_HandValues()
    {
        var dist = new ExponentialDistribution<double>(2.0);
        Assert.Equal(0.0, dist.Cdf(0.0), Tolerance);
        Assert.Equal(1.0 - Math.Exp(-2.0), dist.Cdf(1.0), Tolerance);
        Assert.Equal(1.0 - Math.Exp(-4.0), dist.Cdf(2.0), Tolerance);
    }

    [Fact]
    public void Exponential_Memoryless_Property()
    {
        // P(X > s+t | X > s) = P(X > t)
        // Equivalently: 1 - CDF(s+t) = (1 - CDF(s)) * (1 - CDF(t))
        var dist = new ExponentialDistribution<double>(1.5);
        double s = 2.0, t = 3.0;
        double survivalSPlusT = 1.0 - dist.Cdf(s + t);
        double survivalS = 1.0 - dist.Cdf(s);
        double survivalT = 1.0 - dist.Cdf(t);
        Assert.Equal(survivalSPlusT, survivalS * survivalT, Tolerance);
    }

    [Fact]
    public void Exponential_Mean_And_Variance()
    {
        var dist = new ExponentialDistribution<double>(3.0);
        Assert.Equal(1.0 / 3.0, dist.Mean, Tolerance);
        Assert.Equal(1.0 / 9.0, dist.Variance, Tolerance);
    }

    [Fact]
    public void Exponential_LogPdf_ConsistentWithPdf()
    {
        var dist = new ExponentialDistribution<double>(2.5);
        double x = 1.0;
        Assert.Equal(Math.Log(dist.Pdf(x)), dist.LogPdf(x), Tolerance);
    }

    [Fact]
    public void Exponential_InverseCdf_Roundtrip()
    {
        var dist = new ExponentialDistribution<double>(2.0);
        double[] probs = { 0.1, 0.25, 0.5, 0.75, 0.9 };
        foreach (double p in probs)
        {
            double x = dist.InverseCdf(p);
            double roundtrip = dist.Cdf(x);
            Assert.Equal(p, roundtrip, Tolerance);
        }
    }

    [Fact]
    public void Exponential_GradLogPdf_MatchesNumerical()
    {
        var dist = new ExponentialDistribution<double>(2.0);
        double x = 1.5;
        var analyticalGrad = dist.GradLogPdf(x);
        double[] numericalGrad = NumericalGradLogPdf(dist, x);
        Assert.Equal(numericalGrad[0], analyticalGrad[0], LooseTolerance);
    }

    // ============================================================
    //  GAMMA DISTRIBUTION
    // ============================================================

    [Fact]
    public void Gamma_Shape1_EqualsExponential()
    {
        // Gamma(1, rate) = Exponential(rate)
        var gamma = new GammaDistribution<double>(1.0, 2.0);
        var expo = new ExponentialDistribution<double>(2.0);

        double[] testPoints = { 0.5, 1.0, 2.0, 3.0 };
        foreach (double x in testPoints)
        {
            Assert.Equal(expo.Pdf(x), gamma.Pdf(x), Tolerance);
            Assert.Equal(expo.Cdf(x), gamma.Cdf(x), LooseTolerance);
        }
    }

    [Fact]
    public void Gamma_Mean_And_Variance()
    {
        var dist = new GammaDistribution<double>(3.0, 2.0); // shape=3, rate=2
        Assert.Equal(1.5, dist.Mean, Tolerance); // 3/2
        Assert.Equal(0.75, dist.Variance, Tolerance); // 3/4
    }

    [Fact]
    public void Gamma_Pdf_HandValue()
    {
        // Gamma(2, 1) at x=1: PDF = 1^(2-1) * e^(-1) / Gamma(2) = 1 * e^(-1) / 1 = e^(-1)
        var dist = new GammaDistribution<double>(2.0, 1.0);
        double pdf = dist.Pdf(1.0);
        Assert.Equal(Math.Exp(-1.0), pdf, Tolerance); // 0.36788...
    }

    [Fact]
    public void Gamma_LogPdf_ConsistentWithPdf()
    {
        var dist = new GammaDistribution<double>(3.0, 2.0);
        double x = 2.0;
        Assert.Equal(Math.Log(dist.Pdf(x)), dist.LogPdf(x), Tolerance);
    }

    [Fact]
    public void Gamma_Pdf_IntegratesToOne()
    {
        var dist = new GammaDistribution<double>(3.0, 2.0);
        double sum = 0;
        double lo = 0.001, hi = 15.0;
        int steps = 10000;
        double dx = (hi - lo) / steps;

        for (int i = 0; i < steps; i++)
        {
            double x0 = lo + i * dx;
            double x1 = x0 + dx;
            sum += 0.5 * (dist.Pdf(x0) + dist.Pdf(x1)) * dx;
        }

        Assert.Equal(1.0, sum, 1e-3);
    }

    [Fact]
    public void Gamma_GradLogPdf_MatchesNumerical()
    {
        var dist = new GammaDistribution<double>(3.0, 2.0);
        double x = 2.0;
        var analyticalGrad = dist.GradLogPdf(x);
        double[] numericalGrad = NumericalGradLogPdf(dist, x);

        Assert.Equal(numericalGrad[0], analyticalGrad[0], LooseTolerance); // shape
        Assert.Equal(numericalGrad[1], analyticalGrad[1], LooseTolerance); // rate
    }

    [Fact]
    public void Gamma_InverseCdf_Roundtrip()
    {
        var dist = new GammaDistribution<double>(2.0, 1.0);
        double[] probs = { 0.1, 0.5, 0.9 };
        foreach (double p in probs)
        {
            double x = dist.InverseCdf(p);
            double roundtrip = dist.Cdf(x);
            Assert.Equal(p, roundtrip, 1e-3);
        }
    }

    // ============================================================
    //  BETA DISTRIBUTION
    // ============================================================

    [Fact]
    public void Beta_Uniform_When_Alpha1_Beta1()
    {
        var dist = new BetaDistribution<double>(1.0, 1.0);
        // Beta(1,1) = Uniform[0,1], PDF = 1 everywhere
        double[] testPoints = { 0.1, 0.3, 0.5, 0.7, 0.9 };
        foreach (double x in testPoints)
        {
            Assert.Equal(1.0, dist.Pdf(x), Tolerance);
        }
    }

    [Fact]
    public void Beta_Cdf_LinearFor_Alpha1_Beta1()
    {
        var dist = new BetaDistribution<double>(1.0, 1.0);
        // CDF of Uniform = x
        double[] testPoints = { 0.1, 0.3, 0.5, 0.7, 0.9 };
        foreach (double x in testPoints)
        {
            Assert.Equal(x, dist.Cdf(x), 1e-3);
        }
    }

    [Fact]
    public void Beta_Mean_HandValue()
    {
        var dist = new BetaDistribution<double>(2.0, 5.0);
        Assert.Equal(2.0 / 7.0, dist.Mean, Tolerance); // alpha/(alpha+beta)
    }

    [Fact]
    public void Beta_Variance_HandValue()
    {
        // Var = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))
        var dist = new BetaDistribution<double>(2.0, 5.0);
        double expected = (2.0 * 5.0) / (49.0 * 8.0); // 10/392 = 0.025510...
        Assert.Equal(expected, dist.Variance, Tolerance);
    }

    [Fact]
    public void Beta_Symmetry()
    {
        // Beta(a,b).PDF(x) = Beta(b,a).PDF(1-x)
        var distAB = new BetaDistribution<double>(2.0, 5.0);
        var distBA = new BetaDistribution<double>(5.0, 2.0);

        double[] testPoints = { 0.1, 0.3, 0.5, 0.7 };
        foreach (double x in testPoints)
        {
            Assert.Equal(distAB.Pdf(x), distBA.Pdf(1.0 - x), Tolerance);
        }
    }

    [Fact]
    public void Beta_Pdf_IntegratesToOne()
    {
        var dist = new BetaDistribution<double>(2.0, 3.0);
        double sum = 0;
        int steps = 10000;
        double dx = 1.0 / steps;

        for (int i = 0; i < steps; i++)
        {
            double x0 = (i + 0.5) * dx; // midpoint to avoid 0 and 1
            double x1 = (i + 1.5) * dx;
            if (x1 > 1.0) x1 = 1.0 - 1e-10;
            sum += dist.Pdf(x0) * dx;
        }

        Assert.Equal(1.0, sum, 1e-3);
    }

    [Fact]
    public void Beta_GradLogPdf_MatchesNumerical()
    {
        var dist = new BetaDistribution<double>(3.0, 4.0);
        double x = 0.4;
        var analyticalGrad = dist.GradLogPdf(x);
        double[] numericalGrad = NumericalGradLogPdf(dist, x);

        Assert.Equal(numericalGrad[0], analyticalGrad[0], LooseTolerance); // alpha
        Assert.Equal(numericalGrad[1], analyticalGrad[1], LooseTolerance); // beta
    }

    [Fact]
    public void Beta_InverseCdf_Roundtrip()
    {
        var dist = new BetaDistribution<double>(2.0, 5.0);
        double[] probs = { 0.1, 0.5, 0.9 };
        foreach (double p in probs)
        {
            double x = dist.InverseCdf(p);
            double roundtrip = dist.Cdf(x);
            Assert.Equal(p, roundtrip, 1e-3);
        }
    }

    [Fact]
    public void Beta_LogPdf_ConsistentWithPdf()
    {
        var dist = new BetaDistribution<double>(3.0, 2.0);
        double x = 0.6;
        Assert.Equal(Math.Log(dist.Pdf(x)), dist.LogPdf(x), Tolerance);
    }

    // ============================================================
    //  LAPLACE DISTRIBUTION
    // ============================================================

    [Fact]
    public void Laplace_Pdf_AtLocation_EqualsExpectedPeak()
    {
        // PDF at location = 1/(2*scale)
        var dist = new LaplaceDistribution<double>(3.0, 2.0);
        Assert.Equal(0.25, dist.Pdf(3.0), Tolerance); // 1/(2*2)
    }

    [Fact]
    public void Laplace_Pdf_Symmetry()
    {
        var dist = new LaplaceDistribution<double>(3.0, 2.0);
        Assert.Equal(dist.Pdf(1.0), dist.Pdf(5.0), Tolerance); // symmetric around 3
    }

    [Fact]
    public void Laplace_Cdf_AtLocation_IsHalf()
    {
        var dist = new LaplaceDistribution<double>(5.0, 3.0);
        Assert.Equal(0.5, dist.Cdf(5.0), Tolerance);
    }

    [Fact]
    public void Laplace_Mean_And_Variance()
    {
        var dist = new LaplaceDistribution<double>(3.0, 2.0);
        Assert.Equal(3.0, dist.Mean, Tolerance);
        Assert.Equal(8.0, dist.Variance, Tolerance); // 2 * scale^2 = 2*4 = 8
    }

    [Fact]
    public void Laplace_Cdf_HandValues()
    {
        // CDF: x < location: 0.5 * exp((x-loc)/scale)
        // CDF: x >= location: 1 - 0.5 * exp(-(x-loc)/scale)
        var dist = new LaplaceDistribution<double>(0.0, 1.0);
        Assert.Equal(0.5 * Math.Exp(-2.0), dist.Cdf(-2.0), Tolerance);
        Assert.Equal(1.0 - 0.5 * Math.Exp(-2.0), dist.Cdf(2.0), Tolerance);
    }

    [Fact]
    public void Laplace_LogPdf_ConsistentWithPdf()
    {
        var dist = new LaplaceDistribution<double>(1.0, 2.0);
        double x = 4.0;
        Assert.Equal(Math.Log(dist.Pdf(x)), dist.LogPdf(x), Tolerance);
    }

    [Fact]
    public void Laplace_InverseCdf_Roundtrip()
    {
        var dist = new LaplaceDistribution<double>(2.0, 3.0);
        double[] probs = { 0.1, 0.25, 0.5, 0.75, 0.9 };
        foreach (double p in probs)
        {
            double x = dist.InverseCdf(p);
            double roundtrip = dist.Cdf(x);
            Assert.Equal(p, roundtrip, Tolerance);
        }
    }

    [Fact]
    public void Laplace_GradLogPdf_MatchesNumerical()
    {
        var dist = new LaplaceDistribution<double>(2.0, 3.0);
        double x = 5.0;
        var analyticalGrad = dist.GradLogPdf(x);
        double[] numericalGrad = NumericalGradLogPdf(dist, x);

        Assert.Equal(numericalGrad[0], analyticalGrad[0], LooseTolerance);
        Assert.Equal(numericalGrad[1], analyticalGrad[1], LooseTolerance);
    }

    [Fact]
    public void Laplace_FisherInformation_HandValues()
    {
        // I = [[1/b^2, 0], [0, 1/b^2]]
        var dist = new LaplaceDistribution<double>(0.0, 3.0);
        var fisher = dist.FisherInformation();
        Assert.Equal(1.0 / 9.0, fisher[0, 0], Tolerance);
        Assert.Equal(0.0, fisher[0, 1], Tolerance);
        Assert.Equal(0.0, fisher[1, 0], Tolerance);
        Assert.Equal(1.0 / 9.0, fisher[1, 1], Tolerance);
    }

    // ============================================================
    //  POISSON DISTRIBUTION
    // ============================================================

    [Fact]
    public void Poisson_Pmf_HandValues()
    {
        // P(X=k) = lambda^k * e^(-lambda) / k!
        var dist = new PoissonDistribution<double>(3.0);

        // P(X=0) = e^(-3) ≈ 0.0498
        Assert.Equal(Math.Exp(-3.0), dist.Pmf(0), Tolerance);

        // P(X=1) = 3 * e^(-3) ≈ 0.1494
        Assert.Equal(3.0 * Math.Exp(-3.0), dist.Pmf(1), Tolerance);

        // P(X=2) = 9/2 * e^(-3) ≈ 0.2240
        Assert.Equal(4.5 * Math.Exp(-3.0), dist.Pmf(2), Tolerance);

        // P(X=3) = 27/6 * e^(-3) = 4.5 * e^(-3) ≈ 0.2240
        Assert.Equal(27.0 / 6.0 * Math.Exp(-3.0), dist.Pmf(3), Tolerance);
    }

    [Fact]
    public void Poisson_Pmf_SumsToOne()
    {
        var dist = new PoissonDistribution<double>(5.0);
        double sum = 0;
        for (int k = 0; k < 50; k++)
        {
            sum += dist.Pmf(k);
        }
        Assert.Equal(1.0, sum, 1e-6);
    }

    [Fact]
    public void Poisson_Mean_Equals_Variance()
    {
        var dist = new PoissonDistribution<double>(7.5);
        Assert.Equal(dist.Mean, dist.Variance, Tolerance);
    }

    [Fact]
    public void Poisson_Cdf_Consistency()
    {
        // CDF(k) = sum_{i=0}^{k} PMF(i)
        var dist = new PoissonDistribution<double>(4.0);
        for (int k = 0; k <= 10; k++)
        {
            double cdfValue = dist.Cdf(k);
            double manualSum = 0;
            for (int i = 0; i <= k; i++)
            {
                manualSum += dist.Pmf(i);
            }
            Assert.Equal(manualSum, cdfValue, 1e-4);
        }
    }

    [Fact]
    public void Poisson_GradLogPdf_MatchesNumerical()
    {
        var dist = new PoissonDistribution<double>(4.0);
        double x = 3.0; // k=3
        var analyticalGrad = dist.GradLogPdf(x);
        double[] numericalGrad = NumericalGradLogPdf(dist, x);
        Assert.Equal(numericalGrad[0], analyticalGrad[0], LooseTolerance);
    }

    // ============================================================
    //  WEIBULL DISTRIBUTION
    // ============================================================

    [Fact]
    public void Weibull_Shape1_EqualsExponential()
    {
        // Weibull(1, scale) = Exponential(1/scale)
        var weibull = new WeibullDistribution<double>(1.0, 2.0);
        var expo = new ExponentialDistribution<double>(0.5); // rate = 1/scale

        double[] testPoints = { 0.5, 1.0, 2.0, 4.0 };
        foreach (double x in testPoints)
        {
            Assert.Equal(expo.Pdf(x), weibull.Pdf(x), Tolerance);
            Assert.Equal(expo.Cdf(x), weibull.Cdf(x), Tolerance);
        }
    }

    [Fact]
    public void Weibull_Cdf_HandValues()
    {
        // CDF = 1 - exp(-(x/scale)^shape)
        var dist = new WeibullDistribution<double>(2.0, 3.0);
        double x = 3.0;
        double expected = 1.0 - Math.Exp(-Math.Pow(3.0 / 3.0, 2.0)); // 1 - e^(-1) ≈ 0.6321
        Assert.Equal(expected, dist.Cdf(x), Tolerance);
    }

    [Fact]
    public void Weibull_InverseCdf_Roundtrip()
    {
        var dist = new WeibullDistribution<double>(2.0, 3.0);
        double[] probs = { 0.1, 0.5, 0.9 };
        foreach (double p in probs)
        {
            double x = dist.InverseCdf(p);
            double roundtrip = dist.Cdf(x);
            Assert.Equal(p, roundtrip, Tolerance);
        }
    }

    [Fact]
    public void Weibull_Pdf_IntegratesToOne()
    {
        var dist = new WeibullDistribution<double>(2.0, 3.0);
        double sum = 0;
        double lo = 0.001, hi = 15.0;
        int steps = 10000;
        double dx = (hi - lo) / steps;

        for (int i = 0; i < steps; i++)
        {
            double x0 = lo + i * dx;
            double x1 = x0 + dx;
            sum += 0.5 * (dist.Pdf(x0) + dist.Pdf(x1)) * dx;
        }

        Assert.Equal(1.0, sum, 1e-3);
    }

    [Fact]
    public void Weibull_LogPdf_ConsistentWithPdf()
    {
        var dist = new WeibullDistribution<double>(2.0, 3.0);
        double x = 2.0;
        Assert.Equal(Math.Log(dist.Pdf(x)), dist.LogPdf(x), Tolerance);
    }

    [Fact]
    public void Weibull_GradLogPdf_MatchesNumerical()
    {
        var dist = new WeibullDistribution<double>(2.0, 3.0);
        double x = 2.5;
        var analyticalGrad = dist.GradLogPdf(x);
        double[] numericalGrad = NumericalGradLogPdf(dist, x);

        Assert.Equal(numericalGrad[0], analyticalGrad[0], LooseTolerance); // shape
        Assert.Equal(numericalGrad[1], analyticalGrad[1], LooseTolerance); // scale
    }

    // ============================================================
    //  STUDENT-T DISTRIBUTION
    // ============================================================

    [Fact]
    public void StudentT_Pdf_Symmetry()
    {
        var dist = new StudentTDistribution<double>(2.0, 1.0, 5.0);
        Assert.Equal(dist.Pdf(0.0), dist.Pdf(4.0), Tolerance); // symmetric around 2.0
        Assert.Equal(dist.Pdf(1.0), dist.Pdf(3.0), Tolerance);
    }

    [Fact]
    public void StudentT_Cdf_AtLocation_IsHalf()
    {
        var dist = new StudentTDistribution<double>(3.0, 2.0, 10.0);
        Assert.Equal(0.5, dist.Cdf(3.0), 1e-3);
    }

    [Fact]
    public void StudentT_HighDf_ApproachesNormal()
    {
        // With large df, Student-t approaches Normal
        var studentT = new StudentTDistribution<double>(0.0, 1.0, 1000.0);
        var normal = new NormalDistribution<double>(0.0, 1.0);

        double[] testPoints = { -2.0, -1.0, 0.0, 1.0, 2.0 };
        foreach (double x in testPoints)
        {
            Assert.Equal(normal.Pdf(x), studentT.Pdf(x), 1e-3);
        }
    }

    [Fact]
    public void StudentT_Mean_And_Variance()
    {
        var dist = new StudentTDistribution<double>(3.0, 2.0, 5.0);
        Assert.Equal(3.0, dist.Mean, Tolerance);
        // Variance = sigma^2 * nu / (nu - 2) = 4 * 5/3 ≈ 6.667
        Assert.Equal(4.0 * 5.0 / 3.0, dist.Variance, Tolerance);
    }

    [Fact]
    public void StudentT_LogPdf_ConsistentWithPdf()
    {
        var dist = new StudentTDistribution<double>(0.0, 1.0, 5.0);
        double x = 2.0;
        Assert.Equal(Math.Log(dist.Pdf(x)), dist.LogPdf(x), Tolerance);
    }

    [Fact]
    public void StudentT_GradLogPdf_MatchesNumerical()
    {
        var dist = new StudentTDistribution<double>(1.0, 2.0, 5.0);
        double x = 3.0;
        var analyticalGrad = dist.GradLogPdf(x);
        double[] numericalGrad = NumericalGradLogPdf(dist, x);

        Assert.Equal(numericalGrad[0], analyticalGrad[0], LooseTolerance); // location
        Assert.Equal(numericalGrad[1], analyticalGrad[1], LooseTolerance); // scale
        Assert.Equal(numericalGrad[2], analyticalGrad[2], LooseTolerance); // df
    }

    [Fact]
    public void StudentT_Pdf_IntegratesToOne()
    {
        var dist = new StudentTDistribution<double>(0.0, 1.0, 3.0);
        double sum = 0;
        double lo = -30.0, hi = 30.0;
        int steps = 10000;
        double dx = (hi - lo) / steps;

        for (int i = 0; i < steps; i++)
        {
            double x0 = lo + i * dx;
            double x1 = x0 + dx;
            sum += 0.5 * (dist.Pdf(x0) + dist.Pdf(x1)) * dx;
        }

        Assert.Equal(1.0, sum, 1e-3);
    }

    // ============================================================
    //  LOGNORMAL DISTRIBUTION
    // ============================================================

    [Fact]
    public void LogNormal_Mean_HandValue()
    {
        // Mean = exp(mu + sigma^2/2)
        var dist = new LogNormalDistribution<double>(1.0, 0.5);
        double expected = Math.Exp(1.0 + 0.25 / 2.0); // exp(1.125) ≈ 3.08022
        Assert.Equal(expected, dist.Mean, Tolerance);
    }

    [Fact]
    public void LogNormal_Variance_HandValue()
    {
        // Var = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
        var dist = new LogNormalDistribution<double>(1.0, 0.5);
        double sigma2 = 0.25;
        double expected = (Math.Exp(sigma2) - 1.0) * Math.Exp(2.0 + sigma2);
        Assert.Equal(expected, dist.Variance, Tolerance);
    }

    [Fact]
    public void LogNormal_Pdf_RelatedToNormal()
    {
        // LogNormal PDF(x) = NormalPDF(log(x)) / x
        var logNorm = new LogNormalDistribution<double>(0.0, 1.0);
        var normal = new NormalDistribution<double>(0.0, 1.0);

        double[] testPoints = { 0.5, 1.0, 2.0, 3.0 };
        foreach (double x in testPoints)
        {
            double expected = normal.Pdf(Math.Log(x)) / x;
            Assert.Equal(expected, logNorm.Pdf(x), Tolerance);
        }
    }

    [Fact]
    public void LogNormal_Cdf_RelatedToNormalCdf()
    {
        // LogNormal CDF(x) = Normal_CDF((log(x) - mu) / sigma)
        var logNorm = new LogNormalDistribution<double>(1.0, 0.5);
        var stdNormal = new NormalDistribution<double>(0.0, 1.0);

        double x = 3.0;
        double z = (Math.Log(x) - 1.0) / 0.5;
        Assert.Equal(stdNormal.Cdf(z), logNorm.Cdf(x), LooseTolerance);
    }

    [Fact]
    public void LogNormal_LogPdf_ConsistentWithPdf()
    {
        var dist = new LogNormalDistribution<double>(0.0, 1.0);
        double x = 2.0;
        Assert.Equal(Math.Log(dist.Pdf(x)), dist.LogPdf(x), Tolerance);
    }

    [Fact]
    public void LogNormal_InverseCdf_Roundtrip()
    {
        var dist = new LogNormalDistribution<double>(0.0, 1.0);
        double[] probs = { 0.1, 0.5, 0.9 };
        foreach (double p in probs)
        {
            double x = dist.InverseCdf(p);
            double roundtrip = dist.Cdf(x);
            Assert.Equal(p, roundtrip, 1e-3);
        }
    }

    [Fact]
    public void LogNormal_GradLogPdf_MatchesNumerical()
    {
        var dist = new LogNormalDistribution<double>(1.0, 0.5);
        double x = 3.0;
        var analyticalGrad = dist.GradLogPdf(x);
        double[] numericalGrad = NumericalGradLogPdf(dist, x);

        Assert.Equal(numericalGrad[0], analyticalGrad[0], LooseTolerance); // mu
        Assert.Equal(numericalGrad[1], analyticalGrad[1], LooseTolerance); // sigma
    }

    [Fact]
    public void LogNormal_Pdf_IntegratesToOne()
    {
        var dist = new LogNormalDistribution<double>(0.0, 1.0);
        double sum = 0;
        double lo = 0.001, hi = 20.0;
        int steps = 10000;
        double dx = (hi - lo) / steps;

        for (int i = 0; i < steps; i++)
        {
            double x0 = lo + i * dx;
            double x1 = x0 + dx;
            sum += 0.5 * (dist.Pdf(x0) + dist.Pdf(x1)) * dx;
        }

        Assert.Equal(1.0, sum, 1e-2); // LogNormal has heavy tail
    }

    // ============================================================
    //  CROSS-DISTRIBUTION IDENTITIES
    // ============================================================

    [Fact]
    public void CrossDistribution_NormalVariance1_CdfSymmetry()
    {
        // For standard Normal: CDF(x) + CDF(-x) = 1
        var dist = new NormalDistribution<double>(0.0, 1.0);
        double[] testPoints = { 0.5, 1.0, 1.5, 2.0, 3.0 };
        foreach (double x in testPoints)
        {
            double sum = dist.Cdf(x) + dist.Cdf(-x);
            Assert.Equal(1.0, sum, LooseTolerance);
        }
    }

    [Fact]
    public void CrossDistribution_ExponentialSurvivalFunction()
    {
        // For Exponential: S(x) = 1 - CDF(x) = exp(-rate*x)
        var dist = new ExponentialDistribution<double>(2.0);
        double[] testPoints = { 0.5, 1.0, 2.0, 5.0 };
        foreach (double x in testPoints)
        {
            double survival = 1.0 - dist.Cdf(x);
            double expected = Math.Exp(-2.0 * x);
            Assert.Equal(expected, survival, Tolerance);
        }
    }

    [Fact]
    public void CrossDistribution_GammaScaling()
    {
        // If X ~ Gamma(shape, rate), then cX ~ Gamma(shape, rate/c)
        // So Gamma(3,2).CDF(x) should relate to Gamma(3,1).CDF(x*2/1) kind of...
        // Actually: P(X <= x) where X ~ Gamma(3,2) = P(Y <= 2x) where Y ~ Gamma(3,1)
        var gamma1 = new GammaDistribution<double>(3.0, 2.0);
        var gamma2 = new GammaDistribution<double>(3.0, 1.0);

        double x = 1.5;
        // Gamma(3,2) CDF at x should equal Gamma(3,1) CDF at 2x
        // Because if X~Gamma(3,2), Y=2X~Gamma(3,1)
        // P(X<=1.5) = P(Y<=3.0)
        Assert.Equal(gamma1.Cdf(x), gamma2.Cdf(2.0 * x), 1e-3);
    }

    [Fact]
    public void CrossDistribution_LaplaceCdf_HandVerification()
    {
        // Standard Laplace CDF at x:
        // x < 0: 0.5 * exp(x)
        // x >= 0: 1 - 0.5 * exp(-x)
        var dist = new LaplaceDistribution<double>(0.0, 1.0);

        Assert.Equal(0.5 * Math.Exp(-1.0), dist.Cdf(-1.0), Tolerance);
        Assert.Equal(0.5, dist.Cdf(0.0), Tolerance);
        Assert.Equal(1.0 - 0.5 * Math.Exp(-1.0), dist.Cdf(1.0), Tolerance);
    }

    // ============================================================
    //  FISHER INFORMATION - POSITIVE DEFINITENESS
    // ============================================================

    [Fact]
    public void FisherInformation_AllDistributions_PositiveDiagonal()
    {
        // Fisher information diagonal entries should be positive
        var distributions = new (string name, Func<Matrix<double>> getFisher)[]
        {
            ("Normal", () => new NormalDistribution<double>(0.0, 2.0).FisherInformation()),
            ("Exponential", () => new ExponentialDistribution<double>(2.0).FisherInformation()),
            ("Gamma", () => new GammaDistribution<double>(3.0, 2.0).FisherInformation()),
            ("Beta", () => new BetaDistribution<double>(2.0, 3.0).FisherInformation()),
            ("Laplace", () => new LaplaceDistribution<double>(0.0, 2.0).FisherInformation()),
            ("Weibull", () => new WeibullDistribution<double>(2.0, 3.0).FisherInformation()),
            ("LogNormal", () => new LogNormalDistribution<double>(0.0, 1.0).FisherInformation()),
        };

        foreach (var (name, getFisher) in distributions)
        {
            var fisher = getFisher();
            for (int i = 0; i < fisher.Rows; i++)
            {
                Assert.True(fisher[i, i] > 0, $"{name} Fisher[{i},{i}] should be positive, got {fisher[i, i]}");
            }
        }
    }

    [Fact]
    public void FisherInformation_Symmetric()
    {
        // Fisher information should be symmetric
        var distributions = new (string name, Func<Matrix<double>> getFisher)[]
        {
            ("Normal", () => new NormalDistribution<double>(0.0, 2.0).FisherInformation()),
            ("Gamma", () => new GammaDistribution<double>(3.0, 2.0).FisherInformation()),
            ("Beta", () => new BetaDistribution<double>(2.0, 3.0).FisherInformation()),
            ("Weibull", () => new WeibullDistribution<double>(2.0, 3.0).FisherInformation()),
        };

        foreach (var (name, getFisher) in distributions)
        {
            var fisher = getFisher();
            for (int i = 0; i < fisher.Rows; i++)
            {
                for (int j = i + 1; j < fisher.Columns; j++)
                {
                    Assert.Equal(fisher[i, j], fisher[j, i], Tolerance);
                }
            }
        }
    }

    // ============================================================
    //  EDGE CASES AND NUMERICAL STABILITY
    // ============================================================

    [Fact]
    public void Normal_Cdf_ExtremeValues()
    {
        var dist = new NormalDistribution<double>(0.0, 1.0);
        // CDF at very negative should be near 0
        double cdfLeft = dist.Cdf(-6.0);
        Assert.True(cdfLeft > 0 && cdfLeft < 0.01, $"CDF(-6) = {cdfLeft} should be near 0");

        // CDF at very positive should be near 1
        double cdfRight = dist.Cdf(6.0);
        Assert.True(cdfRight > 0.99 && cdfRight <= 1.0, $"CDF(6) = {cdfRight} should be near 1");
    }

    [Fact]
    public void Exponential_Pdf_NegativeInput_IsZero()
    {
        var dist = new ExponentialDistribution<double>(2.0);
        Assert.Equal(0.0, dist.Pdf(-1.0), Tolerance);
        Assert.Equal(0.0, dist.Cdf(-1.0), Tolerance);
    }

    [Fact]
    public void Gamma_Pdf_NegativeInput_IsZero()
    {
        var dist = new GammaDistribution<double>(3.0, 2.0);
        Assert.Equal(0.0, dist.Pdf(-1.0), Tolerance);
    }

    [Fact]
    public void Beta_Pdf_OutOfRange_IsZero()
    {
        var dist = new BetaDistribution<double>(2.0, 3.0);
        Assert.Equal(0.0, dist.Pdf(-0.5), Tolerance);
        Assert.Equal(0.0, dist.Pdf(1.5), Tolerance);
    }

    [Fact]
    public void LogNormal_Pdf_NegativeInput_IsZero()
    {
        var dist = new LogNormalDistribution<double>(0.0, 1.0);
        Assert.Equal(0.0, dist.Pdf(-1.0), Tolerance);
    }

    // ============================================================
    //  CDF MONOTONICITY
    // ============================================================

    [Fact]
    public void AllContinuousDistributions_Cdf_IsMonotone()
    {
        var distributions = new (string name, Func<double, double> cdf, double lo, double hi)[]
        {
            ("Normal", x => new NormalDistribution<double>(0.0, 1.0).Cdf(x), -5.0, 5.0),
            ("Exponential", x => new ExponentialDistribution<double>(2.0).Cdf(x), 0.0, 5.0),
            ("Gamma", x => new GammaDistribution<double>(3.0, 2.0).Cdf(x), 0.01, 8.0),
            ("Beta", x => new BetaDistribution<double>(2.0, 3.0).Cdf(x), 0.01, 0.99),
            ("Laplace", x => new LaplaceDistribution<double>(0.0, 1.0).Cdf(x), -5.0, 5.0),
            ("Weibull", x => new WeibullDistribution<double>(2.0, 3.0).Cdf(x), 0.01, 10.0),
            ("LogNormal", x => new LogNormalDistribution<double>(0.0, 1.0).Cdf(x), 0.01, 10.0),
        };

        foreach (var (name, cdf, lo, hi) in distributions)
        {
            double prev = cdf(lo);
            for (int i = 1; i <= 100; i++)
            {
                double x = lo + (hi - lo) * i / 100.0;
                double current = cdf(x);
                Assert.True(current >= prev - 1e-10, $"{name} CDF not monotone at x={x}: prev={prev}, current={current}");
                prev = current;
            }
        }
    }

    // ============================================================
    //  SAMPLING STATISTICS
    // ============================================================

    [Fact]
    public void Normal_SampleMean_ApproachesTheoreticalMean()
    {
        var dist = new NormalDistribution<double>(5.0, 4.0);
        var rng = new Random(42);
        int n = 10000;
        double sum = 0;
        for (int i = 0; i < n; i++)
        {
            sum += dist.Sample(rng);
        }
        double sampleMean = sum / n;
        Assert.Equal(5.0, sampleMean, 0.1);
    }

    [Fact]
    public void Exponential_SampleMean_ApproachesTheoreticalMean()
    {
        var dist = new ExponentialDistribution<double>(2.0);
        var rng = new Random(42);
        int n = 10000;
        double sum = 0;
        for (int i = 0; i < n; i++)
        {
            sum += dist.Sample(rng);
        }
        double sampleMean = sum / n;
        Assert.Equal(0.5, sampleMean, 0.05);
    }

    [Fact]
    public void Beta_Samples_InUnitInterval()
    {
        var dist = new BetaDistribution<double>(2.0, 5.0);
        var rng = new Random(42);
        for (int i = 0; i < 1000; i++)
        {
            double s = dist.Sample(rng);
            Assert.True(s >= 0.0 && s <= 1.0, $"Beta sample {s} outside [0,1]");
        }
    }

    [Fact]
    public void Poisson_SampleMean_ApproachesLambda()
    {
        var dist = new PoissonDistribution<double>(5.0);
        var rng = new Random(42);
        int n = 10000;
        double sum = 0;
        for (int i = 0; i < n; i++)
        {
            sum += dist.Sample(rng);
        }
        double sampleMean = sum / n;
        Assert.Equal(5.0, sampleMean, 0.2);
    }

    // ============================================================
    //  CLONE CONSISTENCY
    // ============================================================

    [Fact]
    public void Clone_AllDistributions_ProduceSameResults()
    {
        var distributions = new (string name, ISamplingDistribution<double> dist)[]
        {
            ("Normal", new NormalDistribution<double>(1.0, 2.0)),
            ("Exponential", new ExponentialDistribution<double>(3.0)),
            ("Gamma", new GammaDistribution<double>(2.0, 1.5)),
            ("Beta", new BetaDistribution<double>(2.0, 5.0)),
            ("Laplace", new LaplaceDistribution<double>(1.0, 3.0)),
            ("Weibull", new WeibullDistribution<double>(2.0, 3.0)),
            ("LogNormal", new LogNormalDistribution<double>(0.0, 1.0)),
            ("Poisson", new PoissonDistribution<double>(4.0)),
        };

        foreach (var (name, dist) in distributions)
        {
            var clone = ((IParametricDistribution<double>)dist).Clone();
            var cloneTyped = (ISamplingDistribution<double>)clone;

            double testX = 1.5;
            Assert.Equal(dist.Pdf(testX), cloneTyped.Pdf(testX), Tolerance);
            Assert.Equal(dist.LogPdf(testX), cloneTyped.LogPdf(testX), Tolerance);
            Assert.Equal(dist.Cdf(testX), cloneTyped.Cdf(testX), Tolerance);
        }
    }

    // ============================================================
    //  SPECIFIC HAND-CALCULATED REGRESSION TESTS
    // ============================================================

    [Fact]
    public void Normal_StandardNormal_Pdf_At0_HandValue()
    {
        // Standard Normal PDF at 0 = 1/sqrt(2*pi) ≈ 0.398942280401
        var dist = new NormalDistribution<double>(0.0, 1.0);
        Assert.Equal(1.0 / Math.Sqrt(2.0 * Math.PI), dist.Pdf(0.0), Tolerance);
    }

    [Fact]
    public void Exponential_FisherInformation_HandValue()
    {
        // I = 1/rate^2
        var dist = new ExponentialDistribution<double>(3.0);
        var fisher = dist.FisherInformation();
        Assert.Equal(1.0 / 9.0, fisher[0, 0], Tolerance);
    }

    [Fact]
    public void Gamma_FisherInformation_OffDiagonal()
    {
        // I[0,1] = I[1,0] = -1/beta
        var dist = new GammaDistribution<double>(3.0, 2.0);
        var fisher = dist.FisherInformation();
        Assert.Equal(-0.5, fisher[0, 1], Tolerance); // -1/2
        Assert.Equal(-0.5, fisher[1, 0], Tolerance);
    }

    [Fact]
    public void Weibull_FisherInformation_ScaleScaleDiagonal()
    {
        // I[scale, scale] = k^2/lambda^2
        var dist = new WeibullDistribution<double>(2.0, 3.0);
        var fisher = dist.FisherInformation();
        Assert.Equal(4.0 / 9.0, fisher[1, 1], Tolerance); // k^2/lambda^2 = 4/9
    }
}
