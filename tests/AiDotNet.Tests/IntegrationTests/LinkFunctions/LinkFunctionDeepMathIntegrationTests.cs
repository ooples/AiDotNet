using AiDotNet.Interfaces;
using AiDotNet.LinkFunctions;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.LinkFunctions;

/// <summary>
/// Deep mathematical correctness tests for GLM link functions.
/// Verifies exact hand-calculated values, roundtrip (inverse) consistency,
/// numerical gradient verification of derivatives, chain rule identity,
/// monotonicity, and domain/range correctness.
/// </summary>
public class LinkFunctionDeepMathIntegrationTests
{
    private const double Tolerance = 1e-7;
    private const double LooseTolerance = 1e-5;
    private const double GradientH = 1e-7;

    // ============================================================
    //  IDENTITY LINK: g(mu) = mu
    // ============================================================

    [Fact]
    public void Identity_Link_IsIdentity()
    {
        var link = new IdentityLink<double>();
        double[] values = { -10, -1, 0, 0.5, 1, 10, 100 };
        foreach (double v in values)
        {
            Assert.Equal(v, link.Link(v), Tolerance);
        }
    }

    [Fact]
    public void Identity_InverseLink_IsIdentity()
    {
        var link = new IdentityLink<double>();
        double[] values = { -10, -1, 0, 0.5, 1, 10, 100 };
        foreach (double v in values)
        {
            Assert.Equal(v, link.InverseLink(v), Tolerance);
        }
    }

    [Fact]
    public void Identity_LinkDerivative_IsOne()
    {
        var link = new IdentityLink<double>();
        double[] values = { -10, -1, 0, 0.5, 1, 10, 100 };
        foreach (double v in values)
        {
            Assert.Equal(1.0, link.LinkDerivative(v), Tolerance);
        }
    }

    [Fact]
    public void Identity_InverseLinkDerivative_IsOne()
    {
        var link = new IdentityLink<double>();
        double[] values = { -10, -1, 0, 0.5, 1, 10, 100 };
        foreach (double v in values)
        {
            Assert.Equal(1.0, link.InverseLinkDerivative(v), Tolerance);
        }
    }

    [Fact]
    public void Identity_Variance_IsOne()
    {
        var link = new IdentityLink<double>();
        double[] values = { 0.1, 0.5, 1.0, 5.0 };
        foreach (double v in values)
        {
            Assert.Equal(1.0, link.Variance(v), Tolerance);
        }
    }

    // ============================================================
    //  LOGIT LINK: g(mu) = log(mu / (1-mu))
    // ============================================================

    [Fact]
    public void Logit_HandValues()
    {
        var link = new LogitLink<double>();

        // logit(0.5) = log(0.5/0.5) = log(1) = 0
        Assert.Equal(0.0, link.Link(0.5), Tolerance);

        // logit(0.75) = log(0.75/0.25) = log(3) ≈ 1.0986
        Assert.Equal(Math.Log(3), link.Link(0.75), Tolerance);

        // logit(0.25) = log(0.25/0.75) = log(1/3) ≈ -1.0986
        Assert.Equal(Math.Log(1.0 / 3.0), link.Link(0.25), Tolerance);

        // logit(0.9) = log(9) ≈ 2.1972
        Assert.Equal(Math.Log(9), link.Link(0.9), Tolerance);
    }

    [Fact]
    public void Logit_InverseHandValues()
    {
        var link = new LogitLink<double>();

        // sigmoid(0) = 0.5
        Assert.Equal(0.5, link.InverseLink(0.0), Tolerance);

        // sigmoid(log(9)) = 0.9
        Assert.Equal(0.9, link.InverseLink(Math.Log(9)), Tolerance);

        // sigmoid(-log(9)) = 0.1
        Assert.Equal(0.1, link.InverseLink(-Math.Log(9)), Tolerance);
    }

    [Fact]
    public void Logit_Roundtrip()
    {
        var link = new LogitLink<double>();
        double[] probs = { 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99 };
        foreach (double p in probs)
        {
            double eta = link.Link(p);
            double recovered = link.InverseLink(eta);
            Assert.Equal(p, recovered, LooseTolerance);
        }
    }

    [Fact]
    public void Logit_NumericalGradient_Link()
    {
        var link = new LogitLink<double>();
        double[] muValues = { 0.1, 0.3, 0.5, 0.7, 0.9 };
        foreach (double mu in muValues)
        {
            double analytical = link.LinkDerivative(mu);
            double numerical = (link.Link(mu + GradientH) - link.Link(mu - GradientH)) / (2 * GradientH);
            Assert.Equal(numerical, analytical, LooseTolerance);
        }
    }

    [Fact]
    public void Logit_NumericalGradient_InverseLink()
    {
        var link = new LogitLink<double>();
        double[] etaValues = { -3, -1, 0, 1, 3 };
        foreach (double eta in etaValues)
        {
            double analytical = link.InverseLinkDerivative(eta);
            double numerical = (link.InverseLink(eta + GradientH) - link.InverseLink(eta - GradientH)) / (2 * GradientH);
            Assert.Equal(numerical, analytical, LooseTolerance);
        }
    }

    [Fact]
    public void Logit_ChainRule_DerivativeProduct()
    {
        // g'(mu) * (g^{-1})'(g(mu)) = 1
        var link = new LogitLink<double>();
        double[] muValues = { 0.1, 0.3, 0.5, 0.7, 0.9 };
        foreach (double mu in muValues)
        {
            double eta = link.Link(mu);
            double product = link.LinkDerivative(mu) * link.InverseLinkDerivative(eta);
            Assert.Equal(1.0, product, LooseTolerance);
        }
    }

    [Fact]
    public void Logit_Monotonicity()
    {
        var link = new LogitLink<double>();
        double prev = link.Link(0.01);
        for (double p = 0.05; p < 1.0; p += 0.05)
        {
            double curr = link.Link(p);
            Assert.True(curr > prev, $"Logit should be monotonically increasing: logit({p}) = {curr} <= logit({p - 0.05}) = {prev}");
            prev = curr;
        }
    }

    [Fact]
    public void Logit_Antisymmetry()
    {
        // logit(p) = -logit(1-p)
        var link = new LogitLink<double>();
        double[] probs = { 0.1, 0.2, 0.3, 0.4 };
        foreach (double p in probs)
        {
            double left = link.Link(p);
            double right = -link.Link(1 - p);
            Assert.Equal(right, left, Tolerance);
        }
    }

    [Fact]
    public void Logit_Variance_IsMuTimeOneMinusMu()
    {
        var link = new LogitLink<double>();
        double[] probs = { 0.1, 0.3, 0.5, 0.7, 0.9 };
        foreach (double p in probs)
        {
            double expected = p * (1 - p);
            Assert.Equal(expected, link.Variance(p), Tolerance);
        }
    }

    [Fact]
    public void Logit_LinkDerivative_Equals_1OverVariance()
    {
        // For logit link: g'(mu) = 1/(mu*(1-mu)) = 1/V(mu)
        var link = new LogitLink<double>();
        double[] probs = { 0.1, 0.3, 0.5, 0.7, 0.9 };
        foreach (double p in probs)
        {
            double deriv = link.LinkDerivative(p);
            double invVar = 1.0 / link.Variance(p);
            Assert.Equal(invVar, deriv, Tolerance);
        }
    }

    // ============================================================
    //  LOG LINK: g(mu) = log(mu)
    // ============================================================

    [Fact]
    public void Log_HandValues()
    {
        var link = new LogLink<double>();

        // log(1) = 0
        Assert.Equal(0.0, link.Link(1.0), Tolerance);

        // log(e) = 1
        Assert.Equal(1.0, link.Link(Math.E), Tolerance);

        // log(e^2) = 2
        Assert.Equal(2.0, link.Link(Math.E * Math.E), Tolerance);

        // log(0.5) ≈ -0.6931
        Assert.Equal(Math.Log(0.5), link.Link(0.5), Tolerance);
    }

    [Fact]
    public void Log_InverseHandValues()
    {
        var link = new LogLink<double>();

        // exp(0) = 1
        Assert.Equal(1.0, link.InverseLink(0.0), Tolerance);

        // exp(1) = e
        Assert.Equal(Math.E, link.InverseLink(1.0), Tolerance);

        // exp(-1) = 1/e
        Assert.Equal(1.0 / Math.E, link.InverseLink(-1.0), Tolerance);
    }

    [Fact]
    public void Log_Roundtrip()
    {
        var link = new LogLink<double>();
        double[] muValues = { 0.01, 0.1, 1, 5, 10, 100 };
        foreach (double mu in muValues)
        {
            double eta = link.Link(mu);
            double recovered = link.InverseLink(eta);
            Assert.Equal(mu, recovered, LooseTolerance);
        }
    }

    [Fact]
    public void Log_NumericalGradient_Link()
    {
        var link = new LogLink<double>();
        double[] muValues = { 0.1, 0.5, 1.0, 5.0, 10.0 };
        foreach (double mu in muValues)
        {
            double analytical = link.LinkDerivative(mu);
            double numerical = (link.Link(mu + GradientH) - link.Link(mu - GradientH)) / (2 * GradientH);
            Assert.Equal(numerical, analytical, LooseTolerance);
        }
    }

    [Fact]
    public void Log_NumericalGradient_InverseLink()
    {
        var link = new LogLink<double>();
        double[] etaValues = { -3, -1, 0, 1, 3 };
        foreach (double eta in etaValues)
        {
            double analytical = link.InverseLinkDerivative(eta);
            double numerical = (link.InverseLink(eta + GradientH) - link.InverseLink(eta - GradientH)) / (2 * GradientH);
            Assert.Equal(numerical, analytical, LooseTolerance);
        }
    }

    [Fact]
    public void Log_ChainRule()
    {
        var link = new LogLink<double>();
        double[] muValues = { 0.1, 0.5, 1.0, 5.0, 10.0 };
        foreach (double mu in muValues)
        {
            double eta = link.Link(mu);
            double product = link.LinkDerivative(mu) * link.InverseLinkDerivative(eta);
            Assert.Equal(1.0, product, LooseTolerance);
        }
    }

    [Fact]
    public void Log_LinkDerivative_Is1OverMu()
    {
        var link = new LogLink<double>();
        double[] muValues = { 0.1, 0.5, 1.0, 5.0, 10.0 };
        foreach (double mu in muValues)
        {
            Assert.Equal(1.0 / mu, link.LinkDerivative(mu), Tolerance);
        }
    }

    [Fact]
    public void Log_InverseLinkDerivative_EqualsInverseLink()
    {
        // d/deta exp(eta) = exp(eta)
        var link = new LogLink<double>();
        double[] etaValues = { -2, -1, 0, 1, 2 };
        foreach (double eta in etaValues)
        {
            Assert.Equal(link.InverseLink(eta), link.InverseLinkDerivative(eta), Tolerance);
        }
    }

    [Fact]
    public void Log_Monotonicity()
    {
        var link = new LogLink<double>();
        double prev = link.Link(0.01);
        for (double m = 0.1; m < 20; m += 0.5)
        {
            double curr = link.Link(m);
            Assert.True(curr > prev, $"Log should be monotonically increasing");
            prev = curr;
        }
    }

    [Fact]
    public void Log_Variance_IsMu()
    {
        var link = new LogLink<double>();
        double[] muValues = { 0.1, 1.0, 5.0, 10.0 };
        foreach (double mu in muValues)
        {
            Assert.Equal(mu, link.Variance(mu), Tolerance);
        }
    }

    // ============================================================
    //  PROBIT LINK: g(mu) = Phi^{-1}(mu)
    // ============================================================

    [Fact]
    public void Probit_HandValues()
    {
        var link = new ProbitLink<double>();

        // probit(0.5) = 0 (median of standard normal)
        Assert.Equal(0.0, link.Link(0.5), LooseTolerance);

        // probit(0.8413) ≈ 1.0 (Phi(1) ≈ 0.8413)
        Assert.Equal(1.0, link.Link(0.8413), 0.01);

        // probit(0.1587) ≈ -1.0 (Phi(-1) ≈ 0.1587)
        Assert.Equal(-1.0, link.Link(0.1587), 0.01);
    }

    [Fact]
    public void Probit_InverseHandValues()
    {
        var link = new ProbitLink<double>();

        // Phi(0) = 0.5
        Assert.Equal(0.5, link.InverseLink(0.0), LooseTolerance);

        // Phi(1) ≈ 0.8413
        Assert.Equal(0.8413, link.InverseLink(1.0), 0.001);

        // Phi(-1) ≈ 0.1587
        Assert.Equal(0.1587, link.InverseLink(-1.0), 0.001);

        // Phi(2) ≈ 0.9772
        Assert.Equal(0.9772, link.InverseLink(2.0), 0.001);
    }

    [Fact]
    public void Probit_Roundtrip()
    {
        var link = new ProbitLink<double>();
        double[] probs = { 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95 };
        foreach (double p in probs)
        {
            double eta = link.Link(p);
            double recovered = link.InverseLink(eta);
            Assert.Equal(p, recovered, 0.001);
        }
    }

    [Fact]
    public void Probit_NumericalGradient_Link()
    {
        var link = new ProbitLink<double>();
        double[] muValues = { 0.1, 0.3, 0.5, 0.7, 0.9 };
        foreach (double mu in muValues)
        {
            double analytical = link.LinkDerivative(mu);
            double h = 1e-6;
            double numerical = (link.Link(mu + h) - link.Link(mu - h)) / (2 * h);
            Assert.Equal(numerical, analytical, 0.01);
        }
    }

    [Fact]
    public void Probit_NumericalGradient_InverseLink()
    {
        var link = new ProbitLink<double>();
        double[] etaValues = { -2, -1, 0, 1, 2 };
        foreach (double eta in etaValues)
        {
            double analytical = link.InverseLinkDerivative(eta);
            double numerical = (link.InverseLink(eta + GradientH) - link.InverseLink(eta - GradientH)) / (2 * GradientH);
            Assert.Equal(numerical, analytical, LooseTolerance);
        }
    }

    [Fact]
    public void Probit_InverseLinkDerivative_IsNormalPdf()
    {
        // d/deta Phi(eta) = phi(eta) where phi is the standard normal PDF
        var link = new ProbitLink<double>();
        double[] etaValues = { -2, -1, 0, 1, 2 };
        foreach (double eta in etaValues)
        {
            double expected = Math.Exp(-eta * eta / 2) / Math.Sqrt(2 * Math.PI);
            double actual = link.InverseLinkDerivative(eta);
            Assert.Equal(expected, actual, Tolerance);
        }
    }

    [Fact]
    public void Probit_Antisymmetry()
    {
        // probit(p) = -probit(1-p)
        var link = new ProbitLink<double>();
        double[] probs = { 0.1, 0.2, 0.3, 0.4 };
        foreach (double p in probs)
        {
            double left = link.Link(p);
            double right = -link.Link(1 - p);
            Assert.Equal(right, left, 0.001);
        }
    }

    [Fact]
    public void Probit_Monotonicity()
    {
        var link = new ProbitLink<double>();
        double prev = link.Link(0.01);
        for (double p = 0.05; p < 1.0; p += 0.05)
        {
            double curr = link.Link(p);
            Assert.True(curr > prev, "Probit should be monotonically increasing");
            prev = curr;
        }
    }

    // ============================================================
    //  CLOGLOG LINK: g(mu) = log(-log(1-mu))
    // ============================================================

    [Fact]
    public void CLogLog_HandValues()
    {
        var link = new CLogLogLink<double>();

        // cloglog(0.5) = log(-log(0.5)) = log(log(2)) ≈ log(0.6931) ≈ -0.3665
        double expected = Math.Log(-Math.Log(1 - 0.5));
        Assert.Equal(expected, link.Link(0.5), Tolerance);

        // cloglog(1-1/e) = cloglog(0.6321) = log(-log(1/e)) = log(1) = 0
        double p = 1 - 1.0 / Math.E;
        Assert.Equal(0.0, link.Link(p), LooseTolerance);
    }

    [Fact]
    public void CLogLog_InverseHandValues()
    {
        var link = new CLogLogLink<double>();

        // inv_cloglog(0) = 1 - exp(-exp(0)) = 1 - exp(-1) = 1 - 1/e ≈ 0.6321
        Assert.Equal(1 - 1.0 / Math.E, link.InverseLink(0.0), Tolerance);

        // inv_cloglog(-inf) -> 0
        Assert.True(link.InverseLink(-10) < 0.001);

        // inv_cloglog(+inf) -> 1
        Assert.True(link.InverseLink(10) > 0.999);
    }

    [Fact]
    public void CLogLog_Roundtrip()
    {
        var link = new CLogLogLink<double>();
        double[] probs = { 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99 };
        foreach (double p in probs)
        {
            double eta = link.Link(p);
            double recovered = link.InverseLink(eta);
            Assert.Equal(p, recovered, LooseTolerance);
        }
    }

    [Fact]
    public void CLogLog_NumericalGradient_Link()
    {
        var link = new CLogLogLink<double>();
        double[] muValues = { 0.1, 0.3, 0.5, 0.7, 0.9 };
        foreach (double mu in muValues)
        {
            double analytical = link.LinkDerivative(mu);
            double h = 1e-6;
            double numerical = (link.Link(mu + h) - link.Link(mu - h)) / (2 * h);
            Assert.Equal(numerical, analytical, 0.01);
        }
    }

    [Fact]
    public void CLogLog_NumericalGradient_InverseLink()
    {
        var link = new CLogLogLink<double>();
        double[] etaValues = { -2, -1, 0, 1, 2 };
        foreach (double eta in etaValues)
        {
            double analytical = link.InverseLinkDerivative(eta);
            double numerical = (link.InverseLink(eta + GradientH) - link.InverseLink(eta - GradientH)) / (2 * GradientH);
            Assert.Equal(numerical, analytical, LooseTolerance);
        }
    }

    [Fact]
    public void CLogLog_IsAsymmetric()
    {
        // Unlike logit, cloglog(0.5) != 0
        var link = new CLogLogLink<double>();
        Assert.NotEqual(0.0, link.Link(0.5));
    }

    [Fact]
    public void CLogLog_Monotonicity()
    {
        var link = new CLogLogLink<double>();
        double prev = link.Link(0.01);
        for (double p = 0.05; p < 1.0; p += 0.05)
        {
            double curr = link.Link(p);
            Assert.True(curr > prev, "CLogLog should be monotonically increasing");
            prev = curr;
        }
    }

    // ============================================================
    //  RECIPROCAL (INVERSE) LINK: g(mu) = 1/mu
    // ============================================================

    [Fact]
    public void Reciprocal_HandValues()
    {
        var link = new ReciprocalLink<double>();

        // g(2) = 0.5
        Assert.Equal(0.5, link.Link(2.0), Tolerance);

        // g(0.5) = 2
        Assert.Equal(2.0, link.Link(0.5), Tolerance);

        // g(1) = 1
        Assert.Equal(1.0, link.Link(1.0), Tolerance);

        // g(4) = 0.25
        Assert.Equal(0.25, link.Link(4.0), Tolerance);
    }

    [Fact]
    public void Reciprocal_IsInvolution()
    {
        // g(g(x)) = x for reciprocal link
        var link = new ReciprocalLink<double>();
        double[] values = { 0.1, 0.5, 1.0, 2.0, 10.0 };
        foreach (double v in values)
        {
            double gg = link.Link(link.Link(v));
            Assert.Equal(v, gg, Tolerance);
        }
    }

    [Fact]
    public void Reciprocal_Roundtrip()
    {
        var link = new ReciprocalLink<double>();
        double[] muValues = { 0.1, 0.5, 1, 2, 5, 10 };
        foreach (double mu in muValues)
        {
            double eta = link.Link(mu);
            double recovered = link.InverseLink(eta);
            Assert.Equal(mu, recovered, LooseTolerance);
        }
    }

    [Fact]
    public void Reciprocal_LinkDerivative_IsNegative1OverMuSquared()
    {
        var link = new ReciprocalLink<double>();
        double[] muValues = { 0.5, 1.0, 2.0, 5.0 };
        foreach (double mu in muValues)
        {
            double expected = -1.0 / (mu * mu);
            Assert.Equal(expected, link.LinkDerivative(mu), Tolerance);
        }
    }

    [Fact]
    public void Reciprocal_NumericalGradient_Link()
    {
        var link = new ReciprocalLink<double>();
        double[] muValues = { 0.5, 1.0, 2.0, 5.0 };
        foreach (double mu in muValues)
        {
            double analytical = link.LinkDerivative(mu);
            double numerical = (link.Link(mu + GradientH) - link.Link(mu - GradientH)) / (2 * GradientH);
            Assert.Equal(numerical, analytical, LooseTolerance);
        }
    }

    [Fact]
    public void Reciprocal_NumericalGradient_InverseLink()
    {
        var link = new ReciprocalLink<double>();
        double[] etaValues = { 0.1, 0.5, 1.0, 2.0, 5.0 };
        foreach (double eta in etaValues)
        {
            double analytical = link.InverseLinkDerivative(eta);
            double numerical = (link.InverseLink(eta + GradientH) - link.InverseLink(eta - GradientH)) / (2 * GradientH);
            Assert.Equal(numerical, analytical, LooseTolerance);
        }
    }

    [Fact]
    public void Reciprocal_Variance_IsMuSquared()
    {
        var link = new ReciprocalLink<double>();
        double[] muValues = { 0.5, 1.0, 2.0, 5.0 };
        foreach (double mu in muValues)
        {
            Assert.Equal(mu * mu, link.Variance(mu), Tolerance);
        }
    }

    // ============================================================
    //  SQRT LINK: g(mu) = sqrt(mu)
    // ============================================================

    [Fact]
    public void Sqrt_HandValues()
    {
        var link = new SqrtLink<double>();

        // sqrt(4) = 2
        Assert.Equal(2.0, link.Link(4.0), Tolerance);

        // sqrt(1) = 1
        Assert.Equal(1.0, link.Link(1.0), Tolerance);

        // sqrt(0) = 0
        Assert.Equal(0.0, link.Link(0.0), Tolerance);

        // sqrt(9) = 3
        Assert.Equal(3.0, link.Link(9.0), Tolerance);

        // sqrt(0.25) = 0.5
        Assert.Equal(0.5, link.Link(0.25), Tolerance);
    }

    [Fact]
    public void Sqrt_InverseHandValues()
    {
        var link = new SqrtLink<double>();

        // 2^2 = 4
        Assert.Equal(4.0, link.InverseLink(2.0), Tolerance);

        // 3^2 = 9
        Assert.Equal(9.0, link.InverseLink(3.0), Tolerance);

        // 0^2 = 0
        Assert.Equal(0.0, link.InverseLink(0.0), Tolerance);
    }

    [Fact]
    public void Sqrt_Roundtrip()
    {
        var link = new SqrtLink<double>();
        double[] muValues = { 0.01, 0.25, 1, 4, 9, 100 };
        foreach (double mu in muValues)
        {
            double eta = link.Link(mu);
            double recovered = link.InverseLink(eta);
            Assert.Equal(mu, recovered, LooseTolerance);
        }
    }

    [Fact]
    public void Sqrt_LinkDerivative_Is1Over2SqrtMu()
    {
        var link = new SqrtLink<double>();
        double[] muValues = { 0.25, 1.0, 4.0, 9.0 };
        foreach (double mu in muValues)
        {
            double expected = 1.0 / (2.0 * Math.Sqrt(mu));
            Assert.Equal(expected, link.LinkDerivative(mu), Tolerance);
        }
    }

    [Fact]
    public void Sqrt_InverseLinkDerivative_Is2Eta()
    {
        var link = new SqrtLink<double>();
        double[] etaValues = { 0.5, 1.0, 2.0, 5.0 };
        foreach (double eta in etaValues)
        {
            Assert.Equal(2.0 * eta, link.InverseLinkDerivative(eta), Tolerance);
        }
    }

    [Fact]
    public void Sqrt_NumericalGradient_Link()
    {
        var link = new SqrtLink<double>();
        double[] muValues = { 0.25, 1.0, 4.0, 9.0, 25.0 };
        foreach (double mu in muValues)
        {
            double analytical = link.LinkDerivative(mu);
            double numerical = (link.Link(mu + GradientH) - link.Link(mu - GradientH)) / (2 * GradientH);
            Assert.Equal(numerical, analytical, LooseTolerance);
        }
    }

    [Fact]
    public void Sqrt_NumericalGradient_InverseLink()
    {
        var link = new SqrtLink<double>();
        double[] etaValues = { 0.5, 1.0, 2.0, 5.0 };
        foreach (double eta in etaValues)
        {
            double analytical = link.InverseLinkDerivative(eta);
            double numerical = (link.InverseLink(eta + GradientH) - link.InverseLink(eta - GradientH)) / (2 * GradientH);
            Assert.Equal(numerical, analytical, LooseTolerance);
        }
    }

    [Fact]
    public void Sqrt_ChainRule()
    {
        var link = new SqrtLink<double>();
        double[] muValues = { 0.25, 1.0, 4.0, 9.0 };
        foreach (double mu in muValues)
        {
            double eta = link.Link(mu);
            double product = link.LinkDerivative(mu) * link.InverseLinkDerivative(eta);
            Assert.Equal(1.0, product, LooseTolerance);
        }
    }

    // ============================================================
    //  INVERSE SQUARED LINK: g(mu) = 1/mu^2
    // ============================================================

    [Fact]
    public void InverseSquared_HandValues()
    {
        var link = new InverseSquaredLink<double>();

        // g(1) = 1/1 = 1
        Assert.Equal(1.0, link.Link(1.0), Tolerance);

        // g(2) = 1/4 = 0.25
        Assert.Equal(0.25, link.Link(2.0), Tolerance);

        // g(0.5) = 1/0.25 = 4
        Assert.Equal(4.0, link.Link(0.5), Tolerance);

        // g(3) = 1/9
        Assert.Equal(1.0 / 9.0, link.Link(3.0), Tolerance);
    }

    [Fact]
    public void InverseSquared_InverseHandValues()
    {
        var link = new InverseSquaredLink<double>();

        // g^-1(1) = 1/sqrt(1) = 1
        Assert.Equal(1.0, link.InverseLink(1.0), Tolerance);

        // g^-1(4) = 1/sqrt(4) = 0.5
        Assert.Equal(0.5, link.InverseLink(4.0), Tolerance);

        // g^-1(0.25) = 1/sqrt(0.25) = 2
        Assert.Equal(2.0, link.InverseLink(0.25), Tolerance);
    }

    [Fact]
    public void InverseSquared_Roundtrip()
    {
        var link = new InverseSquaredLink<double>();
        double[] muValues = { 0.1, 0.5, 1, 2, 5, 10 };
        foreach (double mu in muValues)
        {
            double eta = link.Link(mu);
            double recovered = link.InverseLink(eta);
            Assert.Equal(mu, recovered, LooseTolerance);
        }
    }

    [Fact]
    public void InverseSquared_LinkDerivative_IsNeg2OverMuCubed()
    {
        var link = new InverseSquaredLink<double>();
        double[] muValues = { 0.5, 1.0, 2.0, 5.0 };
        foreach (double mu in muValues)
        {
            double expected = -2.0 / (mu * mu * mu);
            Assert.Equal(expected, link.LinkDerivative(mu), Tolerance);
        }
    }

    [Fact]
    public void InverseSquared_NumericalGradient_Link()
    {
        var link = new InverseSquaredLink<double>();
        double[] muValues = { 0.5, 1.0, 2.0, 5.0 };
        foreach (double mu in muValues)
        {
            double analytical = link.LinkDerivative(mu);
            double numerical = (link.Link(mu + GradientH) - link.Link(mu - GradientH)) / (2 * GradientH);
            Assert.Equal(numerical, analytical, LooseTolerance);
        }
    }

    [Fact]
    public void InverseSquared_NumericalGradient_InverseLink()
    {
        var link = new InverseSquaredLink<double>();
        double[] etaValues = { 0.1, 0.5, 1.0, 4.0, 16.0 };
        foreach (double eta in etaValues)
        {
            double analytical = link.InverseLinkDerivative(eta);
            double numerical = (link.InverseLink(eta + GradientH) - link.InverseLink(eta - GradientH)) / (2 * GradientH);
            Assert.Equal(numerical, analytical, LooseTolerance);
        }
    }

    [Fact]
    public void InverseSquared_Variance_IsMuCubed()
    {
        var link = new InverseSquaredLink<double>();
        double[] muValues = { 0.5, 1.0, 2.0, 5.0 };
        foreach (double mu in muValues)
        {
            Assert.Equal(mu * mu * mu, link.Variance(mu), Tolerance);
        }
    }

    // ============================================================
    //  CROSS-LINK PROPERTIES
    // ============================================================

    [Fact]
    public void AllLinks_ChainRule_DerivativeProduct()
    {
        // For all valid link functions: g'(mu) * (g^-1)'(g(mu)) = 1
        var logit = new LogitLink<double>();
        var log = new LogLink<double>();
        var sqrt = new SqrtLink<double>();
        var reciprocal = new ReciprocalLink<double>();
        var invSquared = new InverseSquaredLink<double>();

        // Logit (mu in (0,1))
        foreach (double mu in new[] { 0.1, 0.3, 0.5, 0.7, 0.9 })
        {
            double eta = logit.Link(mu);
            double product = logit.LinkDerivative(mu) * logit.InverseLinkDerivative(eta);
            Assert.Equal(1.0, product, LooseTolerance);
        }

        // Log (mu > 0)
        foreach (double mu in new[] { 0.1, 1.0, 5.0, 10.0 })
        {
            double eta = log.Link(mu);
            double product = log.LinkDerivative(mu) * log.InverseLinkDerivative(eta);
            Assert.Equal(1.0, product, LooseTolerance);
        }

        // Sqrt (mu > 0)
        foreach (double mu in new[] { 0.25, 1.0, 4.0, 9.0 })
        {
            double eta = sqrt.Link(mu);
            double product = sqrt.LinkDerivative(mu) * sqrt.InverseLinkDerivative(eta);
            Assert.Equal(1.0, product, LooseTolerance);
        }

        // Reciprocal (mu > 0)
        foreach (double mu in new[] { 0.5, 1.0, 2.0, 5.0 })
        {
            double eta = reciprocal.Link(mu);
            double product = reciprocal.LinkDerivative(mu) * reciprocal.InverseLinkDerivative(eta);
            Assert.Equal(1.0, product, LooseTolerance);
        }

        // InverseSquared (mu > 0)
        foreach (double mu in new[] { 0.5, 1.0, 2.0, 5.0 })
        {
            double eta = invSquared.Link(mu);
            double product = invSquared.LinkDerivative(mu) * invSquared.InverseLinkDerivative(eta);
            Assert.Equal(1.0, product, LooseTolerance);
        }
    }

    [Fact]
    public void BinaryLinks_AtHalf_CompareOutputs()
    {
        // Compare logit, probit, cloglog at mu=0.5
        var logit = new LogitLink<double>();
        var probit = new ProbitLink<double>();
        var cloglog = new CLogLogLink<double>();

        // logit(0.5) = 0, probit(0.5) = 0
        Assert.Equal(0.0, logit.Link(0.5), Tolerance);
        Assert.Equal(0.0, probit.Link(0.5), LooseTolerance);

        // cloglog(0.5) != 0 (asymmetric)
        double cloglogHalf = cloglog.Link(0.5);
        Assert.True(Math.Abs(cloglogHalf) > 0.1, "cloglog(0.5) should not be near 0");
    }

    [Fact]
    public void LogitVsProbit_CloseForModerateValues()
    {
        // Logit and probit are approximately proportional: logit ≈ 1.7 * probit
        var logit = new LogitLink<double>();
        var probit = new ProbitLink<double>();

        double[] probs = { 0.2, 0.3, 0.4, 0.6, 0.7, 0.8 };
        foreach (double p in probs)
        {
            double l = logit.Link(p);
            double pr = probit.Link(p);

            // The scaling factor is approximately pi/sqrt(3) ≈ 1.8138
            // but a common rule of thumb is 1.6-1.8
            double ratio = l / pr;
            Assert.True(ratio > 1.5 && ratio < 2.0,
                $"logit/probit ratio at {p} was {ratio}, expected ~1.7");
        }
    }

    [Fact]
    public void AllCanonicalLinks_ReportCorrectCanonicalStatus()
    {
        // Identity = canonical for Normal
        Assert.True(new IdentityLink<double>().IsCanonical);
        // Logit = canonical for Binomial
        Assert.True(new LogitLink<double>().IsCanonical);
        // Log = canonical for Poisson
        Assert.True(new LogLink<double>().IsCanonical);
        // Reciprocal = canonical for Gamma
        Assert.True(new ReciprocalLink<double>().IsCanonical);
        // InverseSquared = canonical for Inverse Gaussian
        Assert.True(new InverseSquaredLink<double>().IsCanonical);
        // Probit = NOT canonical
        Assert.False(new ProbitLink<double>().IsCanonical);
        // CLogLog = NOT canonical
        Assert.False(new CLogLogLink<double>().IsCanonical);
        // Sqrt = NOT canonical
        Assert.False(new SqrtLink<double>().IsCanonical);
    }

    [Fact]
    public void AllLinks_HaveCorrectNames()
    {
        Assert.Equal("Identity", new IdentityLink<double>().Name);
        Assert.Equal("Logit", new LogitLink<double>().Name);
        Assert.Equal("Log", new LogLink<double>().Name);
        Assert.Equal("Probit", new ProbitLink<double>().Name);
        Assert.Equal("CLogLog", new CLogLogLink<double>().Name);
        Assert.Equal("Inverse", new ReciprocalLink<double>().Name);
        Assert.Equal("Sqrt", new SqrtLink<double>().Name);
        Assert.Equal("InverseSquared", new InverseSquaredLink<double>().Name);
    }
}
