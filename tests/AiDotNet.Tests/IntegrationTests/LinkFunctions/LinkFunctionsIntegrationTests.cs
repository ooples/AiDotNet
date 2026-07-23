using AiDotNet.Interfaces;
using AiDotNet.LinkFunctions;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.LinkFunctions;

/// <summary>
/// Integration tests for all link function classes.
/// Tests Link, InverseLink, round-trip, derivative, and factory behavior.
/// </summary>
public class LinkFunctionsIntegrationTests
{
    private const double Tolerance = 1e-4;

    #region Edge Case Tests

    [Fact(Timeout = 120000)]
    public async Task LogitLink_NearBoundary_ProducesLargeValues()
    {
        var link = new LogitLink<double>();
        // Near 0, logit approaches -infinity
        var nearZero = link.Link(0.0001);
        Assert.True(nearZero < -5.0, $"Logit(0.0001) should be large negative, got {nearZero}");
        // Near 1, logit approaches +infinity
        var nearOne = link.Link(0.9999);
        Assert.True(nearOne > 5.0, $"Logit(0.9999) should be large positive, got {nearOne}");
    }

    [Fact(Timeout = 120000)]
    public async Task ProbitLink_NearBoundary_ProducesLargeValues()
    {
        var link = new ProbitLink<double>();
        var nearZero = link.Link(0.001);
        Assert.True(nearZero < -2.0, $"Probit(0.001) should be large negative, got {nearZero}");
        var nearOne = link.Link(0.999);
        Assert.True(nearOne > 2.0, $"Probit(0.999) should be large positive, got {nearOne}");
    }

    [Fact(Timeout = 120000)]
    public async Task LogLink_VerySmallPositive_ProducesLargeNegative()
    {
        var link = new LogLink<double>();
        var result = link.Link(0.001);
        Assert.True(result < -5.0, $"Log(0.001) should be large negative, got {result}");
    }

    #endregion

    #region LogitLink Tests

    [Fact(Timeout = 120000)]
    public async Task LogitLink_LinkAt05_ReturnsZero()
    {
        var link = new LogitLink<double>();
        var result = link.Link(0.5);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task LogitLink_InverseLinkAtZero_Returns05()
    {
        var link = new LogitLink<double>();
        var result = link.InverseLink(0.0);
        Assert.Equal(0.5, result, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task LogitLink_RoundTrip_ReturnsOriginal()
    {
        var link = new LogitLink<double>();
        double[] testValues = { 0.1, 0.25, 0.5, 0.75, 0.9 };
        foreach (var val in testValues)
        {
            var roundTrip = link.InverseLink(link.Link(val));
            Assert.Equal(val, roundTrip, Tolerance);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task LogitLink_LinkDerivative_IsPositiveInUnitInterval()
    {
        var link = new LogitLink<double>();
        double[] testValues = { 0.1, 0.3, 0.5, 0.7, 0.9 };
        foreach (var val in testValues)
        {
            var deriv = link.LinkDerivative(val);
            Assert.True(deriv > 0, $"LinkDerivative({val}) should be positive, got {deriv}");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task LogitLink_InverseLinkDerivative_MatchesSigmoidDerivative()
    {
        var link = new LogitLink<double>();
        var deriv = link.InverseLinkDerivative(0.0);
        // sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
        Assert.Equal(0.25, deriv, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task LogitLink_Variance_CorrectForBinomial()
    {
        var link = new LogitLink<double>();
        var variance = link.Variance(0.5);
        Assert.Equal(0.25, variance, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task LogitLink_IsCanonical_ReturnsTrue()
    {
        var link = new LogitLink<double>();
        Assert.True(link.IsCanonical);
        Assert.Equal("Logit", link.Name);
    }

    #endregion

    #region ProbitLink Tests

    [Fact(Timeout = 120000)]
    public async Task ProbitLink_LinkAt05_ReturnsZero()
    {
        var link = new ProbitLink<double>();
        var result = link.Link(0.5);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task ProbitLink_InverseLinkAtZero_Returns05()
    {
        var link = new ProbitLink<double>();
        var result = link.InverseLink(0.0);
        Assert.Equal(0.5, result, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task ProbitLink_RoundTrip_ReturnsOriginal()
    {
        var link = new ProbitLink<double>();
        double[] testValues = { 0.1, 0.25, 0.5, 0.75, 0.9 };
        foreach (var val in testValues)
        {
            var roundTrip = link.InverseLink(link.Link(val));
            Assert.Equal(val, roundTrip, Tolerance);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task ProbitLink_IsNotCanonical()
    {
        var link = new ProbitLink<double>();
        Assert.False(link.IsCanonical);
        Assert.Equal("Probit", link.Name);
    }

    [Fact(Timeout = 120000)]
    public async Task ProbitLink_DerivativeChainRule_ProductIsOne()
    {
        // For any link function: LinkDerivative(mu) * InverseLinkDerivative(Link(mu)) = 1
        var link = new ProbitLink<double>();
        double mu = 0.7;
        double eta = link.Link(mu);
        double linkDeriv = link.LinkDerivative(mu);
        double inverseDeriv = link.InverseLinkDerivative(eta);
        Assert.Equal(1.0, linkDeriv * inverseDeriv, Tolerance);
    }

    #endregion

    #region LogLink Tests

    [Fact(Timeout = 120000)]
    public async Task LogLink_LinkAt1_ReturnsZero()
    {
        var link = new LogLink<double>();
        var result = link.Link(1.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task LogLink_InverseLinkAtZero_ReturnsOne()
    {
        var link = new LogLink<double>();
        var result = link.InverseLink(0.0);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task LogLink_RoundTrip_ReturnsOriginal()
    {
        var link = new LogLink<double>();
        double[] testValues = { 0.5, 1.0, 2.0, 5.0, 10.0 };
        foreach (var val in testValues)
        {
            var roundTrip = link.InverseLink(link.Link(val));
            Assert.Equal(val, roundTrip, Tolerance);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task LogLink_IsCanonical_ReturnsTrue()
    {
        var link = new LogLink<double>();
        Assert.True(link.IsCanonical);
        Assert.Equal("Log", link.Name);
    }

    [Fact(Timeout = 120000)]
    public async Task LogLink_LinkDerivative_IsReciprocal()
    {
        var link = new LogLink<double>();
        var deriv = link.LinkDerivative(2.0);
        Assert.Equal(0.5, deriv, Tolerance);
    }

    #endregion

    #region IdentityLink Tests

    [Fact(Timeout = 120000)]
    public async Task IdentityLink_LinkEqualsInput()
    {
        var link = new IdentityLink<double>();
        double[] testValues = { -5.0, 0.0, 3.14, 100.0 };
        foreach (var val in testValues)
        {
            Assert.Equal(val, link.Link(val), Tolerance);
            Assert.Equal(val, link.InverseLink(val), Tolerance);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task IdentityLink_DerivativeIsOne()
    {
        var link = new IdentityLink<double>();
        Assert.Equal(1.0, link.LinkDerivative(42.0), Tolerance);
        Assert.Equal(1.0, link.InverseLinkDerivative(42.0), Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task IdentityLink_IsCanonical_ReturnsTrue()
    {
        var link = new IdentityLink<double>();
        Assert.True(link.IsCanonical);
        Assert.Equal("Identity", link.Name);
    }

    #endregion

    #region CLogLogLink Tests

    [Fact(Timeout = 120000)]
    public async Task CLogLogLink_RoundTrip_ReturnsOriginal()
    {
        var link = new CLogLogLink<double>();
        double[] testValues = { 0.1, 0.3, 0.5, 0.7, 0.9 };
        foreach (var val in testValues)
        {
            var roundTrip = link.InverseLink(link.Link(val));
            Assert.Equal(val, roundTrip, Tolerance);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task CLogLogLink_IsNotCanonical()
    {
        var link = new CLogLogLink<double>();
        Assert.False(link.IsCanonical);
        Assert.Equal("CLogLog", link.Name);
    }

    [Fact(Timeout = 120000)]
    public async Task CLogLogLink_IsAsymmetricAroundHalf()
    {
        var link = new CLogLogLink<double>();
        // cloglog(0.5) is not 0 (unlike logit)
        var atHalf = link.Link(0.5);
        Assert.NotEqual(0.0, atHalf, Tolerance);
    }

    #endregion

    #region SqrtLink Tests

    [Fact(Timeout = 120000)]
    public async Task SqrtLink_LinkAt4_Returns2()
    {
        var link = new SqrtLink<double>();
        var result = link.Link(4.0);
        Assert.Equal(2.0, result, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task SqrtLink_InverseLinkAt2_Returns4()
    {
        var link = new SqrtLink<double>();
        var result = link.InverseLink(2.0);
        Assert.Equal(4.0, result, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task SqrtLink_RoundTrip_ReturnsOriginal()
    {
        var link = new SqrtLink<double>();
        double[] testValues = { 0.25, 1.0, 4.0, 9.0, 16.0 };
        foreach (var val in testValues)
        {
            var roundTrip = link.InverseLink(link.Link(val));
            Assert.Equal(val, roundTrip, Tolerance);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task SqrtLink_IsNotCanonical()
    {
        var link = new SqrtLink<double>();
        Assert.False(link.IsCanonical);
        Assert.Equal("Sqrt", link.Name);
    }

    #endregion

    #region ReciprocalLink Tests

    [Fact(Timeout = 120000)]
    public async Task ReciprocalLink_LinkAt2_ReturnsHalf()
    {
        var link = new ReciprocalLink<double>();
        var result = link.Link(2.0);
        Assert.Equal(0.5, result, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task ReciprocalLink_InverseLinkAtHalf_Returns2()
    {
        var link = new ReciprocalLink<double>();
        var result = link.InverseLink(0.5);
        Assert.Equal(2.0, result, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task ReciprocalLink_RoundTrip_ReturnsOriginal()
    {
        var link = new ReciprocalLink<double>();
        double[] testValues = { 0.5, 1.0, 2.0, 5.0 };
        foreach (var val in testValues)
        {
            var roundTrip = link.InverseLink(link.Link(val));
            Assert.Equal(val, roundTrip, Tolerance);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task ReciprocalLink_IsCanonical_ReturnsTrue()
    {
        var link = new ReciprocalLink<double>();
        Assert.True(link.IsCanonical);
        Assert.Equal("Inverse", link.Name);
    }

    #endregion

    #region InverseSquaredLink Tests

    [Fact(Timeout = 120000)]
    public async Task InverseSquaredLink_KnownValues()
    {
        var link = new InverseSquaredLink<double>();
        // Link(2) = 1/4 = 0.25
        Assert.Equal(0.25, link.Link(2.0), Tolerance);
        // Link(1) = 1/1 = 1.0
        Assert.Equal(1.0, link.Link(1.0), Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task InverseSquaredLink_RoundTrip_ReturnsOriginal()
    {
        var link = new InverseSquaredLink<double>();
        double[] testValues = { 0.5, 1.0, 2.0, 5.0 };
        foreach (var val in testValues)
        {
            var roundTrip = link.InverseLink(link.Link(val));
            Assert.Equal(val, roundTrip, Tolerance);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task InverseSquaredLink_IsCanonical_ReturnsTrue()
    {
        var link = new InverseSquaredLink<double>();
        Assert.True(link.IsCanonical);
        Assert.Equal("InverseSquared", link.Name);
    }

    #endregion

    #region LinkFunctionFactory Tests

    [Theory]
    [InlineData(LinkFunctionType.Identity, "Identity")]
    [InlineData(LinkFunctionType.Logit, "Logit")]
    [InlineData(LinkFunctionType.Log, "Log")]
    [InlineData(LinkFunctionType.Probit, "Probit")]
    [InlineData(LinkFunctionType.Inverse, "Inverse")]
    [InlineData(LinkFunctionType.CLogLog, "CLogLog")]
    [InlineData(LinkFunctionType.Sqrt, "Sqrt")]
    [InlineData(LinkFunctionType.InverseSquared, "InverseSquared")]
    public void Factory_Create_ReturnsCorrectType(LinkFunctionType type, string expectedName)
    {
        var link = LinkFunctionFactory<double>.Create(type);
        Assert.NotNull(link);
        Assert.Equal(expectedName, link.Name);
    }

    [Fact(Timeout = 120000)]
    public async Task Factory_GetAllLinkFunctions_ReturnsAllEight()
    {
        var allLinks = LinkFunctionFactory<double>.GetAllLinkFunctions();
        Assert.Equal(8, allLinks.Count);

        // Verify all expected link functions are present
        var names = new HashSet<string>(allLinks.Values.Select(l => l.Name));
        Assert.Contains("Identity", names);
        Assert.Contains("Logit", names);
        Assert.Contains("Log", names);
        Assert.Contains("Probit", names);
        Assert.Contains("Inverse", names);
        Assert.Contains("CLogLog", names);
        Assert.Contains("Sqrt", names);
        Assert.Contains("InverseSquared", names);
    }

    [Fact(Timeout = 120000)]
    public async Task Factory_GetCanonicalLink_ReturnsCorrectLinks()
    {
        var normalLink = LinkFunctionFactory<double>.GetCanonicalLink(GlmDistributionFamily.Normal);
        Assert.Equal("Identity", normalLink.Name);

        var binomialLink = LinkFunctionFactory<double>.GetCanonicalLink(GlmDistributionFamily.Binomial);
        Assert.Equal("Logit", binomialLink.Name);

        var poissonLink = LinkFunctionFactory<double>.GetCanonicalLink(GlmDistributionFamily.Poisson);
        Assert.Equal("Log", poissonLink.Name);

        var gammaLink = LinkFunctionFactory<double>.GetCanonicalLink(GlmDistributionFamily.Gamma);
        Assert.Equal("Inverse", gammaLink.Name);
    }

    #endregion
}
