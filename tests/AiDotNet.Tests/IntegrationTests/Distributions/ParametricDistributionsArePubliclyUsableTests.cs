using AiDotNet.Distributions;
using AiDotNet.Scoring;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Distributions;

/// <summary>
/// The parametric distributions must be reachable from outside the assembly.
/// </summary>
/// <remarks>
/// <para>
/// <c>IParametricDistribution&lt;T&gt;</c> and <c>ISamplingDistribution&lt;T&gt;</c> are public, and
/// so is a public surface built on them: <c>ScoringRuleBase&lt;T&gt;.Score</c> and
/// <c>ScoreGradient</c> TAKE one as a parameter, and <c>CRPSMetric.ComputeFromDistributions</c> and
/// <c>NGBoostRegression.PredictDistributionsAsync</c> return arrays of them. But every concrete
/// implementation, and the abstract base, was declared <c>internal</c>.
/// </para>
/// <para>
/// The effect was a public API a consumer could not use. You could receive a distribution from
/// NGBoost and hand it straight back to a scorer, but you could not construct one — so scoring
/// against a known distribution, unit-testing a scoring rule, or building any of the eleven shipped
/// distributions meant reimplementing <c>LogPdf</c>, <c>GradLogPdf</c> and <c>FisherInformation</c>
/// yourself. These tests exercise the surface the way an outside caller would, so the accessibility
/// cannot quietly regress.
/// </para>
/// </remarks>
public class ParametricDistributionsArePubliclyUsableTests
{
    [Fact]
    public void EveryShippedDistributionCanBeConstructed()
    {
        var distributions = new IParametricDistribution<double>[]
        {
            new NormalDistribution<double>(15.0, 0.25),
            new LaplaceDistribution<double>(0.0, 1.0),
            new StudentTDistribution<double>(0.0, 1.0, 5.0),
            new LogNormalDistribution<double>(0.0, 1.0),
            new ExponentialDistribution<double>(2.0),
            new GammaDistribution<double>(2.0, 1.5),
            new BetaDistribution<double>(2.0, 3.0),
            new PoissonDistribution<double>(4.0),
            new NegativeBinomialDistribution<double>(5.0, 0.4),
            new WeibullDistribution<double>(1.5, 2.0),
        };

        foreach (var distribution in distributions)
        {
            Assert.True(distribution.NumParameters > 0,
                $"{distribution.GetType().Name} reported {distribution.NumParameters} parameters");
            Assert.Equal(distribution.NumParameters, distribution.ParameterNames.Length);
            Assert.Equal(distribution.NumParameters, distribution.Parameters.Length);
        }
    }

    [Fact]
    public void TheNormalDistributionMatchesItsClosedForm()
    {
        // Graded against arithmetic done independently, not against itself.
        const double mean = 15.0;
        const double variance = 0.25;
        var distribution = new NormalDistribution<double>(mean, variance);

        foreach (double value in new[] { 13.0, 14.5, 15.0, 15.5, 17.0 })
        {
            double error = value - mean;
            double expected =
                -0.5 * Math.Log(2.0 * Math.PI * variance) - error * error / (2.0 * variance);

            Assert.True(Math.Abs(distribution.LogPdf(value) - expected) < 1e-12,
                $"at x = {value}: {distribution.LogPdf(value)} against {expected}");
            Assert.True(Math.Abs(distribution.Pdf(value) - Math.Exp(expected)) < 1e-12,
                $"at x = {value}: Pdf disagreed with exp(LogPdf)");
        }

        // Fisher information for a normal is diag(1/sigma^2, 1/(2 sigma^4)) -- the off-diagonal
        // zeros say the mean and the variance are estimated independently.
        var information = distribution.FisherInformation();

        Assert.True(Math.Abs(information[0, 0] - 1.0 / variance) < 1e-9,
            $"information[0,0] = {information[0, 0]}");
        Assert.True(Math.Abs(information[1, 1] - 1.0 / (2.0 * variance * variance)) < 1e-9,
            $"information[1,1] = {information[1, 1]}");
        Assert.Equal(0.0, information[0, 1], 12);
        Assert.Equal(0.0, information[1, 0], 12);
    }

    [Fact]
    public void AConstructedDistributionCanBeScoredByThePublicScoringRule()
    {
        // The combination that was impossible before: ScoringRuleBase.Score is public and takes an
        // IParametricDistribution, but nothing public could produce one to pass it.
        var distribution = new NormalDistribution<double>(15.0, 0.25);
        var rule = new LogScore<double>();

        double atCentre = rule.Score(distribution, 15.0);
        double offCentre = rule.Score(distribution, 16.0);

        // double.IsFinite is .NET Core 3.0+; this assembly also targets net471, where the
        // equivalent is the NaN/Infinity pair.
        Assert.True(!double.IsNaN(atCentre) && !double.IsInfinity(atCentre),
            $"the score at the mean was {atCentre}");
        Assert.True(!double.IsNaN(offCentre) && !double.IsInfinity(offCentre),
            $"the score away from the mean was {offCentre}");
        Assert.NotEqual(atCentre, offCentre);

        var gradient = rule.ScoreGradient(distribution, 16.0);
        Assert.Equal(distribution.NumParameters, gradient.Length);
    }

    [Fact]
    public void SamplingGoesThroughThePublicInterfaceToo()
    {
        var distribution = new NormalDistribution<double>(15.0, 0.25);
        var random = new Random(20260818);

        double total = 0.0;
        const int draws = 20000;

        for (int i = 0; i < draws; i++) total += distribution.Sample(random);

        // 20000 draws with a standard error of 0.5/sqrt(20000) = 0.0035, so 0.05 is roughly 14
        // standard errors -- loose enough never to flake, tight enough to catch a broken sampler.
        Assert.True(Math.Abs(total / draws - 15.0) < 0.05,
            $"the sample mean of {draws} draws was {total / draws}, expected about 15");
    }

    [Fact]
    public void TheGradientOfTheLogDensityMatchesAFiniteDifference()
    {
        // GradLogPdf is differentiation with respect to the PARAMETERS, which is what makes maximum
        // likelihood an optimization problem an optimizer can take. Checked against the definition.
        const double mean = 15.0;
        const double variance = 0.25;
        const double observation = 16.0;
        const double step = 1e-6;

        var gradient = new NormalDistribution<double>(mean, variance).GradLogPdf(observation);

        double meanDifference =
            (new NormalDistribution<double>(mean + step, variance).LogPdf(observation)
           - new NormalDistribution<double>(mean - step, variance).LogPdf(observation)) / (2.0 * step);

        double varianceDifference =
            (new NormalDistribution<double>(mean, variance + step).LogPdf(observation)
           - new NormalDistribution<double>(mean, variance - step).LogPdf(observation)) / (2.0 * step);

        Assert.True(Math.Abs(gradient[0] - meanDifference) < 1e-5,
            $"d/dmean: analytic {gradient[0]}, numeric {meanDifference}");
        Assert.True(Math.Abs(gradient[1] - varianceDifference) < 1e-4,
            $"d/dvariance: analytic {gradient[1]}, numeric {varianceDifference}");
    }
}
