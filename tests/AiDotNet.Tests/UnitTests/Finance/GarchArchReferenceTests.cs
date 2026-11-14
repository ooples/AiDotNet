using AiDotNet.Finance.Volatility;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// The GARCH family against the arch package (Kevin Sheppard, arch 9.0.0): <c>arch_model(y, mean="Constant",
/// dist="normal", rescale=False).fix(params)</c> with the series below. Matching arch's log-likelihood checks
/// the constant mean, arch's backcast (tau = 75, weights 0.94^i) and the variance recursion together;
/// matching its one-step forecast checks the step past the sample.
/// </summary>
public class GarchArchReferenceTests
{
    private const double Tolerance = 1e-9;

    // y_t = 0.01 sin(0.7 t) + 0.004 cos(2.3 t) + 0.002 sin(0.13 t^2), t = 0..119.
    private static Vector<double> Returns()
    {
        var y = new double[120];
        for (int t = 0; t < y.Length; t++)
        {
            y[t] = (0.01 * Math.Sin(0.7 * t)) + (0.004 * Math.Cos(2.3 * t)) + (0.002 * Math.Sin(0.13 * t * t));
        }

        return new Vector<double>(y);
    }

    private static void AssertMatchesArch(
        ClassicalVolatilityModelBase<double> model, double[] parameters, double archLogLikelihood, double archForecast)
    {
        var p = new Vector<double>(parameters);
        model.SetParameters(p);
        var returns = Returns();

        double negLogLikelihood = model.NegLogLikelihood(p, Tensor<double>.FromVector(returns));
        Assert.Equal(-archLogLikelihood, negLogLikelihood, Tolerance * Math.Abs(archLogLikelihood));

        double forecast = model.ForecastNextVariance(returns);
        Assert.Equal(archForecast, forecast, Tolerance * archForecast);
    }

    [Fact]
    public void Garch11MatchesArch()
        => AssertMatchesArch(new Garch11Model<double>(), new[] { 0.0005, 2e-6, 0.08, 0.90 },
            GarchLogLikelihood, GarchForecast);

    [Fact]
    public void GjrGarchMatchesArch()
        => AssertMatchesArch(new GjrGarchModel<double>(), new[] { 0.0005, 2e-6, 0.05, 0.10, 0.85 },
            GjrLogLikelihood, GjrForecast);

    [Fact]
    public void EGarchMatchesArch()
        => AssertMatchesArch(new EGarchModel<double>(), new[] { 0.0005, -0.4, 0.12, -0.06, 0.95 },
            EGarchLogLikelihood, EGarchForecast);

    // Returns simulated from a GARCH(1,1) with Box-Muller draws from a seeded generator.
    private static double[] SimulatedReturns(int count, int seed)
    {
        var random = new Random(seed);
        var returns = new double[count];
        double variance = 1e-4;
        for (int t = 0; t < count; t++)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = random.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            returns[t] = Math.Sqrt(variance) * z;
            variance = 2e-6 + (0.08 * returns[t] * returns[t]) + (0.9 * variance);
        }

        return returns;
    }

    [Fact(Timeout = 30000)]
    public async Task EGarchFitOf1500ReturnsFinishesInSeconds()
    {
        await Task.Yield();

        // The EGARCH path is a one-pass recursion. Evaluated as a fixed point of whole-tensor sweeps, every
        // likelihood evaluation cost up to n + 1 linear-recurrence solves, and this fit (the size the model
        // family tests use, under their 120 s timeout) did not finish in two minutes.
        var returns = new Vector<double>(SimulatedReturns(1500, seed: 11));
        var model = new EGarchModel<double>();

        model.FitReturns(returns);

        var parameters = model.GetParameters();
        Assert.Equal(model.ParameterCount, parameters.Length);
        Assert.All(parameters.ToArray(), value => Assert.False(double.IsNaN(value) || double.IsInfinity(value)));
    }

    [Fact]
    public void EGarchLikelihoodIsTheCausalRecursion()
    {
        // Strong shock terms, where the path is far from a contraction: ln s2_0 = omega + beta * backcast,
        // ln s2_t = omega + alpha (|z_(t-1)| - sqrt(2 / pi)) + gamma z_(t-1) + beta ln s2_(t-1), z = e / s.
        const double mu = 0.0002, omega = -0.9, alpha = 0.5, gamma = -0.4, beta = 0.9;
        double[] returns = SimulatedReturns(700, seed: 5);

        // arch's backcast: residuals at the sample mean, weights 0.94^i over the first 75, in logs for EGARCH.
        double sampleMean = returns.Average();
        double weightSum = 0.0, backcast = 0.0;
        for (int i = 0; i < 75; i++)
        {
            double weight = Math.Pow(0.94, i);
            weightSum += weight;
            backcast += weight * (returns[i] - sampleMean) * (returns[i] - sampleMean);
        }

        double logVariance = omega + (beta * Math.Log(backcast / weightSum));
        double expected = 0.0;
        for (int t = 0; t < returns.Length; t++)
        {
            double residual = returns[t] - mu;
            expected += 0.5 * (Math.Log(2.0 * Math.PI) + logVariance + (residual * residual * Math.Exp(-logVariance)));
            double z = residual * Math.Exp(-0.5 * logVariance);
            logVariance = omega + (alpha * (Math.Abs(z) - Math.Sqrt(2.0 / Math.PI))) + (gamma * z) + (beta * logVariance);
        }

        var model = new EGarchModel<double>();
        double actual = model.NegLogLikelihood(
            new Vector<double>(new[] { mu, omega, alpha, gamma, beta }),
            Tensor<double>.FromVector(new Vector<double>(returns)));

        Assert.Equal(expected, actual, Tolerance * Math.Abs(expected));
    }

    // arch 9.0.0: res = arch_model(y, mean="Constant", dist="normal", rescale=False).fix(params);
    // res.loglikelihood and res.forecast(horizon=1, reindex=False).variance[-1, 0].
    private const double GarchLogLikelihood = 408.31206719564096;
    private const double GarchForecast = 6.8481728899230196e-05;
    private const double GjrLogLikelihood = 406.04452627088887;
    private const double GjrForecast = 5.2306575840858528e-05;
    private const double EGarchLogLikelihood = 389.11561489597011;
    private const double EGarchForecast = 0.00017796443152331992;
}
