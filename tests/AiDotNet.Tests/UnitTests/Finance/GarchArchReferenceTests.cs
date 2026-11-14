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

    // arch 9.0.0: res = arch_model(y, mean="Constant", dist="normal", rescale=False).fix(params);
    // res.loglikelihood and res.forecast(horizon=1, reindex=False).variance[-1, 0].
    private const double GarchLogLikelihood = 408.31206719564096;
    private const double GarchForecast = 6.8481728899230196e-05;
    private const double GjrLogLikelihood = 406.04452627088887;
    private const double GjrForecast = 5.2306575840858528e-05;
    private const double EGarchLogLikelihood = 389.11561489597011;
    private const double EGarchForecast = 0.00017796443152331992;
}
