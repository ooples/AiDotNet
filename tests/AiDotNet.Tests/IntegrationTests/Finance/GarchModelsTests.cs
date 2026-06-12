using AiDotNet.Finance.Volatility;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Finance;

/// <summary>
/// The classical GARCH-family volatility models (Bollerslev 1986, GJR 1993, Nelson 1991), fit by maximum
/// likelihood. Pins that they estimate stationary parameters, forecast positive variance, capture
/// volatility clustering, and (GJR/EGARCH) the leverage effect — negative shocks raise vol more.
/// </summary>
public class GarchModelsTests
{
    // Deterministic returns with volatility clustering: a calm regime then a turbulent regime.
    private static double[] ClusteredReturns(int n)
    {
        var r = new double[n];
        for (int i = 0; i < n; i++)
        {
            double scale = (i % 80) < 40 ? 0.005 : 0.025;          // alternating calm/turbulent
            double osc = ((i * 2654435761u) % 1000) / 1000.0 - 0.5; // deterministic pseudo-noise in [-0.5,0.5]
            r[i] = scale * osc * 2.0;
        }

        return r;
    }

    private static Tensor<double> ToTensor(double[] r) => new(new[] { r.Length }, new Vector<double>(r));

    [Fact]
    public void Garch11_fits_stationary_params_and_forecasts_positive_variance()
    {
        var m = new Garch11Model<double>();
        var r = ClusteredReturns(400);
        m.FitReturns(r);

        double f = m.ForecastNextVariance(r);
        Assert.True(f > 0, "GARCH forecast variance must be positive");

        // α + β ∈ (0,1) by construction of the transform → covariance-stationary.
        var p = m.GetParameters();
        double alpha = p[1], beta = p[2];
        Assert.InRange(alpha + beta, 0.0, 1.0);
    }

    [Fact]
    public void Garch11_forecasts_higher_vol_right_after_a_turbulent_cluster()
    {
        var m = new Garch11Model<double>();
        var r = ClusteredReturns(400);
        m.FitReturns(r);

        // Set params then compare: a series ending in a calm stretch vs one ending in a shock.
        var calm = new double[60];
        for (int i = 0; i < calm.Length; i++) calm[i] = 0.003 * ((i % 2 == 0) ? 1 : -1);
        var shocked = (double[])calm.Clone();
        for (int i = 50; i < 60; i++) shocked[i] = 0.05 * ((i % 2 == 0) ? 1 : -1);

        Assert.True(m.ForecastNextVariance(shocked) > m.ForecastNextVariance(calm));
    }

    [Fact]
    public void GjrGarch_negative_shock_raises_variance_more_than_equal_positive_shock()
    {
        // Leverage effect: with γ > 0, a down-move feeds more into next variance than an up-move.
        var m = new GjrGarchModel<double>();
        m.SetParameters(new Vector<double>(new[] { 1e-5, 0.03, 0.10, 0.85 })); // ω, α, γ, β  (γ>0)

        var up = new double[] { 0.0, 0.04 };   // last move +4%
        var down = new double[] { 0.0, -0.04 }; // last move −4%
        Assert.True(m.ForecastNextVariance(down) > m.ForecastNextVariance(up));
    }

    [Fact]
    public void EGarch_fits_and_forecasts_positive_finite_vol()
    {
        var m = new EGarchModel<double>();
        var r = ClusteredReturns(400);
        m.FitReturns(r);

        double vol = m.ForecastAnnualizedVol(r, periodsPerYear: 252);
        Assert.True(vol > 0 && !double.IsNaN(vol) && !double.IsInfinity(vol), $"EGARCH annualized vol invalid: {vol}");
    }

    [Fact]
    public void Tensor_surface_works_for_the_IVolatilityModel_interface()
    {
        var m = new Garch11Model<double>();
        var r = ClusteredReturns(300);
        m.Train(ToTensor(r), ToTensor(r));        // IModel.Train
        var fc = m.ForecastVolatility(ToTensor(r), horizon: 3); // IVolatilityModel
        Assert.Equal(3, fc.Shape[0]);
        for (int i = 0; i < 3; i++)
        {
            Assert.True(fc.Data.Span[i] > 0);
        }
    }
}
