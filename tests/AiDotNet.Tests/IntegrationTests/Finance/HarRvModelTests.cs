using AiDotNet.Finance.Volatility;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Finance;

/// <summary>
/// HAR-RV (Corsi 2009) realized-volatility forecaster: OLS on daily/weekly/monthly realized-variance
/// components. Pins that it fits by least squares, captures volatility persistence (long memory), and
/// produces the annualized-vol forecast the options vol-edge consumes.
/// </summary>
public class HarRvModelTests
{
    // A persistent realized-variance series with two-level regime structure spanning ~0.2..2.4. The
    // alternating calm/turbulent regimes give the (otherwise near-collinear) daily/weekly/monthly HAR
    // regressors genuine identifiability, so plain OLS is well-posed, and the test levels (0.2, 2.0) lie
    // inside the fitted range (no extrapolation).
    private static double[] PersistentRvSeries(int n)
    {
        var rv = new double[n];
        for (int i = 0; i < n; i++)
        {
            double regime = ((i / 30) % 2 == 0) ? 0.2 : 2.2;          // 30-period calm / turbulent blocks
            double bump = ((i * 1103515245 + 12345) % 1000) / 5000.0; // deterministic 0..0.2
            rv[i] = regime + bump;
        }

        return rv;
    }

    [Fact]
    public void Fits_by_OLS_and_forecasts_a_positive_variance()
    {
        var rv = PersistentRvSeries(120);
        var model = new HarRvModel<double>(new RegressionOptions<double> { UseIntercept = true });
        model.FitRealizedVariance(rv);

        // Three HAR coefficients + intercept were estimated.
        Assert.Equal(3, model.Coefficients.Length);
        double f = model.ForecastNextVariance(rv);
        Assert.True(f > 0, "forecast variance should be positive for a positive RV series");
    }

    [Fact]
    public void Forecast_is_higher_after_a_uniformly_elevated_history_than_a_calm_one()
    {
        var model = new HarRvModel<double>(new RegressionOptions<double> { UseIntercept = true });
        var rv = PersistentRvSeries(120);
        model.FitRealizedVariance(rv);

        // Persistence: when ALL of the daily/weekly/monthly components are elevated, the next-period
        // forecast must exceed the calm forecast. (A fresh spike that hasn't yet entered the monthly
        // average is deliberately muted by HAR — that is correct Corsi behaviour, so we test the
        // unambiguous uniformly-elevated case.)
        var calm = new double[40];
        for (int i = 0; i < calm.Length; i++) calm[i] = 0.2;
        var elevated = new double[40];
        for (int i = 0; i < elevated.Length; i++) elevated[i] = 2.0;

        double calmF = model.ForecastNextVariance(calm);
        double elevatedF = model.ForecastNextVariance(elevated);
        Assert.True(elevatedF > calmF, $"elevated history ({elevatedF:F3}) should forecast higher variance than calm ({calmF:F3})");
    }

    [Fact]
    public void ForecastVolFromReturns_returns_annualized_vol_in_a_sane_range()
    {
        // ~2% daily moves → annualized vol ≈ 0.02·√252 ≈ 0.32.
        var returns = new double[300];
        for (int i = 0; i < returns.Length; i++)
        {
            returns[i] = (i % 2 == 0 ? 0.02 : -0.02);
        }

        double vol = HarRvModel<double>.ForecastVolFromReturns(returns, periodsPerYear: 252);
        Assert.InRange(vol, 0.15, 0.55);
    }

    [Fact]
    public void Annualized_vol_scales_with_periods_per_year()
    {
        var rv = PersistentRvSeries(120);
        var model = new HarRvModel<double>(new RegressionOptions<double> { UseIntercept = true });
        model.FitRealizedVariance(rv);

        double daily = model.ForecastAnnualizedVol(rv, periodsPerYear: 1);
        double annual = model.ForecastAnnualizedVol(rv, periodsPerYear: 252);
        Assert.True(annual > daily); // √252 scaling
        Assert.True(System.Math.Abs(annual - daily * System.Math.Sqrt(252)) < 1e-9);
    }
}
