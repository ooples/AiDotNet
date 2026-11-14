using System;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// Reference checks for the filters the unobserved components model initializes its cycle with.
/// </summary>
public class UnobservedComponentsReferenceTests
{
    private const double Tolerance = 1e-9;

    private static UnobservedComponentsModel<double, Matrix<double>, Vector<double>> Model() => new();

    [Fact]
    [Trait("category", "unit")]
    public void Hodrick_prescott_trend_matches_statsmodels_hpfilter()
    {
        // statsmodels.tsa.filters.hp_filter.hpfilter(x, lamb=10)[1]
        var x = new Vector<double>(12);
        for (int t = 0; t < 12; t++)
        {
            x[t] = Math.Sin(0.9 * t) + (0.1 * t * t) - (0.4 * Math.Cos(2.3 * t));
        }

        double[] expected =
        {
            -0.09509580768590774, 0.33416742767032376, 0.7329402437951444, 1.1522992105040522,
            1.711897737398252, 2.5162682970469135, 3.6534952888060603, 5.092951080210395,
            6.70814946604045, 8.391955835107082, 10.049761809106625, 11.665793091359202
        };

        var trend = Model().HodrickPrescottFilter(x, 10.0);
        Assert.Equal(expected.Length, trend.Length);
        for (int t = 0; t < expected.Length; t++)
        {
            Assert.Equal(expected[t], trend[t], Tolerance);
        }
    }

    [Fact]
    [Trait("category", "unit")]
    public void Kalman_filter_matches_statsmodels()
    {
        // statsmodels KalmanFilter(k_endog=1, k_states=3): design ones, obs_cov 0.1, transition and state_cov I,
        // initialize_known(0, 2I) (the first prediction of a zero state with unit covariance); filtered_state[0].
        var model = new UnobservedComponentsModel<double, Matrix<double>, Vector<double>>(
            new AiDotNet.Models.Options.UnobservedComponentsOptions<double, Matrix<double>, Vector<double>>
            {
                SeasonalPeriod = 4,
                IncludeCycle = true
            });
        var y = new Vector<double>(8);
        for (int t = 0; t < 8; t++)
        {
            y[t] = Math.Sin(0.8 * t) + (0.2 * t);
        }

        double[] expected =
        {
            0.0, 0.29622467312490997, 0.4611974587953153, 0.42628184140714753, 0.2528101326429333,
            0.08643809568374902, 0.068523601019022, 0.2503724373833497
        };

        model.InitializeKalmanParameters();
        model.KalmanFilter(y);
        var states = model.FilteredStates;
        Assert.Equal(expected.Length, states.Count);
        for (int t = 0; t < expected.Length; t++)
        {
            for (int s = 0; s < 3; s++)
            {
                Assert.Equal(expected[t], states[t][s], Tolerance);
            }
        }
    }

    [Fact]
    [Trait("category", "unit")]
    public void Half_seasonal_indices_divide_by_the_true_occurrence_counts()
    {
        // Period 2 over 5000 points: each position occurs 2500 times. Position 0 is 1 at its first 100 occurrences and
        // 0 after, position 1 always 0, so the indices are 100 / 2500 = 0.04 and 0, centred to +0.02 / -0.02. Counting
        // the occurrences as a 0/1 sum in Half stopped at 2048 and gave +-0.0244.
        const int n = 5000;
        var model = new UnobservedComponentsModel<Half, Matrix<Half>, Vector<Half>>(
            new AiDotNet.Models.Options.UnobservedComponentsOptions<Half, Matrix<Half>, Vector<Half>> { SeasonalPeriod = 2 });
        var y = new Vector<Half>(n);
        for (int t = 0; t < 200; t += 2) y[t] = (Half)1;

        var seasonal = model.InitializeSeasonal(y, new Vector<Half>(n));

        for (int t = 0; t < n; t += 997)
        {
            Assert.Equal(t % 2 == 0 ? 0.02 : -0.02, (double)seasonal[t], 1e-3);
        }
    }

    [Fact]
    [Trait("category", "unit")]
    public void Half_seasonal_sums_of_large_values_do_not_overflow()
    {
        // Period 2 over 400 points: position 0 is always 1000, position 1 always 0, so the indices are 1000 and 0,
        // centred to +500 / -500. Position 0's sum (200 x 1000 = 200,000) is past Half's 65,504; summed in Half it was
        // infinite and every index NaN or infinite.
        const int n = 400;
        var model = new UnobservedComponentsModel<Half, Matrix<Half>, Vector<Half>>(
            new AiDotNet.Models.Options.UnobservedComponentsOptions<Half, Matrix<Half>, Vector<Half>> { SeasonalPeriod = 2 });
        var y = new Vector<Half>(n);
        for (int t = 0; t < n; t += 2) y[t] = (Half)1000;

        var seasonal = model.InitializeSeasonal(y, new Vector<Half>(n));

        for (int t = 0; t < n; t += 37)
        {
            Assert.Equal(t % 2 == 0 ? 500.0 : -500.0, (double)seasonal[t]);
        }
    }

    [Fact]
    [Trait("category", "unit")]
    public void Band_pass_keeps_both_halves_of_the_spectrum_as_numpy()
    {
        // X = np.fft.fft(x); f = np.minimum(k, n - k) / n; np.real(np.fft.ifft(X * ((f >= 1/8) & (f <= 1/4)))).
        // The period-6 and period-4 components pass at full amplitude; the trend and the period-2.4 term do not.
        var x = new Vector<double>(12);
        for (int t = 0; t < 12; t++)
        {
            x[t] = Math.Sin(2 * Math.PI * t / 6) + (0.5 * Math.Sin(2 * Math.PI * t / 2.4)) + (0.3 * t);
        }

        double[] expected =
        {
            -0.6, -0.033974596215561315, 0.8660254037844386, 0.6, -0.5660254037844383, -0.8660254037844385,
            0.0, 0.5660254037844383, 0.2660254037844386, 0.0, 0.033974596215561384, -0.2660254037844386
        };

        var filtered = Model().BandPassFilter(x, 4, 8);
        Assert.Equal(expected.Length, filtered.Length);
        for (int t = 0; t < expected.Length; t++)
        {
            Assert.Equal(expected[t], filtered[t], Tolerance);
        }
    }
}
