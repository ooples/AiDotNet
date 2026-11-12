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
