using AiDotNet.WaveletFunctions;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.WaveletFunctions;

/// <summary>
/// Regression tests for the Meyer wavelet's second transition band (2/3 &lt; f &lt;= 4/3).
/// The band argument was written <c>3 / 2 * freq</c>, which integer-divides to <c>1 * freq</c>.
/// </summary>
public class MeyerWaveletPrecisionTests
{
    private static double Vf(double x) => x * x * (3 - 2 * x);

    [Fact(Timeout = 120000)]
    public async Task GetWaveletCoefficients_SecondBand_UsesThreeHalvesFrequencyScale()
    {
        var wavelet = new MeyerWavelet<double>();
        var coefficients = wavelet.GetWaveletCoefficients();
        int size = coefficients.Length;

        int checkedCount = 0;
        for (int i = 0; i < size; i++)
        {
            double freq = (double)i / size;
            if (freq <= 2.0 / 3 || freq > 4.0 / 3)
            {
                continue;
            }

            double expected = Math.Sin(Math.PI / 2 * Vf(Math.PI * (1.5 * freq - 1)));
            Assert.Equal(expected, coefficients[i], 10);
            checkedCount++;
        }

        Assert.True(checkedCount > 0, "No coefficients fell in the second Meyer band.");
    }

    [Fact(Timeout = 120000)]
    public async Task GetWaveletCoefficients_AtThreeQuarters_MatchesHandComputedValue()
    {
        var wavelet = new MeyerWavelet<double>();
        var coefficients = wavelet.GetWaveletCoefficients();
        int index = coefficients.Length * 3 / 4; // freq = 0.75

        // v = pi * (1.5 * 0.75 - 1) = pi / 8; psi = sin(pi/2 * Vf(pi/8)) ~= 0.5112.
        // The truncated 3 / 2 == 1 version gave v = -pi/4 and psi ~= -0.9612.
        double v = Math.PI / 8;
        double expected = Math.Sin(Math.PI / 2 * Vf(v));
        Assert.Equal(expected, coefficients[index], 10);
        Assert.True(coefficients[index] > 0.0);
    }
}
