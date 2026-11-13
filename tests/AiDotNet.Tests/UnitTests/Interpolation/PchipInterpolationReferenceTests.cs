using AiDotNet.Interpolation;
using AiDotNet.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.Interpolation;

/// <summary>
/// Compares <see cref="PchipInterpolation{T}"/> with SciPy 1.18.1 <c>scipy.interpolate.PchipInterpolator</c>
/// (Fritsch-Butland derivatives, one-sided three-point ends).
/// </summary>
public class PchipInterpolationReferenceTests
{
    private const double Tolerance = 1e-12;

    [Theory(Timeout = 60000)]
    // Uneven spacing: the interior slopes are width-weighted harmonic means of the neighbouring secants
    [InlineData(new[] { 0.0, 1.0, 3.0, 4.0, 7.0, 8.0 }, new[] { 0.0, 1.0, 2.0, 2.5, 1.0, 4.0 },
        new[] { 0.5, 2.0, 3.5, 5.0, 7.5, 8.0 },
        new[] { 0.5592948717948718, 1.5480769230769231, 2.3125, 2.111111111111111, 2.015625, 4.0 })]
    // Strictly increasing data: the second-to-last knot has its own slope (2.885...), not 0
    [InlineData(new[] { 0.0, 0.5, 2.0, 3.0, 5.0 }, new[] { 0.0, 1.0, 1.2, 4.0, 10.0 },
        new[] { 0.25, 1.0, 2.5, 4.0 },
        new[] { 0.6358739837398374, 1.0867849753766354, 2.273183944841172, 6.938040712468194 })]
    // Three points around a maximum: the middle knot is an extremum with slope 0
    [InlineData(new[] { 0.0, 1.0, 4.0 }, new[] { 1.0, 3.0, 2.0 },
        new[] { 0.5, 2.0, 3.0 },
        new[] { 2.3229166666666665, 2.962962962962963, 2.7037037037037037 })]
    // Two points: a straight line
    [InlineData(new[] { 0.0, 2.0 }, new[] { 1.0, 5.0 },
        new[] { 0.5, 1.5 },
        new[] { 2.0, 4.0 })]
    public async Task Interpolate_MatchesScipyPchipInterpolator(double[] x, double[] y, double[] queries, double[] expected)
    {
        await Task.CompletedTask;
        var interpolation = new PchipInterpolation<double>(new Vector<double>(x), new Vector<double>(y));

        for (int i = 0; i < queries.Length; i++)
        {
            Assert.Equal(expected[i], interpolation.Interpolate(queries[i]), Tolerance);
        }
    }
}
