using AiDotNet.Interpolation;
using AiDotNet.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.Interpolation;

/// <summary>
/// Reference values for <see cref="TrigonometricInterpolation{T}"/>. For N = 2n + 1 equally spaced samples over one
/// period P, the trigonometric interpolant is a0 / 2 + sum_k (a_k cos(2 pi k x / P) + b_k sin(2 pi k x / P)) with
/// a_k = (2 / N) sum_j y_j cos(2 pi k x_j / P) (Zygmund, Trigonometric Series, ch. X; also the odd-N formula in
/// https://en.wikipedia.org/wiki/Trigonometric_interpolation). A trigonometric polynomial of degree at most n is
/// reproduced exactly, so the expected values are the closed form f(x) = 3 + 2 cos(2 pi x / 5) + sin(2 pi x / 5),
/// evaluated by hand in double (Python math.cos / math.sin).
/// </summary>
public class TrigonometricInterpolationReferenceTests
{
    private const double Tolerance = 1e-12;

    [Theory(Timeout = 60000)]
    [InlineData(0.0, 5.0)]
    [InlineData(1.0, 4.5690905050450485)]
    [InlineData(2.0, 1.9697512635425785)]
    [InlineData(3.0, 0.7941807589576321)]
    [InlineData(4.0, 2.666977472454741)]
    [InlineData(0.5, 5.205819241042368)]
    [InlineData(1.75, 2.6334464897900016)]
    [InlineData(3.2, 0.9546387777268316)]
    public async Task Interpolate_ReproducesDegreeTwoTrigonometricPolynomial(double query, double expected)
    {
        await Task.CompletedTask;
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
        var y = new Vector<double>(new[] { 5.0, 4.5690905050450485, 1.9697512635425785, 0.7941807589576321, 2.666977472454741 });
        var interpolation = new TrigonometricInterpolation<double>(x, y, 5.0);

        Assert.Equal(expected, interpolation.Interpolate(query), Tolerance);
    }

    [Fact(Timeout = 60000)]
    public async Task Interpolate_ConstantData_ReturnsTheConstant()
    {
        // y = 5 everywhere: a0 = (2 / 3) * 15 = 10, so the interpolant's constant term a0 / 2 is 5 (not 10).
        await Task.CompletedTask;
        var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
        var y = new Vector<double>(new[] { 5.0, 5.0, 5.0 });
        var interpolation = new TrigonometricInterpolation<double>(x, y, 3.0);

        Assert.Equal(5.0, interpolation.Interpolate(0.0), Tolerance);
        Assert.Equal(5.0, interpolation.Interpolate(1.4), Tolerance);
    }
}
