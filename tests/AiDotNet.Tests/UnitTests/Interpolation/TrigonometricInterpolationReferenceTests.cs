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
/// evaluated by hand in double (Python math.cos / math.sin). The two-argument constructor uses the period
/// (max(x) - min(x)) * N / (N - 1), N times the spacing, as the same formula requires.
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

    // f1(x) = 1 + 0.5 cos(2 pi x / 5) - 2 sin(2 pi x / 5) + 0.75 cos(4 pi x / 5) + 1.5 sin(4 pi x / 5) on x = 0..4. It
    // uses the final sine term (k = n = 2); the default period is (4 - 0) * 5 / 4 = 5. Values from Python math.
    private static readonly double[] F1X = { 0.0, 1.0, 2.0, 3.0, 4.0 };
    private static readonly double[] F1Y =
        { 2.25, -0.4726894027453341, -1.7749010304339403, 3.429409527621414, 1.568180905557861 };

    // f2(x) = -0.5 + cos(w x) + 0.25 sin(w x) - 0.6 cos(2 w x) + 0.3 sin(3 w x), w = 2 pi / 3.5, on x = 1.0, 1.5, ..., 4.0
    // (N = 7, spacing 0.5, not starting at 0); the default period is (4 - 1) * 7 / 6 = 3.5. Values from Python math.
    private static readonly double[] F2X = { 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0 };
    private static readonly double[] F2Y =
    {
        -0.17275707990981592, -1.3741134405837225, -2.1760120574515955, -0.19112214651991022,
        -0.06862063011975247, -0.10000000000000026, 0.582625354584797
    };

    [Fact(Timeout = 60000)]
    public async Task DefaultPeriod_ReproducesEverySample()
    {
        await Task.CompletedTask;
        AssertReproducesSamples(F1X, F1Y);
        AssertReproducesSamples(F2X, F2Y);
    }

    [Theory(Timeout = 60000)]
    [InlineData(1, 0.5, 1.8872855128264683)]
    [InlineData(1, 2.25, -0.3684773795549714)]
    [InlineData(1, 3.7, 2.4085720214524837)]
    [InlineData(2, 1.2, -0.5528715805898318)]
    [InlineData(2, 2.75, 0.2539197213927189)]
    [InlineData(2, 3.9, 0.5873881641143365)]
    public async Task DefaultPeriod_MatchesClosedFormBetweenSamples(int dataSet, double query, double expected)
    {
        await Task.CompletedTask;
        var (x, y) = dataSet == 1 ? (F1X, F1Y) : (F2X, F2Y);
        var interpolation = new TrigonometricInterpolation<double>(new Vector<double>(x), new Vector<double>(y));

        Assert.Equal(expected, interpolation.Interpolate(query), Tolerance);
    }

    private static void AssertReproducesSamples(double[] x, double[] y)
    {
        var interpolation = new TrigonometricInterpolation<double>(new Vector<double>(x), new Vector<double>(y));
        for (int i = 0; i < x.Length; i++)
        {
            Assert.Equal(y[i], interpolation.Interpolate(x[i]), Tolerance);
        }
    }
}
