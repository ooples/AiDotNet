using AiDotNet.Preprocessing.Scalers;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Reference tests that pin the preprocessing transformers to scikit-learn / imbalanced-learn (or the original
/// paper) on small inputs whose expected values were computed by hand from the reference definitions.
/// If any test fails, the CODE must be fixed - never adjust expected values.
/// </summary>
public class PreprocessingReferenceTests
{
    private const double Tolerance = 1e-10;

    private static Matrix<double> M(double[,] data) => new Matrix<double>(data);

    // sklearn.preprocessing.StandardScaler().fit_transform([[1], [2], [3], [4]]):
    // mean_ = 2.5, var_ = 1.25 (ddof=0), scale_ = sqrt(1.25)
    [Fact(Timeout = 120000)]
    public async Task StandardScaler_UsesPopulationVariance_LikeSklearn()
    {
        await Task.CompletedTask;
        var scaler = new StandardScaler<double>();
        var result = scaler.FitTransform(M(new double[,] { { 1 }, { 2 }, { 3 }, { 4 } }));

        Assert.Equal(Math.Sqrt(1.25), scaler.StandardDeviation?[0] ?? double.NaN, Tolerance);
        Assert.Equal(-1.3416407864998738, result[0, 0], Tolerance);
        Assert.Equal(-0.4472135954999579, result[1, 0], Tolerance);
        Assert.Equal(0.4472135954999579, result[2, 0], Tolerance);
        Assert.Equal(1.3416407864998738, result[3, 0], Tolerance);
    }

    // sklearn ignores NaN in fit and keeps it in transform:
    // StandardScaler().fit_transform([[1], [nan], [3]]) == [[-1], [nan], [1]]
    [Fact(Timeout = 120000)]
    public async Task StandardScaler_IgnoresNaNInFit_KeepsItInTransform()
    {
        await Task.CompletedTask;
        var scaler = new StandardScaler<double>();
        var result = scaler.FitTransform(M(new double[,] { { 1 }, { double.NaN }, { 3 } }));

        Assert.Equal(2.0, scaler.Mean?[0] ?? double.NaN, Tolerance);
        Assert.Equal(-1.0, result[0, 0], Tolerance);
        Assert.True(double.IsNaN(result[1, 0]));
        Assert.Equal(1.0, result[2, 0], Tolerance);
    }

    // sklearn's _is_constant_feature: a column whose variance is only floating-point noise is left unscaled
    // (scale_ = 1) instead of being blown up by a ~1e-17 standard deviation.
    [Fact(Timeout = 120000)]
    public async Task StandardScaler_NearConstantColumn_IsNotBlownUp()
    {
        await Task.CompletedTask;
        double v = 0.1 + 0.2;
        var scaler = new StandardScaler<double>();
        var result = scaler.FitTransform(M(new double[,] { { v }, { v }, { v }, { 0.3 } }));

        Assert.Equal(1.0, scaler.StandardDeviation?[0] ?? double.NaN, Tolerance);
        for (int i = 0; i < 4; i++)
            Assert.True(Math.Abs(result[i, 0]) < 1e-12, $"row {i}: {result[i, 0]}");
    }

    // sklearn.preprocessing.MinMaxScaler().fit_transform([[5, 1], [5, 2], [5, 3]]): a zero data range is
    // treated as 1, so the constant column maps to the feature-range minimum, and inverse_transform recovers it.
    [Fact(Timeout = 120000)]
    public async Task MinMaxScaler_ConstantColumn_MapsToRangeMin_AndInverts()
    {
        await Task.CompletedTask;
        var scaler = new MinMaxScaler<double>(-1.0, 1.0);
        var data = M(new double[,] { { 5, 1 }, { 5, 2 }, { 5, 3 } });
        var result = scaler.FitTransform(data);

        for (int i = 0; i < 3; i++)
            Assert.Equal(-1.0, result[i, 0], Tolerance);
        Assert.Equal(-1.0, result[0, 1], Tolerance);
        Assert.Equal(0.0, result[1, 1], Tolerance);
        Assert.Equal(1.0, result[2, 1], Tolerance);

        var restored = scaler.InverseTransform(result);
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(5.0, restored[i, 0], Tolerance);
            Assert.Equal(i + 1.0, restored[i, 1], Tolerance);
        }
    }

    // sklearn MinMaxScaler uses nanmin / nanmax: fit_transform([[0], [nan], [10]]) == [[0], [nan], [1]]
    [Fact(Timeout = 120000)]
    public async Task MinMaxScaler_IgnoresNaNInFit()
    {
        await Task.CompletedTask;
        var scaler = new MinMaxScaler<double>();
        var result = scaler.FitTransform(M(new double[,] { { 0 }, { double.NaN }, { 10 } }));

        Assert.Equal(0.0, result[0, 0], Tolerance);
        Assert.True(double.IsNaN(result[1, 0]));
        Assert.Equal(1.0, result[2, 0], Tolerance);
    }

    // Columns outside columnIndices pass through untouched even with a non-[0, 1] feature range.
    [Fact(Timeout = 120000)]
    public async Task MinMaxScaler_UnselectedColumns_PassThrough()
    {
        await Task.CompletedTask;
        var scaler = new MinMaxScaler<double>(-1.0, 1.0, new[] { 1 });
        var result = scaler.FitTransform(M(new double[,] { { 3, 0 }, { 7, 4 } }));

        Assert.Equal(3.0, result[0, 0], Tolerance);
        Assert.Equal(7.0, result[1, 0], Tolerance);
        Assert.Equal(-1.0, result[0, 1], Tolerance);
        Assert.Equal(1.0, result[1, 1], Tolerance);
    }

    // sklearn.preprocessing.MaxAbsScaler: max_abs_ = nanmax(|X|); NaN is kept by transform.
    [Fact(Timeout = 120000)]
    public async Task MaxAbsScaler_MatchesSklearn_WithNaN()
    {
        await Task.CompletedTask;
        var scaler = new MaxAbsScaler<double>();
        var result = scaler.FitTransform(M(new double[,] { { 1, -2 }, { double.NaN, 4 }, { -3, 0 } }));

        Assert.Equal(1.0 / 3.0, result[0, 0], Tolerance);
        Assert.Equal(-0.5, result[0, 1], Tolerance);
        Assert.True(double.IsNaN(result[1, 0]));
        Assert.Equal(1.0, result[1, 1], Tolerance);
        Assert.Equal(-1.0, result[2, 0], Tolerance);
        Assert.Equal(0.0, result[2, 1], Tolerance);
    }

    // sklearn.preprocessing.RobustScaler().fit_transform([[1], [2], [nan], [3], [4], [100]]):
    // center_ = nanmedian = 3, scale_ = nanpercentile(75) - nanpercentile(25) = 4 - 2 = 2
    [Fact(Timeout = 120000)]
    public async Task RobustScaler_MatchesSklearn_WithNaN()
    {
        await Task.CompletedTask;
        var scaler = new RobustScaler<double>();
        var result = scaler.FitTransform(M(new double[,] { { 1 }, { 2 }, { double.NaN }, { 3 }, { 4 }, { 100 } }));

        Assert.Equal(3.0, scaler.Median?[0] ?? double.NaN, Tolerance);
        Assert.Equal(2.0, scaler.InterquartileRange?[0] ?? double.NaN, Tolerance);
        Assert.Equal(-1.0, result[0, 0], Tolerance);
        Assert.Equal(-0.5, result[1, 0], Tolerance);
        Assert.True(double.IsNaN(result[2, 0]));
        Assert.Equal(48.5, result[5, 0], Tolerance);
    }

    // sklearn.preprocessing.Normalizer: rows divided by their norm, all-zero rows left at zero.
    [Fact(Timeout = 120000)]
    public async Task Normalizer_MatchesSklearn_L2AndMax()
    {
        await Task.CompletedTask;
        var l2 = new Normalizer<double>(NormType.L2).FitTransform(M(new double[,] { { 3, 4 }, { 0, 0 } }));
        Assert.Equal(0.6, l2[0, 0], Tolerance);
        Assert.Equal(0.8, l2[0, 1], Tolerance);
        Assert.Equal(0.0, l2[1, 0], Tolerance);
        Assert.Equal(0.0, l2[1, 1], Tolerance);

        var max = new Normalizer<double>(NormType.Max).FitTransform(M(new double[,] { { 1, -4 } }));
        Assert.Equal(0.25, max[0, 0], Tolerance);
        Assert.Equal(-1.0, max[0, 1], Tolerance);

        var l1 = new Normalizer<double>(NormType.L1).FitTransform(M(new double[,] { { 1, -3 } }));
        Assert.Equal(0.25, l1[0, 0], Tolerance);
        Assert.Equal(-0.75, l1[0, 1], Tolerance);
    }

    // Decimal scaling (Han & Kamber): divide by the smallest power of ten above max|x|.
    [Fact(Timeout = 120000)]
    public async Task DecimalScaler_UsesSmallestPowerOfTenAboveMaxAbs()
    {
        await Task.CompletedTask;
        var scaler = new DecimalScaler<double>();
        scaler.Fit(M(new double[,] { { 999, 1000, 0.5, 1, 0 }, { -5, 3, -0.25, 0, 0 } }));

        Assert.Equal(1000.0, scaler.Scale?[0] ?? double.NaN, Tolerance);
        Assert.Equal(10000.0, scaler.Scale?[1] ?? double.NaN, Tolerance);
        Assert.Equal(1.0, scaler.Scale?[2] ?? double.NaN, Tolerance);
        Assert.Equal(10.0, scaler.Scale?[3] ?? double.NaN, Tolerance);
        Assert.Equal(1.0, scaler.Scale?[4] ?? double.NaN, Tolerance);
    }
}
