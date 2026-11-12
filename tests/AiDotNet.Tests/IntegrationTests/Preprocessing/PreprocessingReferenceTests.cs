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
}
