using AiDotNet.AnomalyDetection.Statistical;
using AiDotNet.Preprocessing.OutlierHandling;
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

    // numpy.percentile([1, 2, 3, 4, 100], [25, 75]) == [2, 4] -> IQR 2, bounds 2 - 3 = -1 and 4 + 3 = 7.
    [Fact(Timeout = 120000)]
    public async Task IQRClipper_UsesNumpyPercentiles_AndKeepsNaN()
    {
        await Task.CompletedTask;
        var clipper = new IQRClipper<double>();
        clipper.Fit(M(new double[,] { { 1 }, { 2 }, { double.NaN }, { 3 }, { 4 }, { 100 } }));

        Assert.Equal(2.0, clipper.Q1Values?[0] ?? double.NaN, Tolerance);
        Assert.Equal(4.0, clipper.Q3Values?[0] ?? double.NaN, Tolerance);
        Assert.Equal(-1.0, clipper.LowerBounds?[0] ?? double.NaN, Tolerance);
        Assert.Equal(7.0, clipper.UpperBounds?[0] ?? double.NaN, Tolerance);

        var result = clipper.Transform(M(new double[,] { { -5 }, { double.NaN }, { 100 } }));
        Assert.Equal(-1.0, result[0, 0], Tolerance);
        Assert.True(double.IsNaN(result[1, 0]));
        Assert.Equal(7.0, result[2, 0], Tolerance);
        Assert.Equal(new[] { 2 }, clipper.CountOutliersPerFeature(M(new double[,] { { -5 }, { 0 }, { 100 } })));
    }

    // Iglewicz & Hoaglin: the clip bound is exactly where the modified Z-score 0.6745 * (x - median) / MAD
    // reaches the threshold, so GetModifiedZScores at the bound returns the threshold.
    [Fact(Timeout = 120000)]
    public async Task MADClipper_BoundsAgreeWithModifiedZScore()
    {
        await Task.CompletedTask;
        var clipper = new MADClipper<double>(threshold: 3.5);
        clipper.Fit(M(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 100 } }));

        double upper = clipper.UpperBounds?[0] ?? double.NaN;
        double lower = clipper.LowerBounds?[0] ?? double.NaN;
        Assert.Equal(3.0 + 3.5 / 0.6745, upper, Tolerance);
        var z = clipper.GetModifiedZScores(M(new double[,] { { upper }, { lower } }));
        Assert.Equal(3.5, z[0, 0], Tolerance);
        Assert.Equal(-3.5, z[1, 0], Tolerance);
    }

    // ZScoreClipper: population std, NaN ignored by the fit and kept by transform.
    [Fact(Timeout = 120000)]
    public async Task ZScoreClipper_IgnoresNaNInFit()
    {
        await Task.CompletedTask;
        var clipper = new ZScoreClipper<double>(threshold: 1.0);
        clipper.Fit(M(new double[,] { { 1 }, { double.NaN }, { 3 } }));

        Assert.Equal(2.0, clipper.Means?[0] ?? double.NaN, Tolerance);
        Assert.Equal(1.0, clipper.StandardDeviations?[0] ?? double.NaN, Tolerance);
        var result = clipper.Transform(M(new double[,] { { 10 }, { double.NaN }, { -10 } }));
        Assert.Equal(3.0, result[0, 0], Tolerance);
        Assert.True(double.IsNaN(result[1, 0]));
        Assert.Equal(1.0, result[2, 0], Tolerance);
    }

    // feature-engine Winsorizer(capping_method="iqr", fold=1.5): Q1 - 1.5 IQR, Q3 + 1.5 IQR (pandas quantiles).
    [Fact(Timeout = 120000)]
    public async Task Winsorizer_IqrLimits_MatchFeatureEngine()
    {
        await Task.CompletedTask;
        var winsorizer = new Winsorizer<double>(1.5, 1.5, WinsorizerLimitType.IQR);
        winsorizer.Fit(M(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 100 } }));

        Assert.Equal(-1.0, winsorizer.LowerBounds?[0] ?? double.NaN, Tolerance);
        Assert.Equal(7.0, winsorizer.UpperBounds?[0] ?? double.NaN, Tolerance);
    }

    // OutlierClipper: numpy.percentile([0..10], [10, 90]) == [1, 9].
    [Fact(Timeout = 120000)]
    public async Task OutlierClipper_UsesNumpyPercentiles()
    {
        await Task.CompletedTask;
        var clipper = new OutlierClipper<double>(10, 90);
        var data = new double[11, 1];
        for (int i = 0; i <= 10; i++)
            data[i, 0] = i;
        var result = clipper.FitTransform(M(data));

        Assert.Equal(1.0, result[0, 0], Tolerance);
        Assert.Equal(5.0, result[5, 0], Tolerance);
        Assert.Equal(9.0, result[10, 0], Tolerance);
    }

    // columnIndices of DetectorBasedFilter are "the columns to consider for anomaly detection": the detector
    // must be fitted on and score only those columns.
    [Fact(Timeout = 120000)]
    public async Task DetectorBasedFilter_DetectsOnSelectedColumnsOnly()
    {
        await Task.CompletedTask;
        var data = M(new double[,]
        {
            { 1, 0 }, { 2, 0 }, { 3, 0 }, { 4, 0 }, { 50, 0 }, { 6, 0 }, { 7, 0 }, { 8, 0 }, { 9, 1000 }, { 10, 0 }
        });
        var filter = new DetectorBasedFilter<double>(new ZScoreDetector<double>(), FilterMode.Flag, new[] { 0 });
        filter.Fit(data);

        var reference = new ZScoreDetector<double>();
        var column0 = M(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 50 }, { 6 }, { 7 }, { 8 }, { 9 }, { 10 } });
        reference.Fit(column0);
        var expected = reference.ScoreAnomalies(column0);
        var actual = filter.GetAnomalyScores(data);
        for (int i = 0; i < 10; i++)
            Assert.Equal(expected[i], actual[i], Tolerance);

        var flagged = filter.Transform(data);
        Assert.Equal(3, flagged.Columns);
        var restored = filter.InverseTransform(flagged);
        Assert.Equal(2, restored.Columns);
        Assert.Equal(1000.0, restored[8, 1], Tolerance);
    }
}
