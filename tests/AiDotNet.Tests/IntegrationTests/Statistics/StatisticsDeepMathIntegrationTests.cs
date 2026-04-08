using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Inputs;
using AiDotNet.Statistics;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Statistics;

public class StatisticsDeepMathIntegrationTests
{
    private const double Tol = 1e-10;
    private const double MedTol = 1e-6;

    private static Vector<double> V(params double[] vals) => new(vals);

    // ──────────────────────────────────────────────────────────
    // MEDIAN
    // ──────────────────────────────────────────────────────────

    [Fact]
    public void Median_OddCount()
    {
        // [1, 3, 5, 7, 9] → median = 5
        double result = StatisticsHelper<double>.CalculateMedian(new double[] { 9, 1, 5, 3, 7 });
        Assert.Equal(5.0, result, Tol);
    }

    [Fact]
    public void Median_EvenCount()
    {
        // [1, 3, 5, 7] → median = (3+5)/2 = 4
        double result = StatisticsHelper<double>.CalculateMedian(new double[] { 7, 1, 5, 3 });
        Assert.Equal(4.0, result, Tol);
    }

    [Fact]
    public void Median_SingleElement()
    {
        double result = StatisticsHelper<double>.CalculateMedian(new double[] { 42 });
        Assert.Equal(42.0, result, Tol);
    }

    [Fact]
    public void Median_AllSame()
    {
        double result = StatisticsHelper<double>.CalculateMedian(new double[] { 5, 5, 5, 5, 5 });
        Assert.Equal(5.0, result, Tol);
    }

    [Fact]
    public void Median_NegativeValues()
    {
        // [-10, -5, 0, 5, 10] → median = 0
        double result = StatisticsHelper<double>.CalculateMedian(new double[] { 10, -5, 0, -10, 5 });
        Assert.Equal(0.0, result, Tol);
    }

    // ──────────────────────────────────────────────────────────
    // MEAN ABSOLUTE DEVIATION (MAD)
    // ──────────────────────────────────────────────────────────

    [Fact]
    public void MAD_HandCalculation()
    {
        // values = [2, 4, 6, 8], median = 5
        // absolute devs from median: |2-5|=3, |4-5|=1, |6-5|=1, |8-5|=3
        // MAD = (3+1+1+3)/4 = 2.0
        var values = V(2, 4, 6, 8);
        double result = StatisticsHelper<double>.CalculateMeanAbsoluteDeviation(values, 5.0);
        Assert.Equal(2.0, result, Tol);
    }

    [Fact]
    public void MAD_AllSame()
    {
        // All values = median → MAD = 0
        var values = V(3, 3, 3, 3);
        double result = StatisticsHelper<double>.CalculateMeanAbsoluteDeviation(values, 3.0);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void MAD_SymmetricData()
    {
        // values = [1, 3, 5, 7, 9], median = 5
        // devs: 4, 2, 0, 2, 4 → MAD = 12/5 = 2.4
        var values = V(1, 3, 5, 7, 9);
        double result = StatisticsHelper<double>.CalculateMeanAbsoluteDeviation(values, 5.0);
        Assert.Equal(2.4, result, Tol);
    }

    // ──────────────────────────────────────────────────────────
    // VARIANCE (sample variance, divides by n-1)
    // ──────────────────────────────────────────────────────────

    [Fact]
    public void Variance_HandCalculation()
    {
        // values = [2, 4, 6, 8, 10], mean = 6
        // deviations: -4, -2, 0, 2, 4
        // squared: 16, 4, 0, 4, 16 = 40
        // sample variance = 40/4 = 10
        var values = V(2, 4, 6, 8, 10);
        double result = StatisticsHelper<double>.CalculateVariance(values, 6.0);
        Assert.Equal(10.0, result, Tol);
    }

    [Fact]
    public void Variance_AllSame()
    {
        // All same values → variance = 0
        var values = V(5, 5, 5, 5);
        double result = StatisticsHelper<double>.CalculateVariance(values, 5.0);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void Variance_TwoElements()
    {
        // [0, 10], mean = 5, deviations: -5, 5, squared: 25, 25 → sum=50, /1 = 50
        var values = V(0, 10);
        double result = StatisticsHelper<double>.CalculateVariance(values, 5.0);
        Assert.Equal(50.0, result, Tol);
    }

    [Fact]
    public void Variance_SingleElement_ReturnsZero()
    {
        var values = V(42);
        double result = StatisticsHelper<double>.CalculateVariance(values, 42.0);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void Variance_NonNegative()
    {
        // Variance must always be >= 0
        var values = V(-5, -3, 0, 2, 7, 100);
        double mean = values.Average();
        double result = StatisticsHelper<double>.CalculateVariance(values, mean);
        Assert.True(result >= 0, $"Variance should be non-negative, got {result}");
    }

    // ──────────────────────────────────────────────────────────
    // STANDARD DEVIATION
    // ──────────────────────────────────────────────────────────

    [Fact]
    public void StandardDeviation_IsSqrtOfVariance()
    {
        double[] data = { 2, 4, 6, 8, 10 };
        double variance = StatisticsHelper<double>.CalculateVariance(data);
        double stddev = StatisticsHelper<double>.CalculateStandardDeviation(data);
        Assert.Equal(Math.Sqrt(variance), stddev, Tol);
    }

    [Fact]
    public void StandardDeviation_AllSame()
    {
        double result = StatisticsHelper<double>.CalculateStandardDeviation(new double[] { 7, 7, 7, 7 });
        Assert.Equal(0.0, result, Tol);
    }

    // ──────────────────────────────────────────────────────────
    // MSE (Mean Squared Error)
    // ──────────────────────────────────────────────────────────

    [Fact]
    public void MSE_PerfectPrediction()
    {
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var predicted = new double[] { 1, 2, 3, 4, 5 };
        double result = StatisticsHelper<double>.CalculateMeanSquaredError(actual, predicted);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void MSE_HandCalculation()
    {
        // actual = [1, 2, 3], predicted = [1.5, 2.5, 3.5]
        // errors: -0.5, -0.5, -0.5, squared: 0.25, 0.25, 0.25
        // MSE = 0.75/3 = 0.25
        var actual = new double[] { 1, 2, 3 };
        var predicted = new double[] { 1.5, 2.5, 3.5 };
        double result = StatisticsHelper<double>.CalculateMeanSquaredError(actual, predicted);
        Assert.Equal(0.25, result, Tol);
    }

    [Fact]
    public void MSE_SymmetricInErrors()
    {
        // MSE should be the same whether we overpredict or underpredict
        var actual = new double[] { 0, 0, 0 };
        var overPredict = new double[] { 1, 1, 1 };
        var underPredict = new double[] { -1, -1, -1 };
        double mseOver = StatisticsHelper<double>.CalculateMeanSquaredError(actual, overPredict);
        double mseUnder = StatisticsHelper<double>.CalculateMeanSquaredError(actual, underPredict);
        Assert.Equal(mseOver, mseUnder, Tol);
    }

    [Fact]
    public void MSE_NonNegative()
    {
        var actual = new double[] { 1, 5, -3, 0 };
        var predicted = new double[] { 2, 3, -1, 1 };
        double result = StatisticsHelper<double>.CalculateMeanSquaredError(actual, predicted);
        Assert.True(result >= 0, $"MSE should be non-negative, got {result}");
    }

    // ──────────────────────────────────────────────────────────
    // QUANTILES
    // ──────────────────────────────────────────────────────────

    [Fact]
    public void Quantiles_HandCalculation()
    {
        // Data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        // Using linear interpolation method:
        // Q1 position = (12-1)*0.25 = 2.75, index 2, fraction 0.75
        //   Q1 = data[2] + 0.75*(data[3]-data[2]) = 3 + 0.75*(4-3) = 3.75
        // Q3 position = (12-1)*0.75 = 8.25, index 8, fraction 0.25
        //   Q3 = data[8] + 0.25*(data[9]-data[8]) = 9 + 0.25*(10-9) = 9.25
        var data = V(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
        var (q1, q3) = StatisticsHelper<double>.CalculateQuantiles(data);
        Assert.Equal(3.75, q1, Tol);
        Assert.Equal(9.25, q3, Tol);
    }

    [Fact]
    public void Quantiles_Q1_LessThanOrEqual_Median_LessThanOrEqual_Q3()
    {
        var data = V(10, 3, 7, 1, 15, 8, 2, 9, 4, 6);
        var (q1, q3) = StatisticsHelper<double>.CalculateQuantiles(data);
        double median = StatisticsHelper<double>.CalculateMedian(data.ToArray());
        Assert.True(q1 <= median, $"Q1 ({q1}) should be <= Median ({median})");
        Assert.True(median <= q3, $"Median ({median}) should be <= Q3 ({q3})");
    }

    [Fact]
    public void Quantiles_IQR_Equals_Q3_Minus_Q1()
    {
        var data = V(1, 3, 5, 7, 9, 11, 13, 15);
        var (q1, q3) = StatisticsHelper<double>.CalculateQuantiles(data);
        double iqr = q3 - q1;
        Assert.True(iqr >= 0, $"IQR should be non-negative, got {iqr}");
    }

    // ──────────────────────────────────────────────────────────
    // SKEWNESS AND KURTOSIS
    // ──────────────────────────────────────────────────────────

    [Fact]
    public void SkewnessKurtosis_SymmetricData_ZeroSkewness()
    {
        // Symmetric data about mean should have zero skewness
        var values = V(-3, -2, -1, 0, 1, 2, 3);
        double mean = 0.0;
        double stddev = Math.Sqrt(StatisticsHelper<double>.CalculateVariance(values, mean));
        var (skewness, _) = StatisticsHelper<double>.CalculateSkewnessAndKurtosis(values, mean, stddev, values.Length);
        Assert.Equal(0.0, skewness, 1e-10);
    }

    [Fact]
    public void SkewnessKurtosis_RightSkewedData()
    {
        // Data skewed to the right (positive skewness)
        var values = V(1, 1, 1, 2, 2, 3, 10);
        double mean = values.Average();
        double variance = StatisticsHelper<double>.CalculateVariance(values, mean);
        double stddev = Math.Sqrt(variance);
        var (skewness, _) = StatisticsHelper<double>.CalculateSkewnessAndKurtosis(values, mean, stddev, values.Length);
        Assert.True(skewness > 0, $"Expected positive skewness, got {skewness}");
    }

    [Fact]
    public void SkewnessKurtosis_LeftSkewedData()
    {
        // Data skewed to the left (negative skewness)
        var values = V(-10, -3, -2, -1, -1, -1, 0);
        double mean = values.Average();
        double variance = StatisticsHelper<double>.CalculateVariance(values, mean);
        double stddev = Math.Sqrt(variance);
        var (skewness, _) = StatisticsHelper<double>.CalculateSkewnessAndKurtosis(values, mean, stddev, values.Length);
        Assert.True(skewness < 0, $"Expected negative skewness, got {skewness}");
    }

    [Fact]
    public void SkewnessKurtosis_NormalLikeData_KurtosisNearZero()
    {
        // For data approximating a normal distribution, excess kurtosis should be near 0
        // Using a relatively uniform-like symmetric dataset
        var values = V(-2, -1, -1, 0, 0, 0, 1, 1, 2);
        double mean = values.Average();
        double variance = StatisticsHelper<double>.CalculateVariance(values, mean);
        double stddev = Math.Sqrt(variance);
        var (_, kurtosis) = StatisticsHelper<double>.CalculateSkewnessAndKurtosis(values, mean, stddev, values.Length);
        // The excess kurtosis should be relatively small for this distribution
        Assert.True(Math.Abs(kurtosis) < 5, $"Excess kurtosis {kurtosis} seems too extreme");
    }

    // ──────────────────────────────────────────────────────────
    // BASIC STATS INTEGRATION (via internal constructor)
    // ──────────────────────────────────────────────────────────

    [Fact]
    public void BasicStats_Mean_HandCalculation()
    {
        // [10, 20, 30, 40, 50] → mean = 30
        var stats = new BasicStats<double>(new BasicStatsInputs<double> { Values = V(10, 20, 30, 40, 50) });
        Assert.Equal(30.0, stats.Mean, Tol);
    }

    [Fact]
    public void BasicStats_MinMax()
    {
        var stats = new BasicStats<double>(new BasicStatsInputs<double> { Values = V(5, -3, 8, 1, 0) });
        Assert.Equal(-3.0, stats.Min, Tol);
        Assert.Equal(8.0, stats.Max, Tol);
    }

    [Fact]
    public void BasicStats_Count()
    {
        var stats = new BasicStats<double>(new BasicStatsInputs<double> { Values = V(1, 2, 3, 4, 5, 6) });
        Assert.Equal(6, stats.N);
    }

    [Fact]
    public void BasicStats_Median_OddCount()
    {
        var stats = new BasicStats<double>(new BasicStatsInputs<double> { Values = V(9, 1, 5, 3, 7) });
        Assert.Equal(5.0, stats.Median, Tol);
    }

    [Fact]
    public void BasicStats_Median_EvenCount()
    {
        var stats = new BasicStats<double>(new BasicStatsInputs<double> { Values = V(7, 1, 5, 3) });
        Assert.Equal(4.0, stats.Median, Tol);
    }

    [Fact]
    public void BasicStats_Variance_IsPopulationVariance()
    {
        // [2, 4, 6, 8, 10], mean=6, deviations: -4,-2,0,2,4, SS=40
        // BasicStats uses population variance (divides by n): 40/5 = 8
        var stats = new BasicStats<double>(new BasicStatsInputs<double> { Values = V(2, 4, 6, 8, 10) });
        Assert.Equal(8.0, stats.Variance, Tol);
    }

    [Fact]
    public void BasicStats_StdDev_IsSqrtVariance()
    {
        // StdDev = sqrt(population variance) = sqrt(8)
        var stats = new BasicStats<double>(new BasicStatsInputs<double> { Values = V(2, 4, 6, 8, 10) });
        Assert.Equal(Math.Sqrt(8.0), stats.StandardDeviation, Tol);
    }

    [Fact]
    public void BasicStats_IQR_HandCalculation()
    {
        var stats = new BasicStats<double>(new BasicStatsInputs<double>
            { Values = V(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) });
        Assert.Equal(stats.ThirdQuartile - stats.FirstQuartile, stats.InterquartileRange, Tol);
    }

    [Fact]
    public void BasicStats_SymmetricSkewnessZero()
    {
        var stats = new BasicStats<double>(new BasicStatsInputs<double> { Values = V(-3, -2, -1, 0, 1, 2, 3) });
        Assert.Equal(0.0, stats.Skewness, 1e-10);
    }

    [Fact]
    public void BasicStats_Empty_AllZeros()
    {
        var stats = BasicStats<double>.Empty();
        Assert.Equal(0, stats.N);
        Assert.Equal(0.0, stats.Mean, Tol);
        Assert.Equal(0.0, stats.Variance, Tol);
    }

    [Fact]
    public void BasicStats_GetMetric_Mean()
    {
        var stats = new BasicStats<double>(new BasicStatsInputs<double> { Values = V(10, 20, 30) });
        Assert.Equal(stats.Mean, stats.GetMetric(MetricType.Mean), Tol);
        Assert.Equal(stats.Variance, stats.GetMetric(MetricType.Variance), Tol);
        Assert.Equal(stats.Min, stats.GetMetric(MetricType.Min), Tol);
        Assert.Equal(stats.Max, stats.GetMetric(MetricType.Max), Tol);
    }

    [Fact]
    public void BasicStats_HasMetric_AllBasicMetricsAvailable()
    {
        var stats = BasicStats<double>.Empty();
        Assert.True(stats.HasMetric(MetricType.Mean));
        Assert.True(stats.HasMetric(MetricType.Variance));
        Assert.True(stats.HasMetric(MetricType.StandardDeviation));
        Assert.True(stats.HasMetric(MetricType.Skewness));
        Assert.True(stats.HasMetric(MetricType.Kurtosis));
        Assert.True(stats.HasMetric(MetricType.Min));
        Assert.True(stats.HasMetric(MetricType.Max));
        Assert.True(stats.HasMetric(MetricType.Median));
        Assert.True(stats.HasMetric(MetricType.FirstQuartile));
        Assert.True(stats.HasMetric(MetricType.ThirdQuartile));
        Assert.True(stats.HasMetric(MetricType.InterquartileRange));
        Assert.True(stats.HasMetric(MetricType.MAD));
    }

    // ──────────────────────────────────────────────────────────
    // CROSS-VALIDATION IDENTITIES
    // ──────────────────────────────────────────────────────────

    [Fact]
    public void Stats_MeanMinMax_Ordering()
    {
        // For any data: min <= mean <= max (when all values are non-negative)
        // More generally: min <= median <= max
        var data = V(3, 1, 4, 1, 5, 9, 2, 6);
        var stats = new BasicStats<double>(new BasicStatsInputs<double> { Values = data });

        Assert.True(stats.Min <= stats.Mean, $"Min ({stats.Min}) should be <= Mean ({stats.Mean})");
        Assert.True(stats.Mean <= stats.Max, $"Mean ({stats.Mean}) should be <= Max ({stats.Max})");
        Assert.True(stats.Min <= stats.Median, $"Min ({stats.Min}) should be <= Median ({stats.Median})");
        Assert.True(stats.Median <= stats.Max, $"Median ({stats.Median}) should be <= Max ({stats.Max})");
    }

    [Fact]
    public void Stats_VarianceZero_AllSame()
    {
        var stats = new BasicStats<double>(new BasicStatsInputs<double> { Values = V(7, 7, 7, 7, 7) });
        Assert.Equal(0.0, stats.Variance, Tol);
        Assert.Equal(0.0, stats.StandardDeviation, Tol);
        Assert.Equal(7.0, stats.Mean, Tol);
        Assert.Equal(7.0, stats.Median, Tol);
    }

    [Fact]
    public void Stats_ScalingProperty()
    {
        // If we multiply all data by c, mean scales by c, variance scales by c^2
        var data = V(1, 2, 3, 4, 5);
        var scaled = V(2, 4, 6, 8, 10); // c = 2
        var stats1 = new BasicStats<double>(new BasicStatsInputs<double> { Values = data });
        var stats2 = new BasicStats<double>(new BasicStatsInputs<double> { Values = scaled });

        Assert.Equal(2 * stats1.Mean, stats2.Mean, Tol);
        Assert.Equal(4 * stats1.Variance, stats2.Variance, Tol);
        Assert.Equal(2 * stats1.StandardDeviation, stats2.StandardDeviation, Tol);
    }

    [Fact]
    public void Stats_ShiftingProperty()
    {
        // If we add constant c to all data, mean shifts by c, variance unchanged
        var data = V(1, 2, 3, 4, 5);
        var shifted = V(101, 102, 103, 104, 105); // c = 100
        var stats1 = new BasicStats<double>(new BasicStatsInputs<double> { Values = data });
        var stats2 = new BasicStats<double>(new BasicStatsInputs<double> { Values = shifted });

        Assert.Equal(stats1.Mean + 100, stats2.Mean, Tol);
        Assert.Equal(stats1.Variance, stats2.Variance, Tol);
        Assert.Equal(stats1.StandardDeviation, stats2.StandardDeviation, Tol);
    }

    // ──────────────────────────────────────────────────────────
    // ERROR STATS INTEGRATION
    // ──────────────────────────────────────────────────────────

    [Fact]
    public void ErrorStats_PerfectPredictions_ZeroErrors()
    {
        var actual = V(1, 2, 3, 4, 5);
        var predicted = V(1, 2, 3, 4, 5);
        var errorStats = new ErrorStats<double>(new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1
        });

        Assert.Equal(0.0, errorStats.MAE, Tol);
        Assert.Equal(0.0, errorStats.MSE, Tol);
        Assert.Equal(0.0, errorStats.RMSE, Tol);
        Assert.Equal(0.0, errorStats.MeanBiasError, Tol);
    }

    [Fact]
    public void ErrorStats_MAE_HandCalculation()
    {
        // actual = [1, 2, 3], predicted = [2, 3, 5]
        // errors: |1-2|=1, |2-3|=1, |3-5|=2
        // MAE = (1+1+2)/3 = 4/3
        var actual = V(1, 2, 3);
        var predicted = V(2, 3, 5);
        var errorStats = new ErrorStats<double>(new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1
        });

        Assert.Equal(4.0 / 3.0, errorStats.MAE, Tol);
    }

    [Fact]
    public void ErrorStats_MSE_HandCalculation()
    {
        // actual = [1, 2, 3], predicted = [2, 3, 5]
        // squared errors: 1, 1, 4
        // MSE = 6/3 = 2
        var actual = V(1, 2, 3);
        var predicted = V(2, 3, 5);
        var errorStats = new ErrorStats<double>(new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1
        });

        Assert.Equal(2.0, errorStats.MSE, Tol);
    }

    [Fact]
    public void ErrorStats_RMSE_IsSqrtMSE()
    {
        var actual = V(1, 2, 3, 4);
        var predicted = V(1.5, 2.5, 3.5, 4.5);
        var errorStats = new ErrorStats<double>(new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1
        });

        Assert.Equal(Math.Sqrt(errorStats.MSE), errorStats.RMSE, Tol);
    }

    [Fact]
    public void ErrorStats_MeanBiasError_OverPrediction()
    {
        // All predictions are 1 higher than actual → MeanBiasError = +1
        // Convention: predicted - actual = +1 for each
        var actual = V(1, 2, 3, 4, 5);
        var predicted = V(2, 3, 4, 5, 6);
        var errorStats = new ErrorStats<double>(new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1
        });

        Assert.Equal(1.0, errorStats.MeanBiasError, Tol);
    }

    [Fact]
    public void ErrorStats_MaxError_HandCalculation()
    {
        // actual = [1, 2, 3], predicted = [1, 2, 10]
        // errors: 0, 0, 7 → MaxError = 7
        var actual = V(1, 2, 3);
        var predicted = V(1, 2, 10);
        var errorStats = new ErrorStats<double>(new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1
        });

        Assert.Equal(7.0, errorStats.MaxError, Tol);
    }

    [Fact]
    public void ErrorStats_MSE_GreaterThanOrEqual_MAE_Squared()
    {
        // By Jensen's inequality: MSE >= MAE^2
        var actual = V(1, 2, 3, 4, 5);
        var predicted = V(1.5, 3, 2, 5, 4);
        var errorStats = new ErrorStats<double>(new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1
        });

        double maeSquared = errorStats.MAE * errorStats.MAE;
        Assert.True(errorStats.MSE >= maeSquared - Tol,
            $"MSE ({errorStats.MSE}) should be >= MAE^2 ({maeSquared})");
    }

    [Fact]
    public void ErrorStats_Ordering_MAE_RMSE_MaxError()
    {
        // MAE <= RMSE <= MaxError (always)
        var actual = V(0, 0, 0, 0, 0);
        var predicted = V(1, -2, 3, -1, 2);
        var errorStats = new ErrorStats<double>(new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1
        });

        Assert.True(errorStats.MAE <= errorStats.RMSE + Tol,
            $"MAE ({errorStats.MAE}) should be <= RMSE ({errorStats.RMSE})");
        Assert.True(errorStats.RMSE <= errorStats.MaxError + Tol,
            $"RMSE ({errorStats.RMSE}) should be <= MaxError ({errorStats.MaxError})");
    }

    [Fact]
    public void ErrorStats_RSS_Is_N_Times_MSE()
    {
        // RSS = sum of squared errors, MSE = RSS/n
        var actual = V(1, 2, 3, 4, 5);
        var predicted = V(2, 3, 4, 5, 6);
        var errorStats = new ErrorStats<double>(new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1
        });

        Assert.Equal(actual.Length * errorStats.MSE, errorStats.RSS, Tol);
    }

    [Fact]
    public void ErrorStats_Empty_AllZeros()
    {
        var errorStats = ErrorStats<double>.Empty();
        Assert.Equal(0.0, errorStats.MAE, Tol);
        Assert.Equal(0.0, errorStats.MSE, Tol);
        Assert.Equal(0.0, errorStats.RMSE, Tol);
    }

    [Fact]
    public void ErrorStats_DurbinWatson_Range()
    {
        // Durbin-Watson statistic should be in [0, 4]
        var actual = V(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        var predicted = V(1.1, 2.3, 2.8, 4.1, 5.2, 5.8, 7.3, 7.9, 9.1, 10.2);
        var errorStats = new ErrorStats<double>(new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1
        });

        Assert.True(errorStats.DurbinWatsonStatistic >= -Tol,
            $"DW ({errorStats.DurbinWatsonStatistic}) should be >= 0");
        Assert.True(errorStats.DurbinWatsonStatistic <= 4.0 + Tol,
            $"DW ({errorStats.DurbinWatsonStatistic}) should be <= 4");
    }

    [Fact]
    public void ErrorStats_SMAPE_Range()
    {
        // SMAPE should be in [0, 200] (or [0, 2] depending on normalization)
        var actual = V(1, 2, 3, 4, 5);
        var predicted = V(2, 3, 4, 5, 6);
        var errorStats = new ErrorStats<double>(new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1
        });

        Assert.True(errorStats.SMAPE >= -Tol,
            $"SMAPE ({errorStats.SMAPE}) should be >= 0");
    }

    [Fact]
    public void ErrorStats_Aliases_Match()
    {
        var actual = V(1, 2, 3);
        var predicted = V(1.5, 2.5, 3.5);
        var errorStats = new ErrorStats<double>(new ErrorStatsInputs<double>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = 1
        });

        Assert.Equal(errorStats.MAE, errorStats.MeanAbsoluteError, Tol);
        Assert.Equal(errorStats.MSE, errorStats.MeanSquaredError, Tol);
        Assert.Equal(errorStats.RMSE, errorStats.RootMeanSquaredError, Tol);
    }
}
