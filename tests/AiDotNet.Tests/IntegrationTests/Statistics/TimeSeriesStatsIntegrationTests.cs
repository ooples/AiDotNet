using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Statistics;

/// <summary>
/// Integration tests for time series statistics (Autocorrelation, PACF, DTW, Durbin-Watson).
/// Ground truth values verified against standard implementations.
/// </summary>
public class TimeSeriesStatsIntegrationTests
{
    private const double Tolerance = 1e-4;

    #region Durbin-Watson Statistic Tests

    [Fact]
    public void DurbinWatson_NoAutocorrelation_ReturnsTwo()
    {
        // Random-ish residuals with minimal autocorrelation should give DW close to 2
        // Residuals: [0.2, 0.3, -0.1, 0.4, 0.1, -0.2, 0.2, -0.1] - no clear pattern
        var actual = Vector<double>.FromArray([1.2, 2.3, 2.9, 4.4, 5.1, 5.8, 7.2, 7.9]);
        var predicted = Vector<double>.FromArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        var result = StatisticsHelper<double>.CalculateDurbinWatsonStatistic(actual, predicted);

        // DW around 2 indicates no autocorrelation
        Assert.True(result >= 1.5 && result <= 2.5, $"Expected DW near 2 for no autocorrelation, got {result}");
    }

    [Fact]
    public void DurbinWatson_ZeroResiduals_ThrowsArgumentException()
    {
        // Perfect prediction (all residuals = 0) should throw because DW is undefined
        var actual = Vector<double>.FromArray([1.0, 2.0, 3.0, 4.0, 5.0]);
        var predicted = Vector<double>.FromArray([1.0, 2.0, 3.0, 4.0, 5.0]);

        var ex = Assert.Throws<ArgumentException>(() =>
            StatisticsHelper<double>.CalculateDurbinWatsonStatistic(actual, predicted));

        Assert.Contains("zero", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void DurbinWatson_ConstantResiduals_ReturnsZero()
    {
        // If all residuals are the same, sum of squared differences = 0
        var residuals = new List<double> { 1.0, 1.0, 1.0, 1.0, 1.0 };

        var result = StatisticsHelper<double>.CalculateDurbinWatsonStatistic(residuals);

        // numerator = 0 (no differences), so DW = 0
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void DurbinWatson_PositiveAutocorrelation_LessThanTwo()
    {
        // Residuals that increase steadily (positive autocorrelation)
        var residuals = new List<double> { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };

        var result = StatisticsHelper<double>.CalculateDurbinWatsonStatistic(residuals);

        // Positive autocorrelation gives DW < 2
        Assert.True(result > 0 && result < 2, $"Expected DW < 2 for positive autocorrelation, got {result}");
    }

    [Fact]
    public void DurbinWatson_NegativeAutocorrelation_GreaterThanTwo()
    {
        // Residuals that alternate (negative autocorrelation)
        var residuals = new List<double> { 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0 };

        var result = StatisticsHelper<double>.CalculateDurbinWatsonStatistic(residuals);

        // Negative autocorrelation gives DW > 2
        Assert.True(result > 2 && result <= 4, $"Expected 2 < DW <= 4 for negative autocorrelation, got {result}");
    }

    [Fact]
    public void DurbinWatson_AlternatingPerfectly_ReturnsFour()
    {
        // Perfect alternation: e[i] = -e[i-1]
        // DW = sum((e[i] - e[i-1])^2) / sum(e[i]^2)
        // With alternating 1,-1: differences are all 2, so sum = 4*(n-1)
        // Sum of squares = n
        // DW = 4*(n-1)/n ≈ 4 as n increases
        var residuals = new List<double> { 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0 };

        var result = StatisticsHelper<double>.CalculateDurbinWatsonStatistic(residuals);

        // For n=10: DW = 4*9/10 = 3.6
        Assert.Equal(3.6, result, 0.1);
    }

    [Fact]
    public void DurbinWatson_VectorOverload_MatchesListOverload()
    {
        var actual = Vector<double>.FromArray([10.0, 20.0, 30.0, 40.0, 50.0]);
        var predicted = Vector<double>.FromArray([12.0, 18.0, 32.0, 38.0, 52.0]);

        var residuals = new List<double> { -2.0, 2.0, -2.0, 2.0, -2.0 };

        var resultVector = StatisticsHelper<double>.CalculateDurbinWatsonStatistic(actual, predicted);
        var resultList = StatisticsHelper<double>.CalculateDurbinWatsonStatistic(residuals);

        Assert.Equal(resultList, resultVector, Tolerance);
    }

    [Fact]
    public void DurbinWatson_IsInValidRange()
    {
        var residuals = new List<double> { 1.5, -0.5, 2.3, -1.2, 0.8, -0.3, 1.1 };

        var result = StatisticsHelper<double>.CalculateDurbinWatsonStatistic(residuals);

        Assert.True(result >= 0 && result <= 4, $"DW should be in [0,4], got {result}");
    }

    #endregion

    #region Dynamic Time Warping Tests

    [Fact]
    public void DTW_IdenticalSequences_ReturnsZero()
    {
        var series1 = Vector<double>.FromArray([1.0, 2.0, 3.0, 4.0, 5.0]);
        var series2 = Vector<double>.FromArray([1.0, 2.0, 3.0, 4.0, 5.0]);

        var result = StatisticsHelper<double>.CalculateDynamicTimeWarping(series1, series2);

        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void DTW_ShiftedSequences_ReturnsNonZero()
    {
        var series1 = Vector<double>.FromArray([1.0, 2.0, 3.0, 4.0, 5.0]);
        var series2 = Vector<double>.FromArray([2.0, 3.0, 4.0, 5.0, 6.0]);

        var result = StatisticsHelper<double>.CalculateDynamicTimeWarping(series1, series2);

        Assert.True(result > 0, "DTW of shifted sequences should be positive");
    }

    [Fact]
    public void DTW_IsSymmetric()
    {
        var series1 = Vector<double>.FromArray([1.0, 3.0, 5.0, 7.0, 9.0]);
        var series2 = Vector<double>.FromArray([2.0, 4.0, 6.0, 8.0, 10.0]);

        var dtw12 = StatisticsHelper<double>.CalculateDynamicTimeWarping(series1, series2);
        var dtw21 = StatisticsHelper<double>.CalculateDynamicTimeWarping(series2, series1);

        Assert.Equal(dtw12, dtw21, Tolerance);
    }

    [Fact]
    public void DTW_ConstantSequences_ReturnsDifference()
    {
        var series1 = Vector<double>.FromArray([5.0, 5.0, 5.0, 5.0, 5.0]);
        var series2 = Vector<double>.FromArray([3.0, 3.0, 3.0, 3.0, 3.0]);

        var result = StatisticsHelper<double>.CalculateDynamicTimeWarping(series1, series2);

        // Optimal alignment matches each point, total cost = n * |5-3| = 5 * 2 = 10
        Assert.Equal(10.0, result, Tolerance);
    }

    [Fact]
    public void DTW_DifferentLengths_StillComputes()
    {
        var series1 = Vector<double>.FromArray([1.0, 2.0, 3.0]);
        var series2 = Vector<double>.FromArray([1.0, 2.0, 3.0, 4.0, 5.0]);

        var result = StatisticsHelper<double>.CalculateDynamicTimeWarping(series1, series2);

        Assert.True(result >= 0, "DTW should be non-negative");
    }

    [Fact]
    public void DTW_SingleElements_ReturnsAbsDifference()
    {
        var series1 = Vector<double>.FromArray([5.0]);
        var series2 = Vector<double>.FromArray([3.0]);

        var result = StatisticsHelper<double>.CalculateDynamicTimeWarping(series1, series2);

        Assert.Equal(2.0, result, Tolerance);
    }

    [Fact]
    public void DTW_StretchedSequence_FindsOptimalAlignment()
    {
        // series1 is stretched version of series2
        var series1 = Vector<double>.FromArray([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        var series2 = Vector<double>.FromArray([1.0, 2.0, 3.0]);

        var result = StatisticsHelper<double>.CalculateDynamicTimeWarping(series1, series2);

        // DTW should find alignment with zero or near-zero cost
        Assert.True(result < 1.0, $"DTW for stretched sequence should be small, got {result}");
    }

    #endregion

    #region Autocorrelation Function Tests

    [Fact]
    public void ACF_Lag0_ReturnsOne()
    {
        var series = Vector<double>.FromArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        var acf = StatisticsHelper<double>.CalculateAutoCorrelationFunction(series, 5);

        // Autocorrelation at lag 0 is always 1
        Assert.Equal(1.0, acf[0], Tolerance);
    }

    [Fact]
    public void ACF_PositiveTrend_HighPositiveCorrelation()
    {
        // Increasing sequence has positive autocorrelation at low lags
        var series = Vector<double>.FromArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        var acf = StatisticsHelper<double>.CalculateAutoCorrelationFunction(series, 3);

        // Lag 1 should have high positive autocorrelation for trending data
        Assert.True(acf[1] > 0.5, $"ACF at lag 1 should be high for trending data, got {acf[1]}");
    }

    [Fact]
    public void ACF_AlternatingSequence_NegativeLag1()
    {
        // Alternating sequence has negative autocorrelation at lag 1
        var series = Vector<double>.FromArray([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);

        var acf = StatisticsHelper<double>.CalculateAutoCorrelationFunction(series, 3);

        // Lag 1 should have negative autocorrelation for alternating data
        Assert.True(acf[1] < 0, $"ACF at lag 1 should be negative for alternating data, got {acf[1]}");
    }

    [Fact]
    public void ACF_ReturnsCorrectLength()
    {
        var series = Vector<double>.FromArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        var acf = StatisticsHelper<double>.CalculateAutoCorrelationFunction(series, 5);

        // Should return maxLag + 1 values (lags 0 to maxLag)
        Assert.Equal(6, acf.Length);
    }

    [Fact]
    public void ACF_AllValuesInValidRange()
    {
        var series = Vector<double>.FromArray([1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0, 8.0, 10.0]);

        var acf = StatisticsHelper<double>.CalculateAutoCorrelationFunction(series, 5);

        // ACF is mathematically bounded by [-1, 1] (Cauchy-Schwarz inequality)
        // Allow small epsilon for floating-point tolerance
        const double epsilon = 1e-10;
        for (int i = 0; i < acf.Length; i++)
        {
            Assert.True(acf[i] >= -1.0 - epsilon && acf[i] <= 1.0 + epsilon,
                $"ACF[{i}] = {acf[i]} should be in [-1, 1]");
        }
    }

    [Fact]
    public void ACF_ConstantSeries_ThrowsArgumentException()
    {
        // Constant series has zero variance, so ACF is undefined
        var series = Vector<double>.FromArray([5.0, 5.0, 5.0, 5.0, 5.0]);

        var ex = Assert.Throws<ArgumentException>(() =>
            StatisticsHelper<double>.CalculateAutoCorrelationFunction(series, 2));

        Assert.Contains("zero variance", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    #endregion

    #region Partial Autocorrelation Function Tests

    [Fact]
    public void PACF_Lag0_ReturnsOne()
    {
        var series = Vector<double>.FromArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        var pacf = StatisticsHelper<double>.CalculatePartialAutoCorrelationFunction(series, 5);

        // PACF at lag 0 is always 1
        Assert.Equal(1.0, pacf[0], Tolerance);
    }

    [Fact]
    public void PACF_ReturnsCorrectLength()
    {
        var series = Vector<double>.FromArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        var pacf = StatisticsHelper<double>.CalculatePartialAutoCorrelationFunction(series, 4);

        // Should return maxLag + 1 values (lags 0 to maxLag)
        Assert.Equal(5, pacf.Length);
    }

    [Fact]
    public void PACF_LongerSeries_ComputesWithoutError()
    {
        // For PACF to work correctly, we need a longer series
        var series = Vector<double>.FromArray([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0
        ]);

        var pacf = StatisticsHelper<double>.CalculatePartialAutoCorrelationFunction(series, 5);

        // PACF at lag 0 should always be 1.0
        Assert.False(double.IsNaN(pacf[0]), "PACF[0] should not be NaN");
        Assert.Equal(1.0, pacf[0], Tolerance);
        Assert.Equal(6, pacf.Length);
    }

    #endregion

    #region Float Type Tests

    [Fact]
    public void DurbinWatson_FloatType_ReturnsCorrectValue()
    {
        var residuals = new List<float> { 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f };

        var result = StatisticsHelper<float>.CalculateDurbinWatsonStatistic(residuals);

        Assert.True(result > 2 && result <= 4, $"Expected DW > 2 for alternating residuals, got {result}");
    }

    [Fact]
    public void DTW_FloatType_ReturnsCorrectValue()
    {
        var series1 = Vector<float>.FromArray([1.0f, 2.0f, 3.0f, 4.0f, 5.0f]);
        var series2 = Vector<float>.FromArray([1.0f, 2.0f, 3.0f, 4.0f, 5.0f]);

        var result = StatisticsHelper<float>.CalculateDynamicTimeWarping(series1, series2);

        Assert.Equal(0.0f, result, 1e-4f);
    }

    [Fact]
    public void ACF_FloatType_ReturnsValidValues()
    {
        var series = Vector<float>.FromArray([1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f]);

        var acf = StatisticsHelper<float>.CalculateAutoCorrelationFunction(series, 3);

        Assert.Equal(1.0f, acf[0], 1e-4f);
        Assert.Equal(4, acf.Length);
    }

    #endregion
}
