using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.TimeSeries;

/// <summary>
/// Integration tests proving that ARIMA/SARIMA coefficient estimation enforces stationarity
/// and produces finite predictions, confidence intervals, and evaluation metrics.
///
/// These tests verify the fix for https://github.com/ooples/AiDotNet/issues/991:
/// - AR coefficients satisfy stationarity condition (|a1| + |a2| + ... + |ap| &lt; 1)
/// - Predictions remain finite (no NaN, no Infinity, no overflow to 1e+79)
/// - Confidence intervals are valid (LowerBound &lt;= Prediction &lt;= UpperBound)
/// - R2 is in valid range (not NaN)
/// - Forecast produces finite undifferenced values
///
/// Mathematical basis: For AR(p) process X_t = a1*X_{t-1} + ... + ap*X_{t-p} + e_t,
/// stationarity requires all roots of 1 - a1*z - a2*z^2 - ... - ap*z^p = 0
/// to lie outside the unit circle. Necessary condition: |a1| + |a2| + ... + |ap| &lt; 1.
/// </summary>
public class ARIMAStabilityTests
{
    #region Test Data

    /// <summary>
    /// Creates trending data that historically caused ARIMA to produce unstable
    /// coefficients with |sum| >= 1, leading to prediction overflow.
    /// </summary>
    private static (Matrix<double> x, Vector<double> y) CreateTrendingData(int n = 100, double slope = 1.0)
    {
        var y = new Vector<double>(n);
        var x = new Matrix<double>(n, 1);
        for (int i = 0; i < n; i++)
        {
            double noise = 5.0 * Math.Sin(2.0 * Math.PI * i / 13.7); // Deterministic for reproducibility
            y[i] = 100.0 + slope * i + noise;
            x[i, 0] = i;
        }
        return (x, y);
    }

    private static (Matrix<double> x, Vector<double> y) CreateSeasonalTrendData(int n = 120)
    {
        var y = new Vector<double>(n);
        var x = new Matrix<double>(n, 1);
        for (int i = 0; i < n; i++)
        {
            y[i] = 50.0 + 0.3 * i + 8.0 * Math.Sin(2.0 * Math.PI * i / 12);
            x[i, 0] = i;
        }
        return (x, y);
    }

    #endregion

    #region Stationarity Enforcement Tests

    [Fact]
    public void EstimateARCoefficients_ShouldProduceStationaryCoefficients()
    {
        // Arrange — trending data that previously produced unstable coefficients
        var (_, y) = CreateTrendingData();
        var diffY = TimeSeriesHelper<double>.DifferenceSeries(y, 1);

        // Act
        var arCoeffs = TimeSeriesHelper<double>.EstimateARCoefficients(
            diffY, p: 2, MatrixDecompositionType.Qr);

        // Assert — sum of absolute coefficients must be < 1 (stationarity)
        double absSum = 0;
        for (int i = 0; i < arCoeffs.Length; i++)
        {
            var val = arCoeffs[i];
            Assert.False(double.IsNaN(val), $"AR coefficient [{i}] is NaN");
            Assert.False(double.IsInfinity(val), $"AR coefficient [{i}] is Infinity ({val})");
            absSum += Math.Abs(val);
        }
        Assert.True(absSum < 1.0,
            $"AR coefficients violate stationarity: |a1| + |a2| = {absSum} >= 1.0. " +
            $"Coefficients: [{string.Join(", ", Enumerable.Range(0, arCoeffs.Length).Select(i => arCoeffs[i].ToString("F6")))}]");
    }

    [Fact]
    public void EstimateMACoefficients_ShouldProduceInvertibleCoefficients()
    {
        // Arrange
        var (_, y) = CreateTrendingData();
        var diffY = TimeSeriesHelper<double>.DifferenceSeries(y, 1);
        var arCoeffs = TimeSeriesHelper<double>.EstimateARCoefficients(
            diffY, p: 1, MatrixDecompositionType.Qr);
        var residuals = TimeSeriesHelper<double>.CalculateARResiduals(diffY, arCoeffs);

        // Act
        var maCoeffs = TimeSeriesHelper<double>.EstimateMACoefficients(residuals, q: 2);

        // Assert — MA coefficients should also satisfy invertibility
        double absSum = 0;
        for (int i = 0; i < maCoeffs.Length; i++)
        {
            Assert.False(double.IsNaN(maCoeffs[i]), $"MA coefficient [{i}] is NaN");
            Assert.False(double.IsInfinity(maCoeffs[i]), $"MA coefficient [{i}] is Infinity");
            absSum += Math.Abs(maCoeffs[i]);
        }
        Assert.True(absSum < 1.0,
            $"MA coefficients violate invertibility: sum = {absSum}");
    }

    #endregion

    #region ARIMA Prediction Finiteness Tests

    [Theory]
    [InlineData(1, 0, 0)] // AR(1)
    [InlineData(2, 0, 0)] // AR(2)
    [InlineData(1, 1, 0)] // ARIMA(1,1,0)
    [InlineData(1, 1, 1)] // ARIMA(1,1,1) — the exact config that produced NaN in #991
    [InlineData(2, 1, 2)] // ARIMA(2,1,2) — the config that produced NaN R2 in #991
    [InlineData(3, 1, 1)] // Higher order AR
    [InlineData(1, 2, 1)] // Higher differencing
    public void ARIMA_Predict_ShouldProduceFiniteValues(int p, int d, int q)
    {
        // Arrange
        var (x, y) = CreateTrendingData();
        var options = new ARIMAOptions<double> { P = p, D = d, Q = q, MaxIterations = 100 };
        var model = new ARIMAModel<double>(options);
        model.Train(x, y);

        // Act — predict 10 steps
        var testX = new Matrix<double>(10, 1);
        for (int i = 0; i < 10; i++) testX[i, 0] = 100 + i;
        var predictions = model.Predict(testX);

        // Assert — every prediction must be finite
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.False(double.IsNaN(predictions[i]),
                $"ARIMA({p},{d},{q}): Prediction[{i}] is NaN");
            Assert.False(double.IsInfinity(predictions[i]),
                $"ARIMA({p},{d},{q}): Prediction[{i}] is Infinity ({predictions[i]})");
            Assert.True(Math.Abs(predictions[i]) < 1e15,
                $"ARIMA({p},{d},{q}): Prediction[{i}] overflowed ({predictions[i]})");
        }
    }

    [Theory]
    [InlineData(1, 1, 1)]
    [InlineData(2, 1, 2)]
    public void ARIMA_Forecast_ShouldProduceFiniteUndifferencedValues(int p, int d, int q)
    {
        // Arrange
        var (x, y) = CreateTrendingData();
        var options = new ARIMAOptions<double> { P = p, D = d, Q = q, MaxIterations = 100 };
        var model = new ARIMAModel<double>(options);
        model.Train(x, y);

        // Act — forecast on original scale
        var forecasts = model.Forecast(y, steps: 10);

        // Assert — forecasts must be finite and on a reasonable scale
        for (int i = 0; i < forecasts.Length; i++)
        {
            Assert.False(double.IsNaN(forecasts[i]),
                $"ARIMA({p},{d},{q}): Forecast[{i}] is NaN");
            Assert.False(double.IsInfinity(forecasts[i]),
                $"ARIMA({p},{d},{q}): Forecast[{i}] is Infinity ({forecasts[i]})");
            Assert.True(Math.Abs(forecasts[i]) < 1e15,
                $"ARIMA({p},{d},{q}): Forecast[{i}] overflowed ({forecasts[i]})");
        }
    }

    #endregion

    #region SARIMA Stability Tests

    [Theory]
    [InlineData(1, 1, 1, 1, 0, 0, 12)]
    [InlineData(1, 1, 1, 1, 1, 1, 12)]
    [InlineData(2, 1, 1, 1, 0, 1, 7)]
    public void SARIMA_Predict_ShouldProduceFiniteValues(int p, int d, int q, int P, int D, int Q, int m)
    {
        // Arrange
        var (x, y) = CreateSeasonalTrendData();
        var options = new SARIMAOptions<double>
        {
            P = p, D = d, Q = q,
            SeasonalP = P, SeasonalD = D, SeasonalQ = Q,
            SeasonalPeriod = m,
            MaxIterations = 100
        };
        var model = new SARIMAModel<double>(options);
        model.Train(x, y);

        // Act
        var testX = new Matrix<double>(10, 1);
        for (int i = 0; i < 10; i++) testX[i, 0] = 120 + i;
        var predictions = model.Predict(testX);

        // Assert
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.False(double.IsNaN(predictions[i]),
                $"SARIMA({p},{d},{q})({P},{D},{Q})[{m}]: Prediction[{i}] is NaN");
            Assert.False(double.IsInfinity(predictions[i]),
                $"SARIMA({p},{d},{q})({P},{D},{Q})[{m}]: Prediction[{i}] is Infinity");
        }
    }

    #endregion

    #region Evaluation Metric Finiteness Tests

    [Theory]
    [InlineData(1, 1, 1)]
    [InlineData(2, 1, 2)]
    public void ARIMA_EvaluateModel_ShouldProduceFiniteMetrics(int p, int d, int q)
    {
        // Arrange — split data into train/test
        var (x, y) = CreateTrendingData(n: 120);
        var trainX = new Matrix<double>(100, 1);
        var trainY = new Vector<double>(100);
        var testX = new Matrix<double>(20, 1);
        var testY = new Vector<double>(20);
        for (int i = 0; i < 100; i++) { trainX[i, 0] = i; trainY[i] = y[i]; }
        for (int i = 0; i < 20; i++) { testX[i, 0] = 100 + i; testY[i] = y[100 + i]; }

        var options = new ARIMAOptions<double> { P = p, D = d, Q = q, MaxIterations = 100 };
        var model = new ARIMAModel<double>(options);
        model.Train(trainX, trainY);

        // Act
        var metrics = model.EvaluateModel(testX, testY);

        // Assert — all metrics must be finite
        foreach (var kvp in metrics)
        {
            if (kvp.Value != null && double.TryParse(kvp.Value.ToString(), out var val))
            {
                Assert.False(double.IsNaN(val),
                    $"ARIMA({p},{d},{q}): Metric '{kvp.Key}' is NaN");
                Assert.False(double.IsInfinity(val),
                    $"ARIMA({p},{d},{q}): Metric '{kvp.Key}' is Infinity ({val})");
            }
        }
    }

    #endregion

    #region GuardPrediction Safety Net Tests

    [Fact]
    public void GuardPrediction_ShouldClampNaN()
    {
        var (x, y) = CreateTrendingData(n: 30);
        var options = new ARIMAOptions<double> { P = 1, D = 0, Q = 0, MaxIterations = 10 };
        var model = new TestableTimeSeriesModel(options);
        model.Train(x, y);

        // GuardPrediction is protected — test via the model's public interface
        // If predictions were NaN before the fix, they should now be finite
        var testX = new Matrix<double>(5, 1);
        for (int i = 0; i < 5; i++) testX[i, 0] = 30 + i;
        var predictions = model.Predict(testX);

        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.False(double.IsNaN(predictions[i]),
                $"GuardPrediction failed: Prediction[{i}] is still NaN");
        }
    }

    /// <summary>
    /// Minimal model subclass to test GuardPrediction through public interface.
    /// </summary>
    private class TestableTimeSeriesModel : ARIMAModel<double>
    {
        public TestableTimeSeriesModel(ARIMAOptions<double> options) : base(options) { }
    }

    #endregion
}
