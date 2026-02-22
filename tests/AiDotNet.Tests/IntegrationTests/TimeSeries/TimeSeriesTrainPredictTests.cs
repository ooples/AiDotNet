using System;
using AiDotNet.Models.Options;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.TimeSeries;

/// <summary>
/// Integration tests verifying that time series models produce valid (non-NaN, non-Infinity)
/// predictions after training, and that Predict(Matrix), Forecast(history, horizon),
/// PredictSingle, and EvaluateModel all work correctly.
///
/// These tests catch bugs like:
/// - Predict(Matrix) using input feature column instead of training data (ARModel #878)
/// - Predict(Matrix) initializing state from zeros instead of training state (ARIMA, SARIMA)
/// - NaN/Infinity in predictions after training
/// - EvaluateModel producing NaN metrics
/// - SARIMA crash when Q=0 (ArgumentOutOfRangeException at line 486)
/// </summary>
public class TimeSeriesTrainPredictTests
{
    #region Test Data Helpers

    /// <summary>
    /// Creates stationary mean-reverting data suitable for AR and ARMA models.
    /// Uses a sine wave centered around zero with small amplitude to avoid gradient explosion
    /// in AR model's un-normalized gradient descent training.
    /// </summary>
    private static (Matrix<double> x, Vector<double> y) CreateStationaryData(int n, double mean = 0.0, double amplitude = 1.0, double period = 20.0)
    {
        var y = new Vector<double>(n);
        var x = new Matrix<double>(n, 1);
        for (int i = 0; i < n; i++)
        {
            y[i] = mean + amplitude * Math.Sin(2.0 * Math.PI * i / period);
            x[i, 0] = i;
        }
        return (x, y);
    }

    /// <summary>
    /// Creates data with a linear trend plus noise, suitable for ARIMA (d=1).
    /// After differencing, the data has variance due to the noise component.
    /// </summary>
    private static (Matrix<double> x, Vector<double> y) CreateTrendPlusNoiseData(int n, double slope = 0.5, double intercept = 10.0, double noiseAmplitude = 3.0)
    {
        var y = new Vector<double>(n);
        var x = new Matrix<double>(n, 1);
        for (int i = 0; i < n; i++)
        {
            // Deterministic noise via sine with non-harmonic frequency
            double noise = noiseAmplitude * Math.Sin(2.0 * Math.PI * i / 7.3);
            y[i] = slope * i + intercept + noise;
            x[i, 0] = i;
        }
        return (x, y);
    }

    /// <summary>
    /// Creates seasonal data suitable for SARIMA models.
    /// Clear repeating seasonal pattern centered around a mean.
    /// </summary>
    private static (Matrix<double> x, Vector<double> y) CreateSeasonalData(int n, int period = 12, double amplitude = 5.0, double offset = 50.0)
    {
        var y = new Vector<double>(n);
        var x = new Matrix<double>(n, 1);
        for (int i = 0; i < n; i++)
        {
            y[i] = amplitude * Math.Sin(2.0 * Math.PI * i / period) + offset;
            x[i, 0] = i;
        }
        return (x, y);
    }

    /// <summary>
    /// Creates trend + seasonal data for SARIMA with differencing.
    /// </summary>
    private static (Matrix<double> x, Vector<double> y) CreateTrendPlusSeasonalData(int n, double slope = 0.3, double intercept = 20.0, int period = 12, double amplitude = 3.0)
    {
        var y = new Vector<double>(n);
        var x = new Matrix<double>(n, 1);
        for (int i = 0; i < n; i++)
        {
            y[i] = slope * i + intercept + amplitude * Math.Sin(2.0 * Math.PI * i / period);
            x[i, 0] = i;
        }
        return (x, y);
    }

    /// <summary>
    /// Creates linear trend data for ExponentialSmoothing.
    /// </summary>
    private static (Matrix<double> x, Vector<double> y) CreateLinearTrendData(int n, double slope = 2.0, double intercept = 10.0)
    {
        var y = new Vector<double>(n);
        var x = new Matrix<double>(n, 1);
        for (int i = 0; i < n; i++)
        {
            y[i] = slope * i + intercept;
            x[i, 0] = i;
        }
        return (x, y);
    }

    private static void AssertFiniteVector(Vector<double> v, string label)
    {
        Assert.True(v.Length > 0, $"{label} is empty — expected at least one element");
        for (int i = 0; i < v.Length; i++)
        {
            Assert.False(double.IsNaN(v[i]), $"{label}[{i}] is NaN");
            Assert.False(double.IsInfinity(v[i]), $"{label}[{i}] is Infinity");
        }
    }

    private static void AssertFiniteMetrics(Dictionary<string, double> metrics)
    {
        Assert.True(metrics.Count > 0, "Metrics dictionary is empty — expected at least one metric");
        foreach (var kvp in metrics)
        {
            Assert.False(double.IsNaN(kvp.Value), $"Metric '{kvp.Key}' is NaN");
            Assert.False(double.IsInfinity(kvp.Value), $"Metric '{kvp.Key}' is Infinity");
        }
    }

    #endregion

    #region ARModel Tests

    [Fact]
    public void ARModel_TrainAndPredict_ProducesFinitePredictions()
    {
        // AR requires stationary data — use mean-reverting sinusoidal
        var (x, y) = CreateStationaryData(100);
        var model = new ARModel<double>(new ARModelOptions<double> { AROrder = 3, MaxIterations = 100 });

        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.Equal(x.Rows, predictions.Length);
        AssertFiniteVector(predictions, "ARModel.Predict");
    }

    [Fact]
    public void ARModel_TrainAndPredict_NotAllZeros()
    {
        // Regression test: before fix, Predict(Matrix) used input column (time indices)
        // instead of training data, producing NaN or all-zero predictions
        var (x, y) = CreateStationaryData(100);
        var model = new ARModel<double>(new ARModelOptions<double> { AROrder = 3, MaxIterations = 100 });

        model.Train(x, y);
        var predictions = model.Predict(x);

        // At least some predictions should be non-zero (after the first few AR-order values)
        bool hasNonZero = false;
        for (int i = 5; i < predictions.Length; i++)
        {
            if (Math.Abs(predictions[i]) > 1e-10)
            {
                hasNonZero = true;
                break;
            }
        }
        Assert.True(hasNonZero, "ARModel predictions are all zero after training on non-zero data");
    }

    [Fact]
    public void ARModel_Forecast_ProducesFiniteResults()
    {
        var (x, y) = CreateStationaryData(100);
        var model = new ARModel<double>(new ARModelOptions<double> { AROrder = 3, MaxIterations = 100 });

        model.Train(x, y);
        var history = new Vector<double>(50);
        for (int i = 0; i < 50; i++)
            history[i] = y[50 + i];
        // Short horizon (3 steps) to avoid recursive forecast instability
        // from AR coefficient root magnitudes near unit circle
        var forecasts = model.Forecast(history, 3);

        Assert.Equal(3, forecasts.Length);
        AssertFiniteVector(forecasts, "ARModel.Forecast");
    }

    [Fact]
    public void ARModel_EvaluateModel_ProducesFiniteMetrics()
    {
        var (x, y) = CreateStationaryData(100);
        var model = new ARModel<double>(new ARModelOptions<double> { AROrder = 3, MaxIterations = 100 });

        model.Train(x, y);
        var metrics = model.EvaluateModel(x, y);

        Assert.True(metrics.Count > 0, "EvaluateModel returned no metrics");
        AssertFiniteMetrics(metrics);
    }

    [Fact]
    public void ARModel_PredictSingle_ProducesFiniteResult()
    {
        var (x, y) = CreateStationaryData(100);
        var model = new ARModel<double>(new ARModelOptions<double> { AROrder = 3, MaxIterations = 100 });

        model.Train(x, y);
        // Provide last 10 values as input for PredictSingle
        var input = new Vector<double>(10);
        for (int i = 0; i < 10; i++)
            input[i] = y[90 + i];
        var prediction = model.PredictSingle(input);

        Assert.False(double.IsNaN(prediction), "PredictSingle returned NaN");
        Assert.False(double.IsInfinity(prediction), "PredictSingle returned Infinity");
    }

    [Fact]
    public void ARModel_Clone_PreservesTrainedState()
    {
        var (x, y) = CreateStationaryData(100);
        var model = new ARModel<double>(new ARModelOptions<double> { AROrder = 3, MaxIterations = 100 });

        model.Train(x, y);
        var original = model.Predict(x);
        var clone = (ARModel<double>)model.Clone();
        var clonePredictions = clone.Predict(x);

        Assert.Equal(original.Length, clonePredictions.Length);
        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(original[i], clonePredictions[i], precision: 8);
        }
    }

    #endregion

    #region MAModel Tests

    [Fact]
    public void MAModel_TrainAndPredict_ProducesFinitePredictions()
    {
        var (x, y) = CreateStationaryData(100);
        var model = new MAModel<double>(new MAModelOptions<double> { MAOrder = 3 });

        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.Equal(x.Rows, predictions.Length);
        AssertFiniteVector(predictions, "MAModel.Predict");
    }

    [Fact]
    public void MAModel_Forecast_ProducesFiniteResults()
    {
        var (x, y) = CreateStationaryData(100);
        var model = new MAModel<double>(new MAModelOptions<double> { MAOrder = 3 });

        model.Train(x, y);
        var history = new Vector<double>(50);
        for (int i = 0; i < 50; i++)
            history[i] = y[50 + i];
        var forecasts = model.Forecast(history, 5);

        Assert.Equal(5, forecasts.Length);
        AssertFiniteVector(forecasts, "MAModel.Forecast");
    }

    #endregion

    #region ARMAModel Tests

    [Fact]
    public void ARMAModel_TrainAndPredict_ProducesFinitePredictions()
    {
        var (x, y) = CreateStationaryData(100);
        var model = new ARMAModel<double>(new ARMAOptions<double> { AROrder = 2, MAOrder = 1 });

        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.Equal(x.Rows, predictions.Length);
        AssertFiniteVector(predictions, "ARMAModel.Predict");
    }

    [Fact]
    public void ARMAModel_Forecast_ProducesFiniteResults()
    {
        var (x, y) = CreateStationaryData(100);
        var model = new ARMAModel<double>(new ARMAOptions<double> { AROrder = 2, MAOrder = 1 });

        model.Train(x, y);
        var history = new Vector<double>(50);
        for (int i = 0; i < 50; i++)
            history[i] = y[50 + i];
        var forecasts = model.Forecast(history, 5);

        Assert.Equal(5, forecasts.Length);
        AssertFiniteVector(forecasts, "ARMAModel.Forecast");
    }

    #endregion

    #region ARIMAModel Tests

    [Fact]
    public void ARIMAModel_TrainAndPredict_ProducesFinitePredictions()
    {
        var (x, y) = CreateTrendPlusNoiseData(100);
        var model = new ARIMAModel<double>(new ARIMAOptions<double> { P = 2, D = 1, Q = 0, MaxIterations = 50 });

        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.Equal(x.Rows, predictions.Length);
        AssertFiniteVector(predictions, "ARIMAModel.Predict");
    }

    [Fact]
    public void ARIMAModel_TrainAndPredict_UsesStoredState()
    {
        // Regression test: before fix, ARIMA Predict(Matrix) started from zeros.
        // ARIMA Predict(Matrix) works in the differenced domain where predictions
        // naturally converge to a constant (the mean of the differenced series).
        // This is mathematically correct — we verify the stored training state is
        // used by checking that the initial transient differs from converged value.
        var (x, y) = CreateTrendPlusNoiseData(100, slope: 1.0, intercept: 50.0, noiseAmplitude: 5.0);
        var model = new ARIMAModel<double>(new ARIMAOptions<double> { P = 2, D = 1, Q = 1, MaxIterations = 50 });

        model.Train(x, y);
        var predictions = model.Predict(x);

        AssertFiniteVector(predictions, "ARIMAModel.Predict");

        // With stored training state, the first prediction should incorporate AR values
        // from the end of training (non-zero). Without stored state, all predictions
        // would be constant from the start (just the constant term with zero AR/MA inputs).
        // Check that predictions[0] differs from the converged value at predictions[50].
        // A 0-initialized model would produce constant output from the very first prediction.
        double convergedValue = predictions[50];
        bool hasTransient = Math.Abs(predictions[0] - convergedValue) > 1e-10;
        Assert.True(hasTransient,
            $"ARIMA predictions show no transient (predictions[0]={predictions[0]:F6}, " +
            $"converged={convergedValue:F6}). This suggests stored training state is not being used.");
    }

    [Fact]
    public void ARIMAModel_Forecast_ProducesFiniteResults()
    {
        var (x, y) = CreateTrendPlusNoiseData(100);
        var model = new ARIMAModel<double>(new ARIMAOptions<double> { P = 2, D = 1, Q = 0, MaxIterations = 50 });

        model.Train(x, y);
        var history = new Vector<double>(50);
        for (int i = 0; i < 50; i++)
            history[i] = y[50 + i];
        var forecasts = model.Forecast(history, 10);

        Assert.Equal(10, forecasts.Length);
        AssertFiniteVector(forecasts, "ARIMAModel.Forecast");
    }

    [Fact]
    public void ARIMAModel_EvaluateModel_ProducesFiniteMetrics()
    {
        var (x, y) = CreateTrendPlusNoiseData(100);
        var model = new ARIMAModel<double>(new ARIMAOptions<double> { P = 2, D = 1, Q = 0, MaxIterations = 50 });

        model.Train(x, y);
        var metrics = model.EvaluateModel(x, y);

        Assert.True(metrics.Count > 0);
        AssertFiniteMetrics(metrics);
    }

    #endregion

    #region SARIMAModel Tests

    /// <summary>
    /// Tests SARIMA with Q=0 (pure AR, no MA components).
    /// Regression test for crash at line 486 where lastErrors had length 0.
    /// This is distinct from SARIMAModel_PureARNoMA_DoesNotCrash which tests
    /// both Q=0 AND seasonal Q=0 with AR order 2.
    /// </summary>
    [Fact]
    public void SARIMAModel_TrainAndPredict_PureAR_ProducesFinitePredictions()
    {
        // Test SARIMA with Q=0 (no MA components) — previously crashed at line 486
        // with ArgumentOutOfRangeException because lastErrors had length 0
        var (x, y) = CreateSeasonalData(120, period: 12);
        var model = new SARIMAModel<double>(new SARIMAOptions<double>
        {
            P = 1, D = 0, Q = 0,
            SeasonalP = 1, SeasonalD = 0, SeasonalQ = 0,
            SeasonalPeriod = 12,
            MaxIterations = 50
        });

        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.Equal(x.Rows, predictions.Length);
        AssertFiniteVector(predictions, "SARIMAModel.Predict(Q=0)");
    }

    [Fact]
    public void SARIMAModel_TrainAndPredict_WithMA_ProducesFinitePredictions()
    {
        // Test SARIMA with MA components (Q > 0)
        var (x, y) = CreateSeasonalData(120, period: 12);
        var model = new SARIMAModel<double>(new SARIMAOptions<double>
        {
            P = 1, D = 0, Q = 1,
            SeasonalP = 1, SeasonalD = 0, SeasonalQ = 1,
            SeasonalPeriod = 12,
            MaxIterations = 50
        });

        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.Equal(x.Rows, predictions.Length);
        AssertFiniteVector(predictions, "SARIMAModel.Predict(Q=1)");
    }

    [Fact]
    public void SARIMAModel_TrainAndPredict_UsesStoredState()
    {
        // SARIMA Predict(Matrix) works in the differenced domain where predictions
        // converge. Verify stored training state via transient behavior.
        var (x, y) = CreateTrendPlusSeasonalData(120, period: 12);
        var model = new SARIMAModel<double>(new SARIMAOptions<double>
        {
            P = 1, D = 1, Q = 1,
            SeasonalP = 1, SeasonalD = 0, SeasonalQ = 1,
            SeasonalPeriod = 12,
            MaxIterations = 50
        });

        model.Train(x, y);
        var predictions = model.Predict(x);

        AssertFiniteVector(predictions, "SARIMAModel.Predict");

        // With stored state, first prediction uses actual AR/MA values from training.
        // Without stored state (zeros), all predictions are constant from the start.
        double convergedValue = predictions[60];
        bool hasTransient = Math.Abs(predictions[0] - convergedValue) > 1e-10;
        Assert.True(hasTransient,
            $"SARIMA predictions show no transient (predictions[0]={predictions[0]:F6}, " +
            $"converged={convergedValue:F6}). This suggests stored training state is not being used.");
    }

    [Fact]
    public void SARIMAModel_Forecast_ProducesFiniteResults()
    {
        var (x, y) = CreateSeasonalData(120, period: 12);
        var model = new SARIMAModel<double>(new SARIMAOptions<double>
        {
            P = 1, D = 0, Q = 1,
            SeasonalP = 1, SeasonalD = 0, SeasonalQ = 1,
            SeasonalPeriod = 12,
            MaxIterations = 50
        });

        model.Train(x, y);
        var history = new Vector<double>(60);
        for (int i = 0; i < 60; i++)
            history[i] = y[60 + i];
        var forecasts = model.Forecast(history, 12);

        Assert.Equal(12, forecasts.Length);
        AssertFiniteVector(forecasts, "SARIMAModel.Forecast");
    }

    #endregion

    #region ExponentialSmoothingModel Tests

    [Fact]
    public void ExponentialSmoothing_TrainAndPredict_ProducesFinitePredictions()
    {
        var (x, y) = CreateLinearTrendData(100, slope: 2.0);
        var model = new ExponentialSmoothingModel<double>(new ExponentialSmoothingOptions<double>
        {
            InitialAlpha = 0.3,
            InitialBeta = 0.3,
            UseTrend = true,
            UseSeasonal = false
        });

        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.Equal(x.Rows, predictions.Length);
        AssertFiniteVector(predictions, "ExponentialSmoothing.Predict");
    }

    [Fact]
    public void ExponentialSmoothing_Forecast_MonotonicallyIncreasingForUpwardTrend()
    {
        var (x, y) = CreateLinearTrendData(100, slope: 2.0, intercept: 10.0);
        var model = new ExponentialSmoothingModel<double>(new ExponentialSmoothingOptions<double>
        {
            InitialAlpha = 0.3,
            InitialBeta = 0.3,
            UseTrend = true,
            UseSeasonal = false
        });

        model.Train(x, y);
        var history = new Vector<double>(50);
        for (int i = 0; i < 50; i++)
            history[i] = y[50 + i];
        var forecasts = model.Forecast(history, 5);

        AssertFiniteVector(forecasts, "ExponentialSmoothing.Forecast");
        for (int i = 1; i < forecasts.Length; i++)
        {
            Assert.True(forecasts[i] > forecasts[i - 1],
                $"Forecast[{i}] ({forecasts[i]:F4}) should be > Forecast[{i - 1}] ({forecasts[i - 1]:F4}) for upward trend");
        }
    }

    #endregion

    #region StateSpaceModel Tests

    [Fact]
    public void StateSpaceModel_TrainAndPredict_ProducesFinitePredictions()
    {
        var (x, y) = CreateStationaryData(60);
        var model = new StateSpaceModel<double>(new StateSpaceModelOptions<double>());

        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.Equal(x.Rows, predictions.Length);
        AssertFiniteVector(predictions, "StateSpaceModel.Predict");
    }

    #endregion

    #region Serialization Round-trip Tests

    [Fact]
    public void ARModel_SerializeAndDeserialize_PreservesPredictions()
    {
        // Use stationary data to avoid coefficient instability
        var (x, y) = CreateStationaryData(100);
        var model = new ARModel<double>(new ARModelOptions<double> { AROrder = 3, MaxIterations = 100 });
        model.Train(x, y);

        var originalPredictions = model.Predict(x);
        AssertFiniteVector(originalPredictions, "Original ARModel.Predict");

        // Serialize
        byte[] serialized = model.Serialize();
        Assert.True(serialized.Length > 0, "Serialized data is empty");

        // Deserialize into new model
        var restoredModel = new ARModel<double>(new ARModelOptions<double> { AROrder = 3 });
        restoredModel.Deserialize(serialized);
        var restoredPredictions = restoredModel.Predict(x);

        // Predictions should match (skip first 5 where AR coefficients have limited history)
        Assert.Equal(originalPredictions.Length, restoredPredictions.Length);
        AssertFiniteVector(restoredPredictions, "Restored ARModel.Predict");
        for (int i = 5; i < originalPredictions.Length; i++)
        {
            Assert.Equal(originalPredictions[i], restoredPredictions[i], precision: 8);
        }
    }

    #endregion

    #region Cross-Model Consistency Tests

    [Fact]
    public void ARIMA_And_ExponentialSmoothing_ForecastProducesFiniteResults()
    {
        // Verify that ARIMA and ExponentialSmoothing both produce finite forecasts
        // on trend data without asserting specific value thresholds
        var (x, y) = CreateLinearTrendData(100, slope: 2.0, intercept: 10.0);
        var history = new Vector<double>(50);
        for (int i = 0; i < 50; i++)
            history[i] = y[50 + i];

        // ARIMA model (handles trends via differencing)
        var arimaModel = new ARIMAModel<double>(new ARIMAOptions<double> { P = 2, D = 1, Q = 0, MaxIterations = 50 });
        arimaModel.Train(x, y);
        var arimaForecast = arimaModel.Forecast(history, 3);
        AssertFiniteVector(arimaForecast, "ARIMA.Forecast");

        // ExponentialSmoothing with trend
        var esModel = new ExponentialSmoothingModel<double>(new ExponentialSmoothingOptions<double>
        {
            InitialAlpha = 0.3, InitialBeta = 0.3, UseTrend = true, UseSeasonal = false
        });
        esModel.Train(x, y);
        var esForecast = esModel.Forecast(history, 3);
        AssertFiniteVector(esForecast, "ES.Forecast");

        // Both forecasts should be positive (since original data is all positive)
        Assert.True(arimaForecast[0] > 0, $"ARIMA forecast ({arimaForecast[0]:F2}) should be positive");
        Assert.True(esForecast[0] > 0, $"ES forecast ({esForecast[0]:F2}) should be positive");
    }

    #endregion

    #region Edge Case Tests

    [Fact]
    public void ARModel_DummyFeatureMatrix_StillProducesValidPredictions()
    {
        // This is the exact scenario that caused #878: training data has meaningful y values
        // but the feature matrix x is just zeros (dummy features).
        // Uses zero-centered stationary data to avoid gradient explosion.
        var n = 100;
        var y = new Vector<double>(n);
        var x = new Matrix<double>(n, 1); // All zeros - dummy features
        for (int i = 0; i < n; i++)
        {
            y[i] = Math.Sin(2.0 * Math.PI * i / 20.0); // Zero-centered, amplitude 1
            x[i, 0] = 0; // Dummy feature - all zeros
        }

        var model = new ARModel<double>(new ARModelOptions<double> { AROrder = 3, MaxIterations = 100 });
        model.Train(x, y);
        var predictions = model.Predict(x);

        AssertFiniteVector(predictions, "ARModel.Predict(dummy features)");

        // Predictions should use training data, not the dummy zeros
        bool hasReasonableValues = false;
        for (int i = 5; i < predictions.Length; i++)
        {
            if (Math.Abs(predictions[i]) > 0.01)
            {
                hasReasonableValues = true;
                break;
            }
        }
        Assert.True(hasReasonableValues,
            "ARModel.Predict produced near-zero values despite training on sinusoidal data. " +
            "This indicates Predict is using the dummy feature matrix instead of stored training data.");
    }

    [Fact]
    public void ARModel_Forecast_AfterPredictMatrix_StillWorks()
    {
        // Uses stationary data
        var (x, y) = CreateStationaryData(100);
        var model = new ARModel<double>(new ARModelOptions<double> { AROrder = 3, MaxIterations = 100 });
        model.Train(x, y);

        // Call Predict(Matrix) first
        var predictions = model.Predict(x);
        AssertFiniteVector(predictions, "ARModel.Predict");

        // Then call Forecast - should still work
        var history = new Vector<double>(50);
        for (int i = 0; i < 50; i++)
            history[i] = y[50 + i];
        var forecasts = model.Forecast(history, 5);
        AssertFiniteVector(forecasts, "ARModel.Forecast after Predict");
    }

    /// <summary>
    /// Tests SARIMA with both Q=0 AND seasonal Q=0 with AR order 2.
    /// Regression test for crash where lastErrors vector had length 0.
    /// This is distinct from SARIMAModel_TrainAndPredict_PureAR which tests
    /// Q=0 with AR order 1 and seasonal AR order 1.
    /// </summary>
    [Fact]
    public void SARIMAModel_PureARNoMA_DoesNotCrash()
    {
        // Regression test: SARIMA with Q=0 and seasonal Q=0 crashed at line 486
        // because lastErrors vector had length 0 and code tried to access index 0
        var (x, y) = CreateSeasonalData(120, period: 12);
        var model = new SARIMAModel<double>(new SARIMAOptions<double>
        {
            P = 2, D = 0, Q = 0,
            SeasonalP = 1, SeasonalD = 0, SeasonalQ = 0,
            SeasonalPeriod = 12,
            MaxIterations = 50
        });

        model.Train(x, y);

        // This should not throw ArgumentOutOfRangeException
        var predictions = model.Predict(x);
        Assert.Equal(x.Rows, predictions.Length);
        AssertFiniteVector(predictions, "SARIMAModel.Predict(Q=0,SQ=0)");
    }

    #endregion

    #region Additional Coverage Tests

    [Fact]
    public void MAModel_EvaluateModel_ProducesFiniteMetrics()
    {
        var (x, y) = CreateStationaryData(100);
        var model = new MAModel<double>(new MAModelOptions<double> { MAOrder = 3 });

        model.Train(x, y);
        var metrics = model.EvaluateModel(x, y);

        Assert.True(metrics.Count > 0, "EvaluateModel returned no metrics");
        AssertFiniteMetrics(metrics);
    }

    [Fact]
    public void MAModel_Clone_PreservesTrainedState()
    {
        var (x, y) = CreateStationaryData(100);
        var model = new MAModel<double>(new MAModelOptions<double> { MAOrder = 3 });

        model.Train(x, y);
        var original = model.Predict(x);
        var clone = (MAModel<double>)model.Clone();
        var clonePredictions = clone.Predict(x);

        Assert.Equal(original.Length, clonePredictions.Length);
        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(original[i], clonePredictions[i], precision: 8);
        }
    }

    [Fact]
    public void ARIMAModel_SerializeAndDeserialize_PreservesPredictions()
    {
        var (x, y) = CreateTrendPlusNoiseData(100);
        var model = new ARIMAModel<double>(new ARIMAOptions<double> { P = 2, D = 1, Q = 0, MaxIterations = 50 });
        model.Train(x, y);

        var originalPredictions = model.Predict(x);
        AssertFiniteVector(originalPredictions, "Original ARIMAModel.Predict");

        // Serialize
        byte[] serialized = model.Serialize();
        Assert.True(serialized.Length > 0, "Serialized data is empty");

        // Deserialize into new model
        var restoredModel = new ARIMAModel<double>(new ARIMAOptions<double> { P = 2, D = 1, Q = 0 });
        restoredModel.Deserialize(serialized);
        var restoredPredictions = restoredModel.Predict(x);

        // Predictions should match (skip initial transient)
        Assert.Equal(originalPredictions.Length, restoredPredictions.Length);
        AssertFiniteVector(restoredPredictions, "Restored ARIMAModel.Predict");
        for (int i = 10; i < originalPredictions.Length; i++)
        {
            Assert.Equal(originalPredictions[i], restoredPredictions[i], precision: 8);
        }
    }

    #endregion
}
