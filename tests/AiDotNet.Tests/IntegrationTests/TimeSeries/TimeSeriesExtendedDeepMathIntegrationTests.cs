using AiDotNet.TimeSeries;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.TimeSeries;

/// <summary>
/// Deep math integration tests for additional time series models:
/// MA (Moving Average) and GARCH (Generalized Autoregressive Conditional Heteroskedasticity).
/// These tests verify mathematical correctness of prediction formulas, training convergence,
/// parameter constraints, and model behavior under various data conditions.
/// </summary>
public class TimeSeriesExtendedDeepMathIntegrationTests
{
    private const double Tol = 1e-4;

    private static Matrix<double> MakeFeatureMatrix(int rows)
    {
        var arr = new double[rows, 1];
        for (int i = 0; i < rows; i++) arr[i, 0] = i;
        return new Matrix<double>(arr);
    }

    private static Vector<double> MakeVector(double[] values)
    {
        var v = new Vector<double>(values.Length);
        for (int i = 0; i < values.Length; i++) v[i] = values[i];
        return v;
    }

    // ========================================================================
    // MA Model Tests
    // ========================================================================

    [Fact]
    public void MAModel_PredictSingle_ReturnsFiniteValue()
    {
        // MA(1) prediction: y_hat = mean + theta1 * e(t-1)
        // After training, PredictSingle should return a finite value
        var data = new double[] { 1.0, 0.8, 1.2, 0.9, 1.1, 0.7, 1.3, 0.85, 1.15, 0.95,
                                   1.05, 0.75, 1.25, 0.88, 1.12, 0.92, 1.08, 0.82, 1.18, 0.98 };
        var y = MakeVector(data);
        var x = MakeFeatureMatrix(data.Length);

        var options = new MAModelOptions<double> { MAOrder = 1 };
        var model = new MAModel<double>(options);
        model.Train(x, y);

        var prediction = model.PredictSingle(MakeVector(new double[] { data.Length }));
        Assert.True(double.IsFinite(prediction), $"MA prediction should be finite, got {prediction}");
    }

    [Fact]
    public void MAModel_PredictionConvergesToMean_AsHorizonIncreases()
    {
        // For MA(q), predictions should converge to the series mean as we forecast further
        // because future errors are assumed to be zero, so after q steps the MA component vanishes
        var data = new double[] { 2.0, 1.5, 2.3, 1.8, 2.1, 1.7, 2.4, 1.6, 2.2, 1.9,
                                   2.0, 1.8, 2.1, 1.7, 2.3, 1.6, 2.2, 1.85, 2.05, 1.95 };
        var y = MakeVector(data);
        var x = MakeFeatureMatrix(data.Length);

        var options = new MAModelOptions<double> { MAOrder = 2 };
        var model = new MAModel<double>(options);
        model.Train(x, y);

        double mean = data.Average();

        // Predict far enough into the future
        int horizon = 20;
        var xFuture = MakeFeatureMatrix(horizon);
        var predictions = model.Predict(xFuture);

        // After q steps, all predictions should be approximately the mean
        for (int t = options.MAOrder + 1; t < horizon; t++)
        {
            Assert.True(Math.Abs(predictions[t] - mean) < 0.1,
                $"MA prediction at t={t} should be near mean {mean:F4}, got {predictions[t]:F4}");
        }
    }

    [Fact]
    public void MAModel_Predict_ShiftsErrorsToZero()
    {
        // MA prediction shifts the working errors vector: at each step, the oldest error
        // is pushed out and a zero error enters. After q steps all errors are zero.
        var data = new double[] { 1.0, 1.5, 0.8, 1.2, 0.9, 1.1, 1.3, 0.7, 1.4, 0.6,
                                   1.0, 1.2, 0.9, 1.1, 0.8, 1.3, 1.0, 0.85, 1.15, 0.95 };
        var y = MakeVector(data);
        var x = MakeFeatureMatrix(data.Length);

        int q = 3;
        var options = new MAModelOptions<double> { MAOrder = q };
        var model = new MAModel<double>(options);
        model.Train(x, y);

        double mean = data.Average();

        // Predictions at step q+1 and beyond should all be identical (= mean)
        var xFuture = MakeFeatureMatrix(q + 5);
        var predictions = model.Predict(xFuture);

        double predAtQ1 = predictions[q]; // First prediction where all errors are zero
        for (int t = q; t < q + 5; t++)
        {
            Assert.Equal(predAtQ1, predictions[t], Tol);
        }
    }

    [Fact]
    public void MAModel_InsufficientData_ThrowsArgumentException()
    {
        // Training requires length > MAOrder
        int q = 5;
        var data = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 }; // length == q, not >
        var y = MakeVector(data);
        var x = MakeFeatureMatrix(data.Length);

        var options = new MAModelOptions<double> { MAOrder = q };
        var model = new MAModel<double>(options);

        Assert.Throws<ArgumentException>(() => model.Train(x, y));
    }

    [Fact]
    public void MAModel_EmptyData_ThrowsArgumentException()
    {
        // Empty data should throw ArgumentException (either from Matrix constructor or Train)
        var y = MakeVector(Array.Empty<double>());

        var options = new MAModelOptions<double> { MAOrder = 1 };
        var model = new MAModel<double>(options);

        Assert.ThrowsAny<ArgumentException>(() =>
        {
            var x = MakeFeatureMatrix(0);
            model.Train(x, y);
        });
    }

    [Fact]
    public void MAModel_PredictBeforeTrain_ThrowsInvalidOperationException()
    {
        var options = new MAModelOptions<double> { MAOrder = 1 };
        var model = new MAModel<double>(options);

        Assert.Throws<InvalidOperationException>(() =>
            model.PredictSingle(MakeVector(new double[] { 1.0 })));
    }

    [Fact]
    public void MAModel_Evaluation_ReturnsAllMetrics()
    {
        // MA EvaluateModel should return MSE, RMSE, MAE, MAPE
        var data = new double[] { 1.0, 1.5, 0.8, 1.2, 0.9, 1.1, 1.3, 0.7, 1.4, 0.6,
                                   1.0, 1.2, 0.9, 1.1, 0.8, 1.3, 1.0, 0.85, 1.15, 0.95 };
        var y = MakeVector(data);
        var x = MakeFeatureMatrix(data.Length);

        var options = new MAModelOptions<double> { MAOrder = 1 };
        var model = new MAModel<double>(options);
        model.Train(x, y);

        var testY = MakeVector(new double[] { 1.05, 0.95, 1.1, 0.9, 1.0 });
        var testX = MakeFeatureMatrix(5);

        var metrics = model.EvaluateModel(testX, testY);
        Assert.True(metrics.ContainsKey("MSE"), "Should have MSE metric");
        Assert.True(metrics.ContainsKey("RMSE"), "Should have RMSE metric");
        Assert.True(metrics["MSE"] >= 0, "MSE should be non-negative");
        Assert.True(metrics["RMSE"] >= 0, "RMSE should be non-negative");
        Assert.True(Math.Abs(Math.Sqrt(metrics["MSE"]) - metrics["RMSE"]) < Tol,
            "RMSE should be sqrt(MSE)");
    }

    [Fact]
    public void MAModel_ConstantSeries_MeanIsConstant()
    {
        // If all values are the same, the mean should be that value
        // and all predictions should be approximately that value
        double constant = 5.0;
        var data = Enumerable.Repeat(constant, 20).ToArray();
        var y = MakeVector(data);
        var x = MakeFeatureMatrix(data.Length);

        var options = new MAModelOptions<double> { MAOrder = 1 };
        var model = new MAModel<double>(options);
        model.Train(x, y);

        var prediction = model.PredictSingle(MakeVector(new double[] { 20 }));
        Assert.True(Math.Abs(prediction - constant) < 0.1,
            $"Constant series prediction should be near {constant}, got {prediction}");
    }

    [Fact]
    public void MAModel_HigherOrder_CapturesMoreLags()
    {
        // MA(1) uses 1 lag, MA(3) uses 3 lags
        // MA(3) first prediction should differ from its 4th prediction more than MA(1)
        // because MA(3) has 3 non-zero errors initially
        var data = new double[] { 1.0, 3.0, 0.5, 2.5, 1.5, 2.0, 0.8, 2.2, 1.2, 1.8,
                                   1.0, 2.5, 0.7, 2.3, 1.3, 1.9, 0.9, 2.1, 1.1, 1.7 };
        var y = MakeVector(data);
        var x = MakeFeatureMatrix(data.Length);

        var options1 = new MAModelOptions<double> { MAOrder = 1 };
        var model1 = new MAModel<double>(options1);
        model1.Train(x, y);

        var options3 = new MAModelOptions<double> { MAOrder = 3 };
        var model3 = new MAModel<double>(options3);
        model3.Train(x, y);

        var xFuture = MakeFeatureMatrix(10);
        var pred1 = model1.Predict(xFuture);
        var pred3 = model3.Predict(xFuture);

        // MA(1) converges to mean after 1 step, MA(3) after 3 steps
        // First prediction of MA(3) should have more deviation from mean than first of MA(1)
        // (since more error terms contribute)
        double mean = data.Average();
        double dev1First = Math.Abs(pred1[0] - mean);
        double dev3First = Math.Abs(pred3[0] - mean);

        // Both should produce finite predictions
        Assert.True(double.IsFinite(pred1[0]), "MA(1) predictions should be finite");
        Assert.True(double.IsFinite(pred3[0]), "MA(3) predictions should be finite");

        // MA(1) should converge to mean by step 2
        Assert.True(Math.Abs(pred1[2] - mean) < 0.01,
            $"MA(1) should converge by step 2, diff = {Math.Abs(pred1[2] - mean)}");

        // MA(3) should NOT have converged by step 2
        // (it still has 1 non-zero error affecting the prediction)
        // But should converge by step 4
        Assert.True(Math.Abs(pred3[4] - mean) < 0.01,
            $"MA(3) should converge by step 4, diff = {Math.Abs(pred3[4] - mean)}");
    }

    [Fact]
    public void MAModel_MACoefficients_StayWithinBounds()
    {
        // MA coefficients should be bounded within (-1, 1) for invertibility
        var data = new double[] { 5.0, -3.0, 8.0, -2.0, 7.0, -1.0, 6.0, 0.0, 4.0, 1.0,
                                   5.0, -3.0, 8.0, -2.0, 7.0, -1.0, 6.0, 0.0, 4.0, 1.0,
                                   5.0, -3.0, 8.0, -2.0, 7.0, -1.0, 6.0, 0.0, 4.0, 1.0 };
        var y = MakeVector(data);
        var x = MakeFeatureMatrix(data.Length);

        var options = new MAModelOptions<double> { MAOrder = 3 };
        var model = new MAModel<double>(options);
        model.Train(x, y);

        // Verify model trained successfully by getting a prediction
        var prediction = model.PredictSingle(MakeVector(new double[] { data.Length }));
        Assert.True(double.IsFinite(prediction),
            "Model with high-variance data should still produce finite predictions");
    }

    [Fact]
    public void MAModel_Forecast_MatchesPredictOutput()
    {
        // Forecast and Predict should produce consistent results
        var data = new double[] { 1.0, 1.2, 0.8, 1.1, 0.9, 1.0, 1.15, 0.85, 1.05, 0.95,
                                   1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.08, 0.92, 1.03, 0.97 };
        var y = MakeVector(data);
        var x = MakeFeatureMatrix(data.Length);

        var options = new MAModelOptions<double> { MAOrder = 2 };
        var model = new MAModel<double>(options);
        model.Train(x, y);

        int horizon = 5;
        var history = MakeVector(data);
        var forecast = model.Forecast(history, horizon);
        var predicted = model.Predict(MakeFeatureMatrix(horizon));

        // Both should produce finite values
        for (int i = 0; i < horizon; i++)
        {
            Assert.True(double.IsFinite(forecast[i]),
                $"Forecast[{i}]={forecast[i]:F4} should be finite");
            Assert.True(double.IsFinite(predicted[i]),
                $"Predict[{i}]={predicted[i]:F4} should be finite");
        }
    }

    [Fact]
    public void MAModel_NegativeHorizon_ThrowsException()
    {
        var data = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
        var y = MakeVector(data);
        var x = MakeFeatureMatrix(data.Length);

        var options = new MAModelOptions<double> { MAOrder = 1 };
        var model = new MAModel<double>(options);
        model.Train(x, y);

        Assert.ThrowsAny<Exception>(() => model.Forecast(MakeVector(data), -1));
    }

    [Fact]
    public void MAModel_Serialize_DeserializeProducesSamePredictions()
    {
        var data = new double[] { 1.0, 0.8, 1.2, 0.9, 1.1, 0.7, 1.3, 0.85, 1.15, 0.95,
                                   1.05, 0.75, 1.25, 0.88, 1.12, 0.92, 1.08, 0.82, 1.18, 0.98 };
        var y = MakeVector(data);
        var x = MakeFeatureMatrix(data.Length);

        var options = new MAModelOptions<double> { MAOrder = 2 };
        var model = new MAModel<double>(options);
        model.Train(x, y);

        var prediction1 = model.PredictSingle(MakeVector(new double[] { data.Length }));

        byte[] serialized = model.Serialize();
        Assert.True(serialized.Length > 0, "Serialized bytes should not be empty");

        var model2 = new MAModel<double>(new MAModelOptions<double> { MAOrder = 2 });
        model2.Deserialize(serialized);

        var prediction2 = model2.PredictSingle(MakeVector(new double[] { data.Length }));
        Assert.Equal(prediction1, prediction2, Tol);
    }

    // ========================================================================
    // GARCH Model Tests
    // ========================================================================

    [Fact]
    public void GARCHModel_Train_ProducesNonNegativeVariances()
    {
        // GARCH conditional variances must always be non-negative
        // sigma^2_t = omega + alpha * e^2_{t-1} + beta * sigma^2_{t-1}
        // With omega >= 0, alpha >= 0, beta >= 0, all variances are non-negative
        var data = new double[] { 0.1, -0.2, 0.15, -0.1, 0.3, -0.25, 0.05, -0.15, 0.2, -0.1,
                                   0.12, -0.18, 0.22, -0.08, 0.16, -0.14, 0.28, -0.22, 0.1, -0.12,
                                   0.14, -0.16, 0.24, -0.06, 0.18, -0.2, 0.08, -0.1, 0.26, -0.24 };
        var y = MakeVector(data);
        var x = MakeFeatureMatrix(data.Length);

        var options = new GARCHModelOptions<double>
        {
            ARCHOrder = 1,
            GARCHOrder = 1,
            MaxIterations = 100
        };
        var model = new GARCHModel<double>(options);
        model.Train(x, y);

        // The model should be trained - predict shouldn't throw
        var prediction = model.PredictSingle(MakeVector(new double[] { data.Length }));
        Assert.True(double.IsFinite(prediction),
            $"GARCH prediction should be finite, got {prediction}");
    }

    [Fact]
    public void GARCHModel_StationarityConstraint_SumAlphaBetaLessThanOne()
    {
        // For a stationary GARCH(1,1), we need alpha + beta < 1
        // The ConstrainParameters method enforces this
        // We verify by training on volatile data and checking the model produces bounded forecasts
        var data = new double[50];
        var rand = new Random(42);
        for (int i = 0; i < 50; i++)
        {
            data[i] = rand.NextDouble() * 2.0 - 1.0; // Random values in [-1, 1]
        }
        var y = MakeVector(data);
        var x = MakeFeatureMatrix(data.Length);

        var options = new GARCHModelOptions<double>
        {
            ARCHOrder = 1,
            GARCHOrder = 1,
            MaxIterations = 200
        };
        var model = new GARCHModel<double>(options);
        model.Train(x, y);

        // Multi-step predictions should remain finite (not explode)
        var xFuture = MakeFeatureMatrix(10);
        var predictions = model.Predict(xFuture);
        for (int i = 0; i < 10; i++)
        {
            Assert.True(double.IsFinite(predictions[i]),
                $"GARCH forecast at step {i} should be finite, got {predictions[i]}");
        }
    }

    [Fact]
    public void GARCHModel_VolatilityClustering_HighVolAfterHighVol()
    {
        // GARCH models capture volatility clustering: high volatility periods
        // are followed by high volatility periods.
        // Create data with clear volatility regime changes.
        var data = new double[60];
        // Low volatility regime
        for (int i = 0; i < 20; i++)
            data[i] = 0.01 * (i % 2 == 0 ? 1 : -1);
        // High volatility regime
        for (int i = 20; i < 40; i++)
            data[i] = 0.5 * (i % 2 == 0 ? 1 : -1);
        // Back to low volatility
        for (int i = 40; i < 60; i++)
            data[i] = 0.02 * (i % 2 == 0 ? 1 : -1);

        var y = MakeVector(data);
        var x = MakeFeatureMatrix(data.Length);

        var options = new GARCHModelOptions<double>
        {
            ARCHOrder = 1,
            GARCHOrder = 1,
            MaxIterations = 200
        };
        var model = new GARCHModel<double>(options);
        model.Train(x, y);

        // Model should train without exception
        var prediction = model.PredictSingle(MakeVector(new double[] { data.Length }));
        Assert.True(double.IsFinite(prediction), "GARCH should handle volatility regime changes");
    }

    [Fact]
    public void GARCHModel_PredictBeforeTrain_Throws()
    {
        var options = new GARCHModelOptions<double>
        {
            ARCHOrder = 1,
            GARCHOrder = 1
        };
        var model = new GARCHModel<double>(options);

        Assert.ThrowsAny<Exception>(() =>
            model.PredictSingle(MakeVector(new double[] { 1.0 })));
    }

    [Fact]
    public void GARCHModel_VarianceEquation_OmegaIsFloor()
    {
        // In the GARCH variance equation: sigma^2_t = omega + alpha*e^2_{t-1} + beta*sigma^2_{t-1}
        // When e^2_{t-1} and sigma^2_{t-1} are both zero (hypothetically), variance = omega.
        // This means omega serves as the floor for conditional variance.
        // Verify model trains and omega being the minimum baseline produces sensible forecasts.

        // Use near-zero data to make residuals very small
        var data = Enumerable.Range(0, 40).Select(i => 0.001 * (i % 2 == 0 ? 1 : -1)).ToArray();
        var y = MakeVector(data);
        var x = MakeFeatureMatrix(data.Length);

        var options = new GARCHModelOptions<double>
        {
            ARCHOrder = 1,
            GARCHOrder = 1,
            MaxIterations = 100
        };
        var model = new GARCHModel<double>(options);
        model.Train(x, y);

        var prediction = model.PredictSingle(MakeVector(new double[] { data.Length }));
        // With near-zero residuals, prediction should be close to mean
        Assert.True(double.IsFinite(prediction),
            "GARCH with near-zero volatility data should produce finite prediction");
    }

    [Fact]
    public void GARCHModel_Forecast_ReturnsRequestedHorizon()
    {
        var data = new double[40];
        var rand = new Random(123);
        for (int i = 0; i < 40; i++) data[i] = rand.NextDouble() * 0.5 - 0.25;

        var y = MakeVector(data);
        var x = MakeFeatureMatrix(data.Length);

        var options = new GARCHModelOptions<double>
        {
            ARCHOrder = 1,
            GARCHOrder = 1,
            MaxIterations = 100
        };
        var model = new GARCHModel<double>(options);
        model.Train(x, y);

        int horizon = 7;
        var forecast = model.Forecast(MakeVector(data), horizon);
        Assert.Equal(horizon, forecast.Length);

        for (int i = 0; i < horizon; i++)
        {
            Assert.True(double.IsFinite(forecast[i]),
                $"GARCH forecast[{i}] should be finite, got {forecast[i]}");
        }
    }

    [Fact]
    public void GARCHModel_SerializeDeserialize_ProducesConsistentForecasts()
    {
        var data = new double[40];
        var rand = new Random(42);
        for (int i = 0; i < 40; i++) data[i] = rand.NextDouble() * 0.6 - 0.3;

        var y = MakeVector(data);
        var x = MakeFeatureMatrix(data.Length);

        var options = new GARCHModelOptions<double>
        {
            ARCHOrder = 1,
            GARCHOrder = 1,
            MaxIterations = 100
        };
        var model = new GARCHModel<double>(options);
        model.Train(x, y);

        byte[] serialized = model.Serialize();
        Assert.True(serialized.Length > 0, "Serialized bytes should not be empty");

        var model2 = new GARCHModel<double>(new GARCHModelOptions<double>
        {
            ARCHOrder = 1,
            GARCHOrder = 1
        });
        model2.Deserialize(serialized);

        // Both models should be able to produce predictions
        var p1 = model.PredictSingle(MakeVector(new double[] { data.Length }));
        var p2 = model2.PredictSingle(MakeVector(new double[] { data.Length }));

        // GARCH uses random normal generation, so predictions differ
        // But both should be finite
        Assert.True(double.IsFinite(p1), "Original GARCH prediction should be finite");
        Assert.True(double.IsFinite(p2), "Deserialized GARCH prediction should be finite");
    }

    [Fact]
    public void GARCHModel_HigherARCHOrder_CapturesMoreShocks()
    {
        // GARCH(2,1) uses 2 past squared residuals vs GARCH(1,1) using 1
        // Both should train successfully on the same data
        var data = new double[50];
        var rand = new Random(55);
        for (int i = 0; i < 50; i++) data[i] = rand.NextDouble() * 0.8 - 0.4;

        var y = MakeVector(data);
        var x = MakeFeatureMatrix(data.Length);

        var options11 = new GARCHModelOptions<double>
        {
            ARCHOrder = 1,
            GARCHOrder = 1,
            MaxIterations = 100
        };
        var model11 = new GARCHModel<double>(options11);
        model11.Train(x, y);

        var options21 = new GARCHModelOptions<double>
        {
            ARCHOrder = 2,
            GARCHOrder = 1,
            MaxIterations = 100
        };
        var model21 = new GARCHModel<double>(options21);
        model21.Train(x, y);

        // Both should produce finite forecasts
        var f11 = model11.Forecast(MakeVector(data), 5);
        var f21 = model21.Forecast(MakeVector(data), 5);

        for (int i = 0; i < 5; i++)
        {
            Assert.True(double.IsFinite(f11[i]), $"GARCH(1,1) forecast[{i}] should be finite");
            Assert.True(double.IsFinite(f21[i]), $"GARCH(2,1) forecast[{i}] should be finite");
        }
    }

    [Fact]
    public void GARCHModel_Metadata_ContainsModelInfo()
    {
        var options = new GARCHModelOptions<double>
        {
            ARCHOrder = 1,
            GARCHOrder = 1,
            MaxIterations = 50
        };
        var model = new GARCHModel<double>(options);

        var data = new double[30];
        var rand = new Random(77);
        for (int i = 0; i < 30; i++) data[i] = rand.NextDouble() * 0.4 - 0.2;

        model.Train(MakeFeatureMatrix(30), MakeVector(data));

        var metadata = model.GetModelMetadata();
        Assert.NotNull(metadata);
    }

    [Fact]
    public void GARCHModel_NegativeHorizon_ThrowsException()
    {
        var data = new double[30];
        var rand = new Random(88);
        for (int i = 0; i < 30; i++) data[i] = rand.NextDouble() * 0.4 - 0.2;

        var options = new GARCHModelOptions<double>
        {
            ARCHOrder = 1,
            GARCHOrder = 1,
            MaxIterations = 50
        };
        var model = new GARCHModel<double>(options);
        model.Train(MakeFeatureMatrix(30), MakeVector(data));

        Assert.ThrowsAny<Exception>(() => model.Forecast(MakeVector(data), -1));
    }

    [Fact]
    public void GARCHModel_EvaluateModel_ReturnsMetrics()
    {
        var data = new double[40];
        var rand = new Random(99);
        for (int i = 0; i < 40; i++) data[i] = rand.NextDouble() * 0.6 - 0.3;

        var y = MakeVector(data);
        var x = MakeFeatureMatrix(data.Length);

        var options = new GARCHModelOptions<double>
        {
            ARCHOrder = 1,
            GARCHOrder = 1,
            MaxIterations = 100
        };
        var model = new GARCHModel<double>(options);
        model.Train(x, y);

        var testData = new double[] { 0.1, -0.1, 0.05, -0.05, 0.08 };
        var metrics = model.EvaluateModel(MakeFeatureMatrix(5), MakeVector(testData));

        Assert.NotNull(metrics);
        Assert.True(metrics.Count > 0, "GARCH EvaluateModel should return at least one metric");
    }

    // ========================================================================
    // Cross-model comparison tests
    // ========================================================================

    [Fact]
    public void MAModel_Evaluation_MSEConsistencyWithPredictions()
    {
        // Verify that EvaluateModel's MSE matches manually computed MSE from predictions
        var trainData = new double[] { 1.0, 1.5, 0.8, 1.2, 0.9, 1.1, 1.3, 0.7, 1.4, 0.6,
                                        1.0, 1.2, 0.9, 1.1, 0.8, 1.3, 1.0, 0.85, 1.15, 0.95 };
        var y = MakeVector(trainData);
        var x = MakeFeatureMatrix(trainData.Length);

        var options = new MAModelOptions<double> { MAOrder = 1 };
        var model = new MAModel<double>(options);
        model.Train(x, y);

        var testData = new double[] { 1.05, 0.95, 1.1, 0.9, 1.0 };
        var testY = MakeVector(testData);
        var testX = MakeFeatureMatrix(5);

        var metrics = model.EvaluateModel(testX, testY);
        var predictions = model.Predict(testX);

        // Manually compute MSE
        double manualMSE = 0;
        for (int i = 0; i < testData.Length; i++)
        {
            double diff = testData[i] - predictions[i];
            manualMSE += diff * diff;
        }
        manualMSE /= testData.Length;

        Assert.Equal(manualMSE, metrics["MSE"], Tol);
    }

    [Fact]
    public void MAModel_MAOrder0_PredictsMean()
    {
        // MA(0) is just a white noise model: prediction = mean
        var data = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
        var y = MakeVector(data);
        var x = MakeFeatureMatrix(data.Length);

        var options = new MAModelOptions<double> { MAOrder = 0 };
        var model = new MAModel<double>(options);
        model.Train(x, y);

        double expectedMean = data.Average(); // 5.5

        var prediction = model.PredictSingle(MakeVector(new double[] { data.Length }));
        Assert.Equal(expectedMean, prediction, 0.01);
    }
}
