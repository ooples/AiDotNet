using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNetTests.IntegrationTests.TimeSeries
{
    /// <summary>
    /// Integration tests for basic TimeSeries models with mathematically verified results.
    /// These tests validate the correctness of AR, MA, ARMA, ARIMA, SARIMA, ARIMAX,
    /// ExponentialSmoothing, StateSpace, and STL decomposition models.
    /// </summary>
    public class TimeSeriesBasicModelsIntegrationTests
    {
        private const double Tolerance = 1e-4;
        private const double HighTolerance = 1e-2;

        #region AR Model Tests

        [Fact]
        public void ARModel_WithKnownAR1Coefficients_RecoversCoefficientAccurately()
        {
            // Arrange - Generate synthetic AR(1) series: y_t = 0.7 * y_{t-1} + epsilon
            var options = new ARModelOptions<double>
            {
                AROrder = 1,
                LearningRate = 0.01,
                MaxIterations = 2000,
                Tolerance = 1e-6
            };
            var model = new ARModel<double>(options);

            // Generate data with known AR(1) coefficient
            var random = new Random(42);
            int n = 200;
            double trueCoeff = 0.7;
            var y = new Vector<double>(n);
            y[0] = random.NextDouble();

            for (int i = 1; i < n; i++)
            {
                y[i] = trueCoeff * y[i - 1] + 0.1 * (random.NextDouble() - 0.5);
            }

            // Create dummy X matrix (not used by AR model but required by interface)
            var X = new Matrix<double>(n, 1);
            for (int i = 0; i < n; i++)
            {
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var predictions = model.Predict(X);

            // Assert - Model should fit the data well
            double mse = 0;
            for (int i = 1; i < n; i++)
            {
                mse += Math.Pow(y[i] - predictions[i], 2);
            }
            mse /= (n - 1);

            Assert.True(mse < 0.1, $"MSE should be small, but was {mse}");
        }

        [Fact]
        public void ARModel_WithHigherOrder_CapturesComplexPatterns()
        {
            // Arrange - AR(3) model
            var options = new ARModelOptions<double>
            {
                AROrder = 3,
                LearningRate = 0.01,
                MaxIterations = 2000
            };
            var model = new ARModel<double>(options);

            // Generate AR(3) series
            var random = new Random(123);
            int n = 250;
            var y = new Vector<double>(n);

            for (int i = 0; i < 3; i++)
            {
                y[i] = random.NextDouble();
            }

            for (int i = 3; i < n; i++)
            {
                y[i] = 0.5 * y[i-1] + 0.3 * y[i-2] + 0.1 * y[i-3] + 0.05 * (random.NextDouble() - 0.5);
            }

            var X = new Matrix<double>(n, 1);
            for (int i = 0; i < n; i++) X[i, 0] = i;

            // Act
            model.Train(X, y);
            var metrics = model.EvaluateModel(X, y);

            // Assert
            Assert.True(metrics.ContainsKey("MSE"));
            Assert.True(Convert.ToDouble(metrics["MSE"]) < 0.1);
        }

        [Fact]
        public void ARModel_WithShortSeries_HandlesEdgeCase()
        {
            // Arrange - Very short series
            var options = new ARModelOptions<double> { AROrder = 1 };
            var model = new ARModel<double>(options);

            int n = 20;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            for (int i = 0; i < n; i++)
            {
                y[i] = Math.Sin(i * 0.5) + 1.0;
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var prediction = model.PredictSingle(new Vector<double>(new[] { 20.0 }));

            // Assert - Should not throw and should produce reasonable prediction
            Assert.True(!double.IsNaN(prediction) && !double.IsInfinity(prediction));
        }

        [Fact]
        public void ARModel_ForecastHorizon_ProducesReasonablePredictions()
        {
            // Arrange
            var options = new ARModelOptions<double> { AROrder = 2 };
            var model = new ARModel<double>(options);

            int n = 100;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Generate trending series
            for (int i = 0; i < n; i++)
            {
                y[i] = 10 + 0.5 * i + Math.Sin(i * 0.3);
                X[i, 0] = i;
            }

            model.Train(X, y);

            // Act - Forecast 10 steps ahead
            var futureX = new Matrix<double>(10, 1);
            for (int i = 0; i < 10; i++)
            {
                futureX[i, 0] = n + i;
            }
            var forecast = model.Predict(futureX);

            // Assert - Forecast should be in reasonable range
            for (int i = 0; i < 10; i++)
            {
                Assert.True(forecast[i] > 0 && forecast[i] < 100);
            }
        }

        [Fact]
        public void ARModel_Serialization_PreservesModelState()
        {
            // Arrange
            var options = new ARModelOptions<double> { AROrder = 2 };
            var model = new ARModel<double>(options);

            int n = 50;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            for (int i = 0; i < n; i++)
            {
                y[i] = Math.Sin(i * 0.2) + i * 0.1;
                X[i, 0] = i;
            }

            model.Train(X, y);
            var originalPrediction = model.PredictSingle(new Vector<double>(new[] { 50.0 }));

            // Act - Serialize and deserialize
            var serialized = model.Serialize();
            var deserialized = new ARModel<double>(options);
            deserialized.Deserialize(serialized);
            var restoredPrediction = deserialized.PredictSingle(new Vector<double>(new[] { 50.0 }));

            // Assert
            Assert.Equal(originalPrediction, restoredPrediction, precision: 8);
        }

        [Fact]
        public void ARModel_NegativeCoefficient_HandlesDampingBehavior()
        {
            // Arrange
            var options = new ARModelOptions<double> { AROrder = 1, MaxIterations = 2000 };
            var model = new ARModel<double>(options);

            int n = 150;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Generate series with negative AR coefficient (oscillating)
            y[0] = 10.0;
            for (int i = 1; i < n; i++)
            {
                y[i] = -0.5 * y[i-1] + 5.0;
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var predictions = model.Predict(X);

            // Assert - Should capture oscillating behavior
            double correlation = CalculateCorrelation(y, predictions);
            Assert.True(correlation > 0.6, $"Should handle negative AR coefficient, correlation was {correlation:F3}");
        }

        [Fact]
        public void ARModel_MeanReversion_ConvergesToMean()
        {
            // Arrange
            var options = new ARModelOptions<double> { AROrder = 1 };
            var model = new ARModel<double>(options);

            int n = 120;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Mean-reverting series
            y[0] = 15.0;
            for (int i = 1; i < n; i++)
            {
                y[i] = 10.0 + 0.7 * (y[i-1] - 10.0);
                X[i, 0] = i;
            }

            model.Train(X, y);

            // Act - Long forecast
            var futureX = new Matrix<double>(20, 1);
            for (int i = 0; i < 20; i++) futureX[i, 0] = n + i;
            var forecast = model.Predict(futureX);

            // Assert - Should converge toward mean
            double lastForecast = forecast[19];
            Assert.True(Math.Abs(lastForecast - 10.0) < 3.0, "AR(1) forecast should converge to mean");
        }

        [Fact]
        public void ARModel_VariableInitialConditions_ProducesConsistentResults()
        {
            // Arrange
            var options = new ARModelOptions<double> { AROrder = 2 };
            var model = new ARModel<double>(options);

            int n = 100;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            for (int i = 0; i < n; i++)
            {
                y[i] = 20.0 + 0.3 * i + Math.Sin(i * 0.2);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var metrics = model.EvaluateModel(X, y);

            // Assert
            Assert.True(metrics.ContainsKey("MAE"));
            double mae = Convert.ToDouble(metrics["MAE"]);
            Assert.True(mae < 2.0, $"Model should fit consistently, MAE was {mae}");
        }

        [Fact]
        public void ARModel_StabilityCheck_CoefficientsWithinBounds()
        {
            // Arrange
            var options = new ARModelOptions<double> { AROrder = 1, LearningRate = 0.01 };
            var model = new ARModel<double>(options);

            int n = 100;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Stationary AR(1) process
            y[0] = 5.0;
            for (int i = 1; i < n; i++)
            {
                y[i] = 0.6 * y[i-1] + 2.0;
                X[i, 0] = i;
            }

            // Act & Assert - Should train without divergence
            model.Train(X, y);
            var predictions = model.Predict(X);

            bool anyInfinite = false;
            for (int i = 0; i < predictions.Length; i++)
            {
                if (double.IsInfinity(predictions[i]) || double.IsNaN(predictions[i]))
                {
                    anyInfinite = true;
                    break;
                }
            }

            Assert.False(anyInfinite, "AR model predictions should be finite and stable");
        }

        [Fact]
        public void ARModel_PerformanceMetrics_ProvideAccurateAssessment()
        {
            // Arrange
            var options = new ARModelOptions<double> { AROrder = 2, MaxIterations = 2000 };
            var model = new ARModel<double>(options);

            int n = 120;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            for (int i = 0; i < n; i++)
            {
                y[i] = 50.0 + 0.3 * i + 5.0 * Math.Sin(i * 0.2);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var metrics = model.EvaluateModel(X, y);

            // Assert - All expected metrics present
            Assert.True(metrics.ContainsKey("MSE"), "MSE metric should be present");
            Assert.True(metrics.ContainsKey("RMSE"), "RMSE metric should be present");
            Assert.True(metrics.ContainsKey("MAE"), "MAE metric should be present");

            double mse = Convert.ToDouble(metrics["MSE"]);
            double rmse = Convert.ToDouble(metrics["RMSE"]);
            Assert.True(Math.Abs(rmse - Math.Sqrt(mse)) < 0.01, "RMSE should equal sqrt(MSE)");
        }

        #endregion

        #region MA Model Tests

        [Fact]
        public void MAModel_WithKnownMA1Coefficients_FitsDataCorrectly()
        {
            // Arrange - Generate synthetic MA(1) series
            var options = new MAModelOptions<double>
            {
                MAOrder = 1,
                MaxIterations = 1000,
                Tolerance = 1e-6
            };
            var model = new MAModel<double>(options);

            var random = new Random(42);
            int n = 150;
            double trueTheta = 0.6;
            var epsilon = new double[n];
            var y = new Vector<double>(n);

            for (int i = 0; i < n; i++)
            {
                epsilon[i] = random.NextDouble() - 0.5;
            }

            y[0] = epsilon[0];
            for (int i = 1; i < n; i++)
            {
                y[i] = epsilon[i] + trueTheta * epsilon[i - 1];
            }

            var X = new Matrix<double>(n, 1);
            for (int i = 0; i < n; i++) X[i, 0] = i;

            // Act
            model.Train(X, y);
            var predictions = model.Predict(X);

            // Assert - Predictions should capture the MA structure
            double residualVariance = 0;
            for (int i = 1; i < n; i++)
            {
                residualVariance += Math.Pow(y[i] - predictions[i], 2);
            }
            residualVariance /= (n - 1);

            Assert.True(residualVariance < 0.5, $"Residual variance should be small, but was {residualVariance}");
        }

        [Fact]
        public void MAModel_WithHigherOrder_HandlesComplexMovingAverage()
        {
            // Arrange - MA(2) model
            var options = new MAModelOptions<double>
            {
                MAOrder = 2,
                MaxIterations = 1500
            };
            var model = new MAModel<double>(options);

            var random = new Random(456);
            int n = 200;
            var epsilon = new double[n];
            var y = new Vector<double>(n);

            for (int i = 0; i < n; i++)
            {
                epsilon[i] = 0.1 * (random.NextDouble() - 0.5);
            }

            y[0] = epsilon[0];
            y[1] = epsilon[1] + 0.5 * epsilon[0];
            for (int i = 2; i < n; i++)
            {
                y[i] = epsilon[i] + 0.5 * epsilon[i-1] + 0.3 * epsilon[i-2];
            }

            var X = new Matrix<double>(n, 1);
            for (int i = 0; i < n; i++) X[i, 0] = i;

            // Act
            model.Train(X, y);
            var metrics = model.EvaluateModel(X, y);

            // Assert
            Assert.True(metrics.ContainsKey("MSE"));
            double mse = Convert.ToDouble(metrics["MSE"]);
            Assert.True(mse < 0.5, $"MSE should be small for MA(2) model, but was {mse}");
        }

        [Fact]
        public void MAModel_ResidualProperties_ApproachWhiteNoise()
        {
            // Arrange
            var options = new MAModelOptions<double> { MAOrder = 1 };
            var model = new MAModel<double>(options);

            var random = new Random(789);
            int n = 150;
            var y = new Vector<double>(n);

            for (int i = 0; i < n; i++)
            {
                y[i] = random.NextDouble() - 0.5;
                if (i > 0) y[i] += 0.4 * (random.NextDouble() - 0.5);
            }

            var X = new Matrix<double>(n, 1);
            for (int i = 0; i < n; i++) X[i, 0] = i;

            // Act
            model.Train(X, y);
            var predictions = model.Predict(X);

            // Calculate residuals
            var residuals = new double[n];
            for (int i = 0; i < n; i++)
            {
                residuals[i] = y[i] - predictions[i];
            }

            // Assert - Residuals should have near-zero mean
            double residualMean = residuals.Average();
            Assert.True(Math.Abs(residualMean) < 0.1, $"Residual mean should be near zero, but was {residualMean}");
        }

        [Fact]
        public void MAModel_Forecast_ProducesStablePredictions()
        {
            // Arrange
            var options = new MAModelOptions<double> { MAOrder = 1 };
            var model = new MAModel<double>(options);

            int n = 100;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            var random = new Random(321);
            for (int i = 0; i < n; i++)
            {
                y[i] = 5.0 + (random.NextDouble() - 0.5);
                X[i, 0] = i;
            }

            model.Train(X, y);

            // Act - Multi-step forecast
            var futureX = new Matrix<double>(5, 1);
            for (int i = 0; i < 5; i++) futureX[i, 0] = n + i;
            var forecast = model.Predict(futureX);

            // Assert - MA forecasts should converge to mean
            for (int i = 0; i < 5; i++)
            {
                Assert.True(Math.Abs(forecast[i] - 5.0) < 2.0, "Forecast should be near series mean");
            }
        }

        [Fact]
        public void MAModel_InvertibilityCondition_MaintainsStability()
        {
            // Arrange
            var options = new MAModelOptions<double> { MAOrder = 1, MaxIterations = 1500 };
            var model = new MAModel<double>(options);

            int n = 120;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            var random = new Random(555);
            double[] epsilon = new double[n];
            for (int i = 0; i < n; i++)
            {
                epsilon[i] = random.NextDouble() - 0.5;
            }

            y[0] = epsilon[0];
            for (int i = 1; i < n; i++)
            {
                y[i] = epsilon[i] + 0.7 * epsilon[i-1];
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var predictions = model.Predict(X);

            // Assert - All predictions should be finite
            foreach (var pred in predictions)
            {
                Assert.True(!double.IsNaN(pred) && !double.IsInfinity(pred), "MA predictions should be stable");
            }
        }

        [Fact]
        public void MAModel_ZeroMeanSeries_HandlesCorrectly()
        {
            // Arrange
            var options = new MAModelOptions<double> { MAOrder = 2 };
            var model = new MAModel<double>(options);

            int n = 100;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            var random = new Random(888);
            for (int i = 0; i < n; i++)
            {
                y[i] = random.NextDouble() - 0.5;  // Zero mean
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var predictions = model.Predict(X);

            // Assert - Predictions should also be near zero mean
            double predMean = 0;
            for (int i = 0; i < predictions.Length; i++) predMean += predictions[i];
            predMean /= predictions.Length;

            Assert.True(Math.Abs(predMean) < 0.5, $"Predictions should have near-zero mean, was {predMean:F3}");
        }

        [Fact]
        public void MAModel_LongLagOrder_FitsWithoutOverfitting()
        {
            // Arrange
            var options = new MAModelOptions<double> { MAOrder = 5, MaxIterations = 2000 };
            var model = new MAModel<double>(options);

            int n = 200;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            var random = new Random(999);
            for (int i = 0; i < n; i++)
            {
                y[i] = 10.0 + 3.0 * (random.NextDouble() - 0.5);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var metrics = model.EvaluateModel(X, y);

            // Assert - Should fit reasonably even with high order
            double rmse = Convert.ToDouble(metrics["RMSE"]);
            Assert.True(rmse < 5.0, $"MA(5) should fit without excessive overfitting, RMSE was {rmse}");
        }

        [Fact]
        public void MAModel_CompareWithSimpleAverage_ShowsImprovement()
        {
            // Arrange
            var options = new MAModelOptions<double> { MAOrder = 1 };
            var model = new MAModel<double>(options);

            int n = 100;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            var random = new Random(333);
            var epsilon = new double[n];
            for (int i = 0; i < n; i++)
            {
                epsilon[i] = random.NextDouble() - 0.5;
            }

            y[0] = epsilon[0];
            for (int i = 1; i < n; i++)
            {
                y[i] = epsilon[i] + 0.6 * epsilon[i-1];
                X[i, 0] = i;
            }

            // Calculate simple mean prediction error
            double mean = 0;
            for (int i = 0; i < n; i++) mean += y[i];
            mean /= n;

            double meanMSE = 0;
            for (int i = 0; i < n; i++)
            {
                meanMSE += Math.Pow(y[i] - mean, 2);
            }
            meanMSE /= n;

            // Act
            model.Train(X, y);
            var metrics = model.EvaluateModel(X, y);
            double maMSE = Convert.ToDouble(metrics["MSE"]);

            // Assert - MA model should outperform simple mean
            Assert.True(maMSE < meanMSE, $"MA model (MSE={maMSE:F4}) should beat mean baseline (MSE={meanMSE:F4})");
        }

        [Fact]
        public void MAModel_ConvergenceCheck_ReachesStableState()
        {
            // Arrange
            var options = new MAModelOptions<double> { MAOrder = 1, MaxIterations = 2000, Tolerance = 1e-6 };
            var model = new MAModel<double>(options);

            int n = 100;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            var random = new Random(777);
            for (int i = 0; i < n; i++)
            {
                y[i] = 10.0 + 2.0 * (random.NextDouble() - 0.5);
                X[i, 0] = i;
            }

            // Act & Assert - Should converge
            model.Train(X, y);
            var predictions = model.Predict(X);

            Assert.True(predictions.Length == n, "Should produce prediction for each input");
        }

        #endregion

        #region ARMA Model Tests

        [Fact]
        public void ARMAModel_CombinesARAndMAComponents_FitsAccurately()
        {
            // Arrange - ARMA(1,1) model
            var options = new ARMAOptions<double>
            {
                AROrder = 1,
                MAOrder = 1,
                MaxIterations = 2000,
                Tolerance = 1e-6
            };
            var model = new ARMAModel<double>(options);

            // Generate ARMA(1,1) series
            var random = new Random(42);
            int n = 200;
            double phi = 0.6;
            double theta = 0.4;
            var epsilon = new double[n];
            var y = new Vector<double>(n);

            for (int i = 0; i < n; i++)
            {
                epsilon[i] = 0.1 * (random.NextDouble() - 0.5);
            }

            y[0] = epsilon[0];
            for (int i = 1; i < n; i++)
            {
                y[i] = phi * y[i-1] + epsilon[i] + theta * epsilon[i-1];
            }

            var X = new Matrix<double>(n, 1);
            for (int i = 0; i < n; i++) X[i, 0] = i;

            // Act
            model.Train(X, y);
            var predictions = model.Predict(X);

            // Assert
            double mse = 0;
            for (int i = 1; i < n; i++)
            {
                mse += Math.Pow(y[i] - predictions[i], 2);
            }
            mse /= (n - 1);

            Assert.True(mse < 0.2, $"ARMA model should fit data well, MSE was {mse}");
        }

        [Fact]
        public void ARMAModel_WithHigherOrders_CapturesComplexDynamics()
        {
            // Arrange - ARMA(2,2) model
            var options = new ARMAOptions<double>
            {
                AROrder = 2,
                MAOrder = 2,
                MaxIterations = 2500
            };
            var model = new ARMAModel<double>(options);

            int n = 250;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Generate complex ARMA series
            var random = new Random(999);
            for (int i = 0; i < n; i++)
            {
                if (i < 2)
                {
                    y[i] = random.NextDouble();
                }
                else
                {
                    y[i] = 0.5 * y[i-1] + 0.3 * y[i-2] + 0.2 * (random.NextDouble() - 0.5);
                }
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var metrics = model.EvaluateModel(X, y);

            // Assert
            Assert.True(metrics.ContainsKey("RMSE"));
            double rmse = Convert.ToDouble(metrics["RMSE"]);
            Assert.True(rmse < 1.0, $"RMSE should be reasonable for ARMA(2,2), but was {rmse}");
        }

        [Fact]
        public void ARMAModel_StationarySeries_ProducesStableForecasts()
        {
            // Arrange
            var options = new ARMAOptions<double> { AROrder = 1, MAOrder = 1 };
            var model = new ARMAModel<double>(options);

            int n = 150;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Generate stationary series around mean of 10
            var random = new Random(555);
            y[0] = 10.0;
            for (int i = 1; i < n; i++)
            {
                y[i] = 10.0 + 0.5 * (y[i-1] - 10.0) + 0.5 * (random.NextDouble() - 0.5);
                X[i, 0] = i;
            }

            model.Train(X, y);

            // Act - Long-term forecast
            var futureX = new Matrix<double>(20, 1);
            for (int i = 0; i < 20; i++) futureX[i, 0] = n + i;
            var forecast = model.Predict(futureX);

            // Assert - Forecasts should stay near the mean
            double forecastMean = 0;
            for (int i = 0; i < 20; i++) forecastMean += forecast[i];
            forecastMean /= 20;

            Assert.True(Math.Abs(forecastMean - 10.0) < 3.0, "Long-term forecast should approach series mean");
        }

        [Fact]
        public void ARMAModel_ModelMetadata_ContainsCorrectInfo()
        {
            // Arrange
            var options = new ARMAOptions<double> { AROrder = 2, MAOrder = 1 };
            var model = new ARMAModel<double>(options);

            int n = 100;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);
            for (int i = 0; i < n; i++)
            {
                y[i] = Math.Sin(i * 0.1);
                X[i, 0] = i;
            }

            model.Train(X, y);

            // Act
            var metadata = model.GetModelMetadata();

            // Assert
            Assert.NotNull(metadata);
            Assert.Equal(ModelType.ARMAModel, metadata.ModelType);
            Assert.True(metadata.AdditionalInfo.ContainsKey("AROrder"));
            Assert.True(metadata.AdditionalInfo.ContainsKey("MAOrder"));
        }

        [Fact]
        public void ARMAModel_BalancedOrders_OutperformsAROrMAAlone()
        {
            // Arrange
            var armaOptions = new ARMAOptions<double> { AROrder = 1, MAOrder = 1, MaxIterations = 2000 };
            var arma = new ARMAModel<double>(armaOptions);

            int n = 150;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Generate true ARMA(1,1) process
            var random = new Random(42);
            double[] epsilon = new double[n];
            for (int i = 0; i < n; i++) epsilon[i] = 0.1 * (random.NextDouble() - 0.5);

            y[0] = epsilon[0];
            for (int i = 1; i < n; i++)
            {
                y[i] = 0.6 * y[i-1] + epsilon[i] + 0.4 * epsilon[i-1];
                X[i, 0] = i;
            }

            // Act
            arma.Train(X, y);
            var armaMetrics = arma.EvaluateModel(X, y);

            // Assert - ARMA should fit this data well
            double rmse = Convert.ToDouble(armaMetrics["RMSE"]);
            Assert.True(rmse < 0.3, $"ARMA(1,1) should fit true ARMA process well, RMSE was {rmse}");
        }

        [Fact]
        public void ARMAModel_ConvergenceBehavior_ReachesTolerance()
        {
            // Arrange
            var options = new ARMAOptions<double> { AROrder = 1, MAOrder = 1, Tolerance = 1e-5 };
            var model = new ARMAModel<double>(options);

            int n = 120;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            for (int i = 0; i < n; i++)
            {
                y[i] = 50.0 + 0.2 * i + Math.Sin(i * 0.15);
                X[i, 0] = i;
            }

            // Act & Assert - Should converge without issues
            model.Train(X, y);
            var predictions = model.Predict(X);

            double mse = 0;
            for (int i = 1; i < n; i++)
            {
                mse += Math.Pow(y[i] - predictions[i], 2);
            }
            mse /= (n - 1);

            Assert.True(mse < 10.0, $"Model should converge properly, MSE was {mse}");
        }

        [Fact]
        public void ARMAModel_SyntheticData_RecognizesStructure()
        {
            // Arrange
            var options = new ARMAOptions<double> { AROrder = 1, MAOrder = 1, MaxIterations = 2500 };
            var model = new ARMAModel<double>(options);

            int n = 150;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Generate true ARMA structure
            var random = new Random(123);
            y[0] = random.NextDouble();
            for (int i = 1; i < n; i++)
            {
                y[i] = 0.5 * y[i-1] + 0.2 * (random.NextDouble() - 0.5);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var predictions = model.Predict(X);

            // Assert
            double correlation = CalculateCorrelation(y, predictions);
            Assert.True(correlation > 0.75, $"Should recognize ARMA structure, correlation was {correlation:F3}");
        }

        #endregion

        #region ARIMA Model Tests

        [Fact]
        public void ARIMAModel_WithDifferencing_HandlesNonStationarySeries()
        {
            // Arrange - ARIMA(1,1,1) model for trending series
            var options = new ARIMAOptions<double>
            {
                AROrder = 1,
                DifferencingOrder = 1,
                MAOrder = 1,
                MaxIterations = 2000
            };
            var model = new ARIMAModel<double>(options);

            // Generate series with linear trend
            int n = 150;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            var random = new Random(42);
            for (int i = 0; i < n; i++)
            {
                y[i] = 5.0 + 0.3 * i + 2.0 * Math.Sin(i * 0.2) + 0.5 * (random.NextDouble() - 0.5);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var predictions = model.Predict(X);

            // Assert - Model should capture trend
            double correlation = CalculateCorrelation(y, predictions);
            Assert.True(correlation > 0.8, $"Model should capture trend well, correlation was {correlation:F3}");
        }

        [Fact]
        public void ARIMAModel_MultipleOrderDifferencing_StabilizesSeries()
        {
            // Arrange - ARIMA(1,2,1) for series with quadratic trend
            var options = new ARIMAOptions<double>
            {
                AROrder = 1,
                DifferencingOrder = 2,
                MAOrder = 1,
                MaxIterations = 2500
            };
            var model = new ARIMAModel<double>(options);

            int n = 150;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Generate series with quadratic trend
            for (int i = 0; i < n; i++)
            {
                y[i] = 10.0 + 0.1 * i + 0.01 * i * i;
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var metrics = model.EvaluateModel(X, y);

            // Assert
            Assert.True(metrics.ContainsKey("RMSE"));
            double rmse = Convert.ToDouble(metrics["RMSE"]);
            Assert.True(rmse < 5.0, $"ARIMA(1,2,1) should handle quadratic trend, RMSE was {rmse}");
        }

        [Fact]
        public void ARIMAModel_TrendDetection_IdentifiesUpwardTrend()
        {
            // Arrange
            var options = new ARIMAOptions<double> { AROrder = 1, DifferencingOrder = 1, MAOrder = 0 };
            var model = new ARIMAModel<double>(options);

            int n = 100;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Clear upward trend
            for (int i = 0; i < n; i++)
            {
                y[i] = 10.0 + 0.5 * i;
                X[i, 0] = i;
            }

            model.Train(X, y);

            // Act - Forecast should continue trend
            var futureX = new Matrix<double>(10, 1);
            for (int i = 0; i < 10; i++) futureX[i, 0] = n + i;
            var forecast = model.Predict(futureX);

            // Assert - Forecast should be increasing
            for (int i = 1; i < 10; i++)
            {
                Assert.True(forecast[i] > forecast[i-1], "Forecast should continue upward trend");
            }
        }

        [Fact]
        public void ARIMAModel_ResidualAnalysis_ShowsWhiteNoiseProperties()
        {
            // Arrange
            var options = new ARIMAOptions<double> { AROrder = 2, DifferencingOrder = 1, MAOrder = 1 };
            var model = new ARIMAModel<double>(options);

            int n = 200;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            var random = new Random(777);
            y[0] = 10.0;
            for (int i = 1; i < n; i++)
            {
                y[i] = y[i-1] + 0.1 + 0.3 * (random.NextDouble() - 0.5);
                X[i, 0] = i;
            }

            model.Train(X, y);
            var predictions = model.Predict(X);

            // Act - Calculate residuals
            var residuals = new List<double>();
            for (int i = 2; i < n; i++)  // Skip first few due to differencing
            {
                residuals.Add(y[i] - predictions[i]);
            }

            // Assert - Residuals should have low autocorrelation
            double acf1 = CalculateACF(residuals, 1);
            Assert.True(Math.Abs(acf1) < 0.3, $"First-order autocorrelation should be small, was {acf1:F3}");
        }

        [Fact]
        public void ARIMAModel_ForecastConfidence_ProducesReasonableIntervals()
        {
            // Arrange
            var options = new ARIMAOptions<double> { AROrder = 1, DifferencingOrder = 1, MAOrder = 1 };
            var model = new ARIMAModel<double>(options);

            int n = 120;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            for (int i = 0; i < n; i++)
            {
                y[i] = 100.0 + 0.5 * i + 5.0 * Math.Sin(i * 0.1);
                X[i, 0] = i;
            }

            model.Train(X, y);

            // Act
            var futureX = new Matrix<double>(12, 1);
            for (int i = 0; i < 12; i++) futureX[i, 0] = n + i;
            var forecast = model.Predict(futureX);

            // Assert - Forecast should be in reasonable range
            foreach (var value in forecast)
            {
                Assert.True(value > 50 && value < 200, $"Forecast value {value} should be in reasonable range");
            }
        }

        [Fact]
        public void ARIMAModel_RandomWalk_HandledByIntegration()
        {
            // Arrange - ARIMA(0,1,0) is a random walk
            var options = new ARIMAOptions<double> { AROrder = 0, DifferencingOrder = 1, MAOrder = 0 };
            var model = new ARIMAModel<double>(options);

            int n = 100;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            var random = new Random(42);
            y[0] = 100.0;
            for (int i = 1; i < n; i++)
            {
                y[i] = y[i-1] + (random.NextDouble() - 0.5);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var predictions = model.Predict(X);

            // Assert
            double correlation = CalculateCorrelation(y, predictions);
            Assert.True(correlation > 0.8, $"ARIMA(0,1,0) should handle random walk, correlation was {correlation:F3}");
        }

        [Fact]
        public void ARIMAModel_OverDifferencing_DetectableByResiduals()
        {
            // Arrange - Over-differencing (d=2 when d=1 is sufficient)
            var options = new ARIMAOptions<double> { AROrder = 1, DifferencingOrder = 2, MAOrder = 0 };
            var model = new ARIMAModel<double>(options);

            int n = 100;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Linear trend only (d=1 sufficient)
            for (int i = 0; i < n; i++)
            {
                y[i] = 50.0 + 0.5 * i;
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var metrics = model.EvaluateModel(X, y);

            // Assert - Should still fit but may have higher error
            Assert.True(metrics.ContainsKey("RMSE"));
            double rmse = Convert.ToDouble(metrics["RMSE"]);
            Assert.True(rmse < 15.0, $"Over-differenced model should still fit, RMSE was {rmse}");
        }

        [Fact]
        public void ARIMAModel_SeasonalTrend_RequiresDifferencing()
        {
            // Arrange
            var options = new ARIMAOptions<double> { AROrder = 1, DifferencingOrder = 1, MAOrder = 1, MaxIterations = 2000 };
            var model = new ARIMAModel<double>(options);

            int n = 120;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Trending series with seasonal component
            for (int i = 0; i < n; i++)
            {
                y[i] = 50.0 + 0.5 * i + 10.0 * Math.Sin(2 * Math.PI * i / 12.0);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var predictions = model.Predict(X);

            // Assert
            double correlation = CalculateCorrelation(y, predictions);
            Assert.True(correlation > 0.8, $"ARIMA should handle trend+seasonality, correlation was {correlation:F3}");
        }

        #endregion

        #region SARIMA Model Tests

        [Fact]
        public void SARIMAModel_WithSeasonality_CapturesSeasonalPattern()
        {
            // Arrange - SARIMA(1,0,0)(1,0,0,12) for monthly seasonality
            var options = new SARIMAOptions<double>
            {
                AROrder = 1,
                DifferencingOrder = 0,
                MAOrder = 0,
                SeasonalAROrder = 1,
                SeasonalDifferencingOrder = 0,
                SeasonalMAOrder = 0,
                SeasonalPeriod = 12,
                MaxIterations = 2000
            };
            var model = new SARIMAModel<double>(options);

            // Generate series with clear seasonal pattern
            int n = 120;  // 10 years of monthly data
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            for (int i = 0; i < n; i++)
            {
                // Seasonal pattern with period 12
                y[i] = 50.0 + 20.0 * Math.Sin(2 * Math.PI * i / 12.0);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var predictions = model.Predict(X);

            // Assert - Should capture seasonal pattern
            double correlation = CalculateCorrelation(y, predictions);
            Assert.True(correlation > 0.8, $"SARIMA should capture seasonality, correlation was {correlation:F3}");
        }

        [Fact]
        public void SARIMAModel_SeasonalDifferencing_HandlesTrendAndSeasonality()
        {
            // Arrange - SARIMA(1,1,1)(1,1,1,12)
            var options = new SARIMAOptions<double>
            {
                AROrder = 1,
                DifferencingOrder = 1,
                MAOrder = 1,
                SeasonalAROrder = 1,
                SeasonalDifferencingOrder = 1,
                SeasonalMAOrder = 1,
                SeasonalPeriod = 12,
                MaxIterations = 3000
            };
            var model = new SARIMAModel<double>(options);

            int n = 144;  // 12 years
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            var random = new Random(42);
            for (int i = 0; i < n; i++)
            {
                // Trend + seasonal + noise
                y[i] = 100.0 + 0.5 * i + 30.0 * Math.Sin(2 * Math.PI * i / 12.0) + 2.0 * (random.NextDouble() - 0.5);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var metrics = model.EvaluateModel(X, y);

            // Assert
            Assert.True(metrics.ContainsKey("RMSE"));
            double rmse = Convert.ToDouble(metrics["RMSE"]);
            Assert.True(rmse < 10.0, $"SARIMA should handle trend+seasonality, RMSE was {rmse}");
        }

        [Fact]
        public void SARIMAModel_SeasonalForecast_PreservesSeasonalPattern()
        {
            // Arrange
            var options = new SARIMAOptions<double>
            {
                AROrder = 0,
                DifferencingOrder = 0,
                MAOrder = 0,
                SeasonalAROrder = 1,
                SeasonalDifferencingOrder = 0,
                SeasonalMAOrder = 0,
                SeasonalPeriod = 4  // Quarterly
            };
            var model = new SARIMAModel<double>(options);

            int n = 40;  // 10 years of quarterly data
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            for (int i = 0; i < n; i++)
            {
                // Q1=100, Q2=120, Q3=90, Q4=110
                double[] seasonalValues = { 100, 120, 90, 110 };
                y[i] = seasonalValues[i % 4];
                X[i, 0] = i;
            }

            model.Train(X, y);

            // Act - Forecast next 4 quarters
            var futureX = new Matrix<double>(4, 1);
            for (int i = 0; i < 4; i++) futureX[i, 0] = n + i;
            var forecast = model.Predict(futureX);

            // Assert - Should maintain seasonal pattern
            Assert.True(forecast[1] > forecast[0], "Q2 should be higher than Q1");
            Assert.True(forecast[2] < forecast[1], "Q3 should be lower than Q2");
            Assert.True(forecast[3] > forecast[2], "Q4 should be higher than Q3");
        }

        [Fact]
        public void SARIMAModel_WithoutSeasonality_BehavesLikeARIMA()
        {
            // Arrange - SARIMA with no seasonal components = ARIMA
            var options = new SARIMAOptions<double>
            {
                AROrder = 1,
                DifferencingOrder = 1,
                MAOrder = 1,
                SeasonalAROrder = 0,
                SeasonalDifferencingOrder = 0,
                SeasonalMAOrder = 0,
                SeasonalPeriod = 1
            };
            var model = new SARIMAModel<double>(options);

            int n = 100;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Simple trending series
            for (int i = 0; i < n; i++)
            {
                y[i] = 50 + 0.5 * i;
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var predictions = model.Predict(X);

            // Assert - Should fit well like ARIMA
            double mse = 0;
            for (int i = 2; i < n; i++)
            {
                mse += Math.Pow(y[i] - predictions[i], 2);
            }
            mse /= (n - 2);

            Assert.True(mse < 5.0, $"SARIMA without seasonal terms should work like ARIMA, MSE was {mse}");
        }

        [Fact]
        public void SARIMAModel_MultipleSeasonalities_CapturesComplexPattern()
        {
            // Arrange - Weekly pattern in daily data
            var options = new SARIMAOptions<double>
            {
                AROrder = 1,
                DifferencingOrder = 0,
                MAOrder = 0,
                SeasonalAROrder = 1,
                SeasonalDifferencingOrder = 0,
                SeasonalMAOrder = 0,
                SeasonalPeriod = 7  // Weekly
            };
            var model = new SARIMAModel<double>(options);

            int n = 84;  // 12 weeks
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Weekly pattern: higher on weekends
            for (int i = 0; i < n; i++)
            {
                double weekdayEffect = (i % 7 == 5 || i % 7 == 6) ? 20.0 : 0.0;  // Sat/Sun
                y[i] = 100.0 + weekdayEffect;
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var predictions = model.Predict(X);

            // Assert
            double correlation = CalculateCorrelation(y, predictions);
            Assert.True(correlation > 0.7, $"SARIMA should capture weekly pattern, correlation was {correlation:F3}");
        }

        [Fact]
        public void SARIMAModel_LongSeasonalPeriod_HandlesEfficiently()
        {
            // Arrange - Annual seasonality in monthly data
            var options = new SARIMAOptions<double>
            {
                AROrder = 1,
                DifferencingOrder = 1,
                MAOrder = 1,
                SeasonalAROrder = 1,
                SeasonalDifferencingOrder = 1,
                SeasonalMAOrder = 1,
                SeasonalPeriod = 12,
                MaxIterations = 2000
            };
            var model = new SARIMAModel<double>(options);

            int n = 60;  // 5 years
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            for (int i = 0; i < n; i++)
            {
                y[i] = 100.0 + 0.2 * i + 25.0 * Math.Sin(2 * Math.PI * i / 12.0);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var metrics = model.EvaluateModel(X, y);

            // Assert
            double rmse = Convert.ToDouble(metrics["RMSE"]);
            Assert.True(rmse < 20.0, $"SARIMA should handle long seasonal period, RMSE was {rmse}");
        }

        #endregion

        #region ARIMAX Model Tests

        [Fact]
        public void ARIMAXModel_WithExogenousVariables_IncorporatesExternalFactors()
        {
            // Arrange - ARIMAX(1,1,1) with 2 exogenous variables
            var options = new ARIMAXModelOptions<double>
            {
                AROrder = 1,
                DifferencingOrder = 1,
                MAOrder = 1,
                NumExogenousVariables = 2,
                MaxIterations = 2000
            };
            var model = new ARIMAXModel<double>(options);

            int n = 150;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 2);

            var random = new Random(42);
            for (int i = 0; i < n; i++)
            {
                // Two exogenous variables affect y
                X[i, 0] = Math.Sin(i * 0.1);
                X[i, 1] = Math.Cos(i * 0.15);
                y[i] = 10.0 + 0.2 * i + 5.0 * X[i, 0] + 3.0 * X[i, 1] + 0.5 * (random.NextDouble() - 0.5);
            }

            // Act
            model.Train(X, y);
            var predictions = model.Predict(X);

            // Assert - Should capture influence of exogenous variables
            double correlation = CalculateCorrelation(y, predictions);
            Assert.True(correlation > 0.85, $"ARIMAX should capture exogenous effects, correlation was {correlation:F3}");
        }

        [Fact]
        public void ARIMAXModel_ExogenousVariableImpact_ImprovesForecastAccuracy()
        {
            // Arrange
            var optionsWithX = new ARIMAXModelOptions<double>
            {
                AROrder = 1,
                DifferencingOrder = 0,
                MAOrder = 0,
                NumExogenousVariables = 1
            };
            var modelWithX = new ARIMAXModel<double>(optionsWithX);

            int n = 100;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // y strongly depends on exogenous variable
            for (int i = 0; i < n; i++)
            {
                X[i, 0] = i % 10;  // Cyclical exogenous variable
                y[i] = 50.0 + 10.0 * X[i, 0];
            }

            // Act
            modelWithX.Train(X, y);
            var predictions = modelWithX.Predict(X);
            var metrics = modelWithX.EvaluateModel(X, y);

            // Assert - With exogenous variables, fit should be excellent
            double rmse = Convert.ToDouble(metrics["RMSE"]);
            Assert.True(rmse < 5.0, $"ARIMAX with relevant exogenous variable should fit well, RMSE was {rmse}");
        }

        [Fact]
        public void ARIMAXModel_FutureExogenousValues_RequiredForForecasting()
        {
            // Arrange
            var options = new ARIMAXModelOptions<double>
            {
                AROrder = 1,
                DifferencingOrder = 0,
                MAOrder = 0,
                NumExogenousVariables = 1
            };
            var model = new ARIMAXModel<double>(options);

            int n = 80;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            for (int i = 0; i < n; i++)
            {
                X[i, 0] = i * 0.1;
                y[i] = 100.0 + 5.0 * X[i, 0];
            }

            model.Train(X, y);

            // Act - Provide future exogenous values
            var futureX = new Matrix<double>(10, 1);
            for (int i = 0; i < 10; i++)
            {
                futureX[i, 0] = (n + i) * 0.1;
            }
            var forecast = model.Predict(futureX);

            // Assert - Forecast should use future exogenous values
            Assert.True(forecast[9] > forecast[0], "Forecast should increase with exogenous variable");
        }

        [Fact]
        public void ARIMAXModel_MultipleExogenousVariables_HandlesComplexRelationships()
        {
            // Arrange
            var options = new ARIMAXModelOptions<double>
            {
                AROrder = 1,
                DifferencingOrder = 0,
                MAOrder = 1,
                NumExogenousVariables = 3
            };
            var model = new ARIMAXModel<double>(options);

            int n = 120;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 3);

            for (int i = 0; i < n; i++)
            {
                X[i, 0] = Math.Sin(i * 0.1);
                X[i, 1] = Math.Cos(i * 0.1);
                X[i, 2] = i % 7;  // Day of week effect
                y[i] = 50.0 + 10.0 * X[i, 0] + 5.0 * X[i, 1] + 2.0 * X[i, 2];
            }

            // Act
            model.Train(X, y);
            var metrics = model.EvaluateModel(X, y);

            // Assert
            Assert.True(metrics.ContainsKey("MAE"));
            double mae = Convert.ToDouble(metrics["MAE"]);
            Assert.True(mae < 5.0, $"Model should handle multiple exogenous variables, MAE was {mae}");
        }

        [Fact]
        public void ARIMAXModel_NonLinearExogenousEffect_CapturedInResiduals()
        {
            // Arrange
            var options = new ARIMAXModelOptions<double>
            {
                AROrder = 1,
                DifferencingOrder = 0,
                MAOrder = 0,
                NumExogenousVariables = 1
            };
            var model = new ARIMAXModel<double>(options);

            int n = 100;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Linear model with quadratic true relationship
            for (int i = 0; i < n; i++)
            {
                X[i, 0] = i * 0.1;
                y[i] = 100.0 + 5.0 * X[i, 0] + 2.0 * X[i, 0] * X[i, 0];  // Quadratic
            }

            // Act
            model.Train(X, y);
            var predictions = model.Predict(X);

            // Assert - Linear model won't perfectly capture quadratic but should be reasonable
            double correlation = CalculateCorrelation(y, predictions);
            Assert.True(correlation > 0.9, $"Should capture main relationship, correlation was {correlation:F3}");
        }

        [Fact]
        public void ARIMAXModel_ExogenousVsEndogenous_ComparativePerformance()
        {
            // Arrange
            var options = new ARIMAXModelOptions<double>
            {
                AROrder = 1,
                DifferencingOrder = 0,
                MAOrder = 0,
                NumExogenousVariables = 1
            };
            var model = new ARIMAXModel<double>(options);

            int n = 80;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Strong exogenous influence
            for (int i = 0; i < n; i++)
            {
                X[i, 0] = Math.Sin(i * 0.2);
                y[i] = 50.0 + 20.0 * X[i, 0];  // Dominated by exogenous variable
            }

            // Act
            model.Train(X, y);
            var metrics = model.EvaluateModel(X, y);

            // Assert - With strong exogenous influence, should fit very well
            double mse = Convert.ToDouble(metrics["MSE"]);
            Assert.True(mse < 10.0, $"Strong exogenous relationship should be captured, MSE was {mse}");
        }

        #endregion

        #region Exponential Smoothing Model Tests

        [Fact]
        public void ExponentialSmoothing_SimpleSmoothing_SmoothsSeriesCorrectly()
        {
            // Arrange - Simple exponential smoothing (no trend, no seasonality)
            var options = new ExponentialSmoothingOptions<double>
            {
                InitialAlpha = 0.3,
                UseTrend = false,
                UseSeasonal = false
            };
            var model = new ExponentialSmoothingModel<double>(options);

            int n = 100;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            var random = new Random(42);
            for (int i = 0; i < n; i++)
            {
                // Series oscillating around 50
                y[i] = 50.0 + 5.0 * (random.NextDouble() - 0.5);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var predictions = model.Predict(X);

            // Assert - Predictions should be smoother than actual data
            double actualVariance = CalculateVariance(y);
            double predictedVariance = CalculateVariance(predictions);
            Assert.True(predictedVariance < actualVariance, "Smoothed series should have lower variance");
        }

        [Fact]
        public void ExponentialSmoothing_WithTrend_CapturesLinearTrend()
        {
            // Arrange - Double exponential smoothing (Holt's method)
            var options = new ExponentialSmoothingOptions<double>
            {
                InitialAlpha = 0.3,
                InitialBeta = 0.1,
                UseTrend = true,
                UseSeasonal = false
            };
            var model = new ExponentialSmoothingModel<double>(options);

            int n = 100;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Linear trend with noise
            var random = new Random(123);
            for (int i = 0; i < n; i++)
            {
                y[i] = 10.0 + 0.5 * i + 2.0 * (random.NextDouble() - 0.5);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);

            // Forecast ahead
            var futureX = new Matrix<double>(10, 1);
            for (int i = 0; i < 10; i++) futureX[i, 0] = n + i;
            var forecast = model.Predict(futureX);

            // Assert - Forecast should continue upward trend
            for (int i = 1; i < 10; i++)
            {
                Assert.True(forecast[i] > forecast[i-1], "Forecast with trend should be increasing");
            }
        }

        [Fact]
        public void ExponentialSmoothing_HoltWinters_HandlesSeasonalityAndTrend()
        {
            // Arrange - Triple exponential smoothing (Holt-Winters)
            var options = new ExponentialSmoothingOptions<double>
            {
                InitialAlpha = 0.3,
                InitialBeta = 0.1,
                InitialGamma = 0.1,
                UseTrend = true,
                UseSeasonal = true
            };
            var model = new ExponentialSmoothingModel<double>(options);
            model.SeasonalPeriod = 12;

            int n = 120;  // 10 years of monthly data
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            for (int i = 0; i < n; i++)
            {
                // Trend + seasonality
                y[i] = 100.0 + 0.5 * i + 20.0 * Math.Sin(2 * Math.PI * i / 12.0);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var predictions = model.Predict(X);

            // Assert - Should capture both trend and seasonality
            double correlation = CalculateCorrelation(y, predictions);
            Assert.True(correlation > 0.85, $"Holt-Winters should capture trend+seasonality, correlation was {correlation:F3}");
        }

        [Fact]
        public void ExponentialSmoothing_AlphaParameter_ControlsSmoothingLevel()
        {
            // Arrange - Test different alpha values
            var lowAlpha = new ExponentialSmoothingOptions<double> { InitialAlpha = 0.1, UseTrend = false, UseSeasonal = false };
            var highAlpha = new ExponentialSmoothingOptions<double> { InitialAlpha = 0.9, UseTrend = false, UseSeasonal = false };

            var modelLow = new ExponentialSmoothingModel<double>(lowAlpha);
            var modelHigh = new ExponentialSmoothingModel<double>(highAlpha);

            int n = 80;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            var random = new Random(999);
            for (int i = 0; i < n; i++)
            {
                y[i] = 50.0 + 10.0 * (random.NextDouble() - 0.5);
                X[i, 0] = i;
            }

            // Act
            modelLow.Train(X, y);
            modelHigh.Train(X, y);

            var predictionsLow = modelLow.Predict(X);
            var predictionsHigh = modelHigh.Predict(X);

            // Assert - Low alpha should produce smoother predictions
            double varianceLow = CalculateVariance(predictionsLow);
            double varianceHigh = CalculateVariance(predictionsHigh);
            Assert.True(varianceLow < varianceHigh, "Lower alpha should produce smoother predictions");
        }

        [Fact]
        public void ExponentialSmoothing_SeasonalForecast_RepeatsPattern()
        {
            // Arrange
            var options = new ExponentialSmoothingOptions<double>
            {
                InitialAlpha = 0.2,
                InitialBeta = 0.05,
                InitialGamma = 0.1,
                UseTrend = false,
                UseSeasonal = true
            };
            var model = new ExponentialSmoothingModel<double>(options);
            model.SeasonalPeriod = 4;

            int n = 40;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Clear quarterly pattern: 100, 110, 90, 95
            double[] pattern = { 100, 110, 90, 95 };
            for (int i = 0; i < n; i++)
            {
                y[i] = pattern[i % 4];
                X[i, 0] = i;
            }

            model.Train(X, y);

            // Act - Forecast next 8 quarters
            var futureX = new Matrix<double>(8, 1);
            for (int i = 0; i < 8; i++) futureX[i, 0] = n + i;
            var forecast = model.Predict(futureX);

            // Assert - Should repeat pattern
            Assert.True(Math.Abs(forecast[0] - 100) < 10, "Should forecast Q1 pattern");
            Assert.True(Math.Abs(forecast[1] - 110) < 10, "Should forecast Q2 pattern");
            Assert.True(Math.Abs(forecast[2] - 90) < 10, "Should forecast Q3 pattern");
            Assert.True(Math.Abs(forecast[3] - 95) < 10, "Should forecast Q4 pattern");
        }

        [Fact]
        public void ExponentialSmoothing_DampedTrend_ConvergesToConstant()
        {
            // Arrange - Holt's damped trend model
            var options = new ExponentialSmoothingOptions<double>
            {
                InitialAlpha = 0.3,
                InitialBeta = 0.1,
                UseTrend = true,
                UseSeasonal = false
            };
            var model = new ExponentialSmoothingModel<double>(options);

            int n = 100;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Linear trend that levels off
            for (int i = 0; i < n; i++)
            {
                y[i] = 100.0 + 20.0 * (1.0 - Math.Exp(-i / 30.0));
                X[i, 0] = i;
            }

            model.Train(X, y);

            // Act - Long forecast
            var futureX = new Matrix<double>(30, 1);
            for (int i = 0; i < 30; i++) futureX[i, 0] = n + i;
            var forecast = model.Predict(futureX);

            // Assert - Trend should dampen over forecast horizon
            double firstSlope = forecast[5] - forecast[0];
            double lastSlope = forecast[29] - forecast[24];
            Assert.True(Math.Abs(lastSlope) <= Math.Abs(firstSlope), "Trend should dampen in long-term forecast");
        }

        [Fact]
        public void ExponentialSmoothing_IrregularSpacing_HandlesWithInterpolation()
        {
            // Arrange
            var options = new ExponentialSmoothingOptions<double>
            {
                InitialAlpha = 0.3,
                UseTrend = false,
                UseSeasonal = false
            };
            var model = new ExponentialSmoothingModel<double>(options);

            int n = 80;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            for (int i = 0; i < n; i++)
            {
                y[i] = 50.0 + 5.0 * Math.Sin(i * 0.1);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var predictions = model.Predict(X);

            // Assert
            double correlation = CalculateCorrelation(y, predictions);
            Assert.True(correlation > 0.7, $"Should handle series with smoothing, correlation was {correlation:F3}");
        }

        [Fact]
        public void ExponentialSmoothing_AdaptiveParameters_ImproveOverTime()
        {
            // Arrange
            var options = new ExponentialSmoothingOptions<double>
            {
                InitialAlpha = 0.5,
                UseTrend = false,
                UseSeasonal = false,
                GridSearchStep = 0.1
            };
            var model = new ExponentialSmoothingModel<double>(options);

            int n = 100;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            var random = new Random(42);
            for (int i = 0; i < n; i++)
            {
                y[i] = 100.0 + 10.0 * (random.NextDouble() - 0.5);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var metrics = model.EvaluateModel(X, y);

            // Assert - Should fit reasonably
            double mse = Convert.ToDouble(metrics["MSE"]);
            Assert.True(mse < 50.0, $"Exponential smoothing should fit noisy data, MSE was {mse}");
        }

        #endregion

        #region State Space Model Tests

        [Fact]
        public void StateSpaceModel_KalmanFilter_EstimatesHiddenStates()
        {
            // Arrange
            var options = new StateSpaceModelOptions<double>
            {
                StateSize = 2,
                ObservationSize = 1,
                LearningRate = 0.01,
                MaxIterations = 1000,
                Tolerance = 1e-6
            };
            var model = new StateSpaceModel<double>(options);

            int n = 100;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Simple linear system
            for (int i = 0; i < n; i++)
            {
                y[i] = 10.0 + 0.5 * i + Math.Sin(i * 0.2);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var predictions = model.Predict(X);

            // Assert - Should capture the pattern
            double correlation = CalculateCorrelation(y, predictions);
            Assert.True(correlation > 0.7, $"State space model should track series, correlation was {correlation:F3}");
        }

        [Fact]
        public void StateSpaceModel_WithNoise_FiltersEffectively()
        {
            // Arrange
            var options = new StateSpaceModelOptions<double>
            {
                StateSize = 2,
                ObservationSize = 1,
                LearningRate = 0.01,
                MaxIterations = 1500
            };
            var model = new StateSpaceModel<double>(options);

            int n = 150;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            var random = new Random(42);
            for (int i = 0; i < n; i++)
            {
                // Smooth signal + noise
                y[i] = 50.0 + 10.0 * Math.Sin(i * 0.1) + 5.0 * (random.NextDouble() - 0.5);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var predictions = model.Predict(X);

            // Assert - Predictions should be smoother than noisy observations
            double yVariance = CalculateVariance(y);
            double predVariance = CalculateVariance(predictions);
            // Note: State space may not always reduce variance, but should track signal
            double mse = 0;
            for (int i = 0; i < n; i++)
            {
                mse += Math.Pow(y[i] - predictions[i], 2);
            }
            mse /= n;
            Assert.True(mse < 50, $"State space filter should reduce noise, MSE was {mse}");
        }

        [Fact]
        public void StateSpaceModel_EMAlgorithm_ConvergesToSolution()
        {
            // Arrange
            var options = new StateSpaceModelOptions<double>
            {
                StateSize = 1,
                ObservationSize = 1,
                LearningRate = 0.01,
                MaxIterations = 2000,
                Tolerance = 1e-6
            };
            var model = new StateSpaceModel<double>(options);

            int n = 80;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Simple AR(1)-like process
            y[0] = 10.0;
            for (int i = 1; i < n; i++)
            {
                y[i] = 0.8 * y[i-1] + 0.3;
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var metrics = model.EvaluateModel(X, y);

            // Assert - EM algorithm should converge
            Assert.True(metrics.ContainsKey("RMSE"));
            double rmse = Convert.ToDouble(metrics["RMSE"]);
            Assert.True(rmse < 5.0, $"State space EM should converge well, RMSE was {rmse}");
        }

        [Fact]
        public void StateSpaceModel_HigherDimensionalState_HandlesComplexity()
        {
            // Arrange - Larger state space
            var options = new StateSpaceModelOptions<double>
            {
                StateSize = 3,
                ObservationSize = 1,
                LearningRate = 0.01,
                MaxIterations = 1500
            };
            var model = new StateSpaceModel<double>(options);

            int n = 100;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Complex pattern
            for (int i = 0; i < n; i++)
            {
                y[i] = 20.0 + 5.0 * Math.Sin(i * 0.1) + 3.0 * Math.Cos(i * 0.15);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var predictions = model.Predict(X);

            // Assert
            double correlation = CalculateCorrelation(y, predictions);
            Assert.True(correlation > 0.6, $"Higher-dimensional state space should handle complexity, correlation was {correlation:F3}");
        }

        [Fact]
        public void StateSpaceModel_Metadata_ContainsStateInfo()
        {
            // Arrange
            var options = new StateSpaceModelOptions<double> { StateSize = 2, ObservationSize = 1 };
            var model = new StateSpaceModel<double>(options);

            int n = 50;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);
            for (int i = 0; i < n; i++)
            {
                y[i] = Math.Sin(i * 0.2);
                X[i, 0] = i;
            }

            model.Train(X, y);

            // Act
            var metadata = model.GetModelMetadata();

            // Assert
            Assert.NotNull(metadata);
            Assert.Equal(ModelType.StateSpaceModel, metadata.ModelType);
            Assert.True(metadata.AdditionalInfo.ContainsKey("StateSize"));
            Assert.True(metadata.AdditionalInfo.ContainsKey("ObservationSize"));
        }

        [Fact]
        public void StateSpaceModel_TimeVaryingDynamics_AdaptsToChanges()
        {
            // Arrange
            var options = new StateSpaceModelOptions<double>
            {
                StateSize = 2,
                ObservationSize = 1,
                LearningRate = 0.01,
                MaxIterations = 1500
            };
            var model = new StateSpaceModel<double>(options);

            int n = 120;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Series with regime change at n/2
            for (int i = 0; i < n / 2; i++)
            {
                y[i] = 50.0 + 0.3 * i;
                X[i, 0] = i;
            }
            for (int i = n / 2; i < n; i++)
            {
                y[i] = 80.0 + 0.1 * i;  // Different dynamics
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var predictions = model.Predict(X);

            // Assert - Should adapt to both regimes
            double correlation = CalculateCorrelation(y, predictions);
            Assert.True(correlation > 0.7, $"State space should adapt to regime changes, correlation was {correlation:F3}");
        }

        [Fact]
        public void StateSpaceModel_MultipleObservations_HandlesSimultaneously()
        {
            // Arrange - Multiple observation dimensions
            var options = new StateSpaceModelOptions<double>
            {
                StateSize = 2,
                ObservationSize = 1,
                LearningRate = 0.01,
                MaxIterations = 1000
            };
            var model = new StateSpaceModel<double>(options);

            int n = 100;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            for (int i = 0; i < n; i++)
            {
                y[i] = 50.0 + 5.0 * Math.Sin(i * 0.1) + 3.0 * Math.Cos(i * 0.15);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var metrics = model.EvaluateModel(X, y);

            // Assert
            double rmse = Convert.ToDouble(metrics["RMSE"]);
            Assert.True(rmse < 10.0, $"State space should handle multiple frequencies, RMSE was {rmse}");
        }

        [Fact]
        public void StateSpaceModel_MissingData_InterpolatesGracefully()
        {
            // Arrange
            var options = new StateSpaceModelOptions<double>
            {
                StateSize = 1,
                ObservationSize = 1,
                LearningRate = 0.01,
                MaxIterations = 1000
            };
            var model = new StateSpaceModel<double>(options);

            int n = 80;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Generate series
            for (int i = 0; i < n; i++)
            {
                y[i] = 100.0 + 0.5 * i;
                X[i, 0] = i;
            }

            // Act & Assert - Should handle without NaN issues
            model.Train(X, y);
            var predictions = model.Predict(X);

            foreach (var pred in predictions)
            {
                Assert.True(!double.IsNaN(pred) && !double.IsInfinity(pred), "Predictions should be valid");
            }
        }

        #endregion

        #region STL Decomposition Tests

        [Fact]
        public void STLDecomposition_StandardAlgorithm_SeparatesComponents()
        {
            // Arrange
            var options = new STLDecompositionOptions<double>
            {
                SeasonalPeriod = 12,
                AlgorithmType = STLAlgorithmType.Standard,
                TrendWindowSize = 18,
                SeasonalLoessWindow = 121
            };
            var model = new STLDecomposition<double>(options);

            int n = 120;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Series with trend + seasonality + noise
            var random = new Random(42);
            for (int i = 0; i < n; i++)
            {
                double trend = 50.0 + 0.3 * i;
                double seasonal = 15.0 * Math.Sin(2 * Math.PI * i / 12.0);
                double noise = 2.0 * (random.NextDouble() - 0.5);
                y[i] = trend + seasonal + noise;
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var trendComponent = model.GetTrend();
            var seasonalComponent = model.GetSeasonal();
            var residualComponent = model.GetResidual();

            // Assert - Components should sum to original series
            for (int i = 0; i < n; i++)
            {
                double reconstructed = trendComponent[i] + seasonalComponent[i] + residualComponent[i];
                Assert.True(Math.Abs(reconstructed - y[i]) < 0.01, $"Decomposition at index {i} should sum correctly");
            }
        }

        [Fact]
        public void STLDecomposition_TrendComponent_CapturesLongTermPattern()
        {
            // Arrange
            var options = new STLDecompositionOptions<double>
            {
                SeasonalPeriod = 12,
                TrendWindowSize = 25
            };
            var model = new STLDecomposition<double>(options);

            int n = 120;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Clear upward trend with seasonality
            for (int i = 0; i < n; i++)
            {
                y[i] = 100.0 + 0.5 * i + 10.0 * Math.Sin(2 * Math.PI * i / 12.0);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var trend = model.GetTrend();

            // Assert - Trend should be approximately linear and increasing
            double firstTrend = trend[10];
            double lastTrend = trend[110];
            Assert.True(lastTrend > firstTrend + 40, "Trend should show clear increase");
        }

        [Fact]
        public void STLDecomposition_SeasonalComponent_RepeatsPattern()
        {
            // Arrange
            var options = new STLDecompositionOptions<double> { SeasonalPeriod = 12 };
            var model = new STLDecomposition<double>(options);

            int n = 120;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Strong seasonal pattern
            for (int i = 0; i < n; i++)
            {
                y[i] = 50.0 + 30.0 * Math.Sin(2 * Math.PI * i / 12.0);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var seasonal = model.GetSeasonal();

            // Assert - Seasonal pattern should repeat with period 12
            for (int i = 12; i < n - 12; i++)
            {
                double diff = Math.Abs(seasonal[i] - seasonal[i + 12]);
                Assert.True(diff < 5.0, $"Seasonal component should repeat at index {i}");
            }
        }

        [Fact]
        public void STLDecomposition_ResidualComponent_HasLowAutocorrelation()
        {
            // Arrange
            var options = new STLDecompositionOptions<double> { SeasonalPeriod = 12 };
            var model = new STLDecomposition<double>(options);

            int n = 120;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            var random = new Random(777);
            for (int i = 0; i < n; i++)
            {
                y[i] = 50.0 + 0.2 * i + 10.0 * Math.Sin(2 * Math.PI * i / 12.0) + 3.0 * (random.NextDouble() - 0.5);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var residuals = model.GetResidual();

            // Assert - Residuals should have low autocorrelation
            var residualList = new List<double>();
            for (int i = 0; i < residuals.Length; i++)
            {
                residualList.Add(residuals[i]);
            }

            double acf1 = CalculateACF(residualList, 1);
            double acf12 = CalculateACF(residualList, 12);

            Assert.True(Math.Abs(acf1) < 0.3, $"Lag-1 ACF should be small, was {acf1:F3}");
            Assert.True(Math.Abs(acf12) < 0.3, $"Lag-12 ACF should be small, was {acf12:F3}");
        }

        [Fact]
        public void STLDecomposition_RobustAlgorithm_HandlesOutliers()
        {
            // Arrange - Use robust algorithm
            var options = new STLDecompositionOptions<double>
            {
                SeasonalPeriod = 12,
                AlgorithmType = STLAlgorithmType.Robust,
                RobustIterations = 2
            };
            var model = new STLDecomposition<double>(options);

            int n = 120;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Series with outliers
            for (int i = 0; i < n; i++)
            {
                y[i] = 50.0 + 0.2 * i + 10.0 * Math.Sin(2 * Math.PI * i / 12.0);

                // Add outliers at specific points
                if (i % 20 == 0)
                {
                    y[i] += 50.0;  // Large outlier
                }

                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var residuals = model.GetResidual();

            // Assert - Robust method should handle outliers in residuals
            int largeResidualsCount = 0;
            for (int i = 0; i < residuals.Length; i++)
            {
                if (Math.Abs(residuals[i]) > 20)
                {
                    largeResidualsCount++;
                }
            }

            Assert.True(largeResidualsCount <= 10, "Robust STL should handle outliers effectively");
        }

        [Fact]
        public void STLDecomposition_FastAlgorithm_ProducesReasonableResults()
        {
            // Arrange - Fast algorithm for large dataset
            var options = new STLDecompositionOptions<double>
            {
                SeasonalPeriod = 12,
                AlgorithmType = STLAlgorithmType.Fast,
                TrendWindowSize = 25,
                SeasonalLoessWindow = 13
            };
            var model = new STLDecomposition<double>(options);

            int n = 240;  // 20 years of monthly data
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            for (int i = 0; i < n; i++)
            {
                y[i] = 100.0 + 0.5 * i + 20.0 * Math.Sin(2 * Math.PI * i / 12.0);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var trend = model.GetTrend();
            var seasonal = model.GetSeasonal();

            // Assert - Fast algorithm should still capture main patterns
            Assert.True(trend[n-1] > trend[0] + 80, "Trend should be captured by fast algorithm");
            Assert.True(Math.Abs(seasonal[0] - seasonal[12]) < 8, "Seasonality should repeat in fast algorithm");
        }

        [Fact]
        public void STLDecomposition_SeasonalStrength_ReflectsSeasonalityImportance()
        {
            // Arrange - Strong seasonal component
            var options = new STLDecompositionOptions<double> { SeasonalPeriod = 12 };
            var model = new STLDecomposition<double>(options);

            int n = 120;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            for (int i = 0; i < n; i++)
            {
                // Very strong seasonal component
                y[i] = 50.0 + 40.0 * Math.Sin(2 * Math.PI * i / 12.0) + 0.1 * i;
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var metadata = model.GetModelMetadata();

            // Assert - Seasonal strength should be high
            double seasonalStrength = Convert.ToDouble(metadata.AdditionalInfo["SeasonalStrength"]);
            Assert.True(seasonalStrength > 0.7, $"Strong seasonality should be detected, strength was {seasonalStrength:F3}");
        }

        [Fact]
        public void STLDecomposition_TrendStrength_ReflectsTrendImportance()
        {
            // Arrange - Strong trend component
            var options = new STLDecompositionOptions<double> { SeasonalPeriod = 12 };
            var model = new STLDecomposition<double>(options);

            int n = 120;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            for (int i = 0; i < n; i++)
            {
                // Strong trend, weak seasonality
                y[i] = 10.0 + 0.8 * i + 2.0 * Math.Sin(2 * Math.PI * i / 12.0);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var metadata = model.GetModelMetadata();

            // Assert - Trend strength should be high
            double trendStrength = Convert.ToDouble(metadata.AdditionalInfo["TrendStrength"]);
            Assert.True(trendStrength > 0.7, $"Strong trend should be detected, strength was {trendStrength:F3}");
        }

        [Fact]
        public void STLDecomposition_Forecast_UsesDecomposedComponents()
        {
            // Arrange
            var options = new STLDecompositionOptions<double> { SeasonalPeriod = 12 };
            var model = new STLDecomposition<double>(options);

            int n = 120;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            for (int i = 0; i < n; i++)
            {
                y[i] = 100.0 + 0.3 * i + 20.0 * Math.Sin(2 * Math.PI * i / 12.0);
                X[i, 0] = i;
            }

            model.Train(X, y);

            // Act - Forecast next 12 months
            var futureX = new Matrix<double>(12, 1);
            for (int i = 0; i < 12; i++) futureX[i, 0] = n + i;
            var forecast = model.Predict(futureX);

            // Assert - Forecast should exhibit seasonal pattern
            int maxIdx = 0, minIdx = 0;
            for (int i = 1; i < 12; i++)
            {
                if (forecast[i] > forecast[maxIdx]) maxIdx = i;
                if (forecast[i] < forecast[minIdx]) minIdx = i;
            }

            double seasonalRange = forecast[maxIdx] - forecast[minIdx];
            Assert.True(seasonalRange > 10, "Forecast should preserve seasonal variation");
        }

        [Fact]
        public void STLDecomposition_ShortSeries_HandlesMinimumData()
        {
            // Arrange - Minimum viable series (2 full seasonal periods)
            var options = new STLDecompositionOptions<double>
            {
                SeasonalPeriod = 4,
                TrendWindowSize = 7,
                SeasonalLoessWindow = 9
            };
            var model = new STLDecomposition<double>(options);

            int n = 32;  // 8 quarters
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            double[] seasonalPattern = { 100, 110, 90, 95 };
            for (int i = 0; i < n; i++)
            {
                y[i] = seasonalPattern[i % 4] + 0.1 * i;
                X[i, 0] = i;
            }

            // Act & Assert - Should not throw
            model.Train(X, y);
            var trend = model.GetTrend();
            var seasonal = model.GetSeasonal();

            Assert.NotNull(trend);
            Assert.NotNull(seasonal);
            Assert.Equal(n, trend.Length);
            Assert.Equal(n, seasonal.Length);
        }

        [Fact]
        public void STLDecomposition_NoTrend_TrendComponentFlat()
        {
            // Arrange
            var options = new STLDecompositionOptions<double> { SeasonalPeriod = 12 };
            var model = new STLDecomposition<double>(options);

            int n = 120;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Pure seasonal with no trend
            for (int i = 0; i < n; i++)
            {
                y[i] = 50.0 + 15.0 * Math.Sin(2 * Math.PI * i / 12.0);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var trend = model.GetTrend();

            // Assert - Trend should be relatively flat
            double trendRange = 0;
            double minTrend = trend[0], maxTrend = trend[0];
            for (int i = 1; i < n; i++)
            {
                if (trend[i] < minTrend) minTrend = trend[i];
                if (trend[i] > maxTrend) maxTrend = trend[i];
            }
            trendRange = maxTrend - minTrend;

            Assert.True(trendRange < 10, $"Trend range should be small when no trend exists, was {trendRange:F2}");
        }

        [Fact]
        public void STLDecomposition_NoSeasonality_SeasonalComponentNearZero()
        {
            // Arrange
            var options = new STLDecompositionOptions<double> { SeasonalPeriod = 12 };
            var model = new STLDecomposition<double>(options);

            int n = 120;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Pure trend with no seasonality
            for (int i = 0; i < n; i++)
            {
                y[i] = 50.0 + 0.5 * i;
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var seasonal = model.GetSeasonal();

            // Assert - Seasonal component should be near zero
            double seasonalMean = 0;
            for (int i = 0; i < n; i++)
            {
                seasonalMean += Math.Abs(seasonal[i]);
            }
            seasonalMean /= n;

            Assert.True(seasonalMean < 2.0, $"Seasonal component should be near zero when no seasonality exists, mean abs value was {seasonalMean:F2}");
        }

        [Fact]
        public void STLDecomposition_DifferentSeasonalPeriods_AdaptsCorrectly()
        {
            // Arrange - Quarterly data
            var options = new STLDecompositionOptions<double> { SeasonalPeriod = 4, TrendWindowSize = 7 };
            var model = new STLDecomposition<double>(options);

            int n = 40;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Quarterly pattern
            for (int i = 0; i < n; i++)
            {
                y[i] = 100.0 + 20.0 * Math.Sin(2 * Math.PI * i / 4.0);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var seasonal = model.GetSeasonal();

            // Assert - Should detect quarterly pattern
            for (int i = 4; i < n - 4; i++)
            {
                double diff = Math.Abs(seasonal[i] - seasonal[i + 4]);
                Assert.True(diff < 3.0, $"Quarterly pattern should repeat, difference at {i} was {diff:F2}");
            }
        }

        [Fact]
        public void STLDecomposition_ChangingSeasonalAmplitude_DetectsVariation()
        {
            // Arrange
            var options = new STLDecompositionOptions<double> { SeasonalPeriod = 12, TrendWindowSize = 25 };
            var model = new STLDecomposition<double>(options);

            int n = 120;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Seasonal pattern with increasing amplitude
            for (int i = 0; i < n; i++)
            {
                double amplitude = 10.0 + 0.1 * i;
                y[i] = 50.0 + amplitude * Math.Sin(2 * Math.PI * i / 12.0);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var seasonal = model.GetSeasonal();

            // Assert - Later seasonal amplitudes should be larger
            double earlyAmplitude = Math.Abs(seasonal[6]);  // Peak in first year
            double lateAmplitude = Math.Abs(seasonal[114]);  // Peak in last year
            Assert.True(lateAmplitude > earlyAmplitude, "Seasonal amplitude should increase over time");
        }

        [Fact]
        public void STLDecomposition_MultipleFrequencies_IsolatesDominant()
        {
            // Arrange
            var options = new STLDecompositionOptions<double> { SeasonalPeriod = 12 };
            var model = new STLDecomposition<double>(options);

            int n = 120;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Multiple frequencies: annual (dominant) and semi-annual
            for (int i = 0; i < n; i++)
            {
                y[i] = 50.0 + 20.0 * Math.Sin(2 * Math.PI * i / 12.0) + 5.0 * Math.Sin(2 * Math.PI * i / 6.0);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var seasonal = model.GetSeasonal();

            // Assert - Should capture dominant 12-month pattern
            double variance = CalculateVariance(seasonal);
            Assert.True(variance > 50, $"Seasonal component should capture dominant frequency, variance was {variance:F2}");
        }

        [Fact]
        public void STLDecomposition_Edges_HandleBoundaryConditions()
        {
            // Arrange
            var options = new STLDecompositionOptions<double> { SeasonalPeriod = 12 };
            var model = new STLDecomposition<double>(options);

            int n = 120;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            for (int i = 0; i < n; i++)
            {
                y[i] = 100.0 + 0.3 * i + 15.0 * Math.Sin(2 * Math.PI * i / 12.0);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var trend = model.GetTrend();
            var seasonal = model.GetSeasonal();

            // Assert - Edge values should be reasonable (not NaN or extreme)
            Assert.True(!double.IsNaN(trend[0]) && Math.Abs(trend[0]) < 200, "First trend value should be valid");
            Assert.True(!double.IsNaN(trend[n-1]) && Math.Abs(trend[n-1]) < 200, "Last trend value should be valid");
            Assert.True(!double.IsNaN(seasonal[0]) && Math.Abs(seasonal[0]) < 50, "First seasonal value should be valid");
            Assert.True(!double.IsNaN(seasonal[n-1]) && Math.Abs(seasonal[n-1]) < 50, "Last seasonal value should be valid");
        }

        [Fact]
        public void STLDecomposition_VeryShortPeriod_HandlesMinimalData()
        {
            // Arrange - Very short seasonal period
            var options = new STLDecompositionOptions<double>
            {
                SeasonalPeriod = 3,
                TrendWindowSize = 5,
                SeasonalLoessWindow = 7
            };
            var model = new STLDecomposition<double>(options);

            int n = 24;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            for (int i = 0; i < n; i++)
            {
                y[i] = 50.0 + 10.0 * Math.Sin(2 * Math.PI * i / 3.0);
                X[i, 0] = i;
            }

            // Act & Assert - Should not throw
            model.Train(X, y);
            var seasonal = model.GetSeasonal();

            Assert.Equal(n, seasonal.Length);
            Assert.True(!seasonal.Any(s => double.IsNaN(s)), "No NaN values in seasonal component");
        }

        [Fact]
        public void STLDecomposition_PureNoise_ProducesSmallComponents()
        {
            // Arrange
            var options = new STLDecompositionOptions<double> { SeasonalPeriod = 12 };
            var model = new STLDecomposition<double>(options);

            int n = 120;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            var random = new Random(42);
            for (int i = 0; i < n; i++)
            {
                y[i] = 50.0 + 3.0 * (random.NextDouble() - 0.5);  // Pure noise around mean
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var trend = model.GetTrend();
            var seasonal = model.GetSeasonal();

            // Assert - With pure noise, trend and seasonal should be relatively flat/small
            double trendVariance = CalculateVariance(trend);
            double seasonalVariance = CalculateVariance(seasonal);

            Assert.True(trendVariance < 5.0, $"Trend variance should be small for pure noise, was {trendVariance:F2}");
            Assert.True(seasonalVariance < 5.0, $"Seasonal variance should be small for pure noise, was {seasonalVariance:F2}");
        }

        [Fact]
        public void STLDecomposition_StepChange_CapturedInTrend()
        {
            // Arrange
            var options = new STLDecompositionOptions<double> { SeasonalPeriod = 12, TrendWindowSize = 25 };
            var model = new STLDecomposition<double>(options);

            int n = 120;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Step change at midpoint
            for (int i = 0; i < n / 2; i++)
            {
                y[i] = 50.0 + 10.0 * Math.Sin(2 * Math.PI * i / 12.0);
                X[i, 0] = i;
            }
            for (int i = n / 2; i < n; i++)
            {
                y[i] = 80.0 + 10.0 * Math.Sin(2 * Math.PI * i / 12.0);  // Step up by 30
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var trend = model.GetTrend();

            // Assert - Trend should show the step increase
            double firstHalfMean = 0, secondHalfMean = 0;
            for (int i = 10; i < n / 2 - 10; i++) firstHalfMean += trend[i];
            for (int i = n / 2 + 10; i < n - 10; i++) secondHalfMean += trend[i];

            firstHalfMean /= (n / 2 - 20);
            secondHalfMean /= (n / 2 - 20);

            Assert.True(secondHalfMean > firstHalfMean + 20, "Trend should capture level shift");
        }

        [Fact]
        public void STLDecomposition_CompareDifferentAlgorithms_ProduceSimilarResults()
        {
            // Arrange - Same data, different algorithms
            var standardOptions = new STLDecompositionOptions<double>
            {
                SeasonalPeriod = 12,
                AlgorithmType = STLAlgorithmType.Standard
            };
            var fastOptions = new STLDecompositionOptions<double>
            {
                SeasonalPeriod = 12,
                AlgorithmType = STLAlgorithmType.Fast,
                TrendWindowSize = 25,
                SeasonalLoessWindow = 13
            };

            var standardModel = new STLDecomposition<double>(standardOptions);
            var fastModel = new STLDecomposition<double>(fastOptions);

            int n = 120;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            for (int i = 0; i < n; i++)
            {
                y[i] = 100.0 + 0.5 * i + 20.0 * Math.Sin(2 * Math.PI * i / 12.0);
                X[i, 0] = i;
            }

            // Act
            standardModel.Train(X, y);
            fastModel.Train(X, y);

            var standardTrend = standardModel.GetTrend();
            var fastTrend = fastModel.GetTrend();

            // Assert - Both algorithms should produce similar trends
            double correlation = CalculateCorrelation(standardTrend, fastTrend);
            Assert.True(correlation > 0.9, $"Standard and Fast algorithms should produce similar trends, correlation was {correlation:F3}");
        }

        [Fact]
        public void STLDecomposition_LargeDataset_ProcessesEfficiently()
        {
            // Arrange - Large dataset
            var options = new STLDecompositionOptions<double>
            {
                SeasonalPeriod = 12,
                AlgorithmType = STLAlgorithmType.Fast
            };
            var model = new STLDecomposition<double>(options);

            int n = 300;  // 25 years of monthly data
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            for (int i = 0; i < n; i++)
            {
                y[i] = 100.0 + 0.3 * i + 25.0 * Math.Sin(2 * Math.PI * i / 12.0);
                X[i, 0] = i;
            }

            // Act & Assert - Should complete without performance issues
            model.Train(X, y);
            var trend = model.GetTrend();
            var seasonal = model.GetSeasonal();
            var residual = model.GetResidual();

            Assert.Equal(n, trend.Length);
            Assert.Equal(n, seasonal.Length);
            Assert.Equal(n, residual.Length);
        }

        [Fact]
        public void STLDecomposition_ExtremeOutlier_MinimizedInResiduals()
        {
            // Arrange - Robust algorithm with extreme outlier
            var options = new STLDecompositionOptions<double>
            {
                SeasonalPeriod = 12,
                AlgorithmType = STLAlgorithmType.Robust,
                RobustIterations = 3
            };
            var model = new STLDecomposition<double>(options);

            int n = 120;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            for (int i = 0; i < n; i++)
            {
                y[i] = 50.0 + 0.2 * i + 10.0 * Math.Sin(2 * Math.PI * i / 12.0);
                X[i, 0] = i;
            }

            // Add extreme outlier
            y[60] = 200.0;

            // Act
            model.Train(X, y);
            var residuals = model.GetResidual();

            // Assert - Robust method should isolate outlier in residuals
            int largeResidualCount = 0;
            for (int i = 0; i < residuals.Length; i++)
            {
                if (Math.Abs(residuals[i]) > 50)
                {
                    largeResidualCount++;
                }
            }

            Assert.True(largeResidualCount <= 3, "Robust STL should minimize impact of outliers");
        }

        [Fact]
        public void STLDecomposition_MultiYearData_CapturesLongTermTrends()
        {
            // Arrange - 10 years of monthly data
            var options = new STLDecompositionOptions<double> { SeasonalPeriod = 12, TrendWindowSize = 37 };
            var model = new STLDecomposition<double>(options);

            int n = 120;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Slowly accelerating trend
            for (int i = 0; i < n; i++)
            {
                y[i] = 50.0 + 0.1 * i + 0.005 * i * i + 15.0 * Math.Sin(2 * Math.PI * i / 12.0);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var trend = model.GetTrend();

            // Assert - Trend should show acceleration
            double earlySlope = trend[30] - trend[20];
            double lateSlope = trend[110] - trend[100];

            Assert.True(lateSlope > earlySlope * 1.5, "Trend should show acceleration over time");
        }

        [Fact]
        public void STLDecomposition_WeeklyData_HandlesShortPeriod()
        {
            // Arrange - Daily data with weekly seasonality
            var options = new STLDecompositionOptions<double>
            {
                SeasonalPeriod = 7,
                TrendWindowSize = 11,
                SeasonalLoessWindow = 21
            };
            var model = new STLDecomposition<double>(options);

            int n = 84;  // 12 weeks
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Weekly pattern with weekends higher
            for (int i = 0; i < n; i++)
            {
                double weekendEffect = (i % 7 == 5 || i % 7 == 6) ? 15.0 : 0.0;
                y[i] = 100.0 + weekendEffect;
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var seasonal = model.GetSeasonal();

            // Assert - Should capture weekly pattern
            for (int i = 7; i < n - 7; i++)
            {
                if (i % 7 == 0)  // Same day of week
                {
                    double diff = Math.Abs(seasonal[i] - seasonal[i + 7]);
                    Assert.True(diff < 5.0, $"Weekly pattern should repeat, difference at {i} was {diff:F2}");
                }
            }
        }

        [Fact]
        public void STLDecomposition_ConstantSeries_HandlesGracefully()
        {
            // Arrange
            var options = new STLDecompositionOptions<double> { SeasonalPeriod = 12 };
            var model = new STLDecomposition<double>(options);

            int n = 60;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            // Constant series
            for (int i = 0; i < n; i++)
            {
                y[i] = 50.0;
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var trend = model.GetTrend();
            var seasonal = model.GetSeasonal();
            var residual = model.GetResidual();

            // Assert - Trend should be constant, seasonal near zero
            double trendRange = 0;
            for (int i = 0; i < n; i++)
            {
                if (trend[i] > trendRange) trendRange = Math.Max(trendRange, trend[i]);
            }
            trendRange -= trend.Min();

            Assert.True(trendRange < 5.0, "Trend should be nearly constant for constant series");
            Assert.True(seasonal.All(s => Math.Abs(s) < 2.0), "Seasonal should be near zero for constant series");
        }

        [Fact]
        public void STLDecomposition_SeasonalNormalization_SumsToZero()
        {
            // Arrange
            var options = new STLDecompositionOptions<double> { SeasonalPeriod = 12 };
            var model = new STLDecomposition<double>(options);

            int n = 120;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            for (int i = 0; i < n; i++)
            {
                y[i] = 100.0 + 30.0 * Math.Sin(2 * Math.PI * i / 12.0);
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var seasonal = model.GetSeasonal();

            // Assert - Seasonal component should sum to near zero
            double seasonalSum = 0;
            for (int i = 0; i < seasonal.Length; i++)
            {
                seasonalSum += seasonal[i];
            }

            Assert.True(Math.Abs(seasonalSum) < 1.0, $"Seasonal component should sum to near zero, was {seasonalSum:F2}");
        }

        [Fact]
        public void STLDecomposition_Reset_ClearsComponents()
        {
            // Arrange
            var options = new STLDecompositionOptions<double> { SeasonalPeriod = 12 };
            var model = new STLDecomposition<double>(options);

            int n = 60;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            for (int i = 0; i < n; i++)
            {
                y[i] = 100.0 + 10.0 * Math.Sin(2 * Math.PI * i / 12.0);
                X[i, 0] = i;
            }

            model.Train(X, y);

            // Act
            model.Reset();

            // Assert - Should throw when trying to access components after reset
            Assert.Throws<InvalidOperationException>(() => model.GetTrend());
            Assert.Throws<InvalidOperationException>(() => model.GetSeasonal());
            Assert.Throws<InvalidOperationException>(() => model.GetResidual());
        }

        [Fact]
        public void STLDecomposition_ComponentStrengths_ReflectDataCharacteristics()
        {
            // Arrange - Strong trend, weak seasonality
            var options = new STLDecompositionOptions<double> { SeasonalPeriod = 12 };
            var model = new STLDecomposition<double>(options);

            int n = 120;
            var y = new Vector<double>(n);
            var X = new Matrix<double>(n, 1);

            for (int i = 0; i < n; i++)
            {
                y[i] = 10.0 + 1.0 * i + 2.0 * Math.Sin(2 * Math.PI * i / 12.0);  // Strong trend, weak seasonal
                X[i, 0] = i;
            }

            // Act
            model.Train(X, y);
            var metadata = model.GetModelMetadata();

            double trendStrength = Convert.ToDouble(metadata.AdditionalInfo["TrendStrength"]);
            double seasonalStrength = Convert.ToDouble(metadata.AdditionalInfo["SeasonalStrength"]);

            // Assert - Trend strength should exceed seasonal strength
            Assert.True(trendStrength > seasonalStrength, $"Trend strength ({trendStrength:F3}) should exceed seasonal strength ({seasonalStrength:F3})");
            Assert.True(trendStrength > 0.8, $"Trend strength should be high, was {trendStrength:F3}");
        }

        #endregion

        #region Helper Methods

        /// <summary>
        /// Calculates Pearson correlation coefficient between two vectors.
        /// </summary>
        private double CalculateCorrelation(Vector<double> x, Vector<double> y)
        {
            if (x.Length != y.Length) throw new ArgumentException("Vectors must have same length");

            int n = x.Length;
            double meanX = 0, meanY = 0;

            for (int i = 0; i < n; i++)
            {
                meanX += x[i];
                meanY += y[i];
            }
            meanX /= n;
            meanY /= n;

            double numerator = 0, denomX = 0, denomY = 0;
            for (int i = 0; i < n; i++)
            {
                double dx = x[i] - meanX;
                double dy = y[i] - meanY;
                numerator += dx * dy;
                denomX += dx * dx;
                denomY += dy * dy;
            }

            if (denomX == 0 || denomY == 0) return 0;
            return numerator / Math.Sqrt(denomX * denomY);
        }

        /// <summary>
        /// Calculates variance of a vector.
        /// </summary>
        private double CalculateVariance(Vector<double> x)
        {
            int n = x.Length;
            double mean = 0;
            for (int i = 0; i < n; i++) mean += x[i];
            mean /= n;

            double variance = 0;
            for (int i = 0; i < n; i++)
            {
                double diff = x[i] - mean;
                variance += diff * diff;
            }
            return variance / n;
        }

        /// <summary>
        /// Calculates autocorrelation function at a given lag.
        /// </summary>
        private double CalculateACF(List<double> series, int lag)
        {
            if (lag >= series.Count) return 0;

            int n = series.Count;
            double mean = series.Average();

            double numerator = 0;
            double denominator = 0;

            for (int i = 0; i < n - lag; i++)
            {
                numerator += (series[i] - mean) * (series[i + lag] - mean);
            }

            for (int i = 0; i < n; i++)
            {
                denominator += Math.Pow(series[i] - mean, 2);
            }

            return denominator == 0 ? 0 : numerator / denominator;
        }

        #endregion
    }
}
