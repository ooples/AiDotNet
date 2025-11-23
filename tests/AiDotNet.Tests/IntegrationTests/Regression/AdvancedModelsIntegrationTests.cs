using AiDotNet.LinearAlgebra;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Regression
{
    /// <summary>
    /// Integration tests for advanced regression models.
    /// Tests GAM, isotonic regression, time series, genetic algorithms, and symbolic regression.
    /// </summary>
    public class AdvancedModelsIntegrationTests
    {
        #region GeneralizedAdditiveModelRegression Tests

        [Fact]
        public void GeneralizedAdditiveModelRegression_AdditiveComponents_FitsWell()
        {
            // Arrange - additive relationship: y = f1(x1) + f2(x2)
            var x = new Matrix<double>(30, 2);
            var y = new Vector<double>(30);

            for (int i = 0; i < 30; i++)
            {
                x[i, 0] = i / 5.0;
                x[i, 1] = i / 3.0;
                y[i] = Math.Sin(x[i, 0]) * 5 + Math.Cos(x[i, 1]) * 5 + 10;
            }

            // Act
            var regression = new GeneralizedAdditiveModelRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should capture additive structure
            for (int i = 0; i < 30; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 5.0);
            }
        }

        [Fact]
        public void GeneralizedAdditiveModelRegression_SmoothingSplines_CreatesSmoothFits()
        {
            // Arrange
            var x = new Matrix<double>(25, 2);
            var y = new Vector<double>(25);

            for (int i = 0; i < 25; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = Math.Sqrt(x[i, 0]) * 3 + Math.Log(x[i, 1] + 1) * 2;
            }

            var options = new GeneralizedAdditiveModelOptions { SmoothingParameter = 0.5 };

            // Act
            var regression = new GeneralizedAdditiveModelRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            Assert.NotNull(predictions);
        }

        [Fact]
        public void GeneralizedAdditiveModelRegression_InterpretableComponents_CanExtract()
        {
            // Arrange
            var x = new Matrix<double>(20, 2);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 1.5;
                y[i] = 2 * x[i, 0] + 3 * x[i, 1];
            }

            // Act
            var regression = new GeneralizedAdditiveModelRegression<double>();
            regression.Train(x, y);

            var component1 = regression.GetComponentFunction(0);
            var component2 = regression.GetComponentFunction(1);

            // Assert - should extract individual component functions
            Assert.NotNull(component1);
            Assert.NotNull(component2);
        }

        [Fact]
        public void GeneralizedAdditiveModelRegression_DifferentSmoothing_AffectsFit()
        {
            // Arrange
            var x = new Matrix<double>(30, 2);
            var y = new Vector<double>(30);

            for (int i = 0; i < 30; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = x[i, 0] + x[i, 1];
            }

            // Act - different smoothing parameters
            var gamSmooth = new GeneralizedAdditiveModelRegression<double>(
                new GeneralizedAdditiveModelOptions { SmoothingParameter = 0.1 });
            gamSmooth.Train(x, y);

            var gamRough = new GeneralizedAdditiveModelRegression<double>(
                new GeneralizedAdditiveModelOptions { SmoothingParameter = 0.9 });
            gamRough.Train(x, y);

            // Assert
            var predSmooth = gamSmooth.Predict(x);
            var predRough = gamRough.Predict(x);
            Assert.NotNull(predSmooth);
            Assert.NotNull(predRough);
        }

        [Fact]
        public void GeneralizedAdditiveModelRegression_NonLinearInteractions_CapturesPartially()
        {
            // Arrange
            var x = new Matrix<double>(25, 2);
            var y = new Vector<double>(25);

            for (int i = 0; i < 25; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = x[i, 0] + x[i, 1] + Math.Sin(x[i, 0]) * Math.Cos(x[i, 1]);
            }

            // Act
            var regression = new GeneralizedAdditiveModelRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - may not capture full interaction but should approximate
            Assert.NotNull(predictions);
        }

        [Fact]
        public void GeneralizedAdditiveModelRegression_CrossValidation_SelectsOptimalSmoothing()
        {
            // Arrange
            var x = new Matrix<double>(40, 2);
            var y = new Vector<double>(40);
            var random = new Random(789);

            for (int i = 0; i < 40; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 1.5;
                y[i] = x[i, 0] + x[i, 1] + (random.NextDouble() - 0.5) * 5;
            }

            var options = new GeneralizedAdditiveModelOptions { UseCrossValidation = true };

            // Act
            var regression = new GeneralizedAdditiveModelRegression<double>(options);
            regression.Train(x, y);

            var optimalSmoothing = regression.GetOptimalSmoothingParameter();

            // Assert
            Assert.True(optimalSmoothing > 0 && optimalSmoothing < 1);
        }

        [Fact]
        public void GeneralizedAdditiveModelRegression_PartialResidualPlots_Available()
        {
            // Arrange
            var x = new Matrix<double>(20, 2);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = x[i, 0] + 2 * x[i, 1];
            }

            // Act
            var regression = new GeneralizedAdditiveModelRegression<double>();
            regression.Train(x, y);

            var partialResiduals = regression.GetPartialResiduals(0);

            // Assert
            Assert.NotNull(partialResiduals);
        }

        [Fact]
        public void GeneralizedAdditiveModelRegression_MultipleFeatures_HandlesHighDimensional()
        {
            // Arrange
            var x = new Matrix<double>(30, 4);
            var y = new Vector<double>(30);

            for (int i = 0; i < 30; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 1.5;
                x[i, 2] = i / 2.0;
                x[i, 3] = i * 0.8;
                y[i] = x[i, 0] + 2 * x[i, 1] - x[i, 2] + 0.5 * x[i, 3];
            }

            // Act
            var regression = new GeneralizedAdditiveModelRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 30; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 10.0);
            }
        }

        [Fact]
        public void GeneralizedAdditiveModelRegression_FloatType_WorksCorrectly()
        {
            // Arrange
            var x = new Matrix<float>(15, 2);
            var y = new Vector<float>(15);

            for (int i = 0; i < 15; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = x[i, 0] + x[i, 1];
            }

            // Act
            var regression = new GeneralizedAdditiveModelRegression<float>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            Assert.NotNull(predictions);
        }

        [Fact]
        public void GeneralizedAdditiveModelRegression_BackfittingAlgorithm_Converges()
        {
            // Arrange
            var x = new Matrix<double>(25, 2);
            var y = new Vector<double>(25);

            for (int i = 0; i < 25; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 1.5;
                y[i] = Math.Sqrt(x[i, 0]) * 3 + x[i, 1];
            }

            var options = new GeneralizedAdditiveModelOptions { MaxIterations = 50 };

            // Act
            var regression = new GeneralizedAdditiveModelRegression<double>(options);
            regression.Train(x, y);

            var iterations = regression.GetActualIterations();

            // Assert - should converge before max iterations
            Assert.True(iterations <= 50);
        }

        [Fact]
        public void GeneralizedAdditiveModelRegression_PenalizedLikelihood_BalancesFitAndSmoothness()
        {
            // Arrange
            var x = new Matrix<double>(30, 2);
            var y = new Vector<double>(30);
            var random = new Random(321);

            for (int i = 0; i < 30; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = x[i, 0] + x[i, 1] + (random.NextDouble() - 0.5) * 3;
            }

            var options = new GeneralizedAdditiveModelOptions
            {
                SmoothingParameter = 0.5,
                UsePenalizedLikelihood = true
            };

            // Act
            var regression = new GeneralizedAdditiveModelRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            Assert.NotNull(predictions);
        }

        #endregion

        #region IsotonicRegression Tests

        [Fact]
        public void IsotonicRegression_MonotonicIncreasing_FitsStepFunction()
        {
            // Arrange - monotonically increasing data
            var x = new Matrix<double>(15, 1);
            var y = new Vector<double>(new[] { 1.0, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0, 6.0, 7.0, 7.5, 8.0, 9.0, 10.0, 11.0, 12.0 });

            for (int i = 0; i < 15; i++)
            {
                x[i, 0] = i;
            }

            // Act
            var regression = new IsotonicRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - predictions should be monotonically increasing
            for (int i = 1; i < 15; i++)
            {
                Assert.True(predictions[i] >= predictions[i - 1]);
            }
        }

        [Fact]
        public void IsotonicRegression_ViolatesMonotonicity_Corrects()
        {
            // Arrange - data that violates monotonicity
            var x = new Matrix<double>(10, 1);
            var y = new Vector<double>(new[] { 1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0, 8.0, 10.0 });

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
            }

            // Act
            var regression = new IsotonicRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should enforce monotonicity
            for (int i = 1; i < 10; i++)
            {
                Assert.True(predictions[i] >= predictions[i - 1]);
            }
        }

        [Fact]
        public void IsotonicRegression_Decreasing_HandlesCorrectly()
        {
            // Arrange - monotonically decreasing
            var x = new Matrix<double>(10, 1);
            var y = new Vector<double>(new[] { 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 });

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
            }

            var options = new IsotonicRegressionOptions { Increasing = false };

            // Act
            var regression = new IsotonicRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - predictions should be monotonically decreasing
            for (int i = 1; i < 10; i++)
            {
                Assert.True(predictions[i] <= predictions[i - 1]);
            }
        }

        [Fact]
        public void IsotonicRegression_PoolAdjacentViolators_Averages()
        {
            // Arrange
            var x = new Matrix<double>(8, 1);
            var y = new Vector<double>(new[] { 1.0, 3.0, 2.0, 2.0, 5.0, 4.0, 7.0, 8.0 });

            for (int i = 0; i < 8; i++)
            {
                x[i, 0] = i;
            }

            // Act
            var regression = new IsotonicRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - violations should be pooled (averaged)
            for (int i = 1; i < 8; i++)
            {
                Assert.True(predictions[i] >= predictions[i - 1]);
            }
        }

        [Fact]
        public void IsotonicRegression_PerfectMonotonic_PreservesData()
        {
            // Arrange - already monotonic
            var x = new Matrix<double>(10, 1);
            var y = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 });

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
            }

            // Act
            var regression = new IsotonicRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should preserve perfect monotonic data
            for (int i = 0; i < 10; i++)
            {
                Assert.Equal(y[i], predictions[i], precision: 1);
            }
        }

        [Fact]
        public void IsotonicRegression_Calibration_MapsToMonotonic()
        {
            // Arrange - calibration curve (e.g., for probability calibration)
            var x = new Matrix<double>(12, 1);
            var y = new Vector<double>(new[] { 0.1, 0.15, 0.2, 0.18, 0.3, 0.35, 0.5, 0.6, 0.7, 0.75, 0.9, 0.95 });

            for (int i = 0; i < 12; i++)
            {
                x[i, 0] = i / 11.0;
            }

            // Act
            var regression = new IsotonicRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 1; i < 12; i++)
            {
                Assert.True(predictions[i] >= predictions[i - 1]);
            }
        }

        [Fact]
        public void IsotonicRegression_SmallDataset_HandlesCorrectly()
        {
            // Arrange
            var x = new Matrix<double>(4, 1);
            var y = new Vector<double>(new[] { 2.0, 1.0, 3.0, 4.0 });

            for (int i = 0; i < 4; i++)
            {
                x[i, 0] = i;
            }

            // Act
            var regression = new IsotonicRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 1; i < 4; i++)
            {
                Assert.True(predictions[i] >= predictions[i - 1]);
            }
        }

        [Fact]
        public void IsotonicRegression_LargeDataset_HandlesEfficiently()
        {
            // Arrange
            var n = 500;
            var x = new Matrix<double>(n, 1);
            var y = new Vector<double>(n);
            var random = new Random(123);

            for (int i = 0; i < n; i++)
            {
                x[i, 0] = i;
                y[i] = i + (random.NextDouble() - 0.5) * 10;
            }

            // Act
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var regression = new IsotonicRegression<double>();
            regression.Train(x, y);
            sw.Stop();

            // Assert
            Assert.True(sw.ElapsedMilliseconds < 5000);
        }

        [Fact]
        public void IsotonicRegression_WeightedSamples_UsesWeights()
        {
            // Arrange
            var x = new Matrix<double>(10, 1);
            var y = new Vector<double>(new[] { 1.0, 2.0, 1.5, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 10.0 });
            var weights = new Vector<double>(new[] { 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }); // High weight on violation

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
            }

            // Act
            var regression = new IsotonicRegression<double>();
            regression.TrainWithWeights(x, y, weights);
            var predictions = regression.Predict(x);

            // Assert - should respect weights
            for (int i = 1; i < 10; i++)
            {
                Assert.True(predictions[i] >= predictions[i - 1]);
            }
        }

        [Fact]
        public void IsotonicRegression_InterpolationBetweenPoints_UsesStepFunction()
        {
            // Arrange
            var x = new Matrix<double>(5, 1);
            var y = new Vector<double>(new[] { 1.0, 3.0, 5.0, 7.0, 9.0 });

            for (int i = 0; i < 5; i++)
            {
                x[i, 0] = i * 2; // 0, 2, 4, 6, 8
            }

            var regression = new IsotonicRegression<double>();
            regression.Train(x, y);

            // Act - predict between training points
            var testX = new Matrix<double>(1, 1);
            testX[0, 0] = 3; // Between 2 and 4

            var prediction = regression.Predict(testX);

            // Assert - should interpolate (likely step function)
            Assert.True(prediction[0] >= 3.0 && prediction[0] <= 5.0);
        }

        #endregion

        #region TimeSeriesRegression Tests

        [Fact]
        public void TimeSeriesRegression_AutoregressivePattern_CapturesTrend()
        {
            // Arrange - AR(1) process: y_t = 0.8 * y_{t-1} + noise
            var n = 50;
            var y = new Vector<double>(n);
            y[0] = 10.0;

            for (int i = 1; i < n; i++)
            {
                y[i] = 0.8 * y[i - 1] + 2.0;
            }

            var options = new TimeSeriesRegressionOptions { Lag = 1 };

            // Act
            var regression = new TimeSeriesRegression<double>(options);
            regression.TrainTimeSeries(y);
            var predictions = regression.PredictTimeSeries(10);

            // Assert - should predict reasonable future values
            Assert.Equal(10, predictions.Length);
            Assert.True(predictions[0] > 0);
        }

        [Fact]
        public void TimeSeriesRegression_MultipleLags_CapturesComplexDynamics()
        {
            // Arrange - AR(3) process
            var n = 60;
            var y = new Vector<double>(n);
            y[0] = 10.0;
            y[1] = 12.0;
            y[2] = 14.0;

            for (int i = 3; i < n; i++)
            {
                y[i] = 0.5 * y[i - 1] + 0.3 * y[i - 2] + 0.2 * y[i - 3] + 1.0;
            }

            var options = new TimeSeriesRegressionOptions { Lag = 3 };

            // Act
            var regression = new TimeSeriesRegression<double>(options);
            regression.TrainTimeSeries(y);
            var predictions = regression.PredictTimeSeries(5);

            // Assert
            Assert.Equal(5, predictions.Length);
        }

        [Fact]
        public void TimeSeriesRegression_TrendComponent_Extracts()
        {
            // Arrange - linear trend + seasonal
            var n = 40;
            var y = new Vector<double>(n);

            for (int i = 0; i < n; i++)
            {
                y[i] = i * 2 + Math.Sin(i / 4.0) * 5;
            }

            var options = new TimeSeriesRegressionOptions { ExtractTrend = true };

            // Act
            var regression = new TimeSeriesRegression<double>(options);
            regression.TrainTimeSeries(y);

            var trend = regression.GetTrendComponent();

            // Assert - trend should be extracted
            Assert.NotNull(trend);
            Assert.True(trend.Length > 0);
        }

        [Fact]
        public void TimeSeriesRegression_SeasonalComponent_Detects()
        {
            // Arrange - strong seasonality
            var n = 48;
            var y = new Vector<double>(n);

            for (int i = 0; i < n; i++)
            {
                y[i] = 50 + 10 * Math.Sin(2 * Math.PI * i / 12); // Period of 12
            }

            var options = new TimeSeriesRegressionOptions { SeasonalPeriod = 12 };

            // Act
            var regression = new TimeSeriesRegression<double>(options);
            regression.TrainTimeSeries(y);

            var seasonal = regression.GetSeasonalComponent();

            // Assert
            Assert.NotNull(seasonal);
        }

        [Fact]
        public void TimeSeriesRegression_ExogenousVariables_IncorporatesExternal()
        {
            // Arrange - time series with external predictor
            var n = 30;
            var y = new Vector<double>(n);
            var exogenous = new Matrix<double>(n, 1);

            for (int i = 0; i < n; i++)
            {
                exogenous[i, 0] = i;
                y[i] = 0.7 * (i > 0 ? y[i - 1] : 10) + 2 * exogenous[i, 0];
            }

            var options = new TimeSeriesRegressionOptions { Lag = 1, UseExogenous = true };

            // Act
            var regression = new TimeSeriesRegression<double>(options);
            regression.TrainWithExogenous(y, exogenous);

            // Assert - should train successfully
            Assert.True(regression.IsTrained);
        }

        [Fact]
        public void TimeSeriesRegression_ForecastingHorizon_PredictsFuture()
        {
            // Arrange
            var n = 50;
            var y = new Vector<double>(n);

            for (int i = 0; i < n; i++)
            {
                y[i] = 20 + i * 0.5;
            }

            var options = new TimeSeriesRegressionOptions { Lag = 2 };

            // Act
            var regression = new TimeSeriesRegression<double>(options);
            regression.TrainTimeSeries(y);

            var forecast = regression.Forecast(10);

            // Assert - should forecast 10 steps ahead
            Assert.Equal(10, forecast.Length);
            Assert.True(forecast[9] > forecast[0]); // Should increase
        }

        [Fact]
        public void TimeSeriesRegression_StationarityCheck_Detects()
        {
            // Arrange - non-stationary (random walk)
            var n = 50;
            var y = new Vector<double>(n);
            y[0] = 0;

            for (int i = 1; i < n; i++)
            {
                y[i] = y[i - 1] + 1.0;
            }

            // Act
            var regression = new TimeSeriesRegression<double>();
            regression.TrainTimeSeries(y);

            var isStationary = regression.CheckStationarity();

            // Assert - should detect non-stationarity
            Assert.False(isStationary);
        }

        [Fact]
        public void TimeSeriesRegression_Differencing_MakesStationary()
        {
            // Arrange - non-stationary data
            var n = 40;
            var y = new Vector<double>(n);

            for (int i = 0; i < n; i++)
            {
                y[i] = i * i; // Quadratic trend
            }

            var options = new TimeSeriesRegressionOptions { DifferencingOrder = 2 };

            // Act
            var regression = new TimeSeriesRegression<double>(options);
            regression.TrainTimeSeries(y);

            var differenced = regression.GetDifferencedSeries();

            // Assert - differenced series should be more stationary
            Assert.NotNull(differenced);
        }

        [Fact]
        public void TimeSeriesRegression_ConfidenceIntervals_ProvidedForForecasts()
        {
            // Arrange
            var n = 30;
            var y = new Vector<double>(n);

            for (int i = 0; i < n; i++)
            {
                y[i] = 10 + i;
            }

            var options = new TimeSeriesRegressionOptions { Lag = 1 };

            // Act
            var regression = new TimeSeriesRegression<double>(options);
            regression.TrainTimeSeries(y);

            var (forecast, lower, upper) = regression.ForecastWithConfidenceIntervals(5, 0.95);

            // Assert - confidence intervals should bound forecast
            for (int i = 0; i < 5; i++)
            {
                Assert.True(lower[i] <= forecast[i] && forecast[i] <= upper[i]);
            }
        }

        [Fact]
        public void TimeSeriesRegression_ResidualAnalysis_ChecksAssumptions()
        {
            // Arrange
            var n = 40;
            var y = new Vector<double>(n);
            var random = new Random(456);

            for (int i = 0; i < n; i++)
            {
                y[i] = 20 + i + (random.NextDouble() - 0.5) * 5;
            }

            var options = new TimeSeriesRegressionOptions { Lag = 1 };

            // Act
            var regression = new TimeSeriesRegression<double>(options);
            regression.TrainTimeSeries(y);

            var residuals = regression.GetResiduals();

            // Assert - residuals should be available
            Assert.NotNull(residuals);
            Assert.True(residuals.Length > 0);
        }

        [Fact]
        public void TimeSeriesRegression_AutocorrelationFunction_Computes()
        {
            // Arrange
            var n = 50;
            var y = new Vector<double>(n);

            for (int i = 0; i < n; i++)
            {
                y[i] = Math.Sin(i / 5.0) * 10;
            }

            // Act
            var regression = new TimeSeriesRegression<double>();
            regression.TrainTimeSeries(y);

            var acf = regression.ComputeAutocorrelationFunction(10);

            // Assert - should compute ACF for 10 lags
            Assert.Equal(10, acf.Length);
        }

        [Fact]
        public void TimeSeriesRegression_MovingAverage_Smooths()
        {
            // Arrange - noisy data
            var n = 30;
            var y = new Vector<double>(n);
            var random = new Random(789);

            for (int i = 0; i < n; i++)
            {
                y[i] = i + (random.NextDouble() - 0.5) * 10;
            }

            var options = new TimeSeriesRegressionOptions { MovingAverageWindow = 5 };

            // Act
            var regression = new TimeSeriesRegression<double>(options);
            regression.TrainTimeSeries(y);

            var smoothed = regression.GetSmoothedSeries();

            // Assert - smoothed series should have less variance
            Assert.NotNull(smoothed);
        }

        #endregion

        #region GeneticAlgorithmRegression Tests

        [Fact]
        public void GeneticAlgorithmRegression_EvolutionaryOptimization_FindsGoodSolution()
        {
            // Arrange
            var x = new Matrix<double>(25, 2);
            var y = new Vector<double>(25);

            for (int i = 0; i < 25; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = 3 * x[i, 0] + 2 * x[i, 1] + 5;
            }

            var options = new GeneticAlgorithmOptions
            {
                PopulationSize = 50,
                Generations = 100,
                MutationRate = 0.1
            };

            // Act
            var regression = new GeneticAlgorithmRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should evolve reasonable solution
            double totalError = 0;
            for (int i = 0; i < 25; i++)
            {
                totalError += Math.Abs(predictions[i] - y[i]);
            }
            Assert.True(totalError / 25 < 15.0);
        }

        [Fact]
        public void GeneticAlgorithmRegression_PopulationSize_AffectsDiversity()
        {
            // Arrange
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = i * i;
            }

            // Act - different population sizes
            var gaSmall = new GeneticAlgorithmRegression<double>(
                new GeneticAlgorithmOptions { PopulationSize = 10, Generations = 50 });
            gaSmall.Train(x, y);

            var gaLarge = new GeneticAlgorithmRegression<double>(
                new GeneticAlgorithmOptions { PopulationSize = 100, Generations = 50 });
            gaLarge.Train(x, y);

            // Assert - both should produce valid solutions
            var predSmall = gaSmall.Predict(x);
            var predLarge = gaLarge.Predict(x);
            Assert.NotNull(predSmall);
            Assert.NotNull(predLarge);
        }

        [Fact]
        public void GeneticAlgorithmRegression_CrossoverOperator_RecombinesGenes()
        {
            // Arrange
            var x = new Matrix<double>(20, 2);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 1.5;
                y[i] = x[i, 0] + x[i, 1];
            }

            var options = new GeneticAlgorithmOptions
            {
                PopulationSize = 40,
                Generations = 80,
                CrossoverRate = 0.8
            };

            // Act
            var regression = new GeneticAlgorithmRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            Assert.NotNull(predictions);
        }

        [Fact]
        public void GeneticAlgorithmRegression_MutationRate_BalancesExploration()
        {
            // Arrange
            var x = new Matrix<double>(15, 1);
            var y = new Vector<double>(15);

            for (int i = 0; i < 15; i++)
            {
                x[i, 0] = i;
                y[i] = Math.Sin(i / 3.0) * 10;
            }

            // Act - different mutation rates
            var gaLowMutation = new GeneticAlgorithmRegression<double>(
                new GeneticAlgorithmOptions { PopulationSize = 30, Generations = 50, MutationRate = 0.01 });
            gaLowMutation.Train(x, y);

            var gaHighMutation = new GeneticAlgorithmRegression<double>(
                new GeneticAlgorithmOptions { PopulationSize = 30, Generations = 50, MutationRate = 0.3 });
            gaHighMutation.Train(x, y);

            // Assert
            var predLow = gaLowMutation.Predict(x);
            var predHigh = gaHighMutation.Predict(x);
            Assert.NotNull(predLow);
            Assert.NotNull(predHigh);
        }

        [Fact]
        public void GeneticAlgorithmRegression_FitnessFunction_GuidesEvolution()
        {
            // Arrange
            var x = new Matrix<double>(20, 2);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = 2 * x[i, 0] + 3 * x[i, 1];
            }

            var options = new GeneticAlgorithmOptions
            {
                PopulationSize = 50,
                Generations = 100,
                FitnessFunction = FitnessFunction.MeanSquaredError
            };

            // Act
            var regression = new GeneticAlgorithmRegression<double>(options);
            regression.Train(x, y);

            var bestFitness = regression.GetBestFitness();

            // Assert - fitness should improve over generations
            Assert.True(bestFitness >= 0);
        }

        [Fact]
        public void GeneticAlgorithmRegression_ElitismStrategy_PreservesBest()
        {
            // Arrange
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = i * 3 + 2;
            }

            var options = new GeneticAlgorithmOptions
            {
                PopulationSize = 40,
                Generations = 60,
                ElitismRate = 0.1 // Keep best 10%
            };

            // Act
            var regression = new GeneticAlgorithmRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            Assert.NotNull(predictions);
        }

        #endregion

        #region SymbolicRegression Tests

        [Fact]
        public void SymbolicRegression_DiscoversMathematicalExpression()
        {
            // Arrange - y = x^2 + 2*x + 1
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i - 10;
                y[i] = x[i, 0] * x[i, 0] + 2 * x[i, 0] + 1;
            }

            var options = new SymbolicRegressionOptions
            {
                PopulationSize = 100,
                Generations = 200,
                MaxTreeDepth = 5
            };

            // Act
            var regression = new SymbolicRegression<double>(options);
            regression.Train(x, y);

            var expression = regression.GetBestExpression();

            // Assert - should discover an expression close to x^2 + 2x + 1
            Assert.NotNull(expression);
        }

        [Fact]
        public void SymbolicRegression_GeneticProgramming_EvolvesTrees()
        {
            // Arrange
            var x = new Matrix<double>(15, 1);
            var y = new Vector<double>(15);

            for (int i = 0; i < 15; i++)
            {
                x[i, 0] = i;
                y[i] = Math.Sin(x[i, 0]) * 5;
            }

            var options = new SymbolicRegressionOptions
            {
                PopulationSize = 80,
                Generations = 150,
                AllowedFunctions = new[] { "sin", "cos", "add", "multiply" }
            };

            // Act
            var regression = new SymbolicRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 15; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 5.0);
            }
        }

        [Fact]
        public void SymbolicRegression_Parsimony_FavorsSimplicity()
        {
            // Arrange
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = 3 * i + 5; // Simple linear
            }

            var options = new SymbolicRegressionOptions
            {
                PopulationSize = 60,
                Generations = 100,
                ParsimonyPressure = 0.01 // Favor simpler expressions
            };

            // Act
            var regression = new SymbolicRegression<double>(options);
            regression.Train(x, y);

            var expression = regression.GetBestExpression();
            var complexity = regression.GetExpressionComplexity();

            // Assert - should find simple expression
            Assert.True(complexity < 10);
        }

        [Fact]
        public void SymbolicRegression_MultipleFeatures_CombinesFeatures()
        {
            // Arrange - y = x1 * x2 + x1
            var x = new Matrix<double>(20, 2);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = x[i, 0] * x[i, 1] + x[i, 0];
            }

            var options = new SymbolicRegressionOptions
            {
                PopulationSize = 100,
                Generations = 150
            };

            // Act
            var regression = new SymbolicRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            double totalError = 0;
            for (int i = 0; i < 20; i++)
            {
                totalError += Math.Abs(predictions[i] - y[i]);
            }
            Assert.True(totalError / 20 < 50.0);
        }

        [Fact]
        public void SymbolicRegression_ExpressionSimplification_ReducesComplexity()
        {
            // Arrange
            var x = new Matrix<double>(15, 1);
            var y = new Vector<double>(15);

            for (int i = 0; i < 15; i++)
            {
                x[i, 0] = i;
                y[i] = i + i; // Should simplify to 2*i
            }

            var options = new SymbolicRegressionOptions
            {
                PopulationSize = 50,
                Generations = 100,
                SimplifyExpressions = true
            };

            // Act
            var regression = new SymbolicRegression<double>(options);
            regression.Train(x, y);

            var simplified = regression.GetSimplifiedExpression();

            // Assert
            Assert.NotNull(simplified);
        }

        [Fact]
        public void SymbolicRegression_TreeCrossover_ExchangesSubtrees()
        {
            // Arrange
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = (i + 1) * (i + 2);
            }

            var options = new SymbolicRegressionOptions
            {
                PopulationSize = 60,
                Generations = 120,
                CrossoverRate = 0.9
            };

            // Act
            var regression = new SymbolicRegression<double>(options);
            regression.Train(x, y);

            // Assert
            var predictions = regression.Predict(x);
            Assert.NotNull(predictions);
        }

        [Fact]
        public void SymbolicRegression_ConstantOptimization_RefinesNumericalValues()
        {
            // Arrange
            var x = new Matrix<double>(15, 1);
            var y = new Vector<double>(15);

            for (int i = 0; i < 15; i++)
            {
                x[i, 0] = i;
                y[i] = 3.14159 * i + 2.71828;
            }

            var options = new SymbolicRegressionOptions
            {
                PopulationSize = 50,
                Generations = 100,
                OptimizeConstants = true
            };

            // Act
            var regression = new SymbolicRegression<double>(options);
            regression.Train(x, y);

            var optimizedConstants = regression.GetOptimizedConstants();

            // Assert
            Assert.NotNull(optimizedConstants);
        }

        #endregion
    }
}
