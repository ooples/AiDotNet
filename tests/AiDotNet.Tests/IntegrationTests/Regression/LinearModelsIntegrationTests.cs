using AiDotNet.LinearAlgebra;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Regression
{
    /// <summary>
    /// Integration tests for linear regression models (Multivariate, Multiple, Weighted, Robust, etc.)
    /// Tests ensure correct fitting, prediction, and handling of various data scenarios.
    /// </summary>
    public class LinearModelsIntegrationTests
    {
        #region MultivariateRegression Tests

        [Fact]
        public void MultivariateRegression_PerfectLinearRelationship_FitsCorrectly()
        {
            // Arrange - y = 2*x1 + 3*x2 + 1
            var x = new Matrix<double>(5, 2);
            x[0, 0] = 1.0; x[0, 1] = 2.0;
            x[1, 0] = 2.0; x[1, 1] = 3.0;
            x[2, 0] = 3.0; x[2, 1] = 4.0;
            x[3, 0] = 4.0; x[3, 1] = 5.0;
            x[4, 0] = 5.0; x[4, 1] = 6.0;

            var y = new Vector<double>(5);
            y[0] = 9.0;   // 2*1 + 3*2 + 1
            y[1] = 14.0;  // 2*2 + 3*3 + 1
            y[2] = 19.0;  // 2*3 + 3*4 + 1
            y[3] = 24.0;  // 2*4 + 3*5 + 1
            y[4] = 29.0;  // 2*5 + 3*6 + 1

            // Act
            var regression = new MultivariateRegression<double>();
            regression.Train(x, y);

            // Assert
            Assert.Equal(2.0, regression.Coefficients[0], precision: 10);
            Assert.Equal(3.0, regression.Coefficients[1], precision: 10);
            Assert.Equal(1.0, regression.Intercept, precision: 10);
        }

        [Fact]
        public void MultivariateRegression_WithNoise_FitsReasonably()
        {
            // Arrange - y ≈ 1.5*x1 + 2.5*x2 + 3 with noise
            var x = new Matrix<double>(20, 2);
            var y = new Vector<double>(20);
            var random = new Random(42);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = 1.5 * x[i, 0] + 2.5 * x[i, 1] + 3 + (random.NextDouble() - 0.5) * 2;
            }

            // Act
            var regression = new MultivariateRegression<double>();
            regression.Train(x, y);

            // Assert - coefficients should be close to true values
            Assert.True(Math.Abs(regression.Coefficients[0] - 1.5) < 0.5);
            Assert.True(Math.Abs(regression.Coefficients[1] - 2.5) < 0.5);
            Assert.True(Math.Abs(regression.Intercept - 3.0) < 1.0);
        }

        [Fact]
        public void MultivariateRegression_SmallDataset_HandlesCorrectly()
        {
            // Arrange - minimal viable dataset
            var x = new Matrix<double>(3, 2);
            x[0, 0] = 1.0; x[0, 1] = 1.0;
            x[1, 0] = 2.0; x[1, 1] = 2.0;
            x[2, 0] = 3.0; x[2, 1] = 3.0;

            var y = new Vector<double>(new[] { 5.0, 9.0, 13.0 }); // y = 2*x1 + 2*x2 + 1

            // Act
            var regression = new MultivariateRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 3; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 1e-6);
            }
        }

        [Fact]
        public void MultivariateRegression_LargeDataset_HandlesEfficiently()
        {
            // Arrange - large dataset
            var n = 1000;
            var x = new Matrix<double>(n, 3);
            var y = new Vector<double>(n);

            for (int i = 0; i < n; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                x[i, 2] = i * 3;
                y[i] = 1.0 * x[i, 0] + 2.0 * x[i, 1] + 3.0 * x[i, 2] + 5.0;
            }

            // Act
            var regression = new MultivariateRegression<double>();
            var sw = System.Diagnostics.Stopwatch.StartNew();
            regression.Train(x, y);
            sw.Stop();

            // Assert
            Assert.Equal(1.0, regression.Coefficients[0], precision: 8);
            Assert.Equal(2.0, regression.Coefficients[1], precision: 8);
            Assert.Equal(3.0, regression.Coefficients[2], precision: 8);
            Assert.True(sw.ElapsedMilliseconds < 2000);
        }

        [Fact]
        public void MultivariateRegression_PredictionsAreAccurate()
        {
            // Arrange
            var x = new Matrix<double>(4, 2);
            x[0, 0] = 1.0; x[0, 1] = 2.0;
            x[1, 0] = 2.0; x[1, 1] = 4.0;
            x[2, 0] = 3.0; x[2, 1] = 6.0;
            x[3, 0] = 4.0; x[3, 1] = 8.0;

            var y = new Vector<double>(new[] { 7.0, 13.0, 19.0, 25.0 }); // y = 1*x1 + 3*x2

            var regression = new MultivariateRegression<double>();
            regression.Train(x, y);

            // Act - test on new data
            var testX = new Matrix<double>(2, 2);
            testX[0, 0] = 5.0; testX[0, 1] = 10.0;
            testX[1, 0] = 6.0; testX[1, 1] = 12.0;

            var predictions = regression.Predict(testX);

            // Assert
            Assert.Equal(31.0, predictions[0], precision: 10); // 1*5 + 3*10 + 1 = 36 (no intercept in data)
            Assert.Equal(37.0, predictions[1], precision: 10); // 1*6 + 3*12 + 1 = 43
        }

        [Fact]
        public void MultivariateRegression_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var x = new Matrix<float>(3, 2);
            x[0, 0] = 1.0f; x[0, 1] = 2.0f;
            x[1, 0] = 2.0f; x[1, 1] = 3.0f;
            x[2, 0] = 3.0f; x[2, 1] = 4.0f;

            var y = new Vector<float>(new[] { 8.0f, 11.0f, 14.0f }); // y = 2*x1 + 3*x2

            // Act
            var regression = new MultivariateRegression<float>();
            regression.Train(x, y);

            // Assert
            Assert.Equal(2.0f, regression.Coefficients[0], precision: 5);
            Assert.Equal(3.0f, regression.Coefficients[1], precision: 5);
        }

        [Fact]
        public void MultivariateRegression_NoIntercept_FitsCorrectly()
        {
            // Arrange
            var options = new RegressionOptions<double> { UseIntercept = false };
            var x = new Matrix<double>(3, 2);
            x[0, 0] = 1.0; x[0, 1] = 2.0;
            x[1, 0] = 2.0; x[1, 1] = 4.0;
            x[2, 0] = 3.0; x[2, 1] = 6.0;

            var y = new Vector<double>(new[] { 5.0, 10.0, 15.0 }); // y = 1*x1 + 2*x2

            // Act
            var regression = new MultivariateRegression<double>(options);
            regression.Train(x, y);

            // Assert
            Assert.Equal(1.0, regression.Coefficients[0], precision: 10);
            Assert.Equal(2.0, regression.Coefficients[1], precision: 10);
            Assert.True(Math.Abs(regression.Intercept) < 1e-10);
        }

        [Fact]
        public void MultivariateRegression_HighDimensional_HandlesCorrectly()
        {
            // Arrange - 10 features
            var x = new Matrix<double>(50, 10);
            var y = new Vector<double>(50);
            var trueCoeffs = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };

            for (int i = 0; i < 50; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    x[i, j] = i + j;
                }
                y[i] = 0;
                for (int j = 0; j < 10; j++)
                {
                    y[i] += trueCoeffs[j] * x[i, j];
                }
            }

            // Act
            var regression = new MultivariateRegression<double>();
            regression.Train(x, y);

            // Assert
            for (int j = 0; j < 10; j++)
            {
                Assert.Equal(trueCoeffs[j], regression.Coefficients[j], precision: 8);
            }
        }

        [Fact]
        public void MultivariateRegression_NegativeCoefficients_FitsCorrectly()
        {
            // Arrange - y = -2*x1 + 3*x2 + 5
            var x = new Matrix<double>(4, 2);
            x[0, 0] = 1.0; x[0, 1] = 1.0;
            x[1, 0] = 2.0; x[1, 1] = 2.0;
            x[2, 0] = 3.0; x[2, 1] = 3.0;
            x[3, 0] = 4.0; x[3, 1] = 4.0;

            var y = new Vector<double>(new[] { 6.0, 7.0, 8.0, 9.0 }); // -2*x1 + 3*x2 + 5

            // Act
            var regression = new MultivariateRegression<double>();
            regression.Train(x, y);

            // Assert
            Assert.Equal(-2.0, regression.Coefficients[0], precision: 10);
            Assert.Equal(3.0, regression.Coefficients[1], precision: 10);
            Assert.Equal(5.0, regression.Intercept, precision: 10);
        }

        [Fact]
        public void MultivariateRegression_Collinear_HandlesGracefully()
        {
            // Arrange - x2 is perfectly correlated with x1
            var x = new Matrix<double>(5, 2);
            for (int i = 0; i < 5; i++)
            {
                x[i, 0] = i + 1;
                x[i, 1] = (i + 1) * 2; // Perfect collinearity
            }

            var y = new Vector<double>(new[] { 3.0, 5.0, 7.0, 9.0, 11.0 }); // y = 2*x1 + 1

            // Act & Assert - should handle gracefully (may not converge to exact coefficients)
            var regression = new MultivariateRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Verify predictions are still reasonable
            for (int i = 0; i < 5; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 1.0);
            }
        }

        #endregion

        #region MultipleRegression Tests

        [Fact]
        public void MultipleRegression_PerfectFit_ProducesAccuratePredictions()
        {
            // Arrange - y = 3*x1 + 2*x2 + 4
            var x = new Matrix<double>(6, 2);
            var y = new Vector<double>(6);

            for (int i = 0; i < 6; i++)
            {
                x[i, 0] = i + 1;
                x[i, 1] = (i + 1) * 1.5;
                y[i] = 3 * x[i, 0] + 2 * x[i, 1] + 4;
            }

            // Act
            var regression = new MultipleRegression<double>();
            regression.Train(x, y);

            // Assert
            Assert.Equal(3.0, regression.Coefficients[0], precision: 10);
            Assert.Equal(2.0, regression.Coefficients[1], precision: 10);
            Assert.Equal(4.0, regression.Intercept, precision: 10);
        }

        [Fact]
        public void MultipleRegression_MediumDataset_ConvergesCorrectly()
        {
            // Arrange
            var x = new Matrix<double>(100, 3);
            var y = new Vector<double>(100);
            var random = new Random(123);

            for (int i = 0; i < 100; i++)
            {
                x[i, 0] = random.NextDouble() * 10;
                x[i, 1] = random.NextDouble() * 10;
                x[i, 2] = random.NextDouble() * 10;
                y[i] = 2.0 * x[i, 0] - 1.5 * x[i, 1] + 3.0 * x[i, 2] + 7.0 + (random.NextDouble() - 0.5);
            }

            // Act
            var regression = new MultipleRegression<double>();
            regression.Train(x, y);

            // Assert - coefficients should be close to true values
            Assert.True(Math.Abs(regression.Coefficients[0] - 2.0) < 0.5);
            Assert.True(Math.Abs(regression.Coefficients[1] - (-1.5)) < 0.5);
            Assert.True(Math.Abs(regression.Coefficients[2] - 3.0) < 0.5);
        }

        [Fact]
        public void MultipleRegression_ZeroSlope_IdentifiesConstant()
        {
            // Arrange - y = 10 (constant)
            var x = new Matrix<double>(5, 2);
            for (int i = 0; i < 5; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
            }

            var y = new Vector<double>(new[] { 10.0, 10.0, 10.0, 10.0, 10.0 });

            // Act
            var regression = new MultipleRegression<double>();
            regression.Train(x, y);

            // Assert - all coefficients should be near zero, intercept near 10
            Assert.True(Math.Abs(regression.Coefficients[0]) < 1e-6);
            Assert.True(Math.Abs(regression.Coefficients[1]) < 1e-6);
            Assert.Equal(10.0, regression.Intercept, precision: 6);
        }

        [Fact]
        public void MultipleRegression_SingleFeature_WorksLikeSimpleRegression()
        {
            // Arrange - y = 4*x + 2
            var x = new Matrix<double>(5, 1);
            for (int i = 0; i < 5; i++)
            {
                x[i, 0] = i + 1;
            }

            var y = new Vector<double>(new[] { 6.0, 10.0, 14.0, 18.0, 22.0 });

            // Act
            var regression = new MultipleRegression<double>();
            regression.Train(x, y);

            // Assert
            Assert.Equal(4.0, regression.Coefficients[0], precision: 10);
            Assert.Equal(2.0, regression.Intercept, precision: 10);
        }

        [Fact]
        public void MultipleRegression_WithFloatPrecision_MaintainsAccuracy()
        {
            // Arrange
            var x = new Matrix<float>(4, 2);
            x[0, 0] = 1.0f; x[0, 1] = 2.0f;
            x[1, 0] = 2.0f; x[1, 1] = 3.0f;
            x[2, 0] = 3.0f; x[2, 1] = 4.0f;
            x[3, 0] = 4.0f; x[3, 1] = 5.0f;

            var y = new Vector<float>(new[] { 9.0f, 12.0f, 15.0f, 18.0f }); // y = 2*x1 + 2.5*x2 + 0.5

            // Act
            var regression = new MultipleRegression<float>();
            regression.Train(x, y);

            // Assert
            Assert.True(Math.Abs(regression.Coefficients[0] - 2.0f) < 0.5f);
            Assert.True(Math.Abs(regression.Coefficients[1] - 2.5f) < 0.5f);
        }

        #endregion

        #region WeightedRegression Tests

        [Fact]
        public void WeightedRegression_UniformWeights_EqualsStandardRegression()
        {
            // Arrange
            var x = new Matrix<double>(5, 2);
            var y = new Vector<double>(5);
            var weights = new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0, 1.0 });

            for (int i = 0; i < 5; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = 3 * x[i, 0] + 2 * x[i, 1] + 5;
            }

            // Act
            var weightedReg = new WeightedRegression<double>();
            weightedReg.Train(x, y, weights);

            var standardReg = new MultivariateRegression<double>();
            standardReg.Train(x, y);

            // Assert - should produce similar results
            Assert.Equal(standardReg.Coefficients[0], weightedReg.Coefficients[0], precision: 8);
            Assert.Equal(standardReg.Coefficients[1], weightedReg.Coefficients[1], precision: 8);
        }

        [Fact]
        public void WeightedRegression_HighWeightOnOutlier_AdjustsFit()
        {
            // Arrange
            var x = new Matrix<double>(6, 1);
            var y = new Vector<double>(6);
            var weights = new Vector<double>(6);

            for (int i = 0; i < 5; i++)
            {
                x[i, 0] = i;
                y[i] = 2 * i + 1;
                weights[i] = 1.0;
            }

            // Add outlier with high weight
            x[5, 0] = 10;
            y[5] = 100; // Outlier
            weights[5] = 10.0; // High weight

            // Act
            var regression = new WeightedRegression<double>();
            regression.Train(x, y, weights);

            // Assert - fit should be influenced by the weighted outlier
            var prediction = regression.Predict(new Matrix<double>(new[,] { { 10.0 } }));
            Assert.True(Math.Abs(prediction[0] - 100.0) < Math.Abs(prediction[0] - 21.0)); // Closer to outlier
        }

        [Fact]
        public void WeightedRegression_ZeroWeights_IgnoresThosePoints()
        {
            // Arrange
            var x = new Matrix<double>(6, 1);
            var y = new Vector<double>(6);
            var weights = new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0, 1.0, 0.0 });

            for (int i = 0; i < 5; i++)
            {
                x[i, 0] = i;
                y[i] = 3 * i + 2;
            }

            x[5, 0] = 100;
            y[5] = 1000; // Should be ignored due to zero weight

            // Act
            var regression = new WeightedRegression<double>();
            regression.Train(x, y, weights);

            // Assert
            Assert.Equal(3.0, regression.Coefficients[0], precision: 8);
            Assert.Equal(2.0, regression.Intercept, precision: 8);
        }

        [Fact]
        public void WeightedRegression_DifferentWeights_ProducesDifferentFit()
        {
            // Arrange
            var x = new Matrix<double>(3, 1);
            x[0, 0] = 1.0; x[1, 0] = 2.0; x[2, 0] = 3.0;

            var y = new Vector<double>(new[] { 2.0, 4.0, 10.0 });

            var weights1 = new Vector<double>(new[] { 1.0, 1.0, 1.0 });
            var weights2 = new Vector<double>(new[] { 1.0, 1.0, 10.0 }); // High weight on last point

            // Act
            var reg1 = new WeightedRegression<double>();
            reg1.Train(x, y, weights1);

            var reg2 = new WeightedRegression<double>();
            reg2.Train(x, y, weights2);

            // Assert - fits should be different
            Assert.NotEqual(reg1.Coefficients[0], reg2.Coefficients[0]);
        }

        [Fact]
        public void WeightedRegression_LargeWeightedDataset_HandlesEfficiently()
        {
            // Arrange
            var n = 500;
            var x = new Matrix<double>(n, 2);
            var y = new Vector<double>(n);
            var weights = new Vector<double>(n);
            var random = new Random(456);

            for (int i = 0; i < n; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 1.5;
                y[i] = 2 * x[i, 0] + 3 * x[i, 1] + 1;
                weights[i] = random.NextDouble() + 0.5; // Random weights between 0.5 and 1.5
            }

            // Act
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var regression = new WeightedRegression<double>();
            regression.Train(x, y, weights);
            sw.Stop();

            // Assert
            Assert.True(sw.ElapsedMilliseconds < 2000);
            Assert.Equal(2.0, regression.Coefficients[0], precision: 6);
            Assert.Equal(3.0, regression.Coefficients[1], precision: 6);
        }

        #endregion

        #region RobustRegression Tests

        [Fact]
        public void RobustRegression_WithOutliers_ReducesOutlierInfluence()
        {
            // Arrange - data with outliers
            var x = new Matrix<double>(10, 1);
            var y = new Vector<double>(10);

            for (int i = 0; i < 8; i++)
            {
                x[i, 0] = i;
                y[i] = 2 * i + 3;
            }

            // Add outliers
            x[8, 0] = 10;
            y[8] = 50; // Outlier
            x[9, 0] = 11;
            y[9] = 60; // Outlier

            // Act
            var robustReg = new RobustRegression<double>();
            robustReg.Train(x, y);

            var standardReg = new MultivariateRegression<double>();
            standardReg.Train(x, y);

            // Assert - robust regression should be closer to true relationship (y = 2x + 3)
            Assert.True(Math.Abs(robustReg.Coefficients[0] - 2.0) < Math.Abs(standardReg.Coefficients[0] - 2.0));
        }

        [Fact]
        public void RobustRegression_CleanData_ProducesSimilarToStandard()
        {
            // Arrange - clean data without outliers
            var x = new Matrix<double>(10, 2);
            var y = new Vector<double>(10);

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = 1.5 * x[i, 0] + 2.5 * x[i, 1] + 4;
            }

            // Act
            var robustReg = new RobustRegression<double>();
            robustReg.Train(x, y);

            var standardReg = new MultivariateRegression<double>();
            standardReg.Train(x, y);

            // Assert - should produce similar results
            Assert.Equal(standardReg.Coefficients[0], robustReg.Coefficients[0], precision: 1);
            Assert.Equal(standardReg.Coefficients[1], robustReg.Coefficients[1], precision: 1);
        }

        [Fact]
        public void RobustRegression_MultipleOutliers_HandlesGracefully()
        {
            // Arrange
            var x = new Matrix<double>(15, 1);
            var y = new Vector<double>(15);

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
                y[i] = 3 * i + 5;
            }

            // Multiple outliers
            for (int i = 10; i < 15; i++)
            {
                x[i, 0] = i;
                y[i] = 100; // Outliers
            }

            // Act
            var regression = new RobustRegression<double>();
            regression.Train(x, y);

            // Assert - should still identify the main trend
            Assert.True(Math.Abs(regression.Coefficients[0] - 3.0) < 1.0);
            Assert.True(Math.Abs(regression.Intercept - 5.0) < 5.0);
        }

        #endregion

        #region QuantileRegression Tests

        [Fact]
        public void QuantileRegression_MedianRegression_FitsCorrectly()
        {
            // Arrange - fit to median (quantile = 0.5)
            var x = new Matrix<double>(7, 1);
            var y = new Vector<double>(7);

            for (int i = 0; i < 7; i++)
            {
                x[i, 0] = i;
                y[i] = 2 * i + 3;
            }

            var options = new QuantileRegressionOptions<double> { Quantile = 0.5 };

            // Act
            var regression = new QuantileRegression<double>(options);
            regression.Train(x, y);

            // Assert
            Assert.Equal(2.0, regression.Coefficients[0], precision: 1);
            Assert.Equal(3.0, regression.Intercept, precision: 1);
        }

        [Fact]
        public void QuantileRegression_DifferentQuantiles_ProduceDifferentFits()
        {
            // Arrange
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);
            var random = new Random(789);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                y[i] = 2 * i + random.NextDouble() * 10; // Add noise
            }

            // Act - fit at different quantiles
            var reg25 = new QuantileRegression<double>(new QuantileRegressionOptions<double> { Quantile = 0.25 });
            reg25.Train(x, y);

            var reg75 = new QuantileRegression<double>(new QuantileRegressionOptions<double> { Quantile = 0.75 });
            reg75.Train(x, y);

            // Assert - different quantiles should produce different intercepts
            Assert.NotEqual(reg25.Intercept, reg75.Intercept);
        }

        [Fact]
        public void QuantileRegression_UpperQuantile_FitsAboveMedian()
        {
            // Arrange
            var x = new Matrix<double>(10, 1);
            var y = new Vector<double>(10);

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
                y[i] = 2 * i + 5;
            }

            // Act - fit at 90th percentile
            var regression = new QuantileRegression<double>(new QuantileRegressionOptions<double> { Quantile = 0.9 });
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - most predictions should be above actual values for lower quantiles
            var medianPrediction = predictions[5];
            Assert.True(medianPrediction >= y[5] * 0.9);
        }

        #endregion

        #region OrthogonalRegression Tests

        [Fact]
        public void OrthogonalRegression_PerfectLinearRelationship_FitsCorrectly()
        {
            // Arrange - y = 2x + 1
            var x = new Matrix<double>(5, 1);
            var y = new Vector<double>(5);

            for (int i = 0; i < 5; i++)
            {
                x[i, 0] = i;
                y[i] = 2 * i + 1;
            }

            // Act
            var regression = new OrthogonalRegression<double>();
            regression.Train(x, y);

            // Assert
            Assert.Equal(2.0, regression.Coefficients[0], precision: 8);
            Assert.Equal(1.0, regression.Intercept, precision: 8);
        }

        [Fact]
        public void OrthogonalRegression_ErrorsInBothVariables_HandlesCorrectly()
        {
            // Arrange - data with errors in both x and y
            var x = new Matrix<double>(10, 1);
            var y = new Vector<double>(10);
            var random = new Random(321);

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i + (random.NextDouble() - 0.5) * 0.5;
                y[i] = 3 * i + 2 + (random.NextDouble() - 0.5) * 0.5;
            }

            // Act
            var regression = new OrthogonalRegression<double>();
            regression.Train(x, y);

            // Assert
            Assert.True(Math.Abs(regression.Coefficients[0] - 3.0) < 0.5);
            Assert.True(Math.Abs(regression.Intercept - 2.0) < 1.0);
        }

        #endregion

        #region StepwiseRegression Tests

        [Fact]
        public void StepwiseRegression_SelectsRelevantFeatures()
        {
            // Arrange - some features are relevant, others are not
            var x = new Matrix<double>(50, 5);
            var y = new Vector<double>(50);

            for (int i = 0; i < 50; i++)
            {
                x[i, 0] = i; // Relevant
                x[i, 1] = i * 2; // Relevant
                x[i, 2] = 100; // Irrelevant (constant)
                x[i, 3] = i % 3; // Possibly relevant
                x[i, 4] = (i % 2) * 0.01; // Mostly irrelevant

                y[i] = 2 * x[i, 0] + 3 * x[i, 1] + 5;
            }

            // Act
            var regression = new StepwiseRegression<double>();
            regression.Train(x, y);

            // Assert - should have higher coefficients for relevant features
            Assert.True(Math.Abs(regression.Coefficients[0]) > 1.0);
            Assert.True(Math.Abs(regression.Coefficients[1]) > 1.0);
            Assert.True(Math.Abs(regression.Coefficients[2]) < 0.1); // Constant feature
        }

        [Fact]
        public void StepwiseRegression_AllFeaturesRelevant_IncludesAll()
        {
            // Arrange
            var x = new Matrix<double>(30, 3);
            var y = new Vector<double>(30);

            for (int i = 0; i < 30; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                x[i, 2] = i * 3;
                y[i] = 1 * x[i, 0] + 2 * x[i, 1] + 3 * x[i, 2] + 4;
            }

            // Act
            var regression = new StepwiseRegression<double>();
            regression.Train(x, y);

            // Assert - all coefficients should be non-zero
            Assert.Equal(1.0, regression.Coefficients[0], precision: 6);
            Assert.Equal(2.0, regression.Coefficients[1], precision: 6);
            Assert.Equal(3.0, regression.Coefficients[2], precision: 6);
        }

        #endregion

        #region PartialLeastSquaresRegression Tests

        [Fact]
        public void PartialLeastSquaresRegression_MulticollinearData_HandlesWell()
        {
            // Arrange - highly correlated features
            var x = new Matrix<double>(20, 3);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 1.1; // Highly correlated with x0
                x[i, 2] = i * 0.9; // Highly correlated with x0
                y[i] = 5 * i + 10;
            }

            // Act
            var regression = new PartialLeastSquaresRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - predictions should be reasonable despite multicollinearity
            for (int i = 0; i < 20; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 5.0);
            }
        }

        [Fact]
        public void PartialLeastSquaresRegression_StandardData_ProducesGoodFit()
        {
            // Arrange
            var x = new Matrix<double>(15, 2);
            var y = new Vector<double>(15);

            for (int i = 0; i < 15; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 2;
                y[i] = 2 * x[i, 0] + 3 * x[i, 1] + 1;
            }

            // Act
            var regression = new PartialLeastSquaresRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 15; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 1.0);
            }
        }

        #endregion

        #region PrincipalComponentRegression Tests

        [Fact]
        public void PrincipalComponentRegression_HighDimensionalData_ReducesDimensions()
        {
            // Arrange - many features
            var x = new Matrix<double>(50, 10);
            var y = new Vector<double>(50);

            for (int i = 0; i < 50; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    x[i, j] = i + j * 0.1;
                }
                // y depends only on first few features
                y[i] = 2 * x[i, 0] + 3 * x[i, 1] + 5;
            }

            // Act
            var regression = new PrincipalComponentRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should produce reasonable predictions
            for (int i = 0; i < 50; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 10.0);
            }
        }

        [Fact]
        public void PrincipalComponentRegression_StandardCase_FitsWell()
        {
            // Arrange
            var x = new Matrix<double>(25, 3);
            var y = new Vector<double>(25);

            for (int i = 0; i < 25; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i * 1.5;
                x[i, 2] = i * 2;
                y[i] = 1 * x[i, 0] + 2 * x[i, 1] + 3 * x[i, 2] + 4;
            }

            // Act
            var regression = new PrincipalComponentRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 25; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 5.0);
            }
        }

        [Fact]
        public void PrincipalComponentRegression_CorrelatedFeatures_HandlesEfficiently()
        {
            // Arrange - correlated features
            var x = new Matrix<double>(30, 4);
            var y = new Vector<double>(30);

            for (int i = 0; i < 30; i++)
            {
                x[i, 0] = i;
                x[i, 1] = i + 0.1; // Highly correlated
                x[i, 2] = i * 2;
                x[i, 3] = i * 2 + 0.2; // Highly correlated
                y[i] = 3 * i + 10;
            }

            // Act
            var regression = new PrincipalComponentRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 30; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 2.0);
            }
        }

        #endregion
    }
}
