using AiDotNet.LinearAlgebra;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Regression
{
    /// <summary>
    /// Integration tests for polynomial and spline regression models.
    /// Tests ensure correct fitting of non-linear relationships using polynomial and spline approaches.
    /// </summary>
    public class PolynomialAndSplineIntegrationTests
    {
        #region PolynomialRegression Tests

        [Fact]
        public void PolynomialRegression_QuadraticRelationship_FitsCorrectly()
        {
            // Arrange - y = x^2 + 2x + 3
            var x = new Matrix<double>(10, 1);
            var y = new Vector<double>(10);

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i - 5;
                y[i] = x[i, 0] * x[i, 0] + 2 * x[i, 0] + 3;
            }

            var options = new PolynomialRegressionOptions<double> { Degree = 2 };

            // Act
            var regression = new PolynomialRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should fit perfectly
            for (int i = 0; i < 10; i++)
            {
                Assert.Equal(y[i], predictions[i], precision: 6);
            }
        }

        [Fact]
        public void PolynomialRegression_CubicRelationship_FitsAccurately()
        {
            // Arrange - y = 2x^3 - 3x^2 + 4x + 5
            var x = new Matrix<double>(15, 1);
            var y = new Vector<double>(15);

            for (int i = 0; i < 15; i++)
            {
                double val = i - 7;
                x[i, 0] = val;
                y[i] = 2 * val * val * val - 3 * val * val + 4 * val + 5;
            }

            var options = new PolynomialRegressionOptions<double> { Degree = 3 };

            // Act
            var regression = new PolynomialRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 15; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 1e-4);
            }
        }

        [Fact]
        public void PolynomialRegression_DegreeOne_EqualsLinearRegression()
        {
            // Arrange - y = 3x + 2
            var x = new Matrix<double>(8, 1);
            var y = new Vector<double>(8);

            for (int i = 0; i < 8; i++)
            {
                x[i, 0] = i;
                y[i] = 3 * i + 2;
            }

            var options = new PolynomialRegressionOptions<double> { Degree = 1 };

            // Act
            var polyReg = new PolynomialRegression<double>(options);
            polyReg.Train(x, y);

            var linearReg = new MultivariateRegression<double>();
            linearReg.Train(x, y);

            // Assert - should produce similar results
            var polyPred = polyReg.Predict(x);
            var linearPred = linearReg.Predict(x);

            for (int i = 0; i < 8; i++)
            {
                Assert.Equal(linearPred[i], polyPred[i], precision: 6);
            }
        }

        [Fact]
        public void PolynomialRegression_HighDegree_FitsComplexCurve()
        {
            // Arrange - complex polynomial
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                double val = (i - 10) / 10.0;
                x[i, 0] = val;
                // Complex polynomial relationship
                y[i] = val * val * val * val - 2 * val * val * val + val * val + 3 * val + 1;
            }

            var options = new PolynomialRegressionOptions<double> { Degree = 4 };

            // Act
            var regression = new PolynomialRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 20; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 0.1);
            }
        }

        [Fact]
        public void PolynomialRegression_WithNoise_FitsReasonably()
        {
            // Arrange
            var x = new Matrix<double>(30, 1);
            var y = new Vector<double>(30);
            var random = new Random(42);

            for (int i = 0; i < 30; i++)
            {
                double val = i / 5.0;
                x[i, 0] = val;
                y[i] = val * val - 2 * val + 3 + (random.NextDouble() - 0.5) * 0.5;
            }

            var options = new PolynomialRegressionOptions<double> { Degree = 2 };

            // Act
            var regression = new PolynomialRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should fit reasonably well despite noise
            double totalError = 0;
            for (int i = 0; i < 30; i++)
            {
                totalError += Math.Abs(predictions[i] - y[i]);
            }
            double avgError = totalError / 30;
            Assert.True(avgError < 1.0);
        }

        [Fact]
        public void PolynomialRegression_SmallDataset_HandlesCorrectly()
        {
            // Arrange
            var x = new Matrix<double>(5, 1);
            var y = new Vector<double>(new[] { 1.0, 4.0, 9.0, 16.0, 25.0 }); // y = x^2

            for (int i = 0; i < 5; i++)
            {
                x[i, 0] = i + 1;
            }

            var options = new PolynomialRegressionOptions<double> { Degree = 2 };

            // Act
            var regression = new PolynomialRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 5; i++)
            {
                Assert.Equal(y[i], predictions[i], precision: 4);
            }
        }

        [Fact]
        public void PolynomialRegression_LargeDataset_HandlesEfficiently()
        {
            // Arrange
            var n = 500;
            var x = new Matrix<double>(n, 1);
            var y = new Vector<double>(n);

            for (int i = 0; i < n; i++)
            {
                double val = i / 100.0;
                x[i, 0] = val;
                y[i] = val * val + 2 * val + 1;
            }

            var options = new PolynomialRegressionOptions<double> { Degree = 2 };

            // Act
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var regression = new PolynomialRegression<double>(options);
            regression.Train(x, y);
            sw.Stop();

            // Assert
            Assert.True(sw.ElapsedMilliseconds < 3000);
            var predictions = regression.Predict(x);
            Assert.True(Math.Abs(predictions[100] - y[100]) < 0.01);
        }

        [Fact]
        public void PolynomialRegression_ExtrapolationWarning_StillWorks()
        {
            // Arrange - train on limited range
            var x = new Matrix<double>(10, 1);
            var y = new Vector<double>(10);

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
                y[i] = i * i;
            }

            var options = new PolynomialRegressionOptions<double> { Degree = 2 };
            var regression = new PolynomialRegression<double>(options);
            regression.Train(x, y);

            // Act - predict outside training range
            var testX = new Matrix<double>(1, 1);
            testX[0, 0] = 20; // Outside training range

            var prediction = regression.Predict(testX);

            // Assert - should extrapolate (though may be less accurate)
            Assert.True(prediction[0] > 100); // 20^2 = 400
        }

        [Fact]
        public void PolynomialRegression_MultipleFeatures_HandlesCorrectly()
        {
            // Arrange - y = x1^2 + x2^2
            var x = new Matrix<double>(15, 2);
            var y = new Vector<double>(15);

            for (int i = 0; i < 15; i++)
            {
                x[i, 0] = i / 5.0;
                x[i, 1] = (i + 1) / 5.0;
                y[i] = x[i, 0] * x[i, 0] + x[i, 1] * x[i, 1];
            }

            var options = new PolynomialRegressionOptions<double> { Degree = 2 };

            // Act
            var regression = new PolynomialRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 15; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 0.5);
            }
        }

        [Fact]
        public void PolynomialRegression_FloatPrecision_WorksCorrectly()
        {
            // Arrange
            var x = new Matrix<float>(8, 1);
            var y = new Vector<float>(8);

            for (int i = 0; i < 8; i++)
            {
                x[i, 0] = i;
                y[i] = i * i + 2 * i + 1;
            }

            var options = new PolynomialRegressionOptions<float> { Degree = 2 };

            // Act
            var regression = new PolynomialRegression<float>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 8; i++)
            {
                Assert.Equal(y[i], predictions[i], precision: 3);
            }
        }

        [Fact]
        public void PolynomialRegression_NegativeValues_HandlesCorrectly()
        {
            // Arrange - symmetric polynomial around zero
            var x = new Matrix<double>(11, 1);
            var y = new Vector<double>(11);

            for (int i = 0; i < 11; i++)
            {
                x[i, 0] = i - 5; // -5 to 5
                y[i] = x[i, 0] * x[i, 0] + 1; // Even function
            }

            var options = new PolynomialRegressionOptions<double> { Degree = 2 };

            // Act
            var regression = new PolynomialRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 11; i++)
            {
                Assert.Equal(y[i], predictions[i], precision: 5);
            }
        }

        [Fact]
        public void PolynomialRegression_OverfittingCheck_HighDegreeWithFewPoints()
        {
            // Arrange - few points with high degree can lead to overfitting
            var x = new Matrix<double>(6, 1);
            var y = new Vector<double>(new[] { 1.0, 2.0, 3.0, 2.5, 2.0, 1.5 });

            for (int i = 0; i < 6; i++)
            {
                x[i, 0] = i;
            }

            var options = new PolynomialRegressionOptions<double> { Degree = 5 };

            // Act
            var regression = new PolynomialRegression<double>(options);
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should fit training data very well (possibly too well)
            for (int i = 0; i < 6; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 0.5);
            }
        }

        #endregion

        #region SplineRegression Tests

        [Fact]
        public void SplineRegression_SmoothCurve_FitsCorrectly()
        {
            // Arrange - smooth curve
            var x = new Matrix<double>(20, 1);
            var y = new Vector<double>(20);

            for (int i = 0; i < 20; i++)
            {
                double val = i / 2.0;
                x[i, 0] = val;
                y[i] = Math.Sin(val);
            }

            // Act
            var regression = new SplineRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should fit reasonably well
            for (int i = 0; i < 20; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 0.5);
            }
        }

        [Fact]
        public void SplineRegression_LinearSegments_FitsWell()
        {
            // Arrange - piecewise linear function
            var x = new Matrix<double>(15, 1);
            var y = new Vector<double>(15);

            for (int i = 0; i < 15; i++)
            {
                x[i, 0] = i;
                if (i < 5)
                    y[i] = 2 * i;
                else if (i < 10)
                    y[i] = 10 + 3 * (i - 5);
                else
                    y[i] = 25 + 1 * (i - 10);
            }

            // Act
            var regression = new SplineRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 15; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 2.0);
            }
        }

        [Fact]
        public void SplineRegression_NonMonotonicData_HandlesCorrectly()
        {
            // Arrange - non-monotonic relationship
            var x = new Matrix<double>(12, 1);
            var y = new Vector<double>(new[] { 1.0, 3.0, 5.0, 7.0, 6.0, 4.0, 2.0, 3.0, 5.0, 7.0, 8.0, 9.0 });

            for (int i = 0; i < 12; i++)
            {
                x[i, 0] = i;
            }

            // Act
            var regression = new SplineRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should capture the pattern
            for (int i = 0; i < 12; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 3.0);
            }
        }

        [Fact]
        public void SplineRegression_SmallDataset_HandlesGracefully()
        {
            // Arrange
            var x = new Matrix<double>(5, 1);
            var y = new Vector<double>(new[] { 1.0, 4.0, 9.0, 16.0, 25.0 });

            for (int i = 0; i < 5; i++)
            {
                x[i, 0] = i + 1;
            }

            // Act
            var regression = new SplineRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 5; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 2.0);
            }
        }

        [Fact]
        public void SplineRegression_LargeDataset_HandlesEfficiently()
        {
            // Arrange
            var n = 500;
            var x = new Matrix<double>(n, 1);
            var y = new Vector<double>(n);

            for (int i = 0; i < n; i++)
            {
                double val = i / 50.0;
                x[i, 0] = val;
                y[i] = Math.Sin(val) + Math.Cos(val * 2);
            }

            // Act
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var regression = new SplineRegression<double>();
            regression.Train(x, y);
            sw.Stop();

            // Assert
            Assert.True(sw.ElapsedMilliseconds < 5000);
        }

        [Fact]
        public void SplineRegression_WithNoise_SmoothsAppropriately()
        {
            // Arrange
            var x = new Matrix<double>(30, 1);
            var y = new Vector<double>(30);
            var random = new Random(123);

            for (int i = 0; i < 30; i++)
            {
                x[i, 0] = i;
                y[i] = i * 2 + (random.NextDouble() - 0.5) * 5; // Linear with noise
            }

            // Act
            var regression = new SplineRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - spline should smooth the noise somewhat
            double totalError = 0;
            for (int i = 0; i < 30; i++)
            {
                totalError += Math.Abs(predictions[i] - y[i]);
            }
            Assert.True(totalError / 30 < 10.0);
        }

        [Fact]
        public void SplineRegression_InterpolationBetweenPoints_IsSmooth()
        {
            // Arrange - sparse training points
            var trainX = new Matrix<double>(5, 1);
            var trainY = new Vector<double>(5);

            for (int i = 0; i < 5; i++)
            {
                trainX[i, 0] = i * 2;
                trainY[i] = (i * 2) * (i * 2);
            }

            var regression = new SplineRegression<double>();
            regression.Train(trainX, trainY);

            // Act - predict at intermediate points
            var testX = new Matrix<double>(9, 1);
            for (int i = 0; i < 9; i++)
            {
                testX[i, 0] = i;
            }

            var predictions = regression.Predict(testX);

            // Assert - intermediate predictions should be reasonable
            for (int i = 0; i < 9; i++)
            {
                double expected = i * i;
                Assert.True(Math.Abs(predictions[i] - expected) < 5.0);
            }
        }

        [Fact]
        public void SplineRegression_DuplicateXValues_HandlesGracefully()
        {
            // Arrange - some duplicate x values
            var x = new Matrix<double>(8, 1);
            var y = new Vector<double>(new[] { 1.0, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0, 5.5 });

            x[0, 0] = 0; x[1, 0] = 1; x[2, 0] = 1; x[3, 0] = 2;
            x[4, 0] = 3; x[5, 0] = 3; x[6, 0] = 4; x[7, 0] = 5;

            // Act & Assert - should handle gracefully
            var regression = new SplineRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            Assert.NotNull(predictions);
            Assert.Equal(8, predictions.Length);
        }

        [Fact]
        public void SplineRegression_QuadraticData_ApproximatesWell()
        {
            // Arrange
            var x = new Matrix<double>(15, 1);
            var y = new Vector<double>(15);

            for (int i = 0; i < 15; i++)
            {
                x[i, 0] = i;
                y[i] = i * i - 2 * i + 3;
            }

            // Act
            var regression = new SplineRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 15; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 2.0);
            }
        }

        [Fact]
        public void SplineRegression_FloatType_WorksCorrectly()
        {
            // Arrange
            var x = new Matrix<float>(10, 1);
            var y = new Vector<float>(10);

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
                y[i] = i * i;
            }

            // Act
            var regression = new SplineRegression<float>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert
            for (int i = 0; i < 10; i++)
            {
                Assert.True(Math.Abs(predictions[i] - y[i]) < 3.0f);
            }
        }

        [Fact]
        public void SplineRegression_MonotonicIncreasing_PreservesMonotonicity()
        {
            // Arrange - monotonically increasing data
            var x = new Matrix<double>(10, 1);
            var y = new Vector<double>(10);

            for (int i = 0; i < 10; i++)
            {
                x[i, 0] = i;
                y[i] = Math.Sqrt(i + 1);
            }

            // Act
            var regression = new SplineRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - predictions should generally be increasing
            for (int i = 1; i < 10; i++)
            {
                Assert.True(predictions[i] >= predictions[i - 1] - 0.5); // Allow small violations
            }
        }

        [Fact]
        public void SplineRegression_ExponentialData_ApproximatesReasonably()
        {
            // Arrange
            var x = new Matrix<double>(12, 1);
            var y = new Vector<double>(12);

            for (int i = 0; i < 12; i++)
            {
                x[i, 0] = i;
                y[i] = Math.Exp(i / 5.0);
            }

            // Act
            var regression = new SplineRegression<double>();
            regression.Train(x, y);
            var predictions = regression.Predict(x);

            // Assert - should approximate reasonably (splines may struggle with exponential)
            double totalRelativeError = 0;
            for (int i = 0; i < 12; i++)
            {
                totalRelativeError += Math.Abs(predictions[i] - y[i]) / Math.Max(y[i], 1.0);
            }
            Assert.True(totalRelativeError / 12 < 0.5);
        }

        #endregion
    }
}
