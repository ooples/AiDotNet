using AiDotNet.LinearAlgebra;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Regression
{
    /// <summary>
    /// Integration tests for SimpleRegression (linear regression) with mathematically verified results.
    /// Tests ensure the regression correctly fits known linear relationships.
    /// </summary>
    public class SimpleRegressionIntegrationTests
    {
        [Fact]
        public void SimpleRegression_PerfectLinearRelationship_FitsCorrectly()
        {
            // Arrange - Perfect linear relationship: y = 2x + 1
            var x = new Vector<double>(5);
            x[0] = 1.0; x[1] = 2.0; x[2] = 3.0; x[3] = 4.0; x[4] = 5.0;

            var y = new Vector<double>(5);
            y[0] = 3.0; y[1] = 5.0; y[2] = 7.0; y[3] = 9.0; y[4] = 11.0; // y = 2x + 1

            // Act
            var regression = new SimpleRegression<double>();
            regression.Fit(x, y);

            // Assert - Should find slope = 2.0, intercept = 1.0
            Assert.Equal(2.0, regression.Slope, precision: 10);
            Assert.Equal(1.0, regression.Intercept, precision: 10);

            // Verify predictions
            var prediction = regression.Predict(new Vector<double>(new[] { 10.0 }));
            Assert.Equal(21.0, prediction[0], precision: 10); // 2 * 10 + 1 = 21
        }

        [Fact]
        public void SimpleRegression_RealWorldData_FitsReasonably()
        {
            // Arrange - Realistic data with some noise
            var x = new Vector<double>(10);
            x[0] = 1.0; x[1] = 2.0; x[2] = 3.0; x[3] = 4.0; x[4] = 5.0;
            x[5] = 6.0; x[6] = 7.0; x[7] = 8.0; x[8] = 9.0; x[9] = 10.0;

            var y = new Vector<double>(10);
            // Approximately y = 3x + 2, with small noise
            y[0] = 5.1; y[1] = 8.2; y[2] = 11.0; y[3] = 13.9; y[4] = 17.1;
            y[5] = 20.0; y[6] = 22.8; y[7] = 26.2; y[8] = 29.0; y[9] = 32.1;

            // Act
            var regression = new SimpleRegression<double>();
            regression.Fit(x, y);

            // Assert - Should be close to slope = 3.0, intercept = 2.0
            Assert.True(Math.Abs(regression.Slope - 3.0) < 0.2);
            Assert.True(Math.Abs(regression.Intercept - 2.0) < 0.5);

            // R-squared should be very high (close to 1.0) for good fit
            var rSquared = regression.RSquared;
            Assert.True(rSquared > 0.99);
        }

        [Fact]
        public void SimpleRegression_NegativeSlope_FitsCorrectly()
        {
            // Arrange - Negative linear relationship: y = -1.5x + 10
            var x = new Vector<double>(5);
            x[0] = 0.0; x[1] = 2.0; x[2] = 4.0; x[3] = 6.0; x[4] = 8.0;

            var y = new Vector<double>(5);
            y[0] = 10.0; y[1] = 7.0; y[2] = 4.0; y[3] = 1.0; y[4] = -2.0;

            // Act
            var regression = new SimpleRegression<double>();
            regression.Fit(x, y);

            // Assert
            Assert.Equal(-1.5, regression.Slope, precision: 10);
            Assert.Equal(10.0, regression.Intercept, precision: 10);
        }

        [Fact]
        public void SimpleRegression_ZeroIntercept_FitsCorrectly()
        {
            // Arrange - Line through origin: y = 4x
            var x = new Vector<double>(5);
            x[0] = 1.0; x[1] = 2.0; x[2] = 3.0; x[3] = 4.0; x[4] = 5.0;

            var y = new Vector<double>(5);
            y[0] = 4.0; y[1] = 8.0; y[2] = 12.0; y[3] = 16.0; y[4] = 20.0;

            // Act
            var regression = new SimpleRegression<double>();
            regression.Fit(x, y);

            // Assert
            Assert.Equal(4.0, regression.Slope, precision: 10);
            Assert.True(Math.Abs(regression.Intercept) < 1e-10); // Should be very close to 0
        }

        [Fact]
        public void SimpleRegression_MultiplePoints_PredictionsAreAccurate()
        {
            // Arrange
            var x = new Vector<double>(6);
            x[0] = 1.0; x[1] = 2.0; x[2] = 3.0; x[3] = 4.0; x[4] = 5.0; x[5] = 6.0;

            var y = new Vector<double>(6);
            y[0] = 2.5; y[1] = 5.0; y[2] = 7.5; y[3] = 10.0; y[4] = 12.5; y[5] = 15.0; // y = 2.5x

            // Act
            var regression = new SimpleRegression<double>();
            regression.Fit(x, y);

            // Test multiple predictions
            var testX = new Vector<double>(3);
            testX[0] = 7.0; testX[1] = 8.0; testX[2] = 9.0;

            var predictions = regression.Predict(testX);

            // Assert
            Assert.Equal(17.5, predictions[0], precision: 10); // 2.5 * 7
            Assert.Equal(20.0, predictions[1], precision: 10); // 2.5 * 8
            Assert.Equal(22.5, predictions[2], precision: 10); // 2.5 * 9
        }

        [Fact]
        public void SimpleRegression_StandardError_IsCalculatedCorrectly()
        {
            // Arrange - Data with known standard error
            var x = new Vector<double>(5);
            x[0] = 1.0; x[1] = 2.0; x[2] = 3.0; x[3] = 4.0; x[4] = 5.0;

            var y = new Vector<double>(5);
            y[0] = 2.0; y[1] = 4.0; y[2] = 6.0; y[3] = 8.0; y[4] = 10.0; // Perfect fit: y = 2x

            // Act
            var regression = new SimpleRegression<double>();
            regression.Fit(x, y);

            // Assert - Standard error should be very small for perfect fit
            var standardError = regression.StandardError;
            Assert.True(standardError < 1e-10);
        }

        [Fact]
        public void SimpleRegression_ResidualAnalysis_IsCorrect()
        {
            // Arrange
            var x = new Vector<double>(4);
            x[0] = 1.0; x[1] = 2.0; x[2] = 3.0; x[3] = 4.0;

            var y = new Vector<double>(4);
            y[0] = 3.0; y[1] = 5.5; y[2] = 7.0; y[3] = 9.5; // Approximately y = 2.5x + 0.5

            // Act
            var regression = new SimpleRegression<double>();
            regression.Fit(x, y);
            var residuals = regression.GetResiduals(x, y);

            // Assert - Sum of residuals should be close to zero
            var sumResiduals = 0.0;
            for (int i = 0; i < residuals.Length; i++)
            {
                sumResiduals += residuals[i];
            }
            Assert.True(Math.Abs(sumResiduals) < 1e-10);
        }

        [Fact]
        public void SimpleRegression_ConfidenceIntervals_AreReasonable()
        {
            // Arrange
            var x = new Vector<double>(20);
            var y = new Vector<double>(20);
            for (int i = 0; i < 20; i++)
            {
                x[i] = i + 1;
                y[i] = 2.0 * x[i] + 3.0; // y = 2x + 3
            }

            // Act
            var regression = new SimpleRegression<double>();
            regression.Fit(x, y);

            // Get 95% confidence interval for slope
            var (lowerBound, upperBound) = regression.GetSlopeConfidenceInterval(0.95);

            // Assert - True slope (2.0) should be within confidence interval
            Assert.True(lowerBound <= 2.0 && upperBound >= 2.0);
        }

        [Fact]
        public void SimpleRegression_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var x = new Vector<float>(5);
            x[0] = 1.0f; x[1] = 2.0f; x[2] = 3.0f; x[3] = 4.0f; x[4] = 5.0f;

            var y = new Vector<float>(5);
            y[0] = 3.0f; y[1] = 5.0f; y[2] = 7.0f; y[3] = 9.0f; y[4] = 11.0f; // y = 2x + 1

            // Act
            var regression = new SimpleRegression<float>();
            regression.Fit(x, y);

            // Assert
            Assert.Equal(2.0f, regression.Slope, precision: 6);
            Assert.Equal(1.0f, regression.Intercept, precision: 6);
        }

        [Fact]
        public void SimpleRegression_LargeDataset_HandlesEfficiently()
        {
            // Arrange - Large dataset
            var n = 1000;
            var x = new Vector<double>(n);
            var y = new Vector<double>(n);
            for (int i = 0; i < n; i++)
            {
                x[i] = i;
                y[i] = 1.5 * i + 2.0; // y = 1.5x + 2
            }

            // Act
            var regression = new SimpleRegression<double>();
            var sw = System.Diagnostics.Stopwatch.StartNew();
            regression.Fit(x, y);
            sw.Stop();

            // Assert - Should complete quickly and fit correctly
            Assert.Equal(1.5, regression.Slope, precision: 10);
            Assert.Equal(2.0, regression.Intercept, precision: 10);
            Assert.True(sw.ElapsedMilliseconds < 1000); // Should be very fast
        }
    }
}
