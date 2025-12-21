using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.LossFunctions
{
    public class RootMeanSquaredErrorLossTests
    {
        [Fact]
        public void CalculateLoss_WithPerfectPredictions_ReturnsZero()
        {
            // Arrange
            var loss = new RootMeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            Assert.Equal(0.0, result, 10);
        }

        [Fact]
        public void CalculateLoss_WithDifferentPredictions_ReturnsCorrectValue()
        {
            // Arrange
            var loss = new RootMeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 2.0, 4.0, 6.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // MSE = ((2-1)^2 + (4-2)^2 + (6-3)^2) / 3 = (1 + 4 + 9) / 3 = 14 / 3 = 4.666...
            // RMSE = sqrt(4.666...) = 2.160...
            Assert.Equal(2.160246899469287, result, 10);
        }

        [Fact]
        public void CalculateLoss_WithNegativeValues_ReturnsCorrectValue()
        {
            // Arrange
            var loss = new RootMeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { -1.0, -2.0, -3.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // MSE = ((-1-1)^2 + (-2-2)^2 + (-3-3)^2) / 3 = (4 + 16 + 36) / 3 = 56 / 3 = 18.666...
            // RMSE = sqrt(18.666...) = 4.320...
            Assert.Equal(4.320493798938574, result, 10);
        }

        [Fact]
        public void CalculateLoss_WithSingleValue_ReturnsCorrectValue()
        {
            // Arrange
            var loss = new RootMeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 5.0 });
            var actual = new Vector<double>(new double[] { 3.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // MSE = (5-3)^2 / 1 = 4
            // RMSE = sqrt(4) = 2
            Assert.Equal(2.0, result, 10);
        }

        [Fact]
        public void CalculateLoss_WithDifferentLengths_ThrowsArgumentException()
        {
            // Arrange
            var loss = new RootMeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => loss.CalculateLoss(predicted, actual));
        }

        [Fact]
        public void CalculateDerivative_WithPerfectPredictions_ReturnsZero()
        {
            // Arrange
            var loss = new RootMeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = loss.CalculateDerivative(predicted, actual);

            // Assert
            Assert.All(result, item => Assert.Equal(0.0, item, 10));
        }

        [Fact]
        public void CalculateDerivative_WithDifferentPredictions_ReturnsCorrectValues()
        {
            // Arrange
            var loss = new RootMeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 2.0, 4.0, 6.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = loss.CalculateDerivative(predicted, actual);

            // Assert
            // Differences: [1, 2, 3]
            // RMSE = 2.160246899469287
            // Derivative = difference / (n * RMSE) = [1, 2, 3] / (3 * 2.160246899469287)
            Assert.Equal(0.15430334996209191, result[0], 10);
            Assert.Equal(0.30860669992418383, result[1], 10);
            Assert.Equal(0.46291004988627574, result[2], 10);
        }

        [Fact]
        public void CalculateDerivative_WithNegativeDifferences_ReturnsNegativeValues()
        {
            // Arrange
            var loss = new RootMeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new double[] { 2.0, 4.0, 6.0 });

            // Act
            var result = loss.CalculateDerivative(predicted, actual);

            // Assert
            // Differences: [-1, -2, -3]
            // All values should be negative
            Assert.True(result[0] < 0);
            Assert.True(result[1] < 0);
            Assert.True(result[2] < 0);
            Assert.Equal(-0.15430334996209191, result[0], 10);
            Assert.Equal(-0.30860669992418383, result[1], 10);
            Assert.Equal(-0.46291004988627574, result[2], 10);
        }

        [Fact]
        public void CalculateDerivative_WithDifferentLengths_ThrowsArgumentException()
        {
            // Arrange
            var loss = new RootMeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => loss.CalculateDerivative(predicted, actual));
        }

        [Fact]
        public void CalculateLoss_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var loss = new RootMeanSquaredErrorLoss<float>();
            var predicted = new Vector<float>(new float[] { 2.0f, 4.0f, 6.0f });
            var actual = new Vector<float>(new float[] { 1.0f, 2.0f, 3.0f });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // RMSE = sqrt(14/3) = 2.160...
            Assert.Equal(2.160246899469287f, result, 5);
        }

        [Fact]
        public void CalculateDerivative_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var loss = new RootMeanSquaredErrorLoss<float>();
            var predicted = new Vector<float>(new float[] { 2.0f, 4.0f, 6.0f });
            var actual = new Vector<float>(new float[] { 1.0f, 2.0f, 3.0f });

            // Act
            var result = loss.CalculateDerivative(predicted, actual);

            // Assert
            Assert.Equal(0.15430334996209191f, result[0], 5);
            Assert.Equal(0.30860669992418383f, result[1], 5);
            Assert.Equal(0.46291004988627574f, result[2], 5);
        }

        [Fact]
        public void CalculateLoss_WithLargeErrors_ReturnsSquareRootOfMSE()
        {
            // Arrange
            var loss = new RootMeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 10.0, 20.0, 30.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // MSE = ((10-1)^2 + (20-2)^2 + (30-3)^2) / 3 = (81 + 324 + 729) / 3 = 1134 / 3 = 378
            // RMSE = sqrt(378) = 19.442...
            Assert.Equal(19.44222209522358, result, 10);
        }

        [Fact]
        public void CalculateLoss_WithSmallErrors_ReturnsCorrectValue()
        {
            // Arrange
            var loss = new RootMeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 1.1, 2.1, 3.1 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // MSE = (0.1^2 + 0.1^2 + 0.1^2) / 3 = 0.03 / 3 = 0.01
            // RMSE = sqrt(0.01) = 0.1
            Assert.Equal(0.1, result, 10);
        }

        [Fact]
        public void CalculateLoss_ReturnsNonNegativeValue()
        {
            // Arrange
            var loss = new RootMeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { -10.0, 5.0, 15.0 });
            var actual = new Vector<double>(new double[] { 10.0, -5.0, -15.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // RMSE should always be non-negative
            Assert.True(result >= 0.0);
        }
    }
}
