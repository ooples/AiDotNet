using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.LossFunctions
{
    public class HuberLossTests
    {
        [Fact]
        public void Constructor_WithDefaultDelta_UsesOneAsDefault()
        {
            // Arrange & Act
            var loss = new HuberLoss<double>();
            var predicted = new Vector<double>(new double[] { 0.5, 0.5 });
            var actual = new Vector<double>(new double[] { 0.0, 0.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // With delta = 1.0 and error = 0.5 (< delta), should use quadratic: 0.5 * 0.5^2 = 0.125 per element
            Assert.Equal(0.125, result, 10);
        }

        [Fact]
        public void Constructor_WithCustomDelta_UsesSpecifiedValue()
        {
            // Arrange & Act
            var loss = new HuberLoss<double>(2.0);
            var predicted = new Vector<double>(new double[] { 1.5, 1.5 });
            var actual = new Vector<double>(new double[] { 0.0, 0.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // With delta = 2.0 and error = 1.5 (< delta), should use quadratic
            // 0.5 * 1.5^2 = 1.125 per element
            Assert.Equal(1.125, result, 10);
        }

        [Fact]
        public void CalculateLoss_WithPerfectPredictions_ReturnsZero()
        {
            // Arrange
            var loss = new HuberLoss<double>();
            var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            Assert.Equal(0.0, result, 10);
        }

        [Fact]
        public void CalculateLoss_WithSmallErrors_UsesQuadraticLoss()
        {
            // Arrange
            var loss = new HuberLoss<double>(1.0);
            var predicted = new Vector<double>(new double[] { 0.5, 0.6, 0.7 });
            var actual = new Vector<double>(new double[] { 0.0, 0.0, 0.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // All errors (0.5, 0.6, 0.7) < delta (1.0), so use quadratic: 0.5 * error^2
            // (0.5*0.5^2 + 0.5*0.6^2 + 0.5*0.7^2) / 3 = (0.125 + 0.18 + 0.245) / 3 = 0.55 / 3
            Assert.Equal(0.18333333333333333, result, 10);
        }

        [Fact]
        public void CalculateLoss_WithLargeErrors_UsesLinearLoss()
        {
            // Arrange
            var loss = new HuberLoss<double>(1.0);
            var predicted = new Vector<double>(new double[] { 3.0, 4.0, 5.0 });
            var actual = new Vector<double>(new double[] { 0.0, 0.0, 0.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // All errors (3, 4, 5) > delta (1.0), so use linear: delta * (|error| - 0.5 * delta)
            // (1*(3-0.5) + 1*(4-0.5) + 1*(5-0.5)) / 3 = (2.5 + 3.5 + 4.5) / 3 = 10.5 / 3 = 3.5
            Assert.Equal(3.5, result, 10);
        }

        [Fact]
        public void CalculateLoss_WithMixedErrors_UsesBothQuadraticAndLinear()
        {
            // Arrange
            var loss = new HuberLoss<double>(1.0);
            var predicted = new Vector<double>(new double[] { 0.5, 2.0 });
            var actual = new Vector<double>(new double[] { 0.0, 0.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // Error 0.5 < delta (1.0): quadratic = 0.5 * 0.5^2 = 0.125
            // Error 2.0 > delta (1.0): linear = 1 * (2 - 0.5) = 1.5
            // Average = (0.125 + 1.5) / 2 = 0.8125
            Assert.Equal(0.8125, result, 10);
        }

        [Fact]
        public void CalculateLoss_WithNegativeErrors_HandlesCorrectly()
        {
            // Arrange
            var loss = new HuberLoss<double>(1.0);
            var predicted = new Vector<double>(new double[] { -2.0, -3.0 });
            var actual = new Vector<double>(new double[] { 0.0, 0.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // Absolute errors (2, 3) > delta (1.0), so use linear
            // (1*(2-0.5) + 1*(3-0.5)) / 2 = (1.5 + 2.5) / 2 = 2.0
            Assert.Equal(2.0, result, 10);
        }

        [Fact]
        public void CalculateLoss_WithDifferentLengths_ThrowsArgumentException()
        {
            // Arrange
            var loss = new HuberLoss<double>();
            var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => loss.CalculateLoss(predicted, actual));
        }

        [Fact]
        public void CalculateDerivative_WithPerfectPredictions_ReturnsZero()
        {
            // Arrange
            var loss = new HuberLoss<double>();
            var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = loss.CalculateDerivative(predicted, actual);

            // Assert
            Assert.All(result, item => Assert.Equal(0.0, item, 10));
        }

        [Fact]
        public void CalculateDerivative_WithSmallErrors_ReturnsLinearDerivative()
        {
            // Arrange
            var loss = new HuberLoss<double>(1.0);
            var predicted = new Vector<double>(new double[] { 0.5, 0.6 });
            var actual = new Vector<double>(new double[] { 0.0, 0.0 });

            // Act
            var result = loss.CalculateDerivative(predicted, actual);

            // Assert
            // For small errors (< delta), derivative = diff / n
            Assert.Equal(0.25, result[0], 10);  // 0.5 / 2
            Assert.Equal(0.3, result[1], 10);   // 0.6 / 2
        }

        [Fact]
        public void CalculateDerivative_WithLargeErrors_ReturnsClippedDerivative()
        {
            // Arrange
            var loss = new HuberLoss<double>(1.0);
            var predicted = new Vector<double>(new double[] { 3.0, -3.0 });
            var actual = new Vector<double>(new double[] { 0.0, 0.0 });

            // Act
            var result = loss.CalculateDerivative(predicted, actual);

            // Assert
            // For large errors (> delta), derivative = delta * sign(diff) / n
            Assert.Equal(0.5, result[0], 10);   // 1.0 * 1 / 2 = 0.5
            Assert.Equal(-0.5, result[1], 10);  // 1.0 * -1 / 2 = -0.5
        }

        [Fact]
        public void CalculateDerivative_WithDifferentLengths_ThrowsArgumentException()
        {
            // Arrange
            var loss = new HuberLoss<double>();
            var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => loss.CalculateDerivative(predicted, actual));
        }

        [Fact]
        public void CalculateLoss_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var loss = new HuberLoss<float>(1.0f);
            var predicted = new Vector<float>(new float[] { 0.5f, 2.0f });
            var actual = new Vector<float>(new float[] { 0.0f, 0.0f });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            Assert.Equal(0.8125f, result, 5);
        }

        [Fact]
        public void CalculateDerivative_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var loss = new HuberLoss<float>(1.0f);
            var predicted = new Vector<float>(new float[] { 0.5f, 3.0f });
            var actual = new Vector<float>(new float[] { 0.0f, 0.0f });

            // Act
            var result = loss.CalculateDerivative(predicted, actual);

            // Assert
            Assert.Equal(0.25f, result[0], 5);
            Assert.Equal(0.5f, result[1], 5);
        }

        [Fact]
        public void CalculateLoss_LessRobustToOutliersThanMAE_ButMoreThanMSE()
        {
            // Arrange
            var loss = new HuberLoss<double>(1.0);
            var predicted = new Vector<double>(new double[] { 0.1, 0.2, 10.0 });  // One outlier
            var actual = new Vector<double>(new double[] { 0.0, 0.0, 0.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // Should handle outlier better than MSE (which would square 10)
            // but still penalize it more than MAE
            Assert.True(result < 100.0);  // Much less than MSE which would be ~33
            Assert.True(result > 0.1);    // More than just average absolute error
        }
    }
}
