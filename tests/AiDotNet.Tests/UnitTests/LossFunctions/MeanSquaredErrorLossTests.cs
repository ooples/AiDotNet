using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.LossFunctions
{
    public class MeanSquaredErrorLossTests
    {
        [Fact]
        public void CalculateLoss_WithPerfectPredictions_ReturnsZero()
        {
            // Arrange
            var loss = new MeanSquaredErrorLoss<double>();
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
            var loss = new MeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 2.0, 4.0, 6.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // MSE = ((2-1)^2 + (4-2)^2 + (6-3)^2) / 3 = (1 + 4 + 9) / 3 = 14 / 3 = 4.666...
            Assert.Equal(4.6666666666666667, result, 10);
        }

        [Fact]
        public void CalculateLoss_WithNegativeValues_ReturnsCorrectValue()
        {
            // Arrange
            var loss = new MeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { -1.0, -2.0, -3.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // MSE = ((-1-1)^2 + (-2-2)^2 + (-3-3)^2) / 3 = (4 + 16 + 36) / 3 = 56 / 3 = 18.666...
            Assert.Equal(18.666666666666667, result, 10);
        }

        [Fact]
        public void CalculateLoss_WithSingleValue_ReturnsCorrectValue()
        {
            // Arrange
            var loss = new MeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 5.0 });
            var actual = new Vector<double>(new double[] { 3.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // MSE = (5-3)^2 / 1 = 4
            Assert.Equal(4.0, result, 10);
        }

        [Fact]
        public void CalculateLoss_WithDifferentLengths_ThrowsArgumentException()
        {
            // Arrange
            var loss = new MeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => loss.CalculateLoss(predicted, actual));
        }

        [Fact]
        public void CalculateDerivative_WithPerfectPredictions_ReturnsZero()
        {
            // Arrange
            var loss = new MeanSquaredErrorLoss<double>();
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
            var loss = new MeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 2.0, 4.0, 6.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = loss.CalculateDerivative(predicted, actual);

            // Assert
            // Derivative = 2*(predicted-actual)/n = 2*[1,2,3]/3 = [2/3, 4/3, 6/3]
            Assert.Equal(0.6666666666666666, result[0], 10);
            Assert.Equal(1.3333333333333333, result[1], 10);
            Assert.Equal(2.0, result[2], 10);
        }

        [Fact]
        public void CalculateDerivative_WithNegativeDifferences_ReturnsNegativeValues()
        {
            // Arrange
            var loss = new MeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new double[] { 2.0, 4.0, 6.0 });

            // Act
            var result = loss.CalculateDerivative(predicted, actual);

            // Assert
            // Derivative = 2*(predicted-actual)/n = 2*[-1,-2,-3]/3 = [-2/3, -4/3, -6/3]
            Assert.Equal(-0.6666666666666666, result[0], 10);
            Assert.Equal(-1.3333333333333333, result[1], 10);
            Assert.Equal(-2.0, result[2], 10);
        }

        [Fact]
        public void CalculateDerivative_WithDifferentLengths_ThrowsArgumentException()
        {
            // Arrange
            var loss = new MeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => loss.CalculateDerivative(predicted, actual));
        }

        [Fact]
        public void CalculateLoss_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var loss = new MeanSquaredErrorLoss<float>();
            var predicted = new Vector<float>(new float[] { 2.0f, 4.0f, 6.0f });
            var actual = new Vector<float>(new float[] { 1.0f, 2.0f, 3.0f });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // MSE = ((2-1)^2 + (4-2)^2 + (6-3)^2) / 3 = (1 + 4 + 9) / 3 = 14 / 3 = 4.666...
            Assert.Equal(4.6666666666666667f, result, 5);
        }

        [Fact]
        public void CalculateDerivative_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var loss = new MeanSquaredErrorLoss<float>();
            var predicted = new Vector<float>(new float[] { 2.0f, 4.0f, 6.0f });
            var actual = new Vector<float>(new float[] { 1.0f, 2.0f, 3.0f });

            // Act
            var result = loss.CalculateDerivative(predicted, actual);

            // Assert
            // Derivative = 2*(predicted-actual)/n = 2*[1,2,3]/3
            Assert.Equal(0.6666666666666666f, result[0], 5);
            Assert.Equal(1.3333333333333333f, result[1], 5);
            Assert.Equal(2.0f, result[2], 5);
        }

        [Fact]
        public void CalculateLoss_WithLargeErrors_PenalizesHeavily()
        {
            // Arrange
            var loss = new MeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 10.0, 20.0, 30.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // MSE = ((10-1)^2 + (20-2)^2 + (30-3)^2) / 3 = (81 + 324 + 729) / 3 = 1134 / 3 = 378
            Assert.Equal(378.0, result, 10);
        }
    }
}
