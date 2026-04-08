using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.LossFunctions
{
    public class MeanBiasErrorLossTests
    {
        [Fact]
        public void CalculateLoss_WithPerfectPredictions_ReturnsZero()
        {
            // Arrange
            var loss = new MeanBiasErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            Assert.Equal(0.0, result, 10);
        }

        [Fact]
        public void CalculateLoss_WithUnderPredictions_ReturnsPositiveValue()
        {
            // Arrange
            var loss = new MeanBiasErrorLoss<double>();
            // Predictions are all too low
            var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new double[] { 2.0, 4.0, 6.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // MBE = mean(actual - predicted) = mean([1, 2, 3]) = 2.0
            Assert.Equal(2.0, result, 10);
        }

        [Fact]
        public void CalculateLoss_WithOverPredictions_ReturnsNegativeValue()
        {
            // Arrange
            var loss = new MeanBiasErrorLoss<double>();
            // Predictions are all too high
            var predicted = new Vector<double>(new double[] { 2.0, 4.0, 6.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // MBE = mean(actual - predicted) = mean([-1, -2, -3]) = -2.0
            Assert.Equal(-2.0, result, 10);
        }

        [Fact]
        public void CalculateLoss_WithMixedErrors_ReturnsCorrectValue()
        {
            // Arrange
            var loss = new MeanBiasErrorLoss<double>();
            // Some over, some under predictions
            var predicted = new Vector<double>(new double[] { 1.0, 5.0, 3.0, 7.0 });
            var actual = new Vector<double>(new double[] { 2.0, 4.0, 4.0, 6.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // MBE = mean([1, -1, 1, -1]) = 0.0
            Assert.Equal(0.0, result, 10);
        }

        [Fact]
        public void CalculateLoss_WithSingleValue_ReturnsCorrectValue()
        {
            // Arrange
            var loss = new MeanBiasErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 3.0 });
            var actual = new Vector<double>(new double[] { 5.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // MBE = (5-3) / 1 = 2.0
            Assert.Equal(2.0, result, 10);
        }

        [Fact]
        public void CalculateLoss_WithNegativeValues_HandlesCorrectly()
        {
            // Arrange
            var loss = new MeanBiasErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { -1.0, -2.0, -3.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // MBE = mean([2, 4, 6]) = 4.0
            Assert.Equal(4.0, result, 10);
        }

        [Fact]
        public void CalculateLoss_WithDifferentLengths_ThrowsArgumentException()
        {
            // Arrange
            var loss = new MeanBiasErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => loss.CalculateLoss(predicted, actual));
        }

        [Fact]
        public void CalculateDerivative_ReturnsConstantNegativeValue()
        {
            // Arrange
            var loss = new MeanBiasErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 2.0, 4.0, 6.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = loss.CalculateDerivative(predicted, actual);

            // Assert
            // Derivative = -1/n = -1/3 = -0.3333... for all elements
            Assert.Equal(3, result.Length);
            Assert.All(result, item => Assert.Equal(-0.3333333333333333, item, 10));
        }

        [Fact]
        public void CalculateDerivative_WithPerfectPredictions_ReturnsConstantNegativeValue()
        {
            // Arrange
            var loss = new MeanBiasErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = loss.CalculateDerivative(predicted, actual);

            // Assert
            // Derivative is always -1/n regardless of prediction accuracy
            Assert.Equal(3, result.Length);
            Assert.All(result, item => Assert.Equal(-0.3333333333333333, item, 10));
        }

        [Fact]
        public void CalculateDerivative_WithSingleValue_ReturnsMinusOne()
        {
            // Arrange
            var loss = new MeanBiasErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 5.0 });
            var actual = new Vector<double>(new double[] { 3.0 });

            // Act
            var result = loss.CalculateDerivative(predicted, actual);

            // Assert
            // Derivative = -1/1 = -1.0
            Assert.Equal(1, result.Length);
            Assert.Equal(-1.0, result[0], 10);
        }

        [Fact]
        public void CalculateDerivative_WithDifferentLengths_ThrowsArgumentException()
        {
            // Arrange
            var loss = new MeanBiasErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => loss.CalculateDerivative(predicted, actual));
        }

        [Fact]
        public void CalculateLoss_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var loss = new MeanBiasErrorLoss<float>();
            var predicted = new Vector<float>(new float[] { 1.0f, 2.0f, 3.0f });
            var actual = new Vector<float>(new float[] { 2.0f, 4.0f, 6.0f });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // MBE = mean([1, 2, 3]) = 2.0
            Assert.Equal(2.0f, result, 5);
        }

        [Fact]
        public void CalculateDerivative_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var loss = new MeanBiasErrorLoss<float>();
            var predicted = new Vector<float>(new float[] { 2.0f, 4.0f, 6.0f });
            var actual = new Vector<float>(new float[] { 1.0f, 2.0f, 3.0f });

            // Act
            var result = loss.CalculateDerivative(predicted, actual);

            // Assert
            // Derivative = -1/3 for all elements
            Assert.All(result, item => Assert.Equal(-0.3333333333333333f, item, 5));
        }

        [Fact]
        public void CalculateLoss_WithSmallBias_ReturnsSmallValue()
        {
            // Arrange
            var loss = new MeanBiasErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 1.1, 2.1, 3.1 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // MBE = mean([-0.1, -0.1, -0.1]) = -0.1
            Assert.Equal(-0.1, result, 10);
        }

        [Fact]
        public void CalculateLoss_CanBePositive_Negative_OrZero()
        {
            // Arrange
            var loss = new MeanBiasErrorLoss<double>();

            // Test positive bias (under-prediction)
            var predicted1 = new Vector<double>(new double[] { 1.0, 2.0 });
            var actual1 = new Vector<double>(new double[] { 3.0, 4.0 });
            var result1 = loss.CalculateLoss(predicted1, actual1);
            Assert.True(result1 > 0.0);

            // Test negative bias (over-prediction)
            var predicted2 = new Vector<double>(new double[] { 3.0, 4.0 });
            var actual2 = new Vector<double>(new double[] { 1.0, 2.0 });
            var result2 = loss.CalculateLoss(predicted2, actual2);
            Assert.True(result2 < 0.0);

            // Test zero bias (balanced errors)
            var predicted3 = new Vector<double>(new double[] { 1.0, 3.0 });
            var actual3 = new Vector<double>(new double[] { 2.0, 2.0 });
            var result3 = loss.CalculateLoss(predicted3, actual3);
            Assert.Equal(0.0, result3, 10);
        }

        [Fact]
        public void CalculateLoss_WithLargeDataset_ComputesCorrectly()
        {
            // Arrange
            var loss = new MeanBiasErrorLoss<double>();
            var predicted = new Vector<double>(new double[]
            {
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0
            });
            var actual = new Vector<double>(new double[]
            {
                1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5
            });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // MBE = mean([0.5, 0.5, ..., 0.5]) = 0.5
            Assert.Equal(0.5, result, 10);
        }

        [Fact]
        public void CalculateDerivative_ScalesWithVectorLength()
        {
            // Arrange
            var loss = new MeanBiasErrorLoss<double>();

            // Length 2: derivative should be -0.5
            var predicted2 = new Vector<double>(new double[] { 1.0, 2.0 });
            var actual2 = new Vector<double>(new double[] { 1.0, 2.0 });
            var result2 = loss.CalculateDerivative(predicted2, actual2);
            Assert.All(result2, item => Assert.Equal(-0.5, item, 10));

            // Length 5: derivative should be -0.2
            var predicted5 = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var actual5 = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var result5 = loss.CalculateDerivative(predicted5, actual5);
            Assert.All(result5, item => Assert.Equal(-0.2, item, 10));

            // Length 10: derivative should be -0.1
            var predicted10 = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 });
            var actual10 = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 });
            var result10 = loss.CalculateDerivative(predicted10, actual10);
            Assert.All(result10, item => Assert.Equal(-0.1, item, 10));
        }
    }
}
