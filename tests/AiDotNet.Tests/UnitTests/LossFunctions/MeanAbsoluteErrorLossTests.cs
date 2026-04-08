using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.LossFunctions
{
    public class MeanAbsoluteErrorLossTests
    {
        [Fact]
        public void CalculateLoss_WithPerfectPredictions_ReturnsZero()
        {
            // Arrange
            var loss = new MeanAbsoluteErrorLoss<double>();
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
            var loss = new MeanAbsoluteErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 2.0, 4.0, 6.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // MAE = (|2-1| + |4-2| + |6-3|) / 3 = (1 + 2 + 3) / 3 = 2.0
            Assert.Equal(2.0, result, 10);
        }

        [Fact]
        public void CalculateLoss_WithNegativeValues_HandlesCorrectly()
        {
            // Arrange
            var loss = new MeanAbsoluteErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { -1.0, -2.0, -3.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // MAE = (|-1-1| + |-2-2| + |-3-3|) / 3 = (2 + 4 + 6) / 3 = 4.0
            Assert.Equal(4.0, result, 10);
        }

        [Fact]
        public void CalculateLoss_WithSingleValue_ReturnsCorrectValue()
        {
            // Arrange
            var loss = new MeanAbsoluteErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 5.0 });
            var actual = new Vector<double>(new double[] { 3.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // MAE = |5-3| = 2.0
            Assert.Equal(2.0, result, 10);
        }

        [Fact]
        public void CalculateLoss_WithDifferentLengths_ThrowsArgumentException()
        {
            // Arrange
            var loss = new MeanAbsoluteErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => loss.CalculateLoss(predicted, actual));
        }

        [Fact]
        public void CalculateDerivative_WithPerfectPredictions_ReturnsZero()
        {
            // Arrange
            var loss = new MeanAbsoluteErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = loss.CalculateDerivative(predicted, actual);

            // Assert
            Assert.All(result, item => Assert.Equal(0.0, item, 10));
        }

        [Fact]
        public void CalculateDerivative_WithPositiveDifferences_ReturnsPositiveValues()
        {
            // Arrange
            var loss = new MeanAbsoluteErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 2.0, 4.0, 6.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = loss.CalculateDerivative(predicted, actual);

            // Assert
            // Derivative = sign(predicted - actual) / n
            Assert.True(result[0] > 0.0);
            Assert.True(result[1] > 0.0);
            Assert.True(result[2] > 0.0);
        }

        [Fact]
        public void CalculateDerivative_WithNegativeDifferences_ReturnsNegativeValues()
        {
            // Arrange
            var loss = new MeanAbsoluteErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new double[] { 2.0, 4.0, 6.0 });

            // Act
            var result = loss.CalculateDerivative(predicted, actual);

            // Assert
            // Derivative = sign(predicted - actual) / n
            Assert.True(result[0] < 0.0);
            Assert.True(result[1] < 0.0);
            Assert.True(result[2] < 0.0);
        }

        [Fact]
        public void CalculateDerivative_WithDifferentLengths_ThrowsArgumentException()
        {
            // Arrange
            var loss = new MeanAbsoluteErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => loss.CalculateDerivative(predicted, actual));
        }

        [Fact]
        public void CalculateLoss_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var loss = new MeanAbsoluteErrorLoss<float>();
            var predicted = new Vector<float>(new float[] { 2.0f, 4.0f, 6.0f });
            var actual = new Vector<float>(new float[] { 1.0f, 2.0f, 3.0f });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            Assert.Equal(2.0f, result, 5);
        }

        [Fact]
        public void CalculateDerivative_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var loss = new MeanAbsoluteErrorLoss<float>();
            var predicted = new Vector<float>(new float[] { 2.0f, 4.0f, 6.0f });
            var actual = new Vector<float>(new float[] { 1.0f, 2.0f, 3.0f });

            // Act
            var result = loss.CalculateDerivative(predicted, actual);

            // Assert
            Assert.All(result, item => Assert.True(item > 0.0f));
        }

        [Fact]
        public void CalculateLoss_IsRobustToOutliers_ComparedToMSE()
        {
            // Arrange
            var maeLoss = new MeanAbsoluteErrorLoss<double>();
            var mseLoss = new MeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 1.0, 2.0, 100.0 });  // One outlier
            var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var maeResult = maeLoss.CalculateLoss(predicted, actual);
            var mseResult = mseLoss.CalculateLoss(predicted, actual);

            // Assert
            // MAE = (0 + 0 + 97) / 3 = 32.33
            // MSE = (0 + 0 + 9409) / 3 = 3136.33
            // MAE should be much smaller than MSE due to outlier
            Assert.True(maeResult < mseResult);
            Assert.Equal(32.333333333333333, maeResult, 10);
        }

        [Fact]
        public void CalculateLoss_WithMixedPositiveAndNegativeErrors_HandlesCorrectly()
        {
            // Arrange
            var loss = new MeanAbsoluteErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 2.0, 1.0, 4.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // MAE = (|2-1| + |1-2| + |4-3|) / 3 = (1 + 1 + 1) / 3 = 1.0
            Assert.Equal(1.0, result, 10);
        }

        [Fact]
        public void CalculateDerivative_MagnitudeIsConstant()
        {
            // Arrange
            var loss = new MeanAbsoluteErrorLoss<double>();
            var predicted = new Vector<double>(new double[] { 2.0, 5.0, 10.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = loss.CalculateDerivative(predicted, actual);

            // Assert
            // MAE derivative has constant magnitude (1/n) regardless of error size
            var n = predicted.Length;
            Assert.All(result, item => Assert.Equal(1.0 / n, Math.Abs(item), 10));
        }
    }
}
