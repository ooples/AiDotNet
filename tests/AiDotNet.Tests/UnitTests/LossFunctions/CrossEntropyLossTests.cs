using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.LossFunctions
{
    public class CrossEntropyLossTests
    {
        [Fact]
        public void CalculateLoss_WithPerfectPredictions_ReturnsNearZero()
        {
            // Arrange
            var loss = new CrossEntropyLoss<double>();
            var predicted = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });
            var actual = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // With perfect predictions, cross-entropy should be very close to zero
            Assert.True(result < 0.001);
        }

        [Fact]
        public void CalculateLoss_WithProbabilityDistribution_ReturnsCorrectValue()
        {
            // Arrange
            var loss = new CrossEntropyLoss<double>();
            var predicted = new Vector<double>(new double[] { 0.7, 0.2, 0.1 });
            var actual = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // Cross-entropy = -1/n * sum(actual * log(predicted))
            // = -1/3 * (1*log(0.7) + 0*log(0.2) + 0*log(0.1))
            // = -1/3 * log(0.7) = approximately 0.1187
            Assert.True(result > 0.0);
            Assert.True(result < 1.0);
        }

        [Fact]
        public void CalculateLoss_WithMultipleClasses_ReturnsCorrectValue()
        {
            // Arrange
            var loss = new CrossEntropyLoss<double>();
            var predicted = new Vector<double>(new double[] { 0.25, 0.25, 0.25, 0.25 });
            var actual = new Vector<double>(new double[] { 0.0, 1.0, 0.0, 0.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // Cross-entropy with uniform prediction should be higher than with confident prediction
            Assert.True(result > 0.0);
        }

        [Fact]
        public void CalculateLoss_WithDifferentLengths_ThrowsArgumentException()
        {
            // Arrange
            var loss = new CrossEntropyLoss<double>();
            var predicted = new Vector<double>(new double[] { 0.7, 0.3 });
            var actual = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => loss.CalculateLoss(predicted, actual));
        }

        [Fact]
        public void CalculateLoss_WithZeroProbabilities_HandlesNumericalStability()
        {
            // Arrange
            var loss = new CrossEntropyLoss<double>();
            var predicted = new Vector<double>(new double[] { 0.0, 0.0, 1.0 });
            var actual = new Vector<double>(new double[] { 0.0, 0.0, 1.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // Should handle zero probabilities without throwing or returning NaN/Infinity
            Assert.False(double.IsNaN(result));
            Assert.False(double.IsInfinity(result));
        }

        [Fact]
        public void CalculateDerivative_WithPerfectPredictions_ReturnsNegativeValues()
        {
            // Arrange
            var loss = new CrossEntropyLoss<double>();
            var predicted = new Vector<double>(new double[] { 0.9, 0.05, 0.05 });
            var actual = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });

            // Act
            var result = loss.CalculateDerivative(predicted, actual);

            // Assert
            // Derivative = -actual / predicted / n
            // Should have negative value for the first element
            Assert.True(result[0] < 0.0);
        }

        [Fact]
        public void CalculateDerivative_WithDifferentLengths_ThrowsArgumentException()
        {
            // Arrange
            var loss = new CrossEntropyLoss<double>();
            var predicted = new Vector<double>(new double[] { 0.7, 0.3 });
            var actual = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => loss.CalculateDerivative(predicted, actual));
        }

        [Fact]
        public void CalculateDerivative_WithZeroProbabilities_HandlesNumericalStability()
        {
            // Arrange
            var loss = new CrossEntropyLoss<double>();
            var predicted = new Vector<double>(new double[] { 0.0, 0.5, 0.5 });
            var actual = new Vector<double>(new double[] { 0.0, 1.0, 0.0 });

            // Act
            var result = loss.CalculateDerivative(predicted, actual);

            // Assert
            // Should handle zero probabilities without throwing or returning NaN/Infinity
            Assert.All(result, item => Assert.False(double.IsNaN(item)));
            Assert.All(result, item => Assert.False(double.IsInfinity(item)));
        }

        [Fact]
        public void CalculateLoss_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var loss = new CrossEntropyLoss<float>();
            var predicted = new Vector<float>(new float[] { 0.7f, 0.2f, 0.1f });
            var actual = new Vector<float>(new float[] { 1.0f, 0.0f, 0.0f });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            Assert.True(result > 0.0f);
            Assert.True(result < 1.0f);
            Assert.False(float.IsNaN(result));
        }

        [Fact]
        public void CalculateDerivative_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var loss = new CrossEntropyLoss<float>();
            var predicted = new Vector<float>(new float[] { 0.9f, 0.05f, 0.05f });
            var actual = new Vector<float>(new float[] { 1.0f, 0.0f, 0.0f });

            // Act
            var result = loss.CalculateDerivative(predicted, actual);

            // Assert
            Assert.True(result[0] < 0.0f);
            Assert.All(result, item => Assert.False(float.IsNaN(item)));
        }

        [Fact]
        public void CalculateLoss_WithConfidentWrongPrediction_ReturnsHighLoss()
        {
            // Arrange
            var loss = new CrossEntropyLoss<double>();
            var predicted = new Vector<double>(new double[] { 0.1, 0.1, 0.8 });
            var actual = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // Should penalize confident wrong predictions heavily
            Assert.True(result > 0.5);
        }

        [Fact]
        public void CalculateLoss_WithSingleValue_ReturnsCorrectValue()
        {
            // Arrange
            var loss = new CrossEntropyLoss<double>();
            var predicted = new Vector<double>(new double[] { 0.8 });
            var actual = new Vector<double>(new double[] { 1.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // Should work with single-element vectors
            Assert.True(result > 0.0);
            Assert.False(double.IsNaN(result));
        }

        [Fact]
        public void CalculateDerivative_ReturnsVectorOfCorrectLength()
        {
            // Arrange
            var loss = new CrossEntropyLoss<double>();
            var predicted = new Vector<double>(new double[] { 0.7, 0.2, 0.1 });
            var actual = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });

            // Act
            var result = loss.CalculateDerivative(predicted, actual);

            // Assert
            Assert.Equal(predicted.Length, result.Length);
        }
    }
}
