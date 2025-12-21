using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.LossFunctions
{
    public class SparseCategoricalCrossEntropyLossTests
    {
        [Fact]
        public void CalculateLoss_WithPerfectPrediction_SingleSample_ReturnsNearZero()
        {
            // Arrange
            var loss = new SparseCategoricalCrossEntropyLoss<double>();
            // 3 classes, perfect prediction for class 1
            var predicted = new Vector<double>(new double[] { 0.001, 0.998, 0.001 });
            var actual = new Vector<double>(new double[] { 1.0 }); // class index 1

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // Loss = -log(0.998) ≈ 0.002
            Assert.True(result < 0.01);
            Assert.True(result >= 0.0);
        }

        [Fact]
        public void CalculateLoss_WithSingleSample_ReturnsCorrectValue()
        {
            // Arrange
            var loss = new SparseCategoricalCrossEntropyLoss<double>();
            // 4 classes with probabilities
            var predicted = new Vector<double>(new double[] { 0.1, 0.2, 0.6, 0.1 });
            var actual = new Vector<double>(new double[] { 2.0 }); // class index 2

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // Loss = -log(0.6) = 0.5108...
            Assert.Equal(0.5108256237659907, result, 10);
        }

        [Fact]
        public void CalculateLoss_WithMultipleSamples_ReturnsAverageLoss()
        {
            // Arrange
            var loss = new SparseCategoricalCrossEntropyLoss<double>();
            // One set of predicted probabilities for 3 classes
            var predicted = new Vector<double>(new double[] { 0.7, 0.2, 0.1 });
            // Two samples with different target classes: sample 1 targets class 0, sample 2 targets class 1
            // This tests evaluating the same prediction vector against multiple class indices
            var actual = new Vector<double>(new double[] { 0.0, 1.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // Sample 1 loss: -log(predicted[0]) = -log(0.7) = 0.35667494393873245
            // Sample 2 loss: -log(predicted[1]) = -log(0.2) = 1.6094379124341003
            // Average: (0.35667494393873245 + 1.6094379124341003) / 2 = 0.9830564281864163
            Assert.Equal(0.9830564281864163, result, 10);
        }

        [Fact]
        public void CalculateLoss_WithPoorPrediction_ReturnsHighLoss()
        {
            // Arrange
            var loss = new SparseCategoricalCrossEntropyLoss<double>();
            // 3 classes, very low probability for correct class
            var predicted = new Vector<double>(new double[] { 0.8, 0.15, 0.05 });
            var actual = new Vector<double>(new double[] { 2.0 }); // class index 2 has only 0.05 prob

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // Loss = -log(0.05) = 2.9957...
            Assert.Equal(2.995732273553991, result, 10);
        }

        [Fact]
        public void CalculateLoss_WithInvalidClassIndex_ThrowsArgumentException()
        {
            // Arrange
            var loss = new SparseCategoricalCrossEntropyLoss<double>();
            var predicted = new Vector<double>(new double[] { 0.5, 0.3, 0.2 });
            var actual = new Vector<double>(new double[] { 5.0 }); // index 5 is out of bounds

            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(() => loss.CalculateLoss(predicted, actual));
            Assert.Contains("out of bounds", exception.Message);
        }

        [Fact]
        public void CalculateLoss_WithNegativeClassIndex_ThrowsArgumentException()
        {
            // Arrange
            var loss = new SparseCategoricalCrossEntropyLoss<double>();
            var predicted = new Vector<double>(new double[] { 0.5, 0.3, 0.2 });
            var actual = new Vector<double>(new double[] { -1.0 }); // negative index

            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(() => loss.CalculateLoss(predicted, actual));
            Assert.Contains("out of bounds", exception.Message);
        }

        [Fact]
        public void CalculateLoss_WithEmptyPredicted_ThrowsArgumentException()
        {
            // Arrange
            var loss = new SparseCategoricalCrossEntropyLoss<double>();
            var predicted = new Vector<double>(new double[] { });
            var actual = new Vector<double>(new double[] { 0.0 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => loss.CalculateLoss(predicted, actual));
        }

        [Fact]
        public void CalculateDerivative_WithSingleSample_ReturnsCorrectGradient()
        {
            // Arrange
            var loss = new SparseCategoricalCrossEntropyLoss<double>();
            var predicted = new Vector<double>(new double[] { 0.1, 0.2, 0.6, 0.1 });
            var actual = new Vector<double>(new double[] { 2.0 }); // class index 2

            // Act
            var result = loss.CalculateDerivative(predicted, actual);

            // Assert
            // Gradient should be 0 for classes 0, 1, 3
            // Gradient should be -1/0.6 = -1.6666... for class 2
            Assert.Equal(4, result.Length);
            Assert.Equal(0.0, result[0], 10);
            Assert.Equal(0.0, result[1], 10);
            Assert.Equal(-1.6666666666666667, result[2], 10);
            Assert.Equal(0.0, result[3], 10);
        }

        [Fact]
        public void CalculateDerivative_WithMultipleSamples_ReturnsAveragedGradient()
        {
            // Arrange
            var loss = new SparseCategoricalCrossEntropyLoss<double>();
            var predicted = new Vector<double>(new double[] { 0.7, 0.2, 0.1 });
            // Two samples, both targeting class 0 (class indices [0, 0])
            var actual = new Vector<double>(new double[] { 0.0, 0.0 });

            // Act
            var result = loss.CalculateDerivative(predicted, actual);

            // Assert
            // Gradient for class 0: -1/0.7 = -1.4285... (accumulated twice, then averaged)
            // Average: -2 * (1/0.7) / 2 = -1.4285...
            Assert.Equal(-1.4285714285714286, result[0], 10);
            Assert.Equal(0.0, result[1], 10);
            Assert.Equal(0.0, result[2], 10);
        }

        [Fact]
        public void CalculateDerivative_WithDifferentClassIndices_ReturnsCorrectGradient()
        {
            // Arrange
            var loss = new SparseCategoricalCrossEntropyLoss<double>();
            var predicted = new Vector<double>(new double[] { 0.5, 0.3, 0.2 });
            // Three samples with class indices [0, 1, 2]
            var actual = new Vector<double>(new double[] { 0.0, 1.0, 2.0 });

            // Act
            var result = loss.CalculateDerivative(predicted, actual);

            // Assert
            // Gradient for class 0: -1/0.5 / 3 = -0.6666...
            // Gradient for class 1: -1/0.3 / 3 = -1.1111...
            // Gradient for class 2: -1/0.2 / 3 = -1.6666...
            Assert.Equal(-0.6666666666666666, result[0], 10);
            Assert.Equal(-1.1111111111111112, result[1], 10);
            Assert.Equal(-1.6666666666666667, result[2], 10);
        }

        [Fact]
        public void CalculateDerivative_WithInvalidClassIndex_ThrowsArgumentException()
        {
            // Arrange
            var loss = new SparseCategoricalCrossEntropyLoss<double>();
            var predicted = new Vector<double>(new double[] { 0.5, 0.3, 0.2 });
            var actual = new Vector<double>(new double[] { 3.0 }); // out of bounds

            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(() => loss.CalculateDerivative(predicted, actual));
            Assert.Contains("out of bounds", exception.Message);
        }

        [Fact]
        public void CalculateLoss_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var loss = new SparseCategoricalCrossEntropyLoss<float>();
            var predicted = new Vector<float>(new float[] { 0.1f, 0.2f, 0.6f, 0.1f });
            var actual = new Vector<float>(new float[] { 2.0f });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // Loss = -log(0.6) = 0.5108...
            Assert.Equal(0.5108256237659907f, result, 5);
        }

        [Fact]
        public void CalculateDerivative_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var loss = new SparseCategoricalCrossEntropyLoss<float>();
            var predicted = new Vector<float>(new float[] { 0.1f, 0.2f, 0.6f, 0.1f });
            var actual = new Vector<float>(new float[] { 2.0f });

            // Act
            var result = loss.CalculateDerivative(predicted, actual);

            // Assert
            Assert.Equal(0.0f, result[0], 5);
            Assert.Equal(0.0f, result[1], 5);
            Assert.Equal(-1.6666666666666667f, result[2], 5);
            Assert.Equal(0.0f, result[3], 5);
        }

        [Fact]
        public void CalculateLoss_WithManyClasses_HandlesCorrectly()
        {
            // Arrange
            var loss = new SparseCategoricalCrossEntropyLoss<double>();
            // 10 classes
            var predicted = new Vector<double>(new double[]
            {
                0.05, 0.05, 0.1, 0.2, 0.3, 0.15, 0.05, 0.05, 0.03, 0.02
            });
            var actual = new Vector<double>(new double[] { 4.0 }); // class 4 has prob 0.3

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // Loss = -log(0.3) = 1.2039...
            Assert.Equal(1.2039728043259361, result, 10);
        }

        [Fact]
        public void CalculateLoss_WithVeryLowProbability_HandlesClamping()
        {
            // Arrange
            var loss = new SparseCategoricalCrossEntropyLoss<double>();
            // Predicted has essentially 0 probability for class 2
            var predicted = new Vector<double>(new double[] { 0.49999, 0.49999, 0.00001 });
            var actual = new Vector<double>(new double[] { 2.0 });

            // Act
            var result = loss.CalculateLoss(predicted, actual);

            // Assert
            // Should handle near-zero probability without overflow
            // Loss = -log(0.00001) ≈ 11.51
            Assert.True(result > 10.0);
            Assert.True(result < 15.0);
        }
    }
}
