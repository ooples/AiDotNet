using System;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Regularization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.Regularization
{
    public class L2RegularizationTests
    {
        [Fact]
        public void Constructor_WithDefaultOptions_UsesDefaultStrength()
        {
            // Arrange & Act
            var regularization = new L2Regularization<double, Vector<double>, Vector<double>>();

            // Assert
            Assert.Equal(RegularizationType.L2, regularization.GetOptions().Type);
            Assert.Equal(0.01, regularization.GetOptions().Strength);
            Assert.Equal(0.0, regularization.GetOptions().L1Ratio);
        }

        [Fact]
        public void Constructor_WithCustomOptions_UsesCustomStrength()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.L2,
                Strength = 0.05,
                L1Ratio = 0.0
            };

            // Act
            var regularization = new L2Regularization<double, Vector<double>, Vector<double>>(options);

            // Assert
            Assert.Equal(0.05, regularization.GetOptions().Strength);
        }

        [Fact]
        public void Regularize_VectorGradient_AddsL2Penalty()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.L2,
                Strength = 0.1,
                L1Ratio = 0.0
            };
            var regularization = new L2Regularization<double, Vector<double>, Vector<double>>(options);
            var gradient = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var coefficients = new Vector<double>(new double[] { 2.0, 3.0, 4.0 });

            // Act
            var result = regularization.Regularize(gradient, coefficients);

            // Assert
            // L2 adds coefficient * strength to gradient
            // gradient[0] = 1.0 + 2.0 * 0.1 = 1.2
            // gradient[1] = 2.0 + 3.0 * 0.1 = 2.3
            // gradient[2] = 3.0 + 4.0 * 0.1 = 3.4
            Assert.Equal(1.2, result[0], 10);
            Assert.Equal(2.3, result[1], 10);
            Assert.Equal(3.4, result[2], 10);
        }

        [Fact]
        public void Regularize_VectorGradient_WithZeroCoefficients_DoesNotAddPenalty()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.L2,
                Strength = 0.1,
                L1Ratio = 0.0
            };
            var regularization = new L2Regularization<double, Vector<double>, Vector<double>>(options);
            var gradient = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var coefficients = new Vector<double>(new double[] { 0.0, 0.0, 0.0 });

            // Act
            var result = regularization.Regularize(gradient, coefficients);

            // Assert
            // With zero coefficients, no penalty should be added
            Assert.Equal(1.0, result[0], 10);
            Assert.Equal(2.0, result[1], 10);
            Assert.Equal(3.0, result[2], 10);
        }

        [Fact]
        public void Regularize_Vector_ShrinksAllValues()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.L2,
                Strength = 0.1,
                L1Ratio = 0.0
            };
            var regularization = new L2Regularization<double, Vector<double>, Vector<double>>(options);
            var data = new Vector<double>(new double[] { 2.0, 3.0, 4.0, 5.0 });

            // Act
            var result = regularization.Regularize(data);

            // Assert
            // L2 multiplies by (1 - strength)
            // shrinkageFactor = 1 - 0.1 = 0.9
            Assert.Equal(1.8, result[0], 10);  // 2.0 * 0.9
            Assert.Equal(2.7, result[1], 10);  // 3.0 * 0.9
            Assert.Equal(3.6, result[2], 10);  // 4.0 * 0.9
            Assert.Equal(4.5, result[3], 10);  // 5.0 * 0.9
        }

        [Fact]
        public void Regularize_Vector_DoesNotProduceSparseSolution()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.L2,
                Strength = 0.5,
                L1Ratio = 0.0
            };
            var regularization = new L2Regularization<double, Vector<double>, Vector<double>>(options);
            var data = new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4 });

            // Act
            var result = regularization.Regularize(data);

            // Assert
            // L2 does not produce sparse solutions - all values should remain non-zero
            Assert.All(result, item => Assert.NotEqual(0.0, item));
            // All values should be smaller than original
            for (int i = 0; i < data.Length; i++)
            {
                Assert.True(Math.Abs(result[i]) < Math.Abs(data[i]));
            }
        }

        [Fact]
        public void Regularize_Matrix_ShrinksAllValues()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.L2,
                Strength = 0.1,
                L1Ratio = 0.0
            };
            var regularization = new L2Regularization<double, Matrix<double>, Matrix<double>>(options);
            var data = new Matrix<double>(2, 2);
            data[0, 0] = 2.0;
            data[0, 1] = 3.0;
            data[1, 0] = 4.0;
            data[1, 1] = 5.0;

            // Act
            var result = regularization.Regularize(data);

            // Assert
            // shrinkageFactor = 1 - 0.1 = 0.9
            Assert.Equal(1.8, result[0, 0], 10);
            Assert.Equal(2.7, result[0, 1], 10);
            Assert.Equal(3.6, result[1, 0], 10);
            Assert.Equal(4.5, result[1, 1], 10);
        }

        [Fact]
        public void Regularize_TensorGradient_WorksCorrectly()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.L2,
                Strength = 0.1,
                L1Ratio = 0.0
            };
            var regularization = new L2Regularization<double, Tensor<double>, Tensor<double>>(options);
            var gradient = new Tensor<double>(new int[] { 2, 2 });
            gradient[0, 0] = 1.0;
            gradient[0, 1] = 2.0;
            gradient[1, 0] = 3.0;
            gradient[1, 1] = 4.0;

            var coefficients = new Tensor<double>(new int[] { 2, 2 });
            coefficients[0, 0] = 2.0;
            coefficients[0, 1] = 3.0;
            coefficients[1, 0] = 4.0;
            coefficients[1, 1] = 5.0;

            // Act
            var result = regularization.Regularize(gradient, coefficients);

            // Assert
            Assert.Equal(2, result.Shape[0]);
            Assert.Equal(2, result.Shape[1]);
            // Values should have L2 penalty added
            Assert.Equal(1.2, result[0, 0], 10);
            Assert.Equal(2.3, result[0, 1], 10);
            Assert.Equal(3.4, result[1, 0], 10);
            Assert.Equal(4.5, result[1, 1], 10);
        }

        [Fact]
        public void Regularize_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.L2,
                Strength = 0.1,
                L1Ratio = 0.0
            };
            var regularization = new L2Regularization<float, Vector<float>, Vector<float>>(options);
            var data = new Vector<float>(new float[] { 2.0f, 3.0f, 4.0f });

            // Act
            var result = regularization.Regularize(data);

            // Assert
            Assert.Equal(1.8f, result[0], 5);
            Assert.Equal(2.7f, result[1], 5);
            Assert.Equal(3.6f, result[2], 5);
        }

        [Fact]
        public void Regularize_WithHighStrength_ShrinksMore()
        {
            // Arrange
            var lowStrength = new L2Regularization<double, Vector<double>, Vector<double>>(
                new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.1, L1Ratio = 0.0 });
            var highStrength = new L2Regularization<double, Vector<double>, Vector<double>>(
                new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.5, L1Ratio = 0.0 });
            var data = new Vector<double>(new double[] { 2.0, 3.0, 4.0, 5.0 });

            // Act
            var resultLow = lowStrength.Regularize(data);
            var resultHigh = highStrength.Regularize(data);

            // Assert
            // High strength should shrink values more
            for (int i = 0; i < data.Length; i++)
            {
                Assert.True(Math.Abs(resultHigh[i]) < Math.Abs(resultLow[i]));
            }
        }

        [Fact]
        public void Regularize_Vector_PreservesSign()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.L2,
                Strength = 0.5,
                L1Ratio = 0.0
            };
            var regularization = new L2Regularization<double, Vector<double>, Vector<double>>(options);
            var data = new Vector<double>(new double[] { 2.0, -3.0, 4.0, -5.0 });

            // Act
            var result = regularization.Regularize(data);

            // Assert
            // L2 should preserve the sign of all values
            Assert.True(result[0] > 0.0);  // Positive input -> positive output
            Assert.True(result[1] < 0.0);  // Negative input -> negative output
            Assert.True(result[2] > 0.0);  // Positive input -> positive output
            Assert.True(result[3] < 0.0);  // Negative input -> negative output
        }

        [Fact]
        public void Regularize_Vector_ShrinksProportionally()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.L2,
                Strength = 0.5,
                L1Ratio = 0.0
            };
            var regularization = new L2Regularization<double, Vector<double>, Vector<double>>(options);
            var data = new Vector<double>(new double[] { 2.0, 4.0, 6.0 });

            // Act
            var result = regularization.Regularize(data);

            // Assert
            // All values should be shrunk by the same factor
            var shrinkageFactor = result[0] / data[0];
            Assert.Equal(shrinkageFactor, result[1] / data[1], 10);
            Assert.Equal(shrinkageFactor, result[2] / data[2], 10);
        }

        [Fact]
        public void Regularize_WithNegativeCoefficients_WorksCorrectly()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.L2,
                Strength = 0.1,
                L1Ratio = 0.0
            };
            var regularization = new L2Regularization<double, Vector<double>, Vector<double>>(options);
            var gradient = new Vector<double>(new double[] { 1.0, 2.0 });
            var coefficients = new Vector<double>(new double[] { -2.0, -3.0 });

            // Act
            var result = regularization.Regularize(gradient, coefficients);

            // Assert
            // gradient[0] = 1.0 + (-2.0) * 0.1 = 1.0 - 0.2 = 0.8
            // gradient[1] = 2.0 + (-3.0) * 0.1 = 2.0 - 0.3 = 1.7
            Assert.Equal(0.8, result[0], 10);
            Assert.Equal(1.7, result[1], 10);
        }
    }
}
