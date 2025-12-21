using System;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Regularization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.Regularization
{
    public class L1RegularizationTests
    {
        [Fact]
        public void Constructor_WithDefaultOptions_UsesDefaultStrength()
        {
            // Arrange & Act
            var regularization = new L1Regularization<double, Vector<double>, Vector<double>>();

            // Assert
            Assert.Equal(RegularizationType.L1, regularization.GetOptions().Type);
            Assert.Equal(0.1, regularization.GetOptions().Strength);
            Assert.Equal(1.0, regularization.GetOptions().L1Ratio);
        }

        [Fact]
        public void Constructor_WithCustomOptions_UsesCustomStrength()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.L1,
                Strength = 0.05,
                L1Ratio = 1.0
            };

            // Act
            var regularization = new L1Regularization<double, Vector<double>, Vector<double>>(options);

            // Assert
            Assert.Equal(0.05, regularization.GetOptions().Strength);
        }

        [Fact]
        public void Regularize_VectorGradient_AddsL1Penalty()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.L1,
                Strength = 0.1,
                L1Ratio = 1.0
            };
            var regularization = new L1Regularization<double, Vector<double>, Vector<double>>(options);
            var gradient = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var coefficients = new Vector<double>(new double[] { 2.0, -3.0, 4.0 });

            // Act
            var result = regularization.Regularize(gradient, coefficients);

            // Assert
            // L1 adds sign(coefficient) * strength to gradient
            // gradient[0] = 1.0 + sign(2.0) * 0.1 = 1.0 + 0.1 = 1.1
            // gradient[1] = 2.0 + sign(-3.0) * 0.1 = 2.0 - 0.1 = 1.9
            // gradient[2] = 3.0 + sign(4.0) * 0.1 = 3.0 + 0.1 = 3.1
            Assert.Equal(1.1, result[0], 10);
            Assert.Equal(1.9, result[1], 10);
            Assert.Equal(3.1, result[2], 10);
        }

        [Fact]
        public void Regularize_VectorGradient_WithZeroCoefficients_DoesNotAddPenalty()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.L1,
                Strength = 0.1,
                L1Ratio = 1.0
            };
            var regularization = new L1Regularization<double, Vector<double>, Vector<double>>(options);
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
        public void Regularize_Vector_AppliesSoftThresholding()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.L1,
                Strength = 0.5,
                L1Ratio = 1.0
            };
            var regularization = new L1Regularization<double, Vector<double>, Vector<double>>(options);
            var data = new Vector<double>(new double[] { 2.0, -3.0, 0.3, -0.2 });

            // Act
            var result = regularization.Regularize(data);

            // Assert
            // Soft thresholding: sign(x) * max(0, |x| - strength)
            // 2.0: sign(2.0) * max(0, 2.0 - 0.5) = 1 * 1.5 = 1.5
            // -3.0: sign(-3.0) * max(0, 3.0 - 0.5) = -1 * 2.5 = -2.5
            // 0.3: sign(0.3) * max(0, 0.3 - 0.5) = 1 * 0 = 0
            // -0.2: sign(-0.2) * max(0, 0.2 - 0.5) = -1 * 0 = 0
            Assert.Equal(1.5, result[0], 10);
            Assert.Equal(-2.5, result[1], 10);
            Assert.Equal(0.0, result[2], 10);
            Assert.Equal(0.0, result[3], 10);
        }

        [Fact]
        public void Regularize_Vector_ProducesSparseSolution()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.L1,
                Strength = 1.0,
                L1Ratio = 1.0
            };
            var regularization = new L1Regularization<double, Vector<double>, Vector<double>>(options);
            var data = new Vector<double>(new double[] { 0.5, 0.8, 1.5, 2.0 });

            // Act
            var result = regularization.Regularize(data);

            // Assert
            // Values below threshold (1.0) should be zero (sparse)
            Assert.Equal(0.0, result[0], 10);  // 0.5 < 1.0
            Assert.Equal(0.0, result[1], 10);  // 0.8 < 1.0
            Assert.Equal(0.5, result[2], 10);  // 1.5 - 1.0 = 0.5
            Assert.Equal(1.0, result[3], 10);  // 2.0 - 1.0 = 1.0
        }

        [Fact]
        public void Regularize_Matrix_AppliesSoftThresholding()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.L1,
                Strength = 0.5,
                L1Ratio = 1.0
            };
            var regularization = new L1Regularization<double, Matrix<double>, Matrix<double>>(options);
            var data = new Matrix<double>(2, 2);
            data[0, 0] = 2.0;
            data[0, 1] = -3.0;
            data[1, 0] = 0.3;
            data[1, 1] = -0.2;

            // Act
            var result = regularization.Regularize(data);

            // Assert
            Assert.Equal(1.5, result[0, 0], 10);
            Assert.Equal(-2.5, result[0, 1], 10);
            Assert.Equal(0.0, result[1, 0], 10);
            Assert.Equal(0.0, result[1, 1], 10);
        }

        [Fact]
        public void Regularize_TensorGradient_WorksCorrectly()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.L1,
                Strength = 0.1,
                L1Ratio = 1.0
            };
            var regularization = new L1Regularization<double, Tensor<double>, Tensor<double>>(options);
            var gradient = new Tensor<double>(new int[] { 2, 2 });
            gradient[0, 0] = 1.0;
            gradient[0, 1] = 2.0;
            gradient[1, 0] = 3.0;
            gradient[1, 1] = 4.0;

            var coefficients = new Tensor<double>(new int[] { 2, 2 });
            coefficients[0, 0] = 2.0;
            coefficients[0, 1] = -3.0;
            coefficients[1, 0] = 4.0;
            coefficients[1, 1] = -5.0;

            // Act
            var result = regularization.Regularize(gradient, coefficients);

            // Assert
            Assert.Equal(2, result.Shape[0]);
            Assert.Equal(2, result.Shape[1]);
            // Values should have L1 penalty added
            Assert.True(result[0, 0] > 1.0);
            Assert.True(result[0, 1] < 2.0);
        }

        [Fact]
        public void Regularize_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.L1,
                Strength = 0.5,
                L1Ratio = 1.0
            };
            var regularization = new L1Regularization<float, Vector<float>, Vector<float>>(options);
            var data = new Vector<float>(new float[] { 2.0f, -3.0f, 0.3f });

            // Act
            var result = regularization.Regularize(data);

            // Assert
            Assert.Equal(1.5f, result[0], 5);
            Assert.Equal(-2.5f, result[1], 5);
            Assert.Equal(0.0f, result[2], 5);
        }

        [Fact]
        public void Regularize_WithHighStrength_ProducesMoreSparsity()
        {
            // Arrange
            var lowStrength = new L1Regularization<double, Vector<double>, Vector<double>>(
                new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.1, L1Ratio = 1.0 });
            var highStrength = new L1Regularization<double, Vector<double>, Vector<double>>(
                new RegularizationOptions { Type = RegularizationType.L1, Strength = 1.0, L1Ratio = 1.0 });
            var data = new Vector<double>(new double[] { 0.5, 1.0, 1.5, 2.0 });

            // Act
            var resultLow = lowStrength.Regularize(data);
            var resultHigh = highStrength.Regularize(data);

            // Assert
            // High strength should produce more zeros
            var zerosLow = 0;
            var zerosHigh = 0;
            for (int i = 0; i < resultLow.Length; i++)
            {
                if (Math.Abs(resultLow[i]) < 0.001) zerosLow++;
                if (Math.Abs(resultHigh[i]) < 0.001) zerosHigh++;
            }
            Assert.True(zerosHigh > zerosLow);
        }

        [Fact]
        public void Regularize_Vector_PreservesSign()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.L1,
                Strength = 0.5,
                L1Ratio = 1.0
            };
            var regularization = new L1Regularization<double, Vector<double>, Vector<double>>(options);
            var data = new Vector<double>(new double[] { 2.0, -3.0, 4.0, -5.0 });

            // Act
            var result = regularization.Regularize(data);

            // Assert
            // L1 should preserve the sign of all non-zero values
            Assert.True(result[0] > 0.0);  // Positive input -> positive output
            Assert.True(result[1] < 0.0);  // Negative input -> negative output
            Assert.True(result[2] > 0.0);  // Positive input -> positive output
            Assert.True(result[3] < 0.0);  // Negative input -> negative output
        }
    }
}
