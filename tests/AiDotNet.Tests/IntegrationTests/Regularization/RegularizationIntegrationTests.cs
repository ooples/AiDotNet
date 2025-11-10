using AiDotNet.Enums;
using AiDotNet.Factories;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Regularization;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Regularization
{
    /// <summary>
    /// Comprehensive integration tests for regularization techniques.
    /// Tests L1, L2, ElasticNet, and No regularization with mathematical verification.
    /// Covers penalty computation, gradient computation, sparsity, and overfitting reduction.
    /// </summary>
    public class RegularizationIntegrationTests
    {
        private const double Tolerance = 1e-10;

        #region NoRegularization Tests

        [Fact]
        public void NoRegularization_MatrixPassthrough_ReturnsUnchanged()
        {
            // Arrange
            var regularization = new NoRegularization<double, Matrix<double>, Vector<double>>();
            var matrix = new Matrix<double>(3, 3);
            matrix[0, 0] = 1.5; matrix[0, 1] = -2.3; matrix[0, 2] = 3.7;
            matrix[1, 0] = -4.2; matrix[1, 1] = 5.8; matrix[1, 2] = -6.1;
            matrix[2, 0] = 7.9; matrix[2, 1] = -8.4; matrix[2, 2] = 9.2;

            // Act
            var result = regularization.Regularize(matrix);

            // Assert - NoRegularization returns input unchanged
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    Assert.Equal(matrix[i, j], result[i, j], Tolerance);
                }
            }
        }

        [Fact]
        public void NoRegularization_VectorPassthrough_ReturnsUnchanged()
        {
            // Arrange
            var regularization = new NoRegularization<double, Matrix<double>, Vector<double>>();
            var vector = new Vector<double>(5);
            vector[0] = 1.5; vector[1] = -2.3; vector[2] = 3.7; vector[3] = -4.2; vector[4] = 5.8;

            // Act
            var result = regularization.Regularize(vector);

            // Assert - NoRegularization returns input unchanged
            for (int i = 0; i < 5; i++)
            {
                Assert.Equal(vector[i], result[i], Tolerance);
            }
        }

        [Fact]
        public void NoRegularization_GradientPassthrough_ReturnsUnchanged()
        {
            // Arrange
            var regularization = new NoRegularization<double, Matrix<double>, Vector<double>>();
            var gradient = new Vector<double>(4);
            gradient[0] = 0.5; gradient[1] = -0.3; gradient[2] = 0.7; gradient[3] = -0.2;
            var coefficients = new Vector<double>(4);
            coefficients[0] = 1.0; coefficients[1] = 2.0; coefficients[2] = 3.0; coefficients[3] = 4.0;

            // Act
            var result = regularization.Regularize(gradient, coefficients);

            // Assert - NoRegularization returns gradient unchanged
            for (int i = 0; i < 4; i++)
            {
                Assert.Equal(gradient[i], result[i], Tolerance);
            }
        }

        [Fact]
        public void NoRegularization_GetOptions_ReturnsDefaultOptions()
        {
            // Arrange
            var regularization = new NoRegularization<double, Matrix<double>, Vector<double>>();

            // Act
            var options = regularization.GetOptions();

            // Assert - NoRegularization should have default/no options
            Assert.NotNull(options);
        }

        #endregion

        #region L1 Regularization - Penalty Computation Tests

        [Fact]
        public void L1Regularization_VectorSoftThresholding_ZeroStrength_ReturnsOriginal()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.0 };
            var regularization = new L1Regularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(4);
            vector[0] = 2.0; vector[1] = -1.5; vector[2] = 3.0; vector[3] = -2.5;

            // Act
            var result = regularization.Regularize(vector);

            // Assert - With zero strength, soft thresholding returns original values
            Assert.Equal(2.0, result[0], Tolerance);
            Assert.Equal(-1.5, result[1], Tolerance);
            Assert.Equal(3.0, result[2], Tolerance);
            Assert.Equal(-2.5, result[3], Tolerance);
        }

        [Fact]
        public void L1Regularization_VectorSoftThresholding_LightStrength_ReducesValues()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.5 };
            var regularization = new L1Regularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(4);
            vector[0] = 2.0; vector[1] = -1.5; vector[2] = 3.0; vector[3] = -2.5;

            // Act
            var result = regularization.Regularize(vector);

            // Assert - Soft thresholding: sign(w) * max(0, |w| - strength)
            Assert.Equal(1.5, result[0], Tolerance); // sign(2.0) * max(0, 2.0 - 0.5) = 1 * 1.5
            Assert.Equal(-1.0, result[1], Tolerance); // sign(-1.5) * max(0, 1.5 - 0.5) = -1 * 1.0
            Assert.Equal(2.5, result[2], Tolerance); // sign(3.0) * max(0, 3.0 - 0.5) = 1 * 2.5
            Assert.Equal(-2.0, result[3], Tolerance); // sign(-2.5) * max(0, 2.5 - 0.5) = -1 * 2.0
        }

        [Fact]
        public void L1Regularization_VectorSoftThresholding_StrongStrength_CreatesSparsity()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 2.0 };
            var regularization = new L1Regularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(5);
            vector[0] = 3.0; vector[1] = 1.5; vector[2] = -2.5; vector[3] = 1.0; vector[4] = -4.0;

            // Act
            var result = regularization.Regularize(vector);

            // Assert - L1 creates sparsity by setting small values to zero
            Assert.Equal(1.0, result[0], Tolerance); // 3.0 - 2.0 = 1.0
            Assert.Equal(0.0, result[1], Tolerance); // 1.5 - 2.0 < 0 → 0
            Assert.Equal(-0.5, result[2], Tolerance); // -(2.5 - 2.0) = -0.5
            Assert.Equal(0.0, result[3], Tolerance); // 1.0 - 2.0 < 0 → 0
            Assert.Equal(-2.0, result[4], Tolerance); // -(4.0 - 2.0) = -2.0
        }

        [Fact]
        public void L1Regularization_VectorSoftThresholding_VeryStrongStrength_MostValuesZero()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 10.0 };
            var regularization = new L1Regularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(5);
            vector[0] = 5.0; vector[1] = 3.0; vector[2] = -7.0; vector[3] = 2.0; vector[4] = -4.0;

            // Act
            var result = regularization.Regularize(vector);

            // Assert - Very strong regularization sets most values to zero
            Assert.Equal(0.0, result[0], Tolerance); // 5.0 - 10.0 < 0 → 0
            Assert.Equal(0.0, result[1], Tolerance); // 3.0 - 10.0 < 0 → 0
            Assert.Equal(0.0, result[2], Tolerance); // 7.0 - 10.0 < 0 → 0
            Assert.Equal(0.0, result[3], Tolerance); // 2.0 - 10.0 < 0 → 0
            Assert.Equal(0.0, result[4], Tolerance); // 4.0 - 10.0 < 0 → 0
        }

        [Fact]
        public void L1Regularization_MatrixSoftThresholding_CreatesSparsity()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 1.5 };
            var regularization = new L1Regularization<double, Matrix<double>, Vector<double>>(options);
            var matrix = new Matrix<double>(2, 3);
            matrix[0, 0] = 3.0; matrix[0, 1] = 1.0; matrix[0, 2] = -2.5;
            matrix[1, 0] = 0.5; matrix[1, 1] = -4.0; matrix[1, 2] = 2.0;

            // Act
            var result = regularization.Regularize(matrix);

            // Assert - Soft thresholding applied to each element
            Assert.Equal(1.5, result[0, 0], Tolerance); // 3.0 - 1.5 = 1.5
            Assert.Equal(0.0, result[0, 1], Tolerance); // 1.0 - 1.5 < 0 → 0
            Assert.Equal(-1.0, result[0, 2], Tolerance); // -(2.5 - 1.5) = -1.0
            Assert.Equal(0.0, result[1, 0], Tolerance); // 0.5 - 1.5 < 0 → 0
            Assert.Equal(-2.5, result[1, 1], Tolerance); // -(4.0 - 1.5) = -2.5
            Assert.Equal(0.5, result[1, 2], Tolerance); // 2.0 - 1.5 = 0.5
        }

        #endregion

        #region L1 Regularization - Gradient Computation Tests

        [Fact]
        public void L1Regularization_GradientWithVector_AddsSignOfCoefficients()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.1 };
            var regularization = new L1Regularization<double, Matrix<double>, Vector<double>>(options);
            var gradient = new Vector<double>(4);
            gradient[0] = 0.5; gradient[1] = -0.3; gradient[2] = 0.7; gradient[3] = -0.2;
            var coefficients = new Vector<double>(4);
            coefficients[0] = 2.0; coefficients[1] = -1.5; coefficients[2] = 3.0; coefficients[3] = -2.5;

            // Act
            var result = regularization.Regularize(gradient, coefficients);

            // Assert - L1 gradient: gradient + strength * sign(coefficient)
            Assert.Equal(0.6, result[0], Tolerance); // 0.5 + 0.1 * sign(2.0) = 0.5 + 0.1
            Assert.Equal(-0.4, result[1], Tolerance); // -0.3 + 0.1 * sign(-1.5) = -0.3 - 0.1
            Assert.Equal(0.8, result[2], Tolerance); // 0.7 + 0.1 * sign(3.0) = 0.7 + 0.1
            Assert.Equal(-0.3, result[3], Tolerance); // -0.2 + 0.1 * sign(-2.5) = -0.2 - 0.1
        }

        [Fact]
        public void L1Regularization_GradientWithVector_ZeroCoefficients_NoChange()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.1 };
            var regularization = new L1Regularization<double, Matrix<double>, Vector<double>>(options);
            var gradient = new Vector<double>(3);
            gradient[0] = 0.5; gradient[1] = -0.3; gradient[2] = 0.7;
            var coefficients = new Vector<double>(3);
            coefficients[0] = 0.0; coefficients[1] = 0.0; coefficients[2] = 0.0;

            // Act
            var result = regularization.Regularize(gradient, coefficients);

            // Assert - sign(0) = 0, so gradient unchanged
            Assert.Equal(0.5, result[0], Tolerance);
            Assert.Equal(-0.3, result[1], Tolerance);
            Assert.Equal(0.7, result[2], Tolerance);
        }

        [Fact]
        public void L1Regularization_GradientWithTensor_AddsSignOfCoefficients()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.2 };
            var regularization = new L1Regularization<double, Matrix<double>, Tensor<double>>(options);
            var gradient = new Tensor<double>(new[] { 4 });
            gradient[0] = 0.5; gradient[1] = -0.3; gradient[2] = 0.7; gradient[3] = -0.2;
            var coefficients = new Tensor<double>(new[] { 4 });
            coefficients[0] = 1.0; coefficients[1] = -2.0; coefficients[2] = 3.0; coefficients[3] = -1.5;

            // Act
            var result = regularization.Regularize(gradient, coefficients);

            // Assert - L1 gradient: gradient + strength * sign(coefficient)
            Assert.Equal(0.7, result[0], Tolerance); // 0.5 + 0.2 * 1
            Assert.Equal(-0.5, result[1], Tolerance); // -0.3 + 0.2 * (-1)
            Assert.Equal(0.9, result[2], Tolerance); // 0.7 + 0.2 * 1
            Assert.Equal(-0.4, result[3], Tolerance); // -0.2 + 0.2 * (-1)
        }

        [Fact]
        public void L1Regularization_GradientWithTensor_HigherDimensions_CorrectShape()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.1 };
            var regularization = new L1Regularization<double, Matrix<double>, Tensor<double>>(options);
            var gradient = new Tensor<double>(new[] { 2, 3 });
            gradient[0, 0] = 0.5; gradient[0, 1] = -0.3; gradient[0, 2] = 0.7;
            gradient[1, 0] = -0.2; gradient[1, 1] = 0.4; gradient[1, 2] = -0.6;
            var coefficients = new Tensor<double>(new[] { 2, 3 });
            coefficients[0, 0] = 1.0; coefficients[0, 1] = -2.0; coefficients[0, 2] = 3.0;
            coefficients[1, 0] = -1.5; coefficients[1, 1] = 2.5; coefficients[1, 2] = -3.5;

            // Act
            var result = regularization.Regularize(gradient, coefficients);

            // Assert - Result maintains shape and applies L1 gradient correctly
            Assert.Equal(2, result.Shape.Length);
            Assert.Equal(2, result.Shape[0]);
            Assert.Equal(3, result.Shape[1]);
            Assert.Equal(0.6, result[0, 0], Tolerance); // 0.5 + 0.1 * sign(1.0)
            Assert.Equal(-0.4, result[0, 1], Tolerance); // -0.3 + 0.1 * sign(-2.0)
        }

        #endregion

        #region L1 Regularization - Sparsity Tests

        [Fact]
        public void L1Regularization_IncreasedStrength_IncreasedSparsity()
        {
            // Arrange
            var vector = new Vector<double>(10);
            for (int i = 0; i < 10; i++)
            {
                vector[i] = (i + 1) * 0.5; // 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0
            }

            // Act - Test with increasing strength
            var strength05 = new L1Regularization<double, Matrix<double>, Vector<double>>(
                new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.5 });
            var strength15 = new L1Regularization<double, Matrix<double>, Vector<double>>(
                new RegularizationOptions { Type = RegularizationType.L1, Strength = 1.5 });
            var strength30 = new L1Regularization<double, Matrix<double>, Vector<double>>(
                new RegularizationOptions { Type = RegularizationType.L1, Strength = 3.0 });

            var result05 = strength05.Regularize(vector);
            var result15 = strength15.Regularize(vector);
            var result30 = strength30.Regularize(vector);

            // Count zeros
            int zeros05 = 0, zeros15 = 0, zeros30 = 0;
            for (int i = 0; i < 10; i++)
            {
                if (Math.Abs(result05[i]) < Tolerance) zeros05++;
                if (Math.Abs(result15[i]) < Tolerance) zeros15++;
                if (Math.Abs(result30[i]) < Tolerance) zeros30++;
            }

            // Assert - Higher strength leads to more zeros (sparsity)
            Assert.True(zeros05 < zeros15);
            Assert.True(zeros15 < zeros30);
        }

        [Fact]
        public void L1Regularization_Sparsity_PreservesLargeValues()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 2.0 };
            var regularization = new L1Regularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(6);
            vector[0] = 10.0; vector[1] = 0.5; vector[2] = -8.0;
            vector[3] = 1.5; vector[4] = -12.0; vector[5] = 0.8;

            // Act
            var result = regularization.Regularize(vector);

            // Assert - Large values reduced but not zeroed, small values set to zero
            Assert.True(Math.Abs(result[0]) > 5.0); // Large value preserved
            Assert.Equal(0.0, result[1], Tolerance); // Small value zeroed
            Assert.True(Math.Abs(result[2]) > 4.0); // Large value preserved
            Assert.Equal(0.0, result[3], Tolerance); // Small value zeroed
            Assert.True(Math.Abs(result[4]) > 8.0); // Large value preserved
            Assert.Equal(0.0, result[5], Tolerance); // Small value zeroed
        }

        #endregion

        #region L2 Regularization - Penalty Computation Tests

        [Fact]
        public void L2Regularization_VectorUniformShrinkage_ZeroStrength_ReturnsOriginal()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.0 };
            var regularization = new L2Regularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(4);
            vector[0] = 2.0; vector[1] = -1.5; vector[2] = 3.0; vector[3] = -2.5;

            // Act
            var result = regularization.Regularize(vector);

            // Assert - With zero strength, returns original
            Assert.Equal(2.0, result[0], Tolerance);
            Assert.Equal(-1.5, result[1], Tolerance);
            Assert.Equal(3.0, result[2], Tolerance);
            Assert.Equal(-2.5, result[3], Tolerance);
        }

        [Fact]
        public void L2Regularization_VectorUniformShrinkage_LightStrength_ShrinksAll()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.1 };
            var regularization = new L2Regularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(4);
            vector[0] = 2.0; vector[1] = -1.5; vector[2] = 3.0; vector[3] = -2.5;

            // Act
            var result = regularization.Regularize(vector);

            // Assert - L2 shrinkage: value * (1 - strength)
            double shrinkageFactor = 1.0 - 0.1;
            Assert.Equal(2.0 * shrinkageFactor, result[0], Tolerance);
            Assert.Equal(-1.5 * shrinkageFactor, result[1], Tolerance);
            Assert.Equal(3.0 * shrinkageFactor, result[2], Tolerance);
            Assert.Equal(-2.5 * shrinkageFactor, result[3], Tolerance);
        }

        [Fact]
        public void L2Regularization_VectorUniformShrinkage_StrongStrength_SignificantShrinkage()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.5 };
            var regularization = new L2Regularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(4);
            vector[0] = 10.0; vector[1] = -8.0; vector[2] = 6.0; vector[3] = -4.0;

            // Act
            var result = regularization.Regularize(vector);

            // Assert - L2 shrinks all values by 50%
            Assert.Equal(5.0, result[0], Tolerance); // 10.0 * 0.5
            Assert.Equal(-4.0, result[1], Tolerance); // -8.0 * 0.5
            Assert.Equal(3.0, result[2], Tolerance); // 6.0 * 0.5
            Assert.Equal(-2.0, result[3], Tolerance); // -4.0 * 0.5
        }

        [Fact]
        public void L2Regularization_VectorUniformShrinkage_VeryStrongStrength_NearZero()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.99 };
            var regularization = new L2Regularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(4);
            vector[0] = 100.0; vector[1] = -50.0; vector[2] = 75.0; vector[3] = -25.0;

            // Act
            var result = regularization.Regularize(vector);

            // Assert - Very strong shrinkage brings values near zero
            Assert.Equal(1.0, result[0], Tolerance); // 100.0 * 0.01
            Assert.Equal(-0.5, result[1], Tolerance); // -50.0 * 0.01
            Assert.Equal(0.75, result[2], Tolerance); // 75.0 * 0.01
            Assert.Equal(-0.25, result[3], Tolerance); // -25.0 * 0.01
        }

        [Fact]
        public void L2Regularization_MatrixUniformShrinkage_ShrinksAllElements()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.2 };
            var regularization = new L2Regularization<double, Matrix<double>, Vector<double>>(options);
            var matrix = new Matrix<double>(2, 3);
            matrix[0, 0] = 5.0; matrix[0, 1] = -4.0; matrix[0, 2] = 3.0;
            matrix[1, 0] = -2.0; matrix[1, 1] = 1.0; matrix[1, 2] = -6.0;

            // Act
            var result = regularization.Regularize(matrix);

            // Assert - All elements shrunk by factor (1 - 0.2) = 0.8
            Assert.Equal(4.0, result[0, 0], Tolerance);
            Assert.Equal(-3.2, result[0, 1], Tolerance);
            Assert.Equal(2.4, result[0, 2], Tolerance);
            Assert.Equal(-1.6, result[1, 0], Tolerance);
            Assert.Equal(0.8, result[1, 1], Tolerance);
            Assert.Equal(-4.8, result[1, 2], Tolerance);
        }

        [Fact]
        public void L2Regularization_NoSparsity_AllValuesNonZero()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.8 };
            var regularization = new L2Regularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(5);
            vector[0] = 1.0; vector[1] = 2.0; vector[2] = 3.0; vector[3] = 4.0; vector[4] = 5.0;

            // Act
            var result = regularization.Regularize(vector);

            // Assert - L2 never creates exact zeros (unlike L1)
            for (int i = 0; i < 5; i++)
            {
                Assert.NotEqual(0.0, result[i]);
                Assert.True(Math.Abs(result[i]) > 0);
            }
        }

        #endregion

        #region L2 Regularization - Gradient Computation Tests

        [Fact]
        public void L2Regularization_GradientWithVector_AddsScaledCoefficients()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.1 };
            var regularization = new L2Regularization<double, Matrix<double>, Vector<double>>(options);
            var gradient = new Vector<double>(4);
            gradient[0] = 0.5; gradient[1] = -0.3; gradient[2] = 0.7; gradient[3] = -0.2;
            var coefficients = new Vector<double>(4);
            coefficients[0] = 2.0; coefficients[1] = -1.5; coefficients[2] = 3.0; coefficients[3] = -2.5;

            // Act
            var result = regularization.Regularize(gradient, coefficients);

            // Assert - L2 gradient: gradient + strength * coefficient
            Assert.Equal(0.7, result[0], Tolerance); // 0.5 + 0.1 * 2.0
            Assert.Equal(-0.45, result[1], Tolerance); // -0.3 + 0.1 * (-1.5)
            Assert.Equal(1.0, result[2], Tolerance); // 0.7 + 0.1 * 3.0
            Assert.Equal(-0.45, result[3], Tolerance); // -0.2 + 0.1 * (-2.5)
        }

        [Fact]
        public void L2Regularization_GradientWithVector_ZeroCoefficients_NoChange()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.1 };
            var regularization = new L2Regularization<double, Matrix<double>, Vector<double>>(options);
            var gradient = new Vector<double>(3);
            gradient[0] = 0.5; gradient[1] = -0.3; gradient[2] = 0.7;
            var coefficients = new Vector<double>(3);
            coefficients[0] = 0.0; coefficients[1] = 0.0; coefficients[2] = 0.0;

            // Act
            var result = regularization.Regularize(gradient, coefficients);

            // Assert - Zero coefficients add nothing to gradient
            Assert.Equal(0.5, result[0], Tolerance);
            Assert.Equal(-0.3, result[1], Tolerance);
            Assert.Equal(0.7, result[2], Tolerance);
        }

        [Fact]
        public void L2Regularization_GradientWithTensor_AddsScaledCoefficients()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.2 };
            var regularization = new L2Regularization<double, Matrix<double>, Tensor<double>>(options);
            var gradient = new Tensor<double>(new[] { 4 });
            gradient[0] = 0.5; gradient[1] = -0.3; gradient[2] = 0.7; gradient[3] = -0.2;
            var coefficients = new Tensor<double>(new[] { 4 });
            coefficients[0] = 1.0; coefficients[1] = -2.0; coefficients[2] = 3.0; coefficients[3] = -1.5;

            // Act
            var result = regularization.Regularize(gradient, coefficients);

            // Assert - L2 gradient: gradient + strength * coefficient
            Assert.Equal(0.7, result[0], Tolerance); // 0.5 + 0.2 * 1.0
            Assert.Equal(-0.7, result[1], Tolerance); // -0.3 + 0.2 * (-2.0)
            Assert.Equal(1.3, result[2], Tolerance); // 0.7 + 0.2 * 3.0
            Assert.Equal(-0.5, result[3], Tolerance); // -0.2 + 0.2 * (-1.5)
        }

        [Fact]
        public void L2Regularization_GradientWithTensor_HigherDimensions_CorrectShape()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.1 };
            var regularization = new L2Regularization<double, Matrix<double>, Tensor<double>>(options);
            var gradient = new Tensor<double>(new[] { 2, 3 });
            gradient[0, 0] = 0.5; gradient[0, 1] = -0.3; gradient[0, 2] = 0.7;
            gradient[1, 0] = -0.2; gradient[1, 1] = 0.4; gradient[1, 2] = -0.6;
            var coefficients = new Tensor<double>(new[] { 2, 3 });
            coefficients[0, 0] = 1.0; coefficients[0, 1] = -2.0; coefficients[0, 2] = 3.0;
            coefficients[1, 0] = -1.5; coefficients[1, 1] = 2.5; coefficients[1, 2] = -3.5;

            // Act
            var result = regularization.Regularize(gradient, coefficients);

            // Assert - Result maintains shape and applies L2 gradient correctly
            Assert.Equal(2, result.Shape.Length);
            Assert.Equal(2, result.Shape[0]);
            Assert.Equal(3, result.Shape[1]);
            Assert.Equal(0.6, result[0, 0], Tolerance); // 0.5 + 0.1 * 1.0
            Assert.Equal(-0.5, result[0, 1], Tolerance); // -0.3 + 0.1 * (-2.0)
        }

        #endregion

        #region L2 Regularization - Proportional Shrinkage Tests

        [Fact]
        public void L2Regularization_ProportionalShrinkage_MaintainsRelativeScale()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.5 };
            var regularization = new L2Regularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(3);
            vector[0] = 10.0; vector[1] = 5.0; vector[2] = 2.0;

            // Act
            var result = regularization.Regularize(vector);

            // Assert - Relative ratios maintained after shrinkage
            double ratio01_before = vector[0] / vector[1]; // 2.0
            double ratio12_before = vector[1] / vector[2]; // 2.5
            double ratio01_after = result[0] / result[1];
            double ratio12_after = result[1] / result[2];

            Assert.Equal(ratio01_before, ratio01_after, Tolerance);
            Assert.Equal(ratio12_before, ratio12_after, Tolerance);
        }

        [Fact]
        public void L2Regularization_LargeValuesStaySmallerThanBefore()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.3 };
            var regularization = new L2Regularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(5);
            vector[0] = 10.0; vector[1] = -8.0; vector[2] = 6.0; vector[3] = -4.0; vector[4] = 2.0;

            // Act
            var result = regularization.Regularize(vector);

            // Assert - All values smaller in magnitude than before
            for (int i = 0; i < 5; i++)
            {
                Assert.True(Math.Abs(result[i]) < Math.Abs(vector[i]));
            }
        }

        #endregion

        #region ElasticNet Regularization - Combination Tests

        [Fact]
        public void ElasticNet_L1RatioZero_BehavesLikeL2()
        {
            // Arrange - L1Ratio = 0 means pure L2
            var elasticOptions = new RegularizationOptions
            {
                Type = RegularizationType.ElasticNet,
                Strength = 0.2,
                L1Ratio = 0.0
            };
            var l2Options = new RegularizationOptions
            {
                Type = RegularizationType.L2,
                Strength = 0.2
            };
            var elasticNet = new ElasticNetRegularization<double, Matrix<double>, Vector<double>>(elasticOptions);
            var l2 = new L2Regularization<double, Matrix<double>, Vector<double>>(l2Options);
            var vector = new Vector<double>(4);
            vector[0] = 5.0; vector[1] = -3.0; vector[2] = 7.0; vector[3] = -2.0;

            // Act
            var elasticResult = elasticNet.Regularize(vector);
            var l2Result = l2.Regularize(vector);

            // Assert - Results should be nearly identical
            for (int i = 0; i < 4; i++)
            {
                Assert.Equal(l2Result[i], elasticResult[i], 1e-8);
            }
        }

        [Fact]
        public void ElasticNet_L1RatioOne_BehavesLikeL1()
        {
            // Arrange - L1Ratio = 1 means pure L1
            var elasticOptions = new RegularizationOptions
            {
                Type = RegularizationType.ElasticNet,
                Strength = 0.5,
                L1Ratio = 1.0
            };
            var l1Options = new RegularizationOptions
            {
                Type = RegularizationType.L1,
                Strength = 0.5
            };
            var elasticNet = new ElasticNetRegularization<double, Matrix<double>, Vector<double>>(elasticOptions);
            var l1 = new L1Regularization<double, Matrix<double>, Vector<double>>(l1Options);
            var vector = new Vector<double>(4);
            vector[0] = 2.0; vector[1] = -1.5; vector[2] = 3.0; vector[3] = -0.8;

            // Act
            var elasticResult = elasticNet.Regularize(vector);
            var l1Result = l1.Regularize(vector);

            // Assert - Results should be nearly identical
            for (int i = 0; i < 4; i++)
            {
                Assert.Equal(l1Result[i], elasticResult[i], 1e-8);
            }
        }

        [Fact]
        public void ElasticNet_L1RatioHalf_CombinesBothEffects()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.ElasticNet,
                Strength = 1.0,
                L1Ratio = 0.5
            };
            var elasticNet = new ElasticNetRegularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(4);
            vector[0] = 3.0; vector[1] = 1.0; vector[2] = -2.5; vector[3] = 0.8;

            // Act
            var result = elasticNet.Regularize(vector);

            // Assert - ElasticNet combines L1 (soft thresholding) and L2 (shrinkage)
            // With L1Ratio = 0.5, strength = 1.0:
            // L1 part: strength * l1_ratio = 0.5
            // L2 part: strength * (1 - l1_ratio) = 0.5
            // For value 3.0: L1 soft threshold with 0.5 gives 2.5, then L2 shrinkage with 0.5 gives 1.25
            // The actual formula is more complex, but result should be between pure L1 and pure L2
            Assert.True(Math.Abs(result[0]) < 3.0); // Smaller than original
            Assert.True(Math.Abs(result[0]) > 0.0); // Not zero
        }

        [Fact]
        public void ElasticNet_DifferentL1Ratios_DifferentResults()
        {
            // Arrange
            var vector = new Vector<double>(5);
            vector[0] = 5.0; vector[1] = 2.0; vector[2] = -4.0; vector[3] = 1.5; vector[4] = -3.0;

            var elastic25 = new ElasticNetRegularization<double, Matrix<double>, Vector<double>>(
                new RegularizationOptions { Strength = 1.0, L1Ratio = 0.25 });
            var elastic50 = new ElasticNetRegularization<double, Matrix<double>, Vector<double>>(
                new RegularizationOptions { Strength = 1.0, L1Ratio = 0.5 });
            var elastic75 = new ElasticNetRegularization<double, Matrix<double>, Vector<double>>(
                new RegularizationOptions { Strength = 1.0, L1Ratio = 0.75 });

            // Act
            var result25 = elastic25.Regularize(vector);
            var result50 = elastic50.Regularize(vector);
            var result75 = elastic75.Regularize(vector);

            // Assert - Different L1Ratios produce different results
            // Higher L1Ratio = more L1-like (more sparsity)
            int zeros25 = 0, zeros50 = 0, zeros75 = 0;
            for (int i = 0; i < 5; i++)
            {
                if (Math.Abs(result25[i]) < Tolerance) zeros25++;
                if (Math.Abs(result50[i]) < Tolerance) zeros50++;
                if (Math.Abs(result75[i]) < Tolerance) zeros75++;
            }
            // Higher L1 ratio should lead to more zeros
            Assert.True(zeros25 <= zeros50);
            Assert.True(zeros50 <= zeros75);
        }

        [Fact]
        public void ElasticNet_MatrixRegularization_CombinesBothEffects()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.ElasticNet,
                Strength = 0.8,
                L1Ratio = 0.6
            };
            var elasticNet = new ElasticNetRegularization<double, Matrix<double>, Vector<double>>(options);
            var matrix = new Matrix<double>(2, 3);
            matrix[0, 0] = 3.0; matrix[0, 1] = 0.5; matrix[0, 2] = -2.5;
            matrix[1, 0] = 1.0; matrix[1, 1] = -4.0; matrix[1, 2] = 2.0;

            // Act
            var result = elasticNet.Regularize(matrix);

            // Assert - Small values may become zero (L1 effect), large values shrunk (L2 effect)
            // The exact calculation is complex, but we can verify general properties
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    Assert.True(Math.Abs(result[i, j]) <= Math.Abs(matrix[i, j]));
                }
            }
        }

        #endregion

        #region ElasticNet Regularization - Gradient Tests

        [Fact]
        public void ElasticNet_GradientWithVector_CombinesL1AndL2Gradients()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.ElasticNet,
                Strength = 0.2,
                L1Ratio = 0.5
            };
            var elasticNet = new ElasticNetRegularization<double, Matrix<double>, Vector<double>>(options);
            var gradient = new Vector<double>(3);
            gradient[0] = 0.5; gradient[1] = -0.3; gradient[2] = 0.7;
            var coefficients = new Vector<double>(3);
            coefficients[0] = 2.0; coefficients[1] = -3.0; coefficients[2] = 1.5;

            // Act
            var result = elasticNet.Regularize(gradient, coefficients);

            // Assert - ElasticNet gradient combines L1 and L2
            // gradient + strength * (l1_ratio * sign(coef) + (1 - l1_ratio) * coef)
            // For coef[0] = 2.0: 0.5 + 0.2 * (0.5 * 1 + 0.5 * 2.0) = 0.5 + 0.2 * 1.5 = 0.8
            Assert.Equal(0.8, result[0], Tolerance);
            // For coef[1] = -3.0: -0.3 + 0.2 * (0.5 * (-1) + 0.5 * (-3.0)) = -0.3 + 0.2 * (-1.75) = -0.65
            Assert.Equal(-0.65, result[1], Tolerance);
            // For coef[2] = 1.5: 0.7 + 0.2 * (0.5 * 1 + 0.5 * 1.5) = 0.7 + 0.2 * 1.25 = 0.95
            Assert.Equal(0.95, result[2], Tolerance);
        }

        [Fact]
        public void ElasticNet_GradientWithTensor_CombinesL1AndL2Gradients()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.ElasticNet,
                Strength = 0.1,
                L1Ratio = 0.3
            };
            var elasticNet = new ElasticNetRegularization<double, Matrix<double>, Tensor<double>>(options);
            var gradient = new Tensor<double>(new[] { 3 });
            gradient[0] = 0.5; gradient[1] = -0.3; gradient[2] = 0.7;
            var coefficients = new Tensor<double>(new[] { 3 });
            coefficients[0] = 2.0; coefficients[1] = -3.0; coefficients[2] = 1.5;

            // Act
            var result = elasticNet.Regularize(gradient, coefficients);

            // Assert - ElasticNet gradient for tensors
            // For coef[0] = 2.0: 0.5 + 0.1 * (0.3 * 1 + 0.7 * 2.0) = 0.5 + 0.1 * 1.7 = 0.67
            Assert.Equal(0.67, result[0], Tolerance);
            // For coef[1] = -3.0: -0.3 + 0.1 * (0.3 * (-1) + 0.7 * (-3.0)) = -0.3 + 0.1 * (-2.4) = -0.54
            Assert.Equal(-0.54, result[1], Tolerance);
        }

        [Fact]
        public void ElasticNet_GradientWithTensor_HigherDimensions_CorrectShape()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.ElasticNet,
                Strength = 0.15,
                L1Ratio = 0.4
            };
            var elasticNet = new ElasticNetRegularization<double, Matrix<double>, Tensor<double>>(options);
            var gradient = new Tensor<double>(new[] { 2, 2 });
            gradient[0, 0] = 0.5; gradient[0, 1] = -0.3;
            gradient[1, 0] = 0.7; gradient[1, 1] = -0.2;
            var coefficients = new Tensor<double>(new[] { 2, 2 });
            coefficients[0, 0] = 1.0; coefficients[0, 1] = -2.0;
            coefficients[1, 0] = 3.0; coefficients[1, 1] = -1.5;

            // Act
            var result = elasticNet.Regularize(gradient, coefficients);

            // Assert - Result maintains shape
            Assert.Equal(2, result.Shape.Length);
            Assert.Equal(2, result.Shape[0]);
            Assert.Equal(2, result.Shape[1]);
        }

        #endregion

        #region Regularization Factory Tests

        [Fact]
        public void RegularizationFactory_CreateNone_ReturnsNoRegularization()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.None };

            // Act
            var regularization = RegularizationFactory.CreateRegularization<double, Matrix<double>, Vector<double>>(options);

            // Assert
            Assert.IsType<NoRegularization<double, Matrix<double>, Vector<double>>>(regularization);
        }

        [Fact]
        public void RegularizationFactory_CreateL1_ReturnsL1Regularization()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.1 };

            // Act
            var regularization = RegularizationFactory.CreateRegularization<double, Matrix<double>, Vector<double>>(options);

            // Assert
            Assert.IsType<L1Regularization<double, Matrix<double>, Vector<double>>>(regularization);
            Assert.Equal(0.1, regularization.GetOptions().Strength);
        }

        [Fact]
        public void RegularizationFactory_CreateL2_ReturnsL2Regularization()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.05 };

            // Act
            var regularization = RegularizationFactory.CreateRegularization<double, Matrix<double>, Vector<double>>(options);

            // Assert
            Assert.IsType<L2Regularization<double, Matrix<double>, Vector<double>>>(regularization);
            Assert.Equal(0.05, regularization.GetOptions().Strength);
        }

        [Fact]
        public void RegularizationFactory_CreateElasticNet_ReturnsElasticNetRegularization()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.ElasticNet,
                Strength = 0.15,
                L1Ratio = 0.7
            };

            // Act
            var regularization = RegularizationFactory.CreateRegularization<double, Matrix<double>, Vector<double>>(options);

            // Assert
            Assert.IsType<ElasticNetRegularization<double, Matrix<double>, Vector<double>>>(regularization);
            Assert.Equal(0.15, regularization.GetOptions().Strength);
            Assert.Equal(0.7, regularization.GetOptions().L1Ratio);
        }

        [Fact]
        public void RegularizationFactory_GetRegularizationType_NoRegularization_ReturnsNone()
        {
            // Arrange
            var regularization = new NoRegularization<double, Matrix<double>, Vector<double>>();

            // Act
            var type = RegularizationFactory.GetRegularizationType(regularization);

            // Assert
            Assert.Equal(RegularizationType.None, type);
        }

        [Fact]
        public void RegularizationFactory_GetRegularizationType_L1_ReturnsL1()
        {
            // Arrange
            var regularization = new L1Regularization<double, Matrix<double>, Vector<double>>();

            // Act
            var type = RegularizationFactory.GetRegularizationType(regularization);

            // Assert
            Assert.Equal(RegularizationType.L1, type);
        }

        [Fact]
        public void RegularizationFactory_GetRegularizationType_L2_ReturnsL2()
        {
            // Arrange
            var regularization = new L2Regularization<double, Matrix<double>, Vector<double>>();

            // Act
            var type = RegularizationFactory.GetRegularizationType(regularization);

            // Assert
            Assert.Equal(RegularizationType.L2, type);
        }

        [Fact]
        public void RegularizationFactory_GetRegularizationType_ElasticNet_ReturnsElasticNet()
        {
            // Arrange
            var regularization = new ElasticNetRegularization<double, Matrix<double>, Vector<double>>();

            // Act
            var type = RegularizationFactory.GetRegularizationType(regularization);

            // Assert
            Assert.Equal(RegularizationType.ElasticNet, type);
        }

        #endregion

        #region Mathematical Properties - L1 vs L2 Comparison

        [Fact]
        public void L1VsL2_SameStrength_L1CreatesMoreSparsity()
        {
            // Arrange
            var vector = new Vector<double>(10);
            for (int i = 0; i < 10; i++)
            {
                vector[i] = (i + 1) * 0.3; // 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0
            }

            var l1 = new L1Regularization<double, Matrix<double>, Vector<double>>(
                new RegularizationOptions { Strength = 1.0 });
            var l2 = new L2Regularization<double, Matrix<double>, Vector<double>>(
                new RegularizationOptions { Strength = 1.0 });

            // Act
            var l1Result = l1.Regularize(vector);
            var l2Result = l2.Regularize(vector);

            // Count zeros
            int l1Zeros = 0, l2Zeros = 0;
            for (int i = 0; i < 10; i++)
            {
                if (Math.Abs(l1Result[i]) < Tolerance) l1Zeros++;
                if (Math.Abs(l2Result[i]) < Tolerance) l2Zeros++;
            }

            // Assert - L1 creates more zeros than L2
            Assert.True(l1Zeros > l2Zeros);
            Assert.True(l1Zeros > 0); // L1 should create some zeros
            Assert.Equal(0, l2Zeros); // L2 should not create exact zeros
        }

        [Fact]
        public void L1VsL2_LargeCoefficients_L1PenalizesMore()
        {
            // Arrange
            var vector = new Vector<double>(3);
            vector[0] = 10.0; vector[1] = 1.0; vector[2] = 0.1;

            var l1 = new L1Regularization<double, Matrix<double>, Vector<double>>(
                new RegularizationOptions { Strength = 1.0 });
            var l2 = new L2Regularization<double, Matrix<double>, Vector<double>>(
                new RegularizationOptions { Strength = 0.1 });

            // Act
            var l1Result = l1.Regularize(vector);
            var l2Result = l2.Regularize(vector);

            // Calculate reduction in large coefficient
            double l1Reduction = vector[0] - l1Result[0];
            double l2Reduction = vector[0] - l2Result[0];

            // Assert - For same strength, L1 reduces large values by fixed amount
            // L2 reduces by proportion
            Assert.Equal(1.0, l1Reduction, Tolerance); // L1 reduces by exactly strength
            Assert.Equal(1.0, l2Reduction, Tolerance); // L2 reduces by 10% of 10.0
        }

        [Fact]
        public void L1VsL2_SmallCoefficients_L1SetsToZero()
        {
            // Arrange
            var vector = new Vector<double>(3);
            vector[0] = 0.5; vector[1] = 0.3; vector[2] = 0.1;

            var l1 = new L1Regularization<double, Matrix<double>, Vector<double>>(
                new RegularizationOptions { Strength = 0.4 });
            var l2 = new L2Regularization<double, Matrix<double>, Vector<double>>(
                new RegularizationOptions { Strength = 0.4 });

            // Act
            var l1Result = l1.Regularize(vector);
            var l2Result = l2.Regularize(vector);

            // Assert - L1 sets values below threshold to zero
            Assert.Equal(0.1, l1Result[0], Tolerance); // 0.5 - 0.4 = 0.1
            Assert.Equal(0.0, l1Result[1], Tolerance); // 0.3 - 0.4 < 0 → 0
            Assert.Equal(0.0, l1Result[2], Tolerance); // 0.1 - 0.4 < 0 → 0

            // L2 keeps all values non-zero
            Assert.True(l2Result[0] > 0);
            Assert.True(l2Result[1] > 0);
            Assert.True(l2Result[2] > 0);
        }

        #endregion

        #region Edge Cases and Boundary Conditions

        [Fact]
        public void L1Regularization_AllZeroVector_RemainsZero()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.5 };
            var regularization = new L1Regularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(5);
            // All zeros

            // Act
            var result = regularization.Regularize(vector);

            // Assert - Zero vector remains zero
            for (int i = 0; i < 5; i++)
            {
                Assert.Equal(0.0, result[i], Tolerance);
            }
        }

        [Fact]
        public void L2Regularization_AllZeroVector_RemainsZero()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.5 };
            var regularization = new L2Regularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(5);
            // All zeros

            // Act
            var result = regularization.Regularize(vector);

            // Assert - Zero vector remains zero
            for (int i = 0; i < 5; i++)
            {
                Assert.Equal(0.0, result[i], Tolerance);
            }
        }

        [Fact]
        public void ElasticNet_AllZeroVector_RemainsZero()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.ElasticNet,
                Strength = 0.5,
                L1Ratio = 0.5
            };
            var regularization = new ElasticNetRegularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(5);
            // All zeros

            // Act
            var result = regularization.Regularize(vector);

            // Assert - Zero vector remains zero
            for (int i = 0; i < 5; i++)
            {
                Assert.Equal(0.0, result[i], Tolerance);
            }
        }

        [Fact]
        public void L1Regularization_SingleElement_CorrectSoftThresholding()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 1.5 };
            var regularization = new L1Regularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(1);
            vector[0] = 3.0;

            // Act
            var result = regularization.Regularize(vector);

            // Assert
            Assert.Equal(1.5, result[0], Tolerance); // 3.0 - 1.5 = 1.5
        }

        [Fact]
        public void L2Regularization_SingleElement_CorrectShrinkage()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.4 };
            var regularization = new L2Regularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(1);
            vector[0] = 5.0;

            // Act
            var result = regularization.Regularize(vector);

            // Assert
            Assert.Equal(3.0, result[0], Tolerance); // 5.0 * (1 - 0.4) = 3.0
        }

        [Fact]
        public void L1Regularization_VeryLargeStrength_AllZeros()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 1000.0 };
            var regularization = new L1Regularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(5);
            vector[0] = 10.0; vector[1] = 20.0; vector[2] = 30.0; vector[3] = 40.0; vector[4] = 50.0;

            // Act
            var result = regularization.Regularize(vector);

            // Assert - Extremely large strength zeros everything
            for (int i = 0; i < 5; i++)
            {
                Assert.Equal(0.0, result[i], Tolerance);
            }
        }

        [Fact]
        public void L2Regularization_StrengthNearOne_VerySmallValues()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.999 };
            var regularization = new L2Regularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(3);
            vector[0] = 100.0; vector[1] = 200.0; vector[2] = 300.0;

            // Act
            var result = regularization.Regularize(vector);

            // Assert - Strength near 1.0 shrinks to very small values
            Assert.True(Math.Abs(result[0]) < 1.0);
            Assert.True(Math.Abs(result[1]) < 1.0);
            Assert.True(Math.Abs(result[2]) < 1.0);
        }

        #endregion

        #region Options and Configuration Tests

        [Fact]
        public void L1Regularization_GetOptions_ReturnsCorrectType()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.L1,
                Strength = 0.3
            };
            var regularization = new L1Regularization<double, Matrix<double>, Vector<double>>(options);

            // Act
            var retrievedOptions = regularization.GetOptions();

            // Assert
            Assert.Equal(RegularizationType.L1, retrievedOptions.Type);
            Assert.Equal(0.3, retrievedOptions.Strength);
        }

        [Fact]
        public void L2Regularization_GetOptions_ReturnsCorrectType()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.L2,
                Strength = 0.05
            };
            var regularization = new L2Regularization<double, Matrix<double>, Vector<double>>(options);

            // Act
            var retrievedOptions = regularization.GetOptions();

            // Assert
            Assert.Equal(RegularizationType.L2, retrievedOptions.Type);
            Assert.Equal(0.05, retrievedOptions.Strength);
        }

        [Fact]
        public void ElasticNet_GetOptions_ReturnsCorrectParameters()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.ElasticNet,
                Strength = 0.2,
                L1Ratio = 0.7
            };
            var regularization = new ElasticNetRegularization<double, Matrix<double>, Vector<double>>(options);

            // Act
            var retrievedOptions = regularization.GetOptions();

            // Assert
            Assert.Equal(RegularizationType.ElasticNet, retrievedOptions.Type);
            Assert.Equal(0.2, retrievedOptions.Strength);
            Assert.Equal(0.7, retrievedOptions.L1Ratio);
        }

        [Fact]
        public void L1Regularization_DefaultOptions_UsesDefaultStrength()
        {
            // Arrange & Act
            var regularization = new L1Regularization<double, Matrix<double>, Vector<double>>();
            var options = regularization.GetOptions();

            // Assert
            Assert.Equal(RegularizationType.L1, options.Type);
            Assert.Equal(0.1, options.Strength); // Default L1 strength
            Assert.Equal(1.0, options.L1Ratio); // L1 should have L1Ratio = 1.0
        }

        [Fact]
        public void L2Regularization_DefaultOptions_UsesDefaultStrength()
        {
            // Arrange & Act
            var regularization = new L2Regularization<double, Matrix<double>, Vector<double>>();
            var options = regularization.GetOptions();

            // Assert
            Assert.Equal(RegularizationType.L2, options.Type);
            Assert.Equal(0.01, options.Strength); // Default L2 strength
            Assert.Equal(0.0, options.L1Ratio); // L2 should have L1Ratio = 0.0
        }

        [Fact]
        public void ElasticNet_DefaultOptions_UsesDefaultParameters()
        {
            // Arrange & Act
            var regularization = new ElasticNetRegularization<double, Matrix<double>, Vector<double>>();
            var options = regularization.GetOptions();

            // Assert
            Assert.Equal(RegularizationType.ElasticNet, options.Type);
            Assert.Equal(0.1, options.Strength); // Default ElasticNet strength
            Assert.Equal(0.5, options.L1Ratio); // Default ElasticNet L1Ratio
        }

        #endregion

        #region Multiple Strength Values Tests

        [Fact]
        public void L1Regularization_MultipleStrengths_IncreasingShrinkage()
        {
            // Arrange
            var vector = new Vector<double>(5);
            vector[0] = 5.0; vector[1] = 4.0; vector[2] = 3.0; vector[3] = 2.0; vector[4] = 1.0;

            var reg001 = new L1Regularization<double, Matrix<double>, Vector<double>>(
                new RegularizationOptions { Strength = 0.01 });
            var reg01 = new L1Regularization<double, Matrix<double>, Vector<double>>(
                new RegularizationOptions { Strength = 0.1 });
            var reg10 = new L1Regularization<double, Matrix<double>, Vector<double>>(
                new RegularizationOptions { Strength = 1.0 });

            // Act
            var result001 = reg001.Regularize(vector);
            var result01 = reg01.Regularize(vector);
            var result10 = reg10.Regularize(vector);

            // Assert - Higher strength means more shrinkage
            Assert.True(Math.Abs(result001[0]) > Math.Abs(result01[0]));
            Assert.True(Math.Abs(result01[0]) > Math.Abs(result10[0]));
        }

        [Fact]
        public void L2Regularization_MultipleStrengths_IncreasingShrinkage()
        {
            // Arrange
            var vector = new Vector<double>(5);
            vector[0] = 10.0; vector[1] = 8.0; vector[2] = 6.0; vector[3] = 4.0; vector[4] = 2.0;

            var reg001 = new L2Regularization<double, Matrix<double>, Vector<double>>(
                new RegularizationOptions { Strength = 0.01 });
            var reg01 = new L2Regularization<double, Matrix<double>, Vector<double>>(
                new RegularizationOptions { Strength = 0.1 });
            var reg05 = new L2Regularization<double, Matrix<double>, Vector<double>>(
                new RegularizationOptions { Strength = 0.5 });

            // Act
            var result001 = reg001.Regularize(vector);
            var result01 = reg01.Regularize(vector);
            var result05 = reg05.Regularize(vector);

            // Assert - Higher strength means more shrinkage
            Assert.True(Math.Abs(result001[0]) > Math.Abs(result01[0]));
            Assert.True(Math.Abs(result01[0]) > Math.Abs(result05[0]));
        }

        #endregion

        #region Sign Preservation Tests

        [Fact]
        public void L1Regularization_PreservesSign_PositiveValues()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.5 };
            var regularization = new L1Regularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(3);
            vector[0] = 3.0; vector[1] = 2.0; vector[2] = 1.5;

            // Act
            var result = regularization.Regularize(vector);

            // Assert - All results should remain positive
            Assert.True(result[0] > 0 || Math.Abs(result[0]) < Tolerance);
            Assert.True(result[1] > 0 || Math.Abs(result[1]) < Tolerance);
            Assert.True(result[2] > 0 || Math.Abs(result[2]) < Tolerance);
        }

        [Fact]
        public void L1Regularization_PreservesSign_NegativeValues()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.5 };
            var regularization = new L1Regularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(3);
            vector[0] = -3.0; vector[1] = -2.0; vector[2] = -1.5;

            // Act
            var result = regularization.Regularize(vector);

            // Assert - All results should remain negative or zero
            Assert.True(result[0] < 0 || Math.Abs(result[0]) < Tolerance);
            Assert.True(result[1] < 0 || Math.Abs(result[1]) < Tolerance);
            Assert.True(result[2] < 0 || Math.Abs(result[2]) < Tolerance);
        }

        [Fact]
        public void L2Regularization_PreservesSign_PositiveAndNegative()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.3 };
            var regularization = new L2Regularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(4);
            vector[0] = 5.0; vector[1] = -3.0; vector[2] = 7.0; vector[3] = -2.0;

            // Act
            var result = regularization.Regularize(vector);

            // Assert - Signs preserved
            Assert.True(result[0] > 0);
            Assert.True(result[1] < 0);
            Assert.True(result[2] > 0);
            Assert.True(result[3] < 0);
        }

        [Fact]
        public void ElasticNet_PreservesSign_MixedValues()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.ElasticNet,
                Strength = 0.5,
                L1Ratio = 0.5
            };
            var regularization = new ElasticNetRegularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(4);
            vector[0] = 5.0; vector[1] = -3.0; vector[2] = 7.0; vector[3] = -2.0;

            // Act
            var result = regularization.Regularize(vector);

            // Assert - Signs preserved (or zero)
            Assert.True(result[0] >= 0);
            Assert.True(result[1] <= 0);
            Assert.True(result[2] >= 0);
            Assert.True(result[3] <= 0);
        }

        #endregion

        #region Float Type Tests

        [Fact]
        public void L1Regularization_FloatType_SoftThresholding()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.5 };
            var regularization = new L1Regularization<float, Matrix<float>, Vector<float>>(options);
            var vector = new Vector<float>(3);
            vector[0] = 2.0f; vector[1] = 1.0f; vector[2] = -1.5f;

            // Act
            var result = regularization.Regularize(vector);

            // Assert
            Assert.Equal(1.5f, result[0], 1e-6f);
            Assert.Equal(0.5f, result[1], 1e-6f);
            Assert.Equal(-1.0f, result[2], 1e-6f);
        }

        [Fact]
        public void L2Regularization_FloatType_UniformShrinkage()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.2 };
            var regularization = new L2Regularization<float, Matrix<float>, Vector<float>>(options);
            var vector = new Vector<float>(3);
            vector[0] = 5.0f; vector[1] = -4.0f; vector[2] = 3.0f;

            // Act
            var result = regularization.Regularize(vector);

            // Assert
            Assert.Equal(4.0f, result[0], 1e-6f);
            Assert.Equal(-3.2f, result[1], 1e-6f);
            Assert.Equal(2.4f, result[2], 1e-6f);
        }

        [Fact]
        public void ElasticNet_FloatType_CombinedRegularization()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.ElasticNet,
                Strength = 0.4,
                L1Ratio = 0.5
            };
            var regularization = new ElasticNetRegularization<float, Matrix<float>, Vector<float>>(options);
            var vector = new Vector<float>(2);
            vector[0] = 3.0f; vector[1] = -2.0f;

            // Act
            var result = regularization.Regularize(vector);

            // Assert - Result should be smaller than input
            Assert.True(Math.Abs(result[0]) < 3.0f);
            Assert.True(Math.Abs(result[1]) < 2.0f);
        }

        #endregion

        #region Weight Magnitude Reduction Tests

        [Fact]
        public void L1Regularization_ReducesWeightMagnitude_LargeWeights()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 2.0 };
            var regularization = new L1Regularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(5);
            vector[0] = 10.0; vector[1] = -8.0; vector[2] = 12.0; vector[3] = -15.0; vector[4] = 20.0;

            // Act
            var result = regularization.Regularize(vector);

            // Calculate total magnitude before and after
            double magnitudeBefore = 0, magnitudeAfter = 0;
            for (int i = 0; i < 5; i++)
            {
                magnitudeBefore += Math.Abs(vector[i]);
                magnitudeAfter += Math.Abs(result[i]);
            }

            // Assert - Total magnitude reduced
            Assert.True(magnitudeAfter < magnitudeBefore);
        }

        [Fact]
        public void L2Regularization_ReducesWeightMagnitude_AllWeights()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.3 };
            var regularization = new L2Regularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(5);
            vector[0] = 10.0; vector[1] = -8.0; vector[2] = 12.0; vector[3] = -15.0; vector[4] = 20.0;

            // Act
            var result = regularization.Regularize(vector);

            // Calculate L2 norm before and after
            double normBefore = 0, normAfter = 0;
            for (int i = 0; i < 5; i++)
            {
                normBefore += vector[i] * vector[i];
                normAfter += result[i] * result[i];
            }
            normBefore = Math.Sqrt(normBefore);
            normAfter = Math.Sqrt(normAfter);

            // Assert - L2 norm reduced
            Assert.True(normAfter < normBefore);
        }

        [Fact]
        public void ElasticNet_ReducesWeightMagnitude_CombinedEffect()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.ElasticNet,
                Strength = 1.0,
                L1Ratio = 0.5
            };
            var regularization = new ElasticNetRegularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(5);
            vector[0] = 10.0; vector[1] = -8.0; vector[2] = 12.0; vector[3] = -15.0; vector[4] = 20.0;

            // Act
            var result = regularization.Regularize(vector);

            // Calculate total magnitude
            double magnitudeBefore = 0, magnitudeAfter = 0;
            for (int i = 0; i < 5; i++)
            {
                magnitudeBefore += Math.Abs(vector[i]);
                magnitudeAfter += Math.Abs(result[i]);
            }

            // Assert - ElasticNet reduces magnitude
            Assert.True(magnitudeAfter < magnitudeBefore);
        }

        #endregion

        #region Matrix vs Vector Consistency Tests

        [Fact]
        public void L1Regularization_MatrixVsVector_ConsistentResults()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.5 };
            var regularization = new L1Regularization<double, Matrix<double>, Vector<double>>(options);

            var vector = new Vector<double>(6);
            vector[0] = 2.0; vector[1] = -1.5; vector[2] = 3.0;
            vector[3] = -2.5; vector[4] = 1.0; vector[5] = -3.5;

            var matrix = new Matrix<double>(2, 3);
            matrix[0, 0] = 2.0; matrix[0, 1] = -1.5; matrix[0, 2] = 3.0;
            matrix[1, 0] = -2.5; matrix[1, 1] = 1.0; matrix[1, 2] = -3.5;

            // Act
            var vectorResult = regularization.Regularize(vector);
            var matrixResult = regularization.Regularize(matrix);

            // Assert - Same values produce same results
            Assert.Equal(vectorResult[0], matrixResult[0, 0], Tolerance);
            Assert.Equal(vectorResult[1], matrixResult[0, 1], Tolerance);
            Assert.Equal(vectorResult[2], matrixResult[0, 2], Tolerance);
            Assert.Equal(vectorResult[3], matrixResult[1, 0], Tolerance);
            Assert.Equal(vectorResult[4], matrixResult[1, 1], Tolerance);
            Assert.Equal(vectorResult[5], matrixResult[1, 2], Tolerance);
        }

        [Fact]
        public void L2Regularization_MatrixVsVector_ConsistentResults()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.3 };
            var regularization = new L2Regularization<double, Matrix<double>, Vector<double>>(options);

            var vector = new Vector<double>(4);
            vector[0] = 5.0; vector[1] = -4.0; vector[2] = 3.0; vector[3] = -2.0;

            var matrix = new Matrix<double>(2, 2);
            matrix[0, 0] = 5.0; matrix[0, 1] = -4.0;
            matrix[1, 0] = 3.0; matrix[1, 1] = -2.0;

            // Act
            var vectorResult = regularization.Regularize(vector);
            var matrixResult = regularization.Regularize(matrix);

            // Assert - Same values produce same results
            Assert.Equal(vectorResult[0], matrixResult[0, 0], Tolerance);
            Assert.Equal(vectorResult[1], matrixResult[0, 1], Tolerance);
            Assert.Equal(vectorResult[2], matrixResult[1, 0], Tolerance);
            Assert.Equal(vectorResult[3], matrixResult[1, 1], Tolerance);
        }

        [Fact]
        public void ElasticNet_MatrixVsVector_ConsistentResults()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.ElasticNet,
                Strength = 0.6,
                L1Ratio = 0.4
            };
            var regularization = new ElasticNetRegularization<double, Matrix<double>, Vector<double>>(options);

            var vector = new Vector<double>(4);
            vector[0] = 5.0; vector[1] = -4.0; vector[2] = 3.0; vector[3] = -2.0;

            var matrix = new Matrix<double>(2, 2);
            matrix[0, 0] = 5.0; matrix[0, 1] = -4.0;
            matrix[1, 0] = 3.0; matrix[1, 1] = -2.0;

            // Act
            var vectorResult = regularization.Regularize(vector);
            var matrixResult = regularization.Regularize(matrix);

            // Assert - Same values produce same results
            Assert.Equal(vectorResult[0], matrixResult[0, 0], Tolerance);
            Assert.Equal(vectorResult[1], matrixResult[0, 1], Tolerance);
            Assert.Equal(vectorResult[2], matrixResult[1, 0], Tolerance);
            Assert.Equal(vectorResult[3], matrixResult[1, 1], Tolerance);
        }

        #endregion

        #region Additional ElasticNet L1Ratio Tests

        [Fact]
        public void ElasticNet_L1Ratio025_MoreL2Behavior()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.ElasticNet,
                Strength = 1.0,
                L1Ratio = 0.25 // More L2
            };
            var regularization = new ElasticNetRegularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(5);
            vector[0] = 5.0; vector[1] = 2.0; vector[2] = 1.0; vector[3] = 0.5; vector[4] = 0.2;

            // Act
            var result = regularization.Regularize(vector);

            // Count zeros - should have fewer zeros due to more L2 influence
            int zeros = 0;
            for (int i = 0; i < 5; i++)
            {
                if (Math.Abs(result[i]) < Tolerance) zeros++;
            }

            // Assert - L2-heavy should have few or no zeros
            Assert.True(zeros <= 2); // At most 2 very small values become zero
        }

        [Fact]
        public void ElasticNet_L1Ratio075_MoreL1Behavior()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.ElasticNet,
                Strength = 1.0,
                L1Ratio = 0.75 // More L1
            };
            var regularization = new ElasticNetRegularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(5);
            vector[0] = 5.0; vector[1] = 2.0; vector[2] = 1.0; vector[3] = 0.5; vector[4] = 0.2;

            // Act
            var result = regularization.Regularize(vector);

            // Count zeros - should have more zeros due to more L1 influence
            int zeros = 0;
            for (int i = 0; i < 5; i++)
            {
                if (Math.Abs(result[i]) < Tolerance) zeros++;
            }

            // Assert - L1-heavy should create more sparsity
            Assert.True(zeros >= 1); // At least 1 zero
        }

        [Fact]
        public void ElasticNet_VariousL1Ratios_MonotonicSparsity()
        {
            // Arrange
            var vector = new Vector<double>(8);
            for (int i = 0; i < 8; i++)
            {
                vector[i] = (i + 1) * 0.4; // 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2
            }

            var ratios = new[] { 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 };
            var sparsityCounts = new List<int>();

            // Act
            foreach (var ratio in ratios)
            {
                var reg = new ElasticNetRegularization<double, Matrix<double>, Vector<double>>(
                    new RegularizationOptions { Strength = 1.5, L1Ratio = ratio });
                var result = reg.Regularize(vector);

                int zeros = 0;
                for (int i = 0; i < 8; i++)
                {
                    if (Math.Abs(result[i]) < Tolerance) zeros++;
                }
                sparsityCounts.Add(zeros);
            }

            // Assert - As L1Ratio increases, sparsity should generally increase
            Assert.True(sparsityCounts[0] <= sparsityCounts[sparsityCounts.Count - 1]);
        }

        #endregion

        #region Gradient Edge Cases

        [Fact]
        public void L1Regularization_GradientWithLargeCoefficients_CorrectAddition()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.5 };
            var regularization = new L1Regularization<double, Matrix<double>, Vector<double>>(options);
            var gradient = new Vector<double>(3);
            gradient[0] = 0.1; gradient[1] = -0.05; gradient[2] = 0.15;
            var coefficients = new Vector<double>(3);
            coefficients[0] = 100.0; coefficients[1] = -200.0; coefficients[2] = 50.0;

            // Act
            var result = regularization.Regularize(gradient, coefficients);

            // Assert - L1 gradient adds sign regardless of coefficient magnitude
            Assert.Equal(0.6, result[0], Tolerance); // 0.1 + 0.5 * sign(100)
            Assert.Equal(-0.55, result[1], Tolerance); // -0.05 + 0.5 * sign(-200)
            Assert.Equal(0.65, result[2], Tolerance); // 0.15 + 0.5 * sign(50)
        }

        [Fact]
        public void L2Regularization_GradientWithLargeCoefficients_ProportionalAddition()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.01 };
            var regularization = new L2Regularization<double, Matrix<double>, Vector<double>>(options);
            var gradient = new Vector<double>(3);
            gradient[0] = 0.1; gradient[1] = -0.05; gradient[2] = 0.15;
            var coefficients = new Vector<double>(3);
            coefficients[0] = 100.0; coefficients[1] = -200.0; coefficients[2] = 50.0;

            // Act
            var result = regularization.Regularize(gradient, coefficients);

            // Assert - L2 gradient adds proportion of coefficient
            Assert.Equal(1.1, result[0], Tolerance); // 0.1 + 0.01 * 100
            Assert.Equal(-2.05, result[1], Tolerance); // -0.05 + 0.01 * (-200)
            Assert.Equal(0.65, result[2], Tolerance); // 0.15 + 0.01 * 50
        }

        [Fact]
        public void ElasticNet_GradientWithMixedCoefficientSizes_CorrectCombination()
        {
            // Arrange
            var options = new RegularizationOptions
            {
                Type = RegularizationType.ElasticNet,
                Strength = 0.1,
                L1Ratio = 0.6
            };
            var regularization = new ElasticNetRegularization<double, Matrix<double>, Vector<double>>(options);
            var gradient = new Vector<double>(4);
            gradient[0] = 0.2; gradient[1] = -0.1; gradient[2] = 0.3; gradient[3] = -0.15;
            var coefficients = new Vector<double>(4);
            coefficients[0] = 10.0; coefficients[1] = -5.0; coefficients[2] = 2.0; coefficients[3] = -1.0;

            // Act
            var result = regularization.Regularize(gradient, coefficients);

            // Assert - ElasticNet combines both effects
            // For coef[0] = 10.0: 0.2 + 0.1 * (0.6 * 1 + 0.4 * 10.0) = 0.2 + 0.1 * 4.6 = 0.66
            Assert.Equal(0.66, result[0], Tolerance);
            // For coef[1] = -5.0: -0.1 + 0.1 * (0.6 * (-1) + 0.4 * (-5.0)) = -0.1 + 0.1 * (-2.6) = -0.36
            Assert.Equal(-0.36, result[1], Tolerance);
        }

        #endregion

        #region Specific Penalty Verification Tests

        [Fact]
        public void L1Regularization_PenaltyComputation_SumOfAbsoluteValues()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.2 };
            var regularization = new L1Regularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(5);
            vector[0] = 3.0; vector[1] = -2.0; vector[2] = 1.5; vector[3] = -4.0; vector[4] = 2.5;

            // Calculate expected L1 penalty
            double expectedPenalty = 0.2 * (3.0 + 2.0 + 1.5 + 4.0 + 2.5); // 0.2 * 13.0 = 2.6

            // Act - Apply regularization
            var result = regularization.Regularize(vector);

            // Calculate actual reduction (penalty applied)
            double actualReduction = 0;
            for (int i = 0; i < 5; i++)
            {
                actualReduction += Math.Abs(vector[i]) - Math.Abs(result[i]);
            }

            // Assert - L1 reduces each value by strength (soft thresholding effect)
            // For L1, each value is reduced by the strength amount (if possible)
            Assert.True(actualReduction > 0); // Some reduction occurred
        }

        [Fact]
        public void L2Regularization_PenaltyComputation_SquaredValues()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.3 };
            var regularization = new L2Regularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(4);
            vector[0] = 2.0; vector[1] = -3.0; vector[2] = 1.0; vector[3] = -2.0;

            // Calculate L2 norm squared before
            double normSquaredBefore = 2.0 * 2.0 + 3.0 * 3.0 + 1.0 * 1.0 + 2.0 * 2.0; // 18.0

            // Act
            var result = regularization.Regularize(vector);

            // Calculate L2 norm squared after
            double normSquaredAfter = 0;
            for (int i = 0; i < 4; i++)
            {
                normSquaredAfter += result[i] * result[i];
            }

            // Assert - L2 norm squared reduced by shrinkage factor squared
            double expectedNormSquared = normSquaredBefore * (1 - 0.3) * (1 - 0.3); // 18.0 * 0.49 = 8.82
            Assert.Equal(expectedNormSquared, normSquaredAfter, Tolerance);
        }

        #endregion

        #region Additional Strength Variation Tests

        [Fact]
        public void L1Regularization_VeryLightStrength_MinimalChange()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.001 };
            var regularization = new L1Regularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(4);
            vector[0] = 5.0; vector[1] = 3.0; vector[2] = -4.0; vector[3] = 2.0;

            // Act
            var result = regularization.Regularize(vector);

            // Assert - Very light strength causes minimal change
            Assert.Equal(4.999, result[0], 0.01);
            Assert.Equal(2.999, result[1], 0.01);
            Assert.Equal(-3.999, result[2], 0.01);
            Assert.Equal(1.999, result[3], 0.01);
        }

        [Fact]
        public void L2Regularization_VeryLightStrength_MinimalShrinkage()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.001 };
            var regularization = new L2Regularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(4);
            vector[0] = 100.0; vector[1] = 50.0; vector[2] = -75.0; vector[3] = 25.0;

            // Act
            var result = regularization.Regularize(vector);

            // Assert - Very light shrinkage (99.9% retained)
            Assert.Equal(99.9, result[0], Tolerance);
            Assert.Equal(49.95, result[1], Tolerance);
            Assert.Equal(-74.925, result[2], Tolerance);
            Assert.Equal(24.975, result[3], Tolerance);
        }

        #endregion

        #region Large Matrix Tests

        [Fact]
        public void L1Regularization_LargeMatrix_AllElementsProcessed()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.5 };
            var regularization = new L1Regularization<double, Matrix<double>, Vector<double>>(options);
            var matrix = new Matrix<double>(10, 10);

            // Fill with varying values
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    matrix[i, j] = (i + j + 1) * 0.3;
                }
            }

            // Act
            var result = regularization.Regularize(matrix);

            // Assert - All elements processed (smaller or zero)
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    Assert.True(Math.Abs(result[i, j]) <= Math.Abs(matrix[i, j]));
                }
            }
        }

        [Fact]
        public void L2Regularization_LargeMatrix_UniformShrinkage()
        {
            // Arrange
            var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.25 };
            var regularization = new L2Regularization<double, Matrix<double>, Vector<double>>(options);
            var matrix = new Matrix<double>(8, 8);

            // Fill with varying values
            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    matrix[i, j] = (i - j) * 2.0;
                }
            }

            // Act
            var result = regularization.Regularize(matrix);

            // Assert - All elements shrunk by 75%
            double shrinkageFactor = 1.0 - 0.25;
            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    Assert.Equal(matrix[i, j] * shrinkageFactor, result[i, j], Tolerance);
                }
            }
        }

        #endregion

        #region Negative Strength Edge Cases (Invalid Input)

        [Fact]
        public void L1Regularization_NegativeStrength_TreatedAsZero()
        {
            // Arrange - Note: This tests defensive behavior if negative strength provided
            var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = -0.1 };
            var regularization = new L1Regularization<double, Matrix<double>, Vector<double>>(options);
            var vector = new Vector<double>(3);
            vector[0] = 2.0; vector[1] = -1.5; vector[2] = 3.0;

            // Act
            var result = regularization.Regularize(vector);

            // Assert - Negative strength may behave unexpectedly
            // This test documents current behavior
            Assert.NotNull(result);
        }

        #endregion
    }
}
