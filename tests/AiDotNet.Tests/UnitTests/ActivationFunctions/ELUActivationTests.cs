using System;
using AiDotNet.ActivationFunctions;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.ActivationFunctions
{
    public class ELUActivationTests
    {
        [Fact]
        public void Constructor_WithDefaultAlpha_UsesOneAsDefault()
        {
            // Arrange & Act
            var activation = new ELUActivation<double>();

            // Assert - Verify by testing activation behavior
            var result = activation.Activate(-10.0);
            Assert.True(result > -1.1);  // Should approach -alpha = -1.0
            Assert.True(result < 0.0);
        }

        [Fact]
        public void Constructor_WithCustomAlpha_UsesSpecifiedValue()
        {
            // Arrange & Act
            var activation = new ELUActivation<double>(2.0);

            // Assert - Verify by testing activation behavior
            var result = activation.Activate(-10.0);
            Assert.True(result > -2.1);  // Should approach -alpha = -2.0
            Assert.True(result < 0.0);
        }

        [Fact]
        public void Activate_WithPositiveValue_ReturnsInputUnchanged()
        {
            // Arrange
            var activation = new ELUActivation<double>();

            // Act
            var result1 = activation.Activate(1.0);
            var result2 = activation.Activate(5.0);
            var result3 = activation.Activate(100.0);

            // Assert
            Assert.Equal(1.0, result1, 10);
            Assert.Equal(5.0, result2, 10);
            Assert.Equal(100.0, result3, 10);
        }

        [Fact]
        public void Activate_WithZero_ReturnsZero()
        {
            // Arrange
            var activation = new ELUActivation<double>();

            // Act
            var result = activation.Activate(0.0);

            // Assert
            Assert.Equal(0.0, result, 10);
        }

        [Fact]
        public void Activate_WithNegativeValue_ReturnsExponentialCurve()
        {
            // Arrange
            var activation = new ELUActivation<double>(1.0);

            // Act
            var result1 = activation.Activate(-1.0);
            var result2 = activation.Activate(-2.0);

            // Assert
            // ELU(-1) = alpha * (e^-1 - 1) = 1 * (0.3678... - 1) = -0.6321...
            Assert.Equal(-0.6321205588285577, result1, 10);
            // ELU(-2) = alpha * (e^-2 - 1) = 1 * (0.1353... - 1) = -0.8646...
            Assert.Equal(-0.8646647167633873, result2, 10);
        }

        [Fact]
        public void Activate_WithLargeNegativeValue_ApproachesNegativeAlpha()
        {
            // Arrange
            var activation = new ELUActivation<double>(1.0);

            // Act
            var result = activation.Activate(-10.0);

            // Assert
            // Should be very close to -alpha = -1.0
            Assert.True(result > -1.0001);
            Assert.True(result < -0.999);
        }

        [Fact]
        public void Activate_Vector_AppliesActivationToAllElements()
        {
            // Arrange
            var activation = new ELUActivation<double>(1.0);
            var input = new Vector<double>(new double[] { -2.0, -1.0, 0.0, 1.0, 2.0 });

            // Act
            var result = activation.Activate(input);

            // Assert
            Assert.Equal(5, result.Length);
            Assert.True(result[0] < 0.0);  // Negative input
            Assert.True(result[1] < 0.0);  // Negative input
            Assert.Equal(0.0, result[2], 10);  // Zero input
            Assert.Equal(1.0, result[3], 10);  // Positive input
            Assert.Equal(2.0, result[4], 10);  // Positive input
        }

        [Fact]
        public void Derivative_WithPositiveValue_ReturnsOne()
        {
            // Arrange
            var activation = new ELUActivation<double>();

            // Act
            var result1 = activation.Derivative(1.0);
            var result2 = activation.Derivative(5.0);
            var result3 = activation.Derivative(100.0);

            // Assert
            Assert.Equal(1.0, result1, 10);
            Assert.Equal(1.0, result2, 10);
            Assert.Equal(1.0, result3, 10);
        }

        [Fact]
        public void Derivative_WithZero_ReturnsAlpha()
        {
            // Arrange
            var activation = new ELUActivation<double>(1.0);

            // Act
            var result = activation.Derivative(0.0);

            // Assert
            // At x=0: derivative = ELU(0) + alpha = 0 + 1 = 1
            Assert.Equal(1.0, result, 10);
        }

        [Fact]
        public void Derivative_WithNegativeValue_ReturnsCorrectValue()
        {
            // Arrange
            var activation = new ELUActivation<double>(1.0);

            // Act
            var result1 = activation.Derivative(-1.0);
            var result2 = activation.Derivative(-2.0);

            // Assert
            // derivative = ELU(x) + alpha
            // At x=-1: ELU(-1) + 1 = -0.6321... + 1 = 0.3678...
            Assert.True(result1 > 0.0);
            Assert.True(result1 < 1.0);
            // At x=-2: should be smaller than at x=-1
            Assert.True(result2 < result1);
        }

        [Fact]
        public void Derivative_Vector_ReturnsJacobianMatrix()
        {
            // Arrange
            var activation = new ELUActivation<double>(1.0);
            var input = new Vector<double>(new double[] { -1.0, 0.0, 1.0 });

            // Act
            var result = activation.Derivative(input);

            // Assert
            Assert.Equal(3, result.Rows);
            Assert.Equal(3, result.Columns);
            // Diagonal elements should contain derivatives
            Assert.True(result[0, 0] > 0.0);
            Assert.Equal(1.0, result[1, 1], 10);
            Assert.Equal(1.0, result[2, 2], 10);
            // Off-diagonal elements should be zero
            Assert.Equal(0.0, result[0, 1], 10);
            Assert.Equal(0.0, result[0, 2], 10);
            Assert.Equal(0.0, result[1, 0], 10);
        }

        [Fact]
        public void Activate_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var activation = new ELUActivation<float>(1.0f);

            // Act
            var result1 = activation.Activate(2.0f);
            var result2 = activation.Activate(-1.0f);

            // Assert
            Assert.Equal(2.0f, result1, 5);
            Assert.True(result2 < 0.0f);
        }

        [Fact]
        public void Derivative_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var activation = new ELUActivation<float>(1.0f);

            // Act
            var result1 = activation.Derivative(2.0f);
            var result2 = activation.Derivative(-1.0f);

            // Assert
            Assert.Equal(1.0f, result1, 5);
            Assert.True(result2 > 0.0f && result2 < 1.0f);
        }

        [Fact]
        public void Activate_WithDifferentAlpha_AffectsNegativeValues()
        {
            // Arrange
            var activation1 = new ELUActivation<double>(1.0);
            var activation2 = new ELUActivation<double>(2.0);
            var input = -2.0;

            // Act
            var result1 = activation1.Activate(input);
            var result2 = activation2.Activate(input);

            // Assert
            // Larger alpha should give more negative values
            Assert.True(result2 < result1);
        }

        [Fact]
        public void Activate_PreventsDyingNeurons_UnlikeReLU()
        {
            // Arrange
            var activation = new ELUActivation<double>(1.0);
            var negativeInput = -5.0;

            // Act
            var result = activation.Activate(negativeInput);
            var derivative = activation.Derivative(negativeInput);

            // Assert
            // ELU has non-zero outputs and gradients for negative inputs
            Assert.NotEqual(0.0, result);
            Assert.NotEqual(0.0, derivative);
            Assert.True(derivative > 0.0);  // Gradient exists for backpropagation
        }

        [Fact]
        public void Activate_IsContinuous_AtZero()
        {
            // Arrange
            var activation = new ELUActivation<double>(1.0);
            var epsilon = 0.0001;

            // Act
            var leftLimit = activation.Activate(-epsilon);
            var rightLimit = activation.Activate(epsilon);
            var atZero = activation.Activate(0.0);

            // Assert
            // Function should be continuous at x=0
            Assert.True(Math.Abs(leftLimit - atZero) < 0.01);
            Assert.True(Math.Abs(rightLimit - atZero) < 0.01);
        }
    }
}
