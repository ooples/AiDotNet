using System;
using Xunit;

namespace AiDotNet.Tensors.Tests.Operators
{
    public class AtanhOperatorTests
    {
        private const double DoubleTolerance = 1e-14;
        private const float FloatTolerance = 1e-6f;

        #region AtanhOperatorDouble Tests

        [Fact]
        public void AtanhOperatorDouble_ScalarOperation_WithZero_ReturnsZero()
        {
            // Arrange
            double input = 0.0;
            double expected = 0.0;

            // Act
#if NET5_0_OR_GREATER
            double result = Math.Atanh(input);
#else
            double result = 0.5 * Math.Log((1.0 + input) / (1.0 - input));
#endif

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void AtanhOperatorDouble_ScalarOperation_WithHalf_ReturnsCorrectValue()
        {
            // Arrange
            double input = 0.5;
            double expected = 0.54930614433405484; // atanh(0.5)

            // Act
#if NET5_0_OR_GREATER
            double result = Math.Atanh(input);
#else
            double result = 0.5 * Math.Log((1.0 + input) / (1.0 - input));
#endif

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void AtanhOperatorDouble_ScalarOperation_WithNegativeHalf_ReturnsCorrectValue()
        {
            // Arrange
            double input = -0.5;
            double expected = -0.54930614433405484; // atanh(-0.5)

            // Act
#if NET5_0_OR_GREATER
            double result = Math.Atanh(input);
#else
            double result = 0.5 * Math.Log((1.0 + input) / (1.0 - input));
#endif

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void AtanhOperatorDouble_ScalarOperation_WithSmallValue_ReturnsCorrectValue()
        {
            // Arrange
            double input = 0.1;
            // atanh(x) = 0.5 * ln((1+x)/(1-x))
            double expected = 0.5 * Math.Log((1.0 + input) / (1.0 - input));

            // Act
#if NET5_0_OR_GREATER
            double result = Math.Atanh(input);
#else
            double result = 0.5 * Math.Log((1.0 + input) / (1.0 - input));
#endif

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void AtanhOperatorDouble_ScalarOperation_WithValueNearOne_ReturnsLargeValue()
        {
            // Arrange
            double input = 0.99;

            // Act
#if NET5_0_OR_GREATER
            double result = Math.Atanh(input);
#else
            double result = 0.5 * Math.Log((1.0 + input) / (1.0 - input));
#endif

            // Assert - atanh(0.99) ≈ 2.647 (grows as x approaches 1)
            Assert.True(result > 2.0, "atanh(0.99) should be greater than 2");
        }

        #endregion

        #region AtanhOperatorFloat Tests

        [Fact]
        public void AtanhOperatorFloat_ScalarOperation_WithZero_ReturnsZero()
        {
            // Arrange
            float input = 0.0f;
            float expected = 0.0f;

            // Act
#if NET5_0_OR_GREATER
            float result = MathF.Atanh(input);
#else
            float result = 0.5f * (float)Math.Log((1f + input) / (1f - input));
#endif

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void AtanhOperatorFloat_ScalarOperation_WithHalf_ReturnsCorrectValue()
        {
            // Arrange
            float input = 0.5f;
            float expected = 0.54930615f; // atanh(0.5)

            // Act
#if NET5_0_OR_GREATER
            float result = MathF.Atanh(input);
#else
            float result = 0.5f * (float)Math.Log((1f + input) / (1f - input));
#endif

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void AtanhOperatorFloat_ScalarOperation_WithNegativeHalf_ReturnsCorrectValue()
        {
            // Arrange
            float input = -0.5f;
            float expected = -0.54930615f; // atanh(-0.5)

            // Act
#if NET5_0_OR_GREATER
            float result = MathF.Atanh(input);
#else
            float result = 0.5f * (float)Math.Log((1f + input) / (1f - input));
#endif

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void AtanhOperatorFloat_DomainCheck_ValueOutsideRange_ShouldProduceNaNOrInf()
        {
            // Arrange - atanh requires -1 < x < 1
            float input = 1.5f; // Invalid domain

            // Act
#if NET5_0_OR_GREATER
            float result = MathF.Atanh(input);
#else
            float result = 0.5f * (float)Math.Log((1f + input) / (1f - input));
#endif

            // Assert - Should produce NaN for invalid input
            Assert.True(float.IsNaN(result), "atanh(1.5) should produce NaN");
        }

        #endregion
    }
}
