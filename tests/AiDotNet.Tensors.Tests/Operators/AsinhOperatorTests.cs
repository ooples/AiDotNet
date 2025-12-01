using System;
using Xunit;

namespace AiDotNet.Tensors.Tests.Operators
{
    public class AsinhOperatorTests
    {
        private const double DoubleTolerance = 1e-14;
        private const float FloatTolerance = 1e-6f;

        #region AsinhOperatorDouble Tests

        [Fact]
        public void AsinhOperatorDouble_ScalarOperation_WithZero_ReturnsZero()
        {
            // Arrange
            double input = 0.0;
            double expected = 0.0;

            // Act
#if NET5_0_OR_GREATER
            double result = Math.Asinh(input);
#else
            double result = Math.Log(input + Math.Sqrt(input * input + 1.0));
#endif

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void AsinhOperatorDouble_ScalarOperation_WithOne_ReturnsCorrectValue()
        {
            // Arrange
            double input = 1.0;
            double expected = 0.88137358701954305; // asinh(1)

            // Act
#if NET5_0_OR_GREATER
            double result = Math.Asinh(input);
#else
            double result = Math.Log(input + Math.Sqrt(input * input + 1.0));
#endif

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void AsinhOperatorDouble_ScalarOperation_WithNegativeOne_ReturnsCorrectValue()
        {
            // Arrange
            double input = -1.0;
            double expected = -0.88137358701954305; // asinh(-1)

            // Act
#if NET5_0_OR_GREATER
            double result = Math.Asinh(input);
#else
            double result = Math.Log(input + Math.Sqrt(input * input + 1.0));
#endif

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void AsinhOperatorDouble_ScalarOperation_WithTwo_ReturnsCorrectValue()
        {
            // Arrange
            double input = 2.0;
            double expected = 1.4436354751788103; // asinh(2)

            // Act
#if NET5_0_OR_GREATER
            double result = Math.Asinh(input);
#else
            double result = Math.Log(input + Math.Sqrt(input * input + 1.0));
#endif

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void AsinhOperatorDouble_ScalarOperation_WithLargeValue_ReturnsCorrectValue()
        {
            // Arrange
            double input = 100.0;

            // Act
#if NET5_0_OR_GREATER
            double result = Math.Asinh(input);
#else
            double result = Math.Log(input + Math.Sqrt(input * input + 1.0));
#endif

            // Assert - asinh(100) = ln(100 + sqrt(100^2 + 1)) ≈ ln(200.005)
            double expected = Math.Log(100.0 + Math.Sqrt(100.0 * 100.0 + 1.0));
            Assert.Equal(expected, result, DoubleTolerance);
        }

        #endregion

        #region AsinhOperatorFloat Tests

        [Fact]
        public void AsinhOperatorFloat_ScalarOperation_WithZero_ReturnsZero()
        {
            // Arrange
            float input = 0.0f;
            float expected = 0.0f;

            // Act
#if NET5_0_OR_GREATER
            float result = MathF.Asinh(input);
#else
            float result = (float)Math.Log(input + Math.Sqrt(input * input + 1f));
#endif

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void AsinhOperatorFloat_ScalarOperation_WithOne_ReturnsCorrectValue()
        {
            // Arrange
            float input = 1.0f;
            float expected = 0.88137359f; // asinh(1)

            // Act
#if NET5_0_OR_GREATER
            float result = MathF.Asinh(input);
#else
            float result = (float)Math.Log(input + Math.Sqrt(input * input + 1f));
#endif

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void AsinhOperatorFloat_ScalarOperation_WithNegativeOne_ReturnsCorrectValue()
        {
            // Arrange
            float input = -1.0f;
            float expected = -0.88137359f; // asinh(-1)

            // Act
#if NET5_0_OR_GREATER
            float result = MathF.Asinh(input);
#else
            float result = (float)Math.Log(input + Math.Sqrt(input * input + 1f));
#endif

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void AsinhOperatorFloat_ScalarOperation_WithTwo_ReturnsCorrectValue()
        {
            // Arrange
            float input = 2.0f;
            float expected = 1.44363547f; // asinh(2)

            // Act
#if NET5_0_OR_GREATER
            float result = MathF.Asinh(input);
#else
            float result = (float)Math.Log(input + Math.Sqrt(input * input + 1f));
#endif

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        #endregion
    }
}
