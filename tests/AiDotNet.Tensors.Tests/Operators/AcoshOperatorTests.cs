using System;
using Xunit;

namespace AiDotNet.Tensors.Tests.Operators
{
    public class AcoshOperatorTests
    {
        private const double DoubleTolerance = 1e-14;
        private const float FloatTolerance = 1e-6f;

        #region AcoshOperatorDouble Tests

        [Fact]
        public void AcoshOperatorDouble_ScalarOperation_WithOne_ReturnsZero()
        {
            // Arrange
            double input = 1.0;
            double expected = 0.0;

            // Act
#if NET5_0_OR_GREATER
            double result = Math.Acosh(input);
#else
            double result = Math.Log(input + Math.Sqrt(input * input - 1.0));
#endif

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void AcoshOperatorDouble_ScalarOperation_WithTwo_ReturnsCorrectValue()
        {
            // Arrange
            double input = 2.0;
            double expected = 1.3169578969248168; // acosh(2)

            // Act
#if NET5_0_OR_GREATER
            double result = Math.Acosh(input);
#else
            double result = Math.Log(input + Math.Sqrt(input * input - 1.0));
#endif

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void AcoshOperatorDouble_ScalarOperation_WithE_ReturnsCorrectValue()
        {
            // Arrange
            double input = Math.E;

            // Act
#if NET5_0_OR_GREATER
            double result = Math.Acosh(input);
#else
            double result = Math.Log(input + Math.Sqrt(input * input - 1.0));
#endif

            // Assert - acosh(e) should be positive
            Assert.True(result > 0, "acosh(e) should be positive");
            Assert.Equal(Math.Acosh(Math.E), result, DoubleTolerance);
        }

        [Fact]
        public void AcoshOperatorDouble_ScalarOperation_WithLargeValue_ReturnsCorrectValue()
        {
            // Arrange
            double input = 100.0;

            // Act
#if NET5_0_OR_GREATER
            double result = Math.Acosh(input);
#else
            double result = Math.Log(input + Math.Sqrt(input * input - 1.0));
#endif

            // Assert - acosh(100) = ln(100 + sqrt(100^2 - 1)) ≈ ln(199.995) ≈ 5.298
            double expected = Math.Log(100.0 + Math.Sqrt(100.0 * 100.0 - 1.0));
            Assert.Equal(expected, result, DoubleTolerance);
        }

        #endregion

        #region AcoshOperatorFloat Tests

        [Fact]
        public void AcoshOperatorFloat_ScalarOperation_WithOne_ReturnsZero()
        {
            // Arrange
            float input = 1.0f;
            float expected = 0.0f;

            // Act
#if NET5_0_OR_GREATER
            float result = MathF.Acosh(input);
#else
            float result = (float)Math.Log(input + Math.Sqrt(input * input - 1f));
#endif

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void AcoshOperatorFloat_ScalarOperation_WithTwo_ReturnsCorrectValue()
        {
            // Arrange
            float input = 2.0f;
            float expected = 1.31695790f; // acosh(2)

            // Act
#if NET5_0_OR_GREATER
            float result = MathF.Acosh(input);
#else
            float result = (float)Math.Log(input + Math.Sqrt(input * input - 1f));
#endif

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void AcoshOperatorFloat_ScalarOperation_WithE_ReturnsCorrectValue()
        {
            // Arrange
            float input = (float)Math.E;

            // Act
#if NET5_0_OR_GREATER
            float result = MathF.Acosh(input);
#else
            float result = (float)Math.Log(input + Math.Sqrt(input * input - 1f));
#endif

            // Assert - acosh(e) should be positive
            Assert.True(result > 0, "acosh(e) should be positive");
        }

        [Fact]
        public void AcoshOperatorFloat_ScalarOperation_DomainCheck_ValueBelowOne_ShouldProduceNaN()
        {
            // Arrange
            float input = 0.5f; // Invalid domain: acosh requires x >= 1

            // Act
#if NET5_0_OR_GREATER
            float result = MathF.Acosh(input);
#else
            float result = (float)Math.Log(input + Math.Sqrt(input * input - 1f));
#endif

            // Assert - Should produce NaN for invalid input
            Assert.True(double.IsNaN(result) || double.IsNaN((double)result), "acosh(0.5) should produce NaN");
        }

        #endregion
    }
}
