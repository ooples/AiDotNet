using System;
using AiDotNet.Tensors.Tests.TestHelpers;
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
            double input = 1.0;
            double expected = 0.0;
            double result = MathCompat.Acosh(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void AcoshOperatorDouble_ScalarOperation_WithTwo_ReturnsCorrectValue()
        {
            double input = 2.0;
            double expected = 1.3169578969248168; // acosh(2)
            double result = MathCompat.Acosh(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void AcoshOperatorDouble_ScalarOperation_WithE_ReturnsCorrectValue()
        {
            double input = Math.E;
            double result = MathCompat.Acosh(input);
            // acosh(e) should be positive
            Assert.True(result > 0, "acosh(e) should be positive");
            double expected = MathCompat.Acosh(Math.E);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void AcoshOperatorDouble_ScalarOperation_WithLargeValue_ReturnsCorrectValue()
        {
            double input = 100.0;
            double result = MathCompat.Acosh(input);
            // acosh(100) = ln(100 + sqrt(100^2 - 1))
            double expected = Math.Log(100.0 + Math.Sqrt(100.0 * 100.0 - 1.0));
            Assert.Equal(expected, result, DoubleTolerance);
        }

        #endregion

        #region AcoshOperatorFloat Tests

        [Fact]
        public void AcoshOperatorFloat_ScalarOperation_WithOne_ReturnsZero()
        {
            float input = 1.0f;
            float expected = 0.0f;
            float result = MathCompat.Acosh(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void AcoshOperatorFloat_ScalarOperation_WithTwo_ReturnsCorrectValue()
        {
            float input = 2.0f;
            float expected = 1.31695790f; // acosh(2)
            float result = MathCompat.Acosh(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void AcoshOperatorFloat_ScalarOperation_WithE_ReturnsCorrectValue()
        {
            float input = (float)Math.E;
            float result = MathCompat.Acosh(input);
            // acosh(e) should be positive
            Assert.True(result > 0, "acosh(e) should be positive");
        }

        [Fact]
        public void AcoshOperatorFloat_ScalarOperation_DomainCheck_ValueBelowOne_ShouldProduceNaN()
        {
            float input = 0.5f; // Invalid domain: acosh requires x >= 1
            float result = MathCompat.Acosh(input);
            // Should produce NaN for invalid input
            Assert.True(float.IsNaN(result), "acosh(0.5) should produce NaN");
        }

        #endregion
    }
}
