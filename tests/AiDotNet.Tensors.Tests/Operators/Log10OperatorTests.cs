using System;
using Xunit;

namespace AiDotNet.Tensors.Tests.Operators
{
    public class Log10OperatorTests
    {
        private const double DoubleTolerance = 1e-14;
        private const float FloatTolerance = 1e-6f;

        #region Log10OperatorDouble Tests

        [Fact]
        public void Log10OperatorDouble_ScalarOperation_WithOne_ReturnsZero()
        {
            double input = 1.0;
            double expected = 0.0;
            double result = Math.Log10(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void Log10OperatorDouble_ScalarOperation_WithTen_ReturnsOne()
        {
            double input = 10.0;
            double expected = 1.0;
            double result = Math.Log10(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void Log10OperatorDouble_ScalarOperation_WithHundred_ReturnsTwo()
        {
            double input = 100.0;
            double expected = 2.0;
            double result = Math.Log10(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void Log10OperatorDouble_ScalarOperation_WithThousand_ReturnsThree()
        {
            double input = 1000.0;
            double expected = 3.0;
            double result = Math.Log10(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void Log10OperatorDouble_ScalarOperation_WithTenThousand_ReturnsFour()
        {
            double input = 10000.0;
            double expected = 4.0;
            double result = Math.Log10(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void Log10OperatorDouble_ScalarOperation_WithTenth_ReturnsNegativeOne()
        {
            double input = 0.1;
            double expected = -1.0;
            double result = Math.Log10(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void Log10OperatorDouble_ScalarOperation_WithHundredth_ReturnsNegativeTwo()
        {
            double input = 0.01;
            double expected = -2.0;
            double result = Math.Log10(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void Log10OperatorDouble_ScalarOperation_ExponentialProperty_IsCorrect()
        {
            // log10(10^x) = x
            double exponent = 5.0;
            double input = Math.Pow(10.0, exponent);
            double result = Math.Log10(input);
            Assert.Equal(exponent, result, DoubleTolerance);
        }

        #endregion

        #region Log10OperatorFloat Tests

        [Fact]
        public void Log10OperatorFloat_ScalarOperation_WithOne_ReturnsZero()
        {
            float input = 1.0f;
            float expected = 0.0f;
            float result = MathF.Log10(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void Log10OperatorFloat_ScalarOperation_WithTen_ReturnsOne()
        {
            float input = 10.0f;
            float expected = 1.0f;
            float result = MathF.Log10(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void Log10OperatorFloat_ScalarOperation_WithHundred_ReturnsTwo()
        {
            float input = 100.0f;
            float expected = 2.0f;
            float result = MathF.Log10(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void Log10OperatorFloat_ScalarOperation_WithThousand_ReturnsThree()
        {
            float input = 1000.0f;
            float expected = 3.0f;
            float result = MathF.Log10(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void Log10OperatorFloat_ScalarOperation_WithTenThousand_ReturnsFour()
        {
            float input = 10000.0f;
            float expected = 4.0f;
            float result = MathF.Log10(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void Log10OperatorFloat_ScalarOperation_WithTenth_ReturnsNegativeOne()
        {
            float input = 0.1f;
            float expected = -1.0f;
            float result = MathF.Log10(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void Log10OperatorFloat_ScalarOperation_WithHundredth_ReturnsNegativeTwo()
        {
            float input = 0.01f;
            float expected = -2.0f;
            float result = MathF.Log10(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void Log10OperatorFloat_ScalarOperation_ExponentialProperty_IsCorrect()
        {
            // log10(10^x) = x
            float exponent = 5.0f;
            float input = MathF.Pow(10.0f, exponent);
            float result = MathF.Log10(input);
            Assert.Equal(exponent, result, FloatTolerance);
        }

        #endregion

        #region Edge Case Tests

        [Fact]
        public void Log10OperatorDouble_WithPowerOfTen_ReturnsExactInteger()
        {
            for (int exp = 0; exp <= 6; exp++)
            {
                double input = Math.Pow(10.0, exp);
                double result = Math.Log10(input);
                Assert.Equal((double)exp, result, DoubleTolerance);
            }
        }

        [Fact]
        public void Log10OperatorFloat_WithPowerOfTen_ReturnsExactInteger()
        {
            for (int exp = 0; exp <= 6; exp++)
            {
                float input = MathF.Pow(10.0f, exp);
                float result = MathF.Log10(input);
                Assert.Equal((float)exp, result, FloatTolerance);
            }
        }

        [Fact]
        public void Log10OperatorDouble_WithLargeValue_ReturnsCorrectValue()
        {
            double input = 1000000.0; // 10^6
            double expected = 6.0;
            double result = Math.Log10(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void Log10OperatorFloat_WithLargeValue_ReturnsCorrectValue()
        {
            float input = 1000000.0f; // 10^6
            float expected = 6.0f;
            float result = MathF.Log10(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        #endregion
    }
}
