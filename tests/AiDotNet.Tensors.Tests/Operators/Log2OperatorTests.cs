using System;
using Xunit;

namespace AiDotNet.Tensors.Tests.Operators
{
    public class Log2OperatorTests
    {
        private const double DoubleTolerance = 1e-14;
        private const float FloatTolerance = 1e-6f;

        #region Log2OperatorDouble Tests

        [Fact]
        public void Log2OperatorDouble_ScalarOperation_WithOne_ReturnsZero()
        {
            double input = 1.0;
            double expected = 0.0;
            double result = Math.Log2(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void Log2OperatorDouble_ScalarOperation_WithTwo_ReturnsOne()
        {
            double input = 2.0;
            double expected = 1.0;
            double result = Math.Log2(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void Log2OperatorDouble_ScalarOperation_WithFour_ReturnsTwo()
        {
            double input = 4.0;
            double expected = 2.0;
            double result = Math.Log2(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void Log2OperatorDouble_ScalarOperation_WithEight_ReturnsThree()
        {
            double input = 8.0;
            double expected = 3.0;
            double result = Math.Log2(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void Log2OperatorDouble_ScalarOperation_WithSixteen_ReturnsFour()
        {
            double input = 16.0;
            double expected = 4.0;
            double result = Math.Log2(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void Log2OperatorDouble_ScalarOperation_WithHalf_ReturnsNegativeOne()
        {
            double input = 0.5;
            double expected = -1.0;
            double result = Math.Log2(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void Log2OperatorDouble_ScalarOperation_WithQuarter_ReturnsNegativeTwo()
        {
            double input = 0.25;
            double expected = -2.0;
            double result = Math.Log2(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void Log2OperatorDouble_ScalarOperation_ExponentialProperty_IsCorrect()
        {
            // log2(2^x) = x
            double exponent = 5.0;
            double input = Math.Pow(2.0, exponent);
            double result = Math.Log2(input);
            Assert.Equal(exponent, result, DoubleTolerance);
        }

        #endregion

        #region Log2OperatorFloat Tests

        [Fact]
        public void Log2OperatorFloat_ScalarOperation_WithOne_ReturnsZero()
        {
            float input = 1.0f;
            float expected = 0.0f;
            float result = MathF.Log2(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void Log2OperatorFloat_ScalarOperation_WithTwo_ReturnsOne()
        {
            float input = 2.0f;
            float expected = 1.0f;
            float result = MathF.Log2(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void Log2OperatorFloat_ScalarOperation_WithFour_ReturnsTwo()
        {
            float input = 4.0f;
            float expected = 2.0f;
            float result = MathF.Log2(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void Log2OperatorFloat_ScalarOperation_WithEight_ReturnsThree()
        {
            float input = 8.0f;
            float expected = 3.0f;
            float result = MathF.Log2(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void Log2OperatorFloat_ScalarOperation_WithSixteen_ReturnsFour()
        {
            float input = 16.0f;
            float expected = 4.0f;
            float result = MathF.Log2(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void Log2OperatorFloat_ScalarOperation_WithHalf_ReturnsNegativeOne()
        {
            float input = 0.5f;
            float expected = -1.0f;
            float result = MathF.Log2(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void Log2OperatorFloat_ScalarOperation_WithQuarter_ReturnsNegativeTwo()
        {
            float input = 0.25f;
            float expected = -2.0f;
            float result = MathF.Log2(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void Log2OperatorFloat_ScalarOperation_ExponentialProperty_IsCorrect()
        {
            // log2(2^x) = x
            float exponent = 5.0f;
            float input = MathF.Pow(2.0f, exponent);
            float result = MathF.Log2(input);
            Assert.Equal(exponent, result, FloatTolerance);
        }

        #endregion

        #region Edge Case Tests

        [Fact]
        public void Log2OperatorDouble_WithPowerOfTwo_ReturnsExactInteger()
        {
            for (int exp = 0; exp <= 10; exp++)
            {
                double input = Math.Pow(2.0, exp);
                double result = Math.Log2(input);
                Assert.Equal((double)exp, result, DoubleTolerance);
            }
        }

        [Fact]
        public void Log2OperatorFloat_WithPowerOfTwo_ReturnsExactInteger()
        {
            for (int exp = 0; exp <= 10; exp++)
            {
                float input = MathF.Pow(2.0f, exp);
                float result = MathF.Log2(input);
                Assert.Equal((float)exp, result, FloatTolerance);
            }
        }

        [Fact]
        public void Log2OperatorDouble_WithLargeValue_ReturnsCorrectValue()
        {
            double input = 1024.0; // 2^10
            double expected = 10.0;
            double result = Math.Log2(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void Log2OperatorFloat_WithLargeValue_ReturnsCorrectValue()
        {
            float input = 1024.0f; // 2^10
            float expected = 10.0f;
            float result = MathF.Log2(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        #endregion
    }
}
