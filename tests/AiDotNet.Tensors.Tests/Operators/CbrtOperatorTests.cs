using System;
using AiDotNet.Tensors.Tests.TestHelpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Operators
{
    public class CbrtOperatorTests
    {
        private const double DoubleTolerance = 1e-14;
        private const float FloatTolerance = 1e-6f;

        #region CbrtOperatorDouble Tests

        [Fact]
        public void CbrtOperatorDouble_ScalarOperation_WithZero_ReturnsZero()
        {
            double input = 0.0;
            double expected = 0.0;
            double result = MathCompat.Cbrt(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void CbrtOperatorDouble_ScalarOperation_WithOne_ReturnsOne()
        {
            double input = 1.0;
            double expected = 1.0;
            double result = MathCompat.Cbrt(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void CbrtOperatorDouble_ScalarOperation_WithEight_ReturnsTwo()
        {
            double input = 8.0;
            double expected = 2.0;
            double result = MathCompat.Cbrt(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void CbrtOperatorDouble_ScalarOperation_WithTwentySeven_ReturnsThree()
        {
            double input = 27.0;
            double expected = 3.0;
            double result = MathCompat.Cbrt(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void CbrtOperatorDouble_ScalarOperation_WithNegativeEight_ReturnsNegativeTwo()
        {
            double input = -8.0;
            double expected = -2.0;
            double result = MathCompat.Cbrt(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void CbrtOperatorDouble_ScalarOperation_WithSmallValue_ReturnsCorrectValue()
        {
            double input = 0.125;
            double expected = 0.5;
            double result = MathCompat.Cbrt(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void CbrtOperatorDouble_ScalarOperation_WithLargeValue_ReturnsCorrectValue()
        {
            double input = 1000.0;
            double expected = 10.0;
            double result = MathCompat.Cbrt(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void CbrtOperatorDouble_ScalarOperation_CubeProperty_IsCorrect()
        {
            // cbrt(x^3) = x
            double input = 5.0;
            double cubed = input * input * input;
            double result = MathCompat.Cbrt(cubed);
            Assert.Equal(input, result, DoubleTolerance);
        }

        #endregion

        #region CbrtOperatorFloat Tests

        [Fact]
        public void CbrtOperatorFloat_ScalarOperation_WithZero_ReturnsZero()
        {
            float input = 0.0f;
            float expected = 0.0f;
            float result = MathCompat.Cbrt(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void CbrtOperatorFloat_ScalarOperation_WithOne_ReturnsOne()
        {
            float input = 1.0f;
            float expected = 1.0f;
            float result = MathCompat.Cbrt(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void CbrtOperatorFloat_ScalarOperation_WithEight_ReturnsTwo()
        {
            float input = 8.0f;
            float expected = 2.0f;
            float result = MathCompat.Cbrt(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void CbrtOperatorFloat_ScalarOperation_WithTwentySeven_ReturnsThree()
        {
            float input = 27.0f;
            float expected = 3.0f;
            float result = MathCompat.Cbrt(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void CbrtOperatorFloat_ScalarOperation_WithNegativeEight_ReturnsNegativeTwo()
        {
            float input = -8.0f;
            float expected = -2.0f;
            float result = MathCompat.Cbrt(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void CbrtOperatorFloat_ScalarOperation_WithSmallValue_ReturnsCorrectValue()
        {
            float input = 0.125f;
            float expected = 0.5f;
            float result = MathCompat.Cbrt(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void CbrtOperatorFloat_ScalarOperation_WithLargeValue_ReturnsCorrectValue()
        {
            float input = 1000.0f;
            float expected = 10.0f;
            float result = MathCompat.Cbrt(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void CbrtOperatorFloat_ScalarOperation_CubeProperty_IsCorrect()
        {
            // cbrt(x^3) = x
            float input = 5.0f;
            float cubed = input * input * input;
            float result = MathCompat.Cbrt(cubed);
            Assert.Equal(input, result, FloatTolerance);
        }

        #endregion

        #region Edge Case Tests

        [Fact]
        public void CbrtOperatorDouble_WithNegativeCube_ReturnsNegativeRoot()
        {
            double input = -27.0;
            double expected = -3.0;
            double result = MathCompat.Cbrt(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void CbrtOperatorFloat_WithNegativeCube_ReturnsNegativeRoot()
        {
            float input = -27.0f;
            float expected = -3.0f;
            float result = MathCompat.Cbrt(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void CbrtOperatorDouble_WithFractionalPower_ReturnsCorrectValue()
        {
            double input = 64.0;
            double expected = 4.0;
            double result = MathCompat.Cbrt(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void CbrtOperatorFloat_WithFractionalPower_ReturnsCorrectValue()
        {
            float input = 64.0f;
            float expected = 4.0f;
            float result = MathCompat.Cbrt(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        #endregion
    }
}
