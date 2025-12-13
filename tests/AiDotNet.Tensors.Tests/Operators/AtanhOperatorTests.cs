using System;
using AiDotNet.Tensors.Tests.TestHelpers;
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
            double input = 0.0;
            double expected = 0.0;
            double result = MathCompat.Atanh(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void AtanhOperatorDouble_ScalarOperation_WithHalf_ReturnsCorrectValue()
        {
            double input = 0.5;
            double expected = 0.54930614433405484; // atanh(0.5)
            double result = MathCompat.Atanh(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void AtanhOperatorDouble_ScalarOperation_WithNegativeHalf_ReturnsCorrectValue()
        {
            double input = -0.5;
            double expected = -0.54930614433405484; // atanh(-0.5)
            double result = MathCompat.Atanh(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void AtanhOperatorDouble_ScalarOperation_WithSmallValue_ReturnsCorrectValue()
        {
            double input = 0.1;
            // atanh(x) = 0.5 * ln((1+x)/(1-x))
            double expected = 0.5 * Math.Log((1.0 + input) / (1.0 - input));
            double result = MathCompat.Atanh(input);
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void AtanhOperatorDouble_ScalarOperation_WithValueNearOne_ReturnsLargeValue()
        {
            double input = 0.99;
            double result = MathCompat.Atanh(input);
            // atanh(0.99) grows as x approaches 1
            Assert.True(result > 2.0, "atanh(0.99) should be greater than 2");
        }

        #endregion

        #region AtanhOperatorFloat Tests

        [Fact]
        public void AtanhOperatorFloat_ScalarOperation_WithZero_ReturnsZero()
        {
            float input = 0.0f;
            float expected = 0.0f;
            float result = MathCompat.Atanh(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void AtanhOperatorFloat_ScalarOperation_WithHalf_ReturnsCorrectValue()
        {
            float input = 0.5f;
            float expected = 0.54930615f; // atanh(0.5)
            float result = MathCompat.Atanh(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void AtanhOperatorFloat_ScalarOperation_WithNegativeHalf_ReturnsCorrectValue()
        {
            float input = -0.5f;
            float expected = -0.54930615f; // atanh(-0.5)
            float result = MathCompat.Atanh(input);
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void AtanhOperatorFloat_DomainCheck_ValueOutsideRange_ShouldProduceNaNOrInf()
        {
            // atanh requires -1 < x < 1
            float input = 1.5f; // Invalid domain
            float result = MathCompat.Atanh(input);
            // Should produce NaN for invalid input
            Assert.True(float.IsNaN(result), "atanh(1.5) should produce NaN");
        }

        #endregion
    }
}
