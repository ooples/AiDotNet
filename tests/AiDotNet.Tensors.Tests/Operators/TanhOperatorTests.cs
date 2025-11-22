using System;
using System.Numerics;
using System.Runtime.Intrinsics;
using Xunit;

namespace AiDotNet.Tensors.Tests.Operators
{
    public class TanhOperatorTests
    {
        private const double DoubleTolerance = 1e-14;
        private const float FloatTolerance = 1e-6f;

        #region TanhOperatorDouble Tests

        [Fact]
        public void TanhOperatorDouble_ScalarOperation_WithZero_ReturnsZero()
        {
            // Arrange
            double input = 0.0;
            double expected = 0.0;

            // Act
            double result = Math.Tanh(input);

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void TanhOperatorDouble_ScalarOperation_WithOne_ReturnsCorrectValue()
        {
            // Arrange
            double input = 1.0;
            double expected = 0.7615941559557649;

            // Act
            double result = Math.Tanh(input);

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void TanhOperatorDouble_ScalarOperation_WithNegativeOne_ReturnsCorrectValue()
        {
            // Arrange
            double input = -1.0;
            double expected = -0.7615941559557649;

            // Act
            double result = Math.Tanh(input);

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void TanhOperatorDouble_ScalarOperation_WithTwo_ReturnsCorrectValue()
        {
            // Arrange
            double input = 2.0;
            double expected = 0.9640275800758169;

            // Act
            double result = Math.Tanh(input);

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void TanhOperatorDouble_ScalarOperation_WithSmallValue_ReturnsCorrectValue()
        {
            // Arrange
            double input = 0.5;
            double expected = 0.46211715726000974;

            // Act
            double result = Math.Tanh(input);

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void TanhOperatorDouble_ScalarOperation_WithLargePositiveValue_ApproachesOne()
        {
            // Arrange
            double input = 5.0;

            // Act
            double result = Math.Tanh(input);

            // Assert - tanh approaches 1 as x approaches infinity
            Assert.True(result > 0.99);
            Assert.True(result < 1.0);
        }

        [Fact]
        public void TanhOperatorDouble_ScalarOperation_WithLargeNegativeValue_ApproachesNegativeOne()
        {
            // Arrange
            double input = -5.0;

            // Act
            double result = Math.Tanh(input);

            // Assert - tanh approaches -1 as x approaches negative infinity
            Assert.True(result < -0.99);
            Assert.True(result > -1.0);
        }

        [Fact]
        public void TanhOperatorDouble_ScalarOperation_IsOddFunction()
        {
            // Arrange
            double input = 2.5;

            // Act
            double posResult = Math.Tanh(input);
            double negResult = Math.Tanh(-input);

            // Assert - tanh(-x) should equal -tanh(x)
            Assert.Equal(-posResult, negResult, DoubleTolerance);
        }

        [Fact]
        public void TanhOperatorDouble_ScalarOperation_IsBounded()
        {
            // Arrange & Act - tanh is bounded: -1 < tanh(x) < 1
            double result1 = Math.Tanh(-10.0);
            double result2 = Math.Tanh(0.0);
            double result3 = Math.Tanh(10.0);

            // Assert
            Assert.True(result1 > -1.0 && result1 < 1.0);
            Assert.True(result2 > -1.0 && result2 < 1.0);
            Assert.True(result3 > -1.0 && result3 < 1.0);
        }

#if NET5_0_OR_GREATER
        [Fact]
        public void TanhOperatorDouble_Vector128_WithHardwareAcceleration_ReturnsCorrectValues()
        {
            if (!Vector128.IsHardwareAccelerated)
            {
                // Skip test if hardware acceleration not available
                return;
            }

            // Arrange
            Vector128<double> input = Vector128.Create(0.0, 1.0);

            // Act
            Vector128<double> result = Vector128.Create(
                Math.Tanh(input[0]),
                Math.Tanh(input[1])
            );

            // Assert
            Assert.Equal(0.0, result[0], DoubleTolerance);
            Assert.Equal(0.7615941559557649, result[1], DoubleTolerance);
        }

        [Fact]
        public void TanhOperatorDouble_Vector256_WithHardwareAcceleration_ReturnsCorrectValues()
        {
            if (!Vector256.IsHardwareAccelerated)
            {
                // Skip test if hardware acceleration not available
                return;
            }

            // Arrange
            Vector256<double> input = Vector256.Create(0.0, 1.0, -1.0, 2.0);

            // Act
            Vector256<double> result = Vector256.Create(
                Math.Tanh(input[0]),
                Math.Tanh(input[1]),
                Math.Tanh(input[2]),
                Math.Tanh(input[3])
            );

            // Assert
            Assert.Equal(0.0, result[0], DoubleTolerance);
            Assert.Equal(0.7615941559557649, result[1], DoubleTolerance);
            Assert.Equal(-0.7615941559557649, result[2], DoubleTolerance);
            Assert.Equal(0.9640275800758169, result[3], DoubleTolerance);
        }

        [Fact]
        public void TanhOperatorDouble_Vector512_WithHardwareAcceleration_ReturnsCorrectValues()
        {
            if (!Vector512.IsHardwareAccelerated)
            {
                // Skip test if hardware acceleration not available
                return;
            }

            // Arrange
            Vector512<double> input = Vector512.Create(0.0, 1.0, -1.0, 2.0, 0.5, -0.5, 3.0, -3.0);

            // Act
            Vector512<double> result = Vector512.Create(
                Math.Tanh(input[0]),
                Math.Tanh(input[1]),
                Math.Tanh(input[2]),
                Math.Tanh(input[3]),
                Math.Tanh(input[4]),
                Math.Tanh(input[5]),
                Math.Tanh(input[6]),
                Math.Tanh(input[7])
            );

            // Assert
            Assert.Equal(0.0, result[0], DoubleTolerance);
            Assert.Equal(0.7615941559557649, result[1], DoubleTolerance);
            Assert.Equal(-0.7615941559557649, result[2], DoubleTolerance);
            Assert.Equal(0.9640275800758169, result[3], DoubleTolerance);
            Assert.Equal(0.46211715726000974, result[4], DoubleTolerance);
            Assert.Equal(-0.46211715726000974, result[5], DoubleTolerance);
        }
#endif

        #endregion

        #region TanhOperatorFloat Tests

        [Fact]
        public void TanhOperatorFloat_ScalarOperation_WithZero_ReturnsZero()
        {
            // Arrange
            float input = 0.0f;
            float expected = 0.0f;

            // Act
            float result = MathF.Tanh(input);

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void TanhOperatorFloat_ScalarOperation_WithOne_ReturnsCorrectValue()
        {
            // Arrange
            float input = 1.0f;
            float expected = 0.7615942f;

            // Act
            float result = MathF.Tanh(input);

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void TanhOperatorFloat_ScalarOperation_WithNegativeOne_ReturnsCorrectValue()
        {
            // Arrange
            float input = -1.0f;
            float expected = -0.7615942f;

            // Act
            float result = MathF.Tanh(input);

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void TanhOperatorFloat_ScalarOperation_WithTwo_ReturnsCorrectValue()
        {
            // Arrange
            float input = 2.0f;
            float expected = 0.9640276f;

            // Act
            float result = MathF.Tanh(input);

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void TanhOperatorFloat_ScalarOperation_WithSmallValue_ReturnsCorrectValue()
        {
            // Arrange
            float input = 0.5f;
            float expected = 0.46211717f;

            // Act
            float result = MathF.Tanh(input);

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void TanhOperatorFloat_ScalarOperation_WithLargePositiveValue_ApproachesOne()
        {
            // Arrange
            float input = 5.0f;

            // Act
            float result = MathF.Tanh(input);

            // Assert - tanh approaches 1 as x approaches infinity
            Assert.True(result > 0.99f);
            Assert.True(result < 1.0f);
        }

        [Fact]
        public void TanhOperatorFloat_ScalarOperation_WithLargeNegativeValue_ApproachesNegativeOne()
        {
            // Arrange
            float input = -5.0f;

            // Act
            float result = MathF.Tanh(input);

            // Assert - tanh approaches -1 as x approaches negative infinity
            Assert.True(result < -0.99f);
            Assert.True(result > -1.0f);
        }

        [Fact]
        public void TanhOperatorFloat_ScalarOperation_IsOddFunction()
        {
            // Arrange
            float input = 2.5f;

            // Act
            float posResult = MathF.Tanh(input);
            float negResult = MathF.Tanh(-input);

            // Assert - tanh(-x) should equal -tanh(x)
            Assert.Equal(-posResult, negResult, FloatTolerance);
        }

        [Fact]
        public void TanhOperatorFloat_ScalarOperation_IsBounded()
        {
            // Arrange & Act - tanh is bounded: -1 < tanh(x) < 1
            float result1 = MathF.Tanh(-10.0f);
            float result2 = MathF.Tanh(0.0f);
            float result3 = MathF.Tanh(10.0f);

            // Assert
            Assert.True(result1 > -1.0f && result1 < 1.0f);
            Assert.True(result2 > -1.0f && result2 < 1.0f);
            Assert.True(result3 > -1.0f && result3 < 1.0f);
        }

#if NET5_0_OR_GREATER
        [Fact]
        public void TanhOperatorFloat_Vector128_WithHardwareAcceleration_ReturnsCorrectValues()
        {
            if (!Vector128.IsHardwareAccelerated)
            {
                // Skip test if hardware acceleration not available
                return;
            }

            // Arrange
            Vector128<float> input = Vector128.Create(0.0f, 1.0f, -1.0f, 2.0f);

            // Act
            Vector128<float> result = Vector128.Create(
                MathF.Tanh(input[0]),
                MathF.Tanh(input[1]),
                MathF.Tanh(input[2]),
                MathF.Tanh(input[3])
            );

            // Assert
            Assert.Equal(0.0f, result[0], FloatTolerance);
            Assert.Equal(0.7615942f, result[1], FloatTolerance);
            Assert.Equal(-0.7615942f, result[2], FloatTolerance);
            Assert.Equal(0.9640276f, result[3], FloatTolerance);
        }

        [Fact]
        public void TanhOperatorFloat_Vector256_WithHardwareAcceleration_ReturnsCorrectValues()
        {
            if (!Vector256.IsHardwareAccelerated)
            {
                // Skip test if hardware acceleration not available
                return;
            }

            // Arrange
            Vector256<float> input = Vector256.Create(0.0f, 1.0f, -1.0f, 2.0f, 0.5f, -0.5f, 3.0f, -3.0f);

            // Act
            Vector256<float> result = Vector256.Create(
                MathF.Tanh(input[0]),
                MathF.Tanh(input[1]),
                MathF.Tanh(input[2]),
                MathF.Tanh(input[3]),
                MathF.Tanh(input[4]),
                MathF.Tanh(input[5]),
                MathF.Tanh(input[6]),
                MathF.Tanh(input[7])
            );

            // Assert
            Assert.Equal(0.0f, result[0], FloatTolerance);
            Assert.Equal(0.7615942f, result[1], FloatTolerance);
            Assert.Equal(-0.7615942f, result[2], FloatTolerance);
            Assert.Equal(0.9640276f, result[3], FloatTolerance);
            Assert.Equal(0.46211717f, result[4], FloatTolerance);
            Assert.Equal(-0.46211717f, result[5], FloatTolerance);
        }

        [Fact]
        public void TanhOperatorFloat_Vector512_WithHardwareAcceleration_ReturnsCorrectValues()
        {
            if (!Vector512.IsHardwareAccelerated)
            {
                // Skip test if hardware acceleration not available
                return;
            }

            // Arrange
            Vector512<float> input = Vector512.Create(
                0.0f, 1.0f, -1.0f, 2.0f,
                0.5f, -0.5f, 3.0f, -3.0f,
                0.25f, -0.25f, 1.5f, -1.5f,
                4.0f, -4.0f, 0.1f, -0.1f
            );

            // Act
            Vector512<float> result = Vector512.Create(
                MathF.Tanh(input[0]),
                MathF.Tanh(input[1]),
                MathF.Tanh(input[2]),
                MathF.Tanh(input[3]),
                MathF.Tanh(input[4]),
                MathF.Tanh(input[5]),
                MathF.Tanh(input[6]),
                MathF.Tanh(input[7]),
                MathF.Tanh(input[8]),
                MathF.Tanh(input[9]),
                MathF.Tanh(input[10]),
                MathF.Tanh(input[11]),
                MathF.Tanh(input[12]),
                MathF.Tanh(input[13]),
                MathF.Tanh(input[14]),
                MathF.Tanh(input[15])
            );

            // Assert
            Assert.Equal(0.0f, result[0], FloatTolerance);
            Assert.Equal(0.7615942f, result[1], FloatTolerance);
            Assert.Equal(-0.7615942f, result[2], FloatTolerance);
            Assert.Equal(0.9640276f, result[3], FloatTolerance);
        }
#endif

        #endregion

        #region Edge Case Tests

        [Fact]
        public void TanhOperatorDouble_WithVerySmallValue_ReturnsApproximatelyInput()
        {
            // Arrange - for very small x, tanh(x) ≈ x
            double input = 1e-10;

            // Act
            double result = Math.Tanh(input);

            // Assert
            Assert.Equal(input, result, 1e-15);
        }

        [Fact]
        public void TanhOperatorFloat_WithVerySmallValue_ReturnsApproximatelyInput()
        {
            // Arrange - for very small x, tanh(x) ≈ x
            float input = 1e-7f;

            // Act
            float result = MathF.Tanh(input);

            // Assert
            Assert.Equal(input, result, 1e-10f);
        }

        [Fact]
        public void TanhOperatorDouble_RelationshipToSinhCosh_IsCorrect()
        {
            // Arrange - tanh(x) = sinh(x) / cosh(x)
            double input = 1.5;

            // Act
            double tanhResult = Math.Tanh(input);
            double expectedFromRatio = Math.Sinh(input) / Math.Cosh(input);

            // Assert
            Assert.Equal(expectedFromRatio, tanhResult, DoubleTolerance);
        }

        [Fact]
        public void TanhOperatorFloat_RelationshipToSinhCosh_IsCorrect()
        {
            // Arrange - tanh(x) = sinh(x) / cosh(x)
            float input = 1.5f;

            // Act
            float tanhResult = MathF.Tanh(input);
            float expectedFromRatio = MathF.Sinh(input) / MathF.Cosh(input);

            // Assert
            Assert.Equal(expectedFromRatio, tanhResult, FloatTolerance);
        }

        [Fact]
        public void TanhOperatorDouble_AsActivationFunction_HasGoodGradient()
        {
            // Arrange - tanh derivative at 0 is 1 (sech^2(0) = 1)
            double input = 0.0;

            // Act - derivative of tanh(x) = 1 - tanh^2(x)
            double tanhValue = Math.Tanh(input);
            double derivative = 1.0 - tanhValue * tanhValue;

            // Assert
            Assert.Equal(1.0, derivative, DoubleTolerance);
        }

        [Fact]
        public void TanhOperatorFloat_AsActivationFunction_HasGoodGradient()
        {
            // Arrange - tanh derivative at 0 is 1 (sech^2(0) = 1)
            float input = 0.0f;

            // Act - derivative of tanh(x) = 1 - tanh^2(x)
            float tanhValue = MathF.Tanh(input);
            float derivative = 1.0f - tanhValue * tanhValue;

            // Assert
            Assert.Equal(1.0f, derivative, FloatTolerance);
        }

        [Fact]
        public void TanhOperatorDouble_ExponentialFormula_IsCorrect()
        {
            // Arrange - tanh(x) = (e^x - e^-x) / (e^x + e^-x)
            double input = 1.5;

            // Act
            double tanhResult = Math.Tanh(input);
            double expPos = Math.Exp(input);
            double expNeg = Math.Exp(-input);
            double expectedFromExp = (expPos - expNeg) / (expPos + expNeg);

            // Assert
            Assert.Equal(expectedFromExp, tanhResult, DoubleTolerance);
        }

        [Fact]
        public void TanhOperatorFloat_ExponentialFormula_IsCorrect()
        {
            // Arrange - tanh(x) = (e^x - e^-x) / (e^x + e^-x)
            float input = 1.5f;

            // Act
            float tanhResult = MathF.Tanh(input);
            float expPos = MathF.Exp(input);
            float expNeg = MathF.Exp(-input);
            float expectedFromExp = (expPos - expNeg) / (expPos + expNeg);

            // Assert
            Assert.Equal(expectedFromExp, tanhResult, FloatTolerance);
        }

        #endregion
    }
}
