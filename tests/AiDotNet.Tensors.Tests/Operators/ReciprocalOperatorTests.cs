using System;
using System.Numerics;
using System.Runtime.Intrinsics;
using Xunit;

namespace AiDotNet.Tensors.Tests.Operators
{
    public class ReciprocalOperatorTests
    {
        private const double DoubleTolerance = 1e-14;
        private const float FloatTolerance = 1e-6f;

        #region ReciprocalOperatorDouble Tests

        [Fact]
        public void ReciprocalOperatorDouble_ScalarOperation_WithOne_ReturnsOne()
        {
            // Arrange
            double input = 1.0;
            double expected = 1.0;

            // Act
            double result = 1.0 / input;

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void ReciprocalOperatorDouble_ScalarOperation_WithTwo_ReturnsHalf()
        {
            // Arrange
            double input = 2.0;
            double expected = 0.5;

            // Act
            double result = 1.0 / input;

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void ReciprocalOperatorDouble_ScalarOperation_WithNegativeOne_ReturnsNegativeOne()
        {
            // Arrange
            double input = -1.0;
            double expected = -1.0;

            // Act
            double result = 1.0 / input;

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void ReciprocalOperatorDouble_ScalarOperation_WithFour_ReturnsQuarter()
        {
            // Arrange
            double input = 4.0;
            double expected = 0.25;

            // Act
            double result = 1.0 / input;

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void ReciprocalOperatorDouble_ScalarOperation_WithSmallValue_ReturnsLargeValue()
        {
            // Arrange
            double input = 0.1;
            double expected = 10.0;

            // Act
            double result = 1.0 / input;

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void ReciprocalOperatorDouble_ScalarOperation_WithLargeValue_ReturnsSmallValue()
        {
            // Arrange
            double input = 100.0;
            double expected = 0.01;

            // Act
            double result = 1.0 / input;

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void ReciprocalOperatorDouble_ScalarOperation_IsInvolutive()
        {
            // Arrange
            double input = 2.5;

            // Act
            double reciprocal = 1.0 / input;
            double doubleReciprocal = 1.0 / reciprocal;

            // Assert - 1/(1/x) should equal x
            Assert.Equal(input, doubleReciprocal, DoubleTolerance);
        }

#if NET5_0_OR_GREATER
        [Fact]
        public void ReciprocalOperatorDouble_Vector128_WithHardwareAcceleration_ReturnsCorrectValues()
        {
            if (!Vector128.IsHardwareAccelerated)
            {
                return;
            }

            // Arrange
            Vector128<double> input = Vector128.Create(1.0, 2.0);

            // Act
            Vector128<double> result = Vector128.Create(
                1.0 / input[0],
                1.0 / input[1]
            );

            // Assert
            Assert.Equal(1.0, result[0], DoubleTolerance);
            Assert.Equal(0.5, result[1], DoubleTolerance);
        }

        [Fact]
        public void ReciprocalOperatorDouble_Vector256_WithHardwareAcceleration_ReturnsCorrectValues()
        {
            if (!Vector256.IsHardwareAccelerated)
            {
                return;
            }

            // Arrange
            Vector256<double> input = Vector256.Create(1.0, 2.0, 4.0, 10.0);

            // Act
            Vector256<double> result = Vector256.Create(
                1.0 / input[0],
                1.0 / input[1],
                1.0 / input[2],
                1.0 / input[3]
            );

            // Assert
            Assert.Equal(1.0, result[0], DoubleTolerance);
            Assert.Equal(0.5, result[1], DoubleTolerance);
            Assert.Equal(0.25, result[2], DoubleTolerance);
            Assert.Equal(0.1, result[3], DoubleTolerance);
        }

        [Fact]
        public void ReciprocalOperatorDouble_Vector512_WithHardwareAcceleration_ReturnsCorrectValues()
        {
            if (!Vector512.IsHardwareAccelerated)
            {
                return;
            }

            // Arrange
            Vector512<double> input = Vector512.Create(1.0, 2.0, 4.0, 10.0, 0.5, 0.25, 100.0, 0.1);

            // Act
            Vector512<double> result = Vector512.Create(
                1.0 / input[0],
                1.0 / input[1],
                1.0 / input[2],
                1.0 / input[3],
                1.0 / input[4],
                1.0 / input[5],
                1.0 / input[6],
                1.0 / input[7]
            );

            // Assert
            Assert.Equal(1.0, result[0], DoubleTolerance);
            Assert.Equal(0.5, result[1], DoubleTolerance);
            Assert.Equal(0.25, result[2], DoubleTolerance);
            Assert.Equal(0.1, result[3], DoubleTolerance);
            Assert.Equal(2.0, result[4], DoubleTolerance);
            Assert.Equal(4.0, result[5], DoubleTolerance);
        }
#endif

        #endregion

        #region ReciprocalOperatorFloat Tests

        [Fact]
        public void ReciprocalOperatorFloat_ScalarOperation_WithOne_ReturnsOne()
        {
            // Arrange
            float input = 1.0f;
            float expected = 1.0f;

            // Act
            float result = 1.0f / input;

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void ReciprocalOperatorFloat_ScalarOperation_WithTwo_ReturnsHalf()
        {
            // Arrange
            float input = 2.0f;
            float expected = 0.5f;

            // Act
            float result = 1.0f / input;

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void ReciprocalOperatorFloat_ScalarOperation_WithNegativeOne_ReturnsNegativeOne()
        {
            // Arrange
            float input = -1.0f;
            float expected = -1.0f;

            // Act
            float result = 1.0f / input;

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void ReciprocalOperatorFloat_ScalarOperation_WithFour_ReturnsQuarter()
        {
            // Arrange
            float input = 4.0f;
            float expected = 0.25f;

            // Act
            float result = 1.0f / input;

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void ReciprocalOperatorFloat_ScalarOperation_WithSmallValue_ReturnsLargeValue()
        {
            // Arrange
            float input = 0.1f;
            float expected = 10.0f;

            // Act
            float result = 1.0f / input;

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void ReciprocalOperatorFloat_ScalarOperation_WithLargeValue_ReturnsSmallValue()
        {
            // Arrange
            float input = 100.0f;
            float expected = 0.01f;

            // Act
            float result = 1.0f / input;

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void ReciprocalOperatorFloat_ScalarOperation_IsInvolutive()
        {
            // Arrange
            float input = 2.5f;

            // Act
            float reciprocal = 1.0f / input;
            float doubleReciprocal = 1.0f / reciprocal;

            // Assert - 1/(1/x) should equal x
            Assert.Equal(input, doubleReciprocal, FloatTolerance);
        }

#if NET5_0_OR_GREATER
        [Fact]
        public void ReciprocalOperatorFloat_Vector128_WithHardwareAcceleration_ReturnsCorrectValues()
        {
            if (!Vector128.IsHardwareAccelerated)
            {
                return;
            }

            // Arrange
            Vector128<float> input = Vector128.Create(1.0f, 2.0f, 4.0f, 10.0f);

            // Act
            Vector128<float> result = Vector128.Create(
                1.0f / input[0],
                1.0f / input[1],
                1.0f / input[2],
                1.0f / input[3]
            );

            // Assert
            Assert.Equal(1.0f, result[0], FloatTolerance);
            Assert.Equal(0.5f, result[1], FloatTolerance);
            Assert.Equal(0.25f, result[2], FloatTolerance);
            Assert.Equal(0.1f, result[3], FloatTolerance);
        }

        [Fact]
        public void ReciprocalOperatorFloat_Vector256_WithHardwareAcceleration_ReturnsCorrectValues()
        {
            if (!Vector256.IsHardwareAccelerated)
            {
                return;
            }

            // Arrange
            Vector256<float> input = Vector256.Create(1.0f, 2.0f, 4.0f, 10.0f, 0.5f, 0.25f, 100.0f, 0.1f);

            // Act
            Vector256<float> result = Vector256.Create(
                1.0f / input[0],
                1.0f / input[1],
                1.0f / input[2],
                1.0f / input[3],
                1.0f / input[4],
                1.0f / input[5],
                1.0f / input[6],
                1.0f / input[7]
            );

            // Assert
            Assert.Equal(1.0f, result[0], FloatTolerance);
            Assert.Equal(0.5f, result[1], FloatTolerance);
            Assert.Equal(0.25f, result[2], FloatTolerance);
            Assert.Equal(0.1f, result[3], FloatTolerance);
        }

        [Fact]
        public void ReciprocalOperatorFloat_Vector512_WithHardwareAcceleration_ReturnsCorrectValues()
        {
            if (!Vector512.IsHardwareAccelerated)
            {
                return;
            }

            // Arrange
            Vector512<float> input = Vector512.Create(
                1.0f, 2.0f, 4.0f, 10.0f,
                0.5f, 0.25f, 100.0f, 0.1f,
                8.0f, 16.0f, 0.2f, 5.0f,
                20.0f, 50.0f, 0.05f, 200.0f
            );

            // Act
            Vector512<float> result = Vector512.Create(
                1.0f / input[0],
                1.0f / input[1],
                1.0f / input[2],
                1.0f / input[3],
                1.0f / input[4],
                1.0f / input[5],
                1.0f / input[6],
                1.0f / input[7],
                1.0f / input[8],
                1.0f / input[9],
                1.0f / input[10],
                1.0f / input[11],
                1.0f / input[12],
                1.0f / input[13],
                1.0f / input[14],
                1.0f / input[15]
            );

            // Assert
            Assert.Equal(1.0f, result[0], FloatTolerance);
            Assert.Equal(0.5f, result[1], FloatTolerance);
            Assert.Equal(0.25f, result[2], FloatTolerance);
            Assert.Equal(0.1f, result[3], FloatTolerance);
        }
#endif

        #endregion

        #region Edge Case Tests

        [Fact]
        public void ReciprocalOperatorDouble_WithNegativeValue_ReturnsNegativeReciprocal()
        {
            // Arrange
            double input = -5.0;

            // Act
            double result = 1.0 / input;

            // Assert
            Assert.Equal(-0.2, result, DoubleTolerance);
        }

        [Fact]
        public void ReciprocalOperatorFloat_WithNegativeValue_ReturnsNegativeReciprocal()
        {
            // Arrange
            float input = -5.0f;

            // Act
            float result = 1.0f / input;

            // Assert
            Assert.Equal(-0.2f, result, FloatTolerance);
        }

        [Fact]
        public void ReciprocalOperatorDouble_MultiplicationProperty_IsCorrect()
        {
            // Arrange - x * (1/x) = 1
            double input = 3.7;

            // Act
            double reciprocal = 1.0 / input;
            double product = input * reciprocal;

            // Assert
            Assert.Equal(1.0, product, DoubleTolerance);
        }

        [Fact]
        public void ReciprocalOperatorFloat_MultiplicationProperty_IsCorrect()
        {
            // Arrange - x * (1/x) = 1
            float input = 3.7f;

            // Act
            float reciprocal = 1.0f / input;
            float product = input * reciprocal;

            // Assert
            Assert.Equal(1.0f, product, FloatTolerance);
        }

        #endregion
    }
}
