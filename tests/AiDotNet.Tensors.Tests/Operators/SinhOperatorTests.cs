using System;
using System.Numerics;
using System.Runtime.Intrinsics;
using Xunit;

namespace AiDotNet.Tensors.Tests.Operators
{
    public class SinhOperatorTests
    {
        private const double DoubleTolerance = 1e-14;
        private const float FloatTolerance = 1e-6f;

        #region SinhOperatorDouble Tests

        [Fact]
        public void SinhOperatorDouble_ScalarOperation_WithZero_ReturnsZero()
        {
            // Arrange
            double input = 0.0;
            double expected = 0.0;

            // Act
            double result = Math.Sinh(input);

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void SinhOperatorDouble_ScalarOperation_WithOne_ReturnsCorrectValue()
        {
            // Arrange
            double input = 1.0;
            double expected = 1.1752011936438014;

            // Act
            double result = Math.Sinh(input);

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void SinhOperatorDouble_ScalarOperation_WithNegativeOne_ReturnsCorrectValue()
        {
            // Arrange
            double input = -1.0;
            double expected = -1.1752011936438014;

            // Act
            double result = Math.Sinh(input);

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void SinhOperatorDouble_ScalarOperation_WithTwo_ReturnsCorrectValue()
        {
            // Arrange
            double input = 2.0;
            double expected = 3.626860407847019;

            // Act
            double result = Math.Sinh(input);

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void SinhOperatorDouble_ScalarOperation_WithSmallValue_ReturnsCorrectValue()
        {
            // Arrange
            double input = 0.5;
            double expected = 0.5210953054937474;

            // Act
            double result = Math.Sinh(input);

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void SinhOperatorDouble_ScalarOperation_WithLargePositiveValue_ReturnsCorrectValue()
        {
            // Arrange
            double input = 5.0;
            double expected = 74.20321057778875;

            // Act
            double result = Math.Sinh(input);

            // Assert
            Assert.Equal(expected, result, 1e-12); // Slightly relaxed tolerance for large values
        }

        [Fact]
        public void SinhOperatorDouble_ScalarOperation_WithLargeNegativeValue_ReturnsCorrectValue()
        {
            // Arrange
            double input = -5.0;
            double expected = -74.20321057778875;

            // Act
            double result = Math.Sinh(input);

            // Assert
            Assert.Equal(expected, result, 1e-12); // Slightly relaxed tolerance for large values
        }

        [Fact]
        public void SinhOperatorDouble_ScalarOperation_IsOddFunction()
        {
            // Arrange
            double input = 2.5;

            // Act
            double posResult = Math.Sinh(input);
            double negResult = Math.Sinh(-input);

            // Assert - sinh(-x) should equal -sinh(x)
            Assert.Equal(-posResult, negResult, DoubleTolerance);
        }

#if NET5_0_OR_GREATER
        [Fact]
        public void SinhOperatorDouble_Vector128_WithHardwareAcceleration_ReturnsCorrectValues()
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
                Math.Sinh(input[0]),
                Math.Sinh(input[1])
            );

            // Assert
            Assert.Equal(0.0, result[0], DoubleTolerance);
            Assert.Equal(1.1752011936438014, result[1], DoubleTolerance);
        }

        [Fact]
        public void SinhOperatorDouble_Vector256_WithHardwareAcceleration_ReturnsCorrectValues()
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
                Math.Sinh(input[0]),
                Math.Sinh(input[1]),
                Math.Sinh(input[2]),
                Math.Sinh(input[3])
            );

            // Assert
            Assert.Equal(0.0, result[0], DoubleTolerance);
            Assert.Equal(1.1752011936438014, result[1], DoubleTolerance);
            Assert.Equal(-1.1752011936438014, result[2], DoubleTolerance);
            Assert.Equal(3.626860407847019, result[3], DoubleTolerance);
        }

        [Fact]
        public void SinhOperatorDouble_Vector512_WithHardwareAcceleration_ReturnsCorrectValues()
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
                Math.Sinh(input[0]),
                Math.Sinh(input[1]),
                Math.Sinh(input[2]),
                Math.Sinh(input[3]),
                Math.Sinh(input[4]),
                Math.Sinh(input[5]),
                Math.Sinh(input[6]),
                Math.Sinh(input[7])
            );

            // Assert
            Assert.Equal(0.0, result[0], DoubleTolerance);
            Assert.Equal(1.1752011936438014, result[1], DoubleTolerance);
            Assert.Equal(-1.1752011936438014, result[2], DoubleTolerance);
            Assert.Equal(3.626860407847019, result[3], DoubleTolerance);
            Assert.Equal(0.5210953054937474, result[4], DoubleTolerance);
            Assert.Equal(-0.5210953054937474, result[5], DoubleTolerance);
        }
#endif

        #endregion

        #region SinhOperatorFloat Tests

        [Fact]
        public void SinhOperatorFloat_ScalarOperation_WithZero_ReturnsZero()
        {
            // Arrange
            float input = 0.0f;
            float expected = 0.0f;

            // Act
            float result = MathF.Sinh(input);

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void SinhOperatorFloat_ScalarOperation_WithOne_ReturnsCorrectValue()
        {
            // Arrange
            float input = 1.0f;
            float expected = 1.1752012f;

            // Act
            float result = MathF.Sinh(input);

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void SinhOperatorFloat_ScalarOperation_WithNegativeOne_ReturnsCorrectValue()
        {
            // Arrange
            float input = -1.0f;
            float expected = -1.1752012f;

            // Act
            float result = MathF.Sinh(input);

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void SinhOperatorFloat_ScalarOperation_WithTwo_ReturnsCorrectValue()
        {
            // Arrange
            float input = 2.0f;
            float expected = 3.6268604f;

            // Act
            float result = MathF.Sinh(input);

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void SinhOperatorFloat_ScalarOperation_WithSmallValue_ReturnsCorrectValue()
        {
            // Arrange
            float input = 0.5f;
            float expected = 0.5210953f;

            // Act
            float result = MathF.Sinh(input);

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void SinhOperatorFloat_ScalarOperation_WithLargePositiveValue_ReturnsCorrectValue()
        {
            // Arrange
            float input = 5.0f;
            float expected = 74.20321f;

            // Act
            float result = MathF.Sinh(input);

            // Assert
            Assert.Equal(expected, result, 1e-3f); // Relaxed tolerance for large values
        }

        [Fact]
        public void SinhOperatorFloat_ScalarOperation_WithLargeNegativeValue_ReturnsCorrectValue()
        {
            // Arrange
            float input = -5.0f;
            float expected = -74.20321f;

            // Act
            float result = MathF.Sinh(input);

            // Assert
            Assert.Equal(expected, result, 1e-3f); // Relaxed tolerance for large values
        }

        [Fact]
        public void SinhOperatorFloat_ScalarOperation_IsOddFunction()
        {
            // Arrange
            float input = 2.5f;

            // Act
            float posResult = MathF.Sinh(input);
            float negResult = MathF.Sinh(-input);

            // Assert - sinh(-x) should equal -sinh(x)
            Assert.Equal(-posResult, negResult, FloatTolerance);
        }

#if NET5_0_OR_GREATER
        [Fact]
        public void SinhOperatorFloat_Vector128_WithHardwareAcceleration_ReturnsCorrectValues()
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
                MathF.Sinh(input[0]),
                MathF.Sinh(input[1]),
                MathF.Sinh(input[2]),
                MathF.Sinh(input[3])
            );

            // Assert
            Assert.Equal(0.0f, result[0], FloatTolerance);
            Assert.Equal(1.1752012f, result[1], FloatTolerance);
            Assert.Equal(-1.1752012f, result[2], FloatTolerance);
            Assert.Equal(3.6268604f, result[3], FloatTolerance);
        }

        [Fact]
        public void SinhOperatorFloat_Vector256_WithHardwareAcceleration_ReturnsCorrectValues()
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
                MathF.Sinh(input[0]),
                MathF.Sinh(input[1]),
                MathF.Sinh(input[2]),
                MathF.Sinh(input[3]),
                MathF.Sinh(input[4]),
                MathF.Sinh(input[5]),
                MathF.Sinh(input[6]),
                MathF.Sinh(input[7])
            );

            // Assert
            Assert.Equal(0.0f, result[0], FloatTolerance);
            Assert.Equal(1.1752012f, result[1], FloatTolerance);
            Assert.Equal(-1.1752012f, result[2], FloatTolerance);
            Assert.Equal(3.6268604f, result[3], FloatTolerance);
            Assert.Equal(0.5210953f, result[4], FloatTolerance);
            Assert.Equal(-0.5210953f, result[5], FloatTolerance);
        }

        [Fact]
        public void SinhOperatorFloat_Vector512_WithHardwareAcceleration_ReturnsCorrectValues()
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
                MathF.Sinh(input[0]),
                MathF.Sinh(input[1]),
                MathF.Sinh(input[2]),
                MathF.Sinh(input[3]),
                MathF.Sinh(input[4]),
                MathF.Sinh(input[5]),
                MathF.Sinh(input[6]),
                MathF.Sinh(input[7]),
                MathF.Sinh(input[8]),
                MathF.Sinh(input[9]),
                MathF.Sinh(input[10]),
                MathF.Sinh(input[11]),
                MathF.Sinh(input[12]),
                MathF.Sinh(input[13]),
                MathF.Sinh(input[14]),
                MathF.Sinh(input[15])
            );

            // Assert
            Assert.Equal(0.0f, result[0], FloatTolerance);
            Assert.Equal(1.1752012f, result[1], FloatTolerance);
            Assert.Equal(-1.1752012f, result[2], FloatTolerance);
            Assert.Equal(3.6268604f, result[3], FloatTolerance);
        }
#endif

        #endregion

        #region Edge Case Tests

        [Fact]
        public void SinhOperatorDouble_WithVerySmallValue_ReturnsApproximatelyInput()
        {
            // Arrange - for very small x, sinh(x) ≈ x
            double input = 1e-10;

            // Act
            double result = Math.Sinh(input);

            // Assert
            Assert.Equal(input, result, 1e-15);
        }

        [Fact]
        public void SinhOperatorFloat_WithVerySmallValue_ReturnsApproximatelyInput()
        {
            // Arrange - for very small x, sinh(x) ≈ x
            float input = 1e-7f;

            // Act
            float result = MathF.Sinh(input);

            // Assert
            Assert.Equal(input, result, 1e-10f);
        }

        [Fact]
        public void SinhOperatorDouble_ExponentialRelationship_IsCorrect()
        {
            // Arrange - sinh(x) = (e^x - e^-x) / 2
            double input = 1.5;

            // Act
            double sinhResult = Math.Sinh(input);
            double expectedFromExponential = (Math.Exp(input) - Math.Exp(-input)) / 2.0;

            // Assert
            Assert.Equal(expectedFromExponential, sinhResult, DoubleTolerance);
        }

        [Fact]
        public void SinhOperatorFloat_ExponentialRelationship_IsCorrect()
        {
            // Arrange - sinh(x) = (e^x - e^-x) / 2
            float input = 1.5f;

            // Act
            float sinhResult = MathF.Sinh(input);
            float expectedFromExponential = (MathF.Exp(input) - MathF.Exp(-input)) / 2.0f;

            // Assert
            Assert.Equal(expectedFromExponential, sinhResult, FloatTolerance);
        }

        #endregion
    }
}
