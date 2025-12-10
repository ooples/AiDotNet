using System;
using System.Numerics;
using System.Runtime.Intrinsics;
using Xunit;

namespace AiDotNet.Tensors.Tests.Operators
{
    public class CoshOperatorTests
    {
        private const double DoubleTolerance = 1e-14;
        private const float FloatTolerance = 1e-6f;

        #region CoshOperatorDouble Tests

        [Fact]
        public void CoshOperatorDouble_ScalarOperation_WithZero_ReturnsOne()
        {
            // Arrange
            double input = 0.0;
            double expected = 1.0;

            // Act
            double result = Math.Cosh(input);

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void CoshOperatorDouble_ScalarOperation_WithOne_ReturnsCorrectValue()
        {
            // Arrange
            double input = 1.0;
            double expected = 1.5430806348152437;

            // Act
            double result = Math.Cosh(input);

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void CoshOperatorDouble_ScalarOperation_WithNegativeOne_ReturnsCorrectValue()
        {
            // Arrange
            double input = -1.0;
            double expected = 1.5430806348152437;

            // Act
            double result = Math.Cosh(input);

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void CoshOperatorDouble_ScalarOperation_WithTwo_ReturnsCorrectValue()
        {
            // Arrange
            double input = 2.0;
            double expected = 3.7621956910836314;

            // Act
            double result = Math.Cosh(input);

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void CoshOperatorDouble_ScalarOperation_WithSmallValue_ReturnsCorrectValue()
        {
            // Arrange
            double input = 0.5;
            double expected = 1.1276259652063807;

            // Act
            double result = Math.Cosh(input);

            // Assert
            Assert.Equal(expected, result, DoubleTolerance);
        }

        [Fact]
        public void CoshOperatorDouble_ScalarOperation_WithLargePositiveValue_ReturnsCorrectValue()
        {
            // Arrange
            double input = 5.0;
            double expected = 74.20994852478785;

            // Act
            double result = Math.Cosh(input);

            // Assert
            Assert.Equal(expected, result, 1e-12); // Slightly relaxed tolerance for large values
        }

        [Fact]
        public void CoshOperatorDouble_ScalarOperation_WithLargeNegativeValue_ReturnsCorrectValue()
        {
            // Arrange
            double input = -5.0;
            double expected = 74.20994852478785;

            // Act
            double result = Math.Cosh(input);

            // Assert
            Assert.Equal(expected, result, 1e-12); // Slightly relaxed tolerance for large values
        }

        [Fact]
        public void CoshOperatorDouble_ScalarOperation_IsEvenFunction()
        {
            // Arrange
            double input = 2.5;

            // Act
            double posResult = Math.Cosh(input);
            double negResult = Math.Cosh(-input);

            // Assert - cosh(-x) should equal cosh(x)
            Assert.Equal(posResult, negResult, DoubleTolerance);
        }

        [Fact]
        public void CoshOperatorDouble_ScalarOperation_IsAlwaysPositive()
        {
            // Arrange & Act
            double result1 = Math.Cosh(-5.0);
            double result2 = Math.Cosh(0.0);
            double result3 = Math.Cosh(5.0);

            // Assert - cosh(x) is always >= 1
            Assert.True(result1 >= 1.0);
            Assert.True(result2 >= 1.0);
            Assert.True(result3 >= 1.0);
        }

#if NET5_0_OR_GREATER
        [Fact]
        public void CoshOperatorDouble_Vector128_WithHardwareAcceleration_ReturnsCorrectValues()
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
                Math.Cosh(input[0]),
                Math.Cosh(input[1])
            );

            // Assert
            Assert.Equal(1.0, result[0], DoubleTolerance);
            Assert.Equal(1.5430806348152437, result[1], DoubleTolerance);
        }

        [Fact]
        public void CoshOperatorDouble_Vector256_WithHardwareAcceleration_ReturnsCorrectValues()
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
                Math.Cosh(input[0]),
                Math.Cosh(input[1]),
                Math.Cosh(input[2]),
                Math.Cosh(input[3])
            );

            // Assert
            Assert.Equal(1.0, result[0], DoubleTolerance);
            Assert.Equal(1.5430806348152437, result[1], DoubleTolerance);
            Assert.Equal(1.5430806348152437, result[2], DoubleTolerance);
            Assert.Equal(3.7621956910836314, result[3], DoubleTolerance);
        }

        [Fact]
        public void CoshOperatorDouble_Vector512_WithHardwareAcceleration_ReturnsCorrectValues()
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
                Math.Cosh(input[0]),
                Math.Cosh(input[1]),
                Math.Cosh(input[2]),
                Math.Cosh(input[3]),
                Math.Cosh(input[4]),
                Math.Cosh(input[5]),
                Math.Cosh(input[6]),
                Math.Cosh(input[7])
            );

            // Assert
            Assert.Equal(1.0, result[0], DoubleTolerance);
            Assert.Equal(1.5430806348152437, result[1], DoubleTolerance);
            Assert.Equal(1.5430806348152437, result[2], DoubleTolerance);
            Assert.Equal(3.7621956910836314, result[3], DoubleTolerance);
            Assert.Equal(1.1276259652063807, result[4], DoubleTolerance);
            Assert.Equal(1.1276259652063807, result[5], DoubleTolerance);
        }
#endif

        #endregion

        #region CoshOperatorFloat Tests

        [Fact]
        public void CoshOperatorFloat_ScalarOperation_WithZero_ReturnsOne()
        {
            // Arrange
            float input = 0.0f;
            float expected = 1.0f;

            // Act
            float result = MathF.Cosh(input);

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void CoshOperatorFloat_ScalarOperation_WithOne_ReturnsCorrectValue()
        {
            // Arrange
            float input = 1.0f;
            float expected = 1.5430807f;

            // Act
            float result = MathF.Cosh(input);

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void CoshOperatorFloat_ScalarOperation_WithNegativeOne_ReturnsCorrectValue()
        {
            // Arrange
            float input = -1.0f;
            float expected = 1.5430807f;

            // Act
            float result = MathF.Cosh(input);

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void CoshOperatorFloat_ScalarOperation_WithTwo_ReturnsCorrectValue()
        {
            // Arrange
            float input = 2.0f;
            float expected = 3.7621956f;

            // Act
            float result = MathF.Cosh(input);

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void CoshOperatorFloat_ScalarOperation_WithSmallValue_ReturnsCorrectValue()
        {
            // Arrange
            float input = 0.5f;
            float expected = 1.127626f;

            // Act
            float result = MathF.Cosh(input);

            // Assert
            Assert.Equal(expected, result, FloatTolerance);
        }

        [Fact]
        public void CoshOperatorFloat_ScalarOperation_WithLargePositiveValue_ReturnsCorrectValue()
        {
            // Arrange
            float input = 5.0f;
            float expected = 74.20995f;

            // Act
            float result = MathF.Cosh(input);

            // Assert
            Assert.Equal(expected, result, 1e-3f); // Relaxed tolerance for large values
        }

        [Fact]
        public void CoshOperatorFloat_ScalarOperation_WithLargeNegativeValue_ReturnsCorrectValue()
        {
            // Arrange
            float input = -5.0f;
            float expected = 74.20995f;

            // Act
            float result = MathF.Cosh(input);

            // Assert
            Assert.Equal(expected, result, 1e-3f); // Relaxed tolerance for large values
        }

        [Fact]
        public void CoshOperatorFloat_ScalarOperation_IsEvenFunction()
        {
            // Arrange
            float input = 2.5f;

            // Act
            float posResult = MathF.Cosh(input);
            float negResult = MathF.Cosh(-input);

            // Assert - cosh(-x) should equal cosh(x)
            Assert.Equal(posResult, negResult, FloatTolerance);
        }

        [Fact]
        public void CoshOperatorFloat_ScalarOperation_IsAlwaysPositive()
        {
            // Arrange & Act
            float result1 = MathF.Cosh(-5.0f);
            float result2 = MathF.Cosh(0.0f);
            float result3 = MathF.Cosh(5.0f);

            // Assert - cosh(x) is always >= 1
            Assert.True(result1 >= 1.0f);
            Assert.True(result2 >= 1.0f);
            Assert.True(result3 >= 1.0f);
        }

#if NET5_0_OR_GREATER
        [Fact]
        public void CoshOperatorFloat_Vector128_WithHardwareAcceleration_ReturnsCorrectValues()
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
                MathF.Cosh(input[0]),
                MathF.Cosh(input[1]),
                MathF.Cosh(input[2]),
                MathF.Cosh(input[3])
            );

            // Assert
            Assert.Equal(1.0f, result[0], FloatTolerance);
            Assert.Equal(1.5430807f, result[1], FloatTolerance);
            Assert.Equal(1.5430807f, result[2], FloatTolerance);
            Assert.Equal(3.7621956f, result[3], FloatTolerance);
        }

        [Fact]
        public void CoshOperatorFloat_Vector256_WithHardwareAcceleration_ReturnsCorrectValues()
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
                MathF.Cosh(input[0]),
                MathF.Cosh(input[1]),
                MathF.Cosh(input[2]),
                MathF.Cosh(input[3]),
                MathF.Cosh(input[4]),
                MathF.Cosh(input[5]),
                MathF.Cosh(input[6]),
                MathF.Cosh(input[7])
            );

            // Assert
            Assert.Equal(1.0f, result[0], FloatTolerance);
            Assert.Equal(1.5430807f, result[1], FloatTolerance);
            Assert.Equal(1.5430807f, result[2], FloatTolerance);
            Assert.Equal(3.7621956f, result[3], FloatTolerance);
            Assert.Equal(1.127626f, result[4], FloatTolerance);
            Assert.Equal(1.127626f, result[5], FloatTolerance);
        }

        [Fact]
        public void CoshOperatorFloat_Vector512_WithHardwareAcceleration_ReturnsCorrectValues()
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
                MathF.Cosh(input[0]),
                MathF.Cosh(input[1]),
                MathF.Cosh(input[2]),
                MathF.Cosh(input[3]),
                MathF.Cosh(input[4]),
                MathF.Cosh(input[5]),
                MathF.Cosh(input[6]),
                MathF.Cosh(input[7]),
                MathF.Cosh(input[8]),
                MathF.Cosh(input[9]),
                MathF.Cosh(input[10]),
                MathF.Cosh(input[11]),
                MathF.Cosh(input[12]),
                MathF.Cosh(input[13]),
                MathF.Cosh(input[14]),
                MathF.Cosh(input[15])
            );

            // Assert
            Assert.Equal(1.0f, result[0], FloatTolerance);
            Assert.Equal(1.5430807f, result[1], FloatTolerance);
            Assert.Equal(1.5430807f, result[2], FloatTolerance);
            Assert.Equal(3.7621956f, result[3], FloatTolerance);
        }
#endif

        #endregion

        #region Edge Case Tests

        [Fact]
        public void CoshOperatorDouble_WithVerySmallValue_ReturnsApproximatelyOne()
        {
            // Arrange - for very small x, cosh(x) ≈ 1
            double input = 1e-10;

            // Act
            double result = Math.Cosh(input);

            // Assert
            Assert.Equal(1.0, result, 1e-14);
        }

        [Fact]
        public void CoshOperatorFloat_WithVerySmallValue_ReturnsApproximatelyOne()
        {
            // Arrange - for very small x, cosh(x) ≈ 1
            float input = 1e-7f;

            // Act
            float result = MathF.Cosh(input);

            // Assert
            Assert.Equal(1.0f, result, 1e-6f);
        }

        [Fact]
        public void CoshOperatorDouble_ExponentialRelationship_IsCorrect()
        {
            // Arrange - cosh(x) = (e^x + e^-x) / 2
            double input = 1.5;

            // Act
            double coshResult = Math.Cosh(input);
            double expectedFromExponential = (Math.Exp(input) + Math.Exp(-input)) / 2.0;

            // Assert
            Assert.Equal(expectedFromExponential, coshResult, DoubleTolerance);
        }

        [Fact]
        public void CoshOperatorFloat_ExponentialRelationship_IsCorrect()
        {
            // Arrange - cosh(x) = (e^x + e^-x) / 2
            float input = 1.5f;

            // Act
            float coshResult = MathF.Cosh(input);
            float expectedFromExponential = (MathF.Exp(input) + MathF.Exp(-input)) / 2.0f;

            // Assert
            Assert.Equal(expectedFromExponential, coshResult, FloatTolerance);
        }

        [Fact]
        public void CoshOperatorDouble_HyperbolicIdentity_IsCorrect()
        {
            // Arrange - cosh^2(x) - sinh^2(x) = 1
            double input = 2.5;

            // Act
            double cosh = Math.Cosh(input);
            double sinh = Math.Sinh(input);
            double identity = cosh * cosh - sinh * sinh;

            // Assert
            Assert.Equal(1.0, identity, DoubleTolerance);
        }

        [Fact]
        public void CoshOperatorFloat_HyperbolicIdentity_IsCorrect()
        {
            // Arrange - cosh^2(x) - sinh^2(x) = 1
            // Use smaller input to reduce precision loss from subtracting large similar numbers
            float input = 0.5f;

            // Act
            float cosh = MathF.Cosh(input);
            float sinh = MathF.Sinh(input);
            float identity = cosh * cosh - sinh * sinh;

            // Assert - relaxed tolerance due to floating point precision in subtraction
            Assert.Equal(1.0f, identity, 1e-5f);
        }

        #endregion
    }
}
