using System;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
#endif
using AiDotNet.Tensors.Operators;
using Xunit;

namespace AiDotNet.Tensors.Tests.Operators;

/// <summary>
/// Unit tests for CosOperator implementations (scalar and SIMD).
/// </summary>
public class CosOperatorTests
{
    // Scalar operations use Math.Cos/MathF.Cos, so very high accuracy
    private const double ScalarDoubleTolerance = 1e-14;  // Machine epsilon for double
    private const float ScalarFloatTolerance = 1e-6f;    // Machine epsilon for float

    // SIMD operations use polynomial approximations, so lower accuracy but still excellent
    private const double SimdDoubleTolerance = 1e-4;  // 4 decimal places for SIMD double
    private const float SimdFloatTolerance = 1e-3f;    // 3 decimal places for SIMD float

    #region Scalar Double Tests

    [Theory]
    [InlineData(0.0, 1.0)]
    [InlineData(Math.PI / 6, 0.8660254037844387)]  // 30 degrees
    [InlineData(Math.PI / 4, 0.7071067811865476)]  // 45 degrees
    [InlineData(Math.PI / 3, 0.5)]                 // 60 degrees
    [InlineData(Math.PI / 2, 0.0)]                 // 90 degrees
    [InlineData(Math.PI, -1.0)]                    // 180 degrees
    [InlineData(3 * Math.PI / 2, 0.0)]             // 270 degrees
    [InlineData(2 * Math.PI, 1.0)]                 // 360 degrees
    public void CosOperatorDouble_Invoke_Scalar_KnownValues(double input, double expected)
    {
        var op = new CosOperatorDouble();
        double result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    [Theory]
    [InlineData(-Math.PI / 6, 0.8660254037844387)]
    [InlineData(-Math.PI / 4, 0.7071067811865476)]
    [InlineData(-Math.PI / 2, 0.0)]
    [InlineData(-Math.PI, -1.0)]
    public void CosOperatorDouble_Invoke_Scalar_NegativeValues(double input, double expected)
    {
        var op = new CosOperatorDouble();
        double result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    [Theory]
    [InlineData(10 * Math.PI)]      // Large positive
    [InlineData(-10 * Math.PI)]     // Large negative
    [InlineData(100 * Math.PI)]     // Very large positive
    [InlineData(-100 * Math.PI)]    // Very large negative
    public void CosOperatorDouble_Invoke_Scalar_LargeValues(double input)
    {
        var op = new CosOperatorDouble();
        double result = op.Invoke(input);
        double expected = Math.Cos(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    #endregion

    #region Scalar Float Tests

    [Theory]
    [InlineData(0.0f, 1.0f)]
    [InlineData(MathF.PI / 6, 0.86602540f)]        // 30 degrees
    [InlineData(MathF.PI / 4, 0.70710678f)]        // 45 degrees
    [InlineData(MathF.PI / 3, 0.5f)]               // 60 degrees
    [InlineData(MathF.PI / 2, 0.0f)]               // 90 degrees
    [InlineData(MathF.PI, -1.0f)]                  // 180 degrees
    [InlineData(3 * MathF.PI / 2, 0.0f)]           // 270 degrees
    [InlineData(2 * MathF.PI, 1.0f)]               // 360 degrees
    public void CosOperatorFloat_Invoke_Scalar_KnownValues(float input, float expected)
    {
        var op = new CosOperatorFloat();
        float result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarFloatTolerance);
    }

    [Theory]
    [InlineData(-MathF.PI / 6, 0.86602540f)]
    [InlineData(-MathF.PI / 4, 0.70710678f)]
    [InlineData(-MathF.PI / 2, 0.0f)]
    [InlineData(-MathF.PI, -1.0f)]
    public void CosOperatorFloat_Invoke_Scalar_NegativeValues(float input, float expected)
    {
        var op = new CosOperatorFloat();
        float result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarFloatTolerance);
    }

    #endregion

    #region Vector128 Double Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void CosOperatorDouble_Invoke_Vector128_KnownValues()
    {
        var op = new CosOperatorDouble();

        // Test [0, Ï€/2] â†’ [1, 0]
        Vector128<double> input = Vector128.Create(0.0, Math.PI / 2);
        Vector128<double> result = op.Invoke(input);

        Assert.Equal(1.0, result[0], SimdDoubleTolerance);
        Assert.Equal(0.0, result[1], SimdDoubleTolerance);
    }

    [Fact]
    public void CosOperatorDouble_Invoke_Vector128_RangeReduction()
    {
        var op = new CosOperatorDouble();

        // Test values outside [-Ï€, Ï€] to verify range reduction
        // cos(2Ï€) = cos(0) = 1, cos(4Ï€) = cos(0) = 1
        Vector128<double> input = Vector128.Create(2 * Math.PI, 4 * Math.PI);
        Vector128<double> result = op.Invoke(input);

        Assert.Equal(1.0, result[0], SimdDoubleTolerance);
        Assert.Equal(1.0, result[1], SimdDoubleTolerance);
    }

    [Fact]
    public void CosOperatorDouble_Invoke_Vector128_NegativeValues()
    {
        var op = new CosOperatorDouble();

        // Test negative values: cos(-Ï€/3) = 0.5, cos(-Ï€/2) = 0
        Vector128<double> input = Vector128.Create(-Math.PI / 3, -Math.PI / 2);
        Vector128<double> result = op.Invoke(input);

        Assert.Equal(0.5, result[0], SimdDoubleTolerance);
        Assert.Equal(0.0, result[1], SimdDoubleTolerance);
    }
#endif

    #endregion

    #region Vector128 Float Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void CosOperatorFloat_Invoke_Vector128_KnownValues()
    {
        var op = new CosOperatorFloat();

        // Test [0, Ï€/6, Ï€/4, Ï€/2] â†’ [1, 0.866..., 0.707..., 0]
        Vector128<float> input = Vector128.Create(0.0f, MathF.PI / 6, MathF.PI / 4, MathF.PI / 2);
        Vector128<float> result = op.Invoke(input);

        Assert.Equal(1.0f, result[0], SimdFloatTolerance);
        Assert.Equal(0.86602540f, result[1], SimdFloatTolerance);
        Assert.Equal(0.70710678f, result[2], SimdFloatTolerance);
        Assert.Equal(0.0f, result[3], SimdFloatTolerance);
    }

    [Fact]
    public void CosOperatorFloat_Invoke_Vector128_RangeReduction()
    {
        var op = new CosOperatorFloat();

        // Test large values to verify range reduction
        // cos(2nÏ€) = 1, cos((2n+1)Ï€) = -1
        Vector128<float> input = Vector128.Create(10 * MathF.PI, -10 * MathF.PI, 11 * MathF.PI, -11 * MathF.PI);
        Vector128<float> result = op.Invoke(input);

        // Even multiples of Ï€ â†’ 1
        Assert.Equal(1.0f, result[0], SimdFloatTolerance);
        Assert.Equal(1.0f, result[1], SimdFloatTolerance);
        // Odd multiples of Ï€ â†’ -1
        Assert.Equal(-1.0f, result[2], SimdFloatTolerance);
        Assert.Equal(-1.0f, result[3], SimdFloatTolerance);
    }
#endif

    #endregion

    #region Vector256 Double Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void CosOperatorDouble_Invoke_Vector256_KnownValues()
    {
        if (!Vector256.IsHardwareAccelerated)
        {
            return; // Skip test if AVX2 not available
        }

        var op = new CosOperatorDouble();

        // Test 4 values: [0, Ï€/6, Ï€/4, Ï€/2]
        Vector256<double> input = Vector256.Create(0.0, Math.PI / 6, Math.PI / 4, Math.PI / 2);
        Vector256<double> result = op.Invoke(input);

        Assert.Equal(1.0, result[0], SimdDoubleTolerance);
        Assert.Equal(0.8660254037844387, result[1], SimdDoubleTolerance);
        Assert.Equal(0.7071067811865476, result[2], SimdDoubleTolerance);
        Assert.Equal(0.0, result[3], SimdDoubleTolerance);
    }

    [Fact]
    public void CosOperatorDouble_Invoke_Vector256_RangeReduction()
    {
        if (!Vector256.IsHardwareAccelerated)
        {
            return; // Skip test if AVX2 not available
        }

        var op = new CosOperatorDouble();

        // Test extreme values
        Vector256<double> input = Vector256.Create(
            1000 * Math.PI,
            -1000 * Math.PI,
            1001 * Math.PI,
            -1001 * Math.PI);

        Vector256<double> result = op.Invoke(input);

        // Even multiples of Ï€ â†’ 1
        Assert.Equal(1.0, result[0], SimdDoubleTolerance);
        Assert.Equal(1.0, result[1], SimdDoubleTolerance);
        // Odd multiples of Ï€ â†’ -1
        Assert.Equal(-1.0, result[2], SimdDoubleTolerance);
        Assert.Equal(-1.0, result[3], SimdDoubleTolerance);
    }
#endif

    #endregion

    #region Vector256 Float Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void CosOperatorFloat_Invoke_Vector256_KnownValues()
    {
        if (!Vector256.IsHardwareAccelerated)
        {
            return; // Skip test if AVX2 not available
        }

        var op = new CosOperatorFloat();

        // Test 8 values
        Vector256<float> input = Vector256.Create(
            0.0f, MathF.PI / 6, MathF.PI / 4, MathF.PI / 3,
            MathF.PI / 2, MathF.PI, 3 * MathF.PI / 2, 2 * MathF.PI);

        Vector256<float> result = op.Invoke(input);

        Assert.Equal(1.0f, result[0], SimdFloatTolerance);
        Assert.Equal(0.86602540f, result[1], SimdFloatTolerance);
        Assert.Equal(0.70710678f, result[2], SimdFloatTolerance);
        Assert.Equal(0.5f, result[3], SimdFloatTolerance);
        Assert.Equal(0.0f, result[4], SimdFloatTolerance);
        Assert.Equal(-1.0f, result[5], SimdFloatTolerance);
        Assert.Equal(0.0f, result[6], SimdFloatTolerance);
        Assert.Equal(1.0f, result[7], SimdFloatTolerance);
    }
#endif

    #endregion

    #region Vector512 Double Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void CosOperatorDouble_Invoke_Vector512_KnownValues()
    {
        if (!Vector512.IsHardwareAccelerated)
        {
            return; // Skip test if AVX-512 not available
        }

        var op = new CosOperatorDouble();

        // Test 8 values
        Vector512<double> input = Vector512.Create(
            0.0, Math.PI / 6, Math.PI / 4, Math.PI / 3,
            Math.PI / 2, Math.PI, 3 * Math.PI / 2, 2 * Math.PI);

        Vector512<double> result = op.Invoke(input);

        Assert.Equal(1.0, result[0], SimdDoubleTolerance);
        Assert.Equal(0.8660254037844387, result[1], SimdDoubleTolerance);
        Assert.Equal(0.7071067811865476, result[2], SimdDoubleTolerance);
        Assert.Equal(0.5, result[3], SimdDoubleTolerance);
        Assert.Equal(0.0, result[4], SimdDoubleTolerance);
        Assert.Equal(-1.0, result[5], SimdDoubleTolerance);
        Assert.Equal(0.0, result[6], SimdDoubleTolerance);
        Assert.Equal(1.0, result[7], SimdDoubleTolerance);
    }
#endif

    #endregion

    #region Vector512 Float Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void CosOperatorFloat_Invoke_Vector512_KnownValues()
    {
        if (!Vector512.IsHardwareAccelerated)
        {
            return; // Skip test if AVX-512 not available
        }

        var op = new CosOperatorFloat();

        // Test 16 values (full Vector512<float>)
        Vector512<float> input = Vector512.Create(
            0.0f, MathF.PI / 6, MathF.PI / 4, MathF.PI / 3,
            MathF.PI / 2, MathF.PI, 3 * MathF.PI / 2, 2 * MathF.PI,
            -MathF.PI / 6, -MathF.PI / 4, -MathF.PI / 3, -MathF.PI / 2,
            -MathF.PI, -3 * MathF.PI / 2, -2 * MathF.PI, 0.0f);

        Vector512<float> result = op.Invoke(input);

        // Positive values
        Assert.Equal(1.0f, result[0], SimdFloatTolerance);
        Assert.Equal(0.86602540f, result[1], SimdFloatTolerance);
        Assert.Equal(0.70710678f, result[2], SimdFloatTolerance);
        Assert.Equal(0.5f, result[3], SimdFloatTolerance);
        Assert.Equal(0.0f, result[4], SimdFloatTolerance);
        Assert.Equal(-1.0f, result[5], SimdFloatTolerance);
        Assert.Equal(0.0f, result[6], SimdFloatTolerance);
        Assert.Equal(1.0f, result[7], SimdFloatTolerance);

        // Negative values (cos is even function, so cos(-x) = cos(x))
        Assert.Equal(0.86602540f, result[8], SimdFloatTolerance);
        Assert.Equal(0.70710678f, result[9], SimdFloatTolerance);
        Assert.Equal(0.5f, result[10], SimdFloatTolerance);
        Assert.Equal(0.0f, result[11], SimdFloatTolerance);
        Assert.Equal(-1.0f, result[12], SimdFloatTolerance);
        Assert.Equal(0.0f, result[13], SimdFloatTolerance);
        Assert.Equal(1.0f, result[14], SimdFloatTolerance);
        Assert.Equal(1.0f, result[15], SimdFloatTolerance);
    }

    [Fact]
    public void CosOperatorFloat_Invoke_Vector512_RangeReduction()
    {
        if (!Vector512.IsHardwareAccelerated)
        {
            return; // Skip test if AVX-512 not available
        }

        var op = new CosOperatorFloat();

        // Test extreme range reduction with 16 large values
        Vector512<float> input = Vector512.Create(
            100 * MathF.PI, -100 * MathF.PI, 1000 * MathF.PI, -1000 * MathF.PI,
            101 * MathF.PI, -101 * MathF.PI, 1001 * MathF.PI, -1001 * MathF.PI,
            100 * MathF.PI + MathF.PI / 3, -100 * MathF.PI - MathF.PI / 3,
            1000 * MathF.PI + MathF.PI / 4, -1000 * MathF.PI - MathF.PI / 4,
            50 * MathF.PI, -50 * MathF.PI, 51 * MathF.PI, -51 * MathF.PI);

        Vector512<float> result = op.Invoke(input);

        // Even multiples of Ï€ should be 1
        Assert.Equal(1.0f, result[0], SimdFloatTolerance);
        Assert.Equal(1.0f, result[1], SimdFloatTolerance);
        Assert.Equal(1.0f, result[2], SimdFloatTolerance);
        Assert.Equal(1.0f, result[3], SimdFloatTolerance);

        // Odd multiples of Ï€ should be -1
        Assert.Equal(-1.0f, result[4], SimdFloatTolerance);
        Assert.Equal(-1.0f, result[5], SimdFloatTolerance);
        Assert.Equal(-1.0f, result[6], SimdFloatTolerance);
        Assert.Equal(-1.0f, result[7], SimdFloatTolerance);

        // n*Ï€ + Ï€/3 should be Â±0.5
        Assert.Equal(0.5f, result[8], SimdFloatTolerance);
        Assert.Equal(0.5f, result[9], SimdFloatTolerance);

        // n*Ï€ + Ï€/4 should be Â±0.707...
        Assert.Equal(0.70710678f, result[10], SimdFloatTolerance);
        Assert.Equal(0.70710678f, result[11], SimdFloatTolerance);
    }
#endif

    #endregion

    #region Accuracy Comparison Tests

    [Theory]
    [InlineData(0.1)]
    [InlineData(0.5)]
    [InlineData(1.0)]
    [InlineData(1.5)]
    [InlineData(2.0)]
    [InlineData(3.0)]
    public void CosOperatorDouble_Invoke_Scalar_AccuracyVsMathCos(double input)
    {
        var op = new CosOperatorDouble();
        double result = op.Invoke(input);
        double expected = Math.Cos(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    [Theory]
    [InlineData(0.1f)]
    [InlineData(0.5f)]
    [InlineData(1.0f)]
    [InlineData(1.5f)]
    [InlineData(2.0f)]
    [InlineData(3.0f)]
    public void CosOperatorFloat_Invoke_Scalar_AccuracyVsMathFCos(float input)
    {
        var op = new CosOperatorFloat();
        float result = op.Invoke(input);
        float expected = MathF.Cos(input);
        Assert.Equal(expected, result, ScalarFloatTolerance);
    }

    #endregion
}
