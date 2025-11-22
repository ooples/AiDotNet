using System;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
#endif
using AiDotNet.Tensors.Operators;
using Xunit;

namespace AiDotNet.Tensors.Tests.Operators;

/// <summary>
/// Unit tests for Exp10Operator implementations (scalar and SIMD).
/// Tests 10^x functionality.
/// </summary>
public class Exp10OperatorTests
{
    // Scalar operations use Math.Pow, so very high accuracy
    private const double ScalarDoubleTolerance = 1e-14;  // Machine epsilon for double
    private const float ScalarFloatTolerance = 1e-5f;    // Machine epsilon for float

    // SIMD currently uses scalar fallback, so same accuracy as scalar
    // NOTE: When polynomial approximations are added, these will need to be relaxed
    private const double SimdDoubleTolerance = 1e-14;
    private const float SimdFloatTolerance = 1e-5f;

    #region Scalar Double Tests

    [Theory]
    [InlineData(0.0, 1.0)]                 // 10^0 = 1
    [InlineData(1.0, 10.0)]                // 10^1 = 10
    [InlineData(2.0, 100.0)]               // 10^2 = 100
    [InlineData(3.0, 1000.0)]              // 10^3 = 1000
    [InlineData(-1.0, 0.1)]                // 10^-1 = 0.1
    [InlineData(0.5, 3.1622776601683795)] // 10^0.5 = sqrt(10)
    public void Exp10OperatorDouble_Invoke_Scalar_KnownValues(double input, double expected)
    {
        var op = new Exp10OperatorDouble();
        double result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    [Theory]
    [InlineData(-0.5, 0.31622776601683794)]  // 10^-0.5 = 1/sqrt(10)
    [InlineData(-2.0, 0.01)]                 // 10^-2 = 0.01
    [InlineData(-3.0, 0.001)]                // 10^-3 = 0.001
    public void Exp10OperatorDouble_Invoke_Scalar_NegativeValues(double input, double expected)
    {
        var op = new Exp10OperatorDouble();
        double result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    [Theory]
    [InlineData(10.0)]
    [InlineData(-10.0)]
    [InlineData(5.0)]
    [InlineData(-5.0)]
    public void Exp10OperatorDouble_Invoke_Scalar_LargeValues(double input)
    {
        var op = new Exp10OperatorDouble();
        double result = op.Invoke(input);
        double expected = Math.Pow(10.0, input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    #endregion

    #region Scalar Float Tests

    [Theory]
    [InlineData(0.0f, 1.0f)]
    [InlineData(1.0f, 10.0f)]
    [InlineData(2.0f, 100.0f)]
    [InlineData(3.0f, 1000.0f)]
    [InlineData(-1.0f, 0.1f)]
    [InlineData(0.5f, 3.1622777f)]
    public void Exp10OperatorFloat_Invoke_Scalar_KnownValues(float input, float expected)
    {
        var op = new Exp10OperatorFloat();
        float result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarFloatTolerance);
    }

    [Theory]
    [InlineData(-0.5f, 0.3162278f)]
    [InlineData(-2.0f, 0.01f)]
    [InlineData(-3.0f, 0.001f)]
    public void Exp10OperatorFloat_Invoke_Scalar_NegativeValues(float input, float expected)
    {
        var op = new Exp10OperatorFloat();
        float result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarFloatTolerance);
    }

    #endregion

    #region Vector128 Double Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void Exp10OperatorDouble_Invoke_Vector128_KnownValues()
    {
        var op = new Exp10OperatorDouble();

        // Test [0, 1] -> [1, 10]
        Vector128<double> input = Vector128.Create(0.0, 1.0);
        Vector128<double> result = op.Invoke(input);

        Assert.Equal(1.0, result[0], SimdDoubleTolerance);
        Assert.Equal(10.0, result[1], SimdDoubleTolerance);
    }

    [Fact]
    public void Exp10OperatorDouble_Invoke_Vector128_NegativeValues()
    {
        var op = new Exp10OperatorDouble();

        // Test negative values
        Vector128<double> input = Vector128.Create(-1.0, -2.0);
        Vector128<double> result = op.Invoke(input);

        Assert.Equal(0.1, result[0], SimdDoubleTolerance);
        Assert.Equal(0.01, result[1], SimdDoubleTolerance);
    }
#endif

    #endregion

    #region Vector128 Float Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void Exp10OperatorFloat_Invoke_Vector128_KnownValues()
    {
        var op = new Exp10OperatorFloat();

        // Test [0, 1, -1, 0.5]
        Vector128<float> input = Vector128.Create(0.0f, 1.0f, -1.0f, 0.5f);
        Vector128<float> result = op.Invoke(input);

        Assert.Equal(1.0f, result[0], SimdFloatTolerance);
        Assert.Equal(10.0f, result[1], SimdFloatTolerance);
        Assert.Equal(0.1f, result[2], SimdFloatTolerance);
        Assert.Equal(3.1622777f, result[3], SimdFloatTolerance);
    }
#endif

    #endregion

    #region Vector256 Double Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void Exp10OperatorDouble_Invoke_Vector256_KnownValues()
    {
        if (!Vector256.IsHardwareAccelerated)
        {
            return; // Skip test if AVX2 not available
        }

        var op = new Exp10OperatorDouble();

        // Test 4 values
        Vector256<double> input = Vector256.Create(0.0, 1.0, -1.0, 2.0);
        Vector256<double> result = op.Invoke(input);

        Assert.Equal(1.0, result[0], SimdDoubleTolerance);
        Assert.Equal(10.0, result[1], SimdDoubleTolerance);
        Assert.Equal(0.1, result[2], SimdDoubleTolerance);
        Assert.Equal(100.0, result[3], SimdDoubleTolerance);
    }
#endif

    #endregion

    #region Vector256 Float Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void Exp10OperatorFloat_Invoke_Vector256_KnownValues()
    {
        if (!Vector256.IsHardwareAccelerated)
        {
            return; // Skip test if AVX2 not available
        }

        var op = new Exp10OperatorFloat();

        // Test 8 values
        Vector256<float> input = Vector256.Create(
            0.0f, 1.0f, -1.0f, 0.5f,
            -0.5f, 2.0f, -2.0f, 3.0f);

        Vector256<float> result = op.Invoke(input);

        Assert.Equal(1.0f, result[0], SimdFloatTolerance);
        Assert.Equal(10.0f, result[1], SimdFloatTolerance);
        Assert.Equal(0.1f, result[2], SimdFloatTolerance);
        Assert.Equal(3.1622777f, result[3], SimdFloatTolerance);
        Assert.Equal(0.3162278f, result[4], SimdFloatTolerance);
        Assert.Equal(100.0f, result[5], SimdFloatTolerance);
        Assert.Equal(0.01f, result[6], SimdFloatTolerance);
        Assert.Equal(1000.0f, result[7], SimdFloatTolerance);
    }
#endif

    #endregion

    #region Vector512 Double Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void Exp10OperatorDouble_Invoke_Vector512_KnownValues()
    {
        if (!Vector512.IsHardwareAccelerated)
        {
            return; // Skip test if AVX-512 not available
        }

        var op = new Exp10OperatorDouble();

        // Test 8 values
        Vector512<double> input = Vector512.Create(
            0.0, 1.0, -1.0, 2.0,
            -2.0, 0.5, -0.5, 3.0);

        Vector512<double> result = op.Invoke(input);

        Assert.Equal(1.0, result[0], SimdDoubleTolerance);
        Assert.Equal(10.0, result[1], SimdDoubleTolerance);
        Assert.Equal(0.1, result[2], SimdDoubleTolerance);
        Assert.Equal(100.0, result[3], SimdDoubleTolerance);
        Assert.Equal(0.01, result[4], SimdDoubleTolerance);
        Assert.Equal(3.1622776601683795, result[5], SimdDoubleTolerance);
        Assert.Equal(0.31622776601683794, result[6], SimdDoubleTolerance);
        Assert.Equal(1000.0, result[7], SimdDoubleTolerance);
    }
#endif

    #endregion

    #region Vector512 Float Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void Exp10OperatorFloat_Invoke_Vector512_KnownValues()
    {
        if (!Vector512.IsHardwareAccelerated)
        {
            return; // Skip test if AVX-512 not available
        }

        var op = new Exp10OperatorFloat();

        // Test 16 values
        Vector512<float> input = Vector512.Create(
            0.0f, 1.0f, -1.0f, 0.5f,
            -0.5f, 2.0f, -2.0f, 3.0f,
            -3.0f, 0.1f, -0.1f, 1.5f,
            -1.5f, 4.0f, -4.0f, 0.0f);

        Vector512<float> result = op.Invoke(input);

        Assert.Equal(1.0f, result[0], SimdFloatTolerance);
        Assert.Equal(10.0f, result[1], SimdFloatTolerance);
        Assert.Equal(0.1f, result[2], SimdFloatTolerance);
        Assert.Equal(3.1622777f, result[3], SimdFloatTolerance);
        Assert.Equal(0.3162278f, result[4], SimdFloatTolerance);
        Assert.Equal(100.0f, result[5], SimdFloatTolerance);
        Assert.Equal(0.01f, result[6], SimdFloatTolerance);
        Assert.Equal(1000.0f, result[7], SimdFloatTolerance);
        Assert.Equal(0.001f, result[8], SimdFloatTolerance);
        Assert.Equal(1.2589255f, result[9], SimdFloatTolerance);
        Assert.Equal(0.79432825f, result[10], SimdFloatTolerance);
        Assert.Equal(31.622776f, result[11], SimdFloatTolerance);
        Assert.Equal(0.031622776f, result[12], SimdFloatTolerance);
        Assert.Equal(10000.0f, result[13], SimdFloatTolerance);
        Assert.Equal(0.0001f, result[14], SimdFloatTolerance);
        Assert.Equal(1.0f, result[15], SimdFloatTolerance);
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
    public void Exp10OperatorDouble_Invoke_Scalar_AccuracyVsMathPow(double input)
    {
        var op = new Exp10OperatorDouble();
        double result = op.Invoke(input);
        double expected = Math.Pow(10.0, input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    [Theory]
    [InlineData(0.1f)]
    [InlineData(0.5f)]
    [InlineData(1.0f)]
    [InlineData(1.5f)]
    [InlineData(2.0f)]
    [InlineData(3.0f)]
    public void Exp10OperatorFloat_Invoke_Scalar_AccuracyVsMathFPow(float input)
    {
        var op = new Exp10OperatorFloat();
        float result = op.Invoke(input);
        float expected = MathF.Pow(10.0f, input);
        Assert.Equal(expected, result, ScalarFloatTolerance);
    }

    #endregion
}
