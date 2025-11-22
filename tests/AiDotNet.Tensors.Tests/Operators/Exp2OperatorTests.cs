using System;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
#endif
using AiDotNet.Tensors.Operators;
using Xunit;

namespace AiDotNet.Tensors.Tests.Operators;

/// <summary>
/// Unit tests for Exp2Operator implementations (scalar and SIMD).
/// Tests 2^x functionality.
/// </summary>
public class Exp2OperatorTests
{
    // Scalar operations use Math.Pow, so very high accuracy
    private const double ScalarDoubleTolerance = 1e-14;  // Machine epsilon for double
    private const float ScalarFloatTolerance = 1e-6f;    // Machine epsilon for float

    // SIMD currently uses scalar fallback, so same accuracy as scalar
    // NOTE: When polynomial approximations are added, these will need to be relaxed
    private const double SimdDoubleTolerance = 1e-14;
    private const float SimdFloatTolerance = 1e-6f;

    #region Scalar Double Tests

    [Theory]
    [InlineData(0.0, 1.0)]                 // 2^0 = 1
    [InlineData(1.0, 2.0)]                 // 2^1 = 2
    [InlineData(2.0, 4.0)]                 // 2^2 = 4
    [InlineData(3.0, 8.0)]                 // 2^3 = 8
    [InlineData(-1.0, 0.5)]                // 2^-1 = 0.5
    [InlineData(0.5, 1.4142135623730951)]  // 2^0.5 = sqrt(2)
    public void Exp2OperatorDouble_Invoke_Scalar_KnownValues(double input, double expected)
    {
        var op = new Exp2OperatorDouble();
        double result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    [Theory]
    [InlineData(-0.5, 0.7071067811865476)]  // 2^-0.5 = 1/sqrt(2)
    [InlineData(-2.0, 0.25)]                // 2^-2 = 0.25
    [InlineData(-3.0, 0.125)]               // 2^-3 = 0.125
    public void Exp2OperatorDouble_Invoke_Scalar_NegativeValues(double input, double expected)
    {
        var op = new Exp2OperatorDouble();
        double result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    [Theory]
    [InlineData(10.0)]
    [InlineData(-10.0)]
    [InlineData(5.0)]
    [InlineData(-5.0)]
    public void Exp2OperatorDouble_Invoke_Scalar_LargeValues(double input)
    {
        var op = new Exp2OperatorDouble();
        double result = op.Invoke(input);
        double expected = Math.Pow(2.0, input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    #endregion

    #region Scalar Float Tests

    [Theory]
    [InlineData(0.0f, 1.0f)]
    [InlineData(1.0f, 2.0f)]
    [InlineData(2.0f, 4.0f)]
    [InlineData(3.0f, 8.0f)]
    [InlineData(-1.0f, 0.5f)]
    [InlineData(0.5f, 1.4142136f)]
    public void Exp2OperatorFloat_Invoke_Scalar_KnownValues(float input, float expected)
    {
        var op = new Exp2OperatorFloat();
        float result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarFloatTolerance);
    }

    [Theory]
    [InlineData(-0.5f, 0.7071068f)]
    [InlineData(-2.0f, 0.25f)]
    [InlineData(-3.0f, 0.125f)]
    public void Exp2OperatorFloat_Invoke_Scalar_NegativeValues(float input, float expected)
    {
        var op = new Exp2OperatorFloat();
        float result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarFloatTolerance);
    }

    #endregion

    #region Vector128 Double Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void Exp2OperatorDouble_Invoke_Vector128_KnownValues()
    {
        var op = new Exp2OperatorDouble();

        // Test [0, 1] -> [1, 2]
        Vector128<double> input = Vector128.Create(0.0, 1.0);
        Vector128<double> result = op.Invoke(input);

        Assert.Equal(1.0, result[0], SimdDoubleTolerance);
        Assert.Equal(2.0, result[1], SimdDoubleTolerance);
    }

    [Fact]
    public void Exp2OperatorDouble_Invoke_Vector128_NegativeValues()
    {
        var op = new Exp2OperatorDouble();

        // Test negative values
        Vector128<double> input = Vector128.Create(-1.0, -2.0);
        Vector128<double> result = op.Invoke(input);

        Assert.Equal(0.5, result[0], SimdDoubleTolerance);
        Assert.Equal(0.25, result[1], SimdDoubleTolerance);
    }
#endif

    #endregion

    #region Vector128 Float Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void Exp2OperatorFloat_Invoke_Vector128_KnownValues()
    {
        var op = new Exp2OperatorFloat();

        // Test [0, 1, -1, 0.5]
        Vector128<float> input = Vector128.Create(0.0f, 1.0f, -1.0f, 0.5f);
        Vector128<float> result = op.Invoke(input);

        Assert.Equal(1.0f, result[0], SimdFloatTolerance);
        Assert.Equal(2.0f, result[1], SimdFloatTolerance);
        Assert.Equal(0.5f, result[2], SimdFloatTolerance);
        Assert.Equal(1.4142136f, result[3], SimdFloatTolerance);
    }
#endif

    #endregion

    #region Vector256 Double Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void Exp2OperatorDouble_Invoke_Vector256_KnownValues()
    {
        if (!Vector256.IsHardwareAccelerated)
        {
            return; // Skip test if AVX2 not available
        }

        var op = new Exp2OperatorDouble();

        // Test 4 values
        Vector256<double> input = Vector256.Create(0.0, 1.0, -1.0, 2.0);
        Vector256<double> result = op.Invoke(input);

        Assert.Equal(1.0, result[0], SimdDoubleTolerance);
        Assert.Equal(2.0, result[1], SimdDoubleTolerance);
        Assert.Equal(0.5, result[2], SimdDoubleTolerance);
        Assert.Equal(4.0, result[3], SimdDoubleTolerance);
    }
#endif

    #endregion

    #region Vector256 Float Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void Exp2OperatorFloat_Invoke_Vector256_KnownValues()
    {
        if (!Vector256.IsHardwareAccelerated)
        {
            return; // Skip test if AVX2 not available
        }

        var op = new Exp2OperatorFloat();

        // Test 8 values
        Vector256<float> input = Vector256.Create(
            0.0f, 1.0f, -1.0f, 0.5f,
            -0.5f, 2.0f, -2.0f, 3.0f);

        Vector256<float> result = op.Invoke(input);

        Assert.Equal(1.0f, result[0], SimdFloatTolerance);
        Assert.Equal(2.0f, result[1], SimdFloatTolerance);
        Assert.Equal(0.5f, result[2], SimdFloatTolerance);
        Assert.Equal(1.4142136f, result[3], SimdFloatTolerance);
        Assert.Equal(0.7071068f, result[4], SimdFloatTolerance);
        Assert.Equal(4.0f, result[5], SimdFloatTolerance);
        Assert.Equal(0.25f, result[6], SimdFloatTolerance);
        Assert.Equal(8.0f, result[7], SimdFloatTolerance);
    }
#endif

    #endregion

    #region Vector512 Double Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void Exp2OperatorDouble_Invoke_Vector512_KnownValues()
    {
        if (!Vector512.IsHardwareAccelerated)
        {
            return; // Skip test if AVX-512 not available
        }

        var op = new Exp2OperatorDouble();

        // Test 8 values
        Vector512<double> input = Vector512.Create(
            0.0, 1.0, -1.0, 2.0,
            -2.0, 0.5, -0.5, 3.0);

        Vector512<double> result = op.Invoke(input);

        Assert.Equal(1.0, result[0], SimdDoubleTolerance);
        Assert.Equal(2.0, result[1], SimdDoubleTolerance);
        Assert.Equal(0.5, result[2], SimdDoubleTolerance);
        Assert.Equal(4.0, result[3], SimdDoubleTolerance);
        Assert.Equal(0.25, result[4], SimdDoubleTolerance);
        Assert.Equal(1.4142135623730951, result[5], SimdDoubleTolerance);
        Assert.Equal(0.7071067811865476, result[6], SimdDoubleTolerance);
        Assert.Equal(8.0, result[7], SimdDoubleTolerance);
    }
#endif

    #endregion

    #region Vector512 Float Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void Exp2OperatorFloat_Invoke_Vector512_KnownValues()
    {
        if (!Vector512.IsHardwareAccelerated)
        {
            return; // Skip test if AVX-512 not available
        }

        var op = new Exp2OperatorFloat();

        // Test 16 values
        Vector512<float> input = Vector512.Create(
            0.0f, 1.0f, -1.0f, 0.5f,
            -0.5f, 2.0f, -2.0f, 3.0f,
            -3.0f, 0.1f, -0.1f, 1.5f,
            -1.5f, 4.0f, -4.0f, 0.0f);

        Vector512<float> result = op.Invoke(input);

        Assert.Equal(1.0f, result[0], SimdFloatTolerance);
        Assert.Equal(2.0f, result[1], SimdFloatTolerance);
        Assert.Equal(0.5f, result[2], SimdFloatTolerance);
        Assert.Equal(1.4142136f, result[3], SimdFloatTolerance);
        Assert.Equal(0.7071068f, result[4], SimdFloatTolerance);
        Assert.Equal(4.0f, result[5], SimdFloatTolerance);
        Assert.Equal(0.25f, result[6], SimdFloatTolerance);
        Assert.Equal(8.0f, result[7], SimdFloatTolerance);
        Assert.Equal(0.125f, result[8], SimdFloatTolerance);
        Assert.Equal(1.0717734f, result[9], SimdFloatTolerance);
        Assert.Equal(0.9330329f, result[10], SimdFloatTolerance);
        Assert.Equal(2.8284271f, result[11], SimdFloatTolerance);
        Assert.Equal(0.3535534f, result[12], SimdFloatTolerance);
        Assert.Equal(16.0f, result[13], SimdFloatTolerance);
        Assert.Equal(0.0625f, result[14], SimdFloatTolerance);
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
    public void Exp2OperatorDouble_Invoke_Scalar_AccuracyVsMathPow(double input)
    {
        var op = new Exp2OperatorDouble();
        double result = op.Invoke(input);
        double expected = Math.Pow(2.0, input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    [Theory]
    [InlineData(0.1f)]
    [InlineData(0.5f)]
    [InlineData(1.0f)]
    [InlineData(1.5f)]
    [InlineData(2.0f)]
    [InlineData(3.0f)]
    public void Exp2OperatorFloat_Invoke_Scalar_AccuracyVsMathFPow(float input)
    {
        var op = new Exp2OperatorFloat();
        float result = op.Invoke(input);
        float expected = MathF.Pow(2.0f, input);
        Assert.Equal(expected, result, ScalarFloatTolerance);
    }

    #endregion
}
