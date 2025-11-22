using System;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
#endif
using AiDotNet.Tensors.Operators;
using Xunit;

namespace AiDotNet.Tensors.Tests.Operators;

/// <summary>
/// Unit tests for LogOperator implementations (scalar and SIMD).
/// </summary>
public class LogOperatorTests
{
    // Scalar operations use Math.Log/MathF.Log, so very high accuracy
    private const double ScalarDoubleTolerance = 1e-14;  // Machine epsilon for double
    private const float ScalarFloatTolerance = 1e-6f;    // Machine epsilon for float

    // SIMD currently uses scalar fallback, so same accuracy as scalar
    // NOTE: When polynomial approximations are added, these will need to be relaxed
    private const double SimdDoubleTolerance = 1e-14;
    private const float SimdFloatTolerance = 1e-6f;

    #region Scalar Double Tests

    [Theory]
    [InlineData(1.0, 0.0)]                         // ln(1) = 0
    [InlineData(Math.E, 1.0)]                      // ln(e) = 1
    [InlineData(Math.E * Math.E, 2.0)]             // ln(e^2) = 2
    [InlineData(10.0, 2.302585092994046)]          // ln(10)
    [InlineData(2.0, 0.6931471805599453)]          // ln(2)
    public void LogOperatorDouble_Invoke_Scalar_KnownValues(double input, double expected)
    {
        var op = new LogOperatorDouble();
        double result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    [Theory]
    [InlineData(0.5, -0.6931471805599453)]         // ln(0.5) = -ln(2)
    [InlineData(0.1, -2.302585092994046)]          // ln(0.1) = -ln(10)
    [InlineData(0.01, -4.605170185988092)]         // ln(0.01) = -ln(100)
    public void LogOperatorDouble_Invoke_Scalar_SmallValues(double input, double expected)
    {
        var op = new LogOperatorDouble();
        double result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    [Theory]
    [InlineData(100.0)]
    [InlineData(1000.0)]
    [InlineData(0.001)]
    [InlineData(0.0001)]
    public void LogOperatorDouble_Invoke_Scalar_LargeAndSmallValues(double input)
    {
        var op = new LogOperatorDouble();
        double result = op.Invoke(input);
        double expected = Math.Log(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    #endregion

    #region Scalar Float Tests

    [Theory]
    [InlineData(1.0f, 0.0f)]
    [InlineData(2.7182818f, 1.0f)]
    [InlineData(10.0f, 2.3025851f)]
    [InlineData(2.0f, 0.6931472f)]
    public void LogOperatorFloat_Invoke_Scalar_KnownValues(float input, float expected)
    {
        var op = new LogOperatorFloat();
        float result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarFloatTolerance);
    }

    [Theory]
    [InlineData(0.5f, -0.6931472f)]
    [InlineData(0.1f, -2.3025851f)]
    [InlineData(0.01f, -4.6051702f)]
    public void LogOperatorFloat_Invoke_Scalar_SmallValues(float input, float expected)
    {
        var op = new LogOperatorFloat();
        float result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarFloatTolerance);
    }

    #endregion

    #region Vector128 Double Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void LogOperatorDouble_Invoke_Vector128_KnownValues()
    {
        var op = new LogOperatorDouble();

        // Test [1, e] -> [0, 1]
        Vector128<double> input = Vector128.Create(1.0, Math.E);
        Vector128<double> result = op.Invoke(input);

        Assert.Equal(0.0, result[0], SimdDoubleTolerance);
        Assert.Equal(1.0, result[1], SimdDoubleTolerance);
    }

    [Fact]
    public void LogOperatorDouble_Invoke_Vector128_SmallValues()
    {
        var op = new LogOperatorDouble();

        // Test small values
        Vector128<double> input = Vector128.Create(0.5, 0.1);
        Vector128<double> result = op.Invoke(input);

        Assert.Equal(-0.6931471805599453, result[0], SimdDoubleTolerance);
        Assert.Equal(-2.302585092994046, result[1], SimdDoubleTolerance);
    }
#endif

    #endregion

    #region Vector128 Float Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void LogOperatorFloat_Invoke_Vector128_KnownValues()
    {
        var op = new LogOperatorFloat();

        // Test [1, e, 10, 2]
        Vector128<float> input = Vector128.Create(1.0f, 2.7182818f, 10.0f, 2.0f);
        Vector128<float> result = op.Invoke(input);

        Assert.Equal(0.0f, result[0], SimdFloatTolerance);
        Assert.Equal(1.0f, result[1], SimdFloatTolerance);
        Assert.Equal(2.3025851f, result[2], SimdFloatTolerance);
        Assert.Equal(0.6931472f, result[3], SimdFloatTolerance);
    }
#endif

    #endregion

    #region Vector256 Double Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void LogOperatorDouble_Invoke_Vector256_KnownValues()
    {
        if (!Vector256.IsHardwareAccelerated)
        {
            return; // Skip test if AVX2 not available
        }

        var op = new LogOperatorDouble();

        // Test 4 values
        Vector256<double> input = Vector256.Create(1.0, Math.E, 10.0, 2.0);
        Vector256<double> result = op.Invoke(input);

        Assert.Equal(0.0, result[0], SimdDoubleTolerance);
        Assert.Equal(1.0, result[1], SimdDoubleTolerance);
        Assert.Equal(2.302585092994046, result[2], SimdDoubleTolerance);
        Assert.Equal(0.6931471805599453, result[3], SimdDoubleTolerance);
    }
#endif

    #endregion

    #region Vector256 Float Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void LogOperatorFloat_Invoke_Vector256_KnownValues()
    {
        if (!Vector256.IsHardwareAccelerated)
        {
            return; // Skip test if AVX2 not available
        }

        var op = new LogOperatorFloat();

        // Test 8 values
        Vector256<float> input = Vector256.Create(
            1.0f, 2.7182818f, 10.0f, 2.0f,
            0.5f, 0.1f, 100.0f, 5.0f);

        Vector256<float> result = op.Invoke(input);

        Assert.Equal(0.0f, result[0], SimdFloatTolerance);
        Assert.Equal(1.0f, result[1], SimdFloatTolerance);
        Assert.Equal(2.3025851f, result[2], SimdFloatTolerance);
        Assert.Equal(0.6931472f, result[3], SimdFloatTolerance);
        Assert.Equal(-0.6931472f, result[4], SimdFloatTolerance);
        Assert.Equal(-2.3025851f, result[5], SimdFloatTolerance);
        Assert.Equal(4.6051702f, result[6], SimdFloatTolerance);
        Assert.Equal(1.6094379f, result[7], SimdFloatTolerance);
    }
#endif

    #endregion

    #region Vector512 Double Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void LogOperatorDouble_Invoke_Vector512_KnownValues()
    {
        if (!Vector512.IsHardwareAccelerated)
        {
            return; // Skip test if AVX-512 not available
        }

        var op = new LogOperatorDouble();

        // Test 8 values
        Vector512<double> input = Vector512.Create(
            1.0, Math.E, 10.0, 2.0,
            0.5, 0.1, 100.0, 5.0);

        Vector512<double> result = op.Invoke(input);

        Assert.Equal(0.0, result[0], SimdDoubleTolerance);
        Assert.Equal(1.0, result[1], SimdDoubleTolerance);
        Assert.Equal(2.302585092994046, result[2], SimdDoubleTolerance);
        Assert.Equal(0.6931471805599453, result[3], SimdDoubleTolerance);
        Assert.Equal(-0.6931471805599453, result[4], SimdDoubleTolerance);
        Assert.Equal(-2.302585092994046, result[5], SimdDoubleTolerance);
        Assert.Equal(4.605170185988092, result[6], SimdDoubleTolerance);
        Assert.Equal(1.6094379124341003, result[7], SimdDoubleTolerance);
    }
#endif

    #endregion

    #region Vector512 Float Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void LogOperatorFloat_Invoke_Vector512_KnownValues()
    {
        if (!Vector512.IsHardwareAccelerated)
        {
            return; // Skip test if AVX-512 not available
        }

        var op = new LogOperatorFloat();

        // Test 16 values
        Vector512<float> input = Vector512.Create(
            1.0f, 2.7182818f, 10.0f, 2.0f,
            0.5f, 0.1f, 100.0f, 5.0f,
            3.0f, 0.3f, 1000.0f, 7.0f,
            0.7f, 0.07f, 50.0f, 1.0f);

        Vector512<float> result = op.Invoke(input);

        Assert.Equal(0.0f, result[0], SimdFloatTolerance);
        Assert.Equal(1.0f, result[1], SimdFloatTolerance);
        Assert.Equal(2.3025851f, result[2], SimdFloatTolerance);
        Assert.Equal(0.6931472f, result[3], SimdFloatTolerance);
        Assert.Equal(-0.6931472f, result[4], SimdFloatTolerance);
        Assert.Equal(-2.3025851f, result[5], SimdFloatTolerance);
        Assert.Equal(4.6051702f, result[6], SimdFloatTolerance);
        Assert.Equal(1.6094379f, result[7], SimdFloatTolerance);
        Assert.Equal(1.0986123f, result[8], SimdFloatTolerance);
        Assert.Equal(-1.2039728f, result[9], SimdFloatTolerance);
        Assert.Equal(6.9077554f, result[10], SimdFloatTolerance);
        Assert.Equal(1.9459101f, result[11], SimdFloatTolerance);
        Assert.Equal(-0.35667494f, result[12], SimdFloatTolerance);
        Assert.Equal(-2.6592600f, result[13], SimdFloatTolerance);
        Assert.Equal(3.9120231f, result[14], SimdFloatTolerance);
        Assert.Equal(0.0f, result[15], SimdFloatTolerance);
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
    [InlineData(10.0)]
    public void LogOperatorDouble_Invoke_Scalar_AccuracyVsMathLog(double input)
    {
        var op = new LogOperatorDouble();
        double result = op.Invoke(input);
        double expected = Math.Log(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    [Theory]
    [InlineData(0.1f)]
    [InlineData(0.5f)]
    [InlineData(1.0f)]
    [InlineData(1.5f)]
    [InlineData(2.0f)]
    [InlineData(10.0f)]
    public void LogOperatorFloat_Invoke_Scalar_AccuracyVsMathFLog(float input)
    {
        var op = new LogOperatorFloat();
        float result = op.Invoke(input);
        float expected = MathF.Log(input);
        Assert.Equal(expected, result, ScalarFloatTolerance);
    }

    #endregion
}
