using System;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
#endif
using AiDotNet.Tensors.Operators;
using Xunit;

namespace AiDotNet.Tensors.Tests.Operators;

/// <summary>
/// Unit tests for ExpOperator implementations (scalar and SIMD).
/// </summary>
public class ExpOperatorTests
{
    // Scalar operations use Math.Exp/MathF.Exp, so very high accuracy
    private const double ScalarDoubleTolerance = 1e-14;  // Machine epsilon for double
    private const float ScalarFloatTolerance = 1e-6f;    // Machine epsilon for float

    // SIMD currently uses scalar fallback, so same accuracy as scalar
    // NOTE: When polynomial approximations are added, these will need to be relaxed
    private const double SimdDoubleTolerance = 1e-14;
    private const float SimdFloatTolerance = 1e-6f;

    #region Scalar Double Tests

    [Theory]
    [InlineData(0.0, 1.0)]                 // e^0 = 1
    [InlineData(1.0, 2.718281828459045)]   // e^1 = e
    [InlineData(2.0, 7.38905609893065)]    // e^2
    [InlineData(-1.0, 0.36787944117144233)]  // e^-1 = 1/e
    [InlineData(0.5, 1.6487212707001282)]  // e^0.5
    public void ExpOperatorDouble_Invoke_Scalar_KnownValues(double input, double expected)
    {
        var op = new ExpOperatorDouble();
        double result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    [Theory]
    [InlineData(-0.5, 0.6065306597126334)]
    [InlineData(-2.0, 0.1353352832366127)]
    [InlineData(-5.0, 0.006737946999085467)]
    public void ExpOperatorDouble_Invoke_Scalar_NegativeValues(double input, double expected)
    {
        var op = new ExpOperatorDouble();
        double result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    [Theory]
    [InlineData(10.0)]
    [InlineData(-10.0)]
    [InlineData(5.0)]
    [InlineData(-5.0)]
    public void ExpOperatorDouble_Invoke_Scalar_LargeValues(double input)
    {
        var op = new ExpOperatorDouble();
        double result = op.Invoke(input);
        double expected = Math.Exp(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    #endregion

    #region Scalar Float Tests

    [Theory]
    [InlineData(0.0f, 1.0f)]
    [InlineData(1.0f, 2.7182818f)]
    [InlineData(2.0f, 7.389056f)]
    [InlineData(-1.0f, 0.36787945f)]
    [InlineData(0.5f, 1.6487213f)]
    public void ExpOperatorFloat_Invoke_Scalar_KnownValues(float input, float expected)
    {
        var op = new ExpOperatorFloat();
        float result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarFloatTolerance);
    }

    [Theory]
    [InlineData(-0.5f, 0.60653067f)]
    [InlineData(-2.0f, 0.13533528f)]
    [InlineData(-5.0f, 0.006737947f)]
    public void ExpOperatorFloat_Invoke_Scalar_NegativeValues(float input, float expected)
    {
        var op = new ExpOperatorFloat();
        float result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarFloatTolerance);
    }

    #endregion

    #region Vector128 Double Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void ExpOperatorDouble_Invoke_Vector128_KnownValues()
    {
        var op = new ExpOperatorDouble();

        // Test [0, 1] -> [1, e]
        Vector128<double> input = Vector128.Create(0.0, 1.0);
        Vector128<double> result = op.Invoke(input);

        Assert.Equal(1.0, result[0], SimdDoubleTolerance);
        Assert.Equal(2.718281828459045, result[1], SimdDoubleTolerance);
    }

    [Fact]
    public void ExpOperatorDouble_Invoke_Vector128_NegativeValues()
    {
        var op = new ExpOperatorDouble();

        // Test negative values
        Vector128<double> input = Vector128.Create(-1.0, -2.0);
        Vector128<double> result = op.Invoke(input);

        Assert.Equal(0.36787944117144233, result[0], SimdDoubleTolerance);
        Assert.Equal(0.1353352832366127, result[1], SimdDoubleTolerance);
    }
#endif

    #endregion

    #region Vector128 Float Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void ExpOperatorFloat_Invoke_Vector128_KnownValues()
    {
        var op = new ExpOperatorFloat();

        // Test [0, 1, -1, 0.5]
        Vector128<float> input = Vector128.Create(0.0f, 1.0f, -1.0f, 0.5f);
        Vector128<float> result = op.Invoke(input);

        Assert.Equal(1.0f, result[0], SimdFloatTolerance);
        Assert.Equal(2.7182818f, result[1], SimdFloatTolerance);
        Assert.Equal(0.36787945f, result[2], SimdFloatTolerance);
        Assert.Equal(1.6487213f, result[3], SimdFloatTolerance);
    }
#endif

    #endregion

    #region Vector256 Double Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void ExpOperatorDouble_Invoke_Vector256_KnownValues()
    {
        if (!Vector256.IsHardwareAccelerated)
        {
            return; // Skip test if AVX2 not available
        }

        var op = new ExpOperatorDouble();

        // Test 4 values
        Vector256<double> input = Vector256.Create(0.0, 1.0, -1.0, 2.0);
        Vector256<double> result = op.Invoke(input);

        Assert.Equal(1.0, result[0], SimdDoubleTolerance);
        Assert.Equal(2.718281828459045, result[1], SimdDoubleTolerance);
        Assert.Equal(0.36787944117144233, result[2], SimdDoubleTolerance);
        Assert.Equal(7.38905609893065, result[3], SimdDoubleTolerance);
    }
#endif

    #endregion

    #region Vector256 Float Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void ExpOperatorFloat_Invoke_Vector256_KnownValues()
    {
        if (!Vector256.IsHardwareAccelerated)
        {
            return; // Skip test if AVX2 not available
        }

        var op = new ExpOperatorFloat();

        // Test 8 values
        Vector256<float> input = Vector256.Create(
            0.0f, 1.0f, -1.0f, 0.5f,
            -0.5f, 2.0f, -2.0f, 3.0f);

        Vector256<float> result = op.Invoke(input);

        Assert.Equal(1.0f, result[0], SimdFloatTolerance);
        Assert.Equal(2.7182818f, result[1], SimdFloatTolerance);
        Assert.Equal(0.36787945f, result[2], SimdFloatTolerance);
        Assert.Equal(1.6487213f, result[3], SimdFloatTolerance);
        Assert.Equal(0.60653067f, result[4], SimdFloatTolerance);
        Assert.Equal(7.389056f, result[5], SimdFloatTolerance);
        Assert.Equal(0.13533528f, result[6], SimdFloatTolerance);
        Assert.Equal(20.085537f, result[7], SimdFloatTolerance);
    }
#endif

    #endregion

    #region Vector512 Double Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void ExpOperatorDouble_Invoke_Vector512_KnownValues()
    {
        if (!Vector512.IsHardwareAccelerated)
        {
            return; // Skip test if AVX-512 not available
        }

        var op = new ExpOperatorDouble();

        // Test 8 values
        Vector512<double> input = Vector512.Create(
            0.0, 1.0, -1.0, 2.0,
            -2.0, 0.5, -0.5, 3.0);

        Vector512<double> result = op.Invoke(input);

        Assert.Equal(1.0, result[0], SimdDoubleTolerance);
        Assert.Equal(2.718281828459045, result[1], SimdDoubleTolerance);
        Assert.Equal(0.36787944117144233, result[2], SimdDoubleTolerance);
        Assert.Equal(7.38905609893065, result[3], SimdDoubleTolerance);
        Assert.Equal(0.1353352832366127, result[4], SimdDoubleTolerance);
        Assert.Equal(1.6487212707001282, result[5], SimdDoubleTolerance);
        Assert.Equal(0.6065306597126334, result[6], SimdDoubleTolerance);
        Assert.Equal(20.085536923187668, result[7], SimdDoubleTolerance);
    }
#endif

    #endregion

    #region Vector512 Float Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void ExpOperatorFloat_Invoke_Vector512_KnownValues()
    {
        if (!Vector512.IsHardwareAccelerated)
        {
            return; // Skip test if AVX-512 not available
        }

        var op = new ExpOperatorFloat();

        // Test 16 values
        Vector512<float> input = Vector512.Create(
            0.0f, 1.0f, -1.0f, 0.5f,
            -0.5f, 2.0f, -2.0f, 3.0f,
            -3.0f, 0.1f, -0.1f, 1.5f,
            -1.5f, 4.0f, -4.0f, 0.0f);

        Vector512<float> result = op.Invoke(input);

        Assert.Equal(1.0f, result[0], SimdFloatTolerance);
        Assert.Equal(2.7182818f, result[1], SimdFloatTolerance);
        Assert.Equal(0.36787945f, result[2], SimdFloatTolerance);
        Assert.Equal(1.6487213f, result[3], SimdFloatTolerance);
        Assert.Equal(0.60653067f, result[4], SimdFloatTolerance);
        Assert.Equal(7.389056f, result[5], SimdFloatTolerance);
        Assert.Equal(0.13533528f, result[6], SimdFloatTolerance);
        Assert.Equal(20.085537f, result[7], SimdFloatTolerance);
        Assert.Equal(0.04978707f, result[8], SimdFloatTolerance);
        Assert.Equal(1.1051709f, result[9], SimdFloatTolerance);
        Assert.Equal(0.9048374f, result[10], SimdFloatTolerance);
        Assert.Equal(4.4816891f, result[11], SimdFloatTolerance);
        Assert.Equal(0.22313017f, result[12], SimdFloatTolerance);
        Assert.Equal(54.59815f, result[13], SimdFloatTolerance);
        Assert.Equal(0.018315639f, result[14], SimdFloatTolerance);
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
    public void ExpOperatorDouble_Invoke_Scalar_AccuracyVsMathExp(double input)
    {
        var op = new ExpOperatorDouble();
        double result = op.Invoke(input);
        double expected = Math.Exp(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    [Theory]
    [InlineData(0.1f)]
    [InlineData(0.5f)]
    [InlineData(1.0f)]
    [InlineData(1.5f)]
    [InlineData(2.0f)]
    [InlineData(3.0f)]
    public void ExpOperatorFloat_Invoke_Scalar_AccuracyVsMathFExp(float input)
    {
        var op = new ExpOperatorFloat();
        float result = op.Invoke(input);
        float expected = MathF.Exp(input);
        Assert.Equal(expected, result, ScalarFloatTolerance);
    }

    #endregion
}
