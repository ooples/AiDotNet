using System;
using System.Runtime.Intrinsics;
using AiDotNet.Tensors.Operators;
using Xunit;

namespace AiDotNet.Tensors.Tests.Operators;

public class RoundOperatorTests
{
    private const double ScalarDoubleTolerance = 1e-14;
    private const float ScalarFloatTolerance = 1e-6f;
    private const double SimdDoubleTolerance = 1e-14;
    private const float SimdFloatTolerance = 1e-6f;

    #region RoundOperatorDouble - Scalar Tests

    [Theory]
    [InlineData(0.0, 0.0)]
    [InlineData(1.0, 1.0)]
    [InlineData(-1.0, -1.0)]
    [InlineData(5.5, 6.0)]
    [InlineData(5.4, 5.0)]
    [InlineData(-5.5, -6.0)]
    [InlineData(-5.4, -5.0)]
    [InlineData(100.3, 100.0)]
    [InlineData(-100.7, -101.0)]
    [InlineData(2.5, 2.0)]  // Banker's rounding
    [InlineData(3.5, 4.0)]  // Banker's rounding
    public void RoundOperatorDouble_Invoke_Scalar_KnownValues(double input, double expected)
    {
        var op = new RoundOperatorDouble();
        double result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    [Fact]
    public void RoundOperatorDouble_Invoke_Scalar_Zero()
    {
        var op = new RoundOperatorDouble();
        double result = op.Invoke(0.0);
        Assert.Equal(0.0, result, ScalarDoubleTolerance);
    }

    [Fact]
    public void RoundOperatorDouble_Invoke_Scalar_PositiveValue()
    {
        var op = new RoundOperatorDouble();
        double input = 42.7;
        double expected = Math.Round(input);
        double result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    [Fact]
    public void RoundOperatorDouble_Invoke_Scalar_NegativeValue()
    {
        var op = new RoundOperatorDouble();
        double input = -42.3;
        double expected = Math.Round(input);
        double result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    #endregion

    #region RoundOperatorDouble - SIMD Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void RoundOperatorDouble_Invoke_Vector128()
    {
        if (!Vector128.IsHardwareAccelerated)
        {
            return;
        }

        var op = new RoundOperatorDouble();
        var input = Vector128.Create(-5.7, 5.3);
        var result = op.Invoke(input);

        Span<double> resultValues = stackalloc double[Vector128<double>.Count];
        result.CopyTo(resultValues);

        Assert.Equal(Math.Round(-5.7), resultValues[0], SimdDoubleTolerance);
        Assert.Equal(Math.Round(5.3), resultValues[1], SimdDoubleTolerance);
    }

    [Fact]
    public void RoundOperatorDouble_Invoke_Vector256()
    {
        if (!Vector256.IsHardwareAccelerated)
        {
            return;
        }

        var op = new RoundOperatorDouble();
        var input = Vector256.Create(-5.7, 5.3, 10.5, -10.5);
        var result = op.Invoke(input);

        Span<double> resultValues = stackalloc double[Vector256<double>.Count];
        result.CopyTo(resultValues);

        Assert.Equal(Math.Round(-5.7), resultValues[0], SimdDoubleTolerance);
        Assert.Equal(Math.Round(5.3), resultValues[1], SimdDoubleTolerance);
        Assert.Equal(Math.Round(10.5), resultValues[2], SimdDoubleTolerance);
        Assert.Equal(Math.Round(-10.5), resultValues[3], SimdDoubleTolerance);
    }
#endif

    #endregion

    #region RoundOperatorFloat - Scalar Tests

    [Theory]
    [InlineData(0.0f, 0.0f)]
    [InlineData(1.0f, 1.0f)]
    [InlineData(-1.0f, -1.0f)]
    [InlineData(5.5f, 6.0f)]
    [InlineData(5.4f, 5.0f)]
    [InlineData(-5.5f, -6.0f)]
    [InlineData(-5.4f, -5.0f)]
    [InlineData(100.3f, 100.0f)]
    [InlineData(-100.7f, -101.0f)]
    public void RoundOperatorFloat_Invoke_Scalar_KnownValues(float input, float expected)
    {
        var op = new RoundOperatorFloat();
        float result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarFloatTolerance);
    }

    [Fact]
    public void RoundOperatorFloat_Invoke_Scalar_Zero()
    {
        var op = new RoundOperatorFloat();
        float result = op.Invoke(0.0f);
        Assert.Equal(0.0f, result, ScalarFloatTolerance);
    }

    [Fact]
    public void RoundOperatorFloat_Invoke_Scalar_PositiveValue()
    {
        var op = new RoundOperatorFloat();
        float input = 42.7f;
        float expected = MathF.Round(input);
        float result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarFloatTolerance);
    }

    [Fact]
    public void RoundOperatorFloat_Invoke_Scalar_NegativeValue()
    {
        var op = new RoundOperatorFloat();
        float input = -42.3f;
        float expected = MathF.Round(input);
        float result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarFloatTolerance);
    }

    #endregion

    #region RoundOperatorFloat - SIMD Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void RoundOperatorFloat_Invoke_Vector128()
    {
        if (!Vector128.IsHardwareAccelerated)
        {
            return;
        }

        var op = new RoundOperatorFloat();
        var input = Vector128.Create(-5.7f, 5.3f, 10.5f, -10.5f);
        var result = op.Invoke(input);

        Span<float> resultValues = stackalloc float[Vector128<float>.Count];
        result.CopyTo(resultValues);

        Assert.Equal(MathF.Round(-5.7f), resultValues[0], SimdFloatTolerance);
        Assert.Equal(MathF.Round(5.3f), resultValues[1], SimdFloatTolerance);
        Assert.Equal(MathF.Round(10.5f), resultValues[2], SimdFloatTolerance);
        Assert.Equal(MathF.Round(-10.5f), resultValues[3], SimdFloatTolerance);
    }

    [Fact]
    public void RoundOperatorFloat_Invoke_Vector256()
    {
        if (!Vector256.IsHardwareAccelerated)
        {
            return;
        }

        var op = new RoundOperatorFloat();
        var input = Vector256.Create(-5.7f, 5.3f, 10.5f, -10.5f, 2.2f, -2.8f, 7.9f, -7.1f);
        var result = op.Invoke(input);

        Span<float> resultValues = stackalloc float[Vector256<float>.Count];
        result.CopyTo(resultValues);

        for (int i = 0; i < 8; i++)
        {
            Span<float> inputValues = stackalloc float[8];
            input.CopyTo(inputValues);
            Assert.Equal(MathF.Round(inputValues[i]), resultValues[i], SimdFloatTolerance);
        }
    }
#endif

    #endregion
}
