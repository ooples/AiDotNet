using System;
using System.Runtime.Intrinsics;
using AiDotNet.Tensors.Operators;
using Xunit;

namespace AiDotNet.Tensors.Tests.Operators;

public class AbsOperatorTests
{
    private const double ScalarDoubleTolerance = 1e-14;
    private const float ScalarFloatTolerance = 1e-6f;
    private const double SimdDoubleTolerance = 1e-14;  // Same as scalar (fallback)
    private const float SimdFloatTolerance = 1e-6f;

    #region AbsOperatorDouble - Scalar Tests

    [Theory]
    [InlineData(0.0, 0.0)]
    [InlineData(1.0, 1.0)]
    [InlineData(-1.0, 1.0)]
    [InlineData(5.5, 5.5)]
    [InlineData(-5.5, 5.5)]
    [InlineData(100.0, 100.0)]
    [InlineData(-100.0, 100.0)]
    public void AbsOperatorDouble_Invoke_Scalar_KnownValues(double input, double expected)
    {
        var op = new AbsOperatorDouble();
        double result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    [Fact]
    public void AbsOperatorDouble_Invoke_Scalar_Zero()
    {
        var op = new AbsOperatorDouble();
        double result = op.Invoke(0.0);
        Assert.Equal(0.0, result, ScalarDoubleTolerance);
    }

    [Fact]
    public void AbsOperatorDouble_Invoke_Scalar_PositiveValue()
    {
        var op = new AbsOperatorDouble();
        double input = 42.5;
        double expected = Math.Abs(input);
        double result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    [Fact]
    public void AbsOperatorDouble_Invoke_Scalar_NegativeValue()
    {
        var op = new AbsOperatorDouble();
        double input = -42.5;
        double expected = Math.Abs(input);
        double result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    [Fact]
    public void AbsOperatorDouble_Invoke_Scalar_VerySmallNegative()
    {
        var op = new AbsOperatorDouble();
        double input = -1e-10;
        double expected = Math.Abs(input);
        double result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    #endregion

    #region AbsOperatorDouble - SIMD Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void AbsOperatorDouble_Invoke_Vector128()
    {
        if (!Vector128.IsHardwareAccelerated)
        {
            return; // Skip test if hardware acceleration not available
        }

        var op = new AbsOperatorDouble();
        var input = Vector128.Create(-5.0, 5.0);
        var result = op.Invoke(input);

        Span<double> resultValues = stackalloc double[Vector128<double>.Count];
        result.CopyTo(resultValues);

        Assert.Equal(Math.Abs(-5.0), resultValues[0], SimdDoubleTolerance);
        Assert.Equal(Math.Abs(5.0), resultValues[1], SimdDoubleTolerance);
    }

    [Fact]
    public void AbsOperatorDouble_Invoke_Vector256()
    {
        if (!Vector256.IsHardwareAccelerated)
        {
            return; // Skip test if hardware acceleration not available
        }

        var op = new AbsOperatorDouble();
        var input = Vector256.Create(-5.0, 5.0, -10.0, 10.0);
        var result = op.Invoke(input);

        Span<double> resultValues = stackalloc double[Vector256<double>.Count];
        result.CopyTo(resultValues);

        Assert.Equal(Math.Abs(-5.0), resultValues[0], SimdDoubleTolerance);
        Assert.Equal(Math.Abs(5.0), resultValues[1], SimdDoubleTolerance);
        Assert.Equal(Math.Abs(-10.0), resultValues[2], SimdDoubleTolerance);
        Assert.Equal(Math.Abs(10.0), resultValues[3], SimdDoubleTolerance);
    }

    [Fact]
    public void AbsOperatorDouble_Invoke_Vector512()
    {
        if (!Vector512.IsHardwareAccelerated)
        {
            return; // Skip test if hardware acceleration not available
        }

        var op = new AbsOperatorDouble();
        var input = Vector512.Create(-5.0, 5.0, -10.0, 10.0, -15.0, 15.0, -20.0, 20.0);
        var result = op.Invoke(input);

        Span<double> resultValues = stackalloc double[Vector512<double>.Count];
        result.CopyTo(resultValues);

        for (int i = 0; i < Vector512<double>.Count; i++)
        {
            double expectedValue = Math.Abs(Vector512.GetElement(input, i));
            Assert.Equal(expectedValue, resultValues[i], SimdDoubleTolerance);
        }
    }

    [Fact]
    public void AbsOperatorDouble_Invoke_Vector256_AllNegative()
    {
        if (!Vector256.IsHardwareAccelerated)
        {
            return; // Skip test if hardware acceleration not available
        }

        var op = new AbsOperatorDouble();
        var input = Vector256.Create(-1.0, -2.0, -3.0, -4.0);
        var result = op.Invoke(input);

        Span<double> resultValues = stackalloc double[Vector256<double>.Count];
        result.CopyTo(resultValues);

        Assert.Equal(1.0, resultValues[0], SimdDoubleTolerance);
        Assert.Equal(2.0, resultValues[1], SimdDoubleTolerance);
        Assert.Equal(3.0, resultValues[2], SimdDoubleTolerance);
        Assert.Equal(4.0, resultValues[3], SimdDoubleTolerance);
    }
#endif

    #endregion

    #region AbsOperatorFloat - Scalar Tests

    [Theory]
    [InlineData(0.0f, 0.0f)]
    [InlineData(1.0f, 1.0f)]
    [InlineData(-1.0f, 1.0f)]
    [InlineData(5.5f, 5.5f)]
    [InlineData(-5.5f, 5.5f)]
    [InlineData(100.0f, 100.0f)]
    [InlineData(-100.0f, 100.0f)]
    public void AbsOperatorFloat_Invoke_Scalar_KnownValues(float input, float expected)
    {
        var op = new AbsOperatorFloat();
        float result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarFloatTolerance);
    }

    [Fact]
    public void AbsOperatorFloat_Invoke_Scalar_Zero()
    {
        var op = new AbsOperatorFloat();
        float result = op.Invoke(0.0f);
        Assert.Equal(0.0f, result, ScalarFloatTolerance);
    }

    [Fact]
    public void AbsOperatorFloat_Invoke_Scalar_PositiveValue()
    {
        var op = new AbsOperatorFloat();
        float input = 42.5f;
        float expected = MathF.Abs(input);
        float result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarFloatTolerance);
    }

    [Fact]
    public void AbsOperatorFloat_Invoke_Scalar_NegativeValue()
    {
        var op = new AbsOperatorFloat();
        float input = -42.5f;
        float expected = MathF.Abs(input);
        float result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarFloatTolerance);
    }

    [Fact]
    public void AbsOperatorFloat_Invoke_Scalar_VerySmallNegative()
    {
        var op = new AbsOperatorFloat();
        float input = -1e-6f;
        float expected = MathF.Abs(input);
        float result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarFloatTolerance);
    }

    #endregion

    #region AbsOperatorFloat - SIMD Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void AbsOperatorFloat_Invoke_Vector128()
    {
        if (!Vector128.IsHardwareAccelerated)
        {
            return; // Skip test if hardware acceleration not available
        }

        var op = new AbsOperatorFloat();
        var input = Vector128.Create(-5.0f, 5.0f, -10.0f, 10.0f);
        var result = op.Invoke(input);

        Span<float> resultValues = stackalloc float[Vector128<float>.Count];
        result.CopyTo(resultValues);

        Assert.Equal(MathF.Abs(-5.0f), resultValues[0], SimdFloatTolerance);
        Assert.Equal(MathF.Abs(5.0f), resultValues[1], SimdFloatTolerance);
        Assert.Equal(MathF.Abs(-10.0f), resultValues[2], SimdFloatTolerance);
        Assert.Equal(MathF.Abs(10.0f), resultValues[3], SimdFloatTolerance);
    }

    [Fact]
    public void AbsOperatorFloat_Invoke_Vector256()
    {
        if (!Vector256.IsHardwareAccelerated)
        {
            return; // Skip test if hardware acceleration not available
        }

        var op = new AbsOperatorFloat();
        var input = Vector256.Create(-5.0f, 5.0f, -10.0f, 10.0f, -15.0f, 15.0f, -20.0f, 20.0f);
        var result = op.Invoke(input);

        Span<float> resultValues = stackalloc float[Vector256<float>.Count];
        result.CopyTo(resultValues);

        for (int i = 0; i < Vector256<float>.Count; i++)
        {
            float expectedValue = MathF.Abs(Vector256.GetElement(input, i));
            Assert.Equal(expectedValue, resultValues[i], SimdFloatTolerance);
        }
    }

    [Fact]
    public void AbsOperatorFloat_Invoke_Vector512()
    {
        if (!Vector512.IsHardwareAccelerated)
        {
            return; // Skip test if hardware acceleration not available
        }

        var op = new AbsOperatorFloat();
        var input = Vector512.Create(-5.0f, 5.0f, -10.0f, 10.0f, -15.0f, 15.0f, -20.0f, 20.0f,
                                      -25.0f, 25.0f, -30.0f, 30.0f, -35.0f, 35.0f, -40.0f, 40.0f);
        var result = op.Invoke(input);

        Span<float> resultValues = stackalloc float[Vector512<float>.Count];
        result.CopyTo(resultValues);

        for (int i = 0; i < Vector512<float>.Count; i++)
        {
            float expectedValue = MathF.Abs(Vector512.GetElement(input, i));
            Assert.Equal(expectedValue, resultValues[i], SimdFloatTolerance);
        }
    }

    [Fact]
    public void AbsOperatorFloat_Invoke_Vector256_AllNegative()
    {
        if (!Vector256.IsHardwareAccelerated)
        {
            return; // Skip test if hardware acceleration not available
        }

        var op = new AbsOperatorFloat();
        var input = Vector256.Create(-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f);
        var result = op.Invoke(input);

        Span<float> resultValues = stackalloc float[Vector256<float>.Count];
        result.CopyTo(resultValues);

        for (int i = 0; i < Vector256<float>.Count; i++)
        {
            Assert.Equal((float)(i + 1), resultValues[i], SimdFloatTolerance);
        }
    }
#endif

    #endregion
}
