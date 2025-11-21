using System;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
#endif
using AiDotNet.Tensors.Operators;
using Xunit;

namespace AiDotNet.Tensors.Tests.Operators;

public class SqrtOperatorTests
{
    private const double ScalarDoubleTolerance = 1e-14;
    private const float ScalarFloatTolerance = 1e-6f;
    private const double SimdDoubleTolerance = 1e-14;  // Same as scalar (fallback)
    private const float SimdFloatTolerance = 1e-6f;

    #region SqrtOperatorDouble - Scalar Tests

    [Theory]
    [InlineData(0.0, 0.0)]
    [InlineData(1.0, 1.0)]
    [InlineData(4.0, 2.0)]
    [InlineData(9.0, 3.0)]
    [InlineData(16.0, 4.0)]
    [InlineData(2.0, 1.4142135623730951)]
    [InlineData(100.0, 10.0)]
    public void SqrtOperatorDouble_Invoke_Scalar_KnownValues(double input, double expected)
    {
        var op = new SqrtOperatorDouble();
        double result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    [Fact]
    public void SqrtOperatorDouble_Invoke_Scalar_Zero()
    {
        var op = new SqrtOperatorDouble();
        double result = op.Invoke(0.0);
        Assert.Equal(0.0, result, ScalarDoubleTolerance);
    }

    [Fact]
    public void SqrtOperatorDouble_Invoke_Scalar_SmallValue()
    {
        var op = new SqrtOperatorDouble();
        double input = 0.01;
        double expected = Math.Sqrt(input);
        double result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    [Fact]
    public void SqrtOperatorDouble_Invoke_Scalar_LargeValue()
    {
        var op = new SqrtOperatorDouble();
        double input = 10000.0;
        double expected = Math.Sqrt(input);
        double result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    #endregion

    #region SqrtOperatorDouble - SIMD Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void SqrtOperatorDouble_Invoke_Vector128()
    {
        if (!Vector128.IsHardwareAccelerated)
        {
            return; // Skip test if hardware acceleration not available
        }

        var op = new SqrtOperatorDouble();
        var input = Vector128.Create(0.0, 1.0);
        var result = op.Invoke(input);

        Span<double> resultValues = stackalloc double[Vector128<double>.Count];
        result.CopyTo(resultValues);

        Assert.Equal(Math.Sqrt(0.0), resultValues[0], SimdDoubleTolerance);
        Assert.Equal(Math.Sqrt(1.0), resultValues[1], SimdDoubleTolerance);
    }

    [Fact]
    public void SqrtOperatorDouble_Invoke_Vector256()
    {
        if (!Vector256.IsHardwareAccelerated)
        {
            return; // Skip test if hardware acceleration not available
        }

        var op = new SqrtOperatorDouble();
        var input = Vector256.Create(0.0, 1.0, 4.0, 9.0);
        var result = op.Invoke(input);

        Span<double> resultValues = stackalloc double[Vector256<double>.Count];
        result.CopyTo(resultValues);

        Assert.Equal(Math.Sqrt(0.0), resultValues[0], SimdDoubleTolerance);
        Assert.Equal(Math.Sqrt(1.0), resultValues[1], SimdDoubleTolerance);
        Assert.Equal(Math.Sqrt(4.0), resultValues[2], SimdDoubleTolerance);
        Assert.Equal(Math.Sqrt(9.0), resultValues[3], SimdDoubleTolerance);
    }

    [Fact]
    public void SqrtOperatorDouble_Invoke_Vector512()
    {
        if (!Vector512.IsHardwareAccelerated)
        {
            return; // Skip test if hardware acceleration not available
        }

        var op = new SqrtOperatorDouble();
        var input = Vector512.Create(0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0);
        var result = op.Invoke(input);

        Span<double> resultValues = stackalloc double[Vector512<double>.Count];
        result.CopyTo(resultValues);

        for (int i = 0; i < Vector512<double>.Count; i++)
        {
            double expectedValue = Math.Sqrt(Vector512.GetElement(input, i));
            Assert.Equal(expectedValue, resultValues[i], SimdDoubleTolerance);
        }
    }
#endif

    #endregion

    #region SqrtOperatorFloat - Scalar Tests

    [Theory]
    [InlineData(0.0f, 0.0f)]
    [InlineData(1.0f, 1.0f)]
    [InlineData(4.0f, 2.0f)]
    [InlineData(9.0f, 3.0f)]
    [InlineData(16.0f, 4.0f)]
    [InlineData(2.0f, 1.4142135f)]
    [InlineData(100.0f, 10.0f)]
    public void SqrtOperatorFloat_Invoke_Scalar_KnownValues(float input, float expected)
    {
        var op = new SqrtOperatorFloat();
        float result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarFloatTolerance);
    }

    [Fact]
    public void SqrtOperatorFloat_Invoke_Scalar_Zero()
    {
        var op = new SqrtOperatorFloat();
        float result = op.Invoke(0.0f);
        Assert.Equal(0.0f, result, ScalarFloatTolerance);
    }

    [Fact]
    public void SqrtOperatorFloat_Invoke_Scalar_SmallValue()
    {
        var op = new SqrtOperatorFloat();
        float input = 0.01f;
        float expected = MathF.Sqrt(input);
        float result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarFloatTolerance);
    }

    [Fact]
    public void SqrtOperatorFloat_Invoke_Scalar_LargeValue()
    {
        var op = new SqrtOperatorFloat();
        float input = 10000.0f;
        float expected = MathF.Sqrt(input);
        float result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarFloatTolerance);
    }

    #endregion

    #region SqrtOperatorFloat - SIMD Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void SqrtOperatorFloat_Invoke_Vector128()
    {
        if (!Vector128.IsHardwareAccelerated)
        {
            return; // Skip test if hardware acceleration not available
        }

        var op = new SqrtOperatorFloat();
        var input = Vector128.Create(0.0f, 1.0f, 4.0f, 9.0f);
        var result = op.Invoke(input);

        Span<float> resultValues = stackalloc float[Vector128<float>.Count];
        result.CopyTo(resultValues);

        Assert.Equal(MathF.Sqrt(0.0f), resultValues[0], SimdFloatTolerance);
        Assert.Equal(MathF.Sqrt(1.0f), resultValues[1], SimdFloatTolerance);
        Assert.Equal(MathF.Sqrt(4.0f), resultValues[2], SimdFloatTolerance);
        Assert.Equal(MathF.Sqrt(9.0f), resultValues[3], SimdFloatTolerance);
    }

    [Fact]
    public void SqrtOperatorFloat_Invoke_Vector256()
    {
        if (!Vector256.IsHardwareAccelerated)
        {
            return; // Skip test if hardware acceleration not available
        }

        var op = new SqrtOperatorFloat();
        var input = Vector256.Create(0.0f, 1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f);
        var result = op.Invoke(input);

        Span<float> resultValues = stackalloc float[Vector256<float>.Count];
        result.CopyTo(resultValues);

        for (int i = 0; i < Vector256<float>.Count; i++)
        {
            float expectedValue = MathF.Sqrt(Vector256.GetElement(input, i));
            Assert.Equal(expectedValue, resultValues[i], SimdFloatTolerance);
        }
    }

    [Fact]
    public void SqrtOperatorFloat_Invoke_Vector512()
    {
        if (!Vector512.IsHardwareAccelerated)
        {
            return; // Skip test if hardware acceleration not available
        }

        var op = new SqrtOperatorFloat();
        var input = Vector512.Create(0.0f, 1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f,
                                      64.0f, 81.0f, 100.0f, 121.0f, 144.0f, 169.0f, 196.0f, 225.0f);
        var result = op.Invoke(input);

        Span<float> resultValues = stackalloc float[Vector512<float>.Count];
        result.CopyTo(resultValues);

        for (int i = 0; i < Vector512<float>.Count; i++)
        {
            float expectedValue = MathF.Sqrt(Vector512.GetElement(input, i));
            Assert.Equal(expectedValue, resultValues[i], SimdFloatTolerance);
        }
    }
#endif

    #endregion
}
