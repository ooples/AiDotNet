using System;
using System.Runtime.Intrinsics;
using AiDotNet.Tensors.Operators;
using Xunit;

namespace AiDotNet.Tensors.Tests.Operators;

public class TanOperatorTests
{
    private const double ScalarDoubleTolerance = 1e-14;
    private const float ScalarFloatTolerance = 1e-6f;
    private const double SimdDoubleTolerance = 1e-14;  // Same as scalar (fallback)
    private const float SimdFloatTolerance = 1e-6f;

    #region TanOperatorDouble - Scalar Tests

    [Theory]
    [InlineData(0.0, 0.0)]
    [InlineData(Math.PI / 4, 1.0)]
    [InlineData(-Math.PI / 4, -1.0)]
    [InlineData(Math.PI / 6, 0.57735026918962576)]  // tan(30°) ≈ 1/√3
    public void TanOperatorDouble_Invoke_Scalar_KnownValues(double input, double expected)
    {
        var op = new TanOperatorDouble();
        double result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    [Fact]
    public void TanOperatorDouble_Invoke_Scalar_Zero()
    {
        var op = new TanOperatorDouble();
        double result = op.Invoke(0.0);
        Assert.Equal(0.0, result, ScalarDoubleTolerance);
    }

    [Fact]
    public void TanOperatorDouble_Invoke_Scalar_Negative()
    {
        var op = new TanOperatorDouble();
        double input = -0.5;
        double expected = Math.Tan(input);
        double result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    [Fact]
    public void TanOperatorDouble_Invoke_Scalar_LargeValue()
    {
        var op = new TanOperatorDouble();
        double input = 100.0;
        double expected = Math.Tan(input);
        double result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarDoubleTolerance);
    }

    #endregion

    #region TanOperatorDouble - SIMD Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void TanOperatorDouble_Invoke_Vector128()
    {
        if (!Vector128.IsHardwareAccelerated)
        {
            return; // Skip test if hardware acceleration not available
        }

        var op = new TanOperatorDouble();
        var input = Vector128.Create(0.0, Math.PI / 4);
        var result = op.Invoke(input);

        Span<double> resultValues = stackalloc double[Vector128<double>.Count];
        result.CopyTo(resultValues);

        Assert.Equal(Math.Tan(0.0), resultValues[0], SimdDoubleTolerance);
        Assert.Equal(Math.Tan(Math.PI / 4), resultValues[1], SimdDoubleTolerance);
    }

    [Fact]
    public void TanOperatorDouble_Invoke_Vector256()
    {
        if (!Vector256.IsHardwareAccelerated)
        {
            return; // Skip test if hardware acceleration not available
        }

        var op = new TanOperatorDouble();
        var input = Vector256.Create(0.0, Math.PI / 4, -Math.PI / 4, Math.PI / 6);
        var result = op.Invoke(input);

        Span<double> resultValues = stackalloc double[Vector256<double>.Count];
        result.CopyTo(resultValues);

        Assert.Equal(Math.Tan(0.0), resultValues[0], SimdDoubleTolerance);
        Assert.Equal(Math.Tan(Math.PI / 4), resultValues[1], SimdDoubleTolerance);
        Assert.Equal(Math.Tan(-Math.PI / 4), resultValues[2], SimdDoubleTolerance);
        Assert.Equal(Math.Tan(Math.PI / 6), resultValues[3], SimdDoubleTolerance);
    }

    [Fact]
    public void TanOperatorDouble_Invoke_Vector512()
    {
        if (!Vector512.IsHardwareAccelerated)
        {
            return; // Skip test if hardware acceleration not available
        }

        var op = new TanOperatorDouble();
        var input = Vector512.Create(0.0, Math.PI / 4, -Math.PI / 4, Math.PI / 6, -Math.PI / 6, 0.5, -0.5, 1.0);
        var result = op.Invoke(input);

        Span<double> resultValues = stackalloc double[Vector512<double>.Count];
        result.CopyTo(resultValues);

        for (int i = 0; i < Vector512<double>.Count; i++)
        {
            double expectedValue = Math.Tan(Vector512.GetElement(input, i));
            Assert.Equal(expectedValue, resultValues[i], SimdDoubleTolerance);
        }
    }
#endif

    #endregion

    #region TanOperatorFloat - Scalar Tests

    [Theory]
    [InlineData(0.0f, 0.0f)]
    [InlineData(MathF.PI / 4, 1.0f)]
    [InlineData(-MathF.PI / 4, -1.0f)]
    [InlineData(MathF.PI / 6, 0.57735026f)]  // tan(30°) ≈ 1/√3
    public void TanOperatorFloat_Invoke_Scalar_KnownValues(float input, float expected)
    {
        var op = new TanOperatorFloat();
        float result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarFloatTolerance);
    }

    [Fact]
    public void TanOperatorFloat_Invoke_Scalar_Zero()
    {
        var op = new TanOperatorFloat();
        float result = op.Invoke(0.0f);
        Assert.Equal(0.0f, result, ScalarFloatTolerance);
    }

    [Fact]
    public void TanOperatorFloat_Invoke_Scalar_Negative()
    {
        var op = new TanOperatorFloat();
        float input = -0.5f;
        float expected = MathF.Tan(input);
        float result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarFloatTolerance);
    }

    [Fact]
    public void TanOperatorFloat_Invoke_Scalar_LargeValue()
    {
        var op = new TanOperatorFloat();
        float input = 100.0f;
        float expected = MathF.Tan(input);
        float result = op.Invoke(input);
        Assert.Equal(expected, result, ScalarFloatTolerance);
    }

    #endregion

    #region TanOperatorFloat - SIMD Tests

#if NET5_0_OR_GREATER
    [Fact]
    public void TanOperatorFloat_Invoke_Vector128()
    {
        if (!Vector128.IsHardwareAccelerated)
        {
            return; // Skip test if hardware acceleration not available
        }

        var op = new TanOperatorFloat();
        var input = Vector128.Create(0.0f, MathF.PI / 4, -MathF.PI / 4, MathF.PI / 6);
        var result = op.Invoke(input);

        Span<float> resultValues = stackalloc float[Vector128<float>.Count];
        result.CopyTo(resultValues);

        Assert.Equal(MathF.Tan(0.0f), resultValues[0], SimdFloatTolerance);
        Assert.Equal(MathF.Tan(MathF.PI / 4), resultValues[1], SimdFloatTolerance);
        Assert.Equal(MathF.Tan(-MathF.PI / 4), resultValues[2], SimdFloatTolerance);
        Assert.Equal(MathF.Tan(MathF.PI / 6), resultValues[3], SimdFloatTolerance);
    }

    [Fact]
    public void TanOperatorFloat_Invoke_Vector256()
    {
        if (!Vector256.IsHardwareAccelerated)
        {
            return; // Skip test if hardware acceleration not available
        }

        var op = new TanOperatorFloat();
        var input = Vector256.Create(0.0f, MathF.PI / 4, -MathF.PI / 4, MathF.PI / 6, -MathF.PI / 6, 0.5f, -0.5f, 1.0f);
        var result = op.Invoke(input);

        Span<float> resultValues = stackalloc float[Vector256<float>.Count];
        result.CopyTo(resultValues);

        for (int i = 0; i < Vector256<float>.Count; i++)
        {
            float expectedValue = MathF.Tan(Vector256.GetElement(input, i));
            Assert.Equal(expectedValue, resultValues[i], SimdFloatTolerance);
        }
    }

    [Fact]
    public void TanOperatorFloat_Invoke_Vector512()
    {
        if (!Vector512.IsHardwareAccelerated)
        {
            return; // Skip test if hardware acceleration not available
        }

        var op = new TanOperatorFloat();
        var input = Vector512.Create(0.0f, MathF.PI / 4, -MathF.PI / 4, MathF.PI / 6, -MathF.PI / 6, 0.5f, -0.5f, 1.0f,
                                      MathF.PI / 3, -MathF.PI / 3, 0.25f, -0.25f, 2.0f, -2.0f, 0.1f, -0.1f);
        var result = op.Invoke(input);

        Span<float> resultValues = stackalloc float[Vector512<float>.Count];
        result.CopyTo(resultValues);

        for (int i = 0; i < Vector512<float>.Count; i++)
        {
            float expectedValue = MathF.Tan(Vector512.GetElement(input, i));
            Assert.Equal(expectedValue, resultValues[i], SimdFloatTolerance);
        }
    }
#endif

    #endregion
}
