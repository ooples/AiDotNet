using System;
using AiDotNet.Tensors.Operators;
using Xunit;

namespace AiDotNet.Tensors.Tests.Operators;

public class TruncateOperatorTests
{
    private const double DoubleTolerance = 1e-14;
    private const float FloatTolerance = 1e-6f;

    [Theory]
    [InlineData(0.0, 0.0)]
    [InlineData(1.0, 1.0)]
    [InlineData(-1.0, -1.0)]
    [InlineData(5.5, 5.0)]
    [InlineData(5.9, 5.0)]
    [InlineData(-5.5, -5.0)]
    [InlineData(-5.9, -5.0)]
    [InlineData(100.3, 100.0)]
    [InlineData(-100.7, -100.0)]
    public void TruncateOperatorDouble_Invoke_Scalar_KnownValues(double input, double expected)
    {
        var op = new TruncateOperatorDouble();
        double result = op.Invoke(input);
        Assert.Equal(expected, result, DoubleTolerance);
    }

    [Fact]
    public void TruncateOperatorDouble_Invoke_Scalar_Zero()
    {
        var op = new TruncateOperatorDouble();
        double result = op.Invoke(0.0);
        Assert.Equal(0.0, result, DoubleTolerance);
    }

    [Theory]
    [InlineData(0.0f, 0.0f)]
    [InlineData(1.0f, 1.0f)]
    [InlineData(-1.0f, -1.0f)]
    [InlineData(5.5f, 5.0f)]
    [InlineData(5.9f, 5.0f)]
    [InlineData(-5.5f, -5.0f)]
    [InlineData(-5.9f, -5.0f)]
    public void TruncateOperatorFloat_Invoke_Scalar_KnownValues(float input, float expected)
    {
        var op = new TruncateOperatorFloat();
        float result = op.Invoke(input);
        Assert.Equal(expected, result, FloatTolerance);
    }

    [Fact]
    public void TruncateOperatorFloat_Invoke_Scalar_Zero()
    {
        var op = new TruncateOperatorFloat();
        float result = op.Invoke(0.0f);
        Assert.Equal(0.0f, result, FloatTolerance);
    }
}
