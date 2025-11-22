using System;
using AiDotNet.Tensors.Operators;
using Xunit;

namespace AiDotNet.Tensors.Tests.Operators;

public class CeilingOperatorTests
{
    private const double DoubleTolerance = 1e-14;
    private const float FloatTolerance = 1e-6f;

    [Theory]
    [InlineData(0.0, 0.0)]
    [InlineData(1.0, 1.0)]
    [InlineData(-1.0, -1.0)]
    [InlineData(5.5, 6.0)]
    [InlineData(5.1, 6.0)]
    [InlineData(-5.5, -5.0)]
    [InlineData(-5.9, -5.0)]
    [InlineData(100.3, 101.0)]
    [InlineData(-100.7, -100.0)]
    public void CeilingOperatorDouble_Invoke_Scalar_KnownValues(double input, double expected)
    {
        var op = new CeilingOperatorDouble();
        double result = op.Invoke(input);
        Assert.Equal(expected, result, DoubleTolerance);
    }

    [Fact]
    public void CeilingOperatorDouble_Invoke_Scalar_Zero()
    {
        var op = new CeilingOperatorDouble();
        double result = op.Invoke(0.0);
        Assert.Equal(0.0, result, DoubleTolerance);
    }

    [Theory]
    [InlineData(0.0f, 0.0f)]
    [InlineData(1.0f, 1.0f)]
    [InlineData(-1.0f, -1.0f)]
    [InlineData(5.5f, 6.0f)]
    [InlineData(5.1f, 6.0f)]
    [InlineData(-5.5f, -5.0f)]
    [InlineData(-5.9f, -5.0f)]
    public void CeilingOperatorFloat_Invoke_Scalar_KnownValues(float input, float expected)
    {
        var op = new CeilingOperatorFloat();
        float result = op.Invoke(input);
        Assert.Equal(expected, result, FloatTolerance);
    }

    [Fact]
    public void CeilingOperatorFloat_Invoke_Scalar_Zero()
    {
        var op = new CeilingOperatorFloat();
        float result = op.Invoke(0.0f);
        Assert.Equal(0.0f, result, FloatTolerance);
    }
}
