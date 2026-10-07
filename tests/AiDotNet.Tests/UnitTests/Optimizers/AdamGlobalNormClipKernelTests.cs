using System;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// #2087: global-norm clipping ran per element through the numeric-operations interface, 1.75 s of a 14 s training
/// step on a 477.6M-parameter U-Net. The vectorized float/double path must keep the arithmetic: squares summed in
/// double, and each element scaled as (T)((double)x * scale).
/// </summary>
public class AdamGlobalNormClipKernelTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(33)]
    [InlineData(1000)]
    public void Float_ScaleMatchesTheScalarDefinitionExactly_AndTheSumMatchesClosely(int length)
    {
        var values = RandomFloats(length, seed: length + 1);
        var expectedScaled = new float[length];
        double expectedSum = 0;
        const double scale = 0.3141592653589793;
        for (int i = 0; i < length; i++)
        {
            double v = values[i];
            expectedSum += v * v;
            expectedScaled[i] = (float)(values[i] * scale);
        }

        double sum = AdamOptimizer<float, Tensor<float>, Tensor<float>>.SumOfSquares(values.AsSpan());
        Assert.Equal(expectedSum, sum, 1e-9 * Math.Max(1.0, expectedSum));

        AdamOptimizer<float, Tensor<float>, Tensor<float>>.ScaleInPlace(values.AsSpan(), scale);
        for (int i = 0; i < length; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits(expectedScaled[i]), BitConverter.SingleToInt32Bits(values[i]));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(3)]
    [InlineData(17)]
    [InlineData(1001)]
    public void Double_ScaleMatchesTheScalarDefinitionExactly_AndTheSumMatchesClosely(int length)
    {
        var rng = RandomHelper.CreateSeededRandom(length + 2);
        var values = new double[length];
        for (int i = 0; i < length; i++) values[i] = rng.NextDouble() * 4 - 2;
        var expectedScaled = new double[length];
        double expectedSum = 0;
        const double scale = 1.7320508075688772;
        for (int i = 0; i < length; i++)
        {
            expectedSum += values[i] * values[i];
            expectedScaled[i] = values[i] * scale;
        }

        double sum = AdamOptimizer<double, Tensor<double>, Tensor<double>>.SumOfSquares(values.AsSpan());
        Assert.Equal(expectedSum, sum, 1e-12 * Math.Max(1.0, expectedSum));

        AdamOptimizer<double, Tensor<double>, Tensor<double>>.ScaleInPlace(values.AsSpan(), scale);
        for (int i = 0; i < length; i++)
            Assert.Equal(BitConverter.DoubleToInt64Bits(expectedScaled[i]), BitConverter.DoubleToInt64Bits(values[i]));
    }

    private static float[] RandomFloats(int length, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var values = new float[length];
        for (int i = 0; i < length; i++) values[i] = (float)(rng.NextDouble() * 4 - 2);
        return values;
    }
}
