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
            Assert.Equal(Bits(expectedScaled[i]), Bits(values[i]));
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

    [Fact]
    public void Float_SumOfSquares_WidensBeforeSquaring_SoLargeMagnitudesDoNotOverflow()
    {
        // (1e30f)^2 is 1e60, far past float's 3.4e38: squaring in float would give +Infinity and the
        // global norm would clip every update to nothing. 33 elements reach the vector body and a tail.
        var values = new float[33];
        for (int i = 0; i < values.Length; i++) values[i] = 1e30f;

        double sum = AdamOptimizer<float, Tensor<float>, Tensor<float>>.SumOfSquares(values.AsSpan());

        double expected = 33 * (double)1e30f * (double)1e30f;
        Assert.False(double.IsInfinity(sum));
        Assert.Equal(expected, sum, 1e-9 * expected);
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    public void Float_SumOfSquares_PropagatesANonFiniteElement(float bad)
    {
        // The caller's non-finite check depends on a NaN or infinity reaching the sum, wherever it sits.
        foreach (int position in new[] { 0, 16, 32 })
        {
            var values = RandomFloats(33, seed: position + 3);
            values[position] = bad;

            double sum = AdamOptimizer<float, Tensor<float>, Tensor<float>>.SumOfSquares(values.AsSpan());

            Assert.True(double.IsNaN(sum) || double.IsInfinity(sum), $"position {position} gave {sum}");
        }
    }

    [Fact]
    public void Generic_ElementType_UsesTheScalarDefinition()
    {
        // decimal takes neither vectorized path, so this covers the numeric-operations loop that every
        // other element type -- and float/double on net471 -- runs.
        var values = new decimal[] { -1.5m, 0.25m, 2m, -0.125m, 3m };
        double expectedSum = 0;
        foreach (var v in values) expectedSum += (double)v * (double)v;
        const double scale = 0.5;

        double sum = AdamOptimizer<decimal, Tensor<decimal>, Tensor<decimal>>.SumOfSquares(values.AsSpan());
        Assert.Equal(expectedSum, sum, 12);

        AdamOptimizer<decimal, Tensor<decimal>, Tensor<decimal>>.ScaleInPlace(values.AsSpan(), scale);
        Assert.Equal(new[] { -0.75m, 0.125m, 1m, -0.0625m, 1.5m }, values);
    }

    // BitConverter.SingleToInt32Bits is not available on net471, which this project also targets.
    private static int Bits(float value) => BitConverter.ToInt32(BitConverter.GetBytes(value), 0);

    private static float[] RandomFloats(int length, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var values = new float[length];
        for (int i = 0; i < length; i++) values[i] = (float)(rng.NextDouble() * 4 - 2);
        return values;
    }
}
