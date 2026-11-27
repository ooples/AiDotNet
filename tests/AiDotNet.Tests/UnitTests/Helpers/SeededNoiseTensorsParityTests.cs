using System;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Helpers;

/// <summary>
/// Pins <see cref="SeededNoiseTensors.StandardNormal{T}"/> to the per-element loop the video and text-to-speech
/// models each carried: <c>t[i] = NumOps.FromDouble(scale * sqrt(-2 ln(1 - u1)) * cos(2 pi u2))</c>.
/// </summary>
public class SeededNoiseTensorsParityTests
{
    private const int Seed = 20490;

    private static Tensor<T> Reference<T>(int[] shape, Random random, double scale)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var t = new Tensor<T>(shape);
        for (int i = 0; i < t.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            t[i] = ops.FromDouble(scale * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }

        return t;
    }

    private static Tensor<T> ReferenceUnscaled<T>(int[] shape, Random random)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var t = new Tensor<T>(shape);
        for (int i = 0; i < t.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            t[i] = ops.FromDouble(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }

        return t;
    }

    private static void Matches<T>(int[] shape, double scale)
    {
        var expectedRandom = RandomHelper.CreateSeededRandom(Seed);
        var actualRandom = RandomHelper.CreateSeededRandom(Seed);

        var expected = Reference<T>(shape, expectedRandom, scale);
        var actual = SeededNoiseTensors.StandardNormal<T>(shape, actualRandom, scale);

        Assert.Equal(shape, actual.Shape.ToArray());
        Assert.Equal(expected.ToArray(), actual.ToArray());
        // The helper took exactly as many uniforms as the loop: the next draw of both generators agrees.
        Assert.Equal(expectedRandom.NextDouble(), actualRandom.NextDouble());
    }

    [Theory]
    [InlineData(1.0)]
    [InlineData(0.667)]
    [InlineData(0.0)]
    [InlineData(3.5)]
    public void StandardNormal_MatchesTheElementLoop_Double(double scale)
    {
        Matches<double>(new[] { 7 }, scale);
        Matches<double>(new[] { 2, 3, 5 }, scale);
        Matches<double>(new[] { 1, 4, 3, 3 }, scale);
    }

    [Theory]
    [InlineData(1.0)]
    [InlineData(0.667)]
    [InlineData(0.0)]
    [InlineData(3.5)]
    public void StandardNormal_MatchesTheElementLoop_Float(double scale)
    {
        Matches<float>(new[] { 7 }, scale);
        Matches<float>(new[] { 2, 3, 5 }, scale);
        Matches<float>(new[] { 1, 4, 3, 3 }, scale);
    }

    [Fact]
    public void StandardNormal_DefaultScale_MatchesTheLoopWithoutAScaleFactor()
    {
        // The video models and two of the speech bases had no scale factor at all.
        var shape = new[] { 3, 4, 5 };

        Assert.Equal(
            ReferenceUnscaled<double>(shape, RandomHelper.CreateSeededRandom(Seed)).ToArray(),
            SeededNoiseTensors.StandardNormal<double>(shape, RandomHelper.CreateSeededRandom(Seed)).ToArray());
        Assert.Equal(
            ReferenceUnscaled<float>(shape, RandomHelper.CreateSeededRandom(Seed)).ToArray(),
            SeededNoiseTensors.StandardNormal<float>(shape, RandomHelper.CreateSeededRandom(Seed)).ToArray());
    }

    [Fact]
    public void StandardNormal_EmptyShape_ReturnsAnEmptyTensorAndDrawsNothing()
    {
        var used = RandomHelper.CreateSeededRandom(Seed);
        var untouched = RandomHelper.CreateSeededRandom(Seed);

        var tensor = SeededNoiseTensors.StandardNormal<double>(new[] { 2, 0, 3 }, used);

        Assert.Equal(0, tensor.Length);
        Assert.Equal(new[] { 2, 0, 3 }, tensor.Shape.ToArray());
        Assert.Equal(untouched.NextDouble(), used.NextDouble());
    }

    [Fact]
    public void StandardNormal_SameSeed_SameTensor_DifferentSeed_DifferentTensor()
    {
        var shape = new[] { 4, 4 };
        var first = SeededNoiseTensors.StandardNormal<double>(shape, RandomHelper.CreateSeededRandom(1));
        var again = SeededNoiseTensors.StandardNormal<double>(shape, RandomHelper.CreateSeededRandom(1));
        var other = SeededNoiseTensors.StandardNormal<double>(shape, RandomHelper.CreateSeededRandom(2));

        Assert.Equal(first.ToArray(), again.ToArray());
        Assert.NotEqual(first.ToArray(), other.ToArray());
    }
}
