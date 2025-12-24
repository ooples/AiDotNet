using System;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

public class SimdKernelsTests
{
    [Fact]
    public void VectorAdd_MatchesScalar()
    {
        var a = new float[] { 1, 2, 3, 4, 5 };
        var b = new float[] { 10, 20, 30, 40, 50 };
        var result = new float[a.Length];

        SimdKernels.VectorAdd(a, b, result);

        for (int i = 0; i < a.Length; i++)
        {
            Assert.Equal(a[i] + b[i], result[i]);
        }
    }

    [Fact]
    public void VectorMultiply_MatchesScalar()
    {
        var a = new float[] { 1, 2, 3, 4, 5 };
        var b = new float[] { 10, 20, 30, 40, 50 };
        var result = new float[a.Length];

        SimdKernels.VectorMultiply(a, b, result);

        for (int i = 0; i < a.Length; i++)
        {
            Assert.Equal(a[i] * b[i], result[i]);
        }
    }

    [Fact]
    public void DotProduct_MatchesScalar()
    {
        var a = new float[] { 1, 2, 3, 4 };
        var b = new float[] { 10, 20, 30, 40 };

        float dot = SimdKernels.DotProduct(a, b);

        Assert.Equal(1 * 10 + 2 * 20 + 3 * 30 + 4 * 40, dot);
    }

    [Fact]
    public void ScalarMultiplyAdd_MatchesScalar()
    {
        var a = new float[] { 1, 2, 3, 4 };
        var b = new float[] { 10, 20, 30, 40 };
        var result = new float[a.Length];

        SimdKernels.ScalarMultiplyAdd(a, b, scalar: 0.5f, result);

        for (int i = 0; i < a.Length; i++)
        {
            Assert.Equal(a[i] + (b[i] * 0.5f), result[i]);
        }
    }

    [Fact]
    public void ReLU_ZeroesNegatives()
    {
        var input = new float[] { -1, 0, 2, -3, 4 };
        var output = new float[input.Length];

        SimdKernels.ReLU(input, output);

        Assert.Equal(new float[] { 0, 0, 2, 0, 4 }, output);
    }

    [Fact]
    public void Exp_MatchesMathExpWithinTolerance()
    {
        var input = new float[] { -1f, 0f, 1f };
        var output = new float[input.Length];

        SimdKernels.Exp(input, output);

        for (int i = 0; i < input.Length; i++)
        {
            Assert.InRange(output[i], (float)Math.Exp(input[i]) * 0.999f, (float)Math.Exp(input[i]) * 1.001f);
        }
    }

    [Fact]
    public void Sum_MatchesScalar()
    {
        var input = new float[] { 1, 2, 3, 4, 5 };

        float sum = SimdKernels.Sum(input);

        Assert.Equal(15f, sum);
    }

    [Fact]
    public void Floor_MatchesMathFloor()
    {
        var input = new float[] { 1.5f, 2.3f, -1.7f, 3.9f, -0.5f };
        var output = new float[input.Length];

        SimdKernels.Floor(input, output);

        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal((float)Math.Floor(input[i]), output[i]);
        }
    }

    [Fact]
    public void Ceiling_MatchesMathCeiling()
    {
        var input = new float[] { 1.5f, 2.3f, -1.7f, 3.9f, -0.5f };
        var output = new float[input.Length];

        SimdKernels.Ceiling(input, output);

        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal((float)Math.Ceiling(input[i]), output[i]);
        }
    }

    [Fact]
    public void Frac_ReturnsFractionalPart()
    {
        var input = new float[] { 1.5f, 2.3f, 3.9f };
        var output = new float[input.Length];

        SimdKernels.Frac(input, output);

        for (int i = 0; i < input.Length; i++)
        {
            float expected = input[i] - (float)Math.Floor(input[i]);
            Assert.Equal(expected, output[i], precision: 5);
        }
    }

    [Fact]
    public void Sin_MatchesMathSin()
    {
        var input = new float[] { 0f, (float)(Math.PI / 2), (float)Math.PI, (float)(3 * Math.PI / 2) };
        var output = new float[input.Length];

        SimdKernels.Sin(input, output);

        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal((float)Math.Sin(input[i]), output[i], precision: 5);
        }
    }

    [Fact]
    public void Cos_MatchesMathCos()
    {
        var input = new float[] { 0f, (float)(Math.PI / 2), (float)Math.PI, (float)(3 * Math.PI / 2) };
        var output = new float[input.Length];

        SimdKernels.Cos(input, output);

        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal((float)Math.Cos(input[i]), output[i], precision: 5);
        }
    }

    [Fact]
    public void FloorDouble_MatchesMathFloor()
    {
        var input = new double[] { 1.5, 2.3, -1.7, 3.9, -0.5 };
        var output = new double[input.Length];

        SimdKernels.Floor(input, output);

        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal(Math.Floor(input[i]), output[i]);
        }
    }

    [Fact]
    public void CeilingDouble_MatchesMathCeiling()
    {
        var input = new double[] { 1.5, 2.3, -1.7, 3.9, -0.5 };
        var output = new double[input.Length];

        SimdKernels.Ceiling(input, output);

        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal(Math.Ceiling(input[i]), output[i]);
        }
    }
}

