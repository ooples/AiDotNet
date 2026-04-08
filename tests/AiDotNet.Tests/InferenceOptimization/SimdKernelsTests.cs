using System;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.InferenceOptimization;

public class SimdKernelsTests
{
    [Fact(Timeout = 60000)]
    public async Task VectorAdd_MatchesScalar()
    {
        await Task.Run(() =>
        {
            var a = CreateInput(32, 1);
            var b = CreateInput(32, 17);
            var result = new float[a.Length];
            var expected = new float[a.Length];

            for (int i = 0; i < a.Length; i++)
            {
                expected[i] = a[i] + b[i];
            }

            SimdKernels.VectorAdd(a, b, result);

            AssertEqual(expected, result);
        });
    }

    [Fact(Timeout = 60000)]
    public async Task VectorMultiply_MatchesScalar()
    {
        await Task.Run(() =>
        {
            var a = CreateInput(32, 3);
            var b = CreateInput(32, 9);
            var result = new float[a.Length];
            var expected = new float[a.Length];

            for (int i = 0; i < a.Length; i++)
            {
                expected[i] = a[i] * b[i];
            }

            SimdKernels.VectorMultiply(a, b, result);

            AssertEqual(expected, result);
        });
    }

    [Fact(Timeout = 60000)]
    public async Task DotProduct_MatchesScalar()
    {
        await Task.Run(() =>
        {
            var a = CreateInput(37, 5);
            var b = CreateInput(37, 11);

            float expected = 0f;
            for (int i = 0; i < a.Length; i++)
            {
                expected += a[i] * b[i];
            }

            float actual = SimdKernels.DotProduct(a, b);
            Assert.Equal(expected, actual, 5);
        });
    }

    [Fact(Timeout = 60000)]
    public async Task ScalarMultiplyAdd_MatchesScalar()
    {
        await Task.Run(() =>
        {
            var a = CreateInput(31, 7);
            var b = CreateInput(31, 13);
            var result = new float[a.Length];
            var expected = new float[a.Length];

            float scalar = 0.25f;
            for (int i = 0; i < a.Length; i++)
            {
                expected[i] = a[i] + scalar * b[i];
            }

            SimdKernels.ScalarMultiplyAdd(a, b, scalar, result);

            AssertEqual(expected, result);
        });
    }

    [Fact(Timeout = 60000)]
    public async Task ReLU_MatchesScalar()
    {
        await Task.Run(() =>
        {
            var input = CreateSignedInput(33);
            var output = new float[input.Length];
            var expected = new float[input.Length];

            for (int i = 0; i < input.Length; i++)
            {
                expected[i] = Math.Max(0f, input[i]);
            }

            SimdKernels.ReLU(input, output);

            AssertEqual(expected, output);
        });
    }

    [Fact(Timeout = 60000)]
    public async Task Sum_MatchesScalar()
    {
        await Task.Run(() =>
        {
            var input = CreateInput(100, 23);
            float expected = 0f;
            for (int i = 0; i < input.Length; i++)
            {
                expected += input[i];
            }

            float actual = SimdKernels.Sum(input);
            Assert.Equal(expected, actual, 5);
        });
    }

    private static float[] CreateInput(int length, int seed)
    {
        var data = new float[length];
        for (int i = 0; i < length; i++)
        {
            data[i] = DeterministicValue(i + seed);
        }

        return data;
    }

    private static float[] CreateSignedInput(int length)
    {
        var data = new float[length];
        for (int i = 0; i < length; i++)
        {
            float v = DeterministicValue(i);
            data[i] = (i % 2 == 0) ? v : -v;
        }

        return data;
    }

    private static float DeterministicValue(int i)
    {
        unchecked
        {
            uint x = (uint)(i * 1664525 + 1013904223);
            return (x & 0x00FFFFFF) / 16777216f;
        }
    }

    private static void AssertEqual(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i], 5);
        }
    }
}
