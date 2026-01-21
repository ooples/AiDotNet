using System;
using AiDotNet.InferenceOptimization.Kernels;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.InferenceOptimization;

public class AttentionKernelValidationTests
{
    [Fact]
    public void Execute_MatchesNaiveAttention()
    {
        var kernel = new AttentionKernel();

        // [batch=1, seq=2, d=4]
        var q = CreateTensor(new[] { 1, 2, 4 }, new float[] { 1, 0, 0, 0, 0, 1, 0, 0 });
        var k = CreateTensor(new[] { 1, 2, 4 }, new float[] { 1, 0, 0, 0, 0, 1, 0, 0 });
        var v = CreateTensor(new[] { 1, 2, 4 }, new float[] { 10, 11, 12, 13, 20, 21, 22, 23 });

        var actual = kernel.Execute(q, k, v);
        var expected = NaiveAttention(q, k, v);

        Assert.Equal(expected.Shape, actual.Shape);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i], 5);
        }
    }

    [Fact]
    public void Execute_WithMask_RespectsMaskZeros()
    {
        var kernel = new AttentionKernel();

        // [batch=1, seq=2, d=2]
        var q = CreateTensor(new[] { 1, 2, 2 }, new float[] { 1, 0, 0, 1 });
        var k = CreateTensor(new[] { 1, 2, 2 }, new float[] { 1, 0, 0, 1 });
        var v = CreateTensor(new[] { 1, 2, 2 }, new float[] { 1, 2, 100, 200 });

        // Mask: allow only attending to token 0 (mask value 1), disallow token 1 (mask value 0)
        var mask = CreateTensor(new[] { 1, 2, 2 }, new float[] { 1, 0, 1, 0 });

        var actual = kernel.Execute(q, k, v, mask);

        // With token 1 masked out, both rows should match v0.
        Assert.Equal(1f, actual[0], 5);
        Assert.Equal(2f, actual[1], 5);
        Assert.Equal(1f, actual[2], 5);
        Assert.Equal(2f, actual[3], 5);
    }

    private static Tensor<float> NaiveAttention(Tensor<float> q, Tensor<float> k, Tensor<float> v)
    {
        int seq = q.Shape[1];
        int d = q.Shape[2];
        float scale = 1f / MathF.Sqrt(d);

        var scores = new float[seq * seq];
        for (int i = 0; i < seq; i++)
        {
            for (int j = 0; j < seq; j++)
            {
                float dot = 0f;
                for (int t = 0; t < d; t++)
                {
                    dot += q[i * d + t] * k[j * d + t];
                }

                scores[i * seq + j] = dot * scale;
            }
        }

        for (int i = 0; i < seq; i++)
        {
            float max = float.NegativeInfinity;
            for (int j = 0; j < seq; j++)
            {
                max = Math.Max(max, scores[i * seq + j]);
            }

            float sum = 0f;
            for (int j = 0; j < seq; j++)
            {
                scores[i * seq + j] = MathF.Exp(scores[i * seq + j] - max);
                sum += scores[i * seq + j];
            }

            for (int j = 0; j < seq; j++)
            {
                scores[i * seq + j] /= sum;
            }
        }

        var result = new Tensor<float>(new[] { 1, seq, v.Shape[2] });
        int dV = v.Shape[2];
        for (int i = 0; i < seq; i++)
        {
            for (int j = 0; j < dV; j++)
            {
                float sum = 0f;
                for (int t = 0; t < seq; t++)
                {
                    sum += scores[i * seq + t] * v[t * dV + j];
                }

                result[i * dV + j] = sum;
            }
        }

        return result;
    }

    private static Tensor<float> CreateTensor(int[] shape, float[] data)
    {
        var t = new Tensor<float>(shape);
        Assert.Equal(t.Length, data.Length);
        t.CopyFromArray(data);
        return t;
    }
}
