using AiDotNet.InferenceOptimization.Kernels;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.InferenceOptimization;

public class GemmKernelValidationTests
{
    [Fact]
    public void Execute_MatchesNaiveGemm()
    {
        var kernel = new GemmKernel();

        // A: 2x3
        var a = CreateTensor(new[] { 2, 3 }, new float[] { 1, 2, 3, 4, 5, 6 });
        // B: 3x2
        var b = CreateTensor(new[] { 3, 2 }, new float[] { 7, 8, 9, 10, 11, 12 });

        var actual = kernel.Execute(a, b);
        var expected = NaiveGemm(a, b);

        Assert.Equal(expected.Shape, actual.Shape);
        Assert.Equal(expected.ToArray(), actual.ToArray());
    }

    [Fact]
    public void GemmTransposeB_MatchesNaive()
    {
        var kernel = new GemmKernel();

        // A: 2x3
        var a = CreateTensor(new[] { 2, 3 }, new float[] { 1, 2, 3, 4, 5, 6 });
        // B: 2x3 (represents B^T; result is 2x2)
        var b = CreateTensor(new[] { 2, 3 }, new float[] { 7, 8, 9, 10, 11, 12 });

        var actual = kernel.GemmTransposeB(a, b);
        var expected = NaiveGemmTransposeB(a, b);

        Assert.Equal(expected.Shape, actual.Shape);
        Assert.Equal(expected.ToArray(), actual.ToArray());
    }

    private static Tensor<float> NaiveGemm(Tensor<float> a, Tensor<float> b)
    {
        int m = a.Shape[0];
        int k = a.Shape[1];
        int n = b.Shape[1];

        var c = new Tensor<float>(new[] { m, n });

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                float sum = 0f;
                for (int t = 0; t < k; t++)
                {
                    sum += a[i * k + t] * b[t * n + j];
                }

                c[i * n + j] = sum;
            }
        }

        return c;
    }

    private static Tensor<float> NaiveGemmTransposeB(Tensor<float> a, Tensor<float> b)
    {
        int m = a.Shape[0];
        int k = a.Shape[1];
        int n = b.Shape[0];

        var c = new Tensor<float>(new[] { m, n });

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                float sum = 0f;
                for (int t = 0; t < k; t++)
                {
                    sum += a[i * k + t] * b[j * k + t];
                }

                c[i * n + j] = sum;
            }
        }

        return c;
    }

    private static Tensor<float> CreateTensor(int[] shape, float[] data)
    {
        var t = new Tensor<float>(shape);
        Assert.Equal(t.Length, data.Length);
        t.CopyFromArray(data);
        return t;
    }
}
