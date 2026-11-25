using System;
using AiDotNet.Models.Options;
using AiDotNet.NeuralRadianceFields.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralRadianceFields;

/// <summary>
/// The field is evaluated per point on the host. #2524 had written it as about seventy engine ops over [points,
/// Gaussians] blocks: at the census size every op was dispatch overhead (steady forward 0.1 -> 1.0 ms, cold forward
/// 23 -> 76 ms of first-use JIT), and a large query materialized several [points, Gaussians] temporaries per op.
/// </summary>
public class GaussianSplattingFieldEvaluationTests
{
    [Fact]
    public void SingleGaussian_DensityIsOpacityTimesTheGaussianFalloff()
    {
        var cloud = new Matrix<float>(1, 3);
        var model = new GaussianSplatting<float>(new GaussianSplattingOptions(), cloud);
        var positions = new Tensor<float>(new[] { 2, 3 });
        positions[0, 0] = 0.02f;   // 0.02 along x
        positions[1, 1] = 0.5f;    // far outside
        var directions = new Tensor<float>(new[] { 2, 3 });
        directions[0, 2] = 1f;
        directions[1, 2] = 1f;

        var (_, density) = model.QueryField(positions, directions);

        // A lone point takes the default spacing 0.05, so scale 0.025 on every axis; opacity logit 0 is 0.5.
        double variance = 0.025 * 0.025 + 1e-6;
        Assert.Equal(0.5 * Math.Exp(-0.5 * 0.0004 / variance), density[0, 0], 4);
        Assert.Equal(0.0, density[1, 0], 6);
    }

    [Fact]
    public void ForwardRows_MatchQueryField()
    {
        var model = new GaussianSplatting<float>();
        var random = new Random(5);
        var input = new Tensor<float>(new[] { 16, 6 });
        var positions = new Tensor<float>(new[] { 16, 3 });
        var directions = new Tensor<float>(new[] { 16, 3 });
        for (int i = 0; i < 16; i++)
        {
            for (int c = 0; c < 6; c++)
            {
                float value = (float)(random.NextDouble() - 0.5);
                input[i, c] = value;
                if (c < 3) positions[i, c] = value; else directions[i, c - 3] = value;
            }
        }

        var forward = model.Predict(input);
        var (rgb, density) = model.QueryField(positions, directions);

        for (int i = 0; i < 16; i++)
        {
            for (int c = 0; c < 3; c++) Assert.Equal(rgb[i, c], forward[i, c]);
            Assert.Equal(density[i, 0], forward[i, 3]);
        }
    }

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task LargeQuery_AllocatesOnlyItsInputsAndOutputs()
    {
        await System.Threading.Tasks.Task.Yield();
        const int gaussians = 2000;
        const int points = 500;
        var random = new Random(9);
        var cloud = new Matrix<float>(gaussians, 3);
        for (int i = 0; i < gaussians; i++)
        {
            for (int c = 0; c < 3; c++) cloud[i, c] = (float)random.NextDouble();
        }

        var model = new GaussianSplatting<float>(new GaussianSplattingOptions(), cloud);
        var positions = new Tensor<float>(new[] { points, 3 });
        var directions = new Tensor<float>(new[] { points, 3 });
        for (int i = 0; i < points; i++)
        {
            for (int c = 0; c < 3; c++) positions[i, c] = (float)random.NextDouble();
            directions[i, 2] = 1f;
        }

        model.QueryField(positions, directions);   // first use: one-time setup and JIT
        long before = GC.GetAllocatedBytesForCurrentThread();
        model.QueryField(positions, directions);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        // The per-Gaussian tables are about 0.5 MB. One [points, Gaussians] float temporary alone is 4 MB, and the
        // tensor form made dozens per query.
        Assert.True(allocated < 4L * 1024 * 1024, $"allocated {allocated / (1024.0 * 1024.0):F1} MB");
    }
}
