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
    public void PointCloudColors_MustHaveOneRgbRowPerPoint()
    {
        var cloud = new Matrix<float>(2, 3);
        cloud[1, 0] = 1f;

        // One colour row for two points: rejected before any initialization work.
        Assert.Throws<ArgumentException>(() => new GaussianSplatting<float>(new GaussianSplattingOptions(), cloud, new Matrix<float>(1, 3)));
        // A fourth colour column would be silently ignored.
        Assert.Throws<ArgumentException>(() => new GaussianSplatting<float>(new GaussianSplattingOptions(), cloud, new Matrix<float>(2, 4)));
    }

    [Fact]
    public void QueryField_RejectsDirectionRowCountMismatch()
    {
        var model = new GaussianSplatting<float>(new GaussianSplattingOptions { UseSphericalHarmonics = false }, new Matrix<float>(1, 3));

        Assert.Throws<ArgumentException>(() =>
            model.QueryField(new Tensor<float>(new[] { 2, 3 }), new Tensor<float>(new[] { 1, 3 })));
    }

    [Fact]
    public void CoincidentPoints_TakeTheDefaultSpacingScale()
    {
        // Three points at the origin: each has a neighbour at distance 0, so each falls back to the default spacing
        // (scale 0.025), and the three identical Gaussians add their densities.
        var model = new GaussianSplatting<float>(new GaussianSplattingOptions(), new Matrix<float>(3, 3));
        var positions = new Tensor<float>(new[] { 1, 3 });
        positions[0, 0] = 0.02f;
        var directions = new Tensor<float>(new[] { 1, 3 });
        directions[0, 2] = 1f;

        var (_, density) = model.QueryField(positions, directions);

        double variance = 0.025 * 0.025 + 1e-6;
        Assert.Equal(3 * 0.5 * Math.Exp(-0.5 * 0.0004 / variance), density[0, 0], 4);
    }

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task FullyCoincidentLargeCloud_SkipsTheNeighbourSearch()
    {
        await System.Threading.Tasks.Task.Yield();

        // 100,000 points in one grid bucket: the pairwise scan was ~10^10 distance evaluations (minutes); the
        // duplicate collapse makes it linear.
        var watch = System.Diagnostics.Stopwatch.StartNew();
        var model = new GaussianSplatting<float>(
            new GaussianSplattingOptions { UseSphericalHarmonics = false }, new Matrix<float>(100_000, 3));
        watch.Stop();

        Assert.Equal(100_000, model.GaussianCount);
        Assert.True(watch.Elapsed < TimeSpan.FromSeconds(30), $"construction took {watch.Elapsed.TotalSeconds:F1} s");
    }

    [Fact]
    public void SpatialIndex_CellMembershipUsesFloatRounding()
    {
        // Unit-scale Gaussians at x = 2e-7 and x = 2.4: cell width 2.5, radius 2. In float, the query at x = 7.5 is
        // (7.5f - 2e-7f) * 0.4f = 3 exactly, three cells from the second Gaussian (cell 0), so it is excluded; double
        // arithmetic gives cell 2 and would include it, turning the result red.
        var model = new GaussianSplatting<float>(new GaussianSplattingOptions { UseSphericalHarmonics = false }, new Matrix<float>(2, 3));
        var centres = new Tensor<float>(new[] { 2, 3 });
        centres[0, 0] = 2e-7f;
        centres[1, 0] = 2.4f;
        var rotations = new Tensor<float>(new[] { 2, 4 });
        rotations[0, 0] = 1f;
        rotations[1, 0] = 1f;
        var sh = new Tensor<float>(new[] { 2, 1, 3 });
        sh[1, 0, 0] = 1f;
        model.SetGaussianStateForTest(centres, rotations, new Tensor<float>(new[] { 2, 3 }), new Tensor<float>(new[] { 2 }), sh);
        var positions = new Tensor<float>(new[] { 1, 3 });
        positions[0, 0] = 7.5f;
        var directions = new Tensor<float>(new[] { 1, 3 });
        directions[0, 2] = 1f;

        var (rgb, density) = model.QueryField(positions, directions);

        Assert.Equal(0f, density[0, 0]);
        Assert.Equal(0f, rgb[0, 0]);

        // Without the index the second Gaussian's alpha (0.5 exp(-5.1^2 / 2) ~ 1.1e-6) clears the 1e-8 cutoff.
        model.EnableSpatialIndex = false;
        var (unindexedRgb, unindexedDensity) = model.QueryField(positions, directions);
        Assert.True(unindexedDensity[0, 0] > 1e-8f);
        Assert.Equal(1f, unindexedRgb[0, 0], 5);
    }

    [Fact]
    public void LargeQuery_MatchesPointByPointEvaluation()
    {
        // The query is evaluated in parallel blocks of points; every row must equal the same point queried alone.
        const int gaussians = 300;
        const int points = 1000;
        var random = new Random(11);
        var cloud = new Matrix<float>(gaussians, 3);
        for (int i = 0; i < gaussians; i++)
            for (int c = 0; c < 3; c++) cloud[i, c] = (float)random.NextDouble();
        var model = new GaussianSplatting<float>(new GaussianSplattingOptions(), cloud);
        var positions = new Tensor<float>(new[] { points, 3 });
        var directions = new Tensor<float>(new[] { points, 3 });
        for (int i = 0; i < points; i++)
        {
            for (int c = 0; c < 3; c++)
            {
                positions[i, c] = (float)random.NextDouble();
                directions[i, c] = (float)(random.NextDouble() - 0.5);
            }
        }

        var (rgb, density) = model.QueryField(positions, directions);

        for (int i = 0; i < points; i += 37)
        {
            var onePosition = new Tensor<float>(new[] { 1, 3 });
            var oneDirection = new Tensor<float>(new[] { 1, 3 });
            for (int c = 0; c < 3; c++)
            {
                onePosition[0, c] = positions[i, c];
                oneDirection[0, c] = directions[i, c];
            }

            var (oneRgb, oneDensity) = model.QueryField(onePosition, oneDirection);
            Assert.Equal(oneDensity[0, 0], density[i, 0]);
            for (int c = 0; c < 3; c++) Assert.Equal(oneRgb[0, c], rgb[i, c]);
        }
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

    [SkippableFact(Timeout = 120000)]
    public async System.Threading.Tasks.Task LargeQuery_AllocatesOnlyItsInputsAndOutputs()
    {
#if !NETCOREAPP3_0_OR_GREATER
        // net471 has no per-thread allocation counter, and a process-wide one counts parallel tests too.
        Skip.If(true, "GC.GetAllocatedBytesForCurrentThread needs .NET Core 3.0 or later.");
#else
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
#endif
    }
}
