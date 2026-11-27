using System;
using AiDotNet.Models.Options;
using AiDotNet.NeuralRadianceFields.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralRadianceFields;

/// <summary>
/// The initial scale of each Gaussian is its nearest-neighbour distance found through a grid hash, which is O(N).
/// #2524 rewrote it as a chunked all-pairs engine computation (O(N^2) time and pair buffers) and the census saw its
/// first construction go 16.8 -> 145.6 ms, almost all of it JIT for the forty engine ops it used once.
/// </summary>
public class GaussianSplattingInitialScaleTests
{
    [Fact]
    public void DefaultCube_EveryCornerTakesHalfTheMeanSpacing()
    {
        var model = new GaussianSplatting<float>();

        var gaussians = model.GetGaussianDiagnostics();

        Assert.Equal(8, gaussians.Count);
        for (int i = 0; i < 8; i++)
        {
            // Corner i has x = bit 2, y = bit 1, z = bit 0 of i, at +-0.5. The mean spacing is (1 / 8)^(1/3) = 0.5,
            // so cells are 0.5 wide and neighbouring corners (1.0 apart) sit two cells away, outside the 3x3x3
            // search block: every corner falls back to the mean spacing, times InitialScaleMultiplier 0.5.
            Assert.Equal(((i >> 2) & 1) - 0.5, gaussians[i].PosX, 6);
            Assert.Equal(((i >> 1) & 1) - 0.5, gaussians[i].PosY, 6);
            Assert.Equal((i & 1) - 0.5, gaussians[i].PosZ, 6);
            Assert.Equal(0.25, gaussians[i].MaxScale, 5);
        }
    }

    [Fact]
    public void CoincidentPoints_FallBackToTheMeanSpacing()
    {
        var cloud = new Matrix<float>(3, 3);
        cloud[2, 0] = 4f;   // two points at the origin, one at (4, 0, 0): a flat cloud, spacing 4 / sqrt(3)
        var options = new GaussianSplattingOptions();
        var model = new GaussianSplatting<float>(options, cloud);

        var gaussians = model.GetGaussianDiagnostics();

        // The two coincident points are 0 apart, which is no scale, so they take the mean spacing; the third point's
        // nearest neighbour is 4 away, in the adjacent grid cell (cells are one spacing wide).
        double spacing = 4.0 / Math.Sqrt(3.0);
        Assert.Equal(spacing * options.InitialScaleMultiplier, gaussians[0].MaxScale, 4);
        Assert.Equal(spacing * options.InitialScaleMultiplier, gaussians[1].MaxScale, 4);
        Assert.Equal(4.0 * options.InitialScaleMultiplier, gaussians[2].MaxScale, 4);
    }

    [SkippableFact(Timeout = 120000)]
    public async System.Threading.Tasks.Task LargeCloud_InitializesInLinearMemory()
    {
#if !NETCOREAPP3_0_OR_GREATER
        // net471 has no per-thread allocation counter, and a process-wide one counts parallel tests too.
        Skip.If(true, "GC.GetAllocatedBytesForCurrentThread needs .NET Core 3.0 or later.");
#else
        await System.Threading.Tasks.Task.Yield();
        const int points = 20000;
        var random = new Random(3);
        var cloud = new Matrix<float>(points, 3);
        for (int i = 0; i < points; i++)
        {
            cloud[i, 0] = (float)random.NextDouble();
            cloud[i, 1] = (float)random.NextDouble();
            cloud[i, 2] = (float)random.NextDouble();
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        var model = new GaussianSplatting<float>(new GaussianSplattingOptions(), cloud);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.Equal(points, model.GaussianCount);
        // About 9 MB: the per-point tables and the grid. The all-pairs search materialized [rows, N, 3] blocks of
        // pair offsets, gigabytes in total for 20000 points.
        Assert.True(allocated < 64L * 1024 * 1024, $"allocated {allocated / (1024.0 * 1024.0):F1} MB");
#endif
    }
}
