using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Helpers;

/// <summary>
/// OrderedTopKSelection must choose what a stable descending sort chooses (larger first, ties keep the earlier
/// entry), slot by slot, on the CPU engine and on a GPU engine.
/// </summary>
public class OrderedTopKSelectionTests
{
    // Rows with ties (row 0: three 2s and two 5s), negatives, and a constant row (every entry tied).
    private static readonly float[][] Rows =
    {
        new[] { 2f, 5f, 2f, -1f, 5f, 2f, 0f },
        new[] { -3f, -0.5f, -7f, 4f, 4f, -0.5f, 1f },
        new[] { 1f, 1f, 1f, 1f, 1f, 1f, 1f },
    };

    private static int[][] Expected(int k)
        => Rows.Select(r => Enumerable.Range(0, r.Length).OrderByDescending(i => r[i]).Take(k).ToArray()).ToArray();

    private static void AssertSelects(IEngine engine, int k)
    {
        int n = Rows[0].Length;
        var scores = new Tensor<float>(Rows.SelectMany(r => r).ToArray(), new[] { Rows.Length, n });
        var oneHot = OrderedTopKSelection<float>.Select(engine, scores, k).ToArray();
        var expected = Expected(k);
        Assert.Equal(Rows.Length * k * n, oneHot.Length);
        for (int r = 0; r < Rows.Length; r++)
            for (int u = 0; u < k; u++)
                for (int i = 0; i < n; i++)
                    Assert.True(oneHot[(r * k + u) * n + i] == (expected[r][u] == i ? 1f : 0f),
                        $"row {r} slot {u}: expected entry {expected[r][u]}, got one-hot {string.Join(",", Enumerable.Range(0, n).Select(j => oneHot[(r * k + u) * n + j]))}");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(7)]
    public void Cpu_selection_matches_a_stable_descending_sort(int k) => AssertSelects(new CpuEngine(), k);

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(4)]
    public void Gpu_selection_matches_a_stable_descending_sort(int k)
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException or InvalidOperationException) { }
        try
        {
            Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU backend did not resolve.");
            AssertSelects(gpu!, k);
        }
        finally { gpu?.Dispose(); }
    }
}
