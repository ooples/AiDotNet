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
    [Fact]
    public void BFloat16_rank_257_is_a_distinct_one_hot_slot_and_replays_changed_scores()
    {
        var engine = new CpuEngine();
        var scores = new Tensor<AiDotNet.Tensors.NumericOperations.BFloat16>(new[] { 1, 258 });
        scores.Fill((AiDotNet.Tensors.NumericOperations.BFloat16)1f);
        AssertBFloatSelection(OrderedTopKSelection<AiDotNet.Tensors.NumericOperations.BFloat16>.Select(engine, scores, 257), false);
        var parameter = new Tensor<AiDotNet.Tensors.NumericOperations.BFloat16>(new[] { 1, 257, 258 });
        parameter.Fill((AiDotNet.Tensors.NumericOperations.BFloat16)1f);
        AiDotNet.Tensors.Engines.Compilation.CompiledTrainingPlan<AiDotNet.Tensors.NumericOperations.BFloat16> plan;
        Tensor<AiDotNet.Tensors.NumericOperations.BFloat16> selection;
        using (var scope = AiDotNet.Tensors.Engines.Compilation.GraphMode.EnableTraining(new[] { parameter }))
        {
            using (AiDotNet.Tensors.Engines.Autodiff.GradientTape<AiDotNet.Tensors.NumericOperations.BFloat16>.NoGrad())
                selection = OrderedTopKSelection<AiDotNet.Tensors.NumericOperations.BFloat16>.Select(engine, scores, 257);
            var loss = engine.ReduceSum(engine.TensorMultiply(parameter, selection), null, false);
            Assert.Null(scope.HostReadOp);
            plan = scope.CompileTraining(new[] { parameter }, loss);
        }
        using (plan)
        {
            plan.Step();
            AssertBFloatSelection(selection, false);
            scores[0, 257] = (AiDotNet.Tensors.NumericOperations.BFloat16)2f;
            plan.Step();
            AssertBFloatSelection(selection, true);
        }
    }

    private static void AssertBFloatSelection(Tensor<AiDotNet.Tensors.NumericOperations.BFloat16> selection, bool lastFirst)
    {
        var values = selection.ToArray();
        for (int u = 0; u < 257; u++)
            for (int i = 0; i < 258; i++)
            {
                int expected = lastFirst ? (u == 0 ? 257 : u - 1) : u;
                Assert.Equal(expected == i ? 1f : 0f, (float)values[u * 258 + i]);
            }
    }
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
            if (gpu is null || !gpu.IsGpuAvailable)
            {
                Skip.If(true, "GPU backend did not resolve.");
                return;
            }
            AssertSelects(gpu, k);
        }
        finally { gpu?.Dispose(); }
    }
}
