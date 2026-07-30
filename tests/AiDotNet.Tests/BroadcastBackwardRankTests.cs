using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.Diagnostics;

/// <summary>
/// Backward finiteness for broadcast ops whose second operand is singleton on EVERY axis.
/// </summary>
/// <remarks>
/// RevIN's reverse step does forecast * std + mean with std/mean shaped [batch, 1]. At batch = 1 that
/// operand is [1, 1] -- singleton on BOTH axes -- so the broadcast backward has to reduce-sum over two
/// broadcast axes instead of one. RWKVForecaster&lt;float&gt; takes ALL 17,576 parameters to NaN after a
/// single step at batch 1 while batch 2 and 4 are clean, which points here.
/// </remarks>
public class BroadcastBackwardRankTests
{
    private readonly ITestOutputHelper _out;
    public BroadcastBackwardRankTests(ITestOutputHelper output) => _out = output;

    private static bool Fin(double v) => !double.IsNaN(v) && !double.IsInfinity(v);

    private static Tensor<double> T(int[] shape, double start)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t.Data.Span[i] = start + (i * 0.25);
        return t;
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    public void BroadcastMultiplyThenAdd_Backward_IsFinite(int batch)
    {
        var engine = AiDotNetEngine.Current;
        var work = T([batch, 8], 0.5);
        var std = T([batch, 1], 1.5);
        var mean = T([batch, 1], -0.25);

        using var tape = new GradientTape<double>();
        var scaled = engine.TensorBroadcastMultiply(work, std);
        var shifted = engine.TensorBroadcastAdd(scaled, mean);

        var axes = new int[shifted.Shape.Length];
        for (int i = 0; i < axes.Length; i++) axes[i] = i;
        var loss = engine.ReduceSum(shifted, axes, keepDims: false);

        var grads = tape.ComputeGradients(loss, new[] { work, std, mean });

        foreach (var kvp in grads)
        {
            var g = kvp.Value;
            Assert.NotNull(g);
            int bad = 0;
            for (int i = 0; i < g!.Length; i++) if (!Fin(g[i])) bad++;
            _out.WriteLine($"batch={batch} operand[{string.Join("x", kvp.Key.Shape.ToArray())}] " +
                           $"grad[{string.Join("x", g.Shape.ToArray())}] nonFinite={bad}/{g.Length} g[0]={g[0]}");
            Assert.True(bad == 0,
                $"batch={batch}: {bad}/{g.Length} gradient values non-finite for the " +
                $"[{string.Join("x", kvp.Key.Shape.ToArray())}] operand.");
        }
    }
}
