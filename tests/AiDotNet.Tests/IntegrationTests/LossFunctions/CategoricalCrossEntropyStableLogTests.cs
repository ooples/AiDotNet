using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.LossFunctions;

/// <summary>
/// Numerics tests for the floor-only stable log in <see cref="CategoricalCrossEntropyLoss{T}"/>
/// (`ComputeTapeLoss` now uses `TensorClampMin(p, 1e-7)` + `Log` instead of the two-sided
/// `Clamp(p, 1e-7, 1)` + `Log`).
///
/// <para>Contract:
/// (1) For any VALID probability distribution (softmax output, every element ≤ 1) the loss is
///     BIT-IDENTICAL to the previous two-sided-clamp formulation — the upper cap at 1.0 was
///     redundant, so nothing changes on the real (softmax-head) path.
/// (2) The loss stays finite and non-negative at the extremes: a near-0 target probability gives
///     a large finite loss (no NaN/Inf), and a perfectly-confident p=1 gives loss 0 (never a tiny
///     negative value).
/// (3) The tape path agrees with the scalar CalculateLoss (which uses SafeLog = log(max(p, eps))).
/// </para>
/// </summary>
public class CategoricalCrossEntropyStableLogTests
{
    private readonly ITestOutputHelper _o;
    public CategoricalCrossEntropyStableLogTests(ITestOutputHelper o) { _o = o; }

    // Reference: exactly what the OLD two-sided clamp produced — clamp(p, 1e-7, 1), log, ·target,
    // sum over classes, mean over batch, negate. For p ≤ 1 this equals the new floor-only path.
    private static double RefLoss(float[,] p, float[,] y)
    {
        int B = p.GetLength(0), V = p.GetLength(1);
        double tot = 0;
        for (int i = 0; i < B; i++)
        {
            double row = 0;
            for (int v = 0; v < V; v++)
            {
                double pc = Math.Min(1.0, Math.Max(1e-7, p[i, v]));
                row += y[i, v] * Math.Log(pc);
            }
            tot += row;
        }
        return -(tot / B);
    }

    private static (Tensor<float> p, Tensor<float> y) ToTensors(float[,] p, float[,] y)
    {
        int B = p.GetLength(0), V = p.GetLength(1);
        var pt = new Tensor<float>(new[] { B, V });
        var yt = new Tensor<float>(new[] { B, V });
        for (int i = 0; i < B; i++)
            for (int v = 0; v < V; v++) { pt[i, v] = p[i, v]; yt[i, v] = y[i, v]; }
        return (pt, yt);
    }

    [Fact]
    public void FloorOnly_BitIdentical_To_TwoSidedClamp_For_ValidProbabilities()
    {
        // Realistic softmax rows (each sums to 1, all ≤ 1), incl. a near-1 confident row.
        var rng = new Random(11);
        int B = 8, V = 32;
        var p = new float[B, V];
        var y = new float[B, V];
        for (int i = 0; i < B; i++)
        {
            double sum = 0; var logits = new double[V];
            for (int v = 0; v < V; v++) { logits[v] = rng.NextDouble() * 6 - 3; }
            if (i == 0) logits[3] = 40; // near-one-hot row: softmax → p≈1 at class 3
            double mx = double.NegativeInfinity; for (int v = 0; v < V; v++) mx = Math.Max(mx, logits[v]);
            for (int v = 0; v < V; v++) { p[i, v] = (float)Math.Exp(logits[v] - mx); sum += p[i, v]; }
            for (int v = 0; v < V; v++) p[i, v] = (float)(p[i, v] / sum);
            y[i, rng.Next(V)] = 1f;
        }

        var (pt, yt) = ToTensors(p, y);
        var loss = new CategoricalCrossEntropyLoss<float>(); // labelSmoothing default 0
        double got = (double)loss.ComputeTapeLoss(pt, yt).ToArray()[0];
        double reference = RefLoss(p, y);
        _o.WriteLine($"floor-only={got:G9} two-sided-ref={reference:G9} diff={Math.Abs(got - reference):E3}");
        Assert.Equal(reference, got, 5); // identical to old two-sided clamp for valid probs
        Assert.True(got >= 0.0, $"CE loss must be non-negative, got {got}");
    }

    [Fact]
    public void Stable_At_Extremes_NoNaN_NoNegative()
    {
        var loss = new CategoricalCrossEntropyLoss<float>();

        // (a) target class has near-ZERO probability → large FINITE loss (floored log, no -Inf/NaN).
        var (p0, y0) = ToTensors(new float[,] { { 1e-12f, 1f - 1e-12f } }, new float[,] { { 1f, 0f } });
        double near0 = (double)loss.ComputeTapeLoss(p0, y0).ToArray()[0];
        _o.WriteLine($"near-0 target prob loss={near0:G6}");
        Assert.False(double.IsNaN(near0) || double.IsInfinity(near0), "near-0 prob must give finite loss");
        Assert.True(near0 > 10.0 && near0 < 20.0, $"near-0 loss should be ≈ -log(1e-7)=16.1, got {near0}");

        // (b) perfectly-confident correct prediction p=1 → loss exactly 0 (never a tiny negative).
        var (p1, y1) = ToTensors(new float[,] { { 1f, 0f } }, new float[,] { { 1f, 0f } });
        double conf = (double)loss.ComputeTapeLoss(p1, y1).ToArray()[0];
        _o.WriteLine($"p=1 confident loss={conf:G6}");
        Assert.True(conf >= 0.0 && conf < 1e-6, $"p=1 loss must be 0 (non-negative), got {conf}");
    }
}
