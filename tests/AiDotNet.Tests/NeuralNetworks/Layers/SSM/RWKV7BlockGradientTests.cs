using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.NeuralNetworks.Layers.SSM;

/// <summary>
/// Backward-pass finiteness invariants for <see cref="RWKV7Block{T}"/> (Peng et al., arXiv 2503.14456).
/// </summary>
/// <remarks>
/// <para>
/// These exist because RWKVForecaster's generated FP32 fixture produced NaN PARAMETERS after a single
/// optimizer step at a learning rate of 1e-4, while its forward pass stayed finite and its 15-step
/// memorization trajectory decreased normally. A learning rate that small cannot turn a finite
/// gradient into NaN, so the gradient itself must arrive non-finite -- which localises the defect to
/// the backward sweep of the fused delta-rule kernel or to the block's own backward composition.
/// </para>
/// <para>
/// The block is exercised at exactly the fixture's shape (seqLen 32, modelDim 32, 4 heads, so
/// headDim 8). Both precisions are covered on purpose: <c>float</c> runs the kernel's GENERIC
/// backward while <c>double</c> runs a separate specialised one, and a discrepancy between those two
/// implementations is exactly the kind of defect that reproduces in one precision only.
/// </para>
/// <para>
/// Gradients come from <see cref="GradientTape{T}"/>, not a layer-level Backward method: Backward was
/// removed in the tape-based autodiff migration, so the tape IS the backward pass here.
/// </para>
/// </remarks>
public class RWKV7BlockGradientTests
{
    private const int SeqLen = 32;
    private const int ModelDim = 32;
    private const int NumHeads = 4;

    /// <summary>net471-safe finiteness check (<c>double.IsFinite</c> does not exist there).</summary>
    private static bool IsFinite(double v) => !double.IsNaN(v) && !double.IsInfinity(v);

    private static Tensor<T> Fill<T>(int[] shape, Func<int, double> value)
    {
        var t = new Tensor<T>(shape);
        var ops = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < t.Length; i++)
        {
            t.Data.Span[i] = ops.FromDouble(value(i));
        }

        return t;
    }

    /// <summary>Deterministic input in a LayerNorm-like range; no RNG, so failures are reproducible.</summary>
    private static Tensor<T> Input<T>(int batch)
        => Fill<T>([batch, SeqLen, ModelDim], i => (Math.Sin(i * 0.37) * 0.5) + (Math.Cos(i * 0.11) * 0.25));

    /// <summary>
    /// Forwards the block under a tape, differentiates L = sum(output), and asserts every trainable
    /// parameter gradient is finite. The forward output is verified finite FIRST so a failure can only
    /// be attributed to the backward, and L = sum(output) gives dL/doutput = 1 everywhere, so the loss
    /// itself cannot be the source of a non-finite value.
    /// </summary>
    private static void AssertBackwardIsFinite<T>(RWKV7Block<T> block, Tensor<T> input, string context)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        block.SetTrainingMode(true);

        using var tape = new GradientTape<T>();
        var output = block.Forward(input);

        for (int i = 0; i < output.Length; i++)
        {
            double v = ops.ToDouble(output.Data.Span[i]);
            Assert.True(IsFinite(v), $"{context}: forward output[{i}] = {v} is not finite — precondition failed, this is a FORWARD defect.");
        }

        // Collect AFTER Forward: lazy-init sub-layers reassign their tensor fields inside the first
        // forward, so pre-Forward references are stale placeholders the tape never saw.
        Assert.True(block is ITrainableLayer<T>, $"{context}: block does not expose ITrainableLayer<T>.");
        var trainableParams = ((ITrainableLayer<T>)block).GetTrainableParameters();
        Assert.True(trainableParams.Count > 0, $"{context}: block reported no trainable parameters.");

        var allAxes = new int[output.Shape.Length];
        for (int i = 0; i < allAxes.Length; i++)
        {
            allAxes[i] = i;
        }

        var loss = AiDotNetEngine.Current.ReduceSum(output, allAxes, keepDims: false);
        var grads = tape.ComputeGradients(loss, trainableParams);

        int inspected = 0, nonFinite = 0, firstFlat = -1;
        double firstValue = 0.0, maxAbs = 0.0;
        string firstShape = string.Empty;

        foreach (var kvp in grads)
        {
            var grad = kvp.Value;
            if (grad is null)
            {
                continue;
            }

            for (int i = 0; i < grad.Length; i++)
            {
                double g = ops.ToDouble(grad[i]);
                inspected++;
                if (!IsFinite(g))
                {
                    nonFinite++;
                    if (firstFlat < 0)
                    {
                        firstFlat = i;
                        firstValue = g;
                        firstShape = string.Join("x", kvp.Key.Shape.ToArray());
                    }
                }
                else if (Math.Abs(g) > maxAbs)
                {
                    maxAbs = Math.Abs(g);
                }
            }
        }

        Assert.True(inspected > 0, $"{context}: the tape returned no gradient values to inspect.");
        Assert.True(
            nonFinite == 0,
            $"{context}: {nonFinite}/{inspected} parameter gradients are non-finite " +
            $"(first in the [{firstShape}] parameter at flat index {firstFlat} = {firstValue}); " +
            $"largest finite |grad| = {maxAbs:G6}. The forward was verified finite, so the backward is at fault.");
    }

    [Fact]
    public void Backward_Float_Batch1_ProducesFiniteParameterGradients()
        => AssertBackwardIsFinite(
            new RWKV7Block<float>(SeqLen, ModelDim, NumHeads), Input<float>(1), "float / batch 1");

    /// <summary>
    /// Batch 2 is the shape every FAILING invariant uses, while
    /// <c>LossStrictlyDecreasesOnMemorizationTask</c> trains on a single sample and PASSES — so batch
    /// is the prime suspect dimension.
    /// </summary>
    [Fact]
    public void Backward_Float_Batch2_ProducesFiniteParameterGradients()
        => AssertBackwardIsFinite(
            new RWKV7Block<float>(SeqLen, ModelDim, NumHeads), Input<float>(2), "float / batch 2");

    /// <summary>The double path runs a separate specialised kernel; both must agree on finiteness.</summary>
    [Fact]
    public void Backward_Double_Batch2_ProducesFiniteParameterGradients()
        => AssertBackwardIsFinite(
            new RWKV7Block<double>(SeqLen, ModelDim, NumHeads), Input<double>(2), "double / batch 2");

    /// <summary>
    /// Edge case: an all-zero input drives kappa to zero, so Eq 15's per-head L2 normalisation
    /// kappaHat = kappa / ||kappa|| hits its guarded 0/0 branch in BOTH sweeps — and the backward
    /// additionally divides the kappaHat adjoint by that same norm.
    /// </summary>
    [Fact]
    public void Backward_ZeroInput_KappaNormGuard_ProducesFiniteParameterGradients()
        => AssertBackwardIsFinite(
            new RWKV7Block<float>(SeqLen, ModelDim, NumHeads),
            Fill<float>([2, SeqLen, ModelDim], _ => 0.0),
            "float / zero input (kappa-norm guard)");

    /// <summary>
    /// The Global ICLR Multiplier c must not change finiteness. c = 1 is what RWKV-7 language
    /// modeling uses; c = exp(-e^-0.5) is the paper's own restricted range (footnote 4) — the largest
    /// c keeping w - c*a positive so the transition product stays bounded.
    /// </summary>
    [Theory]
    [InlineData(1.0)]
    [InlineData(0.5452392118926051)]
    [InlineData(0.25)]
    public void Backward_AcrossGlobalIclrMultiplier_ProducesFiniteParameterGradients(double c)
        => AssertBackwardIsFinite(
            new RWKV7Block<float>(SeqLen, ModelDim, NumHeads, globalIclrMultiplier: c),
            Input<float>(2),
            $"float / c = {c}");

    /// <summary>
    /// Multi-draw FORWARD finiteness for the block at batch 1 vs 2.
    /// </summary>
    /// <remarks>
    /// The single-draw tests above lacked the power to clear the block: the forecaster fails on ~44% of
    /// draws at batch 1, and the failing draw reports LastLoss=NaN, so the NaN is produced in the
    /// FORWARD. Each trial builds a fresh block (fresh unseeded init) and varies the input, then checks
    /// the forward for non-finite values and records the largest magnitude — an overflow to Inf in the
    /// 32-step WKV recurrence would show up as a growing maxAbs before it tips over.
    /// </remarks>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void Forward_OverManyDraws_StaysFinite(int batch)
    {
        const int trials = 25;
        int bad = 0;
        double worstMax = 0;

        for (int t = 0; t < trials; t++)
        {
            var block = new RWKV7Block<float>(SeqLen, ModelDim, NumHeads);
            block.SetTrainingMode(true);

            var x = Fill<float>([batch, SeqLen, ModelDim],
                i => (Math.Sin((i + t) * 0.37) * 0.5) + (Math.Cos((i + t) * 0.11) * 0.25));

            var y = block.Forward(x);
            int nf = 0;
            double mx = 0;
            for (int i = 0; i < y.Length; i++)
            {
                double v = y.Data.Span[i];
                if (!IsFinite(v)) nf++; else mx = Math.Max(mx, Math.Abs(v));
            }
            worstMax = Math.Max(worstMax, mx);
            if (nf > 0)
            {
                bad++;
                if (bad == 1) Console.WriteLine($"  first bad draw t={t}: {nf}/{y.Length} non-finite, finite maxAbs={mx:G6}");
            }
        }

        Assert.True(bad == 0,
            $"batch={batch}: {bad}/{trials} fresh blocks produced a NON-FINITE forward; largest finite |out| seen = {worstMax:G6}.");
    }
}
