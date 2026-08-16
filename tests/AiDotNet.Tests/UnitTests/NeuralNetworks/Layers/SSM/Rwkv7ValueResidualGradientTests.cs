using System;
using System.Linq;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers.SSM;

/// <summary>
/// Proves the RWKV-7 value residual is differentiable ACROSS layers, not merely computed.
/// </summary>
/// <remarks>
/// <para>
/// The residual <c>v = v + (v_first - v) * sigmoid(v0 + (x_v @ v1) @ v2)</c> is the one term in the
/// block that reaches backwards to a DIFFERENT layer: every layer above the first consumes the first
/// layer's value projection. That makes it the one term whose gradient can be wrong in a way nothing
/// else notices — shapes stay correct, loss still moves, and the model simply trains the wrong way.
/// </para>
/// <para>
/// An earlier attempt threaded v_first through a shared mutable carrier instead of returning it as a
/// value. It compiled, ran, and doubled the failures in the model it was meant to fix. These tests
/// exist so that class of mistake fails loudly here instead of quietly during training.
/// </para>
/// </remarks>
public class Rwkv7ValueResidualGradientTests
{
    private const int SeqLen = 6;
    private const int ModelDim = 32;
    private const int NumHeads = 4;
    private const int NumLayers = 3;   // >1 so v_first is actually threaded

    private static Tensor<double> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() * 0.2 - 0.1;
        return t;
    }

    /// <summary>
    /// Builds a stack whose time-mixing branch actually reaches the output.
    /// </summary>
    /// <remarks>
    /// RWKV-7 initializes the time-mixing output projection to exactly zero (the reference does
    /// <c>self.output.weight.data.zero_()</c>) so each block starts as an identity on the residual
    /// stream. That is correct, and it makes <c>dL/dθ = 0</c> for every parameter upstream of it —
    /// the value projection, the value residual, all of it. Measured at initialization, a gradient
    /// test would read zero everywhere and could not tell a healthy graph from a severed one.
    /// So we move the output projection off zero, exactly as one optimizer step would.
    /// </remarks>
    private static Rwkv7Stack<double> MakeTrainedLikeStack(int seed = 1)
    {
        var stack = new Rwkv7Stack<double>(NumLayers, SeqLen, ModelDim, NumHeads);
        int s = seed;
        foreach (var block in stack.Blocks)
        {
            var w = block.OutputProjectionWeights;
            var replacement = Rand(w.Shape.ToArray(), s++);
            for (int i = 0; i < w.Length; i++) w[i] = replacement[i];
        }
        return stack;
    }

    /// <summary>
    /// Floor check: every cross-layer parameter above the first layer receives a finite, non-zero
    /// gradient.
    /// </summary>
    /// <remarks>
    /// Layer 0's v0/v1/v2 are deliberately excluded. It is handed no v_first, so the residual is
    /// inactive there and a zero gradient is correct — asserting otherwise would be asserting a bug.
    /// v1 is also excluded: RWKV-7 zero-initializes it, and with <c>v1 = 0</c> the LoRA branch
    /// <c>(x_v @ v1) @ v2</c> contributes nothing to v2's gradient either. Both are checked in
    /// <see cref="ValueResidualLoraMatrices_TrainOnceOffZero"/> instead.
    /// </remarks>
    [Fact]
    public void ValueResidualParameters_ReceiveFiniteNonZeroGradients()
    {
        var engine = new CpuEngine();
        var stack = MakeTrainedLikeStack();
        var input = Rand([1, SeqLen, ModelDim], 3);

        // v0 only: the gate bias is live from step one because v1 starts at zero.
        var vParams = stack.Blocks
            .Skip(1)                                   // layer 0 never consumes a v_first
            .Select(b => b.ValueResidualParameters[0])
            .ToArray();
        Assert.NotEmpty(vParams);

        using var tape = new GradientTape<double>();
        var loss = engine.ReduceSum(stack.Forward(input), null, keepDims: false);
        var grads = tape.ComputeGradients(loss, vParams);

        for (int p = 0; p < vParams.Length; p++)
        {
            Assert.True(grads.ContainsKey(vParams[p]),
                $"layer {p + 1} v0 received no gradient — the cross-layer blend is detached.");

            AssertFiniteAndNonZero(grads[vParams[p]], $"layer {p + 1} v0");
        }
    }

    /// <summary>
    /// Once v1 is off its zero init, both LoRA matrices of the gate train.
    /// </summary>
    [Fact]
    public void ValueResidualLoraMatrices_TrainOnceOffZero()
    {
        var engine = new CpuEngine();
        var stack = MakeTrainedLikeStack(seed: 40);
        var input = Rand([1, SeqLen, ModelDim], 7);

        var block = stack.Blocks[NumLayers - 1];
        var v1 = block.ValueResidualParameters[1];
        var v2 = block.ValueResidualParameters[2];

        var seeded = Rand(v1.Shape.ToArray(), 61);
        for (int i = 0; i < v1.Length; i++) v1[i] = seeded[i];

        using var tape = new GradientTape<double>();
        var loss = engine.ReduceSum(stack.Forward(input), null, keepDims: false);
        var grads = tape.ComputeGradients(loss, new[] { v1, v2 });

        AssertFiniteAndNonZero(grads[v1], "top-layer v1");
        AssertFiniteAndNonZero(grads[v2], "top-layer v2");
    }

    /// <summary>
    /// The real proof: cross-layer gradients match central finite differences.
    /// </summary>
    /// <remarks>
    /// Non-zero and finite would still pass if the fan-out from several consuming layers back into
    /// one producing layer accumulated wrongly — say counted once instead of summed, or scaled. Only
    /// comparing against a numerical derivative of the actual loss catches that.
    /// </remarks>
    [Fact]
    public void ValueResidualGradients_MatchFiniteDifferences()
    {
        var engine = new CpuEngine();
        var input = Rand([1, SeqLen, ModelDim], 11);

        var stack = MakeTrainedLikeStack(seed: 20);
        var target = stack.Blocks[NumLayers - 1].ValueResidualParameters[0];   // the top layer's v0

        double Loss()
        {
            stack.ResetState();
            return engine.ReduceSum(stack.Forward(input), null, keepDims: false)[0];
        }

        stack.ResetState();
        Tensor<double> analytic;
        using (var tape = new GradientTape<double>())
        {
            var value = engine.ReduceSum(stack.Forward(input), null, keepDims: false);
            analytic = tape.ComputeGradients(value, new[] { target })[target];
        }

        const double eps = 1e-6;
        int probes = Math.Min(4, target.Length);
        double largestNumeric = 0;

        for (int k = 0; k < probes; k++)
        {
            double original = target[k];

            target[k] = original + eps;
            double plus = Loss();

            target[k] = original - eps;
            double minus = Loss();

            target[k] = original;

            double numeric = (plus - minus) / (2 * eps);
            largestNumeric = Math.Max(largestNumeric, Math.Abs(numeric));

            double tolerance = 2e-4 * Math.Max(1.0, Math.Abs(numeric));
            Assert.True(Math.Abs(numeric - analytic[k]) <= tolerance,
                $"v0[{k}] analytic gradient {analytic[k]:G10} disagrees with finite differences " +
                $"{numeric:G10}. The cross-layer blend is differentiable but computes the wrong " +
                "derivative.");
        }

        // Without this the test passes when analytic and numeric are both ~0 — which is exactly what
        // happens at the zero-initialized output projection, and would prove nothing.
        Assert.True(largestNumeric > 1e-6,
            $"finite differences are all ~0 (max {largestNumeric:G10}); the comparison above was " +
            "vacuous and did not exercise the value residual.");
    }

    /// <summary>
    /// The first layer's value projection must receive gradient from the layers ABOVE it, not only
    /// from its own output path.
    /// </summary>
    /// <remarks>
    /// This is the fan-out the carrier design got wrong: v_first is consumed by every later layer, so
    /// the producing layer's gradient is a SUM over those consumers. To show the cross-layer edges
    /// specifically, this compares against a stack whose upper blocks cannot contribute — if the only
    /// gradient reaching layer 0 came from its own output, the two would be identical.
    /// </remarks>
    [Fact]
    public void FirstLayerValueProjection_ReceivesGradientFromLayersAbove()
    {
        var engine = new CpuEngine();
        var input = Rand([1, SeqLen, ModelDim], 17);

        double GradMagnitude(Rwkv7Stack<double> stack)
        {
            var w = stack.Blocks[0].ValueProjectionWeights;
            using var tape = new GradientTape<double>();
            var loss = engine.ReduceSum(stack.Forward(input), null, keepDims: false);
            var grads = tape.ComputeGradients(loss, new[] { w });

            Assert.True(grads.ContainsKey(w),
                "the first layer's value projection received no gradient at all");

            AssertFiniteAndNonZero(grads[w], "first-layer value projection");
            double m = 0;
            for (int i = 0; i < grads[w].Length; i++) m += Math.Abs(grads[w][i]);
            return m;
        }

        double withUpperLayers = GradMagnitude(MakeTrainedLikeStack(seed: 5));

        // Same stack, but every upper block's value-residual gate is forced shut (sigmoid(-40) ~ 0),
        // so v_first is blended in with weight ~0 and the cross-layer edges carry nothing.
        var gated = MakeTrainedLikeStack(seed: 5);
        foreach (var block in gated.Blocks.Skip(1))
        {
            var v0 = block.ValueResidualParameters[0];
            for (int i = 0; i < v0.Length; i++) v0[i] = -40.0;
        }
        double withoutUpperLayers = GradMagnitude(gated);

        Assert.True(Math.Abs(withUpperLayers - withoutUpperLayers) > 1e-9,
            $"closing the upper layers' value-residual gates did not change the first layer's " +
            $"value-projection gradient ({withUpperLayers:G10} vs {withoutUpperLayers:G10}); the " +
            "cross-layer edge from the consuming layers is missing.");
    }

    private static void AssertFiniteAndNonZero(Tensor<double> g, string what)
    {
        double magnitude = 0;
        for (int i = 0; i < g.Length; i++)
        {
            Assert.True(!double.IsNaN(g[i]) && !double.IsInfinity(g[i]),
                $"{what} gradient[{i}] is {g[i]}");
            magnitude += Math.Abs(g[i]);
        }
        Assert.True(magnitude > 1e-12,
            $"{what} has an all-zero gradient (sum |g| = {magnitude}); it cannot train.");
    }
}
