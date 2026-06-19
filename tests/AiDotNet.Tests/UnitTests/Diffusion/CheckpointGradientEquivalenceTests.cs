using System.Linq;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion;

/// <summary>
/// Verifies that the package activation-checkpointing primitive
/// <see cref="AiDotNet.Tensors.Engines.Autodiff.GradientCheckpointing{T}.Checkpoint(System.Func{Tensor{T},Tensor{T}}[],Tensor{T},int)"/>
/// (G4, #1624) is gradient-equivalent to a non-checkpointed block under a bare
/// <c>new GradientTape&lt;T&gt;()</c> — the exact tape construction <c>DiffusionModelBase.Train</c> uses.
/// Running a block eager vs checkpointed must yield IDENTICAL gradients w.r.t. the block input AND
/// every block parameter. The driver uses a NON-uniform output weighting so the gradient flowing into
/// the block (gradOutput) is non-constant — exactly the case the prior ones-seeded VJP got wrong
/// (AiDotNet#1341, fixed in #361, shipped in 0.101.2). Checkpointing only changes WHEN activations are
/// materialized, never their values or gradients.
/// </summary>
/// <remarks>
/// This is the consumer-side proof that the package primitive — unlike a consumer-recorded
/// <c>tape.Record</c> custom node, which does NOT link into the eager tape's backward walk — links
/// correctly in a bare tape and therefore makes diffusion G4 viable through <c>CheckpointBlock</c>.
/// </remarks>
public class CheckpointGradientEquivalenceTests
{
    private static IEngine Engine => AiDotNetEngine.Current;

    private static Tensor<double> Filled(int[] shape, double start, double step)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = start + step * i;
        return t;
    }

    // Runs loss = sum(blockChain(input) * outWeight) and returns grads for [input, ...params].
    private static System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> RunGrads(
        System.Func<Tensor<double>, Tensor<double>>[] blocks, Tensor<double> input,
        Tensor<double>[] pars, Tensor<double> outWeight, bool checkpoint)
    {
        using var tape = new GradientTape<double>();

        Tensor<double> output;
        if (checkpoint)
        {
            // segmentSize 1 = checkpoint every block (most aggressive; recompute each in backward).
            output = GradientCheckpointing<double>.Checkpoint(blocks, input, 1);
        }
        else
        {
            output = input;
            foreach (var b in blocks) output = b(output);
        }

        var loss = Engine.ReduceSum(Engine.TensorMultiply(output, outWeight), null);
        var sources = new[] { input }.Concat(pars).ToArray();
        // sources: null = differentiate the FULL graph (the mode NeuralNetworkBase tape-training uses
        // at NeuralNetworkBase.cs:3404). The package checkpoint scatters param grads into the full
        // result; an explicit pruned source list misses the recompute's param contributions.
        var grads = tape.ComputeGradients(loss, null);

        var result = new System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>>();
        foreach (var s in sources)
            if (grads.TryGetValue(s, out var g) && g is not null) result[s] = g;
        return result;
    }

    [Fact]
    public void PackageCheckpoint_GradientsMatchEager_ForInputAndParameters()
    {
        // A two-block chain with non-linear activations so the Jacobian is non-trivial and
        // shape-aware (the case the ones-seed VJP got wrong).
        var l0 = new DenseLayer<double>(
            6, (AiDotNet.Interfaces.IActivationFunction<double>)new AiDotNet.ActivationFunctions.GELUActivation<double>());
        var l1 = new DenseLayer<double>(
            6, (AiDotNet.Interfaces.IActivationFunction<double>)new AiDotNet.ActivationFunctions.GELUActivation<double>());
        l0.SetTrainingMode(true);
        l1.SetTrainingMode(true);
        var input = Filled(new[] { 2, 5 }, start: -0.3, step: 0.017);

        // Resolve lazy weights once (un-taped) so both runs share identical parameters.
        using (new NoGradScope<double>()) { _ = l1.Forward(l0.Forward(input)); }
        var pars = l0.GetTrainableParameters().Concat(l1.GetTrainableParameters()).ToArray();
        Assert.True(pars.Length > 0, "Blocks have no trainable parameters to check.");

        var blocks = new System.Func<Tensor<double>, Tensor<double>>[] { l0.Forward, l1.Forward };
        var outWeight = Filled(new[] { 2, 6 }, start: 0.11, step: 0.037); // NON-uniform gradOutput driver

        var eager = RunGrads(blocks, input, pars, outWeight, checkpoint: false);
        var ckpt = RunGrads(blocks, input, pars, outWeight, checkpoint: true);

        // Input gradient present and identical.
        Assert.True(eager.ContainsKey(input) && ckpt.ContainsKey(input),
            $"Missing input gradient — eager: {eager.ContainsKey(input)}, checkpointed: {ckpt.ContainsKey(input)} " +
            $"(eager count={eager.Count}, ckpt count={ckpt.Count}, params={pars.Length})");
        Assert.Equal(eager[input].Length, ckpt[input].Length);
        for (int i = 0; i < eager[input].Length; i++)
            Assert.Equal(eager[input][i], ckpt[input][i], 10);

        // Every parameter gradient present and identical.
        foreach (var p in pars)
        {
            Assert.True(eager.ContainsKey(p), "Eager run missing a parameter gradient.");
            Assert.True(ckpt.ContainsKey(p), "Checkpointed run missing a parameter gradient (the bug class).");
            Assert.Equal(eager[p].Length, ckpt[p].Length);
            for (int i = 0; i < eager[p].Length; i++)
                Assert.Equal(eager[p][i], ckpt[p][i], 10);
        }
    }
}
