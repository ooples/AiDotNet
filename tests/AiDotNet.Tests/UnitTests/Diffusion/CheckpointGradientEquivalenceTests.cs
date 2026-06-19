using System.Collections.Generic;
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
/// <c>new GradientTape&lt;T&gt;()</c> — the exact tape construction <c>DiffusionModelBase.Train</c> uses —
/// for real <see cref="DenseLayer{T}"/> blocks. Running a block eager vs checkpointed must yield
/// IDENTICAL gradients w.r.t. the block input AND every block weight. A NON-uniform output weighting
/// makes the upstream gradient non-constant (the case the prior ones-seeded VJP got wrong, #1341/#361).
/// </summary>
/// <remarks>
/// Each run builds FRESH layers + a FRESH input (with identical parameter VALUES copied from a master),
/// so there is no shared mutable gradient state between the eager and checkpointed runs — the comparison
/// isolates the checkpoint mechanism itself, not cross-call gradient accumulation on reused tensors.
/// </remarks>
public class CheckpointGradientEquivalenceTests : System.IDisposable
{
    // Pin CPU for this gradient-correctness suite — a [ModuleInitializer] can flip
    // AiDotNetEngine.Current to an optimized engine (same convention as the Tensors repo's
    // GradientCorrectnessTests, which pins CPU so the comparison hits the reference autodiff path).
    private readonly IEngine _priorEngine = AiDotNetEngine.Current;
    public CheckpointGradientEquivalenceTests() { AiDotNetEngine.Current = new CpuEngine(); }
    public void Dispose() { AiDotNetEngine.Current = _priorEngine; }

    private static IEngine Engine => AiDotNetEngine.Current;

    private static Tensor<double> Filled(int[] shape, double start, double step)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = start + step * i;
        return t;
    }

    private static DenseLayer<double> NewBlock() => new(
        6, (AiDotNet.Interfaces.IActivationFunction<double>)new AiDotNet.ActivationFunctions.GELUActivation<double>());

    // Builds two fresh blocks, copies in the given parameter values, runs loss = sum(chain(input)*outW)
    // ONCE on a fresh input, and returns the flattened gradient vector for [input, ...all block params].
    private static double[] RunOnce(IReadOnlyList<Tensor<double>> p0, IReadOnlyList<Tensor<double>> p1, bool checkpoint)
    {
        var l0 = NewBlock();
        var l1 = NewBlock();
        l0.SetTrainingMode(true);
        l1.SetTrainingMode(true);
        var input = Filled(new[] { 2, 5 }, start: -0.3, step: 0.017);

        // Resolve lazy weights, then overwrite with the shared master values so both runs are identical.
        using (new NoGradScope<double>()) { _ = l1.Forward(l0.Forward(input)); }
        l0.SetTrainableParameters(p0.Select(t => t.Clone()).ToList());
        l1.SetTrainableParameters(p1.Select(t => t.Clone()).ToList());

        var pars = l0.GetTrainableParameters().Concat(l1.GetTrainableParameters()).ToArray();
        var outWeight = Filled(new[] { 2, 6 }, start: 0.11, step: 0.037); // NON-uniform gradOutput driver

        using var tape = new GradientTape<double>();
        var blockFns = new System.Func<Tensor<double>, Tensor<double>>[] { l0.Forward, l1.Forward };
        Tensor<double> output = checkpoint
            // segmentSize 1 → MULTIPLE segments (one per block), the cross-segment recompute path
            // NoisePredictorBase.CheckpointBlocks uses via sqrt(N). Requires the multi-segment
            // double-count fix (AiDotNet.Tensors >= 0.101.5 / #645).
            ? GradientCheckpointing<double>.Checkpoint(blockFns, input, 1)
            : l1.Forward(l0.Forward(input));
        var loss = Engine.ReduceSum(Engine.TensorMultiply(output, outWeight), null);

        var sources = new[] { input }.Concat(pars).ToArray();
        var grads = tape.ComputeGradients(loss, sources);

        var flat = new List<double>();
        foreach (var s in sources)
        {
            Assert.True(grads.TryGetValue(s, out var g) && g is not null,
                "A source (input or block weight) received no gradient — checkpointed weight gradients must flow.");
            for (int i = 0; i < g.Length; i++) flat.Add(g[i]);
        }
        return flat.ToArray();
    }

    // ISOLATION: the exact raw-matmul checkpoint test that PASSES in the AiDotNet.Tensors repo against
    // local source. If this passes here too (against the published 0.101.4 NuGet) the primitive is fine
    // and the DenseLayer failure is consumer-side; if it fails, the published package differs from source.
    [Fact]
    public void PackageCheckpoint_RawMatmul_GradientsMatchEager()
    {
        var x = Filled(new[] { 2, 3 }, -0.4, 0.05);
        var w = Filled(new[] { 3, 4 }, -0.3, 0.03);
        var outW = Filled(new[] { 2, 4 }, 0.1, 0.02);
        System.Func<Tensor<double>, Tensor<double>> seg = inp => Engine.ReLU(Engine.TensorMatMul(inp, w));

        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> Run(bool checkpoint)
        {
            using var tape = new GradientTape<double>();
            var o = checkpoint
                ? GradientCheckpointing<double>.Checkpoint(new[] { seg }, x, segmentSize: 1)
                : seg(x);
            var loss = Engine.ReduceSum(Engine.TensorMultiply(o, outW), null);
            return tape.ComputeGradients(loss, new[] { x, w });
        }

        var eager = Run(false);
        var ckpt = Run(true);
        for (int i = 0; i < eager[w].Length; i++) Assert.Equal(eager[w][i], ckpt[w][i], 8);
        for (int i = 0; i < eager[x].Length; i++) Assert.Equal(eager[x][i], ckpt[x][i], 8);
    }

    [Fact]
    public void PackageCheckpoint_GradientsMatchEager_ForInputAndParameters()
    {
        // Master parameter values: resolve once, snapshot, then both runs use copies of these.
        var m0 = NewBlock();
        var m1 = NewBlock();
        var probe = Filled(new[] { 2, 5 }, start: -0.3, step: 0.017);
        using (new NoGradScope<double>()) { _ = m1.Forward(m0.Forward(probe)); }
        var p0 = m0.GetTrainableParameters().Select(t => t.Clone()).ToArray();
        var p1 = m1.GetTrainableParameters().Select(t => t.Clone()).ToArray();
        Assert.True(p0.Length + p1.Length > 0, "Blocks have no trainable parameters to check.");

        var eager = RunOnce(p0, p1, checkpoint: false);
        var ckpt = RunOnce(p0, p1, checkpoint: true);

        Assert.Equal(eager.Length, ckpt.Length);
        for (int i = 0; i < eager.Length; i++)
            Assert.Equal(eager[i], ckpt[i], 10);
    }
}
