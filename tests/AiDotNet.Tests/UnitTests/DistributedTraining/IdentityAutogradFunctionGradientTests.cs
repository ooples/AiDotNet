using System;
using AiDotNet.DistributedTraining;
using AiDotNet.DistributedTraining.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.DistributedTraining;

/// <summary>
/// The tensor-parallel and FSDP conjugate operators are identity in the forward pass. Recorded on the tape they must
/// behave as an identity in the backward pass too: the gradient that reaches the tensor they were applied to, and
/// everything upstream of it, is the gradient that reached their output -- once.
/// </summary>
/// <remarks>
/// AiDotNet.Tensors' <c>AutogradFunction.Apply</c> keys the recorded step by the tensor Forward returned. When that
/// was the input itself, the step's output and input were one key: the input's gradient was added to itself (2x), and
/// the step took the place of the input's own producer, so an upstream op received no gradient at all.
/// </remarks>
public class IdentityAutogradFunctionGradientTests
{
    private static InMemoryCommunicationBackend<double> SingleRank() => new(0, 1);

    private static Tensor<double> Input() => new(new[] { 0.5, -1.25, 2.0, 0.75 }, new[] { 2, 2 });

    // loss = sum(f(3x)^2) with f the identity => d loss / d x = 18x. The multiply by 3 is the upstream producer.
    private static double[] GradientThroughUpstreamOp(Func<Tensor<double>, Tensor<double>> identityOp, Tensor<double> x)
    {
        var engine = AiDotNetEngine.Current;
        using var tape = new GradientTape<double>();
        var upstream = engine.TensorMultiplyScalar(x, 3.0);
        var y = identityOp(upstream);
        var loss = engine.ReduceSum(engine.TensorMultiply(y, y), new[] { 0, 1 }, keepDims: false);
        var grads = tape.ComputeGradients(loss, new[] { x });
        Assert.True(grads.TryGetValue(x, out var g), "the input upstream of the identity op received no gradient");
        return g!.ToArray();
    }

    private static void AssertEighteenX(Tensor<double> x, double[] gradient)
    {
        var values = x.ToArray();
        Assert.Equal(values.Length, gradient.Length);
        for (int i = 0; i < values.Length; i++) Assert.Equal(18.0 * values[i], gradient[i], 10);
    }

    [Fact]
    public void CopyToTensorParallelRegion_PassesTheGradientUpstreamOnce()
    {
        var x = Input();
        AssertEighteenX(x, GradientThroughUpstreamOp(t => new CopyToTensorParallelRegion<double>(SingleRank()).Apply(t), x));
    }

    [Fact]
    public void ReduceFromTensorParallelRegion_SingleRank_PassesTheGradientUpstreamOnce()
    {
        var x = Input();
        AssertEighteenX(x, GradientThroughUpstreamOp(t => new ReduceFromTensorParallelRegion<double>(SingleRank()).Apply(t), x));
    }

    [Fact]
    public void GatherFromTensorParallelRegion_SingleRank_PassesTheGradientUpstreamOnce()
    {
        var x = Input();
        AssertEighteenX(x, GradientThroughUpstreamOp(t => new GatherFromTensorParallelRegion<double>(SingleRank(), 2).Apply(t), x));
    }

    [Fact]
    public void AverageGradientAcrossRanks_PassesTheGradientUpstreamOnce()
    {
        var x = Input();
        AssertEighteenX(x, GradientThroughUpstreamOp(t => new AverageGradientAcrossRanks<double>(SingleRank()).Apply(t), x));
    }

    [Fact]
    public void IdentityOp_AppliedToALeaf_GivesTheLeafItsGradientOnce()
    {
        // The bias of a Stage-3 sharded linear is a leaf routed straight through AverageGradientAcrossRanks.
        var engine = AiDotNetEngine.Current;
        var bias = Input();
        using var tape = new GradientTape<double>();
        var synced = new AverageGradientAcrossRanks<double>(SingleRank()).Apply(bias);
        var loss = engine.ReduceSum(engine.TensorMultiply(synced, synced), new[] { 0, 1 }, keepDims: false);
        var grads = tape.ComputeGradients(loss, new[] { bias });

        Assert.True(grads.TryGetValue(bias, out var g));
        var values = bias.ToArray();
        var gradient = g!.ToArray();
        for (int i = 0; i < values.Length; i++) Assert.Equal(2.0 * values[i], gradient[i], 10);
    }

    [Fact]
    public void IdentityOp_OutputSharesTheInputStorage()
    {
        // The forward stays a no-op: no copy of the activations, and an in-place update of the input stays visible.
        var x = Input();
        var y = new CopyToTensorParallelRegion<double>(SingleRank()).Apply(x);
        Assert.NotSame(x, y);
        x[0, 0] = 42.0;
        Assert.Equal(42.0, y[0, 0]);
    }
}
