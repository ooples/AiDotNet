using System;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Helpers;

[Trait("category", "unit")]
public sealed class PolicyDistributionNumericsTests
{
    [Fact]
    public void Half_ExtremeLogits_UnselectedActionDoesNotPoisonLogProbOrGradients()
    {
        var engine = AiDotNetEngine.Current;
        var logits = new Tensor<Half>(
            new[] { (Half)0, (Half)(-100) },
            new[] { 1, 2 });

        using var tape = new GradientTape<Half>();
        var logProbs = PolicyDistributionHelper<Half>.ComputeDiscreteLogProb(engine, logits, new[] { 0 });
        var grads = tape.ComputeGradients(logProbs, new[] { logits });

        Assert.Single(logProbs.ToArray());
        double selectedLogProb = (double)logProbs[0];
        Assert.False(double.IsNaN(selectedLogProb));
        Assert.False(double.IsInfinity(selectedLogProb));
        Assert.Equal(0.0, selectedLogProb, 6);

        Assert.True(grads.TryGetValue(logits, out var logitGrad));
        Assert.NotNull(logitGrad);
        foreach (Half g in logitGrad.ToArray())
        {
            Assert.False(Half.IsNaN(g));
            Assert.False(Half.IsInfinity(g));
            Assert.Equal(0.0, (double)g, 6);
        }
    }

    [Fact]
    public void Half_ExtremeLogits_SelectedLowProbabilityAction_PreservesExactLogProbAndNonZeroGradients()
    {
        var engine = AiDotNetEngine.Current;
        var logits = new Tensor<Half>(
            new[] { (Half)0, (Half)(-100) },
            new[] { 1, 2 });

        using var tape = new GradientTape<Half>();
        var logProbs = PolicyDistributionHelper<Half>.ComputeDiscreteLogProb(engine, logits, new[] { 1 });
        var grads = tape.ComputeGradients(logProbs, new[] { logits });

        double selectedLogProb = (double)logProbs[0];
        Assert.False(double.IsNaN(selectedLogProb));
        Assert.False(double.IsInfinity(selectedLogProb));
        // Exact log-softmax without an arbitrary epsilon floor: log_softmax([0, -100])[1] == -100.
        Assert.Equal(-100.0, selectedLogProb, 3);

        Assert.True(grads.TryGetValue(logits, out var logitGrad));
        Assert.NotNull(logitGrad);
        Assert.Equal(-1.0, (double)logitGrad[0, 0], 3);
        Assert.Equal(1.0, (double)logitGrad[0, 1], 3);
    }

    [Fact]
    public void Double_ExtremeLogits_HasNoEpsilonFloorClamp()
    {
        var engine = AiDotNetEngine.Current;
        var logits = new Tensor<double>(new[] { 0.0, -50.0 }, new[] { 1, 2 });

        var logProbs = PolicyDistributionHelper<double>.ComputeDiscreteLogProb(engine, logits, new[] { 1 });

        // With a 1e-8 probability floor, this clamped to log(1e-8) ≈ -18.42068 instead of -50.0.
        Assert.Equal(-50.0, logProbs[0], 12);
    }

    [Fact]
    public void Half_MaskedNegativeInfinityLogits_LogProbAndEntropy_KeepGradientsFiniteAndMaskedLaneZero()
    {
        var engine = AiDotNetEngine.Current;
        var logits = new Tensor<Half>(
            new[] { (Half)0, (Half)(-1), Half.NegativeInfinity },
            new[] { 1, 3 });

        double p0 = 1.0 / (1.0 + Math.Exp(-1.0));
        double p1 = Math.Exp(-1.0) / (1.0 + Math.Exp(-1.0));
        double expectedLogProb0 = Math.Log(p0);
        double expectedEntropy = -(p0 * Math.Log(p0) + p1 * Math.Log(p1));

        using (var logProbTape = new GradientTape<Half>())
        {
            var logProbs = PolicyDistributionHelper<Half>.ComputeDiscreteLogProb(engine, logits, new[] { 0 });
            var grads = logProbTape.ComputeGradients(logProbs, new[] { logits });

            Assert.Equal(expectedLogProb0, (double)logProbs[0], 2);
            Assert.True(grads.TryGetValue(logits, out var g));
            Assert.NotNull(g);
            Assert.Equal(1.0 - p0, (double)g[0, 0], 2);
            Assert.Equal(-p1, (double)g[0, 1], 2);
            Assert.Equal(0.0, (double)g[0, 2], 6);
        }

        using (var zeroSeedTape = new GradientTape<Half>())
        {
            var logProbs = PolicyDistributionHelper<Half>.ComputeDiscreteLogProb(engine, logits, new[] { 0 });
            var zeroScaled = engine.TensorMultiplyScalar(logProbs, (Half)0);
            var grads = zeroSeedTape.ComputeGradients(zeroScaled, new[] { logits });

            Assert.True(grads.TryGetValue(logits, out var g));
            Assert.NotNull(g);
            foreach (Half value in g.ToArray())
            {
                Assert.False(Half.IsNaN(value));
                Assert.Equal(0.0, (double)value, 6);
            }
        }

        using (var entropyTape = new GradientTape<Half>())
        {
            var entropy = PolicyDistributionHelper<Half>.ComputeDiscreteEntropy(engine, logits);
            var grads = entropyTape.ComputeGradients(entropy, new[] { logits });

            Assert.Equal(expectedEntropy, (double)entropy[0], 2);
            Assert.True(grads.TryGetValue(logits, out var g));
            Assert.NotNull(g);
            double expectedGrad0 = -p0 * (Math.Log(p0) + expectedEntropy);
            double expectedGrad1 = -p1 * (Math.Log(p1) + expectedEntropy);
            Assert.Equal(expectedGrad0, (double)g[0, 0], 2);
            Assert.Equal(expectedGrad1, (double)g[0, 1], 2);
            Assert.Equal(0.0, (double)g[0, 2], 6);
        }
    }
}
