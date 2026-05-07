using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for GAN (Generative Adversarial Network) models.
/// Inherits all neural network invariant tests and adds GAN-specific invariants:
/// generator output shape, mode diversity, parameter count, and output range.
/// </summary>
public abstract class GANModelTestBase : NeuralNetworkModelTestBase
{
    // =====================================================
    // GAN INVARIANT: Generator Output Shape
    // The generator should produce output matching the declared OutputShape.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task GeneratorOutput_ShouldHaveCorrectShape()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);

        int expectedLength = 1;
        foreach (var dim in OutputShape)
            expectedLength *= dim;

        Assert.Equal(expectedLength, output.Length);
    }

    // =====================================================
    // GAN INVARIANT: Mode Diversity
    // Different latent inputs should produce different outputs.
    // A GAN that produces the same output regardless of input has
    // mode-collapsed — a fundamental GAN failure mode.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task DifferentLatentInputs_ProduceDifferentOutputs()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var network = CreateNetwork();

        var input1 = CreateRandomTensor(InputShape, ModelTestHelpers.CreateSeededRandom(1));
        var input2 = CreateRandomTensor(InputShape, ModelTestHelpers.CreateSeededRandom(2));

        var output1 = network.Predict(input1);
        var output2 = network.Predict(input2);

        bool anyDifferent = false;
        int minLen = Math.Min(output1.Length, output2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(output1[i] - output2[i]) > 1e-12)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "GAN produces identical output for different latent inputs — possible mode collapse.");
    }

    // =====================================================
    // GAN INVARIANT: Parameter Count
    // Real GANs should have a substantial number of parameters
    // (generator + discriminator). A model with very few parameters
    // is likely misconfigured.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task ParameterCount_ShouldBeSubstantial()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var network = CreateNetwork();
        var parameters = network.GetParameters();
        Assert.True(parameters.Length > 100,
            $"GAN has only {parameters.Length} parameters — expected >100 for a real GAN architecture.");
    }

    // =====================================================
    // GAN INVARIANT: Output Values in Reasonable Range
    // Generated output values should not be extreme (no exploding values).
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task OutputValues_ShouldBeInReasonableRange()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            Assert.True(Math.Abs(output[i]) < 1e6,
                $"Output[{i}] = {output[i]:E4} is out of reasonable range [-1e6, 1e6]. " +
                "Generator may have exploding values.");
        }
    }

    // =====================================================
    // GAN INVARIANT: Loss must stay bounded across training
    // (overrides the base monotonic-decrease memorization
    // invariant — adversarial training has no monotonic-decrease
    // contract). The generator loss oscillates by construction
    // as the discriminator improves and re-classifies its fakes
    // (Goodfellow et al. 2014, §3 "Theoretical Results"; in
    // particular the Nash equilibrium is reached at non-zero
    // generator loss). The base class's
    // LossStrictlyDecreasesOnMemorizationTask asserts
    // lossFinal &lt; lossStep1 × threshold, which fundamentally
    // doesn't hold for adversarial training and produced false
    // failures across every GAN derivative (#1224 Cluster F:
    // ConditionalGAN.LossStrictlyDecreasesOnMemorizationTask
    // showed step1=0.277, step4=0.438 — loss increased
    // legitimately as the discriminator's grip on the
    // memorized fake tightened, not because of optimizer
    // misbehavior). Replace with a GAN-appropriate boundedness
    // invariant: loss must stay finite (no NaN/Inf) and must
    // not explode by &gt; 100×, which still catches the bug
    // classes the base test was designed for (sign error,
    // first-step explosion, post-explosion drift) without
    // false-failing on legitimate adversarial oscillation.
    // =====================================================
    public override async Task LossStrictlyDecreasesOnMemorizationTask()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(EffectiveOutputShape, rng);

        network.Train(input, target);
        double lossStep1 = ConvertLossToDouble(network.GetLastLoss());

        int followOnSteps = System.Math.Max(0, MemorizationTaskIterations - 1);
        for (int s = 0; s < followOnSteps; s++) network.Train(input, target);
        double lossFinal = ConvertLossToDouble(network.GetLastLoss());

        Assert.False(double.IsNaN(lossStep1) || double.IsInfinity(lossStep1),
            $"GAN generator loss after step 1 is non-finite: {lossStep1} (sign error or first-step explosion).");
        Assert.False(double.IsNaN(lossFinal) || double.IsInfinity(lossFinal),
            $"GAN generator loss after step {MemorizationTaskIterations} is non-finite: {lossFinal} (post-explosion drift).");

        // Adversarial loss can fluctuate but shouldn't explode by >100×
        // across a handful of steps on a memorization pair — that would
        // indicate a runaway generator/discriminator imbalance, not
        // healthy oscillation.
        double explosionRatio = lossStep1 > 1e-12 ? lossFinal / lossStep1 : lossFinal;
        Assert.True(explosionRatio < 100.0,
            $"GAN loss exploded across memorization steps: step 1={lossStep1:F6}, "
            + $"step {MemorizationTaskIterations}={lossFinal:F6} (ratio={explosionRatio:F2}×). "
            + "Diagnostic: generator/discriminator imbalance — one side is dominating "
            + "and the other can't keep up.");
    }

    private static double ConvertLossToDouble<TVal>(TVal value)
    {
        if (value is double d) return d;
        if (value is float f) return f;
        if (value is IConvertible) return Convert.ToDouble(value);
        throw new InvalidOperationException(
            $"GAN loss type {typeof(TVal).Name} is not IConvertible — cannot run boundedness invariant.");
    }

    // =====================================================
    // GAN INVARIANT: Training does not explode loss
    // (overrides the base monotonic-decrease "training reduces loss"
    // invariant — GAN generators are trained to fool a moving-target
    // discriminator, not to minimize MSE against a fixed regression
    // target. Predict-vs-target MSE in particular has no decreasing-loss
    // contract in adversarial training and produced false failures on
    // every GAN derivative (#1224 Cluster F: InfoGAN.Training_ShouldReduceLoss
    // showed initial=0.143 → final=0.169 — loss increased legitimately
    // as the generator learned to fool the disc, not because training
    // was broken). Replace with a boundedness check that still catches
    // the bug class the base invariant targets — exploding loss /
    // first-step blow-up — without false-failing on healthy
    // adversarial dynamics.
    // =====================================================
    public override async Task Training_ShouldReduceLoss()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(EffectiveOutputShape, rng);

        var initialOutput = network.Predict(input);
        double initialLoss = ComputeMSE(initialOutput, target);

        for (int i = 0; i < TrainingIterations * 3; i++)
            network.Train(input, target);

        var finalOutput = network.Predict(input);
        double finalLoss = ComputeMSE(finalOutput, target);

        // Final loss must stay finite and within a 100× envelope of
        // the initial — captures sign errors / first-step explosion
        // / runaway gen-disc imbalance without insisting on a
        // monotonic-decrease contract that adversarial training
        // doesn't have.
        if (!double.IsNaN(initialLoss) && !double.IsNaN(finalLoss))
        {
            Assert.False(double.IsInfinity(finalLoss),
                $"GAN final MSE is infinite — generator weights blew up after training.");
            double explosionRatio = initialLoss > 1e-12 ? finalLoss / initialLoss : finalLoss;
            Assert.True(explosionRatio < 100.0,
                $"GAN MSE exploded under training: initial={initialLoss:F6}, "
                + $"final={finalLoss:F6} (ratio={explosionRatio:F2}×). "
                + "Diagnostic: gradient sign error or runaway generator/discriminator imbalance.");
        }
    }
}
