using AiDotNet.Interfaces;
using AiDotNet.Safety.Adversarial;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using System.Threading.Tasks;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual test scaffold for AdversarialImageEvaluator. Per Xu et al. 2018
/// (Feature Squeezing) the detector takes an NCHW image and produces a per-image
/// adversarial-detection score in [0, 1] via a learnable weighted ensemble of
/// three heuristic features (HF energy, histogram gaps, feature-squeezing
/// residual). The auto-generator can't construct it (parameterless ctor with
/// default-arg threshold isn't recognised); this manual class supplies the
/// ctor explicitly.
/// </summary>
public class AdversarialImageEvaluatorTests : NeuralNetworkModelTestBase<float>
{
    protected override int[] InputShape => [1, 3, 32, 32];
    protected override int[] OutputShape => [1, 1];

    /// <summary>
    /// AIE has only 4 learnable parameters (Dense(3 → 1) = 3 weights + 1 bias)
    /// trying to fit per-pixel random targets — it can't. The base
    /// <c>TrainingError_ShouldNotExceedTestError</c> invariant assumes a
    /// model with enough capacity to drive train-MSE meaningfully below
    /// test-MSE. AIE's tiny head will produce essentially the same
    /// sigmoid-bounded output for any feature vector close to the training
    /// one, so train/test MSE both end up near 0.25 (typical sigmoid-vs-
    /// uniform-random-target gap) with stochastic ordering. Use a permissive
    /// multiplier so the invariant catches the bug class it's designed for
    /// (training EXPLODES error — train MSE = 100 × test MSE) without
    /// false-failing on low-capacity-vs-random-target stochasticity.
    /// </summary>
    protected override double TrainingErrorMultiplier => 100.0;

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new AdversarialImageEvaluator<float>(threshold: 0.5);

    /// <summary>
    /// Override the base "different uniform inputs → different outputs"
    /// invariant. AIE's three statistical features (HF energy, histogram
    /// smoothness, feature-squeezing residual) are ZERO by mathematical
    /// construction for any uniform image: no high-frequency content,
    /// single-bin smooth histogram, identity bit-depth quantization. The
    /// base test's <c>CreateConstantTensor(0.1)</c> vs
    /// <c>CreateConstantTensor(0.9)</c> both produce feature vector
    /// <c>[0, 0, 0]</c> → identical Dense(3 → 1) → identical sigmoid
    /// output. That isn't a model bug, it's a paper-faithful invariance
    /// (Xu et al. §3 explicitly position the detector against
    /// adversarial perturbations, which inject HF/histogram structure).
    /// Use varied inputs that actually exercise the heuristics.
    /// </summary>
    [Fact(Timeout = 120000)]
    public override async Task DifferentInputs_ShouldProduceDifferentOutputs()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        using var network = CreateNetwork();
        var rng1 = ModelTestHelpers.CreateSeededRandom();
        var rng2 = ModelTestHelpers.CreateSeededRandom(seed: 1729);

        var input1 = CreateRandomTensor(InputShape, rng1);
        var input2 = CreateRandomTensor(InputShape, rng2);

        var output1 = network.Predict(input1);
        var output2 = network.Predict(input2);

        double sumSquared = 0;
        int minLen = System.Math.Min(output1.Length, output2.Length);
        for (int i = 0; i < minLen; i++)
        {
            double d = output1[i] - output2[i];
            sumSquared += d * d;
        }
        double l2Distance = System.Math.Sqrt(sumSquared);

        Assert.True(l2Distance > 1e-9,
            $"AIE produces identical output for varied random inputs: " +
            $"L2 distance = {l2Distance:E3}. Either the feature heuristics " +
            $"collapsed all variation to zero (HF / histogram / squeezing " +
            $"computations are broken) or Dense(3 → 1) weights cancel the " +
            $"feature differences out.");
    }

    /// <summary>
    /// Same rationale as <see cref="DifferentInputs_ShouldProduceDifferentOutputs"/>:
    /// the base implementation feeds two CONSTANT tensors at the assertion
    /// step, which AIE's image-statistics features map to identical
    /// <c>[0, 0, 0]</c>. Override with varied random inputs so the
    /// post-training output-divergence assertion is testing what it
    /// claims to test.
    /// </summary>
    [Fact(Timeout = 120000)]
    public override async Task DifferentInputs_AfterTraining_ShouldProduceDifferentOutputs()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var network = CreateNetwork();

        var trainInput = CreateRandomTensor(InputShape, rng);
        var trainTarget = CreateRandomTargetTensor(EffectiveOutputShape, rng);
        for (int i = 0; i < TrainingIterations; i++)
            network.Train(trainInput, trainTarget);

        var rng1 = ModelTestHelpers.CreateSeededRandom();
        var rng2 = ModelTestHelpers.CreateSeededRandom(seed: 1729);
        var input1 = CreateRandomTensor(InputShape, rng1);
        var input2 = CreateRandomTensor(InputShape, rng2);

        var output1 = network.Predict(input1);
        var output2 = network.Predict(input2);

        double sumSquared = 0;
        int minLen = System.Math.Min(output1.Length, output2.Length);
        for (int i = 0; i < minLen; i++)
        {
            double d = output1[i] - output2[i];
            sumSquared += d * d;
        }
        double l2Distance = System.Math.Sqrt(sumSquared);

        Assert.True(l2Distance > 1e-9,
            $"AIE produces identical output for varied inputs AFTER training: " +
            $"L2 distance = {l2Distance:E3}. The single Dense(3 → 1) head " +
            $"collapsed to a constant during training, or gradient flow " +
            $"into Dense weights is broken.");
    }
}
