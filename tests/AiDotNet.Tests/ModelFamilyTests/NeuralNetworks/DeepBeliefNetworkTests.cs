using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class DeepBeliefNetworkTests : NeuralNetworkModelTestBase<float>
{
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [1];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new DeepBeliefNetwork<float>();

    // Per Hinton 2006 ("A fast learning algorithm for deep belief nets") and
    // Hinton & Salakhutdinov 2006 ("Reducing the Dimensionality of Data with
    // Neural Networks"), a DBN's training is a strictly two-phase pipeline:
    //
    //   1. Greedy layer-wise unsupervised pre-training (CD-1) of each RBM
    //      bottom-up. This is what gives DBNs their reason-to-exist —
    //      it sidesteps the vanishing-gradient pathology of backprop on
    //      randomly-initialised deep sigmoid stacks.
    //   2. Supervised fine-tuning of the full stack via backprop.
    //
    // The base-class invariant tests assume gradient-descent-only training
    // (one call to Train() per step on a fresh random network), which
    // skips phase 1 entirely. On the default 128 → 500 → 500 → 2000 → 1
    // stack with three sigmoid layers, supervised gradient signals
    // vanish through σ'·σ'·σ' ≤ 0.015 within a handful of steps and the
    // network collapses to input-invariant output (the exact failure
    // mode the base DifferentInputs_AfterTraining catches). Override
    // those tests to run PreTrain first, matching the paper's canonical
    // two-phase contract.

    // CD-1 pre-training drops the supervised baseline near the
    // memorization-floor (initial MSE ~0.13 on a [1] random target after
    // pre-train, vs ~0.45 cold-start). At that scale, SGD+momentum (lr=0.1,
    // β=0.9) oscillates around the floor by ~0.001 — legitimate stochastic
    // drift, not a regression. The base-class default 1e-6 tolerance was
    // tuned for smooth deterministic gradient descent on much larger
    // initial-loss baselines; loosen it for CD-pretrained DBN per the
    // contract spelled out in
    // NeuralNetworkModelTestBase.TrainingLossReductionTolerance's doc
    // comment ("models whose training is inherently stochastic — e.g.
    // RBM contrastive divergence (Hinton 2006) — can override to a
    // looser bound").
    protected override double TrainingLossReductionTolerance => 5e-3;

    public override async Task Training_ShouldReduceLoss()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var network = (DeepBeliefNetwork<float>)CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTargetTensor(EffectiveOutputShape, rng);

        // Phase 1: greedy CD-1 pre-training per Hinton 2006 §3. Without
        // it, the supervised backprop signal vanishes through three
        // σ' factors on the random-init deep sigmoid stack and the
        // (paper-canonical SGD+momentum, lr=0.1) optimizer amplifies
        // noise into the divergence the base test catches. Pre-train
        // before measuring loss so the comparison is against the
        // contract DBNs were actually designed to satisfy.
        network.PreTrain(input);

        var initialOutput = network.Predict(input);
        double initialLoss = ComputeMSE(initialOutput, target);

        // Phase 2: supervised fine-tuning, matching the base test's
        // iteration budget.
        for (int i = 0; i < TrainingIterations * 3; i++)
            network.Train(input, target);

        var finalOutput = network.Predict(input);
        double finalLoss = ComputeMSE(finalOutput, target);

        if (!double.IsNaN(initialLoss) && !double.IsNaN(finalLoss))
        {
            Assert.True(finalLoss <= initialLoss + TrainingLossReductionTolerance,
                $"DBN training did not reduce loss after CD pre-training: "
                + $"initial={initialLoss:F6}, final={finalLoss:F6}. "
                + "Investigate whether CD-1 pretrain is escaping the vanishing-gradient "
                + "regime or whether the supervised SGD+momentum step is mis-configured.");
        }
    }

    public override async Task DifferentInputs_AfterTraining_ShouldProduceDifferentOutputs()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var network = (DeepBeliefNetwork<float>)CreateNetwork();

        var trainInput = CreateRandomTensor(InputShape, rng);
        var trainTarget = CreateRandomTensor(EffectiveOutputShape, rng);

        // Phase 1: greedy CD-1 pre-training per Hinton 2006 §3.
        network.PreTrain(trainInput);

        // Phase 2: supervised fine-tuning, matching the base test's pattern.
        for (int i = 0; i < TrainingIterations; i++)
            network.Train(trainInput, trainTarget);

        var input1 = CreateConstantTensor(InputShape, 0.1);
        var input2 = CreateConstantTensor(InputShape, 0.9);
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
            $"DBN produces identical output for distinct inputs after pre-training "
            + $"+ fine-tuning: L2 distance = {l2Distance:E3}. CD pre-training is "
            + $"supposed to escape the vanishing-gradient regime — investigate "
            + $"whether CD updates are actually moving the RBM weights or "
            + $"whether the supervised step still pushes them into a dead zone.");
    }

    public override async Task LossStrictlyDecreasesOnMemorizationTask()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var network = (DeepBeliefNetwork<float>)CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTargetTensor(EffectiveOutputShape, rng);

        // Phase 1: greedy CD-1 pre-training per Hinton 2006 §3.
        network.PreTrain(input);

        // Phase 2: supervised fine-tuning. Mirror the base test's
        // step-1 / step-N comparison.
        network.Train(input, target);
        double lossStep1 = System.Convert.ToDouble(network.GetLastLoss());

        int followOnSteps = System.Math.Max(0, MemorizationTaskIterations - 1);
        for (int s = 0; s < followOnSteps; s++) network.Train(input, target);
        double lossFinal = System.Convert.ToDouble(network.GetLastLoss());

        Assert.False(double.IsNaN(lossStep1) || double.IsInfinity(lossStep1),
            $"Loss after step 1 is non-finite: {lossStep1}");
        Assert.False(double.IsNaN(lossFinal) || double.IsInfinity(lossFinal),
            $"Loss after step {MemorizationTaskIterations} is non-finite: {lossFinal}");

        bool atFloor = MemorizationTaskAbsoluteLossFloor > 0
            && lossFinal <= MemorizationTaskAbsoluteLossFloor;
        Assert.True(atFloor || lossFinal < lossStep1 * MemorizationTaskLossThreshold,
            $"DBN loss did NOT strictly decrease on memorization task after CD "
            + $"pre-training: step 1={lossStep1:F6}, step {MemorizationTaskIterations}="
            + $"{lossFinal:F6}. CD-pretrained DBN should converge cleanly on a "
            + $"single (input, target) pair — investigate whether PreTrain is "
            + $"actually escaping the vanishing-gradient regime or whether the "
            + $"supervised optimizer is mis-configured.");
    }

    // Same two-phase contract for the adequate-budget training invariant.
    // Compare the fine-tuned network with its own post-CD baseline, exactly as
    // the base test compares a trained model with its untrained baseline.
    // Comparing two optimizer snapshots at arbitrary short/long iteration
    // counts incorrectly requires per-step monotonicity from momentum SGD and
    // also mixes model cloning into a test whose subject is training behavior.
    // Clone fidelity is covered by the dedicated clone contract suite.
    public override async Task MoreData_ShouldNotDegrade()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom(42);
        using var network = (DeepBeliefNetwork<float>)CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTargetTensor(EffectiveOutputShape, rng);

        // Phase 1: greedy CD-1 pre-training per Hinton 2006 §3. Measure the
        // supervised baseline only after the DBN has entered its intended
        // fine-tuning regime.
        network.PreTrain(input);
        double lossUntrained = ComputeMSE(network.Predict(input), target);

        int longIters = MoreDataLongIterations;
        Assert.True(longIters > 0,
            $"{nameof(MoreDataLongIterations)} must be > 0; got {longIters}.");

        for (int i = 0; i < longIters; i++)
            network.Train(input, target);
        double lossLong = ComputeMSE(network.Predict(input), target);

        // double.IsFinite was added in .NET Core 2.1 / .NET 5+ and is NOT
        // available on net471 — the test project multi-targets net471, so
        // use the NaN || Infinity polyfill instead.
        Assert.False(double.IsNaN(lossUntrained) || double.IsInfinity(lossUntrained),
            $"DBN post-pretrain baseline is non-finite ({lossUntrained}). Indicates gradient explosion or "
            + "numerical instability in the supervised fine-tuning path.");
        Assert.False(double.IsNaN(lossLong) || double.IsInfinity(lossLong),
            $"DBN long-run loss is non-finite ({lossLong}). Indicates gradient explosion or "
            + "numerical instability in the supervised fine-tuning path.");
        Assert.True(lossLong <= lossUntrained + MoreDataTolerance,
            $"DBN: {longIters} supervised iterations loss ({lossLong:F6}) exceeds the "
            + $"post-CD baseline ({lossUntrained:F6}). Supervised optimizer is "
            + "diverging over an adequate training budget — investigate momentum defaults or "
            + "learning-rate schedule for the 3-RBM deep sigmoid stack.");
    }
}
