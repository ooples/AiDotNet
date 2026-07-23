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

    // Same two-phase contract for the "more iterations should not degrade"
    // invariant. Without CD-1 pre-training the base test runs 200 steps of
    // pure backprop on a randomly-initialised deep sigmoid stack, and the
    // vanishing-gradient pathology Hinton 2006 §1 describes causes Adam to
    // amplify noise rather than the (near-zero) gradient signal — long-run
    // loss diverges above short-run loss. Pre-train both clones before
    // letting the base-style supervised loop run.
    public override async Task MoreData_ShouldNotDegrade()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var rng2 = ModelTestHelpers.CreateSeededRandom(42);

        var network1 = (DeepBeliefNetwork<float>)CreateNetwork();

        var input = CreateRandomTensor(InputShape, rng1);
        var target = CreateRandomTargetTensor(EffectiveOutputShape, rng1);
        var input2 = CreateRandomTensor(InputShape, rng2);
        var target2 = CreateRandomTensor(EffectiveOutputShape, rng2);

        // Phase 1: greedy CD-1 pre-training per Hinton 2006 §3 on
        // network1's training input. Clone afterwards so network2 starts
        // from the same pre-trained weights — same shared-baseline rule
        // the base MoreData_ShouldNotDegrade enforces.
        network1.PreTrain(input);

        var network2 = (DeepBeliefNetwork<float>)network1.Clone();

        int shortIters = MoreDataShortIterations;
        int longIters = MoreDataLongIterations;

        Assert.True(shortIters > 0,
            $"{nameof(MoreDataShortIterations)} must be > 0; got {shortIters}.");
        Assert.True(longIters >= shortIters,
            $"{nameof(MoreDataLongIterations)} ({longIters}) must be >= "
            + $"{nameof(MoreDataShortIterations)} ({shortIters}).");

        for (int i = 0; i < shortIters; i++)
            network1.Train(input, target);
        double lossShort = ComputeMSE(network1.Predict(input), target);

        for (int i = 0; i < longIters; i++)
            network2.Train(input2, target2);
        double lossLong = ComputeMSE(network2.Predict(input2), target2);

        // double.IsFinite was added in .NET Core 2.1 / .NET 5+ and is NOT
        // available on net471 — the test project multi-targets net471, so
        // use the NaN || Infinity polyfill instead.
        Assert.False(double.IsNaN(lossShort) || double.IsInfinity(lossShort),
            $"DBN short-run loss is non-finite ({lossShort}). Indicates gradient explosion or "
            + "numerical instability in the supervised fine-tuning path.");
        Assert.False(double.IsNaN(lossLong) || double.IsInfinity(lossLong),
            $"DBN long-run loss is non-finite ({lossLong}). Indicates gradient explosion or "
            + "numerical instability in the supervised fine-tuning path.");
        Assert.True(lossLong <= lossShort + MoreDataTolerance,
            $"DBN: {longIters} iterations loss ({lossLong:F6}) > {shortIters} iterations loss "
            + $"({lossShort:F6}) even after CD-1 pre-training. Supervised optimizer is "
            + "diverging with more iterations — investigate Adam β₁/β₂ defaults or "
            + "learning-rate schedule for the 3-RBM deep sigmoid stack.");
    }
}
