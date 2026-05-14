using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class DeepBeliefNetworkTests : NeuralNetworkModelTestBase
{
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [1];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new DeepBeliefNetwork<double>();

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

    public override async Task DifferentInputs_AfterTraining_ShouldProduceDifferentOutputs()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var network = (DeepBeliefNetwork<double>)CreateNetwork();

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
        using var network = (DeepBeliefNetwork<double>)CreateNetwork();
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
}
