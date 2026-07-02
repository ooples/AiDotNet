using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class RestrictedBoltzmannMachineTests : NeuralNetworkModelTestBase<float>
{
    // RBM default: visibleSize=128, hiddenSize=64
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [64];

    // RBM trains via contrastive divergence (Hinton 2006, "A Fast Learning
    // Algorithm for Deep Belief Nets" §3.3) which uses Gibbs sampling —
    // the reconstruction-error loss is intrinsically stochastic and CAN
    // step up between iterations even though the long-run trend decreases.
    // The default 1e-6 tolerance on Training_ShouldReduceLoss is too strict
    // for a CD-k regime over the handful of iterations the smoke suite
    // runs. 0.1 still catches a genuinely broken gradient (which diverges
    // by orders of magnitude) while tolerating the paper's sampling noise.
    protected override double TrainingLossReductionTolerance => 0.1;

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new RestrictedBoltzmannMachine<float>();

    // RBM trains via unsupervised Contrastive Divergence (Hinton 2006, §3.3): it minimizes
    // the RECONSTRUCTION error of its input and never optimizes toward an external target.
    // The base MoreData_ShouldNotDegrade measures MSE(Predict(input), randomTarget) — a metric
    // RBM does not and should not reduce. As CD sharpens the learned features, that arbitrary-
    // target MSE actually rises slightly and then plateaus (measured on this config:
    // reconstruction error falls monotonically 0.00093 @ 50 iters -> 0.000022 @ 200 -> 0.000004
    // @ 500, while MSE-vs-target drifts 0.1388 -> 0.1408 and plateaus ~0.140 — bounded, NOT
    // divergence, so the base's 1e-4 target-MSE bound false-fails on healthy training). Measure
    // the model's genuine objective instead, keeping this a load-bearing "more training must not
    // worsen learning" invariant: a truly diverging RBM blows reconstruction error up, which this
    // still catches. Pairs with the TrainingLossReductionTolerance override above (same root cause).
    public override async Task MoreData_ShouldNotDegrade()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var rng2 = ModelTestHelpers.CreateSeededRandom(42);

        // RBM weight init (seeded) and Gibbs sampling (now seeded via _samplingRandom) are both
        // deterministic, so two fresh instances are bit-identical at construction — the same
        // shared baseline the base test achieves by cloning.
        var rbmShort = new RestrictedBoltzmannMachine<float>();
        var rbmLong = new RestrictedBoltzmannMachine<float>();

        var input = CreateRandomTensor(InputShape, rng1);
        var target = CreateRandomTargetTensor(EffectiveOutputShape, rng1);
        var input2 = CreateRandomTensor(InputShape, rng2);
        var target2 = CreateRandomTargetTensor(EffectiveOutputShape, rng2);

        for (int i = 0; i < MoreDataShortIterations; i++) rbmShort.Train(input, target);
        double reconShort = System.Convert.ToDouble(rbmShort.ComputeReconstructionError(input));

        for (int i = 0; i < MoreDataLongIterations; i++) rbmLong.Train(input2, target2);
        double reconLong = System.Convert.ToDouble(rbmLong.ComputeReconstructionError(input2));

        Assert.False(double.IsNaN(reconShort) || double.IsNaN(reconLong),
            $"Reconstruction error became NaN during training: short={reconShort}, long={reconLong}.");
        Assert.True(reconLong <= reconShort + MoreDataTolerance,
            $"{MoreDataLongIterations} iterations reconstruction error ({reconLong:F6}) > "
            + $"{MoreDataShortIterations} iterations ({reconShort:F6}). CD training may be diverging with more steps.");
    }
}
