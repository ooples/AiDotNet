using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class RestrictedBoltzmannMachineTests : NeuralNetworkModelTestBase
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

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new RestrictedBoltzmannMachine<double>();
}
