using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class EchoStateNetworkTests : NeuralNetworkModelTestBase<float>
{
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [1];

    // ESN training is Jaeger 2001 §3.4 closed-form ridge regression on the
    // readout, NOT iterative gradient descent. The first Train(input,
    // target) call jumps the readout discretely from a random
    // initialization to the ridge solution, so the post-train L2 norm
    // can legitimately be much smaller than the pre-train (random) L2.
    // The [0.5×, 2×] default bound exists to catch Adam first-step
    // explosion / wrong-sign updates in gradient trainers; it doesn't
    // model closed-form solver semantics. Disable the lower bound and
    // keep the upper bound — any post-train L2 explosion in a ridge
    // solve still indicates a real bug (e.g. λ→0 ill-conditioned solve
    // producing huge weights).
    protected override double OptimizerStepL2LowerBound => 0.0;

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new EchoStateNetwork<float>();
}
