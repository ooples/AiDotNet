using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class FeedForwardNeuralNetworkTests : NeuralNetworkModelTestBase<float>
{
    // FFNN default: inputSize=128, outputSize=1, 1D tensors
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [1];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new FeedForwardNeuralNetwork<float>();

    // Gradcheck canary (#1872): FeedForwardNeuralNetwork is a plain Dense-stack, standard-forward
    // model with a known-correct backward, so it opts into the finite-difference gradcheck as the
    // reference that keeps the harness itself honest (validated: passes on the correct backward,
    // fails on an injected factor-of-2 gradient bug). Broader per-model enablement lands in Phase 2.
    protected override bool GradientCheckApplicable => true;
}
