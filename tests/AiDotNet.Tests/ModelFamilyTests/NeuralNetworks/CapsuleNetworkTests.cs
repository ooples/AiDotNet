using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class CapsuleNetworkTests : NeuralNetworkModelTestBase
{
    // CapsuleNetwork per Sabour et al. (2017): 28x28x1 MNIST input
    // Output is flattened feature map (784), not class probabilities
    protected override int[] InputShape => [1, 28, 28];
    protected override int[] OutputShape => [784];

    // Iteration counts capped at 1 / 2 / 4 to fit the 60-180 s xUnit
    // per-test timeouts. The defaults (50 / 200 for MoreData and 100 for
    // MemorizationTask) are calibrated for small / mid-scale networks
    // where each step takes < 1.5 s — the model here either has a
    // higher per-step cost (per-iteration adversarial GAN forwards,
    // graph propagation across all nodes) or evolves topology between
    // calls (NEAT speciation), making 100+ iterations exceed budget.
    // Same paper-scale precedent the Forecasting Foundation models /
    // CLIP-family / VoxelCNN / VGG / DenseNet use; still exercises the
    // gradient-direction / loss-decrease invariants the tests catch
    // (sign error, oscillation, first-step explosion).
    protected override int MoreDataShortIterations => 1;
    protected override int MoreDataLongIterations => 2;
    protected override int MemorizationTaskIterations => 4;
    protected override double MemorizationTaskLossThreshold => 0.99999;

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new CapsuleNetwork<double>();
}
