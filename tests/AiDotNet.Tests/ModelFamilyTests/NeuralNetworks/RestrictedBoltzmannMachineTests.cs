using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class RestrictedBoltzmannMachineTests : NeuralNetworkModelTestBase
{
    // RBM default: visibleSize=128, hiddenSize=64
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [64];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new RestrictedBoltzmannMachine<double>();
}
