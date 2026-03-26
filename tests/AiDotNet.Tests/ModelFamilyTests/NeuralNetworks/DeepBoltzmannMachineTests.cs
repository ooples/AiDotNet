using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class DeepBoltzmannMachineTests : NeuralNetworkModelTestBase
{
    // DBM is generative: output matches visible (input) layer size
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [128];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new DeepBoltzmannMachine<double>();
}
