using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class DeepBoltzmannMachineTests : NeuralNetworkModelTestBase
{
    protected override INeuralNetworkModel<double> CreateNetwork()
        => new DeepBoltzmannMachine<double>();
}
