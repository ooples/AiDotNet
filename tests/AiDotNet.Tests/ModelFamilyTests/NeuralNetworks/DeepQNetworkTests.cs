using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class DeepQNetworkTests : NeuralNetworkModelTestBase
{
    // DQN default: inputSize=4 (state), outputSize=2 (actions)
    protected override int[] InputShape => [4];
    protected override int[] OutputShape => [2];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new DeepQNetwork<double>();
}
