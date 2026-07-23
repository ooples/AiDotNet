using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class DeepQNetworkTests : NeuralNetworkModelTestBase<float>
{
    // DQN default: inputSize=4 (state), outputSize=2 (actions)
    protected override int[] InputShape => [4];
    protected override int[] OutputShape => [2];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new DeepQNetwork<float>();
}
