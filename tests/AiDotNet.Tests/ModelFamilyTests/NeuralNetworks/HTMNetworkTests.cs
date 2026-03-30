using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class HTMNetworkTests : NeuralNetworkModelTestBase
{
    // HTM default: inputSize=128, output through Dense(->1) + Softmax
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [1]; // Dense(->1) + Softmax output from full layer chain

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new HTMNetwork<double>();
}
