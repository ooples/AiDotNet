using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class CapsuleNetworkTests : NeuralNetworkModelTestBase<float>
{
    // CapsuleNetwork per Sabour et al. 2017 ("Dynamic Routing Between Capsules"):
    // 28×28×1 MNIST input. Output is the flattened final feature map.
    protected override int[] InputShape => [1, 28, 28];
    protected override int[] OutputShape => [784];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new CapsuleNetwork<float>();
}
