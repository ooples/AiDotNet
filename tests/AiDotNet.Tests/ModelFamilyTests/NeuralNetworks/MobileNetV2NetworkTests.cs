using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class MobileNetV2NetworkTests : NeuralNetworkModelTestBase
{
    protected override int[] InputShape => [3, 64, 64];
    protected override int[] OutputShape => [10];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new MobileNetV2Network<double>();
}
