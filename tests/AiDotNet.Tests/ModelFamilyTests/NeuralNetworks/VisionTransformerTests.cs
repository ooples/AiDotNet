using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class VisionTransformerTests : NeuralNetworkModelTestBase
{
    protected override int[] InputShape => [1, 3, 32, 32];
    protected override int[] OutputShape => [1000];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new VisionTransformer<double>();
}
