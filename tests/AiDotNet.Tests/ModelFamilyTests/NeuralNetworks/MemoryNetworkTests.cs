using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class MemoryNetworkTests : NeuralNetworkModelTestBase<float>
{
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [128];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new MemoryNetwork<float>();
}
