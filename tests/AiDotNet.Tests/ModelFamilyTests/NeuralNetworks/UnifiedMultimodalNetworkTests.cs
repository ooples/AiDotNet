using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class UnifiedMultimodalNetworkTests : NeuralNetworkModelTestBase<float>
{
    // Default: inputSize=768 (embedding dim), outputSize=100
    protected override int[] InputShape => [768];
    protected override int[] OutputShape => [100];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new UnifiedMultimodalNetwork<float>();
}
