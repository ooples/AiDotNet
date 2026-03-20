using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class HTMNetworkTests : NeuralNetworkModelTestBase
{
    // HTM default: inputSize=128, output is sparse distributed representation
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [4097]; // SDR output size from temporal memory

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new HTMNetwork<double>();
}
