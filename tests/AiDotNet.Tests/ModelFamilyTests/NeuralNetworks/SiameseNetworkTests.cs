using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class SiameseNetworkTests : NeuralNetworkModelTestBase
{
    // SiameseNetwork needs [batch, 2, features] input for pair comparison
    // Default: inputSize=128, outputSize=1 (similarity score)
    protected override int[] InputShape => [1, 2, 128];
    protected override int[] OutputShape => [1];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new SiameseNetwork<double>();
}
