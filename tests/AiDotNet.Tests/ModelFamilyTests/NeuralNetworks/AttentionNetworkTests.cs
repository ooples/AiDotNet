using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class AttentionNetworkTests : NeuralNetworkModelTestBase
{
    // AttentionNetwork default: inputSize=128, outputSize=128, 1D tensors
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [128];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new AttentionNetwork<double>();
}
