using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class VariationalAutoencoderTests : NeuralNetworkModelTestBase
{
    // VAE default: inputSize=128, outputSize=128 (reconstructs input)
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [128];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new VariationalAutoencoder<double>();
}
