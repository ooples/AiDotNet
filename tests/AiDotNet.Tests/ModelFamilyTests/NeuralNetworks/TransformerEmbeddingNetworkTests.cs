using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class TransformerEmbeddingNetworkTests : EmbeddingModelTestBase
{
    protected override int[] InputShape => [768];
    protected override int[] OutputShape => [768];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new TransformerEmbeddingNetwork<double>();
}
