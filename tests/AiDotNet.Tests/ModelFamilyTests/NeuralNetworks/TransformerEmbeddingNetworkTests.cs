using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class TransformerEmbeddingNetworkTests : EmbeddingModelTestBase
{
    protected override INeuralNetworkModel<double> CreateNetwork()
        => new TransformerEmbeddingNetwork<double>();
}
