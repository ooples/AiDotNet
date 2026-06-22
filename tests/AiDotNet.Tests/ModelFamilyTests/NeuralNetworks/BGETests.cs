using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class BGETests : NeuralNetworkModelTestBase<float>
{
    // BGE embedding model: input/output both use BERT-base 768-dim embeddings
    protected override int[] InputShape => [768];
    protected override int[] OutputShape => [768];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new BGE<float>();
}
