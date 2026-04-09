using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class SimCSETests : NeuralNetworkModelTestBase
{
    // Per Gao et al. (2021): SimCSE outputs [CLS] embeddings of size embeddingDimension (768)
    protected override int[] InputShape => [1, 768];
    protected override int[] OutputShape => [1, 768];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new SimCSE<double>();
}
