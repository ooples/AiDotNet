using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class ColBERTTests : NeuralNetworkModelTestBase
{
    // ColBERT projects 768-dim BERT embeddings down to 128-dim for late interaction
    protected override int[] InputShape => [768];
    protected override int[] OutputShape => [128];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new ColBERT<double>();
}
