using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class ColBERTTests : NeuralNetworkModelTestBase
{
    // ColBERT (Khattab & Zaharia 2020) projects 768-dim BERT embeddings
    // down to 128-dim for late interaction retrieval.
    // Uses full paper parameters: 12 layers, 12 heads, 768 hidden, 3072 FFN.
    protected override int[] InputShape => [768];
    protected override int[] OutputShape => [128];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new ColBERT<double>();
}
