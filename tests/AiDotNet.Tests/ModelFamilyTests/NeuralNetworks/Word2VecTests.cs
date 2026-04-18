using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class Word2VecTests : NeuralNetworkModelTestBase
{
    // Word2Vec's default ctor uses vocabSize=10000 — the last layer emits
    // a 10000-dim softmax over the vocabulary, so predicted output length
    // is 10000 per sample, not the [1, 1] implied by the base-class
    // default OutputShape. Align both sides so
    // OutputDimension_ShouldMatchExpectedShape compares like with like.
    protected override int[] InputShape => [1, 4];
    protected override int[] OutputShape => [1, 10000];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new Word2Vec<double>();
}
