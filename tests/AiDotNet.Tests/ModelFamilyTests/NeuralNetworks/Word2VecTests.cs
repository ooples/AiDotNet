using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Tests for Word2Vec per Mikolov et al. (2013)
/// "Efficient Estimation of Word Representations in Vector Space".
/// Default: inputSize=768 (embedding dim), outputSize=10000 (vocab size).
/// </summary>
public class Word2VecTests : NeuralNetworkModelTestBase
{
    // Input: word index encoded as one-hot or embedding lookup [768]
    protected override int[] InputShape => [768];
    // Output: vocabulary probability distribution [10000]
    protected override int[] OutputShape => [10000];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new Word2Vec<double>();
}
