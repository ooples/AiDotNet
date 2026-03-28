using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Tests for FastText per Bojanowski et al. (2017)
/// "Enriching Word Vectors with Subword Information".
/// Default: inputSize=768 (embedding dim), outputSize=10000 (vocab size).
/// </summary>
public class FastTextTests : NeuralNetworkModelTestBase
{
    protected override int[] InputShape => [768];
    protected override int[] OutputShape => [10000];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new FastText<double>();
}
