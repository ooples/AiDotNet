using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.NeuralNetworks;

public sealed class TransformerDecoderInitializationSeedTests
{
    [Fact]
    public void LazyChildren_UseParentSeed_AfterConstructionScopeEnds()
    {
        const int seed = 1729;
        var previousFallback = LayerInitializationSeedScope.AmbientFallbackSeed;

        try
        {
            var first = CreateSeededDecoder(seed);
            LayerInitializationSeedScope.ResetForModelConstruction(null);
            Initialize(first);

            for (int i = 0; i < 1_000; i++)
                _ = RandomHelper.ThreadSafeRandom.Next();

            var second = CreateSeededDecoder(seed);
            LayerInitializationSeedScope.ResetForModelConstruction(null);
            Initialize(second);

            var firstParameters = first.GetParameters();
            var secondParameters = second.GetParameters();

            Assert.Equal(firstParameters.Length, secondParameters.Length);
            for (int i = 0; i < firstParameters.Length; i++)
                Assert.Equal(firstParameters[i], secondParameters[i]);
        }
        finally
        {
            LayerInitializationSeedScope.AmbientFallbackSeed = previousFallback;
            LayerInitializationSeedScope.ResetForModelConstruction(null);
        }
    }

    private static TransformerDecoderLayer<double> CreateSeededDecoder(int seed)
    {
        LayerInitializationSeedScope.AmbientFallbackSeed = seed;
        LayerInitializationSeedScope.ResetForModelConstruction(null);
        var decoder = new TransformerDecoderLayer<double>(
            numHeads: 2,
            feedForwardDim: 8,
            sequenceLength: 2);
        LayerInitializationSeedScope.AmbientFallbackSeed = null;
        return decoder;
    }

    private static void Initialize(TransformerDecoderLayer<double> decoder)
    {
        var input = new Tensor<double>([1, 2, 4]);
        for (int i = 0; i < input.Length; i++)
            input[i] = (i + 1) * 0.01;

        _ = decoder.Forward(input);
    }
}
