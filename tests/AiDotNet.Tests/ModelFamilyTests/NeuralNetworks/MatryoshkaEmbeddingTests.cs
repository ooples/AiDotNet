using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class MatryoshkaEmbeddingTests : EmbeddingModelTestBase
{
    // MatryoshkaEmbedding (Kusupati et al. 2022 §3) emits a maxEmbeddingDimension-
    // wide vector at full resolution; nested sub-vectors of decreasing length
    // share the same forward pass. The parameterless ctor sets
    // maxEmbeddingDimension=1536, so a fresh network produces [1, 1536]
    // outputs and the test base's MSE target must match that shape.
    protected override int[] InputShape => [1, 4];
    protected override int[] OutputShape => [1, 1536];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new MatryoshkaEmbedding<double>();
}
