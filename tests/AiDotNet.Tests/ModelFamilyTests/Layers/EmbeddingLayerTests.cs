using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class EmbeddingLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new EmbeddingLayer<double>(vocabularySize: 100, embeddingDimension: 16);

    // Embedding expects integer token indices (but as doubles)
    protected override int[] InputShape => [1, 4];

    // Embedding lookup is discrete — constant inputs like 0.1 and 0.9 both truncate to token 0
    protected override bool ExpectsDifferentOutputForConstantInputs => false;
}
