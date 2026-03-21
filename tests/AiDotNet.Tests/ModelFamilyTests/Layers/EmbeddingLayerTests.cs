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

    // Embedding lookup is discrete — constant inputs like 0.1 and 0.9 both round to 0
    // so DifferentInputs won't detect differences for small constants
    protected override double Tolerance => 0.5;
}
