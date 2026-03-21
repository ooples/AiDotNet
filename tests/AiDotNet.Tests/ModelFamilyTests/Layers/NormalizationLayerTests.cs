using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class LayerNormalizationLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new LayerNormalizationLayer<double>(4);

    protected override int[] InputShape => [2, 4];
}
