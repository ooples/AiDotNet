using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class DenseLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new DenseLayer<double>(4, 8);

    protected override int[] InputShape => [1, 4];
}
