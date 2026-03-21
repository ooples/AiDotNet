using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class BatchNormalizationLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new BatchNormalizationLayer<double>(4);

    protected override int[] InputShape => [2, 4]; // BatchNorm needs batch > 1 for training stats
}
