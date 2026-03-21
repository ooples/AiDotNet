using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class DenseBlockLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new DenseBlockLayer<double>(inputChannels: 4, growthRate: 4, height: 4, width: 4);
    protected override int[] InputShape => [1, 4, 4, 4];
}

public class DenseBlockTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new DenseBlock<double>(inputChannels: 4, numLayers: 2, growthRate: 4,
            inputHeight: 4, inputWidth: 4);
    protected override int[] InputShape => [1, 4, 4, 4];
}

public class TransitionLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new TransitionLayer<double>(inputChannels: 4, inputHeight: 4, inputWidth: 4);
    protected override int[] InputShape => [1, 4, 4, 4];
}

public class InvertedResidualBlockTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new InvertedResidualBlock<double>(inChannels: 4, outChannels: 4,
            inputHeight: 8, inputWidth: 8, expansionRatio: 2);
    protected override int[] InputShape => [1, 4, 8, 8];
}
