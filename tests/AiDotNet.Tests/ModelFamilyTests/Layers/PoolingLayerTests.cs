using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class MaxPoolingLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new MaxPoolingLayer<double>(inputShape: [1, 4, 4], poolSize: 2, stride: 2);

    protected override int[] InputShape => [1, 1, 4, 4];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
}
