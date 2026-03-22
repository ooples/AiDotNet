using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class SpikingLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new SpikingLayer<double>(inputSize: 4, outputSize: 8);

    protected override int[] InputShape => [4]; // SpikingLayer expects 1D input
}
