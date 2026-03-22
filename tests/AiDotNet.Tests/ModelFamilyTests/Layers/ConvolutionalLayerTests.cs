using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class ConvolutionalLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new ConvolutionalLayer<double>(
            inputDepth: 1, inputHeight: 8, inputWidth: 8, outputDepth: 4, kernelSize: 3);

    // Conv expects [batch, channels, height, width]
    protected override int[] InputShape => [1, 1, 8, 8];
}
