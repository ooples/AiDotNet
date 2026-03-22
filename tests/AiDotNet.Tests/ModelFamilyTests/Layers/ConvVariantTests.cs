using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class DepthwiseSeparableConvLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new DepthwiseSeparableConvolutionalLayer<double>(
            inputDepth: 2, outputDepth: 4, kernelSize: 3,
            inputHeight: 8, inputWidth: 8, stride: 1, padding: 0,
            activation: new AiDotNet.ActivationFunctions.ReLUActivation<double>() as IActivationFunction<double>);
    protected override int[] InputShape => [1, 2, 8, 8];
}

public class DilatedConvolutionalLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new DilatedConvolutionalLayer<double>(
            inputDepth: 1, outputDepth: 2, kernelSize: 3, inputHeight: 8, inputWidth: 8,
            dilation: 2, stride: 1, padding: 0,
            activation: new AiDotNet.ActivationFunctions.ReLUActivation<double>() as IActivationFunction<double>);
    protected override int[] InputShape => [1, 1, 8, 8];
}
