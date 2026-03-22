using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class BidirectionalLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new BidirectionalLayer<double>(
            new RecurrentLayer<double>(inputSize: 4, hiddenSize: 8,
                activationFunction: new TanhActivation<double>()),
            mergeMode: true,
            activationFunction: new TanhActivation<double>() as IActivationFunction<double>);
    protected override int[] InputShape => [1, 4];
}

public class ConvLSTMLayerTests : LayerTestBase
{
    // ConvLSTM per Shi et al. 2015: inputShape is NHWC [batch, H, W, C]
    protected override ILayer<double> CreateLayer()
        => new ConvLSTMLayer<double>(inputShape: [1, 4, 4, 1], kernelSize: 3, filters: 2,
            padding: 1, strides: 1,
            activationFunction: new TanhActivation<double>() as IActivationFunction<double>);
    protected override int[] InputShape => [1, 4, 4, 1]; // NHWC
}
