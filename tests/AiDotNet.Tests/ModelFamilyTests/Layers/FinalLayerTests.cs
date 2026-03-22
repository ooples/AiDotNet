using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class ContinuumMemorySystemLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new ContinuumMemorySystemLayer<double>(inputShape: [4], hiddenDim: 8);
    protected override int[] InputShape => [1, 4];
}

public class DeformableConvolutionalLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new DeformableConvolutionalLayer<double>(
            inputHeight: 8, inputWidth: 8, inputChannels: 1, outputChannels: 2, kernelSize: 3);
    protected override int[] InputShape => [1, 1, 8, 8];
}

public class MixtureOfExpertsLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
    {
        var experts = new System.Collections.Generic.List<ILayer<double>>
        {
            new DenseLayer<double>(4, 8),
            new DenseLayer<double>(4, 8)
        };
        var router = new DenseLayer<double>(4, 2, new SoftmaxActivation<double>() as IActivationFunction<double>);
        return new MixtureOfExpertsLayer<double>(experts, router,
            inputShape: [4], outputShape: [8], topK: 1);
    }
    protected override int[] InputShape => [1, 4];
}
