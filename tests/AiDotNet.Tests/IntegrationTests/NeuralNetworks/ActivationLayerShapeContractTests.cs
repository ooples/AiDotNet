using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

public sealed class ActivationLayerShapeContractTests
{
    [Fact]
    public void Forward_ResolvesPerSampleShapeWithoutBatchDimension()
    {
        var layer = new ActivationLayer<float>(
            (IVectorActivationFunction<float>)new SoftmaxActivation<float>());
        var input = new Tensor<float>([38, 2]);

        Tensor<float> output = layer.Forward(input);

        Assert.Equal([38, 2], output.Shape.ToArray());
        Assert.Equal([2], layer.GetInputShape());
        Assert.Equal([2], layer.GetOutputShape());
    }
}
