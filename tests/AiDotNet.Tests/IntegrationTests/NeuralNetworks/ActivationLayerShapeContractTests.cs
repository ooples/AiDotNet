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

    [Fact]
    public void Forward_RankOneInputAfterSampleShapeProbe_ReplacesTheProbedShape()
    {
        var layer = new ActivationLayer<float>((IActivationFunction<float>)new ReLUActivation<float>());
        layer.ResolveShapesOnlyFromSampleShape([1, 4]);
        Assert.Equal([1, 4], layer.GetOutputShape());

        Tensor<float> output = layer.Forward(new Tensor<float>([1f, -2f, 3f, -4f], [4]));

        Assert.Equal([4], output.Shape.ToArray());
        Assert.Equal([4], layer.GetInputShape());
        Assert.Equal([4], layer.GetOutputShape());
    }
}
