using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

public class KairosMultiSizePatchLayerTests
{
    [Fact]
    public void ConstructorKnownWidths_ParametersExistAndStayStableAfterForward()
    {
        var layer = new KairosMultiSizePatchLayer<double>(
            contextLength: 16,
            hiddenDim: 4,
            patchSizes: new[] { 4, 8 });

        var before = layer.GetParameters();
        Assert.Equal(90, before.Length);

        var input = new Tensor<double>(new double[16], new[] { 1, 16 });
        var output = layer.Forward(input);
        var after = layer.GetParameters();

        Assert.Equal(new[] { 1, 4 }, output.Shape.ToArray());
        Assert.Equal(before.Length, after.Length);
    }
}
