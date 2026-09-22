using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Cloning;

public class LayerConstructionShapeCloneTests
{
    // No generated or registered factory and no shape field of its own: construction
    // must recover the array forwarded to the shared base class.
    private sealed class ForwardedShapeLayer : InputLayer<float>
    {
        public ForwardedShapeLayer(int[] inputShape) : base(inputShape) { }
    }

    [Theory]
    [InlineData(4)]
    [InlineData(2, 4)]
    public void InputLayerClone_PreservesIndependentConstructionShape(params int[] shape)
    {
        var source = new ForwardedShapeLayer(shape);
        var clone = Assert.IsType<ForwardedShapeLayer>(LayerCloning.Clone(source));
        Assert.Equal(source.GetInputShape(), clone.GetInputShape());
        Assert.Equal(source.GetOutputShape(), clone.GetOutputShape());
        Assert.NotSame(source.GetInputShape(), clone.GetInputShape());
        clone.GetInputShape()[0] += 1;
        Assert.Equal(shape, source.GetInputShape());
    }
}
