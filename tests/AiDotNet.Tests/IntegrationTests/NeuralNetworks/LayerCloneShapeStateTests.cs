using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

public class LayerCloneShapeStateTests
{
    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void InputLayer_ClonesPreserveDeclaredShape(bool multidimensional, bool configurationOnly)
    {
        CheckInputLayer<float>(multidimensional, configurationOnly);
        CheckInputLayer<double>(multidimensional, configurationOnly);
    }

    private static void CheckInputLayer<T>(bool multidimensional, bool configurationOnly)
    {
        var shape = multidimensional ? new[] { 2, 3 } : new[] { 6 };
        var source = multidimensional ? new InputLayer<T>(shape) : new InputLayer<T>(6);
        object result = configurationOnly
            ? ((IConfigurationCloneable)source).CloneConfiguration()
            : LayerCloning.Clone(source);
        var clone = Assert.IsType<InputLayer<T>>(result);
        Assert.NotSame(source, clone);
        Assert.Equal(shape, clone.GetInputShape());
        Assert.Equal(shape, clone.GetOutputShape());
        Assert.Equal(source.ParameterCount, clone.ParameterCount);
        Assert.NotSame(source.GetInputShape(), clone.GetInputShape());
        Assert.NotSame(source.GetOutputShape(), clone.GetOutputShape());
    }

    [Fact]
    public void ConsumerLayer_ConstructorForwardedShape_IsRecoverable()
    {
        var source = new ConsumerShapeLayer(new[] { 2, 3 });
        var clone = Assert.IsType<ConsumerShapeLayer>(LayerCloning.Clone(source));
        Assert.NotSame(source, clone);
        Assert.Equal(new[] { 2, 3 }, clone.GetInputShape());
        Assert.Equal(new[] { 2, 3 }, clone.GetOutputShape());
        clone.GetInputShape()[0] = 7;
        clone.GetOutputShape()[1] = 9;
        Assert.Equal(new[] { 2, 3 }, source.GetInputShape());
        Assert.Equal(new[] { 2, 3 }, source.GetOutputShape());
    }

    public sealed class ConsumerShapeLayer : InputLayer<double>
    {
        public ConsumerShapeLayer(int[] inputShape) : base(inputShape) { }
    }
}
