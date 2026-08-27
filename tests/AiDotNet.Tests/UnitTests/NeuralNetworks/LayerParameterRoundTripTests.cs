using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public sealed class LayerParameterRoundTripTests
{
    [Fact]
    public async Task SetParameters_ConcreteRegistryWithDeferredInput_RoundTripsWithoutGrowing()
    {
        await Task.Yield();

        var layer = new DuelingCombinationLayer<double>(featureDim: 8, actionSize: 3, seed: 17);
        var original = layer.GetParameters();

        layer.SetParameters(original);
        var firstRoundTrip = layer.GetParameters();
        layer.SetParameters(firstRoundTrip);
        var secondRoundTrip = layer.GetParameters();

        Assert.Equal(original.Length, firstRoundTrip.Length);
        Assert.Equal(original.Length, secondRoundTrip.Length);
        Assert.Equal(original.Length, layer.ParameterCount);
        Assert.Equal(original.ToArray(), secondRoundTrip.ToArray());
    }

    [Fact]
    public void ShapePreservingLayer_DeserializeRestoresBothResolvedShapes()
    {
        var source = new ActivationLayer<double>(
            (IActivationFunction<double>)new IdentityActivation<double>());
        source.ResolveFromShape([3, 5]);
        int[] expectedInputShape = source.GetInputShape();
        int[] expectedOutputShape = source.GetOutputShape();
        Assert.Equal(expectedInputShape, expectedOutputShape);
        Assert.All(expectedInputShape, dimension => Assert.True(dimension > 0));

        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            source.Serialize(writer);
            writer.Flush();
        }

        var restored = new ActivationLayer<double>(
            (IActivationFunction<double>)new IdentityActivation<double>());
        stream.Position = 0;
        using var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);
        restored.Deserialize(reader);

        Assert.Equal(expectedInputShape, restored.GetInputShape());
        Assert.Equal(expectedOutputShape, restored.GetOutputShape());
        Assert.Equal(stream.Length, stream.Position);
    }
}
