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
}
