using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

public class CollectionSubLayerProbe
{
    private readonly ITestOutputHelper _out;
    public CollectionSubLayerProbe(ITestOutputHelper o) => _out = o;

    [Fact]
    public void CitrinetBlockExposesItsCollectionHeldChildren()
    {
        var layer = new CitrinetBlockLayer<double>(channels: 4, kernelSize: 3, numSubBlocks: 2);

        // Registration is lazy: the generated EnsureSubLayersRegistered runs on first forward,
        // not in the constructor, so drive one pass before asking.
        typeof(CitrinetBlockLayer<double>)
            .GetMethod("EnsureSubLayersRegistered", BindingFlags.Instance | BindingFlags.NonPublic)
            ?.Invoke(layer, null);

        var subs = layer.GetSubLayers();
        _out.WriteLine($"GetSubLayers() returns {subs.Count}: " +
                       string.Join(", ", subs.Select(s => s.GetType().Name)));

        // Count the children actually held in fields, including the List<> ones.
        int held = 0;
        foreach (var f in typeof(CitrinetBlockLayer<double>)
                     .GetFields(BindingFlags.Instance | BindingFlags.NonPublic))
        {
            var v = f.GetValue(layer);
            if (v is ILayer<double>) held++;
            else if (v is System.Collections.IEnumerable e && v is not string)
                held += e.Cast<object>().Count(x => x is ILayer<double>);
        }
        _out.WriteLine($"child layers actually held in fields: {held}");

        Assert.True(subs.Count >= held,
            $"GetSubLayers() reports {subs.Count} children but the layer holds {held}. The " +
            $"{held - subs.Count} missing ones live in List<> fields, which the source generator's " +
            "EnsureSubLayersRegistered does not walk — so the tape training step never reaches " +
            "their parameters and they silently never train.");
    }
}
