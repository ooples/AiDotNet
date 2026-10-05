using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers.SSM;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers.SSM;

/// <summary>
/// MambaBlock and RWKV7Block draw their random initial values (dt init, orthogonal LoRA init) on first use, not in
/// their constructors. Built into an explicit architecture layer list, they are constructed before the network
/// opens its seed scope; the network seeds them at its construction, before anything reads their weights, so
/// equal seeds reproduce the initial weights. Weights already read, set or trained are never redrawn (#2290 review).
/// </summary>
public class SsmLateSeedInitializationTests
{
    public static IEnumerable<object[]> Layers()
    {
        yield return new object[] { "MambaBlock", (Func<LayerBase<double>>)(() => new MambaBlock<double>(4, 8, 4)) };
        yield return new object[] { "RWKV7Block", (Func<LayerBase<double>>)(() => new RWKV7Block<double>(4, 8, 2)) };
    }

    /// <summary>Constructs a layer with no seed scope open, as a layer built ahead of its network is.</summary>
    private static LayerBase<double> ConstructUnseeded(Func<LayerBase<double>> create)
    {
        var scope = LayerInitializationSeedScope.CaptureScope();
        var ambient = LayerInitializationSeedScope.AmbientFallbackSeed;
        try
        {
            LayerInitializationSeedScope.AmbientFallbackSeed = null;
            LayerInitializationSeedScope.ResetForModelConstruction(null);
            var layer = create();
            Assert.Null(layer.RandomSeed);
            return layer;
        }
        finally
        {
            LayerInitializationSeedScope.AmbientFallbackSeed = ambient;
            LayerInitializationSeedScope.RestoreScope(scope);
        }
    }

    [Theory]
    [MemberData(nameof(Layers))]
    public void ASeedAssignedAfterConstruction_ReproducesTheInitialWeights(string name, Func<LayerBase<double>> create)
    {
        // Control, on separate instances so the layers under test stay unread: unseeded, the draws differ.
        Assert.False(ConstructUnseeded(create).GetParameters().ToArray()
                .SequenceEqual(ConstructUnseeded(create).GetParameters().ToArray()),
            $"{name}: two unseeded constructions came out identical, so this test could not tell anything apart.");

        var first = ConstructUnseeded(create);
        var second = ConstructUnseeded(create);
        first.RandomSeed = 5;
        second.RandomSeed = 5;

        Assert.Equal(first.GetParameters().ToArray(), second.GetParameters().ToArray());
    }

    [Theory]
    [MemberData(nameof(Layers))]
    public void ASeedAssignedAfterTheWeightsChanged_KeepsThem(string name, Func<LayerBase<double>> create)
    {
        var layer = ConstructUnseeded(create);
        var changed = layer.GetParameters().Clone();
        for (int i = 0; i < changed.Length; i++) changed[i] = 0.25 + i * 1e-4;
        layer.SetParameters(changed);

        layer.RandomSeed = 5;

        Assert.True(changed.ToArray().SequenceEqual(layer.GetParameters().ToArray()),
            $"{name}: assigning a seed overwrote parameters that had been set after construction.");
    }
}
