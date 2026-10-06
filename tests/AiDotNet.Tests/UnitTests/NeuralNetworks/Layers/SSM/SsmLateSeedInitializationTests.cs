using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers.SSM;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers.SSM;

/// <summary>
/// MambaBlock and RWKV7Block draw random values in their constructors (dt init, orthogonal LoRA init). Built
/// into an explicit architecture layer list, they are constructed before the network opens its seed scope, so
/// they initialise unseeded and receive their seed only later, when the network wires layer seeds. A seed that
/// arrives while their parameters are still the constructed ones now redoes the initialisation from it, so equal
/// seeds reproduce the initial weights; parameters changed since are kept (#2290 review).
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
        var first = ConstructUnseeded(create);
        var second = ConstructUnseeded(create);
        Assert.False(first.GetParameters().ToArray().SequenceEqual(second.GetParameters().ToArray()),
            $"{name}: two unseeded constructions came out identical, so this test could not tell anything apart.");

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
