using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class ZonosOptionContextTests
{
    private static Zonos<double> Create(ZonosOptions options) => new(new NeuralNetworkArchitecture<double>(
        InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1) { RandomSeed = 7 }, options);

    [Fact]
    public void SelectingHybridBuildsMambaLayersWithoutAnExplicitAttentionList()
    {
        var options = ZonosFixture.Options();
        options.Backbone = ZonosBackbone.Hybrid;
        using var model = Create(options);
        Assert.NotNull(model.Network.Layers[0].SelfAttention);
        Assert.NotNull(model.Network.Layers[1].Ssm);
        var codes = new Tensor<double>(new[] { 6, options.Codebooks });
        for (int i = 0; i < codes.Length; i++) codes[i] = i % options.Codec.CodebookSize;
        var sample = new TtsTrainingSample<double>
        {
            Tokens = new Tensor<double>(new[] { 3 }, new Vector<double>(new[] { 1.0, 2.0, 3.0 })), CodecTokens = codes,
        };
        var before = model.GetParameters().ToArray();
        Assert.True(TtsPaperNumerics.IsFinite(model.Train(sample)));
        Assert.False(before.SequenceEqual(model.GetParameters()));
    }

    [Fact]
    public void ExplicitAttentionLayersOverrideTheHybridDefaultPattern()
    {
        var options = ZonosFixture.Options();
        options.Backbone = ZonosBackbone.Hybrid;
        options.AttentionLayers = new[] { 1 };
        using var model = Create(options);
        Assert.NotNull(model.Network.Layers[0].Ssm);
        Assert.NotNull(model.Network.Layers[1].SelfAttention);
    }

    [Fact]
    public void NullSharedSeedRetainsTheArchitectureInitializationFallback()
    {
        var options = ZonosFixture.Options();
        using var implicitSeed = Create(options);
        using var explicitSeed = Create(new ZonosOptions(options) { Seed = 7 });
        Assert.Equal(implicitSeed.GetParameters().ToArray(), explicitSeed.GetParameters().ToArray());
        Assert.Equal(implicitSeed.Codec.GetParameters().ToArray(), explicitSeed.Codec.GetParameters().ToArray());
    }

    [Fact]
    public void InitializationSeedControlsBackboneAndCodecIndependentlyOfSampling()
    {
        var options = ZonosFixture.Options(); options.Seed = 11;
        using var first = Create(options);
        using var repeat = Create(new ZonosOptions(options) { SamplingSeed = 31 });
        using var changed = Create(new ZonosOptions(options) { Seed = 12 });
        Assert.Equal(first.GetParameters().ToArray(), repeat.GetParameters().ToArray());
        Assert.False(first.GetParameters().SequenceEqual(changed.GetParameters()));
        Assert.Equal(first.Codec.GetParameters().ToArray(), repeat.Codec.GetParameters().ToArray());
        Assert.False(first.Codec.GetParameters().SequenceEqual(changed.Codec.GetParameters()));
    }
}
