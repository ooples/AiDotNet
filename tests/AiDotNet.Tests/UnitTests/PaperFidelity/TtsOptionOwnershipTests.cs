using AiDotNet.TextToSpeech.Latest;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PaperFidelity;

public class TtsOptionOwnershipTests
{
    [Theory]
    [InlineData(typeof(BigVGANOptions))]
    [InlineData(typeof(HiFiGANOptions))]
    [InlineData(typeof(ISTFTNetOptions))]
    public void TransposedConvolutionKernelsBelongToTheirConsumingModels(Type optionsType)
    {
        var property = optionsType.GetProperty("UpsampleKernelSizes");
        Assert.NotNull(property);
        Assert.Equal(optionsType, property.DeclaringType);
        var options = Activator.CreateInstance(optionsType);
        property.SetValue(options, new[] { 12, 8 });
        var constructor = optionsType.GetConstructor(new[] { optionsType });
        Assert.NotNull(constructor);
        var copy = constructor.Invoke(new[] { options });
        var copiedKernels = Assert.IsType<int[]>(property.GetValue(copy));
        Assert.Equal(new[] { 12, 8 }, copiedKernels);
        Assert.NotSame(property.GetValue(options), copiedKernels);
    }

    [Theory]
    [InlineData(typeof(APNetOptions))]
    [InlineData(typeof(APNet2Options))]
    public void FrameLevelVocoderDoesNotAdvertiseTransposedConvolutionKernels(Type optionsType)
        => Assert.Null(optionsType.GetProperty("UpsampleKernelSizes"));

    [Theory]
    [InlineData(typeof(KaniTTSOptions))]
    [InlineData(typeof(KaniTTS2Options))]
    public void Lfm2OptionsExposeTheirOwnTopology(Type optionsType)
    {
        foreach (var removed in new[] { "NumHeads", "NumEncoderLayers", "NumDecoderLayers", "DropoutRate", "MelChannels", "FftSize", "MaxMelLength" })
            Assert.Null(optionsType.GetProperty(removed));
        Assert.NotNull(optionsType.GetProperty(nameof(KaniTTSOptions.Heads)));
        Assert.NotNull(optionsType.GetProperty(nameof(KaniTTSOptions.AttentionLayers)));
    }

    [Fact]
    public void CodecHopIsDerivedAndCopiedWithItsRates()
    {
        var source = new KaniTTSOptions { Seed = 2294, CodecDownsampleRates = new[] { 2, 3 } };
        var copy = new KaniTTSOptions(source);
        Assert.Equal(2294, copy.Seed);
        Assert.Equal(6, copy.HopSize);
        Assert.NotSame(source.CodecDownsampleRates, copy.CodecDownsampleRates);
        copy.CodecDownsampleRates[1] = 5;
        Assert.Equal(10, copy.HopSize);
        Assert.Equal(6, source.HopSize);
    }
}
