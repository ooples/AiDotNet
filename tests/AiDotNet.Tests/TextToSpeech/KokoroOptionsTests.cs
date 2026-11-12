using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.EndToEnd;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class KokoroOptionsTests
{
    [Fact]
    public void DefaultsPreserveTheReleasedGraph()
    {
        var options = new KokoroOptions();
        Assert.Equal(3, options.NumEncoderLayers);
        Assert.Equal(3, options.NumDecoderLayers);
        Assert.Equal(12, options.NumHeads);
        Assert.Equal(2048, options.FilterChannels);
        Assert.Equal(.2, options.DropoutRate);
    }

    [Theory]
    [InlineData("HiddenDim", 256)]
    [InlineData("EncoderDim", 256)]
    [InlineData("DecoderDim", 256)]
    [InlineData("StyleDim", 64)]
    [InlineData("VocabSize", 256)]
    [InlineData("SampleRate", 22050)]
    [InlineData("NumEncoderLayers", 0)]
    [InlineData("NumDecoderLayers", 0)]
    [InlineData("NumHeads", 5)]
    [InlineData("FilterChannels", 0)]
    [InlineData("MaxTextLength", 0)]
    public void RejectsIncompatibleGraphOptionsBeforeAllocation(string property, int value)
    {
        var options = new KokoroOptions();
        typeof(KokoroOptions).GetProperty(property)!.SetValue(options, value);
        Assert.Throws<ArgumentException>(() => new Kokoro<float>(Architecture(), options));
    }

    [Theory]
    [InlineData(-.1)]
    [InlineData(1.0)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void RejectsInvalidDropout(double probability)
        => Assert.Throws<ArgumentException>(() => new Kokoro<float>(Architecture(), new KokoroOptions { DropoutRate = probability }));

    private static NeuralNetworkArchitecture<float> Architecture()
        => new(InputType.OneDimensional, NeuralNetworkTaskType.Generative, inputSize: 4, outputSize: 1200);
}
