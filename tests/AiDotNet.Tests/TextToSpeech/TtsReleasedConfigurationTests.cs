using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class TtsReleasedConfigurationTests
{
    private static NeuralNetworkArchitecture<double> Architecture() => new(
        InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);

    [Theory]
    [InlineData(false, "SampleRate", 24000)]
    [InlineData(true, "SampleRate", 24000)]
    [InlineData(false, "HopSize", 128)]
    [InlineData(true, "HopSize", 128)]
    [InlineData(false, "NumDecoderLayers", 3)]
    [InlineData(true, "NumDecoderLayers", 3)]
    public void OpenVoiceRejectsGeometryOutsideTheReleasedConverter(bool versionTwo, string property, int value)
    {
        Exception error;
        if (versionTwo)
        {
            var options = new OpenVoiceV2Options();
            Assert.Equal(4, options.NumDecoderLayers);
            options.GetType().GetProperty(property)!.SetValue(options, value);
            error = Assert.Throws<ArgumentException>(() => new OpenVoiceV2<double>(Architecture(), options));
        }
        else
        {
            var options = new OpenVoiceOptions();
            Assert.Equal(4, options.NumDecoderLayers);
            options.GetType().GetProperty(property)!.SetValue(options, value);
            error = Assert.Throws<ArgumentException>(() => new OpenVoice<double>(Architecture(), options));
        }
        Assert.Contains(property == "NumDecoderLayers" ? "four upsampling stages" : "22050 Hz", error.Message);
    }

    [Fact]
    public void UniAudioRejectsDropoutAbsentFromTheReleasedTransformers()
    {
        var options = new UniAudioOptions { DropoutRate = .1 };
        var error = Assert.Throws<ArgumentException>(() => new UniAudio<double>(Architecture(), options));
        Assert.Contains("no dropout", error.Message);
    }
}
