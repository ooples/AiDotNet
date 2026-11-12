using AiDotNet.Models;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class OpenVoiceOptionAliasesTests
{
    [Theory]
    [InlineData(false, "NumEncoderLayers", "IpaLayers", 2, 3)]
    [InlineData(true, "NumEncoderLayers", "IpaLayers", 2, 3)]
    [InlineData(false, "NumHeads", "IpaHeads", 1, 2)]
    [InlineData(true, "NumHeads", "IpaHeads", 1, 2)]
    [InlineData(false, "MelChannels", "MelChannels", 20, 30)]
    [InlineData(true, "MelChannels", "MelChannels", 20, 30)]
    [InlineData(false, "DropoutRate", "IpaDropout", .25, .35)]
    [InlineData(true, "DropoutRate", "IpaDropout", .25, .35)]
    public void InheritedAndNativeSettingsStayCoherentAcrossIndependentConfigurationCopies(
        bool versionTwo, string inheritedName, string nativeName, double first, double second)
    {
        TtsModelOptions options = versionTwo ? new OpenVoiceV2Options() : new OpenVoiceOptions();
        var inherited = typeof(TtsModelOptions).GetProperty(inheritedName)!;
        var native = typeof(OpenVoiceTrainingOptions).GetProperty(nativeName)!;
        var settings = Settings(options);
        var initial = native.GetValue(settings);
        var firstValue = Convert.ChangeType(first, inherited.PropertyType);
        var secondValue = Convert.ChangeType(second, inherited.PropertyType);

        inherited.SetValue(options, firstValue);
        Assert.Equal(firstValue, native.GetValue(settings));
        var copy = (TtsModelOptions)CloneEngine.CopyConfiguration(options);
        var copiedSettings = Settings(copy);
        Assert.NotSame(settings, copiedSettings);
        Assert.Equal(firstValue, inherited.GetValue(copy));
        Assert.Equal(firstValue, native.GetValue(copiedSettings));
        inherited.SetValue(copy, secondValue);
        Assert.Equal(secondValue, native.GetValue(copiedSettings));
        Assert.Equal(firstValue, native.GetValue(settings));

        native.SetValue(settings, initial);
        Assert.Equal(initial, inherited.GetValue(options));
        Assert.Equal(secondValue, inherited.GetValue(copy));
    }

    private static OpenVoiceTrainingOptions Settings(TtsModelOptions options) => options switch
    {
        OpenVoiceOptions one => one.Training,
        OpenVoiceV2Options two => two.Training,
        _ => throw new ArgumentException("Expected an OpenVoice configuration.", nameof(options))
    };
}
