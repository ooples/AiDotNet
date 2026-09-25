using System;
using System.IO;
using System.Reflection;
using System.Xml.Linq;
using AiDotNet.NeuralNetworks.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Models.Options;

/// <summary>Checks the actual compiler-emitted public options documentation and unchanged implementation defaults.</summary>
public sealed class VisionLanguageOptionsDocumentationTests
{
    public static TheoryData<Type, string, object, string> DocumentedProperties => new()
    {
        { typeof(VideoCLIPOptions), nameof(VideoCLIPOptions.NumFrames), 8, "frames" },
        { typeof(VideoCLIPOptions), nameof(VideoCLIPOptions.FrameRate), 1.0, "frames per second" },
        { typeof(VideoCLIPOptions), nameof(VideoCLIPOptions.TextHiddenDim), 512, "features" },
        { typeof(VideoCLIPOptions), nameof(VideoCLIPOptions.NumFrameEncoderLayers), 12, "blocks" },
        { typeof(VideoCLIPOptions), nameof(VideoCLIPOptions.NumTemporalLayers), 4, "blocks" },
        { typeof(VideoCLIPOptions), nameof(VideoCLIPOptions.NumTextLayers), 12, "blocks" },
        { typeof(AudioVisualCorrespondenceOptions), nameof(AudioVisualCorrespondenceOptions.AudioSampleRate), 16000, "samples per second" },
        { typeof(AudioVisualCorrespondenceOptions), nameof(AudioVisualCorrespondenceOptions.VideoFrameRate), 25.0, "frames per second" },
        { typeof(LLaVAOptions), nameof(LLaVAOptions.NumLmLayers), 32, "blocks" },
        { typeof(BlipOptions), nameof(BlipOptions.NumDecoderLayers), 12, "blocks" },
        { typeof(BlipOptions), nameof(BlipOptions.MlpDim), 3072, "features" },
        { typeof(Blip2Options), nameof(Blip2Options.QformerHiddenDim), 768, "features" },
        { typeof(Blip2Options), nameof(Blip2Options.LmHiddenDim), 2560, "features" },
        { typeof(Blip2Options), nameof(Blip2Options.NumQformerLayers), 12, "blocks" },
        { typeof(Blip2Options), nameof(Blip2Options.NumQueryTokens), 32, "tokens" },
        { typeof(Blip2Options), nameof(Blip2Options.NumLmDecoderLayers), 6, "blocks" },
        { typeof(ImageBindOptions), nameof(ImageBindOptions.AudioSampleRate), 16000, "samples per second" },
        { typeof(ImageBindOptions), nameof(ImageBindOptions.AudioMaxDuration), 10, "seconds" },
        { typeof(ImageBindOptions), nameof(ImageBindOptions.ImuTimesteps), 2000, "observations" },
        { typeof(ImageBindOptions), nameof(ImageBindOptions.NumVideoFrames), 2, "frames" }
    };

    [Theory]
    [MemberData(nameof(DocumentedProperties))]
    public void PropertiesDocumentUnitsProvenanceAndBeginnerEffectWithoutChangingDefaults(Type optionsType,
        string propertyName, object expectedDefault, string unit)
    {
        PropertyInfo property = optionsType.GetProperty(propertyName, BindingFlags.Public | BindingFlags.Instance)
            ?? throw new InvalidOperationException("Missing typed options property: " + propertyName);
        object options = Activator.CreateInstance(optionsType)
            ?? throw new InvalidOperationException("The options constructor did not return an instance.");
        Assert.Equal(expectedDefault, property.GetValue(options));
        var member = Member(optionsType, propertyName);
        var value = member.Element("value") ?? throw new InvalidOperationException("Missing value documentation for " + propertyName);
        Assert.Contains(unit, value.Value);
        Assert.Contains("implementation", value.Value);
        Assert.Contains("For Beginners:", member.Element("remarks")?.Value ?? string.Empty);
    }

    [Theory]
    [InlineData(typeof(VideoCLIPOptions), nameof(VideoCLIPOptions.FrameRate))]
    [InlineData(typeof(AudioVisualCorrespondenceOptions), nameof(AudioVisualCorrespondenceOptions.VideoFrameRate))]
    public void NominalVideoRateDocumentationDoesNotPromiseAutomaticResampling(Type optionsType, string propertyName)
    {
        var member = Member(optionsType, propertyName);
        string remarks = member.Element("remarks")?.Value ?? string.Empty;
        Assert.Contains("metadata", remarks);
        Assert.Contains("does not", remarks);
        Assert.Contains("resample", remarks);
    }

    private static XElement Member(Type type, string propertyName)
    {
        // Framework hosts shadow-copy DLLs without their XML; the app base is the actual output.
        string path = Path.Combine(AppContext.BaseDirectory, type.Assembly.GetName().Name + ".xml");
        var documentation = XDocument.Load(path);
        string memberName = $"P:{type.FullName}.{propertyName}";
        return Assert.Single(documentation.Descendants("member"), member => (string?)member.Attribute("name") == memberName);
    }
}
