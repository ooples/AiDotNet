using AiDotNet.VisionLanguage.Encoders;
using Xunit;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

/// <summary>
/// Guards the paper-facing production preset independently from bounded generated fixtures.
/// </summary>
public sealed class MedCLIPPaperDefaultContractTests
{
    [Fact]
    public void DefaultOptions_RetainOfficialMedClipEncoderPreset()
    {
        var options = new MedCLIPOptions();

        Assert.Equal(224, options.ImageSize);
        Assert.Equal("Swin-Tiny", options.VisionBackbone);
        Assert.Equal(768, options.VisionEmbeddingDim);
        Assert.Equal(TextEncoderVariant.BERT, options.TextEncoderVariant);
        Assert.Equal(768, options.TextEmbeddingDim);
        Assert.Equal(512, options.ProjectionDim);
        Assert.Equal(DomainSpecialization.Medical, options.Domain);
        Assert.True(options.UseEntityExtraction);
        Assert.Equal(1.0, options.SemanticMatchingWeight);
        Assert.Equal(0.07, options.Temperature, precision: 10);
    }
}
