using AiDotNet.VisionLanguage.Encoders;
using Xunit;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

/// <summary>
/// Guards the production option defaults independently from bounded generated fixtures.
/// </summary>
/// <remarks>
/// This is intentionally an options contract, not a paper-fidelity claim. Matching scalar defaults
/// cannot prove that the constructed vision encoder, clinical text encoder, projections, loss and
/// inference paths implement the reference MedCLIP system.
/// </remarks>
public sealed class MedCLIPDefaultOptionsContractTests
{
    [Fact]
    public void DefaultOptions_AreNotChangedByBoundedGeneratedFixtures()
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
