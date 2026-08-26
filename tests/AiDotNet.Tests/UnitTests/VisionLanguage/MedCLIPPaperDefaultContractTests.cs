using AiDotNet.VisionLanguage.Encoders;
using Xunit;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

/// <summary>
/// Guards the production paper defaults independently from bounded generated fixtures.
/// </summary>
public sealed class MedCLIPDefaultOptionsContractTests
{
    [Fact]
    public void DefaultOptions_AreNotChangedByBoundedGeneratedFixtures()
    {
        var options = new MedCLIPOptions();

        Assert.Equal(224, options.ImageSize);
        Assert.Equal("ResNet50", options.VisionBackbone);
        Assert.Equal("torchvision/resnet50", options.VisionModelId);
        Assert.Equal(2048, options.VisionEmbeddingDim);
        Assert.Equal(TextEncoderVariant.BERT, options.TextEncoderVariant);
        Assert.Equal(768, options.TextEmbeddingDim);
        Assert.Equal("emilyalsentzer/Bio_ClinicalBERT", options.TextModelId);
        Assert.Equal(12, options.NumTextLayers);
        Assert.Equal(12, options.NumTextHeads);
        Assert.Equal(0.1, options.DropoutRate, precision: 10);
        Assert.Equal(28996, options.VocabSize);
        Assert.Equal(512, options.MaxSequenceLength);
        Assert.Equal(512, options.ProjectionDim);
        Assert.Equal(DomainSpecialization.Medical, options.Domain);
        Assert.True(options.UseEntityExtraction);
        Assert.Equal(1.0, options.SemanticMatchingWeight);
        Assert.Equal(0.07, options.Temperature, precision: 10);
        Assert.Equal(2e-5, options.LearningRate, precision: 10);
        Assert.Equal(1e-4, options.WeightDecay, precision: 10);
        Assert.Equal(3, options.ImageMean.Length);
        Assert.Equal(3, options.ImageStd.Length);
        Assert.All(options.ImageMean,
            value => Assert.Equal(0.5862785803043838, value, precision: 12));
        Assert.All(options.ImageStd,
            value => Assert.Equal(0.27950088968644304, value, precision: 12));
        Assert.Null(options.TokenizerDirectory);
    }
}
