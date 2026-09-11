using System;
using AiDotNet.NeuralNetworks.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Models.Options;

public class Gpt4VisionOptionsContractTests
{
    [Fact]
    public void SharedVisionWidth_RetainsTheFormerConstructorDefault()
    {
        var options = new Gpt4VisionOptions();
        Assert.Equal(1024, options.VisionDim);
        Assert.Equal(4096, options.EmbeddingDimension);
        Assert.Equal(4096, options.HiddenDim);
        Assert.Equal(24, options.VisionLayers);
        Assert.Equal(32, options.NumLanguageLayers);
        options.Validate();
    }

    [Fact]
    public void Options_DoNotExposeTheDuplicateVisionWidth()
    {
        // Identity of the removed migration-only API, not a dispatch policy.
        Assert.Null(typeof(Gpt4VisionOptions).GetProperty("VisionEmbeddingDim"));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(int.MinValue)]
    public void SharedVisionWidth_RejectsInvalidValuesAtTheOptionsBoundary(int width)
    {
        var options = new Gpt4VisionOptions { VisionDim = 16 };
        options.Validate();
        options.VisionDim = width;
        var exception = Assert.Throws<ArgumentException>(options.Validate);
        Assert.Equal("options", exception.ParamName);
        Assert.Contains($"{nameof(Gpt4VisionOptions)}.{nameof(options.VisionDim)}", exception.Message);
    }

    [Theory]
    [InlineData(8)]
    [InlineData(16)]
    public void SharedVisionWidth_RemainsConfigurableThroughTheFamilyBase(int width)
    {
        var options = new Gpt4VisionOptions();
        VisionLanguageModelOptions familyOptions = options;
        familyOptions.VisionDim = width;
        options.Validate();
        Assert.Equal(width, options.VisionDim);
        Assert.Equal(4096, options.EmbeddingDimension);
    }
}
