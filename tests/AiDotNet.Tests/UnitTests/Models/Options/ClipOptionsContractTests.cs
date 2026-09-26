using System;
using System.Linq;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Models.Options;

public sealed class ClipOptionsContractTests
{
    [Theory]
    [InlineData(nameof(VisionLanguageModelOptions.PatchSize))]
    [InlineData(nameof(VisionLanguageModelOptions.Channels))]
    [InlineData(nameof(VisionLanguageModelOptions.VocabSize))]
    [InlineData(nameof(VisionLanguageModelOptions.NumHeads))]
    [InlineData(nameof(VisionLanguageModelOptions.HiddenDim))]
    [InlineData(nameof(VisionLanguageModelOptions.NumEncoderLayers))]
    [InlineData(nameof(VisionLanguageModelOptions.VisionDim))]
    [InlineData(nameof(VisionLanguageModelOptions.VisionLayers))]
    public void OnnxOnlyOptionsDoNotExposeNativeTowerSetters(string propertyName)
    {
        Assert.Null(typeof(ClipOptions).GetProperty(propertyName));
        // Native models keep their existing, independently configurable geometry.
        Assert.NotNull(typeof(VisionLanguageModelOptions).GetProperty(propertyName));
    }

    [Fact]
    public void OnlyTheThreeConsumedVisionLanguageDimensionsArePublic()
    {
        string[] names = typeof(ClipOptions).GetProperties()
            .Where(property => property.DeclaringType?.Namespace == typeof(ClipOptions).Namespace)
            .Select(property => property.Name).OrderBy(name => name).ToArray();

        Assert.Equal(new[] { nameof(ClipOptions.EmbeddingDimension), nameof(ClipOptions.ImageSize),
            nameof(ClipOptions.MaxSequenceLength) }, names);
        Assert.True(typeof(NeuralNetworkOptions).IsAssignableFrom(typeof(ClipOptions)));
        Assert.False(typeof(VisionLanguageModelOptions).IsAssignableFrom(typeof(ClipOptions)));
    }

    [Fact]
    public void ShippedInputDefaultsRemainUnchanged()
    {
        var options = new ClipOptions();
        Assert.Equal(512, options.EmbeddingDimension);
        Assert.Equal(77, options.MaxSequenceLength);
        Assert.Equal(224, options.ImageSize);
        options.Validate();
    }

    [Theory]
    [InlineData(4, 8, 2)]
    [InlineData(768, 64, 336)]
    public void PositiveInputDimensionsAreRetained(int embedding, int context, int image)
    {
        var options = new ClipOptions
        {
            EmbeddingDimension = embedding, MaxSequenceLength = context, ImageSize = image
        };
        options.Validate();
        Assert.Equal(embedding, options.EmbeddingDimension);
        Assert.Equal(context, options.MaxSequenceLength);
        Assert.Equal(image, options.ImageSize);
    }

    [Theory]
    [InlineData(nameof(ClipOptions.EmbeddingDimension), 0)]
    [InlineData(nameof(ClipOptions.EmbeddingDimension), -1)]
    [InlineData(nameof(ClipOptions.MaxSequenceLength), 0)]
    [InlineData(nameof(ClipOptions.MaxSequenceLength), -1)]
    [InlineData(nameof(ClipOptions.ImageSize), 0)]
    [InlineData(nameof(ClipOptions.ImageSize), -1)]
    public void InvalidInputsStillIdentifyTheOffendingOption(string propertyName, int value)
    {
        var options = new ClipOptions();
        var property = typeof(ClipOptions).GetProperty(propertyName);
        Assert.NotNull(property);
        property.SetValue(options, value);

        var error = Assert.Throws<ArgumentException>(options.Validate);
        Assert.Equal("options", error.ParamName);
        Assert.Contains(nameof(ClipOptions) + "." + propertyName, error.Message);
    }
}
