using AiDotNet.Models.Options;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

public class PANNsPoolingLayerTests
{
    [Fact]
    public void Forward_NchwFeatures_UsesFrequencyMeanThenTemporalMaxPlusMean()
    {
        // Channel 0 frequency means by time: [2, 6] -> max 6 + mean 4 = 10.
        // Channel 1 frequency means by time: [3, 8] -> max 8 + mean 5.5 = 13.5.
        var input = new Tensor<double>(
            new[] { 1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 10.0 },
            new[] { 1, 2, 2, 2 });
        var layer = new PANNsPoolingLayer<double>();

        var output = layer.Forward(input);

        Assert.Equal(new[] { 1, 2 }, output.Shape.ToArray());
        Assert.Equal(10.0, output[0], 12);
        Assert.Equal(13.5, output[1], 12);
    }

    [Fact]
    public void Options_DefaultsMatchCnn14_AndCopyPreservesCustomization()
    {
        var defaults = new PANNsModelOptions();
        Assert.Equal(32_000, defaults.SampleRate);
        Assert.Equal(1024, defaults.StftWindowSize);
        Assert.Equal(320, defaults.HopLength);
        Assert.Equal(64, defaults.NumMelBands);
        Assert.Equal(50.0, defaults.MinFrequency);
        Assert.Equal(14_000.0, defaults.MaxFrequency);
        Assert.Equal(6, defaults.NumBlocks);
        Assert.Equal(64, defaults.BaseChannels);
        Assert.Equal(2048, defaults.EmbeddingDim);
        Assert.Equal(0.2, defaults.DropoutRate);
        Assert.Equal(0.5, defaults.HeadDropoutRate);
        Assert.Equal(527, defaults.NumClasses);

        var customized = new PANNsModelOptions
        {
            MinFrequency = 80.0,
            MaxFrequency = 7_600.0,
            HeadDropoutRate = 0.25,
            BaseChannels = 12,
            NumBlocks = 3
        };
        var copy = new PANNsModelOptions(customized);

        Assert.Equal(customized.MinFrequency, copy.MinFrequency);
        Assert.Equal(customized.MaxFrequency, copy.MaxFrequency);
        Assert.Equal(customized.HeadDropoutRate, copy.HeadDropoutRate);
        Assert.Equal(customized.BaseChannels, copy.BaseChannels);
        Assert.Equal(customized.NumBlocks, copy.NumBlocks);
    }

    [Fact]
    public void ConfigurableFactory_ConvWidthsAreIndependentFromEmbeddingHeadWidth()
    {
        var convolutionWidths = LayerHelper<double>.CreateDefaultPANNsLayers(
                numMels: 16,
                baseChannels: 12,
                numBlocks: 3,
                embeddingDim: 8,
                numClasses: 4,
                dropoutRate: 0.0,
                headDropoutRate: 0.0)
            .OfType<ConvolutionalLayer<double>>()
            .Select(layer => layer.OutputDepth)
            .ToArray();

        Assert.Equal(new[] { 12, 12, 24, 24, 48, 48 }, convolutionWidths);
    }
}
