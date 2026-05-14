#nullable disable
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.VisionLanguage.Reasoning;
using System.Collections.Generic;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.VisionLanguage;

public class KimiVLReviewRegressionIntegrationTests
{
    [Fact]
    public void Constructor_WithInvalidImageSize_ThrowsBeforeLayerInitialization()
    {
        var options = CreateOptions();
        options.ImageSize = 0;

        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new KimiVL<double>(CreateArchitectureWithCustomLayers(), options));

        Assert.Contains("ImageSize", ex.Message);
    }

    [Fact]
    public void Constructor_WithTinyNativeConfiguration_UsesSharedBoundaryHelpers()
    {
        var model = new KimiVL<double>(CreateDefaultLayerArchitecture(), CreateOptions());

        Assert.NotEmpty(model.Layers);
    }

    private static KimiVLOptions CreateOptions()
    {
        return new KimiVLOptions
        {
            ImageSize = 32,
            MaxVisualTokens = 16,
            VisionDim = 8,
            DecoderDim = 8,
            NumVisionLayers = 1,
            NumDecoderLayers = 1,
            NumHeads = 1,
            VocabSize = 128,
            MaxSequenceLength = 16
        };
    }

    private static NeuralNetworkArchitecture<double> CreateDefaultLayerArchitecture()
    {
        return new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 8);
    }

    private static NeuralNetworkArchitecture<double> CreateArchitectureWithCustomLayers()
    {
        var layers = new List<ILayer<double>>
        {
            new DenseLayer<double>(outputSize: 8),
            new DenseLayer<double>(outputSize: 8)
        };

        return new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 8,
            layers: layers);
    }
}
